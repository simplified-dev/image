package dev.simplified.image.codec.webp;

import dev.simplified.annotations.SilentThrows;
import dev.simplified.collection.Concurrent;
import dev.simplified.collection.ConcurrentList;
import dev.simplified.image.AnimatedImageData;
import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.PixelBuffer;
import dev.simplified.image.codec.ImageWriter;
import dev.simplified.image.codec.common.LittleEndian;
import dev.simplified.image.codec.webp.vp8.VP8Encoder;
import dev.simplified.image.codec.webp.vp8.VP8EncoderSession;
import dev.simplified.image.codec.webp.vp8l.NearLosslessPreprocess;
import dev.simplified.image.codec.webp.vp8l.VP8LEncoder;
import dev.simplified.image.frame.FrameBlend;
import dev.simplified.image.frame.FrameDisposal;
import dev.simplified.image.frame.ImageFrame;
import dev.simplified.image.frame.LoopCount;
import dev.simplified.image.option.ImageWriteOptions;
import dev.simplified.image.option.WebPWriteOptions;
import dev.simplified.image.pixel.AlphaMode;
import org.jetbrains.annotations.ApiStatus;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.IntStream;

/**
 * Writes WebP images (static and animated) using a pure Java implementation
 * of the VP8L lossless and VP8 lossy codecs.
 * <p>
 * An animation is written as the canvases its frames compose to, {@link ImageData#toCanvasFrames()},
 * so a frame may be any size at any offset, blend over the canvas and dispose of itself in any way,
 * {@link FrameDisposal#RESTORE_TO_PREVIOUS} included: each written frame shows what that frame
 * leaves on the canvas.
 */
@ApiStatus.Internal
public class WebPImageWriter implements ImageWriter {

    /**
     * The largest play count an {@code ANIM} chunk's 16-bit loop count field declares.
     */
    private static final int MAX_ANIM_LOOP_COUNT = 0xFFFF;

    @Override
    public @NotNull ImageFormat getFormat() {
        return ImageFormat.WEBP;
    }

    @Override
    public byte @NotNull [] write(@NotNull ImageData data, @Nullable ImageWriteOptions options) {
        boolean lossless = true;
        float quality = 0.75f;
        Optional<LoopCount> loopCount = Optional.empty();
        boolean multithreaded = true;
        boolean alphaCompression = true;
        boolean usePFrames = false;
        int forceKeyframeEvery = -1;
        int motionSearchThreads = -1;
        boolean autoSegment = false;
        int nearLossless = 100;

        if (options instanceof WebPWriteOptions webpOptions) {
            lossless = webpOptions.isLossless();
            quality = webpOptions.getQuality();
            loopCount = webpOptions.getLoopCount();
            multithreaded = webpOptions.isMultithreaded();
            alphaCompression = webpOptions.isAlphaCompression();
            usePFrames = webpOptions.isUsePFrames();
            forceKeyframeEvery = webpOptions.getForceKeyframeEvery();
            motionSearchThreads = webpOptions.getMotionSearchThreads();
            autoSegment = webpOptions.isAutoSegment();
            nearLossless = webpOptions.getNearLossless();
        }

        // Composed and converted after the dispatch, not before: both answer an AnimatedImageData for any
        // animated image, so doing either first would send an animated ImageData implemented elsewhere
        // down the animated path. An AnimatedImageData composes and converts to an AnimatedImageData, so
        // the cast holds. Composing comes first, so OVER frames are laid down in the convention the
        // frames are held in.
        if (data instanceof AnimatedImageData animated) {
            LoopCount declared = loopCount.orElse(animated.getLoopCount());
            AnimatedImageData canvases = (AnimatedImageData) animated.toCanvasFrames().toAlphaMode(AlphaMode.STRAIGHT);
            return writeAnimated(canvases, lossless, quality, declared, multithreaded, usePFrames, forceKeyframeEvery, motionSearchThreads, autoSegment, nearLossless);
        }

        return writeStatic(data, lossless, quality, alphaCompression, autoSegment, nearLossless);
    }

    /**
     * Auto-picks a sensible {@code forceKeyframeEvery} when the caller left the
     * option at its sentinel default ({@code -1}): tooltip-length animations
     * (<= 60 frames) stay at single-keyframe max compression, longer animations
     * switch to a 30-frame interval so viewers can seek. Callers who set an
     * explicit value (including {@code 0}) are honoured as-is.
     */
    private static int resolveForceKeyframeEvery(int configured, int frameCount) {
        if (configured != -1) return configured;
        return frameCount > 60 ? 30 : 0;
    }

    private byte @NotNull [] writeStatic(@NotNull ImageData data, boolean lossless, float quality, boolean alphaCompression, boolean autoSegment, int nearLossless) {
        PixelBuffer pixels = data.toPixelBuffer(AlphaMode.STRAIGHT);
        byte[] payload;

        if (lossless) {
            payload = VP8LEncoder.encode(applyNearLosslessIfEnabled(pixels, nearLossless));
            // Simple lossless: just RIFF [ VP8L ]
            ConcurrentList<WebPChunk> chunks = Concurrent.newList();
            chunks.add(RiffContainer.createChunk(WebPChunk.Type.VP8L, payload));
            return RiffContainer.write(chunks);
        }

        payload = autoSegment
            ? VP8Encoder.encodeWithAutoSegment(pixels, quality)
            : VP8Encoder.encode(pixels, quality);
        boolean hasAlpha = data.hasAlpha();

        ConcurrentList<WebPChunk> chunks = Concurrent.newList();

        if (hasAlpha) {
            // Extended: VP8X + ALPH + VP8
            chunks.add(RiffContainer.createChunk(WebPChunk.Type.VP8X, buildVP8XPayload(pixels.width(), pixels.height(), false, true)));
            chunks.add(RiffContainer.createChunk(WebPChunk.Type.ALPH, encodeAlphaPlane(pixels, alphaCompression)));
            chunks.add(RiffContainer.createChunk(WebPChunk.Type.VP8, payload));
        } else {
            chunks.add(RiffContainer.createChunk(WebPChunk.Type.VP8, payload));
        }

        return RiffContainer.write(chunks);
    }

    private byte @NotNull [] writeAnimated(
        @NotNull AnimatedImageData data,
        boolean lossless,
        float quality,
        @NotNull LoopCount loopCount,
        boolean multithreaded,
        boolean usePFrames,
        int forceKeyframeEvery,
        int motionSearchThreads,
        boolean autoSegment,
        int nearLossless
    ) {
        int animLoopCount = animLoopCount(loopCount);
        ConcurrentList<ImageFrame> frames = data.getFrames();
        boolean hasAlpha = data.hasAlpha();
        // Per-frame lossy + alpha: emit an ALPH sub-chunk before the VP8 sub-chunk.
        boolean perFrameAlpha = hasAlpha && !lossless;

        // Compute the per-frame partial-frame rectangles. Frame 0 is always the
        // full canvas (it seeds the decoded state); frames 1..N use the bounding
        // box of pixels that differ from frame i-1. This drops file size ~10x+
        // on content where only a small region of the canvas changes per frame
        // (tooltip animations where the body is static, only the bottom line of
        // obfuscated text cycles), matches what libwebp's WebPAnimEncoder and
        // GIF89a both do, and avoids the "animated WebP shows as static" issue
        // that strict viewers (Windows Photos) exhibit on very large full-frame
        // animated WebPs. Partial frames are skipped on the P-frame path because
        // VP8 inter prediction requires the full-canvas reference - a partial
        // frame would leave the previously-cached reference state inconsistent.
        int canvasW = data.getWidth();
        int canvasH = data.getHeight();
        int numFrames = frames.size();
        int[] bboxX = new int[numFrames];
        int[] bboxY = new int[numFrames];
        int[] bboxW = new int[numFrames];
        int[] bboxH = new int[numFrames];
        PixelBuffer[] framePixels = new PixelBuffer[numFrames];
        boolean usePartialFrames = !usePFrames || lossless;
        if (!usePartialFrames) {
            // P-frame path: every ANMF covers the full canvas, as required by
            // VP8 inter prediction.
            for (int i = 0; i < numFrames; i++) {
                bboxX[i] = 0; bboxY[i] = 0;
                bboxW[i] = canvasW; bboxH[i] = canvasH;
                framePixels[i] = frames.get(i).pixels();
            }
        } else {
            // Frame 0: full canvas.
            bboxX[0] = 0; bboxY[0] = 0;
            bboxW[0] = canvasW; bboxH[0] = canvasH;
            framePixels[0] = frames.getFirst().pixels();
            for (int i = 1; i < numFrames; i++) {
                PixelBuffer prev = frames.get(i - 1).pixels();
                PixelBuffer curr = frames.get(i).pixels();
                int[] bbox = FrameDiffUtil.computeBoundingBox(prev, curr);
                if (bbox == null) {
                    // Identical to previous frame: emit a 1x1 placeholder so the
                    // ANMF count matches the input frame count and the duration
                    // contribution is preserved in the animation timeline.
                    bboxX[i] = 0; bboxY[i] = 0;
                    bboxW[i] = 1; bboxH[i] = 1;
                    framePixels[i] = FrameDiffUtil.extractSubBuffer(curr, 0, 0, 1, 1);
                } else {
                    bboxX[i] = bbox[0]; bboxY[i] = bbox[1];
                    bboxW[i] = bbox[2]; bboxH[i] = bbox[3];
                    framePixels[i] = FrameDiffUtil.extractSubBuffer(curr, bbox[0], bbox[1], bbox[2], bbox[3]);
                }
            }
        }

        // Encode all frames. Lossy animated with usePFrames enabled runs sequentially so
        // a shared VP8EncoderSession can carry reference-frame state across frames and
        // emit P-frames for stationary macroblocks. Otherwise each frame is encoded
        // independently (optionally in parallel).
        ConcurrentList<byte[]> encodedPayloads;
        ConcurrentList<byte[]> alphaPayloads;
        if (!lossless && usePFrames) {
            int mvThreads = motionSearchThreads == -1
                ? Runtime.getRuntime().availableProcessors()
                : motionSearchThreads;
            VP8EncoderSession vp8Session = new VP8EncoderSession()
                .withMotionSearchThreads(mvThreads)
                .withAutoSegment(autoSegment);
            int keyInterval = resolveForceKeyframeEvery(forceKeyframeEvery, frames.size());
            encodedPayloads = Concurrent.newList();
            for (int i = 0; i < frames.size(); i++) {
                boolean forceKey = (i == 0)
                    || (keyInterval > 0 && i % keyInterval == 0);
                encodedPayloads.add(vp8Session.encode(framePixels[i], quality, forceKey));
            }
            if (perFrameAlpha) {
                alphaPayloads = Concurrent.newList();
                for (int i = 0; i < numFrames; i++)
                    alphaPayloads.add(encodeAlphaPlane(framePixels[i], true));
            } else {
                alphaPayloads = null;
            }
        } else if (multithreaded) {
            try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
                var futures = IntStream.range(0, numFrames)
                    .mapToObj(i -> CompletableFuture.supplyAsync(
                        () -> encodeFramePixels(framePixels[i], lossless, quality, autoSegment, nearLossless),
                        executor
                    ))
                    .toList();
                encodedPayloads = futures.stream()
                    .map(WebPImageWriter::join)
                    .collect(Concurrent.toList());
                if (perFrameAlpha) {
                    var alphaFutures = IntStream.range(0, numFrames)
                        .mapToObj(i -> CompletableFuture.supplyAsync(() -> encodeAlphaPlane(framePixels[i], true), executor))
                        .toList();
                    alphaPayloads = alphaFutures.stream()
                        .map(WebPImageWriter::join)
                        .collect(Concurrent.toList());
                } else {
                    alphaPayloads = null;
                }
            }
        } else {
            encodedPayloads = Concurrent.newList();
            for (int i = 0; i < numFrames; i++)
                encodedPayloads.add(encodeFramePixels(framePixels[i], lossless, quality, autoSegment, nearLossless));
            if (perFrameAlpha) {
                alphaPayloads = Concurrent.newList();
                for (int i = 0; i < numFrames; i++)
                    alphaPayloads.add(encodeAlphaPlane(framePixels[i], true));
            } else {
                alphaPayloads = null;
            }
        }

        // Assemble RIFF
        ConcurrentList<WebPChunk> chunks = Concurrent.newList();

        // VP8X
        chunks.add(RiffContainer.createChunk(WebPChunk.Type.VP8X,
            buildVP8XPayload(data.getWidth(), data.getHeight(), true, hasAlpha)));

        // ANIM
        chunks.add(RiffContainer.createChunk(WebPChunk.Type.ANIM, buildAnimPayload(data.getBackgroundColor(), animLoopCount)));

        // ANMF frames. When using partial frames we override the input frame's
        // disposal / blend metadata with NO_DISPOSE + NO_BLEND (flag=0x02) so
        // the canvas regions outside each ANMF's rectangle retain the previous
        // frame's pixels and the new frame's pixels overwrite inside the
        // rectangle - the delta-accumulation model that partial frames require.
        // Every frame is a composed canvas, the whole picture its source frame
        // leaves, so overwriting the rectangle that changed reproduces it.
        WebPChunk.Type innerType = lossless ? WebPChunk.Type.VP8L : WebPChunk.Type.VP8;
        for (int i = 0; i < frames.size(); i++) {
            ImageFrame frame = frames.get(i);
            byte[] framePayload = encodedPayloads.get(i);
            byte[] alphaPayload = alphaPayloads != null ? alphaPayloads.get(i) : null;
            int flags = usePartialFrames
                ? 0x02   // NO_DISPOSE + NO_BLEND: partial-frame overwrite
                : computeFrameFlags(frame);
            chunks.add(RiffContainer.createChunk(WebPChunk.Type.ANMF,
                buildAnmfPayload(bboxX[i], bboxY[i], bboxW[i], bboxH[i], frame.delayMs(),
                    flags, framePayload, innerType, alphaPayload)));
        }

        return RiffContainer.write(chunks);
    }

    /**
     * Reads the disposal + blend flags from an {@link ImageFrame}'s metadata.
     */
    private static int computeFrameFlags(@NotNull ImageFrame frame) {
        int flags = 0;
        if (frame.disposal() == FrameDisposal.RESTORE_TO_BACKGROUND) flags |= 0x01;
        if (frame.blend() == FrameBlend.SOURCE) flags |= 0x02;
        return flags;
    }

    /**
     * Encodes a frame's pixel buffer (may be a full-canvas or a partial-frame
     * sub-buffer) into a raw VP8 or VP8L payload. Partial-frame-aware callers
     * pass the already-extracted sub-buffer; full-frame callers pass the
     * canvas pixels directly.
     */
    private byte @NotNull [] encodeFramePixels(@NotNull PixelBuffer pixels, boolean lossless, float quality, boolean autoSegment, int nearLossless) {
        if (lossless)
            return VP8LEncoder.encode(applyNearLosslessIfEnabled(pixels, nearLossless));

        return autoSegment
            ? VP8Encoder.encodeWithAutoSegment(pixels, quality)
            : VP8Encoder.encode(pixels, quality);
    }

    /**
     * Applies libwebp-style near-lossless preprocessing when {@code level < 100},
     * returning a new {@link PixelBuffer} with snapped pixel values. When the
     * level is at its off-default {@code 100}, returns {@code pixels} unchanged -
     * no buffer copy, no behaviour change vs the original lossless path.
     */
    private static @NotNull PixelBuffer applyNearLosslessIfEnabled(@NotNull PixelBuffer pixels, int level) {
        if (level >= 100) return pixels;
        int[] out = NearLosslessPreprocess.apply(
            pixels.data(), pixels.width(), pixels.height(), level);
        return PixelBuffer.of(out, pixels.width(), pixels.height());
    }

    private static byte @NotNull [] buildVP8XPayload(int width, int height, boolean animation, boolean alpha) {
        byte[] payload = new byte[10];
        int flags = 0;
        if (animation) flags |= 0x02;
        if (alpha) flags |= 0x10;
        payload[0] = (byte) flags;

        LittleEndian.writeUint24(payload, 4, width - 1);
        LittleEndian.writeUint24(payload, 7, height - 1);

        return payload;
    }

    /**
     * Resolves the count an {@code ANIM} chunk declares for a loop count.
     * <p>
     * The chunk counts plays, the first included, and {@code 0} plays forever, so it declares one
     * more than the repeats a finite count holds. A count that plays more often than the chunk's
     * 16-bit field holds is declared as {@value #MAX_ANIM_LOOP_COUNT} plays, as libwebp's
     * {@code gif2webp} declares a GIF repeating 65535 times: one play short of what it asks rather
     * than no file at all.
     *
     * @param loopCount the loop count to declare
     * @return the count the chunk declares
     */
    private static int animLoopCount(@NotNull LoopCount loopCount) {
        return Math.min(loopCount.plays().orElse(0), MAX_ANIM_LOOP_COUNT);
    }

    private static byte @NotNull [] buildAnimPayload(int backgroundColor, int loopCount) {
        byte[] payload = new byte[6];
        LittleEndian.writeUint32(payload, 0, backgroundColor);
        LittleEndian.writeUint16(payload, 4, loopCount);
        return payload;
    }

    /**
     * Builds the payload of an ANMF (Animation Frame) chunk.
     * <p>
     * Per the WebP Extended File Format spec, the 16-byte animation header is followed
     * by <i>sub-chunks</i> (an optional ALPH followed by exactly one VP8L or VP8) each
     * with its own 8-byte FourCC + size header plus an optional pad byte to keep the
     * total even-aligned. Writing the raw encoded bitstream directly - without wrapping
     * it in its own FourCC-prefixed sub-chunk - produces a file that libwebp will
     * reject with {@code VP8_STATUS_BITSTREAM_ERROR}.
     *
     * @param alphaPayload optional ALPH chunk payload (compressed alpha bytes); pass
     *                     {@code null} for lossless or alpha-less frames
     */
    private static byte @NotNull [] buildAnmfPayload(
        int pixelOffsetX, int pixelOffsetY, int width, int height, int durationMs,
        int flags,
        byte @NotNull [] frameBitstream,
        @NotNull WebPChunk.Type innerType,
        byte @Nullable [] alphaPayload
    ) {
        int alphaSubSize = alphaPayload != null ? 8 + alphaPayload.length + (alphaPayload.length & 1) : 0;
        int innerPayloadLen = frameBitstream.length;
        int innerPad = innerPayloadLen & 1;
        int innerSubSize = 8 + innerPayloadLen + innerPad;
        byte[] payload = new byte[16 + alphaSubSize + innerSubSize];

        // ANMF stores x/y as (pixelOffset / 2) in 3 bytes each - the actual
        // decoded pixel offset is 2 * raw value, so the offset MUST be even.
        // Callers are expected to have aligned to even coordinates before
        // reaching here (see FrameDiffUtil.computeBoundingBox).
        LittleEndian.writeUint24(payload, 0, pixelOffsetX / 2);
        LittleEndian.writeUint24(payload, 3, pixelOffsetY / 2);
        LittleEndian.writeUint24(payload, 6, width - 1);
        LittleEndian.writeUint24(payload, 9, height - 1);
        LittleEndian.writeUint24(payload, 12, durationMs);
        payload[15] = (byte) flags;

        int offset = 16;
        if (alphaPayload != null)
            offset = writeSubChunk(payload, offset, "ALPH", alphaPayload);
        writeSubChunk(payload, offset, innerType.getFourCC(), frameBitstream);
        return payload;
    }

    /**
     * Writes a {@code FourCC + LE32 size + payload + optional pad} sub-chunk into
     * {@code dst} starting at {@code offset}, returning the offset just past the
     * sub-chunk (including the pad byte).
     */
    private static int writeSubChunk(byte @NotNull [] dst, int offset, @NotNull String fourCC, byte @NotNull [] body) {
        byte[] fc = fourCC.getBytes();
        dst[offset]     = fc[0];
        dst[offset + 1] = fc[1];
        dst[offset + 2] = fc[2];
        dst[offset + 3] = fc[3];
        int size = body.length;
        LittleEndian.writeUint32(dst, offset + 4, size);
        System.arraycopy(body, 0, dst, offset + 8, size);
        // Trailing pad byte (if any) is already zero from array allocation.
        return offset + 8 + size + (size & 1);
    }

    private static byte @NotNull [] encodeAlphaPlane(@NotNull PixelBuffer pixels, boolean compressed) {
        int[] data = pixels.data();
        byte[] rawAlpha = new byte[data.length];

        for (int i = 0; i < data.length; i++)
            rawAlpha[i] = (byte) ((data[i] >> 24) & 0xFF);

        if (compressed) {
            // VP8L-compressed ALPH: alpha bytes are packed into the GREEN channel of a
            // synthetic ARGB image (matching libwebp's WebPDispatchAlphaToGreen, see
            // src/enc/alpha_enc.c), then VP8L-encoded. The standard 5-byte VP8L header
            // (signature + dimensions + version) is stripped because libwebp's ALPH
            // decoder calls DecodeImageStream directly without parsing the standalone
            // VP8L header (see VP8LDecodeAlphaHeader in src/dec/vp8l_dec.c) - dimensions
            // are inferred from the parent VP8 chunk instead.
            PixelBuffer alphaBuf = PixelBuffer.of(
                IntStream.range(0, data.length)
                    .map(i -> 0xFF000000 | ((rawAlpha[i] & 0xFF) << 8))
                    .toArray(),
                pixels.width(), pixels.height()
            );
            // libwebp's ALPH decoder (src/dec/alpha_dec.c's VP8LDecodeAlphaHeader)
            // disallows the predictor / cross-color / subtract-green transforms; it only
            // accepts the color-indexing (palette) transform per WebP spec section
            // "ALPH Chunk". Emit without predictor to keep cross-decode parity.
            byte[] vp8lPayload = encodeAlphaPlane(alphaBuf);
            // Strip the 5-byte VP8L header. The remainder is byte-aligned because the
            // header is exactly 40 bits.
            int headerBytes = 5;
            byte[] result = new byte[1 + (vp8lPayload.length - headerBytes)];
            result[0] = 0x01; // filtering=0, compression=1 (VP8L)
            System.arraycopy(vp8lPayload, headerBytes, result, 1, vp8lPayload.length - headerBytes);
            return result;
        }

        // Uncompressed: header byte + raw alpha data
        byte[] result = new byte[1 + rawAlpha.length];
        result[0] = 0; // filtering=0, compression=0
        System.arraycopy(rawAlpha, 0, result, 1, rawAlpha.length);
        return result;
    }

    /**
     * VP8L-encodes an alpha-plane PixelBuffer without the predictor / cross-color /
     * subtract-green transforms that libwebp's ALPH decoder rejects (see
     * {@code VP8LDecodeAlphaHeader} in libwebp's {@code src/dec/vp8l_dec.c}). Still
     * tries palette vs literal x cache off/on and keeps the smallest output, matching
     * the general {@link VP8LEncoder#encode(PixelBuffer)} heuristic minus the three
     * disallowed transforms.
     */
    private static byte @NotNull [] encodeAlphaPlane(@NotNull PixelBuffer alphaBuf) {
        byte[] best = null;
        for (int cacheBits : new int[] { 0, 10 }) {
            byte[] cand = VP8LEncoder.encode(alphaBuf, VP8LEncoder.TransformMode.NONE, cacheBits);
            if (best == null || cand.length < best.length) best = cand;
            cand = VP8LEncoder.encode(alphaBuf, VP8LEncoder.TransformMode.PALETTE, cacheBits);
            if (cand.length < best.length) best = cand;
        }
        return best;
    }

    /**
     * Waits for an encode and returns its result, throwing what the encode threw rather than the
     * {@link CompletionException} that wraps it.
     *
     * @param future the encode
     * @param <R> the result type
     * @return the encode's result
     */
    @SilentThrows
    private static <R> R join(@NotNull CompletableFuture<R> future) {
        try {
            return future.join();
        } catch (CompletionException ex) {
            throw ex.getCause();
        }
    }

}
