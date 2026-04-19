package dev.simplified.image.codec.webp;


import dev.simplified.collection.Concurrent;
import dev.simplified.collection.ConcurrentList;
import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.codec.ImageWriteOptions;
import dev.simplified.image.codec.ImageWriter;
import dev.simplified.image.codec.webp.lossless.NearLosslessPreprocess;
import dev.simplified.image.codec.webp.lossless.VP8LEncoder;
import dev.simplified.image.codec.webp.lossy.VP8Encoder;
import dev.simplified.image.codec.webp.lossy.VP8EncoderSession;
import dev.simplified.image.data.AnimatedImageData;
import dev.simplified.image.data.FrameBlend;
import dev.simplified.image.data.FrameDisposal;
import dev.simplified.image.data.ImageFrame;
import dev.simplified.image.pixel.PixelBuffer;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.concurrent.CompletableFuture;

/**
 * Writes WebP images (static and animated) using a pure Java implementation
 * of the VP8L lossless and VP8 lossy codecs.
 */
public class WebPImageWriter implements ImageWriter {

    @Override
    public @NotNull ImageFormat getFormat() {
        return ImageFormat.WEBP;
    }

    @Override
    public byte @NotNull [] write(@NotNull ImageData data, @Nullable ImageWriteOptions options) {
        boolean lossless = true;
        float quality = 0.75f;
        int loopCount = 0;
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

        if (data instanceof AnimatedImageData animated) {
            if (loopCount == 0 && animated.getLoopCount() != 0)
                loopCount = animated.getLoopCount();

            return writeAnimated(animated, lossless, quality, loopCount, multithreaded, usePFrames, forceKeyframeEvery, motionSearchThreads, autoSegment, nearLossless);
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
        PixelBuffer pixels = data.toPixelBuffer();
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
        int loopCount,
        boolean multithreaded,
        boolean usePFrames,
        int forceKeyframeEvery,
        int motionSearchThreads,
        boolean autoSegment,
        int nearLossless
    ) {
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
        // GIF89a both do, and removes the "animated WebP shows as static" issue
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
            framePixels[0] = frames.get(0).pixels();
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
        // emit P-frames for stationary macroblocks. Otherwise the existing per-frame
        // independent path is used (optionally parallel).
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
            PixelBuffer[] fpForLambda = framePixels;
            var futures = java.util.stream.IntStream.range(0, numFrames)
                .mapToObj(i -> CompletableFuture.supplyAsync(
                    () -> encodeFramePixels(fpForLambda[i], lossless, quality, autoSegment, nearLossless),
                    java.util.concurrent.Executors.newVirtualThreadPerTaskExecutor()
                ))
                .collect(Concurrent.toList());
            encodedPayloads = futures.stream()
                .map(CompletableFuture::join)
                .collect(Concurrent.toList());
            if (perFrameAlpha) {
                var alphaFutures = java.util.stream.IntStream.range(0, numFrames)
                    .mapToObj(i -> CompletableFuture.supplyAsync(
                        () -> encodeAlphaPlane(fpForLambda[i], true),
                        java.util.concurrent.Executors.newVirtualThreadPerTaskExecutor()
                    ))
                    .collect(Concurrent.toList());
                alphaPayloads = alphaFutures.stream()
                    .map(CompletableFuture::join)
                    .collect(Concurrent.toList());
            } else {
                alphaPayloads = null;
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
        chunks.add(RiffContainer.createChunk(WebPChunk.Type.ANIM, buildAnimPayload(data.getBackgroundColor(), loopCount)));

        // ANMF frames. When using partial frames we override the input frame's
        // disposal / blend metadata with NO_DISPOSE + NO_BLEND (flag=0x02) so
        // the canvas regions outside each ANMF's rectangle retain the previous
        // frame's pixels and the new frame's pixels overwrite inside the
        // rectangle - the delta-accumulation model that partial frames require.
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

    /** Reads the disposal + blend flags from an {@link ImageFrame}'s metadata. */
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

        int w = width - 1;
        int h = height - 1;
        payload[4] = (byte) (w & 0xFF);
        payload[5] = (byte) ((w >> 8) & 0xFF);
        payload[6] = (byte) ((w >> 16) & 0xFF);
        payload[7] = (byte) (h & 0xFF);
        payload[8] = (byte) ((h >> 8) & 0xFF);
        payload[9] = (byte) ((h >> 16) & 0xFF);

        return payload;
    }

    private static byte @NotNull [] buildAnimPayload(int backgroundColor, int loopCount) {
        byte[] payload = new byte[6];
        payload[0] = (byte) (backgroundColor & 0xFF);
        payload[1] = (byte) ((backgroundColor >> 8) & 0xFF);
        payload[2] = (byte) ((backgroundColor >> 16) & 0xFF);
        payload[3] = (byte) ((backgroundColor >> 24) & 0xFF);
        payload[4] = (byte) (loopCount & 0xFF);
        payload[5] = (byte) ((loopCount >> 8) & 0xFF);
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
        byte @org.jetbrains.annotations.Nullable [] alphaPayload
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
        int x = pixelOffsetX / 2;
        int y = pixelOffsetY / 2;
        int w = width - 1;
        int h = height - 1;

        payload[0] = (byte) (x & 0xFF);
        payload[1] = (byte) ((x >> 8) & 0xFF);
        payload[2] = (byte) ((x >> 16) & 0xFF);
        payload[3] = (byte) (y & 0xFF);
        payload[4] = (byte) ((y >> 8) & 0xFF);
        payload[5] = (byte) ((y >> 16) & 0xFF);
        payload[6] = (byte) (w & 0xFF);
        payload[7] = (byte) ((w >> 8) & 0xFF);
        payload[8] = (byte) ((w >> 16) & 0xFF);
        payload[9] = (byte) (h & 0xFF);
        payload[10] = (byte) ((h >> 8) & 0xFF);
        payload[11] = (byte) ((h >> 16) & 0xFF);
        payload[12] = (byte) (durationMs & 0xFF);
        payload[13] = (byte) ((durationMs >> 8) & 0xFF);
        payload[14] = (byte) ((durationMs >> 16) & 0xFF);
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
        dst[offset + 4] = (byte) (size & 0xFF);
        dst[offset + 5] = (byte) ((size >> 8) & 0xFF);
        dst[offset + 6] = (byte) ((size >> 16) & 0xFF);
        dst[offset + 7] = (byte) ((size >> 24) & 0xFF);
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
                java.util.stream.IntStream.range(0, data.length)
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

}
