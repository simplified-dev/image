package dev.simplified.image.codec.webp;


import dev.simplified.collection.Concurrent;
import dev.simplified.collection.ConcurrentList;
import dev.simplified.image.data.AnimatedImageData;
import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.data.FrameBlend;
import dev.simplified.image.data.FrameDisposal;
import dev.simplified.image.data.ImageFrame;
import dev.simplified.image.pixel.PixelBuffer;
import dev.simplified.image.codec.ImageWriteOptions;
import dev.simplified.image.codec.ImageWriter;
import dev.simplified.image.codec.webp.lossless.VP8LEncoder;
import dev.simplified.image.codec.webp.lossy.VP8Encoder;
import dev.simplified.image.codec.webp.lossy.VP8EncoderSession;
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

        if (options instanceof WebPWriteOptions webpOptions) {
            lossless = webpOptions.isLossless();
            quality = webpOptions.getQuality();
            loopCount = webpOptions.getLoopCount();
            multithreaded = webpOptions.isMultithreaded();
            alphaCompression = webpOptions.isAlphaCompression();
            usePFrames = webpOptions.isUsePFrames();
        }

        if (data instanceof AnimatedImageData animated) {
            if (loopCount == 0 && animated.getLoopCount() != 0)
                loopCount = animated.getLoopCount();

            return writeAnimated(animated, lossless, quality, loopCount, multithreaded, usePFrames);
        }

        return writeStatic(data, lossless, quality, alphaCompression);
    }

    private byte @NotNull [] writeStatic(@NotNull ImageData data, boolean lossless, float quality, boolean alphaCompression) {
        PixelBuffer pixels = data.toPixelBuffer();
        byte[] payload;

        if (lossless) {
            payload = VP8LEncoder.encode(pixels);
            // Simple lossless: just RIFF [ VP8L ]
            ConcurrentList<WebPChunk> chunks = Concurrent.newList();
            chunks.add(RiffContainer.createChunk(WebPChunk.Type.VP8L, payload));
            return RiffContainer.write(chunks);
        }

        payload = VP8Encoder.encode(pixels, quality);
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
        boolean usePFrames
    ) {
        ConcurrentList<ImageFrame> frames = data.getFrames();
        boolean hasAlpha = data.hasAlpha();
        // Per-frame lossy + alpha: emit an ALPH sub-chunk before the VP8 sub-chunk.
        boolean perFrameAlpha = hasAlpha && !lossless;

        // Encode all frames. Lossy animated with usePFrames enabled runs sequentially so
        // a shared VP8EncoderSession can carry reference-frame state across frames and
        // emit P-frames for stationary macroblocks. Otherwise the existing per-frame
        // independent path is used (optionally parallel).
        ConcurrentList<byte[]> encodedPayloads;
        ConcurrentList<byte[]> alphaPayloads;
        if (!lossless && usePFrames) {
            VP8EncoderSession vp8Session = new VP8EncoderSession();
            encodedPayloads = Concurrent.newList();
            for (int i = 0; i < frames.size(); i++) {
                ImageFrame frame = frames.get(i);
                boolean forceKey = (i == 0);
                encodedPayloads.add(vp8Session.encode(frame.pixels(), quality, forceKey));
            }
            if (perFrameAlpha) {
                alphaPayloads = frames.stream()
                    .map(frame -> encodeAlphaPlane(frame.pixels(), true))
                    .collect(Concurrent.toList());
            } else {
                alphaPayloads = null;
            }
        } else if (multithreaded) {
            var futures = frames.stream()
                .map(frame -> CompletableFuture.supplyAsync(
                    () -> encodeFrame(frame, lossless, quality),
                    java.util.concurrent.Executors.newVirtualThreadPerTaskExecutor()
                ))
                .collect(Concurrent.toList());
            encodedPayloads = futures.stream()
                .map(CompletableFuture::join)
                .collect(Concurrent.toList());
            if (perFrameAlpha) {
                var alphaFutures = frames.stream()
                    .map(frame -> CompletableFuture.supplyAsync(
                        () -> encodeAlphaPlane(frame.pixels(), true),
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
            encodedPayloads = frames.stream()
                .map(frame -> encodeFrame(frame, lossless, quality))
                .collect(Concurrent.toList());
            if (perFrameAlpha) {
                alphaPayloads = frames.stream()
                    .map(frame -> encodeAlphaPlane(frame.pixels(), true))
                    .collect(Concurrent.toList());
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

        // ANMF frames
        WebPChunk.Type innerType = lossless ? WebPChunk.Type.VP8L : WebPChunk.Type.VP8;
        for (int i = 0; i < frames.size(); i++) {
            ImageFrame frame = frames.get(i);
            byte[] framePayload = encodedPayloads.get(i);
            byte[] alphaPayload = alphaPayloads != null ? alphaPayloads.get(i) : null;
            chunks.add(RiffContainer.createChunk(WebPChunk.Type.ANMF,
                buildAnmfPayload(frame, framePayload, innerType, alphaPayload)));
        }

        return RiffContainer.write(chunks);
    }

    private byte @NotNull [] encodeFrame(@NotNull ImageFrame frame, boolean lossless, float quality) {
        PixelBuffer pixels = frame.pixels();

        if (lossless)
            return VP8LEncoder.encode(pixels);

        return VP8Encoder.encode(pixels, quality);
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
        @NotNull ImageFrame frame,
        byte @NotNull [] frameBitstream,
        @NotNull WebPChunk.Type innerType,
        byte @org.jetbrains.annotations.Nullable [] alphaPayload
    ) {
        int alphaSubSize = alphaPayload != null ? 8 + alphaPayload.length + (alphaPayload.length & 1) : 0;
        int innerPayloadLen = frameBitstream.length;
        int innerPad = innerPayloadLen & 1;
        int innerSubSize = 8 + innerPayloadLen + innerPad;
        byte[] payload = new byte[16 + alphaSubSize + innerSubSize];

        int x = frame.offsetX() / 2;
        int y = frame.offsetY() / 2;
        int w = frame.pixels().width() - 1;
        int h = frame.pixels().height() - 1;
        int dur = frame.delayMs();

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
        payload[12] = (byte) (dur & 0xFF);
        payload[13] = (byte) ((dur >> 8) & 0xFF);
        payload[14] = (byte) ((dur >> 16) & 0xFF);

        int flags = 0;
        if (frame.disposal() == FrameDisposal.RESTORE_TO_BACKGROUND) flags |= 0x01;
        if (frame.blend() == FrameBlend.SOURCE) flags |= 0x02;
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
        int[] data = pixels.pixels();
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
            byte[] vp8lPayload = VP8LEncoder.encode(alphaBuf);
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

}
