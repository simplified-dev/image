package dev.simplified.image.codec.webp;


import dev.simplified.collection.Concurrent;
import dev.simplified.collection.ConcurrentList;
import dev.simplified.image.AnimatedImageData;
import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.ImageFrame;
import dev.simplified.image.PixelBuffer;
import dev.simplified.image.codec.ImageWriteOptions;
import dev.simplified.image.codec.ImageWriter;
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

        if (options instanceof WebPWriteOptions webpOptions) {
            lossless = webpOptions.isLossless();
            quality = webpOptions.getQuality();
            loopCount = webpOptions.getLoopCount();
            multithreaded = webpOptions.isMultithreaded();
            alphaCompression = webpOptions.isAlphaCompression();
        }

        if (data instanceof AnimatedImageData animated) {
            if (loopCount == 0 && animated.getLoopCount() != 0)
                loopCount = animated.getLoopCount();

            return writeAnimated(animated, lossless, quality, loopCount, multithreaded);
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
            chunks.add(RiffContainer.createChunk(WebPChunkType.VP8L, payload));
            return RiffContainer.write(chunks);
        }

        payload = VP8Encoder.encode(pixels, quality);
        boolean hasAlpha = data.hasAlpha();

        ConcurrentList<WebPChunk> chunks = Concurrent.newList();

        if (hasAlpha) {
            // Extended: VP8X + ALPH + VP8
            chunks.add(RiffContainer.createChunk(WebPChunkType.VP8X, buildVP8XPayload(pixels.width(), pixels.height(), false, true)));
            chunks.add(RiffContainer.createChunk(WebPChunkType.ALPH, encodeAlphaPlane(pixels, alphaCompression)));
            chunks.add(RiffContainer.createChunk(WebPChunkType.VP8, payload));
        } else {
            chunks.add(RiffContainer.createChunk(WebPChunkType.VP8, payload));
        }

        return RiffContainer.write(chunks);
    }

    private byte @NotNull [] writeAnimated(
        @NotNull AnimatedImageData data,
        boolean lossless,
        float quality,
        int loopCount,
        boolean multithreaded
    ) {
        ConcurrentList<ImageFrame> frames = data.getFrames();

        // Encode all frames (optionally in parallel)
        ConcurrentList<byte[]> encodedPayloads;
        if (multithreaded) {
            var futures = frames.stream()
                .map(frame -> CompletableFuture.supplyAsync(
                    () -> encodeFrame(frame, lossless, quality),
                    java.util.concurrent.Executors.newVirtualThreadPerTaskExecutor()
                ))
                .collect(Concurrent.toList());
            encodedPayloads = futures.stream()
                .map(CompletableFuture::join)
                .collect(Concurrent.toList());
        } else {
            encodedPayloads = frames.stream()
                .map(frame -> encodeFrame(frame, lossless, quality))
                .collect(Concurrent.toList());
        }

        // Assemble RIFF
        ConcurrentList<WebPChunk> chunks = Concurrent.newList();

        // VP8X
        chunks.add(RiffContainer.createChunk(WebPChunkType.VP8X,
            buildVP8XPayload(data.getWidth(), data.getHeight(), true, data.hasAlpha())));

        // ANIM
        chunks.add(RiffContainer.createChunk(WebPChunkType.ANIM, buildAnimPayload(data.getBackgroundColor(), loopCount)));

        // ANMF frames
        for (int i = 0; i < frames.size(); i++) {
            ImageFrame frame = frames.get(i);
            byte[] framePayload = encodedPayloads.get(i);
            chunks.add(RiffContainer.createChunk(WebPChunkType.ANMF,
                buildAnmfPayload(frame, framePayload)));
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

    private static byte @NotNull [] buildAnmfPayload(@NotNull ImageFrame frame, byte @NotNull [] frameBitstream) {
        byte[] payload = new byte[16 + frameBitstream.length];

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
        if (frame.disposal() == ImageFrame.Disposal.RESTORE_TO_BACKGROUND) flags |= 0x01;
        if (frame.blend() == ImageFrame.Blend.SOURCE) flags |= 0x02;
        payload[15] = (byte) flags;

        System.arraycopy(frameBitstream, 0, payload, 16, frameBitstream.length);
        return payload;
    }

    private static byte @NotNull [] encodeAlphaPlane(@NotNull PixelBuffer pixels, boolean compressed) {
        int[] data = pixels.getPixels();
        byte[] rawAlpha = new byte[data.length];

        for (int i = 0; i < data.length; i++)
            rawAlpha[i] = (byte) ((data[i] >> 24) & 0xFF);

        if (compressed) {
            // VP8L-compressed alpha: header byte with compression flag + VP8L bitstream of alpha channel
            byte[] header = new byte[]{0x01}; // filtering=0, compression=1 (VP8L)
            PixelBuffer alphaBuf = PixelBuffer.of(
                java.util.stream.IntStream.range(0, data.length)
                    .map(i -> (rawAlpha[i] & 0xFF) << 24) // pack alpha into ARGB alpha channel
                    .toArray(),
                pixels.width(), pixels.height()
            );
            byte[] vp8lPayload = VP8LEncoder.encode(alphaBuf);
            byte[] result = new byte[1 + vp8lPayload.length];
            result[0] = header[0];
            System.arraycopy(vp8lPayload, 0, result, 1, vp8lPayload.length);
            return result;
        }

        // Uncompressed: header byte + raw alpha data
        byte[] result = new byte[1 + rawAlpha.length];
        result[0] = 0; // filtering=0, compression=0
        System.arraycopy(rawAlpha, 0, result, 1, rawAlpha.length);
        return result;
    }

}
