package dev.sbs.api.io.image.codec.webp;

import dev.sbs.api.collection.concurrent.Concurrent;
import dev.sbs.api.collection.concurrent.ConcurrentList;
import dev.sbs.api.io.image.AnimatedImageData;
import dev.sbs.api.io.image.ImageData;
import dev.sbs.api.io.image.ImageFormat;
import dev.sbs.api.io.image.ImageFrame;
import dev.sbs.api.io.image.ImageFrame.FrameBlend;
import dev.sbs.api.io.image.ImageFrame.FrameDisposal;
import dev.sbs.api.io.image.PixelBuffer;
import dev.sbs.api.io.image.StaticImageData;
import dev.sbs.api.io.image.codec.ImageReadOptions;
import dev.sbs.api.io.image.codec.ImageReader;
import dev.sbs.api.io.image.exception.ImageDecodeException;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.awt.image.BufferedImage;

/**
 * Reads WebP images (static and animated, lossless and lossy) using a pure Java
 * implementation of the VP8L and VP8 codecs.
 */
public class WebPImageReader implements ImageReader {

    @Override
    public @NotNull ImageFormat getFormat() {
        return ImageFormat.WEBP;
    }

    @Override
    public boolean canRead(byte @NotNull [] data) {
        return ImageFormat.WEBP.matches(data);
    }

    @Override
    public @NotNull ImageData read(byte @NotNull [] data, @Nullable ImageReadOptions options) {
        ConcurrentList<WebPChunk> chunks = RiffContainer.parse(data);

        if (chunks.isEmpty())
            throw new ImageDecodeException("WebP file contains no chunks");

        // Check for VP8X extended header
        WebPChunk vp8xChunk = findChunk(chunks, WebPChunkType.VP8X);

        if (vp8xChunk != null)
            return readExtended(chunks, vp8xChunk);

        // Simple lossy (VP8)
        WebPChunk vp8Chunk = findChunk(chunks, WebPChunkType.VP8);

        if (vp8Chunk != null)
            return decodeVP8(vp8Chunk);

        // Simple lossless (VP8L)
        WebPChunk vp8lChunk = findChunk(chunks, WebPChunkType.VP8L);

        if (vp8lChunk != null)
            return decodeVP8L(vp8lChunk);

        throw new ImageDecodeException("WebP file contains no VP8, VP8L, or VP8X chunk");
    }

    private @NotNull ImageData readExtended(@NotNull ConcurrentList<WebPChunk> chunks, @NotNull WebPChunk vp8x) {
        byte[] payload = vp8x.getPayload();

        if (payload.length < 10)
            throw new ImageDecodeException("VP8X chunk too short");

        int flags = payload[0] & 0xFF;
        boolean hasAnimation = (flags & 0x02) != 0;
        boolean hasAlpha = (flags & 0x10) != 0;

        int canvasWidth = 1 + ((payload[4] & 0xFF) | ((payload[5] & 0xFF) << 8) | ((payload[6] & 0xFF) << 16));
        int canvasHeight = 1 + ((payload[7] & 0xFF) | ((payload[8] & 0xFF) << 8) | ((payload[9] & 0xFF) << 16));

        if (canvasWidth > 16383 || canvasHeight > 16383)
            throw new ImageDecodeException("VP8X canvas dimensions exceed spec maximum: %dx%d", canvasWidth, canvasHeight);

        if (!hasAnimation) {
            // Static extended image (may have alpha + VP8 or VP8L)
            WebPChunk vp8l = findChunk(chunks, WebPChunkType.VP8L);
            if (vp8l != null) return decodeVP8L(vp8l);

            WebPChunk vp8 = findChunk(chunks, WebPChunkType.VP8);
            WebPChunk alph = hasAlpha ? findChunk(chunks, WebPChunkType.ALPH) : null;

            if (vp8 != null)
                return decodeVP8WithAlpha(vp8, alph);

            throw new ImageDecodeException("Extended WebP has no image data");
        }

        // Animated
        WebPChunk animChunk = findChunk(chunks, WebPChunkType.ANIM);
        int backgroundColor = 0;
        int loopCount = 0;

        if (animChunk != null && animChunk.getPayloadLength() >= 6) {
            byte[] animPayload = animChunk.getPayload();
            backgroundColor = readLE32(animPayload, 0);
            loopCount = (animPayload[4] & 0xFF) | ((animPayload[5] & 0xFF) << 8);
        }

        ConcurrentList<ImageFrame> frames = Concurrent.newList();

        for (WebPChunk chunk : chunks) {
            if (chunk.getType() != WebPChunkType.ANMF) continue;

            byte[] anmf = chunk.getPayload();

            if (anmf.length < 16)
                throw new ImageDecodeException("ANMF chunk too short");

            int frameX = 2 * ((anmf[0] & 0xFF) | ((anmf[1] & 0xFF) << 8) | ((anmf[2] & 0xFF) << 16));
            int frameY = 2 * ((anmf[3] & 0xFF) | ((anmf[4] & 0xFF) << 8) | ((anmf[5] & 0xFF) << 16));
            int frameW = 1 + ((anmf[6] & 0xFF) | ((anmf[7] & 0xFF) << 8) | ((anmf[8] & 0xFF) << 16));
            int frameH = 1 + ((anmf[9] & 0xFF) | ((anmf[10] & 0xFF) << 8) | ((anmf[11] & 0xFF) << 16));
            int durationMs = (anmf[12] & 0xFF) | ((anmf[13] & 0xFF) << 8) | ((anmf[14] & 0xFF) << 16);
            int flagsByte = anmf[15] & 0xFF;

            FrameDisposal disposal = (flagsByte & 0x01) != 0
                ? FrameDisposal.RESTORE_TO_BACKGROUND
                : FrameDisposal.DO_NOT_DISPOSE;
            FrameBlend blend = (flagsByte & 0x02) != 0
                ? FrameBlend.SOURCE
                : FrameBlend.OVER;

            // Frame bitstream starts at offset 16
            byte[] frameBitstream = new byte[anmf.length - 16];
            System.arraycopy(anmf, 16, frameBitstream, 0, frameBitstream.length);

            BufferedImage frameImage = decodeFrameBitstream(frameBitstream);

            if (frameImage.getWidth() != frameW || frameImage.getHeight() != frameH)
                throw new ImageDecodeException(
                    "ANMF frame size mismatch: declared %dx%d but decoded %dx%d",
                    frameW, frameH, frameImage.getWidth(), frameImage.getHeight()
                );

            frames.add(ImageFrame.of(frameImage, durationMs, frameX, frameY, disposal, blend));
        }

        if (frames.isEmpty())
            throw new ImageDecodeException("Animated WebP contains no ANMF frames");

        return AnimatedImageData.builder()
            .withWidth(canvasWidth)
            .withHeight(canvasHeight)
            .withFrames(frames)
            .withLoopCount(loopCount)
            .withBackgroundColor(backgroundColor)
            .build();
    }

    private @NotNull StaticImageData decodeVP8(@NotNull WebPChunk chunk) {
        byte[] payload = chunk.getPayload();
        PixelBuffer pixels = VP8Decoder.decode(payload);
        return StaticImageData.of(pixels.toBufferedImage());
    }

    private @NotNull StaticImageData decodeVP8L(@NotNull WebPChunk chunk) {
        byte[] payload = chunk.getPayload();
        PixelBuffer pixels = VP8LDecoder.decode(payload);
        return StaticImageData.of(pixels.toBufferedImage());
    }

    private @NotNull StaticImageData decodeVP8WithAlpha(@NotNull WebPChunk vp8, @Nullable WebPChunk alph) {
        PixelBuffer colorPixels = VP8Decoder.decode(vp8.getPayload());

        if (alph == null)
            return StaticImageData.of(colorPixels.toBufferedImage());

        // Merge alpha plane into decoded color data
        byte[] alphPayload = alph.getPayload();

        if (alphPayload.length < 2)
            return StaticImageData.of(colorPixels.toBufferedImage());

        int header = alphPayload[0] & 0xFF;
        int compression = header & 0x03;
        int[] pixels = colorPixels.getPixels();

        if (compression == 0) {
            // Uncompressed alpha
            int alphaOffset = 1;
            for (int i = 0; i < pixels.length && alphaOffset + i < alphPayload.length; i++)
                pixels[i] = (pixels[i] & 0x00FFFFFF) | ((alphPayload[alphaOffset + i] & 0xFF) << 24);
        } else if (compression == 1) {
            // VP8L-compressed alpha
            byte[] alphBitstream = new byte[alphPayload.length - 1];
            System.arraycopy(alphPayload, 1, alphBitstream, 0, alphBitstream.length);
            PixelBuffer alphDecoded = VP8LDecoder.decode(alphBitstream);
            int[] alphPixels = alphDecoded.getPixels();
            for (int i = 0; i < pixels.length && i < alphPixels.length; i++)
                pixels[i] = (pixels[i] & 0x00FFFFFF) | (alphPixels[i] & 0xFF000000);
        }

        return StaticImageData.of(colorPixels.toBufferedImage());
    }

    private @NotNull BufferedImage decodeFrameBitstream(byte @NotNull [] bitstream) {
        // Detect if frame is VP8L or VP8 based on signature
        if (bitstream.length > 0 && bitstream[0] == 0x2F) {
            // VP8L signature
            return VP8LDecoder.decode(bitstream).toBufferedImage();
        }

        // VP8 lossy
        return VP8Decoder.decode(bitstream).toBufferedImage();
    }

    private static @Nullable WebPChunk findChunk(@NotNull ConcurrentList<WebPChunk> chunks, @NotNull WebPChunkType type) {
        return chunks.stream()
            .filter(chunk -> chunk.getType() == type)
            .findFirst()
            .orElse(null);
    }

    private static int readLE32(byte @NotNull [] data, int offset) {
        return (data[offset] & 0xFF)
            | ((data[offset + 1] & 0xFF) << 8)
            | ((data[offset + 2] & 0xFF) << 16)
            | ((data[offset + 3] & 0xFF) << 24);
    }

}
