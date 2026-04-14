package dev.simplified.image.codec.webp;

import dev.simplified.collection.Concurrent;
import dev.simplified.collection.ConcurrentList;
import dev.simplified.image.data.AnimatedImageData;
import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.data.ImageFrame;
import dev.simplified.image.data.FrameBlend;
import dev.simplified.image.data.FrameDisposal;
import dev.simplified.image.pixel.PixelBuffer;
import dev.simplified.image.data.StaticImageData;
import dev.simplified.image.codec.ImageReadOptions;
import dev.simplified.image.codec.ImageReader;
import dev.simplified.image.codec.webp.lossless.VP8LDecoder;
import dev.simplified.image.codec.webp.lossy.VP8Decoder;
import dev.simplified.image.codec.webp.lossy.VP8DecoderSession;
import dev.simplified.image.exception.ImageDecodeException;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

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
        WebPChunk vp8xChunk = findChunk(chunks, WebPChunk.Type.VP8X);

        if (vp8xChunk != null)
            return readExtended(chunks, vp8xChunk);

        // Simple lossy (VP8)
        WebPChunk vp8Chunk = findChunk(chunks, WebPChunk.Type.VP8);

        if (vp8Chunk != null)
            return decodeVP8(vp8Chunk);

        // Simple lossless (VP8L)
        WebPChunk vp8lChunk = findChunk(chunks, WebPChunk.Type.VP8L);

        if (vp8lChunk != null)
            return decodeVP8L(vp8lChunk);

        throw new ImageDecodeException("WebP file contains no VP8, VP8L, or VP8X chunk");
    }

    private @NotNull ImageData readExtended(@NotNull ConcurrentList<WebPChunk> chunks, @NotNull WebPChunk vp8x) {
        byte[] payload = vp8x.payload();

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
            WebPChunk vp8l = findChunk(chunks, WebPChunk.Type.VP8L);
            if (vp8l != null) return decodeVP8L(vp8l);

            WebPChunk vp8 = findChunk(chunks, WebPChunk.Type.VP8);
            WebPChunk alph = hasAlpha ? findChunk(chunks, WebPChunk.Type.ALPH) : null;

            if (vp8 != null)
                return decodeVP8WithAlpha(vp8, alph);

            throw new ImageDecodeException("Extended WebP has no image data");
        }

        // Animated
        WebPChunk animChunk = findChunk(chunks, WebPChunk.Type.ANIM);
        int backgroundColor = 0;
        int loopCount = 0;

        if (animChunk != null && animChunk.payloadLength() >= 6) {
            byte[] animPayload = animChunk.payload();
            backgroundColor = readLE32(animPayload, 0);
            loopCount = (animPayload[4] & 0xFF) | ((animPayload[5] & 0xFF) << 8);
        }

        ConcurrentList<ImageFrame> frames = Concurrent.newList();

        // Shared VP8 decoder session for animated lossy frames - enables P-frame decoding
        // where a frame references the prior frame's reconstruction.
        VP8DecoderSession vp8Session = new VP8DecoderSession();

        for (WebPChunk chunk : chunks) {
            if (chunk.type() != WebPChunk.Type.ANMF) continue;

            byte[] anmf = chunk.payload();

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

            // Frame body at offset 16 is a sequence of sub-chunks (4-byte FourCC + 4-byte
            // LE size + payload + optional pad byte), per WebP Extended File Format:
            // an optional ALPH sub-chunk followed by either VP8 or VP8L.
            byte[] vp8Sub = null;
            byte[] vp8lSub = null;
            byte[] alphSub = null;
            int subOffset = 16;
            while (subOffset + 8 <= anmf.length) {
                String tag = new String(anmf, subOffset, 4);
                int subSize = readLE32(anmf, subOffset + 4);
                int dataStart = subOffset + 8;
                if (dataStart + subSize > anmf.length)
                    throw new ImageDecodeException("ANMF sub-chunk %s overruns frame", tag);
                byte[] subPayload = new byte[subSize];
                System.arraycopy(anmf, dataStart, subPayload, 0, subSize);
                switch (tag) {
                    case "VP8 " -> vp8Sub = subPayload;
                    case "VP8L" -> vp8lSub = subPayload;
                    case "ALPH" -> alphSub = subPayload;
                    default -> { /* unknown sub-chunk, ignore for forward compatibility */ }
                }
                subOffset = dataStart + subSize + (subSize & 1);
            }

            PixelBuffer framePixels;
            if (vp8lSub != null) {
                framePixels = VP8LDecoder.decode(vp8lSub);
            } else if (vp8Sub != null) {
                framePixels = vp8Session.decode(vp8Sub);
                if (alphSub != null)
                    mergeAlphaPlane(framePixels, alphSub);
            } else {
                throw new ImageDecodeException("ANMF frame has no VP8 or VP8L sub-chunk");
            }

            if (framePixels.width() != frameW || framePixels.height() != frameH)
                throw new ImageDecodeException(
                    "ANMF frame size mismatch: declared %dx%d but decoded %dx%d",
                    frameW, frameH, framePixels.width(), framePixels.height()
                );

            frames.add(ImageFrame.of(framePixels, durationMs, frameX, frameY, disposal, blend));
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
        byte[] payload = chunk.payload();
        return StaticImageData.of(VP8Decoder.decode(payload));
    }

    private @NotNull StaticImageData decodeVP8L(@NotNull WebPChunk chunk) {
        byte[] payload = chunk.payload();
        return StaticImageData.of(VP8LDecoder.decode(payload));
    }

    private @NotNull StaticImageData decodeVP8WithAlpha(@NotNull WebPChunk vp8, @Nullable WebPChunk alph) {
        PixelBuffer colorPixels = VP8Decoder.decode(vp8.payload());
        if (alph != null)
            mergeAlphaPlane(colorPixels, alph.payload());
        return StaticImageData.of(colorPixels);
    }

    /**
     * Merges an ALPH chunk payload into {@code colorPixels}'s alpha channel. Supports the
     * uncompressed ({@code compression=0}) form and the VP8L-compressed
     * ({@code compression=1}) form where alpha bytes are carried in the green channel of
     * the VP8L bitstream (libwebp's {@code WebPDispatchAlphaToGreen}).
     */
    private static void mergeAlphaPlane(@NotNull PixelBuffer colorPixels, byte @NotNull [] alphPayload) {
        if (alphPayload.length < 2) return;

        int header = alphPayload[0] & 0xFF;
        int compression = header & 0x03;
        int[] pixels = colorPixels.pixels();

        if (compression == 0) {
            int alphaOffset = 1;
            for (int i = 0; i < pixels.length && alphaOffset + i < alphPayload.length; i++)
                pixels[i] = (pixels[i] & 0x00FFFFFF) | ((alphPayload[alphaOffset + i] & 0xFF) << 24);
        } else if (compression == 1) {
            // ALPH-internal VP8L omits the standard 5-byte header (libwebp's
            // VP8LDecodeAlphaHeader takes dimensions from the parent VP8 chunk).
            // Synthesize the header from colorPixels' size before delegating to our
            // standalone VP8L decoder.
            byte[] alphBitstream = prependVp8lHeader(alphPayload, 1, colorPixels.width(), colorPixels.height());
            PixelBuffer alphDecoded = VP8LDecoder.decode(alphBitstream);
            int[] alphPixels = alphDecoded.pixels();
            for (int i = 0; i < pixels.length && i < alphPixels.length; i++)
                pixels[i] = (pixels[i] & 0x00FFFFFF) | (((alphPixels[i] >> 8) & 0xFF) << 24);
        }
    }

    /**
     * Prepends a synthetic 5-byte VP8L header
     * ({@code 0x2F | 14b w-1 | 14b h-1 | 1b alpha | 3b version}) to {@code alphPayload},
     * skipping the 1-byte ALPH chunk header. Lets our {@link VP8LDecoder} consume an
     * ALPH-internal VP8L stream that omits the standard header.
     */
    private static byte @NotNull [] prependVp8lHeader(byte @NotNull [] alphPayload, int payloadStart, int width, int height) {
        int bodyLen = alphPayload.length - payloadStart;
        byte[] result = new byte[5 + bodyLen];
        result[0] = 0x2F;
        int wM1 = width - 1;
        int hM1 = height - 1;
        // Bit-packed LE: bits 0..13 = wM1, 14..27 = hM1, 28 = alpha=0, 29..31 = version=0.
        result[1] = (byte) (wM1 & 0xFF);
        result[2] = (byte) (((wM1 >> 8) & 0x3F) | ((hM1 & 0x03) << 6));
        result[3] = (byte) ((hM1 >> 2) & 0xFF);
        result[4] = (byte) ((hM1 >> 10) & 0x0F);
        System.arraycopy(alphPayload, payloadStart, result, 5, bodyLen);
        return result;
    }

    private @NotNull PixelBuffer decodeFrameBitstream(byte @NotNull [] bitstream) {
        // Detect if frame is VP8L or VP8 based on signature
        if (bitstream.length > 0 && bitstream[0] == 0x2F)
            return VP8LDecoder.decode(bitstream);

        return VP8Decoder.decode(bitstream);
    }

    private static @Nullable WebPChunk findChunk(@NotNull ConcurrentList<WebPChunk> chunks, @NotNull WebPChunk.Type type) {
        return chunks.stream()
            .filter(chunk -> chunk.type() == type)
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
