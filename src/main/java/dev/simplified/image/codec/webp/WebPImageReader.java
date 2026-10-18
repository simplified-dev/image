package dev.simplified.image.codec.webp;

import dev.simplified.collection.Concurrent;
import dev.simplified.collection.ConcurrentList;
import dev.simplified.image.AnimatedImageData;
import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.PixelBuffer;
import dev.simplified.image.StaticImageData;
import dev.simplified.image.codec.ImageReader;
import dev.simplified.image.codec.common.DecodeBudget;
import dev.simplified.image.codec.common.LittleEndian;
import dev.simplified.image.codec.webp.vp8.VP8Decoder;
import dev.simplified.image.codec.webp.vp8.VP8DecoderSession;
import dev.simplified.image.codec.webp.vp8l.VP8LDecoder;
import dev.simplified.image.exception.ImageDecodeException;
import dev.simplified.image.frame.FrameBlend;
import dev.simplified.image.frame.FrameDisposal;
import dev.simplified.image.frame.ImageFrame;
import dev.simplified.image.frame.LoopCount;
import dev.simplified.image.option.ImageReadOptions;
import org.jetbrains.annotations.ApiStatus;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

/**
 * Reads WebP images (static and animated, lossless and lossy) using a pure Java
 * implementation of the VP8L and VP8 codecs.
 * <p>
 * An animated image's frames are read composed: each is the whole canvas as it stands once its
 * frame has been drawn, at the origin, blended {@link FrameBlend#SOURCE} and disposed
 * {@link FrameDisposal#RESTORE_TO_BACKGROUND}, so it replaces the frame before it whatever blend
 * and disposal the file declares for its frame.
 */
@ApiStatus.Internal
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

        int canvasWidth = 1 + LittleEndian.readUint24(payload, 4);
        int canvasHeight = 1 + LittleEndian.readUint24(payload, 7);

        if (canvasWidth > 16383 || canvasHeight > 16383)
            throw new ImageDecodeException("VP8X canvas dimensions exceed spec maximum: %dx%d", canvasWidth, canvasHeight);

        DecodeBudget budget = DecodeBudget.open("WebP").declare(canvasWidth, canvasHeight);

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
        // ANIM counts plays, the first included, and 0 plays forever. The container requires an
        // ANIM chunk in an animated file and libwebp rejects one without it; this reader accepts
        // such a file and reads it as playing once.
        LoopCount loopCount = LoopCount.Preset.NONE;

        if (animChunk != null && animChunk.payloadLength() >= 6) {
            byte[] animPayload = animChunk.payload();
            backgroundColor = LittleEndian.readUint32(animPayload, 0);
            int plays = LittleEndian.readUint16(animPayload, 4);
            loopCount = plays == 0 ? LoopCount.Preset.INFINITE : LoopCount.of(plays - 1);
        }

        ConcurrentList<ImageFrame> frames = Concurrent.newList();

        // Shared VP8 decoder session for animated lossy frames - enables P-frame decoding
        // where a frame references the prior frame's reconstruction.
        VP8DecoderSession vp8Session = new VP8DecoderSession();

        // Canvas state for partial-frame composition. Each ANMF's decoded sub-
        // buffer is composited onto this canvas per the frame's blend flag; the
        // canvas BEFORE compositing is disposed per the prior frame's disposal
        // flag (DISPOSE_BACKGROUND clears the prior bbox region to the ANIM
        // chunk's background colour, DO_NOT_DISPOSE leaves it). The resulting
        // full-canvas snapshot is what we hand back as each frame's pixels -
        // matches libwebp's WebPAnimDecoder behaviour and makes
        // AnimatedImageData callers see each frame as a complete image. The working canvas
        // and one snapshot per frame are claimed together before any of them is allocated.
        long frameCount = chunks.stream().filter(chunk -> chunk.type() == WebPChunk.Type.ANMF).count();
        budget.claim(canvasWidth, canvasHeight, frameCount + 1);
        int[] canvas = new int[canvasWidth * canvasHeight];
        java.util.Arrays.fill(canvas, backgroundColor);
        int priorFrameX = 0, priorFrameY = 0, priorFrameW = 0, priorFrameH = 0;
        FrameDisposal priorDisposal = FrameDisposal.DO_NOT_DISPOSE;

        for (WebPChunk chunk : chunks) {
            if (chunk.type() != WebPChunk.Type.ANMF) continue;

            byte[] anmf = chunk.payload();

            if (anmf.length < 16)
                throw new ImageDecodeException("ANMF chunk too short");

            int frameX = 2 * LittleEndian.readUint24(anmf, 0);
            int frameY = 2 * LittleEndian.readUint24(anmf, 3);
            int frameW = 1 + LittleEndian.readUint24(anmf, 6);
            int frameH = 1 + LittleEndian.readUint24(anmf, 9);
            int durationMs = LittleEndian.readUint24(anmf, 12);
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
                int subSize = LittleEndian.readUint32(anmf, subOffset + 4);
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

            PixelBuffer subPixels;
            if (vp8lSub != null) {
                subPixels = VP8LDecoder.decode(vp8lSub);
            } else if (vp8Sub != null) {
                subPixels = vp8Session.decode(vp8Sub);
                if (alphSub != null)
                    mergeAlphaPlane(subPixels, alphSub);
            } else {
                throw new ImageDecodeException("ANMF frame has no VP8 or VP8L sub-chunk");
            }

            if (subPixels.width() != frameW || subPixels.height() != frameH)
                throw new ImageDecodeException(
                    "ANMF frame size mismatch: declared %dx%d but decoded %dx%d",
                    frameW, frameH, subPixels.width(), subPixels.height()
                );

            // Apply PRIOR frame's disposal to the canvas (the spec puts
            // disposal at the START of the next frame's rendering, not after
            // the prior frame was emitted).
            if (priorDisposal == FrameDisposal.RESTORE_TO_BACKGROUND) {
                int x0 = Math.max(0, priorFrameX);
                int y0 = Math.max(0, priorFrameY);
                int x1 = Math.min(canvasWidth, priorFrameX + priorFrameW);
                int y1 = Math.min(canvasHeight, priorFrameY + priorFrameH);
                for (int y = y0; y < y1; y++)
                    for (int x = x0; x < x1; x++)
                        canvas[y * canvasWidth + x] = backgroundColor;
            }

            // Composite the partial-frame pixels onto the canvas per blend flag.
            int[] subPx = subPixels.data();
            int dstX0 = Math.max(0, frameX);
            int dstY0 = Math.max(0, frameY);
            int dstX1 = Math.min(canvasWidth, frameX + frameW);
            int dstY1 = Math.min(canvasHeight, frameY + frameH);
            for (int y = dstY0; y < dstY1; y++) {
                int srcRow = (y - frameY) * frameW;
                int dstRow = y * canvasWidth;
                for (int x = dstX0; x < dstX1; x++) {
                    int src = subPx[srcRow + (x - frameX)];
                    if (blend == FrameBlend.SOURCE) {
                        canvas[dstRow + x] = src;
                    } else {
                        // FrameBlend.OVER: alpha-composite src over dst.
                        int srcA = (src >>> 24) & 0xFF;
                        if (srcA == 0xFF) {
                            canvas[dstRow + x] = src;
                        } else if (srcA != 0) {
                            int dst = canvas[dstRow + x];
                            int dstA = (dst >>> 24) & 0xFF;
                            int outA = srcA + dstA * (255 - srcA) / 255;
                            if (outA == 0) {
                                canvas[dstRow + x] = 0;
                            } else {
                                int sr = (src >>> 16) & 0xFF, sg = (src >>> 8) & 0xFF, sb = src & 0xFF;
                                int dr = (dst >>> 16) & 0xFF, dg = (dst >>> 8) & 0xFF, db = dst & 0xFF;
                                int or = (sr * srcA + dr * dstA * (255 - srcA) / 255) / outA;
                                int og = (sg * srcA + dg * dstA * (255 - srcA) / 255) / outA;
                                int ob = (sb * srcA + db * dstA * (255 - srcA) / 255) / outA;
                                canvas[dstRow + x] = (outA << 24) | (or << 16) | (og << 8) | ob;
                            }
                        }
                        // srcA == 0 leaves canvas pixel unchanged.
                    }
                }
            }

            // Snapshot the full canvas as this frame's pixels so callers see
            // composed full-image data (matching libwebp's WebPAnimDecoder).
            // The snapshot has already taken this frame's blend and the prior
            // frame's disposal, so it carries neither: it is a whole canvas
            // that replaces the one before it.
            int[] snapshot = canvas.clone();
            PixelBuffer composed = PixelBuffer.of(snapshot, canvasWidth, canvasHeight);
            frames.add(ImageFrame.of(composed, durationMs));

            priorFrameX = frameX;
            priorFrameY = frameY;
            priorFrameW = frameW;
            priorFrameH = frameH;
            priorDisposal = disposal;
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
        int[] pixels = colorPixels.data();

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
            int[] alphPixels = alphDecoded.data();
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

    private static @Nullable WebPChunk findChunk(@NotNull ConcurrentList<WebPChunk> chunks, @NotNull WebPChunk.Type type) {
        return chunks.stream()
            .filter(chunk -> chunk.type() == type)
            .findFirst()
            .orElse(null);
    }

}
