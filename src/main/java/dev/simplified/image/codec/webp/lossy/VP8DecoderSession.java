package dev.simplified.image.codec.webp.lossy;

import dev.simplified.image.pixel.PixelBuffer;
import org.jetbrains.annotations.NotNull;

/**
 * Stateful VP8 decoder carrying reference-frame buffers across calls.
 * <p>
 * The session holds the reconstructed Y/U/V planes from the most recently decoded
 * frame (post loop filter), which will be consumed as the {@code LAST} reference
 * by Phase 1+ P-frame decoding. Phase 0 (current) still requires every frame to
 * be a keyframe; the reference is captured but not yet consumed.
 * <p>
 * For stateless single-frame decoding prefer {@link VP8Decoder#decode(byte[])},
 * which internally allocates a throwaway session.
 *
 * @see VP8Decoder
 * @see VP8EncoderSession
 */
public final class VP8DecoderSession {

    /** Last reconstructed luma plane, or {@code null} before the first decode. */
    short[] refY;
    /** Last reconstructed U chroma plane, or {@code null} before the first decode. */
    short[] refU;
    /** Last reconstructed V chroma plane, or {@code null} before the first decode. */
    short[] refV;
    /** Luma plane stride in samples (MB-grid-aligned). */
    int refLumaStride;
    /** Chroma plane stride in samples (MB-grid-aligned). */
    int refChromaStride;
    /** Macroblock columns in the reference planes. */
    int refMbCols;
    /** Macroblock rows in the reference planes. */
    int refMbRows;
    /** Reference frame width in pixels. */
    int refWidth;
    /** Reference frame height in pixels. */
    int refHeight;

    /** Constructs a new {@code VP8DecoderSession} with no cached reference. */
    public VP8DecoderSession() { }

    /**
     * Decodes a VP8 bitstream (keyframe or P-frame) into pixel data. P-frames require
     * a cached reference from a prior decode within this session; the dimensions are
     * carried over from the most recent keyframe.
     *
     * @param data the raw VP8 payload
     * @return the decoded pixel buffer
     */
    public @NotNull PixelBuffer decode(byte @NotNull [] data) {
        return VP8Decoder.decodeFrame(data, this);
    }

    /** {@code true} when a reference frame is available for Phase 1+ P-frame decoding. */
    public boolean hasReference() {
        return refY != null;
    }

    /** Clears the cached reference so the next decode starts from a clean state. */
    public void reset() {
        refY = null;
        refU = null;
        refV = null;
        refLumaStride = 0;
        refChromaStride = 0;
        refMbCols = 0;
        refMbRows = 0;
        refWidth = 0;
        refHeight = 0;
    }

    /**
     * Copies the reconstructed planes into this session's reference buffers. Called by
     * {@link VP8Decoder} after a successful decode (post loop filter). Reallocates the
     * backing arrays only when their length changes across calls.
     */
    void captureReference(
        short @NotNull [] reconY, short @NotNull [] reconU, short @NotNull [] reconV,
        int lumaStride, int chromaStride, int mbCols, int mbRows, int width, int height
    ) {
        int lumaLen = lumaStride * mbRows * 16;
        int chromaLen = chromaStride * mbRows * 8;
        if (refY == null || refY.length != lumaLen) refY = new short[lumaLen];
        if (refU == null || refU.length != chromaLen) refU = new short[chromaLen];
        if (refV == null || refV.length != chromaLen) refV = new short[chromaLen];
        System.arraycopy(reconY, 0, refY, 0, lumaLen);
        System.arraycopy(reconU, 0, refU, 0, chromaLen);
        System.arraycopy(reconV, 0, refV, 0, chromaLen);
        refLumaStride = lumaStride;
        refChromaStride = chromaStride;
        refMbCols = mbCols;
        refMbRows = mbRows;
        refWidth = width;
        refHeight = height;
    }

}
