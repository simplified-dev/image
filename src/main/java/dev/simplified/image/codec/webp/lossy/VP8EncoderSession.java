package dev.simplified.image.codec.webp.lossy;

import dev.simplified.image.pixel.PixelBuffer;
import org.jetbrains.annotations.NotNull;

/**
 * Stateful VP8 encoder carrying reference-frame buffers across calls.
 * <p>
 * The session holds the reconstructed Y/U/V planes from the most recently encoded
 * frame, which will be consumed as the {@code LAST} reference by Phase 1+ P-frame
 * encoding. Phase 0 (current) emits a keyframe for every call regardless of the
 * {@code forceKeyframe} flag; the reference is captured but not yet used.
 * <p>
 * For stateless single-frame encoding prefer {@link VP8Encoder#encode(PixelBuffer, float)},
 * which internally allocates a throwaway session.
 *
 * @see VP8Encoder
 * @see VP8DecoderSession
 */
public final class VP8EncoderSession {

    /** Last reconstructed luma plane, or {@code null} before the first encode. */
    short[] refY;
    /** Last reconstructed U chroma plane, or {@code null} before the first encode. */
    short[] refU;
    /** Last reconstructed V chroma plane, or {@code null} before the first encode. */
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

    /** Constructs a new {@code VP8EncoderSession} with no cached reference. */
    public VP8EncoderSession() { }

    /**
     * Encodes {@code pixels} as a VP8 frame. Phase 0 always emits a keyframe;
     * Phase 1+ may emit a P-frame when a reference is available.
     *
     * @param pixels source pixel buffer
     * @param quality encoding quality in {@code [0.0, 1.0]}
     * @return encoded VP8 payload bytes
     */
    public byte @NotNull [] encode(@NotNull PixelBuffer pixels, float quality) {
        return encode(pixels, quality, false);
    }

    /**
     * Encodes {@code pixels} as a VP8 frame, optionally forcing a keyframe. When a
     * reference is cached and its dimensions match {@code pixels}, and
     * {@code forceKeyframe} is {@code false}, emits a P-frame; otherwise emits a keyframe.
     *
     * @param pixels source pixel buffer
     * @param quality encoding quality in {@code [0.0, 1.0]}
     * @param forceKeyframe forces a keyframe even if a reference is available
     * @return encoded VP8 payload bytes
     */
    public byte @NotNull [] encode(@NotNull PixelBuffer pixels, float quality, boolean forceKeyframe) {
        boolean needsKeyframe = forceKeyframe
            || !hasReference()
            || pixels.width() != refWidth
            || pixels.height() != refHeight;
        if (needsKeyframe)
            return VP8Encoder.encodeKeyframe(pixels, quality, this);
        return VP8Encoder.encodePFrame(pixels, quality, this);
    }

    /** {@code true} when a reference frame is available for Phase 1+ P-frame encoding. */
    public boolean hasReference() {
        return refY != null;
    }

    /** Clears the cached reference so the next encode starts from a clean state. */
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
     * {@link VP8Encoder} after a successful encode and after the encoder has applied the
     * loop filter, so the captured reference matches the post-filter state the decoder
     * will produce. Reallocates the backing arrays only when their length changes across
     * calls.
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
