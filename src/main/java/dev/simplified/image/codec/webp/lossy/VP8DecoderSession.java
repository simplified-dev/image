package dev.simplified.image.codec.webp.lossy;

import dev.simplified.image.pixel.PixelBuffer;
import org.jetbrains.annotations.NotNull;

/**
 * Stateful VP8 decoder carrying the three VP8 reference-frame slots - {@code LAST},
 * {@code GOLDEN}, and {@code ALTREF} - across calls.
 * <p>
 * Each slot holds a post-loop-filter reconstruction of a prior frame, sized to the
 * shared session dimensions (all three slots use the same width / height / stride
 * since VP8 requires frame-size consistency across a sequence). A keyframe implicitly
 * refreshes all three slots with the current reconstruction. A P-frame's header
 * {@code refresh_last} / {@code refresh_golden} / {@code refresh_alt} flags and the
 * {@code copy_buffer_to_golden} / {@code copy_buffer_to_alt} selectors determine
 * which slots change and which propagate from another slot.
 * <p>
 * For stateless single-frame decoding prefer {@link VP8Decoder#decode(byte[])},
 * which internally allocates a throwaway session.
 *
 * @see VP8Decoder
 * @see VP8EncoderSession
 * @see <a href="https://datatracker.ietf.org/doc/html/rfc6386#section-9.7">RFC 6386 section 9.7</a>
 */
public final class VP8DecoderSession {

    // ── LAST reference (legacy field names, retained for call-site stability) ──
    /** Last reconstructed luma plane, or {@code null} before the first decode. */
    short[] refY;
    /** Last reconstructed U chroma plane, or {@code null} before the first decode. */
    short[] refU;
    /** Last reconstructed V chroma plane, or {@code null} before the first decode. */
    short[] refV;

    // ── GOLDEN reference ──
    /** Golden-slot luma plane, or {@code null} when {@link #hasReferenceGolden()} is false. */
    short[] goldenY;
    short[] goldenU;
    short[] goldenV;

    // ── ALTREF reference ──
    /** Altref-slot luma plane, or {@code null} when {@link #hasReferenceAltref()} is false. */
    short[] altrefY;
    short[] altrefU;
    short[] altrefV;

    /** Luma plane stride in samples (MB-grid-aligned); shared across all 3 slots. */
    int refLumaStride;
    /** Chroma plane stride in samples (MB-grid-aligned); shared across all 3 slots. */
    int refChromaStride;
    /** Macroblock columns in the reference planes. */
    int refMbCols;
    /** Macroblock rows in the reference planes. */
    int refMbRows;
    /** Reference frame width in pixels. */
    int refWidth;
    /** Reference frame height in pixels. */
    int refHeight;

    /**
     * Sign-bias flag for the golden reference (RFC 6386 section 9.7). When set and a
     * neighbour MB referenced {@code GOLDEN}, {@link NearMvs} flips that neighbour's
     * MV sign relative to the current MB's reference before reuse.
     */
    boolean signBiasGolden;
    /** Sign-bias flag for the altref reference (symmetric to {@link #signBiasGolden}). */
    boolean signBiasAltref;

    /** Constructs a new {@code VP8DecoderSession} with no cached references. */
    public VP8DecoderSession() { }

    /**
     * Decodes a VP8 bitstream (keyframe or P-frame) into pixel data. P-frames require
     * at least the {@code LAST} reference to be cached from a prior decode within this
     * session; the dimensions are carried over from the most recent keyframe.
     *
     * @param data the raw VP8 payload
     * @return the decoded pixel buffer
     */
    public @NotNull PixelBuffer decode(byte @NotNull [] data) {
        return VP8Decoder.decodeFrame(data, this);
    }

    /** {@code true} when the {@code LAST} reference is available. */
    public boolean hasReference() {
        return refY != null;
    }

    /** {@code true} when the {@code GOLDEN} reference is available. */
    public boolean hasReferenceGolden() {
        return goldenY != null;
    }

    /** {@code true} when the {@code ALTREF} reference is available. */
    public boolean hasReferenceAltref() {
        return altrefY != null;
    }

    /** Clears all cached references so the next decode starts from a clean state. */
    public void reset() {
        refY = refU = refV = null;
        goldenY = goldenU = goldenV = null;
        altrefY = altrefU = altrefV = null;
        refLumaStride = refChromaStride = 0;
        refMbCols = refMbRows = 0;
        refWidth = refHeight = 0;
        signBiasGolden = signBiasAltref = false;
    }

    /**
     * Returns the luma plane of the reference frame selected by {@code refFrame}, one of
     * {@link LoopFilter#REF_LAST}, {@link LoopFilter#REF_GOLDEN}, or
     * {@link LoopFilter#REF_ALTREF}.
     */
    short @NotNull [] lumaRef(int refFrame) {
        return switch (refFrame) {
            case LoopFilter.REF_LAST -> refY;
            case LoopFilter.REF_GOLDEN -> goldenY;
            case LoopFilter.REF_ALTREF -> altrefY;
            default -> throw new IllegalStateException("not a valid inter reference: " + refFrame);
        };
    }

    /** Chroma U plane for the selected reference frame. */
    short @NotNull [] uRef(int refFrame) {
        return switch (refFrame) {
            case LoopFilter.REF_LAST -> refU;
            case LoopFilter.REF_GOLDEN -> goldenU;
            case LoopFilter.REF_ALTREF -> altrefU;
            default -> throw new IllegalStateException("not a valid inter reference: " + refFrame);
        };
    }

    /** Chroma V plane for the selected reference frame. */
    short @NotNull [] vRef(int refFrame) {
        return switch (refFrame) {
            case LoopFilter.REF_LAST -> refV;
            case LoopFilter.REF_GOLDEN -> goldenV;
            case LoopFilter.REF_ALTREF -> altrefV;
            default -> throw new IllegalStateException("not a valid inter reference: " + refFrame);
        };
    }

    /**
     * Snapshots the reconstructed planes into this session's {@code LAST} slot. Reallocates
     * only when the buffer size changes.
     */
    void captureReferenceLast(
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
        this.refLumaStride = lumaStride;
        this.refChromaStride = chromaStride;
        this.refMbCols = mbCols;
        this.refMbRows = mbRows;
        this.refWidth = width;
        this.refHeight = height;
    }

    /** Snapshots the reconstructed planes into this session's {@code GOLDEN} slot. */
    void captureReferenceGolden(
        short @NotNull [] reconY, short @NotNull [] reconU, short @NotNull [] reconV,
        int lumaStride, int chromaStride, int mbCols, int mbRows, int width, int height
    ) {
        int lumaLen = lumaStride * mbRows * 16;
        int chromaLen = chromaStride * mbRows * 8;
        if (goldenY == null || goldenY.length != lumaLen) goldenY = new short[lumaLen];
        if (goldenU == null || goldenU.length != chromaLen) goldenU = new short[chromaLen];
        if (goldenV == null || goldenV.length != chromaLen) goldenV = new short[chromaLen];
        System.arraycopy(reconY, 0, goldenY, 0, lumaLen);
        System.arraycopy(reconU, 0, goldenU, 0, chromaLen);
        System.arraycopy(reconV, 0, goldenV, 0, chromaLen);
        this.refLumaStride = lumaStride;
        this.refChromaStride = chromaStride;
        this.refMbCols = mbCols;
        this.refMbRows = mbRows;
        this.refWidth = width;
        this.refHeight = height;
    }

    /** Snapshots the reconstructed planes into this session's {@code ALTREF} slot. */
    void captureReferenceAltref(
        short @NotNull [] reconY, short @NotNull [] reconU, short @NotNull [] reconV,
        int lumaStride, int chromaStride, int mbCols, int mbRows, int width, int height
    ) {
        int lumaLen = lumaStride * mbRows * 16;
        int chromaLen = chromaStride * mbRows * 8;
        if (altrefY == null || altrefY.length != lumaLen) altrefY = new short[lumaLen];
        if (altrefU == null || altrefU.length != chromaLen) altrefU = new short[chromaLen];
        if (altrefV == null || altrefV.length != chromaLen) altrefV = new short[chromaLen];
        System.arraycopy(reconY, 0, altrefY, 0, lumaLen);
        System.arraycopy(reconU, 0, altrefU, 0, chromaLen);
        System.arraycopy(reconV, 0, altrefV, 0, chromaLen);
        this.refLumaStride = lumaStride;
        this.refChromaStride = chromaStride;
        this.refMbCols = mbCols;
        this.refMbRows = mbRows;
        this.refWidth = width;
        this.refHeight = height;
    }

    /** Implements {@code copy_buffer_to_golden = 1} (copy LAST -> GOLDEN). */
    void copyLastToGolden() {
        goldenY = ensureLen(goldenY, refY.length);
        goldenU = ensureLen(goldenU, refU.length);
        goldenV = ensureLen(goldenV, refV.length);
        System.arraycopy(refY, 0, goldenY, 0, refY.length);
        System.arraycopy(refU, 0, goldenU, 0, refU.length);
        System.arraycopy(refV, 0, goldenV, 0, refV.length);
    }

    /** Implements {@code copy_buffer_to_golden = 2} (copy ALTREF -> GOLDEN). */
    void copyAltrefToGolden() {
        goldenY = ensureLen(goldenY, altrefY.length);
        goldenU = ensureLen(goldenU, altrefU.length);
        goldenV = ensureLen(goldenV, altrefV.length);
        System.arraycopy(altrefY, 0, goldenY, 0, altrefY.length);
        System.arraycopy(altrefU, 0, goldenU, 0, altrefU.length);
        System.arraycopy(altrefV, 0, goldenV, 0, altrefV.length);
    }

    /** Implements {@code copy_buffer_to_alt = 1} (copy LAST -> ALTREF). */
    void copyLastToAltref() {
        altrefY = ensureLen(altrefY, refY.length);
        altrefU = ensureLen(altrefU, refU.length);
        altrefV = ensureLen(altrefV, refV.length);
        System.arraycopy(refY, 0, altrefY, 0, refY.length);
        System.arraycopy(refU, 0, altrefU, 0, refU.length);
        System.arraycopy(refV, 0, altrefV, 0, refV.length);
    }

    /** Implements {@code copy_buffer_to_alt = 2} (copy GOLDEN -> ALTREF). */
    void copyGoldenToAltref() {
        altrefY = ensureLen(altrefY, goldenY.length);
        altrefU = ensureLen(altrefU, goldenU.length);
        altrefV = ensureLen(altrefV, goldenV.length);
        System.arraycopy(goldenY, 0, altrefY, 0, goldenY.length);
        System.arraycopy(goldenU, 0, altrefU, 0, goldenU.length);
        System.arraycopy(goldenV, 0, altrefV, 0, goldenV.length);
    }

    private static short[] ensureLen(short[] arr, int len) {
        return (arr == null || arr.length != len) ? new short[len] : arr;
    }

}
