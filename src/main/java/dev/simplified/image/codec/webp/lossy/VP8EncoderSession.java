package dev.simplified.image.codec.webp.lossy;

import dev.simplified.image.pixel.PixelBuffer;
import org.jetbrains.annotations.NotNull;

/**
 * Stateful VP8 encoder carrying the three VP8 reference-frame slots - {@code LAST},
 * {@code GOLDEN}, and {@code ALTREF} - across calls.
 * <p>
 * Each slot holds a post-loop-filter reconstruction, snapshotted after the encoder
 * applies the same loop filter the decoder will. A keyframe populates all three
 * slots identically. A P-frame's emitted {@code refresh_last} / {@code refresh_golden}
 * / {@code refresh_alt} flags and {@code copy_buffer_to_golden} / {@code copy_buffer_to_alt}
 * selectors decide which slots update at the end of encode. Our default encoder emits
 * {@code refresh_last = 1} and {@code refresh_golden = refresh_alt = 0} with no
 * buffer copies, so only LAST updates per P-frame; extension to long-range prediction
 * via GOLDEN / ALTREF is left to future work.
 * <p>
 * For stateless single-frame encoding prefer {@link VP8Encoder#encode(PixelBuffer, float)},
 * which internally allocates a throwaway session.
 *
 * @see VP8Encoder
 * @see VP8DecoderSession
 * @see <a href="https://datatracker.ietf.org/doc/html/rfc6386#section-9.7">RFC 6386 section 9.7</a>
 */
public final class VP8EncoderSession {

    // ── LAST reference (legacy field names) ──
    /** Last reconstructed luma plane, or {@code null} before the first encode. */
    short[] refY;
    short[] refU;
    short[] refV;

    // ── GOLDEN reference ──
    short[] goldenY;
    short[] goldenU;
    short[] goldenV;

    // ── ALTREF reference ──
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

    /** Constructs a new {@code VP8EncoderSession} with no cached references. */
    public VP8EncoderSession() { }

    /**
     * Encodes {@code pixels} as a VP8 frame. Emits a keyframe when no LAST reference is
     * cached; otherwise emits a P-frame against the cached LAST reference.
     *
     * @param pixels source pixel buffer
     * @param quality encoding quality in {@code [0.0, 1.0]}
     * @return encoded VP8 payload bytes
     */
    public byte @NotNull [] encode(@NotNull PixelBuffer pixels, float quality) {
        return encode(pixels, quality, false);
    }

    /**
     * Encodes {@code pixels}, optionally forcing a keyframe even when a LAST reference
     * is available. When forcing a keyframe, all three reference slots are refreshed.
     *
     * @param pixels source pixel buffer
     * @param quality encoding quality in {@code [0.0, 1.0]}
     * @param forceKeyframe forces a keyframe even if a LAST reference is available
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

    /** Clears all cached references so the next encode starts from a clean state. */
    public void reset() {
        refY = refU = refV = null;
        goldenY = goldenU = goldenV = null;
        altrefY = altrefU = altrefV = null;
        refLumaStride = refChromaStride = 0;
        refMbCols = refMbRows = 0;
        refWidth = refHeight = 0;
    }

    /**
     * Snapshots the reconstructed planes into this session's {@code LAST} slot. Called by
     * {@link VP8Encoder} after the encoder has applied the loop filter, so the captured
     * reference matches the post-filter state the decoder will produce.
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
