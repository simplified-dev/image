package dev.simplified.image.codec.webp.lossy;

import org.jetbrains.annotations.NotNull;

/**
 * VP8 sub-pel motion-compensation prediction using the 6-tap separable filter.
 * <p>
 * Line-for-line port of libvpx's {@code vp8_sixtap_predict*_c} in
 * {@code vp8/common/filter.c}. Two-pass operation: horizontal 6-tap then vertical
 * 6-tap, with the intermediate buffer clamped to {@code [0, 255]} between passes
 * (matching the reference integer-math implementation bit-for-bit).
 * <p>
 * The filter kernels in {@link #SUBPEL_FILTERS} are indexed by 1/8-pel sub-pel
 * position {@code [0..7]}. Luma motion vectors are emitted in 1/4-pel wire units,
 * so the internal luma MV (wire &lt;&lt; 1) only visits sub-pel positions
 * {0, 2, 4, 6} (full, 1/4, 1/2, 3/4 pel). Chroma MVs, derived from luma via
 * {@code round_away_zero(mv_luma / 2)}, may land on any of the 8 sub-pel positions.
 * <p>
 * Reference samples outside the provided plane bounds are clamped to the edge
 * (pixel replication), matching libvpx's handling of edge MBs.
 */
final class SubpelPrediction {

    /** Sum of the 6-tap coefficients (scales rounding + output normalisation). */
    private static final int FILTER_WEIGHT = 128;

    /** Right-shift amount used to normalise 6-tap products back into sample range. */
    private static final int FILTER_SHIFT = 7;

    /**
     * 6-tap sub-pel filter kernels (libvpx {@code vp8_sub_pel_filters}). Rows are 1/8-pel
     * positions {@code 0..7}; columns are taps {@code [-2, -1, 0, 1, 2, 3]} relative to
     * the integer-pel anchor.
     */
    static final int[][] SUBPEL_FILTERS = {
        {  0,   0, 128,   0,   0,  0 },   // 0/8: full pel (identity)
        {  0,  -6, 123,  12,  -1,  0 },   // 1/8
        {  2, -11, 108,  36,  -8,  1 },   // 2/8: 1/4 pel
        {  0,  -9,  93,  50,  -6,  0 },   // 3/8
        {  3, -16,  77,  77, -16,  3 },   // 4/8: 1/2 pel
        {  0,  -6,  50,  93,  -9,  0 },   // 5/8
        {  1,  -8,  36, 108, -11,  2 },   // 6/8: 3/4 pel
        {  0,  -1,  12, 123,  -6,  0 }    // 7/8
    };

    private SubpelPrediction() { }

    /**
     * Computes a motion-compensated prediction block using 2D 6-tap sub-pel filtering.
     * <p>
     * The prediction region is {@code blockW x blockH} samples anchored at integer-pel
     * position ({@code refX}, {@code refY}) in {@code ref}, offset by sub-pel positions
     * ({@code xoffset}, {@code yoffset}) in 1/8-pel units. The filter reads samples
     * from {@code [refX - 2, refX + blockW + 2]} horizontally and
     * {@code [refY - 2, refY + blockH + 2]} vertically; out-of-plane reads are clamped
     * to the plane edge.
     *
     * @param ref source reference plane (row-major, samples in {@code [0, 255]})
     * @param refStride {@code ref} row stride in samples
     * @param refH {@code ref} height in samples
     * @param refX integer-pel x of the top-left of the prediction region
     * @param refY integer-pel y of the top-left of the prediction region
     * @param xoffset sub-pel x in {@code [0, 7]} (1/8-pel units)
     * @param yoffset sub-pel y in {@code [0, 7]} (1/8-pel units)
     * @param dst output buffer (MB-local, size {@code blockW * blockH})
     * @param dstStride {@code dst} row stride (usually {@code blockW})
     * @param blockW prediction block width in samples
     * @param blockH prediction block height in samples
     */
    static void predict6tap(
        short @NotNull [] ref, int refStride, int refH,
        int refX, int refY, int xoffset, int yoffset,
        short @NotNull [] dst, int dstStride, int blockW, int blockH
    ) {
        int[] hFilter = SUBPEL_FILTERS[xoffset];
        int[] vFilter = SUBPEL_FILTERS[yoffset];

        // First pass: horizontal 6-tap. Produces (blockH + 5) rows of blockW samples,
        // covering rows [refY - 2, refY + blockH + 2]. Each output sample is clamped
        // to [0, 255] to match libvpx's integer-math reference.
        int fdataRows = blockH + 5;
        int[] fdata = new int[fdataRows * blockW];
        for (int fy = 0; fy < fdataRows; fy++) {
            int srcY = refY - 2 + fy;
            int clampY = Math.clamp(srcY, 0, refH - 1);
            int rowOff = clampY * refStride;
            for (int fx = 0; fx < blockW; fx++) {
                int srcXBase = refX + fx;
                int t = 0;
                for (int k = 0; k < 6; k++) {
                    int sx = Math.clamp(srcXBase + k - 2, 0, refStride - 1);
                    t += ref[rowOff + sx] * hFilter[k];
                }
                t = (t + (FILTER_WEIGHT >> 1)) >> FILTER_SHIFT;
                fdata[fy * blockW + fx] = Math.clamp(t, 0, 255);
            }
        }

        // Second pass: vertical 6-tap over fdata. Reads fdata rows [fy, fy+5] (relative
        // to output row fy) - fdata row 2 aligns with prediction row 0.
        for (int oy = 0; oy < blockH; oy++) {
            int dstOff = oy * dstStride;
            for (int ox = 0; ox < blockW; ox++) {
                int t = 0;
                for (int k = 0; k < 6; k++)
                    t += fdata[(oy + k) * blockW + ox] * vFilter[k];
                t = (t + (FILTER_WEIGHT >> 1)) >> FILTER_SHIFT;
                dst[dstOff + ox] = (short) Math.clamp(t, 0, 255);
            }
        }
    }

}
