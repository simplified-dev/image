package dev.simplified.image.codec.webp.lossy;

import org.jetbrains.annotations.NotNull;

/**
 * Neighbour-MV derivation shared by {@link VP8Encoder} and {@link VP8Decoder} for
 * P-frame NEAREST / NEAR / NEW / ZERO mode context.
 * <p>
 * Line-for-line port of libvpx's {@code vp8_find_near_mvs} in
 * {@code vp8/common/findnearmv.c}. Examines the above, left, and above-left
 * neighbours (weights 2, 2, 1), classifies each as intra, zero-MV, or a distinct
 * non-zero MV, and populates nearest/near/best MV slots plus the 4-slot vote
 * counter used to index into {@link VP8Tables#MODE_CONTEXTS}.
 * <p>
 * Both encoder and decoder must produce bit-identical output for each MB position
 * given the same neighbour state; mismatches break NEAREST / NEAR decoding because
 * the decoder has no wire MV to fall back on. All MV values are in libvpx's
 * 1/8-pel internal representation (wire quarter-pel shifted left by one).
 */
final class NearMvs {

    /** Index into {@link Result#cnt} for the "intra or zero-MV" neighbour count. */
    static final int CNT_INTRA = 0;

    /** Index for the "matches nearest_mv" neighbour count. */
    static final int CNT_NEAREST = 1;

    /** Index for the "matches near_mv" neighbour count. */
    static final int CNT_NEAR = 2;

    /** Index for the SPLITMV neighbour count (always 0 in our encoder - we never emit SPLITMV). */
    static final int CNT_SPLITMV = 3;

    /** Output bundle - nearest / near / best MVs plus the 4-slot vote counter. */
    static final class Result {
        /** Nearest-neighbour MV (1/8-pel internal units). */
        int nearestRow, nearestCol;
        /** Near-neighbour MV. */
        int nearRow, nearCol;
        /** Best-guess MV for NEW_MV initial search / ZEROMV fallback. */
        int bestRow, bestCol;
        /** Vote counts indexed by {@link #CNT_INTRA}, {@link #CNT_NEAREST}, etc. */
        final int[] cnt = new int[4];
    }

    private NearMvs() { }

    /**
     * Returns the RFC 6386 section 18.3 sign-bias flag for a reference frame. INTRA and
     * LAST are always 0; GOLDEN and ALTREF take the per-frame flags emitted in the
     * key-frame header. Mirrors libvpx's {@code ref_frame_sign_bias[]} table.
     *
     * @param refFrame one of {@link LoopFilter#REF_INTRA} / {@link LoopFilter#REF_LAST}
     *                / {@link LoopFilter#REF_GOLDEN} / {@link LoopFilter#REF_ALTREF}
     * @param signBiasGolden per-frame sign_bias_golden flag
     * @param signBiasAltref per-frame sign_bias_altref flag
     * @return 0 or 1
     */
    private static int biasOf(int refFrame, boolean signBiasGolden, boolean signBiasAltref) {
        if (refFrame == LoopFilter.REF_GOLDEN) return signBiasGolden ? 1 : 0;
        if (refFrame == LoopFilter.REF_ALTREF) return signBiasAltref ? 1 : 0;
        return 0;                          // INTRA and LAST always 0
    }

    /**
     * Derives nearest / near / best MVs and neighbour vote counts for the MB at
     * ({@code mbX}, {@code mbY}). Read-only over {@code mbIsInter}, {@code mbMvRow},
     * {@code mbMvCol}, {@code mbRefFrame} which must already carry state for MBs earlier
     * in raster order.
     * <p>
     * Implements libvpx's {@code vp8_mv_bias} cross-reference sign handling: each
     * neighbour MV whose reference-frame sign bias differs from the current MB's sign bias
     * has its row and column negated before the distinct-MV comparison and vote bump.
     * Required for interop with libvpx-produced streams that set a non-zero
     * {@code sign_bias_golden} / {@code sign_bias_altref}.
     *
     * @param mbIsInter per-MB "has a reference frame" flag
     * @param mbMvRow per-MB MV row (1/8-pel internal) - unused when {@code !mbIsInter[i]}
     * @param mbMvCol per-MB MV col (1/8-pel internal) - unused when {@code !mbIsInter[i]}
     * @param mbRefFrame per-MB reference-frame index (REF_INTRA / REF_LAST / REF_GOLDEN /
     *                   REF_ALTREF); used to resolve each inter neighbour's sign bias
     * @param mbCols macroblock grid width
     * @param mbX current MB column
     * @param mbY current MB row
     * @param currentRefFrame reference frame the current MB is being evaluated against
     * @param signBiasGolden per-frame {@code sign_bias_golden} flag
     * @param signBiasAltref per-frame {@code sign_bias_altref} flag
     * @param out populated with the derived values (caller-allocated to avoid GC churn)
     */
    static void find(
        boolean @NotNull [] mbIsInter, int @NotNull [] mbMvRow, int @NotNull [] mbMvCol,
        int @NotNull [] mbRefFrame, int mbCols, int mbX, int mbY, int currentRefFrame,
        boolean signBiasGolden, boolean signBiasAltref, @NotNull Result out
    ) {
        // libvpx stashes best/nearest/near/split MVs in a 4-slot array and walks a "current"
        // pointer forward as new distinct non-zero neighbour MVs arrive. cntx advances in
        // lockstep, so the first slot that a weight is added to always matches the MV slot
        // the pointer currently refers to.
        int[] mvRows = { 0, 0, 0, 0 };
        int[] mvCols = { 0, 0, 0, 0 };
        int[] cnt = out.cnt;
        cnt[0] = cnt[1] = cnt[2] = cnt[3] = 0;
        int mvIdx = 0;
        int curBias = biasOf(currentRefFrame, signBiasGolden, signBiasAltref);

        // Above neighbour (weight 2).
        if (mbY > 0) {
            int idx = (mbY - 1) * mbCols + mbX;
            if (mbIsInter[idx]) {
                int r = mbMvRow[idx], c = mbMvCol[idx];
                if (biasOf(mbRefFrame[idx], signBiasGolden, signBiasAltref) != curBias) {
                    r = -r; c = -c;
                }
                if ((r | c) != 0) {
                    mvIdx++;
                    mvRows[mvIdx] = r;
                    mvCols[mvIdx] = c;
                }
                cnt[mvIdx] += 2;
            }
        }

        // Left neighbour (weight 2). Libvpx: zero-MV non-intra neighbours bump CNT_INTRA
        // explicitly here (unlike above, where the initial cntx position coincides with
        // CNT_INTRA, so the implicit increment lands in the same slot).
        if (mbX > 0) {
            int idx = mbY * mbCols + mbX - 1;
            if (mbIsInter[idx]) {
                int r = mbMvRow[idx], c = mbMvCol[idx];
                if (biasOf(mbRefFrame[idx], signBiasGolden, signBiasAltref) != curBias) {
                    r = -r; c = -c;
                }
                if ((r | c) != 0) {
                    if (r != mvRows[mvIdx] || c != mvCols[mvIdx]) {
                        mvIdx++;
                        mvRows[mvIdx] = r;
                        mvCols[mvIdx] = c;
                    }
                    cnt[mvIdx] += 2;
                } else {
                    cnt[CNT_INTRA] += 2;
                }
            }
        }

        // Above-left neighbour (weight 1).
        if (mbX > 0 && mbY > 0) {
            int idx = (mbY - 1) * mbCols + mbX - 1;
            if (mbIsInter[idx]) {
                int r = mbMvRow[idx], c = mbMvCol[idx];
                if (biasOf(mbRefFrame[idx], signBiasGolden, signBiasAltref) != curBias) {
                    r = -r; c = -c;
                }
                if ((r | c) != 0) {
                    if (r != mvRows[mvIdx] || c != mvCols[mvIdx]) {
                        mvIdx++;
                        mvRows[mvIdx] = r;
                        mvCols[mvIdx] = c;
                    }
                    cnt[mvIdx] += 1;
                } else {
                    cnt[CNT_INTRA] += 1;
                }
            }
        }

        // When three distinct non-zero neighbour MVs were found, libvpx checks whether
        // the current (third-distinct) MV happens to equal nearest_mv and if so, bumps
        // the NEAREST count.
        if (cnt[CNT_SPLITMV] != 0 && mvRows[mvIdx] == mvRows[CNT_NEAREST]
                                  && mvCols[mvIdx] == mvCols[CNT_NEAREST]) {
            cnt[CNT_NEAREST] += 1;
        }

        // Our encoder never emits SPLITMV, so the "neighbour coded SPLITMV" count
        // collapses to 0. Libvpx's formula: cnt[SPLITMV] = (above==SPLIT)*2 + (left==SPLIT)*2 + (aboveleft==SPLIT).
        cnt[CNT_SPLITMV] = 0;

        // Swap near and nearest if near accumulated more votes.
        if (cnt[CNT_NEAR] > cnt[CNT_NEAREST]) {
            int tmp;
            tmp = cnt[CNT_NEAREST]; cnt[CNT_NEAREST] = cnt[CNT_NEAR]; cnt[CNT_NEAR] = tmp;
            tmp = mvRows[CNT_NEAREST]; mvRows[CNT_NEAREST] = mvRows[CNT_NEAR]; mvRows[CNT_NEAR] = tmp;
            tmp = mvCols[CNT_NEAREST]; mvCols[CNT_NEAREST] = mvCols[CNT_NEAR]; mvCols[CNT_NEAR] = tmp;
        }

        // If nearest votes >= intra/zero votes, promote nearest into the best slot.
        if (cnt[CNT_NEAREST] >= cnt[CNT_INTRA]) {
            mvRows[CNT_INTRA] = mvRows[CNT_NEAREST];
            mvCols[CNT_INTRA] = mvCols[CNT_NEAREST];
        }

        out.bestRow = mvRows[CNT_INTRA];
        out.bestCol = mvCols[CNT_INTRA];
        out.nearestRow = mvRows[CNT_NEAREST];
        out.nearestCol = mvCols[CNT_NEAREST];
        out.nearRow = mvRows[CNT_NEAR];
        out.nearCol = mvCols[CNT_NEAR];
    }

    /**
     * Computes the MV-reference tree probabilities from a 4-slot vote counter, mirroring
     * libvpx's {@code vp8_mv_ref_probs}. Shared by encoder + decoder so both sides agree
     * on the probability vector used by the MV-ref tree walk.
     */
    static void refProbs(int @NotNull [] cnt, int @NotNull [] outProbs) {
        int maxIdx = VP8Tables.MODE_CONTEXTS.length - 1;
        for (int i = 0; i < 4; i++)
            outProbs[i] = VP8Tables.MODE_CONTEXTS[Math.min(cnt[i], maxIdx)][i];
    }

}
