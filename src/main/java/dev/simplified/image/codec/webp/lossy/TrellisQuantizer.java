package dev.simplified.image.codec.webp.lossy;

import org.jetbrains.annotations.NotNull;

/**
 * Trellis-optimized quantization for a single 4x4 coefficient block.
 * <p>
 * Port of libwebp's {@code TrellisQuantizeBlock} ({@code src/enc/quant_enc.c}).
 * For each coefficient position in zig-zag order, evaluates two candidate quantized
 * levels (the plain-quant result and {@code +1}) and picks the path through the trellis
 * whose rate-distortion score {@code rate * lambda + 256 * distortion} is minimum.
 * <p>
 * Rate is computed using {@link VP8Costs#REMAPPED_COSTS}; distortion is the weighted
 * squared error between the dequantized level and the original coefficient.
 */
final class TrellisQuantizer {

    /**
     * Coefficient weight used in trellis distortion (libwebp {@code USE_TDISTO=1} default).
     * Weighs low-frequency coefficients more heavily than high-frequency ones - matches
     * the human eye's greater sensitivity to low-frequency quantization error.
     * <p>
     * Indexed in zig-zag order {@code n} (to match libwebp's indexing {@code kWeightTrellis[j]}
     * where {@code j = kZigzag[n]} - in libwebp {@code j} is the raster position, but our
     * trellis accesses this by the raster index, so the table values are in raster order).
     */
    private static final int[] WEIGHT_TRELLIS = {
        30, 27, 19, 11, 27, 24, 17, 10, 19, 17, 12, 8, 11, 10, 8, 6
    };

    /** Search range: we test {@code level0} and {@code level0 + 1} at each coefficient. */
    private static final int MIN_DELTA = 0;
    private static final int MAX_DELTA = 1;
    private static final int NUM_NODES = MIN_DELTA + 1 + MAX_DELTA;

    /** {@link VP8Costs#MAX_VARIABLE_LEVEL}. */
    private static final int MAX_VARIABLE_LEVEL = VP8Costs.MAX_VARIABLE_LEVEL;
    /** {@link VP8Costs#MAX_LEVEL}. */
    private static final int MAX_LEVEL = VP8Costs.MAX_LEVEL;
    /** {@link VP8Costs#RD_DISTO_MULT}. */
    private static final int RD_DISTO_MULT = VP8Costs.RD_DISTO_MULT;

    /** Sentinel for an unreachable trellis node. */
    private static final long MAX_COST = 0x7fffffffffffffL;

    private TrellisQuantizer() { }

    /**
     * Emulates libwebp's {@code QUANTDIV(n, iQ, B) = (n * iQ + B) >> QFIX}. All inputs are
     * treated as unsigned 32-bit.
     */
    private static int quantDiv(int n, int iq, int bias) {
        long p = (long) n * (iq & 0xFFFFFFFFL) + (bias & 0xFFFFFFFFL);
        return (int) (p >>> QuantMatrix.QFIX);
    }

    /**
     * Trellis-quantizes a single 4x4 block. Writes zig-zag-ordered levels into {@code out}
     * and dequantized coefficients (raster order) back into {@code in}, matching libwebp's
     * convention so the caller can feed {@code in} directly to the inverse DCT.
     *
     * @param in raster-order DCT coefficients; on return, contains dequantized values
     * @param out zig-zag-order quantized levels (output); {@code out[0]} is preserved for
     *            {@code TYPE_I16_AC} (DC is carried by Y2)
     * @param ctx0 initial surrounding-nonzero context ({@code top_nz + left_nz}, 0..2)
     * @param coeffType {@code TYPE_I16_AC | TYPE_I16_DC | TYPE_CHROMA_A | TYPE_I4_AC}
     * @param mtx per-coefficient quantization matrix
     * @param lambda R-D lambda (higher = favor rate over distortion)
     * @return {@code 1} if at least one non-zero coefficient remains after trellis, else {@code 0}
     */
    static int quantize(
        short @NotNull [] in, short @NotNull [] out, int ctx0, int coeffType,
        @NotNull QuantMatrix mtx, int lambda
    ) {
        int first = (coeffType == VP8Tables.TYPE_I16_AC) ? 1 : 0;

        // Trellis storage:
        //   nodes[n][m]   - best path to "coded level = level0(n) + m" at position n
        //   ss[phase][m]  - partial R-D score + cost-table pointer for the "previous" column
        int[] nodesLevel = new int[16 * NUM_NODES];
        byte[] nodesPrev = new byte[16 * NUM_NODES];
        byte[] nodesSign = new byte[16 * NUM_NODES];
        long[] scoreScore = new long[2 * NUM_NODES];
        int[][] scoreCosts = new int[2 * NUM_NODES][];
        int curPhase = 0;

        int[] bestPath = { -1, -1, -1 }; // {last, nodeIdx, prevIdx}
        int[][][] remapped = VP8Costs.REMAPPED_COSTS[coeffType];

        // Position of the "last interesting coefficient" - libwebp widens by +1 to give the
        // trellis room to turn zero coefficients into non-zero ones when it reduces total cost.
        int last;
        {
            int thresh = mtx.q[1] * mtx.q[1] / 4;
            last = first - 1;
            for (int n = 15; n >= first; n--) {
                int j = VP8Tables.ZIGZAG[n];
                int err = in[j] * in[j];
                if (err > thresh) { last = n; break; }
            }
            if (last < 15) last++;
        }

        int lastProba = VP8Tables.COEFFS_PROBA_0[coeffType][VP8Tables.COEF_BANDS[first]][ctx0][0];

        // Skip-block (all-zero) score is the max achievable without any non-zero levels.
        long bestScore = rdScore(lambda, VP8Costs.bitCost(0, lastProba), 0);

        // Seed source node.
        for (int m = -MIN_DELTA; m <= MAX_DELTA; m++) {
            int rate = (ctx0 == 0) ? VP8Costs.bitCost(1, lastProba) : 0;
            int idx = curPhase * NUM_NODES + (m + MIN_DELTA);
            scoreScore[idx] = rdScore(lambda, rate, 0);
            scoreCosts[idx] = remapped[first][ctx0];
        }

        for (int n = first; n <= last; n++) {
            int j = VP8Tables.ZIGZAG[n];
            int Q = mtx.q[j];
            int iQ = mtx.iq[j];
            int B = QuantMatrix.bias(0x00); // neutral bias for trellis
            boolean neg = in[j] < 0;
            int coeff0 = (neg ? -in[j] : in[j]) + mtx.sharpen[j];
            int level0 = Math.min(quantDiv(coeff0, iQ, B), MAX_LEVEL);
            int threshLevel = Math.min(quantDiv(coeff0, iQ, QuantMatrix.bias(0x80)), MAX_LEVEL);

            // Swap current and previous score columns.
            int prevPhase = curPhase;
            curPhase ^= 1;

            for (int m = -MIN_DELTA; m <= MAX_DELTA; m++) {
                int level = level0 + m;
                int ctx = (level > 2) ? 2 : level;
                int band = VP8Tables.COEF_BANDS[n + 1];

                int idx = curPhase * NUM_NODES + (m + MIN_DELTA);
                scoreCosts[idx] = (n + 1 < 16) ? remapped[n + 1][ctx] : null;

                if (level < 0 || level > threshLevel) {
                    scoreScore[idx] = MAX_COST;
                    continue;
                }

                // Delta distortion vs the "all zero from here" baseline, weighted.
                int newError = coeff0 - level * Q;
                long deltaError = (long) WEIGHT_TRELLIS[j] * ((long) newError * newError - (long) coeff0 * coeff0);
                long baseScore = rdScore(lambda, 0, deltaError);

                // Walk all predecessor columns, pick the best one.
                int bestPrev = -MIN_DELTA;
                long bestCur;
                {
                    int prevIdx0 = prevPhase * NUM_NODES + 0;
                    int cost = VP8Costs.levelCost(scoreCosts[prevIdx0], level);
                    bestCur = scoreScore[prevIdx0] + rdScore(lambda, cost, 0);
                }
                for (int p = -MIN_DELTA + 1; p <= MAX_DELTA; p++) {
                    int prevIdx = prevPhase * NUM_NODES + (p + MIN_DELTA);
                    int cost = VP8Costs.levelCost(scoreCosts[prevIdx], level);
                    long score = scoreScore[prevIdx] + rdScore(lambda, cost, 0);
                    if (score < bestCur) {
                        bestCur = score;
                        bestPrev = p;
                    }
                }
                bestCur += baseScore;

                int nodeIdx = n * NUM_NODES + (m + MIN_DELTA);
                nodesSign[nodeIdx] = (byte) (neg ? 1 : 0);
                nodesLevel[nodeIdx] = level;
                nodesPrev[nodeIdx] = (byte) bestPrev;
                scoreScore[idx] = bestCur;

                // Record best terminal node - "stop here, EOB or end-of-block-at-15".
                if (level != 0 && bestCur < bestScore) {
                    long lastPosCost = (n < 15)
                        ? VP8Costs.bitCost(0, VP8Tables.COEFFS_PROBA_0[coeffType][band][ctx][0])
                        : 0;
                    long lastPosScore = rdScore(lambda, (int) lastPosCost, 0);
                    long score = bestCur + lastPosScore;
                    if (score < bestScore) {
                        bestScore = score;
                        bestPath[0] = n;
                        bestPath[1] = m;
                        bestPath[2] = bestPrev;
                    }
                }
            }
        }

        // Reset output. Preserve in[0]/out[0] for I16 AC since DC is coded separately in Y2.
        if (coeffType == VP8Tables.TYPE_I16_AC) {
            for (int i = 1; i < 16; i++) { in[i] = 0; out[i] = 0; }
        } else {
            for (int i = 0; i < 16; i++) { in[i] = 0; out[i] = 0; }
        }

        if (bestPath[0] == -1)
            return 0; // skip the whole block

        int nz = 0;
        int bestNode = bestPath[1];
        int nLast = bestPath[0];
        // Patch terminal node's prev field with the best terminal predecessor.
        nodesPrev[nLast * NUM_NODES + (bestNode + MIN_DELTA)] = (byte) bestPath[2];
        for (int n = nLast; n >= first; n--) {
            int nodeIdx = n * NUM_NODES + (bestNode + MIN_DELTA);
            int level = nodesLevel[nodeIdx];
            int sign = nodesSign[nodeIdx];
            int j = VP8Tables.ZIGZAG[n];
            out[n] = (short) (sign != 0 ? -level : level);
            nz |= level;
            in[j] = (short) (out[n] * mtx.q[j]);
            bestNode = nodesPrev[nodeIdx];
        }
        return nz != 0 ? 1 : 0;
    }

    private static long rdScore(int lambda, int rate, long distortion) {
        return (long) rate * lambda + (long) RD_DISTO_MULT * distortion;
    }

}
