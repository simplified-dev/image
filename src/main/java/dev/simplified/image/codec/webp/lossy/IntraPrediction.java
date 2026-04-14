package dev.simplified.image.codec.webp.lossy;

import org.jetbrains.annotations.NotNull;

/**
 * VP8 intra-frame spatial prediction for luma and chroma blocks.
 * <p>
 * Supports 5 luma macroblock modes (4 whole-block 16x16 modes plus B_PRED),
 * 10 luma 4x4 sub-block modes, and 4 chroma 8x8 modes.
 */
final class IntraPrediction {

    /** 16x16 luma prediction modes. */
    static final int DC_PRED = 0;
    static final int V_PRED = 1;
    static final int H_PRED = 2;
    static final int TM_PRED = 3;
    /** Per-sub-block prediction mode - each 4x4 block has its own mode. */
    static final int B_PRED = 4;

    /**
     * 4x4 sub-block prediction modes. Numerical values match libwebp's enum in
     * {@code src/dec/common_dec.h} so {@link VP8Tables#KF_BMODE_PROB} (which uses
     * libwebp's ordering) and {@link VP8Tables#BMODE_TREE} (whose leaves are
     * {@code -mode} per libwebp's positional layout) are consistent.
     */
    static final int B_DC_PRED = 0;
    static final int B_TM_PRED = 1;
    static final int B_VE_PRED = 2;
    static final int B_HE_PRED = 3;
    static final int B_RD_PRED = 4;
    static final int B_VR_PRED = 5;
    static final int B_LD_PRED = 6;
    static final int B_VL_PRED = 7;
    static final int B_HD_PRED = 8;
    static final int B_HU_PRED = 9;

    /** Number of 16x16 luma modes (excluding B_PRED). */
    static final int NUM_16x16_MODES = 4;
    /** Number of macroblock luma modes (including B_PRED). */
    static final int NUM_YMODES = 5;
    /** Number of 4x4 sub-block modes. */
    static final int NUM_4x4_MODES = 10;
    /** Number of 8x8 chroma modes. */
    static final int NUM_CHROMA_MODES = 4;

    private IntraPrediction() { }

    /**
     * Fills a 4x4 block with predicted values using a B_PRED sub-block mode.
     *
     * @param predicted output 16-element predicted block
     * @param above 8-element row - 4 above pixels followed by 4 above-right pixels (null if top row)
     * @param left 4-element column of pixels to the left (null if left column)
     * @param aboveLeft the pixel above-left of the block (128 if unavailable)
     * @param mode the sub-block prediction mode (0-9)
     */
    static void predict4x4(
        short @NotNull [] predicted,
        short[] above,
        short[] left,
        short aboveLeft,
        int mode
    ) {
        // Materialize the neighbour pixels with libwebp's default fills so the per-mode
        // formulas stay simple and bit-compatible: 127 for missing top, 129 for missing
        // left, 128 for missing above-left. Matches libwebp's dsp/dec.c predictors,
        // which rely on the surrounding buffer being pre-seeded with those values.
        short[] top = above;
        if (top == null) {
            top = new short[8];
            java.util.Arrays.fill(top, (short) 127);
        }
        short[] lt = left;
        if (lt == null) {
            lt = new short[4];
            java.util.Arrays.fill(lt, (short) 129);
        }
        final int P = aboveLeft;
        final int A = top[0], B = top[1], C = top[2], D = top[3];
        final int E = top[4], F = top[5], G = top[6], H = top[7];
        final int I = lt[0], J = lt[1], K = lt[2], L = lt[3];

        switch (mode) {
            case B_DC_PRED -> {
                int sum = 4 + A + B + C + D + I + J + K + L;
                short dc = (short) (sum >> 3);
                java.util.Arrays.fill(predicted, dc);
            }
            case B_TM_PRED -> {
                for (int y = 0; y < 4; y++)
                    for (int x = 0; x < 4; x++)
                        predicted[y * 4 + x] = (short) clamp(top[x] + lt[y] - P);
            }
            case B_VE_PRED -> {
                short c0 = avg3((short) P, (short) A, (short) B);
                short c1 = avg3((short) A, (short) B, (short) C);
                short c2 = avg3((short) B, (short) C, (short) D);
                short c3 = avg3((short) C, (short) D, (short) E);
                for (int y = 0; y < 4; y++) {
                    predicted[y * 4] = c0;
                    predicted[y * 4 + 1] = c1;
                    predicted[y * 4 + 2] = c2;
                    predicted[y * 4 + 3] = c3;
                }
            }
            case B_HE_PRED -> {
                short r0 = avg3((short) P, (short) I, (short) J);
                short r1 = avg3((short) I, (short) J, (short) K);
                short r2 = avg3((short) J, (short) K, (short) L);
                short r3 = avg3((short) K, (short) L, (short) L);
                for (int x = 0; x < 4; x++) {
                    predicted[x] = r0;
                    predicted[4 + x] = r1;
                    predicted[8 + x] = r2;
                    predicted[12 + x] = r3;
                }
            }
            case B_LD_PRED -> {
                predicted[0] = avg3((short) A, (short) B, (short) C);
                predicted[1] = predicted[4] = avg3((short) B, (short) C, (short) D);
                predicted[2] = predicted[5] = predicted[8] = avg3((short) C, (short) D, (short) E);
                predicted[3] = predicted[6] = predicted[9] = predicted[12] = avg3((short) D, (short) E, (short) F);
                predicted[7] = predicted[10] = predicted[13] = avg3((short) E, (short) F, (short) G);
                predicted[11] = predicted[14] = avg3((short) F, (short) G, (short) H);
                predicted[15] = avg3((short) G, (short) H, (short) H);
            }
            case B_RD_PRED -> {
                predicted[12] = avg3((short) J, (short) K, (short) L);
                predicted[8] = predicted[13] = avg3((short) I, (short) J, (short) K);
                predicted[4] = predicted[9] = predicted[14] = avg3((short) P, (short) I, (short) J);
                predicted[0] = predicted[5] = predicted[10] = predicted[15] = avg3((short) I, (short) P, (short) A);
                predicted[1] = predicted[6] = predicted[11] = avg3((short) P, (short) A, (short) B);
                predicted[2] = predicted[7] = avg3((short) A, (short) B, (short) C);
                predicted[3] = avg3((short) B, (short) C, (short) D);
            }
            case B_VR_PRED -> {
                predicted[12] = avg3((short) K, (short) J, (short) I);
                predicted[8] = avg3((short) J, (short) I, (short) P);
                predicted[4] = predicted[13] = avg3((short) I, (short) P, (short) A);
                predicted[0] = predicted[9] = avg2((short) P, (short) A);
                predicted[5] = predicted[14] = avg3((short) P, (short) A, (short) B);
                predicted[1] = predicted[10] = avg2((short) A, (short) B);
                predicted[6] = predicted[15] = avg3((short) A, (short) B, (short) C);
                predicted[2] = predicted[11] = avg2((short) B, (short) C);
                predicted[7] = avg3((short) B, (short) C, (short) D);
                predicted[3] = avg2((short) C, (short) D);
            }
            case B_VL_PRED -> {
                predicted[0] = avg2((short) A, (short) B);
                predicted[4] = avg3((short) A, (short) B, (short) C);
                predicted[1] = predicted[8] = avg2((short) B, (short) C);
                predicted[5] = predicted[12] = avg3((short) B, (short) C, (short) D);
                predicted[2] = predicted[9] = avg2((short) C, (short) D);
                predicted[6] = predicted[13] = avg3((short) C, (short) D, (short) E);
                predicted[3] = predicted[10] = avg2((short) D, (short) E);
                predicted[7] = predicted[14] = avg3((short) D, (short) E, (short) F);
                predicted[11] = avg2((short) E, (short) F);
                predicted[15] = avg3((short) E, (short) F, (short) G);
            }
            case B_HD_PRED -> {
                predicted[12] = avg2((short) L, (short) K);
                predicted[13] = avg3((short) L, (short) K, (short) J);
                predicted[8] = predicted[14] = avg2((short) K, (short) J);
                predicted[9] = predicted[15] = avg3((short) K, (short) J, (short) I);
                predicted[4] = predicted[10] = avg2((short) J, (short) I);
                predicted[5] = predicted[11] = avg3((short) J, (short) I, (short) P);
                predicted[0] = predicted[6] = avg2((short) I, (short) P);
                predicted[1] = predicted[7] = avg3((short) I, (short) P, (short) A);
                predicted[2] = avg3((short) P, (short) A, (short) B);
                predicted[3] = avg3((short) A, (short) B, (short) C);
            }
            case B_HU_PRED -> {
                predicted[0] = avg2((short) I, (short) J);
                predicted[1] = avg3((short) I, (short) J, (short) K);
                predicted[2] = predicted[4] = avg2((short) J, (short) K);
                predicted[3] = predicted[5] = avg3((short) J, (short) K, (short) L);
                predicted[6] = predicted[8] = avg2((short) K, (short) L);
                predicted[7] = predicted[9] = avg3((short) K, (short) L, (short) L);
                predicted[10] = predicted[11] = predicted[12] = predicted[13] =
                    predicted[14] = predicted[15] = (short) L;
            }
            default -> java.util.Arrays.fill(predicted, (short) 128);
        }
    }

    /**
     * Fills an 8x8 block with predicted values using the given mode.
     *
     * @param predicted output 64-element predicted block
     * @param above 8-element row of pixels above the block (null if top row)
     * @param left 8-element column of pixels to the left (null if left column)
     * @param aboveLeft the pixel above-left of the block (128 if unavailable)
     * @param mode the prediction mode (0-3)
     */
    static void predict8x8(
        short @NotNull [] predicted,
        short[] above,
        short[] left,
        short aboveLeft,
        int mode
    ) {
        switch (mode) {
            case DC_PRED -> {
                int sum = 0;
                int count = 0;
                if (above != null) { for (short v : above) sum += v; count += 8; }
                if (left != null) { for (short v : left) sum += v; count += 8; }
                short dc = count > 0 ? (short) ((sum + count / 2) / count) : (short) 128;
                java.util.Arrays.fill(predicted, dc);
            }
            case V_PRED -> {
                if (above == null) { java.util.Arrays.fill(predicted, (short) 127); return; }
                for (int y = 0; y < 8; y++)
                    System.arraycopy(above, 0, predicted, y * 8, 8);
            }
            case H_PRED -> {
                if (left == null) { java.util.Arrays.fill(predicted, (short) 129); return; }
                for (int y = 0; y < 8; y++)
                    for (int x = 0; x < 8; x++)
                        predicted[y * 8 + x] = left[y];
            }
            case TM_PRED -> {
                for (int y = 0; y < 8; y++)
                    for (int x = 0; x < 8; x++) {
                        int a = above != null ? above[x] : 127;
                        int l = left != null ? left[y] : 129;
                        predicted[y * 8 + x] = (short) clamp(a + l - aboveLeft);
                    }
            }
            default -> java.util.Arrays.fill(predicted, (short) 128);
        }
    }

    /**
     * Fills a 16x16 block with predicted values using the given mode.
     *
     * @param predicted output 256-element predicted block
     * @param above 16-element row of pixels above the block (null if top row)
     * @param left 16-element column of pixels to the left (null if left column)
     * @param aboveLeft the pixel above-left of the block (128 if unavailable)
     * @param mode the prediction mode (0-3)
     */
    static void predict16x16(
        short @NotNull [] predicted,
        short[] above,
        short[] left,
        short aboveLeft,
        int mode
    ) {
        switch (mode) {
            case DC_PRED -> {
                int sum = 0;
                int count = 0;
                if (above != null) { for (short v : above) sum += v; count += 16; }
                if (left != null) { for (short v : left) sum += v; count += 16; }
                short dc = count > 0 ? (short) ((sum + count / 2) / count) : (short) 128;
                java.util.Arrays.fill(predicted, dc);
            }
            case V_PRED -> {
                if (above == null) { java.util.Arrays.fill(predicted, (short) 127); return; }
                for (int y = 0; y < 16; y++)
                    System.arraycopy(above, 0, predicted, y * 16, 16);
            }
            case H_PRED -> {
                if (left == null) { java.util.Arrays.fill(predicted, (short) 129); return; }
                for (int y = 0; y < 16; y++)
                    for (int x = 0; x < 16; x++)
                        predicted[y * 16 + x] = left[y];
            }
            case TM_PRED -> {
                for (int y = 0; y < 16; y++)
                    for (int x = 0; x < 16; x++) {
                        int a = above != null ? above[x] : 127;
                        int l = left != null ? left[y] : 129;
                        predicted[y * 16 + x] = (short) clamp(a + l - aboveLeft);
                    }
            }
            default -> java.util.Arrays.fill(predicted, (short) 128);
        }
    }

    /**
     * Computes the sum of squared differences between two blocks.
     *
     * @param original the original pixel block
     * @param predicted the predicted pixel block
     * @return the sum of squared differences
     */
    static long computeSSD(short @NotNull [] original, short @NotNull [] predicted) {
        long ssd = 0;
        for (int i = 0; i < original.length; i++) {
            int diff = original[i] - predicted[i];
            ssd += (long) diff * diff;
        }
        return ssd;
    }

    private static short avg2(int a, int b) {
        return (short) ((a + b + 1) >> 1);
    }

    private static short avg3(int a, int b, int c) {
        return (short) ((a + 2 * b + c + 2) >> 2);
    }

    private static int clamp(int value) {
        return Math.clamp(value, 0, 255);
    }

}
