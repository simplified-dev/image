package dev.simplified.image.codec.webp;

import org.jetbrains.annotations.NotNull;

/**
 * Forward and inverse 4x4 transforms for VP8 lossy coding.
 * <p>
 * These are ports of libwebp's {@code FTransform_C} / {@code TransformOne_C}
 * (forward and inverse DCT in {@code src/dsp/enc.c} and {@code src/dsp/dec.c})
 * plus {@code FTransformWHT_C} / {@code TransformWHT_C} (forward and inverse
 * Walsh-Hadamard). They must match libwebp bit-for-bit: libwebp's decoder
 * applies its own inverse transforms when reading our bitstream, so any
 * rounding drift between our forward transform and theirs becomes pixel
 * error in the decoded image.
 */
final class DCT {

    /** WEBP_TRANSFORM_AC3_C1 - first cosine constant (~20091/65536). */
    private static final int C1 = 20091;
    /** WEBP_TRANSFORM_AC3_C2 - second cosine constant (~35468/65536). */
    private static final int C2 = 35468;

    private DCT() { }

    /**
     * Forward 4x4 DCT matching libwebp's {@code FTransform_C}.
     *
     * @param input 16-element row-major residual block (values typically in {@code [-255, 255]})
     * @param output 16-element row-major DCT coefficients
     */
    static void forwardDCT(short @NotNull [] input, short @NotNull [] output) {
        int[] tmp = new int[16];

        // Horizontal pass (per row i).
        for (int i = 0; i < 4; i++) {
            int row = i * 4;
            int d0 = input[row + 0];
            int d1 = input[row + 1];
            int d2 = input[row + 2];
            int d3 = input[row + 3];
            int a0 = d0 + d3;
            int a1 = d1 + d2;
            int a2 = d1 - d2;
            int a3 = d0 - d3;
            tmp[row + 0] = (a0 + a1) * 8;
            tmp[row + 1] = (a2 * 2217 + a3 * 5352 + 1812) >> 9;
            tmp[row + 2] = (a0 - a1) * 8;
            tmp[row + 3] = (a3 * 2217 - a2 * 5352 + 937) >> 9;
        }

        // Vertical pass (per column i).
        for (int i = 0; i < 4; i++) {
            int a0 = tmp[i + 0] + tmp[i + 12];
            int a1 = tmp[i + 4] + tmp[i + 8];
            int a2 = tmp[i + 4] - tmp[i + 8];
            int a3 = tmp[i + 0] - tmp[i + 12];
            output[i +  0] = (short) ((a0 + a1 + 7) >> 4);
            output[i +  4] = (short) (((a2 * 2217 + a3 * 5352 + 12000) >> 16) + (a3 != 0 ? 1 : 0));
            output[i +  8] = (short) ((a0 - a1 + 7) >> 4);
            output[i + 12] = (short) ((a3 * 2217 - a2 * 5352 + 51000) >> 16);
        }
    }

    /**
     * Inverse 4x4 DCT matching libwebp's {@code TransformOne_C}. Output is the
     * reconstructed residual scaled by 1 (caller adds prediction and clamps).
     *
     * @param input 16-element row-major dequantized coefficients
     * @param output 16-element row-major reconstructed residuals
     */
    static void inverseDCT(short @NotNull [] input, short @NotNull [] output) {
        int[] tmp = new int[16];

        // Vertical pass (per column): accesses input[col], input[col+4], input[col+8], input[col+12].
        for (int col = 0; col < 4; col++) {
            int a = input[col] + input[col + 8];
            int b = input[col] - input[col + 8];
            int c = mul2(input[col + 4]) - mul1(input[col + 12]);
            int d = mul1(input[col + 4]) + mul2(input[col + 12]);
            // Write transposed into tmp: column 'col' of input -> row 'col' of tmp.
            tmp[col * 4 + 0] = a + d;
            tmp[col * 4 + 1] = b + c;
            tmp[col * 4 + 2] = b - c;
            tmp[col * 4 + 3] = a - d;
        }

        // Horizontal pass (per row in original space, which is 'col' in tmp space).
        for (int row = 0; row < 4; row++) {
            int dc = tmp[row + 0] + 4;                  // libwebp adds 4 here before any shift
            int a = dc + tmp[row + 8];
            int b = dc - tmp[row + 8];
            int c = mul2(tmp[row + 4]) - mul1(tmp[row + 12]);
            int d = mul1(tmp[row + 4]) + mul2(tmp[row + 12]);
            output[row * 4 + 0] = (short) ((a + d) >> 3);
            output[row * 4 + 1] = (short) ((b + c) >> 3);
            output[row * 4 + 2] = (short) ((b - c) >> 3);
            output[row * 4 + 3] = (short) ((a - d) >> 3);
        }
    }

    /** {@code WEBP_TRANSFORM_AC3_MUL1}: approximates {@code v * C1/65536 + v}. */
    private static int mul1(int v) {
        return ((v * C1) >> 16) + v;
    }

    /** {@code WEBP_TRANSFORM_AC3_MUL2}: approximates {@code v * C2/65536}. */
    private static int mul2(int v) {
        return (v * C2) >> 16;
    }

    /**
     * Forward 4x4 Walsh-Hadamard matching libwebp's {@code FTransformWHT_C}.
     * Used to compress the 16 DC coefficients of a 16x16 luma macroblock.
     *
     * @param input 16 DC values from each 4x4 sub-block, row-major
     * @param output 16 WHT coefficients, row-major
     */
    static void forwardWHT(short @NotNull [] input, short @NotNull [] output) {
        int[] tmp = new int[16];

        // Horizontal pass: reads rows of input.
        for (int i = 0; i < 4; i++) {
            int row = i * 4;
            int a0 = input[row + 0] + input[row + 2];
            int a1 = input[row + 1] + input[row + 3];
            int a2 = input[row + 1] - input[row + 3];
            int a3 = input[row + 0] - input[row + 2];
            tmp[row + 0] = a0 + a1;
            tmp[row + 1] = a3 + a2;
            tmp[row + 2] = a3 - a2;
            tmp[row + 3] = a0 - a1;
        }

        // Vertical pass: reads columns of tmp.
        for (int i = 0; i < 4; i++) {
            int a0 = tmp[i + 0] + tmp[i + 8];
            int a1 = tmp[i + 4] + tmp[i + 12];
            int a2 = tmp[i + 4] - tmp[i + 12];
            int a3 = tmp[i + 0] - tmp[i + 8];
            output[i +  0] = (short) ((a0 + a1) >> 1);
            output[i +  4] = (short) ((a3 + a2) >> 1);
            output[i +  8] = (short) ((a3 - a2) >> 1);
            output[i + 12] = (short) ((a0 - a1) >> 1);
        }
    }

    /**
     * Inverse 4x4 Walsh-Hadamard matching libwebp's {@code TransformWHT_C}.
     *
     * @param input 16 dequantized WHT coefficients, row-major
     * @param output 16 reconstructed DC values, row-major
     */
    static void inverseWHT(short @NotNull [] input, short @NotNull [] output) {
        int[] tmp = new int[16];

        // Vertical pass: reads columns of input.
        for (int i = 0; i < 4; i++) {
            int a0 = input[i + 0] + input[i + 12];
            int a1 = input[i + 4] + input[i + 8];
            int a2 = input[i + 4] - input[i + 8];
            int a3 = input[i + 0] - input[i + 12];
            tmp[i + 0] = a0 + a1;
            tmp[i + 4] = a3 + a2;
            tmp[i + 8] = a0 - a1;
            tmp[i + 12] = a3 - a2;
        }

        // Horizontal pass: reads rows of tmp, with +3 rounder on the DC element.
        for (int i = 0; i < 4; i++) {
            int row = i * 4;
            int dc = tmp[row + 0] + 3;
            int a0 = dc + tmp[row + 3];
            int a1 = tmp[row + 1] + tmp[row + 2];
            int a2 = tmp[row + 1] - tmp[row + 2];
            int a3 = dc - tmp[row + 3];
            output[row + 0] = (short) ((a0 + a1) >> 3);
            output[row + 1] = (short) ((a3 + a2) >> 3);
            output[row + 2] = (short) ((a0 - a1) >> 3);
            output[row + 3] = (short) ((a3 - a2) >> 3);
        }
    }

}
