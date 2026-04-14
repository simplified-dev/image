package dev.simplified.image.codec.webp.lossy;

import org.jetbrains.annotations.NotNull;

/**
 * Per-coefficient quantization parameters for trellis quantization and coefficient
 * dequantization. Mirrors libwebp's {@code VP8Matrix} ({@code src/enc/vp8i_enc.h})
 * as expanded by {@code ExpandMatrix} ({@code src/enc/quant_enc.c}).
 * <p>
 * In VP8 the base quantizer is flat across all 16 coefficient positions except for
 * the DC coefficient (index 0), so {@link #q} is effectively {@code [dc, ac, ac, ...,
 * ac]}. {@link #iq} is the Q17 fixed-point reciprocal and {@link #bias} is the rounding
 * bias baked into the quantization divide. {@link #sharpen} raises the luma AC
 * coefficients per libwebp's {@code kFreqSharpening} curve to preserve edge detail
 * at mid-bitrate.
 */
final class QuantMatrix {

    /** Q17 fixed-point shift used by {@link #iq} and {@link #bias}. */
    static final int QFIX = 17;

    /** Per-coefficient quantization bias shift. {@code BIAS(b) = b << 9}. */
    static int bias(int b) {
        return b << (QFIX - 8);
    }

    /**
     * libwebp's sharpening curve (only applied to luma AC coefficients, position 0 is
     * the DC and always gets 0). Descaled by {@code >> 11} against the quant step.
     * See {@code kFreqSharpening} in {@code src/enc/quant_enc.c}.
     */
    private static final int[] FREQ_SHARPENING = {
        0, 30, 60, 90, 30, 60, 90, 90, 60, 90, 90, 90, 90, 90, 90, 90
    };
    private static final int SHARPEN_BITS = 11;

    /** Quantizer-bias matrix {@code [type][is_ac_coeff]}, type 0=luma Y1, 1=luma Y2, 2=chroma. */
    private static final int[][] BIAS_MATRIX = {
        { 96, 110 },   // luma Y1 (I16 AC + I4 AC): DC bias 96, AC bias 110
        { 96, 108 },   // luma Y2 (WHT): DC bias 96, AC bias 108
        { 110, 115 }   // chroma: DC bias 110, AC bias 115
    };

    /** Per-coefficient quantizer step. {@code q[0]} is DC; {@code q[1..15]} are AC (flat). */
    final int[] q = new int[16];
    /** Per-coefficient Q17 fixed-point reciprocal: {@code iq[i] = (1<<17) / q[i]}. */
    final int[] iq = new int[16];
    /** Per-coefficient rounding bias in Q17: {@code bias[i] = kBias[type][ac] << 9}. */
    final int[] bias = new int[16];
    /** Luma AC sharpening (raises coefficients by a small amount); zero for Y2 and chroma. */
    final int[] sharpen = new int[16];

    private QuantMatrix(int dcQ, int acQ, int type) {
        q[0] = dcQ;
        iq[0] = (1 << QFIX) / dcQ;
        bias[0] = bias(BIAS_MATRIX[type][0]);
        for (int i = 1; i < 16; i++) {
            q[i] = acQ;
            iq[i] = (1 << QFIX) / acQ;
            bias[i] = bias(BIAS_MATRIX[type][1]);
        }
        if (type == 0) {
            // Sharpening is only applied to luma Y1 AC coefficients.
            for (int i = 0; i < 16; i++)
                sharpen[i] = (FREQ_SHARPENING[i] * q[i]) >> SHARPEN_BITS;
        }
    }

    /** Builds a luma Y1 matrix (used for I16 AC, I4 AC, and B_PRED sub-block quant). */
    static @NotNull QuantMatrix luma(int dcQ, int acQ) {
        return new QuantMatrix(dcQ, acQ, 0);
    }

    /** Builds a luma Y2 (WHT) matrix for the 16 DC coefficients of an I16 macroblock. */
    static @NotNull QuantMatrix lumaY2(int dcQ, int acQ) {
        return new QuantMatrix(dcQ, acQ, 1);
    }

    /** Builds a chroma matrix. */
    static @NotNull QuantMatrix chroma(int dcQ, int acQ) {
        return new QuantMatrix(dcQ, acQ, 2);
    }

}
