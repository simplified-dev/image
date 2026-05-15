package dev.simplified.image.codec.webp.lossy;

import org.jetbrains.annotations.NotNull;

/**
 * VP8 quantization parameter management and coefficient quantize/dequantize.
 * <p>
 * Separate Y (luma) and UV (chroma) quantization matrices with distinct
 * DC and AC quantization steps. Quality maps to QP value.
 */
final class Quantizer {

    /**
     * VP8 DC quantization table (indexed by QP 0-127).
     */
    private static final int[] DC_TABLE = buildDcTable();
    /**
     * VP8 AC quantization table (indexed by QP 0-127).
     */
    private static final int[] AC_TABLE = buildAcTable();

    private final int yDc;
    private final int yAc;
    private final int uvDc;
    private final int uvAc;
    private final int qp;

    /**
     * Creates a quantizer for the given quality parameter.
     *
     * @param quality the encoding quality (0.0 - 1.0, higher is better)
     */
    Quantizer(float quality) {
        // Map quality 0.0-1.0 to QP 127-0 (inverted: lower QP = higher quality)
        this.qp = Math.clamp((int) ((1.0f - quality) * 127), 0, 127);
        this.yDc = DC_TABLE[qp];
        this.yAc = AC_TABLE[qp];
        this.uvDc = DC_TABLE[qp];
        this.uvAc = AC_TABLE[qp];
    }

    /**
     * Quantizes luma DCT coefficients in-place.
     *
     * @param coefficients the 16-element coefficient array
     */
    void quantizeY(short @NotNull [] coefficients) {
        coefficients[0] = (short) (coefficients[0] / yDc);
        for (int i = 1; i < 16; i++)
            coefficients[i] = (short) (coefficients[i] / yAc);
    }

    /**
     * Dequantizes luma DCT coefficients in-place.
     *
     * @param coefficients the 16-element coefficient array
     */
    void dequantizeY(short @NotNull [] coefficients) {
        coefficients[0] = (short) (coefficients[0] * yDc);
        for (int i = 1; i < 16; i++)
            coefficients[i] = (short) (coefficients[i] * yAc);
    }

    /**
     * Quantizes chroma DCT coefficients in-place.
     *
     * @param coefficients the 16-element coefficient array
     */
    void quantizeUV(short @NotNull [] coefficients) {
        coefficients[0] = (short) (coefficients[0] / uvDc);
        for (int i = 1; i < 16; i++)
            coefficients[i] = (short) (coefficients[i] / uvAc);
    }

    /**
     * Dequantizes chroma DCT coefficients in-place.
     *
     * @param coefficients the 16-element coefficient array
     */
    void dequantizeUV(short @NotNull [] coefficients) {
        coefficients[0] = (short) (coefficients[0] * uvDc);
        for (int i = 1; i < 16; i++)
            coefficients[i] = (short) (coefficients[i] * uvAc);
    }

    /**
     * The quantization parameter (0-127).
     */
    int getQP() {
        return qp;
    }

    private static int[] buildDcTable() {
        int[] table = new int[128];
        for (int i = 0; i < 128; i++)
            table[i] = Math.max(1, (i < 8) ? i + 2 : (i < 25) ? (i * 2 - 4) : (i * 3 - 26));
        return table;
    }

    private static int[] buildAcTable() {
        int[] table = new int[128];
        for (int i = 0; i < 128; i++)
            table[i] = Math.max(1, (i < 4) ? i + 2 : (i < 24) ? (i + i / 2) : (i * 2 - 16));
        return table;
    }

}
