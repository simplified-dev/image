package dev.simplified.image.codec.webp.lossy;

import org.jetbrains.annotations.NotNull;

/**
 * Rate-distortion mode decision engine for the VP8 encoder.
 * <p>
 * Evaluates intra prediction modes and selects the one minimizing
 * {@code distortion + lambda * rate}, where distortion is SSD and
 * rate is estimated from quantized coefficient count.
 */
final class RateDistortion {

    private final float lambda;

    /**
     * Creates a rate-distortion optimizer for the given quantization parameter.
     *
     * @param qp the quantization parameter (0-127)
     */
    RateDistortion(int qp) {
        // Lambda increases with QP: at low QP (high quality) we care more
        // about distortion; at high QP we care more about rate
        this.lambda = 0.5f + qp * 0.1f;
    }

    /**
     * Selects the best 16x16 luma prediction mode for a macroblock.
     *
     * @param mb the macroblock with original Y samples
     * @param aboveRow the 16 luma samples from the row above (null if top edge)
     * @param leftCol the 16 luma samples from the column to the left (null if left edge)
     * @param aboveLeft the above-left corner sample
     * @param quantizer the quantizer for rate estimation
     * @return the best mode index (0-3)
     */
    int selectBest16x16Mode(
        @NotNull Macroblock mb,
        short[] aboveRow,
        short[] leftCol,
        short aboveLeft,
        @NotNull Quantizer quantizer
    ) {
        int bestMode = IntraPrediction.DC_PRED;
        double bestCost = Double.MAX_VALUE;

        for (int mode = 0; mode < IntraPrediction.NUM_16x16_MODES; mode++) {
            // Generate 16x16 prediction by predicting each 4x4 sub-block
            short[] predicted = new short[256];
            predict16x16(predicted, aboveRow, leftCol, aboveLeft, mode);

            // Compute distortion (SSD)
            long ssd = IntraPrediction.computeSSD(mb.y, predicted);

            // Estimate rate (count of non-zero quantized coefficients)
            int nonZero = estimateRate(mb.y, predicted, quantizer);

            double cost = ssd + lambda * nonZero;

            if (cost < bestCost) {
                bestCost = cost;
                bestMode = mode;
            }
        }

        return bestMode;
    }

    /**
     * Selects the best 8x8 chroma prediction mode.
     *
     * @param cbSamples the 64-element Cb samples
     * @param crSamples the 64-element Cr samples
     * @param aboveCb 8 Cb samples above
     * @param leftCb 8 Cb samples left
     * @param aboveCr 8 Cr samples above
     * @param leftCr 8 Cr samples left
     * @param aboveLeftCb the Cb above-left
     * @param aboveLeftCr the Cr above-left
     * @return the best chroma mode (0-3)
     */
    int selectBestChromaMode(
        short @NotNull [] cbSamples, short @NotNull [] crSamples,
        short[] aboveCb, short[] leftCb,
        short[] aboveCr, short[] leftCr,
        short aboveLeftCb, short aboveLeftCr
    ) {
        int bestMode = IntraPrediction.DC_PRED;
        double bestCost = Double.MAX_VALUE;

        for (int mode = 0; mode < IntraPrediction.NUM_CHROMA_MODES; mode++) {
            short[] predCb = new short[64];
            short[] predCr = new short[64];
            predict8x8(predCb, aboveCb, leftCb, aboveLeftCb, mode);
            predict8x8(predCr, aboveCr, leftCr, aboveLeftCr, mode);

            double cost = (double) IntraPrediction.computeSSD(cbSamples, predCb) + IntraPrediction.computeSSD(crSamples, predCr);

            if (cost < bestCost) {
                bestCost = cost;
                bestMode = mode;
            }
        }

        return bestMode;
    }

    private void predict16x16(short[] predicted, short[] aboveRow, short[] leftCol, short aboveLeft, int mode) {
        switch (mode) {
            case IntraPrediction.DC_PRED -> {
                int sum = 0;
                int count = 0;
                if (aboveRow != null) { for (int i = 0; i < 16; i++) sum += aboveRow[i]; count += 16; }
                if (leftCol != null) { for (int i = 0; i < 16; i++) sum += leftCol[i]; count += 16; }
                short dc = count > 0 ? (short) ((sum + count / 2) / count) : (short) 128;
                java.util.Arrays.fill(predicted, dc);
            }
            case IntraPrediction.V_PRED -> {
                if (aboveRow == null) { java.util.Arrays.fill(predicted, (short) 127); return; }
                for (int y = 0; y < 16; y++)
                    System.arraycopy(aboveRow, 0, predicted, y * 16, 16);
            }
            case IntraPrediction.H_PRED -> {
                if (leftCol == null) { java.util.Arrays.fill(predicted, (short) 129); return; }
                for (int y = 0; y < 16; y++)
                    for (int x = 0; x < 16; x++)
                        predicted[y * 16 + x] = leftCol[y];
            }
            case IntraPrediction.TM_PRED -> {
                for (int y = 0; y < 16; y++)
                    for (int x = 0; x < 16; x++) {
                        int a = aboveRow != null ? aboveRow[x] : 127;
                        int l = leftCol != null ? leftCol[y] : 129;
                        predicted[y * 16 + x] = (short) Math.clamp(a + l - aboveLeft, 0, 255);
                    }
            }
        }
    }

    private void predict8x8(short[] predicted, short[] above, short[] left, short aboveLeft, int mode) {
        switch (mode) {
            case IntraPrediction.DC_PRED -> {
                int sum = 0;
                int count = 0;
                if (above != null) { for (int i = 0; i < 8; i++) sum += above[i]; count += 8; }
                if (left != null) { for (int i = 0; i < 8; i++) sum += left[i]; count += 8; }
                short dc = count > 0 ? (short) ((sum + count / 2) / count) : (short) 128;
                java.util.Arrays.fill(predicted, dc);
            }
            case IntraPrediction.V_PRED -> {
                if (above == null) { java.util.Arrays.fill(predicted, (short) 127); return; }
                for (int y = 0; y < 8; y++)
                    System.arraycopy(above, 0, predicted, y * 8, 8);
            }
            case IntraPrediction.H_PRED -> {
                if (left == null) { java.util.Arrays.fill(predicted, (short) 129); return; }
                for (int y = 0; y < 8; y++)
                    for (int x = 0; x < 8; x++)
                        predicted[y * 8 + x] = left[y];
            }
            case IntraPrediction.TM_PRED -> {
                for (int y = 0; y < 8; y++)
                    for (int x = 0; x < 8; x++) {
                        int a = above != null ? above[x] : 127;
                        int l = left != null ? left[y] : 129;
                        predicted[y * 8 + x] = (short) Math.clamp(a + l - aboveLeft, 0, 255);
                    }
            }
        }
    }

    private int estimateRate(short[] original, short[] predicted, Quantizer quantizer) {
        int nonZero = 0;
        short[] residual = new short[16];
        short[] coefficients = new short[16];

        // Check a few 4x4 sub-blocks for non-zero quantized coefficients
        for (int by = 0; by < 4; by++) {
            for (int bx = 0; bx < 4; bx++) {
                // Extract 4x4 block residual
                for (int y = 0; y < 4; y++)
                    for (int x = 0; x < 4; x++)
                        residual[y * 4 + x] = (short) (original[(by * 4 + y) * 16 + bx * 4 + x]
                            - predicted[(by * 4 + y) * 16 + bx * 4 + x]);

                DCT.forwardDCT(residual, coefficients);
                quantizer.quantizeY(coefficients);

                for (short c : coefficients)
                    if (c != 0) nonZero++;
            }
        }

        return nonZero;
    }

}
