package dev.simplified.image.codec.webp.vp8;

import dev.simplified.image.PixelBuffer;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

/**
 * Pins the encoder's intra mode decision to libwebp's rate-distortion scorer: the lambdas its
 * {@code SetupMatrices} derives, the fixed mode costs it reads, and the choice it makes.
 */
class VP8RateDistortionTest {

    /**
     * Returns libwebp's {@code ExpandMatrix} step: the rounded mean of a matrix's 16 steps, its DC
     * step and 15 copies of its AC step.
     */
    private static int meanStep(int dc, int ac) {
        return (dc + 15 * ac + 8) >> 4;
    }

    @Test
    @DisplayName("every quantizer's R-D lambdas are libwebp's SetupMatrices values over the mean step")
    void lambdasAreLibwebpSetupMatrices() {
        for (int qi = 0; qi <= 127; qi++) {
            VP8Encoder.State s = new VP8Encoder.State(16, 16, qi, true, null);
            int qI4 = meanStep(s.y1Dc, s.y1Ac);
            int qI16 = meanStep(s.y2Dc, s.y2Ac);
            int qUv = meanStep(s.uvDc, s.uvAc);
            String at = "qi " + qi + " ";

            assertThat(at + "lambda_i4", s.lambdaI4, is(Math.max(1, (3 * qI4 * qI4) >> 7)));
            assertThat(at + "lambda_i16", s.lambdaI16, is(Math.max(1, 3 * qI16 * qI16)));
            assertThat(at + "lambda_uv", s.lambdaUv, is(Math.max(1, (3 * qUv * qUv) >> 6)));
            assertThat(at + "lambda_mode", s.lambdaMode, is(Math.max(1, (qI4 * qI4) >> 7)));
            assertThat(at + "lambda_trellis_i4", s.lambdaTrellisI4, is(Math.max(1, (7 * qI4 * qI4) >> 3)));
            assertThat(at + "lambda_trellis_i16", s.lambdaTrellisI16, is(Math.max(1, (qI16 * qI16) >> 2)));
            assertThat(at + "lambda_trellis_uv", s.lambdaTrellisUv, is(Math.max(1, (qUv * qUv) << 1)));
        }
        // Without spatial noise shaping libwebp scales tlambda by 0, and its floor of 1 holds.
        assertThat("tlambda", VP8Encoder.TLAMBDA, is(1));
    }

    @Test
    @DisplayName("the fixed 16x16 and chroma mode costs are libwebp's, each at this codec's number for its mode")
    void fixedModeCostsFollowTheModeNumbers() {
        // libwebp's VP8FixedCostsI16 and VP8FixedCostsUV, in libwebp's mode order: DC, TM, V, H.
        int[] i16 = { 663, 919, 872, 919 };
        int[] uv = { 302, 984, 439, 642 };
        int[] libwebpOrder = {
            IntraPrediction.DC_PRED, IntraPrediction.TM_PRED, IntraPrediction.V_PRED, IntraPrediction.H_PRED
        };

        for (int k = 0; k < 4; k++) {
            assertThat("16x16 mode " + libwebpOrder[k], VP8Costs.FIXED_COSTS_I16[libwebpOrder[k]], is(i16[k]));
            assertThat("chroma mode " + libwebpOrder[k], VP8Costs.FIXED_COSTS_UV[libwebpOrder[k]], is(uv[k]));
        }
    }

    @Test
    @DisplayName("the texture distortion is libwebp's weighted Hadamard difference")
    void textureDistortionIsWeightedHadamard() {
        // One sample of 255 against black puts 255 into every Hadamard coefficient, and libwebp's
        // weights sum to 256, so the distortion is 255 * 256 / 32 whichever block comes first.
        short[] black = new short[16];
        short[] impulse = new short[16];
        impulse[0] = 255;
        assertThat(DCT.textureDistortion4x4(black, impulse, 0, 4), is(2040));
        assertThat(DCT.textureDistortion4x4(impulse, black, 0, 4), is(2040));

        Random random = new Random(42);
        short[] a = new short[256];
        short[] b = new short[256];
        for (int trial = 0; trial < 100; trial++) {
            for (int i = 0; i < 256; i++) {
                a[i] = (short) random.nextInt(256);
                b[i] = (short) random.nextInt(256);
            }
            int total = 0;
            for (int y = 0; y < 16; y += 4) {
                for (int x = 0; x < 16; x += 4) {
                    int offset = y * 16 + x;
                    int expected = Math.abs(weightedHadamard(b, offset) - weightedHadamard(a, offset)) >> 5;
                    assertThat(DCT.textureDistortion4x4(a, b, offset, 16), is(expected));
                    total += expected;
                }
            }
            assertThat(DCT.textureDistortion16x16(a, b), is(total));
        }
    }

    /**
     * Weighs a 4x4 block's absolute Hadamard coefficients by libwebp's {@code kWeightY}, taking the
     * transform as the matrix product {@code H * X * H^T} rather than libwebp's butterflies.
     */
    private static int weightedHadamard(short[] samples, int offset) {
        int[][] hadamard = { { 1, 1, 1, 1 }, { 1, 1, -1, -1 }, { 1, -1, -1, 1 }, { 1, -1, 1, -1 } };
        int[] weightY = { 38, 32, 20, 9, 32, 28, 17, 7, 20, 17, 10, 4, 9, 7, 4, 2 };
        int sum = 0;
        for (int v = 0; v < 4; v++) {
            for (int h = 0; h < 4; h++) {
                int coefficient = 0;
                for (int y = 0; y < 4; y++)
                    for (int x = 0; x < 4; x++)
                        coefficient += hadamard[v][y] * samples[offset + y * 16 + x] * hadamard[h][x];
                sum += weightY[v * 4 + h] * Math.abs(coefficient);
            }
        }
        return sum;
    }

    @Test
    @DisplayName("signalling B_PRED costs the 211 libwebp starts a B_PRED score at")
    void bPredSignalCost() {
        assertThat(VP8Costs.bitCost(0, 145), is(211));
    }

    @Test
    @DisplayName("a flat frame codes every macroblock with 16x16 prediction")
    void flatFramePredictsWhole() {
        PixelBuffer flat = PixelBuffer.create(48, 48);
        flat.fill(0xFF5A7D3C);

        for (float quality : new float[] { 0.25f, 0.75f, 1.0f }) {
            boolean[] i4 = VP8Decoder.decodeWithMetadata(VP8Encoder.encode(flat, quality)).metadata().mbIsI4x4();
            for (int mb = 0; mb < i4.length; mb++)
                assertThat("quality " + quality + " macroblock " + mb + " is B_PRED", i4[mb], is(false));
        }
    }

}
