package dev.simplified.image.pixel;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.lessThanOrEqualTo;
import static org.hamcrest.Matchers.not;

class ColorMathTest {

    @Test
    @DisplayName("pack/unpack roundtrips an ARGB triple")
    void packUnpackRoundtrip() {
        int packed = ColorMath.pack(0x80, 0x12, 0x34, 0x56);
        assertThat(ColorMath.alpha(packed), equalTo(0x80));
        assertThat(ColorMath.red(packed), equalTo(0x12));
        assertThat(ColorMath.green(packed), equalTo(0x34));
        assertThat(ColorMath.blue(packed), equalTo(0x56));
    }

    @Test
    @DisplayName("hsvToArgb produces opaque white for value=1 saturation=0")
    void hsvPureValue() {
        int white = ColorMath.hsvToArgb(0f, 0f, 1f);
        assertThat(ColorMath.alpha(white), equalTo(0xFF));
        assertThat(ColorMath.red(white), equalTo(0xFF));
        assertThat(ColorMath.green(white), equalTo(0xFF));
        assertThat(ColorMath.blue(white), equalTo(0xFF));
    }

    @Test
    @DisplayName("hsvToArgb at hue=0 saturation=1 value=1 produces red")
    void hsvPureRed() {
        int red = ColorMath.hsvToArgb(0f, 1f, 1f);
        assertThat(ColorMath.red(red), equalTo(0xFF));
        assertThat(ColorMath.green(red), equalTo(0x00));
        assertThat(ColorMath.blue(red), equalTo(0x00));
    }

    @Test
    @DisplayName("hsvToArgb at hue=120 saturation=1 value=1 produces green")
    void hsvPureGreen() {
        int green = ColorMath.hsvToArgb(120f, 1f, 1f);
        assertThat(ColorMath.red(green), equalTo(0x00));
        assertThat(ColorMath.green(green), equalTo(0xFF));
        assertThat(ColorMath.blue(green), equalTo(0x00));
    }

    @Test
    @DisplayName("blend NORMAL with opaque src returns src unchanged")
    void blendNormalOpaqueSource() {
        int src = 0xFF123456;
        int dst = 0xFFABCDEF;
        assertThat(ColorMath.blend(src, dst, BlendMode.NORMAL), equalTo(src));
    }

    @Test
    @DisplayName("blend NORMAL with transparent src returns dst unchanged")
    void blendNormalTransparentSource() {
        int src = 0x00FFFFFF;
        int dst = 0xFFABCDEF;
        assertThat(ColorMath.blend(src, dst, BlendMode.NORMAL), equalTo(dst));
    }

    @Test
    @DisplayName("blend REPLACE returns src verbatim whatever its alpha")
    void blendReplaceReturnsSource() {
        int dst = 0xFFABCDEF;
        assertThat(ColorMath.blend(0xFF123456, dst, BlendMode.REPLACE), equalTo(0xFF123456));
        assertThat(ColorMath.blend(0x80123456, dst, BlendMode.REPLACE), equalTo(0x80123456));
        assertThat(ColorMath.blend(0x00FFFFFF, dst, BlendMode.REPLACE), equalTo(0x00FFFFFF));
    }

    @Test
    @DisplayName("blend REPLACE agrees with NORMAL at full alpha and differs at partial alpha")
    void blendReplaceAgreesWithNormalAtFullAlpha() {
        int dst = 0xFFABCDEF;
        int opaque = 0xFF123456;
        assertThat(ColorMath.blend(opaque, dst, BlendMode.REPLACE),
            equalTo(ColorMath.blend(opaque, dst, BlendMode.NORMAL)));

        int partial = 0x80123456;
        assertThat(ColorMath.blend(partial, dst, BlendMode.REPLACE),
            not(equalTo(ColorMath.blend(partial, dst, BlendMode.NORMAL))));
    }

    @Test
    @DisplayName("blend NORMAL darkens a pixel drawn over transparent a second time")
    void blendNormalRedrawDarkens() {
        int drawn = ColorMath.blend(0x80FFFFFF, ColorMath.TRANSPARENT, BlendMode.NORMAL);
        assertThat(drawn, equalTo(0x80808080));
        assertThat(ColorMath.blend(drawn, ColorMath.TRANSPARENT, BlendMode.NORMAL), equalTo(0x80404040));
    }

    @Test
    @DisplayName("blend PREMULTIPLIED lays a pixel drawn over transparent back over transparent unchanged")
    void blendPremultipliedOverTransparentIsIdentity() {
        for (int rgb : new int[]{ 0xFFFFFF, 0x123456, 0xFF8000 }) {
            for (int alpha = 1; alpha < 255; alpha++) {
                int drawn = ColorMath.blend(ColorMath.withAlpha(rgb, alpha), ColorMath.TRANSPARENT, BlendMode.NORMAL);
                assertThat(ColorMath.blend(drawn, ColorMath.TRANSPARENT, BlendMode.PREMULTIPLIED), equalTo(drawn));
            }
        }
    }

    @Test
    @DisplayName("blend PREMULTIPLIED over an opaque pixel matches drawing the source there directly within one unit")
    void blendPremultipliedMatchesDirectDraw() {
        int[] destinations = { 0xFF000000, 0xFFFFFFFF, 0xFF204060, 0xFFABCDEF };
        for (int rgb : new int[]{ 0xFFFFFF, 0x123456, 0xFF8000 }) {
            for (int alpha = 1; alpha < 255; alpha++) {
                int src = ColorMath.withAlpha(rgb, alpha);
                int drawn = ColorMath.blend(src, ColorMath.TRANSPARENT, BlendMode.NORMAL);

                for (int dst : destinations) {
                    int direct = ColorMath.blend(src, dst, BlendMode.NORMAL);
                    int laid = ColorMath.blend(drawn, dst, BlendMode.PREMULTIPLIED);
                    assertThat(ColorMath.alpha(laid), equalTo(0xFF));
                    assertThat(Math.abs(ColorMath.red(laid) - ColorMath.red(direct)), lessThanOrEqualTo(1));
                    assertThat(Math.abs(ColorMath.green(laid) - ColorMath.green(direct)), lessThanOrEqualTo(1));
                    assertThat(Math.abs(ColorMath.blue(laid) - ColorMath.blue(direct)), lessThanOrEqualTo(1));
                }
            }
        }
    }

    @Test
    @DisplayName("blend PREMULTIPLIED agrees with NORMAL at zero and full source alpha")
    void blendPremultipliedAgreesWithNormalAtTheEnds() {
        int dst = 0x80ABCDEF;
        assertThat(ColorMath.blend(0xFF123456, dst, BlendMode.PREMULTIPLIED),
            equalTo(ColorMath.blend(0xFF123456, dst, BlendMode.NORMAL)));
        assertThat(ColorMath.blend(0x00FFFFFF, dst, BlendMode.PREMULTIPLIED),
            equalTo(ColorMath.blend(0x00FFFFFF, dst, BlendMode.NORMAL)));
    }

    @Test
    @DisplayName("blend PREMULTIPLIED accumulates alpha and clamps a colour that exceeds it")
    void blendPremultipliedAccumulatesAndClamps() {
        // 0x40 + 0x40 * (1 - 128/255) = 95.9, and 128 + 128 * (1 - 128/255) = 191.7
        assertThat(ColorMath.blend(0x80404040, 0x80404040, BlendMode.PREMULTIPLIED), equalTo(0xC0606060));
        assertThat(ColorMath.blend(0x80FFFFFF, 0xFFFFFFFF, BlendMode.PREMULTIPLIED), equalTo(0xFFFFFFFF));
    }

    @Test
    @DisplayName("blend MULTIPLY darkens towards black for mid-gray src")
    void blendMultiplyDarkens() {
        int src = 0xFF808080;
        int dst = 0xFFFFFFFF;
        int blended = ColorMath.blend(src, dst, BlendMode.MULTIPLY);
        int r = ColorMath.red(blended);
        assertThat(r, lessThanOrEqualTo(0x81));
    }

    @Test
    @DisplayName("blend ADD saturates at 0xFF")
    void blendAddSaturates() {
        int src = 0xFFFFFFFF;
        int dst = 0xFF808080;
        int blended = ColorMath.blend(src, dst, BlendMode.ADD);
        assertThat(ColorMath.red(blended), equalTo(0xFF));
        assertThat(ColorMath.green(blended), equalTo(0xFF));
        assertThat(ColorMath.blue(blended), equalTo(0xFF));
    }

    @Test
    @DisplayName("blend MULTIPLY rounds half-up to match OpenGL normalized fixed-point")
    void blendMultiplyRoundsHalfUp() {
        // 200 * 200 / 255 = 156.862... Truncation gives 156, round-half-up gives 157.
        // Vanilla GLSL writes a normalized-fixed-point byte via floor(v * 255 + 0.5), so the
        // correct value is 157. Pinning this catches regression to integer truncation.
        int blended = ColorMath.blend(0xFFC8C8C8, 0xFFC8C8C8, BlendMode.MULTIPLY);
        assertThat(ColorMath.red(blended), equalTo(157));
        assertThat(ColorMath.green(blended), equalTo(157));
        assertThat(ColorMath.blue(blended), equalTo(157));
    }

    @Test
    @DisplayName("blend OVERLAY rounds half-up in both dark/light branches")
    void blendOverlayRoundsHalfUp() {
        // dark branch d < 128: 2 * 100 * 100 / 255 = 78.43... -> 78
        // light branch d >= 128: 255 - (2 * 100 * 100 / 255) where the inner uses (255-s)(255-d)
        //   for s=d=156: 255 - (2 * 99 * 99 / 255) = 255 - 19602/255 = 255 - 76.87... = 255 - 77 = 178
        int darkBranch = ColorMath.blend(0xFF646464, 0xFF646464, BlendMode.OVERLAY);
        assertThat(ColorMath.red(darkBranch), equalTo(78));

        int lightBranch = ColorMath.blend(0xFF9C9C9C, 0xFF9C9C9C, BlendMode.OVERLAY);
        assertThat(ColorMath.red(lightBranch), equalTo(178));
    }

    @Test
    @DisplayName("blend QUADRATIC_ADD squares the source (round-half-up) and adds the destination")
    void blendQuadraticAddSquaresAndAdds() {
        // out = min(255, src*src/255 + dst). src=200: 200*200/255 = 156.86 -> 157; + dst 32 = 189.
        int blended = ColorMath.blend(0xFFC8C8C8, 0xFF202020, BlendMode.QUADRATIC_ADD);
        assertThat(ColorMath.red(blended), equalTo(189));
        assertThat(ColorMath.green(blended), equalTo(189));
        assertThat(ColorMath.blue(blended), equalTo(189));
    }

    @Test
    @DisplayName("blend QUADRATIC_ADD saturates at 0xFF and preserves the destination alpha")
    void blendQuadraticAddSaturatesAndKeepsDestinationAlpha() {
        // src=255: 255*255/255 = 255; + dst 64 -> clamp 255. Alpha must come from dst, not src.
        int blended = ColorMath.blend(0xFFFFFFFF, 0x80404040, BlendMode.QUADRATIC_ADD);
        assertThat(ColorMath.alpha(blended), equalTo(0x80));
        assertThat(ColorMath.red(blended), equalTo(0xFF));
        assertThat(ColorMath.green(blended), equalTo(0xFF));
        assertThat(ColorMath.blue(blended), equalTo(0xFF));
    }

    @Test
    @DisplayName("blend QUADRATIC_ADD with a fully transparent source is a no-op")
    void blendQuadraticAddTransparentSourceNoOp() {
        int dst = 0xFF123456;
        assertThat(ColorMath.blend(0x00FFFFFF, dst, BlendMode.QUADRATIC_ADD), equalTo(dst));
    }

    @Test
    @DisplayName("blend STRAIGHT lays straight colour over straight colour and answers straight colour")
    void blendStraightOverStraight() {
        assertThat(ColorMath.blend(0x80FF0000, 0xFF0000FF, BlendMode.STRAIGHT), equalTo(0xFF80007F));
        assertThat(ColorMath.blend(0x80FF0000, 0x800000FF, BlendMode.STRAIGHT), equalTo(0xC0AA0055));
    }

    @Test
    @DisplayName("blend STRAIGHT keeps either pixel whole where the other is fully transparent, and an opaque source")
    void blendStraightEdges() {
        assertThat(ColorMath.blend(0x00FFFFFF, 0x80123456, BlendMode.STRAIGHT), equalTo(0x80123456));
        assertThat(ColorMath.blend(0x80123456, 0x00FFFFFF, BlendMode.STRAIGHT), equalTo(0x80123456));
        assertThat(ColorMath.blend(0xFF123456, 0x80ABCDEF, BlendMode.STRAIGHT), equalTo(0xFF123456));
    }

}
