package dev.simplified.image.pixel;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;

class PixelDiffTest {

    @Nested
    @DisplayName("ABSOLUTE")
    class Absolute {

        @Test
        @DisplayName("both transparent → transparent")
        void bothTransparent() {
            PixelBuffer a = PixelBuffer.create(1, 1);
            PixelBuffer b = PixelBuffer.create(1, 1);
            assertThat(PixelDiff.absolute(a, b).getPixel(0, 0), equalTo(ColorMath.TRANSPARENT));
        }

        @Test
        @DisplayName("matching opaque pixels → faint dark grey match marker")
        void matchInSilhouette() {
            PixelBuffer a = filledOpaque(1, 1, 0x40, 0x80, 0xC0);
            PixelBuffer b = filledOpaque(1, 1, 0x40, 0x80, 0xC0);
            assertThat(PixelDiff.absolute(a, b).getPixel(0, 0), equalTo(PixelDiff.ABSOLUTE_MATCH));
        }

        @Test
        @DisplayName("coverage mismatch → magenta")
        void coverageMismatch() {
            PixelBuffer a = filledOpaque(1, 1, 0x40, 0x80, 0xC0);
            PixelBuffer b = PixelBuffer.create(1, 1);
            assertThat(PixelDiff.absolute(a, b).getPixel(0, 0), equalTo(PixelDiff.COVERAGE_MAGENTA));
        }

        @Test
        @DisplayName("per-channel delta is amplified ×4")
        void deltaAmplification() {
            PixelBuffer a = filledOpaque(1, 1, 0x40, 0x00, 0x00);
            PixelBuffer b = filledOpaque(1, 1, 0x30, 0x00, 0x00);
            int out = PixelDiff.absolute(a, b).getPixel(0, 0);
            assertThat(ColorMath.red(out), equalTo(0x40));
            assertThat(ColorMath.green(out), equalTo(0));
            assertThat(ColorMath.blue(out), equalTo(0));
        }

        @Test
        @DisplayName("amplified channel saturates at 255")
        void deltaSaturates() {
            PixelBuffer a = filledOpaque(1, 1, 0xFF, 0x00, 0x00);
            PixelBuffer b = filledOpaque(1, 1, 0x00, 0x00, 0x00);
            int out = PixelDiff.absolute(a, b).getPixel(0, 0);
            assertThat(ColorMath.red(out), equalTo(255));
        }
    }

    @Nested
    @DisplayName("OVER_WHITE")
    class OverWhite {

        @Test
        @DisplayName("matching opaque pixels → transparent (perceptual match)")
        void matchTransparent() {
            PixelBuffer a = filledOpaque(1, 1, 0x40, 0x80, 0xC0);
            PixelBuffer b = filledOpaque(1, 1, 0x40, 0x80, 0xC0);
            assertThat(PixelDiff.overWhite(a, b).getPixel(0, 0), equalTo(ColorMath.TRANSPARENT));
        }

        @Test
        @DisplayName("both transparent → transparent (composite-over-white collapses to identity)")
        void bothTransparent() {
            PixelBuffer a = PixelBuffer.create(1, 1);
            PixelBuffer b = PixelBuffer.create(1, 1);
            assertThat(PixelDiff.overWhite(a, b).getPixel(0, 0), equalTo(ColorMath.TRANSPARENT));
        }

        @Test
        @DisplayName("AA edge spill (same raw RGB but different alphas) reads as small delta")
        void aaEdgeSpill() {
            // Both pixels are nearly transparent; ABSOLUTE would show large raw RGB delta,
            // OVER_WHITE compositing collapses to a small perceptual delta.
            int rgb = 0xFF; // saturated red component
            int aA = 0x10;
            int aB = 0x14;
            PixelBuffer a = PixelBuffer.create(1, 1);
            PixelBuffer b = PixelBuffer.create(1, 1);
            a.setPixel(0, 0, ColorMath.pack(aA, rgb, 0, 0));
            b.setPixel(0, 0, ColorMath.pack(aB, rgb, 0, 0));
            int out = PixelDiff.overWhite(a, b).getPixel(0, 0);
            // Perceptual delta is bounded; amplifying ×4 still keeps it well under 255.
            assertThat(ColorMath.red(out) < 64, equalTo(true));
        }
    }

    @Nested
    @DisplayName("SIGNED_LUMA")
    class SignedLuma {

        @Test
        @DisplayName("matching pixels → mid-grey baseline")
        void match() {
            PixelBuffer a = filledOpaque(1, 1, 0x40, 0x80, 0xC0);
            PixelBuffer b = filledOpaque(1, 1, 0x40, 0x80, 0xC0);
            int out = PixelDiff.signedLuma(a, b).getPixel(0, 0);
            assertThat(ColorMath.red(out), equalTo(128));
            assertThat(ColorMath.green(out), equalTo(128));
            assertThat(ColorMath.blue(out), equalTo(128));
        }

        @Test
        @DisplayName("a brighter than b → warm shift toward red")
        void aBrighter() {
            PixelBuffer a = filledOpaque(1, 1, 0xFF, 0xFF, 0xFF);
            PixelBuffer b = filledOpaque(1, 1, 0x00, 0x00, 0x00);
            int out = PixelDiff.signedLuma(a, b).getPixel(0, 0);
            assertThat(ColorMath.red(out) > ColorMath.blue(out), equalTo(true));
        }

        @Test
        @DisplayName("b brighter than a → cool shift toward blue")
        void bBrighter() {
            PixelBuffer a = filledOpaque(1, 1, 0x00, 0x00, 0x00);
            PixelBuffer b = filledOpaque(1, 1, 0xFF, 0xFF, 0xFF);
            int out = PixelDiff.signedLuma(a, b).getPixel(0, 0);
            assertThat(ColorMath.blue(out) > ColorMath.red(out), equalTo(true));
        }
    }

    @Nested
    @DisplayName("SIGNED_RGB")
    class SignedRgb {

        @Test
        @DisplayName("matching pixels → mid-grey")
        void match() {
            PixelBuffer a = filledOpaque(1, 1, 0x40, 0x80, 0xC0);
            PixelBuffer b = filledOpaque(1, 1, 0x40, 0x80, 0xC0);
            int out = PixelDiff.signedRgb(a, b).getPixel(0, 0);
            assertThat(out, equalTo(ColorMath.pack(255, 128, 128, 128)));
        }

        @Test
        @DisplayName("green-only divergence → green tint")
        void greenOnly() {
            PixelBuffer a = filledOpaque(1, 1, 0x80, 0x90, 0x80);
            PixelBuffer b = filledOpaque(1, 1, 0x80, 0x80, 0x80);
            int out = PixelDiff.signedRgb(a, b).getPixel(0, 0);
            assertThat(ColorMath.red(out), equalTo(128));
            assertThat(ColorMath.green(out), equalTo(128 + 16 * 2));
            assertThat(ColorMath.blue(out), equalTo(128));
        }
    }

    @Nested
    @DisplayName("COVERAGE")
    class Coverage {

        @Test
        @DisplayName("both transparent → transparent")
        void bothTransparent() {
            PixelBuffer a = PixelBuffer.create(1, 1);
            PixelBuffer b = PixelBuffer.create(1, 1);
            assertThat(PixelDiff.coverage(a, b).getPixel(0, 0), equalTo(ColorMath.TRANSPARENT));
        }

        @Test
        @DisplayName("both opaque → dark grey")
        void bothOpaque() {
            PixelBuffer a = filledOpaque(1, 1, 0xFF, 0x00, 0x00);
            PixelBuffer b = filledOpaque(1, 1, 0x00, 0x00, 0xFF);
            assertThat(PixelDiff.coverage(a, b).getPixel(0, 0), equalTo(PixelDiff.COVERAGE_BOTH));
        }

        @Test
        @DisplayName("a-only → magenta")
        void aOnly() {
            PixelBuffer a = filledOpaque(1, 1, 0xFF, 0xFF, 0xFF);
            PixelBuffer b = PixelBuffer.create(1, 1);
            assertThat(PixelDiff.coverage(a, b).getPixel(0, 0), equalTo(PixelDiff.COVERAGE_MAGENTA));
        }

        @Test
        @DisplayName("b-only → cyan")
        void bOnly() {
            PixelBuffer a = PixelBuffer.create(1, 1);
            PixelBuffer b = filledOpaque(1, 1, 0xFF, 0xFF, 0xFF);
            assertThat(PixelDiff.coverage(a, b).getPixel(0, 0), equalTo(PixelDiff.COVERAGE_CYAN));
        }
    }

    @Nested
    @DisplayName("dispatcher + PixelBuffer.diff delegate")
    class Dispatcher {

        @Test
        @DisplayName("PixelDiff.of routes ABSOLUTE through absolute()")
        void ofRoutes() {
            PixelBuffer a = filledOpaque(2, 2, 0xFF, 0x00, 0x00);
            PixelBuffer b = filledOpaque(2, 2, 0x00, 0x00, 0x00);
            for (DiffType type : DiffType.values()) {
                PixelBuffer viaOf = PixelDiff.of(a, b, type);
                PixelBuffer viaInstance = a.diff(b, type);
                for (int y = 0; y < 2; y++) {
                    for (int x = 0; x < 2; x++) {
                        assertThat(
                            "instance method delegates to PixelDiff.of for " + type,
                            viaInstance.getPixel(x, y),
                            equalTo(viaOf.getPixel(x, y))
                        );
                    }
                }
            }
        }

        @Test
        @DisplayName("result is sized to min(w, h) intersection")
        void intersectionSize() {
            PixelBuffer a = PixelBuffer.create(4, 6);
            PixelBuffer b = PixelBuffer.create(5, 3);
            PixelBuffer out = PixelDiff.absolute(a, b);
            assertThat(out.width(), equalTo(4));
            assertThat(out.height(), equalTo(3));
        }
    }

    private static PixelBuffer filledOpaque(int w, int h, int r, int g, int b) {
        PixelBuffer buf = PixelBuffer.create(w, h);
        int argb = ColorMath.pack(0xFF, r, g, b);
        buf.fill(argb);
        return buf;
    }

}
