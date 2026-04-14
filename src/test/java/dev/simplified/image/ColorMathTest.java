package dev.simplified.image;

import dev.simplified.image.pixel.BlendMode;
import dev.simplified.image.pixel.ColorMath;
import dev.simplified.image.pixel.PixelBuffer;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.lessThanOrEqualTo;

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
    @DisplayName("tint multiplies RGB channels and preserves alpha")
    void tintPreservesAlpha() {
        PixelBuffer source = PixelBuffer.of(new int[]{ 0x80FFFFFF }, 1, 1);
        PixelBuffer tinted = ColorMath.tint(source, 0xFFFF0000);
        assertThat(ColorMath.alpha(tinted.getPixel(0, 0)), equalTo(0x80));
        assertThat(ColorMath.red(tinted.getPixel(0, 0)), equalTo(0xFF));
        assertThat(ColorMath.green(tinted.getPixel(0, 0)), equalTo(0x00));
        assertThat(ColorMath.blue(tinted.getPixel(0, 0)), equalTo(0x00));
    }

}
