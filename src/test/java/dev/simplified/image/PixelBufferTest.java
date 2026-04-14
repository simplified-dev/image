package dev.simplified.image;

import dev.simplified.image.pixel.BlendMode;
import dev.simplified.image.pixel.ColorMath;
import dev.simplified.image.pixel.PixelBuffer;
import dev.simplified.image.pixel.Resample;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.awt.Rectangle;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class PixelBufferTest {

    private static PixelBuffer gradient(int w, int h) {
        int[] data = new int[w * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                data[y * w + x] = ColorMath.pack(255, x & 0xFF, y & 0xFF, (x + y) & 0xFF);
        return PixelBuffer.of(data, w, h);
    }

    @Test
    @DisplayName("getPixel / setPixel roundtrip")
    void pixelRoundtrip() {
        PixelBuffer buffer = PixelBuffer.create(4, 4);
        buffer.setPixel(2, 3, 0xDEADBEEF);
        assertThat(buffer.getPixel(2, 3), equalTo(0xDEADBEEF));
    }

    @Test
    @DisplayName("getRow / setRow roundtrip")
    void rowRoundtrip() {
        PixelBuffer buffer = PixelBuffer.create(3, 2);
        int[] row = { 0x11111111, 0x22222222, 0x33333333 };
        buffer.setRow(1, row);
        assertArrayEquals(row, buffer.getRow(1));
    }

    @Test
    @DisplayName("getPixels / setPixels roundtrip a sub-region")
    void regionRoundtrip() {
        PixelBuffer buffer = gradient(8, 8);
        int[] region = buffer.getPixels(2, 2, 3, 3, null, 0, 0);

        PixelBuffer target = PixelBuffer.create(8, 8);
        target.setPixels(2, 2, 3, 3, region, 0, 3);
        assertThat(target.getPixel(3, 3), equalTo(buffer.getPixel(3, 3)));
        assertThat(target.getPixel(0, 0), equalTo(0));
    }

    @Test
    @DisplayName("fillRect clips out-of-bounds regions")
    void fillRectClips() {
        PixelBuffer buffer = PixelBuffer.create(4, 4);
        buffer.fillRect(-2, -2, 4, 4, 0xFF00FF00);
        assertThat(buffer.getPixel(0, 0), equalTo(0xFF00FF00));
        assertThat(buffer.getPixel(2, 2), equalTo(0));
    }

    @Test
    @DisplayName("contains reports in-bounds coordinates")
    void containsCoordinates() {
        PixelBuffer buffer = PixelBuffer.create(4, 4);
        assertThat(buffer.contains(0, 0), is(true));
        assertThat(buffer.contains(3, 3), is(true));
        assertThat(buffer.contains(4, 3), is(false));
        assertThat(buffer.contains(-1, 0), is(false));
    }

    @Test
    @DisplayName("flipHorizontal twice is identity")
    void flipHorizontalIdentity() {
        PixelBuffer a = gradient(5, 3);
        PixelBuffer b = a.copy();
        b.flipHorizontal();
        b.flipHorizontal();
        assertThat(b, equalTo(a));
    }

    @Test
    @DisplayName("flipVertical twice is identity")
    void flipVerticalIdentity() {
        PixelBuffer a = gradient(4, 5);
        PixelBuffer b = a.copy();
        b.flipVertical();
        b.flipVertical();
        assertThat(b, equalTo(a));
    }

    @Test
    @DisplayName("rotate90 four times is identity")
    void rotate90Identity() {
        PixelBuffer a = gradient(4, 3);
        PixelBuffer r = a.rotate90().rotate90().rotate90().rotate90();
        assertThat(r, equalTo(a));
    }

    @Test
    @DisplayName("rotate180 twice is identity")
    void rotate180Identity() {
        PixelBuffer a = gradient(4, 3);
        assertThat(a.rotate180().rotate180(), equalTo(a));
    }

    @Test
    @DisplayName("rotate270 is the inverse of rotate90")
    void rotate270InverseOfRotate90() {
        PixelBuffer a = gradient(4, 3);
        assertThat(a.rotate90().rotate270(), equalTo(a));
    }

    @Test
    @DisplayName("crop returns expected dimensions and contents")
    void cropRegion() {
        PixelBuffer a = gradient(6, 6);
        PixelBuffer c = a.crop(1, 2, 3, 2);
        assertThat(c.width(), equalTo(3));
        assertThat(c.height(), equalTo(2));
        assertThat(c.getPixel(0, 0), equalTo(a.getPixel(1, 2)));
        assertThat(c.getPixel(2, 1), equalTo(a.getPixel(3, 3)));
    }

    @Test
    @DisplayName("crop rejects out-of-bounds regions")
    void cropRejectsOutOfBounds() {
        PixelBuffer a = PixelBuffer.create(4, 4);
        assertThrows(IllegalArgumentException.class, () -> a.crop(2, 2, 5, 5));
    }

    @Test
    @DisplayName("resize NEAREST agrees with blitScaled")
    void resizeNearestMatchesBlitScaled() {
        PixelBuffer a = gradient(6, 6);
        PixelBuffer nearest = a.resize(3, 3, Resample.NEAREST);

        PixelBuffer scaled = PixelBuffer.create(3, 3);
        scaled.blitScaled(a, 0, 0, 3, 3);
        assertThat(nearest, equalTo(scaled));
    }

    @Test
    @DisplayName("resize BILINEAR returns requested dimensions")
    void resizeBilinearDimensions() {
        PixelBuffer a = gradient(8, 8);
        PixelBuffer r = a.resize(4, 4, Resample.BILINEAR);
        assertThat(r.width(), equalTo(4));
        assertThat(r.height(), equalTo(4));
    }

    @Test
    @DisplayName("resize rejects non-positive dimensions")
    void resizeRejectsInvalidDimensions() {
        PixelBuffer a = PixelBuffer.create(4, 4);
        assertThrows(IllegalArgumentException.class, () -> a.resize(0, 4, Resample.NEAREST));
    }

    @Test
    @DisplayName("invert XORs RGB, preserving alpha")
    void invertPreservesAlpha() {
        PixelBuffer a = PixelBuffer.create(1, 1);
        a.setPixel(0, 0, 0x80112233);
        a.invert();
        assertThat(a.getPixel(0, 0), equalTo(0x80EEDDCC));
    }

    @Test
    @DisplayName("grayscale produces equal RGB channels")
    void grayscaleUniformChannels() {
        PixelBuffer a = PixelBuffer.create(1, 1);
        a.setPixel(0, 0, ColorMath.pack(255, 200, 100, 50));
        a.grayscale();
        int p = a.getPixel(0, 0);
        assertThat(ColorMath.red(p), equalTo(ColorMath.green(p)));
        assertThat(ColorMath.green(p), equalTo(ColorMath.blue(p)));
    }

    @Test
    @DisplayName("multiplyAlpha halves alpha and preserves RGB")
    void multiplyAlphaHalves() {
        PixelBuffer a = PixelBuffer.create(1, 1);
        a.setPixel(0, 0, 0xFF112233);
        a.multiplyAlpha(0.5f);
        int p = a.getPixel(0, 0);
        assertThat(ColorMath.alpha(p), equalTo(128));
        assertThat(p & 0x00FFFFFF, equalTo(0x00112233));
    }

    @Test
    @DisplayName("premultiply then unpremultiply roundtrips within tolerance")
    void premultiplyRoundtrip() {
        PixelBuffer a = PixelBuffer.create(1, 1);
        a.setPixel(0, 0, 0x80808080);
        a.premultiplyAlpha();
        a.unpremultiplyAlpha();
        int p = a.getPixel(0, 0);
        assertThat(ColorMath.alpha(p), equalTo(128));
        assertThat(Math.abs(ColorMath.red(p) - 128), org.hamcrest.Matchers.lessThanOrEqualTo(2));
    }

    @Test
    @DisplayName("opaqueBounds finds the tight box")
    void opaqueBoundsTight() {
        PixelBuffer a = PixelBuffer.create(10, 10);
        a.fillRect(2, 3, 4, 5, 0xFFFF0000);
        Rectangle r = a.opaqueBounds();
        assertThat(r, equalTo(new Rectangle(2, 3, 4, 5)));
    }

    @Test
    @DisplayName("opaqueBounds is zero-size for fully transparent buffer")
    void opaqueBoundsEmpty() {
        PixelBuffer a = PixelBuffer.create(4, 4);
        Rectangle r = a.opaqueBounds();
        assertThat(r.width, equalTo(0));
        assertThat(r.height, equalTo(0));
    }

    @Test
    @DisplayName("trim crops to opaque bounds")
    void trimCropsToOpaque() {
        PixelBuffer a = PixelBuffer.create(10, 10);
        a.fillRect(2, 3, 4, 5, 0xFFFF0000);
        PixelBuffer t = a.trim();
        assertThat(t.width(), equalTo(4));
        assertThat(t.height(), equalTo(5));
    }

    @Test
    @DisplayName("blit opaque source matches generic blend result")
    void blitOpaqueFastPathMatchesGeneric() {
        PixelBuffer src = gradient(4, 4);
        PixelBuffer opaqueSrc = PixelBuffer.of(src.pixels().clone(), 4, 4, false);

        PixelBuffer fast = PixelBuffer.create(8, 8);
        fast.blit(opaqueSrc, 2, 2, BlendMode.NORMAL);

        PixelBuffer generic = PixelBuffer.create(8, 8);
        generic.blit(src, 2, 2, BlendMode.NORMAL);

        assertThat(fast, equalTo(generic));
    }

    @Test
    @DisplayName("equals / hashCode follow deep pixel equality")
    void equalsAndHashCode() {
        PixelBuffer a = gradient(3, 3);
        PixelBuffer b = gradient(3, 3);
        PixelBuffer c = gradient(3, 3);
        c.setPixel(0, 0, 0);

        assertThat(a, equalTo(b));
        assertThat(a.hashCode(), equalTo(b.hashCode()));
        assertThat(a, not(equalTo(c)));
    }

}
