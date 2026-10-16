package dev.simplified.image;

import dev.simplified.image.pixel.BlendMode;
import dev.simplified.image.pixel.ColorMath;
import dev.simplified.image.pixel.PixelBuffer;
import dev.simplified.image.pixel.Resample;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.awt.*;
import java.util.Random;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.*;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class PixelBufferTest {

    /**
     * Source width, source height, destination width, destination height, destination origin, target
     * width and target height of each scaled blit held to the frozen per-pixel loop: a 16x
     * magnification, a halving, magnifications by 100 / 16 and 13 / 7, a shrink by 4 / 9, a
     * one-to-one copy, an origin left of and above the target that starts the blit part way through a
     * magnified texel, a destination wider than the target clipped on both sides, an origin that clips
     * the right and bottom, a magnification tall enough for its rows to run in parallel, and a
     * destination so wide that each column's product overflows an int and wraps.
     */
    private static final int[][] SCALED_BLITS = {
        { 16, 16, 256, 256, 0, 0, 256, 256 },
        { 512, 512, 256, 256, 0, 0, 256, 256 },
        { 16, 16, 100, 100, 0, 0, 100, 100 },
        { 7, 7, 13, 13, 0, 0, 13, 13 },
        { 9, 9, 4, 4, 1, 1, 6, 6 },
        { 13, 11, 13, 11, 2, -3, 20, 20 },
        { 16, 16, 64, 48, -9, -5, 40, 40 },
        { 16, 8, 300, 40, -37, 3, 200, 50 },
        { 16, 16, 256, 256, 100, 120, 256, 256 },
        { 5, 3, 64, 70, -3, -1, 70, 80 },
        { 50000, 2, 100000, 2, -60000, -1, 10, 1 }
    };

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
    @DisplayName("blitScaled PREMULTIPLIED onto a fresh buffer agrees with resize NEAREST for a translucent source")
    void blitScaledPremultipliedMatchesResizeNearest() {
        PixelBuffer render = PixelBuffer.create(6, 6);
        for (int y = 0; y < 6; y++)
            for (int x = 0; x < 6; x++)
                render.setPixel(x, y, ColorMath.blend(ColorMath.pack(20 + 40 * x, 0xFF, 8 * y, 0x80), ColorMath.TRANSPARENT, BlendMode.NORMAL));

        for (int size : new int[]{ 3, 6, 12 }) {
            PixelBuffer laid = PixelBuffer.create(size, size);
            laid.blitScaled(render, 0, 0, size, size, BlendMode.PREMULTIPLIED);
            assertArrayEquals(render.resize(size, size, Resample.NEAREST).data(), laid.data());
        }

        PixelBuffer darkened = PixelBuffer.create(6, 6);
        darkened.blitScaled(render, 0, 0, 6, 6);
        assertThat(darkened.getPixel(5, 0), not(equalTo(render.getPixel(5, 0))));
    }

    @Test
    @DisplayName("blitScaled lays down what a frozen per-pixel copy of its loop lays down")
    void blitScaledMatchesTheFrozenLoop() {
        int seed = 0;

        for (int[] blit : SCALED_BLITS) {
            int srcW = blit[0], srcH = blit[1], dw = blit[2], dh = blit[3], dx = blit[4], dy = blit[5], width = blit[6], height = blit[7];
            int[] drawn = texels(srcW, srcH, ++seed);
            int[] backdrop = texels(width, height, ++seed);

            for (boolean hasAlpha : new boolean[]{ true, false })
                for (BlendMode mode : BlendMode.values()) {
                    String label = "%dx%d -> %dx%d at (%d, %d) onto %dx%d, hasAlpha %s, %s"
                        .formatted(srcW, srcH, dw, dh, dx, dy, width, height, hasAlpha, mode);
                    PixelBuffer source = PixelBuffer.of(drawn.clone(), srcW, srcH, hasAlpha);

                    PixelBuffer expected = PixelBuffer.of(backdrop.clone(), width, height);
                    frozenBlitScaled(expected, source, dx, dy, dw, dh, mode);

                    PixelBuffer actual = PixelBuffer.of(backdrop.clone(), width, height);
                    actual.blitScaled(source, dx, dy, dw, dh, mode);

                    assertArrayEquals(expected.data(), actual.data(), label);
                    assertArrayEquals(drawn, source.data(), label + " left its source as drawn");
                }
        }
    }

    @Test
    @DisplayName("blitScaled of a buffer onto itself reads each texel after every write ahead of it in its row")
    void blitScaledOntoItselfMatchesTheFrozenLoop() {
        // Doubling from the origin, the first pixel of row 0 writes the texel the second pixel reads,
        // so that texel is held translucent to make the write change it in every mode but REPLACE.
        int[] drawn = texels(24, 20, 1);
        drawn[0] = 0x80C08040;

        for (BlendMode mode : BlendMode.values()) {
            PixelBuffer expected = PixelBuffer.of(drawn.clone(), 24, 20);
            frozenBlitScaled(expected, expected, 0, 0, 48, 40, mode);

            PixelBuffer actual = PixelBuffer.of(drawn.clone(), 24, 20);
            actual.blitScaled(actual, 0, 0, 48, 40, mode);

            assertArrayEquals(expected.data(), actual.data(), mode.name());
        }
    }

    @Test
    @DisplayName("blitScaled reads no blend mode over a source whose every texel has no alpha")
    void blitScaledReadsNoModeOverATransparentSource() {
        int[] drawn = texels(16, 16, 2);
        for (int i = 0; i < drawn.length; i++)
            drawn[i] &= 0x00FFFFFF;

        PixelBuffer source = PixelBuffer.of(drawn, 16, 16);
        int[] backdrop = texels(64, 64, 3);

        for (int size : new int[]{ 8, 16, 64 }) {
            PixelBuffer target = PixelBuffer.of(backdrop.clone(), 64, 64);
            target.blitScaled(source, 0, 0, size, size, null);
            assertArrayEquals(backdrop, target.data(), "%d to %d".formatted(16, size));
        }
    }

    @Test
    @DisplayName("blit PREMULTIPLIED opaque source matches generic blend result")
    void blitPremultipliedOpaqueFastPathMatchesGeneric() {
        PixelBuffer src = gradient(4, 4);
        PixelBuffer opaqueSrc = PixelBuffer.of(src.data().clone(), 4, 4, false);

        PixelBuffer fast = PixelBuffer.create(8, 8);
        fast.fill(0x80204060);
        fast.blit(opaqueSrc, 2, 2, BlendMode.PREMULTIPLIED);

        PixelBuffer generic = PixelBuffer.create(8, 8);
        generic.fill(0x80204060);
        generic.blit(src, 2, 2, BlendMode.PREMULTIPLIED);

        assertThat(fast, equalTo(generic));
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
    @DisplayName("resize BILINEAR keeps an opaque white picture opaque white at every size")
    void resizeBilinearKeepsOpaqueWhite() {
        // Four taps whose weights sum to one in float can land a hair under 255; rounding keeps them
        // there, where truncating would leave 254.
        for (int size = 2; size <= 12; size++) {
            PixelBuffer white = PixelBuffer.create(size, size);
            white.fill(0xFFFFFFFF);

            for (int target = 1; target <= 24; target++)
                for (int pixel : white.resize(target, target, Resample.BILINEAR).data())
                    assertThat("%d to %d".formatted(size, target), pixel, equalTo(0xFFFFFFFF));
        }
    }

    @Test
    @DisplayName("resize BILINEAR reads past an edge as the edge pixel")
    void resizeBilinearClampsToTheEdgePixel() {
        // A ramp over four columns, doubled: the outermost outputs sit a quarter pixel past the first
        // and last columns, so each is that column exactly rather than a mix shifted one inward.
        PixelBuffer ramp = PixelBuffer.create(4, 1);
        for (int x = 0; x < 4; x++)
            ramp.setPixel(x, 0, ColorMath.pack(255, 0x40 * x, 0, 0));

        PixelBuffer wide = ramp.resize(8, 1, Resample.BILINEAR);
        assertThat(ColorMath.red(wide.getPixel(0, 0)), equalTo(0x00));
        assertThat(ColorMath.red(wide.getPixel(7, 0)), equalTo(0xC0));
    }

    @Test
    @DisplayName("resize BILINEAR samples a one-pixel-wide source as its one column")
    void resizeBilinearAcceptsAOnePixelWideSource() {
        PixelBuffer column = PixelBuffer.create(1, 3);
        column.fill(0xFF336699);

        for (int pixel : column.resize(4, 6, Resample.BILINEAR).data())
            assertThat(pixel, equalTo(0xFF336699));
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
        PixelBuffer opaqueSrc = PixelBuffer.of(src.data().clone(), 4, 4, false);

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

    /**
     * Draws random texels in four even shares: no alpha over colour channels that still hold bits,
     * full alpha, and two shares of alpha drawn from 1 to 254.
     *
     * @param width the picture width
     * @param height the picture height
     * @param seed the random seed
     * @return the pixels
     */
    private static int[] texels(int width, int height, int seed) {
        Random random = new Random(seed);
        int[] pixels = new int[width * height];

        for (int i = 0; i < pixels.length; i++) {
            int rgb = random.nextInt() & 0x00FFFFFF;
            int alpha = switch (random.nextInt(4)) {
                case 0 -> 0;
                case 1 -> 0xFF;
                default -> 1 + random.nextInt(254);
            };
            pixels[i] = alpha << 24 | rgb;
        }

        return pixels;
    }

    /**
     * A frozen copy of the scaled blit as it stood computing each source column with its own integer
     * divide: the clip, the row and column products in int arithmetic, rows in order, every pixel
     * reading its texel afresh, alpha 0 skipped and every other texel blended.
     *
     * @param target the buffer composited onto
     * @param source the buffer composited
     * @param dx the destination x origin
     * @param dy the destination y origin
     * @param dw the destination width
     * @param dh the destination height
     * @param mode the blend mode
     */
    private static void frozenBlitScaled(PixelBuffer target, PixelBuffer source, int dx, int dy, int dw, int dh, BlendMode mode) {
        int[] data = target.data();
        int[] pixels = source.data();
        int x0 = Math.max(0, -dx);
        int y0 = Math.max(0, -dy);
        int x1 = Math.min(dw, target.width() - dx);
        int y1 = Math.min(dh, target.height() - dy);

        if (x0 >= x1 || y0 >= y1) return;

        for (int y = y0; y < y1; y++) {
            int dstRow = (dy + y) * target.width();
            int sy = (y * source.height()) / dh;

            for (int x = x0; x < x1; x++) {
                int sx = (x * source.width()) / dw;
                int src = pixels[sy * source.width() + sx];
                if (ColorMath.alpha(src) == 0) continue;
                int dstIdx = dstRow + dx + x;
                data[dstIdx] = ColorMath.blend(src, data[dstIdx], mode);
            }
        }
    }

}
