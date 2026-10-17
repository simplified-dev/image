package dev.simplified.image;

import dev.simplified.image.pixel.AlphaMode;
import dev.simplified.image.pixel.BlendMode;
import dev.simplified.image.pixel.ColorMath;
import dev.simplified.image.pixel.PixelBuffer;
import dev.simplified.image.pixel.Resample;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.awt.*;
import java.util.Arrays;
import java.util.Random;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.*;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

class PixelBufferTest {

    /**
     * Source width, source height, destination width, destination height, destination origin, target
     * width and target height of each scaled blit held to the frozen per-pixel loop: a 16x
     * magnification, a halving, magnifications by 100 / 16 and 13 / 7, a shrink by 4 / 9, a
     * one-to-one copy and a shrink by 2 / 5 from origins left of the target, an origin left of and
     * above the target that starts the blit part way through a magnified texel, a destination wider
     * than the target clipped on both sides, an origin that clips the right and bottom, a
     * magnification tall enough for its rows to run in parallel, and a destination so wide that each
     * column's product overflows an int and wraps.
     */
    private static final int[][] SCALED_BLITS = {
        { 16, 16, 256, 256, 0, 0, 256, 256 },
        { 512, 512, 256, 256, 0, 0, 256, 256 },
        { 16, 16, 100, 100, 0, 0, 100, 100 },
        { 7, 7, 13, 13, 0, 0, 13, 13 },
        { 9, 9, 4, 4, 1, 1, 6, 6 },
        { 13, 11, 13, 11, -2, -3, 20, 20 },
        { 30, 20, 12, 8, -5, 2, 6, 10 },
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
    @DisplayName("blitScaled lays down what a frozen per-pixel copy of its loop lays down, over random and transparent targets")
    void blitScaledMatchesTheFrozenLoop() {
        // Each geometry is drawn over random texels and over a fresh transparent target, the backdrop
        // most composites land on, so a run filled from what the target holds rather than from the
        // texel alone differs from the frozen loop.
        int seed = 0;

        for (int[] blit : SCALED_BLITS) {
            int srcW = blit[0], srcH = blit[1], dw = blit[2], dh = blit[3], dx = blit[4], dy = blit[5], width = blit[6], height = blit[7];
            int[] drawn = texels(srcW, srcH, ++seed);
            int[] random = texels(width, height, ++seed);

            for (int[] backdrop : new int[][]{ random, new int[width * height] })
                for (boolean hasAlpha : new boolean[]{ true, false })
                    for (BlendMode mode : BlendMode.values()) {
                        String label = "%dx%d -> %dx%d at (%d, %d) onto %dx%d %s, hasAlpha %s, %s"
                            .formatted(srcW, srcH, dw, dh, dx, dy, width, height, backdrop == random ? "random" : "transparent", hasAlpha, mode);
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
    @DisplayName("blitScaled and blitScaledFxaa return silently and leave the target as it was at a destination of no width or no height")
    void scaledBlitsOfNoSizeLeaveTheTarget() {
        // Origin and size of each composite: no width, a negative width, no width from an origin left
        // of the target, a negative width and height from an origin left of and above it, no height,
        // no height from an origin above it, and a negative height. Where the clipped columns run
        // backwards a column table sized from them could not be built.
        int[][] sizes = {
            { 0, 0, 0, 16 },
            { 0, 0, -5, 8 },
            { -4, 0, 0, 16 },
            { -4, -3, -6, -2 },
            { 0, 0, 16, 0 },
            { 3, -2, 16, 0 },
            { 0, 0, 16, -7 }
        };
        PixelBuffer source = PixelBuffer.of(texels(16, 16, 4), 16, 16);
        int[] backdrop = texels(32, 32, 5);

        for (int[] size : sizes) {
            int dx = size[0], dy = size[1], dw = size[2], dh = size[3];

            for (BlendMode mode : BlendMode.values()) {
                String label = "%dx%d at (%d, %d), %s".formatted(dw, dh, dx, dy, mode);
                PixelBuffer target = PixelBuffer.of(backdrop.clone(), 32, 32);

                assertDoesNotThrow(() -> target.blitScaled(source, dx, dy, dw, dh, mode), label);
                for (AlphaMode alphaMode : AlphaMode.values())
                    assertDoesNotThrow(() -> target.blitScaledFxaa(source, dx, dy, dw, dh, mode, alphaMode), label + ", " + alphaMode);

                assertArrayEquals(backdrop, target.data(), label);
            }
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
    @DisplayName("resize BILINEAR mixes the alpha bytes a buffer without alpha holds, as a frozen copy of its loop does")
    void resizeBilinearMixesTheAlphaABufferWithoutAlphaHolds() {
        // The texels hold alpha 0 over colour bits, partial alpha and full alpha. A resize that read
        // a buffer without alpha as opaque, as FXAA does, would write 0xFF wherever they mix.
        int seed = 40;

        for (int[] size : new int[][]{ { 7, 5, 16, 11 }, { 16, 16, 9, 6 }, { 1, 4, 3, 8 }, { 12, 70, 5, 66 } }) {
            int width = size[0], height = size[1], w = size[2], h = size[3];
            String label = "%dx%d to %dx%d".formatted(width, height, w, h);
            int[] drawn = texels(width, height, ++seed);

            PixelBuffer withoutAlpha = PixelBuffer.of(drawn.clone(), width, height, false);
            PixelBuffer resized = withoutAlpha.resize(w, h, Resample.BILINEAR);
            int[] expected = frozenResizeBilinear(drawn, width, height, w, h);

            assertArrayEquals(expected, resized.data(), label);
            assertThat(label + " carries no alpha", resized.hasAlpha(), is(false));
            assertThat(label + " holds alpha under 0xFF", Arrays.stream(expected).anyMatch(pixel -> ColorMath.alpha(pixel) < 0xFF), is(true));
        }
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
    @DisplayName("premultiply then unpremultiply takes a half-alpha grey back exactly")
    void premultiplyRoundtrip() {
        PixelBuffer a = PixelBuffer.create(1, 1);
        a.setPixel(0, 0, 0x80808080);
        a.premultiplyAlpha();
        assertThat(a.getPixel(0, 0), equalTo(0x80404040));
        a.unpremultiplyAlpha();
        assertThat(a.getPixel(0, 0), equalTo(0x80808080));
    }

    @Test
    @DisplayName("unpremultiply then premultiply returns every valid premultiplied pixel exactly")
    void unpremultiplyInvertsPremultiply() {
        // Row y holds alpha y + 1 and column x a channel of x % (alpha + 1), so every row walks each
        // channel value its alpha allows - every valid premultiplied pixel, 32,639 of them - with green
        // running down as red runs up and blue at half of red.
        int width = 255;
        int height = 254;
        int[] fixture = new int[width * height];

        for (int y = 0; y < height; y++) {
            int alpha = y + 1;
            for (int x = 0; x < width; x++) {
                int channel = x % (alpha + 1);
                fixture[y * width + x] = ColorMath.pack(alpha, channel, alpha - channel, channel >> 1);
            }
        }

        PixelBuffer buffer = PixelBuffer.of(fixture.clone(), width, height);
        buffer.unpremultiplyAlpha();
        buffer.premultiplyAlpha();
        assertArrayEquals(fixture, buffer.data());
    }

    @Test
    @DisplayName("premultiply then unpremultiply lands within half a premultiplied step of every straight pixel")
    void premultiplyRoundtripBound() {
        for (int alpha = 1; alpha < 255; alpha++) {
            int[] fixture = new int[256];
            for (int straight = 0; straight < 256; straight++)
                fixture[straight] = ColorMath.pack(alpha, straight, straight, 255 - straight);

            PixelBuffer buffer = PixelBuffer.of(fixture.clone(), 256, 1);
            buffer.premultiplyAlpha();
            buffer.unpremultiplyAlpha();

            // Premultiplying keeps alpha + 1 of the 256 straight levels, so a level comes back to the
            // nearest one kept: within half a step, ceil(255 / (2 * alpha)), and within 1 from 128 up.
            int bound = alpha >= 128 ? 1 : (255 + 2 * alpha - 1) / (2 * alpha);
            for (int straight = 0; straight < 256; straight++) {
                int pixel = buffer.data()[straight];
                String label = "alpha %d, straight %d".formatted(alpha, straight);
                assertThat(label, ColorMath.alpha(pixel), equalTo(alpha));
                assertThat(label, Math.abs(ColorMath.red(pixel) - straight), lessThanOrEqualTo(bound));
                assertThat(label, Math.abs(ColorMath.blue(pixel) - (255 - straight)), lessThanOrEqualTo(bound));
            }
        }
    }

    @Test
    @DisplayName("unpremultiply rounds each channel to nearest and leaves alpha 0 and 255 as they are")
    void unpremultiplyGoldens() {
        int[] fixture = { 0x00000000, 0x00123456, 0xFF123456, 0x01010101, 0x02010000, 0xFE7F3F00, 0x80404040, 0x80FF0000 };
        PixelBuffer buffer = PixelBuffer.of(fixture.clone(), fixture.length, 1);

        buffer.unpremultiplyAlpha();

        // 0x02010000 floors to 0x027F0000 and 0xFE7F3F00 to itself; 0x80FF0000 carries more red than
        // alpha, which no premultiplied pixel does, and is held at 255.
        assertArrayEquals(
            new int[] { 0x00000000, 0x00123456, 0xFF123456, 0x01FFFFFF, 0x02800000, 0xFE803F00, 0x80808080, 0x80FF0000 },
            buffer.data()
        );
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
     * Composites {@code source} onto {@code target} exactly as the scaled blit did when it computed
     * each source column with its own integer divide: the clip, the row and column products in int
     * arithmetic, rows in order, every pixel reading its texel afresh, alpha 0 skipped and every
     * other texel blended.
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

    /**
     * Resizes {@code data} bilinearly exactly as resize did while its sampler read the buffer's own
     * size: each output pixel's centre mapped into the source, the four taps around it clamped to the
     * edge, every channel - alpha included - mixed as it is held whatever the buffer says its alpha
     * carries, and each channel rounded once.
     *
     * @param data the pixels to resize, left unmodified
     * @param width the source width
     * @param height the source height
     * @param w the resized width
     * @param h the resized height
     * @return the resized pixels
     */
    private static int[] frozenResizeBilinear(int[] data, int width, int height, int w, int h) {
        int[] result = new int[w * h];
        float[] sample = new float[4];

        for (int y = 0; y < h; y++) {
            float srcY = ((y + 0.5f) * height) / h - 0.5f;

            for (int x = 0; x < w; x++) {
                float srcX = ((x + 0.5f) * width) / w - 0.5f;
                int floorX = (int) Math.floor(srcX);
                int floorY = (int) Math.floor(srcY);
                float fx = srcX - floorX;
                float fy = srcY - floorY;

                int x0 = Math.clamp(floorX, 0, width - 1);
                int x1 = Math.clamp(floorX + 1, 0, width - 1);
                int row0 = Math.clamp(floorY, 0, height - 1) * width;
                int row1 = Math.clamp(floorY + 1, 0, height - 1) * width;

                Arrays.fill(sample, 0f);
                accumulateTap(data[row0 + x0], (1 - fx) * (1 - fy), sample);
                accumulateTap(data[row0 + x1], fx * (1 - fy), sample);
                accumulateTap(data[row1 + x0], (1 - fx) * fy, sample);
                accumulateTap(data[row1 + x1], fx * fy, sample);

                result[y * w + x] = ColorMath.pack(channel(sample[0]), channel(sample[1]), channel(sample[2]), channel(sample[3]));
            }
        }

        return result;
    }

    /**
     * Adds one bilinear tap's share of every channel, each as it is held, to a sample being
     * accumulated.
     *
     * @param argb the tap's packed ARGB pixel
     * @param weight the tap's bilinear weight
     * @param sample the array accumulating alpha, red, green and blue
     */
    private static void accumulateTap(int argb, float weight, float[] sample) {
        sample[0] += weight * ColorMath.alpha(argb);
        sample[1] += weight * ColorMath.red(argb);
        sample[2] += weight * ColorMath.green(argb);
        sample[3] += weight * ColorMath.blue(argb);
    }

    /**
     * Rounds a channel value half up and holds it in {@code [0, 255]}.
     *
     * @param value the channel value
     * @return the rounded channel
     */
    private static int channel(float value) {
        return Math.clamp(Math.round(value), 0, 255);
    }

}
