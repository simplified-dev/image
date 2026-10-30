package dev.simplified.image;

import dev.simplified.image.pixel.AlphaMode;
import dev.simplified.image.pixel.BlendMode;
import dev.simplified.image.pixel.ColorMath;
import dev.simplified.image.pixel.Resample;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.ComponentColorModel;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferInt;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
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
        return PixelBuffer.ofStraight(data, w, h);
    }

    /**
     * Wraps pixels in a buffer holding the given convention. A premultiplied buffer whose alpha carries
     * no data is the straight one converted, which shares its array.
     *
     * @param alphaMode the convention the buffer holds
     * @param pixels the pixels, not copied
     * @param width the buffer width
     * @param height the buffer height
     * @param hasAlpha whether the alpha channel carries meaningful data
     * @return the buffer
     */
    private static PixelBuffer held(AlphaMode alphaMode, int[] pixels, int width, int height, boolean hasAlpha) {
        return switch (alphaMode) {
            case STRAIGHT -> PixelBuffer.ofStraight(pixels, width, height, hasAlpha);
            case PREMULTIPLIED -> hasAlpha
                ? PixelBuffer.ofPremultiplied(pixels, width, height)
                : PixelBuffer.ofStraight(pixels, width, height, false).toAlphaMode(AlphaMode.PREMULTIPLIED);
        };
    }

    /**
     * Answers a buffer of the given convention whose every pixel holds {@code argb}, written as it is.
     *
     * @param alphaMode the convention the buffer holds
     * @param width the buffer width
     * @param height the buffer height
     * @param argb the pixel, in that convention
     * @return the buffer
     */
    private static PixelBuffer filled(AlphaMode alphaMode, int width, int height, int argb) {
        int[] pixels = new int[width * height];
        Arrays.fill(pixels, argb);
        return held(alphaMode, pixels, width, height, true);
    }

    @Test
    @DisplayName("getPixel / setPixel roundtrip")
    void pixelRoundtrip() {
        PixelBuffer buffer = PixelBuffer.create(4, 4);
        buffer.setPixel(2, 3, 0xDEADBEEF);
        assertThat(buffer.getPixel(2, 3), equalTo(0xDEADBEEF));
    }

    @Test
    @DisplayName("create refuses dimensions whose pixel count overflows an int")
    void createRefusesAnOverflowingPixelCount() {
        // 65535 * 65535 wraps to a negative count, 65536 * 65537 to the positive 65536.
        assertThrows(ArithmeticException.class, () -> PixelBuffer.create(65535, 65535));
        assertThrows(ArithmeticException.class, () -> PixelBuffer.create(65536, 65537));
    }

    @Test
    @DisplayName("ofStraight and ofPremultiplied refuse an array whose length matches the dimensions only once their product wraps")
    void ofRefusesALengthMatchingAWrappedPixelCount() {
        assertThrows(IllegalArgumentException.class, () -> PixelBuffer.ofStraight(new int[65536], 65536, 65537));
        assertThrows(IllegalArgumentException.class, () -> PixelBuffer.ofStraight(new int[65536], 65536, 65537, false));
        assertThrows(IllegalArgumentException.class, () -> PixelBuffer.ofPremultiplied(new int[65536], 65536, 65537));
    }

    @Test
    @DisplayName("a buffer holds the convention it was made in")
    void aBufferHoldsTheConventionItWasMadeIn() {
        int[] pixels = { 0x80800000, 0xFF336699 };

        PixelBuffer created = PixelBuffer.create(3, 2);
        assertThat(created.alphaMode(), is(AlphaMode.STRAIGHT));
        assertThat(created.hasAlpha(), is(true));
        assertArrayEquals(new int[6], created.data());

        PixelBuffer canvas = PixelBuffer.canvas(3, 2);
        assertThat(canvas.alphaMode(), is(AlphaMode.PREMULTIPLIED));
        assertThat(canvas.hasAlpha(), is(true));
        assertArrayEquals(new int[6], canvas.data());
        assertThrows(ArithmeticException.class, () -> PixelBuffer.canvas(65535, 65535));

        PixelBuffer straight = PixelBuffer.ofStraight(pixels, 2, 1);
        assertThat(straight.alphaMode(), is(AlphaMode.STRAIGHT));
        assertThat(straight.hasAlpha(), is(true));
        assertThat(straight.data(), is(sameInstance(pixels)));

        PixelBuffer opaque = PixelBuffer.ofStraight(pixels, 2, 1, false);
        assertThat(opaque.alphaMode(), is(AlphaMode.STRAIGHT));
        assertThat(opaque.hasAlpha(), is(false));
        assertThat(opaque.data(), is(sameInstance(pixels)));

        PixelBuffer premultiplied = PixelBuffer.ofPremultiplied(pixels, 2, 1);
        assertThat(premultiplied.alphaMode(), is(AlphaMode.PREMULTIPLIED));
        assertThat(premultiplied.hasAlpha(), is(true));
        assertThat(premultiplied.data(), is(sameInstance(pixels)));
    }

    @Test
    @DisplayName("wrap answers straight colour for every layout holding it, the gray and alpha one included")
    void wrapAnswersStraightColourForEveryStraightLayout() {
        // The gray and alpha image is TYPE_CUSTOM, which wrap reads by its colour model, and its alpha
        // carries data. TYPE_USHORT_565_RGB has no unpacker of its own and is drawn into an ARGB image.
        ComponentColorModel grayAndAlpha = new ComponentColorModel(
            ColorSpace.getInstance(ColorSpace.CS_GRAY), true, false, Transparency.TRANSLUCENT, DataBuffer.TYPE_BYTE
        );
        BufferedImage grayAndAlphaImage = new BufferedImage(grayAndAlpha, grayAndAlpha.createCompatibleWritableRaster(2, 2), false, null);
        assertThat(grayAndAlphaImage.getType(), is(BufferedImage.TYPE_CUSTOM));
        assertThat(PixelBuffer.wrap(grayAndAlphaImage).hasAlpha(), is(true));

        String[] layouts = { "TYPE_INT_RGB", "TYPE_INT_BGR", "TYPE_BYTE_GRAY", "gray and alpha", "TYPE_BYTE_BINARY", "TYPE_USHORT_565_RGB" };
        BufferedImage[] images = {
            new BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB),
            new BufferedImage(2, 2, BufferedImage.TYPE_INT_BGR),
            new BufferedImage(2, 2, BufferedImage.TYPE_BYTE_GRAY),
            grayAndAlphaImage,
            new BufferedImage(2, 2, BufferedImage.TYPE_BYTE_BINARY),
            new BufferedImage(2, 2, BufferedImage.TYPE_USHORT_565_RGB)
        };

        for (int i = 0; i < images.length; i++)
            assertThat(layouts[i], PixelBuffer.wrap(images[i]).alphaMode(), is(AlphaMode.STRAIGHT));
    }

    @Test
    @DisplayName("a buffer derived from another holds its convention")
    void derivedBuffersKeepTheConvention() {
        // A translucent square inside a transparent border, so trim crops, and an opaque picture, so
        // trim copies.
        int[] bordered = new int[6 * 5];
        for (int y = 1; y < 4; y++)
            for (int x = 1; x < 5; x++)
                bordered[y * 6 + x] = ColorMath.pack(0x80 + x, 0x40, y * 0x10, 0x20);

        for (AlphaMode mode : AlphaMode.values()) {
            AlphaMode other = mode == AlphaMode.STRAIGHT ? AlphaMode.PREMULTIPLIED : AlphaMode.STRAIGHT;
            PixelBuffer buffer = held(mode, bordered.clone(), 6, 5, true);
            PixelBuffer opaque = held(mode, gradient(6, 5).data(), 6, 5, true);
            PixelBuffer withoutAlpha = held(mode, gradient(6, 5).data(), 6, 5, false);

            assertThat(mode + " crop", buffer.crop(1, 1, 3, 2).alphaMode(), is(mode));
            assertThat(mode + " copy", buffer.copy().alphaMode(), is(mode));
            assertThat(mode + " rotate90", buffer.rotate90().alphaMode(), is(mode));
            assertThat(mode + " rotate180", buffer.rotate180().alphaMode(), is(mode));
            assertThat(mode + " rotate270", buffer.rotate270().alphaMode(), is(mode));
            assertThat(mode + " trim cropping", buffer.trim().width(), is(4));
            assertThat(mode + " trim cropping", buffer.trim().alphaMode(), is(mode));
            assertThat(mode + " trim copying", opaque.trim().alphaMode(), is(mode));
            assertThat(mode + " tint", buffer.tint(0xFF80C0E0).alphaMode(), is(mode));
            assertThat(mode + " lerp", PixelBuffer.lerp(buffer, held(other, bordered.clone(), 6, 5, true), 0.5f).alphaMode(), is(mode));

            for (Resample filter : Resample.values()) {
                assertThat(mode + " resize " + filter, buffer.resize(4, 7, filter).alphaMode(), is(mode));
                assertThat(mode + " resize " + filter + " without alpha", withoutAlpha.resize(4, 7, filter).alphaMode(), is(mode));
            }
        }
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
    @DisplayName("fillRect NORMAL lays F0100010 onto a clear pixel as F00F000F, and 505000FF over that as F523005A")
    void fillRectNormalBlendsTheColour() {
        PixelBuffer buffer = PixelBuffer.create(3, 2);
        buffer.fillRect(0, 0, 3, 2, 0xF0100010, BlendMode.NORMAL);
        for (int pixel : buffer.data())
            assertThat(Integer.toHexString(pixel), pixel, equalTo(0xF00F000F));

        buffer.fillRect(1, 0, 2, 1, 0x505000FF, BlendMode.NORMAL);
        assertArrayEquals(
            new int[]{ 0xF00F000F, 0xF523005A, 0xF523005A, 0xF00F000F, 0xF00F000F, 0xF00F000F },
            buffer.data()
        );
    }

    @Test
    @DisplayName("fillRect with a blend lays a colour of alpha 0 nowhere under every mode but REPLACE")
    void fillRectOfNoAlphaLaysNothing() {
        int[] backdrop = texels(9, 70, 6);

        for (BlendMode mode : BlendMode.values()) {
            if (mode == BlendMode.REPLACE) continue;

            for (int colour : new int[]{ 0x00000000, 0x00FFFFFF, 0x00123456 }) {
                PixelBuffer buffer = PixelBuffer.ofStraight(backdrop.clone(), 9, 70);
                buffer.fillRect(-1, -1, 11, 72, colour, mode);
                assertArrayEquals(backdrop, buffer.data(), "%08X, %s".formatted(colour, mode));
            }
        }
    }

    @Test
    @DisplayName("fillRect with a blend covers exactly the pixels the overwriting fillRect covers")
    void fillRectWithABlendClipsAsFillRectClips() {
        // A 2x2 rectangle from (-1, -1) lays only (0, 0), and one overhanging the right and bottom
        // edges only the pixels inside them.
        PixelBuffer corner = PixelBuffer.create(4, 3);
        corner.fillRect(-1, -1, 2, 2, 0xF0100010, BlendMode.NORMAL);
        assertArrayEquals(new int[]{ 0xF00F000F, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, corner.data());

        PixelBuffer overhang = PixelBuffer.create(4, 3);
        overhang.fillRect(2, 1, 5, 5, 0xF0100010, BlendMode.NORMAL);
        assertArrayEquals(new int[]{ 0, 0, 0, 0, 0, 0, 0xF00F000F, 0xF00F000F, 0, 0, 0xF00F000F, 0xF00F000F }, overhang.data());

        // Origin and size of each rectangle: left of and above the buffer, overhanging the right and
        // bottom, overhanging every side, wholly inside, no width, a negative width, no height, a
        // negative height from an origin above the buffer, and wholly outside on each side.
        int[][] rects = {
            { -1, -1, 3, 3 },
            { 4, 3, 10, 10 },
            { -2, -3, 20, 20 },
            { 1, 2, 3, 4 },
            { 2, 2, 0, 4 },
            { 2, 2, -3, 4 },
            { 2, 2, 4, 0 },
            { 2, -1, 4, -2 },
            { -5, 0, 3, 2 },
            { 0, -9, 3, 2 },
            { 8, 0, 3, 2 },
            { 0, 7, 3, 2 }
        };
        int[] backdrop = texels(8, 7, 7);

        for (int[] rect : rects) {
            int x = rect[0], y = rect[1], w = rect[2], h = rect[3];

            for (int colour : new int[]{ 0xF0100010, 0x505000FF, 0xFF336699 }) {
                String label = "%dx%d at (%d, %d), %08X".formatted(w, h, x, y, colour);
                PixelBuffer covered = PixelBuffer.create(8, 7);
                covered.fillRect(x, y, w, h, 0xFFFFFFFF);

                int[] expected = backdrop.clone();
                for (int i = 0; i < expected.length; i++)
                    if (covered.data()[i] != 0)
                        expected[i] = ColorMath.blend(colour, backdrop[i], BlendMode.NORMAL);

                PixelBuffer buffer = PixelBuffer.ofStraight(backdrop.clone(), 8, 7);
                buffer.fillRect(x, y, w, h, colour, BlendMode.NORMAL);
                assertArrayEquals(expected, buffer.data(), label);
            }
        }
    }

    @Test
    @DisplayName("fillRect REPLACE lays what the overwriting fillRect lays, a colour of alpha 0 included")
    void fillRectReplaceIsTheOverwritingFillRect() {
        int[] backdrop = texels(9, 70, 8);

        for (int[] rect : new int[][]{ { -2, -3, 6, 80 }, { 3, 1, 4, 5 }, { 7, 60, 9, 20 }, { 2, 2, 0, 5 } })
            for (int colour : new int[]{ 0x00000000, 0x00123456, 0x01FFFFFF, 0x80808080, 0xF0100010, 0xFF336699 }) {
                int x = rect[0], y = rect[1], w = rect[2], h = rect[3];
                String label = "%dx%d at (%d, %d), %08X".formatted(w, h, x, y, colour);

                PixelBuffer overwritten = PixelBuffer.ofStraight(backdrop.clone(), 9, 70);
                overwritten.fillRect(x, y, w, h, colour);

                PixelBuffer replaced = PixelBuffer.ofStraight(backdrop.clone(), 9, 70);
                replaced.fillRect(x, y, w, h, colour, BlendMode.REPLACE);

                assertArrayEquals(overwritten.data(), replaced.data(), label);
            }
    }

    @Test
    @DisplayName("fillRect with a blend lays what a blit of a one-pixel image of its colour lays at every pixel it covers")
    void fillRectWithABlendLaysWhatABlitOfOnePixelLays() {
        // The backdrop holds no alpha over colour bits, full alpha and partial alpha, and is tall
        // enough for the fill's rows to run in parallel. Under REPLACE a blit discards a source pixel
        // of alpha 0, which the fill writes as the overwriting fillRect does, so that one pairing is
        // held by the REPLACE test instead.
        int width = 9, height = 70;
        int x = -2, y = 3, w = 8, h = 70;
        int[] backdrop = texels(width, height, 9);

        for (BlendMode mode : BlendMode.values())
            for (int colour : new int[]{ 0x00000000, 0x00FFFFFF, 0x01FFFFFF, 0x505000FF, 0x80808080, 0xF0100010, 0xFEFFFFFF, 0xFF336699 }) {
                if (mode == BlendMode.REPLACE && ColorMath.alpha(colour) == 0) continue;
                String label = "%08X, %s".formatted(colour, mode);

                PixelBuffer blitted = PixelBuffer.ofStraight(backdrop.clone(), width, height);
                for (int py = y; py < y + h; py++)
                    for (int px = x; px < x + w; px++)
                        blitted.blit(PixelBuffer.ofStraight(new int[]{ colour }, 1, 1), px, py, mode);

                PixelBuffer filled = PixelBuffer.ofStraight(backdrop.clone(), width, height);
                filled.fillRect(x, y, w, h, colour, mode);

                assertArrayEquals(blitted.data(), filled.data(), label);
            }
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
    @DisplayName("resize NEAREST takes the top-left source pixel of each output pixel's footprint")
    void resizeNearestTakesTheTopLeftOfEachFootprint() {
        PixelBuffer row = PixelBuffer.create(9, 1);
        for (int x = 0; x < 9; x++)
            row.setPixel(x, 0, 0xFF000000 | x);

        // A 3x downscale reads source 0, 3 and 6, where the centre-closest pixels are 1, 4 and 7.
        assertArrayEquals(new int[]{ 0xFF000000, 0xFF000003, 0xFF000006 }, row.resize(3, 1, Resample.NEAREST).data());
        // Upscaling 2 to 3 reads source 0, 0 and 1: each output column rounds down onto the source.
        assertArrayEquals(new int[]{ 0xFF000000, 0xFF000000, 0xFF000001 }, row.crop(0, 0, 2, 1).resize(3, 1, Resample.NEAREST).data());
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
                        PixelBuffer source = PixelBuffer.ofStraight(drawn.clone(), srcW, srcH, hasAlpha);

                        PixelBuffer expected = PixelBuffer.ofStraight(backdrop.clone(), width, height);
                        frozenBlitScaled(expected, source, dx, dy, dw, dh, mode);

                        PixelBuffer actual = PixelBuffer.ofStraight(backdrop.clone(), width, height);
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
            PixelBuffer expected = PixelBuffer.ofStraight(drawn.clone(), 24, 20);
            frozenBlitScaled(expected, expected, 0, 0, 48, 40, mode);

            PixelBuffer actual = PixelBuffer.ofStraight(drawn.clone(), 24, 20);
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

        PixelBuffer source = PixelBuffer.ofStraight(drawn, 16, 16);
        int[] backdrop = texels(64, 64, 3);

        for (int size : new int[]{ 8, 16, 64 }) {
            PixelBuffer target = PixelBuffer.ofStraight(backdrop.clone(), 64, 64);
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
        int[] drawn = texels(16, 16, 4);
        PixelBuffer[] sources = { PixelBuffer.ofStraight(drawn, 16, 16), PixelBuffer.ofPremultiplied(drawn, 16, 16) };
        int[] backdrop = texels(32, 32, 5);

        for (int[] size : sizes) {
            int dx = size[0], dy = size[1], dw = size[2], dh = size[3];

            for (BlendMode mode : BlendMode.values()) {
                String label = "%dx%d at (%d, %d), %s".formatted(dw, dh, dx, dy, mode);
                PixelBuffer target = PixelBuffer.ofStraight(backdrop.clone(), 32, 32);

                assertDoesNotThrow(() -> target.blitScaled(sources[0], dx, dy, dw, dh, mode), label);
                for (PixelBuffer source : sources)
                    assertDoesNotThrow(() -> target.blitScaledFxaa(source, dx, dy, dw, dh, mode), label + ", " + source.alphaMode());

                assertArrayEquals(backdrop, target.data(), label);
            }
        }
    }

    @Test
    @DisplayName("blit PREMULTIPLIED opaque source matches generic blend result")
    void blitPremultipliedOpaqueFastPathMatchesGeneric() {
        PixelBuffer src = gradient(4, 4);
        PixelBuffer opaqueSrc = PixelBuffer.ofStraight(src.data().clone(), 4, 4, false);

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
    @DisplayName("resize BILINEAR keeps a straight pixel's colour beside a transparent one, and mixes premultiplied pixels as held")
    void resizeBilinearHonoursTheConvention() {
        PixelBuffer straight = PixelBuffer.create(2, 1);
        straight.setPixel(0, 0, 0xFFFF0000);
        PixelBuffer premultiplied = PixelBuffer.canvas(2, 1);
        premultiplied.setPixel(0, 0, 0xFFFF0000);

        assertThat(straight.resize(1, 1, Resample.BILINEAR).getPixel(0, 0), equalTo(0x80FF0000));
        assertThat(premultiplied.resize(1, 1, Resample.BILINEAR).getPixel(0, 0), equalTo(0x80800000));

        // A partly transparent premultiplied pixel carries its alpha in its colour already; weighing it
        // by its alpha again would leave half the red, 0x10.
        PixelBuffer translucent = PixelBuffer.canvas(2, 1);
        translucent.setPixel(0, 0, 0x80400000);
        assertThat(translucent.resize(1, 1, Resample.BILINEAR).getPixel(0, 0), equalTo(0x40200000));
    }

    @Test
    @DisplayName("resize BICUBIC of premultiplied pixels holds every colour at or under its alpha, and of straight pixels keeps the colour")
    void resizeBicubicPremultipliedStaysPremultiplied() {
        PixelBuffer premultiplied = PixelBuffer.canvas(8, 4);
        PixelBuffer straight = PixelBuffer.create(8, 4);
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++) {
                premultiplied.setPixel(x, y, 0xFFFF0000);
                straight.setPixel(x, y, 0xFFFF0000);
            }

        int partial = 0;
        for (int pixel : premultiplied.resize(3, 2, Resample.BICUBIC).data()) {
            assertThat(Integer.toHexString(pixel), ColorMath.red(pixel) <= ColorMath.alpha(pixel), equalTo(true));
            if (ColorMath.alpha(pixel) > 0 && ColorMath.alpha(pixel) < 0xF0) partial++;
        }
        assertThat("the premultiplied resize mixed an edge", partial, greaterThan(0));

        for (int pixel : straight.resize(3, 2, Resample.BICUBIC).data())
            if (ColorMath.alpha(pixel) > 0)
                assertThat(Integer.toHexString(pixel), ColorMath.red(pixel), greaterThanOrEqualTo(0xF0));
    }

    @Test
    @DisplayName("resize BICUBIC keeps an opaque buffer opaque")
    void resizeBicubicKeepsOpaque() {
        for (AlphaMode mode : AlphaMode.values()) {
            PixelBuffer resized = held(mode, gradient(8, 8).data(), 8, 8, false).resize(5, 3, Resample.BICUBIC);
            assertThat(resized.hasAlpha(), equalTo(false));
            for (int pixel : resized.data())
                assertThat(ColorMath.alpha(pixel), equalTo(0xFF));
        }
    }

    @Test
    @DisplayName("resize AREA averages every source pixel a destination pixel covers")
    void resizeAreaAveragesTheFootprint() {
        PixelBuffer quad = PixelBuffer.ofStraight(new int[]{ 0xFFFF0000, 0xFF00FF00, 0xFF0000FF, 0xFFFFFFFF }, 2, 2, false);
        assertThat(quad.resize(1, 1, Resample.AREA).getPixel(0, 0), equalTo(0xFF808080));

        int[] lone = { 0xFFFF0000, 0, 0, 0 };
        assertThat(PixelBuffer.ofStraight(lone.clone(), 2, 2).resize(1, 1, Resample.AREA).getPixel(0, 0), equalTo(0x40FF0000));
        assertThat(PixelBuffer.ofPremultiplied(lone.clone(), 2, 2).resize(1, 1, Resample.AREA).getPixel(0, 0), equalTo(0x40400000));

        // A partly transparent premultiplied pixel carries its alpha in its colour already; weighing it
        // by its alpha again would leave half the red, 0x08.
        int[] translucent = { 0x80400000, 0, 0, 0 };
        assertThat(PixelBuffer.ofPremultiplied(translucent, 2, 2).resize(1, 1, Resample.AREA).getPixel(0, 0), equalTo(0x20100000));
    }

    @Test
    @DisplayName("resize AREA weights a pixel a destination pixel only partly covers by the share it covers")
    void resizeAreaWeightsPartialCoverage() {
        PixelBuffer row = PixelBuffer.ofStraight(new int[]{ 0xFF000000, 0xFF780000, 0xFFF00000 }, 3, 1, false);
        PixelBuffer shrunk = row.resize(2, 1, Resample.AREA);

        assertThat(ColorMath.red(shrunk.getPixel(0, 0)), equalTo(40));
        assertThat(ColorMath.red(shrunk.getPixel(1, 0)), equalTo(200));
    }

    @Test
    @DisplayName("resize AREA reads a growing axis as NEAREST does")
    void resizeAreaMagnifiesAsNearest() {
        PixelBuffer source = gradient(5, 3);
        assertArrayEquals(source.resize(12, 7, Resample.NEAREST).data(), source.resize(12, 7, Resample.AREA).data());
    }

    @Test
    @DisplayName("blitScaled with a filter lays down what resizing with it and then blitting lays down, in either convention")
    void blitScaledWithAFilter() {
        int[] drawn = new int[256];
        for (int i = 0; i < 256; i++)
            drawn[i] = ((i * 37) & 0xFF) << 24 | (i * 0x010203 & 0xFFFFFF);
        PixelBuffer straight = PixelBuffer.ofStraight(drawn, 16, 16);

        for (PixelBuffer hiRes : new PixelBuffer[]{ straight, straight.toAlphaMode(AlphaMode.PREMULTIPLIED) }) {
            AlphaMode alphaMode = hiRes.alphaMode();
            BlendMode mode = alphaMode == AlphaMode.PREMULTIPLIED ? BlendMode.PREMULTIPLIED : BlendMode.STRAIGHT;

            PixelBuffer viaBlit = filled(alphaMode, 6, 6, 0x80204060);
            viaBlit.blitScaled(hiRes, 1, 1, 4, 4, mode, Resample.AREA);

            PixelBuffer viaResize = filled(alphaMode, 6, 6, 0x80204060);
            viaResize.blit(hiRes.resize(4, 4, Resample.AREA), 1, 1, mode);
            assertArrayEquals(viaResize.data(), viaBlit.data(), alphaMode.name());

            PixelBuffer nearest = filled(alphaMode, 6, 6, 0);
            nearest.blitScaled(hiRes, 1, 1, 4, 4, mode, Resample.NEAREST);
            PixelBuffer unfiltered = filled(alphaMode, 6, 6, 0);
            unfiltered.blitScaled(hiRes, 1, 1, 4, 4, mode);
            assertArrayEquals(unfiltered.data(), nearest.data(), alphaMode.name());
        }

        // The area average weighs straight colour by alpha and mixes premultiplied colour as held, so
        // the source's convention decides the resampled pixels.
        PixelBuffer fromStraight = PixelBuffer.create(4, 4);
        fromStraight.blitScaled(straight, 0, 0, 4, 4, BlendMode.REPLACE, Resample.AREA);
        PixelBuffer fromPremultiplied = PixelBuffer.create(4, 4);
        fromPremultiplied.blitScaled(straight.toAlphaMode(AlphaMode.PREMULTIPLIED), 0, 0, 4, 4, BlendMode.REPLACE, Resample.AREA);
        assertArrayEquals(straight.resize(4, 4, Resample.AREA).data(), fromStraight.data());
        assertArrayEquals(straight.toAlphaMode(AlphaMode.PREMULTIPLIED).resize(4, 4, Resample.AREA).data(), fromPremultiplied.data());
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

            PixelBuffer withoutAlpha = PixelBuffer.ofStraight(drawn.clone(), width, height, false);
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
    @DisplayName("invert of a premultiplied buffer takes each channel from its alpha, the premultiplied form of the inverted colour")
    void invertOfAPremultipliedBuffer() {
        // F00F000F is straight F0100010 premultiplied; inverted, that is F0EFFFEF, which premultiplies
        // to F0E1F0E1. 80FF0000 and 4000FF90 hold channels above their alpha, which no premultiplied
        // pixel holds, and each of those inverts to 0 rather than below it.
        PixelBuffer premultiplied = PixelBuffer.ofPremultiplied(new int[]{ 0xF00F000F, 0xFF336699, 0x00000000, 0x80800000, 0x80FF0000, 0x4000FF90 }, 6, 1);
        premultiplied.invert();
        assertArrayEquals(new int[]{ 0xF0E1F0E1, 0xFFCC9966, 0x00000000, 0x80008080, 0x80008080, 0x40400000 }, premultiplied.data());
        assertThat(PixelBuffer.ofStraight(new int[]{ 0xF0EFFFEF }, 1, 1).toAlphaMode(AlphaMode.PREMULTIPLIED).getPixel(0, 0), equalTo(0xF0E1F0E1));

        // A buffer whose alpha carries no data inverts as opaque colour, whatever its alpha bytes hold.
        PixelBuffer withoutAlpha = held(AlphaMode.PREMULTIPLIED, new int[]{ 0x00112233 }, 1, 1, false);
        withoutAlpha.invert();
        assertThat(withoutAlpha.getPixel(0, 0), equalTo(0x00EEDDCC));
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
    @DisplayName("tint multiplies RGB channels and preserves alpha")
    void tintPreservesAlpha() {
        PixelBuffer source = PixelBuffer.ofStraight(new int[]{ 0x80FFFFFF }, 1, 1);
        PixelBuffer tinted = source.tint(0xFFFF0000);
        assertThat(ColorMath.alpha(tinted.getPixel(0, 0)), equalTo(0x80));
        assertThat(ColorMath.red(tinted.getPixel(0, 0)), equalTo(0xFF));
        assertThat(ColorMath.green(tinted.getPixel(0, 0)), equalTo(0x00));
        assertThat(ColorMath.blue(tinted.getPixel(0, 0)), equalTo(0x00));
    }

    @Test
    @DisplayName("tint SIMD prefix and scalar tail produce bit-identical per-pixel output")
    void tintSimdScalarParity() {
        // SPECIES_PREFERRED is typically 4-8 int lanes. A 1-pixel buffer exercises only the
        // scalar fallback (loopBound(1) = 0). A 64-pixel buffer exercises the SIMD prefix for
        // most/all pixels. Both should produce identical per-pixel output - this catches drift
        // between scalar and SIMD rounding conventions.
        int sourcePixel = 0xC0C8C8C8; // alpha=192, RGB=200; mid-range to expose rounding
        int tint = 0xFFC8C8C8;        // tint=200 per channel

        PixelBuffer scalarOnly = PixelBuffer.ofStraight(new int[]{ sourcePixel }, 1, 1);
        PixelBuffer scalarTinted = scalarOnly.tint(tint);
        int scalarResult = scalarTinted.getPixel(0, 0);

        int[] bulk = new int[64];
        Arrays.fill(bulk, sourcePixel);
        PixelBuffer simdInput = PixelBuffer.ofStraight(bulk, 64, 1);
        PixelBuffer simdTinted = simdInput.tint(tint);
        for (int i = 0; i < 64; i++)
            assertThat("pixel " + i + " (SIMD lane vs scalar)",
                simdTinted.getPixel(i, 0), equalTo(scalarResult));
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
    @DisplayName("multiplyAlpha scales a premultiplied pixel's colour with its alpha, and a straight pixel's alpha alone")
    void multiplyAlphaKeepsAPremultipliedBufferPremultiplied() {
        PixelBuffer straight = PixelBuffer.ofStraight(new int[]{ 0xF0100010 }, 1, 1);
        straight.multiplyAlpha(0.5f);
        assertThat(straight.getPixel(0, 0), equalTo(0x78100010));

        // Every premultiplied pixel at every alpha, each channel up to its alpha, at a factor whose
        // products fall on and between halves: each channel is rounded once and none passes its alpha.
        int[] pixels = new int[256 * 256];
        for (int i = 0; i < pixels.length; i++) {
            int a = i >>> 8;
            int c = Math.min(i & 0xFF, a);
            pixels[i] = ColorMath.pack(a, c, a - c, c >>> 1);
        }

        for (float factor : new float[]{ 0.5f, 0.3f, 0.999f, 0f }) {
            PixelBuffer premultiplied = PixelBuffer.ofPremultiplied(pixels.clone(), 256, 256);
            premultiplied.multiplyAlpha(factor);

            for (int i = 0; i < pixels.length; i++) {
                int expected = ColorMath.pack(
                    Math.round(ColorMath.alpha(pixels[i]) * factor),
                    Math.round(ColorMath.red(pixels[i]) * factor),
                    Math.round(ColorMath.green(pixels[i]) * factor),
                    Math.round(ColorMath.blue(pixels[i]) * factor)
                );
                int actual = premultiplied.data()[i];
                String label = "%08X at %s".formatted(pixels[i], factor);
                assertThat(label, actual, equalTo(expected));
                assertThat(label, Math.max(ColorMath.red(actual), Math.max(ColorMath.green(actual), ColorMath.blue(actual))), lessThanOrEqualTo(ColorMath.alpha(actual)));
            }
        }

        PixelBuffer tooltip = PixelBuffer.ofPremultiplied(new int[]{ 0xF00F000F }, 1, 1);
        tooltip.multiplyAlpha(0.5f);
        assertThat(tooltip.getPixel(0, 0), equalTo(0x78080008));

        // A buffer whose alpha carries no data is scaled as its convention says and keeps reporting no
        // alpha: premultiplied colour darkens with the alpha byte, so the array is a valid premultiplied
        // pixel to wrap again, and straight colour stays.
        PixelBuffer opaquePremultiplied = held(AlphaMode.PREMULTIPLIED, new int[]{ 0xFF336699 }, 1, 1, false);
        opaquePremultiplied.multiplyAlpha(0.5f);
        assertThat(opaquePremultiplied.getPixel(0, 0), equalTo(0x801A334D));
        assertThat(opaquePremultiplied.hasAlpha(), is(false));

        PixelBuffer opaqueStraight = held(AlphaMode.STRAIGHT, new int[]{ 0xFF336699 }, 1, 1, false);
        opaqueStraight.multiplyAlpha(0.5f);
        assertThat(opaqueStraight.getPixel(0, 0), equalTo(0x80336699));
        assertThat(opaqueStraight.hasAlpha(), is(false));
    }

    @Test
    @DisplayName("toAlphaMode answers the buffer itself in the convention it holds and a converted copy in the other, rounding each channel to nearest")
    void toAlphaModeConvertsOrAnswersItself() {
        // The held convention answers the same buffer.
        for (AlphaMode mode : AlphaMode.values()) {
            PixelBuffer buffer = held(mode, new int[]{ 0x80808080 }, 1, 1, true);
            assertThat(mode.name(), buffer.toAlphaMode(mode), is(sameInstance(buffer)));
        }

        // A half-alpha grey premultiplies and comes back exactly; the conversion copies and leaves its
        // source as it was.
        PixelBuffer grey = PixelBuffer.ofStraight(new int[]{ 0x80808080 }, 1, 1);
        PixelBuffer premultipliedGrey = grey.toAlphaMode(AlphaMode.PREMULTIPLIED);
        assertThat(premultipliedGrey.alphaMode(), is(AlphaMode.PREMULTIPLIED));
        assertThat(premultipliedGrey.getPixel(0, 0), equalTo(0x80404040));
        assertThat(premultipliedGrey.data(), is(not(sameInstance(grey.data()))));
        assertThat(grey.getPixel(0, 0), equalTo(0x80808080));
        PixelBuffer straightGrey = premultipliedGrey.toAlphaMode(AlphaMode.STRAIGHT);
        assertThat(straightGrey.alphaMode(), is(AlphaMode.STRAIGHT));
        assertThat(straightGrey.getPixel(0, 0), equalTo(0x80808080));
        assertThat(premultipliedGrey.getPixel(0, 0), equalTo(0x80404040));

        // Un-premultiplying rounds each channel to nearest and leaves alpha 0 and 255 as they are:
        // 0x02010000 floors to 0x027F0000 and 0xFE7F3F00 to itself; 0x80FF0000 carries more red than
        // alpha, which no premultiplied pixel does, and is held at 255.
        int[] goldens = { 0x00000000, 0x00123456, 0xFF123456, 0x01010101, 0x02010000, 0xFE7F3F00, 0x80404040, 0x80FF0000 };
        assertArrayEquals(
            new int[]{ 0x00000000, 0x00123456, 0xFF123456, 0x01FFFFFF, 0x02800000, 0xFE803F00, 0x80808080, 0x80FF0000 },
            PixelBuffer.ofPremultiplied(goldens.clone(), goldens.length, 1).toAlphaMode(AlphaMode.STRAIGHT).data()
        );

        // Un-premultiplying then premultiplying returns every valid premultiplied pixel exactly. Row y
        // holds alpha y + 1 and column x a channel of x % (alpha + 1), so every row walks each channel
        // value its alpha allows - every valid premultiplied pixel, 32,639 of them - with green running
        // down as red runs up and blue at half of red.
        int width = 255;
        int height = 254;
        int[] valid = new int[width * height];

        for (int y = 0; y < height; y++) {
            int alpha = y + 1;
            for (int x = 0; x < width; x++) {
                int channel = x % (alpha + 1);
                valid[y * width + x] = ColorMath.pack(alpha, channel, alpha - channel, channel >> 1);
            }
        }

        PixelBuffer premultiplied = PixelBuffer.ofPremultiplied(valid.clone(), width, height);
        assertArrayEquals(valid, premultiplied.toAlphaMode(AlphaMode.STRAIGHT).toAlphaMode(AlphaMode.PREMULTIPLIED).data());

        // Premultiplying then un-premultiplying lands within half a premultiplied step, rounded to
        // nearest, of every straight pixel: premultiplying keeps alpha + 1 of the 256 straight levels,
        // so a level comes back to the nearest one kept, within (255 + alpha) / (2 * alpha), which is 1
        // from alpha 86 up. A division that floors lands one level further at 57 alphas.
        for (int alpha = 1; alpha < 255; alpha++) {
            int[] straight = new int[256];
            for (int level = 0; level < 256; level++)
                straight[level] = ColorMath.pack(alpha, level, level, 255 - level);

            int[] roundTrip = PixelBuffer.ofStraight(straight, 256, 1)
                .toAlphaMode(AlphaMode.PREMULTIPLIED)
                .toAlphaMode(AlphaMode.STRAIGHT)
                .data();
            int bound = (255 + alpha) / (2 * alpha);

            for (int level = 0; level < 256; level++) {
                int pixel = roundTrip[level];
                String label = "alpha %d, straight %d".formatted(alpha, level);
                assertThat(label, ColorMath.alpha(pixel), equalTo(alpha));
                assertThat(label, Math.abs(ColorMath.red(pixel) - level), lessThanOrEqualTo(bound));
                assertThat(label, Math.abs(ColorMath.blue(pixel) - (255 - level)), lessThanOrEqualTo(bound));
            }
        }

        // A buffer whose alpha carries no data holds the same colour in either convention: it answers
        // a buffer sharing its array under the asked convention, no pixel changed.
        int[] opaque = { 0x00123456, 0x80FF0000 };
        PixelBuffer withoutAlpha = PixelBuffer.ofStraight(opaque, 2, 1, false);
        PixelBuffer relabelled = withoutAlpha.toAlphaMode(AlphaMode.PREMULTIPLIED);
        assertThat(relabelled.alphaMode(), is(AlphaMode.PREMULTIPLIED));
        assertThat(relabelled.hasAlpha(), is(false));
        assertThat(relabelled.data(), is(sameInstance(opaque)));
        assertThat(relabelled.toAlphaMode(AlphaMode.STRAIGHT).data(), is(sameInstance(opaque)));
        assertArrayEquals(new int[]{ 0x00123456, 0x80FF0000 }, opaque);
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
        PixelBuffer opaqueSrc = PixelBuffer.ofStraight(src.data().clone(), 4, 4, false);

        PixelBuffer fast = PixelBuffer.create(8, 8);
        fast.blit(opaqueSrc, 2, 2, BlendMode.NORMAL);

        PixelBuffer generic = PixelBuffer.create(8, 8);
        generic.blit(src, 2, 2, BlendMode.NORMAL);

        assertThat(fast, equalTo(generic));
    }

    @Test
    @DisplayName("equals / hashCode follow deep pixel equality and the convention the pixels are held in")
    void equalsAndHashCode() {
        PixelBuffer a = gradient(3, 3);
        PixelBuffer b = gradient(3, 3);
        PixelBuffer c = gradient(3, 3);
        c.setPixel(0, 0, 0);

        assertThat(a, equalTo(b));
        assertThat(a.hashCode(), equalTo(b.hashCode()));
        assertThat(a, not(equalTo(c)));

        // One array under two conventions is two pictures.
        int[] shared = { 0x80800000, 0x40202020, 0xFF336699, 0x00000000 };
        PixelBuffer straight = PixelBuffer.ofStraight(shared, 2, 2);
        PixelBuffer premultiplied = PixelBuffer.ofPremultiplied(shared, 2, 2);
        assertThat(straight, not(equalTo(premultiplied)));
        assertThat(premultiplied, not(equalTo(straight)));
        assertThat(straight.hashCode(), not(equalTo(premultiplied.hashCode())));
        assertThat(premultiplied, equalTo(PixelBuffer.ofPremultiplied(shared.clone(), 2, 2)));
        assertThat(premultiplied.hashCode(), equalTo(PixelBuffer.ofPremultiplied(shared.clone(), 2, 2).hashCode()));
    }

    @Test
    @DisplayName("toBufferedImage shares the array as TYPE_INT_ARGB_PRE for a canvas and TYPE_INT_ARGB for straight colour")
    void toBufferedImageOfACanvasIsArgbPre() throws IOException {
        PixelBuffer canvas = PixelBuffer.canvas(2, 1);
        canvas.setPixel(0, 0, 0x80800000);
        BufferedImage image = canvas.toBufferedImage();

        assertThat(image.getType(), equalTo(BufferedImage.TYPE_INT_ARGB_PRE));
        assertThat(image.isAlphaPremultiplied(), is(true));
        assertThat(((DataBufferInt) image.getRaster().getDataBuffer()).getData(), is(sameInstance(canvas.data())));

        // ImageIO writes the straight colour a premultiplied image stands for.
        ByteArrayOutputStream png = new ByteArrayOutputStream();
        assertThat(ImageIO.write(image, "png", png), is(true));
        assertThat(Integer.toHexString(ImageIO.read(new ByteArrayInputStream(png.toByteArray())).getRGB(0, 0)), equalTo(Integer.toHexString(0x80FF0000)));
        assertThat(canvas.getPixel(0, 0), equalTo(0x80800000));

        PixelBuffer straight = PixelBuffer.ofStraight(new int[]{ 0x80FF0000, 0 }, 2, 1);
        BufferedImage straightImage = straight.toBufferedImage();
        assertThat(straightImage.getType(), equalTo(BufferedImage.TYPE_INT_ARGB));
        assertThat(straightImage.isAlphaPremultiplied(), is(false));
        assertThat(((DataBufferInt) straightImage.getRaster().getDataBuffer()).getData(), is(sameInstance(straight.data())));
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
