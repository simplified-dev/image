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
import java.awt.image.DirectColorModel;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
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

    /** Every blend mode a colour fill takes: all but the two that name the convention of a source. */
    private static final BlendMode[] FILL_MODES = Arrays.stream(BlendMode.values())
        .filter(mode -> mode != BlendMode.PREMULTIPLIED && mode != BlendMode.STRAIGHT)
        .toArray(BlendMode[]::new);

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
        PixelBuffer buffer = PixelBuffer.straight(4, 4);
        buffer.setPixel(2, 3, 0xDEADBEEF);
        assertThat(buffer.getPixel(2, 3), equalTo(0xDEADBEEF));
    }

    @Test
    @DisplayName("straight and canvas refuse dimensions whose pixel count overflows an int")
    void aBlankBufferRefusesAnOverflowingPixelCount() {
        // 65535 * 65535 wraps to a negative count, 65536 * 65537 to the positive 65536.
        assertThrows(ArithmeticException.class, () -> PixelBuffer.straight(65535, 65535));
        assertThrows(ArithmeticException.class, () -> PixelBuffer.straight(65536, 65537));
        assertThrows(ArithmeticException.class, () -> PixelBuffer.canvas(65535, 65535));
        assertThrows(ArithmeticException.class, () -> PixelBuffer.canvas(65536, 65537));
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

        PixelBuffer blank = PixelBuffer.straight(3, 2);
        assertThat(blank.alphaMode(), is(AlphaMode.STRAIGHT));
        assertThat(blank.hasAlpha(), is(true));
        assertArrayEquals(new int[6], blank.data());

        PixelBuffer canvas = PixelBuffer.canvas(3, 2);
        assertThat(canvas.alphaMode(), is(AlphaMode.PREMULTIPLIED));
        assertThat(canvas.hasAlpha(), is(true));
        assertArrayEquals(new int[6], canvas.data());

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
    @DisplayName("every blank factory is named for the convention it answers: straight and canvas, and no other")
    void everyBlankFactoryNamesItsConvention() {
        // A public static method that takes a width and a height and answers a buffer makes a blank
        // one. Each answers one convention and is named for it, so a blank under any other name holds
        // a convention its name does not say.
        List<String> blanks = Arrays.stream(PixelBuffer.class.getDeclaredMethods())
            .filter(method -> Modifier.isPublic(method.getModifiers()) && Modifier.isStatic(method.getModifiers()))
            .filter(method -> method.getReturnType() == PixelBuffer.class)
            .filter(method -> Arrays.equals(method.getParameterTypes(), new Class<?>[]{ int.class, int.class }))
            .map(Method::getName)
            .sorted()
            .toList();

        assertThat(blanks, contains("canvas", "straight"));
        assertThat(PixelBuffer.straight(1, 1).alphaMode(), is(AlphaMode.STRAIGHT));
        assertThat(PixelBuffer.canvas(1, 1).alphaMode(), is(AlphaMode.PREMULTIPLIED));
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

        // Translucent ints in A, B, G, R order under a colour model that does not premultiply are
        // TYPE_CUSTOM too, with no unpacker of their own, and are drawn into an ARGB image. Its
        // 80FF0000 reads as it is held, where a premultiplied scratch would answer 80800000.
        DirectColorModel abgr = new DirectColorModel(32, 0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000);
        BufferedImage abgrImage = new BufferedImage(abgr, abgr.createCompatibleWritableRaster(3, 1), false, null);
        abgrImage.getRaster().setDataElements(0, 0, 3, 1, new int[]{ 0x800000FF, 0xFF996633, 0x00000000 });
        assertThat(abgrImage.getType(), is(BufferedImage.TYPE_CUSTOM));
        PixelBuffer abgrWrapped = PixelBuffer.wrap(abgrImage);
        assertThat(abgrWrapped.hasAlpha(), is(true));
        assertArrayEquals(new int[]{ 0x80FF0000, 0xFF336699, 0x00000000 }, abgrWrapped.data());

        String[] layouts = { "TYPE_INT_RGB", "TYPE_INT_BGR", "TYPE_BYTE_GRAY", "gray and alpha", "TYPE_BYTE_BINARY", "TYPE_USHORT_565_RGB", "translucent A, B, G, R ints" };
        BufferedImage[] images = {
            new BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB),
            new BufferedImage(2, 2, BufferedImage.TYPE_INT_BGR),
            new BufferedImage(2, 2, BufferedImage.TYPE_BYTE_GRAY),
            grayAndAlphaImage,
            new BufferedImage(2, 2, BufferedImage.TYPE_BYTE_BINARY),
            new BufferedImage(2, 2, BufferedImage.TYPE_USHORT_565_RGB),
            abgrImage
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
        PixelBuffer buffer = PixelBuffer.straight(3, 2);
        int[] row = { 0x11111111, 0x22222222, 0x33333333 };
        buffer.setRow(1, row);
        assertArrayEquals(row, buffer.getRow(1));
    }

    @Test
    @DisplayName("getPixels / setPixels roundtrip a sub-region")
    void regionRoundtrip() {
        PixelBuffer buffer = gradient(8, 8);
        int[] region = buffer.getPixels(2, 2, 3, 3, null, 0, 0);

        PixelBuffer target = PixelBuffer.straight(8, 8);
        target.setPixels(2, 2, 3, 3, region, 0, 3);
        assertThat(target.getPixel(3, 3), equalTo(buffer.getPixel(3, 3)));
        assertThat(target.getPixel(0, 0), equalTo(0));
    }

    @Test
    @DisplayName("fillRect clips out-of-bounds regions")
    void fillRectClips() {
        PixelBuffer buffer = PixelBuffer.straight(4, 4);
        buffer.fillRect(-2, -2, 4, 4, 0xFF00FF00);
        assertThat(buffer.getPixel(0, 0), equalTo(0xFF00FF00));
        assertThat(buffer.getPixel(2, 2), equalTo(0));
    }

    @Test
    @DisplayName("fillRect NORMAL lays F0100010 onto a clear pixel of a canvas as F00F000F, and 505000FF over that as F523005A")
    void fillRectNormalBlendsTheColour() {
        PixelBuffer buffer = PixelBuffer.canvas(3, 2);
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
    @DisplayName("fillRect with a blend lays a colour of alpha 0 nowhere, on a buffer of either convention, under every mode a fill takes but REPLACE")
    void fillRectOfNoAlphaLaysNothing() {
        int[] backdrop = texels(9, 70, 6);

        for (AlphaMode alphaMode : AlphaMode.values())
            for (BlendMode mode : FILL_MODES) {
                if (mode == BlendMode.REPLACE) continue;

                for (int colour : new int[]{ 0x00000000, 0x00FFFFFF, 0x00123456 }) {
                    PixelBuffer buffer = held(alphaMode, backdrop.clone(), 9, 70, true);
                    buffer.fillRect(-1, -1, 11, 72, colour, mode);
                    assertArrayEquals(backdrop, buffer.data(), "%08X, %s, %s".formatted(colour, mode, alphaMode));
                }
            }
    }

    @Test
    @DisplayName("fillRect with a blend covers exactly the pixels the overwriting fillRect covers, on a canvas")
    void fillRectWithABlendClipsAsFillRectClips() {
        // A 2x2 rectangle from (-1, -1) lays only (0, 0), and one overhanging the right and bottom
        // edges only the pixels inside them.
        PixelBuffer corner = PixelBuffer.canvas(4, 3);
        corner.fillRect(-1, -1, 2, 2, 0xF0100010, BlendMode.NORMAL);
        assertArrayEquals(new int[]{ 0xF00F000F, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, corner.data());

        PixelBuffer overhang = PixelBuffer.canvas(4, 3);
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
                PixelBuffer covered = PixelBuffer.straight(8, 7);
                covered.fillRect(x, y, w, h, 0xFFFFFFFF);

                int[] expected = backdrop.clone();
                for (int i = 0; i < expected.length; i++)
                    if (covered.data()[i] != 0)
                        expected[i] = ColorMath.blend(colour, backdrop[i], BlendMode.NORMAL);

                PixelBuffer buffer = PixelBuffer.ofPremultiplied(backdrop.clone(), 8, 7);
                buffer.fillRect(x, y, w, h, colour, BlendMode.NORMAL);
                assertArrayEquals(expected, buffer.data(), label);
            }
        }
    }

    @Test
    @DisplayName("fillRect REPLACE lays what the overwriting fillRect lays, a colour of alpha 0 included, on a buffer of either convention")
    void fillRectReplaceIsTheOverwritingFillRect() {
        int[] backdrop = texels(9, 70, 8);

        for (AlphaMode alphaMode : AlphaMode.values())
            for (int[] rect : new int[][]{ { -2, -3, 6, 80 }, { 3, 1, 4, 5 }, { 7, 60, 9, 20 }, { 2, 2, 0, 5 } })
                for (int colour : new int[]{ 0x00000000, 0x00123456, 0x01FFFFFF, 0x80808080, 0xF0100010, 0xFF336699 }) {
                    int x = rect[0], y = rect[1], w = rect[2], h = rect[3];
                    String label = "%dx%d at (%d, %d), %08X, %s".formatted(w, h, x, y, colour, alphaMode);

                    PixelBuffer overwritten = held(alphaMode, backdrop.clone(), 9, 70, true);
                    overwritten.fillRect(x, y, w, h, colour);

                    PixelBuffer replaced = held(alphaMode, backdrop.clone(), 9, 70, true);
                    replaced.fillRect(x, y, w, h, colour, BlendMode.REPLACE);

                    assertArrayEquals(overwritten.data(), replaced.data(), label);
                }
    }

    @Test
    @DisplayName("fillRect with a blend lays what a blit of a straight one-pixel image of its colour lays at every pixel it covers, onto either convention")
    void fillRectWithABlendLaysWhatABlitOfOnePixelLays() {
        // The backdrop holds no alpha over colour bits, full alpha and partial alpha, and is tall
        // enough for the fill's rows to run in parallel. A fill colour is straight colour, so it is
        // held to a straight image. Under REPLACE a blit discards a source pixel of alpha 0, which the
        // fill writes as the overwriting fillRect does, so that one pairing is held by the REPLACE
        // test instead.
        int width = 9, height = 70;
        int x = -2, y = 3, w = 8, h = 70;
        int[] backdrop = texels(width, height, 9);

        for (AlphaMode alphaMode : AlphaMode.values())
            for (BlendMode mode : FILL_MODES)
                for (int colour : new int[]{ 0x00000000, 0x00FFFFFF, 0x01FFFFFF, 0x505000FF, 0x80808080, 0xF0100010, 0xFEFFFFFF, 0xFF336699 }) {
                    if (mode == BlendMode.REPLACE && ColorMath.alpha(colour) == 0) continue;
                    String label = "%08X, %s onto %s".formatted(colour, mode, alphaMode);

                    PixelBuffer blitted = held(alphaMode, backdrop.clone(), width, height, true);
                    for (int py = y; py < y + h; py++)
                        for (int px = x; px < x + w; px++)
                            blitted.blit(PixelBuffer.ofStraight(new int[]{ colour }, 1, 1), px, py, mode);

                    PixelBuffer filled = held(alphaMode, backdrop.clone(), width, height, true);
                    filled.fillRect(x, y, w, h, colour, mode);

                    assertArrayEquals(blitted.data(), filled.data(), label);
                }
    }

    @Test
    @DisplayName("fill and the overwriting fillRect lay a straight colour in the convention the buffer holds")
    void aFillLaysAStraightColourInTheBuffersConvention() {
        // F0100010 premultiplies to F00F000F and 80FF0000 to 80800000; a colour of alpha 0 is
        // transparent black in premultiplied colour, whatever its colour bits.
        PixelBuffer canvas = PixelBuffer.canvas(3, 2);
        canvas.fill(0xF0100010);
        for (int pixel : canvas.data())
            assertThat(hex(pixel), equalTo(hex(0xF00F000F)));

        canvas.fillRect(1, 0, 2, 1, 0x80FF0000);
        canvas.fillRect(0, 1, 1, 1, 0x00123456);
        assertArrayEquals(new int[]{ 0xF00F000F, 0x80800000, 0x80800000, 0x00000000, 0xF00F000F, 0xF00F000F }, canvas.data());

        PixelBuffer straight = PixelBuffer.straight(3, 2);
        straight.fill(0xF0100010);
        for (int pixel : straight.data())
            assertThat(hex(pixel), equalTo(hex(0xF0100010)));

        straight.fillRect(1, 0, 2, 1, 0x80FF0000);
        straight.fillRect(0, 1, 1, 1, 0x00123456);
        assertArrayEquals(new int[]{ 0xF0100010, 0x80FF0000, 0x80FF0000, 0x00123456, 0xF0100010, 0xF0100010 }, straight.data());
    }

    @Test
    @DisplayName("fillRect NORMAL composites source-over in the convention the buffer holds, and a fill refuses the two modes that name a source's convention")
    void fillRectSourceOverFollowsTheBuffer() {
        // 80FF0000 over the premultiplied F00F000F stores NORMAL's F8870007, and over the straight
        // F0100010 STRAIGHT's F88C0008: what Java2D's SrcOver stores on a TYPE_INT_ARGB_PRE and a
        // TYPE_INT_ARGB image. NORMAL's formula on the straight pixel would store F8880008.
        PixelBuffer canvas = PixelBuffer.ofPremultiplied(new int[]{ 0xF00F000F, 0xF00F000F }, 2, 1);
        canvas.fillRect(0, 0, 1, 1, 0x80FF0000, BlendMode.NORMAL);
        assertThat(hex(canvas.getPixel(0, 0)), equalTo(hex(0xF8870007)));
        assertThat(hex(canvas.getPixel(1, 0)), equalTo(hex(0xF00F000F)));

        PixelBuffer straight = PixelBuffer.ofStraight(new int[]{ 0xF0100010, 0xF0100010 }, 2, 1);
        straight.fillRect(0, 0, 1, 1, 0x80FF0000, BlendMode.NORMAL);
        assertThat(hex(straight.getPixel(0, 0)), equalTo(hex(0xF88C0008)));
        assertThat(hex(straight.getPixel(1, 0)), equalTo(hex(0xF0100010)));

        // A colour argument is straight colour and has no buffer to say otherwise, so the two modes
        // that name a source's convention are refused before anything is laid, a colour of alpha 0
        // and a rectangle of no size included.
        for (BlendMode mode : new BlendMode[]{ BlendMode.PREMULTIPLIED, BlendMode.STRAIGHT })
            for (PixelBuffer buffer : new PixelBuffer[]{ PixelBuffer.canvas(2, 1), PixelBuffer.straight(2, 1) })
                for (int[] fill : new int[][]{ { 0, 0, 2, 1, 0x80FF0000 }, { 0, 0, 2, 1, 0x00123456 }, { 0, 0, 0, 0, 0x80FF0000 } }) {
                    IllegalArgumentException refused = assertThrows(IllegalArgumentException.class,
                        () -> buffer.fillRect(fill[0], fill[1], fill[2], fill[3], fill[4], mode));
                    assertThat(refused.getMessage(), equalTo("A fill colour is straight colour and takes 'NORMAL', not '%s'".formatted(mode)));
                    assertArrayEquals(new int[2], buffer.data(), mode + " left the buffer as it was");
                }
    }

    @Test
    @DisplayName("sourceOverFormula names the formula a one-pixel NORMAL fill lays a straight colour by, in every convention")
    void sourceOverFormulaLaysWhatAOnePixelFillLays() {
        // The backdrop holds no alpha over colour bits, full alpha and partial alpha, in either
        // convention, with and without alpha data. Every pixel is laid once by each side, at every
        // alpha a colour can carry - 0, where the fill skips, and FF, where it writes the colour as
        // given, included.
        int width = 9, height = 7;
        int[] backdrop = texels(width, height, 9);

        for (AlphaMode alphaMode : AlphaMode.values())
            for (boolean hasAlpha : new boolean[]{ true, false })
                for (int alpha = 0; alpha <= 0xFF; alpha++)
                    for (int rgb : new int[]{ 0x000000, 0xFFFFFF, 0x5000FF, 0x100010 }) {
                        int colour = ColorMath.withAlpha(rgb, alpha);
                        String label = "%08X onto %s%s".formatted(colour, alphaMode, hasAlpha ? "" : " without alpha");

                        PixelBuffer filled = held(alphaMode, backdrop.clone(), width, height, hasAlpha);
                        PixelBuffer laid = held(alphaMode, backdrop.clone(), width, height, hasAlpha);
                        BlendMode sourceOver = laid.sourceOverFormula();
                        for (int py = 0; py < height; py++)
                            for (int px = 0; px < width; px++) {
                                filled.fillRect(px, py, 1, 1, colour, BlendMode.NORMAL);
                                laid.setPixel(px, py, ColorMath.blend(colour, laid.getPixel(px, py), sourceOver));
                            }

                        assertArrayEquals(filled.data(), laid.data(), label);
                    }

        // STRAIGHT's formula onto straight colour whose alpha carries data, NORMAL's onto everything else.
        assertThat(PixelBuffer.canvas(1, 1).sourceOverFormula(), is(BlendMode.NORMAL));
        assertThat(PixelBuffer.straight(1, 1).sourceOverFormula(), is(BlendMode.STRAIGHT));
        assertThat(PixelBuffer.ofStraight(new int[1], 1, 1, false).sourceOverFormula(), is(BlendMode.NORMAL));
        assertThat(held(AlphaMode.PREMULTIPLIED, new int[1], 1, 1, false).sourceOverFormula(), is(BlendMode.NORMAL));
    }

    @Test
    @DisplayName("contains reports in-bounds coordinates")
    void containsCoordinates() {
        PixelBuffer buffer = PixelBuffer.straight(4, 4);
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
        PixelBuffer a = PixelBuffer.straight(4, 4);
        assertThrows(IllegalArgumentException.class, () -> a.crop(2, 2, 5, 5));
    }

    @Test
    @DisplayName("resize NEAREST agrees with blitScaled")
    void resizeNearestMatchesBlitScaled() {
        PixelBuffer a = gradient(6, 6);
        PixelBuffer nearest = a.resize(3, 3, Resample.NEAREST);

        PixelBuffer scaled = PixelBuffer.straight(3, 3);
        scaled.blitScaled(a, 0, 0, 3, 3);
        assertThat(nearest, equalTo(scaled));
    }

    @Test
    @DisplayName("resize NEAREST takes the top-left source pixel of each output pixel's footprint")
    void resizeNearestTakesTheTopLeftOfEachFootprint() {
        PixelBuffer row = PixelBuffer.straight(9, 1);
        for (int x = 0; x < 9; x++)
            row.setPixel(x, 0, 0xFF000000 | x);

        // A 3x downscale reads source 0, 3 and 6, where the centre-closest pixels are 1, 4 and 7.
        assertArrayEquals(new int[]{ 0xFF000000, 0xFF000003, 0xFF000006 }, row.resize(3, 1, Resample.NEAREST).data());
        // Upscaling 2 to 3 reads source 0, 0 and 1: each output column rounds down onto the source.
        assertArrayEquals(new int[]{ 0xFF000000, 0xFF000000, 0xFF000001 }, row.crop(0, 0, 2, 1).resize(3, 1, Resample.NEAREST).data());
    }

    @Test
    @DisplayName("blitScaled of a translucent render onto a fresh canvas agrees with resize NEAREST, under PREMULTIPLIED and the default mode alike")
    void blitScaledPremultipliedMatchesResizeNearest() {
        PixelBuffer render = PixelBuffer.canvas(6, 6);
        for (int y = 0; y < 6; y++)
            for (int x = 0; x < 6; x++)
                render.setPixel(x, y, ColorMath.blend(ColorMath.pack(20 + 40 * x, 0xFF, 8 * y, 0x80), ColorMath.TRANSPARENT, BlendMode.NORMAL));

        for (int size : new int[]{ 3, 6, 12 }) {
            PixelBuffer laid = PixelBuffer.canvas(size, size);
            laid.blitScaled(render, 0, 0, size, size, BlendMode.PREMULTIPLIED);
            assertArrayEquals(render.resize(size, size, Resample.NEAREST).data(), laid.data());
        }

        // The default source-over takes its formula from the two canvases, so it lays the render as it
        // is rather than weighing its colour by its alpha a second time.
        PixelBuffer laidByDefault = PixelBuffer.canvas(6, 6);
        laidByDefault.blitScaled(render, 0, 0, 6, 6);
        assertThat(laidByDefault.getPixel(5, 0), equalTo(render.getPixel(5, 0)));
        assertArrayEquals(render.data(), laidByDefault.data());
    }

    @Test
    @DisplayName("blitScaled lays down what a frozen per-pixel copy of its loop lays down, over random and transparent targets, in each pairing of conventions")
    void blitScaledMatchesTheFrozenLoop() {
        // Each geometry is drawn over random texels and over a fresh transparent target, the backdrop
        // most composites land on, so a run filled from what the target holds rather than from the
        // texel alone differs from the frozen loop. The frozen loop lays each texel as the table of
        // conventions says, which in each mode's own pairing is ColorMath.blend of the pixels as held.
        int seed = 0;

        for (int[] blit : SCALED_BLITS) {
            int srcW = blit[0], srcH = blit[1], dw = blit[2], dh = blit[3], dx = blit[4], dy = blit[5], width = blit[6], height = blit[7];
            int[] drawn = texels(srcW, srcH, ++seed);
            int[] random = texels(width, height, ++seed);

            for (int[] backdrop : new int[][]{ random, new int[width * height] })
                for (boolean hasAlpha : new boolean[]{ true, false })
                    for (AlphaMode from : AlphaMode.values())
                        for (AlphaMode to : AlphaMode.values())
                            for (BlendMode mode : BlendMode.values()) {
                                String label = "%dx%d -> %dx%d at (%d, %d) onto %dx%d %s, hasAlpha %s, %s onto %s, %s"
                                    .formatted(srcW, srcH, dw, dh, dx, dy, width, height, backdrop == random ? "random" : "transparent", hasAlpha, from, to, mode);
                                PixelBuffer source = held(from, drawn.clone(), srcW, srcH, hasAlpha);

                                PixelBuffer expected = held(to, backdrop.clone(), width, height, true);
                                frozenBlitScaled(expected, source, dx, dy, dw, dh, mode);

                                PixelBuffer actual = held(to, backdrop.clone(), width, height, true);
                                actual.blitScaled(source, dx, dy, dw, dh, mode);

                                assertArrayEquals(expected.data(), actual.data(), label);
                                assertArrayEquals(drawn, source.data(), label + " left its source as drawn");
                            }
        }
    }

    @Test
    @DisplayName("blitScaled of a buffer onto itself reads each texel after every write ahead of it in its row, in either convention")
    void blitScaledOntoItselfMatchesTheFrozenLoop() {
        // Doubling from the origin, the first pixel of row 0 writes the texel the second pixel reads,
        // so that texel is held translucent to make the write change it in every mode but REPLACE.
        int[] drawn = texels(24, 20, 1);
        drawn[0] = 0x80C08040;

        for (AlphaMode alphaMode : AlphaMode.values())
            for (BlendMode mode : BlendMode.values()) {
                PixelBuffer expected = held(alphaMode, drawn.clone(), 24, 20, true);
                frozenBlitScaled(expected, expected, 0, 0, 48, 40, mode);

                PixelBuffer actual = held(alphaMode, drawn.clone(), 24, 20, true);
                actual.blitScaled(actual, 0, 0, 48, 40, mode);

                assertArrayEquals(expected.data(), actual.data(), alphaMode + ", " + mode);
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

        PixelBuffer fast = PixelBuffer.straight(8, 8);
        fast.fill(0x80204060);
        fast.blit(opaqueSrc, 2, 2, BlendMode.PREMULTIPLIED);

        PixelBuffer generic = PixelBuffer.straight(8, 8);
        generic.fill(0x80204060);
        generic.blit(src, 2, 2, BlendMode.PREMULTIPLIED);

        assertThat(fast, equalTo(generic));
    }

    @Test
    @DisplayName("a composite lays its source in the destination's convention, by the formula the two conventions give")
    void aCompositeLaysTheSourceInTheDestinationsConvention() {
        // Texels of no, full and partial alpha over a backdrop of the same, in each pairing of
        // conventions and under every mode. Each composite is held to the frozen loop laying every
        // pixel as the table of conventions says: blit as a scale of one to one, blitScaled shrinking
        // and magnifying, blitScaledFxaa as FXAA and then the scaled loop, and blitTinted as the tint
        // and then the blit.
        int[] drawn = texels(10, 8, 31);
        int[] backdrop = texels(24, 20, 32);

        for (AlphaMode from : AlphaMode.values())
            for (AlphaMode to : AlphaMode.values())
                for (BlendMode mode : BlendMode.values()) {
                    String label = "%s onto %s, %s".formatted(from, to, mode);
                    PixelBuffer source = held(from, drawn.clone(), 10, 8, true);
                    PixelBuffer filtered = source.copy();
                    filtered.applyFxaa();
                    PixelBuffer tinted = source.tint(0xFF60A0F0);

                    PixelBuffer blitted = held(to, backdrop.clone(), 24, 20, true);
                    blitted.blit(source, 3, 2, mode);
                    PixelBuffer frozenBlit = held(to, backdrop.clone(), 24, 20, true);
                    frozenBlitScaled(frozenBlit, source, 3, 2, 10, 8, mode);
                    assertArrayEquals(frozenBlit.data(), blitted.data(), "blit " + label);

                    for (int[] size : new int[][]{ { -1, 1, 5, 4 }, { -2, -1, 23, 19 } }) {
                        PixelBuffer scaled = held(to, backdrop.clone(), 24, 20, true);
                        scaled.blitScaled(source, size[0], size[1], size[2], size[3], mode);
                        PixelBuffer frozenScaled = held(to, backdrop.clone(), 24, 20, true);
                        frozenBlitScaled(frozenScaled, source, size[0], size[1], size[2], size[3], mode);
                        assertArrayEquals(frozenScaled.data(), scaled.data(), "blitScaled to %dx%d %s".formatted(size[2], size[3], label));
                    }

                    PixelBuffer smoothed = held(to, backdrop.clone(), 24, 20, true);
                    smoothed.blitScaledFxaa(source, 2, 2, 20, 16, mode);
                    PixelBuffer frozenSmoothed = held(to, backdrop.clone(), 24, 20, true);
                    frozenBlitScaled(frozenSmoothed, filtered, 2, 2, 20, 16, mode);
                    assertArrayEquals(frozenSmoothed.data(), smoothed.data(), "blitScaledFxaa " + label);

                    PixelBuffer tintedBlit = held(to, backdrop.clone(), 24, 20, true);
                    tintedBlit.blitTinted(source, 3, 2, 0xFF60A0F0, mode);
                    PixelBuffer frozenTinted = held(to, backdrop.clone(), 24, 20, true);
                    frozenBlitScaled(frozenTinted, tinted, 3, 2, 10, 8, mode);
                    assertArrayEquals(frozenTinted.data(), tintedBlit.data(), "blitTinted " + label);
                }
    }

    @Test
    @DisplayName("the three source-over modes lay one composite, the default blit's, whichever conventions the two buffers hold")
    void theSourceOverModesAreOneModeInAComposite() {
        int[] drawn = texels(10, 8, 33);
        int[] backdrop = texels(24, 20, 34);

        for (AlphaMode from : AlphaMode.values())
            for (AlphaMode to : AlphaMode.values()) {
                PixelBuffer source = held(from, drawn.clone(), 10, 8, true);
                PixelBuffer byDefault = held(to, backdrop.clone(), 24, 20, true);
                byDefault.blit(source, 3, 2);
                PixelBuffer scaledByDefault = held(to, backdrop.clone(), 24, 20, true);
                scaledByDefault.blitScaled(source, -2, -1, 23, 19);

                for (BlendMode mode : new BlendMode[]{ BlendMode.NORMAL, BlendMode.PREMULTIPLIED, BlendMode.STRAIGHT }) {
                    String label = "%s onto %s, %s".formatted(from, to, mode);

                    PixelBuffer blitted = held(to, backdrop.clone(), 24, 20, true);
                    blitted.blit(source, 3, 2, mode);
                    assertArrayEquals(byDefault.data(), blitted.data(), "blit " + label);

                    PixelBuffer scaled = held(to, backdrop.clone(), 24, 20, true);
                    scaled.blitScaled(source, -2, -1, 23, 19, mode);
                    assertArrayEquals(scaledByDefault.data(), scaled.data(), "blitScaled " + label);
                }
            }
    }

    @Test
    @DisplayName("a straight see-through child laid with PREMULTIPLIED onto a canvas stores NORMAL's 50190050 and encodes as its own colour")
    void aStraightChildLaidAsPremultipliedIsNotTooBright() {
        // Laid by PREMULTIPLIED's formula as it is held, the child would stay 505000FF, colour above its
        // alpha on a canvas, and encode as 50FF00FF.
        PixelBuffer child = PixelBuffer.ofStraight(new int[]{ 0x505000FF }, 1, 1);

        PixelBuffer canvas = PixelBuffer.canvas(1, 1);
        canvas.blit(child, 0, 0, BlendMode.PREMULTIPLIED);
        assertThat(hex(canvas.getPixel(0, 0)), equalTo(hex(0x50190050)));
        assertThat(hex(canvas.toAlphaMode(AlphaMode.STRAIGHT).getPixel(0, 0)), equalTo(hex(0x505000FF)));

        PixelBuffer scaled = PixelBuffer.canvas(3, 2);
        scaled.blitScaled(child, 0, 0, 3, 2, BlendMode.PREMULTIPLIED);
        for (int pixel : scaled.data())
            assertThat(hex(pixel), equalTo(hex(0x50190050)));

        byte[] png = ImageFactory.toByteArray(StaticImageData.of(canvas), ImageFormat.PNG);
        assertThat(hex(ImageFactory.fromByteArray(png).toPixelBuffer().getPixel(0, 0)), equalTo(hex(0x505000FF)));
    }

    @Test
    @DisplayName("REPLACE writes the source in the destination's convention, both ways, through blit and both scaled paths")
    void replaceConvertsIntoTheDestinationsConvention() {
        // Straight 80FF0000 is premultiplied 80800000 and F0100010 is F00F000F; an opaque pixel is the
        // same int in both, and a source already in the destination's convention is written as held.
        int[] straight = { 0x80FF0000, 0xF0100010, 0xFF336699 };
        int[] premultiplied = { 0x80800000, 0xF00F000F, 0xFF336699 };

        assertReplaceWrites(PixelBuffer.ofStraight(straight.clone(), 3, 1), AlphaMode.PREMULTIPLIED, premultiplied);
        assertReplaceWrites(PixelBuffer.ofPremultiplied(premultiplied.clone(), 3, 1), AlphaMode.STRAIGHT, straight);
        assertReplaceWrites(PixelBuffer.ofStraight(straight.clone(), 3, 1), AlphaMode.STRAIGHT, straight);
        assertReplaceWrites(PixelBuffer.ofPremultiplied(premultiplied.clone(), 3, 1), AlphaMode.PREMULTIPLIED, premultiplied);
    }

    @Test
    @DisplayName("ADD reads a premultiplied source as the straight colour it stands for")
    void addReadsAPremultipliedSourceAsStraight() {
        // Premultiplied 80800000 stands for straight 80FF0000, which ADD weighs by its alpha onto the
        // premultiplied 40000040: C0800040. Read as it is held, its red would be weighed twice, to 40.
        PixelBuffer source = PixelBuffer.ofPremultiplied(new int[]{ 0x80800000 }, 1, 1);
        PixelBuffer canvas = PixelBuffer.ofPremultiplied(new int[]{ 0x40000040 }, 1, 1);
        canvas.blit(source, 0, 0, BlendMode.ADD);

        assertThat(hex(canvas.getPixel(0, 0)), equalTo(hex(0xC0800040)));
        assertThat(canvas.getPixel(0, 0), equalTo(ColorMath.blend(0x80FF0000, 0x40000040, BlendMode.ADD)));
    }

    @Test
    @DisplayName("ADD, QUADRATIC_ADD and OVERLAY lay onto a pixel in the convention their formula reads, MULTIPLY onto it as held, and source-over onto a buffer without alpha as onto an opaque one")
    void theOtherModesWorkInTheirOwnConvention() {
        // ADD adds premultiplied colour: straight 80FF0000 is premultiplied 80800000, and 800000FF
        // weighed by its alpha adds blue 80 and alpha 80, FF800080. Added to the straight red as held
        // it would be FFFF0080.
        assertFillAndBlitLay(PixelBuffer.ofStraight(new int[]{ 0x80FF0000 }, 1, 1), 0x800000FF, BlendMode.ADD, 0xFF800080);

        // QUADRATIC_ADD adds the squared colour to premultiplied colour: straight 80400000 is
        // 80200000, plus 10 from each channel of 80404040 is 80301010, straight 80602020. Added to the
        // straight colour as held it would be 80501010.
        assertFillAndBlitLay(PixelBuffer.ofStraight(new int[]{ 0x80400000 }, 1, 1), 0x80404040, BlendMode.QUADRATIC_ADD, 0x80602020);

        // OVERLAY reads straight colour: premultiplied 80700000 is straight 80DF0000, which white
        // overlays to 80FF0000, premultiplied 80800000. Overlaid as held, the red of 70 reads as dark
        // and becomes E0, above its alpha.
        assertFillAndBlitLay(PixelBuffer.ofPremultiplied(new int[]{ 0x80700000 }, 1, 1), 0xFFFFFFFF, BlendMode.OVERLAY, 0x80800000);

        // MULTIPLY scales each channel and keeps alpha, which holds in either convention, so it lays
        // onto the pixel as held: straight 03FF0000 by 80 is 03800000, where converting it to
        // premultiplied colour and back around the product would round it to 03AA0000.
        assertFillAndBlitLay(PixelBuffer.ofStraight(new int[]{ 0x03FF0000 }, 1, 1), 0xFF808080, BlendMode.MULTIPLY, 0x03800000);
        assertFillAndBlitLay(PixelBuffer.ofPremultiplied(new int[]{ 0x80800000 }, 1, 1), 0xFF808080, BlendMode.MULTIPLY, 0x80400000);

        // A buffer whose alpha carries no data holds the same colour in either convention, so its pixel
        // is read as held whatever its alpha byte: 00FF0000 plus 800000FF is 80FF0080. Premultiplied at
        // its alpha byte of 0, the red would be lost, to 800000FF.
        assertFillAndBlitLay(PixelBuffer.ofStraight(new int[]{ 0x00FF0000 }, 1, 1, false), 0x800000FF, BlendMode.ADD, 0x80FF0080);

        // Source-over counts such a buffer as opaque, held straight as its premultiplied twin over the
        // same array: 800000FF over the red stores 807F0080, and so does the premultiplied 80000080.
        // By STRAIGHT's formula its alpha byte of 0 would leave the source alone, 800000FF.
        PixelBuffer withoutAlpha = PixelBuffer.ofStraight(new int[]{ 0x00FF0000 }, 1, 1, false);
        assertFillAndBlitLay(withoutAlpha, 0x800000FF, BlendMode.NORMAL, 0x807F0080);
        assertFillAndBlitLay(withoutAlpha.toAlphaMode(AlphaMode.PREMULTIPLIED), 0x800000FF, BlendMode.NORMAL, 0x807F0080);

        PixelBuffer underPremultiplied = withoutAlpha.copy();
        underPremultiplied.blit(PixelBuffer.ofPremultiplied(new int[]{ 0x80000080 }, 1, 1), 0, 0);
        assertThat(hex(underPremultiplied.getPixel(0, 0)), equalTo(hex(0x807F0080)));
    }

    @Test
    @DisplayName("lerp converts its second buffer into the first one's convention before mixing")
    void lerpConvertsItsSecondBuffer() {
        // Straight 80FF0000 and premultiplied 80800000 are one picture, so halfway between them is
        // that picture again, in the first buffer's convention. Mixed as held, the red would be C0.
        PixelBuffer premultiplied = PixelBuffer.ofPremultiplied(new int[]{ 0x80800000 }, 1, 1);
        PixelBuffer straight = PixelBuffer.ofStraight(new int[]{ 0x80FF0000 }, 1, 1);

        PixelBuffer intoPremultiplied = PixelBuffer.lerp(premultiplied, straight, 0.5f);
        assertThat(intoPremultiplied.alphaMode(), is(AlphaMode.PREMULTIPLIED));
        assertThat(hex(intoPremultiplied.getPixel(0, 0)), equalTo(hex(0x80800000)));

        PixelBuffer intoStraight = PixelBuffer.lerp(straight, premultiplied, 0.5f);
        assertThat(intoStraight.alphaMode(), is(AlphaMode.STRAIGHT));
        assertThat(hex(intoStraight.getPixel(0, 0)), equalTo(hex(0x80FF0000)));

        assertThat(premultiplied.getPixel(0, 0), equalTo(0x80800000));
        assertThat(straight.getPixel(0, 0), equalTo(0x80FF0000));
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
            PixelBuffer white = PixelBuffer.straight(size, size);
            white.fill(0xFFFFFFFF);

            for (int target = 1; target <= 24; target++)
                for (int pixel : white.resize(target, target, Resample.BILINEAR).data())
                    assertThat("%d to %d".formatted(size, target), pixel, equalTo(0xFFFFFFFF));
        }
    }

    @Test
    @DisplayName("resize BILINEAR keeps a straight pixel's colour beside a transparent one, and mixes premultiplied pixels as held")
    void resizeBilinearHonoursTheConvention() {
        PixelBuffer straight = PixelBuffer.straight(2, 1);
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
        PixelBuffer straight = PixelBuffer.straight(8, 4);
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
        // the source's convention decides the resampled pixels, which REPLACE writes as they are onto
        // a buffer of that convention.
        PixelBuffer fromStraight = PixelBuffer.straight(4, 4);
        fromStraight.blitScaled(straight, 0, 0, 4, 4, BlendMode.REPLACE, Resample.AREA);
        PixelBuffer fromPremultiplied = PixelBuffer.canvas(4, 4);
        fromPremultiplied.blitScaled(straight.toAlphaMode(AlphaMode.PREMULTIPLIED), 0, 0, 4, 4, BlendMode.REPLACE, Resample.AREA);
        assertArrayEquals(straight.resize(4, 4, Resample.AREA).data(), fromStraight.data());
        assertArrayEquals(straight.toAlphaMode(AlphaMode.PREMULTIPLIED).resize(4, 4, Resample.AREA).data(), fromPremultiplied.data());
    }

    @Test
    @DisplayName("resize BILINEAR reads past an edge as the edge pixel")
    void resizeBilinearClampsToTheEdgePixel() {
        // A ramp over four columns, doubled: the outermost outputs sit a quarter pixel past the first
        // and last columns, so each is that column exactly rather than a mix shifted one inward.
        PixelBuffer ramp = PixelBuffer.straight(4, 1);
        for (int x = 0; x < 4; x++)
            ramp.setPixel(x, 0, ColorMath.pack(255, 0x40 * x, 0, 0));

        PixelBuffer wide = ramp.resize(8, 1, Resample.BILINEAR);
        assertThat(ColorMath.red(wide.getPixel(0, 0)), equalTo(0x00));
        assertThat(ColorMath.red(wide.getPixel(7, 0)), equalTo(0xC0));
    }

    @Test
    @DisplayName("resize BILINEAR samples a one-pixel-wide source as its one column")
    void resizeBilinearAcceptsAOnePixelWideSource() {
        PixelBuffer column = PixelBuffer.straight(1, 3);
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
        PixelBuffer a = PixelBuffer.straight(4, 4);
        assertThrows(IllegalArgumentException.class, () -> a.resize(0, 4, Resample.NEAREST));
    }

    @Test
    @DisplayName("invert XORs RGB, preserving alpha")
    void invertPreservesAlpha() {
        PixelBuffer a = PixelBuffer.straight(1, 1);
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
        PixelBuffer a = PixelBuffer.straight(1, 1);
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
        PixelBuffer a = PixelBuffer.straight(1, 1);
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
        PixelBuffer a = PixelBuffer.straight(10, 10);
        a.fillRect(2, 3, 4, 5, 0xFFFF0000);
        Rectangle r = a.opaqueBounds();
        assertThat(r, equalTo(new Rectangle(2, 3, 4, 5)));
    }

    @Test
    @DisplayName("opaqueBounds is zero-size for fully transparent buffer")
    void opaqueBoundsEmpty() {
        PixelBuffer a = PixelBuffer.straight(4, 4);
        Rectangle r = a.opaqueBounds();
        assertThat(r.width, equalTo(0));
        assertThat(r.height, equalTo(0));
    }

    @Test
    @DisplayName("trim crops to opaque bounds")
    void trimCropsToOpaque() {
        PixelBuffer a = PixelBuffer.straight(10, 10);
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

        PixelBuffer fast = PixelBuffer.straight(8, 8);
        fast.blit(opaqueSrc, 2, 2, BlendMode.NORMAL);

        PixelBuffer generic = PixelBuffer.straight(8, 8);
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
     * Formats a pixel as eight hex digits, so a failed comparison names both pixels legibly.
     *
     * @param argb the packed ARGB pixel
     * @return the pixel in hex
     */
    private static String hex(int argb) {
        return "%08X".formatted(argb);
    }

    /**
     * Asserts that {@link BlendMode#REPLACE} writes every pixel of a one-row source as {@code expected}
     * onto a transparent buffer of the given convention, through blit, a one-to-one blitScaled and a
     * blitScaled doubling the row, whose runs each fill with one converted texel.
     *
     * @param source the one-row source
     * @param to the convention of the buffer written to
     * @param expected the pixels REPLACE writes, one per source pixel
     */
    private static void assertReplaceWrites(PixelBuffer source, AlphaMode to, int[] expected) {
        int width = source.width();
        String label = "%s onto %s".formatted(source.alphaMode(), to);

        PixelBuffer blitted = held(to, new int[width], width, 1, true);
        blitted.blit(source, 0, 0, BlendMode.REPLACE);
        assertArrayEquals(expected, blitted.data(), "blit " + label);

        PixelBuffer copied = held(to, new int[width], width, 1, true);
        copied.blitScaled(source, 0, 0, width, 1, BlendMode.REPLACE);
        assertArrayEquals(expected, copied.data(), "blitScaled " + label);

        PixelBuffer doubled = held(to, new int[width * 2], width * 2, 1, true);
        doubled.blitScaled(source, 0, 0, width * 2, 1, BlendMode.REPLACE);
        for (int x = 0; x < width * 2; x++)
            assertThat("magnified " + label + " at " + x, hex(doubled.getPixel(x, 0)), equalTo(hex(expected[x / 2])));
    }

    /**
     * Asserts that a straight colour laid with {@code mode} onto a one-pixel buffer stores
     * {@code expected}, filled with fillRect and blitted as a straight one-pixel image alike.
     *
     * @param pixel the one-pixel buffer laid onto, left as it is
     * @param colour the straight colour laid
     * @param mode the blend mode
     * @param expected the pixel each lays
     */
    private static void assertFillAndBlitLay(PixelBuffer pixel, int colour, BlendMode mode, int expected) {
        String label = "%08X by %s onto %s %08X".formatted(colour, mode, pixel.alphaMode(), pixel.getPixel(0, 0));

        PixelBuffer filled = pixel.copy();
        filled.fillRect(0, 0, 1, 1, colour, mode);
        assertThat("fillRect " + label, hex(filled.getPixel(0, 0)), equalTo(hex(expected)));

        PixelBuffer blitted = pixel.copy();
        blitted.blit(PixelBuffer.ofStraight(new int[]{ colour }, 1, 1), 0, 0, mode);
        assertThat("blit " + label, hex(blitted.getPixel(0, 0)), equalTo(hex(expected)));
    }

    /**
     * Composites {@code source} onto {@code target} exactly as the scaled blit did when it computed
     * each source column with its own integer divide: the clip, the row and column products in int
     * arithmetic, rows in order, every pixel reading its texel afresh, alpha 0 skipped and every
     * other texel laid by {@link #laid}.
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
                data[dstIdx] = laid(src, data[dstIdx], mode, source, target);
            }
        }
    }

    /**
     * Lays one source pixel over one destination pixel as a composite between buffers of these
     * conventions lays it, computed with {@link ColorMath#blend} and {@link PixelBuffer#toAlphaMode}
     * as the table of conventions says.
     * <p>
     * The three source-over modes are one mode, the formula taken from the two conventions:
     * {@link BlendMode#NORMAL}'s lays a straight source onto premultiplied colour,
     * {@link BlendMode#PREMULTIPLIED}'s a premultiplied source onto it, and {@link BlendMode#STRAIGHT}'s
     * a source read as straight colour onto straight colour. {@link BlendMode#REPLACE} writes the source
     * in the destination's convention. The other four read the source as straight colour:
     * {@link BlendMode#ADD} and {@link BlendMode#QUADRATIC_ADD} lay it onto the destination pixel
     * premultiplied and {@link BlendMode#OVERLAY} onto it straight, converted there and back, and
     * {@link BlendMode#MULTIPLY} onto it as held. A buffer whose alpha carries no data has none of its
     * pixels converted, and source-over onto one lays as onto premultiplied colour.
     *
     * @param src the source pixel, as its buffer holds it
     * @param dst the destination pixel, as its buffer holds it
     * @param mode the blend mode the composite is called with
     * @param source the buffer the source pixel comes from
     * @param target the buffer the destination pixel is in
     * @return the pixel the composite writes
     */
    private static int laid(int src, int dst, BlendMode mode, PixelBuffer source, PixelBuffer target) {
        AlphaMode from = source.alphaMode();
        AlphaMode to = target.alphaMode();
        int straightSource = source.hasAlpha() && from == AlphaMode.PREMULTIPLIED ? converted(src, from, AlphaMode.STRAIGHT) : src;

        return switch (mode) {
            case NORMAL, PREMULTIPLIED, STRAIGHT -> to == AlphaMode.STRAIGHT && target.hasAlpha()
                ? ColorMath.blend(straightSource, dst, BlendMode.STRAIGHT)
                : ColorMath.blend(src, dst, from == AlphaMode.PREMULTIPLIED ? BlendMode.PREMULTIPLIED : BlendMode.NORMAL);
            case REPLACE -> source.hasAlpha() ? converted(src, from, to) : src;
            case MULTIPLY -> ColorMath.blend(straightSource, dst, mode);
            case ADD, QUADRATIC_ADD -> laidIn(AlphaMode.PREMULTIPLIED, straightSource, dst, mode, target);
            case OVERLAY -> laidIn(AlphaMode.STRAIGHT, straightSource, dst, mode, target);
        };
    }

    /**
     * Blends a straight source pixel over a destination pixel converted into the convention a
     * formula reads, and converts the result back into the destination's.
     *
     * @param working the convention the formula reads the destination in
     * @param src the straight source pixel
     * @param dst the destination pixel, as its buffer holds it
     * @param mode the blend mode
     * @param target the buffer the destination pixel is in
     * @return the blended pixel, in the destination's convention
     */
    private static int laidIn(AlphaMode working, int src, int dst, BlendMode mode, PixelBuffer target) {
        if (!target.hasAlpha() || target.alphaMode() == working)
            return ColorMath.blend(src, dst, mode);

        return converted(ColorMath.blend(src, converted(dst, target.alphaMode(), working), mode), working, target.alphaMode());
    }

    /**
     * Converts one pixel between conventions through {@link PixelBuffer#toAlphaMode}.
     *
     * @param pixel the pixel, held in {@code from}
     * @param from the convention the pixel is held in
     * @param to the convention to hold it in
     * @return the pixel held in {@code to}
     */
    private static int converted(int pixel, AlphaMode from, AlphaMode to) {
        return held(from, new int[]{ pixel }, 1, 1, true).toAlphaMode(to).getPixel(0, 0);
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
