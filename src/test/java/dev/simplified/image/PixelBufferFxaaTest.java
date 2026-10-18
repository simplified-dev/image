package dev.simplified.image;

import dev.simplified.image.pixel.AlphaMode;
import dev.simplified.image.pixel.BlendMode;
import dev.simplified.image.pixel.ColorMath;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.everyItem;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.lessThanOrEqualTo;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Holds FXAA to the picture it is given: a softened edge keeps the colour of the shape it belongs to
 * at its own coverage in either alpha convention, the two conventions smooth one picture alike, and
 * an opaque picture stays opaque. The scaled composite that filters as it samples lays down what
 * filtering the whole source and then scaling it lays down, and the filter in place writes what a
 * frozen copy of its per-pixel body writes.
 */
class PixelBufferFxaaTest {

    /** The one colour every test shape is drawn in. */
    private static final int FACE = 0xFFFFE070;

    /**
     * Source width, source height, destination width, destination height and destination origin of
     * each scaled composite: whole factors of 2, 3 and 4 on frames that are not square, a clipped
     * origin, a factor that is not whole, an upscale that reaches the source's last row and column,
     * sources too narrow or too short for FXAA to filter anything, and a halving tall enough for its
     * rows to split across the pool.
     */
    private static final int[][] GEOMETRIES = {
        { 74, 46, 37, 23, 0, 0 },
        { 111, 69, 37, 23, 0, 0 },
        { 148, 92, 37, 23, 0, 0 },
        { 52, 44, 26, 22, -4, 3 },
        { 50, 31, 17, 12, 5, -2 },
        { 7, 5, 20, 13, 0, 0 },
        { 2, 40, 1, 20, 0, 0 },
        { 40, 2, 20, 1, 0, 0 },
        { 2, 2, 1, 1, 0, 0 },
        { 3, 3, 3, 3, 0, 0 },
        { 256, 192, 128, 96, 0, 0 }
    };

    /** Width of each picture the neighbourhood test draws. */
    private static final int STENCIL_WIDTH = 23;

    /** Height of each picture the neighbourhood test draws. */
    private static final int STENCIL_HEIGHT = 19;

    /**
     * Field, island and checkerboard partner of each picture the neighbourhood test draws: opaque,
     * islands on a flat field, translucent, a transparent island holding colour bits, a transparent
     * field holding colour bits, premultiplied colour above its alpha, and one colour at three alphas,
     * whose neighbourhoods differ in the alpha byte alone.
     */
    private static final int[][] PALETTES = {
        { 0xFF303030, 0xFFE0E0E0, 0xFF2060C0 },
        { 0xFF303030, 0xFFE0E0E0, 0xFF303030 },
        { 0x80406080, 0x80F0E0C0, 0x40C02020 },
        { 0xFFC0A060, 0x00FFFFFF, 0x00000000 },
        { 0x00FFFFFF, 0xFFC0A060, 0x0000FF00 },
        { 0x40303030, 0x40FFFFFF, 0x40102030 },
        { 0xFFFFE070, 0x80FFE070, 0x20FFE070 }
    };

    /**
     * Draws a disc and a staircase of uneven steps in one pixel over another. Along a curved or broken
     * edge FXAA's two samples land at different coverage, which is where a mix that mishandles alpha
     * shows.
     *
     * @param size the width and height of the picture
     * @param inside the pixel the shapes are drawn in
     * @param outside the pixel everywhere else
     * @return the picture
     */
    private static PixelBuffer shapes(int size, int inside, int outside) {
        PixelBuffer buffer = PixelBuffer.create(size, size);
        float centre = size * 0.3f;
        float radius = size * 0.22f;

        for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++) {
                float dx = x + 0.5f - centre;
                float dy = y + 0.5f - centre;
                boolean disc = dx * dx + dy * dy < radius * radius;
                boolean stairs = x > size / 2 && y > size / 2 && x - size / 2 < (y - size / 2) * (y % 3 + 1) / 2;
                buffer.setPixel(x, y, disc || stairs ? inside : outside);
            }

        return buffer;
    }

    /**
     * Asserts a premultiplied pixel holds {@code colour} scaled by its own alpha, to within one unit of
     * rounding.
     *
     * @param pixel the premultiplied pixel
     * @param colour the colour it must carry, read at full strength
     */
    private static void assertCarries(int pixel, int colour) {
        int alpha = ColorMath.alpha(pixel);
        int[] held = { ColorMath.red(pixel), ColorMath.green(pixel), ColorMath.blue(pixel) };
        int[] full = { ColorMath.red(colour), ColorMath.green(colour), ColorMath.blue(colour) };

        for (int c = 0; c < 3; c++)
            assertThat("%08X carries %06X".formatted(pixel, colour & 0xFFFFFF), (double) held[c], closeTo(full[c] * alpha / 255.0, 1.0));
    }

    @Test
    @DisplayName("a softened premultiplied edge carries the shape's colour at its own coverage")
    void aPremultipliedEdgeCarriesTheShapesColour() {
        PixelBuffer buffer = shapes(48, FACE, ColorMath.TRANSPARENT);
        buffer.applyFxaa(AlphaMode.PREMULTIPLIED);
        int softened = 0;

        for (int pixel : buffer.data()) {
            int alpha = ColorMath.alpha(pixel);
            if (alpha > 0 && alpha < 0xFF) softened++;
            assertCarries(pixel, FACE);
        }

        assertThat(softened, is(greaterThan(0)));
    }

    @Test
    @DisplayName("a softened straight edge keeps the shape's colour whatever the transparent pixels hold")
    void aStraightEdgeKeepsTheShapesColour() {
        // Transparent pixels whose colour channels hold green: straight colour at no alpha carries
        // nothing, so none of it may reach the edge, and neither may the black a mix would otherwise
        // pull toward.
        PixelBuffer buffer = shapes(48, FACE, 0x0000FF00);
        buffer.applyFxaa(AlphaMode.STRAIGHT);
        int softened = 0;

        for (int pixel : buffer.data()) {
            int alpha = ColorMath.alpha(pixel);
            if (alpha == 0) continue;
            if (alpha < 0xFF) softened++;
            assertThat("%08X".formatted(pixel), pixel & 0xFFFFFF, is(FACE & 0xFFFFFF));
        }

        assertThat(softened, is(greaterThan(0)));
    }

    @Test
    @DisplayName("straight and premultiplied FXAA smooth one picture alike, opaque or translucent")
    void bothConventionsSmoothOnePictureAlike() {
        // At a coverage of 2 the carried colour's luma sits just above the filter's absolute contrast
        // floor, so an edge test reading the luma at the wrong scale skips edges the other convention
        // smooths.
        for (int coverage : new int[]{ 0xFF, 0x80, 0x18, 0x02 }) {
            PixelBuffer straight = shapes(48, (FACE & 0xFFFFFF) | coverage << 24, 0x0000FF00);
            PixelBuffer premultiplied = straight.copy();
            premultiplied.premultiplyAlpha();

            straight.applyFxaa(AlphaMode.STRAIGHT);
            premultiplied.applyFxaa(AlphaMode.PREMULTIPLIED);
            straight.premultiplyAlpha();

            for (int i = 0; i < straight.data().length; i++) {
                int expected = straight.data()[i];
                int actual = premultiplied.data()[i];
                String label = "coverage %d pixel %d: %08X against %08X".formatted(coverage, i, actual, expected);

                assertThat(label, ColorMath.alpha(actual), is(ColorMath.alpha(expected)));
                assertThat(label, Math.abs(ColorMath.red(actual) - ColorMath.red(expected)), is(lessThanOrEqualTo(2)));
                assertThat(label, Math.abs(ColorMath.green(actual) - ColorMath.green(expected)), is(lessThanOrEqualTo(2)));
                assertThat(label, Math.abs(ColorMath.blue(actual) - ColorMath.blue(expected)), is(lessThanOrEqualTo(2)));
            }
        }
    }

    @Test
    @DisplayName("FXAA over an opaque picture leaves every pixel opaque, in either convention")
    void anOpaquePictureStaysOpaque() {
        for (AlphaMode alphaMode : AlphaMode.values()) {
            PixelBuffer buffer = shapes(96, FACE, 0xFF2040A0);
            int[] drawn = buffer.data().clone();
            buffer.applyFxaa(alphaMode);
            int softened = 0;

            for (int i = 0; i < drawn.length; i++) {
                assertThat("%s pixel %d".formatted(alphaMode, i), ColorMath.alpha(buffer.data()[i]), is(0xFF));
                if (buffer.data()[i] != drawn[i]) softened++;
            }

            assertThat(alphaMode + " softened the edges", softened, is(greaterThan(0)));
        }
    }

    @Test
    @DisplayName("a buffer whose alpha carries no data is smoothed as opaque, in either convention")
    void aBufferWithoutAlphaIsSmoothedAsOpaque() {
        for (AlphaMode alphaMode : AlphaMode.values()) {
            PixelBuffer opaque = shapes(48, FACE, 0xFF2040A0);
            int[] unset = opaque.data().clone();
            Arrays.setAll(unset, i -> unset[i] & 0xFFFFFF);
            PixelBuffer withoutAlpha = PixelBuffer.of(unset, 48, 48, false);

            opaque.applyFxaa(alphaMode);
            withoutAlpha.applyFxaa(alphaMode);

            for (int i = 0; i < unset.length; i++)
                assertThat("%s pixel %d".formatted(alphaMode, i), withoutAlpha.data()[i] & 0xFFFFFF, is(opaque.data()[i] & 0xFFFFFF));
        }
    }

    @Test
    @DisplayName("a pixel under the contrast threshold keeps its exact bytes, in either convention")
    void aFlatPictureKeepsItsBytes() {
        // Uniform, so no neighbourhood has contrast and nothing is rewritten. Divided through by its
        // alpha and scaled back, this pixel would round to 199: what is left alone must not round.
        int drawn = ColorMath.pack(201, 200, 200, 200);

        for (AlphaMode alphaMode : AlphaMode.values()) {
            PixelBuffer buffer = PixelBuffer.create(16, 16);
            buffer.fill(drawn);
            buffer.applyFxaa(alphaMode);
            assertThat(alphaMode.name(), Arrays.stream(buffer.data()).boxed().toList(), everyItem(is(drawn)));
        }
    }

    @Test
    @DisplayName("a dark edge under FXAA's contrast floor keeps its bytes, though the relative threshold alone would smooth it")
    void aDarkEdgeUnderTheFloorKeepsItsBytes() {
        // FXAA's floor is 0.0156 of a luma in [0, 1], about 4 of this filter's [0, 255]. Shapes at
        // luma 5 on a field at 2 have a contrast of 3: past the relative threshold, 0.0312 of their
        // luma, but under the floor. A bright edge of the same contrast is held by the relative
        // threshold instead, and a dark edge past the floor is smoothed.
        int darkField = ColorMath.pack(255, 2, 2, 2);
        int[][] unfiltered = {
            { ColorMath.pack(255, 5, 5, 5), darkField },
            { ColorMath.pack(255, 205, 205, 205), ColorMath.pack(255, 202, 202, 202) }
        };

        for (AlphaMode alphaMode : AlphaMode.values()) {
            for (int[] pair : unfiltered) {
                PixelBuffer drawn = shapes(40, pair[0], pair[1]);
                PixelBuffer buffer = drawn.copy();
                buffer.applyFxaa(alphaMode);
                assertArrayEquals(drawn.data(), buffer.data(), "%s, %08X on %08X".formatted(alphaMode, pair[0], pair[1]));
            }

            PixelBuffer drawn = shapes(40, ColorMath.pack(255, 12, 12, 12), darkField);
            PixelBuffer buffer = drawn.copy();
            buffer.applyFxaa(alphaMode);
            assertThat("%s, a dark edge past the floor is smoothed".formatted(alphaMode),
                Arrays.equals(drawn.data(), buffer.data()), is(false));
        }
    }

    @Test
    @DisplayName("the scaled composite that filters as it samples lays down what FXAA and then the scaled blit lay down")
    void theFilteringCompositeMatchesFxaaThenTheScaledBlit() {
        int seed = 0;
        int smoothed = 0;

        for (int[] geometry : GEOMETRIES) {
            int srcW = geometry[0], srcH = geometry[1], dw = geometry[2], dh = geometry[3], dx = geometry[4], dy = geometry[5];

            for (boolean blocky : new boolean[]{ false, true }) {
                int[] drawn = blocky ? blocky(srcW, srcH, ++seed) : noise(srcW, srcH, ++seed);
                PixelBuffer destination = PixelBuffer.of(noise(dw + 3, dh + 2, ++seed), dw + 3, dh + 2);

                for (boolean hasAlpha : new boolean[]{ true, false })
                    for (AlphaMode alphaMode : AlphaMode.values())
                        for (BlendMode mode : BlendMode.values()) {
                            String label = "%dx%d -> %dx%d at (%d, %d), %s, hasAlpha %s, %s, %s"
                                .formatted(srcW, srcH, dw, dh, dx, dy, blocky ? "blocky" : "noise", hasAlpha, alphaMode, mode);
                            PixelBuffer source = PixelBuffer.of(drawn.clone(), srcW, srcH, hasAlpha);

                            PixelBuffer filtered = PixelBuffer.of(drawn.clone(), srcW, srcH, hasAlpha);
                            filtered.applyFxaa(alphaMode);
                            PixelBuffer expected = destination.copy();
                            expected.blitScaled(filtered, dx, dy, dw, dh, mode);

                            PixelBuffer actual = destination.copy();
                            actual.blitScaledFxaa(source, dx, dy, dw, dh, mode, alphaMode);

                            assertArrayEquals(expected.data(), actual.data(), label);
                            assertArrayEquals(drawn, source.data(), label + " left its source as drawn");

                            PixelBuffer unfiltered = destination.copy();
                            unfiltered.blitScaled(source, dx, dy, dw, dh, mode);
                            if (!Arrays.equals(unfiltered.data(), actual.data())) smoothed++;
                        }
            }
        }

        assertThat("the composites smoothed something", smoothed, is(greaterThan(0)));
    }

    @Test
    @DisplayName("the scaled composite that filters as it samples refuses a source holding the target's pixels")
    void theFilteringCompositeRefusesItsOwnPixels() {
        PixelBuffer buffer = PixelBuffer.of(noise(8, 8, 1), 8, 8);
        PixelBuffer sharing = PixelBuffer.of(buffer.data(), 8, 8);

        assertThrows(IllegalArgumentException.class,
            () -> buffer.blitScaledFxaa(buffer, 0, 0, 4, 4, BlendMode.PREMULTIPLIED, AlphaMode.PREMULTIPLIED));
        assertThrows(IllegalArgumentException.class,
            () -> buffer.blitScaledFxaa(sharing, 0, 0, 4, 4, BlendMode.PREMULTIPLIED, AlphaMode.PREMULTIPLIED));
    }

    @Test
    @DisplayName("FXAA in place writes what a frozen copy of its per-pixel filter writes, in either convention, with or without alpha")
    void fxaaInPlaceMatchesTheFrozenFilter() {
        int seed = 100;
        int smoothed = 0;

        for (int[] size : new int[][]{ { 128, 96 }, { 61, 97 }, { 3, 3 }, { 2, 9 }, { 9, 2 } })
            for (boolean blocky : new boolean[]{ false, true }) {
                int width = size[0], height = size[1];
                int[] drawn = blocky ? blocky(width, height, ++seed) : noise(width, height, ++seed);

                for (boolean hasAlpha : new boolean[]{ true, false })
                    for (AlphaMode alphaMode : AlphaMode.values()) {
                        PixelBuffer buffer = PixelBuffer.of(drawn.clone(), width, height, hasAlpha);
                        buffer.applyFxaa(alphaMode);
                        int[] expected = FrozenFxaa.apply(drawn, width, height, hasAlpha, alphaMode);

                        assertArrayEquals(expected, buffer.data(), "%dx%d %s, hasAlpha %s, %s"
                            .formatted(width, height, blocky ? "blocky" : "noise", hasAlpha, alphaMode));
                        if (!Arrays.equals(drawn, expected)) smoothed++;
                    }
            }

        assertThat("the frozen filter smoothed something", smoothed, is(greaterThan(0)));
    }

    @Test
    @DisplayName("FXAA in place skips only a neighbourhood of one value, writing what the frozen filter writes everywhere else")
    void fxaaInPlaceSkipsOnlyANeighbourhoodOfOneValue() {
        // Four corners equal with a different centre, or a centre and three corners equal with the
        // fourth different, are filtered wherever their contrast clears the threshold; an island
        // makes both, the second on each of its four corners. A buffer whose alpha carries no data
        // keeps the alpha bits it holds wherever nothing is filtered.
        Map<String, int[]> pictures = new LinkedHashMap<>();

        for (int[] palette : PALETTES) {
            String colours = "%08X/%08X/%08X".formatted(palette[0], palette[1], palette[2]);
            pictures.put("islands " + colours, islands(palette[0], palette[1], palette[2]));
            pictures.put("inverted islands " + colours, islands(palette[1], palette[0], palette[2]));
            pictures.put("flat %08X".formatted(palette[0]), flat(palette[0]));
        }

        pictures.put("translucent ramp", ramp());
        int smoothed = 0;

        for (Map.Entry<String, int[]> picture : pictures.entrySet()) {
            int[] drawn = picture.getValue();

            for (boolean hasAlpha : new boolean[]{ true, false })
                for (AlphaMode alphaMode : AlphaMode.values()) {
                    PixelBuffer buffer = PixelBuffer.of(drawn.clone(), STENCIL_WIDTH, STENCIL_HEIGHT, hasAlpha);
                    buffer.applyFxaa(alphaMode);
                    int[] expected = FrozenFxaa.apply(drawn, STENCIL_WIDTH, STENCIL_HEIGHT, hasAlpha, alphaMode);

                    assertArrayEquals(expected, buffer.data(), "%s, hasAlpha %s, %s"
                        .formatted(picture.getKey(), hasAlpha, alphaMode));
                    if (!Arrays.equals(drawn, expected)) smoothed++;
                }
        }

        assertThat("the frozen filter smoothed something", smoothed, is(greaterThan(0)));
    }

    /**
     * Draws a checkerboard of a field and its partner with single-pixel islands on the field's
     * squares, four apart each way. An island's neighbourhood differs from the field at its centre
     * alone, and each of its diagonal neighbours' at the one corner the island sits on, while the
     * partner squares give the bilinear samples a second value to mix in. Every other neighbourhood
     * holds one value.
     *
     * @param field the pixel on the squares the islands sit among
     * @param island the pixel each island holds
     * @param partner the pixel on the other squares
     * @return the pixels, {@link #STENCIL_WIDTH} by {@link #STENCIL_HEIGHT}
     */
    private static int[] islands(int field, int island, int partner) {
        int[] pixels = new int[STENCIL_WIDTH * STENCIL_HEIGHT];

        for (int y = 0; y < STENCIL_HEIGHT; y++)
            for (int x = 0; x < STENCIL_WIDTH; x++)
                pixels[y * STENCIL_WIDTH + x] = x % 4 == 2 && y % 4 == 2 ? island : (x + y) % 2 == 0 ? field : partner;

        return pixels;
    }

    /**
     * Draws one pixel everywhere.
     *
     * @param pixel the pixel
     * @return the pixels, {@link #STENCIL_WIDTH} by {@link #STENCIL_HEIGHT}
     */
    private static int[] flat(int pixel) {
        int[] pixels = new int[STENCIL_WIDTH * STENCIL_HEIGHT];
        Arrays.fill(pixels, pixel);
        return pixels;
    }

    /**
     * Draws translucent pixels whose alpha and colour channels step by a few units between
     * neighbours, so no two neighbouring pixels hold one value and the contrast between them stays
     * mostly under the threshold.
     *
     * @return the pixels, {@link #STENCIL_WIDTH} by {@link #STENCIL_HEIGHT}
     */
    private static int[] ramp() {
        int[] pixels = new int[STENCIL_WIDTH * STENCIL_HEIGHT];

        for (int y = 0; y < STENCIL_HEIGHT; y++)
            for (int x = 0; x < STENCIL_WIDTH; x++)
                pixels[y * STENCIL_WIDTH + x] = ColorMath.pack(0x90 + (x + y) % 3, 100 + x % 5, 120 + y % 4, 140 + (x * y) % 3);

        return pixels;
    }

    /**
     * Draws random pixels of random alpha, a fifth of them with no alpha over colour channels that
     * still hold bits.
     *
     * @param width the picture width
     * @param height the picture height
     * @param seed the random seed
     * @return the pixels
     */
    private static int[] noise(int width, int height, int seed) {
        Random random = new Random(seed);
        int[] pixels = new int[width * height];

        for (int i = 0; i < pixels.length; i++) {
            int argb = random.nextInt();
            pixels[i] = random.nextInt(5) == 0 ? argb & 0x00FFFFFF : argb;
        }

        return pixels;
    }

    /**
     * Draws random cells of one random pixel each, a fifth of them with no alpha over colour channels
     * that still hold bits. Inside a cell nothing has contrast, so FXAA leaves those pixels as they
     * are, and along a cell's edge it smooths.
     *
     * @param width the picture width
     * @param height the picture height
     * @param seed the random seed
     * @return the pixels
     */
    private static int[] blocky(int width, int height, int seed) {
        Random random = new Random(seed);
        int cell = 3 + random.nextInt(4);
        int columns = width / cell + 1;
        int[] colours = noise(columns, height / cell + 1, seed + 7919);
        int[] pixels = new int[width * height];

        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                pixels[y * width + x] = colours[(y / cell) * columns + x / cell];

        return pixels;
    }

    /**
     * A frozen copy of FXAA in place as it stood before its per-pixel body was shared with the scaled
     * composite, and before a neighbourhood of one value was skipped ahead of its lumas: the whole
     * source cloned and forced opaque up front when its alpha carries no data, the edge test run on
     * every interior pixel, the float operations in their order, the bilinear taps clamped and strided
     * by the buffer's own size, and the luma the edge test reads, with FXAA's two absolute constants -
     * the contrast floor and the direction floor, written for a luma in [0, 1] - scaled to that
     * luma's [0, 255].
     */
    private static final class FrozenFxaa {

        /**
         * Filters a copy of {@code data} the way FXAA in place filters it.
         *
         * @param data the pixels to filter, left unmodified
         * @param width the picture width
         * @param height the picture height
         * @param hasAlpha whether the alpha channel carries meaningful data
         * @param alphaMode the convention the pixels are held in
         * @return the filtered pixels
         */
        static int[] apply(int[] data, int width, int height, boolean hasAlpha, AlphaMode alphaMode) {
            int[] out = data.clone();
            boolean straight = switch (alphaMode) {
                case STRAIGHT -> true;
                case PREMULTIPLIED -> false;
            };
            int[] temp = data.clone();
            if (!hasAlpha)
                Arrays.setAll(temp, i -> temp[i] | 0xFF000000);

            final float fxaaReduceMin = 1.0f / 128.0f * 255.0f;
            final float fxaaReduceMul = 1.0f / 4.0f;
            final float fxaaSpanMax = 8.0f;

            for (int y = 1; y < height - 1; y++) {
                float[] samples = new float[8];

                for (int x = 1; x < width - 1; x++) {
                    int rgbNw = temp[(y - 1) * width + (x - 1)];
                    int rgbNe = temp[(y - 1) * width + (x + 1)];
                    int rgbSw = temp[(y + 1) * width + (x - 1)];
                    int rgbSe = temp[(y + 1) * width + (x + 1)];
                    int rgbM = temp[y * width + x];

                    float lumaNw = edgeLuma(rgbNw, alphaMode);
                    float lumaNe = edgeLuma(rgbNe, alphaMode);
                    float lumaSw = edgeLuma(rgbSw, alphaMode);
                    float lumaSe = edgeLuma(rgbSe, alphaMode);
                    float lumaM = edgeLuma(rgbM, alphaMode);

                    float lumaMin = Math.min(lumaM, Math.min(Math.min(lumaNw, lumaNe), Math.min(lumaSw, lumaSe)));
                    float lumaMax = Math.max(lumaM, Math.max(Math.max(lumaNw, lumaNe), Math.max(lumaSw, lumaSe)));

                    float contrast = lumaMax - lumaMin;

                    if (contrast < Math.max(0.0156f * 255.0f, lumaMax * 0.0312f))
                        continue;

                    float dirX = -((lumaNw + lumaNe) - (lumaSw + lumaSe));
                    float dirY = (lumaNw + lumaSw) - (lumaNe + lumaSe);

                    float dirReduce = Math.max(
                        (lumaNw + lumaNe + lumaSw + lumaSe) * (0.25f * fxaaReduceMul),
                        fxaaReduceMin
                    );
                    float rcpDirMin = 1.0f / (Math.min(Math.abs(dirX), Math.abs(dirY)) + dirReduce);

                    dirX = Math.clamp(dirX * rcpDirMin, -fxaaSpanMax, fxaaSpanMax);
                    dirY = Math.clamp(dirY * rcpDirMin, -fxaaSpanMax, fxaaSpanMax);

                    sampleBilinear(temp, width, height,
                        x + dirX * (1.0f / 3.0f - 0.5f),
                        y + dirY * (1.0f / 3.0f - 0.5f),
                        straight, samples, 0);
                    sampleBilinear(temp, width, height,
                        x + dirX * (2.0f / 3.0f - 0.5f),
                        y + dirY * (2.0f / 3.0f - 0.5f),
                        straight, samples, 4);

                    out[y * width + x] = packSample(
                        (samples[0] + samples[4]) * 0.5f,
                        (samples[1] + samples[5]) * 0.5f,
                        (samples[2] + samples[6]) * 0.5f,
                        (samples[3] + samples[7]) * 0.5f,
                        alphaMode
                    );
                }
            }

            return out;
        }

        /**
         * Computes the luma the edge test reads in either convention.
         *
         * @param argb the packed ARGB pixel
         * @param alphaMode the convention the pixel is held in
         * @return the luma in {@code [0, 255]}
         */
        private static float edgeLuma(int argb, AlphaMode alphaMode) {
            return switch (alphaMode) {
                case STRAIGHT -> luma(argb);
                case PREMULTIPLIED -> luma(argb | 0xFF000000);
            };
        }

        /**
         * Computes the Rec. 601 luminance of a pixel, weighted by its alpha.
         *
         * @param argb the packed ARGB pixel
         * @return the luminance in {@code [0, 255]}
         */
        private static float luma(int argb) {
            int a = ColorMath.alpha(argb);
            int r = ColorMath.red(argb);
            int g = ColorMath.green(argb);
            int b = ColorMath.blue(argb);
            return (r * 0.299f + g * 0.587f + b * 0.114f) * (a / 255f);
        }

        /**
         * Packs a filtered sample into a pixel of the given convention, rounding each channel once.
         *
         * @param a the sample's coverage
         * @param r the red the coverage carries
         * @param g the green the coverage carries
         * @param b the blue the coverage carries
         * @param alphaMode the convention to pack into
         * @return the packed ARGB pixel
         */
        private static int packSample(float a, float r, float g, float b, AlphaMode alphaMode) {
            int alpha = channel(a, 255);
            if (alpha == 0) return 0;

            return switch (alphaMode) {
                case STRAIGHT -> {
                    float scale = 255f / a;
                    yield ColorMath.pack(alpha, channel(r * scale, 255), channel(g * scale, 255), channel(b * scale, 255));
                }
                case PREMULTIPLIED -> ColorMath.pack(alpha, channel(r, alpha), channel(g, alpha), channel(b, alpha));
            };
        }

        /**
         * Rounds a channel value half up and holds it in {@code [0, max]}.
         *
         * @param value the channel value
         * @param max the largest value the channel may take
         * @return the rounded channel
         */
        private static int channel(float value, int max) {
            return Math.clamp(Math.round(value), 0, max);
        }

        /**
         * Samples {@code source} bilinearly at a point into {@code out} from {@code offset}, clamping
         * and striding by the picture's own size.
         *
         * @param source the pixels to sample
         * @param width the picture width
         * @param height the picture height
         * @param x the horizontal sample position
         * @param y the vertical sample position
         * @param weightByAlpha whether each tap's colour is scaled by its alpha before the mix
         * @param out the array receiving alpha, red, green and blue
         * @param offset the index in {@code out} the four channels start at
         */
        private static void sampleBilinear(int[] source, int width, int height, float x, float y, boolean weightByAlpha, float[] out, int offset) {
            int floorX = (int) Math.floor(x);
            int floorY = (int) Math.floor(y);
            float fx = x - floorX;
            float fy = y - floorY;

            int x0 = Math.clamp(floorX, 0, width - 1);
            int x1 = Math.clamp(floorX + 1, 0, width - 1);
            int row0 = Math.clamp(floorY, 0, height - 1) * width;
            int row1 = Math.clamp(floorY + 1, 0, height - 1) * width;

            Arrays.fill(out, offset, offset + 4, 0f);
            accumulateTap(source[row0 + x0], (1 - fx) * (1 - fy), weightByAlpha, out, offset);
            accumulateTap(source[row0 + x1], fx * (1 - fy), weightByAlpha, out, offset);
            accumulateTap(source[row1 + x0], (1 - fx) * fy, weightByAlpha, out, offset);
            accumulateTap(source[row1 + x1], fx * fy, weightByAlpha, out, offset);
        }

        /**
         * Adds one bilinear tap's share to a sample being accumulated.
         *
         * @param argb the tap's packed ARGB pixel
         * @param weight the tap's bilinear weight
         * @param weightByAlpha whether the tap's colour is scaled by its alpha
         * @param out the array accumulating alpha, red, green and blue
         * @param offset the index in {@code out} the four channels start at
         */
        private static void accumulateTap(int argb, float weight, boolean weightByAlpha, float[] out, int offset) {
            int alpha = ColorMath.alpha(argb);
            float colourWeight = weightByAlpha ? weight * alpha / 255f : weight;

            out[offset] += weight * alpha;
            out[offset + 1] += colourWeight * ColorMath.red(argb);
            out[offset + 2] += colourWeight * ColorMath.green(argb);
            out[offset + 3] += colourWeight * ColorMath.blue(argb);
        }

    }

}

