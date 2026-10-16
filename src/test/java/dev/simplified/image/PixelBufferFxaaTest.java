package dev.simplified.image;

import dev.simplified.image.pixel.AlphaMode;
import dev.simplified.image.pixel.ColorMath;
import dev.simplified.image.pixel.PixelBuffer;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.everyItem;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.lessThanOrEqualTo;

/**
 * Holds FXAA to the picture it is given: a softened edge keeps the colour of the shape it belongs to
 * at its own coverage in either alpha convention, the two conventions smooth one picture alike, and
 * an opaque picture stays opaque.
 */
class PixelBufferFxaaTest {

    /** The one colour every test shape is drawn in. */
    private static final int FACE = 0xFFFFE070;

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

}
