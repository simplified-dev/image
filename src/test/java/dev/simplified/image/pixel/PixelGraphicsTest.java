package dev.simplified.image.pixel;

import dev.simplified.image.PixelBuffer;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

class PixelGraphicsTest {

    private static final int RED = 0xFFFF0000;
    private static final int GREEN = 0xFF00FF00;
    private static final int BLUE = 0xFF0000FF;
    private static final int TRANSLUCENT_RED = 0x80FF0000;

    private static BufferedImage image(int... argb) {
        BufferedImage image = new BufferedImage(argb.length, 1, BufferedImage.TYPE_INT_ARGB);
        for (int x = 0; x < argb.length; x++)
            image.setRGB(x, 0, argb[x]);
        return image;
    }

    private static PixelBuffer filled(int width, int argb) {
        PixelBuffer buffer = PixelBuffer.create(width, 1);
        buffer.fill(argb);
        return buffer;
    }

    @Test
    @DisplayName("the rectangle form blends a translucent pixel over the destination")
    void rectangleFormBlends() {
        PixelBuffer target = filled(2, BLUE);
        new PixelGraphics(target).drawImage(image(TRANSLUCENT_RED), 0, 0, 1, 1, 0, 0, 1, 1, null);

        assertThat(target.getPixel(0, 0), is(ColorMath.blend(TRANSLUCENT_RED, BLUE, BlendMode.NORMAL)));
        assertThat(target.getPixel(1, 0), is(BLUE));
    }

    @Test
    @DisplayName("the rectangle form mirrors when one corner pair is reversed, and not when both are")
    void rectangleFormKeepsItsFlip() {
        PixelBuffer flipped = PixelBuffer.create(2, 1);
        new PixelGraphics(flipped).drawImage(image(RED, GREEN), 0, 0, 2, 1, 2, 0, 0, 1, null);
        assertThat(flipped.getPixel(0, 0), is(GREEN));
        assertThat(flipped.getPixel(1, 0), is(RED));

        PixelBuffer doubled = PixelBuffer.create(2, 1);
        new PixelGraphics(doubled).drawImage(image(RED, GREEN), 2, 0, 0, 1, 2, 0, 0, 1, null);
        assertThat(doubled.getPixel(0, 0), is(RED));
        assertThat(doubled.getPixel(1, 0), is(GREEN));
    }

    @Test
    @DisplayName("the transform form blends a translucent pixel over the destination")
    void transformFormBlends() {
        PixelBuffer target = filled(2, BLUE);
        new PixelGraphics(target).drawImage(image(TRANSLUCENT_RED), AffineTransform.getTranslateInstance(1, 0), null);

        assertThat(target.getPixel(0, 0), is(BLUE));
        assertThat(target.getPixel(1, 0), is(ColorMath.blend(TRANSLUCENT_RED, BLUE, BlendMode.NORMAL)));
    }

    @Test
    @DisplayName("every drawImage form draws only inside the clip")
    void everyFormHonoursTheClip() {
        BufferedImage red = image(RED, RED, RED, RED);

        for (int form = 0; form < 5; form++) {
            PixelBuffer target = PixelBuffer.create(4, 1);
            PixelGraphics graphics = new PixelGraphics(target);
            graphics.setClip(1, 0, 2, 1);

            switch (form) {
                case 0 -> graphics.drawImage(red, 0, 0, null);
                case 1 -> graphics.drawImage(red, 0, 0, 4, 1, null);
                case 2 -> graphics.drawImage(red, 0, 0, 4, 1, 0, 0, 4, 1, null);
                case 3 -> graphics.drawImage(red, AffineTransform.getScaleInstance(1, 1.0001), null);
                default -> graphics.drawImage(red, null, 0, 0);
            }

            assertThat("form " + form + " at x=0", target.getPixel(0, 0), is(0));
            assertThat("form " + form + " at x=1", target.getPixel(1, 0), is(RED));
            assertThat("form " + form + " at x=2", target.getPixel(2, 0), is(RED));
            assertThat("form " + form + " at x=3", target.getPixel(3, 0), is(0));
        }
    }

    @Test
    @DisplayName("the clip is read in user space under the current translation")
    void clipFollowsTheTranslation() {
        PixelBuffer target = PixelBuffer.create(4, 1);
        PixelGraphics graphics = new PixelGraphics(target);
        graphics.translate(2, 0);
        graphics.setClip(0, 0, 1, 1);
        graphics.drawImage(image(RED, RED), 0, 0, null);

        assertThat(target.getPixel(2, 0), is(RED));
        assertThat(target.getPixel(3, 0), is(0));
    }

    @Test
    @DisplayName("a clipped scaled draw samples the source as the unclipped one does")
    void clippedScaledDrawSamplesAsUnclipped() {
        BufferedImage source = image(RED, GREEN, BLUE);
        PixelBuffer unclipped = PixelBuffer.create(5, 1);
        new PixelGraphics(unclipped).drawImage(source, 0, 0, 5, 1, null);

        PixelBuffer clipped = PixelBuffer.create(5, 1);
        PixelGraphics graphics = new PixelGraphics(clipped);
        graphics.setClip(0, 0, 5, 1);
        graphics.drawImage(source, 0, 0, 5, 1, null);

        assertThat(clipped.data(), is(unclipped.data()));
    }

}
