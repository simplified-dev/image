package dev.simplified.image.pixel;

import dev.simplified.image.PixelBuffer;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

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
        PixelBuffer buffer = PixelBuffer.canvas(width, 1);
        buffer.fill(argb);
        return buffer;
    }

    /**
     * Draws an image at the origin through one of the five drawImage forms that lay it: the plain
     * form, the scaled form at the image's size, the sub-rectangle form over the whole image, the
     * transform form with a scale a hair over one, and the form taking an image filter.
     *
     * @param graphics the graphics drawn through
     * @param image the image drawn
     * @param form which form draws it, from 0 to 4
     */
    private static void draw(PixelGraphics graphics, BufferedImage image, int form) {
        int width = image.getWidth();
        int height = image.getHeight();

        switch (form) {
            case 0 -> graphics.drawImage(image, 0, 0, null);
            case 1 -> graphics.drawImage(image, 0, 0, width, height, null);
            case 2 -> graphics.drawImage(image, 0, 0, width, height, 0, 0, width, height, null);
            case 3 -> graphics.drawImage(image, AffineTransform.getScaleInstance(1, 1.0001), null);
            default -> graphics.drawImage(image, null, 0, 0);
        }
    }

    @Test
    @DisplayName("every drawImage form, clipped and unclipped, lays a premultiplied image onto a canvas as the premultiplied colour it holds")
    void drawImageLaysAPremultipliedImageOntoACanvas() {
        // 80800000 is straight 80FF0000 premultiplied; over F00F000F it stores F8870007, what Java2D's
        // SrcOver stores on a TYPE_INT_ARGB_PRE image. 19020000 lays as held to F110000E, where
        // dividing it back to straight colour and laying that rounds twice, to F10F000E.
        BufferedImage image = PixelBuffer.ofPremultiplied(new int[]{ 0x80800000, 0x19020000 }, 2, 1).toBufferedImage();
        assertThat(image.getType(), is(BufferedImage.TYPE_INT_ARGB_PRE));
        assertThat(ColorMath.blend(0x19020000, 0xF00F000F, BlendMode.PREMULTIPLIED), is(0xF110000E));

        for (boolean clipped : new boolean[]{ false, true })
            for (int form = 0; form < 5; form++) {
                String label = "form %d, %s".formatted(form, clipped ? "clipped" : "unclipped");
                PixelBuffer target = PixelBuffer.ofPremultiplied(new int[]{ 0xF00F000F, 0xF00F000F, 0xF00F000F }, 3, 1);
                PixelGraphics graphics = new PixelGraphics(target);
                if (clipped)
                    graphics.setClip(1, 0, 2, 1);
                draw(graphics, image, form);

                assertThat(label, Integer.toHexString(target.getPixel(0, 0)), is(Integer.toHexString(clipped ? 0xF00F000F : 0xF8870007)));
                assertThat(label, Integer.toHexString(target.getPixel(1, 0)), is(Integer.toHexString(0xF110000E)));
                assertThat(label, Integer.toHexString(target.getPixel(2, 0)), is(Integer.toHexString(0xF00F000F)));
            }
    }

    @Test
    @DisplayName("every drawImage form, clipped and unclipped, composites onto a straight buffer as straight colour")
    void drawImageOntoAStraightBufferCompositesStraight() {
        // Over the straight F0100010, 80FF0000 stores STRAIGHT's F88C0008, what Java2D's SrcOver
        // stores on a TYPE_INT_ARGB image; a premultiplied 80800000 is that colour, and lays the same.
        // NORMAL's formula on the straight pixel would store F8880008.
        BufferedImage straight = image(TRANSLUCENT_RED, TRANSLUCENT_RED);
        BufferedImage premultiplied = PixelBuffer.ofPremultiplied(new int[]{ 0x80800000, 0x80800000 }, 2, 1).toBufferedImage();

        for (BufferedImage image : new BufferedImage[]{ straight, premultiplied })
            for (boolean clipped : new boolean[]{ false, true })
                for (int form = 0; form < 5; form++) {
                    String label = "form %d, %s, image type %d".formatted(form, clipped ? "clipped" : "unclipped", image.getType());
                    PixelBuffer target = PixelBuffer.ofStraight(new int[]{ 0xF0100010, 0xF0100010, 0xF0100010 }, 3, 1);
                    PixelGraphics graphics = new PixelGraphics(target);
                    if (clipped)
                        graphics.setClip(1, 0, 2, 1);
                    draw(graphics, image, form);

                    assertThat(label, Integer.toHexString(target.getPixel(0, 0)), is(Integer.toHexString(clipped ? 0xF0100010 : 0xF88C0008)));
                    assertThat(label, Integer.toHexString(target.getPixel(1, 0)), is(Integer.toHexString(0xF88C0008)));
                    assertThat(label, Integer.toHexString(target.getPixel(2, 0)), is(Integer.toHexString(0xF0100010)));
                }
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
    @DisplayName("every drawImage form keeps the rows a clip cuts above and below it")
    void everyFormHonoursTheClipsRows() {
        // A column of three translucent reds over blue, clipped to its middle row: the rows above and
        // below keep the blue and the middle one takes the composite.
        BufferedImage column = new BufferedImage(1, 3, BufferedImage.TYPE_INT_ARGB);
        for (int y = 0; y < 3; y++)
            column.setRGB(0, y, TRANSLUCENT_RED);
        int laid = ColorMath.blend(TRANSLUCENT_RED, BLUE, BlendMode.NORMAL);

        for (int form = 0; form < 5; form++) {
            PixelBuffer target = PixelBuffer.canvas(1, 3);
            target.fill(BLUE);
            PixelGraphics graphics = new PixelGraphics(target);
            graphics.setClip(0, 1, 1, 1);
            draw(graphics, column, form);

            assertThat("form " + form + " at y=0", Integer.toHexString(target.getPixel(0, 0)), is(Integer.toHexString(BLUE)));
            assertThat("form " + form + " at y=1", Integer.toHexString(target.getPixel(0, 1)), is(Integer.toHexString(laid)));
            assertThat("form " + form + " at y=2", Integer.toHexString(target.getPixel(0, 2)), is(Integer.toHexString(BLUE)));
        }
    }

    @Test
    @DisplayName("a sub-rectangle or transformed draw far larger than the target samples only what lands on it")
    void aDrawFarLargerThanTheTargetSamplesWhatLandsOnIt() {
        // Two by two texels stretched over 131072 pixels a side, whose pixel count overflows an int,
        // placed so the texel boundaries cross the middle of a four by four target: each quadrant
        // takes one texel. Clipped to the centre under a translation of one, only the centre four take
        // them and the rest keep the backdrop.
        BufferedImage texels = new BufferedImage(2, 2, BufferedImage.TYPE_INT_ARGB);
        texels.setRGB(0, 0, RED);
        texels.setRGB(1, 0, GREEN);
        texels.setRGB(0, 1, BLUE);
        texels.setRGB(1, 1, 0xFFFFFFFF);
        int[] quadrants = { RED, RED, GREEN, GREEN, RED, RED, GREEN, GREEN, BLUE, BLUE, 0xFFFFFFFF, 0xFFFFFFFF, BLUE, BLUE, 0xFFFFFFFF, 0xFFFFFFFF };
        int[] centre = { 0, 0, 0, 0, 0, RED, GREEN, 0, 0, BLUE, 0xFFFFFFFF, 0, 0, 0, 0, 0 };

        for (boolean clipped : new boolean[]{ false, true })
            for (boolean transformed : new boolean[]{ false, true }) {
                String label = "%s, %s".formatted(transformed ? "transformed" : "sub-rectangle", clipped ? "clipped" : "unclipped");
                PixelBuffer target = PixelBuffer.canvas(4, 4);
                PixelGraphics graphics = new PixelGraphics(target);
                int shift = 0;
                if (clipped) {
                    graphics.translate(1, 1);
                    graphics.setClip(0, 0, 2, 2);
                    shift = 1;
                }

                // The sub-rectangle form maps a pixel to a texel by flooring, the boundary 65534 pixels
                // in; the transform form by rounding, the boundary 32766 pixels in.
                if (transformed)
                    graphics.drawImage(texels, new AffineTransform(65536, 0, 0, 65536, -32766 - shift, -32766 - shift), null);
                else
                    graphics.drawImage(texels, -65534 - shift, -65534 - shift, 65538 - shift, 65538 - shift, 0, 0, 2, 2, null);

                assertArrayEquals(clipped ? centre : quadrants, target.data(), label);
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
    @DisplayName("filled and cleared rectangles draw only inside the clip, read under the current translation")
    void rectanglesHonourTheClip() {
        PixelBuffer target = PixelBuffer.create(6, 1);
        PixelGraphics graphics = new PixelGraphics(target);
        graphics.translate(1, 0);
        graphics.setClip(1, 0, 2, 1);
        graphics.setColor(new java.awt.Color(RED, true));
        graphics.fillRect(-1, 0, 6, 1);

        assertThat(target.getPixel(1, 0), is(0));
        assertThat(target.getPixel(2, 0), is(RED));
        assertThat(target.getPixel(3, 0), is(RED));
        assertThat(target.getPixel(4, 0), is(0));

        graphics.setBackground(new java.awt.Color(BLUE, true));
        graphics.clearRect(-1, 0, 6, 1);
        assertThat(target.getPixel(2, 0), is(BLUE));
        assertThat(target.getPixel(4, 0), is(0));
    }

    @Test
    @DisplayName("lines, ovals and polygons draw only inside the clip")
    void shapesHonourTheClip() {
        PixelBuffer target = PixelBuffer.create(8, 8);
        PixelGraphics graphics = new PixelGraphics(target);
        graphics.setClip(0, 0, 4, 4);
        graphics.setColor(new java.awt.Color(RED, true));
        graphics.drawLine(0, 0, 7, 7);
        graphics.drawLine(0, 6, 7, 6);
        graphics.fillOval(0, 0, 7, 7);
        graphics.fillPolygon(new int[]{ 0, 7, 7 }, new int[]{ 0, 0, 7 }, 3);

        for (int y = 0; y < 8; y++)
            for (int x = 0; x < 8; x++)
                if (x >= 4 || y >= 4)
                    assertThat("(" + x + ", " + y + ")", target.getPixel(x, y), is(0));
        assertThat(target.getPixel(0, 0), is(RED));
    }

    @Test
    @DisplayName("copyArea writes only inside the clip")
    void copyAreaHonoursTheClip() {
        PixelBuffer target = PixelBuffer.create(4, 1);
        target.setPixel(0, 0, RED);
        target.setPixel(1, 0, RED);
        PixelGraphics graphics = new PixelGraphics(target);
        graphics.setClip(2, 0, 1, 1);
        graphics.copyArea(0, 0, 2, 1, 2, 0);

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
