package dev.simplified.image.pixel;

import dev.simplified.image.PixelBuffer;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.awt.AlphaComposite;
import java.awt.Color;
import java.awt.Composite;
import java.awt.Graphics2D;
import java.awt.Polygon;
import java.awt.geom.AffineTransform;
import java.awt.geom.Ellipse2D;
import java.awt.geom.QuadCurve2D;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.sameInstance;
import static org.hamcrest.Matchers.startsWith;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

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
     * Creates an array of one pixel repeated.
     *
     * @param count the number of pixels
     * @param argb the pixel, as it is to be held
     * @return the pixels
     */
    private static int[] uniform(int count, int argb) {
        int[] pixels = new int[count];
        Arrays.fill(pixels, argb);
        return pixels;
    }

    /**
     * Finds the pixels a draw covers: the draw made in opaque red on a transparent canvas, where a
     * pixel holds the red however many times the draw reaches it.
     *
     * @param width the canvas width
     * @param height the canvas height
     * @param draw the draw
     * @return whether each pixel, in row order, is covered
     */
    private static boolean[] covered(int width, int height, Consumer<PixelGraphics> draw) {
        PixelBuffer canvas = PixelBuffer.canvas(width, height);
        PixelGraphics graphics = new PixelGraphics(canvas);
        graphics.setColor(new Color(RED, true));
        draw.accept(graphics);

        boolean[] covered = new boolean[width * height];
        for (int i = 0; i < covered.length; i++)
            covered[i] = canvas.data()[i] != 0;
        return covered;
    }

    /**
     * Draws in translucent red onto a target and asserts that every pixel the draw covers holds one
     * composite of the colour and every other keeps what it held.
     *
     * @param label the draw's name, for the failure message
     * @param target the buffer drawn onto, holding {@code under} throughout
     * @param under the pixel the target holds before the draw
     * @param laid the pixel a covered pixel holds after one composite
     * @param draw the draw
     */
    private static void assertLaysEachCoveredPixelOnce(String label, PixelBuffer target, int under, int laid, Consumer<PixelGraphics> draw) {
        boolean[] covered = covered(target.width(), target.height(), draw);
        PixelGraphics graphics = new PixelGraphics(target);
        graphics.setColor(new Color(TRANSLUCENT_RED, true));
        draw.accept(graphics);

        int count = 0;
        for (int i = 0; i < covered.length; i++) {
            String at = "%s at (%d, %d)".formatted(label, i % target.width(), i / target.width());
            assertThat(at, Integer.toHexString(target.data()[i]), is(Integer.toHexString(covered[i] ? laid : under)));
            if (covered[i])
                count++;
        }
        assertThat(label + " covers a pixel", count > 0, is(true));
    }

    /**
     * Collects the colour draws the translucent fill tests make on an eight by eight buffer: a
     * rectangle, laid at once, and draws that reach the buffer a row or a pixel at a time, among them
     * an outlined and a filled oval too narrow or too flat to span a row, which plot their centre.
     *
     * @return each draw by name
     */
    private static Map<String, Consumer<PixelGraphics>> fills() {
        Map<String, Consumer<PixelGraphics>> fills = new LinkedHashMap<>();
        fills.put("fillRect", g -> g.fillRect(1, 1, 5, 4));
        fills.put("a diagonal drawLine", g -> g.drawLine(0, 0, 7, 7));
        fills.put("fillOval", g -> g.fillOval(0, 0, 7, 7));
        fills.put("a drawOval one pixel wide", g -> g.drawOval(3, 1, 1, 5));
        fills.put("a fillOval of no height", g -> g.fillOval(1, 3, 5, 0));
        fills.put("fillArc", g -> g.fillArc(0, 0, 7, 7, 30, 200));
        fills.put("fillPolygon", g -> g.fillPolygon(new int[]{ 0, 7, 3 }, new int[]{ 0, 2, 7 }, 3));
        fills.put("fill of an ellipse", g -> g.fill(new Ellipse2D.Double(1, 1, 6, 5)));
        return fills;
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
    @DisplayName("a clip whose device edges pass the int range cuts a draw where the clip reaches")
    void aClipPastTheIntRangeCutsWhereItReaches() {
        // A clip's device edges are its origin, plus its size for the far edge, plus the translation.
        // Under the first two clips the far edge passes Integer.MAX_VALUE and the clip reaches past the
        // target's right and bottom edges, as the bounded clip beside it does. Under the last two the
        // near edge passes it as well, so the clip lies wholly right of or below the target, and the
        // draws start at the user point the translation takes to the target's origin, where an edge
        // that wrapped round would let them through. Each draw covers what it covers under the bounded
        // clip; the counts are Java2D's under the clip itself.
        record Clip(String label, int x, int y, Consumer<PixelGraphics> past, Consumer<PixelGraphics> bounded, int[] counts) { }
        List<Clip> clips = List.of(
            new Clip("setClip(1, 0, MAX_VALUE, 10)", 0, 0,
                g -> g.setClip(1, 0, Integer.MAX_VALUE, 10),
                g -> g.setClip(1, 0, 100, 10),
                new int[]{ 5, 13, 12, 12, 10 }),
            new Clip("clipRect(0, 0, MAX_VALUE, MAX_VALUE) then translate(1, 1)", 0, 0,
                g -> { g.clipRect(0, 0, Integer.MAX_VALUE, Integer.MAX_VALUE); g.translate(1, 1); },
                g -> { g.clipRect(0, 0, 100, 100); g.translate(1, 1); },
                new int[]{ 6, 16, 16, 16, 12 }),
            new Clip("translate(5, 0) then setClip(MAX_VALUE - 1, 0, MAX_VALUE, 10)", -5, 0,
                g -> { g.translate(5, 0); g.setClip(Integer.MAX_VALUE - 1, 0, Integer.MAX_VALUE, 10); },
                g -> { g.translate(5, 0); g.setClip(100, 0, 10, 10); },
                new int[]{ 0, 0, 0, 0, 0 }),
            new Clip("translate(0, 5) then setClip(0, MAX_VALUE - 1, 10, MAX_VALUE)", 0, -5,
                g -> { g.translate(0, 5); g.setClip(0, Integer.MAX_VALUE - 1, 10, Integer.MAX_VALUE); },
                g -> { g.translate(0, 5); g.setClip(0, 100, 10, 10); },
                new int[]{ 0, 0, 0, 0, 0 })
        );

        BufferedImage strip = new BufferedImage(6, 2, BufferedImage.TYPE_INT_ARGB);
        strip.setRGB(0, 0, 6, 2, uniform(12, RED), 0, 6);

        for (Clip clip : clips) {
            int x = clip.x();
            int y = clip.y();
            Map<String, Consumer<PixelGraphics>> draws = new LinkedHashMap<>();
            draws.put("drawLine", g -> g.drawLine(x, y, x + 5, y + 3));
            draws.put("drawOval", g -> g.drawOval(x, y, 6, 6));
            draws.put("fillRect", g -> g.fillRect(x, y, 4, 4));
            draws.put("clearRect", g -> {
                g.setBackground(new Color(BLUE, true));
                g.clearRect(x, y, 4, 4);
            });
            draws.put("drawImage", g -> g.drawImage(strip, x, y, null));

            int index = 0;
            for (Map.Entry<String, Consumer<PixelGraphics>> draw : draws.entrySet()) {
                String label = draw.getKey() + " under " + clip.label();
                boolean[] past = covered(8, 8, clip.past().andThen(draw.getValue()));
                boolean[] bounded = covered(8, 8, clip.bounded().andThen(draw.getValue()));
                int count = 0;
                for (boolean pixel : past)
                    count += pixel ? 1 : 0;

                assertArrayEquals(bounded, past, label);
                assertThat(label, count, is(clip.counts()[index++]));
            }
        }
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

    @Test
    @DisplayName("a translucent fill or shape composites over a canvas by NORMAL's formula")
    void aTranslucentFillCompositesOverACanvas() {
        // 80FF0000 over the premultiplied F00F000F stores F8870007, what Java2D's SrcOver stores on a
        // TYPE_INT_ARGB_PRE image.
        assertThat(Integer.toHexString(ColorMath.blend(TRANSLUCENT_RED, 0xF00F000F, BlendMode.NORMAL)), is("f8870007"));

        fills().forEach((label, draw) -> assertLaysEachCoveredPixelOnce(
            label, PixelBuffer.ofPremultiplied(uniform(64, 0xF00F000F), 8, 8), 0xF00F000F, 0xF8870007, draw
        ));
    }

    @Test
    @DisplayName("a translucent fill or shape composites over a straight buffer by STRAIGHT's formula")
    void aTranslucentFillCompositesOverAStraightBuffer() {
        // 80FF0000 over the straight F0100010 stores F88C0008, what Java2D's SrcOver stores on a
        // TYPE_INT_ARGB image.
        assertThat(Integer.toHexString(ColorMath.blend(TRANSLUCENT_RED, 0xF0100010, BlendMode.STRAIGHT)), is("f88c0008"));

        fills().forEach((label, draw) -> assertLaysEachCoveredPixelOnce(
            label, PixelBuffer.ofStraight(uniform(64, 0xF0100010), 8, 8), 0xF0100010, 0xF88C0008, draw
        ));
    }

    @Test
    @DisplayName("clearRect writes the background colour in the buffer's convention over what is there")
    void clearRectWritesTheBackgroundAsSrc() {
        // Java2D's clearRect fills with the background under Src, which writes 80FF0000 as 80800000 on
        // a TYPE_INT_ARGB_PRE image and as it is on a TYPE_INT_ARGB one.
        PixelBuffer canvas = PixelBuffer.ofPremultiplied(uniform(3, 0xF00F000F), 3, 1);
        PixelGraphics onCanvas = new PixelGraphics(canvas);
        onCanvas.setBackground(new Color(TRANSLUCENT_RED, true));
        onCanvas.clearRect(0, 0, 2, 1);
        assertArrayEquals(new int[]{ 0x80800000, 0x80800000, 0xF00F000F }, canvas.data());

        PixelBuffer straight = PixelBuffer.ofStraight(uniform(3, 0xF0100010), 3, 1);
        PixelGraphics onStraight = new PixelGraphics(straight);
        onStraight.setBackground(new Color(TRANSLUCENT_RED, true));
        onStraight.clearRect(0, 0, 2, 1);
        assertArrayEquals(new int[]{ TRANSLUCENT_RED, TRANSLUCENT_RED, 0xF0100010 }, straight.data());
    }

    @Test
    @DisplayName("an outline or shape composites each pixel it covers once, however often it reaches one")
    void anOutlineCompositesEachPixelOnce() {
        // Each draw reaches a pixel more than once: corners and vertices two edges share, the points
        // on an oval's axes, the rows a round rectangle's body shares with its corner bands, a node two
        // spans share, repeated arc samples, the joins of a path's lines and of a curve's segments. One
        // NORMAL composite of 80FF0000 over transparency is 80800000; a second over that would make it
        // C0C00000. An overwrite there stores the colour premultiplied, 80800000 too, so each draw is
        // made over a canvas's F00F000F and a straight F0100010 as well, where one composite stores
        // F8870007 and F88C0008, an overwrite 80800000 and 80FF0000, and a second composite neither.
        int once = ColorMath.blend(TRANSLUCENT_RED, 0, BlendMode.NORMAL);
        assertThat(Integer.toHexString(once), is("80800000"));
        assertThat(Integer.toHexString(ColorMath.blend(TRANSLUCENT_RED, 0xF00F000F, BlendMode.NORMAL)), is("f8870007"));
        assertThat(Integer.toHexString(ColorMath.blend(TRANSLUCENT_RED, 0xF0100010, BlendMode.STRAIGHT)), is("f88c0008"));

        Map<String, Consumer<PixelGraphics>> draws = new LinkedHashMap<>();
        draws.put("drawRect", g -> g.drawRect(1, 1, 7, 5));
        draws.put("drawRect without width", g -> g.drawRect(2, 1, 0, 6));
        draws.put("drawRect without height", g -> g.drawRect(1, 3, 7, 0));
        draws.put("drawRoundRect", g -> g.drawRoundRect(0, 0, 9, 8, 4, 4));
        draws.put("drawRoundRect with both arc sizes below 2", g -> g.drawRoundRect(1, 1, 7, 5, 1, 1));
        draws.put("fillRoundRect", g -> g.fillRoundRect(0, 0, 9, 8, 4, 4));
        draws.put("drawOval", g -> g.drawOval(0, 0, 9, 7));
        draws.put("drawPolyline", g -> g.drawPolyline(new int[]{ 0, 6, 6, 1 }, new int[]{ 0, 0, 6, 9 }, 4));
        draws.put("drawPolygon", g -> g.drawPolygon(new int[]{ 0, 8, 4 }, new int[]{ 1, 1, 9 }, 3));
        draws.put("drawArc", g -> g.drawArc(0, 0, 9, 9, 0, 270));
        draws.put("fill of a notched polygon", g -> g.fill(new Polygon(new int[]{ 0, 4, 8, 8, 0 }, new int[]{ 0, 3, 0, 8, 8 }, 5)));
        draws.put("draw of a polygon", g -> g.draw(new Polygon(new int[]{ 0, 8, 4 }, new int[]{ 1, 1, 9 }, 3)));
        draws.put("draw of a quadratic curve", g -> g.draw(new QuadCurve2D.Double(0, 9, 4, -5, 9, 9)));
        draws.put("draw of an ellipse", g -> g.draw(new Ellipse2D.Double(0, 0, 9, 9)));
        draws.put("drawPolygon under a translation and a clip", g -> {
            g.translate(1, 1);
            g.clipRect(0, 0, 5, 9);
            g.drawPolygon(new int[]{ -1, 7, 3 }, new int[]{ 0, 0, 7 }, 3);
        });
        draws.forEach((label, draw) -> {
            assertLaysEachCoveredPixelOnce(label, PixelBuffer.canvas(10, 10), 0, once, draw);
            assertLaysEachCoveredPixelOnce(label + " over a canvas", PixelBuffer.ofPremultiplied(uniform(100, 0xF00F000F), 10, 10), 0xF00F000F, 0xF8870007, draw);
            assertLaysEachCoveredPixelOnce(label + " over a straight buffer", PixelBuffer.ofStraight(uniform(100, 0xF0100010), 10, 10), 0xF0100010, 0xF88C0008, draw);
        });

        // A draw covers the pixels it reaches: a frame round the whole canvas keeps both the last pixel
        // of each row and the first of the next, which sit side by side in the pixel array, as does a
        // fill wider and taller than the canvas, whose rows each end where the next begins; a diagonal
        // line keeps each of its single pixels, and a shallow one each pixel Bresenham's walk plots
        // after the one before it on a row - x 0 to 2 on row 0, 3 to 6 on row 1, 7 to 9 on row 2.
        boolean[] frame = covered(10, 10, g -> g.drawRect(0, 0, 9, 9));
        boolean[] whole = covered(10, 10, g -> g.fillPolygon(new int[]{ -5, 20, 20, -5 }, new int[]{ -5, -5, 20, 20 }, 4));
        boolean[] diagonal = covered(10, 10, g -> g.drawLine(0, 0, 9, 9));
        boolean[] shallow = covered(10, 10, g -> g.drawLine(0, 0, 9, 2));
        for (int i = 0; i < 100; i++) {
            int x = i % 10;
            int y = i / 10;
            assertThat("frame at (%d, %d)".formatted(x, y), frame[i], is(x == 0 || x == 9 || y == 0 || y == 9));
            assertThat("whole canvas at (%d, %d)".formatted(x, y), whole[i], is(true));
            assertThat("diagonal at (%d, %d)".formatted(x, y), diagonal[i], is(x == y));
            assertThat("shallow line at (%d, %d)".formatted(x, y), shallow[i], is(y == 0 && x <= 2 || y == 1 && x >= 3 && x <= 6 || y == 2 && x >= 7));
        }
    }

    @Test
    @DisplayName("setComposite takes SrcOver alone and refuses every composite it would not honour")
    void setCompositeRefusesWhatItCannotHonour() {
        PixelBuffer target = PixelBuffer.ofPremultiplied(uniform(2, 0xF00F000F), 2, 1);
        PixelGraphics graphics = new PixelGraphics(target);
        assertThat(graphics.getComposite(), is(sameInstance(AlphaComposite.SrcOver)));

        List<Composite> refused = List.of(
            AlphaComposite.Src,
            AlphaComposite.DstOver,
            AlphaComposite.SrcOver.derive(0.5f),
            (source, destination, hints) -> null
        );
        for (Composite composite : refused) {
            UnsupportedOperationException thrown = assertThrows(UnsupportedOperationException.class, () -> graphics.setComposite(composite));
            assertThat(thrown.getMessage(), startsWith("Only AlphaComposite.SrcOver is supported, not '"));
            assertThat(graphics.getComposite(), is(sameInstance(AlphaComposite.SrcOver)));
        }

        // SrcOver is taken, asked for either way, and a copy and setPaintMode hand it back.
        graphics.setComposite(AlphaComposite.SrcOver);
        graphics.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f));
        assertThat(graphics.getComposite(), is(sameInstance(AlphaComposite.SrcOver)));
        assertThat(((Graphics2D) graphics.create()).getComposite(), is(sameInstance(AlphaComposite.SrcOver)));
        graphics.setPaintMode();
        assertThat(graphics.getComposite(), is(sameInstance(AlphaComposite.SrcOver)));

        // After the refusals a colour and an image still lay source-over, neither written as Src would
        // write it (80800000).
        graphics.setColor(new Color(TRANSLUCENT_RED, true));
        graphics.fillRect(0, 0, 1, 1);
        graphics.drawImage(image(TRANSLUCENT_RED), 1, 0, null);
        assertThat(Integer.toHexString(target.getPixel(0, 0)), is("f8870007"));
        assertThat(Integer.toHexString(target.getPixel(1, 0)), is("f8870007"));
    }

}
