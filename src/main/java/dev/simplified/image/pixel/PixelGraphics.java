package dev.simplified.image.pixel;

import dev.simplified.image.PixelBuffer;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.awt.*;
import java.awt.font.FontRenderContext;
import java.awt.font.GlyphVector;
import java.awt.geom.AffineTransform;
import java.awt.geom.PathIterator;
import java.awt.image.BufferedImage;
import java.awt.image.BufferedImageOp;
import java.awt.image.ImageObserver;
import java.awt.image.RenderedImage;
import java.awt.image.renderable.RenderableImage;
import java.text.AttributedCharacterIterator;
import java.util.Arrays;
import java.util.Map;

/**
 * A {@link Graphics2D} implementation backed by a {@link PixelBuffer} - all drawing operations
 * write directly into the buffer's {@code int[]} pixel array with no AWT rendering pipeline.
 * <p>
 * Supports 2D primitives (rectangles, rounded rectangles, lines, ovals, arcs, polygons),
 * image blitting, rectangular clipping, and translation.
 * <p>
 * Every draw composites source-over: {@link AlphaComposite#SrcOver} is the one composite it takes, and
 * {@link #setComposite(Composite)} refuses any other with {@link UnsupportedOperationException}, as
 * {@link #setXORMode(Color)} refuses XOR. A colour draw lays its colour, straight colour, as
 * {@link PixelBuffer#fillRect(int, int, int, int, int, BlendMode)} lays it under
 * {@link BlendMode#NORMAL}, whose javadoc says which formula a buffer takes: {@code NORMAL}'s on
 * premultiplied colour or a buffer whose alpha carries no data, and {@link BlendMode#STRAIGHT}'s on
 * straight colour whose alpha does. That is the compositing Java2D's {@code SrcOver} does on a
 * {@link BufferedImage#TYPE_INT_ARGB_PRE} image and on a {@link BufferedImage#TYPE_INT_ARGB} image, in
 * this library's rounding: each channel within 1 of Java2D's on the first, while on the second Java2D
 * rounds the destination through 8-bit premultiplied colour and loses colour this keeps. A translucent
 * {@code 80FF0000} over a canvas's {@code F00F000F} stores {@code F8870007}, as Java2D does. An outline or
 * a shape gathers every pixel it reaches before it lays any, so a pixel its edges, vertices or
 * segments reach more than once takes the colour once. {@link #clearRect(int, int, int, int)} clears
 * rather than draws: it writes the background colour in the buffer's convention over what is there,
 * as Java2D's does.
 * <p>
 * An image draw reads the image in the convention it holds, through
 * {@link PixelBuffer#wrap(BufferedImage)}, and lays it source-over in the convention the target holds,
 * as {@link PixelBuffer#blit(PixelBuffer, int, int)} lays it, so a
 * {@link BufferedImage#TYPE_INT_ARGB_PRE} image lays as the premultiplied colour it holds. A
 * sub-rectangle or transformed draw samples the image into a buffer of the image's convention first,
 * only over the part of its destination that lands on the target inside the clip; a scaled draw under
 * a clip samples its whole destination so, and a clipped draw lays only the part inside the clip.
 * <p>
 * Text rendering ({@code drawString}) throws {@link UnsupportedOperationException} at this
 * level - subclasses that provide a glyph source should override it.
 *
 * @see PixelBuffer#createGraphics()
 */
public class PixelGraphics extends Graphics2D {

    private final @NotNull PixelBuffer target;
    private @NotNull Font currentFont;
    private int colorArgb;
    private int backgroundArgb;
    private int translateX;
    private int translateY;
    private @Nullable Rectangle clip;
    private @NotNull RenderingHints hints;
    private @NotNull Paint paint;
    private @NotNull Stroke stroke;
    private @NotNull AffineTransform transform;

    /**
     * Creates a new pixel graphics context for the given buffer.
     *
     * @param target the pixel buffer to draw onto
     */
    public PixelGraphics(@NotNull PixelBuffer target) {
        this.target = target;
        this.currentFont = new Font(Font.SANS_SERIF, Font.PLAIN, 12);
        this.colorArgb = ColorMath.BLACK;
        this.backgroundArgb = ColorMath.TRANSPARENT;
        this.translateX = 0;
        this.translateY = 0;
        this.clip = null;
        this.hints = new RenderingHints(null);
        this.paint = Color.BLACK;
        this.stroke = new BasicStroke();
        this.transform = new AffineTransform();
    }

    /**
     * Copy constructor for {@link #create()}.
     */
    protected PixelGraphics(@NotNull PixelGraphics source) {
        this.target = source.target;
        this.currentFont = source.currentFont;
        this.colorArgb = source.colorArgb;
        this.backgroundArgb = source.backgroundArgb;
        this.translateX = source.translateX;
        this.translateY = source.translateY;
        this.clip = source.clip != null ? new Rectangle(source.clip) : null;
        this.hints = (RenderingHints) source.hints.clone();
        this.paint = source.paint;
        this.stroke = source.stroke;
        this.transform = new AffineTransform(source.transform);
    }

    /**
     * The underlying pixel buffer this graphics context draws onto.
     *
     * @return the target buffer
     */
    public @NotNull PixelBuffer target() {
        return this.target;
    }

    /**
     * The current drawing color as a packed ARGB int.
     *
     * @return the color ARGB value
     */
    public int colorArgb() {
        return this.colorArgb;
    }

    /**
     * The current translation X offset.
     *
     * @return the X translation
     */
    protected int translateX() {
        return this.translateX;
    }

    /**
     * The current translation Y offset.
     *
     * @return the Y translation
     */
    protected int translateY() {
        return this.translateY;
    }

    // --- pixel helper ---

    /**
     * Whether a device pixel lies inside the clip, which is read in user space under the current
     * translation. Every pixel is inside when no clip is set.
     *
     * @param x the device x coordinate
     * @param y the device y coordinate
     * @return {@code true} when the pixel may be drawn
     */
    private boolean inClip(int x, int y) {
        return this.clip == null || this.clip.contains(x - this.translateX, y - this.translateY);
    }

    /**
     * The pixels one outline or shape covers, gathered while it is rasterized and laid once it is
     * whole, so that a pixel the draw reaches more than once takes its colour once.
     * <p>
     * Each added rectangle is cut to the target and the clip, as
     * {@link #drawableArea(int, int, int, int)} cuts it, and held as one run per row: the span of
     * indices the row covers in the target's pixel array, its start in the high half of a {@code long}
     * and its end in the low half, so that sorting the runs orders them by row and then by column. A
     * run that starts where the last one added ends, on the same row, lengthens it instead, so a draw
     * that plots a row's pixels one after another costs one run for them, and a draw costs memory in
     * its runs rather than in the area it spans.
     */
    private final class Coverage {

        private long @NotNull [] runs = new long[16];
        private int count;

        /**
         * Adds one device pixel.
         *
         * @param x the device x coordinate
         * @param y the device y coordinate
         */
        void add(int x, int y) {
            add(x, y, 1, 1);
        }

        /**
         * Adds a device-space rectangle, one run for each of its rows that lands on the target inside
         * the clip.
         *
         * @param x the device x coordinate of the left edge
         * @param y the device y coordinate of the top edge
         * @param width the rectangle width
         * @param height the rectangle height
         */
        void add(int x, int y, int width, int height) {
            Rectangle area = drawableArea(x, y, width, height);
            int stride = target.width();

            for (int row = area.y; row < area.y + area.height; row++) {
                int start = row * stride + area.x;
                int end = start + area.width;

                if (area.x > 0 && this.count > 0 && (int) this.runs[this.count - 1] == start)
                    this.runs[this.count - 1] = (this.runs[this.count - 1] & 0xFFFFFFFF00000000L) | end;
                else {
                    if (this.count == this.runs.length)
                        this.runs = Arrays.copyOf(this.runs, this.count * 2);
                    this.runs[this.count++] = ((long) start << 32) | end;
                }
            }
        }

        /**
         * Lays a colour once on every covered pixel, as
         * {@link PixelBuffer#fillRect(int, int, int, int, int, BlendMode)} lays it under
         * {@link BlendMode#NORMAL}: the runs of a row that overlap or touch are merged, and each merged
         * stretch is laid with one call.
         *
         * @param argb the colour, straight
         */
        void lay(int argb) {
            if (this.count == 0) return;
            Arrays.sort(this.runs, 0, this.count);
            int stride = target.width();
            int start = (int) (this.runs[0] >>> 32);
            int end = (int) this.runs[0];

            for (int i = 1; i < this.count; i++) {
                int nextStart = (int) (this.runs[i] >>> 32);
                int nextEnd = (int) this.runs[i];

                if (nextStart <= end && nextStart / stride == start / stride)
                    end = Math.max(end, nextEnd);
                else {
                    target.fillRect(start % stride, start / stride, end - start, 1, argb, BlendMode.NORMAL);
                    start = nextStart;
                    end = nextEnd;
                }
            }

            target.fillRect(start % stride, start / stride, end - start, 1, argb, BlendMode.NORMAL);
        }

    }

    /**
     * Lays an image onto the target with its top-left corner at device coordinates {@code (x, y)},
     * source-over in the target's convention as {@link PixelBuffer#blit(PixelBuffer, int, int)} lays
     * it. With a clip set, only the part of the image inside the clip is laid: the image is cut to the
     * clip and the target, as {@link #drawableArea(int, int, int, int)} cuts it, and the cut part
     * blitted.
     *
     * @param image the image pixels, in the convention they are held in
     * @param x the device x coordinate of the top-left corner
     * @param y the device y coordinate of the top-left corner
     */
    private void layImage(@NotNull PixelBuffer image, int x, int y) {
        if (this.clip == null) {
            this.target.blit(image, x, y);
            return;
        }

        Rectangle area = drawableArea(x, y, image.width(), image.height());
        if (!area.isEmpty())
            this.target.blit(image.crop(area.x - x, area.y - y, area.width, area.height), area.x, area.y);
    }

    /**
     * Cuts a device-space rectangle to the target's bounds and, with a clip set, to the clip, read in
     * user space under the current translation. The clip's device edges are summed in {@code long}, so
     * a clip whose origin, size and translation pass the int range between them cuts where it reaches.
     *
     * @param x the device x coordinate of the left edge
     * @param y the device y coordinate of the top edge
     * @param width the rectangle width
     * @param height the rectangle height
     * @return the part of the rectangle a draw may write, empty where there is none
     */
    private @NotNull Rectangle drawableArea(int x, int y, int width, int height) {
        long x0 = Math.max(x, 0);
        long y0 = Math.max(y, 0);
        long x1 = Math.min(x + width, this.target.width());
        long y1 = Math.min(y + height, this.target.height());

        if (this.clip != null) {
            long clipX = (long) this.clip.x + this.translateX;
            long clipY = (long) this.clip.y + this.translateY;
            x0 = Math.max(x0, clipX);
            y0 = Math.max(y0, clipY);
            x1 = Math.min(x1, clipX + this.clip.width);
            y1 = Math.min(y1, clipY + this.clip.height);
        }

        return x0 < x1 && y0 < y1 ? new Rectangle((int) x0, (int) y0, (int) (x1 - x0), (int) (y1 - y0)) : new Rectangle();
    }

    /**
     * Creates a transparent buffer holding the convention {@code image} holds, for pixels sampled from
     * it to be held in as they are.
     *
     * @param image the image whose convention the buffer holds
     * @param width the buffer width
     * @param height the buffer height
     * @return a transparent buffer of that convention
     */
    private static @NotNull PixelBuffer blankLike(@NotNull PixelBuffer image, int width, int height) {
        return switch (image.alphaMode()) {
            case STRAIGHT -> PixelBuffer.straight(width, height);
            case PREMULTIPLIED -> PixelBuffer.canvas(width, height);
        };
    }

    // --- text rendering (subclass responsibility) ---

    @Override
    public void drawString(@NotNull String str, float x, float y) {
        this.drawString(str, Math.round(x), Math.round(y));
    }

    @Override
    public void drawString(@NotNull String str, int x, int y) {
        throw new UnsupportedOperationException("drawString requires a glyph source - use a subclass that provides one");
    }

    @Override
    public void drawString(@NotNull AttributedCharacterIterator iterator, int x, int y) {
        throw new UnsupportedOperationException("AttributedCharacterIterator text rendering is not supported");
    }

    @Override
    public void drawString(@NotNull AttributedCharacterIterator iterator, float x, float y) {
        throw new UnsupportedOperationException("AttributedCharacterIterator text rendering is not supported");
    }

    @Override
    public void drawGlyphVector(@NotNull GlyphVector gv, float x, float y) {
        throw new UnsupportedOperationException("GlyphVector rendering is not supported");
    }

    // --- font and metrics ---

    @Override
    public void setFont(@NotNull Font font) {
        this.currentFont = font;
    }

    @Override
    public @NotNull Font getFont() {
        return this.currentFont;
    }

    @Override
    public @NotNull FontMetrics getFontMetrics(@NotNull Font f) {
        BufferedImage temp = new BufferedImage(1, 1, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = temp.createGraphics();
        g.setFont(f);
        FontMetrics fm = g.getFontMetrics();
        g.dispose();
        return fm;
    }

    @Override
    public @NotNull FontMetrics getFontMetrics() {
        return getFontMetrics(this.currentFont);
    }

    @Override
    public @NotNull FontRenderContext getFontRenderContext() {
        return new FontRenderContext(null, true, false);
    }

    // --- color and paint ---

    @Override
    public void setColor(@NotNull Color c) {
        this.colorArgb = c.getRGB();
        this.paint = c;
    }

    @Override
    public @NotNull Color getColor() {
        return new Color(this.colorArgb, true);
    }

    @Override
    public void setPaint(@NotNull Paint paint) {
        this.paint = paint;
        if (paint instanceof Color c)
            this.colorArgb = c.getRGB();
    }

    @Override
    public @NotNull Paint getPaint() {
        return this.paint;
    }

    @Override
    public void setBackground(@NotNull Color color) {
        this.backgroundArgb = color.getRGB();
    }

    @Override
    public @NotNull Color getBackground() {
        return new Color(this.backgroundArgb, true);
    }

    // --- rectangles ---

    @Override
    public void fillRect(int x, int y, int width, int height) {
        Rectangle area = drawableArea(this.translateX + x, this.translateY + y, width, height);
        if (!area.isEmpty())
            this.target.fillRect(area.x, area.y, area.width, area.height, this.colorArgb, BlendMode.NORMAL);
    }

    @Override
    public void clearRect(int x, int y, int width, int height) {
        Rectangle area = drawableArea(this.translateX + x, this.translateY + y, width, height);
        if (!area.isEmpty())
            this.target.fillRect(area.x, area.y, area.width, area.height, this.backgroundArgb);
    }

    @Override
    public void drawRect(int x, int y, int width, int height) {
        int tx = this.translateX + x;
        int ty = this.translateY + y;
        Coverage coverage = new Coverage();
        coverage.add(tx, ty, width + 1, 1);
        coverage.add(tx, ty + height, width + 1, 1);
        coverage.add(tx, ty, 1, height + 1);
        coverage.add(tx + width, ty, 1, height + 1);
        coverage.lay(this.colorArgb);
    }

    @Override
    public void drawRoundRect(int x, int y, int width, int height, int arcWidth, int arcHeight) {
        int rx = arcWidth / 2;
        int ry = arcHeight / 2;
        int tx = this.translateX + x;
        int ty = this.translateY + y;
        Coverage coverage = new Coverage();

        // Straight edges
        coverage.add(tx + rx, ty, width - 2 * rx + 1, 1);
        coverage.add(tx + rx, ty + height, width - 2 * rx + 1, 1);
        coverage.add(tx, ty + ry, 1, height - 2 * ry + 1);
        coverage.add(tx + width, ty + ry, 1, height - 2 * ry + 1);

        // Corner arcs via midpoint ellipse
        drawCornerArcs(coverage, tx + rx, ty + ry, tx + width - rx, ty + height - ry, rx, ry);
        coverage.lay(this.colorArgb);
    }

    @Override
    public void fillRoundRect(int x, int y, int width, int height, int arcWidth, int arcHeight) {
        int rx = arcWidth / 2;
        int ry = arcHeight / 2;
        int tx = this.translateX + x;
        int ty = this.translateY + y;
        Coverage coverage = new Coverage();

        // Central body
        coverage.add(tx, ty + ry, width + 1, height - 2 * ry + 1);

        // Top and bottom bands with rounded corners via scanline
        fillCornerArcs(coverage, tx + rx, ty + ry, tx + width - rx, ty + height - ry, rx, ry);
        coverage.lay(this.colorArgb);
    }

    private void drawCornerArcs(@NotNull Coverage coverage, int cx1, int cy1, int cx2, int cy2, int rx, int ry) {
        if (rx <= 0 || ry <= 0) return;
        int x = 0, y = ry;
        long rx2 = (long) rx * rx, ry2 = (long) ry * ry;
        long err = ry2 - rx2 * (long) ry + rx2 / 4;

        while (ry2 * x <= rx2 * y) {
            coverage.add(cx2 + x, cy1 - y);
            coverage.add(cx1 - x, cy1 - y);
            coverage.add(cx2 + x, cy2 + y);
            coverage.add(cx1 - x, cy2 + y);
            x++;
            if (err < 0)
                err += ry2 * (2L * x + 1);
            else {
                y--;
                err += ry2 * (2L * x + 1) - rx2 * 2L * y;
            }
        }

        err = ry2 * (2L * x + 1) * (2L * x + 1) / 4 + rx2 * ((long)(y - 1) * (y - 1) - ry2);
        while (y >= 0) {
            coverage.add(cx2 + x, cy1 - y);
            coverage.add(cx1 - x, cy1 - y);
            coverage.add(cx2 + x, cy2 + y);
            coverage.add(cx1 - x, cy2 + y);
            y--;
            if (err > 0)
                err -= rx2 * (2L * y + 1);
            else {
                x++;
                err += ry2 * (2L * x + 1) - rx2 * (2L * y + 1);
            }
        }
    }

    private void fillCornerArcs(@NotNull Coverage coverage, int cx1, int cy1, int cx2, int cy2, int rx, int ry) {
        if (rx <= 0 || ry <= 0) return;
        int px = 0, py = ry;
        long rx2 = (long) rx * rx, ry2 = (long) ry * ry;
        long err = ry2 - rx2 * (long) ry + rx2 / 4;

        int lastPy = -1;
        while (ry2 * px <= rx2 * py) {
            if (py != lastPy) {
                coverage.add(cx1 - px, cy1 - py, cx2 - cx1 + 2 * px + 1, 1);
                coverage.add(cx1 - px, cy2 + py, cx2 - cx1 + 2 * px + 1, 1);
                lastPy = py;
            }
            px++;
            if (err < 0)
                err += ry2 * (2L * px + 1);
            else {
                py--;
                err += ry2 * (2L * px + 1) - rx2 * 2L * py;
            }
        }

        err = ry2 * (2L * px + 1) * (2L * px + 1) / 4 + rx2 * ((long)(py - 1) * (py - 1) - ry2);
        while (py >= 0) {
            if (py != lastPy) {
                coverage.add(cx1 - px, cy1 - py, cx2 - cx1 + 2 * px + 1, 1);
                coverage.add(cx1 - px, cy2 + py, cx2 - cx1 + 2 * px + 1, 1);
                lastPy = py;
            }
            py--;
            if (err > 0)
                err -= rx2 * (2L * py + 1);
            else {
                px++;
                err += ry2 * (2L * px + 1) - rx2 * (2L * py + 1);
            }
        }
    }

    // --- lines ---

    @Override
    public void drawLine(int x1, int y1, int x2, int y2) {
        Coverage coverage = new Coverage();
        drawLine(coverage, x1, y1, x2, y2);
        coverage.lay(this.colorArgb);
    }

    /**
     * Adds the pixels of a line between two user-space points to a coverage: one rectangle for a
     * horizontal or vertical line, and otherwise each pixel Bresenham's walk steps on.
     *
     * @param coverage the coverage the pixels are added to
     * @param x1 the x coordinate of the first point
     * @param y1 the y coordinate of the first point
     * @param x2 the x coordinate of the second point
     * @param y2 the y coordinate of the second point
     */
    private void drawLine(@NotNull Coverage coverage, int x1, int y1, int x2, int y2) {
        int tx1 = this.translateX + x1;
        int ty1 = this.translateY + y1;
        int tx2 = this.translateX + x2;
        int ty2 = this.translateY + y2;

        if (ty1 == ty2) {
            int minX = Math.min(tx1, tx2);
            coverage.add(minX, ty1, Math.abs(tx2 - tx1) + 1, 1);
            return;
        }
        if (tx1 == tx2) {
            int minY = Math.min(ty1, ty2);
            coverage.add(tx1, minY, 1, Math.abs(ty2 - ty1) + 1);
            return;
        }

        int dx = Math.abs(tx2 - tx1);
        int dy = Math.abs(ty2 - ty1);
        int sx = tx1 < tx2 ? 1 : -1;
        int sy = ty1 < ty2 ? 1 : -1;
        int err = dx - dy;

        while (true) {
            coverage.add(tx1, ty1);
            if (tx1 == tx2 && ty1 == ty2) break;
            int e2 = 2 * err;
            if (e2 > -dy) { err -= dy; tx1 += sx; }
            if (e2 < dx) { err += dx; ty1 += sy; }
        }
    }

    // --- ovals ---

    @Override
    public void drawOval(int x, int y, int width, int height) {
        int cx = this.translateX + x + width / 2;
        int cy = this.translateY + y + height / 2;
        int rx = width / 2;
        int ry = height / 2;
        Coverage coverage = new Coverage();
        drawEllipse(coverage, cx, cy, rx, ry);
        coverage.lay(this.colorArgb);
    }

    @Override
    public void fillOval(int x, int y, int width, int height) {
        int cx = this.translateX + x + width / 2;
        int cy = this.translateY + y + height / 2;
        int rx = width / 2;
        int ry = height / 2;
        Coverage coverage = new Coverage();
        fillEllipse(coverage, cx, cy, rx, ry);
        coverage.lay(this.colorArgb);
    }

    private void drawEllipse(@NotNull Coverage coverage, int cx, int cy, int rx, int ry) {
        if (rx <= 0 || ry <= 0) { coverage.add(cx, cy); return; }
        int x = 0, y = ry;
        long rx2 = (long) rx * rx, ry2 = (long) ry * ry;
        long err = ry2 - rx2 * (long) ry + rx2 / 4;

        while (ry2 * x <= rx2 * y) {
            coverage.add(cx + x, cy + y);
            coverage.add(cx - x, cy + y);
            coverage.add(cx + x, cy - y);
            coverage.add(cx - x, cy - y);
            x++;
            if (err < 0) err += ry2 * (2L * x + 1);
            else { y--; err += ry2 * (2L * x + 1) - rx2 * 2L * y; }
        }

        err = ry2 * (2L * x + 1) * (2L * x + 1) / 4 + rx2 * ((long)(y - 1) * (y - 1) - ry2);
        while (y >= 0) {
            coverage.add(cx + x, cy + y);
            coverage.add(cx - x, cy + y);
            coverage.add(cx + x, cy - y);
            coverage.add(cx - x, cy - y);
            y--;
            if (err > 0) err -= rx2 * (2L * y + 1);
            else { x++; err += ry2 * (2L * x + 1) - rx2 * (2L * y + 1); }
        }
    }

    private void fillEllipse(@NotNull Coverage coverage, int cx, int cy, int rx, int ry) {
        if (rx <= 0 || ry <= 0) { coverage.add(cx, cy); return; }

        for (int y = -ry; y <= ry; y++) {
            int halfWidth = (int) Math.round(rx * Math.sqrt(1.0 - (double) y * y / ((double) ry * ry)));
            coverage.add(cx - halfWidth, cy + y, 2 * halfWidth + 1, 1);
        }
    }

    // --- arcs ---

    @Override
    public void drawArc(int x, int y, int width, int height, int startAngle, int arcAngle) {
        arcImpl(x, y, width, height, startAngle, arcAngle, false);
    }

    @Override
    public void fillArc(int x, int y, int width, int height, int startAngle, int arcAngle) {
        arcImpl(x, y, width, height, startAngle, arcAngle, true);
    }

    private void arcImpl(int x, int y, int width, int height, int startAngle, int arcAngle, boolean fill) {
        int cx = this.translateX + x + width / 2;
        int cy = this.translateY + y + height / 2;
        int rx = width / 2;
        int ry = height / 2;
        if (rx <= 0 || ry <= 0) return;

        double startRad = Math.toRadians(startAngle);
        double extentRad = Math.toRadians(arcAngle);
        double endRad = startRad + extentRad;
        if (extentRad < 0) { double t = startRad; startRad = endRad; endRad = t; }
        Coverage coverage = new Coverage();

        if (fill) {
            double normStart = startRad % (2 * Math.PI);
            if (normStart < 0) normStart += 2 * Math.PI;
            double normEnd = endRad % (2 * Math.PI);
            if (normEnd < 0) normEnd += 2 * Math.PI;

            for (int py = -ry; py <= ry; py++) {
                int halfWidth = (int) Math.round(rx * Math.sqrt(1.0 - (double) py * py / ((double) ry * ry)));

                for (int px = -halfWidth; px <= halfWidth; px++) {
                    double angle = Math.atan2(-py, px);
                    if (angle < 0) angle += 2 * Math.PI;

                    boolean inRange = normStart <= normEnd
                        ? angle >= normStart && angle <= normEnd
                        : angle >= normStart || angle <= normEnd;

                    if (inRange)
                        coverage.add(cx + px, cy + py);
                }
            }
        } else {
            int steps = Math.max(36, Math.abs(arcAngle) * 2);
            double step = extentRad / steps;
            for (int i = 0; i <= steps; i++) {
                double angle = startRad + i * step;
                int px = cx + (int) Math.round(rx * Math.cos(angle));
                int py = cy - (int) Math.round(ry * Math.sin(angle));
                coverage.add(px, py);
            }
        }

        coverage.lay(this.colorArgb);
    }

    // --- polyline and polygon ---

    @Override
    public void drawPolyline(int @NotNull [] xPoints, int @NotNull [] yPoints, int nPoints) {
        Coverage coverage = new Coverage();
        drawPolyline(coverage, xPoints, yPoints, nPoints);
        coverage.lay(this.colorArgb);
    }

    /**
     * Adds the pixels of the lines joining consecutive user-space points to a coverage.
     *
     * @param coverage the coverage the pixels are added to
     * @param xPoints the x coordinates of the points
     * @param yPoints the y coordinates of the points
     * @param nPoints the number of points
     */
    private void drawPolyline(@NotNull Coverage coverage, int @NotNull [] xPoints, int @NotNull [] yPoints, int nPoints) {
        for (int i = 0; i < nPoints - 1; i++)
            drawLine(coverage, xPoints[i], yPoints[i], xPoints[i + 1], yPoints[i + 1]);
    }

    @Override
    public void drawPolygon(int @NotNull [] xPoints, int @NotNull [] yPoints, int nPoints) {
        Coverage coverage = new Coverage();
        drawPolyline(coverage, xPoints, yPoints, nPoints);
        if (nPoints > 1)
            drawLine(coverage, xPoints[nPoints - 1], yPoints[nPoints - 1], xPoints[0], yPoints[0]);
        coverage.lay(this.colorArgb);
    }

    @Override
    public void fillPolygon(int @NotNull [] xPoints, int @NotNull [] yPoints, int nPoints) {
        if (nPoints < 3) return;

        // Find Y bounds
        int minY = Integer.MAX_VALUE, maxY = Integer.MIN_VALUE;
        for (int i = 0; i < nPoints; i++) {
            int ty = this.translateY + yPoints[i];
            minY = Math.min(minY, ty);
            maxY = Math.max(maxY, ty);
        }

        // Scanline fill
        Coverage coverage = new Coverage();
        int[] nodeX = new int[nPoints];
        for (int y = minY; y <= maxY; y++) {
            int nodes = 0;
            int j = nPoints - 1;
            for (int i = 0; i < nPoints; i++) {
                int yi = this.translateY + yPoints[i];
                int yj = this.translateY + yPoints[j];
                if ((yi < y && yj >= y) || (yj < y && yi >= y)) {
                    int xi = this.translateX + xPoints[i];
                    int xj = this.translateX + xPoints[j];
                    nodeX[nodes++] = xi + (y - yi) * (xj - xi) / (yj - yi);
                }
                j = i;
            }

            Arrays.sort(nodeX, 0, nodes);
            for (int i = 0; i < nodes - 1; i += 2)
                coverage.add(nodeX[i], y, nodeX[i + 1] - nodeX[i] + 1, 1);
        }

        coverage.lay(this.colorArgb);
    }

    // --- shapes ---

    @Override
    public void draw(@NotNull Shape s) {
        if (s instanceof Rectangle r) {
            drawRect(r.x, r.y, r.width, r.height);
            return;
        }

        PathIterator pi = s.getPathIterator(null);
        double[] coords = new double[6];
        double startX = 0, startY = 0, lastX = 0, lastY = 0;
        Coverage coverage = new Coverage();

        while (!pi.isDone()) {
            int type = pi.currentSegment(coords);
            switch (type) {
                case PathIterator.SEG_MOVETO -> {
                    startX = lastX = coords[0];
                    startY = lastY = coords[1];
                }
                case PathIterator.SEG_LINETO -> {
                    drawLine(coverage, (int) Math.round(lastX), (int) Math.round(lastY),
                        (int) Math.round(coords[0]), (int) Math.round(coords[1]));
                    lastX = coords[0];
                    lastY = coords[1];
                }
                case PathIterator.SEG_QUADTO -> {
                    drawQuadCurve(coverage, lastX, lastY, coords[0], coords[1], coords[2], coords[3]);
                    lastX = coords[2];
                    lastY = coords[3];
                }
                case PathIterator.SEG_CUBICTO -> {
                    drawCubicCurve(coverage, lastX, lastY, coords[0], coords[1], coords[2], coords[3], coords[4], coords[5]);
                    lastX = coords[4];
                    lastY = coords[5];
                }
                case PathIterator.SEG_CLOSE -> {
                    drawLine(coverage, (int) Math.round(lastX), (int) Math.round(lastY),
                        (int) Math.round(startX), (int) Math.round(startY));
                    lastX = startX;
                    lastY = startY;
                }
            }
            pi.next();
        }

        coverage.lay(this.colorArgb);
    }

    @Override
    public void fill(@NotNull Shape s) {
        if (s instanceof Rectangle r) {
            fillRect(r.x, r.y, r.width, r.height);
            return;
        }

        // Flatten curves to line segments, then scanline fill
        PathIterator pi = s.getPathIterator(null, 1.0);
        double[] coords = new double[6];
        int capacity = 64;
        int[] pxArr = new int[capacity], pyArr = new int[capacity];
        int count = 0;

        while (!pi.isDone()) {
            int type = pi.currentSegment(coords);
            if (type == PathIterator.SEG_MOVETO || type == PathIterator.SEG_LINETO) {
                if (count >= capacity) {
                    capacity *= 2;
                    pxArr = Arrays.copyOf(pxArr, capacity);
                    pyArr = Arrays.copyOf(pyArr, capacity);
                }
                pxArr[count] = (int) Math.round(coords[0]) - this.translateX;
                pyArr[count] = (int) Math.round(coords[1]) - this.translateY;
                count++;
            }
            pi.next();
        }

        if (count >= 3)
            fillPolygon(pxArr, pyArr, count);
    }

    @Override
    public boolean hit(@NotNull Rectangle rect, @NotNull Shape s, boolean onStroke) {
        return s.intersects(rect);
    }

    private void drawQuadCurve(@NotNull Coverage coverage, double x0, double y0, double cx, double cy, double x1, double y1) {
        int steps = Math.max(8, (int) (Math.hypot(x1 - x0, y1 - y0) / 2));
        double prevX = x0, prevY = y0;
        for (int i = 1; i <= steps; i++) {
            double t = (double) i / steps;
            double u = 1 - t;
            double px = u * u * x0 + 2 * u * t * cx + t * t * x1;
            double py = u * u * y0 + 2 * u * t * cy + t * t * y1;
            drawLine(coverage, (int) Math.round(prevX), (int) Math.round(prevY),
                (int) Math.round(px), (int) Math.round(py));
            prevX = px;
            prevY = py;
        }
    }

    private void drawCubicCurve(@NotNull Coverage coverage, double x0, double y0, double cx1, double cy1, double cx2, double cy2, double x1, double y1) {
        int steps = Math.max(12, (int) (Math.hypot(x1 - x0, y1 - y0) / 2));
        double prevX = x0, prevY = y0;
        for (int i = 1; i <= steps; i++) {
            double t = (double) i / steps;
            double u = 1 - t;
            double px = u * u * u * x0 + 3 * u * u * t * cx1 + 3 * u * t * t * cx2 + t * t * t * x1;
            double py = u * u * u * y0 + 3 * u * u * t * cy1 + 3 * u * t * t * cy2 + t * t * t * y1;
            drawLine(coverage, (int) Math.round(prevX), (int) Math.round(prevY),
                (int) Math.round(px), (int) Math.round(py));
            prevX = px;
            prevY = py;
        }
    }

    // --- image drawing ---

    @Override
    public boolean drawImage(@NotNull Image img, int x, int y, @Nullable ImageObserver observer) {
        if (!(img instanceof BufferedImage bi)) return false;
        layImage(PixelBuffer.wrap(bi), this.translateX + x, this.translateY + y);
        return true;
    }

    @Override
    public boolean drawImage(@NotNull Image img, int x, int y, int width, int height, @Nullable ImageObserver observer) {
        if (!(img instanceof BufferedImage bi)) return false;
        PixelBuffer source = PixelBuffer.wrap(bi);

        if (this.clip == null) {
            this.target.blitScaled(source, this.translateX + x, this.translateY + y, width, height);
            return true;
        }

        if (width <= 0 || height <= 0) return true;
        PixelBuffer scaled = blankLike(source, width, height);
        scaled.blitScaled(source, 0, 0, width, height, BlendMode.REPLACE);
        layImage(scaled, this.translateX + x, this.translateY + y);
        return true;
    }

    @Override
    public boolean drawImage(@NotNull Image img, int x, int y, @Nullable Color bgcolor, @Nullable ImageObserver observer) {
        return drawImage(img, x, y, observer);
    }

    @Override
    public boolean drawImage(@NotNull Image img, int x, int y, int width, int height, @Nullable Color bgcolor, @Nullable ImageObserver observer) {
        return drawImage(img, x, y, width, height, observer);
    }

    @Override
    public boolean drawImage(@NotNull Image img, int dx1, int dy1, int dx2, int dy2, int sx1, int sy1, int sx2, int sy2, @Nullable ImageObserver observer) {
        if (!(img instanceof BufferedImage bi)) return false;
        PixelBuffer source = PixelBuffer.wrap(bi);
        int sw = Math.abs(sx2 - sx1);
        int sh = Math.abs(sy2 - sy1);
        int dw = Math.abs(dx2 - dx1);
        int dh = Math.abs(dy2 - dy1);
        if (sw == 0 || sh == 0 || dw == 0 || dh == 0) return true;

        int srcMinX = Math.min(sx1, sx2), srcMinY = Math.min(sy1, sy2);
        int dstMinX = Math.min(dx1, dx2), dstMinY = Math.min(dy1, dy2);
        // A corner pair reversed on one side only mirrors the image along that axis: (dx1, dy1)
        // always receives (sx1, sy1).
        boolean flipX = (dx2 < dx1) != (sx2 < sx1);
        boolean flipY = (dy2 < dy1) != (sy2 < sy1);
        int left = this.translateX + dstMinX;
        int top = this.translateY + dstMinY;
        Rectangle area = drawableArea(left, top, dw, dh);
        if (area.isEmpty()) return true;
        PixelBuffer sampled = blankLike(source, area.width, area.height);

        for (int py = area.y; py < area.y + area.height; py++) {
            int offY = (py - top) * sh / dh;
            int srcY = flipY ? srcMinY + sh - 1 - offY : srcMinY + offY;
            for (int px = area.x; px < area.x + area.width; px++) {
                int offX = (px - left) * sw / dw;
                int srcX = flipX ? srcMinX + sw - 1 - offX : srcMinX + offX;
                if (srcX >= 0 && srcX < source.width() && srcY >= 0 && srcY < source.height())
                    sampled.setPixel(px - area.x, py - area.y, source.getPixel(srcX, srcY));
            }
        }

        this.target.blit(sampled, area.x, area.y);
        return true;
    }

    @Override
    public boolean drawImage(@NotNull Image img, int dx1, int dy1, int dx2, int dy2, int sx1, int sy1, int sx2, int sy2, @Nullable Color bgcolor, @Nullable ImageObserver observer) {
        return drawImage(img, dx1, dy1, dx2, dy2, sx1, sy1, sx2, sy2, observer);
    }

    @Override
    public boolean drawImage(@NotNull Image img, @NotNull AffineTransform xform, @Nullable ImageObserver obs) {
        if (!(img instanceof BufferedImage bi)) return false;
        if (xform.isIdentity()) return drawImage(img, 0, 0, obs);

        PixelBuffer source = PixelBuffer.wrap(bi);
        try {
            AffineTransform inv = xform.createInverse();
            Rectangle bounds = xform.createTransformedShape(new Rectangle(0, 0, source.width(), source.height())).getBounds();
            Rectangle area = drawableArea(this.translateX + bounds.x, this.translateY + bounds.y, bounds.width, bounds.height);
            if (area.isEmpty()) return true;
            PixelBuffer sampled = blankLike(source, area.width, area.height);

            for (int py = area.y; py < area.y + area.height; py++) {
                for (int px = area.x; px < area.x + area.width; px++) {
                    double[] src = new double[2];
                    inv.transform(new double[]{ px - this.translateX, py - this.translateY }, 0, src, 0, 1);
                    int srcX = (int) Math.round(src[0]);
                    int srcY = (int) Math.round(src[1]);
                    if (srcX >= 0 && srcX < source.width() && srcY >= 0 && srcY < source.height())
                        sampled.setPixel(px - area.x, py - area.y, source.getPixel(srcX, srcY));
                }
            }

            this.target.blit(sampled, area.x, area.y);
        } catch (java.awt.geom.NoninvertibleTransformException e) {
            return false;
        }
        return true;
    }

    @Override
    public void drawImage(@NotNull BufferedImage img, @Nullable BufferedImageOp op, int x, int y) {
        drawImage(img, x, y, null);
    }

    @Override
    public void drawRenderedImage(@NotNull RenderedImage img, @NotNull AffineTransform xform) {
        if (img instanceof BufferedImage bi)
            drawImage(bi, xform, null);
    }

    @Override
    public void drawRenderableImage(@NotNull RenderableImage img, @NotNull AffineTransform xform) {
        drawRenderedImage(img.createDefaultRendering(), xform);
    }

    // --- transform ---

    @Override
    public void translate(int x, int y) {
        this.translateX += x;
        this.translateY += y;
        this.transform.translate(x, y);
    }

    @Override
    public void translate(double tx, double ty) {
        translate((int) Math.round(tx), (int) Math.round(ty));
    }

    @Override
    public void rotate(double theta) {
        this.transform.rotate(theta);
    }

    @Override
    public void rotate(double theta, double x, double y) {
        this.transform.rotate(theta, x, y);
    }

    @Override
    public void scale(double sx, double sy) {
        this.transform.scale(sx, sy);
    }

    @Override
    public void shear(double shx, double shy) {
        this.transform.shear(shx, shy);
    }

    @Override
    public void transform(@NotNull AffineTransform tx) {
        this.transform.concatenate(tx);
        this.translateX = (int) this.transform.getTranslateX();
        this.translateY = (int) this.transform.getTranslateY();
    }

    @Override
    public void setTransform(@NotNull AffineTransform tx) {
        this.transform = new AffineTransform(tx);
        this.translateX = (int) tx.getTranslateX();
        this.translateY = (int) tx.getTranslateY();
    }

    @Override
    public @NotNull AffineTransform getTransform() {
        return new AffineTransform(this.transform);
    }

    // --- clip ---

    @Override
    public void setClip(int x, int y, int width, int height) {
        this.clip = new Rectangle(x, y, width, height);
    }

    @Override
    public void setClip(@Nullable Shape shape) {
        this.clip = shape != null ? shape.getBounds() : null;
    }

    @Override
    public @Nullable Shape getClip() {
        return this.clip;
    }

    @Override
    public @Nullable Rectangle getClipBounds() {
        return this.clip != null ? new Rectangle(this.clip) : null;
    }

    @Override
    public void clipRect(int x, int y, int width, int height) {
        Rectangle r = new Rectangle(x, y, width, height);
        this.clip = this.clip != null ? this.clip.intersection(r) : r;
    }

    @Override
    public void clip(@NotNull Shape s) {
        Rectangle b = s.getBounds();
        clipRect(b.x, b.y, b.width, b.height);
    }

    // --- rendering hints ---

    @Override
    public void setRenderingHint(@NotNull RenderingHints.Key hintKey, @Nullable Object hintValue) {
        this.hints.put(hintKey, hintValue);
    }

    @Override
    public @Nullable Object getRenderingHint(@NotNull RenderingHints.Key hintKey) {
        return this.hints.get(hintKey);
    }

    @Override
    public void setRenderingHints(@NotNull Map<?, ?> hints) {
        this.hints = new RenderingHints(null);
        this.hints.putAll(hints);
    }

    @Override
    public void addRenderingHints(@NotNull Map<?, ?> hints) {
        this.hints.putAll(hints);
    }

    @Override
    public @NotNull RenderingHints getRenderingHints() {
        return this.hints;
    }

    // --- composite and stroke ---

    @Override
    public void setComposite(@NotNull Composite comp) {
        if (!AlphaComposite.SrcOver.equals(comp))
            throw new UnsupportedOperationException("Only AlphaComposite.SrcOver is supported, not '%s'".formatted(comp));
    }

    @Override
    public @NotNull Composite getComposite() {
        return AlphaComposite.SrcOver;
    }

    @Override
    public void setStroke(@NotNull Stroke s) {
        this.stroke = s;
    }

    @Override
    public @NotNull Stroke getStroke() {
        return this.stroke;
    }

    // --- mode ---

    @Override
    public void setPaintMode() {
        // no-op - every draw composites source-over, the paint mode, and XOR is refused
    }

    @Override
    public void setXORMode(@NotNull Color c1) {
        throw new UnsupportedOperationException("XOR mode is not supported");
    }

    // --- copy and lifecycle ---

    @Override
    public void copyArea(int x, int y, int width, int height, int dx, int dy) {
        int srcX = this.translateX + x;
        int srcY = this.translateY + y;
        int tw = this.target.width();
        int th = this.target.height();

        // Copy to temp to handle overlapping regions
        int[] temp = new int[width * height];
        for (int row = 0; row < height; row++) {
            int sy = srcY + row;
            if (sy < 0 || sy >= th) continue;
            for (int col = 0; col < width; col++) {
                int sx = srcX + col;
                if (sx >= 0 && sx < tw)
                    temp[row * width + col] = this.target.getPixel(sx, sy);
            }
        }

        for (int row = 0; row < height; row++) {
            int dy2 = srcY + dy + row;
            if (dy2 < 0 || dy2 >= th) continue;
            for (int col = 0; col < width; col++) {
                int dx2 = srcX + dx + col;
                if (dx2 >= 0 && dx2 < tw && inClip(dx2, dy2))
                    this.target.setPixel(dx2, dy2, temp[row * width + col]);
            }
        }
    }

    @Override
    public @NotNull Graphics create() {
        return new PixelGraphics(this);
    }

    @Override
    public @Nullable GraphicsConfiguration getDeviceConfiguration() {
        return null;
    }

    @Override
    public void dispose() {
        // no-op — no native resources
    }

}
