package dev.simplified.image.pixel;

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
    private @NotNull Composite composite;
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
        this.composite = AlphaComposite.SrcOver;
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
        this.composite = source.composite;
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

    private void setPixelSafe(int x, int y, int argb) {
        if (x >= 0 && x < this.target.width() && y >= 0 && y < this.target.height())
            this.target.setPixel(x, y, argb);
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
        this.target.fillRect(this.translateX + x, this.translateY + y, width, height, this.colorArgb);
    }

    @Override
    public void clearRect(int x, int y, int width, int height) {
        this.target.fillRect(this.translateX + x, this.translateY + y, width, height, this.backgroundArgb);
    }

    @Override
    public void drawRect(int x, int y, int width, int height) {
        int tx = this.translateX + x;
        int ty = this.translateY + y;
        this.target.fillRect(tx, ty, width + 1, 1, this.colorArgb);
        this.target.fillRect(tx, ty + height, width + 1, 1, this.colorArgb);
        this.target.fillRect(tx, ty, 1, height + 1, this.colorArgb);
        this.target.fillRect(tx + width, ty, 1, height + 1, this.colorArgb);
    }

    @Override
    public void drawRoundRect(int x, int y, int width, int height, int arcWidth, int arcHeight) {
        int rx = arcWidth / 2;
        int ry = arcHeight / 2;
        int tx = this.translateX + x;
        int ty = this.translateY + y;

        // Straight edges
        this.target.fillRect(tx + rx, ty, width - 2 * rx + 1, 1, this.colorArgb);
        this.target.fillRect(tx + rx, ty + height, width - 2 * rx + 1, 1, this.colorArgb);
        this.target.fillRect(tx, ty + ry, 1, height - 2 * ry + 1, this.colorArgb);
        this.target.fillRect(tx + width, ty + ry, 1, height - 2 * ry + 1, this.colorArgb);

        // Corner arcs via midpoint ellipse
        drawCornerArcs(tx + rx, ty + ry, tx + width - rx, ty + height - ry, rx, ry);
    }

    @Override
    public void fillRoundRect(int x, int y, int width, int height, int arcWidth, int arcHeight) {
        int rx = arcWidth / 2;
        int ry = arcHeight / 2;
        int tx = this.translateX + x;
        int ty = this.translateY + y;

        // Central body
        this.target.fillRect(tx, ty + ry, width + 1, height - 2 * ry + 1, this.colorArgb);

        // Top and bottom bands with rounded corners via scanline
        fillCornerArcs(tx + rx, ty + ry, tx + width - rx, ty + height - ry, rx, ry);
    }

    private void drawCornerArcs(int cx1, int cy1, int cx2, int cy2, int rx, int ry) {
        if (rx <= 0 || ry <= 0) return;
        int x = 0, y = ry;
        long rx2 = (long) rx * rx, ry2 = (long) ry * ry;
        long err = ry2 - rx2 * (long) ry + rx2 / 4;

        while (ry2 * x <= rx2 * y) {
            setPixelSafe(cx2 + x, cy1 - y, this.colorArgb);
            setPixelSafe(cx1 - x, cy1 - y, this.colorArgb);
            setPixelSafe(cx2 + x, cy2 + y, this.colorArgb);
            setPixelSafe(cx1 - x, cy2 + y, this.colorArgb);
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
            setPixelSafe(cx2 + x, cy1 - y, this.colorArgb);
            setPixelSafe(cx1 - x, cy1 - y, this.colorArgb);
            setPixelSafe(cx2 + x, cy2 + y, this.colorArgb);
            setPixelSafe(cx1 - x, cy2 + y, this.colorArgb);
            y--;
            if (err > 0)
                err -= rx2 * (2L * y + 1);
            else {
                x++;
                err += ry2 * (2L * x + 1) - rx2 * (2L * y + 1);
            }
        }
    }

    private void fillCornerArcs(int cx1, int cy1, int cx2, int cy2, int rx, int ry) {
        if (rx <= 0 || ry <= 0) return;
        int px = 0, py = ry;
        long rx2 = (long) rx * rx, ry2 = (long) ry * ry;
        long err = ry2 - rx2 * (long) ry + rx2 / 4;

        int lastPy = -1;
        while (ry2 * px <= rx2 * py) {
            if (py != lastPy) {
                this.target.fillRect(cx1 - px, cy1 - py, cx2 - cx1 + 2 * px + 1, 1, this.colorArgb);
                this.target.fillRect(cx1 - px, cy2 + py, cx2 - cx1 + 2 * px + 1, 1, this.colorArgb);
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
                this.target.fillRect(cx1 - px, cy1 - py, cx2 - cx1 + 2 * px + 1, 1, this.colorArgb);
                this.target.fillRect(cx1 - px, cy2 + py, cx2 - cx1 + 2 * px + 1, 1, this.colorArgb);
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
        int tx1 = this.translateX + x1;
        int ty1 = this.translateY + y1;
        int tx2 = this.translateX + x2;
        int ty2 = this.translateY + y2;

        if (ty1 == ty2) {
            int minX = Math.min(tx1, tx2);
            this.target.fillRect(minX, ty1, Math.abs(tx2 - tx1) + 1, 1, this.colorArgb);
            return;
        }
        if (tx1 == tx2) {
            int minY = Math.min(ty1, ty2);
            this.target.fillRect(tx1, minY, 1, Math.abs(ty2 - ty1) + 1, this.colorArgb);
            return;
        }

        int dx = Math.abs(tx2 - tx1);
        int dy = Math.abs(ty2 - ty1);
        int sx = tx1 < tx2 ? 1 : -1;
        int sy = ty1 < ty2 ? 1 : -1;
        int err = dx - dy;

        while (true) {
            setPixelSafe(tx1, ty1, this.colorArgb);
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
        drawEllipse(cx, cy, rx, ry);
    }

    @Override
    public void fillOval(int x, int y, int width, int height) {
        int cx = this.translateX + x + width / 2;
        int cy = this.translateY + y + height / 2;
        int rx = width / 2;
        int ry = height / 2;
        fillEllipse(cx, cy, rx, ry);
    }

    private void drawEllipse(int cx, int cy, int rx, int ry) {
        if (rx <= 0 || ry <= 0) { setPixelSafe(cx, cy, this.colorArgb); return; }
        int x = 0, y = ry;
        long rx2 = (long) rx * rx, ry2 = (long) ry * ry;
        long err = ry2 - rx2 * (long) ry + rx2 / 4;

        while (ry2 * x <= rx2 * y) {
            setPixelSafe(cx + x, cy + y, this.colorArgb);
            setPixelSafe(cx - x, cy + y, this.colorArgb);
            setPixelSafe(cx + x, cy - y, this.colorArgb);
            setPixelSafe(cx - x, cy - y, this.colorArgb);
            x++;
            if (err < 0) err += ry2 * (2L * x + 1);
            else { y--; err += ry2 * (2L * x + 1) - rx2 * 2L * y; }
        }

        err = ry2 * (2L * x + 1) * (2L * x + 1) / 4 + rx2 * ((long)(y - 1) * (y - 1) - ry2);
        while (y >= 0) {
            setPixelSafe(cx + x, cy + y, this.colorArgb);
            setPixelSafe(cx - x, cy + y, this.colorArgb);
            setPixelSafe(cx + x, cy - y, this.colorArgb);
            setPixelSafe(cx - x, cy - y, this.colorArgb);
            y--;
            if (err > 0) err -= rx2 * (2L * y + 1);
            else { x++; err += ry2 * (2L * x + 1) - rx2 * (2L * y + 1); }
        }
    }

    private void fillEllipse(int cx, int cy, int rx, int ry) {
        if (rx <= 0 || ry <= 0) { setPixelSafe(cx, cy, this.colorArgb); return; }

        for (int y = -ry; y <= ry; y++) {
            int halfWidth = (int) Math.round(rx * Math.sqrt(1.0 - (double) y * y / ((double) ry * ry)));
            this.target.fillRect(cx - halfWidth, cy + y, 2 * halfWidth + 1, 1, this.colorArgb);
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
                        setPixelSafe(cx + px, cy + py, this.colorArgb);
                }
            }
        } else {
            int steps = Math.max(36, Math.abs(arcAngle) * 2);
            double step = extentRad / steps;
            for (int i = 0; i <= steps; i++) {
                double angle = startRad + i * step;
                int px = cx + (int) Math.round(rx * Math.cos(angle));
                int py = cy - (int) Math.round(ry * Math.sin(angle));
                setPixelSafe(px, py, this.colorArgb);
            }
        }
    }

    // --- polyline and polygon ---

    @Override
    public void drawPolyline(int @NotNull [] xPoints, int @NotNull [] yPoints, int nPoints) {
        for (int i = 0; i < nPoints - 1; i++)
            drawLine(xPoints[i], yPoints[i], xPoints[i + 1], yPoints[i + 1]);
    }

    @Override
    public void drawPolygon(int @NotNull [] xPoints, int @NotNull [] yPoints, int nPoints) {
        drawPolyline(xPoints, yPoints, nPoints);
        if (nPoints > 1)
            drawLine(xPoints[nPoints - 1], yPoints[nPoints - 1], xPoints[0], yPoints[0]);
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
                this.target.fillRect(nodeX[i], y, nodeX[i + 1] - nodeX[i] + 1, 1, this.colorArgb);
        }
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

        while (!pi.isDone()) {
            int type = pi.currentSegment(coords);
            switch (type) {
                case PathIterator.SEG_MOVETO -> {
                    startX = lastX = coords[0];
                    startY = lastY = coords[1];
                }
                case PathIterator.SEG_LINETO -> {
                    drawLine((int) Math.round(lastX), (int) Math.round(lastY),
                        (int) Math.round(coords[0]), (int) Math.round(coords[1]));
                    lastX = coords[0];
                    lastY = coords[1];
                }
                case PathIterator.SEG_QUADTO -> {
                    drawQuadCurve(lastX, lastY, coords[0], coords[1], coords[2], coords[3]);
                    lastX = coords[2];
                    lastY = coords[3];
                }
                case PathIterator.SEG_CUBICTO -> {
                    drawCubicCurve(lastX, lastY, coords[0], coords[1], coords[2], coords[3], coords[4], coords[5]);
                    lastX = coords[4];
                    lastY = coords[5];
                }
                case PathIterator.SEG_CLOSE -> {
                    drawLine((int) Math.round(lastX), (int) Math.round(lastY),
                        (int) Math.round(startX), (int) Math.round(startY));
                    lastX = startX;
                    lastY = startY;
                }
            }
            pi.next();
        }
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

    private void drawQuadCurve(double x0, double y0, double cx, double cy, double x1, double y1) {
        int steps = Math.max(8, (int) (Math.hypot(x1 - x0, y1 - y0) / 2));
        double prevX = x0, prevY = y0;
        for (int i = 1; i <= steps; i++) {
            double t = (double) i / steps;
            double u = 1 - t;
            double px = u * u * x0 + 2 * u * t * cx + t * t * x1;
            double py = u * u * y0 + 2 * u * t * cy + t * t * y1;
            drawLine((int) Math.round(prevX), (int) Math.round(prevY),
                (int) Math.round(px), (int) Math.round(py));
            prevX = px;
            prevY = py;
        }
    }

    private void drawCubicCurve(double x0, double y0, double cx1, double cy1, double cx2, double cy2, double x1, double y1) {
        int steps = Math.max(12, (int) (Math.hypot(x1 - x0, y1 - y0) / 2));
        double prevX = x0, prevY = y0;
        for (int i = 1; i <= steps; i++) {
            double t = (double) i / steps;
            double u = 1 - t;
            double px = u * u * u * x0 + 3 * u * u * t * cx1 + 3 * u * t * t * cx2 + t * t * t * x1;
            double py = u * u * u * y0 + 3 * u * u * t * cy1 + 3 * u * t * t * cy2 + t * t * t * y1;
            drawLine((int) Math.round(prevX), (int) Math.round(prevY),
                (int) Math.round(px), (int) Math.round(py));
            prevX = px;
            prevY = py;
        }
    }

    // --- image drawing ---

    @Override
    public boolean drawImage(@NotNull Image img, int x, int y, @Nullable ImageObserver observer) {
        if (img instanceof BufferedImage bi) {
            this.target.blit(PixelBuffer.wrap(bi), this.translateX + x, this.translateY + y);
            return true;
        }
        return false;
    }

    @Override
    public boolean drawImage(@NotNull Image img, int x, int y, int width, int height, @Nullable ImageObserver observer) {
        if (img instanceof BufferedImage bi) {
            this.target.blitScaled(PixelBuffer.wrap(bi), this.translateX + x, this.translateY + y, width, height);
            return true;
        }
        return false;
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

        for (int y = 0; y < dh; y++) {
            int srcY = srcMinY + y * sh / dh;
            for (int x = 0; x < dw; x++) {
                int srcX = srcMinX + x * sw / dw;
                if (srcX >= 0 && srcX < source.width() && srcY >= 0 && srcY < source.height()) {
                    int pixel = source.getPixel(srcX, srcY);
                    if (ColorMath.alpha(pixel) > 0)
                        setPixelSafe(this.translateX + dstMinX + x, this.translateY + dstMinY + y, pixel);
                }
            }
        }
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

            for (int dy = bounds.y; dy < bounds.y + bounds.height; dy++) {
                for (int dx = bounds.x; dx < bounds.x + bounds.width; dx++) {
                    double[] src = new double[2];
                    inv.transform(new double[]{dx, dy}, 0, src, 0, 1);
                    int srcX = (int) Math.round(src[0]);
                    int srcY = (int) Math.round(src[1]);
                    if (srcX >= 0 && srcX < source.width() && srcY >= 0 && srcY < source.height()) {
                        int pixel = source.getPixel(srcX, srcY);
                        if (ColorMath.alpha(pixel) > 0)
                            setPixelSafe(this.translateX + dx, this.translateY + dy, pixel);
                    }
                }
            }
        } catch (java.awt.geom.NoninvertibleTransformException e) {
            return false;
        }
        return true;
    }

    @Override
    public void drawImage(@NotNull BufferedImage img, @Nullable BufferedImageOp op, int x, int y) {
        this.target.blit(PixelBuffer.wrap(img), this.translateX + x, this.translateY + y);
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
        this.composite = comp;
    }

    @Override
    public @NotNull Composite getComposite() {
        return this.composite;
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
        // no-op
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
                if (dx2 >= 0 && dx2 < tw)
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
