package dev.simplified.image.pixel;

import org.jetbrains.annotations.NotNull;

/**
 * A per-pixel boolean coverage mask over a rectangular buffer. A pixel is either marked or bare; the
 * mask records which pixels a later pass should act on (a foil overlay restricted to certain geometry,
 * a selection, a dirty region).
 * <p>
 * Backed by a flat {@code byte} per pixel (1 = marked). Out-of-range coordinates are ignored on
 * {@link #mark} and read as unmarked on {@link #marked}.
 */
public final class PixelMask {

    /**
     * Flat row-major coverage buffer indexed as {@code y * width + x}; {@code 1} marks a pixel,
     * {@code 0} leaves it bare.
     */
    private final byte @NotNull [] data;

    /**
     * Mask width in pixels.
     */
    private final int width;

    /**
     * Mask height in pixels.
     */
    private final int height;

    /**
     * Constructs an all-unmarked mask of the given dimensions.
     *
     * @param width the mask width in pixels
     * @param height the mask height in pixels
     */
    public PixelMask(int width, int height) {
        this.width = width;
        this.height = height;
        this.data = new byte[Math.max(0, width * height)];
    }

    /**
     * Mask width in pixels.
     */
    public int width() {
        return this.width;
    }

    /**
     * Mask height in pixels.
     */
    public int height() {
        return this.height;
    }

    /**
     * Marks the pixel at {@code (x, y)}. Out-of-range coordinates are ignored.
     *
     * @param x the pixel x coordinate
     * @param y the pixel y coordinate
     */
    public void mark(int x, int y) {
        if (x < 0 || x >= this.width || y < 0 || y >= this.height) return;
        this.data[y * this.width + x] = 1;
    }

    /**
     * Whether the pixel at {@code (x, y)} is marked.
     *
     * @param x the pixel x coordinate
     * @param y the pixel y coordinate
     * @return {@code true} when the pixel is marked, {@code false} when unmarked or out of range
     */
    public boolean marked(int x, int y) {
        if (x < 0 || x >= this.width || y < 0 || y >= this.height) return false;
        return this.data[y * this.width + x] != 0;
    }

    /**
     * Box-downsamples this mask to {@code outW x outH} using an "any subpixel marked produces a marked
     * output pixel" rule. This keeps the marked region from eroding at its silhouette when a
     * supersampled buffer is scaled down. Returns {@code this} unchanged when the dimensions already
     * match.
     *
     * @param outW the target width in pixels
     * @param outH the target height in pixels
     * @return the downsampled mask
     */
    public @NotNull PixelMask downsample(int outW, int outH) {
        if (outW == this.width && outH == this.height) return this;
        PixelMask out = new PixelMask(outW, outH);
        for (int oy = 0; oy < outH; oy++) {
            int y0 = (int) ((long) oy * this.height / outH);
            int y1 = Math.max(y0 + 1, (int) ((long) (oy + 1) * this.height / outH));
            for (int ox = 0; ox < outW; ox++) {
                int x0 = (int) ((long) ox * this.width / outW);
                int x1 = Math.max(x0 + 1, (int) ((long) (ox + 1) * this.width / outW));
                if (anyMarked(x0, y0, Math.min(x1, this.width), Math.min(y1, this.height)))
                    out.mark(ox, oy);
            }
        }
        return out;
    }

    /**
     * Whether any pixel in the half-open box {@code [x0, x1) x [y0, y1)} of this mask is marked - the
     * "any subpixel marked wins" predicate {@link #downsample} evaluates per output pixel.
     *
     * @param x0 inclusive left bound
     * @param y0 inclusive top bound
     * @param x1 exclusive right bound
     * @param y1 exclusive bottom bound
     * @return {@code true} when at least one pixel in the box is marked
     */
    private boolean anyMarked(int x0, int y0, int x1, int y1) {
        for (int y = y0; y < y1; y++)
            for (int x = x0; x < x1; x++)
                if (this.data[y * this.width + x] != 0) return true;
        return false;
    }

}
