package dev.simplified.image.pixel;

import lombok.Getter;
import lombok.experimental.Accessors;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.awt.AlphaComposite;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferInt;
import java.awt.image.Raster;
import java.awt.image.SinglePixelPackedSampleModel;
import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * A zero-copy wrapper over {@code int[]} ARGB pixel data.
 * <p>
 * For {@link BufferedImage#TYPE_INT_ARGB} images, the backing array is referenced directly without
 * copying. For other image types, pixel data is extracted once into a new array, then accessed
 * without further copies.
 */
@Getter
@Accessors(fluent = true)
public class PixelBuffer {

    private final int @NotNull [] pixels;
    private final int width;
    private final int height;
    private final boolean hasAlpha;

    private PixelBuffer(int @NotNull [] pixels, int width, int height, boolean hasAlpha) {
        if (pixels.length != width * height)
            throw new IllegalArgumentException("Pixel array length %d does not match dimensions %dx%d".formatted(pixels.length, width, height));

        this.pixels = pixels;
        this.width = width;
        this.height = height;
        this.hasAlpha = hasAlpha;
    }

    /**
     * Creates a blank pixel buffer of the given dimensions, initially filled with transparent black.
     * No {@link BufferedImage} is allocated.
     *
     * @param width the buffer width in pixels
     * @param height the buffer height in pixels
     * @return a new pixel buffer
     */
    public static @NotNull PixelBuffer create(int width, int height) {
        return new PixelBuffer(new int[width * height], width, height, true);
    }

    /**
     * Creates a pixel buffer from an existing ARGB pixel array, assuming alpha is present.
     *
     * @param pixels the ARGB pixel data (not copied)
     * @param width the image width
     * @param height the image height
     * @return a pixel buffer wrapping the array
     */
    public static @NotNull PixelBuffer of(int @NotNull [] pixels, int width, int height) {
        return new PixelBuffer(pixels, width, height, true);
    }

    /**
     * Creates a pixel buffer from an existing ARGB pixel array.
     *
     * @param pixels the ARGB pixel data (not copied)
     * @param width the image width
     * @param height the image height
     * @param hasAlpha whether the alpha channel carries meaningful data
     * @return a pixel buffer wrapping the array
     */
    public static @NotNull PixelBuffer of(int @NotNull [] pixels, int width, int height, boolean hasAlpha) {
        return new PixelBuffer(pixels, width, height, hasAlpha);
    }

    /**
     * Wraps the pixel data of a {@link BufferedImage}.
     * <p>
     * If the image is {@link BufferedImage#TYPE_INT_ARGB}, the underlying data buffer is referenced
     * directly (zero copy). Otherwise, pixel data is extracted via
     * {@link BufferedImage#getRGB(int, int, int, int, int[], int, int)}.
     *
     * @param image the source image
     * @return a pixel buffer wrapping the image data
     */
    public static @NotNull PixelBuffer wrap(@NotNull BufferedImage image) {
        int w = image.getWidth();
        int h = image.getHeight();
        boolean alpha = image.getColorModel().hasAlpha();

        if (image.getType() == BufferedImage.TYPE_INT_ARGB) {
            int[] data = ((DataBufferInt) image.getRaster().getDataBuffer()).getData();
            return new PixelBuffer(data, w, h, alpha);
        }

        int[] pixels = image.getRGB(0, 0, w, h, null, 0, w);
        return new PixelBuffer(pixels, w, h, alpha);
    }

    // --- bounds queries ---

    /**
     * Returns whether the given coordinates fall within the buffer.
     *
     * @param x the column index
     * @param y the row index
     * @return {@code true} if {@code (x, y)} lies inside the buffer
     */
    public boolean contains(int x, int y) {
        return x >= 0 && y >= 0 && x < this.width && y < this.height;
    }

    /**
     * Computes the tightest axis-aligned rectangle containing every non-transparent pixel.
     * <p>
     * Returns a zero-size rectangle when the buffer is fully transparent.
     *
     * @return the bounding rectangle of opaque pixels
     */
    public @NotNull Rectangle opaqueBounds() {
        int minX = this.width, minY = this.height, maxX = -1, maxY = -1;
        for (int y = 0; y < this.height; y++) {
            int row = y * this.width;
            for (int x = 0; x < this.width; x++) {
                if (((this.pixels[row + x] >>> 24) & 0xFF) == 0) continue;
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
        if (maxX < 0) return new Rectangle(0, 0, 0, 0);
        return new Rectangle(minX, minY, maxX - minX + 1, maxY - minY + 1);
    }

    // --- pixel I/O ---

    /**
     * Returns the ARGB value at the given coordinates.
     *
     * @param x the column index
     * @param y the row index
     * @return the packed ARGB pixel value
     */
    public int getPixel(int x, int y) {
        return this.pixels[y * this.width + x];
    }

    /**
     * Copies a rectangular region of pixels into a target array using the given stride layout.
     *
     * @param x the left edge of the region
     * @param y the top edge of the region
     * @param w the region width
     * @param h the region height
     * @param target the destination array, or {@code null} to allocate a tightly packed result
     * @param offset the offset into {@code target} of the first written pixel
     * @param stride the number of {@code int}s between consecutive rows in {@code target}
     * @return {@code target} if provided, otherwise the newly allocated array
     */
    public int @NotNull [] getPixels(int x, int y, int w, int h, int @Nullable [] target, int offset, int stride) {
        int[] out = target != null ? target : new int[w * h];
        int outStride = target != null ? stride : w;
        for (int row = 0; row < h; row++)
            System.arraycopy(this.pixels, (y + row) * this.width + x, out, offset + row * outStride, w);
        return out;
    }

    /**
     * Returns a newly-allocated copy of the pixels in row {@code y}.
     *
     * @param y the row index
     * @return a new array containing the row's pixels
     */
    public int @NotNull [] getRow(int y) {
        int[] row = new int[this.width];
        System.arraycopy(this.pixels, y * this.width, row, 0, this.width);
        return row;
    }

    /**
     * Sets the ARGB value at the given coordinates.
     *
     * @param x the column index
     * @param y the row index
     * @param argb the packed ARGB pixel value
     */
    public void setPixel(int x, int y, int argb) {
        this.pixels[y * this.width + x] = argb;
    }

    /**
     * Writes a rectangular region of pixels from a source array using the given stride layout.
     *
     * @param x the left edge of the region
     * @param y the top edge of the region
     * @param w the region width
     * @param h the region height
     * @param src the source pixel array
     * @param offset the offset into {@code src} of the first read pixel
     * @param stride the number of {@code int}s between consecutive rows in {@code src}
     */
    public void setPixels(int x, int y, int w, int h, int @NotNull [] src, int offset, int stride) {
        for (int row = 0; row < h; row++)
            System.arraycopy(src, offset + row * stride, this.pixels, (y + row) * this.width + x, w);
    }

    /**
     * Overwrites row {@code y} with the given pixels.
     *
     * @param y the row index
     * @param row the pixels to copy; must have length {@link #width()}
     */
    public void setRow(int y, int @NotNull [] row) {
        System.arraycopy(row, 0, this.pixels, y * this.width, this.width);
    }

    // --- geometry ---

    /**
     * Returns a new pixel buffer containing a copy of the given sub-region.
     *
     * @param x the left edge of the region
     * @param y the top edge of the region
     * @param w the region width
     * @param h the region height
     * @return a new pixel buffer of size {@code w x h}
     * @throws IllegalArgumentException if the region exceeds the buffer bounds
     */
    public @NotNull PixelBuffer crop(int x, int y, int w, int h) {
        if (w < 0 || h < 0)
            throw new IllegalArgumentException("Crop dimensions must be non-negative");
        if (x < 0 || y < 0 || x + w > this.width || y + h > this.height)
            throw new IllegalArgumentException("Crop region (%d,%d %dx%d) exceeds buffer bounds %dx%d".formatted(x, y, w, h, this.width, this.height));

        int[] result = new int[w * h];
        for (int row = 0; row < h; row++)
            System.arraycopy(this.pixels, (y + row) * this.width + x, result, row * w, w);
        return new PixelBuffer(result, w, h, this.hasAlpha);
    }

    /**
     * Mirrors the buffer in place along the vertical axis, swapping columns left-to-right.
     */
    public void flipHorizontal() {
        for (int y = 0; y < this.height; y++) {
            int row = y * this.width;
            int left = 0;
            int right = this.width - 1;
            while (left < right) {
                int tmp = this.pixels[row + left];
                this.pixels[row + left] = this.pixels[row + right];
                this.pixels[row + right] = tmp;
                left++;
                right--;
            }
        }
    }

    /**
     * Mirrors the buffer in place along the horizontal axis, swapping rows top-to-bottom.
     */
    public void flipVertical() {
        int[] tmp = new int[this.width];
        int top = 0;
        int bottom = this.height - 1;
        while (top < bottom) {
            int topOff = top * this.width;
            int botOff = bottom * this.width;
            System.arraycopy(this.pixels, topOff, tmp, 0, this.width);
            System.arraycopy(this.pixels, botOff, this.pixels, topOff, this.width);
            System.arraycopy(tmp, 0, this.pixels, botOff, this.width);
            top++;
            bottom--;
        }
    }

    /**
     * Resamples this buffer to the given dimensions using the specified filter.
     * <p>
     * {@link Resample#NEAREST} and {@link Resample#BILINEAR} sample directly from the pixel array;
     * {@link Resample#BICUBIC} delegates to {@link Graphics2D} with high-quality hints.
     *
     * @param w the target width
     * @param h the target height
     * @param filter the resampling filter
     * @return a new pixel buffer of size {@code w x h}
     * @throws IllegalArgumentException if {@code w} or {@code h} is not positive
     */
    public @NotNull PixelBuffer resize(int w, int h, @NotNull Resample filter) {
        if (w <= 0 || h <= 0)
            throw new IllegalArgumentException("Resize dimensions must be positive, got %dx%d".formatted(w, h));

        if (filter == Resample.BICUBIC) {
            BufferedImage src = this.toBufferedImage();
            BufferedImage dst = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
            Graphics2D g = dst.createGraphics();
            try {
                g.setComposite(AlphaComposite.Src);
                g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
                g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
                g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                g.drawImage(src, 0, 0, w, h, null);
            } finally {
                g.dispose();
            }
            return wrap(dst);
        }

        int[] result = new int[w * h];
        for (int y = 0; y < h; y++) {
            int outRow = y * w;
            if (filter == Resample.NEAREST) {
                int sy = (y * this.height) / h;
                for (int x = 0; x < w; x++) {
                    int sx = (x * this.width) / w;
                    result[outRow + x] = this.pixels[sy * this.width + sx];
                }
            } else {
                float srcY = ((y + 0.5f) * this.height) / h - 0.5f;
                for (int x = 0; x < w; x++) {
                    float srcX = ((x + 0.5f) * this.width) / w - 0.5f;
                    result[outRow + x] = sampleBilinear(this.pixels, srcX, srcY);
                }
            }
        }

        return new PixelBuffer(result, w, h, this.hasAlpha);
    }

    /**
     * Returns a new pixel buffer rotated 180 degrees.
     *
     * @return a rotated copy with the same dimensions
     */
    public @NotNull PixelBuffer rotate180() {
        int n = this.pixels.length;
        int[] result = new int[n];

        for (int i = 0; i < n; i++)
            result[i] = this.pixels[n - 1 - i];

        return new PixelBuffer(result, this.width, this.height, this.hasAlpha);
    }

    /**
     * Returns a new pixel buffer rotated 270 degrees clockwise (90 degrees counter-clockwise).
     *
     * @return a rotated copy with swapped dimensions
     */
    public @NotNull PixelBuffer rotate270() {
        int w = this.width;
        int h = this.height;
        int[] result = new int[w * h];

        for (int y = 0; y < h; y++) {
            int row = y * w;
            for (int x = 0; x < w; x++)
                result[(w - 1 - x) * h + y] = this.pixels[row + x];
        }

        return new PixelBuffer(result, h, w, this.hasAlpha);
    }

    /**
     * Returns a new pixel buffer rotated 90 degrees clockwise.
     *
     * @return a rotated copy with swapped dimensions
     */
    public @NotNull PixelBuffer rotate90() {
        int w = this.width;
        int h = this.height;
        int[] result = new int[w * h];

        for (int y = 0; y < h; y++) {
            int row = y * w;

            for (int x = 0; x < w; x++)
                result[x * h + (h - 1 - y)] = this.pixels[row + x];
        }

        return new PixelBuffer(result, h, w, this.hasAlpha);
    }

    /**
     * Returns a new pixel buffer cropped to {@link #opaqueBounds()}.
     * <p>
     * Falls back to a plain {@link #copy()} when the buffer is fully opaque or fully transparent.
     *
     * @return a trimmed pixel buffer
     */
    public @NotNull PixelBuffer trim() {
        Rectangle r = opaqueBounds();
        if (r.width == 0 || r.height == 0) return copy();
        if (r.width == this.width && r.height == this.height) return copy();
        return crop(r.x, r.y, r.width, r.height);
    }

    // --- color operations ---

    /**
     * Fills the entire buffer with the given ARGB colour. Discards any existing content.
     *
     * @param argb the fill colour
     */
    public void fill(int argb) {
        Arrays.fill(this.pixels, argb);
    }

    /**
     * Fills a rectangular region with the given ARGB colour. Out-of-bounds areas are silently
     * clipped; each row is written via {@link Arrays#fill}.
     *
     * @param x the left edge
     * @param y the top edge
     * @param w the rectangle width
     * @param h the rectangle height
     * @param argb the fill colour
     */
    public void fillRect(int x, int y, int w, int h, int argb) {
        int x0 = Math.max(0, x);
        int y0 = Math.max(0, y);
        int x1 = Math.min(this.width, x + w);
        int y1 = Math.min(this.height, y + h);

        for (int row = y0; row < y1; row++) {
            int offset = row * this.width;
            Arrays.fill(this.pixels, offset + x0, offset + x1, argb);
        }
    }

    /**
     * Converts every pixel to its luma-weighted grayscale equivalent in place, preserving alpha.
     */
    public void grayscale() {
        for (int i = 0; i < this.pixels.length; i++) {
            int p = this.pixels[i];
            int a = (p >>> 24) & 0xFF;
            int y = Math.clamp((int) ColorMath.luma(p | 0xFF000000), 0, 255);
            this.pixels[i] = (a << 24) | (y << 16) | (y << 8) | y;
        }
    }

    /**
     * Inverts the RGB channels of every pixel in place, preserving alpha.
     */
    public void invert() {
        for (int i = 0; i < this.pixels.length; i++)
            this.pixels[i] ^= 0x00FFFFFF;
    }

    /**
     * Multiplies every pixel's alpha channel by the given factor in place.
     * <p>
     * The factor is clamped to {@code [0, 1]}. RGB channels are left untouched. Note that the
     * {@link #hasAlpha()} flag is not changed by this call; callers that begin with an opaque
     * buffer and want the result to report meaningful alpha should re-wrap the pixel array with
     * {@link #of(int[], int, int, boolean)}.
     *
     * @param factor the alpha multiplier, clamped to {@code [0, 1]}
     */
    public void multiplyAlpha(float factor) {
        factor = Math.clamp(factor, 0f, 1f);
        if (factor == 1f) return;
        for (int i = 0; i < this.pixels.length; i++) {
            int p = this.pixels[i];
            int a = (p >>> 24) & 0xFF;
            int newA = Math.round(a * factor);
            this.pixels[i] = (newA << 24) | (p & 0x00FFFFFF);
        }
    }

    /**
     * Premultiplies each pixel's RGB channels by its alpha in place.
     * <p>
     * Required for correct filtering, resampling, or compositing with libraries that expect
     * premultiplied inputs.
     */
    public void premultiplyAlpha() {
        for (int i = 0; i < this.pixels.length; i++) {
            int p = this.pixels[i];
            int a = (p >>> 24) & 0xFF;
            if (a == 0xFF) continue;
            if (a == 0) {
                this.pixels[i] = 0;
                continue;
            }
            int r = (((p >>> 16) & 0xFF) * a) / 255;
            int g = (((p >>> 8) & 0xFF) * a) / 255;
            int b = ((p & 0xFF) * a) / 255;
            this.pixels[i] = (a << 24) | (r << 16) | (g << 8) | b;
        }
    }

    /**
     * Un-premultiplies each pixel's RGB channels in place, dividing by alpha.
     * <p>
     * Inverse of {@link #premultiplyAlpha()}. Results are clamped to {@code [0, 255]} to guard
     * against accumulated floating-point drift.
     */
    public void unpremultiplyAlpha() {
        for (int i = 0; i < this.pixels.length; i++) {
            int p = this.pixels[i];
            int a = (p >>> 24) & 0xFF;
            if (a == 0xFF || a == 0) continue;
            int r = Math.min(255, (((p >>> 16) & 0xFF) * 255) / a);
            int g = Math.min(255, (((p >>> 8) & 0xFF) * 255) / a);
            int b = Math.min(255, ((p & 0xFF) * 255) / a);
            this.pixels[i] = (a << 24) | (r << 16) | (g << 8) | b;
        }
    }

    // --- compositing ---

    /**
     * Composites the source buffer onto this buffer at the given origin using standard source-over
     * alpha compositing.
     *
     * @param source the source buffer
     * @param dx the destination x origin
     * @param dy the destination y origin
     */
    public void blit(@NotNull PixelBuffer source, int dx, int dy) {
        blit(source, dx, dy, BlendMode.NORMAL);
    }

    /**
     * Composites the source buffer onto this buffer at the given origin using the specified blend
     * mode. Out-of-bounds pixels are silently clipped.
     * <p>
     * Fast paths are taken when {@code mode} is {@link BlendMode#NORMAL}: opaque sources use a
     * per-row {@link System#arraycopy}; alpha sources inline the fully-opaque and fully-transparent
     * short-circuits.
     *
     * @param source the source buffer
     * @param dx the destination x origin
     * @param dy the destination y origin
     * @param mode the blend mode to use for compositing
     */
    public void blit(@NotNull PixelBuffer source, int dx, int dy, @NotNull BlendMode mode) {
        int srcX0 = Math.max(0, -dx);
        int srcY0 = Math.max(0, -dy);
        int srcX1 = Math.min(source.width, this.width - dx);
        int srcY1 = Math.min(source.height, this.height - dy);

        if (srcX0 >= srcX1 || srcY0 >= srcY1) return;

        if (mode == BlendMode.NORMAL && !source.hasAlpha) {
            int rowWidth = srcX1 - srcX0;
            for (int y = srcY0; y < srcY1; y++) {
                int srcOff = y * source.width + srcX0;
                int dstOff = (dy + y) * this.width + dx + srcX0;
                System.arraycopy(source.pixels, srcOff, this.pixels, dstOff, rowWidth);
            }
            return;
        }

        if (mode == BlendMode.NORMAL) {
            for (int y = srcY0; y < srcY1; y++) {
                int srcRow = y * source.width;
                int dstRow = (dy + y) * this.width + dx;
                for (int x = srcX0; x < srcX1; x++) {
                    int src = source.pixels[srcRow + x];
                    int sa = (src >>> 24) & 0xFF;
                    if (sa == 0) continue;
                    if (sa == 0xFF) {
                        this.pixels[dstRow + x] = src;
                        continue;
                    }
                    int dstIdx = dstRow + x;
                    this.pixels[dstIdx] = ColorMath.blend(src, this.pixels[dstIdx], BlendMode.NORMAL);
                }
            }
            return;
        }

        for (int y = srcY0; y < srcY1; y++) {
            int srcRow = y * source.width;
            int dstRow = (dy + y) * this.width + dx;
            for (int x = srcX0; x < srcX1; x++) {
                int src = source.pixels[srcRow + x];
                if (ColorMath.alpha(src) == 0) continue;
                int dstIdx = dstRow + x;
                this.pixels[dstIdx] = ColorMath.blend(src, this.pixels[dstIdx], mode);
            }
        }
    }

    /**
     * Composites the source buffer onto this buffer at the given origin, rescaling to the specified
     * width and height using nearest-neighbor sampling with source-over alpha compositing.
     *
     * @param source the source buffer
     * @param dx the destination x origin
     * @param dy the destination y origin
     * @param dw the destination width
     * @param dh the destination height
     */
    public void blitScaled(@NotNull PixelBuffer source, int dx, int dy, int dw, int dh) {
        int x0 = Math.max(0, -dx);
        int y0 = Math.max(0, -dy);
        int x1 = Math.min(dw, this.width - dx);
        int y1 = Math.min(dh, this.height - dy);

        for (int y = y0; y < y1; y++) {
            int dstRow = (dy + y) * this.width;
            int sy = (y * source.height) / dh;

            for (int x = x0; x < x1; x++) {
                int sx = (x * source.width) / dw;
                int src = source.pixels[sy * source.width + sx];
                if (ColorMath.alpha(src) == 0) continue;
                int dstIdx = dstRow + dx + x;
                this.pixels[dstIdx] = ColorMath.blend(src, this.pixels[dstIdx], BlendMode.NORMAL);
            }
        }
    }

    /**
     * Tints the source buffer and composites it onto this buffer at the given origin using the
     * specified blend mode.
     *
     * @param source the source buffer
     * @param dx the destination x origin
     * @param dy the destination y origin
     * @param argbTint the ARGB tint colour
     * @param mode the blend mode to use for compositing
     */
    public void blitTinted(@NotNull PixelBuffer source, int dx, int dy, int argbTint, @NotNull BlendMode mode) {
        blit(ColorMath.tint(source, argbTint), dx, dy, mode);
    }

    /**
     * Creates a new pixel buffer by linearly interpolating between two buffers per pixel.
     * <p>
     * Each ARGB channel is blended independently. When the two buffers have different dimensions,
     * the result uses the minimum width and height.
     *
     * @param a the source buffer (alpha = 0)
     * @param b the target buffer (alpha = 1)
     * @param alpha the blend factor, clamped to {@code [0, 1]}
     * @return a new pixel buffer containing the blended result
     */
    public static @NotNull PixelBuffer lerp(@NotNull PixelBuffer a, @NotNull PixelBuffer b, float alpha) {
        alpha = Math.clamp(alpha, 0f, 1f);
        float inverse = 1f - alpha;
        int width = Math.min(a.width, b.width);
        int height = Math.min(a.height, b.height);
        int[] result = new int[width * height];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int s = a.pixels[y * a.width + x];
                int d = b.pixels[y * b.width + x];

                int ra = clampByte(((s >> 24) & 0xFF) * inverse + ((d >> 24) & 0xFF) * alpha);
                int rr = clampByte(((s >> 16) & 0xFF) * inverse + ((d >> 16) & 0xFF) * alpha);
                int rg = clampByte(((s >> 8) & 0xFF) * inverse + ((d >> 8) & 0xFF) * alpha);
                int rb = clampByte((s & 0xFF) * inverse + (d & 0xFF) * alpha);

                result[y * width + x] = (ra << 24) | (rr << 16) | (rg << 8) | rb;
            }
        }

        return PixelBuffer.of(result, width, height, a.hasAlpha || b.hasAlpha);
    }

    // --- filters ---

    /**
     * Applies FXAA post-processing in place, smoothing aliased edges.
     * <p>
     * Uses luma-based edge detection with bilinear sampling along detected edge directions.
     * Processing is parallelized across scanlines. The one-pixel border is left unmodified.
     */
    public void applyFxaa() {
        int[] temp = this.pixels.clone();

        final float fxaaReduceMin = 1.0f / 128.0f;
        final float fxaaReduceMul = 1.0f / 4.0f;
        final float fxaaSpanMax = 8.0f;

        IntStream.range(1, this.height - 1).parallel().forEach(y -> {
            for (int x = 1; x < this.width - 1; x++) {
                int rgbNw = temp[(y - 1) * this.width + (x - 1)];
                int rgbNe = temp[(y - 1) * this.width + (x + 1)];
                int rgbSw = temp[(y + 1) * this.width + (x - 1)];
                int rgbSe = temp[(y + 1) * this.width + (x + 1)];
                int rgbM = temp[y * this.width + x];

                float lumaNw = ColorMath.luma(rgbNw);
                float lumaNe = ColorMath.luma(rgbNe);
                float lumaSw = ColorMath.luma(rgbSw);
                float lumaSe = ColorMath.luma(rgbSe);
                float lumaM = ColorMath.luma(rgbM);

                float lumaMin = Math.min(lumaM, Math.min(Math.min(lumaNw, lumaNe), Math.min(lumaSw, lumaSe)));
                float lumaMax = Math.max(lumaM, Math.max(Math.max(lumaNw, lumaNe), Math.max(lumaSw, lumaSe)));

                float contrast = lumaMax - lumaMin;

                if (contrast < Math.max(0.0156f, lumaMax * 0.0312f))
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

                int sample1 = sampleBilinear(temp,
                    x + dirX * (1.0f / 3.0f - 0.5f),
                    y + dirY * (1.0f / 3.0f - 0.5f));
                int sample2 = sampleBilinear(temp,
                    x + dirX * (2.0f / 3.0f - 0.5f),
                    y + dirY * (2.0f / 3.0f - 0.5f));

                int s1a = (sample1 >> 24) & 0xFF;
                int s1r = (sample1 >> 16) & 0xFF;
                int s1g = (sample1 >> 8) & 0xFF;
                int s1b = sample1 & 0xFF;

                int s2a = (sample2 >> 24) & 0xFF;
                int s2r = (sample2 >> 16) & 0xFF;
                int s2g = (sample2 >> 8) & 0xFF;
                int s2b = sample2 & 0xFF;

                float r = (s1r * s1a + s2r * s2a) * 0.5f;
                float g = (s1g * s1a + s2g * s2a) * 0.5f;
                float b = (s1b * s1a + s2b * s2a) * 0.5f;
                float a = (s1a + s2a) * 0.5f;

                if (a > 0) {
                    int ri = Math.clamp((int) (r / a), 0, 255);
                    int gi = Math.clamp((int) (g / a), 0, 255);
                    int bi = Math.clamp((int) (b / a), 0, 255);
                    int ai = Math.clamp((int) a, 0, 255);
                    this.pixels[y * this.width + x] = (ai << 24) | (ri << 16) | (gi << 8) | bi;
                } else {
                    this.pixels[y * this.width + x] = 0;
                }
            }
        });
    }

    // --- conversion ---

    /**
     * Returns a new pixel buffer with a cloned pixel array.
     *
     * @return an independent copy of this buffer
     */
    public @NotNull PixelBuffer copy() {
        return new PixelBuffer(this.pixels.clone(), this.width, this.height, this.hasAlpha);
    }

    /**
     * Creates a {@link BufferedImage} backed by this buffer's pixel array (zero copy).
     * <p>
     * The returned image shares the same {@code int[]} as this buffer, so edits made through either
     * API (pixel buffer or {@link Graphics2D}) are immediately visible from the other without a
     * flush step.
     *
     * @return a buffered image sharing this buffer's pixel array
     */
    public @NotNull BufferedImage toBufferedImage() {
        DataBufferInt db = new DataBufferInt(this.pixels, this.pixels.length);
        int[] masks = { 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000 };
        SinglePixelPackedSampleModel sm = new SinglePixelPackedSampleModel(
            DataBuffer.TYPE_INT, this.width, this.height, masks
        );
        return new BufferedImage(
            ColorModel.getRGBdefault(),
            Raster.createWritableRaster(sm, db, null),
            false,
            null
        );
    }

    // --- value semantics ---

    /**
     * Compares this buffer with another for deep pixel equality.
     *
     * @param o the object to compare
     * @return {@code true} if the other object is a pixel buffer with the same dimensions, alpha
     *     flag, and pixel contents
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof PixelBuffer other)) return false;
        return this.width == other.width
            && this.height == other.height
            && this.hasAlpha == other.hasAlpha
            && Arrays.equals(this.pixels, other.pixels);
    }

    /**
     * Computes a hash code consistent with {@link #equals(Object)}.
     *
     * @return a hash derived from the dimensions, alpha flag, and pixel contents
     */
    @Override
    public int hashCode() {
        int h = Integer.hashCode(this.width);
        h = 31 * h + Integer.hashCode(this.height);
        h = 31 * h + Boolean.hashCode(this.hasAlpha);
        h = 31 * h + Arrays.hashCode(this.pixels);
        return h;
    }

    // --- private helpers ---

    private static int clampByte(float value) {
        return (int) Math.clamp(value, 0f, 255f);
    }

    private int sampleBilinear(int @NotNull [] source, float x, float y) {
        int ix = (int) Math.floor(x);
        int iy = (int) Math.floor(y);
        float fx = x - ix;
        float fy = y - iy;

        ix = Math.clamp(ix, 0, this.width - 2);
        iy = Math.clamp(iy, 0, this.height - 2);

        int c00 = source[iy * this.width + ix];
        int c10 = source[iy * this.width + ix + 1];
        int c01 = source[(iy + 1) * this.width + ix];
        int c11 = source[(iy + 1) * this.width + ix + 1];

        float w00 = (1 - fx) * (1 - fy);
        float w10 = fx * (1 - fy);
        float w01 = (1 - fx) * fy;
        float w11 = fx * fy;

        float r = ((c00 >> 16) & 0xFF) * w00 + ((c10 >> 16) & 0xFF) * w10
            + ((c01 >> 16) & 0xFF) * w01 + ((c11 >> 16) & 0xFF) * w11;
        float g = ((c00 >> 8) & 0xFF) * w00 + ((c10 >> 8) & 0xFF) * w10
            + ((c01 >> 8) & 0xFF) * w01 + ((c11 >> 8) & 0xFF) * w11;
        float b = (c00 & 0xFF) * w00 + (c10 & 0xFF) * w10
            + (c01 & 0xFF) * w01 + (c11 & 0xFF) * w11;
        float a = ((c00 >> 24) & 0xFF) * w00 + ((c10 >> 24) & 0xFF) * w10
            + ((c01 >> 24) & 0xFF) * w01 + ((c11 >> 24) & 0xFF) * w11;

        int ri = Math.clamp((int) r, 0, 255);
        int gi = Math.clamp((int) g, 0, 255);
        int bi = Math.clamp((int) b, 0, 255);
        int ai = Math.clamp((int) a, 0, 255);

        return (ai << 24) | (ri << 16) | (gi << 8) | bi;
    }

}
