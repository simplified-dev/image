package dev.simplified.image;

import lombok.Getter;
import lombok.experimental.Accessors;
import org.jetbrains.annotations.NotNull;

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
 * For {@link BufferedImage#TYPE_INT_ARGB} images, the backing array is referenced
 * directly without copying. For other image types, pixel data is extracted once
 * into a new array, then accessed without further copies.
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
     * Wraps the pixel data of a {@link BufferedImage}.
     * <p>
     * If the image is {@link BufferedImage#TYPE_INT_ARGB}, the underlying data buffer
     * is referenced directly (zero copy). Otherwise, pixel data is extracted via
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

    /**
     * Creates a {@link BufferedImage} backed by this buffer's pixel array (zero copy).
     * <p>
     * The returned image shares the same {@code int[]} as this buffer, so edits made through
     * either API (pixel buffer or {@link java.awt.Graphics2D}) are immediately visible from
     * the other without a flush step.
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
     * Returns a new pixel buffer with a cloned pixel array.
     *
     * @return an independent copy of this buffer
     */
    public @NotNull PixelBuffer copy() {
        return new PixelBuffer(this.pixels.clone(), this.width, this.height, this.hasAlpha);
    }

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
     * clipped. Each row is filled via {@link Arrays#fill} for JVM vectorization.
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
     * Composites the source buffer onto this buffer at the given origin using standard
     * source-over alpha compositing.
     *
     * @param source the source buffer
     * @param dx the destination x origin
     * @param dy the destination y origin
     */
    public void blit(@NotNull PixelBuffer source, int dx, int dy) {
        blit(source, dx, dy, BlendMode.NORMAL);
    }

    /**
     * Composites the source buffer onto this buffer at the given origin using the specified
     * blend mode. Out-of-bounds pixels are silently clipped.
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

        for (int y = srcY0; y < srcY1; y++) {
            int dstRow = (dy + y) * this.width;
            int srcRow = y * source.width;

            for (int x = srcX0; x < srcX1; x++) {
                int src = source.pixels[srcRow + x];
                if (ColorMath.alpha(src) == 0) continue;
                int dst = this.pixels[dstRow + dx + x];
                this.pixels[dstRow + dx + x] = ColorMath.blend(src, dst, mode);
            }
        }
    }

    /**
     * Composites the source buffer onto this buffer at the given origin, rescaling to the
     * specified width and height using nearest-neighbor sampling with source-over alpha
     * compositing.
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
     * Each ARGB channel is blended independently. When the two buffers have different
     * dimensions, the result uses the minimum width and height.
     *
     * @param a the source buffer (alpha = 0)
     * @param b the target buffer (alpha = 1)
     * @param alpha the blend factor, clamped to [0, 1]
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

    private static int clampByte(float value) {
        return (int) Math.clamp(value, 0f, 255f);
    }

    /**
     * Applies FXAA post-processing in place, smoothing aliased edges.
     * <p>
     * Uses luma-based edge detection with bilinear sampling along detected edge
     * directions. Processing is parallelized across scanlines. The one-pixel border
     * is left unmodified.
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

                float lumaNw = luma(rgbNw);
                float lumaNe = luma(rgbNe);
                float lumaSw = luma(rgbSw);
                float lumaSe = luma(rgbSe);
                float lumaM = luma(rgbM);

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

    private static float luma(int argb) {
        int a = (argb >> 24) & 0xFF;
        int r = (argb >> 16) & 0xFF;
        int g = (argb >> 8) & 0xFF;
        int b = argb & 0xFF;
        return (r * 0.299f + g * 0.587f + b * 0.114f) * (a / 255f);
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
