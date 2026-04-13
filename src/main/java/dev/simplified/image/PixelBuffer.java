package dev.simplified.image;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.experimental.Accessors;
import org.jetbrains.annotations.NotNull;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.util.stream.IntStream;

/**
 * A zero-copy wrapper over {@code int[]} ARGB pixel data.
 * <p>
 * For {@link BufferedImage#TYPE_INT_ARGB} images, the backing array is referenced
 * directly without copying. For other image types, pixel data is extracted once
 * into a new array, then accessed without further copies.
 */
@Getter
@RequiredArgsConstructor(access = AccessLevel.PRIVATE)
public class PixelBuffer {

    private final int @NotNull [] pixels;
    @Accessors(fluent = true)
    private final int width;
    @Accessors(fluent = true)
    private final int height;
    @Accessors(fluent = true)
    private final boolean hasAlpha;

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
     * Creates a new {@link BufferedImage} from this buffer's pixel data.
     * <p>
     * The returned image is {@link BufferedImage#TYPE_INT_ARGB} with the pixel array
     * set directly on its raster.
     *
     * @return a new buffered image containing this buffer's pixels
     */
    public @NotNull BufferedImage toBufferedImage() {
        BufferedImage image = new BufferedImage(this.width, this.height, BufferedImage.TYPE_INT_ARGB);
        image.setRGB(0, 0, this.width, this.height, this.pixels, 0, this.width);
        return image;
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

    private int sampleBilinear(int[] source, float x, float y) {
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
