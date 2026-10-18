package dev.simplified.image.codec.webp.vp8;

import org.jetbrains.annotations.NotNull;

/**
 * VP8 macroblock source samples for a 16x16 pixel region.
 * <p>
 * Holds the 256 luma samples and the 8x8 Cb and Cr samples the encoder predicts and codes,
 * converted from ARGB by {@link #fromARGB}.
 */
final class Macroblock {

    /**
     * Luma samples (16x16 = 256 values).
     */
    final short[] y = new short[256];
    /**
     * Cb chroma samples (8x8 = 64 values).
     */
    final short[] cb = new short[64];
    /**
     * Cr chroma samples (8x8 = 64 values).
     */
    final short[] cr = new short[64];

    /**
     * Fills this macroblock from ARGB pixel data at the given position, with ITU-R BT.601 coefficients.
     * <p>
     * Each luma sample converts its own pixel. Each chroma sample converts the sum of its 2x2 block of
     * pixels, as libwebp's {@code AccumulateRGBA} does: a block that is fully opaque or fully
     * transparent sums its colour as it is, and a partly transparent block weights each pixel's colour
     * by its alpha, so a transparent pixel adds no colour. A pixel past the image's right or bottom edge
     * reads the edge pixel.
     *
     * @param pixels the full image ARGB pixel array, holding straight colour
     * @param mbX the macroblock column index
     * @param mbY the macroblock row index
     * @param imageWidth the full image width
     * @param imageHeight the full image height
     */
    void fromARGB(int @NotNull [] pixels, int mbX, int mbY, int imageWidth, int imageHeight) {
        int startX = mbX * 16;
        int startY = mbY * 16;

        for (int dy = 0; dy < 16; dy++) {
            int py = Math.min(startY + dy, imageHeight - 1);

            for (int dx = 0; dx < 16; dx++) {
                int argb = pixels[py * imageWidth + Math.min(startX + dx, imageWidth - 1)];
                int r = (argb >> 16) & 0xFF;
                int g = (argb >> 8) & 0xFF;
                int b = argb & 0xFF;
                y[dy * 16 + dx] = (short) clamp(((66 * r + 129 * g + 25 * b + 128) >> 8) + 16);
            }
        }

        for (int cy = 0; cy < 8; cy++) {
            int row0 = Math.min(startY + 2 * cy, imageHeight - 1) * imageWidth;
            int row1 = Math.min(startY + 2 * cy + 1, imageHeight - 1) * imageWidth;

            for (int cx = 0; cx < 8; cx++) {
                int col0 = Math.min(startX + 2 * cx, imageWidth - 1);
                int col1 = Math.min(startX + 2 * cx + 1, imageWidth - 1);
                int[] sum = sumBlock(pixels[row0 + col0], pixels[row0 + col1], pixels[row1 + col0], pixels[row1 + col1]);

                // The block sum is four pixels' worth, so the rounding and the shift each carry the
                // factor of four a single pixel's (x + 128) >> 8 would not.
                cb[cy * 8 + cx] = (short) clamp(((-38 * sum[0] - 74 * sum[1] + 112 * sum[2] + 512) >> 10) + 128);
                cr[cy * 8 + cx] = (short) clamp(((112 * sum[0] - 94 * sum[1] - 18 * sum[2] + 512) >> 10) + 128);
            }
        }
    }

    /**
     * Sums the red, green and blue of a 2x2 block at four times one pixel's scale. A block whose alpha
     * sums to zero or to four full alphas sums its colour as it is; any other block weights each pixel
     * by its alpha and scales the weighted mean back to four pixels' worth, rounded once.
     *
     * @param p0 the top-left pixel
     * @param p1 the top-right pixel
     * @param p2 the bottom-left pixel
     * @param p3 the bottom-right pixel
     * @return the red, green and blue sums
     */
    private static int @NotNull [] sumBlock(int p0, int p1, int p2, int p3) {
        int[] block = { p0, p1, p2, p3 };
        int alpha = 0;
        for (int pixel : block)
            alpha += pixel >>> 24;

        int[] sum = new int[3];
        boolean weigh = alpha != 0 && alpha != 4 * 0xFF;
        for (int pixel : block) {
            int weight = weigh ? pixel >>> 24 : 1;
            sum[0] += ((pixel >> 16) & 0xFF) * weight;
            sum[1] += ((pixel >> 8) & 0xFF) * weight;
            sum[2] += (pixel & 0xFF) * weight;
        }

        if (weigh) {
            for (int i = 0; i < 3; i++)
                sum[i] = (4 * sum[i] + (alpha >> 1)) / alpha;
        }

        return sum;
    }

    private static int clamp(int value) {
        return Math.clamp(value, 0, 255);
    }

}
