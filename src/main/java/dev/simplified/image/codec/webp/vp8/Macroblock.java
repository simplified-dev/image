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
     * Fills this macroblock from ARGB pixel data at the given position.
     *
     * @param pixels the full image ARGB pixel array
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
                int px = Math.min(startX + dx, imageWidth - 1);
                int argb = pixels[py * imageWidth + px];

                int r = (argb >> 16) & 0xFF;
                int g = (argb >> 8) & 0xFF;
                int b = argb & 0xFF;

                // ITU-R BT.601 conversion
                y[dy * 16 + dx] = (short) clamp(((66 * r + 129 * g + 25 * b + 128) >> 8) + 16);

                if ((dy & 1) == 0 && (dx & 1) == 0) {
                    int cx = dx >> 1;
                    int cy = dy >> 1;
                    cb[cy * 8 + cx] = (short) clamp(((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128);
                    cr[cy * 8 + cx] = (short) clamp(((112 * r - 94 * g - 18 * b + 128) >> 8) + 128);
                }
            }
        }
    }

    private static int clamp(int value) {
        return Math.clamp(value, 0, 255);
    }

}
