package dev.simplified.image.codec.webp.lossy;

import org.jetbrains.annotations.NotNull;

/**
 * VP8 macroblock data structure representing a 16x16 pixel region.
 * <p>
 * Contains 16 luma 4x4 sub-blocks, 4 Cb and 4 Cr 4x4 sub-blocks,
 * prediction modes, quantized coefficients, and reconstructed pixels.
 * Handles RGB-to-YCbCr conversion on input and YCbCr-to-RGB on output.
 */
final class Macroblock {

    /** Luma samples (16x16 = 256 values). */
    final short[] y = new short[256];
    /** Cb chroma samples (8x8 = 64 values). */
    final short[] cb = new short[64];
    /** Cr chroma samples (8x8 = 64 values). */
    final short[] cr = new short[64];

    /** Reconstructed luma (after decode or encode round-trip). */
    final short[] reconY = new short[256];
    /** Reconstructed Cb. */
    final short[] reconCb = new short[64];
    /** Reconstructed Cr. */
    final short[] reconCr = new short[64];

    /** Selected 16x16 prediction mode. */
    int yMode = IntraPrediction.DC_PRED;
    /** Selected chroma prediction mode. */
    int uvMode = IntraPrediction.DC_PRED;

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

    /**
     * Converts reconstructed YCbCr data back to ARGB and writes to the pixel array.
     *
     * @param pixels the output ARGB pixel array
     * @param mbX the macroblock column index
     * @param mbY the macroblock row index
     * @param imageWidth the full image width
     * @param imageHeight the full image height
     */
    void toARGB(int @NotNull [] pixels, int mbX, int mbY, int imageWidth, int imageHeight) {
        int startX = mbX * 16;
        int startY = mbY * 16;

        for (int dy = 0; dy < 16; dy++) {
            int py = startY + dy;
            if (py >= imageHeight) break;

            for (int dx = 0; dx < 16; dx++) {
                int px = startX + dx;
                if (px >= imageWidth) break;

                int yVal = reconY[dy * 16 + dx] - 16;
                int cbVal = reconCb[(dy >> 1) * 8 + (dx >> 1)] - 128;
                int crVal = reconCr[(dy >> 1) * 8 + (dx >> 1)] - 128;

                // ITU-R BT.601 inverse
                int r = clamp((298 * yVal + 409 * crVal + 128) >> 8);
                int g = clamp((298 * yVal - 100 * cbVal - 208 * crVal + 128) >> 8);
                int b = clamp((298 * yVal + 516 * cbVal + 128) >> 8);

                pixels[py * imageWidth + px] = 0xFF000000 | (r << 16) | (g << 8) | b;
            }
        }
    }

    private static int clamp(int value) {
        return Math.clamp(value, 0, 255);
    }

}
