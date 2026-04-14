package dev.simplified.image.codec.webp.lossless;

import org.jetbrains.annotations.NotNull;

/**
 * VP8L reversible image transforms applied during lossless encoding and decoding.
 * <p>
 * Transforms are applied in order during encoding (forward) and in reverse
 * order during decoding (inverse). All transforms operate on {@code int[]}
 * ARGB pixel arrays in-place.
 */
sealed interface VP8LTransform {

    /** Applies the inverse (decoding) transform in-place. */
    void inverseTransform(int @NotNull [] pixels, int width, int height);

    /** Applies the forward (encoding) transform in-place. */
    void forwardTransform(int @NotNull [] pixels, int width, int height);

    // -----------------------------------------------------------------------
    // Subtract Green Transform
    // -----------------------------------------------------------------------

    /**
     * Subtracts the green channel from red and blue during encoding,
     * adds it back during decoding. Stateless.
     */
    record SubtractGreen() implements VP8LTransform {

        @Override
        public void inverseTransform(int @NotNull [] pixels, int width, int height) {
            for (int i = 0; i < pixels.length; i++) {
                int argb = pixels[i];
                int green = (argb >> 8) & 0xFF;
                int red = ((argb >> 16) & 0xFF) + green;
                int blue = (argb & 0xFF) + green;
                pixels[i] = (argb & 0xFF00FF00) | ((red & 0xFF) << 16) | (blue & 0xFF);
            }
        }

        @Override
        public void forwardTransform(int @NotNull [] pixels, int width, int height) {
            for (int i = 0; i < pixels.length; i++) {
                int argb = pixels[i];
                int green = (argb >> 8) & 0xFF;
                int red = ((argb >> 16) & 0xFF) - green;
                int blue = (argb & 0xFF) - green;
                pixels[i] = (argb & 0xFF00FF00) | ((red & 0xFF) << 16) | (blue & 0xFF);
            }
        }

    }

    // -----------------------------------------------------------------------
    // Color Indexing Transform (Palette)
    // -----------------------------------------------------------------------

    /**
     * Maps pixels to palette indices when the image has 256 or fewer unique colors.
     * <p>
     * Sub-pixel packing is used for small palettes: 2 colors use 1 bit/pixel,
     * 4 colors use 2 bits, 16 colors use 4 bits.
     */
    record ColorIndexing(int @NotNull [] palette, int bitsPerPixel) implements VP8LTransform {

        @Override
        public void inverseTransform(int @NotNull [] pixels, int width, int height) {
            // VP8L packs palette indices into the GREEN channel of each decoded pixel,
            // not into the low bits of the whole ARGB word. Other channels carry no
            // index data at this stage.
            if (bitsPerPixel < 8) {
                // Sub-bit-packed: each packed pixel's green byte carries `pixelsPerByte`
                // indices, least-significant bits first. Walk backwards so the source
                // slot is always read before it's overwritten by the larger expanded row.
                int pixelsPerByte = 8 / bitsPerPixel;
                int mask = (1 << bitsPerPixel) - 1;
                int packedWidth = (width + pixelsPerByte - 1) / pixelsPerByte;

                for (int y = height - 1; y >= 0; y--) {
                    for (int x = width - 1; x >= 0; x--) {
                        int packedX = x / pixelsPerByte;
                        int bitOffset = (x % pixelsPerByte) * bitsPerPixel;
                        int packedPixel = pixels[y * packedWidth + packedX];
                        int index = ((packedPixel >> 8) >> bitOffset) & mask;
                        pixels[y * width + x] = index < palette.length ? palette[index] : 0;
                    }
                }
            } else {
                // 1 index per pixel (8-bit): the index is the green byte.
                for (int i = pixels.length - 1; i >= 0; i--) {
                    int index = (pixels[i] >> 8) & 0xFF;
                    pixels[i] = index < palette.length ? palette[index] : 0;
                }
            }
        }

        @Override
        public void forwardTransform(int @NotNull [] pixels, int width, int height) {
            // Replace ARGB values with palette indices stored in the green channel with
            // alpha = 0xFF (matches the inverse transform's read location).
            java.util.Map<Integer, Integer> lookup = new java.util.HashMap<>();
            for (int i = 0; i < palette.length; i++)
                lookup.put(palette[i], i);

            for (int i = 0; i < pixels.length; i++) {
                Integer index = lookup.get(pixels[i]);
                int idx = index != null ? index : 0;
                pixels[i] = 0xFF000000 | (idx << 8);
            }
        }

    }

    // -----------------------------------------------------------------------
    // Predictor Transform
    // -----------------------------------------------------------------------

    /**
     * Spatial prediction filter with 14 modes.
     * <p>
     * Each block of pixels uses a prediction mode selected by a sub-resolution
     * "meta" image. The forward transform computes residuals (original - prediction);
     * the inverse transform adds prediction back to residuals.
     */
    record Predictor(int blockBits, int @NotNull [] blockModes, int blockWidth) implements VP8LTransform {

        /** Number of prediction filters defined in the VP8L spec. */
        static final int NUM_FILTERS = 14;

        /** Opaque black - the implicit predictor value for the top-left pixel (mode 0). */
        private static final int ARGB_BLACK = 0xFF000000;

        @Override
        public void inverseTransform(int @NotNull [] pixels, int width, int height) {
            // Top-left pixel: add the implicit opaque-black predictor (mode 0 / ARGB_BLACK).
            // libwebp's encoder emits the top-left residual relative to this constant; the
            // decoder must add it back, not pass the raw pixel through.
            pixels[0] = addPixels(pixels[0], ARGB_BLACK);

            // First row: L (left-pixel) prediction for every remaining pixel.
            for (int x = 1; x < width; x++)
                pixels[x] = addPixels(pixels[x], pixels[x - 1]);

            // Remaining rows
            for (int y = 1; y < height; y++) {
                int blockY = y >> blockBits;
                int rowStart = y * width;

                // First column: T (top-pixel) prediction.
                pixels[rowStart] = addPixels(pixels[rowStart], pixels[rowStart - width]);

                for (int x = 1; x < width; x++) {
                    int blockX = x >> blockBits;
                    int modeIdx = blockY * blockWidth + blockX;
                    int mode = modeIdx < blockModes.length ? (blockModes[modeIdx] >> 8) & 0xF : 0;
                    int predicted = predict(mode, pixels, x, y, width);
                    pixels[rowStart + x] = addPixels(pixels[rowStart + x], predicted);
                }
            }
        }

        @Override
        public void forwardTransform(int @NotNull [] pixels, int width, int height) {
            // Compute residuals (reverse of inverse)
            for (int y = height - 1; y >= 1; y--) {
                int blockY = y >> blockBits;
                int rowStart = y * width;

                for (int x = width - 1; x >= 1; x--) {
                    int blockX = x >> blockBits;
                    int modeIdx = blockY * blockWidth + blockX;
                    int mode = modeIdx < blockModes.length ? (blockModes[modeIdx] >> 8) & 0xF : 0;
                    int predicted = predict(mode, pixels, x, y, width);
                    pixels[rowStart + x] = subPixels(pixels[rowStart + x], predicted);
                }

                pixels[rowStart] = subPixels(pixels[rowStart], pixels[rowStart - width]);
            }

            for (int x = width - 1; x >= 1; x--)
                pixels[x] = subPixels(pixels[x], pixels[x - 1]);

            pixels[0] = subPixels(pixels[0], ARGB_BLACK);
        }

        private static int predict(int mode, int[] pixels, int x, int y, int width) {
            int idx = y * width + x;
            int left = pixels[idx - 1];
            int top = pixels[idx - width];
            int topLeft = pixels[idx - width - 1];
            int topRight = x + 1 < width ? pixels[idx - width + 1] : top;

            return switch (mode) {
                case 0 -> 0xFF000000; // Black
                case 1 -> left;
                case 2 -> top;
                case 3 -> topRight;
                case 4 -> topLeft;
                case 5 -> average2(average2(left, topRight), top);
                case 6 -> average2(left, topLeft);
                case 7 -> average2(left, top);
                case 8 -> average2(topLeft, top);
                case 9 -> average2(top, topRight);
                case 10 -> average2(average2(left, topLeft), average2(top, topRight));
                case 11 -> select(left, top, topLeft);
                case 12 -> clampAddSubFull(left, top, topLeft);
                case 13 -> clampAddSubHalf(average2(left, top), topLeft);
                default -> 0xFF000000;
            };
        }

    }

    // -----------------------------------------------------------------------
    // Color Transform
    // -----------------------------------------------------------------------

    /**
     * Cross-channel decorrelation transform.
     * <p>
     * Reduces redundancy between R, G, B channels by storing per-block
     * decorrelation coefficients in a sub-resolution "meta" image.
     */
    record ColorXform(int blockBits, int @NotNull [] transformData, int blockWidth) implements VP8LTransform {

        @Override
        public void inverseTransform(int @NotNull [] pixels, int width, int height) {
            for (int y = 0; y < height; y++) {
                int blockY = y >> blockBits;

                for (int x = 0; x < width; x++) {
                    int blockX = x >> blockBits;
                    int transform = transformData[blockY * blockWidth + blockX];

                    int greenToRed = (byte) ((transform >> 0) & 0xFF);
                    int greenToBlue = (byte) ((transform >> 8) & 0xFF);
                    int redToBlue = (byte) ((transform >> 16) & 0xFF);

                    int idx = y * width + x;
                    int argb = pixels[idx];
                    int green = (argb >> 8) & 0xFF;
                    int red = ((argb >> 16) & 0xFF) + colorTransformDelta(greenToRed, green);
                    int blue = (argb & 0xFF)
                        + colorTransformDelta(greenToBlue, green)
                        + colorTransformDelta(redToBlue, red & 0xFF);

                    pixels[idx] = (argb & 0xFF00FF00) | ((red & 0xFF) << 16) | (blue & 0xFF);
                }
            }
        }

        @Override
        public void forwardTransform(int @NotNull [] pixels, int width, int height) {
            for (int y = height - 1; y >= 0; y--) {
                int blockY = y >> blockBits;

                for (int x = width - 1; x >= 0; x--) {
                    int blockX = x >> blockBits;
                    int transform = transformData[blockY * blockWidth + blockX];

                    int greenToRed = (byte) ((transform >> 0) & 0xFF);
                    int greenToBlue = (byte) ((transform >> 8) & 0xFF);
                    int redToBlue = (byte) ((transform >> 16) & 0xFF);

                    int idx = y * width + x;
                    int argb = pixels[idx];
                    int green = (argb >> 8) & 0xFF;
                    int red = ((argb >> 16) & 0xFF) - colorTransformDelta(greenToRed, green);
                    int blue = (argb & 0xFF)
                        - colorTransformDelta(greenToBlue, green)
                        - colorTransformDelta(redToBlue, red & 0xFF);

                    pixels[idx] = (argb & 0xFF00FF00) | ((red & 0xFF) << 16) | (blue & 0xFF);
                }
            }
        }

        private static int colorTransformDelta(int t, int c) {
            return (t * c) >> 5;
        }

    }

    // -----------------------------------------------------------------------
    // Pixel arithmetic helpers (shared across transforms)
    // -----------------------------------------------------------------------

    private static int addPixels(int a, int b) {
        int alpha = ((a >> 24) & 0xFF) + ((b >> 24) & 0xFF);
        int red = ((a >> 16) & 0xFF) + ((b >> 16) & 0xFF);
        int green = ((a >> 8) & 0xFF) + ((b >> 8) & 0xFF);
        int blue = (a & 0xFF) + (b & 0xFF);
        return ((alpha & 0xFF) << 24) | ((red & 0xFF) << 16) | ((green & 0xFF) << 8) | (blue & 0xFF);
    }

    private static int subPixels(int a, int b) {
        int alpha = ((a >> 24) & 0xFF) - ((b >> 24) & 0xFF);
        int red = ((a >> 16) & 0xFF) - ((b >> 16) & 0xFF);
        int green = ((a >> 8) & 0xFF) - ((b >> 8) & 0xFF);
        int blue = (a & 0xFF) - (b & 0xFF);
        return ((alpha & 0xFF) << 24) | ((red & 0xFF) << 16) | ((green & 0xFF) << 8) | (blue & 0xFF);
    }

    private static int average2(int a, int b) {
        int alpha = (((a >> 24) & 0xFF) + ((b >> 24) & 0xFF)) / 2;
        int red = (((a >> 16) & 0xFF) + ((b >> 16) & 0xFF)) / 2;
        int green = (((a >> 8) & 0xFF) + ((b >> 8) & 0xFF)) / 2;
        int blue = ((a & 0xFF) + (b & 0xFF)) / 2;
        return (alpha << 24) | (red << 16) | (green << 8) | blue;
    }

    private static int select(int left, int top, int topLeft) {
        int pAlpha = ((left >> 24) & 0xFF) + ((top >> 24) & 0xFF) - ((topLeft >> 24) & 0xFF);
        int pRed = ((left >> 16) & 0xFF) + ((top >> 16) & 0xFF) - ((topLeft >> 16) & 0xFF);
        int pGreen = ((left >> 8) & 0xFF) + ((top >> 8) & 0xFF) - ((topLeft >> 8) & 0xFF);
        int pBlue = (left & 0xFF) + (top & 0xFF) - (topLeft & 0xFF);

        int dLeft = Math.abs(pAlpha - ((left >> 24) & 0xFF)) + Math.abs(pRed - ((left >> 16) & 0xFF))
            + Math.abs(pGreen - ((left >> 8) & 0xFF)) + Math.abs(pBlue - (left & 0xFF));
        int dTop = Math.abs(pAlpha - ((top >> 24) & 0xFF)) + Math.abs(pRed - ((top >> 16) & 0xFF))
            + Math.abs(pGreen - ((top >> 8) & 0xFF)) + Math.abs(pBlue - (top & 0xFF));

        return dLeft <= dTop ? left : top;
    }

    private static int clampAddSubFull(int a, int b, int c) {
        return clampChannel(a, b, c, 24) | clampChannel(a, b, c, 16) | clampChannel(a, b, c, 8) | clampChannel(a, b, c, 0);
    }

    private static int clampAddSubHalf(int a, int b) {
        int alpha = clamp(((a >> 24) & 0xFF) + (((a >> 24) & 0xFF) - ((b >> 24) & 0xFF)) / 2);
        int red = clamp(((a >> 16) & 0xFF) + (((a >> 16) & 0xFF) - ((b >> 16) & 0xFF)) / 2);
        int green = clamp(((a >> 8) & 0xFF) + (((a >> 8) & 0xFF) - ((b >> 8) & 0xFF)) / 2);
        int blue = clamp((a & 0xFF) + ((a & 0xFF) - (b & 0xFF)) / 2);
        return (alpha << 24) | (red << 16) | (green << 8) | blue;
    }

    private static int clampChannel(int a, int b, int c, int shift) {
        int va = (a >> shift) & 0xFF;
        int vb = (b >> shift) & 0xFF;
        int vc = (c >> shift) & 0xFF;
        return clamp(va + vb - vc) << shift;
    }

    private static int clamp(int value) {
        return Math.clamp(value, 0, 255);
    }

}
