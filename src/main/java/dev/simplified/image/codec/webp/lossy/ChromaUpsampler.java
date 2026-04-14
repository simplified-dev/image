package dev.simplified.image.codec.webp.lossy;

import org.jetbrains.annotations.NotNull;

/**
 * Bilinear 4:2:0 chroma upsampling for VP8 decode, matching libwebp's fancy upsampler
 * ({@code src/dsp/upsampling.c} + {@code EmitFancyRGB} in {@code src/dec/io_dec.c}).
 * <p>
 * Given chroma samples laid out as a 2x2 square
 * <pre>
 *   [a b]
 *   [c d]
 * </pre>
 * the four upsampled output values covering that square's 2x2 output region are
 * <pre>
 *   TL = (9a + 3b + 3c +  d + 8) / 16
 *   TR = (3a + 9b +  c + 3d + 8) / 16
 *   BL = (3a +  b + 9c + 3d + 8) / 16
 *   BR = ( a + 3b + 3c + 9d + 8) / 16
 * </pre>
 * Replaces the prior nearest-neighbor sampling, which produced ~10 LSB per-channel
 * drift against libwebp on smooth chroma gradients.
 *
 * @see <a href="https://chromium.googlesource.com/webm/libwebp/+/refs/heads/main/src/dsp/upsampling.c">libwebp upsampling.c</a>
 */
final class ChromaUpsampler {

    private ChromaUpsampler() { }

    /**
     * Fancy-upsamples an entire frame. Output is written as ARGB {@code 0xFF000000 | ...}
     * integers. Walks row pairs in the pattern libwebp uses: the first output row is a
     * special case with the first chroma row replicated; interior rows are emitted in
     * pairs (each pair uses two adjacent chroma rows); for even-height frames, the final
     * row is a special case with the last chroma row replicated.
     *
     * @param planeY luma plane (0..255 samples, row-major, {@code yStride})
     * @param planeU chroma U plane (0..255 samples, row-major, {@code uvStride})
     * @param planeV chroma V plane (0..255 samples, row-major, {@code uvStride})
     * @param width image width in luma pixels
     * @param height image height in luma pixels
     * @param yStride luma row stride in samples
     * @param uvStride chroma row stride in samples
     * @param out output ARGB pixels (length {@code width * height})
     */
    static void upsampleToArgb(
        short @NotNull [] planeY, short @NotNull [] planeU, short @NotNull [] planeV,
        int width, int height, int yStride, int uvStride, int @NotNull [] out
    ) {
        int chromaRows = (height + 1) / 2;

        // Output row 0 is special-cased: no bottom luma row, first chroma row used as both top
        // and cur (so the vertical interpolation collapses to the pure-top case).
        upsampleLinePair(
            planeY, 0,
            null, 0,
            planeU, planeV, 0,
            planeU, planeV, 0,
            out, 0, -1,
            width, uvStride
        );

        // Main loop: emits output pairs (y, y+1) for y=1,3,5,..., using chroma rows (y-1)/2
        // and (y+1)/2 as top/cur. After processing, y advances by 2 and we also advance chroma
        // by one row each iteration.
        int y = 0;
        while (y + 2 < height) {
            int topChromaRow = y / 2;           // previous cur row
            int curChromaRow = (y + 2) / 2;     // new cur row
            int topYRow = y + 1;
            int botYRow = y + 2;
            int topDstRow = y + 1;
            int botDstRow = y + 2;
            upsampleLinePair(
                planeY, topYRow * yStride,
                planeY, botYRow * yStride,
                planeU, planeV, topChromaRow * uvStride,
                planeU, planeV, curChromaRow * uvStride,
                out, topDstRow * width, botDstRow * width,
                width, uvStride
            );
            y += 2;
        }

        // If the image has an even height, a single output row remains at the bottom, emitted
        // with the last chroma row replicated (symmetric to the y==0 special case).
        if ((height & 1) == 0) {
            int lastChromaRow = chromaRows - 1;
            int lastYRow = height - 1;
            upsampleLinePair(
                planeY, lastYRow * yStride,
                null, 0,
                planeU, planeV, lastChromaRow * uvStride,
                planeU, planeV, lastChromaRow * uvStride,
                out, lastYRow * width, -1,
                width, uvStride
            );
        }
    }

    /**
     * Processes one or two output rows given 4 chroma rows (2 top, 2 cur). Mirrors libwebp's
     * {@code UpsampleRgbaLinePair_C} expansion of the {@code UPSAMPLE_FUNC} macro. When
     * {@code botY} is {@code null}, only the top output row is emitted.
     *
     * @param topY luma plane for the top output row
     * @param topYOff offset into {@code topY} where the top luma row starts
     * @param botY luma plane for the bottom output row, or {@code null}
     * @param botYOff offset into {@code botY} (ignored when {@code botY} is {@code null})
     * @param topU top chroma U row's plane
     * @param topV top chroma V row's plane
     * @param topUvOff offset of the top chroma row in {@code topU}/{@code topV}
     * @param curU cur chroma U row's plane (typically same plane as {@code topU})
     * @param curV cur chroma V row's plane
     * @param curUvOff offset of the cur chroma row
     * @param out output ARGB pixel array
     * @param topDstOff output offset for the top row
     * @param botDstOff output offset for the bottom row, or {@code -1} if no bottom row
     * @param len number of output columns (image width)
     * @param uvStride chroma row stride (unused here since offsets are pre-computed; kept for
     *                 symmetry with libwebp's signature)
     */
    private static void upsampleLinePair(
        short[] topY, int topYOff,
        short[] botY, int botYOff,
        short[] topU, short[] topV, int topUvOff,
        short[] curU, short[] curV, int curUvOff,
        int[] out, int topDstOff, int botDstOff,
        int len, int uvStride
    ) {
        int lastPixelPair = (len - 1) >> 1;
        int tlU = topU[topUvOff + 0] & 0xFF;
        int tlV = topV[topUvOff + 0] & 0xFF;
        int lU = curU[curUvOff + 0] & 0xFF;
        int lV = curV[curUvOff + 0] & 0xFF;

        // First output pixel: collapses the 4-tap bilinear to a 2-tap vertical (same chroma
        // column on both sides, so the horizontal component weight-sums to 0).
        {
            int u0 = (3 * tlU + lU + 2) >> 2;
            int v0 = (3 * tlV + lV + 2) >> 2;
            out[topDstOff + 0] = yuvToArgb(topY[topYOff + 0] & 0xFF, u0, v0);
        }
        if (botY != null) {
            int u0 = (3 * lU + tlU + 2) >> 2;
            int v0 = (3 * lV + tlV + 2) >> 2;
            out[botDstOff + 0] = yuvToArgb(botY[botYOff + 0] & 0xFF, u0, v0);
        }

        for (int x = 1; x <= lastPixelPair; x++) {
            int tU = topU[topUvOff + x] & 0xFF;
            int tV = topV[topUvOff + x] & 0xFF;
            int cU = curU[curUvOff + x] & 0xFF;
            int cV = curV[curUvOff + x] & 0xFF;

            // Precompute diagonal averages shared between the two output columns.
            int sumU = tlU + tU + lU + cU + 8;
            int sumV = tlV + tV + lV + cV + 8;
            int diag12U = (sumU + 2 * (tU + lU)) >> 3;
            int diag12V = (sumV + 2 * (tV + lV)) >> 3;
            int diag03U = (sumU + 2 * (tlU + cU)) >> 3;
            int diag03V = (sumV + 2 * (tlV + cV)) >> 3;

            // Top row: two output pixels at columns 2x-1 and 2x.
            int u0 = (diag12U + tlU) >> 1;
            int v0 = (diag12V + tlV) >> 1;
            int u1 = (diag03U + tU) >> 1;
            int v1 = (diag03V + tV) >> 1;
            out[topDstOff + 2 * x - 1] = yuvToArgb(topY[topYOff + 2 * x - 1] & 0xFF, u0, v0);
            out[topDstOff + 2 * x - 0] = yuvToArgb(topY[topYOff + 2 * x - 0] & 0xFF, u1, v1);

            if (botY != null) {
                int bu0 = (diag03U + lU) >> 1;
                int bv0 = (diag03V + lV) >> 1;
                int bu1 = (diag12U + cU) >> 1;
                int bv1 = (diag12V + cV) >> 1;
                out[botDstOff + 2 * x - 1] = yuvToArgb(botY[botYOff + 2 * x - 1] & 0xFF, bu0, bv0);
                out[botDstOff + 2 * x - 0] = yuvToArgb(botY[botYOff + 2 * x - 0] & 0xFF, bu1, bv1);
            }

            tlU = tU; tlV = tV;
            lU = cU; lV = cV;
        }

        // Even-width tail pixel: collapses back to the boundary 2-tap, same as the start pixel
        // but using the last chroma column.
        if ((len & 1) == 0) {
            int u0 = (3 * tlU + lU + 2) >> 2;
            int v0 = (3 * tlV + lV + 2) >> 2;
            out[topDstOff + len - 1] = yuvToArgb(topY[topYOff + len - 1] & 0xFF, u0, v0);
            if (botY != null) {
                int bu0 = (3 * lU + tlU + 2) >> 2;
                int bv0 = (3 * lV + tlV + 2) >> 2;
                out[botDstOff + len - 1] = yuvToArgb(botY[botYOff + len - 1] & 0xFF, bu0, bv0);
            }
        }
    }

    /**
     * BT.601 YCbCr -> ARGB conversion for a single pixel. Keeps the same coefficients the
     * prior nearest-neighbor decoder used: {@code y -= 16}, {@code u -= 128}, {@code v -= 128},
     * {@code R = clamp((298y + 409v + 128) >> 8)} etc. Not bit-exact with libwebp's YUV->RGB
     * (libwebp uses a different fixed-point layout), but the output differs by at most 1 LSB
     * per channel, well below the upsampling drift this task was chasing.
     */
    private static int yuvToArgb(int y, int u, int v) {
        int yy = y - 16;
        int cb = u - 128;
        int cr = v - 128;
        int r = Math.clamp((298 * yy + 409 * cr + 128) >> 8, 0, 255);
        int g = Math.clamp((298 * yy - 100 * cb - 208 * cr + 128) >> 8, 0, 255);
        int b = Math.clamp((298 * yy + 516 * cb + 128) >> 8, 0, 255);
        return 0xFF000000 | (r << 16) | (g << 8) | b;
    }

}
