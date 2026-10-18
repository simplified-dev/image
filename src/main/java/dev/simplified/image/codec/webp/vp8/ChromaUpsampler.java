package dev.simplified.image.codec.webp.vp8;

import dev.simplified.annotations.UtilityClass;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

/**
 * Bilinear 4:2:0 chroma upsampling and YCbCr-to-RGB conversion for VP8 decode, matching
 * libwebp's fancy upsampler ({@code src/dsp/upsampling.c} + {@code EmitFancyRGB} in
 * {@code src/dec/io_dec.c}) and its {@code VP8YuvToRgb} ({@code src/dsp/yuv.h}) bit for bit.
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
 *
 * @see <a href="https://chromium.googlesource.com/webm/libwebp/+/refs/heads/main/src/dsp/upsampling.c">libwebp upsampling.c</a>
 */
@UtilityClass
final class ChromaUpsampler {

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
            width
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
                width
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
                width
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
     */
    private static void upsampleLinePair(
        short @NotNull [] topY, int topYOff,
        short @Nullable [] botY, int botYOff,
        short @NotNull [] topU, short @NotNull [] topV, int topUvOff,
        short @NotNull [] curU, short @NotNull [] curV, int curUvOff,
        int @NotNull [] out, int topDstOff, int botDstOff,
        int len
    ) {
        int lastPixelPair = (len - 1) >> 1;
        int tlU = topU[topUvOff] & 0xFF;
        int tlV = topV[topUvOff] & 0xFF;
        int lU = curU[curUvOff] & 0xFF;
        int lV = curV[curUvOff] & 0xFF;

        // First output pixel: collapses the 4-tap bilinear to a 2-tap vertical (same chroma
        // column on both sides, so the horizontal component weight-sums to 0).
        {
            int u0 = (3 * tlU + lU + 2) >> 2;
            int v0 = (3 * tlV + lV + 2) >> 2;
            out[topDstOff] = yuvToArgb(topY[topYOff] & 0xFF, u0, v0);
        }
        if (botY != null) {
            int u0 = (3 * lU + tlU + 2) >> 2;
            int v0 = (3 * lV + tlV + 2) >> 2;
            out[botDstOff] = yuvToArgb(botY[botYOff] & 0xFF, u0, v0);
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
            out[topDstOff + 2 * x] = yuvToArgb(topY[topYOff + 2 * x] & 0xFF, u1, v1);

            if (botY != null) {
                int bu0 = (diag03U + lU) >> 1;
                int bv0 = (diag03V + lV) >> 1;
                int bu1 = (diag12U + cU) >> 1;
                int bv1 = (diag12V + cV) >> 1;
                out[botDstOff + 2 * x - 1] = yuvToArgb(botY[botYOff + 2 * x - 1] & 0xFF, bu0, bv0);
                out[botDstOff + 2 * x] = yuvToArgb(botY[botYOff + 2 * x] & 0xFF, bu1, bv1);
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
     * Converts one BT.601 YCbCr pixel to opaque ARGB as libwebp's {@code VP8YuvToRgb} does
     * ({@code src/dsp/yuv.h}): each channel sums {@link #multHi} terms and an offset in 14-bit
     * fixed point, then {@link #clip8} drops the six fraction bits.
     */
    private static int yuvToArgb(int y, int u, int v) {
        int r = clip8(multHi(y, 19077) + multHi(v, 26149) - 14234);
        int g = clip8(multHi(y, 19077) - multHi(u, 6419) - multHi(v, 13320) + 8708);
        int b = clip8(multHi(y, 19077) + multHi(u, 33050) - 17685);
        return 0xFF000000 | (r << 16) | (g << 8) | b;
    }

    /**
     * Returns the high bits of a sample's product with a coefficient, libwebp's {@code MultHi}.
     */
    private static int multHi(int value, int coefficient) {
        return (value * coefficient) >> 8;
    }

    /**
     * Drops the six fraction bits of a 14-bit fixed-point channel, clamping it to {@code [0, 255]},
     * libwebp's {@code VP8Clip8}.
     */
    private static int clip8(int value) {
        return (value & ~0x3FFF) == 0 ? value >> 6 : value < 0 ? 0 : 255;
    }

}
