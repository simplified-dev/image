package dev.simplified.image.codec.webp.lossy;

import org.jetbrains.annotations.NotNull;

/**
 * VP8 deblocking loop filter applied at macroblock and sub-block boundaries.
 * <p>
 * Reduces blocking artifacts by smoothing boundary pixels based on the
 * quantization level and a sharpness parameter.
 */
final class LoopFilter {

    private LoopFilter() { }

    /**
     * Applies the simple loop filter to reconstructed luma samples
     * at macroblock column boundaries.
     *
     * @param samples the full luma plane (row-major)
     * @param width the plane width
     * @param height the plane height
     * @param filterLevel the filter strength (0-63)
     * @param sharpness the sharpness parameter (0-7)
     */
    static void filterSimple(
        short @NotNull [] samples,
        int width,
        int height,
        int filterLevel,
        int sharpness
    ) {
        if (filterLevel == 0) return;

        int limit = computeLimit(filterLevel, sharpness);

        // Vertical edges (between macroblock columns)
        for (int mbX = 1; mbX < (width + 15) / 16; mbX++) {
            int x = mbX * 16;
            if (x >= width) break;

            for (int y = 0; y < height; y++) {
                int idx = y * width + x;
                filterEdge4(samples, idx - 2, idx - 1, idx, idx + 1, limit);
            }
        }

        // Horizontal edges (between macroblock rows)
        for (int mbY = 1; mbY < (height + 15) / 16; mbY++) {
            int y = mbY * 16;
            if (y >= height) break;

            for (int x = 0; x < width; x++) {
                int idx0 = (y - 2) * width + x;
                int idx1 = (y - 1) * width + x;
                int idx2 = y * width + x;
                int idx3 = (y + 1) * width + x;
                filterEdge4(samples, idx0, idx1, idx2, idx3, limit);
            }
        }
    }

    private static void filterEdge4(short[] samples, int p1, int p0, int q0, int q1, int limit) {
        if (p1 < 0 || p0 < 0 || q0 < 0 || q1 < 0
            || p1 >= samples.length || p0 >= samples.length
            || q0 >= samples.length || q1 >= samples.length) return;

        int a = 3 * (samples[q0] - samples[p0]);
        if (Math.abs(a) > limit * 4) return;

        a += clampFilter(samples[p1] - samples[q1]);
        a = clampFilter(a);

        int a1 = (a + 4) >> 3;
        int a2 = (a + 3) >> 3;

        samples[q0] = (short) clamp(samples[q0] - a1);
        samples[p0] = (short) clamp(samples[p0] + a2);
    }

    private static int computeLimit(int filterLevel, int sharpness) {
        int limit = filterLevel;
        if (sharpness > 0) {
            limit >>= (sharpness > 4 ? 2 : 1);
            limit = Math.max(1, limit);
        }
        return limit;
    }

    private static int clampFilter(int value) {
        return Math.clamp(value, -128, 127);
    }

    private static int clamp(int value) {
        return Math.clamp(value, 0, 255);
    }

}
