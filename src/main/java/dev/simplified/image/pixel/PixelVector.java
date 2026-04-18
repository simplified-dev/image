package dev.simplified.image.pixel;

/**
 * Dispatches per-pixel operations to a SIMD implementation when the JDK Vector API is resolvable
 * at runtime, otherwise runs a scalar fallback. A single source covers both paths so bit-for-bit
 * semantics are preserved.
 * <p>
 * Availability is detected once via {@link Class#forName(String)} against {@code IntVector}. The
 * {@link VectorOps} class - which holds every reference to {@code jdk.incubator.vector} types -
 * is only loaded when the detection succeeds, so consumers whose JVM does not resolve the
 * incubator module never trigger an unresolved symbol.
 */
final class PixelVector {

    private static final boolean ENABLED = detectVectorApi();

    private static boolean detectVectorApi() {
        try {
            Class.forName("jdk.incubator.vector.IntVector");
            return true;
        } catch (Throwable ignored) {
            return false;
        }
    }

    private PixelVector() {}

    static void grayscale(int[] p, int off, int len) {
        int start = ENABLED ? VectorOps.grayscale(p, off, len) : 0;
        int end = off + len;
        for (int i = off + start; i < end; i++) {
            int px = p[i];
            int a = (px >>> 24) & 0xFF;
            int y = Math.clamp((int) ColorMath.luma(px | 0xFF000000), 0, 255);
            p[i] = (a << 24) | (y << 16) | (y << 8) | y;
        }
    }

    static void multiplyAlpha(int[] p, int off, int len, float factor) {
        int start = ENABLED ? VectorOps.multiplyAlpha(p, off, len, factor) : 0;
        int end = off + len;
        for (int i = off + start; i < end; i++) {
            int px = p[i];
            int a = (px >>> 24) & 0xFF;
            int newA = Math.round(a * factor);
            p[i] = (newA << 24) | (px & 0x00FFFFFF);
        }
    }

    static void premultiply(int[] p, int off, int len) {
        int start = ENABLED ? VectorOps.premultiply(p, off, len) : 0;
        int end = off + len;
        for (int i = off + start; i < end; i++) {
            int px = p[i];
            int a = (px >>> 24) & 0xFF;
            if (a == 0xFF) continue;
            if (a == 0) {
                p[i] = 0;
                continue;
            }
            int r = (((px >>> 16) & 0xFF) * a) / 255;
            int g = (((px >>> 8) & 0xFF) * a) / 255;
            int b = ((px & 0xFF) * a) / 255;
            p[i] = (a << 24) | (r << 16) | (g << 8) | b;
        }
    }

    static void tint(int[] src, int[] dst, int off, int len, int tr, int tg, int tb) {
        int start = ENABLED ? VectorOps.tint(src, dst, off, len, tr, tg, tb) : 0;
        int end = off + len;
        for (int i = off + start; i < end; i++) {
            int pixel = src[i];
            int a = (pixel >>> 24) & 0xFF;
            if (a == 0) {
                dst[i] = pixel;
                continue;
            }
            int r = (((pixel >>> 16) & 0xFF) * tr) / 255;
            int g = (((pixel >>> 8) & 0xFF) * tg) / 255;
            int b = ((pixel & 0xFF) * tb) / 255;
            dst[i] = (a << 24) | (r << 16) | (g << 8) | b;
        }
    }

    static void lerp(int[] a, int aOff, int[] b, int bOff, int[] out, int outOff,
                     int len, float blend, float inverse) {
        int start = ENABLED ? VectorOps.lerp(a, aOff, b, bOff, out, outOff, len, blend, inverse) : 0;
        for (int x = start; x < len; x++) {
            int s = a[aOff + x];
            int d = b[bOff + x];
            int ra = clampByte(((s >> 24) & 0xFF) * inverse + ((d >> 24) & 0xFF) * blend);
            int rr = clampByte(((s >> 16) & 0xFF) * inverse + ((d >> 16) & 0xFF) * blend);
            int rg = clampByte(((s >> 8) & 0xFF) * inverse + ((d >> 8) & 0xFF) * blend);
            int rb = clampByte((s & 0xFF) * inverse + (d & 0xFF) * blend);
            out[outOff + x] = (ra << 24) | (rr << 16) | (rg << 8) | rb;
        }
    }

    private static int clampByte(float value) {
        return (int) Math.clamp(value, 0f, 255f);
    }
}
