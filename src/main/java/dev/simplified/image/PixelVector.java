package dev.simplified.image;

import dev.simplified.annotations.UtilityClass;
import dev.simplified.image.pixel.ColorMath;
import org.jetbrains.annotations.NotNull;

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
@UtilityClass
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

    static void grayscale(int @NotNull [] p, int off, int len) {
        int start = ENABLED ? VectorOps.grayscale(p, off, len) : 0;
        int end = off + len;
        for (int i = off + start; i < end; i++) {
            int px = p[i];
            int a = (px >>> 24) & 0xFF;
            int y = Math.clamp(Math.round(ColorMath.luma(px | 0xFF000000)), 0, 255);
            p[i] = (a << 24) | (y << 16) | (y << 8) | y;
        }
    }

    static void multiplyAlpha(int @NotNull [] p, int off, int len, float factor) {
        int start = ENABLED ? VectorOps.multiplyAlpha(p, off, len, factor) : 0;
        int end = off + len;
        for (int i = off + start; i < end; i++) {
            int px = p[i];
            int a = (px >>> 24) & 0xFF;
            int newA = Math.round(a * factor);
            p[i] = (newA << 24) | (px & 0x00FFFFFF);
        }
    }

    /**
     * Scales all four channels of each premultiplied pixel in the run by {@code factor}, each rounded
     * half up once, so a colour at or under its alpha stays there. Scalar only; the straight form,
     * {@link #multiplyAlpha}, scales alpha alone.
     *
     * @param p the pixels
     * @param off the index of the first pixel of the run
     * @param len the number of pixels in the run
     * @param factor the multiplier, in {@code [0, 1]}
     */
    static void multiplyPremultiplied(int @NotNull [] p, int off, int len, float factor) {
        int end = off + len;
        for (int i = off; i < end; i++) {
            int px = p[i];
            int a = Math.round(((px >>> 24) & 0xFF) * factor);
            int r = Math.round(((px >>> 16) & 0xFF) * factor);
            int g = Math.round(((px >>> 8) & 0xFF) * factor);
            int b = Math.round((px & 0xFF) * factor);
            p[i] = (a << 24) | (r << 16) | (g << 8) | b;
        }
    }

    static void premultiply(int @NotNull [] p, int off, int len) {
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
            int r = div255(((px >>> 16) & 0xFF) * a);
            int g = div255(((px >>> 8) & 0xFF) * a);
            int b = div255((px & 0xFF) * a);
            p[i] = (a << 24) | (r << 16) | (g << 8) | b;
        }
    }

    static void tint(int @NotNull [] src, int @NotNull [] dst, int off, int len, int tr, int tg, int tb) {
        int start = ENABLED ? VectorOps.tint(src, dst, off, len, tr, tg, tb) : 0;
        int end = off + len;
        for (int i = off + start; i < end; i++) {
            int pixel = src[i];
            int a = (pixel >>> 24) & 0xFF;
            if (a == 0) {
                dst[i] = pixel;
                continue;
            }
            int r = div255(((pixel >>> 16) & 0xFF) * tr);
            int g = div255(((pixel >>> 8) & 0xFF) * tg);
            int b = div255((pixel & 0xFF) * tb);
            dst[i] = (a << 24) | (r << 16) | (g << 8) | b;
        }
    }

    static void lerp(int @NotNull [] a, int aOff, int @NotNull [] b, int bOff, int @NotNull [] out, int outOff,
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
        return Math.round(Math.clamp(value, 0f, 255f));
    }

    /**
     * Divides a non-negative product of two 8-bit channels by 255 with round-half-up semantics,
     * matching the OpenGL normalized fixed-point conversion {@code floor(v * 255 + 0.5)}. Mirrors
     * {@code ColorMath#div255} so scalar and SIMD compose paths stay bit-identical.
     *
     * @param product the channel product in {@code [0, 65025]}
     * @return the round-half-up quotient
     */
    private static int div255(int product) {
        return (product + 127) / 255;
    }
}
