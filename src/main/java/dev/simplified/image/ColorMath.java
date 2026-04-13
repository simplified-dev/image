package dev.simplified.image;

import lombok.experimental.UtilityClass;
import org.jetbrains.annotations.NotNull;

/**
 * Static helpers for ARGB colour math, blending, and tinting.
 * <p>
 * All methods operate on packed 32-bit ARGB ints in the native byte order
 * {@code 0xAARRGGBB}. No object allocation - pure bit math.
 */
@UtilityClass
public class ColorMath {

    /** A fully transparent ARGB pixel. */
    public static final int TRANSPARENT = 0x00000000;

    /** An opaque white ARGB pixel. */
    public static final int WHITE = 0xFFFFFFFF;

    /** An opaque black ARGB pixel. */
    public static final int BLACK = 0xFF000000;

    // --- channel accessors ---

    public static int alpha(int argb) {
        return (argb >>> 24) & 0xFF;
    }

    public static int red(int argb) {
        return (argb >>> 16) & 0xFF;
    }

    public static int green(int argb) {
        return (argb >>> 8) & 0xFF;
    }

    public static int blue(int argb) {
        return argb & 0xFF;
    }

    /**
     * Packs individual ARGB channel values into a single 32-bit int.
     *
     * @param a the alpha channel in {@code [0, 255]}
     * @param r the red channel in {@code [0, 255]}
     * @param g the green channel in {@code [0, 255]}
     * @param b the blue channel in {@code [0, 255]}
     * @return the packed ARGB pixel
     */
    public static int pack(int a, int r, int g, int b) {
        return ((a & 0xFF) << 24) | ((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF);
    }

    /**
     * Combines a 24-bit RGB integer with an explicit alpha channel into a packed ARGB pixel.
     *
     * @param rgb the 24-bit RGB value
     * @param alpha the alpha channel in {@code [0, 255]}
     * @return the packed ARGB pixel
     */
    public static int withAlpha(int rgb, int alpha) {
        return ((alpha & 0xFF) << 24) | (rgb & 0x00FFFFFF);
    }

    // --- HSV conversion ---

    /**
     * Converts an HSV triplet into a packed ARGB pixel with full opacity.
     *
     * @param hue the hue in {@code [0, 360]} degrees
     * @param saturation the saturation in {@code [0, 1]}
     * @param value the value in {@code [0, 1]}
     * @return the packed ARGB pixel
     */
    public static int hsvToArgb(float hue, float saturation, float value) {
        return hsvToArgb(hue, saturation, value, 255);
    }

    /**
     * Converts an HSV triplet with an explicit alpha into a packed ARGB pixel.
     *
     * @param hue the hue in {@code [0, 360]} degrees
     * @param saturation the saturation in {@code [0, 1]}
     * @param value the value in {@code [0, 1]}
     * @param alpha the alpha channel in {@code [0, 255]}
     * @return the packed ARGB pixel
     */
    public static int hsvToArgb(float hue, float saturation, float value, int alpha) {
        float h = (((hue % 360f) + 360f) % 360f) / 60f;
        float c = value * saturation;
        float x = c * (1 - Math.abs((h % 2) - 1));
        float m = value - c;

        float r = 0, g = 0, b = 0;
        int sector = (int) Math.floor(h);
        switch (sector) {
            case 0 -> { r = c; g = x; }
            case 1 -> { r = x; g = c; }
            case 2 -> { g = c; b = x; }
            case 3 -> { g = x; b = c; }
            case 4 -> { r = x; b = c; }
            case 5, 6 -> { r = c; b = x; }
        }

        int ri = Math.round((r + m) * 255f);
        int gi = Math.round((g + m) * 255f);
        int bi = Math.round((b + m) * 255f);
        return pack(alpha, ri, gi, bi);
    }

    // --- blending ---

    /**
     * Blends {@code src} on top of {@code dst} using the given blend mode.
     *
     * @param src the source (incoming) pixel
     * @param dst the destination (existing) pixel
     * @param mode the blend mode
     * @return the composited ARGB pixel
     */
    public static int blend(int src, int dst, @NotNull BlendMode mode) {
        return switch (mode) {
            case NORMAL -> blendNormal(src, dst);
            case ADD -> blendAdd(src, dst);
            case MULTIPLY -> blendMultiply(src, dst);
            case OVERLAY -> blendOverlay(src, dst);
        };
    }

    private static int blendNormal(int src, int dst) {
        int sa = alpha(src);
        if (sa == 0xFF) return src;
        if (sa == 0) return dst;

        int da = alpha(dst);
        float srcA = sa / 255f;
        float invSrcA = 1f - srcA;

        int r = Math.round(red(src) * srcA + red(dst) * invSrcA);
        int g = Math.round(green(src) * srcA + green(dst) * invSrcA);
        int b = Math.round(blue(src) * srcA + blue(dst) * invSrcA);
        int a = Math.round(sa + da * invSrcA);
        return pack(a, r, g, b);
    }

    private static int blendAdd(int src, int dst) {
        int sa = alpha(src);
        if (sa == 0) return dst;

        float srcA = sa / 255f;
        int r = Math.min(255, red(dst) + Math.round(red(src) * srcA));
        int g = Math.min(255, green(dst) + Math.round(green(src) * srcA));
        int b = Math.min(255, blue(dst) + Math.round(blue(src) * srcA));
        int a = Math.min(255, alpha(dst) + sa);
        return pack(a, r, g, b);
    }

    private static int blendMultiply(int src, int dst) {
        int sa = alpha(src);
        if (sa == 0) return dst;

        int r = (red(src) * red(dst)) / 255;
        int g = (green(src) * green(dst)) / 255;
        int b = (blue(src) * blue(dst)) / 255;
        return pack(alpha(dst), r, g, b);
    }

    private static int blendOverlay(int src, int dst) {
        int sa = alpha(src);
        if (sa == 0) return dst;

        int r = overlayChannel(red(src), red(dst));
        int g = overlayChannel(green(src), green(dst));
        int b = overlayChannel(blue(src), blue(dst));
        return pack(alpha(dst), r, g, b);
    }

    private static int overlayChannel(int s, int d) {
        return d < 128
            ? (2 * s * d) / 255
            : 255 - (2 * (255 - s) * (255 - d)) / 255;
    }

    // --- tinting ---

    /**
     * Multiplies every pixel's RGB channels by the given tint, preserving alpha.
     *
     * @param source the source pixel buffer
     * @param argbTint the packed ARGB tint colour
     * @return a new tinted pixel buffer with the same dimensions
     */
    public static @NotNull PixelBuffer tint(@NotNull PixelBuffer source, int argbTint) {
        int w = source.width();
        int h = source.height();
        int[] result = new int[w * h];
        int[] src = source.pixels();

        int tr = red(argbTint);
        int tg = green(argbTint);
        int tb = blue(argbTint);

        for (int i = 0; i < src.length; i++) {
            int pixel = src[i];
            int a = alpha(pixel);
            if (a == 0) {
                result[i] = pixel;
                continue;
            }
            int r = (red(pixel) * tr) / 255;
            int g = (green(pixel) * tg) / 255;
            int b = (blue(pixel) * tb) / 255;
            result[i] = pack(a, r, g, b);
        }

        return PixelBuffer.of(result, w, h);
    }

}
