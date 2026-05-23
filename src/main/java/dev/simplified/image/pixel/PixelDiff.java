package dev.simplified.image.pixel;

import lombok.experimental.UtilityClass;
import org.jetbrains.annotations.NotNull;

/**
 * Per-pixel difference visualisations between two {@link PixelBuffer}s.
 * <p>
 * Each method returns a new {@code PixelBuffer} sized to the intersection of the inputs
 * ({@code min(a.width, b.width) × min(a.height, b.height)}). Out-of-canvas pixels (both inputs
 * fully transparent) are written as {@link ColorMath#TRANSPARENT} so the caller can composite the
 * result over its own background. Coverage mismatches (exactly one input has alpha at a pixel) are
 * surfaced as a magenta marker so they remain visible even when their composited colour delta
 * would otherwise read as small.
 * <p>
 * The diff direction is {@code a - b}: for {@link #signedLuma} / {@link #signedRgb} a value past
 * the mid-grey baseline means {@code a} exceeded {@code b} on that channel; for {@link #coverage}
 * magenta marks an {@code a}-only pixel and cyan marks a {@code b}-only pixel. Caller-defined
 * orientation - typically {@code a} is the reference image and {@code b} is the candidate.
 *
 * <p>The {@link #of(PixelBuffer, PixelBuffer, DiffType)} dispatcher delegates to the per-mode
 * methods so callers that pin a {@link DiffType} at runtime ({@code PixelBuffer.diff(other, type)})
 * compile through the same code path as direct callers.
 */
@UtilityClass
public class PixelDiff {

    /**
     * Magenta marker used for coverage mismatches in the colour modes ({@link #absolute},
     * {@link #signedLuma}, {@link #signedRgb}) and for {@code a}-only pixels in {@link #coverage}.
     */
    public static final int COVERAGE_MAGENTA = 0xFFFF00FF;

    /**
     * Cyan marker used for {@code b}-only pixels in {@link #coverage}.
     */
    public static final int COVERAGE_CYAN = 0xFF00FFFF;

    /**
     * Flat dark grey written by {@link #absolute} when both inputs agree exactly within the
     * silhouette - a "checked here, no diff" diagnostic that's still distinguishable from the
     * transparent canvas around the silhouette.
     */
    public static final int ABSOLUTE_MATCH = 0xFF101010;

    /**
     * Flat dark grey written by {@link #coverage} for both-opaque silhouette interior pixels.
     */
    public static final int COVERAGE_BOTH = 0xFF303034;

    /**
     * Computes a diff buffer of the requested type.
     *
     * @param a the left-side input buffer
     * @param b the right-side input buffer
     * @param type the diff visualisation mode
     * @return a new buffer sized {@code min(a.width, b.width) × min(a.height, b.height)}
     */
    public static @NotNull PixelBuffer of(@NotNull PixelBuffer a, @NotNull PixelBuffer b, @NotNull DiffType type) {
        return switch (type) {
            case ABSOLUTE -> absolute(a, b);
            case OVER_WHITE -> overWhite(a, b);
            case SIGNED_LUMA -> signedLuma(a, b);
            case SIGNED_RGB -> signedRgb(a, b);
            case COVERAGE -> coverage(a, b);
        };
    }

    /**
     * Per-channel absolute difference after each pixel is composited over white, amplified ×4
     * for visibility. See {@link DiffType#OVER_WHITE} for the full pixel-mode breakdown.
     *
     * @param a the left-side input buffer
     * @param b the right-side input buffer
     * @return a new buffer carrying the composite-over-white diff
     */
    public static @NotNull PixelBuffer overWhite(@NotNull PixelBuffer a, @NotNull PixelBuffer b) {
        int w = Math.min(a.width(), b.width());
        int h = Math.min(a.height(), b.height());
        PixelBuffer out = PixelBuffer.create(w, h);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int pa = a.getPixel(x, y);
                int pb = b.getPixel(x, y);
                int aa = ColorMath.alpha(pa);
                int ab = ColorMath.alpha(pb);
                int dr = Math.abs(compositeOverWhite(ColorMath.red(pa), aa) - compositeOverWhite(ColorMath.red(pb), ab));
                int dg = Math.abs(compositeOverWhite(ColorMath.green(pa), aa) - compositeOverWhite(ColorMath.green(pb), ab));
                int db = Math.abs(compositeOverWhite(ColorMath.blue(pa), aa) - compositeOverWhite(ColorMath.blue(pb), ab));
                if (dr == 0 && dg == 0 && db == 0) {
                    out.setPixel(x, y, ColorMath.TRANSPARENT);
                    continue;
                }
                if ((aa == 0) ^ (ab == 0)) {
                    out.setPixel(x, y, COVERAGE_MAGENTA);
                    continue;
                }
                int rr = Math.min(255, dr * 4);
                int gg = Math.min(255, dg * 4);
                int bb = Math.min(255, db * 4);
                out.setPixel(x, y, ColorMath.pack(255, rr, gg, bb));
            }
        }
        return out;
    }

    /**
     * Per-channel absolute difference amplified ×4 for visibility. See {@link DiffType#ABSOLUTE}
     * for the full pixel-mode breakdown.
     *
     * @param a the left-side input buffer
     * @param b the right-side input buffer
     * @return a new buffer carrying the absolute diff
     */
    public static @NotNull PixelBuffer absolute(@NotNull PixelBuffer a, @NotNull PixelBuffer b) {
        int w = Math.min(a.width(), b.width());
        int h = Math.min(a.height(), b.height());
        PixelBuffer out = PixelBuffer.create(w, h);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int pa = a.getPixel(x, y);
                int pb = b.getPixel(x, y);
                int aa = ColorMath.alpha(pa);
                int ab = ColorMath.alpha(pb);
                if (aa == 0 && ab == 0) {
                    out.setPixel(x, y, ColorMath.TRANSPARENT);
                    continue;
                }
                if ((aa == 0) ^ (ab == 0)) {
                    out.setPixel(x, y, COVERAGE_MAGENTA);
                    continue;
                }
                int dr = Math.abs(ColorMath.red(pa) - ColorMath.red(pb));
                int dg = Math.abs(ColorMath.green(pa) - ColorMath.green(pb));
                int db = Math.abs(ColorMath.blue(pa) - ColorMath.blue(pb));
                int da = Math.abs(aa - ab);
                if (da == 0 && dr == 0 && dg == 0 && db == 0) {
                    out.setPixel(x, y, ABSOLUTE_MATCH);
                    continue;
                }
                int rr = Math.min(255, dr * 4);
                int gg = Math.min(255, dg * 4);
                int bb = Math.min(255, db * 4);
                out.setPixel(x, y, ColorMath.pack(255, rr, gg, bb));
            }
        }
        return out;
    }

    /**
     * Signed luma delta on a red↔blue divergent palette. See {@link DiffType#SIGNED_LUMA} for
     * the full pixel-mode breakdown.
     *
     * @param a the left-side input buffer
     * @param b the right-side input buffer
     * @return a new buffer carrying the signed luma diff
     */
    public static @NotNull PixelBuffer signedLuma(@NotNull PixelBuffer a, @NotNull PixelBuffer b) {
        int w = Math.min(a.width(), b.width());
        int h = Math.min(a.height(), b.height());
        PixelBuffer out = PixelBuffer.create(w, h);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int pa = a.getPixel(x, y);
                int pb = b.getPixel(x, y);
                int aa = ColorMath.alpha(pa);
                int ab = ColorMath.alpha(pb);
                if (aa == 0 && ab == 0) {
                    out.setPixel(x, y, ColorMath.TRANSPARENT);
                    continue;
                }
                if ((aa == 0) ^ (ab == 0)) {
                    out.setPixel(x, y, COVERAGE_MAGENTA);
                    continue;
                }
                float vL = unweightedLuma(pa);
                float jL = unweightedLuma(pb);
                float delta = vL - jL;
                int mag = (int) Math.min(127, Math.abs(delta) * 2);
                int r, g, bl;
                if (delta >= 0) {
                    r = 128 + mag;
                    g = 128 - mag / 2;
                    bl = 128 - mag / 2;
                } else {
                    r = 128 - mag / 2;
                    g = 128 - mag / 2;
                    bl = 128 + mag;
                }
                out.setPixel(x, y, ColorMath.pack(255, r, g, bl));
            }
        }
        return out;
    }

    /**
     * Per-channel signed delta centred at mid-grey. See {@link DiffType#SIGNED_RGB} for the full
     * pixel-mode breakdown.
     *
     * @param a the left-side input buffer
     * @param b the right-side input buffer
     * @return a new buffer carrying the per-channel signed diff
     */
    public static @NotNull PixelBuffer signedRgb(@NotNull PixelBuffer a, @NotNull PixelBuffer b) {
        int w = Math.min(a.width(), b.width());
        int h = Math.min(a.height(), b.height());
        PixelBuffer out = PixelBuffer.create(w, h);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int pa = a.getPixel(x, y);
                int pb = b.getPixel(x, y);
                int aa = ColorMath.alpha(pa);
                int ab = ColorMath.alpha(pb);
                if (aa == 0 && ab == 0) {
                    out.setPixel(x, y, ColorMath.TRANSPARENT);
                    continue;
                }
                if ((aa == 0) ^ (ab == 0)) {
                    out.setPixel(x, y, COVERAGE_MAGENTA);
                    continue;
                }
                int dR = (ColorMath.red(pa) - ColorMath.red(pb)) * 2;
                int dG = (ColorMath.green(pa) - ColorMath.green(pb)) * 2;
                int dB = (ColorMath.blue(pa) - ColorMath.blue(pb)) * 2;
                int r = Math.clamp(128 + dR, 0, 255);
                int g = Math.clamp(128 + dG, 0, 255);
                int bl = Math.clamp(128 + dB, 0, 255);
                out.setPixel(x, y, ColorMath.pack(255, r, g, bl));
            }
        }
        return out;
    }

    /**
     * Coverage-only silhouette difference. See {@link DiffType#COVERAGE} for the full pixel-mode
     * breakdown.
     *
     * @param a the left-side input buffer
     * @param b the right-side input buffer
     * @return a new buffer carrying the coverage diff
     */
    public static @NotNull PixelBuffer coverage(@NotNull PixelBuffer a, @NotNull PixelBuffer b) {
        int w = Math.min(a.width(), b.width());
        int h = Math.min(a.height(), b.height());
        PixelBuffer out = PixelBuffer.create(w, h);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int aa = ColorMath.alpha(a.getPixel(x, y));
                int ab = ColorMath.alpha(b.getPixel(x, y));
                if (aa == 0 && ab == 0) {
                    out.setPixel(x, y, ColorMath.TRANSPARENT);
                } else if (aa > 0 && ab > 0) {
                    out.setPixel(x, y, COVERAGE_BOTH);
                } else if (aa > 0) {
                    out.setPixel(x, y, COVERAGE_MAGENTA);
                } else {
                    out.setPixel(x, y, COVERAGE_CYAN);
                }
            }
        }
        return out;
    }

    /**
     * Unweighted Rec. 601 luminance. Used by {@link #signedLuma} - the alpha gate has already
     * fired by the time this is called (only invoked when both pixels have alpha &gt; 0), so the
     * alpha-weighted variant on {@link ColorMath} would just multiply by 1 here.
     */
    private static float unweightedLuma(int argb) {
        return 0.299f * ColorMath.red(argb)
            + 0.587f * ColorMath.green(argb)
            + 0.114f * ColorMath.blue(argb);
    }

    /**
     * Composites a single channel value over a fully-opaque white background. The result is
     * {@code channel * alpha/255 + 255 * (1 - alpha/255)} rounded to integer. Used by
     * {@link #overWhite} to perceptually-correct AA edge spill.
     */
    private static int compositeOverWhite(int channel, int alpha) {
        return (channel * alpha + 255 * (255 - alpha) + 127) / 255;
    }

}
