package dev.simplified.image.pixel;

import dev.simplified.annotations.Getter;
import dev.simplified.annotations.RequiredArgsConstructor;

/**
 * Per-pixel alpha compositing modes for ARGB pixel blending.
 * <p>
 * Mode semantics operate on ARGB pixel pairs where {@code src} is the incoming pixel and
 * {@code dst} is the pixel already on the surface. Each mode states which convention its colour is
 * read and answered in.
 */
@Getter
@RequiredArgsConstructor
public enum BlendMode {

    /**
     * Standard source-over alpha compositing. The result equals
     * {@code src * src.a + dst * (1 - src.a)} per channel.
     */
    NORMAL(true),

    /**
     * Porter-Duff <i>source</i>: the source replaces the destination verbatim, alpha included. The
     * result equals {@code src}, and {@code dst} is not read.
     * <p>
     * This is the absence of a blend rather than another kind of one, which is what a GPU pipeline
     * declaring no blend function does with each fragment that survives its depth and alpha tests.
     * <p>
     * It agrees with {@link #NORMAL} on every source whose alpha is {@code 0} or {@code 255} - at full
     * alpha source-over returns the source unchanged, and a fully transparent source is normally
     * discarded before it reaches a composite - so the two can differ only where the source carries
     * partial alpha.
     */
    REPLACE(false),

    /**
     * Source-over for a source whose colour already carries its alpha. The result equals
     * {@code src + dst * (1 - src.a)} per colour channel, clamped to byte range, and
     * {@code src.a + dst.a * (1 - src.a)} for alpha.
     * <p>
     * This is how one finished image is laid over another. An image drawn with {@link #NORMAL}
     * onto a transparent surface holds every colour already scaled by its alpha, so laying it down
     * with {@link #NORMAL} scales it a second time and darkens each partially transparent pixel.
     * This mode lays such an image over a transparent pixel unchanged, and over an opaque pixel it
     * gives what {@link #NORMAL} gives drawing the image's own source there directly, to within one
     * unit of rounding.
     * <p>
     * It agrees with {@link #NORMAL} on every source whose alpha is {@code 0} or {@code 255}, so the
     * two can differ only where the source carries partial alpha.
     */
    PREMULTIPLIED(true),

    /**
     * Source-over with both pixels held in straight colour, answered in straight colour. The result
     * alpha equals {@code src.a + dst.a * (1 - src.a)}, and each colour channel equals
     * {@code (src * src.a + dst * dst.a * (1 - src.a)) / alpha}, rounded to nearest.
     * <p>
     * This is how one straight image is laid over another - a decoded image over a fill, or a frame
     * over the canvas an animation has drawn so far. A fully transparent destination takes the source
     * unchanged, so a translucent pixel keeps its own colour rather than darkening.
     * <p>
     * It agrees with {@link #NORMAL} on every source whose alpha is {@code 0} or {@code 255}, and over
     * an opaque destination the two differ only by rounding.
     */
    STRAIGHT(true),

    /**
     * Additive blend, clamped to byte range. The result equals
     * {@code min(255, src + dst)} per channel.
     */
    ADD(false),

    /**
     * Multiplicative blend. The result equals {@code src * dst / 255} per channel, preserving
     * the destination alpha.
     */
    MULTIPLY(false),

    /**
     * Photoshop-style overlay: {@code dst < 128 ? 2*src*dst/255 : 255 - 2*(255-src)*(255-dst)/255}.
     */
    OVERLAY(false),

    /**
     * Quadratic additive blend: the source is squared (in normalized space) before adding the
     * destination. The result equals {@code min(255, src * src / 255 + dst)} per channel, with the
     * destination alpha preserved ({@code out.a = dst.a}); a fully transparent source is a no-op.
     * Squaring biases dark sources toward black and only lets bright sources contribute strongly, a
     * softer, more selective add than {@link #ADD}.
     * <p>
     * Equivalent to the GPU blend with source factor {@code SRC_COLOR} and destination factor
     * {@code ONE} ({@code out = src * src_color + dst}). Any per-source intensity scaling should be
     * pre-multiplied into the source RGB by the caller, since the square applies to the source as-is.
     */
    QUADRATIC_ADD(false);

    /**
     * Whether this mode is source-over: it answers a fully opaque source unchanged and leaves the
     * destination untouched under a fully transparent one, so a compositor may copy the one and skip
     * the other without blending.
     */
    private final boolean sourceOver;

}
