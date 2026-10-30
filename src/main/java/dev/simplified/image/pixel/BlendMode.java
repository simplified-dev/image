package dev.simplified.image.pixel;

import dev.simplified.annotations.Getter;
import dev.simplified.annotations.RequiredArgsConstructor;
import dev.simplified.image.PixelBuffer;

/**
 * Per-pixel alpha compositing modes for ARGB pixel blending.
 * <p>
 * Mode semantics operate on ARGB pixel pairs where {@code src} is the incoming pixel and
 * {@code dst} is the pixel already on the surface. Each constant names the formula
 * {@link ColorMath#blend(int, int, BlendMode)} applies to raw ints, and states which convention that
 * formula reads and answers its colour in, for a caller writing pixels itself.
 * <p>
 * A composite of one {@link PixelBuffer} onto another, and a colour filled into one, take the formula
 * from the conventions the buffers hold instead, as {@link PixelBuffer#blit(PixelBuffer, int, int, BlendMode)}
 * states. There {@link #NORMAL}, {@link #PREMULTIPLIED} and {@link #STRAIGHT} are one source-over mode,
 * laid by whichever of their three formulas fits the two conventions; {@link #REPLACE} writes the
 * source in the destination's convention; and {@link #ADD}, {@link #MULTIPLY}, {@link #OVERLAY} and
 * {@link #QUADRATIC_ADD} read the source as straight colour and the destination in the convention each
 * formula takes, converting it there and back.
 */
@Getter
@RequiredArgsConstructor
public enum BlendMode {

    /**
     * Standard source-over alpha compositing of a straight source over premultiplied colour, rounded
     * once as a GPU blends a straight texel into its framebuffer. The result equals
     * {@code src * src.a + dst * (1 - src.a)} per channel, and is premultiplied colour.
     * <p>
     * In a composite or a colour fill it is the source-over mode, laid by whichever source-over
     * formula the two conventions give.
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
     * <p>
     * In a composite the source is written in the destination's convention, converted where it holds
     * the other one, and a colour fill writes the colour in the buffer's convention.
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
     * <p>
     * In a composite it is the source-over mode, laid by whichever source-over formula the two
     * conventions give, so a straight source laid with it onto premultiplied colour takes
     * {@link #NORMAL}'s formula. A colour argument, straight by contract, refuses it.
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
     * <p>
     * In a composite it is the source-over mode, laid by whichever source-over formula the two
     * conventions give. A colour argument, straight by contract, refuses it.
     */
    STRAIGHT(true),

    /**
     * Additive blend, clamped to byte range. The result equals {@code min(255, src * src.a + dst)} per
     * colour channel and {@code min(255, src.a + dst.a)} for alpha: a straight source, weighed by its
     * alpha, added to premultiplied colour.
     * <p>
     * In a composite or a colour fill a premultiplied source is divided back to straight colour, and a
     * destination of straight colour is premultiplied for the sum and divided back after.
     */
    ADD(false),

    /**
     * Multiplicative blend. The result equals {@code src * dst / 255} per channel, preserving
     * the destination alpha. It scales each destination channel and keeps its alpha, which holds in
     * either convention, and reads the source as straight colour.
     * <p>
     * In a composite or a colour fill a premultiplied source is divided back to straight colour, and the
     * destination is read as it is held.
     */
    MULTIPLY(false),

    /**
     * Photoshop-style overlay: {@code dst < 128 ? 2*src*dst/255 : 255 - 2*(255-src)*(255-dst)/255}, with
     * the destination alpha preserved. Its threshold reads straight colour on both sides.
     * <p>
     * In a composite or a colour fill a premultiplied source is divided back to straight colour, and a
     * destination of premultiplied colour is divided back for the formula and premultiplied after.
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
     * {@code ONE} ({@code out = src * src_color + dst}), which adds to the premultiplied colour a
     * framebuffer drawn onto from transparency holds. Any per-source intensity scaling should be
     * pre-multiplied into the source RGB by the caller, since the square applies to the source as-is.
     * <p>
     * In a composite or a colour fill a premultiplied source is divided back to straight colour, and a
     * destination of straight colour is premultiplied for the sum and divided back after.
     */
    QUADRATIC_ADD(false);

    /**
     * Whether this mode is source-over: it answers a fully opaque source unchanged and leaves the
     * destination untouched under a fully transparent one, so a compositor may copy the one and skip
     * the other without blending.
     */
    private final boolean sourceOver;

}
