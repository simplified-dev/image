package dev.simplified.image.pixel;

/**
 * Per-pixel alpha compositing modes for ARGB pixel blending.
 * <p>
 * Mode semantics operate on ARGB pixel pairs where {@code src} is the incoming pixel and
 * {@code dst} is the pixel already on the surface. Alpha is always premultiplied into the output.
 */
public enum BlendMode {

    /**
     * Standard source-over alpha compositing. The result equals
     * {@code src * src.a + dst * (1 - src.a)} per channel.
     */
    NORMAL,

    /**
     * Additive blend, clamped to byte range. The result equals
     * {@code min(255, src + dst)} per channel.
     */
    ADD,

    /**
     * Multiplicative blend. The result equals {@code src * dst / 255} per channel, preserving
     * the destination alpha.
     */
    MULTIPLY,

    /**
     * Photoshop-style overlay: {@code dst < 128 ? 2*src*dst/255 : 255 - 2*(255-src)*(255-dst)/255}.
     */
    OVERLAY,

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
    QUADRATIC_ADD

}
