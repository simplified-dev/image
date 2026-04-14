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
    OVERLAY

}
