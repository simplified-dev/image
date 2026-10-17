/**
 * Colour, compositing and drawing work on a {@link dev.simplified.image.PixelBuffer PixelBuffer}.
 * <p>
 * {@link dev.simplified.image.pixel.ColorMath ColorMath} holds the per-channel arithmetic.
 * {@link dev.simplified.image.pixel.BlendMode BlendMode} names how one pixel is composited over
 * another, {@link dev.simplified.image.pixel.AlphaMode AlphaMode} the alpha convention a buffer's
 * colour is held in, and {@link dev.simplified.image.pixel.Background Background} the fill an image is
 * laid over. {@link dev.simplified.image.pixel.PixelGraphics PixelGraphics} draws onto a buffer through
 * the {@code Graphics2D} contract, {@link dev.simplified.image.pixel.PixelMask PixelMask} marks the
 * pixels a later pass acts on, {@link dev.simplified.image.pixel.PixelDiff PixelDiff} draws the
 * difference of two buffers as a {@link dev.simplified.image.pixel.DiffType DiffType} asks,
 * {@link dev.simplified.image.pixel.Resample Resample} names a resize filter, and
 * {@link dev.simplified.image.pixel.PixelBufferPool PixelBufferPool} lends scratch buffers out on one
 * thread.
 * <p>
 * A type belongs here when it works on pixels or names a choice that pixel work takes. The buffer
 * itself is a form a whole image is held as, so it sits in {@link dev.simplified.image the root
 * package}.
 */
package dev.simplified.image.pixel;
