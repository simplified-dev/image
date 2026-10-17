/**
 * The GIF reader and writer - internal.
 * <p>
 * {@link dev.simplified.image.codec.gif.GifImageReader GifImageReader} and
 * {@link dev.simplified.image.codec.gif.GifImageWriter GifImageWriter} read and write still and animated
 * GIF through ImageIO, with each frame's delay, offset and disposal. A caller reaches them through
 * {@link dev.simplified.image.ImageFactory ImageFactory} with
 * {@link dev.simplified.image.ImageFormat#GIF ImageFormat.GIF}, never directly: both are public only
 * because the factory constructs them from the root package, and each carries
 * {@code @ApiStatus.Internal}.
 * <p>
 * A type belongs here when only GIF uses it. A helper a second codec package needs moves to
 * {@link dev.simplified.image.codec.common codec.common}, and the options a GIF write takes are
 * {@link dev.simplified.image.option.GifWriteOptions GifWriteOptions}, in
 * {@link dev.simplified.image.option option}.
 */
package dev.simplified.image.codec.gif;
