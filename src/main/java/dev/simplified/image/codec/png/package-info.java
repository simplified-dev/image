/**
 * The PNG reader and writer - internal.
 * <p>
 * {@link dev.simplified.image.codec.png.PngImageReader PngImageReader} and
 * {@link dev.simplified.image.codec.png.PngImageWriter PngImageWriter} read and write PNG through
 * ImageIO. A caller reaches them through {@link dev.simplified.image.ImageFactory ImageFactory} with
 * {@link dev.simplified.image.ImageFormat#PNG ImageFormat.PNG}, never directly: both are public only
 * because the factory constructs them from the root package, and each carries
 * {@code @ApiStatus.Internal}.
 * <p>
 * A type belongs here when only PNG uses it. A helper a second codec package needs moves to
 * {@link dev.simplified.image.codec.common codec.common}, and the options a PNG write takes are
 * {@link dev.simplified.image.option.PngWriteOptions PngWriteOptions}, in
 * {@link dev.simplified.image.option option}.
 */
package dev.simplified.image.codec.png;
