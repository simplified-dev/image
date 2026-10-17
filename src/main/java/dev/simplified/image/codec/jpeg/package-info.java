/**
 * The JPEG reader and writer - internal.
 * <p>
 * {@link dev.simplified.image.codec.jpeg.JpegImageReader JpegImageReader} and
 * {@link dev.simplified.image.codec.jpeg.JpegImageWriter JpegImageWriter} read and write JPEG through
 * ImageIO. A caller reaches them through {@link dev.simplified.image.ImageFactory ImageFactory} with
 * {@link dev.simplified.image.ImageFormat#JPEG ImageFormat.JPEG}, never directly: both are public only
 * because the factory constructs them from the root package, and each carries
 * {@code @ApiStatus.Internal}.
 * <p>
 * A type belongs here when only JPEG uses it. A helper a second codec package needs moves to
 * {@link dev.simplified.image.codec.common codec.common}, and the options a JPEG write takes are
 * {@link dev.simplified.image.option.JpegWriteOptions JpegWriteOptions}, in
 * {@link dev.simplified.image.option option}.
 */
package dev.simplified.image.codec.jpeg;
