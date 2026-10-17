/**
 * The BMP reader and writer - internal.
 * <p>
 * {@link dev.simplified.image.codec.bmp.BmpImageReader BmpImageReader} and
 * {@link dev.simplified.image.codec.bmp.BmpImageWriter BmpImageWriter} read and write BMP through
 * ImageIO. A caller reaches them through {@link dev.simplified.image.ImageFactory ImageFactory} with
 * {@link dev.simplified.image.ImageFormat#BMP ImageFormat.BMP}, never directly: both are public only
 * because the factory constructs them from the root package, and each carries
 * {@code @ApiStatus.Internal}.
 * <p>
 * A type belongs here when only BMP uses it. A helper a second codec package needs moves to
 * {@link dev.simplified.image.codec.common codec.common}, and a BMP write takes no options of its own.
 */
package dev.simplified.image.codec.bmp;
