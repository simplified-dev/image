/**
 * The QOI reader and writer - internal.
 * <p>
 * {@link dev.simplified.image.codec.qoi.QoiImageReader QoiImageReader} and
 * {@link dev.simplified.image.codec.qoi.QoiImageWriter QoiImageWriter} read and write the Quite OK Image
 * format. A caller reaches them through {@link dev.simplified.image.ImageFactory ImageFactory} with
 * {@link dev.simplified.image.ImageFormat#QOI ImageFormat.QOI}, never directly: both are public only
 * because the factory constructs them from the root package, and each carries
 * {@code @ApiStatus.Internal}.
 * <p>
 * A type belongs here when only QOI uses it. A helper a second codec package needs moves to
 * {@link dev.simplified.image.codec.common codec.common}, and a QOI write takes no options of its own.
 */
package dev.simplified.image.codec.qoi;
