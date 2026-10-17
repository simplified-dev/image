/**
 * The TIFF reader and writer - internal.
 * <p>
 * {@link dev.simplified.image.codec.tiff.TiffImageReader TiffImageReader} and
 * {@link dev.simplified.image.codec.tiff.TiffImageWriter TiffImageWriter} read and write single- and
 * multi-page TIFF through ImageIO's baseline TIFF plugin. A caller reaches them through
 * {@link dev.simplified.image.ImageFactory ImageFactory} with
 * {@link dev.simplified.image.ImageFormat#TIFF ImageFormat.TIFF}, never directly: both are public only
 * because the factory constructs them from the root package, and each carries
 * {@code @ApiStatus.Internal}.
 * <p>
 * A type belongs here when only TIFF uses it. A helper a second codec package needs moves to
 * {@link dev.simplified.image.codec.common codec.common}, and the options a TIFF write takes are
 * {@link dev.simplified.image.option.TiffWriteOptions TiffWriteOptions}, in
 * {@link dev.simplified.image.option option}.
 */
package dev.simplified.image.codec.tiff;
