/**
 * The WebP reader, writer and RIFF container - internal.
 * <p>
 * {@link dev.simplified.image.codec.webp.WebPImageReader WebPImageReader} and
 * {@link dev.simplified.image.codec.webp.WebPImageWriter WebPImageWriter} read and write still and
 * animated WebP, framing each image in the RIFF container and handing its bitstream to
 * {@link dev.simplified.image.codec.webp.vp8 vp8} for lossy coding or
 * {@link dev.simplified.image.codec.webp.vp8l vp8l} for lossless coding and for a lossy image's alpha
 * plane. The container's parser and chunk model, and the diff that finds the rectangle a partial
 * animation frame is written as, are package-private.
 * <p>
 * A caller reaches the pair through {@link dev.simplified.image.ImageFactory ImageFactory} with
 * {@link dev.simplified.image.ImageFormat#WEBP ImageFormat.WEBP}, never directly: both are public only
 * because the factory constructs them from the root package, and each carries
 * {@code @ApiStatus.Internal}. The options a WebP write takes are
 * {@link dev.simplified.image.option.WebPWriteOptions WebPWriteOptions}, in
 * {@link dev.simplified.image.option option}.
 * <p>
 * A type belongs here when it frames a WebP file or chooses between its bitstreams; the bitstreams
 * themselves sit one level down, one package each.
 */
package dev.simplified.image.codec.webp;
