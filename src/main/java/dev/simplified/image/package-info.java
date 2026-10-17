/**
 * The static facade that moves images to and from bytes, the format a caller names to choose an
 * encoding, and the forms a whole image is held as.
 * <p>
 * {@link dev.simplified.image.ImageFactory ImageFactory} is the one route to a codec: it detects a
 * format from its magic bytes, reads bytes, files, streams and URLs into an
 * {@link dev.simplified.image.ImageData ImageData}, and writes one out as an
 * {@link dev.simplified.image.ImageFormat ImageFormat}, through static methods over a closed set of
 * built-in readers and writers. A whole image is held as a
 * {@link dev.simplified.image.StaticImageData StaticImageData} or an
 * {@link dev.simplified.image.AnimatedImageData AnimatedImageData}, and as pixels in a
 * {@link dev.simplified.image.PixelBuffer PixelBuffer}.
 * <p>
 * A public type belongs here when it is the facade, the format, or a form a whole image is held as.
 * Parts of an image, choices passed to a method, static toolboxes and options go to the package of the
 * task they serve - {@link dev.simplified.image.frame frame}, {@link dev.simplified.image.pixel pixel}
 * or {@link dev.simplified.image.option option}. A root type's package-private helpers sit beside it,
 * because that is the only place Java keeps them hidden: the Vector API kernels and the
 * {@code BufferedImage} unpacker behind {@code PixelBuffer} are here for that reason.
 */
package dev.simplified.image;
