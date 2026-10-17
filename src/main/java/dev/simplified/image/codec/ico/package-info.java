/**
 * The ICO reader and writer - internal.
 * <p>
 * {@link dev.simplified.image.codec.ico.IcoImageReader IcoImageReader} and
 * {@link dev.simplified.image.codec.ico.IcoImageWriter IcoImageWriter} read the largest image of a
 * Windows icon and write a one-image icon whose payload is an embedded PNG, which the writer encodes
 * through {@link dev.simplified.image.codec.png.PngImageWriter PngImageWriter}. A caller reaches them
 * through {@link dev.simplified.image.ImageFactory ImageFactory} with
 * {@link dev.simplified.image.ImageFormat#ICO ImageFormat.ICO}, never directly: both are public only
 * because the factory constructs them from the root package, and each carries
 * {@code @ApiStatus.Internal}.
 * <p>
 * A type belongs here when only ICO uses it. A helper a second codec package needs moves to
 * {@link dev.simplified.image.codec.common codec.common}, and an ICO write takes no options of its own.
 */
package dev.simplified.image.codec.ico;
