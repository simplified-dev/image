/**
 * The TGA reader and writer - internal.
 * <p>
 * {@link dev.simplified.image.codec.tga.TgaImageReader TgaImageReader} and
 * {@link dev.simplified.image.codec.tga.TgaImageWriter TgaImageWriter} read and write Truevision TGA,
 * uncompressed or run-length encoded. A caller reaches them through
 * {@link dev.simplified.image.ImageFactory ImageFactory} with
 * {@link dev.simplified.image.ImageFormat#TGA ImageFormat.TGA}, never directly: both are public only
 * because the factory constructs them from the root package, and each carries
 * {@code @ApiStatus.Internal}.
 * <p>
 * A type belongs here when only TGA uses it. A helper a second codec package needs moves to
 * {@link dev.simplified.image.codec.common codec.common}, and the options a TGA write takes are
 * {@link dev.simplified.image.option.TgaWriteOptions TgaWriteOptions}, in
 * {@link dev.simplified.image.option option}.
 */
package dev.simplified.image.codec.tga;
