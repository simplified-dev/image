/**
 * The contract every built-in codec implements: an {@link dev.simplified.image.codec.ImageReader
 * ImageReader} decodes bytes into an {@link dev.simplified.image.ImageData ImageData}, and an
 * {@link dev.simplified.image.codec.ImageWriter ImageWriter} encodes one.
 * <p>
 * A caller reads and writes through {@link dev.simplified.image.ImageFactory ImageFactory}, which holds
 * one reader and one writer for each {@link dev.simplified.image.ImageFormat ImageFormat}. The set is
 * closed, so no caller implements this contract or hands one to the factory.
 * <p>
 * This package holds the contract and nothing else. Each format's reader and writer sit in a package of
 * their own one level below it, and every package below it is internal.
 */
package dev.simplified.image.codec;
