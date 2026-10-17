/**
 * The VP8L bitstream: lossless WebP, plus a lossy image's alpha plane - internal.
 * <p>
 * {@link dev.simplified.image.codec.webp.vp8l.VP8LEncoder VP8LEncoder} and
 * {@link dev.simplified.image.codec.webp.vp8l.VP8LDecoder VP8LDecoder} code one image, and
 * {@link dev.simplified.image.codec.webp.vp8l.NearLosslessPreprocess NearLosslessPreprocess} adjusts
 * its pixels within a bounded deviation ahead of a near-lossless encode. The rest - the bit reader and
 * writer, the Huffman trees, the colour cache, the LZ77 backward references and the transforms - is
 * package-private.
 * <p>
 * The three entry points are public only because the WebP reader and writer call them from
 * {@link dev.simplified.image.codec.webp codec.webp}, and each carries {@code @ApiStatus.Internal}: a
 * caller writes lossless WebP through {@link dev.simplified.image.ImageFactory ImageFactory}. Nothing
 * here names the lossy bitstream.
 */
package dev.simplified.image.codec.webp.vp8l;
