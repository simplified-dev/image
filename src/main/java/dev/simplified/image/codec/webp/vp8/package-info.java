/**
 * The VP8 bitstream: lossy WebP, plus opt-in inter frames - internal.
 * <p>
 * {@link dev.simplified.image.codec.webp.vp8.VP8Encoder VP8Encoder} and
 * {@link dev.simplified.image.codec.webp.vp8.VP8Decoder VP8Decoder} code one frame on its own, and
 * {@link dev.simplified.image.codec.webp.vp8.VP8EncoderSession VP8EncoderSession} and
 * {@link dev.simplified.image.codec.webp.vp8.VP8DecoderSession VP8DecoderSession} carry the reference
 * frames a sequence of inter frames predicts from. The rest - the boolean range coder, the transforms,
 * prediction, quantisation, token coding, the loop filter and the tables taken from libwebp - is
 * package-private.
 * <p>
 * The four entry points are public only because the WebP reader and writer call them from
 * {@link dev.simplified.image.codec.webp codec.webp}, and each carries {@code @ApiStatus.Internal}: a
 * caller writes lossy WebP through {@link dev.simplified.image.ImageFactory ImageFactory}. Nothing here
 * names the lossless bitstream.
 */
package dev.simplified.image.codec.webp.vp8;
