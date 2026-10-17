# image

Pure-Java codec library for ten formats: BMP, GIF, ICO, JPEG, PNG, PNM, QOI, TGA, TIFF and WebP (VP8 lossy + VP8L lossless). Animated frames with per-frame timing, disposal, and blend; virtual-thread batch encode; zero-copy `PixelBuffer`.

## Packages

A caller imports from the first six packages and never from below `codec`. `ImageFactory` is the only route to a codec: a static facade over a closed set of ten built-in readers and writers, which its static initializer creates once.

- `dev.simplified.image` - `ImageFactory` (`fromFile`/`toFile`/`fromByteArray`/`toByteArray` plus `*s` batch variants and `detectFormat`), `ImageFormat`, `ImageData` and its two forms `StaticImageData` and `AnimatedImageData`, and `PixelBuffer`. Package-private beside `PixelBuffer`: `PixelVector`, `VectorOps` (the one source importing `jdk.incubator.vector`), `BufferedImageUnpacker`
- `.frame` - `ImageFrame`, `FrameBlend`, `FrameDisposal`, `FrameNormalizer`, `FitMode`
- `.pixel` - `ColorMath`, `BlendMode`, `AlphaMode`, `Background`, `PixelGraphics`, `PixelMask`, `PixelDiff`, `DiffType`, `Resample`, `PixelBufferPool`
- `.option` - `ImageReadOptions`/`ImageWriteOptions` and every format's write options: `GifWriteOptions`, `JpegWriteOptions`, `PngWriteOptions`, `PnmWriteOptions`, `TgaWriteOptions`, `TiffWriteOptions`, `WebPWriteOptions`. An option enum nests in its type (`GifWriteOptions.DelayFidelity`, `PnmWriteOptions.Variant`)
- `.exception` - `ImageException` hierarchy (`ImageDecodeException`, `ImageEncodeException`, `UnsupportedFormatException`)
- `.codec` - the `ImageReader`/`ImageWriter` SPI and nothing else

Internal, every package below `codec`. A type there is public only where another package calls it, and then carries `@ApiStatus.Internal`:

- `.codec.{bmp,gif,ico,jpeg,png,pnm,qoi,tga,tiff}` - `<Format>ImageReader`/`<Format>ImageWriter`, constructed by `ImageFactory`; `IcoImageWriter` encodes its payload through `PngImageWriter`
- `.codec.common` - helpers two or more codec packages share: `LittleEndian`, `ImageIOBridge`. It names no format
- `.codec.webp` - `WebPImageReader`, `WebPImageWriter`. Package-private: `RiffContainer`, `WebPChunk` (+ nested `Type` enum for FourCC), `FrameDiffUtil`
- `.codec.webp.vp8` - the VP8 bitstream: lossy WebP, plus opt-in inter frames. Public: `VP8Encoder`, `VP8Decoder`, `VP8EncoderSession`, `VP8DecoderSession`. Package-private: `BooleanEncoder`/`BooleanDecoder`, `DCT`, `IntraPrediction`, `SubpelPrediction`, `Quantizer`, `QuantMatrix`, `TrellisQuantizer`, `Macroblock`, `NearMvs`, `VP8Tables`, `VP8Costs`, `VP8TokenEncoder`/`VP8TokenDecoder`, `LoopFilter`, `RateDistortion`, `ChromaUpsampler`
- `.codec.webp.vp8l` - the VP8L bitstream: lossless WebP, plus a lossy image's alpha plane. Public: `VP8LEncoder` (+ nested `TransformMode`), `VP8LDecoder`, `NearLosslessPreprocess`. Package-private: `BitReader`, `BitWriter`, `HuffmanTree`, `ColorCache`, `LZ77`, `VP8LTransform`

Every package has a `package-info.java` charter saying what belongs in it. `guard/CallerSurfaceTest` holds the layout over the main sources: every package classified and chartered, no caller source importing an internal type except `ImageFactory`, no caller signature naming one, options types in `.option` and readers and writers in their format's package, `.codec.common` and every package below it naming no format, and every public internal type tagged and named from another package.

## VP8 codec state

The encoder writes keyframes and, opt-in, P-frames. A keyframe macroblock is coded either whole, as 16x16 DC/V/H/TM picked by distortion plus mode cost, or as B_PRED, each 4x4 sub-block taking the best of the ten sub-block modes by rate-distortion score; the lower score of the two wins. Chroma takes one of the four 8x8 modes the same way. Coefficients are trellis-quantized against the frame's token costs, the normal loop filter runs at a level derived from the quantizer (zero, so off, at the top of the quality range), and tokens go out over four partitions.

P-frames (`WebPWriteOptions.usePFrames`, through `VP8EncoderSession`) predict from the LAST reference only. Each macroblock takes the best rate-distortion score of ZEROMV skip, NEAREST and NEAR with and without residual, NEWMV from a motion search, SPLITMV, and intra 16x16 or B_PRED, and the motion-vector, mode and token probabilities update with a one-frame lag. GOLDEN and ALTREF are filled by each keyframe and are never refreshed or predicted from. `WebPWriteOptions.autoSegment` adds variance-based segmentation: four segments, each with its own quantizer delta. libwebp decodes the keyframe output and rejects every P-frame, so P-frame output reads back only through this library and libvpx-based decoders.

The tables are taken from the two reference codecs and must stay bit-exact, or a decoder misreads the bitstream: the token tree and coefficient probabilities from libwebp's `src/dec/tree_dec.c`, the bit costs from `src/enc/cost_enc.c` and `cost_enc.h`, the quantizer bias and sharpening matrices from `src/enc/quant_enc.c`, and the inter-frame mode, motion-vector and SPLITMV tables from libvpx, chiefly `vp8/common/entropymode.c`, `vp8/common/modecont.c` and `vp8/decoder/decodemv.c`.

The decoder reads keyframes and P-frames, the latter against all three references with their refresh, copy and sign-bias flags, plus SPLITMV, segmentation, loop-filter deltas, both loop filters and every token partition. Coefficients go through the full token tree in `VP8TokenDecoder`, after libwebp's `GetCoeffsFast`, and chroma is upsampled with libwebp's fancy bilinear filter. A P-frame decodes only through a `VP8DecoderSession` holding its reference.

## Builders

Write options live in `.option` and use fluent builders: `WebPWriteOptions.builder().isLossless(bool).withQuality(float).isMultithreaded().withLoopCount(int).build()`. Same pattern for `GifWriteOptions`, `JpegWriteOptions`, `PngWriteOptions`, `PnmWriteOptions`, `TgaWriteOptions`. They are handed to the factory beside the format they configure: `ImageFactory.toFile(image, ImageFormat.WEBP, out, options)`.

## Build

```bash
./gradlew build                                                       # compile + test
./gradlew test                                                        # tests only
./gradlew test --tests dev.simplified.image.codec.webp.vp8.VP8CodecTest
```

VP8 lossy tests shell out to Python `webp` package (libwebp bindings) for round-trip validation. Tests abort gracefully via `TestAbortedException` when Python or the package is missing, rather than failing.

## Info

- Java 21 toolchain (`build.gradle.kts`)
- Internal deps: Simplified-Dev `collections`, `utils` (both pulled via JitPack, pinned to a commit)
- External deps: Log4j2, Simplified Annotations, JetBrains annotations
- Third-party code: `NOTICE` names every class ported from libwebp, libvpx or RFC 6386 and carries their licences; the jar packs it into `META-INF`. Code or tables taken from another codec get a line there too
- Tests: JUnit 5 + Hamcrest
- 94 main sources plus a `package-info.java` per package; 22 test sources
