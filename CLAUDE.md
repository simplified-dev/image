# image

Pure-Java codec library for ten formats: BMP, GIF, ICO, JPEG, PNG, PNM, QOI, TGA, TIFF and WebP (VP8 lossy + VP8L lossless). Animated frames with per-frame timing, disposal, and blend; virtual-thread batch encode; zero-copy `PixelBuffer`.

## Packages

A caller imports from the first six packages and never from below `codec`. `ImageFactory` is the only route to a codec: a static facade over a closed set of ten built-in readers and writers, which its static initializer creates once.

- `dev.simplified.image` - `ImageFactory` (`fromFile`/`toFile`/`fromByteArray`/`toByteArray` plus `*s` batch variants and `detectFormat`), `ImageFormat`, `ImageData` and its two forms `StaticImageData` and `AnimatedImageData`, and `PixelBuffer`. Package-private beside `PixelBuffer`: `PixelVector`, `VectorOps` (the one source importing `jdk.incubator.vector`), `BufferedImageUnpacker`
- `.frame` - `ImageFrame`, `FrameBlend`, `FrameDisposal`, `LoopCount` (+ nested `Preset` enum, `Custom` record and `Builder`), `FrameNormalizer`, `FitMode`
- `.pixel` - `ColorMath`, `BlendMode`, `AlphaMode`, `Background`, `PixelGraphics`, `PixelMask`, `PixelDiff`, `DiffType`, `Resample`, `PixelBufferPool`
- `.option` - `ImageReadOptions`/`ImageWriteOptions` and every format's write options: `GifWriteOptions`, `JpegWriteOptions`, `PngWriteOptions`, `PnmWriteOptions`, `TgaWriteOptions`, `TiffWriteOptions`, `WebPWriteOptions`. An option enum nests in its type (`GifWriteOptions.DelayFidelity`, `PnmWriteOptions.Variant`)
- `.exception` - `ImageException` hierarchy (`ImageDecodeException`, `ImageEncodeException`, `UnsupportedFormatException`)
- `.codec` - the `ImageReader`/`ImageWriter` SPI and nothing else

Internal, every package below `codec`. A type there is public only where another package calls it, and then carries `@ApiStatus.Internal`:

- `.codec.{bmp,gif,ico,jpeg,png,pnm,qoi,tga,tiff}` - `<Format>ImageReader`/`<Format>ImageWriter`, constructed by `ImageFactory`; `IcoImageWriter` encodes its payload through `PngImageWriter`
- `.codec.common` - helpers two or more codec packages share: `LittleEndian`, `ImageIOBridge`. It names no format
- `.codec.webp` - `WebPImageReader`, `WebPImageWriter`. Package-private: `RiffContainer`, `WebPChunk` (+ nested `Type` enum for FourCC), `FrameDiffUtil`
- `.codec.webp.vp8` - the VP8 bitstream: lossy WebP, plus opt-in inter frames. Public: `VP8Encoder`, `VP8Decoder`, `VP8EncoderSession`, `VP8DecoderSession`. Package-private: `BooleanEncoder`/`BooleanDecoder`, `DCT`, `IntraPrediction`, `SubpelPrediction`, `QuantMatrix`, `TrellisQuantizer`, `Macroblock`, `NearMvs`, `VP8Tables`, `VP8Costs`, `VP8TokenEncoder`/`VP8TokenDecoder`, `LoopFilter`, `ChromaUpsampler`
- `.codec.webp.vp8l` - the VP8L bitstream: lossless WebP, plus a lossy image's alpha plane. Public: `VP8LEncoder` (+ nested `TransformMode`), `VP8LDecoder`, `NearLosslessPreprocess`. Package-private: `BitReader`, `BitWriter`, `HuffmanTree`, `ColorCache`, `LZ77`, `VP8LTransform`

Every package has a `package-info.java` charter saying what belongs in it. `guard/CallerSurfaceTest` holds the layout over the main sources: every package classified and chartered, no caller source importing an internal type except `ImageFactory`, no caller signature naming one, options types in `.option` and readers and writers in their format's package, `.codec.common` and every package below it naming no format, and every internal type another package can reach tagged and named from another package. Reach is effective visibility, read from the compiled classes: public, or nested and protected, all the way out, a type declared inside an interface being public whether or not it says so.

## VP8 codec state

The encoder writes keyframes and, opt-in, P-frames. Each source chroma sample averages its 2x2 block of pixels, weighting a partly transparent block by alpha as libwebp's `AccumulateRGBA` does. A keyframe macroblock's modes are picked as libwebp's rate-distortion search picks them (`PickBestIntra16`, `PickBestIntra4`, `PickBestUV`): every candidate is quantized and reconstructed and scored on distortion, texture distortion, mode cost and residual rate under the lambdas libwebp's `SetupMatrices` derives, and the best 16x16 mode and the B_PRED sub-blocks are compared on `lambda_mode`. Chroma takes the best of the four 8x8 modes. Coefficients are trellis-quantized against the frame's token costs, the normal loop filter runs at a level derived from the quantizer (zero, so off, at the top of the quality range), and tokens go out over four partitions.

P-frames (`WebPWriteOptions.usePFrames`, through `VP8EncoderSession`) predict from the LAST reference only. Each macroblock takes the best rate-distortion score of ZEROMV skip, NEAREST and NEAR with and without residual, NEWMV from a motion search, SPLITMV, and intra 16x16 or B_PRED, and the motion-vector, mode and token probabilities update with a one-frame lag. GOLDEN and ALTREF are filled by each keyframe and are never refreshed or predicted from. `WebPWriteOptions.autoSegment` adds variance-based segmentation: four segments, each with its own quantizer delta, the map coded at the probabilities its histogram gives, as libwebp's `SetSegmentProbas` codes it. libwebp decodes the keyframe output and rejects every P-frame, so P-frame output reads back only through this library and libvpx-based decoders.

The tables are taken from the two reference codecs and must stay bit-exact, or a decoder misreads the bitstream: the token tree and coefficient probabilities from libwebp's `src/dec/tree_dec.c`, the bit costs from `src/enc/cost_enc.c` and `cost_enc.h`, the quantizer bias and sharpening matrices from `src/enc/quant_enc.c`, and the inter-frame mode, motion-vector and SPLITMV tables from libvpx, chiefly `vp8/common/entropymode.c`, `vp8/common/entropymv.c`, `vp8/common/modecont.c` and `vp8/decoder/decodemv.c`.

The decoder reads keyframes and P-frames, the latter against all three references with their refresh, copy and sign-bias flags, carrying the token, mode and MV probabilities from frame to frame as each frame's `refresh_entropy_probs` says, plus SPLITMV, segmentation, loop-filter deltas, both loop filters and every token partition. Coefficients go through the full token tree in `VP8TokenDecoder`, after libwebp's `GetCoeffsFast`, and chroma is upsampled with libwebp's fancy bilinear filter and converted to RGB in libwebp's fixed point. A keyframe decodes to exactly the pixels libwebp decodes it to, whichever encoder wrote it, and the encoder reconstructs every frame exactly as the decoder will; `VP8DecodeParityTest` holds both. A P-frame decodes only through a `VP8DecoderSession` holding its reference.

## Builders

Write options live in `.option` and use fluent builders: `WebPWriteOptions.builder().isLossless(bool).withQuality(float).isMultithreaded().withLoopCount(LoopCount.Preset.INFINITE).build()`. Same pattern for `GifWriteOptions`, `JpegWriteOptions`, `PngWriteOptions`, `PnmWriteOptions`, `TgaWriteOptions`. They are handed to the factory beside the format they configure: `ImageFactory.toFile(image, ImageFormat.WEBP, out, options)`.

## Build

```bash
./gradlew build                                                       # compile + test
./gradlew test                                                        # tests only
./gradlew test --tests dev.simplified.image.codec.webp.vp8.VP8CodecTest
```

The VP8 and WebP tests check this library's output against libwebp through Python's `webp` package (libwebp bindings), always by way of `LibwebpOracle` in the test sources. It takes the first interpreter that starts and imports `webp` and `numpy`, trying `-Dvp8.pythonBin=<path>` (the build forwards it to the test JVM) or else the `VP8_PYTHON_BIN` environment variable, then `python3`, `python` and `py`, and it hands each script its file paths as arguments. When none qualifies, a test that consults libwebp aborts with `TestAbortedException` and so reports skipped, never passed; a check of this library's own output that the test makes first still runs, and still fails on its own. A script that runs and exits non-zero fails its test. Three diagnostics in `WebPRoundTripTest` abort the same way when the local tooltip files they read are absent.

`VP8LibvpxParityTest` holds P-frames to libvpx the same way, through `LibvpxOracle` and an `ffmpeg` build that lists `libvpx` among both its encoders and decoders: `-Dvp8.ffmpegBin=<path>` (forwarded likewise), else the `VP8_FFMPEG_BIN` environment variable, then `ffmpeg`. Streams travel as IVF files, libvpx decodes on four threads, and an ffmpeg run that exits non-zero or prints an error fails its test. Without such a build those tests abort and report skipped.

## Info

- Java 21 toolchain (`build.gradle.kts`)
- Internal deps: Simplified-Dev `collections`, `utils` (both pulled via JitPack, pinned to a commit)
- External deps: Log4j2, Simplified Annotations, JetBrains annotations
- Third-party code: `NOTICE` names every class ported from libwebp, libvpx or RFC 6386 and carries their licences; the jar packs it into `META-INF` with `LICENSE.md`, the Apache-2.0 text it points to. Code or tables taken from another codec get a line there too
- Tests: JUnit 5 + Hamcrest
- 93 main sources plus a `package-info.java` per package; 40 test sources
