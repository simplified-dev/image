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

## VP8 encoder state

Keyframe-only, 16x16 intra. All four luma and chroma modes (DC/V/H/TM) with SSE-based mode selection. No B_PRED 4x4 sub-blocks, no inter-frame. Quantizer / token-tree / coefficient probability tables are copied verbatim from libwebp (`src/enc/tree_enc.c`, `src/enc/frame_enc.c`, `src/enc/quant_enc.c`) - they must stay bit-exact or libwebp will reject the bitstream. `WebPImageWriter` with `isLossless(false)` is production-ready.

The VP8 decoder handles the full intra-4x4 B_PRED sub-mode tree for reading libwebp-produced files, but also still carries shortcuts (fixed-width 11-bit coefficient reads) from the old sketch encoder era - a spec-compliant decoder rewrite is a future task.

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
- Tests: JUnit 5 + Hamcrest
- 94 main sources plus a `package-info.java` per package; 22 test sources
