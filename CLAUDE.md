# image

Pure-Java codec library: BMP, GIF, JPEG, PNG, WebP (VP8 lossy + VP8L lossless). Animated frames with per-frame timing, disposal, and blend; virtual-thread batch encode; zero-copy `PixelBuffer`.

## Packages

- `dev.simplified.image` - `ImageData`, `ImageFormat`, `ImageFactory` (instance facade - `fromFile`/`toFile`/`fromByteArray`/`toByteArray` plus `*s` batch variants)
- `.data` - `StaticImageData`, `AnimatedImageData`, `ImageFrame`, `FrameBlend`, `FrameDisposal`
- `.pixel` - `PixelBuffer`, `PixelGraphics`, `ColorMath`, `BlendMode`, `Resample`
- `.transform` - `FrameNormalizer`, `FitMode`
- `.exception` - `ImageException` hierarchy (`ImageDecodeException`, `ImageEncodeException`, `UnsupportedFormatException`)
- `.codec` - `ImageReader`/`ImageWriter` SPI + `ImageReadOptions`/`ImageWriteOptions`
- `.codec.{bmp,gif,jpeg,png}` - `<Format>ImageReader`/`<Format>ImageWriter` (+ `<Format>WriteOptions` builders where relevant)
- `.codec.webp` - `WebPImageReader`, `WebPImageWriter`, `WebPWriteOptions`, `RiffContainer`, `WebPChunk` (+ nested `Type` enum for FourCC)
- `.codec.webp.lossless` - VP8L internals. Public: `VP8LEncoder`, `VP8LDecoder`. Package-private: `BitReader`, `BitWriter`, `HuffmanTree`, `ColorCache`, `LZ77`, `VP8LTransform`.
- `.codec.webp.lossy` - VP8 internals. Public: `VP8Encoder`, `VP8Decoder`. Package-private: `BooleanEncoder`/`BooleanDecoder`, `DCT`, `IntraPrediction`, `Quantizer`, `Macroblock`, `VP8Tables`, `VP8TokenEncoder`/`VP8TokenDecoder`, `LoopFilter`, `RateDistortion`.

## VP8 encoder state

Keyframe-only, 16x16 intra. All four luma and chroma modes (DC/V/H/TM) with SSE-based mode selection. No B_PRED 4x4 sub-blocks, no inter-frame. Quantizer / token-tree / coefficient probability tables are copied verbatim from libwebp (`src/enc/tree_enc.c`, `src/enc/frame_enc.c`, `src/enc/quant_enc.c`) - they must stay bit-exact or libwebp will reject the bitstream. `WebPImageWriter` with `isLossless(false)` is production-ready.

The VP8 decoder handles the full intra-4x4 B_PRED sub-mode tree for reading libwebp-produced files, but also still carries shortcuts (fixed-width 11-bit coefficient reads) from the old sketch encoder era - a spec-compliant decoder rewrite is a future task.

## Builders

Write options use fluent builders: `WebPWriteOptions.builder().isLossless(bool).withQuality(float).isMultithreaded().withLoopCount(int).build()`. Same pattern for `GifWriteOptions`, `JpegWriteOptions`, `PngWriteOptions`.

## Build

```bash
./gradlew build                                                       # compile + test
./gradlew test                                                        # tests only
./gradlew test --tests dev.simplified.image.codec.webp.lossy.VP8CodecTest
```

VP8 lossy tests shell out to Python `webp` package (libwebp bindings) for round-trip validation. Tests abort gracefully via `TestAbortedException` when Python or the package is missing, rather than failing.

## Info

- Java 21 toolchain (`build.gradle.kts`)
- Internal deps: Simplified-Dev `collections`, `utils`, `reflection` (all pulled via JitPack)
- External deps: Log4j2, Lombok, JetBrains annotations
- Tests: JUnit 5 + Hamcrest
- 60 source classes, 8 test classes
