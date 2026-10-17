# Image

Pure-Java multi-format image codec library with full VP8/VP8L WebP encoding and decoding, animated image support, and parallel processing. Reads and writes BMP, GIF, ICO, JPEG, PNG, PNM, QOI, TGA, TIFF and WebP through one static facade, with per-frame timing, loop control, disposal and blend modes, frame normalization, and frame interpolation - no native dependencies or JNI bindings required.

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Auto-Detection](#auto-detection)
  - [Reading an Image](#reading-an-image)
  - [Writing an Image](#writing-an-image)
  - [Animated Images](#animated-images)
  - [Batch Processing](#batch-processing)
  - [Pixel Manipulation](#pixel-manipulation)
- [Supported Formats](#supported-formats)
- [Architecture](#architecture)
  - [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Pure-Java VP8 and VP8L** - Lossy (VP8) and lossless (VP8L) WebP encoding and decoding with no native libraries. The VP8 encoder picks each macroblock's 16x16 or 4x4 (B_PRED) intra modes by rate-distortion score, with trellis quantization and the normal loop filter, and writes keyframes libwebp decodes; opt-in P-frames add motion-compensated prediction from the previous frame, and opt-in segmentation gives detailed regions a coarser quantizer. The VP8 decoder reads keyframes and P-frames. VP8L brings LZ77 backward references, a colour cache, meta-Huffman groups, the four reversible transforms and near-lossless preprocessing
- **Ten image formats** - Read and write BMP, GIF, ICO, JPEG, PNG, PNM, QOI, TGA, TIFF and WebP with format-specific options for quality, compression, and lossless/lossy mode
- **Animated image support** - Multi-frame GIF and WebP with per-frame timing, loop count, background color, disposal methods (none, do not dispose, restore to background, restore to previous), blend modes (source replace, alpha-over), and partial-frame ANMF encoding that diffs consecutive frames to emit minimal sub-rectangles
- **Parallel frame encoding** - Animated WebP encodes frames concurrently using virtual threads for significantly faster output
- **Batch processing** - `ImageFactory` bulk operations (`fromFiles`, `toFiles`, `fromByteArrays`, `toByteArrays`) process multiple images in parallel with virtual threads
- **Zero-copy pixel access** - `PixelBuffer` wraps `TYPE_INT_ARGB` arrays directly without copying, providing per-pixel read/write access
- **Frame normalization** - `FrameNormalizer` handles variable-sized animation frames with configurable fit modes (contain, cover, stretch), background fill, upscale control, and bicubic interpolation
- **Frame interpolation** - `AnimatedImageData` supports time-based frame lookup with per-pixel linear interpolation and bilinear sampling for smooth animation playback
- **Auto-format detection** - `ImageFactory` identifies formats from magic bytes and routes to the correct codec automatically
- **Multiple I/O sources** - Read from and write to byte arrays, files, streams, URLs, Base64, `BufferedImage`, and classpath resources
- **FXAA anti-aliasing** - `PixelBuffer` applies per-pixel FXAA with parallel scanline processing and luma-based edge detection, to straight or premultiplied colour
- **Pure Java** - No native dependencies, JNI bindings, or platform-specific code
- **JitPack distribution** - Add as a Gradle dependency with no manual installation

## Getting Started

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| [Java](https://adoptium.net/) | **21+** | Required (LTS recommended) |
| [Gradle](https://gradle.org/) | **9.4+** | Or use the included `./gradlew` wrapper |
| [Git](https://git-scm.com/) | **2.x+** | For cloning the repository |

### Installation

Add the JitPack repository and dependency to your `build.gradle.kts`:

```kotlin
repositories {
    maven(url = "https://jitpack.io")
}

dependencies {
    implementation("com.github.simplified-dev:image:master-SNAPSHOT")
}
```

<details>
<summary>Gradle (Groovy)</summary>

```groovy
repositories {
    maven { url 'https://jitpack.io' }
}

dependencies {
    implementation 'com.github.simplified-dev:image:master-SNAPSHOT'
}
```

</details>

<details>
<summary>Maven</summary>

```xml
<repositories>
    <repository>
        <id>jitpack.io</id>
        <url>https://jitpack.io</url>
    </repository>
</repositories>

<dependency>
    <groupId>com.github.simplified-dev</groupId>
    <artifactId>image</artifactId>
    <version>master-SNAPSHOT</version>
</dependency>
```

</details>

> [!NOTE]
> This library depends on other Simplified-Dev modules (`collections`, `utils`) which are also resolved from JitPack automatically.

## Usage

### Auto-Detection

`ImageFactory` is the one route to a codec: a static facade over the ten built-in
readers and writers. It detects the format from magic bytes and routes to the right
codec, so there is no factory to construct and no codec class to name:

```java
import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFactory;
import dev.simplified.image.ImageFormat;

ImageData image = ImageFactory.fromFile(inputFile);
ImageFactory.toFile(image, ImageFormat.WEBP, outputFile);
```

The codec set is closed: each format has exactly one reader and one writer, created
once when `ImageFactory` initialises and shared by every call and thread.

### Reading an Image

Every read goes through `ImageFactory` - from a byte array, file, stream, URL, Base64
string, classpath resource or `BufferedImage`. Name the format to skip detection:

```java
import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFactory;
import dev.simplified.image.ImageFormat;

ImageData detected = ImageFactory.fromByteArray(Files.readAllBytes(path));
ImageData png = ImageFactory.fromByteArray(Files.readAllBytes(path), ImageFormat.PNG);
```

### Writing an Image

Every write option type lives in `dev.simplified.image.option` and uses a fluent
builder. Hand the options to `ImageFactory` beside the format they configure:

```java
import dev.simplified.image.ImageFactory;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.option.WebPWriteOptions;

// Lossless (default)
ImageFactory.toFile(image, ImageFormat.WEBP, outputFile,
    WebPWriteOptions.builder().isLossless().build());

// Lossy with quality control
ImageFactory.toFile(image, ImageFormat.WEBP, outputFile,
    WebPWriteOptions.builder()
        .isLossless(false)
        .withQuality(0.80f)
        .isMultithreaded()
        .build());
```

### Animated Images

The two forms of an image, `StaticImageData` and `AnimatedImageData`, sit beside
`ImageData` in the root package; one frame and its disposal and blend, and the
`LoopCount` an animation plays by, come from `dev.simplified.image.frame`.

A `LoopCount` counts the repeats after the first play: `LoopCount.Preset.NONE`
plays once, `ONCE` twice and `INFINITE` forever, and `LoopCount.of(n)` plays
`n + 1` times. Each codec converts it to its container's own unit - a GIF's
NETSCAPE2.0 block counts repeats, a WebP's `ANIM` chunk counts plays. Both hold
at most 65535: a GIF refuses a count that repeats more often, and a WebP
declares a count that plays more often as 65535 plays. Write options that set
no loop count declare the image's own.

```java
import dev.simplified.image.AnimatedImageData;
import dev.simplified.image.ImageFactory;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.frame.ImageFrame;
import dev.simplified.image.frame.LoopCount;
import dev.simplified.image.option.GifWriteOptions;
import dev.simplified.image.option.WebPWriteOptions;

// Read any animated format (GIF, WebP) - auto-detected.
AnimatedImageData animated = (AnimatedImageData) ImageFactory.fromFile(inputFile);

for (ImageFrame frame : animated.getFrames()) {
    // Access per-frame timing, position, disposal, blend mode, and pixel data.
}

// Re-encode as animated WebP; frames encoded in parallel by default.
ImageFactory.toFile(animated, ImageFormat.WEBP, outputFile,
    WebPWriteOptions.builder().isLossless(false).withQuality(0.9f).build());

// Re-encode as a GIF that plays twice, one repeat after the first play.
ImageFactory.toFile(animated, ImageFormat.GIF, outputFile,
    GifWriteOptions.builder().withLoopCount(LoopCount.Preset.ONCE).build());
```

### Batch Processing

```java
// Decode multiple files in parallel using virtual threads.
ConcurrentList<File> files = Concurrent.newList(...);
ConcurrentList<ImageData> images = ImageFactory.fromFiles(files);

// Encode multiple images in parallel.
ImageFactory.toFiles(images, ImageFormat.PNG, outputFiles);
```

### Pixel Manipulation

`PixelBuffer` is a whole image as pixels, so it sits in the root package; colour,
compositing and drawing work on it comes from `dev.simplified.image.pixel`.

```java
import dev.simplified.image.PixelBuffer;
import dev.simplified.image.pixel.AlphaMode;

PixelBuffer buffer = image.toPixelBuffer();

// Direct zero-copy pixel access.
int argb = buffer.getPixel(x, y);
buffer.setPixel(x, y, 0xFFFF0000); // solid red

// Apply FXAA anti-aliasing (parallel scanline processing) to straight colour, as a decoded image holds.
buffer.applyFxaa();

// A buffer drawn onto transparency with BlendMode.NORMAL holds its colour already scaled by alpha; say so.
PixelBuffer render = PixelBuffer.create(256, 256);
render.blit(buffer, 0, 0);
render.applyFxaa(AlphaMode.PREMULTIPLIED);
```

## Supported Formats

Every format reads and writes through `ImageFactory`, named by its `ImageFormat`
constant. The write options are in `dev.simplified.image.option`.

| Format | `ImageFormat` | Animated | Write options |
|---|---|---|---|
| BMP | `BMP` | No | None (lossless) |
| GIF | `GIF` | Yes | `GifWriteOptions` - loop count, transparency, the background and alpha threshold partial alpha is flattened by, delay fidelity |
| ICO | `ICO` | No | None - reads the largest image, writes one image as an embedded PNG |
| JPEG | `JPEG` | No | `JpegWriteOptions` - quality (0.0-1.0, default 0.75) |
| PNG | `PNG` | No | `PngWriteOptions` - compression level (0-9, default 6) |
| PNM | `PNM` | No | `PnmWriteOptions` - PBM, PGM or PPM (default PPM), ASCII or binary (default binary) |
| QOI | `QOI` | No | None (lossless) |
| TGA | `TGA` | No | `TgaWriteOptions` - run-length encoding (default on) |
| TIFF | `TIFF` | Multi-page | `TiffWriteOptions` - compression: none, LZW, Deflate (default) or PackBits |
| WebP | `WEBP` | Yes | `WebPWriteOptions` - lossless (default) or lossy, quality (0.0-1.0), loop count, multithreaded, alpha compression, P-frames, near-lossless |

## Architecture

### Project Structure

```
src/main/java/dev/simplified/image/
├── ImageFactory.java           # static facade - the one route to a codec
├── ImageFormat.java            # the ten formats and their magic bytes
├── ImageData.java              # what every read returns
├── StaticImageData.java        # a single image
├── AnimatedImageData.java      # timed frames, loop count and canvas
├── PixelBuffer.java            # zero-copy ARGB pixels, with package-private SIMD kernels beside it
├── frame/                      # ImageFrame, FrameBlend, FrameDisposal, LoopCount, FrameNormalizer, FitMode
├── pixel/                      # ColorMath, BlendMode, AlphaMode, Background, PixelGraphics, PixelMask, PixelDiff, DiffType, Resample, PixelBufferPool
├── option/                     # ImageReadOptions, ImageWriteOptions and every format's *WriteOptions
├── exception/                  # ImageException hierarchy
└── codec/
    ├── ImageReader.java        # reader SPI
    ├── ImageWriter.java        # writer SPI
    ├── common/                 # LittleEndian, ImageIOBridge - helpers two or more formats share
    ├── bmp/                    # BmpImageReader, BmpImageWriter
    ├── gif/                    # GifImageReader, GifImageWriter
    ├── ico/                    # IcoImageReader, IcoImageWriter
    ├── jpeg/                   # JpegImageReader, JpegImageWriter
    ├── png/                    # PngImageReader, PngImageWriter
    ├── pnm/                    # PnmImageReader, PnmImageWriter
    ├── qoi/                    # QoiImageReader, QoiImageWriter
    ├── tga/                    # TgaImageReader, TgaImageWriter
    ├── tiff/                   # TiffImageReader, TiffImageWriter
    └── webp/
        ├── WebPImageReader.java
        ├── WebPImageWriter.java
        ├── RiffContainer.java  # RIFF parse/write
        ├── WebPChunk.java      # chunk model + Type enum for FourCC
        ├── FrameDiffUtil.java  # partial-frame ANMF bounding-box diffing
        ├── vp8/                # VP8Encoder, VP8Decoder, VP8EncoderSession, VP8DecoderSession + BooleanEncoder/Decoder, DCT, IntraPrediction, SubpelPrediction, Quantizer, QuantMatrix, TrellisQuantizer, Macroblock, NearMvs, VP8Tables, VP8Costs, VP8TokenEncoder/Decoder, LoopFilter, RateDistortion, ChromaUpsampler
        └── vp8l/               # VP8LEncoder, VP8LDecoder, NearLosslessPreprocess + BitReader/Writer, HuffmanTree, ColorCache, LZ77, VP8LTransform
```

| Package | Description |
|---|---|
| `dev.simplified.image` | The static facade `ImageFactory`, the `ImageFormat` enum, `ImageData` and its two forms, and `PixelBuffer` |
| `dev.simplified.image.frame` | One frame of an image - `ImageFrame` and its disposal and blend enums - and `FrameNormalizer`, which fits a frame sequence to one canvas |
| `dev.simplified.image.pixel` | Colour math, blend and alpha modes, background fills, `PixelGraphics` drawing, masks, diffs, resampling and the buffer pool |
| `dev.simplified.image.option` | Every read and write options type, with its builder and nested enums |
| `dev.simplified.image.exception` | `ImageException` hierarchy for decode/encode/unsupported-format errors |
| `dev.simplified.image.codec` | The `ImageReader` / `ImageWriter` SPI every built-in codec implements |
| `dev.simplified.image.codec.*` | Internal - each format's reader and writer, the helpers they share in `codec.common`, and the WebP bitstreams in `codec.webp.vp8` (lossy) and `codec.webp.vp8l` (lossless, and a lossy image's alpha plane) |

> [!TIP]
> A caller imports from the six packages above and never from below `codec`.
> The readers and writers stay public only because `ImageFactory` constructs them
> from the root package; each carries `@ApiStatus.Internal`, so IntelliJ flags a
> use outside the library, and the library's own `CallerSurfaceTest` fails if a
> caller type starts importing one or naming one in a signature.

## Dependencies

| Dependency | Version | Scope |
|---|---|---|
| [Log4j2](https://logging.apache.org/log4j/) | 2.25.3 | API |
| [JetBrains Annotations](https://github.com/JetBrains/java-annotations) | 26.0.2 | API |
| [Simplified Annotations](https://github.com/simplified-dev/annotations) | 2.7.0 | Compile-only |
| [collections](https://github.com/Simplified-Dev/collections) | master-SNAPSHOT | API (Simplified-Dev) |
| [utils](https://github.com/Simplified-Dev/utils) | master-SNAPSHOT | API (Simplified-Dev) |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style
guidelines, and how to submit a pull request.

## License

This project is licensed under the **Apache License 2.0** - see
[LICENSE.md](LICENSE.md) for the full text.

The WebP codecs port code and tables from [libwebp](https://chromium.googlesource.com/webm/libwebp)
and [libvpx](https://chromium.googlesource.com/webm/libvpx), both BSD-licensed, and from RFC 6386.
[NOTICE](NOTICE) names what each one supplies and carries their copyright notices and licence
terms; the jar ships it under `META-INF`.
