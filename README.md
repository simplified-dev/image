# Image

Pure-Java multi-format image codec library with full VP8/VP8L WebP encoding and decoding, animated image support, and parallel processing. Reads and writes BMP, GIF, JPEG, PNG, and WebP with per-frame timing, loop control, disposal and blend modes, frame normalization, and frame interpolation - no native dependencies or JNI bindings required.

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

- **Pure-Java VP8 and VP8L** - Lossy (VP8) and lossless (VP8L) WebP encoding and decoding with no native libraries, including the VP8 boolean range coder, VP8L Huffman + LZ77 compression with ColorCache, DCT/WHT transforms, trellis quantization, near-lossless preprocessing, and SSE-based 16x16 intra-prediction mode selection (DC/V/H/TM). The VP8 encoder emits keyframes libwebp accepts; the VP8 decoder supports the full intra-4x4 B_PRED sub-mode tree
- **Five image formats** - Read and write BMP, GIF, JPEG, PNG, and WebP with format-specific options for quality, compression level, and lossless/lossy mode
- **Animated image support** - Multi-frame GIF and WebP with per-frame timing, loop count, background color, disposal methods (none, do not dispose, restore to background, restore to previous), blend modes (source replace, alpha-over), and partial-frame ANMF encoding that diffs consecutive frames to emit minimal sub-rectangles
- **Parallel frame encoding** - Animated WebP encodes frames concurrently using virtual threads for significantly faster output
- **Batch processing** - `ImageFactory` bulk operations (`fromFiles`, `toFiles`, `fromByteArrays`, `toByteArrays`) process multiple images in parallel with virtual threads
- **Zero-copy pixel access** - `PixelBuffer` wraps `TYPE_INT_ARGB` arrays directly without copying, providing per-pixel read/write access
- **Frame normalization** - `FrameNormalizer` handles variable-sized animation frames with configurable fit modes (contain, cover, stretch), background fill, upscale control, and bicubic interpolation
- **Frame interpolation** - `AnimatedImageData` supports time-based frame lookup with per-pixel linear interpolation and bilinear sampling for smooth animation playback
- **Auto-format detection** - `ImageFactory` identifies formats from magic bytes and routes to the correct codec automatically
- **Multiple I/O sources** - Read from and write to byte arrays, files, streams, URLs, Base64, `BufferedImage`, and classpath resources
- **FXAA anti-aliasing** - `PixelBuffer` applies per-pixel FXAA with parallel scanline processing and luma-based edge detection
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

`ImageFactory` is a small instance-based facade that detects the format from magic
bytes and routes to the correct codec:

```java
import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFactory;
import dev.simplified.image.ImageFormat;

ImageFactory factory = new ImageFactory();
ImageData image = factory.fromFile(inputFile);
factory.toFile(image, ImageFormat.WEBP, outputFile);
```

### Reading an Image

```java
import dev.simplified.image.ImageData;
import dev.simplified.image.codec.png.PngImageReader;

PngImageReader reader = new PngImageReader();
ImageData image = reader.read(Files.readAllBytes(path), null);
```

### Writing an Image

All codec write options use a fluent builder. Construct the options, hand them to
`ImageFactory` or the codec-specific writer:

```java
import dev.simplified.image.ImageFactory;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.codec.webp.WebPWriteOptions;

ImageFactory factory = new ImageFactory();

// Lossless (default)
factory.toFile(image, ImageFormat.WEBP, outputFile,
    WebPWriteOptions.builder().isLossless().build());

// Lossy with quality control
factory.toFile(image, ImageFormat.WEBP, outputFile,
    WebPWriteOptions.builder()
        .isLossless(false)
        .withQuality(0.80f)
        .isMultithreaded()
        .build());
```

### Animated Images

```java
import dev.simplified.image.ImageFactory;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.data.AnimatedImageData;
import dev.simplified.image.data.ImageFrame;
import dev.simplified.image.codec.webp.WebPWriteOptions;

ImageFactory factory = new ImageFactory();

// Read any animated format (GIF, WebP) - auto-detected.
AnimatedImageData animated = (AnimatedImageData) factory.fromFile(inputFile);

for (ImageFrame frame : animated.getFrames()) {
    // Access per-frame timing, position, disposal, blend mode, and pixel data.
}

// Re-encode as animated WebP; frames encoded in parallel by default.
factory.toFile(animated, ImageFormat.WEBP, outputFile,
    WebPWriteOptions.builder().isLossless(false).withQuality(0.9f).build());
```

### Batch Processing

```java
// Decode multiple files in parallel using virtual threads.
ConcurrentList<File> files = Concurrent.newList(...);
ConcurrentList<ImageData> images = factory.fromFiles(files);

// Encode multiple images in parallel.
factory.toFiles(images, ImageFormat.PNG, outputFiles);
```

### Pixel Manipulation

```java
import dev.simplified.image.pixel.PixelBuffer;

PixelBuffer buffer = image.toPixelBuffer();

// Direct zero-copy pixel access.
int argb = buffer.getPixel(x, y);
buffer.setPixel(x, y, 0xFFFF0000); // solid red

// Apply FXAA anti-aliasing (parallel scanline processing).
buffer.applyFxaa();
```

## Supported Formats

| Format | Read | Write | Animated | Options |
|---|---|---|---|---|
| BMP | `BmpImageReader` | `BmpImageWriter` | No | None (lossless) |
| GIF | `GifImageReader` | `GifImageWriter` | Yes | Loop count, transparency, disposal |
| JPEG | `JpegImageReader` | `JpegImageWriter` | No | Quality (0.0-1.0, default 0.75) |
| PNG | `PngImageReader` | `PngImageWriter` | No | Compression level (0-9, default 6) |
| WebP | `WebPImageReader` | `WebPImageWriter` | Yes | Lossy/lossless, quality (0.0-1.0), multithreaded, alpha compression |

## Architecture

### Project Structure

```
src/main/java/dev/simplified/image/
├── ImageData.java              # sealed base type
├── ImageFactory.java           # auto-detect facade
├── ImageFormat.java
├── data/                       # StaticImageData, AnimatedImageData, ImageFrame, FrameBlend, FrameDisposal
├── pixel/                      # PixelBuffer, PixelGraphics, ColorMath, BlendMode, Resample
├── transform/                  # FrameNormalizer, FitMode
├── exception/                  # ImageException hierarchy
└── codec/
    ├── ImageReader.java        # reader SPI + ImageReadOptions
    ├── ImageWriter.java        # writer SPI + ImageWriteOptions
    ├── bmp/                    # BmpImageReader, BmpImageWriter
    ├── gif/                    # GifImageReader, GifImageWriter, GifWriteOptions
    ├── jpeg/                   # JpegImageReader, JpegImageWriter, JpegWriteOptions
    ├── png/                    # PngImageReader, PngImageWriter, PngWriteOptions
    └── webp/
        ├── WebPImageReader.java
        ├── WebPImageWriter.java
        ├── WebPWriteOptions.java
        ├── RiffContainer.java  # RIFF parse/write
        ├── WebPChunk.java      # chunk model + Type enum for FourCC
        ├── FrameDiffUtil.java  # partial-frame ANMF bounding-box diffing
        ├── lossless/           # VP8LEncoder, VP8LDecoder + BitReader/Writer, HuffmanTree, ColorCache, LZ77, VP8LTransform, NearLosslessPreprocess
        └── lossy/              # VP8Encoder, VP8Decoder + BooleanEncoder/Decoder, DCT, IntraPrediction, Quantizer, TrellisQuantizer, Macroblock, VP8Tables, VP8TokenEncoder/Decoder, LoopFilter, RateDistortion, VP8Costs, ChromaUpsampler
```

| Package | Description |
|---|---|
| `dev.simplified.image` | Core data container (`ImageData`), auto-detect facade (`ImageFactory`), and format enum |
| `dev.simplified.image.data` | Frame-level types: `StaticImageData`, `AnimatedImageData`, `ImageFrame`, and disposal/blend enums |
| `dev.simplified.image.pixel` | Zero-copy `PixelBuffer`, `PixelGraphics`, color math, blend modes, and resampling |
| `dev.simplified.image.transform` | `FrameNormalizer` and animation fit modes |
| `dev.simplified.image.exception` | `ImageException` hierarchy for decode/encode/unsupported-format errors |
| `dev.simplified.image.codec.*` | Format-specific readers/writers plus their write-options builders |
| `dev.simplified.image.codec.webp.lossless` | VP8L codec internals (public entry: `VP8LEncoder` / `VP8LDecoder`) |
| `dev.simplified.image.codec.webp.lossy` | VP8 codec internals (public entry: `VP8Encoder` / `VP8Decoder`) |

> [!TIP]
> Each codec package follows the same pattern: a `*ImageReader` for decoding
> and a `*ImageWriter` for encoding. All readers and writers operate on the
> shared `ImageData` / `AnimatedImageData` types.

## Dependencies

| Dependency | Version | Scope |
|---|---|---|
| [Log4j2](https://logging.apache.org/log4j/) | 2.25.3 | API |
| [JetBrains Annotations](https://github.com/JetBrains/java-annotations) | 26.0.2 | API |
| [Lombok](https://projectlombok.org/) | 1.18.36 | Compile-only |
| [collections](https://github.com/Simplified-Dev/collections) | master-SNAPSHOT | API (Simplified-Dev) |
| [utils](https://github.com/Simplified-Dev/utils) | master-SNAPSHOT | API (Simplified-Dev) |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style
guidelines, and how to submit a pull request.

## License

This project is licensed under the **Apache License 2.0** - see
[LICENSE.md](LICENSE.md) for the full text.
