# Image

Pure-Java multi-format image codec library with full VP8/VP8L WebP encoding and decoding, animated image support, and parallel processing. Reads and writes BMP, GIF, JPEG, PNG, and WebP with per-frame timing, loop control, disposal and blend modes, frame normalization, and frame interpolation - no native dependencies or JNI bindings required.

> [!IMPORTANT]
> This library is under active development. APIs may change between releases
> until a stable `1.0.0` release is published.

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

- **Pure-Java VP8 and VP8L** - Full lossy (VP8) and lossless (VP8L) WebP encoding and decoding with no native libraries, including boolean arithmetic coding, Huffman + LZ77 compression, all 14 spatial prediction modes, DCT/WHT transforms, and rate-distortion mode selection
- **Five image formats** - Read and write BMP, GIF, JPEG, PNG, and WebP with format-specific options for quality, compression level, and lossless/lossy mode
- **Animated image support** - Multi-frame GIF and WebP with per-frame timing, loop count, background color, disposal methods (none, do not dispose, restore to background, restore to previous), and blend modes (source replace, alpha-over)
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
|-------------|---------|-------|
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
> This library depends on other Simplified-Dev modules (`collections`, `utils`,
> `reflection`) which are also resolved from JitPack automatically.

## Usage

### Auto-Detection

`ImageFactory` detects the format from magic bytes and routes to the correct codec:

```java
import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFactory;

ImageData image = ImageFactory.fromFile(path);
ImageFactory.toFile(image, outputPath, ImageFormat.WEBP);
```

### Reading an Image

```java
import dev.simplified.image.ImageData;
import dev.simplified.image.codec.png.PngImageReader;

PngImageReader reader = new PngImageReader();
ImageData image = reader.read(inputStream);
```

### Writing an Image

```java
import dev.simplified.image.codec.webp.WebPImageWriter;
import dev.simplified.image.codec.webp.WebPWriteOptions;

WebPImageWriter writer = new WebPImageWriter();

// Lossless (default)
writer.write(image, outputStream);

// Lossy with quality control
writer.write(image, outputStream, new WebPWriteOptions(
    false,  // lossy
    0.80f,  // quality
    0,      // loop count
    true,   // multithreaded
    true    // alpha compression
));
```

### Animated Images

```java
import dev.simplified.image.AnimatedImageData;
import dev.simplified.image.ImageFrame;
import dev.simplified.image.codec.gif.GifImageReader;
import dev.simplified.image.codec.webp.WebPImageWriter;

// Read animated GIF
GifImageReader reader = new GifImageReader();
AnimatedImageData animated = reader.readAnimated(inputStream);

for (ImageFrame frame : animated.getFrames()) {
    // Access per-frame timing, position, disposal, blend mode, and pixel data
}

// Time-based frame lookup with interpolation
ImageFrame frame = animated.getFrameAtTime(elapsedMs, true);

// Write as animated WebP (frames encoded in parallel)
WebPImageWriter writer = new WebPImageWriter();
writer.writeAnimated(animated, outputStream);
```

### Batch Processing

```java
// Decode multiple files in parallel using virtual threads
List<ImageData> images = ImageFactory.fromFiles(paths);

// Encode multiple images in parallel
ImageFactory.toFiles(images, outputPaths, ImageFormat.PNG);
```

### Pixel Manipulation

```java
import dev.simplified.image.PixelBuffer;

PixelBuffer buffer = image.getPixelBuffer();

// Direct zero-copy pixel access
int argb = buffer.getPixel(x, y);
buffer.setPixel(x, y, 0xFFFF0000); // solid red

// Apply FXAA anti-aliasing (parallel scanline processing)
buffer.applyFxaa();
```

## Supported Formats

| Format | Read | Write | Animated | Options |
|--------|------|-------|----------|---------|
| BMP | `BmpImageReader` | `BmpImageWriter` | No | None (lossless) |
| GIF | `GifImageReader` | `GifImageWriter` | Yes | Loop count, transparency, disposal |
| JPEG | `JpegImageReader` | `JpegImageWriter` | No | Quality (0.0-1.0, default 0.75) |
| PNG | `PngImageReader` | `PngImageWriter` | No | Compression level (0-9, default 6) |
| WebP | `WebPImageReader` | `WebPImageWriter` | Yes | Lossy/lossless, quality (0.0-1.0), multithreaded, alpha compression |

## Architecture

### Project Structure

```
src/main/java/dev/simplified/image/
├── AnimatedImageData.java
├── FrameNormalizer.java
├── ImageData.java
├── ImageFactory.java
├── ImageFormat.java
├── ImageFrame.java
├── PixelBuffer.java
└── codec/
    ├── bmp/
    │   ├── BmpImageReader.java
    │   └── BmpImageWriter.java
    ├── gif/
    │   ├── GifImageReader.java
    │   └── GifImageWriter.java
    ├── jpeg/
    │   ├── JpegImageReader.java
    │   └── JpegImageWriter.java
    ├── png/
    │   ├── PngImageReader.java
    │   └── PngImageWriter.java
    └── webp/
        ├── WebPImageReader.java
        ├── WebPImageWriter.java
        └── (VP8/VP8L codec internals)
```

| Package | Description |
|---------|-------------|
| `dev.simplified.image` | Core data structures, `ImageFactory` facade, `FrameNormalizer`, `PixelBuffer`, and format detection |
| `dev.simplified.image.codec.*` | Format-specific reader and writer implementations with per-format write options |

> [!TIP]
> Each codec package follows the same pattern: a `*ImageReader` for decoding
> and a `*ImageWriter` for encoding. All readers and writers operate on the
> shared `ImageData` / `AnimatedImageData` types.

## Dependencies

| Dependency | Version | Scope |
|------------|---------|-------|
| [Log4j2](https://logging.apache.org/log4j/) | 2.25.3 | API |
| [JetBrains Annotations](https://github.com/JetBrains/java-annotations) | 26.0.2 | API |
| [Lombok](https://projectlombok.org/) | 1.18.36 | Compile-only |
| [collections](https://github.com/Simplified-Dev/collections) | master-SNAPSHOT | API (Simplified-Dev) |
| [utils](https://github.com/Simplified-Dev/utils) | master-SNAPSHOT | API (Simplified-Dev) |
| [reflection](https://github.com/Simplified-Dev/reflection) | master-SNAPSHOT | API (Simplified-Dev) |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style
guidelines, and how to submit a pull request.

## License

This project is licensed under the **Apache License 2.0** - see
[LICENSE.md](LICENSE.md) for the full text.
