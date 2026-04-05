# Image

Multi-format image codec library with animated image support. Provides pure-Java
readers and writers for BMP, GIF, JPEG, PNG, and WebP formats with per-frame
timing, loop control, background colors, and frame interpolation.

> [!IMPORTANT]
> This library is under active development. APIs may change between releases
> until a stable `1.0.0` release is published.

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Reading an Image](#reading-an-image)
  - [Writing an Image](#writing-an-image)
  - [Animated Images](#animated-images)
  - [Pixel Manipulation](#pixel-manipulation)
- [Supported Formats](#supported-formats)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Five image formats** - Read and write BMP, GIF, JPEG, PNG, and WebP
- **Animated image support** - Multi-frame images with per-frame timing, loop control, and background colors
- **PixelBuffer abstraction** - Direct raw pixel manipulation independent of format
- **Frame interpolation** - Smooth transitions between animation frames
- **Pure Java** - No native dependencies or JNI bindings required
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

### Reading an Image

```java
import dev.simplified.image.ImageData;
import dev.simplified.image.codec.png.PngImageReader;

PngImageReader reader = new PngImageReader();
ImageData image = reader.read(inputStream);
```

### Writing an Image

```java
import dev.simplified.image.codec.png.PngImageWriter;

PngImageWriter writer = new PngImageWriter();
writer.write(image, outputStream);
```

### Animated Images

```java
import dev.simplified.image.AnimatedImageData;
import dev.simplified.image.ImageFrame;
import dev.simplified.image.codec.gif.GifImageReader;

GifImageReader reader = new GifImageReader();
AnimatedImageData animated = reader.readAnimated(inputStream);

for (ImageFrame frame : animated.getFrames()) {
    // Access per-frame timing, position, and pixel data
}
```

### Pixel Manipulation

```java
import dev.simplified.image.PixelBuffer;

PixelBuffer buffer = image.getPixelBuffer();
// Direct pixel-level read/write access
```

## Supported Formats

| Format | Read | Write | Animated |
|--------|------|-------|----------|
| BMP | `BmpImageReader` | `BmpImageWriter` | No |
| GIF | `GifImageReader` | `GifImageWriter` | Yes |
| JPEG | `JpegImageReader` | `JpegImageWriter` | No |
| PNG | `PngImageReader` | `PngImageWriter` | No |
| WebP | `WebPImageReader` | `WebPImageWriter` | Yes |

## Project Structure

```
src/main/java/dev/simplified/image/
├── AnimatedImageData.java
├── ImageData.java
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
        └── WebPImageWriter.java
```

| Directory | Description |
|-----------|-------------|
| `dev.simplified.image` | Core data structures - ImageData, ImageFrame, AnimatedImageData, PixelBuffer |
| `dev.simplified.image.codec.*` | Format-specific reader and writer implementations |

> [!TIP]
> Each codec package follows the same pattern: a `*ImageReader` for decoding
> and a `*ImageWriter` for encoding. All readers and writers operate on the
> shared `ImageData` / `AnimatedImageData` types.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style
guidelines, and how to submit a pull request.

## License

This project is licensed under the **Apache License 2.0** - see
[LICENSE.md](LICENSE.md) for the full text.
