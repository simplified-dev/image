# image

Multi-format image codec library with animated image support (BMP, GIF, JPEG, PNG, WebP).

## Package Structure
- `dev.simplified.image` - core data structures (ImageData, ImageFrame, AnimatedImageData, PixelBuffer)
- `dev.simplified.image.codec.bmp` - BMP reader/writer
- `dev.simplified.image.codec.gif` - GIF reader/writer
- `dev.simplified.image.codec.jpeg` - JPEG reader/writer
- `dev.simplified.image.codec.png` - PNG reader/writer
- `dev.simplified.image.codec.webp` - WebP reader/writer

## Key Classes
- `ImageData` - single image container
- `AnimatedImageData` - multi-frame animated image with loop control
- `ImageFrame` - individual frame with timing/position
- `PixelBuffer` - raw pixel manipulation abstraction
- `*ImageReader` / `*ImageWriter` - format-specific codecs

## Dependencies
- Internal: `collections`, `utils`, `reflection` (Simplified-Dev)
- External: Log4j2, Lombok, JetBrains annotations
- Test: JUnit 5, Hamcrest

## Build
```bash
./gradlew build
./gradlew test
```

## Info
- Java 21
- Group: `dev.simplified`, artifact: `image`, version: `1.0.0`
- 52 source classes, 2 test classes
