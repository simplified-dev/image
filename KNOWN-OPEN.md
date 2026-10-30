# Known open defects

Defects present in the current tree and not yet fixed, most severe first. Each entry names the cause, the affected code, what it costs today, and what would raise its severity.

## `create` answers straight colour under a name that does not say so
Every `PixelBuffer` holds its pixels in one `AlphaMode`, fixed by the factory that makes it, and every operation reads it from there: a composite lays its source in the destination's convention, a colour argument is straight colour laid in the buffer's, and `wrap` reads the convention off the image. Every blank factory names its convention but one. `canvas` answers a transparent premultiplied buffer, what drawing source-over onto transparency leaves; `create(int, int)` answers a transparent straight one under a name that says nothing, because minecraft-text's compiled glyph code calls it for its glyph bitmaps, which are straight. A render drawn into a `create` buffer by a raw writer - `setPixel` of `ColorMath.blend(.., BlendMode.NORMAL)`, as minecraft-text lays its text, or asset-renderer's rasterizer - holds premultiplied colour the buffer calls straight: `getAlphaMode()` answers straight, a composite lays it by the straight formulas, `resize` and FXAA weigh it by its alpha again, and the writers encode each see-through pixel too dark, its colour scaled by its alpha. A buffer drawn into that way is made with `canvas`. It closes when minecraft-text next re-pins image and `create` gives way to a straight blank factory whose name says so. MEDIUM until a consumer re-pins image; HIGH if one leaves a render on `create`.

- Affected: `src/main/java/dev/simplified/image/PixelBuffer.java` (`create`)
- Severity: **MEDIUM**
- Type: **RISK**
- Status: **OPEN**

## No ICC / Exif / XMP metadata model; WebP ICC profiles are dropped
`WebPChunk.Type` names `ICCP`, `EXIF` and `XMP `, but no reader or writer carries them and `ImageData` has no metadata field, so colour profiles and Exif are lost on every WebP round trip. MEDIUM once a wide-gamut or HDR format depends on its colour metadata.

- Affected: `src/main/java/dev/simplified/image/codec/webp/WebPChunk.java:39-54`, `src/main/java/dev/simplified/image/codec/webp/WebPImageReader.java`, `src/main/java/dev/simplified/image/codec/webp/WebPImageWriter.java`, `src/main/java/dev/simplified/image/ImageData.java`
- Severity: **LOW**
- Type: **GAP**
- Status: **OPEN**
