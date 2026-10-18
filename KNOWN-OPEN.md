# Known open defects

Defects present in the current tree and not yet fixed, most severe first. Each entry names the cause, the affected code, what it costs today, and what would raise its severity.

## No reader bounds the pixels a header asks for
`GifImageReader` refuses a canvas longer than 16383 on a side, as `WebPImageReader` refuses a VP8X canvas, and the PNM, QOI and TGA readers throw `ArithmeticException` on a pixel count that overflows an `int`, but no reader bounds how many pixels it allocates, so a few header bytes can ask for gigabytes. Measured on a 512 MB heap: a 41-byte still GIF whose logical screen is 16383x16383, over one 1x1 frame, is laid on a canvas of 268,402,689 pixels, 1 GiB of `int`, and throws `OutOfMemoryError`, and so does the 19-byte PNM header `P6 40000 40000 255`. An animated GIF declares such a canvas without allocating it, leaving the allocation to whatever composites it. A pixel budget each reader checks before it allocates, as Pillow's `MAX_IMAGE_PIXELS` is, would bound it. asset-renderer decodes pack textures and skins by format detection, so a crafted file in a pack it is handed reaches a reader. HIGH once a service renders packs or skins it does not vet.

- Affected: `src/main/java/dev/simplified/image/codec/gif/GifImageReader.java:190-191` (`read`), `:248` (`onCanvas`), `src/main/java/dev/simplified/image/codec/pnm/PnmImageReader.java:64`, `src/main/java/dev/simplified/image/codec/qoi/QoiImageReader.java:60`, `src/main/java/dev/simplified/image/codec/tga/TgaImageReader.java:69`, `src/main/java/dev/simplified/image/codec/webp/WebPImageReader.java:88-89`
- Severity: **MEDIUM**
- Type: **RISK**
- Status: **OPEN**

## No ICC / Exif / XMP metadata model; WebP ICC profiles are dropped
`WebPChunk.Type` names `ICCP`, `EXIF` and `XMP `, but no reader or writer carries them and `ImageData` has no metadata field, so colour profiles and Exif are lost on every WebP round trip. MEDIUM once a wide-gamut or HDR format depends on its colour metadata.

- Affected: `src/main/java/dev/simplified/image/codec/webp/WebPChunk.java:39-54`, `src/main/java/dev/simplified/image/codec/webp/WebPImageReader.java`, `src/main/java/dev/simplified/image/codec/webp/WebPImageWriter.java`, `src/main/java/dev/simplified/image/ImageData.java`
- Severity: **LOW**
- Type: **GAP**
- Status: **OPEN**
