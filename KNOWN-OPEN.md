# Known open defects

Defects present in the current tree and not yet fixed, most severe first. Each entry names the cause, the affected code, what it costs today, and what would raise its severity.

## VP8 encoder point-samples 4:2:0 chroma instead of averaging
`Macroblock.fromARGB` takes each 2x2 block's Cb/Cr from its top-left pixel rather than averaging the four pixels, so every lossy WebP encode aliases thin colour detail such as one-pixel lines and coloured text edges. The bitstream stays valid; the cost is fidelity.

- Affected: `src/main/java/dev/simplified/image/codec/webp/vp8/Macroblock.java:53-58` (`fromARGB`); callers `src/main/java/dev/simplified/image/codec/webp/vp8/VP8Encoder.java:1594`, `:2208`, `:3105`
- Severity: **MEDIUM**
- Type: **BUG**
- Status: **OPEN**

## GIF writer misreads frames of different sizes and drops frame offsets
`flattenAllFrames` packs every frame's pixels back to back, but the write loop reads frame i at `i * frameStride`, where `frameStride` is the largest frame's area, so an animation whose frames differ in size is written from the wrong slices and the last one overruns the array: frames of 4x4, 2x2 and 2x2 throw `ArrayIndexOutOfBoundsException: Index 32 out of bounds for length 24`. `configureFrameMetadata` writes only the GraphicControlExtension and never the ImageDescriptor's `imageLeftPosition` / `imageTopPosition`, so every frame is written at the origin. A GIF this library decodes - partial frames at offsets - therefore cannot be re-encoded: it throws, or, where its partial frames happen to share one size, it is written with every frame moved to the top-left corner. The writer hands ImageIO no stream metadata either, so the Logical Screen Descriptor declares the first frame's size rather than the image's canvas: two 8x8 frames on a 16x16 canvas are written with an 8x8 screen and read back 8x8, and a decoded GIF whose screen is larger than its first frame is written with the smaller one. HIGH once a caller re-encodes decoded GIFs.

- Affected: `src/main/java/dev/simplified/image/codec/gif/GifImageWriter.java:117-143` (`write`, the `frameStride` read, and `prepareWriteSequence(null)` at `:119`), `src/main/java/dev/simplified/image/codec/gif/GifImageWriter.java:179` (`flattenAllFrames`), `src/main/java/dev/simplified/image/codec/gif/GifImageWriter.java:589-607` (`configureFrameMetadata`)
- Severity: **MEDIUM**
- Type: **BUG**
- Status: **OPEN**

## WebP writer refuses an animation whose frames do not cover its canvas from the origin
`writeAnimated` encodes every frame as though its pixels spanned the canvas from the origin - frame 0's ANMF is declared canvas-sized and each later frame is diffed pixel for pixel against the one before - so `requireCanvasFrames` refuses with `ImageEncodeException` any animation holding a frame of another size or at an offset. A decoded GIF whose first frame is smaller than its logical screen, or whose later frames are partial, is such an animation, and cannot be written as WebP at all: two 4x4 frames on an 8x8 screen, the second at (4, 4), are refused at frame 0, and a 4x4 frame followed by a 2x2 one at (1, 1) at frame 1. An ANMF chunk carries its own offset, at even coordinates, and its own blend and disposal flags, so the container can hold such frames as they are; GIF's `RESTORE_TO_PREVIOUS` disposal has no WebP flag. `ImageData#toCanvasFrames` composes such an animation into whole-canvas frames the writer accepts, but the writer does not call it, so a caller must. Nothing in the workspace re-encodes a decoded GIF as WebP today. HIGH once a caller converts GIFs to WebP.

- Affected: `src/main/java/dev/simplified/image/codec/webp/WebPImageWriter.java:145` (`writeAnimated`), `:309-320` (`requireCanvasFrames`)
- Severity: **MEDIUM**
- Type: **GAP**
- Status: **OPEN**

## No downscale averages: `blitScaled` decimates
`blitScaled` and `blitScaledFxaa` map each destination pixel to one source pixel (`x * sw / dw`, read from the column table `sourceColumns` builds) and offer no other filter, and `resize` offers `NEAREST`, a 2x2 `BILINEAR` with no prefilter, and AWT `BICUBIC`. The library's one box filter, `PixelMask.downsample`, is boolean. A supersampled render scaled down through either composite keeps one sample of each block and discards the rest, so the supersampling averages nothing, and a `BILINEAR` downscale beyond 2x aliases the same way. asset-renderer's `RasterPass` scales down through `blitScaledFxaa` when it anti-aliases and through `blitScaled` when it does not: FXAA smooths the hi-res pixels the scale-down samples, and the scale-down still keeps one sample per block - a decimation of a filtered source. What such a caller needs is an area-averaging downsample of premultiplied colour.

- Affected: `src/main/java/dev/simplified/image/PixelBuffer.java:755-828` (`blitScaled`), `src/main/java/dev/simplified/image/PixelBuffer.java:853-888` (`blitScaledFxaa`), `src/main/java/dev/simplified/image/PixelBuffer.java:1281-1288` (`sourceColumns`), `src/main/java/dev/simplified/image/PixelBuffer.java:379-468` (`resize`, `resizeBicubic`), `src/main/java/dev/simplified/image/pixel/PixelMask.java:90` (`downsample`)
- Severity: **MEDIUM**
- Type: **GAP**
- Status: **OPEN**

## The WebP writer's partial-frame path drops an OVER frame's blend
When `usePartialFrames` holds - every lossless write, and every lossy write without P-frames - `WebPImageWriter` writes each ANMF chunk as do-not-dispose and do-not-blend over the region that changed. A frame tagged `FrameBlend.OVER` that is transparent where it relies on the frame beneath is therefore written as an overwrite. Measured on an animation whose frame 1 is `OVER`: written as lossless WebP (the default) the relying pixel reads `00000000`, where GIF shows `ffff0000` and lossy WebP with P-frames shows `ffff0005`. A decoded GIF, whose frames read as `OVER`, reaches it when re-encoded as lossless WebP; nothing in the workspace does that today. HIGH once a caller converts GIFs to WebP.

- Affected: `src/main/java/dev/simplified/image/codec/webp/WebPImageWriter.java:169` (`usePartialFrames`), `:286-287`
- Severity: **MEDIUM**
- Type: **BUG**
- Status: **OPEN**

## No reader bounds the pixels a header asks for
`GifImageReader` refuses a canvas longer than 16383 on a side, as `WebPImageReader` refuses a VP8X canvas, and the PNM, QOI and TGA readers throw `ArithmeticException` on a pixel count that overflows an `int`, but no reader bounds how many pixels it allocates, so a few header bytes can ask for gigabytes. Measured on a 512 MB heap: a 41-byte still GIF whose logical screen is 16383x16383, over one 1x1 frame, is laid on a canvas of 268,402,689 pixels, 1 GiB of `int`, and throws `OutOfMemoryError`, and so does the 19-byte PNM header `P6 40000 40000 255`. An animated GIF declares such a canvas without allocating it, leaving the allocation to whatever composites it. A pixel budget each reader checks before it allocates, as Pillow's `MAX_IMAGE_PIXELS` is, would bound it. asset-renderer decodes pack textures and skins by format detection, so a crafted file in a pack it is handed reaches a reader. HIGH once a service renders packs or skins it does not vet.

- Affected: `src/main/java/dev/simplified/image/codec/gif/GifImageReader.java:190-191` (`read`), `:248` (`onCanvas`), `src/main/java/dev/simplified/image/codec/pnm/PnmImageReader.java:64`, `src/main/java/dev/simplified/image/codec/qoi/QoiImageReader.java:60`, `src/main/java/dev/simplified/image/codec/tga/TgaImageReader.java:69`, `src/main/java/dev/simplified/image/codec/webp/WebPImageReader.java:88-89`
- Severity: **MEDIUM**
- Type: **RISK**
- Status: **OPEN**

## GIF writer floors the flatten of a straight image
`flattenAllFrames` lays a partly transparent straight pixel over the background as `(c * a + bg * (255 - a)) / 255`, an integer division, so every such pixel lands up to a level low: at worst 0.996 of a level from the exact composite over 455,168 cases (backgrounds 0, 1, 16, 127, 128, 198 and 255, every partial alpha, every straight level), where the premultiplied flatten beside it, rounded once, lands within 0.498. A straight image and its premultiplied twin therefore flatten up to a level apart. Rounding it moves the bytes of every GIF written from a straight image with partial alpha. MEDIUM once a caller holds GIF output to an exact composite.

- Affected: `src/main/java/dev/simplified/image/codec/gif/GifImageWriter.java:215-217` (`flattenAllFrames`)
- Severity: **LOW**
- Type: **BUG**
- Status: **OPEN**

## ICO magic collides with a 256-byte ISOBMFF `ftyp` box
The ICO arm of `ImageFormat.matches` accepts `00 00 01 00`, which is also the big-endian size field of a 256-byte `ftyp` box, and `ImageFactory` lists the ICO reader ahead of any format added later, so `detectFormat` tries ICO before it. No ISOBMFF-based format ships today; MEDIUM once one (AVIF, HEIF) is added after ICO.

- Affected: `src/main/java/dev/simplified/image/ImageFormat.java:81-85`, `src/main/java/dev/simplified/image/ImageFactory.java:77`
- Severity: **LOW**
- Type: **RISK**
- Status: **OPEN**

## No ICC / Exif / XMP metadata model; WebP ICC profiles are dropped
`WebPChunk.Type` names `ICCP`, `EXIF` and `XMP `, but no reader or writer carries them and `ImageData` has no metadata field, so colour profiles and Exif are lost on every WebP round trip. MEDIUM once a wide-gamut or HDR format depends on its colour metadata.

- Affected: `src/main/java/dev/simplified/image/codec/webp/WebPChunk.java:39-54`, `src/main/java/dev/simplified/image/codec/webp/WebPImageReader.java`, `src/main/java/dev/simplified/image/codec/webp/WebPImageWriter.java`, `src/main/java/dev/simplified/image/ImageData.java`
- Severity: **LOW**
- Type: **GAP**
- Status: **OPEN**

## NOTICE names no licence for the RFC 6386 code `BooleanEncoder` ports
`NOTICE` carries libwebp's and libvpx's copyright notices and BSD licences for the classes ported from each, and the jar ships it under `META-INF`. `BooleanEncoder` ports the boolean entropy encoder section 7.3 of RFC 6386 sets out, and for that `NOTICE` records only the RFC's copyright line and the IETF Trust provisions it is subject to. RFC 6386 is an Independent Submission; its section 20.26 licenses the reference decoder source it attaches under Google's BSD terms, and nothing in this repo says whether those terms, the Trust's code-component licence, or neither reaches the encoder fragment in the RFC's body (UNVERIFIED either way). Until that is settled `NOTICE` may owe a licence text for the port. MEDIUM once a redistributor needs every licence named.

- Affected: `NOTICE`, `src/main/java/dev/simplified/image/codec/webp/vp8/BooleanEncoder.java`
- Severity: **LOW**
- Type: **RISK**
- Status: **OPEN**

## FXAA's absolute thresholds are 0-to-1 luma constants read against a 0-to-255 luma
`fxaaPixel`, the per-pixel body `applyFxaa` and `blitScaledFxaa` share, skips a neighbourhood whose contrast is under `max(0.0156, lumaMax * 0.0312)` and floors its direction reduction at `1/128`. Those are FXAA's values for luma in `[0, 1]`, but the luma here is in `[0, 255]`, so both floors are effectively inert: the relative threshold decides every case above near-black, and the direction floor never binds there. Not checked against a reference shader; changing either moves every FXAA output.

- Affected: `src/main/java/dev/simplified/image/PixelBuffer.java:1023-1024`, `src/main/java/dev/simplified/image/PixelBuffer.java:1048`, `src/main/java/dev/simplified/image/PixelBuffer.java:1054-1057` (`fxaaPixel`)
- Severity: **LOW**
- Type: **RISK**
- Status: **OPEN**

## `PixelGraphics` shape drawing ignores the clip
`setClip`, `clipRect` and `clip` record a rectangle that only the `drawImage` forms read. `fillRect`, `drawRect`, the round-rect, oval, arc, line and polygon methods write through `setPixelSafe`, which bounds a pixel to the target and not to the clip, so a shape drawn after a clip is set lands outside it. No workspace consumer sets a clip on a `PixelGraphics` today.

- Affected: `src/main/java/dev/simplified/image/pixel/PixelGraphics.java:124-127` (`setPixelSafe`), `:257-305`, `:385-578`
- Severity: **LOW**
- Type: **GAP**
- Status: **OPEN**

## `VP8Encoder` holds libwebp lambdas no scorer reads
`State.lambdaI16`, `lambdaMode` and `tlambda`, and their per-segment arrays `segLambdaI16`, `segLambdaMode` and `segTlambda`, are assigned and never read. Their comments say so. Deleting them, or wiring them into the scorer as libwebp does, would end the dead state.

- Affected: `src/main/java/dev/simplified/image/codec/webp/vp8/VP8Encoder.java:184-207` (`State`), `:235-237`, `:470`
- Severity: **LOW**
- Type: **GAP**
- Status: **OPEN**

## The project's own copyright holder is named nowhere
`LICENSE.md` is the bare Apache-2.0 text with no copyright line, so `NOTICE` opens with the project name and licence only.

- Affected: `LICENSE.md`, `NOTICE:1-2`
- Severity: **LOW**
- Type: **GAP**
- Status: **OPEN**
