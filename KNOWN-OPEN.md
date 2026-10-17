# Known open defects

Defects present in the current tree and not yet fixed, most severe first. Each entry names the cause, the affected code, what it costs today, and what would raise its severity.

## WebP writer refuses an animation whose frames do not cover its canvas from the origin
`writeAnimated` encodes every frame as though its pixels spanned the canvas from the origin - frame 0's ANMF is declared canvas-sized and each later frame is diffed pixel for pixel against the one before - so `requireCanvasFrames` refuses with `ImageEncodeException` any animation holding a frame of another size or at an offset. A decoded GIF whose first frame is smaller than its logical screen, or whose later frames are partial, is such an animation, and cannot be written as WebP at all: two 4x4 frames on an 8x8 screen, the second at (4, 4), are refused at frame 0, and a 4x4 frame followed by a 2x2 one at (1, 1) at frame 1. An ANMF chunk carries its own offset, at even coordinates, and its own blend and disposal flags, so the container can hold such frames as they are; GIF's `RESTORE_TO_PREVIOUS` disposal has no WebP flag. `ImageData#toCanvasFrames` composes such an animation into whole-canvas frames the writer accepts, but the writer does not call it, so a caller must. Nothing in the workspace re-encodes a decoded GIF as WebP today. HIGH once a caller converts GIFs to WebP.

- Affected: `src/main/java/dev/simplified/image/codec/webp/WebPImageWriter.java:145` (`writeAnimated`), `:309-320` (`requireCanvasFrames`)
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

- Affected: `src/main/java/dev/simplified/image/PixelBuffer.java:1146-1147`, `src/main/java/dev/simplified/image/PixelBuffer.java:1171`, `src/main/java/dev/simplified/image/PixelBuffer.java:1177-1180` (`fxaaPixel`)
- Severity: **LOW**
- Type: **RISK**
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
