# Known open defects

Defects present in the current tree and not yet fixed, most severe first. Each entry names the cause, the affected code, what it costs today, and what would raise its severity.

## GIF without a NETSCAPE block reads as "loop forever"
`GifImageReader` starts from `LoopCount.Preset.INFINITE`, and `parseLoopCount` answers it again when no NETSCAPE2.0 application extension declares a count, but a GIF that carries none plays once. A GIF that plays once therefore decodes to an `AnimatedImageData` that claims infinite looping, and any re-encode carries that forward. HIGH once a caller depends on play-once fidelity.

- Affected: `src/main/java/dev/simplified/image/codec/gif/GifImageReader.java:64`, `src/main/java/dev/simplified/image/codec/gif/GifImageReader.java:106-111`, `src/main/java/dev/simplified/image/codec/gif/GifImageReader.java:145-162` (`parseLoopCount`)
- Severity: **MEDIUM**
- Type: **BUG**
- Status: **OPEN**

## VP8 encoder point-samples 4:2:0 chroma instead of averaging
`Macroblock.fromARGB` takes each 2x2 block's Cb/Cr from its top-left pixel rather than averaging the four pixels, so every lossy WebP encode aliases thin colour detail such as one-pixel lines and coloured text edges. The bitstream stays valid; the cost is fidelity.

- Affected: `src/main/java/dev/simplified/image/codec/webp/vp8/Macroblock.java:76-81` (`fromARGB`); callers `src/main/java/dev/simplified/image/codec/webp/vp8/VP8Encoder.java:1602`, `:2216`, `:3111`
- Severity: **MEDIUM**
- Type: **BUG**
- Status: **OPEN**

## GIF writer misreads frames of different sizes and drops frame offsets
`flattenAllFrames` packs every frame's pixels back to back, but the write loop reads frame i at `i * frameStride`, where `frameStride` is the largest frame's area, so an animation whose frames differ in size is written from the wrong slices and the last one overruns the array: frames of 4x4, 2x2 and 2x2 throw `ArrayIndexOutOfBoundsException: Index 32 out of bounds for length 24`. `configureFrameMetadata` writes only the GraphicControlExtension and never the ImageDescriptor's `imageLeftPosition` / `imageTopPosition`, so every frame is written at the origin. A GIF this library decodes - partial frames at offsets - therefore cannot be re-encoded: it throws, or, where its partial frames happen to share one size, it is written with every frame moved to the top-left corner. HIGH once a caller re-encodes decoded GIFs.

- Affected: `src/main/java/dev/simplified/image/codec/gif/GifImageWriter.java:117-143` (`write`, the `frameStride` read), `src/main/java/dev/simplified/image/codec/gif/GifImageWriter.java:179` (`flattenAllFrames`), `src/main/java/dev/simplified/image/codec/gif/GifImageWriter.java:589-607` (`configureFrameMetadata`)
- Severity: **MEDIUM**
- Type: **BUG**
- Status: **OPEN**

## GIF reader sizes the canvas from the first frame
`GifImageReader` never reads the logical screen descriptor and builds the `AnimatedImageData` with no width or height, so the canvas takes the first frame's size, and a GIF whose first frame does not span its screen loses whatever later frames draw outside that rectangle. HIGH once a consumer draws a decoded GIF's frames onto its canvas.

- Affected: `src/main/java/dev/simplified/image/codec/gif/GifImageReader.java:122-125`
- Severity: **MEDIUM**
- Type: **BUG**
- Status: **OPEN**

## WebP reader keeps each frame's disposal and blend on frames it has already composed
`readExtended` composites every ANMF frame onto the canvas - applying its blend and the prior frame's disposal - and emits the whole canvas at offset 0, but tags that full picture with the frame's original `disposal` and `blend`. A consumer that honours the flags composites each frame a second time: an `OVER` frame is blended over itself, and a `RESTORE_TO_BACKGROUND` frame clears a rectangle of a picture that is already whole. A composed frame carries no blend or disposal of its own to apply. HIGH once a consumer composites decoded WebPs by the flags.

- Affected: `src/main/java/dev/simplified/image/codec/webp/WebPImageReader.java:251` (`readExtended`)
- Severity: **MEDIUM**
- Type: **BUG**
- Status: **OPEN**

## No downscale averages: `blitScaled` decimates
`blitScaled` and `blitScaledFxaa` map each destination pixel to one source pixel (`x * sw / dw`, read from the column table `sourceColumns` builds) and offer no other filter, and `resize` offers `NEAREST`, a 2x2 `BILINEAR` with no prefilter, and AWT `BICUBIC`. The library's one box filter, `PixelMask.downsample`, is boolean. A supersampled render scaled down through either composite keeps one sample of each block and discards the rest, so the supersampling averages nothing, and a `BILINEAR` downscale beyond 2x aliases the same way. asset-renderer's `RasterPass` scales down through `blitScaledFxaa` when it anti-aliases and through `blitScaled` when it does not: FXAA smooths the hi-res pixels the scale-down samples, and the scale-down still keeps one sample per block - a decimation of a filtered source. What such a caller needs is an area-averaging downsample of premultiplied colour.

- Affected: `src/main/java/dev/simplified/image/PixelBuffer.java:714-787` (`blitScaled`), `src/main/java/dev/simplified/image/PixelBuffer.java:812-847` (`blitScaledFxaa`), `src/main/java/dev/simplified/image/PixelBuffer.java:1240-1247` (`sourceColumns`), `src/main/java/dev/simplified/image/PixelBuffer.java:383-427` (`resize`), `src/main/java/dev/simplified/image/pixel/PixelMask.java:90` (`downsample`)
- Severity: **MEDIUM**
- Type: **GAP**
- Status: **OPEN**

## ImageIO's GIF reader decodes interlaced frames shorter than 8 rows wrongly
`GifImageWriter` writes every frame interlaced, because it takes ImageIO's default image metadata, whose `interlaceFlag` is TRUE. The JDK GIF reader that `GifImageReader` decodes through reads such a frame wrongly when it is shorter than 8 rows. Measured on still GIFs of alternating red and blue rows: a 2x2 frame decodes 2 pixels wrong, 3x3 6, 4x2 4 and 4x4 8; 8x8 and 16x16 decode exactly. A hand LZW decode and Pillow read the same files correctly, so the files are sound, and a short GIF this library writes does not round-trip through its own reader. Writing non-interlaced avoids it and moves every GIF's bytes. HIGH once a caller round-trips frames under 8 rows.

- Affected: `src/main/java/dev/simplified/image/codec/gif/GifImageWriter.java:127`, `:148` (`getDefaultImageMetadata`), `src/main/java/dev/simplified/image/codec/gif/GifImageReader.java:67` (`reader.read`)
- Severity: **MEDIUM**
- Type: **BUG**
- Status: **OPEN**

## The WebP writer's partial-frame path drops an OVER frame's blend
When `usePartialFrames` holds - every lossless write, and every lossy write without P-frames - `WebPImageWriter` writes each ANMF chunk as do-not-dispose and do-not-blend over the region that changed. A frame tagged `FrameBlend.OVER` that is transparent where it relies on the frame beneath is therefore written as an overwrite. Measured on an animation whose frame 1 is `OVER`: written as lossless WebP (the default) the relying pixel reads `00000000`, where GIF shows `ffff0000` and lossy WebP with P-frames shows `ffff0005`. A decoded GIF, whose frames read as `OVER`, reaches it when re-encoded as lossless WebP; nothing in the workspace does that today. HIGH once a caller converts GIFs to WebP.

- Affected: `src/main/java/dev/simplified/image/codec/webp/WebPImageWriter.java:166` (`usePartialFrames`), `:283-284`
- Severity: **MEDIUM**
- Type: **BUG**
- Status: **OPEN**

## WebP write options cannot force infinite looping over a finite count
`WebPImageWriter` reads a loop count in `WebPWriteOptions` that repeats forever as no count at all, and falls back to the image's own loop count. `withLoopCount(LoopCount.Preset.INFINITE)` is therefore ignored whenever the `AnimatedImageData` carries a finite count.

- Affected: `src/main/java/dev/simplified/image/codec/webp/WebPImageWriter.java:77-78`
- Severity: **LOW**
- Type: **BUG**
- Status: **OPEN**

## GIF writer ignores the image's loop count whenever options are passed
When `GifWriteOptions` are supplied, `GifImageWriter` takes the loop count only from the options, and options that set none declare `LoopCount.Preset.INFINITE`. A finite-loop GIF re-encoded with options, for example to set transparency, loops forever. The WebP writer declares the image's count when its options set none.

- Affected: `src/main/java/dev/simplified/image/codec/gif/GifImageWriter.java:80-89`
- Severity: **LOW**
- Type: **BUG**
- Status: **OPEN**

## GIF writer floors the flatten of a straight image
`flattenAllFrames` lays a partly transparent straight pixel over the background as `(c * a + bg * (255 - a)) / 255`, an integer division, so every such pixel lands up to a level low: at worst 0.996 of a level from the exact composite over 455,168 cases (backgrounds 0, 1, 16, 127, 128, 198 and 255, every partial alpha, every straight level), where the premultiplied flatten beside it, rounded once, lands within 0.498. A straight image and its premultiplied twin therefore flatten up to a level apart. Rounding it moves the bytes of every GIF written from a straight image with partial alpha. MEDIUM once a caller holds GIF output to an exact composite.

- Affected: `src/main/java/dev/simplified/image/codec/gif/GifImageWriter.java:215-217` (`flattenAllFrames`)
- Severity: **LOW**
- Type: **BUG**
- Status: **OPEN**

## `Background.composite` lays every image down as premultiplied colour, whatever it holds
`composite` blits each frame onto the fill with `BlendMode.PREMULTIPLIED`, on the assumption that a finished render's colours already carry their alpha, and premultiplies the fill only under an image declared `PREMULTIPLIED`. A straight image - anything decoded - is therefore laid down with its colour unscaled and comes out too bright wherever it is partly transparent. `getAlphaMode` would let it blend a straight image with `NORMAL`, but every caller in the workspace passes asset-renderer renders, which hold premultiplied colour under the default `STRAIGHT` until that consumer declares them, so honouring the label now would move their bytes. MEDIUM once a caller composites a decoded image.

- Affected: `src/main/java/dev/simplified/image/pixel/Background.java:89-104` (`composite`), `src/main/java/dev/simplified/image/pixel/Background.java:115-124` (`over`)
- Severity: **LOW**
- Type: **RISK**
- Status: **OPEN**

## `Background.composite` drops an animation's canvas size
The `AnimatedImageData` it rebuilds never calls `withWidth` or `withHeight`, so its canvas becomes the first frame's size and an animation whose canvas is larger than its first frame loses the rest. Nothing in the workspace builds such an animation and composites it: the GIF reader and `FrameNormalizer` set no canvas size, and the WebP reader sets one but composes every frame to the whole canvas, so its first frame is canvas-sized. MEDIUM once a caller composites an animation whose first frame is smaller than its canvas.

- Affected: `src/main/java/dev/simplified/image/pixel/Background.java:97-103` (`composite`), `src/main/java/dev/simplified/image/AnimatedImageData.java:216-217` (`Builder.build`)
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

## No disposal/blend-aware timeline renderer
`FrameNormalizer` resizes and positions frames but passes disposal and blend through unchanged, and nothing composites a GIF/WebP timeline (offsets, `OVER` blending, `RESTORE_*` disposal) into full canvases. A target format that only carries full-canvas frames cannot be written from such input without one.

- Affected: `src/main/java/dev/simplified/image/frame/FrameNormalizer.java:31-137`
- Severity: **LOW**
- Type: **GAP**
- Status: **OPEN**

## NOTICE names no licence for the RFC 6386 code `BooleanEncoder` ports
`NOTICE` carries libwebp's and libvpx's copyright notices and BSD licences for the classes ported from each, and the jar ships it under `META-INF`. `BooleanEncoder` ports the boolean entropy encoder section 7.3 of RFC 6386 sets out, and for that `NOTICE` records only the RFC's copyright line and the IETF Trust provisions it is subject to. RFC 6386 is an Independent Submission; its section 20.26 licenses the reference decoder source it attaches under Google's BSD terms, and nothing in this repo says whether those terms, the Trust's code-component licence, or neither reaches the encoder fragment in the RFC's body (UNVERIFIED either way). Until that is settled `NOTICE` may owe a licence text for the port. MEDIUM once a redistributor needs every licence named.

- Affected: `NOTICE`, `src/main/java/dev/simplified/image/codec/webp/vp8/BooleanEncoder.java`
- Severity: **LOW**
- Type: **RISK**
- Status: **OPEN**

## `resize` takes no alpha convention, and its two smoothing filters assume opposite ones
`BILINEAR` mixes every channel as it is held, which is correct for premultiplied colour and darkens a straight pixel beside a transparent one toward the transparent pixel's colour. `BICUBIC` draws through AWT with the raster declared straight, which is correct for straight colour and weights a premultiplied buffer by alpha twice. A caller cannot say which it holds, and switching filter switches the convention assumed. Nothing in the workspace resizes with either today. `resize(w, h, Resample, AlphaMode)` would route the bilinear sampler's alpha weighting and declare the AWT raster premultiplied where asked. `FrameNormalizer` scales frames through the same AWT bicubic from a straight-declared image, so it weights a premultiplied animation's colour by alpha twice, and fills its letterbox with a straight `Color`; it carries the animation's convention onto its output without honouring it. Nothing in the workspace normalizes premultiplied frames.

- Affected: `src/main/java/dev/simplified/image/PixelBuffer.java:383-427` (`resize`), `src/main/java/dev/simplified/image/frame/FrameNormalizer.java:86-137` (`normalize`)
- Severity: **LOW**
- Type: **GAP**
- Status: **OPEN**

## FXAA's absolute thresholds are 0-to-1 luma constants read against a 0-to-255 luma
`fxaaPixel`, the per-pixel body `applyFxaa` and `blitScaledFxaa` share, skips a neighbourhood whose contrast is under `max(0.0156, lumaMax * 0.0312)` and floors its direction reduction at `1/128`. Those are FXAA's values for luma in `[0, 1]`, but the luma here is in `[0, 255]`, so both floors are effectively inert: the relative threshold decides every case above near-black, and the direction floor never binds there. Not checked against a reference shader; changing either moves every FXAA output.

- Affected: `src/main/java/dev/simplified/image/PixelBuffer.java:982-983`, `src/main/java/dev/simplified/image/PixelBuffer.java:1007`, `src/main/java/dev/simplified/image/PixelBuffer.java:1013-1016` (`fxaaPixel`)
- Severity: **LOW**
- Type: **RISK**
- Status: **OPEN**

## `resize(BICUBIC)` always reports `hasAlpha = true`
`NEAREST` and `BILINEAR` keep the source's `hasAlpha`; `BICUBIC` wraps a `TYPE_INT_ARGB` image, so an opaque buffer comes back flagged as carrying alpha and loses `blit`'s opaque fast path.

- Affected: `src/main/java/dev/simplified/image/PixelBuffer.java:389`, `:400` (`resize`)
- Severity: **LOW**
- Type: **BUG**
- Status: **OPEN**

## `PixelGraphics.drawImage`'s rectangle and transform forms overwrite instead of compositing
The source/destination-rectangle form and the `AffineTransform` form write every pixel with alpha above zero through `setPixelSafe`: a partly transparent pixel replaces the destination instead of blending over it, a flipped rectangle loses its flip to the min corners, and the clip is ignored. The `(img, x, y)` and `(img, x, y, w, h)` forms blend with `BlendMode.NORMAL`. No workspace consumer calls the two forms today.

- Affected: `src/main/java/dev/simplified/image/pixel/PixelGraphics.java:695-719`, `src/main/java/dev/simplified/image/pixel/PixelGraphics.java:727-753`
- Severity: **LOW**
- Type: **BUG**
- Status: **OPEN**
