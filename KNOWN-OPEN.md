# Known open defects

Defects present in the current tree and not yet fixed, most severe first. Each entry names the cause, the affected code, what it costs today, and what would raise its severity.

## GIF without a NETSCAPE block reads as "loop forever"
`GifImageReader.parseLoopCount` returns 0 when the file carries no NETSCAPE2.0 application extension, and `AnimatedImageData` defines a loop count of 0 as infinite. A GIF that plays once therefore decodes to an `AnimatedImageData` that claims infinite looping, and any re-encode carries that forward. HIGH once a caller depends on play-once fidelity.

- Affected: `src/main/java/dev/simplified/image/codec/gif/GifImageReader.java:99-104`, `src/main/java/dev/simplified/image/codec/gif/GifImageReader.java:130-145` (`parseLoopCount`), `src/main/java/dev/simplified/image/data/AnimatedImageData.java:175-180` (`Builder.withLoopCount`)
- Severity: **MEDIUM**
- Type: **BUG**
- Status: **OPEN**

## VP8 encoder point-samples 4:2:0 chroma instead of averaging
`Macroblock.fromARGB` takes each 2x2 block's Cb/Cr from its top-left pixel rather than averaging the four pixels, so every lossy WebP encode aliases thin colour detail such as one-pixel lines and coloured text edges. The bitstream stays valid; the cost is fidelity.

- Affected: `src/main/java/dev/simplified/image/codec/webp/lossy/Macroblock.java:76-81` (`fromARGB`); callers `src/main/java/dev/simplified/image/codec/webp/lossy/VP8Encoder.java:1596`, `:2210`, `:3105`
- Severity: **MEDIUM**
- Type: **BUG**
- Status: **OPEN**

## CLAUDE.md "VP8 encoder state" describes an outdated codec
Line 20 says the encoder is keyframe-only 16x16 intra with no B_PRED and no inter frames, but it ships B_PRED, P-frames with golden/altref references, SPLITMV, R-D mode selection, trellis quantization and segmentation. It also cites libwebp `src/enc/tree_enc.c` for tables that `VP8Tables` takes from `src/dec/tree_dec.c`. Line 22 says the decoder still carries fixed-width 11-bit coefficient shortcuts pending a spec-compliant rewrite; that rewrite has landed. Anyone planning or sizing work against this paragraph undercounts the VP8 codec.

- Affected: `CLAUDE.md:20`, `CLAUDE.md:22`
- Severity: **MEDIUM**
- Type: **GAP**
- Status: **OPEN**

## WebP write options cannot force infinite looping over a finite count
`WebPImageWriter` reads a loop count of 0 in `WebPWriteOptions` as "unset" and falls back to the image's own loop count. `withLoopCount(0)`, which means infinite, is therefore ignored whenever the `AnimatedImageData` carries a finite count.

- Affected: `src/main/java/dev/simplified/image/codec/webp/WebPImageWriter.java:61-63`
- Severity: **LOW**
- Type: **BUG**
- Status: **OPEN**

## GIF writer ignores the image's loop count whenever options are passed
When `GifWriteOptions` are supplied, `GifImageWriter` takes the loop count only from the options, whose default of 0 means infinite. A finite-loop GIF re-encoded with options, for example to set transparency, loops forever. The WebP writer applies the opposite precedence.

- Affected: `src/main/java/dev/simplified/image/codec/gif/GifImageWriter.java:59-68`
- Severity: **LOW**
- Type: **BUG**
- Status: **OPEN**

## Four libwebp oracle tests pass instead of skipping when the oracle is missing
Each ends with `if (x == null) return;` when Python or the `webp` package is unavailable, so a run without the oracle reports PASSED for checks that never ran.

- Affected: `src/test/java/dev/simplified/image/codec/webp/lossy/VP8CodecTest.java:681`, `:848`, `:865`, `:873`
- Severity: **LOW**
- Type: **GAP**
- Status: **OPEN**

## CLAUDE.md overstates the test-abort guarantee
Line 36 promises that the VP8 tests abort via `TestAbortedException` when Python or the package is missing; the four silent passes above contradict it.

- Affected: `CLAUDE.md:36`
- Severity: **LOW**
- Type: **GAP**
- Status: **OPEN**

## Python launchers fall through only on `IOException`
Every launcher copy moves to the next interpreter candidate only when `ProcessBuilder.start()` throws, so the first candidate that launches wins even if it cannot import `webp`. The suspected trigger is a Windows Store `python3` stub; the only evidence for its behaviour is the code comment at `VP8CodecTest.java:777-786`, which says it can shadow a working install (UNVERIFIED that it launches and then fails).

- Affected: `src/test/java/dev/simplified/image/codec/webp/lossy/VP8CodecTest.java:257-268`, `:777-805`, `:2251-2279`, `src/test/java/dev/simplified/image/codec/webp/WebPRoundTripTest.java:1528-1547`
- Severity: **LOW**
- Type: **RISK**
- Status: **OPEN**

## `VP8EncoderTests.startPython` ignores the `vp8.pythonBin` override
This launcher tries only `python3`, `python` and `py`, unlike the other copies, which honour `-Dvp8.pythonBin` first. If the first interpreter that starts cannot import `webp` and exits non-zero without printing `NO_WEBP`, the caller fails with `AssertionError("libwebp rejected ...")` instead of skipping.

- Affected: `src/test/java/dev/simplified/image/codec/webp/lossy/VP8CodecTest.java:257-268`
- Severity: **LOW**
- Type: **RISK**
- Status: **OPEN**

## Test file paths are spliced into Python `r'...'` literals
The oracle scripts build `webp.load_image(r'<path>')` and `open(r'<path>')` by string concatenation, so a path containing `'` breaks the script. `Files.createTempFile` paths keep it harmless today; passing paths as argv removes it.

- Affected: `src/test/java/dev/simplified/image/codec/webp/lossy/VP8CodecTest.java:229`, `:452`, `:740`, `:2162`, `:2167`, `:2198`, `src/test/java/dev/simplified/image/codec/webp/WebPRoundTripTest.java:1179`, `:1239`
- Severity: **LOW**
- Type: **RISK**
- Status: **OPEN**

## Python launcher and PSNR helper are duplicated across test classes
There are four copies of the interpreter launcher and three of the PSNR loop, which is how the override and fall-through behaviour above drifted apart.

- Affected: `src/test/java/dev/simplified/image/codec/webp/lossy/VP8CodecTest.java:257-268`, `:777-805`, `:2251-2279`, `src/test/java/dev/simplified/image/codec/webp/WebPRoundTripTest.java:1528-1547` (launchers); `VP8CodecTest.java:399` (`computePsnr`), `:693` (`sourcePsnr`), `:2115` (`pixelPsnr`)
- Severity: **LOW**
- Type: **GAP**
- Status: **OPEN**

## ICO magic collides with a 256-byte ISOBMFF `ftyp` box
The ICO arm of `ImageFormat.matches` accepts `00 00 01 00`, which is also the big-endian size field of a 256-byte `ftyp` box, and `ImageFactory` registers the ICO reader before any format added later. No ISOBMFF-based format is registered today; MEDIUM once one (AVIF, HEIF) is added after ICO.

- Affected: `src/main/java/dev/simplified/image/ImageFormat.java:81-85`, `src/main/java/dev/simplified/image/ImageFactory.java:79`
- Severity: **LOW**
- Type: **RISK**
- Status: **OPEN**

## No ICC / Exif / XMP metadata model; WebP ICC profiles are dropped
`WebPChunk.Type` names `ICCP`, `EXIF` and `XMP `, but no reader or writer carries them and `ImageData` has no metadata field, so colour profiles and Exif are lost on every WebP round trip. MEDIUM once a wide-gamut or HDR format depends on its colour metadata.

- Affected: `src/main/java/dev/simplified/image/codec/webp/WebPChunk.java:39-54`, `src/main/java/dev/simplified/image/codec/webp/WebPImageReader.java`, `src/main/java/dev/simplified/image/codec/webp/WebPImageWriter.java`, `src/main/java/dev/simplified/image/ImageData.java`
- Severity: **LOW**
- Type: **GAP**
- Status: **OPEN**

## Loop-count unit differs per container but is stored raw
The GIF NETSCAPE loop count and the WebP `ANIM` loop count pass unchanged into `AnimatedImageData.loopCount`, although players are widely reported to read the two in different units (repetitions after the first play versus total plays; UNVERIFIED against browser source). `AnimatedImageData` documents no unit beyond "0 for infinite".

- Affected: `src/main/java/dev/simplified/image/codec/gif/GifImageReader.java:103`, `src/main/java/dev/simplified/image/codec/webp/WebPImageReader.java:99-104`, `src/main/java/dev/simplified/image/data/AnimatedImageData.java:25`
- Severity: **LOW**
- Type: **RISK**
- Status: **OPEN**

## No disposal/blend-aware timeline renderer
`FrameNormalizer` resizes and positions frames but passes disposal and blend through unchanged, and nothing composites a GIF/WebP timeline (offsets, `OVER` blending, `RESTORE_*` disposal) into full canvases. A target format that only carries full-canvas frames cannot be written from such input without one.

- Affected: `src/main/java/dev/simplified/image/transform/FrameNormalizer.java:32-138`
- Severity: **LOW**
- Type: **GAP**
- Status: **OPEN**

## No NOTICE / third-party attribution file
`VP8Tables` and `VP8Costs` carry tables copied from libwebp, whose BSD licence requires its copyright notice to travel with redistributed source and binaries. The repo root holds only `LICENSE.md` (Apache-2.0), `README.md`, `CONTRIBUTING.md` and `CLAUDE.md`, and no libwebp notice appears anywhere in the tree. Tables taken from further BSD-licensed codecs would need the same.

- Affected: repository root, `src/main/java/dev/simplified/image/codec/webp/lossy/VP8Tables.java`, `src/main/java/dev/simplified/image/codec/webp/lossy/VP8Costs.java`
- Severity: **LOW**
- Type: **RISK**
- Status: **OPEN**

## CLAUDE.md package inventory, format list, dependencies and class counts drifted
Line 3 lists 5 formats where 10 exist (TIFF, ICO, TGA, QOI and PNM are missing), and line 44 says 60 source and 8 test classes where there are 92 and 13. Line 41 names a `reflection` dependency that `build.gradle.kts` does not declare. The package lists omit 15+ classes, including the public `NearLosslessPreprocess`, `VP8EncoderSession` and `VP8DecoderSession`.

- Affected: `CLAUDE.md:3`, `:7`, `:9`, `:13-16`, `:41`, `:44`
- Severity: **LOW**
- Type: **GAP**
- Status: **OPEN**

## `ImageFactory` javadoc lists 5 auto-registered formats; 10 are registered
The class javadoc names JPEG, PNG, BMP, GIF and WebP, while the constructor registers 10 readers and 10 writers.

- Affected: `src/main/java/dev/simplified/image/ImageFactory.java:56`
- Severity: **LOW**
- Type: **GAP**
- Status: **OPEN**

## Stale class javadocs on VP8/VP8L codec classes
Four class headers describe shipped features as absent or future work: `VP8Encoder` calls V/H/TM and B_PRED future work, `VP8Decoder` says only keyframes are supported, `VP8LEncoder` says there are no multi-Huffman groups, colour cache or transforms, and `VP8TokenDecoder` anticipates a spec-compliant decoder that already exists.

- Affected: `src/main/java/dev/simplified/image/codec/webp/lossy/VP8Encoder.java:10-13`, `src/main/java/dev/simplified/image/codec/webp/lossy/VP8Decoder.java:19-21`, `src/main/java/dev/simplified/image/codec/webp/lossless/VP8LEncoder.java:22-24`, `src/main/java/dev/simplified/image/codec/webp/lossy/VP8TokenDecoder.java:9-11`
- Severity: **LOW**
- Type: **GAP**
- Status: **OPEN**
