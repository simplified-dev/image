# Known open defects

Defects present in the current tree and not yet fixed, most severe first. Each entry names the cause, the affected code, what it costs today, and what would raise its severity.

## `create` answers straight colour under a name that does not say so
Every `PixelBuffer` holds its pixels in one `AlphaMode`, fixed by the factory that makes it, and every operation reads it from there: a composite lays its source in the destination's convention, a colour argument is straight colour laid in the buffer's, and `wrap` reads the convention off the image. Every blank factory names its convention but one. `canvas` answers a transparent premultiplied buffer, what drawing source-over onto transparency leaves; `create(int, int)` answers a transparent straight one under a name that says nothing, because minecraft-text's compiled glyph code calls it for its glyph bitmaps, which are straight. A render drawn into a `create` buffer by a raw writer - `setPixel` of `ColorMath.blend(.., BlendMode.NORMAL)`, as minecraft-text lays its text, or asset-renderer's rasterizer - holds premultiplied colour the buffer calls straight: `getAlphaMode()` answers straight, a composite lays it by the straight formulas, `resize` and FXAA weigh it by its alpha again, and the writers encode each see-through pixel too dark, its colour scaled by its alpha. A buffer drawn into that way is made with `canvas`. It closes when minecraft-text next re-pins image and `create` gives way to a straight blank factory whose name says so. MEDIUM until a consumer re-pins image; HIGH if one leaves a render on `create`.

- Affected: `src/main/java/dev/simplified/image/PixelBuffer.java` (`create`)
- Severity: **MEDIUM**
- Type: **RISK**
- Status: **OPEN**

## PixelGraphics' fills and shapes overwrite the pixel while it reports `SrcOver`
`PixelGraphics` extends `Graphics2D`, starts with `AlphaComposite.SrcOver` and hands back from `getComposite` whatever `setComposite` stored, but no colour draw reads it. `fillDeviceRect` sends every rectangle and every row a fill, outline or shape lays at once to `PixelBuffer#fillRect(int, int, int, int, int)`, an overwrite that writes the colour in the buffer's convention, and `setPixelSafe` sends each pixel a line, oval, arc (`fillArc` included) or rounded corner plots alone to `setPixel`, which writes it as given; only the `drawImage` forms composite, source-over by the formula the image's and the target's conventions give. A translucent colour therefore replaces what lies beneath it rather than laying over it: over a canvas's `F00F000F`, a `0x80FF0000` fill stores `80800000`, the colour premultiplied, and a pixel plotted alone `80FF0000`, colour above its alpha, where `BlendMode.NORMAL`, and Java2D's `SrcOver` on a `TYPE_INT_ARGB_PRE` image, store `F8870007`; over a straight buffer's `F0100010` both store `80FF0000`, where `BlendMode.STRAIGHT`, and Java2D's `SrcOver` on a `TYPE_INT_ARGB` image, store `F88C0008`. Several draws also plot a pixel more than once in one call, which an overwrite hides and a composite would not, since a pixel laid twice takes the colour twice: `drawRect`'s four edges share their corners (and lie one on another when it has no width or height), `drawOval` plots the points on each axis twice, `drawRoundRect`'s corner arcs repeat the end pixels of its straight edges (with both arc sizes below 2 its edges meet at their corners as `drawRect`'s do), `fillRoundRect`'s body shares its top and bottom rows with the corner bands, `drawPolyline` and `drawPolygon` repeat each shared vertex, `fillPolygon` can repeat a node two spans share, `drawArc`'s sampled points can repeat, and `draw(Shape)` repeats the point where two of its segments meet, a curve being drawn as many short lines. No workspace caller draws a translucent fill or shape through a `PixelGraphics`: asset-renderer's text decorations go through minecraft-text's own `fillEffect`, and minecraft-text calls none of the fills it inherits. HIGH once a caller draws a translucent fill or shape through one.

- Affected: `src/main/java/dev/simplified/image/pixel/PixelGraphics.java:132-135` (`setPixelSafe`), `:158-170` (`fillDeviceRect`), `:333-340` (`fillRect`, `clearRect`), `:343-734` (outlines and shapes), `:741-769` (`drawQuadCurve`, `drawCubicCurve`), `:1014-1021` (`setComposite`, `getComposite`)
- Severity: **MEDIUM**
- Type: **BUG**
- Status: **OPEN**

## No ICC / Exif / XMP metadata model; WebP ICC profiles are dropped
`WebPChunk.Type` names `ICCP`, `EXIF` and `XMP `, but no reader or writer carries them and `ImageData` has no metadata field, so colour profiles and Exif are lost on every WebP round trip. MEDIUM once a wide-gamut or HDR format depends on its colour metadata.

- Affected: `src/main/java/dev/simplified/image/codec/webp/WebPChunk.java:39-54`, `src/main/java/dev/simplified/image/codec/webp/WebPImageReader.java`, `src/main/java/dev/simplified/image/codec/webp/WebPImageWriter.java`, `src/main/java/dev/simplified/image/ImageData.java`
- Severity: **LOW**
- Type: **GAP**
- Status: **OPEN**
