package dev.simplified.image.codec.webp;

import dev.simplified.image.pixel.PixelBuffer;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

/**
 * Per-frame bounding-box diff utilities for animated WebP output. Given two
 * consecutive frames, computes the minimum rectangle enclosing the changed
 * pixels so the downstream ANMF encoder can emit just that sub-frame rather
 * than the full canvas. Mirrors libwebp's {@code WebPAnimEncoder} partial-frame
 * behaviour: drastically reduces file size on content where only a small
 * region of the canvas changes per frame (tooltip animations where the body
 * is static and only one line of obfuscated text cycles).
 * <p>
 * The WebP animated spec stores each ANMF's x/y offset as {@code offset / 2}
 * in 3 bytes, so effective offsets are always even pixels. This class aligns
 * the bounding-box offset down to the nearest even coordinate, growing the
 * rectangle by at most one pixel on each axis.
 *
 * @see <a href="https://developers.google.com/speed/webp/docs/riff_container#animation">WebP Animation container</a>
 */
final class FrameDiffUtil {

    private FrameDiffUtil() { }

    /**
     * Bounding box of pixels that differ between {@code prev} and {@code curr},
     * aligned to even x/y coordinates (required by the ANMF offset field). A
     * {@code null} return means the two frames are pixel-identical; callers
     * emitting an ANMF per input frame should fall back to a 1x1 placeholder
     * rectangle at {@code (0, 0)} in that case (matches libwebp's behaviour).
     *
     * @param prev previous frame's canvas-sized pixel buffer
     * @param curr current frame's canvas-sized pixel buffer
     * @return {@code [x, y, width, height]} or {@code null} when identical
     * @throws IllegalArgumentException when dimensions don't match
     */
    static int @Nullable [] computeBoundingBox(@NotNull PixelBuffer prev, @NotNull PixelBuffer curr) {
        int w = prev.width();
        int h = prev.height();
        if (curr.width() != w || curr.height() != h)
            throw new IllegalArgumentException(String.format(
                "dimension mismatch: prev=%dx%d curr=%dx%d",
                w, h, curr.width(), curr.height()));

        int[] pp = prev.pixels();
        int[] cp = curr.pixels();
        int minX = Integer.MAX_VALUE, minY = Integer.MAX_VALUE;
        int maxX = -1, maxY = -1;
        for (int y = 0; y < h; y++) {
            int rowOff = y * w;
            for (int x = 0; x < w; x++) {
                if (pp[rowOff + x] != cp[rowOff + x]) {
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
            }
        }
        if (maxX < 0) return null;

        // Align offset down to even; grow width/height to cover the shift.
        int alignedX = minX & ~1;
        int alignedY = minY & ~1;
        int width = (maxX - alignedX) + 1;
        int height = (maxY - alignedY) + 1;
        return new int[] { alignedX, alignedY, width, height };
    }

    /**
     * Returns a new {@link PixelBuffer} containing the {@code width x height}
     * sub-rectangle of {@code src} starting at {@code (offsetX, offsetY)}.
     * When the rectangle is the full canvas (offset 0,0 and matching
     * dimensions), callers may prefer to pass {@code src} through directly
     * and skip this copy - this helper always allocates.
     */
    static @NotNull PixelBuffer extractSubBuffer(
        @NotNull PixelBuffer src, int offsetX, int offsetY, int width, int height
    ) {
        int sw = src.width();
        int sh = src.height();
        if (offsetX < 0 || offsetY < 0 || width <= 0 || height <= 0
                || offsetX + width > sw || offsetY + height > sh)
            throw new IllegalArgumentException(String.format(
                "sub-buffer out of bounds: offset=(%d,%d) size=%dx%d canvas=%dx%d",
                offsetX, offsetY, width, height, sw, sh));

        int[] out = new int[width * height];
        int[] in = src.pixels();
        for (int y = 0; y < height; y++)
            System.arraycopy(in, (offsetY + y) * sw + offsetX, out, y * width, width);
        return PixelBuffer.of(out, width, height);
    }
}
