package dev.simplified.image.codec.webp.lossless;

import dev.simplified.image.codec.webp.WebPWriteOptions;
import org.jetbrains.annotations.NotNull;

/**
 * Near-lossless image preprocessing that adjusts pixel values in non-smooth
 * regions to help VP8L entropy coding compress more tightly, while guaranteeing
 * a bounded per-channel deviation from the source.
 * <p>
 * Ported from libwebp's {@code src/enc/near_lossless_enc.c} verbatim, including
 * the iterative-refinement pass structure and the small-image bypass. The
 * output of {@link #apply(int[], int, int, int)} is bit-identical to
 * {@code cwebp -near_lossless N} on the same source at matched level.
 * <p>
 * Algorithm summary:
 * <ul>
 *   <li>Level {@code [0..100]}, inverted scale: {@code 100} = off (default),
 *       {@code 0} = most aggressive quantization.</li>
 *   <li>Level maps to a {@code limit_bits} value via
 *       {@code limit_bits = 5 - level / 20} (table: 100 -> 0, 80-99 -> 1,
 *       60-79 -> 2, 40-59 -> 3, 20-39 -> 4, 0-19 -> 5).</li>
 *   <li>For each interior pixel: check 4-connected neighbours (left, right,
 *       up, down). If all channels of all neighbours differ by less than
 *       {@code 1 << limit_bits}, the pixel is "smooth" and passes through
 *       unchanged. Otherwise, each channel is snapped to the closest multiple
 *       of {@code 1 << limit_bits} via bankers' rounding with clamp-at-255.</li>
 *   <li>Small-image bypass: if ({@code width < 64 && height < 64}) or
 *       {@code height < 3}, the preprocessing is skipped and pixels copy
 *       through unchanged.</li>
 *   <li>Iterative refinement: the preprocessing runs {@code limit_bits}
 *       times, with decreasing bit widths. So a level of 0 (5 bits) runs
 *       5 full passes.</li>
 * </ul>
 * <p>
 * Typical usage: invoked by the VP8L encode path when
 * {@link WebPWriteOptions#getNearLossless()}
 * returns a value less than 100. The preprocessed pixel buffer is then
 * encoded normally via {@link VP8LEncoder#encode}; the output is a standard
 * VP8L bitstream decodable by any conformant WebP decoder (including libwebp
 * and browser/OS image stacks built on it).
 *
 * @see <a href="https://chromium.googlesource.com/webm/libwebp/+/refs/heads/main/src/enc/near_lossless_enc.c">libwebp near_lossless_enc.c</a>
 */
public final class NearLosslessPreprocess {

    /**
     * Below this dimension libwebp skips near-lossless entirely (icon heuristic).
     */
    private static final int MIN_DIM_FOR_NEAR_LOSSLESS = 64;
    /**
     * Maximum {@code limit_bits} derived from level 0.
     */
    private static final int MAX_LIMIT_BITS = 5;

    private NearLosslessPreprocess() { }

    /**
     * Runs libwebp near-lossless preprocessing on {@code argb} and returns a
     * new ARGB buffer. When {@code level >= 100} the feature is off and the
     * returned buffer is a plain copy of the input.
     *
     * @param argb source pixel buffer, length {@code width * height}, packed
     *             {@code 0xAARRGGBB}
     * @param width image width in pixels
     * @param height image height in pixels
     * @param level {@code [0..100]} - higher = closer to lossless; {@code 100}
     *              = off
     * @return a new {@code int[width * height]} with the preprocessed pixels
     */
    public static int @NotNull [] apply(int @NotNull [] argb, int width, int height, int level) {
        int[] dst = new int[width * height];
        if (level >= 100) {
            System.arraycopy(argb, 0, dst, 0, width * height);
            return dst;
        }

        // Level -> limit_bits: 0..19 -> 5, 20..39 -> 4, 40..59 -> 3, 60..79 -> 2,
        // 80..99 -> 1. Level 100 is handled above (would map to 0 bits = no-op).
        int limitBits = MAX_LIMIT_BITS - Math.max(0, level) / 20;
        if (limitBits <= 0 || limitBits > MAX_LIMIT_BITS) {
            System.arraycopy(argb, 0, dst, 0, width * height);
            return dst;
        }

        // Small-image bypass matches libwebp: don't preprocess tiny icons,
        // and guarantee ysize >= 3 so the prev/curr/next row rotation has
        // data to work with.
        if ((width < MIN_DIM_FOR_NEAR_LOSSLESS && height < MIN_DIM_FOR_NEAR_LOSSLESS)
            || height < 3) {
            System.arraycopy(argb, 0, dst, 0, width * height);
            return dst;
        }

        int[] scratch = new int[3 * width];
        nearLossless(width, height, argb, width, limitBits, scratch, dst);
        // Iterative refinement with decreasing bit widths; matches libwebp's
        // VP8ApplyNearLossless tail loop exactly. For level 80-99 (limitBits=1)
        // this loop body is skipped entirely.
        for (int i = limitBits - 1; i != 0; i--)
            nearLossless(width, height, dst, width, i, scratch, dst);
        return dst;
    }

    /**
     * One full near-lossless pass over the image at {@code limitBits} bit
     * width. Reads rows from {@code argbSrc} with the given {@code stride},
     * writes preprocessed rows into {@code argbDst} at a tight stride of
     * {@code xsize}. The {@code copyBuffer} is scratch space of length
     * {@code 3 * xsize} used to hold the rolling 3-row neighbourhood so the
     * read-and-write-same-buffer case (iterative passes) works without
     * corruption.
     */
    private static void nearLossless(
        int xsize, int ysize, int @NotNull [] argbSrc, int stride,
        int limitBits, int @NotNull [] copyBuffer, int @NotNull [] argbDst
    ) {
        int limit = 1 << limitBits;
        int prevRowOff = 0;
        int currRowOff = xsize;
        int nextRowOff = 2 * xsize;

        // Seed the rolling window: curr = row 0, next = row 1. prev is
        // uninitialised but never read on y=0 (copy-through branch).
        System.arraycopy(argbSrc, 0, copyBuffer, currRowOff, xsize);
        System.arraycopy(argbSrc, stride, copyBuffer, nextRowOff, xsize);

        int srcRowOff = 0;
        int dstRowOff = 0;
        for (int y = 0; y < ysize; y++) {
            if (y == 0 || y == ysize - 1) {
                // Borders copy verbatim.
                System.arraycopy(argbSrc, srcRowOff, argbDst, dstRowOff, xsize);
            } else {
                // Refresh next_row with argbSrc[y+1]. Safe when argbSrc ==
                // argbDst because argbSrc[(y+1)*stride..] hasn't been written
                // yet in this pass.
                System.arraycopy(argbSrc, srcRowOff + stride, copyBuffer, nextRowOff, xsize);
                // Column borders copy verbatim.
                argbDst[dstRowOff] = argbSrc[srcRowOff];
                argbDst[dstRowOff + xsize - 1] = argbSrc[srcRowOff + xsize - 1];
                for (int x = 1; x < xsize - 1; x++) {
                    int center = copyBuffer[currRowOff + x];
                    if (isSmooth(copyBuffer, prevRowOff, currRowOff, nextRowOff, x, limit))
                        argbDst[dstRowOff + x] = center;
                    else
                        argbDst[dstRowOff + x] = closestDiscretizedArgb(center, limitBits);
                }
            }
            // Three-way rotation: prev := curr, curr := next, next := old prev
            // (which is now scratch - its contents get overwritten at the top
            // of the next interior iteration).
            int tempOff = prevRowOff;
            prevRowOff = currRowOff;
            currRowOff = nextRowOff;
            nextRowOff = tempOff;

            srcRowOff += stride;
            dstRowOff += xsize;
        }
    }

    /**
     * A pixel is "smooth" when each of its 4-connected neighbours is
     * {@link #isNear near} to it on every channel. Mirrors libwebp's
     * {@code IsSmooth}.
     */
    private static boolean isSmooth(
        int @NotNull [] buf, int prevOff, int currOff, int nextOff, int x, int limit
    ) {
        int center = buf[currOff + x];
        return isNear(center, buf[currOff + x - 1], limit)
            && isNear(center, buf[currOff + x + 1], limit)
            && isNear(center, buf[prevOff + x], limit)
            && isNear(center, buf[nextOff + x], limit);
    }

    /**
     * {@code true} when every channel of {@code a} and {@code b} differs by
     * strictly less than {@code limit}. Note the strictness: a delta of
     * exactly {@code +limit} or {@code -limit} is NOT near - this matters on
     * the edge case where adjacent channels differ by exactly
     * {@code 1 << limit_bits}. Mirrors libwebp's {@code IsNear}.
     */
    private static boolean isNear(int a, int b, int limit) {
        for (int k = 0; k < 4; k++) {
            int delta = ((a >>> (k * 8)) & 0xFF) - ((b >>> (k * 8)) & 0xFF);
            if (delta >= limit || delta <= -limit) return false;
        }
        return true;
    }

    /**
     * Applies {@link #findClosestDiscretized} to each channel of {@code argb}.
     */
    private static int closestDiscretizedArgb(int argb, int bits) {
        return (findClosestDiscretized((argb >>> 24) & 0xFF, bits) << 24)
             | (findClosestDiscretized((argb >>> 16) & 0xFF, bits) << 16)
             | (findClosestDiscretized((argb >>> 8) & 0xFF, bits) << 8)
             | findClosestDiscretized(argb & 0xFF, bits);
    }

    /**
     * Snaps an 8-bit channel value to the closest multiple of {@code 1 << bits}
     * (or to {@code 0xFF}), resolving ties via bankers' rounding. Mirrors
     * libwebp's {@code FindClosestDiscretized}; the {@code ((a >> bits) & 1)}
     * term is what makes the rounding banker-correct on exact midpoints.
     */
    private static int findClosestDiscretized(int a, int bits) {
        int mask = (1 << bits) - 1;
        int biased = a + (mask >> 1) + ((a >> bits) & 1);
        if (biased > 0xFF) return 0xFF;
        return biased & ~mask;
    }
}
