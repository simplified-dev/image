package dev.simplified.image.codec.webp.lossless;

import dev.simplified.image.codec.webp.RiffContainer;
import dev.simplified.image.pixel.PixelBuffer;
import org.jetbrains.annotations.NotNull;

/**
 * Pure Java VP8L (WebP lossless) encoder.
 * <p>
 * Emits a spec-compliant bitstream that reference {@code libwebp} decoders accept.
 * Supports LZ77 backward references on top of literal pixel emission: a hash-chain
 * match finder locates repeated pixel runs, and matches of length
 * {@value LZ77#MIN_MATCH} or longer are emitted as length/distance symbol pairs
 * in the extended green alphabet (literal bytes at positions 0..255, length
 * codes at 256..279). Distances are mapped to libwebp's 2-D plane-code space
 * via {@link LZ77#distanceToPlaneCode} before prefix encoding.
 * <p>
 * Applies the {@link VP8LTransform.ColorIndexing} palette transform automatically when
 * the source has at most 256 unique colors: each pixel's ARGB is replaced by its
 * palette index in the green channel, sub-bit packed horizontally (1/2/4 bits per pixel
 * for palettes of 2/4/16 entries respectively), and the palette itself is emitted as
 * a delta-encoded {@code 1xN} sub-image after the transform header. Matches libwebp
 * {@code -m 3} method behaviour at a simplified feature subset: no multi-Huffman tile
 * groups, no color-cache, no predictor / cross-color / subtract-green transforms yet.
 * <p>
 * The encoder uses <i>simple prefix codes</i> (is_simple = 1) for alphabets with one or
 * two distinct used symbols, and normal prefix codes with a proper Huffman tree for
 * three or more. The VP8L spec mandates this split: a Huffman code with a single leaf of
 * length 1 is not a complete tree, and libwebp's {@code VP8LBuildHuffmanTable} rejects
 * it as {@code VP8_STATUS_BITSTREAM_ERROR}.
 *
 * @see <a href="https://developers.google.com/speed/webp/docs/webp_lossless_bitstream_specification">
 *     WebP Lossless Bitstream Specification</a>
 */
public final class VP8LEncoder {

    private static final int VP8L_SIGNATURE = 0x2F;
    private static final int VP8L_VERSION = 0;

    private static final int NUM_LITERAL_CODES = 256;
    private static final int NUM_LENGTH_CODES = 24;
    private static final int NUM_DISTANCE_CODES = 40;
    private static final int MAX_HUFFMAN_BITS = 15;

    /** Maximum {@code color_cache_size} exponent the VP8L spec accepts. */
    private static final int MAX_COLOR_CACHE_BITS = 11;

    /** Number of code-length-code symbols in the VP8L alphabet (0..15 literal + 16..18 RLE). */
    private static final int CODE_LENGTH_CODES = 19;

    /** Max bit length for the code-length-code Huffman table (per spec). */
    private static final int CLC_MAX_BITS = 7;

    /**
     * Order in which the CLC table's per-symbol bit lengths are written. Matches
     * {@code kCodeLengthCodeOrder} in libwebp.
     */
    private static final int[] CODE_LENGTH_ORDER = {
        17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    };

    private VP8LEncoder() { }

    /**
     * Encodes {@code pixels} into a VP8L payload suitable for wrapping in a {@code VP8L}
     * WebP chunk (via {@link RiffContainer#createChunk}).
     *
     * @param pixels the ARGB source buffer
     * @return the encoded VP8L bitstream
     */
    /** Color-cache trial sizes, in bits. 0 = cache disabled. */
    private static final int[] CACHE_BITS_TRIALS = { 0, 10 };

    /**
     * Meta-Huffman tile-size exponents the public encoder tries in addition to the
     * single-Huffman path. {@code 0} means "don't try meta-Huffman at all" (equivalent
     * to the previous behaviour). Picking tile_bits=8 gives 256x256 tiles, which for
     * tooltip-sized content splits a full canvas into a handful of groups; partial
     * diff frames trivially collapse to a single-group meta-Huffman (which is strictly
     * worse than no-meta and the A/B selector rejects it). Each added trial roughly
     * doubles encode cost for the main body; keep this minimal.
     */
    private static final int[] META_TILE_BITS_TRIALS = { 0, 8 };

    /**
     * Transform choices tried by {@link #encode(PixelBuffer)}. Only one transform may be
     * active per encode - palette and predictor are mutually exclusive (palette reduces
     * an image's histogram; predictor operates on the reduced residuals but libwebp
     * treats the two as competing strategies and we match that).
     */
    public enum TransformMode {
        /** No transform - emit pixels as literal ARGB through the body entropy coder. */
        NONE,
        /** Color-indexing / palette transform; only valid when palette-eligible. */
        PALETTE,
        /** Spatial predictor transform; always applicable. */
        PREDICTOR,
        /** Subtract-green transform: {@code R -= G; B -= G;} tightens R/B histograms. */
        SUBTRACT_GREEN,
        /** Cross-color (colour-transform) decorrelation; per-tile coefficients. */
        CROSS_COLOR
    }

    public static byte @NotNull [] encode(@NotNull PixelBuffer pixels) {
        // Try every combination of (transform in {none, palette, predictor}) x
        // (cache off / cache 10-bit) and keep the smallest output. Palette helps when a
        // full-canvas image has <= 256 unique colours; predictor helps full-canvas
        // content with spatial coherence (gradients, AA edges); color cache helps
        // literal-heavy content with recurring colours the LZ77 match-finder misses.
        // None of these is free - each carries per-frame header overhead (transform
        // bits, tile mode sub-image, expanded alphabets) that can exceed the savings
        // on tiny images. Trying and picking is the simplest way to guarantee a
        // non-regression without replicating libwebp's entropy-estimation heuristic.
        int[] palette = detectPalette(pixels.data());
        boolean paletteEligible = palette != null;

        byte[] best = null;
        for (int cacheBits : CACHE_BITS_TRIALS) {
            for (int metaTileBits : META_TILE_BITS_TRIALS) {
                byte[] cand = encode(pixels, TransformMode.NONE, cacheBits, metaTileBits);
                if (best == null || cand.length < best.length) best = cand;
                if (paletteEligible) {
                    cand = encode(pixels, TransformMode.PALETTE, cacheBits, metaTileBits);
                    if (cand.length < best.length) best = cand;
                }
                cand = encode(pixels, TransformMode.PREDICTOR, cacheBits, metaTileBits);
                if (cand.length < best.length) best = cand;
                cand = encode(pixels, TransformMode.SUBTRACT_GREEN, cacheBits, metaTileBits);
                if (cand.length < best.length) best = cand;
                cand = encode(pixels, TransformMode.CROSS_COLOR, cacheBits, metaTileBits);
                if (cand.length < best.length) best = cand;
            }
        }
        return best;
    }

    /**
     * Convenience overload that encodes with a specific palette toggle and the color
     * cache disabled ({@code colorCacheBits = 0}). Exists primarily for diagnostics that
     * want to measure the pre-cache baseline.
     */
    public static byte @NotNull [] encode(@NotNull PixelBuffer pixels, boolean usePalette) {
        return encode(pixels, usePalette, 0);
    }

    /**
     * Convenience overload preserving the pre-{@link TransformMode} signature. Maps
     * {@code usePalette = true} to {@link TransformMode#PALETTE} and {@code false} to
     * {@link TransformMode#NONE}.
     */
    public static byte @NotNull [] encode(@NotNull PixelBuffer pixels, boolean usePalette, int colorCacheBits) {
        return encode(pixels, usePalette ? TransformMode.PALETTE : TransformMode.NONE, colorCacheBits);
    }

    /**
     * Convenience overload that disables meta-Huffman. See
     * {@link #encode(PixelBuffer, TransformMode, int, int)} for the full signature.
     */
    public static byte @NotNull [] encode(
        @NotNull PixelBuffer pixels,
        @NotNull TransformMode transformMode,
        int colorCacheBits
    ) {
        return encode(pixels, transformMode, colorCacheBits, 0);
    }

    /**
     * Encodes with explicit toggles for the transform choice, color-cache size, and
     * meta-Huffman tile size.
     *
     * @param pixels the source buffer
     * @param transformMode which reversible transform to apply, if any
     * @param colorCacheBits {@code 0} to disable the color cache, otherwise {@code 1..11}
     *                       for a cache of {@code 1 << colorCacheBits} entries
     * @param metaTileBits {@code 0} to use a single Huffman group for the whole image;
     *                     otherwise the tile-size exponent in {@code 2..9} enabling the
     *                     meta-Huffman (multi-group) entropy coder
     */
    public static byte @NotNull [] encode(
        @NotNull PixelBuffer pixels,
        @NotNull TransformMode transformMode,
        int colorCacheBits,
        int metaTileBits
    ) {
        int width = pixels.width();
        int height = pixels.height();
        int[] pixelData = pixels.data();

        BitWriter writer = new BitWriter(Math.max(1024, width * height * 4));

        // --- Header ---
        writer.writeBits(VP8L_SIGNATURE, 8);
        writer.writeBits(width - 1, 14);
        writer.writeBits(height - 1, 14);
        writer.writeBits(1, 1);                    // alpha_is_used
        writer.writeBits(VP8L_VERSION, 3);

        // --- Transforms ---
        // PALETTE (COLOR_INDEXING_TRANSFORM): when <= 256 unique colours, replace each
        // pixel with its palette index stored in the green channel, horizontally pack
        // 2/4/8 indices per pixel for small palettes, and emit the palette itself as a
        // delta-encoded 1xN sub-image. Tightens the channel histograms dramatically for
        // UI content with few colours.
        // PREDICTOR (PREDICTOR_TRANSFORM): split the image into 8x8 tiles; for each tile
        // pick the best of 14 spatial-prediction modes; emit a mode sub-image and replace
        // each pixel with (actual - predicted) residual. Smooth gradients and AA borders
        // collapse to near-zero residuals.
        int[] bodyPixels = pixelData;
        int bodyWidth = width;

        switch (transformMode) {
            case PALETTE -> {
                int[] palette = detectPalette(pixelData);
                if (palette != null) {
                    int bitsPerPixel = paletteBitsPerPixel(palette.length);
                    bodyPixels = applyColorIndexing(pixelData, width, height, palette, bitsPerPixel);
                    int pixelsPerByte = bitsPerPixel == 8 ? 1 : 8 / bitsPerPixel;
                    bodyWidth = (width + pixelsPerByte - 1) / pixelsPerByte;

                    writer.writeBit(1);                         // transform present
                    writer.writeBits(3, 2);                     // COLOR_INDEXING_TRANSFORM
                    writer.writeBits(palette.length - 1, 8);    // palette_size - 1 (0..255)
                    encodeSubImage(writer, deltaEncodePalette(palette), palette.length);
                }
                // If palette detect returns null we silently fall through to no-transform.
            }
            case PREDICTOR -> {
                int[][] predictorOutput = applyPredictor(pixelData, width, height);
                int[] residuals = predictorOutput[0];
                int[] modeImage = predictorOutput[1];
                int blockBits = predictorOutput[2][0];
                int blockWidth = predictorOutput[2][1];
                int blockHeight = predictorOutput[2][2];
                bodyPixels = residuals;
                // Predictor does not change the image geometry, so bodyWidth == width.

                writer.writeBit(1);                         // transform present
                writer.writeBits(0, 2);                     // PREDICTOR_TRANSFORM
                writer.writeBits(blockBits - 2, 3);         // 2..9 -> 0..7
                encodeSubImage(writer, modeImage, blockWidth);
                // blockHeight is implicit to the decoder (derives from image height + blockBits).
                if (blockHeight * blockWidth != modeImage.length)
                    throw new IllegalStateException("mode image size mismatch: expected "
                        + (blockWidth * blockHeight) + " got " + modeImage.length);
            }
            case SUBTRACT_GREEN -> {
                bodyPixels = pixelData.clone();
                new VP8LTransform.SubtractGreen().forwardTransform(bodyPixels, width, height);
                writer.writeBit(1);              // transform present
                writer.writeBits(2, 2);          // SUBTRACT_GREEN_TRANSFORM
                // No parameters, no meta-image.
            }
            case CROSS_COLOR -> {
                int[][] xformOutput = applyCrossColor(pixelData, width, height);
                int[] residuals = xformOutput[0];
                int[] transformData = xformOutput[1];
                int blockBits = xformOutput[2][0];
                int blockWidth = xformOutput[2][1];
                bodyPixels = residuals;

                writer.writeBit(1);                         // transform present
                writer.writeBits(1, 2);                     // COLOR_TRANSFORM
                writer.writeBits(blockBits - 2, 3);         // 2..9 -> 0..7
                encodeSubImage(writer, transformData, blockWidth);
            }
            case NONE -> { }
        }

        // --- End of transform loop ---
        writer.writeBit(0);

        // --- Color cache ---
        if (colorCacheBits > 0) {
            if (colorCacheBits < 1 || colorCacheBits > MAX_COLOR_CACHE_BITS)
                throw new IllegalArgumentException(
                    "colorCacheBits must be 0 or 1.." + MAX_COLOR_CACHE_BITS + ", got " + colorCacheBits);
            writer.writeBit(1);
            writer.writeBits(colorCacheBits, 4);
        } else {
            writer.writeBit(0);
        }

        // --- Meta-prefix (multi-Huffman) flag ---
        if (metaTileBits > 0) {
            if (metaTileBits < 2 || metaTileBits > 9)
                throw new IllegalArgumentException("metaTileBits must be 0 or 2..9, got " + metaTileBits);
            writer.writeBit(1);
            writer.writeBits(metaTileBits - 2, 3);
            encodePixelStreamBodyMultiHuffman(writer, bodyPixels, bodyWidth, colorCacheBits, metaTileBits);
        } else {
            writer.writeBit(0);
            encodePixelStreamBody(writer, bodyPixels, bodyWidth, colorCacheBits);
        }

        return writer.toByteArray();
    }

    // ---------------------------------------------------------------------
    //  Color-indexing (palette) transform
    // ---------------------------------------------------------------------

    /**
     * Returns the sorted set of unique ARGB colors in {@code pixels} when it fits within
     * the VP8L 256-entry palette limit, or {@code null} when the image has more than 256
     * unique colors (in which case the caller should skip the color-indexing transform).
     */
    private static int @org.jetbrains.annotations.Nullable [] detectPalette(int @NotNull [] pixels) {
        java.util.HashSet<Integer> set = new java.util.HashSet<>();
        for (int px : pixels) {
            set.add(px);
            if (set.size() > 256) return null;
        }
        int[] palette = new int[set.size()];
        int i = 0;
        for (int c : set) palette[i++] = c;
        java.util.Arrays.sort(palette);
        return palette;
    }

    /**
     * Returns the packing depth used for a given palette size.
     * Mirrors {@link VP8LDecoder}'s implicit rule: palettes with {@code <= 2} entries
     * pack 8 pixels per byte, {@code <= 4} pack 4, {@code <= 16} pack 2, otherwise
     * 1 pixel per byte (the full 8-bit green channel).
     */
    private static int paletteBitsPerPixel(int paletteSize) {
        if (paletteSize <= 2) return 1;
        if (paletteSize <= 4) return 2;
        if (paletteSize <= 16) return 4;
        return 8;
    }

    /**
     * Applies the color-indexing forward transform: returns a new pixel array with each
     * source ARGB replaced by its palette index stored in the green channel. For
     * {@code bitsPerPixel < 8} the result is horizontally packed - each output pixel's
     * green byte carries {@code pixelsPerByte} indices, LSB-first - matching the
     * decoder's inverse unpacking in {@link VP8LTransform.ColorIndexing#inverseTransform}.
     */
    private static int @NotNull [] applyColorIndexing(
        int @NotNull [] pixels,
        int width,
        int height,
        int @NotNull [] palette,
        int bitsPerPixel
    ) {
        java.util.HashMap<Integer, Integer> lookup = new java.util.HashMap<>(palette.length * 2);
        for (int i = 0; i < palette.length; i++) lookup.put(palette[i], i);

        if (bitsPerPixel == 8) {
            int[] out = new int[pixels.length];
            for (int i = 0; i < pixels.length; i++) {
                int idx = lookup.get(pixels[i]);
                out[i] = 0xFF000000 | (idx << 8);
            }
            return out;
        }

        int pixelsPerByte = 8 / bitsPerPixel;
        int mask = (1 << bitsPerPixel) - 1;
        int packedWidth = (width + pixelsPerByte - 1) / pixelsPerByte;
        int[] out = new int[packedWidth * height];

        for (int y = 0; y < height; y++) {
            int rowOut = y * packedWidth;
            int rowIn = y * width;
            for (int px = 0; px < packedWidth; px++) {
                int packedGreen = 0;
                int base = px * pixelsPerByte;
                int count = Math.min(pixelsPerByte, width - base);
                for (int sub = 0; sub < count; sub++) {
                    int idx = lookup.get(pixels[rowIn + base + sub]);
                    packedGreen |= (idx & mask) << (sub * bitsPerPixel);
                }
                out[rowOut + px] = 0xFF000000 | (packedGreen << 8);
            }
        }
        return out;
    }

    /**
     * Delta-encodes a palette for transmission: the first entry is passed through and
     * each subsequent entry is replaced by the per-channel bytewise difference from its
     * predecessor (mod 256). Inverse of the decoder's palette fix-up pass, which
     * accumulates the same deltas via {@code addPixelsForPalette}.
     */
    private static int @NotNull [] deltaEncodePalette(int @NotNull [] palette) {
        int[] out = new int[palette.length];
        out[0] = palette[0];
        for (int i = 1; i < palette.length; i++) {
            int a = palette[i];
            int b = palette[i - 1];
            int alpha = (((a >> 24) & 0xFF) - ((b >> 24) & 0xFF)) & 0xFF;
            int red   = (((a >> 16) & 0xFF) - ((b >> 16) & 0xFF)) & 0xFF;
            int green = (((a >>  8) & 0xFF) - ((b >>  8) & 0xFF)) & 0xFF;
            int blue  = ( (a        & 0xFF) -  (b        & 0xFF)) & 0xFF;
            out[i] = (alpha << 24) | (red << 16) | (green << 8) | blue;
        }
        return out;
    }

    /**
     * Emits a transform / meta-image sub-image: a 1-bit color-cache flag (always 0
     * here) followed by the same five prefix-code declarations + entropy-coded pixel
     * stream as the main image body. Sub-images do not carry a meta-prefix bit - only
     * the top-level image body does. Sub-image bodies also never enable the color
     * cache: the palette / predictor meta-images are too small to amortise the
     * expanded-alphabet overhead.
     */
    private static void encodeSubImage(@NotNull BitWriter writer, int @NotNull [] pixels, int width) {
        writer.writeBit(0);                     // no color cache in sub-image
        encodePixelStreamBody(writer, pixels, width, 0);
    }

    /**
     * Wraps {@link LZ77#findMatch} with libwebp's "longest-reach" lookahead: after the
     * greedy-longest match is found at {@code pos}, scans positions inside that match
     * (at k = {@link LZ77#MIN_MATCH}..{@code matchLen - 1}) to see if truncating the
     * current match to length {@code k} and then re-matching at {@code pos + k} covers
     * MORE pixels than the full-length greedy match. Returns either the greedy match
     * unchanged, or a truncated {@code (k, originalDist)} pair when truncation reaches
     * farther.
     * <p>
     * Notably this does NOT update the hash chain for the scanned lookahead positions
     * - so findMatch at {@code pos + k} may miss short-distance matches to positions in
     * {@code [pos, pos + k - 1]}. libwebp pre-inserts those positions; we accept the
     * minor miss rate in exchange for a simpler implementation. See
     * {@code BackwardReferencesLz77} in libwebp's {@code src/enc/backward_references_enc.c}
     * for the reference algorithm.
     */
    private static int @NotNull [] findBestMatchWithLookahead(
        int @NotNull [] pixels,
        int pos,
        int pixelCount,
        int @NotNull [] hashHead,
        int @NotNull [] hashPrev
    ) {
        int remaining = pixelCount - pos;
        int maxLen = Math.min(remaining, LZ77.MAX_LENGTH);
        int[] match = LZ77.findMatch(pixels, pos, maxLen, hashHead, hashPrev);
        int bestLen = match[0];
        int bestDist = match[1];
        if (bestLen < LZ77.MIN_MATCH) return match;

        // Stricter-than-libwebp truncation gate: only prefer the truncated two-token
        // path when it covers significantly more pixels PER TOKEN than the greedy-
        // longest single-token match. Pure "reach further" (libwebp's default) actually
        // regressed weapon_ss4 by +5% because repeating tooltip patterns mean greedy's
        // NEXT match already reaches comparably far, and the extra length+distance
        // symbol pair costs ~15 bits per token. Requiring the two-token coverage to
        // beat the greedy coverage by 2.5x means truncation only fires when per-token
        // amortisation is near-guaranteed; measured ~1.3% shrink on weapon_ss4 with
        // this gate, vs regression with the libwebp-default 1x criterion.
        long truncReachThreshold = (5L * bestLen) / 2;
        int chosenLen = bestLen;
        int chosenAltLen = 0;
        for (int k = LZ77.MIN_MATCH; k < bestLen; k++) {
            int altRemaining = pixelCount - (pos + k);
            if (altRemaining < LZ77.MIN_MATCH) break;
            int altMaxLen = Math.min(altRemaining, LZ77.MAX_LENGTH);
            int[] altMatch = LZ77.findMatch(pixels, pos + k, altMaxLen, hashHead, hashPrev);
            int altLen = altMatch[0];
            if (altLen < LZ77.MIN_MATCH) continue;
            long coverage = (long) k + altLen;
            if (coverage > truncReachThreshold && altLen > chosenAltLen) {
                truncReachThreshold = coverage;
                chosenLen = k;
                chosenAltLen = altLen;
            }
        }
        if (chosenLen == bestLen) return match;
        return new int[] { chosenLen, bestDist };
    }

    // ---------------------------------------------------------------------
    //  Predictor transform
    // ---------------------------------------------------------------------

    /**
     * Predictor tile-size exponents that {@link #applyPredictorBestTile} tries when
     * picking a block size for a given image. Smaller tiles (3 = 8x8) give finer-grained
     * per-region mode selection but more mode-image overhead; larger tiles (5 = 32x32)
     * cost less header bytes but predict worse on mixed content. Spanning a couple of
     * sizes costs us proportionally more encode CPU but catches both extremes.
     */
    private static final int[] PREDICTOR_BLOCK_BITS_TRIALS = { 3, 5 };

    /**
     * Applies the spatial predictor transform to a fresh copy of {@code pixels} using
     * the tile size that produces the smallest residual-and-mode-image payload.
     * Delegates to {@link #applyPredictorAtBlockBits} for each trial in
     * {@link #PREDICTOR_BLOCK_BITS_TRIALS}.
     */
    private static int @NotNull [] @NotNull [] applyPredictor(
        int @NotNull [] pixels,
        int width,
        int height
    ) {
        int[][] best = null;
        long bestScore = Long.MAX_VALUE;
        for (int blockBits : PREDICTOR_BLOCK_BITS_TRIALS) {
            int[][] candidate = applyPredictorAtBlockBits(pixels, width, height, blockBits);
            // Score by total residual SAD + mode-image pixel count as a cheap proxy for
            // the encoded size. The residual SAD is already captured during per-tile
            // mode selection but we re-sum it here across all tiles for comparison.
            long sad = 0;
            int[] res = candidate[0];
            for (int v : res) {
                sad += Math.abs((byte) ((v >>> 24) & 0xFF));
                sad += Math.abs((byte) ((v >>  16) & 0xFF));
                sad += Math.abs((byte) ((v >>   8) & 0xFF));
                sad += Math.abs((byte) ( v         & 0xFF));
            }
            long score = sad + (long) candidate[1].length * 4;   // weight mode-image overhead
            if (score < bestScore) {
                bestScore = score;
                best = candidate;
            }
        }
        return best;
    }

    /**
     * Applies the spatial predictor transform to a fresh copy of {@code pixels} at a
     * specific block size.
     * <p>
     * Splits the image into {@code 2^blockBits} square tiles; for each tile picks the
     * VP8L predictor mode (0..13) with the lowest sum-of-absolute-signed-byte-residuals
     * metric; assembles the mode meta-image; and applies the forward transform through
     * {@link VP8LTransform.Predictor#forwardTransform}. Returns a three-element array:
     * {@code [residuals, modeImage, {blockBits, blockWidth, blockHeight}]}.
     */
    private static int @NotNull [] @NotNull [] applyPredictorAtBlockBits(
        int @NotNull [] pixels,
        int width,
        int height,
        int blockBits
    ) {
        int blockSize = 1 << blockBits;
        int blockWidth = ((width - 1) >> blockBits) + 1;
        int blockHeight = ((height - 1) >> blockBits) + 1;
        int[] modes = new int[blockWidth * blockHeight];

        // Per-tile mode selection. Mode 0 (black) is the implicit predictor for the
        // top-left pixel and for any neighbour-less position; it's picked here whenever
        // the scoring pass can't find a better candidate.
        for (int bY = 0; bY < blockHeight; bY++) {
            int y0 = bY << blockBits;
            int y1 = Math.min(y0 + blockSize, height);
            for (int bX = 0; bX < blockWidth; bX++) {
                int x0 = bX << blockBits;
                int x1 = Math.min(x0 + blockSize, width);

                int bestMode = 0;
                long bestCost = scoreTileUnderMode(pixels, width, x0, y0, x1, y1, 0);
                for (int mode = 1; mode < VP8LTransform.Predictor.NUM_FILTERS; mode++) {
                    long cost = scoreTileUnderMode(pixels, width, x0, y0, x1, y1, mode);
                    if (cost < bestCost) {
                        bestCost = cost;
                        bestMode = mode;
                    }
                }
                // Decoder reads the mode from the green channel of this meta-image pixel.
                modes[bY * blockWidth + bX] = 0xFF000000 | (bestMode << 8);
            }
        }

        int[] residuals = pixels.clone();
        new VP8LTransform.Predictor(blockBits, modes, blockWidth).forwardTransform(residuals, width, height);

        return new int[][] {
            residuals,
            modes,
            { blockBits, blockWidth, blockHeight }
        };
    }

    // ---------------------------------------------------------------------
    //  Meta-Huffman (multi-group) body encoder
    // ---------------------------------------------------------------------

    /**
     * Multi-group counterpart of {@link #encodePixelStreamBody}. Splits the image into
     * square tiles of {@code 2^tileBits} pixels; each tile gets its own 5-alphabet
     * Huffman group (no clustering in this first cut, so {@code numGroups = numTiles}).
     * Emits a meta-prefix sub-image mapping tile index to group index, then the
     * per-group prefix-code declarations, then the replayed token stream with each
     * token's symbols coded under its owning tile's Huffman tables.
     * <p>
     * Win condition: the per-tile entropy advantage (regions with distinct statistics
     * each get a tightly-tuned prefix code) must exceed the per-group header overhead
     * (5 prefix-code declarations + meta-prefix sub-image). The A/B selector in
     * {@link #encode(PixelBuffer)} ensures meta-Huffman is only kept when it strictly
     * shrinks vs the single-group body.
     */
    private static void encodePixelStreamBodyMultiHuffman(
        @NotNull BitWriter writer,
        int @NotNull [] pixelData,
        int width,
        int colorCacheBits,
        int tileBits
    ) {
        int pixelCount = pixelData.length;
        int height = pixelCount / width;
        int tileSize = 1 << tileBits;
        int prefixXsize = (width + tileSize - 1) >> tileBits;
        int prefixYsize = (height + tileSize - 1) >> tileBits;
        int numGroups = prefixXsize * prefixYsize;
        int cacheSize = colorCacheBits > 0 ? (1 << colorCacheBits) : 0;
        int greenAlphabetSize = NUM_LITERAL_CODES + NUM_LENGTH_CODES + cacheSize;

        int[][] greenFreq = new int[numGroups][greenAlphabetSize];
        int[][] redFreq = new int[numGroups][NUM_LITERAL_CODES];
        int[][] blueFreq = new int[numGroups][NUM_LITERAL_CODES];
        int[][] alphaFreq = new int[numGroups][NUM_LITERAL_CODES];
        int[][] distFreq = new int[numGroups][NUM_DISTANCE_CODES];

        int[] tokenKind = new int[pixelCount];
        int[] tokenLen = new int[pixelCount];
        int[] tokenValue = new int[pixelCount];
        int[] tokenGroup = new int[pixelCount];
        int tokenCount = 0;

        ColorCache cache = new ColorCache(colorCacheBits);
        int[] hashHead = LZ77.newHashHead();
        int[] hashPrev = LZ77.newHashPrev(pixelCount);

        int pos = 0;
        int x = 0;
        int y = 0;
        while (pos < pixelCount) {
            int group = (y >> tileBits) * prefixXsize + (x >> tileBits);
            int[] match = findBestMatchWithLookahead(pixelData, pos, pixelCount, hashHead, hashPrev);
            int matchLen = match[0];
            int matchDist = match[1];

            if (matchLen >= LZ77.MIN_MATCH) {
                int[] lenPrefix = LZ77.prefixEncode(matchLen);
                int lengthSymbol = lenPrefix[0];
                greenFreq[group][NUM_LITERAL_CODES + lengthSymbol]++;

                int planeCode = LZ77.distanceToPlaneCode(width, matchDist);
                int[] distPrefix = LZ77.prefixEncode(planeCode);
                int distSymbol = distPrefix[0];
                if (distSymbol >= NUM_DISTANCE_CODES)
                    throw new IllegalStateException(
                        "distance symbol " + distSymbol + " overflows 40-symbol alphabet");
                distFreq[group][distSymbol]++;

                tokenKind[tokenCount] = 1;
                tokenLen[tokenCount] = matchLen;
                tokenValue[tokenCount] = matchDist;
                tokenGroup[tokenCount] = group;
                tokenCount++;

                for (int k = 0; k < matchLen; k++) {
                    LZ77.updateHash(pixelData, pos + k, hashHead, hashPrev);
                    if (cacheSize > 0) cache.insert(pixelData[pos + k]);
                }
                pos += matchLen;
                // Advance (x, y) over matchLen pixels with row-wrap.
                int newX = x + matchLen;
                y += newX / width;
                x = newX % width;
            } else {
                int argb = pixelData[pos];
                int cacheIdx = -1;
                if (cacheSize > 0) {
                    int idx = cache.hashIndex(argb);
                    if (cache.lookup(idx) == argb) cacheIdx = idx;
                }

                if (cacheIdx >= 0) {
                    greenFreq[group][NUM_LITERAL_CODES + NUM_LENGTH_CODES + cacheIdx]++;
                    tokenKind[tokenCount] = 2;
                    tokenLen[tokenCount] = 1;
                    tokenValue[tokenCount] = cacheIdx;
                    tokenGroup[tokenCount] = group;
                    tokenCount++;
                } else {
                    greenFreq[group][(argb >> 8) & 0xFF]++;
                    redFreq[group][(argb >> 16) & 0xFF]++;
                    blueFreq[group][argb & 0xFF]++;
                    alphaFreq[group][(argb >> 24) & 0xFF]++;

                    tokenKind[tokenCount] = 0;
                    tokenLen[tokenCount] = 1;
                    tokenValue[tokenCount] = argb;
                    tokenGroup[tokenCount] = group;
                    tokenCount++;

                    if (cacheSize > 0) cache.insert(argb);
                }

                LZ77.updateHash(pixelData, pos, hashHead, hashPrev);
                pos++;
                if (++x == width) { x = 0; y++; }
            }
        }

        // Distance-alphabet non-empty guard per group. Every group gets its own 5 prefix
        // code declarations, so every one needs at least one used distance symbol.
        for (int g = 0; g < numGroups; g++) {
            boolean any = false;
            for (int f : distFreq[g]) if (f > 0) { any = true; break; }
            if (!any) distFreq[g][0] = 1;
        }

        PrefixCode[] greenPrefix = new PrefixCode[numGroups];
        PrefixCode[] redPrefix = new PrefixCode[numGroups];
        PrefixCode[] bluePrefix = new PrefixCode[numGroups];
        PrefixCode[] alphaPrefix = new PrefixCode[numGroups];
        PrefixCode[] distPrefix = new PrefixCode[numGroups];
        for (int g = 0; g < numGroups; g++) {
            greenPrefix[g] = buildPrefixCode(greenFreq[g], MAX_HUFFMAN_BITS);
            redPrefix[g] = buildPrefixCode(redFreq[g], MAX_HUFFMAN_BITS);
            bluePrefix[g] = buildPrefixCode(blueFreq[g], MAX_HUFFMAN_BITS);
            alphaPrefix[g] = buildPrefixCode(alphaFreq[g], MAX_HUFFMAN_BITS);
            distPrefix[g] = buildPrefixCode(distFreq[g], MAX_HUFFMAN_BITS);
        }

        // Meta-prefix sub-image: one pixel per tile, carrying the group index in the
        // (red, green) bytes. Since each tile is its own group here the mapping is
        // identity. Alpha is opaque; blue is unused.
        int[] metaPrefixImage = new int[numGroups];
        for (int g = 0; g < numGroups; g++)
            metaPrefixImage[g] = 0xFF000000 | (g << 8);
        encodeSubImage(writer, metaPrefixImage, prefixXsize);

        // Per-group 5-alphabet declarations.
        for (int g = 0; g < numGroups; g++) {
            writePrefixCode(writer, greenPrefix[g], greenAlphabetSize);
            writePrefixCode(writer, redPrefix[g], NUM_LITERAL_CODES);
            writePrefixCode(writer, bluePrefix[g], NUM_LITERAL_CODES);
            writePrefixCode(writer, alphaPrefix[g], NUM_LITERAL_CODES);
            writePrefixCode(writer, distPrefix[g], NUM_DISTANCE_CODES);
        }

        // Replay token stream using each token's owning group's Huffman tables.
        for (int t = 0; t < tokenCount; t++) {
            int g = tokenGroup[t];
            switch (tokenKind[t]) {
                case 0 -> {
                    int argb = tokenValue[t];
                    emitSymbol(writer, greenPrefix[g], (argb >> 8) & 0xFF);
                    emitSymbol(writer, redPrefix[g], (argb >> 16) & 0xFF);
                    emitSymbol(writer, bluePrefix[g], argb & 0xFF);
                    emitSymbol(writer, alphaPrefix[g], (argb >> 24) & 0xFF);
                }
                case 1 -> {
                    int matchLen = tokenLen[t];
                    int matchDist = tokenValue[t];
                    int[] lenPrefix = LZ77.prefixEncode(matchLen);
                    emitSymbol(writer, greenPrefix[g], NUM_LITERAL_CODES + lenPrefix[0]);
                    if (lenPrefix[1] > 0) writer.writeBits(lenPrefix[2], lenPrefix[1]);
                    int planeCode = LZ77.distanceToPlaneCode(width, matchDist);
                    int[] distPx = LZ77.prefixEncode(planeCode);
                    emitSymbol(writer, distPrefix[g], distPx[0]);
                    if (distPx[1] > 0) writer.writeBits(distPx[2], distPx[1]);
                }
                case 2 -> emitSymbol(writer, greenPrefix[g],
                    NUM_LITERAL_CODES + NUM_LENGTH_CODES + tokenValue[t]);
                default -> throw new IllegalStateException("unknown token kind " + tokenKind[t]);
            }
        }
    }

    // ---------------------------------------------------------------------
    //  Cross-color (COLOR_TRANSFORM) transform
    // ---------------------------------------------------------------------

    /**
     * Cross-color tile size exponent: a single 512x512 tile covers any VP8L image
     * (max 16384x16384 would need blockBits=14 but the spec caps at 9, so a single
     * tile only covers up to {@code 512x512}). For our tooltip-sized content one tile
     * suffices and keeps the meta-image at 1x1 - dramatically cheaper than the
     * predictor's 8x8 tiling. libwebp uses per-4x4 or per-16x16 tiles at higher methods
     * but a single-tile first-pass gets the bulk of the decorrelation benefit on
     * natural images for a fraction of the overhead.
     */
    private static final int CROSS_COLOR_BLOCK_BITS = 9;

    /**
     * Applies the cross-color (COLOR_TRANSFORM) to a fresh copy of {@code pixels}.
     * <p>
     * Computes per-tile decorrelation coefficients via a two-pass least-squares fit:
     * first derives {@code green_to_red}, {@code green_to_blue} from the covariance
     * of the raw signed channels; then fits {@code red_to_blue} against the red-vs-
     * post-green-subtract-blue residual. Applies the forward transform through
     * {@link VP8LTransform.ColorXform#forwardTransform} using the computed coefficients.
     * Returns {@code [residuals, transformData, {blockBits, blockWidth, blockHeight}]}.
     */
    private static int @NotNull [] @NotNull [] applyCrossColor(
        int @NotNull [] pixels,
        int width,
        int height
    ) {
        int blockBits = CROSS_COLOR_BLOCK_BITS;
        int blockSize = 1 << blockBits;
        int blockWidth = ((width - 1) >> blockBits) + 1;
        int blockHeight = ((height - 1) >> blockBits) + 1;
        int[] transformData = new int[blockWidth * blockHeight];

        for (int bY = 0; bY < blockHeight; bY++) {
            int y0 = bY << blockBits;
            int y1 = Math.min(y0 + blockSize, height);
            for (int bX = 0; bX < blockWidth; bX++) {
                int x0 = bX << blockBits;
                int x1 = Math.min(x0 + blockSize, width);
                int[] coeffs = computeCrossColorCoeffs(pixels, width, x0, y0, x1, y1);
                int gr = coeffs[0] & 0xFF;
                int gb = coeffs[1] & 0xFF;
                int rb = coeffs[2] & 0xFF;
                transformData[bY * blockWidth + bX] = 0xFF000000 | (rb << 16) | (gb << 8) | gr;
            }
        }

        int[] residuals = pixels.clone();
        new VP8LTransform.ColorXform(blockBits, transformData, blockWidth).forwardTransform(residuals, width, height);
        return new int[][] {
            residuals,
            transformData,
            { blockBits, blockWidth, blockHeight }
        };
    }

    /**
     * Computes {@code {greenToRed, greenToBlue, redToBlue}} coefficients for a tile
     * using two-pass greedy least-squares. All values are returned as signed ints in
     * {@code [-128, 127]}; callers serialise as signed bytes.
     * <p>
     * Pass 1: minimise sum of {@code |R - (gr * G) >> 5|^2} and similarly for blue,
     * yielding the optimum projection coefficient {@code gr = 32 * sum(R*G) / sum(G*G)}.
     * Pass 2: compute the blue residual after subtracting the green contribution, then
     * fit {@code rb = 32 * sum(B_resid * R) / sum(R*R)}. This is an approximation of
     * libwebp's brute-force search but costs {@code O(N)} per tile instead of
     * {@code O(256 * N)}.
     */
    private static int @NotNull [] computeCrossColorCoeffs(
        int @NotNull [] pixels,
        int width,
        int x0,
        int y0,
        int x1,
        int y1
    ) {
        long sumRG = 0, sumBG = 0, sumGG = 0;
        for (int y = y0; y < y1; y++) {
            int rowStart = y * width;
            for (int x = x0; x < x1; x++) {
                int argb = pixels[rowStart + x];
                int r = (byte) ((argb >> 16) & 0xFF);
                int g = (byte) ((argb >>  8) & 0xFF);
                int b = (byte) ( argb        & 0xFF);
                sumRG += (long) r * g;
                sumBG += (long) b * g;
                sumGG += (long) g * g;
            }
        }
        int gr = sumGG == 0 ? 0 : Math.clamp(Math.round(32.0 * sumRG / sumGG), -128, 127);
        int gb = sumGG == 0 ? 0 : Math.clamp(Math.round(32.0 * sumBG / sumGG), -128, 127);

        long sumBResR = 0, sumRR = 0;
        for (int y = y0; y < y1; y++) {
            int rowStart = y * width;
            for (int x = x0; x < x1; x++) {
                int argb = pixels[rowStart + x];
                int r = (byte) ((argb >> 16) & 0xFF);
                int g = (byte) ((argb >>  8) & 0xFF);
                int b = (byte) ( argb        & 0xFF);
                int bResid = b - ((gb * g) >> 5);
                sumBResR += (long) bResid * r;
                sumRR += (long) r * r;
            }
        }
        int rb = sumRR == 0 ? 0 : Math.clamp(Math.round(32.0 * sumBResR / sumRR), -128, 127);

        return new int[] { gr, gb, rb };
    }

    /**
     * Scores a candidate predictor mode over a tile as the sum of absolute signed-byte
     * residuals across all four ARGB channels. Smaller residual magnitudes concentrate
     * the per-channel histogram near zero and produce tighter Huffman codes, so the
     * minimum-SAD mode is a reasonable proxy for the minimum-entropy mode without the
     * cost of actually building a histogram per candidate.
     * <p>
     * For pixels on the top row or leftmost column the effective predictor is fixed
     * (left on first row, top on first column, black at {@code (0, 0)}) per the VP8L
     * spec - mode choice only affects the interior {@code x >= 1, y >= 1} positions.
     */
    private static long scoreTileUnderMode(
        int @NotNull [] pixels,
        int width,
        int x0,
        int y0,
        int x1,
        int y1,
        int mode
    ) {
        long sad = 0;
        for (int y = y0; y < y1; y++) {
            int rowStart = y * width;
            for (int x = x0; x < x1; x++) {
                int predicted;
                if (x == 0 && y == 0) predicted = VP8LTransform.Predictor.ARGB_BLACK;
                else if (y == 0) predicted = pixels[rowStart + x - 1];
                else if (x == 0) predicted = pixels[rowStart - width];
                else predicted = VP8LTransform.Predictor.predict(mode, pixels, x, y, width);
                int src = pixels[rowStart + x];
                int dA = (byte) (((src >>> 24) & 0xFF) - ((predicted >>> 24) & 0xFF));
                int dR = (byte) (((src >>  16) & 0xFF) - ((predicted >>  16) & 0xFF));
                int dG = (byte) (((src >>   8) & 0xFF) - ((predicted >>   8) & 0xFF));
                int dB = (byte) (( src         & 0xFF) - ( predicted         & 0xFF));
                sad += Math.abs(dA) + Math.abs(dR) + Math.abs(dG) + Math.abs(dB);
            }
        }
        return sad;
    }

    // ---------------------------------------------------------------------
    //  Shared pixel-stream body (LZ77 + prefix codes + token emission)
    // ---------------------------------------------------------------------

    /**
     * Writes the entropy-coded pixel body: a first pass walks the pixels with hash-chain
     * LZ77 match finding and records a token stream plus per-alphabet frequency tables;
     * the five prefix codes (G+length, R, B, A, distance) are declared; then the token
     * stream is replayed into the bitstream.
     * <p>
     * The caller is responsible for having emitted the transform / color-cache /
     * meta-prefix header bits before invoking this helper. On the main image body
     * {@code width} is the post-transform "effective" width (equal to
     * {@code ceil(width / pixelsPerByte)} when a sub-bit-packed color-indexing
     * transform was applied); on sub-images it is the sub-image's own width.
     * <p>
     * Libwebp's {@code BackwardReferencesLz77} additionally does a "longest-reach
     * lookahead" that can truncate a match if a later match reaches further; we skip
     * that here (first-pass port).
     */
    private static void encodePixelStreamBody(
        @NotNull BitWriter writer,
        int @NotNull [] pixelData,
        int width,
        int colorCacheBits
    ) {
        int pixelCount = pixelData.length;
        int cacheSize = colorCacheBits > 0 ? (1 << colorCacheBits) : 0;
        int greenAlphabetSize = NUM_LITERAL_CODES + NUM_LENGTH_CODES + cacheSize;

        int[] greenFreq = new int[greenAlphabetSize];
        int[] redFreq = new int[NUM_LITERAL_CODES];
        int[] blueFreq = new int[NUM_LITERAL_CODES];
        int[] alphaFreq = new int[NUM_LITERAL_CODES];
        int[] distFreq = new int[NUM_DISTANCE_CODES];

        // Token kinds: 0 = LITERAL (value=argb), 1 = COPY (len=matchLen, value=dist),
        // 2 = CACHE_INDEX (value=cacheIdx). The replay pass writes the bits recorded
        // here; cache state is only maintained during the token-building pass.
        int[] tokenKind = new int[pixelCount];
        int[] tokenLen = new int[pixelCount];
        int[] tokenValue = new int[pixelCount];
        int tokenCount = 0;

        ColorCache cache = new ColorCache(colorCacheBits);

        int[] hashHead = LZ77.newHashHead();
        int[] hashPrev = LZ77.newHashPrev(pixelCount);
        int pos = 0;
        while (pos < pixelCount) {
            int[] match = findBestMatchWithLookahead(pixelData, pos, pixelCount, hashHead, hashPrev);
            int matchLen = match[0];
            int matchDist = match[1];

            if (matchLen >= LZ77.MIN_MATCH) {
                // Emit COPY token. Length is encoded via the 24-symbol length
                // alphabet at green-alphabet positions 256..279; distance is
                // encoded via the 40-symbol distance alphabet after mapping
                // through the 2-D plane-code space.
                int[] lenPrefix = LZ77.prefixEncode(matchLen);
                int lengthSymbol = lenPrefix[0];
                greenFreq[NUM_LITERAL_CODES + lengthSymbol]++;

                int planeCode = LZ77.distanceToPlaneCode(width, matchDist);
                int[] distPrefix = LZ77.prefixEncode(planeCode);
                int distSymbol = distPrefix[0];
                if (distSymbol >= NUM_DISTANCE_CODES)
                    throw new IllegalStateException(
                        "distance symbol " + distSymbol + " overflows 40-symbol alphabet; "
                        + "dist=" + matchDist + " planeCode=" + planeCode);
                distFreq[distSymbol]++;

                tokenKind[tokenCount] = 1;
                tokenLen[tokenCount] = matchLen;
                tokenValue[tokenCount] = matchDist;
                tokenCount++;

                // Advance the hash chain + color cache over every position in the
                // match. The decoder inserts each copied pixel into the cache as it
                // materialises, so the encoder's cache state must track the same
                // sequence of inserts to keep future cache-hit checks in sync.
                for (int k = 0; k < matchLen; k++) {
                    LZ77.updateHash(pixelData, pos + k, hashHead, hashPrev);
                    if (cacheSize > 0) cache.insert(pixelData[pos + k]);
                }
                pos += matchLen;
            } else {
                int argb = pixelData[pos];

                // Color-cache shortcut: if this ARGB already occupies its hash slot
                // from an earlier literal or copy, emit a cache-index symbol (living
                // in green alphabet positions 280..280+cacheSize-1) instead of four
                // separate G/R/B/A literals. Decoder does NOT insert on a cache-index
                // read, so skip the insert here to keep states in sync.
                int cacheIdx = -1;
                if (cacheSize > 0) {
                    int idx = cache.hashIndex(argb);
                    if (cache.lookup(idx) == argb) cacheIdx = idx;
                }

                if (cacheIdx >= 0) {
                    greenFreq[NUM_LITERAL_CODES + NUM_LENGTH_CODES + cacheIdx]++;

                    tokenKind[tokenCount] = 2;
                    tokenLen[tokenCount] = 1;
                    tokenValue[tokenCount] = cacheIdx;
                    tokenCount++;
                } else {
                    greenFreq[(argb >> 8) & 0xFF]++;
                    redFreq[(argb >> 16) & 0xFF]++;
                    blueFreq[argb & 0xFF]++;
                    alphaFreq[(argb >> 24) & 0xFF]++;

                    tokenKind[tokenCount] = 0;
                    tokenLen[tokenCount] = 1;
                    tokenValue[tokenCount] = argb;
                    tokenCount++;

                    if (cacheSize > 0) cache.insert(argb);
                }

                LZ77.updateHash(pixelData, pos, hashHead, hashPrev);
                pos++;
            }
        }

        // Guarantee the distance alphabet is non-empty.
        ensureDistanceNonEmpty(distFreq);

        // -------------------------------------------------------------------
        //  Trace-backwards cost DP (task 8): re-decide every (literal | match)
        //  choice globally using Huffman bit costs from the initial greedy pass
        //  as a guide, then rebuild Huffman from the optimized token stream.
        // -------------------------------------------------------------------

        // Phase 1: build INITIAL Huffman codes (cost estimator).
        PrefixCode greenPrefixEst = buildPrefixCode(greenFreq, MAX_HUFFMAN_BITS);
        PrefixCode redPrefixEst = buildPrefixCode(redFreq, MAX_HUFFMAN_BITS);
        PrefixCode bluePrefixEst = buildPrefixCode(blueFreq, MAX_HUFFMAN_BITS);
        PrefixCode alphaPrefixEst = buildPrefixCode(alphaFreq, MAX_HUFFMAN_BITS);
        PrefixCode distPrefixEst = buildPrefixCode(distFreq, MAX_HUFFMAN_BITS);

        // Phase 2: pre-compute best match at every pixel position. The hash chain
        // is fully populated from the initial LZ77 pass so findMatch at any pos
        // finds the globally-best backward reference.
        int[] bestMatchLen = new int[pixelCount];
        int[] bestMatchDist = new int[pixelCount];
        for (int i = 0; i < pixelCount; i++) {
            int rem = pixelCount - i;
            int maxL = Math.min(rem, LZ77.MAX_LENGTH);
            int[] m = LZ77.findMatch(pixelData, i, maxL, hashHead, hashPrev);
            bestMatchLen[i] = m[0];
            bestMatchDist[i] = m[1];
        }

        // Phase 2b: estimate cache state. Insert every pixel (conservative:
        // over-estimates cache availability, matching libwebp's approach).
        int[] cacheHitIdx = null;
        if (cacheSize > 0) {
            cacheHitIdx = new int[pixelCount];
            java.util.Arrays.fill(cacheHitIdx, -1);
            ColorCache dpCacheEst = new ColorCache(colorCacheBits);
            for (int i = 0; i < pixelCount; i++) {
                int argb = pixelData[i];
                int idx = dpCacheEst.hashIndex(argb);
                if (dpCacheEst.lookup(idx) == argb) cacheHitIdx[i] = idx;
                dpCacheEst.insert(argb);
            }
        }

        // Phase 3: DP backward. dpCost[i] = minimum estimated bits to encode
        // pixels[i..N-1]. At each i we consider literal, cache-index, and all
        // match lengths from MIN_MATCH to bestMatchLen[i].
        long[] dpCost = new long[pixelCount + 1];
        int[] dpNext = new int[pixelCount];     // next position after optimal choice at i
        int[] dpChoiceKind = new int[pixelCount]; // 0=lit, 1=match, 2=cache
        int[] dpChoiceLen = new int[pixelCount];
        int[] dpChoiceValue = new int[pixelCount]; // argb, dist, or cacheIdx

        dpCost[pixelCount] = 0;
        for (int i = pixelCount - 1; i >= 0; i--) {
            int argb = pixelData[i];

            // Literal cost
            long litCost = (long) symbolBitCost(greenPrefixEst, (argb >> 8) & 0xFF)
                + symbolBitCost(redPrefixEst, (argb >> 16) & 0xFF)
                + symbolBitCost(bluePrefixEst, argb & 0xFF)
                + symbolBitCost(alphaPrefixEst, (argb >> 24) & 0xFF)
                + dpCost[i + 1];
            dpCost[i] = litCost;
            dpChoiceKind[i] = 0;
            dpChoiceLen[i] = 1;
            dpChoiceValue[i] = argb;
            dpNext[i] = i + 1;

            // Cache-index cost
            if (cacheHitIdx != null && cacheHitIdx[i] >= 0) {
                long cacheCost = (long) symbolBitCost(greenPrefixEst,
                    NUM_LITERAL_CODES + NUM_LENGTH_CODES + cacheHitIdx[i])
                    + dpCost[i + 1];
                if (cacheCost < dpCost[i]) {
                    dpCost[i] = cacheCost;
                    dpChoiceKind[i] = 2;
                    dpChoiceLen[i] = 1;
                    dpChoiceValue[i] = cacheHitIdx[i];
                    dpNext[i] = i + 1;
                }
            }

            // Match costs at every valid length
            int mLen = bestMatchLen[i];
            int mDist = bestMatchDist[i];
            if (mLen >= LZ77.MIN_MATCH) {
                int planeCode = LZ77.distanceToPlaneCode(width, mDist);
                int[] distPx = LZ77.prefixEncode(planeCode);
                int distBits = symbolBitCost(distPrefixEst, distPx[0]) + distPx[1];
                for (int L = LZ77.MIN_MATCH; L <= mLen; L++) {
                    int[] lenPx = LZ77.prefixEncode(L);
                    long matchCost = (long) symbolBitCost(greenPrefixEst,
                        NUM_LITERAL_CODES + lenPx[0]) + lenPx[1]
                        + distBits + dpCost[i + L];
                    if (matchCost < dpCost[i]) {
                        dpCost[i] = matchCost;
                        dpChoiceKind[i] = 1;
                        dpChoiceLen[i] = L;
                        dpChoiceValue[i] = mDist;
                        dpNext[i] = i + L;
                    }
                }
            }
        }

        // Phase 4: forward walk → extract DP token stream into SEPARATE arrays
        // (keep greedy tokens in tokenKind/Len/Value intact for comparison).
        int dpTokenCount = 0;
        int[] dpTokenKind = new int[pixelCount];
        int[] dpTokenLen = new int[pixelCount];
        int[] dpTokenValue = new int[pixelCount];
        int pi = 0;
        while (pi < pixelCount) {
            dpTokenKind[dpTokenCount] = dpChoiceKind[pi];
            dpTokenLen[dpTokenCount] = dpChoiceLen[pi];
            dpTokenValue[dpTokenCount] = dpChoiceValue[pi];
            dpTokenCount++;
            pi = dpNext[pi];
        }

        // Phase 5: compare greedy vs DP by computing each token set's bit total
        // under its OWN final Huffman. The DP's estimated cost (dpCost[0]) is always
        // optimal under the INITIAL Huffman, but the actual output size depends on
        // the REBUILT Huffman which can be worse if the DP token distribution diverges
        // from the greedy distribution (estimator drift). This "actual bit cost"
        // comparison is the definitive gate.
        int[] dpGreenFreq = new int[greenAlphabetSize];
        int[] dpRedFreq = new int[NUM_LITERAL_CODES];
        int[] dpBlueFreq = new int[NUM_LITERAL_CODES];
        int[] dpAlphaFreq = new int[NUM_LITERAL_CODES];
        int[] dpDistFreq = new int[NUM_DISTANCE_CODES];
        for (int t = 0; t < dpTokenCount; t++) {
            switch (dpTokenKind[t]) {
                case 0 -> { int a = dpTokenValue[t]; dpGreenFreq[(a >> 8) & 0xFF]++; dpRedFreq[(a >> 16) & 0xFF]++; dpBlueFreq[a & 0xFF]++; dpAlphaFreq[(a >> 24) & 0xFF]++; }
                case 1 -> { int[] lp = LZ77.prefixEncode(dpTokenLen[t]); dpGreenFreq[NUM_LITERAL_CODES + lp[0]]++; int pc = LZ77.distanceToPlaneCode(width, dpTokenValue[t]); int[] dp2 = LZ77.prefixEncode(pc); dpDistFreq[dp2[0]]++; }
                case 2 -> dpGreenFreq[NUM_LITERAL_CODES + NUM_LENGTH_CODES + dpTokenValue[t]]++;
            }
        }
        ensureDistanceNonEmpty(dpDistFreq);
        PrefixCode dpGreen = buildPrefixCode(dpGreenFreq, MAX_HUFFMAN_BITS);
        PrefixCode dpRed = buildPrefixCode(dpRedFreq, MAX_HUFFMAN_BITS);
        PrefixCode dpBlue = buildPrefixCode(dpBlueFreq, MAX_HUFFMAN_BITS);
        PrefixCode dpAlpha = buildPrefixCode(dpAlphaFreq, MAX_HUFFMAN_BITS);
        PrefixCode dpDist = buildPrefixCode(dpDistFreq, MAX_HUFFMAN_BITS);

        long dpBitTotal = computeTokenBitTotal(dpTokenKind, dpTokenLen, dpTokenValue, dpTokenCount,
            dpGreen, dpRed, dpBlue, dpAlpha, dpDist, width);
        long greedyBitTotal = computeTokenBitTotal(tokenKind, tokenLen, tokenValue, tokenCount,
            greenPrefixEst, redPrefixEst, bluePrefixEst, alphaPrefixEst, distPrefixEst, width);

        boolean useDp = dpBitTotal < greedyBitTotal;
        if (useDp) {
            tokenCount = dpTokenCount;
            System.arraycopy(dpTokenKind, 0, tokenKind, 0, tokenCount);
            System.arraycopy(dpTokenLen, 0, tokenLen, 0, tokenCount);
            System.arraycopy(dpTokenValue, 0, tokenValue, 0, tokenCount);
        }

        // Phase 6: rebuild final frequency tables from the surviving tokens.
        // The token arrays already hold either greedy (useDp=false) or DP
        // (useDp=true) choices — no further DP iteration; the actual-cost gate
        // above is the definitive decision.
        java.util.Arrays.fill(greenFreq, 0);
        java.util.Arrays.fill(redFreq, 0);
        java.util.Arrays.fill(blueFreq, 0);
        java.util.Arrays.fill(alphaFreq, 0);
        java.util.Arrays.fill(distFreq, 0);
        for (int t = 0; t < tokenCount; t++) {
            switch (tokenKind[t]) {
                case 0 -> { int a = tokenValue[t]; greenFreq[(a >> 8) & 0xFF]++; redFreq[(a >> 16) & 0xFF]++; blueFreq[a & 0xFF]++; alphaFreq[(a >> 24) & 0xFF]++; }
                case 1 -> { int[] lp = LZ77.prefixEncode(tokenLen[t]); greenFreq[NUM_LITERAL_CODES + lp[0]]++; int pc = LZ77.distanceToPlaneCode(width, tokenValue[t]); int[] dPx = LZ77.prefixEncode(pc); distFreq[dPx[0]]++; }
                case 2 -> greenFreq[NUM_LITERAL_CODES + NUM_LENGTH_CODES + tokenValue[t]]++;
            }
        }
        ensureDistanceNonEmpty(distFreq);

        PrefixCode greenPrefix = buildPrefixCode(greenFreq, MAX_HUFFMAN_BITS);
        PrefixCode redPrefix = buildPrefixCode(redFreq, MAX_HUFFMAN_BITS);
        PrefixCode bluePrefix = buildPrefixCode(blueFreq, MAX_HUFFMAN_BITS);
        PrefixCode alphaPrefix = buildPrefixCode(alphaFreq, MAX_HUFFMAN_BITS);
        PrefixCode distPrefix = buildPrefixCode(distFreq, MAX_HUFFMAN_BITS);

        // --- Five prefix-code declarations (G, R, B, A, distance) ---
        writePrefixCode(writer, greenPrefix, greenAlphabetSize);
        writePrefixCode(writer, redPrefix, NUM_LITERAL_CODES);
        writePrefixCode(writer, bluePrefix, NUM_LITERAL_CODES);
        writePrefixCode(writer, alphaPrefix, NUM_LITERAL_CODES);
        writePrefixCode(writer, distPrefix, NUM_DISTANCE_CODES);

        // --- Entropy-coded pixel data, replayed from the token stream ---
        for (int t = 0; t < tokenCount; t++) {
            switch (tokenKind[t]) {
                case 0 -> {
                    int argb = tokenValue[t];
                    emitSymbol(writer, greenPrefix, (argb >> 8) & 0xFF);
                    emitSymbol(writer, redPrefix, (argb >> 16) & 0xFF);
                    emitSymbol(writer, bluePrefix, argb & 0xFF);
                    emitSymbol(writer, alphaPrefix, (argb >> 24) & 0xFF);
                }
                case 1 -> {
                    int matchLen = tokenLen[t];
                    int matchDist = tokenValue[t];
                    int[] lenPrefix = LZ77.prefixEncode(matchLen);
                    emitSymbol(writer, greenPrefix, NUM_LITERAL_CODES + lenPrefix[0]);
                    if (lenPrefix[1] > 0) writer.writeBits(lenPrefix[2], lenPrefix[1]);

                    int planeCode = LZ77.distanceToPlaneCode(width, matchDist);
                    int[] distPx = LZ77.prefixEncode(planeCode);
                    emitSymbol(writer, distPrefix, distPx[0]);
                    if (distPx[1] > 0) writer.writeBits(distPx[2], distPx[1]);
                }
                case 2 -> emitSymbol(writer, greenPrefix,
                    NUM_LITERAL_CODES + NUM_LENGTH_CODES + tokenValue[t]);
                default -> throw new IllegalStateException("unknown token kind " + tokenKind[t]);
            }
        }
    }

    /**
     * Returns the bit cost of emitting {@code symbol} through {@code prefix}. For simple
     * 1-symbol codes the cost is 0 (degenerate), for 2-symbol codes it is 1, and for
     * normal codes it is the Huffman code length assigned to that symbol.
     */
    private static int symbolBitCost(@NotNull PrefixCode prefix, int symbol) {
        if (prefix.isSimple) return prefix.simpleSymbols.length == 1 ? 0 : 1;
        return symbol < prefix.normalLengths.length ? prefix.normalLengths[symbol] : 0;
    }

    /** Guarantees at least one used symbol in the distance alphabet. */
    private static void ensureDistanceNonEmpty(int @NotNull [] distFreq) {
        for (int f : distFreq) if (f > 0) return;
        distFreq[0] = 1;
    }

    /** Sums the Huffman bit cost of every token in a stream under the given prefix codes. */
    private static long computeTokenBitTotal(
        int @NotNull [] kind, int @NotNull [] len, int @NotNull [] value, int count,
        @NotNull PrefixCode green, @NotNull PrefixCode red,
        @NotNull PrefixCode blue, @NotNull PrefixCode alpha,
        @NotNull PrefixCode dist, int width
    ) {
        long total = 0;
        for (int t = 0; t < count; t++) {
            switch (kind[t]) {
                case 0 -> {
                    int a = value[t];
                    total += symbolBitCost(green, (a >> 8) & 0xFF)
                        + symbolBitCost(red, (a >> 16) & 0xFF)
                        + symbolBitCost(blue, a & 0xFF)
                        + symbolBitCost(alpha, (a >> 24) & 0xFF);
                }
                case 1 -> {
                    int[] lp = LZ77.prefixEncode(len[t]);
                    total += symbolBitCost(green, NUM_LITERAL_CODES + lp[0]) + lp[1];
                    int pc = LZ77.distanceToPlaneCode(width, value[t]);
                    int[] dp = LZ77.prefixEncode(pc);
                    total += symbolBitCost(dist, dp[0]) + dp[1];
                }
                case 2 -> total += symbolBitCost(green,
                    NUM_LITERAL_CODES + NUM_LENGTH_CODES + value[t]);
            }
        }
        return total;
    }

    // ---------------------------------------------------------------------
    //  Prefix code construction and emission
    // ---------------------------------------------------------------------

    /**
     * Builds a valid VP8L prefix code for an alphabet given its symbol frequencies.
     * <p>
     * The result is tagged as either {@code simple} (for 1- or 2-used-symbol alphabets
     * where a Huffman tree would degenerate) or {@code normal} (3+ symbols with a
     * proper canonical Huffman code). The write path uses the tag to choose the matching
     * spec-defined sub-format.
     */
    private static @NotNull PrefixCode buildPrefixCode(int @NotNull [] freq, int maxBits) {
        int[] usedSymbols = new int[freq.length];
        int usedCount = 0;
        for (int i = 0; i < freq.length; i++)
            if (freq[i] > 0) usedSymbols[usedCount++] = i;

        // Simple-mode prefix codes can only represent symbols 0..255 (first
        // symbol: 1 or 8 bits, second symbol: 8 bits per VP8L spec). The
        // extended green alphabet's length codes live at 256..279 so any
        // alphabet with a used symbol > 255 MUST use normal mode, even at
        // one or two used symbols.
        boolean anyLargeSymbol = false;
        for (int i = 0; i < usedCount; i++)
            if (usedSymbols[i] > 255) { anyLargeSymbol = true; break; }

        if (!anyLargeSymbol) {
            if (usedCount <= 1) {
                int sym = usedCount == 1 ? usedSymbols[0] : 0;
                return PrefixCode.simple(new int[]{ sym });
            }
            if (usedCount == 2)
                return PrefixCode.simple(new int[]{ usedSymbols[0], usedSymbols[1] });
        }

        // Normal-mode path. For {@code usedCount <= 1} at symbol > 255 we
        // still need a valid Huffman tree - add a virtual second used symbol
        // at position 0 so the builder produces a 2-leaf tree with both
        // lengths = 1 (decoder accepts it; the virtual symbol never gets
        // emitted because the encoder only writes the real ones).
        int[] workFreq = freq;
        if (usedCount == 1) {
            workFreq = freq.clone();
            int dummy = usedSymbols[0] == 0 ? 1 : 0;
            if (workFreq[dummy] == 0) workFreq[dummy] = 1;
        }
        int[] lengths = buildHuffmanLengths(workFreq, maxBits);
        int[] codes = buildCanonicalCodes(lengths);
        return PrefixCode.normal(lengths, codes);
    }

    /**
     * Writes a prefix-code declaration to the bitstream. {@link PrefixCode#simple}
     * variants use the 1-2 symbol shortcut; {@link PrefixCode#normal} emits the CLC
     * header plus per-symbol lengths.
     */
    private static void writePrefixCode(
        @NotNull BitWriter writer,
        @NotNull PrefixCode prefix,
        int alphabetSize
    ) {
        if (prefix.isSimple) {
            writer.writeBit(1); // is_simple = 1
            int numSymbols = prefix.simpleSymbols.length;
            writer.writeBit(numSymbols - 1); // 0 = 1 symbol, 1 = 2 symbols

            int s0 = prefix.simpleSymbols[0];
            boolean firstIs8Bits = s0 > 1;
            writer.writeBit(firstIs8Bits ? 1 : 0);
            writer.writeBits(s0, firstIs8Bits ? 8 : 1);

            if (numSymbols == 2)
                writer.writeBits(prefix.simpleSymbols[1], 8);
            return;
        }

        writer.writeBit(0); // is_simple = 0

        int[] codeLengths = prefix.normalLengths;

        // Build CLC from this alphabet's lengths only (symbols 0..15; no RLE shortcuts).
        int[] clcFreq = new int[CODE_LENGTH_CODES];
        for (int i = 0; i < alphabetSize; i++) {
            int len = i < codeLengths.length ? codeLengths[i] : 0;
            if (len >= 0 && len < CODE_LENGTH_CODES) clcFreq[len]++;
        }

        // CLC lengths are themselves written as raw 3-bit fields, so they cannot exceed
        // 7 - pass that limit down or the Huffman builder may produce deeper codes that
        // would overflow the 3-bit slot and desync libwebp.
        PrefixCode clc = buildPrefixCode(clcFreq, CLC_MAX_BITS);

        // CLC is written with raw 3-bit lengths regardless of CLC simple/normal - the
        // spec defines a fixed 19-slot layout driven by CODE_LENGTH_ORDER.
        int[] clcLengths = clc.isSimple ? expandSimpleCLCLengths(clc) : clc.normalLengths;
        int[] clcCodes = clc.isSimple ? buildCanonicalCodes(clcLengths) : clc.normalCodes;

        int numCodes = CODE_LENGTH_CODES;
        while (numCodes > 4 && clcLengths[CODE_LENGTH_ORDER[numCodes - 1]] == 0)
            numCodes--;

        writer.writeBits(numCodes - 4, 4);
        for (int i = 0; i < numCodes; i++) {
            int len = clcLengths[CODE_LENGTH_ORDER[i]];
            if (len > CLC_MAX_BITS)
                throw new IllegalStateException("CLC length " + len + " exceeds 7-bit limit");
            writer.writeBits(len, 3);
        }

        // max_symbol is implicit (use full alphabet).
        writer.writeBit(0);

        // Degenerate-CLC short-circuit: a single used CLC length means every main-alphabet
        // symbol has the same length (unused symbols would register a second CLC length
        // at zero). The decoder's {@code HuffmanTree.fromCodeLengths} treats a
        // single-nonzero-length tree as degenerate and reads zero bits per symbol - so the
        // encoder must also emit zero bits per main symbol to keep the bitstream in sync.
        // This case only fires when {@code alphabetSize} is a power of two AND every
        // symbol is used (e.g., an R/B/A alphabet where all 256 literals appear, producing
        // a uniform 8-bit flat Huffman).
        boolean clcDegenerate = clc.isSimple && clc.simpleSymbols.length == 1;

        if (!clcDegenerate) {
            for (int i = 0; i < alphabetSize; i++) {
                int len = i < codeLengths.length ? codeLengths[i] : 0;
                writer.writeBits(clcCodes[len], clcLengths[len]);
            }
        }
    }

    /**
     * Simple-mode CLCs would normally produce only a symbols-array with no lengths, but
     * the outer code path treats CLC as a normal 19-slot code. For the rare cases where
     * the CLC itself collapses to one or two literal-length symbols we synthesize the
     * canonical length vector that matches.
     */
    private static int @NotNull [] expandSimpleCLCLengths(@NotNull PrefixCode simple) {
        int[] out = new int[CODE_LENGTH_CODES];
        int n = simple.simpleSymbols.length;
        if (n == 1) {
            // A single literal-length symbol cannot appear alone - the encoder must use
            // at least one other symbol. This branch only fires from the buildPrefixCode
            // recursion when every non-zero CLC frequency collapses to one bin, which is
            // impossible when the outer alphabet has any symbols at all. Guard anyway:
            out[simple.simpleSymbols[0]] = 1;
            return out;
        }
        out[simple.simpleSymbols[0]] = 1;
        out[simple.simpleSymbols[1]] = 1;
        return out;
    }

    /**
     * Writes one symbol of the main bitstream using the matching prefix code. Simple
     * codes emit nothing for 1-symbol alphabets (the decoder already knows the symbol)
     * and one bit (0 or 1) for 2-symbol alphabets.
     */
    private static void emitSymbol(@NotNull BitWriter writer, @NotNull PrefixCode prefix, int symbol) {
        if (prefix.isSimple) {
            if (prefix.simpleSymbols.length == 1) return;
            int bit = prefix.simpleSymbols[0] == symbol ? 0 : 1;
            writer.writeBit(bit);
            return;
        }

        int len = symbol < prefix.normalLengths.length ? prefix.normalLengths[symbol] : 0;
        if (len == 0) return;
        writer.writeBits(prefix.normalCodes[symbol], len);
    }

    // ---------------------------------------------------------------------
    //  Huffman construction
    // ---------------------------------------------------------------------

    /**
     * Builds a valid canonical Huffman code length vector for the given symbol
     * frequencies, capped at {@code maxBits}. Implements classic Huffman via a
     * heap-sort-based merging pass, then rebalances the tree to satisfy
     * {@code length <= maxBits} while preserving Kraft equality (required for every
     * alphabet with >= 2 used symbols).
     */
    private static int @NotNull [] buildHuffmanLengths(int @NotNull [] freq, int maxBits) {
        int n = freq.length;
        int[] lengths = new int[n];

        int used = 0;
        for (int f : freq) if (f > 0) used++;
        if (used == 0) return lengths;
        if (used == 1) {
            // Caller should have chosen simple mode, but guard: length=1 is invalid,
            // length=0 preserves Kraft for a 1-symbol tree.
            for (int i = 0; i < n; i++) if (freq[i] > 0) lengths[i] = 0;
            return lengths;
        }

        // Classic Huffman: represent each "node" as an index into two parallel arrays.
        // freq[i] and parent[i]; leaves are 0..n-1, internal nodes are n..2n-2.
        int maxNodes = used * 2 - 1;
        long[] nodeFreq = new long[maxNodes];
        int[] parent = new int[maxNodes];
        int[] leafIndex = new int[used];
        int[] leafSymbol = new int[used];
        int li = 0;
        for (int i = 0; i < n; i++) {
            if (freq[i] > 0) {
                nodeFreq[li] = freq[i];
                leafIndex[li] = li;
                leafSymbol[li] = i;
                parent[li] = -1;
                li++;
            }
        }

        // Min-heap keyed by frequency, storing node indices.
        int[] heap = new int[maxNodes + 1];
        int heapSize = 0;
        for (int i = 0; i < used; i++) heapSize = heapInsert(heap, heapSize, i, nodeFreq);

        int nextInternal = used;
        while (heapSize > 1) {
            int a = heapExtract(heap, heapSize--, nodeFreq);
            int b = heapExtract(heap, heapSize--, nodeFreq);
            int merged = nextInternal++;
            nodeFreq[merged] = nodeFreq[a] + nodeFreq[b];
            parent[merged] = -1;
            parent[a] = merged;
            parent[b] = merged;
            heapSize = heapInsert(heap, heapSize, merged, nodeFreq);
        }

        // Assign lengths by walking from each leaf to the root.
        for (int i = 0; i < used; i++) {
            int depth = 0;
            int cur = i;
            while (parent[cur] != -1) {
                depth++;
                cur = parent[cur];
            }
            lengths[leafSymbol[i]] = depth;
        }

        enforceMaxLengths(lengths, maxBits);
        return lengths;
    }

    /**
     * Caps every code length at {@code maxBits} while maintaining Kraft equality
     * ({@code sum(2^-len_i) == 1}). Without this, the canonical code builder generates
     * colliding prefixes and both our own decoder and {@code libwebp} reject the stream
     * with "Invalid Huffman code".
     * <p>
     * Algorithm:
     * <ol>
     *   <li>Clamp every length over {@code maxBits} to {@code maxBits}; Kraft overshoots.</li>
     *   <li>Walk the bit-length histogram, lengthening the shortest populated length by
     *       one repeatedly until Kraft returns to 1. Each lengthening subtracts
     *       {@code 2^-(newLen)} from Kraft so we integer-track the deficit.</li>
     *   <li>If lengthening overshoots the target (possible when the only short length has
     *       a high weight), shorten a long-length symbol to compensate.</li>
     * </ol>
     * Operates entirely on integer math scaled by {@code 2^maxBits} to avoid float drift.
     */
    private static void enforceMaxLengths(int @NotNull [] lengths, int maxBits) {
        int n = lengths.length;

        int actualMax = 0;
        for (int len : lengths) if (len > actualMax) actualMax = len;
        if (actualMax <= maxBits) return;

        // 1. Clamp every overflowed length to maxBits.
        for (int i = 0; i < n; i++)
            if (lengths[i] > maxBits) lengths[i] = maxBits;

        int[] bl = new int[maxBits + 1];
        for (int l : lengths) if (l > 0) bl[l]++;

        // Integer Kraft, scaled by 2^maxBits. Target = 2^maxBits.
        long target = 1L << maxBits;
        long kraft = 0;
        for (int l = 1; l <= maxBits; l++)
            kraft += (long) bl[l] << (maxBits - l);

        // 2. While Kraft > target, lengthen the shortest populated code.
        while (kraft > target) {
            int pickLen = -1;
            for (int l = 1; l < maxBits; l++) {
                if (bl[l] > 0) { pickLen = l; break; }
            }
            if (pickLen < 0) {
                // Every symbol already at maxBits and Kraft still too high - the alphabet
                // has more non-zero symbols than fit at this max length (2^maxBits).
                throw new IllegalStateException("VP8L alphabet too large for length cap " + maxBits);
            }
            bl[pickLen]--;
            bl[pickLen + 1]++;
            for (int i = 0; i < n; i++) {
                if (lengths[i] == pickLen) { lengths[i]++; break; }
            }
            kraft -= 1L << (maxBits - pickLen - 1);
        }

        // 3. If we undershoot (lengthening a short code removed more than needed), shorten
        //    a long code one step at a time.
        while (kraft < target) {
            int pickLen = -1;
            for (int l = maxBits; l > 1; l--) {
                if (bl[l] > 0) { pickLen = l; break; }
            }
            if (pickLen < 0) break;
            long delta = 1L << (maxBits - pickLen);
            if (kraft + delta > target) break; // would overshoot; tree is as close as we can get
            bl[pickLen]--;
            bl[pickLen - 1]++;
            for (int i = 0; i < n; i++) {
                if (lengths[i] == pickLen) { lengths[i]--; break; }
            }
            kraft += delta;
        }
    }

    /**
     * Builds canonical Huffman codes from a length vector. Bits are reversed per symbol
     * because {@link BitWriter} emits LSB-first as VP8L requires.
     */
    private static int @NotNull [] buildCanonicalCodes(int @NotNull [] codeLengths) {
        int n = codeLengths.length;
        int[] codes = new int[n];
        int maxLen = 0;
        for (int len : codeLengths) if (len > maxLen) maxLen = len;
        if (maxLen == 0) return codes;

        int[] blCount = new int[maxLen + 1];
        for (int len : codeLengths) if (len > 0) blCount[len]++;

        int[] nextCode = new int[maxLen + 1];
        int code = 0;
        for (int bits = 1; bits <= maxLen; bits++) {
            code = (code + blCount[bits - 1]) << 1;
            nextCode[bits] = code;
        }

        for (int i = 0; i < n; i++) {
            int len = codeLengths[i];
            if (len > 0)
                codes[i] = reverseBits(nextCode[len]++, len);
        }
        return codes;
    }

    private static int reverseBits(int value, int numBits) {
        int result = 0;
        for (int i = 0; i < numBits; i++) {
            result = (result << 1) | (value & 1);
            value >>>= 1;
        }
        return result;
    }

    // ---------------------------------------------------------------------
    //  Min-heap (used by the Huffman construction above)
    // ---------------------------------------------------------------------

    private static int heapInsert(int @NotNull [] heap, int size, int node, long @NotNull [] freq) {
        heap[size] = node;
        int i = size;
        while (i > 0) {
            int p = (i - 1) >> 1;
            if (freq[heap[p]] <= freq[heap[i]]) break;
            int tmp = heap[p]; heap[p] = heap[i]; heap[i] = tmp;
            i = p;
        }
        return size + 1;
    }

    private static int heapExtract(int @NotNull [] heap, int size, long @NotNull [] freq) {
        int top = heap[0];
        heap[0] = heap[size - 1];
        int i = 0;
        while (true) {
            int l = i * 2 + 1;
            int r = l + 1;
            int s = i;
            if (l < size - 1 && freq[heap[l]] < freq[heap[s]]) s = l;
            if (r < size - 1 && freq[heap[r]] < freq[heap[s]]) s = r;
            if (s == i) break;
            int tmp = heap[s]; heap[s] = heap[i]; heap[i] = tmp;
            i = s;
        }
        return top;
    }

    // ---------------------------------------------------------------------
    //  Value types
    // ---------------------------------------------------------------------

    /**
     * One prefix code for one VP8L alphabet. Either a simple-mode code (1 or 2 distinct
     * symbols emitted directly without a Huffman tree) or a normal canonical Huffman
     * code with per-symbol lengths and codes. libwebp's decoder strictly validates the
     * Huffman tree as complete, so single-symbol alphabets must use the simple variant.
     */
    private static final class PrefixCode {

        final boolean isSimple;
        final int @NotNull [] simpleSymbols;
        final int @NotNull [] normalLengths;
        final int @NotNull [] normalCodes;

        private PrefixCode(boolean isSimple, int @NotNull [] simpleSymbols, int @NotNull [] normalLengths, int @NotNull [] normalCodes) {
            this.isSimple = isSimple;
            this.simpleSymbols = simpleSymbols;
            this.normalLengths = normalLengths;
            this.normalCodes = normalCodes;
        }

        static @NotNull PrefixCode simple(int @NotNull [] symbols) {
            return new PrefixCode(true, symbols, new int[0], new int[0]);
        }

        static @NotNull PrefixCode normal(int @NotNull [] lengths, int @NotNull [] codes) {
            return new PrefixCode(false, new int[0], lengths, codes);
        }

    }

}
