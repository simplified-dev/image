package dev.simplified.image.codec.webp.lossless;

import dev.simplified.image.pixel.PixelBuffer;
import org.jetbrains.annotations.NotNull;

/**
 * Pure Java VP8L (WebP lossless) encoder.
 * <p>
 * Emits a spec-compliant bitstream that reference {@code libwebp} decoders accept. The
 * current implementation is a <i>literal-only</i> encoder: no LZ77 backward references,
 * no color cache, no spatial transforms. Every pixel is emitted as four prefix-coded
 * literals (green, red, blue, alpha). Trade-off is clear - substantially larger output
 * than an LZ77-optimized encoder, but every byte conforms to the
 * <a href="https://developers.google.com/speed/webp/docs/webp_lossless_bitstream_specification">
 * WebP Lossless Bitstream Specification</a>.
 * <p>
 * The encoder uses <i>simple prefix codes</i> (is_simple = 1) for alphabets with one or
 * two distinct used symbols, and normal prefix codes with a proper Huffman tree for
 * three or more. The VP8L spec mandates this split: a Huffman code with a single leaf of
 * length 1 is not a complete tree, and libwebp's {@code VP8LBuildHuffmanTable} rejects
 * it as {@code VP8_STATUS_BITSTREAM_ERROR}. Prior versions of this encoder emitted such
 * incomplete trees whenever the source image had uniform color channels (which for
 * solid-colored regions like the lore tooltip background is essentially always) -
 * that's the root of the long-standing "webp contains errors" diagnostic.
 */
public final class VP8LEncoder {

    private static final int VP8L_SIGNATURE = 0x2F;
    private static final int VP8L_VERSION = 0;

    private static final int NUM_LITERAL_CODES = 256;
    private static final int NUM_LENGTH_CODES = 24;
    private static final int NUM_DISTANCE_CODES = 40;
    private static final int MAX_HUFFMAN_BITS = 15;

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
    public static byte @NotNull [] encode(@NotNull PixelBuffer pixels) {
        int width = pixels.width();
        int height = pixels.height();
        int[] pixelData = pixels.pixels();

        int greenAlphabetSize = NUM_LITERAL_CODES + NUM_LENGTH_CODES;

        int[] greenFreq = new int[greenAlphabetSize];
        int[] redFreq = new int[NUM_LITERAL_CODES];
        int[] blueFreq = new int[NUM_LITERAL_CODES];
        int[] alphaFreq = new int[NUM_LITERAL_CODES];
        int[] distFreq = new int[NUM_DISTANCE_CODES];

        for (int argb : pixelData) {
            greenFreq[(argb >> 8) & 0xFF]++;
            redFreq[(argb >> 16) & 0xFF]++;
            blueFreq[argb & 0xFF]++;
            alphaFreq[(argb >> 24) & 0xFF]++;
        }
        // Distance alphabet is unused but the prefix code must still exist. A single
        // "used" symbol triggers the simple-mode path below, which is the only valid way
        // to describe an unused-but-required alphabet to a VP8L decoder.
        distFreq[0] = 1;

        PrefixCode greenPrefix = buildPrefixCode(greenFreq, MAX_HUFFMAN_BITS);
        PrefixCode redPrefix = buildPrefixCode(redFreq, MAX_HUFFMAN_BITS);
        PrefixCode bluePrefix = buildPrefixCode(blueFreq, MAX_HUFFMAN_BITS);
        PrefixCode alphaPrefix = buildPrefixCode(alphaFreq, MAX_HUFFMAN_BITS);
        PrefixCode distPrefix = buildPrefixCode(distFreq, MAX_HUFFMAN_BITS);

        BitWriter writer = new BitWriter(Math.max(1024, width * height * 4));

        // --- Header ---
        writer.writeBits(VP8L_SIGNATURE, 8);
        writer.writeBits(width - 1, 14);
        writer.writeBits(height - 1, 14);
        writer.writeBits(1, 1);                    // alpha_is_used
        writer.writeBits(VP8L_VERSION, 3);

        // --- No transforms, no color cache, no meta-prefix ---
        writer.writeBit(0);
        writer.writeBit(0);
        writer.writeBit(0);

        // --- Five prefix-code declarations (G, R, B, A, distance) ---
        writePrefixCode(writer, greenPrefix, greenAlphabetSize);
        writePrefixCode(writer, redPrefix, NUM_LITERAL_CODES);
        writePrefixCode(writer, bluePrefix, NUM_LITERAL_CODES);
        writePrefixCode(writer, alphaPrefix, NUM_LITERAL_CODES);
        writePrefixCode(writer, distPrefix, NUM_DISTANCE_CODES);

        // --- Entropy-coded pixel data ---
        for (int argb : pixelData) {
            emitSymbol(writer, greenPrefix, (argb >> 8) & 0xFF);
            emitSymbol(writer, redPrefix, (argb >> 16) & 0xFF);
            emitSymbol(writer, bluePrefix, argb & 0xFF);
            emitSymbol(writer, alphaPrefix, (argb >> 24) & 0xFF);
        }

        return writer.toByteArray();
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

        if (usedCount <= 1) {
            int sym = usedCount == 1 ? usedSymbols[0] : 0;
            return PrefixCode.simple(new int[]{ sym });
        }
        if (usedCount == 2)
            return PrefixCode.simple(new int[]{ usedSymbols[0], usedSymbols[1] });

        int[] lengths = buildHuffmanLengths(freq, maxBits);
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

        for (int i = 0; i < alphabetSize; i++) {
            int len = i < codeLengths.length ? codeLengths[i] : 0;
            writer.writeBits(clcCodes[len], clcLengths[len]);
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
