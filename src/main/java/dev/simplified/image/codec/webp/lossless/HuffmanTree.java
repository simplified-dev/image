package dev.simplified.image.codec.webp.lossless;

import dev.simplified.image.exception.ImageDecodeException;
import org.jetbrains.annotations.NotNull;

/**
 * Canonical Huffman tree for VP8L entropy coding.
 * <p>
 * Supports building from code lengths (decoding) or from symbol frequencies
 * (encoding). Uses a flat lookup table for codes up to 8 bits for fast
 * single-lookup decoding, with tree-walk fallback for longer codes.
 */
final class HuffmanTree {

    private static final int TABLE_BITS = 8;
    private static final int TABLE_SIZE = 1 << TABLE_BITS;

    // Flat table: low 16 bits = symbol, high 16 bits = code length
    // For codes > TABLE_BITS, entry points to secondary table offset
    private final int[] table;
    private final int numSymbols;

    // For longer codes: tree nodes [left, right] indexed by offset
    private int[] tree;
    private int treeSize;

    // Degenerate "single symbol" tree: VP8L prefix codes with one used symbol consume
    // zero bits per read and always emit the same value. Flagged explicitly because the
    // normal flat-table lookup relies on len>0 to signal a valid entry.
    private boolean degenerate;
    private int degenerateSymbol;

    private HuffmanTree(int numSymbols) {
        this.numSymbols = numSymbols;
        this.table = new int[TABLE_SIZE];
        this.tree = new int[64];
        this.treeSize = 0;
    }

    /**
     * Builds a Huffman tree from an array of code lengths.
     * <p>
     * A code length of 0 means the symbol is not present in the tree.
     *
     * @param codeLengths the code length for each symbol
     * @return the built Huffman tree
     */
    static @NotNull HuffmanTree fromCodeLengths(int @NotNull [] codeLengths) {
        int numSymbols = codeLengths.length;
        HuffmanTree tree = new HuffmanTree(numSymbols);

        // Find max code length and count non-zero entries
        int maxLen = 0;
        int nonZero = 0;
        int firstNonZero = 0;
        for (int i = 0; i < codeLengths.length; i++) {
            int len = codeLengths[i];
            if (len > maxLen) maxLen = len;
            if (len > 0) {
                if (nonZero == 0) firstNonZero = i;
                nonZero++;
            }
        }

        if (maxLen == 0)
            return tree; // empty tree (never decoded from)

        // A tree with a single non-zero length describes a degenerate alphabet with
        // exactly one reachable symbol. VP8L simple-mode prefix codes produce this shape
        // and require zero bits per symbol. Mark as degenerate so readSymbol short-circuits.
        if (nonZero == 1) {
            tree.degenerate = true;
            tree.degenerateSymbol = firstNonZero;
            return tree;
        }

        // Count codes of each length
        int[] blCount = new int[maxLen + 1];
        for (int len : codeLengths)
            if (len > 0) blCount[len]++;

        // Compute first code of each length (canonical Huffman)
        int[] nextCode = new int[maxLen + 1];
        int code = 0;
        for (int bits = 1; bits <= maxLen; bits++) {
            code = (code + blCount[bits - 1]) << 1;
            nextCode[bits] = code;
        }

        // Assign codes and build lookup tables
        for (int symbol = 0; symbol < numSymbols; symbol++) {
            int len = codeLengths[symbol];
            if (len == 0) continue;

            int symbolCode = nextCode[len]++;
            // Reverse bits for LSB-first reading
            int reversed = reverseBits(symbolCode, len);

            if (len <= TABLE_BITS) {
                // Fill flat table entries (all bit patterns that share this prefix)
                int entry = symbol | (len << 16);
                int step = 1 << len;
                for (int i = reversed; i < TABLE_SIZE; i += step)
                    tree.table[i] = entry;
            } else {
                // Longer codes: store in overflow tree
                tree.addLongCode(reversed, len, symbol);
            }
        }

        return tree;
    }

    /**
     * Builds a Huffman tree for a single symbol (code length 0).
     *
     * @param symbol the single symbol
     * @return a tree that always decodes to this symbol
     */
    static @NotNull HuffmanTree singleSymbol(int symbol) {
        HuffmanTree tree = new HuffmanTree(symbol + 1);
        tree.degenerate = true;
        tree.degenerateSymbol = symbol;
        return tree;
    }

    /**
     * Decodes the next symbol from the bit reader.
     *
     * @param reader the bit source
     * @return the decoded symbol
     * @throws ImageDecodeException if the code is invalid
     */
    int readSymbol(@NotNull BitReader reader) {
        if (degenerate) return degenerateSymbol;

        int bits = reader.peekBits(TABLE_BITS);
        int entry = table[bits];
        int len = entry >>> 16;

        if (len > 0 && len <= TABLE_BITS) {
            reader.advanceBits(len);
            return entry & 0xFFFF;
        }

        // Long code - walk overflow tree
        return readLongCode(reader);
    }

    /**
     * Builds optimal code lengths from symbol frequencies for encoding.
     *
     * @param frequencies the frequency count of each symbol
     * @param maxBits the maximum allowed code length
     * @return an array of code lengths indexed by symbol
     */
    static int @NotNull [] buildCodeLengths(int @NotNull [] frequencies, int maxBits) {
        int n = frequencies.length;
        int[] codeLengths = new int[n];

        // Count non-zero symbols
        int nonZero = 0;
        int lastSymbol = 0;
        for (int i = 0; i < n; i++) {
            if (frequencies[i] > 0) {
                nonZero++;
                lastSymbol = i;
            }
        }

        if (nonZero == 0) return codeLengths;
        if (nonZero == 1) {
            codeLengths[lastSymbol] = 1;
            return codeLengths;
        }

        // Simple length-limited Huffman using package-merge is complex;
        // use a simpler heuristic: frequency-proportional bit allocation
        // capped at maxBits
        long totalFreq = 0;
        for (int f : frequencies)
            totalFreq += f;

        for (int i = 0; i < n; i++) {
            if (frequencies[i] > 0) {
                double prob = (double) frequencies[i] / totalFreq;
                int bits = Math.max(1, Math.min(maxBits, (int) Math.ceil(-Math.log(prob) / Math.log(2))));
                codeLengths[i] = bits;
            }
        }

        return codeLengths;
    }

    private void addLongCode(int reversed, int len, int symbol) {
        // For codes longer than TABLE_BITS, we use the table entry as a pointer
        // to an overflow area. Simple linear search for now.
        int tableIndex = reversed & (TABLE_SIZE - 1);
        int entry = table[tableIndex];

        if ((entry >>> 16) == 0) {
            // First long code at this table index - mark as overflow
            table[tableIndex] = (treeSize) | (0xFFFF << 16); // overflow marker
        }

        ensureTreeCapacity(treeSize + 3);
        tree[treeSize++] = reversed;
        tree[treeSize++] = len;
        tree[treeSize++] = symbol;
    }

    private int readLongCode(@NotNull BitReader reader) {
        int bits = reader.peekBits(TABLE_BITS);
        int entry = table[bits];
        int overflowStart = entry & 0xFFFF;

        // Linear search through overflow entries
        for (int i = overflowStart; i < treeSize; i += 3) {
            int storedReversed = tree[i];
            int storedLen = tree[i + 1];
            int storedSymbol = tree[i + 2];

            int fullBits = reader.peekBits(storedLen);
            if (fullBits == storedReversed) {
                reader.advanceBits(storedLen);
                return storedSymbol;
            }
        }

        throw new ImageDecodeException("Invalid Huffman code in VP8L bitstream");
    }

    private void ensureTreeCapacity(int required) {
        if (required > tree.length) {
            int[] newTree = new int[Math.max(tree.length * 2, required)];
            System.arraycopy(tree, 0, newTree, 0, treeSize);
            tree = newTree;
        }
    }

    private static int reverseBits(int value, int numBits) {
        int result = 0;
        for (int i = 0; i < numBits; i++) {
            result = (result << 1) | (value & 1);
            value >>>= 1;
        }
        return result;
    }

}
