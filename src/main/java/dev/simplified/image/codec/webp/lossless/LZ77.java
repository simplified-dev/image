package dev.simplified.image.codec.webp.lossless;

import org.jetbrains.annotations.NotNull;

/**
 * LZ77 backward reference encoding and decoding for VP8L.
 * <p>
 * Uses hash-chain based matching for the encoder with a configurable
 * maximum distance window. The decoder interprets length-distance pairs
 * and copies pixel data from the already-decoded buffer.
 */
final class LZ77 {

    /**
     * Number of plane codes (1-indexed) that map through {@link #CODE_TO_PLANE_LUT}.
     * Plane codes larger than this are literal backward-reference distances with
     * {@code distance = plane_code - CODE_TO_PLANE_CODES}.
     */
    static final int CODE_TO_PLANE_CODES = 120;

    /**
     * VP8L 2-D plane code lookup table. Indexed by {@code plane_code - 1}.
     * Each byte encodes {@code (yoffset << 4) | (8 - xoffset)}, producing signed
     * (dy, dx) deltas with {@code dx in [-4..4]} and {@code dy in [0..7]}.
     * Source: libwebp's {@code kCodeToPlane}.
     */
    static final int[] CODE_TO_PLANE_LUT = {
        0x18, 0x07, 0x17, 0x19, 0x28, 0x06, 0x27, 0x29, 0x16, 0x1a, 0x26, 0x2a,
        0x38, 0x05, 0x37, 0x39, 0x15, 0x1b, 0x36, 0x3a, 0x25, 0x2b, 0x48, 0x04,
        0x47, 0x49, 0x14, 0x1c, 0x35, 0x3b, 0x46, 0x4a, 0x24, 0x2c, 0x58, 0x45,
        0x4b, 0x34, 0x3c, 0x03, 0x57, 0x59, 0x13, 0x1d, 0x56, 0x5a, 0x23, 0x2d,
        0x44, 0x4c, 0x55, 0x5b, 0x33, 0x3d, 0x68, 0x02, 0x67, 0x69, 0x12, 0x1e,
        0x66, 0x6a, 0x22, 0x2e, 0x54, 0x5c, 0x43, 0x4d, 0x65, 0x6b, 0x32, 0x3e,
        0x78, 0x01, 0x77, 0x79, 0x53, 0x5d, 0x11, 0x1f, 0x64, 0x6c, 0x42, 0x4e,
        0x76, 0x7a, 0x21, 0x2f, 0x75, 0x7b, 0x31, 0x3f, 0x63, 0x6d, 0x52, 0x5e,
        0x00, 0x74, 0x7c, 0x41, 0x4f, 0x10, 0x20, 0x62, 0x6e, 0x30, 0x73, 0x7d,
        0x51, 0x5f, 0x40, 0x72, 0x7e, 0x61, 0x6f, 0x50, 0x71, 0x7f, 0x60, 0x70
    };

    private static final int HASH_BITS = 16;
    private static final int HASH_SIZE = 1 << HASH_BITS;
    private static final int HASH_MASK = HASH_SIZE - 1;
    private static final int MAX_CHAIN_LENGTH = 64;
    /**
     * Minimum match length libwebp accepts into its backward-reference stream
     * ({@code MIN_LENGTH} in {@code src/enc/backward_references_enc.h}). Matches
     * shorter than 4 pixels cost more bits as length+distance symbols than they
     * save vs literal emission, so libwebp rejects them. Emitting shorter
     * matches would be spec-conformant but produce sub-optimal compression.
     */
    static final int MIN_MATCH = 4;
    /**
     * Maximum match length libwebp emits ({@code MAX_LENGTH} in
     * {@code src/enc/backward_references_enc.h}, equal to {@code (1 << 12) - 1}).
     * Longer matches must be truncated to {@code 4095} and the remainder coded
     * as a new token.
     */
    static final int MAX_LENGTH = (1 << 12) - 1;

    /**
     * libwebp's {@code plane_to_code_lut[128]} ({@code src/enc/backward_references_enc.c}):
     * maps {@code (yoffset, xoffset)} pairs to plane-code-minus-1. Indexed by
     * {@code yoffset * 16 + (8 - xoffset)} in the "left-half" branch of
     * {@link #distanceToPlaneCode} and by {@code (yoffset + 1) * 16 + 8 + (xsize - xoffset)}
     * in the "right-half" wraparound branch.
     */
    private static final int[] PLANE_TO_CODE_LUT = {
         96,  73,  55,  39,  23,  13,   5,   1, 255, 255, 255, 255, 255, 255, 255, 255,
        101,  78,  58,  42,  26,  16,   8,   2,   0,   3,   9,  17,  27,  43,  59,  79,
        102,  86,  62,  46,  32,  20,  10,   6,   4,   7,  11,  21,  33,  47,  63,  87,
        105,  90,  70,  52,  37,  28,  18,  14,  12,  15,  19,  29,  38,  53,  71,  91,
        110,  99,  82,  66,  48,  35,  30,  24,  22,  25,  31,  36,  49,  67,  83, 100,
        115, 108,  94,  76,  64,  50,  44,  40,  34,  41,  45,  51,  65,  77,  95, 109,
        118, 113, 103,  92,  80,  68,  60,  56,  54,  57,  61,  69,  81,  93, 104, 114,
        119, 116, 111, 106,  97,  88,  84,  74,  72,  75,  85,  89,  98, 107, 112, 117
    };

    /**
     * Maps a linear pixel distance to a VP8L "plane code" (the encoder-side
     * inverse of {@link #planeCodeToDistance}). Mirrors libwebp's
     * {@code VP8LDistanceToPlaneCode}: neighbours within (yoffset in [0, 7],
     * xoffset in [-7, 8]) get a 2-D plane code in {@code [1, 120]} via the
     * {@link #PLANE_TO_CODE_LUT}; everything else is encoded as
     * {@code dist + 120} so plane codes above 120 are literal linear distances.
     *
     * @param xsize image width in pixels (used to decompose {@code dist} into
     *              {@code (yoffset, xoffset)})
     * @param dist linear pixel distance ({@code >= 1})
     * @return plane code ({@code >= 1})
     */
    static int distanceToPlaneCode(int xsize, int dist) {
        int yoffset = dist / xsize;
        int xoffset = dist - yoffset * xsize;
        if (xoffset <= 8 && yoffset < 8) {
            return PLANE_TO_CODE_LUT[yoffset * 16 + 8 - xoffset] + 1;
        } else if (xoffset > xsize - 8 && yoffset < 7) {
            return PLANE_TO_CODE_LUT[(yoffset + 1) * 16 + 8 + (xsize - xoffset)] + 1;
        }
        return dist + 120;
    }

    private LZ77() { }

    /**
     * Decodes an LZ77 match length from a VP8L length symbol.
     * <p>
     * Per the VP8L spec: "Length values have the same encoding as distance values".
     * The 24-symbol length alphabet uses the same {@code base + extra_bits + 1} formula
     * as {@link #getCopyDistance}, not a DEFLATE-style length table.
     *
     * @param code the length symbol (0-23, after subtracting 256 from the green symbol)
     * @param reader the bit reader for the symbol's extra bits
     * @return the decoded match length in pixels
     */
    static int decodeLength(int code, @NotNull BitReader reader) {
        return getCopyDistance(code, reader);
    }

    /**
     * Decodes a 2D linear distance from a VP8L distance Huffman symbol.
     * <p>
     * Two-stage: the Huffman alphabet is 40 symbols that expand to a "plane code" via
     * {@link #getCopyDistance}, then the plane code maps to an actual pixel distance via
     * {@link #planeCodeToDistance}. Plane codes 1-120 are 2-D offsets (dy rows down plus
     * dx columns right, where dx is signed); plane codes above 120 are literal linear
     * distances offset by 120.
     *
     * @param distanceSymbol the 40-symbol Huffman alphabet index (0-39)
     * @param reader the bit reader for the symbol's extra bits
     * @param imageWidth the image width in pixels (used to linearize 2-D offsets)
     * @return the decoded linear pixel distance ({@code >= 1})
     */
    static int decodeDistance(int distanceSymbol, @NotNull BitReader reader, int imageWidth) {
        int planeCode = getCopyDistance(distanceSymbol, reader);
        return planeCodeToDistance(planeCode, imageWidth);
    }

    /**
     * Expands a VP8L distance Huffman symbol (0-39) into a plane code using the
     * {@code base + extra_bits + 1} formula from libwebp's {@code GetCopyDistance}.
     */
    static int getCopyDistance(int distanceSymbol, @NotNull BitReader reader) {
        if (distanceSymbol < 4)
            return distanceSymbol + 1;

        int extraBits = (distanceSymbol - 2) >> 1;
        int offset = (2 + (distanceSymbol & 1)) << extraBits;
        return offset + reader.readBits(extraBits) + 1;
    }

    /**
     * Converts a VP8L plane code to a linear pixel distance. Plane codes 1-120 use the
     * {@link #CODE_TO_PLANE_LUT} to derive signed (dy, dx) offsets and linearize them
     * via the image width; plane codes above 120 are literal linear distances.
     *
     * @param planeCode the plane code ({@code >= 1})
     * @param xSize the image width in pixels
     * @return the decoded linear distance ({@code >= 1})
     */
    static int planeCodeToDistance(int planeCode, int xSize) {
        if (planeCode > CODE_TO_PLANE_CODES)
            return planeCode - CODE_TO_PLANE_CODES;

        int distCode = CODE_TO_PLANE_LUT[planeCode - 1];
        int yOffset = distCode >> 4;
        int xOffset = 8 - (distCode & 0xF);
        int dist = yOffset * xSize + xOffset;
        return Math.max(1, dist);
    }

    /**
     * Finds the best backward reference match at the current position
     * using hash-chain matching.
     *
     * @param pixels the pixel array (ARGB values)
     * @param pos the current position
     * @param maxLen the maximum match length
     * @param hashHead the hash chain head table
     * @param hashPrev the hash chain previous-link table
     * @return the match as {length, distance}, or {0, 0} if no match
     */
    static int @NotNull [] findMatch(int @NotNull [] pixels, int pos, int maxLen, int @NotNull [] hashHead, int @NotNull [] hashPrev) {
        if (pos == 0 || maxLen < MIN_MATCH) return new int[]{0, 0};

        int hash = hashPixel(pixels, pos);
        int bestLen = 0;
        int bestDist = 0;
        int chainLen = 0;
        int candidate = hashHead[hash];

        while (candidate >= 0 && chainLen < MAX_CHAIN_LENGTH) {
            int dist = pos - candidate;
            if (dist > 0) {
                int matchLen = matchLength(pixels, candidate, pos, Math.min(maxLen, pixels.length - pos));
                if (matchLen > bestLen) {
                    bestLen = matchLen;
                    bestDist = dist;
                    if (matchLen >= maxLen) break;
                }
            }
            candidate = hashPrev[candidate];
            chainLen++;
        }

        if (bestLen < MIN_MATCH) return new int[]{0, 0};

        return new int[]{bestLen, bestDist};
    }

    /**
     * Updates the hash chain with the pixel at the given position.
     *
     * @param pixels the pixel array
     * @param pos the position to insert
     * @param hashHead the hash chain head table
     * @param hashPrev the hash chain previous-link table
     */
    static void updateHash(int @NotNull [] pixels, int pos, int @NotNull [] hashHead, int @NotNull [] hashPrev) {
        if (pos >= pixels.length) return;

        int hash = hashPixel(pixels, pos);
        hashPrev[pos] = hashHead[hash];
        hashHead[hash] = pos;
    }

    /**
     * Creates a new hash head table initialized to -1.
     * <p>
     * Indexed by hash value, maps to the most recent pixel position with that hash.
     *
     * @return the hash head table
     */
    static int @NotNull [] newHashHead() {
        int[] table = new int[HASH_SIZE];
        java.util.Arrays.fill(table, -1);
        return table;
    }

    /**
     * Creates a new hash previous-link table initialized to -1.
     * <p>
     * Indexed by pixel position, maps to the previous position in the same
     * hash chain. Sized to the pixel array length for collision-free chaining.
     *
     * @param pixelCount the total number of pixels in the image
     * @return the hash prev table
     */
    static int @NotNull [] newHashPrev(int pixelCount) {
        int[] table = new int[pixelCount];
        java.util.Arrays.fill(table, -1);
        return table;
    }

    private static int hashPixel(int[] pixels, int pos) {
        return (pixels[pos] * 0x1E35A7BD) >>> (32 - HASH_BITS);
    }

    /**
     * Encoder-side inverse of {@link #getCopyDistance} / {@link #decodeLength}:
     * splits a value {@code V >= 1} into a Huffman symbol plus extra-bit payload,
     * matching libwebp's {@code VP8LPrefixEncode}. Used for both length codes
     * (positions 256..279 of the extended green alphabet) and distance plane
     * codes (the 40-symbol distance alphabet) - libwebp uses the same prefix
     * formula for both.
     * <p>
     * Small values ({@code V <= 4}) map directly to symbols 0..3 with zero
     * extra bits. Larger values use the
     * {@code code = 2 * highest_bit + second_highest_bit} formula with
     * {@code extra_bits = highest_bit - 1} extra bits carrying the low bits of
     * {@code V - 1}. The symbol range for encoders using a 24-symbol length
     * alphabet is {@code [0, 23]}, corresponding to {@code V in [1, 4095]};
     * for a 40-symbol distance alphabet the range is {@code [0, 39]},
     * corresponding to {@code V in [1, 1 << 21]}.
     *
     * @param value value to encode ({@code >= 1})
     * @return three-element array {@code [symbol, extraBits, extraValue]}
     */
    static int @NotNull [] prefixEncode(int value) {
        if (value < 5) return new int[] { value - 1, 0, 0 };
        // highest_bit = floor(log2(value - 1))
        int v = value - 1;
        int highestBit = 31 - Integer.numberOfLeadingZeros(v);
        int secondBit = (v >> (highestBit - 1)) & 1;
        int symbol = 2 * highestBit + secondBit;
        int extraBits = highestBit - 1;
        int extraValue = v & ((1 << extraBits) - 1);
        return new int[] { symbol, extraBits, extraValue };
    }

    private static int matchLength(int[] pixels, int a, int b, int maxLen) {
        int len = 0;
        while (len < maxLen && pixels[a + len] == pixels[b + len])
            len++;
        return len;
    }

}
