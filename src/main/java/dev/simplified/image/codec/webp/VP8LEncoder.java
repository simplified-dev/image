package dev.simplified.image.codec.webp;

import dev.simplified.image.PixelBuffer;
import dev.simplified.image.exception.ImageEncodeException;
import org.jetbrains.annotations.NotNull;

import java.util.HashSet;
import java.util.Set;

/**
 * Pure Java VP8L (WebP lossless) encoder.
 * <p>
 * Encodes ARGB pixel data into a VP8L bitstream using forward transforms,
 * Huffman coding, LZ77 backward references, and color cache.
 */
final class VP8LEncoder {

    private static final int NUM_LITERAL_CODES = 256;
    private static final int NUM_LENGTH_CODES = 24;
    private static final int MAX_COLOR_CACHE_BITS = 11;
    private static final int CODE_LENGTH_CODES = 19;

    private static final int[] CODE_LENGTH_ORDER = {
        17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    };

    private VP8LEncoder() { }

    /**
     * Encodes pixel data into a VP8L bitstream.
     *
     * @param pixels the source pixel buffer
     * @return the encoded VP8L payload bytes (including 0x2F signature)
     * @throws ImageEncodeException if encoding fails
     */
    static byte @NotNull [] encode(@NotNull PixelBuffer pixels) {
        int width = pixels.width();
        int height = pixels.height();
        int[] pixelData = pixels.getPixels();

        // Analyze image for transforms
        boolean useSubtractGreen = true;
        boolean usePalette = false;
        int[] palette = null;

        Set<Integer> uniqueColors = new HashSet<>();
        for (int p : pixelData) {
            uniqueColors.add(p);
            if (uniqueColors.size() > 256) break;
        }

        if (uniqueColors.size() <= 256) {
            usePalette = true;
            useSubtractGreen = false;
            palette = uniqueColors.stream().mapToInt(Integer::intValue).toArray();
            java.util.Arrays.sort(palette);
        }

        // Make a working copy for transforms
        int[] workPixels = pixelData.clone();

        // Determine color cache bits
        int colorCacheBits = usePalette ? 0 : 6;

        // Build output
        BitWriter writer = new BitWriter(width * height * 2);

        // VP8L signature
        writer.writeBits(0x2F, 8);

        // Header: 14-bit width, 14-bit height, 1-bit alpha, 3-bit version
        writer.writeBits(width - 1, 14);
        writer.writeBits(height - 1, 14);
        writer.writeBits(1, 1); // alpha used
        writer.writeBits(0, 3); // version 0

        // Transforms
        if (usePalette) {
            // Apply subtract-green: no, palette mode
            // Signal transform present
            writer.writeBit(1);
            writer.writeBits(3, 2); // Color indexing transform

            // Write palette size
            writer.writeBits(palette.length - 1, 8);

            // Delta-encode palette and write as sub-image
            int[] deltaPalette = new int[palette.length];
            deltaPalette[0] = palette[0];
            for (int i = 1; i < palette.length; i++)
                deltaPalette[i] = subPixels(palette[i], palette[i - 1]);

            encodeSubImage(writer, deltaPalette);

            // Convert pixels to palette indices
            java.util.Map<Integer, Integer> indexMap = new java.util.HashMap<>();
            for (int i = 0; i < palette.length; i++)
                indexMap.put(palette[i], i);

            for (int i = 0; i < workPixels.length; i++) {
                Integer idx = indexMap.get(workPixels[i]);
                workPixels[i] = idx != null ? idx : 0;
            }
        }

        if (useSubtractGreen) {
            writer.writeBit(1);
            writer.writeBits(2, 2); // Subtract green
            new VP8LTransform.SubtractGreen().forwardTransform(workPixels, width, height);
        }

        // No more transforms
        writer.writeBit(0);

        // Color cache
        if (colorCacheBits > 0) {
            writer.writeBit(1);
            writer.writeBits(colorCacheBits, 4);
        } else {
            writer.writeBit(0);
        }

        // Encode pixels with Huffman + LZ77
        encodePixels(writer, workPixels, width, height, colorCacheBits);

        return writer.toByteArray();
    }

    private static void encodeSubImage(@NotNull BitWriter writer, int @NotNull [] pixels) {
        // Write color cache flag
        writer.writeBit(0); // No color cache for sub-images

        // Use simple Huffman coding for sub-images
        encodePixelsSimple(writer, pixels);
    }

    private static void encodePixelsSimple(@NotNull BitWriter writer, int @NotNull [] pixels) {
        // Collect frequency statistics
        int numGreen = NUM_LITERAL_CODES + NUM_LENGTH_CODES;
        int[] greenFreq = new int[numGreen];
        int[] redFreq = new int[256];
        int[] blueFreq = new int[256];
        int[] alphaFreq = new int[256];

        for (int pixel : pixels) {
            greenFreq[(pixel >> 8) & 0xFF]++;
            redFreq[(pixel >> 16) & 0xFF]++;
            blueFreq[pixel & 0xFF]++;
            alphaFreq[(pixel >> 24) & 0xFF]++;
        }

        // Build Huffman trees
        int[] greenLengths = HuffmanTree.buildCodeLengths(greenFreq, 15);
        int[] redLengths = HuffmanTree.buildCodeLengths(redFreq, 15);
        int[] blueLengths = HuffmanTree.buildCodeLengths(blueFreq, 15);
        int[] alphaLengths = HuffmanTree.buildCodeLengths(alphaFreq, 15);
        int[] distLengths = new int[120]; // No distance codes in simple encoding

        // Write as normal (non-simple) Huffman codes
        writer.writeBit(0); // Not simple codes

        writeCodeLengths(writer, greenLengths, redLengths, blueLengths, alphaLengths, distLengths);

        // Build encoding tables
        int[][] greenCodes = buildCanonicalCodes(greenLengths);
        int[][] redCodes = buildCanonicalCodes(redLengths);
        int[][] blueCodes = buildCanonicalCodes(blueLengths);
        int[][] alphaCodes = buildCanonicalCodes(alphaLengths);

        // Encode literals only (no LZ77 for sub-images)
        for (int pixel : pixels) {
            int green = (pixel >> 8) & 0xFF;
            int red = (pixel >> 16) & 0xFF;
            int blue = pixel & 0xFF;
            int alpha = (pixel >> 24) & 0xFF;

            writeCode(writer, greenCodes, green);
            writeCode(writer, redCodes, red);
            writeCode(writer, blueCodes, blue);
            writeCode(writer, alphaCodes, alpha);
        }
    }

    private static void encodePixels(@NotNull BitWriter writer, int @NotNull [] pixels, int width, int height, int colorCacheBits) {
        int totalPixels = width * height;
        ColorCache cache = new ColorCache(colorCacheBits);
        int numCacheCodes = colorCacheBits > 0 ? (1 << colorCacheBits) : 0;

        // First pass: LZ77 matching + frequency collection
        int numGreen = NUM_LITERAL_CODES + NUM_LENGTH_CODES + numCacheCodes;
        int[] greenFreq = new int[numGreen];
        int[] redFreq = new int[256];
        int[] blueFreq = new int[256];
        int[] alphaFreq = new int[256];
        int numDistanceCodes = 120 + LZ77.DISTANCE_CODES.length;
        int[] distFreq = new int[numDistanceCodes];

        // Encode tokens
        int[] tokens = new int[totalPixels * 4]; // worst case: 4 ints per pixel
        int tokenCount = 0;

        int[] hashHead = LZ77.newHashHead();
        int[] hashPrev = LZ77.newHashPrev(totalPixels);

        int pos = 0;
        while (pos < totalPixels) {
            int[] match = LZ77.findMatch(pixels, pos, Math.min(258, totalPixels - pos), hashHead, hashPrev);

            if (match[0] >= 3) {
                // Backward reference
                int length = match[0];
                int distance = match[1];

                int lengthCode = encodeLengthCode(length);
                int distCode = encodeDistanceCode(distance, width);

                greenFreq[NUM_LITERAL_CODES + lengthCode]++;
                distFreq[Math.min(distCode, distFreq.length - 1)]++;

                if (tokenCount + 5 > tokens.length)
                    tokens = grow(tokens);

                tokens[tokenCount++] = -1; // marker: backward reference
                tokens[tokenCount++] = length;
                tokens[tokenCount++] = lengthCode;
                tokens[tokenCount++] = distance;
                tokens[tokenCount++] = distCode;

                for (int j = 0; j < length; j++) {
                    cache.insert(pixels[pos + j]);
                    LZ77.updateHash(pixels, pos + j, hashHead, hashPrev);
                }
                pos += length;
            } else {
                // Literal or cache hit
                int argb = pixels[pos];
                boolean cacheHit = false;

                if (cache.isEnabled() && pos > 0) {
                    int cacheIdx = cache.hashIndex(argb);
                    if (cache.lookup(cacheIdx) == argb && (argb != 0 || cacheIdx != 0)) {
                        greenFreq[NUM_LITERAL_CODES + NUM_LENGTH_CODES + cacheIdx]++;
                        if (tokenCount + 2 > tokens.length)
                            tokens = grow(tokens);
                        tokens[tokenCount++] = -2; // marker: cache hit
                        tokens[tokenCount++] = cacheIdx;
                        cacheHit = true;
                    }
                }

                if (!cacheHit) {
                    int green = (argb >> 8) & 0xFF;
                    int red = (argb >> 16) & 0xFF;
                    int blue = argb & 0xFF;
                    int alpha = (argb >> 24) & 0xFF;

                    greenFreq[green]++;
                    redFreq[red]++;
                    blueFreq[blue]++;
                    alphaFreq[alpha]++;

                    if (tokenCount + 5 > tokens.length)
                        tokens = grow(tokens);
                    tokens[tokenCount++] = 0; // marker: literal
                    tokens[tokenCount++] = green;
                    tokens[tokenCount++] = red;
                    tokens[tokenCount++] = blue;
                    tokens[tokenCount++] = alpha;
                }

                cache.insert(argb);
                LZ77.updateHash(pixels, pos, hashHead, hashPrev);
                pos++;
            }
        }

        // Build Huffman codes from frequencies
        int[] greenLengths = HuffmanTree.buildCodeLengths(greenFreq, 15);
        int[] redLengths = HuffmanTree.buildCodeLengths(redFreq, 15);
        int[] blueLengths = HuffmanTree.buildCodeLengths(blueFreq, 15);
        int[] alphaLengths = HuffmanTree.buildCodeLengths(alphaFreq, 15);
        int[] distLengths = HuffmanTree.buildCodeLengths(distFreq, 15);

        // Write Huffman codes
        writer.writeBit(0); // Not simple
        writeCodeLengths(writer, greenLengths, redLengths, blueLengths, alphaLengths, distLengths);

        // Build encoding tables
        int[][] greenCodes = buildCanonicalCodes(greenLengths);
        int[][] redCodes = buildCanonicalCodes(redLengths);
        int[][] blueCodes = buildCanonicalCodes(blueLengths);
        int[][] alphaCodes = buildCanonicalCodes(alphaLengths);
        int[][] distCodes = buildCanonicalCodes(distLengths);

        // Second pass: write tokens
        int ti = 0;
        while (ti < tokenCount) {
            int type = tokens[ti++];

            if (type == 0) {
                // Literal
                writeCode(writer, greenCodes, tokens[ti++]);
                writeCode(writer, redCodes, tokens[ti++]);
                writeCode(writer, blueCodes, tokens[ti++]);
                writeCode(writer, alphaCodes, tokens[ti++]);
            } else if (type == -1) {
                // Backward reference
                int length = tokens[ti++];
                int lengthCode = tokens[ti++];
                int distance = tokens[ti++];
                int distCode = tokens[ti++];

                writeCode(writer, greenCodes, NUM_LITERAL_CODES + lengthCode);
                writeLengthExtra(writer, lengthCode, length);
                writeCode(writer, distCodes, Math.min(distCode, distCodes[0].length - 1));
                writeDistanceExtra(writer, distCode, distance);
            } else if (type == -2) {
                // Cache hit
                int cacheIdx = tokens[ti++];
                writeCode(writer, greenCodes, NUM_LITERAL_CODES + NUM_LENGTH_CODES + cacheIdx);
            }
        }
    }

    private static void writeCodeLengths(
        @NotNull BitWriter writer,
        int @NotNull [] greenLengths,
        int @NotNull [] redLengths,
        int @NotNull [] blueLengths,
        int @NotNull [] alphaLengths,
        int @NotNull [] distLengths
    ) {
        int[][] allLengths = {greenLengths, redLengths, blueLengths, alphaLengths, distLengths};

        // Build code-length-code frequencies
        int[] clcFreq = new int[CODE_LENGTH_CODES];
        for (int[] lengths : allLengths)
            for (int len : lengths)
                if (len >= 0 && len < CODE_LENGTH_CODES) clcFreq[len]++;

        int[] clcLengths = HuffmanTree.buildCodeLengths(clcFreq, 7);

        // Find last non-zero code length code
        int numCodes = CODE_LENGTH_CODES;
        while (numCodes > 4 && clcLengths[CODE_LENGTH_ORDER[numCodes - 1]] == 0)
            numCodes--;

        writer.writeBits(numCodes - 4, 4);

        for (int i = 0; i < numCodes; i++)
            writer.writeBits(clcLengths[CODE_LENGTH_ORDER[i]], 3);

        int[][] clcCodes = buildCanonicalCodes(clcLengths);

        // Write each alphabet's code lengths
        for (int[] lengths : allLengths)
            for (int len : lengths)
                writeCode(writer, clcCodes, len);
    }

    /**
     * Builds canonical Huffman codes from code lengths.
     *
     * @return [0] = codes array, [1] = lengths array (mirrored for convenience)
     */
    private static int @NotNull [][] buildCanonicalCodes(int @NotNull [] codeLengths) {
        int n = codeLengths.length;
        int[] codes = new int[n];

        int maxLen = 0;
        for (int len : codeLengths)
            if (len > maxLen) maxLen = len;

        if (maxLen == 0) return new int[][]{codes, codeLengths};

        int[] blCount = new int[maxLen + 1];
        for (int len : codeLengths)
            if (len > 0) blCount[len]++;

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

        return new int[][]{codes, codeLengths};
    }

    private static void writeCode(@NotNull BitWriter writer, int @NotNull [][] codeTable, int symbol) {
        if (symbol >= codeTable[0].length) return;
        int code = codeTable[0][symbol];
        int len = codeTable[1][symbol];
        if (len > 0)
            writer.writeBits(code, len);
    }

    private static int encodeLengthCode(int length) {
        for (int i = LZ77.LENGTH_CODES.length - 1; i >= 0; i--)
            if (length >= LZ77.LENGTH_CODES[i]) return i;
        return 0;
    }

    private static void writeLengthExtra(@NotNull BitWriter writer, int code, int length) {
        if (code < LZ77.LENGTH_EXTRA_BITS.length && LZ77.LENGTH_EXTRA_BITS[code] > 0) {
            int base = LZ77.LENGTH_CODES[code];
            writer.writeBits(length - base, LZ77.LENGTH_EXTRA_BITS[code]);
        }
    }

    private static int encodeDistanceCode(int distance, int width) {
        // Try 2D plane codes 0-119 first (more efficient for nearby references)
        for (int code = 0; code < 120; code++) {
            if (LZ77.planeCodeToDistance(code, width) == distance)
                return code;
        }

        // Fall back to linear distance codes 120+
        for (int i = LZ77.DISTANCE_CODES.length - 1; i >= 0; i--)
            if (distance >= LZ77.DISTANCE_CODES[i]) return 120 + i;

        return 120;
    }

    private static void writeDistanceExtra(@NotNull BitWriter writer, int code, int distance) {
        if (code < 120) return; // Plane codes have no extra bits

        int idx = code - 120;

        if (idx >= 0 && idx < LZ77.DISTANCE_EXTRA_BITS.length && LZ77.DISTANCE_EXTRA_BITS[idx] > 0) {
            int base = LZ77.DISTANCE_CODES[idx];
            writer.writeBits(distance - base, LZ77.DISTANCE_EXTRA_BITS[idx]);
        }
    }

    private static int subPixels(int a, int b) {
        int alpha = (((a >> 24) & 0xFF) - ((b >> 24) & 0xFF)) & 0xFF;
        int red = (((a >> 16) & 0xFF) - ((b >> 16) & 0xFF)) & 0xFF;
        int green = (((a >> 8) & 0xFF) - ((b >> 8) & 0xFF)) & 0xFF;
        int blue = ((a & 0xFF) - (b & 0xFF)) & 0xFF;
        return (alpha << 24) | (red << 16) | (green << 8) | blue;
    }

    private static int reverseBits(int value, int numBits) {
        int result = 0;
        for (int i = 0; i < numBits; i++) {
            result = (result << 1) | (value & 1);
            value >>>= 1;
        }
        return result;
    }

    private static int[] grow(int[] arr) {
        int[] newArr = new int[arr.length * 2];
        System.arraycopy(arr, 0, newArr, 0, arr.length);
        return newArr;
    }

}
