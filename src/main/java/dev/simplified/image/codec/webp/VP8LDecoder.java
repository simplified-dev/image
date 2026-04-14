package dev.simplified.image.codec.webp;

import dev.simplified.image.pixel.PixelBuffer;
import dev.simplified.image.exception.ImageDecodeException;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.List;

/**
 * Pure Java VP8L (WebP lossless) decoder.
 * <p>
 * Decodes a VP8L bitstream into ARGB pixel data using Huffman coding,
 * LZ77 backward references, color cache, and four reversible transforms
 * (predictor, color, subtract-green, color-indexing).
 */
final class VP8LDecoder {

    /** Number of Huffman code groups in VP8L: green+length, red, blue, alpha, distance. */
    private static final int NUM_CODE_GROUPS = 5;
    private static final int NUM_LITERAL_CODES = 256;
    private static final int NUM_LENGTH_CODES = 24;
    private static final int MAX_COLOR_CACHE_BITS = 11;
    private static final int CODE_LENGTH_CODES = 19;

    /** Code length code order per VP8L spec. */
    private static final int[] CODE_LENGTH_ORDER = {
        17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    };

    private VP8LDecoder() { }

    /**
     * Decodes a VP8L bitstream into pixel data.
     *
     * @param data the raw VP8L payload (starting with 0x2F signature byte)
     * @return the decoded pixel buffer
     * @throws ImageDecodeException if decoding fails
     */
    static @NotNull PixelBuffer decode(byte @NotNull [] data) {
        if (data.length < 5)
            throw new ImageDecodeException("VP8L data too short");

        // Validate signature
        if (data[0] != 0x2F)
            throw new ImageDecodeException("Invalid VP8L signature: 0x%02X", data[0] & 0xFF);

        BitReader reader = new BitReader(data, 1, data.length - 1);

        // Read header: 14-bit width, 14-bit height, 1-bit alpha, 3-bit version
        int width = reader.readBits(14) + 1;
        int height = reader.readBits(14) + 1;
        int alphaUsed = reader.readBit();
        int version = reader.readBits(3);

        if (version != 0)
            throw new ImageDecodeException("Unsupported VP8L version: %d", version);

        // Read transforms
        List<VP8LTransform> transforms = new ArrayList<>();
        readTransforms(reader, transforms, width, height);

        // Determine effective dimensions after transforms (color indexing may change width)
        int effectiveWidth = width;

        for (VP8LTransform t : transforms) {
            if (t instanceof VP8LTransform.ColorIndexing ci && ci.bitsPerPixel() < 8) {
                int pixelsPerByte = 8 / ci.bitsPerPixel();
                effectiveWidth = (width + pixelsPerByte - 1) / pixelsPerByte;
            }
        }

        // Read color cache size
        int colorCacheBits = 0;
        if (reader.readBit() == 1) {
            colorCacheBits = reader.readBits(4);
            if (colorCacheBits > MAX_COLOR_CACHE_BITS)
                throw new ImageDecodeException("Invalid color cache bits: %d", colorCacheBits);
        }

        // Read Huffman codes
        int numDistanceCodes = 120; // VP8L 2D distance codes
        HuffmanTree[] trees = readHuffmanCodes(reader, colorCacheBits, numDistanceCodes);

        // Decode pixels
        int[] pixels = new int[effectiveWidth * height];
        ColorCache cache = new ColorCache(colorCacheBits);
        decodePixels(reader, trees, pixels, effectiveWidth, height, cache);

        // Apply inverse transforms in reverse order
        for (int i = transforms.size() - 1; i >= 0; i--)
            transforms.get(i).inverseTransform(pixels, width, height);

        return PixelBuffer.of(pixels, width, height);
    }

    /**
     * Decodes a sub-image (used for transform data / meta images).
     */
    static int @NotNull [] decodeSubImage(@NotNull BitReader reader, int width, int height) {
        // Read color cache
        int colorCacheBits = 0;
        if (reader.readBit() == 1) {
            colorCacheBits = reader.readBits(4);
            if (colorCacheBits > MAX_COLOR_CACHE_BITS)
                throw new ImageDecodeException("Invalid color cache bits in sub-image");
        }

        HuffmanTree[] trees = readHuffmanCodes(reader, colorCacheBits, 120);
        int[] pixels = new int[width * height];
        ColorCache cache = new ColorCache(colorCacheBits);
        decodePixels(reader, trees, pixels, width, height, cache);
        return pixels;
    }

    private static void readTransforms(@NotNull BitReader reader, @NotNull List<VP8LTransform> transforms, int width, int height) {
        while (reader.readBit() == 1) {
            int type = reader.readBits(2);

            switch (type) {
                case 0 -> { // Predictor
                    int blockBits = reader.readBits(3) + 2;
                    int blockWidth = ((width - 1) >> blockBits) + 1;
                    int blockHeight = ((height - 1) >> blockBits) + 1;
                    int[] blockModes = decodeSubImage(reader, blockWidth, blockHeight);
                    transforms.add(new VP8LTransform.Predictor(blockBits, blockModes, blockWidth));
                }
                case 1 -> { // Color transform
                    int blockBits = reader.readBits(3) + 2;
                    int blockWidth = ((width - 1) >> blockBits) + 1;
                    int blockHeight = ((height - 1) >> blockBits) + 1;
                    int[] transformData = decodeSubImage(reader, blockWidth, blockHeight);
                    transforms.add(new VP8LTransform.ColorXform(blockBits, transformData, blockWidth));
                }
                case 2 -> // Subtract green
                    transforms.add(new VP8LTransform.SubtractGreen());
                case 3 -> { // Color indexing
                    int paletteSize = reader.readBits(8) + 1;
                    int[] palette = decodeSubImage(reader, paletteSize, 1);

                    // Delta-decode palette
                    for (int i = 1; i < palette.length; i++) {
                        int prev = palette[i - 1];
                        int curr = palette[i];
                        palette[i] = addPixelsForPalette(prev, curr);
                    }

                    int bitsPerPixel;
                    if (paletteSize <= 2) bitsPerPixel = 1;
                    else if (paletteSize <= 4) bitsPerPixel = 2;
                    else if (paletteSize <= 16) bitsPerPixel = 4;
                    else bitsPerPixel = 8;

                    transforms.add(new VP8LTransform.ColorIndexing(palette, bitsPerPixel));
                }
            }
        }
    }

    private static @NotNull HuffmanTree @NotNull [] readHuffmanCodes(@NotNull BitReader reader, int colorCacheBits, int numDistanceCodes) {
        int numGreenCodes = NUM_LITERAL_CODES + NUM_LENGTH_CODES + (colorCacheBits > 0 ? (1 << colorCacheBits) : 0);

        // Check for simple code (1 or 2 symbols)
        if (reader.readBit() == 1) {
            // Simple code length code
            return readSimpleHuffmanCodes(reader, numGreenCodes, numDistanceCodes);
        }

        // Normal code length codes
        HuffmanTree[] trees = new HuffmanTree[NUM_CODE_GROUPS];
        int[] alphabetSizes = {numGreenCodes, 256, 256, 256, numDistanceCodes};

        // Read code length code
        int numCodeLengthCodes = reader.readBits(4) + 4;
        int[] codeLengthCodeLengths = new int[CODE_LENGTH_CODES];

        for (int i = 0; i < numCodeLengthCodes; i++)
            codeLengthCodeLengths[CODE_LENGTH_ORDER[i]] = reader.readBits(3);

        HuffmanTree codeLengthTree = HuffmanTree.fromCodeLengths(codeLengthCodeLengths);

        // Read each alphabet's code lengths
        for (int g = 0; g < NUM_CODE_GROUPS; g++) {
            int[] codeLengths = readCodeLengths(reader, codeLengthTree, alphabetSizes[g]);
            trees[g] = HuffmanTree.fromCodeLengths(codeLengths);
        }

        return trees;
    }

    private static @NotNull HuffmanTree @NotNull [] readSimpleHuffmanCodes(@NotNull BitReader reader, int numGreenCodes, int numDistanceCodes) {
        int numSymbols = reader.readBit() + 1;
        int firstSymbolBits = reader.readBit() == 0 ? 1 : 8;
        int symbol0 = reader.readBits(firstSymbolBits);

        HuffmanTree[] trees = new HuffmanTree[NUM_CODE_GROUPS];

        if (numSymbols == 1) {
            trees[0] = HuffmanTree.singleSymbol(symbol0);
            for (int i = 1; i < NUM_CODE_GROUPS; i++)
                trees[i] = HuffmanTree.singleSymbol(0);
        } else {
            int symbol1 = reader.readBits(8);
            int[] codeLengths = new int[Math.max(symbol0, symbol1) + 1];
            codeLengths[symbol0] = 1;
            codeLengths[symbol1] = 1;
            trees[0] = HuffmanTree.fromCodeLengths(codeLengths);
            for (int i = 1; i < NUM_CODE_GROUPS; i++)
                trees[i] = HuffmanTree.singleSymbol(0);
        }

        return trees;
    }

    private static int @NotNull [] readCodeLengths(@NotNull BitReader reader, @NotNull HuffmanTree codeLengthTree, int numSymbols) {
        int[] codeLengths = new int[numSymbols];
        int prevCodeLen = 8;
        int i = 0;

        while (i < numSymbols) {
            int code = codeLengthTree.readSymbol(reader);

            if (code < 16) {
                codeLengths[i++] = code;
                if (code != 0) prevCodeLen = code;
            } else if (code == 16) {
                // Repeat previous length 3-6 times
                int repeat = 3 + reader.readBits(2);
                for (int j = 0; j < repeat && i < numSymbols; j++)
                    codeLengths[i++] = prevCodeLen;
            } else if (code == 17) {
                // Repeat zero 3-10 times
                int repeat = 3 + reader.readBits(3);
                i += repeat;
            } else if (code == 18) {
                // Repeat zero 11-138 times
                int repeat = 11 + reader.readBits(7);
                i += repeat;
            }
        }

        return codeLengths;
    }

    private static void decodePixels(
        @NotNull BitReader reader,
        @NotNull HuffmanTree @NotNull [] trees,
        int @NotNull [] pixels,
        int width,
        int height,
        @NotNull ColorCache cache
    ) {
        int totalPixels = width * height;
        int pos = 0;

        while (pos < totalPixels) {
            int green = trees[0].readSymbol(reader);

            if (green < NUM_LITERAL_CODES) {
                // Literal ARGB pixel
                int red = trees[1].readSymbol(reader);
                int blue = trees[2].readSymbol(reader);
                int alpha = trees[3].readSymbol(reader);
                int argb = (alpha << 24) | (red << 16) | (green << 8) | blue;
                pixels[pos++] = argb;
                cache.insert(argb);
            } else if (green < NUM_LITERAL_CODES + NUM_LENGTH_CODES) {
                // LZ77 backward reference
                int lengthCode = green - NUM_LITERAL_CODES;
                int length = LZ77.decodeLength(lengthCode, reader);
                int distCode = trees[4].readSymbol(reader);
                int distance = LZ77.decodeDistance(distCode, reader, width);

                // Copy pixels from back-reference
                for (int j = 0; j < length && pos < totalPixels; j++) {
                    int srcPos = pos - distance;

                    if (srcPos < 0)
                        throw new ImageDecodeException("Invalid backward reference at position %d, distance %d", pos, distance);

                    pixels[pos] = pixels[srcPos];
                    cache.insert(pixels[pos]);
                    pos++;
                }
            } else {
                // Color cache lookup
                int cacheIndex = green - NUM_LITERAL_CODES - NUM_LENGTH_CODES;
                int argb = cache.lookup(cacheIndex);
                pixels[pos++] = argb;
            }
        }
    }

    private static int addPixelsForPalette(int a, int b) {
        int alpha = (((a >> 24) & 0xFF) + ((b >> 24) & 0xFF)) & 0xFF;
        int red = (((a >> 16) & 0xFF) + ((b >> 16) & 0xFF)) & 0xFF;
        int green = (((a >> 8) & 0xFF) + ((b >> 8) & 0xFF)) & 0xFF;
        int blue = ((a & 0xFF) + (b & 0xFF)) & 0xFF;
        return (alpha << 24) | (red << 16) | (green << 8) | blue;
    }

}
