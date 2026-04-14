package dev.simplified.image.codec.webp.lossy;

import dev.simplified.image.pixel.PixelBuffer;
import dev.simplified.image.exception.ImageDecodeException;
import org.jetbrains.annotations.NotNull;

/**
 * Pure Java VP8 (WebP lossy) decoder.
 * <p>
 * Decodes a VP8 bitstream into ARGB pixel data using boolean arithmetic
 * decoding, intra-frame prediction, inverse DCT, and loop filtering.
 * Supports both 16x16 prediction modes and B_PRED (per-sub-block 4x4
 * prediction with all 10 intra modes).
 */
public final class VP8Decoder {

    // @formatter:off

    /** Key frame luma mode tree (5 leaves: B_PRED, DC, V, H, TM). */
    private static final int[] KF_YMODE_TREE = {
        -IntraPrediction.B_PRED, 2,
        -IntraPrediction.DC_PRED, 4,
        -IntraPrediction.V_PRED, 6,
        -IntraPrediction.H_PRED, -IntraPrediction.TM_PRED
    };

    /** Key frame luma mode probabilities (one per internal tree node). */
    private static final int[] KF_YMODE_PROB = { 145, 156, 163, 128 };

    /** Sub-block mode tree (10 leaves: B_DC, B_TM, B_VE, B_HE, B_LD, B_RD, B_VR, B_VL, B_HD, B_HU). */
    private static final int[] BMODE_TREE = {
        -IntraPrediction.B_DC_PRED, 2,
        -IntraPrediction.B_TM_PRED, 4,
        -IntraPrediction.B_VE_PRED, 6,
         8, 12,
        -IntraPrediction.B_HE_PRED, 10,
        -IntraPrediction.B_RD_PRED, -IntraPrediction.B_VR_PRED,
        -IntraPrediction.B_LD_PRED, 14,
        -IntraPrediction.B_VL_PRED, 16,
        -IntraPrediction.B_HD_PRED, -IntraPrediction.B_HU_PRED
    };

    /**
     * Key frame sub-block mode probabilities indexed by {@code [above_mode][left_mode]}.
     * Each entry has 9 probabilities for the 9 internal nodes of the bmode tree.
     * <p>
     * From RFC 6386, Section 12.1.
     */
    private static final int[][][] KF_BMODE_PROB = {
        /* above = B_DC_PRED (0) */
        {
            { 231, 120, 48, 89, 115, 113, 120, 152, 112 },
            { 152, 179, 64, 126, 170, 118, 46, 70, 95 },
            { 175, 69, 143, 80, 85, 82, 72, 155, 103 },
            { 56, 58, 10, 171, 218, 189, 17, 13, 152 },
            { 144, 71, 10, 38, 171, 213, 144, 34, 26 },
            { 114, 26, 17, 163, 44, 195, 21, 10, 173 },
            { 121, 24, 80, 195, 26, 62, 44, 64, 85 },
            { 170, 46, 55, 19, 136, 160, 33, 206, 71 },
            { 63, 20, 8, 114, 114, 208, 12, 9, 226 },
            { 81, 40, 11, 96, 182, 84, 29, 16, 36 },
        },
        /* above = B_TM_PRED (1) */
        {
            { 134, 183, 89, 137, 98, 101, 106, 165, 148 },
            { 72, 187, 100, 130, 157, 111, 32, 75, 80 },
            { 66, 102, 167, 99, 74, 62, 40, 234, 128 },
            { 41, 53, 9, 178, 241, 141, 26, 8, 107 },
            { 104, 79, 12, 27, 217, 255, 87, 17, 7 },
            { 74, 43, 26, 146, 73, 166, 49, 23, 157 },
            { 65, 38, 105, 160, 51, 52, 31, 115, 128 },
            { 87, 68, 71, 44, 114, 51, 15, 186, 23 },
            { 47, 41, 14, 110, 182, 183, 21, 17, 194 },
            { 66, 45, 25, 102, 197, 189, 23, 18, 22 },
        },
        /* above = B_VE_PRED (2) */
        {
            { 88, 88, 147, 150, 42, 46, 45, 196, 205 },
            { 43, 97, 183, 117, 85, 38, 35, 179, 61 },
            { 39, 53, 200, 87, 26, 21, 43, 232, 171 },
            { 56, 34, 51, 104, 114, 102, 29, 93, 77 },
            { 107, 54, 32, 26, 51, 1, 81, 43, 31 },
            { 39, 28, 85, 171, 58, 165, 90, 98, 64 },
            { 34, 22, 116, 206, 23, 34, 43, 166, 73 },
            { 68, 25, 106, 22, 64, 171, 36, 225, 114 },
            { 34, 19, 21, 102, 132, 188, 16, 76, 124 },
            { 62, 18, 78, 95, 85, 57, 50, 48, 51 },
        },
        /* above = B_HE_PRED (3) */
        {
            { 193, 101, 35, 159, 215, 111, 89, 46, 111 },
            { 60, 148, 31, 172, 219, 228, 21, 18, 111 },
            { 112, 113, 77, 85, 179, 255, 38, 120, 114 },
            { 40, 42, 1, 196, 245, 209, 10, 25, 109 },
            { 100, 80, 8, 43, 154, 1, 51, 26, 71 },
            { 88, 43, 29, 140, 166, 213, 37, 43, 154 },
            { 61, 63, 30, 155, 67, 45, 68, 1, 209 },
            { 100, 80, 8, 43, 154, 1, 51, 26, 71 },
            { 68, 35, 6, 142, 208, 189, 14, 1, 168 },
            { 50, 31, 7, 152, 211, 1, 50, 4, 14 },
        },
        /* above = B_LD_PRED (4) */
        {
            { 125, 98, 42, 88, 104, 85, 117, 175, 82 },
            { 95, 84, 53, 89, 128, 100, 113, 101, 45 },
            { 75, 79, 123, 47, 51, 128, 81, 171, 1 },
            { 57, 17, 5, 71, 102, 57, 53, 41, 49 },
            { 115, 21, 2, 10, 102, 255, 166, 23, 6 },
            { 38, 33, 13, 121, 57, 73, 26, 1, 85 },
            { 41, 10, 67, 138, 77, 110, 90, 47, 114 },
            { 101, 29, 16, 10, 85, 128, 101, 196, 26 },
            { 57, 18, 10, 102, 102, 213, 34, 20, 43 },
            { 117, 20, 15, 36, 163, 128, 68, 1, 26 },
        },
        /* above = B_RD_PRED (5) */
        {
            { 102, 61, 71, 37, 34, 53, 31, 243, 192 },
            { 69, 60, 71, 38, 73, 119, 28, 222, 37 },
            { 68, 45, 128, 34, 1, 47, 11, 245, 171 },
            { 62, 17, 19, 70, 146, 85, 55, 62, 70 },
            { 75, 15, 9, 9, 64, 255, 184, 119, 16 },
            { 37, 43, 37, 154, 100, 163, 85, 160, 1 },
            { 63, 9, 92, 136, 28, 64, 32, 201, 85 },
            { 86, 6, 28, 5, 64, 255, 25, 248, 1 },
            { 56, 8, 17, 132, 137, 255, 55, 116, 128 },
            { 58, 15, 20, 82, 135, 57, 26, 121, 40 },
        },
        /* above = B_VR_PRED (6) */
        {
            { 164, 50, 31, 137, 154, 133, 25, 35, 218 },
            { 51, 103, 44, 131, 131, 123, 31, 6, 158 },
            { 86, 40, 64, 135, 148, 224, 45, 183, 128 },
            { 22, 26, 17, 131, 240, 154, 14, 1, 209 },
            { 83, 12, 13, 54, 192, 255, 68, 47, 28 },
            { 45, 16, 21, 91, 64, 222, 7, 1, 197 },
            { 56, 21, 39, 155, 60, 138, 23, 102, 213 },
            { 85, 26, 85, 85, 128, 128, 32, 146, 171 },
            { 18, 11, 7, 63, 144, 171, 4, 4, 246 },
            { 35, 27, 10, 146, 174, 171, 12, 26, 128 },
        },
        /* above = B_VL_PRED (7) */
        {
            { 190, 80, 35, 99, 180, 80, 126, 54, 45 },
            { 85, 126, 47, 87, 176, 51, 41, 20, 32 },
            { 101, 75, 128, 139, 118, 146, 116, 128, 85 },
            { 56, 41, 15, 176, 236, 85, 37, 9, 62 },
            { 146, 36, 19, 30, 171, 255, 97, 27, 20 },
            { 71, 30, 17, 119, 118, 255, 17, 18, 138 },
            { 101, 38, 60, 138, 55, 70, 43, 26, 142 },
            { 138, 45, 61, 62, 219, 1, 81, 188, 64 },
            { 32, 41, 20, 117, 151, 142, 20, 21, 163 },
            { 112, 19, 12, 61, 195, 128, 48, 4, 24 },
        },
        /* above = B_HD_PRED (8) */
        {
            { 104, 55, 44, 218, 9, 54, 53, 130, 226 },
            { 64, 90, 70, 205, 40, 41, 23, 26, 57 },
            { 54, 57, 112, 184, 5, 41, 38, 166, 213 },
            { 30, 34, 26, 133, 152, 116, 10, 32, 134 },
            { 75, 32, 12, 51, 192, 255, 160, 43, 51 },
            { 39, 19, 53, 221, 26, 114, 32, 73, 255 },
            { 31, 9, 65, 234, 2, 15, 1, 118, 73 },
            { 88, 31, 35, 67, 102, 85, 55, 186, 85 },
            { 56, 21, 23, 111, 59, 205, 45, 37, 192 },
            { 55, 38, 70, 124, 73, 102, 1, 34, 98 },
        },
        /* above = B_HU_PRED (9) */
        {
            { 102, 61, 71, 37, 34, 53, 31, 243, 192 },
            { 69, 60, 71, 38, 73, 119, 28, 222, 37 },
            { 68, 45, 128, 34, 1, 47, 11, 245, 171 },
            { 62, 17, 19, 70, 146, 85, 55, 62, 70 },
            { 75, 15, 9, 9, 64, 255, 184, 119, 16 },
            { 37, 43, 37, 154, 100, 163, 85, 160, 1 },
            { 63, 9, 92, 136, 28, 64, 32, 201, 85 },
            { 86, 6, 28, 5, 64, 255, 25, 248, 1 },
            { 56, 8, 17, 132, 137, 255, 55, 116, 128 },
            { 58, 15, 20, 82, 135, 57, 26, 121, 40 },
        },
    };

    // @formatter:on

    private VP8Decoder() { }

    /**
     * Decodes a VP8 bitstream into pixel data.
     *
     * @param data the raw VP8 payload
     * @return the decoded pixel buffer
     * @throws ImageDecodeException if decoding fails
     */
    public static @NotNull PixelBuffer decode(byte @NotNull [] data) {
        if (data.length < 10)
            throw new ImageDecodeException("VP8 data too short");

        // Parse frame tag (3 bytes)
        int frameTag = (data[0] & 0xFF) | ((data[1] & 0xFF) << 8) | ((data[2] & 0xFF) << 16);
        boolean keyFrame = (frameTag & 0x01) == 0;
        int firstPartSize = (frameTag >> 5) & 0x7FFFF;

        if (!keyFrame)
            throw new ImageDecodeException("VP8 inter-frames not supported (only key frames)");

        // Key frame header (7 bytes: 3 sync + 2 width + 2 height)
        int offset = 3;

        if (data.length < offset + 7)
            throw new ImageDecodeException("VP8 key frame header too short");

        // Sync code
        if (data[offset] != (byte) 0x9D || data[offset + 1] != (byte) 0x01 || data[offset + 2] != (byte) 0x2A)
            throw new ImageDecodeException("Invalid VP8 sync code");

        offset += 3;

        int widthField = (data[offset] & 0xFF) | ((data[offset + 1] & 0xFF) << 8);
        int heightField = (data[offset + 2] & 0xFF) | ((data[offset + 3] & 0xFF) << 8);
        int width = widthField & 0x3FFF;
        int height = heightField & 0x3FFF;

        offset += 4;

        if (width == 0 || height == 0)
            throw new ImageDecodeException("Invalid VP8 dimensions: %dx%d", width, height);

        // Decode first partition (header + macroblock modes)
        int partitionEnd = Math.min(offset + firstPartSize, data.length);
        BooleanDecoder headerDecoder = new BooleanDecoder(data, offset, partitionEnd - offset);

        // Colorspace and clamping (consumed but not used - VP8 always BT.601)
        headerDecoder.decodeBool();
        headerDecoder.decodeBool();

        // Skip segmentation
        if (headerDecoder.decodeBool() != 0) {
            int updateMap = headerDecoder.decodeBool();
            if (headerDecoder.decodeBool() != 0) {
                headerDecoder.decodeBool(); // abs or delta
                for (int i = 0; i < 4; i++)
                    if (headerDecoder.decodeBool() != 0) headerDecoder.decodeSint(7);
                for (int i = 0; i < 4; i++)
                    if (headerDecoder.decodeBool() != 0) headerDecoder.decodeSint(6);
            }
            if (updateMap != 0)
                for (int i = 0; i < 3; i++)
                    if (headerDecoder.decodeBool() != 0) headerDecoder.decodeUint(8);
        }

        // Filter parameters
        headerDecoder.decodeBool(); // simple vs normal filter (only simple implemented)
        int filterLevel = headerDecoder.decodeUint(6);
        int sharpness = headerDecoder.decodeUint(3);

        if (headerDecoder.decodeBool() != 0) { // filter adjust
            if (headerDecoder.decodeBool() != 0)
                for (int i = 0; i < 8; i++)
                    if (headerDecoder.decodeBool() != 0) headerDecoder.decodeSint(6);
        }

        // Token partitions
        int numTokenPartitions = 1 << headerDecoder.decodeUint(2);

        // Quantizer indices
        int yAcQi = headerDecoder.decodeUint(7);
        int yDcDelta = headerDecoder.decodeBool() != 0 ? headerDecoder.decodeSint(4) : 0;
        int y2DcDelta = headerDecoder.decodeBool() != 0 ? headerDecoder.decodeSint(4) : 0;
        int y2AcDelta = headerDecoder.decodeBool() != 0 ? headerDecoder.decodeSint(4) : 0;
        int uvDcDelta = headerDecoder.decodeBool() != 0 ? headerDecoder.decodeSint(4) : 0;
        int uvAcDelta = headerDecoder.decodeBool() != 0 ? headerDecoder.decodeSint(4) : 0;

        // Build dequantization steps from QI + deltas
        int yDcQ = lookupDc(Math.clamp(yAcQi + yDcDelta, 0, 127));
        int yAcQ = lookupAc(yAcQi);
        int y2DcQ = lookupDc(Math.clamp(yAcQi + y2DcDelta, 0, 127)) * 2;
        int y2AcQ = Math.max(8, lookupAc(Math.clamp(yAcQi + y2AcDelta, 0, 127)) * 155 / 100);
        int uvDcQ = lookupDc(Math.clamp(yAcQi + uvDcDelta, 0, 127));
        int uvAcQ = lookupAc(Math.clamp(yAcQi + uvAcDelta, 0, 127));

        // Macroblock grid
        int mbCols = (width + 15) / 16;
        int mbRows = (height + 15) / 16;
        int lumaWidth = mbCols * 16;
        int chromaWidth = mbCols * 8;

        // Reconstructed planes
        short[] reconLuma = new short[lumaWidth * mbRows * 16];
        short[] reconCb = new short[chromaWidth * mbRows * 8];
        short[] reconCr = new short[chromaWidth * mbRows * 8];

        // Token partition(s) start after header partition
        int tokenOffset = partitionEnd;
        // Skip partition size fields for multi-partition (3 bytes each for partitions 1..N-1)
        if (numTokenPartitions > 1)
            tokenOffset += (numTokenPartitions - 1) * 3;

        BooleanDecoder tokenDecoder = tokenOffset < data.length
            ? new BooleanDecoder(data, tokenOffset, data.length - tokenOffset)
            : null;

        // Sub-block mode context tracking for B_PRED
        int[] aboveSubBlockModes = new int[mbCols * 4]; // bottom row from macroblock row above
        int[] leftSubBlockModes = new int[4]; // right column from macroblock to the left

        // Decode each macroblock
        for (int mbY = 0; mbY < mbRows; mbY++) {
            java.util.Arrays.fill(leftSubBlockModes, IntraPrediction.B_DC_PRED);

            for (int mbX = 0; mbX < mbCols; mbX++) {
                // Read prediction mode from header partition using tree decoding
                int yMode = headerDecoder.decodeTree(KF_YMODE_TREE, KF_YMODE_PROB);
                int uvMode = headerDecoder.decodeUint(2);

                if (yMode == IntraPrediction.B_PRED)
                    decodeBPred(reconLuma, mbX, mbY, lumaWidth, headerDecoder, tokenDecoder,
                        yDcQ, yAcQ, aboveSubBlockModes, leftSubBlockModes);
                else
                    decode16x16(reconLuma, mbX, mbY, lumaWidth, yMode, tokenDecoder,
                        yAcQ, y2DcQ, y2AcQ, aboveSubBlockModes, leftSubBlockModes);

                // Predict + decode residual for chroma
                decodeChromaPlane(reconCb, mbX, mbY, chromaWidth, uvMode, tokenDecoder, uvDcQ, uvAcQ);
                decodeChromaPlane(reconCr, mbX, mbY, chromaWidth, uvMode, tokenDecoder, uvDcQ, uvAcQ);
            }
        }

        // Loop filter
        if (filterLevel > 0)
            LoopFilter.filterSimple(reconLuma, lumaWidth, mbRows * 16, filterLevel, sharpness);

        // Convert YCbCr to ARGB
        int[] pixels = new int[width * height];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int yVal = reconLuma[y * lumaWidth + x] - 16;
                int cbVal = reconCb[(y / 2) * chromaWidth + (x / 2)] - 128;
                int crVal = reconCr[(y / 2) * chromaWidth + (x / 2)] - 128;

                int r = Math.clamp((298 * yVal + 409 * crVal + 128) >> 8, 0, 255);
                int g = Math.clamp((298 * yVal - 100 * cbVal - 208 * crVal + 128) >> 8, 0, 255);
                int b = Math.clamp((298 * yVal + 516 * cbVal + 128) >> 8, 0, 255);

                pixels[y * width + x] = 0xFF000000 | (r << 16) | (g << 8) | b;
            }
        }

        return PixelBuffer.of(pixels, width, height);
    }

    /**
     * Decodes a macroblock using 16x16 prediction with WHT-coded DC coefficients.
     */
    private static void decode16x16(
        short @NotNull [] reconLuma, int mbX, int mbY, int lumaWidth,
        int yMode, BooleanDecoder tokenDecoder,
        int yAcQ, int y2DcQ, int y2AcQ,
        int @NotNull [] aboveSubBlockModes, int @NotNull [] leftSubBlockModes
    ) {
        // 16x16 luma prediction using macroblock-boundary neighbors
        short[] above16 = mbY > 0
            ? extractRow(reconLuma, mbX * 16, (mbY * 16 - 1) * lumaWidth, 16) : null;
        short[] left16 = mbX > 0
            ? extractCol(reconLuma, mbX * 16 - 1, mbY * 16, lumaWidth, 16) : null;
        short aboveLeft16 = (mbX > 0 && mbY > 0)
            ? reconLuma[(mbY * 16 - 1) * lumaWidth + mbX * 16 - 1] : (short) 128;

        short[] predicted16 = new short[256];
        IntraPrediction.predict16x16(predicted16, above16, left16, aboveLeft16, yMode);

        // Decode Y2 block (WHT-coded DC coefficients for luma sub-blocks)
        short[] y2Coeffs = new short[16];
        if (tokenDecoder != null)
            for (int c = 0; c < 16; c++)
                y2Coeffs[c] = (short) tokenDecoder.decodeSint(11);

        y2Coeffs[0] = (short) (y2Coeffs[0] * y2DcQ);
        for (int c = 1; c < 16; c++)
            y2Coeffs[c] = (short) (y2Coeffs[c] * y2AcQ);

        short[] dcValues = new short[16];
        DCT.inverseWHT(y2Coeffs, dcValues);

        // Decode each 4x4 luma sub-block (AC only - DC comes from WHT)
        for (int by = 0; by < 4; by++) {
            for (int bx = 0; bx < 4; bx++) {
                short[] coefficients = new short[16];
                coefficients[0] = dcValues[by * 4 + bx];

                if (tokenDecoder != null)
                    for (int c = 1; c < 16; c++)
                        coefficients[c] = (short) tokenDecoder.decodeSint(11);

                for (int c = 1; c < 16; c++)
                    coefficients[c] = (short) (coefficients[c] * yAcQ);

                short[] residual = new short[16];
                DCT.inverseDCT(coefficients, residual);

                int baseX = mbX * 16 + bx * 4;
                int baseY = mbY * 16 + by * 4;
                for (int y = 0; y < 4; y++)
                    for (int x = 0; x < 4; x++)
                        reconLuma[(baseY + y) * lumaWidth + baseX + x] = (short) Math.clamp(
                            predicted16[(by * 4 + y) * 16 + bx * 4 + x] + residual[y * 4 + x], 0, 255
                        );
            }
        }

        // 16x16 mode macroblocks provide B_DC_PRED context for neighboring B_PRED sub-blocks
        java.util.Arrays.fill(aboveSubBlockModes, mbX * 4, mbX * 4 + 4, IntraPrediction.B_DC_PRED);
        java.util.Arrays.fill(leftSubBlockModes, IntraPrediction.B_DC_PRED);
    }

    /**
     * Decodes a B_PRED macroblock where each 4x4 sub-block has its own prediction mode.
     */
    private static void decodeBPred(
        short @NotNull [] reconLuma, int mbX, int mbY, int lumaWidth,
        @NotNull BooleanDecoder headerDecoder, BooleanDecoder tokenDecoder,
        int yDcQ, int yAcQ,
        int @NotNull [] aboveSubBlockModes, int @NotNull [] leftSubBlockModes
    ) {
        int[][] subModes = new int[4][4];

        for (int by = 0; by < 4; by++) {
            for (int bx = 0; bx < 4; bx++) {
                // Context: mode of the sub-block above and to the left
                int aboveMode = by > 0
                    ? subModes[by - 1][bx]
                    : aboveSubBlockModes[mbX * 4 + bx];
                int leftMode = bx > 0
                    ? subModes[by][bx - 1]
                    : leftSubBlockModes[by];

                // Decode sub-block mode with context-dependent probabilities
                subModes[by][bx] = headerDecoder.decodeTree(BMODE_TREE, KF_BMODE_PROB[aboveMode][leftMode]);

                // Get neighbor pixels for prediction (8-element above includes above-right)
                int blockX = mbX * 16 + bx * 4;
                int blockY = mbY * 16 + by * 4;

                short[] above8 = getAbove8(reconLuma, blockX, blockY, lumaWidth);
                short[] left4 = getLeft4(reconLuma, blockX, blockY, lumaWidth);
                short aboveLeft = getAboveLeft(reconLuma, blockX, blockY, lumaWidth);

                // Predict
                short[] predicted = new short[16];
                IntraPrediction.predict4x4(predicted, above8, left4, aboveLeft, subModes[by][bx]);

                // Decode all 16 coefficients (no WHT for B_PRED)
                short[] coefficients = new short[16];
                if (tokenDecoder != null)
                    for (int c = 0; c < 16; c++)
                        coefficients[c] = (short) tokenDecoder.decodeSint(11);

                // Dequantize with Y1 params
                coefficients[0] = (short) (coefficients[0] * yDcQ);
                for (int c = 1; c < 16; c++)
                    coefficients[c] = (short) (coefficients[c] * yAcQ);

                // Inverse DCT
                short[] residual = new short[16];
                DCT.inverseDCT(coefficients, residual);

                // Reconstruct and store (prediction is per-sub-block, not whole macroblock)
                for (int y = 0; y < 4; y++)
                    for (int x = 0; x < 4; x++)
                        reconLuma[(blockY + y) * lumaWidth + blockX + x] = (short) Math.clamp(
                            predicted[y * 4 + x] + residual[y * 4 + x], 0, 255
                        );
            }
        }

        // Store bottom row and right column for neighboring macroblock context
        for (int bx = 0; bx < 4; bx++)
            aboveSubBlockModes[mbX * 4 + bx] = subModes[3][bx];
        for (int by = 0; by < 4; by++)
            leftSubBlockModes[by] = subModes[by][3];
    }

    private static void decodeChromaPlane(
        short @NotNull [] recon, int mbX, int mbY, int chromaWidth,
        int uvMode, BooleanDecoder tokenDecoder, int dcQ, int acQ
    ) {
        // 8x8 chroma prediction using macroblock-boundary neighbors
        short[] above8 = mbY > 0
            ? extractRow(recon, mbX * 8, (mbY * 8 - 1) * chromaWidth, 8) : null;
        short[] left8 = mbX > 0
            ? extractCol(recon, mbX * 8 - 1, mbY * 8, chromaWidth, 8) : null;
        short aboveLeft8 = (mbX > 0 && mbY > 0)
            ? recon[(mbY * 8 - 1) * chromaWidth + mbX * 8 - 1] : (short) 128;

        short[] predicted8 = new short[64];
        IntraPrediction.predict8x8(predicted8, above8, left8, aboveLeft8, uvMode);

        for (int by = 0; by < 2; by++) {
            for (int bx = 0; bx < 2; bx++) {
                short[] coefficients = new short[16];
                if (tokenDecoder != null)
                    for (int c = 0; c < 16; c++)
                        coefficients[c] = (short) tokenDecoder.decodeSint(11);

                coefficients[0] = (short) (coefficients[0] * dcQ);
                for (int c = 1; c < 16; c++)
                    coefficients[c] = (short) (coefficients[c] * acQ);

                short[] residual = new short[16];
                DCT.inverseDCT(coefficients, residual);

                int baseX = mbX * 8 + bx * 4;
                int baseY = mbY * 8 + by * 4;
                for (int y = 0; y < 4; y++)
                    for (int x = 0; x < 4; x++)
                        recon[(baseY + y) * chromaWidth + baseX + x] = (short) Math.clamp(
                            predicted8[(by * 4 + y) * 8 + bx * 4 + x] + residual[y * 4 + x], 0, 255
                        );
            }
        }
    }

    private static short[] extractRow(short[] plane, int startX, int rowOffset, int count) {
        short[] row = new short[count];
        System.arraycopy(plane, rowOffset + startX, row, 0, count);
        return row;
    }

    private static short[] extractCol(short[] plane, int x, int startY, int stride, int count) {
        short[] col = new short[count];
        for (int i = 0; i < count; i++)
            col[i] = plane[(startY + i) * stride + x];
        return col;
    }

    /**
     * Gets 8 pixels above a 4x4 sub-block (4 above + 4 above-right), clamped at plane edge.
     */
    private static short[] getAbove8(short[] plane, int blockX, int blockY, int stride) {
        if (blockY == 0) return null;
        short[] above = new short[8];
        int aboveRow = blockY - 1;
        for (int i = 0; i < 8; i++) {
            int x = Math.min(blockX + i, stride - 1);
            above[i] = plane[aboveRow * stride + x];
        }
        return above;
    }

    /**
     * Gets 4 pixels to the left of a 4x4 sub-block.
     */
    private static short[] getLeft4(short[] plane, int blockX, int blockY, int stride) {
        if (blockX == 0) return null;
        short[] left = new short[4];
        for (int i = 0; i < 4; i++)
            left[i] = plane[(blockY + i) * stride + blockX - 1];
        return left;
    }

    /**
     * Gets the pixel above-left of a 4x4 sub-block.
     */
    private static short getAboveLeft(short[] plane, int blockX, int blockY, int stride) {
        if (blockX == 0 || blockY == 0) return 128;
        return plane[(blockY - 1) * stride + blockX - 1];
    }

    private static int lookupDc(int qi) {
        return Math.max(1, (qi < 8) ? qi + 2 : (qi < 25) ? (qi * 2 - 4) : (qi * 3 - 26));
    }

    private static int lookupAc(int qi) {
        return Math.max(1, (qi < 4) ? qi + 2 : (qi < 24) ? (qi + qi / 2) : (qi * 2 - 16));
    }

}
