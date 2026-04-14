package dev.simplified.image.codec.webp;

import dev.simplified.image.pixel.PixelBuffer;
import dev.simplified.image.exception.ImageEncodeException;
import org.jetbrains.annotations.NotNull;

/**
 * Pure Java VP8 (WebP lossy) encoder.
 * <p>
 * Encodes ARGB pixel data into a VP8 bitstream using boolean arithmetic
 * coding, rate-distortion optimized intra-frame prediction, forward DCT,
 * and quantization. Each macroblock (16x16) is encoded with the best
 * prediction mode selected via R-D optimization.
 */
final class VP8Encoder {

    private VP8Encoder() { }

    /**
     * Encodes pixel data into a VP8 bitstream.
     *
     * @param pixels the source pixel buffer
     * @param quality the encoding quality (0.0 - 1.0)
     * @return the encoded VP8 payload bytes
     * @throws ImageEncodeException if encoding fails
     */
    static byte @NotNull [] encode(@NotNull PixelBuffer pixels, float quality) {
        int width = pixels.width();
        int height = pixels.height();
        int[] pixelData = pixels.pixels();

        int mbCols = (width + 15) / 16;
        int mbRows = (height + 15) / 16;

        Quantizer quantizer = new Quantizer(quality);
        RateDistortion rd = new RateDistortion(quantizer.getQP());

        // Allocate reconstructed planes for neighbor prediction
        int lumaWidth = mbCols * 16;
        int lumaHeight = mbRows * 16;
        int chromaWidth = mbCols * 8;
        int chromaHeight = mbRows * 8;

        short[] reconLuma = new short[lumaWidth * lumaHeight];
        short[] reconCb = new short[chromaWidth * chromaHeight];
        short[] reconCr = new short[chromaWidth * chromaHeight];

        java.util.Arrays.fill(reconLuma, (short) 128);
        java.util.Arrays.fill(reconCb, (short) 128);
        java.util.Arrays.fill(reconCr, (short) 128);

        // Token buffer for coefficient encoding
        BooleanEncoder headerEncoder = new BooleanEncoder(mbCols * mbRows * 64);
        BooleanEncoder tokenEncoder = new BooleanEncoder(mbCols * mbRows * 256);

        // Encode macroblocks
        for (int mbY = 0; mbY < mbRows; mbY++) {
            for (int mbX = 0; mbX < mbCols; mbX++) {
                Macroblock mb = new Macroblock();
                mb.fromARGB(pixelData, mbX, mbY, width, height);

                // Get neighbor samples for prediction
                short[] aboveRow = mbY > 0 ? extractAboveRow(reconLuma, mbX, mbY, lumaWidth) : null;
                short[] leftCol = mbX > 0 ? extractLeftCol(reconLuma, mbX, mbY, lumaWidth) : null;
                short aboveLeft = (mbX > 0 && mbY > 0) ? reconLuma[(mbY * 16 - 1) * lumaWidth + mbX * 16 - 1] : (short) 128;

                // Select best luma mode
                mb.yMode = rd.selectBest16x16Mode(mb, aboveRow, leftCol, aboveLeft, quantizer);

                // Select best chroma mode
                short[] aboveCb = mbY > 0 ? extractAboveRow8(reconCb, mbX, mbY, chromaWidth) : null;
                short[] leftCb = mbX > 0 ? extractLeftCol8(reconCb, mbX, mbY, chromaWidth) : null;
                short[] aboveCr = mbY > 0 ? extractAboveRow8(reconCr, mbX, mbY, chromaWidth) : null;
                short[] leftCr = mbX > 0 ? extractLeftCol8(reconCr, mbX, mbY, chromaWidth) : null;
                short aboveLeftCb = (mbX > 0 && mbY > 0) ? reconCb[(mbY * 8 - 1) * chromaWidth + mbX * 8 - 1] : (short) 128;
                short aboveLeftCr = (mbX > 0 && mbY > 0) ? reconCr[(mbY * 8 - 1) * chromaWidth + mbX * 8 - 1] : (short) 128;

                mb.uvMode = rd.selectBestChromaMode(
                    mb.cb, mb.cr, aboveCb, leftCb, aboveCr, leftCr, aboveLeftCb, aboveLeftCr
                );

                // Encode prediction mode in header partition
                headerEncoder.encodeUint(mb.yMode, 2);
                headerEncoder.encodeUint(mb.uvMode, 2);

                // Predict, compute residual, DCT, quantize, dequantize, reconstruct
                short[] predicted = new short[256];
                encodeLuma(mb, predicted, aboveRow, leftCol, aboveLeft, quantizer, tokenEncoder);

                short[] predCb = new short[64];
                short[] predCr = new short[64];
                encodeChroma(mb, predCb, predCr, aboveCb, leftCb, aboveCr, leftCr, aboveLeftCb, aboveLeftCr, quantizer, tokenEncoder);

                // Store reconstructed samples for future prediction
                storeRecon(reconLuma, mb.reconY, mbX, mbY, lumaWidth, 16);
                storeRecon(reconCb, mb.reconCb, mbX, mbY, chromaWidth, 8);
                storeRecon(reconCr, mb.reconCr, mbX, mbY, chromaWidth, 8);
            }
        }

        // Apply loop filter
        int filterLevel = Math.clamp(quantizer.getQP() / 2, 0, 63);
        LoopFilter.filterSimple(reconLuma, lumaWidth, lumaHeight, filterLevel, 0);

        // Assemble VP8 bitstream
        return assembleFrame(width, height, quantizer, headerEncoder, tokenEncoder);
    }

    private static void encodeLuma(
        @NotNull Macroblock mb,
        short @NotNull [] predicted,
        short[] aboveRow,
        short[] leftCol,
        short aboveLeft,
        @NotNull Quantizer quantizer,
        @NotNull BooleanEncoder tokenEncoder
    ) {
        // Generate 16x16 prediction
        generatePrediction16x16(predicted, aboveRow, leftCol, aboveLeft, mb.yMode);

        // For each 4x4 sub-block: residual -> DCT -> quantize -> encode -> dequantize -> IDCT -> reconstruct
        for (int by = 0; by < 4; by++) {
            for (int bx = 0; bx < 4; bx++) {
                short[] residual = new short[16];
                short[] coefficients = new short[16];

                for (int y = 0; y < 4; y++)
                    for (int x = 0; x < 4; x++)
                        residual[y * 4 + x] = (short) (mb.y[(by * 4 + y) * 16 + bx * 4 + x]
                            - predicted[(by * 4 + y) * 16 + bx * 4 + x]);

                DCT.forwardDCT(residual, coefficients);
                quantizer.quantizeY(coefficients);

                // Encode coefficients (simplified: encode as fixed-width values)
                for (short c : coefficients)
                    tokenEncoder.encodeSint(c, 11);

                // Reconstruct
                quantizer.dequantizeY(coefficients);
                short[] reconstructed = new short[16];
                DCT.inverseDCT(coefficients, reconstructed);

                for (int y = 0; y < 4; y++)
                    for (int x = 0; x < 4; x++)
                        mb.reconY[(by * 4 + y) * 16 + bx * 4 + x] =
                            (short) clamp(predicted[(by * 4 + y) * 16 + bx * 4 + x] + reconstructed[y * 4 + x]);
            }
        }
    }

    private static void encodeChroma(
        @NotNull Macroblock mb,
        short @NotNull [] predCb, short @NotNull [] predCr,
        short[] aboveCb, short[] leftCb,
        short[] aboveCr, short[] leftCr,
        short aboveLeftCb, short aboveLeftCr,
        @NotNull Quantizer quantizer,
        @NotNull BooleanEncoder tokenEncoder
    ) {
        generatePrediction8x8(predCb, aboveCb, leftCb, aboveLeftCb, mb.uvMode);
        generatePrediction8x8(predCr, aboveCr, leftCr, aboveLeftCr, mb.uvMode);

        encodeChromaPlane(mb.cb, predCb, mb.reconCb, quantizer, tokenEncoder);
        encodeChromaPlane(mb.cr, predCr, mb.reconCr, quantizer, tokenEncoder);
    }

    private static void encodeChromaPlane(
        short @NotNull [] original, short @NotNull [] predicted, short @NotNull [] recon,
        @NotNull Quantizer quantizer, @NotNull BooleanEncoder tokenEncoder
    ) {
        for (int by = 0; by < 2; by++) {
            for (int bx = 0; bx < 2; bx++) {
                short[] residual = new short[16];
                short[] coefficients = new short[16];

                for (int y = 0; y < 4; y++)
                    for (int x = 0; x < 4; x++)
                        residual[y * 4 + x] = (short) (original[(by * 4 + y) * 8 + bx * 4 + x]
                            - predicted[(by * 4 + y) * 8 + bx * 4 + x]);

                DCT.forwardDCT(residual, coefficients);
                quantizer.quantizeUV(coefficients);

                for (short c : coefficients)
                    tokenEncoder.encodeSint(c, 11);

                quantizer.dequantizeUV(coefficients);
                short[] reconstructed = new short[16];
                DCT.inverseDCT(coefficients, reconstructed);

                for (int y = 0; y < 4; y++)
                    for (int x = 0; x < 4; x++) {
                        int i = (by * 4 + y) * 8 + bx * 4 + x;
                        recon[i] = (short) clamp(predicted[i] + reconstructed[y * 4 + x]);
                    }
            }
        }
    }

    private static void generatePrediction16x16(short[] predicted, short[] above, short[] left, short aboveLeft, int mode) {
        switch (mode) {
            case IntraPrediction.DC_PRED -> {
                int sum = 0, count = 0;
                if (above != null) { for (int i = 0; i < 16; i++) sum += above[i]; count += 16; }
                if (left != null) { for (int i = 0; i < 16; i++) sum += left[i]; count += 16; }
                short dc = count > 0 ? (short) ((sum + count / 2) / count) : (short) 128;
                java.util.Arrays.fill(predicted, dc);
            }
            case IntraPrediction.V_PRED -> {
                if (above == null) { java.util.Arrays.fill(predicted, (short) 127); return; }
                for (int y = 0; y < 16; y++) System.arraycopy(above, 0, predicted, y * 16, 16);
            }
            case IntraPrediction.H_PRED -> {
                if (left == null) { java.util.Arrays.fill(predicted, (short) 129); return; }
                for (int y = 0; y < 16; y++) for (int x = 0; x < 16; x++) predicted[y * 16 + x] = left[y];
            }
            case IntraPrediction.TM_PRED -> {
                for (int y = 0; y < 16; y++)
                    for (int x = 0; x < 16; x++) {
                        int a = above != null ? above[x] : 127;
                        int l = left != null ? left[y] : 129;
                        predicted[y * 16 + x] = (short) clamp(a + l - aboveLeft);
                    }
            }
        }
    }

    private static void generatePrediction8x8(short[] predicted, short[] above, short[] left, short aboveLeft, int mode) {
        switch (mode) {
            case IntraPrediction.DC_PRED -> {
                int sum = 0, count = 0;
                if (above != null) { for (int i = 0; i < 8; i++) sum += above[i]; count += 8; }
                if (left != null) { for (int i = 0; i < 8; i++) sum += left[i]; count += 8; }
                short dc = count > 0 ? (short) ((sum + count / 2) / count) : (short) 128;
                java.util.Arrays.fill(predicted, dc);
            }
            case IntraPrediction.V_PRED -> {
                if (above == null) { java.util.Arrays.fill(predicted, (short) 127); return; }
                for (int y = 0; y < 8; y++) System.arraycopy(above, 0, predicted, y * 8, 8);
            }
            case IntraPrediction.H_PRED -> {
                if (left == null) { java.util.Arrays.fill(predicted, (short) 129); return; }
                for (int y = 0; y < 8; y++) for (int x = 0; x < 8; x++) predicted[y * 8 + x] = left[y];
            }
            case IntraPrediction.TM_PRED -> {
                for (int y = 0; y < 8; y++)
                    for (int x = 0; x < 8; x++) {
                        int a = above != null ? above[x] : 127;
                        int l = left != null ? left[y] : 129;
                        predicted[y * 8 + x] = (short) clamp(a + l - aboveLeft);
                    }
            }
        }
    }

    private static short @NotNull [] extractAboveRow(short @NotNull [] plane, int mbX, int mbY, int planeWidth) {
        short[] row = new short[16];
        int y = mbY * 16 - 1;
        int xStart = mbX * 16;
        System.arraycopy(plane, y * planeWidth + xStart, row, 0, 16);
        return row;
    }

    private static short @NotNull [] extractLeftCol(short @NotNull [] plane, int mbX, int mbY, int planeWidth) {
        short[] col = new short[16];
        int x = mbX * 16 - 1;
        int yStart = mbY * 16;
        for (int i = 0; i < 16; i++)
            col[i] = plane[(yStart + i) * planeWidth + x];
        return col;
    }

    private static short @NotNull [] extractAboveRow8(short @NotNull [] plane, int mbX, int mbY, int planeWidth) {
        short[] row = new short[8];
        int y = mbY * 8 - 1;
        int xStart = mbX * 8;
        System.arraycopy(plane, y * planeWidth + xStart, row, 0, 8);
        return row;
    }

    private static short @NotNull [] extractLeftCol8(short @NotNull [] plane, int mbX, int mbY, int planeWidth) {
        short[] col = new short[8];
        int x = mbX * 8 - 1;
        int yStart = mbY * 8;
        for (int i = 0; i < 8; i++)
            col[i] = plane[(yStart + i) * planeWidth + x];
        return col;
    }

    private static void storeRecon(short @NotNull [] plane, short @NotNull [] block, int mbX, int mbY, int planeWidth, int blockSize) {
        int xStart = mbX * blockSize;
        int yStart = mbY * blockSize;
        for (int y = 0; y < blockSize; y++)
            System.arraycopy(block, y * blockSize, plane, (yStart + y) * planeWidth + xStart, blockSize);
    }

    private static byte @NotNull [] assembleFrame(int width, int height, @NotNull Quantizer quantizer,
                                                    @NotNull BooleanEncoder headerEncoder, @NotNull BooleanEncoder tokenEncoder) {
        byte[] headerData = headerEncoder.toByteArray();
        byte[] tokenData = tokenEncoder.toByteArray();

        // Build VP8 frame
        int totalSize = 10 + headerData.length + tokenData.length;
        byte[] frame = new byte[totalSize];
        int offset = 0;

        // Frame tag (3 bytes): key frame, version 0, show frame, first partition size
        int firstPartSize = headerData.length;
        int frameTag = (0) // key frame = 0
            | (0 << 1) // version = 0
            | (1 << 4) // show frame
            | (firstPartSize << 5);
        frame[offset++] = (byte) (frameTag & 0xFF);
        frame[offset++] = (byte) ((frameTag >> 8) & 0xFF);
        frame[offset++] = (byte) ((frameTag >> 16) & 0xFF);

        // Sync code
        frame[offset++] = (byte) 0x9D;
        frame[offset++] = (byte) 0x01;
        frame[offset++] = (byte) 0x2A;

        // Width and height (little-endian 16-bit each)
        frame[offset++] = (byte) (width & 0xFF);
        frame[offset++] = (byte) ((width >> 8) & 0xFF);
        frame[offset++] = (byte) (height & 0xFF);
        frame[offset++] = (byte) ((height >> 8) & 0xFF);

        // Header partition
        System.arraycopy(headerData, 0, frame, offset, headerData.length);
        offset += headerData.length;

        // Token partition
        System.arraycopy(tokenData, 0, frame, offset, tokenData.length);

        return frame;
    }

    private static int clamp(int value) {
        return Math.clamp(value, 0, 255);
    }

}
