package dev.simplified.image.codec.webp.lossy;

import dev.simplified.image.pixel.PixelBuffer;
import dev.simplified.image.exception.ImageDecodeException;
import org.jetbrains.annotations.NotNull;

/**
 * Pure Java VP8 (WebP lossy) keyframe decoder.
 * <p>
 * Parses the frame/segment/filter/quant/probability headers exactly as libwebp's
 * {@code VP8GetHeaders} does, then walks the macroblock grid consuming per-MB
 * modes from the first partition and coefficients from the token partition via
 * {@link VP8TokenDecoder}. Coefficients are dequantized with the
 * {@link VP8Tables#DC_Q_LOOKUP} / {@link VP8Tables#AC_Q_LOOKUP} /
 * {@link VP8Tables#AC2_Q_LOOKUP} tables, inverse-transformed, added to the
 * predicted samples, and clamped to produce the reconstructed Y/U/V planes.
 * <p>
 * Supports the four 16x16 luma modes (DC, V, H, TM), B_PRED with all ten 4x4
 * sub-block modes, the four 8x8 chroma modes, and per-MB skip bits. Only
 * keyframes are supported.
 *
 * @see <a href="https://datatracker.ietf.org/doc/html/rfc6386">RFC 6386</a>
 */
public final class VP8Decoder {

    private VP8Decoder() { }

    /** Per-frame decoder state threaded through the MB loop. */
    private static final class State {
        final int mbCols, mbRows;
        final int lumaStride, chromaStride;

        final short[] reconY, reconU, reconV;

        // Non-zero context trackers - mirror VP8Encoder.State exactly:
        //   topNz[mbX][0..3] luma sub-block columns (bottom row of MB above)
        //   topNz[mbX][4..5] U sub-block columns
        //   topNz[mbX][6..7] V sub-block columns
        //   topNz[mbX][8]    Y2 block (for i16x16 MBs)
        // leftNz[0..3]=luma rows, [4..5]=U rows, [6..7]=V rows, [8]=Y2.
        final int[][] topNz;
        final int[] leftNz = new int[9];

        // B_PRED neighbor sub-block modes (bottom row of MB above, right column of left MB).
        final int[] intraT;
        final int[] intraL = new int[4];

        // Per-MB is_i4x4 flag (row-major {@code mbY * mbCols + mbX}), consumed by the loop filter
        // to decide whether to filter the three inner 4x4 sub-block edges inside each MB.
        final boolean[] mbIsI4x4;

        // Dequantizer steps (filled after VP8ParseQuant).
        int y1Dc, y1Ac, y2Dc, y2Ac, uvDc, uvAc;

        // Coefficient probabilities after parsing the 1056 update bits.
        final int[][][][] proba;

        // Per-MB skip bit state.
        boolean useSkipProba;
        int skipP;

        // Segment map state (needed to correctly consume per-MB segment bits).
        boolean updateMap;
        final int[] segmentProbs = { 255, 255, 255 };

        State(int width, int height) {
            this.mbCols = (width + 15) / 16;
            this.mbRows = (height + 15) / 16;
            this.lumaStride = mbCols * 16;
            this.chromaStride = mbCols * 8;
            this.reconY = new short[lumaStride * mbRows * 16];
            this.reconU = new short[chromaStride * mbRows * 8];
            this.reconV = new short[chromaStride * mbRows * 8];
            this.topNz = new int[mbCols][9];
            this.intraT = new int[mbCols * 4];
            this.mbIsI4x4 = new boolean[mbCols * mbRows];
            this.proba = cloneProba(VP8Tables.COEFFS_PROBA_0);
        }
    }

    /**
     * Decodes a VP8 keyframe bitstream into pixel data.
     *
     * @param data the raw VP8 payload
     * @return the decoded pixel buffer
     * @throws ImageDecodeException if the bitstream is malformed or uses unsupported features
     */
    public static @NotNull PixelBuffer decode(byte @NotNull [] data) {
        if (data.length < 10)
            throw new ImageDecodeException("VP8 data too short");

        int frameTag = (data[0] & 0xFF) | ((data[1] & 0xFF) << 8) | ((data[2] & 0xFF) << 16);
        boolean keyFrame = (frameTag & 0x01) == 0;
        int firstPartSize = (frameTag >>> 5) & 0x7FFFF;

        if (!keyFrame)
            throw new ImageDecodeException("VP8 inter-frames not supported (only key frames)");

        if (data[3] != (byte) 0x9D || data[4] != (byte) 0x01 || data[5] != (byte) 0x2A)
            throw new ImageDecodeException("Invalid VP8 sync code");

        int width = ((data[6] & 0xFF) | ((data[7] & 0xFF) << 8)) & 0x3FFF;
        int height = ((data[8] & 0xFF) | ((data[9] & 0xFF) << 8)) & 0x3FFF;
        if (width == 0 || height == 0)
            throw new ImageDecodeException("Invalid VP8 dimensions: %dx%d", width, height);

        int headerStart = 10;
        int partitionEnd = Math.min(headerStart + firstPartSize, data.length);
        BooleanDecoder br = new BooleanDecoder(data, headerStart, partitionEnd - headerStart);

        State s = new State(width, height);

        // ── Frame header (keyframe only) ──
        br.decodeBool();   // colorspace
        br.decodeBool();   // clamp_type

        // ── Segment header ──
        boolean useSegment = br.decodeBool() != 0;
        if (useSegment) {
            s.updateMap = br.decodeBool() != 0;
            if (br.decodeBool() != 0) {                       // update_data
                br.decodeBool();                              // absolute_delta
                for (int i = 0; i < 4; i++)
                    if (br.decodeBool() != 0) br.decodeSint(7);
                for (int i = 0; i < 4; i++)
                    if (br.decodeBool() != 0) br.decodeSint(6);
            }
            if (s.updateMap) {
                for (int i = 0; i < 3; i++)
                    s.segmentProbs[i] = br.decodeBool() != 0 ? br.decodeUint(8) : 255;
            }
        }

        // ── Filter header ──
        boolean simpleFilter = br.decodeBool() != 0;
        int filterLevel = br.decodeUint(6);
        int sharpness = br.decodeUint(3);
        if (br.decodeBool() != 0) {                           // use_lf_delta
            if (br.decodeBool() != 0) {                       // update_lf_delta
                for (int i = 0; i < 4; i++)
                    if (br.decodeBool() != 0) br.decodeSint(6); // ref_lf_delta
                for (int i = 0; i < 4; i++)
                    if (br.decodeBool() != 0) br.decodeSint(6); // mode_lf_delta
            }
        }

        // ── Token partitions ──
        int numTokenPartitions = 1 << br.decodeUint(2);

        // ── VP8ParseQuant ──
        int baseQ = br.decodeUint(7);
        int dqy1Dc = br.decodeBool() != 0 ? br.decodeSint(4) : 0;
        int dqy2Dc = br.decodeBool() != 0 ? br.decodeSint(4) : 0;
        int dqy2Ac = br.decodeBool() != 0 ? br.decodeSint(4) : 0;
        int dquvDc = br.decodeBool() != 0 ? br.decodeSint(4) : 0;
        int dquvAc = br.decodeBool() != 0 ? br.decodeSint(4) : 0;

        s.y1Dc = VP8Tables.DC_Q_LOOKUP[Math.clamp(baseQ + dqy1Dc, 0, 127)];
        s.y1Ac = VP8Tables.AC_Q_LOOKUP[Math.clamp(baseQ, 0, 127)];
        s.y2Dc = VP8Tables.DC_Q_LOOKUP[Math.clamp(baseQ + dqy2Dc, 0, 127)] * 2;
        s.y2Ac = Math.max(8, VP8Tables.AC2_Q_LOOKUP[Math.clamp(baseQ + dqy2Ac, 0, 127)]);
        s.uvDc = VP8Tables.DC_Q_LOOKUP[Math.clamp(baseQ + dquvDc, 0, 117)];
        s.uvAc = VP8Tables.AC_Q_LOOKUP[Math.clamp(baseQ + dquvAc, 0, 127)];

        // refresh_entropy_probs (value ignored on keyframes, but the bit is still consumed).
        br.decodeBool();

        // VP8ParseProba: 1056 probability update bits.
        for (int t = 0; t < VP8Tables.NUM_TYPES; t++)
            for (int b = 0; b < VP8Tables.NUM_BANDS; b++)
                for (int c = 0; c < VP8Tables.NUM_CTX; c++)
                    for (int p = 0; p < VP8Tables.NUM_PROBAS; p++)
                        if (br.decodeBit(VP8Tables.COEFFS_UPDATE_PROBA[t][b][c][p]) != 0)
                            s.proba[t][b][c][p] = br.decodeUint(8);

        s.useSkipProba = br.decodeBool() != 0;
        if (s.useSkipProba)
            s.skipP = br.decodeUint(8);

        // ── Token partition decoder ──
        int tokenOffset = partitionEnd;
        if (numTokenPartitions > 1)
            tokenOffset += (numTokenPartitions - 1) * 3;
        if (tokenOffset > data.length)
            throw new ImageDecodeException("VP8 token partition truncated");
        BooleanDecoder td = new BooleanDecoder(data, tokenOffset, data.length - tokenOffset);

        // ── Macroblock loop ──
        for (int mbY = 0; mbY < s.mbRows; mbY++) {
            java.util.Arrays.fill(s.leftNz, 0);
            java.util.Arrays.fill(s.intraL, IntraPrediction.B_DC_PRED);

            for (int mbX = 0; mbX < s.mbCols; mbX++)
                decodeMacroblock(s, br, td, mbX, mbY);
        }

        // Loop filter. Applied whether we emitted simple=1 (our encoder) or a libwebp-
        // produced frame set simple=0. Chroma planes are passed for the normal filter path;
        // the simple path ignores them.
        if (filterLevel > 0) {
            LoopFilter.filterFrame(
                s.reconY, s.reconU, s.reconV,
                s.lumaStride, s.chromaStride,
                s.mbCols, s.mbRows,
                simpleFilter, filterLevel, sharpness, s.mbIsI4x4
            );
        }

        // ── YCbCr -> ARGB (fancy bilinear chroma upsampling, matching libwebp) ──
        int[] pixels = new int[width * height];
        ChromaUpsampler.upsampleToArgb(
            s.reconY, s.reconU, s.reconV,
            width, height, s.lumaStride, s.chromaStride, pixels
        );

        return PixelBuffer.of(pixels, width, height);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Per-macroblock parse + reconstruct
    // ──────────────────────────────────────────────────────────────────────

    private static void decodeMacroblock(
        @NotNull State s, @NotNull BooleanDecoder br, @NotNull BooleanDecoder td,
        int mbX, int mbY
    ) {
        // Segment map (consume bits; value unused since we don't track per-segment state).
        if (s.updateMap) {
            if (br.decodeBit(s.segmentProbs[0]) == 0)
                br.decodeBit(s.segmentProbs[1]);
            else
                br.decodeBit(s.segmentProbs[2]);
        }

        int skip = (s.useSkipProba && br.decodeBit(s.skipP) != 0) ? 1 : 0;

        // is_i4x4 - bit=0 means i4x4, bit=1 means 16x16 prediction (libwebp convention).
        boolean isI4x4 = br.decodeBit(145) == 0;
        s.mbIsI4x4[mbY * s.mbCols + mbX] = isI4x4;

        int yMode;
        int[][] subModes = null;

        if (!isI4x4) {
            // Hardcoded 16x16 intra-mode decision tree.
            if (br.decodeBit(156) != 0)
                yMode = br.decodeBit(128) != 0 ? IntraPrediction.TM_PRED : IntraPrediction.H_PRED;
            else
                yMode = br.decodeBit(163) != 0 ? IntraPrediction.V_PRED : IntraPrediction.DC_PRED;

            // Fill sub-block neighbor context with the analogous B_PRED mode.
            int ctxMode = mb16ToSubMode(yMode);
            java.util.Arrays.fill(s.intraT, mbX * 4, mbX * 4 + 4, ctxMode);
            java.util.Arrays.fill(s.intraL, ctxMode);
        } else {
            yMode = IntraPrediction.B_PRED;
            subModes = new int[4][4];
            for (int by = 0; by < 4; by++) {
                for (int bx = 0; bx < 4; bx++) {
                    int above = by > 0 ? subModes[by - 1][bx] : s.intraT[mbX * 4 + bx];
                    int leftM = bx > 0 ? subModes[by][bx - 1] : s.intraL[by];
                    subModes[by][bx] = br.decodeTree(VP8Tables.BMODE_TREE, VP8Tables.KF_BMODE_PROB[above][leftM]);
                }
            }
            for (int bx = 0; bx < 4; bx++) s.intraT[mbX * 4 + bx] = subModes[3][bx];
            for (int by = 0; by < 4; by++) s.intraL[by] = subModes[by][3];
        }

        // UV mode tree.
        int uvMode;
        if (br.decodeBit(142) == 0)
            uvMode = IntraPrediction.DC_PRED;
        else if (br.decodeBit(114) == 0)
            uvMode = IntraPrediction.V_PRED;
        else
            uvMode = br.decodeBit(183) != 0 ? IntraPrediction.TM_PRED : IntraPrediction.H_PRED;

        // ── Parse residuals ──
        int[] top = s.topNz[mbX];
        int[] left = s.leftNz;

        short[] dcValues = new short[16];
        short[][] yCoefs = new short[16][16];
        short[][] uCoefs = new short[4][16];
        short[][] vCoefs = new short[4][16];

        if (skip != 0) {
            // All residuals are zero; clear nz contexts and fall through to reconstruction.
            for (int i = 0; i < 9; i++) { top[i] = 0; left[i] = 0; }
        } else {
            int first;
            int acType;

            if (!isI4x4) {
                short[] zz = new short[16];
                int ctx = top[8] + left[8];
                int nz = VP8TokenDecoder.decode(td, zz, 0, ctx, VP8Tables.TYPE_I16_DC, s.proba);
                top[8] = left[8] = nz;

                short[] y2Raster = zigzagToRaster(zz);
                y2Raster[0] = (short) (y2Raster[0] * s.y2Dc);
                for (int i = 1; i < 16; i++) y2Raster[i] = (short) (y2Raster[i] * s.y2Ac);
                DCT.inverseWHT(y2Raster, dcValues);

                first = 1;
                acType = VP8Tables.TYPE_I16_AC;
            } else {
                first = 0;
                acType = VP8Tables.TYPE_I4_AC;
            }

            // 16 Y blocks - raster order (y outer, x inner) matching VP8Encoder.emitMbTokens.
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 4; x++) {
                    short[] zz = new short[16];
                    int ctx = top[x] + left[y];
                    int nz = VP8TokenDecoder.decode(td, zz, first, ctx, acType, s.proba);
                    top[x] = left[y] = nz;

                    short[] raster = zigzagToRaster(zz);
                    if (!isI4x4) {
                        // DC comes from the Y2 WHT block; raster[0] at first=1 is already zero.
                        raster[0] = dcValues[y * 4 + x];
                    } else {
                        raster[0] = (short) (raster[0] * s.y1Dc);
                    }
                    for (int i = 1; i < 16; i++) raster[i] = (short) (raster[i] * s.y1Ac);
                    yCoefs[y * 4 + x] = raster;
                }
            }

            // U blocks (2x2).
            for (int y = 0; y < 2; y++) {
                for (int x = 0; x < 2; x++) {
                    short[] zz = new short[16];
                    int ctx = top[4 + x] + left[4 + y];
                    int nz = VP8TokenDecoder.decode(td, zz, 0, ctx, VP8Tables.TYPE_CHROMA_A, s.proba);
                    top[4 + x] = left[4 + y] = nz;

                    short[] raster = zigzagToRaster(zz);
                    raster[0] = (short) (raster[0] * s.uvDc);
                    for (int i = 1; i < 16; i++) raster[i] = (short) (raster[i] * s.uvAc);
                    uCoefs[y * 2 + x] = raster;
                }
            }
            // V blocks (2x2).
            for (int y = 0; y < 2; y++) {
                for (int x = 0; x < 2; x++) {
                    short[] zz = new short[16];
                    int ctx = top[6 + x] + left[6 + y];
                    int nz = VP8TokenDecoder.decode(td, zz, 0, ctx, VP8Tables.TYPE_CHROMA_A, s.proba);
                    top[6 + x] = left[6 + y] = nz;

                    short[] raster = zigzagToRaster(zz);
                    raster[0] = (short) (raster[0] * s.uvDc);
                    for (int i = 1; i < 16; i++) raster[i] = (short) (raster[i] * s.uvAc);
                    vCoefs[y * 2 + x] = raster;
                }
            }
        }

        // ── Reconstruct ──
        if (isI4x4)
            reconstructI4x4(s, mbX, mbY, subModes, yCoefs);
        else
            reconstruct16x16(s, mbX, mbY, yMode, yCoefs);

        reconstructChroma(s.reconU, s.chromaStride, mbX, mbY, uvMode, uCoefs);
        reconstructChroma(s.reconV, s.chromaStride, mbX, mbY, uvMode, vCoefs);
    }

    /** Maps a 16x16 luma macroblock mode to the corresponding sub-block mode for B_PRED neighbor context. */
    private static int mb16ToSubMode(int mbMode) {
        return switch (mbMode) {
            case IntraPrediction.V_PRED -> IntraPrediction.B_VE_PRED;
            case IntraPrediction.H_PRED -> IntraPrediction.B_HE_PRED;
            case IntraPrediction.TM_PRED -> IntraPrediction.B_TM_PRED;
            default -> IntraPrediction.B_DC_PRED;
        };
    }

    private static short[] zigzagToRaster(short @NotNull [] zz) {
        short[] raster = new short[16];
        for (int i = 0; i < 16; i++) raster[VP8Tables.ZIGZAG[i]] = zz[i];
        return raster;
    }

    private static void reconstruct16x16(
        @NotNull State s, int mbX, int mbY, int yMode, short @NotNull [] @NotNull [] yCoefs
    ) {
        short[] above = mbY > 0 ? extractRow(s.reconY, mbX * 16, (mbY * 16 - 1) * s.lumaStride, 16) : null;
        short[] leftCol = mbX > 0 ? extractCol(s.reconY, mbX * 16 - 1, mbY * 16, s.lumaStride, 16) : null;
        short aboveLeft = (mbX > 0 && mbY > 0)
            ? s.reconY[(mbY * 16 - 1) * s.lumaStride + mbX * 16 - 1] : (short) 128;

        short[] predicted = new short[256];
        IntraPrediction.predict16x16(predicted, above, leftCol, aboveLeft, yMode);

        for (int by = 0; by < 4; by++) {
            for (int bx = 0; bx < 4; bx++) {
                short[] residual = new short[16];
                DCT.inverseDCT(yCoefs[by * 4 + bx], residual);

                int baseX = mbX * 16 + bx * 4;
                int baseY = mbY * 16 + by * 4;
                for (int y = 0; y < 4; y++)
                    for (int x = 0; x < 4; x++)
                        s.reconY[(baseY + y) * s.lumaStride + baseX + x] = (short) Math.clamp(
                            predicted[(by * 4 + y) * 16 + bx * 4 + x] + residual[y * 4 + x], 0, 255
                        );
            }
        }
    }

    private static void reconstructI4x4(
        @NotNull State s, int mbX, int mbY,
        int @NotNull [] @NotNull [] subModes, short @NotNull [] @NotNull [] yCoefs
    ) {
        for (int by = 0; by < 4; by++) {
            for (int bx = 0; bx < 4; bx++) {
                int blockX = mbX * 16 + bx * 4;
                int blockY = mbY * 16 + by * 4;

                short[] above8 = getAbove8(s.reconY, blockX, blockY, s.lumaStride);
                short[] left4 = getLeft4(s.reconY, blockX, blockY, s.lumaStride);
                short aboveLeft = getAboveLeft(s.reconY, blockX, blockY, s.lumaStride);

                short[] predicted = new short[16];
                IntraPrediction.predict4x4(predicted, above8, left4, aboveLeft, subModes[by][bx]);

                short[] residual = new short[16];
                DCT.inverseDCT(yCoefs[by * 4 + bx], residual);

                for (int y = 0; y < 4; y++)
                    for (int x = 0; x < 4; x++)
                        s.reconY[(blockY + y) * s.lumaStride + blockX + x] = (short) Math.clamp(
                            predicted[y * 4 + x] + residual[y * 4 + x], 0, 255
                        );
            }
        }
    }

    private static void reconstructChroma(
        short @NotNull [] plane, int stride, int mbX, int mbY, int uvMode,
        short @NotNull [] @NotNull [] coefs
    ) {
        short[] above = mbY > 0 ? extractRow(plane, mbX * 8, (mbY * 8 - 1) * stride, 8) : null;
        short[] leftCol = mbX > 0 ? extractCol(plane, mbX * 8 - 1, mbY * 8, stride, 8) : null;
        short aboveLeft = (mbX > 0 && mbY > 0)
            ? plane[(mbY * 8 - 1) * stride + mbX * 8 - 1] : (short) 128;

        short[] predicted = new short[64];
        IntraPrediction.predict8x8(predicted, above, leftCol, aboveLeft, uvMode);

        for (int by = 0; by < 2; by++) {
            for (int bx = 0; bx < 2; bx++) {
                short[] residual = new short[16];
                DCT.inverseDCT(coefs[by * 2 + bx], residual);

                int baseX = mbX * 8 + bx * 4;
                int baseY = mbY * 8 + by * 4;
                for (int y = 0; y < 4; y++)
                    for (int x = 0; x < 4; x++)
                        plane[(baseY + y) * stride + baseX + x] = (short) Math.clamp(
                            predicted[(by * 4 + y) * 8 + bx * 4 + x] + residual[y * 4 + x], 0, 255
                        );
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Neighbor extraction helpers
    // ──────────────────────────────────────────────────────────────────────

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
     * Gets 8 pixels above a 4x4 sub-block: 4 above plus 4 above-right.
     * <p>
     * For sub-blocks whose above-row falls inside the current MB row
     * ({@code blockY % 16 != 0}), the right neighbour MB is not yet reconstructed,
     * so above-right reads are clamped to the current MB's right edge - replicating
     * the rightmost in-MB pixel. For top-of-MB sub-blocks the above-row is the prior
     * MB row's bottom row (fully reconstructed), so we clamp at plane edge.
     */
    private static short[] getAbove8(short[] plane, int blockX, int blockY, int stride) {
        if (blockY == 0) return null;
        short[] above = new short[8];
        int aboveRow = blockY - 1;
        boolean withinMbRow = (blockY % 16) != 0;
        int rightLimit = withinMbRow ? ((blockX / 16) * 16 + 15) : (stride - 1);
        for (int i = 0; i < 8; i++) {
            int x = Math.min(blockX + i, rightLimit);
            above[i] = plane[aboveRow * stride + x];
        }
        return above;
    }

    private static short[] getLeft4(short[] plane, int blockX, int blockY, int stride) {
        if (blockX == 0) return null;
        short[] left = new short[4];
        for (int i = 0; i < 4; i++)
            left[i] = plane[(blockY + i) * stride + blockX - 1];
        return left;
    }

    private static short getAboveLeft(short[] plane, int blockX, int blockY, int stride) {
        if (blockX == 0 || blockY == 0) return 128;
        return plane[(blockY - 1) * stride + blockX - 1];
    }

    /** Deep-copies the static default proba table so per-frame updates do not mutate the source. */
    private static int[][][][] cloneProba(int[][][][] src) {
        int[][][][] out = new int[src.length][][][];
        for (int t = 0; t < src.length; t++) {
            out[t] = new int[src[t].length][][];
            for (int b = 0; b < src[t].length; b++) {
                out[t][b] = new int[src[t][b].length][];
                for (int c = 0; c < src[t][b].length; c++)
                    out[t][b][c] = src[t][b][c].clone();
            }
        }
        return out;
    }

}
