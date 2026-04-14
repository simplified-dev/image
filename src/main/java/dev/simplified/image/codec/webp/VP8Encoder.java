package dev.simplified.image.codec.webp;

import dev.simplified.image.pixel.PixelBuffer;
import org.jetbrains.annotations.NotNull;

/**
 * Pure Java VP8 (WebP lossy) encoder.
 * <p>
 * <b>Status: DC_PRED 16x16 encoder.</b> Emits a single keyframe where every
 * macroblock uses 16x16 DC prediction (luma) and 8x8 DC prediction (chroma),
 * with the full VP8 transform + quantize + token-tree pipeline. V_PRED,
 * H_PRED, TM_PRED, and the B_PRED 4x4 sub-block path are future work.
 * <p>
 * Pipeline per macroblock:
 * <ol>
 *   <li>Source ARGB is converted to BT.601 YCbCr in {@link Macroblock#fromARGB}.</li>
 *   <li>16x16 luma (and 8x8 chroma) DC prediction uses the reconstructed bottom
 *       row of the MB above and right column of the MB to the left; missing
 *       neighbors default to 128.</li>
 *   <li>Residual = source - predicted is split into sixteen 4x4 sub-blocks,
 *       each forward-DCT'd.</li>
 *   <li>Luma DC coefficients are collected into a Y2 block and Walsh-Hadamard
 *       transformed; the sixteen AC-only Y blocks are quantized with {@code y1Ac}.</li>
 *   <li>Quantized coefficients are dequantized and inverse-transformed to build
 *       the reconstructed plane used by the next MB's prediction.</li>
 *   <li>Tokens are emitted through {@link VP8TokenEncoder} - Y2 first (if i16x16),
 *       then 16 Y AC blocks in raster order, then 4 U + 4 V blocks.</li>
 * </ol>
 *
 * @see <a href="https://datatracker.ietf.org/doc/html/rfc6386">RFC 6386</a>
 */
final class VP8Encoder {

    private VP8Encoder() { }

    /** Per-frame encoding state threaded through the MB loop. */
    private static final class State {
        final int mbCols, mbRows;
        final int lumaStride, chromaStride;

        // Reconstructed luma/chroma planes (padded to the MB grid).
        final short[] reconY, reconU, reconV;

        // Non-zero-flag context:
        //   topNz[mbX][0..3] luma sub-block columns (bottom row of MB above)
        //   topNz[mbX][4..5] U sub-block columns
        //   topNz[mbX][6..7] V sub-block columns
        //   topNz[mbX][8]    Y2 block (for i16x16 MBs)
        final int[][] topNz;
        // leftNz[0..3]=luma rows, [4..5]=U rows, [6..7]=V rows, [8]=Y2 - reset each MB row.
        final int[] leftNz = new int[9];

        // Quantizer steps.
        final int y1Dc, y1Ac, y2Dc, y2Ac, uvDc, uvAc;

        final BooleanEncoder header;   // first partition: frame header + per-MB modes
        final BooleanEncoder tokens;   // token partition: coefficient streams

        State(int width, int height, int qi) {
            this.mbCols = (width + 15) / 16;
            this.mbRows = (height + 15) / 16;
            this.lumaStride = mbCols * 16;
            this.chromaStride = mbCols * 8;
            this.reconY = new short[lumaStride * mbRows * 16];
            this.reconU = new short[chromaStride * mbRows * 8];
            this.reconV = new short[chromaStride * mbRows * 8];
            this.topNz = new int[mbCols][9];

            this.y1Dc = VP8Tables.DC_Q_LOOKUP[qi];
            this.y1Ac = VP8Tables.AC_Q_LOOKUP[qi];
            this.y2Dc = VP8Tables.DC_Q_LOOKUP[qi] * 2;
            this.y2Ac = VP8Tables.AC2_Q_LOOKUP[qi];
            int uvQi = Math.min(qi, 117);          // spec: uvDc saturates at QI 117
            this.uvDc = VP8Tables.DC_Q_LOOKUP[uvQi];
            this.uvAc = VP8Tables.AC_Q_LOOKUP[qi];

            this.header = new BooleanEncoder(2048);
            this.tokens = new BooleanEncoder(mbCols * mbRows * 1024);
        }
    }

    /**
     * Encodes pixel data into a VP8 keyframe bitstream.
     *
     * @param pixels source pixel buffer
     * @param quality encoding quality in {@code [0.0, 1.0]}
     * @return the encoded VP8 payload bytes
     */
    static byte @NotNull [] encode(@NotNull PixelBuffer pixels, float quality) {
        int width = pixels.width();
        int height = pixels.height();
        int qi = qualityToQi(quality);

        State s = new State(width, height, qi);

        writeFrameHeader(s, qi);

        // Per-MB encoding loop.
        int[] argb = pixels.pixels();
        for (int mbY = 0; mbY < s.mbRows; mbY++) {
            java.util.Arrays.fill(s.leftNz, 0);
            for (int mbX = 0; mbX < s.mbCols; mbX++) {
                encodeMacroblock(s, argb, width, height, mbX, mbY);
            }
        }

        byte[] headerBytes = s.header.toByteArray();
        byte[] tokenBytes = s.tokens.toByteArray();
        return assembleFrame(width, height, headerBytes, tokenBytes);
    }

    /** Maps a 0..1 quality value to VP8's 0..127 QI index (higher quality = lower QI). */
    private static int qualityToQi(float quality) {
        return Math.clamp((int) ((1.0f - quality) * 127f + 0.5f), 0, 127);
    }

    /** Emits the boolean-coded first-partition frame header up to the per-MB modes. */
    private static void writeFrameHeader(@NotNull State s, int qi) {
        BooleanEncoder e = s.header;

        // Frame header.
        e.encodeBool(0);           // color_space
        e.encodeBool(0);           // clamp_type
        e.encodeBool(0);           // use_segment

        // Filter header: disabled.
        e.encodeBool(0);           // simple filter
        e.encodeUint(0, 6);        // loop_filter_level
        e.encodeUint(0, 3);        // sharpness
        e.encodeBool(0);           // use_lf_delta

        e.encodeUint(0, 2);        // log2_num_token_partitions - single token partition

        // Quantizer: write base QI, no per-component deltas (matches State's derived steps).
        e.encodeUint(qi, 7);
        e.encodeBool(0);           // y1_dc_delta_present
        e.encodeBool(0);           // y2_dc_delta_present
        e.encodeBool(0);           // y2_ac_delta_present
        e.encodeBool(0);           // uv_dc_delta_present
        e.encodeBool(0);           // uv_ac_delta_present

        e.encodeBool(0);           // refresh_entropy_probs (ignored for keyframe)

        // VP8ParseProba: 1056 "no update" bits.
        for (int t = 0; t < VP8Tables.NUM_TYPES; t++)
            for (int b = 0; b < VP8Tables.NUM_BANDS; b++)
                for (int c = 0; c < VP8Tables.NUM_CTX; c++)
                    for (int p = 0; p < VP8Tables.NUM_PROBAS; p++)
                        e.encodeBit(VP8Tables.COEFFS_UPDATE_PROBA[t][b][c][p], 0);

        e.encodeBool(0);           // use_skip_proba = 0 - no per-MB skip bit emitted below
    }

    // ──────────────────────────────────────────────────────────────────────
    // Per-macroblock encoding
    // ──────────────────────────────────────────────────────────────────────

    private static void encodeMacroblock(
        @NotNull State s, int @NotNull [] argb, int width, int height, int mbX, int mbY
    ) {
        // 1. Load the MB's YCbCr samples from the source (with boundary replication).
        Macroblock mb = new Macroblock();
        mb.fromARGB(argb, mbX, mbY, width, height);

        // 2. Build DC predictions (luma 16x16, chroma 8x8) from reconstructed neighbors.
        short[] predY = new short[256];
        short[] predU = new short[64];
        short[] predV = new short[64];
        buildDcPrediction16x16(predY, s.reconY, s.lumaStride, mbX, mbY);
        buildDcPrediction8x8(predU, s.reconU, s.chromaStride, mbX, mbY);
        buildDcPrediction8x8(predV, s.reconV, s.chromaStride, mbX, mbY);

        // 3. Forward-DCT each 4x4 Y sub-block. Collect DCs into a Y2 block, quantize ACs.
        short[] y2 = new short[16];          // 16 DC coefficients for Y2 WHT
        short[][] yAc = new short[16][16];   // 16 AC-only Y sub-block coefficients
        forwardY(mb.y, predY, y2, yAc, s.y1Ac);

        // 4. Forward-WHT on the DCs, quantize.
        short[] y2Coef = new short[16];
        DCT.forwardWHT(y2, y2Coef);
        y2Coef[0] = (short) (y2Coef[0] / s.y2Dc);
        for (int i = 1; i < 16; i++) y2Coef[i] = (short) (y2Coef[i] / s.y2Ac);

        // 5. Chroma DCT + quant.
        short[][] uCoef = new short[4][16];
        short[][] vCoef = new short[4][16];
        forwardChroma(mb.cb, predU, uCoef, s.uvDc, s.uvAc);
        forwardChroma(mb.cr, predV, vCoef, s.uvDc, s.uvAc);

        // 6. Emit the MB header in the first partition (i16x16 mode, no skip bit).
        BooleanEncoder h = s.header;
        h.encodeBit(145, 1);       // !is_i4x4  -> 16x16 prediction
        h.encodeBit(156, 0);       // ymode tree: DC_PRED
        h.encodeBit(163, 0);
        h.encodeBit(142, 0);       // uvmode tree: DC_PRED

        // 7. Emit the residual tokens in the token partition, in libwebp's order:
        //    Y2 (type=1, first=0), 16 Y AC blocks (type=0, first=1), 4 U + 4 V (type=2, first=0).
        emitMbTokens(s, mbX, y2Coef, yAc, uCoef, vCoef);

        // 8. Reconstruct the MB's samples for future neighbor prediction.
        reconstructMb(s, mb, predY, predU, predV, y2Coef, yAc, uCoef, vCoef, mbX, mbY);
    }

    /**
     * 16x16 DC prediction: average of the row above + column to the left, or 128
     * when neither is available. Matches {@link IntraPrediction#predict16x16} DC_PRED.
     */
    private static void buildDcPrediction16x16(
        short @NotNull [] dst, short @NotNull [] recon, int stride, int mbX, int mbY
    ) {
        int sum = 0, count = 0;
        if (mbY > 0) {
            int rowOff = (mbY * 16 - 1) * stride + mbX * 16;
            for (int x = 0; x < 16; x++) sum += recon[rowOff + x];
            count += 16;
        }
        if (mbX > 0) {
            int colOff = mbY * 16 * stride + mbX * 16 - 1;
            for (int y = 0; y < 16; y++) sum += recon[colOff + y * stride];
            count += 16;
        }
        short dc = count > 0 ? (short) ((sum + count / 2) / count) : (short) 128;
        java.util.Arrays.fill(dst, dc);
    }

    /** 8x8 DC prediction for chroma. Same formula as 16x16 but over 8-pixel edges. */
    private static void buildDcPrediction8x8(
        short @NotNull [] dst, short @NotNull [] recon, int stride, int mbX, int mbY
    ) {
        int sum = 0, count = 0;
        if (mbY > 0) {
            int rowOff = (mbY * 8 - 1) * stride + mbX * 8;
            for (int x = 0; x < 8; x++) sum += recon[rowOff + x];
            count += 8;
        }
        if (mbX > 0) {
            int colOff = mbY * 8 * stride + mbX * 8 - 1;
            for (int y = 0; y < 8; y++) sum += recon[colOff + y * stride];
            count += 8;
        }
        short dc = count > 0 ? (short) ((sum + count / 2) / count) : (short) 128;
        java.util.Arrays.fill(dst, dc);
    }

    /**
     * For each of the 16 4x4 sub-blocks in the 16x16 luma MB: compute residual,
     * forward-DCT, extract DC into {@code y2}, quantize the AC coefficients.
     */
    private static void forwardY(
        short @NotNull [] src, short @NotNull [] pred,
        short @NotNull [] y2, short @NotNull [] @NotNull [] yAc, int y1Ac
    ) {
        for (int by = 0; by < 4; by++) {
            for (int bx = 0; bx < 4; bx++) {
                short[] residual = new short[16];
                for (int y = 0; y < 4; y++)
                    for (int x = 0; x < 4; x++) {
                        int idx = (by * 4 + y) * 16 + bx * 4 + x;
                        residual[y * 4 + x] = (short) (src[idx] - pred[idx]);
                    }
                short[] coef = new short[16];
                DCT.forwardDCT(residual, coef);
                y2[by * 4 + bx] = coef[0];
                coef[0] = 0; // DC is coded in the Y2 block; AC block has zero DC.
                for (int i = 1; i < 16; i++)
                    coef[i] = (short) (coef[i] / y1Ac);
                yAc[by * 4 + bx] = coef;
            }
        }
    }

    /**
     * Forward-DCT + quantize each of the 4 chroma 4x4 sub-blocks (DC included, no WHT).
     */
    private static void forwardChroma(
        short @NotNull [] src, short @NotNull [] pred,
        short @NotNull [] @NotNull [] out, int uvDc, int uvAc
    ) {
        for (int by = 0; by < 2; by++) {
            for (int bx = 0; bx < 2; bx++) {
                short[] residual = new short[16];
                for (int y = 0; y < 4; y++)
                    for (int x = 0; x < 4; x++) {
                        int idx = (by * 4 + y) * 8 + bx * 4 + x;
                        residual[y * 4 + x] = (short) (src[idx] - pred[idx]);
                    }
                short[] coef = new short[16];
                DCT.forwardDCT(residual, coef);
                coef[0] = (short) (coef[0] / uvDc);
                for (int i = 1; i < 16; i++)
                    coef[i] = (short) (coef[i] / uvAc);
                out[by * 2 + bx] = coef;
            }
        }
    }

    /** Emits all coefficient tokens for one MB in libwebp's raster order. */
    private static void emitMbTokens(
        @NotNull State s, int mbX,
        short @NotNull [] y2, short @NotNull [] @NotNull [] yAc,
        short @NotNull [] @NotNull [] uCoef, short @NotNull [] @NotNull [] vCoef
    ) {
        BooleanEncoder t = s.tokens;
        int[] top = s.topNz[mbX];
        int[] left = s.leftNz;

        // Y2 block (DC coefficients of all 16 Y sub-blocks).
        {
            short[] zz = toZigzag(y2);
            int ctx = top[8] + left[8];
            int nz = VP8TokenEncoder.emit(t, zz, 0, ctx, VP8Tables.TYPE_I16_DC, VP8Tables.COEFFS_PROBA_0);
            top[8] = left[8] = nz;
        }

        // Y AC blocks - 16 blocks in raster order, first=1 (DC was already in Y2).
        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 4; x++) {
                short[] zz = toZigzag(yAc[y * 4 + x]);
                int ctx = top[x] + left[y];
                int nz = VP8TokenEncoder.emit(t, zz, 1, ctx, VP8Tables.TYPE_I16_AC, VP8Tables.COEFFS_PROBA_0);
                top[x] = left[y] = nz;
            }
        }

        // U blocks (indices 4..5 column-wise / 4..5 row-wise in nz tables).
        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 2; x++) {
                short[] zz = toZigzag(uCoef[y * 2 + x]);
                int ctx = top[4 + x] + left[4 + y];
                int nz = VP8TokenEncoder.emit(t, zz, 0, ctx, VP8Tables.TYPE_CHROMA_A, VP8Tables.COEFFS_PROBA_0);
                top[4 + x] = left[4 + y] = nz;
            }
        }

        // V blocks (indices 6..7).
        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 2; x++) {
                short[] zz = toZigzag(vCoef[y * 2 + x]);
                int ctx = top[6 + x] + left[6 + y];
                int nz = VP8TokenEncoder.emit(t, zz, 0, ctx, VP8Tables.TYPE_CHROMA_A, VP8Tables.COEFFS_PROBA_0);
                top[6 + x] = left[6 + y] = nz;
            }
        }
    }

    /** Reorders a 4x4 block from raster order into VP8 zig-zag order. */
    private static short[] toZigzag(short @NotNull [] raster) {
        short[] zz = new short[16];
        for (int i = 0; i < 16; i++) zz[i] = raster[VP8Tables.ZIGZAG[i]];
        return zz;
    }

    /**
     * Inverse-quantize, inverse-transform, add prediction, clamp, and write the
     * result back to the reconstructed Y/U/V planes so the next MB can reference it.
     */
    private static void reconstructMb(
        @NotNull State s, @NotNull Macroblock mb,
        short @NotNull [] predY, short @NotNull [] predU, short @NotNull [] predV,
        short @NotNull [] y2Coef,
        short @NotNull [] @NotNull [] yAc,
        short @NotNull [] @NotNull [] uCoef, short @NotNull [] @NotNull [] vCoef,
        int mbX, int mbY
    ) {
        // Inverse-WHT to get the 16 reconstructed DC values (still quantized).
        short[] dq = new short[16];
        dq[0] = (short) (y2Coef[0] * s.y2Dc);
        for (int i = 1; i < 16; i++) dq[i] = (short) (y2Coef[i] * s.y2Ac);
        short[] dcValues = new short[16];
        DCT.inverseWHT(dq, dcValues);

        // Reconstruct luma: for each 4x4 block, dequant AC, fold in the corresponding DC.
        for (int by = 0; by < 4; by++) {
            for (int bx = 0; bx < 4; bx++) {
                short[] coef = yAc[by * 4 + bx].clone();
                coef[0] = dcValues[by * 4 + bx];       // DC from WHT
                for (int i = 1; i < 16; i++) coef[i] = (short) (coef[i] * s.y1Ac);
                short[] residual = new short[16];
                DCT.inverseDCT(coef, residual);
                for (int y = 0; y < 4; y++)
                    for (int x = 0; x < 4; x++) {
                        int mbIdx = (by * 4 + y) * 16 + bx * 4 + x;
                        int reconIdx = (mbY * 16 + by * 4 + y) * s.lumaStride
                            + mbX * 16 + bx * 4 + x;
                        s.reconY[reconIdx] = (short) clamp(predY[mbIdx] + residual[y * 4 + x]);
                    }
            }
        }

        // Reconstruct chroma similarly.
        reconstructChroma(s.reconU, s.chromaStride, mbX, mbY, predU, uCoef, s.uvDc, s.uvAc);
        reconstructChroma(s.reconV, s.chromaStride, mbX, mbY, predV, vCoef, s.uvDc, s.uvAc);

        // Make the (still populated) Macroblock object feel useful - avoids an unused-var warning.
        mb.yMode = IntraPrediction.DC_PRED;
        mb.uvMode = IntraPrediction.DC_PRED;
    }

    private static void reconstructChroma(
        short @NotNull [] plane, int stride, int mbX, int mbY,
        short @NotNull [] pred, short @NotNull [] @NotNull [] coef, int dcQ, int acQ
    ) {
        for (int by = 0; by < 2; by++) {
            for (int bx = 0; bx < 2; bx++) {
                short[] c = coef[by * 2 + bx].clone();
                c[0] = (short) (c[0] * dcQ);
                for (int i = 1; i < 16; i++) c[i] = (short) (c[i] * acQ);
                short[] residual = new short[16];
                DCT.inverseDCT(c, residual);
                for (int y = 0; y < 4; y++)
                    for (int x = 0; x < 4; x++) {
                        int predIdx = (by * 4 + y) * 8 + bx * 4 + x;
                        int planeIdx = (mbY * 8 + by * 4 + y) * stride + mbX * 8 + bx * 4 + x;
                        plane[planeIdx] = (short) clamp(pred[predIdx] + residual[y * 4 + x]);
                    }
            }
        }
    }

    private static int clamp(int v) {
        return Math.clamp(v, 0, 255);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Frame assembly
    // ──────────────────────────────────────────────────────────────────────

    private static byte @NotNull [] assembleFrame(
        int width, int height, byte @NotNull [] firstPartition, byte @NotNull [] tokenPartition
    ) {
        int firstSize = firstPartition.length;
        byte[] frame = new byte[10 + firstSize + tokenPartition.length];
        int offset = 0;

        int frameTag = (1 << 4) | (firstSize << 5);
        frame[offset++] = (byte) (frameTag & 0xFF);
        frame[offset++] = (byte) ((frameTag >>> 8) & 0xFF);
        frame[offset++] = (byte) ((frameTag >>> 16) & 0xFF);

        frame[offset++] = (byte) 0x9D;
        frame[offset++] = (byte) 0x01;
        frame[offset++] = (byte) 0x2A;

        frame[offset++] = (byte) (width & 0xFF);
        frame[offset++] = (byte) ((width >>> 8) & 0x3F);
        frame[offset++] = (byte) (height & 0xFF);
        frame[offset++] = (byte) ((height >>> 8) & 0x3F);

        System.arraycopy(firstPartition, 0, frame, offset, firstSize);
        offset += firstSize;
        System.arraycopy(tokenPartition, 0, frame, offset, tokenPartition.length);

        return frame;
    }

}
