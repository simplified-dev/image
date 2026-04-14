package dev.simplified.image.codec.webp.lossy;

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
public final class VP8Encoder {

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

        // Sub-block mode context for B_PRED neighbour lookup. Mirrors VP8Decoder.State:
        //   intraT[mbX*4 + bx] = bottom-row sub-block mode of the MB above
        //   intraL[by]         = right-column sub-block mode of the MB to the left
        // For 16x16 macroblocks, both are filled with the analogous B_*_PRED constant.
        final int[] intraT;
        final int[] intraL = new int[4];

        // Quantizer steps.
        final int y1Dc, y1Ac, y2Dc, y2Ac, uvDc, uvAc;

        // Trellis / R-D matrices (per-coefficient q, iq, bias, sharpen).
        final QuantMatrix y1Mtx, y2Mtx, uvMtx;

        // R-D lambdas, derived from QI via libwebp's {@code SetupMatrices} formulas.
        // {@code score = RD_DISTO_MULT * distortion + rate * lambda}.
        final int lambdaI4, lambdaUv;
        final int lambdaTrellisI4, lambdaTrellisI16, lambdaTrellisUv;

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
            this.intraT = new int[mbCols * 4];

            this.y1Dc = VP8Tables.DC_Q_LOOKUP[qi];
            this.y1Ac = VP8Tables.AC_Q_LOOKUP[qi];
            this.y2Dc = VP8Tables.DC_Q_LOOKUP[qi] * 2;
            this.y2Ac = VP8Tables.AC2_Q_LOOKUP[qi];
            int uvQi = Math.min(qi, 117);          // spec: uvDc saturates at QI 117
            this.uvDc = VP8Tables.DC_Q_LOOKUP[uvQi];
            this.uvAc = VP8Tables.AC_Q_LOOKUP[qi];

            this.y1Mtx = QuantMatrix.luma(y1Dc, y1Ac);
            this.y2Mtx = QuantMatrix.lumaY2(y2Dc, y2Ac);
            this.uvMtx = QuantMatrix.chroma(uvDc, uvAc);

            // libwebp's average-quantizer-driven lambdas. We don't have per-segment quant so
            // q_i4 == y1Ac (average of the flat matrix), q_i16 == y2Ac, q_uv == uvAc.
            int qI4 = y1Ac;
            int qUv = uvAc;
            int qI16 = y2Ac;
            this.lambdaI4 = Math.max(1, (3 * qI4 * qI4) >> 7);
            this.lambdaUv = Math.max(1, (3 * qUv * qUv) >> 6);
            this.lambdaTrellisI4 = Math.max(1, (7 * qI4 * qI4) >> 3);
            this.lambdaTrellisI16 = Math.max(1, (qI16 * qI16) >> 2);
            this.lambdaTrellisUv = Math.max(1, (qUv * qUv) << 1);

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
    public static byte @NotNull [] encode(@NotNull PixelBuffer pixels, float quality) {
        int width = pixels.width();
        int height = pixels.height();
        int qi = qualityToQi(quality);

        State s = new State(width, height, qi);

        writeFrameHeader(s, qi);

        // Per-MB encoding loop.
        int[] argb = pixels.pixels();
        for (int mbY = 0; mbY < s.mbRows; mbY++) {
            java.util.Arrays.fill(s.leftNz, 0);
            java.util.Arrays.fill(s.intraL, IntraPrediction.B_DC_PRED);
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

    /**
     * Picks a deblocking filter level from the AC quantizer step, matching libwebp's
     * {@code SetupFilterStrength} ({@code src/enc/quant_enc.c}) at sharpness 0 with the
     * default {@code filter_strength = 60} and no per-segment beta adjustment.
     * <pre>
     *   qstep         = AC_Q_LOOKUP[qi] >> 2
     *   base_strength = qstep   (for sharpness=0, kLevelsFromDelta is the identity)
     *   level         = base_strength * (5 * 60) / 256 = base_strength * 300 / 256
     * </pre>
     * Values below libwebp's {@code FSTRENGTH_CUTOFF=2} are treated as zero.
     *
     * @return filter level in {@code [0, 63]}
     */
    private static int pickFilterLevel(int qi) {
        int qstep = Math.min(VP8Tables.AC_Q_LOOKUP[qi] >> 2, 63);
        int level = qstep * 300 / 256;
        return level < 2 ? 0 : Math.min(level, 63);
    }

    /** Emits the boolean-coded first-partition frame header up to the per-MB modes. */
    private static void writeFrameHeader(@NotNull State s, int qi) {
        BooleanEncoder e = s.header;

        // Frame header.
        e.encodeBool(0);           // color_space
        e.encodeBool(0);           // clamp_type
        e.encodeBool(0);           // use_segment

        // Filter header: simple filter at qi-derived strength.
        int filterLevel = pickFilterLevel(qi);
        e.encodeBool(1);                // simple filter (matches LoopFilter.filterSimple)
        e.encodeUint(filterLevel, 6);   // loop_filter_level
        e.encodeUint(0, 3);             // sharpness
        e.encodeBool(0);                // use_lf_delta

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

        // 2. Extract neighbors for chroma intra prediction (luma neighbors are computed
        //    inside the per-mode evaluation paths below).
        short[] yAbove = extractAbove(s.reconY, s.lumaStride, mbX * 16, mbY * 16, 16);
        short[] yLeft  = extractLeft(s.reconY, s.lumaStride, mbX * 16, mbY * 16, 16);
        short yAboveLeft = (mbX > 0 && mbY > 0)
            ? s.reconY[(mbY * 16 - 1) * s.lumaStride + mbX * 16 - 1] : (short) 128;
        short[] uAbove = extractAbove(s.reconU, s.chromaStride, mbX * 8, mbY * 8, 8);
        short[] uLeft  = extractLeft(s.reconU, s.chromaStride, mbX * 8, mbY * 8, 8);
        short[] vAbove = extractAbove(s.reconV, s.chromaStride, mbX * 8, mbY * 8, 8);
        short[] vLeft  = extractLeft(s.reconV, s.chromaStride, mbX * 8, mbY * 8, 8);
        short uvAboveLeft = (mbX > 0 && mbY > 0)
            ? s.reconU[(mbY * 8 - 1) * s.chromaStride + mbX * 8 - 1] : (short) 128;
        short vuAboveLeft = (mbX > 0 && mbY > 0)
            ? s.reconV[(mbY * 8 - 1) * s.chromaStride + mbX * 8 - 1] : (short) 128;

        // 3. 16x16 luma path - DCT + trellis + reconstruct into a local 256-sample buffer.
        short[] predY16 = new short[256];
        int yMode16 = selectBest16x16Mode(mb.y, yAbove, yLeft, yAboveLeft, predY16);
        I16Result i16 = trellisI16(s, mb.y, predY16, mbX);
        long sse16 = sumSquaredError(mb.y, i16.recon);
        long rdScore16 = sse16 * VP8Costs.RD_DISTO_MULT + (long) i16.rate * s.lambdaTrellisI16;

        // 4. B_PRED luma path - per-sub-block mode + reconstruct into a local 256-sample buffer.
        BPredResult bpred = encodeBPredLuma(s, mb.y, mbX, mbY);

        // 5. Pick the winner by R-D score. The B_PRED path already accumulates per-sub-block
        //    {@code sse*RD_DISTO_MULT + rate*lambda_i4}; we compare that against the i16 total.
        boolean useBPred = bpred.rdScore < rdScore16;

        // 6. Chroma: mode selection + DCT + trellis.
        short[] predU = new short[64];
        short[] predV = new short[64];
        int uvMode = selectBestChromaMode(
            mb.cb, uAbove, uLeft, uvAboveLeft,
            mb.cr, vAbove, vLeft, vuAboveLeft,
            predU, predV
        );

        // 7. Commit luma reconstruction to the frame's reconY plane.
        commitLumaToRecon(s, mbX, mbY, useBPred ? bpred.recon : i16.recon);

        // 8. Emit per-MB header.
        BooleanEncoder h = s.header;
        if (useBPred) {
            h.encodeBit(145, 0);                            // is_i4x4 = true
            emitBPredModes(s, h, mbX, bpred.modes);
        } else {
            h.encodeBit(145, 1);                            // 16x16 prediction
            emitYMode16x16(h, yMode16);
            // Fill sub-block neighbour context with the analogous B_*_PRED for future MBs.
            int ctxMode = mb16ToSubMode(yMode16);
            java.util.Arrays.fill(s.intraT, mbX * 4, mbX * 4 + 4, ctxMode);
            java.util.Arrays.fill(s.intraL, ctxMode);
        }
        emitUvMode(h, uvMode);

        // 9. Emit tokens + quantize+emit chroma (using trellis in emit-time order so each
        //    block sees the correct {@code top_nz + left_nz} context).
        emitLumaTokens(s, mbX, useBPred, i16.y2ZigZag, useBPred ? bpred.yAc : i16.yAcZigZag);
        ChromaResult uResult = trellisAndEmitChroma(s, mbX, mb.cb, predU, /*isU=*/ true);
        ChromaResult vResult = trellisAndEmitChroma(s, mbX, mb.cr, predV, /*isU=*/ false);

        // 10. Commit chroma reconstruction using the dequantized coefficients from trellis.
        commitChromaFromDequant(s.reconU, s.chromaStride, mbX, mbY, predU, uResult.dequant);
        commitChromaFromDequant(s.reconV, s.chromaStride, mbX, mbY, predV, vResult.dequant);
    }

    /** Output of the per-MB 16x16 luma encode: zig-zag coefficients for emit + reconstruction. */
    private static final class I16Result {
        final short[] y2ZigZag;              // Y2 block, zig-zag order (16 entries)
        final short[][] yAcZigZag;           // 16 Y AC blocks, zig-zag order
        final short[] recon;                 // 256-sample reconstructed MB
        final int rate;                      // sum of residual bit costs across Y2 + 16 Y AC blocks

        I16Result(short[] y2ZigZag, short[][] yAcZigZag, short[] recon, int rate) {
            this.y2ZigZag = y2ZigZag;
            this.yAcZigZag = yAcZigZag;
            this.recon = recon;
            this.rate = rate;
        }
    }

    /**
     * DCTs, trellis-quantizes, and reconstructs a 16x16 luma macroblock. Produces
     * zig-zag-order Y2 + 16 Y AC coefficient blocks for token emission, and a 256-sample
     * MB-local reconstruction buffer. Does NOT mutate the frame-level {@link State#topNz}
     * or {@link State#leftNz} - those are updated by the emit pass.
     */
    private static @NotNull I16Result trellisI16(
        @NotNull State s, short @NotNull [] src, short @NotNull [] pred, int mbX
    ) {
        // DCT each 4x4 sub-block; collect DC coefficients for the Y2 WHT.
        short[][] yDctRaster = new short[16][16];
        short[] y2Raw = new short[16];
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
                y2Raw[by * 4 + bx] = coef[0];
                coef[0] = 0;
                yDctRaster[by * 4 + bx] = coef;
            }
        }

        // Local nz context mirrors what emitLumaTokens will see (pre-MB state).
        int[] topNz = { s.topNz[mbX][0], s.topNz[mbX][1], s.topNz[mbX][2], s.topNz[mbX][3] };
        int[] leftNz = { s.leftNz[0], s.leftNz[1], s.leftNz[2], s.leftNz[3] };
        int topNzY2 = s.topNz[mbX][8], leftNzY2 = s.leftNz[8];

        // Y2: WHT + plain quantize (libwebp's {@code ReconstructIntra16} always uses
        // {@code VP8EncQuantizeBlockWHT} for Y2 regardless of the trellis flag).
        short[] y2Dct = new short[16];
        DCT.forwardWHT(y2Raw, y2Dct);
        short[] y2ZigZag = plainQuantZigZag(y2Dct, s.y2Dc, s.y2Ac);
        int rate = VP8Costs.residualCost(topNzY2 + leftNzY2, VP8Tables.TYPE_I16_DC, 0, y2ZigZag);

        // y2Dct has been overwritten with raster-order dequantized values by plainQuantZigZag.
        // Inverse WHT to recover per-sub-block DC values for reconstruction.
        short[] dcValues = new short[16];
        DCT.inverseWHT(y2Dct, dcValues);

        // Y AC: trellis-quantize each sub-block in raster order, accumulate rate.
        short[][] yAcZigZag = new short[16][16];
        for (int by = 0; by < 4; by++) {
            for (int bx = 0; bx < 4; bx++) {
                int ctx = topNz[bx] + leftNz[by];
                short[] dct = yDctRaster[by * 4 + bx]; // DC was already zeroed
                short[] zz = new short[16];
                int nz = TrellisQuantizer.quantize(
                    dct, zz, ctx, VP8Tables.TYPE_I16_AC, s.y1Mtx, s.lambdaTrellisI16);
                yAcZigZag[by * 4 + bx] = zz;
                rate += VP8Costs.residualCost(ctx, VP8Tables.TYPE_I16_AC, 1, zz);
                topNz[bx] = nz;
                leftNz[by] = nz;
            }
        }

        // Reconstruct the MB using predicted samples + dequantized coefficients (Y2 DC re-
        // instated + per-block AC from trellis).
        short[] recon = new short[256];
        for (int by = 0; by < 4; by++) {
            for (int bx = 0; bx < 4; bx++) {
                short[] dequant = yDctRaster[by * 4 + bx]; // trellis filled these in-place
                dequant[0] = dcValues[by * 4 + bx];
                short[] residual = new short[16];
                DCT.inverseDCT(dequant, residual);
                for (int yy = 0; yy < 4; yy++)
                    for (int xx = 0; xx < 4; xx++) {
                        int idx = (by * 4 + yy) * 16 + bx * 4 + xx;
                        recon[idx] = (short) clamp(pred[idx] + residual[yy * 4 + xx]);
                    }
            }
        }
        return new I16Result(y2ZigZag, yAcZigZag, recon, rate);
    }

    /** Output of the per-channel chroma trellis+emit pass. */
    private static final class ChromaResult {
        final short[][] dequant;            // 4 sub-blocks of raster-order dequantized coefficients
        ChromaResult(short[][] dequant) { this.dequant = dequant; }
    }

    /**
     * DCTs, trellis-quantizes, and emits one chroma channel (U or V). The trellis and emit
     * share the same per-sub-block {@code top + left} nz context, which the emit call updates
     * in {@link State#topNz}/{@link State#leftNz}.
     *
     * @param isU {@code true} for U (context slots 4..5), {@code false} for V (slots 6..7)
     */
    private static @NotNull ChromaResult trellisAndEmitChroma(
        @NotNull State s, int mbX, short @NotNull [] src, short @NotNull [] pred, boolean isU
    ) {
        int[] top = s.topNz[mbX];
        int[] left = s.leftNz;
        int baseIdx = isU ? 4 : 6;
        short[][] dequant = new short[4][];

        for (int by = 0; by < 2; by++) {
            for (int bx = 0; bx < 2; bx++) {
                short[] residual = new short[16];
                for (int y = 0; y < 4; y++)
                    for (int x = 0; x < 4; x++) {
                        int idx = (by * 4 + y) * 8 + bx * 4 + x;
                        residual[y * 4 + x] = (short) (src[idx] - pred[idx]);
                    }
                short[] dct = new short[16];
                DCT.forwardDCT(residual, dct);
                int ctx = top[baseIdx + bx] + left[baseIdx + by];
                short[] zz = new short[16];
                TrellisQuantizer.quantize(
                    dct, zz, ctx, VP8Tables.TYPE_CHROMA_A, s.uvMtx, s.lambdaTrellisUv);
                dequant[by * 2 + bx] = dct;

                int nz = VP8TokenEncoder.emit(
                    s.tokens, zz, 0, ctx, VP8Tables.TYPE_CHROMA_A, VP8Tables.COEFFS_PROBA_0);
                top[baseIdx + bx] = nz;
                left[baseIdx + by] = nz;
            }
        }
        return new ChromaResult(dequant);
    }

    /** Output of the per-MB B_PRED evaluation path. */
    private static final class BPredResult {
        final int[] modes;          // 16 sub-block modes, raster order (by * 4 + bx)
        final short[][] yAc;        // 16 quantized coefficient blocks, <b>zig-zag order</b>
        final short[] recon;        // 256 reconstructed luma samples, MB-local raster
        final long sse;             // sum-of-squared-error vs source
        final long rdScore;         // {@code RD_DISTO_MULT * sse + rate * lambda} summed over sub-blocks

        BPredResult(int[] modes, short[][] yAc, short[] recon, long sse, long rdScore) {
            this.modes = modes;
            this.yAc = yAc;
            this.recon = recon;
            this.sse = sse;
            this.rdScore = rdScore;
        }
    }

    /**
     * Evaluates the B_PRED path: for each of 16 sub-blocks in raster order, picks the best
     * of the 10 4x4 prediction modes by SSE, forward-DCTs + quantizes the residual, and
     * reconstructs into a local 256-sample buffer that subsequent sub-blocks use as
     * intra neighbours. Does NOT mutate {@code s.reconY}.
     */
    private static @NotNull BPredResult encodeBPredLuma(@NotNull State s, short @NotNull [] src, int mbX, int mbY) {
        int[] modes = new int[16];
        short[][] yAc = new short[16][];
        short[] recon = new short[256];
        long totalSse = 0;
        long totalRDScore = 0;

        // Sub-block mode neighbour context tracked locally (so we don't update s.intraT/L
        // until after we know B_PRED has won).
        int[] localTop = new int[4];
        for (int bx = 0; bx < 4; bx++) localTop[bx] = s.intraT[mbX * 4 + bx];
        int[] localLeft = s.intraL.clone();

        // Non-zero sub-block contexts tracked locally so token-cost scoring sees the same
        // {@code top_nz + left_nz} values that libwebp's {@code PickBestIntra4} passes into
        // its {@code VP8Residual}. Mirror layout of {@link State#topNz} (rows 0..3 = luma).
        int[] localTopNz = new int[4];
        for (int bx = 0; bx < 4; bx++) localTopNz[bx] = s.topNz[mbX][bx];
        int[] localLeftNz = { s.leftNz[0], s.leftNz[1], s.leftNz[2], s.leftNz[3] };

        for (int by = 0; by < 4; by++) {
            for (int bx = 0; bx < 4; bx++) {
                short[] above = neighborAbove8(s, recon, mbX, mbY, bx, by);
                short[] left = neighborLeft4(s, recon, mbX, mbY, bx, by);
                short aboveLeft = neighborAboveLeft(s, recon, mbX, mbY, bx, by);

                short[] srcBlock = new short[16];
                for (int yy = 0; yy < 4; yy++)
                    for (int xx = 0; xx < 4; xx++)
                        srcBlock[yy * 4 + xx] = src[(by * 4 + yy) * 16 + bx * 4 + xx];

                // Mode-context for the i4 mode-cost table.
                int aboveMode = by > 0 ? modes[(by - 1) * 4 + bx] : localTop[bx];
                int leftMode = bx > 0 ? modes[by * 4 + bx - 1] : localLeft[by];
                int[] modeCosts = VP8Costs.FIXED_COSTS_I4[aboveMode][leftMode];
                int ctxNz = localTopNz[bx] + localLeftNz[by];

                short[] bestRecon = new short[16];
                short[] bestCoef = new short[16];    // zig-zag order (ready for emit)
                int bestMode = -1;
                int bestNz = 0;
                long bestScore = Long.MAX_VALUE;
                long bestSseValue = Long.MAX_VALUE;
                short[] cand = new short[16];
                short[] candResidual = new short[16];
                short[] candDeq = new short[16];
                short[] candZigZag = new short[16];
                short[] candReconBlk = new short[16];

                for (int mode = 0; mode < IntraPrediction.NUM_4x4_MODES; mode++) {
                    IntraPrediction.predict4x4(cand, above, left, aboveLeft, mode);
                    for (int i = 0; i < 16; i++)
                        candResidual[i] = (short) (srcBlock[i] - cand[i]);
                    DCT.forwardDCT(candResidual, candDeq);
                    // Trellis-quantize (candDeq becomes raster-dequantized, candZigZag is
                    // zig-zag-order quantized levels).
                    java.util.Arrays.fill(candZigZag, (short) 0);
                    int nz = TrellisQuantizer.quantize(
                        candDeq, candZigZag, ctxNz, VP8Tables.TYPE_I4_AC, s.y1Mtx, s.lambdaTrellisI4);

                    DCT.inverseDCT(candDeq, candResidual);
                    for (int i = 0; i < 16; i++)
                        candReconBlk[i] = (short) Math.clamp(cand[i] + candResidual[i], 0, 255);

                    long sse = sumSquaredError(srcBlock, candReconBlk);
                    // R-D: lambda_i4 * mode-tree fixed cost, added to distortion. Residual
                    // cost is intentionally omitted: including it at small quantizers causes
                    // the encoder to prefer simpler modes with larger quantization error.
                    long score = sse * VP8Costs.RD_DISTO_MULT
                        + (long) modeCosts[mode] * s.lambdaI4;
                    if (score < bestScore) {
                        bestScore = score;
                        bestSseValue = sse;
                        bestMode = mode;
                        bestNz = nz;
                        System.arraycopy(candReconBlk, 0, bestRecon, 0, 16);
                        System.arraycopy(candZigZag, 0, bestCoef, 0, 16);
                    }
                }

                modes[by * 4 + bx] = bestMode;
                yAc[by * 4 + bx] = bestCoef;
                totalSse += bestSseValue;
                totalRDScore += bestScore;

                // Write reconstructed sub-block into the local 16x16 buffer.
                for (int yy = 0; yy < 4; yy++)
                    for (int xx = 0; xx < 4; xx++)
                        recon[(by * 4 + yy) * 16 + bx * 4 + xx] = bestRecon[yy * 4 + xx];

                localTop[bx] = bestMode;     // for next row's above-neighbour context
                localLeft[by] = bestMode;    // for next column's left-neighbour context
                localTopNz[bx] = bestNz;
                localLeftNz[by] = bestNz;
            }
        }

        return new BPredResult(modes, yAc, recon, totalSse, totalRDScore);
    }

    /** Maps a 16x16 luma macroblock mode to the analogous B_PRED sub-block constant. */
    private static int mb16ToSubMode(int mbMode) {
        return switch (mbMode) {
            case IntraPrediction.V_PRED -> IntraPrediction.B_VE_PRED;
            case IntraPrediction.H_PRED -> IntraPrediction.B_HE_PRED;
            case IntraPrediction.TM_PRED -> IntraPrediction.B_TM_PRED;
            default -> IntraPrediction.B_DC_PRED;
        };
    }

    /**
     * Above-row neighbours for sub-block ({@code by}, {@code bx}): 4 above + 4 above-right,
     * sourced from {@code mbRecon} for sub-blocks within the current MB and from
     * {@code s.reconY} for the prior MB row. Above-right past the current MB's right edge
     * is replicated from the rightmost in-MB pixel (matching {@link VP8Decoder}).
     */
    private static short[] neighborAbove8(@NotNull State s, short @NotNull [] mbRecon, int mbX, int mbY, int bx, int by) {
        if (by == 0 && mbY == 0) return null;
        short[] above = new short[8];
        for (int i = 0; i < 8; i++) {
            int relX = bx * 4 + i;
            if (by == 0) {
                int absX = mbX * 16 + relX;
                int rightLimit = s.lumaStride - 1;
                int x = Math.min(absX, rightLimit);
                above[i] = s.reconY[(mbY * 16 - 1) * s.lumaStride + x];
            } else {
                int rightLimit = 15;
                int x = Math.min(relX, rightLimit);
                above[i] = mbRecon[(by * 4 - 1) * 16 + x];
            }
        }
        return above;
    }

    /** Left-column 4 neighbours for sub-block ({@code by}, {@code bx}). */
    private static short[] neighborLeft4(@NotNull State s, short @NotNull [] mbRecon, int mbX, int mbY, int bx, int by) {
        if (bx == 0 && mbX == 0) return null;
        short[] left = new short[4];
        for (int i = 0; i < 4; i++) {
            int relY = by * 4 + i;
            if (bx == 0) {
                left[i] = s.reconY[(mbY * 16 + relY) * s.lumaStride + mbX * 16 - 1];
            } else {
                left[i] = mbRecon[relY * 16 + bx * 4 - 1];
            }
        }
        return left;
    }

    private static short neighborAboveLeft(@NotNull State s, short @NotNull [] mbRecon, int mbX, int mbY, int bx, int by) {
        int absX = mbX * 16 + bx * 4 - 1;
        int absY = mbY * 16 + by * 4 - 1;
        if (absX < 0 || absY < 0) return 128;
        if (by > 0 && bx > 0)
            return mbRecon[(by * 4 - 1) * 16 + bx * 4 - 1];
        if (by == 0 && bx > 0)
            return s.reconY[(mbY * 16 - 1) * s.lumaStride + mbX * 16 + bx * 4 - 1];
        if (bx == 0 && by > 0)
            return s.reconY[(mbY * 16 + by * 4 - 1) * s.lumaStride + mbX * 16 - 1];
        return s.reconY[(mbY * 16 - 1) * s.lumaStride + mbX * 16 - 1];
    }

    /**
     * Plain (non-trellis) quantization of a single 4x4 DCT block. Overwrites {@code in[]} with
     * raster-order dequantized values and returns the zig-zag-ordered quantized levels.
     * Used for the Y2 (WHT) block, which libwebp never trellises.
     */
    private static short @NotNull [] plainQuantZigZag(short @NotNull [] in, int dcQ, int acQ) {
        short[] zz = new short[16];
        for (int n = 0; n < 16; n++) {
            int j = VP8Tables.ZIGZAG[n];
            int q = (j == 0) ? dcQ : acQ;
            int level = in[j] / q;
            zz[n] = (short) level;
            in[j] = (short) (level * q);
        }
        return zz;
    }

    /** Copies a 16x16 MB-local reconstruction buffer to the appropriate region of {@code s.reconY}. */
    private static void commitLumaToRecon(@NotNull State s, int mbX, int mbY, short @NotNull [] recon) {
        for (int yy = 0; yy < 16; yy++) {
            int dst = (mbY * 16 + yy) * s.lumaStride + mbX * 16;
            System.arraycopy(recon, yy * 16, s.reconY, dst, 16);
        }
    }

    /**
     * Reconstructs a chroma channel from its prediction plus per-sub-block raster-order
     * dequantized coefficients (as produced by {@link #trellisAndEmitChroma}), and commits
     * the result to the appropriate region of {@code plane}.
     */
    private static void commitChromaFromDequant(
        short @NotNull [] plane, int stride, int mbX, int mbY,
        short @NotNull [] pred, short @NotNull [] @NotNull [] dequant
    ) {
        for (int by = 0; by < 2; by++) {
            for (int bx = 0; bx < 2; bx++) {
                short[] residual = new short[16];
                DCT.inverseDCT(dequant[by * 2 + bx], residual);
                for (int y = 0; y < 4; y++)
                    for (int x = 0; x < 4; x++) {
                        int predIdx = (by * 4 + y) * 8 + bx * 4 + x;
                        int planeIdx = (mbY * 8 + by * 4 + y) * stride + mbX * 8 + bx * 4 + x;
                        plane[planeIdx] = (short) clamp(pred[predIdx] + residual[y * 4 + x]);
                    }
            }
        }
    }

    /**
     * Emits the luma coefficient tokens for one MB in libwebp's order: Y2 first
     * (if {@code !isBPred}), then 16 Y sub-blocks in raster order. Coefficient blocks are
     * already in zig-zag order - trellis writes directly to zig-zag. Updates the
     * {@link State#topNz}/{@link State#leftNz} luma slots.
     */
    private static void emitLumaTokens(
        @NotNull State s, int mbX, boolean isBPred,
        short @NotNull [] y2ZigZag, short @NotNull [] @NotNull [] yAcZigZag
    ) {
        BooleanEncoder t = s.tokens;
        int[] top = s.topNz[mbX];
        int[] left = s.leftNz;

        int yFirst;
        int yType;
        if (isBPred) {
            yFirst = 0;
            yType = VP8Tables.TYPE_I4_AC;
            // B_PRED MBs have no Y2 block - per libwebp ParseResiduals, neither encoder
            // nor decoder touches the Y2 nz context bits (top[8]/left[8]) here.
        } else {
            int ctx = top[8] + left[8];
            int nz = VP8TokenEncoder.emit(
                t, y2ZigZag, 0, ctx, VP8Tables.TYPE_I16_DC, VP8Tables.COEFFS_PROBA_0);
            top[8] = left[8] = nz;
            yFirst = 1;
            yType = VP8Tables.TYPE_I16_AC;
        }

        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 4; x++) {
                int ctx = top[x] + left[y];
                int nz = VP8TokenEncoder.emit(
                    t, yAcZigZag[y * 4 + x], yFirst, ctx, yType, VP8Tables.COEFFS_PROBA_0);
                top[x] = left[y] = nz;
            }
        }
    }

    /**
     * Emits the 16 sub-block modes via the {@link VP8Tables#BMODE_TREE} tree at the
     * context-dependent {@link VP8Tables#KF_BMODE_PROB} probabilities. Updates
     * {@code s.intraT}/{@code s.intraL} so subsequent MBs see the correct neighbour modes.
     */
    private static void emitBPredModes(@NotNull State s, @NotNull BooleanEncoder h, int mbX, int @NotNull [] modes) {
        int[][] subModes = new int[4][4];
        for (int by = 0; by < 4; by++)
            for (int bx = 0; bx < 4; bx++)
                subModes[by][bx] = modes[by * 4 + bx];

        for (int by = 0; by < 4; by++) {
            for (int bx = 0; bx < 4; bx++) {
                int aboveMode = by > 0 ? subModes[by - 1][bx] : s.intraT[mbX * 4 + bx];
                int leftMode = bx > 0 ? subModes[by][bx - 1] : s.intraL[by];
                int[] probs = VP8Tables.KF_BMODE_PROB[aboveMode][leftMode];
                emitBMode(h, subModes[by][bx], probs);
            }
        }

        // Update neighbour-mode context for future MBs.
        for (int bx = 0; bx < 4; bx++) s.intraT[mbX * 4 + bx] = subModes[3][bx];
        for (int by = 0; by < 4; by++) s.intraL[by] = subModes[by][3];
    }

    /**
     * Walks {@link VP8Tables#BMODE_TREE} to the {@code mode} leaf, emitting one bit per
     * internal node at the corresponding probability. Inverse of
     * {@link BooleanDecoder#decodeTree}.
     */
    private static void emitBMode(@NotNull BooleanEncoder h, int mode, int @NotNull [] probs) {
        int[] tree = VP8Tables.BMODE_TREE;
        int node = 0;
        // Trees can have at most ~9 internal-node hops for the bmode tree.
        int[] pathBranches = new int[9];
        int[] pathNodes = new int[9];
        int depth = findTreePath(tree, 0, -mode, pathBranches, pathNodes, 0);
        for (int i = 0; i < depth; i++)
            h.encodeBit(probs[pathNodes[i] >> 1], pathBranches[i]);
    }

    /** Depth-first walks {@code tree} looking for {@code targetLeaf}, recording the branch
     *  bits taken to reach it. Returns the depth on success, 0 on failure. */
    private static int findTreePath(int[] tree, int node, int targetLeaf,
                                    int[] branches, int[] nodes, int depth) {
        for (int branch = 0; branch < 2; branch++) {
            int next = tree[node + branch];
            branches[depth] = branch;
            nodes[depth] = node;
            if (next <= 0) {
                if (next == targetLeaf) return depth + 1;
            } else {
                int d = findTreePath(tree, next, targetLeaf, branches, nodes, depth + 1);
                if (d > 0) return d;
            }
        }
        return 0;
    }

    /**
     * Picks the 16x16 luma mode that minimises {@code RD_DISTO_MULT * sse + modeCost * lambda_d}
     * against the source samples, where {@code modeCost = VP8Costs.FIXED_COSTS_I16[mode]}
     * and {@code lambda_d = 106} (libwebp's {@code lambda_d_i16} constant from
     * {@code RefineUsingDistortion} in {@code src/enc/quant_enc.c}). Fills {@code predOut}
     * with the winning prediction.
     *
     * @return the selected {@link IntraPrediction} mode constant
     */
    private static int selectBest16x16Mode(
        short @NotNull [] src, short[] above, short[] left, short aboveLeft,
        short @NotNull [] predOut
    ) {
        int[] modes = { IntraPrediction.DC_PRED, IntraPrediction.V_PRED,
                        IntraPrediction.H_PRED, IntraPrediction.TM_PRED };
        int bestMode = IntraPrediction.DC_PRED;
        long bestScore = Long.MAX_VALUE;
        short[] candidate = new short[256];
        for (int mode : modes) {
            IntraPrediction.predict16x16(candidate, above, left, aboveLeft, mode);
            long sse = sumSquaredError(src, candidate);
            long score = sse * VP8Costs.RD_DISTO_MULT + (long) VP8Costs.FIXED_COSTS_I16[mode] * 106;
            if (score < bestScore) {
                bestScore = score;
                bestMode = mode;
                System.arraycopy(candidate, 0, predOut, 0, 256);
            }
        }
        return bestMode;
    }

    /**
     * Picks the shared chroma 8x8 mode that minimises
     * {@code RD_DISTO_MULT * (sseU + sseV) + VP8Costs.FIXED_COSTS_UV[mode] * 120}, where
     * {@code 120} is libwebp's {@code lambda_d_uv} ({@code src/enc/quant_enc.c}). Fills
     * {@code predU}/{@code predV} with the winning predictions.
     */
    private static int selectBestChromaMode(
        short @NotNull [] srcU, short[] aboveU, short[] leftU, short aboveLeftU,
        short @NotNull [] srcV, short[] aboveV, short[] leftV, short aboveLeftV,
        short @NotNull [] predU, short @NotNull [] predV
    ) {
        int[] modes = { IntraPrediction.DC_PRED, IntraPrediction.V_PRED,
                        IntraPrediction.H_PRED, IntraPrediction.TM_PRED };
        int bestMode = IntraPrediction.DC_PRED;
        long bestScore = Long.MAX_VALUE;
        short[] candU = new short[64];
        short[] candV = new short[64];
        for (int mode : modes) {
            IntraPrediction.predict8x8(candU, aboveU, leftU, aboveLeftU, mode);
            IntraPrediction.predict8x8(candV, aboveV, leftV, aboveLeftV, mode);
            long sse = sumSquaredError(srcU, candU) + sumSquaredError(srcV, candV);
            long score = sse * VP8Costs.RD_DISTO_MULT + (long) VP8Costs.FIXED_COSTS_UV[mode] * 120;
            if (score < bestScore) {
                bestScore = score;
                bestMode = mode;
                System.arraycopy(candU, 0, predU, 0, 64);
                System.arraycopy(candV, 0, predV, 0, 64);
            }
        }
        return bestMode;
    }

    private static long sumSquaredError(short @NotNull [] a, short @NotNull [] b) {
        long sum = 0;
        for (int i = 0; i < a.length; i++) {
            int d = a[i] - b[i];
            sum += (long) d * d;
        }
        return sum;
    }

    /**
     * Emits the 4-way 16x16 luma mode tree, matching libwebp's
     * {@code PutI16Mode} (src/enc/tree_enc.c). Mode bits:
     * <pre>
     *   DC_PRED -> (0, 0) at probs (156, 163)
     *   V_PRED  -> (0, 1) at probs (156, 163)
     *   H_PRED  -> (1, 0) at probs (156, 128)
     *   TM_PRED -> (1, 1) at probs (156, 128)
     * </pre>
     */
    private static void emitYMode16x16(@NotNull BooleanEncoder e, int mode) {
        boolean tmOrH = (mode == IntraPrediction.TM_PRED || mode == IntraPrediction.H_PRED);
        e.encodeBit(156, tmOrH ? 1 : 0);
        if (tmOrH)
            e.encodeBit(128, mode == IntraPrediction.TM_PRED ? 1 : 0);
        else
            e.encodeBit(163, mode == IntraPrediction.V_PRED ? 1 : 0);
    }

    /**
     * Emits the 4-way chroma mode tree, matching libwebp's {@code PutUVMode}.
     * <pre>
     *   DC_PRED -> (0)       at prob 142
     *   V_PRED  -> (1, 0)    at probs (142, 114)
     *   H_PRED  -> (1, 1, 0) at probs (142, 114, 183)
     *   TM_PRED -> (1, 1, 1) at probs (142, 114, 183)
     * </pre>
     */
    private static void emitUvMode(@NotNull BooleanEncoder e, int mode) {
        if (mode == IntraPrediction.DC_PRED) {
            e.encodeBit(142, 0);
            return;
        }
        e.encodeBit(142, 1);
        if (mode == IntraPrediction.V_PRED) {
            e.encodeBit(114, 0);
            return;
        }
        e.encodeBit(114, 1);
        e.encodeBit(183, mode == IntraPrediction.TM_PRED ? 1 : 0);
    }

    /**
     * Extracts the bottom row of the MB above (length {@code n}, or {@code null}
     * when this is the top MB row).
     */
    private static short[] extractAbove(short @NotNull [] plane, int stride, int x0, int y0, int n) {
        if (y0 == 0) return null;
        short[] row = new short[n];
        int off = (y0 - 1) * stride + x0;
        System.arraycopy(plane, off, row, 0, n);
        return row;
    }

    /**
     * Extracts the right column of the MB to the left (length {@code n}, or
     * {@code null} when this is the first MB in a row).
     */
    private static short[] extractLeft(short @NotNull [] plane, int stride, int x0, int y0, int n) {
        if (x0 == 0) return null;
        short[] col = new short[n];
        for (int i = 0; i < n; i++) col[i] = plane[(y0 + i) * stride + x0 - 1];
        return col;
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
