package dev.simplified.image.codec.webp.lossy;

import dev.simplified.image.pixel.PixelBuffer;
import dev.simplified.image.exception.ImageDecodeException;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

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
        /** {@code true} for a keyframe, {@code false} for a P-frame (inter frame). */
        final boolean isKeyframe;
        /** Backing session for P-frame reference buffers, or {@code null} for keyframe-only decode. */
        @Nullable final VP8DecoderSession session;

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

        // Per-MB inter flag (row-major). Fed to the MV-ref probability derivation to
        // mirror libvpx's {@code FindNearMVs} for P-frame parse.
        final boolean[] mbIsInter;

        // Per-MB MV in libvpx 1/8-pel internal units (valid only when mbIsInter is true).
        // Populated on inter-skip, NEAREST, NEAR, and NEW decode paths so downstream MBs'
        // NearMvs.find() produces the same nearest/near/best MVs that the encoder used.
        final int[] mbMvRow;
        final int[] mbMvCol;

        // Per-MB loop-filter classification tables (row-major mbY*mbCols + mbX):
        //   mbRefFrame[i]   = LoopFilter.REF_INTRA / REF_LAST / REF_GOLDEN / REF_ALTREF
        //   mbModeLfIdx[i]  = LoopFilter.MODE_{NON_BPRED_INTRA, BPRED, ZEROMV, OTHER_INTER, SPLITMV}
        // Consumed by the end-of-frame loop filter to apply RFC 6386 section 15 deltas.
        final int[] mbRefFrame;
        final int[] mbModeLfIdx;

        // Filter-delta header state populated by the filter-header parse:
        //   useLfDelta    - the use_lf_delta bit
        //   refLfDelta    - 4 ref_lf_delta entries (LAST, GOLDEN, ALTREF, INTRA in RFC order,
        //                   but we store indexed by LoopFilter.REF_* which puts INTRA at 0)
        //   modeLfDelta   - 4 mode_lf_delta entries (B_PRED, ZEROMV, NEWMV, SPLITMV)
        boolean useLfDelta;
        final int[] refLfDelta = new int[4];
        final int[] modeLfDelta = new int[4];

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

        // Inter-frame header state (populated only when !isKeyframe).
        int probIntra;
        int probLast;
        int probGf;
        final int[] yModeProba = VP8Tables.YMODE_PROBA_INTER.clone();
        final int[] uvModeProba = VP8Tables.UV_MODE_PROBA_INTER.clone();

        State(int width, int height, boolean isKeyframe, @Nullable VP8DecoderSession session) {
            this.isKeyframe = isKeyframe;
            this.session = session;
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
            this.mbIsInter = new boolean[mbCols * mbRows];
            this.mbMvRow = new int[mbCols * mbRows];
            this.mbMvCol = new int[mbCols * mbRows];
            this.mbRefFrame = new int[mbCols * mbRows];       // defaults to REF_INTRA = 0
            this.mbModeLfIdx = new int[mbCols * mbRows];       // defaults to MODE_NON_BPRED_INTRA = 0
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
        return decodeFrame(data, null);
    }

    /**
     * Decodes a VP8 frame (keyframe or P-frame), optionally snapshotting reconstructed
     * planes into {@code session} for later P-frame reference use. Package-private entry
     * point used by {@link VP8DecoderSession#decode(byte[])}.
     * <p>
     * Keyframe parse follows RFC 6386 section 9; P-frame parse requires {@code session}
     * to hold a matching reference (dimensions from the most recent keyframe) since
     * inter frames omit the sync code and dimensions fields.
     *
     * @param data the raw VP8 payload
     * @param session session holding P-frame reference + reference dimensions, or
     *                {@code null} for a stateless keyframe-only decode
     * @return the decoded pixel buffer
     * @throws ImageDecodeException if the bitstream is malformed or uses unsupported features
     */
    static @NotNull PixelBuffer decodeFrame(byte @NotNull [] data, @Nullable VP8DecoderSession session) {
        if (data.length < 3)
            throw new ImageDecodeException("VP8 data too short");

        int frameTag = (data[0] & 0xFF) | ((data[1] & 0xFF) << 8) | ((data[2] & 0xFF) << 16);
        boolean keyFrame = (frameTag & 0x01) == 0;
        int firstPartSize = (frameTag >>> 5) & 0x7FFFF;

        int width;
        int height;
        int headerStart;

        if (keyFrame) {
            if (data.length < 10)
                throw new ImageDecodeException("VP8 keyframe data too short");
            if (data[3] != (byte) 0x9D || data[4] != (byte) 0x01 || data[5] != (byte) 0x2A)
                throw new ImageDecodeException("Invalid VP8 sync code");
            width = ((data[6] & 0xFF) | ((data[7] & 0xFF) << 8)) & 0x3FFF;
            height = ((data[8] & 0xFF) | ((data[9] & 0xFF) << 8)) & 0x3FFF;
            if (width == 0 || height == 0)
                throw new ImageDecodeException("Invalid VP8 dimensions: %dx%d", width, height);
            headerStart = 10;
        } else {
            if (session == null || !session.hasReference())
                throw new ImageDecodeException("VP8 inter-frame requires a session with a reference");
            width = session.refWidth;
            height = session.refHeight;
            headerStart = 3;
        }

        int partitionEnd = Math.min(headerStart + firstPartSize, data.length);
        BooleanDecoder br = new BooleanDecoder(data, headerStart, partitionEnd - headerStart);

        State s = new State(width, height, keyFrame, session);

        // Keyframe-only fields (color_space + clamp_type) per RFC 6386 section 9.2.
        if (keyFrame) {
            br.decodeBool();   // colorspace
            br.decodeBool();   // clamp_type
        }

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
        s.useLfDelta = br.decodeBool() != 0;
        if (s.useLfDelta) {
            if (br.decodeBool() != 0) {                       // update_lf_delta
                // RFC 6386 section 15: ref_lf_delta and mode_lf_delta are each 4 signed 6-bit
                // values with per-entry update flags. The wire order indexes by reference
                // frame (INTRA, LAST, GOLDEN, ALTREF) and by mode class (B_PRED, ZEROMV,
                // NEWMV/NEAREST/NEAR, SPLITMV).
                for (int i = 0; i < 4; i++)
                    if (br.decodeBool() != 0) s.refLfDelta[i] = br.decodeSint(6);
                for (int i = 0; i < 4; i++)
                    if (br.decodeBool() != 0) s.modeLfDelta[i] = br.decodeSint(6);
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

        // ── Inter-frame reference-buffer management (RFC 6386 section 9.7) ──
        if (!keyFrame) {
            boolean refreshGolden = br.decodeBool() != 0;
            boolean refreshAlt = br.decodeBool() != 0;
            if (!refreshGolden) br.decodeUint(2);    // copy_buffer_to_golden
            if (!refreshAlt) br.decodeUint(2);       // copy_buffer_to_alt
            br.decodeBool();                         // sign_bias_golden
            br.decodeBool();                         // sign_bias_alt
        }

        // refresh_entropy_probs (value ignored on keyframes, but the bit is still consumed).
        br.decodeBool();

        // Inter-only: refresh_last.
        if (!keyFrame)
            br.decodeBool();                         // refresh_last

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

        // ── Inter-frame mode / MV probabilities (RFC 6386 sections 16.2, 17.2) ──
        if (!keyFrame) {
            s.probIntra = br.decodeUint(8);
            s.probLast = br.decodeUint(8);
            s.probGf = br.decodeUint(8);
            // Y-mode probabilities (4 entries) - optional per-entry 8-bit update.
            if (br.decodeBool() != 0) {
                for (int i = 0; i < 4; i++)
                    s.yModeProba[i] = br.decodeUint(8);
            }
            // UV-mode probabilities (3 entries) - optional per-entry 8-bit update.
            if (br.decodeBool() != 0) {
                for (int i = 0; i < 3; i++)
                    s.uvModeProba[i] = br.decodeUint(8);
            }
            // MV probabilities - each of 2*19 slots has an update flag at MV_UPDATE_PROBA.
            for (int c = 0; c < 2; c++)
                for (int p = 0; p < VP8Tables.NUM_MV_PROBAS; p++)
                    if (br.decodeBit(VP8Tables.MV_UPDATE_PROBA[c][p]) != 0)
                        br.decodeUint(7);            // new MV prob (Phase 1: never actually updated)
        }

        // ── Token partition decoders ──
        // The layout following the first partition is:
        //   (numParts - 1) * 3 bytes of little-endian uint24 size prefixes for partitions 0..N-2;
        //   concatenated partition bodies, with the last partition's size implied by file length.
        int partSizesOffset = partitionEnd;
        int partBodiesOffset = partSizesOffset + (numTokenPartitions - 1) * 3;
        if (partBodiesOffset > data.length)
            throw new ImageDecodeException("VP8 token partition size prefixes truncated");
        BooleanDecoder[] tokenDecoders = new BooleanDecoder[numTokenPartitions];
        int cursor = partBodiesOffset;
        for (int p = 0; p < numTokenPartitions; p++) {
            int partSize;
            if (p < numTokenPartitions - 1) {
                int o = partSizesOffset + p * 3;
                partSize = (data[o] & 0xFF)
                    | ((data[o + 1] & 0xFF) << 8)
                    | ((data[o + 2] & 0xFF) << 16);
            } else {
                partSize = data.length - cursor;
            }
            if (partSize < 0 || cursor + partSize > data.length)
                throw new ImageDecodeException(
                    "VP8 token partition %d out of range: offset=%d size=%d total=%d",
                    p, cursor, partSize, data.length);
            tokenDecoders[p] = new BooleanDecoder(data, cursor, partSize);
            cursor += partSize;
        }
        int partMask = numTokenPartitions - 1;

        // ── Macroblock loop ──
        // MB row y reads tokens from partition (y & (numTokenPartitions - 1)), matching the
        // encoder's interleave. Mode bits come from the first partition (br) as before.
        for (int mbY = 0; mbY < s.mbRows; mbY++) {
            java.util.Arrays.fill(s.leftNz, 0);
            java.util.Arrays.fill(s.intraL, IntraPrediction.B_DC_PRED);

            BooleanDecoder td = tokenDecoders[mbY & partMask];
            for (int mbX = 0; mbX < s.mbCols; mbX++) {
                if (keyFrame)
                    decodeMacroblock(s, br, td, mbX, mbY);
                else
                    decodeInterMacroblock(s, br, td, mbX, mbY);
            }
        }

        // Loop filter. Normal + simple modes are both driven by the same per-MB
        // classification tables; ref_lf_delta / mode_lf_delta are applied when the
        // frame header set use_lf_delta = 1.
        if (filterLevel > 0) {
            LoopFilter.filterFrame(
                s.reconY, s.reconU, s.reconV,
                s.lumaStride, s.chromaStride,
                s.mbCols, s.mbRows,
                simpleFilter, filterLevel, sharpness,
                s.mbRefFrame, s.mbModeLfIdx,
                s.useLfDelta, s.refLfDelta, s.modeLfDelta
            );
        }

        // Snapshot the post-filter planes into the session so Phase 1+ P-frame
        // decoding can read them as the LAST reference.
        if (session != null) {
            session.captureReference(
                s.reconY, s.reconU, s.reconV,
                s.lumaStride, s.chromaStride,
                s.mbCols, s.mbRows, width, height
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
        int mbIdxKf = mbY * s.mbCols + mbX;
        s.mbIsI4x4[mbIdxKf] = isI4x4;
        s.mbRefFrame[mbIdxKf] = LoopFilter.REF_INTRA;
        s.mbModeLfIdx[mbIdxKf] = isI4x4 ? LoopFilter.MODE_BPRED : LoopFilter.MODE_NON_BPRED_INTRA;

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

    // ──────────────────────────────────────────────────────────────────────
    // Per-macroblock parse + reconstruct (inter / P-frame path)
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Parses and reconstructs one macroblock in a P-frame. Supports the two Phase 1
     * cases: inter-skip {@code (is_inter=1, ref=LAST, mb_mode=ZEROMV, skip=1)} which
     * copies the reference MB into the current reconstruction, and intra-in-P which
     * follows the existing intra reconstruction path using inter-frame mode probas.
     * Other inter modes (NEAREST/NEAR/NEW/SPLIT) throw - Phase 2 work.
     */
    private static void decodeInterMacroblock(
        @NotNull State s, @NotNull BooleanDecoder br, @NotNull BooleanDecoder td,
        int mbX, int mbY
    ) {
        // Segment map (consume bits if enabled).
        if (s.updateMap) {
            if (br.decodeBit(s.segmentProbs[0]) == 0)
                br.decodeBit(s.segmentProbs[1]);
            else
                br.decodeBit(s.segmentProbs[2]);
        }

        int skip = (s.useSkipProba && br.decodeBit(s.skipP) != 0) ? 1 : 0;
        boolean isInter = br.decodeBit(s.probIntra) != 0;

        if (isInter) {
            // Reference frame: 0 = LAST, 1 = GOLDEN/ALTREF (branch we never emit).
            int refBit = br.decodeBit(s.probLast);
            if (refBit != 0) {
                br.decodeBit(s.probGf);  // consume the second ref bit even though we refuse
                throw new ImageDecodeException("VP8 GOLDEN/ALTREF references not supported (Phase 1)");
            }

            // MV-reference tree with context-dependent probabilities (libvpx parity).
            NearMvs.Result near = new NearMvs.Result();
            NearMvs.find(s.mbIsInter, s.mbMvRow, s.mbMvCol, s.mbCols, mbX, mbY, near);
            int[] mvRefProbs = new int[4];
            NearMvs.refProbs(near.cnt, mvRefProbs);
            int mvMode = br.decodeTree(VP8Tables.MV_REF_TREE, mvRefProbs);

            // Resolve the effective MV for this MB per mode (libvpx tree leaves):
            //   0 = ZEROMV     - MV is (0, 0)
            //   1 = NEARESTMV  - MV = near.nearest_mv (no wire bits)
            //   2 = NEARMV     - MV = near.near_mv (no wire bits)
            //   3 = NEWMV      - MV read from wire
            //   4 = SPLITMV    - out of Phase 2 scope
            int effInternalRow;
            int effInternalCol;
            if (mvMode == 0) {
                effInternalRow = 0;
                effInternalCol = 0;
            } else if (mvMode == 1) {
                effInternalRow = near.nearestRow;
                effInternalCol = near.nearestCol;
            } else if (mvMode == 2) {
                effInternalRow = near.nearRow;
                effInternalCol = near.nearCol;
            } else if (mvMode == 3) {
                int wireRow = decodeMvComponent(br, VP8Tables.MV_DEFAULT_PROBA[0]);
                int wireCol = decodeMvComponent(br, VP8Tables.MV_DEFAULT_PROBA[1]);
                effInternalRow = wireRow << 1;
                effInternalCol = wireCol << 1;
            } else {
                throw new ImageDecodeException("VP8 SPLITMV not supported (Phase 2 scope)");
            }

            int mbIdx = mbY * s.mbCols + mbX;
            s.mbIsInter[mbIdx] = true;
            s.mbIsI4x4[mbIdx] = false;
            s.mbMvRow[mbIdx] = effInternalRow;
            s.mbMvCol[mbIdx] = effInternalCol;
            s.mbRefFrame[mbIdx] = LoopFilter.REF_LAST;
            // mvMode: 0=ZEROMV, 1=NEARESTMV, 2=NEARMV, 3=NEWMV. NEAREST/NEAR/NEW all map
            // to MODE_OTHER_INTER per RFC 6386 section 15 (mode_lf_delta[2]).
            s.mbModeLfIdx[mbIdx] = (mvMode == 0) ? LoopFilter.MODE_ZEROMV : LoopFilter.MODE_OTHER_INTER;

            // Reset intra-mode neighbour context for downstream intra MBs.
            java.util.Arrays.fill(s.intraT, mbX * 4, mbX * 4 + 4, IntraPrediction.B_DC_PRED);
            java.util.Arrays.fill(s.intraL, IntraPrediction.B_DC_PRED);

            if (mvMode == 0 && skip != 0) {
                // ZEROMV+skip fast path: straight copy from the LAST reference, no residual.
                copyRefMb16x16(s.session.refY, s.session.refLumaStride, s.reconY, s.lumaStride, mbX, mbY);
                copyRefMb8x8(s.session.refU, s.session.refChromaStride, s.reconU, s.chromaStride, mbX, mbY);
                copyRefMb8x8(s.session.refV, s.session.refChromaStride, s.reconV, s.chromaStride, mbX, mbY);
                int[] top = s.topNz[mbX];
                for (int i = 0; i < 9; i++) { top[i] = 0; s.leftNz[i] = 0; }
                return;
            }

            decodeInterWithMv(s, br, td, mbX, mbY, skip, effInternalRow, effInternalCol);
            return;
        }

        // Intra-in-P-frame: walk the inter Y-mode tree (5 leaves).
        int yMode = br.decodeTree(VP8Tables.YMODE_TREE, s.yModeProba);
        boolean isI4x4 = (yMode == IntraPrediction.B_PRED);
        int mbIdxIp = mbY * s.mbCols + mbX;
        s.mbIsI4x4[mbIdxIp] = isI4x4;
        s.mbRefFrame[mbIdxIp] = LoopFilter.REF_INTRA;
        s.mbModeLfIdx[mbIdxIp] = isI4x4 ? LoopFilter.MODE_BPRED : LoopFilter.MODE_NON_BPRED_INTRA;

        int[][] subModes = null;
        if (isI4x4) {
            subModes = new int[4][4];
            // RFC 6386 section 16.2: inter-frame B_PRED sub-block modes use the fixed
            // context-free {@link VP8Tables#BMODE_PROBA_INTER}, not the keyframe
            // context-dependent {@link VP8Tables#KF_BMODE_PROB[above][left]}.
            for (int by = 0; by < 4; by++) {
                for (int bx = 0; bx < 4; bx++) {
                    subModes[by][bx] = br.decodeTree(
                        VP8Tables.BMODE_TREE, VP8Tables.BMODE_PROBA_INTER
                    );
                }
            }
            for (int bx = 0; bx < 4; bx++) s.intraT[mbX * 4 + bx] = subModes[3][bx];
            for (int by = 0; by < 4; by++) s.intraL[by] = subModes[by][3];
        } else {
            int ctxMode = mb16ToSubMode(yMode);
            java.util.Arrays.fill(s.intraT, mbX * 4, mbX * 4 + 4, ctxMode);
            java.util.Arrays.fill(s.intraL, ctxMode);
        }

        // UV-mode tree (inter variant: same tree shape, different default probas).
        int uvMode = br.decodeTree(VP8Tables.UV_MODE_TREE, s.uvModeProba);

        // Residual parsing is identical to the keyframe intra path. Re-use the existing
        // inline layout by calling a helper that mirrors decodeMacroblock's residual block.
        int[] top = s.topNz[mbX];
        int[] left = s.leftNz;

        short[] dcValues = new short[16];
        short[][] yCoefs = new short[16][16];
        short[][] uCoefs = new short[4][16];
        short[][] vCoefs = new short[4][16];

        if (skip != 0) {
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
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 4; x++) {
                    short[] zz = new short[16];
                    int ctx = top[x] + left[y];
                    int nz = VP8TokenDecoder.decode(td, zz, first, ctx, acType, s.proba);
                    top[x] = left[y] = nz;
                    short[] raster = zigzagToRaster(zz);
                    if (!isI4x4)
                        raster[0] = dcValues[y * 4 + x];
                    else
                        raster[0] = (short) (raster[0] * s.y1Dc);
                    for (int i = 1; i < 16; i++) raster[i] = (short) (raster[i] * s.y1Ac);
                    yCoefs[y * 4 + x] = raster;
                }
            }
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

        if (isI4x4)
            reconstructI4x4(s, mbX, mbY, subModes, yCoefs);
        else
            reconstruct16x16(s, mbX, mbY, yMode, yCoefs);

        reconstructChroma(s.reconU, s.chromaStride, mbX, mbY, uvMode, uCoefs);
        reconstructChroma(s.reconV, s.chromaStride, mbX, mbY, uvMode, vCoefs);
    }

    /**
     * Reconstructs a P-frame inter MB with a known effective MV. Called from every
     * NEAREST / NEAR / NEW / ZERO-with-residual branch in {@link #decodeInterMacroblock}
     * after the mode decision has resolved the MV. Builds the motion-compensated
     * prediction via {@link SubpelPrediction}, parses the residual through the same
     * Y2 + Y AC + chroma pipeline used by intra i16x16, and writes the reconstruction.
     *
     * @param effInternalRow effective MV row in 1/8-pel internal units
     * @param effInternalCol effective MV col in 1/8-pel internal units
     */
    private static void decodeInterWithMv(
        @NotNull State s, @NotNull BooleanDecoder br, @NotNull BooleanDecoder td,
        int mbX, int mbY, int skip, int effInternalRow, int effInternalCol
    ) {
        // buildInterPrediction takes wire (quarter-pel); convert from internal (>>1).
        int wireRow = effInternalRow >> 1;
        int wireCol = effInternalCol >> 1;

        // Build motion-compensated prediction (luma 16x16, chroma U/V 8x8) via 6-tap.
        short[] predY = new short[256];
        short[] predU = new short[64];
        short[] predV = new short[64];
        buildInterPrediction(s, mbX, mbY, wireRow, wireCol, predY, predU, predV);

        int[] top = s.topNz[mbX];
        int[] left = s.leftNz;

        short[] dcValues = new short[16];
        short[][] yCoefs = new short[16][16];
        short[][] uCoefs = new short[4][16];
        short[][] vCoefs = new short[4][16];

        if (skip != 0) {
            // Skip with NEW_MV: no residual, prediction is the final reconstruction.
            for (int i = 0; i < 9; i++) { top[i] = 0; left[i] = 0; }
        } else {
            // Y2 first (inter MBs that aren't B_PRED always have Y2).
            short[] zz = new short[16];
            int ctx = top[8] + left[8];
            int nz = VP8TokenDecoder.decode(td, zz, 0, ctx, VP8Tables.TYPE_I16_DC, s.proba);
            top[8] = left[8] = nz;
            short[] y2Raster = zigzagToRaster(zz);
            y2Raster[0] = (short) (y2Raster[0] * s.y2Dc);
            for (int i = 1; i < 16; i++) y2Raster[i] = (short) (y2Raster[i] * s.y2Ac);
            DCT.inverseWHT(y2Raster, dcValues);

            // 16 Y AC blocks.
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 4; x++) {
                    short[] yzz = new short[16];
                    int yctx = top[x] + left[y];
                    int ynz = VP8TokenDecoder.decode(td, yzz, 1, yctx, VP8Tables.TYPE_I16_AC, s.proba);
                    top[x] = left[y] = ynz;
                    short[] raster = zigzagToRaster(yzz);
                    raster[0] = dcValues[y * 4 + x];
                    for (int i = 1; i < 16; i++) raster[i] = (short) (raster[i] * s.y1Ac);
                    yCoefs[y * 4 + x] = raster;
                }
            }
            // 2x2 U then 2x2 V.
            for (int y = 0; y < 2; y++) {
                for (int x = 0; x < 2; x++) {
                    short[] uzz = new short[16];
                    int uctx = top[4 + x] + left[4 + y];
                    int unz = VP8TokenDecoder.decode(td, uzz, 0, uctx, VP8Tables.TYPE_CHROMA_A, s.proba);
                    top[4 + x] = left[4 + y] = unz;
                    short[] raster = zigzagToRaster(uzz);
                    raster[0] = (short) (raster[0] * s.uvDc);
                    for (int i = 1; i < 16; i++) raster[i] = (short) (raster[i] * s.uvAc);
                    uCoefs[y * 2 + x] = raster;
                }
            }
            for (int y = 0; y < 2; y++) {
                for (int x = 0; x < 2; x++) {
                    short[] vzz = new short[16];
                    int vctx = top[6 + x] + left[6 + y];
                    int vnz = VP8TokenDecoder.decode(td, vzz, 0, vctx, VP8Tables.TYPE_CHROMA_A, s.proba);
                    top[6 + x] = left[6 + y] = vnz;
                    short[] raster = zigzagToRaster(vzz);
                    raster[0] = (short) (raster[0] * s.uvDc);
                    for (int i = 1; i < 16; i++) raster[i] = (short) (raster[i] * s.uvAc);
                    vCoefs[y * 2 + x] = raster;
                }
            }
        }

        // Add residual to prediction (or just commit prediction when skip), clamp, write.
        commitInterRecon16x16(s.reconY, s.lumaStride, mbX * 16, mbY * 16, predY, yCoefs, skip != 0);
        commitInterRecon8x8 (s.reconU, s.chromaStride, mbX * 8,  mbY * 8,  predU, uCoefs, skip != 0);
        commitInterRecon8x8 (s.reconV, s.chromaStride, mbX * 8,  mbY * 8,  predV, vCoefs, skip != 0);
    }

    /**
     * Builds the motion-compensated Y/U/V prediction for an inter MB at wire MV
     * {@code (wireRow, wireCol)}. Mirror of {@code VP8Encoder.buildInterPrediction} -
     * both sides must produce bit-identical predictions.
     */
    private static void buildInterPrediction(
        @NotNull State s, int mbX, int mbY, int wireRow, int wireCol,
        short @NotNull [] predY, short @NotNull [] predU, short @NotNull [] predV
    ) {
        int refStrideY = s.session.refLumaStride;
        int refH = s.session.refMbRows * 16;
        int refStrideC = s.session.refChromaStride;
        int refHC = s.session.refMbRows * 8;

        int lumaIntRow = wireRow << 1;
        int lumaIntCol = wireCol << 1;
        int lumaY = mbY * 16 + (lumaIntRow >> 3);
        int lumaX = mbX * 16 + (lumaIntCol >> 3);
        int lumaSubY = lumaIntRow & 7;
        int lumaSubX = lumaIntCol & 7;
        SubpelPrediction.predict6tap(s.session.refY, refStrideY, refH,
            lumaX, lumaY, lumaSubX, lumaSubY, predY, 16, 16, 16);

        int chromaRow = chromaMv(lumaIntRow);
        int chromaCol = chromaMv(lumaIntCol);
        int chromaY = mbY * 8 + (chromaRow >> 3);
        int chromaX = mbX * 8 + (chromaCol >> 3);
        int chromaSubY = chromaRow & 7;
        int chromaSubX = chromaCol & 7;
        SubpelPrediction.predict6tap(s.session.refU, refStrideC, refHC,
            chromaX, chromaY, chromaSubX, chromaSubY, predU, 8, 8, 8);
        SubpelPrediction.predict6tap(s.session.refV, refStrideC, refHC,
            chromaX, chromaY, chromaSubX, chromaSubY, predV, 8, 8, 8);
    }

    /** Round-away-from-zero division of the internal luma MV by 2 (libvpx parity). */
    private static int chromaMv(int internalLumaMv) {
        int adj = 1 | (internalLumaMv >> 31);
        return (internalLumaMv + adj) / 2;
    }

    /** Commits a 16x16 inter-MB reconstruction = prediction + IDCT(residual), clamped. */
    private static void commitInterRecon16x16(
        short @NotNull [] plane, int planeStride, int baseX, int baseY,
        short @NotNull [] pred, short @NotNull [] @NotNull [] coefs, boolean skip
    ) {
        for (int by = 0; by < 4; by++) {
            for (int bx = 0; bx < 4; bx++) {
                short[] residual = new short[16];
                if (!skip) DCT.inverseDCT(coefs[by * 4 + bx], residual);
                for (int y = 0; y < 4; y++) {
                    for (int x = 0; x < 4; x++) {
                        int p = pred[(by * 4 + y) * 16 + bx * 4 + x] + residual[y * 4 + x];
                        plane[(baseY + by * 4 + y) * planeStride + baseX + bx * 4 + x]
                            = (short) Math.clamp(p, 0, 255);
                    }
                }
            }
        }
    }

    /** Commits an 8x8 chroma inter-MB reconstruction = prediction + IDCT(residual), clamped. */
    private static void commitInterRecon8x8(
        short @NotNull [] plane, int planeStride, int baseX, int baseY,
        short @NotNull [] pred, short @NotNull [] @NotNull [] coefs, boolean skip
    ) {
        for (int by = 0; by < 2; by++) {
            for (int bx = 0; bx < 2; bx++) {
                short[] residual = new short[16];
                if (!skip) DCT.inverseDCT(coefs[by * 2 + bx], residual);
                for (int y = 0; y < 4; y++) {
                    for (int x = 0; x < 4; x++) {
                        int p = pred[(by * 4 + y) * 8 + bx * 4 + x] + residual[y * 4 + x];
                        plane[(baseY + by * 4 + y) * planeStride + baseX + bx * 4 + x]
                            = (short) Math.clamp(p, 0, 255);
                    }
                }
            }
        }
    }

    /**
     * Reads one MV component in wire form. Mirrors libvpx's {@code read_mvcomponent}.
     *
     * @param r input boolean decoder
     * @param probs the 19-element probability vector for this component
     * @return the decoded component value in quarter-pel wire units, sign-inclusive
     */
    private static int decodeMvComponent(@NotNull BooleanDecoder r, int @NotNull [] probs) {
        int x = 0;
        if (r.decodeBit(probs[VP8Tables.MVP_IS_SHORT]) != 0) {
            // Large MV: read bits 0..2, then bits 9..4 reverse, then bit 3 conditionally.
            for (int i = 0; i < 3; i++)
                x += r.decodeBit(probs[VP8Tables.MVP_BITS + i]) << i;
            for (int i = VP8Tables.MV_LONG_BITS - 1; i > 3; i--)
                x += r.decodeBit(probs[VP8Tables.MVP_BITS + i]) << i;
            // Bit 3: implicit 1 when no higher bits set (since x >= 8 with bits 4+ = 0 forces bit 3).
            if ((x & 0xFFF0) == 0 || r.decodeBit(probs[VP8Tables.MVP_BITS + 3]) != 0)
                x += 8;
        } else {
            int[] smallProbs = new int[VP8Tables.MV_SHORT_COUNT - 1];
            System.arraycopy(probs, VP8Tables.MVP_SHORT, smallProbs, 0, smallProbs.length);
            x = r.decodeTree(VP8Tables.MV_SMALL_TREE, smallProbs);
        }
        if (x != 0 && r.decodeBit(probs[VP8Tables.MVP_SIGN]) != 0)
            x = -x;
        return x;
    }


    /** Copies the 16x16 luma MB at ({@code mbX}, {@code mbY}) from {@code ref} to {@code dst}. */
    private static void copyRefMb16x16(
        short @NotNull [] ref, int refStride, short @NotNull [] dst, int dstStride, int mbX, int mbY
    ) {
        for (int y = 0; y < 16; y++) {
            int refOff = (mbY * 16 + y) * refStride + mbX * 16;
            int dstOff = (mbY * 16 + y) * dstStride + mbX * 16;
            System.arraycopy(ref, refOff, dst, dstOff, 16);
        }
    }

    /** Copies the 8x8 chroma MB at ({@code mbX}, {@code mbY}) from {@code ref} to {@code dst}. */
    private static void copyRefMb8x8(
        short @NotNull [] ref, int refStride, short @NotNull [] dst, int dstStride, int mbX, int mbY
    ) {
        for (int y = 0; y < 8; y++) {
            int refOff = (mbY * 8 + y) * refStride + mbX * 8;
            int dstOff = (mbY * 8 + y) * dstStride + mbX * 8;
            System.arraycopy(ref, refOff, dst, dstOff, 8);
        }
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
