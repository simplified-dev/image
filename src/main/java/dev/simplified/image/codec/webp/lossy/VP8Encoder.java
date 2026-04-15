package dev.simplified.image.codec.webp.lossy;

import dev.simplified.image.pixel.PixelBuffer;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

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

    /**
     * Number of token partitions we emit. Must be a power of two in {@code {1, 2, 4, 8}}.
     * libwebp's cwebp defaults to 4. Macroblock rows are interleaved across partitions
     * via {@code mb_y & (NUM_TOKEN_PARTITIONS - 1)}, letting multi-threaded decoders
     * parse rows in parallel.
     */
    private static final int NUM_TOKEN_PARTITIONS = 4;
    /** {@code log2(NUM_TOKEN_PARTITIONS)}; encoded into the 2-bit header field. */
    private static final int LOG2_NUM_TOKEN_PARTITIONS = 2;

    private VP8Encoder() { }

    /**
     * Probability for the {@code prob_skip_false} header field in P-frames. 0x80 = 50/50,
     * a safe default that libwebp also converges near for typical content.
     */
    private static final int INTER_SKIP_PROBA = 0x80;

    /**
     * All-zero delta vector used when {@code use_lf_delta = 0} in the frame header.
     * Shared so the encoder does not allocate a throwaway array per frame.
     */
    private static final int[] EMPTY_LF_DELTA = new int[4];

    /**
     * Probability for the {@code prob_intra} header field in P-frames. Biased low so
     * encoding an inter-MB is cheap - our Phase 1 use case (stationary tooltip frames)
     * has most MBs inter-skip.
     */
    private static final int INTER_PROB_INTRA = 32;

    /**
     * Probability for the {@code prob_last} header field in P-frames. 255 means "always
     * LAST" since Phase 1 never emits GOLDEN or ALTREF references.
     */
    private static final int INTER_PROB_LAST = 255;

    /**
     * Probability for the {@code prob_gf} header field in P-frames. Irrelevant when
     * {@code prob_last = 255} (the tree branch is never entered), set to 128 to match
     * libwebp's default.
     */
    private static final int INTER_PROB_GF = 128;

    /** Per-frame encoding state threaded through the MB loop. */
    static final class State {
        /** {@code true} for a keyframe, {@code false} for a P-frame (inter frame). */
        final boolean isKeyframe;
        /** Backing session for P-frame reference buffers, or {@code null} for a stateless keyframe. */
        @Nullable final VP8EncoderSession session;

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

        // Per-MB flags (row-major mbY*mbCols + mbX).
        // mbIsI4x4: B_PRED MB - consumed by the encoder's loop-filter pass before
        //           captureReference so encoder/decoder references stay in sync.
        // mbIsInter: the MB was coded inter - used by {@link NearMvs#find} for the
        //           NEAREST / NEAR / NEW / ZERO mode context.
        // mbMvRow/mbMvCol: per-MB MV in libvpx 1/8-pel internal units (valid only
        //           when mbIsInter is true). Inter-skip + NEAREST + NEAR + NEW all
        //           write their effective MV here for downstream MBs to reference.
        final boolean[] mbIsI4x4;
        final boolean[] mbIsInter;
        final int[] mbMvRow;
        final int[] mbMvCol;

        // Per-4x4-sub-block MV grid for SPLITMV MBs (RFC 6386 section 17.3). Layout
        // mirrors VP8Decoder.State:
        //   bmiRow[mbIdx * 16 + k] where k is the 4x4 sub-block index (0..15 raster).
        // Populated only for SPLITMV MBs - non-SPLITMV MBs leave their slots at the
        // default 0 because the SPLITMV decode path gates the bmi[] read behind a
        // {@code mbSplitScheme[neighbour] >= 0} check (see VP8Decoder.decodeSplitMv).
        // Read by neighbouring SPLITMV MBs at the boundary to look up the right-column
        // sub-MV (left neighbour) or bottom-row sub-MV (above neighbour) for the 5-way
        // sub-MV-ref context classification.
        final int[] bmiRow;
        final int[] bmiCol;
        // Per-MB SPLITMV partitioning scheme (0..3) or -1 for non-SPLITMV. Sentinel
        // {@code -1} matches VP8Decoder.State and gates whether neighbouring SPLITMV
        // MBs read this MB's bmi[] entries vs the MB-level mv.
        final int[] mbSplitScheme;

        // Per-MB loop-filter classification tables (row-major mbY*mbCols + mbX):
        //   mbRefFrame[i]   = one of LoopFilter.REF_INTRA / REF_LAST / REF_GOLDEN / REF_ALTREF
        //   mbModeLfIdx[i]  = one of LoopFilter.MODE_NON_BPRED_INTRA / MODE_BPRED / MODE_ZEROMV
        //                     / MODE_OTHER_INTER / MODE_SPLITMV
        // Populated as each MB is committed and consumed at the end-of-frame loop-filter
        // pass to apply RFC 6386 section 15 ref_lf_delta + mode_lf_delta per-MB.
        final int[] mbRefFrame;
        final int[] mbModeLfIdx;

        // Quantizer steps.
        final int y1Dc, y1Ac, y2Dc, y2Ac, uvDc, uvAc;

        // Trellis / R-D matrices (per-coefficient q, iq, bias, sharpen).
        final QuantMatrix y1Mtx, y2Mtx, uvMtx;

        // R-D lambdas, derived from QI via libwebp's {@code SetupMatrices} formulas.
        // {@code score = RD_DISTO_MULT * distortion + rate * lambda}.
        final int lambdaI4, lambdaUv;
        final int lambdaTrellisI4, lambdaTrellisI16, lambdaTrellisUv;

        final BooleanEncoder header;                // first partition: frame header + per-MB modes
        final BooleanEncoder[] tokens;              // token partitions: MB row {@code y} writes to
                                                    // {@code tokens[y & (NUM_TOKEN_PARTITIONS - 1)]}

        // Loop-filter delta header state (RFC 6386 section 15). When useLfDelta is true,
        // the frame header emits refLfDelta[4] and modeLfDelta[4], and the post-encode
        // loop filter applies them when computing per-MB filter strength.
        boolean useLfDelta;
        int[] refLfDelta = new int[4];
        int[] modeLfDelta = new int[4];

        // Per-frame sign-bias flags for GOLDEN / ALTREF (RFC 6386 section 18.3). Consumed
        // by {@link NearMvs#find} so cross-reference neighbours get their MVs flipped
        // when bias differs from the current MB's ref. Default false - our encoder emits
        // only LAST-ref MBs so the flip never triggers in self-roundtrip.
        boolean signBiasGolden;
        boolean signBiasAltref;

        // MV component probabilities active for this frame (RFC 6386 section 19.2).
        // Keyframes start from {@link VP8Tables#MV_DEFAULT_PROBA}; P-frames inherit from
        // session and may mutate slots chosen by {@link #writeFrameHeader} when emitting
        // updates. Passed to {@link #emitMvComponent} so tokens match what the decoder
        // will decode.
        final int[][] mvProba;

        // MV branch counts accumulated during this frame's emit pass. Layout
        // {@code [component][slot][outcome]} with outcome in {0, 1}. Stashed on the
        // session at end-of-frame so the NEXT frame's header can compute optimal probs
        // with one-frame lag (libvpx's approach - avoids a two-pass encode).
        final int[][][] mvBranches = new int[2][VP8Tables.NUM_MV_PROBAS][2];

        // Per-frame Y-mode / UV-mode probabilities (RFC 6386 section 16.2). For P-frames
        // the header may emit an all-or-nothing update block overriding the default
        // INTER table; for keyframes these remain at the defaults.
        final int[] yModeProba;
        final int[] uvModeProba;

        // Per-frame Y-mode / UV-mode tree branch observations, accumulated across every
        // intra-in-P MB's tree emit. Stashed on the session at end-of-frame for the next
        // frame's header emit to compute whether a proba update is worth emitting.
        final int[][] yModeBranches = new int[VP8Tables.YMODE_PROBA_INTER.length][2];
        final int[][] uvModeBranches = new int[VP8Tables.UV_MODE_PROBA_INTER.length][2];

        // Per-frame coefficient token probabilities and branch observation counters.
        // tokenProba starts at COEFFS_PROBA_0 every frame (refresh_entropy_probs=0
        // semantics); the header emit may mutate individual slots based on prior-frame
        // observations in session.prevTokenBranches. tokenBranches accumulates this
        // frame's observations during emit for the NEXT frame's header.
        final int[][][][] tokenProba;
        final int[][][][][] tokenBranches =
            new int[VP8Tables.NUM_TYPES][VP8Tables.NUM_BANDS][VP8Tables.NUM_CTX][VP8Tables.NUM_PROBAS][2];

        // Derived per-frame residual-cost bundle, built from tokenProba after writeFrameHeader
        // finalises the updates. Threaded through residual-cost estimation and trellis
        // quantization so R-D decisions use post-update probs instead of COEFFS_PROBA_0
        // defaults.
        VP8Costs.TokenCosts tokenCosts;

        // Parallelism for the motion-search prepass (copied from session). {@code 1}
        // = serial; {@code > 1} uses a throwaway ForkJoinPool. Motion search results
        // are cached in {@code cachedMvWire*} and looked up by {@link #findBestMv}
        // during the serial R-D loop, so thread count does not affect bit output.
        final int motionSearchThreads;
        // Per-MB motion-search results, populated by the prepass. {@code cachedMvFound[i]}
        // = false mirrors the computeBestMv null-return case (all candidates clipped).
        final int[] cachedMvWireRow;
        final int[] cachedMvWireCol;
        final boolean[] cachedMvFound;

        // Segment map state (RFC 6386 section 10). Disabled by default - only the
        // {@link #encodeWithSegmentMap} test hook flips {@code useSegment} on. When
        // enabled, {@code writeFrameHeader} emits the segment sub-header and the per-MB
        // dispatcher emits {@code mbSegmentId} via the 4-leaf segment tree before each
        // macroblock's normal header bits.
        boolean useSegment;
        boolean updateMbSegmentationMap;
        boolean segmentAbsoluteDelta;
        final int[] segmentQuantDelta = new int[4];
        final int[] segmentLfDelta = new int[4];
        final int[] segmentProbs = { 255, 255, 255 };
        final int[] mbSegmentId;
        // Per-segment effective filter level, passed to {@link LoopFilter#filterFrame}
        // so per-segment LF deltas apply at filter time. All 4 entries equal the base
        // filter level when segments are off.
        final int[] segFilterLevel = new int[4];

        State(int width, int height, int qi, boolean isKeyframe, @Nullable VP8EncoderSession session) {
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
            this.bmiRow = new int[mbCols * mbRows * 16];       // defaults to 0; filled per SPLITMV MB
            this.bmiCol = new int[mbCols * mbRows * 16];
            this.mbSplitScheme = new int[mbCols * mbRows];     // sentinel -1 = non-SPLITMV
            java.util.Arrays.fill(this.mbSplitScheme, -1);
            this.mbRefFrame = new int[mbCols * mbRows];       // defaults to REF_INTRA = 0
            this.mbModeLfIdx = new int[mbCols * mbRows];       // defaults to MODE_NON_BPRED_INTRA = 0
            this.mbSegmentId = new int[mbCols * mbRows];       // defaults to 0 (single-segment)

            this.motionSearchThreads = session != null ? session.getMotionSearchThreads() : 1;
            this.cachedMvWireRow = new int[mbCols * mbRows];
            this.cachedMvWireCol = new int[mbCols * mbRows];
            this.cachedMvFound = new boolean[mbCols * mbRows];

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
            int perPartBytes = Math.max(1024, mbCols * mbRows * 1024 / NUM_TOKEN_PARTITIONS);
            this.tokens = new BooleanEncoder[NUM_TOKEN_PARTITIONS];
            for (int i = 0; i < NUM_TOKEN_PARTITIONS; i++)
                this.tokens[i] = new BooleanEncoder(perPartBytes);

            // Keyframes always reset MV probs to defaults; P-frames inherit the probs
            // the decoder carried forward from the prior frame via session.mvProba.
            this.mvProba = new int[2][VP8Tables.NUM_MV_PROBAS];
            if (isKeyframe || session == null) {
                for (int c = 0; c < 2; c++)
                    System.arraycopy(VP8Tables.MV_DEFAULT_PROBA[c], 0, this.mvProba[c], 0,
                        VP8Tables.NUM_MV_PROBAS);
            } else {
                for (int c = 0; c < 2; c++)
                    System.arraycopy(session.mvProba[c], 0, this.mvProba[c], 0,
                        VP8Tables.NUM_MV_PROBAS);
            }
            // Y-mode / UV-mode probs start at the inter-frame defaults every frame.
            // The decoder re-initialises the same way. Writing an update block in the
            // header mutates these for the rest of the frame only.
            this.yModeProba = VP8Tables.YMODE_PROBA_INTER.clone();
            this.uvModeProba = VP8Tables.UV_MODE_PROBA_INTER.clone();

            // Token probabilities also start at defaults every frame (our encoder emits
            // refresh_entropy_probs=0, so updates are per-frame only - decoder resets
            // to defaults at each frame boundary and we must match).
            this.tokenProba = deepCloneTokenProba(VP8Tables.COEFFS_PROBA_0);
        }
    }

    /** Deep-copies the {@code [NUM_TYPES][NUM_BANDS][NUM_CTX][NUM_PROBAS]} proba table. */
    private static int[][][][] deepCloneTokenProba(int @NotNull [][][][] src) {
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

    /**
     * Encodes pixel data into a VP8 keyframe bitstream.
     *
     * @param pixels source pixel buffer
     * @param quality encoding quality in {@code [0.0, 1.0]}
     * @return the encoded VP8 payload bytes
     */
    public static byte @NotNull [] encode(@NotNull PixelBuffer pixels, float quality) {
        return encodeKeyframe(pixels, quality, null);
    }

    /**
     * Encodes a VP8 keyframe, optionally snapshotting reconstructed planes into {@code session}
     * for later P-frame reference use. Package-private entry point used by
     * {@link VP8EncoderSession#encode(PixelBuffer, float, boolean)}.
     *
     * @param pixels source pixel buffer
     * @param quality encoding quality in {@code [0.0, 1.0]}
     * @param session session to capture the reconstructed planes into, or {@code null}
     *                for a stateless encode
     * @return the encoded VP8 payload bytes
     */
    static byte @NotNull [] encodeKeyframe(
        @NotNull PixelBuffer pixels, float quality, @Nullable VP8EncoderSession session
    ) {
        return encodeFrame(pixels, quality, session, /*isKeyframe=*/ true);
    }

    /**
     * Encodes a VP8 P-frame against the reference held in {@code session}. Emits inter-
     * frame header and inter-MB layouts. Package-private entry point used by
     * {@link VP8EncoderSession#encode(PixelBuffer, float, boolean)} when a reference is
     * available and the caller does not force a keyframe.
     *
     * @param pixels source pixel buffer
     * @param quality encoding quality in {@code [0.0, 1.0]}
     * @param session session holding the {@code LAST} reference frame - must satisfy
     *                {@link VP8EncoderSession#hasReference()} and match the source dimensions
     * @return the encoded VP8 payload bytes
     */
    static byte @NotNull [] encodePFrame(
        @NotNull PixelBuffer pixels, float quality, @NotNull VP8EncoderSession session
    ) {
        return encodeFrame(pixels, quality, session, /*isKeyframe=*/ false);
    }

    /**
     * Test / library hook: encodes a keyframe with {@code use_lf_delta = 1} and the
     * supplied {@code ref_lf_delta} / {@code mode_lf_delta} vectors. Enables spec
     * conformance coverage of RFC 6386 section 15 without enabling deltas by default
     * on the public {@link #encode(PixelBuffer, float)} entry point.
     *
     * @param pixels source pixel buffer
     * @param quality encoding quality in {@code [0.0, 1.0]}
     * @param refLfDelta 4-entry delta vector indexed by reference frame
     *                   ({@link LoopFilter#REF_INTRA} etc.)
     * @param modeLfDelta 4-entry delta vector (B_PRED, ZEROMV, NEW/NEAREST/NEAR, SPLITMV)
     * @return the encoded VP8 payload bytes
     */
    public static byte @NotNull [] encodeWithLfDeltas(
        @NotNull PixelBuffer pixels, float quality,
        int @NotNull [] refLfDelta, int @NotNull [] modeLfDelta
    ) {
        if (refLfDelta.length != 4 || modeLfDelta.length != 4)
            throw new IllegalArgumentException("LF delta vectors must have length 4");
        return encodeFrame(pixels, quality, null, /*isKeyframe=*/ true, true, refLfDelta, modeLfDelta);
    }

    /**
     * Test / library hook: encodes a P-frame where every macroblock uses the SPLITMV
     * path with scheme {@link VP8Tables#MBSPLIT_TOP_BOTTOM} and two
     * {@link VP8Tables#SUB_MV_REF_ZERO} sub-MVs. All sub-MVs resolve to {@code (0, 0)}
     * and {@code mb_skip} is set, so the decoded reconstruction is a straight copy of
     * the session's {@code LAST} reference - bit-identical to what ZEROMV+skip would
     * produce, but routed through the RFC 6386 section 17.3 SPLITMV parse path.
     * <p>
     * Exists to exercise the decoder's SPLITMV wire-format coverage without wiring
     * SPLITMV into the R-D candidate set. Source pixels are not used for
     * reconstruction - they only drive the quantizer / filter-level derivations in
     * {@link #writeFrameHeader}.
     *
     * @param pixels source pixel buffer - must match the session's reference dimensions
     * @param quality encoding quality in {@code [0.0, 1.0]}
     * @param session session holding the {@code LAST} reference (seeded by a prior
     *                keyframe encode)
     * @return the encoded VP8 P-frame payload bytes
     */
    static byte @NotNull [] encodeWithSplitMv(
        @NotNull PixelBuffer pixels, float quality, @NotNull VP8EncoderSession session
    ) {
        if (!session.hasReference())
            throw new IllegalStateException("encodeWithSplitMv requires a LAST reference");
        int qi = qualityToQi(quality);
        boolean filtered = pickFilterLevel(qi) > 0;
        return encodeFrame(pixels, quality, session, /*isKeyframe=*/ false,
            filtered, EMPTY_LF_DELTA,
            filtered ? LIBVPX_DEFAULT_P_MODE_LF_DELTA : EMPTY_LF_DELTA,
            /*forceSplitMv=*/ true);
    }

    /**
     * Test / library hook: encodes a keyframe with {@code use_segment = 1} and the
     * supplied per-segment feature data. Assigns segment IDs to macroblocks via
     * {@code mbSegmentAssignment[mbY * mbCols + mbX]} (values {@code 0..3}). Exercises
     * RFC 6386 section 10 (segment header + per-MB segment tree emission) and
     * section 15 (per-segment loop filter deltas).
     * <p>
     * Quant deltas are emitted to the bitstream - the decoder applies them at dequant
     * time - but the encoder itself still uses a single quantizer at the frame's base
     * QI. Bit-stream-wise this is valid but asymmetric; the encoder will over-emit
     * residual bits relative to what a fully segment-aware encoder would pick, and
     * the decoder's reconstructed pixels may differ from a single-segment baseline
     * by the dequant delta. The hook therefore exists to validate decoder behaviour,
     * not to exercise a production encode path.
     *
     * @param pixels source pixel buffer
     * @param quality encoding quality in {@code [0.0, 1.0]}
     * @param mbSegmentAssignment per-MB segment IDs (row-major, length mbCols*mbRows);
     *                            each entry must be in {@code [0, 3]}
     * @param segmentQuantDelta 4-entry per-segment quantizer delta (signed 7-bit)
     * @param segmentLfDelta 4-entry per-segment filter-level delta (signed 6-bit)
     * @param segmentAbsoluteDelta if true, deltas replace the base value rather than add
     * @return the encoded VP8 keyframe payload bytes
     */
    public static byte @NotNull [] encodeWithSegmentMap(
        @NotNull PixelBuffer pixels, float quality,
        int @NotNull [] mbSegmentAssignment,
        int @NotNull [] segmentQuantDelta, int @NotNull [] segmentLfDelta,
        boolean segmentAbsoluteDelta
    ) {
        if (segmentQuantDelta.length != 4 || segmentLfDelta.length != 4)
            throw new IllegalArgumentException("per-segment delta vectors must have length 4");
        int mbCols = (pixels.width() + 15) / 16;
        int mbRows = (pixels.height() + 15) / 16;
        if (mbSegmentAssignment.length != mbCols * mbRows)
            throw new IllegalArgumentException(
                "mbSegmentAssignment length must equal mbCols*mbRows");
        for (int id : mbSegmentAssignment)
            if (id < 0 || id > 3)
                throw new IllegalArgumentException("segment id must be in [0, 3]");
        return encodeSegmentMappedKeyframe(pixels, quality, mbSegmentAssignment,
            segmentQuantDelta, segmentLfDelta, segmentAbsoluteDelta);
    }

    private static byte @NotNull [] encodeSegmentMappedKeyframe(
        @NotNull PixelBuffer pixels, float quality,
        int @NotNull [] mbSegmentAssignment,
        int @NotNull [] segmentQuantDelta, int @NotNull [] segmentLfDelta,
        boolean segmentAbsoluteDelta
    ) {
        int width = pixels.width();
        int height = pixels.height();
        int qi = qualityToQi(quality);

        State s = new State(width, height, qi, /*isKeyframe=*/ true, /*session=*/ null);
        s.useSegment = true;
        s.updateMbSegmentationMap = true;
        s.segmentAbsoluteDelta = segmentAbsoluteDelta;
        System.arraycopy(segmentQuantDelta, 0, s.segmentQuantDelta, 0, 4);
        System.arraycopy(segmentLfDelta, 0, s.segmentLfDelta, 0, 4);
        System.arraycopy(mbSegmentAssignment, 0, s.mbSegmentId, 0, mbSegmentAssignment.length);
        // Uniform segment prior: 50/50 at probs[0], 50/50 inside each half. Works for
        // any assignment without requiring histogram analysis.
        s.segmentProbs[0] = 128;
        s.segmentProbs[1] = 128;
        s.segmentProbs[2] = 128;

        // Derive per-segment effective filter levels for LoopFilter.
        int baseFilterLevel = pickFilterLevel(qi);
        for (int seg = 0; seg < 4; seg++) {
            int level = segmentAbsoluteDelta
                ? segmentLfDelta[seg]
                : baseFilterLevel + segmentLfDelta[seg];
            s.segFilterLevel[seg] = Math.clamp(level, 0, 63);
        }

        writeFrameHeader(s, qi);
        s.tokenCosts = new VP8Costs.TokenCosts(s.tokenProba);

        int[] argb = pixels.pixels();
        for (int mbY = 0; mbY < s.mbRows; mbY++) {
            java.util.Arrays.fill(s.leftNz, 0);
            java.util.Arrays.fill(s.intraL, IntraPrediction.B_DC_PRED);
            for (int mbX = 0; mbX < s.mbCols; mbX++) {
                emitSegmentId(s.header, s.mbSegmentId[mbY * s.mbCols + mbX], s.segmentProbs);
                encodeMacroblock(s, argb, width, height, mbX, mbY);
            }
        }

        boolean anySegmentFilters = false;
        for (int lvl : s.segFilterLevel) if (lvl > 0) { anySegmentFilters = true; break; }
        if (anySegmentFilters) {
            LoopFilter.filterFrame(
                s.reconY, s.reconU, s.reconV,
                s.lumaStride, s.chromaStride,
                s.mbCols, s.mbRows,
                /*simpleFilter=*/ false, baseFilterLevel, /*sharpness=*/ 0,
                s.mbRefFrame, s.mbModeLfIdx,
                s.useLfDelta, s.refLfDelta, s.modeLfDelta,
                s.mbSegmentId, s.segFilterLevel
            );
        }

        byte[] headerBytes = s.header.toByteArray();
        byte[][] tokenBytes = new byte[NUM_TOKEN_PARTITIONS][];
        for (int i = 0; i < NUM_TOKEN_PARTITIONS; i++)
            tokenBytes[i] = s.tokens[i].toByteArray();
        return assembleFrame(width, height, headerBytes, tokenBytes, /*isKeyframe=*/ true);
    }

    /**
     * Emits the 4-leaf segment-ID tree at the current per-frame segment probabilities
     * (RFC 6386 section 10 / 19.3). Wire format: {@code bit[probs[0]]} selects the
     * segment-{0,1} vs segment-{2,3} half, then {@code bit[probs[1]]} or
     * {@code bit[probs[2]]} picks within the half.
     */
    private static void emitSegmentId(
        @NotNull BooleanEncoder h, int segId, int @NotNull [] probs
    ) {
        if (segId < 2) {
            h.encodeBit(probs[0], 0);
            h.encodeBit(probs[1], segId);
        } else {
            h.encodeBit(probs[0], 1);
            h.encodeBit(probs[2], segId - 2);
        }
    }

    /**
     * Mode-LF-delta vector applied to P-frames at non-zero filter level, mirroring
     * libvpx's {@code set_default_lf_deltas} ({@code vp8/encoder/onyx_if.c:1272-1291}).
     * Index layout is {@code [BPRED, ZEROMV, OTHER_INTER, SPLITMV]}. Only the ZEROMV
     * slot gets a non-zero default: {@code -2} reduces (not disables) the loop-filter
     * strength on stationary inter-skip MBs, attenuating cumulative drift across
     * chained P-frames without losing deblocking on moving content.
     */
    private static final int[] LIBVPX_DEFAULT_P_MODE_LF_DELTA = { 0, -2, 0, 0 };

    private static byte @NotNull [] encodeFrame(
        @NotNull PixelBuffer pixels, float quality, @Nullable VP8EncoderSession session, boolean isKeyframe
    ) {
        // P-frames at non-zero filter level pick up libvpx's default mode_lf_deltas so
        // stationary-inter-skip chains don't accumulate loop-filter drift. Keyframes and
        // filter-off paths skip the 16-bit header overhead. Callers with specific delta
        // requirements use {@link #encodeWithLfDeltas} which bypasses this default.
        int qi = qualityToQi(quality);
        if (!isKeyframe && pickFilterLevel(qi) > 0) {
            return encodeFrame(pixels, quality, session, isKeyframe,
                true, EMPTY_LF_DELTA, LIBVPX_DEFAULT_P_MODE_LF_DELTA);
        }
        return encodeFrame(pixels, quality, session, isKeyframe, false, EMPTY_LF_DELTA, EMPTY_LF_DELTA);
    }

    private static byte @NotNull [] encodeFrame(
        @NotNull PixelBuffer pixels, float quality, @Nullable VP8EncoderSession session, boolean isKeyframe,
        boolean useLfDelta, int @NotNull [] refLfDelta, int @NotNull [] modeLfDelta
    ) {
        return encodeFrame(pixels, quality, session, isKeyframe,
            useLfDelta, refLfDelta, modeLfDelta, /*forceSplitMv=*/ false);
    }

    private static byte @NotNull [] encodeFrame(
        @NotNull PixelBuffer pixels, float quality, @Nullable VP8EncoderSession session, boolean isKeyframe,
        boolean useLfDelta, int @NotNull [] refLfDelta, int @NotNull [] modeLfDelta,
        boolean forceSplitMv
    ) {
        int width = pixels.width();
        int height = pixels.height();
        int qi = qualityToQi(quality);

        // Keyframes start a fresh adaptation window: the decoder's proba state resets
        // to defaults on each keyframe, so stats from prior inter frames are stale.
        // Clearing prev* here keeps session encodes bit-identical to static encodes
        // across keyframe boundaries, and avoids the next P-frame picking updates off
        // stale data.
        if (isKeyframe && session != null) {
            session.prevMvBranches = null;
            session.prevYModeBranches = null;
            session.prevUvModeBranches = null;
            session.prevTokenBranches = null;
        }

        State s = new State(width, height, qi, isKeyframe, session);
        s.useLfDelta = useLfDelta;
        System.arraycopy(refLfDelta, 0, s.refLfDelta, 0, 4);
        System.arraycopy(modeLfDelta, 0, s.modeLfDelta, 0, 4);

        // Derive per-segment effective filter levels. Single-segment streams leave
        // segmentLfDelta all zero so every entry equals the base level.
        int baseFilterLevel = pickFilterLevel(qi);
        for (int seg = 0; seg < 4; seg++) {
            int level = s.segmentAbsoluteDelta
                ? s.segmentLfDelta[seg]
                : baseFilterLevel + s.segmentLfDelta[seg];
            s.segFilterLevel[seg] = Math.clamp(level, 0, 63);
        }

        writeFrameHeader(s, qi);

        // Freeze the post-update coefficient probs into a derived cost bundle for the
        // R-D pass. refresh_entropy_probs=0 means tokenProba is stable for the rest of
        // the frame, so a single build suffices.
        s.tokenCosts = new VP8Costs.TokenCosts(s.tokenProba);

        // Per-MB encoding loop.
        int[] argb = pixels.pixels();

        // P-frame motion-search prepass: compute the best NEW-MV per MB up front so
        // the serial R-D loop can look up the cached result instead of re-running
        // the integer+half+quarter pel search. Parallelizable when
        // {@code motionSearchThreads > 1} - the prepass reads only session-snapshot
        // state, never mutates it. Skipped for keyframes (no motion search) and for
        // the forced-SPLITMV test hook (doesn't call findBestMv).
        if (!isKeyframe && !forceSplitMv)
            runMotionSearchPrepass(s, argb, width, height);
        for (int mbY = 0; mbY < s.mbRows; mbY++) {
            java.util.Arrays.fill(s.leftNz, 0);
            java.util.Arrays.fill(s.intraL, IntraPrediction.B_DC_PRED);
            for (int mbX = 0; mbX < s.mbCols; mbX++) {
                // Segment ID emits first in the MB header stream (RFC 6386 section 10),
                // before skip / intra / mode bits. Mirrors the decoder's
                // applySegmentId call at the top of its per-MB parse.
                if (s.useSegment && s.updateMbSegmentationMap)
                    emitSegmentId(s.header, s.mbSegmentId[mbY * s.mbCols + mbX], s.segmentProbs);

                if (isKeyframe)
                    encodeMacroblock(s, argb, width, height, mbX, mbY);
                else if (forceSplitMv)
                    emitForcedSplitMvMacroblock(s, mbX, mbY);
                else
                    encodeInterMacroblock(s, argb, width, height, mbX, mbY);
            }
        }

        // Apply the same loop filter the decoder will so the captured reference matches
        // post-filter state on both sides. Without this, encoder's ZERO-MV skip decision
        // compares source against a pre-filter recon while decoder copies a post-filter
        // ref - producing cumulative drift across P-frames.
        int filterLevel = pickFilterLevel(qi);
        boolean anySegmentFilters = false;
        for (int lvl : s.segFilterLevel) if (lvl > 0) { anySegmentFilters = true; break; }
        if (anySegmentFilters) {
            LoopFilter.filterFrame(
                s.reconY, s.reconU, s.reconV,
                s.lumaStride, s.chromaStride,
                s.mbCols, s.mbRows,
                /*simpleFilter=*/ false, filterLevel, /*sharpness=*/ 0,
                s.mbRefFrame, s.mbModeLfIdx,
                s.useLfDelta, s.refLfDelta, s.modeLfDelta,
                s.mbSegmentId, s.segFilterLevel
            );
        }

        if (session != null) {
            // Stash MV proba state + this-frame branch counts for the next frame's
            // header emit (RFC 6386 section 19.2 one-frame-lag proba updates).
            for (int c = 0; c < 2; c++)
                System.arraycopy(s.mvProba[c], 0, session.mvProba[c], 0, VP8Tables.NUM_MV_PROBAS);
            session.prevMvBranches = s.mvBranches;
            session.prevYModeBranches = s.yModeBranches;
            session.prevUvModeBranches = s.uvModeBranches;
            session.prevTokenBranches = s.tokenBranches;

            // Keyframe refreshes all three reference slots (RFC 6386 section 9.7 -
            // keyframes implicitly overwrite LAST, GOLDEN, and ALTREF). P-frames
            // default to {@code refresh_last = 1, refresh_golden = refresh_alt = 0,
            // copy_buffer_to_golden = copy_buffer_to_alt = 0}, so only LAST updates.
            session.captureReferenceLast(
                s.reconY, s.reconU, s.reconV,
                s.lumaStride, s.chromaStride,
                s.mbCols, s.mbRows, width, height
            );
            if (isKeyframe) {
                session.captureReferenceGolden(
                    s.reconY, s.reconU, s.reconV,
                    s.lumaStride, s.chromaStride,
                    s.mbCols, s.mbRows, width, height
                );
                session.captureReferenceAltref(
                    s.reconY, s.reconU, s.reconV,
                    s.lumaStride, s.chromaStride,
                    s.mbCols, s.mbRows, width, height
                );
            }
        }

        byte[] headerBytes = s.header.toByteArray();
        byte[][] tokenBytes = new byte[NUM_TOKEN_PARTITIONS][];
        for (int i = 0; i < NUM_TOKEN_PARTITIONS; i++)
            tokenBytes[i] = s.tokens[i].toByteArray();
        return assembleFrame(width, height, headerBytes, tokenBytes, isKeyframe);
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

        // Keyframe-only fields (color_space + clamp_type) per RFC 6386 section 9.2.
        if (s.isKeyframe) {
            e.encodeBool(0);       // color_space
            e.encodeBool(0);       // clamp_type
        }

        // Segment header (RFC 6386 section 10 / 19.2). Default is use_segment = 0;
        // the {@link #encodeWithSegmentMap} test hook flips s.useSegment on and
        // populates the per-segment data.
        if (!s.useSegment) {
            e.encodeBool(0);           // use_segment
        } else {
            e.encodeBool(1);           // use_segment
            e.encodeBool(s.updateMbSegmentationMap ? 1 : 0);    // update_mb_segmentation_map

            // update_segment_feature_data. We always emit (simpler than diffing) when
            // either quant or lf deltas are non-zero. Emit a 1-bit with all-zero payload
            // otherwise to keep the decoder in sync.
            boolean anyDelta = false;
            for (int i = 0; i < 4; i++)
                if (s.segmentQuantDelta[i] != 0 || s.segmentLfDelta[i] != 0) {
                    anyDelta = true;
                    break;
                }
            if (!anyDelta) {
                e.encodeBool(0);       // update_segment_feature_data = 0
            } else {
                e.encodeBool(1);       // update_segment_feature_data = 1
                e.encodeBool(s.segmentAbsoluteDelta ? 1 : 0);   // segment_feature_mode
                for (int i = 0; i < 4; i++) {
                    if (s.segmentQuantDelta[i] != 0) {
                        e.encodeBool(1);
                        e.encodeSint(s.segmentQuantDelta[i], 7);
                    } else {
                        e.encodeBool(0);
                    }
                }
                for (int i = 0; i < 4; i++) {
                    if (s.segmentLfDelta[i] != 0) {
                        e.encodeBool(1);
                        e.encodeSint(s.segmentLfDelta[i], 6);
                    } else {
                        e.encodeBool(0);
                    }
                }
            }

            // Segment tree probabilities. Emit all 3 (flag + 8-bit literal per slot).
            if (s.updateMbSegmentationMap) {
                for (int i = 0; i < 3; i++) {
                    if (s.segmentProbs[i] != 255) {
                        e.encodeBool(1);
                        e.encodeUint(s.segmentProbs[i], 8);
                    } else {
                        e.encodeBool(0);
                    }
                }
            }
        }

        // Filter header: normal filter at qi-derived strength. RFC 6386 section 9.1
        // pairs version=0 (which our frame tag emits) with the normal loop filter +
        // 6-tap bicubic sub-pel reconstruction; simple-filter requires version=1 which
        // also mandates bilinear sub-pel. Our sub-pel path is 6-tap everywhere so we
        // must use the normal filter for internal consistency + libvpx interop.
        int filterLevel = pickFilterLevel(qi);
        e.encodeBool(0);                // simple filter bit: 0 = normal (RFC section 15.3)
        e.encodeUint(filterLevel, 6);   // loop_filter_level
        e.encodeUint(0, 3);             // sharpness
        if (!s.useLfDelta) {
            e.encodeBool(0);            // use_lf_delta
        } else {
            e.encodeBool(1);            // use_lf_delta
            e.encodeBool(1);            // update_lf_delta (always update - simpler than diffing)
            // RFC 6386 section 15: 4 ref_lf_delta signed 6-bit entries with per-entry update flags.
            for (int i = 0; i < 4; i++) {
                if (s.refLfDelta[i] != 0) {
                    e.encodeBool(1);
                    e.encodeSint(s.refLfDelta[i], 6);
                } else {
                    e.encodeBool(0);
                }
            }
            for (int i = 0; i < 4; i++) {
                if (s.modeLfDelta[i] != 0) {
                    e.encodeBool(1);
                    e.encodeSint(s.modeLfDelta[i], 6);
                } else {
                    e.encodeBool(0);
                }
            }
        }

        e.encodeUint(LOG2_NUM_TOKEN_PARTITIONS, 2);    // log2 of the number of token partitions

        // Quantizer: write base QI, no per-component deltas (matches State's derived steps).
        e.encodeUint(qi, 7);
        e.encodeBool(0);           // y1_dc_delta_present
        e.encodeBool(0);           // y2_dc_delta_present
        e.encodeBool(0);           // y2_ac_delta_present
        e.encodeBool(0);           // uv_dc_delta_present
        e.encodeBool(0);           // uv_ac_delta_present

        // Inter-frame reference-buffer management (RFC 6386 section 9.7). Phase 1 never
        // uses GOLDEN or ALTREF, so refresh flags are 0 and copy-buffer selectors are 0
        // (copy current LAST into the slot). Sign-bias bits are 0 (no MV flipping).
        if (!s.isKeyframe) {
            e.encodeBool(0);       // refresh_golden_frame
            e.encodeBool(0);       // refresh_alt_frame
            e.encodeUint(0, 2);    // copy_buffer_to_golden (only emitted if refresh_golden=0)
            e.encodeUint(0, 2);    // copy_buffer_to_alt    (only emitted if refresh_alt=0)
            e.encodeBool(0);       // sign_bias_golden
            e.encodeBool(0);       // sign_bias_alt
        }

        e.encodeBool(0);           // refresh_entropy_probs (keep prior-frame proba)

        // Inter-only: refresh_last is always 1 for Phase 1 (we emit a fresh LAST each frame).
        if (!s.isKeyframe)
            e.encodeBool(1);       // refresh_last

        // VP8ParseProba: 1056 per-slot update flags (RFC 6386 paragraph 13). Uses
        // one-frame-lag stats from the session's prevTokenBranches to compute optimal
        // probs and emit updates where cost-beneficial. On keyframes or the first
        // frame of a session, prevTokenBranches is null and we emit all zeros.
        emitTokenProbaUpdates(e, s);

        // Per-MB skip-flag proba. Keyframes keep the existing use_skip_proba=0 behaviour;
        // P-frames enable skip with a 50/50 default so the inter-skip MBs actually save
        // bits (one skip bit per MB instead of a zero Y2 + 16 zero YAC + 8 zero UV tokens).
        if (s.isKeyframe) {
            e.encodeBool(0);       // use_skip_proba = 0
        } else {
            e.encodeBool(1);       // use_skip_proba = 1
            e.encodeUint(INTER_SKIP_PROBA, 8);
        }

        // Inter-frame mode/MV probabilities (RFC 6386 sections 16.2 and 17.2).
        if (!s.isKeyframe) {
            e.encodeUint(INTER_PROB_INTRA, 8);   // prob_intra
            e.encodeUint(INTER_PROB_LAST, 8);    // prob_last
            e.encodeUint(INTER_PROB_GF, 8);      // prob_gf

            // y_mode_probs / uv_mode_probs: 1-bit all-or-nothing gate. When set, 4 or 3
            // full 8-bit probs follow. Uses one-frame-lag stats on the session to decide.
            emitIntraModeProbaUpdate(e, s.yModeProba, s.session != null ? s.session.prevYModeBranches : null);
            emitIntraModeProbaUpdate(e, s.uvModeProba, s.session != null ? s.session.prevUvModeBranches : null);

            // mv_prob_update: 2 components * 19 probs. Uses one-frame-lag branch counts
            // from the prior frame (via session.prevMvBranches) to pick optimal probs
            // per slot. The first P-frame has no prior stats and emits all-zero flags.
            emitMvProbaUpdates(e, s);
        }
    }

    /**
     * Emits the per-frame MV proba update section (RFC 6386 section 19.2). For each of
     * the 2x{@link VP8Tables#NUM_MV_PROBAS} slots, compares the cost of continuing to
     * code prior-frame observations at the current proba vs switching to the optimal
     * quantized proba; emits the 1-bit update flag and optional 7-bit literal
     * accordingly. Mutates {@link State#mvProba} so subsequent MV emits use the
     * post-update value.
     * <p>
     * The MV wire format stores probs as a 7-bit literal mapping 0 -> 1 and non-zero
     * -> (lit << 1), so the reachable set is {1, 2, 4, 6, ..., 254} (128 values).
     * Costs are compared in {@code 1/256} bits via {@link VP8Costs#bitCost}.
     */
    private static void emitMvProbaUpdates(@NotNull BooleanEncoder e, @NotNull State s) {
        int[][][] prevBranches = s.session != null ? s.session.prevMvBranches : null;

        for (int c = 0; c < 2; c++) {
            for (int p = 0; p < VP8Tables.NUM_MV_PROBAS; p++) {
                int currentProb = s.mvProba[c][p];
                int updateFlagProb = VP8Tables.MV_UPDATE_PROBA[c][p];

                // No prior stats (first inter frame) or no observations at this slot:
                // keep current prob with a single "no update" flag bit.
                if (prevBranches == null) {
                    e.encodeBit(updateFlagProb, 0);
                    continue;
                }
                int ct0 = prevBranches[c][p][0];
                int ct1 = prevBranches[c][p][1];
                int total = ct0 + ct1;
                if (total == 0) {
                    e.encodeBit(updateFlagProb, 0);
                    continue;
                }

                // Optimal prob(0) from observations, clamped + quantized to the wire set.
                int rawProb = (ct0 * 256 + total / 2) / total;
                if (rawProb < 1) rawProb = 1;
                if (rawProb > 255) rawProb = 255;
                int lit = rawProb >> 1;
                int quantProb = lit != 0 ? (lit << 1) : 1;

                if (quantProb == currentProb) {
                    e.encodeBit(updateFlagProb, 0);
                    continue;
                }

                // Cost comparison in 1/256 bit units. Both paths include the update-flag
                // bit itself so the comparison is symmetric.
                long keepCost = (long) ct0 * VP8Costs.bitCost(0, currentProb)
                              + (long) ct1 * VP8Costs.bitCost(1, currentProb)
                              + VP8Costs.bitCost(0, updateFlagProb);
                long updateCost = (long) ct0 * VP8Costs.bitCost(0, quantProb)
                                + (long) ct1 * VP8Costs.bitCost(1, quantProb)
                                + VP8Costs.bitCost(1, updateFlagProb)
                                + 7 * 256;                   // 7-bit literal cost

                if (updateCost < keepCost) {
                    e.encodeBit(updateFlagProb, 1);
                    e.encodeUint(lit, 7);
                    s.mvProba[c][p] = quantProb;
                } else {
                    e.encodeBit(updateFlagProb, 0);
                }
            }
        }
    }

    /**
     * Emits the all-or-nothing proba-update block for the inter-frame Y-mode (4 slots)
     * or UV-mode (3 slots) tree (RFC 6386 section 16.2). A single {@code encodeBool}
     * gate at 50/50 prob controls whether the full {@code slots * 8} bits of new probs
     * follow. When the update is cost-beneficial the passed {@code probs} array is
     * mutated to the new values so subsequent MB emits use the updated probs.
     * <p>
     * No-op (emits just the 0-gate bit) when {@code prevBranches} is {@code null} or
     * carries zero observations - matches the first-inter-frame fast path.
     *
     * @param e header encoder
     * @param probs the mutable per-frame proba array (Y-mode or UV-mode)
     * @param prevBranches prior-frame observations (null when no data yet)
     */
    private static void emitIntraModeProbaUpdate(
        @NotNull BooleanEncoder e, int @NotNull [] probs, int[] @Nullable [] prevBranches
    ) {
        if (prevBranches == null) {
            e.encodeBool(0);
            return;
        }

        // Compute optimal prob per slot from observations.
        int slots = probs.length;
        int[] optimal = new int[slots];
        int totalObservations = 0;
        for (int i = 0; i < slots; i++) {
            int ct0 = prevBranches[i][0];
            int ct1 = prevBranches[i][1];
            int total = ct0 + ct1;
            totalObservations += total;
            if (total == 0) {
                optimal[i] = probs[i];                   // no data: keep current
                continue;
            }
            int rawProb = (ct0 * 256 + total / 2) / total;
            if (rawProb < 1) rawProb = 1;
            if (rawProb > 255) rawProb = 255;
            optimal[i] = rawProb;
        }
        // If zero observations anywhere, skip the update entirely.
        if (totalObservations == 0) {
            e.encodeBool(0);
            return;
        }

        // Cost comparison in 1/256-bit units. Both paths include the 50/50 gate bit
        // (bitCost = 256 regardless of outcome at prob=128). Update path adds slots * 8
        // bits of probs payload.
        long keepCost = 0;
        long updateCost = 0;
        for (int i = 0; i < slots; i++) {
            int ct0 = prevBranches[i][0];
            int ct1 = prevBranches[i][1];
            keepCost   += (long) ct0 * VP8Costs.bitCost(0, probs[i])
                        + (long) ct1 * VP8Costs.bitCost(1, probs[i]);
            updateCost += (long) ct0 * VP8Costs.bitCost(0, optimal[i])
                        + (long) ct1 * VP8Costs.bitCost(1, optimal[i]);
        }
        updateCost += (long) slots * 8 * 256;            // new probs payload

        if (updateCost < keepCost) {
            e.encodeBool(1);
            for (int i = 0; i < slots; i++) {
                e.encodeUint(optimal[i], 8);
                probs[i] = optimal[i];
            }
        } else {
            e.encodeBool(0);
        }
    }

    /** Minimum observations at a (type, band, ctx, probaIdx) slot before we consider a coef update. */
    private static final int COEF_UPDATE_MIN_OBSERVATIONS = 4;

    /**
     * Emits the 1056-slot coefficient proba update section (RFC 6386 paragraph 13).
     * For each of {@link VP8Tables#NUM_TYPES}x{@link VP8Tables#NUM_BANDS}x{@link
     * VP8Tables#NUM_CTX}x{@link VP8Tables#NUM_PROBAS} slots, compares the cost of
     * coding prior-frame observations at the current default vs the optimal
     * observed prob; emits an update bit + 8-bit new prob when the savings exceed
     * the update overhead. Mutates {@link State#tokenProba} so subsequent block
     * emits use the updated values.
     * <p>
     * Slots with fewer than {@link #COEF_UPDATE_MIN_OBSERVATIONS} observations skip
     * the comparison to avoid noisy updates driven by 1-2 outlier blocks.
     */
    private static void emitTokenProbaUpdates(@NotNull BooleanEncoder e, @NotNull State s) {
        // Keyframes always start fresh - they reset every proba stream (MV, Y-mode,
        // UV-mode probs all go back to defaults at the decoder's keyframe boundary)
        // so carrying coef stats from a prior inter frame is stale. Behave like a
        // stateless static encode and emit all-zero update flags.
        int[][][][][] prev = (s.isKeyframe || s.session == null)
            ? null : s.session.prevTokenBranches;
        for (int t = 0; t < VP8Tables.NUM_TYPES; t++) {
            for (int b = 0; b < VP8Tables.NUM_BANDS; b++) {
                for (int c = 0; c < VP8Tables.NUM_CTX; c++) {
                    for (int p = 0; p < VP8Tables.NUM_PROBAS; p++) {
                        int flagProb = VP8Tables.COEFFS_UPDATE_PROBA[t][b][c][p];
                        int currentProb = s.tokenProba[t][b][c][p];

                        if (prev == null) {
                            e.encodeBit(flagProb, 0);
                            continue;
                        }
                        int ct0 = prev[t][b][c][p][0];
                        int ct1 = prev[t][b][c][p][1];
                        int total = ct0 + ct1;
                        if (total < COEF_UPDATE_MIN_OBSERVATIONS) {
                            e.encodeBit(flagProb, 0);
                            continue;
                        }

                        int rawProb = (ct0 * 256 + total / 2) / total;
                        if (rawProb < 1) rawProb = 1;
                        if (rawProb > 255) rawProb = 255;
                        if (rawProb == currentProb) {
                            e.encodeBit(flagProb, 0);
                            continue;
                        }

                        // Cost comparison in 1/256-bit units. Both paths include the
                        // update-flag bit so the comparison is symmetric.
                        long keepCost = (long) ct0 * VP8Costs.bitCost(0, currentProb)
                                      + (long) ct1 * VP8Costs.bitCost(1, currentProb)
                                      + VP8Costs.bitCost(0, flagProb);
                        long updateCost = (long) ct0 * VP8Costs.bitCost(0, rawProb)
                                        + (long) ct1 * VP8Costs.bitCost(1, rawProb)
                                        + VP8Costs.bitCost(1, flagProb)
                                        + 8 * 256;               // 8-bit new prob

                        if (updateCost < keepCost) {
                            e.encodeBit(flagProb, 1);
                            e.encodeUint(rawProb, 8);
                            s.tokenProba[t][b][c][p] = rawProb;
                        } else {
                            e.encodeBit(flagProb, 0);
                        }
                    }
                }
            }
        }
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
        int mbIdx = mbY * s.mbCols + mbX;
        s.mbIsI4x4[mbIdx] = useBPred;
        s.mbIsInter[mbIdx] = false;
        s.mbRefFrame[mbIdx] = LoopFilter.REF_INTRA;
        s.mbModeLfIdx[mbIdx] = useBPred ? LoopFilter.MODE_BPRED : LoopFilter.MODE_NON_BPRED_INTRA;

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
        //    block sees the correct {@code top_nz + left_nz} context). MB rows are
        //    interleaved across {@code NUM_TOKEN_PARTITIONS} partitions so a multi-
        //    threaded decoder can parse rows in parallel.
        BooleanEncoder tokenPart = s.tokens[mbY & (NUM_TOKEN_PARTITIONS - 1)];
        emitLumaTokens(s, tokenPart, mbX, useBPred, i16.y2ZigZag, useBPred ? bpred.yAc : i16.yAcZigZag);
        ChromaTrellisResult uResult = trellisAndEmitChroma(s, tokenPart, mbX, mb.cb, predU, /*isU=*/ true);
        ChromaTrellisResult vResult = trellisAndEmitChroma(s, tokenPart, mbX, mb.cr, predV, /*isU=*/ false);

        // 10. Commit chroma reconstruction using the dequantized coefficients from trellis.
        commitChromaFromDequant(s.reconU, s.chromaStride, mbX, mbY, predU, uResult.dequant);
        commitChromaFromDequant(s.reconV, s.chromaStride, mbX, mbY, predV, vResult.dequant);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Per-macroblock encoding (inter / P-frame path)
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Motion search radius in luma pixels. Integer search runs at 1-pel step within
     * {@code [-RADIUS, +RADIUS]} on each axis, followed by an 8-neighbour half-pel
     * refinement through {@link SubpelPrediction}. Chroma prediction uses the same
     * sub-pel filter since luma 1-pel offsets can produce fractional chroma MVs.
     */
    private static final int MOTION_SEARCH_RADIUS = 16;

    /**
     * One evaluated P-frame macroblock candidate, scored by
     * {@code rdScore = RD_DISTO_MULT * sse + rate * lambda}. Concrete subtypes carry
     * enough pre-computed state (prediction, trellis outputs) that the winner can
     * emit its header + residual bits and commit its reconstruction to the frame
     * planes without re-running the scoring pipeline.
     */
    private abstract static class InterCandidate {
        final long sse;
        final int rate;             // 1/256 bits (header + mode-tree + MV + residual)
        final long rdScore;
        InterCandidate(long sse, int rate, int lambda) {
            this.sse = sse;
            this.rate = rate;
            this.rdScore = sse * VP8Costs.RD_DISTO_MULT + (long) rate * lambda;
        }
        abstract void commit(@NotNull State s, int mbX, int mbY);
    }

    /** ZEROMV + skip: reconstruction is a direct copy of the LAST reference MB at ({@code 0, 0}). */
    private static final class ZeroMvSkipCandidate extends InterCandidate {
        final int @NotNull [] mvRefProbs;
        ZeroMvSkipCandidate(long sse, int rate, int lambda, int @NotNull [] mvRefProbs) {
            super(sse, rate, lambda);
            this.mvRefProbs = mvRefProbs;
        }
        @Override
        void commit(@NotNull State s, int mbX, int mbY) {
            assert s.session != null;
            BooleanEncoder h = s.header;
            h.encodeBit(INTER_SKIP_PROBA, 1);
            h.encodeBit(INTER_PROB_INTRA, 1);
            h.encodeBit(INTER_PROB_LAST, 0);
            h.encodeBit(mvRefProbs[0], 0);   // ZEROMV leaf (single bit at probs[0])

            int mbIdx = mbY * s.mbCols + mbX;
            s.mbIsI4x4[mbIdx] = false;
            s.mbIsInter[mbIdx] = true;
            s.mbMvRow[mbIdx] = 0;
            s.mbMvCol[mbIdx] = 0;
            s.mbRefFrame[mbIdx] = LoopFilter.REF_LAST;
            s.mbModeLfIdx[mbIdx] = LoopFilter.MODE_ZEROMV;

            copyRef16x16(s.session.refY, s.session.refLumaStride, s.reconY, s.lumaStride, mbX, mbY);
            copyRef8x8(s.session.refU, s.session.refChromaStride, s.reconU, s.chromaStride, mbX, mbY);
            copyRef8x8(s.session.refV, s.session.refChromaStride, s.reconV, s.chromaStride, mbX, mbY);

            clearNzAndIntraContextForInterSkip(s, mbX);
        }
    }

    /**
     * NEAREST or NEAR + skip: reconstruction is the 6-tap motion-compensated prediction at
     * the neighbour-derived MV, with no coded residual. Saves MV-wire bits over NEW-skip.
     */
    private static final class McSkipCandidate extends InterCandidate {
        final int mvMode;               // 1 = NEAREST, 2 = NEAR
        final int internalRow, internalCol;
        final short @NotNull [] predY, predU, predV;
        final int @NotNull [] mvRefProbs;
        McSkipCandidate(long sse, int rate, int lambda, int mvMode,
                        int internalRow, int internalCol,
                        short @NotNull [] predY, short @NotNull [] predU, short @NotNull [] predV,
                        int @NotNull [] mvRefProbs) {
            super(sse, rate, lambda);
            this.mvMode = mvMode;
            this.internalRow = internalRow;
            this.internalCol = internalCol;
            this.predY = predY;
            this.predU = predU;
            this.predV = predV;
            this.mvRefProbs = mvRefProbs;
        }
        @Override
        void commit(@NotNull State s, int mbX, int mbY) {
            BooleanEncoder h = s.header;
            h.encodeBit(INTER_SKIP_PROBA, 1);
            h.encodeBit(INTER_PROB_INTRA, 1);
            h.encodeBit(INTER_PROB_LAST, 0);
            emitTreeLeaf(h, VP8Tables.MV_REF_TREE, mvMode, mvRefProbs);

            int mbIdx = mbY * s.mbCols + mbX;
            s.mbIsI4x4[mbIdx] = false;
            s.mbIsInter[mbIdx] = true;
            s.mbMvRow[mbIdx] = internalRow;
            s.mbMvCol[mbIdx] = internalCol;
            s.mbRefFrame[mbIdx] = LoopFilter.REF_LAST;
            s.mbModeLfIdx[mbIdx] = LoopFilter.MODE_OTHER_INTER;

            commitLumaToRecon(s, mbX, mbY, predY);
            commitChromaBufferToRecon(s.reconU, s.chromaStride, mbX, mbY, predU);
            commitChromaBufferToRecon(s.reconV, s.chromaStride, mbX, mbY, predV);

            clearNzAndIntraContextForInterSkip(s, mbX);
        }
    }

    /**
     * Any inter MB with a trellised residual (NEAREST, NEAR, or NEW + residual). MV-wire
     * bits are only paid for {@code mvMode == 3} (NEW); NEAREST and NEAR infer the MV
     * from neighbours on decode.
     */
    private static final class InterResidualCandidate extends InterCandidate {
        final int mvMode;               // 1 = NEAREST, 2 = NEAR, 3 = NEW
        final int wireRow, wireCol;
        final int internalRow, internalCol;
        final short @NotNull [] predU, predV;
        final @NotNull I16Result i16;
        final @NotNull ChromaTrellisResult u;
        final @NotNull ChromaTrellisResult v;
        final int @NotNull [] mvRefProbs;
        InterResidualCandidate(long sse, int rate, int lambda, int mvMode,
                               int wireRow, int wireCol, int internalRow, int internalCol,
                               short @NotNull [] predU, short @NotNull [] predV,
                               @NotNull I16Result i16,
                               @NotNull ChromaTrellisResult u, @NotNull ChromaTrellisResult v,
                               int @NotNull [] mvRefProbs) {
            super(sse, rate, lambda);
            this.mvMode = mvMode;
            this.wireRow = wireRow;
            this.wireCol = wireCol;
            this.internalRow = internalRow;
            this.internalCol = internalCol;
            this.predU = predU;
            this.predV = predV;
            this.i16 = i16;
            this.u = u;
            this.v = v;
            this.mvRefProbs = mvRefProbs;
        }
        @Override
        void commit(@NotNull State s, int mbX, int mbY) {
            BooleanEncoder h = s.header;
            h.encodeBit(INTER_SKIP_PROBA, 0);
            h.encodeBit(INTER_PROB_INTRA, 1);
            h.encodeBit(INTER_PROB_LAST, 0);
            emitTreeLeaf(h, VP8Tables.MV_REF_TREE, mvMode, mvRefProbs);
            if (mvMode == 3) {
                emitMvComponent(h, wireRow, s.mvProba[0], s.mvBranches[0]);
                emitMvComponent(h, wireCol, s.mvProba[1], s.mvBranches[1]);
            }

            int mbIdx = mbY * s.mbCols + mbX;
            s.mbIsI4x4[mbIdx] = false;
            s.mbIsInter[mbIdx] = true;
            s.mbMvRow[mbIdx] = internalRow;
            s.mbMvCol[mbIdx] = internalCol;
            s.mbRefFrame[mbIdx] = LoopFilter.REF_LAST;
            // mvMode 0=ZEROMV (unused as candidate), 1=NEAREST, 2=NEAR, 3=NEW.
            // All of NEAREST / NEAR / NEW map to MODE_OTHER_INTER per RFC 6386 section 15.
            s.mbModeLfIdx[mbIdx] = (mvMode == 0) ? LoopFilter.MODE_ZEROMV : LoopFilter.MODE_OTHER_INTER;

            commitLumaToRecon(s, mbX, mbY, i16.recon);

            java.util.Arrays.fill(s.intraT, mbX * 4, mbX * 4 + 4, IntraPrediction.B_DC_PRED);
            java.util.Arrays.fill(s.intraL, IntraPrediction.B_DC_PRED);

            BooleanEncoder tokenPart = s.tokens[mbY & (NUM_TOKEN_PARTITIONS - 1)];
            emitLumaTokens(s, tokenPart, mbX, /*isBPred=*/ false, i16.y2ZigZag, i16.yAcZigZag);
            emitChromaTokensFromTrellis(s, tokenPart, mbX, u, /*isU=*/ true);
            emitChromaTokensFromTrellis(s, tokenPart, mbX, v, /*isU=*/ false);
            commitChromaFromDequant(s.reconU, s.chromaStride, mbX, mbY, predU, u.dequant);
            commitChromaFromDequant(s.reconV, s.chromaStride, mbX, mbY, predV, v.dequant);
        }
    }

    /** Intra 16x16 macroblock inside a P-frame, using inter-frame Y/UV mode probabilities. */
    private static final class IntraInPCandidate extends InterCandidate {
        final int yMode16;
        final int uvMode;
        final short @NotNull [] predU, predV;
        final @NotNull I16Result i16;
        final @NotNull ChromaTrellisResult u;
        final @NotNull ChromaTrellisResult v;
        IntraInPCandidate(long sse, int rate, int lambda, int yMode16, int uvMode,
                          short @NotNull [] predU, short @NotNull [] predV,
                          @NotNull I16Result i16,
                          @NotNull ChromaTrellisResult u, @NotNull ChromaTrellisResult v) {
            super(sse, rate, lambda);
            this.yMode16 = yMode16;
            this.uvMode = uvMode;
            this.predU = predU;
            this.predV = predV;
            this.i16 = i16;
            this.u = u;
            this.v = v;
        }
        @Override
        void commit(@NotNull State s, int mbX, int mbY) {
            BooleanEncoder h = s.header;
            h.encodeBit(INTER_SKIP_PROBA, 0);
            h.encodeBit(INTER_PROB_INTRA, 0);
            emitTreeLeafCounted(h, VP8Tables.YMODE_TREE, yMode16, s.yModeProba, s.yModeBranches);
            emitTreeLeafCounted(h, VP8Tables.UV_MODE_TREE, uvMode, s.uvModeProba, s.uvModeBranches);

            int mbIdx = mbY * s.mbCols + mbX;
            s.mbIsI4x4[mbIdx] = false;
            s.mbIsInter[mbIdx] = false;
            s.mbRefFrame[mbIdx] = LoopFilter.REF_INTRA;
            // Only i16 intra-in-P is currently emitted (no B_PRED-in-P). Extend to
            // MODE_BPRED when B_PRED candidates get wired in.
            s.mbModeLfIdx[mbIdx] = LoopFilter.MODE_NON_BPRED_INTRA;

            commitLumaToRecon(s, mbX, mbY, i16.recon);

            int ctxMode = mb16ToSubMode(yMode16);
            java.util.Arrays.fill(s.intraT, mbX * 4, mbX * 4 + 4, ctxMode);
            java.util.Arrays.fill(s.intraL, ctxMode);

            BooleanEncoder tokenPart = s.tokens[mbY & (NUM_TOKEN_PARTITIONS - 1)];
            emitLumaTokens(s, tokenPart, mbX, /*isBPred=*/ false, i16.y2ZigZag, i16.yAcZigZag);
            emitChromaTokensFromTrellis(s, tokenPart, mbX, u, /*isU=*/ true);
            emitChromaTokensFromTrellis(s, tokenPart, mbX, v, /*isU=*/ false);
            commitChromaFromDequant(s.reconU, s.chromaStride, mbX, mbY, predU, u.dequant);
            commitChromaFromDequant(s.reconV, s.chromaStride, mbX, mbY, predV, v.dequant);
        }
    }

    /**
     * B_PRED 4x4 sub-block-mode macroblock inside a P-frame. Y-mode emits the
     * {@link IntraPrediction#B_PRED} leaf of {@link VP8Tables#YMODE_TREE}, followed by
     * 16 sub-block modes via the context-free {@link VP8Tables#BMODE_PROBA_INTER} per
     * RFC 6386 section 16.2. No Y2 block (each 4x4 sub-block has its own DC + AC coefs).
     */
    private static final class BPredIntraInPCandidate extends InterCandidate {
        final int uvMode;
        final short @NotNull [] predU, predV;
        final @NotNull BPredResult bpred;
        final @NotNull ChromaTrellisResult u;
        final @NotNull ChromaTrellisResult v;
        BPredIntraInPCandidate(long sse, int rate, int lambda, int uvMode,
                               short @NotNull [] predU, short @NotNull [] predV,
                               @NotNull BPredResult bpred,
                               @NotNull ChromaTrellisResult u, @NotNull ChromaTrellisResult v) {
            super(sse, rate, lambda);
            this.uvMode = uvMode;
            this.predU = predU;
            this.predV = predV;
            this.bpred = bpred;
            this.u = u;
            this.v = v;
        }
        @Override
        void commit(@NotNull State s, int mbX, int mbY) {
            BooleanEncoder h = s.header;
            h.encodeBit(INTER_SKIP_PROBA, 0);
            h.encodeBit(INTER_PROB_INTRA, 0);
            emitTreeLeafCounted(h, VP8Tables.YMODE_TREE, IntraPrediction.B_PRED, s.yModeProba, s.yModeBranches);
            emitBPredModesInter(s, h, mbX, bpred.modes);
            emitTreeLeafCounted(h, VP8Tables.UV_MODE_TREE, uvMode, s.uvModeProba, s.uvModeBranches);

            int mbIdx = mbY * s.mbCols + mbX;
            s.mbIsI4x4[mbIdx] = true;
            s.mbIsInter[mbIdx] = false;
            s.mbRefFrame[mbIdx] = LoopFilter.REF_INTRA;
            s.mbModeLfIdx[mbIdx] = LoopFilter.MODE_BPRED;

            commitLumaToRecon(s, mbX, mbY, bpred.recon);

            BooleanEncoder tokenPart = s.tokens[mbY & (NUM_TOKEN_PARTITIONS - 1)];
            emitLumaTokens(s, tokenPart, mbX, /*isBPred=*/ true, /*y2ZigZag=*/ null, bpred.yAc);
            emitChromaTokensFromTrellis(s, tokenPart, mbX, u, /*isU=*/ true);
            emitChromaTokensFromTrellis(s, tokenPart, mbX, v, /*isU=*/ false);
            commitChromaFromDequant(s.reconU, s.chromaStride, mbX, mbY, predU, u.dequant);
            commitChromaFromDequant(s.reconV, s.chromaStride, mbX, mbY, predV, v.dequant);
        }
    }

    /** Returns the lower-R-D of the two candidates, or {@code next} when {@code prev} is null. */
    private static @NotNull InterCandidate chooseBetter(
        @Nullable InterCandidate prev, @NotNull InterCandidate next
    ) {
        return prev == null || next.rdScore < prev.rdScore ? next : prev;
    }

    /** Resets the per-MB nz slots and {@code intraT}/{@code intraL} context for an inter-skip MB. */
    private static void clearNzAndIntraContextForInterSkip(@NotNull State s, int mbX) {
        for (int i = 0; i < 9; i++) {
            s.topNz[mbX][i] = 0;
            s.leftNz[i] = 0;
        }
        java.util.Arrays.fill(s.intraT, mbX * 4, mbX * 4 + 4, IntraPrediction.B_DC_PRED);
        java.util.Arrays.fill(s.intraL, IntraPrediction.B_DC_PRED);
    }

    /** Copies an 8x8 chroma MB-local buffer into {@code plane} at ({@code mbX}, {@code mbY}). */
    private static void commitChromaBufferToRecon(
        short @NotNull [] plane, int stride, int mbX, int mbY, short @NotNull [] buf
    ) {
        for (int y = 0; y < 8; y++) {
            int dst = (mbY * 8 + y) * stride + mbX * 8;
            System.arraycopy(buf, y * 8, plane, dst, 8);
        }
    }

    /**
     * Encodes one macroblock in a P-frame by enumerating all valid candidates and
     * selecting the minimum-R-D one via {@code rdScore = RD_DISTO_MULT * sse + rate * lambda}.
     * Candidates considered:
     * <ul>
     *   <li><b>ZEROMV + skip</b> - direct LAST-ref copy, no residual (always).</li>
     *   <li><b>NEAREST / NEAR + skip</b> - MC prediction as reconstruction, no residual
     *       (when {@link NearMvs}'s neighbour vote counter marks the slot usable and the
     *       MV is non-zero).</li>
     *   <li><b>NEAREST / NEAR + residual</b> - MC prediction with a trellised residual
     *       through the Y2 + Y AC + chroma pipeline (when the slot is usable).</li>
     *   <li><b>NEW + residual</b> - motion-searched MV with residual, when the MV is
     *       non-zero and doesn't coincide with a cheaper NEAREST / NEAR label.</li>
     *   <li><b>Intra i16x16 in P</b> - full 16x16 intra encode with inter-frame mode
     *       probabilities.</li>
     *   <li><b>Intra B_PRED in P</b> - 4x4 sub-block-mode intra with fixed inter-frame
     *       sub-block mode probabilities (RFC 6386 section 16.2). The fallback for
     *       locally textured MBs where i16 is too coarse.</li>
     * </ul>
     * The winner's {@link InterCandidate#commit} emits its header + residual bits, commits
     * reconstruction into the frame planes, and updates the neighbour-mode and nz contexts.
     */
    private static void encodeInterMacroblock(
        @NotNull State s, int @NotNull [] argb, int width, int height, int mbX, int mbY
    ) {
        assert s.session != null && s.session.refY != null : "encodeInterMacroblock requires a session reference";

        Macroblock mb = new Macroblock();
        mb.fromARGB(argb, mbX, mbY, width, height);

        NearMvs.Result near = new NearMvs.Result();
        // Our encoder only emits LAST-ref MBs, so currentRefFrame is always REF_LAST here.
        NearMvs.find(s.mbIsInter, s.mbMvRow, s.mbMvCol, s.mbRefFrame, s.mbCols, mbX, mbY,
            LoopFilter.REF_LAST, s.signBiasGolden, s.signBiasAltref, near);
        int[] mvRefProbs = new int[4];
        NearMvs.refProbs(near.cnt, mvRefProbs);

        int lambda = s.lambdaTrellisI16;
        int baseInterHeader = VP8Costs.bitCost(1, INTER_PROB_INTRA)
                            + VP8Costs.bitCost(0, INTER_PROB_LAST);
        int baseIntraHeader = VP8Costs.bitCost(0, INTER_PROB_INTRA);
        int skipBit1 = VP8Costs.bitCost(1, INTER_SKIP_PROBA);
        int skipBit0 = VP8Costs.bitCost(0, INTER_SKIP_PROBA);

        InterCandidate best = null;

        // --- ZEROMV + skip (direct ref copy, no MC) ---
        {
            long sseY = sseVsRef16x16(s.session.refY, s.session.refLumaStride, mbX * 16, mbY * 16, mb.y);
            long sseU = sseVsRef8x8(s.session.refU, s.session.refChromaStride, mbX * 8, mbY * 8, mb.cb);
            long sseV = sseVsRef8x8(s.session.refV, s.session.refChromaStride, mbX * 8, mbY * 8, mb.cr);
            int rate = skipBit1 + baseInterHeader
                     + VP8Costs.treeLeafBitCost(VP8Tables.MV_REF_TREE, 0, mvRefProbs);
            best = chooseBetter(best, new ZeroMvSkipCandidate(sseY + sseU + sseV, rate, lambda, mvRefProbs));
        }

        // --- NEAREST + skip ---
        if (near.cnt[NearMvs.CNT_NEAREST] > 0 && (near.nearestRow | near.nearestCol) != 0) {
            best = chooseBetter(best, buildMcSkipCandidate(
                s, mb, mbX, mbY, /*mvMode=*/ 1,
                near.nearestRow, near.nearestCol,
                mvRefProbs, skipBit1, baseInterHeader, lambda
            ));
        }

        // --- NEAR + skip ---
        if (near.cnt[NearMvs.CNT_NEAR] > 0 && (near.nearRow | near.nearCol) != 0) {
            best = chooseBetter(best, buildMcSkipCandidate(
                s, mb, mbX, mbY, /*mvMode=*/ 2,
                near.nearRow, near.nearCol,
                mvRefProbs, skipBit1, baseInterHeader, lambda
            ));
        }

        // --- NEAREST + residual ---
        if (near.cnt[NearMvs.CNT_NEAREST] > 0) {
            best = chooseBetter(best, buildInterResidualCandidate(
                s, mb, mbX, mbY, /*mvMode=*/ 1,
                near.nearestRow, near.nearestCol,
                mvRefProbs, skipBit0, baseInterHeader, lambda
            ));
        }

        // --- NEAR + residual ---
        if (near.cnt[NearMvs.CNT_NEAR] > 0) {
            best = chooseBetter(best, buildInterResidualCandidate(
                s, mb, mbX, mbY, /*mvMode=*/ 2,
                near.nearRow, near.nearCol,
                mvRefProbs, skipBit0, baseInterHeader, lambda
            ));
        }

        // --- NEW + residual (motion search; skip when (0,0) or coincides with NEAREST/NEAR) ---
        int[] bestWireMv = findBestMv(s, mb, mbX, mbY);
        if (bestWireMv != null && (bestWireMv[0] != 0 || bestWireMv[1] != 0)) {
            int wireRow = bestWireMv[0];
            int wireCol = bestWireMv[1];
            boolean matchesNearest = near.cnt[NearMvs.CNT_NEAREST] > 0
                && wireRow == (near.nearestRow >> 1) && wireCol == (near.nearestCol >> 1);
            boolean matchesNear = near.cnt[NearMvs.CNT_NEAR] > 0
                && wireRow == (near.nearRow >> 1) && wireCol == (near.nearCol >> 1);
            if (!matchesNearest && !matchesNear) {
                best = chooseBetter(best, buildInterResidualCandidate(
                    s, mb, mbX, mbY, /*mvMode=*/ 3,
                    wireRow << 1, wireCol << 1,   // wire -> 1/8-pel internal
                    mvRefProbs, skipBit0, baseInterHeader, lambda
                ));
            }
        }

        // --- Intra i16x16 in P ---
        best = chooseBetter(best, buildIntraInPCandidate(
            s, mb, mbX, mbY, skipBit0, baseIntraHeader, lambda
        ));

        // --- Intra B_PRED in P (4x4 sub-block modes at fixed probs, RFC 6386 section 16.2) ---
        best = chooseBetter(best, buildBPredIntraInPCandidate(
            s, mb, mbX, mbY, skipBit0, baseIntraHeader, lambda
        ));

        best.commit(s, mbX, mbY);
    }

    /**
     * Forced-SPLITMV per-MB emit used by the {@link #encodeWithSplitMv} test hook.
     * Every MB emits scheme {@link VP8Tables#MBSPLIT_TOP_BOTTOM} with two
     * {@link VP8Tables#SUB_MV_REF_ZERO} sub-MVs and {@code mb_skip = 1}. All
     * neighbour sub-MVs are {@code (0, 0)} by construction, so the 5-way sub-MV
     * context is {@code LEFT_ABOVE_BOTH_ZERO} (ctx 4) for every slot.
     * <p>
     * The encoder's {@code bmiRow} / {@code bmiCol} / {@code mbSplitScheme} mirror
     * is populated to match the decoder so future R-D-integrated SPLITMV emits at
     * non-zero sub-MVs see consistent neighbour context.
     * <p>
     * Reconstruction is a direct {@code LAST}-reference copy, matching the decoder's
     * SPLITMV path at all-zero sub-MVs with {@code skip = 1}.
     */
    private static void emitForcedSplitMvMacroblock(@NotNull State s, int mbX, int mbY) {
        assert s.session != null && s.session.refY != null
            : "emitForcedSplitMvMacroblock requires a session reference";

        // Match the decoder's MV_REF_TREE proba derivation. Canonical prior-MB state
        // is {ref=LAST, mv=(0,0), inter=true}, so neighbour votes in NearMvs are
        // stable across the forced-SPLITMV grid.
        NearMvs.Result near = new NearMvs.Result();
        NearMvs.find(s.mbIsInter, s.mbMvRow, s.mbMvCol, s.mbRefFrame, s.mbCols, mbX, mbY,
            LoopFilter.REF_LAST, s.signBiasGolden, s.signBiasAltref, near);
        int[] mvRefProbs = new int[4];
        NearMvs.refProbs(near.cnt, mvRefProbs);

        BooleanEncoder h = s.header;
        h.encodeBit(INTER_SKIP_PROBA, 1);               // skip = 1
        h.encodeBit(INTER_PROB_INTRA, 1);               // inter
        h.encodeBit(INTER_PROB_LAST, 0);                // ref = LAST
        emitTreeLeaf(h, VP8Tables.MV_REF_TREE, 4, mvRefProbs);   // SPLITMV

        emitTreeLeaf(h, VP8Tables.MBSPLIT_TREE,
            VP8Tables.MBSPLIT_TOP_BOTTOM, VP8Tables.MBSPLIT_PROBS);

        // SUB_MV_REF_PROB[4] is the LEFT_ABOVE_BOTH_ZERO row. TOP_BOTTOM has
        // MBSPLIT_COUNT[0] = 2 sub-MVs.
        int[] zeroCtxProbs = VP8Tables.SUB_MV_REF_PROB[4];
        for (int slot = 0; slot < VP8Tables.MBSPLIT_COUNT[VP8Tables.MBSPLIT_TOP_BOTTOM]; slot++)
            emitTreeLeaf(h, VP8Tables.SUB_MV_REF_TREE, VP8Tables.SUB_MV_REF_ZERO, zeroCtxProbs);

        // Mirror decodeSplitMv's State updates: canonical MB-level MV is bmi[15] which
        // is (0, 0) for this forced pattern. The bmi[16] grid + mbSplitScheme are
        // populated so neighbouring SPLITMV MBs see the same context the decoder will.
        int mbIdx = mbY * s.mbCols + mbX;
        s.mbIsI4x4[mbIdx] = false;
        s.mbIsInter[mbIdx] = true;
        s.mbMvRow[mbIdx] = 0;
        s.mbMvCol[mbIdx] = 0;
        // bmiRow / bmiCol default to 0 from State allocation - explicit fill is a
        // no-op for this all-zero forced pattern but documents the contract for
        // future non-zero SPLITMV emit paths.
        int bmiBase = mbIdx * 16;
        for (int i = 0; i < 16; i++) {
            s.bmiRow[bmiBase + i] = 0;
            s.bmiCol[bmiBase + i] = 0;
        }
        s.mbSplitScheme[mbIdx] = VP8Tables.MBSPLIT_TOP_BOTTOM;
        s.mbRefFrame[mbIdx] = LoopFilter.REF_LAST;
        s.mbModeLfIdx[mbIdx] = LoopFilter.MODE_SPLITMV;

        // Straight LAST-ref copy - all sub-MVs are (0, 0) and skip = 1 kills residual.
        copyRef16x16(s.session.refY, s.session.refLumaStride, s.reconY, s.lumaStride, mbX, mbY);
        copyRef8x8 (s.session.refU, s.session.refChromaStride, s.reconU, s.chromaStride, mbX, mbY);
        copyRef8x8 (s.session.refV, s.session.refChromaStride, s.reconV, s.chromaStride, mbX, mbY);

        clearNzAndIntraContextForInterSkip(s, mbX);
    }

    /**
     * Classifies the sub-MV reference probability context (RFC 6386 section 17.3 /
     * libvpx {@code vp8_sub_mv_ref_prob3}) given the LEFT and ABOVE neighbour
     * sub-MVs. Returns one of the 5 row indices into {@link VP8Tables#SUB_MV_REF_PROB}:
     * {@code 0 = NORMAL}, {@code 1 = LEFT_ZERO}, {@code 2 = ABOVE_ZERO},
     * {@code 3 = LEFT_ABOVE_SAME_NONZERO}, {@code 4 = LEFT_ABOVE_BOTH_ZERO}. Mirrors
     * the classification at {@code VP8Decoder.decodeSplitMv} so encoder rate-cost
     * estimation indexes the same proba row the decoder will read.
     *
     * @param leftRow  left-neighbour sub-MV row component (1/8-pel internal)
     * @param leftCol  left-neighbour sub-MV col component
     * @param aboveRow above-neighbour sub-MV row component
     * @param aboveCol above-neighbour sub-MV col component
     * @return context row index in {@code [0, 4]}
     */
    static int subMvRefContext(int leftRow, int leftCol, int aboveRow, int aboveCol) {
        boolean lez = (leftRow == 0 && leftCol == 0);
        boolean aez = (aboveRow == 0 && aboveCol == 0);
        if (lez && aez) return 4;
        if (lez) return 1;
        if (aez) return 2;
        if (leftRow == aboveRow && leftCol == aboveCol) return 3;
        return 0;
    }

    /**
     * Builds an {@link McSkipCandidate} using the MC prediction at the supplied 1/8-pel
     * internal MV. The MV is passed through unchanged from {@link NearMvs.Result} slots;
     * the wire (quarter-pel) form is derived internally for {@link SubpelPrediction}.
     */
    private static @NotNull InterCandidate buildMcSkipCandidate(
        @NotNull State s, @NotNull Macroblock mb, int mbX, int mbY, int mvMode,
        int internalRow, int internalCol,
        int @NotNull [] mvRefProbs, int skipBit1, int baseInterHeader, int lambda
    ) {
        int wireRow = internalRow >> 1;
        int wireCol = internalCol >> 1;
        short[] predY = new short[256];
        short[] predU = new short[64];
        short[] predV = new short[64];
        buildInterPrediction(s, mbX, mbY, wireRow, wireCol, predY, predU, predV);
        long sse = sumSquaredError(mb.y, predY)
                 + sumSquaredError(mb.cb, predU)
                 + sumSquaredError(mb.cr, predV);
        int rate = skipBit1 + baseInterHeader
                 + VP8Costs.treeLeafBitCost(VP8Tables.MV_REF_TREE, mvMode, mvRefProbs);
        return new McSkipCandidate(sse, rate, lambda, mvMode,
            internalRow, internalCol, predY, predU, predV, mvRefProbs);
    }

    /**
     * Builds an {@link InterResidualCandidate} from an MC prediction + trellised residual.
     * MV is in 1/8-pel internal units. {@code mvMode == 3} (NEW) pays the MV-wire-bit cost;
     * NEAREST and NEAR (modes 1 and 2) infer the MV from neighbours.
     */
    private static @NotNull InterCandidate buildInterResidualCandidate(
        @NotNull State s, @NotNull Macroblock mb, int mbX, int mbY, int mvMode,
        int internalRow, int internalCol,
        int @NotNull [] mvRefProbs, int skipBit0, int baseInterHeader, int lambda
    ) {
        int wireRow = internalRow >> 1;
        int wireCol = internalCol >> 1;
        short[] predY = new short[256];
        short[] predU = new short[64];
        short[] predV = new short[64];
        buildInterPrediction(s, mbX, mbY, wireRow, wireCol, predY, predU, predV);

        I16Result i16 = trellisI16(s, mb.y, predY, mbX);
        ChromaTrellisResult u = trellisChroma(s, mbX, mb.cb, predU, /*isU=*/ true);
        ChromaTrellisResult v = trellisChroma(s, mbX, mb.cr, predV, /*isU=*/ false);

        short[] reconU = reconstructChroma8x8(predU, u.dequant);
        short[] reconV = reconstructChroma8x8(predV, v.dequant);
        long sse = sumSquaredError(mb.y, i16.recon)
                 + sumSquaredError(mb.cb, reconU)
                 + sumSquaredError(mb.cr, reconV);

        int modeRate = VP8Costs.treeLeafBitCost(VP8Tables.MV_REF_TREE, mvMode, mvRefProbs);
        int mvRate = (mvMode == 3) ? VP8Costs.mvWireBitCost(wireRow, wireCol) : 0;
        int rate = skipBit0 + baseInterHeader + modeRate + mvRate
                 + i16.rate + u.rate + v.rate;
        return new InterResidualCandidate(sse, rate, lambda, mvMode,
            wireRow, wireCol, internalRow, internalCol,
            predU, predV, i16, u, v, mvRefProbs);
    }

    /**
     * Builds an {@link IntraInPCandidate} - full 16x16 intra-in-P encode with inter-frame
     * mode probabilities. Runs {@link #selectBest16x16Mode} + {@link #selectBestChromaMode}
     * internally; residual rates are accumulated from the luma + chroma trellis passes.
     */
    private static @NotNull InterCandidate buildIntraInPCandidate(
        @NotNull State s, @NotNull Macroblock mb, int mbX, int mbY,
        int skipBit0, int baseIntraHeader, int lambda
    ) {
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

        short[] predY16 = new short[256];
        int yMode16 = selectBest16x16Mode(mb.y, yAbove, yLeft, yAboveLeft, predY16);
        I16Result i16 = trellisI16(s, mb.y, predY16, mbX);

        short[] predU = new short[64];
        short[] predV = new short[64];
        int uvMode = selectBestChromaMode(
            mb.cb, uAbove, uLeft, uvAboveLeft,
            mb.cr, vAbove, vLeft, vuAboveLeft,
            predU, predV
        );
        ChromaTrellisResult u = trellisChroma(s, mbX, mb.cb, predU, /*isU=*/ true);
        ChromaTrellisResult v = trellisChroma(s, mbX, mb.cr, predV, /*isU=*/ false);

        short[] reconU = reconstructChroma8x8(predU, u.dequant);
        short[] reconV = reconstructChroma8x8(predV, v.dequant);
        long sse = sumSquaredError(mb.y, i16.recon)
                 + sumSquaredError(mb.cb, reconU)
                 + sumSquaredError(mb.cr, reconV);

        int yModeRate = VP8Costs.treeLeafBitCost(VP8Tables.YMODE_TREE, yMode16, s.yModeProba);
        int uvModeRate = VP8Costs.treeLeafBitCost(VP8Tables.UV_MODE_TREE, uvMode, s.uvModeProba);
        int rate = skipBit0 + baseIntraHeader + yModeRate + uvModeRate
                 + i16.rate + u.rate + v.rate;
        return new IntraInPCandidate(sse, rate, lambda, yMode16, uvMode,
            predU, predV, i16, u, v);
    }

    /**
     * Builds a {@link BPredIntraInPCandidate} - B_PRED 4x4 sub-block-mode intra MB inside
     * a P-frame. Runs {@link #encodeBPredLuma} for the 16 sub-blocks (each picks its best
     * of 10 modes by per-sub-block R-D), selects a shared chroma mode via
     * {@link #selectBestChromaMode}, and accumulates the full rate: header + Y-mode tree
     * (B_PRED leaf) + 16 sub-block modes at fixed {@link VP8Tables#BMODE_PROBA_INTER} +
     * UV mode + luma 4x4 token rate (no Y2) + chroma token rate.
     */
    private static @NotNull InterCandidate buildBPredIntraInPCandidate(
        @NotNull State s, @NotNull Macroblock mb, int mbX, int mbY,
        int skipBit0, int baseIntraHeader, int lambda
    ) {
        short[] uAbove = extractAbove(s.reconU, s.chromaStride, mbX * 8, mbY * 8, 8);
        short[] uLeft  = extractLeft(s.reconU, s.chromaStride, mbX * 8, mbY * 8, 8);
        short[] vAbove = extractAbove(s.reconV, s.chromaStride, mbX * 8, mbY * 8, 8);
        short[] vLeft  = extractLeft(s.reconV, s.chromaStride, mbX * 8, mbY * 8, 8);
        short uvAboveLeft = (mbX > 0 && mbY > 0)
            ? s.reconU[(mbY * 8 - 1) * s.chromaStride + mbX * 8 - 1] : (short) 128;
        short vuAboveLeft = (mbX > 0 && mbY > 0)
            ? s.reconV[(mbY * 8 - 1) * s.chromaStride + mbX * 8 - 1] : (short) 128;

        BPredResult bpred = encodeBPredLuma(s, mb.y, mbX, mbY);

        short[] predU = new short[64];
        short[] predV = new short[64];
        int uvMode = selectBestChromaMode(
            mb.cb, uAbove, uLeft, uvAboveLeft,
            mb.cr, vAbove, vLeft, vuAboveLeft,
            predU, predV
        );
        ChromaTrellisResult u = trellisChroma(s, mbX, mb.cb, predU, /*isU=*/ true);
        ChromaTrellisResult v = trellisChroma(s, mbX, mb.cr, predV, /*isU=*/ false);

        short[] reconU = reconstructChroma8x8(predU, u.dequant);
        short[] reconV = reconstructChroma8x8(predV, v.dequant);
        long sse = bpred.sse
                 + sumSquaredError(mb.cb, reconU)
                 + sumSquaredError(mb.cr, reconV);

        int yModeRate = VP8Costs.treeLeafBitCost(
            VP8Tables.YMODE_TREE, IntraPrediction.B_PRED, s.yModeProba);
        int subModesRate = 0;
        for (int i = 0; i < 16; i++)
            subModesRate += VP8Costs.treeLeafBitCost(
                VP8Tables.BMODE_TREE, bpred.modes[i], VP8Tables.BMODE_PROBA_INTER);
        int uvModeRate = VP8Costs.treeLeafBitCost(
            VP8Tables.UV_MODE_TREE, uvMode, s.uvModeProba);
        int bpredLumaTokenRate = bpredLumaTokenRate(s, mbX, bpred.yAc);
        int rate = skipBit0 + baseIntraHeader
                 + yModeRate + subModesRate + uvModeRate
                 + bpredLumaTokenRate + u.rate + v.rate;
        return new BPredIntraInPCandidate(sse, rate, lambda, uvMode,
            predU, predV, bpred, u, v);
    }

    /**
     * Computes the bit cost of token-coding the 16 Y AC sub-blocks of a B_PRED MB. Mirrors
     * {@link #emitLumaTokens}'s {@code isBPred=true} walk: each sub-block is coded as
     * {@link VP8Tables#TYPE_I4_AC} starting at coefficient index 0 with a {@code top+left}
     * nz context that propagates through the 4x4 grid in raster order.
     */
    private static int bpredLumaTokenRate(@NotNull State s, int mbX, short @NotNull [] @NotNull [] yAcZigZag) {
        int[] topLocal = { s.topNz[mbX][0], s.topNz[mbX][1], s.topNz[mbX][2], s.topNz[mbX][3] };
        int[] leftLocal = { s.leftNz[0], s.leftNz[1], s.leftNz[2], s.leftNz[3] };
        int rate = 0;
        for (int by = 0; by < 4; by++) {
            for (int bx = 0; bx < 4; bx++) {
                int ctx = topLocal[bx] + leftLocal[by];
                short[] zz = yAcZigZag[by * 4 + bx];
                rate += VP8Costs.residualCost(s.tokenCosts, ctx, VP8Tables.TYPE_I4_AC, 0, zz);
                int nz = 0;
                for (int i = 0; i < 16; i++) if (zz[i] != 0) { nz = 1; break; }
                topLocal[bx] = nz;
                leftLocal[by] = nz;
            }
        }
        return rate;
    }

    /** Computes SSE between source 16x16 luma samples and the reference plane at {@code (x0, y0)}. */
    private static long sseVsRef16x16(short @NotNull [] ref, int refStride, int x0, int y0, short @NotNull [] src) {
        long sum = 0;
        for (int y = 0; y < 16; y++) {
            int refOff = (y0 + y) * refStride + x0;
            int srcOff = y * 16;
            for (int x = 0; x < 16; x++) {
                int d = src[srcOff + x] - ref[refOff + x];
                sum += (long) d * d;
            }
        }
        return sum;
    }

    /** Computes SSE between source 8x8 chroma samples and the reference plane at {@code (x0, y0)}. */
    private static long sseVsRef8x8(short @NotNull [] ref, int refStride, int x0, int y0, short @NotNull [] src) {
        long sum = 0;
        for (int y = 0; y < 8; y++) {
            int refOff = (y0 + y) * refStride + x0;
            int srcOff = y * 8;
            for (int x = 0; x < 8; x++) {
                int d = src[srcOff + x] - ref[refOff + x];
                sum += (long) d * d;
            }
        }
        return sum;
    }

    /**
     * Cache lookup for the MV previously computed by {@link #runMotionSearchPrepass}.
     * Returns {@code null} when the prepass determined no valid NEW-MV exists for
     * this MB (all candidates fell outside the reference frame bounds).
     */
    private static int @org.jetbrains.annotations.Nullable [] findBestMv(
        @NotNull State s, @NotNull Macroblock mb, int mbX, int mbY
    ) {
        int mbIdx = mbY * s.mbCols + mbX;
        if (!s.cachedMvFound[mbIdx]) return null;
        return new int[] { s.cachedMvWireRow[mbIdx], s.cachedMvWireCol[mbIdx] };
    }

    /**
     * Runs the per-MB motion search for an inter frame in advance of the serial
     * R-D loop, caching the best wire MV into {@link State#cachedMvWireRow} /
     * {@link State#cachedMvWireCol} / {@link State#cachedMvFound}.
     * <p>
     * Thread-safe: each MB's search reads {@code session.refY/refU/refV} (frozen
     * post-capture from the prior frame) and a per-MB {@link Macroblock} extracted
     * from the source pixels. No shared mutable State is touched, so the MB work
     * can run on any worker pool.
     */
    private static void runMotionSearchPrepass(
        @NotNull State s, int @NotNull [] argb, int width, int height
    ) {
        int total = s.mbCols * s.mbRows;
        java.util.function.IntConsumer mvPass = mbIdx -> {
            int mbY = mbIdx / s.mbCols;
            int mbX = mbIdx % s.mbCols;
            Macroblock mb = new Macroblock();
            mb.fromARGB(argb, mbX, mbY, width, height);
            int[] best = computeBestMv(s, mb, mbX, mbY);
            if (best == null) {
                s.cachedMvFound[mbIdx] = false;
            } else {
                s.cachedMvWireRow[mbIdx] = best[0];
                s.cachedMvWireCol[mbIdx] = best[1];
                s.cachedMvFound[mbIdx] = true;
            }
        };

        if (s.motionSearchThreads <= 1) {
            for (int mbIdx = 0; mbIdx < total; mbIdx++)
                mvPass.accept(mbIdx);
            return;
        }

        java.util.concurrent.ForkJoinPool pool =
            new java.util.concurrent.ForkJoinPool(s.motionSearchThreads);
        try {
            pool.submit(() ->
                java.util.stream.IntStream.range(0, total).parallel().forEach(mvPass)
            ).get();
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("motion-search prepass interrupted", ie);
        } catch (java.util.concurrent.ExecutionException ee) {
            throw new RuntimeException("motion-search prepass failed", ee.getCause());
        } finally {
            pool.shutdown();
        }
    }

    /**
     * Computes the best NEW-MV for a single inter MB. Integer SAD search within
     * {@link #MOTION_SEARCH_RADIUS}, then half-pel, then quarter-pel refinement (the
     * quarter-pel pass runs only when half-pel improved on integer; see RFC 6386
     * paragraph 17.1). Returns {@code null} when every candidate lies outside the
     * reference frame bounds.
     * <p>
     * Called exactly once per MB from {@link #runMotionSearchPrepass}; the per-MB
     * serial R-D loop reads the cached result via {@link #findBestMv}.
     */
    private static int @org.jetbrains.annotations.Nullable [] computeBestMv(
        @NotNull State s, @NotNull Macroblock mb, int mbX, int mbY
    ) {
        int baseX = mbX * 16;
        int baseY = mbY * 16;
        int refStride = s.session.refLumaStride;
        int refH = s.session.refMbRows * 16;

        // Integer search (SAD-based). 1-pel steps so we can refine to half-pel.
        int bestDx = 0, bestDy = 0;
        long bestSad = sadLuma16x16(s.session.refY, refStride, baseX, baseY, mb.y);
        for (int dy = -MOTION_SEARCH_RADIUS; dy <= MOTION_SEARCH_RADIUS; dy++) {
            int refY = baseY + dy;
            if (refY < 0 || refY + 16 > refH) continue;
            for (int dx = -MOTION_SEARCH_RADIUS; dx <= MOTION_SEARCH_RADIUS; dx++) {
                if (dx == 0 && dy == 0) continue;
                int refX = baseX + dx;
                if (refX < 0 || refX + 16 > refStride) continue;
                long sad = sadLuma16x16(s.session.refY, refStride, refX, refY, mb.y);
                if (sad < bestSad) {
                    bestSad = sad;
                    bestDx = dx;
                    bestDy = dy;
                }
            }
        }

        // Half-pel refinement. Wire step of 2 in quarter-pel = 0.5 luma pel.
        int integerWireRow = bestDy * 4;
        int integerWireCol = bestDx * 4;
        int bestWireRow = integerWireRow;
        int bestWireCol = integerWireCol;
        long bestSse = sseNewMvAtWire(s, mb, mbX, mbY, bestWireRow, bestWireCol);

        for (int dy = -2; dy <= 2; dy += 2) {
            for (int dx = -2; dx <= 2; dx += 2) {
                if (dy == 0 && dx == 0) continue;
                int wireRow = integerWireRow + dy;
                int wireCol = integerWireCol + dx;
                long sse = sseNewMvAtWire(s, mb, mbX, mbY, wireRow, wireCol);
                if (sse < bestSse) {
                    bestSse = sse;
                    bestWireRow = wireRow;
                    bestWireCol = wireCol;
                }
            }
        }

        // Quarter-pel refinement (wire step 1 = 0.25 luma pel). Skip when half-pel
        // didn't improve - motion is integer-aligned and 8 more MC evals won't help.
        if (bestWireRow != integerWireRow || bestWireCol != integerWireCol) {
            int halfPelBestRow = bestWireRow;
            int halfPelBestCol = bestWireCol;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dy == 0 && dx == 0) continue;
                    int wireRow = halfPelBestRow + dy;
                    int wireCol = halfPelBestCol + dx;
                    long sse = sseNewMvAtWire(s, mb, mbX, mbY, wireRow, wireCol);
                    if (sse < bestSse) {
                        bestSse = sse;
                        bestWireRow = wireRow;
                        bestWireCol = wireCol;
                    }
                }
            }
        }
        return new int[] { bestWireRow, bestWireCol };
    }

    /** Sum of absolute differences between the 16x16 source block and a reference block. */
    private static long sadLuma16x16(
        short @NotNull [] ref, int refStride, int refX, int refY, short @NotNull [] src
    ) {
        long sad = 0;
        for (int y = 0; y < 16; y++) {
            int srcOff = y * 16;
            int refOff = (refY + y) * refStride + refX;
            for (int x = 0; x < 16; x++)
                sad += Math.abs(src[srcOff + x] - ref[refOff + x]);
        }
        return sad;
    }

    /**
     * Computes the best NEW-MV for a luma sub-block of a SPLITMV MB. The sub-block
     * is anchored at {@code (mbX*16 + blockX0, mbY*16 + blockY0)} with size
     * {@code blockW x blockH}, and the source samples are taken from {@code mb.y}
     * at the corresponding offset. Searches integer SAD within
     * {@link #MOTION_SEARCH_RADIUS}, then half-pel, then quarter-pel (the quarter-pel
     * pass runs only when half-pel improved on integer; matches {@link #computeBestMv}'s
     * RFC 6386 paragraph 17.1 short-circuit).
     * <p>
     * Luma-only by construction: SPLITMV decode derives chroma MVs from a 4-way average
     * of luma sub-MVs (RFC 6386 section 17.4), so per-sub-block chroma SSE is not a
     * meaningful R-D signal at search time.
     * <p>
     * Returns {@code null} when every candidate lies outside the reference frame bounds.
     *
     * @param s       encoder state (provides session ref planes)
     * @param mb      MB-local source samples ({@code mb.y} carries the full 16x16 luma)
     * @param mbX     MB column
     * @param mbY     MB row
     * @param blockX0 sub-block x offset within the MB, in samples ({@code 0}, {@code 4},
     *                {@code 8}, or {@code 12})
     * @param blockY0 sub-block y offset within the MB, in samples
     * @param blockW  sub-block width ({@code 4}, {@code 8}, or {@code 16})
     * @param blockH  sub-block height ({@code 4}, {@code 8}, or {@code 16})
     * @return wire MV in {@code [bestWireRow, bestWireCol]} (quarter-pel units), or
     *         {@code null} if every candidate clipped
     */
    static int @org.jetbrains.annotations.Nullable [] computeBestSubBlockMv(
        @NotNull State s, @NotNull Macroblock mb, int mbX, int mbY,
        int blockX0, int blockY0, int blockW, int blockH
    ) {
        int baseX = mbX * 16 + blockX0;
        int baseY = mbY * 16 + blockY0;
        int refStride = s.session.refLumaStride;
        int refH = s.session.refMbRows * 16;

        // Integer SAD search. Anchor the source extraction at (blockX0, blockY0)
        // inside mb.y (16-sample stride); compare against ref at (baseX+dx, baseY+dy).
        // Initial candidate is dx=dy=0 (zero-MV); skip it inside the loop.
        long bestSad = sadLumaSubBlock(s.session.refY, refStride, baseX, baseY,
            mb.y, 16, blockX0, blockY0, blockW, blockH);
        int bestDx = 0, bestDy = 0;
        boolean anyValid = (baseY >= 0 && baseY + blockH <= refH
                         && baseX >= 0 && baseX + blockW <= refStride);
        for (int dy = -MOTION_SEARCH_RADIUS; dy <= MOTION_SEARCH_RADIUS; dy++) {
            int refY = baseY + dy;
            if (refY < 0 || refY + blockH > refH) continue;
            for (int dx = -MOTION_SEARCH_RADIUS; dx <= MOTION_SEARCH_RADIUS; dx++) {
                if (dx == 0 && dy == 0) continue;
                int refX = baseX + dx;
                if (refX < 0 || refX + blockW > refStride) continue;
                anyValid = true;
                long sad = sadLumaSubBlock(s.session.refY, refStride, refX, refY,
                    mb.y, 16, blockX0, blockY0, blockW, blockH);
                if (sad < bestSad) {
                    bestSad = sad;
                    bestDx = dx;
                    bestDy = dy;
                }
            }
        }
        if (!anyValid) return null;

        // Half-pel refinement (wire step 2 = 0.5 luma pel).
        int integerWireRow = bestDy * 4;
        int integerWireCol = bestDx * 4;
        int bestWireRow = integerWireRow;
        int bestWireCol = integerWireCol;
        long bestSse = sseSubBlockAtWire(s, mb, mbX, mbY,
            blockX0, blockY0, blockW, blockH, bestWireRow, bestWireCol);
        for (int dy = -2; dy <= 2; dy += 2) {
            for (int dx = -2; dx <= 2; dx += 2) {
                if (dy == 0 && dx == 0) continue;
                int wireRow = integerWireRow + dy;
                int wireCol = integerWireCol + dx;
                long sse = sseSubBlockAtWire(s, mb, mbX, mbY,
                    blockX0, blockY0, blockW, blockH, wireRow, wireCol);
                if (sse < bestSse) {
                    bestSse = sse;
                    bestWireRow = wireRow;
                    bestWireCol = wireCol;
                }
            }
        }

        // Quarter-pel refinement (wire step 1 = 0.25 luma pel). Skip when half-pel
        // didn't improve - matches computeBestMv's short-circuit.
        if (bestWireRow != integerWireRow || bestWireCol != integerWireCol) {
            int halfPelBestRow = bestWireRow;
            int halfPelBestCol = bestWireCol;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dy == 0 && dx == 0) continue;
                    int wireRow = halfPelBestRow + dy;
                    int wireCol = halfPelBestCol + dx;
                    long sse = sseSubBlockAtWire(s, mb, mbX, mbY,
                        blockX0, blockY0, blockW, blockH, wireRow, wireCol);
                    if (sse < bestSse) {
                        bestSse = sse;
                        bestWireRow = wireRow;
                        bestWireCol = wireCol;
                    }
                }
            }
        }
        return new int[] { bestWireRow, bestWireCol };
    }

    /**
     * Sum of absolute differences for a {@code blockW x blockH} luma sub-block.
     * Source samples come from {@code src[(srcY0 + y) * srcStride + (srcX0 + x)]};
     * reference samples come from {@code ref[(refY + y) * refStride + (refX + x)]}.
     */
    private static long sadLumaSubBlock(
        short @NotNull [] ref, int refStride, int refX, int refY,
        short @NotNull [] src, int srcStride, int srcX0, int srcY0,
        int blockW, int blockH
    ) {
        long sad = 0;
        for (int y = 0; y < blockH; y++) {
            int srcOff = (srcY0 + y) * srcStride + srcX0;
            int refOff = (refY + y) * refStride + refX;
            for (int x = 0; x < blockW; x++)
                sad += Math.abs(src[srcOff + x] - ref[refOff + x]);
        }
        return sad;
    }

    /**
     * Sum of squared errors for a luma sub-block prediction at the given wire MV.
     * Uses {@link SubpelPrediction#predict6tap} so sub-pel MVs invoke 6-tap interpolation
     * and integer MVs resolve to the identity filter. Luma-only by construction (see
     * {@link #computeBestSubBlockMv}).
     */
    private static long sseSubBlockAtWire(
        @NotNull State s, @NotNull Macroblock mb, int mbX, int mbY,
        int blockX0, int blockY0, int blockW, int blockH, int wireRow, int wireCol
    ) {
        int refStrideY = s.session.refLumaStride;
        int refH = s.session.refMbRows * 16;
        int lumaIntRow = wireRow << 1;
        int lumaIntCol = wireCol << 1;
        int lumaY = mbY * 16 + blockY0 + (lumaIntRow >> 3);
        int lumaX = mbX * 16 + blockX0 + (lumaIntCol >> 3);
        int lumaSubY = lumaIntRow & 7;
        int lumaSubX = lumaIntCol & 7;
        short[] pred = new short[blockW * blockH];
        SubpelPrediction.predict6tap(s.session.refY, refStrideY, refH,
            lumaX, lumaY, lumaSubX, lumaSubY, pred, blockW, blockW, blockH);
        long sse = 0;
        for (int y = 0; y < blockH; y++) {
            int srcOff = (blockY0 + y) * 16 + blockX0;
            int predOff = y * blockW;
            for (int x = 0; x < blockW; x++) {
                int d = mb.y[srcOff + x] - pred[predOff + x];
                sse += (long) d * d;
            }
        }
        return sse;
    }

    /**
     * Sum of squared errors for the full luma + chroma prediction at the given wire MV
     * (quarter-pel units). Uses {@link SubpelPrediction} for both planes, so sub-pel MVs
     * invoke the 6-tap interpolation and full-pel MVs resolve to the identity filter.
     */
    private static long sseNewMvAtWire(
        @NotNull State s, @NotNull Macroblock mb, int mbX, int mbY, int wireRow, int wireCol
    ) {
        short[] predY = new short[256];
        short[] predU = new short[64];
        short[] predV = new short[64];
        buildInterPrediction(s, mbX, mbY, wireRow, wireCol, predY, predU, predV);
        return sumSquaredError(mb.y, predY)
             + sumSquaredError(mb.cb, predU)
             + sumSquaredError(mb.cr, predV);
    }

    /**
     * Builds the motion-compensated Y/U/V prediction for an inter MB at wire MV
     * {@code (wireRow, wireCol)}. Luma uses the 6-tap filter at 1/4-pel positions;
     * chroma MVs are derived per RFC 6386 section 17.4 (round-away-zero division by 2)
     * and also use 6-tap, matching libvpx's {@code vp8_build_inter16x16_predictors_mbuv}.
     */
    private static void buildInterPrediction(
        @NotNull State s, int mbX, int mbY, int wireRow, int wireCol,
        short @NotNull [] predY, short @NotNull [] predU, short @NotNull [] predV
    ) {
        int refStrideY = s.session.refLumaStride;
        int refH = s.session.refMbRows * 16;
        int refStrideC = s.session.refChromaStride;
        int refHC = s.session.refMbRows * 8;

        // Luma internal MV = wire << 1 (1/8-pel). Integer part = internal >> 3; sub-pel = internal & 7.
        int lumaIntRow = wireRow << 1;
        int lumaIntCol = wireCol << 1;
        int lumaY = mbY * 16 + (lumaIntRow >> 3);
        int lumaX = mbX * 16 + (lumaIntCol >> 3);
        int lumaSubY = lumaIntRow & 7;
        int lumaSubX = lumaIntCol & 7;
        SubpelPrediction.predict6tap(s.session.refY, refStrideY, refH,
            lumaX, lumaY, lumaSubX, lumaSubY, predY, 16, 16, 16);

        // Chroma MV (libvpx round-away-zero /2 over the internal luma MV).
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

    /**
     * Round-away-from-zero division of an internal luma MV component by 2, producing
     * the corresponding internal chroma MV component. Bit-exact port of libvpx's
     * {@code mv_row += 1 | (mv_row >> 31); mv_row /= 2;} pattern.
     */
    private static int chromaMv(int internalLumaMv) {
        int adj = 1 | (internalLumaMv >> 31);
        return (internalLumaMv + adj) / 2;
    }

    /** Copies the 16x16 luma MB at ({@code mbX}, {@code mbY}) from {@code ref} to {@code dst}. */
    private static void copyRef16x16(
        short @NotNull [] ref, int refStride, short @NotNull [] dst, int dstStride, int mbX, int mbY
    ) {
        for (int y = 0; y < 16; y++) {
            int refOff = (mbY * 16 + y) * refStride + mbX * 16;
            int dstOff = (mbY * 16 + y) * dstStride + mbX * 16;
            System.arraycopy(ref, refOff, dst, dstOff, 16);
        }
    }

    /** Copies the 8x8 chroma MB at ({@code mbX}, {@code mbY}) from {@code ref} to {@code dst}. */
    private static void copyRef8x8(
        short @NotNull [] ref, int refStride, short @NotNull [] dst, int dstStride, int mbX, int mbY
    ) {
        for (int y = 0; y < 8; y++) {
            int refOff = (mbY * 8 + y) * refStride + mbX * 8;
            int dstOff = (mbY * 8 + y) * dstStride + mbX * 8;
            System.arraycopy(ref, refOff, dst, dstOff, 8);
        }
    }

    /**
     * Walks {@code tree} from the root to the leaf labelled {@code -mode}, emitting one
     * bit per internal node at the matching probability. Generalises {@link #emitBMode}
     * over any tree + proba pair (used for the inter-frame Y and UV mode trees).
     */
    private static void emitTreeLeaf(
        @NotNull BooleanEncoder h, int @NotNull [] tree, int mode, int @NotNull [] probs
    ) {
        int[] pathBranches = new int[16];
        int[] pathNodes = new int[16];
        int depth = findTreePath(tree, 0, -mode, pathBranches, pathNodes, 0);
        for (int i = 0; i < depth; i++)
            h.encodeBit(probs[pathNodes[i] >> 1], pathBranches[i]);
    }

    /**
     * Emits a single MV component in wire form. Mirrors libvpx's {@code encode_mvcomponent}.
     * Values with magnitude {@code < 8} use the small-MV tree; larger values emit magnitude
     * bits directly. Sign is only emitted for non-zero magnitudes.
     *
     * @param w output boolean encoder
     * @param v the component value in quarter-pel wire units
     * @param probs the 19-element probability vector for this component
     */
    private static void emitMvComponent(
        @NotNull BooleanEncoder w, int v, int @NotNull [] probs, int[] @Nullable [] branches
    ) {
        int x = Math.abs(v);
        if (x < VP8Tables.MV_SHORT_COUNT) {
            w.encodeBit(probs[VP8Tables.MVP_IS_SHORT], 0);
            countBranch(branches, VP8Tables.MVP_IS_SHORT, 0);
            // Small tree uses probs[MVP_SHORT..MVP_SHORT+6].
            int[] smallProbs = new int[VP8Tables.MV_SHORT_COUNT - 1];
            System.arraycopy(probs, VP8Tables.MVP_SHORT, smallProbs, 0, smallProbs.length);
            emitSmallMvTreeLeaf(w, x, smallProbs, branches);
            if (x == 0) return;
        } else {
            w.encodeBit(probs[VP8Tables.MVP_IS_SHORT], 1);
            countBranch(branches, VP8Tables.MVP_IS_SHORT, 1);
            // Emit bits 0..2 in order.
            for (int i = 0; i < 3; i++) {
                int bit = (x >> i) & 1;
                w.encodeBit(probs[VP8Tables.MVP_BITS + i], bit);
                countBranch(branches, VP8Tables.MVP_BITS + i, bit);
            }
            // Emit bits 9..4 in reverse order (skipping bit 3 - see below).
            for (int i = VP8Tables.MV_LONG_BITS - 1; i > 3; i--) {
                int bit = (x >> i) & 1;
                w.encodeBit(probs[VP8Tables.MVP_BITS + i], bit);
                countBranch(branches, VP8Tables.MVP_BITS + i, bit);
            }
            // Bit 3 is only emitted when any higher bit is set; otherwise implicit 1
            // (since x >= 8 always has bit 3 set when high bits are 0).
            if ((x & 0xFFF0) != 0) {
                int bit = (x >> 3) & 1;
                w.encodeBit(probs[VP8Tables.MVP_BITS + 3], bit);
                countBranch(branches, VP8Tables.MVP_BITS + 3, bit);
            }
        }
        int signBit = v < 0 ? 1 : 0;
        w.encodeBit(probs[VP8Tables.MVP_SIGN], signBit);
        countBranch(branches, VP8Tables.MVP_SIGN, signBit);
    }

    /**
     * Walks {@link VP8Tables#MV_SMALL_TREE} for small-MV magnitude {@code value}, emits
     * each branch bit using {@code smallProbs}, and also increments {@code branches} at
     * the full-MV slot index (with {@link VP8Tables#MVP_SHORT} offset) when non-null.
     * Piggybacks on {@link #findTreePath} so the tree traversal logic lives in one place.
     */
    private static void emitSmallMvTreeLeaf(
        @NotNull BooleanEncoder w, int value, int @NotNull [] smallProbs, int[] @Nullable [] branches
    ) {
        int[] pathBranches = new int[16];
        int[] pathNodes = new int[16];
        int depth = findTreePath(VP8Tables.MV_SMALL_TREE, 0, -value, pathBranches, pathNodes, 0);
        for (int i = 0; i < depth; i++) {
            int probaIdx = pathNodes[i] >> 1;
            w.encodeBit(smallProbs[probaIdx], pathBranches[i]);
            countBranch(branches, VP8Tables.MVP_SHORT + probaIdx, pathBranches[i]);
        }
    }

    /** Increments {@code branches[slot][outcome]} when {@code branches} is non-null. */
    private static void countBranch(int[] @Nullable [] branches, int slot, int outcome) {
        if (branches != null) branches[slot][outcome]++;
    }

    /**
     * Variant of {@link #emitTreeLeaf} that also records the branch decisions into
     * {@code branches[slot][outcome]}. Used for the inter-frame Y-mode / UV-mode trees
     * so the next frame's header emit can compute optimal proba updates.
     */
    private static void emitTreeLeafCounted(
        @NotNull BooleanEncoder h, int @NotNull [] tree, int mode,
        int @NotNull [] probs, int[] @Nullable [] branches
    ) {
        int[] pathBranches = new int[16];
        int[] pathNodes = new int[16];
        int depth = findTreePath(tree, 0, -mode, pathBranches, pathNodes, 0);
        for (int i = 0; i < depth; i++) {
            int slot = pathNodes[i] >> 1;
            h.encodeBit(probs[slot], pathBranches[i]);
            countBranch(branches, slot, pathBranches[i]);
        }
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
        int rate = VP8Costs.residualCost(s.tokenCosts, topNzY2 + leftNzY2, VP8Tables.TYPE_I16_DC, 0, y2ZigZag);

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
                    s.tokenCosts, dct, zz, ctx, VP8Tables.TYPE_I16_AC, s.y1Mtx, s.lambdaTrellisI16);
                yAcZigZag[by * 4 + bx] = zz;
                rate += VP8Costs.residualCost(s.tokenCosts, ctx, VP8Tables.TYPE_I16_AC, 1, zz);
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

    /**
     * Output of the per-channel chroma trellis pass: per-sub-block dequantized + zig-zag
     * coefficients, nz flags, and the token-emit bit cost. Packed so R-D scoring can
     * enumerate multiple prediction candidates without mutating {@link State#topNz}/
     * {@link State#leftNz} until a winner is picked.
     */
    private static final class ChromaTrellisResult {
        final short[][] dequant;            // 4 sub-blocks of raster-order dequantized coefficients
        final short[][] zz;                 // 4 sub-blocks of zig-zag-order quantized levels
        final int[] nz;                     // 4 nz flags in layout {@code [by*2+bx]}
        final int rate;                     // token-emit bit cost in {@code 1/256} bits
        ChromaTrellisResult(short[][] dequant, short[][] zz, int[] nz, int rate) {
            this.dequant = dequant;
            this.zz = zz;
            this.nz = nz;
            this.rate = rate;
        }
    }

    /**
     * DCTs + trellis-quantizes one chroma channel (U or V) without mutating frame state.
     * Reads the current {@code top + left} nz context for the initial scoring position
     * but propagates subsequent per-sub-block contexts through a local copy, so multiple
     * candidate predictions can be evaluated back-to-back.
     *
     * @param isU {@code true} for U (context slots 4..5), {@code false} for V (slots 6..7)
     */
    private static @NotNull ChromaTrellisResult trellisChroma(
        @NotNull State s, int mbX, short @NotNull [] src, short @NotNull [] pred, boolean isU
    ) {
        int[] top = s.topNz[mbX];
        int[] left = s.leftNz;
        int baseIdx = isU ? 4 : 6;
        int[] localTop = { top[baseIdx], top[baseIdx + 1] };
        int[] localLeft = { left[baseIdx], left[baseIdx + 1] };

        short[][] dequant = new short[4][];
        short[][] zzBlocks = new short[4][];
        int[] nzFlags = new int[4];
        int rate = 0;

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
                int ctx = localTop[bx] + localLeft[by];
                short[] zz = new short[16];
                int nz = TrellisQuantizer.quantize(
                    s.tokenCosts, dct, zz, ctx, VP8Tables.TYPE_CHROMA_A, s.uvMtx, s.lambdaTrellisUv);
                dequant[by * 2 + bx] = dct;
                zzBlocks[by * 2 + bx] = zz;
                nzFlags[by * 2 + bx] = nz;
                rate += VP8Costs.residualCost(s.tokenCosts, ctx, VP8Tables.TYPE_CHROMA_A, 0, zz);
                localTop[bx] = nz;
                localLeft[by] = nz;
            }
        }
        return new ChromaTrellisResult(dequant, zzBlocks, nzFlags, rate);
    }

    /**
     * Emits the token-coded coefficients from a {@link ChromaTrellisResult}, committing
     * per-sub-block nz flags into {@code s.topNz[mbX]} / {@code s.leftNz}. Must be called
     * in the same channel order as the trellis (U before V) with the same {@code isU}
     * argument so the context slots line up.
     */
    private static void emitChromaTokensFromTrellis(
        @NotNull State s, @NotNull BooleanEncoder tokens, int mbX,
        @NotNull ChromaTrellisResult tr, boolean isU
    ) {
        int[] top = s.topNz[mbX];
        int[] left = s.leftNz;
        int baseIdx = isU ? 4 : 6;
        for (int by = 0; by < 2; by++) {
            for (int bx = 0; bx < 2; bx++) {
                int ctx = top[baseIdx + bx] + left[baseIdx + by];
                VP8TokenEncoder.emit(
                    tokens, tr.zz[by * 2 + bx], 0, ctx,
                    VP8Tables.TYPE_CHROMA_A, s.tokenProba, s.tokenBranches);
                int nz = tr.nz[by * 2 + bx];
                top[baseIdx + bx] = nz;
                left[baseIdx + by] = nz;
            }
        }
    }

    /**
     * Convenience combination of {@link #trellisChroma} and {@link #emitChromaTokensFromTrellis}
     * that preserves the pre-Phase-2e call shape used by the keyframe path and the winning
     * inter / intra candidates. The returned {@link ChromaTrellisResult#dequant} feeds
     * {@link #commitChromaFromDequant}.
     *
     * @param isU {@code true} for U (context slots 4..5), {@code false} for V (slots 6..7)
     */
    private static @NotNull ChromaTrellisResult trellisAndEmitChroma(
        @NotNull State s, @NotNull BooleanEncoder tokens, int mbX,
        short @NotNull [] src, short @NotNull [] pred, boolean isU
    ) {
        ChromaTrellisResult tr = trellisChroma(s, mbX, src, pred, isU);
        emitChromaTokensFromTrellis(s, tokens, mbX, tr, isU);
        return tr;
    }

    /**
     * Rebuilds a single chroma channel's 8x8 reconstruction from its prediction plus
     * per-sub-block raster-order dequantized coefficients. Used by R-D scoring when
     * the candidate's reconstruction is needed for an SSE comparison but has not yet
     * been committed to the frame plane.
     */
    private static short @NotNull [] reconstructChroma8x8(
        short @NotNull [] pred, short @NotNull [] @NotNull [] dequant
    ) {
        short[] recon = new short[64];
        for (int by = 0; by < 2; by++) {
            for (int bx = 0; bx < 2; bx++) {
                short[] residual = new short[16];
                DCT.inverseDCT(dequant[by * 2 + bx], residual);
                for (int y = 0; y < 4; y++)
                    for (int x = 0; x < 4; x++) {
                        int idx = (by * 4 + y) * 8 + bx * 4 + x;
                        recon[idx] = (short) clamp(pred[idx] + residual[y * 4 + x]);
                    }
            }
        }
        return recon;
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
                        s.tokenCosts, candDeq, candZigZag, ctxNz, VP8Tables.TYPE_I4_AC, s.y1Mtx, s.lambdaTrellisI4);

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
        @NotNull State s, @NotNull BooleanEncoder t, int mbX, boolean isBPred,
        short @Nullable [] y2ZigZag, short @NotNull [] @NotNull [] yAcZigZag
    ) {
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
                t, y2ZigZag, 0, ctx, VP8Tables.TYPE_I16_DC, s.tokenProba, s.tokenBranches);
            top[8] = left[8] = nz;
            yFirst = 1;
            yType = VP8Tables.TYPE_I16_AC;
        }

        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 4; x++) {
                int ctx = top[x] + left[y];
                int nz = VP8TokenEncoder.emit(
                    t, yAcZigZag[y * 4 + x], yFirst, ctx, yType, s.tokenProba, s.tokenBranches);
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
     * Emits the 16 sub-block modes for an inter-frame B_PRED macroblock. RFC 6386
     * section 16.2 specifies that inter-frame B_PRED uses the <b>context-free</b>
     * fixed {@link VP8Tables#BMODE_PROBA_INTER} per sub-block, unlike the keyframe
     * variant in {@link #emitBPredModes} which indexes the context-dependent
     * {@link VP8Tables#KF_BMODE_PROB} by neighbouring modes. Still updates
     * {@code s.intraT} / {@code s.intraL} so a downstream keyframe MB (unlikely in
     * practice but legal) sees a consistent neighbour context.
     */
    private static void emitBPredModesInter(
        @NotNull State s, @NotNull BooleanEncoder h, int mbX, int @NotNull [] modes
    ) {
        for (int i = 0; i < 16; i++)
            emitBMode(h, modes[i], VP8Tables.BMODE_PROBA_INTER);

        // Populate cross-MB neighbour context from the bottom-row / right-column sub-blocks,
        // matching the keyframe emitBPredModes bookkeeping.
        for (int bx = 0; bx < 4; bx++) s.intraT[mbX * 4 + bx] = modes[3 * 4 + bx];
        for (int by = 0; by < 4; by++) s.intraL[by] = modes[by * 4 + 3];
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

    /**
     * Packs the frame tag, optional sync code + dimensions (keyframe only), first
     * partition, {@code (N-1)} token-partition size prefixes (little-endian uint24),
     * and {@code N} concatenated token partitions into a VP8 payload. Mirrors libwebp's
     * {@code EmitPartitionsSize} + frame-tag emission in {@code src/enc/syntax_enc.c}.
     * <p>
     * Inter frames omit the 3-byte sync code and 4-byte dimensions per RFC 6386 section
     * 9.1, shrinking the inter-frame header overhead by 7 bytes compared to a keyframe.
     */
    private static byte @NotNull [] assembleFrame(
        int width, int height, byte @NotNull [] firstPartition,
        byte @NotNull [] @NotNull [] tokenPartitions, boolean isKeyframe
    ) {
        int firstSize = firstPartition.length;
        int numParts = tokenPartitions.length;
        int sizePrefixBytes = (numParts - 1) * 3;
        int totalTokenBytes = 0;
        for (byte[] part : tokenPartitions) totalTokenBytes += part.length;

        int headerBytes = isKeyframe ? 10 : 3;
        byte[] frame = new byte[headerBytes + firstSize + sizePrefixBytes + totalTokenBytes];
        int offset = 0;

        // Frame tag: bit 0 = key_frame flag (0 = keyframe, 1 = inter), bit 4 = show_frame,
        // bits 5.. = first-partition size in bytes. Bits 1-3 are version and per our encoder
        // stay 0 (simple-filter bilinear reconstruction profile).
        int frameTag = (1 << 4) | (firstSize << 5);
        if (!isKeyframe) frameTag |= 1;
        frame[offset++] = (byte) (frameTag & 0xFF);
        frame[offset++] = (byte) ((frameTag >>> 8) & 0xFF);
        frame[offset++] = (byte) ((frameTag >>> 16) & 0xFF);

        if (isKeyframe) {
            frame[offset++] = (byte) 0x9D;
            frame[offset++] = (byte) 0x01;
            frame[offset++] = (byte) 0x2A;

            frame[offset++] = (byte) (width & 0xFF);
            frame[offset++] = (byte) ((width >>> 8) & 0x3F);
            frame[offset++] = (byte) (height & 0xFF);
            frame[offset++] = (byte) ((height >>> 8) & 0x3F);
        }

        System.arraycopy(firstPartition, 0, frame, offset, firstSize);
        offset += firstSize;

        // (N-1) size prefixes for partitions 0..N-2; the last partition's size is implicit.
        for (int p = 0; p < numParts - 1; p++) {
            int partSize = tokenPartitions[p].length;
            frame[offset++] = (byte) (partSize & 0xFF);
            frame[offset++] = (byte) ((partSize >>> 8) & 0xFF);
            frame[offset++] = (byte) ((partSize >>> 16) & 0xFF);
        }

        // Concatenated partition bodies.
        for (byte[] part : tokenPartitions) {
            System.arraycopy(part, 0, frame, offset, part.length);
            offset += part.length;
        }

        return frame;
    }

}
