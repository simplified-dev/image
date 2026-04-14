package dev.simplified.image.codec.webp;

import dev.simplified.image.pixel.PixelBuffer;
import org.jetbrains.annotations.NotNull;

/**
 * Pure Java VP8 (WebP lossy) encoder.
 * <p>
 * <b>Status: DC-only keyframe output.</b> Currently emits a single keyframe
 * where every macroblock is DC_PRED with all coefficients skipped, producing
 * a uniform mid-gray image regardless of input pixels. The purpose at this
 * stage is to exercise every required frame-header field so libwebp accepts
 * the bitstream; pixel fidelity is a later step.
 * <p>
 * The bitstream layout follows RFC 6386 and libwebp's reference decoder:
 * <ol>
 *   <li>10-byte uncompressed chunk: 3-byte frame tag (keyframe flag,
 *       version, show-frame flag, first_partition_size), 3-byte sync
 *       {@code 9D 01 2A}, 2-byte width (14 bits + 2-bit scale), 2-byte
 *       height.</li>
 *   <li>First partition (boolean-coded): color space, clamp type, segment
 *       header (disabled), filter header (disabled), token partition count,
 *       quantizer, refresh_entropy_probs, 1056 coefficient-probability
 *       update bits (all "no update"), use_skip_proba, skip_p, then the
 *       per-macroblock intra-mode headers.</li>
 *   <li>Token partition(s): empty - every macroblock signals skip=1 in its
 *       header so the decoder never descends into coefficient parsing.</li>
 * </ol>
 *
 * @see <a href="https://datatracker.ietf.org/doc/html/rfc6386">RFC 6386</a>
 */
final class VP8Encoder {

    /** Base quantizer index written into the frame header. Unused while all MBs are skipped. */
    private static final int DEFAULT_Y_AC_QI = 36;

    /** Skip probability written into the frame header. Low value makes skip=1 the cheap path. */
    private static final int SKIP_PROB = 1;

    private VP8Encoder() { }

    /**
     * Encodes pixel data into a VP8 keyframe bitstream.
     *
     * @param pixels source pixel buffer (contents ignored at the DC-only stage)
     * @param quality encoding quality in {@code [0.0, 1.0]} (unused; reserved)
     * @return the encoded VP8 payload bytes
     */
    static byte @NotNull [] encode(@NotNull PixelBuffer pixels, float quality) {
        int width = pixels.width();
        int height = pixels.height();
        int mbCols = (width + 15) / 16;
        int mbRows = (height + 15) / 16;

        byte[] firstPartition = encodeFirstPartition(mbCols, mbRows);
        byte[] tokenPartition = encodeTokenPartition();

        return assembleFrame(width, height, firstPartition, tokenPartition);
    }

    /**
     * Emits the boolean-coded first partition: frame header followed by per-macroblock
     * intra-mode decisions. Layout must match the order consumed by libwebp's
     * {@code VP8GetHeaders} / {@code VP8ParseProba} / {@code ParseIntraMode}.
     */
    private static byte @NotNull [] encodeFirstPartition(int mbCols, int mbRows) {
        BooleanEncoder e = new BooleanEncoder(2048);

        // ── Frame header (VP8GetHeaders) ─────────────────────────────────
        e.encodeBool(0);                       // color_space (YUV)
        e.encodeBool(0);                       // clamp_type (clamping required)

        // Segment header: segmentation disabled.
        e.encodeBool(0);                       // use_segment

        // Filter header: no loop filter, no delta adjustments.
        e.encodeBool(0);                       // simple filter flag (arbitrary; level=0 disables filtering anyway)
        e.encodeUint(0, 6);                    // loop_filter_level
        e.encodeUint(0, 3);                    // sharpness
        e.encodeBool(0);                       // use_lf_delta

        // Token partition count: one partition total (log2 = 0).
        e.encodeUint(0, 2);                    // log2_num_token_partitions

        // Quantizer: fixed base QI, no per-component deltas.
        e.encodeUint(DEFAULT_Y_AC_QI, 7);      // base_qi
        e.encodeBool(0);                       // y1_dc_delta_present
        e.encodeBool(0);                       // y2_dc_delta_present
        e.encodeBool(0);                       // y2_ac_delta_present
        e.encodeBool(0);                       // uv_dc_delta_present
        e.encodeBool(0);                       // uv_ac_delta_present

        // refresh_entropy_probs: decoder ignores for keyframes but the bit is required.
        e.encodeBool(0);

        // VP8ParseProba: 4*8*3*11 update bits (all "no update"), then use_skip_proba + skip_p.
        for (int t = 0; t < VP8Tables.NUM_TYPES; t++)
            for (int b = 0; b < VP8Tables.NUM_BANDS; b++)
                for (int c = 0; c < VP8Tables.NUM_CTX; c++)
                    for (int p = 0; p < VP8Tables.NUM_PROBAS; p++)
                        e.encodeBit(VP8Tables.COEFFS_UPDATE_PROBA[t][b][c][p], 0);

        e.encodeBool(1);                       // use_skip_proba
        e.encodeUint(SKIP_PROB, 8);            // skip_p

        // ── Per-macroblock intra-mode decisions (ParseIntraMode) ─────────
        // With segmentation disabled there is no segment bit. We emit:
        //   skip bit (prob = skip_p)
        //   is_i4x4 bit (prob 145) - 1 means "use 16x16 mode"
        //   y-mode tree (probs 156, 163 for DC_PRED) - two 0 bits
        //   uv-mode tree (prob 142 for DC_PRED) - one 0 bit
        for (int mbY = 0; mbY < mbRows; mbY++) {
            for (int mbX = 0; mbX < mbCols; mbX++) {
                e.encodeBit(SKIP_PROB, 1);     // skip = 1 (skip all tokens)
                e.encodeBit(145, 1);           // !is_i4x4 -> 16x16 prediction
                e.encodeBit(156, 0);           // 16x16 y-mode: DC or V
                e.encodeBit(163, 0);           // 16x16 y-mode: DC_PRED
                e.encodeBit(142, 0);           // uv-mode: DC_PRED
            }
        }

        return e.toByteArray();
    }

    /** Token partition is empty - every MB was marked skip=1, so no coefficient tokens follow. */
    private static byte @NotNull [] encodeTokenPartition() {
        return new BooleanEncoder(16).toByteArray();
    }

    /**
     * Assembles the uncompressed 10-byte frame tag, sync code, dimensions, and
     * partition payloads into a complete VP8 keyframe.
     */
    private static byte @NotNull [] assembleFrame(
        int width, int height,
        byte @NotNull [] firstPartition,
        byte @NotNull [] tokenPartition
    ) {
        int firstSize = firstPartition.length;
        byte[] frame = new byte[10 + firstSize + tokenPartition.length];
        int offset = 0;

        // 3-byte frame tag: keyframe=0 | version=0<<1 | show_frame=1<<4 | size<<5
        int frameTag = (1 << 4) | (firstSize << 5);
        frame[offset++] = (byte) (frameTag & 0xFF);
        frame[offset++] = (byte) ((frameTag >>> 8) & 0xFF);
        frame[offset++] = (byte) ((frameTag >>> 16) & 0xFF);

        // 3-byte start code.
        frame[offset++] = (byte) 0x9D;
        frame[offset++] = (byte) 0x01;
        frame[offset++] = (byte) 0x2A;

        // 2-byte width with top 2 bits = horizontal scale (0).
        frame[offset++] = (byte) (width & 0xFF);
        frame[offset++] = (byte) ((width >>> 8) & 0x3F);

        // 2-byte height with top 2 bits = vertical scale (0).
        frame[offset++] = (byte) (height & 0xFF);
        frame[offset++] = (byte) ((height >>> 8) & 0x3F);

        System.arraycopy(firstPartition, 0, frame, offset, firstSize);
        offset += firstSize;
        System.arraycopy(tokenPartition, 0, frame, offset, tokenPartition.length);

        return frame;
    }

}
