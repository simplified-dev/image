package dev.simplified.image.codec.webp;

import org.jetbrains.annotations.NotNull;

/**
 * Emits a 4x4 block of quantized DCT coefficients through the VP8 token tree.
 * <p>
 * Coefficients are assumed to be in zig-zag scan order (use {@link VP8Tables#ZIGZAG}
 * to reorder from raster). The emitted token sequence follows RFC 6386 paragraph 13,
 * which is realised here as a line-for-line port of libwebp's {@code PutCoeffs}
 * (in {@code src/enc/frame_enc.c}).
 * <p>
 * Each coefficient position consults the probability triple
 * {@code probas[coeffType][COEF_BANDS[n]][ctx]} where {@code ctx} is the "surrounding
 * non-zero" context carried forward from the previous coefficient:
 * <ul>
 *   <li>{@code 0} - previous coefficient was zero (or this is the first coefficient
 *       and the caller passed {@code 0} as the above/left context sum)</li>
 *   <li>{@code 1} - previous coefficient had magnitude 1</li>
 *   <li>{@code 2} - previous coefficient had magnitude greater than 1</li>
 * </ul>
 * The returned value (0 if the block is empty, 1 otherwise) is suitable for
 * carrying into the next block's context computation.
 */
final class VP8TokenEncoder {

    private VP8TokenEncoder() { }

    /**
     * Emits a single 4x4 block's coefficients through the token tree.
     *
     * @param enc boolean encoder to write into
     * @param coeffs 16 quantized coefficients in zig-zag order
     * @param first index of the first coefficient to emit (0 for most types, 1 for
     *              {@link VP8Tables#TYPE_I16_AC} since DC is carried by the Y2 block)
     * @param ctx0 initial surrounding-nonzero context, typically
     *             {@code above_nonzero + left_nonzero} (0, 1, or 2)
     * @param coeffType one of {@code TYPE_I16_AC}, {@code TYPE_I16_DC},
     *                  {@code TYPE_CHROMA_A}, {@code TYPE_I4_AC}
     * @param probas the {@code [NUM_TYPES][NUM_BANDS][NUM_CTX][NUM_PROBAS]} probability
     *               table - typically {@link VP8Tables#COEFFS_PROBA_0} for a simple
     *               encoder that never updates token probabilities
     * @return {@code 1} if any non-zero coefficient was emitted, {@code 0} otherwise
     */
    static int emit(
        @NotNull BooleanEncoder enc,
        short @NotNull [] coeffs,
        int first,
        int ctx0,
        int coeffType,
        int @NotNull [] @NotNull [] @NotNull [] @NotNull [] probas
    ) {
        // Locate the last non-zero coefficient (or -1 if all zero).
        int last = -1;
        for (int i = 15; i >= first; i--) {
            if (coeffs[i] != 0) { last = i; break; }
        }

        int n = first;
        // For n=0 or 1, probas[type][COEF_BANDS[n]] is equivalent to probas[type][n]
        // since COEF_BANDS[0]=0 and COEF_BANDS[1]=1. libwebp exploits this; we follow.
        int[] p = probas[coeffType][n][ctx0];

        // Initial "not EOB" bit. If the whole block is zero we emit 0 and bail.
        if (last < 0) {
            enc.encodeBit(p[0], 0);
            return 0;
        }
        enc.encodeBit(p[0], 1);

        while (n < 16) {
            int c = coeffs[n++];
            int sign = c < 0 ? 1 : 0;
            int v = sign != 0 ? -c : c;

            // "is zero" bit. If zero, reset context to 0 and skip to the next position
            // (no EOB check here - EOB is only queried after a non-zero coefficient).
            if (v == 0) {
                enc.encodeBit(p[1], 0);
                p = probas[coeffType][VP8Tables.COEF_BANDS[n]][0];
                continue;
            }
            enc.encodeBit(p[1], 1);

            // Magnitude tree.
            if (v == 1) {
                enc.encodeBit(p[2], 0);
                p = probas[coeffType][VP8Tables.COEF_BANDS[n]][1];
            } else {
                enc.encodeBit(p[2], 1);
                if (v <= 4) {
                    // Small values 2..4.
                    enc.encodeBit(p[3], 0);
                    if (v == 2) {
                        enc.encodeBit(p[4], 0);
                    } else {
                        enc.encodeBit(p[4], 1);
                        enc.encodeBit(p[5], v == 4 ? 1 : 0);
                    }
                } else if (v <= 10) {
                    // Mid values 5..10.
                    enc.encodeBit(p[3], 1);
                    enc.encodeBit(p[6], 0);
                    if (v <= 6) {
                        enc.encodeBit(p[7], 0);
                        enc.encodeBit(159, v == 6 ? 1 : 0);
                    } else {
                        enc.encodeBit(p[7], 1);
                        enc.encodeBit(165, v >= 9 ? 1 : 0);
                        enc.encodeBit(145, (v & 1) == 0 ? 1 : 0);
                    }
                } else {
                    // Large values 11..2048 via CAT3/4/5/6 extra bits.
                    enc.encodeBit(p[3], 1);
                    enc.encodeBit(p[6], 1);
                    emitCategory(enc, p, v);
                }
                p = probas[coeffType][VP8Tables.COEF_BANDS[n]][2];
            }

            // Sign bit (uniform).
            enc.encodeBit(128, sign);

            if (n == 16) return 1;

            // EOB bit at the next position.
            if (n > last) {
                enc.encodeBit(p[0], 0);
                return 1;
            }
            enc.encodeBit(p[0], 1);
        }

        return 1;
    }

    /**
     * Emits the category flags (p[8], p[9] or p[8], p[10]) plus the variable-length
     * extra bits that identify a coefficient magnitude of 11 or greater.
     */
    private static void emitCategory(@NotNull BooleanEncoder enc, int @NotNull [] p, int v) {
        int residue;
        int mask;
        int[] tab;
        if (v < 3 + (8 << 1)) {                // Cat3: v in 11..18, 3 extra bits
            enc.encodeBit(p[8], 0);
            enc.encodeBit(p[9], 0);
            residue = v - (3 + (8 << 0));
            mask = 1 << 2;
            tab = VP8Tables.CAT3;
        } else if (v < 3 + (8 << 2)) {         // Cat4: v in 19..34, 4 extra bits
            enc.encodeBit(p[8], 0);
            enc.encodeBit(p[9], 1);
            residue = v - (3 + (8 << 1));
            mask = 1 << 3;
            tab = VP8Tables.CAT4;
        } else if (v < 3 + (8 << 3)) {         // Cat5: v in 35..66, 5 extra bits
            enc.encodeBit(p[8], 1);
            enc.encodeBit(p[10], 0);
            residue = v - (3 + (8 << 2));
            mask = 1 << 4;
            tab = VP8Tables.CAT5;
        } else {                               // Cat6: v in 67..2048, 11 extra bits
            enc.encodeBit(p[8], 1);
            enc.encodeBit(p[10], 1);
            residue = v - (3 + (8 << 3));
            mask = 1 << 10;
            tab = VP8Tables.CAT6;
        }

        int i = 0;
        while (mask != 0) {
            enc.encodeBit(tab[i++], (residue & mask) != 0 ? 1 : 0);
            mask >>>= 1;
        }
    }

}
