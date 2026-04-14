package dev.simplified.image.codec.webp.lossy;

import org.jetbrains.annotations.NotNull;

/**
 * Inverse of {@link VP8TokenEncoder}: reads a 4x4 block of quantized DCT
 * coefficients back out of a VP8 token stream.
 * <p>
 * Intended both for round-tripping our own encoder's output and for a future
 * spec-compliant {@link VP8Decoder}. Mirrors the branching of libwebp's
 * {@code GetCoeffsFast} in {@code src/dec/vp8_dec.c}.
 */
final class VP8TokenDecoder {

    private VP8TokenDecoder() { }

    /**
     * Decodes a 4x4 block's coefficients from the bit stream.
     *
     * @param dec boolean decoder to read from
     * @param out destination array; zig-zag positions 0..15 are written
     * @param first index of the first coefficient expected in the stream (see
     *              {@link VP8TokenEncoder#emit})
     * @param ctx0 initial surrounding-nonzero context
     * @param coeffType coefficient type; see constants on {@link VP8Tables}
     * @param probas probability table; typically {@link VP8Tables#COEFFS_PROBA_0}
     * @return {@code 1} if any non-zero coefficient was decoded, {@code 0} otherwise
     */
    static int decode(
        @NotNull BooleanDecoder dec,
        short @NotNull [] out,
        int first,
        int ctx0,
        int coeffType,
        int @NotNull [] @NotNull [] @NotNull [] @NotNull [] probas
    ) {
        // Caller provides a cleared buffer for the AC path so positions below `first`
        // keep their existing value; we zero the range we cover.
        for (int i = first; i < 16; i++) out[i] = 0;

        int n = first;
        int[] p = probas[coeffType][n][ctx0];

        // EOB check at the first position.
        if (dec.decodeBit(p[0]) == 0) return 0;

        while (n < 16) {
            // "is zero" bit.
            if (dec.decodeBit(p[1]) == 0) {
                out[n++] = 0;
                if (n == 16) return 1;
                p = probas[coeffType][VP8Tables.COEF_BANDS[n]][0];
                continue;
            }

            int v;
            if (dec.decodeBit(p[2]) == 0) {
                v = 1;
                p = probas[coeffType][VP8Tables.COEF_BANDS[n + 1]][1];
            } else {
                if (dec.decodeBit(p[3]) == 0) {
                    // Small: 2..4.
                    if (dec.decodeBit(p[4]) == 0) {
                        v = 2;
                    } else {
                        v = (dec.decodeBit(p[5]) == 0) ? 3 : 4;
                    }
                } else if (dec.decodeBit(p[6]) == 0) {
                    // Mid: 5..10.
                    if (dec.decodeBit(p[7]) == 0) {
                        v = (dec.decodeBit(159) == 0) ? 5 : 6;
                    } else {
                        int hi = dec.decodeBit(165);         // 0 => 7 or 8, 1 => 9 or 10
                        int even = dec.decodeBit(145);        // 1 => even
                        // Decode: (hi, even) -> value
                        //   (0, 0) -> 7, (0, 1) -> 8, (1, 0) -> 9, (1, 1) -> 10
                        v = 7 + 2 * hi + even;
                    }
                } else {
                    v = decodeCategory(dec, p);
                }
                p = probas[coeffType][VP8Tables.COEF_BANDS[n + 1]][2];
            }

            int sign = dec.decodeBit(128);
            out[n++] = (short) (sign != 0 ? -v : v);

            if (n == 16) return 1;

            // EOB at the next position.
            if (dec.decodeBit(p[0]) == 0) return 1;
        }

        return 1;
    }

    /** Reads category flags plus extra bits for a coefficient magnitude of 11 or greater. */
    private static int decodeCategory(@NotNull BooleanDecoder dec, int @NotNull [] p) {
        int base;
        int bits;
        int[] tab;
        if (dec.decodeBit(p[8]) == 0) {
            if (dec.decodeBit(p[9]) == 0) {           // Cat3
                base = 3 + (8 << 0);
                bits = 3;
                tab = VP8Tables.CAT3;
            } else {                                   // Cat4
                base = 3 + (8 << 1);
                bits = 4;
                tab = VP8Tables.CAT4;
            }
        } else {
            if (dec.decodeBit(p[10]) == 0) {          // Cat5
                base = 3 + (8 << 2);
                bits = 5;
                tab = VP8Tables.CAT5;
            } else {                                   // Cat6
                base = 3 + (8 << 3);
                bits = 11;
                tab = VP8Tables.CAT6;
            }
        }

        int v = 0;
        for (int i = 0; i < bits; i++)
            v = (v << 1) | dec.decodeBit(tab[i]);
        return base + v;
    }

}
