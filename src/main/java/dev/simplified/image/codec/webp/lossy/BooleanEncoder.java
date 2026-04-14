package dev.simplified.image.codec.webp.lossy;

import org.jetbrains.annotations.NotNull;

/**
 * VP8 boolean arithmetic (range) encoder.
 * <p>
 * Literal port of the reference implementation in RFC 6386 section 7.3.
 * Encodes bits into a VP8 bitstream where each bit has an associated 8-bit
 * probability; dual to {@link BooleanDecoder}.
 * <p>
 * The algorithm maintains two pieces of state:
 * <ul>
 *   <li>{@code range} - the size of the current probability interval, kept in
 *     {@code [128, 255]} via renormalization shifts.</li>
 *   <li>{@code bottom} - the interval's low endpoint, held as a 32-bit
 *     accumulator; bits shift off the top into the output buffer one byte at
 *     a time.</li>
 * </ul>
 * <p>
 * Carries produced by interval updates that cross a power-of-two boundary are
 * propagated back through already-emitted bytes via {@link #addOneToOutput}.
 *
 * @see <a href="https://datatracker.ietf.org/doc/html/rfc6386#section-7.3">RFC 6386 section 7.3</a>
 */
final class BooleanEncoder {

    private byte[] data;
    private int bytePos;
    private int range;     // 128 <= range <= 255
    private int bottom;    // 32-bit accumulator (treated as unsigned)
    private int bitCount;  // # of shifts before the next output byte is available

    /**
     * Creates a boolean encoder with the specified initial capacity.
     *
     * @param initialCapacity initial byte array capacity (at least 16)
     */
    BooleanEncoder(int initialCapacity) {
        this.data = new byte[Math.max(16, initialCapacity)];
        this.bytePos = 0;
        this.range = 255;
        this.bottom = 0;
        this.bitCount = 24;
    }

    /**
     * Encodes a single bit with the given probability.
     *
     * @param probability probability of the bit being 0 (0-255)
     * @param bit 0 or 1
     */
    void encodeBit(int probability, int bit) {
        int split = 1 + (((range - 1) * probability) >>> 8);

        if (bit != 0) {
            bottom += split;
            range -= split;
        } else {
            range = split;
        }

        // Renormalize one bit at a time so the bit-31 carry check is exact.
        while (range < 128) {
            range <<= 1;

            if ((bottom >>> 31) != 0)
                addOneToOutput();

            bottom <<= 1;

            if (--bitCount == 0) {
                ensureCapacity(1);
                data[bytePos++] = (byte) ((bottom >>> 24) & 0xFF);
                bottom &= 0xFFFFFF;
                bitCount = 8;
            }
        }
    }

    /**
     * Encodes a boolean at equal probability (p=128).
     *
     * @param bit 0 or 1
     */
    void encodeBool(int bit) {
        encodeBit(128, bit);
    }

    /**
     * Encodes an unsigned integer of the given bit width, most significant bit first.
     *
     * @param value value to encode
     * @param bits number of bits
     */
    void encodeUint(int value, int bits) {
        for (int i = bits - 1; i >= 0; i--)
            encodeBool((value >>> i) & 1);
    }

    /**
     * Encodes a signed integer as magnitude bits followed by a sign bit.
     *
     * @param value signed value to encode
     * @param bits number of magnitude bits (excluding sign)
     */
    void encodeSint(int value, int bits) {
        int abs = Math.abs(value);
        encodeUint(abs, bits);
        encodeBool(value < 0 ? 1 : 0);
    }

    /**
     * Flushes remaining state and returns the encoded bytes.
     * <p>
     * Implements {@code flush_bool_encoder} from RFC 6386 section 7.3: emits
     * a final carry if pending, then pads with trailing bits so the decoder's
     * two-byte read-ahead is satisfied.
     *
     * @return the encoded byte array
     */
    byte @NotNull [] toByteArray() {
        int c = bitCount;
        int v = bottom;

        if ((v & (1 << (32 - c))) != 0)
            addOneToOutput();

        v <<= c & 7;
        c >>= 3;
        while (--c >= 0)
            v <<= 8;

        ensureCapacity(4);
        c = 4;
        while (--c >= 0) {
            data[bytePos++] = (byte) ((v >>> 24) & 0xFF);
            v <<= 8;
        }

        byte[] result = new byte[bytePos];
        System.arraycopy(data, 0, result, 0, bytePos);
        return result;
    }

    /**
     * Propagates a carry backward through already-written output bytes.
     * <p>
     * Equivalent to RFC 6386 section 7.3 {@code add_one_to_output}: walks back
     * over a run of {@code 0xFF} bytes converting each to {@code 0x00}, then
     * increments the first non-{@code 0xFF} byte encountered.
     */
    private void addOneToOutput() {
        int pos = bytePos - 1;
        while (pos >= 0 && data[pos] == (byte) 0xFF) {
            data[pos] = 0;
            pos--;
        }
        if (pos >= 0) data[pos]++;
    }

    private void ensureCapacity(int additional) {
        if (bytePos + additional > data.length) {
            byte[] newData = new byte[Math.max(data.length * 2, bytePos + additional)];
            System.arraycopy(data, 0, newData, 0, bytePos);
            data = newData;
        }
    }

}
