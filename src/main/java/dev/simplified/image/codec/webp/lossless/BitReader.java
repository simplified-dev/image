package dev.simplified.image.codec.webp.lossless;

import dev.simplified.image.exception.ImageDecodeException;
import org.jetbrains.annotations.NotNull;

/**
 * LSB-first bitstream reader with 64-bit buffering for VP8L decoding.
 * <p>
 * Reads bits from a byte array in least-significant-bit-first order,
 * maintaining a 64-bit buffer to minimize per-bit array access.
 */
final class BitReader {

    private final byte @NotNull [] data;
    private final int length;
    private int bytePos;
    private long buffer;
    private int bitsInBuffer;

    /**
     * Creates a bit reader for the given byte array.
     *
     * @param data the source bytes
     */
    BitReader(byte @NotNull [] data) {
        this(data, 0, data.length);
    }

    /**
     * Creates a bit reader for a sub-range of the given byte array.
     *
     * @param data the source bytes
     * @param offset the starting byte offset
     * @param length the number of bytes available
     */
    BitReader(byte @NotNull [] data, int offset, int length) {
        this.data = data;
        this.bytePos = offset;
        this.length = offset + length;
        this.buffer = 0;
        this.bitsInBuffer = 0;
        fillBuffer();
    }

    /**
     * Reads the specified number of bits.
     *
     * @param numBits the number of bits to read (1-32)
     * @return the value read, with bits in LSB order
     * @throws ImageDecodeException if not enough bits remain
     */
    int readBits(int numBits) {
        if (numBits == 0) return 0;

        if (bitsInBuffer < numBits)
            fillBuffer();

        if (bitsInBuffer < numBits)
            throw new ImageDecodeException("Unexpected end of VP8L bitstream");

        int value = (int) (buffer & ((1L << numBits) - 1));
        buffer >>>= numBits;
        bitsInBuffer -= numBits;
        return value;
    }

    /**
     * Reads a single bit.
     *
     * @return 0 or 1
     */
    int readBit() {
        return readBits(1);
    }

    /**
     * Peeks at the next bits without consuming them.
     *
     * @param numBits the number of bits to peek (1-32)
     * @return the peeked value
     */
    int peekBits(int numBits) {
        if (bitsInBuffer < numBits)
            fillBuffer();

        return (int) (buffer & ((1L << numBits) - 1));
    }

    /**
     * Advances past the specified number of bits.
     *
     * @param numBits the number of bits to skip
     */
    void advanceBits(int numBits) {
        buffer >>>= numBits;
        bitsInBuffer -= numBits;
    }

    /**
     * Returns the number of bits remaining in the stream.
     *
     * @return the remaining bit count
     */
    int remainingBits() {
        return bitsInBuffer + (length - bytePos) * 8;
    }

    private void fillBuffer() {
        while (bitsInBuffer <= 56 && bytePos < length) {
            buffer |= ((long) (data[bytePos++] & 0xFF)) << bitsInBuffer;
            bitsInBuffer += 8;
        }
    }

}
