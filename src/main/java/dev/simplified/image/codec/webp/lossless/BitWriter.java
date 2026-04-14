package dev.simplified.image.codec.webp.lossless;

import org.jetbrains.annotations.NotNull;

/**
 * LSB-first bitstream writer with growable buffer for VP8L encoding.
 * <p>
 * Accumulates bits in a 64-bit buffer and flushes to a growable byte array
 * in least-significant-bit-first order.
 */
final class BitWriter {

    private byte[] data;
    private int bytePos;
    private long buffer;
    private int bitsInBuffer;

    /**
     * Creates a bit writer with the specified initial capacity.
     *
     * @param initialCapacity the initial byte array capacity
     */
    BitWriter(int initialCapacity) {
        this.data = new byte[Math.max(16, initialCapacity)];
        this.bytePos = 0;
        this.buffer = 0;
        this.bitsInBuffer = 0;
    }

    /**
     * Creates a bit writer with a default initial capacity.
     */
    BitWriter() {
        this(1024);
    }

    /**
     * Writes the specified number of bits.
     *
     * @param value the value to write (lower numBits bits are used)
     * @param numBits the number of bits to write (0-32)
     */
    void writeBits(int value, int numBits) {
        if (numBits == 0) return;

        buffer |= ((long) value & ((1L << numBits) - 1)) << bitsInBuffer;
        bitsInBuffer += numBits;

        while (bitsInBuffer >= 8) {
            ensureCapacity(1);
            data[bytePos++] = (byte) (buffer & 0xFF);
            buffer >>>= 8;
            bitsInBuffer -= 8;
        }
    }

    /**
     * Writes a single bit.
     *
     * @param bit 0 or 1
     */
    void writeBit(int bit) {
        writeBits(bit, 1);
    }

    /**
     * Flushes any remaining bits in the buffer (zero-padded to byte boundary).
     */
    void flush() {
        if (bitsInBuffer > 0) {
            ensureCapacity(1);
            data[bytePos++] = (byte) (buffer & 0xFF);
            buffer = 0;
            bitsInBuffer = 0;
        }
    }

    /**
     * Returns the accumulated bytes, flushing any remaining bits first.
     *
     * @return a new byte array containing the written data
     */
    byte @NotNull [] toByteArray() {
        flush();
        byte[] result = new byte[bytePos];
        System.arraycopy(data, 0, result, 0, bytePos);
        return result;
    }

    /**
     * Returns the current number of bytes written (excluding unflushed bits).
     *
     * @return the byte count
     */
    int size() {
        return bytePos;
    }

    private void ensureCapacity(int additional) {
        if (bytePos + additional > data.length) {
            int newCapacity = Math.max(data.length * 2, bytePos + additional);
            byte[] newData = new byte[newCapacity];
            System.arraycopy(data, 0, newData, 0, bytePos);
            data = newData;
        }
    }

}
