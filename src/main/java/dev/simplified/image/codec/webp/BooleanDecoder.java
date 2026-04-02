package dev.sbs.api.io.image.codec.webp;

import org.jetbrains.annotations.NotNull;

/**
 * VP8 boolean arithmetic (range) decoder.
 * <p>
 * Decodes bits from a VP8 bitstream where each bit has an associated
 * probability. Used for all VP8 lossy decoding operations.
 */
final class BooleanDecoder {

    private final byte[] data;
    private int offset;
    private int range;
    private int value;
    private int bitsLeft;

    /**
     * Creates a boolean decoder for the given data.
     *
     * @param data the VP8 bitstream bytes
     * @param offset the starting offset
     * @param length the number of available bytes
     */
    BooleanDecoder(byte @NotNull [] data, int offset, int length) {
        this.data = data;
        this.offset = offset;

        // Initialize with first two bytes
        this.value = 0;
        this.bitsLeft = 0;
        this.range = 255;

        // Load initial value
        for (int i = 0; i < 2 && this.offset < offset + length; i++) {
            this.value = (this.value << 8) | (this.data[this.offset++] & 0xFF);
            this.bitsLeft += 8;
        }
        this.bitsLeft -= 8; // VP8 starts with 8 bits consumed
    }

    /**
     * Decodes a single bit with the given probability (0-255).
     *
     * @param probability the probability of the bit being 0 (0-255, with 128 = 50%)
     * @return 0 or 1
     */
    int decodeBit(int probability) {
        int split = 1 + (((range - 1) * probability) >> 8);
        int bigSplit = split << 8;
        int bit;

        if (value < bigSplit) {
            range = split;
            bit = 0;
        } else {
            range -= split;
            value -= bigSplit;
            bit = 1;
        }

        // Renormalize
        while (range < 128) {
            range <<= 1;
            value <<= 1;
            bitsLeft--;

            if (bitsLeft <= 0 && offset < data.length) {
                value |= (data[offset++] & 0xFF);
                bitsLeft += 8;
            }
        }

        return bit;
    }

    /**
     * Decodes a boolean (equal probability).
     *
     * @return 0 or 1
     */
    int decodeBool() {
        return decodeBit(128);
    }

    /**
     * Decodes an unsigned integer of the given bit width.
     *
     * @param bits the number of bits
     * @return the decoded value
     */
    int decodeUint(int bits) {
        int value = 0;
        for (int i = bits - 1; i >= 0; i--)
            value |= decodeBool() << i;
        return value;
    }

    /**
     * Decodes a value from a VP8 probability tree.
     * <p>
     * The tree is a flat array of pairs. Starting at index 0, a bit is decoded
     * using the probability at {@code probs[index >> 1]}. If the resulting
     * tree entry is positive, it is the next index; if zero or negative, its
     * negation is the decoded leaf value.
     *
     * @param tree the tree array (2 entries per internal node)
     * @param probs one probability per internal node
     * @return the decoded leaf value
     */
    int decodeTree(int @NotNull [] tree, int @NotNull [] probs) {
        int i = 0;

        do {
            i = tree[i + decodeBit(probs[i >> 1])];
        } while (i > 0);

        return -i;
    }

    /**
     * Decodes a signed integer of the given bit width.
     *
     * @param bits the number of bits (excluding sign)
     * @return the decoded signed value
     */
    int decodeSint(int bits) {
        int value = decodeUint(bits);
        return decodeBool() != 0 ? -value : value;
    }

}
