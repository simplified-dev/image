package dev.simplified.image.codec.gif;

import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.List;

/**
 * Walks the block structure of an encoded GIF, so a test reads what a writer declared out of the
 * bytes themselves rather than through a reader's interpretation of them.
 */
final class GifBlocks {

    private GifBlocks() {
    }

    /**
     * Returns the byte offset of every graphic control extension's introducer, {@code 0x21}, in file
     * order.
     * <p>
     * The extension is laid out {@code 21 F9 04 [packed] [delay lo] [delay hi] [transparent index] 00}.
     * The block structure is walked rather than scanned for that signature, which could match the
     * same bytes inside compressed image data.
     *
     * @param gif the encoded GIF
     * @return the offset of each graphic control extension
     * @throws IllegalStateException if the walk meets a block it does not know
     */
    static @NotNull List<Integer> graphicControlOffsets(byte @NotNull [] gif) {
        List<Integer> offsets = new ArrayList<>();
        int position = 6; // header: "GIF89a"

        int screenPacked = gif[position + 4] & 0xFF;
        position += 7; // logical screen descriptor
        if ((screenPacked & 0x80) != 0)
            position += colorTableBytes(screenPacked);

        while (position < gif.length) {
            int block = gif[position++] & 0xFF;

            if (block == 0x3B) // trailer
                break;

            if (block == 0x21) { // extension
                int label = gif[position++] & 0xFF;

                if (label == 0xF9)
                    offsets.add(position - 2);

                position = skipSubBlocks(gif, position);
                continue;
            }

            if (block == 0x2C) { // image descriptor
                position += 8; // left, top, width, height
                int imagePacked = gif[position++] & 0xFF;
                if ((imagePacked & 0x80) != 0)
                    position += colorTableBytes(imagePacked);

                position++; // LZW minimum code size
                position = skipSubBlocks(gif, position);
                continue;
            }

            throw new IllegalStateException("Unexpected GIF block 0x%02X at %d".formatted(block, position - 1));
        }

        return offsets;
    }

    private static int colorTableBytes(int packed) {
        return 3 * (1 << ((packed & 0x07) + 1));
    }

    private static int skipSubBlocks(byte @NotNull [] gif, int position) {
        while (true) {
            int length = gif[position++] & 0xFF;
            if (length == 0) return position;
            position += length;
        }
    }

}
