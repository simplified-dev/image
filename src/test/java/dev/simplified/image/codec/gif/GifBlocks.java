package dev.simplified.image.codec.gif;

import org.jetbrains.annotations.NotNull;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * Walks the block structure of an encoded GIF, so a test reads what a writer declared out of the
 * bytes themselves rather than through a reader's interpretation of them.
 */
final class GifBlocks {

    /** The application identifier and authentication code of the extension that declares a loop count. */
    private static final String NETSCAPE = "NETSCAPE2.0";

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
        walkExtensions(gif, (label, introducer) -> {
            if (label == 0xF9)
                offsets.add(introducer);
        });
        return offsets;
    }

    /**
     * Returns the loop count every NETSCAPE2.0 application extension declares, in file order.
     * <p>
     * The extension is laid out {@code 21 FF 0B "NETSCAPE2.0" 03 01 [count lo] [count hi] 00}, the
     * count being the repeats after the first play and {@code 0} repeating forever. A GIF that plays
     * once carries none.
     *
     * @param gif the encoded GIF
     * @return the count each NETSCAPE2.0 extension declares
     * @throws IllegalStateException if the walk meets a block it does not know, or a NETSCAPE2.0
     *     extension whose first sub-block is not a loop count
     */
    static @NotNull List<Integer> netscapeLoopCounts(byte @NotNull [] gif) {
        List<Integer> counts = new ArrayList<>();
        walkExtensions(gif, (label, introducer) -> {
            int position = introducer + 2;

            if (label != 0xFF || (gif[position] & 0xFF) != 11 || !NETSCAPE.equals(new String(gif, position + 1, 11, StandardCharsets.US_ASCII)))
                return;

            position += 12;
            if ((gif[position] & 0xFF) != 3 || gif[position + 1] != 1)
                throw new IllegalStateException("NETSCAPE2.0 extension at %d holds no loop count".formatted(introducer));

            counts.add((gif[position + 2] & 0xFF) | ((gif[position + 3] & 0xFF) << 8));
        });
        return counts;
    }

    /**
     * Walks the block structure of a GIF, handing every extension to a visitor.
     *
     * @param gif the encoded GIF
     * @param visitor what receives each extension's label and the offset of its introducer
     * @throws IllegalStateException if the walk meets a block it does not know
     */
    private static void walkExtensions(byte @NotNull [] gif, @NotNull ExtensionVisitor visitor) {
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
                visitor.visit(label, position - 2);
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
    }

    /**
     * Receives one extension of a GIF being walked.
     */
    @FunctionalInterface
    private interface ExtensionVisitor {

        /**
         * Receives one extension.
         *
         * @param label the extension's label, the byte after its {@code 0x21} introducer
         * @param introducer the offset of its introducer
         */
        void visit(int label, int introducer);

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
