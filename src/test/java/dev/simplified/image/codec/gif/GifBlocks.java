package dev.simplified.image.codec.gif;

import dev.simplified.annotations.UtilityClass;
import org.jetbrains.annotations.NotNull;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.function.IntConsumer;

/**
 * Walks the block structure of an encoded GIF, so a test reads what a writer declared out of the
 * bytes themselves rather than through a reader's interpretation of them.
 */
@UtilityClass
final class GifBlocks {

    /** The application identifier and authentication code of the extension that declares a loop count. */
    private static final String NETSCAPE = "NETSCAPE2.0";

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
     * Returns whether each image descriptor declares its frame interlaced, in file order.
     * <p>
     * The descriptor is laid out {@code 2C [left] [top] [width] [height] [packed]}, each position and
     * size two bytes, and bit {@code 0x40} of its packed field is the interlace flag.
     *
     * @param gif the encoded GIF
     * @return the interlace flag of each image descriptor
     * @throws IllegalStateException if the walk meets a block it does not know
     */
    static @NotNull List<Boolean> interlaceFlags(byte @NotNull [] gif) {
        List<Boolean> flags = new ArrayList<>();
        walk(gif, (label, introducer) -> { }, packed -> flags.add((gif[packed] & 0x40) != 0));
        return flags;
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
        for (int identifier : netscapeIdentifiers(gif)) {
            int position = identifier + 11;
            if ((gif[position] & 0xFF) != 3 || gif[position + 1] != 1)
                throw new IllegalStateException("NETSCAPE2.0 extension at %d holds no loop count".formatted(identifier - 3));

            counts.add((gif[position + 2] & 0xFF) | ((gif[position + 3] & 0xFF) << 8));
        }
        return counts;
    }

    /**
     * Returns a copy of {@code gif} in which every NETSCAPE2.0 application extension carries another
     * application identifier and authentication code, every other byte as it was.
     *
     * @param gif the encoded GIF
     * @param identifier the eleven characters to write in place of {@code NETSCAPE2.0}
     * @return the GIF with each NETSCAPE2.0 extension renamed
     * @throws IllegalArgumentException if {@code identifier} is not eleven characters
     * @throws IllegalStateException if the walk meets a block it does not know
     */
    static byte @NotNull [] withNetscapeRenamed(byte @NotNull [] gif, @NotNull String identifier) {
        byte[] name = identifier.getBytes(StandardCharsets.US_ASCII);
        if (name.length != 11)
            throw new IllegalArgumentException("Identifier '%s' is not eleven characters".formatted(identifier));

        byte[] renamed = gif.clone();
        for (int position : netscapeIdentifiers(gif))
            System.arraycopy(name, 0, renamed, position, name.length);
        return renamed;
    }

    /**
     * Returns a copy of {@code gif} in which the first sub-block of every NETSCAPE2.0 application
     * extension carries another sub-block id, every other byte as it was.
     * <p>
     * Sub-block {@code 1} is the loop count; {@code 2} is the buffering sub-block, whose payload is
     * a buffer size rather than a count.
     *
     * @param gif the encoded GIF
     * @param id the sub-block id to write
     * @return the GIF with each NETSCAPE2.0 extension's first sub-block re-labelled
     * @throws IllegalStateException if the walk meets a block it does not know
     */
    static byte @NotNull [] withNetscapeSubBlockId(byte @NotNull [] gif, int id) {
        byte[] relabelled = gif.clone();
        for (int position : netscapeIdentifiers(gif))
            relabelled[position + 12] = (byte) id;
        return relabelled;
    }

    /**
     * Returns the offset of every NETSCAPE2.0 application extension's eleven-byte identifier, in
     * file order.
     *
     * @param gif the encoded GIF
     * @return the offset of each NETSCAPE2.0 identifier
     * @throws IllegalStateException if the walk meets a block it does not know
     */
    private static @NotNull List<Integer> netscapeIdentifiers(byte @NotNull [] gif) {
        List<Integer> offsets = new ArrayList<>();
        walkExtensions(gif, (label, introducer) -> {
            int position = introducer + 2;

            if (label == 0xFF && (gif[position] & 0xFF) == 11 && NETSCAPE.equals(new String(gif, position + 1, 11, StandardCharsets.US_ASCII)))
                offsets.add(position + 1);
        });
        return offsets;
    }

    /**
     * Walks the block structure of a GIF, handing every extension to a visitor.
     *
     * @param gif the encoded GIF
     * @param visitor what receives each extension's label and the offset of its introducer
     * @throws IllegalStateException if the walk meets a block it does not know
     */
    private static void walkExtensions(byte @NotNull [] gif, @NotNull ExtensionVisitor visitor) {
        walk(gif, visitor, packed -> { });
    }

    /**
     * Walks the block structure of a GIF, handing every extension and every image descriptor to a
     * visitor.
     *
     * @param gif the encoded GIF
     * @param extensions what receives each extension's label and the offset of its introducer
     * @param descriptors what receives the offset of each image descriptor's packed field
     * @throws IllegalStateException if the walk meets a block it does not know
     */
    private static void walk(byte @NotNull [] gif, @NotNull ExtensionVisitor extensions, @NotNull IntConsumer descriptors) {
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
                extensions.visit(label, position - 2);
                position = skipSubBlocks(gif, position);
                continue;
            }

            if (block == 0x2C) { // image descriptor
                position += 8; // left, top, width, height
                descriptors.accept(position);
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
