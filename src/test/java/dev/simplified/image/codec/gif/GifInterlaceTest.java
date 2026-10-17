package dev.simplified.image.codec.gif;

import dev.simplified.image.AnimatedImageData;
import dev.simplified.image.ImageData;
import dev.simplified.image.PixelBuffer;
import dev.simplified.image.StaticImageData;
import dev.simplified.image.frame.ImageFrame;
import dev.simplified.image.frame.LoopCount;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.hasSize;

/**
 * Pins that {@link GifImageReader} puts every row of an interlaced frame where its image has it, at
 * every height and whoever wrote the frame.
 * <p>
 * An interlaced frame stores its rows in four passes: rows 0, 8, 16 and on, then 4, 12, 20 and on,
 * then 2, 6, 10 and on, then every odd row. A frame of 4 rows or fewer has no row in the second
 * pass, and one of 2 rows none in the third either, which is what a reader has to get right; every
 * height through 8 is read here, so a frame that has a row in every pass is covered too. Each row
 * of a fixture is a colour no other row of its image holds, so a row read into the wrong place,
 * or left unread, shows.
 */
class GifInterlaceTest {

    /** The widths every height is read at. */
    private static final int[] WIDTHS = { 1, 2, 3, 5, 8, 13 };

    /** The tallest frame read. */
    private static final int MAX_HEIGHT = 8;

    /** The frames of each animated fixture. */
    private static final int FRAMES = 3;

    /** The first row of each pass. */
    private static final int[] PASS_START = { 0, 4, 2, 1 };

    /** The rows between one row of a pass and the next. */
    private static final int[] PASS_STEP = { 8, 8, 4, 2 };

    // ──── what this library writes ────

    @Test
    @DisplayName("a still frame this library writes reads back row for row at every height")
    void stillFrameRoundTrips() {
        for (int height = 1; height <= MAX_HEIGHT; height++) {
            for (int width : WIDTHS) {
                PixelBuffer pixels = rows(width, height, 0);
                byte[] gif = new GifImageWriter().write(StaticImageData.of(pixels));

                assertThat(GifBlocks.interlaceFlags(gif), equalTo(List.of(true)));
                assertThat(width + "x" + height, rowsOf(new GifImageReader().read(gif)), equalTo(List.of(rowsOf(pixels))));
            }
        }
    }

    @Test
    @DisplayName("every frame of an animation this library writes reads back row for row at every height")
    void animationRoundTrips() {
        for (int height = 1; height <= MAX_HEIGHT; height++) {
            for (int width : WIDTHS) {
                AnimatedImageData.Builder builder = AnimatedImageData.builder().withLoopCount(LoopCount.Preset.INFINITE);
                List<List<List<Integer>>> expected = new ArrayList<>();

                for (int frame = 0; frame < FRAMES; frame++) {
                    PixelBuffer pixels = rows(width, height, frame);
                    builder.withFrame(ImageFrame.of(pixels, 100));
                    expected.add(rowsOf(pixels));
                }

                byte[] gif = new GifImageWriter().write(builder.build());

                assertThat(GifBlocks.interlaceFlags(gif), equalTo(Collections.nCopies(FRAMES, true)));
                assertThat(width + "x" + height, rowsOf(new GifImageReader().read(gif)), equalTo(expected));
            }
        }
    }

    // ──── what another encoder writes ────

    @Test
    @DisplayName("an interlaced still frame another encoder writes reads row for row at every height")
    void foreignStillFrameReads() {
        for (int height = 1; height <= MAX_HEIGHT; height++) {
            for (int width : WIDTHS) {
                ForeignGif gif = ForeignGif.encode(List.of(new FrameShape(width, height, true)), false);

                assertThat(GifBlocks.interlaceFlags(gif.bytes()), equalTo(List.of(true)));
                assertThat(width + "x" + height, rowsOf(new GifImageReader().read(gif.bytes())), equalTo(gif.frames()));
            }
        }
    }

    @Test
    @DisplayName("an interlaced animation another encoder writes, each frame with its own colour table, reads row for row at every height")
    void foreignAnimationReads() {
        for (int height = 1; height <= MAX_HEIGHT; height++) {
            for (int width : WIDTHS) {
                ForeignGif gif = ForeignGif.encode(Collections.nCopies(FRAMES, new FrameShape(width, height, true)), true);

                assertThat(GifBlocks.interlaceFlags(gif.bytes()), equalTo(Collections.nCopies(FRAMES, true)));
                assertThat(width + "x" + height, rowsOf(new GifImageReader().read(gif.bytes())), equalTo(gif.frames()));
            }
        }
    }

    @Test
    @DisplayName("a frame stored in row order reads in row order at every height")
    void sequentialFrameReads() {
        for (int height = 1; height <= MAX_HEIGHT; height++) {
            for (int width : WIDTHS) {
                ForeignGif gif = ForeignGif.encode(List.of(new FrameShape(width, height, false)), false);

                assertThat(GifBlocks.interlaceFlags(gif.bytes()), equalTo(List.of(false)));
                assertThat(width + "x" + height, rowsOf(new GifImageReader().read(gif.bytes())), equalTo(gif.frames()));
            }
        }
    }

    @Test
    @DisplayName("each frame of an animation mixing heights and interlacing reads as its own descriptor declares")
    void mixedAnimationReads() {
        List<FrameShape> shapes = new ArrayList<>();
        for (int height = 1; height <= MAX_HEIGHT; height++) {
            shapes.add(new FrameShape(5, height, true));
            shapes.add(new FrameShape(3, height, false));
        }

        for (boolean localTables : new boolean[] { false, true }) {
            ForeignGif gif = ForeignGif.encode(shapes, localTables);
            List<List<List<Integer>>> decoded = rowsOf(new GifImageReader().read(gif.bytes()));

            assertThat(GifBlocks.interlaceFlags(gif.bytes()), equalTo(shapes.stream().map(FrameShape::interlaced).toList()));
            assertThat(decoded, hasSize(shapes.size()));
            for (int frame = 0; frame < shapes.size(); frame++)
                assertThat(shapes.get(frame) + " with local tables " + localTables, decoded.get(frame), equalTo(gif.frames().get(frame)));
        }
    }

    // ──── fixtures ────

    /**
     * Builds a frame in which every row is one colour no other row of the image holds, frame
     * {@code frame} of an animation taking colours no other frame does.
     */
    private static @NotNull PixelBuffer rows(int width, int height, int frame) {
        int[] pixels = new int[width * height];
        for (int y = 0; y < height; y++) {
            int argb = 0xFF000000 | ((y + 1) * 0x1F) << 16 | ((frame + 1) * 0x3D) << 8 | 0x55;
            for (int x = 0; x < width; x++)
                pixels[y * width + x] = argb;
        }

        return PixelBuffer.of(pixels, width, height);
    }

    /** Lists the pixels of each row of a frame, top row first. */
    private static @NotNull List<List<Integer>> rowsOf(@NotNull PixelBuffer pixels) {
        int[] data = pixels.data();
        List<List<Integer>> rows = new ArrayList<>();
        for (int y = 0; y < pixels.height(); y++) {
            List<Integer> row = new ArrayList<>();
            for (int x = 0; x < pixels.width(); x++)
                row.add(data[y * pixels.width() + x]);
            rows.add(row);
        }

        return rows;
    }

    /** Lists the rows of each frame of a decoded image, first frame first. */
    private static @NotNull List<List<List<Integer>>> rowsOf(@NotNull ImageData image) {
        List<List<List<Integer>>> frames = new ArrayList<>();
        for (ImageFrame frame : image.getFrames())
            frames.add(rowsOf(frame.pixels()));
        return frames;
    }

    /**
     * Lists the rows of a frame in the order an interlaced frame stores them.
     */
    private static @NotNull List<Integer> interlacedOrder(int height) {
        List<Integer> order = new ArrayList<>();
        for (int pass = 0; pass < PASS_START.length; pass++)
            for (int y = PASS_START[pass]; y < height; y += PASS_STEP[pass])
                order.add(y);
        return order;
    }

    /**
     * The size of one frame of a {@link ForeignGif}, and whether it is stored interlaced.
     *
     * @param width the frame's width
     * @param height the frame's height
     * @param interlaced whether its rows are stored interlaced
     */
    private record FrameShape(int width, int height, boolean interlaced) {
    }

    /**
     * A GIF encoded here rather than through ImageIO, and the rows of each frame it holds.
     * <p>
     * Its LZW data is literal codes only, a clear code coming before the code table would grow to a
     * wider code, so the encoding is plain to check by hand. Each frame sits at the origin of a
     * screen as large as the largest frame, with a graphic control extension and a comment
     * extension before it.
     *
     * @param bytes the encoded GIF
     * @param frames the rows of each frame, as a reader should decode them
     */
    private record ForeignGif(byte @NotNull [] bytes, @NotNull List<List<List<Integer>>> frames) {

        /** The size, as a power of two, of a frame's local colour table. */
        private static final int LOCAL_TABLE_BITS = 3;

        /** The size, as a power of two, of the global colour table. */
        private static final int GLOBAL_TABLE_BITS = 8;

        /**
         * Encodes frames of the given shapes.
         * <p>
         * With one global table of 256 colours, row {@code y} of frame {@code f} is index
         * {@code 1 + y + 8 * f}, so each row's colour is unique to it. With a local table of 8
         * colours per frame, it is index {@code y % 8} of a table no other frame shares, which is
         * unique within every frame of up to 8 rows.
         *
         * @param shapes the size and storage of each frame
         * @param localTables whether each frame carries a local colour table in place of one global
         *     table
         * @return the encoded GIF and the rows it holds
         */
        static @NotNull ForeignGif encode(@NotNull List<FrameShape> shapes, boolean localTables) {
            int screenWidth = shapes.stream().mapToInt(FrameShape::width).max().orElseThrow();
            int screenHeight = shapes.stream().mapToInt(FrameShape::height).max().orElseThrow();
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            List<List<List<Integer>>> frames = new ArrayList<>();

            out.writeBytes("GIF89a".getBytes(StandardCharsets.US_ASCII));
            writeShort(out, screenWidth);
            writeShort(out, screenHeight);
            out.write(localTables ? 0x70 : 0xF0 | (GLOBAL_TABLE_BITS - 1));
            out.write(0); // background colour index
            out.write(0); // pixel aspect ratio
            if (!localTables)
                writeTable(out, GLOBAL_TABLE_BITS, -1);

            if (shapes.size() > 1)
                out.writeBytes(new byte[] { 0x21, (byte) 0xFF, 0x0B, 'N', 'E', 'T', 'S', 'C', 'A', 'P', 'E', '2', '.', '0', 0x03, 0x01, 0x00, 0x00, 0x00 });

            for (int f = 0; f < shapes.size(); f++) {
                FrameShape shape = shapes.get(f);
                int tableBits = localTables ? LOCAL_TABLE_BITS : GLOBAL_TABLE_BITS;
                int[] index = new int[shape.height()];
                for (int y = 0; y < shape.height(); y++)
                    index[y] = localTables ? y % (1 << LOCAL_TABLE_BITS) : 1 + y + 8 * f;

                out.writeBytes(new byte[] { 0x21, (byte) 0xF9, 0x04, 0x04, 0x0A, 0x00, 0x00, 0x00 });
                out.writeBytes(new byte[] { 0x21, (byte) 0xFE, 0x05, 'f', 'r', 'a', 'm', 'e', 0x00 });
                out.write(0x2C);
                writeShort(out, 0);
                writeShort(out, 0);
                writeShort(out, shape.width());
                writeShort(out, shape.height());
                out.write((localTables ? 0x80 | (LOCAL_TABLE_BITS - 1) : 0) | (shape.interlaced() ? 0x40 : 0));
                if (localTables)
                    writeTable(out, LOCAL_TABLE_BITS, f);

                List<Integer> stored = shape.interlaced() ? interlacedOrder(shape.height()) : rowOrder(shape.height());
                int[] codes = new int[shape.width() * shape.height()];
                int at = 0;
                for (int y : stored)
                    for (int x = 0; x < shape.width(); x++)
                        codes[at++] = index[y];

                int minimumCodeSize = Math.max(2, tableBits);
                out.write(minimumCodeSize);
                writeSubBlocks(out, literalCodes(codes, minimumCodeSize));

                List<List<Integer>> rows = new ArrayList<>();
                for (int y = 0; y < shape.height(); y++)
                    rows.add(Collections.nCopies(shape.width(), colour(index[y], localTables ? f : -1)));
                frames.add(rows);
            }

            out.write(0x3B);
            return new ForeignGif(out.toByteArray(), frames);
        }

        /**
         * The colour of entry {@code index} of the global table, or of frame {@code frame}'s local
         * table.
         */
        private static int colour(int index, int frame) {
            return 0xFF000000 | index << 16 | (frame < 0 ? 0xA0 : frame * 0x0B) << 8 | (255 - index);
        }

        private static void writeTable(@NotNull ByteArrayOutputStream out, int bits, int frame) {
            for (int index = 0; index < 1 << bits; index++) {
                int argb = colour(index, frame);
                out.write(argb >> 16 & 0xFF);
                out.write(argb >> 8 & 0xFF);
                out.write(argb & 0xFF);
            }
        }

        private static @NotNull List<Integer> rowOrder(int height) {
            List<Integer> order = new ArrayList<>();
            for (int y = 0; y < height; y++)
                order.add(y);
            return order;
        }

        /**
         * Encodes palette indices as LZW literal codes, least significant bit first.
         * <p>
         * Every code is one bit wider than the minimum code size. A decoder adds a table entry for
         * each code after the first following a clear code, and widens its codes once the table
         * reaches twice the clear code, so a clear code goes out after every
         * {@code clear - 2} literals, before that happens.
         */
        private static byte @NotNull [] literalCodes(int @NotNull [] indices, int minimumCodeSize) {
            int clear = 1 << minimumCodeSize;
            int width = minimumCodeSize + 1;
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            long buffer = 0;
            int bits = 0;
            List<Integer> codes = new ArrayList<>();
            codes.add(clear);
            int sinceClear = 0;
            for (int index : indices) {
                if (sinceClear == clear - 2) {
                    codes.add(clear);
                    sinceClear = 0;
                }
                codes.add(index);
                sinceClear++;
            }
            codes.add(clear + 1); // end of information

            for (int code : codes) {
                buffer |= (long) code << bits;
                bits += width;
                while (bits >= 8) {
                    out.write((int) (buffer & 0xFF));
                    buffer >>>= 8;
                    bits -= 8;
                }
            }
            if (bits > 0)
                out.write((int) (buffer & 0xFF));

            return out.toByteArray();
        }

        private static void writeSubBlocks(@NotNull ByteArrayOutputStream out, byte @NotNull [] data) {
            for (int start = 0; start < data.length; start += 255) {
                int length = Math.min(255, data.length - start);
                out.write(length);
                out.write(data, start, length);
            }
            out.write(0); // block terminator
        }

        private static void writeShort(@NotNull ByteArrayOutputStream out, int value) {
            out.write(value & 0xFF);
            out.write(value >> 8 & 0xFF);
        }

    }

}
