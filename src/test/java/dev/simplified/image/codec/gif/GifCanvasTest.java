package dev.simplified.image.codec.gif;

import dev.simplified.image.AnimatedImageData;
import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFactory;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.PixelBuffer;
import dev.simplified.image.StaticImageData;
import dev.simplified.image.exception.ImageDecodeException;
import dev.simplified.image.exception.ImageEncodeException;
import dev.simplified.image.frame.ImageFrame;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.instanceOf;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Pins the canvas {@link GifImageReader} decodes a GIF onto: the logical screen its Logical Screen
 * Descriptor declares, each frame kept at its own size and offset on it, and no longer than 16383
 * on a side - and that the WebP writer refuses such a canvas where its frames do not cover it from
 * the origin, rather than writing a file the WebP reader refuses.
 * <p>
 * Every fixture is a {@link HandBuiltGif}, so it declares a screen and frame offsets no writer here
 * produces. A picture is read as rows of characters: {@code R}, {@code B} and {@code G} for opaque
 * red, blue and green, and {@code .} for a transparent pixel.
 */
class GifCanvasTest {

    private static final int T = 0;
    private static final int R = 1;
    private static final int B = 2;
    private static final int G = 3;

    // ──── an animation ────

    @Test
    @DisplayName("an animation's canvas is its logical screen, not its first frame")
    void screenLargerThanTheFirstFrameIsTheCanvas() {
        ImageData decoded = read(HandBuiltGif.screen(8, 8)
            .frame(0, 0, 4, 4, fill(16, R))
            .frame(4, 4, 4, 4, fill(16, B))
            .bytes());

        assertThat(decoded, instanceOf(AnimatedImageData.class));
        assertThat(decoded.getWidth(), is(8));
        assertThat(decoded.getHeight(), is(8));
    }

    @Test
    @DisplayName("each frame keeps its own size and offset")
    void framesKeepTheirSizesAndOffsets() {
        ImageData decoded = read(HandBuiltGif.screen(8, 8)
            .frame(0, 0, 4, 4, fill(16, R))
            .frame(4, 4, 4, 4, fill(16, B))
            .frame(5, 1, 2, 3, G, T, G, G, T, G)
            .bytes());

        assertThat(decoded.getFrames().stream().map(GifCanvasTest::placement).toList(), contains(
            "4x4 at (0, 0)",
            "4x4 at (4, 4)",
            "2x3 at (5, 1)"
        ));
        assertThat(picture(decoded.getFrames().getLast().pixels()), contains(
            "G.",
            "GG",
            ".G"
        ));
    }

    @Test
    @DisplayName("what a later frame draws outside the first frame's rectangle is on the canvas")
    void laterFramesDrawOutsideTheFirstFrame() {
        ImageData decoded = read(HandBuiltGif.screen(8, 8)
            .frame(0, 0, 4, 4, fill(16, R))
            .frame(4, 4, 4, 4, fill(16, B))
            .frame(5, 1, 2, 3, G, T, G, G, T, G)
            .bytes());

        assertThat(picture(drawn(decoded), decoded.getWidth()), contains(
            "RRRR....",
            "RRRR.G..",
            "RRRR.GG.",
            "RRRR..G.",
            "....BBBB",
            "....BBBB",
            "....BBBB",
            "....BBBB"
        ));
    }

    @Test
    @DisplayName("a frame reaching past the logical screen grows the canvas to hold it")
    void frameReachingPastTheScreenGrowsTheCanvas() {
        ImageData decoded = read(HandBuiltGif.screen(4, 4)
            .frame(0, 0, 4, 4, fill(16, R))
            .frame(2, 2, 4, 4, fill(16, B))
            .bytes());

        assertThat(picture(drawn(decoded), decoded.getWidth()), contains(
            "RRRR..",
            "RRRR..",
            "RRBBBB",
            "RRBBBB",
            "..BBBB",
            "..BBBB"
        ));
    }

    @Test
    @DisplayName("a logical screen declared 0 by 0 is the extent of the frames")
    void zeroScreenIsTheExtentOfTheFrames() {
        ImageData decoded = read(HandBuiltGif.screen(0, 0)
            .frame(1, 2, 2, 2, fill(4, R))
            .frame(0, 0, 2, 2, fill(4, B))
            .bytes());

        assertThat(picture(drawn(decoded), decoded.getWidth()), contains(
            "BB.",
            "BB.",
            ".RR",
            ".RR"
        ));
    }

    // ──── a still ────

    @Test
    @DisplayName("a still whose frame does not span its logical screen is laid on it at its offset")
    void stillIsLaidOnItsScreen() {
        ImageData decoded = read(HandBuiltGif.screen(8, 8)
            .frame(3, 3, 2, 2, R, B, G, T)
            .bytes());

        assertThat(decoded, instanceOf(StaticImageData.class));
        assertThat(picture(decoded.toPixelBuffer()), contains(
            "........",
            "........",
            "........",
            "...RB...",
            "...G....",
            "........",
            "........",
            "........"
        ));
    }

    @Test
    @DisplayName("a still reaching past its logical screen grows the canvas to hold it")
    void stillReachingPastItsScreenGrowsTheCanvas() {
        ImageData decoded = read(HandBuiltGif.screen(2, 2)
            .frame(1, 0, 3, 3, R, R, R, B, B, B, G, G, G)
            .bytes());

        assertThat(picture(decoded.toPixelBuffer()), contains(
            ".RRR",
            ".BBB",
            ".GGG"
        ));
    }

    @Test
    @DisplayName("a still that spans its logical screen from the origin is decoded as it is")
    void stillSpanningItsScreenIsDecodedAsItIs() {
        ImageData decoded = read(HandBuiltGif.screen(2, 2)
            .frame(0, 0, 2, 2, R, B, G, T)
            .bytes());

        assertThat(picture(decoded.toPixelBuffer()), contains(
            "RB",
            "G."
        ));
    }

    // ──── a canvas past the bound ────

    @Test
    @DisplayName("a still whose logical screen is longer than 16383 on a side is refused")
    void stillOnAScreenPastTheBoundIsRefused() {
        byte[] gif = HandBuiltGif.screen(65535, 65535).frame(0, 0, 1, 1, R).bytes();

        ImageDecodeException refused = assertThrows(ImageDecodeException.class, () -> read(gif));
        assertThat(refused.getMessage(), is("GIF canvas '65535x65535' exceeds '16383' on a side"));
    }

    @Test
    @DisplayName("an animation whose logical screen is longer than 16383 on a side is refused")
    void animationOnAScreenPastTheBoundIsRefused() {
        byte[] gif = HandBuiltGif.screen(1, 16384)
            .frame(0, 0, 1, 1, R)
            .frame(0, 0, 1, 1, B)
            .bytes();

        assertThrows(ImageDecodeException.class, () -> read(gif));
    }

    @Test
    @DisplayName("a frame whose offset grows the canvas past 16383 on a side is refused")
    void frameGrowingTheCanvasPastTheBoundIsRefused() {
        byte[] gif = HandBuiltGif.screen(1, 1).frame(16383, 0, 1, 1, R).bytes();

        assertThrows(ImageDecodeException.class, () -> read(gif));
    }

    @Test
    @DisplayName("a canvas 16383 long on a side is decoded")
    void canvasAtTheBoundIsDecoded() {
        ImageData decoded = read(HandBuiltGif.screen(16383, 1).frame(0, 0, 1, 1, R).bytes());

        assertThat(decoded.getWidth(), is(16383));
        assertThat(decoded.getHeight(), is(1));
    }

    // ──── written as WebP ────

    @Test
    @DisplayName("an animation whose first frame is smaller than its screen is refused by the WebP writer")
    void firstFrameSmallerThanTheScreenIsRefusedAsWebP() {
        ImageData decoded = read(HandBuiltGif.screen(8, 8)
            .frame(0, 0, 4, 4, fill(16, R))
            .frame(4, 4, 4, 4, fill(16, B))
            .bytes());

        ImageEncodeException refused = assertThrows(ImageEncodeException.class, () -> ImageFactory.toByteArray(decoded, ImageFormat.WEBP));
        assertThat(refused.getMessage(), is("WebP writer takes only frames that cover the canvas from the origin: frame '0' is '4x4' at '(0, 0)' on a '8x8' canvas"));
    }

    @Test
    @DisplayName("an animation with a later frame smaller than its screen is refused by the WebP writer")
    void laterFrameSmallerThanTheScreenIsRefusedAsWebP() {
        ImageData decoded = read(HandBuiltGif.screen(4, 4)
            .frame(0, 0, 4, 4, fill(16, R))
            .frame(1, 1, 2, 2, fill(4, B))
            .bytes());

        assertThrows(ImageEncodeException.class, () -> ImageFactory.toByteArray(decoded, ImageFormat.WEBP));
    }

    @Test
    @DisplayName("an animation whose frames cover its screen from the origin is written as WebP and reads back")
    void framesCoveringTheScreenAreWrittenAsWebP() {
        ImageData decoded = read(HandBuiltGif.screen(2, 2)
            .frame(0, 0, 2, 2, R, B, G, T)
            .frame(0, 0, 2, 2, G, G, R, B)
            .bytes());

        ImageData webp = ImageFactory.fromByteArray(ImageFactory.toByteArray(decoded, ImageFormat.WEBP));

        assertThat(webp.getFrames().stream().map(GifCanvasTest::placement).toList(), contains(
            "2x2 at (0, 0)",
            "2x2 at (0, 0)"
        ));
        assertThat(picture(webp.getFrames().getLast().pixels()), contains(
            "GG",
            "RB"
        ));
    }

    // ──── fixtures ────

    private static @NotNull ImageData read(byte @NotNull [] gif) {
        return new GifImageReader().read(gif);
    }

    private static int @NotNull [] fill(int count, int index) {
        int[] indices = new int[count];
        Arrays.fill(indices, index);
        return indices;
    }

    private static @NotNull String placement(@NotNull ImageFrame frame) {
        return "%dx%d at (%d, %d)".formatted(frame.width(), frame.height(), frame.offsetX(), frame.offsetY());
    }

    /**
     * Draws every frame of {@code image} in order onto a transparent canvas the image's size, at
     * the frame's offset, a transparent pixel leaving what lies beneath and a pixel past the canvas
     * clipped. Every fixture leaves each frame in place, so this is the picture a player shows after
     * the last frame.
     */
    private static int @NotNull [] drawn(@NotNull ImageData image) {
        int width = image.getWidth();
        int height = image.getHeight();
        int[] canvas = new int[width * height];

        for (ImageFrame frame : image.getFrames()) {
            for (int y = 0; y < frame.height(); y++) {
                for (int x = 0; x < frame.width(); x++) {
                    int argb = frame.pixels().getPixel(x, y);
                    int canvasX = frame.offsetX() + x;
                    int canvasY = frame.offsetY() + y;

                    if ((argb >>> 24) != 0 && canvasX < width && canvasY < height)
                        canvas[canvasY * width + canvasX] = argb;
                }
            }
        }

        return canvas;
    }

    private static @NotNull List<String> picture(@NotNull PixelBuffer pixels) {
        return picture(pixels.data(), pixels.width());
    }

    /**
     * Reads {@code pixels} as rows of characters, {@code ?} standing for a colour no fixture draws.
     */
    private static @NotNull List<String> picture(int @NotNull [] pixels, int width) {
        List<String> rows = new ArrayList<>();

        for (int row = 0; row < pixels.length / width; row++) {
            StringBuilder line = new StringBuilder();

            for (int x = 0; x < width; x++) {
                int argb = pixels[row * width + x];
                line.append((argb >>> 24) == 0 ? '.' : switch (argb) {
                    case 0xFFFF0000 -> 'R';
                    case 0xFF0000FF -> 'B';
                    case 0xFF00FF00 -> 'G';
                    default -> '?';
                });
            }

            rows.add(line.toString());
        }

        return rows;
    }

    /**
     * Builds a GIF89a byte by byte.
     * <p>
     * Every frame indexes one four-entry global colour table - index 0 transparent, then red, blue
     * and green - is left in place by its graphic control extension, and is not interlaced. Its indices are coded as LZW literals only: with a minimum code size of 2 the
     * clear code is 4, the end code 5 and the first free table entry 6, and a decoder adds an entry
     * for every literal after the first since a clear, widening its codes once the next entry would
     * be 8. A clear before every second literal therefore keeps every code three bits wide.
     */
    private static final class HandBuiltGif {

        private static final int[] PALETTE = { 0x000000, 0xFF0000, 0x0000FF, 0x00FF00 };
        private static final int MIN_CODE_SIZE = 2;
        private static final int CLEAR_CODE = 4;
        private static final int END_CODE = 5;
        private static final int CODE_BITS = 3;
        private static final int LITERALS_PER_CLEAR = 2;

        private final ByteArrayOutputStream bytes = new ByteArrayOutputStream();

        private HandBuiltGif(int screenWidth, int screenHeight) {
            this.bytes.writeBytes("GIF89a".getBytes(StandardCharsets.US_ASCII));
            this.u16(screenWidth);
            this.u16(screenHeight);
            this.bytes.write(0xF1); // a global colour table of four entries, eight bits per primary
            this.bytes.write(0); // background colour index
            this.bytes.write(0); // pixel aspect ratio

            for (int rgb : PALETTE) {
                this.bytes.write(rgb >> 16);
                this.bytes.write((rgb >> 8) & 0xFF);
                this.bytes.write(rgb & 0xFF);
            }
        }

        /**
         * Starts a GIF whose Logical Screen Descriptor declares the given size.
         *
         * @param width the logical screen width
         * @param height the logical screen height
         * @return a GIF holding no frame yet
         */
        static @NotNull HandBuiltGif screen(int width, int height) {
            return new HandBuiltGif(width, height);
        }

        /**
         * Appends a frame whose Image Descriptor places it at the given offset.
         *
         * @param left the frame's left edge on the logical screen
         * @param top the frame's top edge on the logical screen
         * @param width the frame width
         * @param height the frame height
         * @param indices the frame's colour table indices, row by row
         * @return this GIF
         * @throws IllegalArgumentException if {@code indices} does not hold {@code width * height} entries
         */
        @NotNull HandBuiltGif frame(int left, int top, int width, int height, int @NotNull ... indices) {
            if (indices.length != width * height)
                throw new IllegalArgumentException("A %dx%d frame holds %d indices, not %d".formatted(width, height, width * height, indices.length));

            // graphic control extension: leave in place, transparent index 0, 10 cs
            this.bytes.writeBytes(new byte[] { 0x21, (byte) 0xF9, 4, (1 << 2) | 1, 10, 0, 0, 0 });

            this.bytes.write(0x2C); // image descriptor
            this.u16(left);
            this.u16(top);
            this.u16(width);
            this.u16(height);
            this.bytes.write(0); // no local colour table, not interlaced

            this.bytes.write(MIN_CODE_SIZE);
            byte[] packed = lzwLiterals(indices);

            for (int start = 0; start < packed.length; start += 255) {
                int length = Math.min(255, packed.length - start);
                this.bytes.write(length);
                this.bytes.write(packed, start, length);
            }

            this.bytes.write(0); // block terminator
            return this;
        }

        /**
         * Returns the GIF built so far, closed by its trailer.
         *
         * @return the encoded GIF
         */
        byte @NotNull [] bytes() {
            byte[] body = this.bytes.toByteArray();
            byte[] gif = Arrays.copyOf(body, body.length + 1);
            gif[body.length] = 0x3B;
            return gif;
        }

        private void u16(int value) {
            this.bytes.write(value & 0xFF);
            this.bytes.write((value >> 8) & 0xFF);
        }

        private static byte @NotNull [] lzwLiterals(int @NotNull [] indices) {
            List<Integer> codes = new ArrayList<>();

            for (int i = 0; i < indices.length; i++) {
                if (i % LITERALS_PER_CLEAR == 0)
                    codes.add(CLEAR_CODE);

                codes.add(indices[i]);
            }

            codes.add(END_CODE);

            ByteArrayOutputStream packed = new ByteArrayOutputStream();
            int buffer = 0;
            int bits = 0;

            for (int code : codes) {
                buffer |= code << bits;
                bits += CODE_BITS;

                while (bits >= 8) {
                    packed.write(buffer & 0xFF);
                    buffer >>>= 8;
                    bits -= 8;
                }
            }

            if (bits > 0)
                packed.write(buffer & 0xFF);

            return packed.toByteArray();
        }

    }

}
