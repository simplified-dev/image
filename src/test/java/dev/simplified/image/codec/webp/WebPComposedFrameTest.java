package dev.simplified.image.codec.webp;

import dev.simplified.collection.Concurrent;
import dev.simplified.collection.ConcurrentList;
import dev.simplified.image.AnimatedImageData;
import dev.simplified.image.ImageData;
import dev.simplified.image.PixelBuffer;
import dev.simplified.image.codec.webp.vp8l.VP8LEncoder;
import dev.simplified.image.frame.FrameBlend;
import dev.simplified.image.frame.FrameDisposal;
import dev.simplified.image.frame.ImageFrame;
import dev.simplified.image.option.WebPWriteOptions;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.instanceOf;

/**
 * Pins the disposal and blend {@link WebPImageReader} gives the frames of an animated WebP, each of
 * which it has already composed onto the whole canvas.
 * <p>
 * A composed frame is the picture the canvas shows once the frame has been drawn, so it replaces the
 * picture before it outright: a player that honours its flags, or a writer that declares them, must
 * show its pixels as they are rather than composite them a second time.
 * <p>
 * The source is built chunk by chunk, because its frames are lossless partial rectangles whose blend
 * and disposal decide what the canvas shows, and {@link WebPImageWriter} declares a frame's own flags
 * only on full-canvas lossy frames.
 */
class WebPComposedFrameTest {

    private static final int HALF_RED = 0x80FF0000;
    private static final int BLUE = 0xFF0000FF;
    private static final int GREEN = 0xFF00FF00;
    private static final int CLEAR = 0x00000000;

    /** The side of the source's canvas. */
    private static final int SIZE = 16;

    /** The {@code ANMF} flag bit that restores a frame's rectangle to the background once it has shown. */
    private static final int DISPOSE_TO_BACKGROUND = 0x01;

    /** The {@code ANMF} flag bit that writes a frame over its rectangle rather than blending it there. */
    private static final int NO_BLEND = 0x02;

    // ──── the decoded frames ────

    @Test
    @DisplayName("each decoded frame is the canvas the source's blend and disposal draw")
    void framesAreTheComposedCanvases() {
        assertPictures(pictures(read(source())), composed());
    }

    @Test
    @DisplayName("every decoded frame is a whole canvas that replaces the one before it")
    void decodedFramesReplaceTheCanvas() {
        AnimatedImageData decoded = read(source());

        assertThat(decoded.getFrames(), hasSize(4));
        for (ImageFrame frame : decoded.getFrames()) {
            assertThat(frame.offsetX(), equalTo(0));
            assertThat(frame.offsetY(), equalTo(0));
            assertThat(frame.width(), equalTo(SIZE));
            assertThat(frame.height(), equalTo(SIZE));
            assertThat(frame.blend(), equalTo(FrameBlend.SOURCE));
            assertThat(frame.disposal(), equalTo(FrameDisposal.RESTORE_TO_BACKGROUND));
        }
    }

    @Test
    @DisplayName("a player that honours the decoded flags shows each frame's own pixels")
    void playerShowsEachFramesOwnPixels() {
        AnimatedImageData decoded = read(source());

        assertPictures(play(decoded), pictures(decoded));
    }

    // ──── written again ────

    @Test
    @DisplayName("a decoded animation written again losslessly decodes to every frame's pixels")
    void losslessRewriteReproducesEveryFrame() {
        AnimatedImageData rewritten = read(new WebPImageWriter().write(read(source())));

        assertPictures(pictures(rewritten), composed());
    }

    @Test
    @DisplayName("a decoded animation written again with P-frames, which declare each frame's flags, decodes to every frame's alpha")
    void pFrameRewriteReproducesEveryAlpha() {
        WebPWriteOptions options = WebPWriteOptions.builder().isLossless(false).withQuality(1.0f).usePFrames(true).build();
        AnimatedImageData rewritten = read(new WebPImageWriter().write(read(source()), options));

        // VP8 colour is lossy and the alpha plane is not, so the alpha is what shows whether a frame
        // was laid over the one before it a second time.
        assertPictures(alphas(pictures(rewritten)), alphas(composed()));
    }

    // ──── fixtures ────

    /**
     * An animated WebP on a transparent background, playing forever, whose four lossless frames rely
     * on their blend and disposal:
     * <ul>
     *     <li>half-transparent red, blended over the whole canvas</li>
     *     <li>an 8x8 rectangle at the origin, blue on its left half and transparent on its right,
     *     blended so the red shows through its right half, and restored to the background once it
     *     has shown</li>
     *     <li>opaque green written over the 8x8 rectangle at (8, 8)</li>
     *     <li>transparency written over the 4x4 rectangle at (12, 0), clearing the red there</li>
     * </ul>
     */
    private static byte @NotNull [] source() {
        int[] blueLeft = new int[8 * 8];
        for (int i = 0; i < blueLeft.length; i++)
            blueLeft[i] = i % 8 < 4 ? BLUE : CLEAR;

        // VP8X: the animation and alpha flags, then the canvas's width and height less one.
        byte[] vp8x = new byte[10];
        vp8x[0] = 0x02 | 0x10;
        vp8x[4] = SIZE - 1;
        vp8x[7] = SIZE - 1;

        // ANIM: a transparent background, then 0 plays, which plays forever.
        ConcurrentList<WebPChunk> chunks = Concurrent.newList();
        chunks.add(RiffContainer.createChunk(WebPChunk.Type.VP8X, vp8x));
        chunks.add(RiffContainer.createChunk(WebPChunk.Type.ANIM, new byte[6]));
        chunks.add(anmf(0, 0, filled(SIZE, HALF_RED), 0));
        chunks.add(anmf(0, 0, PixelBuffer.of(blueLeft, 8, 8), DISPOSE_TO_BACKGROUND));
        chunks.add(anmf(8, 8, filled(8, GREEN), NO_BLEND));
        chunks.add(anmf(12, 0, filled(4, CLEAR), NO_BLEND));
        return RiffContainer.write(chunks);
    }

    /**
     * The canvas the source shows after each of its frames.
     */
    private static @NotNull List<int[]> composed() {
        int[] first = paint(new int[SIZE * SIZE], 0, 0, SIZE, SIZE, HALF_RED);
        int[] second = paint(first, 0, 0, 4, 8, BLUE);
        int[] third = paint(paint(first, 0, 0, 8, 8, CLEAR), 8, 8, 8, 8, GREEN);
        int[] fourth = paint(third, 12, 0, 4, 4, CLEAR);
        return List.of(first, second, third, fourth);
    }

    /**
     * An {@code ANMF} chunk drawing {@code pixels} at ({@code x}, {@code y}) as a VP8L frame, under
     * {@code flags}.
     */
    private static @NotNull WebPChunk anmf(int x, int y, @NotNull PixelBuffer pixels, int flags) {
        byte[] bitstream = VP8LEncoder.encode(pixels);
        byte[] payload = new byte[16 + 8 + bitstream.length + (bitstream.length & 1)];
        putLittleEndian(payload, 0, 3, x / 2);
        putLittleEndian(payload, 3, 3, y / 2);
        putLittleEndian(payload, 6, 3, pixels.width() - 1);
        putLittleEndian(payload, 9, 3, pixels.height() - 1);
        putLittleEndian(payload, 12, 3, 100);
        payload[15] = (byte) flags;
        System.arraycopy("VP8L".getBytes(StandardCharsets.US_ASCII), 0, payload, 16, 4);
        putLittleEndian(payload, 20, 4, bitstream.length);
        System.arraycopy(bitstream, 0, payload, 24, bitstream.length);
        return RiffContainer.createChunk(WebPChunk.Type.ANMF, payload);
    }

    /**
     * Writes the low {@code bytes} bytes of {@code value} into {@code data} at {@code offset}, low
     * byte first.
     */
    private static void putLittleEndian(byte @NotNull [] data, int offset, int bytes, int value) {
        for (int i = 0; i < bytes; i++)
            data[offset + i] = (byte) ((value >> (8 * i)) & 0xFF);
    }

    private static @NotNull PixelBuffer filled(int side, int argb) {
        int[] pixels = new int[side * side];
        Arrays.fill(pixels, argb);
        return PixelBuffer.of(pixels, side, side);
    }

    /**
     * Returns a copy of the {@link #SIZE}-wide {@code picture} with the {@code width} by
     * {@code height} rectangle at ({@code x}, {@code y}) filled with {@code argb}.
     */
    private static int @NotNull [] paint(int @NotNull [] picture, int x, int y, int width, int height, int argb) {
        int[] painted = picture.clone();

        for (int row = y; row < y + height; row++)
            Arrays.fill(painted, row * SIZE + x, row * SIZE + x + width, argb);

        return painted;
    }

    // ──── playing ────

    /**
     * Plays {@code animation} as a WebP player does, honouring each frame's flags, and returns the
     * canvas shown after each frame.
     * <p>
     * The canvas starts as the background colour. Before a frame is drawn, the frame before it is
     * disposed: restoring to the background clears that frame's rectangle, and doing nothing leaves
     * it. The frame is then drawn at its offset, written over the canvas when it blends
     * {@link FrameBlend#SOURCE} and laid over it when it blends {@link FrameBlend#OVER}.
     *
     * @throws IllegalStateException if a frame restores to the previous canvas, which WebP does not
     *     declare
     */
    private static @NotNull List<int[]> play(@NotNull AnimatedImageData animation) {
        int width = animation.getWidth();
        int[] canvas = new int[width * animation.getHeight()];
        Arrays.fill(canvas, animation.getBackgroundColor());
        List<int[]> shown = new ArrayList<>();
        ImageFrame prior = null;

        for (ImageFrame frame : animation.getFrames()) {
            if (prior != null) {
                switch (prior.disposal()) {
                    case RESTORE_TO_BACKGROUND -> {
                        for (int row = 0; row < prior.height(); row++) {
                            int start = (prior.offsetY() + row) * width + prior.offsetX();
                            Arrays.fill(canvas, start, start + prior.width(), animation.getBackgroundColor());
                        }
                    }
                    case RESTORE_TO_PREVIOUS -> throw new IllegalStateException("WebP declares no frame restoring to the previous canvas");
                    case NONE, DO_NOT_DISPOSE -> { }
                }
            }

            int[] pixels = frame.pixels().data();
            for (int row = 0; row < frame.height(); row++) {
                for (int column = 0; column < frame.width(); column++) {
                    int at = (frame.offsetY() + row) * width + frame.offsetX() + column;
                    int src = pixels[row * frame.width() + column];
                    canvas[at] = frame.blend() == FrameBlend.SOURCE ? src : over(src, canvas[at]);
                }
            }

            shown.add(canvas.clone());
            prior = frame;
        }

        return shown;
    }

    /**
     * Lays the straight-alpha {@code src} over {@code dst} as the WebP container blends: the alpha is
     * {@code src.A + dst.A * (1 - src.A)} and each colour channel
     * {@code (src.C * src.A + dst.C * dst.A * (1 - src.A)) / alpha}, alphas taken in {@code [0, 1]},
     * and every channel is zero where the alpha is.
     */
    private static int over(int src, int dst) {
        double srcAlpha = (src >>> 24) / 255.0;
        double dstWeight = (dst >>> 24) / 255.0 * (1 - srcAlpha);
        double alpha = srcAlpha + dstWeight;

        if (alpha == 0)
            return CLEAR;

        int argb = (int) Math.round(alpha * 255) << 24;
        for (int shift = 0; shift < 24; shift += 8) {
            double channel = (((src >> shift) & 0xFF) * srcAlpha + ((dst >> shift) & 0xFF) * dstWeight) / alpha;
            argb |= (int) Math.round(channel) << shift;
        }

        return argb;
    }

    // ──── reading ────

    private static @NotNull AnimatedImageData read(byte @NotNull [] webp) {
        ImageData decoded = new WebPImageReader().read(webp);
        assertThat(decoded, instanceOf(AnimatedImageData.class));
        return (AnimatedImageData) decoded;
    }

    private static @NotNull List<int[]> pictures(@NotNull AnimatedImageData animation) {
        List<int[]> pictures = new ArrayList<>();

        for (ImageFrame frame : animation.getFrames())
            pictures.add(frame.pixels().data());

        return pictures;
    }

    /**
     * Returns each of {@code pictures} with every pixel's colour dropped and its alpha kept.
     */
    private static @NotNull List<int[]> alphas(@NotNull List<int[]> pictures) {
        List<int[]> alphas = new ArrayList<>();

        for (int[] picture : pictures)
            alphas.add(Arrays.stream(picture).map(argb -> argb & 0xFF000000).toArray());

        return alphas;
    }

    private static void assertPictures(@NotNull List<int[]> actual, @NotNull List<int[]> expected) {
        assertThat(actual, hasSize(expected.size()));

        for (int i = 0; i < expected.size(); i++)
            assertThat("frame " + i, hex(actual.get(i)), equalTo(hex(expected.get(i))));
    }

    private static @NotNull List<String> hex(int @NotNull [] picture) {
        return Arrays.stream(picture).mapToObj(argb -> String.format("%08x", argb)).toList();
    }

}
