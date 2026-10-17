package dev.simplified.image.codec.gif;

import dev.simplified.collection.Concurrent;
import dev.simplified.collection.ConcurrentList;
import dev.simplified.image.AnimatedImageData;
import dev.simplified.image.ImageData;
import dev.simplified.image.StaticImageData;
import dev.simplified.image.frame.FrameBlend;
import dev.simplified.image.frame.FrameDisposal;
import dev.simplified.image.frame.ImageFrame;
import dev.simplified.image.option.GifWriteOptions;
import dev.simplified.image.pixel.PixelBuffer;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.hasSize;

/**
 * Pins the disposal {@link GifImageWriter} declares for each frame, read out of the packed byte of
 * the frame's graphic control extension, and what a player shows for it.
 * <p>
 * GIF has no blend flag: a frame is always laid over what the previous frame's disposal left. The
 * writer honours a frame's {@link FrameBlend#SOURCE} through the disposal of the frame before it, so
 * what is checked here is the disposal written, never the one a frame carries.
 * <p>
 * For the same reason {@link GifImageReader} reads every frame {@link FrameBlend#OVER}, so a decoded
 * GIF is written again with the disposals it was read with.
 */
class GifDisposalTest {

    private static final int RED = 0xFFFF0000;
    private static final int BLUE = 0xFF0000FF;
    private static final int CLEAR = 0x00000000;

    /** The options every played fixture is written with: a transparent palette slot. */
    private static final GifWriteOptions TRANSPARENT = GifWriteOptions.builder().isTransparent().build();

    /**
     * The side of a decoded fixture: ImageIO's GIF reader, which {@link GifImageReader} decodes
     * through, decodes the rows of an interlaced frame shorter than eight rows wrongly, and this
     * writer interlaces every frame.
     */
    private static final int SIZE = 8;

    // ──── declared disposal ────

    @Test
    @DisplayName("every frame of an all-SOURCE animation is written restore-to-background")
    void allSourceFramesRestoreToBackground() {
        AnimatedImageData data = animation(
            frame(0, FrameDisposal.NONE, FrameBlend.SOURCE),
            frame(1, FrameDisposal.DO_NOT_DISPOSE, FrameBlend.SOURCE),
            frame(2, FrameDisposal.RESTORE_TO_PREVIOUS, FrameBlend.SOURCE)
        );

        assertThat(writtenDisposals(data, null), contains(2, 2, 2));
    }

    @Test
    @DisplayName("a frame followed by an OVER frame keeps its own disposal")
    void frameBeforeOverKeepsItsDisposal() {
        AnimatedImageData data = animation(
            frame(0, FrameDisposal.NONE, FrameBlend.SOURCE),
            frame(1, FrameDisposal.DO_NOT_DISPOSE, FrameBlend.OVER),
            frame(2, FrameDisposal.RESTORE_TO_PREVIOUS, FrameBlend.OVER),
            frame(3, FrameDisposal.NONE, FrameBlend.OVER)
        );

        // Frames 0, 1 and 2 are followed by an OVER frame and keep their own disposal; frame 3 is
        // followed by frame 0, which blends SOURCE.
        assertThat(writtenDisposals(data, null), contains(0, 1, 3, 2));
    }

    @Test
    @DisplayName("the last frame is followed by the first")
    void lastFrameWrapsToTheFirst() {
        AnimatedImageData sourceFirst = animation(
            frame(0, FrameDisposal.NONE, FrameBlend.SOURCE),
            frame(1, FrameDisposal.NONE, FrameBlend.OVER)
        );
        AnimatedImageData overFirst = animation(
            frame(0, FrameDisposal.NONE, FrameBlend.OVER),
            frame(1, FrameDisposal.NONE, FrameBlend.OVER)
        );

        assertThat(writtenDisposals(sourceFirst, null), contains(0, 2));
        assertThat(writtenDisposals(overFirst, null), contains(0, 0));
    }

    @Test
    @DisplayName("the disposal written follows the blend whatever options are passed")
    void optionsDoNotChangeTheDisposalWritten() {
        AnimatedImageData data = animation(
            frame(0, FrameDisposal.NONE, FrameBlend.SOURCE),
            frame(1, FrameDisposal.DO_NOT_DISPOSE, FrameBlend.OVER)
        );
        GifWriteOptions options = GifWriteOptions.builder().isTransparent().withLoopCount(3).build();

        assertThat(writtenDisposals(data, options), contains(0, 2));
    }

    @Test
    @DisplayName("a single frame is written with its own disposal")
    void singleFrameKeepsItsDisposal() {
        assertThat(writtenDisposals(new Still(frame(0, FrameDisposal.NONE, FrameBlend.SOURCE)), null), contains(0));
        assertThat(writtenDisposals(new Still(frame(0, FrameDisposal.DO_NOT_DISPOSE, FrameBlend.SOURCE)), null), contains(1));
        assertThat(writtenDisposals(new Still(frame(0, FrameDisposal.RESTORE_TO_PREVIOUS, FrameBlend.SOURCE)), null), contains(3));
        assertThat(writtenDisposals(StaticImageData.of(fill(2, 2, RED)), null), contains(2));

        // A one-frame animation is followed by itself; it keeps its own disposal all the same.
        assertThat(writtenDisposals(animation(frame(0, FrameDisposal.NONE, FrameBlend.SOURCE)), null), contains(0));
    }

    // ──── what a player shows ────

    @Test
    @DisplayName("a SOURCE frame shows no pixel of the frame before it where it is transparent")
    void sourceFrameDoesNotSmear() {
        List<int[]> shown = play(halves(FrameBlend.SOURCE));

        assertThat(shown, hasSize(2));
        assertThat(boxed(shown.getFirst()), equalTo(boxed(half(true, RED))));
        assertThat(boxed(shown.getLast()), equalTo(boxed(half(false, BLUE))));
    }

    @Test
    @DisplayName("an OVER frame shows the frame before it where it is transparent")
    void overFrameShowsTheFrameBeneath() {
        List<int[]> shown = play(halves(FrameBlend.OVER));

        int[] both = half(true, RED);
        for (int i = 0; i < both.length; i++) {
            if (both[i] == CLEAR)
                both[i] = BLUE;
        }

        assertThat(shown, hasSize(2));
        assertThat(boxed(shown.getFirst()), equalTo(boxed(half(true, RED))));
        assertThat(boxed(shown.getLast()), equalTo(boxed(both)));
    }

    // ──── a decoded GIF ────

    @Test
    @DisplayName("every frame of a decoded GIF blends OVER")
    void decodedFramesBlendOver() {
        ImageData decoded = new GifImageReader().read(new GifImageWriter().write(blockOverRed(), TRANSPARENT));

        assertThat(decoded.getFrames(), hasSize(2));
        for (ImageFrame frame : decoded.getFrames())
            assertThat(frame.blend(), equalTo(FrameBlend.OVER));
    }

    @Test
    @DisplayName("a decoded GIF is written again with the disposals it was read with")
    void decodedGifKeepsItsDisposals() {
        byte[] original = new GifImageWriter().write(blockOverRed(), TRANSPARENT);
        byte[] rewritten = new GifImageWriter().write(new GifImageReader().read(original), TRANSPARENT);

        // Frame 0 is followed by an OVER frame and keeps its own disposal; frame 1 is followed by
        // frame 0, which blends SOURCE.
        assertThat(disposals(original), contains(0, 2));
        assertThat(disposals(rewritten), equalTo(disposals(original)));
    }

    @Test
    @DisplayName("a decoded GIF written again shows the frame beneath where a later frame is transparent")
    void decodedGifShowsTheFrameBeneath() {
        byte[] original = new GifImageWriter().write(blockOverRed(), TRANSPARENT);
        byte[] rewritten = new GifImageWriter().write(new GifImageReader().read(original), TRANSPARENT);
        int beneath = 5 * SIZE + 5;

        assertThat(Integer.toHexString(play(original).getLast()[beneath]), equalTo(Integer.toHexString(RED)));
        assertThat(Integer.toHexString(play(rewritten).getLast()[beneath]), equalTo(Integer.toHexString(RED)));
    }

    // ──── fixtures ────

    /**
     * Two frames each disposed {@link FrameDisposal#NONE}: opaque red over the whole canvas, then a
     * 2x2 blue block at (1, 1) laid {@link FrameBlend#OVER} it, transparent everywhere else.
     */
    private static @NotNull AnimatedImageData blockOverRed() {
        int[] red = new int[SIZE * SIZE];
        Arrays.fill(red, RED);
        int[] block = new int[SIZE * SIZE];

        for (int y = 1; y < 3; y++) {
            for (int x = 1; x < 3; x++)
                block[y * SIZE + x] = BLUE;
        }

        return animation(
            ImageFrame.of(PixelBuffer.of(red, SIZE, SIZE), 100, 0, 0, FrameDisposal.NONE, FrameBlend.SOURCE),
            ImageFrame.of(PixelBuffer.of(block, SIZE, SIZE), 100, 0, 0, FrameDisposal.NONE, FrameBlend.OVER)
        );
    }

    /**
     * Two transparent frames whose opaque pixels do not overlap - red on the left half, then blue on
     * the right - each disposed {@link FrameDisposal#NONE}, so the second frame's blend alone decides
     * whether the first shows through it.
     */
    private static @NotNull AnimatedImageData halves(@NotNull FrameBlend secondBlend) {
        return animation(
            ImageFrame.of(PixelBuffer.of(half(true, RED), SIZE, SIZE), 100, 0, 0, FrameDisposal.NONE, FrameBlend.SOURCE),
            ImageFrame.of(PixelBuffer.of(half(false, BLUE), SIZE, SIZE), 100, 0, 0, FrameDisposal.NONE, secondBlend)
        );
    }

    /**
     * A square {@link #SIZE} pixels on a side, holding {@code argb} on its left half when {@code left}
     * is set and on its right half otherwise, and transparent on the other half.
     */
    private static int @NotNull [] half(boolean left, int argb) {
        int[] pixels = new int[SIZE * SIZE];

        for (int i = 0; i < pixels.length; i++)
            pixels[i] = (i % SIZE < SIZE / 2) == left ? argb : CLEAR;

        return pixels;
    }

    private static @NotNull AnimatedImageData animation(@NotNull ImageFrame @NotNull ... frames) {
        AnimatedImageData.Builder builder = AnimatedImageData.builder();

        for (ImageFrame frame : frames)
            builder.withFrame(frame);

        return builder.build();
    }

    /**
     * A 2x2 frame of one colour per index, so consecutive frames differ - an encoder is free to
     * handle an unchanged frame differently.
     */
    private static @NotNull ImageFrame frame(int index, @NotNull FrameDisposal disposal, @NotNull FrameBlend blend) {
        int color = switch (index % 4) {
            case 0 -> RED;
            case 1 -> BLUE;
            case 2 -> 0xFF00FF00;
            default -> 0xFFFFFFFF;
        };

        return ImageFrame.of(fill(2, 2, color), 100, 0, 0, disposal, blend);
    }

    private static @NotNull PixelBuffer fill(int width, int height, int argb) {
        PixelBuffer pixels = PixelBuffer.create(width, height);
        pixels.fill(argb);
        return pixels;
    }

    /**
     * A still image holding one frame exactly as it is given, which {@link StaticImageData#of(PixelBuffer)}
     * cannot: it builds its frame with the default disposal.
     *
     * @param frame the one frame
     */
    private record Still(@NotNull ImageFrame frame) implements ImageData {

        @Override
        public int getWidth() {
            return this.frame.width();
        }

        @Override
        public int getHeight() {
            return this.frame.height();
        }

        @Override
        public boolean hasAlpha() {
            return this.frame.hasAlpha();
        }

        @Override
        public boolean isAnimated() {
            return false;
        }

        @Override
        public @NotNull ConcurrentList<ImageFrame> getFrames() {
            ConcurrentList<ImageFrame> frames = Concurrent.newList();
            frames.add(this.frame);
            return frames;
        }

    }

    // ──── GIF bytes ────

    /**
     * Writes {@code data} and returns the disposal method each graphic control extension declares,
     * {@code (packed >> 2) & 7}, in file order.
     */
    private static @NotNull List<Integer> writtenDisposals(@NotNull ImageData data, @Nullable GifWriteOptions options) {
        return disposals(options == null
            ? new GifImageWriter().write(data)
            : new GifImageWriter().write(data, options));
    }

    /**
     * Returns the disposal method each graphic control extension of {@code gif} declares,
     * {@code (packed >> 2) & 7}, in file order.
     */
    private static @NotNull List<Integer> disposals(byte @NotNull [] gif) {
        List<Integer> disposals = new ArrayList<>();

        for (int offset : GifBlocks.graphicControlOffsets(gif)) {
            assertThat(gif[offset] & 0xFF, equalTo(0x21));
            assertThat(gif[offset + 1] & 0xFF, equalTo(0xF9));
            assertThat(gif[offset + 2] & 0xFF, equalTo(0x04));
            int packed = gif[offset + 3] & 0xFF;
            disposals.add((packed >> 2) & 0x07);
        }

        return disposals;
    }

    /**
     * Writes {@code data} with a transparent palette slot and plays it, returning the canvas shown
     * after each frame.
     */
    private static @NotNull List<int[]> play(@NotNull ImageData data) {
        return play(new GifImageWriter().write(data, TRANSPARENT));
    }

    /**
     * Decodes {@code gif} and plays it as a GIF player does, returning the canvas shown after each
     * frame.
     * <p>
     * Each frame is laid over the canvas the previous frame's disposal left, a transparent pixel
     * showing what lies beneath. Every frame here spans the canvas from the origin, so restoring to
     * the background clears the whole canvas.
     *
     * @throws IllegalStateException if a decoded frame restores to the previous canvas, which no
     *     fixture here declares
     */
    private static @NotNull List<int[]> play(byte @NotNull [] gif) {
        ImageData decoded = new GifImageReader().read(gif);
        int[] canvas = new int[decoded.getWidth() * decoded.getHeight()];
        FrameDisposal pending = FrameDisposal.NONE;
        List<int[]> shown = new ArrayList<>();

        for (ImageFrame frame : decoded.getFrames()) {
            switch (pending) {
                case RESTORE_TO_BACKGROUND -> Arrays.fill(canvas, CLEAR);
                case RESTORE_TO_PREVIOUS -> throw new IllegalStateException("No fixture restores to the previous canvas");
                case NONE, DO_NOT_DISPOSE -> { }
            }

            int[] pixels = frame.pixels().data();
            for (int i = 0; i < pixels.length; i++) {
                if ((pixels[i] >>> 24) != 0)
                    canvas[i] = pixels[i];
            }

            shown.add(canvas.clone());
            pending = frame.disposal();
        }

        return shown;
    }

    private static @NotNull List<Integer> boxed(int @NotNull [] values) {
        List<Integer> list = new ArrayList<>(values.length);
        for (int value : values) list.add(value);
        return list;
    }

}
