package dev.simplified.image;

import dev.simplified.image.frame.FrameBlend;
import dev.simplified.image.frame.FrameDisposal;
import dev.simplified.image.frame.ImageFrame;
import dev.simplified.image.frame.LoopCount;
import dev.simplified.image.pixel.AlphaMode;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.instanceOf;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.sameInstance;

class ImageDataCanvasFramesTest {

    private static final int RED = 0xFFFF0000;
    private static final int GREEN = 0xFF00FF00;
    private static final int BLUE = 0xFF0000FF;

    private static PixelBuffer filled(int width, int height, int argb) {
        PixelBuffer buffer = PixelBuffer.create(width, height);
        buffer.fill(argb);
        return buffer;
    }

    private static ImageFrame frame(PixelBuffer pixels, int delayMs, int x, int y, FrameDisposal disposal, FrameBlend blend) {
        return ImageFrame.of(pixels, delayMs, x, y, disposal, blend);
    }

    private static AnimatedImageData canvas(int width, int height, ImageFrame... frames) {
        AnimatedImageData.Builder builder = AnimatedImageData.builder().withWidth(width).withHeight(height);
        for (ImageFrame frame : frames)
            builder.withFrame(frame);
        return builder.build();
    }

    private static void assertWholeCanvas(ImageData image, int width, int height) {
        for (ImageFrame frame : image.getFrames()) {
            assertThat(frame.width(), is(width));
            assertThat(frame.height(), is(height));
            assertThat(frame.offsetX(), is(0));
            assertThat(frame.offsetY(), is(0));
            assertThat(frame.blend(), is(FrameBlend.SOURCE));
            assertThat(frame.disposal(), is(FrameDisposal.RESTORE_TO_BACKGROUND));
        }
    }

    @Test
    @DisplayName("an image whose frames already cover the canvas from the origin and replace it is answered as it is")
    void wholeCanvasFramesAreAnsweredAsTheyAre() {
        ImageData still = StaticImageData.of(filled(4, 4, RED));
        ImageData animated = AnimatedImageData.builder()
            .withFrame(ImageFrame.of(filled(4, 4, RED), 40))
            .withFrame(ImageFrame.of(filled(4, 4, BLUE), 60))
            .build();

        assertThat(still.toCanvasFrames(), is(sameInstance(still)));
        assertThat(animated.toCanvasFrames(), is(sameInstance(animated)));
    }

    @Test
    @DisplayName("frames at offsets are laid onto the whole canvas as they play")
    void offsetFramesComposeOntoTheCanvas() {
        AnimatedImageData image = canvas(4, 4,
            frame(filled(2, 2, RED), 40, 0, 0, FrameDisposal.DO_NOT_DISPOSE, FrameBlend.SOURCE),
            frame(filled(2, 2, BLUE), 60, 2, 2, FrameDisposal.DO_NOT_DISPOSE, FrameBlend.OVER));

        ImageData composed = image.toCanvasFrames();
        assertWholeCanvas(composed, 4, 4);
        assertThat(composed.getFrames().getFirst().delayMs(), is(40));
        assertThat(composed.getFrames().getLast().delayMs(), is(60));

        PixelBuffer first = composed.getFrames().getFirst().pixels();
        assertThat(first.getPixel(0, 0), is(RED));
        assertThat(first.getPixel(3, 3), is(0));

        PixelBuffer second = composed.getFrames().getLast().pixels();
        assertThat(second.getPixel(0, 0), is(RED));
        assertThat(second.getPixel(3, 3), is(BLUE));
        assertThat(second.getPixel(0, 3), is(0));
    }

    @Test
    @DisplayName("restore-to-background clears the frame's rectangle to transparent before the next frame")
    void restoreToBackgroundClearsTheRectangle() {
        AnimatedImageData image = canvas(4, 4,
            frame(filled(2, 2, RED), 40, 0, 0, FrameDisposal.RESTORE_TO_BACKGROUND, FrameBlend.SOURCE),
            frame(filled(2, 2, BLUE), 60, 2, 2, FrameDisposal.DO_NOT_DISPOSE, FrameBlend.OVER));

        PixelBuffer second = image.toCanvasFrames().getFrames().getLast().pixels();
        assertThat(second.getPixel(0, 0), is(0));
        assertThat(second.getPixel(2, 2), is(BLUE));
    }

    @Test
    @DisplayName("restore-to-previous puts back what the frame's rectangle held before it was drawn")
    void restoreToPreviousPutsBackTheRectangle() {
        AnimatedImageData image = canvas(4, 4,
            frame(filled(4, 4, RED), 40, 0, 0, FrameDisposal.DO_NOT_DISPOSE, FrameBlend.SOURCE),
            frame(filled(2, 2, GREEN), 40, 0, 0, FrameDisposal.RESTORE_TO_PREVIOUS, FrameBlend.SOURCE),
            frame(PixelBuffer.create(1, 1), 40, 3, 3, FrameDisposal.DO_NOT_DISPOSE, FrameBlend.OVER));

        ImageData composed = image.toCanvasFrames();
        assertThat(composed.getFrames().get(1).pixels().getPixel(0, 0), is(GREEN));
        assertThat(composed.getFrames().get(2).pixels().getPixel(0, 0), is(RED));
    }

    @Test
    @DisplayName("a SOURCE frame replaces its rectangle, transparent pixels included")
    void sourceFrameReplacesItsRectangle() {
        AnimatedImageData image = canvas(4, 4,
            frame(filled(4, 4, RED), 40, 0, 0, FrameDisposal.DO_NOT_DISPOSE, FrameBlend.SOURCE),
            frame(PixelBuffer.create(2, 2), 40, 1, 1, FrameDisposal.DO_NOT_DISPOSE, FrameBlend.SOURCE));

        PixelBuffer second = image.toCanvasFrames().getFrames().getLast().pixels();
        assertThat(second.getPixel(1, 1), is(0));
        assertThat(second.getPixel(0, 0), is(RED));
    }

    @Test
    @DisplayName("an OVER frame composites in the convention the image is held in")
    void overFrameBlendsInTheHeldConvention() {
        AnimatedImageData straight = canvas(2, 1,
            frame(filled(2, 1, BLUE), 40, 0, 0, FrameDisposal.DO_NOT_DISPOSE, FrameBlend.SOURCE),
            frame(filled(1, 1, 0x80FF0000), 40, 0, 0, FrameDisposal.DO_NOT_DISPOSE, FrameBlend.OVER));
        AnimatedImageData premultiplied = AnimatedImageData.builder()
            .withWidth(2)
            .withHeight(1)
            .withFrame(frame(filled(2, 1, BLUE), 40, 0, 0, FrameDisposal.DO_NOT_DISPOSE, FrameBlend.SOURCE))
            .withFrame(frame(filled(1, 1, 0x80800000), 40, 0, 0, FrameDisposal.DO_NOT_DISPOSE, FrameBlend.OVER))
            .withAlphaMode(AlphaMode.PREMULTIPLIED)
            .build();

        assertThat(straight.toCanvasFrames().getFrames().getLast().pixels().getPixel(0, 0), is(0xFF80007F));
        assertThat(premultiplied.toCanvasFrames().getFrames().getLast().pixels().getPixel(0, 0), is(0xFF80007F));
    }

    @Test
    @DisplayName("the composed animation keeps the canvas, loop count, background colour and convention")
    void keepsTheAnimationsProperties() {
        AnimatedImageData image = AnimatedImageData.builder()
            .withWidth(6)
            .withHeight(5)
            .withFrame(frame(filled(2, 2, RED), 40, 1, 1, FrameDisposal.DO_NOT_DISPOSE, FrameBlend.OVER))
            .withFrame(frame(filled(3, 3, BLUE), 60, 3, 2, FrameDisposal.RESTORE_TO_PREVIOUS, FrameBlend.SOURCE))
            .withLoopCount(LoopCount.of(3))
            .withBackgroundColor(0xFF00FF00)
            .withAlphaMode(AlphaMode.PREMULTIPLIED)
            .build();

        ImageData composed = image.toCanvasFrames();
        assertThat(composed, is(instanceOf(AnimatedImageData.class)));
        AnimatedImageData animated = (AnimatedImageData) composed;
        assertThat(animated.getWidth(), is(6));
        assertThat(animated.getHeight(), is(5));
        assertThat(animated.getLoopCount(), is(LoopCount.of(3)));
        assertThat(animated.getBackgroundColor(), is(0xFF00FF00));
        assertThat(animated.getAlphaMode(), is(AlphaMode.PREMULTIPLIED));
        assertThat(animated.getTotalDurationMs(), is(100));
        assertWholeCanvas(animated, 6, 5);
    }

}
