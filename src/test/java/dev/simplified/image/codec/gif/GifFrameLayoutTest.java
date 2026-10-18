package dev.simplified.image.codec.gif;

import dev.simplified.image.AnimatedImageData;
import dev.simplified.image.ImageData;
import dev.simplified.image.PixelBuffer;
import dev.simplified.image.exception.ImageEncodeException;
import dev.simplified.image.frame.FrameBlend;
import dev.simplified.image.frame.FrameDisposal;
import dev.simplified.image.frame.ImageFrame;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Pins how {@link GifImageWriter} lays out an animation whose frames differ in size or sit at offsets,
 * and the logical screen it declares.
 */
class GifFrameLayoutTest {

    private static @NotNull PixelBuffer filled(int width, int height, int argb) {
        PixelBuffer buffer = PixelBuffer.create(width, height);
        buffer.fill(argb);
        return buffer;
    }

    private static @NotNull ImageFrame frame(@NotNull PixelBuffer pixels, int x, int y) {
        return ImageFrame.of(pixels, 100, x, y, FrameDisposal.DO_NOT_DISPOSE, FrameBlend.OVER);
    }

    /** The logical screen width and height a GIF's header declares, little-endian at bytes 6 to 9. */
    private static int[] logicalScreen(byte @NotNull [] gif) {
        return new int[]{ (gif[6] & 0xFF) | (gif[7] & 0xFF) << 8, (gif[8] & 0xFF) | (gif[9] & 0xFF) << 8 };
    }

    @Test
    @DisplayName("frames of different sizes are written whole, each at its own offset")
    void framesOfDifferentSizesKeepTheirPixelsAndOffsets() {
        AnimatedImageData animation = AnimatedImageData.builder()
            .withWidth(4)
            .withHeight(4)
            .withFrame(frame(filled(4, 4, 0xFFFF0000), 0, 0))
            .withFrame(frame(filled(2, 2, 0xFF00FF00), 1, 1))
            .withFrame(frame(filled(2, 2, 0xFF0000FF), 2, 2))
            .build();

        ImageData decoded = new GifImageReader().read(new GifImageWriter().write(animation));
        assertThat(decoded.getFrames().size(), is(3));

        for (int i = 0; i < 3; i++) {
            ImageFrame written = animation.getFrames().get(i);
            ImageFrame read = decoded.getFrames().get(i);
            assertThat("frame " + i + " width", read.width(), is(written.width()));
            assertThat("frame " + i + " height", read.height(), is(written.height()));
            assertThat("frame " + i + " x", read.offsetX(), is(written.offsetX()));
            assertThat("frame " + i + " y", read.offsetY(), is(written.offsetY()));
            for (int pixel : read.pixels().data())
                assertThat("frame " + i + " colour", pixel, is(written.pixels().getPixel(0, 0)));
        }
    }

    @Test
    @DisplayName("the logical screen is the image's canvas, not its first frame")
    void logicalScreenIsTheCanvas() {
        AnimatedImageData animation = AnimatedImageData.builder()
            .withWidth(16)
            .withHeight(12)
            .withFrame(frame(filled(8, 8, 0xFFFF0000), 0, 0))
            .withFrame(frame(filled(8, 4, 0xFF00FF00), 8, 8))
            .build();

        byte[] gif = new GifImageWriter().write(animation);
        assertThat(logicalScreen(gif)[0], is(16));
        assertThat(logicalScreen(gif)[1], is(12));

        ImageData decoded = new GifImageReader().read(gif);
        assertThat(decoded.getWidth(), is(16));
        assertThat(decoded.getHeight(), is(12));
    }

    @Test
    @DisplayName("a frame at a negative offset is refused, since a GIF holds only offsets of zero or more")
    void negativeOffsetIsRefused() {
        AnimatedImageData animation = AnimatedImageData.builder()
            .withWidth(4)
            .withHeight(4)
            .withFrame(frame(filled(4, 4, 0xFFFF0000), 0, 0))
            .withFrame(frame(filled(2, 2, 0xFF00FF00), -1, 0))
            .build();

        assertThrows(ImageEncodeException.class, () -> new GifImageWriter().write(animation));
    }

}
