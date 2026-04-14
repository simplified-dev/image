package dev.simplified.image;

import dev.simplified.image.data.FrameBlend;
import dev.simplified.image.data.FrameDisposal;
import dev.simplified.image.data.ImageFrame;
import dev.simplified.image.pixel.PixelBuffer;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.sameInstance;
import static org.junit.jupiter.api.Assertions.assertThrows;

class ImageFrameTest {

    @Test
    @DisplayName("of(pixels) produces a zero-delay static frame")
    void ofPixelsDefaults() {
        PixelBuffer pixels = PixelBuffer.create(2, 2);
        ImageFrame frame = ImageFrame.of(pixels);
        assertThat(frame.delayMs(), equalTo(0));
        assertThat(frame.offsetX(), equalTo(0));
        assertThat(frame.offsetY(), equalTo(0));
        assertThat(frame.disposal(), equalTo(FrameDisposal.NONE));
        assertThat(frame.blend(), equalTo(FrameBlend.SOURCE));
    }

    @Test
    @DisplayName("compact constructor rejects negative delayMs")
    void rejectsNegativeDelay() {
        PixelBuffer pixels = PixelBuffer.create(1, 1);
        assertThrows(IllegalArgumentException.class, () -> new ImageFrame(pixels, -1, 0, 0, FrameDisposal.NONE, FrameBlend.SOURCE));
    }

    @Test
    @DisplayName("delegating accessors return pixel buffer dimensions")
    void delegatingAccessors() {
        ImageFrame frame = ImageFrame.of(PixelBuffer.create(5, 7));
        assertThat(frame.width(), equalTo(5));
        assertThat(frame.height(), equalTo(7));
        assertThat(frame.hasAlpha(), equalTo(true));
    }

    @Test
    @DisplayName("withers update one component at a time")
    void witherIndependence() {
        PixelBuffer pixels = PixelBuffer.create(1, 1);
        ImageFrame base = ImageFrame.of(pixels, 50);

        ImageFrame delayed = base.withDelayMs(100);
        assertThat(delayed.delayMs(), equalTo(100));
        assertThat(delayed.pixels(), sameInstance(base.pixels()));

        ImageFrame offset = base.withOffset(3, 4);
        assertThat(offset.offsetX(), equalTo(3));
        assertThat(offset.offsetY(), equalTo(4));

        ImageFrame disposed = base.withDisposal(FrameDisposal.RESTORE_TO_BACKGROUND);
        assertThat(disposed.disposal(), equalTo(FrameDisposal.RESTORE_TO_BACKGROUND));

        ImageFrame blended = base.withBlend(FrameBlend.OVER);
        assertThat(blended.blend(), equalTo(FrameBlend.OVER));
    }

    @Test
    @DisplayName("FrameDisposal.of(int) resolves known values and falls back to NONE")
    void disposalOfInt() {
        assertThat(FrameDisposal.of(0), equalTo(FrameDisposal.NONE));
        assertThat(FrameDisposal.of(2), equalTo(FrameDisposal.RESTORE_TO_BACKGROUND));
        assertThat(FrameDisposal.of(99), equalTo(FrameDisposal.NONE));
        assertThat(FrameDisposal.of(-1), equalTo(FrameDisposal.NONE));
    }

    @Test
    @DisplayName("FrameDisposal.of(String) is case-insensitive and falls back to NONE")
    void disposalOfString() {
        assertThat(FrameDisposal.of("doNotDispose"), equalTo(FrameDisposal.DO_NOT_DISPOSE));
        assertThat(FrameDisposal.of("DONOTDISPOSE"), equalTo(FrameDisposal.DO_NOT_DISPOSE));
        assertThat(FrameDisposal.of("unknown"), equalTo(FrameDisposal.NONE));
    }

    @Test
    @DisplayName("FrameBlend.of(int) resolves known values and falls back to SOURCE")
    void blendOfInt() {
        assertThat(FrameBlend.of(0), equalTo(FrameBlend.SOURCE));
        assertThat(FrameBlend.of(1), equalTo(FrameBlend.OVER));
        assertThat(FrameBlend.of(99), equalTo(FrameBlend.SOURCE));
    }

}
