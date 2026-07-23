package dev.simplified.image.data;

import dev.simplified.image.pixel.PixelBuffer;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.sameInstance;

/**
 * Pins the animation timeline: that the reported duration is the frames' own summed timing, and
 * that walking it reaches every frame that occupies time.
 */
class AnimatedImageDataTest {

    // ──── declared duration ────

    @Test
    @DisplayName("total duration is the summed declared delays")
    void totalDurationIsTheHonestSum() {
        assertThat(animation(17, 17, 16).getTotalDurationMs(), equalTo(50));
        assertThat(animation(33, 33, 33).getTotalDurationMs(), equalTo(99));
        assertThat(animation(50, 100, 150).getTotalDurationMs(), equalTo(300));
    }

    @Test
    @DisplayName("a delay below any playback minimum is still reported as itself")
    void shortDelaysAreNotInflated() {
        // The reported duration describes the animation, not what a given player would show; a
        // consumer that wants a viewer's behavior has to model that itself.
        assertThat(animation(1, 1, 1).getTotalDurationMs(), equalTo(3));
    }

    // ──── walking the timeline ────

    @Test
    @DisplayName("each frame occupies exactly its own delay")
    void framesOccupyTheirDeclaredDelay() {
        AnimatedImageData animation = animation(10, 20, 30);

        assertThat(frameAt(animation, 0), equalTo(0));
        assertThat(frameAt(animation, 9), equalTo(0));
        assertThat(frameAt(animation, 10), equalTo(1));
        assertThat(frameAt(animation, 29), equalTo(1));
        assertThat(frameAt(animation, 30), equalTo(2));
        assertThat(frameAt(animation, 59), equalTo(2));
    }

    @Test
    @DisplayName("the timeline wraps at the declared total")
    void timelineWraps() {
        AnimatedImageData animation = animation(10, 20, 30);
        assertThat(frameAt(animation, 60), equalTo(0));
        assertThat(frameAt(animation, 95), equalTo(2));
    }

    @Test
    @DisplayName("the last frame is reachable")
    void lastFrameIsReachable() {
        AnimatedImageData animation = animation(50, 50, 50);
        assertThat(frameAt(animation, 149), equalTo(2));
    }

    // ──── zero-delay frames ────

    @Test
    @DisplayName("a zero-delay frame occupies no time and never displaces one that does")
    void zeroDelayFramesAreSkipped() {
        AnimatedImageData animation = animation(0, 50);

        assertThat(animation.getTotalDurationMs(), equalTo(50));
        assertThat(frameAt(animation, 0), equalTo(1));
        assertThat(frameAt(animation, 25), equalTo(1));
        assertThat(frameAt(animation, 49), equalTo(1));
    }

    @Test
    @DisplayName("a zero-delay frame between two timed frames does not swallow the tail")
    void zeroDelayFrameBetweenTimedFrames() {
        AnimatedImageData animation = animation(20, 0, 30);

        assertThat(animation.getTotalDurationMs(), equalTo(50));
        assertThat(frameAt(animation, 0), equalTo(0));
        assertThat(frameAt(animation, 19), equalTo(0));
        assertThat(frameAt(animation, 20), equalTo(2));
        assertThat(frameAt(animation, 49), equalTo(2));
    }

    @Test
    @DisplayName("an animation with no duration at all resolves to its first frame")
    void zeroLengthAnimationResolvesToFirstFrame() {
        AnimatedImageData animation = animation(0, 0, 0);

        assertThat(animation.getTotalDurationMs(), equalTo(0));
        // The guard that returns here is also what keeps the timeline's modulo from dividing by
        // zero, so it has to hold for every query, not just the first.
        assertThat(frameAt(animation, 0), equalTo(0));
        assertThat(frameAt(animation, 1_000), equalTo(0));
    }

    // ──── interpolation ────

    @Test
    @DisplayName("interpolation blends across a frame's own delay")
    void interpolationSpansTheDeclaredDelay() {
        AnimatedImageData animation = animation(20, 20);

        assertThat(animation.getFrameAtTime(0, true).interpolated(), equalTo(false));
        assertThat(animation.getFrameAtTime(10, true).interpolated(), equalTo(true));
        assertThat(animation.getFrameAtTime(19, true).interpolated(), equalTo(true));
    }

    @Test
    @DisplayName("interpolation resolves to the next frame once it is close enough to the end")
    void interpolationResolvesToNextFrameAtTheEnd() {
        // The cutoff is a fraction of the frame's own delay, so only a long frame has the
        // resolution to reach it at all.
        AnimatedImageData animation = animation(2_000, 2_000);
        AnimatedImageData.FrameAtTimeResult atEnd = animation.getFrameAtTime(1_999, true);

        assertThat(atEnd.interpolated(), equalTo(false));
        assertThat(atEnd.frame(), sameInstance(animation.getFrames().get(1)));
    }

    // ──── fixtures ────

    private static int frameAt(@NotNull AnimatedImageData animation, long elapsedMs) {
        return animation.getFrames().indexOf(animation.getFrameAtTime(elapsedMs, false).frame());
    }

    private static @NotNull AnimatedImageData animation(int @NotNull ... delaysMs) {
        AnimatedImageData.Builder builder = AnimatedImageData.builder();

        for (int index = 0; index < delaysMs.length; index++) {
            // Frames compare by pixel value, so identical buffers would make two frames equal and
            // resolving one back to its position would find the wrong index.
            PixelBuffer pixels = PixelBuffer.create(2, 2);
            pixels.fill(0xFF000000 | (index + 1));
            builder.withFrame(ImageFrame.of(pixels, delaysMs[index]));
        }

        return builder.build();
    }

}
