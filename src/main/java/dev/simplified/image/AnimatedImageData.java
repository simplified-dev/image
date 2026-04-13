package dev.simplified.image;

import dev.simplified.collection.Concurrent;
import dev.simplified.collection.ConcurrentList;
import dev.simplified.reflection.builder.BuildFlag;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.experimental.Accessors;
import org.jetbrains.annotations.NotNull;

/**
 * A multi-frame animated image with timing, loop control, and background color metadata.
 */
@Getter
@RequiredArgsConstructor(access = AccessLevel.PRIVATE)
public class AnimatedImageData implements ImageData {

    private static final int MIN_FRAME_DURATION_MS = 50;

    private final int width;
    private final int height;
    @Accessors(fluent = true)
    private final boolean hasAlpha;
    private final @NotNull ConcurrentList<ImageFrame> frames;
    private final int loopCount;
    private final int backgroundColor;
    private final int totalDurationMs;

    @Override
    public boolean isAnimated() {
        return true;
    }

    /**
     * Resolves the frame that should be displayed at the given elapsed time.
     * <p>
     * When interpolation is enabled and the elapsed time falls between two keyframes,
     * the returned frame is a per-pixel blend of the adjacent keyframes using
     * {@link PixelBuffer#lerp(PixelBuffer, PixelBuffer, float)}.
     *
     * @param elapsedMs the elapsed time since animation start in milliseconds
     * @param interpolate whether to blend between keyframes
     * @return the resolved frame and whether it was synthesized
     */
    public @NotNull FrameAtTimeResult getFrameAtTime(long elapsedMs, boolean interpolate) {
        if (this.frames.isEmpty())
            throw new IllegalStateException("Animation does not contain frames");

        if (this.totalDurationMs <= 0)
            return new FrameAtTimeResult(this.frames.getFirst(), false);

        int normalized = (int) (elapsedMs % this.totalDurationMs);
        int accumulated = 0;

        for (int index = 0; index < this.frames.size(); index++) {
            ImageFrame frame = this.frames.get(index);
            int duration = Math.max(frame.delayMs(), MIN_FRAME_DURATION_MS);
            int nextAccumulated = accumulated + duration;

            if (normalized < nextAccumulated) {
                if (!interpolate || this.frames.size() == 1)
                    return new FrameAtTimeResult(frame, false);

                int spanWithinFrame = normalized - accumulated;
                if (spanWithinFrame <= 0)
                    return new FrameAtTimeResult(frame, false);

                double progress = spanWithinFrame / (double) duration;
                if (progress <= 0d)
                    return new FrameAtTimeResult(frame, false);

                int nextIndex = (index + 1) % this.frames.size();
                if (progress >= 0.999d)
                    return new FrameAtTimeResult(this.frames.get(nextIndex), false);

                ImageFrame nextFrame = this.frames.get(nextIndex);
                PixelBuffer blended = PixelBuffer.lerp(
                    frame.pixels(),
                    nextFrame.pixels(),
                    (float) progress
                );
                ImageFrame interpolatedFrame = ImageFrame.of(blended, frame.delayMs());
                return new FrameAtTimeResult(interpolatedFrame, true);
            }

            accumulated = nextAccumulated;
        }

        return new FrameAtTimeResult(this.frames.getLast(), false);
    }

    /**
     * Returns a new builder for constructing animated image data.
     *
     * @return a new builder instance
     */
    public static @NotNull Builder builder() {
        return new Builder();
    }

    /**
     * Builds {@link AnimatedImageData} instances with configurable animation parameters.
     */
    public static class Builder {

        @BuildFlag(nonNull = true, notEmpty = true)
        private ConcurrentList<ImageFrame> frames = Concurrent.newList();
        private int width = -1;
        private int height = -1;
        private int loopCount = 0;
        private int backgroundColor = 0;

        /**
         * Appends a frame to the animation sequence.
         *
         * @param frame the frame to add
         * @return this builder for chaining
         */
        public @NotNull Builder withFrame(@NotNull ImageFrame frame) {
            this.frames.add(frame);
            return this;
        }

        /**
         * Replaces the frame list with the given frames.
         *
         * @param frames the frames to use
         * @return this builder for chaining
         */
        public @NotNull Builder withFrames(@NotNull ConcurrentList<ImageFrame> frames) {
            this.frames = frames;
            return this;
        }

        /**
         * Sets the canvas width explicitly.
         * <p>
         * When not set, the width is derived from the first frame.
         *
         * @param width the canvas width in pixels
         * @return this builder for chaining
         */
        public @NotNull Builder withWidth(int width) {
            this.width = width;
            return this;
        }

        /**
         * Sets the canvas height explicitly.
         * <p>
         * When not set, the height is derived from the first frame.
         *
         * @param height the canvas height in pixels
         * @return this builder for chaining
         */
        public @NotNull Builder withHeight(int height) {
            this.height = height;
            return this;
        }

        /**
         * Sets the animation loop count.
         *
         * @param loopCount the number of times to loop (0 for infinite)
         * @return this builder for chaining
         */
        public @NotNull Builder withLoopCount(int loopCount) {
            this.loopCount = loopCount;
            return this;
        }

        /**
         * Sets the canvas background color as a packed ARGB integer.
         *
         * @param backgroundColor the ARGB background color
         * @return this builder for chaining
         */
        public @NotNull Builder withBackgroundColor(int backgroundColor) {
            this.backgroundColor = backgroundColor;
            return this;
        }

        public @NotNull AnimatedImageData build() {
            ImageFrame first = this.frames.getFirst();
            int w = this.width > 0 ? this.width : first.pixels().width();
            int h = this.height > 0 ? this.height : first.pixels().height();
            boolean alpha = first.pixels().hasAlpha();
            int totalDuration = this.frames.stream()
                .mapToInt(frame -> Math.max(frame.delayMs(), MIN_FRAME_DURATION_MS))
                .sum();

            return new AnimatedImageData(w, h, alpha, this.frames, this.loopCount, this.backgroundColor, totalDuration);
        }

    }

    /**
     * The result of resolving a frame at a specific point in time.
     *
     * @param frame the resolved frame
     * @param interpolated whether the frame was synthesized by blending two keyframes
     */
    public record FrameAtTimeResult(@NotNull ImageFrame frame, boolean interpolated) {}

}
