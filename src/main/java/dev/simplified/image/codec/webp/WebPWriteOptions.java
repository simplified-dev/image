package dev.simplified.image.codec.webp;

import dev.simplified.image.codec.ImageWriteOptions;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.jetbrains.annotations.NotNull;

/**
 * WebP-specific encoding options supporting both lossless (VP8L) and lossy (VP8) modes.
 */
@Getter
@RequiredArgsConstructor(access = AccessLevel.PRIVATE)
public class WebPWriteOptions implements ImageWriteOptions {

    private final boolean lossless;
    private final float quality;
    private final int loopCount;
    private final boolean multithreaded;
    private final boolean alphaCompression;
    private final boolean usePFrames;
    /**
     * Interval at which the animated-lossy writer forces a keyframe. {@code 0}
     * disables intermediate keyframes (only the first frame is a keyframe); a
     * positive value forces frame {@code i} to be a keyframe when {@code i == 0}
     * or {@code i % forceKeyframeEvery == 0}; {@code -1} defers to the writer's
     * content-length-dependent default (30 for animations longer than 60 frames,
     * 0 otherwise). Ignored when {@code usePFrames == false}.
     */
    private final int forceKeyframeEvery;
    /**
     * Parallelism for the VP8 P-frame motion-search prepass. {@code -1} defers to
     * the writer ({@code Runtime.availableProcessors()}); {@code 1} forces
     * deterministic serial search; {@code N >= 1} uses exactly {@code N} worker
     * threads. Motion-searched MVs are a hint, not a correctness requirement, so
     * output is bit-identical across thread counts. Ignored for lossless encodes
     * and when {@code usePFrames == false}.
     */
    private final int motionSearchThreads;

    /**
     * Returns a new builder for WebP write options.
     *
     * @return a new builder instance
     */
    public static @NotNull Builder builder() {
        return new Builder();
    }

    /**
     * Builds {@link WebPWriteOptions} instances.
     */
    public static class Builder {

        private boolean lossless = true;
        private float quality = 0.75f;
        private int loopCount = 0;
        private boolean multithreaded = true;
        private boolean alphaCompression = true;
        private boolean usePFrames = false;
        private int forceKeyframeEvery = -1;
        private int motionSearchThreads = -1;

        /**
         * Enables lossless encoding.
         *
         * @return this builder for chaining
         */
        public @NotNull Builder isLossless() {
            return this.isLossless(true);
        }

        /**
         * Sets whether to use lossless or lossy encoding.
         *
         * @param lossless true for VP8L lossless, false for VP8 lossy
         * @return this builder for chaining
         */
        public @NotNull Builder isLossless(boolean lossless) {
            this.lossless = lossless;
            return this;
        }

        /**
         * Sets the encoding quality for lossy mode.
         *
         * @param quality the quality value (0.0 - 1.0, ignored for lossless)
         * @return this builder for chaining
         */
        public @NotNull Builder withQuality(float quality) {
            this.quality = Math.clamp(quality, 0.0f, 1.0f);
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
         * Enables multi-threaded frame encoding.
         *
         * @return this builder for chaining
         */
        public @NotNull Builder isMultithreaded() {
            return this.isMultithreaded(true);
        }

        /**
         * Sets whether to use multi-threaded encoding for animated images.
         *
         * @param multithreaded true to encode frames in parallel
         * @return this builder for chaining
         */
        public @NotNull Builder isMultithreaded(boolean multithreaded) {
            this.multithreaded = multithreaded;
            return this;
        }

        /**
         * Sets whether to lossless-compress the alpha channel in lossy mode.
         *
         * @param alphaCompression true to compress the ALPH chunk
         * @return this builder for chaining
         */
        public @NotNull Builder isAlphaCompression(boolean alphaCompression) {
            this.alphaCompression = alphaCompression;
            return this;
        }

        /**
         * Enables VP8 P-frame emission for animated lossy output. When enabled, the
         * writer threads a shared-state session across frames so stationary macroblocks
         * reuse the prior frame's reconstruction (inter-skip), typically cutting file
         * size by 50-60% on text/tooltip animations.
         * <p>
         * Off by default: libwebp's reference VP8 decoder unconditionally rejects
         * non-keyframe bitstreams ({@code src/dec/vp8_dec.c} {@code VP8GetHeaders}
         * returns {@code VP8_STATUS_UNSUPPORTED_FEATURE} "Not a key frame."), and
         * {@code WebPAnimDecoder} re-creates the VP8 decoder per ANMF frame with
         * no cross-frame reference handoff - so even bypassing the check would
         * fail structurally. The WebP container spec (RFC 9649 / Google WebP
         * container spec) does not forbid P-frames at the container level, but
         * the codec decoder gate makes the constraint hard. Output with P-frames
         * enabled is therefore only decodable by tools that accept arbitrary VP8
         * bitstreams with synthesised reference state (this project's own reader,
         * libvpx-based decoders); libwebp-based tools (Chrome, Firefox,
         * Pillow-via-libwebp, ffmpeg-built-against-libwebp, Android / iOS system
         * decoders) will not accept the output. See
         * {@code research/WebP-ANMF-PFrame-Interop.md} for the full investigation.
         *
         * @param usePFrames true to enable P-frames for animated lossy encode
         * @return this builder for chaining
         */
        public @NotNull Builder usePFrames(boolean usePFrames) {
            this.usePFrames = usePFrames;
            return this;
        }

        /**
         * Sets how often the animated-lossy writer forces a keyframe. Intermediate
         * keyframes let viewers seek to an arbitrary frame at the cost of one large
         * keyframe-sized payload per interval. Only meaningful when
         * {@link #usePFrames(boolean)} is on.
         * <p>
         * Values:
         * <ul>
         *   <li>{@code -1} (default) - defer to the writer: {@code 30} for
         *       animations longer than 60 frames, {@code 0} otherwise. Short
         *       tooltip animations stay at single-keyframe maximum compression;
         *       long animations get seekable.</li>
         *   <li>{@code 0} - never force intermediate keyframes (only frame 0).
         *       Smallest output, no seekability.</li>
         *   <li>{@code N > 0} - force a keyframe every N frames (plus frame 0).</li>
         * </ul>
         *
         * @param every interval in frames, or {@code -1} for the content-length default
         * @return this builder for chaining
         */
        public @NotNull Builder withForceKeyframeEvery(int every) {
            if (every < -1)
                throw new IllegalArgumentException("forceKeyframeEvery must be >= -1");
            this.forceKeyframeEvery = every;
            return this;
        }

        /**
         * Sets the VP8 P-frame motion-search parallelism. Only meaningful for lossy
         * animated output with {@link #usePFrames(boolean)} enabled; ignored otherwise.
         * <p>
         * Values:
         * <ul>
         *   <li>{@code -1} (default) - defer to the writer:
         *       {@link Runtime#availableProcessors()}.</li>
         *   <li>{@code 1} - serial / deterministic: no pool overhead, identical output
         *       across environments.</li>
         *   <li>{@code N > 1} - use exactly {@code N} worker threads. Output is
         *       bit-identical regardless of thread count.</li>
         * </ul>
         *
         * @param threads worker-thread count, or {@code -1} for the writer default
         * @return this builder for chaining
         */
        public @NotNull Builder withMotionSearchThreads(int threads) {
            if (threads != -1 && threads < 1)
                throw new IllegalArgumentException("motionSearchThreads must be -1 or >= 1");
            this.motionSearchThreads = threads;
            return this;
        }

        public @NotNull WebPWriteOptions build() {
            return new WebPWriteOptions(this.lossless, this.quality, this.loopCount, this.multithreaded, this.alphaCompression, this.usePFrames, this.forceKeyframeEvery, this.motionSearchThreads);
        }

    }

}
