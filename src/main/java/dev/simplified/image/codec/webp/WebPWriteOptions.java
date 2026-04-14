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
         * Off by default: the WebP container specification historically restricts the
         * ANMF VP8 payload to keyframes, so output with P-frames enabled is only
         * decodable by tools that accept arbitrary VP8 bitstreams (this project's own
         * reader, libvpx-based decoders). libwebp's reference WebP decoder will not
         * accept the output.
         *
         * @param usePFrames true to enable P-frames for animated lossy encode
         * @return this builder for chaining
         */
        public @NotNull Builder usePFrames(boolean usePFrames) {
            this.usePFrames = usePFrames;
            return this;
        }

        public @NotNull WebPWriteOptions build() {
            return new WebPWriteOptions(this.lossless, this.quality, this.loopCount, this.multithreaded, this.alphaCompression, this.usePFrames);
        }

    }

}
