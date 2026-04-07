package dev.simplified.image.codec.gif;

import dev.simplified.image.codec.ImageWriteOptions;
import org.jetbrains.annotations.NotNull;

/**
 * GIF-specific encoding options.
 */
public record GifWriteOptions(int loopCount, boolean transparent, int transparentColorIndex) implements ImageWriteOptions {

    /**
     * Returns a new builder for GIF write options.
     *
     * @return a new builder instance
     */
    public static @NotNull Builder builder() {
        return new Builder();
    }

    /**
     * Builds {@link GifWriteOptions} instances.
     */
    public static class Builder {

        private int loopCount = 0;
        private boolean transparent = false;
        private int transparentColorIndex = 0;

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
         * Enables transparency rendering.
         *
         * @return this builder for chaining
         */
        public @NotNull Builder isTransparent() {
            return this.isTransparent(true);
        }

        /**
         * Sets whether the GIF should be rendered with transparency.
         *
         * @param transparent true to enable transparency
         * @return this builder for chaining
         */
        public @NotNull Builder isTransparent(boolean transparent) {
            this.transparent = transparent;
            return this;
        }

        /**
         * Sets the palette index to treat as transparent.
         *
         * @param colorIndex the transparent color index (0-255)
         * @return this builder for chaining
         */
        public @NotNull Builder withTransparentColorIndex(int colorIndex) {
            this.transparentColorIndex = colorIndex;
            return this;
        }

        public @NotNull GifWriteOptions build() {
            return new GifWriteOptions(this.loopCount, this.transparent, this.transparentColorIndex);
        }

    }

}
