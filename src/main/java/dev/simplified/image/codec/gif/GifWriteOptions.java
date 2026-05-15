package dev.simplified.image.codec.gif;

import dev.simplified.image.codec.ImageWriteOptions;
import org.jetbrains.annotations.NotNull;

import java.awt.Graphics2D;

/**
 * GIF-specific encoding options.
 *
 * @param loopCount animation loop count ({@code 0} for infinite)
 * @param transparent whether one palette slot should be reserved as fully transparent
 * @param transparentColorIndex the palette slot to mark as transparent when {@code transparent} is set
 * @param backgroundRgb the RGB color to composite partial-alpha pixels onto before
 *     quantizing. GIF supports only 1-bit transparency, so partial alpha otherwise gets
 *     rendered as a dithered checkerboard by {@link Graphics2D}. Any pixel with
 *     {@code alpha &lt; alphaThreshold} becomes fully transparent; everything else is
 *     flattened onto this color and becomes fully opaque.
 * @param alphaThreshold pixels with alpha strictly below this value ({@code 0}-{@code 255})
 *     are treated as fully transparent; pixels at or above are flattened onto
 *     {@code backgroundRgb}. Default {@code 128}.
 */
public record GifWriteOptions(
    int loopCount,
    boolean transparent,
    int transparentColorIndex,
    int backgroundRgb,
    int alphaThreshold
) implements ImageWriteOptions {

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
        private int backgroundRgb = 0x000000;
        private int alphaThreshold = 128;

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

        /**
         * Sets the RGB color that partial-alpha pixels are composited onto before
         * quantizing. Avoids the GIF-writer dithering partial transparency into an RGB
         * checkerboard.
         *
         * @param rgb the 24-bit RGB color (defaults to {@code 0x000000})
         * @return this builder for chaining
         */
        public @NotNull Builder withBackgroundRgb(int rgb) {
            this.backgroundRgb = rgb & 0xFFFFFF;
            return this;
        }

        /**
         * Sets the alpha threshold below which a pixel is treated as fully transparent.
         * At or above the threshold, the pixel is composited onto the configured
         * background color and becomes fully opaque.
         *
         * @param threshold the threshold in {@code [0, 255]}; default {@code 128}
         * @return this builder for chaining
         */
        public @NotNull Builder withAlphaThreshold(int threshold) {
            this.alphaThreshold = Math.max(0, Math.min(255, threshold));
            return this;
        }

        public @NotNull GifWriteOptions build() {
            return new GifWriteOptions(
                this.loopCount,
                this.transparent,
                this.transparentColorIndex,
                this.backgroundRgb,
                this.alphaThreshold
            );
        }

    }

}
