package dev.simplified.image.codec.jpeg;

import dev.simplified.image.codec.ImageWriteOptions;
import org.jetbrains.annotations.NotNull;

/**
 * JPEG-specific encoding options.
 */
public record JpegWriteOptions(float quality) implements ImageWriteOptions {

    /**
     * Validates and clamps the quality component to the {@code [0.0, 1.0]} range.
     */
    public JpegWriteOptions {
        quality = Math.clamp(quality, 0.0f, 1.0f);
    }

    /**
     * Returns a new builder for JPEG write options.
     *
     * @return a new builder instance
     */
    public static @NotNull Builder builder() {
        return new Builder();
    }

    /**
     * Builds {@link JpegWriteOptions} instances.
     */
    public static class Builder {

        private float quality = 0.75f;

        /**
         * Sets the JPEG compression quality.
         *
         * @param quality the quality value (0.0 - 1.0)
         * @return this builder for chaining
         */
        public @NotNull Builder withQuality(float quality) {
            this.quality = quality;
            return this;
        }

        public @NotNull JpegWriteOptions build() {
            return new JpegWriteOptions(this.quality);
        }

    }

}
