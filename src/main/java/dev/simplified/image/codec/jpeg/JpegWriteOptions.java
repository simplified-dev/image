package dev.sbs.api.io.image.codec.jpeg;

import dev.sbs.api.io.image.codec.ImageWriteOptions;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.jetbrains.annotations.NotNull;

/**
 * JPEG-specific encoding options.
 */
@Getter
@RequiredArgsConstructor(access = AccessLevel.PRIVATE)
public class JpegWriteOptions implements ImageWriteOptions {

    private final float quality;

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
            this.quality = Math.clamp(quality, 0.0f, 1.0f);
            return this;
        }

        public @NotNull JpegWriteOptions build() {
            return new JpegWriteOptions(this.quality);
        }

    }

}
