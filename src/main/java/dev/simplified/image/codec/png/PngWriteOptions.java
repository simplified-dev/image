package dev.simplified.image.codec.png;

import dev.simplified.image.codec.ImageWriteOptions;
import dev.simplified.util.NumberUtil;
import org.jetbrains.annotations.NotNull;

/**
 * PNG-specific encoding options.
 */
public record PngWriteOptions(int compressionLevel) implements ImageWriteOptions {

    /**
     * Validates and clamps the compression level component to the {@code [0, 9]} range.
     */
    public PngWriteOptions {
        compressionLevel = NumberUtil.ensureRange(compressionLevel, 0, 9);
    }

    /**
     * Returns a new builder for PNG write options.
     *
     * @return a new builder instance
     */
    public static @NotNull Builder builder() {
        return new Builder();
    }

    /**
     * Builds {@link PngWriteOptions} instances.
     */
    public static class Builder {

        private int compressionLevel = 6;

        /**
         * Sets the PNG deflate compression level.
         *
         * @param compressionLevel the compression level (0-9)
         * @return this builder for chaining
         */
        public @NotNull Builder withCompressionLevel(int compressionLevel) {
            this.compressionLevel = compressionLevel;
            return this;
        }

        public @NotNull PngWriteOptions build() {
            return new PngWriteOptions(this.compressionLevel);
        }

    }

}
