package dev.simplified.image.codec.tga;

import dev.simplified.image.codec.ImageWriteOptions;
import org.jetbrains.annotations.NotNull;

/**
 * TGA-specific encoding options.
 */
public record TgaWriteOptions(boolean rle) implements ImageWriteOptions {

    /**
     * Returns a new builder for TGA write options.
     *
     * @return a new builder instance
     */
    public static @NotNull Builder builder() {
        return new Builder();
    }

    /**
     * Builds {@link TgaWriteOptions} instances.
     */
    public static class Builder {

        private boolean rle = true;

        /**
         * Enables run-length encoding (image type 10). This is the default.
         *
         * @return this builder for chaining
         */
        public @NotNull Builder isRle() {
            return this.isRle(true);
        }

        /**
         * Sets run-length encoding on or off. RLE on -> image type 10; RLE off -> image type 2
         * (uncompressed true-color).
         *
         * @param rle true to enable RLE compression
         * @return this builder for chaining
         */
        public @NotNull Builder isRle(boolean rle) {
            this.rle = rle;
            return this;
        }

        public @NotNull TgaWriteOptions build() {
            return new TgaWriteOptions(this.rle);
        }

    }

}
