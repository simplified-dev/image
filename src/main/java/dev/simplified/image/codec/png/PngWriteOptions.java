package dev.sbs.api.io.image.codec.png;

import dev.sbs.api.io.image.codec.ImageWriteOptions;
import dev.sbs.api.util.NumberUtil;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.jetbrains.annotations.NotNull;

/**
 * PNG-specific encoding options.
 */
@Getter
@RequiredArgsConstructor(access = AccessLevel.PRIVATE)
public class PngWriteOptions implements ImageWriteOptions {

    private final int compressionLevel;

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
            this.compressionLevel = NumberUtil.ensureRange(compressionLevel, 0, 9);
            return this;
        }

        public @NotNull PngWriteOptions build() {
            return new PngWriteOptions(this.compressionLevel);
        }

    }

}
