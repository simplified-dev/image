package dev.simplified.image.codec.tiff;

import dev.simplified.image.codec.ImageWriteOptions;
import org.jetbrains.annotations.NotNull;

/**
 * TIFF-specific encoding options selecting the compression scheme applied to each page.
 */
public record TiffWriteOptions(@NotNull Compression compression) implements ImageWriteOptions {

    /**
     * Returns a new builder for TIFF write options.
     *
     * @return a new builder instance
     */
    public static @NotNull Builder builder() {
        return new Builder();
    }

    /**
     * TIFF compression scheme. All values are lossless except {@link #NONE} (which is still lossless
     * by nature of being uncompressed).
     */
    public enum Compression {

        /** Uncompressed baseline TIFF. Largest output; highest compatibility. */
        NONE,

        /** Lempel-Ziv-Welch. Good compression ratio on indexed/synthetic imagery. */
        LZW,

        /** Deflate (zlib). Strong general-purpose compression. */
        DEFLATE,

        /** Apple PackBits byte-level RLE. Weakest compression but universally supported. */
        PACKBITS

    }

    /**
     * Builds {@link TiffWriteOptions} instances.
     */
    public static class Builder {

        private Compression compression = Compression.DEFLATE;

        /**
         * Sets the compression scheme.
         *
         * @param compression the compression scheme to apply
         * @return this builder for chaining
         */
        public @NotNull Builder withCompression(@NotNull Compression compression) {
            this.compression = compression;
            return this;
        }

        public @NotNull TiffWriteOptions build() {
            return new TiffWriteOptions(this.compression);
        }

    }

}
