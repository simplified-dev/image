package dev.simplified.image.codec.tiff;

import dev.simplified.annotations.ClassBuilder;
import dev.simplified.annotations.SetterNames;
import dev.simplified.image.codec.ImageWriteOptions;
import org.jetbrains.annotations.NotNull;

/**
 * TIFF-specific encoding options selecting the compression scheme applied to each page.
 *
 * @param compression the compression scheme applied to each page, {@link Compression#DEFLATE} when
 *     left unset
 */
@ClassBuilder(setters = @SetterNames(set = "with{}"))
public record TiffWriteOptions(@NotNull Compression compression) implements ImageWriteOptions {

    public TiffWriteOptions {
        if (compression == null) compression = Compression.DEFLATE;
    }

    /**
     * TIFF compression scheme. All values are lossless except {@link #NONE} (which is still lossless
     * by nature of being uncompressed).
     */
    public enum Compression {

        /**
         * Uncompressed baseline TIFF. Largest output; highest compatibility.
         */
        NONE,

        /**
         * Lempel-Ziv-Welch. Good compression ratio on indexed/synthetic imagery.
         */
        LZW,

        /**
         * Deflate (zlib). Strong general-purpose compression.
         */
        DEFLATE,

        /**
         * Apple PackBits byte-level RLE. Weakest compression but universally supported.
         */
        PACKBITS

    }

}
