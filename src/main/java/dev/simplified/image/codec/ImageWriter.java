package dev.simplified.image.codec;

import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.exception.ImageEncodeException;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

/**
 * Encodes format-agnostic {@link ImageData} into raw image bytes.
 */
public interface ImageWriter {

    /**
     * The image format this writer produces.
     */
    @NotNull ImageFormat getFormat();

    /**
     * Encodes image data to a byte array using default options.
     *
     * @param data the image data to encode
     * @return the encoded bytes
     * @throws ImageEncodeException if encoding fails
     */
    default byte @NotNull [] write(@NotNull ImageData data) {
        return this.write(data, null);
    }

    /**
     * Encodes image data to a byte array with the given options.
     *
     * @param data the image data to encode
     * @param options format-specific write options, or null for defaults
     * @return the encoded bytes
     * @throws ImageEncodeException if encoding fails
     */
    byte @NotNull [] write(@NotNull ImageData data, @Nullable ImageWriteOptions options);

}
