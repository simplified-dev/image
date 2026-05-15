package dev.simplified.image.codec;

import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.exception.ImageDecodeException;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

/**
 * Decodes raw image bytes into format-agnostic {@link ImageData}.
 */
public interface ImageReader {

    /**
     * The image format this reader handles.
     */
    @NotNull ImageFormat getFormat();

    /**
     * Returns whether this reader can decode the given byte data.
     *
     * @param data the raw image bytes
     * @return true if this reader supports the data's format
     */
    boolean canRead(byte @NotNull [] data);

    /**
     * Decodes image data from a byte array using default options.
     *
     * @param data the raw image bytes
     * @return the decoded image data
     * @throws ImageDecodeException if decoding fails
     */
    default @NotNull ImageData read(byte @NotNull [] data) {
        return this.read(data, null);
    }

    /**
     * Decodes image data from a byte array with the given options.
     *
     * @param data the raw image bytes
     * @param options format-specific read options, or null for defaults
     * @return the decoded image data
     * @throws ImageDecodeException if decoding fails
     */
    @NotNull ImageData read(byte @NotNull [] data, @Nullable ImageReadOptions options);

}
