package dev.simplified.image;

import dev.simplified.collection.ConcurrentList;
import dev.simplified.image.data.AnimatedImageData;
import dev.simplified.image.data.ImageFrame;
import dev.simplified.image.data.StaticImageData;
import dev.simplified.image.pixel.PixelBuffer;
import org.jetbrains.annotations.NotNull;

import java.awt.image.BufferedImage;

/**
 * Format-agnostic representation of decoded image data.
 * <p>
 * Implementations include {@link StaticImageData} for single-frame images
 * and {@link AnimatedImageData} for multi-frame animated images.
 */
public interface ImageData {

    /** The canvas width in pixels. */
    int getWidth();

    /** The canvas height in pixels. */
    int getHeight();

    /** Whether this image contains an alpha channel. */
    boolean hasAlpha();

    /** Whether this image contains multiple animation frames. */
    boolean isAnimated();

    /**
     * Returns all frames in this image.
     * <p>
     * Static images return a single-element list.
     */
    @NotNull ConcurrentList<ImageFrame> getFrames();

    /**
     * Returns the first frame's pixel data.
     *
     * @return the first frame's pixel buffer
     */
    default @NotNull PixelBuffer toPixelBuffer() {
        return getFrames().getFirst().pixels();
    }

    /**
     * Returns the first frame as a {@link BufferedImage}.
     * <p>
     * The image is constructed on demand from the underlying pixel data.
     *
     * @return the first frame's pixel data as a buffered image
     */
    default @NotNull BufferedImage toBufferedImage() {
        return toPixelBuffer().toBufferedImage();
    }

}
