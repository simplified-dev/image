package dev.simplified.image;

import dev.simplified.collection.ConcurrentList;
import dev.simplified.image.frame.ImageFrame;
import dev.simplified.image.pixel.AlphaMode;
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

    /**
     * The canvas width in pixels.
     */
    int getWidth();

    /**
     * The canvas height in pixels.
     */
    int getHeight();

    /**
     * Whether this image contains an alpha channel.
     */
    boolean hasAlpha();

    /**
     * Whether this image contains multiple animation frames.
     */
    boolean isAnimated();

    /**
     * The convention every frame's pixels are held in - {@link AlphaMode#STRAIGHT} for a decoded image
     * and for any image not built to say otherwise.
     */
    default @NotNull AlphaMode getAlphaMode() {
        return AlphaMode.STRAIGHT;
    }

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
     * Returns the first frame's pixels held in the given convention - the frame's own buffer when they
     * already are, otherwise a converted copy.
     * <p>
     * A buffer whose alpha channel carries no data holds the same colour in either convention and is
     * answered as it is.
     *
     * @param alphaMode the convention to hold the pixels in
     * @return the first frame's pixel buffer, or a converted copy of it
     */
    default @NotNull PixelBuffer toPixelBuffer(@NotNull AlphaMode alphaMode) {
        PixelBuffer pixels = toPixelBuffer();
        return getAlphaMode() == alphaMode ? pixels : convert(pixels, alphaMode);
    }

    /**
     * Returns this image with its pixels held in the given convention - this image itself when they
     * already are, otherwise a copy whose every frame is converted.
     * <p>
     * A still image is copied as a {@link StaticImageData} of its first frame's converted pixels. An
     * animation is copied as an {@link AnimatedImageData} keeping this image's canvas size and each
     * frame's timing, offset, disposal and blend, and - when this image is an
     * {@link AnimatedImageData} - its loop count and background colour. A frame whose alpha channel
     * carries no data keeps its own buffer.
     *
     * @param alphaMode the convention to hold the pixels in
     * @return this image, or a converted copy of it
     */
    default @NotNull ImageData toAlphaMode(@NotNull AlphaMode alphaMode) {
        if (getAlphaMode() == alphaMode)
            return this;

        if (!isAnimated())
            return StaticImageData.of(convert(toPixelBuffer(), alphaMode), alphaMode);

        AnimatedImageData.Builder builder = AnimatedImageData.builder()
            .withWidth(getWidth())
            .withHeight(getHeight())
            .withAlphaMode(alphaMode);

        if (this instanceof AnimatedImageData animated)
            builder.withLoopCount(animated.getLoopCount()).withBackgroundColor(animated.getBackgroundColor());

        for (ImageFrame frame : getFrames())
            builder.withFrame(frame.withPixels(convert(frame.pixels(), alphaMode)));

        return builder.build();
    }

    /**
     * Returns the first frame as a {@link BufferedImage} holding straight colour.
     * <p>
     * The image is constructed on demand from the underlying pixel data. For an image held in
     * {@link AlphaMode#STRAIGHT} it shares the first frame's pixel array, so edits made through either
     * are visible from the other; for one held in {@link AlphaMode#PREMULTIPLIED} it holds a straight
     * copy, and edits made through it do not reach the frame.
     *
     * @return the first frame's pixel data as a buffered image
     */
    default @NotNull BufferedImage toBufferedImage() {
        return toPixelBuffer(AlphaMode.STRAIGHT).toBufferedImage();
    }

    /**
     * Converts a buffer into a copy held in the given convention, or answers it as it is when its alpha
     * channel carries no data.
     *
     * @param pixels the buffer to convert, held in the other convention
     * @param alphaMode the convention to hold the copy in
     * @return the converted copy, or {@code pixels} when it carries no alpha
     */
    private static @NotNull PixelBuffer convert(@NotNull PixelBuffer pixels, @NotNull AlphaMode alphaMode) {
        if (!pixels.hasAlpha())
            return pixels;

        PixelBuffer copy = pixels.copy();

        switch (alphaMode) {
            case STRAIGHT -> copy.unpremultiplyAlpha();
            case PREMULTIPLIED -> copy.premultiplyAlpha();
        }

        return copy;
    }

}
