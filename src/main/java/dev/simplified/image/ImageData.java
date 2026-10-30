package dev.simplified.image;

import dev.simplified.collection.Concurrent;
import dev.simplified.collection.ConcurrentList;
import dev.simplified.image.frame.FrameBlend;
import dev.simplified.image.frame.FrameDisposal;
import dev.simplified.image.frame.ImageFrame;
import dev.simplified.image.pixel.AlphaMode;
import dev.simplified.image.pixel.BlendMode;
import dev.simplified.image.pixel.ColorMath;
import org.jetbrains.annotations.NotNull;

import java.awt.image.BufferedImage;
import java.util.Arrays;

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
     * The convention every frame's pixels are held in, as the first frame's buffer holds it -
     * {@link AlphaMode#STRAIGHT} for a decoded image.
     */
    default @NotNull AlphaMode getAlphaMode() {
        return getFrames().getFirst().pixels().alphaMode();
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
     * already are, otherwise what {@link PixelBuffer#toAlphaMode(AlphaMode)} answers for it.
     * <p>
     * A buffer whose alpha channel carries no data holds the same colour in either convention and is
     * answered as a buffer sharing its pixel array under the asked convention.
     *
     * @param alphaMode the convention to hold the pixels in
     * @return the first frame's pixel buffer, or a converted copy of it
     */
    default @NotNull PixelBuffer toPixelBuffer(@NotNull AlphaMode alphaMode) {
        return toPixelBuffer().toAlphaMode(alphaMode);
    }

    /**
     * Returns this image with its pixels held in the given convention - this image itself when they
     * already are, otherwise a copy whose every frame is converted with
     * {@link PixelBuffer#toAlphaMode(AlphaMode)}.
     * <p>
     * A still image is copied as a {@link StaticImageData} of its first frame's converted pixels. An
     * animation is copied as an {@link AnimatedImageData} keeping this image's canvas size and each
     * frame's timing, offset, disposal and blend, and - when this image is an
     * {@link AnimatedImageData} - its loop count and background colour. A frame whose alpha channel
     * carries no data keeps its own pixel array.
     *
     * @param alphaMode the convention to hold the pixels in
     * @return this image, or a converted copy of it
     */
    default @NotNull ImageData toAlphaMode(@NotNull AlphaMode alphaMode) {
        if (getAlphaMode() == alphaMode)
            return this;

        if (!isAnimated())
            return StaticImageData.of(toPixelBuffer().toAlphaMode(alphaMode));

        AnimatedImageData.Builder builder = AnimatedImageData.builder()
            .withWidth(getWidth())
            .withHeight(getHeight());

        if (this instanceof AnimatedImageData animated)
            builder.withLoopCount(animated.getLoopCount()).withBackgroundColor(animated.getBackgroundColor());

        for (ImageFrame frame : getFrames())
            builder.withFrame(frame.withPixels(frame.pixels().toAlphaMode(alphaMode)));

        return builder.build();
    }

    /**
     * Returns this image with every frame composed onto the whole canvas as the frames play - each laid
     * at its offset, by its blend, over what the frame before left once its disposal ran.
     * <p>
     * Every frame of the result covers the canvas from the origin, blends {@link FrameBlend#SOURCE} and
     * disposes to the background, so it replaces the frame before it, and keeps its source frame's
     * delay. The canvas starts transparent. A {@link FrameBlend#SOURCE} frame replaces its rectangle,
     * transparent pixels included, and a {@link FrameBlend#OVER} frame composites over it in the
     * convention this image's frames hold, which every composed frame holds too. Once a frame has
     * shown, {@link FrameDisposal#RESTORE_TO_BACKGROUND}
     * clears its rectangle to transparent, {@link FrameDisposal#RESTORE_TO_PREVIOUS} puts back what the
     * rectangle held before the frame was drawn, and the others leave it as drawn.
     * <p>
     * An image whose every frame already covers the canvas from the origin and blends
     * {@link FrameBlend#SOURCE} is answered as it is. A still image is composed into a
     * {@link StaticImageData}; an animation into an {@link AnimatedImageData} keeping this image's canvas
     * size, convention and - when it is an {@link AnimatedImageData} - its loop count and background
     * colour.
     *
     * @return this image, or a copy whose every frame is a whole canvas
     */
    default @NotNull ImageData toCanvasFrames() {
        int width = getWidth();
        int height = getHeight();
        ConcurrentList<ImageFrame> frames = getFrames();

        boolean wholeCanvas = true;
        for (ImageFrame frame : frames) {
            if (frame.offsetX() != 0 || frame.offsetY() != 0 || frame.width() != width || frame.height() != height
                || (frame.blend() != FrameBlend.SOURCE && frames.size() > 1)) {
                wholeCanvas = false;
                break;
            }
        }
        if (wholeCanvas)
            return this;

        AlphaMode alphaMode = getAlphaMode();
        BlendMode over = alphaMode == AlphaMode.PREMULTIPLIED ? BlendMode.PREMULTIPLIED : BlendMode.STRAIGHT;
        int[] canvas = new int[width * height];
        ConcurrentList<PixelBuffer> composed = Concurrent.newList();

        for (ImageFrame frame : frames) {
            int[] previous = frame.disposal() == FrameDisposal.RESTORE_TO_PREVIOUS ? canvas.clone() : null;
            draw(canvas, width, height, frame, over);
            composed.add(new PixelBuffer(canvas.clone(), width, height, true, alphaMode));

            switch (frame.disposal()) {
                case RESTORE_TO_BACKGROUND -> clear(canvas, width, height, frame);
                case RESTORE_TO_PREVIOUS -> System.arraycopy(previous, 0, canvas, 0, canvas.length);
                default -> { }
            }
        }

        if (!isAnimated())
            return StaticImageData.of(composed.getFirst());

        AnimatedImageData.Builder builder = AnimatedImageData.builder()
            .withWidth(width)
            .withHeight(height);

        if (this instanceof AnimatedImageData animated)
            builder.withLoopCount(animated.getLoopCount()).withBackgroundColor(animated.getBackgroundColor());

        for (int i = 0; i < frames.size(); i++)
            builder.withFrame(ImageFrame.of(composed.get(i), frames.get(i).delayMs()));

        return builder.build();
    }

    /**
     * Returns the first frame as a {@link BufferedImage} sharing its pixel array, in the convention the
     * frame holds.
     * <p>
     * The image is constructed on demand from the underlying pixel data, as
     * {@link PixelBuffer#toBufferedImage()} constructs it: a {@link BufferedImage#TYPE_INT_ARGB} image
     * for an image held in {@link AlphaMode#STRAIGHT}, a {@link BufferedImage#TYPE_INT_ARGB_PRE} image
     * for one held in {@link AlphaMode#PREMULTIPLIED}. Either way edits made through the image or the
     * frame are visible from the other, and Java2D and ImageIO read the colour the pixels stand for.
     *
     * @return the first frame's pixel data as a buffered image
     */
    default @NotNull BufferedImage toBufferedImage() {
        return toPixelBuffer().toBufferedImage();
    }

    /**
     * Lays one frame onto the canvas at its offset, clipped to the canvas - replacing its rectangle under
     * {@link FrameBlend#SOURCE}, compositing over it with {@code over} under {@link FrameBlend#OVER}.
     *
     * @param canvas the canvas pixels, row-major
     * @param width the canvas width
     * @param height the canvas height
     * @param frame the frame to lay down
     * @param over the source-over mode for the convention the frame's pixels are held in
     */
    private static void draw(int @NotNull [] canvas, int width, int height, @NotNull ImageFrame frame, @NotNull BlendMode over) {
        PixelBuffer pixels = frame.pixels();
        int[] source = pixels.data();
        int x0 = Math.max(0, frame.offsetX());
        int y0 = Math.max(0, frame.offsetY());
        int x1 = Math.min(width, frame.offsetX() + pixels.width());
        int y1 = Math.min(height, frame.offsetY() + pixels.height());
        boolean replace = frame.blend() == FrameBlend.SOURCE;

        for (int y = y0; y < y1; y++) {
            int sourceRow = (y - frame.offsetY()) * pixels.width() - frame.offsetX();
            int canvasRow = y * width;
            for (int x = x0; x < x1; x++) {
                int src = source[sourceRow + x];
                canvas[canvasRow + x] = replace ? src : ColorMath.blend(src, canvas[canvasRow + x], over);
            }
        }
    }

    /**
     * Clears a frame's rectangle on the canvas to transparent, clipped to the canvas.
     *
     * @param canvas the canvas pixels, row-major
     * @param width the canvas width
     * @param height the canvas height
     * @param frame the frame whose rectangle is cleared
     */
    private static void clear(int @NotNull [] canvas, int width, int height, @NotNull ImageFrame frame) {
        int x0 = Math.max(0, frame.offsetX());
        int x1 = Math.min(width, frame.offsetX() + frame.width());
        int y1 = Math.min(height, frame.offsetY() + frame.height());

        for (int y = Math.max(0, frame.offsetY()); y < y1; y++) {
            if (x0 < x1)
                Arrays.fill(canvas, y * width + x0, y * width + x1, 0);
        }
    }

}
