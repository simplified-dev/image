package dev.simplified.image.data;

import dev.simplified.image.pixel.PixelBuffer;
import org.jetbrains.annotations.NotNull;

/**
 * A single frame within an image, carrying pixel data and animation metadata.
 *
 * @param pixels the frame pixel data
 * @param delayMs the display duration in milliseconds
 * @param offsetX the horizontal offset within the canvas
 * @param offsetY the vertical offset within the canvas
 * @param disposal the disposal method after displaying this frame
 * @param blend the blending method when rendering this frame
 */
public record ImageFrame(
    @NotNull PixelBuffer pixels,
    int delayMs,
    int offsetX,
    int offsetY,
    @NotNull FrameDisposal disposal,
    @NotNull FrameBlend blend
) {

    /**
     * Validates the frame components.
     *
     * @throws IllegalArgumentException if {@code delayMs} is negative
     */
    public ImageFrame {
        if (delayMs < 0)
            throw new IllegalArgumentException("delayMs must be non-negative, got %d".formatted(delayMs));
    }

    // --- factories ---

    /**
     * Creates a static (zero-delay) frame at the canvas origin.
     *
     * @param pixels the frame pixel data
     * @return a new image frame
     */
    public static @NotNull ImageFrame of(@NotNull PixelBuffer pixels) {
        return new ImageFrame(pixels, 0, 0, 0, FrameDisposal.NONE, FrameBlend.SOURCE);
    }

    /**
     * Creates a frame with the given pixels and delay, using default offset and disposal.
     *
     * @param pixels the frame pixel data
     * @param delayMs the display duration in milliseconds
     * @return a new image frame
     */
    public static @NotNull ImageFrame of(@NotNull PixelBuffer pixels, int delayMs) {
        return new ImageFrame(pixels, delayMs, 0, 0, FrameDisposal.NONE, FrameBlend.SOURCE);
    }

    /**
     * Creates a frame with full control over all animation parameters.
     *
     * @param pixels the frame pixel data
     * @param delayMs the display duration in milliseconds
     * @param offsetX the horizontal offset within the canvas
     * @param offsetY the vertical offset within the canvas
     * @param disposal the disposal method after displaying this frame
     * @param blend the blending method when rendering this frame
     * @return a new image frame
     */
    public static @NotNull ImageFrame of(
        @NotNull PixelBuffer pixels,
        int delayMs,
        int offsetX,
        int offsetY,
        @NotNull FrameDisposal disposal,
        @NotNull FrameBlend blend
    ) {
        return new ImageFrame(pixels, delayMs, offsetX, offsetY, disposal, blend);
    }

    // --- withers ---

    /**
     * Returns a copy of this frame with the given blend method.
     *
     * @param blend the new blend method
     * @return a frame with the updated blend
     */
    public @NotNull ImageFrame withBlend(@NotNull FrameBlend blend) {
        return new ImageFrame(this.pixels, this.delayMs, this.offsetX, this.offsetY, this.disposal, blend);
    }

    /**
     * Returns a copy of this frame with the given display duration.
     *
     * @param delayMs the new display duration in milliseconds
     * @return a frame with the updated delay
     */
    public @NotNull ImageFrame withDelayMs(int delayMs) {
        return new ImageFrame(this.pixels, delayMs, this.offsetX, this.offsetY, this.disposal, this.blend);
    }

    /**
     * Returns a copy of this frame with the given disposal method.
     *
     * @param disposal the new disposal method
     * @return a frame with the updated disposal
     */
    public @NotNull ImageFrame withDisposal(@NotNull FrameDisposal disposal) {
        return new ImageFrame(this.pixels, this.delayMs, this.offsetX, this.offsetY, disposal, this.blend);
    }

    /**
     * Returns a copy of this frame with the given canvas offset.
     *
     * @param offsetX the new horizontal offset
     * @param offsetY the new vertical offset
     * @return a frame with the updated offset
     */
    public @NotNull ImageFrame withOffset(int offsetX, int offsetY) {
        return new ImageFrame(this.pixels, this.delayMs, offsetX, offsetY, this.disposal, this.blend);
    }

    /**
     * Returns a copy of this frame with the given pixel data.
     *
     * @param pixels the new pixel data
     * @return a frame with the updated pixels
     */
    public @NotNull ImageFrame withPixels(@NotNull PixelBuffer pixels) {
        return new ImageFrame(pixels, this.delayMs, this.offsetX, this.offsetY, this.disposal, this.blend);
    }

    // --- delegating accessors ---

    /**
     * Returns whether the frame's pixel data carries a meaningful alpha channel.
     *
     * @return {@code true} if alpha is present
     */
    public boolean hasAlpha() {
        return this.pixels.hasAlpha();
    }

    /**
     * Returns the frame height in pixels.
     *
     * @return the height of the underlying {@link PixelBuffer}
     */
    public int height() {
        return this.pixels.height();
    }

    /**
     * Returns the frame width in pixels.
     *
     * @return the width of the underlying {@link PixelBuffer}
     */
    public int width() {
        return this.pixels.width();
    }

}
