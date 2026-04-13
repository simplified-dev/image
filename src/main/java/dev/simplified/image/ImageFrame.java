package dev.simplified.image;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;

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
    @NotNull Disposal disposal,
    @NotNull Blend blend
) {

    /**
     * Creates a frame with the given pixels and delay, using default offset and disposal.
     *
     * @param pixels the frame pixel data
     * @param delayMs the display duration in milliseconds
     * @return a new image frame
     */
    public static @NotNull ImageFrame of(@NotNull PixelBuffer pixels, int delayMs) {
        return new ImageFrame(pixels, delayMs, 0, 0, Disposal.NONE, Blend.SOURCE);
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
        @NotNull Disposal disposal,
        @NotNull ImageFrame.Blend blend
    ) {
        return new ImageFrame(pixels, delayMs, offsetX, offsetY, disposal, blend);
    }

    /**
     * Determines the action to take with the canvas after this frame is displayed.
     */
    @Getter
    @RequiredArgsConstructor
    public enum Disposal {

        /** No disposal action specified. */
        NONE(0, "none"),
        /** Leave the canvas as-is after displaying this frame. */
        DO_NOT_DISPOSE(1, "doNotDispose"),
        /** Restore the canvas to the background color. */
        RESTORE_TO_BACKGROUND(2, "restoreToBackgroundColor"),
        /** Restore the canvas to its state before this frame was rendered. */
        RESTORE_TO_PREVIOUS(3, "restoreToPrevious");

        private final int value;
        private final @NotNull String method;

        /**
         * Returns the disposal method for the given numeric value.
         *
         * @param value the disposal method identifier
         * @return the matching disposal method, or {@link #NONE} if unrecognized
         */
        public static @NotNull Disposal of(int value) {
            return Arrays.stream(values())
                .filter(disposal -> disposal.getValue() == value)
                .findFirst()
                .orElse(NONE);
        }

        /**
         * Returns the disposal method for the given string value.
         *
         * @param value the disposal method identifier
         * @return the matching disposal method, or {@link #NONE} if unrecognized
         */
        public static @NotNull Disposal of(@NotNull String value) {
            return Arrays.stream(values())
                .filter(disposal -> disposal.getMethod().equalsIgnoreCase(value))
                .findFirst()
                .orElse(NONE);
        }

    }

    /**
     * Determines how the frame is composited onto the canvas.
     */
    @Getter
    @RequiredArgsConstructor
    public enum Blend {

        /** Replace the canvas region with this frame's pixels. */
        SOURCE(0),
        /** Alpha-blend this frame over the existing canvas content. */
        OVER(1);

        private final int value;

        /**
         * Returns the blend mode for the given numeric value.
         *
         * @param value the blend mode identifier
         * @return the matching blend mode, or {@link #SOURCE} if unrecognized
         */
        public static @NotNull Blend of(int value) {
            return Arrays.stream(values())
                .filter(blend -> blend.getValue() == value)
                .findFirst()
                .orElse(SOURCE);
        }

    }

}
