package dev.simplified.image.transform;

/**
 * Defines how animation frames are fitted onto a target canvas during normalization.
 * <p>
 * These modes control how source frames of differing dimensions are scaled
 * and positioned onto a uniform canvas size.
 */
public enum FitMode {

    /**
     * Scale the frame to fit entirely within the canvas while preserving aspect ratio.
     * <p>
     * The entire source frame will be visible on the canvas. If the aspect ratios
     * differ, the background color fills the remaining space (letterboxing).
     */
    CONTAIN,

    /**
     * Scale the frame to completely fill the canvas while preserving aspect ratio.
     * <p>
     * The canvas will be entirely covered by the frame. If the aspect ratios
     * differ, parts of the source frame extending beyond the canvas are cropped.
     */
    COVER,

    /**
     * Stretch the frame to exactly match the canvas dimensions.
     * <p>
     * The frame is stretched or compressed to fill the entire canvas,
     * potentially distorting the aspect ratio.
     */
    STRETCH

}
