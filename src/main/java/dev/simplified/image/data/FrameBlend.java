package dev.simplified.image.data;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.jetbrains.annotations.NotNull;

/**
 * How a frame is composited onto the canvas when rendered.
 */
@Getter
@RequiredArgsConstructor
public enum FrameBlend {

    /** Replace the canvas region with this frame's pixels. */
    SOURCE(0),

    /** Alpha-blend this frame over the existing canvas content. */
    OVER(1);

    private static final @NotNull FrameBlend[] BY_VALUE;

    static {
        FrameBlend[] values = values();
        int max = 0;
        for (FrameBlend b : values) max = Math.max(max, b.value);
        BY_VALUE = new FrameBlend[max + 1];
        for (FrameBlend b : values) BY_VALUE[b.value] = b;
    }

    private final int value;

    /**
     * Returns the blend mode for the given numeric value.
     *
     * @param value the blend mode identifier
     * @return the matching blend mode, or {@link #SOURCE} if unrecognized
     */
    public static @NotNull FrameBlend of(int value) {
        if (value < 0 || value >= BY_VALUE.length) return SOURCE;
        return BY_VALUE[value];
    }

}
