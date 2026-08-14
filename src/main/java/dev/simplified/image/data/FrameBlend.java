package dev.simplified.image.data;

import dev.simplified.annotations.EnumLookup;
import dev.simplified.annotations.Getter;
import dev.simplified.annotations.KeyField;
import dev.simplified.annotations.RequiredArgsConstructor;
import org.jetbrains.annotations.NotNull;

/**
 * How a frame is composited onto the canvas when rendered.
 */
@Getter
@EnumLookup
@RequiredArgsConstructor
public enum FrameBlend {

    /**
     * Replace the canvas region with this frame's pixels.
     */
    SOURCE(0),

    /**
     * Alpha-blend this frame over the existing canvas content.
     */
    OVER(1);

    @KeyField(strictKeys = true)
    private final int value;

    /**
     * Returns the blend mode for the given numeric value.
     *
     * @param value the blend mode identifier
     * @return the matching blend mode, or {@link #SOURCE} if unrecognized
     */
    public static @NotNull FrameBlend of(int value) {
        return findByValue(value).orElse(SOURCE);
    }

}
