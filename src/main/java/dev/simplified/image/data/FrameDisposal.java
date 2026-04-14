package dev.simplified.image.data;

import dev.simplified.collection.Concurrent;
import dev.simplified.collection.ConcurrentMap;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.jetbrains.annotations.NotNull;

import java.util.Locale;

/**
 * The action to take with the canvas after a frame is displayed.
 */
@Getter
@RequiredArgsConstructor
public enum FrameDisposal {

    /** No disposal action specified. */
    NONE(0, "none"),

    /** Leave the canvas as-is after displaying this frame. */
    DO_NOT_DISPOSE(1, "doNotDispose"),

    /** Restore the canvas to the background color. */
    RESTORE_TO_BACKGROUND(2, "restoreToBackgroundColor"),

    /** Restore the canvas to its state before this frame was rendered. */
    RESTORE_TO_PREVIOUS(3, "restoreToPrevious");

    private static final @NotNull FrameDisposal[] BY_VALUE;
    private static final @NotNull ConcurrentMap<String, FrameDisposal> BY_METHOD;

    static {
        FrameDisposal[] values = values();
        int max = 0;
        for (FrameDisposal d : values) max = Math.max(max, d.value);
        BY_VALUE = new FrameDisposal[max + 1];
        for (FrameDisposal d : values) BY_VALUE[d.value] = d;

        BY_METHOD = Concurrent.newMap();
        for (FrameDisposal d : values) BY_METHOD.put(d.method.toLowerCase(Locale.ROOT), d);
    }

    private final int value;
    private final @NotNull String method;

    /**
     * Returns the disposal method for the given numeric value.
     *
     * @param value the disposal method identifier
     * @return the matching disposal method, or {@link #NONE} if unrecognized
     */
    public static @NotNull FrameDisposal of(int value) {
        if (value < 0 || value >= BY_VALUE.length) return NONE;
        return BY_VALUE[value];
    }

    /**
     * Returns the disposal method for the given string value.
     *
     * @param value the disposal method identifier
     * @return the matching disposal method, or {@link #NONE} if unrecognized
     */
    public static @NotNull FrameDisposal of(@NotNull String value) {
        return BY_METHOD.getOrDefault(value.toLowerCase(Locale.ROOT), NONE);
    }

}
