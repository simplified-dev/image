package dev.simplified.image.data;

import dev.simplified.annotations.EnumLookup;
import dev.simplified.annotations.Getter;
import dev.simplified.annotations.KeyField;
import dev.simplified.annotations.RequiredArgsConstructor;
import org.jetbrains.annotations.NotNull;

/**
 * The action to take with the canvas after a frame is displayed.
 */
@Getter
@EnumLookup
@RequiredArgsConstructor
public enum FrameDisposal {

    /**
     * No disposal action specified.
     */
    NONE(0, "none"),

    /**
     * Leave the canvas as-is after displaying this frame.
     */
    DO_NOT_DISPOSE(1, "doNotDispose"),

    /**
     * Restore the canvas to the background color.
     */
    RESTORE_TO_BACKGROUND(2, "restoreToBackgroundColor"),

    /**
     * Restore the canvas to its state before this frame was rendered.
     */
    RESTORE_TO_PREVIOUS(3, "restoreToPrevious");

    @KeyField(strictKeys = true)
    private final int value;
    @KeyField(strictKeys = true, ignoreCase = true)
    private final @NotNull String method;

    /**
     * Returns the disposal method for the given numeric value.
     *
     * @param value the disposal method identifier
     * @return the matching disposal method, or {@link #NONE} if unrecognized
     */
    public static @NotNull FrameDisposal of(int value) {
        return findByValue(value).orElse(NONE);
    }

    /**
     * Returns the disposal method for the given string value, ignoring case.
     *
     * @param value the disposal method identifier
     * @return the matching disposal method, or {@link #NONE} if unrecognized
     */
    public static @NotNull FrameDisposal of(@NotNull String value) {
        return findByMethod(value).orElse(NONE);
    }

}
