package dev.simplified.image.exception;

import org.intellij.lang.annotations.PrintFormat;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

/**
 * Thrown when a read would hold more pixels than the pixel budget allows, before it allocates them.
 */
public class ImageTooLargeException extends ImageDecodeException {

    /**
     * Constructs a new {@code ImageTooLargeException} with the specified cause.
     *
     * @param cause the underlying throwable that caused this exception
     */
    public ImageTooLargeException(@NotNull Throwable cause) {
        super(cause);
    }

    /**
     * Constructs a new {@code ImageTooLargeException} with the specified detail message.
     *
     * @param message the detail message
     */
    public ImageTooLargeException(@NotNull String message) {
        super(message);
    }

    /**
     * Constructs a new {@code ImageTooLargeException} with the specified cause and detail message.
     *
     * @param cause the underlying throwable that caused this exception
     * @param message the detail message
     */
    public ImageTooLargeException(@NotNull Throwable cause, @NotNull String message) {
        super(cause, message);
    }

    /**
     * Constructs a new {@code ImageTooLargeException} with a formatted detail message.
     *
     * @param message the format string
     * @param args the format arguments
     */
    public ImageTooLargeException(@NotNull @PrintFormat String message, @Nullable Object... args) {
        super(message, args);
    }

    /**
     * Constructs a new {@code ImageTooLargeException} with the specified cause and a formatted detail message.
     *
     * @param cause the underlying throwable that caused this exception
     * @param message the format string
     * @param args the format arguments
     */
    public ImageTooLargeException(@NotNull Throwable cause, @NotNull @PrintFormat String message, @Nullable Object... args) {
        super(cause, message, args);
    }

}
