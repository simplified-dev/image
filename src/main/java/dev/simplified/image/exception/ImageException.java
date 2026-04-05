package dev.simplified.image.exception;

import org.intellij.lang.annotations.PrintFormat;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

/**
 * Thrown when the image pipeline encounters a read, write, or conversion error.
 */
public class ImageException extends RuntimeException {

    /**
     * Constructs a new {@code ImageException} with the specified cause.
     *
     * @param cause the underlying throwable that caused this exception
     */
    public ImageException(@NotNull Throwable cause) {
        super(cause);
    }

    /**
     * Constructs a new {@code ImageException} with the specified detail message.
     *
     * @param message the detail message
     */
    public ImageException(@NotNull String message) {
        super(message);
    }

    /**
     * Constructs a new {@code ImageException} with the specified cause and detail message.
     *
     * @param cause the underlying throwable that caused this exception
     * @param message the detail message
     */
    public ImageException(@NotNull Throwable cause, @NotNull String message) {
        super(message, cause);
    }

    /**
     * Constructs a new {@code ImageException} with a formatted detail message.
     *
     * @param message the format string
     * @param args the format arguments
     */
    public ImageException(@NotNull @PrintFormat String message, @Nullable Object... args) {
        super(String.format(message, args));
    }

    /**
     * Constructs a new {@code ImageException} with the specified cause and a formatted detail message.
     *
     * @param cause the underlying throwable that caused this exception
     * @param message the format string
     * @param args the format arguments
     */
    public ImageException(@NotNull Throwable cause, @NotNull @PrintFormat String message, @Nullable Object... args) {
        super(String.format(message, args), cause);
    }

}
