package dev.sbs.api.io.exception;

import org.intellij.lang.annotations.PrintFormat;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

/**
 * Thrown when the I/O layer encounters a stream, file, or serialization error.
 */
public class IoException extends RuntimeException {

    /**
     * Constructs a new {@code IoException} with the specified cause.
     *
     * @param cause the underlying throwable that caused this exception
     */
    public IoException(@NotNull Throwable cause) {
        super(cause);
    }

    /**
     * Constructs a new {@code IoException} with the specified detail message.
     *
     * @param message the detail message
     */
    public IoException(@NotNull String message) {
        super(message);
    }

    /**
     * Constructs a new {@code IoException} with the specified cause and detail message.
     *
     * @param cause the underlying throwable that caused this exception
     * @param message the detail message
     */
    public IoException(@NotNull Throwable cause, @NotNull String message) {
        super(message, cause);
    }

    /**
     * Constructs a new {@code IoException} with a formatted detail message.
     *
     * @param message the format string
     * @param args the format arguments
     */
    public IoException(@NotNull @PrintFormat String message, @Nullable Object... args) {
        super(String.format(message, args));
    }

    /**
     * Constructs a new {@code IoException} with the specified cause and a formatted detail message.
     *
     * @param cause the underlying throwable that caused this exception
     * @param message the format string
     * @param args the format arguments
     */
    public IoException(@NotNull Throwable cause, @NotNull @PrintFormat String message, @Nullable Object... args) {
        super(String.format(message, args), cause);
    }

}
