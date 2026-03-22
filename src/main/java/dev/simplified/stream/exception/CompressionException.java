package dev.sbs.api.io.stream.exception;

import dev.sbs.api.io.exception.IoException;
import org.intellij.lang.annotations.PrintFormat;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

/**
 * Thrown when the compression layer encounters an encoding or decoding error.
 */
public class CompressionException extends IoException {

    /**
     * Constructs a new {@code CompressionException} with the specified cause.
     *
     * @param cause the underlying throwable that caused this exception
     */
    public CompressionException(@NotNull Throwable cause) {
        super(cause);
    }

    /**
     * Constructs a new {@code CompressionException} with the specified detail message.
     *
     * @param message the detail message
     */
    public CompressionException(@NotNull String message) {
        super(message);
    }

    /**
     * Constructs a new {@code CompressionException} with the specified cause and detail message.
     *
     * @param cause the underlying throwable that caused this exception
     * @param message the detail message
     */
    public CompressionException(@NotNull Throwable cause, @NotNull String message) {
        super(cause, message);
    }

    /**
     * Constructs a new {@code CompressionException} with a formatted detail message.
     *
     * @param message the format string
     * @param args the format arguments
     */
    public CompressionException(@NotNull @PrintFormat String message, @Nullable Object... args) {
        super(message, args);
    }

    /**
     * Constructs a new {@code CompressionException} with the specified cause and a formatted detail message.
     *
     * @param cause the underlying throwable that caused this exception
     * @param message the format string
     * @param args the format arguments
     */
    public CompressionException(@NotNull Throwable cause, @NotNull @PrintFormat String message, @Nullable Object... args) {
        super(cause, message, args);
    }

}
