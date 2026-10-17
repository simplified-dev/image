package dev.simplified.image.exception;

import dev.simplified.image.ImageFactory;
import org.intellij.lang.annotations.PrintFormat;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

/**
 * Thrown when data matches the magic bytes of no known image format, or when the codec set built into
 * {@link ImageFactory} holds no reader or writer for the requested format.
 */
public class UnsupportedFormatException extends ImageException {

    /**
     * Constructs a new {@code UnsupportedFormatException} with the specified cause.
     *
     * @param cause the underlying throwable that caused this exception
     */
    public UnsupportedFormatException(@NotNull Throwable cause) {
        super(cause);
    }

    /**
     * Constructs a new {@code UnsupportedFormatException} with the specified detail message.
     *
     * @param message the detail message
     */
    public UnsupportedFormatException(@NotNull String message) {
        super(message);
    }

    /**
     * Constructs a new {@code UnsupportedFormatException} with the specified cause and detail message.
     *
     * @param cause the underlying throwable that caused this exception
     * @param message the detail message
     */
    public UnsupportedFormatException(@NotNull Throwable cause, @NotNull String message) {
        super(cause, message);
    }

    /**
     * Constructs a new {@code UnsupportedFormatException} with a formatted detail message.
     *
     * @param message the format string
     * @param args the format arguments
     */
    public UnsupportedFormatException(@NotNull @PrintFormat String message, @Nullable Object... args) {
        super(message, args);
    }

    /**
     * Constructs a new {@code UnsupportedFormatException} with the specified cause and a formatted detail message.
     *
     * @param cause the underlying throwable that caused this exception
     * @param message the format string
     * @param args the format arguments
     */
    public UnsupportedFormatException(@NotNull Throwable cause, @NotNull @PrintFormat String message, @Nullable Object... args) {
        super(cause, message, args);
    }

}
