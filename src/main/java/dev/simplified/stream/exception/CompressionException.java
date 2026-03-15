package dev.sbs.api.io.stream.exception;

import dev.sbs.api.io.exception.IoException;
import org.intellij.lang.annotations.PrintFormat;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

public class CompressionException extends IoException {

    public CompressionException(@NotNull Throwable cause) {
        super(cause);
    }

    public CompressionException(@NotNull String message) {
        super(message);
    }

    public CompressionException(@NotNull Throwable cause, @NotNull String message) {
        super(cause, message);
    }

    public CompressionException(@NotNull @PrintFormat String message, @Nullable Object... args) {
        super(message, args);
    }

    public CompressionException(@NotNull Throwable cause, @NotNull @PrintFormat String message, @Nullable Object... args) {
        super(cause, message, args);
    }

}
