package dev.sbs.api.io.exception;

import org.intellij.lang.annotations.PrintFormat;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

public class IoException extends RuntimeException {

    public IoException(@NotNull Throwable cause) {
        super(cause);
    }

    public IoException(@NotNull String message) {
        super(message);
    }

    public IoException(@NotNull Throwable cause, @NotNull String message) {
        super(message, cause);
    }

    public IoException(@NotNull @PrintFormat String message, @Nullable Object... args) {
        super(String.format(message, args));
    }

    public IoException(@NotNull Throwable cause, @NotNull @PrintFormat String message, @Nullable Object... args) {
        super(String.format(message, args), cause);
    }

}
