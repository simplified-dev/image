package dev.simplified.image.codec.common;

import dev.simplified.image.exception.ImageTooLargeException;
import org.jetbrains.annotations.ApiStatus;
import org.jetbrains.annotations.NotNull;

/**
 * The most pixels one read may hold, and the tally a read keeps against it.
 * <p>
 * The limit is process-wide: every reader opens a budget when it starts, which fixes the limit for
 * that read, and claims each frame, page or canvas against it before allocating it, so a header that
 * asks for more is refused with {@link ImageTooLargeException} before any memory is taken. A read's
 * claims add up across what it builds the image from - each frame of an animation, each page of a
 * TIFF, each canvas an animated WebP is composed onto - and a codec decoding one frame, VP8 or VP8L,
 * claims that frame on a budget of its own. A canvas a reader declares without allocating is held to
 * the limit on its own, since whatever composes onto it later allocates it whole.
 */
@ApiStatus.Internal
public final class DecodeBudget {

    /** The limit a read holds to until one is set: 8192 by 8192 pixels, 256 MiB of ARGB. */
    public static final long DEFAULT_LIMIT = 8192L * 8192L;

    /** The limit every read opened from now on holds to. */
    private static volatile long limit = DEFAULT_LIMIT;

    /** The format named when a claim is refused. */
    private final @NotNull String format;

    /** The limit fixed for this read when it opened. */
    private final long readLimit;

    /** The pixels claimed so far. */
    private long held;

    private DecodeBudget(@NotNull String format, long readLimit) {
        this.format = format;
        this.readLimit = readLimit;
    }

    /**
     * Returns the limit every read opened from now on holds to.
     *
     * @return the most pixels one read may hold
     */
    public static long limit() {
        return limit;
    }

    /**
     * Sets the limit every read opened from now on holds to; a read already under way keeps the one it
     * opened with.
     *
     * @param pixels the most pixels one read may hold
     * @throws IllegalArgumentException if {@code pixels} is not positive
     */
    public static void setLimit(long pixels) {
        if (pixels <= 0)
            throw new IllegalArgumentException(String.format("Pixel budget '%d' is not positive", pixels));

        limit = pixels;
    }

    /**
     * Opens a budget for one read at the current limit.
     *
     * @param format the format being read, named when a claim is refused
     * @return a budget holding nothing yet
     */
    public static @NotNull DecodeBudget open(@NotNull String format) {
        return new DecodeBudget(format, limit);
    }

    /**
     * Claims one buffer of {@code width} by {@code height} pixels.
     *
     * @param width the buffer's width
     * @param height the buffer's height
     * @return this budget
     * @throws ImageTooLargeException if the read would then hold more than its limit
     */
    public @NotNull DecodeBudget claim(long width, long height) {
        return claim(width, height, 1);
    }

    /**
     * Claims {@code count} buffers of {@code width} by {@code height} pixels each.
     *
     * @param width each buffer's width
     * @param height each buffer's height
     * @param count how many such buffers
     * @return this budget
     * @throws ImageTooLargeException if the read would then hold more than its limit
     */
    public @NotNull DecodeBudget claim(long width, long height, long count) {
        long room = this.readLimit - this.held;
        long each = width * height;

        if (each > room || (count > 1 && each > 0 && count > room / each))
            throw new ImageTooLargeException(
                "%s needs '%dx%d' pixels%s with '%d' already held, past its '%d'-pixel budget",
                this.format, width, height, count > 1 ? " '" + count + "' times" : "", this.held, this.readLimit);

        this.held += each * count;
        return this;
    }

    /**
     * Holds a canvas the read declares, without allocating it, to the limit on its own.
     *
     * @param width the canvas width
     * @param height the canvas height
     * @return this budget
     * @throws ImageTooLargeException if the canvas alone holds more than the limit
     */
    public @NotNull DecodeBudget declare(long width, long height) {
        if (width * height > this.readLimit)
            throw new ImageTooLargeException(
                "%s declares a '%dx%d' canvas, past its '%d'-pixel budget",
                this.format, width, height, this.readLimit);

        return this;
    }

}
