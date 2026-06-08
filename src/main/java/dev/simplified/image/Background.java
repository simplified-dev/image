package dev.simplified.image;

import dev.simplified.image.data.AnimatedImageData;
import dev.simplified.image.data.ImageFrame;
import dev.simplified.image.data.StaticImageData;
import dev.simplified.image.pixel.PixelBuffer;
import org.jetbrains.annotations.NotNull;

/**
 * A canvas background fill composited behind a finished render. Either a single solid ARGB colour
 * or a repeating two-colour checkerboard (the checkerboard makes the transparent extent of a
 * render visible without an external image editor).
 * <p>
 * {@link #composite(ImageData)} applies the fill as the very last step - the finished frame is
 * blitted over a freshly filled background buffer so source-over alpha stays correct. A
 * {@link #TRANSPARENT} background is a no-op and leaves output byte-identical to the input.
 */
public sealed interface Background permits Background.Solid, Background.Checkerboard {

    /**
     * A fully transparent background. Compositing against it is a no-op.
     */
    @NotNull Background TRANSPARENT = solid(0x00000000);

    /**
     * Returns a solid single-colour background.
     *
     * @param argb the fill colour
     * @return a solid background
     */
    static @NotNull Background solid(int argb) {
        return new Solid(argb);
    }

    /**
     * Returns the default checkerboard - a dark 8-pixel grid.
     *
     * @return a checkerboard background
     */
    static @NotNull Background checkerboard() {
        return new Checkerboard(0xFF202024, 0xFF181820, 8);
    }

    /**
     * Returns a checkerboard with explicit colours and cell size.
     *
     * @param argb0 the colour of the cell at the top-left origin
     * @param argb1 the alternating colour
     * @param cellSize the square cell edge length in pixels
     * @return a checkerboard background
     */
    static @NotNull Background checkerboard(int argb0, int argb1, int cellSize) {
        return new Checkerboard(argb0, argb1, cellSize);
    }

    /**
     * Whether this background contributes nothing, letting callers skip compositing entirely. True
     * only for a fully transparent solid fill.
     *
     * @return {@code true} if compositing against this background is a no-op
     */
    boolean isTransparent();

    /**
     * Fills {@code target} with this background across its full extent.
     *
     * @param target the buffer to fill in place
     */
    void fill(@NotNull PixelBuffer target);

    /**
     * Composites {@code image} over this background, applied as the very last step so source-over
     * alpha stays correct. Each frame is blitted onto a freshly filled background buffer of the
     * same size; per-frame timing and metadata are preserved. A {@linkplain #isTransparent()
     * transparent} background returns {@code image} unchanged.
     *
     * @param image the finished render
     * @return the backgrounded image, or {@code image} unchanged when this background is transparent
     */
    default @NotNull ImageData composite(@NotNull ImageData image) {
        if (isTransparent()) return image;

        if (!image.isAnimated())
            return StaticImageData.of(over(image.getFrames().getFirst().pixels()));

        AnimatedImageData.Builder builder = AnimatedImageData.builder();
        if (image instanceof AnimatedImageData animated)
            builder.withLoopCount(animated.getLoopCount()).withBackgroundColor(animated.getBackgroundColor());
        for (ImageFrame frame : image.getFrames())
            builder.withFrame(frame.withPixels(over(frame.pixels())));

        return builder.build();
    }

    /**
     * Returns a new buffer the size of {@code rendered} with this background filled underneath and
     * {@code rendered} composited on top (source-over).
     */
    private @NotNull PixelBuffer over(@NotNull PixelBuffer rendered) {
        PixelBuffer out = PixelBuffer.create(rendered.width(), rendered.height());
        this.fill(out);
        out.blit(rendered, 0, 0);
        return out;
    }

    /**
     * A single solid ARGB fill.
     *
     * @param argb the fill colour, fully transparent when its alpha byte is zero
     */
    record Solid(int argb) implements Background {

        @Override
        public boolean isTransparent() {
            return (this.argb >>> 24) == 0;
        }

        @Override
        public void fill(@NotNull PixelBuffer target) {
            target.fill(this.argb);
        }

    }

    /**
     * A repeating two-colour checkerboard anchored at the top-left origin.
     *
     * @param argb0 the colour of the cell at the top-left origin
     * @param argb1 the alternating colour
     * @param cellSize the square cell edge length in pixels
     */
    record Checkerboard(int argb0, int argb1, int cellSize) implements Background {

        @Override
        public boolean isTransparent() {
            return false;
        }

        @Override
        public void fill(@NotNull PixelBuffer target) {
            int size = Math.max(1, this.cellSize);

            for (int y = 0; y < target.height(); y++) {
                for (int x = 0; x < target.width(); x++)
                    target.setPixel(x, y, ((x / size) + (y / size)) % 2 == 0 ? this.argb0 : this.argb1);
            }
        }

    }

}
