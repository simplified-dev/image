package dev.simplified.image.pixel;

import dev.simplified.image.AnimatedImageData;
import dev.simplified.image.ImageData;
import dev.simplified.image.PixelBuffer;
import dev.simplified.image.StaticImageData;
import dev.simplified.image.frame.ImageFrame;
import org.jetbrains.annotations.NotNull;

/**
 * A canvas background fill composited behind a finished render. Either a single solid ARGB colour
 * or a repeating two-colour checkerboard (the checkerboard makes the transparent extent of a
 * render visible without an external image editor).
 * <p>
 * {@link #composite(ImageData)} applies the fill as the very last step - each frame, composed onto
 * the whole canvas, is laid source-over a freshly filled canvas-sized buffer of the convention the
 * image is held in, which {@link PixelBuffer#blit(PixelBuffer, int, int)} lays by the formula for two
 * buffers of that convention: {@link BlendMode#PREMULTIPLIED}'s under an image held in
 * {@link AlphaMode#PREMULTIPLIED}, whose colours already carry their alpha, and
 * {@link BlendMode#STRAIGHT}'s under a straight one. A partially transparent pixel therefore shows the
 * fill through it as it would had the image been drawn over the fill directly. A {@link #TRANSPARENT}
 * background is a no-op and leaves output byte-identical to the input.
 * <p>
 * A fill colour is straight colour, laid in the convention of the buffer it fills as
 * {@link PixelBuffer#fill(int)} and {@link PixelBuffer#fillRect(int, int, int, int, int)} lay it. Under
 * an image held in {@link AlphaMode#PREMULTIPLIED} the fill is premultiplied, so the composite holds
 * premultiplied colour throughout and a translucent fill encodes as the colour given.
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
     * Composites {@code image} over this background, applied as the very last step. The image is first
     * {@linkplain ImageData#toCanvasFrames() composed onto whole canvases}, so the fill covers the
     * canvas outside every frame, a checkerboard is anchored at the canvas origin, and a partly
     * transparent frame shows the frames beneath it rather than the fill. Each canvas is then laid
     * source-over a freshly filled buffer of the canvas size and of the convention the canvas holds -
     * a premultiplied fill under {@link AlphaMode#PREMULTIPLIED}, the fill as given under
     * {@link AlphaMode#STRAIGHT} - and the result holds that convention. A
     * {@linkplain #isTransparent() transparent} background returns {@code image} unchanged.
     * <p>
     * An animation keeps its canvas size, each frame's timing, and - when it is an
     * {@link AnimatedImageData} - its loop count and background colour. Each frame of the result is
     * the canvas the composed frame covers, so it replaces the frame before it.
     *
     * @param image the finished render
     * @return the backgrounded image, or {@code image} unchanged when this background is transparent
     */
    default @NotNull ImageData composite(@NotNull ImageData image) {
        if (isTransparent()) return image;

        ImageData canvas = image.toCanvasFrames();

        if (!canvas.isAnimated())
            return StaticImageData.of(over(canvas.getFrames().getFirst().pixels()));

        AnimatedImageData.Builder builder = AnimatedImageData.builder()
            .withWidth(canvas.getWidth())
            .withHeight(canvas.getHeight());

        if (canvas instanceof AnimatedImageData animated)
            builder.withLoopCount(animated.getLoopCount()).withBackgroundColor(animated.getBackgroundColor());
        for (ImageFrame frame : canvas.getFrames())
            builder.withFrame(frame.withPixels(over(frame.pixels())));

        return builder.build();
    }

    /**
     * Returns a new buffer the size of {@code rendered} and of the convention it holds, with this
     * background filled underneath, laid in that convention, and {@code rendered} laid on top
     * source-over.
     *
     * @param rendered the composed canvas laid over the fill
     * @return a new buffer holding {@code rendered} over the fill, in its convention
     */
    private @NotNull PixelBuffer over(@NotNull PixelBuffer rendered) {
        PixelBuffer out = switch (rendered.alphaMode()) {
            case STRAIGHT -> PixelBuffer.create(rendered.width(), rendered.height());
            case PREMULTIPLIED -> PixelBuffer.canvas(rendered.width(), rendered.height());
        };

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

            for (int y = 0; y < target.height(); y += size) {
                for (int x = 0; x < target.width(); x += size)
                    target.fillRect(x, y, size, size, ((x / size) + (y / size)) % 2 == 0 ? this.argb0 : this.argb1);
            }
        }

    }

}
