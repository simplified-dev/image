package dev.simplified.image;

import dev.simplified.annotations.EqualsAndHashCode;
import dev.simplified.annotations.Getter;
import dev.simplified.annotations.NamingStyle;
import dev.simplified.image.pixel.AlphaMode;
import dev.simplified.image.pixel.BlendMode;
import dev.simplified.image.pixel.ColorMath;
import dev.simplified.image.pixel.DiffType;
import dev.simplified.image.pixel.PixelDiff;
import dev.simplified.image.pixel.PixelGraphics;
import dev.simplified.image.pixel.PixelMask;
import dev.simplified.image.pixel.Resample;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.awt.*;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.ComponentColorModel;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferInt;
import java.awt.image.DirectColorModel;
import java.awt.image.ImageObserver;
import java.awt.image.Raster;
import java.awt.image.SinglePixelPackedSampleModel;
import java.util.Arrays;
import java.util.Optional;
import java.util.concurrent.ForkJoinTask;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;

/**
 * A zero-copy wrapper over {@code int[]} ARGB pixel data.
 * <p>
 * {@link #wrap(BufferedImage)} references a {@link BufferedImage#TYPE_INT_ARGB} image's backing array
 * directly without copying; for other image types, pixel data is extracted once into a new array, then
 * accessed without further copies. {@link #toBufferedImage()} shares this buffer's array the other way,
 * as an image of the convention the buffer holds.
 *
 * <h2>Alpha convention</h2>
 * <p>
 * Every buffer holds its pixels in one {@link AlphaMode}, fixed by the factory that makes it and
 * reported by {@code alphaMode()}: {@link #create(int, int)}, {@link #ofStraight(int[], int, int)} and
 * {@link #wrap(BufferedImage)} answer {@link AlphaMode#STRAIGHT} colour, what every file format stores,
 * and {@link #canvas(int, int)} and {@link #ofPremultiplied(int[], int, int)} answer
 * {@link AlphaMode#PREMULTIPLIED} colour, what drawing source-over onto a transparent surface leaves.
 * {@link #toAlphaMode(AlphaMode)} converts to the other one, and equality counts it: the same ints held
 * in two conventions are two pictures.
 * <p>
 * An operation that reads one buffer reads its convention - {@link #resize}, {@link #applyFxaa()},
 * {@link #multiplyAlpha}, {@link #invert()} - and a buffer derived from another holds the convention
 * of the one it is derived from. The composites and the colour fills read none: the blend mode states
 * how a source or a colour and each pixel are read. The raw reads and writes - {@link #getPixel},
 * {@link #setPixel}, {@link #setPixels}, {@link #setRow} and {@code data()} - pass the stored ints as
 * they are, so a caller writing through them writes in the buffer's convention.
 *
 * <h2>SIMD acceleration</h2>
 * <p>
 * Selected per-pixel methods ({@link #grayscale}, {@link #multiplyAlpha} on straight colour,
 * {@link #toAlphaMode} converting to premultiplied colour, {@link #lerp}, and {@link #tint})
 * transparently use the JDK Vector API when the incubator module is resolvable at runtime. To opt in,
 * pass the following flag to the consuming JVM:
 * <pre><code>--add-modules jdk.incubator.vector</code></pre>
 * Without the flag, those methods fall back to a bit-equivalent scalar path - no other action is
 * required to use this library.
 */
@Getter(style = NamingStyle.FLUENT)
@EqualsAndHashCode(identity = EqualsAndHashCode.Identity.INSTANCE_OF, exclude = "mask")
public class PixelBuffer {

    /**
     * Minimum row count before a row loop splits its rows across a fork-join pool. Below this, the
     * fork-join overhead dominates the per-row work. {@link #forEachRow} splits only when called off
     * every pool, and called on a fork-join worker runs its rows on that worker whatever their count;
     * {@link #forEachRowAcrossPool} splits wherever it is called.
     */
    private static final int MIN_PARALLEL_ROWS = 64;

    /**
     * The value {@link #fxaaPixel} answers for a pixel FXAA leaves as it is, outside the range every
     * packed pixel falls in.
     */
    private static final long UNFILTERED = Long.MIN_VALUE;

    /** Buffer width in pixels. */
    private final int width;

    /** Buffer height in pixels. */
    private final int height;

    /** Whether the alpha channel carries meaningful data. */
    private final boolean hasAlpha;

    /** Packed ARGB pixel data in row-major order, of length {@code width * height}. */
    private final int @NotNull [] data;

    /** The convention every pixel of {@link #data} is held in, fixed when the buffer is made. */
    private final @NotNull AlphaMode alphaMode;

    /**
     * Optional per-pixel coverage mask owned by this buffer - auxiliary render metadata, excluded from
     * value identity ({@link #equals}/{@link #hashCode}) and never carried by {@link #copy()}. Absent
     * ({@link Optional#empty()}) means no mask; present but unmarked and present-with-marks are
     * distinct states the consumer reads through {@link #mask()}.
     */
    private @NotNull Optional<PixelMask> mask = Optional.empty();

    /**
     * Constructs a buffer over {@code data}, held in {@code alphaMode}. Every caller outside this
     * package makes one through a factory named for the convention it answers.
     *
     * @param data the packed ARGB pixels, not copied
     * @param width the buffer width in pixels
     * @param height the buffer height in pixels
     * @param hasAlpha whether the alpha channel carries meaningful data
     * @param alphaMode the convention every pixel of {@code data} is held in
     * @throws IllegalArgumentException if the length of {@code data} is not {@code width * height}
     */
    PixelBuffer(int @NotNull [] data, int width, int height, boolean hasAlpha, @NotNull AlphaMode alphaMode) {
        if (data.length != (long) width * height)
            throw new IllegalArgumentException("Pixel array length %d does not match dimensions %dx%d".formatted(data.length, width, height));

        this.data = data;
        this.width = width;
        this.height = height;
        this.hasAlpha = hasAlpha;
        this.alphaMode = alphaMode;
    }

    /**
     * Creates a blank pixel buffer of the given dimensions holding {@link AlphaMode#STRAIGHT} colour,
     * initially filled with transparent black. No {@link BufferedImage} is allocated.
     * <p>
     * A buffer drawn onto from transparency with a source-over blend holds premultiplied colour, and is
     * made with {@link #canvas(int, int)} instead.
     *
     * @param width the buffer width in pixels
     * @param height the buffer height in pixels
     * @return a new pixel buffer
     * @throws ArithmeticException if {@code width * height} overflows an int
     */
    public static @NotNull PixelBuffer create(int width, int height) {
        return new PixelBuffer(new int[Math.multiplyExact(width, height)], width, height, true, AlphaMode.STRAIGHT);
    }

    /**
     * Creates a blank canvas of the given dimensions - a pixel buffer holding
     * {@link AlphaMode#PREMULTIPLIED} colour, initially filled with transparent black, what drawing
     * source-over onto it from transparency leaves. No {@link BufferedImage} is allocated.
     *
     * @param width the canvas width in pixels
     * @param height the canvas height in pixels
     * @return a new canvas
     * @throws ArithmeticException if {@code width * height} overflows an int
     */
    public static @NotNull PixelBuffer canvas(int width, int height) {
        return new PixelBuffer(new int[Math.multiplyExact(width, height)], width, height, true, AlphaMode.PREMULTIPLIED);
    }

    /**
     * Wraps an existing ARGB pixel array holding {@link AlphaMode#STRAIGHT} colour, assuming alpha is
     * present.
     *
     * @param pixels the straight ARGB pixel data (not copied)
     * @param width the image width
     * @param height the image height
     * @return a pixel buffer wrapping the array
     * @throws IllegalArgumentException if the length of {@code pixels} is not {@code width * height}
     */
    public static @NotNull PixelBuffer ofStraight(int @NotNull [] pixels, int width, int height) {
        return new PixelBuffer(pixels, width, height, true, AlphaMode.STRAIGHT);
    }

    /**
     * Wraps an existing ARGB pixel array holding {@link AlphaMode#STRAIGHT} colour.
     *
     * @param pixels the straight ARGB pixel data (not copied)
     * @param width the image width
     * @param height the image height
     * @param hasAlpha whether the alpha channel carries meaningful data
     * @return a pixel buffer wrapping the array
     * @throws IllegalArgumentException if the length of {@code pixels} is not {@code width * height}
     */
    public static @NotNull PixelBuffer ofStraight(int @NotNull [] pixels, int width, int height, boolean hasAlpha) {
        return new PixelBuffer(pixels, width, height, hasAlpha, AlphaMode.STRAIGHT);
    }

    /**
     * Wraps an existing ARGB pixel array holding {@link AlphaMode#PREMULTIPLIED} colour, alpha present.
     * <p>
     * A buffer whose alpha carries no data holds the same colour in either convention; one is wrapped
     * with {@link #ofStraight(int[], int, int, boolean)}.
     *
     * @param pixels the premultiplied ARGB pixel data (not copied)
     * @param width the image width
     * @param height the image height
     * @return a pixel buffer wrapping the array
     * @throws IllegalArgumentException if the length of {@code pixels} is not {@code width * height}
     */
    public static @NotNull PixelBuffer ofPremultiplied(int @NotNull [] pixels, int width, int height) {
        return new PixelBuffer(pixels, width, height, true, AlphaMode.PREMULTIPLIED);
    }

    /**
     * Wraps the pixel data of a {@link BufferedImage}, holding {@link AlphaMode#STRAIGHT} colour
     * whatever the source's layout.
     * <p>
     * {@link BufferedImage#TYPE_INT_ARGB} is the only layout that shares memory with this buffer's
     * {@code int[]} ARGB store; for that type the backing array is referenced directly (zero-copy).
     * Every other standard sRGB layout - {@link BufferedImage#TYPE_INT_RGB TYPE_INT_RGB},
     * {@link BufferedImage#TYPE_INT_BGR TYPE_INT_BGR}, {@link BufferedImage#TYPE_4BYTE_ABGR
     * TYPE_4BYTE_ABGR}, {@link BufferedImage#TYPE_3BYTE_BGR TYPE_3BYTE_BGR} - is repacked into
     * ARGB with no colour-space transform, giving a bit-perfect copy of every channel.
     * <p>
     * The two premultiplied layouts, {@link BufferedImage#TYPE_INT_ARGB_PRE TYPE_INT_ARGB_PRE} and
     * {@link BufferedImage#TYPE_4BYTE_ABGR_PRE TYPE_4BYTE_ABGR_PRE}, hold each colour already scaled
     * by its alpha. They are copied - the straight colour cannot share the source's array - and each
     * channel is divided back by its alpha as {@link #toAlphaMode(AlphaMode)} divides it converting to
     * straight colour, so converting the result back to premultiplied colour gives every source sample
     * exactly.
     * <p>
     * Indexed and grayscale layouts - {@link BufferedImage#TYPE_BYTE_INDEXED TYPE_BYTE_INDEXED},
     * {@link BufferedImage#TYPE_BYTE_GRAY TYPE_BYTE_GRAY}, plus the 2-band gray+alpha
     * (tRNS-keyed grayscale) variant that lands as {@link BufferedImage#TYPE_CUSTOM TYPE_CUSTOM}
     * with a {@link ComponentColorModel} of {@link ColorSpace#TYPE_GRAY} - read raster samples
     * directly rather than going through
     * {@link Graphics2D#drawImage(Image, int, int, ImageObserver)}. The compositor applies an
     * sRGB-gamma transform that inflates raw byte values (a calibrated-gray byte of {@code 170}
     * becomes {@code 213} once sampled through {@code drawImage} or
     * {@link BufferedImage#getRGB(int, int) getRGB}); bypassing the compositor preserves the
     * raw byte exactly, which is what a GL texture sampler reads from the source PNG.
     * <p>
     * All other types (16-bit RGB, custom colour models, premultiplied ones included, ...) fall
     * back to {@code Graphics2D.drawImage} for an sRGB conversion and may lose up to one
     * least-significant bit per channel at partial alpha due to the compositor's
     * premultiply/un-premultiply roundtrip.
     *
     * @param image the source image
     * @return a pixel buffer wrapping the image data
     */
    public static @NotNull PixelBuffer wrap(@NotNull BufferedImage image) {
        int w = image.getWidth();
        int h = image.getHeight();
        boolean alpha = image.getColorModel().hasAlpha();

        return switch (image.getType()) {
            case BufferedImage.TYPE_INT_ARGB       -> BufferedImageUnpacker.intArgb(image, w, h, alpha);
            case BufferedImage.TYPE_INT_ARGB_PRE   -> BufferedImageUnpacker.intArgbPre(image, w, h, alpha);
            case BufferedImage.TYPE_INT_RGB        -> BufferedImageUnpacker.intRgb(image, w, h, alpha);
            case BufferedImage.TYPE_INT_BGR        -> BufferedImageUnpacker.intBgr(image, w, h, alpha);
            case BufferedImage.TYPE_4BYTE_ABGR     -> BufferedImageUnpacker.fourByteAbgr(image, w, h, alpha);
            case BufferedImage.TYPE_4BYTE_ABGR_PRE -> BufferedImageUnpacker.fourByteAbgrPre(image, w, h, alpha);
            case BufferedImage.TYPE_3BYTE_BGR      -> BufferedImageUnpacker.threeByteBgr(image, w, h, alpha);
            case BufferedImage.TYPE_BYTE_INDEXED   -> BufferedImageUnpacker.indexed(image, w, h, alpha);
            case BufferedImage.TYPE_BYTE_GRAY      -> BufferedImageUnpacker.grayscale(image, w, h, alpha);
            default                                -> BufferedImageUnpacker.byColorModel(image, w, h, alpha);
        };
    }

    // --- bounds queries ---

    /**
     * Returns whether the given coordinates fall within the buffer.
     *
     * @param x the column index
     * @param y the row index
     * @return {@code true} if {@code (x, y)} lies inside the buffer
     */
    public boolean contains(int x, int y) {
        return x >= 0 && y >= 0 && x < this.width && y < this.height;
    }

    /**
     * Computes the tightest axis-aligned rectangle containing every non-transparent pixel.
     * <p>
     * Returns a zero-size rectangle when the buffer is fully transparent.
     *
     * @return the bounding rectangle of opaque pixels
     */
    public @NotNull Rectangle opaqueBounds() {
        int minX = this.width, minY = this.height, maxX = -1, maxY = -1;
        for (int y = 0; y < this.height; y++) {
            int row = y * this.width;
            for (int x = 0; x < this.width; x++) {
                if (((this.data[row + x] >>> 24) & 0xFF) == 0) continue;
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
        if (maxX < 0) return new Rectangle(0, 0, 0, 0);
        return new Rectangle(minX, minY, maxX - minX + 1, maxY - minY + 1);
    }

    // --- pixel I/O ---

    /**
     * Returns the ARGB value at the given coordinates.
     *
     * @param x the column index
     * @param y the row index
     * @return the packed ARGB pixel value
     */
    public int getPixel(int x, int y) {
        return this.data[y * this.width + x];
    }

    /**
     * Copies a rectangular region of pixels into a target array using the given stride layout.
     *
     * @param x the left edge of the region
     * @param y the top edge of the region
     * @param w the region width
     * @param h the region height
     * @param target the destination array, or {@code null} to allocate a tightly packed result
     * @param offset the offset into {@code target} of the first written pixel
     * @param stride the number of {@code int}s between consecutive rows in {@code target}
     * @return {@code target} if provided, otherwise the newly allocated array
     */
    public int @NotNull [] getPixels(int x, int y, int w, int h, int @Nullable [] target, int offset, int stride) {
        int[] out = target != null ? target : new int[w * h];
        int outStride = target != null ? stride : w;
        for (int row = 0; row < h; row++)
            System.arraycopy(this.data, (y + row) * this.width + x, out, offset + row * outStride, w);
        return out;
    }

    /**
     * Returns a newly-allocated copy of the pixels in row {@code y}.
     *
     * @param y the row index
     * @return a new array containing the row's pixels
     */
    public int @NotNull [] getRow(int y) {
        int[] row = new int[this.width];
        System.arraycopy(this.data, y * this.width, row, 0, this.width);
        return row;
    }

    /**
     * Sets the ARGB value at the given coordinates.
     *
     * @param x the column index
     * @param y the row index
     * @param argb the packed ARGB pixel value
     */
    public void setPixel(int x, int y, int argb) {
        this.data[y * this.width + x] = argb;
    }

    /**
     * Writes a rectangular region of pixels from a source array using the given stride layout.
     *
     * @param x the left edge of the region
     * @param y the top edge of the region
     * @param w the region width
     * @param h the region height
     * @param src the source pixel array
     * @param offset the offset into {@code src} of the first read pixel
     * @param stride the number of {@code int}s between consecutive rows in {@code src}
     */
    public void setPixels(int x, int y, int w, int h, int @NotNull [] src, int offset, int stride) {
        for (int row = 0; row < h; row++)
            System.arraycopy(src, offset + row * stride, this.data, (y + row) * this.width + x, w);
    }

    /**
     * Creates a {@link PixelGraphics} context for this buffer. The returned context draws
     * directly into this buffer's pixel array with no AWT rendering pipeline.
     *
     * @return a new graphics context backed by this buffer
     */
    public @NotNull PixelGraphics createGraphics() {
        return new PixelGraphics(this);
    }

    /**
     * Overwrites row {@code y} with the given pixels.
     *
     * @param y the row index
     * @param row the pixels to copy; must have length {@link #width()}
     */
    public void setRow(int y, int @NotNull [] row) {
        System.arraycopy(row, 0, this.data, y * this.width, this.width);
    }

    // --- geometry ---

    /**
     * Returns a new pixel buffer containing a copy of the given sub-region.
     *
     * @param x the left edge of the region
     * @param y the top edge of the region
     * @param w the region width
     * @param h the region height
     * @return a new pixel buffer of size {@code w x h}
     * @throws IllegalArgumentException if the region exceeds the buffer bounds
     */
    public @NotNull PixelBuffer crop(int x, int y, int w, int h) {
        if (w < 0 || h < 0)
            throw new IllegalArgumentException("Crop dimensions must be non-negative");
        if (x < 0 || y < 0 || x + w > this.width || y + h > this.height)
            throw new IllegalArgumentException("Crop region (%d,%d %dx%d) exceeds buffer bounds %dx%d".formatted(x, y, w, h, this.width, this.height));

        int[] result = new int[w * h];
        for (int row = 0; row < h; row++)
            System.arraycopy(this.data, (y + row) * this.width + x, result, row * w, w);
        return new PixelBuffer(result, w, h, this.hasAlpha, this.alphaMode);
    }

    /**
     * Mirrors the buffer in place along the vertical axis, swapping columns left-to-right.
     */
    public void flipHorizontal() {
        forEachRow(0, this.height, y -> {
            int row = y * this.width;
            int left = 0;
            int right = this.width - 1;
            while (left < right) {
                int tmp = this.data[row + left];
                this.data[row + left] = this.data[row + right];
                this.data[row + right] = tmp;
                left++;
                right--;
            }
        });
    }

    /**
     * Mirrors the buffer in place along the horizontal axis, swapping rows top-to-bottom.
     */
    public void flipVertical() {
        int pairs = this.height / 2;
        forEachRow(0, pairs, top -> {
            int bottom = this.height - 1 - top;
            int topOff = top * this.width;
            int botOff = bottom * this.width;
            int[] tmp = new int[this.width];
            System.arraycopy(this.data, topOff, tmp, 0, this.width);
            System.arraycopy(this.data, botOff, this.data, topOff, this.width);
            System.arraycopy(tmp, 0, this.data, botOff, this.width);
        });
    }

    /**
     * Resamples this buffer to the given dimensions using the specified filter, mixing its pixels in
     * the convention this buffer holds.
     * <p>
     * {@link Resample#NEAREST} copies pixels and reads no convention. {@link Resample#BILINEAR}
     * samples the pixel array directly: {@link AlphaMode#STRAIGHT} colour is weighted by its alpha
     * for the mix and divided back after, so a transparent neighbour adds no colour, and
     * {@link AlphaMode#PREMULTIPLIED} colour mixes as it is held, each channel at or under its alpha.
     * {@link Resample#AREA} weighs and sums each footprint the same way. {@link Resample#BICUBIC}
     * draws through {@link Graphics2D} with high-quality hints, declaring the raster in the same
     * convention. The result holds the convention this buffer holds and carries alpha only when this
     * buffer does. A buffer whose alpha carries no data mixes every channel as it is held under either
     * convention, which hold the same colour there.
     *
     * @param w the target width
     * @param h the target height
     * @param filter the resampling filter
     * @return a new pixel buffer of size {@code w x h}
     * @throws IllegalArgumentException if {@code w} or {@code h} is not positive
     */
    public @NotNull PixelBuffer resize(int w, int h, @NotNull Resample filter) {
        if (w <= 0 || h <= 0)
            throw new IllegalArgumentException("Resize dimensions must be positive, got %dx%d".formatted(w, h));

        if (filter == Resample.BICUBIC)
            return resizeBicubic(w, h);
        if (filter == Resample.AREA)
            return resizeArea(w, h);

        int[] result = new int[w * h];
        boolean nearest = filter == Resample.NEAREST;
        boolean weighByAlpha = this.hasAlpha && this.alphaMode == AlphaMode.STRAIGHT;
        forEachRow(0, h, y -> {
            int outRow = y * w;
            if (nearest) {
                int sy = (y * this.height) / h;
                for (int x = 0; x < w; x++) {
                    int sx = (x * this.width) / w;
                    result[outRow + x] = this.data[sy * this.width + sx];
                }
            } else {
                float[] sample = new float[4];
                float srcY = ((y + 0.5f) * this.height) / h - 0.5f;
                for (int x = 0; x < w; x++) {
                    float srcX = ((x + 0.5f) * this.width) / w - 0.5f;
                    sampleBilinear(this.data, this.width, this.height, false, srcX, srcY, weighByAlpha, sample, 0);
                    result[outRow + x] = this.hasAlpha
                        ? packSample(sample[0], sample[1], sample[2], sample[3], this.alphaMode)
                        : ColorMath.pack(channel(sample[0], 255), channel(sample[1], 255), channel(sample[2], 255), channel(sample[3], 255));
                }
            }
        });

        return new PixelBuffer(result, w, h, this.hasAlpha, this.alphaMode);
    }

    /**
     * Resamples this buffer by area averaging. Each output pixel sums the source pixels its footprint
     * covers, each weighted by the share of it covered along each axis - the footprint narrowed to the
     * one pixel {@link Resample#NEAREST} reads along an axis that does not shrink. Straight colour is
     * weighted by its alpha and divided back after; premultiplied colour, and every channel of a buffer
     * whose alpha carries no data, sums as it is held. The result holds this buffer's convention.
     *
     * @param w the target width
     * @param h the target height
     * @return a new pixel buffer of size {@code w x h}, carrying alpha only when this buffer does
     */
    private @NotNull PixelBuffer resizeArea(int w, int h) {
        AreaAxis columns = AreaAxis.of(this.width, w);
        AreaAxis rows = AreaAxis.of(this.height, h);
        boolean weighByAlpha = this.hasAlpha && this.alphaMode == AlphaMode.STRAIGHT;
        int[] result = new int[w * h];

        forEachRow(0, h, y -> {
            float[] sum = new float[4];
            for (int x = 0; x < w; x++) {
                Arrays.fill(sum, 0f);
                for (int j = 0; j < rows.count[y]; j++) {
                    int sourceRow = (rows.first[y] + j) * this.width;
                    float rowWeight = rows.weights[y][j];
                    for (int i = 0; i < columns.count[x]; i++)
                        accumulateTap(this.data[sourceRow + columns.first[x] + i], rowWeight * columns.weights[x][i], weighByAlpha, sum, 0);
                }
                result[y * w + x] = this.hasAlpha
                    ? packSample(sum[0], sum[1], sum[2], sum[3], this.alphaMode)
                    : ColorMath.pack(channel(sum[0], 255), channel(sum[1], 255), channel(sum[2], 255), channel(sum[3], 255));
            }
        });

        return new PixelBuffer(result, w, h, this.hasAlpha, this.alphaMode);
    }

    /**
     * The source pixels each output pixel of one axis averages, and the weight of each, summing to one.
     *
     * @param first each output pixel's first source pixel
     * @param count how many source pixels each output pixel averages
     * @param weights each output pixel's weights, one per source pixel it averages
     */
    private record AreaAxis(int @NotNull [] first, int @NotNull [] count, float @NotNull [] @NotNull [] weights) {

        /**
         * Maps an axis of {@code source} pixels onto {@code target} output pixels. A shrinking axis gives
         * each output pixel the source span {@code [x * source / target, (x + 1) * source / target)},
         * every pixel in it weighted by the share of it inside; an axis that does not shrink gives each
         * the one pixel {@code x * source / target}, rounded down.
         *
         * @param source the source length
         * @param target the output length
         * @return the mapping
         */
        static @NotNull AreaAxis of(int source, int target) {
            int[] first = new int[target];
            int[] count = new int[target];
            float[][] weights = new float[target][];

            for (int x = 0; x < target; x++) {
                if (source <= target) {
                    first[x] = (int) ((long) x * source / target);
                    count[x] = 1;
                    weights[x] = new float[]{ 1f };
                    continue;
                }

                double start = (double) x * source / target;
                double end = (double) (x + 1) * source / target;
                int from = (int) Math.floor(start);
                int to = Math.min(source, (int) Math.ceil(end));
                first[x] = from;
                count[x] = to - from;
                weights[x] = new float[to - from];
                double span = end - start;
                for (int i = from; i < to; i++)
                    weights[x][i - from] = (float) ((Math.min(end, i + 1) - Math.max(start, i)) / span);
            }

            return new AreaAxis(first, count, weights);
        }

    }

    /**
     * Resamples this buffer bicubically through {@link Graphics2D}, declaring the source and target
     * rasters in the convention this buffer holds so the interpolation weights colour by alpha exactly
     * once. A buffer whose alpha carries no data is drawn as straight colour whatever convention it
     * holds, the two holding the same colour there. The result holds this buffer's convention.
     *
     * @param w the target width
     * @param h the target height
     * @return a new pixel buffer of size {@code w x h}, carrying alpha only when this buffer does
     */
    private @NotNull PixelBuffer resizeBicubic(int w, int h) {
        int type = this.hasAlpha && this.alphaMode == AlphaMode.PREMULTIPLIED
            ? BufferedImage.TYPE_INT_ARGB_PRE
            : BufferedImage.TYPE_INT_ARGB;
        BufferedImage src = new BufferedImage(this.width, this.height, type);
        System.arraycopy(this.data, 0, ((DataBufferInt) src.getRaster().getDataBuffer()).getData(), 0, this.data.length);

        BufferedImage dst = new BufferedImage(w, h, type);
        Graphics2D g = dst.createGraphics();
        try {
            g.setComposite(AlphaComposite.Src);
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
            g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g.drawImage(src, 0, 0, w, h, null);
        } finally {
            g.dispose();
        }

        return new PixelBuffer(((DataBufferInt) dst.getRaster().getDataBuffer()).getData(), w, h, this.hasAlpha, this.alphaMode);
    }

    /**
     * Returns a new pixel buffer rotated 180 degrees.
     *
     * @return a rotated copy with the same dimensions
     */
    public @NotNull PixelBuffer rotate180() {
        int w = this.width;
        int h = this.height;
        int[] result = new int[w * h];

        forEachRow(0, h, y -> {
            int srcRow = y * w;
            int dstRow = (h - 1 - y) * w;
            for (int x = 0; x < w; x++)
                result[dstRow + (w - 1 - x)] = this.data[srcRow + x];
        });

        return new PixelBuffer(result, w, h, this.hasAlpha, this.alphaMode);
    }

    /**
     * Returns a new pixel buffer rotated 270 degrees clockwise (90 degrees counter-clockwise).
     *
     * @return a rotated copy with swapped dimensions
     */
    public @NotNull PixelBuffer rotate270() {
        int w = this.width;
        int h = this.height;
        int[] result = new int[w * h];

        forEachRow(0, h, y -> {
            int row = y * w;
            for (int x = 0; x < w; x++)
                result[(w - 1 - x) * h + y] = this.data[row + x];
        });

        return new PixelBuffer(result, h, w, this.hasAlpha, this.alphaMode);
    }

    /**
     * Returns a new pixel buffer rotated 90 degrees clockwise.
     *
     * @return a rotated copy with swapped dimensions
     */
    public @NotNull PixelBuffer rotate90() {
        int w = this.width;
        int h = this.height;
        int[] result = new int[w * h];

        forEachRow(0, h, y -> {
            int row = y * w;
            for (int x = 0; x < w; x++)
                result[x * h + (h - 1 - y)] = this.data[row + x];
        });

        return new PixelBuffer(result, h, w, this.hasAlpha, this.alphaMode);
    }

    /**
     * Returns a new pixel buffer cropped to {@link #opaqueBounds()}.
     * <p>
     * Falls back to a plain {@link #copy()} when the buffer is fully opaque or fully transparent.
     *
     * @return a trimmed pixel buffer
     */
    public @NotNull PixelBuffer trim() {
        Rectangle r = opaqueBounds();
        if (r.width == 0 || r.height == 0) return copy();
        if (r.width == this.width && r.height == this.height) return copy();
        return crop(r.x, r.y, r.width, r.height);
    }

    // --- color operations ---

    /**
     * Fills the entire buffer with the given ARGB colour. Discards any existing content.
     *
     * @param argb the fill colour
     */
    public void fill(int argb) {
        Arrays.fill(this.data, argb);
    }

    /**
     * Fills a rectangular region with the given ARGB colour. Out-of-bounds areas are silently
     * clipped; each row is written via {@link Arrays#fill}.
     *
     * @param x the left edge
     * @param y the top edge
     * @param w the rectangle width
     * @param h the rectangle height
     * @param argb the fill colour
     */
    public void fillRect(int x, int y, int w, int h, int argb) {
        int x0 = Math.max(0, x);
        int y0 = Math.max(0, y);
        int x1 = Math.min(this.width, x + w);
        int y1 = Math.min(this.height, y + h);

        if (x0 >= x1 || y0 >= y1) return;

        forEachRow(y0, y1, row -> {
            int offset = row * this.width;
            Arrays.fill(this.data, offset + x0, offset + x1, argb);
        });
    }

    /**
     * Composites a rectangle of one colour onto this buffer with the given blend mode. Out-of-bounds
     * areas are silently clipped, as {@link #fillRect(int, int, int, int, int)} clips them, and a
     * rectangle of no width or height lays nothing.
     * <p>
     * Each covered pixel becomes {@link ColorMath#blend(int, int, BlendMode)} of {@code argb} over it,
     * the blend {@link #blit(PixelBuffer, int, int, BlendMode)} lays each pixel of an image with, so
     * the rectangle lays what a blit of a one-pixel image holding {@code argb}, its alpha carrying
     * data, lays at each pixel it covers. {@link BlendMode#REPLACE} is the one mode the two part on:
     * it writes {@code argb} whatever its alpha, as {@link #fillRect(int, int, int, int, int)} does,
     * where a blit discards a source pixel of alpha {@code 0}.
     * <p>
     * Fast paths are taken where they lay what the blend lays: under {@link BlendMode#REPLACE}, and
     * for an opaque colour under a source-over mode, one whose {@code isSourceOver()} answers
     * {@code true}, each row is written via {@link Arrays#fill}; under every other mode a colour of
     * alpha {@code 0} returns at once.
     * <p>
     * This reads no alpha convention, as a blit reads none: the mode states how the colour and each
     * pixel are read, whichever convention the buffer holds. On a buffer holding premultiplied colour,
     * {@link BlendMode#NORMAL} lays a straight colour and leaves premultiplied colour - over a
     * transparent pixel, {@code F0100010} becomes {@code F00F000F} - and
     * {@link BlendMode#PREMULTIPLIED} lays a colour already premultiplied. On a buffer holding
     * straight colour, {@link BlendMode#STRAIGHT} lays a straight colour and leaves straight colour.
     *
     * @param x the left edge
     * @param y the top edge
     * @param w the rectangle width
     * @param h the rectangle height
     * @param argb the fill colour
     * @param mode the blend mode to composite the colour with
     */
    public void fillRect(int x, int y, int w, int h, int argb, @NotNull BlendMode mode) {
        int alpha = ColorMath.alpha(argb);
        if (mode == BlendMode.REPLACE || alpha == 0xFF && mode.isSourceOver()) {
            fillRect(x, y, w, h, argb);
            return;
        }
        if (alpha == 0) return;

        int x0 = Math.max(0, x);
        int y0 = Math.max(0, y);
        int x1 = Math.min(this.width, x + w);
        int y1 = Math.min(this.height, y + h);

        if (x0 >= x1 || y0 >= y1) return;

        forEachRow(y0, y1, row -> {
            int offset = row * this.width;
            for (int i = offset + x0; i < offset + x1; i++)
                this.data[i] = ColorMath.blend(argb, this.data[i], mode);
        });
    }

    /**
     * Converts every pixel to its luma-weighted grayscale equivalent in place, preserving alpha.
     */
    public void grayscale() {
        forEachRow(0, this.height, y -> PixelVector.grayscale(this.data, y * this.width, this.width));
    }

    /**
     * Inverts the colour of every pixel in place, preserving alpha, in the convention this buffer
     * holds.
     * <p>
     * Each channel of a {@link AlphaMode#STRAIGHT} pixel becomes {@code 255 - c}. Each channel of a
     * {@link AlphaMode#PREMULTIPLIED} pixel becomes {@code alpha - c}, the premultiplied form of its
     * straight colour inverted; a channel above its alpha, which no premultiplied pixel holds, becomes
     * {@code 0}. A buffer whose alpha carries no data is read as opaque, and inverts as straight colour
     * in either convention.
     */
    public void invert() {
        boolean fromAlpha = this.hasAlpha && this.alphaMode == AlphaMode.PREMULTIPLIED;

        forEachRow(0, this.height, y -> {
            int rowOff = y * this.width;
            int rowEnd = rowOff + this.width;
            for (int i = rowOff; i < rowEnd; i++) {
                int p = this.data[i];
                if (!fromAlpha) {
                    this.data[i] = p ^ 0x00FFFFFF;
                    continue;
                }

                int a = (p >>> 24) & 0xFF;
                int r = Math.max(0, a - ((p >>> 16) & 0xFF));
                int g = Math.max(0, a - ((p >>> 8) & 0xFF));
                int b = Math.max(0, a - (p & 0xFF));
                this.data[i] = (a << 24) | (r << 16) | (g << 8) | b;
            }
        });
    }

    /**
     * Multiplies every pixel's alpha by the given factor in place, in the convention this buffer
     * holds.
     * <p>
     * The factor is clamped to {@code [0, 1]}. On {@link AlphaMode#STRAIGHT} colour the RGB channels
     * are left untouched, the colour standing at full strength whatever its alpha. On
     * {@link AlphaMode#PREMULTIPLIED} colour each channel is scaled with alpha, each rounded once, so
     * every colour stays at or under its alpha.
     * <p>
     * A buffer whose alpha carries no data is scaled the same way by the convention it holds, and the
     * {@link #hasAlpha()} flag is not changed by this call: a straight one keeps its colour, and a
     * premultiplied one still reads as opaque with every channel scaled by the factor, so it shows
     * darker. Either holds a valid pixel of its convention at the scaled alpha, so a caller that begins
     * with an opaque buffer and wants the result to report meaningful alpha wraps its pixel array again
     * with {@link #ofStraight(int[], int, int)} or {@link #ofPremultiplied(int[], int, int)}, by the
     * convention it holds.
     *
     * @param factor the alpha multiplier, clamped to {@code [0, 1]}
     */
    public void multiplyAlpha(float factor) {
        float clamped = Math.clamp(factor, 0f, 1f);
        if (clamped == 1f) return;

        switch (this.alphaMode) {
            case STRAIGHT -> forEachRow(0, this.height, y -> PixelVector.multiplyAlpha(this.data, y * this.width, this.width, clamped));
            case PREMULTIPLIED -> forEachRow(0, this.height, y -> PixelVector.multiplyPremultiplied(this.data, y * this.width, this.width, clamped));
        }
    }

    /**
     * Multiplies every pixel's RGB channels by the given tint, preserving alpha. Each channel is scaled
     * alone, which holds in either convention, and the result holds this buffer's convention.
     *
     * @param argbTint the packed ARGB tint colour
     * @return a new tinted pixel buffer with the same dimensions
     */
    public @NotNull PixelBuffer tint(int argbTint) {
        int w = this.width;
        int h = this.height;
        int[] result = new int[w * h];
        int[] src = this.data;

        int tr = ColorMath.red(argbTint);
        int tg = ColorMath.green(argbTint);
        int tb = ColorMath.blue(argbTint);

        forEachRow(0, h, y -> PixelVector.tint(src, result, y * w, w, tr, tg, tb));

        return new PixelBuffer(result, w, h, true, this.alphaMode);
    }

    // --- compositing ---

    /**
     * Composites the source buffer onto this buffer at the given origin using standard source-over
     * alpha compositing.
     *
     * @param source the source buffer
     * @param dx the destination x origin
     * @param dy the destination y origin
     */
    public void blit(@NotNull PixelBuffer source, int dx, int dy) {
        blit(source, dx, dy, BlendMode.NORMAL);
    }

    /**
     * Composites the source buffer onto this buffer at the given origin using the specified blend
     * mode. Out-of-bounds pixels are silently clipped.
     * <p>
     * Fast paths are taken when {@code mode} {@linkplain BlendMode#isSourceOver() is source-over}:
     * opaque sources use a per-row {@link System#arraycopy}; alpha sources inline the fully-opaque and
     * fully-transparent short-circuits.
     *
     * @param source the source buffer
     * @param dx the destination x origin
     * @param dy the destination y origin
     * @param mode the blend mode to use for compositing
     */
    public void blit(@NotNull PixelBuffer source, int dx, int dy, @NotNull BlendMode mode) {
        int srcX0 = Math.max(0, -dx);
        int srcY0 = Math.max(0, -dy);
        int srcX1 = Math.min(source.width, this.width - dx);
        int srcY1 = Math.min(source.height, this.height - dy);

        if (srcX0 >= srcX1 || srcY0 >= srcY1) return;

        boolean sourceOver = mode.isSourceOver();
        if (sourceOver && !source.hasAlpha) {
            int rowWidth = srcX1 - srcX0;
            forEachRow(srcY0, srcY1, y -> {
                int srcOff = y * source.width + srcX0;
                int dstOff = (dy + y) * this.width + dx + srcX0;
                System.arraycopy(source.data, srcOff, this.data, dstOff, rowWidth);
            });
            return;
        }

        if (sourceOver) {
            forEachRow(srcY0, srcY1, y -> {
                int srcRow = y * source.width;
                int dstRow = (dy + y) * this.width + dx;
                for (int x = srcX0; x < srcX1; x++) {
                    int src = source.data[srcRow + x];
                    int sa = (src >>> 24) & 0xFF;
                    if (sa == 0) continue;
                    if (sa == 0xFF) {
                        this.data[dstRow + x] = src;
                        continue;
                    }
                    int dstIdx = dstRow + x;
                    this.data[dstIdx] = ColorMath.blend(src, this.data[dstIdx], mode);
                }
            });
            return;
        }

        forEachRow(srcY0, srcY1, y -> {
            int srcRow = y * source.width;
            int dstRow = (dy + y) * this.width + dx;
            for (int x = srcX0; x < srcX1; x++) {
                int src = source.data[srcRow + x];
                if (ColorMath.alpha(src) == 0) continue;
                int dstIdx = dstRow + x;
                this.data[dstIdx] = ColorMath.blend(src, this.data[dstIdx], mode);
            }
        });
    }

    /**
     * Composites the source buffer onto this buffer at the given origin, rescaling to the specified
     * width and height using nearest-neighbor sampling with source-over alpha compositing.
     *
     * @param source the source buffer
     * @param dx the destination x origin
     * @param dy the destination y origin
     * @param dw the destination width
     * @param dh the destination height
     */
    public void blitScaled(@NotNull PixelBuffer source, int dx, int dy, int dw, int dh) {
        blitScaled(source, dx, dy, dw, dh, BlendMode.NORMAL);
    }

    /**
     * Composites the source buffer onto this buffer at the given origin, rescaling to the specified
     * width and height using nearest-neighbor sampling and the specified blend mode. Out-of-bounds
     * pixels are silently clipped.
     * <p>
     * A downscale keeps one source pixel per output pixel and skips the rest. To average them instead,
     * as scaling a supersampled render down needs, pass {@link Resample#AREA} to
     * {@link #blitScaled(PixelBuffer, int, int, int, int, BlendMode, Resample)}.
     *
     * @param source the source buffer
     * @param dx the destination x origin
     * @param dy the destination y origin
     * @param dw the destination width
     * @param dh the destination height
     * @param mode the blend mode to use for compositing
     */
    public void blitScaled(@NotNull PixelBuffer source, int dx, int dy, int dw, int dh, @NotNull BlendMode mode) {
        int x0 = Math.max(0, -dx);
        int y0 = Math.max(0, -dy);
        int x1 = Math.min(dw, this.width - dx);
        int y1 = Math.min(dh, this.height - dy);

        if (x0 >= x1 || y0 >= y1) return;

        int[] columns = sourceColumns(x0, x1, source.width, dw);

        if (source.width >= dw) {
            forEachRow(y0, y1, y -> {
                int dstRow = (dy + y) * this.width;
                int sy = (y * source.height) / dh;

                for (int x = x0; x < x1; x++) {
                    int sx = columns[x - x0];
                    int src = source.data[sy * source.width + sx];
                    if (ColorMath.alpha(src) == 0) continue;
                    int dstIdx = dstRow + dx + x;
                    this.data[dstIdx] = ColorMath.blend(src, this.data[dstIdx], mode);
                }
            });
            return;
        }

        // Magnifying, each run of columns reading one texel is decided once from that texel's own
        // alpha: skipped at 0, filled where the blend answers the texel whatever lies beneath it, and
        // otherwise blended pixel by pixel, each pixel reading the texel afresh because a source may
        // hold this buffer's pixel array.
        forEachRow(y0, y1, y -> {
            int dstRow = (dy + y) * this.width;
            int sy = (y * source.height) / dh;

            for (int x = x0, end; x < x1; x = end) {
                int sx = columns[x - x0];
                end = x + 1;
                while (end < x1 && columns[end - x0] == sx) end++;

                int srcIdx = sy * source.width + sx;
                int src = source.data[srcIdx];
                int sa = ColorMath.alpha(src);
                if (sa == 0) continue;

                if (mode == BlendMode.REPLACE || sa == 0xFF && mode.isSourceOver()) {
                    Arrays.fill(this.data, dstRow + dx + x, dstRow + dx + end, src);
                    continue;
                }

                for (int i = x; i < end; i++) {
                    int texel = source.data[srcIdx];
                    if (ColorMath.alpha(texel) == 0) continue;
                    int dstIdx = dstRow + dx + i;
                    this.data[dstIdx] = ColorMath.blend(texel, this.data[dstIdx], mode);
                }
            }
        });
    }

    /**
     * Composites the source buffer onto this buffer at the given origin, rescaling it to the specified
     * width and height with the given filter and the specified blend mode. Out-of-bounds pixels are
     * silently clipped.
     * <p>
     * {@link Resample#NEAREST} lays down what
     * {@link #blitScaled(PixelBuffer, int, int, int, int, BlendMode)} lays down; any other filter
     * resamples the source with {@link #resize(int, int, Resample)}, in the convention the source
     * holds, and lays the result down with {@link #blit(PixelBuffer, int, int, BlendMode)}.
     * {@link Resample#AREA} is the filter for scaling a supersampled render down: every source pixel
     * counts toward the output.
     *
     * @param source the source buffer
     * @param dx the destination x origin
     * @param dy the destination y origin
     * @param dw the destination width
     * @param dh the destination height
     * @param mode the blend mode to use for compositing
     * @param filter the resampling filter
     */
    public void blitScaled(@NotNull PixelBuffer source, int dx, int dy, int dw, int dh, @NotNull BlendMode mode,
                           @NotNull Resample filter) {
        if (filter == Resample.NEAREST) {
            blitScaled(source, dx, dy, dw, dh, mode);
            return;
        }

        if (dw <= 0 || dh <= 0) return;
        blit(source.resize(dw, dh, filter), dx, dy, mode);
    }

    /**
     * Composites the source buffer, smoothed by FXAA, onto this buffer at the given origin, rescaling
     * to the specified width and height using nearest-neighbor sampling and the specified blend mode.
     * Out-of-bounds pixels are silently clipped and the source is left unmodified.
     * <p>
     * Lays down exactly what {@link #applyFxaa()} on the source followed by
     * {@link #blitScaled(PixelBuffer, int, int, int, int, BlendMode)} lays down, filtering only the
     * source pixels the rescale reads: each one is the pixel FXAA writes there, in the convention the
     * source holds, read from the unfiltered source, so a downscale filters a fraction of the source
     * and never copies it.
     * <p>
     * A composite of many rows splits them across a fork-join pool - the calling worker's own when it
     * is called on one - so a single composite handed to a pool still spreads its filtering over the
     * pool's workers.
     *
     * @param source the source buffer, holding a pixel array other than this buffer's
     * @param dx the destination x origin
     * @param dy the destination y origin
     * @param dw the destination width
     * @param dh the destination height
     * @param mode the blend mode to use for compositing
     * @throws IllegalArgumentException if {@code source} holds this buffer's pixel array
     */
    public void blitScaledFxaa(@NotNull PixelBuffer source, int dx, int dy, int dw, int dh, @NotNull BlendMode mode) {
        if (source.data == this.data)
            throw new IllegalArgumentException("Source must hold a pixel array other than the target's");

        int x0 = Math.max(0, -dx);
        int y0 = Math.max(0, -dy);
        int x1 = Math.min(dw, this.width - dx);
        int y1 = Math.min(dh, this.height - dy);

        if (x0 >= x1 || y0 >= y1) return;

        int[] pixels = source.data;
        int srcW = source.width;
        int srcH = source.height;
        boolean forceAlpha = !source.hasAlpha;
        AlphaMode alphaMode = source.alphaMode;
        int[] columns = sourceColumns(x0, x1, srcW, dw);

        forEachRowAcrossPool(y0, y1, y -> {
            float[] samples = new float[8];
            int dstRow = (dy + y) * this.width;
            int sy = (y * srcH) / dh;
            boolean interiorRow = sy >= 1 && sy <= srcH - 2;

            for (int x = x0; x < x1; x++) {
                int sx = columns[x - x0];
                int src = pixels[sy * srcW + sx];
                if (interiorRow && sx >= 1 && sx <= srcW - 2) {
                    long filtered = fxaaPixel(pixels, srcW, srcH, forceAlpha, sx, sy, alphaMode, samples);
                    if (filtered != UNFILTERED) src = (int) filtered;
                }
                if (ColorMath.alpha(src) == 0) continue;
                int dstIdx = dstRow + dx + x;
                this.data[dstIdx] = ColorMath.blend(src, this.data[dstIdx], mode);
            }
        });
    }

    /**
     * Tints the source buffer and composites it onto this buffer at the given origin using the
     * specified blend mode.
     *
     * @param source the source buffer
     * @param dx the destination x origin
     * @param dy the destination y origin
     * @param argbTint the ARGB tint colour
     * @param mode the blend mode to use for compositing
     */
    public void blitTinted(@NotNull PixelBuffer source, int dx, int dy, int argbTint, @NotNull BlendMode mode) {
        blit(source.tint(argbTint), dx, dy, mode);
    }

    // --- diff ---

    /**
     * Computes a per-pixel difference visualisation against another buffer.
     * <p>
     * Delegates to {@link PixelDiff#of(PixelBuffer, PixelBuffer, DiffType)}. The receiver is the
     * {@code a} (left-side / reference) argument and {@code other} is the {@code b} (right-side /
     * candidate) argument - see {@link DiffType} for per-mode pixel semantics and the meaning of
     * the directional shifts.
     * <p>
     * The result is sized to the intersection of the two buffers
     * ({@code min(this.width, other.width) × min(this.height, other.height)}). Out-of-canvas
     * pixels (both inputs fully transparent) are written as {@link ColorMath#TRANSPARENT} so the
     * caller can composite the diff over its own background.
     *
     * @param other the buffer to diff against; receiver is the left-side argument
     * @param type the diff visualisation mode
     * @return a new buffer carrying the requested diff
     */
    public @NotNull PixelBuffer diff(@NotNull PixelBuffer other, @NotNull DiffType type) {
        return PixelDiff.of(this, other, type);
    }

    /**
     * Creates a new pixel buffer by linearly interpolating between two buffers per pixel.
     * <p>
     * Each ARGB channel is blended independently, as it is held. When the two buffers have different
     * dimensions, the result uses the minimum width and height. The result holds the convention
     * {@code a} holds.
     *
     * @param a the source buffer (alpha = 0)
     * @param b the target buffer (alpha = 1)
     * @param alpha the blend factor, clamped to {@code [0, 1]}
     * @return a new pixel buffer containing the blended result
     */
    public static @NotNull PixelBuffer lerp(@NotNull PixelBuffer a, @NotNull PixelBuffer b, float alpha) {
        float blend = Math.clamp(alpha, 0f, 1f);
        float inverse = 1f - blend;
        int width = Math.min(a.width, b.width);
        int height = Math.min(a.height, b.height);
        int[] result = new int[width * height];

        forEachRow(0, height, y ->
            PixelVector.lerp(a.data, y * a.width, b.data, y * b.width, result, y * width, width, blend, inverse));

        return new PixelBuffer(result, width, height, a.hasAlpha || b.hasAlpha, a.alphaMode);
    }

    // --- filters ---

    /**
     * Applies FXAA post-processing in place, in the alpha convention this buffer holds, smoothing
     * aliased edges.
     * <p>
     * Uses luma-based edge detection with bilinear sampling along detected edge directions.
     * Processing is parallelized across scanlines. The one-pixel border is left unmodified, and so is
     * every pixel whose neighbourhood falls under the contrast threshold, byte for byte.
     * <p>
     * A smoothed pixel is the mean of two bilinear samples taken along the edge. Both mix colour in
     * proportion to alpha, so a transparent neighbour takes its share of the mix with no coverage and
     * no colour - it dilutes the coverage and leaves the colour the coverage carries untouched - and
     * the mean is written back in the convention the buffer holds, each channel rounded once.
     * <p>
     * Edge detection reads the luma of each pixel's colour carried by its alpha in either convention.
     * A premultiplied pixel holds that colour rounded, which at low alpha can turn the direction a
     * smoothed pixel is sampled along, so the two conventions agree on one picture only to within
     * that rounding. A buffer whose alpha carries no data is read as opaque.
     */
    public void applyFxaa() {
        int[] source = this.data.clone();
        boolean forceAlpha = !this.hasAlpha;
        AlphaMode alphaMode = this.alphaMode;

        IntStream.range(1, this.height - 1).parallel().forEach(y -> {
            float[] samples = new float[8];

            for (int x = 1; x < this.width - 1; x++) {
                long filtered = fxaaPixel(source, this.width, this.height, forceAlpha, x, y, alphaMode, samples);
                if (filtered != UNFILTERED) this.data[y * this.width + x] = (int) filtered;
            }
        });
    }

    /**
     * Computes the pixel FXAA writes at one interior position of {@code source}, reading only
     * {@code source}, or answers {@link #UNFILTERED} when the neighbourhood falls under the contrast
     * threshold and the pixel stays as it is.
     * <p>
     * The edge test is FXAA's local-contrast check, as FXAA 3.11 makes it: a neighbourhood whose luma
     * range is under the larger of a floor and a share of its brightest luma is left alone. The edge
     * direction and the two samples along it are the simple FXAA shader's. FXAA writes the floor and
     * the direction's floor for a luma in [0, 1], and both are scaled to the [0, 255] this luma runs
     * over.
     * <p>
     * A neighbourhood whose five edge-test reads hold one value answers {@link #UNFILTERED} before any
     * luma is computed: its contrast is zero, under a threshold that never drops below its positive
     * floor, so the edge test would leave it unfiltered too.
     * <p>
     * Every read strides and clamps by the {@code width} and {@code height} passed in, the source's
     * own. When {@code forceAlpha} is set each of the thirteen reads - the five the edge test takes
     * and the eight bilinear taps - is read as opaque, for a source whose alpha carries no data.
     *
     * @param source the pixels to filter
     * @param width the source width
     * @param height the source height
     * @param forceAlpha whether every pixel is read as opaque
     * @param x the pixel's column, in {@code [1, width - 2]}
     * @param y the pixel's row, in {@code [1, height - 2]}
     * @param alphaMode the convention the source's pixels are held in
     * @param samples the scratch the two bilinear samples are written into, eight long
     * @return the filtered pixel, or {@link #UNFILTERED}
     */
    private static long fxaaPixel(int @NotNull [] source, int width, int height, boolean forceAlpha, int x, int y,
                                  @NotNull AlphaMode alphaMode, float @NotNull [] samples) {
        // FXAA writes its constants for luma in [0, 1]; edgeLuma runs over [0, 255], so the two
        // absolute ones - the contrast floor and the direction floor - are scaled by 255, and the
        // relative ones, each a factor on a luma, stand as written.
        final float fxaaReduceMin = 1.0f / 128.0f * 255.0f;
        final float fxaaReduceMul = 1.0f / 4.0f;
        final float fxaaSpanMax = 8.0f;
        int fill = forceAlpha ? 0xFF000000 : 0;

        int rgbNw = source[(y - 1) * width + (x - 1)] | fill;
        int rgbNe = source[(y - 1) * width + (x + 1)] | fill;
        int rgbSw = source[(y + 1) * width + (x - 1)] | fill;
        int rgbSe = source[(y + 1) * width + (x + 1)] | fill;
        int rgbM = source[y * width + x] | fill;

        if (rgbNw == rgbM && rgbNe == rgbM && rgbSw == rgbM && rgbSe == rgbM)
            return UNFILTERED;

        float lumaNw = edgeLuma(rgbNw, alphaMode);
        float lumaNe = edgeLuma(rgbNe, alphaMode);
        float lumaSw = edgeLuma(rgbSw, alphaMode);
        float lumaSe = edgeLuma(rgbSe, alphaMode);
        float lumaM = edgeLuma(rgbM, alphaMode);

        float lumaMin = Math.min(lumaM, Math.min(Math.min(lumaNw, lumaNe), Math.min(lumaSw, lumaSe)));
        float lumaMax = Math.max(lumaM, Math.max(Math.max(lumaNw, lumaNe), Math.max(lumaSw, lumaSe)));

        float contrast = lumaMax - lumaMin;

        if (contrast < Math.max(0.0156f * 255.0f, lumaMax * 0.0312f))
            return UNFILTERED;

        float dirX = -((lumaNw + lumaNe) - (lumaSw + lumaSe));
        float dirY = (lumaNw + lumaSw) - (lumaNe + lumaSe);

        float dirReduce = Math.max(
            (lumaNw + lumaNe + lumaSw + lumaSe) * (0.25f * fxaaReduceMul),
            fxaaReduceMin
        );
        float rcpDirMin = 1.0f / (Math.min(Math.abs(dirX), Math.abs(dirY)) + dirReduce);

        dirX = Math.clamp(dirX * rcpDirMin, -fxaaSpanMax, fxaaSpanMax);
        dirY = Math.clamp(dirY * rcpDirMin, -fxaaSpanMax, fxaaSpanMax);

        boolean straight = switch (alphaMode) {
            case STRAIGHT -> true;
            case PREMULTIPLIED -> false;
        };
        sampleBilinear(source, width, height, forceAlpha,
            x + dirX * (1.0f / 3.0f - 0.5f),
            y + dirY * (1.0f / 3.0f - 0.5f),
            straight, samples, 0);
        sampleBilinear(source, width, height, forceAlpha,
            x + dirX * (2.0f / 3.0f - 0.5f),
            y + dirY * (2.0f / 3.0f - 0.5f),
            straight, samples, 4);

        // Each sample holds a coverage and the colour that coverage carries, so their mean is the
        // pixel's coverage and its colour as it is carried - the premultiplied value.
        return packSample(
            (samples[0] + samples[4]) * 0.5f,
            (samples[1] + samples[5]) * 0.5f,
            (samples[2] + samples[6]) * 0.5f,
            (samples[3] + samples[7]) * 0.5f,
            alphaMode
        );
    }

    /**
     * Computes the luma FXAA's edge test reads: a pixel's colour carried by its alpha, which a
     * straight pixel is scaled to and a premultiplied pixel holds already.
     *
     * @param argb the packed ARGB pixel
     * @param alphaMode the convention the pixel is held in
     * @return the luma in {@code [0, 255]}
     */
    private static float edgeLuma(int argb, @NotNull AlphaMode alphaMode) {
        return switch (alphaMode) {
            case STRAIGHT -> ColorMath.luma(argb);
            case PREMULTIPLIED -> ColorMath.luma(argb | 0xFF000000);
        };
    }

    /**
     * Packs a filtered sample into a pixel of the given convention, rounding each channel once. A
     * premultiplied pixel takes the carried colour as it is, held at or under its alpha; a straight
     * pixel takes it divided by the unrounded coverage. A sample that rounds to no coverage packs
     * to transparent black.
     *
     * @param a the sample's coverage, in {@code [0, 255]}
     * @param r the red the coverage carries, in {@code [0, a]}
     * @param g the green the coverage carries, in {@code [0, a]}
     * @param b the blue the coverage carries, in {@code [0, a]}
     * @param alphaMode the convention to pack into
     * @return the packed ARGB pixel
     */
    private static int packSample(float a, float r, float g, float b, @NotNull AlphaMode alphaMode) {
        int alpha = channel(a, 255);
        if (alpha == 0) return 0;

        return switch (alphaMode) {
            case STRAIGHT -> {
                float scale = 255f / a;
                yield ColorMath.pack(alpha, channel(r * scale, 255), channel(g * scale, 255), channel(b * scale, 255));
            }
            case PREMULTIPLIED -> ColorMath.pack(alpha, channel(r, alpha), channel(g, alpha), channel(b, alpha));
        };
    }

    /**
     * Rounds an unrounded channel value half up and holds it in {@code [0, max]}.
     *
     * @param value the channel value
     * @param max the largest value the channel may take
     * @return the rounded channel
     */
    private static int channel(float value, int max) {
        return Math.clamp(Math.round(value), 0, max);
    }

    // --- conversion ---

    /**
     * Returns a new pixel buffer with a cloned pixel array, in the convention this buffer holds.
     *
     * @return an independent copy of this buffer
     */
    public @NotNull PixelBuffer copy() {
        return new PixelBuffer(this.data.clone(), this.width, this.height, this.hasAlpha, this.alphaMode);
    }

    /**
     * Returns this buffer's pixels held in the given convention - this buffer itself when it already
     * holds it, otherwise a converted copy, which carries no mask.
     * <p>
     * Premultiplying scales each colour channel by its alpha, {@code (c * a + 127) / 255}, rounding half
     * up. Un-premultiplying divides each back, {@code (c * 255 + a / 2) / a}, rounding to nearest with a
     * tie rounding up, so every premultiplied pixel becomes the straight pixel that premultiplies back
     * to it exactly; a pixel at alpha {@code 0} or {@code 255} is left as it is, and a channel above its
     * alpha, which no premultiplied pixel holds, is held at {@code 255}. Each conversion rounds once, so
     * a straight pixel taken to premultiplied colour and back can land a level or more from where it
     * started at low alpha.
     * <p>
     * A buffer whose alpha carries no data holds the same colour in either convention: it answers a new
     * buffer sharing this buffer's pixel array under the asked convention, no pixel changed.
     *
     * @param alphaMode the convention to hold the pixels in
     * @return this buffer, or a buffer holding its pixels in {@code alphaMode}
     */
    public @NotNull PixelBuffer toAlphaMode(@NotNull AlphaMode alphaMode) {
        if (this.alphaMode == alphaMode)
            return this;

        if (!this.hasAlpha)
            return new PixelBuffer(this.data, this.width, this.height, false, alphaMode);

        int[] converted = this.data.clone();

        switch (alphaMode) {
            case PREMULTIPLIED -> forEachRow(0, this.height, y -> PixelVector.premultiply(converted, y * this.width, this.width));
            case STRAIGHT -> forEachRow(0, this.height, y -> unpremultiply(converted, y * this.width, this.width));
        }

        return new PixelBuffer(converted, this.width, this.height, true, alphaMode);
    }

    /**
     * Divides each colour channel of a run of premultiplied pixels back by its alpha in place, as
     * {@link #toAlphaMode(AlphaMode)} states.
     *
     * @param pixels the pixels
     * @param offset the index of the first pixel of the run
     * @param length the number of pixels in the run
     */
    private static void unpremultiply(int @NotNull [] pixels, int offset, int length) {
        for (int i = offset; i < offset + length; i++) {
            int p = pixels[i];
            int a = (p >>> 24) & 0xFF;
            if (a == 0xFF || a == 0) continue;
            int half = a >> 1;
            int r = Math.min(255, (((p >>> 16) & 0xFF) * 255 + half) / a);
            int g = Math.min(255, (((p >>> 8) & 0xFF) * 255 + half) / a);
            int b = Math.min(255, ((p & 0xFF) * 255 + half) / a);
            pixels[i] = (a << 24) | (r << 16) | (g << 8) | b;
        }
    }

    /**
     * Creates a {@link BufferedImage} backed by this buffer's pixel array (zero copy), in the convention
     * this buffer holds.
     * <p>
     * The returned image shares the same {@code int[]} as this buffer, so edits made through either
     * API (pixel buffer or {@link Graphics2D}) are immediately visible from the other without a
     * flush step. A buffer holding {@link AlphaMode#STRAIGHT} colour answers a
     * {@link BufferedImage#TYPE_INT_ARGB} image and one holding {@link AlphaMode#PREMULTIPLIED} colour
     * a {@link BufferedImage#TYPE_INT_ARGB_PRE} image, so Java2D and ImageIO read the colour the pixels
     * stand for - ImageIO writing a premultiplied image's straight colour to a file.
     *
     * @return a buffered image sharing this buffer's pixel array
     */
    public @NotNull BufferedImage toBufferedImage() {
        DataBufferInt db = new DataBufferInt(this.data, this.data.length);
        int[] masks = { 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000 };
        SinglePixelPackedSampleModel sm = new SinglePixelPackedSampleModel(
            DataBuffer.TYPE_INT, this.width, this.height, masks
        );
        ColorModel model = switch (this.alphaMode) {
            case STRAIGHT -> ColorModel.getRGBdefault();
            case PREMULTIPLIED -> new DirectColorModel(
                ColorSpace.getInstance(ColorSpace.CS_sRGB), 32,
                masks[0], masks[1], masks[2], masks[3],
                true, DataBuffer.TYPE_INT
            );
        };
        return new BufferedImage(
            model,
            Raster.createWritableRaster(sm, db, null),
            model.isAlphaPremultiplied(),
            null
        );
    }

    // --- mask ---

    /**
     * Enables an empty coverage mask sized to this buffer, replacing any existing mask.
     *
     * @return this buffer
     */
    public @NotNull PixelBuffer enableMask() {
        this.mask = Optional.of(new PixelMask(this.width, this.height));
        return this;
    }

    /**
     * Attaches the given coverage mask to this buffer, replacing any existing mask.
     *
     * @param mask the mask to attach
     * @return this buffer
     */
    public @NotNull PixelBuffer attachMask(@NotNull PixelMask mask) {
        this.mask = Optional.of(mask);
        return this;
    }

    /**
     * Clears this buffer's coverage mask, restoring the unmasked state.
     *
     * @return this buffer
     */
    public @NotNull PixelBuffer clearMask() {
        this.mask = Optional.empty();
        return this;
    }

    // --- private helpers ---

    /**
     * Iterates rows in {@code [start, end)}, dispatching to the common fork-join pool when the range
     * covers at least {@link #MIN_PARALLEL_ROWS} rows and the calling thread is not a fork-join
     * worker, and otherwise running every row on the calling thread in ascending order. A call on
     * any fork-join worker, of any pool, runs its rows itself, whether it is its pool's only task or
     * one of many, so a single operation handed to a pool runs its rows on one thread. A row loop
     * whose rows repay a fork there runs through {@link #forEachRowAcrossPool} instead.
     * <p>
     * Each body reads and writes only the target pixels its own index owns, and otherwise reads only
     * a source no body writes, so the bodies commute and either schedule writes the same bytes. A
     * composite whose source holds the target's own pixel array breaks that: {@code blit(this, ...)},
     * or a {@link PixelGraphics} drawing a straight target's own {@link #toBufferedImage()}, whose
     * array {@link #wrap(BufferedImage)} shares rather than copies, reads pixels other rows write, and
     * writes bytes that depend on the order its rows run in under either schedule.
     *
     * @param start the first row index, inclusive
     * @param end the last row index, exclusive
     * @param body the per-row action
     */
    static void forEachRow(int start, int end, @NotNull IntConsumer body) {
        int count = end - start;

        if (count >= MIN_PARALLEL_ROWS && !ForkJoinTask.inForkJoinPool()) {
            IntStream.range(start, end)
                .parallel()
                .forEach(body);
        } else {
            for (int y = start; y < end; y++)
                body.accept(y);
        }
    }

    /**
     * Iterates rows in {@code [start, end)}, splitting them across a fork-join pool when the range
     * covers at least {@link #MIN_PARALLEL_ROWS} rows - the calling worker's own pool when it is
     * called on one, the common pool otherwise - and otherwise running every row on the calling
     * thread in ascending order.
     * <p>
     * A row loop whose rows each carry work enough to repay a fork wherever it is called runs through
     * this rather than {@link #forEachRow}, so a single call on a pool worker spreads its rows over
     * the pool's workers. Each body is held to the footprint {@link #forEachRow} states, and any
     * scratch a body writes is its own, allocated per row.
     *
     * @param start the first row index, inclusive
     * @param end the last row index, exclusive
     * @param body the per-row action
     */
    static void forEachRowAcrossPool(int start, int end, @NotNull IntConsumer body) {
        int count = end - start;

        if (count >= MIN_PARALLEL_ROWS) {
            IntStream.range(start, end)
                .parallel()
                .forEach(body);
        } else {
            for (int y = start; y < end; y++)
                body.accept(y);
        }
    }

    /**
     * Computes the source column a nearest-neighbour rescale to {@code dw} columns reads at each
     * destination column in {@code [x0, x1)}, as {@code (x * sourceWidth) / dw} in int arithmetic,
     * wrapping where the product overflows.
     * <p>
     * The table is built once per composite and only read by its row tasks.
     *
     * @param x0 the first destination column, inclusive
     * @param x1 the last destination column, exclusive
     * @param sourceWidth the source width
     * @param dw the destination width, positive
     * @return the source column of each destination column {@code x}, at index {@code x - x0}
     */
    private static int @NotNull [] sourceColumns(int x0, int x1, int sourceWidth, int dw) {
        int[] columns = new int[x1 - x0];

        for (int x = x0; x < x1; x++)
            columns[x - x0] = (x * sourceWidth) / dw;

        return columns;
    }

    /**
     * Samples {@code source} bilinearly at a point, writing alpha and the three colour channels -
     * each in {@code [0, 255]} and unrounded - into {@code out} from {@code offset}, in that order.
     * <p>
     * A tap past an edge reads the edge pixel and keeps the point's own weights, so every tap reads a
     * pixel inside the buffer and a one-pixel-wide source samples as its one column. When
     * {@code weightByAlpha} is set each tap's colour is scaled by its alpha before the mix, which
     * straight colour needs for a transparent tap to add no colour, and the colour written is the
     * colour the coverage carries; otherwise every channel mixes as it is held.
     *
     * @param source the pixels to sample, in rows of {@code width}
     * @param width the width of {@code source}, which every tap strides and clamps by
     * @param height the height of {@code source}, which every tap clamps by
     * @param forceAlpha whether each tap is read as opaque
     * @param x the horizontal sample position, in pixels
     * @param y the vertical sample position, in pixels
     * @param weightByAlpha whether each tap's colour is scaled by its alpha before the mix
     * @param out the array receiving alpha, red, green and blue
     * @param offset the index in {@code out} the four channels start at
     */
    private static void sampleBilinear(int @NotNull [] source, int width, int height, boolean forceAlpha, float x, float y,
                                       boolean weightByAlpha, float @NotNull [] out, int offset) {
        int fill = forceAlpha ? 0xFF000000 : 0;
        int floorX = (int) Math.floor(x);
        int floorY = (int) Math.floor(y);
        float fx = x - floorX;
        float fy = y - floorY;

        int x0 = Math.clamp(floorX, 0, width - 1);
        int x1 = Math.clamp(floorX + 1, 0, width - 1);
        int row0 = Math.clamp(floorY, 0, height - 1) * width;
        int row1 = Math.clamp(floorY + 1, 0, height - 1) * width;

        Arrays.fill(out, offset, offset + 4, 0f);
        accumulateTap(source[row0 + x0] | fill, (1 - fx) * (1 - fy), weightByAlpha, out, offset);
        accumulateTap(source[row0 + x1] | fill, fx * (1 - fy), weightByAlpha, out, offset);
        accumulateTap(source[row1 + x0] | fill, (1 - fx) * fy, weightByAlpha, out, offset);
        accumulateTap(source[row1 + x1] | fill, fx * fy, weightByAlpha, out, offset);
    }

    /**
     * Adds one bilinear tap's share to a sample being accumulated.
     *
     * @param argb the tap's packed ARGB pixel
     * @param weight the tap's bilinear weight
     * @param weightByAlpha whether the tap's colour is scaled by its alpha
     * @param out the array accumulating alpha, red, green and blue
     * @param offset the index in {@code out} the four channels start at
     */
    private static void accumulateTap(int argb, float weight, boolean weightByAlpha, float @NotNull [] out, int offset) {
        int alpha = ColorMath.alpha(argb);
        float colourWeight = weightByAlpha ? weight * alpha / 255f : weight;

        out[offset] += weight * alpha;
        out[offset + 1] += colourWeight * ColorMath.red(argb);
        out[offset + 2] += colourWeight * ColorMath.green(argb);
        out[offset + 3] += colourWeight * ColorMath.blue(argb);
    }

}
