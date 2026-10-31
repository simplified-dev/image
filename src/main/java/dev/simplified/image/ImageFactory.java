package dev.simplified.image;

import dev.simplified.annotations.SilentThrows;
import dev.simplified.annotations.UtilityClass;
import dev.simplified.collection.Concurrent;
import dev.simplified.collection.ConcurrentList;
import dev.simplified.image.codec.ImageReader;
import dev.simplified.image.codec.ImageWriter;
import dev.simplified.image.codec.bmp.BmpImageReader;
import dev.simplified.image.codec.bmp.BmpImageWriter;
import dev.simplified.image.codec.common.DecodeBudget;
import dev.simplified.image.codec.gif.GifImageReader;
import dev.simplified.image.codec.gif.GifImageWriter;
import dev.simplified.image.codec.ico.IcoImageReader;
import dev.simplified.image.codec.ico.IcoImageWriter;
import dev.simplified.image.codec.jpeg.JpegImageReader;
import dev.simplified.image.codec.jpeg.JpegImageWriter;
import dev.simplified.image.codec.png.PngImageReader;
import dev.simplified.image.codec.png.PngImageWriter;
import dev.simplified.image.codec.pnm.PnmImageReader;
import dev.simplified.image.codec.pnm.PnmImageWriter;
import dev.simplified.image.codec.qoi.QoiImageReader;
import dev.simplified.image.codec.qoi.QoiImageWriter;
import dev.simplified.image.codec.tga.TgaImageReader;
import dev.simplified.image.codec.tga.TgaImageWriter;
import dev.simplified.image.codec.tiff.TiffImageReader;
import dev.simplified.image.codec.tiff.TiffImageWriter;
import dev.simplified.image.codec.webp.WebPImageReader;
import dev.simplified.image.codec.webp.WebPImageWriter;
import dev.simplified.image.exception.ImageDecodeException;
import dev.simplified.image.exception.ImageException;
import dev.simplified.image.exception.ImageTooLargeException;
import dev.simplified.image.exception.UnsupportedFormatException;
import dev.simplified.image.frame.ImageFrame;
import dev.simplified.image.option.ImageReadOptions;
import dev.simplified.image.option.ImageWriteOptions;
import dev.simplified.image.pixel.AlphaMode;
import dev.simplified.util.StringUtil;
import dev.simplified.util.SystemUtil;
import dev.simplified.util.io.ByteArrayDataOutput;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URL;
import java.nio.file.Files;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Function;
import java.util.stream.IntStream;

/**
 * Central facade for reading and writing images across all supported formats.
 * <p>
 * The ten built-in formats - JPEG, PNG, BMP, GIF, WebP, TIFF, ICO, TGA, QOI and PNM - each have
 * one reader and one writer, created once when the class initialises and shared by every call.
 * The set is closed: nothing adds or replaces a codec at run time.
 * <p>
 * All input methods converge to {@link #fromByteArray(byte[], ImageFormat, ImageReadOptions)}.
 * All output methods start from {@link #toByteArray(ImageData, ImageFormat, ImageWriteOptions)}.
 * <p>
 * Every read holds to a process-wide pixel budget, {@link #getMaxPixels()}: a reader claims every
 * frame, page and canvas it builds the image from, counted together, and refuses an image whose
 * header asks for more with {@link ImageTooLargeException} before allocating it. A canvas the image
 * declares is held to the budget on its own.
 */
@UtilityClass
public class ImageFactory {

    /**
     * The pixel budget every read holds to until {@link #setMaxPixels(long)} sets another: 8192 by
     * 8192 pixels, 256 MiB of ARGB.
     */
    public static final long DEFAULT_MAX_PIXELS = DecodeBudget.DEFAULT_LIMIT;

    /** the built-in readers, in the order {@link #detectFormat} tries them */
    private static final @NotNull List<ImageReader> READERS;

    /** the built-in writers, one per format */
    private static final @NotNull List<ImageWriter> WRITERS;

    static {
        READERS = List.of(
            new JpegImageReader(),
            new PngImageReader(),
            new BmpImageReader(),
            new GifImageReader(),
            new WebPImageReader(),
            new TiffImageReader(),
            new IcoImageReader(),
            new TgaImageReader(),
            new QoiImageReader(),
            new PnmImageReader()
        );

        WRITERS = List.of(
            new JpegImageWriter(),
            new PngImageWriter(),
            new BmpImageWriter(),
            new GifImageWriter(),
            new WebPImageWriter(),
            new TiffImageWriter(),
            new IcoImageWriter(),
            new TgaImageWriter(),
            new QoiImageWriter(),
            new PnmImageWriter()
        );
    }

    // === Pixel Budget ===

    /**
     * Returns the most pixels one read may hold, across every frame, page and canvas it builds the
     * image from.
     *
     * @return the pixel budget every read opened from now on holds to
     */
    public static long getMaxPixels() {
        return DecodeBudget.limit();
    }

    /**
     * Sets the most pixels one read may hold, across every frame, page and canvas it builds the image
     * from, for every read in the process opened from now on; a read already under way keeps the
     * budget it opened with. {@link Long#MAX_VALUE} lifts the budget.
     *
     * @param maxPixels the pixel budget
     * @throws IllegalArgumentException if {@code maxPixels} is not positive
     */
    public static void setMaxPixels(long maxPixels) {
        DecodeBudget.setLimit(maxPixels);
    }

    // === Input Methods ===

    /**
     * Decodes image data from a byte array with auto-format detection.
     *
     * @param bytes the raw image bytes
     * @return the decoded image data
     * @throws ImageException if decoding fails or format is unrecognized
     */
    public static @NotNull ImageData fromByteArray(byte @NotNull [] bytes) {
        return fromByteArray(bytes, detectFormat(bytes), null);
    }

    /**
     * Decodes image data from a byte array with the specified format.
     *
     * @param bytes the raw image bytes
     * @param format the image format to use
     * @return the decoded image data
     * @throws ImageException if decoding fails
     */
    public static @NotNull ImageData fromByteArray(byte @NotNull [] bytes, @NotNull ImageFormat format) {
        return fromByteArray(bytes, format, null);
    }

    /**
     * Decodes image data from a byte array with auto-format detection and options.
     *
     * @param bytes the raw image bytes
     * @param options format-specific read options
     * @return the decoded image data
     * @throws ImageException if decoding fails or format is unrecognized
     */
    public static @NotNull ImageData fromByteArray(byte @NotNull [] bytes, @Nullable ImageReadOptions options) {
        return fromByteArray(bytes, detectFormat(bytes), options);
    }

    /**
     * Decodes image data from a byte array with the specified format and options.
     *
     * @param bytes the raw image bytes
     * @param format the image format to use
     * @param options format-specific read options, or null for defaults
     * @return the decoded image data
     * @throws ImageException if decoding fails
     */
    public static @NotNull ImageData fromByteArray(byte @NotNull [] bytes, @NotNull ImageFormat format, @Nullable ImageReadOptions options) {
        ImageReader reader = READERS.stream()
            .filter(r -> r.getFormat() == format)
            .findFirst()
            .orElseThrow(() -> new UnsupportedFormatException("No built-in reader for format '%s'", format.getFormatName()));

        return reader.read(bytes, options);
    }

    /**
     * Decodes image data from a Base64-encoded string.
     *
     * @param encoded the Base64-encoded image data
     * @return the decoded image data
     * @throws ImageException if decoding fails
     */
    public static @NotNull ImageData fromBase64(@NotNull String encoded) {
        return fromByteArray(StringUtil.decodeBase64(encoded));
    }

    /**
     * Decodes image data from a file.
     *
     * @param file the image file
     * @return the decoded image data
     * @throws ImageException if reading or decoding fails
     */
    @SilentThrows
    public static @NotNull ImageData fromFile(@NotNull File file) {
        return fromByteArray(Files.readAllBytes(file.toPath()));
    }

    /**
     * Decodes image data from a classpath resource.
     *
     * @param path the resource path
     * @return the decoded image data
     * @throws ImageException if the resource is not found or decoding fails
     */
    public static @NotNull ImageData fromResource(@NotNull String path) {
        InputStream stream = SystemUtil.getResource(path);

        if (stream == null)
            throw new ImageDecodeException("Resource not found: '%s'", path);

        return fromStream(stream);
    }

    /**
     * Decodes image data from an input stream.
     *
     * @param inputStream the input stream containing image data
     * @return the decoded image data
     * @throws ImageException if reading or decoding fails
     */
    @SilentThrows
    public static @NotNull ImageData fromStream(@NotNull InputStream inputStream) {
        ByteArrayDataOutput buffer = new ByteArrayDataOutput();
        inputStream.transferTo(buffer);
        return fromByteArray(buffer.toByteArray());
    }

    /**
     * Decodes image data from a URL.
     *
     * @param url the URL to read from
     * @return the decoded image data
     * @throws ImageException if reading or decoding fails
     */
    @SilentThrows
    public static @NotNull ImageData fromUrl(@NotNull URL url) {
        try (InputStream stream = url.openStream()) {
            return fromStream(stream);
        }
    }

    /**
     * Wraps a {@link BufferedImage} as static image data without decoding.
     *
     * @param image the source image
     * @return a static image data instance
     */
    public static @NotNull StaticImageData fromImage(@NotNull BufferedImage image) {
        return StaticImageData.of(PixelBuffer.wrap(image));
    }

    /**
     * Wraps multiple {@link BufferedImage} instances as animated image data.
     * <p>
     * Each image is wrapped in the convention it holds, as {@link PixelBuffer#wrap(BufferedImage)}
     * wraps it, and every frame holds the first image's: an image holding the other one is converted
     * with {@link PixelBuffer#toAlphaMode(AlphaMode)}.
     *
     * @param images the source frames
     * @param delayMs the display duration per frame in milliseconds
     * @return an animated image data instance
     */
    public static @NotNull AnimatedImageData fromImages(@NotNull ConcurrentList<BufferedImage> images, int delayMs) {
        AnimatedImageData.Builder builder = AnimatedImageData.builder();
        AlphaMode alphaMode = null;

        for (BufferedImage image : images) {
            PixelBuffer pixels = PixelBuffer.wrap(image);

            if (alphaMode == null)
                alphaMode = pixels.alphaMode();

            builder.withFrame(ImageFrame.of(pixels.toAlphaMode(alphaMode), delayMs));
        }

        return builder.build();
    }

    // === Output Methods ===

    /**
     * Encodes image data to a byte array in the given format.
     *
     * @param data the image data to encode
     * @param format the target image format
     * @return the encoded bytes
     * @throws ImageException if encoding fails
     */
    public static byte @NotNull [] toByteArray(@NotNull ImageData data, @NotNull ImageFormat format) {
        return toByteArray(data, format, null);
    }

    /**
     * Encodes image data to a byte array with format-specific options.
     *
     * @param data the image data to encode
     * @param format the target image format
     * @param options format-specific write options, or null for defaults
     * @return the encoded bytes
     * @throws ImageException if encoding fails
     */
    public static byte @NotNull [] toByteArray(@NotNull ImageData data, @NotNull ImageFormat format, @Nullable ImageWriteOptions options) {
        ImageWriter writer = WRITERS.stream()
            .filter(w -> w.getFormat() == format)
            .findFirst()
            .orElseThrow(() -> new UnsupportedFormatException("No built-in writer for format '%s'", format.getFormatName()));

        return writer.write(data, options);
    }

    /**
     * Encodes image data to a Base64 string.
     *
     * @param data the image data to encode
     * @param format the target image format
     * @return the Base64-encoded string
     * @throws ImageException if encoding fails
     */
    public static @NotNull String toBase64(@NotNull ImageData data, @NotNull ImageFormat format) {
        return StringUtil.encodeBase64ToString(toByteArray(data, format));
    }

    /**
     * Encodes image data to a file.
     *
     * @param data the image data to encode
     * @param format the target image format
     * @param file the output file
     * @throws ImageException if encoding fails
     */
    public static void toFile(@NotNull ImageData data, @NotNull ImageFormat format, @NotNull File file) {
        toFile(data, format, file, null);
    }

    /**
     * Encodes image data to a file with format-specific options.
     *
     * @param data the image data to encode
     * @param format the target image format
     * @param file the output file
     * @param options format-specific write options, or null for defaults
     * @throws ImageException if encoding fails
     */
    @SilentThrows
    public static void toFile(@NotNull ImageData data, @NotNull ImageFormat format, @NotNull File file, @Nullable ImageWriteOptions options) {
        Files.write(file.toPath(), toByteArray(data, format, options));
    }

    /**
     * Encodes image data to an output stream.
     *
     * @param data the image data to encode
     * @param format the target image format
     * @param outputStream the target output stream
     * @throws ImageException if encoding fails
     */
    @SilentThrows
    public static void toStream(@NotNull ImageData data, @NotNull ImageFormat format, @NotNull OutputStream outputStream) {
        outputStream.write(toByteArray(data, format));
    }

    /**
     * Returns the first frame as a {@link BufferedImage}.
     *
     * @param data the image data
     * @return the first frame's pixel data
     */
    public static @NotNull BufferedImage toImage(@NotNull ImageData data) {
        return data.toBufferedImage();
    }

    // === Bulk Processing ===

    /**
     * Decodes multiple image files in parallel using virtual threads.
     *
     * @param files the image files to read
     * @return a list of decoded image data in the same order as the input files
     * @throws ImageException if any file fails to decode
     */
    public static @NotNull ConcurrentList<ImageData> fromFiles(@NotNull ConcurrentList<File> files) {
        return inParallel(files, ImageFactory::fromFile);
    }

    /**
     * Decodes multiple byte arrays in parallel using virtual threads.
     *
     * @param arrays the raw image byte arrays to decode
     * @return a list of decoded image data in the same order as the input arrays
     * @throws ImageException if any array fails to decode
     */
    public static @NotNull ConcurrentList<ImageData> fromByteArrays(@NotNull ConcurrentList<byte[]> arrays) {
        return inParallel(arrays, ImageFactory::fromByteArray);
    }

    /**
     * Encodes multiple images to byte arrays in parallel using virtual threads.
     *
     * @param data the images to encode
     * @param format the target image format
     * @return a list of encoded byte arrays in the same order as the input images
     * @throws ImageException if any image fails to encode
     */
    public static @NotNull ConcurrentList<byte[]> toByteArrays(@NotNull ConcurrentList<ImageData> data, @NotNull ImageFormat format) {
        return toByteArrays(data, format, null);
    }

    /**
     * Encodes multiple images to byte arrays in parallel with options.
     *
     * @param data the images to encode
     * @param format the target image format
     * @param options format-specific write options, or null for defaults
     * @return a list of encoded byte arrays in the same order as the input images
     * @throws ImageException if any image fails to encode
     */
    public static @NotNull ConcurrentList<byte[]> toByteArrays(
        @NotNull ConcurrentList<ImageData> data,
        @NotNull ImageFormat format,
        @Nullable ImageWriteOptions options
    ) {
        return inParallel(data, image -> toByteArray(image, format, options));
    }

    /**
     * Encodes multiple images to files in parallel using virtual threads.
     *
     * @param data the images to encode
     * @param format the target image format
     * @param files the output files (must match the size of data)
     * @throws ImageException if any image fails to encode
     * @throws IllegalArgumentException if data and files have different sizes
     */
    public static void toFiles(@NotNull ConcurrentList<ImageData> data, @NotNull ImageFormat format, @NotNull ConcurrentList<File> files) {
        if (data.size() != files.size())
            throw new IllegalArgumentException("Data and file lists must have the same size");

        inParallel(IntStream.range(0, data.size()).boxed().toList(), index -> {
            toFile(data.get(index), format, files.get(index));
            return files.get(index);
        });
    }

    /**
     * Runs a task on every item, each on a virtual thread of its own, and gathers the results in
     * the order of the items.
     *
     * @param items the inputs
     * @param task what each input becomes
     * @param <T> the input type
     * @param <R> the result type
     * @return the results, in the order of {@code items}
     * @throws ImageException the first failure in the order of {@code items}, as the task threw it
     */
    private static <T, R> @NotNull ConcurrentList<R> inParallel(@NotNull List<T> items, @NotNull Function<T, R> task) {
        try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
            List<CompletableFuture<R>> futures = items.stream()
                .map(item -> CompletableFuture.supplyAsync(() -> task.apply(item), executor))
                .toList();

            return futures.stream()
                .map(ImageFactory::join)
                .collect(Concurrent.toList());
        }
    }

    /**
     * Waits for a task and returns its result, throwing what the task threw rather than the
     * {@link CompletionException} that wraps it.
     *
     * @param future the task
     * @param <R> the result type
     * @return the task's result
     */
    @SilentThrows
    private static <R> R join(@NotNull CompletableFuture<R> future) {
        try {
            return future.join();
        } catch (CompletionException ex) {
            throw ex.getCause();
        }
    }

    // === Format Detection ===

    /**
     * Detects the image format of the given byte array by examining magic bytes.
     *
     * @param bytes the raw image bytes
     * @return the detected format
     * @throws UnsupportedFormatException if no format matches
     */
    public static @NotNull ImageFormat detectFormat(byte @NotNull [] bytes) {
        return READERS.stream()
            .filter(reader -> reader.canRead(bytes))
            .map(ImageReader::getFormat)
            .findFirst()
            .orElseGet(() -> ImageFormat.detect(bytes));
    }

}
