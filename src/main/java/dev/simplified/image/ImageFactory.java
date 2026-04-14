package dev.simplified.image;


import dev.simplified.collection.Concurrent;
import dev.simplified.collection.ConcurrentList;
import dev.simplified.image.data.AnimatedImageData;
import dev.simplified.image.data.ImageFrame;
import dev.simplified.image.data.StaticImageData;
import dev.simplified.image.pixel.PixelBuffer;
import dev.simplified.image.codec.ImageReadOptions;
import dev.simplified.image.codec.ImageReader;
import dev.simplified.image.codec.ImageWriteOptions;
import dev.simplified.image.codec.ImageWriter;
import dev.simplified.image.codec.bmp.BmpImageReader;
import dev.simplified.image.codec.bmp.BmpImageWriter;
import dev.simplified.image.codec.gif.GifImageReader;
import dev.simplified.image.codec.gif.GifImageWriter;
import dev.simplified.image.codec.jpeg.JpegImageReader;
import dev.simplified.image.codec.jpeg.JpegImageWriter;
import dev.simplified.image.codec.png.PngImageReader;
import dev.simplified.image.codec.png.PngImageWriter;
import dev.simplified.image.codec.webp.WebPImageReader;
import dev.simplified.image.codec.webp.WebPImageWriter;
import dev.simplified.image.exception.ImageDecodeException;
import dev.simplified.image.exception.ImageException;
import dev.simplified.image.exception.UnsupportedFormatException;
import dev.simplified.stream.ByteArrayDataOutput;
import dev.simplified.util.StringUtil;
import dev.simplified.util.SystemUtil;
import lombok.Getter;
import lombok.SneakyThrows;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URL;
import java.nio.file.Files;
import java.util.concurrent.CompletableFuture;

/**
 * Central facade for reading and writing images across all supported formats.
 * <p>
 * Built-in readers and writers for JPEG, PNG, BMP, GIF, and WebP are registered
 * automatically. Custom format support can be added at runtime via
 * {@link #registerReader(ImageReader)} and {@link #registerWriter(ImageWriter)}.
 * <p>
 * All input methods converge to {@link #fromByteArray(byte[], ImageFormat, ImageReadOptions)}.
 * All output methods start from {@link #toByteArray(ImageData, ImageFormat, ImageWriteOptions)}.
 */
@Getter
public class ImageFactory {

    private final @NotNull ConcurrentList<ImageReader> readers = Concurrent.newList();
    private final @NotNull ConcurrentList<ImageWriter> writers = Concurrent.newList();

    /**
     * Registers all built-in readers and writers.
     */
    public ImageFactory() {
        this.readers.add(new JpegImageReader());
        this.readers.add(new PngImageReader());
        this.readers.add(new BmpImageReader());
        this.readers.add(new GifImageReader());
        this.readers.add(new WebPImageReader());

        this.writers.add(new JpegImageWriter());
        this.writers.add(new PngImageWriter());
        this.writers.add(new BmpImageWriter());
        this.writers.add(new GifImageWriter());
        this.writers.add(new WebPImageWriter());
    }

    // === Registry ===

    /**
     * Registers a custom image reader.
     *
     * @param reader the reader to register
     */
    public void registerReader(@NotNull ImageReader reader) {
        this.readers.add(reader);
    }

    /**
     * Registers a custom image writer.
     *
     * @param writer the writer to register
     */
    public void registerWriter(@NotNull ImageWriter writer) {
        this.writers.add(writer);
    }

    // === Input Methods ===

    /**
     * Decodes image data from a byte array with auto-format detection.
     *
     * @param bytes the raw image bytes
     * @return the decoded image data
     * @throws ImageException if decoding fails or format is unrecognized
     */
    public @NotNull ImageData fromByteArray(byte @NotNull [] bytes) {
        return this.fromByteArray(bytes, this.detectFormat(bytes), null);
    }

    /**
     * Decodes image data from a byte array with the specified format.
     *
     * @param bytes the raw image bytes
     * @param format the image format to use
     * @return the decoded image data
     * @throws ImageException if decoding fails
     */
    public @NotNull ImageData fromByteArray(byte @NotNull [] bytes, @NotNull ImageFormat format) {
        return this.fromByteArray(bytes, format, null);
    }

    /**
     * Decodes image data from a byte array with auto-format detection and options.
     *
     * @param bytes the raw image bytes
     * @param options format-specific read options
     * @return the decoded image data
     * @throws ImageException if decoding fails or format is unrecognized
     */
    public @NotNull ImageData fromByteArray(byte @NotNull [] bytes, @Nullable ImageReadOptions options) {
        return this.fromByteArray(bytes, this.detectFormat(bytes), options);
    }

    /**
     * Decodes image data from a byte array with the specified format and options.
     *
     * @param bytes the raw image bytes
     * @param format the image format to use
     * @param options format-specific read options, or null for defaults
     * @return the decoded image data
     * @throws ImageException if decoding fails or no reader is registered for the format
     */
    public @NotNull ImageData fromByteArray(byte @NotNull [] bytes, @NotNull ImageFormat format, @Nullable ImageReadOptions options) {
        ImageReader reader = this.readers.stream()
            .filter(r -> r.getFormat() == format)
            .findFirst()
            .orElseThrow(() -> new UnsupportedFormatException("No reader registered for format '%s'", format.getFormatName()));

        return reader.read(bytes, options);
    }

    /**
     * Decodes image data from a Base64-encoded string.
     *
     * @param encoded the Base64-encoded image data
     * @return the decoded image data
     * @throws ImageException if decoding fails
     */
    public @NotNull ImageData fromBase64(@NotNull String encoded) {
        return this.fromByteArray(StringUtil.decodeBase64(encoded));
    }

    /**
     * Decodes image data from a file.
     *
     * @param file the image file
     * @return the decoded image data
     * @throws ImageException if reading or decoding fails
     */
    @SneakyThrows
    public @NotNull ImageData fromFile(@NotNull File file) {
        return this.fromByteArray(Files.readAllBytes(file.toPath()));
    }

    /**
     * Decodes image data from a classpath resource.
     *
     * @param path the resource path
     * @return the decoded image data
     * @throws ImageException if the resource is not found or decoding fails
     */
    public @NotNull ImageData fromResource(@NotNull String path) {
        InputStream stream = SystemUtil.getResource(path);

        if (stream == null)
            throw new ImageDecodeException("Resource not found: '%s'", path);

        return this.fromStream(stream);
    }

    /**
     * Decodes image data from an input stream.
     *
     * @param inputStream the input stream containing image data
     * @return the decoded image data
     * @throws ImageException if reading or decoding fails
     */
    @SneakyThrows
    public @NotNull ImageData fromStream(@NotNull InputStream inputStream) {
        ByteArrayDataOutput buffer = new ByteArrayDataOutput();
        inputStream.transferTo(buffer);
        return this.fromByteArray(buffer.toByteArray());
    }

    /**
     * Decodes image data from a URL.
     *
     * @param url the URL to read from
     * @return the decoded image data
     * @throws ImageException if reading or decoding fails
     */
    @SneakyThrows
    public @NotNull ImageData fromUrl(@NotNull URL url) {
        try (InputStream stream = url.openStream()) {
            return this.fromStream(stream);
        }
    }

    /**
     * Wraps a {@link BufferedImage} as static image data without decoding.
     *
     * @param image the source image
     * @return a static image data instance
     */
    public @NotNull StaticImageData fromImage(@NotNull BufferedImage image) {
        return StaticImageData.of(PixelBuffer.wrap(image));
    }

    /**
     * Wraps multiple {@link BufferedImage} instances as animated image data.
     *
     * @param images the source frames
     * @param delayMs the display duration per frame in milliseconds
     * @return an animated image data instance
     */
    public @NotNull AnimatedImageData fromImages(@NotNull ConcurrentList<BufferedImage> images, int delayMs) {
        AnimatedImageData.Builder builder = AnimatedImageData.builder();

        for (BufferedImage image : images)
            builder.withFrame(ImageFrame.of(PixelBuffer.wrap(image), delayMs));

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
    public byte @NotNull [] toByteArray(@NotNull ImageData data, @NotNull ImageFormat format) {
        return this.toByteArray(data, format, null);
    }

    /**
     * Encodes image data to a byte array with format-specific options.
     *
     * @param data the image data to encode
     * @param format the target image format
     * @param options format-specific write options, or null for defaults
     * @return the encoded bytes
     * @throws ImageException if encoding fails or no writer is registered for the format
     */
    public byte @NotNull [] toByteArray(@NotNull ImageData data, @NotNull ImageFormat format, @Nullable ImageWriteOptions options) {
        ImageWriter writer = this.writers.stream()
            .filter(w -> w.getFormat() == format)
            .findFirst()
            .orElseThrow(() -> new UnsupportedFormatException("No writer registered for format '%s'", format.getFormatName()));

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
    public @NotNull String toBase64(@NotNull ImageData data, @NotNull ImageFormat format) {
        return StringUtil.encodeBase64ToString(this.toByteArray(data, format));
    }

    /**
     * Encodes image data to a file.
     *
     * @param data the image data to encode
     * @param format the target image format
     * @param file the output file
     * @throws ImageException if encoding fails
     */
    public void toFile(@NotNull ImageData data, @NotNull ImageFormat format, @NotNull File file) {
        this.toFile(data, format, file, null);
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
    @SneakyThrows
    public void toFile(@NotNull ImageData data, @NotNull ImageFormat format, @NotNull File file, @Nullable ImageWriteOptions options) {
        Files.write(file.toPath(), this.toByteArray(data, format, options));
    }

    /**
     * Encodes image data to an output stream.
     *
     * @param data the image data to encode
     * @param format the target image format
     * @param outputStream the target output stream
     * @throws ImageException if encoding fails
     */
    @SneakyThrows
    public void toStream(@NotNull ImageData data, @NotNull ImageFormat format, @NotNull OutputStream outputStream) {
        outputStream.write(this.toByteArray(data, format));
    }

    /**
     * Returns the first frame as a {@link BufferedImage}.
     *
     * @param data the image data
     * @return the first frame's pixel data
     */
    public @NotNull BufferedImage toImage(@NotNull ImageData data) {
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
    @SneakyThrows
    public @NotNull ConcurrentList<ImageData> fromFiles(@NotNull ConcurrentList<File> files) {
        var futures = files.stream()
            .map(file -> CompletableFuture.supplyAsync(() -> this.fromFile(file), java.util.concurrent.Executors.newVirtualThreadPerTaskExecutor()))
            .collect(Concurrent.toList());

        return futures.stream()
            .map(CompletableFuture::join)
            .collect(Concurrent.toList());
    }

    /**
     * Decodes multiple byte arrays in parallel using virtual threads.
     *
     * @param arrays the raw image byte arrays to decode
     * @return a list of decoded image data in the same order as the input arrays
     * @throws ImageException if any array fails to decode
     */
    public @NotNull ConcurrentList<ImageData> fromByteArrays(@NotNull ConcurrentList<byte[]> arrays) {
        var futures = arrays.stream()
            .map(bytes -> CompletableFuture.supplyAsync(() -> this.fromByteArray(bytes), java.util.concurrent.Executors.newVirtualThreadPerTaskExecutor()))
            .collect(Concurrent.toList());

        return futures.stream()
            .map(CompletableFuture::join)
            .collect(Concurrent.toList());
    }

    /**
     * Encodes multiple images to byte arrays in parallel using virtual threads.
     *
     * @param data the images to encode
     * @param format the target image format
     * @return a list of encoded byte arrays in the same order as the input images
     * @throws ImageException if any image fails to encode
     */
    public @NotNull ConcurrentList<byte[]> toByteArrays(@NotNull ConcurrentList<ImageData> data, @NotNull ImageFormat format) {
        return this.toByteArrays(data, format, null);
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
    public @NotNull ConcurrentList<byte[]> toByteArrays(
        @NotNull ConcurrentList<ImageData> data,
        @NotNull ImageFormat format,
        @Nullable ImageWriteOptions options
    ) {
        var futures = data.stream()
            .map(img -> CompletableFuture.supplyAsync(() -> this.toByteArray(img, format, options), java.util.concurrent.Executors.newVirtualThreadPerTaskExecutor()))
            .collect(Concurrent.toList());

        return futures.stream()
            .map(CompletableFuture::join)
            .collect(Concurrent.toList());
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
    public void toFiles(@NotNull ConcurrentList<ImageData> data, @NotNull ImageFormat format, @NotNull ConcurrentList<File> files) {
        if (data.size() != files.size())
            throw new IllegalArgumentException("Data and file lists must have the same size");

        var futures = Concurrent.<CompletableFuture<Void>>newList();

        for (int i = 0; i < data.size(); i++) {
            int index = i;
            futures.add(CompletableFuture.runAsync(
                () -> this.toFile(data.get(index), format, files.get(index)),
                java.util.concurrent.Executors.newVirtualThreadPerTaskExecutor()
            ));
        }

        futures.forEach(CompletableFuture::join);
    }

    // === Format Detection ===

    /**
     * Detects the image format of the given byte array by examining magic bytes.
     *
     * @param bytes the raw image bytes
     * @return the detected format
     * @throws UnsupportedFormatException if no format matches
     */
    public @NotNull ImageFormat detectFormat(byte @NotNull [] bytes) {
        return this.readers.stream()
            .filter(reader -> reader.canRead(bytes))
            .map(ImageReader::getFormat)
            .findFirst()
            .orElseGet(() -> ImageFormat.detect(bytes));
    }

}
