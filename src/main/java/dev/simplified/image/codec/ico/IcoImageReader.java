package dev.simplified.image.codec.ico;

import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.codec.ImageReadOptions;
import dev.simplified.image.codec.ImageReader;
import dev.simplified.image.data.StaticImageData;
import dev.simplified.image.exception.ImageDecodeException;
import dev.simplified.image.pixel.PixelBuffer;
import lombok.SneakyThrows;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.util.Arrays;

/**
 * Reads Windows ICO files, returning the largest subimage.
 * <p>
 * Subimages are either embedded PNGs (common for icons 256x256 and above since Windows
 * Vista) or DIBs (BITMAPINFOHEADER + XOR pixel data + AND transparency mask). PNG blobs
 * are decoded via {@link ImageIO}. DIB blobs are decoded manually and currently require
 * 32 bits per pixel (BGRA); other color depths throw {@link ImageDecodeException}.
 */
public class IcoImageReader implements ImageReader {

    @Override
    public @NotNull ImageFormat getFormat() {
        return ImageFormat.ICO;
    }

    @Override
    public boolean canRead(byte @NotNull [] data) {
        return ImageFormat.ICO.matches(data);
    }

    @Override
    @SneakyThrows
    public @NotNull ImageData read(byte @NotNull [] data, @Nullable ImageReadOptions options) {
        if (data.length < 6)
            throw new ImageDecodeException("ICO data too short: '%d' bytes", data.length);

        if (readUint16LE(data, 0) != 0 || readUint16LE(data, 2) != 1)
            throw new ImageDecodeException("Not an ICO file");

        int count = readUint16LE(data, 4);

        if (count == 0)
            throw new ImageDecodeException("ICO contains no subimages");

        if (data.length < 6 + count * 16)
            throw new ImageDecodeException("ICO directory truncated");

        int bestIndex = 0;
        int bestPixels = -1;

        for (int i = 0; i < count; i++) {
            int off = 6 + i * 16;
            int width = effectiveDimension(data[off] & 0xFF);
            int height = effectiveDimension(data[off + 1] & 0xFF);
            int pixels = width * height;

            if (pixels > bestPixels) {
                bestPixels = pixels;
                bestIndex = i;
            }
        }

        int entryOff = 6 + bestIndex * 16;
        int width = effectiveDimension(data[entryOff] & 0xFF);
        int height = effectiveDimension(data[entryOff + 1] & 0xFF);
        int blobSize = readUint32LE(data, entryOff + 8);
        int blobOff = readUint32LE(data, entryOff + 12);

        if (blobOff < 0 || blobSize <= 0 || blobOff + blobSize > data.length)
            throw new ImageDecodeException("ICO subimage data out of bounds");

        byte[] blob = Arrays.copyOfRange(data, blobOff, blobOff + blobSize);

        if (isPng(blob))
            return decodePng(blob);

        return StaticImageData.of(decodeDib(blob, width, height));
    }

    private static int effectiveDimension(int raw) {
        return raw == 0 ? 256 : raw;
    }

    private static boolean isPng(byte @NotNull [] blob) {
        return blob.length >= 8
            && (blob[0] & 0xFF) == 0x89
            && blob[1] == 'P'
            && blob[2] == 'N'
            && blob[3] == 'G';
    }

    private static @NotNull ImageData decodePng(byte @NotNull [] blob) throws java.io.IOException {
        BufferedImage image = ImageIO.read(new ByteArrayInputStream(blob));

        if (image == null)
            throw new ImageDecodeException("Failed to decode embedded PNG in ICO");

        return StaticImageData.of(PixelBuffer.wrap(image));
    }

    private static @NotNull PixelBuffer decodeDib(byte @NotNull [] dib, int width, int height) {
        if (dib.length < 40)
            throw new ImageDecodeException("ICO DIB header truncated");

        int headerSize = readUint32LE(dib, 0);
        int bpp = readUint16LE(dib, 14);

        if (bpp != 32)
            throw new ImageDecodeException("ICO DIB with '%d' bpp is not supported", bpp);

        int pixelOff = headerSize;
        int rowStride = width * 4;

        if (pixelOff + rowStride * height > dib.length)
            throw new ImageDecodeException("ICO DIB pixel data truncated");

        int[] pixels = new int[width * height];

        for (int y = 0; y < height; y++) {
            int rowStart = pixelOff + (height - 1 - y) * rowStride;

            for (int x = 0; x < width; x++) {
                int p = rowStart + x * 4;
                int b = dib[p] & 0xFF;
                int g = dib[p + 1] & 0xFF;
                int r = dib[p + 2] & 0xFF;
                int a = dib[p + 3] & 0xFF;
                pixels[y * width + x] = (a << 24) | (r << 16) | (g << 8) | b;
            }
        }

        return PixelBuffer.of(pixels, width, height, true);
    }

    private static int readUint16LE(byte @NotNull [] data, int offset) {
        return (data[offset] & 0xFF) | ((data[offset + 1] & 0xFF) << 8);
    }

    private static int readUint32LE(byte @NotNull [] data, int offset) {
        return (data[offset] & 0xFF)
            | ((data[offset + 1] & 0xFF) << 8)
            | ((data[offset + 2] & 0xFF) << 16)
            | ((data[offset + 3] & 0xFF) << 24);
    }

}
