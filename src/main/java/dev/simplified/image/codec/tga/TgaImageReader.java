package dev.simplified.image.codec.tga;

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

/**
 * Reads Truevision TGA images. Supports image types 2 (uncompressed true-color) and 10
 * (RLE true-color) at 24 or 32 bits per pixel. Honors the image-descriptor origin bit
 * (bit 5) for top-down vs. bottom-up row order. Color-mapped and grayscale variants are
 * rejected.
 */
public class TgaImageReader implements ImageReader {

    private static final int TYPE_TRUE_COLOR = 2;
    private static final int TYPE_RLE_TRUE_COLOR = 10;

    @Override
    public @NotNull ImageFormat getFormat() {
        return ImageFormat.TGA;
    }

    @Override
    public boolean canRead(byte @NotNull [] data) {
        return ImageFormat.TGA.matches(data);
    }

    @Override
    @SneakyThrows
    public @NotNull ImageData read(byte @NotNull [] data, @Nullable ImageReadOptions options) {
        if (data.length < 18)
            throw new ImageDecodeException("TGA data too short: '%d' bytes", data.length);

        int idLength = data[0] & 0xFF;
        int colorMapType = data[1] & 0xFF;
        int imageType = data[2] & 0xFF;

        if (colorMapType != 0)
            throw new ImageDecodeException("TGA color-mapped images are not supported");

        if (imageType != TYPE_TRUE_COLOR && imageType != TYPE_RLE_TRUE_COLOR)
            throw new ImageDecodeException("TGA image type '%d' is not supported", imageType);

        int width = readUint16LE(data, 12);
        int height = readUint16LE(data, 14);
        int depth = data[16] & 0xFF;
        int descriptor = data[17] & 0xFF;

        if (width <= 0 || height <= 0)
            throw new ImageDecodeException("Invalid TGA dimensions: '%dx%d'", width, height);

        if (depth != 24 && depth != 32)
            throw new ImageDecodeException("TGA pixel depth '%d' is not supported", depth);

        boolean hasAlpha = depth == 32;
        boolean topDown = (descriptor & 0x20) != 0;
        int bpp = depth / 8;
        int pos = 18 + idLength;
        int[] pixels = new int[Math.multiplyExact(width, height)];

        if (imageType == TYPE_TRUE_COLOR) {
            decodeRaw(data, pos, pixels, width, height, bpp, hasAlpha, topDown);
        } else {
            decodeRle(data, pos, pixels, width, height, bpp, hasAlpha, topDown);
        }

        return StaticImageData.of(PixelBuffer.of(pixels, width, height, hasAlpha));
    }

    private static void decodeRaw(byte @NotNull [] data, int pos, int @NotNull [] pixels, int width, int height, int bpp, boolean hasAlpha, boolean topDown) {
        int total = width * height;

        for (int i = 0; i < total; i++) {
            int argb = readPixel(data, pos, bpp, hasAlpha);
            pixels[destIndex(i, width, height, topDown)] = argb;
            pos += bpp;
        }
    }

    private static void decodeRle(byte @NotNull [] data, int pos, int @NotNull [] pixels, int width, int height, int bpp, boolean hasAlpha, boolean topDown) {
        int total = width * height;
        int i = 0;

        while (i < total) {
            int header = data[pos++] & 0xFF;
            int count = (header & 0x7F) + 1;

            if ((header & 0x80) != 0) {
                int argb = readPixel(data, pos, bpp, hasAlpha);
                pos += bpp;

                for (int j = 0; j < count; j++) {
                    if (i >= total)
                        throw new ImageDecodeException("TGA RLE packet overflows pixel count");

                    pixels[destIndex(i, width, height, topDown)] = argb;
                    i++;
                }
            } else {
                for (int j = 0; j < count; j++) {
                    if (i >= total)
                        throw new ImageDecodeException("TGA raw packet overflows pixel count");

                    pixels[destIndex(i, width, height, topDown)] = readPixel(data, pos, bpp, hasAlpha);
                    pos += bpp;
                    i++;
                }
            }
        }
    }

    private static int readPixel(byte @NotNull [] data, int pos, int bpp, boolean hasAlpha) {
        int b = data[pos] & 0xFF;
        int g = data[pos + 1] & 0xFF;
        int r = data[pos + 2] & 0xFF;
        int a = hasAlpha ? (data[pos + 3] & 0xFF) : 0xFF;
        return (a << 24) | (r << 16) | (g << 8) | b;
    }

    private static int destIndex(int linearIndex, int width, int height, boolean topDown) {
        if (topDown) return linearIndex;
        int y = linearIndex / width;
        int x = linearIndex % width;
        return (height - 1 - y) * width + x;
    }

    private static int readUint16LE(byte @NotNull [] data, int offset) {
        return (data[offset] & 0xFF) | ((data[offset + 1] & 0xFF) << 8);
    }

}
