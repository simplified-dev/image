package dev.simplified.image.codec.qoi;

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
 * Decodes QOI images per the Quite OK Image format specification.
 *
 * @see <a href="https://qoiformat.org/qoi-specification.pdf">QOI specification</a>
 */
public class QoiImageReader implements ImageReader {

    private static final int QOI_OP_RGB = 0xFE;
    private static final int QOI_OP_RGBA = 0xFF;
    private static final int QOI_OP_INDEX = 0x00;
    private static final int QOI_OP_DIFF = 0x40;
    private static final int QOI_OP_LUMA = 0x80;
    private static final int QOI_OP_RUN = 0xC0;
    private static final int QOI_MASK_2 = 0xC0;

    @Override
    public @NotNull ImageFormat getFormat() {
        return ImageFormat.QOI;
    }

    @Override
    public boolean canRead(byte @NotNull [] data) {
        return ImageFormat.QOI.matches(data);
    }

    @Override
    @SneakyThrows
    public @NotNull ImageData read(byte @NotNull [] data, @Nullable ImageReadOptions options) {
        if (data.length < 14 + 8)
            throw new ImageDecodeException("QOI data too short: '%d' bytes", data.length);

        if (data[0] != 'q' || data[1] != 'o' || data[2] != 'i' || data[3] != 'f')
            throw new ImageDecodeException("Invalid QOI magic bytes");

        int width = readUint32BE(data, 4);
        int height = readUint32BE(data, 8);
        int channels = data[12] & 0xFF;

        if (width <= 0 || height <= 0)
            throw new ImageDecodeException("Invalid QOI dimensions: '%dx%d'", width, height);

        if (channels != 3 && channels != 4)
            throw new ImageDecodeException("Invalid QOI channel count: '%d'", channels);

        int totalPixels = Math.multiplyExact(width, height);
        int[] pixels = new int[totalPixels];
        int[] table = new int[64];
        int r = 0, g = 0, b = 0, a = 255;
        int pos = 14;
        int end = data.length - 8;
        int run = 0;

        for (int i = 0; i < totalPixels; i++) {
            if (run > 0) {
                run--;
            } else if (pos < end) {
                int b1 = data[pos++] & 0xFF;

                if (b1 == QOI_OP_RGB) {
                    r = data[pos++] & 0xFF;
                    g = data[pos++] & 0xFF;
                    b = data[pos++] & 0xFF;
                } else if (b1 == QOI_OP_RGBA) {
                    r = data[pos++] & 0xFF;
                    g = data[pos++] & 0xFF;
                    b = data[pos++] & 0xFF;
                    a = data[pos++] & 0xFF;
                } else {
                    int tag2 = b1 & QOI_MASK_2;

                    if (tag2 == QOI_OP_INDEX) {
                        int argb = table[b1 & 0x3F];
                        a = (argb >>> 24) & 0xFF;
                        r = (argb >>> 16) & 0xFF;
                        g = (argb >>> 8) & 0xFF;
                        b = argb & 0xFF;
                    } else if (tag2 == QOI_OP_DIFF) {
                        r = (r + ((b1 >>> 4) & 0x3) - 2) & 0xFF;
                        g = (g + ((b1 >>> 2) & 0x3) - 2) & 0xFF;
                        b = (b + (b1 & 0x3) - 2) & 0xFF;
                    } else if (tag2 == QOI_OP_LUMA) {
                        int b2 = data[pos++] & 0xFF;
                        int dg = (b1 & 0x3F) - 32;
                        r = (r + dg - 8 + ((b2 >>> 4) & 0xF)) & 0xFF;
                        g = (g + dg) & 0xFF;
                        b = (b + dg - 8 + (b2 & 0xF)) & 0xFF;
                    } else {
                        run = b1 & 0x3F;
                    }
                }
            } else {
                throw new ImageDecodeException("QOI stream truncated at pixel '%d' of '%d'", i, totalPixels);
            }

            int argb = (a << 24) | (r << 16) | (g << 8) | b;
            pixels[i] = argb;
            table[(r * 3 + g * 5 + b * 7 + a * 11) & 0x3F] = argb;
        }

        return StaticImageData.of(PixelBuffer.of(pixels, width, height, channels == 4));
    }

    private static int readUint32BE(byte @NotNull [] data, int offset) {
        return ((data[offset] & 0xFF) << 24)
            | ((data[offset + 1] & 0xFF) << 16)
            | ((data[offset + 2] & 0xFF) << 8)
            | (data[offset + 3] & 0xFF);
    }

}
