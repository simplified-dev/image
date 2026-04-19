package dev.simplified.image.codec.pnm;

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

import java.nio.charset.StandardCharsets;

/**
 * Decodes the Netpbm family of images: P1 (PBM ASCII), P2 (PGM ASCII), P3 (PPM ASCII),
 * P4 (PBM binary), P5 (PGM binary), and P6 (PPM binary).
 * <p>
 * Sample values are rescaled from the file's {@code maxval} to 8 bits per channel. PBM
 * stores '1' as black and '0' as white; this reader follows that convention. All output
 * is fully opaque ARGB.
 */
public class PnmImageReader implements ImageReader {

    @Override
    public @NotNull ImageFormat getFormat() {
        return ImageFormat.PNM;
    }

    @Override
    public boolean canRead(byte @NotNull [] data) {
        return ImageFormat.PNM.matches(data);
    }

    @Override
    @SneakyThrows
    public @NotNull ImageData read(byte @NotNull [] data, @Nullable ImageReadOptions options) {
        if (data.length < 2 || data[0] != 'P')
            throw new ImageDecodeException("Not a PNM file");

        int magic = data[1] - '0';

        if (magic < 1 || magic > 6)
            throw new ImageDecodeException("Unknown PNM variant: 'P%d'", magic);

        int[] cursor = {2};
        int width = readHeaderInt(data, cursor);
        int height = readHeaderInt(data, cursor);
        boolean isPbm = magic == 1 || magic == 4;
        int maxval = isPbm ? 1 : readHeaderInt(data, cursor);

        if (width <= 0 || height <= 0)
            throw new ImageDecodeException("Invalid PNM dimensions: '%dx%d'", width, height);

        if (maxval < 1 || maxval > 65535)
            throw new ImageDecodeException("Invalid PNM maxval: '%d'", maxval);

        if (cursor[0] < data.length)
            cursor[0]++;

        int total = Math.multiplyExact(width, height);
        int[] pixels = new int[total];
        int bps = maxval < 256 ? 1 : 2;

        switch (magic) {
            case 1 -> decodePbmAscii(data, cursor, pixels, width, height);
            case 4 -> decodePbmBinary(data, cursor[0], pixels, width, height);
            case 2 -> decodePgmAscii(data, cursor, pixels, total, maxval);
            case 5 -> decodePgmBinary(data, cursor[0], pixels, total, maxval, bps);
            case 3 -> decodePpmAscii(data, cursor, pixels, total, maxval);
            case 6 -> decodePpmBinary(data, cursor[0], pixels, total, maxval, bps);
            default -> throw new ImageDecodeException("Unreachable PNM magic: 'P%d'", magic);
        }

        return StaticImageData.of(PixelBuffer.of(pixels, width, height, false));
    }

    private static void decodePbmAscii(byte @NotNull [] data, int @NotNull [] cursor, int @NotNull [] pixels, int width, int height) {
        for (int i = 0; i < width * height; i++) {
            int bit = readBodyInt(data, cursor);
            int luma = bit == 0 ? 255 : 0;
            pixels[i] = 0xFF000000 | (luma << 16) | (luma << 8) | luma;
        }
    }

    private static void decodePbmBinary(byte @NotNull [] data, int pos, int @NotNull [] pixels, int width, int height) {
        int rowBytes = (width + 7) >>> 3;

        for (int y = 0; y < height; y++) {
            int rowStart = pos + y * rowBytes;

            for (int x = 0; x < width; x++) {
                int bit = (data[rowStart + (x >>> 3)] >>> (7 - (x & 7))) & 1;
                int luma = bit == 0 ? 255 : 0;
                pixels[y * width + x] = 0xFF000000 | (luma << 16) | (luma << 8) | luma;
            }
        }
    }

    private static void decodePgmAscii(byte @NotNull [] data, int @NotNull [] cursor, int @NotNull [] pixels, int total, int maxval) {
        for (int i = 0; i < total; i++) {
            int sample = readBodyInt(data, cursor);
            int luma = scale(sample, maxval);
            pixels[i] = 0xFF000000 | (luma << 16) | (luma << 8) | luma;
        }
    }

    private static void decodePgmBinary(byte @NotNull [] data, int pos, int @NotNull [] pixels, int total, int maxval, int bps) {
        for (int i = 0; i < total; i++) {
            int sample;

            if (bps == 1) {
                sample = data[pos++] & 0xFF;
            } else {
                sample = ((data[pos] & 0xFF) << 8) | (data[pos + 1] & 0xFF);
                pos += 2;
            }

            int luma = scale(sample, maxval);
            pixels[i] = 0xFF000000 | (luma << 16) | (luma << 8) | luma;
        }
    }

    private static void decodePpmAscii(byte @NotNull [] data, int @NotNull [] cursor, int @NotNull [] pixels, int total, int maxval) {
        for (int i = 0; i < total; i++) {
            int r = scale(readBodyInt(data, cursor), maxval);
            int g = scale(readBodyInt(data, cursor), maxval);
            int b = scale(readBodyInt(data, cursor), maxval);
            pixels[i] = 0xFF000000 | (r << 16) | (g << 8) | b;
        }
    }

    private static void decodePpmBinary(byte @NotNull [] data, int pos, int @NotNull [] pixels, int total, int maxval, int bps) {
        for (int i = 0; i < total; i++) {
            int r, g, b;

            if (bps == 1) {
                r = data[pos++] & 0xFF;
                g = data[pos++] & 0xFF;
                b = data[pos++] & 0xFF;
            } else {
                r = ((data[pos] & 0xFF) << 8) | (data[pos + 1] & 0xFF); pos += 2;
                g = ((data[pos] & 0xFF) << 8) | (data[pos + 1] & 0xFF); pos += 2;
                b = ((data[pos] & 0xFF) << 8) | (data[pos + 1] & 0xFF); pos += 2;
            }

            pixels[i] = 0xFF000000 | (scale(r, maxval) << 16) | (scale(g, maxval) << 8) | scale(b, maxval);
        }
    }

    private static int scale(int sample, int maxval) {
        if (maxval == 255) return sample & 0xFF;
        return Math.min(255, sample * 255 / maxval);
    }

    private static int readHeaderInt(byte @NotNull [] data, int @NotNull [] cursor) {
        skipWhitespaceAndComments(data, cursor);
        int start = cursor[0];

        while (cursor[0] < data.length && !isWhitespace(data[cursor[0]]) && data[cursor[0]] != '#')
            cursor[0]++;

        if (cursor[0] == start)
            throw new ImageDecodeException("PNM header truncated");

        return Integer.parseInt(new String(data, start, cursor[0] - start, StandardCharsets.US_ASCII));
    }

    private static int readBodyInt(byte @NotNull [] data, int @NotNull [] cursor) {
        skipWhitespaceAndComments(data, cursor);
        int start = cursor[0];

        while (cursor[0] < data.length && !isWhitespace(data[cursor[0]]))
            cursor[0]++;

        if (cursor[0] == start)
            throw new ImageDecodeException("PNM body truncated");

        return Integer.parseInt(new String(data, start, cursor[0] - start, StandardCharsets.US_ASCII));
    }

    private static void skipWhitespaceAndComments(byte @NotNull [] data, int @NotNull [] cursor) {
        while (cursor[0] < data.length) {
            byte b = data[cursor[0]];

            if (isWhitespace(b)) {
                cursor[0]++;
            } else if (b == '#') {
                while (cursor[0] < data.length && data[cursor[0]] != '\n')
                    cursor[0]++;
            } else {
                break;
            }
        }
    }

    private static boolean isWhitespace(byte b) {
        return b == ' ' || b == '\t' || b == '\n' || b == '\r' || b == '\f' || b == 0x0B;
    }

}
