package dev.simplified.image.codec.tga;

import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.codec.ImageWriteOptions;
import dev.simplified.image.codec.ImageWriter;
import dev.simplified.image.pixel.PixelBuffer;
import dev.simplified.stream.ByteArrayDataOutput;
import lombok.Cleanup;
import lombok.SneakyThrows;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.io.IOException;

/**
 * Writes Truevision TGA images in top-down true-color format, with or without RLE
 * compression. Output always carries the TGA v2 footer so the result is self-identifying
 * via {@code ImageFormat.detect}.
 * <p>
 * 32 bits per pixel (BGRA) is used when the input has an alpha channel, otherwise 24 bpp
 * (BGR). RLE packets never cross row boundaries, matching the conservative interpretation
 * of the TGA 2.0 spec.
 */
public class TgaImageWriter implements ImageWriter {

    private static final int TYPE_TRUE_COLOR = 2;
    private static final int TYPE_RLE_TRUE_COLOR = 10;
    private static final byte[] FOOTER_SIGNATURE = {
        'T', 'R', 'U', 'E', 'V', 'I', 'S', 'I', 'O', 'N',
        '-', 'X', 'F', 'I', 'L', 'E', '.', 0x00
    };

    @Override
    public @NotNull ImageFormat getFormat() {
        return ImageFormat.TGA;
    }

    @Override
    @SneakyThrows
    public byte @NotNull [] write(@NotNull ImageData data, @Nullable ImageWriteOptions options) {
        boolean rle = true;

        if (options instanceof TgaWriteOptions tga)
            rle = tga.rle();

        PixelBuffer buffer = data.getFrames().getFirst().pixels();
        int width = buffer.width();
        int height = buffer.height();
        boolean hasAlpha = buffer.hasAlpha();
        int bpp = hasAlpha ? 4 : 3;
        int depth = bpp * 8;
        int alphaBits = hasAlpha ? 8 : 0;
        int descriptor = alphaBits | 0x20;
        int imageType = rle ? TYPE_RLE_TRUE_COLOR : TYPE_TRUE_COLOR;

        @Cleanup ByteArrayDataOutput out = new ByteArrayDataOutput();

        out.writeByte(0);
        out.writeByte(0);
        out.writeByte(imageType);
        writeUint16LE(out, 0);
        writeUint16LE(out, 0);
        out.writeByte(0);
        writeUint16LE(out, 0);
        writeUint16LE(out, 0);
        writeUint16LE(out, width);
        writeUint16LE(out, height);
        out.writeByte(depth);
        out.writeByte(descriptor);

        if (rle) {
            writeRle(out, buffer, width, height, hasAlpha, bpp);
        } else {
            writeRaw(out, buffer, width, height, hasAlpha);
        }

        writeUint32LE(out, 0);
        writeUint32LE(out, 0);
        out.write(FOOTER_SIGNATURE);

        return out.toByteArray();
    }

    private static void writeRaw(@NotNull ByteArrayDataOutput out, @NotNull PixelBuffer buffer, int width, int height, boolean hasAlpha) throws IOException {
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                writePixel(out, buffer.getPixel(x, y), hasAlpha);
    }

    private static void writeRle(@NotNull ByteArrayDataOutput out, @NotNull PixelBuffer buffer, int width, int height, boolean hasAlpha, int bpp) throws IOException {
        int[] row = new int[width];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++)
                row[x] = buffer.getPixel(x, y);

            int x = 0;

            while (x < width) {
                int runLen = 1;

                while (runLen < 128 && x + runLen < width && row[x + runLen] == row[x])
                    runLen++;

                if (runLen >= 2) {
                    out.writeByte(0x80 | (runLen - 1));
                    writePixel(out, row[x], hasAlpha);
                    x += runLen;
                } else {
                    int rawStart = x;
                    int rawLen = 1;
                    x++;

                    while (rawLen < 128 && x < width) {
                        int peek = 1;

                        while (peek < 3 && x + peek < width && row[x + peek] == row[x])
                            peek++;

                        if (peek >= 2) break;

                        rawLen++;
                        x++;
                    }

                    out.writeByte(rawLen - 1);

                    for (int i = 0; i < rawLen; i++)
                        writePixel(out, row[rawStart + i], hasAlpha);
                }
            }
        }
    }

    private static void writePixel(@NotNull ByteArrayDataOutput out, int argb, boolean hasAlpha) throws IOException {
        out.writeByte(argb & 0xFF);
        out.writeByte((argb >>> 8) & 0xFF);
        out.writeByte((argb >>> 16) & 0xFF);

        if (hasAlpha)
            out.writeByte((argb >>> 24) & 0xFF);
    }

    private static void writeUint16LE(@NotNull ByteArrayDataOutput out, int value) throws IOException {
        out.writeByte(value & 0xFF);
        out.writeByte((value >>> 8) & 0xFF);
    }

    private static void writeUint32LE(@NotNull ByteArrayDataOutput out, int value) throws IOException {
        out.writeByte(value & 0xFF);
        out.writeByte((value >>> 8) & 0xFF);
        out.writeByte((value >>> 16) & 0xFF);
        out.writeByte((value >>> 24) & 0xFF);
    }

}
