package dev.simplified.image.codec.pnm;

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
 * Encodes Netpbm images. The output variant (PBM/PGM/PPM) and ASCII-vs-binary encoding
 * are selected via {@link PnmWriteOptions}; defaults to P6 (PPM binary, {@code maxval = 255}).
 * <p>
 * PGM output collapses RGB to Rec. 601 luma. PBM output collapses to 1-bit via a fixed
 * luma threshold of 128 (below = black / bit set, at-or-above = white / bit clear).
 * Alpha is discarded for all three variants.
 */
public class PnmImageWriter implements ImageWriter {

    private static final int PBM_LUMA_THRESHOLD = 128;

    @Override
    public @NotNull ImageFormat getFormat() {
        return ImageFormat.PNM;
    }

    @Override
    @SneakyThrows
    public byte @NotNull [] write(@NotNull ImageData data, @Nullable ImageWriteOptions options) {
        PnmWriteOptions.Variant variant = PnmWriteOptions.Variant.PPM;
        boolean ascii = false;

        if (options instanceof PnmWriteOptions pnm) {
            variant = pnm.variant();
            ascii = pnm.ascii();
        }

        PixelBuffer buffer = data.getFrames().getFirst().pixels();
        int width = buffer.width();
        int height = buffer.height();
        int magic = magicFor(variant, ascii);

        @Cleanup ByteArrayDataOutput out = new ByteArrayDataOutput();
        out.writeBytes("P" + magic + "\n" + width + " " + height + "\n");

        if (variant != PnmWriteOptions.Variant.PBM)
            out.writeBytes("255\n");

        switch (magic) {
            case 1 -> encodePbmAscii(out, buffer, width, height);
            case 4 -> encodePbmBinary(out, buffer, width, height);
            case 2 -> encodePgmAscii(out, buffer, width, height);
            case 5 -> encodePgmBinary(out, buffer, width, height);
            case 3 -> encodePpmAscii(out, buffer, width, height);
            case 6 -> encodePpmBinary(out, buffer, width, height);
            default -> throw new IllegalStateException("Unreachable PNM magic: " + magic);
        }

        return out.toByteArray();
    }

    private static int magicFor(@NotNull PnmWriteOptions.Variant variant, boolean ascii) {
        return switch (variant) {
            case PBM -> ascii ? 1 : 4;
            case PGM -> ascii ? 2 : 5;
            case PPM -> ascii ? 3 : 6;
        };
    }

    private static void encodePbmAscii(@NotNull ByteArrayDataOutput out, @NotNull PixelBuffer buffer, int width, int height) throws IOException {
        StringBuilder line = new StringBuilder(width * 2);

        for (int y = 0; y < height; y++) {
            line.setLength(0);

            for (int x = 0; x < width; x++) {
                if (x > 0) line.append(' ');
                line.append(luma(buffer.getPixel(x, y)) < PBM_LUMA_THRESHOLD ? '1' : '0');
            }

            line.append('\n');
            out.writeBytes(line.toString());
        }
    }

    private static void encodePbmBinary(@NotNull ByteArrayDataOutput out, @NotNull PixelBuffer buffer, int width, int height) throws IOException {
        int rowBytes = (width + 7) >>> 3;
        byte[] row = new byte[rowBytes];

        for (int y = 0; y < height; y++) {
            java.util.Arrays.fill(row, (byte) 0);

            for (int x = 0; x < width; x++) {
                if (luma(buffer.getPixel(x, y)) < PBM_LUMA_THRESHOLD)
                    row[x >>> 3] |= (byte) (1 << (7 - (x & 7)));
            }

            out.write(row);
        }
    }

    private static void encodePgmAscii(@NotNull ByteArrayDataOutput out, @NotNull PixelBuffer buffer, int width, int height) throws IOException {
        StringBuilder line = new StringBuilder(width * 4);

        for (int y = 0; y < height; y++) {
            line.setLength(0);

            for (int x = 0; x < width; x++) {
                if (x > 0) line.append(' ');
                line.append(luma(buffer.getPixel(x, y)));
            }

            line.append('\n');
            out.writeBytes(line.toString());
        }
    }

    private static void encodePgmBinary(@NotNull ByteArrayDataOutput out, @NotNull PixelBuffer buffer, int width, int height) throws IOException {
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                out.writeByte(luma(buffer.getPixel(x, y)));
    }

    private static void encodePpmAscii(@NotNull ByteArrayDataOutput out, @NotNull PixelBuffer buffer, int width, int height) throws IOException {
        StringBuilder line = new StringBuilder(width * 12);

        for (int y = 0; y < height; y++) {
            line.setLength(0);

            for (int x = 0; x < width; x++) {
                int argb = buffer.getPixel(x, y);
                if (x > 0) line.append(' ');
                line.append((argb >>> 16) & 0xFF).append(' ')
                    .append((argb >>> 8) & 0xFF).append(' ')
                    .append(argb & 0xFF);
            }

            line.append('\n');
            out.writeBytes(line.toString());
        }
    }

    private static void encodePpmBinary(@NotNull ByteArrayDataOutput out, @NotNull PixelBuffer buffer, int width, int height) throws IOException {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int argb = buffer.getPixel(x, y);
                out.writeByte((argb >>> 16) & 0xFF);
                out.writeByte((argb >>> 8) & 0xFF);
                out.writeByte(argb & 0xFF);
            }
        }
    }

    private static int luma(int argb) {
        int r = (argb >>> 16) & 0xFF;
        int g = (argb >>> 8) & 0xFF;
        int b = argb & 0xFF;
        return (r * 299 + g * 587 + b * 114 + 500) / 1000;
    }

}
