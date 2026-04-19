package dev.simplified.image.codec.ico;

import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.codec.ImageWriteOptions;
import dev.simplified.image.codec.ImageWriter;
import dev.simplified.image.codec.png.PngImageWriter;
import dev.simplified.image.data.StaticImageData;
import dev.simplified.stream.ByteArrayDataOutput;
import lombok.Cleanup;
import lombok.SneakyThrows;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.io.IOException;

/**
 * Writes a single-image Windows ICO file with the payload encoded as an embedded PNG.
 * PNG embedding is valid per the ICO specification since Windows Vista and sidesteps the
 * need to synthesize a BITMAPINFOHEADER + AND mask.
 * <p>
 * Multi-frame input ({@link dev.simplified.image.data.AnimatedImageData AnimatedImageData})
 * is not supported for ICO - only the first frame is written, since ICO's multi-entry
 * structure describes resolutions rather than animation frames.
 */
public class IcoImageWriter implements ImageWriter {

    private static final int MAX_ICO_DIMENSION = 256;

    @Override
    public @NotNull ImageFormat getFormat() {
        return ImageFormat.ICO;
    }

    @Override
    @SneakyThrows
    public byte @NotNull [] write(@NotNull ImageData data, @Nullable ImageWriteOptions options) {
        var frame = data.getFrames().getFirst();
        int width = frame.width();
        int height = frame.height();

        if (width > MAX_ICO_DIMENSION || height > MAX_ICO_DIMENSION)
            throw new dev.simplified.image.exception.ImageEncodeException(
                "ICO dimensions must be <= %d; got '%dx%d'", MAX_ICO_DIMENSION, width, height);

        byte[] png = new PngImageWriter().write(StaticImageData.of(frame.pixels()));

        @Cleanup ByteArrayDataOutput out = new ByteArrayDataOutput(6 + 16 + png.length);
        writeUint16LE(out, 0);
        writeUint16LE(out, 1);
        writeUint16LE(out, 1);

        out.writeByte(width == 256 ? 0 : width);
        out.writeByte(height == 256 ? 0 : height);
        out.writeByte(0);
        out.writeByte(0);
        writeUint16LE(out, 1);
        writeUint16LE(out, 32);
        writeUint32LE(out, png.length);
        writeUint32LE(out, 6 + 16);

        out.write(png);

        return out.toByteArray();
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
