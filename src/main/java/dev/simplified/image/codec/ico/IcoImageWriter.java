package dev.simplified.image.codec.ico;

import dev.simplified.annotations.Cleanup;
import dev.simplified.annotations.SilentThrows;
import dev.simplified.image.AnimatedImageData;
import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.StaticImageData;
import dev.simplified.image.codec.ImageWriter;
import dev.simplified.image.codec.common.LittleEndian;
import dev.simplified.image.codec.png.PngImageWriter;
import dev.simplified.image.option.ImageWriteOptions;
import dev.simplified.util.io.ByteArrayDataOutput;
import org.jetbrains.annotations.ApiStatus;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

/**
 * Writes a single-image Windows ICO file with the payload encoded as an embedded PNG.
 * PNG embedding is valid per the ICO specification since Windows Vista and sidesteps the
 * need to synthesize a BITMAPINFOHEADER + AND mask.
 * <p>
 * Multi-frame input ({@link AnimatedImageData AnimatedImageData})
 * is not supported for ICO - only the first frame is written, since ICO's multi-entry
 * structure describes resolutions rather than animation frames.
 */
@ApiStatus.Internal
public class IcoImageWriter implements ImageWriter {

    private static final int MAX_ICO_DIMENSION = 256;

    @Override
    public @NotNull ImageFormat getFormat() {
        return ImageFormat.ICO;
    }

    @Override
    @SilentThrows
    public byte @NotNull [] write(@NotNull ImageData data, @Nullable ImageWriteOptions options) {
        var frame = data.getFrames().getFirst();
        int width = frame.width();
        int height = frame.height();

        if (width > MAX_ICO_DIMENSION || height > MAX_ICO_DIMENSION)
            throw new dev.simplified.image.exception.ImageEncodeException(
                "ICO dimensions must be <= %d; got '%dx%d'", MAX_ICO_DIMENSION, width, height);

        byte[] png = new PngImageWriter().write(StaticImageData.of(frame.pixels(), data.getAlphaMode()));

        @Cleanup ByteArrayDataOutput out = new ByteArrayDataOutput(6 + 16 + png.length);
        LittleEndian.writeUint16(out, 0);
        LittleEndian.writeUint16(out, 1);
        LittleEndian.writeUint16(out, 1);

        out.writeByte(width == 256 ? 0 : width);
        out.writeByte(height == 256 ? 0 : height);
        out.writeByte(0);
        out.writeByte(0);
        LittleEndian.writeUint16(out, 1);
        LittleEndian.writeUint16(out, 32);
        LittleEndian.writeUint32(out, png.length);
        LittleEndian.writeUint32(out, 6 + 16);

        out.write(png);

        return out.toByteArray();
    }

}
