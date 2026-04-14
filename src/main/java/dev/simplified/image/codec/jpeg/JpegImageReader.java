package dev.simplified.image.codec.jpeg;

import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.pixel.PixelBuffer;
import dev.simplified.image.data.StaticImageData;
import dev.simplified.image.codec.ImageReadOptions;
import dev.simplified.image.codec.ImageReader;
import dev.simplified.image.exception.ImageDecodeException;
import lombok.SneakyThrows;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import javax.imageio.ImageIO;
import java.io.ByteArrayInputStream;

/**
 * Reads JPEG images via {@link ImageIO}.
 */
public class JpegImageReader implements ImageReader {

    @Override
    public @NotNull ImageFormat getFormat() {
        return ImageFormat.JPEG;
    }

    @Override
    public boolean canRead(byte @NotNull [] data) {
        return ImageFormat.JPEG.matches(data);
    }

    @Override
    @SneakyThrows
    public @NotNull ImageData read(byte @NotNull [] data, @Nullable ImageReadOptions options) {
        var image = ImageIO.read(new ByteArrayInputStream(data));

        if (image == null)
            throw new ImageDecodeException("Failed to decode JPEG image");

        return StaticImageData.of(PixelBuffer.wrap(image));
    }

}
