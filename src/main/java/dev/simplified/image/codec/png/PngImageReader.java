package dev.simplified.image.codec.png;

import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.PixelBuffer;
import dev.simplified.image.StaticImageData;
import dev.simplified.image.codec.ImageReadOptions;
import dev.simplified.image.codec.ImageReader;
import dev.simplified.image.exception.ImageDecodeException;
import lombok.SneakyThrows;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import javax.imageio.ImageIO;
import java.io.ByteArrayInputStream;

/**
 * Reads PNG images via {@link ImageIO}.
 */
public class PngImageReader implements ImageReader {

    @Override
    public @NotNull ImageFormat getFormat() {
        return ImageFormat.PNG;
    }

    @Override
    public boolean canRead(byte @NotNull [] data) {
        return ImageFormat.PNG.matches(data);
    }

    @Override
    @SneakyThrows
    public @NotNull ImageData read(byte @NotNull [] data, @Nullable ImageReadOptions options) {
        var image = ImageIO.read(new ByteArrayInputStream(data));

        if (image == null)
            throw new ImageDecodeException("Failed to decode PNG image");

        return StaticImageData.of(PixelBuffer.wrap(image));
    }

}
