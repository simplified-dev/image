package dev.simplified.image.codec.bmp;

import dev.simplified.annotations.SilentThrows;
import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.PixelBuffer;
import dev.simplified.image.StaticImageData;
import dev.simplified.image.codec.ImageReader;
import dev.simplified.image.codec.common.ImageIOBridge;
import dev.simplified.image.option.ImageReadOptions;
import org.jetbrains.annotations.ApiStatus;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import javax.imageio.ImageIO;

/**
 * Reads BMP images via {@link ImageIO}.
 */
@ApiStatus.Internal
public class BmpImageReader implements ImageReader {

    @Override
    public @NotNull ImageFormat getFormat() {
        return ImageFormat.BMP;
    }

    @Override
    public boolean canRead(byte @NotNull [] data) {
        return ImageFormat.BMP.matches(data);
    }

    @Override
    @SilentThrows
    public @NotNull ImageData read(byte @NotNull [] data, @Nullable ImageReadOptions options) {
        var image = ImageIOBridge.decode(data, "Failed to decode BMP image");
        return StaticImageData.of(PixelBuffer.wrap(image));
    }

}
