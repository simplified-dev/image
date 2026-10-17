package dev.simplified.image.codec.common;

import dev.simplified.annotations.UtilityClass;
import dev.simplified.image.exception.ImageDecodeException;
import org.jetbrains.annotations.ApiStatus;
import org.jetbrains.annotations.NotNull;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;

/**
 * Decodes encoded image bytes through {@link ImageIO}, for the readers that hand a whole format to it.
 */
@UtilityClass
@ApiStatus.Internal
public class ImageIOBridge {

    /**
     * Decodes encoded image bytes through the {@link ImageIO} reader that claims them.
     *
     * @param data the encoded image
     * @param failure the message thrown when no {@link ImageIO} reader decodes {@code data}
     * @return the decoded image
     * @throws ImageDecodeException if no {@link ImageIO} reader decodes {@code data}
     * @throws IOException if the claiming reader fails while it reads
     */
    public static @NotNull BufferedImage decode(byte @NotNull [] data, @NotNull String failure) throws IOException {
        BufferedImage image = ImageIO.read(new ByteArrayInputStream(data));

        if (image == null)
            throw new ImageDecodeException(failure);

        return image;
    }

}
