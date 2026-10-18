package dev.simplified.image.codec.common;

import dev.simplified.annotations.UtilityClass;
import dev.simplified.image.exception.ImageDecodeException;
import dev.simplified.image.exception.ImageTooLargeException;
import org.jetbrains.annotations.ApiStatus;
import org.jetbrains.annotations.NotNull;

import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.stream.ImageInputStream;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.Iterator;
import java.util.Locale;

/**
 * Decodes encoded image bytes through {@link ImageIO}, for the readers that hand a whole format to it.
 */
@UtilityClass
@ApiStatus.Internal
public class ImageIOBridge {

    /**
     * Decodes encoded image bytes through the {@link ImageIO} reader that claims them, as
     * {@link ImageIO#read(ImageInputStream)} does, once the size its header declares fits the
     * {@link DecodeBudget}.
     *
     * @param data the encoded image
     * @param failure the message thrown when no {@link ImageIO} reader decodes {@code data}
     * @return the decoded image
     * @throws ImageDecodeException if no {@link ImageIO} reader decodes {@code data}
     * @throws ImageTooLargeException if the image declares more pixels than the budget allows
     * @throws IOException if the claiming reader fails while it reads
     */
    public static @NotNull BufferedImage decode(byte @NotNull [] data, @NotNull String failure) throws IOException {
        try (ImageInputStream stream = ImageIO.createImageInputStream(new ByteArrayInputStream(data))) {
            Iterator<ImageReader> readers = ImageIO.getImageReaders(stream);

            if (!readers.hasNext())
                throw new ImageDecodeException(failure);

            ImageReader reader = readers.next();

            try {
                reader.setInput(stream, true, true);
                DecodeBudget.open(reader.getFormatName().toUpperCase(Locale.ROOT)).claim(reader.getWidth(0), reader.getHeight(0));
                return reader.read(0, reader.getDefaultReadParam());
            } finally {
                reader.dispose();
            }
        }
    }

}
