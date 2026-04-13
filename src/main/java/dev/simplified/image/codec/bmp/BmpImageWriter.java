package dev.simplified.image.codec.bmp;

import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.codec.ImageWriteOptions;
import dev.simplified.image.codec.ImageWriter;
import dev.simplified.stream.ByteArrayDataOutput;
import lombok.Cleanup;
import lombok.SneakyThrows;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;

/**
 * Writes BMP images via {@link ImageIO}.
 */
public class BmpImageWriter implements ImageWriter {

    @Override
    public @NotNull ImageFormat getFormat() {
        return ImageFormat.BMP;
    }

    @Override
    @SneakyThrows
    public byte @NotNull [] write(@NotNull ImageData data, @Nullable ImageWriteOptions options) {
        // BMP does not support alpha - always convert to RGB
        BufferedImage argb = data.toBufferedImage();
        BufferedImage image = new BufferedImage(argb.getWidth(), argb.getHeight(), BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = image.createGraphics();
        try {
            g2d.drawImage(argb, 0, 0, null);
        } finally {
            g2d.dispose();
        }

        @Cleanup ByteArrayDataOutput dataOutput = new ByteArrayDataOutput();
        ImageIO.write(image, "bmp", dataOutput);
        return dataOutput.toByteArray();
    }

}
