package dev.simplified.image.codec.jpeg;

import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.pixel.PixelBuffer;
import dev.simplified.image.codec.ImageWriteOptions;
import dev.simplified.image.codec.ImageWriter;
import dev.simplified.stream.ByteArrayDataOutput;
import lombok.Cleanup;
import lombok.SneakyThrows;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import javax.imageio.IIOImage;
import javax.imageio.ImageIO;
import javax.imageio.ImageWriteParam;
import javax.imageio.stream.ImageOutputStream;
import java.awt.*;
import java.awt.image.BufferedImage;

/**
 * Writes JPEG images via {@link ImageIO} with configurable quality.
 */
public class JpegImageWriter implements ImageWriter {

    @Override
    public @NotNull ImageFormat getFormat() {
        return ImageFormat.JPEG;
    }

    @Override
    @SneakyThrows
    public byte @NotNull [] write(@NotNull ImageData data, @Nullable ImageWriteOptions options) {
        float quality = 0.75f;

        if (options instanceof JpegWriteOptions jpegOptions)
            quality = jpegOptions.quality();

        // JPEG does not support alpha - always convert to RGB
        PixelBuffer pixels = data.toPixelBuffer();
        BufferedImage argb = pixels.toBufferedImage();
        BufferedImage image = new BufferedImage(argb.getWidth(), argb.getHeight(), BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = image.createGraphics();
        try {
            g2d.drawImage(argb, 0, 0, null);
        } finally {
            g2d.dispose();
        }

        javax.imageio.ImageWriter writer = ImageIO.getImageWritersByFormatName("jpeg").next();
        @Cleanup ByteArrayDataOutput dataOutput = new ByteArrayDataOutput();
        @Cleanup ImageOutputStream outputStream = ImageIO.createImageOutputStream(dataOutput);
        writer.setOutput(outputStream);

        ImageWriteParam param = writer.getDefaultWriteParam();
        param.setCompressionMode(ImageWriteParam.MODE_EXPLICIT);
        param.setCompressionQuality(quality);

        writer.write(null, new IIOImage(image, null, null), param);
        writer.dispose();

        return dataOutput.toByteArray();
    }

}
