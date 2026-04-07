package dev.simplified.image.codec.png;

import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFormat;
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
import java.awt.image.BufferedImage;

/**
 * Writes PNG images via {@link ImageIO} with configurable compression level.
 */
public class PngImageWriter implements ImageWriter {

    @Override
    public @NotNull ImageFormat getFormat() {
        return ImageFormat.PNG;
    }

    @Override
    @SneakyThrows
    public byte @NotNull [] write(@NotNull ImageData data, @Nullable ImageWriteOptions options) {
        BufferedImage image = data.toBufferedImage();

        javax.imageio.ImageWriter writer = ImageIO.getImageWritersByFormatName("png").next();
        @Cleanup ByteArrayDataOutput dataOutput = new ByteArrayDataOutput();
        @Cleanup ImageOutputStream outputStream = ImageIO.createImageOutputStream(dataOutput);
        writer.setOutput(outputStream);

        ImageWriteParam param = writer.getDefaultWriteParam();

        if (options instanceof PngWriteOptions pngOptions && param.canWriteCompressed()) {
            param.setCompressionMode(ImageWriteParam.MODE_EXPLICIT);
            // PNG compression level 0-9 maps to quality 1.0-0.0 (inverted)
            param.setCompressionQuality(1.0f - (pngOptions.compressionLevel() / 9.0f));
        }

        writer.write(null, new IIOImage(image, null, null), param);
        writer.dispose();

        return dataOutput.toByteArray();
    }

}
