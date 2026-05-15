package dev.simplified.image.codec.tiff;

import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.codec.ImageWriteOptions;
import dev.simplified.image.codec.ImageWriter;
import dev.simplified.image.data.AnimatedImageData;
import dev.simplified.image.data.ImageFrame;
import dev.simplified.image.data.StaticImageData;
import dev.simplified.image.exception.ImageEncodeException;
import dev.simplified.stream.ByteArrayDataOutput;
import lombok.Cleanup;
import lombok.SneakyThrows;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.awt.image.BufferedImage;
import javax.imageio.IIOImage;
import javax.imageio.ImageIO;
import javax.imageio.ImageWriteParam;
import javax.imageio.stream.ImageOutputStream;

/**
 * Writes TIFF images (single- and multi-page) via {@link ImageIO}'s baseline TIFF plugin.
 * <p>
 * {@link AnimatedImageData AnimatedImageData} is encoded as a
 * multi-page TIFF using {@code writeToSequence}; {@link StaticImageData
 * StaticImageData} is encoded as a single page. Compression scheme is selected via
 * {@link TiffWriteOptions}; the default is {@link TiffWriteOptions.Compression#DEFLATE}.
 */
public class TiffImageWriter implements ImageWriter {

    @Override
    public @NotNull ImageFormat getFormat() {
        return ImageFormat.TIFF;
    }

    @Override
    @SneakyThrows
    public byte @NotNull [] write(@NotNull ImageData data, @Nullable ImageWriteOptions options) {
        TiffWriteOptions.Compression compression = TiffWriteOptions.Compression.DEFLATE;

        if (options instanceof TiffWriteOptions tiff)
            compression = tiff.compression();

        var writers = ImageIO.getImageWritersByFormatName("tiff");

        if (!writers.hasNext())
            throw new ImageEncodeException("No TIFF writer available in this JDK");

        javax.imageio.ImageWriter iioWriter = writers.next();
        ImageWriteParam param = iioWriter.getDefaultWriteParam();
        configureCompression(param, compression);

        @Cleanup ByteArrayDataOutput out = new ByteArrayDataOutput();
        @Cleanup ImageOutputStream ios = ImageIO.createImageOutputStream(out);
        iioWriter.setOutput(ios);

        if (data.isAnimated()) {
            iioWriter.prepareWriteSequence(null);

            for (ImageFrame frame : data.getFrames()) {
                BufferedImage image = frame.pixels().toBufferedImage();
                iioWriter.writeToSequence(new IIOImage(image, null, null), param);
            }

            iioWriter.endWriteSequence();
        } else {
            BufferedImage image = data.toBufferedImage();
            iioWriter.write(null, new IIOImage(image, null, null), param);
        }

        iioWriter.dispose();
        ios.flush();
        return out.toByteArray();
    }

    private static void configureCompression(@NotNull ImageWriteParam param, @NotNull TiffWriteOptions.Compression compression) {
        switch (compression) {
            case NONE -> param.setCompressionMode(ImageWriteParam.MODE_DISABLED);
            case LZW -> {
                param.setCompressionMode(ImageWriteParam.MODE_EXPLICIT);
                param.setCompressionType("LZW");
            }
            case DEFLATE -> {
                param.setCompressionMode(ImageWriteParam.MODE_EXPLICIT);
                param.setCompressionType("Deflate");
            }
            case PACKBITS -> {
                param.setCompressionMode(ImageWriteParam.MODE_EXPLICIT);
                param.setCompressionType("PackBits");
            }
        }
    }

}
