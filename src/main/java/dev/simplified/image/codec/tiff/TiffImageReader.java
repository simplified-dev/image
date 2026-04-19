package dev.simplified.image.codec.tiff;

import dev.simplified.collection.Concurrent;
import dev.simplified.collection.ConcurrentList;
import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.codec.ImageReadOptions;
import dev.simplified.image.codec.ImageReader;
import dev.simplified.image.data.AnimatedImageData;
import dev.simplified.image.data.FrameBlend;
import dev.simplified.image.data.FrameDisposal;
import dev.simplified.image.data.ImageFrame;
import dev.simplified.image.data.StaticImageData;
import dev.simplified.image.exception.ImageDecodeException;
import dev.simplified.image.pixel.PixelBuffer;
import lombok.Cleanup;
import lombok.SneakyThrows;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import javax.imageio.ImageIO;
import javax.imageio.stream.ImageInputStream;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;

/**
 * Reads TIFF images (single- and multi-page) via {@link ImageIO}'s baseline TIFF plugin.
 * <p>
 * Multi-page TIFFs are returned as {@link AnimatedImageData} with a synthesized
 * {@value #PAGE_DELAY_MS}-ms per-page delay, matching the GIF/WebP convention of
 * promoting any multi-frame image to animated output.
 */
public class TiffImageReader implements ImageReader {

    private static final int PAGE_DELAY_MS = 100;

    @Override
    public @NotNull ImageFormat getFormat() {
        return ImageFormat.TIFF;
    }

    @Override
    public boolean canRead(byte @NotNull [] data) {
        return ImageFormat.TIFF.matches(data);
    }

    @Override
    @SneakyThrows
    public @NotNull ImageData read(byte @NotNull [] data, @Nullable ImageReadOptions options) {
        @Cleanup ImageInputStream stream = ImageIO.createImageInputStream(new ByteArrayInputStream(data));
        var readers = ImageIO.getImageReadersByFormatName("tiff");

        if (!readers.hasNext())
            throw new ImageDecodeException("No TIFF reader available in this JDK");

        javax.imageio.ImageReader reader = readers.next();
        reader.setInput(stream);

        int pages = reader.getNumImages(true);

        if (pages <= 0)
            throw new ImageDecodeException("TIFF contains no pages");

        if (pages == 1) {
            BufferedImage image = reader.read(0);
            reader.dispose();
            return StaticImageData.of(PixelBuffer.wrap(image));
        }

        ConcurrentList<ImageFrame> frames = Concurrent.newList();

        for (int i = 0; i < pages; i++) {
            BufferedImage image = reader.read(i);
            frames.add(ImageFrame.of(PixelBuffer.wrap(image), PAGE_DELAY_MS, 0, 0, FrameDisposal.NONE, FrameBlend.SOURCE));
        }

        reader.dispose();

        return AnimatedImageData.builder()
            .withFrames(frames)
            .build();
    }

}
