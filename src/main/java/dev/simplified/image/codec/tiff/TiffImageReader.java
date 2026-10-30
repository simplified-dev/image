package dev.simplified.image.codec.tiff;

import dev.simplified.annotations.Cleanup;
import dev.simplified.annotations.SilentThrows;
import dev.simplified.collection.Concurrent;
import dev.simplified.collection.ConcurrentList;
import dev.simplified.image.AnimatedImageData;
import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.PixelBuffer;
import dev.simplified.image.StaticImageData;
import dev.simplified.image.codec.ImageReader;
import dev.simplified.image.codec.common.DecodeBudget;
import dev.simplified.image.exception.ImageDecodeException;
import dev.simplified.image.frame.FrameBlend;
import dev.simplified.image.frame.FrameDisposal;
import dev.simplified.image.frame.ImageFrame;
import dev.simplified.image.option.ImageReadOptions;
import dev.simplified.image.pixel.AlphaMode;
import org.jetbrains.annotations.ApiStatus;
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
 * <p>
 * Every page is answered as {@link AlphaMode#STRAIGHT} colour, as every reader answers. ImageIO decodes a
 * page stored with associated alpha to premultiplied colour, which is divided back to straight colour
 * with {@link PixelBuffer#toAlphaMode(AlphaMode)}'s rounding.
 */
@ApiStatus.Internal
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
    @SilentThrows
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

        DecodeBudget budget = DecodeBudget.open("TIFF");

        if (pages == 1) {
            budget.claim(reader.getWidth(0), reader.getHeight(0));
            BufferedImage image = reader.read(0);
            reader.dispose();
            return StaticImageData.of(PixelBuffer.wrap(image).toAlphaMode(AlphaMode.STRAIGHT));
        }

        ConcurrentList<ImageFrame> frames = Concurrent.newList();

        for (int i = 0; i < pages; i++) {
            budget.claim(reader.getWidth(i), reader.getHeight(i));
            BufferedImage image = reader.read(i);
            frames.add(ImageFrame.of(PixelBuffer.wrap(image).toAlphaMode(AlphaMode.STRAIGHT), PAGE_DELAY_MS, 0, 0, FrameDisposal.NONE, FrameBlend.SOURCE));
        }

        reader.dispose();

        return AnimatedImageData.builder()
            .withFrames(frames)
            .build();
    }

}
