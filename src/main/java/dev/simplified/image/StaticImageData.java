package dev.simplified.image;

import dev.simplified.collection.Concurrent;
import dev.simplified.collection.ConcurrentList;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.experimental.Accessors;
import org.jetbrains.annotations.NotNull;

import java.awt.image.BufferedImage;

/**
 * A single-frame image.
 */
@Getter
@RequiredArgsConstructor(access = AccessLevel.PRIVATE)
public class StaticImageData implements ImageData {

    private final int width;
    private final int height;
    @Accessors(fluent = true)
    private final boolean hasAlpha;
    private final @NotNull ConcurrentList<ImageFrame> frames;

    /**
     * Wraps a {@link BufferedImage} as static image data.
     *
     * @param image the source image
     * @return a new static image data instance
     */
    public static @NotNull StaticImageData of(@NotNull BufferedImage image) {
        boolean alpha = image.getColorModel().hasAlpha();
        ImageFrame frame = ImageFrame.of(image, 0);
        ConcurrentList<ImageFrame> frames = Concurrent.newList();
        frames.add(frame);

        return new StaticImageData(image.getWidth(), image.getHeight(), alpha, frames);
    }

    @Override
    public boolean isAnimated() {
        return false;
    }

}
