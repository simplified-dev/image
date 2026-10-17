package dev.simplified.image;

import dev.simplified.annotations.AccessLevel;
import dev.simplified.annotations.Getter;
import dev.simplified.annotations.NamingStyle;
import dev.simplified.annotations.RequiredArgsConstructor;
import dev.simplified.collection.Concurrent;
import dev.simplified.collection.ConcurrentList;
import dev.simplified.image.frame.ImageFrame;
import dev.simplified.image.pixel.AlphaMode;
import dev.simplified.image.pixel.PixelBuffer;
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
    @Getter(style = NamingStyle.FLUENT)
    private final boolean hasAlpha;
    private final @NotNull ConcurrentList<ImageFrame> frames;
    /** the convention the frame's pixels are held in */
    private final @NotNull AlphaMode alphaMode;

    /**
     * Wraps a {@link PixelBuffer} holding straight colour as static image data.
     *
     * @param pixels the source pixel data
     * @return a new static image data instance
     */
    public static @NotNull StaticImageData of(@NotNull PixelBuffer pixels) {
        return of(pixels, AlphaMode.STRAIGHT);
    }

    /**
     * Wraps a {@link PixelBuffer} holding pixels in the given convention as static image data.
     *
     * @param pixels the source pixel data
     * @param alphaMode the convention the pixels are held in
     * @return a new static image data instance
     */
    public static @NotNull StaticImageData of(@NotNull PixelBuffer pixels, @NotNull AlphaMode alphaMode) {
        ImageFrame frame = ImageFrame.of(pixels, 0);
        ConcurrentList<ImageFrame> frames = Concurrent.newList();
        frames.add(frame);

        return new StaticImageData(pixels.width(), pixels.height(), pixels.hasAlpha(), frames, alphaMode);
    }

    /**
     * Wraps a {@link BufferedImage} as static image data.
     *
     * @param image the source image
     * @return a new static image data instance
     */
    public static @NotNull StaticImageData of(@NotNull BufferedImage image) {
        return of(PixelBuffer.wrap(image));
    }

    @Override
    public boolean isAnimated() {
        return false;
    }

}
