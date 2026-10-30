package dev.simplified.image.frame;

import dev.simplified.annotations.UtilityClass;
import dev.simplified.collection.Concurrent;
import dev.simplified.collection.ConcurrentList;
import dev.simplified.image.AnimatedImageData;
import dev.simplified.image.PixelBuffer;
import dev.simplified.image.pixel.AlphaMode;
import dev.simplified.image.pixel.Resample;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.awt.Color;

/**
 * Normalizes animation frames of varying dimensions to a common canvas size.
 * <p>
 * When creating animated images from frames of different sizes, all frames must
 * have identical dimensions. This utility resizes and positions frames onto a
 * common canvas with configurable fit strategies.
 */
@UtilityClass
public final class FrameNormalizer {

    /**
     * Normalizes frames using default settings - min dimensions across frames,
     * {@link FitMode#CONTAIN}, transparent background, no upscaling.
     *
     * @param data the animated image data containing frames to normalize
     * @return a new animated image data with normalized frame dimensions
     */
    public static @NotNull AnimatedImageData normalize(@NotNull AnimatedImageData data) {
        return normalize(data, null, null, FitMode.CONTAIN, false, new Color(0, 0, 0, 0));
    }

    /**
     * Normalizes frames to a common canvas with configurable strategy.
     * <p>
     * Frames are scaled bicubically in the convention {@code data} is held in, and the background fill,
     * given as straight colour, is premultiplied for an animation held in {@link AlphaMode#PREMULTIPLIED}.
     * A scaled frame replaces the fill it covers, transparent pixels included. Every frame of the result
     * holds the convention {@code data} holds.
     *
     * @param data the animated image data containing frames to normalize
     * @param targetWidth the target canvas width, or null to derive from frames
     * @param targetHeight the target canvas height, or null to derive from frames
     * @param fitMode how to fit frames onto the canvas
     * @param allowUpscale whether to allow upscaling smaller images
     * @param backgroundColor the background fill color
     * @return a new animated image data with normalized frame dimensions
     */
    public static @NotNull AnimatedImageData normalize(
        @NotNull AnimatedImageData data,
        @Nullable Integer targetWidth,
        @Nullable Integer targetHeight,
        @NotNull FitMode fitMode,
        boolean allowUpscale,
        @NotNull Color backgroundColor
    ) {
        ConcurrentList<ImageFrame> frames = data.getFrames();
        AlphaMode alphaMode = data.getAlphaMode();

        boolean needMinDimensions = (targetWidth == null || targetWidth <= 0)
            || (targetHeight == null || targetHeight <= 0);

        int minW = Integer.MAX_VALUE;
        int minH = Integer.MAX_VALUE;

        if (needMinDimensions) {
            for (ImageFrame frame : frames) {
                minW = Math.min(minW, frame.pixels().width());
                minH = Math.min(minH, frame.pixels().height());
            }
        }

        int canvasW = (targetWidth != null && targetWidth > 0) ? targetWidth : minW;
        int canvasH = (targetHeight != null && targetHeight > 0) ? targetHeight : minH;

        ConcurrentList<ImageFrame> normalized = Concurrent.newList();

        for (ImageFrame frame : frames) {
            PixelBuffer src = frame.pixels();
            int sw = src.width();
            int sh = src.height();

            // Fast path: no transformation needed
            if (sw == canvasW && sh == canvasH && fitMode != FitMode.COVER) {
                normalized.add(ImageFrame.of(src.copy(), frame.delayMs(), 0, 0, frame.disposal(), frame.blend()));
                continue;
            }

            int dw, dh, dx, dy;

            switch (fitMode) {
                case COVER -> {
                    double scale = Math.max((double) canvasW / sw, (double) canvasH / sh);
                    if (!allowUpscale) scale = Math.min(1.0, scale);
                    dw = Math.max(1, (int) Math.round(sw * scale));
                    dh = Math.max(1, (int) Math.round(sh * scale));
                    dx = (canvasW - dw) / 2;
                    dy = (canvasH - dh) / 2;
                }
                case STRETCH -> {
                    if (allowUpscale) {
                        dw = canvasW;
                        dh = canvasH;
                        dx = 0;
                        dy = 0;
                    } else {
                        dw = Math.min(canvasW, sw);
                        dh = Math.min(canvasH, sh);
                        dx = (canvasW - dw) / 2;
                        dy = (canvasH - dh) / 2;
                    }
                }
                default -> { // CONTAIN
                    double scale = Math.min((double) canvasW / sw, (double) canvasH / sh);
                    if (!allowUpscale) scale = Math.min(1.0, scale);
                    dw = Math.max(1, (int) Math.round(sw * scale));
                    dh = Math.max(1, (int) Math.round(sh * scale));
                    dx = (canvasW - dw) / 2;
                    dy = (canvasH - dh) / 2;
                }
            }

            PixelBuffer fill = PixelBuffer.create(canvasW, canvasH);
            fill.fill(backgroundColor.getRGB());
            PixelBuffer canvas = fill.toAlphaMode(alphaMode);

            PixelBuffer scaled = src.resize(dw, dh, Resample.BICUBIC);
            replace(canvas, scaled, dx, dy);
            normalized.add(ImageFrame.of(canvas, frame.delayMs(), 0, 0, frame.disposal(), frame.blend()));
        }

        return new AnimatedImageData.Builder()
            .withFrames(normalized)
            .withLoopCount(data.getLoopCount())
            .withBackgroundColor(data.getBackgroundColor())
            .build();
    }

    /**
     * Copies {@code source} onto {@code canvas} with its top-left corner at {@code (x, y)}, replacing
     * the canvas pixels it covers - transparent ones included - and clipping to the canvas.
     *
     * @param canvas the buffer written to
     * @param source the pixels copied
     * @param x the canvas x coordinate of the source's left edge
     * @param y the canvas y coordinate of the source's top edge
     */
    private static void replace(@NotNull PixelBuffer canvas, @NotNull PixelBuffer source, int x, int y) {
        int x0 = Math.max(0, x);
        int x1 = Math.min(canvas.width(), x + source.width());
        if (x0 >= x1) return;

        for (int row = Math.max(0, y); row < Math.min(canvas.height(), y + source.height()); row++)
            System.arraycopy(source.data(), (row - y) * source.width() + (x0 - x), canvas.data(), row * canvas.width() + x0, x1 - x0);
    }

}
