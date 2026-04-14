package dev.simplified.image.transform;

import dev.simplified.collection.Concurrent;
import dev.simplified.collection.ConcurrentList;
import dev.simplified.image.data.AnimatedImageData;
import dev.simplified.image.data.ImageFrame;
import dev.simplified.image.pixel.PixelBuffer;
import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.awt.*;
import java.awt.image.BufferedImage;

/**
 * Normalizes animation frames of varying dimensions to a common canvas size.
 * <p>
 * When creating animated images from frames of different sizes, all frames must
 * have identical dimensions. This utility resizes and positions frames onto a
 * common canvas with configurable fit strategies.
 */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
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
                PixelBuffer copy = PixelBuffer.of(src.pixels().clone(), canvasW, canvasH, src.hasAlpha());
                normalized.add(ImageFrame.of(copy, frame.delayMs(), 0, 0, frame.disposal(), frame.blend()));
                continue;
            }

            BufferedImage srcImage = src.toBufferedImage();
            BufferedImage canvas = new BufferedImage(canvasW, canvasH, BufferedImage.TYPE_INT_ARGB);
            Graphics2D g2d = canvas.createGraphics();
            try {
                g2d.setComposite(AlphaComposite.Src);
                g2d.setColor(backgroundColor);
                g2d.fillRect(0, 0, canvasW, canvasH);

                g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
                g2d.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
                g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

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

                g2d.drawImage(srcImage, dx, dy, dw, dh, null);
            } finally {
                g2d.dispose();
            }

            normalized.add(ImageFrame.of(PixelBuffer.wrap(canvas), frame.delayMs(), 0, 0, frame.disposal(), frame.blend()));
        }

        return new AnimatedImageData.Builder()
            .withFrames(normalized)
            .withLoopCount(data.getLoopCount())
            .withBackgroundColor(data.getBackgroundColor())
            .build();
    }

}
