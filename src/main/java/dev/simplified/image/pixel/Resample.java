package dev.simplified.image.pixel;

/**
 * Resampling filters for image scaling operations.
 * <p>
 * Filters differ in their trade-off between speed and output quality. {@link #NEAREST} is fastest
 * and preserves hard edges; {@link #BILINEAR} is a reasonable middle ground; {@link #BICUBIC}
 * produces the highest quality output at the greatest cost; {@link #AREA} averages a downscale
 * without aliasing, which is what a supersampled render needs to be scaled down by.
 */
public enum Resample {

    /**
     * Area averaging. Along an axis that shrinks, each output pixel is the average of every source pixel
     * it covers, each weighted by the share of it covered, so an integer-ratio downscale averages whole
     * blocks and no source pixel is skipped. Along an axis that grows or keeps its size, it reads the
     * source as {@link #NEAREST} does. Colour is averaged in the convention the pixels are held in.
     */
    AREA,

    /**
     * Bicubic resampling using a 4x4 neighbourhood. Produces the smoothest output, at the highest
     * computational cost. Best for downscaling photographs or any content with fine gradients.
     */
    BICUBIC,

    /**
     * Bilinear resampling using a 2x2 neighbourhood. Produces smooth output without the ringing
     * artifacts sometimes introduced by {@link #BICUBIC}. A reasonable default for general
     * purpose scaling.
     */
    BILINEAR,

    /**
     * Nearest-neighbor resampling. Output pixel {@code (x, y)} of a {@code w x h} result takes source
     * pixel {@code (x * W / w, y * H / h)} of a {@code W x H} source, each division rounding down: the
     * top-left pixel of the source area the output pixel covers, which is not always the one whose
     * centre lies closest. A 3x downscale keeps source pixels 0, 3 and 6, where the closest centres are
     * those of 1, 4 and 7. Preserves hard edges and keeps pixel-art crisp, but introduces aliasing on
     * smooth content.
     */
    NEAREST

}
