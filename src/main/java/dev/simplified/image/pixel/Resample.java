package dev.simplified.image.pixel;

/**
 * Resampling filters for image scaling operations.
 * <p>
 * Filters differ in their trade-off between speed and output quality. {@link #NEAREST} is fastest
 * and preserves hard edges; {@link #BILINEAR} is a reasonable middle ground; {@link #BICUBIC}
 * produces the highest quality output at the greatest cost.
 */
public enum Resample {

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
     * Nearest-neighbor resampling. Picks the source pixel whose centre is closest to the sampled
     * location. Preserves hard edges and keeps pixel-art crisp, but introduces aliasing on
     * smooth content.
     */
    NEAREST

}
