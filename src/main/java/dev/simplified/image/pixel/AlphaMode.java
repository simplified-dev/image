package dev.simplified.image.pixel;

import dev.simplified.image.PixelBuffer;

/**
 * How a pixel's colour channels relate to its alpha - the convention a {@link PixelBuffer} holds its
 * pixels in, set by the factory that makes the buffer and fixed for its life, which a filter mixing
 * neighbouring pixels reads to mix them correctly.
 * <p>
 * The two hold the same picture differently. A colour mixed with a transparent neighbour has to lose
 * the neighbour's share in proportion to how transparent it is: {@link #STRAIGHT} pixels carry no
 * record of that, so each is scaled by its alpha for the mix and divided back after, where
 * {@link #PREMULTIPLIED} pixels carry it already and mix as they are.
 */
public enum AlphaMode {

    /**
     * Colour held at full strength, independent of alpha - what an image decoded from a file holds,
     * and what {@link PixelBuffer#straight(int, int)} and {@link PixelBuffer#ofStraight(int[], int, int)}
     * make. A fully transparent pixel's colour channels carry nothing.
     */
    STRAIGHT,

    /**
     * Colour held already scaled by the pixel's alpha, so no channel exceeds the alpha - what drawing
     * with {@link BlendMode#NORMAL} onto a transparent surface leaves behind, what
     * {@link BlendMode#PREMULTIPLIED} lays down, and what {@link PixelBuffer#canvas(int, int)} and
     * {@link PixelBuffer#ofPremultiplied(int[], int, int)} make.
     */
    PREMULTIPLIED

}
