package dev.simplified.image.pixel;

/**
 * How a pixel's colour channels relate to its alpha - the convention a buffer's pixels are held in,
 * which a filter mixing neighbouring pixels has to know to mix them correctly.
 * <p>
 * The two hold the same picture differently. A colour mixed with a transparent neighbour has to lose
 * the neighbour's share in proportion to how transparent it is: {@link #STRAIGHT} pixels carry no
 * record of that, so each is scaled by its alpha for the mix and divided back after, where
 * {@link #PREMULTIPLIED} pixels carry it already and mix as they are.
 */
public enum AlphaMode {

    /**
     * Colour held at full strength, independent of alpha - what an image decoded from a file holds.
     * A fully transparent pixel's colour channels carry nothing.
     */
    STRAIGHT,

    /**
     * Colour held already scaled by the pixel's alpha, so no channel exceeds the alpha - what drawing
     * with {@link BlendMode#NORMAL} onto a transparent surface leaves behind, and what
     * {@link BlendMode#PREMULTIPLIED} lays down.
     */
    PREMULTIPLIED

}
