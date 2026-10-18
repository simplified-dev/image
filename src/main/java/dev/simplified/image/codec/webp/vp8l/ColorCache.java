package dev.simplified.image.codec.webp.vp8l;

import org.jetbrains.annotations.NotNull;

/**
 * A power-of-2 hash table of recently used ARGB pixel values for VP8L
 * encoding and decoding.
 * <p>
 * The color cache provides an additional symbol source during entropy coding,
 * allowing recently seen colors to be referenced by their hash index instead
 * of being encoded as full ARGB values.
 */
final class ColorCache {

    private final int @NotNull [] colors;
    private final int hashShift;

    /**
     * Creates a color cache with the given size in bits.
     *
     * @param sizeBits the log2 of the cache size (0 to disable, 1-11 for active cache)
     */
    ColorCache(int sizeBits) {
        this.colors = new int[sizeBits > 0 ? 1 << sizeBits : 0];
        this.hashShift = sizeBits > 0 ? 32 - sizeBits : 0;
    }

    /**
     * Returns the cached color at the given index.
     *
     * @param index the cache index
     * @return the ARGB color value
     */
    int lookup(int index) {
        return colors[index];
    }

    /**
     * Inserts a color into the cache, replacing any existing entry at the hash position.
     *
     * @param argb the ARGB color value
     */
    void insert(int argb) {
        if (colors.length != 0)
            colors[hashIndex(argb)] = argb;
    }

    /**
     * Returns the hash index for a given color.
     *
     * @param argb the ARGB color value
     * @return the cache index
     */
    int hashIndex(int argb) {
        return (argb * 0x1E35A7BD) >>> hashShift;
    }

    /**
     * Returns the number of entries in this cache.
     *
     * @return the cache size
     */
    int size() {
        return colors.length;
    }

    /**
     * Returns whether this cache is enabled.
     *
     * @return true if the cache has capacity > 0
     */
    boolean isEnabled() {
        return colors.length != 0;
    }

}
