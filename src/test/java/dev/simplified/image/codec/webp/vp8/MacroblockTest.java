package dev.simplified.image.codec.webp.vp8;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.function.IntBinaryOperator;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

/**
 * Pins how {@link Macroblock#fromARGB} takes each 8x8 chroma sample from its 2x2 block of pixels.
 */
class MacroblockTest {

    private static final int RED = 0xFFFF0000;
    private static final int BLUE = 0xFF0000FF;

    private static Macroblock convert(IntBinaryOperator pixelAt) {
        int[] pixels = new int[16 * 16];
        for (int y = 0; y < 16; y++)
            for (int x = 0; x < 16; x++)
                pixels[y * 16 + x] = pixelAt.applyAsInt(x, y);

        Macroblock mb = new Macroblock();
        mb.fromARGB(pixels, 0, 0, 16, 16);
        return mb;
    }

    @Test
    @DisplayName("each chroma sample averages its 2x2 block rather than taking the top-left pixel")
    void chromaAveragesTheBlock() {
        Macroblock mb = convert((x, y) -> (y & 1) == 0 ? RED : BLUE);

        // Red over blue averages to (127.5, 0, 127.5): Cb 165 and Cr 175. The top-left pixel alone, red,
        // would give Cb 90 and Cr 240.
        for (int i = 0; i < 64; i++) {
            assertThat(mb.cb[i], is((short) 165));
            assertThat(mb.cr[i], is((short) 175));
        }
    }

    @Test
    @DisplayName("a transparent pixel's colour does not reach the chroma of a partly transparent block")
    void transparentPixelsCarryNoChroma() {
        // The transparent green sits at each block's top-left, which the top-left pick would take
        // (Cb 54, Cr 34) and an unweighted average would mix in.
        Macroblock mb = convert((x, y) -> (y & 1) == 0 ? 0x0000FF00 : RED);

        for (int i = 0; i < 64; i++) {
            assertThat(mb.cb[i], is((short) 90));
            assertThat(mb.cr[i], is((short) 240));
        }
    }

    @Test
    @DisplayName("a uniform block keeps the chroma its one colour converts to")
    void uniformBlockKeepsItsColour() {
        int colour = 0xFF3C82C8;
        Macroblock mb = convert((x, y) -> colour);
        int r = 0x3C, g = 0x82, b = 0xC8;

        for (int i = 0; i < 64; i++) {
            assertThat(mb.cb[i], is((short) Math.clamp(((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128, 0, 255)));
            assertThat(mb.cr[i], is((short) Math.clamp(((112 * r - 94 * g - 18 * b + 128) >> 8) + 128, 0, 255)));
        }
    }

}
