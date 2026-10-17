package dev.simplified.image.codec.gif;

import dev.simplified.image.AnimatedImageData;
import dev.simplified.image.ImageData;
import dev.simplified.image.PixelBuffer;
import dev.simplified.image.StaticImageData;
import dev.simplified.image.frame.FrameBlend;
import dev.simplified.image.frame.FrameDisposal;
import dev.simplified.image.frame.ImageFrame;
import dev.simplified.image.option.GifWriteOptions;
import dev.simplified.image.pixel.AlphaMode;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;

/**
 * Pins how {@link GifImageWriter} flattens partly transparent pixels onto its opaque background: a
 * premultiplied image's colour is laid over the background as premultiplied colour, rounded once,
 * and a straight image's as straight colour.
 * <p>
 * Every fixture holds fewer colours than a palette, so the palette is exact and a decoded pixel is the
 * flattened pixel itself.
 */
class GifFlattenTest {

    /** One premultiplied pixel per case the flatten treats differently, alpha 0 and 255 included. */
    private static final int[] PREMULTIPLIED = {
        0x80800000, 0x40202020, 0x01010101, 0x00000000, 0xFF336699, 0x80404040
    };

    /** A pixel {@link GifImageReader} decodes from the transparent palette slot. */
    private static final int CLEAR = 0x00000000;

    /** The rows of a fixture. */
    private static final int ROWS = 8;

    // ──── premultiplied colour over the background ────

    @Test
    @DisplayName("premultiplied colour is laid over a black background as it is")
    void premultipliedOverBlack() {
        assertFlattens(0x000000, false, AlphaMode.PREMULTIPLIED, 0xFF800000, 0xFF202020, 0xFF010101, 0xFF000000, 0xFF336699, 0xFF404040);
    }

    @Test
    @DisplayName("premultiplied colour is laid over a white background")
    void premultipliedOverWhite() {
        assertFlattens(0xFFFFFF, false, AlphaMode.PREMULTIPLIED, 0xFFFF7F7F, 0xFFDFDFDF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFF336699, 0xFFBFBFBF);
    }

    @Test
    @DisplayName("premultiplied colour is laid over a grey background, rounded once")
    void premultipliedOverGrey() {
        // Reading the premultiplied colour as straight would give FF480808 and FF101010 at the first
        // and third pixels.
        assertFlattens(0x101010, false, AlphaMode.PREMULTIPLIED, 0xFF880808, 0xFF2C2C2C, 0xFF111111, 0xFF101010, 0xFF336699, 0xFF484848);
    }

    @Test
    @DisplayName("premultiplied colour is laid over a background of differing channels")
    void premultipliedOverPurple() {
        assertFlattens(0x100010, false, AlphaMode.PREMULTIPLIED, 0xFF880008, 0xFF2C202C, 0xFF110111, 0xFF100010, 0xFF336699, 0xFF484048);
    }

    // ──── the transparent slot ────

    @Test
    @DisplayName("a pixel below the alpha threshold is transparent whichever convention the image holds")
    void transparentBelowThresholdEitherWay() {
        // Alpha 128 is the threshold: a pixel there is flattened rather than turned transparent, by the
        // rule of the convention the image holds.
        assertFlattens(0x101010, true, AlphaMode.PREMULTIPLIED, 0xFF880808, CLEAR, CLEAR, CLEAR, 0xFF336699, 0xFF484848);
        assertFlattens(0x101010, true, AlphaMode.STRAIGHT, 0xFF480808, CLEAR, CLEAR, CLEAR, 0xFF336699, 0xFF282828);
    }

    // ──── straight colour over the background ────

    @Test
    @DisplayName("straight colour is laid over the background as straight colour, rounded once")
    void straightOverGrey() {
        // Flooring the composite would give FF480707 and FF0F0F0F at the first and third pixels.
        assertFlattens(0x101010, false, AlphaMode.STRAIGHT, 0xFF480808, 0xFF141414, 0xFF101010, 0xFF101010, 0xFF336699, 0xFF282828);
    }

    // ──── animations ────

    @Test
    @DisplayName("every frame of a premultiplied animation is laid over the background as premultiplied colour")
    void premultipliedAnimationFlattensEveryFrame() {
        int[] reversed = reversed(PREMULTIPLIED);
        AnimatedImageData data = AnimatedImageData.builder()
            .withFrame(ImageFrame.of(rows(PREMULTIPLIED), 100, 0, 0, FrameDisposal.RESTORE_TO_BACKGROUND, FrameBlend.SOURCE))
            .withFrame(ImageFrame.of(rows(reversed), 100, 0, 0, FrameDisposal.RESTORE_TO_BACKGROUND, FrameBlend.SOURCE))
            .withAlphaMode(AlphaMode.PREMULTIPLIED)
            .build();
        int[] expected = { 0xFF880808, 0xFF2C2C2C, 0xFF111111, 0xFF101010, 0xFF336699, 0xFF484848 };

        ImageData decoded = new GifImageReader().read(new GifImageWriter().write(data, options(0x101010, false)));

        assertThat(decoded.getFrames().size(), equalTo(2));
        assertRows(decoded.getFrames().getFirst().pixels(), expected);
        assertRows(decoded.getFrames().getLast().pixels(), reversed(expected));
        assertThat(data.getFrames().getFirst().pixels().data(), equalTo(rows(PREMULTIPLIED).data()));
    }

    // ──── assertions ────

    private static void assertFlattens(int backgroundRgb, boolean transparent, @NotNull AlphaMode alphaMode, int @NotNull ... expected) {
        PixelBuffer pixels = rows(PREMULTIPLIED);
        byte[] gif = new GifImageWriter().write(StaticImageData.of(pixels, alphaMode), options(backgroundRgb, transparent));

        assertRows(new GifImageReader().read(gif).toPixelBuffer(), expected);
        assertThat(pixels.data(), equalTo(rows(PREMULTIPLIED).data()));
    }

    private static void assertRows(@NotNull PixelBuffer decoded, int @NotNull [] expected) {
        for (int y = 0; y < ROWS; y++) {
            for (int x = 0; x < expected.length; x++) {
                int actual = decoded.getPixel(x, y);
                // A transparent slot decodes to whatever colour the palette holds there, at alpha 0.
                int compared = (expected[x] >>> 24) == 0 ? actual & 0xFF000000 : actual;
                assertThat("pixel " + x + "," + y, Integer.toHexString(compared), equalTo(Integer.toHexString(expected[x])));
            }
        }
    }

    // ──── fixtures ────

    private static @NotNull GifWriteOptions options(int backgroundRgb, boolean transparent) {
        return GifWriteOptions.builder()
            .withBackgroundRgb(backgroundRgb)
            .isTransparent(transparent)
            .build();
    }

    /** A buffer {@link #ROWS} tall whose every row is {@code row}. */
    private static @NotNull PixelBuffer rows(int @NotNull [] row) {
        int[] pixels = new int[row.length * ROWS];

        for (int y = 0; y < ROWS; y++)
            System.arraycopy(row, 0, pixels, y * row.length, row.length);

        return PixelBuffer.of(pixels, row.length, ROWS);
    }

    private static int @NotNull [] reversed(int @NotNull [] pixels) {
        int[] out = new int[pixels.length];

        for (int i = 0; i < pixels.length; i++)
            out[i] = pixels[pixels.length - 1 - i];

        return out;
    }

}
