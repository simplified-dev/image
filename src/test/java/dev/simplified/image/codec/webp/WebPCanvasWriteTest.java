package dev.simplified.image.codec.webp;

import dev.simplified.image.AnimatedImageData;
import dev.simplified.image.ImageData;
import dev.simplified.image.PixelBuffer;
import dev.simplified.image.frame.FrameBlend;
import dev.simplified.image.frame.FrameDisposal;
import dev.simplified.image.frame.ImageFrame;
import dev.simplified.image.option.WebPWriteOptions;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.greaterThanOrEqualTo;
import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.instanceOf;

/**
 * Pins how {@link WebPImageWriter} writes an animation whose frames do not each cover the canvas
 * from the origin and replace it: a decoded GIF's frames, smaller than the canvas and at offsets,
 * laid over what the frame before left and disposed to the background or to the previous canvas.
 * However it is encoded, each frame it decodes to is the canvas {@link ImageData#toCanvasFrames()}
 * composes for that frame.
 */
class WebPCanvasWriteTest {

    private static final int RED = 0xFFFF0000;
    private static final int BLUE = 0xFF0000FF;
    private static final int CLEAR = 0x00000000;

    // The GIF-like fixture's palette: greys, so a lossy frame's colour error is the codec's quantizing
    // rather than 4:2:0 chroma averaging saturated hues across a frame's edge.
    private static final int DARK = 0xFF303030;
    private static final int LIGHT = 0xFFD8D8D8;
    private static final int MID = 0xFF909090;
    private static final int SHADE = 0xFF606060;

    /** The canvas's width, even so that libwebp's player sees no odd-offset rounding. */
    private static final int WIDTH = 16;

    /** The canvas's height. */
    private static final int HEIGHT = 12;

    /** The least PSNR, in dB, a lossy frame's colour keeps against its composed canvas. */
    private static final double LOSSY_PSNR = 30.0;

    @Test
    @DisplayName("a GIF-like animation written losslessly decodes to its composed canvases")
    void losslessDecodesToTheComposedCanvases() {
        List<int[]> decoded = pictures(read(new WebPImageWriter().write(gifLike())));

        assertPictures(decoded, composed());
    }

    @Test
    @DisplayName("a GIF-like animation written lossy, with or without P-frames, decodes to its composed alpha and close to its colour")
    void lossyDecodesToTheComposedAlpha() {
        for (boolean pFrames : new boolean[] { false, true }) {
            WebPWriteOptions options = WebPWriteOptions.builder().isLossless(false).withQuality(0.9f).usePFrames(pFrames).build();
            List<int[]> decoded = pictures(read(new WebPImageWriter().write(gifLike(), options)));
            List<int[]> expected = composed();
            String how = pFrames ? "lossy with P-frames" : "lossy";

            assertPictures(alphas(decoded), alphas(expected));
            for (int i = 0; i < expected.size(); i++)
                assertThat(how + " frame " + i + " PSNR over its covered pixels", psnr(decoded.get(i), expected.get(i)), greaterThanOrEqualTo(LOSSY_PSNR));
        }
    }

    @Test
    @DisplayName("libwebp plays the written animation's frames to their composed alpha, lossless or lossy")
    void libwebpPlaysTheComposedAlpha() throws IOException {
        List<int[]> expected = alphaValues(composed());
        WebPWriteOptions[] writes = {
            WebPWriteOptions.builder().build(),
            WebPWriteOptions.builder().isLossless(false).withQuality(0.9f).build()
        };

        for (WebPWriteOptions options : writes) {
            Path file = Files.createTempFile("webp-canvas-", ".webp");
            try {
                Files.write(file, new WebPImageWriter().write(gifLike(), options));
                int[][] played = LibwebpOracle.decodeAnimatedAlpha(file, WIDTH, HEIGHT, expected.size());
                for (int i = 0; i < expected.size(); i++)
                    assertThat("lossless " + options.isLossless() + " frame " + i, hex(played[i]), equalTo(hex(expected.get(i))));
            } finally {
                Files.deleteIfExists(file);
            }
        }
    }

    @Test
    @DisplayName("a pixel an OVER frame leaves transparent shows the frame beneath it once written losslessly")
    void transparentPixelOfAnOverFrameShowsTheFrameBeneath() {
        int[] holed = new int[8 * 8];
        Arrays.fill(holed, BLUE);
        holed[3 * 8 + 3] = CLEAR;
        AnimatedImageData animation = AnimatedImageData.builder()
            .withWidth(8)
            .withHeight(8)
            .withFrame(ImageFrame.of(filled(8, 8, RED), 100, 0, 0, FrameDisposal.DO_NOT_DISPOSE, FrameBlend.SOURCE))
            .withFrame(ImageFrame.of(PixelBuffer.ofStraight(holed, 8, 8), 100, 0, 0, FrameDisposal.DO_NOT_DISPOSE, FrameBlend.OVER))
            .build();

        int shown = read(new WebPImageWriter().write(animation)).getFrames().getLast().pixels().getPixel(3, 3);

        assertThat(String.format("%08x", shown), equalTo(String.format("%08x", RED)));
    }

    // ──── fixtures ────

    /**
     * An animation as a decoded GIF holds it: frames smaller than the canvas, at offsets, each laid
     * over the canvas the frame before left:
     * <ul>
     *     <li>a 10x8 dark frame at (3, 1), transparent at its centre, left as drawn</li>
     *     <li>a 6x6 light square at (7, 4), transparent at its centre so the dark shows through,
     *     restored to the previous canvas once it has shown</li>
     *     <li>a 4x4 mid-grey frame at (1, 6), restored to the background once it has shown</li>
     *     <li>a 5x3 frame at (11, 9), shaded with a transparent stripe, left as drawn</li>
     * </ul>
     */
    private static @NotNull AnimatedImageData gifLike() {
        int[] holedDark = filled(10, 8, DARK).data().clone();
        for (int y = 3; y < 5; y++)
            for (int x = 4; x < 6; x++)
                holedDark[y * 10 + x] = CLEAR;

        int[] holedLight = filled(6, 6, LIGHT).data().clone();
        for (int y = 2; y < 4; y++)
            for (int x = 2; x < 4; x++)
                holedLight[y * 6 + x] = CLEAR;

        int[] striped = filled(5, 3, SHADE).data().clone();
        for (int x = 0; x < 5; x++)
            striped[5 + x] = CLEAR;

        return AnimatedImageData.builder()
            .withWidth(WIDTH)
            .withHeight(HEIGHT)
            .withFrame(ImageFrame.of(PixelBuffer.ofStraight(holedDark, 10, 8), 80, 3, 1, FrameDisposal.DO_NOT_DISPOSE, FrameBlend.OVER))
            .withFrame(ImageFrame.of(PixelBuffer.ofStraight(holedLight, 6, 6), 90, 7, 4, FrameDisposal.RESTORE_TO_PREVIOUS, FrameBlend.OVER))
            .withFrame(ImageFrame.of(filled(4, 4, MID), 100, 1, 6, FrameDisposal.RESTORE_TO_BACKGROUND, FrameBlend.OVER))
            .withFrame(ImageFrame.of(PixelBuffer.ofStraight(striped, 5, 3), 110, 11, 9, FrameDisposal.DO_NOT_DISPOSE, FrameBlend.OVER))
            .build();
    }

    /**
     * The canvas each frame of {@link #gifLike()} shows, as {@link ImageData#toCanvasFrames()}
     * composes it.
     */
    private static @NotNull List<int[]> composed() {
        return pictures(gifLike().toCanvasFrames());
    }

    private static @NotNull PixelBuffer filled(int width, int height, int argb) {
        int[] pixels = new int[width * height];
        Arrays.fill(pixels, argb);
        return PixelBuffer.ofStraight(pixels, width, height);
    }

    // ──── reading ────

    private static @NotNull AnimatedImageData read(byte @NotNull [] webp) {
        ImageData decoded = new WebPImageReader().read(webp);
        assertThat(decoded, instanceOf(AnimatedImageData.class));
        return (AnimatedImageData) decoded;
    }

    private static @NotNull List<int[]> pictures(@NotNull ImageData image) {
        List<int[]> pictures = new ArrayList<>();

        for (ImageFrame frame : image.getFrames())
            pictures.add(frame.pixels().data());

        return pictures;
    }

    /**
     * Returns each of {@code pictures} with every pixel's colour dropped and its alpha kept.
     */
    private static @NotNull List<int[]> alphas(@NotNull List<int[]> pictures) {
        List<int[]> alphas = new ArrayList<>();

        for (int[] picture : pictures)
            alphas.add(Arrays.stream(picture).map(argb -> argb & 0xFF000000).toArray());

        return alphas;
    }

    /**
     * Returns each of {@code pictures} as one alpha value from 0 to 255 per pixel.
     */
    private static @NotNull List<int[]> alphaValues(@NotNull List<int[]> pictures) {
        List<int[]> values = new ArrayList<>();

        for (int[] picture : pictures)
            values.add(Arrays.stream(picture).map(argb -> argb >>> 24).toArray());

        return values;
    }

    /**
     * Measures how closely {@code actual}'s colour follows {@code expected}'s over the pixels
     * {@code expected} covers at all, a transparent pixel's colour being no part of the picture.
     */
    private static double psnr(int @NotNull [] actual, int @NotNull [] expected) {
        double squared = 0;
        int samples = 0;

        for (int i = 0; i < expected.length; i++) {
            if (expected[i] >>> 24 == 0)
                continue;
            for (int shift = 0; shift < 24; shift += 8) {
                int difference = ((actual[i] >> shift) & 0xFF) - ((expected[i] >> shift) & 0xFF);
                squared += difference * difference;
                samples++;
            }
        }

        double mse = squared / samples;
        return mse == 0 ? Double.POSITIVE_INFINITY : 10 * Math.log10(255.0 * 255.0 / mse);
    }

    private static void assertPictures(@NotNull List<int[]> actual, @NotNull List<int[]> expected) {
        assertThat(actual, hasSize(expected.size()));

        for (int i = 0; i < expected.size(); i++)
            assertThat("frame " + i, hex(actual.get(i)), equalTo(hex(expected.get(i))));
    }

    private static @NotNull List<String> hex(int @NotNull [] picture) {
        return Arrays.stream(picture).mapToObj(argb -> String.format("%08x", argb)).toList();
    }

}
