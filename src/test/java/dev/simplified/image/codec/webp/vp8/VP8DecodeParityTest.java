package dev.simplified.image.codec.webp.vp8;

import dev.simplified.image.ImageFactory;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.PixelBuffer;
import dev.simplified.image.StaticImageData;
import dev.simplified.image.codec.webp.LibwebpOracle;
import dev.simplified.image.option.WebPWriteOptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

/**
 * Pins this library's VP8 decode to libwebp's, pixel for pixel. A lossy WebP that libwebp encodes,
 * under each filter, segment and quality setting, and one this library encodes are each read back
 * by both decoders; a sample predicted from the wrong edge value, a sub-block predictor, a filtered
 * or unfiltered edge, or a rounding in the colour conversion shows as a pixel that differs.
 */
class VP8DecodeParityTest {

    /**
     * A frame neither of whose sides is a multiple of 16, mixing smooth shading, noise, a hard
     * diagonal, fine detail and a flat patch, so the encoders pick from every mode and the loop
     * filter meets every kind of edge.
     */
    private static final PixelBuffer SCENE = scene(170, 134);

    @Test
    @DisplayName("libwebp's lossy output decodes to libwebp's own pixels")
    void libwebpOutputDecodesAsLibwebpDoes() throws IOException {
        String[][] settings = {
            {},
            { "method=6", "sns_strength=0", "segments=1" },
            { "sns_strength=100" },
            { "filter_type=0" },
            { "filter_sharpness=5" },
            { "filter_strength=100", "filter_sharpness=2" },
        };
        for (int quality : new int[] { 10, 50, 90 })
            for (String[] setting : settings)
                assertDecodesAsLibwebp(LibwebpOracle.encodeLossyFile(SCENE, quality, setting),
                    "libwebp at quality " + quality + " " + String.join(" ", setting));
    }

    @Test
    @DisplayName("this library's lossy output decodes to the pixels libwebp decodes it to")
    void ownOutputDecodesAsLibwebpDoes() throws IOException {
        StaticImageData image = StaticImageData.of(SCENE);
        for (float quality : new float[] { 0.1f, 0.5f, 0.9f, 1.0f }) {
            for (boolean autoSegment : new boolean[] { false, true }) {
                byte[] webp = ImageFactory.toByteArray(image, ImageFormat.WEBP, WebPWriteOptions.builder()
                    .isLossless(false).withQuality(quality).withAutoSegment(autoSegment).build());
                assertDecodesAsLibwebp(webp, "this library at quality " + quality + (autoSegment ? " with autoSegment" : ""));
            }
        }
    }

    @Test
    @DisplayName("the encoder reconstructs every frame exactly as the decoder does")
    void encoderReconstructsAsDecoderDoes() {
        for (float quality : new float[] { 0.1f, 0.5f, 0.9f }) {
            VP8EncoderSession encoder = new VP8EncoderSession();
            VP8DecoderSession decoder = new VP8DecoderSession();
            for (int frame = 0; frame < 4; frame++) {
                PixelBuffer pixels = PixelBuffer.create(SCENE.width(), SCENE.height());
                for (int y = 0; y < SCENE.height(); y++)
                    for (int x = 0; x < SCENE.width(); x++)
                        pixels.setPixel(x, y, SCENE.getPixel(Math.min(x + 3 * frame, SCENE.width() - 1), y));
                decoder.decode(encoder.encode(pixels, quality, frame == 0));
                String at = "quality " + quality + " frame " + frame + " ";
                assertPlane(at + "Y", encoder.refY, decoder.refY);
                assertPlane(at + "U", encoder.refU, decoder.refU);
                assertPlane(at + "V", encoder.refV, decoder.refV);
            }
        }
    }

    @Test
    @DisplayName("a skipped B_PRED macroblock leaves the Y2 context where a skipped 16x16 one clears it")
    void skippedMacroblockContexts() {
        // libwebp's VP8DecodeMB clears every non-zero context of a skipped macroblock but keeps the
        // Y2 one when the macroblock is B_PRED, which codes no Y2 block. No stream libwebp's encoder
        // wrote for these tests skips a B_PRED macroblock beside one that reads that context, so
        // the rule is held here directly.
        int[] top = { 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        int[] left = { 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        VP8Decoder.clearNzForSkip(top, left, true);
        assertThat(top, is(new int[] { 0, 0, 0, 0, 0, 0, 0, 0, 1 }));
        assertThat(left, is(new int[] { 0, 0, 0, 0, 0, 0, 0, 0, 1 }));

        Arrays.fill(top, 1);
        Arrays.fill(left, 1);
        VP8Decoder.clearNzForSkip(top, left, false);
        assertThat(top, is(new int[9]));
        assertThat(left, is(new int[9]));
    }

    private static void assertPlane(String what, short[] encoder, short[] decoder) {
        if (encoder.length != decoder.length)
            throw new AssertionError(what + ": the encoder holds " + encoder.length + " samples, the decoder " + decoder.length);
        for (int i = 0; i < encoder.length; i++)
            if (encoder[i] != decoder[i])
                throw new AssertionError(what + ": sample " + i + " reconstructs to " + encoder[i]
                    + " in the encoder and " + decoder[i] + " in the decoder");
    }

    /**
     * Decodes a still WebP through this library and through libwebp, and fails on the first pixel
     * where they part.
     */
    private static void assertDecodesAsLibwebp(byte[] webp, String what) throws IOException {
        int w = SCENE.width();
        int h = SCENE.height();
        int[] expected = LibwebpOracle.decodeArgb(webp, w, h);
        PixelBuffer actual = ImageFactory.fromByteArray(webp).toPixelBuffer();
        int differing = 0;
        int first = -1;
        for (int i = 0; i < expected.length; i++) {
            if (actual.getPixel(i % w, i / w) != expected[i]) {
                if (differing == 0)
                    first = i;
                differing++;
            }
        }
        if (differing > 0)
            throw new AssertionError(String.format(
                "%s: %d of %d pixels differ from libwebp's decode, the first at (%d, %d), %08x where libwebp has %08x",
                what, differing, expected.length, first % w, first / w, actual.getPixel(first % w, first / w), expected[first]));
    }

    private static PixelBuffer scene(int w, int h) {
        Random random = new Random(7);
        PixelBuffer pixels = PixelBuffer.create(w, h);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                double shade = 128 + 50 * Math.sin(x * 0.11 + y * 0.03) + 30 * Math.sin(y * 0.23) + 20 * Math.cos((x - y) * 0.31);
                int r = (int) shade + random.nextInt(17) - 8;
                int g = (int) (shade * 0.8) + 24 + random.nextInt(17) - 8;
                int b = 255 - (int) shade + random.nextInt(17) - 8;
                if (x < 32 && y >= h - 32) {
                    r = 90;
                    g = 125;
                    b = 60;
                } else if (Math.floorMod(x - 2 * y, 29) < 2) {
                    r = 230;
                    g = 40;
                    b = 40;
                } else if (x >= w - 60 && y < 40 && ((x / 3 + y / 3) & 1) == 0) {
                    r = g = b = 240;
                }
                pixels.setPixel(x, y, 0xFF000000 | Math.clamp(r, 0, 255) << 16 | Math.clamp(g, 0, 255) << 8 | Math.clamp(b, 0, 255));
            }
        }
        return pixels;
    }

}
