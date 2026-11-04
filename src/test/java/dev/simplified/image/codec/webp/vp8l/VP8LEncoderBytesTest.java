package dev.simplified.image.codec.webp.vp8l;

import dev.simplified.image.PixelBuffer;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.function.Supplier;
import java.util.stream.Stream;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.greaterThan;

/**
 * Pins the bytes {@link VP8LEncoder} writes for fixed images: seven holding more than 256 colours so
 * no palette applies, one of 16 colours that takes it, and one pixel whose trials tie.
 * <p>
 * Each digest was taken from an encoder that applied the transform and walked the LZ77 parse afresh
 * for every color-cache and meta-Huffman trial, so they hold the encoder, which does both once per
 * transform and shares them across its trials, to that encoder's bytes. The unforced cases pin the
 * trial {@link VP8LEncoder#encode(PixelBuffer)} chooses as well as its bytes; the forced cases pin
 * each transform's header, sub-image and body with the color cache and meta-Huffman on.
 * <p>
 * Four images pin how the trials share a transform. The color cache wins the cached picks and the
 * copied rows, and meta-Huffman wins the split halves, each a later trial of its transform than the
 * first, so their digests hold a trial to its transform's own header, pixels and parse rather than
 * what an earlier trial left in them, and of the three the copied rows' parse holds matches. The
 * one-pixel image codes to 12 bytes three ways, each different - no transform and subtract-green
 * with the cache off, and no transform with it on - so its digest pins the order the trials run in
 * and that a tie keeps the first. A trial that wrote to the parse's longest matches would change
 * only a later single-group trial that the dynamic-programming pass decides, and no fixture's
 * chosen trial is one, so {@link #theTrialsOfOneTransformLeaveItUnchanged()} runs every trial of
 * one shared transform and holds the transform to what it was.
 */
class VP8LEncoderBytesTest {

    /** The order {@link VP8LEncoder#encode(PixelBuffer)} tries its transforms in. */
    private static final List<VP8LEncoder.TransformMode> TRIAL_ORDER = List.of(
        VP8LEncoder.TransformMode.NONE,
        VP8LEncoder.TransformMode.PALETTE,
        VP8LEncoder.TransformMode.PREDICTOR,
        VP8LEncoder.TransformMode.SUBTRACT_GREEN,
        VP8LEncoder.TransformMode.CROSS_COLOR
    );

    /** The color-cache sizes {@link VP8LEncoder#encode(PixelBuffer)} tries, its outer loop. */
    private static final int[] CACHE_BITS_TRIALS = { 0, 10 };

    /** The meta-Huffman tile sizes {@link VP8LEncoder#encode(PixelBuffer)} tries, inside the cache loop. */
    private static final int[] META_TILE_BITS_TRIALS = { 0, 8 };

    private static final List<Image> IMAGES = List.of(
        new Image("NOISY_GRADIENT", VP8LEncoderBytesTest::noisyGradient, true),
        new Image("TRANSLUCENT_NOISE", VP8LEncoderBytesTest::translucentNoise, true),
        new Image("CYCLE_320", VP8LEncoderBytesTest::cycle320, true),
        new Image("SMOOTH_GRADIENT", VP8LEncoderBytesTest::smoothGradient, true),
        new Image("CACHED_PICKS", VP8LEncoderBytesTest::cachedPicks, true),
        new Image("SPLIT_HALVES", VP8LEncoderBytesTest::splitHalves, true),
        new Image("COPIED_ROWS", VP8LEncoderBytesTest::copiedRows, true),
        new Image("PALETTE_16", VP8LEncoderBytesTest::palette16, false),
        new Image("ONE_PIXEL", VP8LEncoderBytesTest::onePixel, false)
    );

    private static final List<Case> CASES = List.of(
        new Case("NOISY_GRADIENT", VP8LEncoderBytesTest::noisyGradient, null),
        new Case("TRANSLUCENT_NOISE", VP8LEncoderBytesTest::translucentNoise, null),
        new Case("CYCLE_320", VP8LEncoderBytesTest::cycle320, null),
        new Case("SMOOTH_GRADIENT", VP8LEncoderBytesTest::smoothGradient, null),
        new Case("CACHED_PICKS", VP8LEncoderBytesTest::cachedPicks, null),
        new Case("SPLIT_HALVES", VP8LEncoderBytesTest::splitHalves, null),
        new Case("COPIED_ROWS", VP8LEncoderBytesTest::copiedRows, null),
        new Case("PALETTE_16", VP8LEncoderBytesTest::palette16, null),
        new Case("ONE_PIXEL", VP8LEncoderBytesTest::onePixel, null),
        new Case("NOISY_GRADIENT_NONE", VP8LEncoderBytesTest::noisyGradient, VP8LEncoder.TransformMode.NONE),
        new Case("PALETTE_16_PALETTE", VP8LEncoderBytesTest::palette16, VP8LEncoder.TransformMode.PALETTE),
        new Case("NOISY_GRADIENT_PREDICTOR", VP8LEncoderBytesTest::noisyGradient, VP8LEncoder.TransformMode.PREDICTOR),
        new Case("NOISY_GRADIENT_SUBTRACT_GREEN", VP8LEncoderBytesTest::noisyGradient, VP8LEncoder.TransformMode.SUBTRACT_GREEN),
        new Case("NOISY_GRADIENT_CROSS_COLOR", VP8LEncoderBytesTest::noisyGradient, VP8LEncoder.TransformMode.CROSS_COLOR)
    );

    /** The SHA-256 of each case's output, keyed by case name. */
    private static final Map<String, String> DIGESTS = Map.ofEntries(
        Map.entry("NOISY_GRADIENT", "6269727c933704b8206ec115ae926df0ef0a7b1aafea875af6ca6f6e4622363e"),
        Map.entry("TRANSLUCENT_NOISE", "8f4f5cb9d6aff77490f5294d85743f1b7a96a5e24ebaa3b103ba9caa3bbb1e48"),
        Map.entry("CYCLE_320", "30aadfeb1fbfe7550266901457b2fe0af3a06e28b3a5d2528cf95982e6cf6b33"),
        Map.entry("SMOOTH_GRADIENT", "fdab050669f2ed93b41299d79b0b1e29591a42414ce7d174228b2d5086003d1b"),
        Map.entry("CACHED_PICKS", "444b00839c11f5f147e12efb7dd806009a962c638528d875ba36d46fea5ee8d6"),
        Map.entry("SPLIT_HALVES", "39f44660b416b74ebb59a5214464f81443b11254c61ca63be2159e579f5d95ad"),
        Map.entry("COPIED_ROWS", "d8a7a0c50e56f5eb5737e39b9729ec42179c71ab726c59b4cf983eed576b226c"),
        Map.entry("PALETTE_16", "eeeaf45e202dfe418b6c260a45c713e6af7dab9fb10a27993b0af8d40e463c3d"),
        Map.entry("ONE_PIXEL", "745fd448c000140e096947520944da5b6d41a69ba680f91b293638b082a70f7b"),
        Map.entry("NOISY_GRADIENT_NONE", "cef8adbc014cad3dd4498788398cad5011cc39340fc401c332e9e1ab1eb65ddb"),
        Map.entry("PALETTE_16_PALETTE", "9fceb1d9480b61694a1790e1fea70be9ce60f19db3301cfe15a7ff6783b029b3"),
        Map.entry("NOISY_GRADIENT_PREDICTOR", "baadecccb4e1e059647dba4dba8127821466c91a4ce5ba4e147ade774278b903"),
        Map.entry("NOISY_GRADIENT_SUBTRACT_GREEN", "31915f0e215ee09e95a4e8bc810701bfd5074f89e5385e6df14001aeff213e1e"),
        Map.entry("NOISY_GRADIENT_CROSS_COLOR", "b1d3b98b66a4adcab9a2845b1a10f32411f6cc51c783a5c8987e5fa0029e30cf")
    );

    @TestFactory
    Stream<DynamicTest> encodedBytesAreUnchanged() {
        return CASES.stream()
            .map(entry -> DynamicTest.dynamicTest(entry.name(), () ->
                assertThat(sha256(entry.encode()), equalTo(DIGESTS.get(entry.name())))
            ));
    }

    @TestFactory
    Stream<DynamicTest> theChosenEncodingIsTheFirstShortestTrial() {
        return IMAGES.stream()
            .map(image -> DynamicTest.dynamicTest(image.name(), () -> {
                PixelBuffer pixels = image.pixels().get();
                boolean paletteEligible = !image.overPalette();
                byte[] best = null;

                for (int cacheBits : CACHE_BITS_TRIALS) {
                    for (int metaTileBits : META_TILE_BITS_TRIALS) {
                        for (VP8LEncoder.TransformMode mode : TRIAL_ORDER) {
                            if (mode == VP8LEncoder.TransformMode.PALETTE && !paletteEligible)
                                continue;

                            byte[] candidate = VP8LEncoder.encode(pixels, mode, cacheBits, metaTileBits);

                            if (best == null || candidate.length < best.length)
                                best = candidate;
                        }
                    }
                }

                assertThat(VP8LEncoder.encode(pixels), equalTo(best));
            }));
    }

    @TestFactory
    Stream<DynamicTest> theTrialsOfOneTransformLeaveItUnchanged() {
        return IMAGES.stream()
            .flatMap(image -> TRIAL_ORDER.stream()
                .filter(mode -> mode != VP8LEncoder.TransformMode.PALETTE || !image.overPalette())
                .map(mode -> DynamicTest.dynamicTest(image.name() + "_" + mode, () -> {
                    PixelBuffer pixels = image.pixels().get();
                    VP8LEncoder.Transformed shared = VP8LEncoder.transform(pixels, mode);
                    VP8LEncoder.Parse parse = shared.parse();
                    int[] bodyPixels = shared.bodyPixels().clone();
                    int[] stepLength = parse.stepLength().clone();
                    int[] stepDistance = parse.stepDistance().clone();
                    int[] matchLength = parse.matchLength().clone();
                    int[] matchDistance = parse.matchDistance().clone();

                    for (int cacheBits : CACHE_BITS_TRIALS) {
                        for (int metaTileBits : META_TILE_BITS_TRIALS) {
                            assertThat(
                                cacheBits + "/" + metaTileBits,
                                VP8LEncoder.encodeTransformed(shared, cacheBits, metaTileBits),
                                equalTo(VP8LEncoder.encode(pixels, mode, cacheBits, metaTileBits))
                            );
                        }
                    }

                    assertThat("body pixels", shared.bodyPixels(), equalTo(bodyPixels));
                    assertThat("step lengths", parse.stepLength(), equalTo(stepLength));
                    assertThat("step distances", parse.stepDistance(), equalTo(stepDistance));
                    assertThat("match lengths", parse.matchLength(), equalTo(matchLength));
                    assertThat("match distances", parse.matchDistance(), equalTo(matchDistance));
                })));
    }

    @TestFactory
    Stream<DynamicTest> theUnpalettedImagesHoldMoreThan256Colours() {
        return IMAGES.stream()
            .filter(Image::overPalette)
            .map(image -> DynamicTest.dynamicTest(image.name(), () -> {
                Set<Integer> colours = new HashSet<>();

                for (int pixel : image.pixels().get().data())
                    colours.add(pixel);

                assertThat(colours.size(), greaterThan(256));
            }));
    }

    // ──── fixtures ────

    /** A 96x64 diagonal gradient with up to 6 levels of seeded noise on each colour channel. */
    private static @NotNull PixelBuffer noisyGradient() {
        Random random = new Random(0x5EED);
        int[] pixels = new int[96 * 64];

        for (int y = 0; y < 64; y++) {
            for (int x = 0; x < 96; x++) {
                int r = Math.clamp(x * 2 + random.nextInt(13) - 6, 0, 255);
                int g = Math.clamp(y * 3 + random.nextInt(13) - 6, 0, 255);
                int b = Math.clamp(x + y + random.nextInt(13) - 6, 0, 255);
                pixels[y * 96 + x] = 0xFF000000 | (r << 16) | (g << 8) | b;
            }
        }

        return PixelBuffer.ofStraight(pixels, 96, 64);
    }

    /** A 48x48 field of seeded random colours, alpha included. */
    private static @NotNull PixelBuffer translucentNoise() {
        Random random = new Random(17);
        int[] pixels = new int[48 * 48];

        for (int i = 0; i < pixels.length; i++)
            pixels[i] = random.nextInt();

        return PixelBuffer.ofStraight(pixels, 48, 48);
    }

    /** A 64x64 run of 320 distinct opaque colours repeating in scan order, the kind the color cache and LZ77 reward. */
    private static @NotNull PixelBuffer cycle320() {
        int[] pixels = new int[64 * 64];

        for (int i = 0; i < pixels.length; i++) {
            int c = i % 320;
            pixels[i] = 0xFF000000 | ((c * 7) & 0xFF) << 16 | ((c >> 1) & 0xFF) << 8 | (c & 0xFF);
        }

        return PixelBuffer.ofStraight(pixels, 64, 64);
    }

    /** A 128x64 smooth opaque gradient, the kind the predictor reduces to small residuals. */
    private static @NotNull PixelBuffer smoothGradient() {
        int[] pixels = new int[128 * 64];

        for (int y = 0; y < 64; y++)
            for (int x = 0; x < 128; x++)
                pixels[y * 128 + x] = 0xFF000000 | (x * 2) << 16 | (y * 4) << 8 | (x + y);

        return PixelBuffer.ofStraight(pixels, 128, 64);
    }

    /**
     * A 32x32 image whose every pixel is one of 400 seeded opaque colours picked at random, so its
     * colours recur without the runs LZ77 matches and the color cache codes the repeats.
     */
    private static @NotNull PixelBuffer cachedPicks() {
        Random random = new Random(400);
        int[] colours = new int[400];

        for (int i = 0; i < colours.length; i++)
            colours[i] = 0xFF000000 | random.nextInt(0x01000000);

        int[] pixels = new int[32 * 32];

        for (int i = 0; i < pixels.length; i++)
            pixels[i] = colours[random.nextInt(colours.length)];

        return PixelBuffer.ofStraight(pixels, 32, 32);
    }

    /**
     * A 512x16 image of seeded noise in red on its left 256 columns and in blue on its right 256,
     * whose two 256x256 meta-Huffman tiles each want a code of their own.
     */
    private static @NotNull PixelBuffer splitHalves() {
        Random random = new Random(9);
        int[] pixels = new int[512 * 16];

        for (int y = 0; y < 16; y++)
            for (int x = 0; x < 512; x++)
                pixels[y * 512 + x] = x < 256 ? 0xFF000000 | random.nextInt(256) << 16 : 0xFF000000 | random.nextInt(256);

        return PixelBuffer.ofStraight(pixels, 512, 16);
    }

    /**
     * A 32x32 image of seeded random picks from 400 opaque colours, row by row, where each row from
     * the fifth on copies a seeded earlier row half the time: LZ77 matches the copies and the color
     * cache codes the recurring colours of the rest.
     */
    private static @NotNull PixelBuffer copiedRows() {
        Random random = new Random(3);
        int[] colours = new int[400];

        for (int i = 0; i < colours.length; i++)
            colours[i] = 0xFF000000 | random.nextInt(0x01000000);

        int[] pixels = new int[32 * 32];

        for (int y = 0; y < 32; y++) {
            if (y >= 4 && random.nextDouble() < 0.5) {
                System.arraycopy(pixels, random.nextInt(y) * 32, pixels, y * 32, 32);
            } else {
                for (int x = 0; x < 32; x++)
                    pixels[y * 32 + x] = colours[random.nextInt(colours.length)];
            }
        }

        return PixelBuffer.ofStraight(pixels, 32, 32);
    }

    /** A 40x24 diagonal pattern of 16 opaque colours, which the palette packs two to a pixel. */
    private static @NotNull PixelBuffer palette16() {
        int[] colours = new int[16];

        for (int i = 0; i < 16; i++)
            colours[i] = 0xFF000000 | (i * 16) << 16 | (i * 8) << 8 | (255 - i * 16);

        int[] pixels = new int[40 * 24];

        for (int y = 0; y < 24; y++)
            for (int x = 0; x < 40; x++)
                pixels[y * 40 + x] = colours[(x * 5 + y * 3) & 0xF];

        return PixelBuffer.ofStraight(pixels, 40, 24);
    }

    /** A 1x1 image of one seeded opaque colour. */
    private static @NotNull PixelBuffer onePixel() {
        return PixelBuffer.ofStraight(new int[] { 0xFF000000 | new Random(1).nextInt(0x01000000) }, 1, 1);
    }

    private static @NotNull String sha256(byte @NotNull [] bytes) throws NoSuchAlgorithmException {
        return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
    }

    /**
     * One fixture image.
     *
     * @param name the image's name
     * @param pixels the image, made afresh on every call
     * @param overPalette whether the image holds more than 256 colours, so no palette applies
     */
    private record Image(@NotNull String name, @NotNull Supplier<PixelBuffer> pixels, boolean overPalette) { }

    /**
     * One image encoded one way.
     *
     * @param name the case's name, and its key in {@link #DIGESTS}
     * @param pixels the image, made afresh on every call
     * @param mode the transform forced with a 10-bit color cache and 256x256 meta-Huffman tiles, or
     *             {@code null} for the encoder's own choice of every trial
     */
    private record Case(@NotNull String name, @NotNull Supplier<PixelBuffer> pixels, VP8LEncoder.@Nullable TransformMode mode) {

        byte @NotNull [] encode() {
            return this.mode == null
                ? VP8LEncoder.encode(this.pixels.get())
                : VP8LEncoder.encode(this.pixels.get(), this.mode, 10, 8);
        }

    }

}
