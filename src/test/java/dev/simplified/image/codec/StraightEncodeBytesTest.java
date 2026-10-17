package dev.simplified.image.codec;

import dev.simplified.image.AnimatedImageData;
import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFactory;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.PixelBuffer;
import dev.simplified.image.StaticImageData;
import dev.simplified.image.codec.bmp.BmpImageWriter;
import dev.simplified.image.codec.gif.GifImageWriter;
import dev.simplified.image.codec.ico.IcoImageWriter;
import dev.simplified.image.codec.jpeg.JpegImageWriter;
import dev.simplified.image.codec.png.PngImageWriter;
import dev.simplified.image.codec.pnm.PnmImageWriter;
import dev.simplified.image.codec.qoi.QoiImageWriter;
import dev.simplified.image.codec.tga.TgaImageWriter;
import dev.simplified.image.codec.tiff.TiffImageWriter;
import dev.simplified.image.codec.webp.WebPImageWriter;
import dev.simplified.image.frame.FrameBlend;
import dev.simplified.image.frame.FrameDisposal;
import dev.simplified.image.frame.ImageFrame;
import dev.simplified.image.option.ImageWriteOptions;
import dev.simplified.image.option.TiffWriteOptions;
import dev.simplified.image.option.WebPWriteOptions;
import dev.simplified.image.pixel.AlphaMode;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;

/**
 * Pins the bytes every writer writes for straight colour, which every image holds unless it is built
 * to say otherwise.
 * <p>
 * The digests cover the writers whose bytes come from this library's own code or ImageIO's Java
 * plugins: QOI, TGA, PNM, BMP, GIF, uncompressed TIFF and the WebP paths. PNG, ICO's embedded PNG,
 * Deflate-compressed TIFF and JPEG compress through the JDK's zlib and libjpeg, and a JDK built against
 * the system's copies of those writes other bytes for the same pixels, so those four are held only to
 * the bytes the same JVM writes for the same image declared straight, and through the factory.
 */
class StraightEncodeBytesTest {

    private static final int WIDTH = 16;
    private static final int HEIGHT = 8;

    /** The first row of every fixture frame: partial alpha at several levels, alpha 0 and 255 included. */
    private static final int[] ROW = {
        0x80800000, 0x40202020, 0x01010101, 0x00000000, 0xFF336699, 0x80404040, 0xFE7F3F00, 0x02010000
    };

    private static final TiffWriteOptions TIFF_UNCOMPRESSED = TiffWriteOptions.builder()
        .withCompression(TiffWriteOptions.Compression.NONE)
        .build();

    private static final WebPWriteOptions WEBP_LOSSY = WebPWriteOptions.builder()
        .isLossless(false)
        .build();

    private static final WebPWriteOptions WEBP_LOSSY_P_FRAMES = WebPWriteOptions.builder()
        .isLossless(false)
        .usePFrames(true)
        .withMotionSearchThreads(1)
        .build();

    private static final List<Case> CASES = List.of(
        new Case("QOI", ImageFormat.QOI, new QoiImageWriter(), null, false),
        new Case("TGA", ImageFormat.TGA, new TgaImageWriter(), null, false),
        new Case("PNM", ImageFormat.PNM, new PnmImageWriter(), null, false),
        new Case("BMP", ImageFormat.BMP, new BmpImageWriter(), null, false),
        new Case("GIF_STILL", ImageFormat.GIF, new GifImageWriter(), null, false),
        new Case("GIF_ANIMATED", ImageFormat.GIF, new GifImageWriter(), null, true),
        new Case("TIFF_NONE_STILL", ImageFormat.TIFF, new TiffImageWriter(), TIFF_UNCOMPRESSED, false),
        new Case("TIFF_NONE_ANIMATED", ImageFormat.TIFF, new TiffImageWriter(), TIFF_UNCOMPRESSED, true),
        new Case("WEBP_LOSSLESS_STILL", ImageFormat.WEBP, new WebPImageWriter(), null, false),
        new Case("WEBP_LOSSY_STILL", ImageFormat.WEBP, new WebPImageWriter(), WEBP_LOSSY, false),
        new Case("WEBP_LOSSLESS_ANIMATED", ImageFormat.WEBP, new WebPImageWriter(), null, true),
        new Case("WEBP_LOSSY_ANIMATED", ImageFormat.WEBP, new WebPImageWriter(), WEBP_LOSSY, true),
        new Case("WEBP_LOSSY_ANIMATED_P_FRAMES", ImageFormat.WEBP, new WebPImageWriter(), WEBP_LOSSY_P_FRAMES, true),
        new Case("PNG", ImageFormat.PNG, new PngImageWriter(), null, false),
        new Case("ICO", ImageFormat.ICO, new IcoImageWriter(), null, false),
        new Case("TIFF_DEFLATE_STILL", ImageFormat.TIFF, new TiffImageWriter(), null, false),
        new Case("TIFF_DEFLATE_ANIMATED", ImageFormat.TIFF, new TiffImageWriter(), null, true),
        new Case("JPEG", ImageFormat.JPEG, new JpegImageWriter(), null, false)
    );

    /** The SHA-256 of each case's output, keyed by case name, for the cases whose bytes are the JDK's own Java code. */
    private static final Map<String, String> DIGESTS = Map.ofEntries(
        Map.entry("QOI", "7a2ca0ab43e2c05b5086e4f52bfe5f014423046b028fbd3acd26dbd840be63f8"),
        Map.entry("TGA", "032e9598dd4d763b0cf7d9d3a2e775b693a550e0004f2fa59f95fab54112251b"),
        Map.entry("PNM", "aea6a6773f6f34bfa2d13b1ce3324defb5216d68e9cc8f39e698ae839174ebbe"),
        Map.entry("BMP", "9d0a50e5b26701b54f04122401c77c8066e61eca97bbb14462cd01eed9a3aa04"),
        Map.entry("GIF_STILL", "bb7e353daa238171fa46bf1f9e4cd6233a9f82c4441030ec3152caf7be35c206"),
        Map.entry("GIF_ANIMATED", "ab4b17741f43325ae07433d2ae7cdf0698e670f1e09f0b3e90908a6874dc84ce"),
        Map.entry("TIFF_NONE_STILL", "8dc6591655bd62af99881727fb9e64f5809dbec0cdf4838f4be5e1fce4d6f09e"),
        Map.entry("TIFF_NONE_ANIMATED", "ea02eaab2c1328f3b8756cc8c195613439ee61cee8028f546793ca4d13656155"),
        Map.entry("WEBP_LOSSLESS_STILL", "642ce11754e544861cfe26efc47cd952fc33e4c8fd38139fed9d87977722dc5d"),
        Map.entry("WEBP_LOSSY_STILL", "957cabbb37b4474f15bdb55a05658f42498cf91c4fc21df6ede4d36643edbf94"),
        Map.entry("WEBP_LOSSLESS_ANIMATED", "0b137cc105d6423c9378f3720288c9021e887e67f71893213025020ae81b90f0"),
        Map.entry("WEBP_LOSSY_ANIMATED", "4a817ee2674fe2fc6438d70365ba6a25928095c56e8a2cebd1c58702f8517112"),
        Map.entry("WEBP_LOSSY_ANIMATED_P_FRAMES", "87a3207aee2bf1d291e965f4a961700fcfd5afc2982e88ced35196fa3b25b114")
    );

    @TestFactory
    Stream<DynamicTest> straightBytesAreUnchanged() {
        return CASES.stream()
            .filter(entry -> DIGESTS.containsKey(entry.name()))
            .map(entry -> DynamicTest.dynamicTest(entry.name(), () ->
                assertThat(sha256(entry.write(image(entry.animated(), null))), equalTo(DIGESTS.get(entry.name())))
            ));
    }

    @TestFactory
    Stream<DynamicTest> declaredStraightWritesTheDefaultBytes() {
        return CASES.stream()
            .map(entry -> DynamicTest.dynamicTest(entry.name(), () ->
                assertThat(entry.write(image(entry.animated(), AlphaMode.STRAIGHT)), equalTo(entry.write(image(entry.animated(), null))))
            ));
    }

    @TestFactory
    Stream<DynamicTest> factoryWritesTheWriterBytes() {
        return CASES.stream()
            .map(entry -> DynamicTest.dynamicTest(entry.name(), () -> {
                ImageData image = image(entry.animated(), null);
                assertThat(ImageFactory.toByteArray(image, entry.format(), entry.options()), equalTo(entry.write(image)));
            }));
    }

    // ──── fixtures ────

    /**
     * The still fixture, or the three-frame animation when {@code animated} is set, built the way an
     * image is built without a convention when {@code alphaMode} is {@code null} and declaring it
     * otherwise.
     */
    private static @NotNull ImageData image(boolean animated, @Nullable AlphaMode alphaMode) {
        if (!animated)
            return alphaMode == null ? StaticImageData.of(frame(0)) : StaticImageData.of(frame(0), alphaMode);

        AnimatedImageData.Builder builder = AnimatedImageData.builder();

        for (int index = 0; index < 3; index++)
            builder.withFrame(ImageFrame.of(frame(index), 100, 0, 0, FrameDisposal.RESTORE_TO_BACKGROUND, FrameBlend.SOURCE));

        if (alphaMode != null)
            builder.withAlphaMode(alphaMode);

        return builder.build();
    }

    /**
     * A frame whose first row is {@link #ROW} twice over and whose other rows are a pattern of partial
     * alpha. Frame {@code index} differs from frame 0 only in a 4x4 block, so consecutive frames of the
     * animation differ in part of the canvas.
     */
    private static @NotNull PixelBuffer frame(int index) {
        int[] pixels = new int[WIDTH * HEIGHT];

        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                boolean changed = index > 0 && x >= 4 * index && x < 4 * index + 4 && y >= 2 && y < 6;
                pixels[y * WIDTH + x] = y == 0 ? ROW[x % ROW.length] : pattern(x, y, changed ? index : 0);
            }
        }

        return PixelBuffer.of(pixels, WIDTH, HEIGHT);
    }

    private static int pattern(int x, int y, int seed) {
        int a = (x * 37 + y * 23 + seed * 51) & 0xFF;
        int r = (x * 16 + y * 3 + seed * 40) & 0xFF;
        int g = (y * 32 + x * 5 + seed * 20) & 0xFF;
        int b = (x * y * 5 + seed * 60) & 0xFF;
        return (a << 24) | (r << 16) | (g << 8) | b;
    }

    private static @NotNull String sha256(byte @NotNull [] bytes) throws NoSuchAlgorithmException {
        return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
    }

    /**
     * One writer called one way.
     *
     * @param name the case's name, and its key in {@link #DIGESTS}
     * @param format the format the writer writes
     * @param writer the writer
     * @param options the options it is called with, or {@code null} for its defaults
     * @param animated whether it writes the animation rather than the still
     */
    private record Case(@NotNull String name, @NotNull ImageFormat format, @NotNull ImageWriter writer, @Nullable ImageWriteOptions options, boolean animated) {

        byte @NotNull [] write(@NotNull ImageData data) {
            return this.options == null ? this.writer.write(data) : this.writer.write(data, this.options);
        }

    }

}
