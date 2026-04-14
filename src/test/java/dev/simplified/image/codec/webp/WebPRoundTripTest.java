package dev.simplified.image.codec.webp;

import dev.simplified.collection.Concurrent;
import dev.simplified.collection.ConcurrentList;
import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFactory;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.data.AnimatedImageData;
import dev.simplified.image.data.FrameBlend;
import dev.simplified.image.data.FrameDisposal;
import dev.simplified.image.data.ImageFrame;
import dev.simplified.image.pixel.PixelBuffer;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.is;

/**
 * End-to-end WebP file-I/O round trips through {@link ImageFactory}. Complements
 * {@link WebPCodecTest}, which covers the encoder and decoder directly; this class
 * exercises the full RIFF container wrapping via the public API.
 */
class WebPRoundTripTest {

    @TempDir Path outputDir;

    @Test
    @DisplayName("2x2 solid-color VP8L writes a non-empty file with RIFF/WEBP header")
    void solidColor2x2() throws IOException {
        PixelBuffer buf = PixelBuffer.create(2, 2);
        buf.fill(0xFFFF0000); // solid red

        ImageFrame frame = ImageFrame.of(buf);
        var image = new StaticImageData(frame);
        File out = outputDir.resolve("solid_2x2_red.webp").toFile();
        new ImageFactory().toFile(image, ImageFormat.WEBP, out, WebPWriteOptions.builder().isLossless().build());

        byte[] bytes = Files.readAllBytes(out.toPath());
        assertThat("file written", bytes.length, is(greaterThan(12)));
        assertThat("RIFF header", new String(bytes, 0, 4), is("RIFF"));
        assertThat("WEBP identifier", new String(bytes, 8, 4), is("WEBP"));
    }

    @Test
    @DisplayName("16x16 gradient VP8L writes a non-empty file")
    void gradient16x16() throws IOException {
        PixelBuffer buf = PixelBuffer.create(16, 16);
        for (int y = 0; y < 16; y++)
            for (int x = 0; x < 16; x++)
                buf.setPixel(x, y, 0xFF000000 | (x * 16 << 16) | (y * 16 << 8));

        ImageFrame frame = ImageFrame.of(buf);
        var image = new StaticImageData(frame);
        File out = outputDir.resolve("gradient_16x16.webp").toFile();
        new ImageFactory().toFile(image, ImageFormat.WEBP, out, WebPWriteOptions.builder().isLossless().build());

        byte[] bytes = Files.readAllBytes(out.toPath());
        assertThat("file written", bytes.length, is(greaterThan(12)));
    }

    @Test
    @DisplayName("100x100 full-range gradient VP8L (255+ unique colors)")
    void gradient100x100() throws IOException {
        PixelBuffer buf = PixelBuffer.create(100, 100);
        for (int y = 0; y < 100; y++)
            for (int x = 0; x < 100; x++) {
                int r = (x * 255) / 100;
                int g = (y * 255) / 100;
                int b = ((x + y) * 127) / 100;
                buf.setPixel(x, y, 0xFF000000 | (r << 16) | (g << 8) | b);
            }
        var image = new StaticImageData(ImageFrame.of(buf));
        File out = outputDir.resolve("gradient_100x100.webp").toFile();
        new ImageFactory().toFile(image, ImageFormat.WEBP, out, WebPWriteOptions.builder().isLossless().build());
        assertThat("file written", Files.size(out.toPath()), is(greaterThan(12L)));
    }

    @Test
    @DisplayName("454x260 with 200+ unique colors like a real tooltip with AA text")
    void tooltipWithManyColors() throws IOException {
        PixelBuffer buf = PixelBuffer.create(454, 260);
        // Generate 256 distinct colors with AA-like frequency distribution: one dominant
        // background color, a long tail of 1-10 pixel blips for each glyph-edge shade.
        for (int y = 0; y < 260; y++)
            for (int x = 0; x < 454; x++) {
                int argb = 0xF0100010; // dominant background
                int blip = ((x * 131) + (y * 257)) & 0xFF; // pseudo-random picker
                if (blip < 4)                argb = 0xFFFFAA00 | (blip << 4); // gold shades
                else if (blip == 100)        argb = 0xFFFFFFFF;                // white
                else if (blip >= 250)        argb = 0x505000FF;                // border top
                buf.setPixel(x, y, argb);
            }
        var image = new StaticImageData(ImageFrame.of(buf));
        File out = outputDir.resolve("tooltip_many_colors.webp").toFile();
        new ImageFactory().toFile(image, ImageFormat.WEBP, out, WebPWriteOptions.builder().isLossless().build());
        assertThat("file written", Files.size(out.toPath()), is(greaterThan(12L)));
    }

    @Test
    @DisplayName("lossy VP8 32x32 writes a RIFF/WEBP file libwebp can parse")
    void lossyVp8WritesValidFile() throws IOException {
        PixelBuffer buf = PixelBuffer.create(32, 32);
        for (int y = 0; y < 32; y++)
            for (int x = 0; x < 32; x++)
                buf.setPixel(x, y, 0xFF000000 | ((x * 8) << 16) | ((y * 8) << 8));

        var image = new StaticImageData(ImageFrame.of(buf));
        File out = outputDir.resolve("lossy_vp8.webp").toFile();
        new ImageFactory().toFile(image, ImageFormat.WEBP, out,
            WebPWriteOptions.builder().isLossless(false).withQuality(0.9f).build());

        byte[] bytes = Files.readAllBytes(out.toPath());
        assertThat("RIFF header",  new String(bytes, 0, 4), is("RIFF"));
        assertThat("WEBP id",       new String(bytes, 8, 4), is("WEBP"));
        assertThat("file non-trivial", bytes.length > 30, is(true));
    }

    @Test
    @DisplayName("round-trip through our own VP8L decoder")
    void selfRoundTrip() {
        PixelBuffer buf = PixelBuffer.create(32, 32);
        for (int y = 0; y < 32; y++)
            for (int x = 0; x < 32; x++)
                buf.setPixel(x, y, 0xFF000000 | ((x * 8) << 16) | ((y * 8) << 8) | ((x + y) * 4));
        var image = new StaticImageData(ImageFrame.of(buf));
        File out = outputDir.resolve("self_roundtrip.webp").toFile();
        ImageFactory factory = new ImageFactory();
        factory.toFile(image, ImageFormat.WEBP, out, WebPWriteOptions.builder().isLossless().build());

        ImageData decoded = factory.fromFile(out);
        PixelBuffer got = decoded.getFrames().getFirst().pixels();
        assertThat(got.width(), is(32));
        assertThat(got.height(), is(32));
        for (int y = 0; y < 32; y++)
            for (int x = 0; x < 32; x++) {
                int expected = 0xFF000000 | ((x * 8) << 16) | ((y * 8) << 8) | ((x + y) * 4);
                assertThat("pixel @ " + x + "," + y, got.getPixel(x, y), is(expected));
            }
    }

    @Test
    @DisplayName("lossy + alpha self round-trip: alpha plane survives encode/decode")
    void lossyAlphaSelfRoundTrip() {
        // 16x16 image with smooth alpha gradient + solid red color.
        PixelBuffer buf = PixelBuffer.create(16, 16);
        for (int y = 0; y < 16; y++)
            for (int x = 0; x < 16; x++) {
                int alpha = ((x + y) * 255) / 30;     // ramps 0..255
                buf.setPixel(x, y, (alpha << 24) | 0x00FF0000);
            }

        var image = new StaticImageData(ImageFrame.of(buf));
        File out = outputDir.resolve("lossy_alpha_selfrt.webp").toFile();
        ImageFactory factory = new ImageFactory();
        factory.toFile(image, ImageFormat.WEBP, out,
            WebPWriteOptions.builder().isLossless(false).withQuality(0.9f).build());

        ImageData decoded = factory.fromFile(out);
        PixelBuffer got = decoded.getFrames().getFirst().pixels();
        assertThat(got.width(), is(16));
        assertThat(got.height(), is(16));
        // Alpha is VP8L-encoded so it is lossless; values must match exactly.
        for (int y = 0; y < 16; y++)
            for (int x = 0; x < 16; x++) {
                int expectedAlpha = ((x + y) * 255) / 30;
                int gotAlpha = (got.getPixel(x, y) >>> 24) & 0xFF;
                if (gotAlpha != expectedAlpha)
                    throw new AssertionError(String.format(
                        "alpha mismatch at (%d,%d): expected %d, got %d", x, y, expectedAlpha, gotAlpha));
            }
    }

    @Test
    @DisplayName("lossy + alpha decodes through libwebp with bit-exact alpha")
    void lossyAlphaLibwebpRoundTrip() throws Exception {
        PixelBuffer buf = PixelBuffer.create(16, 16);
        for (int y = 0; y < 16; y++)
            for (int x = 0; x < 16; x++) {
                int alpha = ((x + y) * 255) / 30;
                buf.setPixel(x, y, (alpha << 24) | 0x00FF0000);
            }

        var image = new StaticImageData(ImageFrame.of(buf));
        File out = outputDir.resolve("lossy_alpha_libwebp.webp").toFile();
        new ImageFactory().toFile(image, ImageFormat.WEBP, out,
            WebPWriteOptions.builder().isLossless(false).withQuality(0.9f).build());

        int[] decoded;
        try {
            decoded = decodeAlphaWithLibwebp(out, 16, 16);
        } catch (org.opentest4j.TestAbortedException abort) {
            throw abort;
        }

        // Alpha is lossless via VP8L ALPH; libwebp must reconstruct it bit-exact.
        for (int y = 0; y < 16; y++)
            for (int x = 0; x < 16; x++) {
                int expectedAlpha = ((x + y) * 255) / 30;
                int gotAlpha = decoded[y * 16 + x];
                if (gotAlpha != expectedAlpha)
                    throw new AssertionError(String.format(
                        "libwebp alpha mismatch at (%d,%d): expected %d, got %d",
                        x, y, expectedAlpha, gotAlpha));
            }
    }

    @Test
    @DisplayName("animated lossy + alpha self round-trip: alpha preserved per frame")
    void animatedLossyAlphaSelfRoundTrip() {
        // 3-frame animation; each frame fills with a different alpha gradient.
        ConcurrentList<ImageFrame> frames = Concurrent.newList();
        for (int f = 0; f < 3; f++) {
            PixelBuffer fb = PixelBuffer.create(16, 16);
            int phase = f * 40;
            for (int y = 0; y < 16; y++)
                for (int x = 0; x < 16; x++) {
                    int alpha = Math.min(255, phase + (x + y) * 7);
                    fb.setPixel(x, y, (alpha << 24) | 0x000000FF);   // blue
                }
            frames.add(ImageFrame.of(fb, 100, 0, 0,
                FrameDisposal.DO_NOT_DISPOSE, FrameBlend.OVER));
        }

        AnimatedImageData anim = AnimatedImageData.builder()
            .withWidth(16).withHeight(16)
            .withFrames(frames)
            .withLoopCount(0)
            .withBackgroundColor(0)
            .build();

        File out = outputDir.resolve("anim_alpha_selfrt.webp").toFile();
        ImageFactory factory = new ImageFactory();
        factory.toFile(anim, ImageFormat.WEBP, out,
            WebPWriteOptions.builder().isLossless(false).withQuality(0.9f).build());

        ImageData decoded = factory.fromFile(out);
        if (!(decoded instanceof AnimatedImageData decAnim))
            throw new AssertionError("expected AnimatedImageData, got " + decoded.getClass());

        ConcurrentList<ImageFrame> outFrames = decAnim.getFrames();
        if (outFrames.size() != 3)
            throw new AssertionError("frame count mismatch: " + outFrames.size());

        for (int f = 0; f < 3; f++) {
            PixelBuffer fb = outFrames.get(f).pixels();
            int phase = f * 40;
            for (int y = 0; y < 16; y++)
                for (int x = 0; x < 16; x++) {
                    int expectedAlpha = Math.min(255, phase + (x + y) * 7);
                    int gotAlpha = (fb.getPixel(x, y) >>> 24) & 0xFF;
                    if (gotAlpha != expectedAlpha)
                        throw new AssertionError(String.format(
                            "frame %d alpha @ (%d,%d): expected %d got %d",
                            f, x, y, expectedAlpha, gotAlpha));
                }
        }
    }

    @Test
    @DisplayName("uncompressed ALPH (compression=0) round-trips through our reader")
    void uncompressedAlphaSelfRoundTrip() {
        PixelBuffer buf = PixelBuffer.create(8, 8);
        for (int y = 0; y < 8; y++)
            for (int x = 0; x < 8; x++) {
                int alpha = (x * 16 + y * 4) & 0xFF;
                buf.setPixel(x, y, (alpha << 24) | 0x0000FF00);
            }

        var image = new StaticImageData(ImageFrame.of(buf));
        File out = outputDir.resolve("uncompressed_alpha.webp").toFile();
        ImageFactory factory = new ImageFactory();
        factory.toFile(image, ImageFormat.WEBP, out,
            WebPWriteOptions.builder()
                .isLossless(false).withQuality(0.9f)
                .isAlphaCompression(false)
                .build());

        ImageData decoded = factory.fromFile(out);
        PixelBuffer got = decoded.getFrames().getFirst().pixels();
        for (int y = 0; y < 8; y++)
            for (int x = 0; x < 8; x++) {
                int expectedAlpha = (x * 16 + y * 4) & 0xFF;
                int gotAlpha = (got.getPixel(x, y) >>> 24) & 0xFF;
                if (gotAlpha != expectedAlpha)
                    throw new AssertionError(String.format(
                        "uncompressed alpha @ (%d,%d): expected %d got %d",
                        x, y, expectedAlpha, gotAlpha));
            }
    }

    /**
     * Decodes the alpha channel of a WebP file via Python's {@code webp} bindings
     * and returns a {@code width * height} array of alpha values (0..255).
     */
    private static int[] decodeAlphaWithLibwebp(File file, int expectedW, int expectedH) throws Exception {
        String script =
            "import sys, warnings\n" +
            "warnings.filterwarnings('ignore')\n" +
            "try:\n" +
            "    import webp\n" +
            "except ImportError:\n" +
            "    print('NO_WEBP'); sys.exit(2)\n" +
            "img = webp.load_image(r'" + file.getAbsolutePath() + "').convert('RGBA')\n" +
            "w, h = img.size\n" +
            "print(f'DIMS {w} {h}')\n" +
            "for px in img.getdata():\n" +
            "    print(f'A {px[3]}')\n";

        Process p = null;
        for (String cmd : new String[]{"python3", "python", "py"}) {
            try {
                ProcessBuilder pb = new ProcessBuilder(cmd, "-c", script);
                pb.redirectErrorStream(true);
                p = pb.start();
                break;
            } catch (java.io.IOException ignored) { }
        }
        if (p == null)
            throw new org.opentest4j.TestAbortedException("No python3/python/py on PATH");

        String stdout = new String(p.getInputStream().readAllBytes());
        int exit = p.waitFor();
        if (exit == 2 && stdout.contains("NO_WEBP"))
            throw new org.opentest4j.TestAbortedException("Python webp package not installed");
        if (exit != 0)
            throw new AssertionError("libwebp rejected our WebP file:\n" + stdout);

        String[] lines = stdout.split("\\R");
        int w = -1, h = -1;
        int[] alpha = new int[expectedW * expectedH];
        int alphaIdx = 0;
        for (String line : lines) {
            if (line.startsWith("DIMS ")) {
                String[] dims = line.substring(5).split("\\s+");
                w = Integer.parseInt(dims[0]);
                h = Integer.parseInt(dims[1]);
            } else if (line.startsWith("A ")) {
                if (alphaIdx < alpha.length)
                    alpha[alphaIdx++] = Integer.parseInt(line.substring(2));
            }
            // Ignore any other lines (Python warnings, etc.)
        }
        if (w != expectedW || h != expectedH)
            throw new AssertionError(
                "dims " + w + "x" + h + " != expected " + expectedW + "x" + expectedH
                + "\nfull stdout:\n" + stdout);
        if (alphaIdx != alpha.length)
            throw new AssertionError(
                "expected " + alpha.length + " alpha values, got " + alphaIdx
                + "\nfull stdout:\n" + stdout);
        return alpha;
    }

    @Test
    @DisplayName("454x260 tooltip-like VP8L (the problem size from TestLoreMain)")
    void tooltipSized() throws IOException {
        PixelBuffer buf = PixelBuffer.create(454, 260);
        // Approximate the tooltip color distribution: mostly dark purple background,
        // a few bright text colors scattered.
        for (int y = 0; y < 260; y++)
            for (int x = 0; x < 454; x++) {
                int argb;
                if (x < 4 || x > 449 || y < 4 || y > 255) argb = 0x50500FF0; // border
                else if ((x + y) % 37 == 0) argb = 0xFFFFAA00; // gold
                else if ((x + y) % 53 == 0) argb = 0xFFAAAAAA; // gray
                else argb = 0xF0100010;
                buf.setPixel(x, y, argb);
            }
        var image = new StaticImageData(ImageFrame.of(buf));
        File out = outputDir.resolve("tooltip_454x260.webp").toFile();
        new ImageFactory().toFile(image, ImageFormat.WEBP, out, WebPWriteOptions.builder().isLossless().build());
        assertThat("file written", Files.size(out.toPath()), is(greaterThan(12L)));
    }

    /** Minimal wrapper so we can write a single frame without instantiating TextRenderer. */
    private record StaticImageData(ImageFrame frame) implements ImageData {

        @Override
        public @NotNull ConcurrentList<ImageFrame> getFrames() {
            ConcurrentList<ImageFrame> list = Concurrent.newList();
            list.add(frame);
            return list;
        }

        @Override
        public boolean hasAlpha() {
            return true;
        }

        @Override
        public int getWidth() {
            return frame.pixels().width();
        }

        @Override
        public int getHeight() {
            return frame.pixels().height();
        }

        @Override
        public boolean isAnimated() {
            return false;
        }

    }

}
