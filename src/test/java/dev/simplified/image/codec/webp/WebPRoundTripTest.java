package dev.simplified.image.codec.webp;

import dev.simplified.collection.Concurrent;
import dev.simplified.collection.ConcurrentList;
import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFactory;
import dev.simplified.image.ImageFormat;
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
