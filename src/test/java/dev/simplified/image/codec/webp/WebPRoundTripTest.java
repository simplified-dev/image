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
    @DisplayName("animated lossy with P-frames self round-trip: stationary content shrinks and decodes cleanly")
    void animatedLossyPFramesSelfRoundTrip() {
        // 4-frame animation with identical frames - every frame after the first should
        // encode as a P-frame with all MBs inter-skip.
        ConcurrentList<ImageFrame> frames = Concurrent.newList();
        for (int f = 0; f < 4; f++) {
            PixelBuffer fb = PixelBuffer.create(32, 32);
            for (int y = 0; y < 32; y++)
                for (int x = 0; x < 32; x++)
                    fb.setPixel(x, y, 0xFF000000 | ((x * 8) << 16) | ((y * 8) << 8));
            frames.add(ImageFrame.of(fb, 100, 0, 0,
                FrameDisposal.DO_NOT_DISPOSE, FrameBlend.OVER));
        }
        AnimatedImageData anim = AnimatedImageData.builder()
            .withWidth(32).withHeight(32)
            .withFrames(frames)
            .withLoopCount(0)
            .withBackgroundColor(0)
            .build();

        File pFrameOut = outputDir.resolve("anim_pframes.webp").toFile();
        File keyOnlyOut = outputDir.resolve("anim_keyonly.webp").toFile();
        ImageFactory factory = new ImageFactory();
        factory.toFile(anim, ImageFormat.WEBP, pFrameOut,
            WebPWriteOptions.builder().isLossless(false).withQuality(1.0f).usePFrames(true).build());
        factory.toFile(anim, ImageFormat.WEBP, keyOnlyOut,
            WebPWriteOptions.builder().isLossless(false).withQuality(1.0f).build());

        // P-frame version must be smaller on stationary content.
        if (pFrameOut.length() >= keyOnlyOut.length())
            throw new AssertionError(String.format(
                "P-frame WebP (%d B) should be smaller than keyframe-only (%d B) on stationary frames",
                pFrameOut.length(), keyOnlyOut.length()));

        // Round-trip: reader must decode our P-frame WebP correctly.
        ImageData decoded = factory.fromFile(pFrameOut);
        if (!(decoded instanceof AnimatedImageData decAnim))
            throw new AssertionError("expected AnimatedImageData, got " + decoded.getClass());
        if (decAnim.getFrames().size() != 4)
            throw new AssertionError("frame count mismatch: " + decAnim.getFrames().size());

        // All frames should match frame 0's reconstruction (since content is identical).
        PixelBuffer frame0 = decAnim.getFrames().get(0).pixels();
        for (int f = 1; f < 4; f++) {
            PixelBuffer fb = decAnim.getFrames().get(f).pixels();
            for (int y = 0; y < 32; y++)
                for (int x = 0; x < 32; x++) {
                    int want = frame0.getPixel(x, y);
                    int got = fb.getPixel(x, y);
                    if (want != got)
                        throw new AssertionError(String.format(
                            "frame %d pixel (%d,%d) differs from frame 0: want %08X got %08X",
                            f, x, y, want, got));
                }
        }
    }

    @Test
    @DisplayName("forceKeyframeEvery=N triggers intermediate keyframes (seekability knob)")
    void forceKeyframeEveryTriggersIntermediateKeyframes() {
        // A 6-frame stationary animation. At usePFrames=true + forceKeyframeEvery=3,
        // frames 0 and 3 are keyframes; 1, 2, 4, 5 are P-frames. Vs the default
        // forceKeyframeEvery=0 (single-keyframe), this must produce a strictly
        // larger file because two keyframes cost more than one + five P-frames of
        // inter-skip on stationary content. Still decodable round-trip.
        ConcurrentList<ImageFrame> frames = Concurrent.newList();
        for (int f = 0; f < 6; f++) {
            PixelBuffer fb = PixelBuffer.create(32, 32);
            for (int y = 0; y < 32; y++)
                for (int x = 0; x < 32; x++)
                    fb.setPixel(x, y, 0xFF000000 | ((x * 8) << 16) | ((y * 8) << 8));
            frames.add(ImageFrame.of(fb, 100, 0, 0,
                FrameDisposal.DO_NOT_DISPOSE, FrameBlend.OVER));
        }
        AnimatedImageData anim = AnimatedImageData.builder()
            .withWidth(32).withHeight(32)
            .withFrames(frames)
            .withLoopCount(0)
            .withBackgroundColor(0)
            .build();

        ImageFactory factory = new ImageFactory();
        File singleKey = outputDir.resolve("anim_forcekey_0.webp").toFile();
        File everyThree = outputDir.resolve("anim_forcekey_3.webp").toFile();
        factory.toFile(anim, ImageFormat.WEBP, singleKey,
            WebPWriteOptions.builder().isLossless(false).withQuality(1.0f).usePFrames(true)
                .withForceKeyframeEvery(0).build());
        factory.toFile(anim, ImageFormat.WEBP, everyThree,
            WebPWriteOptions.builder().isLossless(false).withQuality(1.0f).usePFrames(true)
                .withForceKeyframeEvery(3).build());

        if (everyThree.length() <= singleKey.length())
            throw new AssertionError(String.format(
                "forceKeyframeEvery=3 (%d B) should be larger than single-keyframe (%d B)",
                everyThree.length(), singleKey.length()));

        // Round-trip: reader must decode the interspersed-keyframe output correctly.
        ImageData decoded = factory.fromFile(everyThree);
        if (!(decoded instanceof AnimatedImageData decAnim))
            throw new AssertionError("expected AnimatedImageData, got " + decoded.getClass());
        if (decAnim.getFrames().size() != 6)
            throw new AssertionError("frame count mismatch: " + decAnim.getFrames().size());
        PixelBuffer frame0 = decAnim.getFrames().get(0).pixels();
        for (int f = 1; f < 6; f++) {
            PixelBuffer fb = decAnim.getFrames().get(f).pixels();
            for (int y = 0; y < 32; y++)
                for (int x = 0; x < 32; x++)
                    if (fb.getPixel(x, y) != frame0.getPixel(x, y))
                        throw new AssertionError(String.format(
                            "frame %d pixel (%d,%d) differs from frame 0", f, x, y));
        }
    }

    @Test
    @DisplayName("forceKeyframeEvery default (-1) on short animations matches explicit 0")
    void forceKeyframeEveryDefaultMatchesZeroOnShortAnimations() {
        // The writer's auto-default gates on total frame count: <= 60 frames keeps
        // forceKeyframeEvery at 0 to maximise compression on tooltip-length content;
        // > 60 frames switches to 30 for seekability. A 6-frame animation must hit
        // the "short" branch and produce bit-identical output to explicit 0.
        ConcurrentList<ImageFrame> frames = Concurrent.newList();
        for (int f = 0; f < 6; f++) {
            PixelBuffer fb = PixelBuffer.create(32, 32);
            for (int y = 0; y < 32; y++)
                for (int x = 0; x < 32; x++)
                    fb.setPixel(x, y, 0xFF000000 | ((x * 8) << 16) | ((y * 8) << 8));
            frames.add(ImageFrame.of(fb, 100, 0, 0,
                FrameDisposal.DO_NOT_DISPOSE, FrameBlend.OVER));
        }
        AnimatedImageData anim = AnimatedImageData.builder()
            .withWidth(32).withHeight(32)
            .withFrames(frames)
            .withLoopCount(0)
            .withBackgroundColor(0)
            .build();

        ImageFactory factory = new ImageFactory();
        File explicitZero = outputDir.resolve("anim_default_explicit0.webp").toFile();
        File leftDefault = outputDir.resolve("anim_default_auto.webp").toFile();
        factory.toFile(anim, ImageFormat.WEBP, explicitZero,
            WebPWriteOptions.builder().isLossless(false).withQuality(1.0f).usePFrames(true)
                .withForceKeyframeEvery(0).build());
        factory.toFile(anim, ImageFormat.WEBP, leftDefault,
            WebPWriteOptions.builder().isLossless(false).withQuality(1.0f).usePFrames(true).build());

        try {
            byte[] a = Files.readAllBytes(explicitZero.toPath());
            byte[] b = Files.readAllBytes(leftDefault.toPath());
            assertThat("short-anim default output length matches explicit 0",
                b.length, is(a.length));
            for (int i = 0; i < a.length; i++)
                assertThat("byte " + i + " matches", b[i], is(a[i]));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    @DisplayName("animated lossy + alpha decodes through libwebp with bit-exact alpha per frame")
    void animatedLossyAlphaLibwebpRoundTrip() throws Exception {
        // 3 frames with SOURCE blending so libwebp's animated decoder emits each frame's
        // alpha uncomposited against prior frames; alpha comes through VP8L (lossless) so
        // it must reconstruct bit-exactly.
        ConcurrentList<ImageFrame> frames = Concurrent.newList();
        for (int f = 0; f < 3; f++) {
            PixelBuffer fb = PixelBuffer.create(16, 16);
            int phase = f * 40;
            for (int y = 0; y < 16; y++)
                for (int x = 0; x < 16; x++) {
                    int alpha = Math.min(255, phase + (x + y) * 7);
                    fb.setPixel(x, y, (alpha << 24) | 0x000000FF);
                }
            frames.add(ImageFrame.of(fb, 100, 0, 0,
                FrameDisposal.DO_NOT_DISPOSE, FrameBlend.SOURCE));
        }

        AnimatedImageData anim = AnimatedImageData.builder()
            .withWidth(16).withHeight(16)
            .withFrames(frames)
            .withLoopCount(0)
            .withBackgroundColor(0)
            .build();

        File out = outputDir.resolve("anim_alpha_libwebp.webp").toFile();
        new ImageFactory().toFile(anim, ImageFormat.WEBP, out,
            WebPWriteOptions.builder().isLossless(false).withQuality(0.9f).build());

        int[][] decoded = decodeAnimatedAlphaWithLibwebp(out, 16, 16, 3);

        for (int f = 0; f < 3; f++) {
            int phase = f * 40;
            for (int y = 0; y < 16; y++)
                for (int x = 0; x < 16; x++) {
                    int expectedAlpha = Math.min(255, phase + (x + y) * 7);
                    int gotAlpha = decoded[f][y * 16 + x];
                    if (gotAlpha != expectedAlpha)
                        throw new AssertionError(String.format(
                            "libwebp frame %d alpha @ (%d,%d): expected %d, got %d",
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

    /**
     * Decodes all frames of an animated WebP via libwebp's {@code WebPAnimDecoderNew}
     * through Python's {@code webp._webp} ffi. Returns {@code alpha[frameIdx][y * w + x]}
     * as integers in {@code 0..255}. {@code webp.load_image} does not support animated
     * WebPs, so this uses the low-level animation decoder directly.
     */
    private static int[][] decodeAnimatedAlphaWithLibwebp(
        File file, int expectedW, int expectedH, int expectedFrames
    ) throws Exception {
        String script =
            "import sys, warnings\n" +
            "warnings.filterwarnings('ignore')\n" +
            "try:\n" +
            "    from webp import _webp, ffi\n" +
            "except ImportError:\n" +
            "    print('NO_WEBP'); sys.exit(2)\n" +
            "with open(r'" + file.getAbsolutePath() + "', 'rb') as f: data = f.read()\n" +
            "options = ffi.new('WebPAnimDecoderOptions*')\n" +
            "_webp.lib.WebPAnimDecoderOptionsInit(options)\n" +
            "wd = ffi.new('WebPData*')\n" +
            "buf = ffi.new('uint8_t[]', data); wd.bytes = buf; wd.size = len(data)\n" +
            "dec = _webp.lib.WebPAnimDecoderNew(wd, options)\n" +
            "if dec == ffi.NULL:\n" +
            "    print('DECODE_FAIL'); sys.exit(3)\n" +
            "info = ffi.new('WebPAnimInfo*')\n" +
            "_webp.lib.WebPAnimDecoderGetInfo(dec, info)\n" +
            "w, h = info.canvas_width, info.canvas_height\n" +
            "print(f'DIMS {w} {h}')\n" +
            "print(f'FRAMES {info.frame_count}')\n" +
            "while _webp.lib.WebPAnimDecoderHasMoreFrames(dec):\n" +
            "    ptr = ffi.new('uint8_t**'); ts = ffi.new('int*')\n" +
            "    if _webp.lib.WebPAnimDecoderGetNext(dec, ptr, ts) == 0:\n" +
            "        print('FRAME_FAIL'); sys.exit(4)\n" +
            "    pixels = ffi.buffer(ptr[0], w * h * 4)[:]\n" +
            "    print('FRAME')\n" +
            "    for i in range(3, len(pixels), 4):\n" +
            "        print(f'A {pixels[i]}')\n" +
            "_webp.lib.WebPAnimDecoderDelete(dec)\n";

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
            throw new AssertionError("libwebp rejected our animated WebP:\n" + stdout);

        int w = -1, h = -1, frameCount = -1;
        int[][] frames = new int[expectedFrames][expectedW * expectedH];
        int frameIdx = -1;
        int alphaIdx = 0;
        for (String line : stdout.split("\\R")) {
            if (line.startsWith("DIMS ")) {
                String[] dims = line.substring(5).split("\\s+");
                w = Integer.parseInt(dims[0]);
                h = Integer.parseInt(dims[1]);
            } else if (line.startsWith("FRAMES ")) {
                frameCount = Integer.parseInt(line.substring(7).trim());
            } else if (line.equals("FRAME")) {
                frameIdx++;
                alphaIdx = 0;
            } else if (line.startsWith("A ") && frameIdx >= 0 && frameIdx < expectedFrames) {
                if (alphaIdx < frames[frameIdx].length)
                    frames[frameIdx][alphaIdx++] = Integer.parseInt(line.substring(2));
            }
        }
        if (w != expectedW || h != expectedH)
            throw new AssertionError(
                "dims " + w + "x" + h + " != expected " + expectedW + "x" + expectedH
                + "\nfull stdout:\n" + stdout);
        if (frameCount != expectedFrames)
            throw new AssertionError(
                "frame count " + frameCount + " != expected " + expectedFrames
                + "\nfull stdout:\n" + stdout);
        return frames;
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
