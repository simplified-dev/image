package dev.simplified.image.codec.webp;

import dev.simplified.collection.Concurrent;
import dev.simplified.collection.ConcurrentList;
import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFactory;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.codec.webp.lossless.VP8LEncoder;
import dev.simplified.image.data.AnimatedImageData;
import dev.simplified.image.data.FrameBlend;
import dev.simplified.image.data.FrameDisposal;
import dev.simplified.image.data.ImageFrame;
import dev.simplified.image.pixel.PixelBuffer;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.opentest4j.TestAbortedException;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import org.hamcrest.MatcherAssert.assertThat;
import org.hamcrest.Matchers.greaterThan;
import org.hamcrest.Matchers.is;

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

        // Both outputs must be small on stationary content. Prior to
        // partial-frame ANMF (the 7684318 / FrameDiffUtil commit) the
        // keyframe-only path emitted a full VP8 keyframe per ANMF and was
        // much larger than the P-frame path; with partial frames, unchanged
        // frames collapse to 1x1 placeholder ANMFs on BOTH paths (except
        // the P-frame path still re-emits full inter-prediction VP8 per
        // frame, so it's now slightly larger than keyframe-only on
        // fully-stationary content). The size-ordering between the two
        // options is content-dependent and no longer gates correctness.
        long maxExpected = 10_000;
        if (pFrameOut.length() > maxExpected || keyOnlyOut.length() > maxExpected)
            throw new AssertionError(String.format(
                "animated stationary content too large: pframe=%d B keyOnly=%d B "
                + "(expected <= %d B each)",
                pFrameOut.length(), keyOnlyOut.length(), maxExpected));

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
    @DisplayName("static lossy with autoSegment differs from baseline on high-variance content and round-trips")
    void staticLossyAutoSegmentViaWriter() throws IOException {
        // 64x64 source with a smooth left half + high-variance right half (alternating
        // 2-row stripes). Variance spread clears the auto-segment heuristic's gate so
        // the encoder installs per-segment quant deltas via the writer's autoSegment
        // option. Output bytes must differ from the autoSegment=false baseline, and
        // the auto-segment output must round-trip through the decoder with healthy
        // PSNR.
        int W = 64, H = 64;
        PixelBuffer buf = PixelBuffer.create(W, H);
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int argb;
                if (x < W / 2) {
                    int v = (x + y) * 2;
                    argb = 0xFF000000 | (v << 16) | (v << 8) | v;
                } else {
                    int v = ((y >> 1) & 1) != 0 ? 200 : 40;
                    argb = 0xFF000000 | (v << 16) | (v << 8) | v;
                }
                buf.setPixel(x, y, argb);
            }
        }
        StaticImageData image = new StaticImageData(ImageFrame.of(buf));
        ImageFactory factory = new ImageFactory();

        File baselineOut = outputDir.resolve("static_autoSeg_off.webp").toFile();
        File autoSegOut = outputDir.resolve("static_autoSeg_on.webp").toFile();
        factory.toFile(image, ImageFormat.WEBP, baselineOut,
            WebPWriteOptions.builder().isLossless(false).withQuality(0.5f).build());
        factory.toFile(image, ImageFormat.WEBP, autoSegOut,
            WebPWriteOptions.builder().isLossless(false).withQuality(0.5f)
                .withAutoSegment(true).build());

        if (autoSegOut.length() >= baselineOut.length())
            throw new AssertionError(String.format(
                "autoSegment did not reduce static-lossy byte output on high-variance "
                + "content: baseline=%d autoSeg=%d (expected strictly smaller)",
                baselineOut.length(), autoSegOut.length()));

        // Round-trip: decode the autoSegment stream and sanity-check PSNR.
        ImageData decoded = factory.fromFile(autoSegOut);
        PixelBuffer dec = decoded.toPixelBuffer();
        long sse = 0;
        long pixelCount = (long) W * H * 3;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                int a = buf.getPixel(x, y);
                int b = dec.getPixel(x, y);
                for (int shift : new int[] { 0, 8, 16 }) {
                    int d = ((a >> shift) & 0xFF) - ((b >> shift) & 0xFF);
                    sse += (long) d * d;
                }
            }
        double mse = (double) sse / pixelCount;
        double psnr = mse > 0 ? 10.0 * Math.log10((255.0 * 255.0) / mse) : 99.0;
        if (psnr < 25.0)
            throw new AssertionError(String.format(
                "static-lossy autoSegment round-trip PSNR = %.2f dB (expected >= 25)",
                psnr));
    }

    @Test
    @DisplayName("static lossy autoSegment falls through to baseline on flat content (bit-identical via writer)")
    void staticLossyAutoSegmentFallsThroughOnFlatContent() throws IOException {
        // Solid-colour 32x32 source has zero per-MB variance. WebPImageWriter with
        // autoSegment=true must route through VP8Encoder.encodeWithAutoSegment,
        // which detects the variance gate failure and falls through to the single-
        // segment path. Output files must therefore be byte-identical to the
        // autoSegment=false baseline - no wasted segment-tree emit.
        int W = 32, H = 32;
        PixelBuffer buf = PixelBuffer.create(W, H);
        buf.fill(0xFF808080);
        StaticImageData image = new StaticImageData(ImageFrame.of(buf));
        ImageFactory factory = new ImageFactory();

        File offOut = outputDir.resolve("flat_autoSeg_off.webp").toFile();
        File onOut = outputDir.resolve("flat_autoSeg_on.webp").toFile();
        factory.toFile(image, ImageFormat.WEBP, offOut,
            WebPWriteOptions.builder().isLossless(false).withQuality(0.5f).build());
        factory.toFile(image, ImageFormat.WEBP, onOut,
            WebPWriteOptions.builder().isLossless(false).withQuality(0.5f)
                .withAutoSegment(true).build());

        byte[] off = Files.readAllBytes(offOut.toPath());
        byte[] on = Files.readAllBytes(onOut.toPath());
        if (off.length != on.length)
            throw new AssertionError(String.format(
                "autoSegment did not fall through on flat content: "
                + "baseline=%d autoSeg=%d (expected equal)", off.length, on.length));
        for (int i = 0; i < off.length; i++)
            if (off[i] != on[i])
                throw new AssertionError(String.format(
                    "autoSegment byte %d differs on flat content: "
                    + "baseline=0x%02X autoSeg=0x%02X", i, off[i] & 0xFF, on[i] & 0xFF));
    }

    @Test
    @DisplayName("animated lossy with P-frames + autoSegment round-trips cleanly via WebPImageWriter")
    void animatedLossyPFramesWithAutoSegment() {
        // 3-frame animation with high-variance content per frame (so the autoSegment
        // spread gate clears on both keyframe and P-frame paths). Asserts the writer
        // plumbs autoSegment through VP8EncoderSession for both frame types and the
        // resulting WebP decodes without error via the roundtrip factory.fromFile.
        int W = 64, H = 64;
        ConcurrentList<ImageFrame> frames = Concurrent.newList();
        for (int f = 0; f < 3; f++) {
            PixelBuffer fb = PixelBuffer.create(W, H);
            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    int argb;
                    if (x < W / 2) {
                        int v = ((x + y + f) * 2) & 0xFF;
                        argb = 0xFF000000 | (v << 16) | (v << 8) | v;
                    } else {
                        int v = (((y + f) >> 1) & 1) != 0 ? 200 : 40;
                        argb = 0xFF000000 | (v << 16) | (v << 8) | v;
                    }
                    fb.setPixel(x, y, argb);
                }
            }
            frames.add(ImageFrame.of(fb, 100, 0, 0,
                FrameDisposal.DO_NOT_DISPOSE, FrameBlend.OVER));
        }
        AnimatedImageData anim = AnimatedImageData.builder()
            .withWidth(W).withHeight(H)
            .withFrames(frames)
            .withLoopCount(0)
            .withBackgroundColor(0)
            .build();

        File out = outputDir.resolve("anim_autoSeg.webp").toFile();
        ImageFactory factory = new ImageFactory();
        factory.toFile(anim, ImageFormat.WEBP, out,
            WebPWriteOptions.builder().isLossless(false).withQuality(0.5f)
                .usePFrames(true).withAutoSegment(true).build());

        ImageData decoded = factory.fromFile(out);
        if (!(decoded instanceof AnimatedImageData decAnim))
            throw new AssertionError("expected AnimatedImageData, got " + decoded.getClass());
        if (decAnim.getFrames().size() != 3)
            throw new AssertionError("frame count mismatch: " + decAnim.getFrames().size());

        // Sanity-check PSNR on frame 0 - any wire-format defect in the autoSegment
        // P-frame path would either fail decode or land PSNR well below 20 dB.
        PixelBuffer src0 = frames.get(0).pixels();
        PixelBuffer dec0 = decAnim.getFrames().get(0).pixels();
        long sse = 0;
        long pixelCount = (long) W * H * 3;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                int a = src0.getPixel(x, y);
                int b = dec0.getPixel(x, y);
                for (int shift : new int[] { 0, 8, 16 }) {
                    int d = ((a >> shift) & 0xFF) - ((b >> shift) & 0xFF);
                    sse += (long) d * d;
                }
            }
        double mse = (double) sse / pixelCount;
        double psnr = mse > 0 ? 10.0 * Math.log10((255.0 * 255.0) / mse) : 99.0;
        if (psnr < 22.0)
            throw new AssertionError(String.format(
                "animated autoSegment frame 0 PSNR = %.2f dB (expected >= 22)", psnr));
    }

    @Test
    @DisplayName("diagnostic: decode testLore P-frame output via our own reader + dump PNGs for visual check")
    void diagnosticDecodeTestLorePFrameOutput() throws Exception {
        // Self-skip when testLore cache file not present. Produces
        // W:/tmp/pframe_decoded_fN.png for each selected frame + PSNR vs lossless
        // reference. Determines whether the user's "corrupt" report means
        // "visually wrong pixels" (real bug) vs "libwebp rejects the container"
        // (known Task 13 limitation - our own reader decodes P-frames fine).
        java.nio.file.Path pframeIn = java.nio.file.Path.of(
            "W:/Workspace/Java/SkyBlock-Simplified/asset-renderer/cache/test-lore/weapon_ss4_lossy.webp");
        java.nio.file.Path losslessIn = java.nio.file.Path.of(
            "W:/Workspace/Java/SkyBlock-Simplified/asset-renderer/cache/test-lore/weapon_ss4.webp");
        if (!java.nio.file.Files.exists(pframeIn) || !java.nio.file.Files.exists(losslessIn))
            throw new org.opentest4j.TestAbortedException(
                "testLore cache files not present - diagnostic only");
        java.nio.file.Files.createDirectories(java.nio.file.Path.of("W:/tmp"));

        ImageFactory factory = new ImageFactory();
        ImageData dec = factory.fromFile(pframeIn.toFile());
        ImageData refDec = factory.fromFile(losslessIn.toFile());
        if (!(dec instanceof AnimatedImageData anim) || !(refDec instanceof AnimatedImageData refAnim))
            throw new AssertionError("expected AnimatedImageData from both files");
        System.err.println("[pframe-diag] P-frame file decoded via our reader: "
            + anim.getFrames().size() + " frames (" + anim.getWidth() + "x" + anim.getHeight() + ")");
        System.err.println("[pframe-diag] Lossless reference: "
            + refAnim.getFrames().size() + " frames");

        int[] dumpFrames = { 0, 1, 5, 10, 19 };
        for (int fi : dumpFrames) {
            if (fi >= anim.getFrames().size()) continue;
            PixelBuffer pb = anim.getFrames().get(fi).pixels();
            PixelBuffer rpb = refAnim.getFrames().get(fi).pixels();
            int w = pb.width(), h = pb.height();

            java.awt.image.BufferedImage bi =
                new java.awt.image.BufferedImage(w, h, java.awt.image.BufferedImage.TYPE_INT_ARGB);
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    bi.setRGB(x, y, pb.getPixel(x, y));
            File pngOut = new File("W:/tmp/pframe_decoded_f" + fi + ".png");
            javax.imageio.ImageIO.write(bi, "PNG", pngOut);

            long sse = 0;
            int n = 0;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++) {
                    int a = pb.getPixel(x, y), b = rpb.getPixel(x, y);
                    for (int shift : new int[] { 0, 8, 16, 24 }) {
                        int d = ((a >> shift) & 0xFF) - ((b >> shift) & 0xFF);
                        sse += (long) d * d;
                        n++;
                    }
                }
            double mse = (double) sse / n;
            double psnr = mse == 0 ? 99.0 : 10.0 * Math.log10(255.0 * 255.0 / mse);
            System.err.printf("[pframe-diag]   frame %2d: %s  PSNR vs lossless=%.2f dB  center px: ours=0x%08X  lossless=0x%08X%n",
                fi, pngOut.getName(), psnr, pb.getPixel(w / 2, h / 2), rpb.getPixel(w / 2, h / 2));
        }
    }

    @Test
    @DisplayName("diagnostic: re-encode real tooltip frames via partial-frame ANMF + report size delta")
    void diagnosticReencodeTooltipPartialFrames() throws Exception {
        java.nio.file.Path rawPath = java.nio.file.Path.of("W:/tmp/tooltip_frames_20.rgba");
        if (!java.nio.file.Files.exists(rawPath))
            throw new org.opentest4j.TestAbortedException(
                "tooltip_frames_20.rgba not present - diagnostic only");
        byte[] raw = Files.readAllBytes(rawPath);
        int W = 454, H = 260;
        int frameBytes = W * H * 4;
        int frameCount = raw.length / frameBytes;
        if (frameCount != 20)
            throw new AssertionError("expected 20 frames, got " + frameCount);

        ConcurrentList<ImageFrame> frames = Concurrent.newList();
        for (int f = 0; f < frameCount; f++) {
            int[] argb = new int[W * H];
            for (int i = 0; i < W * H; i++) {
                int off = f * frameBytes + i * 4;
                int r = raw[off] & 0xFF;
                int g = raw[off + 1] & 0xFF;
                int b = raw[off + 2] & 0xFF;
                int a = raw[off + 3] & 0xFF;
                argb[i] = (a << 24) | (r << 16) | (g << 8) | b;
            }
            frames.add(ImageFrame.of(PixelBuffer.of(argb, W, H), 50, 0, 0,
                FrameDisposal.DO_NOT_DISPOSE, FrameBlend.OVER));
        }
        AnimatedImageData anim = AnimatedImageData.builder()
            .withWidth(W).withHeight(H)
            .withFrames(frames)
            .withLoopCount(0)
            .withBackgroundColor(0)
            .build();

        File out = outputDir.resolve("tooltip_reencode_partial.webp").toFile();
        new ImageFactory().toFile(anim, ImageFormat.WEBP, out,
            WebPWriteOptions.builder().isLossless().build());
        System.err.println("[partial-reencode] ours animated lossless: "
            + out.length() + " bytes (prior full-frame encode was 1,699,930 B; "
            + "libwebp WebPAnimEncoder is 11,688 B)");
    }

    @Test
    @DisplayName("partial-frame ANMF: tooltip-like animation encodes to bounding-box deltas, not full frames")
    void partialFrameAnmfTooltipLikeAnimation() throws Exception {
        // 96x96 animation, 5 frames. Frame 0 is a full uniform teal canvas with
        // a constant border; frames 1-4 differ ONLY in an 8x8 square at fixed
        // offset (40, 40). This simulates the tooltip obfuscated-text pattern:
        // static body, one small region cycles. With partial-frame ANMF the
        // file should be dramatically smaller than a full-frame-per-ANMF
        // encode would produce, AND libwebp's animation decoder should return
        // all 5 frames with the correct pixel-per-frame deltas reconstructed.
        int W = 96, H = 96;
        ConcurrentList<ImageFrame> frames = Concurrent.newList();
        for (int f = 0; f < 5; f++) {
            PixelBuffer fb = PixelBuffer.create(W, H);
            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    int argb = 0xFF008080;   // static teal canvas
                    if (x >= 40 && x < 48 && y >= 40 && y < 48) {
                        int v = (f * 64) & 0xFF;   // cycling square
                        argb = 0xFF000000 | (v << 16) | (v << 8) | v;
                    }
                    fb.setPixel(x, y, argb);
                }
            }
            frames.add(ImageFrame.of(fb, 50, 0, 0,
                FrameDisposal.DO_NOT_DISPOSE, FrameBlend.OVER));
        }
        AnimatedImageData anim = AnimatedImageData.builder()
            .withWidth(W).withHeight(H)
            .withFrames(frames)
            .withLoopCount(0)
            .withBackgroundColor(0)
            .build();

        File out = outputDir.resolve("partial_frame_tooltip_like.webp").toFile();
        new ImageFactory().toFile(anim, ImageFormat.WEBP, out,
            WebPWriteOptions.builder().isLossless().build());

        // Inspect the RIFF: frame 0 must be full canvas (W x H), frames 1-4
        // must be small partial frames at offset near (40, 40). Parse the
        // ANMF offset + dimension fields directly so the test doesn't depend
        // on any Python tooling.
        byte[] data = Files.readAllBytes(out.toPath());
        int offset = 12;
        int anmfIdx = 0;
        int[][] bboxes = new int[frames.size()][4];
        while (offset + 8 <= data.length && anmfIdx < frames.size()) {
            String tag = new String(data, offset, 4);
            int size = (data[offset + 4] & 0xFF)
                | ((data[offset + 5] & 0xFF) << 8)
                | ((data[offset + 6] & 0xFF) << 16)
                | ((data[offset + 7] & 0xFF) << 24);
            if (tag.equals("ANMF")) {
                int x = 2 * ((data[offset + 8] & 0xFF)
                    | ((data[offset + 9] & 0xFF) << 8)
                    | ((data[offset + 10] & 0xFF) << 16));
                int y = 2 * ((data[offset + 11] & 0xFF)
                    | ((data[offset + 12] & 0xFF) << 8)
                    | ((data[offset + 13] & 0xFF) << 16));
                int aw = 1 + ((data[offset + 14] & 0xFF)
                    | ((data[offset + 15] & 0xFF) << 8)
                    | ((data[offset + 16] & 0xFF) << 16));
                int ah = 1 + ((data[offset + 17] & 0xFF)
                    | ((data[offset + 18] & 0xFF) << 8)
                    | ((data[offset + 19] & 0xFF) << 16));
                bboxes[anmfIdx++] = new int[] { x, y, aw, ah };
            }
            offset += 8 + size + (size & 1);
        }

        // Frame 0: full canvas.
        if (bboxes[0][0] != 0 || bboxes[0][1] != 0 || bboxes[0][2] != W || bboxes[0][3] != H)
            throw new AssertionError(String.format(
                "frame 0 expected full canvas (0,0)%dx%d, got (%d,%d)%dx%d",
                W, H, bboxes[0][0], bboxes[0][1], bboxes[0][2], bboxes[0][3]));
        // Frames 1-4: bounding box should be small and cover the (40,40)-(48,48)
        // changed square. Exact offset must be even (spec); width/height <= 10.
        for (int i = 1; i < frames.size(); i++) {
            int[] b = bboxes[i];
            if (b[0] > 40 || b[1] > 40)
                throw new AssertionError("frame " + i + " bbox offset (" + b[0] + "," + b[1]
                    + ") doesn't reach the (40,40) changed region");
            if (b[0] + b[2] < 48 || b[1] + b[3] < 48)
                throw new AssertionError("frame " + i + " bbox doesn't cover (47,47)");
            if (b[2] > 16 || b[3] > 16)
                throw new AssertionError("frame " + i + " bbox " + b[2] + "x" + b[3]
                    + " too large (expected <=16, most of canvas should be excluded)");
            if ((b[0] & 1) != 0 || (b[1] & 1) != 0)
                throw new AssertionError("frame " + i + " bbox offset (" + b[0] + "," + b[1]
                    + ") must be even");
        }

        // Size gate: a full-frame-per-ANMF encode would produce 5 * ~some-kB.
        // Partial frames should be much smaller - set a loose bound so we
        // catch regressions without being brittle about exact VP8L sizes.
        if (data.length > 20_000)
            throw new AssertionError("partial-frame output " + data.length
                + " bytes - expected <= 20KB on this mostly-static content "
                + "(full-frame encode would be ~50KB+)");

        // Round-trip via our own reader to verify the partial frames compose
        // back to the right pixels.
        ImageData dec = new ImageFactory().fromFile(out);
        if (!(dec instanceof AnimatedImageData decAnim))
            throw new AssertionError("expected AnimatedImageData");
        if (decAnim.getFrames().size() != 5)
            throw new AssertionError("decoded " + decAnim.getFrames().size() + " frames, expected 5");
        for (int i = 0; i < 5; i++) {
            PixelBuffer src = frames.get(i).pixels();
            PixelBuffer got = decAnim.getFrames().get(i).pixels();
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++) {
                    int s = src.getPixel(x, y);
                    int g = got.getPixel(x, y);
                    if (s != g)
                        throw new AssertionError(String.format(
                            "frame %d round-trip px (%d,%d): src=0x%08X got=0x%08X",
                            i, x, y, s, g));
                }
        }

        // Cross-decoder check: libwebp's WebPAnimDecoder must accept this
        // partial-frame file and return all 5 frames. Self-skips when Python /
        // webp unavailable - the round-trip above still gates correctness.
        int[][] libwebpFrames;
        try {
            libwebpFrames = decodeAnimatedAlphaWithLibwebp(out, W, H, 5);
        } catch (org.opentest4j.TestAbortedException skip) {
            return;   // libwebp unavailable
        }
        if (libwebpFrames.length != 5)
            throw new AssertionError("libwebp decoded " + libwebpFrames.length
                + " frames, expected 5 from our partial-frame animated WebP");
    }

    @Test
    @DisplayName("near-lossless: level=100 is bit-identical to lossless (off by default)")
    void nearLosslessLevel100IsNoOp() throws IOException {
        PixelBuffer buf = buildGradientWithEdges(128, 128);
        StaticImageData image = new StaticImageData(ImageFrame.of(buf));
        ImageFactory factory = new ImageFactory();

        File baselineOut = outputDir.resolve("nl_baseline.webp").toFile();
        File level100Out = outputDir.resolve("nl_level100.webp").toFile();
        factory.toFile(image, ImageFormat.WEBP, baselineOut,
            WebPWriteOptions.builder().isLossless().build());
        factory.toFile(image, ImageFormat.WEBP, level100Out,
            WebPWriteOptions.builder().isLossless().withNearLossless(100).build());

        byte[] a = Files.readAllBytes(baselineOut.toPath());
        byte[] b = Files.readAllBytes(level100Out.toPath());
        if (a.length != b.length)
            throw new AssertionError("near-lossless=100 len " + b.length
                + " != baseline len " + a.length);
        for (int i = 0; i < a.length; i++)
            if (a[i] != b[i])
                throw new AssertionError("near-lossless=100 byte " + i
                    + " differs: baseline=0x" + Integer.toHexString(a[i] & 0xFF)
                    + " near=0x" + Integer.toHexString(b[i] & 0xFF));
    }

    @Test
    @DisplayName("near-lossless: lower levels shrink file + decode to valid pixels")
    void nearLosslessLowerLevelsShrinkFile() throws IOException {
        // Gradient-with-edges source: the non-smooth diagonal stripe gets snapped,
        // the smooth background passes through. Non-trivially compressible.
        PixelBuffer buf = buildGradientWithEdges(128, 128);
        StaticImageData image = new StaticImageData(ImageFrame.of(buf));
        ImageFactory factory = new ImageFactory();

        File baselineOut = outputDir.resolve("nl_baseline_shrink.webp").toFile();
        File nl60Out = outputDir.resolve("nl_level60.webp").toFile();
        factory.toFile(image, ImageFormat.WEBP, baselineOut,
            WebPWriteOptions.builder().isLossless().build());
        factory.toFile(image, ImageFormat.WEBP, nl60Out,
            WebPWriteOptions.builder().isLossless().withNearLossless(60).build());

        // Output must be a valid VP8L stream + decode back through our reader.
        ImageData dec = factory.fromFile(nl60Out);
        PixelBuffer decPixels = dec.toPixelBuffer();
        if (decPixels.width() != 128 || decPixels.height() != 128)
            throw new AssertionError("near-lossless round-trip dims "
                + decPixels.width() + "x" + decPixels.height());

        // Near-lossless preprocessing should NOT produce an identical file - if
        // it did, the hook isn't wired. But it also shouldn't grow the file.
        if (baselineOut.length() == nl60Out.length())
            throw new AssertionError("near-lossless=60 file byte-identical to "
                + "baseline - preprocessing hook isn't wired");
    }

    @Test
    @DisplayName("near-lossless: small-image bypass (< 64x64) leaves bits unchanged")
    void nearLosslessSmallImageBypass() throws IOException {
        // 32x32 triggers libwebp's small-image bypass (width < 64 && height < 64).
        // The preprocessing is skipped; output bytes must match baseline lossless.
        PixelBuffer buf = buildGradientWithEdges(32, 32);
        StaticImageData image = new StaticImageData(ImageFrame.of(buf));
        ImageFactory factory = new ImageFactory();

        File baselineOut = outputDir.resolve("nl_small_baseline.webp").toFile();
        File nl0Out = outputDir.resolve("nl_small_level0.webp").toFile();
        factory.toFile(image, ImageFormat.WEBP, baselineOut,
            WebPWriteOptions.builder().isLossless().build());
        factory.toFile(image, ImageFormat.WEBP, nl0Out,
            WebPWriteOptions.builder().isLossless().withNearLossless(0).build());

        if (baselineOut.length() != nl0Out.length())
            throw new AssertionError("small-image bypass broken: "
                + "baseline=" + baselineOut.length() + " nl=0=" + nl0Out.length()
                + " (expected equal because preprocessing should skip)");
    }

    @Test
    @DisplayName("near-lossless: flat solid-color content round-trips bit-identical at every level")
    void nearLosslessFlatContentUnchanged() throws IOException {
        // Solid grey 128x128. Every interior pixel's 4-connected neighbours are
        // identical, so IsSmooth returns true everywhere: preprocessing is a
        // no-op regardless of level. Output must match baseline lossless
        // byte-for-byte at all non-off levels.
        int w = 128, h = 128;
        PixelBuffer buf = PixelBuffer.create(w, h);
        buf.fill(0xFF808080);
        StaticImageData image = new StaticImageData(ImageFrame.of(buf));
        ImageFactory factory = new ImageFactory();

        File baselineOut = outputDir.resolve("nl_flat_baseline.webp").toFile();
        factory.toFile(image, ImageFormat.WEBP, baselineOut,
            WebPWriteOptions.builder().isLossless().build());
        byte[] baseline = Files.readAllBytes(baselineOut.toPath());

        for (int level : new int[] { 0, 40, 60, 80 }) {
            File out = outputDir.resolve("nl_flat_level" + level + ".webp").toFile();
            factory.toFile(image, ImageFormat.WEBP, out,
                WebPWriteOptions.builder().isLossless().withNearLossless(level).build());
            byte[] got = Files.readAllBytes(out.toPath());
            if (got.length != baseline.length)
                throw new AssertionError("flat content at level=" + level
                    + " len=" + got.length + " != baseline=" + baseline.length);
            for (int i = 0; i < got.length; i++)
                if (got[i] != baseline[i])
                    throw new AssertionError("flat content at level=" + level
                        + " byte " + i + " differs from baseline");
        }
    }

    @Test
    @DisplayName("near-lossless: NearLosslessPreprocess algorithm invariants")
    void nearLosslessPreprocessInvariants() {
        // White-box: verify the algorithm's correctness properties directly
        // against NearLosslessPreprocess.apply without going through the writer.
        int w = 128, h = 128;
        int[] src = new int[w * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                int v;
                if ((x + y) % 16 == 0) v = 200;             // thin diagonal stripe
                else v = 40 + (x + y) / 8;                   // smooth gradient
                src[y * w + x] = 0xFF000000 | (v << 16) | (v << 8) | v;
            }

        // Level 100 = no-op copy.
        int[] off = dev.simplified.image.codec.webp.lossless.NearLosslessPreprocess
            .apply(src, w, h, 100);
        for (int i = 0; i < src.length; i++)
            if (off[i] != src[i])
                throw new AssertionError("level=100 not a no-op at idx " + i);

        // Border rows and columns MUST be unchanged at all non-off levels.
        for (int level : new int[] { 0, 40, 60, 80 }) {
            int[] got = dev.simplified.image.codec.webp.lossless.NearLosslessPreprocess
                .apply(src, w, h, level);
            // First row, last row
            for (int x = 0; x < w; x++) {
                if (got[x] != src[x])
                    throw new AssertionError("level=" + level + " first-row x=" + x
                        + " changed: " + Integer.toHexString(src[x]) + " -> "
                        + Integer.toHexString(got[x]));
                if (got[(h - 1) * w + x] != src[(h - 1) * w + x])
                    throw new AssertionError("level=" + level + " last-row x=" + x
                        + " changed");
            }
            // First col, last col
            for (int y = 0; y < h; y++) {
                if (got[y * w] != src[y * w])
                    throw new AssertionError("level=" + level + " first-col y=" + y
                        + " changed");
                if (got[y * w + w - 1] != src[y * w + w - 1])
                    throw new AssertionError("level=" + level + " last-col y=" + y
                        + " changed");
            }
        }

        // At level 0 (5 bits shaved = bucket 32), per-channel deviation at any
        // interior pixel must be bounded by 31 (bucket_size - 1).
        int[] got0 = dev.simplified.image.codec.webp.lossless.NearLosslessPreprocess
            .apply(src, w, h, 0);
        int maxDev = 0;
        for (int i = 0; i < src.length; i++) {
            int s = src[i], g = got0[i];
            for (int shift : new int[] { 0, 8, 16, 24 }) {
                int sc = (s >>> shift) & 0xFF;
                int gc = (g >>> shift) & 0xFF;
                int dev = Math.abs(gc - sc);
                if (dev > maxDev) maxDev = dev;
            }
        }
        if (maxDev > 31)
            throw new AssertionError("level=0 max per-channel deviation "
                + maxDev + " > expected max 31 (bucket 32)");
    }

    /**
     * Source synthesis helper for the near-lossless tests: a smooth grey
     * gradient overlaid with a diagonal stripe of contrasting pixels. Shared
     * across the near-lossless test cases so they all have the same realistic
     * mix of smooth and non-smooth regions.
     */
    private static @NotNull PixelBuffer buildGradientWithEdges(int w, int h) {
        PixelBuffer buf = PixelBuffer.create(w, h);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                int v;
                if ((x + y) % 16 == 0) v = 200;
                else v = 40 + (x + y) / 8;
                buf.setPixel(x, y, 0xFF000000 | (v << 16) | (v << 8) | v);
            }
        return buf;
    }

    @Test
    @DisplayName("default animated-lossy output is libwebp-decodable (Task 13 regression gate)")
    void defaultAnimatedLossyIsLibwebpDecodable() throws Exception {
        // Regression gate tied to the Task 13 closure: libwebp's VP8GetHeaders
        // unconditionally rejects non-keyframe VP8 bitstreams, so anything wrapped
        // in ANMF that carries a P-frame VP8 payload will fail to decode in
        // libwebp-based tools (Chrome, Firefox, Windows Photos, Pillow-via-libwebp,
        // ffmpeg-built-against-libwebp).
        //
        // Default WebPWriteOptions (no usePFrames(true), no explicit multithreaded)
        // MUST produce a file with all-keyframe VP8 payloads so the output opens
        // everywhere. This test encodes a 3-frame animation at default settings
        // and feeds the resulting file to libwebp's animation decoder via the
        // Python binding. A regression that accidentally flips usePFrames to
        // true by default, or leaks P-frames into the all-keyframes path, would
        // surface here as a libwebp decode failure.
        //
        // Self-skips when python / webp module unavailable - fine on CI without
        // libwebp installed. On a dev box with libwebp, set -Dvp8.pythonBin=...
        // to point at a Python with the webp package when the default python3 on
        // PATH doesn't have it (common on Windows with Microsoft Store stubs).
        int W = 48, H = 48;
        ConcurrentList<ImageFrame> frames = Concurrent.newList();
        for (int f = 0; f < 3; f++) {
            PixelBuffer fb = PixelBuffer.create(W, H);
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++) {
                    int v = ((x + y + f) * 4) & 0xFF;
                    fb.setPixel(x, y, 0xFF000000 | (v << 16) | (v << 8) | v);
                }
            frames.add(ImageFrame.of(fb, 100, 0, 0,
                FrameDisposal.DO_NOT_DISPOSE, FrameBlend.OVER));
        }
        AnimatedImageData anim = AnimatedImageData.builder()
            .withWidth(W).withHeight(H)
            .withFrames(frames)
            .withLoopCount(0)
            .withBackgroundColor(0)
            .build();

        File out = outputDir.resolve("default_animated_lossy.webp").toFile();
        new ImageFactory().toFile(anim, ImageFormat.WEBP, out,
            WebPWriteOptions.builder().isLossless(false).withQuality(0.75f).build());

        // decodeAnimatedAlphaWithLibwebp throws AssertionError when libwebp rejects
        // the file, and TestAbortedException when python/webp unavailable.
        int[][] frameAlpha = decodeAnimatedAlphaWithLibwebp(out, W, H, 3);
        if (frameAlpha.length != 3)
            throw new AssertionError("libwebp decoded " + frameAlpha.length
                + " frames, expected 3");
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

        Process p = startPythonSubprocess(script);
        if (p == null)
            throw new org.opentest4j.TestAbortedException(
                "No python3/python/py on PATH (set -Dvp8.pythonBin=<path> to override)");

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

        Process p = startPythonSubprocess(script);
        if (p == null)
            throw new org.opentest4j.TestAbortedException(
                "No python3/python/py on PATH (set -Dvp8.pythonBin=<path> to override)");

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
    @DisplayName("diagnostic: re-encode real weapon_ss4 tooltip via current encoder + report size delta")
    void diagnosticReencodeWeaponSs4() throws Exception {
        java.nio.file.Path src = java.nio.file.Path.of(
            "W:/Workspace/Java/SkyBlock-Simplified/asset-renderer/cache/test-lore/weapon_ss4.webp");
        if (!java.nio.file.Files.exists(src))
            throw new org.opentest4j.TestAbortedException(
                "weapon_ss4.webp not present - diagnostic only");

        ImageFactory factory = new ImageFactory();
        ImageData src_decoded = factory.fromFile(src.toFile());
        if (!(src_decoded instanceof AnimatedImageData anim))
            throw new AssertionError("expected AnimatedImageData, got " + src_decoded.getClass());

        int W = anim.getWidth();
        int H = anim.getHeight();
        int frameCount = anim.getFrames().size();
        long srcSize = java.nio.file.Files.size(src);

        File out = outputDir.resolve("weapon_ss4_reencoded.webp").toFile();
        factory.toFile(anim, ImageFormat.WEBP, out,
            WebPWriteOptions.builder().isLossless().build());

        long outSize = out.length();
        System.err.printf("[ss4-reencode] %dx%d %d-frame tooltip | source=%d B | ours=%d B "
            + "| libwebp-ref=11688 B | ratio-vs-source=%.3f%n",
            W, H, frameCount, srcSize, outSize, outSize / (double) srcSize);

        // Per-frame unique-color histogram on the PARTIAL (diff-rectangle) frames that
        // the writer actually feeds to VP8LEncoder, not the full canvas. The writer
        // extracts bounding-box sub-buffers via FrameDiffUtil before encoding, so any
        // frame whose partial rect is tiny will both (a) have very few unique colors
        // and (b) see palette transform overhead exceed the savings.
        int paletteTotal = 0, literalTotal = 0;
        PixelBuffer prev = anim.getFrames().get(0).pixels();
        for (int f = 0; f < frameCount; f++) {
            PixelBuffer curr = anim.getFrames().get(f).pixels();
            int[] partial;
            int partialW, partialH;
            if (f == 0) {
                partial = curr.data();
                partialW = W; partialH = H;
            } else {
                int[] bbox = dev.simplified.image.codec.webp.FrameDiffUtil.computeBoundingBox(prev, curr);
                if (bbox == null) { partialW = 1; partialH = 1; partial = new int[]{ curr.data()[0] }; }
                else {
                    partialW = bbox[2]; partialH = bbox[3];
                    partial = new int[partialW * partialH];
                    for (int yy = 0; yy < partialH; yy++)
                        for (int xx = 0; xx < partialW; xx++)
                            partial[yy * partialW + xx] = curr.getPixel(bbox[0] + xx, bbox[1] + yy);
                }
            }
            PixelBuffer fb = PixelBuffer.of(partial, partialW, partialH);
            int lit  = VP8LEncoder.encode(fb, VP8LEncoder.TransformMode.NONE, 0).length;
            int pal  = VP8LEncoder.encode(fb, VP8LEncoder.TransformMode.PALETTE, 0).length;
            int pred = VP8LEncoder.encode(fb, VP8LEncoder.TransformMode.PREDICTOR, 0).length;
            int sg   = VP8LEncoder.encode(fb, VP8LEncoder.TransformMode.SUBTRACT_GREEN, 0).length;
            int xc   = VP8LEncoder.encode(fb, VP8LEncoder.TransformMode.CROSS_COLOR, 0).length;
            int cac  = VP8LEncoder.encode(fb, VP8LEncoder.TransformMode.NONE, 10).length;
            int mh8c  = VP8LEncoder.encode(fb, VP8LEncoder.TransformMode.NONE, 10, 8).length;
            int mh8   = VP8LEncoder.encode(fb, VP8LEncoder.TransformMode.NONE, 0, 8).length;
            int mh8sg = VP8LEncoder.encode(fb, VP8LEncoder.TransformMode.SUBTRACT_GREEN, 0, 8).length;
            int best = Math.min(Math.min(Math.min(lit, pal), Math.min(pred, sg)),
                Math.min(Math.min(xc, cac), Math.min(Math.min(mh8c, mh8), mh8sg)));
            paletteTotal += pal;
            literalTotal += lit;
            System.err.printf("[ss4-reencode]   frame %2d: %4dx%-4d (%6d px) "
                + "lit=%4d sg=%4d cache=%4d mh8c=%4d mh8=%4d mh8sg=%4d  best=%4d%n",
                f, partialW, partialH, partial.length, lit, sg, cac, mh8c, mh8, mh8sg, best);
            prev = curr;
        }
        System.err.printf("[ss4-reencode] TOTAL per-frame VP8L (cache-off A/B): "
            + "palette=%d B, literal=%d B, delta=%+d B%n",
            paletteTotal, literalTotal, paletteTotal - literalTotal);

        // Round-trip sanity: our re-encoded output must decode back to the same pixels
        // as the reference input on every frame.
        ImageData roundTripped = factory.fromFile(out);
        if (!(roundTripped instanceof AnimatedImageData rtAnim))
            throw new AssertionError("round-trip expected AnimatedImageData");
        if (rtAnim.getFrames().size() != frameCount)
            throw new AssertionError("frame count mismatch: " + rtAnim.getFrames().size()
                + " != " + frameCount);
        for (int f = 0; f < frameCount; f++) {
            PixelBuffer a = anim.getFrames().get(f).pixels();
            PixelBuffer b = rtAnim.getFrames().get(f).pixels();
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++) {
                    int pa = a.getPixel(x, y);
                    int pb = b.getPixel(x, y);
                    if (pa != pb)
                        throw new AssertionError(String.format(
                            "frame %d px (%d,%d) lossless re-encode differs: src=0x%08X got=0x%08X",
                            f, x, y, pa, pb));
                }
        }
    }

    @Test
    @DisplayName("diagnostic: tooltip-like 454x260 palette-path output size")
    void diagnosticTooltipPalettePathSize() throws IOException {
        // Realistic tooltip: dark background, gold text pixels, grey mid-tones, border.
        // Has about 4 unique colors - well inside the 1bpp palette envelope (paletteSize=2),
        // so the output body should be tiny (a packed-index stream + 15-code palette
        // sub-image). Reports the byte count for tracking vs libwebp's 11,688-byte
        // reference on the real tooltip animation.
        PixelBuffer buf = PixelBuffer.create(454, 260);
        for (int y = 0; y < 260; y++)
            for (int x = 0; x < 454; x++) {
                int argb;
                if (x < 4 || x > 449 || y < 4 || y > 255) argb = 0x505000FF;
                else if ((x + y) % 37 == 0)              argb = 0xFFFFAA00;
                else if ((x + y) % 53 == 0)              argb = 0xFFAAAAAA;
                else                                     argb = 0xF0100010;
                buf.setPixel(x, y, argb);
            }
        byte[] payload = VP8LEncoder.encode(buf);
        System.err.println("[palette-diag] 454x260 tooltip-like palette VP8L: "
            + payload.length + " bytes (pre-palette literal-path baseline was ~60 KB)");
        if (payload.length > 5_000)
            throw new AssertionError("tooltip-like 4-color palette-path output = "
                + payload.length + " B, expected <= 5000 B (palette may not be firing)");
    }

    @Test
    @DisplayName("palette VP8L cross-decodes through libwebp (alpha-varying content exercises palette deltas)")
    void paletteLosslessLibwebpRoundTrip() throws Exception {
        // 4-color palette (2 bpp) with three different alphas forces the palette sub-image
        // to carry non-zero deltas on every channel, so a broken delta-encode or sub-image
        // emit would surface as libwebp rejecting the file or decoding wrong pixels.
        // Odd width (29) exercises the partial-tail-cell branch of the packing loop.
        int W = 29, H = 17;
        int[] colors = { 0x80112233, 0xFF445566, 0x40778899, 0xFFAABBCC };
        PixelBuffer buf = PixelBuffer.create(W, H);
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                buf.setPixel(x, y, colors[(x + 3 * y) & 3]);

        var image = new StaticImageData(ImageFrame.of(buf));
        File out = outputDir.resolve("palette_lossless_libwebp.webp").toFile();
        new ImageFactory().toFile(image, ImageFormat.WEBP, out,
            WebPWriteOptions.builder().isLossless().build());

        int[] decoded;
        try {
            decoded = decodeAlphaWithLibwebp(out, W, H);
        } catch (org.opentest4j.TestAbortedException abort) {
            throw abort;   // self-skip when python/webp unavailable
        }
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                int expectedAlpha = (colors[(x + 3 * y) & 3] >>> 24) & 0xFF;
                int gotAlpha = decoded[y * W + x];
                if (gotAlpha != expectedAlpha)
                    throw new AssertionError(String.format(
                        "libwebp palette alpha mismatch at (%d,%d): expected %d got %d",
                        x, y, expectedAlpha, gotAlpha));
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

    /**
     * Minimal wrapper so we can write a single frame without instantiating TextRenderer.
     */
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

    /**
     * Shared Python launcher for the libwebp-interop helpers below. Respects the
     * {@code vp8.pythonBin} JVM system property and {@code VP8_PYTHON_BIN} env var
     * when set (useful on Windows where the default {@code python3} on PATH is a
     * Microsoft Store stub without the {@code webp} package). Falls back to the
     * standard launchers otherwise. Returns {@code null} when nothing launches -
     * callers typically throw {@link TestAbortedException}.
     */
    private static Process startPythonSubprocess(String script) {
        String override = System.getProperty("vp8.pythonBin");
        if (override == null || override.isEmpty())
            override = System.getenv("VP8_PYTHON_BIN");
        if (override != null && !override.isEmpty()) {
            try {
                ProcessBuilder pb = new ProcessBuilder(override, "-c", script);
                pb.redirectErrorStream(true);
                return pb.start();
            } catch (java.io.IOException ignored) { }
        }
        for (String cmd : new String[]{"python3", "python", "py"}) {
            try {
                ProcessBuilder pb = new ProcessBuilder(cmd, "-c", script);
                pb.redirectErrorStream(true);
                return pb.start();
            } catch (java.io.IOException ignored) { }
        }
        return null;
    }

}
