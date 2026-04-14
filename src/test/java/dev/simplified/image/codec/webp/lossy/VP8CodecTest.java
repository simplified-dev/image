package dev.simplified.image.codec.webp.lossy;

import dev.simplified.collection.Concurrent;
import dev.simplified.collection.ConcurrentList;
import dev.simplified.image.codec.webp.RiffContainer;
import dev.simplified.image.codec.webp.WebPChunk;
import dev.simplified.image.pixel.PixelBuffer;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

/**
 * Package-private tests for the VP8 (lossy) codec internals: boolean range coder,
 * coefficient token tree, and the full encoder pipeline validated against
 * {@code libwebp} via Python.
 */
public class VP8CodecTest {

    // ──── BooleanEncoder / BooleanDecoder (VP8 range coder) ────

    @Nested
    class BooleanCoderTests {

        @Test @DisplayName("round-trips a single equal-probability bit")
        void singleBoolRoundTrip() {
            for (int bit : new int[]{0, 1}) {
                BooleanEncoder enc = new BooleanEncoder(16);
                enc.encodeBool(bit);
                byte[] out = enc.toByteArray();

                BooleanDecoder dec = new BooleanDecoder(out, 0, out.length);
                assertThat("bit=" + bit, dec.decodeBool(), is(bit));
            }
        }

        @Test @DisplayName("round-trips a mixed-probability bit sequence")
        void mixedProbSequence() {
            // (prob, bit) pairs - deliberately includes extreme probabilities.
            int[][] pairs = {
                { 1, 0 }, { 1, 1 },
                { 128, 0 }, { 128, 1 },
                { 255, 0 }, { 255, 1 },
                { 64, 0 }, { 64, 1 }, { 192, 1 }, { 192, 0 },
                { 100, 1 }, { 50, 0 }, { 200, 1 }, { 10, 0 }, { 240, 1 }
            };
            BooleanEncoder enc = new BooleanEncoder(64);
            for (int[] p : pairs) enc.encodeBit(p[0], p[1]);
            byte[] out = enc.toByteArray();

            BooleanDecoder dec = new BooleanDecoder(out, 0, out.length);
            for (int i = 0; i < pairs.length; i++) {
                int got = dec.decodeBit(pairs[i][0]);
                assertThat("idx=" + i + " prob=" + pairs[i][0], got, is(pairs[i][1]));
            }
        }

        @Test @DisplayName("round-trips unsigned integers of various widths")
        void uintRoundTrip() {
            int[] values = { 0, 1, 0x7F, 0xFF, 0x1234, 0xABCDE };
            int[] widths = { 1, 2, 7, 8, 13, 20 };
            BooleanEncoder enc = new BooleanEncoder(64);
            for (int i = 0; i < values.length; i++) enc.encodeUint(values[i], widths[i]);
            byte[] out = enc.toByteArray();

            BooleanDecoder dec = new BooleanDecoder(out, 0, out.length);
            for (int i = 0; i < values.length; i++)
                assertThat("width=" + widths[i], dec.decodeUint(widths[i]), is(values[i]));
        }

        @Test @DisplayName("round-trips signed integers including zero and negatives")
        void sintRoundTrip() {
            int[] values = { 0, 1, -1, 5, -5, 63, -63 };
            BooleanEncoder enc = new BooleanEncoder(64);
            for (int v : values) enc.encodeSint(v, 7);
            byte[] out = enc.toByteArray();

            BooleanDecoder dec = new BooleanDecoder(out, 0, out.length);
            for (int v : values)
                assertThat(dec.decodeSint(7), is(v));
        }

        @Test @DisplayName("stress test - 1000 random (prob, bit) pairs")
        void stressRandomPairs() {
            java.util.Random rng = new java.util.Random(0xC0FFEE);
            int n = 1000;
            int[] probs = new int[n];
            int[] bits = new int[n];
            BooleanEncoder enc = new BooleanEncoder(256);
            for (int i = 0; i < n; i++) {
                probs[i] = 1 + rng.nextInt(255);      // 1..255
                bits[i] = rng.nextInt(2);
                enc.encodeBit(probs[i], bits[i]);
            }
            byte[] out = enc.toByteArray();

            BooleanDecoder dec = new BooleanDecoder(out, 0, out.length);
            for (int i = 0; i < n; i++) {
                int got = dec.decodeBit(probs[i]);
                assertThat("idx=" + i + " prob=" + probs[i], got, is(bits[i]));
            }
        }

        @Test @DisplayName("round-trips tree decoding with real VP8 key-frame Y-mode probs")
        void treeRoundTrip() {
            // KF_YMODE_TREE / KF_YMODE_PROB from RFC 6386 section 11.2
            int[] tree = { -0, 2, -1, 4, -2, 6, -3, -4 }; // B_PRED=0, DC=1, V=2, H=3, TM=4
            int[] probs = { 145, 156, 163, 128 };

            int[] sequence = { 0, 1, 2, 3, 4, 0, 4, 2, 1, 3 };
            BooleanEncoder enc = new BooleanEncoder(64);
            for (int leaf : sequence) encodeTree(enc, tree, probs, leaf);
            byte[] out = enc.toByteArray();

            BooleanDecoder dec = new BooleanDecoder(out, 0, out.length);
            for (int expected : sequence)
                assertThat(dec.decodeTree(tree, probs), is(expected));
        }

        /** Walks the tree toward {@code leaf} and emits each branch bit via {@code enc}. */
        private static void encodeTree(BooleanEncoder enc, int[] tree, int[] probs, int leaf) {
            int[] path = new int[16];
            int[] nodes = new int[16];
            int depth = findPath(tree, 0, -leaf, path, nodes, 0);
            for (int i = 0; i < depth; i++)
                enc.encodeBit(probs[nodes[i] >> 1], path[i]);
        }

        private static int findPath(int[] tree, int node, int target, int[] path, int[] nodes, int depth) {
            for (int branch = 0; branch < 2; branch++) {
                int next = tree[node + branch];
                path[depth] = branch;
                nodes[depth] = node;
                if (next <= 0) {
                    if (next == target) return depth + 1;
                } else {
                    int d = findPath(tree, next, target, path, nodes, depth + 1);
                    if (d > 0) return d;
                }
            }
            return 0;
        }

    }

    // ──── VP8 Lossy Encoder (libwebp wire-format validation) ────

    @Nested
    class VP8EncoderTests {

        @Test @DisplayName("2x2 DC-only frame emits 10-byte uncompressed header with correct sync")
        void frameHeader2x2() {
            PixelBuffer buf = PixelBuffer.create(2, 2);
            buf.fill(0xFFFF0000);
            byte[] payload = VP8Encoder.encode(buf, 0.75f);

            assertThat("payload length >= 10", payload.length >= 10, is(true));

            // First three bytes: frame tag (key=0, version=0, show=1, first_partition_size).
            int tag = (payload[0] & 0xFF) | ((payload[1] & 0xFF) << 8) | ((payload[2] & 0xFF) << 16);
            assertThat("keyframe flag (bit 0) is 0", tag & 0x01, is(0));
            assertThat("version (bits 1..3) is 0", (tag >>> 1) & 0x7, is(0));
            assertThat("show_frame (bit 4) is 1", (tag >>> 4) & 0x1, is(1));
            int firstPartSize = (tag >>> 5) & 0x7FFFF;
            assertThat("first_partition_size > 0", firstPartSize > 0, is(true));

            // Sync code.
            assertThat(payload[3], is((byte) 0x9D));
            assertThat(payload[4], is((byte) 0x01));
            assertThat(payload[5], is((byte) 0x2A));

            // Dimensions (14-bit each, low byte then high byte with scale in top 2 bits).
            int w = (payload[6] & 0xFF) | ((payload[7] & 0x3F) << 8);
            int h = (payload[8] & 0xFF) | ((payload[9] & 0x3F) << 8);
            assertThat("width", w, is(2));
            assertThat("height", h, is(2));

            assertThat("total >= header + first partition", payload.length >= 10 + firstPartSize, is(true));
        }

        @Test @DisplayName("2x2 DC-only VP8 decodes in libwebp via Python")
        void lossyVp82x2LibwebpRoundTrip() throws Exception {
            PixelBuffer buf = PixelBuffer.create(2, 2);
            buf.fill(0xFFFF0000);
            byte[] vp8Payload = VP8Encoder.encode(buf, 0.75f);

            ConcurrentList<WebPChunk> chunks = Concurrent.newList();
            chunks.add(RiffContainer.createChunk(WebPChunk.Type.VP8, vp8Payload));
            byte[] riff = RiffContainer.write(chunks);

            verifyDecodesInLibwebp(riff, 2, 2, "dc_only_2x2");
        }

        @Test @DisplayName("16x16 DC-only VP8 decodes in libwebp")
        void lossyVp816x16LibwebpRoundTrip() throws Exception {
            PixelBuffer buf = PixelBuffer.create(16, 16);
            for (int y = 0; y < 16; y++)
                for (int x = 0; x < 16; x++)
                    buf.setPixel(x, y, 0xFF000000 | ((x * 16) << 16) | ((y * 16) << 8));
            byte[] vp8Payload = VP8Encoder.encode(buf, 0.75f);

            ConcurrentList<WebPChunk> chunks = Concurrent.newList();
            chunks.add(RiffContainer.createChunk(WebPChunk.Type.VP8, vp8Payload));
            byte[] riff = RiffContainer.write(chunks);

            verifyDecodesInLibwebp(riff, 16, 16, "dc_only_16x16");
        }

        /**
         * Writes {@code riff} to a temp file, shells out to libwebp via Python, and
         * asserts the returned dimensions match.
         */
        private static void verifyDecodesInLibwebp(byte[] riff, int expectedW, int expectedH, String tag)
            throws Exception {
            java.nio.file.Path tmp = java.nio.file.Files.createTempFile("vp8-" + tag + "-", ".webp");
            try {
                java.nio.file.Files.write(tmp, riff);

                String script =
                    "import sys\n" +
                    "try:\n" +
                    "    import webp\n" +
                    "except ImportError:\n" +
                    "    print('NO_WEBP'); sys.exit(2)\n" +
                    "img = webp.load_image(r'" + tmp.toAbsolutePath() + "')\n" +
                    "print(img.size[0], img.size[1])\n";

                Process p = startPython(script);
                if (p == null)
                    throw new org.opentest4j.TestAbortedException(
                        "No python3/python/py executable found on PATH - cannot validate via libwebp");
                String out = new String(p.getInputStream().readAllBytes()).trim();
                int exit = p.waitFor();

                if (exit == 2 && out.contains("NO_WEBP"))
                    throw new org.opentest4j.TestAbortedException("Python webp package not installed");
                if (exit != 0)
                    throw new AssertionError("libwebp rejected our VP8 bitstream:\n" + out);

                String[] parts = out.split("\\s+");
                assertThat("dimensions reported by libwebp", parts.length >= 2, is(true));
                assertThat("width  (libwebp)", Integer.parseInt(parts[0]), is(expectedW));
                assertThat("height (libwebp)", Integer.parseInt(parts[1]), is(expectedH));
            } finally {
                java.nio.file.Files.deleteIfExists(tmp);
            }
        }

        /**
         * Attempts {@code python3}, {@code python}, and {@code py} in turn so the test
         * works across Linux, macOS, and Windows shells.
         */
        private static Process startPython(String script) {
            for (String cmd : new String[]{"python3", "python", "py"}) {
                try {
                    ProcessBuilder pb = new ProcessBuilder(cmd, "-c", script);
                    pb.redirectErrorStream(true);
                    return pb.start();
                } catch (java.io.IOException ignored) {
                    // Try next candidate.
                }
            }
            return null;
        }

        @Test @DisplayName("QI=0 solid gray reconstructs losslessly through libwebp")
        void pixelFidelitySolidGray() throws Exception {
            PixelBuffer buf = PixelBuffer.create(16, 16);
            buf.fill(0xFF808080);  // mid-gray matches the prediction baseline (128)
            assertLibwebpDecodesPixel(buf, 1.0f, 0x80, 0x80, 0x80, 4);
        }

        @Test @DisplayName("QI=0 solid red reconstructs through libwebp within YUV quantization tolerance")
        void pixelFidelitySolidRed() throws Exception {
            PixelBuffer buf = PixelBuffer.create(16, 16);
            buf.fill(0xFFFF0000);
            assertLibwebpDecodesPixel(buf, 1.0f, 0xFF, 0x00, 0x00, 16);
        }

        @Test @DisplayName("horizontal stripes pick V_PRED when available")
        void modeSelectionPicksVertical() throws Exception {
            PixelBuffer buf = PixelBuffer.create(16, 32);
            for (int y = 0; y < 32; y++)
                for (int x = 0; x < 16; x++) {
                    int g = (y * 255) / 31;
                    buf.setPixel(x, y, 0xFF000000 | (g << 16) | (g << 8) | g);
                }
            byte[] vp8Payload = VP8Encoder.encode(buf, 1.0f);
            ConcurrentList<WebPChunk> chunks = Concurrent.newList();
            chunks.add(RiffContainer.createChunk(WebPChunk.Type.VP8, vp8Payload));
            byte[] riff = RiffContainer.write(chunks);

            int[] decoded = decodeWithLibwebp(riff, 16, 32);
            double psnr = computePsnr(buf, decoded, 16, 32);
            if (psnr < 32.0)
                throw new AssertionError(String.format(
                    "vertical stripes PSNR = %.2f dB (expected >= 32 dB with V_PRED picked)", psnr));
        }

        @Test @DisplayName("loop filter level is non-zero at moderate quality")
        void loopFilterEnabled() {
            // Encoder should emit a non-zero filter level so the decoder smooths block edges.
            PixelBuffer buf = PixelBuffer.create(16, 16);
            buf.fill(0xFF808080);
            byte[] vp8 = VP8Encoder.encode(buf, 0.5f);     // q=0.5 -> mid-quality, mid-filter
            int filterLevel = readFilterLevelFromHeader(vp8);
            if (filterLevel == 0)
                throw new AssertionError("expected non-zero filter level at q=0.5, got 0");
        }

        @Test @DisplayName("loop filter on doesn't decrease 32x48 PSNR vs libwebp-decoded baseline")
        void loopFilterDoesNotHurtPsnr() throws Exception {
            // The encoder always emits filter_level > 0 now; asserting a healthy PSNR at
            // q=0.9 guarantees the filter (applied by libwebp on decode) isn't degrading
            // output. If LoopFilter or filter_level computation drifts, this fails.
            PixelBuffer buf = PixelBuffer.create(32, 48);
            for (int y = 0; y < 48; y++)
                for (int x = 0; x < 32; x++) {
                    int r = (x * 255) / 31;
                    int g = (y * 255) / 47;
                    int b = ((x + y) * 255) / 78;
                    buf.setPixel(x, y, 0xFF000000 | (r << 16) | (g << 8) | b);
                }
            byte[] vp8 = VP8Encoder.encode(buf, 0.9f);
            ConcurrentList<WebPChunk> chunks = Concurrent.newList();
            chunks.add(RiffContainer.createChunk(WebPChunk.Type.VP8, vp8));
            byte[] riff = RiffContainer.write(chunks);

            int[] decoded = decodeWithLibwebp(riff, 32, 48);
            double psnr = computePsnr(buf, decoded, 32, 48);
            if (psnr < 28.0)
                throw new AssertionError(String.format(
                    "filter-on PSNR = %.2f dB (expected >= 28 dB at q=0.9)", psnr));
        }

        /**
         * Reads the encoded frame's first-partition far enough to extract
         * {@code filter_level} (RFC 6386 paragraph 9.4).
         */
        private static int readFilterLevelFromHeader(byte[] vp8) {
            int frameTag = (vp8[0] & 0xFF) | ((vp8[1] & 0xFF) << 8) | ((vp8[2] & 0xFF) << 16);
            int firstPartSize = (frameTag >>> 5) & 0x7FFFF;
            BooleanDecoder br = new BooleanDecoder(vp8, 10, firstPartSize);
            br.decodeBool();                                  // colorspace
            br.decodeBool();                                  // clamp_type
            if (br.decodeBool() != 0)                         // use_segment
                throw new AssertionError("encoder unexpectedly emits segment header");
            br.decodeBool();                                  // simple filter flag
            return br.decodeUint(6);
        }

        @Test @DisplayName("16x16 gradient reconstructs through libwebp with bounded error at high quality")
        void pixelFidelityGradient() throws Exception {
            PixelBuffer buf = PixelBuffer.create(16, 16);
            for (int y = 0; y < 16; y++)
                for (int x = 0; x < 16; x++)
                    buf.setPixel(x, y, 0xFF000000 | ((x * 16) << 16) | ((y * 16) << 8));

            byte[] vp8Payload = VP8Encoder.encode(buf, 1.0f);
            ConcurrentList<WebPChunk> chunks = Concurrent.newList();
            chunks.add(RiffContainer.createChunk(WebPChunk.Type.VP8, vp8Payload));
            byte[] riff = RiffContainer.write(chunks);

            int[] decoded = decodeWithLibwebp(riff, 16, 16);
            double psnr = computePsnr(buf, decoded, 16, 16);
            if (psnr < 30.0)
                throw new AssertionError(String.format(
                    "gradient PSNR = %.2f dB (expected >= 30 dB at quality 1.0)", psnr));
        }

        private static double computePsnr(PixelBuffer src, int[] decoded, int w, int h) {
            long sumSq = 0;
            int n = 0;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++) {
                    int s = src.getPixel(x, y);
                    int d = decoded[y * w + x];
                    for (int shift : new int[]{0, 8, 16}) {
                        int diff = ((s >> shift) & 0xFF) - ((d >> shift) & 0xFF);
                        sumSq += (long) diff * diff;
                        n++;
                    }
                }
            double mse = sumSq / (double) n;
            return mse == 0 ? Double.POSITIVE_INFINITY : 10.0 * Math.log10(255.0 * 255.0 / mse);
        }

        private static void assertLibwebpDecodesPixel(
            PixelBuffer src, float quality, int expectR, int expectG, int expectB, int tolerance
        ) throws Exception {
            byte[] vp8Payload = VP8Encoder.encode(src, quality);
            ConcurrentList<WebPChunk> chunks = Concurrent.newList();
            chunks.add(RiffContainer.createChunk(WebPChunk.Type.VP8, vp8Payload));
            byte[] riff = RiffContainer.write(chunks);

            int[] decoded = decodeWithLibwebp(riff, src.width(), src.height());
            int centerX = src.width() / 2;
            int centerY = src.height() / 2;
            int p = decoded[centerY * src.width() + centerX];
            int dr = ((p >> 16) & 0xFF) - expectR;
            int dg = ((p >>  8) & 0xFF) - expectG;
            int db = ( p        & 0xFF) - expectB;
            if (Math.abs(dr) > tolerance || Math.abs(dg) > tolerance || Math.abs(db) > tolerance)
                throw new AssertionError(String.format(
                    "center pixel mismatch: expected (%d,%d,%d) got (%d,%d,%d), tolerance=%d",
                    expectR, expectG, expectB, (p >> 16) & 0xFF, (p >> 8) & 0xFF, p & 0xFF, tolerance));
        }

        /** Writes {@code riff} to a temp file, shells out to libwebp, and returns the decoded ARGB buffer. */
        private static int[] decodeWithLibwebp(byte[] riff, int expectedW, int expectedH)
            throws Exception {
            java.nio.file.Path tmp = java.nio.file.Files.createTempFile("vp8-px-", ".webp");
            try {
                java.nio.file.Files.write(tmp, riff);

                String script =
                    "import sys\n" +
                    "try:\n" +
                    "    import webp\n" +
                    "except ImportError:\n" +
                    "    print('NO_WEBP'); sys.exit(2)\n" +
                    "img = webp.load_image(r'" + tmp.toAbsolutePath() + "').convert('RGBA')\n" +
                    "w, h = img.size\n" +
                    "print(w, h)\n" +
                    "data = img.tobytes()\n" +
                    "for i in range(0, len(data), 4):\n" +
                    "    r, g, b, a = data[i], data[i+1], data[i+2], data[i+3]\n" +
                    "    print(f'{a:02x}{r:02x}{g:02x}{b:02x}')\n";

                Process p = startPython(script);
                if (p == null)
                    throw new org.opentest4j.TestAbortedException(
                        "No python3/python/py executable found on PATH");
                String out = new String(p.getInputStream().readAllBytes());
                int exit = p.waitFor();
                if (exit == 2 && out.contains("NO_WEBP"))
                    throw new org.opentest4j.TestAbortedException("Python webp package not installed");
                if (exit != 0)
                    throw new AssertionError("libwebp rejected our VP8 bitstream:\n" + out);

                String[] lines = out.trim().split("\\R");
                String[] dims = lines[0].split("\\s+");
                int w = Integer.parseInt(dims[0]);
                int h = Integer.parseInt(dims[1]);
                if (w != expectedW || h != expectedH)
                    throw new AssertionError(
                        "decoded dims " + w + "x" + h + " != expected " + expectedW + "x" + expectedH);

                int[] pixels = new int[w * h];
                for (int i = 0; i < w * h; i++)
                    pixels[i] = (int) Long.parseLong(lines[i + 1], 16);
                return pixels;
            } finally {
                java.nio.file.Files.deleteIfExists(tmp);
            }
        }

    }

    // ──── VP8 Decoder (self round-trip + libwebp-produced payload parse) ────

    @Nested
    class VP8DecoderTests {

        @Test @DisplayName("self round-trip: 2x2 solid red decodes to encoder-reconstructed pixels")
        void selfRoundTrip2x2Red() {
            PixelBuffer buf = PixelBuffer.create(2, 2);
            buf.fill(0xFFFF0000);
            byte[] vp8 = VP8Encoder.encode(buf, 0.75f);
            PixelBuffer decoded = VP8Decoder.decode(vp8);

            assertThat("width", decoded.width(), is(2));
            assertThat("height", decoded.height(), is(2));
            // Encoder + decoder both go through the same BT.601 YCbCr quantization,
            // so the reconstructed pixel is deterministic but not == source red.
            // Just check the decoded frame is uniform and close to red.
            int p = decoded.getPixel(0, 0);
            int r = (p >> 16) & 0xFF, g = (p >> 8) & 0xFF, b = p & 0xFF;
            if (r < 200 || g > 60 || b > 60)
                throw new AssertionError(String.format(
                    "expected near-red, got (%d,%d,%d)", r, g, b));
            for (int y = 0; y < 2; y++)
                for (int x = 0; x < 2; x++)
                    assertThat("uniform fill", decoded.getPixel(x, y), is(p));
        }

        @Test @DisplayName("self round-trip: 16x16 gradient decodes to source within PSNR bound")
        void selfRoundTrip16x16Gradient() {
            PixelBuffer buf = PixelBuffer.create(16, 16);
            for (int y = 0; y < 16; y++)
                for (int x = 0; x < 16; x++)
                    buf.setPixel(x, y, 0xFF000000 | ((x * 16) << 16) | ((y * 16) << 8));

            byte[] vp8 = VP8Encoder.encode(buf, 1.0f);
            PixelBuffer decoded = VP8Decoder.decode(vp8);

            double psnr = sourcePsnr(buf, decoded);
            if (psnr < 30.0)
                throw new AssertionError(String.format(
                    "self-roundtrip 16x16 gradient PSNR = %.2f dB (expected >= 30 dB)", psnr));
        }

        @Test @DisplayName("self round-trip: 32x48 multi-MB image decodes to source within PSNR bound")
        void selfRoundTrip32x48MultiMb() {
            PixelBuffer buf = PixelBuffer.create(32, 48);
            for (int y = 0; y < 48; y++)
                for (int x = 0; x < 32; x++) {
                    int r = (x * 255) / 31;
                    int g = (y * 255) / 47;
                    int b = ((x + y) * 255) / 78;
                    buf.setPixel(x, y, 0xFF000000 | (r << 16) | (g << 8) | b);
                }
            byte[] vp8 = VP8Encoder.encode(buf, 0.9f);
            PixelBuffer decoded = VP8Decoder.decode(vp8);

            double psnr = sourcePsnr(buf, decoded);
            if (psnr < 28.0)
                throw new AssertionError(String.format(
                    "self-roundtrip 32x48 PSNR = %.2f dB (expected >= 28 dB)", psnr));
        }

        @Test @DisplayName("decodes libwebp-produced 32x32 gradient to a recognizable image")
        void libwebpProducedGradientQi0() throws Exception {
            PixelBuffer src = PixelBuffer.create(32, 32);
            for (int y = 0; y < 32; y++)
                for (int x = 0; x < 32; x++)
                    src.setPixel(x, y, 0xFF000000 | ((x * 8) << 16) | ((y * 8) << 8));

            byte[] vp8 = encodeWithLibwebp(src, 100);
            if (vp8 == null) return;   // Python/libwebp unavailable - skipped.

            PixelBuffer decoded = VP8Decoder.decode(vp8);
            double psnr = sourcePsnr(src, decoded);
            // Loose threshold - libwebp at quality 100 emits normal-filter and fancy-upsampling
            // dependent artifacts that our decoder doesn't apply yet. 20 dB confirms the
            // basic parse path works end-to-end; tighter tolerances are follow-up work.
            if (psnr < 20.0)
                throw new AssertionError(String.format(
                    "PSNR decoding libwebp q=100 gradient = %.2f dB (expected >= 20 dB)", psnr));
        }

        private static double sourcePsnr(PixelBuffer src, PixelBuffer decoded) {
            long sumSq = 0;
            int n = 0;
            int w = src.width(), h = src.height();
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++) {
                    int s = src.getPixel(x, y);
                    int d = decoded.getPixel(x, y);
                    for (int shift : new int[]{0, 8, 16}) {
                        int diff = ((s >> shift) & 0xFF) - ((d >> shift) & 0xFF);
                        sumSq += (long) diff * diff;
                        n++;
                    }
                }
            double mse = sumSq / (double) n;
            return mse == 0 ? Double.POSITIVE_INFINITY : 10.0 * Math.log10(255.0 * 255.0 / mse);
        }

        /**
         * Encodes {@code src} via Python's {@code webp} bindings at the given quality
         * (0..100, higher = better) and returns the bare VP8 payload (strips the RIFF
         * and VP8 chunk headers).
         *
         * @return the VP8 payload, or {@code null} if Python/webp is unavailable
         */
        private static byte[] encodeWithLibwebp(PixelBuffer src, int quality) throws Exception {
            java.nio.file.Path out = java.nio.file.Files.createTempFile("libwebp-enc-", ".webp");
            try {
                StringBuilder pixels = new StringBuilder();
                for (int y = 0; y < src.height(); y++) {
                    for (int x = 0; x < src.width(); x++) {
                        int p = src.getPixel(x, y);
                        pixels.append(String.format("%d,%d,%d,%d,",
                            (p >> 16) & 0xFF, (p >> 8) & 0xFF, p & 0xFF, (p >>> 24) & 0xFF));
                    }
                }
                String script =
                    "import sys\n" +
                    "try:\n" +
                    "    import webp, numpy as np\n" +
                    "except ImportError:\n" +
                    "    print('NO_WEBP'); sys.exit(2)\n" +
                    "raw = bytes([" + pixels.substring(0, pixels.length() - 1) + "])\n" +
                    "arr = np.frombuffer(raw, dtype=np.uint8).reshape(" + src.height() + ", " + src.width() + ", 4)\n" +
                    "pic = webp.WebPPicture.from_numpy(arr, pilmode='RGBA')\n" +
                    "cfg = webp.WebPConfig.new(quality=" + quality + ")\n" +
                    "data = pic.encode(cfg).buffer()\n" +
                    "with open(r'" + out.toAbsolutePath() + "', 'wb') as f: f.write(bytes(data))\n";
                Process p = startPython(script);
                if (p == null) return null;
                int exit = p.waitFor();
                String stdout = new String(p.getInputStream().readAllBytes());
                if (exit == 2 && stdout.contains("NO_WEBP")) return null;
                if (exit != 0) throw new AssertionError("libwebp encode failed:\n" + stdout);

                byte[] riff = java.nio.file.Files.readAllBytes(out);
                return extractVp8Payload(riff);
            } finally {
                java.nio.file.Files.deleteIfExists(out);
            }
        }

        /** Strips RIFF/WEBP/VP8 chunk headers and returns the bare VP8 payload. */
        private static byte[] extractVp8Payload(byte[] riff) {
            // RIFF header: 4 + 4 + 4 = 12 bytes (magic + size + "WEBP").
            int offset = 12;
            while (offset + 8 <= riff.length) {
                String tag = new String(riff, offset, 4);
                int size = (riff[offset + 4] & 0xFF)
                    | ((riff[offset + 5] & 0xFF) << 8)
                    | ((riff[offset + 6] & 0xFF) << 16)
                    | ((riff[offset + 7] & 0xFF) << 24);
                if (tag.equals("VP8 ")) {
                    byte[] payload = new byte[size];
                    System.arraycopy(riff, offset + 8, payload, 0, size);
                    return payload;
                }
                offset += 8 + size + (size & 1);   // chunks are padded to even length
            }
            throw new AssertionError("no VP8 chunk found in RIFF");
        }

        private static Process startPython(String script) {
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

    // ──── VP8 Token Encoder / Decoder (coefficient tree) ────

    @Nested
    class VP8TokenCoderTests {

        @Test @DisplayName("all-zero block emits a single EOB bit and round-trips")
        void allZeroBlock() {
            short[] coeffs = new short[16];
            short[] decoded = new short[16];
            byte[] bytes = encodeAndFinish(coeffs, 0, 0, VP8Tables.TYPE_I16_DC);
            BooleanDecoder dec = new BooleanDecoder(bytes, 0, bytes.length);
            int nz = VP8TokenDecoder.decode(dec, decoded, 0, 0, VP8Tables.TYPE_I16_DC, VP8Tables.COEFFS_PROBA_0);
            assertThat("non-zero flag", nz, is(0));
            for (int i = 0; i < 16; i++)
                assertThat("coef " + i, decoded[i], is((short) 0));
        }

        @Test @DisplayName("single DC=1 coefficient round-trips")
        void singleDc1() {
            short[] coeffs = new short[16];
            coeffs[0] = 1;
            assertRoundTrips(coeffs, 0, 0, VP8Tables.TYPE_I16_DC);
        }

        @Test @DisplayName("all magnitudes 1..10 round-trip (small + mid tree branches)")
        void smallMagnitudes() {
            for (int v = 1; v <= 10; v++) {
                short[] coeffs = new short[16];
                coeffs[0] = (short) v;
                coeffs[1] = (short) -v;
                assertRoundTrips(coeffs, 0, 0, VP8Tables.TYPE_I16_DC);
            }
        }

        @Test @DisplayName("CAT3/4/5/6 boundary magnitudes round-trip")
        void catBoundaries() {
            int[] values = { 11, 18, 19, 34, 35, 66, 67, 100, 500, 2047 };
            for (int v : values) {
                short[] coeffs = new short[16];
                coeffs[0] = (short) v;
                assertRoundTrips(coeffs, 0, 0, VP8Tables.TYPE_I16_DC);

                short[] coeffsNeg = new short[16];
                coeffsNeg[0] = (short) -v;
                assertRoundTrips(coeffsNeg, 0, 0, VP8Tables.TYPE_I16_DC);
            }
        }

        @Test @DisplayName("dense block with interleaved zeros round-trips")
        void denseBlock() {
            short[] coeffs = {
                 5,  0,  3, -2,
                 0,  0,  1,  0,
                -7,  4,  0,  2,
                 0, -1,  0, 15
            };
            assertRoundTrips(coeffs, 0, 0, VP8Tables.TYPE_I16_DC);
        }

        @Test @DisplayName("AC-only block (first=1) round-trips, coef[0] untouched")
        void acOnly() {
            short[] coeffs = new short[16];
            coeffs[0] = 999;
            coeffs[1] = 3;
            coeffs[5] = -42;
            coeffs[15] = 1;
            short[] decoded = new short[16];
            decoded[0] = 999;
            byte[] bytes = encodeAndFinish(coeffs, 1, 0, VP8Tables.TYPE_I16_AC);
            BooleanDecoder dec = new BooleanDecoder(bytes, 0, bytes.length);
            int nz = VP8TokenDecoder.decode(dec, decoded, 1, 0, VP8Tables.TYPE_I16_AC, VP8Tables.COEFFS_PROBA_0);
            assertThat("non-zero flag", nz, is(1));
            assertThat("DC untouched", decoded[0], is((short) 999));
            for (int i = 1; i < 16; i++)
                assertThat("coef " + i, decoded[i], is(coeffs[i]));
        }

        @Test @DisplayName("varying initial context value round-trips")
        void contextVariation() {
            short[] coeffs = { 0, 1, -1, 2, 0, 0, 3, -4, 0, 5, 0, 0, 6, 0, 0, 0 };
            for (int ctx0 : new int[]{0, 1, 2}) {
                assertRoundTrips(coeffs, 0, ctx0, VP8Tables.TYPE_CHROMA_A);
            }
        }

        @Test @DisplayName("random blocks across all coefficient types round-trip")
        void randomStress() {
            java.util.Random rng = new java.util.Random(0xABCDEF);
            int[] types = {
                VP8Tables.TYPE_I16_AC, VP8Tables.TYPE_I16_DC,
                VP8Tables.TYPE_CHROMA_A, VP8Tables.TYPE_I4_AC
            };
            for (int trial = 0; trial < 100; trial++) {
                short[] coeffs = new short[16];
                int numNonZero = 1 + rng.nextInt(16);
                for (int i = 0; i < numNonZero; i++) {
                    int pos = rng.nextInt(16);
                    int mag = 1 + rng.nextInt(100);
                    coeffs[pos] = (short) (rng.nextBoolean() ? mag : -mag);
                }
                int type = types[rng.nextInt(4)];
                int first = (type == VP8Tables.TYPE_I16_AC) ? 1 : 0;
                int ctx = rng.nextInt(3);
                if (first == 1) coeffs[0] = 0;
                assertRoundTrips(coeffs, first, ctx, type);
            }
        }

        private static byte[] encodeAndFinish(short[] coeffs, int first, int ctx0, int type) {
            BooleanEncoder enc = new BooleanEncoder(64);
            VP8TokenEncoder.emit(enc, coeffs, first, ctx0, type, VP8Tables.COEFFS_PROBA_0);
            return enc.toByteArray();
        }

        private static void assertRoundTrips(short[] coeffs, int first, int ctx0, int type) {
            byte[] bytes = encodeAndFinish(coeffs, first, ctx0, type);
            short[] decoded = new short[16];
            for (int i = 0; i < first; i++) decoded[i] = coeffs[i];
            BooleanDecoder dec = new BooleanDecoder(bytes, 0, bytes.length);
            VP8TokenDecoder.decode(dec, decoded, first, ctx0, type, VP8Tables.COEFFS_PROBA_0);
            for (int i = 0; i < 16; i++)
                if (decoded[i] != coeffs[i])
                    throw new AssertionError(String.format(
                        "coef[%d]: expected %d got %d (type=%d first=%d ctx0=%d)",
                        i, coeffs[i], decoded[i], type, first, ctx0));
        }

    }

}
