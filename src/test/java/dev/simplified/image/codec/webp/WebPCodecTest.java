package dev.simplified.image.codec.webp;

import dev.simplified.collection.Concurrent;
import dev.simplified.collection.ConcurrentList;
import dev.simplified.image.pixel.PixelBuffer;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class WebPCodecTest {

    // ──── RIFF Container ────

    @Nested
    class RiffContainerTests {

        @Test
        void parseRejectsNonRiff() {
            byte[] garbage = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B};
            assertThrows(Exception.class, () -> RiffContainer.parse(garbage));
        }

        @Test
        void writeAndParseRoundTrip() {
            byte[] testPayload = {0x01, 0x02, 0x03, 0x04};
            ConcurrentList<WebPChunk> chunks = Concurrent.newList();
            chunks.add(RiffContainer.createChunk(WebPChunkType.VP8L, testPayload));

            byte[] riffBytes = RiffContainer.write(chunks);
            ConcurrentList<WebPChunk> parsed = RiffContainer.parse(riffBytes);

            assertThat(parsed, hasSize(1));
            assertThat(parsed.getFirst().type(), is(WebPChunkType.VP8L));
            assertThat(parsed.getFirst().payloadLength(), is(4));

            byte[] roundTripped = parsed.getFirst().payload();
            assertThat(roundTripped[0], is((byte) 0x01));
            assertThat(roundTripped[3], is((byte) 0x04));
        }

        @Test
        void multipleChunksPreserved() {
            byte[] payload1 = {0x10, 0x20};
            byte[] payload2 = {0x30, 0x40, 0x50};

            ConcurrentList<WebPChunk> chunks = Concurrent.newList();
            chunks.add(RiffContainer.createChunk(WebPChunkType.VP8X, payload1));
            chunks.add(RiffContainer.createChunk(WebPChunkType.VP8L, payload2));

            byte[] riffBytes = RiffContainer.write(chunks);
            ConcurrentList<WebPChunk> parsed = RiffContainer.parse(riffBytes);

            assertThat(parsed, hasSize(2));
            assertThat(parsed.get(0).type(), is(WebPChunkType.VP8X));
            assertThat(parsed.get(1).type(), is(WebPChunkType.VP8L));
            assertThat(parsed.get(1).payloadLength(), is(3));
        }

    }

    // ──── BitReader / BitWriter ────

    @Nested
    class BitReaderWriterTests {

        @Test
        void writeThenReadRoundTrip() {
            BitWriter writer = new BitWriter();
            writer.writeBits(5, 3);
            writer.writeBits(13, 4);
            writer.writeBits(0, 1);
            byte[] data = writer.toByteArray();

            BitReader reader = new BitReader(data);
            assertThat(reader.readBits(3), is(5));
            assertThat(reader.readBits(4), is(13));
            assertThat(reader.readBits(1), is(0));
        }

        @Test
        void multiByteValuePreserved() {
            BitWriter writer = new BitWriter();
            writer.writeBits(0xABCD, 16);
            byte[] data = writer.toByteArray();

            BitReader reader = new BitReader(data);
            assertThat(reader.readBits(16), is(0xABCD));
        }

        @Test
        void singleBitOperations() {
            BitWriter writer = new BitWriter();
            writer.writeBit(1);
            writer.writeBit(0);
            writer.writeBit(1);
            writer.writeBit(1);
            byte[] data = writer.toByteArray();

            BitReader reader = new BitReader(data);
            assertThat(reader.readBit(), is(1));
            assertThat(reader.readBit(), is(0));
            assertThat(reader.readBit(), is(1));
            assertThat(reader.readBit(), is(1));
        }

        @Test
        void zeroBitsReturnsZero() {
            BitWriter writer = new BitWriter();
            writer.writeBits(0xFF, 8);
            byte[] data = writer.toByteArray();

            BitReader reader = new BitReader(data);
            assertThat(reader.readBits(0), is(0));
            assertThat(reader.readBits(8), is(0xFF));
        }

    }

    // ──── VP8L Encoder/Decoder Round-Trip (no RIFF wrapper) ────

    @Nested
    class VP8LRoundTripTests {

        private void assertRoundTrips(PixelBuffer src) {
            byte[] payload = VP8LEncoder.encode(src);
            PixelBuffer out = VP8LDecoder.decode(payload);
            assertThat("width", out.width(), is(src.width()));
            assertThat("height", out.height(), is(src.height()));
            int[] expected = src.pixels();
            int[] actual = out.pixels();
            for (int i = 0; i < expected.length; i++) {
                if (expected[i] != actual[i]) {
                    int x = i % src.width();
                    int y = i / src.width();
                    throw new AssertionError(String.format(
                        "Pixel mismatch at (%d,%d): expected 0x%08X got 0x%08X",
                        x, y, expected[i], actual[i]));
                }
            }
        }

        @Test @DisplayName("2x2 solid red round-trips")
        void solid2x2() {
            PixelBuffer buf = PixelBuffer.create(2, 2);
            buf.fill(0xFFFF0000);
            assertRoundTrips(buf);
        }

        @Test @DisplayName("4x4 two colors round-trips")
        void twoColors4x4() {
            PixelBuffer buf = PixelBuffer.create(4, 4);
            for (int y = 0; y < 4; y++)
                for (int x = 0; x < 4; x++)
                    buf.setPixel(x, y, ((x + y) & 1) == 0 ? 0xFFFF0000 : 0xFF00FF00);
            assertRoundTrips(buf);
        }

        @Test @DisplayName("8x8 gradient round-trips")
        void gradient8x8() {
            PixelBuffer buf = PixelBuffer.create(8, 8);
            for (int y = 0; y < 8; y++)
                for (int x = 0; x < 8; x++)
                    buf.setPixel(x, y, 0xFF000000 | ((x * 32) << 16) | ((y * 32) << 8));
            assertRoundTrips(buf);
        }

        @Test @DisplayName("32x32 gradient round-trips (the failing case)")
        void gradient32x32() {
            PixelBuffer buf = PixelBuffer.create(32, 32);
            for (int y = 0; y < 32; y++)
                for (int x = 0; x < 32; x++)
                    buf.setPixel(x, y, 0xFF000000 | ((x * 8) << 16) | ((y * 8) << 8) | ((x + y) * 4));
            assertRoundTrips(buf);
        }

        /**
         * Decodes a VP8L payload produced by libwebp (via Pillow). Exercises features our
         * encoder never emits so that the decoder also handles reference-encoder output,
         * not just our own: ColorIndexing sub-bit-packed palette + LZ77 backward refs.
         */
        @Test @DisplayName("decodes libwebp-produced 2x2 solid red (ColorIndexing)")
        void libwebpSolid2x2() {
            byte[] payload = hexToBytes("2f014000000710fd8ffe0722a2ff01");
            PixelBuffer out = VP8LDecoder.decode(payload);
            assertThat(out.width(), is(2));
            assertThat(out.height(), is(2));
            for (int i = 0; i < 4; i++)
                assertThat("pixel " + i, out.pixels()[i], is(0xFFFF0000));
        }

        @Test @DisplayName("decodes libwebp-produced 4x4 noisy pattern")
        void libwebp4x4Noisy() {
            byte[] payload = hexToBytes(
                "2f03c000001f201048de1f3a8df9171014f93fdafc474e0e40204084583467426344ff630c");
            int[] expected = {
                0xFFFF0000, 0xFF00FF00, 0xFF0000FF, 0xFFFFFF00,
                0xFFFFFF00, 0xFFFF0000, 0xFF00FF00, 0xFF0000FF,
                0xFF0000FF, 0xFFFFFF00, 0xFFFF0000, 0xFF00FF00,
                0xFF00FF00, 0xFF0000FF, 0xFFFFFF00, 0xFFFF0000,
            };
            PixelBuffer out = VP8LDecoder.decode(payload);
            assertThat(out.width(), is(4));
            assertThat(out.height(), is(4));
            for (int i = 0; i < 16; i++)
                if (out.pixels()[i] != expected[i])
                    throw new AssertionError(String.format(
                        "libwebp 4x4 pixel idx=%d: expected 0x%08X got 0x%08X",
                        i, expected[i], out.pixels()[i]));
        }

        @Test @DisplayName("decodes libwebp-produced 16x16 gradient (LZ77)")
        void libwebpGradient16x16() {
            byte[] payload = hexToBytes(
                "2f0fc00300b93244f43f7611d1ff0091b64d25dcbfe1c1d381889800ac43fd07");
            PixelBuffer out = VP8LDecoder.decode(payload);
            assertThat(out.width(), is(16));
            assertThat(out.height(), is(16));
            for (int y = 0; y < 16; y++)
                for (int x = 0; x < 16; x++) {
                    int expected = 0xFF000000 | ((x * 16) << 16) | ((y * 16) << 8);
                    int got = out.pixels()[y * 16 + x];
                    if (got != expected)
                        throw new AssertionError(String.format(
                            "libwebp 16x16 pixel (%d,%d): expected 0x%08X got 0x%08X",
                            x, y, expected, got));
                }
        }

        private static byte[] hexToBytes(String hex) {
            byte[] out = new byte[hex.length() / 2];
            for (int i = 0; i < out.length; i++)
                out[i] = (byte) Integer.parseInt(hex.substring(i * 2, i * 2 + 2), 16);
            return out;
        }

    }

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
            // Build the bit-path by DFS over the flat tree.
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

            // Frame layout: 10-byte header + first_partition_size + (possibly) token partition.
            assertThat("total >= header + first partition", payload.length >= 10 + firstPartSize, is(true));
        }

        @Test @DisplayName("2x2 DC-only VP8 decodes in libwebp via Python")
        void lossyVp82x2LibwebpRoundTrip() throws Exception {
            PixelBuffer buf = PixelBuffer.create(2, 2);
            buf.fill(0xFFFF0000);
            byte[] vp8Payload = VP8Encoder.encode(buf, 0.75f);

            ConcurrentList<WebPChunk> chunks = Concurrent.newList();
            chunks.add(RiffContainer.createChunk(WebPChunkType.VP8, vp8Payload));
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
            chunks.add(RiffContainer.createChunk(WebPChunkType.VP8, vp8Payload));
            byte[] riff = RiffContainer.write(chunks);

            verifyDecodesInLibwebp(riff, 16, 16, "dc_only_16x16");
        }

        /**
         * Writes {@code riff} to a temp file, shells out to libwebp via Python, and
         * asserts the returned dimensions match. Skips the test (via an assumption)
         * if Python or the webp package isn't available on this host.
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
         * Attempts to launch a Python interpreter running the given script via {@code -c}.
         * Tries {@code python3}, {@code python}, and {@code py} (the Windows launcher) in
         * turn so the test works across Linux, macOS, and Windows shells.
         *
         * @return the started process, or {@code null} if no candidate launched
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
            // Two stacked 16x16 MBs where the lower MB's best Y16 mode should be V_PRED:
            // the top MB forces a strong vertical gradient that the lower MB can inherit.
            PixelBuffer buf = PixelBuffer.create(16, 32);
            for (int y = 0; y < 32; y++)
                for (int x = 0; x < 16; x++) {
                    int g = (y * 255) / 31;
                    buf.setPixel(x, y, 0xFF000000 | (g << 16) | (g << 8) | g);
                }
            byte[] vp8Payload = VP8Encoder.encode(buf, 1.0f);
            ConcurrentList<WebPChunk> chunks = Concurrent.newList();
            chunks.add(RiffContainer.createChunk(WebPChunkType.VP8, vp8Payload));
            byte[] riff = RiffContainer.write(chunks);

            int[] decoded = decodeWithLibwebp(riff, 16, 32);
            double psnr = computePsnr(buf, decoded, 16, 32);
            if (psnr < 32.0)
                throw new AssertionError(String.format(
                    "vertical stripes PSNR = %.2f dB (expected >= 32 dB with V_PRED picked)", psnr));
        }

        @Test @DisplayName("16x16 gradient reconstructs through libwebp with bounded error at high quality")
        void pixelFidelityGradient() throws Exception {
            PixelBuffer buf = PixelBuffer.create(16, 16);
            for (int y = 0; y < 16; y++)
                for (int x = 0; x < 16; x++)
                    buf.setPixel(x, y, 0xFF000000 | ((x * 16) << 16) | ((y * 16) << 8));

            byte[] vp8Payload = VP8Encoder.encode(buf, 1.0f);
            ConcurrentList<WebPChunk> chunks = Concurrent.newList();
            chunks.add(RiffContainer.createChunk(WebPChunkType.VP8, vp8Payload));
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
            chunks.add(RiffContainer.createChunk(WebPChunkType.VP8, vp8Payload));
            byte[] riff = RiffContainer.write(chunks);

            int[] decoded = decodeWithLibwebp(riff, src.width(), src.height());
            // Sample the center pixel - edges may drift slightly due to chroma sub-sampling.
            int centerX = src.width() / 2;
            int centerY = src.height() / 2;
            int p = decoded[centerY * src.width() + centerX];
            int dr = ((p >> 16) & 0xFF) - expectR;
            int dg = ((p >>  8) & 0xFF) - expectG;
            int db = ((p >>  0) & 0xFF) - expectB;
            if (Math.abs(dr) > tolerance || Math.abs(dg) > tolerance || Math.abs(db) > tolerance)
                throw new AssertionError(String.format(
                    "center pixel mismatch: expected (%d,%d,%d) got (%d,%d,%d), tolerance=%d",
                    expectR, expectG, expectB, (p >> 16) & 0xFF, (p >> 8) & 0xFF, p & 0xFF, tolerance));
        }

        /**
         * Writes {@code riff} to a temp file, shells out to libwebp, and returns
         * the fully decoded ARGB pixel buffer.
         */
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
            // Boundaries: 11, 18 (Cat3 bounds), 19, 34 (Cat4), 35, 66 (Cat5), 67, 2047 (Cat6).
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
            coeffs[0] = 999; // DC placeholder that must not be emitted or overwritten
            coeffs[1] = 3;
            coeffs[5] = -42;
            coeffs[15] = 1;
            short[] decoded = new short[16];
            decoded[0] = 999; // decoder preserves whatever was below 'first'
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
                if (first == 1) coeffs[0] = 0; // AC-only path should have no DC.
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

    // ──── ColorCache ────

    @Nested
    class ColorCacheTests {

        @Test
        void insertAndLookup() {
            ColorCache cache = new ColorCache(4); // 16 entries
            int color = 0xFFAA5500;
            cache.insert(color);

            int index = cache.hashIndex(color);
            assertThat(cache.lookup(index), is(color));
        }

        @Test
        void disabledCacheHasNoSize() {
            ColorCache cache = new ColorCache(0);
            assertThat(cache.isEnabled(), is(false));
            assertThat(cache.size(), is(0));
        }

        @Test
        void enabledCacheHasPowerOfTwoSize() {
            ColorCache cache = new ColorCache(6);
            assertThat(cache.isEnabled(), is(true));
            assertThat(cache.size(), is(64));
        }

    }

}
