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
