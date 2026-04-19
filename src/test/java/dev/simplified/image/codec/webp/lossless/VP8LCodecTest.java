package dev.simplified.image.codec.webp.lossless;

import dev.simplified.image.pixel.PixelBuffer;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

/**
 * Package-private tests for the VP8L (lossless) codec internals: the bit reader/writer,
 * the encoder/decoder round-trip, the {@link ColorCache}, and cross-compat decoding of
 * libwebp-produced payloads.
 */
public class VP8LCodecTest {

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
            int[] expected = src.data();
            int[] actual = out.data();
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
                assertThat("pixel " + i, out.data()[i], is(0xFFFF0000));
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
                if (out.data()[i] != expected[i])
                    throw new AssertionError(String.format(
                        "libwebp 4x4 pixel idx=%d: expected 0x%08X got 0x%08X",
                        i, expected[i], out.data()[i]));
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
                    int got = out.data()[y * 16 + x];
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

    // ──── Color-Indexing (Palette) Transform ────

    /**
     * Covers the encoder's color-indexing transform path across every sub-bit-packed
     * width (1/2/4/8 bpp) plus the fallback to literal ARGB encoding when the image
     * has > 256 unique colors. Each case must round-trip bit-exact through our own
     * decoder, which is the minimum gate for a lossless transform.
     */
    @Nested
    class PaletteTransformTests {

        private void assertRoundTrips(PixelBuffer src) {
            byte[] payload = VP8LEncoder.encode(src);
            PixelBuffer out = VP8LDecoder.decode(payload);
            assertThat("width", out.width(), is(src.width()));
            assertThat("height", out.height(), is(src.height()));
            int[] expected = src.data();
            int[] actual = out.data();
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

        @Test @DisplayName("1 bpp: 2-color checkerboard packs 8 pixels per byte")
        void palette1Bpp() {
            // Odd width forces the encoder to handle a partial tail byte where the last
            // packed cell only holds 17 / 8 = 2 leftover indices instead of a full 8.
            PixelBuffer buf = PixelBuffer.create(17, 5);
            for (int y = 0; y < 5; y++)
                for (int x = 0; x < 17; x++)
                    buf.setPixel(x, y, ((x + y) & 1) == 0 ? 0xFF336699 : 0xFFCC3366);
            assertRoundTrips(buf);
        }

        @Test @DisplayName("2 bpp: 4-color stripes pack 4 pixels per byte")
        void palette2Bpp() {
            PixelBuffer buf = PixelBuffer.create(13, 7);
            int[] colors = { 0xFF112233, 0xFF445566, 0xFF778899, 0xFFAABBCC };
            for (int y = 0; y < 7; y++)
                for (int x = 0; x < 13; x++)
                    buf.setPixel(x, y, colors[(x + 3 * y) % 4]);
            assertRoundTrips(buf);
        }

        @Test @DisplayName("4 bpp: 16-color pattern packs 2 pixels per byte")
        void palette4Bpp() {
            PixelBuffer buf = PixelBuffer.create(23, 11);
            int[] colors = new int[16];
            for (int i = 0; i < 16; i++)
                colors[i] = 0xFF000000 | (i * 16) << 16 | (i * 8) << 8 | (255 - i * 16);
            for (int y = 0; y < 11; y++)
                for (int x = 0; x < 23; x++)
                    buf.setPixel(x, y, colors[(x * 5 + y * 3) & 0xF]);
            assertRoundTrips(buf);
        }

        @Test @DisplayName("8 bpp: 200-color spread uses one index per pixel")
        void palette8BppLarge() {
            PixelBuffer buf = PixelBuffer.create(40, 40);
            int[] colors = new int[200];
            for (int i = 0; i < 200; i++)
                colors[i] = 0xFF000000 | (i * 83 & 0xFF) << 16 | (i * 37 & 0xFF) << 8 | (i * 11 & 0xFF);
            for (int i = 0; i < 40 * 40; i++)
                buf.data()[i] = colors[i % 200];
            assertRoundTrips(buf);
        }

        @Test @DisplayName("boundary: exactly 256 colors still uses the 8 bpp palette path")
        void paletteExactly256Colors() {
            PixelBuffer buf = PixelBuffer.create(32, 32);
            for (int i = 0; i < 256; i++)
                buf.data()[i] = 0xFF000000 | (i << 16) | (i << 8) | i;
            // Fill the remaining 768 pixels by repeating the 256 unique colors.
            for (int i = 256; i < 32 * 32; i++)
                buf.data()[i] = buf.data()[i % 256];
            assertRoundTrips(buf);
        }

        @Test @DisplayName("fallback: 257 unique colors skip palette and take the literal path")
        void paletteFallbackAt257Colors() {
            // 257 unique colors forces detectPalette to return null - encoder must fall
            // back to the untransformed literal-ARGB path and still round-trip cleanly.
            PixelBuffer buf = PixelBuffer.create(32, 32);
            for (int i = 0; i < 32 * 32; i++) {
                int c = i % 257;
                buf.data()[i] = 0xFF000000 | ((c * 3) & 0xFF) << 16 | ((c * 5) & 0xFF) << 8 | (c & 0xFF);
            }
            assertRoundTrips(buf);
        }

        @Test @DisplayName("1-color solid: 1bpp palette still round-trips")
        void paletteSingleColor() {
            // Single-entry palette exercises the degenerate paletteSize=1 case: simple-mode
            // prefix codes across every alphabet plus a 1x1 palette sub-image.
            PixelBuffer buf = PixelBuffer.create(9, 5);
            buf.fill(0xFF48ACFF);
            assertRoundTrips(buf);
        }

        @Test @DisplayName("size gate: 16-color content compresses tighter than literal path")
        void paletteShrinksOutputOn16ColorContent() {
            // 64x64 with 16 repeating colors. Under palette+4bpp packing the pixel body
            // encodes ~16 pixels per 32-bit green literal; under the old literal-only path
            // every pixel cost ~4 Huffman codes (ARGB). File must land well under 2KB if
            // the palette path is actually firing.
            PixelBuffer buf = PixelBuffer.create(64, 64);
            int[] colors = new int[16];
            for (int i = 0; i < 16; i++)
                colors[i] = 0xFF000000 | (i * 16) << 16 | (i * 8) << 8 | (255 - i * 16);
            for (int i = 0; i < 64 * 64; i++)
                buf.data()[i] = colors[i & 0xF];
            byte[] payload = VP8LEncoder.encode(buf);
            if (payload.length > 2_000)
                throw new AssertionError(String.format(
                    "16-color 64x64 palette-path VP8L payload = %d B, expected <= 2000 B "
                    + "(palette/packing may not be firing)", payload.length));
            // Round-trip sanity as a belt-and-braces assertion.
            assertRoundTrips(buf);
        }

    }

    // ──── Predictor Transform Encoder Integration ────

    /**
     * Covers the encoder's predictor-transform path: per-tile mode selection, mode
     * sub-image emission, forward-residual computation, and the decoder's ability to
     * round-trip the result bit-exact via the inverse pass through all 14 prediction
     * modes plus the (0,0) / first-row / first-column fixed-predictor fallbacks.
     */
    @Nested
    class PredictorTransformTests {

        private void assertRoundTripsWithPredictor(PixelBuffer src) {
            byte[] payload = VP8LEncoder.encode(src, VP8LEncoder.TransformMode.PREDICTOR, 0);
            PixelBuffer out = VP8LDecoder.decode(payload);
            assertThat("width", out.width(), is(src.width()));
            assertThat("height", out.height(), is(src.height()));
            int[] expected = src.data();
            int[] actual = out.data();
            for (int i = 0; i < expected.length; i++) {
                if (expected[i] != actual[i]) {
                    int x = i % src.width();
                    int y = i / src.width();
                    throw new AssertionError(String.format(
                        "pixel mismatch at (%d,%d): expected 0x%08X got 0x%08X",
                        x, y, expected[i], actual[i]));
                }
            }
        }

        @Test @DisplayName("solid color: predictor mode 0/1 covers degenerate no-neighbour pixels")
        void predictorSolidColor() {
            PixelBuffer buf = PixelBuffer.create(24, 24);
            buf.fill(0xFF336699);
            assertRoundTripsWithPredictor(buf);
        }

        @Test @DisplayName("smooth horizontal gradient: best mode should be 1 (left)")
        void predictorHorizontalGradient() {
            PixelBuffer buf = PixelBuffer.create(32, 32);
            for (int y = 0; y < 32; y++)
                for (int x = 0; x < 32; x++)
                    buf.setPixel(x, y, 0xFF000000 | (x * 8) << 16 | (x * 8) << 8 | (x * 8));
            assertRoundTripsWithPredictor(buf);
        }

        @Test @DisplayName("smooth vertical gradient: best mode should be 2 (top)")
        void predictorVerticalGradient() {
            PixelBuffer buf = PixelBuffer.create(32, 32);
            for (int y = 0; y < 32; y++)
                for (int x = 0; x < 32; x++)
                    buf.setPixel(x, y, 0xFF000000 | (y * 8) << 16 | (y * 8) << 8 | (y * 8));
            assertRoundTripsWithPredictor(buf);
        }

        @Test @DisplayName("diagonal gradient: mixed modes across 8x8 tiles")
        void predictorDiagonalGradient() {
            PixelBuffer buf = PixelBuffer.create(48, 48);
            for (int y = 0; y < 48; y++)
                for (int x = 0; x < 48; x++) {
                    int v = ((x + y) * 255) / 94;
                    buf.setPixel(x, y, 0xFF000000 | (v << 16) | (v << 8) | v);
                }
            assertRoundTripsWithPredictor(buf);
        }

        @Test @DisplayName("noisy image: residuals with varied patterns")
        void predictorNoisyImage() {
            PixelBuffer buf = PixelBuffer.create(40, 40);
            java.util.Random rng = new java.util.Random(42);
            for (int i = 0; i < buf.data().length; i++)
                buf.data()[i] = 0xFF000000 | rng.nextInt(0x01000000);
            assertRoundTripsWithPredictor(buf);
        }

        @Test @DisplayName("non-tile-aligned dimensions: partial tiles at the right/bottom")
        void predictorNonAlignedDimensions() {
            // 23x17 doesn't divide 8; predictor's last tile column is 7 wide and last row
            // is 1 tall. Tests the boundary loop clipping in scoreTileUnderMode.
            PixelBuffer buf = PixelBuffer.create(23, 17);
            for (int y = 0; y < 17; y++)
                for (int x = 0; x < 23; x++)
                    buf.setPixel(x, y, 0xFF000000 | ((x * 5 + y * 7) & 0xFF) << 16
                        | ((x * 11) & 0xFF) << 8 | ((y * 13) & 0xFF));
            assertRoundTripsWithPredictor(buf);
        }

        @Test @DisplayName("1x1 image: predictor degenerates cleanly")
        void predictor1x1() {
            PixelBuffer buf = PixelBuffer.create(1, 1);
            buf.setPixel(0, 0, 0xFF0066CC);
            assertRoundTripsWithPredictor(buf);
        }

        @Test @DisplayName("256x256 gradient: predictor shrinks output vs literal path")
        void predictorShrinksSmoothContent() {
            // Large smooth gradient - spatial prediction should produce tiny residuals
            // so the predictor-path output must be meaningfully smaller than the
            // literal-path baseline.
            PixelBuffer buf = PixelBuffer.create(256, 256);
            for (int y = 0; y < 256; y++)
                for (int x = 0; x < 256; x++) {
                    int r = x;
                    int g = y;
                    int b = (x + y) / 2;
                    buf.setPixel(x, y, 0xFF000000 | (r << 16) | (g << 8) | b);
                }
            byte[] literalOnly = VP8LEncoder.encode(buf, VP8LEncoder.TransformMode.NONE, 0);
            byte[] predictor = VP8LEncoder.encode(buf, VP8LEncoder.TransformMode.PREDICTOR, 0);
            if (predictor.length >= literalOnly.length)
                throw new AssertionError("predictor did not shrink smooth gradient: predictor="
                    + predictor.length + " B, literal=" + literalOnly.length + " B");
            assertRoundTripsWithPredictor(buf);
        }

        @Test @DisplayName("public encode() picks predictor on smooth content")
        void publicEncoderPicksPredictor() {
            // 128x128 smooth gradient with > 256 unique colors. Palette not applicable;
            // predictor should beat both literal and color-cache variants.
            PixelBuffer buf = PixelBuffer.create(128, 128);
            for (int y = 0; y < 128; y++)
                for (int x = 0; x < 128; x++) {
                    int r = x * 2;
                    int g = y * 2;
                    int b = (x + y);
                    buf.setPixel(x, y, 0xFF000000 | (r << 16) | (g << 8) | b);
                }
            byte[] withPredictor = VP8LEncoder.encode(buf, VP8LEncoder.TransformMode.PREDICTOR, 0);
            byte[] best = VP8LEncoder.encode(buf);
            if (best.length > withPredictor.length)
                throw new AssertionError("public encode() ignored a smaller predictor variant: "
                    + "best=" + best.length + " B, predictor=" + withPredictor.length + " B");
        }

    }

    // ──── Meta-Huffman Multi-Group Tests ────

    /**
     * Covers the encoder's meta-Huffman (multi-Huffman-group) path and the matching
     * decoder support. Each case must round-trip bit-exact through our own decoder;
     * the meta-prefix sub-image + per-group Huffman declarations + per-tile symbol
     * emission are verified end-to-end.
     */
    @Nested
    class MetaHuffmanTests {

        private void assertRoundTripsMetaHuffman(PixelBuffer src, int tileBits, int cacheBits) {
            byte[] payload = VP8LEncoder.encode(src, VP8LEncoder.TransformMode.NONE, cacheBits, tileBits);
            PixelBuffer out = VP8LDecoder.decode(payload);
            assertThat("width", out.width(), is(src.width()));
            assertThat("height", out.height(), is(src.height()));
            int[] expected = src.data();
            int[] actual = out.data();
            for (int i = 0; i < expected.length; i++) {
                if (expected[i] != actual[i]) {
                    int x = i % src.width();
                    int y = i / src.width();
                    throw new AssertionError(String.format(
                        "meta-Huffman tileBits=%d cache=%d mismatch at (%d,%d): "
                        + "expected 0x%08X got 0x%08X",
                        tileBits, cacheBits, x, y, expected[i], actual[i]));
                }
            }
        }

        @Test @DisplayName("small image + single tile: meta-prefix with 1x1 prefix image")
        void singleTile() {
            // tileBits=9 (512 tile) on 32x32 image -> 1x1 prefix image, 1 group.
            PixelBuffer buf = PixelBuffer.create(32, 32);
            for (int i = 0; i < 1024; i++) buf.data()[i] = 0xFF000000 | (i & 0xFF) << 16 | (i & 0xFF);
            assertRoundTripsMetaHuffman(buf, 9, 0);
        }

        @Test @DisplayName("4 tiles: tileBits=5 on 64x64 image splits into 2x2")
        void fourTiles() {
            // 64x64 with tileBits=5 (32-px tiles) -> 2x2 = 4 groups.
            PixelBuffer buf = PixelBuffer.create(64, 64);
            // Quadrant-patterned content so each tile has distinct statistics.
            for (int y = 0; y < 64; y++)
                for (int x = 0; x < 64; x++) {
                    int q = ((y >= 32) ? 2 : 0) + ((x >= 32) ? 1 : 0);
                    int r = (q == 0) ? 240 : (q == 1) ? 20 : (q == 2) ? 120 : 60;
                    int g = (q == 0) ? 20 : (q == 1) ? 240 : (q == 2) ? 60 : 120;
                    int b = (q == 0) ? 60 : (q == 1) ? 120 : (q == 2) ? 240 : 20;
                    buf.setPixel(x, y, 0xFF000000 | (r << 16) | (g << 8) | b);
                }
            assertRoundTripsMetaHuffman(buf, 5, 0);
        }

        @Test @DisplayName("non-aligned dimensions: right/bottom tiles partially clipped")
        void nonAlignedDimensions() {
            // 100x60 with tileBits=5 (32-px tiles) -> ceil(100/32)=4, ceil(60/32)=2 = 8 groups.
            // Rightmost column is 100-96=4 px wide, bottom row is 60-32=28 px tall.
            PixelBuffer buf = PixelBuffer.create(100, 60);
            for (int y = 0; y < 60; y++)
                for (int x = 0; x < 100; x++)
                    buf.setPixel(x, y, 0xFF000000 | ((x * 3 + y * 5) & 0xFF) << 16
                        | ((x * 7) & 0xFF) << 8 | ((y * 11) & 0xFF));
            assertRoundTripsMetaHuffman(buf, 5, 0);
        }

        @Test @DisplayName("meta-Huffman + color cache: both features coexist")
        void metaHuffmanWithCache() {
            // 4-group image + 10-bit cache. Cache hits are routed through the per-group
            // green alphabet's cache-index region (positions 280..1303), so each group
            // has its own set of cache-index counts and Huffman codes.
            PixelBuffer buf = PixelBuffer.create(64, 64);
            for (int i = 0; i < 64 * 64; i++) {
                int c = i % 300;
                buf.data()[i] = 0xFF000000 | ((c * 7) & 0xFF) << 16 | ((c * 11) & 0xFF) << 8 | ((c * 13) & 0xFF);
            }
            assertRoundTripsMetaHuffman(buf, 5, 10);
        }

        @Test @DisplayName("tile boundary crossing LZ77 match: decoder reads len/dist from start-tile's trees")
        void lz77MatchCrossesTileBoundary() {
            // 64x64 with tileBits=5. Match spans horizontally across the x=32 tile
            // boundary - decoder must read all the match-related symbols from the
            // starting pixel's group and copy pixels through the boundary without
            // further Huffman reads. Regression gate for the pos/(x,y) advance code
            // in decodePixels + the encoder's token-group assignment.
            PixelBuffer buf = PixelBuffer.create(64, 64);
            for (int y = 0; y < 64; y++)
                for (int x = 0; x < 64; x++) {
                    // Repeating horizontal pattern so LZ77 can match across boundaries.
                    buf.setPixel(x, y, 0xFF000000 | ((x % 7) << 20) | ((y % 5) << 12));
                }
            assertRoundTripsMetaHuffman(buf, 5, 0);
        }

    }

    // ──── Subtract-Green + Cross-Color Transform Tests ────

    /**
     * Covers the encoder's SUBTRACT_GREEN and COLOR_TRANSFORM (cross-color) integration.
     * Each case must round-trip bit-exact through our own decoder, which gates the
     * lossless guarantee and confirms the forward/inverse sign semantics match
     * libwebp's {@code TransformColor} / {@code ColorSpaceInverseTransform}.
     */
    @Nested
    class SubtractGreenAndCrossColorTests {

        private void assertRoundTripsWithMode(PixelBuffer src, VP8LEncoder.TransformMode mode) {
            byte[] payload = VP8LEncoder.encode(src, mode, 0);
            PixelBuffer out = VP8LDecoder.decode(payload);
            assertThat("width", out.width(), is(src.width()));
            assertThat("height", out.height(), is(src.height()));
            int[] expected = src.data();
            int[] actual = out.data();
            for (int i = 0; i < expected.length; i++) {
                if (expected[i] != actual[i]) {
                    int x = i % src.width();
                    int y = i / src.width();
                    throw new AssertionError(String.format(
                        "%s mismatch at (%d,%d): expected 0x%08X got 0x%08X",
                        mode, x, y, expected[i], actual[i]));
                }
            }
        }

        @Test @DisplayName("subtract-green: smooth natural-image gradient")
        void subtractGreenSmoothContent() {
            PixelBuffer buf = PixelBuffer.create(64, 64);
            for (int y = 0; y < 64; y++)
                for (int x = 0; x < 64; x++) {
                    int v = (x + y) * 2;
                    buf.setPixel(x, y, 0xFF000000 | ((v + 50) & 0xFF) << 16 | (v & 0xFF) << 8 | ((v - 20) & 0xFF));
                }
            assertRoundTripsWithMode(buf, VP8LEncoder.TransformMode.SUBTRACT_GREEN);
        }

        @Test @DisplayName("subtract-green: high-contrast edges (residuals can hit 255-adjacent values)")
        void subtractGreenHighContrast() {
            PixelBuffer buf = PixelBuffer.create(32, 32);
            for (int y = 0; y < 32; y++)
                for (int x = 0; x < 32; x++)
                    buf.setPixel(x, y, (x & 1) == 0 ? 0xFFFF00FF : 0xFF00FF00);
            assertRoundTripsWithMode(buf, VP8LEncoder.TransformMode.SUBTRACT_GREEN);
        }

        @Test @DisplayName("subtract-green: alpha-varying content leaves alpha channel unchanged")
        void subtractGreenAlphaPreserved() {
            PixelBuffer buf = PixelBuffer.create(16, 16);
            for (int y = 0; y < 16; y++)
                for (int x = 0; x < 16; x++) {
                    int a = ((x + y) * 255 / 30) & 0xFF;
                    int v = (x * 7 + y * 13) & 0xFF;
                    buf.setPixel(x, y, (a << 24) | (v << 16) | (v << 8) | ((255 - v) & 0xFF));
                }
            assertRoundTripsWithMode(buf, VP8LEncoder.TransformMode.SUBTRACT_GREEN);
        }

        @Test @DisplayName("cross-color: natural-image-style R=G+X, B=G+Y correlations")
        void crossColorNaturalCorrelations() {
            PixelBuffer buf = PixelBuffer.create(48, 48);
            for (int y = 0; y < 48; y++)
                for (int x = 0; x < 48; x++) {
                    int g = (x * 5 + y * 3) & 0xFF;
                    // R, B are close to G + small deltas - textbook case for cross-color.
                    int r = Math.clamp(g + 20, 0, 255);
                    int b = Math.clamp(g - 15, 0, 255);
                    buf.setPixel(x, y, 0xFF000000 | (r << 16) | (g << 8) | b);
                }
            assertRoundTripsWithMode(buf, VP8LEncoder.TransformMode.CROSS_COLOR);
        }

        @Test @DisplayName("cross-color: saturated RGB values (>= 128 triggers sign edge cases)")
        void crossColorSaturatedChannels() {
            // Half the pixels have channel values >= 128 so the signed-vs-unsigned sign
            // bug that used to lurk in ColorXform would surface here as a round-trip
            // mismatch. Regression gate for the sign-semantics fix.
            PixelBuffer buf = PixelBuffer.create(32, 32);
            for (int y = 0; y < 32; y++)
                for (int x = 0; x < 32; x++) {
                    int r = 128 + ((x * 3) & 0x7F);
                    int g = 128 + ((y * 5) & 0x7F);
                    int b = 128 + (((x + y) * 7) & 0x7F);
                    buf.setPixel(x, y, 0xFF000000 | (r << 16) | (g << 8) | b);
                }
            assertRoundTripsWithMode(buf, VP8LEncoder.TransformMode.CROSS_COLOR);
        }

        @Test @DisplayName("cross-color: random content (near-zero best coefficients)")
        void crossColorRandomContent() {
            PixelBuffer buf = PixelBuffer.create(40, 40);
            java.util.Random rng = new java.util.Random(17);
            for (int i = 0; i < buf.data().length; i++)
                buf.data()[i] = 0xFF000000 | (rng.nextInt(0x01000000));
            assertRoundTripsWithMode(buf, VP8LEncoder.TransformMode.CROSS_COLOR);
        }

        @Test @DisplayName("cross-color: 1x1 image degenerates cleanly")
        void crossColor1x1() {
            PixelBuffer buf = PixelBuffer.create(1, 1);
            buf.setPixel(0, 0, 0xFFD24422);
            assertRoundTripsWithMode(buf, VP8LEncoder.TransformMode.CROSS_COLOR);
        }

        @Test @DisplayName("subtract-green: shrinks output on natural-image content vs literal path")
        void subtractGreenShrinksNaturalImage() {
            // Smooth 128x128 natural-image-style gradient where R, G, B all increase
            // together. R - G and B - G produce tight small-magnitude residuals so the
            // output should be strictly smaller than the literal-path baseline.
            PixelBuffer buf = PixelBuffer.create(128, 128);
            for (int y = 0; y < 128; y++)
                for (int x = 0; x < 128; x++) {
                    int g = Math.clamp(x + y, 0, 255);
                    int r = Math.clamp(g + 20 - ((x ^ y) & 0xF), 0, 255);
                    int b = Math.clamp(g - 15 + ((x | y) & 0xF), 0, 255);
                    buf.setPixel(x, y, 0xFF000000 | (r << 16) | (g << 8) | b);
                }
            byte[] lit = VP8LEncoder.encode(buf, VP8LEncoder.TransformMode.NONE, 0);
            byte[] sg  = VP8LEncoder.encode(buf, VP8LEncoder.TransformMode.SUBTRACT_GREEN, 0);
            if (sg.length >= lit.length)
                throw new AssertionError("subtract-green did not shrink natural-image content: "
                    + "SG=" + sg.length + " B, literal=" + lit.length + " B");
        }

    }

    // ──── Color Cache Encoder Integration ────

    /**
     * Covers the encoder's color-cache integration: the cache-index symbol path in the
     * green alphabet must round-trip through our own decoder at every cache size, and
     * the cache-index emit shortcut must only fire on true hash slot matches.
     */
    @Nested
    class ColorCacheEncodeTests {

        private void assertRoundTripsWithCacheBits(PixelBuffer src, int cacheBits) {
            byte[] payload = VP8LEncoder.encode(src, false, cacheBits);
            PixelBuffer out = VP8LDecoder.decode(payload);
            assertThat("width", out.width(), is(src.width()));
            assertThat("height", out.height(), is(src.height()));
            int[] expected = src.data();
            int[] actual = out.data();
            for (int i = 0; i < expected.length; i++) {
                if (expected[i] != actual[i]) {
                    int x = i % src.width();
                    int y = i / src.width();
                    throw new AssertionError(String.format(
                        "cacheBits=%d pixel mismatch at (%d,%d): expected 0x%08X got 0x%08X",
                        cacheBits, x, y, expected[i], actual[i]));
                }
            }
        }

        @Test @DisplayName("flat uniform Huffman: all-256 red literals (degenerate-CLC path)")
        void degenerateCLCPath() {
            // Synthesise a 16x16 image whose red-channel histogram is dead flat - every
            // value 0..255 appears exactly once across 256 pixels - so the red alphabet's
            // code lengths collapse to a single value (8) across all 256 symbols. The CLC
            // for red then has only one used length, which the decoder treats as a
            // degenerate zero-bit-per-read tree; if the encoder forgets the matching
            // zero-bit emission it writes 256 stray bits and desyncs the rest of the
            // stream. Regression gate for the fix introduced alongside color-cache.
            PixelBuffer buf = PixelBuffer.create(16, 16);
            for (int i = 0; i < 256; i++)
                buf.data()[i] = 0xFF000000 | (i << 16) | ((i * 7 & 0xFF) << 8) | (i * 11 & 0xFF);
            assertRoundTripsWithCacheBits(buf, 0);
            assertRoundTripsWithCacheBits(buf, 1);
            assertRoundTripsWithCacheBits(buf, 10);
        }

        @Test @DisplayName("cache off vs on: tiny 3-color image round-trips both ways")
        void roundTripCacheOffAndOn() {
            // Small synthetic with >256 unique colors so the palette path is skipped.
            // Cache should hit heavily because the histogram is tight: 257 distinct
            // colors cycling across 16384 pixels.
            PixelBuffer buf = PixelBuffer.create(128, 128);
            for (int i = 0; i < buf.data().length; i++) {
                int c = i % 257;
                buf.data()[i] = 0xFF000000 | ((c * 3) & 0xFF) << 16 | ((c * 5) & 0xFF) << 8 | (c & 0xFF);
            }
            assertRoundTripsWithCacheBits(buf, 0);
            assertRoundTripsWithCacheBits(buf, 10);
        }

        @Test @DisplayName("cache=8 round-trips a 300-color mandala")
        void cache8Roundtrip() {
            PixelBuffer buf = PixelBuffer.create(64, 64);
            for (int y = 0; y < 64; y++)
                for (int x = 0; x < 64; x++) {
                    int v = (x * 7 + y * 13) % 300;
                    buf.setPixel(x, y,
                        0xFF000000 | ((v & 0xFF) << 16) | (((v * 3) & 0xFF) << 8) | ((v * 5) & 0xFF));
                }
            assertRoundTripsWithCacheBits(buf, 8);
        }

        @Test @DisplayName("cache=1 smallest valid cache still round-trips")
        void cache1Roundtrip() {
            // 2-entry cache is the smallest enabled size. Cache-index symbols take only
            // 1 bit from that alphabet (after Huffman), but the cache-enabled header
            // overhead is still paid. Mostly tests the edge where the cache alphabet
            // grows the green alphabet by just 2 symbols.
            PixelBuffer buf = PixelBuffer.create(32, 32);
            for (int i = 0; i < buf.data().length; i++) {
                int c = i % 300;
                buf.data()[i] = 0xFF000000 | ((c * 7) & 0xFF) << 16 | ((c * 11) & 0xFF) << 8 | ((c * 13) & 0xFF);
            }
            assertRoundTripsWithCacheBits(buf, 1);
        }

        @Test @DisplayName("cache=11 largest valid cache still round-trips")
        void cache11Roundtrip() {
            // 2048-entry cache is the spec's maximum. Stresses the upper green-alphabet
            // range (positions 280..2327) during freq-count, prefix-code build, and emit.
            PixelBuffer buf = PixelBuffer.create(64, 64);
            for (int i = 0; i < buf.data().length; i++) {
                int c = i % 500;
                buf.data()[i] = 0xFF000000 | ((c * 7) & 0xFF) << 16 | ((c * 11) & 0xFF) << 8 | ((c * 13) & 0xFF);
            }
            assertRoundTripsWithCacheBits(buf, 11);
        }

        @Test @DisplayName("public encode() picks the smaller of cache-off / cache-10 / palette variants")
        void publicEncoderPicksSmaller() {
            // 128x128 recurring content where color cache must beat literal-only. The
            // public encode() has to choose cache-on as the smallest variant.
            PixelBuffer buf = PixelBuffer.create(128, 128);
            for (int i = 0; i < buf.data().length; i++) {
                int c = i % 300;
                buf.data()[i] = 0xFF000000 | ((c * 7) & 0xFF) << 16 | ((c * 11) & 0xFF) << 8 | ((c * 13) & 0xFF);
            }
            byte[] cacheOff = VP8LEncoder.encode(buf, false, 0);
            byte[] best = VP8LEncoder.encode(buf);
            if (best.length > cacheOff.length)
                throw new AssertionError("public encode() grew output: best=" + best.length
                    + " > cacheOff=" + cacheOff.length + " (A/B selector is broken)");
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
