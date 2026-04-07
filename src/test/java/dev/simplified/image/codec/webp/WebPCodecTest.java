package dev.simplified.image.codec.webp;

import dev.simplified.collection.Concurrent;
import dev.simplified.collection.ConcurrentList;
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
