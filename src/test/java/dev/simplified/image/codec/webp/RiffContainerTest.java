package dev.simplified.image.codec.webp;

import dev.simplified.collection.Concurrent;
import dev.simplified.collection.ConcurrentList;
import org.junit.jupiter.api.Test;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertThrows;

/** Tests for the WebP RIFF container - parsing, assembly, and round-trip. */
public class RiffContainerTest {

    @Test
    void parseRejectsNonRiff() {
        byte[] garbage = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B};
        assertThrows(Exception.class, () -> RiffContainer.parse(garbage));
    }

    @Test
    void writeAndParseRoundTrip() {
        byte[] testPayload = {0x01, 0x02, 0x03, 0x04};
        ConcurrentList<WebPChunk> chunks = Concurrent.newList();
        chunks.add(RiffContainer.createChunk(WebPChunk.Type.VP8L, testPayload));

        byte[] riffBytes = RiffContainer.write(chunks);
        ConcurrentList<WebPChunk> parsed = RiffContainer.parse(riffBytes);

        assertThat(parsed, hasSize(1));
        assertThat(parsed.getFirst().type(), is(WebPChunk.Type.VP8L));
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
        chunks.add(RiffContainer.createChunk(WebPChunk.Type.VP8X, payload1));
        chunks.add(RiffContainer.createChunk(WebPChunk.Type.VP8L, payload2));

        byte[] riffBytes = RiffContainer.write(chunks);
        ConcurrentList<WebPChunk> parsed = RiffContainer.parse(riffBytes);

        assertThat(parsed, hasSize(2));
        assertThat(parsed.get(0).type(), is(WebPChunk.Type.VP8X));
        assertThat(parsed.get(1).type(), is(WebPChunk.Type.VP8L));
        assertThat(parsed.get(1).payloadLength(), is(3));
    }

}
