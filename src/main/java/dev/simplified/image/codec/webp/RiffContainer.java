package dev.simplified.image.codec.webp;

import dev.simplified.collection.Concurrent;
import dev.simplified.collection.ConcurrentList;
import dev.simplified.image.exception.ImageDecodeException;
import dev.simplified.image.exception.ImageEncodeException;
import org.jetbrains.annotations.NotNull;

/**
 * Parses and writes WebP RIFF containers.
 * <p>
 * A WebP file is a RIFF container: {@code RIFF [fileSize] WEBP [chunks...]}.
 * Each chunk is {@code [4-byte FourCC] [4-byte LE size] [payload] [padding]}.
 * Chunks are word-aligned (padded to even byte boundaries).
 */
final class RiffContainer {

    private RiffContainer() { }

    /**
     * Parses a WebP byte array into its constituent RIFF chunks.
     * <p>
     * Chunk payloads are stored as offset/length references into the source
     * array to avoid copying.
     *
     * @param data the raw WebP file bytes
     * @return the parsed chunks
     * @throws ImageDecodeException if the RIFF/WEBP header is invalid
     */
    static @NotNull ConcurrentList<WebPChunk> parse(byte @NotNull [] data) {
        if (data.length < 12)
            throw new ImageDecodeException("Data too short for RIFF header");

        // Validate RIFF header
        if (data[0] != 'R' || data[1] != 'I' || data[2] != 'F' || data[3] != 'F')
            throw new ImageDecodeException("Missing RIFF signature");

        // Validate WEBP identifier
        if (data[8] != 'W' || data[9] != 'E' || data[10] != 'B' || data[11] != 'P')
            throw new ImageDecodeException("Missing WEBP identifier");

        int fileSize = readLE32(data, 4);
        int end = Math.min(fileSize + 8, data.length);
        int offset = 12; // Skip RIFF header + WEBP

        ConcurrentList<WebPChunk> chunks = Concurrent.newList();

        while (offset + 8 <= end) {
            String fourCC = new String(data, offset, 4);
            int chunkSize = readLE32(data, offset + 4);
            int payloadOffset = offset + 8;

            if (payloadOffset + chunkSize > data.length)
                break;

            WebPChunkType type = WebPChunkType.of(fourCC);
            chunks.add(new WebPChunk(type, fourCC, data, payloadOffset, chunkSize));

            // Advance past payload + word-alignment padding
            offset = payloadOffset + chunkSize + (chunkSize & 1);
        }

        return chunks;
    }

    /**
     * Assembles RIFF chunks into a complete WebP file byte array.
     *
     * @param chunks the chunks to write
     * @return the assembled WebP file bytes
     * @throws ImageEncodeException if assembly fails
     */
    static byte @NotNull [] write(@NotNull ConcurrentList<WebPChunk> chunks) {
        // Calculate total size
        int dataSize = 4; // "WEBP"

        for (WebPChunk chunk : chunks) {
            dataSize += 8 + chunk.getPayloadLength();
            if ((chunk.getPayloadLength() & 1) != 0) dataSize++; // padding
        }

        byte[] output = new byte[8 + dataSize]; // RIFF + fileSize + data

        // RIFF header
        output[0] = 'R'; output[1] = 'I'; output[2] = 'F'; output[3] = 'F';
        writeLE32(output, 4, dataSize);
        output[8] = 'W'; output[9] = 'E'; output[10] = 'B'; output[11] = 'P';

        int offset = 12;

        for (WebPChunk chunk : chunks) {
            // FourCC
            byte[] fourCCBytes = chunk.getFourCC().getBytes();
            System.arraycopy(fourCCBytes, 0, output, offset, 4);
            offset += 4;

            // Size
            writeLE32(output, offset, chunk.getPayloadLength());
            offset += 4;

            // Payload
            System.arraycopy(chunk.getSource(), chunk.getPayloadOffset(), output, offset, chunk.getPayloadLength());
            offset += chunk.getPayloadLength();

            // Word-alignment padding
            if ((chunk.getPayloadLength() & 1) != 0)
                output[offset++] = 0;
        }

        return output;
    }

    /**
     * Builds a new {@link WebPChunk} from a type and payload byte array.
     *
     * @param type the chunk type
     * @param payload the chunk payload
     * @return a new chunk wrapping the payload
     */
    static @NotNull WebPChunk createChunk(@NotNull WebPChunkType type, byte @NotNull [] payload) {
        return new WebPChunk(type, type.getFourCC(), payload, 0, payload.length);
    }

    private static int readLE32(byte[] data, int offset) {
        return (data[offset] & 0xFF)
            | ((data[offset + 1] & 0xFF) << 8)
            | ((data[offset + 2] & 0xFF) << 16)
            | ((data[offset + 3] & 0xFF) << 24);
    }

    private static void writeLE32(byte[] data, int offset, int value) {
        data[offset] = (byte) (value & 0xFF);
        data[offset + 1] = (byte) ((value >> 8) & 0xFF);
        data[offset + 2] = (byte) ((value >> 16) & 0xFF);
        data[offset + 3] = (byte) ((value >> 24) & 0xFF);
    }

}
