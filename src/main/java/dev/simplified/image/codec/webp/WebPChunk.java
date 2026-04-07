package dev.simplified.image.codec.webp;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

/**
 * A parsed RIFF chunk consisting of a type identifier and payload reference.
 * <p>
 * To minimize data copying, the payload is stored as an offset and length
 * into the original byte array rather than as a copied sub-array.
 */
record WebPChunk(@Nullable WebPChunkType type,
                 @NotNull String fourCC,
                 byte @NotNull [] source,
                 int payloadOffset,
                 int payloadLength) {

    /**
     * Returns a copy of this chunk's payload bytes.
     *
     * @return a new byte array containing the payload
     */
    byte @NotNull [] payload() {
        byte[] copy = new byte[this.payloadLength];
        System.arraycopy(this.source, this.payloadOffset, copy, 0, this.payloadLength);
        return copy;
    }

}
