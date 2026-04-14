package dev.simplified.image.codec.webp;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.Arrays;

/**
 * A parsed RIFF chunk consisting of a type identifier and payload reference.
 * <p>
 * To minimize data copying, the payload is stored as an offset and length
 * into the original byte array rather than as a copied sub-array.
 */
public record WebPChunk(@Nullable Type type,
                        @NotNull String fourCC,
                        byte @NotNull [] source,
                        int payloadOffset,
                        int payloadLength) {

    /**
     * Returns a copy of this chunk's payload bytes.
     *
     * @return a new byte array containing the payload
     */
    public byte @NotNull [] payload() {
        byte[] copy = new byte[this.payloadLength];
        System.arraycopy(this.source, this.payloadOffset, copy, 0, this.payloadLength);
        return copy;
    }

    /**
     * WebP RIFF chunk types identified by their four-character code (FourCC).
     */
    @Getter
    @RequiredArgsConstructor
    public enum Type {

        VP8("VP8 "),
        VP8L("VP8L"),
        VP8X("VP8X"),
        ANIM("ANIM"),
        ANMF("ANMF"),
        ALPH("ALPH"),
        ICCP("ICCP"),
        EXIF("EXIF"),
        XMP("XMP ");

        private final @NotNull String fourCC;

        /**
         * Returns the chunk type matching the given FourCC string.
         *
         * @param fourCC the four-character code
         * @return the matching chunk type, or {@code null} if unrecognized
         */
        public static @Nullable Type of(@NotNull String fourCC) {
            return Arrays.stream(values())
                .filter(type -> type.getFourCC().equals(fourCC))
                .findFirst()
                .orElse(null);
        }

    }

}
