package dev.simplified.image;

import dev.simplified.image.exception.UnsupportedFormatException;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;

/**
 * Supported image formats with magic byte detection and animation capability metadata.
 */
@Getter
@RequiredArgsConstructor
public enum ImageFormat {

    JPEG("jpeg", false),
    PNG("png", false),
    BMP("bmp", false),
    GIF("gif", true),
    WEBP("webp", true),
    TIFF("tiff", true),
    ICO("ico", false),
    TGA("tga", false),
    QOI("qoi", false),
    PNM("pnm", false);

    private final @NotNull String formatName;
    private final boolean supportsAnimation;

    /**
     * Detects the image format of the given byte array by examining magic bytes.
     *
     * @param data the raw image bytes to examine
     * @return the detected image format
     * @throws UnsupportedFormatException if no known format matches the data
     */
    public static @NotNull ImageFormat detect(byte @NotNull [] data) {
        return Arrays.stream(values())
            .filter(format -> format.matches(data))
            .findFirst()
            .orElseThrow(() -> new UnsupportedFormatException("Unable to detect image format from magic bytes"));
    }

    /**
     * Returns whether the given byte array matches this format's magic bytes.
     *
     * @param data the raw image bytes to check
     * @return true if the data starts with this format's magic bytes
     */
    public boolean matches(byte @NotNull [] data) {
        return switch (this) {
            case JPEG -> data.length >= 3
                && (data[0] & 0xFF) == 0xFF
                && (data[1] & 0xFF) == 0xD8
                && (data[2] & 0xFF) == 0xFF;
            case PNG -> data.length >= 4
                && (data[0] & 0xFF) == 0x89
                && data[1] == 0x50
                && data[2] == 0x4E
                && data[3] == 0x47;
            case BMP -> data.length >= 2
                && data[0] == 0x42
                && data[1] == 0x4D;
            case GIF -> data.length >= 3
                && data[0] == 0x47
                && data[1] == 0x49
                && data[2] == 0x46;
            case WEBP -> data.length >= 12
                && data[0] == 0x52   // R
                && data[1] == 0x49   // I
                && data[2] == 0x46   // F
                && data[3] == 0x46   // F
                && data[8] == 0x57   // W
                && data[9] == 0x45   // E
                && data[10] == 0x42  // B
                && data[11] == 0x50; // P
            case TIFF -> data.length >= 4
                && ((data[0] == 0x49 && data[1] == 0x49 && data[2] == 0x2A && data[3] == 0x00)
                    || (data[0] == 0x4D && data[1] == 0x4D && data[2] == 0x00 && data[3] == 0x2A));
            case ICO -> data.length >= 4
                && data[0] == 0x00
                && data[1] == 0x00
                && data[2] == 0x01
                && data[3] == 0x00;
            case TGA -> matchesTgaFooter(data);
            case QOI -> data.length >= 4
                && data[0] == 0x71   // q
                && data[1] == 0x6F   // o
                && data[2] == 0x69   // i
                && data[3] == 0x66;  // f
            case PNM -> data.length >= 2
                && data[0] == 0x50   // P
                && data[1] >= 0x31   // 1
                && data[1] <= 0x36;  // 6
        };
    }

    private static boolean matchesTgaFooter(byte @NotNull [] data) {
        if (data.length < 26) return false;

        int off = data.length - 26;
        return data[off + 8] == 'T' && data[off + 9] == 'R' && data[off + 10] == 'U' && data[off + 11] == 'E'
            && data[off + 12] == 'V' && data[off + 13] == 'I' && data[off + 14] == 'S' && data[off + 15] == 'I'
            && data[off + 16] == 'O' && data[off + 17] == 'N' && data[off + 18] == '-' && data[off + 19] == 'X'
            && data[off + 20] == 'F' && data[off + 21] == 'I' && data[off + 22] == 'L' && data[off + 23] == 'E'
            && data[off + 24] == '.' && data[off + 25] == 0x00;
    }

}
