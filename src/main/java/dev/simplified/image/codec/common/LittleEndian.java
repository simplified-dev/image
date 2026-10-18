package dev.simplified.image.codec.common;

import dev.simplified.annotations.UtilityClass;
import org.jetbrains.annotations.ApiStatus;
import org.jetbrains.annotations.NotNull;

import java.io.DataOutput;
import java.io.IOException;

/**
 * Reads and writes the 16-, 24- and 32-bit little-endian integers that GIF, ICO, TGA and RIFF
 * headers, WebP chunk payloads and VP8 frame headers are built from, through a {@link DataOutput}
 * or in place in an array.
 * <p>
 * A 32-bit read answers the value's bits as an {@code int}, so a value of {@code 2^31} or more reads
 * negative. A write keeps only the low bits its width holds.
 */
@UtilityClass
@ApiStatus.Internal
public class LittleEndian {

    /**
     * Reads the unsigned 16-bit integer stored low byte first at an offset.
     *
     * @param data the bytes to read from
     * @param offset the index of the low byte
     * @return the value, in {@code [0, 65535]}
     */
    public static int readUint16(byte @NotNull [] data, int offset) {
        return (data[offset] & 0xFF) | ((data[offset + 1] & 0xFF) << 8);
    }

    /**
     * Reads the unsigned 24-bit integer stored low byte first at an offset.
     *
     * @param data the bytes to read from
     * @param offset the index of the low byte
     * @return the value, in {@code [0, 16777215]}
     */
    public static int readUint24(byte @NotNull [] data, int offset) {
        return (data[offset] & 0xFF)
            | ((data[offset + 1] & 0xFF) << 8)
            | ((data[offset + 2] & 0xFF) << 16);
    }

    /**
     * Reads the 32-bit integer stored low byte first at an offset.
     *
     * @param data the bytes to read from
     * @param offset the index of the low byte
     * @return the value's 32 bits
     */
    public static int readUint32(byte @NotNull [] data, int offset) {
        return (data[offset] & 0xFF)
            | ((data[offset + 1] & 0xFF) << 8)
            | ((data[offset + 2] & 0xFF) << 16)
            | ((data[offset + 3] & 0xFF) << 24);
    }

    /**
     * Writes the low 16 bits of a value, low byte first.
     *
     * @param out the output to write to
     * @param value the value whose low 16 bits are written
     * @throws IOException if the output fails to write
     */
    public static void writeUint16(@NotNull DataOutput out, int value) throws IOException {
        out.writeByte(value & 0xFF);
        out.writeByte((value >>> 8) & 0xFF);
    }

    /**
     * Writes a 32-bit value, low byte first.
     *
     * @param out the output to write to
     * @param value the value to write
     * @throws IOException if the output fails to write
     */
    public static void writeUint32(@NotNull DataOutput out, int value) throws IOException {
        out.writeByte(value & 0xFF);
        out.writeByte((value >>> 8) & 0xFF);
        out.writeByte((value >>> 16) & 0xFF);
        out.writeByte((value >>> 24) & 0xFF);
    }

    /**
     * Writes the low 16 bits of a value into an array at an offset, low byte first.
     *
     * @param data the array to write into
     * @param offset the index the low byte goes to
     * @param value the value whose low 16 bits are written
     */
    public static void writeUint16(byte @NotNull [] data, int offset, int value) {
        data[offset] = (byte) value;
        data[offset + 1] = (byte) (value >>> 8);
    }

    /**
     * Writes the low 24 bits of a value into an array at an offset, low byte first.
     *
     * @param data the array to write into
     * @param offset the index the low byte goes to
     * @param value the value whose low 24 bits are written
     */
    public static void writeUint24(byte @NotNull [] data, int offset, int value) {
        data[offset] = (byte) value;
        data[offset + 1] = (byte) (value >>> 8);
        data[offset + 2] = (byte) (value >>> 16);
    }

    /**
     * Writes a 32-bit value into an array at an offset, low byte first.
     *
     * @param data the array to write into
     * @param offset the index the low byte goes to
     * @param value the value to write
     */
    public static void writeUint32(byte @NotNull [] data, int offset, int value) {
        data[offset] = (byte) value;
        data[offset + 1] = (byte) (value >>> 8);
        data[offset + 2] = (byte) (value >>> 16);
        data[offset + 3] = (byte) (value >>> 24);
    }

}
