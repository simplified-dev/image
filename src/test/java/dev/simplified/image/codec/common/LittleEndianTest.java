package dev.simplified.image.codec.common;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

/**
 * Pins the little-endian integers the format headers are read and written with: each width is
 * stored low byte first, an array write lays down the bytes a stream write does, and a write keeps
 * only the bits its width holds.
 */
class LittleEndianTest {

    @Test
    @DisplayName("each width reads low byte first")
    void readsLowByteFirst() {
        byte[] data = { 0x01, 0x02, 0x03, (byte) 0x84, 0x05 };
        assertThat(LittleEndian.readUint16(data, 0), is(0x0201));
        assertThat(LittleEndian.readUint24(data, 0), is(0x030201));
        assertThat(LittleEndian.readUint24(data, 2), is(0x058403));
        assertThat(LittleEndian.readUint32(data, 0), is(0x84030201));
    }

    @Test
    @DisplayName("an array write lays down the bytes a stream write does, and reads back")
    void arrayWritesMatchStreamWrites() throws IOException {
        byte[] array = new byte[6];
        LittleEndian.writeUint16(array, 0, 0xBEEF);
        LittleEndian.writeUint32(array, 2, 0xCAFEBABE);

        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        DataOutputStream out = new DataOutputStream(stream);
        LittleEndian.writeUint16(out, 0xBEEF);
        LittleEndian.writeUint32(out, 0xCAFEBABE);

        assertThat(array, is(stream.toByteArray()));
        assertThat(LittleEndian.readUint16(array, 0), is(0xBEEF));
        assertThat(LittleEndian.readUint32(array, 2), is(0xCAFEBABE));

        byte[] wide = new byte[3];
        LittleEndian.writeUint24(wide, 0, 0xABCDEF);
        assertThat(wide, is(new byte[] { (byte) 0xEF, (byte) 0xCD, (byte) 0xAB }));
        assertThat(LittleEndian.readUint24(wide, 0), is(0xABCDEF));
    }

    @Test
    @DisplayName("a write keeps only the bits its width holds and touches nothing past them")
    void writesKeepTheirWidth() {
        byte[] data = new byte[6];
        Arrays.fill(data, (byte) 0x55);
        LittleEndian.writeUint24(data, 1, 0x7F123456);
        assertThat(data, is(new byte[] { 0x55, 0x56, 0x34, 0x12, 0x55, 0x55 }));

        Arrays.fill(data, (byte) 0x55);
        LittleEndian.writeUint16(data, 2, -1);
        assertThat(data, is(new byte[] { 0x55, 0x55, (byte) 0xFF, (byte) 0xFF, 0x55, 0x55 }));
    }

}
