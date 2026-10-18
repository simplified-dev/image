package dev.simplified.image.codec.webp.vp8;

import dev.simplified.image.PixelBuffer;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.List;

/**
 * Pins this library's VP8 inter frames to libvpx's, sample for sample. A stream is decoded by this
 * library and by libvpx, and every frame's planes must agree: a mode probability, a motion vector or
 * a token context read differently shows as a sample that differs, or as a frame libvpx refuses.
 */
class VP8LibvpxParityTest {

    @Test
    @DisplayName("a frame of SPLITMV macroblocks decodes in libvpx as it decodes here")
    void splitMvFrameDecodesAsLibvpxDoes() throws IOException {
        // Every macroblock after the first reads its mode with SPLITMV neighbours, whose count
        // picks the probability the SPLITMV branch of the mode tree is coded under.
        PixelBuffer src = gradient(64, 48);
        VP8EncoderSession encoder = new VP8EncoderSession();
        List<byte[]> stream = List.of(
            encoder.encode(src, 1.0f, true),
            VP8Encoder.encodeWithSplitMv(src, 1.0f, encoder),
            VP8Encoder.encodeWithSplitMv(src, 1.0f, encoder)
        );
        assertDecodesAsLibvpx("forced SPLITMV", stream, 64, 48);
    }

    /**
     * Decodes a stream through this library and through libvpx, and fails on the first sample where
     * they part.
     */
    private static void assertDecodesAsLibvpx(String what, List<byte[]> stream, int width, int height) throws IOException {
        List<LibvpxOracle.Frame> expected = LibvpxOracle.decode(stream, width, height);
        VP8DecoderSession decoder = new VP8DecoderSession();
        int chromaW = (width + 1) / 2;
        int chromaH = (height + 1) / 2;
        for (int f = 0; f < stream.size(); f++) {
            decoder.decode(stream.get(f));
            LibvpxOracle.Frame frame = expected.get(f);
            String at = what + " frame " + f + " ";
            assertPlane(at + "Y", frame.y(), width, height, decoder.refY, decoder.refLumaStride);
            assertPlane(at + "U", frame.u(), chromaW, chromaH, decoder.refU, decoder.refChromaStride);
            assertPlane(at + "V", frame.v(), chromaW, chromaH, decoder.refV, decoder.refChromaStride);
        }
    }

    private static void assertPlane(String what, byte[] libvpx, int width, int height, short[] ours, int stride) {
        int differing = 0;
        int first = -1;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (ours[y * stride + x] != (libvpx[y * width + x] & 0xFF)) {
                    if (differing == 0)
                        first = y * width + x;
                    differing++;
                }
            }
        }
        if (differing > 0)
            throw new AssertionError(String.format(
                "%s: %d of %d samples differ from libvpx's decode, the first at (%d, %d), %d where libvpx has %d",
                what, differing, width * height, first % width, first / width,
                ours[first / width * stride + first % width], libvpx[first] & 0xFF));
    }

    private static PixelBuffer gradient(int width, int height) {
        PixelBuffer pixels = PixelBuffer.create(width, height);
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                pixels.setPixel(x, y, 0xFF000000 | (x * 4) << 16 | (y * 5) << 8 | (x + y) * 2);
        return pixels;
    }

}
