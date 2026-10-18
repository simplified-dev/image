package dev.simplified.image.codec.webp.vp8;

import dev.simplified.image.PixelBuffer;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.ArrayList;
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

    @Test
    @DisplayName("NEWMV macroblocks decode in libvpx as they decode here")
    void newMvFramesDecodeAsLibvpxDoes() throws IOException {
        // The two halves of the frame slide opposite ways, so most macroblocks code NEWMV beside
        // neighbours that carry a non-zero best MV, which the MV on the wire is the difference from.
        VP8EncoderSession encoder = new VP8EncoderSession();
        List<byte[]> stream = new ArrayList<>();
        for (int frame = 0; frame < 6; frame++)
            stream.add(encoder.encode(opposed(128, 96, frame), 0.9f, frame == 0));
        assertDecodesAsLibvpx("opposed", stream, 128, 96);
    }

    @Test
    @DisplayName("a frame's probability updates last only that frame in libvpx as here")
    void probabilityUpdatesLastOneFrame() throws IOException {
        // The encoder clears refresh_entropy_probs, so libvpx restores the probabilities a frame
        // started from once the frame is done. This shear sends MV probability updates, and the
        // frame after each reads its MVs under the defaults again.
        VP8EncoderSession encoder = new VP8EncoderSession();
        List<byte[]> stream = new ArrayList<>();
        for (int frame = 0; frame < 6; frame++)
            stream.add(encoder.encode(shear(128, 96, frame), 1.0f, frame == 0));
        assertDecodesAsLibvpx("shear", stream, 128, 96);
    }

    @Test
    @DisplayName("libvpx's SPLITMV residuals decode here as libvpx decodes them")
    void libvpxSplitMvResidualsDecodeAsLibvpxDoes() throws IOException {
        // At a fine quantizer libvpx codes most of these scenes' SPLITMV macroblocks with a
        // residual, which carries no Y2 block: each of its sixteen luma blocks holds its own DC.
        List<PixelBuffer> shear = new ArrayList<>();
        List<PixelBuffer> opposed = new ArrayList<>();
        for (int frame = 0; frame < 10; frame++) {
            shear.add(shear(128, 96, frame));
            opposed.add(opposed(170, 134, frame));
        }
        assertDecodesAsLibvpx("libvpx's shear", LibvpxOracle.encode(shear, 4), 128, 96);
        assertDecodesAsLibvpx("libvpx's opposed", LibvpxOracle.encode(opposed, 4), 170, 134);
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

    /**
     * A shaded checkerboard whose bands of eight rows each slide right one pixel further per frame
     * than the band above.
     */
    private static PixelBuffer shear(int width, int height, int frame) {
        PixelBuffer pixels = PixelBuffer.create(width, height);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int sx = x - (y / 8) * frame;
                double shade = 128 + 60 * Math.sin(sx * 0.21) * Math.cos(y * 0.17) + ((sx / 8 + y / 8) % 2 == 0 ? 25 : -25);
                int c = Math.clamp((int) shade, 0, 255);
                pixels.setPixel(x, y, 0xFF000000 | c << 16 | Math.clamp(255 - c, 0, 255) << 8 | Math.clamp(c / 2 + 60, 0, 255));
            }
        }
        return pixels;
    }

    /**
     * A shaded checkerboard whose upper half slides left two pixels a frame and whose lower half
     * slides right three.
     */
    private static PixelBuffer opposed(int width, int height, int frame) {
        PixelBuffer pixels = PixelBuffer.create(width, height);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int sx = y < height / 2 + 3 ? x - 2 * frame : x + 3 * frame;
                double shade = 128 + 60 * Math.sin(sx * 0.21) * Math.cos(y * 0.17) + ((sx / 8 + y / 8) % 2 == 0 ? 25 : -25);
                int c = Math.clamp((int) shade, 0, 255);
                pixels.setPixel(x, y, 0xFF000000 | c << 16 | Math.clamp(255 - c, 0, 255) << 8 | Math.clamp(c / 2 + 60, 0, 255));
            }
        }
        return pixels;
    }

}
