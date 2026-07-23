package dev.simplified.image.codec.gif;

import dev.simplified.image.ImageData;
import dev.simplified.image.data.AnimatedImageData;
import dev.simplified.image.data.ImageFrame;
import dev.simplified.image.pixel.PixelBuffer;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.hasSize;

/**
 * Pins the delays {@link GifImageWriter} declares, read out of the encoded bytes by walking the
 * GIF block structure. Nothing here asserts through {@link GifImageReader} - a reader applies its
 * own interpretation, so reading back through one could not tell a correct encode from two
 * mistakes that cancel.
 */
class GifDelayTest {

    // ──── writer: cumulative carry ────

    @Test
    @DisplayName("delays already on the centisecond grid are declared unchanged")
    void gridAlignedDelaysAreExact() {
        int[] declared = writeAndReadDelays(frames(4, 50), null);
        assertThat(boxed(declared), contains(5, 5, 5, 5));
        assertThat(sum(declared), equalTo(20));
    }

    @Test
    @DisplayName("30 fps carries the rounding remainder instead of truncating it away")
    void thirtyFpsCarriesRemainder() {
        int[] declared = writeAndReadDelays(frames(60, 33), null);

        // Truncation would emit a flat 3 cs and lose 180 ms over the loop; the carry alternates
        // so the declared total lands exactly on the 1980 ms of intent.
        assertThat(declared.length, equalTo(60));
        assertThat(boxed(Arrays.copyOf(declared, 6)), contains(3, 4, 3, 3, 4, 3));
        assertThat(sum(declared), equalTo(198));
    }

    @Test
    @DisplayName("24 fps and 66 ms strips declare their intended totals")
    void otherCadencesAreExact() {
        assertThat(sum(writeAndReadDelays(frames(50, 42), null)), equalTo(210));
        assertThat(sum(writeAndReadDelays(frames(50, 66), null)), equalTo(330));
    }

    @Test
    @DisplayName("the carry repays its debt over a period longer than a single frame")
    void carrySpansManyFrames() {
        int[] declared = writeAndReadDelays(frames(100, 19), options(DelayFidelity.EXACT));

        // 19 ms is not on the grid at all, so the carry runs a period-10 cycle spending 19 cs per
        // 10 frames rather than drifting by a whole centisecond every frame.
        assertThat(boxed(Arrays.copyOf(declared, 10)), contains(2, 2, 2, 2, 2, 1, 2, 2, 2, 2));
        assertThat(sum(declared), equalTo(190));
    }

    @Test
    @DisplayName("below the honored minimum the carry has nothing left to repay with")
    void carryCannotRepayBelowTheMinimum() {
        int[] declared = writeAndReadDelays(frames(100, 19), options(DelayFidelity.PLAYABLE));

        // Every frame is raised to the 2 cs minimum, so the debt only grows - the declared total
        // overshoots by 5% and no later frame can give the time back. This is the cost of
        // PLAYABLE below 20 ms per frame, and the reason the choice is the caller's.
        assertThat(boxed(Arrays.copyOf(declared, 10)), contains(2, 2, 2, 2, 2, 2, 2, 2, 2, 2));
        assertThat(sum(declared), equalTo(200));
    }

    // ──── writer: fidelity ────

    @Test
    @DisplayName("PLAYABLE never declares the 1 cs delay that players rewrite to 100 ms")
    void playableRaisesSubGridFrames() {
        int[] declared = writeAndReadDelays(subTickFrames(), options(DelayFidelity.PLAYABLE));

        // A 50 ms tick split three ways cannot hold three honored frames - 3 x 2 cs overruns it -
        // so the strip declares 60 ms against 50 ms of intent. Slow, rather than tenfold slow.
        assertThat(boxed(declared), contains(2, 2, 2));
        assertThat(sum(declared), equalTo(6));
    }

    @Test
    @DisplayName("EXACT declares the true timing, 1 cs and all")
    void exactDeclaresTrueTiming() {
        int[] declared = writeAndReadDelays(subTickFrames(), options(DelayFidelity.EXACT));
        assertThat(boxed(declared), contains(2, 1, 2));
        assertThat(sum(declared), equalTo(5));
    }

    @Test
    @DisplayName("the two fidelities agree wherever the minimum cannot bind")
    void fidelitiesAgreeAboveTheMinimum() {
        int[] playable = writeAndReadDelays(frames(20, 33), options(DelayFidelity.PLAYABLE));
        int[] exact = writeAndReadDelays(frames(20, 33), options(DelayFidelity.EXACT));
        assertThat(boxed(playable), equalTo(boxed(exact)));
    }

    @Test
    @DisplayName("PLAYABLE is the default when no options are supplied")
    void playableIsTheDefault() {
        assertThat(
            boxed(writeAndReadDelays(subTickFrames(), null)),
            equalTo(boxed(writeAndReadDelays(subTickFrames(), options(DelayFidelity.PLAYABLE))))
        );
    }

    // ──── reader ────

    @Test
    @DisplayName("a declared delay is decoded as declared")
    void readerReportsDeclaredDelay() {
        ImageData decoded = new GifImageReader().read(new GifImageWriter().write(frames(3, 30)));

        assertThat(decoded.getFrames(), hasSize(3));
        for (ImageFrame frame : decoded.getFrames())
            assertThat(frame.delayMs(), equalTo(30));
    }

    @Test
    @DisplayName("a declared 0 is unspecified timing, not zero time")
    void readerSubstitutesForZeroDelay() {
        byte[] encoded = patchDelays(new GifImageWriter().write(frames(3, 30)), 0);
        ImageData decoded = new GifImageReader().read(encoded);

        // Reading 0 as zero time would sum to a zero-length animation, which freezes the strip on
        // its first frame. Every player supplies a duration of its own instead.
        for (ImageFrame frame : decoded.getFrames())
            assertThat(frame.delayMs(), equalTo(100));

        assertThat(((AnimatedImageData) decoded).getTotalDurationMs(), equalTo(300));
    }

    @Test
    @DisplayName("a fast declared delay survives the round trip instead of being floored")
    void fastDelaysRoundTrip() {
        byte[] encoded = patchDelays(new GifImageWriter().write(frames(2, 30)), 2);
        ImageData decoded = new GifImageReader().read(encoded);

        for (ImageFrame frame : decoded.getFrames())
            assertThat(frame.delayMs(), equalTo(20));
    }

    @Test
    @DisplayName("grid-aligned timing survives a write and read unchanged")
    void writeReadIdentity() {
        for (int delayMs : new int[] { 20, 30, 50, 100 }) {
            ImageData decoded = new GifImageReader().read(new GifImageWriter().write(frames(3, delayMs)));

            for (ImageFrame frame : decoded.getFrames())
                assertThat(frame.delayMs(), equalTo(delayMs));
        }
    }

    // ──── fixtures ────

    private static @NotNull GifWriteOptions options(@NotNull DelayFidelity fidelity) {
        return GifWriteOptions.builder().withDelayFidelity(fidelity).build();
    }

    /**
     * One 50 ms tick split three ways, the leftover millisecond going to the earliest frames.
     */
    private static @NotNull AnimatedImageData subTickFrames() {
        return animation(new int[] { 17, 17, 16 });
    }

    private static @NotNull AnimatedImageData frames(int count, int delayMs) {
        int[] delays = new int[count];
        Arrays.fill(delays, delayMs);
        return animation(delays);
    }

    private static @NotNull AnimatedImageData animation(int @NotNull [] delaysMs) {
        AnimatedImageData.Builder builder = AnimatedImageData.builder();

        for (int index = 0; index < delaysMs.length; index++) {
            // Alternate the fill so consecutive frames differ - the delays are what is under test,
            // and an encoder is free to handle an unchanged frame differently.
            Color color = index % 2 == 0 ? Color.RED : Color.BLUE;
            builder.withFrame(ImageFrame.of(PixelBuffer.wrap(solid(color)), delaysMs[index]));
        }

        return builder.build();
    }

    private static @NotNull BufferedImage solid(@NotNull Color color) {
        BufferedImage image = new BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = image.createGraphics();

        try {
            graphics.setColor(color);
            graphics.fillRect(0, 0, 2, 2);
        } finally {
            graphics.dispose();
        }

        return image;
    }

    // ──── GIF block walking ────

    private static int @NotNull [] writeAndReadDelays(@NotNull ImageData data, @Nullable GifWriteOptions options) {
        byte[] encoded = options == null
            ? new GifImageWriter().write(data)
            : new GifImageWriter().write(data, options);

        List<Integer> offsets = delayOffsets(encoded);
        int[] delays = new int[offsets.size()];

        for (int index = 0; index < offsets.size(); index++) {
            int offset = offsets.get(index);
            delays[index] = (encoded[offset] & 0xFF) | ((encoded[offset + 1] & 0xFF) << 8);
        }

        return delays;
    }

    /**
     * Overwrites every graphic-control delay with {@code delayCs}, producing a file this writer
     * would never emit. It is how the reader gets tested against declarations only foreign
     * encoders produce, such as a bare 0.
     */
    private static byte @NotNull [] patchDelays(byte @NotNull [] gif, int delayCs) {
        byte[] patched = gif.clone();

        for (int offset : delayOffsets(patched)) {
            patched[offset] = (byte) (delayCs & 0xFF);
            patched[offset + 1] = (byte) ((delayCs >> 8) & 0xFF);
        }

        return patched;
    }

    /**
     * Walks the GIF block structure and returns the byte offset of every graphic-control
     * extension's little-endian delay field. Scanning for the extension's byte signature instead
     * would risk matching those same bytes inside compressed image data.
     */
    private static @NotNull List<Integer> delayOffsets(byte @NotNull [] gif) {
        List<Integer> offsets = new ArrayList<>();
        int position = 6; // header: "GIF89a"

        int screenPacked = gif[position + 4] & 0xFF;
        position += 7; // logical screen descriptor
        if ((screenPacked & 0x80) != 0)
            position += colorTableBytes(screenPacked);

        while (position < gif.length) {
            int block = gif[position++] & 0xFF;

            if (block == 0x3B) // trailer
                break;

            if (block == 0x21) { // extension
                int label = gif[position++] & 0xFF;

                // Graphic control: [size=4][packed][delay lo][delay hi][transparent index]
                if (label == 0xF9)
                    offsets.add(position + 2);

                position = skipSubBlocks(gif, position);
                continue;
            }

            if (block == 0x2C) { // image descriptor
                position += 8; // left, top, width, height
                int imagePacked = gif[position++] & 0xFF;
                if ((imagePacked & 0x80) != 0)
                    position += colorTableBytes(imagePacked);

                position++; // LZW minimum code size
                position = skipSubBlocks(gif, position);
                continue;
            }

            throw new IllegalStateException("Unexpected GIF block 0x%02X at %d".formatted(block, position - 1));
        }

        return offsets;
    }

    private static int colorTableBytes(int packed) {
        return 3 * (1 << ((packed & 0x07) + 1));
    }

    private static int skipSubBlocks(byte @NotNull [] gif, int position) {
        while (true) {
            int length = gif[position++] & 0xFF;
            if (length == 0) return position;
            position += length;
        }
    }

    private static @NotNull List<Integer> boxed(int @NotNull [] values) {
        List<Integer> list = new ArrayList<>(values.length);
        for (int value : values) list.add(value);
        return list;
    }

    private static int sum(int @NotNull [] values) {
        int total = 0;
        for (int value : values) total += value;
        return total;
    }

}
