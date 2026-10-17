package dev.simplified.image.codec.gif;

import dev.simplified.image.AnimatedImageData;
import dev.simplified.image.ImageData;
import dev.simplified.image.PixelBuffer;
import dev.simplified.image.StaticImageData;
import dev.simplified.image.exception.ImageEncodeException;
import dev.simplified.image.frame.ImageFrame;
import dev.simplified.image.frame.LoopCount;
import dev.simplified.image.option.GifWriteOptions;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.instanceOf;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.sameInstance;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Pins the loop count {@link GifImageWriter} declares, read out of the NETSCAPE2.0 application
 * extension's bytes, and the one {@link GifImageReader} reads back.
 * <p>
 * The extension counts repeats after the first play, and {@code 0} repeats forever; a GIF that
 * carries none plays once. An ANIMEXTS1.0 extension, which some encoders write in its place, is
 * read the same way.
 */
class GifLoopCountTest {

    private static final int RED = 0xFFFF0000;
    private static final int BLUE = 0xFF0000FF;

    /** The side of a fixture frame. */
    private static final int SIZE = 8;

    // ──── what the writer declares ────

    @Test
    @DisplayName("a count that never repeats is written as no NETSCAPE2.0 extension")
    void neverRepeatingWritesNoExtension() {
        assertThat(GifBlocks.netscapeLoopCounts(write(animation(LoopCount.Preset.NONE), null)), is(empty()));
    }

    @Test
    @DisplayName("a finite count is written as its repeats, and an endless one as 0")
    void repeatsAreWritten() {
        assertThat(GifBlocks.netscapeLoopCounts(write(animation(LoopCount.Preset.ONCE), null)), contains(1));
        assertThat(GifBlocks.netscapeLoopCounts(write(animation(LoopCount.of(4)), null)), contains(4));
        assertThat(GifBlocks.netscapeLoopCounts(write(animation(LoopCount.of(65535)), null)), contains(65535));
        assertThat(GifBlocks.netscapeLoopCounts(write(animation(LoopCount.Preset.INFINITE), null)), contains(0));
    }

    @Test
    @DisplayName("a count beyond the extension's 16 bits is refused")
    void repeatsBeyondSixteenBitsAreRefused() {
        assertThrows(ImageEncodeException.class, () -> write(animation(LoopCount.of(65536)), null));
    }

    @Test
    @DisplayName("a still image is written with no NETSCAPE2.0 extension, whatever count the options carry")
    void stillImageWritesNoExtension() {
        GifWriteOptions options = GifWriteOptions.builder().withLoopCount(LoopCount.Preset.INFINITE).build();

        assertThat(GifBlocks.netscapeLoopCounts(write(StaticImageData.of(fill(RED)), options)), is(empty()));
    }

    @Test
    @DisplayName("options that set no count declare the image's")
    void optionsWithoutACountDeclareTheImages() {
        GifWriteOptions options = GifWriteOptions.builder().isTransparent().build();

        assertThat(GifBlocks.netscapeLoopCounts(write(animation(LoopCount.Preset.NONE), options)), is(empty()));
        assertThat(GifBlocks.netscapeLoopCounts(write(animation(LoopCount.of(4)), options)), contains(4));
        assertThat(GifBlocks.netscapeLoopCounts(write(animation(LoopCount.Preset.INFINITE), options)), contains(0));
    }

    @Test
    @DisplayName("a count in the options is declared over the image's")
    void optionsCountWins() {
        GifWriteOptions options = GifWriteOptions.builder().withLoopCount(LoopCount.Preset.ONCE).build();

        assertThat(GifBlocks.netscapeLoopCounts(write(animation(LoopCount.Preset.INFINITE), options)), contains(1));
        assertThat(GifBlocks.netscapeLoopCounts(write(animation(LoopCount.Preset.NONE), options)), contains(1));
    }

    // ──── what the reader reads ────

    @Test
    @DisplayName("each count reads back as the count written")
    void eachCountRoundTrips() {
        for (LoopCount count : List.of(LoopCount.Preset.NONE, LoopCount.Preset.ONCE, LoopCount.of(4), LoopCount.Preset.INFINITE))
            assertThat(read(write(animation(count), null)).getLoopCount(), equalTo(count));
    }

    @Test
    @DisplayName("a GIF with no NETSCAPE2.0 extension reads as playing once")
    void noExtensionReadsAsPlayingOnce() {
        byte[] gif = write(animation(LoopCount.Preset.NONE), null);

        assertThat(GifBlocks.netscapeLoopCounts(gif), is(empty()));
        assertThat(read(gif).getLoopCount(), sameInstance(LoopCount.Preset.NONE));
    }

    @Test
    @DisplayName("an ANIMEXTS1.0 extension is read as the NETSCAPE2.0 one it stands in for")
    void animextsReadsAsNetscape() {
        for (LoopCount count : List.of(LoopCount.Preset.ONCE, LoopCount.of(2), LoopCount.Preset.INFINITE)) {
            byte[] gif = GifBlocks.withNetscapeRenamed(write(animation(count), null), "ANIMEXTS1.0");

            assertThat(GifBlocks.netscapeLoopCounts(gif), is(empty()));
            assertThat(read(gif).getLoopCount(), equalTo(count));
        }
    }

    @Test
    @DisplayName("a NETSCAPE2.0 extension whose sub-block is not the loop count declares no count")
    void otherSubBlockDeclaresNoCount() {
        byte[] gif = GifBlocks.withNetscapeSubBlockId(write(animation(LoopCount.of(4)), null), 2);

        assertThat(read(gif).getLoopCount(), sameInstance(LoopCount.Preset.NONE));
    }

    @Test
    @DisplayName("a GIF that plays once is written again with no NETSCAPE2.0 extension")
    void playOnceSurvivesAReEncode() {
        byte[] original = write(animation(LoopCount.Preset.NONE), null);
        byte[] rewritten = write(read(original), null);

        assertThat(GifBlocks.netscapeLoopCounts(rewritten), is(empty()));
    }

    // ──── fixtures ────

    /**
     * Two frames of different colours, {@link #SIZE} pixels on a side, repeating {@code loopCount}
     * times after the first play.
     */
    private static @NotNull AnimatedImageData animation(@NotNull LoopCount loopCount) {
        return AnimatedImageData.builder()
            .withFrame(ImageFrame.of(fill(RED), 100))
            .withFrame(ImageFrame.of(fill(BLUE), 100))
            .withLoopCount(loopCount)
            .build();
    }

    private static @NotNull PixelBuffer fill(int argb) {
        PixelBuffer pixels = PixelBuffer.create(SIZE, SIZE);
        pixels.fill(argb);
        return pixels;
    }

    private static byte @NotNull [] write(@NotNull ImageData data, @Nullable GifWriteOptions options) {
        return options == null
            ? new GifImageWriter().write(data)
            : new GifImageWriter().write(data, options);
    }

    private static @NotNull AnimatedImageData read(byte @NotNull [] gif) {
        ImageData decoded = new GifImageReader().read(gif);
        assertThat(decoded, instanceOf(AnimatedImageData.class));
        return (AnimatedImageData) decoded;
    }

}
