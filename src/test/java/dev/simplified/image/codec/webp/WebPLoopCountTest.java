package dev.simplified.image.codec.webp;

import dev.simplified.collection.Concurrent;
import dev.simplified.collection.ConcurrentList;
import dev.simplified.image.AnimatedImageData;
import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFactory;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.PixelBuffer;
import dev.simplified.image.frame.ImageFrame;
import dev.simplified.image.frame.LoopCount;
import dev.simplified.image.option.WebPWriteOptions;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.instanceOf;
import static org.hamcrest.Matchers.sameInstance;

/**
 * Pins the loop count {@link WebPImageWriter} declares, read out of the {@code ANIM} chunk's bytes,
 * and the one {@link WebPImageReader} reads back.
 * <p>
 * The chunk counts plays, the first play included, and {@code 0} plays forever: one more than the
 * repeats a {@link LoopCount} holds.
 */
class WebPLoopCountTest {

    private static final int RED = 0xFFFF0000;
    private static final int BLUE = 0xFF0000FF;

    /** The offset of the loop count in an {@code ANIM} payload, after the background colour. */
    private static final int LOOP_COUNT_OFFSET = 4;

    // ──── what the writer declares ────

    @Test
    @DisplayName("each count is written as its plays, and an endless one as 0")
    void playsAreWritten() {
        assertThat(animLoopCount(write(animation(LoopCount.Preset.NONE), null)), equalTo(1));
        assertThat(animLoopCount(write(animation(LoopCount.Preset.ONCE), null)), equalTo(2));
        assertThat(animLoopCount(write(animation(LoopCount.of(4)), null)), equalTo(5));
        assertThat(animLoopCount(write(animation(LoopCount.of(65534)), null)), equalTo(65535));
        assertThat(animLoopCount(write(animation(LoopCount.Preset.INFINITE), null)), equalTo(0));
    }

    @Test
    @DisplayName("a count whose plays exceed the chunk's 16 bits is written as the most it declares")
    void playsBeyondSixteenBitsAreCapped() {
        assertThat(animLoopCount(write(animation(LoopCount.of(65535)), null)), equalTo(65535));
        assertThat(animLoopCount(write(animation(LoopCount.of(Integer.MAX_VALUE - 1)), null)), equalTo(65535));
    }

    @Test
    @DisplayName("a GIF repeating 65535 times converts to a WebP playing 65535 times")
    void gifRepeatingSixteenBitsConverts() {
        byte[] gif = ImageFactory.toByteArray(animation(LoopCount.of(65535)), ImageFormat.GIF);
        AnimatedImageData decoded = (AnimatedImageData) ImageFactory.fromByteArray(gif, ImageFormat.GIF);

        assertThat(decoded.getLoopCount(), equalTo(LoopCount.of(65535)));
        assertThat(animLoopCount(write(decoded, null)), equalTo(65535));
    }

    @Test
    @DisplayName("options that set no count declare the image's")
    void optionsWithoutACountDeclareTheImages() {
        WebPWriteOptions options = WebPWriteOptions.builder().isLossless(false).build();

        assertThat(animLoopCount(write(animation(LoopCount.Preset.NONE), options)), equalTo(1));
        assertThat(animLoopCount(write(animation(LoopCount.of(4)), options)), equalTo(5));
    }

    @Test
    @DisplayName("a finite count in the options is declared over the image's")
    void finiteOptionsCountWins() {
        WebPWriteOptions options = WebPWriteOptions.builder().withLoopCount(LoopCount.Preset.ONCE).build();

        assertThat(animLoopCount(write(animation(LoopCount.Preset.INFINITE), options)), equalTo(2));
        assertThat(animLoopCount(write(animation(LoopCount.of(4)), options)), equalTo(2));
    }

    @Test
    @DisplayName("an endless count in the options is declared over the image's finite one")
    void endlessOptionsCountWins() {
        WebPWriteOptions options = WebPWriteOptions.builder().withLoopCount(LoopCount.Preset.INFINITE).build();

        assertThat(animLoopCount(write(animation(LoopCount.Preset.NONE), options)), equalTo(0));
        assertThat(animLoopCount(write(animation(LoopCount.of(4)), options)), equalTo(0));
    }

    // ──── what the reader reads ────

    @Test
    @DisplayName("each count reads back as the count written")
    void eachCountRoundTrips() {
        for (LoopCount count : List.of(LoopCount.Preset.NONE, LoopCount.Preset.ONCE, LoopCount.of(4), LoopCount.Preset.INFINITE))
            assertThat(read(write(animation(count), null)).getLoopCount(), equalTo(count));
    }

    @Test
    @DisplayName("the ANIM count is read as plays")
    void animCountIsReadAsPlays() {
        byte[] webp = write(animation(LoopCount.Preset.INFINITE), null);

        assertThat(read(withAnimLoopCount(webp, 0)).getLoopCount(), sameInstance(LoopCount.Preset.INFINITE));
        assertThat(read(withAnimLoopCount(webp, 1)).getLoopCount(), sameInstance(LoopCount.Preset.NONE));
        assertThat(read(withAnimLoopCount(webp, 2)).getLoopCount(), sameInstance(LoopCount.Preset.ONCE));
        assertThat(read(withAnimLoopCount(webp, 3)).getLoopCount(), equalTo(LoopCount.of(2)));
        assertThat(read(withAnimLoopCount(webp, 65535)).getLoopCount(), equalTo(LoopCount.of(65534)));
    }

    @Test
    @DisplayName("an animation with no ANIM chunk, which the container forbids, reads as playing once")
    void missingAnimReadsAsPlayingOnce() {
        ConcurrentList<WebPChunk> chunks = Concurrent.newList();
        for (WebPChunk chunk : RiffContainer.parse(write(animation(LoopCount.Preset.INFINITE), null))) {
            if (chunk.type() != WebPChunk.Type.ANIM)
                chunks.add(chunk);
        }

        assertThat(read(RiffContainer.write(chunks)).getLoopCount(), sameInstance(LoopCount.Preset.NONE));
    }

    // ──── fixtures ────

    /**
     * Two 4x4 frames of different colours, repeating {@code loopCount} times after the first play.
     */
    private static @NotNull AnimatedImageData animation(@NotNull LoopCount loopCount) {
        return AnimatedImageData.builder()
            .withFrame(ImageFrame.of(fill(RED), 100))
            .withFrame(ImageFrame.of(fill(BLUE), 100))
            .withLoopCount(loopCount)
            .build();
    }

    private static @NotNull PixelBuffer fill(int argb) {
        PixelBuffer pixels = PixelBuffer.create(4, 4);
        pixels.fill(argb);
        return pixels;
    }

    // ──── WebP bytes ────

    private static byte @NotNull [] write(@NotNull ImageData data, @Nullable WebPWriteOptions options) {
        return options == null
            ? new WebPImageWriter().write(data)
            : new WebPImageWriter().write(data, options);
    }

    private static @NotNull AnimatedImageData read(byte @NotNull [] webp) {
        ImageData decoded = new WebPImageReader().read(webp);
        assertThat(decoded, instanceOf(AnimatedImageData.class));
        return (AnimatedImageData) decoded;
    }

    /**
     * Returns the loop count the one {@code ANIM} chunk of {@code webp} declares, low byte first.
     */
    private static int animLoopCount(byte @NotNull [] webp) {
        byte[] payload = anim(RiffContainer.parse(webp)).payload();
        return (payload[LOOP_COUNT_OFFSET] & 0xFF) | ((payload[LOOP_COUNT_OFFSET + 1] & 0xFF) << 8);
    }

    /**
     * Returns {@code webp} with its {@code ANIM} chunk declaring {@code loopCount}, every other byte
     * as it was.
     */
    private static byte @NotNull [] withAnimLoopCount(byte @NotNull [] webp, int loopCount) {
        ConcurrentList<WebPChunk> chunks = RiffContainer.parse(webp);
        WebPChunk anim = anim(chunks);
        byte[] payload = anim.payload();
        payload[LOOP_COUNT_OFFSET] = (byte) (loopCount & 0xFF);
        payload[LOOP_COUNT_OFFSET + 1] = (byte) ((loopCount >> 8) & 0xFF);

        ConcurrentList<WebPChunk> patched = Concurrent.newList();
        for (WebPChunk chunk : chunks)
            patched.add(chunk == anim ? RiffContainer.createChunk(WebPChunk.Type.ANIM, payload) : chunk);

        return RiffContainer.write(patched);
    }

    private static @NotNull WebPChunk anim(@NotNull ConcurrentList<WebPChunk> chunks) {
        List<WebPChunk> anims = chunks.stream().filter(chunk -> chunk.type() == WebPChunk.Type.ANIM).toList();
        assertThat("ANIM chunks", anims.size(), equalTo(1));
        return anims.getFirst();
    }

}
