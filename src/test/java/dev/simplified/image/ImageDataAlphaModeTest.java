package dev.simplified.image;

import dev.simplified.collection.Concurrent;
import dev.simplified.collection.ConcurrentList;
import dev.simplified.image.frame.FrameBlend;
import dev.simplified.image.frame.FrameDisposal;
import dev.simplified.image.frame.ImageFrame;
import dev.simplified.image.frame.LoopCount;
import dev.simplified.image.option.WebPWriteOptions;
import dev.simplified.image.pixel.AlphaMode;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.instanceOf;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;
import static org.hamcrest.Matchers.sameInstance;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Pins the alpha convention an {@link ImageData} reports - the one its frames' buffers hold - and what
 * converting it to the other convention keeps and copies.
 */
class ImageDataAlphaModeTest {

    /** One premultiplied pixel per case the conversion treats differently, alpha 0 and 255 included. */
    private static final int[] PREMULTIPLIED = {
        0x80800000, 0x40202020, 0x01010101, 0x00000000, 0xFF336699, 0x80404040, 0xFE7F3F00, 0x02010000
    };

    /** The straight colour each {@link #PREMULTIPLIED} pixel stands for. */
    private static final int[] STRAIGHT = {
        0x80FF0000, 0x40808080, 0x01FFFFFF, 0x00000000, 0xFF336699, 0x80808080, 0xFE803F00, 0x02800000
    };

    // ──── the reported convention ────

    @Test
    @DisplayName("an image reports the convention its frames' buffers hold")
    void anImageReportsItsFramesConvention() {
        for (AlphaMode mode : AlphaMode.values()) {
            assertThat(mode.name(), StaticImageData.of(row(PREMULTIPLIED, mode)).getAlphaMode(), equalTo(mode));
            assertThat(mode.name(), new Still(ImageFrame.of(row(PREMULTIPLIED, mode))).getAlphaMode(), equalTo(mode));
            assertThat(
                mode.name(),
                AnimatedImageData.builder()
                    .withFrame(ImageFrame.of(row(PREMULTIPLIED, mode), 100))
                    .withFrame(ImageFrame.of(row(reversed(PREMULTIPLIED), mode), 100))
                    .build()
                    .getAlphaMode(),
                equalTo(mode)
            );
        }

        assertThat(StaticImageData.of(PixelBuffer.straight(2, 2)).getAlphaMode(), equalTo(AlphaMode.STRAIGHT));
        assertThat(StaticImageData.of(PixelBuffer.canvas(2, 2)).getAlphaMode(), equalTo(AlphaMode.PREMULTIPLIED));
        assertThat(StaticImageData.of(new BufferedImage(2, 2, BufferedImage.TYPE_INT_ARGB)).getAlphaMode(), equalTo(AlphaMode.STRAIGHT));
    }

    @Test
    @DisplayName("an animation refuses a frame holding the other convention from its first")
    void anAnimationRefusesFramesThatDisagree() {
        AnimatedImageData.Builder builder = AnimatedImageData.builder()
            .withFrame(ImageFrame.of(row(STRAIGHT, AlphaMode.STRAIGHT), 100))
            .withFrame(ImageFrame.of(row(STRAIGHT, AlphaMode.STRAIGHT), 100))
            .withFrame(ImageFrame.of(row(PREMULTIPLIED, AlphaMode.PREMULTIPLIED), 100));

        IllegalArgumentException refused = assertThrows(IllegalArgumentException.class, builder::build);
        assertThat(refused.getMessage(), equalTo("Frame 2 holds 'PREMULTIPLIED' colour where the first frame holds 'STRAIGHT'"));
    }

    @Test
    @DisplayName("every reader decodes to straight colour")
    void decodedImagesAreStraight() {
        ImageData source = StaticImageData.of(square(16, 0xC0336699));

        for (ImageFormat format : new ImageFormat[] {
            ImageFormat.PNG, ImageFormat.JPEG, ImageFormat.BMP, ImageFormat.GIF, ImageFormat.WEBP,
            ImageFormat.TIFF, ImageFormat.ICO, ImageFormat.QOI, ImageFormat.TGA, ImageFormat.PNM
        }) {
            assertStraight(format.name(), ImageFactory.fromByteArray(ImageFactory.toByteArray(source, format)));
        }

        // The routes those writes do not reach: an ICO whose image is a DIB, the lossy VP8 decoder, and
        // the canvas an animated WebP's frames are composed on.
        assertStraight("ICO with a BMP payload", ImageFactory.fromByteArray(icoWithBmpPayload(16, 0xC0336699)));

        byte[] lossy = ImageFactory.toByteArray(source, ImageFormat.WEBP, WebPWriteOptions.builder().isLossless(false).build());
        assertStraight("lossy WEBP", ImageFactory.fromByteArray(lossy));

        ImageData animation = AnimatedImageData.builder()
            .withFrame(ImageFrame.of(square(16, 0xC0336699), 100))
            .withFrame(ImageFrame.of(square(16, 0x80996633), 100))
            .build();
        ImageData decodedAnimation = ImageFactory.fromByteArray(ImageFactory.toByteArray(animation, ImageFormat.WEBP));
        assertThat(decodedAnimation.getFrames().size(), equalTo(2));
        assertStraight("animated WEBP", decodedAnimation);
    }

    // ──── converting to the convention already held ────

    @Test
    @DisplayName("converting to the convention an image already holds answers the image and its own buffer")
    void convertingToTheHeldConventionIsTheIdentity() {
        for (AlphaMode mode : AlphaMode.values()) {
            PixelBuffer pixels = row(PREMULTIPLIED, mode);
            ImageData still = StaticImageData.of(pixels);
            ImageData animated = AnimatedImageData.builder()
                .withFrame(ImageFrame.of(pixels, 100))
                .build();

            assertThat(still.toAlphaMode(mode), sameInstance(still));
            assertThat(still.toPixelBuffer(mode), sameInstance(pixels));
            assertThat(animated.toAlphaMode(mode), sameInstance(animated));
            assertThat(animated.toPixelBuffer(mode), sameInstance(pixels));
        }
    }

    // ──── converting to the other convention ────

    @Test
    @DisplayName("a premultiplied still converts to a straight copy and leaves its own pixels alone")
    void premultipliedStillConvertsToAStraightCopy() {
        PixelBuffer pixels = row(PREMULTIPLIED, AlphaMode.PREMULTIPLIED);
        ImageData image = StaticImageData.of(pixels);

        PixelBuffer straight = image.toPixelBuffer(AlphaMode.STRAIGHT);
        assertThat(straight, not(sameInstance(pixels)));
        assertThat(straight.alphaMode(), equalTo(AlphaMode.STRAIGHT));
        assertThat(straight.data(), equalTo(STRAIGHT));
        assertThat(pixels.data(), equalTo(PREMULTIPLIED));

        ImageData converted = image.toAlphaMode(AlphaMode.STRAIGHT);
        assertThat(converted, instanceOf(StaticImageData.class));
        assertThat(converted.getAlphaMode(), equalTo(AlphaMode.STRAIGHT));
        assertThat(converted.toPixelBuffer().data(), equalTo(STRAIGHT));
        assertThat(pixels.data(), equalTo(PREMULTIPLIED));
    }

    @Test
    @DisplayName("a straight still converts to a premultiplied copy")
    void straightStillConvertsToAPremultipliedCopy() {
        PixelBuffer pixels = row(STRAIGHT, AlphaMode.STRAIGHT);
        ImageData image = StaticImageData.of(pixels);

        assertThat(image.toPixelBuffer(AlphaMode.PREMULTIPLIED).alphaMode(), equalTo(AlphaMode.PREMULTIPLIED));
        assertThat(image.toPixelBuffer(AlphaMode.PREMULTIPLIED).data(), equalTo(PREMULTIPLIED));
        assertThat(image.toAlphaMode(AlphaMode.PREMULTIPLIED).getAlphaMode(), equalTo(AlphaMode.PREMULTIPLIED));
        assertThat(pixels.data(), equalTo(STRAIGHT));
    }

    @Test
    @DisplayName("a premultiplied animation converts every frame and keeps the canvas, timing and frame metadata")
    void premultipliedAnimationConvertsEveryFrame() {
        PixelBuffer first = row(PREMULTIPLIED, AlphaMode.PREMULTIPLIED);
        PixelBuffer second = row(reversed(PREMULTIPLIED), AlphaMode.PREMULTIPLIED);
        AnimatedImageData image = AnimatedImageData.builder()
            .withFrame(ImageFrame.of(first, 40, 1, 2, FrameDisposal.DO_NOT_DISPOSE, FrameBlend.OVER))
            .withFrame(ImageFrame.of(second, 70, 3, 0, FrameDisposal.RESTORE_TO_PREVIOUS, FrameBlend.SOURCE))
            .withWidth(12)
            .withHeight(5)
            .withLoopCount(LoopCount.of(3))
            .withBackgroundColor(0x80112233)
            .build();

        ImageData converted = image.toAlphaMode(AlphaMode.STRAIGHT);
        assertThat(converted, instanceOf(AnimatedImageData.class));
        AnimatedImageData animated = (AnimatedImageData) converted;

        assertThat(animated.getAlphaMode(), equalTo(AlphaMode.STRAIGHT));
        assertThat(animated.getWidth(), equalTo(12));
        assertThat(animated.getHeight(), equalTo(5));
        assertThat(animated.getLoopCount(), equalTo(LoopCount.of(3)));
        assertThat(animated.getBackgroundColor(), equalTo(0x80112233));
        assertThat(animated.getTotalDurationMs(), equalTo(110));
        assertThat(animated.getFrames().size(), equalTo(2));

        ImageFrame convertedFirst = animated.getFrames().getFirst();
        assertThat(convertedFirst.pixels().data(), equalTo(STRAIGHT));
        assertThat(convertedFirst.delayMs(), equalTo(40));
        assertThat(convertedFirst.offsetX(), equalTo(1));
        assertThat(convertedFirst.offsetY(), equalTo(2));
        assertThat(convertedFirst.disposal(), equalTo(FrameDisposal.DO_NOT_DISPOSE));
        assertThat(convertedFirst.blend(), equalTo(FrameBlend.OVER));

        ImageFrame convertedSecond = animated.getFrames().getLast();
        assertThat(convertedSecond.pixels().alphaMode(), equalTo(AlphaMode.STRAIGHT));
        assertThat(convertedSecond.pixels().data(), equalTo(reversed(STRAIGHT)));
        assertThat(convertedSecond.delayMs(), equalTo(70));
        assertThat(convertedSecond.offsetX(), equalTo(3));
        assertThat(convertedSecond.offsetY(), equalTo(0));
        assertThat(convertedSecond.disposal(), equalTo(FrameDisposal.RESTORE_TO_PREVIOUS));
        assertThat(convertedSecond.blend(), equalTo(FrameBlend.SOURCE));

        assertThat(first.data(), equalTo(PREMULTIPLIED));
        assertThat(second.data(), equalTo(reversed(PREMULTIPLIED)));
    }

    @Test
    @DisplayName("a buffer whose alpha carries no data converts to a buffer sharing its pixel array")
    void bufferWithoutAlphaIsNotConverted() {
        PixelBuffer pixels = PixelBuffer.ofStraight(PREMULTIPLIED.clone(), PREMULTIPLIED.length, 1, false)
            .toAlphaMode(AlphaMode.PREMULTIPLIED);
        ImageData still = StaticImageData.of(pixels);
        ImageData animated = AnimatedImageData.builder()
            .withFrame(ImageFrame.of(pixels, 100))
            .build();

        PixelBuffer straight = still.toPixelBuffer(AlphaMode.STRAIGHT);
        assertThat(straight.data(), sameInstance(pixels.data()));
        assertThat(straight.alphaMode(), equalTo(AlphaMode.STRAIGHT));
        assertThat(straight.hasAlpha(), is(false));

        ImageData convertedStill = still.toAlphaMode(AlphaMode.STRAIGHT);
        assertThat(convertedStill.getAlphaMode(), equalTo(AlphaMode.STRAIGHT));
        assertThat(convertedStill.toPixelBuffer().data(), sameInstance(pixels.data()));

        ImageData convertedAnimation = animated.toAlphaMode(AlphaMode.STRAIGHT);
        assertThat(convertedAnimation.getAlphaMode(), equalTo(AlphaMode.STRAIGHT));
        assertThat(convertedAnimation.getFrames().getFirst().pixels().data(), sameInstance(pixels.data()));
        assertThat(pixels.alphaMode(), equalTo(AlphaMode.PREMULTIPLIED));
        assertThat(pixels.data(), equalTo(PREMULTIPLIED));
    }

    // ──── as a buffered image ────

    @Test
    @DisplayName("a straight image's buffered image shares the first frame's pixel array")
    void straightBufferedImageIsZeroCopy() {
        PixelBuffer pixels = row(STRAIGHT, AlphaMode.STRAIGHT);
        BufferedImage image = StaticImageData.of(pixels).toBufferedImage();

        assertThat(backing(image), sameInstance(pixels.data()));
        assertThat(image.getType(), equalTo(BufferedImage.TYPE_INT_ARGB));
        assertThat(image.isAlphaPremultiplied(), is(false));
    }

    @Test
    @DisplayName("a premultiplied image's buffered image shares the first frame's pixel array as premultiplied colour")
    void toBufferedImageSharesAPremultipliedFrame() {
        PixelBuffer pixels = row(PREMULTIPLIED, AlphaMode.PREMULTIPLIED);
        BufferedImage image = StaticImageData.of(pixels).toBufferedImage();

        assertThat(backing(image), sameInstance(pixels.data()));
        assertThat(image.getType(), equalTo(BufferedImage.TYPE_INT_ARGB_PRE));
        assertThat(image.isAlphaPremultiplied(), is(true));
        for (int x = 0; x < STRAIGHT.length; x++)
            assertThat("pixel " + x, image.getRGB(x, 0), equalTo(STRAIGHT[x]));

        image.setRGB(0, 0, 0xFF000000);
        assertThat(pixels.getPixel(0, 0), equalTo(0xFF000000));
    }

    @Test
    @DisplayName("wrapping a buffer's buffered image answers its pixel array in its convention, in both conventions")
    void wrapOfToBufferedImageIsZeroCopyInBothConventions() {
        for (AlphaMode mode : AlphaMode.values()) {
            PixelBuffer pixels = row(PREMULTIPLIED, mode);
            PixelBuffer wrapped = PixelBuffer.wrap(pixels.toBufferedImage());

            assertThat(mode.name(), wrapped.data(), sameInstance(pixels.data()));
            assertThat(mode.name(), wrapped.alphaMode(), equalTo(mode));
            assertThat(mode.name(), wrapped, equalTo(pixels));

            ImageData image = StaticImageData.of(pixels);
            ImageData rewrapped = StaticImageData.of(image.toBufferedImage());
            assertThat(mode.name(), rewrapped.getAlphaMode(), equalTo(mode));
            assertThat(mode.name(), rewrapped.toPixelBuffer().data(), sameInstance(pixels.data()));
        }
    }

    // ──── fixtures ────

    /**
     * Asserts that a decoded image and every one of its frames hold straight colour.
     *
     * @param label the route the image was decoded through
     * @param decoded the decoded image
     */
    private static void assertStraight(@NotNull String label, @NotNull ImageData decoded) {
        assertThat(label, decoded.getAlphaMode(), equalTo(AlphaMode.STRAIGHT));
        for (ImageFrame frame : decoded.getFrames())
            assertThat(label, frame.pixels().alphaMode(), equalTo(AlphaMode.STRAIGHT));
    }

    /**
     * Writes an ICO whose one image is a 32-bit DIB, a payload the library's writer never produces,
     * every pixel holding {@code argb}.
     *
     * @param size the image's width and height, at most 255
     * @param argb the straight colour of every pixel
     * @return the ICO bytes
     */
    private static byte @NotNull [] icoWithBmpPayload(int size, int argb) {
        int maskRow = (size + 31) / 32 * 4;
        int dibSize = 40 + size * size * 4 + maskRow * size;
        ByteBuffer ico = ByteBuffer.allocate(6 + 16 + dibSize).order(ByteOrder.LITTLE_ENDIAN);

        // The header, then the one directory entry: dimensions, no palette, one plane, 32 bits, the
        // DIB's size and its offset.
        ico.putShort((short) 0).putShort((short) 1).putShort((short) 1);
        ico.put((byte) size).put((byte) size).put((byte) 0).put((byte) 0);
        ico.putShort((short) 1).putShort((short) 32).putInt(dibSize).putInt(6 + 16);

        // The BITMAPINFOHEADER, its height counting the AND mask; the fields after the bit count are 0.
        ico.putInt(40).putInt(size).putInt(size * 2).putShort((short) 1).putShort((short) 32);
        ico.position(6 + 16 + 40);

        // Bottom-up BGRA rows, which a little-endian ARGB int lays down byte for byte; the AND mask
        // after them is 0.
        for (int i = 0; i < size * size; i++)
            ico.putInt(argb);

        return ico.array();
    }

    /**
     * Wraps a copy of {@code pixels}, one pixel tall, in a buffer holding {@code alphaMode}.
     *
     * @param pixels the pixels, copied
     * @param alphaMode the convention the buffer holds
     * @return the buffer
     */
    private static @NotNull PixelBuffer row(int @NotNull [] pixels, @NotNull AlphaMode alphaMode) {
        return switch (alphaMode) {
            case STRAIGHT -> PixelBuffer.ofStraight(pixels.clone(), pixels.length, 1);
            case PREMULTIPLIED -> PixelBuffer.ofPremultiplied(pixels.clone(), pixels.length, 1);
        };
    }

    private static @NotNull PixelBuffer square(int size, int argb) {
        PixelBuffer pixels = PixelBuffer.straight(size, size);
        pixels.fill(argb);
        return pixels;
    }

    private static int @NotNull [] reversed(int @NotNull [] pixels) {
        int[] out = new int[pixels.length];

        for (int i = 0; i < pixels.length; i++)
            out[i] = pixels[pixels.length - 1 - i];

        return out;
    }

    private static int @NotNull [] backing(@NotNull BufferedImage image) {
        return ((DataBufferInt) image.getRaster().getDataBuffer()).getData();
    }

    /**
     * A still image implemented outside the library, overriding none of the interface's defaults.
     *
     * @param frame the one frame
     */
    private record Still(@NotNull ImageFrame frame) implements ImageData {

        @Override
        public int getWidth() {
            return this.frame.width();
        }

        @Override
        public int getHeight() {
            return this.frame.height();
        }

        @Override
        public boolean hasAlpha() {
            return this.frame.hasAlpha();
        }

        @Override
        public boolean isAnimated() {
            return false;
        }

        @Override
        public @NotNull ConcurrentList<ImageFrame> getFrames() {
            ConcurrentList<ImageFrame> frames = Concurrent.newList();
            frames.add(this.frame);
            return frames;
        }

    }

}
