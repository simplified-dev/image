package dev.simplified.image;

import dev.simplified.collection.Concurrent;
import dev.simplified.collection.ConcurrentList;
import dev.simplified.image.frame.FrameBlend;
import dev.simplified.image.frame.FrameDisposal;
import dev.simplified.image.frame.ImageFrame;
import dev.simplified.image.pixel.AlphaMode;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.instanceOf;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;
import static org.hamcrest.Matchers.sameInstance;

/**
 * Pins the alpha convention an {@link ImageData} declares - what each way of building one answers,
 * and what converting it to the other convention keeps and copies.
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

    // ──── the declared convention ────

    @Test
    @DisplayName("an image built without a convention declares straight colour")
    void unlabelledImagesAreStraight() {
        assertThat(StaticImageData.of(row(STRAIGHT)).getAlphaMode(), equalTo(AlphaMode.STRAIGHT));
        assertThat(StaticImageData.of(new BufferedImage(2, 2, BufferedImage.TYPE_INT_ARGB)).getAlphaMode(), equalTo(AlphaMode.STRAIGHT));
        assertThat(AnimatedImageData.builder().withFrame(ImageFrame.of(row(STRAIGHT), 100)).build().getAlphaMode(), equalTo(AlphaMode.STRAIGHT));
        assertThat(new Still(ImageFrame.of(row(STRAIGHT))).getAlphaMode(), equalTo(AlphaMode.STRAIGHT));
    }

    @Test
    @DisplayName("an image built with a convention declares it")
    void labelledImagesDeclareTheirConvention() {
        assertThat(StaticImageData.of(row(PREMULTIPLIED), AlphaMode.PREMULTIPLIED).getAlphaMode(), equalTo(AlphaMode.PREMULTIPLIED));
        assertThat(StaticImageData.of(row(STRAIGHT), AlphaMode.STRAIGHT).getAlphaMode(), equalTo(AlphaMode.STRAIGHT));
        assertThat(
            AnimatedImageData.builder()
                .withFrame(ImageFrame.of(row(PREMULTIPLIED), 100))
                .withAlphaMode(AlphaMode.PREMULTIPLIED)
                .build()
                .getAlphaMode(),
            equalTo(AlphaMode.PREMULTIPLIED)
        );
    }

    @Test
    @DisplayName("every reader decodes to straight colour")
    void decodedImagesAreStraight() {
        ImageData source = StaticImageData.of(square(16, 0xC0336699));

        for (ImageFormat format : new ImageFormat[] {
            ImageFormat.PNG, ImageFormat.JPEG, ImageFormat.BMP, ImageFormat.GIF, ImageFormat.WEBP,
            ImageFormat.TIFF, ImageFormat.ICO, ImageFormat.QOI, ImageFormat.TGA, ImageFormat.PNM
        }) {
            ImageData decoded = ImageFactory.fromByteArray(ImageFactory.toByteArray(source, format));
            assertThat(format.name(), decoded.getAlphaMode(), equalTo(AlphaMode.STRAIGHT));
        }
    }

    // ──── converting to the convention already held ────

    @Test
    @DisplayName("converting to the convention an image already holds answers the image and its own buffer")
    void convertingToTheHeldConventionIsTheIdentity() {
        for (AlphaMode mode : AlphaMode.values()) {
            PixelBuffer pixels = row(PREMULTIPLIED);
            ImageData still = StaticImageData.of(pixels, mode);
            ImageData animated = AnimatedImageData.builder()
                .withFrame(ImageFrame.of(pixels, 100))
                .withAlphaMode(mode)
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
        PixelBuffer pixels = row(PREMULTIPLIED);
        ImageData image = StaticImageData.of(pixels, AlphaMode.PREMULTIPLIED);

        PixelBuffer straight = image.toPixelBuffer(AlphaMode.STRAIGHT);
        assertThat(straight, not(sameInstance(pixels)));
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
        PixelBuffer pixels = row(STRAIGHT);
        ImageData image = StaticImageData.of(pixels);

        assertThat(image.toPixelBuffer(AlphaMode.PREMULTIPLIED).data(), equalTo(PREMULTIPLIED));
        assertThat(image.toAlphaMode(AlphaMode.PREMULTIPLIED).getAlphaMode(), equalTo(AlphaMode.PREMULTIPLIED));
        assertThat(pixels.data(), equalTo(STRAIGHT));
    }

    @Test
    @DisplayName("a premultiplied animation converts every frame and keeps the canvas, timing and frame metadata")
    void premultipliedAnimationConvertsEveryFrame() {
        PixelBuffer first = row(PREMULTIPLIED);
        PixelBuffer second = row(reversed(PREMULTIPLIED));
        AnimatedImageData image = AnimatedImageData.builder()
            .withFrame(ImageFrame.of(first, 40, 1, 2, FrameDisposal.DO_NOT_DISPOSE, FrameBlend.OVER))
            .withFrame(ImageFrame.of(second, 70, 3, 0, FrameDisposal.RESTORE_TO_PREVIOUS, FrameBlend.SOURCE))
            .withWidth(12)
            .withHeight(5)
            .withLoopCount(3)
            .withBackgroundColor(0x80112233)
            .withAlphaMode(AlphaMode.PREMULTIPLIED)
            .build();

        ImageData converted = image.toAlphaMode(AlphaMode.STRAIGHT);
        assertThat(converted, instanceOf(AnimatedImageData.class));
        AnimatedImageData animated = (AnimatedImageData) converted;

        assertThat(animated.getAlphaMode(), equalTo(AlphaMode.STRAIGHT));
        assertThat(animated.getWidth(), equalTo(12));
        assertThat(animated.getHeight(), equalTo(5));
        assertThat(animated.getLoopCount(), equalTo(3));
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
    @DisplayName("a buffer whose alpha carries no data holds the same colour in either convention")
    void bufferWithoutAlphaIsNotConverted() {
        PixelBuffer pixels = PixelBuffer.of(PREMULTIPLIED.clone(), PREMULTIPLIED.length, 1, false);
        ImageData still = StaticImageData.of(pixels, AlphaMode.PREMULTIPLIED);
        ImageData animated = AnimatedImageData.builder()
            .withFrame(ImageFrame.of(pixels, 100))
            .withAlphaMode(AlphaMode.PREMULTIPLIED)
            .build();

        assertThat(still.toPixelBuffer(AlphaMode.STRAIGHT), sameInstance(pixels));
        assertThat(still.toAlphaMode(AlphaMode.STRAIGHT).toPixelBuffer(), sameInstance(pixels));
        assertThat(animated.toAlphaMode(AlphaMode.STRAIGHT).getFrames().getFirst().pixels(), sameInstance(pixels));
        assertThat(pixels.data(), equalTo(PREMULTIPLIED));
    }

    // ──── as a buffered image ────

    @Test
    @DisplayName("a straight image's buffered image shares the first frame's pixel array")
    void straightBufferedImageIsZeroCopy() {
        PixelBuffer pixels = row(STRAIGHT);
        BufferedImage image = StaticImageData.of(pixels).toBufferedImage();

        assertThat(backing(image), sameInstance(pixels.data()));
        assertThat(image.isAlphaPremultiplied(), is(false));
    }

    @Test
    @DisplayName("a premultiplied image's buffered image holds a straight copy and leaves the frame alone")
    void premultipliedBufferedImageIsAStraightCopy() {
        PixelBuffer pixels = row(PREMULTIPLIED);
        BufferedImage image = StaticImageData.of(pixels, AlphaMode.PREMULTIPLIED).toBufferedImage();

        assertThat(backing(image), not(sameInstance(pixels.data())));
        assertThat(image.isAlphaPremultiplied(), is(false));
        for (int x = 0; x < STRAIGHT.length; x++)
            assertThat("pixel " + x, image.getRGB(x, 0), equalTo(STRAIGHT[x]));

        image.setRGB(0, 0, 0xFF000000);
        assertThat(pixels.data(), equalTo(PREMULTIPLIED));
    }

    // ──── fixtures ────

    private static @NotNull PixelBuffer row(int @NotNull [] pixels) {
        return PixelBuffer.of(pixels.clone(), pixels.length, 1);
    }

    private static @NotNull PixelBuffer square(int size, int argb) {
        PixelBuffer pixels = PixelBuffer.create(size, size);
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
