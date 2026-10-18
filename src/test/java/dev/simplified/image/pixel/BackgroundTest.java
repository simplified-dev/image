package dev.simplified.image.pixel;

import dev.simplified.image.AnimatedImageData;
import dev.simplified.image.ImageData;
import dev.simplified.image.PixelBuffer;
import dev.simplified.image.StaticImageData;
import dev.simplified.image.codec.png.PngImageReader;
import dev.simplified.image.codec.png.PngImageWriter;
import dev.simplified.image.frame.FrameBlend;
import dev.simplified.image.frame.FrameDisposal;
import dev.simplified.image.frame.ImageFrame;
import dev.simplified.image.frame.LoopCount;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.instanceOf;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.lessThanOrEqualTo;
import static org.hamcrest.Matchers.sameInstance;

class BackgroundTest {

    @Test
    @DisplayName("solid background fills every pixel; an alpha-zero solid reports transparent")
    void solidFillAndTransparency() {
        assertThat(Background.TRANSPARENT.isTransparent(), is(true));
        assertThat(Background.solid(0x00123456).isTransparent(), is(true));
        assertThat(Background.solid(0xFF112233).isTransparent(), is(false));

        PixelBuffer buffer = PixelBuffer.create(2, 2);
        Background.solid(0xFF112233).fill(buffer);
        for (int y = 0; y < 2; y++)
            for (int x = 0; x < 2; x++)
                assertThat(buffer.getPixel(x, y), is(0xFF112233));
    }

    @Test
    @DisplayName("checkerboard alternates colour every cell from the top-left origin")
    void checkerboardPattern() {
        PixelBuffer buffer = PixelBuffer.create(16, 16);
        Background.checkerboard(0xFFAAAAAA, 0xFF555555, 8).fill(buffer);
        assertThat(buffer.getPixel(0, 0), is(0xFFAAAAAA));
        assertThat(buffer.getPixel(8, 0), is(0xFF555555));
        assertThat(buffer.getPixel(0, 8), is(0xFF555555));
        assertThat(buffer.getPixel(8, 8), is(0xFFAAAAAA));
    }

    @Test
    @DisplayName("composite is a no-op for a transparent background")
    void transparentNoOp() {
        ImageData image = StaticImageData.of(PixelBuffer.create(4, 4));
        assertThat(Background.TRANSPARENT.composite(image), is(sameInstance(image)));
    }

    @Test
    @DisplayName("composite blits the render source-over a solid fill")
    void compositesOverSolid() {
        PixelBuffer rendered = PixelBuffer.create(2, 1);
        rendered.setPixel(0, 0, 0x00000000); // transparent - background shows through
        rendered.setPixel(1, 0, 0xFF0000FF); // opaque blue - stays blue

        ImageData out = Background.solid(0xFFFF0000).composite(StaticImageData.of(rendered));
        PixelBuffer result = out.toPixelBuffer();
        assertThat(result.getPixel(0, 0), is(0xFFFF0000));
        assertThat(result.getPixel(1, 0), is(0xFF0000FF));
    }

    @Test
    @DisplayName("composite shows the fill through a translucent premultiplied render as drawing it over the fill would")
    void compositesTranslucentRenderAsDrawnOverFill() {
        int source = 0x80FFFFFF;
        int fill = 0xFF204060;
        PixelBuffer rendered = PixelBuffer.create(1, 1);
        rendered.setPixel(0, 0, ColorMath.blend(source, ColorMath.TRANSPARENT, BlendMode.NORMAL));

        PixelBuffer result = Background.solid(fill).composite(StaticImageData.of(rendered, AlphaMode.PREMULTIPLIED)).toPixelBuffer();
        assertThat(result.getPixel(0, 0), is(0xFF90A0B0));
        assertThat(result.getPixel(0, 0), is(ColorMath.blend(source, fill, BlendMode.NORMAL)));
    }

    @Test
    @DisplayName("composite lays a straight image down as straight colour")
    void compositesStraightImageAsStraightColour() {
        PixelBuffer rendered = PixelBuffer.create(1, 1);
        rendered.setPixel(0, 0, 0x80FF0000);

        PixelBuffer result = Background.solid(0xFF0000FF).composite(StaticImageData.of(rendered)).toPixelBuffer();
        assertThat(result.getPixel(0, 0), is(0xFF80007F));
    }

    @Test
    @DisplayName("composite preserves animated frame count and per-frame timing")
    void preservesAnimation() {
        ImageData animated = AnimatedImageData.builder()
            .withFrame(ImageFrame.of(PixelBuffer.create(2, 2), 40))
            .withFrame(ImageFrame.of(PixelBuffer.create(2, 2), 60))
            .build();

        ImageData out = Background.solid(0xFF222222).composite(animated);
        assertThat(out.isAnimated(), is(true));
        assertThat(out.getFrames().size(), is(2));
        assertThat(out.getFrames().getFirst().delayMs(), is(40));
        assertThat(out.getFrames().getLast().delayMs(), is(60));
        assertThat(out.getFrames().getFirst().pixels().getPixel(0, 0), is(0xFF222222));
    }

    @Test
    @DisplayName("composite keeps an animation's canvas and playback when the canvas is larger than its first frame")
    void keepsTheCanvasOfAnAnimation() {
        AnimatedImageData animated = AnimatedImageData.builder()
            .withWidth(8)
            .withHeight(6)
            .withFrame(ImageFrame.of(PixelBuffer.create(2, 2), 40, 1, 1, FrameDisposal.DO_NOT_DISPOSE, FrameBlend.OVER))
            .withFrame(ImageFrame.of(PixelBuffer.create(3, 3), 60, 4, 2, FrameDisposal.RESTORE_TO_PREVIOUS, FrameBlend.SOURCE))
            .withLoopCount(LoopCount.of(3))
            .withBackgroundColor(0xFF00FF00)
            .withAlphaMode(AlphaMode.PREMULTIPLIED)
            .build();

        ImageData out = Background.solid(0xFF222222).composite(animated);
        assertThat(out, is(instanceOf(AnimatedImageData.class)));

        AnimatedImageData composited = (AnimatedImageData) out;
        assertThat(composited.getWidth(), is(8));
        assertThat(composited.getHeight(), is(6));
        assertThat(composited.getLoopCount(), is(LoopCount.of(3)));
        assertThat(composited.getBackgroundColor(), is(0xFF00FF00));
        assertThat(composited.getAlphaMode(), is(AlphaMode.PREMULTIPLIED));
        assertThat(composited.getTotalDurationMs(), is(100));
        assertThat(composited.getFrames().size(), is(2));

        for (int i = 0; i < 2; i++) {
            ImageFrame frame = composited.getFrames().get(i);
            assertThat(frame.width(), is(8));
            assertThat(frame.height(), is(6));
            assertThat(frame.delayMs(), is(animated.getFrames().get(i).delayMs()));
            assertThat(frame.offsetX(), is(0));
            assertThat(frame.offsetY(), is(0));
            assertThat(frame.disposal(), is(FrameDisposal.RESTORE_TO_BACKGROUND));
            assertThat(frame.blend(), is(FrameBlend.SOURCE));
        }
    }

    @Test
    @DisplayName("composite fills the whole canvas, outside every frame too, with a checkerboard anchored at the canvas origin")
    void fillsTheWholeCanvasFromItsOrigin() {
        AnimatedImageData animated = AnimatedImageData.builder()
            .withWidth(8)
            .withHeight(6)
            .withFrame(ImageFrame.of(PixelBuffer.create(2, 2), 40, 4, 2, FrameDisposal.DO_NOT_DISPOSE, FrameBlend.OVER))
            .withFrame(ImageFrame.of(PixelBuffer.create(2, 2), 40, 0, 0, FrameDisposal.DO_NOT_DISPOSE, FrameBlend.OVER))
            .build();

        PixelBuffer solid = Background.solid(0xFF222222).composite(animated).getFrames().getFirst().pixels();
        assertThat(solid.getPixel(0, 0), is(0xFF222222));
        assertThat(solid.getPixel(7, 5), is(0xFF222222));

        PixelBuffer board = Background.checkerboard(0xFFAAAAAA, 0xFF555555, 2).composite(animated).getFrames().getFirst().pixels();
        assertThat(board.getPixel(4, 2), is(0xFF555555));
        assertThat(board.getPixel(0, 0), is(0xFFAAAAAA));
    }

    @Test
    @DisplayName("a partly transparent OVER frame shows the frame beneath it, not the fill")
    void overFrameShowsTheFrameBeneath() {
        PixelBuffer red = PixelBuffer.create(4, 4);
        red.fill(0xFFFF0000);
        AnimatedImageData animated = AnimatedImageData.builder()
            .withWidth(4)
            .withHeight(4)
            .withFrame(ImageFrame.of(red, 40, 0, 0, FrameDisposal.DO_NOT_DISPOSE, FrameBlend.SOURCE))
            .withFrame(ImageFrame.of(PixelBuffer.create(2, 2), 40, 0, 0, FrameDisposal.DO_NOT_DISPOSE, FrameBlend.OVER))
            .build();

        PixelBuffer second = Background.solid(0xFF222222).composite(animated).getFrames().getLast().pixels();
        assertThat(second.getPixel(0, 0), is(0xFFFF0000));
    }

    @Test
    @DisplayName("composite keeps the convention the image is held in")
    void compositeKeepsTheConvention() {
        for (AlphaMode mode : AlphaMode.values()) {
            ImageData still = StaticImageData.of(PixelBuffer.create(2, 2), mode);
            ImageData animated = AnimatedImageData.builder()
                .withFrame(ImageFrame.of(PixelBuffer.create(2, 2), 40))
                .withFrame(ImageFrame.of(PixelBuffer.create(2, 2), 60))
                .withAlphaMode(mode)
                .build();

            assertThat(Background.solid(0xFF000000).composite(still).getAlphaMode(), is(mode));
            assertThat(Background.solid(0xFF000000).composite(animated).getAlphaMode(), is(mode));
            assertThat(Background.checkerboard().composite(still).getAlphaMode(), is(mode));
            assertThat(Background.checkerboard().composite(animated).getAlphaMode(), is(mode));
        }
    }

    @Test
    @DisplayName("a translucent fill under a premultiplied image is premultiplied, so it encodes as the colour given")
    void translucentFillUnderPremultipliedImage() {
        int fill = 0x80336699;
        int premultipliedFill = 0x801A334D;
        int render = 0x80400000;
        PixelBuffer rendered = PixelBuffer.create(2, 1);
        rendered.setPixel(1, 0, render);

        ImageData out = Background.solid(fill).composite(StaticImageData.of(rendered, AlphaMode.PREMULTIPLIED));
        PixelBuffer result = out.toPixelBuffer();
        assertThat(result.getPixel(0, 0), is(premultipliedFill));
        assertThat(result.getPixel(1, 0), is(ColorMath.blend(render, premultipliedFill, BlendMode.PREMULTIPLIED)));

        // Laid down unpremultiplied, the fill would encode as 0x8066CBFF. Premultiplied, it encodes as
        // the colour given to the level a premultiplied pixel at alpha 128 holds: the red of 0x33
        // premultiplies to 0x1A, which divides back to 0x34.
        int decoded = new PngImageReader().read(new PngImageWriter().write(out)).toPixelBuffer().getPixel(0, 0);
        assertThat(Integer.toHexString(decoded), is(Integer.toHexString(0x80346699)));
        for (int shift = 0; shift <= 24; shift += 8)
            assertThat(Math.abs(((decoded >>> shift) & 0xFF) - ((fill >>> shift) & 0xFF)) <= 1, is(true));
    }

    @Test
    @DisplayName("every translucent cell of a checkerboard under a premultiplied image is premultiplied")
    void translucentCheckerboardUnderPremultipliedImage() {
        ImageData image = StaticImageData.of(PixelBuffer.create(2, 1), AlphaMode.PREMULTIPLIED);
        PixelBuffer result = Background.checkerboard(0x80336699, 0x40FFFFFF, 1).composite(image).toPixelBuffer();

        assertThat(result.getPixel(0, 0), is(0x801A334D));
        assertThat(result.getPixel(1, 0), is(0x40404040));
    }

    @Test
    @DisplayName("a translucent fill under a straight image is laid down as given")
    void translucentFillUnderStraightImage() {
        PixelBuffer result = Background.solid(0x80336699).composite(StaticImageData.of(PixelBuffer.create(1, 1))).toPixelBuffer();
        assertThat(result.getPixel(0, 0), is(0x80336699));
    }

    @Test
    @DisplayName("a straight image and its premultiplied copy composite to the same colours over an opaque fill")
    void straightAndPremultipliedCompositeAlike() {
        PixelBuffer straight = PixelBuffer.create(4, 4);
        for (int i = 0; i < 16; i++)
            straight.data()[i] = ((i * 17) << 24) | ((i * 7) << 16) | ((i * 3) << 8) | i;
        PixelBuffer premultiplied = straight.copy();
        premultiplied.premultiplyAlpha();

        for (Background background : new Background[] { Background.solid(0xFF204060), Background.checkerboard() }) {
            int[] fromStraight = background.composite(StaticImageData.of(straight)).toPixelBuffer().data();
            int[] fromPremultiplied = background.composite(StaticImageData.of(premultiplied, AlphaMode.PREMULTIPLIED)).toPixelBuffer().data();
            for (int i = 0; i < 16; i++)
                for (int shift = 0; shift <= 24; shift += 8)
                    assertThat(Math.abs(((fromStraight[i] >>> shift) & 0xFF) - ((fromPremultiplied[i] >>> shift) & 0xFF)), is(lessThanOrEqualTo(1)));
        }
    }

}
