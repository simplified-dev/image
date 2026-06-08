package dev.simplified.image;

import dev.simplified.image.data.AnimatedImageData;
import dev.simplified.image.data.ImageFrame;
import dev.simplified.image.data.StaticImageData;
import dev.simplified.image.pixel.PixelBuffer;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;
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

}
