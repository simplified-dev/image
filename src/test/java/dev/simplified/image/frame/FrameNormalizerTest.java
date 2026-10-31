package dev.simplified.image.frame;

import dev.simplified.image.AnimatedImageData;
import dev.simplified.image.PixelBuffer;
import dev.simplified.image.pixel.AlphaMode;
import dev.simplified.image.pixel.ColorMath;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.awt.Color;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

class FrameNormalizerTest {

    private static AnimatedImageData animation(PixelBuffer... frames) {
        AnimatedImageData.Builder builder = AnimatedImageData.builder();
        for (PixelBuffer frame : frames)
            builder.withFrame(ImageFrame.of(frame, 40));
        return builder.build();
    }

    @Test
    @DisplayName("a premultiplied animation is scaled as premultiplied colour, every colour at or under its alpha")
    void scalesPremultipliedFrames() {
        PixelBuffer halves = PixelBuffer.canvas(8, 4);
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
                halves.setPixel(x, y, 0xFFFF0000);

        AnimatedImageData normalized = FrameNormalizer.normalize(animation(halves, PixelBuffer.canvas(3, 2)));
        assertThat(normalized.getAlphaMode(), is(AlphaMode.PREMULTIPLIED));

        for (ImageFrame frame : normalized.getFrames())
            for (int pixel : frame.pixels().data())
                assertThat(Integer.toHexString(pixel), ColorMath.red(pixel) <= ColorMath.alpha(pixel), is(true));
    }

    @Test
    @DisplayName("the letterbox fill is premultiplied under a premultiplied animation and laid down as given under a straight one")
    void letterboxFollowsTheConvention() {
        Color fill = new Color(0x33, 0x66, 0x99, 0x80);
        PixelBuffer square = PixelBuffer.straight(2, 2);
        square.fill(0xFFFF0000);

        AnimatedImageData premultiplied = FrameNormalizer.normalize(
            animation(square.toAlphaMode(AlphaMode.PREMULTIPLIED)), 6, 2, FitMode.CONTAIN, false, fill);
        AnimatedImageData straight = FrameNormalizer.normalize(
            animation(square), 6, 2, FitMode.CONTAIN, false, fill);

        assertThat(premultiplied.getAlphaMode(), is(AlphaMode.PREMULTIPLIED));
        assertThat(straight.getAlphaMode(), is(AlphaMode.STRAIGHT));
        assertThat(Integer.toHexString(premultiplied.getFrames().getFirst().pixels().getPixel(0, 0)), is(Integer.toHexString(0x801A334D)));
        assertThat(Integer.toHexString(straight.getFrames().getFirst().pixels().getPixel(0, 0)), is(Integer.toHexString(0x80336699)));
        assertThat(premultiplied.getFrames().getFirst().pixels().getPixel(2, 0), is(0xFFFF0000));
    }

}
