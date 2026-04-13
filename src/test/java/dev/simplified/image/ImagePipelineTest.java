package dev.simplified.image;

import dev.simplified.collection.Concurrent;
import dev.simplified.collection.ConcurrentList;
import dev.simplified.image.codec.bmp.BmpImageReader;
import dev.simplified.image.codec.bmp.BmpImageWriter;
import dev.simplified.image.codec.gif.GifImageReader;
import dev.simplified.image.codec.gif.GifImageWriter;
import dev.simplified.image.codec.gif.GifWriteOptions;
import dev.simplified.image.codec.jpeg.JpegImageReader;
import dev.simplified.image.codec.jpeg.JpegImageWriter;
import dev.simplified.image.codec.jpeg.JpegWriteOptions;
import dev.simplified.image.codec.png.PngImageReader;
import dev.simplified.image.codec.png.PngImageWriter;
import dev.simplified.image.exception.UnsupportedFormatException;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.awt.*;
import java.awt.image.BufferedImage;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.*;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class ImagePipelineTest {

    private static BufferedImage createTestImage(int width, int height, int type, Color color) {
        BufferedImage image = new BufferedImage(width, height, type);
        Graphics2D g = image.createGraphics();
        try {
            g.setColor(color);
            g.fillRect(0, 0, width, height);
        } finally {
            g.dispose();
        }
        return image;
    }

    private static BufferedImage createRgbTestImage(int width, int height, Color color) {
        return createTestImage(width, height, BufferedImage.TYPE_INT_RGB, color);
    }

    private static BufferedImage createArgbTestImage(int width, int height, Color color) {
        return createTestImage(width, height, BufferedImage.TYPE_INT_ARGB, color);
    }

    // ──── ImageFormat ────

    @Nested
    class ImageFormatTests {

        @Test
        void detectJpeg() {
            byte[] magic = {(byte) 0xFF, (byte) 0xD8, (byte) 0xFF, (byte) 0xE0, 0, 0};
            assertThat(ImageFormat.detect(magic), is(ImageFormat.JPEG));
        }

        @Test
        void detectPng() {
            byte[] magic = {(byte) 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A};
            assertThat(ImageFormat.detect(magic), is(ImageFormat.PNG));
        }

        @Test
        void detectBmp() {
            byte[] magic = {0x42, 0x4D, 0, 0, 0, 0};
            assertThat(ImageFormat.detect(magic), is(ImageFormat.BMP));
        }

        @Test
        void detectGif() {
            byte[] magic = {0x47, 0x49, 0x46, 0x38, 0x39, 0x61};
            assertThat(ImageFormat.detect(magic), is(ImageFormat.GIF));
        }

        @Test
        void detectWebP() {
            byte[] magic = {0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0, 0x57, 0x45, 0x42, 0x50};
            assertThat(ImageFormat.detect(magic), is(ImageFormat.WEBP));
        }

        @Test
        void detectThrowsForUnknown() {
            byte[] garbage = {0x00, 0x01, 0x02, 0x03};
            assertThrows(UnsupportedFormatException.class, () -> ImageFormat.detect(garbage));
        }

        @Test
        void matchesNegative() {
            byte[] pngMagic = {(byte) 0x89, 0x50, 0x4E, 0x47};
            assertThat(ImageFormat.JPEG.matches(pngMagic), is(false));
            assertThat(ImageFormat.BMP.matches(pngMagic), is(false));
        }

        @Test
        void supportsAnimation() {
            assertThat(ImageFormat.GIF.isSupportsAnimation(), is(true));
            assertThat(ImageFormat.WEBP.isSupportsAnimation(), is(true));
            assertThat(ImageFormat.PNG.isSupportsAnimation(), is(false));
            assertThat(ImageFormat.JPEG.isSupportsAnimation(), is(false));
            assertThat(ImageFormat.BMP.isSupportsAnimation(), is(false));
        }

    }

    // ──── PixelBuffer ────

    @Nested
    class PixelBufferTests {

        @Test
        void wrapArgbZeroCopy() {
            BufferedImage image = createArgbTestImage(4, 4, Color.RED);
            PixelBuffer buffer = PixelBuffer.wrap(image);

            assertThat(buffer.width(), is(4));
            assertThat(buffer.height(), is(4));
            assertThat(buffer.getPixels().length, is(16));
        }

        @Test
        void wrapNonArgbConverts() {
            BufferedImage image = createRgbTestImage(4, 4, Color.BLUE);
            PixelBuffer buffer = PixelBuffer.wrap(image);

            assertThat(buffer.width(), is(4));
            assertThat(buffer.height(), is(4));
        }

        @Test
        void getSetPixel() {
            int[] pixels = new int[4];
            PixelBuffer buffer = PixelBuffer.of(pixels, 2, 2);

            buffer.setPixel(1, 0, 0xFFFF0000);
            assertThat(buffer.getPixel(1, 0), is(0xFFFF0000));
            assertThat(buffer.getPixel(0, 0), is(0));
        }

        @Test
        void toBufferedImageRoundTrip() {
            BufferedImage original = createArgbTestImage(8, 8, Color.GREEN);
            PixelBuffer buffer = PixelBuffer.wrap(original);
            BufferedImage result = buffer.toBufferedImage();

            assertThat(result.getWidth(), is(8));
            assertThat(result.getHeight(), is(8));
            assertThat(result.getRGB(0, 0), is(original.getRGB(0, 0)));
        }

    }

    // ──── StaticImageData ────

    @Nested
    class StaticImageDataTests {

        @Test
        void ofBufferedImage() {
            BufferedImage image = createArgbTestImage(16, 16, Color.CYAN);
            StaticImageData data = StaticImageData.of(image);

            assertThat(data.getWidth(), is(16));
            assertThat(data.getHeight(), is(16));
            assertThat(data.hasAlpha(), is(true));
            assertThat(data.isAnimated(), is(false));
            assertThat(data.getFrames(), hasSize(1));
        }

        @Test
        void ofRgbImageHasNoAlpha() {
            BufferedImage image = createRgbTestImage(4, 4, Color.WHITE);
            StaticImageData data = StaticImageData.of(image);

            assertThat(data.hasAlpha(), is(false));
        }

        @Test
        void toBufferedImageReturnsSameData() {
            BufferedImage image = createArgbTestImage(4, 4, Color.YELLOW);
            StaticImageData data = StaticImageData.of(image);

            assertThat(data.toBufferedImage().getRGB(0, 0), is(image.getRGB(0, 0)));
        }

    }

    // ──── AnimatedImageData ────

    @Nested
    class AnimatedImageDataTests {

        @Test
        void builderCreatesAnimation() {
            BufferedImage frame1 = createArgbTestImage(8, 8, Color.RED);
            BufferedImage frame2 = createArgbTestImage(8, 8, Color.BLUE);

            AnimatedImageData data = AnimatedImageData.builder()
                .withFrame(ImageFrame.of(PixelBuffer.wrap(frame1), 100))
                .withFrame(ImageFrame.of(PixelBuffer.wrap(frame2), 200))
                .withLoopCount(3)
                .build();

            assertThat(data.isAnimated(), is(true));
            assertThat(data.getFrames(), hasSize(2));
            assertThat(data.getLoopCount(), is(3));
            assertThat(data.getWidth(), is(8));
            assertThat(data.getHeight(), is(8));
        }

        @Test
        void toBufferedImageReturnsFirstFrame() {
            BufferedImage frame1 = createArgbTestImage(4, 4, Color.RED);
            BufferedImage frame2 = createArgbTestImage(4, 4, Color.BLUE);

            AnimatedImageData data = AnimatedImageData.builder()
                .withFrame(ImageFrame.of(PixelBuffer.wrap(frame1), 100))
                .withFrame(ImageFrame.of(PixelBuffer.wrap(frame2), 100))
                .build();

            assertThat(data.toBufferedImage().getRGB(0, 0), is(frame1.getRGB(0, 0)));
        }

    }

    // ──── FrameNormalizer ────

    @Nested
    class FrameNormalizerTests {

        @Test
        void normalizeMixedSizeFrames() {
            BufferedImage small = createArgbTestImage(4, 4, Color.RED);
            BufferedImage large = createArgbTestImage(8, 8, Color.BLUE);

            AnimatedImageData data = AnimatedImageData.builder()
                .withFrame(ImageFrame.of(PixelBuffer.wrap(small), 100))
                .withFrame(ImageFrame.of(PixelBuffer.wrap(large), 100))
                .build();

            AnimatedImageData normalized = FrameNormalizer.normalize(data);

            // Default uses min dimensions
            for (ImageFrame frame : normalized.getFrames()) {
                assertThat(frame.pixels().width(), is(4));
                assertThat(frame.pixels().height(), is(4));
            }
        }

        @Test
        void normalizePreservesDelays() {
            BufferedImage img = createArgbTestImage(4, 4, Color.RED);

            AnimatedImageData data = AnimatedImageData.builder()
                .withFrame(ImageFrame.of(PixelBuffer.wrap(img), 100))
                .withFrame(ImageFrame.of(PixelBuffer.wrap(img), 250))
                .build();

            AnimatedImageData normalized = FrameNormalizer.normalize(data);

            assertThat(normalized.getFrames().get(0).delayMs(), is(100));
            assertThat(normalized.getFrames().get(1).delayMs(), is(250));
        }

    }

    // ──── Standard Format Round-Trips ────

    @Nested
    class StandardFormatTests {

        @Test
        void pngRoundTrip() {
            BufferedImage original = createArgbTestImage(16, 16, Color.RED);
            PngImageWriter writer = new PngImageWriter();
            PngImageReader reader = new PngImageReader();

            byte[] encoded = writer.write(StaticImageData.of(original));
            ImageData decoded = reader.read(encoded);

            assertThat(decoded.getWidth(), is(16));
            assertThat(decoded.getHeight(), is(16));
            assertThat(decoded.isAnimated(), is(false));
        }

        @Test
        void pngPreservesAlpha() {
            BufferedImage original = createArgbTestImage(4, 4, new Color(255, 0, 0, 128));
            PngImageWriter writer = new PngImageWriter();
            PngImageReader reader = new PngImageReader();

            byte[] encoded = writer.write(StaticImageData.of(original));
            ImageData decoded = reader.read(encoded);

            int pixel = decoded.toBufferedImage().getRGB(0, 0);
            int alpha = (pixel >> 24) & 0xFF;
            assertThat(alpha, is(128));
        }

        @Test
        void jpegRoundTrip() {
            BufferedImage original = createRgbTestImage(16, 16, Color.BLUE);
            JpegImageWriter writer = new JpegImageWriter();
            JpegImageReader reader = new JpegImageReader();

            byte[] encoded = writer.write(StaticImageData.of(original));
            ImageData decoded = reader.read(encoded);

            assertThat(decoded.getWidth(), is(16));
            assertThat(decoded.getHeight(), is(16));
        }

        @Test
        void jpegStripsAlpha() {
            BufferedImage original = createArgbTestImage(8, 8, new Color(255, 0, 0, 128));
            JpegImageWriter writer = new JpegImageWriter();

            byte[] encoded = writer.write(StaticImageData.of(original));
            assertThat(ImageFormat.JPEG.matches(encoded), is(true));
        }

        @Test
        void jpegQualityOption() {
            BufferedImage original = createRgbTestImage(32, 32, Color.GREEN);
            JpegImageWriter writer = new JpegImageWriter();

            byte[] highQ = writer.write(StaticImageData.of(original), JpegWriteOptions.builder().withQuality(0.95f).build());
            byte[] lowQ = writer.write(StaticImageData.of(original), JpegWriteOptions.builder().withQuality(0.1f).build());

            assertThat(highQ.length, greaterThan(lowQ.length));
        }

        @Test
        void bmpRoundTrip() {
            BufferedImage original = createRgbTestImage(8, 8, Color.MAGENTA);
            BmpImageWriter writer = new BmpImageWriter();
            BmpImageReader reader = new BmpImageReader();

            byte[] encoded = writer.write(StaticImageData.of(original));
            ImageData decoded = reader.read(encoded);

            assertThat(decoded.getWidth(), is(8));
            assertThat(decoded.getHeight(), is(8));
            // BMP is lossless for RGB - pixel values should match
            assertThat(decoded.toBufferedImage().getRGB(4, 4) | 0xFF000000,
                is(original.getRGB(4, 4) | 0xFF000000));
        }

    }

    // ──── GIF ────

    @Nested
    class GifTests {

        @Test
        void staticGifRoundTrip() {
            BufferedImage original = createRgbTestImage(8, 8, Color.RED);
            GifImageWriter writer = new GifImageWriter();
            GifImageReader reader = new GifImageReader();

            byte[] encoded = writer.write(StaticImageData.of(original));
            ImageData decoded = reader.read(encoded);

            assertThat(decoded.getWidth(), is(8));
            assertThat(decoded.getHeight(), is(8));
        }

        @Test
        void animatedGifRoundTrip() {
            BufferedImage frame1 = createRgbTestImage(8, 8, Color.RED);
            BufferedImage frame2 = createRgbTestImage(8, 8, Color.GREEN);
            BufferedImage frame3 = createRgbTestImage(8, 8, Color.BLUE);

            AnimatedImageData animated = AnimatedImageData.builder()
                .withFrame(ImageFrame.of(PixelBuffer.wrap(frame1), 100))
                .withFrame(ImageFrame.of(PixelBuffer.wrap(frame2), 200))
                .withFrame(ImageFrame.of(PixelBuffer.wrap(frame3), 150))
                .build();

            GifImageWriter writer = new GifImageWriter();
            GifImageReader reader = new GifImageReader();

            byte[] encoded = writer.write(animated);
            ImageData decoded = reader.read(encoded);

            assertThat(decoded.isAnimated(), is(true));
            assertThat(decoded.getFrames(), hasSize(3));
        }

        @Test
        void gifLoopCountPreserved() {
            BufferedImage frame1 = createRgbTestImage(4, 4, Color.RED);
            BufferedImage frame2 = createRgbTestImage(4, 4, Color.BLUE);

            AnimatedImageData animated = AnimatedImageData.builder()
                .withFrame(ImageFrame.of(PixelBuffer.wrap(frame1), 100))
                .withFrame(ImageFrame.of(PixelBuffer.wrap(frame2), 100))
                .withLoopCount(5)
                .build();

            GifImageWriter writer = new GifImageWriter();
            GifImageReader reader = new GifImageReader();

            GifWriteOptions options = GifWriteOptions.builder().withLoopCount(5).build();
            byte[] encoded = writer.write(animated, options);
            ImageData decoded = reader.read(encoded);

            assertThat(decoded.isAnimated(), is(true));
            assertThat(((AnimatedImageData) decoded).getLoopCount(), is(5));
        }

    }

    // ──── Format Conversion ────

    @Nested
    class FormatConversionTests {

        @Test
        void pngToBmpConversion() {
            BufferedImage original = createRgbTestImage(8, 8, Color.ORANGE);
            PngImageWriter pngWriter = new PngImageWriter();
            PngImageReader pngReader = new PngImageReader();
            BmpImageWriter bmpWriter = new BmpImageWriter();
            BmpImageReader bmpReader = new BmpImageReader();

            // Write as PNG, read back
            byte[] pngBytes = pngWriter.write(StaticImageData.of(original));
            ImageData fromPng = pngReader.read(pngBytes);

            // Write the PNG data as BMP
            byte[] bmpBytes = bmpWriter.write(fromPng);
            ImageData fromBmp = bmpReader.read(bmpBytes);

            assertThat(fromBmp.getWidth(), is(8));
            assertThat(fromBmp.getHeight(), is(8));
        }

    }

    // ──── ImageFactory ────

    @Nested
    class ImageFactoryTests {

        private final ImageFactory factory = new ImageFactory();

        @Test
        void fromImageWrapsBufferedImage() {
            BufferedImage image = createArgbTestImage(4, 4, Color.CYAN);
            StaticImageData data = factory.fromImage(image);

            assertThat(data.getWidth(), is(4));
            assertThat(data.isAnimated(), is(false));
        }

        @Test
        void fromImagesCreatesAnimated() {
            ConcurrentList<BufferedImage> images = Concurrent.newList();
            images.add(createArgbTestImage(4, 4, Color.RED));
            images.add(createArgbTestImage(4, 4, Color.BLUE));

            AnimatedImageData data = factory.fromImages(images, 100);

            assertThat(data.isAnimated(), is(true));
            assertThat(data.getFrames(), hasSize(2));
            assertThat(data.getFrames().get(0).delayMs(), is(100));
        }

        @Test
        void detectFormatFromPngBytes() {
            BufferedImage image = createRgbTestImage(4, 4, Color.RED);
            byte[] pngBytes = new PngImageWriter().write(StaticImageData.of(image));

            ImageFormat format = factory.detectFormat(pngBytes);
            assertThat(format, is(ImageFormat.PNG));
        }

        @Test
        void toByteArrayAndFromByteArrayRoundTrip() {
            BufferedImage original = createRgbTestImage(8, 8, Color.GREEN);
            StaticImageData data = StaticImageData.of(original);

            byte[] encoded = factory.toByteArray(data, ImageFormat.PNG);
            ImageData decoded = factory.fromByteArray(encoded);

            assertThat(decoded.getWidth(), is(8));
            assertThat(decoded.getHeight(), is(8));
        }

        @Test
        void toBase64AndFromBase64RoundTrip() {
            BufferedImage original = createRgbTestImage(4, 4, Color.BLUE);
            StaticImageData data = StaticImageData.of(original);

            String base64 = factory.toBase64(data, ImageFormat.PNG);
            ImageData decoded = factory.fromBase64(base64);

            assertThat(decoded.getWidth(), is(4));
            assertThat(decoded.getHeight(), is(4));
        }

    }

}
