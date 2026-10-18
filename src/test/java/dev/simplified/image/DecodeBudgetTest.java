package dev.simplified.image;

import dev.simplified.image.exception.ImageTooLargeException;
import dev.simplified.image.frame.ImageFrame;
import dev.simplified.image.option.WebPWriteOptions;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.stream.Stream;
import java.util.zip.CRC32;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Pins the pixel budget every read holds to, {@link ImageFactory#getMaxPixels()}: each reader decodes
 * an image of exactly the budget and refuses one pixel more, a read's frames, pages and canvases
 * count together, and a header asking for more than the default budget is refused with
 * {@link ImageTooLargeException} before its pixels are allocated.
 * <p>
 * The budget is process-wide, so each test puts the default back after it, and each case of a test
 * factory sets the budget it reads under.
 */
class DecodeBudgetTest {

    /** The side of the still each reader's boundary is found on. */
    private static final int SIDE = 16;

    /** The side of each frame of the animation whose frames count together. */
    private static final int FRAME_SIDE = 8;

    @AfterEach
    void restoreTheDefaultBudget() {
        ImageFactory.setMaxPixels(ImageFactory.DEFAULT_MAX_PIXELS);
    }

    // ──── the budget ────

    @Test
    @DisplayName("the default budget is 8192 by 8192 pixels")
    void defaultIs8192Squared() {
        assertThat(ImageFactory.DEFAULT_MAX_PIXELS, is(8192L * 8192L));
        assertThat(ImageFactory.getMaxPixels(), is(ImageFactory.DEFAULT_MAX_PIXELS));
    }

    @Test
    @DisplayName("a budget that is not positive is refused and the budget stays as it was")
    void nonPositiveBudgetIsRefused() {
        assertThrows(IllegalArgumentException.class, () -> ImageFactory.setMaxPixels(0));
        assertThrows(IllegalArgumentException.class, () -> ImageFactory.setMaxPixels(-1));
        assertThat(ImageFactory.getMaxPixels(), is(ImageFactory.DEFAULT_MAX_PIXELS));
    }

    @Test
    @DisplayName("Long.MAX_VALUE lifts the budget, frames counted together included")
    void maxValueLiftsTheBudget() {
        byte[] webp = ImageFactory.toByteArray(animation(), ImageFormat.WEBP);
        ImageFactory.setMaxPixels(Long.MAX_VALUE);

        assertThat(ImageFactory.fromByteArray(webp, ImageFormat.WEBP).getFrames().size(), is(2));
    }

    // ──── each reader's boundary ────

    @TestFactory
    @DisplayName("every reader decodes a still of exactly the budget and refuses one pixel more")
    Stream<DynamicTest> everyReaderHoldsAStillToTheBudget() {
        return Arrays.stream(ImageFormat.values())
            .map(format -> DynamicTest.dynamicTest(format.name(), () ->
                assertBoundary(ImageFactory.toByteArray(still(), format), format, SIDE * SIDE)
            ));
    }

    @Test
    @DisplayName("a lossless WebP still is held to the budget by its VP8L decoder")
    void losslessWebPStill() {
        byte[] webp = ImageFactory.toByteArray(still(), ImageFormat.WEBP, WebPWriteOptions.builder().isLossless(true).build());

        assertThat(firstChunk(webp), is("VP8L"));
        assertBoundary(webp, ImageFormat.WEBP, SIDE * SIDE);
    }

    @Test
    @DisplayName("a lossy WebP still is held to the budget by its VP8 decoder")
    void lossyWebPStill() {
        byte[] webp = ImageFactory.toByteArray(still(), ImageFormat.WEBP, WebPWriteOptions.builder().isLossless(false).build());

        assertThat(firstChunk(webp), is("VP8 "));
        assertBoundary(webp, ImageFormat.WEBP, SIDE * SIDE);
    }

    @Test
    @DisplayName("an ICO whose subimage is a bitmap rather than a PNG is held to the budget")
    void icoBitmapSubimage() {
        assertBoundary(icoBitmap(4, 4), ImageFormat.ICO, 16);
    }

    // ──── what a read counts together ────

    @TestFactory
    @DisplayName("an animation's frames count together")
    Stream<DynamicTest> framesCountTogether() {
        return Stream.of(ImageFormat.GIF, ImageFormat.TIFF)
            .map(format -> DynamicTest.dynamicTest(format.name(), () ->
                assertBoundary(ImageFactory.toByteArray(animation(), format), format, 2 * FRAME_SIDE * FRAME_SIDE)
            ));
    }

    @Test
    @DisplayName("an animated WebP holds its canvas once per frame and once more to compose on")
    void animatedWebPHoldsItsCanvasPerFrame() {
        assertBoundary(ImageFactory.toByteArray(animation(), ImageFormat.WEBP), ImageFormat.WEBP, 3 * FRAME_SIDE * FRAME_SIDE);
    }

    @Test
    @DisplayName("a still GIF's frame and the screen it is laid on count together")
    void stillGifFrameAndScreenCountTogether() {
        assertBoundary(gif(10, 10, 1, 1), ImageFormat.GIF, 1 + 10 * 10);
    }

    // ──── a header asking for more than the default ────

    @Test
    @DisplayName("a 41-byte GIF declaring a 16383 by 16383 screen is refused")
    void gifScreen() {
        byte[] gif = gif(16383, 16383, 1, 1);

        assertThat(gif.length, is(41));
        assertRefused(gif, ImageFormat.GIF, "GIF declares a '16383x16383' canvas");
    }

    @Test
    @DisplayName("a GIF frame larger than the budget is refused before it is read")
    void gifFrame() {
        assertRefused(gif(1, 1, 9000, 9000), ImageFormat.GIF, "GIF needs '9000x9000' pixels");
    }

    @Test
    @DisplayName("the 19-byte PNM header P6 40000 40000 255 is refused")
    void pnmHeader() {
        byte[] pnm = "P6 40000 40000 255\n".getBytes(StandardCharsets.US_ASCII);

        assertThat(pnm.length, is(19));
        assertRefused(pnm, ImageFormat.PNM, "PNM needs '40000x40000' pixels");
    }

    @Test
    @DisplayName("a QOI header declaring 9000 by 9000 pixels is refused")
    void qoiHeader() {
        ByteBuffer qoi = ByteBuffer.allocate(22);
        qoi.put("qoif".getBytes(StandardCharsets.US_ASCII)).putInt(9000).putInt(9000).put((byte) 4).put((byte) 0);
        qoi.put(new byte[] { 0, 0, 0, 0, 0, 0, 0, 1 });

        assertRefused(qoi.array(), ImageFormat.QOI, "QOI needs '9000x9000' pixels");
    }

    @Test
    @DisplayName("a TGA header declaring 9000 by 9000 pixels is refused")
    void tgaHeader() {
        ByteBuffer tga = ByteBuffer.allocate(18).order(ByteOrder.LITTLE_ENDIAN);
        tga.put((byte) 0).put((byte) 0).put((byte) 2).put(new byte[9]); // no id, no colour map, true colour
        tga.putShort((short) 9000).putShort((short) 9000).put((byte) 32).put((byte) 0x28);

        assertRefused(tga.array(), ImageFormat.TGA, "TGA needs '9000x9000' pixels");
    }

    @Test
    @DisplayName("a PNG whose IHDR declares 10000 by 10000 pixels is refused")
    void pngHeader() {
        byte[] png = ImageFactory.toByteArray(still(), ImageFormat.PNG);
        assertThat(new String(png, 12, 4, StandardCharsets.US_ASCII), is("IHDR"));

        ByteBuffer.wrap(png).putInt(16, 10000).putInt(20, 10000);
        CRC32 crc = new CRC32();
        crc.update(png, 12, 17);
        ByteBuffer.wrap(png).putInt(29, (int) crc.getValue());

        assertRefused(png, ImageFormat.PNG, "PNG needs '10000x10000' pixels");
    }

    @Test
    @DisplayName("a JPEG whose frame header declares 10000 by 10000 pixels is refused")
    void jpegHeader() {
        byte[] jpeg = ImageFactory.toByteArray(still(), ImageFormat.JPEG);
        int position = 2;

        // Walk the marker segments to the start of frame, SOF0 to SOF2
        while ((jpeg[position + 1] & 0xFF) < 0xC0 || (jpeg[position + 1] & 0xFF) > 0xC2)
            position += 2 + ((jpeg[position + 2] & 0xFF) << 8 | (jpeg[position + 3] & 0xFF));

        ByteBuffer.wrap(jpeg).putShort(position + 5, (short) 10000).putShort(position + 7, (short) 10000);

        assertRefused(jpeg, ImageFormat.JPEG, "JPEG needs '10000x10000' pixels");
    }

    @Test
    @DisplayName("a BMP whose info header declares 10000 by 10000 pixels is refused")
    void bmpHeader() {
        byte[] bmp = ImageFactory.toByteArray(still(), ImageFormat.BMP);

        // ImageIO holds the pixels to the file size the header declares, so that is declared too
        ByteBuffer.wrap(bmp).order(ByteOrder.LITTLE_ENDIAN).putInt(2, Integer.MAX_VALUE).putInt(18, 10000).putInt(22, 10000);

        assertRefused(bmp, ImageFormat.BMP, "BMP needs '10000x10000' pixels");
    }

    @Test
    @DisplayName("a VP8 keyframe declaring 16383 by 16383 pixels is refused")
    void vp8Header() {
        byte[] webp = ImageFactory.toByteArray(still(), ImageFormat.WEBP, WebPWriteOptions.builder().isLossless(false).build());
        assertThat(firstChunk(webp), is("VP8 "));

        // The 14-bit width and height follow the frame tag and the start code
        ByteBuffer.wrap(webp).order(ByteOrder.LITTLE_ENDIAN).putShort(26, (short) 0x3FFF).putShort(28, (short) 0x3FFF);

        assertRefused(webp, ImageFormat.WEBP, "VP8 needs '16383x16383' pixels");
    }

    @Test
    @DisplayName("a VP8L header declaring 16384 by 16384 pixels is refused")
    void vp8lHeader() {
        byte[] webp = ImageFactory.toByteArray(still(), ImageFormat.WEBP, WebPWriteOptions.builder().isLossless(true).build());
        assertThat(firstChunk(webp), is("VP8L"));

        // Width - 1 and height - 1 are the low 28 bits after the signature byte
        ByteBuffer header = ByteBuffer.wrap(webp).order(ByteOrder.LITTLE_ENDIAN);
        header.putInt(21, header.getInt(21) | 0x0FFFFFFF);

        assertRefused(webp, ImageFormat.WEBP, "VP8L needs '16384x16384' pixels");
    }

    @Test
    @DisplayName("an animated WebP canvas the budget holds once is refused once per frame")
    void animatedWebPCanvas() {
        byte[] webp = ImageFactory.toByteArray(animation(), ImageFormat.WEBP);
        assertThat(firstChunk(webp), is("VP8X"));

        // Canvas width - 1 and height - 1, 24 bits each, after the flags and three reserved bytes
        for (int offset : new int[] { 24, 27 }) {
            webp[offset] = (byte) 0xFF;
            webp[offset + 1] = (byte) 0x1F;
            webp[offset + 2] = 0;
        }

        assertRefused(webp, ImageFormat.WEBP, "WebP needs '8192x8192' pixels '3' times");
    }

    // ──── fixtures ────

    /**
     * Sets the budget to {@code pixels} and reads the image, then sets it one lower and expects the
     * read refused.
     */
    private static void assertBoundary(byte @NotNull [] bytes, @NotNull ImageFormat format, long pixels) {
        ImageFactory.setMaxPixels(pixels);
        ImageFactory.fromByteArray(bytes, format);

        ImageFactory.setMaxPixels(pixels - 1);
        assertThrows(ImageTooLargeException.class, () -> ImageFactory.fromByteArray(bytes, format));
    }

    private static void assertRefused(byte @NotNull [] bytes, @NotNull ImageFormat format, @NotNull String message) {
        ImageTooLargeException refused = assertThrows(ImageTooLargeException.class, () -> ImageFactory.fromByteArray(bytes, format));
        assertThat(refused.getMessage(), containsString(message));
    }

    private static @NotNull StaticImageData still() {
        int[] pixels = new int[SIDE * SIDE];

        for (int i = 0; i < pixels.length; i++)
            pixels[i] = 0xFF000000 | (i % SIDE * 16) << 16 | (i / SIDE * 16) << 8;

        return StaticImageData.of(PixelBuffer.of(pixels, SIDE, SIDE, false));
    }

    private static @NotNull AnimatedImageData animation() {
        return AnimatedImageData.builder()
            .withFrame(ImageFrame.of(solid(0xFFFF0000), 100))
            .withFrame(ImageFrame.of(solid(0xFF0000FF), 100))
            .build();
    }

    private static @NotNull PixelBuffer solid(int argb) {
        PixelBuffer pixels = PixelBuffer.create(FRAME_SIDE, FRAME_SIDE);
        pixels.fill(argb);
        return pixels;
    }

    private static @NotNull String firstChunk(byte @NotNull [] webp) {
        return new String(webp, 12, 4, StandardCharsets.US_ASCII);
    }

    /**
     * Builds a GIF of one frame over a four-colour global table, its screen and frame the sizes
     * given and its image data coding a single pixel, so a 1 by 1 frame makes a complete file.
     */
    private static byte @NotNull [] gif(int screenWidth, int screenHeight, int frameWidth, int frameHeight) {
        ByteBuffer gif = ByteBuffer.allocate(41).order(ByteOrder.LITTLE_ENDIAN);
        gif.put("GIF89a".getBytes(StandardCharsets.US_ASCII));
        gif.putShort((short) screenWidth).putShort((short) screenHeight).put((byte) 0x81).put((byte) 0).put((byte) 0);
        gif.put(new byte[12]);
        gif.put((byte) 0x2C).putShort((short) 0).putShort((short) 0);
        gif.putShort((short) frameWidth).putShort((short) frameHeight).put((byte) 0);
        gif.put((byte) 2).put((byte) 2).put((byte) 0x44).put((byte) 0x01).put((byte) 0); // clear, index 0, end
        gif.put((byte) 0x3B);
        return gif.array();
    }

    /**
     * Builds an ICO of one 32-bit bitmap subimage, transparent black, the size given.
     */
    private static byte @NotNull [] icoBitmap(int width, int height) {
        int pixelBytes = width * height * 4;
        ByteBuffer ico = ByteBuffer.allocate(6 + 16 + 40 + pixelBytes).order(ByteOrder.LITTLE_ENDIAN);
        ico.putShort((short) 0).putShort((short) 1).putShort((short) 1); // an icon of one subimage
        ico.put((byte) width).put((byte) height).put((byte) 0).put((byte) 0);
        ico.putShort((short) 1).putShort((short) 32).putInt(40 + pixelBytes).putInt(6 + 16);
        ico.putInt(40).putInt(width).putInt(2 * height).putShort((short) 1).putShort((short) 32);
        return ico.array();
    }

}
