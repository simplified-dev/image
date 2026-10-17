package dev.simplified.image.codec;

import dev.simplified.image.AnimatedImageData;
import dev.simplified.image.ImageData;
import dev.simplified.image.PixelBuffer;
import dev.simplified.image.StaticImageData;
import dev.simplified.image.codec.bmp.BmpImageWriter;
import dev.simplified.image.codec.ico.IcoImageReader;
import dev.simplified.image.codec.ico.IcoImageWriter;
import dev.simplified.image.codec.jpeg.JpegImageWriter;
import dev.simplified.image.codec.png.PngImageReader;
import dev.simplified.image.codec.png.PngImageWriter;
import dev.simplified.image.codec.pnm.PnmImageReader;
import dev.simplified.image.codec.pnm.PnmImageWriter;
import dev.simplified.image.codec.qoi.QoiImageReader;
import dev.simplified.image.codec.qoi.QoiImageWriter;
import dev.simplified.image.codec.tga.TgaImageReader;
import dev.simplified.image.codec.tga.TgaImageWriter;
import dev.simplified.image.codec.tiff.TiffImageWriter;
import dev.simplified.image.codec.webp.WebPImageReader;
import dev.simplified.image.codec.webp.WebPImageWriter;
import dev.simplified.image.frame.FrameBlend;
import dev.simplified.image.frame.FrameDisposal;
import dev.simplified.image.frame.ImageFrame;
import dev.simplified.image.option.WebPWriteOptions;
import dev.simplified.image.pixel.AlphaMode;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import javax.imageio.stream.ImageInputStream;
import java.awt.image.Raster;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.instanceOf;
import static org.hamcrest.Matchers.lessThanOrEqualTo;

/**
 * Pins that every writer but GIF's stores the straight colour a premultiplied image stands for, so a
 * decoder that multiplies by alpha multiplies once.
 * <p>
 * Every fixture here holds valid premultiplied pixels, no channel above its alpha, and so converts to
 * straight colour and back without loss: a lossless file decoded and premultiplied again is the buffer
 * that was written.
 */
class PremultipliedEncodeTest {

    /** One premultiplied pixel per case the conversion treats differently, alpha 0 and 255 included. */
    private static final int[] PREMULTIPLIED = {
        0x80800000, 0x40202020, 0x01010101, 0x00000000, 0xFF336699, 0x80404040, 0xFE7F3F00, 0x02010000
    };

    /** The straight colour each {@link #PREMULTIPLIED} pixel stands for. */
    private static final int[] STRAIGHT = {
        0x80FF0000, 0x40808080, 0x01FFFFFF, 0x00000000, 0xFF336699, 0x80808080, 0xFE803F00, 0x02800000
    };

    /** A premultiplied pixel the animation fixture changes between its frames, and its straight colour. */
    private static final int CHANGED_PREMULTIPLIED = 0xC0602000;
    private static final int CHANGED_STRAIGHT = 0xC0802B00;

    /** How far a lossy channel may land from the straight colour; the premultiplied misread is 96 off or more. */
    private static final int LOSSY_TOLERANCE = 12;

    // ──── lossless stills ────

    @Test
    @DisplayName("PNG stores straight colour")
    void pngStoresStraight() {
        assertDecodesStraight(data -> new PngImageReader().read(new PngImageWriter().write(data)).toPixelBuffer());
    }

    @Test
    @DisplayName("QOI stores straight colour")
    void qoiStoresStraight() {
        assertDecodesStraight(data -> new QoiImageReader().read(new QoiImageWriter().write(data)).toPixelBuffer());
    }

    @Test
    @DisplayName("TGA stores straight colour at 32 bits per pixel")
    void tgaStoresStraight() {
        assertDecodesStraight(data -> {
            byte[] tga = new TgaImageWriter().write(data);
            assertThat("bits per pixel", tga[16] & 0xFF, equalTo(32));
            return new TgaImageReader().read(tga).toPixelBuffer();
        });
    }

    @Test
    @DisplayName("ICO stores straight colour in its embedded PNG")
    void icoStoresStraight() {
        assertDecodesStraight(data -> new IcoImageReader().read(new IcoImageWriter().write(data)).toPixelBuffer());
    }

    @Test
    @DisplayName("WebP lossless stores straight colour")
    void webpLosslessStoresStraight() {
        assertDecodesStraight(data -> new WebPImageReader().read(new WebPImageWriter().write(data)).toPixelBuffer());
    }

    @Test
    @DisplayName("TIFF stores straight colour as unassociated alpha")
    void tiffStoresStraight() {
        // The raster's own samples, not TiffImageReader's pixels: ImageIO hands a 4-band 8-bit TIFF back
        // as TYPE_CUSTOM, and PixelBuffer.wrap reads that through drawImage, which moves partial-alpha
        // pixels by a level whatever the file holds.
        assertDecodesStraight(data -> tiffPages(new TiffImageWriter().write(data)).getFirst());
    }

    @Test
    @DisplayName("PNM stores straight colour with alpha dropped")
    void pnmStoresStraightWithoutAlpha() {
        PixelBuffer pixels = row(PREMULTIPLIED);
        PixelBuffer decoded = new PnmImageReader().read(new PnmImageWriter().write(StaticImageData.of(pixels, AlphaMode.PREMULTIPLIED))).toPixelBuffer();

        for (int x = 0; x < STRAIGHT.length; x++)
            assertThat("pixel " + x, decoded.getPixel(x, 0), equalTo(0xFF000000 | STRAIGHT[x]));

        assertThat(pixels.data(), equalTo(PREMULTIPLIED));
    }

    // ──── stills flattened over black ────

    @Test
    @DisplayName("BMP writes the premultiplied colour itself, which is that colour over black")
    void bmpWritesThePremultipliedColour() {
        assertWritesPremultipliedColourOverBlack(new BmpImageWriter());
    }

    @Test
    @DisplayName("JPEG writes the premultiplied colour itself, which is that colour over black")
    void jpegWritesThePremultipliedColour() {
        assertWritesPremultipliedColourOverBlack(new JpegImageWriter());
    }

    // ──── lossy WebP ────

    @Test
    @DisplayName("lossy WebP takes its colour from straight pixels and keeps the alpha plane exact")
    void webpLossyTakesStraightColour() {
        PixelBuffer pixels = halves(0x80800000, 0x40202020);
        ImageData data = StaticImageData.of(pixels, AlphaMode.PREMULTIPLIED);
        byte[] webp = new WebPImageWriter().write(data, WebPWriteOptions.builder().isLossless(false).withQuality(0.9f).build());

        assertLossyStraight(new WebPImageReader().read(webp).toPixelBuffer(), 0x80FF0000, 0x40808080);
        assertThat(pixels.data(), equalTo(halves(0x80800000, 0x40202020).data()));
    }

    // ──── animations ────

    @Test
    @DisplayName("a multi-page TIFF stores every page as straight colour")
    void tiffAnimationStoresStraight() {
        AnimatedImageData data = animation(AlphaMode.PREMULTIPLIED);
        List<PixelBuffer> pages = tiffPages(new TiffImageWriter().write(data));

        assertThat(pages.size(), equalTo(2));
        assertThat(pages.getFirst().data(), equalTo(frame(STRAIGHT, false, CHANGED_STRAIGHT).data()));
        assertThat(pages.getLast().data(), equalTo(frame(STRAIGHT, true, CHANGED_STRAIGHT).data()));
        assertAnimationUntouched(data);
    }

    @Test
    @DisplayName("a lossless animated WebP stores every partial frame as straight colour")
    void webpLosslessAnimationStoresStraight() {
        AnimatedImageData data = animation(AlphaMode.PREMULTIPLIED);
        ImageData decoded = new WebPImageReader().read(new WebPImageWriter().write(data));

        assertThat(decoded, instanceOf(AnimatedImageData.class));
        assertThat(decoded.getFrames().size(), equalTo(2));
        assertThat(decoded.getFrames().getFirst().pixels().data(), equalTo(frame(STRAIGHT, false, CHANGED_STRAIGHT).data()));
        assertThat(decoded.getFrames().getLast().pixels().data(), equalTo(frame(STRAIGHT, true, CHANGED_STRAIGHT).data()));
        assertAnimationUntouched(data);
    }

    @Test
    @DisplayName("a lossy animated WebP with P-frames takes every full frame's colour from straight pixels")
    void webpLossyPFrameAnimationTakesStraightColour() {
        AnimatedImageData data = AnimatedImageData.builder()
            .withFrame(ImageFrame.of(halves(0x80800000, 0x40202020), 100, 0, 0, FrameDisposal.RESTORE_TO_BACKGROUND, FrameBlend.SOURCE))
            .withFrame(ImageFrame.of(halves(0x40202020, 0x80800000), 100, 0, 0, FrameDisposal.RESTORE_TO_BACKGROUND, FrameBlend.SOURCE))
            .withAlphaMode(AlphaMode.PREMULTIPLIED)
            .build();
        WebPWriteOptions options = WebPWriteOptions.builder().isLossless(false).withQuality(0.9f).usePFrames(true).build();
        ImageData decoded = new WebPImageReader().read(new WebPImageWriter().write(data, options));

        assertThat(decoded.getFrames().size(), equalTo(2));
        assertLossyStraight(decoded.getFrames().getFirst().pixels(), 0x80FF0000, 0x40808080);
        assertLossyStraight(decoded.getFrames().getLast().pixels(), 0x40808080, 0x80FF0000);
        assertThat(data.getFrames().getFirst().pixels().data(), equalTo(halves(0x80800000, 0x40202020).data()));
        assertThat(data.getFrames().getLast().pixels().data(), equalTo(halves(0x40202020, 0x80800000).data()));
    }

    // ──── assertions ────

    /**
     * Writes the premultiplied row through {@code roundTrip}, which encodes and decodes it, and checks
     * that the decode holds the straight pixels exactly, that premultiplying the decode gives the row
     * back, and that the write left the row alone.
     */
    private static void assertDecodesStraight(@NotNull Function<ImageData, PixelBuffer> roundTrip) {
        PixelBuffer pixels = row(PREMULTIPLIED);
        PixelBuffer decoded = roundTrip.apply(StaticImageData.of(pixels, AlphaMode.PREMULTIPLIED));

        assertThat(decoded.data(), equalTo(STRAIGHT));

        PixelBuffer premultiplied = decoded.copy();
        premultiplied.premultiplyAlpha();
        assertThat(premultiplied.data(), equalTo(PREMULTIPLIED));
        assertThat(pixels.data(), equalTo(PREMULTIPLIED));
    }

    /**
     * Checks that {@code writer} writes the premultiplied row to the same bytes as an opaque row
     * holding the same colour bytes: a format without alpha lays a straight pixel over black, and
     * premultiplied colour is exactly that.
     */
    private static void assertWritesPremultipliedColourOverBlack(@NotNull ImageWriter writer) {
        int[] opaque = new int[PREMULTIPLIED.length];
        for (int x = 0; x < opaque.length; x++)
            opaque[x] = 0xFF000000 | PREMULTIPLIED[x];

        PixelBuffer pixels = row(PREMULTIPLIED);
        byte[] written = writer.write(StaticImageData.of(pixels, AlphaMode.PREMULTIPLIED));

        assertThat(written, equalTo(writer.write(StaticImageData.of(row(opaque)))));
        assertThat(pixels.data(), equalTo(PREMULTIPLIED));
    }

    /**
     * Checks a decoded lossy {@link #halves} fixture: every alpha byte exactly the straight pixel's,
     * and every colour channel away from the seam between the halves within {@link #LOSSY_TOLERANCE}
     * of it.
     */
    private static void assertLossyStraight(@NotNull PixelBuffer decoded, int left, int right) {
        for (int y = 0; y < 16; y++) {
            for (int x = 0; x < 32; x++) {
                int expected = x < 16 ? left : right;
                int actual = decoded.getPixel(x, y);
                String where = "pixel " + x + "," + y;
                assertThat(where + " alpha", actual >>> 24, equalTo(expected >>> 24));

                if (x < 2 || (x >= 14 && x < 18) || x >= 30)
                    continue;

                for (int shift = 0; shift <= 16; shift += 8)
                    assertThat(where + " channel " + shift, Math.abs(((actual >> shift) & 0xFF) - ((expected >> shift) & 0xFF)), lessThanOrEqualTo(LOSSY_TOLERANCE));
            }
        }
    }

    private static void assertAnimationUntouched(@NotNull AnimatedImageData data) {
        assertThat(data.getFrames().getFirst().pixels().data(), equalTo(frame(PREMULTIPLIED, false, CHANGED_PREMULTIPLIED).data()));
        assertThat(data.getFrames().getLast().pixels().data(), equalTo(frame(PREMULTIPLIED, true, CHANGED_PREMULTIPLIED).data()));
    }

    // ──── fixtures ────

    private static @NotNull PixelBuffer row(int @NotNull [] pixels) {
        return PixelBuffer.of(pixels.clone(), pixels.length, 1);
    }

    /**
     * An 8x4 frame whose every row is {@code row} turned left by the row's index, with a 2x2 block at
     * (2, 2) set to {@code changed} when {@code withBlock} is set - so the second frame of an animation
     * differs from the first only inside that block.
     */
    private static @NotNull PixelBuffer frame(int @NotNull [] row, boolean withBlock, int changed) {
        int width = row.length;
        int[] pixels = new int[width * 4];

        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < width; x++) {
                boolean block = withBlock && x >= 2 && x < 4 && y >= 2;
                pixels[y * width + x] = block ? changed : row[(x + y) % width];
            }
        }

        return PixelBuffer.of(pixels, width, 4);
    }

    private static @NotNull AnimatedImageData animation(@NotNull AlphaMode alphaMode) {
        return AnimatedImageData.builder()
            .withFrame(ImageFrame.of(frame(PREMULTIPLIED, false, CHANGED_PREMULTIPLIED), 100, 0, 0, FrameDisposal.RESTORE_TO_BACKGROUND, FrameBlend.SOURCE))
            .withFrame(ImageFrame.of(frame(PREMULTIPLIED, true, CHANGED_PREMULTIPLIED), 100, 0, 0, FrameDisposal.RESTORE_TO_BACKGROUND, FrameBlend.SOURCE))
            .withAlphaMode(alphaMode)
            .build();
    }

    /** A 32x16 buffer, {@code left} over its left half and {@code right} over its right. */
    private static @NotNull PixelBuffer halves(int left, int right) {
        int[] pixels = new int[32 * 16];

        for (int i = 0; i < pixels.length; i++)
            pixels[i] = i % 32 < 16 ? left : right;

        return PixelBuffer.of(pixels, 32, 16);
    }

    /**
     * Reads every page of a TIFF from its rasters' own samples - red, green, blue and alpha bands in
     * that order - rather than through a colour conversion.
     */
    private static @NotNull List<PixelBuffer> tiffPages(byte @NotNull [] tiff) {
        try (ImageInputStream stream = ImageIO.createImageInputStream(new ByteArrayInputStream(tiff))) {
            javax.imageio.ImageReader reader = ImageIO.getImageReadersByFormatName("tiff").next();
            reader.setInput(stream);
            List<PixelBuffer> pages = new ArrayList<>();

            for (int page = 0; page < reader.getNumImages(true); page++) {
                Raster raster = reader.read(page).getRaster();
                assertThat("bands", raster.getNumBands(), equalTo(4));
                int[] pixels = new int[raster.getWidth() * raster.getHeight()];

                for (int y = 0; y < raster.getHeight(); y++) {
                    for (int x = 0; x < raster.getWidth(); x++) {
                        pixels[y * raster.getWidth() + x] = (raster.getSample(x, y, 3) << 24)
                            | (raster.getSample(x, y, 0) << 16)
                            | (raster.getSample(x, y, 1) << 8)
                            | raster.getSample(x, y, 2);
                    }
                }

                pages.add(PixelBuffer.of(pixels, raster.getWidth(), raster.getHeight()));
            }

            reader.dispose();
            return pages;
        } catch (IOException exception) {
            throw new AssertionError("TIFF did not decode", exception);
        }
    }

}
