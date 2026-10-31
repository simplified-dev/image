package dev.simplified.image;

import dev.simplified.annotations.UtilityClass;
import org.jetbrains.annotations.NotNull;

import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.ComponentColorModel;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferInt;
import java.awt.image.ImageObserver;
import java.awt.image.IndexColorModel;
import java.awt.image.Raster;

/**
 * Per-layout {@link BufferedImage}-to-ARGB unpackers used by
 * {@link PixelBuffer#wrap(BufferedImage)}. Each method handles one source layout and answers its
 * pixels as {@code 0xAARRGGBB} ints - packed into a fresh array, or the image's own array in the
 * zero-copy case - each channel as the image holds it, so a premultiplied image's colour stays
 * premultiplied. {@link PixelBuffer#wrap(BufferedImage)} reads the convention off the image and wraps
 * the array in it.
 * <p>
 * Package-private because the methods make assumptions about the source layout that callers
 * outside this package shouldn't have to enforce - {@link PixelBuffer#wrap(BufferedImage)} is
 * the supported entry point.
 */
@UtilityClass
class BufferedImageUnpacker {

    /**
     * Answers the backing array of an image already held as {@code 0xAARRGGBB} packed ints, straight
     * or premultiplied, shared with the image rather than copied.
     *
     * @param image the image, of type {@link BufferedImage#TYPE_INT_ARGB} or {@link BufferedImage#TYPE_INT_ARGB_PRE}
     * @return the {@code 0xAARRGGBB} pixels, each channel as the image holds it
     */
    static int @NotNull [] intArgb(@NotNull BufferedImage image) {
        return ((DataBufferInt) image.getRaster().getDataBuffer()).getData();
    }

    /**
     * Repacks an image of {@code 0x__RRGGBB} packed ints, whose alpha byte is unused, into a fresh
     * array, forcing the alpha byte to {@code 0xFF}.
     *
     * @param image the image, of type {@link BufferedImage#TYPE_INT_RGB}
     * @return the {@code 0xAARRGGBB} pixels, each channel as the image holds it and alpha {@code 0xFF}
     */
    static int @NotNull [] intRgb(@NotNull BufferedImage image) {
        int[] src = ((DataBufferInt) image.getRaster().getDataBuffer()).getData();
        int[] dst = new int[src.length];

        for (int i = 0; i < src.length; i++)
            dst[i] = (src[i] & 0x00FFFFFF) | 0xFF000000;

        return dst;
    }

    /**
     * Repacks an image of {@code 0x__BBGGRR} packed ints, whose alpha byte is unused, into a fresh
     * array, swapping the red and blue bytes into ARGB order and forcing the alpha byte to
     * {@code 0xFF}.
     *
     * @param image the image, of type {@link BufferedImage#TYPE_INT_BGR}
     * @return the {@code 0xAARRGGBB} pixels, each channel as the image holds it and alpha {@code 0xFF}
     */
    static int @NotNull [] intBgr(@NotNull BufferedImage image) {
        int[] src = ((DataBufferInt) image.getRaster().getDataBuffer()).getData();
        int[] dst = new int[src.length];

        for (int i = 0; i < src.length; i++) {
            int p = src[i];
            int r = p & 0xFF;
            int g = (p >>> 8) & 0xFF;
            int b = (p >>> 16) & 0xFF;
            dst[i] = 0xFF000000 | (r << 16) | (g << 8) | b;
        }

        return dst;
    }

    /**
     * Repacks an image held as a flat {@code byte[]} of 4 bytes per pixel in A, B, G, R order,
     * straight or premultiplied, into a fresh array, reading each byte unsigned.
     *
     * @param image the image, of type {@link BufferedImage#TYPE_4BYTE_ABGR} or {@link BufferedImage#TYPE_4BYTE_ABGR_PRE}
     * @param w the image width
     * @param h the image height
     * @return the {@code 0xAARRGGBB} pixels, each channel as the image holds it
     */
    static int @NotNull [] fourByteAbgr(@NotNull BufferedImage image, int w, int h) {
        byte[] src = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        int[] dst = new int[w * h];

        for (int i = 0, j = 0; i < dst.length; i++, j += 4) {
            int a = src[j] & 0xFF;
            int b = src[j + 1] & 0xFF;
            int g = src[j + 2] & 0xFF;
            int r = src[j + 3] & 0xFF;
            dst[i] = (a << 24) | (r << 16) | (g << 8) | b;
        }

        return dst;
    }

    /**
     * Repacks an image held as a flat {@code byte[]} of 3 bytes per pixel in B, G, R order, with no
     * alpha, into a fresh array of {@code 0xFFRRGGBB} pixels.
     *
     * @param image the image, of type {@link BufferedImage#TYPE_3BYTE_BGR}
     * @param w the image width
     * @param h the image height
     * @return the {@code 0xAARRGGBB} pixels, each channel as the image holds it and alpha {@code 0xFF}
     */
    static int @NotNull [] threeByteBgr(@NotNull BufferedImage image, int w, int h) {
        byte[] src = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        int[] dst = new int[w * h];

        for (int i = 0, j = 0; i < dst.length; i++, j += 3) {
            int b = src[j] & 0xFF;
            int g = src[j + 1] & 0xFF;
            int r = src[j + 2] & 0xFF;
            dst[i] = 0xFF000000 | (r << 16) | (g << 8) | b;
        }

        return dst;
    }

    /**
     * Resolves each palette index of an indexed image through {@link IndexColorModel#getRGB(int)} - a
     * flat lookup with no colour-space conversion - rather than letting {@code drawImage}'s compositor
     * inflate the bytes.
     *
     * @param image the image, its colour model an {@link IndexColorModel}
     * @param w the image width
     * @param h the image height
     * @return the {@code 0xAARRGGBB} pixels, each channel as the palette holds it
     */
    static int @NotNull [] indexed(@NotNull BufferedImage image, int w, int h) {
        IndexColorModel indexed = (IndexColorModel) image.getColorModel();
        Raster raster = image.getRaster();
        int[] sample = new int[1];
        int[] dst = new int[w * h];

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                raster.getPixel(x, y, sample);
                dst[y * w + x] = indexed.getRGB(sample[0] & 0xFF);
            }
        }

        return dst;
    }

    /**
     * Reads a grayscale image's raster bands directly (one band for opaque gray, two for gray and
     * alpha), replicating the gray sample across R, G and B and taking alpha from the second band when
     * there is one. Going through {@code drawImage} or {@link BufferedImage#getRGB(int, int) getRGB}
     * would apply the compositor's sRGB-gamma transform and inflate dark bytes (raw {@code 170}
     * inflates to {@code 213}).
     *
     * @param image the image, its colour model a {@link ComponentColorModel} of {@link ColorSpace#TYPE_GRAY}
     * @param w the image width
     * @param h the image height
     * @return the {@code 0xAARRGGBB} pixels, each channel as the image holds it and alpha {@code 0xFF} without an alpha band
     */
    static int @NotNull [] grayscale(@NotNull BufferedImage image, int w, int h) {
        Raster raster = image.getRaster();
        int numBands = raster.getNumBands();
        int[] sample = new int[numBands];
        int[] dst = new int[w * h];

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                raster.getPixel(x, y, sample);
                int gray = sample[0] & 0xFF;
                int a = numBands >= 2 ? sample[1] & 0xFF : 255;
                dst[y * w + x] = (a << 24) | (gray << 16) | (gray << 8) | gray;
            }
        }

        return dst;
    }

    /**
     * Reads a layout that matches no standard {@link BufferedImage#getType() getType()} code by the
     * {@link ColorModel} it carries - notably {@link BufferedImage#TYPE_BYTE_BINARY} (1/2/4-bpp packed
     * pixels with an {@link IndexColorModel}; reports {@code getType() == 12} so the
     * {@code TYPE_BYTE_INDEXED} case in {@link PixelBuffer#wrap wrap}'s switch does not match) and
     * {@link BufferedImage#TYPE_CUSTOM} with a {@link ComponentColorModel} of
     * {@link ColorSpace#TYPE_GRAY} (2-band tRNS-keyed grayscale). Anything else is drawn through
     * {@link #viaDrawImage}.
     *
     * @param image the image
     * @param w the image width
     * @param h the image height
     * @return the {@code 0xAARRGGBB} pixels, each channel as the image holds it
     */
    static int @NotNull [] byColorModel(@NotNull BufferedImage image, int w, int h) {
        ColorModel cm = image.getColorModel();

        if (cm instanceof IndexColorModel)
            return indexed(image, w, h);

        if (cm instanceof ComponentColorModel && cm.getColorSpace().getType() == ColorSpace.TYPE_GRAY)
            return grayscale(image, w, h);

        return viaDrawImage(image, w, h);
    }

    /**
     * Draws an uncommon layout (16-bit colour, a custom colour model, ...) into a fresh scratch image
     * of the source's convention with {@link Graphics2D#drawImage(Image, int, int, ImageObserver)
     * drawImage}, the AWT compositor converting it - {@link BufferedImage#TYPE_INT_ARGB_PRE} for a
     * premultiplied source, so its colour is copied as premultiplied colour rather than divided back,
     * and {@link BufferedImage#TYPE_INT_ARGB} otherwise, whose premultiply/un-premultiply roundtrip can
     * drop one LSB per channel at partial alpha but is faithful for fully opaque or fully transparent
     * pixels.
     *
     * @param image the image
     * @param w the image width
     * @param h the image height
     * @return the {@code 0xAARRGGBB} pixels, in the convention the image holds
     */
    static int @NotNull [] viaDrawImage(@NotNull BufferedImage image, int w, int h) {
        int type = image.isAlphaPremultiplied() ? BufferedImage.TYPE_INT_ARGB_PRE : BufferedImage.TYPE_INT_ARGB;
        BufferedImage argb = new BufferedImage(w, h, type);
        Graphics2D g = argb.createGraphics();

        try {
            g.drawImage(image, 0, 0, null);
        } finally {
            g.dispose();
        }

        return intArgb(argb);
    }

}
