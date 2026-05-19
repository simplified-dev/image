package dev.simplified.image.pixel;

import lombok.experimental.UtilityClass;
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
 * {@link PixelBuffer#wrap(BufferedImage)}. Each method handles one source layout, packs the
 * raster into a fresh ARGB {@code int[]} (or reuses the existing one for the zero-copy case),
 * and returns the wrapping {@link PixelBuffer}.
 * <p>
 * Package-private because the methods make assumptions about the source layout that callers
 * outside this package shouldn't have to enforce - {@link PixelBuffer#wrap(BufferedImage)} is
 * the supported entry point.
 */
@UtilityClass
class BufferedImageUnpacker {

    /**
     * Source is already {@code 0xAARRGGBB} packed ints - share the backing array directly,
     * no copy.
     */
    static @NotNull PixelBuffer intArgb(@NotNull BufferedImage image, int w, int h, boolean alpha) {
        int[] data = ((DataBufferInt) image.getRaster().getDataBuffer()).getData();
        return PixelBuffer.of(data, w, h, alpha);
    }

    /**
     * Source is {@code 0x__RRGGBB} packed ints (alpha byte unused). Force the alpha byte to
     * {@code 0xFF} so the result is a valid {@code 0xFFRRGGBB} ARGB pixel.
     */
    static @NotNull PixelBuffer intRgb(@NotNull BufferedImage image, int w, int h, boolean alpha) {
        int[] src = ((DataBufferInt) image.getRaster().getDataBuffer()).getData();
        int[] dst = new int[src.length];

        for (int i = 0; i < src.length; i++)
            dst[i] = (src[i] & 0x00FFFFFF) | 0xFF000000;

        return PixelBuffer.of(dst, w, h, alpha);
    }

    /**
     * Source is {@code 0x__BBGGRR} packed ints (alpha byte unused). Swap the R and B bytes to
     * land in ARGB order and force the alpha byte to {@code 0xFF}.
     */
    static @NotNull PixelBuffer intBgr(@NotNull BufferedImage image, int w, int h, boolean alpha) {
        int[] src = ((DataBufferInt) image.getRaster().getDataBuffer()).getData();
        int[] dst = new int[src.length];

        for (int i = 0; i < src.length; i++) {
            int p = src[i];
            int r = p & 0xFF;
            int g = (p >>> 8) & 0xFF;
            int b = (p >>> 16) & 0xFF;
            dst[i] = 0xFF000000 | (r << 16) | (g << 8) | b;
        }

        return PixelBuffer.of(dst, w, h, alpha);
    }

    /**
     * Source is a flat {@code byte[]} with 4 bytes per pixel in A, B, G, R order. Read them as
     * unsigned and repack into {@code 0xAARRGGBB}.
     */
    static @NotNull PixelBuffer fourByteAbgr(@NotNull BufferedImage image, int w, int h, boolean alpha) {
        byte[] src = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        int[] dst = new int[w * h];

        for (int i = 0, j = 0; i < dst.length; i++, j += 4) {
            int a = src[j] & 0xFF;
            int b = src[j + 1] & 0xFF;
            int g = src[j + 2] & 0xFF;
            int r = src[j + 3] & 0xFF;
            dst[i] = (a << 24) | (r << 16) | (g << 8) | b;
        }

        return PixelBuffer.of(dst, w, h, alpha);
    }

    /**
     * Source is a flat {@code byte[]} with 3 bytes per pixel in B, G, R order (no alpha). Repack
     * into {@code 0xFFRRGGBB}.
     */
    static @NotNull PixelBuffer threeByteBgr(@NotNull BufferedImage image, int w, int h, boolean alpha) {
        byte[] src = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        int[] dst = new int[w * h];

        for (int i = 0, j = 0; i < dst.length; i++, j += 3) {
            int b = src[j] & 0xFF;
            int g = src[j + 1] & 0xFF;
            int r = src[j + 2] & 0xFF;
            dst[i] = 0xFF000000 | (r << 16) | (g << 8) | b;
        }

        return PixelBuffer.of(dst, w, h, alpha);
    }

    /**
     * Source pixel sample is a palette index; resolve each index through
     * {@link IndexColorModel#getRGB(int)} - a flat lookup with no colour-space conversion -
     * rather than letting {@code drawImage}'s compositor inflate the bytes.
     */
    static @NotNull PixelBuffer indexed(@NotNull BufferedImage image, int w, int h, boolean alpha) {
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

        return PixelBuffer.of(dst, w, h, alpha);
    }

    /**
     * Reads the grayscale raster's bands directly (one band for opaque gray, two for
     * gray+alpha), replicating the gray sample across R, G, B and pulling alpha from the
     * second band when present. Going through {@code drawImage} or
     * {@link BufferedImage#getRGB(int, int) getRGB} would apply the compositor's sRGB-gamma
     * transform and inflate dark bytes (raw {@code 170} inflates to {@code 213}).
     */
    static @NotNull PixelBuffer grayscale(@NotNull BufferedImage image, int w, int h, boolean alpha) {
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

        return PixelBuffer.of(dst, w, h, alpha);
    }

    /**
     * Catches layouts that don't match a standard {@link BufferedImage#getType() getType()} code
     * but still carry a recognised {@link ColorModel} - notably
     * {@link BufferedImage#TYPE_BYTE_BINARY} (1/2/4-bpp packed pixels with an
     * {@link IndexColorModel}; reports {@code getType() == 12} so the {@code TYPE_BYTE_INDEXED}
     * case in {@link PixelBuffer#wrap wrap}'s switch does not match) and
     * {@link BufferedImage#TYPE_CUSTOM} with a {@link ComponentColorModel} of
     * {@link ColorSpace#TYPE_GRAY} (2-band tRNS-keyed grayscale). Anything else falls through
     * to {@link #viaDrawImage}.
     */
    static @NotNull PixelBuffer byColorModel(@NotNull BufferedImage image, int w, int h, boolean alpha) {
        ColorModel cm = image.getColorModel();

        if (cm instanceof IndexColorModel)
            return indexed(image, w, h, alpha);

        if (cm instanceof ComponentColorModel && cm.getColorSpace().getType() == ColorSpace.TYPE_GRAY)
            return grayscale(image, w, h, alpha);

        return viaDrawImage(image, w, h, alpha);
    }

    /**
     * Final fallback for uncommon layouts (alpha-premultiplied, 16-bit colour, custom 4-band
     * colour models, ...): rely on the AWT compositor to convert into a fresh
     * {@link BufferedImage#TYPE_INT_ARGB} scratch image via
     * {@link Graphics2D#drawImage(Image, int, int, ImageObserver) drawImage}. The
     * premultiply/un-premultiply roundtrip can drop one LSB per channel at partial alpha but
     * is faithful for fully opaque or fully transparent pixels.
     */
    static @NotNull PixelBuffer viaDrawImage(@NotNull BufferedImage image, int w, int h, boolean alpha) {
        BufferedImage argb = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = argb.createGraphics();

        try {
            g.drawImage(image, 0, 0, null);
        } finally {
            g.dispose();
        }

        int[] data = ((DataBufferInt) argb.getRaster().getDataBuffer()).getData();
        return PixelBuffer.of(data, w, h, alpha);
    }

}
