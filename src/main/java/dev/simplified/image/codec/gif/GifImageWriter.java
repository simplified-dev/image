package dev.simplified.image.codec.gif;

import dev.simplified.image.data.AnimatedImageData;
import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.data.ImageFrame;
import dev.simplified.image.codec.ImageWriteOptions;
import dev.simplified.image.codec.ImageWriter;
import dev.simplified.stream.ByteArrayDataOutput;
import lombok.Cleanup;
import lombok.SneakyThrows;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import javax.imageio.IIOImage;
import javax.imageio.ImageIO;
import javax.imageio.ImageTypeSpecifier;
import javax.imageio.ImageWriteParam;
import javax.imageio.metadata.IIOMetadata;
import javax.imageio.metadata.IIOMetadataNode;
import javax.imageio.stream.ImageOutputStream;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.awt.image.IndexColorModel;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Writes GIF images (static and animated) via {@link ImageIO}, configuring
 * per-frame disposal, delay, and loop count metadata.
 */
public class GifImageWriter implements ImageWriter {

    @Override
    public @NotNull ImageFormat getFormat() {
        return ImageFormat.GIF;
    }

    @Override
    @SneakyThrows
    public byte @NotNull [] write(@NotNull ImageData data, @Nullable ImageWriteOptions options) {
        int loopCount = 0;
        boolean transparent = false;
        int transparentColorIndex = 0;
        int backgroundRgb = 0x000000;
        int alphaThreshold = 128;

        if (options instanceof GifWriteOptions gifOptions) {
            loopCount = gifOptions.loopCount();
            transparent = gifOptions.transparent();
            transparentColorIndex = gifOptions.transparentColorIndex();
            backgroundRgb = gifOptions.backgroundRgb();
            alphaThreshold = gifOptions.alphaThreshold();
        } else if (data instanceof AnimatedImageData animated) {
            loopCount = animated.getLoopCount();
        }

        // Flatten every source frame onto a solid background color up-front. GIF supports
        // only 1-bit transparency, so passing partial-alpha pixels to the indexed writer
        // makes Graphics2D dither the alpha itself - producing the RGB-checkerboard effect
        // on every non-opaque pixel (the tooltip's 0xF0-alpha background was the most
        // obvious victim). Flattening first turns the render into solid colors that the
        // palette quantizer can then handle cleanly.
        int[] flattenedPixels = flattenAllFrames(data, backgroundRgb, transparent, alphaThreshold);

        javax.imageio.ImageWriter writer = ImageIO.getImageWritersByFormatName("gif").next();
        @Cleanup ByteArrayDataOutput dataOutput = new ByteArrayDataOutput();
        @Cleanup ImageOutputStream outputStream = ImageIO.createImageOutputStream(dataOutput);
        writer.setOutput(outputStream);

        ImageWriteParam param = writer.getDefaultWriteParam();

        // Build a palette shared across every frame so the GIF doesn't re-quantize each
        // frame independently. The default Java GIF writer would otherwise remap pixels to
        // its built-in 216-color web-safe palette, which shifts every non-web-safe color
        // (including the vanilla tooltip background 0x100010) to the nearest web-safe
        // neighbor and produces the "disco" shimmer across animated frames.
        IndexColorModel palette = buildSharedPalette(flattenedPixels, transparent, transparentColorIndex);
        ImageTypeSpecifier specifier = new ImageTypeSpecifier(palette,
            palette.createCompatibleSampleModel(1, 1));

        if (data.isAnimated()) {
            writer.prepareWriteSequence(null);
            boolean firstFrame = true;
            int frameIdx = 0;
            int frameStride = 0;
            for (ImageFrame frame : data.getFrames())
                frameStride = Math.max(frameStride, frame.pixels().width() * frame.pixels().height());

            for (ImageFrame frame : data.getFrames()) {
                IIOMetadata metadata = writer.getDefaultImageMetadata(specifier, param);
                configureFrameMetadata(metadata, frame, transparent, transparentColorIndex);

                if (firstFrame) {
                    configureLoopMetadata(metadata, loopCount);
                    firstFrame = false;
                }

                BufferedImage flatFrame = buildFrameImage(flattenedPixels, frameIdx * frameStride, frame.pixels().width(), frame.pixels().height());
                writer.writeToSequence(new IIOImage(toIndexed(flatFrame, palette), null, metadata), param);
                frameIdx++;
            }

            writer.endWriteSequence();
        } else {
            ImageFrame frame = data.getFrames().getFirst();
            IIOMetadata metadata = writer.getDefaultImageMetadata(specifier, param);
            configureFrameMetadata(metadata, frame, transparent, transparentColorIndex);
            BufferedImage flatFrame = buildFrameImage(flattenedPixels, 0, frame.pixels().width(), frame.pixels().height());
            writer.write(new IIOImage(toIndexed(flatFrame, palette), null, metadata));
        }

        writer.dispose();
        outputStream.flush();
        return dataOutput.toByteArray();
    }

    /**
     * Composites every frame's ARGB pixels onto a solid background and returns a single
     * contiguous int array containing all frames in order. Pixels with alpha strictly
     * below {@code alphaThreshold} become transparent-marker pixels (ARGB {@code 0x00000000});
     * everything else is blended onto {@code backgroundRgb} and emitted fully opaque
     * ({@code 0xFF......}).
     */
    private static int @NotNull [] flattenAllFrames(
        @NotNull ImageData data,
        int backgroundRgb,
        boolean transparent,
        int alphaThreshold
    ) {
        int bgR = (backgroundRgb >> 16) & 0xFF;
        int bgG = (backgroundRgb >> 8) & 0xFF;
        int bgB = backgroundRgb & 0xFF;
        int totalPixels = 0;
        for (ImageFrame frame : data.getFrames())
            totalPixels += frame.pixels().pixels().length;
        int[] out = new int[totalPixels];
        int offset = 0;
        for (ImageFrame frame : data.getFrames()) {
            int[] src = frame.pixels().pixels();
            for (int i = 0; i < src.length; i++) {
                int argb = src[i];
                int a = (argb >>> 24) & 0xFF;
                if (transparent && a < alphaThreshold) {
                    out[offset + i] = 0x00000000;
                    continue;
                }
                if (a >= 255) {
                    out[offset + i] = 0xFF000000 | (argb & 0xFFFFFF);
                    continue;
                }
                int r = (argb >> 16) & 0xFF;
                int g = (argb >> 8) & 0xFF;
                int b = argb & 0xFF;
                int outR = (r * a + bgR * (255 - a)) / 255;
                int outG = (g * a + bgG * (255 - a)) / 255;
                int outB = (b * a + bgB * (255 - a)) / 255;
                out[offset + i] = 0xFF000000 | (outR << 16) | (outG << 8) | outB;
            }
            offset += src.length;
        }
        return out;
    }

    /**
     * Wraps a slice of the flattened pixel array as an ARGB {@link BufferedImage}.
     * Shares the int array - no per-frame copy.
     */
    @NotNull
    private static BufferedImage buildFrameImage(int @NotNull [] flatPixels, int offset, int width, int height) {
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        img.setRGB(0, 0, width, height, flatPixels, offset, width);
        return img;
    }

    /**
     * Builds a single {@link IndexColorModel} shared across every frame from the already
     * alpha-flattened pixel array.
     * <p>
     * When the combined pixel set fits in 256 (or 255 when a transparent slot is reserved)
     * an exact palette is returned - useful for UI renders like tooltips which typically
     * stay well under the cap and deserve pixel-accurate color preservation. When there
     * are too many unique colors a simple uniform 6×6×6 RGB cube plus grayscale palette is
     * used; this is a minimal fallback, not a perceptual quantizer.
     */
    @NotNull
    private static IndexColorModel buildSharedPalette(
        int @NotNull [] flattenedPixels,
        boolean transparent,
        int transparentColorIndex
    ) {
        int capacity = transparent ? 255 : 256;
        Set<Integer> uniqueRgb = new LinkedHashSet<>();
        for (int argb : flattenedPixels) {
            // Skip the transparent marker (ARGB 0x00000000) - the reserved transparent
            // palette slot handles it, it shouldn't consume a color slot.
            if (transparent && (argb >>> 24) == 0) continue;
            uniqueRgb.add(argb & 0xFFFFFF);
            if (uniqueRgb.size() > capacity) break;
        }

        if (uniqueRgb.size() <= capacity)
            return exactPalette(uniqueRgb, transparent, transparentColorIndex);

        return webSafeFallbackPalette(transparent, transparentColorIndex);
    }

    @NotNull
    private static IndexColorModel exactPalette(
        @NotNull Set<Integer> uniqueRgb,
        boolean transparent,
        int transparentColorIndex
    ) {
        int n = uniqueRgb.size() + (transparent ? 1 : 0);
        byte[] r = new byte[n];
        byte[] g = new byte[n];
        byte[] b = new byte[n];
        int i = 0;
        for (int c : uniqueRgb) {
            // Reserve the caller-specified transparent slot.
            if (transparent && i == transparentColorIndex) i++;
            r[i] = (byte) ((c >> 16) & 0xFF);
            g[i] = (byte) ((c >> 8) & 0xFF);
            b[i] = (byte) (c & 0xFF);
            i++;
        }

        if (transparent)
            return new IndexColorModel(8, n, r, g, b, transparentColorIndex);

        return new IndexColorModel(8, n, r, g, b);
    }

    @NotNull
    private static IndexColorModel webSafeFallbackPalette(boolean transparent, int transparentColorIndex) {
        byte[] r = new byte[256];
        byte[] g = new byte[256];
        byte[] b = new byte[256];
        int i = 0;
        // 6*6*6 = 216 web-safe colors
        for (int rs = 0; rs < 6; rs++)
            for (int gs = 0; gs < 6; gs++)
                for (int bs = 0; bs < 6; bs++) {
                    r[i] = (byte) (rs * 51);
                    g[i] = (byte) (gs * 51);
                    b[i] = (byte) (bs * 51);
                    i++;
                }
        // Fill remaining 40 slots with grayscale ramp for shadow/antialias tones.
        for (int step = 0; i < 256; i++, step++) {
            int v = Math.min(255, step * 7);
            r[i] = (byte) v;
            g[i] = (byte) v;
            b[i] = (byte) v;
        }

        if (transparent)
            return new IndexColorModel(8, 256, r, g, b, transparentColorIndex);

        return new IndexColorModel(8, 256, r, g, b);
    }

    /**
     * Renders {@code source} onto a {@link BufferedImage#TYPE_BYTE_INDEXED} image backed by
     * {@code palette}. Java's {@link Graphics2D#drawImage} performs the ARGB→indexed color
     * mapping, dithering partial-alpha glyph edges onto the nearest palette color.
     */
    @NotNull
    private static BufferedImage toIndexed(@NotNull BufferedImage source, @NotNull IndexColorModel palette) {
        BufferedImage indexed = new BufferedImage(
            source.getWidth(),
            source.getHeight(),
            BufferedImage.TYPE_BYTE_INDEXED,
            palette
        );
        Graphics2D g = indexed.createGraphics();
        try {
            g.setRenderingHint(RenderingHints.KEY_DITHERING, RenderingHints.VALUE_DITHER_ENABLE);
            g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
            g.drawImage(source, 0, 0, null);
        } finally {
            g.dispose();
        }
        return indexed;
    }

    @SneakyThrows
    private static void configureFrameMetadata(
        @NotNull IIOMetadata metadata,
        @NotNull ImageFrame frame,
        boolean transparent,
        int transparentColorIndex
    ) {
        String formatName = metadata.getNativeMetadataFormatName();
        IIOMetadataNode root = (IIOMetadataNode) metadata.getAsTree(formatName);

        IIOMetadataNode gce = getOrCreateNode(root, "GraphicControlExtension");
        gce.setAttribute("disposalMethod", frame.disposal().getMethod());
        gce.setAttribute("userInputFlag", "FALSE");
        gce.setAttribute("transparentColorFlag", String.valueOf(transparent));
        gce.setAttribute("delayTime", Integer.toString(Math.max(1, frame.delayMs() / 10)));
        gce.setAttribute("transparentColorIndex", Integer.toString(transparentColorIndex));

        metadata.setFromTree(formatName, root);
    }

    @SneakyThrows
    private static void configureLoopMetadata(@NotNull IIOMetadata metadata, int loopCount) {
        String formatName = metadata.getNativeMetadataFormatName();
        IIOMetadataNode root = (IIOMetadataNode) metadata.getAsTree(formatName);

        IIOMetadataNode appExts = getOrCreateNode(root, "ApplicationExtensions");
        IIOMetadataNode netscape = new IIOMetadataNode("ApplicationExtension");
        netscape.setAttribute("applicationID", "NETSCAPE");
        netscape.setAttribute("authenticationCode", "2.0");
        netscape.setUserObject(new byte[] {
            0x1,
            (byte) (loopCount & 0xFF),
            (byte) ((loopCount >> 8) & 0xFF)
        });
        appExts.appendChild(netscape);

        metadata.setFromTree(formatName, root);
    }

    private static @NotNull IIOMetadataNode getOrCreateNode(@NotNull IIOMetadataNode root, @NotNull String name) {
        for (int i = 0; i < root.getLength(); i++) {
            if (root.item(i).getNodeName().equalsIgnoreCase(name))
                return (IIOMetadataNode) root.item(i);
        }

        IIOMetadataNode node = new IIOMetadataNode(name);
        root.appendChild(node);
        return node;
    }

}
