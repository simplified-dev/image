package dev.simplified.image.codec.gif;

import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.codec.ImageWriteOptions;
import dev.simplified.image.codec.ImageWriter;
import dev.simplified.image.data.AnimatedImageData;
import dev.simplified.image.data.ImageFrame;
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
import java.awt.image.BufferedImage;
import java.awt.image.IndexColorModel;
import java.awt.image.WritableRaster;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Writes GIF images (static and animated) via {@link ImageIO}, configuring
 * per-frame disposal, delay, and loop count metadata.
 */
public class GifImageWriter implements ImageWriter {

    /** Maximum palette entries in a GIF local or global color table. */
    private static final int MAX_PALETTE_SIZE = 256;

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

                BufferedImage indexedFrame = buildIndexedFrame(
                    flattenedPixels, frameIdx * frameStride,
                    frame.pixels().width(), frame.pixels().height(),
                    palette, transparent, transparentColorIndex);
                writer.writeToSequence(new IIOImage(indexedFrame, null, metadata), param);
                frameIdx++;
            }

            writer.endWriteSequence();
        } else {
            ImageFrame frame = data.getFrames().getFirst();
            IIOMetadata metadata = writer.getDefaultImageMetadata(specifier, param);
            configureFrameMetadata(metadata, frame, transparent, transparentColorIndex);
            BufferedImage indexedFrame = buildIndexedFrame(
                flattenedPixels, 0, frame.pixels().width(), frame.pixels().height(),
                palette, transparent, transparentColorIndex);
            writer.write(new IIOImage(indexedFrame, null, metadata));
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
            totalPixels += frame.pixels().data().length;
        int[] out = new int[totalPixels];
        int offset = 0;
        for (ImageFrame frame : data.getFrames()) {
            int[] src = frame.pixels().data();
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
     * Builds a single {@link IndexColorModel} shared across every frame from the already
     * alpha-flattened pixel array.
     * <p>
     * When the combined pixel set fits in 256 (or 255 when a transparent slot is reserved)
     * an exact palette is returned - useful for UI renders like tooltips which typically
     * stay well under the cap and deserve pixel-accurate color preservation. When there
     * are too many unique colors a frequency-weighted median-cut quantizer picks 256 image
     * -adaptive colors; this produces far better results than the default Java GIF writer's
     * hardcoded web-safe cube, which snaps every non-web-safe pixel to one of 216 coarse
     * colors and destroys smooth gradients (the old "lost border gradient" bug).
     */
    @NotNull
    private static IndexColorModel buildSharedPalette(
        int @NotNull [] flattenedPixels,
        boolean transparent,
        int transparentColorIndex
    ) {
        int capacity = transparent ? MAX_PALETTE_SIZE - 1 : MAX_PALETTE_SIZE;
        Map<Integer, Integer> histogram = new LinkedHashMap<>();
        for (int argb : flattenedPixels) {
            // Skip the transparent marker (ARGB 0x00000000) - the reserved transparent
            // palette slot handles it, it shouldn't consume a color slot.
            if (transparent && (argb >>> 24) == 0) continue;
            histogram.merge(argb & 0xFFFFFF, 1, Integer::sum);
        }

        if (histogram.size() <= capacity)
            return exactPalette(histogram.keySet(), transparent, transparentColorIndex);

        return medianCutPalette(histogram, capacity, transparent, transparentColorIndex);
    }

    @NotNull
    private static IndexColorModel exactPalette(
        @NotNull Iterable<Integer> uniqueRgb,
        boolean transparent,
        int transparentColorIndex
    ) {
        int count = 0;
        for (Integer ignored : uniqueRgb) count++;
        int n = count + (transparent ? 1 : 0);
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

    /**
     * Runs a frequency-weighted median-cut quantizer against the pixel histogram to produce
     * an image-adaptive palette of at most {@code capacity} colors.
     * <p>
     * Algorithm:
     * <ol>
     * <li>Start with one bucket containing every unique RGB.</li>
     * <li>Repeatedly find the bucket with the widest single-channel spread and split it at
     * the frequency-weighted median of that channel. A bucket with only one unique color
     * (or zero channel spread) cannot split.</li>
     * <li>Stop when the bucket count reaches the capacity or no bucket can split further.</li>
     * <li>Emit one palette entry per bucket, weighted by pixel count so high-frequency
     * colors (solid backgrounds, text fills) dominate their bucket's representative.</li>
     * </ol>
     * This is the standard "modified median cut" approach used by most quality GIF encoders
     * (Leptonica, GIFLIB's helpers, etc.). It's not perceptually weighted (no YUV transform,
     * no CIE Lab color space) but is massively better than snapping to a hardcoded cube.
     */
    @NotNull
    private static IndexColorModel medianCutPalette(
        @NotNull Map<Integer, Integer> histogram,
        int capacity,
        boolean transparent,
        int transparentColorIndex
    ) {
        List<int[]> colors = new ArrayList<>(histogram.size());
        for (Map.Entry<Integer, Integer> entry : histogram.entrySet()) {
            int rgb = entry.getKey();
            colors.add(new int[] {
                (rgb >> 16) & 0xFF, (rgb >> 8) & 0xFF, rgb & 0xFF, entry.getValue()
            });
        }

        List<Bucket> buckets = new ArrayList<>();
        buckets.add(new Bucket(colors));

        while (buckets.size() < capacity) {
            Bucket toSplit = null;
            int bestSpread = 0;
            for (Bucket bucket : buckets) {
                int spread = bucket.maxChannelSpread();
                if (spread > bestSpread) {
                    bestSpread = spread;
                    toSplit = bucket;
                }
            }
            if (toSplit == null) break;

            Bucket[] halves = toSplit.split();
            buckets.remove(toSplit);
            buckets.add(halves[0]);
            buckets.add(halves[1]);
        }

        int n = buckets.size() + (transparent ? 1 : 0);
        byte[] r = new byte[n];
        byte[] g = new byte[n];
        byte[] b = new byte[n];
        int i = 0;
        for (Bucket bucket : buckets) {
            if (transparent && i == transparentColorIndex) i++;
            int[] avg = bucket.weightedAverage();
            r[i] = (byte) avg[0];
            g[i] = (byte) avg[1];
            b[i] = (byte) avg[2];
            i++;
        }

        if (transparent)
            return new IndexColorModel(8, n, r, g, b, transparentColorIndex);

        return new IndexColorModel(8, n, r, g, b);
    }

    /**
     * A median-cut bucket holding a list of {@code [r, g, b, count]} rows for every unique
     * color in the bucket. Splits are performed on the channel with the widest value range
     * at the frequency-weighted median so large flat regions collapse into single palette
     * slots instead of being merged with nearby detail.
     */
    private static final class Bucket {

        private final @NotNull List<int @NotNull []> rows;

        Bucket(@NotNull List<int @NotNull []> rows) {
            this.rows = rows;
        }

        int maxChannelSpread() {
            if (this.rows.size() <= 1) return 0;
            int minR = 255, minG = 255, minB = 255;
            int maxR = 0, maxG = 0, maxB = 0;
            for (int[] row : this.rows) {
                if (row[0] < minR) minR = row[0];
                if (row[0] > maxR) maxR = row[0];
                if (row[1] < minG) minG = row[1];
                if (row[1] > maxG) maxG = row[1];
                if (row[2] < minB) minB = row[2];
                if (row[2] > maxB) maxB = row[2];
            }
            return Math.max(Math.max(maxR - minR, maxG - minG), maxB - minB);
        }

        @NotNull Bucket @NotNull [] split() {
            int minR = 255, minG = 255, minB = 255;
            int maxR = 0, maxG = 0, maxB = 0;
            for (int[] row : this.rows) {
                if (row[0] < minR) minR = row[0];
                if (row[0] > maxR) maxR = row[0];
                if (row[1] < minG) minG = row[1];
                if (row[1] > maxG) maxG = row[1];
                if (row[2] < minB) minB = row[2];
                if (row[2] > maxB) maxB = row[2];
            }
            int rangeR = maxR - minR;
            int rangeG = maxG - minG;
            int rangeB = maxB - minB;

            int axis;
            if (rangeR >= rangeG && rangeR >= rangeB) axis = 0;
            else if (rangeG >= rangeB) axis = 1;
            else axis = 2;

            int sortAxis = axis;
            this.rows.sort(Comparator.comparingInt(row -> row[sortAxis]));

            long totalCount = 0;
            for (int[] row : this.rows) totalCount += row[3];
            long midpoint = totalCount / 2;

            long cumulative = 0;
            int splitIndex = 1;
            for (int i = 0; i < this.rows.size(); i++) {
                cumulative += this.rows.get(i)[3];
                if (cumulative >= midpoint) {
                    splitIndex = Math.max(1, Math.min(this.rows.size() - 1, i + 1));
                    break;
                }
            }

            List<int[]> left = new ArrayList<>(this.rows.subList(0, splitIndex));
            List<int[]> right = new ArrayList<>(this.rows.subList(splitIndex, this.rows.size()));
            return new Bucket[] { new Bucket(left), new Bucket(right) };
        }

        int @NotNull [] weightedAverage() {
            long sumR = 0, sumG = 0, sumB = 0, sumCount = 0;
            for (int[] row : this.rows) {
                long c = row[3];
                sumR += (long) row[0] * c;
                sumG += (long) row[1] * c;
                sumB += (long) row[2] * c;
                sumCount += c;
            }
            if (sumCount == 0) return new int[] { 0, 0, 0 };
            return new int[] {
                (int) (sumR / sumCount),
                (int) (sumG / sumCount),
                (int) (sumB / sumCount)
            };
        }

    }

    /**
     * Maps a slice of the flattened ARGB pixel array to a {@link BufferedImage#TYPE_BYTE_INDEXED}
     * image using direct nearest-neighbor lookup against the shared palette. No dithering -
     * flat regions stay flat. Caches {@code rgb -> palette index} for repeat pixels so a
     * typical UI render with mostly solid colors resolves the whole frame with one palette
     * search per unique color.
     */
    @NotNull
    private static BufferedImage buildIndexedFrame(
        int @NotNull [] flatPixels,
        int offset,
        int width,
        int height,
        @NotNull IndexColorModel palette,
        boolean transparent,
        int transparentColorIndex
    ) {
        BufferedImage indexed = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_INDEXED, palette);

        int paletteSize = palette.getMapSize();
        int[] pr = new int[paletteSize];
        int[] pg = new int[paletteSize];
        int[] pb = new int[paletteSize];
        for (int i = 0; i < paletteSize; i++) {
            pr[i] = palette.getRed(i);
            pg[i] = palette.getGreen(i);
            pb[i] = palette.getBlue(i);
        }

        Map<Integer, Byte> indexCache = new HashMap<>();
        byte[] indices = new byte[width * height];
        int total = width * height;

        for (int i = 0; i < total; i++) {
            int argb = flatPixels[offset + i];
            if (transparent && (argb >>> 24) == 0) {
                indices[i] = (byte) transparentColorIndex;
                continue;
            }

            int rgb = argb & 0xFFFFFF;
            Byte cached = indexCache.get(rgb);
            if (cached != null) {
                indices[i] = cached;
                continue;
            }

            int r = (rgb >> 16) & 0xFF;
            int g = (rgb >> 8) & 0xFF;
            int b = rgb & 0xFF;
            int bestIdx = 0;
            int bestDist = Integer.MAX_VALUE;
            for (int j = 0; j < paletteSize; j++) {
                if (transparent && j == transparentColorIndex) continue;
                int dr = r - pr[j];
                int dg = g - pg[j];
                int db = b - pb[j];
                int dist = dr * dr + dg * dg + db * db;
                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx = j;
                    if (dist == 0) break;
                }
            }
            byte idxByte = (byte) bestIdx;
            indexCache.put(rgb, idxByte);
            indices[i] = idxByte;
        }

        WritableRaster raster = indexed.getRaster();
        raster.setDataElements(0, 0, width, height, indices);
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
