package dev.simplified.image.codec.gif;

import dev.simplified.annotations.Cleanup;
import dev.simplified.annotations.SilentThrows;
import dev.simplified.collection.Concurrent;
import dev.simplified.collection.ConcurrentList;
import dev.simplified.image.AnimatedImageData;
import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.PixelBuffer;
import dev.simplified.image.StaticImageData;
import dev.simplified.image.codec.ImageReader;
import dev.simplified.image.codec.common.LittleEndian;
import dev.simplified.image.exception.ImageDecodeException;
import dev.simplified.image.frame.FrameBlend;
import dev.simplified.image.frame.FrameDisposal;
import dev.simplified.image.frame.ImageFrame;
import dev.simplified.image.frame.LoopCount;
import dev.simplified.image.option.ImageReadOptions;
import dev.simplified.util.StringUtil;
import org.jetbrains.annotations.ApiStatus;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import javax.imageio.ImageIO;
import javax.imageio.metadata.IIOMetadata;
import javax.imageio.metadata.IIOMetadataNode;
import javax.imageio.stream.ImageInputStream;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.ByteArrayInputStream;
import java.util.BitSet;
import java.util.stream.IntStream;

/**
 * Reads GIF images (static and animated) via {@link ImageIO}, extracting per-frame
 * metadata for disposal, delay, and canvas offset.
 * <p>
 * The canvas is the logical screen the GIF's Logical Screen Descriptor declares, grown to hold any
 * frame that reaches past it, as Pillow grows it. An animation's frames keep their own sizes and
 * offsets on it; a still's one frame is laid on it at its offset, the rest of the canvas left
 * transparent. A canvas longer than 16383 on a side is refused.
 * <p>
 * GIF has no blend flag: every frame is laid over what the previous frame's disposal left, a pixel
 * at the transparent index showing what lies beneath, so every frame is read
 * {@link FrameBlend#OVER}.
 * <p>
 * An interlaced frame stores its rows in four passes, starting at rows 0, 4, 2 and 1. ImageIO starts
 * each pass at its first row without checking that the frame has that row, so in a frame of 4 rows
 * or fewer, where the pass starting at row 4 holds none, it decodes the next stored row as row 4 and
 * discards it, and leaves another row unwritten. Such a frame is handed to ImageIO with its
 * interlace flag cleared, so ImageIO reads its rows in the order they are stored, and its rows are
 * then moved to the rows they belong to. ImageIO reads every other frame as it stands.
 */
@ApiStatus.Internal
public class GifImageReader implements ImageReader {

    /**
     * The bit of an image descriptor's packed field that declares its frame interlaced.
     */
    private static final int INTERLACE_FLAG = 0x40;

    /**
     * The longest side of a canvas this reader decodes onto, the bound the WebP reader holds a VP8X
     * canvas to.
     */
    private static final int MAX_CANVAS_SIDE = 16383;

    /**
     * The tallest interlaced frame ImageIO misplaces rows of, the pass starting at row 4 holding
     * none of its rows.
     */
    private static final int MAX_MISREAD_HEIGHT = 4;

    /**
     * The first row of each interlace pass.
     */
    private static final int[] PASS_START = { 0, 4, 2, 1 };

    /**
     * The rows between one row of an interlace pass and the next.
     */
    private static final int[] PASS_STEP = { 8, 8, 4, 2 };

    @Override
    public @NotNull ImageFormat getFormat() {
        return ImageFormat.GIF;
    }

    @Override
    public boolean canRead(byte @NotNull [] data) {
        return ImageFormat.GIF.matches(data);
    }

    @Override
    @SilentThrows
    public @NotNull ImageData read(byte @NotNull [] data, @Nullable ImageReadOptions options) {
        // Each interlaced frame ImageIO misplaces rows of is handed over declared sequential, and its
        // rows are moved where they belong once it is read.
        int[] descriptors = imageDescriptorFlags(data);
        BitSet storedOrder = new BitSet();
        byte[] source = data;

        for (int i = 0; i < descriptors.length; i++) {
            int flags = descriptors[i];
            int height = LittleEndian.readUint16(data, flags - 2);

            if ((data[flags] & INTERLACE_FLAG) == 0 || height > MAX_MISREAD_HEIGHT)
                continue;

            if (source == data)
                source = data.clone();

            source[flags] &= ~INTERLACE_FLAG;
            storedOrder.set(i);
        }

        @Cleanup ImageInputStream stream = ImageIO.createImageInputStream(new ByteArrayInputStream(source));
        javax.imageio.ImageReader reader = ImageIO.getImageReadersByFormatName("gif").next();
        reader.setInput(stream);

        int frameCount = reader.getNumImages(true);

        if (frameCount <= 0)
            throw new ImageDecodeException("GIF contains no frames");

        ConcurrentList<ImageFrame> frames = Concurrent.newList();
        LoopCount loopCount = LoopCount.Preset.NONE;
        IIOMetadataNode screen = logicalScreen(reader.getStreamMetadata());
        int canvasWidth = screenDimension(screen, "logicalScreenWidth");
        int canvasHeight = screenDimension(screen, "logicalScreenHeight");

        for (int i = 0; i < frameCount; i++) {
            BufferedImage image = reader.read(i);
            if (storedOrder.get(i))
                deinterlace(image);

            IIOMetadata metadata = reader.getImageMetadata(i);

            int delayMs = 100;
            int offsetX = 0;
            int offsetY = 0;
            FrameDisposal disposal = FrameDisposal.NONE;

            if (metadata != null) {
                String formatName = metadata.getNativeMetadataFormatName();
                IIOMetadataNode root = (IIOMetadataNode) metadata.getAsTree(formatName);

                // Extract GraphicControlExtension
                IIOMetadataNode gce = findNode(root, "GraphicControlExtension");
                if (gce != null) {
                    String delayAttr = gce.getAttribute("delayTime");
                    if (StringUtil.isNotEmpty(delayAttr)) {
                        // A declared 0 means "unspecified", not "zero time" - it is the legacy
                        // idiom for "as fast as you like", and every player substitutes its own
                        // duration for it. Anything else is reported as declared, so a file's
                        // own timing survives a decode.
                        int declaredCs = Integer.parseInt(delayAttr);
                        delayMs = declaredCs == 0 ? 100 : declaredCs * 10;
                    }

                    String disposalAttr = gce.getAttribute("disposalMethod");
                    if (StringUtil.isNotEmpty(disposalAttr))
                        disposal = FrameDisposal.of(disposalAttr);
                }

                // Extract ImageDescriptor for offsets
                IIOMetadataNode desc = findNode(root, "ImageDescriptor");
                if (desc != null) {
                    String leftAttr = desc.getAttribute("imageLeftPosition");
                    String topAttr = desc.getAttribute("imageTopPosition");
                    if (StringUtil.isNotEmpty(leftAttr)) offsetX = Integer.parseInt(leftAttr);
                    if (StringUtil.isNotEmpty(topAttr)) offsetY = Integer.parseInt(topAttr);
                }

                // Extract the loop count from the looping extension (first frame only)
                if (i == 0) {
                    IIOMetadataNode appExts = findNode(root, "ApplicationExtensions");
                    if (appExts != null)
                        loopCount = parseLoopCount(appExts);
                }
            }

            frames.add(ImageFrame.of(PixelBuffer.wrap(image), delayMs, offsetX, offsetY, disposal, FrameBlend.OVER));
            canvasWidth = Math.max(canvasWidth, offsetX + image.getWidth());
            canvasHeight = Math.max(canvasHeight, offsetY + image.getHeight());
        }

        reader.dispose();

        if (canvasWidth > MAX_CANVAS_SIDE || canvasHeight > MAX_CANVAS_SIDE)
            throw new ImageDecodeException("GIF canvas '%dx%d' exceeds '%d' on a side", canvasWidth, canvasHeight, MAX_CANVAS_SIDE);

        if (frames.size() == 1)
            return StaticImageData.of(onCanvas(frames.getFirst(), canvasWidth, canvasHeight));

        return AnimatedImageData.builder()
            .withFrames(frames)
            .withWidth(canvasWidth)
            .withHeight(canvasHeight)
            .withLoopCount(loopCount)
            .build();
    }

    /**
     * Finds the Logical Screen Descriptor in a GIF's stream metadata.
     *
     * @param streamMetadata the stream metadata the reader hands over, if any
     * @return the descriptor's node, or {@code null} when the metadata carries none
     */
    private static @Nullable IIOMetadataNode logicalScreen(@Nullable IIOMetadata streamMetadata) {
        if (streamMetadata == null)
            return null;

        IIOMetadataNode root = (IIOMetadataNode) streamMetadata.getAsTree(streamMetadata.getNativeMetadataFormatName());
        return findNode(root, "LogicalScreenDescriptor");
    }

    /**
     * Reads one dimension of the logical screen a GIF declares.
     *
     * @param screen the Logical Screen Descriptor's node, if the metadata carries one
     * @param attribute the attribute holding the dimension
     * @return the dimension declared, or {@code 0} when none is
     */
    private static int screenDimension(@Nullable IIOMetadataNode screen, @NotNull String attribute) {
        if (screen == null)
            return 0;

        String value = screen.getAttribute(attribute);
        return StringUtil.isNotEmpty(value) ? Integer.parseInt(value) : 0;
    }

    /**
     * Lays a still GIF's one frame on its canvas at the frame's offset.
     * <p>
     * A frame spanning the canvas from the origin is the picture as it is; any other leaves the rest
     * of the canvas transparent.
     *
     * @param frame the one frame
     * @param width the canvas width, which holds the frame at its offset
     * @param height the canvas height, which holds the frame at its offset
     * @return the picture the still shows
     */
    private static @NotNull PixelBuffer onCanvas(@NotNull ImageFrame frame, int width, int height) {
        if (frame.offsetX() == 0 && frame.offsetY() == 0 && frame.width() == width && frame.height() == height)
            return frame.pixels();

        PixelBuffer canvas = PixelBuffer.create(width, height);
        canvas.setPixels(frame.offsetX(), frame.offsetY(), frame.width(), frame.height(), frame.pixels().data(), 0, frame.width());
        return canvas;
    }

    /**
     * Finds the packed field of every image descriptor in a GIF, in file order.
     * <p>
     * The blocks are walked as ImageIO walks them to count frames: an extension's label and
     * sub-blocks are skipped, so are the sub-blocks of a block it does not know, and a trailer, a
     * zero byte or the end of the data ends the walk. The descriptor at index {@code i} is
     * therefore the one ImageIO reads as frame {@code i}.
     *
     * @param gif the encoded GIF
     * @return the offset of each image descriptor's packed field, which follows its two-byte height
     */
    private static int @NotNull [] imageDescriptorFlags(byte @NotNull [] gif) {
        IntStream.Builder flags = IntStream.builder();

        if (gif.length < 13)
            return flags.build().toArray();

        int position = 13; // header and logical screen descriptor
        int screenFlags = gif[10] & 0xFF;
        if ((screenFlags & 0x80) != 0)
            position += colorTableBytes(screenFlags);

        while (position < gif.length) {
            int block = gif[position++] & 0xFF;

            if (block == 0x3B || block == 0x00) // the trailer, or a zero byte ImageIO reads as the end
                break;

            if (block == 0x21) { // extension: a label, then sub-blocks
                position = skipSubBlocks(gif, position + 1);
                continue;
            }

            if (block != 0x2C) { // a block ImageIO does not know: sub-blocks
                position = skipSubBlocks(gif, position);
                continue;
            }

            int descriptorFlags = position + 8; // after left, top, width and height
            if (descriptorFlags >= gif.length)
                break;

            flags.add(descriptorFlags);
            position = descriptorFlags + 1;
            if ((gif[descriptorFlags] & 0x80) != 0)
                position += colorTableBytes(gif[descriptorFlags] & 0xFF);

            position = skipSubBlocks(gif, position + 1); // after the LZW minimum code size
        }

        return flags.build().toArray();
    }

    /**
     * Measures a colour table from the packed field that declares it.
     *
     * @param flags a logical screen or image descriptor's packed field
     * @return the bytes of the table it declares
     */
    private static int colorTableBytes(int flags) {
        return 3 * (1 << ((flags & 0x07) + 1));
    }

    /**
     * Skips a run of sub-blocks, each a length byte and that many bytes, ended by a zero length.
     *
     * @param gif the encoded GIF
     * @param position the offset of the first sub-block's length
     * @return the offset after the terminating zero, or one at or past the end of the data
     */
    private static int skipSubBlocks(byte @NotNull [] gif, int position) {
        while (position < gif.length) {
            int length = gif[position] & 0xFF;
            position += length + 1;

            if (length == 0)
                break;
        }

        return position;
    }

    /**
     * Moves the rows of a frame read in the order an interlaced frame stores them to the rows they
     * belong to.
     * <p>
     * The passes store rows 0, 8, 16 and on, then 4, 12, 20 and on, then 2, 6, 10 and on, then every
     * odd row.
     *
     * @param image the frame, its rows in stored order
     */
    private static void deinterlace(@NotNull BufferedImage image) {
        WritableRaster raster = image.getRaster();
        int width = raster.getWidth();
        int height = raster.getHeight();
        Object[] stored = new Object[height];

        for (int y = 0; y < height; y++)
            stored[y] = raster.getDataElements(0, y, width, 1, null);

        int next = 0;
        for (int pass = 0; pass < PASS_START.length; pass++) {
            for (int y = PASS_START[pass]; y < height; y += PASS_STEP[pass])
                raster.setDataElements(0, y, width, 1, stored[next++]);
        }
    }

    private static @Nullable IIOMetadataNode findNode(@NotNull IIOMetadataNode root, @NotNull String name) {
        for (int i = 0; i < root.getLength(); i++) {
            if (root.item(i).getNodeName().equalsIgnoreCase(name))
                return (IIOMetadataNode) root.item(i);
        }

        return null;
    }

    /**
     * Reads the loop count a GIF's looping application extension declares.
     * <p>
     * The extension is NETSCAPE2.0, or ANIMEXTS1.0, which some encoders write in its place with the
     * same sub-blocks; libwebp's {@code gif2webp} and Wuffs read both. Its sub-block {@code 1}
     * counts repeats after the first play, {@code 0} repeating forever. An extension whose first
     * sub-block is another, such as NETSCAPE2.0's buffering sub-block {@code 2}, declares no count,
     * and a GIF that declares none plays once.
     *
     * @param appExts the first frame's application extensions
     * @return the loop count declared, or {@link LoopCount.Preset#NONE} when no extension declares one
     */
    private static @NotNull LoopCount parseLoopCount(@NotNull IIOMetadataNode appExts) {
        for (int i = 0; i < appExts.getLength(); i++) {
            IIOMetadataNode child = (IIOMetadataNode) appExts.item(i);

            if (!isLoopingExtension(child.getAttribute("applicationID"), child.getAttribute("authenticationCode")))
                continue;

            // ImageIO hands over the extension's sub-blocks joined, so the first byte is the first
            // sub-block's id.
            if (child.getUserObject() instanceof byte[] bytes && bytes.length >= 3 && bytes[0] == 1) {
                int count = LittleEndian.readUint16(bytes, 1);
                return count == 0 ? LoopCount.Preset.INFINITE : LoopCount.of(count);
            }
        }

        return LoopCount.Preset.NONE;
    }

    /**
     * Tests whether an application extension is one that declares a loop count.
     *
     * @param appId the extension's eight-character application identifier
     * @param authCode the extension's three-character authentication code
     * @return {@code true} for NETSCAPE2.0 and ANIMEXTS1.0
     */
    private static boolean isLoopingExtension(@NotNull String appId, @NotNull String authCode) {
        return ("NETSCAPE".equals(appId) && "2.0".equals(authCode))
            || ("ANIMEXTS".equals(appId) && "1.0".equals(authCode));
    }

}
