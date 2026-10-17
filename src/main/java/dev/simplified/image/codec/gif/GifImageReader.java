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
import java.io.ByteArrayInputStream;

/**
 * Reads GIF images (static and animated) via {@link ImageIO}, extracting per-frame
 * metadata for disposal, delay, and canvas offset.
 * <p>
 * GIF has no blend flag: every frame is laid over what the previous frame's disposal left, a pixel
 * at the transparent index showing what lies beneath, so every frame is read
 * {@link FrameBlend#OVER}.
 */
@ApiStatus.Internal
public class GifImageReader implements ImageReader {

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
        @Cleanup ImageInputStream stream = ImageIO.createImageInputStream(new ByteArrayInputStream(data));
        javax.imageio.ImageReader reader = ImageIO.getImageReadersByFormatName("gif").next();
        reader.setInput(stream);

        int frameCount = reader.getNumImages(true);

        if (frameCount <= 0)
            throw new ImageDecodeException("GIF contains no frames");

        ConcurrentList<ImageFrame> frames = Concurrent.newList();
        LoopCount loopCount = LoopCount.Preset.NONE;

        for (int i = 0; i < frameCount; i++) {
            var image = reader.read(i);
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
        }

        reader.dispose();

        if (frames.size() == 1)
            return StaticImageData.of(frames.getFirst().pixels());

        return AnimatedImageData.builder()
            .withFrames(frames)
            .withLoopCount(loopCount)
            .build();
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
                int count = (bytes[1] & 0xFF) | ((bytes[2] & 0xFF) << 8);
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
