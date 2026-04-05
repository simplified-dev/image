package dev.simplified.image.codec.gif;

import dev.simplified.collection.Concurrent;
import dev.simplified.collection.ConcurrentList;
import dev.simplified.image.AnimatedImageData;
import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.ImageFrame;
import dev.simplified.image.ImageFrame.Blend;
import dev.simplified.image.ImageFrame.Disposal;
import dev.simplified.image.StaticImageData;
import dev.simplified.image.codec.ImageReadOptions;
import dev.simplified.image.codec.ImageReader;
import dev.simplified.image.exception.ImageDecodeException;
import dev.simplified.util.StringUtil;
import lombok.Cleanup;
import lombok.SneakyThrows;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import javax.imageio.ImageIO;
import javax.imageio.metadata.IIOMetadata;
import javax.imageio.metadata.IIOMetadataNode;
import javax.imageio.stream.ImageInputStream;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;

/**
 * Reads GIF images (static and animated) via {@link ImageIO}, extracting per-frame
 * metadata for disposal, delay, and canvas offset.
 */
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
    @SneakyThrows
    public @NotNull ImageData read(byte @NotNull [] data, @Nullable ImageReadOptions options) {
        @Cleanup ImageInputStream stream = ImageIO.createImageInputStream(new ByteArrayInputStream(data));
        javax.imageio.ImageReader reader = ImageIO.getImageReadersByFormatName("gif").next();
        reader.setInput(stream);

        int frameCount = reader.getNumImages(true);

        if (frameCount <= 0)
            throw new ImageDecodeException("GIF contains no frames");

        ConcurrentList<ImageFrame> frames = Concurrent.newList();
        int loopCount = 0;

        for (int i = 0; i < frameCount; i++) {
            BufferedImage image = reader.read(i);
            IIOMetadata metadata = reader.getImageMetadata(i);

            int delayMs = 100;
            int offsetX = 0;
            int offsetY = 0;
            Disposal disposal = Disposal.NONE;

            if (metadata != null) {
                String formatName = metadata.getNativeMetadataFormatName();
                IIOMetadataNode root = (IIOMetadataNode) metadata.getAsTree(formatName);

                // Extract GraphicControlExtension
                IIOMetadataNode gce = findNode(root, "GraphicControlExtension");
                if (gce != null) {
                    String delayAttr = gce.getAttribute("delayTime");
                    if (StringUtil.isNotEmpty(delayAttr))
                        delayMs = Math.max(10, Integer.parseInt(delayAttr)) * 10;

                    String disposalAttr = gce.getAttribute("disposalMethod");
                    if (StringUtil.isNotEmpty(disposalAttr))
                        disposal = Disposal.of(disposalAttr);
                }

                // Extract ImageDescriptor for offsets
                IIOMetadataNode desc = findNode(root, "ImageDescriptor");
                if (desc != null) {
                    String leftAttr = desc.getAttribute("imageLeftPosition");
                    String topAttr = desc.getAttribute("imageTopPosition");
                    if (StringUtil.isNotEmpty(leftAttr)) offsetX = Integer.parseInt(leftAttr);
                    if (StringUtil.isNotEmpty(topAttr)) offsetY = Integer.parseInt(topAttr);
                }

                // Extract loop count from NETSCAPE2.0 extension (first frame only)
                if (i == 0) {
                    IIOMetadataNode appExts = findNode(root, "ApplicationExtensions");
                    if (appExts != null)
                        loopCount = parseLoopCount(appExts);
                }
            }

            frames.add(ImageFrame.of(image, delayMs, offsetX, offsetY, disposal, Blend.SOURCE));
        }

        reader.dispose();

        if (frames.size() == 1)
            return StaticImageData.of(frames.getFirst().getImage());

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

    private static int parseLoopCount(@NotNull IIOMetadataNode appExts) {
        for (int i = 0; i < appExts.getLength(); i++) {
            IIOMetadataNode child = (IIOMetadataNode) appExts.item(i);
            String appId = child.getAttribute("applicationID");
            String authCode = child.getAttribute("authenticationCode");

            if ("NETSCAPE".equals(appId) && "2.0".equals(authCode)) {
                Object userObject = child.getUserObject();

                if (userObject instanceof byte[] bytes && bytes.length >= 3)
                    return (bytes[1] & 0xFF) | ((bytes[2] & 0xFF) << 8);
            }
        }

        return 0;
    }

}
