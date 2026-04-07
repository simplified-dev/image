package dev.simplified.image.codec.gif;

import dev.simplified.image.AnimatedImageData;
import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.ImageFrame;
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
import java.awt.image.BufferedImage;

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

        if (options instanceof GifWriteOptions gifOptions) {
            loopCount = gifOptions.loopCount();
            transparent = gifOptions.transparent();
            transparentColorIndex = gifOptions.transparentColorIndex();
        } else if (data instanceof AnimatedImageData animated) {
            loopCount = animated.getLoopCount();
        }

        javax.imageio.ImageWriter writer = ImageIO.getImageWritersByFormatName("gif").next();
        @Cleanup ByteArrayDataOutput dataOutput = new ByteArrayDataOutput();
        @Cleanup ImageOutputStream outputStream = ImageIO.createImageOutputStream(dataOutput);
        writer.setOutput(outputStream);

        ImageWriteParam param = writer.getDefaultWriteParam();
        ImageTypeSpecifier specifier = ImageTypeSpecifier.createFromBufferedImageType(BufferedImage.TYPE_BYTE_INDEXED);

        if (data.isAnimated()) {
            writer.prepareWriteSequence(null);
            boolean firstFrame = true;

            for (ImageFrame frame : data.getFrames()) {
                IIOMetadata metadata = writer.getDefaultImageMetadata(specifier, param);
                configureFrameMetadata(metadata, frame, transparent, transparentColorIndex);

                if (firstFrame) {
                    configureLoopMetadata(metadata, loopCount);
                    firstFrame = false;
                }

                writer.writeToSequence(new IIOImage(frame.image(), null, metadata), param);
            }

            writer.endWriteSequence();
        } else {
            ImageFrame frame = data.getFrames().getFirst();
            IIOMetadata metadata = writer.getDefaultImageMetadata(specifier, param);
            configureFrameMetadata(metadata, frame, transparent, transparentColorIndex);
            writer.write(new IIOImage(frame.image(), null, metadata));
        }

        writer.dispose();
        outputStream.flush();
        return dataOutput.toByteArray();
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
