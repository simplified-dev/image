package dev.simplified.image.codec.qoi;

import dev.simplified.image.ImageData;
import dev.simplified.image.ImageFormat;
import dev.simplified.image.codec.ImageWriteOptions;
import dev.simplified.image.codec.ImageWriter;
import dev.simplified.image.pixel.PixelBuffer;
import dev.simplified.stream.ByteArrayDataOutput;
import lombok.Cleanup;
import lombok.SneakyThrows;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

/**
 * Encodes QOI images per the Quite OK Image format specification.
 *
 * @see <a href="https://qoiformat.org/qoi-specification.pdf">QOI specification</a>
 */
public class QoiImageWriter implements ImageWriter {

    private static final int QOI_OP_RGB = 0xFE;
    private static final int QOI_OP_RGBA = 0xFF;
    private static final int QOI_OP_INDEX = 0x00;
    private static final int QOI_OP_DIFF = 0x40;
    private static final int QOI_OP_LUMA = 0x80;
    private static final int QOI_OP_RUN = 0xC0;

    @Override
    public @NotNull ImageFormat getFormat() {
        return ImageFormat.QOI;
    }

    @Override
    @SneakyThrows
    public byte @NotNull [] write(@NotNull ImageData data, @Nullable ImageWriteOptions options) {
        PixelBuffer buffer = data.getFrames().getFirst().pixels();
        int width = buffer.width();
        int height = buffer.height();
        boolean hasAlpha = buffer.hasAlpha();
        int channels = hasAlpha ? 4 : 3;
        int[] src = buffer.data();

        @Cleanup ByteArrayDataOutput out = new ByteArrayDataOutput(14 + src.length * 5 + 8);
        out.writeByte('q');
        out.writeByte('o');
        out.writeByte('i');
        out.writeByte('f');
        out.writeInt(width);
        out.writeInt(height);
        out.writeByte(channels);
        out.writeByte(0);

        int[] table = new int[64];
        int prevR = 0, prevG = 0, prevB = 0, prevA = 255;
        int run = 0;

        for (int i = 0; i < src.length; i++) {
            int argb = src[i];
            int r = (argb >>> 16) & 0xFF;
            int g = (argb >>> 8) & 0xFF;
            int b = argb & 0xFF;
            int a = hasAlpha ? (argb >>> 24) & 0xFF : 255;

            if (r == prevR && g == prevG && b == prevB && a == prevA) {
                run++;

                if (run == 62 || i == src.length - 1) {
                    out.writeByte(QOI_OP_RUN | (run - 1));
                    run = 0;
                }
            } else {
                if (run > 0) {
                    out.writeByte(QOI_OP_RUN | (run - 1));
                    run = 0;
                }

                int pxArgb = (a << 24) | (r << 16) | (g << 8) | b;
                int idx = (r * 3 + g * 5 + b * 7 + a * 11) & 0x3F;

                if (table[idx] == pxArgb) {
                    out.writeByte(QOI_OP_INDEX | idx);
                } else {
                    table[idx] = pxArgb;

                    if (a == prevA) {
                        int dr = (byte) (r - prevR);
                        int dg = (byte) (g - prevG);
                        int db = (byte) (b - prevB);

                        if (dr >= -2 && dr <= 1 && dg >= -2 && dg <= 1 && db >= -2 && db <= 1) {
                            out.writeByte(QOI_OP_DIFF | ((dr + 2) << 4) | ((dg + 2) << 2) | (db + 2));
                        } else {
                            int drDg = dr - dg;
                            int dbDg = db - dg;

                            if (dg >= -32 && dg <= 31 && drDg >= -8 && drDg <= 7 && dbDg >= -8 && dbDg <= 7) {
                                out.writeByte(QOI_OP_LUMA | (dg + 32));
                                out.writeByte(((drDg + 8) << 4) | (dbDg + 8));
                            } else {
                                out.writeByte(QOI_OP_RGB);
                                out.writeByte(r);
                                out.writeByte(g);
                                out.writeByte(b);
                            }
                        }
                    } else {
                        out.writeByte(QOI_OP_RGBA);
                        out.writeByte(r);
                        out.writeByte(g);
                        out.writeByte(b);
                        out.writeByte(a);
                    }
                }

                prevR = r;
                prevG = g;
                prevB = b;
                prevA = a;
            }
        }

        out.writeByte(0);
        out.writeByte(0);
        out.writeByte(0);
        out.writeByte(0);
        out.writeByte(0);
        out.writeByte(0);
        out.writeByte(0);
        out.writeByte(0x01);

        return out.toByteArray();
    }

}
