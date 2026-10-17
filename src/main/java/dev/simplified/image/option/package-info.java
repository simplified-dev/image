/**
 * Everything a caller passes to configure a read or a write, each with its builder and the enums it
 * takes.
 * <p>
 * {@link dev.simplified.image.option.ImageReadOptions ImageReadOptions} and
 * {@link dev.simplified.image.option.ImageWriteOptions ImageWriteOptions} are the two types the codec
 * contract takes. Each format with a knob to turn implements the second:
 * {@link dev.simplified.image.option.GifWriteOptions GifWriteOptions},
 * {@link dev.simplified.image.option.JpegWriteOptions JpegWriteOptions},
 * {@link dev.simplified.image.option.PngWriteOptions PngWriteOptions},
 * {@link dev.simplified.image.option.PnmWriteOptions PnmWriteOptions},
 * {@link dev.simplified.image.option.TgaWriteOptions TgaWriteOptions},
 * {@link dev.simplified.image.option.TiffWriteOptions TiffWriteOptions} and
 * {@link dev.simplified.image.option.WebPWriteOptions WebPWriteOptions}, each passed to
 * {@link dev.simplified.image.ImageFactory ImageFactory} beside the format it configures. An enum an
 * options type takes is nested in it, as {@code GifWriteOptions.DelayFidelity} and
 * {@code PnmWriteOptions.Variant} are.
 * <p>
 * Every options type lives here rather than beside its codec, so a caller configures a write without
 * naming a codec package.
 */
package dev.simplified.image.option;
