package dev.simplified.image.codec.pnm;

import dev.simplified.image.codec.ImageWriteOptions;
import org.jetbrains.annotations.NotNull;

/**
 * PNM-specific encoding options selecting the Netpbm variant and ASCII-vs-binary encoding.
 * <p>
 * The combination of {@link Variant} and {@link #ascii()} selects the output magic:
 * <ul>
 *   <li><b>{@link Variant#PBM PBM}</b> + ascii false -> P4 (binary bitmap)</li>
 *   <li><b>PBM</b> + ascii true -> P1 (ASCII bitmap)</li>
 *   <li><b>{@link Variant#PGM PGM}</b> + ascii false -> P5 (binary graymap)</li>
 *   <li><b>PGM</b> + ascii true -> P2 (ASCII graymap)</li>
 *   <li><b>{@link Variant#PPM PPM}</b> + ascii false -> P6 (binary pixmap)</li>
 *   <li><b>PPM</b> + ascii true -> P3 (ASCII pixmap)</li>
 * </ul>
 */
public record PnmWriteOptions(@NotNull Variant variant, boolean ascii) implements ImageWriteOptions {

    /**
     * Returns a new builder for PNM write options.
     *
     * @return a new builder instance
     */
    public static @NotNull Builder builder() {
        return new Builder();
    }

    /**
     * Netpbm family variant.
     */
    public enum Variant {

        /**
         * 1-bit bitmap (P1 ASCII, P4 binary).
         */
        PBM,

        /**
         * Grayscale graymap (P2 ASCII, P5 binary).
         */
        PGM,

        /**
         * RGB pixmap (P3 ASCII, P6 binary).
         */
        PPM

    }

    /**
     * Builds {@link PnmWriteOptions} instances.
     */
    public static class Builder {

        private Variant variant = Variant.PPM;
        private boolean ascii = false;

        /**
         * Sets the Netpbm family variant.
         *
         * @param variant the target variant
         * @return this builder for chaining
         */
        public @NotNull Builder withVariant(@NotNull Variant variant) {
            this.variant = variant;
            return this;
        }

        /**
         * Enables ASCII encoding (the P1/P2/P3 magics). Defaults to binary (P4/P5/P6).
         *
         * @return this builder for chaining
         */
        public @NotNull Builder isAscii() {
            return this.isAscii(true);
        }

        /**
         * Sets ASCII vs. binary encoding.
         *
         * @param ascii true for ASCII (P1/P2/P3), false for binary (P4/P5/P6)
         * @return this builder for chaining
         */
        public @NotNull Builder isAscii(boolean ascii) {
            this.ascii = ascii;
            return this;
        }

        public @NotNull PnmWriteOptions build() {
            return new PnmWriteOptions(this.variant, this.ascii);
        }

    }

}
