/**
 * The PNM reader and writer - internal.
 * <p>
 * {@link dev.simplified.image.codec.pnm.PnmImageReader PnmImageReader} and
 * {@link dev.simplified.image.codec.pnm.PnmImageWriter PnmImageWriter} read and write the Netpbm family
 * - PBM, PGM and PPM, ASCII and binary. A caller reaches them through
 * {@link dev.simplified.image.ImageFactory ImageFactory} with
 * {@link dev.simplified.image.ImageFormat#PNM ImageFormat.PNM}, never directly: both are public only
 * because the factory constructs them from the root package, and each carries
 * {@code @ApiStatus.Internal}.
 * <p>
 * A type belongs here when only PNM uses it. A helper a second codec package needs moves to
 * {@link dev.simplified.image.codec.common codec.common}, and the options a PNM write takes are
 * {@link dev.simplified.image.option.PnmWriteOptions PnmWriteOptions}, in
 * {@link dev.simplified.image.option option}.
 */
package dev.simplified.image.codec.pnm;
