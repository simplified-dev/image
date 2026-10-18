/**
 * The helpers two or more codec packages share - internal.
 * <p>
 * {@link dev.simplified.image.codec.common.LittleEndian LittleEndian} reads and writes the
 * little-endian integers the GIF, ICO, TGA, RIFF and VP8 headers are built from,
 * {@link dev.simplified.image.codec.common.ImageIOBridge ImageIOBridge} decodes bytes through ImageIO
 * for the readers that hand a whole format to it, and
 * {@link dev.simplified.image.codec.common.DecodeBudget DecodeBudget} is the most pixels one read
 * may hold, which every reader claims against before it allocates.
 * <p>
 * A helper belongs here once a second codec package calls it; one only a single format uses stays
 * private to that format. This package names no format, so no format comes to depend on another
 * through it. Its types are public because the format packages call them, and each carries
 * {@code @ApiStatus.Internal}: a caller has no use for any of them.
 */
package dev.simplified.image.codec.common;
