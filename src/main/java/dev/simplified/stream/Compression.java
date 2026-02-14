package dev.sbs.api.io.stream;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.jetbrains.annotations.NotNull;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.zip.DeflaterOutputStream;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;
import java.util.zip.InflaterInputStream;

/**
 * Defines the types of compression used in stream data, identified by their magic number signatures.
 * <p>
 * Each compression type is detected by reading the first few bytes of the input stream and matching
 * them against known magic byte sequences.
 *
 * @see <a href="https://en.wikipedia.org/wiki/List_of_file_signatures">List of file signatures</a>
 */
@Getter
@RequiredArgsConstructor
public enum Compression {

	/**
	 * No compression.
	 */
	NONE(new int[]{}),
	/**
	 * GZIP compression.
	 * <p>
	 * Magic bytes: {@code 0x1F 0x8B}
	 *
	 * @apiNote {@link GZIPInputStream} and {@link GZIPOutputStream}
	 */
	GZIP(new int[]{ 0x1F, 0x8B }),
	/**
	 * ZLIB compression.
	 * <p>
	 * Magic bytes:
	 * <ul>
	 *   <li>{@code 0x78 0x01} - No compression (no preset dictionary)</li>
	 *   <li>{@code 0x78 0x5E} - Best speed (no preset dictionary)</li>
	 *   <li>{@code 0x78 0x9C} - Default compression (no preset dictionary)</li>
	 *   <li>{@code 0x78 0xDA} - Best compression (no preset dictionary)</li>
	 *   <li>{@code 0x78 0x20} - No compression (with preset dictionary)</li>
	 *   <li>{@code 0x78 0x7D} - Best speed (with preset dictionary)</li>
	 *   <li>{@code 0x78 0xBB} - Default compression (with preset dictionary)</li>
	 *   <li>{@code 0x78 0xF9} - Best compression (with preset dictionary)</li>
	 * </ul>
	 * This enum uses the most common variant ({@code 0x78}) for detection.
	 *
	 * @apiNote {@link InflaterInputStream} and {@link DeflaterOutputStream}
	 */
	ZLIB(new int[]{ 0x78 }),
	/**
	 * BZIP2 compression.
	 * <p>
	 * Magic bytes: {@code 0x42 0x5A 0x68} ("BZh")
	 *
	 * @apiNote Requires Apache Commons Compress or similar library for decompression.
	 */
	BZIP2(new int[]{ 0x42, 0x5A, 0x68 }),
	/**
	 * XZ compression (LZMA2).
	 * <p>
	 * Magic bytes: {@code 0xFD 0x37 0x7A 0x58 0x5A 0x00}
	 *
	 * @apiNote Requires XZ for Java library for decompression.
	 */
	XZ(new int[]{ 0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00 }),
	/**
	 * LZ4 Frame Format compression.
	 * <p>
	 * Magic bytes: {@code 0x04 0x22 0x4D 0x18}
	 *
	 * @apiNote LZ4 block format does not have magic bytes. Requires lz4-java library for decompression.
	 */
	LZ4(new int[]{ 0x04, 0x22, 0x4D, 0x18 }),
	/**
	 * Zstandard (Zstd) compression.
	 * <p>
	 * Magic bytes: {@code 0x28 0xB5 0x2F 0xFD}
	 *
	 * @apiNote Requires zstd-jni library for decompression.
	 */
	ZSTD(new int[]{ 0x28, 0xB5, 0x2F, 0xFD }),
	/**
	 * ZIP/DEFLATE compression.
	 * <p>
	 * Magic bytes: {@code 0x50 0x4B 0x03 0x04} ("PK")
	 *
	 * @apiNote This detects ZIP archives. For raw DEFLATE streams, use ZLIB.
	 */
	ZIP(new int[]{ 0x50, 0x4B, 0x03, 0x04 }),
	/**
	 * 7-Zip compression.
	 * <p>
	 * Magic bytes: {@code 0x37 0x7A 0xBC 0xAF 0x27 0x1C}
	 *
	 * @apiNote Requires Apache Commons Compress or similar library for decompression.
	 */
	SEVENZ(new int[]{ 0x37, 0x7A, 0xBC, 0xAF, 0x27, 0x1C }),
	/**
	 * LZ (LZIP) compression.
	 * <p>
	 * Magic bytes: {@code 0x4C 0x5A 0x49 0x50} ("LZIP")
	 *
	 * @apiNote Requires lzip library for decompression.
	 */
	LZIP(new int[]{ 0x4C, 0x5A, 0x49, 0x50 }),
	/**
	 * LZFSE (Lempel-Ziv Finite State Entropy) compression.
	 * <p>
	 * Magic bytes: {@code 0x62 0x76 0x78 0x32} ("bvx2")
	 *
	 * @apiNote Apple's compression format. Requires platform-specific library.
	 */
	LZFSE(new int[]{ 0x62, 0x76, 0x78, 0x32 }),
	/**
	 * RAR archive compression (version 1.5+).
	 * <p>
	 * Magic bytes: {@code 0x52 0x61 0x72 0x21 0x1A 0x07 0x00} ("Rar!")
	 *
	 * @apiNote Requires UnRAR library for decompression.
	 */
	RAR(new int[]{ 0x52, 0x61, 0x72, 0x21, 0x1A, 0x07, 0x00 });

	/**
	 * Gets the magic bytes (0-255) as integers that identify this compression format.
	 */
	private final int @NotNull [] magicBytes;

	/**
	 * Detects the compression type of the byte array by examining its magic bytes.
	 *
	 * @param data the byte array to examine
	 * @return the detected compression type, or {@link #NONE} if no compression is detected
	 */
	public static @NotNull Compression getType(byte[] data) {
		if (data == null || data.length == 0)
			return NONE;

		// Try to match against known compression formats
		for (Compression compression : values()) {
			if (compression == NONE)
				continue;

			if (data.length >= compression.magicBytes.length) {
				boolean matches = true;

				for (int i = 0; i < compression.magicBytes.length; i++) {
					if ((data[i] & 0xFF) != compression.magicBytes[i]) {
						matches = false;
						break;
					}
				}

				if (matches)
					return compression;
			}
		}

		return NONE;
	}

	/**
	 * Detects the compression type of the input stream by examining its magic bytes.
	 * <p>
	 * This method will mark and reset the stream, so the original data can still be read.
	 * If the stream does not support marking, it will be wrapped in a {@link BufferedInputStream}.
	 *
	 * @param inputStream the input stream to examine
	 * @return the detected compression type, or {@link #NONE} if no compression is detected
	 * @throws IOException if an I/O error occurs
	 */
	public static @NotNull Compression getType(@NotNull InputStream inputStream) throws IOException {
		// Ensure the stream supports marking
		if (!inputStream.markSupported())
			inputStream = new BufferedInputStream(inputStream);

		// Find the longest magic byte sequence to determine the mark limit
		int maxLength = 0;
		for (Compression compression : values()) {
			if (compression.magicBytes.length > maxLength)
				maxLength = compression.magicBytes.length;
		}

		// Read the magic bytes
		inputStream.mark(maxLength);
		byte[] buffer = new byte[maxLength];
		int bytesRead = inputStream.read(buffer, 0, maxLength);
		inputStream.reset();

		// If we couldn't read enough bytes, return NONE
		if (bytesRead <= 0)
			return NONE;

		return getType(buffer);
	}

	/**
	 * Decompresses a byte array if it's compressed.
	 * Returns the original array if no compression is detected.
	 *
	 * @param data the potentially compressed data
	 * @return decompressed data or original if not compressed
	 * @throws IOException if decompression fails
	 * @throws UnsupportedOperationException if the detected compression format is not supported
	 */
	public static byte[] decompress(byte[] data) throws IOException {
		Compression type = getType(data);

		if (type == NONE)
			return data;

		try (InputStream in = wrap(new ByteArrayDataInput(data));
			ByteArrayDataOutput out = new ByteArrayDataOutput()) {
			byte[] buffer = new byte[8192];
			int length;

			while ((length = in.read(buffer)) > 0)
				out.write(buffer, 0, length);

			return out.toByteArray();
		}
	}

	/**
	 * Wraps an input stream with the appropriate decompression stream based on its detected type.
	 *
	 * @param inputStream the input stream to wrap
	 * @return a decompression stream, or the original stream if no compression is detected
	 * @throws IOException if an I/O error occurs
	 * @throws UnsupportedOperationException if the detected compression format is not supported
	 * @apiNote Only GZIP and ZLIB are supported in the standard Java library. Other formats require
	 * additional dependencies.
	 */
	public static @NotNull InputStream wrap(@NotNull InputStream inputStream) throws IOException {
		return switch (Compression.getType(inputStream)) {
			case GZIP -> new GZIPInputStream(inputStream, 65536);
			case ZLIB -> new InflaterInputStream(inputStream);
			case BZIP2, XZ, LZ4, ZSTD, ZIP, SEVENZ, LZIP, LZFSE, RAR ->
				throw new UnsupportedOperationException("Compression format not supported without additional libraries");
			default -> inputStream;
		};
	}
}
