/**
 * What the library throws.
 * <p>
 * {@link dev.simplified.image.exception.ImageException ImageException} is the root, and each subtype
 * names the stage that failed: {@link dev.simplified.image.exception.ImageDecodeException
 * ImageDecodeException} for a read, {@link dev.simplified.image.exception.ImageEncodeException
 * ImageEncodeException} for a write, and {@link dev.simplified.image.exception.UnsupportedFormatException
 * UnsupportedFormatException} for a format no built-in codec handles.
 */
package dev.simplified.image.exception;
