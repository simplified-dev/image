/**
 * One frame of an image - its pixels, timing, offset, disposal and blend - how many times a sequence
 * of frames plays, and the tools that reshape one.
 * <p>
 * {@link dev.simplified.image.frame.ImageFrame ImageFrame} is one frame of an
 * {@link dev.simplified.image.AnimatedImageData AnimatedImageData}, carrying the
 * {@link dev.simplified.image.frame.FrameDisposal FrameDisposal} and
 * {@link dev.simplified.image.frame.FrameBlend FrameBlend} a player applies to it, and a
 * {@link dev.simplified.image.frame.LoopCount LoopCount} says how many times the animation repeats
 * after its first play. {@link dev.simplified.image.frame.FrameNormalizer FrameNormalizer} fits every
 * frame of an animation to one canvas by a {@link dev.simplified.image.frame.FitMode FitMode}.
 * <p>
 * A type belongs here when it describes or reshapes a frame, or says how a sequence of frames plays,
 * rather than holding a whole image; the forms a whole image is held as sit in
 * {@link dev.simplified.image the root package}.
 */
package dev.simplified.image.frame;
