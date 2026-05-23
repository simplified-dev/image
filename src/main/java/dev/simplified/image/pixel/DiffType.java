package dev.simplified.image.pixel;

/**
 * Per-pixel difference visualisations between two {@link PixelBuffer}s.
 * <p>
 * Mode semantics. For each pair of ARGB pixels {@code a} (left side) and {@code b} (right side),
 * the output pixel's interpretation depends on the selected mode. Pixels where both inputs are
 * fully transparent always produce a transparent output, regardless of mode - the diff cell carries
 * no signal outside the union silhouette and the caller chooses how to render that empty region
 * (typical choice: composite the diff over a grid-checker background). Pixels where exactly one
 * input has alpha (coverage mismatch) always produce a magenta marker in the colour modes; the
 * {@link #COVERAGE} mode breaks coverage out per-side with distinct magenta / cyan colours.
 *
 * <p>The diff direction is {@code a - b}: for {@link #SIGNED_LUMA} and {@link #SIGNED_RGB} a value
 * pushed past the mid-grey baseline means {@code a} exceeded {@code b} on that channel; for
 * {@link #COVERAGE} magenta marks an {@code a}-only pixel and cyan marks a {@code b}-only pixel.
 * Caller-defined orientation - typically {@code a} is the reference image and {@code b} is the
 * pipeline output.
 */
public enum DiffType {

    /**
     * Per-channel absolute difference, amplified ×4 for visibility.
     * <p>
     * Match within silhouette → faint dark grey {@code (0x10, 0x10, 0x10)} so the
     * "checked here, no diff" cells stay distinct from the transparent canvas around the
     * silhouette when the diff buffer is composited over a grid background. Coverage mismatch →
     * magenta. Otherwise each output channel is {@code min(255, |a.c - b.c| * 4)}. The 4× factor
     * makes single-bit deltas (which are otherwise invisible to the eye) read as a clear colour;
     * channels saturate at a raw delta of 64 / 255. Operates on raw channel values - AA edge
     * spill at semi-transparent pixels can read as a large delta even though the eye perceives
     * the two pixels as similar; see {@link #OVER_WHITE} for the perceptually-corrected variant.
     */
    ABSOLUTE,

    /**
     * Per-channel absolute difference after each pixel has been composited onto a fully-opaque
     * white background, amplified ×4 for visibility.
     * <p>
     * Match → transparent (out-of-silhouette and in-silhouette match collapse to the same
     * transparent output, suitable for the diff buffer being viewed directly without a
     * background layer). Coverage mismatch → magenta. Otherwise each output channel is
     * {@code min(255, |composite(a.c, a.alpha) - composite(b.c, b.alpha)| * 4)} where
     * {@code composite(c, a) = c * a/255 + 255 * (1 - a/255)}.
     * <p>
     * Compositing-over-white gives the same per-channel diff a viewer would perceive looking at
     * the two PNGs on the default white viewer background - alpha edge spill at semi-transparent
     * pixels no longer reads as a huge raw RGB delta when both pixels are nearly transparent.
     * This is the perceptually-accurate metric over raw {@link #ABSOLUTE} channel deltas.
     */
    OVER_WHITE,

    /**
     * Signed perceptual luminance delta on a red↔blue divergent palette.
     * <p>
     * Match → mid-grey {@code (128, 128, 128)}. {@code a} brighter than {@code b} → warm shift
     * toward red. {@code b} brighter than {@code a} → cool shift toward blue. Magnitude amplified
     * ×2 so a luma delta of 30 saturates the palette. Coverage mismatch → magenta.
     * <p>
     * Lighting bugs (which scale all three channels uniformly) show as solid-coloured regions in
     * this mode - the {@link #SIGNED_RGB} mode collapses those to grey because the per-channel
     * deltas balance out.
     */
    SIGNED_LUMA,

    /**
     * Per-channel signed delta centred at mid-grey.
     * <p>
     * Each output channel encodes {@code clamp(0, 255, 128 + (a.c - b.c) * 2)}. Match → grey;
     * a per-channel shift tints the output in the direction of the channel that diverged. Match
     * regions where {@code a} and {@code b} agree exactly come out as mid-grey; the test pixel
     * is opaque, distinguishing it from the transparent canvas around the silhouette. Coverage
     * mismatch → magenta.
     * <p>
     * Hue shifts that the {@link #SIGNED_LUMA} mode collapses (e.g. wrong tint colour, sampled
     * from the wrong texture region) show as solid colour tints in this mode.
     */
    SIGNED_RGB,

    /**
     * Coverage-only silhouette difference, ignoring colour entirely.
     * <p>
     * {@code a}-only pixels → magenta. {@code b}-only pixels → cyan. Both-opaque
     * silhouette interior → flat dark grey {@code 0xFF303034}. Both-transparent → transparent.
     * <p>
     * Surfaces "missing geometry" or "extra geometry" failures cleanly without any colour signal
     * to distract.
     */
    COVERAGE

}
