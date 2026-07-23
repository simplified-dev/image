package dev.simplified.image.codec.gif;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

/**
 * How faithfully emitted frame delays follow the source timing when a schedule asks for finer
 * resolution than GIF's centisecond grid can express.
 * <p>
 * {@link GifImageWriter} quantizes the cumulative timeline rather than each delay on its own, so
 * the declared total stays exact no matter how many frames there are. This setting governs the one
 * case quantizing alone cannot resolve - a frame whose share of the timeline rounds below the
 * smallest delay the audience honors - and it trades the two accuracies against each other:
 * <ul>
 * <li><b>What the file declares.</b> {@link #EXACT} keeps the declared total right for any
 * schedule averaging at least 10 ms per frame.</li>
 * <li><b>What a player shows.</b> {@link #PLAYABLE} never declares a delay that players rewrite,
 * at the cost of declaring a longer total than intended.</li>
 * </ul>
 * The two differ only below 20 ms per frame. At or above it, rounding yields at least 2 cs on its
 * own and both settings emit identical bytes.
 */
@Getter
@RequiredArgsConstructor
public enum DelayFidelity {

    /**
     * Never declares 1 cs, which every surveyed player substitutes 100 ms for - a tenfold
     * slowdown rather than the fast playback the value asks for. Frames that would round below
     * 2 cs are raised to it, so a strip finer than 20 ms per frame declares a longer total than
     * intended instead of playing at a tenth of its intended speed. The right choice for anything
     * a browser will play, and the default.
     */
    PLAYABLE(2),

    /**
     * Declares the source timing down to GIF's 1 cs minimum, keeping the declared total exact for
     * any schedule averaging at least 10 ms per frame. A 1 cs delay reaches tooling that reads the
     * file's own numbers intact, and is substituted with 100 ms by every browser surveyed, so this
     * suits files that get measured rather than watched.
     */
    EXACT(1);

    /**
     * the smallest delay in centiseconds the writer may declare
     */
    private final int minimumDelayCs;

}
