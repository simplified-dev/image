package dev.simplified.image.frame;

import dev.simplified.annotations.AccessLevel;
import dev.simplified.annotations.EnumLookup;
import dev.simplified.annotations.Getter;
import dev.simplified.annotations.KeyField;
import dev.simplified.annotations.NamingStyle;
import dev.simplified.annotations.NoArgsConstructor;
import dev.simplified.annotations.RequiredArgsConstructor;
import org.jetbrains.annotations.NotNull;

import java.util.OptionalInt;

/**
 * How many times an animation repeats after its first play.
 * <p>
 * A loop count counts repeats, never plays: {@link Preset#NONE} plays once, {@link Preset#ONCE}
 * plays twice, and a count of {@code n} plays {@code n + 1} times. {@link #plays()} answers the
 * total for a caller who thinks in plays.
 * <p>
 * Two flavours are supported:
 * <ul>
 * <li><b>{@link Preset}</b> - the three counts a caller names most: never repeat, repeat once and
 * repeat forever. Callers normally reference these directly (e.g. {@code LoopCount.Preset.INFINITE}).</li>
 * <li><b>{@link Custom}</b> - a record for any other finite count. Construct via {@link #of(int)}, or
 * the {@link #builder() builder} to state the count as plays instead.</li>
 * </ul>
 * Each container declares a loop count in its own unit and its codec converts: a GIF's NETSCAPE2.0
 * application extension counts repeats, {@code 0} repeating forever, and a GIF without one plays
 * once; a WebP's {@code ANIM} chunk counts plays, {@code 0} playing forever.
 */
public sealed interface LoopCount {

    /**
     * The number of times the animation repeats after its first play, empty when it repeats forever.
     *
     * @return the repeat count, or empty for endless repeats
     */
    @NotNull OptionalInt repeats();

    /**
     * The number of times the animation plays in all, its first play included, empty when it plays
     * forever.
     * <p>
     * Endless play is the empty answer rather than a sentinel count, so a caller cannot do
     * arithmetic on it without first deciding what forever means to it.
     *
     * @return one more than {@link #repeats()}, or empty for endless play
     */
    default @NotNull OptionalInt plays() {
        OptionalInt repeats = this.repeats();
        return repeats.isPresent() ? OptionalInt.of(repeats.getAsInt() + 1) : OptionalInt.empty();
    }

    /**
     * Returns the loop count that repeats the given number of times after the first play.
     *
     * @param repeats the number of repeats after the first play, zero or more
     * @return {@link Preset#NONE} for {@code 0}, {@link Preset#ONCE} for {@code 1}, and a
     *     {@link Custom} for any other count
     * @throws IllegalArgumentException when {@code repeats} is negative or {@link Integer#MAX_VALUE}
     */
    static @NotNull LoopCount of(int repeats) {
        return Preset.findByRepeats(OptionalInt.of(repeats))
            .map(LoopCount.class::cast)
            .orElseGet(() -> new Custom(repeats));
    }

    /**
     * Creates a builder for a finite loop count, stated as repeats or as plays.
     *
     * @return a new builder, which builds {@link Preset#NONE} until a count is set
     */
    static @NotNull Builder builder() {
        return new Builder();
    }

    /**
     * The loop counts a caller names most.
     */
    @EnumLookup
    @Getter(style = NamingStyle.FLUENT)
    @RequiredArgsConstructor
    enum Preset implements LoopCount {

        /**
         * Plays once and never repeats.
         */
        NONE(OptionalInt.of(0)),

        /**
         * Repeats once, so plays twice.
         */
        ONCE(OptionalInt.of(1)),

        /**
         * Repeats forever.
         */
        INFINITE(OptionalInt.empty());

        /**
         * The number of times the animation repeats after its first play, empty when it repeats
         * forever.
         */
        @KeyField(strictKeys = true)
        private final @NotNull OptionalInt repeats;

    }

    /**
     * A finite loop count of the caller's choosing.
     * <p>
     * A count a preset names may be held here too, and then repeats as the preset does without
     * equalling it; {@link LoopCount#of(int)} and the {@link Builder} answer the preset for such a
     * count.
     *
     * @param repeatCount the number of times the animation repeats after its first play, from
     *     {@code 0} to one below {@link Integer#MAX_VALUE}, so that its plays fit an {@code int}
     */
    record Custom(int repeatCount) implements LoopCount {

        /**
         * Refuses a repeat count outside the range its plays fit in.
         *
         * @throws IllegalArgumentException when {@code repeatCount} is negative or
         *     {@link Integer#MAX_VALUE}
         */
        public Custom {
            if (repeatCount < 0 || repeatCount == Integer.MAX_VALUE)
                throw new IllegalArgumentException(String.format("Repeat count '%s' is outside [0, %s]", repeatCount, Integer.MAX_VALUE - 1));
        }

        /** {@inheritDoc} */
        @Override
        public @NotNull OptionalInt repeats() {
            return OptionalInt.of(this.repeatCount);
        }

    }

    /**
     * Builds a finite {@link LoopCount}, stated as repeats after the first play or as plays in all.
     */
    @NoArgsConstructor(access = AccessLevel.PACKAGE)
    final class Builder {

        private int repeats = 0;

        /**
         * Sets the number of times the animation repeats after its first play.
         *
         * @param repeats the repeat count, zero or more
         * @return this builder for chaining
         */
        public @NotNull Builder withRepeats(int repeats) {
            this.repeats = repeats;
            return this;
        }

        /**
         * Sets the number of times the animation plays in all, its first play included.
         *
         * @param plays the play count, one or more
         * @return this builder for chaining
         * @throws IllegalArgumentException when {@code plays} is below one
         */
        public @NotNull Builder withPlays(int plays) {
            if (plays < 1)
                throw new IllegalArgumentException(String.format("Play count '%s' is below one", plays));

            this.repeats = plays - 1;
            return this;
        }

        /**
         * Builds the loop count the last count set describes.
         *
         * @return the preset for a count one names, otherwise a {@link Custom}
         * @throws IllegalArgumentException when the repeat count set is negative or
         *     {@link Integer#MAX_VALUE}
         */
        public @NotNull LoopCount build() {
            return LoopCount.of(this.repeats);
        }

    }

}
