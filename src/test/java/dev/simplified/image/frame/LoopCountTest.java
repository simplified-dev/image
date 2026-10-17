package dev.simplified.image.frame;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.OptionalInt;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.instanceOf;
import static org.hamcrest.Matchers.sameInstance;
import static org.junit.jupiter.api.Assertions.assertThrows;

class LoopCountTest {

    @Test
    @DisplayName("each preset counts the repeats after the first play")
    void presetsCountRepeats() {
        assertThat(LoopCount.Preset.NONE.repeats(), equalTo(OptionalInt.of(0)));
        assertThat(LoopCount.Preset.ONCE.repeats(), equalTo(OptionalInt.of(1)));
        assertThat(LoopCount.Preset.INFINITE.repeats(), equalTo(OptionalInt.empty()));
    }

    @Test
    @DisplayName("plays counts the first play too, and is empty for endless play")
    void playsCountTheFirstPlay() {
        assertThat(LoopCount.Preset.NONE.plays(), equalTo(OptionalInt.of(1)));
        assertThat(LoopCount.Preset.ONCE.plays(), equalTo(OptionalInt.of(2)));
        assertThat(LoopCount.Preset.INFINITE.plays(), equalTo(OptionalInt.empty()));
        assertThat(LoopCount.of(5).plays(), equalTo(OptionalInt.of(6)));
        assertThat(LoopCount.of(Integer.MAX_VALUE - 1).plays(), equalTo(OptionalInt.of(Integer.MAX_VALUE)));
    }

    @Test
    @DisplayName("of answers the preset for the counts a preset names, and a Custom for any other")
    void ofAnswersThePresets() {
        assertThat(LoopCount.of(0), sameInstance(LoopCount.Preset.NONE));
        assertThat(LoopCount.of(1), sameInstance(LoopCount.Preset.ONCE));
        assertThat(LoopCount.of(2), instanceOf(LoopCount.Custom.class));
        assertThat(LoopCount.of(2), equalTo(new LoopCount.Custom(2)));
        assertThat(LoopCount.of(2).repeats(), equalTo(OptionalInt.of(2)));
        assertThat(LoopCount.of(65535).repeats(), equalTo(OptionalInt.of(65535)));
    }

    @Test
    @DisplayName("a Custom refuses a count a preset names, so every count has one spelling")
    void customRefusesAPresetsCount() {
        IllegalArgumentException none = assertThrows(IllegalArgumentException.class, () -> new LoopCount.Custom(0));
        IllegalArgumentException once = assertThrows(IllegalArgumentException.class, () -> new LoopCount.Custom(1));

        assertThat(none.getMessage(), equalTo("Repeat count '0' is a preset's; use LoopCount.of"));
        assertThat(once.getMessage(), equalTo("Repeat count '1' is a preset's; use LoopCount.of"));
        assertThat(new LoopCount.Custom(2).repeats(), equalTo(OptionalInt.of(2)));
    }

    @Test
    @DisplayName("a count below zero, or one whose plays would not fit an int, is refused")
    void outOfRangeCountsAreRefused() {
        assertThrows(IllegalArgumentException.class, () -> LoopCount.of(-1));
        assertThrows(IllegalArgumentException.class, () -> new LoopCount.Custom(-1));
        assertThrows(IllegalArgumentException.class, () -> LoopCount.of(Integer.MAX_VALUE));
        assertThrows(IllegalArgumentException.class, () -> new LoopCount.Custom(Integer.MAX_VALUE));
    }

    @Test
    @DisplayName("the builder states a count as repeats or as plays, and answers the preset where one names it")
    void builderStatesRepeatsOrPlays() {
        assertThat(LoopCount.builder().build(), sameInstance(LoopCount.Preset.NONE));
        assertThat(LoopCount.builder().withRepeats(1).build(), sameInstance(LoopCount.Preset.ONCE));
        assertThat(LoopCount.builder().withRepeats(3).build(), equalTo(LoopCount.of(3)));
        assertThat(LoopCount.builder().withPlays(1).build(), sameInstance(LoopCount.Preset.NONE));
        assertThat(LoopCount.builder().withPlays(2).build(), sameInstance(LoopCount.Preset.ONCE));
        assertThat(LoopCount.builder().withPlays(4).build(), equalTo(LoopCount.of(3)));
        assertThat(LoopCount.builder().withPlays(4).withRepeats(7).build(), equalTo(LoopCount.of(7)));
    }

    @Test
    @DisplayName("the builder refuses a count that plays no time at all")
    void builderRefusesNoPlays() {
        assertThrows(IllegalArgumentException.class, () -> LoopCount.builder().withPlays(0));
        assertThrows(IllegalArgumentException.class, () -> LoopCount.builder().withPlays(Integer.MIN_VALUE));
        assertThrows(IllegalArgumentException.class, () -> LoopCount.builder().withRepeats(-1).build());
    }

}
