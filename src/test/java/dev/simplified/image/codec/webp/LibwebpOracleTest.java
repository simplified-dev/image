package dev.simplified.image.codec.webp;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.opentest4j.TestAbortedException;

import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.startsWith;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * How {@link LibwebpOracle} picks its interpreter and hands a script its arguments.
 * <p>
 * This JVM's own {@code java} launcher plays an interpreter that starts but cannot import {@code webp}:
 * it starts, rejects {@code -c} and exits non-zero. A path nothing lives at plays one that does not start.
 * The tests that need a real interpreter abort without one, as every oracle check does.
 */
class LibwebpOracleTest {

    /** A candidate that starts and fails the probe. */
    private static final String STARTS_WITHOUT_WEBP = Path.of(System.getProperty("java.home"), "bin", "java").toString();

    /** A candidate that does not start. */
    private static final String DOES_NOT_START = Path.of(System.getProperty("java.io.tmpdir"), "no-such-python").toString();

    @Test
    @DisplayName("the named interpreter comes first, the property ahead of the variable, then python3, python and py")
    void namedInterpreterComesFirst() {
        assertThat(LibwebpOracle.candidates("/opt/py", "/env/py"), contains("/opt/py", "python3", "python", "py"));
        assertThat(LibwebpOracle.candidates(null, "/env/py"), contains("/env/py", "python3", "python", "py"));
        assertThat(LibwebpOracle.candidates(" ", "/env/py"), contains("/env/py", "python3", "python", "py"));
        assertThat(LibwebpOracle.candidates(null, ""), contains("python3", "python", "py"));
    }

    @Test
    @DisplayName("a candidate that starts but cannot import webp is passed over, as is one that does not start")
    void unusableCandidatesArePassedOver() {
        LibwebpOracle.Resolution resolution = LibwebpOracle.resolve(List.of(DOES_NOT_START, STARTS_WITHOUT_WEBP));

        assertThat(resolution.interpreter(), is(Optional.empty()));
        assertThat(resolution.rejected(), contains(
            startsWith(DOES_NOT_START + ": did not start"),
            startsWith(STARTS_WITHOUT_WEBP + ": exited ")
        ));
    }

    @Test
    @DisplayName("the first candidate that imports webp is taken, past one that starts without it")
    void firstUsableCandidateIsTaken() {
        String usable = LibwebpOracle.interpreter();

        assertThat(LibwebpOracle.resolve(List.of(STARTS_WITHOUT_WEBP, usable)).interpreter(), is(Optional.of(usable)));
    }

    @Test
    @DisplayName("with no usable candidate a script aborts the test, naming each candidate tried")
    void noUsableCandidateAborts() {
        LibwebpOracle.Resolution none = LibwebpOracle.resolve(List.of(DOES_NOT_START, STARTS_WITHOUT_WEBP));

        TestAbortedException aborted = assertThrows(TestAbortedException.class, () -> LibwebpOracle.run(none, "a probe", "print(1)"));
        assertThat(aborted.getMessage(), containsString(DOES_NOT_START));
        assertThat(aborted.getMessage(), containsString(STARTS_WITHOUT_WEBP));
    }

    @Test
    @DisplayName("a script reads its arguments from sys.argv, so a path with a quote in it reaches the script whole")
    void argumentsReachTheScriptWhole() {
        String path = Path.of(System.getProperty("java.io.tmpdir"), "it's a dir", "frame.webp").toString();

        assertThat(LibwebpOracle.run("an echo", "import sys\nprint(sys.argv[1])\n", path).strip(), is(path));
    }

    @Test
    @DisplayName("a script that exits non-zero fails the test with what it printed")
    void failingScriptFails() {
        LibwebpOracle.interpreter();
        AssertionError failed = assertThrows(AssertionError.class,
            () -> LibwebpOracle.run("a failing script", "import sys\nprint('rejected')\nsys.exit(3)\n"));

        assertThat(failed.getMessage(), startsWith("a failing script"));
        assertThat(failed.getMessage(), containsString("rejected"));
    }

}
