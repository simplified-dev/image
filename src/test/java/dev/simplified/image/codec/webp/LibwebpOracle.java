package dev.simplified.image.codec.webp;

import dev.simplified.image.PixelBuffer;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.junit.jupiter.api.Assumptions;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * The libwebp oracle the WebP tests hold this library's codecs to: libwebp itself, reached through
 * Python's {@code webp} package.
 * <p>
 * The interpreter is resolved once per JVM, from the candidates {@link #candidates} lists: the
 * {@value #PROPERTY} system property, or else the {@value #VARIABLE} environment variable, and then
 * {@code python3}, {@code python} and {@code py}. A candidate is taken only when it starts and imports
 * {@code webp} and {@code numpy}, so one that starts without them - a store stub, a bare install, a
 * virtual environment - falls through to the next. With none taken, every script aborts the calling
 * test, which therefore reports skipped rather than passed; a script that runs and exits non-zero
 * fails it.
 * <p>
 * Scripts take file paths and numbers as arguments, read from {@code sys.argv}, and never have them
 * spliced into their source.
 */
public final class LibwebpOracle {

    /** The system property naming the interpreter to try first. */
    public static final String PROPERTY = "vp8.pythonBin";

    /** The environment variable naming the interpreter to try first when {@link #PROPERTY} is unset. */
    public static final String VARIABLE = "VP8_PYTHON_BIN";

    /** The interpreters tried after the named one, in order. */
    private static final List<String> DEFAULTS = List.of("python3", "python", "py");

    /** The script a candidate runs to be taken. */
    private static final String PROBE = "import webp, numpy";

    /** Encodes raw RGBA - path, output path, width, height, quality - with libwebp's lossy encoder. */
    private static final String ENCODE_LOSSY = """
        import sys
        import numpy as np
        import webp
        raw, out, width, height, quality = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
        with open(raw, 'rb') as f:
            arr = np.frombuffer(f.read(), dtype=np.uint8).reshape(height, width, 4)
        pic = webp.WebPPicture.from_numpy(arr, pilmode='RGBA')
        data = pic.encode(webp.WebPConfig.new(quality=quality)).buffer()
        with open(out, 'wb') as f:
            f.write(bytes(data))
        """;

    /** Decodes a still WebP and prints its size, then one {@code AARRGGBB} line per pixel. */
    private static final String DECODE_ARGB = """
        import sys
        import webp
        img = webp.load_image(sys.argv[1]).convert('RGBA')
        w, h = img.size
        print(w, h)
        data = img.tobytes()
        for i in range(0, len(data), 4):
            r, g, b, a = data[i], data[i+1], data[i+2], data[i+3]
            print(f'{a:02x}{r:02x}{g:02x}{b:02x}')
        """;

    /** Decodes a still WebP and prints a {@code DIMS} line, then one {@code A} line per pixel. */
    private static final String DECODE_ALPHA = """
        import sys, warnings
        warnings.filterwarnings('ignore')
        import webp
        img = webp.load_image(sys.argv[1]).convert('RGBA')
        w, h = img.size
        print(f'DIMS {w} {h}')
        for px in img.getdata():
            print(f'A {px[3]}')
        """;

    /**
     * Decodes an animated WebP through {@code WebPAnimDecoder}, which {@code webp.load_image} does not
     * reach, and prints {@code DIMS} and {@code FRAMES} lines, then a {@code FRAME} line and one
     * {@code A} line per pixel for each frame.
     */
    private static final String DECODE_ANIMATED_ALPHA = """
        import sys, warnings
        warnings.filterwarnings('ignore')
        from webp import _webp, ffi
        with open(sys.argv[1], 'rb') as f:
            data = f.read()
        options = ffi.new('WebPAnimDecoderOptions*')
        _webp.lib.WebPAnimDecoderOptionsInit(options)
        wd = ffi.new('WebPData*')
        buf = ffi.new('uint8_t[]', data); wd.bytes = buf; wd.size = len(data)
        dec = _webp.lib.WebPAnimDecoderNew(wd, options)
        if dec == ffi.NULL:
            print('DECODE_FAIL'); sys.exit(3)
        info = ffi.new('WebPAnimInfo*')
        _webp.lib.WebPAnimDecoderGetInfo(dec, info)
        w, h = info.canvas_width, info.canvas_height
        print(f'DIMS {w} {h}')
        print(f'FRAMES {info.frame_count}')
        while _webp.lib.WebPAnimDecoderHasMoreFrames(dec):
            ptr = ffi.new('uint8_t**'); ts = ffi.new('int*')
            if _webp.lib.WebPAnimDecoderGetNext(dec, ptr, ts) == 0:
                print('FRAME_FAIL'); sys.exit(4)
            pixels = ffi.buffer(ptr[0], w * h * 4)[:]
            print('FRAME')
            for i in range(3, len(pixels), 4):
                print(f'A {pixels[i]}')
        _webp.lib.WebPAnimDecoderDelete(dec)
        """;

    private LibwebpOracle() { }

    /**
     * The outcome of trying each candidate in turn.
     *
     * @param interpreter the first candidate that started and imported the packages, if any
     * @param rejected each candidate passed over before it, with why
     */
    record Resolution(@NotNull Optional<String> interpreter, @NotNull List<String> rejected) {}

    /**
     * What a finished process printed, and how it exited.
     *
     * @param exit the exit code
     * @param text standard output and standard error, interleaved as written
     */
    private record Output(int exit, @NotNull String text) {}

    /** The resolution this JVM's tests share, made on first use. */
    private static final class Shared {

        private static final Resolution RESOLUTION = resolve(candidates(System.getProperty(PROPERTY), System.getenv(VARIABLE)));

    }

    /**
     * Lists the interpreters to try, the named one first.
     *
     * @param property the value of {@link #PROPERTY}, or {@code null} when it is unset
     * @param variable the value of {@link #VARIABLE}, or {@code null} when it is unset
     * @return the property when it is set and not blank, else the variable when it is, followed by
     *         {@code python3}, {@code python} and {@code py}
     */
    static @NotNull List<String> candidates(@Nullable String property, @Nullable String variable) {
        List<String> candidates = new ArrayList<>();
        if (property != null && !property.isBlank())
            candidates.add(property);
        else if (variable != null && !variable.isBlank())
            candidates.add(variable);
        candidates.addAll(DEFAULTS);
        return List.copyOf(candidates);
    }

    /**
     * Takes the first candidate that starts and imports {@code webp} and {@code numpy}.
     *
     * @param candidates the interpreters to try, in order
     * @return the one taken, if any, and why each one before it was passed over
     */
    static @NotNull Resolution resolve(@NotNull List<String> candidates) {
        List<String> rejected = new ArrayList<>();
        for (String candidate : candidates) {
            Optional<String> reason = probe(candidate);
            if (reason.isEmpty())
                return new Resolution(Optional.of(candidate), List.copyOf(rejected));
            rejected.add(candidate + ": " + reason.get());
        }
        return new Resolution(Optional.empty(), List.copyOf(rejected));
    }

    /**
     * Runs the probe under one candidate.
     *
     * @param candidate the interpreter
     * @return nothing when it starts and imports the packages, otherwise why it is passed over
     */
    private static @NotNull Optional<String> probe(@NotNull String candidate) {
        Process process;
        try {
            process = new ProcessBuilder(candidate, "-c", PROBE).redirectErrorStream(true).start();
        } catch (IOException ex) {
            return Optional.of("did not start (" + ex.getMessage() + ")");
        }
        Output output = finish(process);
        if (output.exit() == 0)
            return Optional.empty();
        List<String> lines = output.text().strip().lines().toList();
        return Optional.of("exited " + output.exit() + (lines.isEmpty() ? "" : " (" + lines.getLast() + ")"));
    }

    /**
     * Returns the interpreter this JVM's tests consult libwebp through, aborting the calling test when
     * there is none.
     *
     * @return the interpreter
     */
    public static @NotNull String interpreter() {
        return interpreter(Shared.RESOLUTION);
    }

    /**
     * Returns a resolution's interpreter, aborting the calling test when it has none.
     *
     * @param resolution the resolution
     * @return the interpreter
     */
    private static @NotNull String interpreter(@NotNull Resolution resolution) {
        return resolution.interpreter().orElseGet(() -> Assumptions.abort(
            "No Python interpreter imports webp and numpy, so libwebp cannot be consulted. Tried "
                + String.join("; ", resolution.rejected()) + ". Name one with -D" + PROPERTY + "=<path>"));
    }

    /**
     * Runs a script under the shared interpreter.
     *
     * @param what what the script does, which a failure is reported under
     * @param script the Python source, which reads its arguments from {@code sys.argv[1:]}
     * @param args the arguments
     * @return what the script printed, standard error included
     * @throws AssertionError when the script exits non-zero
     */
    public static @NotNull String run(@NotNull String what, @NotNull String script, @NotNull String... args) {
        return run(Shared.RESOLUTION, what, script, args);
    }

    /**
     * Runs a script under a resolution's interpreter, aborting the calling test when it has none.
     *
     * @param resolution the resolution
     * @param what what the script does, which a failure is reported under
     * @param script the Python source, which reads its arguments from {@code sys.argv[1:]}
     * @param args the arguments
     * @return what the script printed, standard error included
     * @throws AssertionError when the script exits non-zero
     */
    static @NotNull String run(@NotNull Resolution resolution, @NotNull String what, @NotNull String script, @NotNull String... args) {
        List<String> command = new ArrayList<>(List.of(interpreter(resolution), "-c", script));
        command.addAll(List.of(args));
        Output output;
        try {
            output = finish(new ProcessBuilder(command).redirectErrorStream(true).start());
        } catch (IOException ex) {
            throw new UncheckedIOException(ex);
        }
        if (output.exit() != 0)
            throw new AssertionError(what + " - the script exited " + output.exit() + ":\n" + output.text());
        return output.text();
    }

    /**
     * Encodes an image with libwebp's lossy encoder.
     *
     * @param src the image
     * @param quality libwebp's quality, from 0 to 100
     * @return the bare VP8 payload, its RIFF and chunk headers stripped
     * @throws IOException if a temporary file cannot be written or read
     */
    public static byte @NotNull [] encodeLossy(@NotNull PixelBuffer src, int quality) throws IOException {
        // The pixels go through a file rather than the script, which keeps a large source clear of the
        // Windows command-line length limit.
        Path raw = Files.createTempFile("libwebp-enc-raw-", ".bin");
        Path out = Files.createTempFile("libwebp-enc-", ".webp");
        try {
            byte[] rgba = new byte[src.width() * src.height() * 4];
            int i = 0;
            for (int y = 0; y < src.height(); y++) {
                for (int x = 0; x < src.width(); x++) {
                    int p = src.getPixel(x, y);
                    rgba[i++] = (byte) ((p >> 16) & 0xFF);
                    rgba[i++] = (byte) ((p >> 8) & 0xFF);
                    rgba[i++] = (byte) (p & 0xFF);
                    rgba[i++] = (byte) ((p >>> 24) & 0xFF);
                }
            }
            Files.write(raw, rgba);
            run("libwebp encode", ENCODE_LOSSY, raw.toString(), out.toString(),
                Integer.toString(src.width()), Integer.toString(src.height()), Integer.toString(quality));
            return vp8Payload(Files.readAllBytes(out));
        } finally {
            Files.deleteIfExists(raw);
            Files.deleteIfExists(out);
        }
    }

    /**
     * Decodes a still WebP with libwebp.
     *
     * @param riff the whole WebP file
     * @param expectedWidth the width libwebp must report
     * @param expectedHeight the height libwebp must report
     * @return the decoded {@code ARGB} pixels, row-major
     * @throws IOException if a temporary file cannot be written
     * @throws AssertionError when libwebp rejects the file or reports another size
     */
    public static int @NotNull [] decodeArgb(byte @NotNull [] riff, int expectedWidth, int expectedHeight) throws IOException {
        Path file = Files.createTempFile("libwebp-dec-", ".webp");
        try {
            Files.write(file, riff);
            String[] lines = run("libwebp rejected the WebP stream", DECODE_ARGB, file.toString()).strip().split("\\R");
            String[] dims = lines[0].split("\\s+");
            int w = Integer.parseInt(dims[0]);
            int h = Integer.parseInt(dims[1]);
            if (w != expectedWidth || h != expectedHeight)
                throw new AssertionError("libwebp decoded " + w + "x" + h + ", expected " + expectedWidth + "x" + expectedHeight);

            int[] pixels = new int[w * h];
            for (int i = 0; i < pixels.length; i++)
                pixels[i] = (int) Long.parseLong(lines[i + 1], 16);
            return pixels;
        } finally {
            Files.deleteIfExists(file);
        }
    }

    /**
     * Decodes the alpha channel of a still WebP with libwebp.
     *
     * @param file the WebP file
     * @param expectedWidth the width libwebp must report
     * @param expectedHeight the height libwebp must report
     * @return one alpha value from 0 to 255 per pixel, row-major
     * @throws AssertionError when libwebp rejects the file or reports another size
     */
    public static int @NotNull [] decodeAlpha(@NotNull Path file, int expectedWidth, int expectedHeight) {
        String output = run("libwebp rejected the WebP file", DECODE_ALPHA, file.toString());
        int w = -1, h = -1;
        int[] alpha = new int[expectedWidth * expectedHeight];
        int count = 0;
        for (String line : output.split("\\R")) {
            if (line.startsWith("DIMS ")) {
                String[] dims = line.substring(5).split("\\s+");
                w = Integer.parseInt(dims[0]);
                h = Integer.parseInt(dims[1]);
            } else if (line.startsWith("A ") && count < alpha.length) {
                alpha[count++] = Integer.parseInt(line.substring(2));
            }
        }
        if (w != expectedWidth || h != expectedHeight)
            throw new AssertionError("libwebp decoded " + w + "x" + h + ", expected " + expectedWidth + "x" + expectedHeight + "\n" + output);
        if (count != alpha.length)
            throw new AssertionError("libwebp gave " + count + " alpha values, expected " + alpha.length + "\n" + output);
        return alpha;
    }

    /**
     * Decodes the alpha channel of every frame of an animated WebP with libwebp's animation decoder.
     *
     * @param file the WebP file
     * @param expectedWidth the canvas width libwebp must report
     * @param expectedHeight the canvas height libwebp must report
     * @param expectedFrames the frame count libwebp must report
     * @return one alpha value from 0 to 255 per canvas pixel, row-major, for each frame
     * @throws AssertionError when libwebp rejects the file or reports another canvas or frame count
     */
    public static int @NotNull [] @NotNull [] decodeAnimatedAlpha(
        @NotNull Path file, int expectedWidth, int expectedHeight, int expectedFrames
    ) {
        String output = run("libwebp rejected the animated WebP", DECODE_ANIMATED_ALPHA, file.toString());
        int w = -1, h = -1, frameCount = -1;
        int[][] frames = new int[expectedFrames][expectedWidth * expectedHeight];
        int frame = -1;
        int count = 0;
        for (String line : output.split("\\R")) {
            if (line.startsWith("DIMS ")) {
                String[] dims = line.substring(5).split("\\s+");
                w = Integer.parseInt(dims[0]);
                h = Integer.parseInt(dims[1]);
            } else if (line.startsWith("FRAMES ")) {
                frameCount = Integer.parseInt(line.substring(7).trim());
            } else if (line.equals("FRAME")) {
                frame++;
                count = 0;
            } else if (line.startsWith("A ") && frame >= 0 && frame < expectedFrames && count < frames[frame].length) {
                frames[frame][count++] = Integer.parseInt(line.substring(2));
            }
        }
        if (w != expectedWidth || h != expectedHeight)
            throw new AssertionError("libwebp decoded a " + w + "x" + h + " canvas, expected " + expectedWidth + "x" + expectedHeight + "\n" + output);
        if (frameCount != expectedFrames)
            throw new AssertionError("libwebp decoded " + frameCount + " frames, expected " + expectedFrames + "\n" + output);
        return frames;
    }

    /**
     * Strips a WebP file down to the payload of its {@code VP8 } chunk.
     *
     * @param riff the whole WebP file
     * @return the bare VP8 payload
     * @throws AssertionError when the file holds no {@code VP8 } chunk
     */
    private static byte @NotNull [] vp8Payload(byte @NotNull [] riff) {
        int offset = 12;
        while (offset + 8 <= riff.length) {
            String tag = new String(riff, offset, 4, StandardCharsets.US_ASCII);
            int size = (riff[offset + 4] & 0xFF)
                | ((riff[offset + 5] & 0xFF) << 8)
                | ((riff[offset + 6] & 0xFF) << 16)
                | ((riff[offset + 7] & 0xFF) << 24);
            if (tag.equals("VP8 ")) {
                byte[] payload = new byte[size];
                System.arraycopy(riff, offset + 8, payload, 0, size);
                return payload;
            }
            offset += 8 + size + (size & 1);
        }
        throw new AssertionError("libwebp wrote no VP8 chunk");
    }

    /**
     * Reads everything a process prints, then waits for it to exit.
     *
     * @param process the process
     * @return what it printed, and its exit code
     */
    private static @NotNull Output finish(@NotNull Process process) {
        try {
            String text = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            return new Output(process.waitFor(), text);
        } catch (IOException ex) {
            throw new UncheckedIOException(ex);
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted waiting for Python", ex);
        }
    }

}
