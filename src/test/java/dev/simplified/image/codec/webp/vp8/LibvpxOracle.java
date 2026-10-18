package dev.simplified.image.codec.webp.vp8;

import dev.simplified.image.PixelBuffer;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.junit.jupiter.api.Assumptions;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * The libvpx oracle the VP8 inter-frame tests hold this library's codec to: libvpx itself, reached
 * through an {@code ffmpeg} build that carries libvpx's VP8 encoder and decoder.
 * <p>
 * The executable is resolved once per JVM: the {@value #PROPERTY} system property, or else the
 * {@value #VARIABLE} environment variable, and then {@code ffmpeg}. A candidate is taken only when it
 * starts and lists {@code libvpx} among both its encoders and its decoders, so a build without
 * libvpx falls through to the next. With none taken, every call aborts the calling test, which
 * therefore reports skipped rather than passed; a run that exits non-zero, or that prints any error,
 * fails it.
 * <p>
 * Frames travel through IVF files, the container libvpx's own tools read and write. The decoder runs
 * on four threads, which is the path that refuses a frame whose bool decoder reads past the end of a
 * partition rather than only marking it corrupt.
 */
final class LibvpxOracle {

    /** The system property naming the executable to try first. */
    static final String PROPERTY = "vp8.ffmpegBin";

    /** The environment variable naming the executable to try first when {@link #PROPERTY} is unset. */
    static final String VARIABLE = "VP8_FFMPEG_BIN";

    /** The executables tried after the named one, in order. */
    private static final List<String> DEFAULTS = List.of("ffmpeg");

    private LibvpxOracle() { }

    /**
     * One decoded frame's planes, each row-major at its own width.
     *
     * @param width the luma width
     * @param height the luma height
     * @param y the luma plane, {@code width * height} samples
     * @param u the U plane, {@code (width + 1) / 2} by {@code (height + 1) / 2} samples
     * @param v the V plane, the size of {@code u}
     */
    record Frame(int width, int height, byte @NotNull [] y, byte @NotNull [] u, byte @NotNull [] v) {}

    /**
     * The outcome of trying each candidate in turn.
     *
     * @param executable the first candidate that started and listed libvpx both ways, if any
     * @param rejected each candidate passed over before it, with why
     */
    private record Resolution(@NotNull Optional<String> executable, @NotNull List<String> rejected) {}

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
     * Lists the executables to try, the named one first.
     *
     * @param property the value of {@link #PROPERTY}, or {@code null} when it is unset
     * @param variable the value of {@link #VARIABLE}, or {@code null} when it is unset
     * @return the property when it is set and not blank, else the variable when it is, followed by
     *         {@code ffmpeg}
     */
    private static @NotNull List<String> candidates(@Nullable String property, @Nullable String variable) {
        List<String> candidates = new ArrayList<>();
        if (property != null && !property.isBlank())
            candidates.add(property);
        else if (variable != null && !variable.isBlank())
            candidates.add(variable);
        candidates.addAll(DEFAULTS);
        return List.copyOf(candidates);
    }

    /**
     * Takes the first candidate that starts and lists libvpx among both its encoders and decoders.
     *
     * @param candidates the executables to try, in order
     * @return the one taken, if any, and why each one before it was passed over
     */
    private static @NotNull Resolution resolve(@NotNull List<String> candidates) {
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
     * Asks one candidate for its encoders and decoders.
     *
     * @param candidate the executable
     * @return nothing when it lists libvpx both ways, otherwise why it is passed over
     */
    private static @NotNull Optional<String> probe(@NotNull String candidate) {
        for (String list : new String[] { "-encoders", "-decoders" }) {
            Output output;
            try {
                output = finish(new ProcessBuilder(candidate, "-hide_banner", list).redirectErrorStream(true).start());
            } catch (IOException ex) {
                return Optional.of("did not start (" + ex.getMessage() + ")");
            }
            if (output.exit() != 0)
                return Optional.of("exited " + output.exit() + " listing " + list.substring(1));
            boolean listed = output.text().lines()
                .map(line -> line.trim().split("\\s+"))
                .anyMatch(fields -> fields.length > 1 && fields[1].equals("libvpx"));
            if (!listed)
                return Optional.of("lists no libvpx among its " + list.substring(1));
        }
        return Optional.empty();
    }

    /**
     * Returns the executable this JVM's tests consult libvpx through, aborting the calling test when
     * there is none.
     *
     * @return the executable
     */
    private static @NotNull String executable() {
        Resolution resolution = Shared.RESOLUTION;
        return resolution.executable().orElseGet(() -> Assumptions.abort(
            "No ffmpeg lists libvpx among both its encoders and decoders, so libvpx cannot be consulted. Tried "
                + String.join("; ", resolution.rejected()) + ". Name one with -D" + PROPERTY + "=<path>"));
    }

    /**
     * Decodes a VP8 stream with libvpx.
     *
     * @param payloads the frames' VP8 payloads, a keyframe first
     * @param width the frame width
     * @param height the frame height
     * @return every frame libvpx decoded, in order
     * @throws IOException if a temporary file cannot be written or read
     * @throws AssertionError when libvpx reports an error or decodes another number of frames
     */
    static @NotNull List<Frame> decode(@NotNull List<byte[]> payloads, int width, int height) throws IOException {
        String ffmpeg = executable();
        Path ivf = Files.createTempFile("libvpx-dec-", ".ivf");
        Path yuv = Files.createTempFile("libvpx-dec-", ".yuv");
        try {
            Files.write(ivf, ivf(payloads, width, height));
            run("libvpx decode", ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-threads", "4",
                "-c:v", "libvpx", "-i", ivf.toString(), "-f", "rawvideo", "-pix_fmt", "yuv420p", yuv.toString());
            byte[] raw = Files.readAllBytes(yuv);
            int chromaW = (width + 1) / 2;
            int chromaH = (height + 1) / 2;
            int frameBytes = width * height + 2 * chromaW * chromaH;
            if (raw.length != payloads.size() * frameBytes)
                throw new AssertionError("libvpx decoded " + raw.length / (double) frameBytes + " frames, expected " + payloads.size());
            List<Frame> frames = new ArrayList<>();
            for (int offset = 0; offset < raw.length; offset += frameBytes) {
                byte[] y = new byte[width * height];
                byte[] u = new byte[chromaW * chromaH];
                byte[] v = new byte[chromaW * chromaH];
                System.arraycopy(raw, offset, y, 0, y.length);
                System.arraycopy(raw, offset + y.length, u, 0, u.length);
                System.arraycopy(raw, offset + y.length + u.length, v, 0, v.length);
                frames.add(new Frame(width, height, y, u, v));
            }
            return frames;
        } finally {
            Files.deleteIfExists(ivf);
            Files.deleteIfExists(yuv);
        }
    }

    /**
     * Encodes frames with libvpx at one fixed quantizer, a keyframe first and every later frame an
     * inter frame predicted from the frames before it, with no alternate reference.
     *
     * @param frames the frames, all the same size
     * @param quantizer libvpx's quantizer, from 0 to 63
     * @return each frame's VP8 payload, in order
     * @throws IOException if a temporary file cannot be written or read
     * @throws AssertionError when libvpx reports an error
     */
    static @NotNull List<byte[]> encode(@NotNull List<PixelBuffer> frames, int quantizer) throws IOException {
        String ffmpeg = executable();
        int width = frames.getFirst().width();
        int height = frames.getFirst().height();
        Path rgb = Files.createTempFile("libvpx-enc-", ".rgb");
        Path ivf = Files.createTempFile("libvpx-enc-", ".ivf");
        try {
            byte[] raw = new byte[frames.size() * width * height * 3];
            int i = 0;
            for (PixelBuffer frame : frames) {
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        int p = frame.getPixel(x, y);
                        raw[i++] = (byte) (p >> 16);
                        raw[i++] = (byte) (p >> 8);
                        raw[i++] = (byte) p;
                    }
                }
            }
            Files.write(rgb, raw);
            String q = Integer.toString(quantizer);
            run("libvpx encode", ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", width + "x" + height, "-r", "30", "-i", rgb.toString(),
                "-c:v", "libvpx", "-threads", "1", "-qmin", q, "-qmax", q, "-b:v", "10M", "-g", "1000",
                "-auto-alt-ref", "0", "-lag-in-frames", "0", "-f", "ivf", ivf.toString());
            return payloads(Files.readAllBytes(ivf));
        } finally {
            Files.deleteIfExists(rgb);
            Files.deleteIfExists(ivf);
        }
    }

    /**
     * Wraps VP8 payloads in an IVF file: a 32-byte header naming the codec and size, then each frame
     * behind its 4-byte size and 8-byte timestamp.
     *
     * @param payloads the frames' VP8 payloads
     * @param width the frame width
     * @param height the frame height
     * @return the IVF file
     */
    private static byte @NotNull [] ivf(@NotNull List<byte[]> payloads, int width, int height) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ByteBuffer header = ByteBuffer.allocate(32).order(ByteOrder.LITTLE_ENDIAN);
        header.put("DKIF".getBytes(StandardCharsets.US_ASCII)).putShort((short) 0).putShort((short) 32)
            .put("VP80".getBytes(StandardCharsets.US_ASCII)).putShort((short) width).putShort((short) height)
            .putInt(30).putInt(1).putInt(payloads.size()).putInt(0);
        out.writeBytes(header.array());
        for (int i = 0; i < payloads.size(); i++) {
            ByteBuffer frameHeader = ByteBuffer.allocate(12).order(ByteOrder.LITTLE_ENDIAN);
            frameHeader.putInt(payloads.get(i).length).putLong(i);
            out.writeBytes(frameHeader.array());
            out.writeBytes(payloads.get(i));
        }
        return out.toByteArray();
    }

    /**
     * Unwraps the VP8 payloads of an IVF file.
     *
     * @param ivf the IVF file
     * @return each frame's VP8 payload, in order
     */
    private static @NotNull List<byte[]> payloads(byte @NotNull [] ivf) {
        ByteBuffer buffer = ByteBuffer.wrap(ivf).order(ByteOrder.LITTLE_ENDIAN);
        List<byte[]> payloads = new ArrayList<>();
        for (int offset = buffer.getShort(6) & 0xFFFF; offset + 12 <= ivf.length; ) {
            int size = buffer.getInt(offset);
            byte[] payload = new byte[size];
            System.arraycopy(ivf, offset + 12, payload, 0, size);
            payloads.add(payload);
            offset += 12 + size;
        }
        return payloads;
    }

    /**
     * Runs ffmpeg and requires it to finish cleanly.
     *
     * @param what what the run does, which a failure is reported under
     * @param command the command line
     * @throws AssertionError when ffmpeg exits non-zero or prints anything at its error level
     */
    private static void run(@NotNull String what, @NotNull String... command) {
        Output output;
        try {
            output = finish(new ProcessBuilder(command).redirectErrorStream(true).start());
        } catch (IOException ex) {
            throw new UncheckedIOException(ex);
        }
        if (output.exit() != 0 || !output.text().isBlank())
            throw new AssertionError(what + " - ffmpeg exited " + output.exit() + ":\n" + output.text());
    }

    /**
     * Reads everything a process prints, then waits for it to exit.
     *
     * @param process the process
     * @return what it printed, and its exit code
     */
    private static @NotNull Output finish(@NotNull Process process) {
        try {
            process.getOutputStream().close();
            String text = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            return new Output(process.waitFor(), text);
        } catch (IOException ex) {
            throw new UncheckedIOException(ex);
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted waiting for ffmpeg", ex);
        }
    }

}
