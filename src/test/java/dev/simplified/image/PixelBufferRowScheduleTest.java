package dev.simplified.image;

import dev.simplified.image.pixel.AlphaMode;
import dev.simplified.image.pixel.BlendMode;
import dev.simplified.image.pixel.Resample;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.ForkJoinWorkerThread;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.function.UnaryOperator;
import java.util.stream.IntStream;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.sameInstance;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

/**
 * Holds the two row loops pixel operations run through to their schedules: the shared loop called on
 * a fork-join worker runs every row on that worker in ascending order, and called off the pool hands
 * a range of at least {@value #THRESHOLD} rows to the pool and keeps a shorter one on the caller; the
 * loop that splits across the pool hands such a range to the pool from a worker as well, and keeps a
 * shorter one on that worker; and every operation writes the same bytes on a worker as off the pool.
 */
class PixelBufferRowScheduleTest {

    /** The row count from which a row loop called off the pool hands its rows to the pool. */
    private static final int THRESHOLD = 64;

    /** Width of the picture every operation is applied to. */
    private static final int WIDTH = 97;

    /**
     * Height of the picture every operation is applied to, tall enough that the row-pair loop of a
     * vertical flip covers {@value #THRESHOLD} pairs.
     */
    private static final int HEIGHT = 150;

    /** How long each row spins, so that a pool given forked rows has its idle workers take some. */
    private static final long SPIN_NANOS = 100_000;

    @Test
    @DisplayName("rows called on a fork-join worker run on that worker in ascending order")
    void rowsOnAWorkerRunOnThatWorkerInOrder() throws Exception {
        int rows = 4 * THRESHOLD;
        Thread[] ranOn = new Thread[rows];
        int[] order = new int[rows];
        AtomicInteger next = new AtomicInteger();

        Thread caller = onPoolWorker(() -> {
            PixelBuffer.forEachRow(0, rows, y -> {
                ranOn[y] = Thread.currentThread();
                order[next.getAndIncrement()] = y;
                spin();
            });
            return Thread.currentThread();
        });

        for (int y = 0; y < rows; y++)
            assertThat("row %d ran on the calling worker".formatted(y), ranOn[y], is(sameInstance(caller)));

        assertArrayEquals(IntStream.range(0, rows).toArray(), order, "rows in ascending order");
    }

    @Test
    @DisplayName("rows called off the pool reach a pool worker from the threshold and stay on the caller below it")
    void rowsOffThePoolReachThePoolFromTheThreshold() throws Exception {
        int below = THRESHOLD - 1;
        Thread[] ranOn = new Thread[below];
        int[] order = new int[below];
        AtomicInteger next = new AtomicInteger();

        Thread caller = onPlainThread(() -> {
            PixelBuffer.forEachRow(0, below, y -> {
                ranOn[y] = Thread.currentThread();
                order[next.getAndIncrement()] = y;
            });
            return Thread.currentThread();
        });

        for (int y = 0; y < below; y++)
            assertThat("row %d of %d ran on the caller".formatted(y, below), ranOn[y], is(sameInstance(caller)));

        assertArrayEquals(IntStream.range(0, below).toArray(), order, "rows below the threshold in ascending order");

        // The caller holds its first row until a worker has run one, or for ten seconds where none does.
        Set<Thread> threads = ConcurrentHashMap.newKeySet();
        CountDownLatch workerRan = new CountDownLatch(1);
        AtomicBoolean held = new AtomicBoolean();

        onPlainThread(() -> {
            Thread self = Thread.currentThread();
            PixelBuffer.forEachRow(0, THRESHOLD, y -> {
                threads.add(Thread.currentThread());
                if (Thread.currentThread() != self)
                    workerRan.countDown();
                else if (held.compareAndSet(false, true))
                    awaitWorker(workerRan);
            });
            return self;
        });

        assertThat("%d rows reached a pool worker".formatted(THRESHOLD),
            threads.stream().anyMatch(ForkJoinWorkerThread.class::isInstance), is(true));
    }

    @Test
    @DisplayName("rows split across the pool from a fork-join worker reach its other workers from the threshold and stay on the caller below it")
    void rowsSplitAcrossThePoolFromAWorkerReachItsOtherWorkers() throws Exception {
        int below = THRESHOLD - 1;
        Thread[] ranOn = new Thread[below];
        int[] order = new int[below];
        AtomicInteger next = new AtomicInteger();

        Thread caller = onPoolWorker(() -> {
            PixelBuffer.forEachRowAcrossPool(0, below, y -> {
                ranOn[y] = Thread.currentThread();
                order[next.getAndIncrement()] = y;
            });
            return Thread.currentThread();
        });

        for (int y = 0; y < below; y++)
            assertThat("row %d of %d ran on the calling worker".formatted(y, below), ranOn[y], is(sameInstance(caller)));

        assertArrayEquals(IntStream.range(0, below).toArray(), order, "rows below the threshold in ascending order");

        // The calling worker holds its first row until another worker has run one, or for ten seconds
        // where none does.
        Set<Thread> threads = ConcurrentHashMap.newKeySet();
        CountDownLatch workerRan = new CountDownLatch(1);
        AtomicBoolean held = new AtomicBoolean();

        Thread self = onPoolWorker(() -> {
            Thread worker = Thread.currentThread();
            PixelBuffer.forEachRowAcrossPool(0, THRESHOLD, y -> {
                threads.add(Thread.currentThread());
                if (Thread.currentThread() != worker)
                    workerRan.countDown();
                else if (held.compareAndSet(false, true))
                    awaitWorker(workerRan);
            });
            return worker;
        });

        assertThat("%d rows reached another pool worker".formatted(THRESHOLD),
            threads.stream().anyMatch(thread -> thread != self && thread instanceof ForkJoinWorkerThread), is(true));
    }

    @Test
    @DisplayName("every row operation and FXAA write the same bytes on a fork-join worker as off the pool")
    void everyRowOperationWritesTheSameBytesOnAWorker() throws Exception {
        Map<String, UnaryOperator<PixelBuffer>> operations = operations();
        PixelBuffer picture = PixelBuffer.of(texels(WIDTH, HEIGHT, 1), WIDTH, HEIGHT);

        Map<String, int[]> onWorker = onPoolWorker(() -> applyAll(operations, picture));
        Map<String, int[]> offPool = onPlainThread(() -> applyAll(operations, picture));

        assertThat(onWorker.keySet(), equalTo(operations.keySet()));

        for (String name : operations.keySet())
            assertArrayEquals(offPool.get(name), onWorker.get(name), name);
    }

    /**
     * Builds every operation that runs its rows through either row loop, each covering at least
     * {@value #THRESHOLD} rows of it, plus FXAA in place in both conventions. Each takes its own copy
     * of the picture and answers the buffer it wrote.
     *
     * @return the operations by name, in a fixed order
     */
    private static Map<String, UnaryOperator<PixelBuffer>> operations() {
        PixelBuffer other = PixelBuffer.of(texels(WIDTH, HEIGHT, 2), WIDTH, HEIGHT);
        PixelBuffer source = PixelBuffer.of(texels(80, 100, 3), 80, 100);
        PixelBuffer opaqueSource = PixelBuffer.of(texels(80, 100, 4), 80, 100, false);
        PixelBuffer large = PixelBuffer.of(texels(200, 180, 5), 200, 180);
        PixelBuffer small = PixelBuffer.of(texels(16, 16, 6), 16, 16);
        PixelBuffer filtered = PixelBuffer.of(texels(160, 140, 7), 160, 140);
        Map<String, UnaryOperator<PixelBuffer>> operations = new LinkedHashMap<>();

        operations.put("flipHorizontal", inPlace(PixelBuffer::flipHorizontal));
        operations.put("flipVertical", inPlace(PixelBuffer::flipVertical));
        operations.put("resize NEAREST", buffer -> buffer.resize(83, 131, Resample.NEAREST));
        operations.put("resize BILINEAR", buffer -> buffer.resize(83, 131, Resample.BILINEAR));
        operations.put("rotate90", PixelBuffer::rotate90);
        operations.put("rotate180", PixelBuffer::rotate180);
        operations.put("rotate270", PixelBuffer::rotate270);
        operations.put("fillRect", inPlace(buffer -> buffer.fillRect(-4, 7, 60, 120, 0x80C04020)));
        operations.put("grayscale", inPlace(PixelBuffer::grayscale));
        operations.put("invert", inPlace(PixelBuffer::invert));
        operations.put("multiplyAlpha", inPlace(buffer -> buffer.multiplyAlpha(0.6f)));
        operations.put("premultiplyAlpha", inPlace(PixelBuffer::premultiplyAlpha));
        operations.put("unpremultiplyAlpha", inPlace(PixelBuffer::unpremultiplyAlpha));
        operations.put("lerp", buffer -> PixelBuffer.lerp(buffer, other, 0.35f));
        operations.put("tint", buffer -> buffer.tint(0xFF80C0E0));

        for (BlendMode mode : BlendMode.values()) {
            operations.put("blit " + mode, inPlace(buffer -> buffer.blit(source, -5, 30, mode)));
            operations.put("blit without alpha " + mode, inPlace(buffer -> buffer.blit(opaqueSource, -5, 30, mode)));
            operations.put("blitTinted " + mode, inPlace(buffer -> buffer.blitTinted(source, -5, 30, 0xFF60A0F0, mode)));
            operations.put("blitScaled shrinking " + mode, inPlace(buffer -> buffer.blitScaled(large, 3, 10, 90, 120, mode)));
            operations.put("blitScaled magnifying " + mode, inPlace(buffer -> buffer.blitScaled(small, -7, 5, 120, 130, mode)));

            for (AlphaMode alphaMode : AlphaMode.values()) {
                operations.put("blitScaledFxaa %s %s".formatted(mode, alphaMode),
                    inPlace(buffer -> buffer.blitScaledFxaa(filtered, 2, 4, 80, 70, mode, alphaMode)));
            }
        }

        for (AlphaMode alphaMode : AlphaMode.values())
            operations.put("applyFxaa " + alphaMode, inPlace(buffer -> buffer.applyFxaa(alphaMode)));

        return operations;
    }

    /**
     * Applies every operation to its own copy of the picture on the calling thread.
     *
     * @param operations the operations by name
     * @param picture the picture each operation copies
     * @return the pixels each operation wrote, by name
     */
    private static Map<String, int[]> applyAll(Map<String, UnaryOperator<PixelBuffer>> operations, PixelBuffer picture) {
        Map<String, int[]> written = new LinkedHashMap<>();

        for (Map.Entry<String, UnaryOperator<PixelBuffer>> operation : operations.entrySet())
            written.put(operation.getKey(), operation.getValue().apply(picture.copy()).data());

        return written;
    }

    /**
     * Wraps an operation that writes a buffer in place as one that answers that buffer.
     *
     * @param operation the in-place operation
     * @return the operation answering the buffer it wrote
     */
    private static UnaryOperator<PixelBuffer> inPlace(Consumer<PixelBuffer> operation) {
        return buffer -> {
            operation.accept(buffer);
            return buffer;
        };
    }

    /**
     * Runs a task on a worker of a fresh fork-join pool and answers its result.
     *
     * @param task the task
     * @param <T> the result type
     * @return the task's result
     * @throws Exception if the task fails
     */
    private static <T> T onPoolWorker(Callable<T> task) throws Exception {
        try (ForkJoinPool pool = new ForkJoinPool(4)) {
            return pool.submit(() -> {
                assertThat("running on a fork-join worker", ForkJoinTask.inForkJoinPool(), is(true));
                return task.call();
            }).get();
        }
    }

    /**
     * Runs a task on a new platform thread outside every fork-join pool and answers its result.
     *
     * @param task the task
     * @param <T> the result type
     * @return the task's result
     * @throws Exception if the task fails
     */
    private static <T> T onPlainThread(Callable<T> task) throws Exception {
        FutureTask<T> future = new FutureTask<>(() -> {
            assertThat("running outside every fork-join pool", ForkJoinTask.inForkJoinPool(), is(false));
            return task.call();
        });
        Thread.ofPlatform().start(future);
        return future.get();
    }

    /**
     * Waits up to ten seconds for a pool worker to run a row.
     *
     * @param workerRan the latch a worker counts down when it runs a row
     */
    private static void awaitWorker(CountDownLatch workerRan) {
        try {
            workerRan.await(10, TimeUnit.SECONDS);
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
        }
    }

    /** Spins for {@link #SPIN_NANOS}. */
    private static void spin() {
        long until = System.nanoTime() + SPIN_NANOS;

        while (System.nanoTime() < until)
            Thread.onSpinWait();
    }

    /**
     * Draws random texels in four even shares: no alpha over colour channels that still hold bits,
     * full alpha, and two shares of alpha drawn from 1 to 254.
     *
     * @param width the picture width
     * @param height the picture height
     * @param seed the random seed
     * @return the pixels
     */
    private static int[] texels(int width, int height, int seed) {
        Random random = new Random(seed);
        int[] pixels = new int[width * height];

        for (int i = 0; i < pixels.length; i++) {
            int rgb = random.nextInt() & 0x00FFFFFF;
            int alpha = switch (random.nextInt(4)) {
                case 0 -> 0;
                case 1 -> 0xFF;
                default -> 1 + random.nextInt(254);
            };
            pixels[i] = alpha << 24 | rgb;
        }

        return pixels;
    }

}
