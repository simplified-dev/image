package dev.simplified.image.pixel;

import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.experimental.UtilityClass;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.awt.image.BufferedImage;
import java.lang.ref.SoftReference;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.Map;

/**
 * A thread-local pool of {@link PixelBuffer} instances keyed by dimensions.
 * <p>
 * Hot rasterization paths allocate large scratch {@link PixelBuffer}s (SSAA hi-res
 * intermediates, shading masks, downsample inputs) inside tight per-tile loops. The
 * backing {@code int[]} escapes every call - the buffer is handed to a {@link BufferedImage}
 * converter or composited into an output frame - so HotSpot's escape analysis cannot
 * stack-allocate it. This pool lets callers reuse a buffer across many calls on the
 * same worker thread instead of allocating per call.
 *
 * <h2>Lifecycle</h2>
 * <ul>
 *   <li>{@link #acquire} returns a {@link Lease} - an {@link AutoCloseable} wrapping a
 *       fresh or pooled {@link PixelBuffer} of the requested dimensions.</li>
 *   <li>{@link Lease#close} clears the buffer to transparent black and returns it to the
 *       per-thread pool. Double-close is a no-op; use-after-close throws
 *       {@link IllegalStateException}.</li>
 *   <li>The pool is bounded at {@value #MAX_PER_SIZE_CLASS} buffers per size class per
 *       thread - releases past the cap drop to GC.</li>
 *   <li>Pooled buffers are held through {@link SoftReference}s, so idle pool memory
 *       yields under GC pressure.</li>
 * </ul>
 *
 * <h2>Thread model</h2>
 * <p>
 * Each thread has its own pool map - acquire/release is lock-free. A lease may be closed
 * on a different thread than it was acquired on; the buffer simply lands in the closing
 * thread's pool rather than the acquiring thread's. Callers that want a buffer to escape
 * their calling scope should call {@link PixelBuffer#create} directly instead - pooling
 * an escape-path buffer forces an extra deep copy that defeats the point.
 *
 * <h2>Typical use</h2>
 * <pre>{@code
 * try (PixelBufferPool.Lease lease = PixelBufferPool.acquire(512, 512)) {
 *     PixelBuffer buffer = lease.buffer();
 *     engine.rasterize(triangles, buffer, ...);
 *     output.blitScaled(buffer, 0, 0, outputSize, outputSize);
 * }
 * }</pre>
 */
@UtilityClass
public final class PixelBufferPool {

    /**
     * Maximum number of pooled buffers per {@code (width, height)} size class per thread.
     * Releases past this cap drop to GC so the pool never grows unboundedly on a worker
     * that handles many distinct sizes.
     */
    static final int MAX_PER_SIZE_CLASS = 4;

    private static final ThreadLocal<Map<Long, Deque<SoftReference<PixelBuffer>>>> LOCAL_POOL =
        ThreadLocal.withInitial(HashMap::new);

    /**
     * Acquires a {@link PixelBuffer} of the given dimensions, either reusing a pooled
     * buffer on the current thread or allocating a fresh one.
     * <p>
     * The returned buffer is guaranteed to be zero-filled (transparent black) - pool
     * releases clear the contents before returning it to the pool.
     *
     * @param width the buffer width in pixels, must be positive
     * @param height the buffer height in pixels, must be positive
     * @return a lease wrapping the acquired buffer
     * @throws IllegalArgumentException if {@code width} or {@code height} is not positive
     */
    public static @NotNull Lease acquire(int width, int height) {
        if (width <= 0 || height <= 0)
            throw new IllegalArgumentException("Dimensions must be positive, got %dx%d".formatted(width, height));

        long key = packKey(width, height);
        Deque<SoftReference<PixelBuffer>> deque = LOCAL_POOL.get().get(key);
        if (deque != null) {
            while (!deque.isEmpty()) {
                SoftReference<PixelBuffer> ref = deque.pop();
                PixelBuffer buffer = ref.get();
                if (buffer != null) return new Lease(buffer, key);
            }
        }
        return new Lease(PixelBuffer.create(width, height), key);
    }

    /**
     * Returns a buffer to the current thread's pool after clearing it to transparent
     * black. Called by {@link Lease#close}.
     */
    static void release(@NotNull PixelBuffer buffer, long key) {
        buffer.fill(0);
        Map<Long, Deque<SoftReference<PixelBuffer>>> local = LOCAL_POOL.get();
        Deque<SoftReference<PixelBuffer>> deque = local.computeIfAbsent(key, k -> new ArrayDeque<>(MAX_PER_SIZE_CLASS));
        if (deque.size() < MAX_PER_SIZE_CLASS)
            deque.push(new SoftReference<>(buffer));
    }

    /**
     * Packs {@code (width, height)} into a single {@code long} key for the pool map.
     */
    private static long packKey(int width, int height) {
        return ((long) width << 32) | (height & 0xFFFFFFFFL);
    }

    /**
     * Clears the current thread's pool. Package-private for test isolation so each
     * test method starts with an empty pool regardless of prior test state.
     */
    static void clearThreadPool() {
        LOCAL_POOL.get().clear();
    }

    /**
     * A borrowed {@link PixelBuffer} whose {@link #close} returns the buffer to the pool.
     * <p>
     * {@code Lease} is not thread-safe - a lease is intended to be used and closed on a
     * single thread. Double-close is tolerated as a no-op so try-with-resources around a
     * body that manually closes still works; use-after-close is a loud error since it
     * indicates the caller is reading a buffer that may already have been handed to
     * another lease.
     */
    @AllArgsConstructor(access = AccessLevel.PACKAGE)
    public static final class Lease implements AutoCloseable {

        private @Nullable PixelBuffer buffer;
        private final long key;

        /**
         * Returns the underlying {@link PixelBuffer} for the duration of the lease.
         *
         * @return the leased buffer
         * @throws IllegalStateException if the lease has already been closed
         */
        public @NotNull PixelBuffer buffer() {
            PixelBuffer local = this.buffer;
            if (local == null)
                throw new IllegalStateException("PixelBufferPool.Lease is already closed - use-after-close");
            return local;
        }

        /**
         * Clears the buffer to transparent black and returns it to the current thread's
         * pool. Subsequent calls are no-ops.
         */
        @Override
        public void close() {
            PixelBuffer local = this.buffer;
            if (local == null) return;
            this.buffer = null;
            PixelBufferPool.release(local, key);
        }

    }

}
