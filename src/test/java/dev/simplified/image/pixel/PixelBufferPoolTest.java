package dev.simplified.image.pixel;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.allOf;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;
import static org.hamcrest.Matchers.sameInstance;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

class PixelBufferPoolTest {

    @BeforeEach
    void clearPool() {
        PixelBufferPool.clearThreadPool();
    }

    @Test
    @DisplayName("acquire returns a buffer with the requested dimensions, zero-filled")
    void acquireReturnsRequestedDimensions() {
        try (PixelBufferPool.Lease lease = PixelBufferPool.acquire(32, 16)) {
            PixelBuffer buffer = lease.buffer();
            assertThat(buffer.width(), equalTo(32));
            assertThat(buffer.height(), equalTo(16));
            for (int y = 0; y < buffer.height(); y++)
                for (int x = 0; x < buffer.width(); x++)
                    assertThat(buffer.getPixel(x, y), equalTo(0));
        }
    }

    @Test
    @DisplayName("acquire rejects non-positive dimensions")
    void acquireRejectsInvalidDimensions() {
        assertThrows(IllegalArgumentException.class, () -> PixelBufferPool.acquire(0, 16));
        assertThrows(IllegalArgumentException.class, () -> PixelBufferPool.acquire(16, 0));
        assertThrows(IllegalArgumentException.class, () -> PixelBufferPool.acquire(-1, 16));
    }

    @Test
    @DisplayName("acquire after release of matching size reuses the pooled buffer")
    void acquireReusesPooledBuffer() {
        PixelBuffer first;
        try (PixelBufferPool.Lease lease = PixelBufferPool.acquire(64, 64)) {
            first = lease.buffer();
        }
        try (PixelBufferPool.Lease lease = PixelBufferPool.acquire(64, 64)) {
            assertThat(lease.buffer(), sameInstance(first));
        }
    }

    @Test
    @DisplayName("acquire with a different size does not return a mismatched pooled buffer")
    void acquireRespectsSizeClass() {
        PixelBuffer first;
        try (PixelBufferPool.Lease lease = PixelBufferPool.acquire(64, 64)) {
            first = lease.buffer();
        }
        try (PixelBufferPool.Lease lease = PixelBufferPool.acquire(32, 32)) {
            PixelBuffer other = lease.buffer();
            assertThat(other, is(not(sameInstance(first))));
            assertThat(other.width(), equalTo(32));
            assertThat(other.height(), equalTo(32));
        }
    }

    @Test
    @DisplayName("release clears buffer contents to transparent black")
    void releaseClearsContents() {
        PixelBuffer first;
        try (PixelBufferPool.Lease lease = PixelBufferPool.acquire(8, 8)) {
            first = lease.buffer();
            first.fill(0xFFABCDEF);
        }
        try (PixelBufferPool.Lease lease = PixelBufferPool.acquire(8, 8)) {
            PixelBuffer reused = lease.buffer();
            assertThat(reused, sameInstance(first));
            for (int y = 0; y < reused.height(); y++)
                for (int x = 0; x < reused.width(); x++)
                    assertThat(reused.getPixel(x, y), equalTo(0));
        }
    }

    @Test
    @DisplayName("double-close is a silent no-op")
    void doubleCloseIsNoOp() {
        PixelBufferPool.Lease lease = PixelBufferPool.acquire(16, 16);
        lease.close();
        assertDoesNotThrow(lease::close);
    }

    @Test
    @DisplayName("use-after-close throws IllegalStateException")
    void useAfterCloseThrows() {
        PixelBufferPool.Lease lease = PixelBufferPool.acquire(16, 16);
        lease.close();
        assertThrows(IllegalStateException.class, lease::buffer);
    }

    @Test
    @DisplayName("try-with-resources returns the buffer to the pool when the body throws")
    void leakOnExceptionStillReturnsToPool() {
        PixelBuffer first = null;
        try {
            try (PixelBufferPool.Lease lease = PixelBufferPool.acquire(24, 24)) {
                first = lease.buffer();
                throw new RuntimeException("simulated failure");
            }
        } catch (RuntimeException expected) {
            // intentional - we want to verify release happened despite the throw
        }
        try (PixelBufferPool.Lease lease = PixelBufferPool.acquire(24, 24)) {
            assertThat(lease.buffer(), sameInstance(first));
        }
    }

    @Test
    @DisplayName("concurrent acquire on two threads returns distinct buffers")
    void concurrentAcquireReturnsDistinctBuffers() throws Exception {
        CountDownLatch bothAcquired = new CountDownLatch(2);
        CountDownLatch canRelease = new CountDownLatch(1);
        AtomicReference<PixelBuffer> a = new AtomicReference<>();
        AtomicReference<PixelBuffer> b = new AtomicReference<>();
        AtomicInteger slot = new AtomicInteger();

        ExecutorService exec = Executors.newFixedThreadPool(2);
        try {
            Runnable worker = () -> {
                int mine = slot.getAndIncrement();
                try (PixelBufferPool.Lease lease = PixelBufferPool.acquire(48, 48)) {
                    (mine == 0 ? a : b).set(lease.buffer());
                    bothAcquired.countDown();
                    try {
                        canRelease.await(5, TimeUnit.SECONDS);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                    }
                }
            };
            exec.submit(worker);
            exec.submit(worker);

            assertThat(bothAcquired.await(5, TimeUnit.SECONDS), is(true));
            canRelease.countDown();
            exec.shutdown();
            assertThat(exec.awaitTermination(5, TimeUnit.SECONDS), is(true));
        } finally {
            if (!exec.isShutdown()) exec.shutdownNow();
        }

        assertThat(a.get(), allOf(
            is(not(sameInstance(b.get()))),
            is(not(equalTo(null)))
        ));
        assertThat(b.get(), is(not(equalTo(null))));
    }

    @Test
    @DisplayName("releasing more than the per-size cap drops the excess to GC")
    void poolCapOverflowFallsThrough() {
        // Acquire MAX + 1 buffers simultaneously so each is a fresh allocation.
        int max = 4;
        PixelBufferPool.Lease[] leases = new PixelBufferPool.Lease[max + 1];
        java.util.IdentityHashMap<PixelBuffer, Boolean> originals = new java.util.IdentityHashMap<>();
        for (int i = 0; i < leases.length; i++) {
            leases[i] = PixelBufferPool.acquire(96, 96);
            originals.put(leases[i].buffer(), Boolean.TRUE);
        }
        // Release them all - the last release should not fit in the pool.
        for (PixelBufferPool.Lease lease : leases) lease.close();

        // Drain the pool by holding every acquired buffer open at once. At most MAX
        // buffers must come back from the pool - the (MAX + 1)th acquire has to allocate.
        PixelBufferPool.Lease[] drained = new PixelBufferPool.Lease[max + 1];
        int recoveredFromPool = 0;
        try {
            for (int i = 0; i < drained.length; i++) {
                drained[i] = PixelBufferPool.acquire(96, 96);
                if (originals.containsKey(drained[i].buffer()))
                    recoveredFromPool++;
            }
        } finally {
            for (PixelBufferPool.Lease lease : drained)
                if (lease != null) lease.close();
        }
        assertThat(recoveredFromPool, equalTo(max));
    }

}
