package dev.simplified.image.codec.webp.lossy;

import org.jetbrains.annotations.NotNull;

/**
 * VP8 deblocking loop filter applied at macroblock and 4x4 sub-block boundaries.
 * <p>
 * Line-for-line port of libwebp's {@code src/dsp/dec.c} edge-filter primitives
 * plus {@code PrecomputeFilterStrengths} + {@code DoFilter} from {@code src/dec/frame_dec.c}.
 * Both filter modes are supported:
 * <ul>
 *   <li><b>Simple</b> (RFC 6386 paragraph 15.2) - {@code NeedsFilter_C} 4-tap check, {@code DoFilter2_C}
 *       2-pixel write. Luma only. Used when the frame header's {@code simple} bit is set.</li>
 *   <li><b>Normal</b> / complex (RFC 6386 paragraph 15.3) - {@code NeedsFilter2_C} 6+2-tap check plus
 *       {@code Hev} high-edge-variance branch: {@code DoFilter2} on HEV, {@code DoFilter6} (MB edge)
 *       or {@code DoFilter4} (inner edge) otherwise. Applied to luma + U + V.</li>
 * </ul>
 * Both modes optionally filter the three inner 4x4 sub-block edges inside the MB when
 * the macroblock was coded with {@code is_i4x4 = true} (B_PRED).
 *
 * @see <a href="https://datatracker.ietf.org/doc/html/rfc6386#section-15">RFC 6386 section 15</a>
 */
final class LoopFilter {

    // ── Reference-frame labels for per-MB loop-filter indexing (RFC 6386 section 15) ──
    /** Reference-frame index for intra-coded MBs. */
    static final int REF_INTRA = 0;
    /** Reference-frame index for MBs referencing {@code LAST}. */
    static final int REF_LAST = 1;
    /** Reference-frame index for MBs referencing {@code GOLDEN}. */
    static final int REF_GOLDEN = 2;
    /** Reference-frame index for MBs referencing {@code ALTREF}. */
    static final int REF_ALTREF = 3;

    // ── Mode-class labels for per-MB loop-filter indexing (RFC 6386 section 15) ──
    /** Intra MB with 16x16 (non-B_PRED) luma prediction. No {@code mode_lf_delta} applied. */
    static final int MODE_NON_BPRED_INTRA = 0;
    /** Intra MB with B_PRED luma. Picks up {@code mode_lf_delta[0]} + inner-edge filtering. */
    static final int MODE_BPRED = 1;
    /** Inter MB coded as ZEROMV. Picks up {@code mode_lf_delta[1]}. */
    static final int MODE_ZEROMV = 2;
    /** Inter MB coded as NEAREST / NEAR / NEW. Picks up {@code mode_lf_delta[2]}. */
    static final int MODE_OTHER_INTER = 3;
    /** Inter MB coded as SPLITMV. Picks up {@code mode_lf_delta[3]}. */
    static final int MODE_SPLITMV = 4;

    /** Per-MB mode-class index into the {@code mode_lf_delta[4]} array, or {@code -1} for no delta. */
    private static final int[] MODE_LF_DELTA_INDEX = {
        -1,   // MODE_NON_BPRED_INTRA (no mode delta)
         0,   // MODE_BPRED
         1,   // MODE_ZEROMV
         2,   // MODE_OTHER_INTER
         3,   // MODE_SPLITMV
    };

    /** Precomputed per-MB filter strength (matches libwebp's {@code VP8FInfo}). */
    static final class FInfo {
        /** Edge-activity threshold {@code 2*level + ilevel}; {@code 0} means no filtering. */
        final int fLimit;
        /** Inner threshold used by {@code NeedsFilter2}. */
        final int fIlevel;
        /** High-edge-variance threshold for the {@code DoFilter2 vs DoFilter6/4} branch. */
        final int hevThresh;
        /** Whether inner 4x4 sub-block edges should also be filtered (true for B_PRED MBs). */
        final boolean fInner;

        FInfo(int limit, int ilevel, int hevThresh, boolean inner) {
            this.fLimit = limit;
            this.fIlevel = ilevel;
            this.hevThresh = hevThresh;
            this.fInner = inner;
        }
    }

    private LoopFilter() { }

    /**
     * Applies the VP8 loop filter to a decoded frame in place. Mirrors libwebp's per-MB
     * {@code DoFilter} iterating across the full MB grid, with full RFC 6386 section 15
     * {@code ref_lf_delta} / {@code mode_lf_delta} support: each MB's effective filter
     * strength is {@code filter_level + ref_lf_delta[mb.ref] + mode_lf_delta[mb.mode]}
     * when {@code useLfDelta} is set, or just {@code filter_level} otherwise.
     *
     * @param planeY luma plane (row-major, stride {@code yStride})
     * @param planeU chroma U plane; may be {@code null} for simple-only filtering
     * @param planeV chroma V plane; may be {@code null} for simple-only filtering
     * @param yStride luma row stride in samples
     * @param uvStride chroma row stride in samples
     * @param mbCols number of 16x16 macroblock columns
     * @param mbRows number of 16x16 macroblock rows
     * @param simple if {@code true}, only the simple 2-tap luma filter is applied;
     *               otherwise the normal luma + chroma filter is applied
     * @param filterLevel base loop filter level {@code [0, 63]} from the frame header
     * @param sharpness sharpness parameter {@code [0, 7]} from the frame header
     * @param mbRefFrame per-MB reference-frame index (row-major), one of {@link #REF_INTRA}
     *                  / {@link #REF_LAST} / {@link #REF_GOLDEN} / {@link #REF_ALTREF}
     * @param mbModeLfIdx per-MB mode-class (row-major), one of the {@code MODE_*} constants
     * @param useLfDelta whether to apply {@code refLfDelta} + {@code modeLfDelta}
     * @param refLfDelta 4-entry delta vector indexed by reference frame
     * @param modeLfDelta 4-entry delta vector indexed by {@link #MODE_LF_DELTA_INDEX}
     */
    static void filterFrame(
        short @NotNull [] planeY, short[] planeU, short[] planeV,
        int yStride, int uvStride, int mbCols, int mbRows,
        boolean simple, int filterLevel, int sharpness,
        int @NotNull [] mbRefFrame, int @NotNull [] mbModeLfIdx,
        boolean useLfDelta, int @NotNull [] refLfDelta, int @NotNull [] modeLfDelta
    ) {
        if (filterLevel == 0) return;

        FInfo[][] strengths = computeStrengthsTable(
            filterLevel, sharpness, useLfDelta, refLfDelta, modeLfDelta);
        for (int mbY = 0; mbY < mbRows; mbY++) {
            for (int mbX = 0; mbX < mbCols; mbX++) {
                int mbIdx = mbY * mbCols + mbX;
                FInfo info = strengths[mbRefFrame[mbIdx]][mbModeLfIdx[mbIdx]];
                if (info.fLimit == 0) continue;
                doFilter(planeY, planeU, planeV, yStride, uvStride, mbX, mbY, info, simple);
            }
        }
    }

    /**
     * Precomputes a {@code [ref][mode]} table of {@link FInfo} for all combinations of
     * reference frame (4) and mode class (5). Line-for-line port of libwebp / libvpx
     * {@code PrecomputeFilterStrengths} at the single-segment configuration, extended
     * with the full RFC 6386 section 15 {@code ref_lf_delta} / {@code mode_lf_delta}
     * pipeline. Entries with {@code fLimit == 0} trigger the no-filter fast path in
     * {@link #filterFrame}.
     */
    private static FInfo @NotNull [] @NotNull [] computeStrengthsTable(
        int filterLevel, int sharpness,
        boolean useLfDelta, int @NotNull [] refLfDelta, int @NotNull [] modeLfDelta
    ) {
        FInfo[][] out = new FInfo[4][5];
        for (int ref = 0; ref < 4; ref++) {
            for (int mode = 0; mode < 5; mode++) {
                int level = filterLevel;
                if (useLfDelta) {
                    level += refLfDelta[ref];
                    int modeDeltaIdx = MODE_LF_DELTA_INDEX[mode];
                    if (modeDeltaIdx >= 0) level += modeLfDelta[modeDeltaIdx];
                }
                level = Math.clamp(level, 0, 63);
                boolean inner = (mode == MODE_BPRED);
                out[ref][mode] = buildFInfo(level, sharpness, inner);
            }
        }
        return out;
    }

    /**
     * Builds a single {@link FInfo} from a resolved {@code level}, encoding the sharpness
     * adjustment of {@code ilevel} (RFC 6386 section 15.2) and the {@code hev_threshold}
     * staircase at {@code level >= 40 / 15 / 0}.
     */
    private static @NotNull FInfo buildFInfo(int level, int sharpness, boolean inner) {
        if (level <= 0) return new FInfo(0, 0, 0, inner);
        int ilevel = level;
        if (sharpness > 0) {
            ilevel >>= (sharpness > 4 ? 2 : 1);
            if (ilevel > 9 - sharpness) ilevel = 9 - sharpness;
        }
        if (ilevel < 1) ilevel = 1;
        int limit = 2 * level + ilevel;
        int hev = (level >= 40) ? 2 : (level >= 15) ? 1 : 0;
        return new FInfo(limit, ilevel, hev, inner);
    }

    /** Per-MB filter dispatch. Mirrors libwebp's {@code DoFilter} in {@code src/dec/frame_dec.c}. */
    private static void doFilter(
        short[] planeY, short[] planeU, short[] planeV,
        int yStride, int uvStride, int mbX, int mbY,
        @NotNull FInfo info, boolean simple
    ) {
        int yPtr = (mbY * 16) * yStride + mbX * 16;
        int limit = info.fLimit;
        int ilevel = info.fIlevel;
        int hevThresh = info.hevThresh;

        if (simple) {
            if (mbX > 0) simpleHFilter16(planeY, yPtr, yStride, limit + 4);
            if (info.fInner) simpleHFilter16i(planeY, yPtr, yStride, limit);
            if (mbY > 0) simpleVFilter16(planeY, yPtr, yStride, limit + 4);
            if (info.fInner) simpleVFilter16i(planeY, yPtr, yStride, limit);
            return;
        }

        int uPtr = (mbY * 8) * uvStride + mbX * 8;
        int vPtr = uPtr;

        if (mbX > 0) {
            hFilter16(planeY, yPtr, yStride, limit + 4, ilevel, hevThresh);
            hFilter8(planeU, planeV, uPtr, vPtr, uvStride, limit + 4, ilevel, hevThresh);
        }
        if (info.fInner) {
            hFilter16i(planeY, yPtr, yStride, limit, ilevel, hevThresh);
            hFilter8i(planeU, planeV, uPtr, vPtr, uvStride, limit, ilevel, hevThresh);
        }
        if (mbY > 0) {
            vFilter16(planeY, yPtr, yStride, limit + 4, ilevel, hevThresh);
            vFilter8(planeU, planeV, uPtr, vPtr, uvStride, limit + 4, ilevel, hevThresh);
        }
        if (info.fInner) {
            vFilter16i(planeY, yPtr, yStride, limit, ilevel, hevThresh);
            vFilter8i(planeU, planeV, uPtr, vPtr, uvStride, limit, ilevel, hevThresh);
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Simple filter (paragraph 15.2) - NeedsFilter_C + DoFilter2_C
    // ──────────────────────────────────────────────────────────────────────

    /** Simple filter on the horizontal edge between MB row {@code mbY-1} and {@code mbY}. */
    private static void simpleVFilter16(short[] p, int ptr, int stride, int thresh) {
        int thresh2 = 2 * thresh + 1;
        for (int i = 0; i < 16; i++)
            if (needsFilter(p, ptr + i, stride, thresh2))
                doFilter2(p, ptr + i, stride);
    }

    /** Simple filter on the vertical edge between MB col {@code mbX-1} and {@code mbX}. */
    private static void simpleHFilter16(short[] p, int ptr, int stride, int thresh) {
        int thresh2 = 2 * thresh + 1;
        for (int i = 0; i < 16; i++)
            if (needsFilter(p, ptr + i * stride, 1, thresh2))
                doFilter2(p, ptr + i * stride, 1);
    }

    /** Simple filter on the three inner horizontal 4x4 edges at offsets 4, 8, 12 rows. */
    private static void simpleVFilter16i(short[] p, int ptr, int stride, int thresh) {
        for (int k = 3; k > 0; k--) {
            ptr += 4 * stride;
            simpleVFilter16(p, ptr, stride, thresh);
        }
    }

    /** Simple filter on the three inner vertical 4x4 edges at offsets 4, 8, 12 columns. */
    private static void simpleHFilter16i(short[] p, int ptr, int stride, int thresh) {
        for (int k = 3; k > 0; k--) {
            ptr += 4;
            simpleHFilter16(p, ptr, stride, thresh);
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Normal filter (paragraph 15.3) - NeedsFilter2_C + DoFilter2/4/6_C
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Complex 16-sample filter along a 16-pixel edge. {@code hstride} is the step between
     * successive filtered samples within a single 4- or 6-tap filter (perpendicular to the
     * edge); {@code vstride} walks along the edge. Mirrors libwebp's {@code FilterLoop26_C}
     * which branches between {@code DoFilter2_C} (HEV path) and {@code DoFilter6_C}.
     */
    private static void filterLoop26(
        short[] plane, int ptr, int hstride, int vstride, int size,
        int thresh, int ithresh, int hevThresh
    ) {
        int thresh2 = 2 * thresh + 1;
        while (size-- > 0) {
            if (needsFilter2(plane, ptr, hstride, thresh2, ithresh)) {
                if (hev(plane, ptr, hstride, hevThresh))
                    doFilter2(plane, ptr, hstride);
                else
                    doFilter6(plane, ptr, hstride);
            }
            ptr += vstride;
        }
    }

    /** Same as {@link #filterLoop26} but uses {@code DoFilter4_C} (inner edges). */
    private static void filterLoop24(
        short[] plane, int ptr, int hstride, int vstride, int size,
        int thresh, int ithresh, int hevThresh
    ) {
        int thresh2 = 2 * thresh + 1;
        while (size-- > 0) {
            if (needsFilter2(plane, ptr, hstride, thresh2, ithresh)) {
                if (hev(plane, ptr, hstride, hevThresh))
                    doFilter2(plane, ptr, hstride);
                else
                    doFilter4(plane, ptr, hstride);
            }
            ptr += vstride;
        }
    }

    private static void vFilter16(short[] p, int ptr, int stride, int thresh, int ithresh, int hevThresh) {
        filterLoop26(p, ptr, stride, 1, 16, thresh, ithresh, hevThresh);
    }

    private static void hFilter16(short[] p, int ptr, int stride, int thresh, int ithresh, int hevThresh) {
        filterLoop26(p, ptr, 1, stride, 16, thresh, ithresh, hevThresh);
    }

    private static void vFilter16i(short[] p, int ptr, int stride, int thresh, int ithresh, int hevThresh) {
        for (int k = 3; k > 0; k--) {
            ptr += 4 * stride;
            filterLoop24(p, ptr, stride, 1, 16, thresh, ithresh, hevThresh);
        }
    }

    private static void hFilter16i(short[] p, int ptr, int stride, int thresh, int ithresh, int hevThresh) {
        for (int k = 3; k > 0; k--) {
            ptr += 4;
            filterLoop24(p, ptr, 1, stride, 16, thresh, ithresh, hevThresh);
        }
    }

    private static void vFilter8(short[] u, short[] v, int uPtr, int vPtr, int stride,
                                 int thresh, int ithresh, int hevThresh) {
        filterLoop26(u, uPtr, stride, 1, 8, thresh, ithresh, hevThresh);
        filterLoop26(v, vPtr, stride, 1, 8, thresh, ithresh, hevThresh);
    }

    private static void hFilter8(short[] u, short[] v, int uPtr, int vPtr, int stride,
                                 int thresh, int ithresh, int hevThresh) {
        filterLoop26(u, uPtr, 1, stride, 8, thresh, ithresh, hevThresh);
        filterLoop26(v, vPtr, 1, stride, 8, thresh, ithresh, hevThresh);
    }

    private static void vFilter8i(short[] u, short[] v, int uPtr, int vPtr, int stride,
                                  int thresh, int ithresh, int hevThresh) {
        filterLoop24(u, uPtr + 4 * stride, stride, 1, 8, thresh, ithresh, hevThresh);
        filterLoop24(v, vPtr + 4 * stride, stride, 1, 8, thresh, ithresh, hevThresh);
    }

    private static void hFilter8i(short[] u, short[] v, int uPtr, int vPtr, int stride,
                                  int thresh, int ithresh, int hevThresh) {
        filterLoop24(u, uPtr + 4, 1, stride, 8, thresh, ithresh, hevThresh);
        filterLoop24(v, vPtr + 4, 1, stride, 8, thresh, ithresh, hevThresh);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Filter kernels (DoFilter2/4/6) + predicates (NeedsFilter/2, Hev)
    // ──────────────────────────────────────────────────────────────────────

    /** 4 pixels in, 2 pixels out. Writes {@code p0} and {@code q0}. */
    private static void doFilter2(short[] p, int ptr, int step) {
        int p1 = p[ptr - 2 * step], p0 = p[ptr - step], q0 = p[ptr], q1 = p[ptr + step];
        int a = 3 * (q0 - p0) + sclip1(p1 - q1);
        int a1 = sclip2((a + 4) >> 3);
        int a2 = sclip2((a + 3) >> 3);
        p[ptr - step] = (short) clip1(p0 + a2);
        p[ptr] = (short) clip1(q0 - a1);
    }

    /** 4 pixels in, 4 pixels out. Used on inner sub-block edges (normal filter). */
    private static void doFilter4(short[] p, int ptr, int step) {
        int p1 = p[ptr - 2 * step], p0 = p[ptr - step], q0 = p[ptr], q1 = p[ptr + step];
        int a = 3 * (q0 - p0);
        int a1 = sclip2((a + 4) >> 3);
        int a2 = sclip2((a + 3) >> 3);
        int a3 = (a1 + 1) >> 1;
        p[ptr - 2 * step] = (short) clip1(p1 + a3);
        p[ptr - step] = (short) clip1(p0 + a2);
        p[ptr] = (short) clip1(q0 - a1);
        p[ptr + step] = (short) clip1(q1 - a3);
    }

    /** 6 pixels in, 6 pixels out. Used on MB-edge non-HEV transitions. */
    private static void doFilter6(short[] p, int ptr, int step) {
        int p2 = p[ptr - 3 * step], p1 = p[ptr - 2 * step], p0 = p[ptr - step];
        int q0 = p[ptr], q1 = p[ptr + step], q2 = p[ptr + 2 * step];
        int a = sclip1(3 * (q0 - p0) + sclip1(p1 - q1));
        int a1 = (27 * a + 63) >> 7;
        int a2 = (18 * a + 63) >> 7;
        int a3 = (9 * a + 63) >> 7;
        p[ptr - 3 * step] = (short) clip1(p2 + a3);
        p[ptr - 2 * step] = (short) clip1(p1 + a2);
        p[ptr - step] = (short) clip1(p0 + a1);
        p[ptr] = (short) clip1(q0 - a1);
        p[ptr + step] = (short) clip1(q1 - a2);
        p[ptr + 2 * step] = (short) clip1(q2 - a3);
    }

    /** High-edge-variance detector: either {@code |p1-p0|} or {@code |q1-q0|} exceeds {@code thresh}. */
    private static boolean hev(short[] p, int ptr, int step, int thresh) {
        int p1 = p[ptr - 2 * step], p0 = p[ptr - step], q0 = p[ptr], q1 = p[ptr + step];
        return Math.abs(p1 - p0) > thresh || Math.abs(q1 - q0) > thresh;
    }

    /** Simple filter edge-activity predicate: {@code 4|p0-q0| + |p1-q1| <= thresh}. */
    private static boolean needsFilter(short[] p, int ptr, int step, int t) {
        int p1 = p[ptr - 2 * step], p0 = p[ptr - step], q0 = p[ptr], q1 = p[ptr + step];
        return (4 * Math.abs(p0 - q0) + Math.abs(p1 - q1)) <= t;
    }

    /** 6-tap needs-filter predicate: simple check plus six inner-level smoothness tests. */
    private static boolean needsFilter2(short[] p, int ptr, int step, int t, int it) {
        int p3 = p[ptr - 4 * step], p2 = p[ptr - 3 * step], p1 = p[ptr - 2 * step];
        int p0 = p[ptr - step], q0 = p[ptr];
        int q1 = p[ptr + step], q2 = p[ptr + 2 * step], q3 = p[ptr + 3 * step];
        if ((4 * Math.abs(p0 - q0) + Math.abs(p1 - q1)) > t) return false;
        return Math.abs(p3 - p2) <= it && Math.abs(p2 - p1) <= it
            && Math.abs(p1 - p0) <= it && Math.abs(q3 - q2) <= it
            && Math.abs(q2 - q1) <= it && Math.abs(q1 - q0) <= it;
    }

    /** Clamps to {@code [-128, 127]} (libwebp's {@code VP8ksclip1}). */
    private static int sclip1(int v) {
        return Math.clamp(v, -128, 127);
    }

    /** Clamps to {@code [-16, 15]} (libwebp's {@code VP8ksclip2}). */
    private static int sclip2(int v) {
        return Math.clamp(v, -16, 15);
    }

    /** Clamps to {@code [0, 255]} (libwebp's {@code VP8kclip1}). */
    private static int clip1(int v) {
        return Math.clamp(v, 0, 255);
    }

}
