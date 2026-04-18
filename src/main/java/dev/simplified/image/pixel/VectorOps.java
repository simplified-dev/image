package dev.simplified.image.pixel;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorSpecies;

import static jdk.incubator.vector.VectorOperators.AND;
import static jdk.incubator.vector.VectorOperators.DIV;
import static jdk.incubator.vector.VectorOperators.F2I;
import static jdk.incubator.vector.VectorOperators.I2F;
import static jdk.incubator.vector.VectorOperators.LSHL;
import static jdk.incubator.vector.VectorOperators.LSHR;
import static jdk.incubator.vector.VectorOperators.OR;

/**
 * JDK Vector API implementations of per-pixel operations.
 * <p>
 * Loaded lazily by {@link PixelVector} when the {@code jdk.incubator.vector} module is resolvable
 * at runtime. Each method processes the SIMD-aligned prefix of its input range and returns the
 * number of elements consumed; the caller handles the scalar tail.
 */
final class VectorOps {

    private static final VectorSpecies<Integer> I = IntVector.SPECIES_PREFERRED;

    private VectorOps() {}

    static int grayscale(int[] p, int off, int len) {
        int bound = I.loopBound(len);
        for (int i = 0; i < bound; i += I.length()) {
            int at = off + i;
            IntVector v = IntVector.fromArray(I, p, at);
            IntVector a = v.lanewise(LSHR, 24).lanewise(AND, 0xFF);
            IntVector r = v.lanewise(LSHR, 16).lanewise(AND, 0xFF);
            IntVector g = v.lanewise(LSHR, 8).lanewise(AND, 0xFF);
            IntVector b = v.lanewise(AND, 0xFF);
            FloatVector luma = ((FloatVector) r.convert(I2F, 0)).mul(0.299f)
                .add(((FloatVector) g.convert(I2F, 0)).mul(0.587f))
                .add(((FloatVector) b.convert(I2F, 0)).mul(0.114f));
            IntVector y = ((IntVector) luma.convert(F2I, 0)).max(0).min(255);
            a.lanewise(LSHL, 24)
                .lanewise(OR, y.lanewise(LSHL, 16))
                .lanewise(OR, y.lanewise(LSHL, 8))
                .lanewise(OR, y)
                .intoArray(p, at);
        }
        return bound;
    }

    static int multiplyAlpha(int[] p, int off, int len, float factor) {
        int bound = I.loopBound(len);
        for (int i = 0; i < bound; i += I.length()) {
            int at = off + i;
            IntVector v = IntVector.fromArray(I, p, at);
            IntVector a = v.lanewise(LSHR, 24).lanewise(AND, 0xFF);
            FloatVector scaled = ((FloatVector) a.convert(I2F, 0)).mul(factor).add(0.5f);
            IntVector newA = (IntVector) scaled.convert(F2I, 0);
            newA.lanewise(LSHL, 24)
                .lanewise(OR, v.lanewise(AND, 0x00FFFFFF))
                .intoArray(p, at);
        }
        return bound;
    }

    static int premultiply(int[] p, int off, int len) {
        int bound = I.loopBound(len);
        for (int i = 0; i < bound; i += I.length()) {
            int at = off + i;
            IntVector v = IntVector.fromArray(I, p, at);
            IntVector a = v.lanewise(LSHR, 24).lanewise(AND, 0xFF);
            IntVector r = v.lanewise(LSHR, 16).lanewise(AND, 0xFF).mul(a).lanewise(DIV, 255);
            IntVector g = v.lanewise(LSHR, 8).lanewise(AND, 0xFF).mul(a).lanewise(DIV, 255);
            IntVector b = v.lanewise(AND, 0xFF).mul(a).lanewise(DIV, 255);
            a.lanewise(LSHL, 24)
                .lanewise(OR, r.lanewise(LSHL, 16))
                .lanewise(OR, g.lanewise(LSHL, 8))
                .lanewise(OR, b)
                .intoArray(p, at);
        }
        return bound;
    }

    static int tint(int[] src, int[] dst, int off, int len, int tr, int tg, int tb) {
        int bound = I.loopBound(len);
        for (int i = 0; i < bound; i += I.length()) {
            int at = off + i;
            IntVector v = IntVector.fromArray(I, src, at);
            IntVector a = v.lanewise(LSHR, 24).lanewise(AND, 0xFF);
            IntVector r = v.lanewise(LSHR, 16).lanewise(AND, 0xFF).mul(tr).lanewise(DIV, 255);
            IntVector g = v.lanewise(LSHR, 8).lanewise(AND, 0xFF).mul(tg).lanewise(DIV, 255);
            IntVector b = v.lanewise(AND, 0xFF).mul(tb).lanewise(DIV, 255);
            IntVector out = a.lanewise(LSHL, 24)
                .lanewise(OR, r.lanewise(LSHL, 16))
                .lanewise(OR, g.lanewise(LSHL, 8))
                .lanewise(OR, b);
            VectorMask<Integer> transparent = a.eq(0);
            out.blend(v, transparent).intoArray(dst, at);
        }
        return bound;
    }

    static int lerp(int[] a, int aOff, int[] b, int bOff, int[] out, int outOff,
                    int len, float blend, float inverse) {
        int bound = I.loopBound(len);
        for (int i = 0; i < bound; i += I.length()) {
            IntVector sv = IntVector.fromArray(I, a, aOff + i);
            IntVector dv = IntVector.fromArray(I, b, bOff + i);
            IntVector ra = blendChannel(sv, dv, 24, blend, inverse);
            IntVector rr = blendChannel(sv, dv, 16, blend, inverse);
            IntVector rg = blendChannel(sv, dv, 8, blend, inverse);
            IntVector rb = blendChannel(sv, dv, 0, blend, inverse);
            ra.lanewise(LSHL, 24)
                .lanewise(OR, rr.lanewise(LSHL, 16))
                .lanewise(OR, rg.lanewise(LSHL, 8))
                .lanewise(OR, rb)
                .intoArray(out, outOff + i);
        }
        return bound;
    }

    private static IntVector blendChannel(IntVector s, IntVector d, int shift, float blend, float inverse) {
        IntVector sc = s.lanewise(LSHR, shift).lanewise(AND, 0xFF);
        IntVector dc = d.lanewise(LSHR, shift).lanewise(AND, 0xFF);
        FloatVector f = ((FloatVector) sc.convert(I2F, 0)).mul(inverse)
            .add(((FloatVector) dc.convert(I2F, 0)).mul(blend));
        return (IntVector) f.max(0f).min(255f).convert(F2I, 0);
    }
}
