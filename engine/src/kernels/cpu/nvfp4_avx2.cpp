// src/kernels/cpu/nvfp4_avx2.cpp - see include/strata/kernels/cpu/nvfp4_avx2.hpp.
//
// Per 64-value block, once: its E2M1 values in element order as two 32-byte registers (ggml's reordering: sub-block
// s keeps value j in the low nibble of qs[8 s + j] and value j + 8 in the high one), their magnitudes, and the four
// sub-block scales. Per token, as ggml_vec_dot_nvfp4_q8_0's AVX2 path: vpsignb moves the sign onto the activation,
// vpmaddubsw + vpmaddwd give 8 int32 lanes (four per 16-value sub-block), one FMA per 32 values applies
// ue4m3(d[s]) * d_q8, and hsum_float_8's order sums the lanes.
//
// Needs AVX2, FMA and F16C - what cpu_avx2_ok() checks before anything here is called.
#include "strata/kernels/cpu/nvfp4_avx2.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

// A compile-time table, not a constructor: this file is compiled for a wide ISA, and a namespace-scope object with a
// constructor runs at program start on EVERY CPU - the compiler vectorises the loop with AVX/FMA, and a CPU without them
// (Westmere, Sandy Bridge, ...) died with an illegal instruction before main (#2324).  Powers of two: exact.
constexpr float pow2i(int e) {
    float r = 1.0f;
    for (; e > 0; --e) r *= 2.0f;
    for (; e < 0; ++e) r *= 0.5f;
    return r;
}
struct Ue4m3Table {   // ggml_ue4m3_to_fp32 (ggml-impl.h): halved for the doubled kvalues_fp4, 0x7F read as 0
    float v[256];
    constexpr Ue4m3Table() : v() {
        for (int x = 0; x < 256; ++x) {
            if (x == 0 || x == 0x7F) { v[x] = 0.0f; continue; }
            const int e = (x >> 3) & 0xF, m = x & 0x7;
            const float raw = e == 0 ? (float) m * pow2i(-9) : (1.0f + (float) m / 8.0f) * pow2i(e - 7);
            v[x] = raw * 0.5f;
        }
    }
};
constexpr Ue4m3Table kUe4m3;

// software prefetch distance in bytes (STRATA_NVFP4_PREFETCH, 0 = off), as the AVX-512 rows
// (read on first use, behind the CPU check: a static initialiser here would run on every CPU, see above)
inline int prefetch_distance() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_NVFP4_PREFETCH");
        return e ? std::atoi(e) : 2048;
    }();
    return v;
}

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }

// Decode all 4 sub-blocks (32 bytes of qs -> 64 signed doubled E2M1 values and magnitudes) in 256-bit AVX2 lanes.
// Bit-identical to calling the 128-bit unpack + kAbsFp4 table twice, using vpabsb and half the shuffle/cross-lane ops.
inline void values64_both(const uint8_t* qs, const __m256i kv,
                          __m256i& v01, __m256i& a01, __m256i& v23, __m256i& a23) {
    const __m256i q = _mm256_loadu_si256((const __m256i*) qs);
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m256i ql = _mm256_and_si256(q, m4);
    const __m256i qh = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
    const __m256i lo = _mm256_shuffle_epi8(kv, ql);
    const __m256i hi = _mm256_shuffle_epi8(kv, qh);
    const __m256i sb02 = _mm256_unpacklo_epi64(lo, hi);
    const __m256i sb13 = _mm256_unpackhi_epi64(lo, hi);
    v01 = _mm256_permute2x128_si256(sb02, sb13, 0x20);
    v23 = _mm256_permute2x128_si256(sb02, sb13, 0x31);
    a01 = _mm256_abs_epi8(v01);
    a23 = _mm256_abs_epi8(v23);
}

inline float hsum8(const __m256 x) {   // ggml's hsum_float_8
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

constexpr int kMaxBlocks = 64;   // rows of up to 4096 values (n_embd 2560)
// tokens per pass: the pool's MAXT, so a window is one pass. Spilled accumulators cost less than another pass over
// the weights: one expert's gate+up, 8 tokens, 0.415 ms in passes of 4 + 4, 0.352 in one of 8 (5 6 7 alike)
constexpr int kMaxW = 8;

struct Acts {   // per token and block, the two Q8_0 blocks' scales; built once per call, reused by every row
    int nt = 0, nb = 0;
    const block_q8_0* y[kMaxW] = {};
    float dy[kMaxW * kMaxBlocks * 2];
    Acts(const void* const* act, int nt_, int nb_) : nt(nt_), nb(nb_) {
        for (int t = 0; t < nt; ++t) {
            y[t] = (const block_q8_0*) act[t];
            for (int ib = 0; ib < 2 * nb; ++ib) dy[(size_t) t * 2 * nb + ib] = h2f(y[t][ib].d);
        }
    }
};

template <int NT>
inline void row_dot(const uint8_t* row, const Acts& a, float* res) {
    const __m256i kv = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) kvalues_fp4));
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    const block_nvfp4* x = (const block_nvfp4*) row;
    const int prefetch_ahead = prefetch_distance();
    for (int ib = 0; ib < a.nb; ++ib) {
        if (prefetch_ahead > 0 && (ib & 1) == 0) _mm_prefetch((const char*) (x + ib) + prefetch_ahead, _MM_HINT_T0);
        __m256i v01, a01, v23, a23;
        values64_both(x[ib].qs, kv, v01, a01, v23, a23);
        const __m256 u01 = _mm256_set_m128(_mm_set1_ps(kUe4m3.v[x[ib].d[1]]), _mm_set1_ps(kUe4m3.v[x[ib].d[0]]));
        const __m256 u23 = _mm256_set_m128(_mm_set1_ps(kUe4m3.v[x[ib].d[3]]), _mm_set1_ps(kUe4m3.v[x[ib].d[2]]));
        for (int t = 0; t < NT; ++t) {
            const block_q8_0* yb = a.y[t] + 2 * ib;
            const __m256i q01 = _mm256_loadu_si256((const __m256i*) yb[0].qs);
            const __m256i q23 = _mm256_loadu_si256((const __m256i*) yb[1].qs);
            const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(a01, _mm256_sign_epi8(q01, v01)), ones);
            const __m256i p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(a23, _mm256_sign_epi8(q23, v23)), ones);
            const float* dy = a.dy + (size_t) t * 2 * a.nb + 2 * ib;
            const __m256 s01 = _mm256_mul_ps(u01, _mm256_broadcast_ss(dy));
            const __m256 s23 = _mm256_mul_ps(u23, _mm256_broadcast_ss(dy + 1));
            acc[t] = _mm256_fmadd_ps(s01, _mm256_cvtepi32_ps(p1), acc[t]);
            acc[t] = _mm256_fmadd_ps(s23, _mm256_cvtepi32_ps(p2), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum8(acc[t]);
}

// Fused 2-row dot (gate+up or adjacent down rows): loads activations ONCE per block & doubles Zen3 FMA ILP, bit-exact.
template <int NT>
inline void row_dot_pair(const uint8_t* row0, const uint8_t* row1, const Acts& a, float* res0, float* res1) {
    const __m256i kv = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) kvalues_fp4));
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc0[NT], acc1[NT];
    for (int t = 0; t < NT; ++t) { acc0[t] = _mm256_setzero_ps(); acc1[t] = _mm256_setzero_ps(); }
    const block_nvfp4* x0 = (const block_nvfp4*) row0;
    const block_nvfp4* x1 = (const block_nvfp4*) row1;
    const int prefetch_ahead = prefetch_distance();
    if constexpr (NT == 1) {
        // Sequential row0 -> row1 per block keeps peak live YMMs <= 13 (zero stack spills) while sharing q01/q23 loads!
        for (int ib = 0; ib < a.nb; ++ib) {
            if (prefetch_ahead > 0 && (ib & 1) == 0) {
                _mm_prefetch((const char*) (x0 + ib) + prefetch_ahead, _MM_HINT_T0);
                _mm_prefetch((const char*) (x1 + ib) + prefetch_ahead, _MM_HINT_T0);
            }
            const block_q8_0* yb = a.y[0] + 2 * ib;
            const __m256i q01 = _mm256_loadu_si256((const __m256i*) yb[0].qs);
            const __m256i q23 = _mm256_loadu_si256((const __m256i*) yb[1].qs);
            const float* dy = a.dy + 2 * ib;
            const __m256 dy0 = _mm256_broadcast_ss(dy);
            const __m256 dy1 = _mm256_broadcast_ss(dy + 1);
            {
                __m256i v01, a01, v23, a23;
                values64_both(x0[ib].qs, kv, v01, a01, v23, a23);
                const __m256 u01 = _mm256_set_m128(_mm_set1_ps(kUe4m3.v[x0[ib].d[1]]), _mm_set1_ps(kUe4m3.v[x0[ib].d[0]]));
                const __m256 u23 = _mm256_set_m128(_mm_set1_ps(kUe4m3.v[x0[ib].d[3]]), _mm_set1_ps(kUe4m3.v[x0[ib].d[2]]));
                const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(a01, _mm256_sign_epi8(q01, v01)), ones);
                const __m256i p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(a23, _mm256_sign_epi8(q23, v23)), ones);
                acc0[0] = _mm256_fmadd_ps(_mm256_mul_ps(u01, dy0), _mm256_cvtepi32_ps(p1), acc0[0]);
                acc0[0] = _mm256_fmadd_ps(_mm256_mul_ps(u23, dy1), _mm256_cvtepi32_ps(p2), acc0[0]);
            }
            {
                __m256i v01, a01, v23, a23;
                values64_both(x1[ib].qs, kv, v01, a01, v23, a23);
                const __m256 u01 = _mm256_set_m128(_mm_set1_ps(kUe4m3.v[x1[ib].d[1]]), _mm_set1_ps(kUe4m3.v[x1[ib].d[0]]));
                const __m256 u23 = _mm256_set_m128(_mm_set1_ps(kUe4m3.v[x1[ib].d[3]]), _mm_set1_ps(kUe4m3.v[x1[ib].d[2]]));
                const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(a01, _mm256_sign_epi8(q01, v01)), ones);
                const __m256i p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(a23, _mm256_sign_epi8(q23, v23)), ones);
                acc1[0] = _mm256_fmadd_ps(_mm256_mul_ps(u01, dy0), _mm256_cvtepi32_ps(p1), acc1[0]);
                acc1[0] = _mm256_fmadd_ps(_mm256_mul_ps(u23, dy1), _mm256_cvtepi32_ps(p2), acc1[0]);
            }
        }
        res0[0] = hsum8(acc0[0]);
        res1[0] = hsum8(acc1[0]);
        return;
    }
    for (int ib = 0; ib < a.nb; ++ib) {
        if (prefetch_ahead > 0 && (ib & 1) == 0) {
            _mm_prefetch((const char*) (x0 + ib) + prefetch_ahead, _MM_HINT_T0);
            _mm_prefetch((const char*) (x1 + ib) + prefetch_ahead, _MM_HINT_T0);
        }
        __m256i v0_01, a0_01, v0_23, a0_23;
        values64_both(x0[ib].qs, kv, v0_01, a0_01, v0_23, a0_23);
        const __m256 u0_01 = _mm256_set_m128(_mm_set1_ps(kUe4m3.v[x0[ib].d[1]]), _mm_set1_ps(kUe4m3.v[x0[ib].d[0]]));
        const __m256 u0_23 = _mm256_set_m128(_mm_set1_ps(kUe4m3.v[x0[ib].d[3]]), _mm_set1_ps(kUe4m3.v[x0[ib].d[2]]));

        __m256i v1_01, a1_01, v1_23, a1_23;
        values64_both(x1[ib].qs, kv, v1_01, a1_01, v1_23, a1_23);
        const __m256 u1_01 = _mm256_set_m128(_mm_set1_ps(kUe4m3.v[x1[ib].d[1]]), _mm_set1_ps(kUe4m3.v[x1[ib].d[0]]));
        const __m256 u1_23 = _mm256_set_m128(_mm_set1_ps(kUe4m3.v[x1[ib].d[3]]), _mm_set1_ps(kUe4m3.v[x1[ib].d[2]]));

        for (int t = 0; t < NT; ++t) {
            const block_q8_0* yb = a.y[t] + 2 * ib;
            const __m256i q01 = _mm256_loadu_si256((const __m256i*) yb[0].qs);
            const __m256i q23 = _mm256_loadu_si256((const __m256i*) yb[1].qs);
            const float* dy = a.dy + (size_t) t * 2 * a.nb + 2 * ib;
            const __m256 dy0 = _mm256_broadcast_ss(dy);
            const __m256 dy1 = _mm256_broadcast_ss(dy + 1);

            const __m256i p0_1 = _mm256_madd_epi16(_mm256_maddubs_epi16(a0_01, _mm256_sign_epi8(q01, v0_01)), ones);
            const __m256i p0_2 = _mm256_madd_epi16(_mm256_maddubs_epi16(a0_23, _mm256_sign_epi8(q23, v0_23)), ones);
            acc0[t] = _mm256_fmadd_ps(_mm256_mul_ps(u0_01, dy0), _mm256_cvtepi32_ps(p0_1), acc0[t]);
            acc0[t] = _mm256_fmadd_ps(_mm256_mul_ps(u0_23, dy1), _mm256_cvtepi32_ps(p0_2), acc0[t]);

            const __m256i p1_1 = _mm256_madd_epi16(_mm256_maddubs_epi16(a1_01, _mm256_sign_epi8(q01, v1_01)), ones);
            const __m256i p1_2 = _mm256_madd_epi16(_mm256_maddubs_epi16(a1_23, _mm256_sign_epi8(q23, v1_23)), ones);
            acc1[t] = _mm256_fmadd_ps(_mm256_mul_ps(u1_01, dy0), _mm256_cvtepi32_ps(p1_1), acc1[t]);
            acc1[t] = _mm256_fmadd_ps(_mm256_mul_ps(u1_23, dy1), _mm256_cvtepi32_ps(p1_2), acc1[t]);
        }
    }
    for (int t = 0; t < NT; ++t) { res0[t] = hsum8(acc0[t]); res1[t] = hsum8(acc1[t]); }
}

template <int NT, bool SITU = false>
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, const Acts& a, float* const* ff, int r0, int r1,
             float sg, float su, float beta = 4.f, float lb = 25.f) {
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        if constexpr (NT <= 3) {
            row_dot_pair<NT>(blob + (size_t) r * gu_row, blob + up_off + (size_t) r * gu_row, a, g, u);
        } else {
            row_dot<NT>(blob + (size_t) r * gu_row, a, g);
            row_dot<NT>(blob + up_off + (size_t) r * gu_row, a, u);
        }
        for (int t = 0; t < NT; ++t) {   // native_gu_rows' ggml path, operation for operation
            const float gs = g[t] * sg, us = u[t] * su;
            if constexpr (SITU) {
                const float tg = beta * std::tanh((beta == 4.0f) ? (gs * 0.25f) : (gs / beta));
                const float tu = lb > 0.f ? lb * std::tanh(us / lb) : us;
                ff[t][r] = (tg / (1.f + std::exp(-gs))) * tu;
            } else {
                ff[t][r] = (gs / (1.f + std::exp(-gs))) * us;
            }
        }
    }
}

template <int NT>
void dot_rows(const uint8_t* w, size_t row_bytes, const Acts& a, float* const* out, int r0, int r1, float scale) {
    float res0[NT], res1[NT];
    int r = r0;
    if constexpr (NT <= 3) {
        for (; r + 1 < r1; r += 2) {
            row_dot_pair<NT>(w + (size_t) r * row_bytes, w + (size_t) (r + 1) * row_bytes, a, res0, res1);
            for (int t = 0; t < NT; ++t) {
                out[t][r] = res0[t] * scale;
                out[t][r + 1] = res1[t] * scale;
            }
        }
    }
    for (; r < r1; ++r) {
        row_dot<NT>(w + (size_t) r * row_bytes, a, res0);
        for (int t = 0; t < NT; ++t) out[t][r] = res0[t] * scale;
    }
}

// windows wider than kMaxW (not from the pool) run in balanced slices of at most kMaxW
template <typename F>
inline void slices(int nt, F&& run) {
    const int k = (nt + kMaxW - 1) / kMaxW;
    for (int i = 0, t0 = 0; i < k; ++i) {
        const int w = nt / k + (i < nt % k ? 1 : 0);
        run(t0, w);
        t0 += w;
    }
}

}  // namespace

bool nvfp4_256_fits(int n) { return n % QK_NVFP4 == 0 && n / QK_NVFP4 <= kMaxBlocks; }

void nvfp4_256_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                       float* const* ff, int r0, int r1, float s_gate, float s_up) {
    slices(nt, [&](int t0, int w) {
        const Acts a(act + t0, w, n / QK_NVFP4);
        float* const* f = ff + t0;
        switch (w) {
            case 1: gu_rows<1, false>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 2: gu_rows<2, false>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 3: gu_rows<3, false>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 4: gu_rows<4, false>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 5: gu_rows<5, false>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 6: gu_rows<6, false>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 7: gu_rows<7, false>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            default: gu_rows<8, false>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
        }
    });
}

void nvfp4_256_gu_rows_situ(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                            float* const* ff, int r0, int r1, float s_gate, float s_up, float beta, float lb) {
    slices(nt, [&](int t0, int w) {
        const Acts a(act + t0, w, n / QK_NVFP4);
        float* const* f = ff + t0;
        switch (w) {
            case 1: gu_rows<1, true>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up, beta, lb); break;
            case 2: gu_rows<2, true>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up, beta, lb); break;
            case 3: gu_rows<3, true>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up, beta, lb); break;
            case 4: gu_rows<4, true>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up, beta, lb); break;
            case 5: gu_rows<5, true>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up, beta, lb); break;
            case 6: gu_rows<6, true>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up, beta, lb); break;
            case 7: gu_rows<7, true>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up, beta, lb); break;
            default: gu_rows<8, true>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up, beta, lb); break;
        }
    });
}

void nvfp4_256_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                    int r0, int r1, float scale) {
    slices(nt, [&](int t0, int k) {
        const Acts a(act + t0, k, n / QK_NVFP4);
        float* const* o = out + t0;
        switch (k) {
            case 1: dot_rows<1>(w, row_bytes, a, o, r0, r1, scale); break;
            case 2: dot_rows<2>(w, row_bytes, a, o, r0, r1, scale); break;
            case 3: dot_rows<3>(w, row_bytes, a, o, r0, r1, scale); break;
            case 4: dot_rows<4>(w, row_bytes, a, o, r0, r1, scale); break;
            case 5: dot_rows<5>(w, row_bytes, a, o, r0, r1, scale); break;
            case 6: dot_rows<6>(w, row_bytes, a, o, r0, r1, scale); break;
            case 7: dot_rows<7>(w, row_bytes, a, o, r0, r1, scale); break;
            default: dot_rows<8>(w, row_bytes, a, o, r0, r1, scale); break;
        }
    });
}

}  // namespace strata::kernels::cpu
