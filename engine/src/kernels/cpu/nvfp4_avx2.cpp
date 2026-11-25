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
inline void values64_decode(__m256i ql, __m256i qh, const __m256i kv,
                            __m256i& v01, __m256i& a01, __m256i& v23, __m256i& a23) {
    const __m256i lo = _mm256_shuffle_epi8(kv, ql);
    const __m256i hi = _mm256_shuffle_epi8(kv, qh);
    const __m256i sb02 = _mm256_unpacklo_epi64(lo, hi);
    const __m256i sb13 = _mm256_unpackhi_epi64(lo, hi);
    v01 = _mm256_permute2x128_si256(sb02, sb13, 0x20);
    v23 = _mm256_permute2x128_si256(sb02, sb13, 0x31);
    a01 = _mm256_abs_epi8(v01);
    a23 = _mm256_abs_epi8(v23);
}

inline void values64_both(const uint8_t* qs, const __m256i kv,
                          __m256i& v01, __m256i& a01, __m256i& v23, __m256i& a23) {
    const __m256i q = _mm256_loadu_si256((const __m256i*) qs);
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m256i ql = _mm256_and_si256(q, m4);
    const __m256i qh = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
    values64_decode(ql, qh, kv, v01, a01, v23, a23);
}

// Zero-copy K-tile decode: loads q0 (16 B) and q1 (16 B) and inverts perm(b) = (l3 h2 h1 h0 l2 l1 l0 h3) in YMM
// registers before running the exact same shuffle/unpack/permute/abs sequence as values64_both.
inline void values64_ktile_both(const uint8_t* q0_ptr, const uint8_t* q1_ptr, const __m256i kv,
                                __m256i& v01, __m256i& a01, __m256i& v23, __m256i& a23) {
    const __m128i q0 = _mm_loadu_si128((const __m128i*) q0_ptr);
    const __m128i q1 = _mm_loadu_si128((const __m128i*) q1_ptr);
    const __m256i qp = _mm256_inserti128_si256(_mm256_castsi128_si256(q0), q1, 1);
    const __m256i sr4 = _mm256_srli_epi16(qp, 4);
    const __m256i m08 = _mm256_set1_epi8(0x08);
    const __m256i m07 = _mm256_set1_epi8(0x07);
    const __m256i ql = _mm256_or_si256(_mm256_and_si256(sr4, m08),
                                       _mm256_and_si256(_mm256_srli_epi16(qp, 1), m07));
    const __m256i qh = _mm256_or_si256(_mm256_and_si256(_mm256_slli_epi16(qp, 3), m08),
                                       _mm256_and_si256(sr4, m07));
    values64_decode(ql, qh, kv, v01, a01, v23, a23);
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

template <int NT>
inline void ktile_accum_rows(const uint8_t* d_base, const uint8_t* q0_base, const uint8_t* q1_base,
                             int w0, int len, const __m256i kv, const __m256i ones,
                             const __m256i* aq01, const __m256i* aq23, const __m256* ady0, const __m256* ady1,
                             __m256 acc[][NT]) {
    for (int i = 0; i < len; ++i) {
        const int w = w0 + i;
        __m256i v01, a01, v23, a23;
        values64_ktile_both(q0_base + (size_t) w * 16, q1_base + (size_t) w * 16, kv, v01, a01, v23, a23);
        const uint8_t* d = d_base + (size_t) w * 4;
        const __m256 u01 = _mm256_set_m128(_mm_set1_ps(kUe4m3.v[d[1]]), _mm_set1_ps(kUe4m3.v[d[0]]));
        const __m256 u23 = _mm256_set_m128(_mm_set1_ps(kUe4m3.v[d[3]]), _mm_set1_ps(kUe4m3.v[d[2]]));
        for (int t = 0; t < NT; ++t) {
            const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(a01, _mm256_sign_epi8(aq01[t], v01)), ones);
            const __m256i p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(a23, _mm256_sign_epi8(aq23[t], v23)), ones);
            acc[i][t] = _mm256_fmadd_ps(_mm256_mul_ps(u01, ady0[t]), _mm256_cvtepi32_ps(p1), acc[i][t]);
            acc[i][t] = _mm256_fmadd_ps(_mm256_mul_ps(u23, ady1[t]), _mm256_cvtepi32_ps(p2), acc[i][t]);
        }
    }
}

template <int NT>
inline void ktile_load_acts(const Acts& a, int kbx, __m256i* aq01, __m256i* aq23, __m256* ady0, __m256* ady1) {
    for (int t = 0; t < NT; ++t) {
        const block_q8_0* yb = a.y[t] + 2 * kbx;
        aq01[t] = _mm256_loadu_si256((const __m256i*) yb[0].qs);
        aq23[t] = _mm256_loadu_si256((const __m256i*) yb[1].qs);
        const float* dy = a.dy + (size_t) t * 2 * a.nb + 2 * kbx;
        ady0[t] = _mm256_broadcast_ss(dy);
        ady1[t] = _mm256_broadcast_ss(dy + 1);
    }
}

template <int NT, bool SITU = false>
void ktile_gu_rows(const uint8_t* blob, const Acts& a, float* const* ff, int r0, int r1,
                   float sg, float su, float beta = 4.f, float lb = 25.f) {
    constexpr size_t kTileBytes = 4608;
    constexpr int kStrip = (NT <= 2) ? 64 : (NT <= 4 ? 32 : 16);
    const __m256i kv = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) kvalues_fp4));
    const __m256i ones = _mm256_set1_epi16(1);
    const int prefetch_ahead = prefetch_distance();
    for (int c0 = r0; c0 < r1;) {
        const int n_tile = c0 / 64;
        const int c1 = std::min(r1, (n_tile + 1) * 64);
        const int wr0 = c0 - n_tile * 64;
        const int cnt = c1 - c0;
        const uint8_t* tile_base = blob + (size_t) n_tile * (size_t) a.nb * kTileBytes;
        for (int s0 = 0; s0 < cnt; s0 += kStrip) {
            const int len = std::min(kStrip, cnt - s0);
            __m256 acc_g[kStrip][NT];
            __m256 acc_u[kStrip][NT];
            for (int i = 0; i < len; ++i) {
                for (int t = 0; t < NT; ++t) {
                    acc_g[i][t] = _mm256_setzero_ps();
                    acc_u[i][t] = _mm256_setzero_ps();
                }
            }
            for (int kbx = 0; kbx < a.nb; ++kbx) {
                const uint8_t* kt = tile_base + (size_t) kbx * kTileBytes;
                if (prefetch_ahead > 0 && kbx + 1 < a.nb) {
                    const char* pf = (const char*) (kt + kTileBytes);
                    _mm_prefetch(pf, _MM_HINT_T0);
                    _mm_prefetch(pf + 512, _MM_HINT_T0);
                    _mm_prefetch(pf + 2560, _MM_HINT_T0);
                    _mm_prefetch(pf + 3072, _MM_HINT_T0);
                }
                __m256i aq01[NT], aq23[NT];
                __m256 ady0[NT], ady1[NT];
                ktile_load_acts<NT>(a, kbx, aq01, aq23, ady0, ady1);
                ktile_accum_rows<NT>(kt, kt + 512, kt + 2560, wr0 + s0, len, kv, ones, aq01, aq23, ady0, ady1, acc_g);
                ktile_accum_rows<NT>(kt, kt + 512, kt + 2560, wr0 + s0 + 64, len, kv, ones, aq01, aq23, ady0, ady1, acc_u);
            }
            for (int i = 0; i < len; ++i) {
                const int r = c0 + s0 + i;
                for (int t = 0; t < NT; ++t) {
                    const float gs = hsum8(acc_g[i][t]) * sg;
                    const float us = hsum8(acc_u[i][t]) * su;
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
        c0 = c1;
    }
}

template <int NT>
void ktile_dot_rows(const uint8_t* w, const Acts& a, float* const* out, int r0, int r1, float scale) {
    constexpr size_t kTileBytes = 4608;
    constexpr int kStrip = (NT <= 2) ? 128 : (NT <= 4 ? 64 : 32);
    const __m256i kv = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) kvalues_fp4));
    const __m256i ones = _mm256_set1_epi16(1);
    const int prefetch_ahead = prefetch_distance();
    for (int c0 = r0; c0 < r1;) {
        const int n_tile = c0 / 128;
        const int c1 = std::min(r1, (n_tile + 1) * 128);
        const int wr0 = c0 - n_tile * 128;
        const int cnt = c1 - c0;
        const uint8_t* tile_base = w + (size_t) n_tile * (size_t) a.nb * kTileBytes;
        for (int s0 = 0; s0 < cnt; s0 += kStrip) {
            const int len = std::min(kStrip, cnt - s0);
            __m256 acc[kStrip][NT];
            for (int i = 0; i < len; ++i)
                for (int t = 0; t < NT; ++t) acc[i][t] = _mm256_setzero_ps();
            for (int kbx = 0; kbx < a.nb; ++kbx) {
                const uint8_t* kt = tile_base + (size_t) kbx * kTileBytes;
                if (prefetch_ahead > 0 && kbx + 1 < a.nb) {
                    const char* pf = (const char*) (kt + kTileBytes);
                    _mm_prefetch(pf, _MM_HINT_T0);
                    _mm_prefetch(pf + 512, _MM_HINT_T0);
                    _mm_prefetch(pf + 2560, _MM_HINT_T0);
                }
                __m256i aq01[NT], aq23[NT];
                __m256 ady0[NT], ady1[NT];
                ktile_load_acts<NT>(a, kbx, aq01, aq23, ady0, ady1);
                ktile_accum_rows<NT>(kt, kt + 512, kt + 2560, wr0 + s0, len, kv, ones, aq01, aq23, ady0, ady1, acc);
            }
            for (int i = 0; i < len; ++i) {
                const int r = c0 + s0 + i;
                for (int t = 0; t < NT; ++t) out[t][r] = hsum8(acc[i][t]) * scale;
            }
        }
        c0 = c1;
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

void nvfp4_256_ktile_gu_rows(const uint8_t* blob, int n, const void* const* act, int nt,
                             float* const* ff, int r0, int r1, float s_gate, float s_up) {
    slices(nt, [&](int t0, int w) {
        const Acts a(act + t0, w, n / QK_NVFP4);
        float* const* f = ff + t0;
        switch (w) {
            case 1: ktile_gu_rows<1, false>(blob, a, f, r0, r1, s_gate, s_up); break;
            case 2: ktile_gu_rows<2, false>(blob, a, f, r0, r1, s_gate, s_up); break;
            case 3: ktile_gu_rows<3, false>(blob, a, f, r0, r1, s_gate, s_up); break;
            case 4: ktile_gu_rows<4, false>(blob, a, f, r0, r1, s_gate, s_up); break;
            case 5: ktile_gu_rows<5, false>(blob, a, f, r0, r1, s_gate, s_up); break;
            case 6: ktile_gu_rows<6, false>(blob, a, f, r0, r1, s_gate, s_up); break;
            case 7: ktile_gu_rows<7, false>(blob, a, f, r0, r1, s_gate, s_up); break;
            default: ktile_gu_rows<8, false>(blob, a, f, r0, r1, s_gate, s_up); break;
        }
    });
}

void nvfp4_256_ktile_gu_rows_situ(const uint8_t* blob, int n, const void* const* act, int nt,
                                  float* const* ff, int r0, int r1, float s_gate, float s_up, float beta, float lb) {
    slices(nt, [&](int t0, int w) {
        const Acts a(act + t0, w, n / QK_NVFP4);
        float* const* f = ff + t0;
        switch (w) {
            case 1: ktile_gu_rows<1, true>(blob, a, f, r0, r1, s_gate, s_up, beta, lb); break;
            case 2: ktile_gu_rows<2, true>(blob, a, f, r0, r1, s_gate, s_up, beta, lb); break;
            case 3: ktile_gu_rows<3, true>(blob, a, f, r0, r1, s_gate, s_up, beta, lb); break;
            case 4: ktile_gu_rows<4, true>(blob, a, f, r0, r1, s_gate, s_up, beta, lb); break;
            case 5: ktile_gu_rows<5, true>(blob, a, f, r0, r1, s_gate, s_up, beta, lb); break;
            case 6: ktile_gu_rows<6, true>(blob, a, f, r0, r1, s_gate, s_up, beta, lb); break;
            case 7: ktile_gu_rows<7, true>(blob, a, f, r0, r1, s_gate, s_up, beta, lb); break;
            default: ktile_gu_rows<8, true>(blob, a, f, r0, r1, s_gate, s_up, beta, lb); break;
        }
    });
}

void nvfp4_256_ktile_rows(const uint8_t* w, int n, const void* const* act, int nt, float* const* out,
                          int r0, int r1, float scale) {
    slices(nt, [&](int t0, int k) {
        const Acts a(act + t0, k, n / QK_NVFP4);
        float* const* o = out + t0;
        switch (k) {
            case 1: ktile_dot_rows<1>(w, a, o, r0, r1, scale); break;
            case 2: ktile_dot_rows<2>(w, a, o, r0, r1, scale); break;
            case 3: ktile_dot_rows<3>(w, a, o, r0, r1, scale); break;
            case 4: ktile_dot_rows<4>(w, a, o, r0, r1, scale); break;
            case 5: ktile_dot_rows<5>(w, a, o, r0, r1, scale); break;
            case 6: ktile_dot_rows<6>(w, a, o, r0, r1, scale); break;
            case 7: ktile_dot_rows<7>(w, a, o, r0, r1, scale); break;
            default: ktile_dot_rows<8>(w, a, o, r0, r1, scale); break;
        }
    });
}

}  // namespace strata::kernels::cpu
