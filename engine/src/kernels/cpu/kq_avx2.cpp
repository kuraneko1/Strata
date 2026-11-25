// src/kernels/cpu/kq_avx2.cpp - Unsloth UD-Q4_K_XL's expert rows (Q4_K gate/up, Q5_1 and Q8_0 down) for several
// tokens at once, BIT-EXACT against ggml-cpu's own dot products.
//
// ggml-cpu is built for AVX2 here (STRATA_PORTABLE: GGML_AVX2=ON, GGML_AVX512=OFF), so its vec_dot for these formats
// is the __AVX2__ branch of ggml-cpu/arch/x86/quants.c: ggml_vec_dot_q4_K_q8_K, ggml_vec_dot_q5_1_q8_1,
// ggml_vec_dot_q8_0_q8_0.  The functions below are those branches with the loop over tokens moved inside the loop
// over blocks: everything that depends only on the weights (the 4-bit unpacking, the 6-bit scales and mins, the
// fifth bits, the absolute values of the Q8_0 weights, the fp16 scales) is done once per block, and every token then
// runs exactly ggml's integer chain and ggml's float operations in ggml's order - so each token's result is ggml's
// to the bit (native_expert_parity checks it), whatever the number of tokens.  Unlike the i-quant kernels (#152)
// the group size therefore never changes an answer.
//
// The gain is the weight-side work and the weight bytes, read once per verify window instead of once per token.
#include "strata/kernels/cpu/kq_avx2.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

constexpr int MAXT = 8;   // the verify window's tokens per group (kVerifyMaxT)

inline float h2f(ggml_half h) {
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) u)));
}
// the fp16 pair {d, dmin} / {d, m} / {d, s} a block starts with (a union in ggml-common.h's C++ declarations)
inline float half_at(const void* block, int i) {
    ggml_half h;
    std::memcpy(&h, (const uint8_t*) block + 2 * i, 2);
    return h2f(h);
}

// ---- ggml's helpers (ggml-cpu/arch/x86/quants.c, the AVX2 / non-VNNI forms), verbatim
inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}
inline __m256i bytes_from_bits_32(const uint8_t* x) {
    uint32_t x32;
    std::memcpy(&x32, x, sizeof(uint32_t));
    const __m256i shuf_mask = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101,
                                                0x0000000000000000);
    __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32((int) x32), shuf_mask);
    const __m256i bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
    bytes = _mm256_or_si256(bytes, bit_mask);
    return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}
inline __m256i bytes_from_nibbles_32(const uint8_t* rsi) {
    const __m128i tmp = _mm_loadu_si128((const __m128i*) rsi);
    const __m256i bytes = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), _mm_srli_epi16(tmp, 4), 1);
    const __m256i lowMask = _mm256_set1_epi8(0xF);
    return _mm256_and_si256(lowMask, bytes);
}
inline __m256 sum_i16_pairs_float(const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(ones, x));
}
inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
    return sum_i16_pairs_float(_mm256_maddubs_epi16(ax, sy));
}
alignas(32) const uint8_t k_shuffle[256] = {
     0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
     4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
     8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
    12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15};
inline __m256i get_scale_shuffle_k4(int i) { return _mm256_loadu_si256((const __m256i*) k_shuffle + i); }

// ---- Q4_K x Q8_K (ggml_vec_dot_q4_K_q8_K, __AVX2__)
void q4k_dot(const block_q4_K* x, int nb, const void* const* act, int nt, float* res) {
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc[MAXT];
    __m128 acc_m[MAXT];
    for (int t = 0; t < nt; ++t) { acc[t] = _mm256_setzero_ps(); acc_m[t] = _mm_setzero_ps(); }
    for (int i = 0; i < nb; ++i) {
        const float xd = half_at(&x[i], 0), xdmin = half_at(&x[i], 1);
        uint32_t utmp[4];
        std::memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32((int) utmp[3], (int) utmp[2],
                                                                           (int) utmp[1], (int) utmp[0]));
        const __m128i mins = _mm256_extracti128_si256(mins_and_scales, 1);
        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);
        __m256i scale_l[4], scale_h[4], q4l[4], q4h[4];
        const uint8_t* q4 = x[i].qs;
        for (int j = 0; j < QK_K / 64; ++j) {
            scale_l[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
            scale_h[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
            const __m256i q4bits = _mm256_loadu_si256((const __m256i*) q4);
            q4 += 32;
            q4l[j] = _mm256_and_si256(q4bits, m4);
            q4h[j] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
        }
        for (int t = 0; t < nt; ++t) {
            const block_q8_K* y = (const block_q8_K*) act[t] + i;
            const float d = y->d * xd;
            const float dmin = -y->d * xdmin;
            const int8_t* q8 = y->qs;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y->bsums);
            const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod = _mm_madd_epi16(mins, q8s);
            acc_m[t] = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m[t]);
            __m256i sumi = _mm256_setzero_si256();
            for (int j = 0; j < QK_K / 64; ++j) {
                const __m256i q8l = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                __m256i p16l = _mm256_maddubs_epi16(q4l[j], q8l);
                p16l = _mm256_madd_epi16(scale_l[j], p16l);
                const __m256i q8h = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                __m256i p16h = _mm256_maddubs_epi16(q4h[j], q8h);
                p16h = _mm256_madd_epi16(scale_h[j], p16h);
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
            }
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) {
        __m128 m = acc_m[t];
        m = _mm_add_ps(m, _mm_movehl_ps(m, m));
        m = _mm_add_ss(m, _mm_movehdup_ps(m));
        res[t] = hsum_float_8(acc[t]) + _mm_cvtss_f32(m);
    }
}

// ---- Q5_K x Q8_K (ggml_vec_dot_q5_K_q8_K, __AVX2__), bit-exact per token
template <int NT>
void q5k_dot_t(const block_q5_K* x, int nb, const void* const* act, float* res) {
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m128i mzero = _mm_setzero_si128();
    const __m256i mone = _mm256_set1_epi8(1);
    __m256 acc[NT];
    float summs[NT];
    for (int t = 0; t < NT; ++t) { acc[t] = _mm256_setzero_ps(); summs[t] = 0.0f; }
    for (int i = 0; i < nb; ++i) {
        if (i + 2 < nb) {
            _mm_prefetch((const char*) (x + i + 2), _MM_HINT_T0);
            _mm_prefetch((const char*) (x + i + 2) + 64, _MM_HINT_T0);
            _mm_prefetch((const char*) (x + i + 2) + 128, _MM_HINT_T0);
        }
        const float xd = half_at(&x[i], 0), xdmin = half_at(&x[i], 1);
        uint32_t utmp[4];
        std::memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32((int) utmp[3], (int) utmp[2],
                                                                           (int) utmp[1], (int) utmp[0]));
        const __m128i mins = _mm256_extracti128_si256(mins_and_scales, 1);
        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);
        const __m256i hbits = _mm256_loadu_si256((const __m256i*) x[i].qh);
        __m256i hmask = mone;
        __m256i scale_0[4], scale_1[4], q5_0[4], q5_1[4];
        const uint8_t* q5 = x[i].qs;
        int bit = 0;
        for (int j = 0; j < QK_K / 64; ++j) {
            scale_0[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
            scale_1[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
            const __m256i q5bits = _mm256_loadu_si256((const __m256i*) q5);
            q5 += 32;
            const __m256i q5l_0 = _mm256_and_si256(q5bits, m4);
            const __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            q5_0[j] = _mm256_add_epi8(q5l_0, q5h_0);
            hmask = _mm256_slli_epi16(hmask, 1);
            const __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
            const __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            q5_1[j] = _mm256_add_epi8(q5l_1, q5h_1);
            hmask = _mm256_slli_epi16(hmask, 1);
        }
        for (int t = 0; t < NT; ++t) {
            const block_q8_K* y = (const block_q8_K*) act[t] + i;
            const float d = y->d * xd;
            const float dmin = -y->d * xdmin;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y->bsums);
            const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod = _mm_madd_epi16(mins, q8s);
            const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
            summs[t] += dmin * _mm_extract_epi32(hsum, 0);
            const int8_t* q8 = y->qs;
            __m256i sumi = _mm256_setzero_si256();
            for (int j = 0; j < QK_K / 64; ++j) {
                const __m256i q8_0 = _mm256_loadu_si256((const __m256i*) q8); q8 += 32;
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) q8); q8 += 32;
                __m256i p16_0 = _mm256_maddubs_epi16(q5_0[j], q8_0);
                __m256i p16_1 = _mm256_maddubs_epi16(q5_1[j], q8_1);
                p16_0 = _mm256_madd_epi16(scale_0[j], p16_0);
                p16_1 = _mm256_madd_epi16(scale_1[j], p16_1);
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            }
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum_float_8(acc[t]) + summs[t];
}

void q5k_dot_nt(const block_q5_K* x, int nb, const void* const* act, int nt, float* res) {
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int k = nt - t0 < 4 ? nt - t0 : 4;
        switch (k) {
            case 1: q5k_dot_t<1>(x, nb, act + t0, res + t0); break;
            case 2: q5k_dot_t<2>(x, nb, act + t0, res + t0); break;
            case 3: q5k_dot_t<3>(x, nb, act + t0, res + t0); break;
            default: q5k_dot_t<4>(x, nb, act + t0, res + t0); break;
        }
    }
}

// ---- Q5_1 x Q8_1 (ggml_vec_dot_q5_1_q8_1, __AVX2__)
template <int NT>
void q5_1_dot(const block_q5_1* x, int nb, const void* const* act, float* res) {
    __m256 acc[NT];
    float summs[NT];
    for (int t = 0; t < NT; ++t) { acc[t] = _mm256_setzero_ps(); summs[t] = 0.0f; }
    for (int ib = 0; ib < nb; ++ib) {
        const __m256 dx = _mm256_set1_ps(half_at(&x[ib], 0));
        const float xm = half_at(&x[ib], 1);
        __m256i qx = bytes_from_nibbles_32(x[ib].qs);
        __m256i bxhi = bytes_from_bits_32(x[ib].qh);
        bxhi = _mm256_and_si256(bxhi, _mm256_set1_epi8(0x10));
        qx = _mm256_or_si256(qx, bxhi);
        for (int t = 0; t < NT; ++t) {
            const block_q8_1* y = (const block_q8_1*) act[t] + ib;
            summs[t] += xm * half_at(y, 1);
            const __m256 dy = _mm256_set1_ps(half_at(y, 0));
            const __m256i qy = _mm256_loadu_si256((const __m256i*) y->qs);
            const __m256 q = mul_sum_us8_pairs_float(qx, qy);
            acc[t] = _mm256_fmadd_ps(q, _mm256_mul_ps(dx, dy), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum_float_8(acc[t]) + summs[t];
}

// ---- Q8_0 x Q8_0 (ggml_vec_dot_q8_0_q8_0, __AVX2__; mul_sum_i8_pairs_float without AVX-VNNI-INT8)
template <int NT>
void q8_0_dot(const block_q8_0* x, int nb, const void* const* act, float* res) {
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        if ((ib & 1) == 0 && ib + 4 < nb) _mm_prefetch((const char*) (x + ib + 4), _MM_HINT_T0);
        const float xd = h2f(x[ib].d);
        const __m256i qx = _mm256_loadu_si256((const __m256i*) x[ib].qs);
        const __m256i ax = _mm256_sign_epi8(qx, qx);
        for (int t = 0; t < NT; ++t) {
            const block_q8_0* y = (const block_q8_0*) act[t] + ib;
            const __m256 d = _mm256_set1_ps(xd * h2f(y->d));
            const __m256i qy = _mm256_loadu_si256((const __m256i*) y->qs);
            const __m256i sy = _mm256_sign_epi8(qy, qx);
            const __m256 q = mul_sum_us8_pairs_float(ax, sy);
            acc[t] = _mm256_fmadd_ps(d, q, acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum_float_8(acc[t]);
}

void q5_1_dot_nt(const block_q5_1* x, int nb, const void* const* act, int nt, float* res) {
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int k = nt - t0 < 4 ? nt - t0 : 4;
        switch (k) {
            case 1: q5_1_dot<1>(x, nb, act + t0, res + t0); break;
            case 2: q5_1_dot<2>(x, nb, act + t0, res + t0); break;
            case 3: q5_1_dot<3>(x, nb, act + t0, res + t0); break;
            default: q5_1_dot<4>(x, nb, act + t0, res + t0); break;
        }
    }
}

void q8_0_dot_nt(const block_q8_0* x, int nb, const void* const* act, int nt, float* res) {
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int k = nt - t0 < 4 ? nt - t0 : 4;
        switch (k) {
            case 1: q8_0_dot<1>(x, nb, act + t0, res + t0); break;
            case 2: q8_0_dot<2>(x, nb, act + t0, res + t0); break;
            case 3: q8_0_dot<3>(x, nb, act + t0, res + t0); break;
            default: q8_0_dot<4>(x, nb, act + t0, res + t0); break;
        }
    }
}

void dot_rows(int type, const uint8_t* row, int n, const void* const* act, int nt, float* res) {
    switch (type) {
        case 12: q4k_dot((const block_q4_K*) row, n / QK_K, act, nt, res); break;
        case 13: q5k_dot_nt((const block_q5_K*) row, n / QK_K, act, nt, res); break;
        case 7: q5_1_dot_nt((const block_q5_1*) row, n / QK5_1, act, nt, res); break;
        case 8: q8_0_dot_nt((const block_q8_0*) row, n / QK8_0, act, nt, res); break;
        default: break;
    }
}

}  // namespace

void bf16_rows_dot_multi(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) {
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int m = nt - t0 < 4 ? nt - t0 : 4;
        const float* x0 = x + (size_t) t0 * cols;
        const float* x1 = x0 + cols;
        const float* x2 = x1 + cols;
        const float* x3 = x2 + cols;
        for (int r = 0; r < rows; ++r) {
            const uint16_t* wr = w + (size_t) r * (size_t) cols;
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(),
                   a3 = _mm256_setzero_ps();
            for (int c = 0; c + 8 <= cols; c += 8) {
                const __m256 wf = _mm256_castsi256_ps(
                    _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*) (wr + c))), 16));
                a0 = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x0 + c), a0);
                if (m > 1) a1 = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x1 + c), a1);
                if (m > 2) a2 = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x2 + c), a2);
                if (m > 3) a3 = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x3 + c), a3);
            }
            out[(size_t) t0 * rows + r] = hsum_float_8(a0);
            if (m > 1) out[(size_t) (t0 + 1) * rows + r] = hsum_float_8(a1);
            if (m > 2) out[(size_t) (t0 + 2) * rows + r] = hsum_float_8(a2);
            if (m > 3) out[(size_t) (t0 + 3) * rows + r] = hsum_float_8(a3);
        }
    }
}

void bf16_rows_dot(const uint16_t* w, int rows, int cols, const float* x, float* out) {
    for (int r = 0; r < rows; ++r) {
        const uint16_t* wr = w + (size_t) r * (size_t) cols;
        __m256 acc = _mm256_setzero_ps();
        int c = 0;
        for (; c + 8 <= cols; c += 8) {
            const __m256i h = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*) (wr + c)));
            acc = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(h, 16)), _mm256_loadu_ps(x + c), acc);
        }
        float s = hsum_float_8(acc);
        for (; c < cols; ++c) {
            const uint32_t b = (uint32_t) wr[c] << 16;
            float f;
            std::memcpy(&f, &b, 4);
            s += f * x[c];
        }
        out[r] = s;
    }
}

void kq256_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1) {
    for (int t0 = 0; t0 < nt; t0 += MAXT) {
        const int m = nt - t0 < MAXT ? nt - t0 : MAXT;
        float g[MAXT], u[MAXT];
        for (int r = r0; r < r1; ++r) {
            dot_rows(type, blob + (size_t) r * gu_row, n, act + t0, m, g);
            dot_rows(type, blob + up_off + (size_t) r * gu_row, n, act + t0, m, u);
            // native_gu_rows' own SwiGLU expression: the same bits as the per-token ggml path
            for (int t = 0; t < m; ++t) ff[t0 + t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
        }
    }
}

// Extract bit BIT (0..7) of every byte in hbits into bit 4 (0x10) using at most 1 immediate shift (0 variable-count shifts).
template <int BIT>
inline __m256i q5k_bit_to_4(const __m256i hbits) {
    const __m256i m = _mm256_and_si256(hbits, _mm256_set1_epi8((char) (1u << BIT)));
    if constexpr (BIT < 4) return _mm256_slli_epi16(m, 4 - BIT);
    else if constexpr (BIT == 4) return m;
    else return _mm256_srli_epi16(m, BIT - 4);
}

// ---- Zero-spill single-row T=1 Q5_K x Q8_K with precomputed q8s, prefetch, and unrolled immediate shifts (bit-exact to dot_q5_K_q8_K)
inline float q5k_dot_t1_fast(const block_q5_K* x, int nb, const block_q8_K* y0, const __m128i* q8s_pre) {
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m128i mzero = _mm_setzero_si128();
    __m256 acc = _mm256_setzero_ps();
    float summs = 0.0f;
    for (int i = 0; i < nb; ++i) {
        if (i + 2 < nb) {
            _mm_prefetch((const char*) (x + i + 2), _MM_HINT_T0);
            _mm_prefetch((const char*) (x + i + 2) + 64, _MM_HINT_T0);
            _mm_prefetch((const char*) (x + i + 2) + 128, _MM_HINT_T0);
        }
        const block_q8_K* y = y0 + i;
        const float yd = y->d;
        const float d = yd * half_at(&x[i], 0);
        const float dmin = -yd * half_at(&x[i], 1);

        uint32_t utmp[4];
        std::memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32((int) utmp[3], (int) utmp[2],
                                                                           (int) utmp[1], (int) utmp[0]));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s_pre[i]);
        const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
        summs += dmin * _mm_extract_epi32(hsum, 0);

        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);
        const __m256i hbits = _mm256_loadu_si256((const __m256i*) x[i].qh);
        const uint8_t* q5 = x[i].qs;
        const int8_t* q8 = y->qs;
        __m256i sumi = _mm256_setzero_si256();

        auto step = [&]<int J>() {
            const __m256i scale_0 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * J + 0));
            const __m256i scale_1 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * J + 1));
            const __m256i q5bits = _mm256_loadu_si256((const __m256i*) (q5 + 32 * J));
            const __m256i q5_0 = _mm256_add_epi8(_mm256_and_si256(q5bits, m4), q5k_bit_to_4<2 * J + 0>(hbits));
            const __m256i q5_1 = _mm256_add_epi8(_mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4), q5k_bit_to_4<2 * J + 1>(hbits));
            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*) (q8 + 64 * J));
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) (q8 + 64 * J + 32));
            const __m256i p16_0 = _mm256_madd_epi16(scale_0, _mm256_maddubs_epi16(q5_0, q8_0));
            const __m256i p16_1 = _mm256_madd_epi16(scale_1, _mm256_maddubs_epi16(q5_1, q8_1));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
        };
        step.template operator()<0>();
        step.template operator()<1>();
        step.template operator()<2>();
        step.template operator()<3>();

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
    return hsum_float_8(acc) + summs;
}

// ---- Paired-row T=1 Q8_0 x Q8_0 (2 adjacent down rows sharing Q8_0 activations & precomputed dy), bit-exact to q8_0_dot<1>
inline void q8_0_dot_t1_pair(const block_q8_0* x0, const block_q8_0* x1, int nb,
                             const block_q8_0* y0, const float* dy_pre, float& r0, float& r1) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        if ((ib & 1) == 0 && ib + 4 < nb) {
            _mm_prefetch((const char*) (x0 + ib + 4), _MM_HINT_T0);
            _mm_prefetch((const char*) (x1 + ib + 4), _MM_HINT_T0);
        }
        const float dy = dy_pre[ib];
        const __m256i qy = _mm256_loadu_si256((const __m256i*) y0[ib].qs);

        const __m256 d0 = _mm256_set1_ps(h2f(x0[ib].d) * dy);
        const __m256i qx0 = _mm256_loadu_si256((const __m256i*) x0[ib].qs);
        const __m256 q0 = mul_sum_us8_pairs_float(_mm256_sign_epi8(qx0, qx0), _mm256_sign_epi8(qy, qx0));
        acc0 = _mm256_fmadd_ps(d0, q0, acc0);

        const __m256 d1 = _mm256_set1_ps(h2f(x1[ib].d) * dy);
        const __m256i qx1 = _mm256_loadu_si256((const __m256i*) x1[ib].qs);
        const __m256 q1 = mul_sum_us8_pairs_float(_mm256_sign_epi8(qx1, qx1), _mm256_sign_epi8(qy, qx1));
        acc1 = _mm256_fmadd_ps(d1, q1, acc1);
    }
    r0 = hsum_float_8(acc0);
    r1 = hsum_float_8(acc1);
}

inline float q8_0_dot_t1_single(const block_q8_0* x0, int nb, const block_q8_0* y0, const float* dy_pre) {
    __m256 acc0 = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        if ((ib & 1) == 0 && ib + 4 < nb) _mm_prefetch((const char*) (x0 + ib + 4), _MM_HINT_T0);
        const __m256 d0 = _mm256_set1_ps(h2f(x0[ib].d) * dy_pre[ib]);
        const __m256i qx0 = _mm256_loadu_si256((const __m256i*) x0[ib].qs);
        const __m256i qy = _mm256_loadu_si256((const __m256i*) y0[ib].qs);
        acc0 = _mm256_fmadd_ps(d0, mul_sum_us8_pairs_float(_mm256_sign_epi8(qx0, qx0), _mm256_sign_epi8(qy, qx0)), acc0);
    }
    return hsum_float_8(acc0);
}

void kq256_gu_rows_situ(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                        int nt, float* const* ff, int r0, int r1, float beta, float lb) {
    if (nt == 1 && type == 13 && n / QK_K <= 64) {
        const int nb = n / QK_K;
        const block_q8_K* y0 = (const block_q8_K*) act[0];
        __m128i q8s_pre[64];
        for (int i = 0; i < nb; ++i) {
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y0[i].bsums);
            q8s_pre[i] = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        }
        float* const f0 = ff[0];
        for (int r = r0; r < r1; ++r) {
            const float gs = q5k_dot_t1_fast((const block_q5_K*) (blob + (size_t) r * gu_row), nb, y0, q8s_pre);
            const float us = q5k_dot_t1_fast((const block_q5_K*) (blob + up_off + (size_t) r * gu_row), nb, y0, q8s_pre);
            const float tg = beta * std::tanh((beta == 4.0f) ? (gs * 0.25f) : (gs / beta));
            const float tu = lb > 0.f ? lb * std::tanh(us / lb) : us;
            f0[r] = (tg / (1.f + std::exp(-gs))) * tu;
        }
        return;
    }
    for (int t0 = 0; t0 < nt; t0 += MAXT) {
        const int m = nt - t0 < MAXT ? nt - t0 : MAXT;
        float g[MAXT], u[MAXT];
        for (int r = r0; r < r1; ++r) {
            dot_rows(type, blob + (size_t) r * gu_row, n, act + t0, m, g);
            dot_rows(type, blob + up_off + (size_t) r * gu_row, n, act + t0, m, u);
            for (int t = 0; t < m; ++t) {
                const float gs = g[t], us = u[t];
                const float tg = beta * std::tanh(gs / beta);
                const float tu = lb > 0.f ? lb * std::tanh(us / lb) : us;
                ff[t0 + t][r] = (tg / (1.f + std::exp(-gs))) * tu;
            }
        }
    }
}

void kq256_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1) {
    if (nt == 1 && type == 8 && n / QK8_0 <= 256) {
        const int nb = n / QK8_0;
        const block_q8_0* y0 = (const block_q8_0*) act[0];
        float dy_pre[256];
        for (int ib = 0; ib < nb; ++ib) dy_pre[ib] = h2f(y0[ib].d);
        float* const o0 = out[0];
        int r = r0;
        for (; r + 1 < r1; r += 2) {
            float s0 = 0.0f, s1 = 0.0f;
            q8_0_dot_t1_pair((const block_q8_0*) (w + (size_t) r * row_bytes),
                             (const block_q8_0*) (w + (size_t) (r + 1) * row_bytes),
                             nb, y0, dy_pre, s0, s1);
            o0[r] = s0;
            o0[r + 1] = s1;
        }
        for (; r < r1; ++r) {
            o0[r] = q8_0_dot_t1_single((const block_q8_0*) (w + (size_t) r * row_bytes), nb, y0, dy_pre);
        }
        return;
    }
    for (int t0 = 0; t0 < nt; t0 += MAXT) {
        const int m = nt - t0 < MAXT ? nt - t0 : MAXT;
        float s[MAXT];
        for (int r = r0; r < r1; ++r) {
            dot_rows(type, w + (size_t) r * row_bytes, n, act + t0, m, s);
            for (int t = 0; t < m; ++t) out[t0 + t][r] = s[t];
        }
    }
}

namespace {

constexpr int kMaxQ5kSbs = 64;     // up to 16384 cols
constexpr int kMaxQ80Pairs = 256;  // up to 16384 cols (512 Q8_0 blocks)

template <int NT>
inline void q5k_ktile_row_step(const uint8_t* hdr, const __m256i hbits,
                               const uint8_t* qs0, const uint8_t* qs1,
                               const uint8_t* qs2, const uint8_t* qs3,
                               const block_q8_K* const* yt, const __m128i (&q8s_pre)[NT][kMaxQ5kSbs], int ksb,
                               __m256 acc[NT], float summs[NT]) {
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m128i mzero = _mm_setzero_si128();
    const float xd = half_at(hdr, 0), xdmin = half_at(hdr, 1);
    uint32_t utmp[4];
    std::memcpy(utmp, hdr + 4, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
    const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32((int) utmp[3], (int) utmp[2],
                                                                       (int) utmp[1], (int) utmp[0]));
    const __m128i mins = _mm256_extracti128_si256(mins_and_scales, 1);
    const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
    const __m256i scales = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);

    const uint8_t* qs[4] = {qs0, qs1, qs2, qs3};
    // Process J=0..3 one chunk at a time so decoded q5_0/q5_1/s0/s1 stay in YMM registers with zero stack spills.
    for (int t = 0; t < NT; ++t) {
        const block_q8_K* y = yt[t] + ksb;
        const float dmin = -y->d * xdmin;
        const __m128i prod = _mm_madd_epi16(mins, q8s_pre[t][ksb]);
        const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
        summs[t] += dmin * _mm_extract_epi32(hsum, 0);
    }
    __m256i sumi[NT];
    for (int t = 0; t < NT; ++t) sumi[t] = _mm256_setzero_si256();
    auto step = [&]<int J>() {
        const __m256i s0 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * J + 0));
        const __m256i s1 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * J + 1));
        const __m256i q5bits = _mm256_loadu_si256((const __m256i*) qs[J]);
        const __m256i q5_0 = _mm256_add_epi8(_mm256_and_si256(q5bits, m4), q5k_bit_to_4<2 * J + 0>(hbits));
        const __m256i q5_1 = _mm256_add_epi8(_mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4), q5k_bit_to_4<2 * J + 1>(hbits));
        for (int t = 0; t < NT; ++t) {
            const int8_t* q8 = yt[t][ksb].qs;
            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*) (q8 + 64 * J));
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) (q8 + 64 * J + 32));
            const __m256i p16_0 = _mm256_madd_epi16(s0, _mm256_maddubs_epi16(q5_0, q8_0));
            const __m256i p16_1 = _mm256_madd_epi16(s1, _mm256_maddubs_epi16(q5_1, q8_1));
            sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(p16_0, p16_1));
        }
    };
    step.template operator()<0>();
    step.template operator()<1>();
    step.template operator()<2>();
    step.template operator()<3>();
    for (int t = 0; t < NT; ++t) {
        const float d = yt[t][ksb].d * xd;
        acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi[t]), acc[t]);
    }
}

template <int NT>
void ktile_q5k_gu_t(const uint8_t* blob, int n, const void* const* act, float* const* ff,
                    int r0, int r1, float beta, float lb) {
    constexpr size_t kTileBytes = 22528;
    constexpr int kStrip = (NT == 1) ? 64 : ((NT == 2) ? 32 : ((NT <= 4) ? 16 : 8));
    const int k_sbs = n / QK_K;
    const block_q8_K* yt[NT];
    __m128i q8s_pre[NT][kMaxQ5kSbs];
    for (int t = 0; t < NT; ++t) {
        yt[t] = (const block_q8_K*) act[t];
        for (int i = 0; i < k_sbs; ++i) {
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) yt[t][i].bsums);
            q8s_pre[t][i] = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        }
    }
    for (int c0 = r0; c0 < r1;) {
        const int n_tile = c0 >> 6;
        const int c1 = std::min(r1, (n_tile + 1) * 64);
        const int wr0 = c0 - n_tile * 64;
        const int cnt = c1 - c0;
        const uint8_t* tile_base = blob + (size_t) n_tile * (size_t) k_sbs * kTileBytes;
        for (int s0 = 0; s0 < cnt; s0 += kStrip) {
            const int len = std::min(kStrip, cnt - s0);
            __m256 acc_g[kStrip][NT];
            __m256 acc_u[kStrip][NT];
            float summs_g[kStrip][NT];
            float summs_u[kStrip][NT];
            for (int i = 0; i < len; ++i) {
                for (int t = 0; t < NT; ++t) {
                    acc_g[i][t] = _mm256_setzero_ps();
                    acc_u[i][t] = _mm256_setzero_ps();
                    summs_g[i][t] = 0.0f;
                    summs_u[i][t] = 0.0f;
                }
            }
            for (int ksb = 0; ksb < k_sbs; ++ksb) {
                const uint8_t* kt = tile_base + (size_t) ksb * kTileBytes;
                if (ksb + 1 < k_sbs) {
                    const char* pf = (const char*) (kt + kTileBytes);
                    _mm_prefetch(pf + (size_t) (wr0 + s0) * 16u, _MM_HINT_T0);
                    _mm_prefetch(pf + 2048u + (size_t) (wr0 + s0) * 32u, _MM_HINT_T0);
                    _mm_prefetch(pf + 6144u + (size_t) (wr0 + s0) * 32u, _MM_HINT_T0);
                    _mm_prefetch(pf + 10240u + (size_t) (wr0 + s0) * 32u, _MM_HINT_T0);
                    _mm_prefetch(pf + 14336u + (size_t) (wr0 + s0) * 32u, _MM_HINT_T0);
                    _mm_prefetch(pf + 18432u + (size_t) (wr0 + s0) * 32u, _MM_HINT_T0);
                }
                const uint8_t* hdr_base = kt;
                const uint8_t* qh_base  = kt + 2048u;
                const uint8_t* qs0_base = kt + 6144u;
                const uint8_t* qs1_base = kt + 10240u;
                const uint8_t* qs2_base = kt + 14336u;
                const uint8_t* qs3_base = kt + 18432u;
                for (int i = 0; i < len; ++i) {
                    const int wg = wr0 + s0 + i;
                    const __m256i hg = _mm256_loadu_si256((const __m256i*) (qh_base + (size_t) wg * 32u));
                    q5k_ktile_row_step<NT>(hdr_base + (size_t) wg * 16u, hg,
                                           qs0_base + (size_t) wg * 32u, qs1_base + (size_t) wg * 32u,
                                           qs2_base + (size_t) wg * 32u, qs3_base + (size_t) wg * 32u,
                                           yt, q8s_pre, ksb, acc_g[i], summs_g[i]);
                    const int wu = wg + 64;
                    const __m256i hu = _mm256_loadu_si256((const __m256i*) (qh_base + (size_t) wu * 32u));
                    q5k_ktile_row_step<NT>(hdr_base + (size_t) wu * 16u, hu,
                                           qs0_base + (size_t) wu * 32u, qs1_base + (size_t) wu * 32u,
                                           qs2_base + (size_t) wu * 32u, qs3_base + (size_t) wu * 32u,
                                           yt, q8s_pre, ksb, acc_u[i], summs_u[i]);
                }
            }
            for (int i = 0; i < len; ++i) {
                const int r = c0 + s0 + i;
                for (int t = 0; t < NT; ++t) {
                    const float gs = hsum_float_8(acc_g[i][t]) + summs_g[i][t];
                    const float us = hsum_float_8(acc_u[i][t]) + summs_u[i][t];
                    const float tg = beta * std::tanh((beta == 4.0f) ? (gs * 0.25f) : (gs / beta));
                    const float tu = lb > 0.f ? lb * std::tanh(us / lb) : us;
                    ff[t][r] = (tg / (1.f + std::exp(-gs))) * tu;
                }
            }
        }
        c0 = c1;
    }
}

template <int NT>
void ktile_q8_0_down_t(const uint8_t* w, int n, const void* const* act, float* const* out, int r0, int r1) {
    constexpr size_t kTileBytes = 8704;
    constexpr int kStrip = (NT == 1) ? 128 : ((NT == 2) ? 64 : ((NT <= 4) ? 32 : 16));
    const int k_64 = n / 64;
    const block_q8_0* yt[NT];
    float dy_pre[NT][kMaxQ80Pairs * 2];
    for (int t = 0; t < NT; ++t) {
        yt[t] = (const block_q8_0*) act[t];
        for (int ib = 0; ib < 2 * k_64; ++ib) dy_pre[t][ib] = h2f(yt[t][ib].d);
    }
    for (int c0 = r0; c0 < r1;) {
        const int n_tile = c0 >> 7;
        const int c1 = std::min(r1, (n_tile + 1) * 128);
        const int wr0 = c0 - n_tile * 128;
        const int cnt = c1 - c0;
        const uint8_t* tile_base = w + (size_t) n_tile * (size_t) k_64 * kTileBytes;
        for (int s0 = 0; s0 < cnt; s0 += kStrip) {
            const int len = std::min(kStrip, cnt - s0);
            __m256 acc[kStrip][NT];
            for (int i = 0; i < len; ++i)
                for (int t = 0; t < NT; ++t) acc[i][t] = _mm256_setzero_ps();
            for (int kb = 0; kb < k_64; ++kb) {
                const uint8_t* kt = tile_base + (size_t) kb * kTileBytes;
                if (kb + 1 < k_64) {
                    const char* pf = (const char*) (kt + kTileBytes);
                    _mm_prefetch(pf + (size_t) (wr0 + s0) * 4u, _MM_HINT_T0);
                    _mm_prefetch(pf + 512u + (size_t) (wr0 + s0) * 32u, _MM_HINT_T0);
                    _mm_prefetch(pf + 4608u + (size_t) (wr0 + s0) * 32u, _MM_HINT_T0);
                }
                const uint8_t* d2_base = kt;
                const uint8_t* qs0_base = kt + 512u;
                const uint8_t* qs1_base = kt + 4608u;
                const int ib0 = 2 * kb, ib1 = ib0 + 1;
                __m256i aq0[NT], aq1[NT];
                __m256 dy0[NT], dy1[NT];
                for (int t = 0; t < NT; ++t) {
                    aq0[t] = _mm256_loadu_si256((const __m256i*) yt[t][ib0].qs);
                    aq1[t] = _mm256_loadu_si256((const __m256i*) yt[t][ib1].qs);
                    dy0[t] = _mm256_set1_ps(dy_pre[t][ib0]);
                    dy1[t] = _mm256_set1_ps(dy_pre[t][ib1]);
                }
                for (int i = 0; i < len; ++i) {
                    const int wd = wr0 + s0 + i;
                    const uint8_t* d2 = d2_base + (size_t) wd * 4u;
                    const __m256 xd0 = _mm256_set1_ps(half_at(d2, 0));
                    const __m256 xd1 = _mm256_set1_ps(half_at(d2, 1));
                    const __m256i qx0 = _mm256_loadu_si256((const __m256i*) (qs0_base + (size_t) wd * 32u));
                    const __m256i ax0 = _mm256_sign_epi8(qx0, qx0);
                    const __m256i qx1 = _mm256_loadu_si256((const __m256i*) (qs1_base + (size_t) wd * 32u));
                    const __m256i ax1 = _mm256_sign_epi8(qx1, qx1);
                    for (int t = 0; t < NT; ++t) {
                        const __m256 d0 = _mm256_mul_ps(xd0, dy0[t]);
                        const __m256 q0 = mul_sum_us8_pairs_float(ax0, _mm256_sign_epi8(aq0[t], qx0));
                        acc[i][t] = _mm256_fmadd_ps(d0, q0, acc[i][t]);
                        const __m256 d1 = _mm256_mul_ps(xd1, dy1[t]);
                        const __m256 q1 = mul_sum_us8_pairs_float(ax1, _mm256_sign_epi8(aq1[t], qx1));
                        acc[i][t] = _mm256_fmadd_ps(d1, q1, acc[i][t]);
                    }
                }
            }
            for (int i = 0; i < len; ++i) {
                const int r = c0 + s0 + i;
                for (int t = 0; t < NT; ++t) out[t][r] = hsum_float_8(acc[i][t]);
            }
        }
        c0 = c1;
    }
}

}  // namespace

void kq256_ktile_gu_rows_situ(const uint8_t* blob, int n, const void* const* act, int nt, float* const* ff,
                              int r0, int r1, float beta, float lb) {
    for (int t0 = 0; t0 < nt; t0 += MAXT) {
        const int k = nt - t0 < MAXT ? nt - t0 : MAXT;
        switch (k) {
            case 1: ktile_q5k_gu_t<1>(blob, n, act + t0, ff + t0, r0, r1, beta, lb); break;
            case 2: ktile_q5k_gu_t<2>(blob, n, act + t0, ff + t0, r0, r1, beta, lb); break;
            case 3: ktile_q5k_gu_t<3>(blob, n, act + t0, ff + t0, r0, r1, beta, lb); break;
            case 4: ktile_q5k_gu_t<4>(blob, n, act + t0, ff + t0, r0, r1, beta, lb); break;
            case 5: ktile_q5k_gu_t<5>(blob, n, act + t0, ff + t0, r0, r1, beta, lb); break;
            case 6: ktile_q5k_gu_t<6>(blob, n, act + t0, ff + t0, r0, r1, beta, lb); break;
            case 7: ktile_q5k_gu_t<7>(blob, n, act + t0, ff + t0, r0, r1, beta, lb); break;
            default: ktile_q5k_gu_t<8>(blob, n, act + t0, ff + t0, r0, r1, beta, lb); break;
        }
    }
}

void kq256_ktile_rows(const uint8_t* w, int n, const void* const* act, int nt, float* const* out,
                      int r0, int r1) {
    for (int t0 = 0; t0 < nt; t0 += MAXT) {
        const int k = nt - t0 < MAXT ? nt - t0 : MAXT;
        switch (k) {
            case 1: ktile_q8_0_down_t<1>(w, n, act + t0, out + t0, r0, r1); break;
            case 2: ktile_q8_0_down_t<2>(w, n, act + t0, out + t0, r0, r1); break;
            case 3: ktile_q8_0_down_t<3>(w, n, act + t0, out + t0, r0, r1); break;
            case 4: ktile_q8_0_down_t<4>(w, n, act + t0, out + t0, r0, r1); break;
            case 5: ktile_q8_0_down_t<5>(w, n, act + t0, out + t0, r0, r1); break;
            case 6: ktile_q8_0_down_t<6>(w, n, act + t0, out + t0, r0, r1); break;
            case 7: ktile_q8_0_down_t<7>(w, n, act + t0, out + t0, r0, r1); break;
            default: ktile_q8_0_down_t<8>(w, n, act + t0, out + t0, r0, r1); break;
        }
    }
}

}  // namespace strata::kernels::cpu
