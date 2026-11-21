// src/inkling2/inkling2_kernels.cuh - CUDA kernels for Inkling-2-Mini (inkling2).
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace strata::inkling2 {

inline constexpr int kThreads = 256;

__device__ __forceinline__ float sigmf(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = lane < (blockDim.x >> 5) ? sh[lane] : 0.0f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) sh[0] = v;
    }
    __syncthreads();
    return sh[0];
}

__global__ void k_f32_to_bf16(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t u;
    memcpy(&u, x + i, sizeof u);
    const uint32_t bias = 0x7FFFu + ((u >> 16) & 1u);
    y[i] = (uint16_t) ((u + bias) >> 16);
}

__global__ void k_f32_to_f16(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    __half h = __float2half_rn(x[i]);
    uint16_t u;
    memcpy(&u, &h, sizeof u);
    y[i] = u;
}

__global__ void k_bf16_to_f32(const uint16_t* __restrict__ x, float* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t u = (uint32_t) x[i] << 16;
    float f;
    memcpy(&f, &u, sizeof f);
    y[i] = f;
}

__global__ void k_add(float* __restrict__ x, const float* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

__global__ void k_rms_norm(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y,
                           int n, float eps) {
    __shared__ float sh[32];
    const int t = blockIdx.x;
    const float* xr = x + (size_t) t * n;
    float* yr = y + (size_t) t * n;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i];
        s += v * v;
    }
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        yr[i] = xr[i] * r * (w ? w[i] : 1.0f);
    }
}

struct BlkPtrs {
    const float* p[8];
};

struct MixOuts {
    float* p[3];
};

/// Normalize D source vectors (blocks[0..D-2], then cur) for T tokens into contiguous stack_norm[T][D][N].
__global__ void k_stack_rms_norm(BlkPtrs ptrs, const float* __restrict__ cur,
                                 float* __restrict__ out, int D, int N, int T, float eps) {
    __shared__ float sh[32];
    const int d = blockIdx.x, t = blockIdx.y;
    const float* src = (d == D - 1) ? (cur + (size_t) t * N) : (ptrs.p[d] + (size_t) t * N);
    float* dst = out + ((size_t) t * D + d) * N;
    float s = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        const float v = src[i];
        s += v * v;
    }
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) N + eps);
    for (int i = threadIdx.x; i < N; i += blockDim.x) dst[i] = src[i] * r;
}

/// mix() across depth stack: out[q][t][N] = sum_d softmax(dot(stack_norm[t][d], query[q]) / sqrt(N))[d] * stack_norm[t][d]
/// grid (n_q, T), block 256. Max depth D <= 8.
__global__ void k_mix(const float* __restrict__ stack_norm, const float* __restrict__ query, MixOuts out,
                      int D, int N, int n_q, float inv_sqrt_n, int swa_broadcast) {
    __shared__ float sh[32];
    __shared__ float w[16];
    const int q = blockIdx.x, t = blockIdx.y;
    const float* qv = query + (size_t) q * N;
    float mx = -1e30f;
    for (int d = 0; d < D; ++d) {
        const float* sv = stack_norm + ((size_t) t * D + d) * N;
        float dot = 0.0f;
        for (int i = threadIdx.x; i < N; i += blockDim.x) dot += sv[i] * qv[i];
        dot = block_sum(dot, sh) * inv_sqrt_n;
        if (threadIdx.x == 0) {
            w[d] = dot;
            if (dot > mx) mx = dot;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float m = -1e30f;
        for (int d = 0; d < D; ++d) if (w[d] > m) m = w[d];
        float sum = 0.0f;
        for (int d = 0; d < D; ++d) {
            w[d] = expf(w[d] - m);
            sum += w[d];
        }
        const float inv = 1.0f / sum;
        for (int d = 0; d < D; ++d) w[d] *= inv;
    }
    __syncthreads();
    if (swa_broadcast) {
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            float acc = 0.0f;
            for (int d = 0; d < D; ++d) acc += w[d] * stack_norm[((size_t) t * D + d) * N + i];
            out.p[0][(size_t) t * N + i] = acc;
            out.p[1][(size_t) t * N + i] = acc;
            out.p[2][(size_t) t * N + i] = acc;
        }
    } else {
        float* dst = out.p[q] + (size_t) t * N;
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            float acc = 0.0f;
            for (int d = 0; d < D; ++d) acc += w[d] * stack_norm[((size_t) t * D + d) * N + i];
            dst[i] = acc;
        }
    }
}

/// control prep: out2N[t][0..N-1] = rms_norm(cur[t]), out2N[t][N..2N-1] = rms_norm(msg[t]). grid(2, T), block 256.
__global__ void k_control_prep(const float* __restrict__ cur, const float* __restrict__ msg,
                               float* __restrict__ out2N, int N, float eps) {
    __shared__ float sh[32];
    const int k = blockIdx.x, t = blockIdx.y;
    const float* src = (k == 0 ? cur : msg) + (size_t) t * N;
    float* dst = out2N + (size_t) t * (2 * N) + (k * N);
    float s = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        const float v = src[i];
        s += v * v;
    }
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) N + eps);
    for (int i = threadIdx.x; i < N; i += blockDim.x) dst[i] = src[i] * r;
}

__global__ void k_control_combine(const float* __restrict__ cur, const float* __restrict__ msg,
                                  const float* __restrict__ gain, const float* __restrict__ forget,
                                  float* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = tanhf(gain[i]);
    const float f = tanhf(forget[i]);
    out[i] = (1.0f - f) * cur[i] + g * msg[i];
}

/// Causal 4-tap shortconv (+ optional dynamic per-head taps for full-attn K/V).
/// hist has [hist_stride, 3] floats? Wait: hist is [n_channels_total][3] for coalesced / per-channel state:
/// specifically hist[channel_offset + c][0..2] holds the 3 oldest->newest prior inputs.
/// Dynamic weight `dyn` is [T][n_heads][4] where lag 0 = current token, lag 3 = t-3.
/// Static weight `w` is [n_ch][4] where w[0] = t-3, w[3] = current token.
__global__ void k_shortconv(const float* __restrict__ cur, const float* __restrict__ w,
                            const float* __restrict__ dyn, float* __restrict__ hist,
                            float* __restrict__ out, int n_ch, int hist_offset, int n_heads, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_ch) return;
    float* hc = hist + (size_t) (hist_offset + c) * 3;
    float h0 = hc[0], h1 = hc[1], h2 = hc[2];
    const float* wc = w + (size_t) c * 4;
    const float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    const int head_dim = n_heads > 0 ? (n_ch / n_heads) : 0;
    const int h_idx = n_heads > 0 ? (c / head_dim) : 0;
    const float scale = rsqrtf(dyn ? 9.0f : 5.0f);
    for (int t = 0; t < T; ++t) {
        const float x = cur[(size_t) t * n_ch + c];
        float s = h0 * w0 + h1 * w1 + h2 * w2 + x * w3;
        if (dyn) {
            const float* dt = dyn + ((size_t) t * n_heads + h_idx) * 4;
            s += x * dt[0] + h2 * dt[1] + h1 * dt[2] + h0 * dt[3];
        }
        out[(size_t) t * n_ch + c] = (x + s) * scale;
        h0 = h1;
        h1 = h2;
        h2 = x;
    }
    hc[0] = h0;
    hc[1] = h1;
    hc[2] = h2;
}

/// clip(x, scale, thr) per [T][n_heads][D]. Also optionally multiplies by per-token tau if alpha > 0.
/// grid (n_heads, T), block min(128, D) or 32 (one warp when D <= 128).
__global__ void k_clip_and_tau(float* __restrict__ x, const float* __restrict__ scale, float thr, int D, int n_heads,
                               int apply_tau, const int32_t* __restrict__ pos0, int floor_n, float alpha) {
    const int h = blockIdx.x, t = blockIdx.y;
    const int lane = threadIdx.x & 31;
    float* v = x + ((size_t) t * n_heads + h) * D;
    float sq = 0.0f;
    float vals[4];
    const int iters = (D + 31) / 32;   // D = 128 -> 4, D = 16 -> 1
    for (int k = 0; k < iters; ++k) {
        const int i = k * 32 + lane;
        float val = 0.0f;
        if (i < D) {
            val = v[i];
            if (scale) val *= scale[(size_t) h * D + i];
        }
        vals[k] = val;
        sq += val * val;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
    const float norm = sqrtf(sq) + 1e-6f;
    float factor = fminf(1.0f, thr / norm);
    if (apply_tau) {
        const int pos = pos0[0] + t;
        const float ratio = fmaxf(1.0f, (float) (pos + 1) / (float) floor_n);
        factor *= (1.0f + alpha * logf(ratio));
    }
    for (int k = 0; k < iters; ++k) {
        const int i = k * 32 + lane;
        if (i < D) v[i] = vals[k] * factor;
    }
}

/// Per-head RMSNorm on Vcur [T][n_heads_kv][head_dim] (no weight). grid (n_heads_kv, T), block 32.
__global__ void k_v_rms_norm(float* __restrict__ v, int D, int n_heads, float eps) {
    const int h = blockIdx.x, t = blockIdx.y, lane = threadIdx.x & 31;
    float* vr = v + ((size_t) t * n_heads + h) * D;
    float sq = 0.0f;
    float vals[4];
    for (int k = 0; k < 4; ++k) {
        const int i = k * 32 + lane;
        const float val = i < D ? vr[i] : 0.0f;
        vals[k] = val;
        sq += val * val;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
    const float r = rsqrtf(sq / (float) D + eps);
    for (int k = 0; k < 4; ++k) {
        const int i = k * 32 + lane;
        if (i < D) vr[i] = vals[k] * r;
    }
}

/// Store K, V [T][kv_dim] into FP16 or Q8_0 cache [cap][kv_dim].
/// When q8_kv is enabled: kc_q8 stores [cap][n_kv_heads][128] int8_t, kc_s stores [cap][n_kv_heads] __half scale (1 scale per 128-dim head).
__global__ void k_kv_store(const float* __restrict__ k, const float* __restrict__ v,
                           __half* __restrict__ kc, __half* __restrict__ vc,
                           int kv_dim, const int32_t* __restrict__ pos0, int cap, int is_swa) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (i >= kv_dim) return;
    const int pos = pos0[0] + t;
    const int slot = is_swa ? (pos % cap) : pos;
    kc[(size_t) slot * kv_dim + i] = __float2half(k[(size_t) t * kv_dim + i]);
    vc[(size_t) slot * kv_dim + i] = __float2half(v[(size_t) t * kv_dim + i]);
}

/// Q8 per-head (128-dim) KV store: grid (n_head_kv, T), block 32 (1 warp per head).
__global__ void k_kv_store_q8(const float* __restrict__ k, const float* __restrict__ v,
                              int8_t* __restrict__ kq, __half* __restrict__ ks,
                              int8_t* __restrict__ vq, __half* __restrict__ vs,
                              int n_head_kv, const int32_t* __restrict__ pos0, int cap, int is_swa) {
    const int hkv = blockIdx.x, t = blockIdx.y, lane = threadIdx.x & 31;
    const int kv_dim = n_head_kv * 128;
    const float* kp = k + (size_t) t * kv_dim + hkv * 128;
    const float* vp = v + (size_t) t * kv_dim + hkv * 128;
    float kv[4], vv[4], ak = 0.0f, av = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        kv[i] = kp[i * 32 + lane];
        vv[i] = vp[i * 32 + lane];
        ak = fmaxf(ak, fabsf(kv[i]));
        av = fmaxf(av, fabsf(vv[i]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        ak = fmaxf(ak, __shfl_xor_sync(0xffffffffu, ak, o));
        av = fmaxf(av, __shfl_xor_sync(0xffffffffu, av, o));
    }
    const float dk = ak / 127.0f, dv = av / 127.0f;
    const float ik = dk > 0.0f ? (127.0f / ak) : 0.0f;
    const float iv = dv > 0.0f ? (127.0f / av) : 0.0f;
    const int pos = pos0[0] + t;
    const int slot = is_swa ? (pos % cap) : pos;
    int8_t* kdst = kq + (size_t) slot * kv_dim + hkv * 128;
    int8_t* vdst = vq + (size_t) slot * kv_dim + hkv * 128;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        kdst[i * 32 + lane] = (int8_t) __float2int_rn(kv[i] * ik);
        vdst[i * 32 + lane] = (int8_t) __float2int_rn(vv[i] * iv);
    }
    if (lane == 0) {
        ks[(size_t) slot * n_head_kv + hkv] = __float2half(dk);
        vs[(size_t) slot * n_head_kv + hkv] = __float2half(dv);
    }
}

/// Compute relative bias distance table minus sink: table[t][h][d] = dot(R[t,h], rel_bias[d]) - sink[t,h]
/// R is [T][n_head][n_rel=16], rel_bias is [n_ext=1024][16] (F32 contiguous along n_rel=16), sink is [T][n_head].
/// Reads *pos0 on device so grid is CUDA-Graph invariant. grid (n_head, T), block 256.
__global__ void k_rel_table(const float* __restrict__ R, const float* __restrict__ rel_bias,
                            const float* __restrict__ sink, float* __restrict__ table,
                            int n_head, int n_rel, int n_ext, const int32_t* __restrict__ pos0, int T) {
    __shared__ float sr[16];
    const int h = blockIdx.x, t = blockIdx.y;
    const int max_d = min(n_ext, pos0[0] + T);
    const float* rr = R + ((size_t) t * n_head + h) * n_rel;
    if (threadIdx.x < n_rel) sr[threadIdx.x] = rr[threadIdx.x];
    __syncthreads();
    const float sk = sink[(size_t) t * n_head + h];
    float* out = table + ((size_t) t * n_head + h) * n_ext;
    for (int d = threadIdx.x; d < max_d; d += blockDim.x) {
        const float* rb = rel_bias + (size_t) d * n_rel;
        float dot = 0.0f;
#pragma unroll
        for (int r = 0; r < 16; ++r) dot += sr[r] * rb[r];
        out[d] = dot - sk;
    }
}

__global__ void k_attn_q8(const float* __restrict__ Q,
                          const int8_t* __restrict__ kq, const __half* __restrict__ ks,
                          const int8_t* __restrict__ vq, const __half* __restrict__ vs,
                          const float* __restrict__ rel_tab, const float* __restrict__ sink, float* __restrict__ out,
                          int n_head, int n_head_kv, int n_ext, int cap, int is_swa, int n_swa,
                          const int32_t* __restrict__ pos0, float kq_scale) {
    __shared__ float sq[128];
    __shared__ float s_m[4];
    __shared__ float s_l[4];
    __shared__ float s_acc[4][128];
    const int h = blockIdx.x, t = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int hkv = h / (n_head / n_head_kv);
    const int kv_dim = n_head_kv * 128;
    const float* qv = Q + ((size_t) t * n_head + h) * 128;
    if (threadIdx.x < 128) sq[threadIdx.x] = qv[threadIdx.x];
    __syncthreads();

    const int pos_q = pos0[0] + t;
    const int j0 = is_swa ? max(0, pos_q - n_swa + 1) : 0;
    const int j1 = pos_q + 1;
    const float* rt = rel_tab + ((size_t) t * n_head + h) * n_ext;
    const float neg_sink = -sink[(size_t) t * n_head + h];

    float m = -1e30f;
    float l = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    if (warp == 0) {
        m = 0.0f;
        l = 1.0f;
    }
    for (int j = j0 + warp; j < j1; j += 4) {
        const int slot = is_swa ? (j % cap) : j;
        const size_t idx = (size_t) slot * n_head_kv + hkv;
        const float d_k = __half2float(ks[idx]);
        const float d_v = __half2float(vs[idx]);
        const int8_t* kj = kq + (size_t) slot * kv_dim + hkv * 128;
        const int8_t* vj = vq + (size_t) slot * kv_dim + hkv * 128;
        float dot = 0.0f;
        dot += sq[lane] * (float) kj[lane];
        dot += sq[32 + lane] * (float) kj[32 + lane];
        dot += sq[64 + lane] * (float) kj[64 + lane];
        dot += sq[96 + lane] * (float) kj[96 + lane];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
        const int dist = pos_q - j;
        const float bias = dist < n_ext ? rt[dist] : neg_sink;
        const float sc = dot * d_k * kq_scale + bias;
        const float m_new = fmaxf(m, sc);
        const float alpha = l > 0.0f ? expf(m - m_new) : 0.0f;
        const float p = expf(sc - m_new);
        const float pv = p * d_v;
        l = l * alpha + p;
        acc0 = acc0 * alpha + pv * (float) vj[lane];
        acc1 = acc1 * alpha + pv * (float) vj[32 + lane];
        acc2 = acc2 * alpha + pv * (float) vj[64 + lane];
        acc3 = acc3 * alpha + pv * (float) vj[96 + lane];
        m = m_new;
    }
    if (lane == 0) {
        s_m[warp] = m;
        s_l[warp] = l;
    }
    s_acc[warp][lane] = acc0;
    s_acc[warp][32 + lane] = acc1;
    s_acc[warp][64 + lane] = acc2;
    s_acc[warp][96 + lane] = acc3;
    __syncthreads();
    if (threadIdx.x < 128) {
        float mg = s_m[0];
        for (int w = 1; w < 4; ++w) if (s_l[w] > 0.0f && s_m[w] > mg) mg = s_m[w];
        float lg = 0.0f;
        float og = 0.0f;
        for (int w = 0; w < 4; ++w) {
            if (s_l[w] > 0.0f) {
                const float c = expf(s_m[w] - mg);
                lg += s_l[w] * c;
                og += s_acc[w][threadIdx.x] * c;
            }
        }
        out[((size_t) t * n_head + h) * 128 + threadIdx.x] = og / lg;
    }
}

/// Decode / single-token-group attention with relative-position bias and zero sink (1 warp per head, single pass for short ctx, or general loop over valid KV).
/// Supports arbitrary T tokens at pos = *pos0 + t, head_dim = 128, n_head = 24, n_head_kv = 8.
/// grid (n_head, T), block 128 (4 warps split KV keys, then reduce across warps in shared mem!).
__global__ void k_attn(const float* __restrict__ Q, const __half* __restrict__ kc, const __half* __restrict__ vc,
                       const float* __restrict__ rel_tab, const float* __restrict__ sink, float* __restrict__ out,
                       int n_head, int n_head_kv, int n_ext, int cap, int is_swa, int n_swa,
                       const int32_t* __restrict__ pos0, float kq_scale) {
    __shared__ float sq[128];
    __shared__ float s_m[4];
    __shared__ float s_l[4];
    __shared__ float s_acc[4][128];
    const int h = blockIdx.x, t = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int hkv = h / (n_head / n_head_kv);
    const int kv_dim = n_head_kv * 128;
    const float* qv = Q + ((size_t) t * n_head + h) * 128;
    if (threadIdx.x < 128) sq[threadIdx.x] = qv[threadIdx.x];
    __syncthreads();

    const int pos_q = pos0[0] + t;
    const int j0 = is_swa ? max(0, pos_q - n_swa + 1) : 0;
    const int j1 = pos_q + 1;   // inclusive pos_q
    const float* rt = rel_tab + ((size_t) t * n_head + h) * n_ext;
    const float neg_sink = -sink[(size_t) t * n_head + h];

    float m = -1e30f;
    float l = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    if (warp == 0) {
        m = 0.0f;
        l = 1.0f;   // zero sink = exp(0)
    }
    for (int j = j0 + warp; j < j1; j += 4) {
        const int slot = is_swa ? (j % cap) : j;
        const __half* kj = kc + (size_t) slot * kv_dim + hkv * 128;
        const __half* vj = vc + (size_t) slot * kv_dim + hkv * 128;
        float dot = 0.0f;
        dot += sq[lane] * __half2float(kj[lane]);
        dot += sq[32 + lane] * __half2float(kj[32 + lane]);
        dot += sq[64 + lane] * __half2float(kj[64 + lane]);
        dot += sq[96 + lane] * __half2float(kj[96 + lane]);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
        const int dist = pos_q - j;
        const float bias = dist < n_ext ? rt[dist] : neg_sink;
        const float sc = dot * kq_scale + bias;
        const float m_new = fmaxf(m, sc);
        const float alpha = l > 0.0f ? expf(m - m_new) : 0.0f;
        const float p = expf(sc - m_new);
        l = l * alpha + p;
        acc0 = acc0 * alpha + p * __half2float(vj[lane]);
        acc1 = acc1 * alpha + p * __half2float(vj[32 + lane]);
        acc2 = acc2 * alpha + p * __half2float(vj[64 + lane]);
        acc3 = acc3 * alpha + p * __half2float(vj[96 + lane]);
        m = m_new;
    }
    if (lane == 0) {
        s_m[warp] = m;
        s_l[warp] = l;
    }
    s_acc[warp][lane] = acc0;
    s_acc[warp][32 + lane] = acc1;
    s_acc[warp][64 + lane] = acc2;
    s_acc[warp][96 + lane] = acc3;
    __syncthreads();
    if (threadIdx.x < 128) {
        float mg = s_m[0];
        for (int w = 1; w < 4; ++w) if (s_l[w] > 0.0f && s_m[w] > mg) mg = s_m[w];
        float lg = 0.0f;
        float og = 0.0f;
        for (int w = 0; w < 4; ++w) {
            if (s_l[w] > 0.0f) {
                const float c = expf(s_m[w] - mg);
                lg += s_l[w] * c;
                og += s_acc[w][threadIdx.x] * c;
            }
        }
        out[((size_t) t * n_head + h) * 128 + threadIdx.x] = og / lg;
    }
}

__global__ void k_mul_sigmoid(float* __restrict__ att, const float* __restrict__ gate, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) att[i] *= sigmf(gate[i]);
}

/// MoE router for Inkling2: sigmoid(logits) + exp_probs_b -> top-k -> normalize weights -> * scale.
/// n_exp = 128, top_k = 8. One block of 128 threads per token (grid T).
__global__ void k_route_inkling2(const float* __restrict__ logits, const float* __restrict__ bias,
                                 int32_t* __restrict__ ids, float* __restrict__ ewts,
                                 int n_exp, int top_k, float scale) {
    __shared__ float s_prob[128];
    __shared__ float s_sel[128];
    const int t = blockIdx.x;
    const int i = threadIdx.x;
    if (i < n_exp) {
        const float p = sigmf(logits[(size_t) t * n_exp + i]);
        s_prob[i] = p;
        s_sel[i] = p + (bias ? bias[i] : 0.0f);
    }
    __syncthreads();
    if (i == 0) {
        float wsum = 0.0f;
        float wlocal[16];
        for (int k = 0; k < top_k; ++k) {
            int best = 0;
            float bestv = -1e30f;
            for (int e = 0; e < n_exp; ++e) {
                if (s_sel[e] > bestv) {
                    bestv = s_sel[e];
                    best = e;
                }
            }
            s_sel[best] = -1e30f;
            ids[(size_t) t * top_k + k] = best;
            const float wk = s_prob[best];
            wlocal[k] = wk;
            wsum += wk;
        }
        const float denom = fmaxf(wsum, 6.103515625e-5f);
        const float fac = scale / denom;
        for (int k = 0; k < top_k; ++k) ewts[(size_t) t * top_k + k] = wlocal[k] * fac;
    }
}

__global__ void k_combine_experts(const float* __restrict__ eparts, const float* __restrict__ ewts,
                                  float* __restrict__ out, int N, int top_k) {
    const int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = 0.0f;
    const size_t base = (size_t) t * top_k;
    for (int k = 0; k < top_k; ++k) s += ewts[base + k] * eparts[(base + k) * N + i];
    out[(size_t) t * N + i] = s;
}

__global__ void k_merge_cpu(float* __restrict__ eparts, const float* __restrict__ ecpu,
                            const int32_t* __restrict__ miss, int N) {
    const int e = blockIdx.x;
    if (!miss[e]) return;
    for (int i = threadIdx.x; i < N; i += blockDim.x) eparts[(size_t) e * N + i] = ecpu[(size_t) e * N + i];
}

__global__ void k_group_experts(const int32_t* __restrict__ ids, const int32_t* __restrict__ slot_of,
                                const uint8_t* __restrict__ blobs, size_t stride,
                                int n_ent, int n_exp, int n_slots,
                                unsigned long long* __restrict__ grp_ptr, int32_t* __restrict__ grp_start,
                                int32_t* __restrict__ n_groups, int32_t* __restrict__ ent_dst,
                                int32_t* __restrict__ ent_tok, int top_k) {
    // 1 block of 128 threads over all n_ent <= 512*8? For T <= 512, n_ent <= 4096.
    __shared__ int32_t cnt[128];
    __shared__ int32_t slot_id[128];
    __shared__ int32_t cur_off[128];
    if (threadIdx.x < n_exp) {
        cnt[threadIdx.x] = 0;
        slot_id[threadIdx.x] = n_slots > 0 ? slot_of[threadIdx.x] : -1;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x) {
        const int x = ids[e];
        if (slot_id[x] >= 0) atomicAdd(&cnt[x], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int ng = 0;
        int off = 0;
        for (int x = 0; x < n_exp; ++x) {
            if (cnt[x] > 0) {
                grp_ptr[ng] = (unsigned long long) (blobs + (size_t) slot_id[x] * stride);
                grp_start[ng] = off;
                cur_off[x] = off;
                off += cnt[x];
                ++ng;
            }
        }
        grp_start[ng] = off;
        *n_groups = ng;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x) {
        const int x = ids[e];
        if (slot_id[x] >= 0) {
            const int pos = atomicAdd(&cur_off[x], 1);
            ent_dst[pos] = e;
            ent_tok[pos] = e / top_k;
        }
    }
}

__global__ void k_softcap(float* __restrict__ logits, int64_t n, float cap) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && cap > 0.0f) logits[i] = cap * tanhf(logits[i] / cap);
}

__global__ void k_gemv_f32(const float* __restrict__ W, const float* __restrict__ x, float* __restrict__ y,
                           int n_in, int n_out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int o = blockIdx.x * 8 + warp;
    if (o >= n_out) return;
    const float4* wr = (const float4*) (W + (size_t) o * n_in);
    const float4* xr = (const float4*) x;
    float acc = 0.0f;
    for (int i = lane; i < n_in / 4; i += 32) {
        const float4 a = wr[i], b = xr[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane == 0) y[o] = acc;
}

__global__ void k_argmax(const float* __restrict__ logits, int32_t* __restrict__ pick, int V) {
    __shared__ float sv[256];
    __shared__ int32_t si[256];
    const float* row = logits + (size_t) blockIdx.x * V;
    float mv = -1e30f;
    int32_t mi = 0;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        const float v = row[i];
        if (v > mv) { mv = v; mi = i; }
    }
    sv[threadIdx.x] = mv;
    si[threadIdx.x] = mi;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sv[threadIdx.x + s] > sv[threadIdx.x] ||
                (sv[threadIdx.x + s] == sv[threadIdx.x] && si[threadIdx.x + s] < si[threadIdx.x])) {
                sv[threadIdx.x] = sv[threadIdx.x + s];
                si[threadIdx.x] = si[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) pick[blockIdx.x] = si[0];
}

}  // namespace strata::inkling2
