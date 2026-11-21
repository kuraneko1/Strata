// src/inkling2/inkling2_kernels.cuh - CUDA kernels for Inkling-2-Mini (inkling2).
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace strata::inkling2 {

inline constexpr int kThreads = 256;

__device__ __forceinline__ float sigmf(float x) { return __fdividef(1.0f, 1.0f + __expf(-x)); }
__device__ __forceinline__ float fast_tanh(float x) {
    const float xc = fmaxf(-9.0f, fminf(9.0f, x));
    return __fdividef(2.0f, 1.0f + __expf(-2.0f * xc)) - 1.0f;
}

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
    if ((n & 3) == 0) {
        const int n4 = n >> 2;
        const float4* x4 = (const float4*) xr;
        float4* y4 = (float4*) yr;
        for (int i = threadIdx.x; i < n4; i += blockDim.x) {
            const float4 v = x4[i];
            s += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        s = block_sum(s, sh);
        const float r = rsqrtf(s / (float) n + eps);
        const float4* w4 = w ? (const float4*) w : nullptr;
        for (int i = threadIdx.x; i < n4; i += blockDim.x) {
            const float4 v = x4[i];
            if (w4) {
                const float4 ww = w4[i];
                y4[i] = make_float4(v.x * r * ww.x, v.y * r * ww.y, v.z * r * ww.z, v.w * r * ww.w);
            } else {
                y4[i] = make_float4(v.x * r, v.y * r, v.z * r, v.w * r);
            }
        }
        return;
    }
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

/// Normalize 3 independent vectors (q, k, v) with shared weight w in a single grid(3, T) launch.
__global__ void k_rms_norm_3(float* __restrict__ q, float* __restrict__ k, float* __restrict__ v,
                             const float* __restrict__ w, int n, float eps) {
    __shared__ float sh[32];
    const int which = blockIdx.x, t = blockIdx.y;
    float* ptr = (which == 0 ? q : (which == 1 ? k : v)) + (size_t) t * n;
    float s = 0.0f;
    if ((n & 3) == 0) {
        const int n4 = n >> 2;
        float4* p4 = (float4*) ptr;
        const float4* w4 = w ? (const float4*) w : nullptr;
        for (int i = threadIdx.x; i < n4; i += blockDim.x) {
            const float4 x = p4[i];
            s += x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
        }
        s = block_sum(s, sh);
        const float r = rsqrtf(s / (float) n + eps);
        for (int i = threadIdx.x; i < n4; i += blockDim.x) {
            const float4 x = p4[i];
            if (w4) {
                const float4 ww = w4[i];
                p4[i] = make_float4(x.x * r * ww.x, x.y * r * ww.y, x.z * r * ww.z, x.w * r * ww.w);
            } else {
                p4[i] = make_float4(x.x * r, x.y * r, x.z * r, x.w * r);
            }
        }
        return;
    }
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float x = ptr[i];
        s += x * x;
    }
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        ptr[i] = ptr[i] * r * (w ? w[i] : 1.0f);
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
            w[d] = __expf(w[d] - m);
            sum += w[d];
        }
        const float inv = __fdividef(1.0f, sum);
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

/// Mix across D already-RMSNormalized pointers (norm_ptrs[0..D-1]): out[q][t][N] = sum_d softmax(dot(v[d], query[q]) / sqrt(N))[d] * v[d].
/// Eliminates redundant stack_norm materialization when block residuals are cached normalized. grid (n_q, T), block 256.
struct NormMixPtrs {
    const float* p[8];
};

__global__ void k_mix_normed(NormMixPtrs ptrs, int D, const float* __restrict__ query, MixOuts out,
                             int n_q, int N, float inv_sqrt_n, int swa_broadcast) {
    __shared__ float sh[32];
    __shared__ float w[16];
    const int q = blockIdx.x, t = blockIdx.y;
    const float* qv = query + (size_t) q * N;
    const int n4 = N >> 2;
    const float4* q4 = (const float4*) qv;
    for (int d = 0; d < D; ++d) {
        const float* sv = ptrs.p[d] + (size_t) t * N;
        float dot = 0.0f;
        if ((N & 3) == 0) {
            const float4* s4 = (const float4*) sv;
            for (int i = threadIdx.x; i < n4; i += blockDim.x) {
                const float4 a = s4[i], b = q4[i];
                dot += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            }
        } else {
            for (int i = threadIdx.x; i < N; i += blockDim.x) dot += sv[i] * qv[i];
        }
        dot = block_sum(dot, sh) * inv_sqrt_n;
        if (threadIdx.x == 0) w[d] = dot;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float m = -1e30f;
        for (int d = 0; d < D; ++d) if (w[d] > m) m = w[d];
        float sum = 0.0f;
        for (int d = 0; d < D; ++d) {
            w[d] = __expf(w[d] - m);
            sum += w[d];
        }
        const float inv = __fdividef(1.0f, sum);
        for (int d = 0; d < D; ++d) w[d] *= inv;
    }
    __syncthreads();
    if ((N & 3) == 0) {
        if (swa_broadcast) {
            float4* o0 = (float4*) (out.p[0] + (size_t) t * N);
            float4* o1 = (float4*) (out.p[1] + (size_t) t * N);
            float4* o2 = (float4*) (out.p[2] + (size_t) t * N);
            for (int i = threadIdx.x; i < n4; i += blockDim.x) {
                float ax = 0.0f, ay = 0.0f, az = 0.0f, aw = 0.0f;
                for (int d = 0; d < D; ++d) {
                    const float4 v = ((const float4*) (ptrs.p[d] + (size_t) t * N))[i];
                    const float wd = w[d];
                    ax += wd * v.x; ay += wd * v.y; az += wd * v.z; aw += wd * v.w;
                }
                const float4 r = make_float4(ax, ay, az, aw);
                o0[i] = r; o1[i] = r; o2[i] = r;
            }
        } else {
            float4* dst = (float4*) (out.p[q] + (size_t) t * N);
            for (int i = threadIdx.x; i < n4; i += blockDim.x) {
                float ax = 0.0f, ay = 0.0f, az = 0.0f, aw = 0.0f;
                for (int d = 0; d < D; ++d) {
                    const float4 v = ((const float4*) (ptrs.p[d] + (size_t) t * N))[i];
                    const float wd = w[d];
                    ax += wd * v.x; ay += wd * v.y; az += wd * v.z; aw += wd * v.w;
                }
                dst[i] = make_float4(ax, ay, az, aw);
            }
        }
        return;
    }
    if (swa_broadcast) {
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            float acc = 0.0f;
            for (int d = 0; d < D; ++d) acc += w[d] * ptrs.p[d][(size_t) t * N + i];
            out.p[0][(size_t) t * N + i] = acc;
            out.p[1][(size_t) t * N + i] = acc;
            out.p[2][(size_t) t * N + i] = acc;
        }
    } else {
        float* dst = out.p[q] + (size_t) t * N;
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            float acc = 0.0f;
            for (int d = 0; d < D; ++d) acc += w[d] * ptrs.p[d][(size_t) t * N + i];
            dst[i] = acc;
        }
    }
}

/// Fused 4-query mix for loop bank + bank_last (`q_attn` + `q_kv` in 1 grid(4, T) launch).
struct MixOuts4 {
    float* p[4];
};

__global__ void k_mix_normed_4(NormMixPtrs ptrs, int D, const float* __restrict__ q0, const float* __restrict__ q1,
                               MixOuts4 out, int N, float inv_sqrt_n) {
    __shared__ float sh[32];
    __shared__ float w[16];
    const int q = blockIdx.x, t = blockIdx.y;
    const float* qv = (q < 2 ? (q0 + (size_t) q * N) : (q1 + (size_t) (q - 2) * N));
    const int n4 = N >> 2;
    const float4* q4 = (const float4*) qv;
    for (int d = 0; d < D; ++d) {
        const float* sv = ptrs.p[d] + (size_t) t * N;
        float dot = 0.0f;
        if ((N & 3) == 0) {
            const float4* s4 = (const float4*) sv;
            for (int i = threadIdx.x; i < n4; i += blockDim.x) {
                const float4 a = s4[i], b = q4[i];
                dot += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            }
        } else {
            for (int i = threadIdx.x; i < N; i += blockDim.x) dot += sv[i] * qv[i];
        }
        dot = block_sum(dot, sh) * inv_sqrt_n;
        if (threadIdx.x == 0) w[d] = dot;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float m = -1e30f;
        for (int d = 0; d < D; ++d) if (w[d] > m) m = w[d];
        float sum = 0.0f;
        for (int d = 0; d < D; ++d) {
            w[d] = __expf(w[d] - m);
            sum += w[d];
        }
        const float inv = __fdividef(1.0f, sum);
        for (int d = 0; d < D; ++d) w[d] *= inv;
    }
    __syncthreads();
    if ((N & 3) == 0) {
        float4* dst = (float4*) (out.p[q] + (size_t) t * N);
        for (int i = threadIdx.x; i < n4; i += blockDim.x) {
            float ax = 0.0f, ay = 0.0f, az = 0.0f, aw = 0.0f;
            for (int d = 0; d < D; ++d) {
                const float4 v = ((const float4*) (ptrs.p[d] + (size_t) t * N))[i];
                const float wd = w[d];
                ax += wd * v.x; ay += wd * v.y; az += wd * v.z; aw += wd * v.w;
            }
            dst[i] = make_float4(ax, ay, az, aw);
        }
        return;
    }
    float* dst = out.p[q] + (size_t) t * N;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float acc = 0.0f;
        for (int d = 0; d < D; ++d) acc += w[d] * ptrs.p[d][(size_t) t * N + i];
        dst[i] = acc;
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
    if ((N & 3) == 0) {
        const int n4 = N >> 2;
        const float4* s4 = (const float4*) src;
        float4* d4 = (float4*) dst;
        for (int i = threadIdx.x; i < n4; i += blockDim.x) {
            const float4 v = s4[i];
            s += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        s = block_sum(s, sh);
        const float r = rsqrtf(s / (float) N + eps);
        for (int i = threadIdx.x; i < n4; i += blockDim.x) {
            const float4 v = s4[i];
            d4[i] = make_float4(v.x * r, v.y * r, v.z * r, v.w * r);
        }
        return;
    }
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

/// Fused T=1 loop_gain + loop_forget ([R=64 -> N] F32) + k_control_combine: 8 warps/block, 4 rows/warp -> 4x fewer CTAs & barriers.
__global__ void k_control_up_combine_t1(const float* __restrict__ cur, const float* __restrict__ msg,
                                        const float* __restrict__ w_gain, const float* __restrict__ w_forget,
                                        const float* __restrict__ ctrl64, float* __restrict__ out,
                                        int R, int N) {
    __shared__ float sx[64];
    if (threadIdx.x < R) sx[threadIdx.x] = ctrl64[threadIdx.x];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int base_o = (blockIdx.x * 8 + warp) * 4;
    const int half = lane >> 4;
    const int i = lane & 15;
    const float4 x = reinterpret_cast<const float4*>(sx)[i];
    const float* wg = half ? w_forget : w_gain;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int o = base_o + k;
        if (o >= N) return;
        const float4 w = reinterpret_cast<const float4*>(wg + (size_t) o * R)[i];
        float acc = w.x * x.x + w.y * x.y + w.z * x.z + w.w * x.w;
#pragma unroll
        for (int m = 8; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
        const float f_val = __shfl_sync(0xffffffffu, acc, 16);
        if (lane == 0) {
            out[o] = (1.0f - fast_tanh(f_val)) * cur[o] + fast_tanh(acc) * msg[o];
        }
    }
}

/// High-occupancy F32 GEMV for small n_out (1 CTA of 128 threads per output row).
__global__ void k_gemv_f32_cta(const float* __restrict__ W, const float* __restrict__ x, float* __restrict__ y,
                               int n_in, int n_out) {
    __shared__ float sh[32];
    const int o = blockIdx.x;
    if (o >= n_out) return;
    const float4* wr = (const float4*) (W + (size_t) o * n_in);
    const float4* xr = (const float4*) x;
    float acc = 0.0f;
    const int n4 = n_in / 4;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 a = wr[i], b = xr[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    acc = block_sum(acc, sh);
    if (threadIdx.x == 0) y[o] = acc;
}

/// Causal 4-tap shortconv (+ optional dynamic per-head taps for full-attn K/V).
/// hist has [hist_stride, 3] floats? Wait: hist is [n_channels_total][3] for coalesced / per-channel state:
/// specifically hist[channel_offset + c][0..2] holds the 3 oldest->newest prior inputs.
/// Dynamic weight `dyn` is [T][n_heads][4] where lag 0 = current token, lag 3 = t-3.
/// Static weight `w` is [n_ch][4] where w[0] = t-3, w[3] = current token.
__global__ void k_shortconv(const float* __restrict__ cur, const float* __restrict__ w,
                            const float* __restrict__ dyn, float* __restrict__ hist,
                            float* __restrict__ out, int n_ch, int hist_offset, int n_heads, int T,
                            const float* __restrict__ res_add = nullptr) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_ch) return;
    float* hc = hist + (size_t) (hist_offset + c) * 3;
    float h0 = hc[0], h1 = hc[1], h2 = hc[2];
    const float4 w4 = *reinterpret_cast<const float4*>(w + (size_t) c * 4);
    const float w0 = w4.x, w1 = w4.y, w2 = w4.z, w3 = w4.w;
    const int head_dim = n_heads > 0 ? (n_ch / n_heads) : 0;
    const int h_idx = n_heads > 0 ? (c / head_dim) : 0;
    const float scale = rsqrtf(dyn ? 9.0f : 5.0f);
    for (int t = 0; t < T; ++t) {
        const size_t idx = (size_t) t * n_ch + c;
        const float x = cur[idx];
        float s = h0 * w0 + h1 * w1 + h2 * w2 + x * w3;
        if (dyn) {
            const float* dt = dyn + ((size_t) t * n_heads + h_idx) * 4;
            s += x * dt[0] + h2 * dt[1] + h1 * dt[2] + h0 * dt[3];
        }
        float y = (x + s) * scale;
        if (res_add) y += res_add[idx];
        out[idx] = y;
        h0 = h1;
        h1 = h2;
        h2 = x;
    }
    hc[0] = h0;
    hc[1] = h1;
    hc[2] = h2;
}

/// Fused shortconv for K [0..kv_dim-1] and V [kv_dim..2*kv_dim-1]: single launch of 2*kv_dim threads.
__global__ void k_shortconv_kv(const float* __restrict__ K, const float* __restrict__ V,
                               const float* __restrict__ wk, const float* __restrict__ wv,
                               const float* __restrict__ dyn_k, const float* __restrict__ dyn_v,
                               float* __restrict__ hist, float* __restrict__ K_out, float* __restrict__ V_out,
                               int kv_dim, int n_heads, int T) {
    const int c_all = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_all >= 2 * kv_dim) return;
    const bool is_v = (c_all >= kv_dim);
    const int c = is_v ? (c_all - kv_dim) : c_all;
    const float* cur = is_v ? V : K;
    const float* dyn = is_v ? dyn_v : dyn_k;
    float* out = is_v ? V_out : K_out;
    float* hc = hist + (size_t) c_all * 3;
    float h0 = hc[0], h1 = hc[1], h2 = hc[2];
    const float4 w4 = *reinterpret_cast<const float4*>((is_v ? wv : wk) + (size_t) c * 4);
    const float w0 = w4.x, w1 = w4.y, w2 = w4.z, w3 = w4.w;
    const int head_dim = n_heads > 0 ? (kv_dim / n_heads) : 0;
    const int h_idx = n_heads > 0 ? (c / head_dim) : 0;
    const float scale = rsqrtf(dyn ? 9.0f : 5.0f);
    for (int t = 0; t < T; ++t) {
        const size_t idx = (size_t) t * kv_dim + c;
        const float x = cur[idx];
        float s = h0 * w0 + h1 * w1 + h2 * w2 + x * w3;
        if (dyn) {
            const float* dt = dyn + ((size_t) t * n_heads + h_idx) * 4;
            s += x * dt[0] + h2 * dt[1] + h1 * dt[2] + h0 * dt[3];
        }
        out[idx] = (x + s) * scale;
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
    const int slot = is_swa ? (((cap & (cap - 1)) == 0) ? (pos & (cap - 1)) : (pos % cap)) : pos;
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
    const int slot = is_swa ? (((cap & (cap - 1)) == 0) ? (pos & (cap - 1)) : (pos % cap)) : pos;
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

/// Fused clip(K) + rms_norm(V) + Q8 KV store (head_dim == 128): eliminates 2 kernel launches + K_conv/V_conv DRAM roundtrips.
__global__ void k_clip_norm_kv_store_q8(const float* __restrict__ k, const float* __restrict__ v,
                                         int8_t* __restrict__ kq, __half* __restrict__ ks,
                                         int8_t* __restrict__ vq, __half* __restrict__ vs,
                                         int n_head_kv, float qk_clip, float rms_eps,
                                         const int32_t* __restrict__ pos0, int cap, int is_swa) {
    const int hkv = blockIdx.x, t = blockIdx.y, lane = threadIdx.x & 31;
    const int kv_dim = n_head_kv * 128;
    const float* kp = k + (size_t) t * kv_dim + hkv * 128;
    const float* vp = v + (size_t) t * kv_dim + hkv * 128;
    float kv[4], vv[4], sk = 0.0f, sv = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        kv[i] = kp[i * 32 + lane];
        vv[i] = vp[i * 32 + lane];
        sk += kv[i] * kv[i];
        sv += vv[i] * vv[i];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        sk += __shfl_xor_sync(0xffffffffu, sk, o);
        sv += __shfl_xor_sync(0xffffffffu, sv, o);
    }
    const float fk = fminf(1.0f, qk_clip / (sqrtf(sk) + 1e-6f));
    const float fv = rsqrtf(sv / 128.0f + rms_eps);
    float ak = 0.0f, av = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        kv[i] *= fk;
        vv[i] *= fv;
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
    const int slot = is_swa ? (((cap & (cap - 1)) == 0) ? (pos & (cap - 1)) : (pos % cap)) : pos;
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
                            int n_head, int n_rel, int n_ext, int max_rel, const int32_t* __restrict__ pos0, int T) {
    __shared__ float sr[16];
    const int h = blockIdx.x, t = blockIdx.y;
    const int max_d = min(min(n_ext, max_rel), pos0[0] + T);
    const float* rr = R + ((size_t) t * n_head + h) * n_rel;
    if (threadIdx.x < n_rel) sr[threadIdx.x] = rr[threadIdx.x];
    __syncthreads();
    const float sk = sink[(size_t) t * n_head + h];
    float* out = table + ((size_t) t * n_head + h) * n_ext;
    const float4 sr0 = *(const float4*) &sr[0];
    const float4 sr1 = *(const float4*) &sr[4];
    const float4 sr2 = *(const float4*) &sr[8];
    const float4 sr3 = *(const float4*) &sr[12];
    for (int d = threadIdx.x; d < max_d; d += blockDim.x) {
        const float4* rb4 = (const float4*) (rel_bias + (size_t) d * n_rel);
        const float4 r0 = rb4[0], r1 = rb4[1], r2 = rb4[2], r3 = rb4[3];
        const float dot = (sr0.x * r0.x + sr0.y * r0.y + sr0.z * r0.z + sr0.w * r0.w)
                        + (sr1.x * r1.x + sr1.y * r1.y + sr1.z * r1.z + sr1.w * r1.w)
                        + (sr2.x * r2.x + sr2.y * r2.y + sr2.z * r2.z + sr2.w * r2.w)
                        + (sr3.x * r3.x + sr3.y * r3.y + sr3.z * r3.z + sr3.w * r3.w);
        out[d] = dot - sk;
    }
}

__global__ void k_attn_q8(const float* __restrict__ Q,
                          const int8_t* __restrict__ kq, const __half* __restrict__ ks,
                          const int8_t* __restrict__ vq, const __half* __restrict__ vs,
                          const float* __restrict__ rel_tab, const float* __restrict__ sink,
                          const float* __restrict__ gate, float* __restrict__ out,
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

    const int base4 = lane * 4;
    const float q0 = sq[base4], q1 = sq[base4 + 1], q2 = sq[base4 + 2], q3 = sq[base4 + 3];

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
        const char4 kc4 = *reinterpret_cast<const char4*>(kj + base4);
        float dot = q0 * (float) kc4.x + q1 * (float) kc4.y + q2 * (float) kc4.z + q3 * (float) kc4.w;
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
        const char4 vc4 = *reinterpret_cast<const char4*>(vj + base4);
        acc0 = acc0 * alpha + pv * (float) vc4.x;
        acc1 = acc1 * alpha + pv * (float) vc4.y;
        acc2 = acc2 * alpha + pv * (float) vc4.z;
        acc3 = acc3 * alpha + pv * (float) vc4.w;
        m = m_new;
    }
    if (lane == 0) {
        s_m[warp] = m;
        s_l[warp] = l;
    }
    s_acc[warp][base4] = acc0;
    s_acc[warp][base4 + 1] = acc1;
    s_acc[warp][base4 + 2] = acc2;
    s_acc[warp][base4 + 3] = acc3;
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
        const size_t oi = ((size_t) t * n_head + h) * 128 + threadIdx.x;
        float val = og / lg;
        if (gate) val *= sigmf(gate[oi]);
        out[oi] = val;
    }
}

inline constexpr int kAttnSplit = 12;

/// GQA-3 prefill Q8 KV attention: 1 CTA of 128 threads processes all 3 Q-heads per KV head, 3x less KV HBM traffic.
/// grid (n_head_kv, T), block 128. Requires n_head == 3 * n_head_kv and head_dim == 128.
__global__ void k_attn_q8_gqa3_pp(const float* __restrict__ Q,
                               const int8_t* __restrict__ kq, const __half* __restrict__ ks,
                               const int8_t* __restrict__ vq, const __half* __restrict__ vs,
                               const float* __restrict__ rel_tab, const float* __restrict__ sink,
                               const float* __restrict__ gate, float* __restrict__ out,
                               int n_head, int n_head_kv, int n_ext, int cap, int is_swa, int n_swa,
                               const int32_t* __restrict__ pos0, float kq_scale) {
    __shared__ float sq[3][128];
    __shared__ float s_m[3][4];
    __shared__ float s_l[3][4];
    __shared__ float s_acc[3][4][128];
    const int hkv = blockIdx.x, t = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int kv_dim = n_head_kv * 128;
    const int h0 = 3 * hkv;
    if (threadIdx.x < 128) {
#pragma unroll
        for (int g = 0; g < 3; ++g) {
            sq[g][threadIdx.x] = Q[((size_t) t * n_head + (h0 + g)) * 128 + threadIdx.x];
        }
    }
    __syncthreads();

    const int base4 = lane * 4;
    float q[3][4];
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        q[g][0] = sq[g][base4];
        q[g][1] = sq[g][base4 + 1];
        q[g][2] = sq[g][base4 + 2];
        q[g][3] = sq[g][base4 + 3];
    }

    const int pos_q = pos0[0] + t;
    const int j0 = is_swa ? max(0, pos_q - n_swa + 1) : 0;
    const int j1 = pos_q + 1;
    const bool pow2_cap = (cap & (cap - 1)) == 0;
    const int cap_mask = cap - 1;

    float neg_sink[3];
    const float* rt[3];
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        const int h = h0 + g;
        rt[g] = rel_tab + ((size_t) t * n_head + h) * n_ext;
        neg_sink[g] = -sink[(size_t) t * n_head + h];
    }

    float m[3], l[3], acc[3][4];
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        m[g] = (warp == 0) ? 0.0f : -1e30f;
        l[g] = (warp == 0) ? 1.0f : 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[g][i] = 0.0f;
    }

    for (int j = j0 + warp; j < j1; j += 4) {
        const int slot = is_swa ? (pow2_cap ? (j & cap_mask) : (j % cap)) : j;
        const size_t idx = (size_t) slot * n_head_kv + hkv;
        const float d_k = __half2float(ks[idx]);
        const float d_v = __half2float(vs[idx]);
        const int8_t* kj = kq + (size_t) slot * kv_dim + hkv * 128;
        const int8_t* vj = vq + (size_t) slot * kv_dim + hkv * 128;
        const char4 kc4 = *reinterpret_cast<const char4*>(kj + base4);
        const float k0 = (float) kc4.x, k1 = (float) kc4.y, k2 = (float) kc4.z, k3 = (float) kc4.w;
        const char4 vc4 = *reinterpret_cast<const char4*>(vj + base4);
        const float v0 = (float) vc4.x, v1 = (float) vc4.y, v2 = (float) vc4.z, v3 = (float) vc4.w;
        const int dist = pos_q - j;
        const float dkkq = d_k * kq_scale;

        float dot0 = q[0][0] * k0 + q[0][1] * k1 + q[0][2] * k2 + q[0][3] * k3;
        float dot1 = q[1][0] * k0 + q[1][1] * k1 + q[1][2] * k2 + q[1][3] * k3;
        float dot2 = q[2][0] * k0 + q[2][1] * k1 + q[2][2] * k2 + q[2][3] * k3;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            dot0 += __shfl_xor_sync(0xffffffffu, dot0, o);
            dot1 += __shfl_xor_sync(0xffffffffu, dot1, o);
            dot2 += __shfl_xor_sync(0xffffffffu, dot2, o);
        }
        const float dots[3] = {dot0, dot1, dot2};
#pragma unroll
        for (int g = 0; g < 3; ++g) {
            const float bias = dist < n_ext ? rt[g][dist] : neg_sink[g];
            const float sc = dots[g] * dkkq + bias;
            const float m_new = fmaxf(m[g], sc);
            const float alpha = l[g] > 0.0f ? __expf(m[g] - m_new) : 0.0f;
            const float p = __expf(sc - m_new);
            const float pv = p * d_v;
            l[g] = l[g] * alpha + p;
            acc[g][0] = acc[g][0] * alpha + pv * v0;
            acc[g][1] = acc[g][1] * alpha + pv * v1;
            acc[g][2] = acc[g][2] * alpha + pv * v2;
            acc[g][3] = acc[g][3] * alpha + pv * v3;
            m[g] = m_new;
        }
    }

#pragma unroll
    for (int g = 0; g < 3; ++g) {
        if (lane == 0) {
            s_m[g][warp] = m[g];
            s_l[g][warp] = l[g];
        }
        s_acc[g][warp][base4] = acc[g][0];
        s_acc[g][warp][base4 + 1] = acc[g][1];
        s_acc[g][warp][base4 + 2] = acc[g][2];
        s_acc[g][warp][base4 + 3] = acc[g][3];
    }
    __syncthreads();
    if (threadIdx.x < 128) {
#pragma unroll
        for (int g = 0; g < 3; ++g) {
            float mg = s_m[g][0];
            for (int w = 1; w < 4; ++w) if (s_l[g][w] > 0.0f && s_m[g][w] > mg) mg = s_m[g][w];
            float lg = 0.0f, og = 0.0f;
            for (int w = 0; w < 4; ++w) {
                if (s_l[g][w] > 0.0f) {
                    const float c = __expf(s_m[g][w] - mg);
                    lg += s_l[g][w] * c;
                    og += s_acc[g][w][threadIdx.x] * c;
                }
            }
            const size_t oi = ((size_t) t * n_head + (h0 + g)) * 128 + threadIdx.x;
            float val = __fdividef(og, lg);
            if (gate) val *= sigmf(gate[oi]);
            out[oi] = val;
        }
    }
}

/// GQA-3 Split-KV Q8 decode attention (T == 1): grid (n_head_kv, SPLIT), block 128 -> fills all SMs & shares KV 3x.
__global__ void k_attn_q8_gqa3_split(const float* __restrict__ Q,
                                     const int8_t* __restrict__ kq, const __half* __restrict__ ks,
                                     const int8_t* __restrict__ vq, const __half* __restrict__ vs,
                                     const float* __restrict__ rel_tab, const float* __restrict__ sink,
                                     float* __restrict__ split_out,
                                     int n_head, int n_head_kv, int n_ext, int cap, int is_swa, int n_swa,
                                     const int32_t* __restrict__ pos0, float kq_scale, int split_n) {
    __shared__ float sq[3][128];
    __shared__ float s_m[3][4];
    __shared__ float s_l[3][4];
    __shared__ float s_acc[3][4][128];
    const int hkv = blockIdx.x, sp = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int kv_dim = n_head_kv * 128;
    const int h0 = 3 * hkv;
    if (threadIdx.x < 128) {
#pragma unroll
        for (int g = 0; g < 3; ++g) {
            sq[g][threadIdx.x] = Q[(size_t) (h0 + g) * 128 + threadIdx.x];
        }
    }
    __syncthreads();

    const int base4 = lane * 4;
    float q[3][4];
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        q[g][0] = sq[g][base4];
        q[g][1] = sq[g][base4 + 1];
        q[g][2] = sq[g][base4 + 2];
        q[g][3] = sq[g][base4 + 3];
    }

    const int pos_q = pos0[0];
    const int j0 = is_swa ? max(0, pos_q - n_swa + 1) : 0;
    const int j1 = pos_q + 1;
    const bool pow2_cap = (cap & (cap - 1)) == 0;
    const int cap_mask = cap - 1;

    float neg_sink[3];
    const float* rt[3];
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        const int h = h0 + g;
        rt[g] = rel_tab + (size_t) h * n_ext;
        neg_sink[g] = -sink[h];
    }

    const bool has_sink = (sp == 0 && warp == 0);
    float m[3], l[3], acc[3][4];
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        m[g] = has_sink ? 0.0f : -1e30f;
        l[g] = has_sink ? 1.0f : 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[g][i] = 0.0f;
    }

    const int stride_j = 4 * split_n;
    for (int j = j0 + sp * 4 + warp; j < j1; j += stride_j) {
        const int slot = is_swa ? (pow2_cap ? (j & cap_mask) : (j % cap)) : j;
        const size_t idx = (size_t) slot * n_head_kv + hkv;
        const float d_k = __half2float(ks[idx]);
        const float d_v = __half2float(vs[idx]);
        const int8_t* kj = kq + (size_t) slot * kv_dim + hkv * 128;
        const int8_t* vj = vq + (size_t) slot * kv_dim + hkv * 128;
        const char4 kc4 = *reinterpret_cast<const char4*>(kj + base4);
        const float k0 = (float) kc4.x, k1 = (float) kc4.y, k2 = (float) kc4.z, k3 = (float) kc4.w;
        const char4 vc4 = *reinterpret_cast<const char4*>(vj + base4);
        const float v0 = (float) vc4.x, v1 = (float) vc4.y, v2 = (float) vc4.z, v3 = (float) vc4.w;
        const int dist = pos_q - j;
        const float dkkq = d_k * kq_scale;

        float dot0 = q[0][0] * k0 + q[0][1] * k1 + q[0][2] * k2 + q[0][3] * k3;
        float dot1 = q[1][0] * k0 + q[1][1] * k1 + q[1][2] * k2 + q[1][3] * k3;
        float dot2 = q[2][0] * k0 + q[2][1] * k1 + q[2][2] * k2 + q[2][3] * k3;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            dot0 += __shfl_xor_sync(0xffffffffu, dot0, o);
            dot1 += __shfl_xor_sync(0xffffffffu, dot1, o);
            dot2 += __shfl_xor_sync(0xffffffffu, dot2, o);
        }
        const float dots[3] = {dot0, dot1, dot2};
#pragma unroll
        for (int g = 0; g < 3; ++g) {
            const float bias = dist < n_ext ? rt[g][dist] : neg_sink[g];
            const float sc = dots[g] * dkkq + bias;
            const float m_new = fmaxf(m[g], sc);
            const float alpha = l[g] > 0.0f ? __expf(m[g] - m_new) : 0.0f;
            const float p = __expf(sc - m_new);
            const float pv = p * d_v;
            l[g] = l[g] * alpha + p;
            acc[g][0] = acc[g][0] * alpha + pv * v0;
            acc[g][1] = acc[g][1] * alpha + pv * v1;
            acc[g][2] = acc[g][2] * alpha + pv * v2;
            acc[g][3] = acc[g][3] * alpha + pv * v3;
            m[g] = m_new;
        }
    }

#pragma unroll
    for (int g = 0; g < 3; ++g) {
        if (lane == 0) {
            s_m[g][warp] = m[g];
            s_l[g][warp] = l[g];
        }
        s_acc[g][warp][base4] = acc[g][0];
        s_acc[g][warp][base4 + 1] = acc[g][1];
        s_acc[g][warp][base4 + 2] = acc[g][2];
        s_acc[g][warp][base4 + 3] = acc[g][3];
    }
    __syncthreads();
    if (threadIdx.x < 128) {
#pragma unroll
        for (int g = 0; g < 3; ++g) {
            const int h = h0 + g;
            float mg = -1e30f;
            for (int w = 0; w < 4; ++w) if (s_l[g][w] > 0.0f && s_m[g][w] > mg) mg = s_m[g][w];
            float lg = 0.0f, og = 0.0f;
            for (int w = 0; w < 4; ++w) {
                if (s_l[g][w] > 0.0f) {
                    const float c = __expf(s_m[g][w] - mg);
                    lg += s_l[g][w] * c;
                    og += s_acc[g][w][threadIdx.x] * c;
                }
            }
            float* dst = split_out + (size_t) (sp * n_head + h) * 130;
            dst[threadIdx.x] = og;
            if (threadIdx.x == 0) {
                dst[128] = mg;
                dst[129] = lg;
            }
        }
    }
}

/// Combine Split-KV partials across `split_n` and apply fused sigmoid gate: grid (n_head), block 128.
__global__ void k_attn_split_combine(const float* __restrict__ split_out, const float* __restrict__ gate,
                                     float* __restrict__ out, int n_head, int split_n) {
    const int h = blockIdx.x, d = threadIdx.x;
    float mg = -1e30f;
    for (int s = 0; s < split_n; ++s) {
        const float* src = split_out + (size_t) (s * n_head + h) * 130;
        const float ls = src[129];
        const float ms = src[128];
        if (ls > 0.0f && ms > mg) mg = ms;
    }
    float lg = 0.0f, og = 0.0f;
    for (int s = 0; s < split_n; ++s) {
        const float* src = split_out + (size_t) (s * n_head + h) * 130;
        const float ls = src[129];
        if (ls > 0.0f) {
            const float c = __expf(src[128] - mg);
            lg += ls * c;
            og += src[d] * c;
        }
    }
    const size_t oi = (size_t) h * 128 + d;
    float val = __fdividef(og, lg);
    if (gate) val *= sigmf(gate[oi]);
    out[oi] = val;
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
/// n_exp <= 128, top_k <= 16. 1 warp (32 threads) per token (grid T).
__global__ void k_route_inkling2(const float* __restrict__ logits, const float* __restrict__ bias,
                                 int32_t* __restrict__ ids, float* __restrict__ ewts,
                                 int n_exp, int top_k, float scale) {
    const int t = blockIdx.x, lane = threadIdx.x;
    float p[4], s[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int e = j * 32 + lane;
        if (e < n_exp) {
            const float pr = sigmf(logits[(size_t) t * n_exp + e]);
            p[j] = pr;
            s[j] = pr + (bias ? bias[e] : 0.0f);
        } else {
            p[j] = 0.0f;
            s[j] = -1e30f;
        }
    }
    float wlocal[16];
    float wsum = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        float bv = s[0];
        int be = lane;
        float bp = p[0];
#pragma unroll
        for (int j = 1; j < 4; ++j) {
            if (s[j] > bv) {
                bv = s[j];
                be = j * 32 + lane;
                bp = p[j];
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oe = __shfl_xor_sync(0xffffffffu, be, o);
            const float op = __shfl_xor_sync(0xffffffffu, bp, o);
            if (ov > bv || (ov == bv && oe < be)) {
                bv = ov;
                be = oe;
                bp = op;
            }
        }
        if (lane == 0) {
            ids[(size_t) t * top_k + k] = be;
            wlocal[k] = bp;
            wsum += bp;
        }
        if ((be & 31) == lane) s[be >> 5] = -1e30f;
    }
    if (lane == 0) {
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

/// Group entries for PCIe-staged misses in expert ID range [e0, e1): matches slot_of[x] < 0 && x >= e0 && x < e1,
/// resolving each weight blob at `stage_blobs + (x - e0) * stride`.
__global__ void k_group_experts_miss_range(const int32_t* __restrict__ ids, const int32_t* __restrict__ slot_of,
                                           const uint8_t* __restrict__ stage_blobs, size_t stride,
                                           int n_ent, int n_exp, int n_slots, int e0, int e1,
                                           unsigned long long* __restrict__ grp_ptr, int32_t* __restrict__ grp_start,
                                           int32_t* __restrict__ n_groups, int32_t* __restrict__ ent_dst,
                                           int32_t* __restrict__ ent_tok, int top_k) {
    __shared__ int32_t cnt[128];
    __shared__ int8_t active[128];
    __shared__ int32_t cur_off[128];
    if (threadIdx.x < n_exp) {
        cnt[threadIdx.x] = 0;
        const int x = (int) threadIdx.x;
        const bool ok = (x >= e0 && x < e1) && (n_slots <= 0 || slot_of[x] < 0);
        active[x] = ok ? 1 : 0;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x) {
        const int x = ids[e];
        if (active[x]) atomicAdd(&cnt[x], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int ng = 0;
        int off = 0;
        for (int x = e0; x < e1; ++x) {
            if (cnt[x] > 0) {
                grp_ptr[ng] = (unsigned long long) (stage_blobs + (size_t) (x - e0) * stride);
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
        if (active[x]) {
            const int pos = atomicAdd(&cur_off[x], 1);
            ent_dst[pos] = e;
            ent_tok[pos] = e / top_k;
        }
    }
}

/// Compact slot mapping inside `pp_stage[b]` for missing experts in one staging batch (-1 = not in batch).
struct PpStageMap {
    int16_t pos[128];
};

/// Group entries for compact PCIe-staged misses where `map.pos[x] >= 0` gives the compact slot inside `stage_blobs`.
__global__ void k_group_experts_miss_map(const int32_t* __restrict__ ids,
                                         const uint8_t* __restrict__ stage_blobs, size_t stride,
                                         int n_ent, int n_exp, PpStageMap map,
                                         unsigned long long* __restrict__ grp_ptr, int32_t* __restrict__ grp_start,
                                         int32_t* __restrict__ n_groups, int32_t* __restrict__ ent_dst,
                                         int32_t* __restrict__ ent_tok, int top_k) {
    __shared__ int32_t cnt[128];
    __shared__ int16_t spos[128];
    __shared__ int32_t cur_off[128];
    if (threadIdx.x < n_exp) {
        cnt[threadIdx.x] = 0;
        spos[threadIdx.x] = map.pos[threadIdx.x];
    }
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x) {
        const int x = ids[e];
        if (spos[x] >= 0) atomicAdd(&cnt[x], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int ng = 0;
        int off = 0;
        for (int x = 0; x < n_exp; ++x) {
            if (cnt[x] > 0) {
                grp_ptr[ng] = (unsigned long long) (stage_blobs + (size_t) spos[x] * stride);
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
        if (spos[x] >= 0) {
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
