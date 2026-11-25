// src/inkling2/inkling2_kernels.cuh - CUDA kernels for Inkling-2-Mini (inkling2).
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
#include <mma.h>
#endif
#include <cstdint>
#include <cmath>

namespace strata::inkling2 {

inline constexpr int kThreads = 256;

__device__ __forceinline__ float sigmf(float x) { return __fdividef(1.0f, 1.0f + __expf(-x)); }
__device__ __forceinline__ float sigmf_fast_sm75(float x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    float t;
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * x));
    return fmaf(0.5f, t, 0.5f);
#else
    return sigmf(x);
#endif
}
__device__ __forceinline__ void stg_f2(float2* p, float vx, float vy) {
#if defined(__CUDA_ARCH__)
    asm volatile("st.global.v2.f32 [%0], {%1, %2};\n" :: "l"(p), "f"(vx), "f"(vy) : "memory");
#else
    *p = make_float2(vx, vy);
#endif
}
__device__ __forceinline__ int dp4a_i8(int a, int b, int c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    int s = c;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int8_t xa = (int8_t) ((a >> (i * 8)) & 0xFF);
        const int8_t xb = (int8_t) ((b >> (i * 8)) & 0xFF);
        s += (int) xa * (int) xb;
    }
    return s;
#endif
}
__device__ __forceinline__ float fast_tanh(float x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    float r;
    asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
#else
    const float xc = fmaxf(-9.0f, fminf(9.0f, x));
    return __fdividef(2.0f, 1.0f + __expf(-2.0f * xc)) - 1.0f;
#endif
}

__device__ __forceinline__ float4 ldcs_f4(const float4* p) {
#if defined(__CUDA_ARCH__)
    return __ldcs(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ float2 ldcs_f2(const float2* p) {
#if defined(__CUDA_ARCH__)
    return __ldcs(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ void stcs_f4(float4* p, float4 v) {
#if defined(__CUDA_ARCH__)
    __stcs(p, v);
#else
    *p = v;
#endif
}

__device__ __forceinline__ void stcs_u2(uint2* p, uint2 v) {
#if defined(__CUDA_ARCH__)
    __stcs(p, v);
#else
    *p = v;
#endif
}

/// Bit-exact Q8_1 block store (36 bytes: half2 ds + int8_t qs[32]) for one full warp (32 values).
/// Uses div.full.f32 (matching native_mmvq.cu --use_fast_math) for both 10x speed and bit-exact parity.
__device__ __forceinline__ void q8_1_warp_store(float xi, void* __restrict__ xq_base, int blk_idx, int lane) {
    float amax = fabsf(xi);
    float sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    constexpr float kHalfMax = 65504.0f;
    const float d_raw = amax * (1.0f / 127.0f);
    const float d = fabsf(d_raw) > kHalfMax ? copysignf(kHalfMax, d_raw) : d_raw;
    int8_t q = 0;
    if (amax != 0.0f) {
        float q_div;
#if defined(__CUDA_ARCH__)
        asm("div.full.f32 %0, %1, %2;" : "=f"(q_div) : "f"(xi), "f"(d));
#else
        q_div = xi / d;
#endif
        const float qf = roundf(q_div);
        q = (int8_t) (qf > 127.0f ? 127.0f : (qf < -127.0f ? -127.0f : qf));
    }
    uint8_t* dst = reinterpret_cast<uint8_t*>(xq_base) + (size_t) blk_idx * 36u;
    reinterpret_cast<int8_t*>(dst + 4)[lane] = q;
    if (lane == 0) {
        const float s_fin = fabsf(sum) > kHalfMax ? copysignf(kHalfMax, sum) : sum;
        *reinterpret_cast<__half2*>(dst) = make_half2(__float2half(d), __float2half(s_fin));
    }
}

/// 8-lane sub-warp Q8_1 store (8 consecutive threads x float4 = 32 elements = 1 block_q8_1, coalesced 32-bit STG).
__device__ __forceinline__ void q8_1_sub8_store(float4 v, void* __restrict__ xq_base, int blk_idx, int sub_lane) {
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    float sum = (v.x + v.y) + (v.z + v.w);
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    constexpr float kHalfMax = 65504.0f;
    const float d_raw = amax * (1.0f / 127.0f);
    const float d = fabsf(d_raw) > kHalfMax ? copysignf(kHalfMax, d_raw) : d_raw;
    auto qf8 = [&](float xi) -> uint32_t {
        if (amax == 0.0f) return 0u;
        float q_div;
#if defined(__CUDA_ARCH__)
        asm("div.full.f32 %0, %1, %2;" : "=f"(q_div) : "f"(xi), "f"(d));
#else
        q_div = xi / d;
#endif
        const float qf = roundf(q_div);
        const int8_t qi = (int8_t) (qf > 127.0f ? 127.0f : (qf < -127.0f ? -127.0f : qf));
        return (uint32_t) (uint8_t) qi;
    };
    const uint32_t pack = qf8(v.x) | (qf8(v.y) << 8) | (qf8(v.z) << 16) | (qf8(v.w) << 24);
    uint8_t* dst = reinterpret_cast<uint8_t*>(xq_base) + (size_t) blk_idx * 36u;
    reinterpret_cast<uint32_t*>(dst + 4)[sub_lane] = pack;
    if (sub_lane == 0) {
        const float s_fin = fabsf(sum) > kHalfMax ? copysignf(kHalfMax, sum) : sum;
        *reinterpret_cast<__half2*>(dst) = make_half2(__float2half(d), __float2half(s_fin));
    }
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
    if ((n & 3) == 0) {
        const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
        const int64_t n4 = n >> 2;
        if (i >= n4) return;
        const float4 v = reinterpret_cast<const float4*>(x)[i];
        const uint32_t u0 = __float_as_uint(v.x), u1 = __float_as_uint(v.y);
        const uint32_t u2 = __float_as_uint(v.z), u3 = __float_as_uint(v.w);
        const uint32_t b0 = (u0 + (0x7FFFu + ((u0 >> 16) & 1u))) >> 16;
        const uint32_t b1 = (u1 + (0x7FFFu + ((u1 >> 16) & 1u))) >> 16;
        const uint32_t b2 = (u2 + (0x7FFFu + ((u2 >> 16) & 1u))) >> 16;
        const uint32_t b3 = (u3 + (0x7FFFu + ((u3 >> 16) & 1u))) >> 16;
        reinterpret_cast<uint2*>(y)[i] = make_uint2(b0 | (b1 << 16), b2 | (b3 << 16));
        return;
    }
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t u = __float_as_uint(x[i]);
    const uint32_t bias = 0x7FFFu + ((u >> 16) & 1u);
    y[i] = (uint16_t) ((u + bias) >> 16);
}

__global__ void k_f32_to_f16(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    if ((n & 3) == 0) {
        const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
        const int64_t n4 = n >> 2;
        if (i >= n4) return;
        const float4 v = reinterpret_cast<const float4*>(x)[i];
        __half2 p0 = __floats2half2_rn(v.x, v.y);
        __half2 p1 = __floats2half2_rn(v.z, v.w);
        uint2 out;
        memcpy(&out.x, &p0, 4);
        memcpy(&out.y, &p1, 4);
        reinterpret_cast<uint2*>(y)[i] = out;
        return;
    }
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

/// Streaming tail-wave zero kernel (Wave 12): uses st.global.cs.v4.f32 so zeroing large cold buffers
/// (e.g. moe_out) never evicts hot activations/weights from L2. Zero SMEM, <=16 regs, STACK: 0!
__global__ void k_zero_f4_cs(float* __restrict__ dst, size_t n_floats) {
    const size_t tid = (size_t) blockIdx.x * (size_t) blockDim.x + (size_t) threadIdx.x;
    const size_t stride = (size_t) gridDim.x * (size_t) blockDim.x;
    if ((reinterpret_cast<uintptr_t>(dst) & 15u) == 0u) {
        const size_t n4 = n_floats >> 2;
        const float4 z4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4* dst4 = reinterpret_cast<float4*>(dst);
        for (size_t i = tid; i < n4; i += stride) {
            stcs_f4(dst4 + i, z4);
        }
        const size_t tail_base = n4 << 2;
        for (size_t i = tail_base + tid; i < n_floats; i += stride) {
            dst[i] = 0.0f;
        }
    } else {
        for (size_t i = tid; i < n_floats; i += stride) {
            dst[i] = 0.0f;
        }
    }
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

/// Masked RMSNorm: applies token_embd_norm ONLY when !is_img[t]; passes image embeddings straight through.
__global__ void k_rms_norm_masked(const float* __restrict__ x, const float* __restrict__ w,
                                  const uint8_t* __restrict__ is_img, float* __restrict__ y,
                                  int n, float eps) {
    __shared__ float sh[32];
    const int t = blockIdx.x;
    const float* xr = x + (size_t) t * n;
    float* yr = y + (size_t) t * n;
    if (is_img && is_img[t]) {
        if ((n & 3) == 0) {
            const int n4 = n >> 2;
            const float4* x4 = (const float4*) xr;
            float4* y4 = (float4*) yr;
            for (int i = threadIdx.x; i < n4; i += blockDim.x) y4[i] = x4[i];
        } else {
            for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i];
        }
        return;
    }
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

/// Fused RMSNorm + Q8_1 quantization: single-pass register-cached for N == 2304 (576 threads = 18 warps = 72 sub8 blocks,
/// 0 s_x SMEM, 0 idle threads), writes both FP32 `y` and `block_q8_1` `xq` (when xq != nullptr) in 1 launch.
/// Grid (T), block (n==2304 ? 576 : 256).
__global__ void __launch_bounds__(576, 2) k_rms_norm_q8_1(const float* __restrict__ x, const float* __restrict__ w,
                                float* __restrict__ y, void* __restrict__ xq_out, int n, float eps) {
    __shared__ float sh[32];
    const int t = blockIdx.x;
    const float* xr = x + (size_t) t * n;
    float* yr = y + (size_t) t * n;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int nb = n >> 5;
    if (n == 2304 && blockDim.x == 576) {
        const float4 v = reinterpret_cast<const float4*>(xr)[tid];
        float s = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) sh[warp] = s;
        __syncthreads();
        if (warp == 0) {
            float tot = (lane < 18) ? sh[lane] : 0.0f;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, o);
            if (lane == 0) sh[0] = rsqrtf(tot / 2304.0f + eps);
        }
        __syncthreads();
        const float r = sh[0];
        float4 o;
        if (w) {
            const float4 ww = __ldg(reinterpret_cast<const float4*>(w) + tid);
            o = make_float4(v.x * r * ww.x, v.y * r * ww.y, v.z * r * ww.z, v.w * r * ww.w);
        } else {
            o = make_float4(v.x * r, v.y * r, v.z * r, v.w * r);
        }
        reinterpret_cast<float4*>(yr)[tid] = o;
        if (xq_out) q8_1_sub8_store(o, xq_out, t * 72 + (tid >> 3), tid & 7);
        return;
    }
    const int n4 = n >> 2;
    const float4* x4 = reinterpret_cast<const float4*>(xr);
    float s = 0.0f;
    for (int i = tid; i < n4; i += blockDim.x) {
        const float4 v = x4[i];
        s += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    const int nwarps = blockDim.x >> 5;
    for (int b = warp; b < nb; b += nwarps) {
        const int idx = (b << 5) + lane;
        const float val = xr[idx] * r * (w ? w[idx] : 1.0f);
        yr[idx] = val;
        if (xq_out) q8_1_warp_store(val, xq_out, t * nb + b, lane);
    }
}

/// Fused 3-way RMSNorm + Q8_1 quantization (Q, K, V inputs, distinct or shared weights): grid (3, T), block (n==2304 ? 576 : 256).
/// Single-pass register-cached for N == 2304 (0 s_x SMEM, 0 idle threads). Collapses up to 6 launches into 1!
__global__ void __launch_bounds__(576, 2) k_rms_norm_3_q8_1(const float* __restrict__ x0, const float* __restrict__ x1, const float* __restrict__ x2,
                                  float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
                                  const float* __restrict__ w0, const float* __restrict__ w1, const float* __restrict__ w2,
                                  void* __restrict__ xq0, void* __restrict__ xq1, void* __restrict__ xq2,
                                  int n, float eps) {
    __shared__ float sh[32];
    const int which = blockIdx.x, t = blockIdx.y;
    const float* xr = (which == 0 ? x0 : (which == 1 ? x1 : x2)) + (size_t) t * n;
    float* yr = (which == 0 ? y0 : (which == 1 ? y1 : y2)) + (size_t) t * n;
    const float* wr = (which == 0 ? w0 : (which == 1 ? w1 : w2));
    void* xqr = (which == 0 ? xq0 : (which == 1 ? xq1 : xq2));
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int nb = n >> 5;
    if (n == 2304 && blockDim.x == 576) {
        const float4 v = reinterpret_cast<const float4*>(xr)[tid];
        float s = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) sh[warp] = s;
        __syncthreads();
        if (warp == 0) {
            float tot = (lane < 18) ? sh[lane] : 0.0f;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, o);
            if (lane == 0) sh[0] = rsqrtf(tot / 2304.0f + eps);
        }
        __syncthreads();
        const float r = sh[0];
        float4 o;
        if (wr) {
            const float4 ww = __ldg(reinterpret_cast<const float4*>(wr) + tid);
            o = make_float4(v.x * r * ww.x, v.y * r * ww.y, v.z * r * ww.z, v.w * r * ww.w);
        } else {
            o = make_float4(v.x * r, v.y * r, v.z * r, v.w * r);
        }
        reinterpret_cast<float4*>(yr)[tid] = o;
        if (xqr) q8_1_sub8_store(o, xqr, t * 72 + (tid >> 3), tid & 7);
        return;
    }
    const int n4 = n >> 2;
    const float4* x4 = reinterpret_cast<const float4*>(xr);
    float s = 0.0f;
    for (int i = tid; i < n4; i += blockDim.x) {
        const float4 v = x4[i];
        s += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    const int nwarps = blockDim.x >> 5;
    for (int b = warp; b < nb; b += nwarps) {
        const int idx = (b << 5) + lane;
        const float val = xr[idx] * r * (wr ? wr[idx] : 1.0f);
        yr[idx] = val;
        if (xqr) q8_1_warp_store(val, xqr, t * nb + b, lane);
    }
}

/// Fused RMSNorm + FP16 output (for prefill Tensor Core paths): writes optional FP32 `y` and FP16 `y16`,
/// plus optional Scaled-Ozaki low-part `out_lo_f16` (normal L2-cached stores for both `y16` and `out_lo_f16` when enabled).
/// Single-pass 576-thr register-cached for N == 2304 (halves HBM reads!), fallback 256-thr. Grid (T).
__global__ void __launch_bounds__(576, 2) k_rms_norm_f16(const float* __restrict__ x, const float* __restrict__ w,
                               float* __restrict__ y, uint16_t* __restrict__ y16, int n, float eps,
                               __half* __restrict__ out_lo_f16 = nullptr) {
    __shared__ float sh[32];
    const int t = blockIdx.x;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const size_t row_off = (size_t) t * n;
    const float4* x4 = reinterpret_cast<const float4*>(x + row_off);
    const float4* w4 = w ? reinterpret_cast<const float4*>(w) : nullptr;
    if (n == 2304 && blockDim.x == 576) {
        const float4 v = x4[tid];
        float s = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) sh[warp] = s;
        __syncthreads();
        if (warp == 0) {
            float tot = (lane < 18) ? sh[lane] : 0.0f;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, o);
            if (lane == 0) sh[0] = rsqrtf(tot / 2304.0f + eps);
        }
        __syncthreads();
        const float r = sh[0];
        float4 o;
        if (w4) {
            const float4 ww = __ldg(w4 + tid);
            o = make_float4(v.x * r * ww.x, v.y * r * ww.y, v.z * r * ww.z, v.w * r * ww.w);
        } else {
            o = make_float4(v.x * r, v.y * r, v.z * r, v.w * r);
        }
        if (y) reinterpret_cast<float4*>(y + row_off)[tid] = o;
        if (y16) {
            uint2* y16_4 = reinterpret_cast<uint2*>(y16 + row_off);
            const __half h0 = __float2half_rn(o.x);
            const __half h1 = __float2half_rn(o.y);
            const __half h2 = __float2half_rn(o.z);
            const __half h3 = __float2half_rn(o.w);
            const __half2 p0 = __halves2half2(h0, h1);
            const __half2 p1 = __halves2half2(h2, h3);
            uint2 u;
            memcpy(&u.x, &p0, 4);
            memcpy(&u.y, &p1, 4);
            if (out_lo_f16) {
                // Normal L2-cached stores so immediately following k_router_ozaki2_tc_presplit_pp hits L2!
                y16_4[tid] = u;
                const __half2 l0 = __halves2half2(__float2half_rn((o.x - __half2float(h0)) * 2048.0f),
                                                  __float2half_rn((o.y - __half2float(h1)) * 2048.0f));
                const __half2 l1 = __halves2half2(__float2half_rn((o.z - __half2float(h2)) * 2048.0f),
                                                  __float2half_rn((o.w - __half2float(h3)) * 2048.0f));
                uint2 ul;
                memcpy(&ul.x, &l0, 4);
                memcpy(&ul.y, &l1, 4);
                reinterpret_cast<uint2*>(out_lo_f16 + row_off)[tid] = ul;
            } else {
                stcs_u2(y16_4 + tid, u);
            }
        }
        return;
    }
    const int n4 = n >> 2;
    float s = 0.0f;
    for (int i = tid; i < n4; i += blockDim.x) {
        const float4 v = x4[i];
        s += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = tid; i < n4; i += blockDim.x) {
        const float4 v = x4[i];
        float4 o;
        if (w4) {
            const float4 ww = w4[i];
            o = make_float4(v.x * r * ww.x, v.y * r * ww.y, v.z * r * ww.z, v.w * r * ww.w);
        } else {
            o = make_float4(v.x * r, v.y * r, v.z * r, v.w * r);
        }
        if (y) reinterpret_cast<float4*>(y + row_off)[i] = o;
        if (y16) {
            uint2* y16_4 = reinterpret_cast<uint2*>(y16 + row_off);
            const __half h0 = __float2half_rn(o.x);
            const __half h1 = __float2half_rn(o.y);
            const __half h2 = __float2half_rn(o.z);
            const __half h3 = __float2half_rn(o.w);
            const __half2 p0 = __halves2half2(h0, h1);
            const __half2 p1 = __halves2half2(h2, h3);
            uint2 u;
            memcpy(&u.x, &p0, 4);
            memcpy(&u.y, &p1, 4);
            if (out_lo_f16) {
                y16_4[i] = u;
                const __half2 l0 = __halves2half2(__float2half_rn((o.x - __half2float(h0)) * 2048.0f),
                                                  __float2half_rn((o.y - __half2float(h1)) * 2048.0f));
                const __half2 l1 = __halves2half2(__float2half_rn((o.z - __half2float(h2)) * 2048.0f),
                                                  __float2half_rn((o.w - __half2float(h3)) * 2048.0f));
                uint2 ul;
                memcpy(&ul.x, &l0, 4);
                memcpy(&ul.y, &l1, 4);
                reinterpret_cast<uint2*>(out_lo_f16 + row_off)[i] = ul;
            } else {
                stcs_u2(y16_4 + i, u);
            }
        }
    }
}

/// Multi-row (4 rows / 256-thr CTA) 3-way RMSNorm + FP16 output for N == 2304.
/// Grid (3, (T + 3) / 4), Block 256 (STACK: 0, 4x fewer CTAs, 1 barrier per 4 rows).
__global__ void __launch_bounds__(256, 4) k_rms_norm_3_f16_x4(
        const float* __restrict__ x0, const float* __restrict__ x1, const float* __restrict__ x2,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        const float* __restrict__ w0, const float* __restrict__ w1, const float* __restrict__ w2,
        uint16_t* __restrict__ y16_0, uint16_t* __restrict__ y16_1, uint16_t* __restrict__ y16_2,
        int T, float eps) {
    __shared__ float sh[4][2];
    const int which = blockIdx.x;
    const int tid = threadIdx.x;
    const int rg = tid >> 6;   // 0..3
    const int rt = tid & 63;   // 0..63
    const int t = (int) blockIdx.y * 4 + rg;
    const bool valid = (t < T);
    const size_t row_off = (size_t) t * 2304u;
    const float* x_base = (which == 0 ? x0 : (which == 1 ? x1 : x2));
    float* y_base = (which == 0 ? y0 : (which == 1 ? y1 : y2));
    const float* wr = (which == 0 ? w0 : (which == 1 ? w1 : w2));
    uint16_t* y16_base = (which == 0 ? y16_0 : (which == 1 ? y16_1 : y16_2));
    float4 v[9];
    float s = 0.0f;
    if (valid) {
        const float4* x4 = reinterpret_cast<const float4*>(x_base + row_off);
#pragma unroll
        for (int i = 0; i < 9; ++i) {
            v[i] = x4[i * 64 + rt];
            s += v[i].x * v[i].x + v[i].y * v[i].y + v[i].z * v[i].z + v[i].w * v[i].w;
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((rt & 31) == 0) sh[rg][rt >> 5] = s;
    __syncthreads();
    if (!valid) return;
    const float r = rsqrtf((sh[rg][0] + sh[rg][1]) / 2304.0f + eps);
    const float4* w4 = wr ? reinterpret_cast<const float4*>(wr) : nullptr;
    float4* y4 = y_base ? reinterpret_cast<float4*>(y_base + row_off) : nullptr;
    uint2* y16_4 = y16_base ? reinterpret_cast<uint2*>(y16_base + row_off) : nullptr;
#pragma unroll
    for (int i = 0; i < 9; ++i) {
        const int idx = i * 64 + rt;
        float4 o;
        if (w4) {
            const float4 ww = __ldg(w4 + idx);
            o = make_float4(v[i].x * r * ww.x, v[i].y * r * ww.y, v[i].z * r * ww.z, v[i].w * r * ww.w);
        } else {
            o = make_float4(v[i].x * r, v[i].y * r, v[i].z * r, v[i].w * r);
        }
        if (y4) y4[idx] = o;
        if (y16_4) {
            const __half2 p0 = __halves2half2(__float2half_rn(o.x), __float2half_rn(o.y));
            const __half2 p1 = __halves2half2(__float2half_rn(o.z), __float2half_rn(o.w));
            uint2 u;
            memcpy(&u.x, &p0, 4);
            memcpy(&u.y, &p1, 4);
            stcs_u2(y16_4 + idx, u);
        }
    }
}

/// Fused 3-way RMSNorm + FP16 output (for prefill Tensor Core paths): writes FP32 y0/y1/y2 and optional FP16 y16_0/y16_1/y16_2.
/// Single-pass 576-thr register-cached for N == 2304 (halves HBM reads!), fallback 256-thr. Grid (3, T).
__global__ void __launch_bounds__(576, 2) k_rms_norm_3_f16(const float* __restrict__ x0, const float* __restrict__ x1, const float* __restrict__ x2,
                                 float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
                                 const float* __restrict__ w0, const float* __restrict__ w1, const float* __restrict__ w2,
                                 uint16_t* __restrict__ y16_0, uint16_t* __restrict__ y16_1, uint16_t* __restrict__ y16_2,
                                 int n, float eps) {
    __shared__ float sh[32];
    const int which = blockIdx.x, t = blockIdx.y;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const float* xr = (which == 0 ? x0 : (which == 1 ? x1 : x2)) + (size_t) t * n;
    float* yr_base = (which == 0 ? y0 : (which == 1 ? y1 : y2));
    const float* wr = (which == 0 ? w0 : (which == 1 ? w1 : w2));
    uint16_t* y16r = (which == 0 ? y16_0 : (which == 1 ? y16_1 : y16_2));
    const float4* x4 = reinterpret_cast<const float4*>(xr);
    float4* y4 = yr_base ? reinterpret_cast<float4*>(yr_base + (size_t) t * n) : nullptr;
    uint2* y16_4 = y16r ? reinterpret_cast<uint2*>(y16r + (size_t) t * n) : nullptr;
    const float4* w4 = wr ? reinterpret_cast<const float4*>(wr) : nullptr;
    if (n == 2304 && blockDim.x == 576) {
        const float4 v = x4[tid];
        float s = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) sh[warp] = s;
        __syncthreads();
        if (warp == 0) {
            float tot = (lane < 18) ? sh[lane] : 0.0f;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, o);
            if (lane == 0) sh[0] = rsqrtf(tot / 2304.0f + eps);
        }
        __syncthreads();
        const float r = sh[0];
        float4 o;
        if (w4) {
            const float4 ww = __ldg(w4 + tid);
            o = make_float4(v.x * r * ww.x, v.y * r * ww.y, v.z * r * ww.z, v.w * r * ww.w);
        } else {
            o = make_float4(v.x * r, v.y * r, v.z * r, v.w * r);
        }
        if (y4) y4[tid] = o;
        if (y16_4) {
            const __half2 p0 = __halves2half2(__float2half_rn(o.x), __float2half_rn(o.y));
            const __half2 p1 = __halves2half2(__float2half_rn(o.z), __float2half_rn(o.w));
            uint2 u;
            memcpy(&u.x, &p0, 4);
            memcpy(&u.y, &p1, 4);
            stcs_u2(y16_4 + tid, u);
        }
        return;
    }
    const int n4 = n >> 2;
    float s = 0.0f;
    for (int i = tid; i < n4; i += blockDim.x) {
        const float4 v = x4[i];
        s += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = tid; i < n4; i += blockDim.x) {
        const float4 v = x4[i];
        float4 o;
        if (w4) {
            const float4 ww = w4[i];
            o = make_float4(v.x * r * ww.x, v.y * r * ww.y, v.z * r * ww.z, v.w * r * ww.w);
        } else {
            o = make_float4(v.x * r, v.y * r, v.z * r, v.w * r);
        }
        if (y4) y4[i] = o;
        if (y16_4) {
            const __half2 p0 = __halves2half2(__float2half_rn(o.x), __float2half_rn(o.y));
            const __half2 p1 = __halves2half2(__float2half_rn(o.z), __float2half_rn(o.w));
            uint2 u;
            memcpy(&u.x, &p0, 4);
            memcpy(&u.y, &p1, 4);
            stcs_u2(y16_4 + i, u);
        }
    }
}

/// Fused 3-buffer F32->FP16 weight conversion for prefill run_control (down + gain + forget in 1 launch).
__global__ void k_f32_to_f16_ctrl_w(const float* __restrict__ w0, const float* __restrict__ w1, const float* __restrict__ w2,
                                    uint16_t* __restrict__ out, int64_t n0, int64_t n1) {
    const int64_t i4 = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t tot4 = (n0 + 2 * n1) >> 2;
    if (i4 >= tot4) return;
    const int64_t n0_4 = n0 >> 2, n1_4 = n1 >> 2;
    float4 v;
    if (i4 < n0_4) v = reinterpret_cast<const float4*>(w0)[i4];
    else if (i4 < n0_4 + n1_4) v = reinterpret_cast<const float4*>(w1)[i4 - n0_4];
    else v = reinterpret_cast<const float4*>(w2)[i4 - n0_4 - n1_4];
    const __half2 p0 = __halves2half2(__float2half_rn(v.x), __float2half_rn(v.y));
    const __half2 p1 = __halves2half2(__float2half_rn(v.z), __float2half_rn(v.w));
    uint2 u;
    memcpy(&u.x, &p0, 4);
    memcpy(&u.y, &p1, 4);
    reinterpret_cast<uint2*>(out)[i4] = u;
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

/// Fused single-pass multi-query mix (prefill, (N&3)==0, D<=8, n_q in {1,3,4}):
/// 1 CTA per token reads each ptrs.p[d] ONCE across all n_q queries (3x-4x less HBM traffic, 3 barriers total vs 3*D).
template <int NQ>
__global__ void k_mix_normed_fused(NormMixPtrs ptrs, int D, const float* __restrict__ q0, const float* __restrict__ q1,
                                   MixOuts4 out, int N, float inv_sqrt_n, int swa_broadcast) {
    __shared__ float s_dot[NQ][8][8];  // [q][d][warp]
    __shared__ float s_w[NQ][8];
    const int t = blockIdx.x;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int n4 = N >> 2;
    const float4* qp[NQ];
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
        if constexpr (NQ == 4) {
            qp[q] = (const float4*) (q < 2 ? (q0 + (size_t) q * N) : (q1 + (size_t) (q - 2) * N));
        } else {
            qp[q] = (const float4*) (q0 + (size_t) q * N);
        }
    }
    for (int d = 0; d < D; ++d) {
        const float4* s4 = (const float4*) (ptrs.p[d] + (size_t) t * N);
        float acc[NQ] = {};
        for (int i = tid; i < n4; i += 256) {
            const float4 v = s4[i];
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                const float4 b = qp[q][i];
                acc[q] += v.x * b.x + v.y * b.y + v.z * b.z + v.w * b.w;
            }
        }
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            float v = acc[q];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0) s_dot[q][d][warp] = v;
        }
    }
    __syncthreads();
    if (tid < NQ * D) {
        const int q = tid / D, d = tid % D;
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) s += s_dot[q][d][w];
        s_w[q][d] = s * inv_sqrt_n;
    }
    __syncthreads();
    if (tid < NQ) {
        const int q = tid;
        float m = -1e30f;
        for (int d = 0; d < D; ++d) m = fmaxf(m, s_w[q][d]);
        float sum = 0.0f;
        for (int d = 0; d < D; ++d) {
            s_w[q][d] = __expf(s_w[q][d] - m);
            sum += s_w[q][d];
        }
        const float inv = __fdividef(1.0f, sum);
        for (int d = 0; d < D; ++d) s_w[q][d] *= inv;
    }
    __syncthreads();
    if constexpr (NQ == 1) {
        if (swa_broadcast) {
            float4* o0 = (float4*) (out.p[0] + (size_t) t * N);
            float4* o1 = (float4*) (out.p[1] + (size_t) t * N);
            float4* o2 = (float4*) (out.p[2] + (size_t) t * N);
            for (int i = tid; i < n4; i += 256) {
                float ax = 0.0f, ay = 0.0f, az = 0.0f, aw = 0.0f;
                for (int d = 0; d < D; ++d) {
                    const float4 v = ((const float4*) (ptrs.p[d] + (size_t) t * N))[i];
                    const float wd = s_w[0][d];
                    ax += wd * v.x; ay += wd * v.y; az += wd * v.z; aw += wd * v.w;
                }
                const float4 r = make_float4(ax, ay, az, aw);
                o0[i] = r; o1[i] = r; o2[i] = r;
            }
            return;
        }
    }
    float4* dst[NQ];
#pragma unroll
    for (int q = 0; q < NQ; ++q) dst[q] = (float4*) (out.p[q] + (size_t) t * N);
    for (int i = tid; i < n4; i += 256) {
        float ax[NQ] = {}, ay[NQ] = {}, az[NQ] = {}, aw[NQ] = {};
        for (int d = 0; d < D; ++d) {
            const float4 v = ((const float4*) (ptrs.p[d] + (size_t) t * N))[i];
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                const float wd = s_w[q][d];
                ax[q] += wd * v.x; ay[q] += wd * v.y; az[q] += wd * v.z; aw[q] += wd * v.w;
            }
        }
#pragma unroll
        for (int q = 0; q < NQ; ++q) dst[q][i] = make_float4(ax[q], ay[q], az[q], aw[q]);
    }
}

/// Optional per-output inv_rms destination for k_mix_normed_fused_1pass (used by fused k_control_down_pp_wmma).
struct MixInvRms4 {
    float* p[4];
};

/// Single-pass register-cached multi-query mix for N == 2304 (n4 = 576, blockDim = 576 = 18 warps, D <= 8):
/// Reads each ptrs.p[d] ONCE into registers (ZERO second HBM/L2 pass!) and runs 18 warps/token.
template <int NQ, bool NORM_LAST = false>
__global__ void __launch_bounds__(576, (NQ >= 3) ? 1 : 2) k_mix_normed_fused_1pass(
    NormMixPtrs ptrs, int D, const float* __restrict__ q0, const float* __restrict__ q1,
    MixOuts4 out, int N, float inv_sqrt_n, int swa_broadcast, float rms_eps = 0.0f,
    MixInvRms4 ir_out = {{nullptr, nullptr, nullptr, nullptr}}) {
    __shared__ float s_dot[NQ][8][18];  // [q][d][warp]
    __shared__ float s_w[NQ][8];
    __shared__ float s_ss[18];
    const int t = blockIdx.x;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

    float4 cv[8];
#pragma unroll
    for (int d = 0; d < 8; ++d) {
        if (d < D) {
            const float4 v = reinterpret_cast<const float4*>(ptrs.p[d] + (size_t) t * N)[tid];
            cv[d] = v;
            if constexpr (NORM_LAST) {
                if (d == D - 1) {
                    float ss = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
                    if (lane == 0) s_ss[warp] = ss;
                }
            }
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                const float4* q_ptr = reinterpret_cast<const float4*>(
                    (NQ == 4 && q >= 2) ? (q1 + (size_t) (q - 2) * N) : (q0 + (size_t) q * N));
                const float4 qv = __ldg(q_ptr + tid);
                float dot = v.x * qv.x + v.y * qv.y + v.z * qv.z + v.w * qv.w;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
                if (lane == 0) s_dot[q][d][warp] = dot;
            }
        }
    }
    __syncthreads();
    if (warp < NQ) {
        const int q = warp;
        float wv = -1e30f;
        float inv_rms_local = 1.0f;
        if (lane < D) {
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < 18; ++w) s += s_dot[q][lane][w];
            if constexpr (NORM_LAST) {
                if (lane == D - 1) {
                    float ss = 0.0f;
#pragma unroll
                    for (int w = 0; w < 18; ++w) ss += s_ss[w];
                    inv_rms_local = rsqrtf(ss / (float) N + rms_eps);
                    s *= inv_rms_local;
                }
            }
            wv = s * inv_sqrt_n;
        }
        float m = wv;
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 4));
        const float ev = (lane < D) ? __expf(wv - m) : 0.0f;
        float sum = 0.0f;
#pragma unroll
        for (int d = 0; d < 8; ++d) {
            const float ed = __shfl_sync(0xffffffffu, ev, d);
            if (d < D) sum += ed;
        }
        if (lane < D) {
            float res = ev * __fdividef(1.0f, sum);
            if constexpr (NORM_LAST) {
                if (lane == D - 1) res *= inv_rms_local;
            }
            s_w[q][lane] = res;
        }
    }
    __syncthreads();
    if constexpr (NQ == 1) {
        float ax = 0.0f, ay = 0.0f, az = 0.0f, aw = 0.0f;
#pragma unroll
        for (int d = 0; d < 8; ++d) {
            if (d < D) {
                const float wd = s_w[0][d];
                ax += wd * cv[d].x; ay += wd * cv[d].y; az += wd * cv[d].z; aw += wd * cv[d].w;
            }
        }
        const float4 r = make_float4(ax, ay, az, aw);
        reinterpret_cast<float4*>(out.p[0] + (size_t) t * N)[tid] = r;
        if (swa_broadcast) {
            reinterpret_cast<float4*>(out.p[1] + (size_t) t * N)[tid] = r;
            reinterpret_cast<float4*>(out.p[2] + (size_t) t * N)[tid] = r;
        }
        if (ir_out.p[0]) {
            float ss = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
            if (lane == 0) s_ss[warp] = ss;
            __syncthreads();
            if (warp == 0) {
                float tot = (lane < 18) ? s_ss[lane] : 0.0f;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, o);
                if (lane == 0) ir_out.p[0][t] = rsqrtf(tot / (float) N + rms_eps);
            }
        }
        return;
    }
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
        float ax = 0.0f, ay = 0.0f, az = 0.0f, aw = 0.0f;
#pragma unroll
        for (int d = 0; d < 8; ++d) {
            if (d < D) {
                const float wd = s_w[q][d];
                ax += wd * cv[d].x; ay += wd * cv[d].y; az += wd * cv[d].z; aw += wd * cv[d].w;
            }
        }
        reinterpret_cast<float4*>(out.p[q] + (size_t) t * N)[tid] = make_float4(ax, ay, az, aw);
        if (ir_out.p[0]) {
            float ss = ax * ax + ay * ay + az * az + aw * aw;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
            if (lane == 0) s_dot[q][0][warp] = ss;
        }
    }
    if (ir_out.p[0]) {
        __syncthreads();
        if (warp < NQ) {
            float* ir_dst = (warp == 0) ? ir_out.p[0]
                          : (warp == 1) ? ir_out.p[1]
                          : (warp == 2) ? ir_out.p[2]
                                        : ir_out.p[3];
            if (ir_dst) {
                float tot = (lane < 18) ? s_dot[warp][0][lane] : 0.0f;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, o);
                if (lane == 0) ir_dst[t] = rsqrtf(tot / (float) N + rms_eps);
            }
        }
    }
}

/// 12 physical core layers x up to 4 query vectors for D=2 Virtual Loop-Bank (il = 18..41).
struct LoopBankQueries12 {
    const float* q[12][4]; // [layer 0..11][mod 0..3], each [N] FP32 (nullptr if unused)
};

/// Computes 2-block softmax weights w_out[((l*4+m)*T + t)*2 + 0..1] and exact
/// inv_rms_out[(l*4+m)*T + t] = rsqrt(mean(msg^2) + eps) for constant D=2 (bank0, bank1)
/// across all 12 layers x 4 modules in ONE pass!
/// Grid: (T), Block: 576 (for N == 2304).
__global__ void __launch_bounds__(576, 2) k_loop_bank_weights_pp(
    const float* __restrict__ bank0, const float* __restrict__ bank1,
    LoopBankQueries12 qs, float* __restrict__ w_out, float* __restrict__ inv_rms_out,
    int T, int N, float eps) {
    __shared__ const float* s_q[48];
    __shared__ float s_b[3][18];      // [ss0, ss1, dot01][warp]
    __shared__ float s_tot[3];        // [tot_ss0, tot_ss1, tot_dot01]
    __shared__ float s_dot[48][2][18]; // [l*4+m][0..1][warp]

    const int t = blockIdx.x;
    if (t >= T) return;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

    // Copy by-value query pointer table to SMEM via constant param offsets (guarantees STACK: 0).
    if (tid == 0) {
#pragma unroll
        for (int l = 0; l < 12; ++l) {
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                s_q[l * 4 + m] = qs.q[l][m];
            }
        }
    }

    const float4 b0 = reinterpret_cast<const float4*>(bank0 + (size_t) t * N)[tid];
    const float4 b1 = reinterpret_cast<const float4*>(bank1 + (size_t) t * N)[tid];

    float ss0 = b0.x * b0.x + b0.y * b0.y + b0.z * b0.z + b0.w * b0.w;
    float ss1 = b1.x * b1.x + b1.y * b1.y + b1.z * b1.z + b1.w * b1.w;
    float d01 = b0.x * b1.x + b0.y * b1.y + b0.z * b1.z + b0.w * b1.w;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        ss0 += __shfl_xor_sync(0xffffffffu, ss0, o);
        ss1 += __shfl_xor_sync(0xffffffffu, ss1, o);
        d01 += __shfl_xor_sync(0xffffffffu, d01, o);
    }
    if (lane == 0) {
        s_b[0][warp] = ss0;
        s_b[1][warp] = ss1;
        s_b[2][warp] = d01;
    }
    __syncthreads();

    if (tid < 3) {
        float tot = 0.0f;
#pragma unroll
        for (int w = 0; w < 18; ++w) tot += s_b[tid][w];
        s_tot[tid] = tot;
    }

    for (int idx = 0; idx < 48; ++idx) {
        const float* qp = s_q[idx];
        if (!qp) continue;
        const float4 qv = __ldg(reinterpret_cast<const float4*>(qp) + tid);
        float d0 = b0.x * qv.x + b0.y * qv.y + b0.z * qv.z + b0.w * qv.w;
        float d1 = b1.x * qv.x + b1.y * qv.y + b1.z * qv.z + b1.w * qv.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            d0 += __shfl_xor_sync(0xffffffffu, d0, o);
            d1 += __shfl_xor_sync(0xffffffffu, d1, o);
        }
        if (lane == 0) {
            s_dot[idx][0][warp] = d0;
            s_dot[idx][1][warp] = d1;
        }
    }
    __syncthreads();

    if (tid < 48) {
        if (s_q[tid] != nullptr) {
            float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
            for (int w = 0; w < 18; ++w) {
                s0 += s_dot[tid][0][w];
                s1 += s_dot[tid][1][w];
            }
            const float inv_sqrt_n = (N == 2304) ? (1.0f / 48.0f) : (1.0f / sqrtf((float) N));
            const float wv0 = s0 * inv_sqrt_n;
            const float wv1 = s1 * inv_sqrt_n;
            const float mx = fmaxf(wv0, wv1);
            const float e0 = __expf(wv0 - mx);
            const float e1 = __expf(wv1 - mx);
            const float inv_sum = __fdividef(1.0f, e0 + e1);
            const float w0 = e0 * inv_sum;
            const float w1 = e1 * inv_sum;
            const size_t base = (size_t) tid * (size_t) T + (size_t) t;
            *reinterpret_cast<float2*>(w_out + base * 2u) = make_float2(w0, w1);
            if (inv_rms_out) {
                const float tot_ss = fmaxf(0.0f, w0 * w0 * s_tot[0] + 2.0f * w0 * w1 * s_tot[2] + w1 * w1 * s_tot[1]);
                inv_rms_out[base] = rsqrtf(tot_ss / (float) N + eps);
            }
        }
    }
}

/// 3-in-1 fused k_rms_norm(last) + k_mix_normed + k_rms_norm_q8_1 for N == 2304 (576 threads = 72 sub8 blocks):
/// Eliminates BOTH pre-mix k_rms_norm AND post-mix k_rms_norm_q8_1 / k_rms_norm_3_q8_1 launches & HBM roundtrips!
struct NormQ8Out3 {
    const float *w0, *w1, *w2;
    void *xq0, *xq1, *xq2;
};

template <int NQ>
__global__ void __launch_bounds__(576, 2) k_mix_normed_fused_1pass_q8(
    NormMixPtrs ptrs, int D, const float* __restrict__ q0,
    MixOuts4 out, NormQ8Out3 nq8, int N, float inv_sqrt_n, float rms_eps) {
    __shared__ float s_dot[NQ][8][18];
    __shared__ float s_w[NQ][8];
    __shared__ float s_ss[18];
    __shared__ float s_inv_rms;
    const int t = blockIdx.x;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int blk_idx = t * 72 + (tid >> 3);
    const int sub_lane = tid & 7;
    float4 qv[NQ];
#pragma unroll
    for (int q = 0; q < NQ; ++q) qv[q] = reinterpret_cast<const float4*>(q0 + (size_t) q * N)[tid];

    float4 cv[8];
#pragma unroll
    for (int d = 0; d < 8; ++d) {
        if (d < D) {
            const float4 v = reinterpret_cast<const float4*>(ptrs.p[d] + (size_t) t * N)[tid];
            cv[d] = v;
            if (d == D - 1) {
                float ss = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
                if (lane == 0) s_ss[warp] = ss;
            }
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                float dot = v.x * qv[q].x + v.y * qv[q].y + v.z * qv[q].z + v.w * qv[q].w;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
                if (lane == 0) s_dot[q][d][warp] = dot;
            }
        }
    }
    __syncthreads();
    if constexpr (NQ == 1) {
        // D <= 8 fits completely inside warp 0: compute dot sums, inv_rms, and softmax with 0 extra __syncthreads()!
        if (warp == 0) {
            float v_w = -1e30f;
            if (lane < D) {
                float s = 0.0f;
#pragma unroll
                for (int w = 0; w < 18; ++w) s += s_dot[0][lane][w];
                if (lane == D - 1) {
                    float ss = 0.0f;
#pragma unroll
                    for (int w = 0; w < 18; ++w) ss += s_ss[w];
                    const float ir = rsqrtf(ss / (float) N + rms_eps);
                    s_inv_rms = ir;
                    s *= ir;
                }
                v_w = s * inv_sqrt_n;
                s_w[0][lane] = v_w;
            }
            __syncwarp();
            if (lane == 0) {
                float m = -1e30f;
                for (int d = 0; d < D; ++d) m = fmaxf(m, s_w[0][d]);
                float sum = 0.0f;
                for (int d = 0; d < D; ++d) {
                    s_w[0][d] = __expf(s_w[0][d] - m);
                    sum += s_w[0][d];
                }
                const float inv = __fdividef(1.0f, sum);
                for (int d = 0; d < D; ++d) s_w[0][d] *= inv;
                s_w[0][D - 1] *= s_inv_rms;
            }
        }
        __syncthreads();
        float ax = 0.0f, ay = 0.0f, az = 0.0f, aw = 0.0f;
#pragma unroll
        for (int d = 0; d < 8; ++d) {
            if (d < D) {
                const float wd = s_w[0][d];
                ax += wd * cv[d].x; ay += wd * cv[d].y; az += wd * cv[d].z; aw += wd * cv[d].w;
            }
        }
        const float4 r = make_float4(ax, ay, az, aw);
        float ss = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        if (lane == 0) s_ss[warp] = ss;
        __syncthreads();
        if (tid == 0) {
            float tot = 0.0f;
#pragma unroll
            for (int w = 0; w < 18; ++w) tot += s_ss[w];
            s_inv_rms = rsqrtf(tot / (float) N + rms_eps);
        }
        __syncthreads();
        const float ir = s_inv_rms;
        float4 o;
        if (nq8.w0) {
            const float4 ww = reinterpret_cast<const float4*>(nq8.w0)[tid];
            o = make_float4(r.x * ir * ww.x, r.y * ir * ww.y, r.z * ir * ww.z, r.w * ir * ww.w);
        } else {
            o = make_float4(r.x * ir, r.y * ir, r.z * ir, r.w * ir);
        }
        reinterpret_cast<float4*>(out.p[0] + (size_t) t * N)[tid] = o;
        if (nq8.xq0) q8_1_sub8_store(o, nq8.xq0, blk_idx, sub_lane);
        return;
    }

    // NQ == 3: NQ*D <= 24 fits in warp 0; compute all 3 softmaxes & all 3 output norms in 2 total barriers (was 8)!
    if (warp == 0) {
        if (lane < NQ * D) {
            const int q = lane / D, d = lane % D;
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < 18; ++w) s += s_dot[q][d][w];
            if (d == D - 1) {
                float ss = 0.0f;
#pragma unroll
                for (int w = 0; w < 18; ++w) ss += s_ss[w];
                const float ir = rsqrtf(ss / (float) N + rms_eps);
                if (q == 0) s_inv_rms = ir;
                s *= ir;
            }
            s_w[q][d] = s * inv_sqrt_n;
        }
        __syncwarp();
        if (lane < NQ) {
            const int q = lane;
            float m = -1e30f;
            for (int d = 0; d < D; ++d) m = fmaxf(m, s_w[q][d]);
            float sum = 0.0f;
            for (int d = 0; d < D; ++d) {
                s_w[q][d] = __expf(s_w[q][d] - m);
                sum += s_w[q][d];
            }
            const float inv = __fdividef(1.0f, sum);
            for (int d = 0; d < D; ++d) s_w[q][d] *= inv;
            s_w[q][D - 1] *= s_inv_rms;
        }
    }
    __syncthreads();
    float ax[NQ] = {}, ay[NQ] = {}, az[NQ] = {}, aw[NQ] = {};
#pragma unroll
    for (int d = 0; d < 8; ++d) {
        if (d < D) {
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                const float wd = s_w[q][d];
                ax[q] += wd * cv[d].x; ay[q] += wd * cv[d].y; az[q] += wd * cv[d].z; aw[q] += wd * cv[d].w;
            }
        }
    }
    __shared__ float s_ss3[NQ][18];
    __shared__ float s_ir3[NQ];
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
        float ss = ax[q] * ax[q] + ay[q] * ay[q] + az[q] * az[q] + aw[q] * aw[q];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        if (lane == 0) s_ss3[q][warp] = ss;
    }
    __syncthreads();
    if (tid < NQ) {
        float tot = 0.0f;
#pragma unroll
        for (int w = 0; w < 18; ++w) tot += s_ss3[tid][w];
        s_ir3[tid] = rsqrtf(tot / (float) N + rms_eps);
    }
    __syncthreads();
    auto store_q = [&](int q, const float* nw, float* y_dst, void* xq_dst) {
        const float ir = s_ir3[q];
        float4 o;
        if (nw) {
            const float4 ww = reinterpret_cast<const float4*>(nw)[tid];
            o = make_float4(ax[q] * ir * ww.x, ay[q] * ir * ww.y, az[q] * ir * ww.z, aw[q] * ir * ww.w);
        } else {
            o = make_float4(ax[q] * ir, ay[q] * ir, az[q] * ir, aw[q] * ir);
        }
        reinterpret_cast<float4*>(y_dst + (size_t) t * N)[tid] = o;
        if (xq_dst) q8_1_sub8_store(o, xq_dst, blk_idx, sub_lane);
    };
    store_q(0, nq8.w0, out.p[0], nq8.xq0);
    if constexpr (NQ >= 2) store_q(1, nq8.w1, out.p[1], nq8.xq1);
    if constexpr (NQ >= 3) store_q(2, nq8.w2, out.p[2], nq8.xq2);
}

/// 3-in-1 fused k_rms_norm(last) + k_mix_normed + k_rms_norm[_3]_f16 for Prefill (!has_msg, N == 2304, 576 threads):
/// Eliminates BOTH pre-mix k_rms_norm AND post-mix k_rms_norm[_3]_f16 launches & ~1.5 GB/chunk HBM roundtrips!
struct NormF16Out3 {
    const float *w0, *w1, *w2;
    uint16_t *y16_0, *y16_1, *y16_2;
};

template <int NQ>
__global__ void __launch_bounds__(576, 2) k_mix_normed_fused_1pass_f16(
    NormMixPtrs ptrs, int D, const float* __restrict__ q0,
    MixOuts4 out, NormF16Out3 nf16, int N, float inv_sqrt_n, float rms_eps) {
    __shared__ float s_dot[NQ][8][18];
    __shared__ float s_w[NQ][8];
    __shared__ float s_ss[18];
    __shared__ float s_inv_rms;
    const int t = blockIdx.x;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

    float4 cv[8];
#pragma unroll
    for (int d = 0; d < 8; ++d) {
        if (d < D) {
            const float4 v = reinterpret_cast<const float4*>(ptrs.p[d] + (size_t) t * N)[tid];
            cv[d] = v;
            if (d == D - 1) {
                float ss = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
                if (lane == 0) s_ss[warp] = ss;
            }
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                const float4 qv = __ldg(reinterpret_cast<const float4*>(q0 + (size_t) q * N) + tid);
                float dot = v.x * qv.x + v.y * qv.y + v.z * qv.z + v.w * qv.w;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
                if (lane == 0) s_dot[q][d][warp] = dot;
            }
        }
    }
    __syncthreads();
    if constexpr (NQ == 1) {
        if (warp == 0) {
            if (lane < D) {
                float s = 0.0f;
#pragma unroll
                for (int w = 0; w < 18; ++w) s += s_dot[0][lane][w];
                if (lane == D - 1) {
                    float ss = 0.0f;
#pragma unroll
                    for (int w = 0; w < 18; ++w) ss += s_ss[w];
                    const float ir = rsqrtf(ss / (float) N + rms_eps);
                    s_inv_rms = ir;
                    s *= ir;
                }
                s_w[0][lane] = s * inv_sqrt_n;
            }
            __syncwarp();
            if (lane == 0) {
                float m = -1e30f;
                for (int d = 0; d < D; ++d) m = fmaxf(m, s_w[0][d]);
                float sum = 0.0f;
                for (int d = 0; d < D; ++d) {
                    s_w[0][d] = __expf(s_w[0][d] - m);
                    sum += s_w[0][d];
                }
                const float inv = __fdividef(1.0f, sum);
                for (int d = 0; d < D; ++d) s_w[0][d] *= inv;
                s_w[0][D - 1] *= s_inv_rms;
            }
        }
        __syncthreads();
        float ax = 0.0f, ay = 0.0f, az = 0.0f, aw = 0.0f;
#pragma unroll
        for (int d = 0; d < 8; ++d) {
            if (d < D) {
                const float wd = s_w[0][d];
                ax += wd * cv[d].x; ay += wd * cv[d].y; az += wd * cv[d].z; aw += wd * cv[d].w;
            }
        }
        const float4 r = make_float4(ax, ay, az, aw);
        float ss = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        if (lane == 0) s_ss[warp] = ss;
        __syncthreads();
        if (tid == 0) {
            float tot = 0.0f;
#pragma unroll
            for (int w = 0; w < 18; ++w) tot += s_ss[w];
            s_inv_rms = rsqrtf(tot / (float) N + rms_eps);
        }
        __syncthreads();
        const float ir = s_inv_rms;
        float4 o;
        if (nf16.w0) {
            const float4 ww = __ldg(reinterpret_cast<const float4*>(nf16.w0) + tid);
            o = make_float4(r.x * ir * ww.x, r.y * ir * ww.y, r.z * ir * ww.z, r.w * ir * ww.w);
        } else {
            o = make_float4(r.x * ir, r.y * ir, r.z * ir, r.w * ir);
        }
        if (out.p[0]) reinterpret_cast<float4*>(out.p[0] + (size_t) t * N)[tid] = o;
        if (nf16.y16_0) {
            const __half2 p0 = __halves2half2(__float2half_rn(o.x), __float2half_rn(o.y));
            const __half2 p1 = __halves2half2(__float2half_rn(o.z), __float2half_rn(o.w));
            uint2 u;
            memcpy(&u.x, &p0, 4);
            memcpy(&u.y, &p1, 4);
#if defined(__CUDA_ARCH__)
            uint2* dst_u2 = reinterpret_cast<uint2*>(nf16.y16_0 + (size_t) t * N) + tid;
            asm volatile("st.global.cs.v2.u32 [%0], {%1, %2};" :: "l"(dst_u2), "r"(u.x), "r"(u.y));
#else
            reinterpret_cast<uint2*>(nf16.y16_0 + (size_t) t * N)[tid] = u;
#endif
        }
        return;
    }

    if (warp == 0) {
        if (lane < NQ * D) {
            const int q = lane / D, d = lane % D;
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < 18; ++w) s += s_dot[q][d][w];
            if (d == D - 1) {
                float ss = 0.0f;
#pragma unroll
                for (int w = 0; w < 18; ++w) ss += s_ss[w];
                const float ir = rsqrtf(ss / (float) N + rms_eps);
                if (q == 0) s_inv_rms = ir;
                s *= ir;
            }
            s_w[q][d] = s * inv_sqrt_n;
        }
        __syncwarp();
        if (lane < NQ) {
            const int q = lane;
            float m = -1e30f;
            for (int d = 0; d < D; ++d) m = fmaxf(m, s_w[q][d]);
            float sum = 0.0f;
            for (int d = 0; d < D; ++d) {
                s_w[q][d] = __expf(s_w[q][d] - m);
                sum += s_w[q][d];
            }
            const float inv = __fdividef(1.0f, sum);
            for (int d = 0; d < D; ++d) s_w[q][d] *= inv;
            s_w[q][D - 1] *= s_inv_rms;
        }
    }
    __syncthreads();
    float ax[NQ] = {}, ay[NQ] = {}, az[NQ] = {}, aw[NQ] = {};
#pragma unroll
    for (int d = 0; d < 8; ++d) {
        if (d < D) {
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                const float wd = s_w[q][d];
                ax[q] += wd * cv[d].x; ay[q] += wd * cv[d].y; az[q] += wd * cv[d].z; aw[q] += wd * cv[d].w;
            }
        }
    }
    __shared__ float s_ss3[NQ][18];
    __shared__ float s_ir3[NQ];
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
        float ss = ax[q] * ax[q] + ay[q] * ay[q] + az[q] * az[q] + aw[q] * aw[q];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        if (lane == 0) s_ss3[q][warp] = ss;
    }
    __syncthreads();
    if (tid < NQ) {
        float tot = 0.0f;
#pragma unroll
        for (int w = 0; w < 18; ++w) tot += s_ss3[tid][w];
        s_ir3[tid] = rsqrtf(tot / (float) N + rms_eps);
    }
    __syncthreads();
    auto store_q = [&](int q, const float* nw, float* y_dst, uint16_t* y16_dst) {
        const float ir = s_ir3[q];
        float4 o;
        if (nw) {
            const float4 ww = __ldg(reinterpret_cast<const float4*>(nw) + tid);
            o = make_float4(ax[q] * ir * ww.x, ay[q] * ir * ww.y, az[q] * ir * ww.z, aw[q] * ir * ww.w);
        } else {
            o = make_float4(ax[q] * ir, ay[q] * ir, az[q] * ir, aw[q] * ir);
        }
        if (y_dst) reinterpret_cast<float4*>(y_dst + (size_t) t * N)[tid] = o;
        if (y16_dst) {
            const __half2 p0 = __halves2half2(__float2half_rn(o.x), __float2half_rn(o.y));
            const __half2 p1 = __halves2half2(__float2half_rn(o.z), __float2half_rn(o.w));
            uint2 u;
            memcpy(&u.x, &p0, 4);
            memcpy(&u.y, &p1, 4);
            reinterpret_cast<uint2*>(y16_dst + (size_t) t * N)[tid] = u;
        }
    };
    store_q(0, nf16.w0, out.p[0], nf16.y16_0);
    if constexpr (NQ >= 2) store_q(1, nf16.w1, out.p[1], nf16.y16_1);
    if constexpr (NQ >= 3) store_q(2, nf16.w2, out.p[2], nf16.y16_2);
}

/// 4-in-1 fused T=1 k_shortconv(wo -> ffn_inp, res=cur) + k_mix_normed_fused_1pass_q8<1>(..., last=ffn_inp -> in_ffn, xq):
/// Eliminates k_shortconv launch AND ffn_inp global read on !do_push && !has_msg && !save_conv_trace layers (24/48 layers)!
__global__ void __launch_bounds__(576, 2) k_shortconv_mix_normed_fused_1pass_q8_t1(
    const float* __restrict__ wo, const float* __restrict__ w_conv, float* __restrict__ hist,
    const float* __restrict__ res_cur, float* __restrict__ ffn_inp_out, int hist_offset,
    NormMixPtrs ptrs, int D, const float* __restrict__ q0,
    float* __restrict__ out0, const float* __restrict__ nw0, void* __restrict__ xq0,
    int N, float inv_sqrt_n, float rms_eps) {
    __shared__ float s_dot[8][18];
    __shared__ float s_w[8];
    __shared__ float s_ss[18];
    __shared__ float s_inv_rms;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int blk_idx = tid >> 3;
    const int sub_lane = tid & 7;
    const float4 qv = reinterpret_cast<const float4*>(q0)[tid];

    // 1. Compute 4 channels of post-attn shortconv + residual add in registers, update hist, write ffn_inp_out
    const float4 wov = reinterpret_cast<const float4*>(wo)[tid];
    const float4 rcv = reinterpret_cast<const float4*>(res_cur)[tid];
    const float scale = rsqrtf(5.0f);
    const int c0 = tid * 4;
    auto do_conv1 = [&](int ci, float x, float res) -> float {
        float* hc = hist + (size_t) (hist_offset + ci) * 3;
        const float h0 = hc[0], h1 = hc[1], h2 = hc[2];
        const float4 w4 = *reinterpret_cast<const float4*>(w_conv + (size_t) ci * 4);
        const float s = h0 * w4.x + h1 * w4.y + h2 * w4.z + x * w4.w;
        hc[0] = h1; hc[1] = h2; hc[2] = x;
        return (x + s) * scale + res;
    };
    const float4 v_last = make_float4(
        do_conv1(c0 + 0, wov.x, rcv.x),
        do_conv1(c0 + 1, wov.y, rcv.y),
        do_conv1(c0 + 2, wov.z, rcv.z),
        do_conv1(c0 + 3, wov.w, rcv.w));
    reinterpret_cast<float4*>(ffn_inp_out)[tid] = v_last;

    // 2. Mix over [norm_blocks..., v_last] (exact same arithmetic as k_mix_normed_fused_1pass_q8<1>)
    float4 cv[8];
#pragma unroll
    for (int d = 0; d < 8; ++d) {
        if (d < D) {
            const float4 v = (d == D - 1) ? v_last : reinterpret_cast<const float4*>(ptrs.p[d])[tid];
            cv[d] = v;
            if (d == D - 1) {
                float ss = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
                if (lane == 0) s_ss[warp] = ss;
            }
            float dot = v.x * qv.x + v.y * qv.y + v.z * qv.z + v.w * qv.w;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
            if (lane == 0) s_dot[d][warp] = dot;
        }
    }
    __syncthreads();
    if (warp == 0) {
        if (lane < D) {
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < 18; ++w) s += s_dot[lane][w];
            if (lane == D - 1) {
                float ss = 0.0f;
#pragma unroll
                for (int w = 0; w < 18; ++w) ss += s_ss[w];
                const float ir = rsqrtf(ss / (float) N + rms_eps);
                s_inv_rms = ir;
                s *= ir;
            }
            s_w[lane] = s * inv_sqrt_n;
        }
        __syncwarp();
        if (lane == 0) {
            float m = -1e30f;
            for (int d = 0; d < D; ++d) m = fmaxf(m, s_w[d]);
            float sum = 0.0f;
            for (int d = 0; d < D; ++d) {
                s_w[d] = __expf(s_w[d] - m);
                sum += s_w[d];
            }
            const float inv = __fdividef(1.0f, sum);
            for (int d = 0; d < D; ++d) s_w[d] *= inv;
            s_w[D - 1] *= s_inv_rms;
        }
    }
    __syncthreads();
    float ax = 0.0f, ay = 0.0f, az = 0.0f, aw = 0.0f;
#pragma unroll
    for (int d = 0; d < 8; ++d) {
        if (d < D) {
            const float wd = s_w[d];
            ax += wd * cv[d].x; ay += wd * cv[d].y; az += wd * cv[d].z; aw += wd * cv[d].w;
        }
    }
    const float4 r = make_float4(ax, ay, az, aw);
    float ss = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if (lane == 0) s_ss[warp] = ss;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
#pragma unroll
        for (int w = 0; w < 18; ++w) tot += s_ss[w];
        s_inv_rms = rsqrtf(tot / (float) N + rms_eps);
    }
    __syncthreads();
    const float ir = s_inv_rms;
    float4 o;
    if (nw0) {
        const float4 ww = reinterpret_cast<const float4*>(nw0)[tid];
        o = make_float4(r.x * ir * ww.x, r.y * ir * ww.y, r.z * ir * ww.z, r.w * ir * ww.w);
    } else {
        o = make_float4(r.x * ir, r.y * ir, r.z * ir, r.w * ir);
    }
    reinterpret_cast<float4*>(out0)[tid] = o;
    if (xq0) q8_1_sub8_store(o, xq0, blk_idx, sub_lane);
}

/// Fuses post-attn k_shortconv_pp(wo -> ffn_inp, res = do_push ? nullptr : cur) + optional do_push rms_norm(ffn_inp -> ndst)
/// + pre-FFN k_mix_normed_fused_1pass[_f16]<1>(blocks[0..B-1], last=ffn_inp -> out_f32, optional out_f16, optional inv_rms)!
/// Eliminates reading ffn_inp (36 MB x 48 = 1.72 GB) and ndst (8 x 36 MB = 0.29 GB) back from HBM!
/// Grid: (T), Block: 576 (N == 2304, 18 warps).
template <bool WRITE_F16, bool NORM_LAST, int B_VAL = -1>
__global__ void __launch_bounds__(576, 2) k_shortconv_mix_normed_fused_1pass_pp(
    const float* __restrict__ wo, const float* __restrict__ w_conv, const float* __restrict__ conv_hist,
    const float* __restrict__ cur_res,     // non-null when !do_push (ffn_inp = sconv(wo) + cur_res); null when do_push
    float* __restrict__ ffn_inp_out,       // [T, N] FP32 written (still needed by MoE post-conv residual!)
    float* __restrict__ ndst_out,          // non-null when do_push: receives rms_norm(ffn_inp)
    NormMixPtrs ptrs, int B,               // B already-normed history blocks (excluding last = ffn_inp)
    const float* __restrict__ q_mix,       // [N] pre-FFN mix query (L.ffn_res_query)
    float* __restrict__ out_f32,           // [T, N] FP32 output (in_ffn)
    const float* __restrict__ f16_rms_w,   // [N] RMSNorm weight for out_f16 (when WRITE_F16)
    uint16_t* __restrict__ out_f16,        // [T, N] FP16 output via st.global.cs (when WRITE_F16)
    float* __restrict__ inv_rms_out,       // [T] optional inv_rms of out_f32 (when !WRITE_F16 && inv_rms_out != nullptr)
    int hist_offset, int T, int N, float eps, int t_offset = 0, bool local_out = false,
    __half* __restrict__ out_lo_f16 = nullptr) {
    __shared__ float s_dot[8][18];
    __shared__ float s_w[8];
    __shared__ float s_ss[18];
    __shared__ float s_inv_rms;

    const int t_loc = (int) blockIdx.x;
    if (t_loc >= T) return;
    const int t = t_offset + t_loc;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int c0 = tid << 2;

    // Step 1: Post-attn shortconv + optional residual add in registers (wo[t-3..t] hits L2 across adjacent CTAs)
    const float4 x = reinterpret_cast<const float4*>(wo + (size_t) t * N)[tid];
    float4 h0, h1, h2;
    if (t >= 3) {
        // Streaming ldcs_f4 on final tap h0 (t-3) so sliding window stops evicting L2-pinned activations!
        h0 = ldcs_f4(reinterpret_cast<const float4*>(wo + (size_t) (t - 3) * N) + tid);
        h1 = reinterpret_cast<const float4*>(wo + (size_t) (t - 2) * N)[tid];
        h2 = reinterpret_cast<const float4*>(wo + (size_t) (t - 1) * N)[tid];
    } else {
        const float* hc0 = conv_hist + (size_t) (hist_offset + c0 + 0) * 3;
        const float* hc1 = conv_hist + (size_t) (hist_offset + c0 + 1) * 3;
        const float* hc2 = conv_hist + (size_t) (hist_offset + c0 + 2) * 3;
        const float* hc3 = conv_hist + (size_t) (hist_offset + c0 + 3) * 3;
        if (t == 0) {
            h0 = make_float4(hc0[0], hc1[0], hc2[0], hc3[0]);
            h1 = make_float4(hc0[1], hc1[1], hc2[1], hc3[1]);
            h2 = make_float4(hc0[2], hc1[2], hc2[2], hc3[2]);
        } else if (t == 1) {
            h0 = make_float4(hc0[1], hc1[1], hc2[1], hc3[1]);
            h1 = make_float4(hc0[2], hc1[2], hc2[2], hc3[2]);
            h2 = reinterpret_cast<const float4*>(wo)[tid];
        } else {
            h0 = make_float4(hc0[2], hc1[2], hc2[2], hc3[2]);
            h1 = reinterpret_cast<const float4*>(wo)[tid];
            h2 = reinterpret_cast<const float4*>(wo + (size_t) N)[tid];
        }
    }
    const float4 wc0 = __ldg(reinterpret_cast<const float4*>(w_conv + (size_t) (c0 + 0) * 4));
    const float4 wc1 = __ldg(reinterpret_cast<const float4*>(w_conv + (size_t) (c0 + 1) * 4));
    const float4 wc2 = __ldg(reinterpret_cast<const float4*>(w_conv + (size_t) (c0 + 2) * 4));
    const float4 wc3 = __ldg(reinterpret_cast<const float4*>(w_conv + (size_t) (c0 + 3) * 4));
    const float scale = rsqrtf(5.0f);
    float4 fi;
    fi.x = (x.x + (h0.x * wc0.x + h1.x * wc0.y + h2.x * wc0.z + x.x * wc0.w)) * scale;
    fi.y = (x.y + (h0.y * wc1.x + h1.y * wc1.y + h2.y * wc1.z + x.y * wc1.w)) * scale;
    fi.z = (x.z + (h0.z * wc2.x + h1.z * wc2.y + h2.z * wc2.z + x.z * wc2.w)) * scale;
    fi.w = (x.w + (h0.w * wc3.x + h1.w * wc3.y + h2.w * wc3.z + x.w * wc3.w)) * scale;
    if (cur_res) {
        const float4 r = ldcs_f4(reinterpret_cast<const float4*>(cur_res + (size_t) t * N) + tid);
        fi.x += r.x; fi.y += r.y; fi.z += r.z; fi.w += r.w;
    }
    // Streaming store (st.global.cs) for cold ffn_inp_out so out_f32 (in_ffn) stays hot in L2 for Router!
    stcs_f4(reinterpret_cast<float4*>(ffn_inp_out + (size_t) t * N) + tid, fi);

    // Step 2 + 3: Exact 1-pass RMSNorm(fi) + pre-FFN mix_normed dot products (bit-exact with k_mix_normed_fused_1pass[_f16]<1>)
    const int b_cnt = (B_VAL >= 0) ? B_VAL : B;
    constexpr int CV_CAP = (B_VAL >= 0) ? ((B_VAL > 0) ? B_VAL : 1) : 7;
    float4 cv[CV_CAP];
    if constexpr (B_VAL > 0) {
#pragma unroll
        for (int d = 0; d < B_VAL; ++d) {
            cv[d] = ldcs_f4(reinterpret_cast<const float4*>(ptrs.p[d] + (size_t) t * N) + tid);
        }
    }

    const float4 qv = __ldg(reinterpret_cast<const float4*>(q_mix) + tid);
    if (NORM_LAST || ndst_out != nullptr) {
        float ss_fi = fi.x * fi.x + fi.y * fi.y + fi.z * fi.z + fi.w * fi.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss_fi += __shfl_xor_sync(0xffffffffu, ss_fi, o);
        if (lane == 0) s_ss[warp] = ss_fi;
    }
    {
        float dot_last = fi.x * qv.x + fi.y * qv.y + fi.z * qv.z + fi.w * qv.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) dot_last += __shfl_xor_sync(0xffffffffu, dot_last, o);
        if (lane == 0) s_dot[b_cnt][warp] = dot_last;
    }

    if constexpr (B_VAL > 0) {
#pragma unroll
        for (int d = 0; d < B_VAL; ++d) {
            const float4 v = cv[d];
            float dot = v.x * qv.x + v.y * qv.y + v.z * qv.z + v.w * qv.w;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
            if (lane == 0) s_dot[d][warp] = dot;
        }
    } else if constexpr (B_VAL < 0) {
#pragma unroll
        for (int d = 0; d < 7; ++d) {
            if (d < B) {
                const float4 v = ldcs_f4(reinterpret_cast<const float4*>(ptrs.p[d] + (size_t) t * N) + tid);
                cv[d] = v;
                float dot = v.x * qv.x + v.y * qv.y + v.z * qv.z + v.w * qv.w;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
                if (lane == 0) s_dot[d][warp] = dot;
            }
        }
    }
    __syncthreads();

    const int D = b_cnt + 1;
    const float inv_sqrt_n = (N == 2304) ? (1.0f / 48.0f) : (1.0f / sqrtf((float) N));
    if (warp == 0) {
        if (lane < D) {
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < 18; ++w) s += s_dot[lane][w];
            if (lane == b_cnt && (NORM_LAST || ndst_out != nullptr)) {
                float ss = 0.0f;
#pragma unroll
                for (int w = 0; w < 18; ++w) ss += s_ss[w];
                const float ir = rsqrtf(ss / (float) N + eps);
                s_inv_rms = ir;
                if constexpr (NORM_LAST) s *= ir;
            }
            s_w[lane] = s * inv_sqrt_n;
        }
        __syncwarp();
        if (lane == 0) {
            float m = -1e30f;
            for (int d = 0; d < D; ++d) m = fmaxf(m, s_w[d]);
            float sum = 0.0f;
            for (int d = 0; d < D; ++d) {
                s_w[d] = __expf(s_w[d] - m);
                sum += s_w[d];
            }
            const float inv = __fdividef(1.0f, sum);
            for (int d = 0; d < D; ++d) s_w[d] *= inv;
            if constexpr (NORM_LAST) s_w[b_cnt] *= s_inv_rms;
        }
    }
    __syncthreads();

    if (ndst_out != nullptr) {
        const float r_fi = s_inv_rms;
        stcs_f4(reinterpret_cast<float4*>(ndst_out + (size_t) t * N) + tid,
                make_float4(fi.x * r_fi, fi.y * r_fi, fi.z * r_fi, fi.w * r_fi));
    }

    float ax = 0.0f, ay = 0.0f, az = 0.0f, aw = 0.0f;
    if constexpr (B_VAL > 0) {
#pragma unroll
        for (int d = 0; d < B_VAL; ++d) {
            const float wd = s_w[d];
            ax += wd * cv[d].x; ay += wd * cv[d].y; az += wd * cv[d].z; aw += wd * cv[d].w;
        }
    } else if constexpr (B_VAL < 0) {
#pragma unroll
        for (int d = 0; d < 7; ++d) {
            if (d < B) {
                const float wd = s_w[d];
                ax += wd * cv[d].x; ay += wd * cv[d].y; az += wd * cv[d].z; aw += wd * cv[d].w;
            }
        }
    }
    {
        const float wb = s_w[b_cnt];
        ax += wb * fi.x; ay += wb * fi.y; az += wb * fi.z; aw += wb * fi.w;
    }
    const float4 r = make_float4(ax, ay, az, aw);

    if constexpr (WRITE_F16) {
        float ss = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        if (lane == 0) s_ss[warp] = ss;
        __syncthreads();
        if (tid == 0) {
            float tot = 0.0f;
#pragma unroll
            for (int w = 0; w < 18; ++w) tot += s_ss[w];
            s_inv_rms = rsqrtf(tot / (float) N + eps);
        }
        __syncthreads();
        const float ir = s_inv_rms;
        float4 o;
        if (f16_rms_w) {
            const float4 ww = __ldg(reinterpret_cast<const float4*>(f16_rms_w) + tid);
            o = make_float4(r.x * ir * ww.x, r.y * ir * ww.y, r.z * ir * ww.z, r.w * ir * ww.w);
        } else {
            o = make_float4(r.x * ir, r.y * ir, r.z * ir, r.w * ir);
        }
        const int t_o = local_out ? t_loc : t;
        if (out_f32 != nullptr) reinterpret_cast<float4*>(out_f32 + (size_t) t_o * N)[tid] = o;
        if (out_f16 != nullptr) {
            const __half h0 = __float2half_rn(o.x);
            const __half h1 = __float2half_rn(o.y);
            const __half h2 = __float2half_rn(o.z);
            const __half h3 = __float2half_rn(o.w);
            const __half2 p0 = __halves2half2(h0, h1);
            const __half2 p1 = __halves2half2(h2, h3);
            uint2 u;
            memcpy(&u.x, &p0, 4);
            memcpy(&u.y, &p1, 4);
            if (out_lo_f16 != nullptr) {
                // Normal L2-cached stores so immediately following Pre-Split Ozaki-2 Router hits L2 100%!
                reinterpret_cast<uint2*>(out_f16 + (size_t) t * N)[tid] = u;
                const __half2 l0 = __halves2half2(__float2half_rn((o.x - __half2float(h0)) * 2048.0f),
                                                  __float2half_rn((o.y - __half2float(h1)) * 2048.0f));
                const __half2 l1 = __halves2half2(__float2half_rn((o.z - __half2float(h2)) * 2048.0f),
                                                  __float2half_rn((o.w - __half2float(h3)) * 2048.0f));
                uint2 ul;
                memcpy(&ul.x, &l0, 4);
                memcpy(&ul.y, &l1, 4);
                reinterpret_cast<uint2*>(out_lo_f16 + (size_t) t_o * N)[tid] = ul;
            } else {
                stcs_u2(reinterpret_cast<uint2*>(out_f16 + (size_t) t * N) + tid, u);
            }
        }
    } else {
        const int t_o = local_out ? t_loc : t;
        if (out_f32) reinterpret_cast<float4*>(out_f32 + (size_t) t_o * N)[tid] = r;
        if (inv_rms_out) {
            float ss = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
            if (lane == 0) s_ss[warp] = ss;
            __syncthreads();
            if (warp == 0) {
                float tot = (lane < 18) ? s_ss[lane] : 0.0f;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, o);
                if (lane == 0) inv_rms_out[t_o] = rsqrtf(tot / (float) N + eps);
            }
        }
    }
}

/// Fused k_control_prep + loop_down ([2*N -> R=64] F32 GEMM) for small T (1..8):
/// Exploits linearity of RMSNorm: dot(cur*rc, W_c) + dot(msg*rm, W_m) = rc*dot(cur, W_c) + rm*dot(msg, W_m).
/// Eliminates k_control_prep launch and all ctrl2N HBM writes/reads! Grid (R), block 256.
template <int MT, bool EXACT_T>
__global__ void k_loop_down_fused_smallT(const float* __restrict__ W, const float* __restrict__ cur,
                                         const float* __restrict__ msg, float* __restrict__ Y,
                                         int N, int R, int T, float eps) {
    __shared__ float s_dc[MT][8];
    __shared__ float s_sc[MT][8];
    __shared__ float s_dm[MT][8];
    __shared__ float s_sm[MT][8];
    const int o = blockIdx.x;
    if (o >= R) return;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int n4 = N >> 2;
    const float4* wc4 = reinterpret_cast<const float4*>(W + (size_t) o * (2 * N));
    const float4* wm4 = reinterpret_cast<const float4*>(W + (size_t) o * (2 * N) + N);
    float dc[MT], sc[MT], dm[MT], sm[MT];
#pragma unroll
    for (int t = 0; t < MT; ++t) { dc[t] = 0.0f; sc[t] = 0.0f; dm[t] = 0.0f; sm[t] = 0.0f; }
    for (int i = tid; i < n4; i += 256) {
        const float4 awc = wc4[i];
        const float4 awm = wm4[i];
#pragma unroll
        for (int t = 0; t < MT; ++t) {
            if (EXACT_T || t < T) {
                const float4 vc = reinterpret_cast<const float4*>(cur + (size_t) t * N)[i];
                const float4 vm = reinterpret_cast<const float4*>(msg + (size_t) t * N)[i];
                dc[t] += awc.x * vc.x + awc.y * vc.y + awc.z * vc.z + awc.w * vc.w;
                sc[t] += vc.x * vc.x + vc.y * vc.y + vc.z * vc.z + vc.w * vc.w;
                dm[t] += awm.x * vm.x + awm.y * vm.y + awm.z * vm.z + awm.w * vm.w;
                sm[t] += vm.x * vm.x + vm.y * vm.y + vm.z * vm.z + vm.w * vm.w;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < MT; ++t) {
        if (EXACT_T || t < T) {
            float vdc = dc[t], vsc = sc[t], vdm = dm[t], vsm = sm[t];
#pragma unroll
            for (int m = 16; m > 0; m >>= 1) {
                vdc += __shfl_xor_sync(0xffffffffu, vdc, m);
                vsc += __shfl_xor_sync(0xffffffffu, vsc, m);
                vdm += __shfl_xor_sync(0xffffffffu, vdm, m);
                vsm += __shfl_xor_sync(0xffffffffu, vsm, m);
            }
            if (lane == 0) {
                s_dc[t][warp] = vdc; s_sc[t][warp] = vsc;
                s_dm[t][warp] = vdm; s_sm[t][warp] = vsm;
            }
        }
    }
    __syncthreads();
    if (tid < (EXACT_T ? MT : T)) {
        float tdc = 0.0f, tsc = 0.0f, tdm = 0.0f, tsm = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            tdc += s_dc[tid][w]; tsc += s_sc[tid][w];
            tdm += s_dm[tid][w]; tsm += s_sm[tid][w];
        }
        const float rc = rsqrtf(tsc / (float) N + eps);
        const float rm = rsqrtf(tsm / (float) N + eps);
        Y[(size_t) tid * R + o] = tdc * rc + tdm * rm;
    }
}

/// 3-module batched k_loop_down_fused_smallT for Q/K/V loop controllers (grid = (R=64, 3) = 192 blocks -> 100% V100 SM fill!).
struct CtrlPtrs3 {
    const float *w0, *w1, *w2;
    const float *c0, *c1, *c2;
    const float *m0, *m1, *m2;
};

template <int MT, bool EXACT_T>
__global__ void k_loop_down_fused_smallT_3(CtrlPtrs3 cp, float* __restrict__ Y_base,
                                           int N, int R, int T, float eps) {
    __shared__ float s_dc[MT][8];
    __shared__ float s_sc[MT][8];
    __shared__ float s_dm[MT][8];
    __shared__ float s_sm[MT][8];
    const int o = blockIdx.x;
    const int mod = blockIdx.y;
    if (o >= R || mod >= 3) return;
    const float* W = (mod == 0) ? cp.w0 : ((mod == 1) ? cp.w1 : cp.w2);
    const float* cur = (mod == 0) ? cp.c0 : ((mod == 1) ? cp.c1 : cp.c2);
    const float* msg = (mod == 0) ? cp.m0 : ((mod == 1) ? cp.m1 : cp.m2);
    float* Y = Y_base + (size_t) mod * (size_t) (EXACT_T ? MT : T) * (size_t) R;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int n4 = N >> 2;
    const float4* wc4 = reinterpret_cast<const float4*>(W + (size_t) o * (2 * N));
    const float4* wm4 = reinterpret_cast<const float4*>(W + (size_t) o * (2 * N) + N);
    float dc[MT], sc[MT], dm[MT], sm[MT];
#pragma unroll
    for (int t = 0; t < MT; ++t) { dc[t] = 0.0f; sc[t] = 0.0f; dm[t] = 0.0f; sm[t] = 0.0f; }
    for (int i = tid; i < n4; i += 256) {
        const float4 awc = wc4[i];
        const float4 awm = wm4[i];
#pragma unroll
        for (int t = 0; t < MT; ++t) {
            if (EXACT_T || t < T) {
                const float4 vc = reinterpret_cast<const float4*>(cur + (size_t) t * N)[i];
                const float4 vm = reinterpret_cast<const float4*>(msg + (size_t) t * N)[i];
                dc[t] += awc.x * vc.x + awc.y * vc.y + awc.z * vc.z + awc.w * vc.w;
                sc[t] += vc.x * vc.x + vc.y * vc.y + vc.z * vc.z + vc.w * vc.w;
                dm[t] += awm.x * vm.x + awm.y * vm.y + awm.z * vm.z + awm.w * vm.w;
                sm[t] += vm.x * vm.x + vm.y * vm.y + vm.z * vm.z + vm.w * vm.w;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < MT; ++t) {
        if (EXACT_T || t < T) {
            float vdc = dc[t], vsc = sc[t], vdm = dm[t], vsm = sm[t];
#pragma unroll
            for (int m = 16; m > 0; m >>= 1) {
                vdc += __shfl_xor_sync(0xffffffffu, vdc, m);
                vsc += __shfl_xor_sync(0xffffffffu, vsc, m);
                vdm += __shfl_xor_sync(0xffffffffu, vdm, m);
                vsm += __shfl_xor_sync(0xffffffffu, vsm, m);
            }
            if (lane == 0) {
                s_dc[t][warp] = vdc; s_sc[t][warp] = vsc;
                s_dm[t][warp] = vdm; s_sm[t][warp] = vsm;
            }
        }
    }
    __syncthreads();
    if (tid < (EXACT_T ? MT : T)) {
        float tdc = 0.0f, tsc = 0.0f, tdm = 0.0f, tsm = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            tdc += s_dc[tid][w]; tsc += s_sc[tid][w];
            tdm += s_dm[tid][w]; tsm += s_sm[tid][w];
        }
        const float rc = rsqrtf(tsc / (float) N + eps);
        const float rm = rsqrtf(tsm / (float) N + eps);
        Y[(size_t) tid * R + o] = tdc * rc + tdm * rm;
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

/// Fused control prep directly into FP16 [T][2*N] for Tensor Core loop_down GEMM (prefill, (N&3)==0).
/// EX-4C: Single-pass 576-thread register-cached path for N==2304 halves HBM reads (22 ms -> ~11 ms!).
__global__ void k_control_prep_f16(const float* __restrict__ cur, const float* __restrict__ msg,
                                   uint16_t* __restrict__ out2N_f16, int N, float eps) {
    __shared__ float sh[32];
    const int k = blockIdx.x, t = blockIdx.y;
    const float* src = (k == 0 ? cur : msg) + (size_t) t * N;
    __half2* dst2 = reinterpret_cast<__half2*>(out2N_f16 + (size_t) t * (2 * N) + (k * N));
    const int n4 = N >> 2;
    const float4* s4 = (const float4*) src;
    if (N == 2304 && blockDim.x == 576) {
        const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
        const float4 v = s4[tid];
        float s = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) sh[warp] = s;
        __syncthreads();
        if (warp == 0) {
            float tot = (lane < 18) ? sh[lane] : 0.0f;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, o);
            if (lane == 0) sh[0] = rsqrtf(tot / 2304.0f + eps);
        }
        __syncthreads();
        const float r = sh[0];
        dst2[tid * 2 + 0] = __halves2half2(__float2half_rn(v.x * r), __float2half_rn(v.y * r));
        dst2[tid * 2 + 1] = __halves2half2(__float2half_rn(v.z * r), __float2half_rn(v.w * r));
        return;
    }
    float s = 0.0f;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 v = s4[i];
        s += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) N + eps);
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 v = s4[i];
        dst2[i * 2 + 0] = __halves2half2(__float2half_rn(v.x * r), __float2half_rn(v.y * r));
        dst2[i * 2 + 1] = __halves2half2(__float2half_rn(v.z * r), __float2half_rn(v.w * r));
    }
}

/// Fast BF16 -> FP16 converter for narrow BF16 weight matrices (N_out <= 64, ~110 KB -> ~1.5 us, eliminates 5.2 ms X->BF16!).
__global__ void k_bf16_to_f16(const uint16_t* __restrict__ src, uint16_t* __restrict__ dst, int64_t n) {
    const int64_t i4 = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) << 2;
    if (i4 + 3 < n) {
        const uint2 v = *reinterpret_cast<const uint2*>(src + i4);
        const float f0 = __uint_as_float((v.x & 0xFFFFu) << 16);
        const float f1 = __uint_as_float(v.x & 0xFFFF0000u);
        const float f2 = __uint_as_float((v.y & 0xFFFFu) << 16);
        const float f3 = __uint_as_float(v.y & 0xFFFF0000u);
        const __half2 h0 = __floats2half2_rn(f0, f1);
        const __half2 h1 = __floats2half2_rn(f2, f3);
        uint2 o;
        memcpy(&o.x, &h0, 4);
        memcpy(&o.y, &h1, 4);
        *reinterpret_cast<uint2*>(dst + i4) = o;
    } else {
        for (int64_t i = i4; i < n; ++i) {
            const float f = __uint_as_float(((uint32_t) src[i]) << 16);
            const __half h = __float2half_rn(f);
            memcpy(dst + i, &h, 2);
        }
    }
}

/// Fast BF16 -> FP16 converter with trailing zero-padded rows (`sm_70` & `sm_89` safe, zero `<cuda_bf16.h>` hazards, STACK: 0).
/// Converts `n_rows * K` BF16 elements into `dst_f16[0 .. n_rows*K)` and zero-fills `dst_f16[n_rows*K .. (n_rows+pad_rows)*K)`.
__global__ void k_bf16_to_f16_pad(
    const uint16_t* __restrict__ src_bf16, uint16_t* __restrict__ dst_f16,
    int n_rows, int pad_rows, int K) {
    const int64_t n_valid = (int64_t) n_rows * K;
    const int64_t n_total = (int64_t) (n_rows + pad_rows) * K;
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    if ((K & 3) == 0) {
        const int64_t v4 = n_valid >> 2;
        const int64_t t4 = n_total >> 2;
        for (int64_t idx4 = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; idx4 < t4; idx4 += stride) {
            if (idx4 < v4) {
                const uint2 v = reinterpret_cast<const uint2*>(src_bf16)[idx4];
                const float f0 = __uint_as_float((v.x & 0xFFFFu) << 16);
                const float f1 = __uint_as_float(v.x & 0xFFFF0000u);
                const float f2 = __uint_as_float((v.y & 0xFFFFu) << 16);
                const float f3 = __uint_as_float(v.y & 0xFFFF0000u);
                const __half2 h0 = __floats2half2_rn(f0, f1);
                const __half2 h1 = __floats2half2_rn(f2, f3);
                uint2 o;
                memcpy(&o.x, &h0, 4);
                memcpy(&o.y, &h1, 4);
                reinterpret_cast<uint2*>(dst_f16)[idx4] = o;
            } else {
                reinterpret_cast<uint2*>(dst_f16)[idx4] = make_uint2(0u, 0u);
            }
        }
    } else {
        for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n_total; i += stride) {
            if (i < n_valid) {
                const float f = __uint_as_float(((uint32_t) src_bf16[i]) << 16);
                const __half h = __float2half_rn(f);
                memcpy(dst_f16 + i, &h, 2);
            } else {
                dst_f16[i] = 0u;
            }
        }
    }
}

/// EX-4C Wave 2 (sm_80+): Fused prefill loop_down (rms_norm(c, msg) -> FP16 -> W_down[64, 2*N]^T -> ctrl64[T, 64]).
/// M_MODS == 3 shares `c` across Q/K/V on SWA layers (reads `in_q` ONCE instead of 3x = 3x less HBM traffic!).
/// Eliminates k_control_prep_f16 (22.3 ms), launch_f32_to_f16(W_down) (1.5 ms), and gemm.f16 (6.3 ms)!
/// Grid: (T + 31) / 32, Block: 256 (8 warps: 2 M-warps x 4 N-warps), SMEM: 14.3 KB (6 blk/SM), STACK: 0!
struct CtrlDownPtrs3 {
    const float* w[3];
    const float* msg[3];
    const float* ir_msg[3];
    float* out64[3];
};

template <int M_MODS>
__global__ void __launch_bounds__(256, 2) k_control_down_pp_wmma(
    const float* __restrict__ c, const float* __restrict__ ir_c,
    CtrlDownPtrs3 p, int T, int N) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int BM = 32, BN = 64, WK = 64, LD = 72;
    __shared__ __align__(16) __half sx[BM * LD];
    __shared__ __align__(16) __half sw[BN * LD];
    __shared__ float s_irc[BM];
    __shared__ float s_irm[M_MODS][BM];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int t0 = (int) blockIdx.x * BM;
    const int rows = min(BM, T - t0);
    if (rows <= 0) return;

    if (tid < BM) {
        s_irc[tid] = (tid < rows) ? ir_c[t0 + tid] : 0.0f;
#pragma unroll
        for (int m = 0; m < M_MODS; ++m) {
            s_irm[m][tid] = (tid < rows) ? p.ir_msg[m][t0 + tid] : 0.0f;
        }
    }
    __syncthreads();

    const int wm = warp >> 2;  // 0..1 (16 rows/warp)
    const int wn = warp & 3;   // 0..3 (16 cols/warp)
    const int a_row = lane & 15, a_col = (lane >> 4) << 3;
    const int b_row = ((lane >> 4) << 3) | (lane & 7), b_col = ((lane >> 3) & 1) << 3;
    const uint32_t sx_a_smem = static_cast<uint32_t>(__cvta_generic_to_shared(sx + (wm * 16 + a_row) * LD + a_col));
    const uint32_t sw_b_smem = static_cast<uint32_t>(__cvta_generic_to_shared(sw + (wn * 16 + b_row) * LD + b_col));

    auto ldmat4 = [](uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
    };
    auto mma16816 = [](float d[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    };

    const int xr = tid >> 3;        // 0..31
    const int xc = (tid & 7) << 3;  // 0, 8, ..., 56 (8 floats = 2 float4s)
    const int wr = tid >> 2;        // 0..63
    const int wc = (tid & 3) << 4;  // 0, 16, 32, 48 (16 floats = 4 float4s)
    const bool x_ok = (xr < rows);
    const float irc_t = s_irc[xr];

    auto load_x_scaled = [&](const float* src, int k0, float sc) {
        __half2* dst2 = reinterpret_cast<__half2*>(sx + xr * LD + xc);
        if (x_ok) {
            const float4* sp = reinterpret_cast<const float4*>(src + (size_t) (t0 + xr) * N + k0 + xc);
            const float4 v0 = sp[0], v1 = sp[1];
            dst2[0] = __floats2half2_rn(v0.x * sc, v0.y * sc);
            dst2[1] = __floats2half2_rn(v0.z * sc, v0.w * sc);
            dst2[2] = __floats2half2_rn(v1.x * sc, v1.y * sc);
            dst2[3] = __floats2half2_rn(v1.z * sc, v1.w * sc);
        } else {
            *reinterpret_cast<uint4*>(dst2) = make_uint4(0, 0, 0, 0);
        }
    };
    auto load_w = [&](const float* w_base, int k_off) {
        __half2* dst2 = reinterpret_cast<__half2*>(sw + wr * LD + wc);
        const float4* wp = reinterpret_cast<const float4*>(w_base + (size_t) wr * (2 * N) + k_off + wc);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float4 v = __ldg(wp + i);
            dst2[i * 2 + 0] = __floats2half2_rn(v.x, v.y);
            dst2[i * 2 + 1] = __floats2half2_rn(v.z, v.w);
        }
    };

    float acc[M_MODS][8];
#pragma unroll
    for (int m = 0; m < M_MODS; ++m) {
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[m][i] = 0.0f;
    }

    for (int k0 = 0; k0 < N; k0 += WK) {
        // Phase A: load c[t0..t0+31, k0..k0+63] ONCE into sx, multiply by W_down[m][0..63, k0..k0+63] for all M_MODS!
        load_x_scaled(c, k0, irc_t);
#pragma unroll
        for (int m = 0; m < M_MODS; ++m) {
            load_w(p.w[m], k0);
            __syncthreads();
#pragma unroll
            for (int kk = 0; kk < WK; kk += 16) {
                uint32_t a0, a1, a2, a3, b0, b1, b2, b3;
                const uint32_t off_b = (uint32_t) kk * 2u;
                ldmat4(a0, a1, a2, a3, sx_a_smem + off_b);
                ldmat4(b0, b1, b2, b3, sw_b_smem + off_b);
                mma16816(&acc[m][0], a0, a1, a2, a3, b0, b1);
                mma16816(&acc[m][4], a0, a1, a2, a3, b2, b3);
            }
            __syncthreads();
        }
        // Phase B: for each module m, load msg[m] + W_down[m][0..63, N + k0..k0+63] and accumulate!
#pragma unroll
        for (int m = 0; m < M_MODS; ++m) {
            load_x_scaled(p.msg[m], k0, s_irm[m][xr]);
            load_w(p.w[m], N + k0);
            __syncthreads();
#pragma unroll
            for (int kk = 0; kk < WK; kk += 16) {
                uint32_t a0, a1, a2, a3, b0, b1, b2, b3;
                const uint32_t off_b = (uint32_t) kk * 2u;
                ldmat4(a0, a1, a2, a3, sx_a_smem + off_b);
                ldmat4(b0, b1, b2, b3, sw_b_smem + off_b);
                mma16816(&acc[m][0], a0, a1, a2, a3, b0, b1);
                mma16816(&acc[m][4], a0, a1, a2, a3, b2, b3);
            }
            __syncthreads();
        }
    }

    const int acc_r0 = wm * 16 + (lane >> 2);
    const int acc_r1 = acc_r0 + 8;
    const int acc_c0 = wn * 16 + (lane & 3) * 2;
    const int acc_c1 = acc_c0 + 8;
#pragma unroll
    for (int m = 0; m < M_MODS; ++m) {
        if (acc_r0 < rows) {
            float* dst0 = p.out64[m] + (size_t) (t0 + acc_r0) * BN;
            *reinterpret_cast<float2*>(dst0 + acc_c0) = make_float2(acc[m][0], acc[m][1]);
            *reinterpret_cast<float2*>(dst0 + acc_c1) = make_float2(acc[m][4], acc[m][5]);
        }
        if (acc_r1 < rows) {
            float* dst1 = p.out64[m] + (size_t) (t0 + acc_r1) * BN;
            *reinterpret_cast<float2*>(dst1 + acc_c0) = make_float2(acc[m][2], acc[m][3]);
            *reinterpret_cast<float2*>(dst1 + acc_c1) = make_float2(acc[m][6], acc[m][7]);
        }
    }
#else
    (void) c; (void) ir_c; (void) p; (void) T; (void) N;
#endif
}

/// Fused Prefill (T >= 32) loop_gain + loop_forget ([64 -> N] FP16 TC) + k_control_combine in 1 launch!
/// Eliminates ctrl64/W_gain/W_forget FP16 conversions, 2x cuBLAS GEMMs, ctrl_g/ctrl_f (18.8 MB/module) HBM round-trip,
/// and k_control_combine launch! Grid = (N / 64, (T + 63) / 64), block = 256 (8 warps).
__global__ void __launch_bounds__(256, 2) k_control_up_combine_pp_wmma(
    const float* __restrict__ cur, const float* __restrict__ msg,
    const float* __restrict__ ctrl64, const float* __restrict__ w_gain,
    const float* __restrict__ w_forget, float* __restrict__ out,
    int T, int N) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    constexpr int BM = 64, BN = 64, K = 64, LD = 72;
#if __CUDA_ARCH__ < 750 || __CUDA_ARCH__ >= 800
    __shared__ __align__(16) __half smem[(BM + 2 * BN) * LD];  // 27,648 B -> 3 blk/SM on Volta & Ada!
#else
    __shared__ __align__(16) float smem_f[2 * BM * BN];        // 32,768 B -> 3 blk/SM on Turing
    __half* smem = reinterpret_cast<__half*>(smem_f);
#endif
    __half* sx  = smem;
    __half* swg = smem + BM * LD;
    __half* swf = smem + (BM + BN) * LD;

    const int tid = threadIdx.x, warp = tid >> 5;
    const int col0 = (int) blockIdx.x * BN;
    const int t0   = (int) blockIdx.y * BM;
    const int rows = min(BM, T - t0);
    if (rows <= 0) return;

    // Load ctrl64[t0..t0+63][0..63], w_gain[col0..col0+63][0..63], w_forget[col0..col0+63][0..63] -> FP16 SMEM
    // 64 rows x 64 cols = 1024 float4s = 4 float4s per thread (coalesced 16-byte loads)
    const int r_base = tid >> 4;  // 0..15
    const int c4     = tid & 15;  // 0..15 (4 floats = 2 half2s)
#pragma unroll
    for (int step = 0; step < 64; step += 16) {
        const int r = r_base + step;
        const int c = c4 << 2;
        if (r < rows) {
            const float4 vx = reinterpret_cast<const float4*>(ctrl64 + (size_t) (t0 + r) * K)[c4];
            *reinterpret_cast<__half2*>(sx + r * LD + c)     = __floats2half2_rn(vx.x, vx.y);
            *reinterpret_cast<__half2*>(sx + r * LD + c + 2) = __floats2half2_rn(vx.z, vx.w);
        } else {
            *reinterpret_cast<uint2*>(sx + r * LD + c) = make_uint2(0, 0);
        }
        const float4 vg = reinterpret_cast<const float4*>(w_gain + (size_t) (col0 + r) * K)[c4];
        *reinterpret_cast<__half2*>(swg + r * LD + c)     = __floats2half2_rn(vg.x, vg.y);
        *reinterpret_cast<__half2*>(swg + r * LD + c + 2) = __floats2half2_rn(vg.z, vg.w);
        const float4 vf = reinterpret_cast<const float4*>(w_forget + (size_t) (col0 + r) * K)[c4];
        *reinterpret_cast<__half2*>(swf + r * LD + c)     = __floats2half2_rn(vf.x, vf.y);
        *reinterpret_cast<__half2*>(swf + r * LD + c + 2) = __floats2half2_rn(vf.z, vf.w);
    }
    __syncthreads();

    const int wm = warp >> 2;  // 0..1 (32 rows/warp)
    const int wn = warp & 3;   // 0..3 (16 paired gain+forget cols/warp)

#if __CUDA_ARCH__ < 750
    const int lane = tid & 31;
    const int qa_row = ((lane & 16) >> 2) + ((lane & 4) << 1) + (lane & 3);
    const int kb_row = ((lane & 16) >> 2) + (lane & 8) + (lane & 3);
    const int acc_r0 = ((lane & 16) >> 2) + ((lane & 4) << 1) + (lane & 1);
    const int acc_r1 = acc_r0 + 2;
    const int acc_c0 = (lane & 8) + (lane & 2);
    const int acc_c1 = acc_c0 + 4;
    const __half* sx_r0  = sx  + (wm * 32 + qa_row) * LD;
    const __half* swg_r0 = swg + (wn * 16 + kb_row) * LD;
    const __half* swf_r0 = swf + (wn * 16 + kb_row) * LD;
    auto hmma_rc = [](float d[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, {%0, %1, %2, %3, %4, %5, %6, %7};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
            : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
    };
    float acc_g[2][8], acc_f[2][8];
#pragma unroll
    for (int mr = 0; mr < 2; ++mr) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { acc_g[mr][i] = 0.0f; acc_f[mr][i] = 0.0f; }
    }
#pragma unroll
    for (int kk = 0; kk < K; kk += 8) {
        const uint4 bg = *reinterpret_cast<const uint4*>(swg_r0 + kk);
        const uint4 bf = *reinterpret_cast<const uint4*>(swf_r0 + kk);
#pragma unroll
        for (int mr = 0; mr < 2; ++mr) {
            const uint4 a = *reinterpret_cast<const uint4*>(sx_r0 + mr * 16 * LD + kk);
            hmma_rc(acc_g[mr], a.x, a.y, bg.x, bg.y);
            hmma_rc(acc_f[mr], a.x, a.y, bf.x, bf.y);
            hmma_rc(acc_g[mr], a.z, a.w, bg.z, bg.w);
            hmma_rc(acc_f[mr], a.z, a.w, bf.z, bf.w);
        }
    }
#pragma unroll
    for (int mr = 0; mr < 2; ++mr) {
        const int r0 = wm * 32 + mr * 16 + acc_r0;
        const int r1 = wm * 32 + mr * 16 + acc_r1;
        if (r0 < rows) {
            const size_t base = (size_t) (t0 + r0) * N + col0 + wn * 16;
            const float2 c0v = *reinterpret_cast<const float2*>(cur + base + acc_c0);
            const float2 m0v = *reinterpret_cast<const float2*>(msg + base + acc_c0);
            *reinterpret_cast<float2*>(out + base + acc_c0) = make_float2(
                (1.0f - fast_tanh(acc_f[mr][0])) * c0v.x + fast_tanh(acc_g[mr][0]) * m0v.x,
                (1.0f - fast_tanh(acc_f[mr][1])) * c0v.y + fast_tanh(acc_g[mr][1]) * m0v.y);
            const float2 c1v = *reinterpret_cast<const float2*>(cur + base + acc_c1);
            const float2 m1v = *reinterpret_cast<const float2*>(msg + base + acc_c1);
            *reinterpret_cast<float2*>(out + base + acc_c1) = make_float2(
                (1.0f - fast_tanh(acc_f[mr][4])) * c1v.x + fast_tanh(acc_g[mr][4]) * m1v.x,
                (1.0f - fast_tanh(acc_f[mr][5])) * c1v.y + fast_tanh(acc_g[mr][5]) * m1v.y);
        }
        if (r1 < rows) {
            const size_t base = (size_t) (t0 + r1) * N + col0 + wn * 16;
            const float2 c0v = *reinterpret_cast<const float2*>(cur + base + acc_c0);
            const float2 m0v = *reinterpret_cast<const float2*>(msg + base + acc_c0);
            *reinterpret_cast<float2*>(out + base + acc_c0) = make_float2(
                (1.0f - fast_tanh(acc_f[mr][2])) * c0v.x + fast_tanh(acc_g[mr][2]) * m0v.x,
                (1.0f - fast_tanh(acc_f[mr][3])) * c0v.y + fast_tanh(acc_g[mr][3]) * m0v.y);
            const float2 c1v = *reinterpret_cast<const float2*>(cur + base + acc_c1);
            const float2 m1v = *reinterpret_cast<const float2*>(msg + base + acc_c1);
            *reinterpret_cast<float2*>(out + base + acc_c1) = make_float2(
                (1.0f - fast_tanh(acc_f[mr][6])) * c1v.x + fast_tanh(acc_g[mr][6]) * m1v.x,
                (1.0f - fast_tanh(acc_f[mr][7])) * c1v.y + fast_tanh(acc_g[mr][7]) * m1v.y);
        }
    }
#elif __CUDA_ARCH__ >= 800
    // EX-4C (Ampere/Ada sm_80+): native ldmatrix.x4 + mma.m16n8k16 + zero-SMEM-bounce register-direct float2 combine!
    const int lane = tid & 31;
    const int a_row = lane & 15, a_col = (lane >> 4) << 3;
    const int b_row = ((lane >> 4) << 3) | (lane & 7), b_col = ((lane >> 3) & 1) << 3;
    const int acc_r0 = lane >> 2, acc_r1 = acc_r0 + 8;
    const int acc_c0 = (lane & 3) * 2, acc_c1 = acc_c0 + 8;
    const __half* sx_a  = sx  + (wm * 32 + a_row) * LD + a_col;
    const __half* swg_b = swg + (wn * 16 + b_row) * LD + b_col;
    const __half* swf_b = swf + (wn * 16 + b_row) * LD + b_col;
    auto ldmat4 = [](uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
        const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(p));
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
    };
    auto mma16816 = [](float d[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    };
    float acc_g[2][8], acc_f[2][8];
#pragma unroll
    for (int mr = 0; mr < 2; ++mr) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { acc_g[mr][i] = 0.0f; acc_f[mr][i] = 0.0f; }
    }
#pragma unroll
    for (int kk = 0; kk < K; kk += 16) {
        uint32_t bg0, bg1, bg2, bg3, bf0, bf1, bf2, bf3;
        ldmat4(bg0, bg1, bg2, bg3, swg_b + kk);
        ldmat4(bf0, bf1, bf2, bf3, swf_b + kk);
#pragma unroll
        for (int mr = 0; mr < 2; ++mr) {
            uint32_t a0, a1, a2, a3;
            ldmat4(a0, a1, a2, a3, sx_a + mr * 16 * LD + kk);
            mma16816(&acc_g[mr][0], a0, a1, a2, a3, bg0, bg1);
            mma16816(&acc_g[mr][4], a0, a1, a2, a3, bg2, bg3);
            mma16816(&acc_f[mr][0], a0, a1, a2, a3, bf0, bf1);
            mma16816(&acc_f[mr][4], a0, a1, a2, a3, bf2, bf3);
        }
    }
#pragma unroll
    for (int mr = 0; mr < 2; ++mr) {
        const int r0 = wm * 32 + mr * 16 + acc_r0;
        const int r1 = wm * 32 + mr * 16 + acc_r1;
        if (r0 < rows) {
            const size_t base = (size_t) (t0 + r0) * N + col0 + wn * 16;
            const float2 c0v = *reinterpret_cast<const float2*>(cur + base + acc_c0);
            const float2 m0v = *reinterpret_cast<const float2*>(msg + base + acc_c0);
            *reinterpret_cast<float2*>(out + base + acc_c0) = make_float2(
                (1.0f - fast_tanh(acc_f[mr][0])) * c0v.x + fast_tanh(acc_g[mr][0]) * m0v.x,
                (1.0f - fast_tanh(acc_f[mr][1])) * c0v.y + fast_tanh(acc_g[mr][1]) * m0v.y);
            const float2 c1v = *reinterpret_cast<const float2*>(cur + base + acc_c1);
            const float2 m1v = *reinterpret_cast<const float2*>(msg + base + acc_c1);
            *reinterpret_cast<float2*>(out + base + acc_c1) = make_float2(
                (1.0f - fast_tanh(acc_f[mr][4])) * c1v.x + fast_tanh(acc_g[mr][4]) * m1v.x,
                (1.0f - fast_tanh(acc_f[mr][5])) * c1v.y + fast_tanh(acc_g[mr][5]) * m1v.y);
        }
        if (r1 < rows) {
            const size_t base = (size_t) (t0 + r1) * N + col0 + wn * 16;
            const float2 c0v = *reinterpret_cast<const float2*>(cur + base + acc_c0);
            const float2 m0v = *reinterpret_cast<const float2*>(msg + base + acc_c0);
            *reinterpret_cast<float2*>(out + base + acc_c0) = make_float2(
                (1.0f - fast_tanh(acc_f[mr][2])) * c0v.x + fast_tanh(acc_g[mr][2]) * m0v.x,
                (1.0f - fast_tanh(acc_f[mr][3])) * c0v.y + fast_tanh(acc_g[mr][3]) * m0v.y);
            const float2 c1v = *reinterpret_cast<const float2*>(cur + base + acc_c1);
            const float2 m1v = *reinterpret_cast<const float2*>(msg + base + acc_c1);
            *reinterpret_cast<float2*>(out + base + acc_c1) = make_float2(
                (1.0f - fast_tanh(acc_f[mr][6])) * c1v.x + fast_tanh(acc_g[mr][6]) * m1v.x,
                (1.0f - fast_tanh(acc_f[mr][7])) * c1v.y + fast_tanh(acc_g[mr][7]) * m1v.y);
        }
    }
#else
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_g[2], acc_f[2];
#pragma unroll
    for (int mr = 0; mr < 2; ++mr) {
        wmma::fill_fragment(acc_g[mr], 0.0f);
        wmma::fill_fragment(acc_f[mr], 0.0f);
    }
#pragma unroll
    for (int kk = 0; kk < K; kk += 16) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fbg, fbf;
        wmma::load_matrix_sync(fbg, swg + (wn * 16) * LD + kk, LD);
        wmma::load_matrix_sync(fbf, swf + (wn * 16) * LD + kk, LD);
#pragma unroll
        for (int mr = 0; mr < 2; ++mr) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
            wmma::load_matrix_sync(fa, sx + (wm * 32 + mr * 16) * LD + kk, LD);
            wmma::mma_sync(acc_g[mr], fa, fbg, acc_g[mr]);
            wmma::mma_sync(acc_f[mr], fa, fbf, acc_f[mr]);
        }
    }
#pragma unroll
    for (int mr = 0; mr < 2; ++mr) {
#pragma unroll
        for (int i = 0; i < acc_g[mr].num_elements; ++i) {
            acc_g[mr].x[i] = fast_tanh(acc_g[mr].x[i]);
            acc_f[mr].x[i] = 1.0f - fast_tanh(acc_f[mr].x[i]);
        }
    }
    __syncthreads();
    float* s_g = smem_f;
    float* s_f = smem_f + BM * BN;
    wmma::store_matrix_sync(s_g + (wm * 32) * BN + wn * 16,      acc_g[0], BN, wmma::mem_row_major);
    wmma::store_matrix_sync(s_g + (wm * 32 + 16) * BN + wn * 16, acc_g[1], BN, wmma::mem_row_major);
    wmma::store_matrix_sync(s_f + (wm * 32) * BN + wn * 16,      acc_f[0], BN, wmma::mem_row_major);
    wmma::store_matrix_sync(s_f + (wm * 32 + 16) * BN + wn * 16, acc_f[1], BN, wmma::mem_row_major);
    __syncthreads();
    for (int i4 = tid; i4 < rows * 16; i4 += 256) {
        const int r = i4 >> 4, c = (i4 & 15) << 2;
        const size_t off = (size_t) (t0 + r) * N + col0 + c;
        const float4 cv = *reinterpret_cast<const float4*>(cur + off);
        const float4 mv = *reinterpret_cast<const float4*>(msg + off);
        const float4 gv = reinterpret_cast<const float4*>(s_g)[i4];
        const float4 fv = reinterpret_cast<const float4*>(s_f)[i4];
        *reinterpret_cast<float4*>(out + off) = make_float4(
            fv.x * cv.x + gv.x * mv.x,
            fv.y * cv.y + gv.y * mv.y,
            fv.z * cv.z + gv.z * mv.z,
            fv.w * cv.w + gv.w * mv.w);
    }
#endif
#else
    (void) cur; (void) msg; (void) ctrl64; (void) w_gain; (void) w_forget; (void) out; (void) T; (void) N;
#endif
}

struct CtrlDownVmsgPtrs3 {
    const float* c[3];       // [T, N] FP32 (for SWA M_MODS=3: c[0]==c[1]==c[2]; for FFN M_MODS=1: c[0])
    const float* inv_c[3];   // [T] FP32 precomputed inv_rms of c[m]
    const float* w_vmsg[3];  // [T, 2] FP32 softmax weights {w0, w1} for msg[m] = w0*bank0 + w1*bank1
    const float* inv_m[3];   // [T] FP32 precomputed inv_rms of msg[m]
    const float* w[3];       // [R, 2*N] FP32 loop_down weights (R=64)
    float* out[3];           // [T, R] FP32 output
};

/// Fused control_down reading virtual msg = w0*bank0 + w1*bank1 (never materializing msg0..3!).
/// Grid: ((T + 31) / 32), Block: 256.
template <int M_MODS>
__global__ void __launch_bounds__(256, 2) k_control_down_pp_wmma_vmsg(
    const float* __restrict__ bank0, const float* __restrict__ bank1,
    CtrlDownVmsgPtrs3 p, int T, int N) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int BM = 32, BN = 64, WK = 64, LD = 72;
    __shared__ __align__(16) __half sx[BM * LD];
    __shared__ __align__(16) __half sw[BN * LD];
    __shared__ float s_irc[M_MODS][BM];
    __shared__ float s_irm[M_MODS][BM];
    __shared__ float s_w0[M_MODS][BM];
    __shared__ float s_w1[M_MODS][BM];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int t0 = (int) blockIdx.x * BM;
    const int rows = min(BM, T - t0);
    if (rows <= 0) return;

    const int xr = tid >> 3;        // 0..31
    const int xc = (tid & 7) << 3;  // 0, 8, ..., 56 (8 floats = 2 float4s)
    const int wr = tid >> 2;        // 0..63
    const int wc = (tid & 3) << 4;  // 0, 16, 32, 48 (16 floats = 4 float4s)
    const bool x_ok = (xr < rows);

#pragma unroll
    for (int m = 0; m < M_MODS; ++m) {
        if (p.inv_c[m] != nullptr) {
            if (tid < BM) {
                s_irc[m][tid] = (tid < rows) ? p.inv_c[m][t0 + tid] : 0.0f;
            }
        } else {
            float ss = 0.0f;
            if (x_ok) {
                const float4* cp = reinterpret_cast<const float4*>(p.c[m] + (size_t) (t0 + xr) * N);
                const int n4 = N >> 2;
                for (int i = (tid & 7); i < n4; i += 8) {
                    const float4 v = cp[i];
                    ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
                }
            }
            ss += __shfl_xor_sync(0xffffffffu, ss, 4);
            ss += __shfl_xor_sync(0xffffffffu, ss, 2);
            ss += __shfl_xor_sync(0xffffffffu, ss, 1);
            if ((tid & 7) == 0) {
                s_irc[m][xr] = x_ok ? rsqrtf(ss / (float) N + 1e-6f) : 0.0f;
            }
        }
    }
    if (tid < BM) {
#pragma unroll
        for (int m = 0; m < M_MODS; ++m) {
            if (tid < rows) {
                s_irm[m][tid] = p.inv_m[m][t0 + tid];
                const float2 wv = reinterpret_cast<const float2*>(p.w_vmsg[m])[t0 + tid];
                s_w0[m][tid] = wv.x;
                s_w1[m][tid] = wv.y;
            } else {
                s_irm[m][tid] = 0.0f;
                s_w0[m][tid] = 0.0f;
                s_w1[m][tid] = 0.0f;
            }
        }
    }
    __syncthreads();

    const int wm = warp >> 2;  // 0..1 (16 rows/warp)
    const int wn = warp & 3;   // 0..3 (16 cols/warp)
    const int a_row = lane & 15, a_col = (lane >> 4) << 3;
    const int b_row = ((lane >> 4) << 3) | (lane & 7), b_col = ((lane >> 3) & 1) << 3;
    const uint32_t sx_a_smem = static_cast<uint32_t>(__cvta_generic_to_shared(sx + (wm * 16 + a_row) * LD + a_col));
    const uint32_t sw_b_smem = static_cast<uint32_t>(__cvta_generic_to_shared(sw + (wn * 16 + b_row) * LD + b_col));

    auto ldmat4 = [](uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
    };
    auto mma16816 = [](float d[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    };
    auto run_mma = [&](float acc_m[8]) {
#pragma unroll
        for (int kk = 0; kk < WK; kk += 16) {
            uint32_t a0, a1, a2, a3, b0, b1, b2, b3;
            const uint32_t off_b = (uint32_t) kk * 2u;
            ldmat4(a0, a1, a2, a3, sx_a_smem + off_b);
            ldmat4(b0, b1, b2, b3, sw_b_smem + off_b);
            mma16816(&acc_m[0], a0, a1, a2, a3, b0, b1);
            mma16816(&acc_m[4], a0, a1, a2, a3, b2, b3);
        }
    };
    auto load_x_scaled = [&](const float* src, int k0, float sc) {
        __half2* dst2 = reinterpret_cast<__half2*>(sx + xr * LD + xc);
        if (x_ok) {
            const float4* sp = reinterpret_cast<const float4*>(src + (size_t) (t0 + xr) * N + k0 + xc);
            const float4 v0 = sp[0], v1 = sp[1];
            dst2[0] = __floats2half2_rn(v0.x * sc, v0.y * sc);
            dst2[1] = __floats2half2_rn(v0.z * sc, v0.w * sc);
            dst2[2] = __floats2half2_rn(v1.x * sc, v1.y * sc);
            dst2[3] = __floats2half2_rn(v1.z * sc, v1.w * sc);
        } else {
            *reinterpret_cast<uint4*>(dst2) = make_uint4(0, 0, 0, 0);
        }
    };
    auto load_w = [&](const float* w_base, int k_off) {
        __half2* dst2 = reinterpret_cast<__half2*>(sw + wr * LD + wc);
        const float4* wp = reinterpret_cast<const float4*>(w_base + (size_t) wr * (2 * N) + k_off + wc);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float4 v = __ldg(wp + i);
            dst2[i * 2 + 0] = __floats2half2_rn(v.x, v.y);
            dst2[i * 2 + 1] = __floats2half2_rn(v.z, v.w);
        }
    };

    const bool shared_c = (M_MODS > 1) && (p.c[0] == p.c[1]) && (p.inv_c[0] == p.inv_c[1])
                          && (M_MODS < 3 || (p.c[0] == p.c[2] && p.inv_c[0] == p.inv_c[2]));

    float acc[M_MODS][8];
#pragma unroll
    for (int m = 0; m < M_MODS; ++m) {
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[m][i] = 0.0f;
    }

    for (int k0 = 0; k0 < N; k0 += WK) {
        // Phase A: c[m] * inv_c[m] x W_down[m][0..63, k0..k0+63] (shared_c reads c[0] ONCE across all M_MODS!)
        if constexpr (M_MODS > 1) {
            if (shared_c) {
                load_x_scaled(p.c[0], k0, s_irc[0][xr]);
#pragma unroll
                for (int m = 0; m < M_MODS; ++m) {
                    load_w(p.w[m], k0);
                    __syncthreads();
                    run_mma(acc[m]);
                    __syncthreads();
                }
            } else {
#pragma unroll
                for (int m = 0; m < M_MODS; ++m) {
                    load_x_scaled(p.c[m], k0, s_irc[m][xr]);
                    load_w(p.w[m], k0);
                    __syncthreads();
                    run_mma(acc[m]);
                    __syncthreads();
                }
            }
        } else {
            load_x_scaled(p.c[0], k0, s_irc[0][xr]);
            load_w(p.w[0], k0);
            __syncthreads();
            run_mma(acc[0]);
            __syncthreads();
        }

        // Phase B: synthesize virtual msg[m] = (w0*bank0 + w1*bank1)*inv_m[m] (bank0/bank1 hit L1 across m=0..2!)
#pragma unroll
        for (int m = 0; m < M_MODS; ++m) {
            __half2* dst2 = reinterpret_cast<__half2*>(sx + xr * LD + xc);
            if (x_ok) {
                const float4* bp0 = reinterpret_cast<const float4*>(bank0 + (size_t) (t0 + xr) * N + k0 + xc);
                const float4* bp1 = reinterpret_cast<const float4*>(bank1 + (size_t) (t0 + xr) * N + k0 + xc);
                const float4 b0_0 = bp0[0], b0_1 = bp0[1];
                const float4 b1_0 = bp1[0], b1_1 = bp1[1];
                const float w0 = s_w0[m][xr], w1 = s_w1[m][xr], im = s_irm[m][xr];
                dst2[0] = __floats2half2_rn((w0 * b0_0.x + w1 * b1_0.x) * im, (w0 * b0_0.y + w1 * b1_0.y) * im);
                dst2[1] = __floats2half2_rn((w0 * b0_0.z + w1 * b1_0.z) * im, (w0 * b0_0.w + w1 * b1_0.w) * im);
                dst2[2] = __floats2half2_rn((w0 * b0_1.x + w1 * b1_1.x) * im, (w0 * b0_1.y + w1 * b1_1.y) * im);
                dst2[3] = __floats2half2_rn((w0 * b0_1.z + w1 * b1_1.z) * im, (w0 * b0_1.w + w1 * b1_1.w) * im);
            } else {
                *reinterpret_cast<uint4*>(dst2) = make_uint4(0, 0, 0, 0);
            }
            load_w(p.w[m], N + k0);
            __syncthreads();
            run_mma(acc[m]);
            __syncthreads();
        }
    }

    const int acc_r0 = wm * 16 + (lane >> 2);
    const int acc_r1 = acc_r0 + 8;
    const int acc_c0 = wn * 16 + (lane & 3) * 2;
    const int acc_c1 = acc_c0 + 8;
#pragma unroll
    for (int m = 0; m < M_MODS; ++m) {
        if (acc_r0 < rows) {
            float* dst0 = p.out[m] + (size_t) (t0 + acc_r0) * BN;
            *reinterpret_cast<float2*>(dst0 + acc_c0) = make_float2(acc[m][0], acc[m][1]);
            *reinterpret_cast<float2*>(dst0 + acc_c1) = make_float2(acc[m][4], acc[m][5]);
        }
        if (acc_r1 < rows) {
            float* dst1 = p.out[m] + (size_t) (t0 + acc_r1) * BN;
            *reinterpret_cast<float2*>(dst1 + acc_c0) = make_float2(acc[m][2], acc[m][3]);
            *reinterpret_cast<float2*>(dst1 + acc_c1) = make_float2(acc[m][6], acc[m][7]);
        }
    }
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    constexpr int BM = 32, BN = 64, WK = 64, LD = 72;
    __shared__ __align__(16) __half sx[BM * LD];
    __shared__ __align__(16) __half sw[BN * LD];
    __shared__ float s_irc[M_MODS][BM];
    __shared__ float s_irm[M_MODS][BM];
    __shared__ float s_w0[M_MODS][BM];
    __shared__ float s_w1[M_MODS][BM];

    const int tid = threadIdx.x, warp = tid >> 5;
    const int t0 = (int) blockIdx.x * BM;
    const int rows = min(BM, T - t0);
    if (rows <= 0) return;

    const int xr = tid >> 3, xc = (tid & 7) << 3;
    const int wr = tid >> 2, wc = (tid & 3) << 4;
    const bool x_ok = (xr < rows);

    if (tid < BM) {
#pragma unroll
        for (int m = 0; m < M_MODS; ++m) {
            if (tid < rows) {
                s_irc[m][tid] = p.inv_c[m] ? p.inv_c[m][t0 + tid] : 1.0f;
                s_irm[m][tid] = p.inv_m[m][t0 + tid];
                const float2 wv = reinterpret_cast<const float2*>(p.w_vmsg[m])[t0 + tid];
                s_w0[m][tid] = wv.x;
                s_w1[m][tid] = wv.y;
            } else {
                s_irc[m][tid] = 0.0f; s_irm[m][tid] = 0.0f; s_w0[m][tid] = 0.0f; s_w1[m][tid] = 0.0f;
            }
        }
    }
    __syncthreads();

    const int wm = warp >> 2, wn = warp & 3;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[M_MODS];
#pragma unroll
    for (int m = 0; m < M_MODS; ++m) wmma::fill_fragment(acc[m], 0.0f);

    auto load_w_v70 = [&](const float* w_base, int k_off) {
        __half2* dst2 = reinterpret_cast<__half2*>(sw + wr * LD + wc);
        const float4* wp = reinterpret_cast<const float4*>(w_base + (size_t) wr * (2 * N) + k_off + wc);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float4 v = __ldg(wp + i);
            dst2[i * 2 + 0] = __floats2half2_rn(v.x, v.y);
            dst2[i * 2 + 1] = __floats2half2_rn(v.z, v.w);
        }
    };
    auto mma_v70 = [&](wmma::fragment<wmma::accumulator, 16, 16, 16, float>& ac) {
#pragma unroll
        for (int kk = 0; kk < WK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
            wmma::load_matrix_sync(fa, sx + (wm * 16) * LD + kk, LD);
            wmma::load_matrix_sync(fb, sw + (wn * 16) * LD + kk, LD);
            wmma::mma_sync(ac, fa, fb, ac);
        }
    };

    for (int k0 = 0; k0 < N; k0 += WK) {
#pragma unroll
        for (int m = 0; m < M_MODS; ++m) {
            __half2* dst2 = reinterpret_cast<__half2*>(sx + xr * LD + xc);
            if (x_ok) {
                const float sc = s_irc[m][xr];
                const float4* sp = reinterpret_cast<const float4*>(p.c[m] + (size_t) (t0 + xr) * N + k0 + xc);
                const float4 v0 = sp[0], v1 = sp[1];
                dst2[0] = __floats2half2_rn(v0.x * sc, v0.y * sc);
                dst2[1] = __floats2half2_rn(v0.z * sc, v0.w * sc);
                dst2[2] = __floats2half2_rn(v1.x * sc, v1.y * sc);
                dst2[3] = __floats2half2_rn(v1.z * sc, v1.w * sc);
            } else {
                *reinterpret_cast<uint4*>(dst2) = make_uint4(0, 0, 0, 0);
            }
            load_w_v70(p.w[m], k0);
            __syncthreads();
            mma_v70(acc[m]);
            __syncthreads();
        }
#pragma unroll
        for (int m = 0; m < M_MODS; ++m) {
            __half2* dst2 = reinterpret_cast<__half2*>(sx + xr * LD + xc);
            if (x_ok) {
                const float w0 = s_w0[m][xr], w1 = s_w1[m][xr], im = s_irm[m][xr];
                const float4* bp0 = reinterpret_cast<const float4*>(bank0 + (size_t) (t0 + xr) * N + k0 + xc);
                const float4* bp1 = reinterpret_cast<const float4*>(bank1 + (size_t) (t0 + xr) * N + k0 + xc);
                const float4 b0_0 = bp0[0], b0_1 = bp0[1], b1_0 = bp1[0], b1_1 = bp1[1];
                dst2[0] = __floats2half2_rn((w0 * b0_0.x + w1 * b1_0.x) * im, (w0 * b0_0.y + w1 * b1_0.y) * im);
                dst2[1] = __floats2half2_rn((w0 * b0_0.z + w1 * b1_0.z) * im, (w0 * b0_0.w + w1 * b1_0.w) * im);
                dst2[2] = __floats2half2_rn((w0 * b0_1.x + w1 * b1_1.x) * im, (w0 * b0_1.y + w1 * b1_1.y) * im);
                dst2[3] = __floats2half2_rn((w0 * b0_1.z + w1 * b1_1.z) * im, (w0 * b0_1.w + w1 * b1_1.w) * im);
            } else {
                *reinterpret_cast<uint4*>(dst2) = make_uint4(0, 0, 0, 0);
            }
            load_w_v70(p.w[m], N + k0);
            __syncthreads();
            mma_v70(acc[m]);
            __syncthreads();
        }
    }
    float* s_out = reinterpret_cast<float*>(sw);
#pragma unroll
    for (int m = 0; m < M_MODS; ++m) {
        wmma::store_matrix_sync(s_out + (wm * 16) * BN + wn * 16, acc[m], BN, wmma::mem_row_major);
        __syncthreads();
        for (int idx = tid; idx < rows * BN; idx += 256) {
            const int r = idx / BN, c_idx = idx % BN;
            p.out[m][(size_t) (t0 + r) * BN + c_idx] = s_out[idx];
        }
        __syncthreads();
    }
#else
    (void) bank0; (void) bank1; (void) p; (void) T; (void) N;
#endif
}

struct CtrlUpRmsVmsgPtrs3 {
    const float* c[3];         // [T, N] FP32
    const float* w_vmsg[3];    // [T, 2] FP32 {w0, w1} for virtual msg = w0*bank0 + w1*bank1
    const float* ctrl_in[3];   // [T, R] FP32 output of control_down (R=64)
    const float* w_gain[3];    // [N, R] FP32
    const float* w_forget[3];  // [N, R] FP32
    const float* rms_w[3];     // [N] FP32 RMSNorm weight (e.g. attn_norm / ffn_norm; nullptr -> 1.0f)
    float* out_f32[3];         // [T, N] optional FP32 normed output (nullptr for Attn Q/K/V -> 0 dead FP32 writes; non-null for FFN in_ffn!)
    uint16_t* out_f16[3];      // [T, N] FP16 normed output (written via st.global.cs!)
};

/// Fused control_up + combine(c, virtual msg) + RMSNorm + FP16 (+ optional FP32) output!
/// Eliminates k_rms_norm_3_f16 (24x), k_rms_norm_f16 (24x), AND dead FP32 in_q/in_k/in_v writes!
/// Grid: ((T + 15) / 16), Block: 768 (24 warps, N == 2304, BM = 16).
template <int M_MODS>
__global__ void __launch_bounds__(768, 1) k_control_up_combine_rms_f16_vmsg_legacy(
    const float* __restrict__ bank0, const float* __restrict__ bank1,
    CtrlUpRmsVmsgPtrs3 p, int T, int N, float rms_eps) {
    constexpr int BM = 16, K = 64;
    const int tid = threadIdx.x, wn = tid >> 5, lane = tid & 31;
    const int t0 = (int) blockIdx.x * BM;
    const int rows = min(BM, T - t0);
    if (rows <= 0) return;

    __shared__ CtrlUpRmsVmsgPtrs3 s_p;
    __shared__ float s_wv[BM][2];
    __shared__ float s_ss[BM][25]; // padded stride 25 -> 0 bank conflicts across rows

    if (tid == 0) {
#pragma unroll
        for (int k = 0; k < M_MODS; ++k) {
            s_p.c[k]        = p.c[k];
            s_p.w_vmsg[k]   = p.w_vmsg[k];
            s_p.ctrl_in[k]  = p.ctrl_in[k];
            s_p.w_gain[k]   = p.w_gain[k];
            s_p.w_forget[k] = p.w_forget[k];
            s_p.rms_w[k]    = p.rms_w[k];
            s_p.out_f32[k]  = p.out_f32[k];
            s_p.out_f16[k]  = p.out_f16[k];
        }
    }
    __syncthreads();

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int LD = 72;
    __shared__ __align__(16) __half s_ctrl[BM * LD];       // 2,304 B
    __shared__ __align__(16) __half s_w[24 * 8 * LD];      // 27,648 B (1,152 B/warp, 0 inter-warp sync!)

    const int a_row = lane & 15, a_col = (lane >> 4) << 3;
    const int b_row = lane & 7,  b_col = ((lane >> 3) & 1) << 3;
    const int acc_r0 = lane >> 2, acc_r1 = acc_r0 + 8;
    const int acc_c0 = (lane & 3) * 2;
    const int col_warp = wn * 96;
    const bool r0_ok = (acc_r0 < rows);
    const bool r1_ok = (acc_r1 < rows);

    __half* sw_warp = s_w + wn * (8 * LD);
    const uint32_t s_ctrl_a_smem = static_cast<uint32_t>(__cvta_generic_to_shared(s_ctrl + a_row * LD + a_col));
    const uint32_t sw_b_smem     = static_cast<uint32_t>(__cvta_generic_to_shared(sw_warp + b_row * LD + b_col));

    auto ldmat4 = [](uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
    };
    auto ldmat2 = [](uint32_t& r0, uint32_t& r1, uint32_t addr) {
        asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
                     : "=r"(r0), "=r"(r1) : "r"(addr));
    };
    auto mma16816 = [](float d[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    };
    auto load_w_warp = [&](const float* w_base, int col_base) {
        const int r_half = lane >> 4;  // 0..1
        const int c4 = lane & 15;      // 0..15 (4 floats = 2 half2s)
        const int c = c4 << 2;
#pragma unroll 1
        for (int step = 0; step < 8; step += 2) {
            const int r = r_half + step;
            const float4 v = __ldg(reinterpret_cast<const float4*>(w_base + (size_t) (col_base + r) * K) + c4);
            *reinterpret_cast<__half2*>(sw_warp + r * LD + c)     = __floats2half2_rn(v.x, v.y);
            *reinterpret_cast<__half2*>(sw_warp + r * LD + c + 2) = __floats2half2_rn(v.z, v.w);
        }
    };
    auto run_mma_k64 = [&](float acc[4]) {
        acc[0] = 0.0f; acc[1] = 0.0f; acc[2] = 0.0f; acc[3] = 0.0f;
#pragma unroll
        for (int kk = 0; kk < K; kk += 16) {
            uint32_t a0, a1, a2, a3, b0, b1;
            const uint32_t off_b = (uint32_t) kk * 2u;
            ldmat4(a0, a1, a2, a3, s_ctrl_a_smem + off_b);
            ldmat2(b0, b1, sw_b_smem + off_b);
            mma16816(acc, a0, a1, a2, a3, b0, b1);
        }
    };

    // Reverse order (m = M_MODS-1 .. 0) ensures out_f32[1..2] are written before out_f32[0] even if out_f32[0] == c[0..2]!
    for (int m = M_MODS - 1; m >= 0; --m) {
        if (tid < 256) {
            const int r = tid >> 4;
            const int c4 = tid & 15;
            const int c = c4 << 2;
            if (r < rows) {
                const float4 vx = reinterpret_cast<const float4*>(s_p.ctrl_in[m] + (size_t) (t0 + r) * K)[c4];
                *reinterpret_cast<__half2*>(s_ctrl + r * LD + c)     = __floats2half2_rn(vx.x, vx.y);
                *reinterpret_cast<__half2*>(s_ctrl + r * LD + c + 2) = __floats2half2_rn(vx.z, vx.w);
            } else {
                *reinterpret_cast<uint2*>(s_ctrl + r * LD + c) = make_uint2(0, 0);
            }
        }
        if (tid < BM) {
            if (tid < rows) {
                const float2 wv = reinterpret_cast<const float2*>(s_p.w_vmsg[m])[t0 + tid];
                s_wv[tid][0] = wv.x;
                s_wv[tid][1] = wv.y;
            } else {
                s_wv[tid][0] = 0.0f;
                s_wv[tid][1] = 0.0f;
            }
        }
        __syncthreads();

        float y_buf[12][4];
        float ss0 = 0.0f, ss1 = 0.0f;

#pragma unroll
        for (int ct = 0; ct < 12; ++ct) {
            const int col_base = col_warp + ct * 8;
            const int col_c0 = col_base + acc_c0;
            float acc[4];

            // Step 1: gain projection + fast_tanh(g) * (w0*bank0 + w1*bank1) -> y_buf[ct]
            load_w_warp(s_p.w_gain[m], col_base);
            __syncwarp();
            run_mma_k64(acc);
            if (r0_ok) {
                const size_t off0 = (size_t) (t0 + acc_r0) * N + col_c0;
                const float2 b0v = *reinterpret_cast<const float2*>(bank0 + off0);
                const float2 b1v = *reinterpret_cast<const float2*>(bank1 + off0);
                const float w0 = s_wv[acc_r0][0], w1 = s_wv[acc_r0][1];
                y_buf[ct][0] = fast_tanh(acc[0]) * (w0 * b0v.x + w1 * b1v.x);
                y_buf[ct][1] = fast_tanh(acc[1]) * (w0 * b0v.y + w1 * b1v.y);
            } else {
                y_buf[ct][0] = 0.0f; y_buf[ct][1] = 0.0f;
            }
            if (r1_ok) {
                const size_t off1 = (size_t) (t0 + acc_r1) * N + col_c0;
                const float2 b0v = *reinterpret_cast<const float2*>(bank0 + off1);
                const float2 b1v = *reinterpret_cast<const float2*>(bank1 + off1);
                const float w0 = s_wv[acc_r1][0], w1 = s_wv[acc_r1][1];
                y_buf[ct][2] = fast_tanh(acc[2]) * (w0 * b0v.x + w1 * b1v.x);
                y_buf[ct][3] = fast_tanh(acc[3]) * (w0 * b0v.y + w1 * b1v.y);
            } else {
                y_buf[ct][2] = 0.0f; y_buf[ct][3] = 0.0f;
            }

            // Step 2: forget projection + (1 - fast_tanh(f)) * c + y_buf[ct] -> y_buf[ct], accumulate ss
            __syncwarp();
            load_w_warp(s_p.w_forget[m], col_base);
            __syncwarp();
            run_mma_k64(acc);
            const float* c_ptr = s_p.c[m];
            if (r0_ok) {
                const size_t off0 = (size_t) (t0 + acc_r0) * N + col_c0;
                const float2 cv = *reinterpret_cast<const float2*>(c_ptr + off0);
                const float y0 = __fmaf_rn(1.0f - fast_tanh(acc[0]), cv.x, y_buf[ct][0]);
                const float y1 = __fmaf_rn(1.0f - fast_tanh(acc[1]), cv.y, y_buf[ct][1]);
                y_buf[ct][0] = y0;
                y_buf[ct][1] = y1;
                ss0 += y0 * y0 + y1 * y1;
            }
            if (r1_ok) {
                const size_t off1 = (size_t) (t0 + acc_r1) * N + col_c0;
                const float2 cv = *reinterpret_cast<const float2*>(c_ptr + off1);
                const float y2 = __fmaf_rn(1.0f - fast_tanh(acc[2]), cv.x, y_buf[ct][2]);
                const float y3 = __fmaf_rn(1.0f - fast_tanh(acc[3]), cv.y, y_buf[ct][3]);
                y_buf[ct][2] = y2;
                y_buf[ct][3] = y3;
                ss1 += y2 * y2 + y3 * y3;
            }
            __syncwarp();
        }

        // Reduce ss across 4 lanes per row (8 cols/strip x 12 strips = 96 cols/warp), then across 24 warps!
        ss0 += __shfl_xor_sync(0xffffffffu, ss0, 1);
        ss0 += __shfl_xor_sync(0xffffffffu, ss0, 2);
        ss1 += __shfl_xor_sync(0xffffffffu, ss1, 1);
        ss1 += __shfl_xor_sync(0xffffffffu, ss1, 2);
        if ((lane & 3) == 0) {
            s_ss[acc_r0][wn] = ss0;
            s_ss[acc_r1][wn] = ss1;
        }
        __syncthreads();

        float tot0 = 0.0f, tot1 = 0.0f;
#pragma unroll
        for (int w = 0; w < 24; ++w) {
            tot0 += s_ss[acc_r0][w];
            tot1 += s_ss[acc_r1][w];
        }
        const float ir0 = rsqrtf(tot0 / (float) N + rms_eps);
        const float ir1 = rsqrtf(tot1 / (float) N + rms_eps);

        const float* rw_ptr  = s_p.rms_w[m];
        float* of32_ptr      = s_p.out_f32[m];
        uint16_t* of16_ptr   = s_p.out_f16[m];

#pragma unroll
        for (int ct = 0; ct < 12; ++ct) {
            const int col_c0 = col_warp + ct * 8 + acc_c0;
            float2 rw = make_float2(1.0f, 1.0f);
            if (rw_ptr) rw = __ldg(reinterpret_cast<const float2*>(rw_ptr + col_c0));
            if (r0_ok) {
                const size_t off0 = (size_t) (t0 + acc_r0) * N + col_c0;
                const float yn0 = y_buf[ct][0] * ir0 * rw.x;
                const float yn1 = y_buf[ct][1] * ir0 * rw.y;
                if (of32_ptr) {
                    *reinterpret_cast<float2*>(of32_ptr + off0) = make_float2(yn0, yn1);
                }
                if (of16_ptr) {
                    const __half2 h2 = __halves2half2(__float2half_rn(yn0), __float2half_rn(yn1));
                    uint32_t u32;
                    memcpy(&u32, &h2, 4);
                    uint32_t* dst = reinterpret_cast<uint32_t*>(of16_ptr + off0);
                    asm volatile("st.global.cs.u32 [%0], %1;" :: "l"(dst), "r"(u32) : "memory");
                }
            }
            if (r1_ok) {
                const size_t off1 = (size_t) (t0 + acc_r1) * N + col_c0;
                const float yn2 = y_buf[ct][2] * ir1 * rw.x;
                const float yn3 = y_buf[ct][3] * ir1 * rw.y;
                if (of32_ptr) {
                    *reinterpret_cast<float2*>(of32_ptr + off1) = make_float2(yn2, yn3);
                }
                if (of16_ptr) {
                    const __half2 h2 = __halves2half2(__float2half_rn(yn2), __float2half_rn(yn3));
                    uint32_t u32;
                    memcpy(&u32, &h2, 4);
                    uint32_t* dst = reinterpret_cast<uint32_t*>(of16_ptr + off1);
                    asm volatile("st.global.cs.u32 [%0], %1;" :: "l"(dst), "r"(u32) : "memory");
                }
            }
        }
        if (m > 0) __syncthreads();
    }
#else
    // Volta sm_70 / pre-Ampere fallback (clean FP32 exact path, 0 stack)
    __shared__ float s_ctrl_f32[BM][K];
    const int col_warp = wn * 96;

    for (int m = M_MODS - 1; m >= 0; --m) {
        for (int idx = tid; idx < BM * K; idx += 768) {
            const int r = idx / K, c = idx % K;
            s_ctrl_f32[r][c] = (r < rows) ? s_p.ctrl_in[m][(size_t) (t0 + r) * K + c] : 0.0f;
        }
        if (tid < BM) {
            if (tid < rows) {
                const float2 wv = reinterpret_cast<const float2*>(s_p.w_vmsg[m])[t0 + tid];
                s_wv[tid][0] = wv.x;
                s_wv[tid][1] = wv.y;
            } else {
                s_wv[tid][0] = 0.0f;
                s_wv[tid][1] = 0.0f;
            }
        }
        __syncthreads();

        // Pass 1: compute per-row sum of squares across all 24 warps (0 y_buf regs -> STACK: 0 on sm_70!)
        for (int r = 0; r < rows; ++r) {
            float ss = 0.0f;
            const float w0 = s_wv[r][0], w1 = s_wv[r][1];
            for (int k = 0; k < 3; ++k) {
                const int col = col_warp + k * 32 + lane;
                float g = 0.0f, f = 0.0f;
                const float4* wg4 = reinterpret_cast<const float4*>(s_p.w_gain[m] + (size_t) col * K);
                const float4* wf4 = reinterpret_cast<const float4*>(s_p.w_forget[m] + (size_t) col * K);
                const float4* x4  = reinterpret_cast<const float4*>(s_ctrl_f32[r]);
#pragma unroll 1
                for (int i = 0; i < 16; ++i) {
                    const float4 xv = x4[i];
                    const float4 gv = wg4[i];
                    const float4 fv = wf4[i];
                    g += gv.x * xv.x + gv.y * xv.y + gv.z * xv.z + gv.w * xv.w;
                    f += fv.x * xv.x + fv.y * xv.y + fv.z * xv.z + fv.w * xv.w;
                }
                const size_t off = (size_t) (t0 + r) * N + col;
                const float msg_v = w0 * bank0[off] + w1 * bank1[off];
                const float yv = (1.0f - fast_tanh(f)) * s_p.c[m][off] + fast_tanh(g) * msg_v;
                ss += yv * yv;
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
            if (lane == 0) s_ss[r][wn] = ss;
        }
        __syncthreads();

        // Pass 2: normalize and emit out_f32 / out_f16
        const float* rw_ptr = s_p.rms_w[m];
        float* of32_ptr     = s_p.out_f32[m];
        uint16_t* of16_ptr  = s_p.out_f16[m];
        for (int r = 0; r < rows; ++r) {
            float tot = 0.0f;
#pragma unroll
            for (int w = 0; w < 24; ++w) tot += s_ss[r][w];
            const float ir = rsqrtf(tot / (float) N + rms_eps);
            const float w0 = s_wv[r][0], w1 = s_wv[r][1];
            for (int k = 0; k < 3; ++k) {
                const int col = col_warp + k * 32 + lane;
                float g = 0.0f, f = 0.0f;
                const float4* wg4 = reinterpret_cast<const float4*>(s_p.w_gain[m] + (size_t) col * K);
                const float4* wf4 = reinterpret_cast<const float4*>(s_p.w_forget[m] + (size_t) col * K);
                const float4* x4  = reinterpret_cast<const float4*>(s_ctrl_f32[r]);
#pragma unroll 1
                for (int i = 0; i < 16; ++i) {
                    const float4 xv = x4[i];
                    const float4 gv = wg4[i];
                    const float4 fv = wf4[i];
                    g += gv.x * xv.x + gv.y * xv.y + gv.z * xv.z + gv.w * xv.w;
                    f += fv.x * xv.x + fv.y * xv.y + fv.z * xv.z + fv.w * xv.w;
                }
                const size_t off = (size_t) (t0 + r) * N + col;
                const float msg_v = w0 * bank0[off] + w1 * bank1[off];
                const float yv = (1.0f - fast_tanh(f)) * s_p.c[m][off] + fast_tanh(g) * msg_v;
                const float rw = rw_ptr ? rw_ptr[col] : 1.0f;
                const float yn = yv * ir * rw;
                if (of32_ptr) of32_ptr[off] = yn;
                if (of16_ptr) {
                    const __half h = __float2half_rn(yn);
                    uint16_t u16;
                    memcpy(&u16, &h, 2);
                    of16_ptr[off] = u16;
                }
            }
        }
        if (m > 0) __syncthreads();
    }
#endif
}

struct CtrlUpVmsgPtrs3 {
    const float* c[3];        // [T, N] FP32 (for SWA M_MODS=3: c[0]==c[1]==c[2])
    const float* w_vmsg[3];   // [T, 2] FP32 {w0, w1} for virtual msg = w0*bank0 + w1*bank1
    const float* ctrl_in[3];  // [T, R] FP32 (R = 64)
    const float* w_gain[3];   // [N, R] FP32
    const float* w_forget[3]; // [N, R] FP32
    float* out[3];            // [T, N] FP32 output (note: out[0] may alias c[0] on SWA, so loop m = M_MODS-1 .. 0!)
    const __half* wg_f16[3] = {nullptr, nullptr, nullptr};
    const __half* wf_f16[3] = {nullptr, nullptr, nullptr};
};

struct CtrlUpRmsOuts {
    const float* rms_w[3] = {nullptr, nullptr, nullptr}; // [N] RMSNorm weight
    uint16_t* out_f16[3] = {nullptr, nullptr, nullptr};  // [T, N] FP16 output (X_hi for M_MODS==1)
    __half* out_lo_f16 = nullptr;                        // [T, N] optional Ozaki-2 X_lo (for M_MODS==1 FFN router!)
    float eps = 1e-6f;
};

/// Wave 23: Fused ControlUp + VirtualMsg + Exact FP32 RMSNorm -> FP16 (+ optional Ozaki-2 X_lo for M_MODS==1).
/// Both M_MODS==3 and M_MODS==1 use BM=16, cp.async.cg double-buffered weights, and register-cached y_reg[14][4] + s_ext1
/// (zero s_y[8][2304] -> 29,696 B dynamic SMEM <= 48 KB default, 40 CTAs = 1 wave at T_CHUNK=640, REG<=96, STACK:0).
template <int M_MODS>
__global__ void __launch_bounds__(576, 1) k_control_up_combine_rms_f16_vmsg(
    const float* __restrict__ bank0, const float* __restrict__ bank1,
    CtrlUpVmsgPtrs3 p, CtrlUpRmsOuts ro, int T, int N) {
    constexpr int BM = 16;
    constexpr int K = 64;
    const int tid = (int) threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int t0 = (int) blockIdx.x * BM;
    const int rows = min(BM, T - t0);
    if (rows <= 0) return;

    auto sel3_c = [](int m, const float* a, const float* b, const float* c) -> const float* {
        if constexpr (M_MODS == 1) return a;
        return (m == 0) ? a : ((m == 1) ? b : c);
    };
    auto sel3_h = [](int m, const __half* a, const __half* b, const __half* c) -> const __half* {
        if constexpr (M_MODS == 1) return a;
        return (m == 0) ? a : ((m == 1) ? b : c);
    };
    auto sel3_o = [](int m, float* a, float* b, float* c) -> float* {
        if constexpr (M_MODS == 1) return a;
        return (m == 0) ? a : ((m == 1) ? b : c);
    };
    auto sel3_u16 = [](int m, uint16_t* a, uint16_t* b, uint16_t* c) -> uint16_t* {
        if constexpr (M_MODS == 1) return a;
        return (m == 0) ? a : ((m == 1) ? b : c);
    };

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // Dynamic SMEM (28,864 B <= 29,696 B for BOTH M_MODS==1 and M_MODS==3):
    //   s_ctrl[16][64]        :  2,048 B (+0)
    //   s_w[2][18][8][16]     :  9,216 B (+2,048)  [double-buffered ping-pong weight tiles]
    //   s_wv[16]              :    128 B (+11,264)
    //   s_ss[16 * 19]         :  1,216 B (+11,392)
    //   s_ir[16]              :     64 B (+12,608)
    //   s_mp3                 :     64 B (+12,672)
    //   s_ext1[576 * 7]       : 16,128 B (+12,736 .. 28,864)
    struct CurModPtrs {
        const __half* wg16;
        const __half* wf16;
        const float*  wg32;
        const float*  wf32;
        const float*  c_ptr;
        const float*  rw_ptr;
        uint16_t*     of16_ptr;
        float*        of32_ptr;
    };
    extern __shared__ __align__(16) char smem_raw[];
    __half* s_ctrl     = reinterpret_cast<__half*>(smem_raw);                    // +0 (2,048 B)
    __half* s_w0       = s_ctrl + 16 * 64;                                       // +2,048 (4,608 B)
    __half* s_w1       = s_w0 + 18 * 8 * 16;                                     // +6,656 (4,608 B)
    float2* s_wv       = reinterpret_cast<float2*>(smem_raw + 11264);            // +11,264 (128 B)
    float*  s_ss       = reinterpret_cast<float*>(smem_raw + 11392);             // +11,392 (1,216 B)
    float*  s_ir       = reinterpret_cast<float*>(smem_raw + 12608);             // +12,608 (64 B)
    CurModPtrs* s_mp3  = reinterpret_cast<CurModPtrs*>(smem_raw + 12672);        // +12,672 (64 B)
    float*  s_ext1     = reinterpret_cast<float*>(smem_raw + 12736);             // +12,736 (16,128 B)

    __half* sw_warp0 = s_w0 + warp * (8 * 16);
    __half* sw_warp1 = s_w1 + warp * (8 * 16);
    const int col_warp = warp * 128;

    const int acc_r0 = lane >> 2;          // 0..7
    const int acc_c0 = (lane & 3) * 2;     // 0, 2, 4, 6
    const bool r0_ok = (acc_r0 < rows);

    auto ldmat4 = [](uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
    };
    auto ldmat2 = [](uint32_t& r0, uint32_t& r1, uint32_t addr) {
        asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
                     : "=r"(r0), "=r"(r1) : "r"(addr));
    };
    auto mma16816 = [](float d[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    };
    auto pack_h2_u2 = [](float a, float b, float c, float d) -> uint2 {
        const __half2 h0 = __floats2half2_rn(a, b);
        const __half2 h1 = __floats2half2_rn(c, d);
        uint2 u;
        memcpy(&u.x, &h0, 4);
        memcpy(&u.y, &h1, 4);
        return u;
    };

    // Double-buffered cp.async.cg BK=16 MMA across K=64 (swizzle c8 ^ ((r >> 2) & 1) -> 0 bank conflicts!).
    auto run_w_mma_k64 = [&](const __half* w16, const float* w32, int col_base, float acc[4]) {
        acc[0] = 0.0f; acc[1] = 0.0f; acc[2] = 0.0f; acc[3] = 0.0f;
        const int a_row = lane & 15;
        const int a_c8_0 = (lane >> 4) ^ (a_row & 7);
        const uint32_t s_ctrl_row = static_cast<uint32_t>(__cvta_generic_to_shared(s_ctrl + a_row * 64));
        const int b_row = lane & 7;
        const int b_swz = ((lane >> 3) & 1) ^ ((b_row >> 2) & 1);
        const uint32_t sw_b_addr0 = static_cast<uint32_t>(__cvta_generic_to_shared(sw_warp0 + b_row * 16 + b_swz * 8));
        const uint32_t sw_b_addr1 = static_cast<uint32_t>(__cvta_generic_to_shared(sw_warp1 + b_row * 16 + b_swz * 8));
        const int wr = lane >> 1;      // 0..7 for lanes 0..15
        const int c8 = lane & 1;       // 0..1
        const int w_swz = c8 ^ ((wr >> 2) & 1);
        const int g_row = col_base + wr;

        if (w16 != nullptr) {
            const uint32_t sw_wr0 = static_cast<uint32_t>(__cvta_generic_to_shared(sw_warp0 + wr * 16 + w_swz * 8));
            const uint32_t sw_wr1 = static_cast<uint32_t>(__cvta_generic_to_shared(sw_warp1 + wr * 16 + w_swz * 8));
            const __half* g_ptr_base = (g_row < N) ? (w16 + (size_t) g_row * K + c8 * 8) : w16;
            const int src_bytes = (g_row < N) ? 16 : 0;
            auto issue_kb = [&](int kb, int buf) {
                if (lane < 16) {
                    const uint32_t dst_s = buf ? sw_wr1 : sw_wr0;
                    const void* src_g = g_ptr_base + kb * 16;
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                                 :: "r"(dst_s), "l"(src_g), "r"(src_bytes) : "memory");
                }
            };
            issue_kb(0, 0);
            asm volatile("cp.async.commit_group;\n" ::: "memory");
#pragma unroll
            for (int kb = 0; kb < 4; ++kb) {
                if (kb < 3) {
                    if (kb == 2) __syncwarp(); // ensure kb==0 buf0 ldmat2 completed before kb==2? No, buf1 completed at kb==1 wait.
                    issue_kb(kb + 1, (kb + 1) & 1);
                    asm volatile("cp.async.commit_group;\n" ::: "memory");
                    asm volatile("cp.async.wait_group 1;\n" ::: "memory");
                } else {
                    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
                }
                __syncwarp();
                uint32_t a0, a1, a2, a3, b0, b1;
                ldmat4(a0, a1, a2, a3, s_ctrl_row + (((uint32_t) a_c8_0 ^ (uint32_t) (kb << 1)) << 4));
                ldmat2(b0, b1, (kb & 1) ? sw_b_addr1 : sw_b_addr0);
                mma16816(acc, a0, a1, a2, a3, b0, b1);
                if (kb < 2) __syncwarp();
            }
        } else {
#pragma unroll 1
            for (int kb = 0; kb < 4; ++kb) {
                if (lane < 16) {
                    if (g_row < N) {
                        const float4* wp = reinterpret_cast<const float4*>(w32 + (size_t) g_row * K + kb * 16 + c8 * 8);
                        const float4 v0 = __ldg(wp + 0);
                        const float4 v1 = __ldg(wp + 1);
                        const uint2 u0 = pack_h2_u2(v0.x, v0.y, v0.z, v0.w);
                        const uint2 u1 = pack_h2_u2(v1.x, v1.y, v1.z, v1.w);
                        *reinterpret_cast<uint4*>(sw_warp0 + wr * 16 + w_swz * 8) = make_uint4(u0.x, u0.y, u1.x, u1.y);
                    } else {
                        *reinterpret_cast<uint4*>(sw_warp0 + wr * 16 + w_swz * 8) = make_uint4(0, 0, 0, 0);
                    }
                }
                __syncwarp();
                uint32_t a0, a1, a2, a3, b0, b1;
                ldmat4(a0, a1, a2, a3, s_ctrl_row + (((uint32_t) a_c8_0 ^ (uint32_t) (kb << 1)) << 4));
                ldmat2(b0, b1, sw_b_addr0);
                mma16816(acc, a0, a1, a2, a3, b0, b1);
                __syncwarp();
            }
        }
    };

    auto stage_ctrl_and_wv = [&](const float* ctrl_ptr, const float* wv_ptr) {
        if (tid < 128) {
            const int r = tid >> 3;        // 0..15
            const int c8 = tid & 7;        // 0..7
            const int swz = c8 ^ (r & 7);
            if (r < rows) {
                const float4* xp = reinterpret_cast<const float4*>(ctrl_ptr + (size_t) (t0 + r) * K) + (c8 << 1);
                const float4 v0 = __ldg(xp + 0);
                const float4 v1 = __ldg(xp + 1);
                const uint2 u0 = pack_h2_u2(v0.x, v0.y, v0.z, v0.w);
                const uint2 u1 = pack_h2_u2(v1.x, v1.y, v1.z, v1.w);
                *reinterpret_cast<uint4*>(s_ctrl + r * 64 + (swz << 3)) = make_uint4(u0.x, u0.y, u1.x, u1.y);
            } else {
                *reinterpret_cast<uint4*>(s_ctrl + r * 64 + (swz << 3)) = make_uint4(0, 0, 0, 0);
            }
        }
        if (tid < BM) {
            s_wv[tid] = (tid < rows) ? __ldg(reinterpret_cast<const float2*>(wv_ptr) + (t0 + tid))
                                     : make_float2(0.0f, 0.0f);
        }
    };

    const float inv_n = (N == 2304) ? (1.0f / 2304.0f) : __fdividef(1.0f, (float) N);

    if constexpr (M_MODS == 3) {
        // Unified BM=16 path for M_MODS==3: 100% m16 TC utilization, 40 CTAs (1 wave at T_CHUNK=640), 29 KB SMEM!
#pragma unroll 1
        for (int m = 2; m >= 0; --m) {
            const float* ctrl_ptr = sel3_c(m, p.ctrl_in[0], p.ctrl_in[1], p.ctrl_in[2]);
            const float* wv_ptr   = sel3_c(m, p.w_vmsg[0], p.w_vmsg[1], p.w_vmsg[2]);
            if (tid == 0) {
                s_mp3->wg16     = sel3_h(m, p.wg_f16[0], p.wg_f16[1], p.wg_f16[2]);
                s_mp3->wf16     = sel3_h(m, p.wf_f16[0], p.wf_f16[1], p.wf_f16[2]);
                s_mp3->wg32     = sel3_c(m, p.w_gain[0], p.w_gain[1], p.w_gain[2]);
                s_mp3->wf32     = sel3_c(m, p.w_forget[0], p.w_forget[1], p.w_forget[2]);
                s_mp3->c_ptr    = sel3_c(m, p.c[0], p.c[1], p.c[2]);
                s_mp3->rw_ptr   = sel3_c(m, ro.rms_w[0], ro.rms_w[1], ro.rms_w[2]);
                s_mp3->of16_ptr = sel3_u16(m, ro.out_f16[0], ro.out_f16[1], ro.out_f16[2]);
                s_mp3->of32_ptr = sel3_o(m, p.out[0], p.out[1], p.out[2]);
            }
            stage_ctrl_and_wv(ctrl_ptr, wv_ptr);
            __syncthreads();

            float y_reg[14][4];
            float y14_0 = 0.0f;
            float* sy_ext = s_ext1 + tid * 7;
            float ss0 = 0.0f, ss1 = 0.0f;

#pragma unroll
            for (int ct = 0; ct < 16; ++ct) {
                const int col_base = col_warp + ct * 8;
                const int col_c0 = col_base + acc_c0;
                const bool c_ok = (col_c0 < N);
                float acc[4];

                run_w_mma_k64(s_mp3->wg16, s_mp3->wg32, col_base, acc);
                float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
                if (r0_ok && c_ok) {
                    const size_t goff0 = (size_t) (t0 + acc_r0) * (size_t) N + (size_t) col_c0;
                    const float2 b0v = __ldg(reinterpret_cast<const float2*>(bank0 + goff0));
                    const float2 b1v = __ldg(reinterpret_cast<const float2*>(bank1 + goff0));
                    const float2 wv0 = s_wv[acc_r0];
                    g0 = fast_tanh(acc[0]) * (wv0.x * b0v.x + wv0.y * b1v.x);
                    g1 = fast_tanh(acc[1]) * (wv0.x * b0v.y + wv0.y * b1v.y);
                }
                if ((acc_r0 + 8 < rows) && c_ok) {
                    const size_t goff1 = (size_t) (t0 + acc_r0 + 8) * (size_t) N + (size_t) col_c0;
                    const float2 b0v = __ldg(reinterpret_cast<const float2*>(bank0 + goff1));
                    const float2 b1v = __ldg(reinterpret_cast<const float2*>(bank1 + goff1));
                    const float2 wv1 = s_wv[acc_r0 + 8];
                    g2 = fast_tanh(acc[2]) * (wv1.x * b0v.x + wv1.y * b1v.x);
                    g3 = fast_tanh(acc[3]) * (wv1.x * b0v.y + wv1.y * b1v.y);
                }

                run_w_mma_k64(s_mp3->wf16, s_mp3->wf32, col_base, acc);
                float y0 = 0.0f, y1 = 0.0f, y2 = 0.0f, y3 = 0.0f;
                if (r0_ok && c_ok) {
                    const size_t goff0 = (size_t) (t0 + acc_r0) * (size_t) N + (size_t) col_c0;
                    const float2 cv = __ldg(reinterpret_cast<const float2*>(s_mp3->c_ptr + goff0));
                    y0 = __fmaf_rn(1.0f - fast_tanh(acc[0]), cv.x, g0);
                    y1 = __fmaf_rn(1.0f - fast_tanh(acc[1]), cv.y, g1);
                    ss0 += y0 * y0 + y1 * y1;
                }
                if ((acc_r0 + 8 < rows) && c_ok) {
                    const size_t goff1 = (size_t) (t0 + acc_r0 + 8) * (size_t) N + (size_t) col_c0;
                    const float2 cv = __ldg(reinterpret_cast<const float2*>(s_mp3->c_ptr + goff1));
                    y2 = __fmaf_rn(1.0f - fast_tanh(acc[2]), cv.x, g2);
                    y3 = __fmaf_rn(1.0f - fast_tanh(acc[3]), cv.y, g3);
                    ss1 += y2 * y2 + y3 * y3;
                }
                if (ct < 14) {
                    y_reg[ct][0] = y0; y_reg[ct][1] = y1; y_reg[ct][2] = y2; y_reg[ct][3] = y3;
                } else if (ct == 14) {
                    y14_0 = y0;
                    sy_ext[0] = y1; sy_ext[1] = y2; sy_ext[2] = y3;
                } else {
                    sy_ext[3] = y0; sy_ext[4] = y1; sy_ext[5] = y2; sy_ext[6] = y3;
                }
            }

            ss0 += __shfl_xor_sync(0xffffffffu, ss0, 1);
            ss0 += __shfl_xor_sync(0xffffffffu, ss0, 2);
            ss1 += __shfl_xor_sync(0xffffffffu, ss1, 1);
            ss1 += __shfl_xor_sync(0xffffffffu, ss1, 2);
            if ((lane & 3) == 0) {
                s_ss[acc_r0 * 19 + warp] = ss0;
                s_ss[(acc_r0 + 8) * 19 + warp] = ss1;
            }
            __syncthreads();

            if (warp == 0 && lane < BM) {
                float tot = 0.0f;
#pragma unroll
                for (int w = 0; w < 18; ++w) tot += s_ss[lane * 19 + w];
                s_ir[lane] = rsqrtf(tot * inv_n + ro.eps);
            }
            __syncthreads();

            const float ir0 = s_ir[acc_r0];
            const float ir1 = s_ir[acc_r0 + 8];
            const float* rw_ptr = s_mp3->rw_ptr;
            uint16_t* of16_ptr  = s_mp3->of16_ptr;
            float* of32_ptr     = s_mp3->of32_ptr;

            auto store_pair3 = [&](int row_l, int ct, float yv0, float yv1, float ir) {
                const int col_c0 = col_warp + ct * 8 + acc_c0;
                if (row_l < rows && col_c0 < N) {
                    const float2 rw = (rw_ptr != nullptr)
                        ? __ldg(reinterpret_cast<const float2*>(rw_ptr + col_c0))
                        : make_float2(1.0f, 1.0f);
                    const float yn0 = yv0 * ir * rw.x;
                    const float yn1 = yv1 * ir * rw.y;
                    const size_t goff = (size_t) (t0 + row_l) * (size_t) N + (size_t) col_c0;
                    if (of32_ptr != nullptr) {
                        *reinterpret_cast<float2*>(of32_ptr + goff) = make_float2(yn0, yn1);
                    }
                    if (of16_ptr != nullptr) {
                        const __half2 h2 = __halves2half2(__float2half_rn(yn0), __float2half_rn(yn1));
                        uint32_t uhi;
                        memcpy(&uhi, &h2, 4);
                        uint32_t* dst = reinterpret_cast<uint32_t*>(of16_ptr + goff);
                        asm volatile("st.global.cs.u32 [%0], %1;" :: "l"(dst), "r"(uhi) : "memory");
                    }
                }
            };

#pragma unroll
            for (int ct = 0; ct < 14; ++ct) {
                store_pair3(acc_r0, ct, y_reg[ct][0], y_reg[ct][1], ir0);
                store_pair3(acc_r0 + 8, ct, y_reg[ct][2], y_reg[ct][3], ir1);
            }
            store_pair3(acc_r0, 14, y14_0, sy_ext[0], ir0);
            store_pair3(acc_r0 + 8, 14, sy_ext[1], sy_ext[2], ir1);
            store_pair3(acc_r0, 15, sy_ext[3], sy_ext[4], ir0);
            store_pair3(acc_r0 + 8, 15, sy_ext[5], sy_ext[6], ir1);
            if (m > 0) __syncthreads();
        }
    } else {
        // M_MODS == 1 (FFN): BM = 16, 100% m16 TC utilization, 29 KB SMEM, cp.async.cg weights, REG <= 94, STACK: 0!
        stage_ctrl_and_wv(p.ctrl_in[0], p.w_vmsg[0]);
        __syncthreads();

        float y_reg[14][4];
        float y14_0 = 0.0f;
        float* sy_ext = s_ext1 + tid * 7; // [0]=y14_1, [1..2]=y14_2..3, [3..6]=y15[0..3]
        float ss0 = 0.0f, ss1 = 0.0f;

#pragma unroll
        for (int ct = 0; ct < 16; ++ct) {
            const int col_base = col_warp + ct * 8;
            const int col_c0 = col_base + acc_c0;
            const bool c_ok = (col_c0 < N);
            float acc[4];

            run_w_mma_k64(p.wg_f16[0], p.w_gain[0], col_base, acc);
            float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
            if (r0_ok && c_ok) {
                const size_t goff0 = (size_t) (t0 + acc_r0) * (size_t) N + (size_t) col_c0;
                const float2 b0v = ldcs_f2(reinterpret_cast<const float2*>(bank0 + goff0));
                const float2 b1v = ldcs_f2(reinterpret_cast<const float2*>(bank1 + goff0));
                const float2 wv0 = s_wv[acc_r0];
                g0 = fast_tanh(acc[0]) * (wv0.x * b0v.x + wv0.y * b1v.x);
                g1 = fast_tanh(acc[1]) * (wv0.x * b0v.y + wv0.y * b1v.y);
            }
            if ((acc_r0 + 8 < rows) && c_ok) {
                const size_t goff1 = (size_t) (t0 + acc_r0 + 8) * (size_t) N + (size_t) col_c0;
                const float2 b0v = ldcs_f2(reinterpret_cast<const float2*>(bank0 + goff1));
                const float2 b1v = ldcs_f2(reinterpret_cast<const float2*>(bank1 + goff1));
                const float2 wv1 = s_wv[acc_r0 + 8];
                g2 = fast_tanh(acc[2]) * (wv1.x * b0v.x + wv1.y * b1v.x);
                g3 = fast_tanh(acc[3]) * (wv1.x * b0v.y + wv1.y * b1v.y);
            }

            run_w_mma_k64(p.wf_f16[0], p.w_forget[0], col_base, acc);
            float y0 = 0.0f, y1 = 0.0f, y2 = 0.0f, y3 = 0.0f;
            if (r0_ok && c_ok) {
                const size_t goff0 = (size_t) (t0 + acc_r0) * (size_t) N + (size_t) col_c0;
                const float2 cv = ldcs_f2(reinterpret_cast<const float2*>(p.c[0] + goff0));
                y0 = __fmaf_rn(1.0f - fast_tanh(acc[0]), cv.x, g0);
                y1 = __fmaf_rn(1.0f - fast_tanh(acc[1]), cv.y, g1);
                ss0 += y0 * y0 + y1 * y1;
            }
            if ((acc_r0 + 8 < rows) && c_ok) {
                const size_t goff1 = (size_t) (t0 + acc_r0 + 8) * (size_t) N + (size_t) col_c0;
                const float2 cv = ldcs_f2(reinterpret_cast<const float2*>(p.c[0] + goff1));
                y2 = __fmaf_rn(1.0f - fast_tanh(acc[2]), cv.x, g2);
                y3 = __fmaf_rn(1.0f - fast_tanh(acc[3]), cv.y, g3);
                ss1 += y2 * y2 + y3 * y3;
            }
            if (ct < 14) {
                y_reg[ct][0] = y0; y_reg[ct][1] = y1; y_reg[ct][2] = y2; y_reg[ct][3] = y3;
            } else if (ct == 14) {
                y14_0 = y0;
                sy_ext[0] = y1; sy_ext[1] = y2; sy_ext[2] = y3;
            } else {
                sy_ext[3] = y0; sy_ext[4] = y1; sy_ext[5] = y2; sy_ext[6] = y3;
            }
        }

        ss0 += __shfl_xor_sync(0xffffffffu, ss0, 1);
        ss0 += __shfl_xor_sync(0xffffffffu, ss0, 2);
        ss1 += __shfl_xor_sync(0xffffffffu, ss1, 1);
        ss1 += __shfl_xor_sync(0xffffffffu, ss1, 2);
        if ((lane & 3) == 0) {
            s_ss[acc_r0 * 19 + warp] = ss0;
            s_ss[(acc_r0 + 8) * 19 + warp] = ss1;
        }
        __syncthreads();

        if (warp == 0 && lane < BM) {
            float tot = 0.0f;
#pragma unroll
            for (int w = 0; w < 18; ++w) tot += s_ss[lane * 19 + w];
            s_ir[lane] = rsqrtf(tot * inv_n + ro.eps);
        }
        __syncthreads();

        const float ir0 = s_ir[acc_r0];
        const float ir1 = s_ir[acc_r0 + 8];
        const float* rw_ptr = ro.rms_w[0];
        uint16_t* of16_ptr  = ro.out_f16[0];
        __half* olo_ptr     = ro.out_lo_f16;
        float* of32_ptr     = p.out[0];
        const bool emit_lo  = (olo_ptr != nullptr);

        auto store_pair = [&](int row_l, int ct, float yv0, float yv1, float ir) {
            const int col_c0 = col_warp + ct * 8 + acc_c0;
            if (row_l < rows && col_c0 < N) {
                const float2 rw = (rw_ptr != nullptr)
                    ? __ldg(reinterpret_cast<const float2*>(rw_ptr + col_c0))
                    : make_float2(1.0f, 1.0f);
                const float yn0 = yv0 * ir * rw.x;
                const float yn1 = yv1 * ir * rw.y;
                const size_t goff = (size_t) (t0 + row_l) * (size_t) N + (size_t) col_c0;
                if (of32_ptr != nullptr) {
                    *reinterpret_cast<float2*>(of32_ptr + goff) = make_float2(yn0, yn1);
                }
                const __half h0 = __float2half_rn(yn0);
                const __half h1 = __float2half_rn(yn1);
                const __half2 h2 = __halves2half2(h0, h1);
                uint32_t uhi;
                memcpy(&uhi, &h2, 4);
                if (emit_lo) {
                    if (of16_ptr != nullptr) {
                        *reinterpret_cast<uint32_t*>(of16_ptr + goff) = uhi;
                    }
                    const __half2 l2 = __halves2half2(__float2half_rn((yn0 - __half2float(h0)) * 2048.0f),
                                                      __float2half_rn((yn1 - __half2float(h1)) * 2048.0f));
                    uint32_t ulo;
                    memcpy(&ulo, &l2, 4);
                    *reinterpret_cast<uint32_t*>(olo_ptr + goff) = ulo;
                } else if (of16_ptr != nullptr) {
                    uint32_t* dst = reinterpret_cast<uint32_t*>(of16_ptr + goff);
                    asm volatile("st.global.cs.u32 [%0], %1;" :: "l"(dst), "r"(uhi) : "memory");
                }
            }
        };

#pragma unroll
        for (int ct = 0; ct < 14; ++ct) {
            store_pair(acc_r0, ct, y_reg[ct][0], y_reg[ct][1], ir0);
            store_pair(acc_r0 + 8, ct, y_reg[ct][2], y_reg[ct][3], ir1);
        }
        store_pair(acc_r0, 14, y14_0, sy_ext[0], ir0);
        store_pair(acc_r0 + 8, 14, sy_ext[1], sy_ext[2], ir1);
        store_pair(acc_r0, 15, sy_ext[3], sy_ext[4], ir0);
        store_pair(acc_r0 + 8, 15, sy_ext[5], sy_ext[6], ir1);
    }
#else
    // Clean exact FP32 fallback for Volta sm_70 / pre-Ampere (STACK: 0, 5,440 B static SMEM).
    __shared__ float s_ctrl_f32[BM][K];
    __shared__ float2 s_wv[BM];
    __shared__ float s_ss[BM][19];
    const int col_warp = warp * 128;

    for (int m = M_MODS - 1; m >= 0; --m) {
        const float* ctrl_ptr = sel3_c(m, p.ctrl_in[0], p.ctrl_in[1], p.ctrl_in[2]);
        const float* wv_ptr   = sel3_c(m, p.w_vmsg[0], p.w_vmsg[1], p.w_vmsg[2]);
        const float* wg_ptr   = sel3_c(m, p.w_gain[0], p.w_gain[1], p.w_gain[2]);
        const float* wf_ptr   = sel3_c(m, p.w_forget[0], p.w_forget[1], p.w_forget[2]);
        const float* c_ptr    = sel3_c(m, p.c[0], p.c[1], p.c[2]);
        const float* rw_ptr   = sel3_c(m, ro.rms_w[0], ro.rms_w[1], ro.rms_w[2]);
        uint16_t* of16_ptr    = sel3_u16(m, ro.out_f16[0], ro.out_f16[1], ro.out_f16[2]);
        float* of32_ptr       = sel3_o(m, p.out[0], p.out[1], p.out[2]);

        for (int idx = tid; idx < BM * K; idx += 576) {
            const int r = idx / K, c = idx % K;
            s_ctrl_f32[r][c] = (r < rows) ? ctrl_ptr[(size_t) (t0 + r) * K + c] : 0.0f;
        }
        if (tid < BM) {
            s_wv[tid] = (tid < rows) ? reinterpret_cast<const float2*>(wv_ptr)[t0 + tid]
                                     : make_float2(0.0f, 0.0f);
        }
        __syncthreads();

        for (int r = 0; r < rows; ++r) {
            float ss = 0.0f;
            const float2 wv = s_wv[r];
            for (int k = 0; k < 4; ++k) {
                const int col = col_warp + k * 32 + lane;
                if (col < N) {
                    float g = 0.0f, f = 0.0f;
                    const float4* wg4 = reinterpret_cast<const float4*>(wg_ptr + (size_t) col * K);
                    const float4* wf4 = reinterpret_cast<const float4*>(wf_ptr + (size_t) col * K);
                    const float4* x4  = reinterpret_cast<const float4*>(s_ctrl_f32[r]);
#pragma unroll 1
                    for (int i = 0; i < 16; ++i) {
                        const float4 xv = x4[i], gv = wg4[i], fv = wf4[i];
                        g += gv.x * xv.x + gv.y * xv.y + gv.z * xv.z + gv.w * xv.w;
                        f += fv.x * xv.x + fv.y * xv.y + fv.z * xv.z + fv.w * xv.w;
                    }
                    const size_t off = (size_t) (t0 + r) * (size_t) N + (size_t) col;
                    const float msg_v = wv.x * bank0[off] + wv.y * bank1[off];
                    const float yv = __fmaf_rn(1.0f - fast_tanh(f), c_ptr[off], fast_tanh(g) * msg_v);
                    ss += yv * yv;
                }
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
            if (lane == 0) s_ss[r][warp] = ss;
        }
        __syncthreads();

        for (int r = 0; r < rows; ++r) {
            float tot = 0.0f;
#pragma unroll
            for (int w = 0; w < 18; ++w) tot += s_ss[r][w];
            const float ir = rsqrtf(tot / (float) N + ro.eps);
            const float2 wv = s_wv[r];
            for (int k = 0; k < 4; ++k) {
                const int col = col_warp + k * 32 + lane;
                if (col < N) {
                    float g = 0.0f, f = 0.0f;
                    const float4* wg4 = reinterpret_cast<const float4*>(wg_ptr + (size_t) col * K);
                    const float4* wf4 = reinterpret_cast<const float4*>(wf_ptr + (size_t) col * K);
                    const float4* x4  = reinterpret_cast<const float4*>(s_ctrl_f32[r]);
#pragma unroll 1
                    for (int i = 0; i < 16; ++i) {
                        const float4 xv = x4[i], gv = wg4[i], fv = wf4[i];
                        g += gv.x * xv.x + gv.y * xv.y + gv.z * xv.z + gv.w * xv.w;
                        f += fv.x * xv.x + fv.y * xv.y + fv.z * xv.z + fv.w * xv.w;
                    }
                    const size_t off = (size_t) (t0 + r) * (size_t) N + (size_t) col;
                    const float msg_v = wv.x * bank0[off] + wv.y * bank1[off];
                    const float yv = __fmaf_rn(1.0f - fast_tanh(f), c_ptr[off], fast_tanh(g) * msg_v);
                    const float rw = (rw_ptr != nullptr) ? rw_ptr[col] : 1.0f;
                    const float yn = yv * ir * rw;
                    if (of32_ptr != nullptr) of32_ptr[off] = yn;
                    const __half h = __float2half_rn(yn);
                    if (of16_ptr != nullptr) {
                        uint16_t u16;
                        memcpy(&u16, &h, 2);
                        of16_ptr[off] = u16;
                    }
                    if constexpr (M_MODS == 1) {
                        if (ro.out_lo_f16 != nullptr) {
                            ro.out_lo_f16[off] = __float2half_rn((yn - __half2float(h)) * 2048.0f);
                        }
                    }
                }
            }
        }
        if (m > 0) __syncthreads();
    }
#endif
}

/// Fast fused control_up + virtual-msg combine (Wave 8.5 Cut B):
/// - M_MODS >= 3 (Q/K/V): BM=64, BN=128, block=512 (16 warps = 4x4, each warp 16x32), 27,648 B SMEM
///   (sx[64*72] + reused sw[128*72]), pre-packed FP16 wg_f16/wf_f16 (0 F2FP), per-ni overlapped bank0/1/c loads,
///   inline PTX st.global.v2.f32, REG <= 64, STACK: 0!
///   Grid = dim3((N+127)/128, (T+63)/64), Block = 512.
/// - M_MODS == 1 (FFN): BM=128 (RT=2x64), BN=64, block=256 (8 warps), 27,648 B SMEM (3 CTAs/SM!),
///   pre-packed FP16 wg_f16/wf_f16, per-mr streaming ldcs_f2 bank0/bank1 (REG <= 72, STACK: 0!), inline PTX st.global.v2.f32.
///   Grid = dim3((N+63)/64, (T+127)/128), Block = 256.
template <int M_MODS>
__global__ void __launch_bounds__((M_MODS >= 3) ? 512 : 256, (M_MODS >= 3) ? 2 : 3) k_control_up_combine_pp_wmma_vmsg(
    const float* __restrict__ bank0, const float* __restrict__ bank1,
    CtrlUpVmsgPtrs3 p, int T, int N) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    constexpr int BM_SUB = 64, K = 64, LD = 72;
    constexpr int BN = 64;
#if __CUDA_ARCH__ < 750 || __CUDA_ARCH__ >= 800
    // sx[64*72] + swg[64*72] + swf[64*72] = 27,648 B (3 CTAs/SM for blk=256, 2 CTAs/SM = 32 warps/SM for blk=512).
    __shared__ __align__(16) __half smem[192 * LD];
#else
    __shared__ __align__(16) float smem_f[2 * BM_SUB * 64];  // 32,768 B -> fits 48 KB default SMEM on Turing
    __half* smem = reinterpret_cast<__half*>(smem_f);
#endif
    __shared__ float2 s_wv[BM_SUB];

    __half* sx  = smem;
    __half* swg = smem + BM_SUB * LD;
    __half* swf = smem + (BM_SUB + 64) * LD;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int col0 = (int) blockIdx.x * BN;

    auto sel3_c = [](int m, const float* a, const float* b, const float* c) -> const float* {
        if constexpr (M_MODS == 1) return a;
        return (m == 0) ? a : ((m == 1) ? b : c);
    };
    auto sel3_h = [](int m, const __half* a, const __half* b, const __half* c) -> const __half* {
        if constexpr (M_MODS == 1) return a;
        return (m == 0) ? a : ((m == 1) ? b : c);
    };
    auto sel3_o = [](int m, float* a, float* b, float* c) -> float* {
        if constexpr (M_MODS == 1) return a;
        return (m == 0) ? a : ((m == 1) ? b : c);
    };

    const int wm = warp >> 2;  // 0..3 for M_MODS>=3 (16 rows/warp), 0..1 for M_MODS==1 (32 rows/warp)
    const int wn = warp & 3;   // 0..3 (32 cols/warp for M_MODS>=3, 16 cols/warp for M_MODS==1)
    const int r_base = tid >> 4;  // 0..31 for 512 thr, 0..15 for 256 thr
    const int c4     = tid & 15;  // 0..15 (4 floats = 2 half2s)
    const int w16_r  = tid >> 3;  // 0..63 for 512 thr, 0..31 for 256 thr
    const int w16_c  = (tid & 7) << 3;  // 0, 8, ..., 56 (8 halfs = 16 B)

#if __CUDA_ARCH__ < 750 || __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 750
    const int qa_row = ((lane & 16) >> 2) + ((lane & 4) << 1) + (lane & 3);
    const int kb_row = ((lane & 16) >> 2) + (lane & 8) + (lane & 3);
    const int acc_r0 = ((lane & 16) >> 2) + ((lane & 4) << 1) + (lane & 1);
    const int acc_r1 = acc_r0 + 2;
    const int acc_c0 = (lane & 8) + (lane & 2);
    const int acc_c1 = acc_c0 + 4;
    const __half* sx_r0  = sx  + (wm * ((M_MODS >= 3) ? 16 : 32) + qa_row) * LD;
    const __half* swg_r0 = swg + (wn * 16 + kb_row) * LD;
    const __half* swf_r0 = swf + (wn * 16 + kb_row) * LD;
    auto hmma_rc = [](float d[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, {%0, %1, %2, %3, %4, %5, %6, %7};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
            : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
    };
#else
    const int a_row = lane & 15, a_col = (lane >> 4) << 3;
    const int b_row = ((lane >> 4) << 3) | (lane & 7), b_col = ((lane >> 3) & 1) << 3;
    const int acc_r0 = lane >> 2, acc_r1 = acc_r0 + 8;
    const int acc_c0 = (lane & 3) * 2, acc_c1 = acc_c0 + 8;
    const uint32_t sx_a_base  = static_cast<uint32_t>(__cvta_generic_to_shared(sx  + (wm * ((M_MODS >= 3) ? 16 : 32) + a_row) * LD + a_col));
    const uint32_t swg_b_base = static_cast<uint32_t>(__cvta_generic_to_shared(swg + (wn * 16 + b_row) * LD + b_col));
    const uint32_t swf_b_base = static_cast<uint32_t>(__cvta_generic_to_shared(swf + (wn * 16 + b_row) * LD + b_col));
    auto ldmat4 = [](uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
    };
    auto mma16816 = [](float d[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    };
#endif

    auto pack_h2_u2 = [](float a, float b, float c, float d) -> uint2 {
        const __half2 h0 = __floats2half2_rn(a, b);
        const __half2 h1 = __floats2half2_rn(c, d);
        uint2 u;
        memcpy(&u.x, &h0, 4);
        memcpy(&u.y, &h1, 4);
        return u;
    };

    if constexpr (M_MODS >= 3) {
        // 512-thread (16 warps = 4x4, BM=64, BN=64): each warp owns ONE 16x16 tile -> only ag[8]+af[8] = 16 acc regs!
        // Wave 9: bank0/1 ldcs_f2 moved AFTER __syncthreads() on m==M_MODS-1 (0 Barrier-5 collision with s_wv!),
        // hoisted 4x batched c_ptr float2 loads before any stg_f2, vectorized STS.64 sx stores -> REG <= 64, STACK: 0!
        const int t0   = (int) blockIdx.y * BM_SUB;
        const int rows = min(BM_SUB, T - t0);
        if (rows <= 0) return;

        const bool shared_c = (M_MODS == 3 && p.c[0] == p.c[1] && p.c[0] == p.c[2]);
        const int r0 = wm * 16 + acc_r0;
        const int d_acc_r = acc_r1 - acc_r0;
        const int d_acc_c = acc_c1 - acc_c0;
        const bool r0_ok = (r0 < rows);
        const bool r1_ok = (r0 + d_acc_r < rows);
        const int col_w  = col0 + wn * 16;
        const bool c0_ok = (col_w + acc_c0 < N);
        const bool c1_ok = (col_w + acc_c1 < N);
        const size_t base0_c0 = (size_t) (t0 + r0) * N + (col_w + acc_c0);

        float2 b0_00{}, b0_01{}, b0_10{}, b0_11{};
        float2 b1_00{}, b1_01{}, b1_10{}, b1_11{};
        float2 c_r0c0{}, c_r0c1{}, c_r1c0{}, c_r1c1{};

        // Reverse order (m = M_MODS - 1 .. 0) ensures out[1..2] are written before out[0] even when out[0] == c[0..2]!
        for (int m = M_MODS - 1; m >= 0; --m) {
            const float* __restrict__ ctrl_ptr = sel3_c(m, p.ctrl_in[0], p.ctrl_in[1], p.ctrl_in[2]);
            const float* __restrict__ wg_ptr   = sel3_c(m, p.w_gain[0], p.w_gain[1], p.w_gain[2]);
            const float* __restrict__ wf_ptr   = sel3_c(m, p.w_forget[0], p.w_forget[1], p.w_forget[2]);
            const __half* __restrict__ wg16    = sel3_h(m, p.wg_f16[0], p.wg_f16[1], p.wg_f16[2]);
            const __half* __restrict__ wf16    = sel3_h(m, p.wf_f16[0], p.wf_f16[1], p.wf_f16[2]);
            const float* __restrict__ wv_ptr   = sel3_c(m, p.w_vmsg[0], p.w_vmsg[1], p.w_vmsg[2]);

            // 1. Stage sx (64x64) + swg (64x64) + swf (64x64) + s_wv (64) simultaneously -> ONLY 1 barrier!
#pragma unroll
            for (int step = 0; step < 64; step += 32) {
                const int r = r_base + step;
                const int c = c4 << 2;
                if (r < rows) {
                    const float4 vx = __ldg(reinterpret_cast<const float4*>(ctrl_ptr + (size_t) (t0 + r) * K) + c4);
                    *reinterpret_cast<uint2*>(sx + r * LD + c) = pack_h2_u2(vx.x, vx.y, vx.z, vx.w);
                } else {
                    *reinterpret_cast<uint2*>(sx + r * LD + c) = make_uint2(0, 0);
                }
            }
            if (wg16 != nullptr && wf16 != nullptr) {
                // 512 threads = 64 rows x 8 uint4s -> ONE single 16B load per thread for swg and swf!
                if (col0 + w16_r < N) {
                    *reinterpret_cast<uint4*>(swg + w16_r * LD + w16_c) =
                        __ldg(reinterpret_cast<const uint4*>(wg16 + (size_t) (col0 + w16_r) * K + w16_c));
                    *reinterpret_cast<uint4*>(swf + w16_r * LD + w16_c) =
                        __ldg(reinterpret_cast<const uint4*>(wf16 + (size_t) (col0 + w16_r) * K + w16_c));
                } else {
                    const uint4 z = make_uint4(0, 0, 0, 0);
                    *reinterpret_cast<uint4*>(swg + w16_r * LD + w16_c) = z;
                    *reinterpret_cast<uint4*>(swf + w16_r * LD + w16_c) = z;
                }
            } else {
#pragma unroll
                for (int step = 0; step < 64; step += 32) {
                    const int r = r_base + step;
                    const int c = c4 << 2;
                    if (col0 + r < N) {
                        const float4 vg = __ldg(reinterpret_cast<const float4*>(wg_ptr + (size_t) (col0 + r) * K) + c4);
                        *reinterpret_cast<uint2*>(swg + r * LD + c) = pack_h2_u2(vg.x, vg.y, vg.z, vg.w);
                        const float4 vf = __ldg(reinterpret_cast<const float4*>(wf_ptr + (size_t) (col0 + r) * K) + c4);
                        *reinterpret_cast<uint2*>(swf + r * LD + c) = pack_h2_u2(vf.x, vf.y, vf.z, vf.w);
                    } else {
                        *reinterpret_cast<uint2*>(swg + r * LD + c) = make_uint2(0, 0);
                        *reinterpret_cast<uint2*>(swf + r * LD + c) = make_uint2(0, 0);
                    }
                }
            }
            if (tid < BM_SUB) {
                s_wv[tid] = (tid < rows) ? __ldg(reinterpret_cast<const float2*>(wv_ptr) + (t0 + tid))
                                         : make_float2(0.0f, 0.0f);
            }
            __syncthreads();

            // Issue bank0/bank1 ldcs_f2 (and shared p.c[0] ONCE on SWA!) right after BAR.SYNC on m==M_MODS-1 (overlaps HMMA!)
            if (m == M_MODS - 1) {
                const size_t base1_c0 = base0_c0 + (size_t) d_acc_r * N;
                if (r0_ok) {
                    if (c0_ok) {
                        b0_00 = ldcs_f2(reinterpret_cast<const float2*>(bank0 + base0_c0));
                        b1_00 = ldcs_f2(reinterpret_cast<const float2*>(bank1 + base0_c0));
                        if (shared_c) c_r0c0 = *reinterpret_cast<const float2*>(p.c[0] + base0_c0);
                    }
                    if (c1_ok) {
                        b0_01 = ldcs_f2(reinterpret_cast<const float2*>(bank0 + base0_c0 + d_acc_c));
                        b1_01 = ldcs_f2(reinterpret_cast<const float2*>(bank1 + base0_c0 + d_acc_c));
                        if (shared_c) c_r0c1 = *reinterpret_cast<const float2*>(p.c[0] + base0_c0 + d_acc_c);
                    }
                }
                if (r1_ok) {
                    if (c0_ok) {
                        b0_10 = ldcs_f2(reinterpret_cast<const float2*>(bank0 + base1_c0));
                        b1_10 = ldcs_f2(reinterpret_cast<const float2*>(bank1 + base1_c0));
                        if (shared_c) c_r1c0 = *reinterpret_cast<const float2*>(p.c[0] + base1_c0);
                    }
                    if (c1_ok) {
                        b0_11 = ldcs_f2(reinterpret_cast<const float2*>(bank0 + base1_c0 + d_acc_c));
                        b1_11 = ldcs_f2(reinterpret_cast<const float2*>(bank1 + base1_c0 + d_acc_c));
                        if (shared_c) c_r1c1 = *reinterpret_cast<const float2*>(p.c[0] + base1_c0 + d_acc_c);
                    }
                }
            }

            // 2. Compute ag[8] + af[8] (single 16x16 per warp)
            float ag[8], af[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) { ag[i] = 0.0f; af[i] = 0.0f; }

#if __CUDA_ARCH__ < 750
#pragma unroll
            for (int kk = 0; kk < K; kk += 8) {
                const uint4 a  = *reinterpret_cast<const uint4*>(sx_r0 + kk);
                const uint4 bg = *reinterpret_cast<const uint4*>(swg_r0 + kk);
                const uint4 bf = *reinterpret_cast<const uint4*>(swf_r0 + kk);
                hmma_rc(ag, a.x, a.y, bg.x, bg.y);
                hmma_rc(ag, a.z, a.w, bg.z, bg.w);
                hmma_rc(af, a.x, a.y, bf.x, bf.y);
                hmma_rc(af, a.z, a.w, bf.z, bf.w);
            }
#else
            (void) swf_b_base;
#pragma unroll
            for (int kk = 0; kk < K; kk += 16) {
                const uint32_t off_b = (uint32_t) kk * 2u;
                uint32_t a0, a1, a2, a3, b0, b1, b2, b3;
                ldmat4(a0, a1, a2, a3, sx_a_base + off_b);
                ldmat4(b0, b1, b2, b3, swg_b_base + off_b);
                mma16816(&ag[0], a0, a1, a2, a3, b0, b1);
                mma16816(&ag[4], a0, a1, a2, a3, b2, b3);
                ldmat4(b0, b1, b2, b3, swg_b_base + (uint32_t) (64 * LD * sizeof(__half)) + off_b);
                mma16816(&af[0], a0, a1, a2, a3, b0, b1);
                mma16816(&af[4], a0, a1, a2, a3, b2, b3);
            }
#endif

            float* out_ptr = sel3_o(m, p.out[0], p.out[1], p.out[2]);
            const size_t base1_c0 = base0_c0 + (size_t) d_acc_r * N;
            if (!shared_c) {
                const float* c_ptr = sel3_c(m, p.c[0], p.c[1], p.c[2]);
                // Hoist all 4 c_ptr float2 loads BEFORE any stg_f2 -> 1 parallel L2/DRAM batch hidden behind 16 MUFU.TANH!
                if (r0_ok) {
                    if (c0_ok) c_r0c0 = *reinterpret_cast<const float2*>(c_ptr + base0_c0);
                    if (c1_ok) c_r0c1 = *reinterpret_cast<const float2*>(c_ptr + base0_c0 + d_acc_c);
                }
                if (r1_ok) {
                    if (c0_ok) c_r1c0 = *reinterpret_cast<const float2*>(c_ptr + base1_c0);
                    if (c1_ok) c_r1c1 = *reinterpret_cast<const float2*>(c_ptr + base1_c0 + d_acc_c);
                }
            }
            const float2 wv0 = r0_ok ? s_wv[r0] : make_float2(0.0f, 0.0f);
            const float2 wv1 = r1_ok ? s_wv[r0 + d_acc_r] : make_float2(0.0f, 0.0f);
            if (r0_ok) {
                if (c0_ok) {
                    const float m0 = wv0.x * b0_00.x + wv0.y * b1_00.x;
                    const float m1 = wv0.x * b0_00.y + wv0.y * b1_00.y;
                    stg_f2(reinterpret_cast<float2*>(out_ptr + base0_c0),
                           __fmaf_rn(1.0f - fast_tanh(af[0]), c_r0c0.x, fast_tanh(ag[0]) * m0),
                           __fmaf_rn(1.0f - fast_tanh(af[1]), c_r0c0.y, fast_tanh(ag[1]) * m1));
                }
                if (c1_ok) {
                    const float m0 = wv0.x * b0_01.x + wv0.y * b1_01.x;
                    const float m1 = wv0.x * b0_01.y + wv0.y * b1_01.y;
                    stg_f2(reinterpret_cast<float2*>(out_ptr + base0_c0 + d_acc_c),
                           __fmaf_rn(1.0f - fast_tanh(af[4]), c_r0c1.x, fast_tanh(ag[4]) * m0),
                           __fmaf_rn(1.0f - fast_tanh(af[5]), c_r0c1.y, fast_tanh(ag[5]) * m1));
                }
            }
            if (r1_ok) {
                if (c0_ok) {
                    const float m0 = wv1.x * b0_10.x + wv1.y * b1_10.x;
                    const float m1 = wv1.x * b0_10.y + wv1.y * b1_10.y;
                    stg_f2(reinterpret_cast<float2*>(out_ptr + base1_c0),
                           __fmaf_rn(1.0f - fast_tanh(af[2]), c_r1c0.x, fast_tanh(ag[2]) * m0),
                           __fmaf_rn(1.0f - fast_tanh(af[3]), c_r1c0.y, fast_tanh(ag[3]) * m1));
                }
                if (c1_ok) {
                    const float m0 = wv1.x * b0_11.x + wv1.y * b1_11.x;
                    const float m1 = wv1.x * b0_11.y + wv1.y * b1_11.y;
                    stg_f2(reinterpret_cast<float2*>(out_ptr + base1_c0 + d_acc_c),
                           __fmaf_rn(1.0f - fast_tanh(af[6]), c_r1c1.x, fast_tanh(ag[6]) * m0),
                           __fmaf_rn(1.0f - fast_tanh(af[7]), c_r1c1.y, fast_tanh(ag[7]) * m1));
                }
            }
            if (m > 0) __syncthreads();
        }
    } else {
        // M_MODS == 1 (FFN): BM=128 (RT=2x64), BN=64, block=256 (8 warps, 3 CTAs/SM!).
        // Wave 9: per-mr bank0/bank1 ldcs_f2 moved AFTER __syncthreads() (0 Barrier-5 collision!),
        // vectorized STS.64 sx stores, batched 4x c_ptr loads -> REG <= 72, STACK: 0!
        const int col_w = col0 + wn * 16;
        const bool c0_ok = (col_w + acc_c0 < N);
        const bool c1_ok = (col_w + acc_c1 < N);
        const int t0_cta = (int) blockIdx.y * (2 * BM_SUB);
        if (t0_cta >= T) return;

        const float* __restrict__ ctrl_ptr = p.ctrl_in[0];
        const float* __restrict__ wg_ptr   = p.w_gain[0];
        const float* __restrict__ wf_ptr   = p.w_forget[0];
        const __half* __restrict__ wg16    = p.wg_f16[0];
        const __half* __restrict__ wf16    = p.wf_f16[0];
        const float* __restrict__ wv_ptr   = p.w_vmsg[0];
        const float* c_ptr                 = p.c[0];
        float* out_ptr                     = p.out[0];

        if (wg16 != nullptr && wf16 != nullptr) {
#pragma unroll
            for (int step = 0; step < 64; step += 32) {
                const int r = w16_r + step;
                if (col0 + r < N) {
                    *reinterpret_cast<uint4*>(swg + r * LD + w16_c) =
                        __ldg(reinterpret_cast<const uint4*>(wg16 + (size_t) (col0 + r) * K + w16_c));
                    *reinterpret_cast<uint4*>(swf + r * LD + w16_c) =
                        __ldg(reinterpret_cast<const uint4*>(wf16 + (size_t) (col0 + r) * K + w16_c));
                } else {
                    const uint4 z = make_uint4(0, 0, 0, 0);
                    *reinterpret_cast<uint4*>(swg + r * LD + w16_c) = z;
                    *reinterpret_cast<uint4*>(swf + r * LD + w16_c) = z;
                }
            }
        } else {
#pragma unroll
            for (int step = 0; step < 64; step += 16) {
                const int r = r_base + step;
                const int c = c4 << 2;
                if (col0 + r < N) {
                    const float4 vg = __ldg(reinterpret_cast<const float4*>(wg_ptr + (size_t) (col0 + r) * K) + c4);
                    *reinterpret_cast<uint2*>(swg + r * LD + c) = pack_h2_u2(vg.x, vg.y, vg.z, vg.w);
                    const float4 vf = __ldg(reinterpret_cast<const float4*>(wf_ptr + (size_t) (col0 + r) * K) + c4);
                    *reinterpret_cast<uint2*>(swf + r * LD + c) = pack_h2_u2(vf.x, vf.y, vf.z, vf.w);
                } else {
                    *reinterpret_cast<uint2*>(swg + r * LD + c) = make_uint2(0, 0);
                    *reinterpret_cast<uint2*>(swf + r * LD + c) = make_uint2(0, 0);
                }
            }
        }

        for (int rt = 0; rt < 2; ++rt) {
            const int t0   = t0_cta + rt * BM_SUB;
            const int rows = min(BM_SUB, T - t0);
            if (rows <= 0) break;

#pragma unroll
            for (int step = 0; step < 64; step += 16) {
                const int r = r_base + step;
                const int c = c4 << 2;
                if (r < rows) {
                    const float4 vx = __ldg(reinterpret_cast<const float4*>(ctrl_ptr + (size_t) (t0 + r) * K) + c4);
                    *reinterpret_cast<uint2*>(sx + r * LD + c) = pack_h2_u2(vx.x, vx.y, vx.z, vx.w);
                } else {
                    *reinterpret_cast<uint2*>(sx + r * LD + c) = make_uint2(0, 0);
                }
            }
            if (tid < BM_SUB) {
                s_wv[tid] = (tid < rows) ? __ldg(reinterpret_cast<const float2*>(wv_ptr) + (t0 + tid))
                                         : make_float2(0.0f, 0.0f);
            }
            __syncthreads();

#pragma unroll
            for (int mr = 0; mr < 2; ++mr) {
                const int r0 = wm * 32 + mr * 16 + acc_r0;
                const int r1 = wm * 32 + mr * 16 + acc_r1;
                const bool r0_ok = (r0 < rows);
                const bool r1_ok = (r1 < rows);
                const size_t base0 = (size_t) (t0 + r0) * N + col_w;
                const size_t base1 = (size_t) (t0 + r1) * N + col_w;

                float2 b0_00{}, b0_01{}, b0_10{}, b0_11{};
                float2 b1_00{}, b1_01{}, b1_10{}, b1_11{};
#if __CUDA_ARCH__ >= 750
                // Issue per-mr bank0/bank1 ldcs_f2 right after BAR.SYNC / before mr WMMA loop (0 Barrier-5 collision!)
                if (r0_ok) {
                    if (c0_ok) {
                        b0_00 = ldcs_f2(reinterpret_cast<const float2*>(bank0 + base0 + acc_c0));
                        b1_00 = ldcs_f2(reinterpret_cast<const float2*>(bank1 + base0 + acc_c0));
                    }
                    if (c1_ok) {
                        b0_01 = ldcs_f2(reinterpret_cast<const float2*>(bank0 + base0 + acc_c1));
                        b1_01 = ldcs_f2(reinterpret_cast<const float2*>(bank1 + base0 + acc_c1));
                    }
                }
                if (r1_ok) {
                    if (c0_ok) {
                        b0_10 = ldcs_f2(reinterpret_cast<const float2*>(bank0 + base1 + acc_c0));
                        b1_10 = ldcs_f2(reinterpret_cast<const float2*>(bank1 + base1 + acc_c0));
                    }
                    if (c1_ok) {
                        b0_11 = ldcs_f2(reinterpret_cast<const float2*>(bank0 + base1 + acc_c1));
                        b1_11 = ldcs_f2(reinterpret_cast<const float2*>(bank1 + base1 + acc_c1));
                    }
                }
#endif

                float ag[8], af[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) { ag[i] = 0.0f; af[i] = 0.0f; }

#if __CUDA_ARCH__ < 750
#pragma unroll 1
                for (int kk = 0; kk < K; kk += 8) {
                    const uint4 a = *reinterpret_cast<const uint4*>(sx_r0 + mr * 16 * LD + kk);
                    {
                        const uint4 bg = *reinterpret_cast<const uint4*>(swg_r0 + kk);
                        hmma_rc(ag, a.x, a.y, bg.x, bg.y);
                        hmma_rc(ag, a.z, a.w, bg.z, bg.w);
                    }
                    {
                        const uint4 bf = *reinterpret_cast<const uint4*>(swf_r0 + kk);
                        hmma_rc(af, a.x, a.y, bf.x, bf.y);
                        hmma_rc(af, a.z, a.w, bf.z, bf.w);
                    }
                }
#else
                const uint32_t sx_mr = sx_a_base + (uint32_t) (mr * 16 * LD * 2);
#pragma unroll
                for (int kk = 0; kk < K; kk += 16) {
                    const uint32_t off_b = (uint32_t) kk * 2u;
                    uint32_t a0, a1, a2, a3, b0, b1, b2, b3;
                    ldmat4(a0, a1, a2, a3, sx_mr + off_b);
                    ldmat4(b0, b1, b2, b3, swg_b_base + off_b);
                    mma16816(&ag[0], a0, a1, a2, a3, b0, b1);
                    mma16816(&ag[4], a0, a1, a2, a3, b2, b3);
                    ldmat4(b0, b1, b2, b3, swf_b_base + off_b);
                    mma16816(&af[0], a0, a1, a2, a3, b0, b1);
                    mma16816(&af[4], a0, a1, a2, a3, b2, b3);
                }
#endif

#if __CUDA_ARCH__ < 750
                if (r0_ok) {
                    if (c0_ok) {
                        b0_00 = ldcs_f2(reinterpret_cast<const float2*>(bank0 + base0 + acc_c0));
                        b1_00 = ldcs_f2(reinterpret_cast<const float2*>(bank1 + base0 + acc_c0));
                    }
                    if (c1_ok) {
                        b0_01 = ldcs_f2(reinterpret_cast<const float2*>(bank0 + base0 + acc_c1));
                        b1_01 = ldcs_f2(reinterpret_cast<const float2*>(bank1 + base0 + acc_c1));
                    }
                }
                if (r1_ok) {
                    if (c0_ok) {
                        b0_10 = ldcs_f2(reinterpret_cast<const float2*>(bank0 + base1 + acc_c0));
                        b1_10 = ldcs_f2(reinterpret_cast<const float2*>(bank1 + base1 + acc_c0));
                    }
                    if (c1_ok) {
                        b0_11 = ldcs_f2(reinterpret_cast<const float2*>(bank0 + base1 + acc_c1));
                        b1_11 = ldcs_f2(reinterpret_cast<const float2*>(bank1 + base1 + acc_c1));
                    }
                }
#endif
                float2 c_r0c0{}, c_r0c1{}, c_r1c0{}, c_r1c1{};
                if (r0_ok) {
                    if (c0_ok) c_r0c0 = *reinterpret_cast<const float2*>(c_ptr + base0 + acc_c0);
                    if (c1_ok) c_r0c1 = *reinterpret_cast<const float2*>(c_ptr + base0 + acc_c1);
                }
                if (r1_ok) {
                    if (c0_ok) c_r1c0 = *reinterpret_cast<const float2*>(c_ptr + base1 + acc_c0);
                    if (c1_ok) c_r1c1 = *reinterpret_cast<const float2*>(c_ptr + base1 + acc_c1);
                }
                if (r0_ok) {
                    const float2 wv = s_wv[r0];
                    if (c0_ok) {
                        const float m0 = wv.x * b0_00.x + wv.y * b1_00.x;
                        const float m1 = wv.x * b0_00.y + wv.y * b1_00.y;
                        stg_f2(reinterpret_cast<float2*>(out_ptr + base0 + acc_c0),
                               __fmaf_rn(1.0f - fast_tanh(af[0]), c_r0c0.x, fast_tanh(ag[0]) * m0),
                               __fmaf_rn(1.0f - fast_tanh(af[1]), c_r0c0.y, fast_tanh(ag[1]) * m1));
                    }
                    if (c1_ok) {
                        const float m0 = wv.x * b0_01.x + wv.y * b1_01.x;
                        const float m1 = wv.x * b0_01.y + wv.y * b1_01.y;
                        stg_f2(reinterpret_cast<float2*>(out_ptr + base0 + acc_c1),
                               __fmaf_rn(1.0f - fast_tanh(af[4]), c_r0c1.x, fast_tanh(ag[4]) * m0),
                               __fmaf_rn(1.0f - fast_tanh(af[5]), c_r0c1.y, fast_tanh(ag[5]) * m1));
                    }
                }
                if (r1_ok) {
                    const float2 wv = s_wv[r1];
                    if (c0_ok) {
                        const float m0 = wv.x * b0_10.x + wv.y * b1_10.x;
                        const float m1 = wv.x * b0_10.y + wv.y * b1_10.y;
                        stg_f2(reinterpret_cast<float2*>(out_ptr + base1 + acc_c0),
                               __fmaf_rn(1.0f - fast_tanh(af[2]), c_r1c0.x, fast_tanh(ag[2]) * m0),
                               __fmaf_rn(1.0f - fast_tanh(af[3]), c_r1c0.y, fast_tanh(ag[3]) * m1));
                    }
                    if (c1_ok) {
                        const float m0 = wv.x * b0_11.x + wv.y * b1_11.x;
                        const float m1 = wv.x * b0_11.y + wv.y * b1_11.y;
                        stg_f2(reinterpret_cast<float2*>(out_ptr + base1 + acc_c1),
                               __fmaf_rn(1.0f - fast_tanh(af[6]), c_r1c1.x, fast_tanh(ag[6]) * m0),
                               __fmaf_rn(1.0f - fast_tanh(af[7]), c_r1c1.y, fast_tanh(ag[7]) * m1));
                    }
                }
            }
            if (rt == 0) __syncthreads();
        }
    }
#else
    // Turing sm_75 WMMA fallback (handles BN=128 for M_MODS>=3 and BN=64 for M_MODS==1 within 32,768 B SMEM)
    constexpr int RT_MAX = (M_MODS >= 3) ? 1 : 2;
    constexpr int NTHR   = (M_MODS >= 3) ? 512 : 256;
    constexpr int RSTEP  = (M_MODS >= 3) ? 32 : 16;
    constexpr int N_NI   = (M_MODS >= 3) ? 2 : 1;
    const int t0_cta = (int) blockIdx.y * (RT_MAX * BM_SUB);
    if (t0_cta >= T) return;
    for (int rt = 0; rt < RT_MAX; ++rt) {
        const int t0 = t0_cta + rt * BM_SUB;
        const int rows = min(BM_SUB, T - t0);
        if (rows <= 0) break;
        for (int m = M_MODS - 1; m >= 0; --m) {
            const float* ctrl_ptr = sel3_c(m, p.ctrl_in[0], p.ctrl_in[1], p.ctrl_in[2]);
            const float* wg_ptr   = sel3_c(m, p.w_gain[0], p.w_gain[1], p.w_gain[2]);
            const float* wf_ptr   = sel3_c(m, p.w_forget[0], p.w_forget[1], p.w_forget[2]);
            const float* wv_ptr   = sel3_c(m, p.w_vmsg[0], p.w_vmsg[1], p.w_vmsg[2]);
            const float* c_ptr    = sel3_c(m, p.c[0], p.c[1], p.c[2]);
            float* out_ptr        = sel3_o(m, p.out[0], p.out[1], p.out[2]);
            for (int ni_blk = 0; ni_blk < N_NI; ++ni_blk) {
                const int col_sub = col0 + ni_blk * 64;
                __half* swg_t = smem + BM_SUB * LD;
                __half* swf_t = smem + (BM_SUB + 64) * LD;
#pragma unroll
                for (int step = 0; step < 64; step += RSTEP) {
                    const int r = r_base + step;
                    const int c = c4 << 2;
                    if (r < rows) {
                        const float4 vx = __ldg(reinterpret_cast<const float4*>(ctrl_ptr + (size_t) (t0 + r) * K) + c4);
                        *reinterpret_cast<__half2*>(sx + r * LD + c)     = __floats2half2_rn(vx.x, vx.y);
                        *reinterpret_cast<__half2*>(sx + r * LD + c + 2) = __floats2half2_rn(vx.z, vx.w);
                    } else {
                        *reinterpret_cast<uint2*>(sx + r * LD + c) = make_uint2(0, 0);
                    }
                    if (col_sub + r < N) {
                        const float4 vg = __ldg(reinterpret_cast<const float4*>(wg_ptr + (size_t) (col_sub + r) * K) + c4);
                        *reinterpret_cast<__half2*>(swg_t + r * LD + c)     = __floats2half2_rn(vg.x, vg.y);
                        *reinterpret_cast<__half2*>(swg_t + r * LD + c + 2) = __floats2half2_rn(vg.z, vg.w);
                        const float4 vf = __ldg(reinterpret_cast<const float4*>(wf_ptr + (size_t) (col_sub + r) * K) + c4);
                        *reinterpret_cast<__half2*>(swf_t + r * LD + c)     = __floats2half2_rn(vf.x, vf.y);
                        *reinterpret_cast<__half2*>(swf_t + r * LD + c + 2) = __floats2half2_rn(vf.z, vf.w);
                    } else {
                        *reinterpret_cast<uint2*>(swg_t + r * LD + c) = make_uint2(0, 0);
                        *reinterpret_cast<uint2*>(swf_t + r * LD + c) = make_uint2(0, 0);
                    }
                }
                if (tid < BM_SUB) {
                    s_wv[tid] = (tid < rows) ? __ldg(reinterpret_cast<const float2*>(wv_ptr) + (t0 + tid))
                                             : make_float2(0.0f, 0.0f);
                }
                __syncthreads();

                constexpr int N_MR = (M_MODS >= 3) ? 1 : 2;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_g[N_MR], acc_f[N_MR];
#pragma unroll
                for (int mr = 0; mr < N_MR; ++mr) {
                    wmma::fill_fragment(acc_g[mr], 0.0f);
                    wmma::fill_fragment(acc_f[mr], 0.0f);
                }
#pragma unroll
                for (int kk = 0; kk < K; kk += 16) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fbg, fbf;
                    wmma::load_matrix_sync(fbg, swg_t + (wn * 16) * LD + kk, LD);
                    wmma::load_matrix_sync(fbf, swf_t + (wn * 16) * LD + kk, LD);
#pragma unroll
                    for (int mr = 0; mr < N_MR; ++mr) {
                        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
                        wmma::load_matrix_sync(fa, sx + (wm * (N_MR * 16) + mr * 16) * LD + kk, LD);
                        wmma::mma_sync(acc_g[mr], fa, fbg, acc_g[mr]);
                        wmma::mma_sync(acc_f[mr], fa, fbf, acc_f[mr]);
                    }
                }
#pragma unroll
                for (int mr = 0; mr < N_MR; ++mr) {
#pragma unroll
                    for (int i = 0; i < acc_g[mr].num_elements; ++i) {
                        acc_g[mr].x[i] = fast_tanh(acc_g[mr].x[i]);
                        acc_f[mr].x[i] = 1.0f - fast_tanh(acc_f[mr].x[i]);
                    }
                }
                __syncthreads();
                float* s_g = smem_f;
                float* s_f = smem_f + BM_SUB * 64;
#pragma unroll
                for (int mr = 0; mr < N_MR; ++mr) {
                    wmma::store_matrix_sync(s_g + (wm * (N_MR * 16) + mr * 16) * 64 + wn * 16, acc_g[mr], 64, wmma::mem_row_major);
                    wmma::store_matrix_sync(s_f + (wm * (N_MR * 16) + mr * 16) * 64 + wn * 16, acc_f[mr], 64, wmma::mem_row_major);
                }
                __syncthreads();

                for (int i4 = tid; i4 < rows * 16; i4 += NTHR) {
                    const int r = i4 >> 4, c = (i4 & 15) << 2;
                    if (col_sub + c >= N) continue;
                    const size_t off = (size_t) (t0 + r) * N + col_sub + c;
                    const float2 wv = s_wv[r];
                    const float4 cv  = *reinterpret_cast<const float4*>(c_ptr + off);
                    const float4 b0v = ldcs_f4(reinterpret_cast<const float4*>(bank0 + off));
                    const float4 b1v = ldcs_f4(reinterpret_cast<const float4*>(bank1 + off));
                    const float4 gv  = reinterpret_cast<const float4*>(s_g)[i4];
                    const float4 fv  = reinterpret_cast<const float4*>(s_f)[i4];
                    *reinterpret_cast<float4*>(out_ptr + off) = make_float4(
                        __fmaf_rn(fv.x, cv.x, gv.x * (wv.x * b0v.x + wv.y * b1v.x)),
                        __fmaf_rn(fv.y, cv.y, gv.y * (wv.x * b0v.y + wv.y * b1v.y)),
                        __fmaf_rn(fv.z, cv.z, gv.z * (wv.x * b0v.z + wv.y * b1v.z)),
                        __fmaf_rn(fv.w, cv.w, gv.w * (wv.x * b0v.w + wv.y * b1v.w)));
                }
                if (ni_blk + 1 < N_NI || m > 0 || rt + 1 < RT_MAX) __syncthreads();
            }
        }
    }
#endif
#else
    (void) bank0; (void) bank1; (void) p; (void) T; (void) N;
#endif
}

__global__ void k_control_combine(const float* __restrict__ cur, const float* __restrict__ msg,
                                  const float* __restrict__ gain, const float* __restrict__ forget,
                                  float* __restrict__ out, int64_t n) {
    if ((n & 3) == 0) {
        const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
        const int64_t n4 = n >> 2;
        if (i >= n4) return;
        const float4 c = reinterpret_cast<const float4*>(cur)[i];
        const float4 m = reinterpret_cast<const float4*>(msg)[i];
        const float4 g = reinterpret_cast<const float4*>(gain)[i];
        const float4 f = reinterpret_cast<const float4*>(forget)[i];
        reinterpret_cast<float4*>(out)[i] = make_float4(
            (1.0f - fast_tanh(f.x)) * c.x + fast_tanh(g.x) * m.x,
            (1.0f - fast_tanh(f.y)) * c.y + fast_tanh(g.y) * m.y,
            (1.0f - fast_tanh(f.z)) * c.z + fast_tanh(g.z) * m.z,
            (1.0f - fast_tanh(f.w)) * c.w + fast_tanh(g.w) * m.w);
        return;
    }
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = fast_tanh(gain[i]);
    const float f = fast_tanh(forget[i]);
    out[i] = (1.0f - f) * cur[i] + g * msg[i];
}

/// Fused T=1 loop_gain + loop_forget ([R=64 -> N] F32) + k_control_combine: 4 warps/block (128 thr), 2 rows/warp = 8 rows/CTA (288 CTAs = 3.6/SM on V100).
__global__ void __launch_bounds__(128, 8) k_control_up_combine_t1(
        const float* __restrict__ cur, const float* __restrict__ msg,
        const float* __restrict__ w_gain, const float* __restrict__ w_forget,
        const float* __restrict__ ctrl64, float* __restrict__ out,
        int R, int N) {
    __shared__ float sx[64];
    if (threadIdx.x < R) sx[threadIdx.x] = ctrl64[threadIdx.x];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int base_o = (blockIdx.x * 4 + warp) * 2;
    const int half = lane >> 4;
    const int i = lane & 15;
    const float4 x = reinterpret_cast<const float4*>(sx)[i];
    const float* wg = half ? w_forget : w_gain;
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        const int o = base_o + k;
        if (o >= N) return;
        const float4 w = reinterpret_cast<const float4*>(wg + (size_t) o * R)[i];
        float acc = w.x * x.x + w.y * x.y + w.z * x.z + w.w * x.w;
#pragma unroll
        for (int m = 8; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
        const float f_val = __shfl_sync(0xffffffffu, acc, 16);
        if (lane == 0) {
            out[o] = (1.0f - fast_tanh(f_val)) * __ldg(&cur[o]) + fast_tanh(acc) * __ldg(&msg[o]);
        }
    }
}

/// Fused small-T (2..8) loop_gain + loop_forget ([R=64 -> N] F32) + k_control_combine:
/// Loads W_gain/W_forget ONCE per output row across all T tokens against SMEM ctrl64[T][64].
/// Eliminates 2x cuBLAS SGEMM + splitKreduce + k_control_combine per module during verify decode.
template <int MT, bool EXACT_T>
__global__ void k_control_up_combine_smallT(const float* __restrict__ cur, const float* __restrict__ msg,
                                            const float* __restrict__ w_gain, const float* __restrict__ w_forget,
                                            const float* __restrict__ ctrl64, float* __restrict__ out,
                                            int T, int N) {
    __shared__ float sx[MT][64];
    const int tot_x = (EXACT_T ? MT : T) * 64;
    for (int idx = threadIdx.x; idx < tot_x; idx += 256) {
        sx[idx >> 6][idx & 63] = ctrl64[idx];
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int base_o = (blockIdx.x * 8 + warp) * 2;
    const int half = lane >> 4;
    const int i = lane & 15;
    const float* wg = half ? w_forget : w_gain;
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        const int o = base_o + k;
        if (o >= N) return;
        const float4 w = reinterpret_cast<const float4*>(wg + (size_t) o * 64)[i];
        float acc[MT];
#pragma unroll
        for (int t = 0; t < MT; ++t) {
            if (EXACT_T || t < T) {
                const float4 x = reinterpret_cast<const float4*>(sx[t])[i];
                acc[t] = w.x * x.x + w.y * x.y + w.z * x.z + w.w * x.w;
            }
        }
#pragma unroll
        for (int t = 0; t < MT; ++t) {
            if (EXACT_T || t < T) {
                float a = acc[t];
#pragma unroll
                for (int m = 8; m > 0; m >>= 1) a += __shfl_xor_sync(0xffffffffu, a, m);
                const float f_val = __shfl_sync(0xffffffffu, a, 16);
                if (lane == 0) {
                    const size_t idx = (size_t) t * N + o;
                    out[idx] = (1.0f - fast_tanh(f_val)) * cur[idx] + fast_tanh(a) * msg[idx];
                }
            }
        }
    }
}

/// 3-module batched k_control_up_combine_smallT for Q/K/V loop controllers (grid = ((N+15)/16, 3)).
struct CtrlUpPtrs3 {
    const float *wg0, *wg1, *wg2;
    const float *wf0, *wf1, *wf2;
    const float *c0, *c1, *c2;
    const float *m0, *m1, *m2;
    float *o0, *o1, *o2;
};

template <int MT, bool EXACT_T>
__global__ void k_control_up_combine_smallT_3(CtrlUpPtrs3 cp, const float* __restrict__ ctrl64_base,
                                              int T, int N) {
    const int mod = blockIdx.y;
    if (mod >= 3) return;
    __shared__ float sx[MT][64];
    const int n_t = EXACT_T ? MT : T;
    const float* ctrl64 = ctrl64_base + (size_t) mod * (size_t) n_t * 64;
    const int tot_x = n_t * 64;
    for (int idx = threadIdx.x; idx < tot_x; idx += 256) {
        sx[idx >> 6][idx & 63] = ctrl64[idx];
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int base_o = (blockIdx.x * 8 + warp) * 2;
    const int half = lane >> 4;
    const int i = lane & 15;
    const float* wg_m = (mod == 0) ? cp.wg0 : ((mod == 1) ? cp.wg1 : cp.wg2);
    const float* wf_m = (mod == 0) ? cp.wf0 : ((mod == 1) ? cp.wf1 : cp.wf2);
    const float* wg = half ? wf_m : wg_m;
    const float* cur = (mod == 0) ? cp.c0 : ((mod == 1) ? cp.c1 : cp.c2);
    const float* msg = (mod == 0) ? cp.m0 : ((mod == 1) ? cp.m1 : cp.m2);
    float* out = (mod == 0) ? cp.o0 : ((mod == 1) ? cp.o1 : cp.o2);
    if constexpr (MT == 1 && EXACT_T) {
        const float4 x0 = reinterpret_cast<const float4*>(sx[0])[i];
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int o = base_o + k;
            if (o >= N) return;
            const float4 w = reinterpret_cast<const float4*>(wg + (size_t) o * 64)[i];
            float a = w.x * x0.x + w.y * x0.y + w.z * x0.z + w.w * x0.w;
#pragma unroll
            for (int m = 8; m > 0; m >>= 1) a += __shfl_xor_sync(0xffffffffu, a, m);
            const float f_val = __shfl_sync(0xffffffffu, a, 16);
            if (lane == 0) {
                out[o] = (1.0f - fast_tanh(f_val)) * __ldg(&cur[o]) + fast_tanh(a) * __ldg(&msg[o]);
            }
        }
        return;
    } else {
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        const int o = base_o + k;
        if (o >= N) return;
        const float4 w = reinterpret_cast<const float4*>(wg + (size_t) o * 64)[i];
        float acc[MT];
#pragma unroll
        for (int t = 0; t < MT; ++t) {
            if (EXACT_T || t < T) {
                const float4 x = reinterpret_cast<const float4*>(sx[t])[i];
                acc[t] = w.x * x.x + w.y * x.y + w.z * x.z + w.w * x.w;
            }
        }
#pragma unroll
        for (int t = 0; t < MT; ++t) {
            if (EXACT_T || t < T) {
                float a = acc[t];
#pragma unroll
                for (int m = 8; m > 0; m >>= 1) a += __shfl_xor_sync(0xffffffffu, a, m);
                const float f_val = __shfl_sync(0xffffffffu, a, 16);
                if (lane == 0) {
                    const size_t idx = (size_t) t * N + o;
                    out[idx] = (1.0f - fast_tanh(f_val)) * cur[idx] + fast_tanh(a) * msg[idx];
                }
            }
        }
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

/// Small-T (2..8) F32 GEMM for small n_out (<= 128): 1 CTA of 256 threads per output row,
/// reads W ONCE across all T tokens (eliminates cuBLAS SGEMM + splitKreduce overhead on loop_down / router).
template <int MT, bool EXACT_T>
__global__ void k_gemm_f32_smallT_cta(const float* __restrict__ W, const float* __restrict__ X, float* __restrict__ Y,
                                      int n_in, int n_out, int T) {
    __shared__ float s_sum[MT][8];
    const int o = blockIdx.x;
    if (o >= n_out) return;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int n4 = n_in >> 2;
    const float4* wr = reinterpret_cast<const float4*>(W + (size_t) o * n_in);
    float acc[MT];
#pragma unroll
    for (int t = 0; t < MT; ++t) acc[t] = 0.0f;
    for (int i = tid; i < n4; i += 256) {
        const float4 a = wr[i];
#pragma unroll
        for (int t = 0; t < MT; ++t) {
            if (EXACT_T || t < T) {
                const float4 b = reinterpret_cast<const float4*>(X + (size_t) t * n_in)[i];
                acc[t] += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < MT; ++t) {
        if (EXACT_T || t < T) {
            float v = acc[t];
#pragma unroll
            for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
            if (lane == 0) s_sum[t][warp] = v;
        }
    }
    __syncthreads();
    if (tid < (EXACT_T ? MT : T)) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) s += s_sum[tid][w];
        Y[(size_t) tid * n_out + o] = s;
    }
}

/// Small-T (2..8) F32 GEMM for larger n_out (> 128): 8 warps/block, 1 warp per output row, reads W ONCE across T tokens.
template <int MT, bool EXACT_T>
__global__ void k_gemm_f32_smallT_warp(const float* __restrict__ W, const float* __restrict__ X, float* __restrict__ Y,
                                       int n_in, int n_out, int T) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int o = blockIdx.x * 8 + warp;
    if (o >= n_out) return;
    const int n4 = n_in >> 2;
    const float4* wr = reinterpret_cast<const float4*>(W + (size_t) o * n_in);
    float acc[MT];
#pragma unroll
    for (int t = 0; t < MT; ++t) acc[t] = 0.0f;
    for (int i = lane; i < n4; i += 32) {
        const float4 a = wr[i];
#pragma unroll
        for (int t = 0; t < MT; ++t) {
            if (EXACT_T || t < T) {
                const float4 b = reinterpret_cast<const float4*>(X + (size_t) t * n_in)[i];
                acc[t] += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < MT; ++t) {
        if (EXACT_T || t < T) {
            float v = acc[t];
#pragma unroll
            for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
            if (lane == 0) Y[(size_t) t * n_out + o] = v;
        }
    }
}

/// Small-T (2..8) BF16-weight x FP32-activation GEMM (n_out <= 128): 1 CTA of 256 threads per output row.
/// Reads BF16 weights ONCE via 16-byte LDG.128 (uint4 = 8 BF16s/load) when n_in % 8 == 0 across all T tokens.
template <int MT, bool EXACT_T>
__global__ void k_gemm_bf16_smallT_cta(const uint16_t* __restrict__ W, const float* __restrict__ X, float* __restrict__ Y,
                                       int n_in, int n_out, int T) {
    __shared__ float s_sum[MT][8];
    const int o = blockIdx.x;
    if (o >= n_out) return;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    float acc[MT];
#pragma unroll
    for (int t = 0; t < MT; ++t) acc[t] = 0.0f;
    if ((n_in & 7) == 0) {
        const int n8 = n_in >> 3;
        const uint4* wr8 = reinterpret_cast<const uint4*>(W + (size_t) o * n_in);
        for (int i = tid; i < n8; i += 256) {
            const uint4 w4 = wr8[i];
            const float f0 = __uint_as_float((w4.x & 0xFFFFu) << 16);
            const float f1 = __uint_as_float(w4.x & 0xFFFF0000u);
            const float f2 = __uint_as_float((w4.y & 0xFFFFu) << 16);
            const float f3 = __uint_as_float(w4.y & 0xFFFF0000u);
            const float f4 = __uint_as_float((w4.z & 0xFFFFu) << 16);
            const float f5 = __uint_as_float(w4.z & 0xFFFF0000u);
            const float f6 = __uint_as_float((w4.w & 0xFFFFu) << 16);
            const float f7 = __uint_as_float(w4.w & 0xFFFF0000u);
#pragma unroll
            for (int t = 0; t < MT; ++t) {
                if (EXACT_T || t < T) {
                    const float4* xt = reinterpret_cast<const float4*>(X + (size_t) t * n_in) + (i << 1);
                    const float4 b0 = xt[0];
                    const float4 b1 = xt[1];
                    acc[t] += (f0 * b0.x + f1 * b0.y + f2 * b0.z + f3 * b0.w) +
                              (f4 * b1.x + f5 * b1.y + f6 * b1.z + f7 * b1.w);
                }
            }
        }
    } else {
        const int n4 = n_in >> 2;
        const uint2* wr = reinterpret_cast<const uint2*>(W + (size_t) o * n_in);
        for (int i = tid; i < n4; i += 256) {
            const uint2 w2 = wr[i];
            const float w0 = __uint_as_float((w2.x & 0xFFFFu) << 16);
            const float w1 = __uint_as_float(w2.x & 0xFFFF0000u);
            const float w2f = __uint_as_float((w2.y & 0xFFFFu) << 16);
            const float w3 = __uint_as_float(w2.y & 0xFFFF0000u);
#pragma unroll
            for (int t = 0; t < MT; ++t) {
                if (EXACT_T || t < T) {
                    const float4 b = reinterpret_cast<const float4*>(X + (size_t) t * n_in)[i];
                    acc[t] += w0 * b.x + w1 * b.y + w2f * b.z + w3 * b.w;
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < MT; ++t) {
        if (EXACT_T || t < T) {
            float v = acc[t];
#pragma unroll
            for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
            if (lane == 0) s_sum[t][warp] = v;
        }
    }
    __syncthreads();
    if (tid < (EXACT_T ? MT : T)) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) s += s_sum[tid][w];
        Y[(size_t) tid * n_out + o] = s;
    }
}

/// Small-T (2..8) BF16-weight x FP32-activation GEMM (n_out > 128): 8 warps/block, 1 warp per output row.
/// Small-T (1..8) BF16-weight x FP32-activation GEMM (n_out > 128): 8 warps/CTA, 1 warp per output row.
/// Uses 16-byte LDG.128 (uint4 = 8 BF16s/load) when n_in % 8 == 0, plus optional PAIR fusion (W1->Y1, W2->Y2).
template <int MT, bool EXACT_T, bool PAIR = false>
__global__ void k_gemm_bf16_smallT_warp(const uint16_t* __restrict__ W, const float* __restrict__ X, float* __restrict__ Y,
                                        int n_in, int n_out, int T,
                                        const uint16_t* __restrict__ W2 = nullptr, float* __restrict__ Y2 = nullptr, int n_out2 = 0) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int o_all = blockIdx.x * 8 + warp;
    const int tot_out = PAIR ? (n_out + n_out2) : n_out;
    if (o_all >= tot_out) return;
    const bool use2 = PAIR && (o_all >= n_out);
    const int o = use2 ? (o_all - n_out) : o_all;
    const uint16_t* Wsrc = use2 ? W2 : W;
    float* Ydst = use2 ? Y2 : Y;
    const int out_stride = use2 ? n_out2 : n_out;

    float acc[MT];
#pragma unroll
    for (int t = 0; t < MT; ++t) acc[t] = 0.0f;

    if ((n_in & 7) == 0) {
        const int n8 = n_in >> 3;
        const uint4* wr = reinterpret_cast<const uint4*>(Wsrc + (size_t) o * n_in);
        for (int i = lane; i < n8; i += 32) {
            const uint4 w4 = wr[i];
            const float w0 = __uint_as_float((w4.x & 0xFFFFu) << 16);
            const float w1 = __uint_as_float(w4.x & 0xFFFF0000u);
            const float w2 = __uint_as_float((w4.y & 0xFFFFu) << 16);
            const float w3 = __uint_as_float(w4.y & 0xFFFF0000u);
            const float w4f = __uint_as_float((w4.z & 0xFFFFu) << 16);
            const float w5 = __uint_as_float(w4.z & 0xFFFF0000u);
            const float w6 = __uint_as_float((w4.w & 0xFFFFu) << 16);
            const float w7 = __uint_as_float(w4.w & 0xFFFF0000u);
#pragma unroll
            for (int t = 0; t < MT; ++t) {
                if (EXACT_T || t < T) {
                    const float4* xr = reinterpret_cast<const float4*>(X + (size_t) t * n_in);
                    const float4 b0 = xr[i * 2];
                    const float4 b1 = xr[i * 2 + 1];
                    acc[t] += (w0 * b0.x + w1 * b0.y + w2 * b0.z + w3 * b0.w)
                            + (w4f * b1.x + w5 * b1.y + w6 * b1.z + w7 * b1.w);
                }
            }
        }
    } else {
        const int n4 = n_in >> 2;
        const uint2* wr = reinterpret_cast<const uint2*>(Wsrc + (size_t) o * n_in);
        for (int i = lane; i < n4; i += 32) {
            const uint2 w2 = wr[i];
            const float w0 = __uint_as_float((w2.x & 0xFFFFu) << 16);
            const float w1 = __uint_as_float(w2.x & 0xFFFF0000u);
            const float w2f = __uint_as_float((w2.y & 0xFFFFu) << 16);
            const float w3 = __uint_as_float(w2.y & 0xFFFF0000u);
#pragma unroll
            for (int t = 0; t < MT; ++t) {
                if (EXACT_T || t < T) {
                    const float4 b = reinterpret_cast<const float4*>(X + (size_t) t * n_in)[i];
                    acc[t] += w0 * b.x + w1 * b.y + w2f * b.z + w3 * b.w;
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < MT; ++t) {
        if (EXACT_T || t < T) {
            float v = acc[t];
#pragma unroll
            for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
            if (lane == 0) Ydst[(size_t) t * out_stride + o] = v;
        }
    }
}

/// Small-T (1..8) FP16-weight x FP32-activation GEMM (n_out <= 128): 1 CTA of 256 threads per output row.
template <int MT, bool EXACT_T>
__global__ void k_gemm_f16_smallT_cta(const uint16_t* __restrict__ W, const float* __restrict__ X, float* __restrict__ Y,
                                      int n_in, int n_out, int T) {
    __shared__ float s_sum[MT][8];
    const int o = blockIdx.x;
    if (o >= n_out) return;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int n4 = n_in >> 2;
    const uint2* wr = reinterpret_cast<const uint2*>(W + (size_t) o * n_in);
    float acc[MT];
#pragma unroll
    for (int t = 0; t < MT; ++t) acc[t] = 0.0f;
    for (int i = tid; i < n4; i += 256) {
        const uint2 w2 = wr[i];
        const float2 f01 = __half22float2(*reinterpret_cast<const __half2*>(&w2.x));
        const float2 f23 = __half22float2(*reinterpret_cast<const __half2*>(&w2.y));
#pragma unroll
        for (int t = 0; t < MT; ++t) {
            if (EXACT_T || t < T) {
                const float4 b = reinterpret_cast<const float4*>(X + (size_t) t * n_in)[i];
                acc[t] += f01.x * b.x + f01.y * b.y + f23.x * b.z + f23.y * b.w;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < MT; ++t) {
        if (EXACT_T || t < T) {
            float v = acc[t];
#pragma unroll
            for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
            if (lane == 0) s_sum[t][warp] = v;
        }
    }
    __syncthreads();
    if (tid < (EXACT_T ? MT : T)) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) s += s_sum[tid][w];
        Y[(size_t) tid * n_out + o] = s;
    }
}

/// Small-T (1..8) FP16-weight x FP32-activation GEMM (n_out > 128): 8 warps/block, 1 warp per output row.
template <int MT, bool EXACT_T>
__global__ void k_gemm_f16_smallT_warp(const uint16_t* __restrict__ W, const float* __restrict__ X, float* __restrict__ Y,
                                       int n_in, int n_out, int T) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int o = blockIdx.x * 8 + warp;
    if (o >= n_out) return;
    const int n4 = n_in >> 2;
    const uint2* wr = reinterpret_cast<const uint2*>(W + (size_t) o * n_in);
    float acc[MT];
#pragma unroll
    for (int t = 0; t < MT; ++t) acc[t] = 0.0f;
    for (int i = lane; i < n4; i += 32) {
        const uint2 w2 = wr[i];
        const float2 f01 = __half22float2(*reinterpret_cast<const __half2*>(&w2.x));
        const float2 f23 = __half22float2(*reinterpret_cast<const __half2*>(&w2.y));
#pragma unroll
        for (int t = 0; t < MT; ++t) {
            if (EXACT_T || t < T) {
                const float4 b = reinterpret_cast<const float4*>(X + (size_t) t * n_in)[i];
                acc[t] += f01.x * b.x + f01.y * b.y + f23.x * b.z + f23.y * b.w;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < MT; ++t) {
        if (EXACT_T || t < T) {
            float v = acc[t];
#pragma unroll
            for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
            if (lane == 0) Y[(size_t) t * n_out + o] = v;
        }
    }
}

/// High-occupancy zero-SMEM, zero-sync Q8_0 x Q8_1 small-T (2..8) MMVQ:
/// 4 warps/block (128 threads, 4 output rows/block), 100% occupancy (0 B SMEM, 0 __syncthreads).
/// Branchless 4-byte aligned LDG.32 + funnelshift for 34-byte Q8_0 blocks (8 blocks/warp/iter -> 0% tail waste for nb=72/96/288).
/// Supports optional PAIR=true (fused 2-matrix launch on shared xq, e.g. Q+Gate, K+V, DSpark Gate+Up).
/// Requires (K_in % 256) == 0, 4-byte aligned W (and W2 when PAIR), and (n_out1 % 4) == 0 when PAIR.
template <int MT, bool EXACT_T, bool PAIR, int WPB = 4>
__global__ void __launch_bounds__(WPB * 32, (WPB == 1) ? 16 : ((WPB == 2) ? 16 : ((MT <= 4) ? 12 : 8))) k_mmvq_q8_0_smallT(
    const uint8_t* __restrict__ w1, const uint32_t* __restrict__ xq_u32, float* __restrict__ y1,
    int n_in, int n_out1, int T,
    const uint8_t* __restrict__ w2 = nullptr, float* __restrict__ y2 = nullptr, int n_out2 = 0) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nb = n_in >> 5;  // Q8_0 / Q8_1 blocks per row (multiple of 8)
    int row = (int) blockIdx.x * WPB + warp;
    const uint8_t* w_cur = w1;
    float* y_cur = y1;
    int n_out_cur = n_out1;
    if constexpr (PAIR) {
        if (row >= n_out1) {
            row -= n_out1;
            w_cur = w2;
            y_cur = y2;
            n_out_cur = n_out2;
        }
    }
    if (row >= n_out_cur) return;
    const uint8_t* w_row = w_cur + (size_t) row * (size_t) nb * 34u;
    const int blk_in_chunk = lane >> 2;  // 0..7
    const int blk_odd = blk_in_chunk & 1;
    const int iqs = (lane & 3) * 2;      // 0, 2, 4, 6
    const uint32_t sh = blk_odd ? 0u : 16u;
    const int p_off = (blk_odd ? 2 : 0) + iqs * 4;

    float acc[MT];
#pragma unroll
    for (int t = 0; t < MT; ++t) acc[t] = 0.0f;

    for (int kb0 = 0; kb0 < nb; kb0 += 8) {
        const int kb = kb0 + blk_in_chunk;
        const uint8_t* bp = w_row + (size_t) kb * 34u;
        const float d0 = __half2float(__ldg(reinterpret_cast<const __half*>(bp)));
        const uint32_t* p32 = reinterpret_cast<const uint32_t*>(bp + p_off);
        const uint32_t w0 = __ldg(p32);
        const uint32_t w1_u = __ldg(p32 + 1);
        const uint32_t w2_u = blk_odd ? 0u : __ldg(p32 + 2);
        const int v0 = (int) __funnelshift_r(w0, w1_u, sh);
        const int v1 = (int) __funnelshift_r(w1_u, w2_u, sh);
#pragma unroll
        for (int t = 0; t < MT; ++t) {
            if (EXACT_T || t < T) {
                const uint32_t* xb = xq_u32 + ((size_t) t * nb + kb) * 9u;
                const uint32_t ds_u = __ldg(xb);
                const float d1 = __low2float(*reinterpret_cast<const __half2*>(&ds_u));
                int sumi = dp4a_i8(v0, (int) __ldg(xb + 1 + iqs), 0);
                sumi = dp4a_i8(v1, (int) __ldg(xb + 2 + iqs), sumi);
                acc[t] += d0 * d1 * (float) sumi;
            }
        }
    }

#pragma unroll
    for (int t = 0; t < MT; ++t) {
        if (EXACT_T || t < T) {
            float v = acc[t];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0) y_cur[(size_t) t * n_out_cur + row] = v;
        }
    }
}

/// Fused 3-way (Q+Gate+R) or 5-way (Q+Gate+R+K+V) zero-SMEM Q8_0 x Q8_1 small-T MMVQ with per-matrix xq pointers.
/// Scalar struct members (no dynamic array indexing) guarantee STACK:0 on sm_70 & sm_89!
struct Q8MultiSpec {
    const uint8_t *w0, *w1, *w2, *w3, *w4;
    const uint32_t *x0, *x1, *x2, *x3, *x4;
    float *y0, *y1, *y2, *y3, *y4;
    int n0, n1, n2, n3, n4;
};

template <int MT, bool EXACT_T, int NMAT, int WPB = 4>
__global__ void __launch_bounds__(WPB * 32, (WPB == 2) ? 16 : ((MT <= 4) ? 10 : 8)) k_mmvq_q8_0_smallT_multi(
    Q8MultiSpec sp, int n_in, int T) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nb = n_in >> 5;
    int row = (int) blockIdx.x * WPB + warp;
    const uint8_t* w_cur = sp.w0;
    const uint32_t* xq_u32 = sp.x0;
    float* y_cur = sp.y0;
    int n_out_cur = sp.n0;
    if (row >= sp.n0) {
        row -= sp.n0;
        if (row < sp.n1) {
            w_cur = sp.w1; xq_u32 = sp.x1; y_cur = sp.y1; n_out_cur = sp.n1;
        } else {
            row -= sp.n1;
            if constexpr (NMAT == 3) {
                w_cur = sp.w2; xq_u32 = sp.x2; y_cur = sp.y2; n_out_cur = sp.n2;
            } else {
                if (row < sp.n2) {
                    w_cur = sp.w2; xq_u32 = sp.x2; y_cur = sp.y2; n_out_cur = sp.n2;
                } else {
                    row -= sp.n2;
                    if (row < sp.n3) {
                        w_cur = sp.w3; xq_u32 = sp.x3; y_cur = sp.y3; n_out_cur = sp.n3;
                    } else {
                        row -= sp.n3;
                        w_cur = sp.w4; xq_u32 = sp.x4; y_cur = sp.y4; n_out_cur = sp.n4;
                    }
                }
            }
        }
    }
    if (row >= n_out_cur) return;
    const uint8_t* w_row = w_cur + (size_t) row * (size_t) nb * 34u;
    const int blk_in_chunk = lane >> 2;
    const int blk_odd = blk_in_chunk & 1;
    const int iqs = (lane & 3) * 2;
    const uint32_t sh = blk_odd ? 0u : 16u;
    const int p_off = (blk_odd ? 2 : 0) + iqs * 4;

    float acc[MT];
#pragma unroll
    for (int t = 0; t < MT; ++t) acc[t] = 0.0f;

    for (int kb0 = 0; kb0 < nb; kb0 += 8) {
        const int kb = kb0 + blk_in_chunk;
        const uint8_t* bp = w_row + (size_t) kb * 34u;
        const float d0 = __half2float(__ldg(reinterpret_cast<const __half*>(bp)));
        const uint32_t* p32 = reinterpret_cast<const uint32_t*>(bp + p_off);
        const uint32_t w0 = __ldg(p32);
        const uint32_t w1_u = __ldg(p32 + 1);
        const uint32_t w2_u = blk_odd ? 0u : __ldg(p32 + 2);
        const int v0 = (int) __funnelshift_r(w0, w1_u, sh);
        const int v1 = (int) __funnelshift_r(w1_u, w2_u, sh);
#pragma unroll
        for (int t = 0; t < MT; ++t) {
            if (EXACT_T || t < T) {
                const uint32_t* xb = xq_u32 + ((size_t) t * nb + kb) * 9u;
                const uint32_t ds_u = __ldg(xb);
                const float d1 = __low2float(*reinterpret_cast<const __half2*>(&ds_u));
                int sumi = dp4a_i8(v0, (int) __ldg(xb + 1 + iqs), 0);
                sumi = dp4a_i8(v1, (int) __ldg(xb + 2 + iqs), sumi);
                acc[t] += d0 * d1 * (float) sumi;
            }
        }
    }

#pragma unroll
    for (int t = 0; t < MT; ++t) {
        if (EXACT_T || t < T) {
            float v = acc[t];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0) y_cur[(size_t) t * n_out_cur + row] = v;
        }
    }
}

/// Causal 4-tap shortconv (+ optional dynamic per-head taps for full-attn K/V).
/// hist has [hist_stride, 3] floats? Wait: hist is [n_channels_total][3] for coalesced / per-channel state:
/// specifically hist[channel_offset + c][0..2] holds the 3 oldest->newest prior inputs.
/// Dynamic weight `dyn` is [T][n_heads][4] where lag 0 = current token, lag 3 = t-3.
/// Static weight `w` is [n_ch][4] where w[0] = t-3, w[3] = current token.
__global__ void k_shortconv(const float* __restrict__ cur, const float* __restrict__ w,
                            const float* __restrict__ dyn, float* __restrict__ hist,
                            float* __restrict__ out, int n_ch, int hist_offset, int n_heads, int T,
                            const float* __restrict__ res_add = nullptr,
                            float* __restrict__ hist_trace = nullptr, size_t hist_trace_stride = 0) {
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
        if (hist_trace) {
            float* ht = hist_trace + (size_t) t * hist_trace_stride + (size_t) (hist_offset + c) * 3;
            ht[0] = h0;
            ht[1] = h1;
            ht[2] = h2;
        }
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
                               int kv_dim, int n_heads, int T,
                               float* __restrict__ hist_trace = nullptr, size_t hist_trace_stride = 0) {
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
        if (hist_trace) {
            float* ht = hist_trace + (size_t) t * hist_trace_stride + (size_t) c_all * 3;
            ht[0] = h0;
            ht[1] = h1;
            ht[2] = h2;
        }
    }
    hc[0] = h0;
    hc[1] = h1;
    hc[2] = h2;
}

/// Parallel tiled shortconv (prefill, T >= 32, read-only on `hist` from blockIdx.y==0):
/// Grid: ((n_ch+255)/256, (T+15)/16), Block: 256. Followed by k_shortconv_hist_push.
__global__ void k_shortconv_pp(const float* __restrict__ cur, const float* __restrict__ w,
                               const float* __restrict__ hist, float* __restrict__ out,
                               int n_ch, int hist_offset, int T, const float* __restrict__ res_add) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_ch) return;
    const int t0 = blockIdx.y * 16;
    const int t1 = min(T, t0 + 16);
    float h0, h1, h2;
    if (t0 == 0) {
        const float* hc = hist + (size_t) (hist_offset + c) * 3;
        h0 = hc[0]; h1 = hc[1]; h2 = hc[2];
    } else {
        h0 = cur[(size_t) (t0 - 3) * n_ch + c];
        h1 = cur[(size_t) (t0 - 2) * n_ch + c];
        h2 = cur[(size_t) (t0 - 1) * n_ch + c];
    }
    const float4 w4 = *reinterpret_cast<const float4*>(w + (size_t) c * 4);
    const float scale = rsqrtf(5.0f);
    for (int t = t0; t < t1; ++t) {
        const size_t idx = (size_t) t * n_ch + c;
        const float x = cur[idx];
        const float s = h0 * w4.x + h1 * w4.y + h2 * w4.z + x * w4.w;
        float y = (x + s) * scale;
        if (res_add) y += res_add[idx];
        out[idx] = y;
        h0 = h1; h1 = h2; h2 = x;
    }
}

/// EX-4C Wave 4: Vectorized float4 + t_tile=64 k_shortconv_pp (cuts 19% overlap re-reads to 4.7%, 549 CTAs for full SM occupancy).
__global__ void k_shortconv_pp_f4(const float* __restrict__ cur, const float* __restrict__ w,
                                  const float* __restrict__ hist, float* __restrict__ out,
                                  int n_ch, int hist_offset, int T, const float* __restrict__ res_add) {
    const int n4 = n_ch >> 2;
    const int i4 = (int) blockIdx.x * blockDim.x + threadIdx.x;
    if (i4 >= n4) return;
    const int c0 = i4 << 2;
    const int t0 = (int) blockIdx.y * 64;
    const int t1 = min(T, t0 + 64);
    float4 h0, h1, h2;
    if (t0 == 0) {
        const float* hc0 = hist + (size_t) (hist_offset + c0 + 0) * 3;
        const float* hc1 = hist + (size_t) (hist_offset + c0 + 1) * 3;
        const float* hc2 = hist + (size_t) (hist_offset + c0 + 2) * 3;
        const float* hc3 = hist + (size_t) (hist_offset + c0 + 3) * 3;
        h0 = make_float4(hc0[0], hc1[0], hc2[0], hc3[0]);
        h1 = make_float4(hc0[1], hc1[1], hc2[1], hc3[1]);
        h2 = make_float4(hc0[2], hc1[2], hc2[2], hc3[2]);
    } else {
        h0 = reinterpret_cast<const float4*>(cur + (size_t) (t0 - 3) * n_ch)[i4];
        h1 = reinterpret_cast<const float4*>(cur + (size_t) (t0 - 2) * n_ch)[i4];
        h2 = reinterpret_cast<const float4*>(cur + (size_t) (t0 - 1) * n_ch)[i4];
    }
    const float4 wc0 = __ldg(reinterpret_cast<const float4*>(w + (size_t) (c0 + 0) * 4));
    const float4 wc1 = __ldg(reinterpret_cast<const float4*>(w + (size_t) (c0 + 1) * 4));
    const float4 wc2 = __ldg(reinterpret_cast<const float4*>(w + (size_t) (c0 + 2) * 4));
    const float4 wc3 = __ldg(reinterpret_cast<const float4*>(w + (size_t) (c0 + 3) * 4));
    const float scale = rsqrtf(5.0f);
    for (int t = t0; t < t1; ++t) {
        const float4 x = reinterpret_cast<const float4*>(cur + (size_t) t * n_ch)[i4];
        float4 y;
        y.x = (x.x + (h0.x * wc0.x + h1.x * wc0.y + h2.x * wc0.z + x.x * wc0.w)) * scale;
        y.y = (x.y + (h0.y * wc1.x + h1.y * wc1.y + h2.y * wc1.z + x.y * wc1.w)) * scale;
        y.z = (x.z + (h0.z * wc2.x + h1.z * wc2.y + h2.z * wc2.z + x.z * wc2.w)) * scale;
        y.w = (x.w + (h0.w * wc3.x + h1.w * wc3.y + h2.w * wc3.z + x.w * wc3.w)) * scale;
        if (res_add) {
            const float4 r = reinterpret_cast<const float4*>(res_add + (size_t) t * n_ch)[i4];
            y.x += r.x; y.y += r.y; y.z += r.z; y.w += r.w;
        }
        reinterpret_cast<float4*>(out + (size_t) t * n_ch)[i4] = y;
        h0 = h1; h1 = h2; h2 = x;
    }
}

__global__ void k_shortconv_kv_pp(const float* __restrict__ K, const float* __restrict__ V,
                                  const float* __restrict__ wk, const float* __restrict__ wv,
                                  const float* __restrict__ dyn_k, const float* __restrict__ dyn_v,
                                  const float* __restrict__ hist, float* __restrict__ K_out, float* __restrict__ V_out,
                                  int kv_dim, int n_heads, int T) {
    const int c_all = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_all >= 2 * kv_dim) return;
    const bool is_v = (c_all >= kv_dim);
    const int c = is_v ? (c_all - kv_dim) : c_all;
    const float* cur = is_v ? V : K;
    const float* dyn = is_v ? dyn_v : dyn_k;
    float* out = is_v ? V_out : K_out;
    const int t0 = blockIdx.y * 16;
    const int t1 = min(T, t0 + 16);
    float h0, h1, h2;
    if (t0 == 0) {
        const float* hc = hist + (size_t) c_all * 3;
        h0 = hc[0]; h1 = hc[1]; h2 = hc[2];
    } else {
        h0 = cur[(size_t) (t0 - 3) * kv_dim + c];
        h1 = cur[(size_t) (t0 - 2) * kv_dim + c];
        h2 = cur[(size_t) (t0 - 1) * kv_dim + c];
    }
    const float4 w4 = *reinterpret_cast<const float4*>((is_v ? wv : wk) + (size_t) c * 4);
    const int head_dim = n_heads > 0 ? (kv_dim / n_heads) : 0;
    const int h_idx = n_heads > 0 ? (c / head_dim) : 0;
    const float scale = rsqrtf(dyn ? 9.0f : 5.0f);
    for (int t = t0; t < t1; ++t) {
        const size_t idx = (size_t) t * kv_dim + c;
        const float x = cur[idx];
        float s = h0 * w4.x + h1 * w4.y + h2 * w4.z + x * w4.w;
        if (dyn) {
            const float* dt = dyn + ((size_t) t * n_heads + h_idx) * 4;
            s += x * dt[0] + h2 * dt[1] + h1 * dt[2] + h0 * dt[3];
        }
        out[idx] = (x + s) * scale;
        h0 = h1; h1 = h2; h2 = x;
    }
}

/// Update `hist` after parallel k_shortconv_pp / k_shortconv_kv_pp (for T >= 3).
__global__ void k_shortconv_hist_push(const float* __restrict__ cur, float* __restrict__ hist,
                                      int n_ch, int hist_offset, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_ch) return;
    float* hc = hist + (size_t) (hist_offset + c) * 3;
    hc[0] = cur[(size_t) (T - 3) * n_ch + c];
    hc[1] = cur[(size_t) (T - 2) * n_ch + c];
    hc[2] = cur[(size_t) (T - 1) * n_ch + c];
}

__global__ void k_shortconv_kv_hist_push(const float* __restrict__ K, const float* __restrict__ V,
                                         float* __restrict__ hist, int kv_dim, int T, int ld_kv = 0) {
    const int c_all = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_all >= 2 * kv_dim) return;
    const bool is_v = (c_all >= kv_dim);
    const int c = is_v ? (c_all - kv_dim) : c_all;
    const int kv_stride = (ld_kv > 0) ? ld_kv : kv_dim;
    const float* cur = is_v ? V : K;
    float* hc = hist + (size_t) c_all * 3;
    hc[0] = cur[(size_t) (T - 3) * kv_stride + c];
    hc[1] = cur[(size_t) (T - 2) * kv_stride + c];
    hc[2] = cur[(size_t) (T - 1) * kv_stride + c];
}

__global__ void k_shortconv_kv_hist_push_f16(
    const __half* __restrict__ K_in, const __half* __restrict__ V_in,
    float* __restrict__ hist, int kv_dim, int T, int ld_kv = 0) {
    const int c_all = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_all >= 2 * kv_dim) return;
    const bool is_v = (c_all >= kv_dim);
    const int c = is_v ? (c_all - kv_dim) : c_all;
    const int kv_stride = (ld_kv > 0) ? ld_kv : kv_dim;
    const __half* cur = is_v ? V_in : K_in;
    float* hc = hist + (size_t) c_all * 3;
    hc[0] = __half2float(cur[(size_t) (T - 3) * kv_stride + c]);
    hc[1] = __half2float(cur[(size_t) (T - 2) * kv_stride + c]);
    hc[2] = __half2float(cur[(size_t) (T - 1) * kv_stride + c]);
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
    const size_t q_off = is_swa ? ((size_t) slot * kv_dim + hkv * 128) : (((size_t) hkv * cap + slot) * 128);
    int8_t* kdst = kq + q_off;
    int8_t* vdst = vq + q_off;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        kdst[i * 32 + lane] = (int8_t) __float2int_rn(kv[i] * ik);
        vdst[i * 32 + lane] = (int8_t) __float2int_rn(vv[i] * iv);
    }
    if (lane == 0) {
        const size_t s_off = is_swa ? ((size_t) slot * n_head_kv + hkv) : ((size_t) hkv * cap + slot);
        ks[s_off] = __float2half(dk);
        vs[s_off] = __float2half(dv);
    }
}

/// Copy `count` KV slots for absolute positions `j = j_start .. j_start + count - 1`
/// from ring `src[j % src_cap]` to ring `dst[j % dst_cap]` (Q8: kq/ks/vq/vs; or FP16: kf/vf).
/// Launch: dim3(count), 256 (for kv_dim <= 2048, 4 bytes/thr -> 100% coalesced 128B transactions!).
__global__ void __launch_bounds__(256, 4) k_swa_kv_ring_copy_q8(
    const int8_t* __restrict__ src_kq, const __half* __restrict__ src_ks,
    const int8_t* __restrict__ src_vq, const __half* __restrict__ src_vs,
    int8_t* __restrict__ dst_kq, __half* __restrict__ dst_ks,
    int8_t* __restrict__ dst_vq, __half* __restrict__ dst_vs,
    int j_start, int count, int src_cap, int dst_cap, int kv_dim, int n_head_kv) {
    const int idx = (int) blockIdx.x;
    if (idx >= count) return;
    const int tid = threadIdx.x;
    const int j = j_start + idx;
    const int s_slot = j % src_cap;
    const int d_slot = j % dst_cap;

    const int8_t* sk = src_kq + (size_t) s_slot * kv_dim;
    const int8_t* sv = src_vq + (size_t) s_slot * kv_dim;
    int8_t* dk = dst_kq + (size_t) d_slot * kv_dim;
    int8_t* dv = dst_vq + (size_t) d_slot * kv_dim;

    if ((kv_dim & 15) == 0) {
        const int n16 = kv_dim >> 4;
        const uint4* sk16 = reinterpret_cast<const uint4*>(sk);
        const uint4* sv16 = reinterpret_cast<const uint4*>(sv);
        uint4* dk16 = reinterpret_cast<uint4*>(dk);
        uint4* dv16 = reinterpret_cast<uint4*>(dv);
        for (int i = tid; i < n16; i += blockDim.x) {
            dk16[i] = sk16[i];
            dv16[i] = sv16[i];
        }
    } else {
        for (int i = tid; i < kv_dim; i += blockDim.x) {
            dk[i] = sk[i];
            dv[i] = sv[i];
        }
    }
    for (int h = tid; h < n_head_kv; h += blockDim.x) {
        dst_ks[(size_t) d_slot * n_head_kv + h] = src_ks[(size_t) s_slot * n_head_kv + h];
        dst_vs[(size_t) d_slot * n_head_kv + h] = src_vs[(size_t) s_slot * n_head_kv + h];
    }
}

/// Convert Token-Major circular SWA ring [ring_cap, n_head_kv, 128] -> Linear Head-Major [n_head_kv, dst_cap, 128].
__global__ void __launch_bounds__(256, 4) k_swa_kv_ring_to_hm_q8(
    const int8_t* __restrict__ src_kq, const __half* __restrict__ src_ks,
    const int8_t* __restrict__ src_vq, const __half* __restrict__ src_vs,
    int8_t* __restrict__ dst_kq, __half* __restrict__ dst_ks,
    int8_t* __restrict__ dst_vq, __half* __restrict__ dst_vs,
    int j_hist, int count, int ring_cap, int dst_cap, int n_head_kv) {
    const int idx = (int) blockIdx.x;
    if (idx >= count) return;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int s_slot = (j_hist + idx) % ring_cap;
    for (int h = warp; h < n_head_kv; h += 8) {
        const size_t s_off = ((size_t) s_slot * n_head_kv + h) * 128u + (size_t) lane * 4u;
        const size_t d_off = ((size_t) h * dst_cap + idx) * 128u + (size_t) lane * 4u;
        *reinterpret_cast<uint32_t*>(dst_kq + d_off) = *reinterpret_cast<const uint32_t*>(src_kq + s_off);
        *reinterpret_cast<uint32_t*>(dst_vq + d_off) = *reinterpret_cast<const uint32_t*>(src_vq + s_off);
        if (lane == 0) {
            dst_ks[(size_t) h * dst_cap + idx] = src_ks[(size_t) s_slot * n_head_kv + h];
            dst_vs[(size_t) h * dst_cap + idx] = src_vs[(size_t) s_slot * n_head_kv + h];
        }
    }
}

/// Convert Linear Head-Major [n_head_kv, src_cap, 128] -> Token-Major circular SWA ring [ring_cap, n_head_kv, 128].
__global__ void __launch_bounds__(256, 4) k_swa_kv_hm_to_ring_q8(
    const int8_t* __restrict__ src_kq, const __half* __restrict__ src_ks,
    const int8_t* __restrict__ src_vq, const __half* __restrict__ src_vs,
    int8_t* __restrict__ dst_kq, __half* __restrict__ dst_ks,
    int8_t* __restrict__ dst_vq, __half* __restrict__ dst_vs,
    int j_keep, int j_hist, int count, int src_cap, int ring_cap, int n_head_kv) {
    const int idx = (int) blockIdx.x;
    if (idx >= count) return;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int j = j_keep + idx;
    const int s_slot = j - j_hist;
    const int d_slot = j % ring_cap;
    for (int h = warp; h < n_head_kv; h += 8) {
        const size_t s_off = ((size_t) h * src_cap + s_slot) * 128u + (size_t) lane * 4u;
        const size_t d_off = ((size_t) d_slot * n_head_kv + h) * 128u + (size_t) lane * 4u;
        *reinterpret_cast<uint32_t*>(dst_kq + d_off) = *reinterpret_cast<const uint32_t*>(src_kq + s_off);
        *reinterpret_cast<uint32_t*>(dst_vq + d_off) = *reinterpret_cast<const uint32_t*>(src_vq + s_off);
        if (lane == 0) {
            dst_ks[(size_t) d_slot * n_head_kv + h] = src_ks[(size_t) h * src_cap + s_slot];
            dst_vs[(size_t) d_slot * n_head_kv + h] = src_vs[(size_t) h * src_cap + s_slot];
        }
    }
}

__global__ void __launch_bounds__(256, 4) k_swa_kv_ring_copy_f16(
    const __half* __restrict__ src_k, const __half* __restrict__ src_v,
    __half* __restrict__ dst_k, __half* __restrict__ dst_v,
    int j_start, int count, int src_cap, int dst_cap, int kv_dim) {
    const int idx = (int) blockIdx.x;
    if (idx >= count) return;
    const int tid = threadIdx.x;
    const int j = j_start + idx;
    const int s_slot = j % src_cap;
    const int d_slot = j % dst_cap;

    const __half* sk = src_k + (size_t) s_slot * kv_dim;
    const __half* sv = src_v + (size_t) s_slot * kv_dim;
    __half* dk = dst_k + (size_t) d_slot * kv_dim;
    __half* dv = dst_v + (size_t) d_slot * kv_dim;

    if ((kv_dim & 7) == 0) {
        const int n8 = kv_dim >> 3;
        const uint4* sk16 = reinterpret_cast<const uint4*>(sk);
        const uint4* sv16 = reinterpret_cast<const uint4*>(sv);
        uint4* dk16 = reinterpret_cast<uint4*>(dk);
        uint4* dv16 = reinterpret_cast<uint4*>(dv);
        for (int i = tid; i < n8; i += blockDim.x) {
            dk16[i] = sk16[i];
            dv16[i] = sv16[i];
        }
    } else {
        for (int i = tid; i < kv_dim; i += blockDim.x) {
            dk[i] = sk[i];
            dv[i] = sv[i];
        }
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
    const size_t q_off = is_swa ? ((size_t) slot * kv_dim + hkv * 128) : (((size_t) hkv * cap + slot) * 128);
    int8_t* kdst = kq + q_off;
    int8_t* vdst = vq + q_off;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        kdst[i * 32 + lane] = (int8_t) __float2int_rn(kv[i] * ik);
        vdst[i * 32 + lane] = (int8_t) __float2int_rn(vv[i] * iv);
    }
    if (lane == 0) {
        const size_t s_off = is_swa ? ((size_t) slot * n_head_kv + hkv) : ((size_t) hkv * cap + slot);
        ks[s_off] = __float2half(dk);
        vs[s_off] = __float2half(dv);
    }
}

/// EX-5 P5: Zero-barrier warp-vectorized fused k_shortconv_kv_pp + k_clip_norm_kv_store_q8 (head_dim == 128, T >= 32, kv_q8).
/// Grid (n_head_kv, (T + 63) / 64), block 64 (2 warps/CTA: warp 0 = K, warp 1 = V; 32 lanes x float4 = 128).
/// 0 SMEM, 0 __syncthreads(), 5-step warp shuffles, 16-reg cached conv weights, coalesced packed uint32_t STG.32!
/// `hist` is read-only here (updated by subsequent k_shortconv_kv_hist_push for zero inter-CTA races).
__global__ void __launch_bounds__(64) k_shortconv_kv_clip_norm_store_q8_pp(
    const float* __restrict__ K, const float* __restrict__ V,
    const float* __restrict__ wk, const float* __restrict__ wv,
    const float* __restrict__ dyn_k, const float* __restrict__ dyn_v,
    const float* __restrict__ hist,
    int8_t* __restrict__ kq, __half* __restrict__ ks,
    int8_t* __restrict__ vq, __half* __restrict__ vs,
    int n_head_kv, float qk_clip, float rms_eps,
    const int32_t* __restrict__ pos0, int cap, int is_swa, int T, int ld_kv = 0) {
    const int hkv = (int) blockIdx.x;
    const int t0 = (int) blockIdx.y * 64;
    const int t1 = min(T, t0 + 64);
    if (t0 >= t1) return;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const bool is_v = (warp != 0);
    const int kv_dim = n_head_kv * 128;
    const int kv_stride = (ld_kv > 0) ? ld_kv : kv_dim;
    const int c0 = hkv * 128 + lane * 4;
    const int c_all0 = is_v ? (kv_dim + c0) : c0;
    const float* cur = is_v ? V : K;
    const float* dyn = is_v ? dyn_v : dyn_k;

    float4 h0, h1, h2;
    if (t0 == 0) {
        const float4* hc4 = reinterpret_cast<const float4*>(hist + (size_t) c_all0 * 3);
        const float4 a = hc4[0], b = hc4[1], c = hc4[2];
        h0 = make_float4(a.x, a.w, b.z, c.y);
        h1 = make_float4(a.y, b.x, b.w, c.z);
        h2 = make_float4(a.z, b.y, c.x, c.w);
    } else {
        h0 = *reinterpret_cast<const float4*>(cur + (size_t) (t0 - 3) * kv_stride + c0);
        h1 = *reinterpret_cast<const float4*>(cur + (size_t) (t0 - 2) * kv_stride + c0);
        h2 = *reinterpret_cast<const float4*>(cur + (size_t) (t0 - 1) * kv_stride + c0);
    }

    const float4* wp = reinterpret_cast<const float4*>((is_v ? wv : wk) + (size_t) c0 * 4);
    const float4 w0 = __ldg(wp + 0);
    const float4 w1 = __ldg(wp + 1);
    const float4 w2 = __ldg(wp + 2);
    const float4 w3 = __ldg(wp + 3);
    const float conv_scale = rsqrtf(dyn ? 9.0f : 5.0f);
    const int p0 = pos0[0];
    const bool pow2_cap = (cap & (cap - 1)) == 0;
    const int cap_mask = cap - 1;
    int8_t* q_dst_base = is_v ? vq : kq;
    __half* s_dst_base = is_v ? vs : ks;

    float4 xv = ldcs_f4(reinterpret_cast<const float4*>(cur + (size_t) t0 * kv_stride + c0));
    float4 dt = dyn ? __ldg(reinterpret_cast<const float4*>(dyn + ((size_t) t0 * n_head_kv + hkv) * 4))
                    : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (int t = t0; t < t1; ++t) {
        float sx = h0.x * w0.x + h1.x * w0.y + h2.x * w0.z + xv.x * w0.w;
        float sy = h0.y * w1.x + h1.y * w1.y + h2.y * w1.z + xv.y * w1.w;
        float sz = h0.z * w2.x + h1.z * w2.y + h2.z * w2.z + xv.z * w2.w;
        float sw = h0.w * w3.x + h1.w * w3.y + h2.w * w3.z + xv.w * w3.w;
        if (dyn) {
            sx += xv.x * dt.x + h2.x * dt.y + h1.x * dt.z + h0.x * dt.w;
            sy += xv.y * dt.x + h2.y * dt.y + h1.y * dt.z + h0.y * dt.w;
            sz += xv.z * dt.x + h2.z * dt.y + h1.z * dt.z + h0.z * dt.w;
            sw += xv.w * dt.x + h2.w * dt.y + h1.w * dt.z + h0.w * dt.w;
        }
        float vx = (xv.x + sx) * conv_scale;
        float vy = (xv.y + sy) * conv_scale;
        float vz = (xv.z + sz) * conv_scale;
        float vw = (xv.w + sw) * conv_scale;
        h0 = h1; h1 = h2; h2 = xv;

        // Issue t+1 streaming prefetch before warp reduction shuffles to overlap HBM/L2 latency!
        float4 xv_next = xv;
        float4 dt_next = dt;
        if (t + 1 < t1) {
            xv_next = ldcs_f4(reinterpret_cast<const float4*>(cur + (size_t) (t + 1) * kv_stride + c0));
            if (dyn) dt_next = __ldg(reinterpret_cast<const float4*>(dyn + ((size_t) (t + 1) * n_head_kv + hkv) * 4));
        }

        float sq = vx * vx + vy * vy + vz * vz + vw * vw;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
        const float fac = is_v ? rsqrtf(sq / 128.0f + rms_eps) : fminf(1.0f, qk_clip / (sqrtf(sq) + 1e-6f));
        vx *= fac; vy *= fac; vz *= fac; vw *= fac;

        float amax = fmaxf(fmaxf(fabsf(vx), fabsf(vy)), fmaxf(fabsf(vz), fabsf(vw)));
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float dk = amax / 127.0f;
        const float ik = dk > 0.0f ? (127.0f / amax) : 0.0f;

        const uint32_t q0 = (uint32_t) (uint8_t) (int8_t) __float2int_rn(vx * ik);
        const uint32_t q1 = (uint32_t) (uint8_t) (int8_t) __float2int_rn(vy * ik);
        const uint32_t q2 = (uint32_t) (uint8_t) (int8_t) __float2int_rn(vz * ik);
        const uint32_t q3 = (uint32_t) (uint8_t) (int8_t) __float2int_rn(vw * ik);
        const uint32_t pack = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);

        const int pos = p0 + t;
        const int slot = is_swa ? (pow2_cap ? (pos & cap_mask) : (pos % cap)) : pos;
        const size_t q_off = is_swa ? ((size_t) slot * kv_dim + hkv * 128) : (((size_t) hkv * cap + slot) * 128);
        reinterpret_cast<uint32_t*>(q_dst_base + q_off)[lane] = pack;
        if (lane == 0) {
            const size_t s_off = is_swa ? ((size_t) slot * n_head_kv + hkv) : ((size_t) hkv * cap + slot);
            s_dst_base[s_off] = __float2half(dk);
        }
        xv = xv_next;
        dt = dt_next;
    }
}

/// Wave 10: FP16-input variant of k_shortconv_kv_clip_norm_store_q8_pp (`STACK: 0`, `sm_70`+ safe).
/// Coalesced 64-bit (`uint2` -> 2x `__half22float2`) loads; all ShortConv/clip/RMSNorm/Q8 math strictly in FP32!
template <bool HAS_DYN, int T_TILE>
__device__ __forceinline__ void k_shortconv_kv_clip_norm_store_q8_pp_f16_impl(
    const __half* __restrict__ K_in, const __half* __restrict__ V_in,
    const float* __restrict__ k_conv, const float* __restrict__ v_conv,
    const float* __restrict__ dyn_k, const float* __restrict__ dyn_v,
    const float* __restrict__ hist,
    int8_t* __restrict__ kq, __half* __restrict__ ks,
    int8_t* __restrict__ vq, __half* __restrict__ vs,
    int n_head_kv, float qk_clip, float rms_eps, const int* __restrict__ pos0_d,
    int kv_cap, int is_swa, int T, int ld_kv, int slot_bias = 0) {
    const int hkv = (int) blockIdx.x;
    const int t0 = (int) blockIdx.y * T_TILE;
    const int t1 = min(T, t0 + T_TILE);
    if (t0 >= t1) return;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const bool is_v = (warp != 0);
    const int kv_dim = n_head_kv * 128;
    const int kv_stride = (ld_kv > 0) ? ld_kv : kv_dim;
    const int c0 = hkv * 128 + lane * 4;
    const int c_all0 = is_v ? (kv_dim + c0) : c0;
    const __half* cur = is_v ? V_in : K_in;
    const float* dyn = HAS_DYN ? (is_v ? dyn_v : dyn_k) : nullptr;

    auto ld_h4_f4 = [](const __half* p, bool streaming) -> float4 {
        uint2 u;
#if defined(__CUDA_ARCH__)
        u = streaming ? __ldcs(reinterpret_cast<const uint2*>(p)) : *reinterpret_cast<const uint2*>(p);
#else
        (void) streaming;
        u = *reinterpret_cast<const uint2*>(p);
#endif
        __half2 h0, h1;
        memcpy(&h0, &u.x, 4);
        memcpy(&h1, &u.y, 4);
        const float2 f0 = __half22float2(h0);
        const float2 f1 = __half22float2(h1);
        return make_float4(f0.x, f0.y, f1.x, f1.y);
    };

    float4 h0, h1, h2;
    if (t0 == 0) {
        const float4* hc4 = reinterpret_cast<const float4*>(hist + (size_t) c_all0 * 3);
        const float4 a = hc4[0], b = hc4[1], c = hc4[2];
        h0 = make_float4(a.x, a.w, b.z, c.y);
        h1 = make_float4(a.y, b.x, b.w, c.z);
        h2 = make_float4(a.z, b.y, c.x, c.w);
    } else {
        h0 = ld_h4_f4(cur + (size_t) (t0 - 3) * kv_stride + c0, false);
        h1 = ld_h4_f4(cur + (size_t) (t0 - 2) * kv_stride + c0, false);
        h2 = ld_h4_f4(cur + (size_t) (t0 - 1) * kv_stride + c0, false);
    }

    const float4* wp = reinterpret_cast<const float4*>((is_v ? v_conv : k_conv) + (size_t) c0 * 4);
    const float4 w0 = __ldg(wp + 0);
    const float4 w1 = __ldg(wp + 1);
    const float4 w2 = __ldg(wp + 2);
    const float4 w3 = __ldg(wp + 3);
    const float conv_scale = HAS_DYN ? rsqrtf(dyn ? 9.0f : 5.0f) : rsqrtf(5.0f);
    const int p0 = pos0_d[0];
    const bool pow2_cap = (kv_cap & (kv_cap - 1)) == 0;
    const int cap_mask = kv_cap - 1;
    const int n_tok = t1 - t0;
    int pos = p0 - slot_bias + t0;
    const size_t q_step = is_swa ? (size_t) kv_dim : 128u;
    int8_t* q_dst_lane = is_v ? vq : kq;
    q_dst_lane += (is_swa ? (size_t) hkv * 128u : ((size_t) hkv * kv_cap * 128u)) + (size_t) lane * 4u;
    __half* s_dst_base = (is_v ? vs : ks) + (is_swa ? (size_t) hkv : ((size_t) hkv * kv_cap));

    const __half* cur_lane = cur + (size_t) t0 * kv_stride + c0;
    float4 xv = ld_h4_f4(cur_lane, true);
    const size_t dyn_row_stride = (size_t) n_head_kv * 4u;
    const float* dyn_ptr = HAS_DYN && dyn ? (dyn + ((size_t) t0 * n_head_kv + hkv) * 4u) : nullptr;
    float4 dt;
    if constexpr (HAS_DYN) {
        dt = dyn_ptr ? __ldg(reinterpret_cast<const float4*>(dyn_ptr))
                     : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    __half s_hdk_lane = __float2half(0.0f);

    for (int r = 0; r < n_tok; ++r) {
        float sx = h0.x * w0.x + h1.x * w0.y + h2.x * w0.z + xv.x * w0.w;
        float sy = h0.y * w1.x + h1.y * w1.y + h2.y * w1.z + xv.y * w1.w;
        float sz = h0.z * w2.x + h1.z * w2.y + h2.z * w2.z + xv.z * w2.w;
        float sw = h0.w * w3.x + h1.w * w3.y + h2.w * w3.z + xv.w * w3.w;
        if constexpr (HAS_DYN) {
            if (dyn_ptr) {
                sx += xv.x * dt.x + h2.x * dt.y + h1.x * dt.z + h0.x * dt.w;
                sy += xv.y * dt.x + h2.y * dt.y + h1.y * dt.z + h0.y * dt.w;
                sz += xv.z * dt.x + h2.z * dt.y + h1.z * dt.z + h0.z * dt.w;
                sw += xv.w * dt.x + h2.w * dt.y + h1.w * dt.z + h0.w * dt.w;
            }
        }
        float vx = (xv.x + sx) * conv_scale;
        float vy = (xv.y + sy) * conv_scale;
        float vz = (xv.z + sz) * conv_scale;
        float vw = (xv.w + sw) * conv_scale;
        h0 = h1; h1 = h2; h2 = xv;

        // Issue t+1 streaming prefetch before warp reduction shuffles to overlap HBM/L2 latency!
        cur_lane += kv_stride;
        float4 xv_next = xv;
        float4 dt_next;
        if constexpr (HAS_DYN) {
            dt_next = dt;
            if (dyn_ptr) dyn_ptr += dyn_row_stride;
        }
        if (r + 1 < n_tok) {
            xv_next = ld_h4_f4(cur_lane, true);
            if constexpr (HAS_DYN) {
                if (dyn_ptr) dt_next = __ldg(reinterpret_cast<const float4*>(dyn_ptr));
            }
        }

        float sq = vx * vx + vy * vy + vz * vz + vw * vw;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
        const float fac = is_v ? rsqrtf(sq / 128.0f + rms_eps) : fminf(1.0f, qk_clip / (sqrtf(sq) + 1e-6f));
        vx *= fac; vy *= fac; vz *= fac; vw *= fac;

        float amax = fmaxf(fmaxf(fabsf(vx), fabsf(vy)), fmaxf(fabsf(vz), fabsf(vw)));
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float dk = amax / 127.0f;
        const float ik = dk > 0.0f ? (127.0f / amax) : 0.0f;

        const uint32_t q0 = (uint32_t) (uint8_t) (int8_t) __float2int_rn(vx * ik);
        const uint32_t q1 = (uint32_t) (uint8_t) (int8_t) __float2int_rn(vy * ik);
        const uint32_t q2 = (uint32_t) (uint8_t) (int8_t) __float2int_rn(vz * ik);
        const uint32_t q3 = (uint32_t) (uint8_t) (int8_t) __float2int_rn(vw * ik);
        const uint32_t pack = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);

        const int slot = is_swa ? (pow2_cap ? (pos & cap_mask) : (pos % kv_cap)) : pos;
        *reinterpret_cast<uint32_t*>(q_dst_lane + (size_t) slot * q_step) = pack;
        if constexpr (T_TILE <= 32) {
            if (is_swa) {
                if (lane == 0) {
                    s_dst_base[(size_t) slot * n_head_kv] = __float2half(dk);
                }
            } else {
                if (lane == r) {
                    s_hdk_lane = __float2half(dk);
                }
            }
        } else {
            if (lane == 0) {
                const size_t s_off = is_swa ? ((size_t) slot * n_head_kv) : (size_t) slot;
                s_dst_base[s_off] = __float2half(dk);
            }
        }
        ++pos;
        xv = xv_next;
        if constexpr (HAS_DYN) {
            dt = dt_next;
        }
    }
    if constexpr (T_TILE <= 32) {
        if (!is_swa && lane < n_tok) {
            s_dst_base[(size_t) (pos - n_tok + lane)] = s_hdk_lane;
        }
    }
}

template <bool HAS_DYN, int T_TILE = 32>
__global__ void __launch_bounds__(64, HAS_DYN ? 1 : 18) k_shortconv_kv_clip_norm_store_q8_pp_f16(
    const __half* __restrict__ K_in, const __half* __restrict__ V_in,
    const float* __restrict__ k_conv, const float* __restrict__ v_conv,
    const float* __restrict__ dyn_k, const float* __restrict__ dyn_v,
    const float* __restrict__ hist,
    int8_t* __restrict__ kq, __half* __restrict__ ks,
    int8_t* __restrict__ vq, __half* __restrict__ vs,
    int n_head_kv, float qk_clip, float rms_eps, const int* __restrict__ pos0_d,
    int kv_cap, int is_swa, int T, int ld_kv = 0, int slot_bias = 0) {
    k_shortconv_kv_clip_norm_store_q8_pp_f16_impl<HAS_DYN, T_TILE>(
        K_in, V_in, k_conv, v_conv, dyn_k, dyn_v, hist,
        kq, ks, vq, vs, n_head_kv, qk_clip, rms_eps, pos0_d,
        kv_cap, is_swa, T, ld_kv, slot_bias);
}

__global__ void __launch_bounds__(64) k_shortconv_kv_clip_norm_store_q8_pp_f16(
    const __half* __restrict__ K_in, const __half* __restrict__ V_in,
    const float* __restrict__ k_conv, const float* __restrict__ v_conv,
    const float* __restrict__ dyn_k, const float* __restrict__ dyn_v,
    const float* __restrict__ hist,
    int8_t* __restrict__ kq, __half* __restrict__ ks,
    int8_t* __restrict__ vq, __half* __restrict__ vs,
    int n_head_kv, float qk_clip, float rms_eps, const int* __restrict__ pos0_d,
    int kv_cap, int is_swa, int T, int ld_kv = 0) {
    k_shortconv_kv_clip_norm_store_q8_pp_f16_impl<true, 64>(
        K_in, V_in, k_conv, v_conv, dyn_k, dyn_v, hist,
        kq, ks, vq, vs, n_head_kv, qk_clip, rms_eps, pos0_d, kv_cap, is_swa, T, ld_kv);
}

/// Fused k_shortconv_kv + k_clip_norm_kv_store_q8 (head_dim == 128, T < 32, kv_q8):
/// Eliminates K_conv/V_conv HBM write+read and merges 2 launches into 1! Grid (n_head_kv, 2), block 128.
__global__ void k_shortconv_kv_clip_norm_store_q8(
    const float* __restrict__ K, const float* __restrict__ V,
    const float* __restrict__ wk, const float* __restrict__ wv,
    const float* __restrict__ dyn_k, const float* __restrict__ dyn_v,
    float* __restrict__ hist,
    int8_t* __restrict__ kq, __half* __restrict__ ks,
    int8_t* __restrict__ vq, __half* __restrict__ vs,
    int n_head_kv, float qk_clip, float rms_eps,
    const int32_t* __restrict__ pos0, int cap, int is_swa, int T,
    float* __restrict__ hist_trace = nullptr, size_t hist_trace_stride = 0) {
    __shared__ float s_red[4];
    const int hkv = blockIdx.x;
    const bool is_v = (blockIdx.y != 0);
    const int d = threadIdx.x;  // 0..127
    const int warp = d >> 5, lane = d & 31;
    const int kv_dim = n_head_kv * 128;
    const int c = hkv * 128 + d;
    const int c_all = is_v ? (kv_dim + c) : c;
    const float* cur = is_v ? V : K;
    const float* dyn = is_v ? dyn_v : dyn_k;
    float* hc = hist + (size_t) c_all * 3;
    float h0 = hc[0], h1 = hc[1], h2 = hc[2];
    const float4 w4 = *reinterpret_cast<const float4*>((is_v ? wv : wk) + (size_t) c * 4);
    const float conv_scale = rsqrtf(dyn ? 9.0f : 5.0f);
    const int p0 = pos0[0];
    const bool pow2_cap = (cap & (cap - 1)) == 0;
    const int cap_mask = cap - 1;
    int8_t* q_dst_base = is_v ? vq : kq;
    __half* s_dst_base = is_v ? vs : ks;

    for (int t = 0; t < T; ++t) {
        const size_t idx = (size_t) t * kv_dim + c;
        const float x = cur[idx];
        float s = h0 * w4.x + h1 * w4.y + h2 * w4.z + x * w4.w;
        if (dyn) {
            const float* dt = dyn + ((size_t) t * n_head_kv + hkv) * 4;
            s += x * dt[0] + h2 * dt[1] + h1 * dt[2] + h0 * dt[3];
        }
        float val = (x + s) * conv_scale;
        h0 = h1; h1 = h2; h2 = x;
        if (hist_trace) {
            float* ht = hist_trace + (size_t) t * hist_trace_stride + (size_t) c_all * 3;
            ht[0] = h0; ht[1] = h1; ht[2] = h2;
        }
        float sq = val * val;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
        if (lane == 0) s_red[warp] = sq;
        __syncthreads();
        const float tot_sq = s_red[0] + s_red[1] + s_red[2] + s_red[3];
        const float fac = is_v ? rsqrtf(tot_sq / 128.0f + rms_eps) : fminf(1.0f, qk_clip / (sqrtf(tot_sq) + 1e-6f));
        val *= fac;
        float am = fabsf(val);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
        __syncthreads();
        if (lane == 0) s_red[warp] = am;
        __syncthreads();
        const float amax = fmaxf(fmaxf(s_red[0], s_red[1]), fmaxf(s_red[2], s_red[3]));
        const float dk = amax / 127.0f;
        const float ik = dk > 0.0f ? (127.0f / amax) : 0.0f;
        const int pos = p0 + t;
        const int slot = is_swa ? (pow2_cap ? (pos & cap_mask) : (pos % cap)) : pos;
        const size_t q_off = is_swa ? ((size_t) slot * kv_dim + hkv * 128) : (((size_t) hkv * cap + slot) * 128);
        q_dst_base[q_off + d] = (int8_t) __float2int_rn(val * ik);
        if (d == 0) {
            const size_t s_off = is_swa ? ((size_t) slot * n_head_kv + hkv) : ((size_t) hkv * cap + slot);
            s_dst_base[s_off] = __float2half(dk);
        }
        __syncthreads();
    }
    hc[0] = h0; hc[1] = h1; hc[2] = h2;
}

/// Fused T=1 pre-attention super-kernel (kv_q8, head_dim=128, n_rel=16, !save_conv_trace):
/// Combines k_shortconv_kv_clip_norm_store_q8 (16 CTAs) + k_clip_and_tau_qr (24 heads) + k_rel_table_split (24*d_splits CTAs)
/// into 1 single kernel launch (-2 launches & -2 serial GPU pipeline drains on all 48 layers!).
/// Grid = (2 * n_head_kv + n_head * d_splits), block = 128.
__global__ void __launch_bounds__(128, 8) k_pre_attn_t1_fused(
    const float* __restrict__ K, const float* __restrict__ V,
    const float* __restrict__ wk, const float* __restrict__ wv,
    const float* __restrict__ dyn_k, const float* __restrict__ dyn_v,
    float* __restrict__ hist,
    int8_t* __restrict__ kq, __half* __restrict__ ks,
    int8_t* __restrict__ vq, __half* __restrict__ vs,
    float* __restrict__ Q, const float* __restrict__ q_scale, float qk_clip,
    float* __restrict__ R, const float* __restrict__ r_scale, float rel_clip,
    const float* __restrict__ rel_bias, float* __restrict__ sink, float* __restrict__ rel_tab,
    int n_head, int n_head_kv, int n_ext, int max_rel, int cap, int is_swa,
    float rms_eps, int apply_tau, const int32_t* __restrict__ pos0, int floor_n, float alpha, int d_splits,
    const uint16_t* __restrict__ w_sink_bf16 = nullptr, const float* __restrict__ in_q_sink = nullptr, int n_embd_sink = 0) {
    const int bid = (int) blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int p0 = pos0[0];
    if (bid < 2 * n_head_kv) {
        __shared__ float s_red[4];
        const bool is_v = (bid >= n_head_kv);
        const int hkv = is_v ? (bid - n_head_kv) : bid;
        const int kv_dim = n_head_kv * 128;
        const int c = hkv * 128 + tid;
        const int c_all = is_v ? (kv_dim + c) : c;
        const float* cur = is_v ? V : K;
        const float* dyn = is_v ? dyn_v : dyn_k;
        float* hc = hist + (size_t) c_all * 3;
        const float h0 = hc[0], h1 = hc[1], h2 = hc[2];
        const float4 w4 = *reinterpret_cast<const float4*>((is_v ? wv : wk) + (size_t) c * 4);
        const float conv_scale = rsqrtf(dyn ? 9.0f : 5.0f);
        const float x = cur[c];
        float s = h0 * w4.x + h1 * w4.y + h2 * w4.z + x * w4.w;
        if (dyn) {
            const float* dt = dyn + (size_t) hkv * 4;
            s += x * dt[0] + h2 * dt[1] + h1 * dt[2] + h0 * dt[3];
        }
        float val = (x + s) * conv_scale;
        hc[0] = h1; hc[1] = h2; hc[2] = x;
        float sq = val * val;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
        if (lane == 0) s_red[warp] = sq;
        __syncthreads();
        const float tot_sq = s_red[0] + s_red[1] + s_red[2] + s_red[3];
        const float fac = is_v ? rsqrtf(tot_sq / 128.0f + rms_eps) : fminf(1.0f, qk_clip / (sqrtf(tot_sq) + 1e-6f));
        val *= fac;
        float am = fabsf(val);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
        __syncthreads();
        if (lane == 0) s_red[warp] = am;
        __syncthreads();
        const float amax = fmaxf(fmaxf(s_red[0], s_red[1]), fmaxf(s_red[2], s_red[3]));
        const float dk = amax / 127.0f;
        const float ik = dk > 0.0f ? (127.0f / amax) : 0.0f;
        const bool pow2_cap = (cap & (cap - 1)) == 0;
        const int slot = is_swa ? (pow2_cap ? (p0 & (cap - 1)) : (p0 % cap)) : p0;
        const size_t q_off = is_swa ? ((size_t) slot * kv_dim + hkv * 128) : (((size_t) hkv * cap + slot) * 128);
        (is_v ? vq : kq)[q_off + tid] = (int8_t) __float2int_rn(val * ik);
        if (tid == 0) {
            const size_t s_off = is_swa ? ((size_t) slot * n_head_kv + hkv) : ((size_t) hkv * cap + slot);
            (is_v ? vs : ks)[s_off] = __float2half(dk);
        }
        return;
    }

    __shared__ float s_r[16];
    __shared__ float s_sk;
    __shared__ float s_sp[8];
    const int b = bid - 2 * n_head_kv;
    const int h = b / d_splits;
    const int ds = b % d_splits;
    if (w_sink_bf16 && in_q_sink && n_embd_sink > 0) {
        // Bit-exact match to bf16_f32_mmvf_kernel<256> using 128 threads x 2 virtual halves (t, t+128)!
        const uint32_t* w2 = reinterpret_cast<const uint32_t*>(w_sink_bf16 + (size_t) h * (size_t) n_embd_sink);
        const float2* x2 = reinterpret_cast<const float2*>(in_q_sink);
        const int half_n = n_embd_sink >> 1;
        float acc_lo = 0.0f, acc_hi = 0.0f;
        for (int p = tid; p < half_n; p += 256) {
            const uint32_t wv = __ldg(w2 + p);
            const float2 xv = __ldg(x2 + p);
            acc_lo = __fmaf_rn(__uint_as_float((wv & 0xFFFFu) << 16), xv.x, acc_lo);
            acc_lo = __fmaf_rn(__uint_as_float(wv & 0xFFFF0000u), xv.y, acc_lo);
        }
        for (int p = tid + 128; p < half_n; p += 256) {
            const uint32_t wv = __ldg(w2 + p);
            const float2 xv = __ldg(x2 + p);
            acc_hi = __fmaf_rn(__uint_as_float((wv & 0xFFFFu) << 16), xv.x, acc_hi);
            acc_hi = __fmaf_rn(__uint_as_float(wv & 0xFFFF0000u), xv.y, acc_hi);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            acc_lo += __shfl_xor_sync(0xffffffffu, acc_lo, o);
            acc_hi += __shfl_xor_sync(0xffffffffu, acc_hi, o);
        }
        if (lane == 0) {
            s_sp[warp] = acc_lo;
            s_sp[4 + warp] = acc_hi;
        }
        __syncthreads();
        if (warp == 0) {
            float v = (lane < 8) ? s_sp[lane] : 0.0f;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0) {
                s_sk = v;
                if (ds == 0) sink[h] = v;
            }
        }
    }
    if (warp == 0) {
        float tau = 1.0f;
        if (apply_tau) {
            const float ratio = fmaxf(1.0f, (float) (p0 + 1) / (float) floor_n);
            tau = 1.0f + alpha * logf(ratio);
        }
        // Clip R[h][0..15] in registers and stage in SMEM for all splits (zero global R round-trip)
        float* rv = R + (size_t) h * 16;
        float rval = 0.0f;
        if (lane < 16) {
            rval = rv[lane];
            if (r_scale) rval *= r_scale[(size_t) h * 16 + lane];
        }
        float rsq = rval * rval;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) rsq += __shfl_xor_sync(0xffffffffu, rsq, o);
        const float rfac = fminf(1.0f, rel_clip / (sqrtf(rsq) + 1e-6f)) * tau;
        if (lane < 16) {
            const float rc = rval * rfac;
            s_r[lane] = rc;
            if (ds == 0) rv[lane] = rc;
        }
        if (!w_sink_bf16 && lane == 0) s_sk = sink[h];
        // ds == 0 also clips Q[h][0..127] in-place (exact match to k_clip_and_tau_qr)
        if (ds == 0) {
            float* qv = Q + (size_t) h * 128;
            float qvals[4], qsq = 0.0f;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int i = k * 32 + lane;
                float val = qv[i];
                if (q_scale) val *= q_scale[(size_t) h * 128 + i];
                qvals[k] = val;
                qsq += val * val;
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) qsq += __shfl_xor_sync(0xffffffffu, qsq, o);
            const float qfac = fminf(1.0f, qk_clip / (sqrtf(qsq) + 1e-6f)) * tau;
#pragma unroll
            for (int k = 0; k < 4; ++k) qv[k * 32 + lane] = qvals[k] * qfac;
        }
    }
    __syncthreads();
    const float4 q0 = reinterpret_cast<const float4*>(s_r)[0];
    const float4 q1 = reinterpret_cast<const float4*>(s_r)[1];
    const float4 q2 = reinterpret_cast<const float4*>(s_r)[2];
    const float4 q3 = reinterpret_cast<const float4*>(s_r)[3];
    const float sk = s_sk;
    const int max_d = min(min(n_ext, max_rel), p0 + 1);
    const int chunk = (max_d + d_splits - 1) / d_splits;
    const int d0 = ds * chunk;
    const int d1 = min(max_d, d0 + chunk);
    float* dst = rel_tab + (size_t) h * n_ext;
    for (int d = d0 + tid; d < d1; d += blockDim.x) {
        const float4* rb4 = reinterpret_cast<const float4*>(rel_bias + (size_t) d * 16);
        const float4 b0 = rb4[0], b1 = rb4[1], b2 = rb4[2], b3 = rb4[3];
        const float dot = (q0.x * b0.x + q0.y * b0.y + q0.z * b0.z + q0.w * b0.w)
                        + (q1.x * b1.x + q1.y * b1.y + q1.z * b1.z + q1.w * b1.w)
                        + (q2.x * b2.x + q2.y * b2.y + q2.z * b2.z + q2.w * b2.w)
                        + (q3.x * b3.x + q3.y * b3.y + q3.z * b3.z + q3.w * b3.w);
        dst[d] = dot - sk;
    }
}

/// Fused k_clip_and_tau for Q (D=128, float4 vectorized) and R (n_rel=16) in 1 launch.
/// Supports multi-warp CTAs (grid ((n_heads + wpb - 1)/wpb, T), block 32*wpb, e.g. 128 = 4 heads/CTA).
__global__ void k_clip_and_tau_qr(float* __restrict__ Q, const float* __restrict__ q_scale, float qk_clip,
                                  float* __restrict__ R, const float* __restrict__ r_scale, float rel_clip,
                                  int n_heads, int n_rel, int apply_tau,
                                  const int32_t* __restrict__ pos0, int floor_n, float alpha) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wpb = blockDim.x >> 5;
    const int h = (int) blockIdx.x * wpb + warp;
    if (h >= n_heads) return;
    const int t = blockIdx.y;
    float tau = 1.0f;
    if (apply_tau) {
        const int pos = pos0[0] + t;
        const float ratio = fmaxf(1.0f, (float) (pos + 1) / (float) floor_n);
        tau = 1.0f + alpha * logf(ratio);
    }
    // 1. Q (head_dim == 128 -> 32 lanes x float4 = 1x LDG.128 + 1x STG.128!)
    {
        float4* q4 = reinterpret_cast<float4*>(Q + ((size_t) t * n_heads + h) * 128);
        float4 v = q4[lane];
        if (q_scale) {
            const float4 sc = __ldg(reinterpret_cast<const float4*>(q_scale + (size_t) h * 128) + lane);
            v.x *= sc.x; v.y *= sc.y; v.z *= sc.z; v.w *= sc.w;
        }
        float sq = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
        const float fac = fminf(1.0f, qk_clip / (sqrtf(sq) + 1e-6f)) * tau;
        q4[lane] = make_float4(v.x * fac, v.y * fac, v.z * fac, v.w * fac);
    }
    // 2. R (n_rel <= 32, 1 element/lane)
    {
        float* rv = R + ((size_t) t * n_heads + h) * n_rel;
        float val = 0.0f;
        if (lane < n_rel) {
            val = rv[lane];
            if (r_scale) val *= __ldg(r_scale + (size_t) h * n_rel + lane);
        }
        float sq = val * val;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
        const float fac = fminf(1.0f, rel_clip / (sqrtf(sq) + 1e-6f)) * tau;
        if (lane < n_rel) rv[lane] = val * fac;
    }
}

/// Parallel d-split k_rel_table for small T (1..8, n_rel == 16): grid (n_head, d_splits), block 128.
__global__ void __launch_bounds__(128, 8) k_rel_table_split(const float* __restrict__ R, const float* __restrict__ rel_bias,
                                  const float* __restrict__ sink, float* __restrict__ table,
                                  int n_head, int n_rel, int n_ext, int max_rel,
                                  const int32_t* __restrict__ pos0, int T, int d_splits) {
    __shared__ float4 sr[8][4];
    __shared__ float s_sk[8];
    const int h = blockIdx.x, ds = blockIdx.y;
    const int p0 = pos0[0];
    if (threadIdx.x < T * 4) {
        const int r = threadIdx.x >> 2, c = threadIdx.x & 3;
        const float* rr = R + ((size_t) r * n_head + h) * n_rel;
        sr[r][c] = reinterpret_cast<const float4*>(rr)[c];
        if (c == 0) s_sk[r] = sink[(size_t) r * n_head + h];
    }
    __syncthreads();
    const int max_d = min(min(n_ext, max_rel), p0 + T);
    const int chunk = (max_d + d_splits - 1) / d_splits;
    const int d0 = ds * chunk;
    const int d1 = min(max_d, d0 + chunk);
    for (int d = d0 + threadIdx.x; d < d1; d += blockDim.x) {
        const float4* rb4 = reinterpret_cast<const float4*>(rel_bias + (size_t) d * n_rel);
        const float4 b0 = rb4[0], b1 = rb4[1], b2 = rb4[2], b3 = rb4[3];
#pragma unroll 1
        for (int r = 0; r < T; ++r) {
            if (d <= p0 + r) {
                const float4 q0 = sr[r][0], q1 = sr[r][1], q2 = sr[r][2], q3 = sr[r][3];
                const float dot = (q0.x * b0.x + q0.y * b0.y + q0.z * b0.z + q0.w * b0.w)
                                + (q1.x * b1.x + q1.y * b1.y + q1.z * b1.z + q1.w * b1.w)
                                + (q2.x * b2.x + q2.y * b2.y + q2.z * b2.z + q2.w * b2.w)
                                + (q3.x * b3.x + q3.y * b3.y + q3.z * b3.z + q3.w * b3.w);
                table[(((size_t) r * n_head + h) * n_ext) + d] = dot - s_sk[r];
            }
        }
    }
}

/// Compute relative bias distance table minus sink: table[t][h][d] = dot(R[t,h], rel_bias[d]) - sink[t,h]
/// R is [T][n_head][n_rel=16], rel_bias is [n_ext=1024][16] (F32 contiguous along n_rel=16), sink is [T][n_head].
/// Reads *pos0 on device so grid is CUDA-Graph invariant. grid (n_head, (T + 7) / 8), block 256.
__global__ void k_rel_table(const float* __restrict__ R, const float* __restrict__ rel_bias,
                            const float* __restrict__ sink, float* __restrict__ table,
                            int n_head, int n_rel, int n_ext, int max_rel, const int32_t* __restrict__ pos0, int T) {
    __shared__ float4 sr[8][4];
    __shared__ float s_sk[8];
    const int h = blockIdx.x, t0 = blockIdx.y * 8;
    const int p0 = pos0[0];
    const int n_rows = min(8, T - t0);
    if (n_rows <= 0) return;
    if (threadIdx.x < n_rows * 4) {
        const int r = threadIdx.x >> 2, c = threadIdx.x & 3;
        const float* rr = R + ((size_t) (t0 + r) * n_head + h) * n_rel;
        sr[r][c] = reinterpret_cast<const float4*>(rr)[c];
        if (c == 0) s_sk[r] = sink[(size_t) (t0 + r) * n_head + h];
    }
    __syncthreads();
    const int max_d = min(min(n_ext, max_rel), p0 + t0 + n_rows);
    for (int d = threadIdx.x; d < max_d; d += blockDim.x) {
        const float4* rb4 = reinterpret_cast<const float4*>(rel_bias + (size_t) d * n_rel);
        const float4 b0 = rb4[0], b1 = rb4[1], b2 = rb4[2], b3 = rb4[3];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < n_rows && d <= p0 + t0 + r) {
                const float4 q0 = sr[r][0], q1 = sr[r][1], q2 = sr[r][2], q3 = sr[r][3];
                const float dot = (q0.x * b0.x + q0.y * b0.y + q0.z * b0.z + q0.w * b0.w)
                                + (q1.x * b1.x + q1.y * b1.y + q1.z * b1.z + q1.w * b1.w)
                                + (q2.x * b2.x + q2.y * b2.y + q2.z * b2.z + q2.w * b2.w)
                                + (q3.x * b3.x + q3.y * b3.y + q3.z * b3.z + q3.w * b3.w);
                table[(((size_t) (t0 + r) * n_head + h) * n_ext) + d] = dot - s_sk[r];
            }
        }
    }
}

/// 32-token-tile FP32 k_rel_table for prefill (T >= 32, n_rel == 16): 4x fewer rel_bias reads & CTAs, bit-identical FP32.
/// grid (n_head, (T + 31) / 32), block 256.
__global__ void k_rel_table_t32(const float* __restrict__ R, const float* __restrict__ rel_bias,
                                const float* __restrict__ sink, float* __restrict__ table,
                                int n_head, int n_rel, int n_ext, int max_rel, const int32_t* __restrict__ pos0, int T) {
    __shared__ float4 sr[32][4];
    __shared__ float s_sk[32];
    const int h = blockIdx.x, t0 = blockIdx.y * 32;
    const int p0 = pos0[0];
    const int n_rows = min(32, T - t0);
    if (n_rows <= 0) return;
    if (threadIdx.x < n_rows * 4) {
        const int r = threadIdx.x >> 2, c = threadIdx.x & 3;
        const float* rr = R + ((size_t) (t0 + r) * n_head + h) * n_rel;
        sr[r][c] = reinterpret_cast<const float4*>(rr)[c];
        if (c == 0) s_sk[r] = sink[(size_t) (t0 + r) * n_head + h];
    }
    __syncthreads();
    const int max_d = min(min(n_ext, max_rel), p0 + t0 + n_rows);
    const int full_d = min(max_d, p0 + t0 + 1);
    for (int d = threadIdx.x; d < max_d; d += blockDim.x) {
        const float4* rb4 = reinterpret_cast<const float4*>(rel_bias + (size_t) d * n_rel);
        const float4 b0 = rb4[0], b1 = rb4[1], b2 = rb4[2], b3 = rb4[3];
        if (n_rows == 32 && d < full_d) {
#pragma unroll 8
            for (int r = 0; r < 32; ++r) {
                const float4 q0 = sr[r][0], q1 = sr[r][1], q2 = sr[r][2], q3 = sr[r][3];
                const float dot = (q0.x * b0.x + q0.y * b0.y + q0.z * b0.z + q0.w * b0.w)
                                + (q1.x * b1.x + q1.y * b1.y + q1.z * b1.z + q1.w * b1.w)
                                + (q2.x * b2.x + q2.y * b2.y + q2.z * b2.z + q2.w * b2.w)
                                + (q3.x * b3.x + q3.y * b3.y + q3.z * b3.z + q3.w * b3.w);
                table[(((size_t) (t0 + r) * n_head + h) * n_ext) + d] = dot - s_sk[r];
            }
        } else {
            for (int r = 0; r < n_rows; ++r) {
                if (d <= p0 + t0 + r) {
                    const float4 q0 = sr[r][0], q1 = sr[r][1], q2 = sr[r][2], q3 = sr[r][3];
                    const float dot = (q0.x * b0.x + q0.y * b0.y + q0.z * b0.z + q0.w * b0.w)
                                    + (q1.x * b1.x + q1.y * b1.y + q1.z * b1.z + q1.w * b1.w)
                                    + (q2.x * b2.x + q2.y * b2.y + q2.z * b2.z + q2.w * b2.w)
                                    + (q3.x * b3.x + q3.y * b3.y + q3.z * b3.z + q3.w * b3.w);
                    table[(((size_t) (t0 + r) * n_head + h) * n_ext) + d] = dot - s_sk[r];
                }
            }
        }
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
        const size_t idx = is_swa ? ((size_t) slot * n_head_kv + hkv) : ((size_t) hkv * cap + slot);
        const size_t q_off = is_swa ? ((size_t) slot * kv_dim + hkv * 128) : (((size_t) hkv * cap + slot) * 128);
        const float d_k = __half2float(ks[idx]);
        const float d_v = __half2float(vs[idx]);
        const int8_t* kj = kq + q_off;
        const int8_t* vj = vq + q_off;
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

inline constexpr int kAttnSplitMax = 240;

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

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530
/// Vectorized 4x int8 -> 2x half2 in-register decode via byte_perm + __hsub2 (bit-exact for [-128, 127]).
__device__ __forceinline__ void i8x4_to_h2x2(uint32_t packed_i8x4, __half2& out0, __half2& out1) {
    const uint32_t u = packed_i8x4 ^ 0x80808080u;
    const uint32_t lo = __byte_perm(u, 0x6400u, 0x5150u);
    const uint32_t hi = __byte_perm(u, 0x6400u, 0x5352u);
    const __half2 bias = __half2half2(__float2half(1152.0f));
    out0 = __hsub2(*reinterpret_cast<const __half2*>(&lo), bias);
    out1 = __hsub2(*reinterpret_cast<const __half2*>(&hi), bias);
}
#endif

/// FlashAttention-style FP16 WMMA prefill attention for GQA Q8 KV (head_dim=128, sm_70+).
/// One CTA (128 threads = 4 warps) computes 16 queries x 128 dims using 64-key tiles:
/// cuts KV HBM traffic 16x and runs Q*K^T & P*V on Tensor Cores while keeping int8 KV exact in FP16.
__global__ void __launch_bounds__(128, 3) k_attn_q8_gqa3_pp_wmma(
        const float* __restrict__ Q,
        const int8_t* __restrict__ kq, const __half* __restrict__ ks,
        const int8_t* __restrict__ vq, const __half* __restrict__ vs,
        const float* __restrict__ rel_tab, const float* __restrict__ sink,
        const float* __restrict__ gate, float* __restrict__ out,
        int n_head, int n_head_kv, int n_ext, int cap, int is_swa, int n_swa,
        const int32_t* __restrict__ pos0, float kq_scale, int T) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    extern __shared__ char smem_raw[];
    // Align each region to 16 bytes:
    // sQ: 16*136 __half (4352 B) | sKV: 64*136 __half (17408 B, aliases s_tile 16*136 float = 8704 B)
    // sS: 16*64 float (4096 B)   | sP: 16*72 __half (2304 B) | scalars (< 1 KB)
    __half* sQ = reinterpret_cast<__half*>(smem_raw);
    __half* sKV = reinterpret_cast<__half*>(smem_raw + 4352);
    float* s_tile = reinterpret_cast<float*>(sKV);
    float* sS = reinterpret_cast<float*>(smem_raw + 4352 + 17408);
    __half* sP = reinterpret_cast<__half*>(smem_raw + 4352 + 17408 + 4096);
    float* s_qscale = reinterpret_cast<float*>(smem_raw + 4352 + 17408 + 4096 + 2304);
    float* s_dk = s_qscale + 16;
    float* s_dv = s_dk + 64;
    float* s_m = s_dv + 64;
    float* s_l = s_m + 16;
    float* s_alpha = s_l + 16;
    float* s_pvmax = s_alpha + 16;

    const int h = blockIdx.x;
    const int gqa = n_head / n_head_kv;
    const int hkv = h / gqa;
    const int t0 = blockIdx.y * 16;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int kv_dim = n_head_kv * 128;

    // Cooperatively load Q[t0..t0+15][h][0..127] (8 threads/row x 16 floats) and normalize per row for FP16 safety.
    {
        const int qr = tid >> 3;
        const int qc0 = (tid & 7) * 16;
        const int t = t0 + qr;
        float amax = 0.0f;
        const float* qp = (t < T) ? (Q + ((size_t) t * n_head + h) * 128 + qc0) : nullptr;
        if (qp) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = *reinterpret_cast<const float4*>(qp + i * 4);
                amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
            }
        }
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
        const float qscale = (amax > 0.0f) ? amax : 1.0f;
        const float inv_qs = __fdividef(1.0f, qscale);
        if ((tid & 7) == 0) {
            s_qscale[qr] = qscale;
            s_m[qr] = 0.0f;
            s_l[qr] = 1.0f;
        }
        __half2* dst2 = reinterpret_cast<__half2*>(sQ + qr * 136 + qc0);
        if (qp) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = *reinterpret_cast<const float4*>(qp + i * 4);
                dst2[i * 2 + 0] = __halves2half2(__float2half(v.x * inv_qs), __float2half(v.y * inv_qs));
                dst2[i * 2 + 1] = __halves2half2(__float2half(v.z * inv_qs), __float2half(v.w * inv_qs));
            }
        } else {
            const __half2 z2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));
#pragma unroll
            for (int i = 0; i < 8; ++i) dst2[i] = z2;
        }
    }
    __syncthreads();

    const int p0 = pos0[0];
    const int t_last = min(t0 + 15, T - 1);
    const int j_start = is_swa ? max(0, (p0 + t0) - n_swa + 1) : 0;
    const int j_end = p0 + t_last + 1;
    const int bulk_end = is_swa ? j_start : max(j_start, min(j_end, p0 + t0 - n_ext + 1));
    const bool pow2_cap = (cap & (cap - 1)) == 0;
    const int cap_mask = cap - 1;

    float acc[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) acc[r] = 0.0f;

    const int kv_r = tid >> 1;
    const int kv_c0 = (tid & 1) * 64;

    auto step_tile = [&](int jb, bool is_bulk) {
        // 1. Load K[jb..jb+63] into sKV[64][136] + s_dk, s_dv
        {
            const int j = jb + kv_r;
            const bool valid = is_bulk ? true : (j < j_end);
            const int slot = is_bulk ? j : (valid ? (is_swa ? (pow2_cap ? (j & cap_mask) : (j % cap)) : j) : 0);
            const size_t idx = is_swa ? ((size_t) slot * n_head_kv + hkv) : ((size_t) hkv * cap + slot);
            if ((tid & 1) == 0) {
                s_dk[kv_r] = valid ? __half2float(__ldg(&ks[idx])) : 0.0f;
                s_dv[kv_r] = valid ? __half2float(__ldg(&vs[idx])) : 0.0f;
            }
            __half* dst = sKV + kv_r * 136 + kv_c0;
            if (valid) {
                const int8_t* kp = kq + (is_swa ? ((size_t) slot * kv_dim + hkv * 128) : (((size_t) hkv * cap + slot) * 128)) + kv_c0;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(kp + i * 16);
                    __half2 h2[8];
                    i8x4_to_h2x2(raw.x, h2[0], h2[1]);
                    i8x4_to_h2x2(raw.y, h2[2], h2[3]);
                    i8x4_to_h2x2(raw.z, h2[4], h2[5]);
                    i8x4_to_h2x2(raw.w, h2[6], h2[7]);
                    *reinterpret_cast<uint4*>(dst + i * 16) = *reinterpret_cast<const uint4*>(h2);
                    *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                }
            } else {
                const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int i = 0; i < 8; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
            }
        }
        __syncthreads();

        // 2. S = Q * K^T ([16, 128] x [64, 128]^T -> [16, 64], 1 16x16 tile per warp)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
            wmma::fill_fragment(fc, 0.0f);
#pragma unroll
            for (int k = 0; k < 128; k += 16) {
                wmma::load_matrix_sync(fa, sQ + k, 136);
                wmma::load_matrix_sync(fb, sKV + (warp * 16) * 136 + k, 136);
                wmma::mma_sync(fc, fa, fb, fc);
            }
            wmma::store_matrix_sync(sS + warp * 16, fc, 64, wmma::mem_row_major);
        }
        __syncthreads();

        // 3. Load V[jb..jb+63] into sKV[64][136] AND run online softmax (4 rows per warp) -> sP[16][72]
        {
            const int j = jb + kv_r;
            const bool valid = is_bulk ? true : (j < j_end);
            __half* dst = sKV + kv_r * 136 + kv_c0;
            if (valid) {
                const int slot = is_bulk ? j : (is_swa ? (pow2_cap ? (j & cap_mask) : (j % cap)) : j);
                const int8_t* vp = vq + (is_swa ? ((size_t) slot * kv_dim + hkv * 128) : (((size_t) hkv * cap + slot) * 128)) + kv_c0;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(vp + i * 16);
                    __half2 h2[8];
                    i8x4_to_h2x2(raw.x, h2[0], h2[1]);
                    i8x4_to_h2x2(raw.y, h2[2], h2[3]);
                    i8x4_to_h2x2(raw.z, h2[4], h2[5]);
                    i8x4_to_h2x2(raw.w, h2[6], h2[7]);
                    *reinterpret_cast<uint4*>(dst + i * 16) = *reinterpret_cast<const uint4*>(h2);
                    *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                }
            } else {
                const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int i = 0; i < 8; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
            }

            const int col0 = lane;
            const int col1 = lane + 32;
            const int j_0 = jb + col0;
            const int j_1 = jb + col1;
            const float dk0 = s_dk[col0], dk1 = s_dk[col1];
            const float dv0 = s_dv[col0], dv1 = s_dv[col1];

#pragma unroll
            for (int rr = 0; rr < 4; ++rr) {
                const int r = warp * 4 + rr;
                const int t = t0 + r;
                const bool t_ok = t < T;
                const float nsink = t_ok ? -sink[(size_t) t * n_head + h] : 0.0f;
                const float qks = s_qscale[r] * kq_scale;

                float sc0, sc1;
                bool v0, v1;
                if (is_bulk) {
                    v0 = t_ok;
                    v1 = t_ok;
                    sc0 = t_ok ? (sS[r * 64 + col0] * qks * dk0 + nsink) : -1e30f;
                    sc1 = t_ok ? (sS[r * 64 + col1] * qks * dk1 + nsink) : -1e30f;
                } else {
                    const int pos_q = p0 + t;
                    const float* rt = t_ok ? (rel_tab + ((size_t) t * n_head + h) * n_ext) : nullptr;
                    const int d0 = pos_q - j_0;
                    const int d1 = pos_q - j_1;
                    v0 = t_ok && (j_0 <= pos_q) && (!is_swa || d0 < n_swa);
                    v1 = t_ok && (j_1 <= pos_q) && (!is_swa || d1 < n_swa);
                    sc0 = v0 ? (sS[r * 64 + col0] * qks * dk0 + (d0 < n_ext ? rt[d0] : nsink)) : -1e30f;
                    sc1 = v1 ? (sS[r * 64 + col1] * qks * dk1 + (d1 < n_ext ? rt[d1] : nsink)) : -1e30f;
                }

                float rmax = fmaxf(sc0, sc1);
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffffu, rmax, o));

                const float m_old = s_m[r];
                const float l_old = s_l[r];
                const float m_new = fmaxf(m_old, rmax);
                const float alpha = l_old > 0.0f ? __expf(m_old - m_new) : 0.0f;
                const float p_0 = v0 ? __expf(sc0 - m_new) : 0.0f;
                const float p_1 = v1 ? __expf(sc1 - m_new) : 0.0f;

                float psum = p_0 + p_1;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) psum += __shfl_xor_sync(0xffffffffu, psum, o);

                const float pv_0 = p_0 * dv0;
                const float pv_1 = p_1 * dv1;
                float pvmax = fmaxf(pv_0, pv_1);
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) pvmax = fmaxf(pvmax, __shfl_xor_sync(0xffffffffu, pvmax, o));
                const float inv_pv = pvmax > 0.0f ? __fdividef(1.0f, pvmax) : 0.0f;

                sP[r * 72 + col0] = __float2half(pv_0 * inv_pv);
                sP[r * 72 + col1] = __float2half(pv_1 * inv_pv);
                if (lane == 0) {
                    s_m[r] = m_new;
                    s_l[r] = l_old * alpha + psum;
                    s_alpha[r] = alpha;
                    s_pvmax[r] = pvmax;
                }
            }
        }
        __syncthreads();

        // 4. O_tile = P * V ([16, 64] x [64, 128] -> [16, 128], 2 16x16 tiles = 32 cols per warp)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb0, fb1;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc0, fc1;
            wmma::fill_fragment(fc0, 0.0f);
            wmma::fill_fragment(fc1, 0.0f);
            const int cbase = warp * 32;
#pragma unroll
            for (int k = 0; k < 64; k += 16) {
                wmma::load_matrix_sync(fa, sP + k, 72);
                wmma::load_matrix_sync(fb0, sKV + k * 136 + cbase, 136);
                wmma::mma_sync(fc0, fa, fb0, fc0);
                wmma::load_matrix_sync(fb1, sKV + k * 136 + cbase + 16, 136);
                wmma::mma_sync(fc1, fa, fb1, fc1);
            }
            __syncthreads();
            wmma::store_matrix_sync(s_tile + cbase, fc0, 136, wmma::mem_row_major);
            wmma::store_matrix_sync(s_tile + cbase + 16, fc1, 136, wmma::mem_row_major);
        }
        __syncthreads();

        // 5. Rescale + accumulate into thread's own column across all 16 rows
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            acc[r] = acc[r] * s_alpha[r] + s_pvmax[r] * s_tile[r * 136 + tid];
        }
        __syncthreads();
    };

    int jb = j_start;
    for (; jb + 64 <= bulk_end; jb += 64) step_tile(jb, true);
    for (; jb < j_end; jb += 64) step_tile(jb, false);

#pragma unroll
    for (int r = 0; r < 16; ++r) {
        const int t = t0 + r;
        if (t < T) {
            const size_t oi = ((size_t) t * n_head + h) * 128 + tid;
            float val = __fdividef(acc[r], s_l[r]);
            if (gate) val *= sigmf(gate[oi]);
            out[oi] = val;
        }
    }
#endif
}

/// 32-query-tile (256 threads = 8 warps) FP16 WMMA prefill attention for GQA Q8 KV (head_dim=128, T >= 128):
/// Cuts KV HBM2 reads & syncs 2x relative to the 16-row tile with balanced 8-warp WMMA utilization.
__global__ void __launch_bounds__(256, 2) k_attn_q8_gqa3_pp_wmma_t32(
        const float* __restrict__ Q,
        const int8_t* __restrict__ kq, const __half* __restrict__ ks,
        const int8_t* __restrict__ vq, const __half* __restrict__ vs,
        const float* __restrict__ rel_tab, const float* __restrict__ sink,
        const float* __restrict__ gate, float* __restrict__ out,
        int n_head, int n_head_kv, int n_ext, int cap, int is_swa, int n_swa,
        const int32_t* __restrict__ pos0, float kq_scale, int T) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    extern __shared__ __align__(16) char smem_raw[];
    // 48,768 B total (fits 2 blk/SM in 96/100 KB): sQ (8704) | sKV (17408, separate!) | sP (4608) |
    // sS (8192) overlaid by s_tile (32*132 = 16896) | scalars (1152). Cuts barriers from 6 -> 4 per tile.
    __half* sQ = reinterpret_cast<__half*>(smem_raw);
    __half* sKV = sQ + 32 * 136;
    __half* sP = sKV + 64 * 136;
    float* sS = reinterpret_cast<float*>(sP + 32 * 72);
    float* s_tile = sS;
    float* s_qscale = s_tile + 32 * 132;
    float* s_dk = s_qscale + 32;
    float* s_dv = s_dk + 64;
    float* s_m = s_dv + 64;
    float* s_l = s_m + 32;
    float* s_alpha = s_l + 32;
    float* s_pvmax = s_alpha + 32;

    const int h = blockIdx.x;
    const int hkv = h / (n_head / n_head_kv);
    const int t0 = blockIdx.y * 32;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int kv_dim = n_head_kv * 128;
    constexpr float kLog2e = 1.4426950408889634f;

    {
        const int qr = tid >> 3;
        const int qc0 = (tid & 7) * 16;
        const int t = t0 + qr;
        float amax = 0.0f;
        const float* qp = (t < T) ? (Q + ((size_t) t * n_head + h) * 128 + qc0) : nullptr;
        if (qp) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = *reinterpret_cast<const float4*>(qp + i * 4);
                amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
            }
        }
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
        const float qscale = (amax > 0.0f) ? amax : 1.0f;
        const float inv_qs = __fdividef(1.0f, qscale);
        if ((tid & 7) == 0) {
            s_qscale[qr] = qscale * kq_scale * kLog2e;
            s_m[qr] = 0.0f;
            s_l[qr] = 1.0f;
        }
        __half2* dst2 = reinterpret_cast<__half2*>(sQ + qr * 136 + qc0);
        if (qp) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = *reinterpret_cast<const float4*>(qp + i * 4);
                dst2[i * 2 + 0] = __halves2half2(__float2half(v.x * inv_qs), __float2half(v.y * inv_qs));
                dst2[i * 2 + 1] = __halves2half2(__float2half(v.z * inv_qs), __float2half(v.w * inv_qs));
            }
        } else {
            const __half2 z2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));
#pragma unroll
            for (int i = 0; i < 8; ++i) dst2[i] = z2;
        }
    }
    __syncthreads();

    const int p0 = pos0[0];
    const int t_last = min(t0 + 31, T - 1);
    const int j_start = is_swa ? max(0, (p0 + t0) - n_swa + 1) : 0;
    const int j_end = p0 + t_last + 1;
    const int bulk_end = is_swa ? j_start : max(j_start, min(j_end, p0 + t0 - n_ext + 1));
    const bool pow2_cap = (cap & (cap - 1)) == 0;
    const int cap_mask = cap - 1;

    float acc[16] = {};
    const int kv_r = tid >> 2;
    const int kv_c0 = (tid & 3) * 32;
    const int my_col = tid & 127;
    const int my_half = tid >> 7;

    auto step_tile = [&](int jb, bool is_bulk) {
        {
            const int j = jb + kv_r;
            const bool valid = is_bulk ? true : (j < j_end);
            const int slot = is_bulk ? j : (valid ? (is_swa ? (pow2_cap ? (j & cap_mask) : (j % cap)) : j) : 0);
            if ((tid & 3) == 0) {
                const size_t idx = (size_t) slot * n_head_kv + hkv;
                s_dk[kv_r] = valid ? __half2float(__ldg(&ks[idx])) : 0.0f;
                s_dv[kv_r] = valid ? __half2float(__ldg(&vs[idx])) : 0.0f;
            }
            __half* dst = sKV + kv_r * 136 + kv_c0;
            if (valid) {
                const int8_t* kp = kq + (size_t) slot * kv_dim + hkv * 128 + kv_c0;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(kp + i * 16);
                    __half2 h2[8];
                    i8x4_to_h2x2(raw.x, h2[0], h2[1]);
                    i8x4_to_h2x2(raw.y, h2[2], h2[3]);
                    i8x4_to_h2x2(raw.z, h2[4], h2[5]);
                    i8x4_to_h2x2(raw.w, h2[6], h2[7]);
                    *reinterpret_cast<uint4*>(dst + i * 16) = *reinterpret_cast<const uint4*>(h2);
                    *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                }
            } else {
                const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int i = 0; i < 4; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
            }
        }
        __syncthreads();

        {
            const int rt = warp >> 2;
            const int ct = warp & 3;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
            wmma::fill_fragment(fc, 0.0f);
#pragma unroll
            for (int k = 0; k < 128; k += 16) {
                wmma::load_matrix_sync(fa, sQ + (rt * 16) * 136 + k, 136);
                wmma::load_matrix_sync(fb, sKV + (ct * 16) * 136 + k, 136);
                wmma::mma_sync(fc, fa, fb, fc);
            }
            wmma::store_matrix_sync(sS + (rt * 16) * 64 + ct * 16, fc, 64, wmma::mem_row_major);
        }
        __syncthreads();

        {
            const int j = jb + kv_r;
            const bool valid = is_bulk ? true : (j < j_end);
            __half* dst = sKV + kv_r * 136 + kv_c0;
            if (valid) {
                const int slot = is_bulk ? j : (is_swa ? (pow2_cap ? (j & cap_mask) : (j % cap)) : j);
                const int8_t* vp = vq + (is_swa ? ((size_t) slot * kv_dim + hkv * 128) : (((size_t) hkv * cap + slot) * 128)) + kv_c0;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(vp + i * 16);
                    __half2 h2[8];
                    i8x4_to_h2x2(raw.x, h2[0], h2[1]);
                    i8x4_to_h2x2(raw.y, h2[2], h2[3]);
                    i8x4_to_h2x2(raw.z, h2[4], h2[5]);
                    i8x4_to_h2x2(raw.w, h2[6], h2[7]);
                    *reinterpret_cast<uint4*>(dst + i * 16) = *reinterpret_cast<const uint4*>(h2);
                    *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                }
            } else {
                const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int i = 0; i < 4; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
            }

            const int col0 = lane;
            const int col1 = lane + 32;
            const int j_0 = jb + col0;
            const int j_1 = jb + col1;
            const float dk0 = s_dk[col0], dk1 = s_dk[col1];
            const float dv0 = s_dv[col0], dv1 = s_dv[col1];

#pragma unroll
            for (int rr = 0; rr < 4; ++rr) {
                const int r = warp * 4 + rr;
                const int t = t0 + r;
                const bool t_ok = t < T;
                const float nsink = t_ok ? (-sink[(size_t) t * n_head + h] * kLog2e) : 0.0f;
                const float qks2 = s_qscale[r];

                float sc0, sc1;
                bool v0, v1;
                if (is_bulk) {
                    v0 = t_ok;
                    v1 = t_ok;
                    sc0 = t_ok ? (sS[r * 64 + col0] * qks2 * dk0 + nsink) : -1e30f;
                    sc1 = t_ok ? (sS[r * 64 + col1] * qks2 * dk1 + nsink) : -1e30f;
                } else {
                    const int pos_q = p0 + t;
                    const float* rt = t_ok ? (rel_tab + ((size_t) t * n_head + h) * n_ext) : nullptr;
                    const int d0 = pos_q - j_0;
                    const int d1 = pos_q - j_1;
                    v0 = t_ok && (j_0 <= pos_q) && (!is_swa || d0 < n_swa);
                    v1 = t_ok && (j_1 <= pos_q) && (!is_swa || d1 < n_swa);
                    sc0 = v0 ? (sS[r * 64 + col0] * qks2 * dk0 + (d0 < n_ext ? (rt[d0] * kLog2e) : nsink)) : -1e30f;
                    sc1 = v1 ? (sS[r * 64 + col1] * qks2 * dk1 + (d1 < n_ext ? (rt[d1] * kLog2e) : nsink)) : -1e30f;
                }

                float rmax = fmaxf(sc0, sc1);
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffffu, rmax, o));
                const float m_old = s_m[r], l_old = s_l[r];
                const float m_new = fmaxf(m_old, rmax);
                const float alpha = l_old > 0.0f ? exp2f(m_old - m_new) : 0.0f;
                const float p_0 = v0 ? exp2f(sc0 - m_new) : 0.0f;
                const float p_1 = v1 ? exp2f(sc1 - m_new) : 0.0f;
                float psum = p_0 + p_1;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) psum += __shfl_xor_sync(0xffffffffu, psum, o);

                const float pv_0 = p_0 * dv0, pv_1 = p_1 * dv1;
                float pvmax = fmaxf(pv_0, pv_1);
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) pvmax = fmaxf(pvmax, __shfl_xor_sync(0xffffffffu, pvmax, o));
                const float inv_pv = pvmax > 0.0f ? __fdividef(1.0f, pvmax) : 0.0f;

                sP[r * 72 + col0] = __float2half(pv_0 * inv_pv);
                sP[r * 72 + col1] = __float2half(pv_1 * inv_pv);
                if (lane == 0) {
                    s_m[r] = m_new;
                    s_l[r] = l_old * alpha + psum;
                    s_alpha[r] = alpha;
                    s_pvmax[r] = pvmax;
                }
            }
        }
        __syncthreads();

        {
            const int cbase = warp * 16;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc0, fc1;
            wmma::fill_fragment(fc0, 0.0f);
            wmma::fill_fragment(fc1, 0.0f);
#pragma unroll
            for (int k = 0; k < 64; k += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa0, fa1;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
                wmma::load_matrix_sync(fa0, sP + k, 72);
                wmma::load_matrix_sync(fa1, sP + 16 * 72 + k, 72);
                wmma::load_matrix_sync(fb, sKV + k * 136 + cbase, 136);
                wmma::mma_sync(fc0, fa0, fb, fc0);
                wmma::mma_sync(fc1, fa1, fb, fc1);
            }
            wmma::store_matrix_sync(s_tile + cbase, fc0, 132, wmma::mem_row_major);
            wmma::store_matrix_sync(s_tile + 16 * 132 + cbase, fc1, 132, wmma::mem_row_major);
        }
        __syncthreads();

        const int row0 = my_half * 16;
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            acc[r] = acc[r] * s_alpha[row0 + r] + s_pvmax[row0 + r] * s_tile[(row0 + r) * 132 + my_col];
        }
    };

    int jb = j_start;
    for (; jb + 64 <= bulk_end; jb += 64) step_tile(jb, true);
    for (; jb < j_end; jb += 64) step_tile(jb, false);

    const int row0 = my_half * 16;
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        const int t = t0 + row0 + r;
        if (t < T) {
            const size_t oi = ((size_t) t * n_head + h) * 128 + my_col;
            float val = __fdividef(acc[r], s_l[row0 + r]);
            if (gate) val *= sigmf(gate[oi]);
            out[oi] = val;
        }
    }
#endif
}

/// GQA-3 Shared-sKV 48-row (16 tokens x 3 GQA heads, 256 threads = 8 warps) FP16 WMMA prefill attention (head_dim=128, sm_70+).
/// Loads and i8x4_to_h2x2-decodes K/V ONCE per BK=64 tile and reuses across all 3 GQA heads in bank-conflict-free 44,224 B SMEM (2 blk/SM).
__global__ void __launch_bounds__(256, 2) k_attn_q8_gqa3_pp_wmma_gqa3(
        const float* __restrict__ Q,
        const int8_t* __restrict__ kq, const __half* __restrict__ ks,
        const int8_t* __restrict__ vq, const __half* __restrict__ vs,
        const float* __restrict__ rel_tab, const float* __restrict__ sink,
        const float* __restrict__ gate, float* __restrict__ out,
        int n_head, int n_head_kv, int n_ext, int cap, int is_swa, int n_swa,
        const int32_t* __restrict__ pos0, float kq_scale, int T) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    extern __shared__ __align__(16) char smem_raw[];
    // 44,224 B <= 49,152 B (2 blk/SM on 96/100 KB SMEM):
    // sQ (48*136*2 = 13056) | sKV (64*136*2 = 17408) | sS (48*64*4 = 12288, reused as sP 48*72*2 = 6912) | scalars (1472).
    // s_tile (48*132*4 = 25344) overlays dead sKV+sS (29696) for single-shot 3-head P*V store.
    __half* sQ = reinterpret_cast<__half*>(smem_raw);
    __half* sKV = sQ + 48 * 136;
    float* sS = reinterpret_cast<float*>(sKV + 64 * 136);
    __half* sP = reinterpret_cast<__half*>(sS);
    float* s_tile = reinterpret_cast<float*>(sKV);
    float* s_qscale = sS + 48 * 64;
    float* s_dk = s_qscale + 48;
    float* s_dv = s_dk + 64;
    float* s_m = s_dv + 64;
    float* s_l = s_m + 48;
    float* s_alpha = s_l + 48;
    float* s_pvmax = s_alpha + 48;

    const int hkv = blockIdx.x;
    const int h0 = 3 * hkv;
    const int t0 = blockIdx.y * 16;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int kv_dim = n_head_kv * 128;
    constexpr float kLog2e = 1.4426950408889634f;

    for (int base_r = 0; base_r < 48; base_r += 32) {
        const int qr = base_r + (tid >> 3);
        const int qc0 = (tid & 7) * 16;
        if (qr < 48) {
            const int g = qr >> 4;
            const int r = qr & 15;
            const int t = t0 + r;
            const int h = h0 + g;
            float amax = 0.0f;
            const float* qp = (t < T) ? (Q + ((size_t) t * n_head + h) * 128 + qc0) : nullptr;
            if (qp) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float4 v = *reinterpret_cast<const float4*>(qp + i * 4);
                    amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
                }
            }
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
            const float qscale = (amax > 0.0f) ? amax : 1.0f;
            const float inv_qs = __fdividef(1.0f, qscale);
            if ((tid & 7) == 0) {
                s_qscale[qr] = qscale * kq_scale * kLog2e;
                s_m[qr] = 0.0f;
                s_l[qr] = 1.0f;
            }
            __half2* dst2 = reinterpret_cast<__half2*>(sQ + qr * 136 + qc0);
            if (qp) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float4 v = *reinterpret_cast<const float4*>(qp + i * 4);
                    dst2[i * 2 + 0] = __halves2half2(__float2half(v.x * inv_qs), __float2half(v.y * inv_qs));
                    dst2[i * 2 + 1] = __halves2half2(__float2half(v.z * inv_qs), __float2half(v.w * inv_qs));
                }
            } else {
                const __half2 z2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));
#pragma unroll
                for (int i = 0; i < 8; ++i) dst2[i] = z2;
            }
        }
    }
    __syncthreads();

    const int p0 = pos0[0];
    const int t_last = min(t0 + 15, T - 1);
    const int j_start = is_swa ? max(0, (p0 + t0) - n_swa + 1) : 0;
    const int j_end = p0 + t_last + 1;
    const int bulk_end = is_swa ? j_start : max(j_start, min(j_end, p0 + t0 - n_ext + 1));
    const bool pow2_cap = (cap & (cap - 1)) == 0;
    const int cap_mask = cap - 1;

    float acc[3][8] = {};
    const int kv_r = tid >> 2;
    const int kv_c0 = (tid & 3) * 32;
    const int my_col = tid & 127;
    const int my_half = tid >> 7;

    auto step_tile = [&](int jb, bool is_bulk) {
        {
            const int j = jb + kv_r;
            const bool valid = is_bulk ? true : (j < j_end);
            const int slot = is_bulk ? j : (valid ? (is_swa ? (pow2_cap ? (j & cap_mask) : (j % cap)) : j) : 0);
            if ((tid & 3) == 0) {
                const size_t idx = (size_t) slot * n_head_kv + hkv;
                s_dk[kv_r] = valid ? __half2float(__ldg(&ks[idx])) : 0.0f;
                s_dv[kv_r] = valid ? __half2float(__ldg(&vs[idx])) : 0.0f;
            }
            __half* dst = sKV + kv_r * 136 + kv_c0;
            if (valid) {
                const int8_t* kp = kq + (size_t) slot * kv_dim + hkv * 128 + kv_c0;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(kp + i * 16);
                    __half2 h2[8];
                    i8x4_to_h2x2(raw.x, h2[0], h2[1]);
                    i8x4_to_h2x2(raw.y, h2[2], h2[3]);
                    i8x4_to_h2x2(raw.z, h2[4], h2[5]);
                    i8x4_to_h2x2(raw.w, h2[6], h2[7]);
                    *reinterpret_cast<uint4*>(dst + i * 16) = *reinterpret_cast<const uint4*>(h2);
                    *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                }
            } else {
                const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int i = 0; i < 4; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
            }
        }
        __syncthreads();

        {
            const int ct = warp & 3;
            const int wh = warp >> 2;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa0, fa1;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
            if (wh == 0) {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc0, fc2;
                wmma::fill_fragment(fc0, 0.0f);
                wmma::fill_fragment(fc2, 0.0f);
#pragma unroll
                for (int k = 0; k < 128; k += 16) {
                    wmma::load_matrix_sync(fb, sKV + (ct * 16) * 136 + k, 136);
                    wmma::load_matrix_sync(fa0, sQ + k, 136);
                    wmma::load_matrix_sync(fa1, sQ + 32 * 136 + k, 136);
                    wmma::mma_sync(fc0, fa0, fb, fc0);
                    wmma::mma_sync(fc2, fa1, fb, fc2);
                }
                wmma::store_matrix_sync(sS + ct * 16, fc0, 64, wmma::mem_row_major);
                wmma::store_matrix_sync(sS + 32 * 64 + ct * 16, fc2, 64, wmma::mem_row_major);
            } else {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc1;
                wmma::fill_fragment(fc1, 0.0f);
#pragma unroll
                for (int k = 0; k < 128; k += 16) {
                    wmma::load_matrix_sync(fb, sKV + (ct * 16) * 136 + k, 136);
                    wmma::load_matrix_sync(fa0, sQ + 16 * 136 + k, 136);
                    wmma::mma_sync(fc1, fa0, fb, fc1);
                }
                wmma::store_matrix_sync(sS + 16 * 64 + ct * 16, fc1, 64, wmma::mem_row_major);
            }
        }
        __syncthreads();

        __half p_reg0[6], p_reg1[6];
        {
            const int j = jb + kv_r;
            const bool valid = is_bulk ? true : (j < j_end);
            __half* dst = sKV + kv_r * 136 + kv_c0;
            if (valid) {
                const int slot = is_bulk ? j : (is_swa ? (pow2_cap ? (j & cap_mask) : (j % cap)) : j);
                const int8_t* vp = vq + (is_swa ? ((size_t) slot * kv_dim + hkv * 128) : (((size_t) hkv * cap + slot) * 128)) + kv_c0;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(vp + i * 16);
                    __half2 h2[8];
                    i8x4_to_h2x2(raw.x, h2[0], h2[1]);
                    i8x4_to_h2x2(raw.y, h2[2], h2[3]);
                    i8x4_to_h2x2(raw.z, h2[4], h2[5]);
                    i8x4_to_h2x2(raw.w, h2[6], h2[7]);
                    *reinterpret_cast<uint4*>(dst + i * 16) = *reinterpret_cast<const uint4*>(h2);
                    *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                }
            } else {
                const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int i = 0; i < 4; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
            }

            const int col0 = lane;
            const int col1 = lane + 32;
            const int j_0 = jb + col0;
            const int j_1 = jb + col1;
            const float dk0 = s_dk[col0], dk1 = s_dk[col1];
            const float dv0 = s_dv[col0], dv1 = s_dv[col1];

#pragma unroll
            for (int rr = 0; rr < 6; ++rr) {
                const int qr = warp * 6 + rr;
                const int g = qr >> 4;
                const int r = qr & 15;
                const int t = t0 + r;
                const int h = h0 + g;
                const bool t_ok = t < T;
                const float nsink = t_ok ? (-sink[(size_t) t * n_head + h] * kLog2e) : 0.0f;
                const float qks2 = s_qscale[qr];

                float sc0, sc1;
                bool v0, v1;
                if (is_bulk) {
                    v0 = t_ok;
                    v1 = t_ok;
                    sc0 = t_ok ? (sS[qr * 64 + col0] * qks2 * dk0 + nsink) : -1e30f;
                    sc1 = t_ok ? (sS[qr * 64 + col1] * qks2 * dk1 + nsink) : -1e30f;
                } else {
                    const int pos_q = p0 + t;
                    const float* rt = t_ok ? (rel_tab + ((size_t) t * n_head + h) * n_ext) : nullptr;
                    const int d0 = pos_q - j_0;
                    const int d1 = pos_q - j_1;
                    v0 = t_ok && (j_0 <= pos_q) && (!is_swa || d0 < n_swa);
                    v1 = t_ok && (j_1 <= pos_q) && (!is_swa || d1 < n_swa);
                    sc0 = v0 ? (sS[qr * 64 + col0] * qks2 * dk0 + (d0 < n_ext ? (rt[d0] * kLog2e) : nsink)) : -1e30f;
                    sc1 = v1 ? (sS[qr * 64 + col1] * qks2 * dk1 + (d1 < n_ext ? (rt[d1] * kLog2e) : nsink)) : -1e30f;
                }

                float rmax = fmaxf(sc0, sc1);
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffffu, rmax, o));
                const float m_old = s_m[qr], l_old = s_l[qr];
                const float m_new = fmaxf(m_old, rmax);
                const float alpha = l_old > 0.0f ? exp2f(m_old - m_new) : 0.0f;
                const float p_0 = v0 ? exp2f(sc0 - m_new) : 0.0f;
                const float p_1 = v1 ? exp2f(sc1 - m_new) : 0.0f;
                float psum = p_0 + p_1;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) psum += __shfl_xor_sync(0xffffffffu, psum, o);

                const float pv_0 = p_0 * dv0, pv_1 = p_1 * dv1;
                float pvmax = fmaxf(pv_0, pv_1);
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) pvmax = fmaxf(pvmax, __shfl_xor_sync(0xffffffffu, pvmax, o));
                const float inv_pv = pvmax > 0.0f ? __fdividef(1.0f, pvmax) : 0.0f;

                p_reg0[rr] = __float2half(pv_0 * inv_pv);
                p_reg1[rr] = __float2half(pv_1 * inv_pv);
                if (lane == 0) {
                    s_m[qr] = m_new;
                    s_l[qr] = l_old * alpha + psum;
                    s_alpha[qr] = alpha;
                    s_pvmax[qr] = pvmax;
                }
            }
        }
        __syncthreads();

        {
            const int col0 = lane;
            const int col1 = lane + 32;
#pragma unroll
            for (int rr = 0; rr < 6; ++rr) {
                const int qr = warp * 6 + rr;
                sP[qr * 72 + col0] = p_reg0[rr];
                sP[qr * 72 + col1] = p_reg1[rr];
            }
        }
        __syncthreads();

        {
            const int cbase = warp * 16;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc0, fc1, fc2;
            wmma::fill_fragment(fc0, 0.0f);
            wmma::fill_fragment(fc1, 0.0f);
            wmma::fill_fragment(fc2, 0.0f);
#pragma unroll
            for (int k = 0; k < 64; k += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa0, fa1, fa2;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
                wmma::load_matrix_sync(fb, sKV + k * 136 + cbase, 136);
                wmma::load_matrix_sync(fa0, sP + k, 72);
                wmma::load_matrix_sync(fa1, sP + 16 * 72 + k, 72);
                wmma::load_matrix_sync(fa2, sP + 32 * 72 + k, 72);
                wmma::mma_sync(fc0, fa0, fb, fc0);
                wmma::mma_sync(fc1, fa1, fb, fc1);
                wmma::mma_sync(fc2, fa2, fb, fc2);
            }
            __syncthreads();
            wmma::store_matrix_sync(s_tile + cbase, fc0, 132, wmma::mem_row_major);
            wmma::store_matrix_sync(s_tile + 16 * 132 + cbase, fc1, 132, wmma::mem_row_major);
            wmma::store_matrix_sync(s_tile + 32 * 132 + cbase, fc2, 132, wmma::mem_row_major);
        }
        __syncthreads();

        const int row0 = my_half * 8;
#pragma unroll
        for (int g = 0; g < 3; ++g) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int qr = g * 16 + row0 + r;
                acc[g][r] = acc[g][r] * s_alpha[qr] + s_pvmax[qr] * s_tile[qr * 132 + my_col];
            }
        }
        __syncthreads();
    };

    int jb = j_start;
    for (; jb + 64 <= bulk_end; jb += 64) step_tile(jb, true);
    for (; jb < j_end; jb += 64) step_tile(jb, false);

    const int row0 = my_half * 8;
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        const int h = h0 + g;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int t = t0 + row0 + r;
            if (t < T) {
                const size_t oi = ((size_t) t * n_head + h) * 128 + my_col;
                float val = __fdividef(acc[g][r], s_l[g * 16 + row0 + r]);
                if (gate) val *= sigmf(gate[oi]);
                out[oi] = val;
            }
        }
    }
#endif
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
__device__ __forceinline__ float fast_exp2f_hw(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void i8x4_to_h2x2_fma(uint32_t packed_i8x4, __half2 sc2, __half2 neg_bias_sc2,
                                                 __half2& out0, __half2& out1) {
    const uint32_t u = packed_i8x4 ^ 0x80808080u;
    const uint32_t lo = __byte_perm(u, 0x6400u, 0x5150u);
    const uint32_t hi = __byte_perm(u, 0x6400u, 0x5352u);
    out0 = __hfma2(*reinterpret_cast<const __half2*>(&lo), sc2, neg_bias_sc2);
    out1 = __hfma2(*reinterpret_cast<const __half2*>(&hi), sc2, neg_bias_sc2);
}

__device__ __forceinline__ void i8x4_to_h2x2_scaled(uint32_t packed_i8x4, __half2 scale2, __half2& out0, __half2& out1) {
    const __half2 neg1152 = __half2half2(__float2half(-1152.0f));
    i8x4_to_h2x2_fma(packed_i8x4, scale2, __hmul2(neg1152, scale2), out0, out1);
}

__device__ __forceinline__ void wmma_acc_layout_16x16(int lane, int& r0, int& r1, int& c0, int& c1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
    r0 = ((lane & 16) >> 2) + ((lane & 4) << 1) + (lane & 1);
    r1 = r0 + 2;
    c0 = (lane & 8) + (lane & 2);
    c1 = c0 + 4;
#else
    r0 = lane >> 2;
    r1 = r0 + 8;
    c0 = (lane & 3) * 2;
    c1 = c0 + 8;
#endif
}

__device__ __forceinline__ float wmma_row_max_16(float v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
#else
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
#endif
    return v;
}

__device__ __forceinline__ float wmma_row_sum_16(float v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 8);
#else
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
#endif
    return v;
}

__device__ __forceinline__ bool wmma_row_leader_16(int lane) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
    return (lane & 10) == 0;
#else
    return (lane & 3) == 0;
#endif
}

__device__ __forceinline__ void wmma_rescale_acc(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>& fc,
                                                 float a0, float a1) {
    fc.x[0] *= a0; fc.x[1] *= a0; fc.x[4] *= a0; fc.x[5] *= a0;
    fc.x[2] *= a1; fc.x[3] *= a1; fc.x[6] *= a1; fc.x[7] *= a1;
}

__device__ __forceinline__ uint32_t f2_to_h2u(float a, float b) {
    const __half2 h = __float22half2_rn(make_float2(a, b));
    uint32_t u;
    memcpy(&u, &h, 4);
    return u;
}

/// Exact in-register converter: 16x16 FP32 WMMA accumulator (after ex2.approx) -> 16x16 FP16 row_major matrix_a!
/// Zero sP SMEM: 6 warp shuffles on Volta sm_70 (verified 0.00000000 max_err), 0 shuffles on Turing/Ampere/Ada sm_75+.
template <bool MASKED>
__device__ __forceinline__ void wmma_acc_to_matA_16x16(
        const nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>& fc,
        float m_new0, float m_new1, float& ps0, float& ps1,
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major>& fa, int lane) {
    const float p00 = (!MASKED || fc.x[0] > -1e20f) ? fast_exp2f_hw(fc.x[0] - m_new0) : 0.0f;
    const float p01 = (!MASKED || fc.x[1] > -1e20f) ? fast_exp2f_hw(fc.x[1] - m_new0) : 0.0f;
    const float p04 = (!MASKED || fc.x[4] > -1e20f) ? fast_exp2f_hw(fc.x[4] - m_new0) : 0.0f;
    const float p05 = (!MASKED || fc.x[5] > -1e20f) ? fast_exp2f_hw(fc.x[5] - m_new0) : 0.0f;
    ps0 += (p00 + p01) + (p04 + p05);
    const float p12 = (!MASKED || fc.x[2] > -1e20f) ? fast_exp2f_hw(fc.x[2] - m_new1) : 0.0f;
    const float p13 = (!MASKED || fc.x[3] > -1e20f) ? fast_exp2f_hw(fc.x[3] - m_new1) : 0.0f;
    const float p16 = (!MASKED || fc.x[6] > -1e20f) ? fast_exp2f_hw(fc.x[6] - m_new1) : 0.0f;
    const float p17 = (!MASKED || fc.x[7] > -1e20f) ? fast_exp2f_hw(fc.x[7] - m_new1) : 0.0f;
    ps1 += (p12 + p13) + (p16 + p17);

    const uint32_t a0 = f2_to_h2u(p00, p01), a1 = f2_to_h2u(p04, p05);
    const uint32_t b0 = f2_to_h2u(p12, p13), b1 = f2_to_h2u(p16, p17);
    uint32_t* fa_u = reinterpret_cast<uint32_t*>(fa.x);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
    const bool want_r1 = (lane & 2) != 0;
    const uint32_t recv0 = __shfl_xor_sync(0xffffffffu, want_r1 ? a0 : b0, 2);
    const uint32_t recv1 = __shfl_xor_sync(0xffffffffu, want_r1 ? a1 : b1, 2);
    const uint32_t h0 = want_r1 ? recv0 : a0, h1 = want_r1 ? b0 : recv0;
    const uint32_t h2 = want_r1 ? recv1 : a1, h3 = want_r1 ? b1 : recv1;
    const uint32_t o0 = __shfl_xor_sync(0xffffffffu, h0, 8), o1 = __shfl_xor_sync(0xffffffffu, h1, 8);
    const uint32_t o2 = __shfl_xor_sync(0xffffffffu, h2, 8), o3 = __shfl_xor_sync(0xffffffffu, h3, 8);
    const bool hi8 = (lane & 8) != 0;
    fa_u[0] = hi8 ? o0 : h0; fa_u[1] = hi8 ? o1 : h1;
    fa_u[2] = hi8 ? o2 : h2; fa_u[3] = hi8 ? o3 : h3;
    fa_u[4] = hi8 ? h0 : o0; fa_u[5] = hi8 ? h1 : o1;
    fa_u[6] = hi8 ? h2 : o2; fa_u[7] = hi8 ? h3 : o3;
#else
    (void) lane;
    fa_u[0] = fa_u[4] = a0;
    fa_u[1] = fa_u[5] = b0;
    fa_u[2] = fa_u[6] = a1;
    fa_u[3] = fa_u[7] = b1;
#endif
}
#endif

/// FlashAttention-2 192-row (64 tokens x 3 GQA heads, 256 threads = 8 warps, 1 blk/SM, 97,280 B SMEM) FP16 WMMA prefill:
/// 8-warp (4 row-groups x 2 col-groups, exact 2 warps/SMSP, 0 register spills): 3x2 Q*K^T -> bar.sync 64-thr half-row max
/// merge -> sP -> 3x4 register-resident P*V (0.583 SMEM loads/WMMA), HFMA2 KV dequant, shifted-nsink bulk loop, lazy acc rescale.
template <bool SPLIT_KV, bool IS_SWA, bool FUSED_REL = false>
__global__ void __launch_bounds__(256, 1) k_attn_q8_gqa3_pp_fa2_192(
        const float* __restrict__ Q,
        const int8_t* __restrict__ kq, const __half* __restrict__ ks,
        const int8_t* __restrict__ vq, const __half* __restrict__ vs,
        const float* __restrict__ rel_tab, const float* __restrict__ sink,
        const float* __restrict__ gate, float* __restrict__ out,
        float* __restrict__ split_out,
        int n_head, int n_head_kv, int n_ext, int cap, int n_swa,
        const int32_t* __restrict__ pos0, float kq_scale, int T, int split_n,
        const float* __restrict__ R_ptr = nullptr, const float* __restrict__ rel_bias_ptr = nullptr,
        uint16_t* __restrict__ out_f16 = nullptr) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    extern __shared__ __align__(16) char smem_raw[];
    // 97,280 B <= 98,304 B (96 KB on sm_70, 100 KB on sm_89):
    // sQ: 192*136*2 = 52,224 B | sKV: 64*136*2 = 17,408 B | sP: 192*72*2 = 27,648 B
    // Scalar metadata packed into unused 16 B tail of each sP row (sP[qr*72 + 64..71]):
    //   +64: half-max cg=0 (and final rm in epilogue), +66: half-max cg=1, +68: final rl0, +70: final rl1.
    __half* sQ = reinterpret_cast<__half*>(smem_raw);
    __half* sKV = sQ + 192 * 136;
    __half* sP = sKV + 64 * 136;
    auto s_hmax_ptr = [&](int qr, int c) -> float* { return reinterpret_cast<float*>(sP + qr * 72 + 64 + (c << 1)); };
    auto s_m_ptr    = [&](int qr) -> float* { return reinterpret_cast<float*>(sP + qr * 72 + 64); };
    auto s_l0_ptr   = [&](int qr) -> float* { return reinterpret_cast<float*>(sP + qr * 72 + 68); };
    auto s_l1_ptr   = [&](int qr) -> float* { return reinterpret_cast<float*>(sP + qr * 72 + 70); };

    // D11: Swapped grid dim3(q_tiles, n_head_kv[, split_n]) maximizes L2 KV reuse across adjacent Q tiles!
    const int t0 = blockIdx.x * 64, hkv = blockIdx.y, h0 = 3 * hkv;
    const int sp = SPLIT_KV ? (int) blockIdx.z : 0;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int rg = warp >> 1;      // 0..3 (48 rows each)
    const int cg = warp & 1;       // 0..1 (QK^T: cols cg*32..+31; P*V: output cols cg*64..+63)
    const int kv_dim = n_head_kv * 128;
    constexpr float kLog2e = 1.4426950408889634f;
    const float qscale_all = kq_scale * kLog2e;

    const int p0 = pos0[0];
    const int t_last = min(t0 + 63, T - 1);
    const int j_start_all = IS_SWA ? max(0, (p0 + t0) - n_swa + 1) : 0;
    const int j_end_all = p0 + t_last + 1;

    int j_start = j_start_all;
    int j_end = j_end_all;
    if constexpr (SPLIT_KV) {
        const int n_tiles = (j_end_all - j_start_all + 63) / 64;
        const int t_per_sp = (n_tiles + split_n - 1) / split_n;
        j_start = j_start_all + sp * t_per_sp * 64;
        j_end = min(j_end_all, j_start + t_per_sp * 64);
        if (j_start >= j_end_all) {
            if (tid < 192) {
                const int g = tid >> 6, r = tid & 63, t = t0 + r;
                if (t < T) {
                    float* dst = split_out + (((size_t) t * split_n + sp) * n_head + (h0 + g)) * 130;
                    dst[128] = -1e30f;
                    dst[129] = 0.0f;
                }
            }
            return;
        }
    }

    // 256 threads load 192x128 Q -> sQ in 6 iterations (32 rows/iter, 8 threads/row, 16 elems/thread).
    for (int base_r = 0; base_r < 192; base_r += 32) {
        const int qr = base_r + (tid >> 3);
        const int qc0 = (tid & 7) * 16;
        const int g = qr >> 6, r = qr & 63, t = t0 + r, h = h0 + g;
        __half2* dst2 = reinterpret_cast<__half2*>(sQ + qr * 136 + qc0);
        if (t < T) {
            const float* qp = Q + ((size_t) t * n_head + h) * 128 + qc0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = *reinterpret_cast<const float4*>(qp + i * 4);
                dst2[i * 2 + 0] = __halves2half2(__float2half(v.x * qscale_all), __float2half(v.y * qscale_all));
                dst2[i * 2 + 1] = __halves2half2(__float2half(v.z * qscale_all), __float2half(v.w * qscale_all));
            }
        } else {
            const __half2 z2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));
#pragma unroll
            for (int i = 0; i < 8; ++i) dst2[i] = z2;
        }
    }
    __syncthreads();

    const int bulk_end = IS_SWA ? j_start : max(j_start, min(j_end, p0 + t0 - n_ext + 1));
    const bool pow2_cap = IS_SWA ? ((cap & (cap - 1)) == 0) : false;
    const int cap_mask = cap - 1;

    int acc_r0, acc_r1, acc_c0, acc_c1;
    wmma_acc_layout_16x16(lane, acc_r0, acc_r1, acc_c0, acc_c1);
    const bool is_leader = wmma_row_leader_16(lane);

    const int qr_base = rg * 48;
    const int kv_cbase = cg * 32;
    const int pv_pc0 = cg * 64;
    const bool has_sink = (!SPLIT_KV || sp == 0);
    const float init_m = has_sink ? 0.0f : -1e30f;
    const float init_l = (has_sink && cg == 0) ? 1.0f : 0.0f;

    float rm0[3], rm1[3], rl0[3], rl1[3];
#pragma unroll
    for (int m = 0; m < 3; ++m) {
        rm0[m] = init_m; rl0[m] = init_l;
        rm1[m] = init_m; rl1[m] = init_l;
    }

    auto load_nsink = [&](int qr) -> float {
        const int tr = t0 + (qr & 63), hr = h0 + (qr >> 6);
        return (tr < T) ? (-__ldg(&sink[(size_t) tr * n_head + hr]) * kLog2e) : 0.0f;
    };

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[3][4];
#pragma unroll
    for (int m = 0; m < 3; ++m) {
#pragma unroll
        for (int n = 0; n < 4; ++n) wmma::fill_fragment(acc[m][n], 0.0f);
    }

    // All 256 threads participate branchlessly in 64x128 KV loading (4 threads/row x 32 elems/thread).
    const int kv_r = tid >> 2, kv_c0 = (tid & 3) * 32;
    const __half2 neg1152 = __half2half2(__float2half(-1152.0f));
    const int8_t* kq_h = kq + (size_t) hkv * cap * 128 + kv_c0;
    const int8_t* vq_h = vq + (size_t) hkv * cap * 128 + kv_c0;
    const __half* ks_h = ks + (size_t) hkv * cap;
    const __half* vs_h = vs + (size_t) hkv * cap;

    // Fast specialized bulk loop (!IS_SWA, dist >= n_ext): shifted-nsink (0 bulk adds), HFMA2 KV prefetch, lazy acc rescale.
    int jb = j_start;
    if constexpr (!IS_SWA) {
    const int bulk_tiles_end = bulk_end - 63;
    if (jb < bulk_tiles_end) {
#pragma unroll
        for (int m = 0; m < 3; ++m) {
            if (rm0[m] > -1e20f) rm0[m] -= load_nsink(qr_base + m * 16 + acc_r0);
            if (rm1[m] > -1e20f) rm1[m] -= load_nsink(qr_base + m * 16 + acc_r1);
        }
        uint4 kraw0, kraw1;
        __half2 dk2;
        {
            const int j = jb + kv_r;
            dk2 = __half2half2(__ldg(&ks_h[j]));
            const int8_t* kp = kq_h + (size_t) j * 128;
            kraw0 = *reinterpret_cast<const uint4*>(kp);
            kraw1 = *reinterpret_cast<const uint4*>(kp + 16);
        }
        for (; jb < bulk_tiles_end; jb += 64) {
            {
                __half* dst = sKV + kv_r * 136 + kv_c0;
                const __half2 kbias = __hmul2(neg1152, dk2);
                __half2 h2[8];
                i8x4_to_h2x2_fma(kraw0.x, dk2, kbias, h2[0], h2[1]); i8x4_to_h2x2_fma(kraw0.y, dk2, kbias, h2[2], h2[3]);
                i8x4_to_h2x2_fma(kraw0.z, dk2, kbias, h2[4], h2[5]); i8x4_to_h2x2_fma(kraw0.w, dk2, kbias, h2[6], h2[7]);
                *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(h2);
                *reinterpret_cast<uint4*>(dst + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                i8x4_to_h2x2_fma(kraw1.x, dk2, kbias, h2[0], h2[1]); i8x4_to_h2x2_fma(kraw1.y, dk2, kbias, h2[2], h2[3]);
                i8x4_to_h2x2_fma(kraw1.z, dk2, kbias, h2[4], h2[5]); i8x4_to_h2x2_fma(kraw1.w, dk2, kbias, h2[6], h2[7]);
                *reinterpret_cast<uint4*>(dst + 16) = *reinterpret_cast<const uint4*>(h2);
                *reinterpret_cast<uint4*>(dst + 24) = *reinterpret_cast<const uint4*>(h2 + 4);
            }
            __syncthreads();

            uint4 vraw0, vraw1;
            __half2 dv2;
            {
                const int j = jb + kv_r;
                dv2 = __half2half2(__ldg(&vs_h[j]));
                const int8_t* vp = vq_h + (size_t) j * 128;
                vraw0 = *reinterpret_cast<const uint4*>(vp);
                vraw1 = *reinterpret_cast<const uint4*>(vp + 16);
            }

            wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc[3][2];
#pragma unroll
            for (int m = 0; m < 3; ++m) {
                wmma::fill_fragment(fc[m][0], 0.0f);
                wmma::fill_fragment(fc[m][1], 0.0f);
            }
#pragma unroll
            for (int k = 0; k < 128; k += 16) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb0, fb1;
                wmma::load_matrix_sync(fb0, sKV + (kv_cbase + 0) * 136 + k, 136);
                wmma::load_matrix_sync(fb1, sKV + (kv_cbase + 16) * 136 + k, 136);
#pragma unroll
                for (int m = 0; m < 3; ++m) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
                    wmma::load_matrix_sync(fa, sQ + (qr_base + m * 16) * 136 + k, 136);
                    wmma::mma_sync(fc[m][0], fa, fb0, fc[m][0]);
                    wmma::mma_sync(fc[m][1], fa, fb1, fc[m][1]);
                }
            }

#pragma unroll
            for (int m = 0; m < 3; ++m) {
                const float mx0 = wmma_row_max_16(
                    fmaxf(fmaxf(fmaxf(fc[m][0].x[0], fc[m][0].x[1]), fmaxf(fc[m][0].x[4], fc[m][0].x[5])),
                          fmaxf(fmaxf(fc[m][1].x[0], fc[m][1].x[1]), fmaxf(fc[m][1].x[4], fc[m][1].x[5]))));
                const float mx1 = wmma_row_max_16(
                    fmaxf(fmaxf(fmaxf(fc[m][0].x[2], fc[m][0].x[3]), fmaxf(fc[m][0].x[6], fc[m][0].x[7])),
                          fmaxf(fmaxf(fc[m][1].x[2], fc[m][1].x[3]), fmaxf(fc[m][1].x[6], fc[m][1].x[7]))));
                if (is_leader) {
                    *s_hmax_ptr(qr_base + m * 16 + acc_r0, cg) = mx0;
                    *s_hmax_ptr(qr_base + m * 16 + acc_r1, cg) = mx1;
                }
            }
            asm volatile("bar.sync %0, 64;" :: "r"(rg + 1) : "memory");

#pragma unroll
            for (int m = 0; m < 3; ++m) {
                const int qr0 = qr_base + m * 16 + acc_r0;
                const int qr1 = qr_base + m * 16 + acc_r1;
                const float mx0 = fmaxf(*s_hmax_ptr(qr0, 0), *s_hmax_ptr(qr0, 1));
                const float mx1 = fmaxf(*s_hmax_ptr(qr1, 0), *s_hmax_ptr(qr1, 1));
                const float m_new0 = fmaxf(rm0[m], mx0);
                const float m_new1 = fmaxf(rm1[m], mx1);
                const float a0 = (rm0[m] > -1e20f) ? ((m_new0 == rm0[m]) ? 1.0f : fast_exp2f_hw(rm0[m] - m_new0)) : 0.0f;
                const float a1 = (rm1[m] > -1e20f) ? ((m_new1 == rm1[m]) ? 1.0f : fast_exp2f_hw(rm1[m] - m_new1)) : 0.0f;
                if (__any_sync(0xffffffffu, a0 != 1.0f || a1 != 1.0f)) {
#pragma unroll
                    for (int n = 0; n < 4; ++n) wmma_rescale_acc(acc[m][n], a0, a1);
                }
                float psum0 = 0.0f, psum1 = 0.0f;
#pragma unroll
                for (int ct = 0; ct < 2; ++ct) {
                    const float p00 = fast_exp2f_hw(fc[m][ct].x[0] - m_new0), p01 = fast_exp2f_hw(fc[m][ct].x[1] - m_new0);
                    const float p04 = fast_exp2f_hw(fc[m][ct].x[4] - m_new0), p05 = fast_exp2f_hw(fc[m][ct].x[5] - m_new0);
                    psum0 += (p00 + p01) + (p04 + p05);
                    *reinterpret_cast<__half2*>(sP + qr0 * 72 + kv_cbase + ct * 16 + acc_c0) = __halves2half2(__float2half(p00), __float2half(p01));
                    *reinterpret_cast<__half2*>(sP + qr0 * 72 + kv_cbase + ct * 16 + acc_c1) = __halves2half2(__float2half(p04), __float2half(p05));

                    const float p12 = fast_exp2f_hw(fc[m][ct].x[2] - m_new1), p13 = fast_exp2f_hw(fc[m][ct].x[3] - m_new1);
                    const float p16 = fast_exp2f_hw(fc[m][ct].x[6] - m_new1), p17 = fast_exp2f_hw(fc[m][ct].x[7] - m_new1);
                    psum1 += (p12 + p13) + (p16 + p17);
                    *reinterpret_cast<__half2*>(sP + qr1 * 72 + kv_cbase + ct * 16 + acc_c0) = __halves2half2(__float2half(p12), __float2half(p13));
                    *reinterpret_cast<__half2*>(sP + qr1 * 72 + kv_cbase + ct * 16 + acc_c1) = __halves2half2(__float2half(p16), __float2half(p17));
                }
                psum0 = wmma_row_sum_16(psum0);
                psum1 = wmma_row_sum_16(psum1);
                rm0[m] = m_new0; rl0[m] = rl0[m] * a0 + psum0;
                rm1[m] = m_new1; rl1[m] = rl1[m] * a1 + psum1;
            }
            __syncthreads();

            {
                __half* dst = sKV + kv_r * 136 + kv_c0;
                const __half2 vbias = __hmul2(neg1152, dv2);
                __half2 h2[8];
                i8x4_to_h2x2_fma(vraw0.x, dv2, vbias, h2[0], h2[1]); i8x4_to_h2x2_fma(vraw0.y, dv2, vbias, h2[2], h2[3]);
                i8x4_to_h2x2_fma(vraw0.z, dv2, vbias, h2[4], h2[5]); i8x4_to_h2x2_fma(vraw0.w, dv2, vbias, h2[6], h2[7]);
                *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(h2);
                *reinterpret_cast<uint4*>(dst + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                i8x4_to_h2x2_fma(vraw1.x, dv2, vbias, h2[0], h2[1]); i8x4_to_h2x2_fma(vraw1.y, dv2, vbias, h2[2], h2[3]);
                i8x4_to_h2x2_fma(vraw1.z, dv2, vbias, h2[4], h2[5]); i8x4_to_h2x2_fma(vraw1.w, dv2, vbias, h2[6], h2[7]);
                *reinterpret_cast<uint4*>(dst + 16) = *reinterpret_cast<const uint4*>(h2);
                *reinterpret_cast<uint4*>(dst + 24) = *reinterpret_cast<const uint4*>(h2 + 4);
            }
            __syncthreads();

            if (jb + 64 < bulk_tiles_end) {
                const int j = jb + 64 + kv_r;
                dk2 = __half2half2(__ldg(&ks_h[j]));
                const int8_t* kp = kq_h + (size_t) j * 128;
                kraw0 = *reinterpret_cast<const uint4*>(kp);
                kraw1 = *reinterpret_cast<const uint4*>(kp + 16);
            }
#pragma unroll
            for (int k = 0; k < 64; k += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[3];
#pragma unroll
                for (int m = 0; m < 3; ++m) {
                    wmma::load_matrix_sync(fa[m], sP + (qr_base + m * 16) * 72 + k, 72);
                }
#pragma unroll
                for (int n = 0; n < 4; ++n) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
                    wmma::load_matrix_sync(fb, sKV + k * 136 + pv_pc0 + n * 16, 136);
#pragma unroll
                    for (int m = 0; m < 3; ++m) {
                        wmma::mma_sync(acc[m][n], fa[m], fb, acc[m][n]);
                    }
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int m = 0; m < 3; ++m) {
            if (rm0[m] > -1e20f) rm0[m] += load_nsink(qr_base + m * 16 + acc_r0);
            if (rm1[m] > -1e20f) rm1[m] += load_nsink(qr_base + m * 16 + acc_r1);
        }
    }
    }

    // Tail / SWA loop (causal masking + relative distance bias + V/K-prefetch + hoisted fa[3] + zero % cap + interior SWA fast-path).
    if (jb < j_end) {
        auto mask_bias = [&](float s, int qr, int j) -> float {
            const int tr = t0 + (qr & 63);
            const int pos_q = p0 + tr;
            const int d = pos_q - j;
            const bool v = (tr < T) && (j <= pos_q) && (!IS_SWA || d < n_swa);
            if (!v) return -1e30f;
            const int hr = h0 + (qr >> 6);
            const float b = (d < n_ext)
                ? (__ldg(&rel_tab[((size_t) tr * n_head + hr) * n_ext + d]) * kLog2e)
                : (-__ldg(&sink[(size_t) tr * n_head + hr]) * kLog2e);
            return s + b;
        };
        int slot_cur = 0;
        if constexpr (IS_SWA) {
            const int j_first = jb + kv_r;
            slot_cur = pow2_cap ? (j_first & cap_mask) : (j_first % cap);
        } else {
            slot_cur = jb + kv_r;
        }
        for (; jb < j_end; jb += 64) {
            const int j_cur = jb + kv_r;
            const bool valid_cur = j_cur < j_end;
            {
                __half* dst = sKV + kv_r * 136 + kv_c0;
                if (valid_cur) {
                    const __half2 dk2_t = __half2half2(__ldg(IS_SWA ? &ks[(size_t) slot_cur * n_head_kv + hkv] : &ks_h[slot_cur]));
                    const __half2 kbias = __hmul2(neg1152, dk2_t);
                    const int8_t* kp = IS_SWA ? (kq + (size_t) slot_cur * kv_dim + hkv * 128 + kv_c0) : (kq_h + (size_t) slot_cur * 128);
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        const uint4 raw = *reinterpret_cast<const uint4*>(kp + i * 16);
                        __half2 h2[8];
                        i8x4_to_h2x2_fma(raw.x, dk2_t, kbias, h2[0], h2[1]); i8x4_to_h2x2_fma(raw.y, dk2_t, kbias, h2[2], h2[3]);
                        i8x4_to_h2x2_fma(raw.z, dk2_t, kbias, h2[4], h2[5]); i8x4_to_h2x2_fma(raw.w, dk2_t, kbias, h2[6], h2[7]);
                        *reinterpret_cast<uint4*>(dst + i * 16) = *reinterpret_cast<const uint4*>(h2);
                        *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                    }
                } else {
                    const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
                    for (int i = 0; i < 4; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
                }
            }
            __syncthreads();

            // Prefetch V in registers overlapping QK^T + softmax!
            uint4 vraw0 = make_uint4(0, 0, 0, 0), vraw1 = make_uint4(0, 0, 0, 0);
            __half2 dv2_t = __half2half2(__float2half(0.0f));
            if (valid_cur) {
                dv2_t = __half2half2(__ldg(IS_SWA ? &vs[(size_t) slot_cur * n_head_kv + hkv] : &vs_h[slot_cur]));
                const int8_t* vp = IS_SWA ? (vq + (size_t) slot_cur * kv_dim + hkv * 128 + kv_c0) : (vq_h + (size_t) slot_cur * 128);
                vraw0 = *reinterpret_cast<const uint4*>(vp);
                vraw1 = *reinterpret_cast<const uint4*>(vp + 16);
            }

            wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc[3][2];
#pragma unroll
            for (int m = 0; m < 3; ++m) {
                wmma::fill_fragment(fc[m][0], 0.0f);
                wmma::fill_fragment(fc[m][1], 0.0f);
            }
#pragma unroll 1
            for (int k = 0; k < 128; k += 16) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb0, fb1;
                wmma::load_matrix_sync(fb0, sKV + (kv_cbase + 0) * 136 + k, 136);
                wmma::load_matrix_sync(fb1, sKV + (kv_cbase + 16) * 136 + k, 136);
#pragma unroll
                for (int m = 0; m < 3; ++m) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
                    wmma::load_matrix_sync(fa, sQ + (qr_base + m * 16) * 136 + k, 136);
                    wmma::mma_sync(fc[m][0], fa, fb0, fc[m][0]);
                    wmma::mma_sync(fc[m][1], fa, fb1, fc[m][1]);
                }
            }

            const bool all_valid_swa = IS_SWA && (t0 + 63 < T) &&
                                       (jb >= (p0 + t0 + 63) - n_swa + 1) && (jb + 63 <= p0 + t0) &&
                                       ((p0 + t0 + 63) - jb < n_ext);

            if constexpr (FUSED_REL) {
                // Reuse idle sKV (17,408 B) to stage rel_bias[d_lo..d_hi][16] (padded stride 20 = 10,240 B, zero bank conflicts).
                // Eliminates k_rel_table_t32 (100 MB write) AND uncoalesced 4-byte global rel_tab reads with bit-identical FP32 math!
                __syncthreads();
                const int d_lo = max(0, (p0 + t0) - (jb + 63));
                const int d_hi = min(n_ext - 1, (p0 + t0 + 63) - jb);
                const int d_cnt = max(0, d_hi - d_lo + 1);
                float* s_rb = reinterpret_cast<float*>(sKV);
                for (int idx = tid; idx < d_cnt * 4; idx += 256) {
                    const int di = idx >> 2, c4 = idx & 3;
                    *reinterpret_cast<float4*>(s_rb + di * 20 + c4 * 4) =
                        reinterpret_cast<const float4*>(rel_bias_ptr + (size_t) (d_lo + di) * 16)[c4];
                }
                __syncthreads();

                auto fused_row_bias_interior = [&](int pos_q, const float4& q0, const float4& q1, const float4& q2, const float4& q3,
                                                   float sk, float s, int j) -> float {
                    const float4* rb4 = reinterpret_cast<const float4*>(s_rb + (pos_q - j - d_lo) * 20);
                    float dot = 0.0f;
                    { const float4 b = rb4[0]; dot += q0.x * b.x + q0.y * b.y + q0.z * b.z + q0.w * b.w; }
                    { const float4 b = rb4[1]; dot += q1.x * b.x + q1.y * b.y + q1.z * b.z + q1.w * b.w; }
                    { const float4 b = rb4[2]; dot += q2.x * b.x + q2.y * b.y + q2.z * b.z + q2.w * b.w; }
                    { const float4 b = rb4[3]; dot += q3.x * b.x + q3.y * b.y + q3.z * b.z + q3.w * b.w; }
                    return s + (dot - sk) * kLog2e;
                };
                auto fused_row_bias = [&](int qr, const float4& q0, const float4& q1, const float4& q2, const float4& q3,
                                          float sk, float s, int j) -> float {
                    const int tr = t0 + (qr & 63);
                    const int pos_q = p0 + tr;
                    const int d = pos_q - j;
                    if (tr >= T || j > pos_q || (IS_SWA && d >= n_swa)) return -1e30f;
                    if (d < n_ext) {
                        const float4* rb4 = reinterpret_cast<const float4*>(s_rb + (d - d_lo) * 20);
                        float dot = 0.0f;
                        { const float4 b = rb4[0]; dot += q0.x * b.x + q0.y * b.y + q0.z * b.z + q0.w * b.w; }
                        { const float4 b = rb4[1]; dot += q1.x * b.x + q1.y * b.y + q1.z * b.z + q1.w * b.w; }
                        { const float4 b = rb4[2]; dot += q2.x * b.x + q2.y * b.y + q2.z * b.z + q2.w * b.w; }
                        { const float4 b = rb4[3]; dot += q3.x * b.x + q3.y * b.y + q3.z * b.z + q3.w * b.w; }
                        return s + (dot - sk) * kLog2e;
                    }
                    return s - sk * kLog2e;
                };

                if constexpr (IS_SWA) {
                    if (all_valid_swa) {
#pragma unroll
                        for (int m = 0; m < 3; ++m) {
                            const int qr0 = qr_base + m * 16 + acc_r0;
                            const int qr1 = qr_base + m * 16 + acc_r1;
                            float mx0 = -1e30f, mx1 = -1e30f;
                            {
                                const int tr0 = t0 + (qr0 & 63), hr0 = h0 + (qr0 >> 6), pq0 = p0 + tr0;
                                const float4* rp = reinterpret_cast<const float4*>(R_ptr + ((size_t) tr0 * n_head + hr0) * 16);
                                const float4 q0 = rp[0], q1 = rp[1], q2 = rp[2], q3 = rp[3];
                                const float sk0 = __ldg(&sink[(size_t) tr0 * n_head + hr0]);
#pragma unroll
                                for (int ct = 0; ct < 2; ++ct) {
                                    const int j0 = jb + kv_cbase + ct * 16 + acc_c0;
                                    const int j1 = jb + kv_cbase + ct * 16 + acc_c1;
                                    fc[m][ct].x[0] = fused_row_bias_interior(pq0, q0, q1, q2, q3, sk0, fc[m][ct].x[0], j0);
                                    fc[m][ct].x[1] = fused_row_bias_interior(pq0, q0, q1, q2, q3, sk0, fc[m][ct].x[1], j0 + 1);
                                    fc[m][ct].x[4] = fused_row_bias_interior(pq0, q0, q1, q2, q3, sk0, fc[m][ct].x[4], j1);
                                    fc[m][ct].x[5] = fused_row_bias_interior(pq0, q0, q1, q2, q3, sk0, fc[m][ct].x[5], j1 + 1);
                                    mx0 = fmaxf(mx0, fmaxf(fmaxf(fc[m][ct].x[0], fc[m][ct].x[1]), fmaxf(fc[m][ct].x[4], fc[m][ct].x[5])));
                                }
                            }
                            {
                                const int tr1 = t0 + (qr1 & 63), hr1 = h0 + (qr1 >> 6), pq1 = p0 + tr1;
                                const float4* rp = reinterpret_cast<const float4*>(R_ptr + ((size_t) tr1 * n_head + hr1) * 16);
                                const float4 q0 = rp[0], q1 = rp[1], q2 = rp[2], q3 = rp[3];
                                const float sk1 = __ldg(&sink[(size_t) tr1 * n_head + hr1]);
#pragma unroll
                                for (int ct = 0; ct < 2; ++ct) {
                                    const int j0 = jb + kv_cbase + ct * 16 + acc_c0;
                                    const int j1 = jb + kv_cbase + ct * 16 + acc_c1;
                                    fc[m][ct].x[2] = fused_row_bias_interior(pq1, q0, q1, q2, q3, sk1, fc[m][ct].x[2], j0);
                                    fc[m][ct].x[3] = fused_row_bias_interior(pq1, q0, q1, q2, q3, sk1, fc[m][ct].x[3], j0 + 1);
                                    fc[m][ct].x[6] = fused_row_bias_interior(pq1, q0, q1, q2, q3, sk1, fc[m][ct].x[6], j1);
                                    fc[m][ct].x[7] = fused_row_bias_interior(pq1, q0, q1, q2, q3, sk1, fc[m][ct].x[7], j1 + 1);
                                    mx1 = fmaxf(mx1, fmaxf(fmaxf(fc[m][ct].x[2], fc[m][ct].x[3]), fmaxf(fc[m][ct].x[6], fc[m][ct].x[7])));
                                }
                            }
                            mx0 = wmma_row_max_16(mx0);
                            mx1 = wmma_row_max_16(mx1);
                            if (is_leader) {
                                *s_hmax_ptr(qr0, cg) = mx0;
                                *s_hmax_ptr(qr1, cg) = mx1;
                            }
                        }
                    } else {
#pragma unroll
                        for (int m = 0; m < 3; ++m) {
                            const int qr0 = qr_base + m * 16 + acc_r0;
                            const int qr1 = qr_base + m * 16 + acc_r1;
                            float mx0 = -1e30f, mx1 = -1e30f;
                            {
                                const int tr0 = t0 + (qr0 & 63), hr0 = h0 + (qr0 >> 6);
                                float4 q0 = make_float4(0,0,0,0), q1 = q0, q2 = q0, q3 = q0;
                                float sk0 = 0.0f;
                                if (tr0 < T) {
                                    const float4* rp = reinterpret_cast<const float4*>(R_ptr + ((size_t) tr0 * n_head + hr0) * 16);
                                    q0 = rp[0]; q1 = rp[1]; q2 = rp[2]; q3 = rp[3];
                                    sk0 = __ldg(&sink[(size_t) tr0 * n_head + hr0]);
                                }
#pragma unroll
                                for (int ct = 0; ct < 2; ++ct) {
                                    const int j0 = jb + kv_cbase + ct * 16 + acc_c0;
                                    const int j1 = jb + kv_cbase + ct * 16 + acc_c1;
                                    fc[m][ct].x[0] = fused_row_bias(qr0, q0, q1, q2, q3, sk0, fc[m][ct].x[0], j0);
                                    fc[m][ct].x[1] = fused_row_bias(qr0, q0, q1, q2, q3, sk0, fc[m][ct].x[1], j0 + 1);
                                    fc[m][ct].x[4] = fused_row_bias(qr0, q0, q1, q2, q3, sk0, fc[m][ct].x[4], j1);
                                    fc[m][ct].x[5] = fused_row_bias(qr0, q0, q1, q2, q3, sk0, fc[m][ct].x[5], j1 + 1);
                                    mx0 = fmaxf(mx0, fmaxf(fmaxf(fc[m][ct].x[0], fc[m][ct].x[1]), fmaxf(fc[m][ct].x[4], fc[m][ct].x[5])));
                                }
                            }
                            {
                                const int tr1 = t0 + (qr1 & 63), hr1 = h0 + (qr1 >> 6);
                                float4 q0 = make_float4(0,0,0,0), q1 = q0, q2 = q0, q3 = q0;
                                float sk1 = 0.0f;
                                if (tr1 < T) {
                                    const float4* rp = reinterpret_cast<const float4*>(R_ptr + ((size_t) tr1 * n_head + hr1) * 16);
                                    q0 = rp[0]; q1 = rp[1]; q2 = rp[2]; q3 = rp[3];
                                    sk1 = __ldg(&sink[(size_t) tr1 * n_head + hr1]);
                                }
#pragma unroll
                                for (int ct = 0; ct < 2; ++ct) {
                                    const int j0 = jb + kv_cbase + ct * 16 + acc_c0;
                                    const int j1 = jb + kv_cbase + ct * 16 + acc_c1;
                                    fc[m][ct].x[2] = fused_row_bias(qr1, q0, q1, q2, q3, sk1, fc[m][ct].x[2], j0);
                                    fc[m][ct].x[3] = fused_row_bias(qr1, q0, q1, q2, q3, sk1, fc[m][ct].x[3], j0 + 1);
                                    fc[m][ct].x[6] = fused_row_bias(qr1, q0, q1, q2, q3, sk1, fc[m][ct].x[6], j1);
                                    fc[m][ct].x[7] = fused_row_bias(qr1, q0, q1, q2, q3, sk1, fc[m][ct].x[7], j1 + 1);
                                    mx1 = fmaxf(mx1, fmaxf(fmaxf(fc[m][ct].x[2], fc[m][ct].x[3]), fmaxf(fc[m][ct].x[6], fc[m][ct].x[7])));
                                }
                            }
                            mx0 = wmma_row_max_16(mx0);
                            mx1 = wmma_row_max_16(mx1);
                            if (is_leader) {
                                *s_hmax_ptr(qr0, cg) = mx0;
                                *s_hmax_ptr(qr1, cg) = mx1;
                            }
                        }
                    }
                } else {
#pragma unroll
                    for (int m = 0; m < 3; ++m) {
                        const int qr0 = qr_base + m * 16 + acc_r0;
                        const int qr1 = qr_base + m * 16 + acc_r1;
                        float mx0 = -1e30f, mx1 = -1e30f;
                        {
                            const int tr0 = t0 + (qr0 & 63), hr0 = h0 + (qr0 >> 6);
                            float4 q0 = make_float4(0,0,0,0), q1 = q0, q2 = q0, q3 = q0;
                            float sk0 = 0.0f;
                            if (tr0 < T) {
                                const float4* rp = reinterpret_cast<const float4*>(R_ptr + ((size_t) tr0 * n_head + hr0) * 16);
                                q0 = rp[0]; q1 = rp[1]; q2 = rp[2]; q3 = rp[3];
                                sk0 = __ldg(&sink[(size_t) tr0 * n_head + hr0]);
                            }
#pragma unroll
                            for (int ct = 0; ct < 2; ++ct) {
                                const int j0 = jb + kv_cbase + ct * 16 + acc_c0;
                                const int j1 = jb + kv_cbase + ct * 16 + acc_c1;
                                fc[m][ct].x[0] = fused_row_bias(qr0, q0, q1, q2, q3, sk0, fc[m][ct].x[0], j0);
                                fc[m][ct].x[1] = fused_row_bias(qr0, q0, q1, q2, q3, sk0, fc[m][ct].x[1], j0 + 1);
                                fc[m][ct].x[4] = fused_row_bias(qr0, q0, q1, q2, q3, sk0, fc[m][ct].x[4], j1);
                                fc[m][ct].x[5] = fused_row_bias(qr0, q0, q1, q2, q3, sk0, fc[m][ct].x[5], j1 + 1);
                                mx0 = fmaxf(mx0, fmaxf(fmaxf(fc[m][ct].x[0], fc[m][ct].x[1]), fmaxf(fc[m][ct].x[4], fc[m][ct].x[5])));
                            }
                        }
                        {
                            const int tr1 = t0 + (qr1 & 63), hr1 = h0 + (qr1 >> 6);
                            float4 q0 = make_float4(0,0,0,0), q1 = q0, q2 = q0, q3 = q0;
                            float sk1 = 0.0f;
                            if (tr1 < T) {
                                const float4* rp = reinterpret_cast<const float4*>(R_ptr + ((size_t) tr1 * n_head + hr1) * 16);
                                q0 = rp[0]; q1 = rp[1]; q2 = rp[2]; q3 = rp[3];
                                sk1 = __ldg(&sink[(size_t) tr1 * n_head + hr1]);
                            }
#pragma unroll
                            for (int ct = 0; ct < 2; ++ct) {
                                const int j0 = jb + kv_cbase + ct * 16 + acc_c0;
                                const int j1 = jb + kv_cbase + ct * 16 + acc_c1;
                                fc[m][ct].x[2] = fused_row_bias(qr1, q0, q1, q2, q3, sk1, fc[m][ct].x[2], j0);
                                fc[m][ct].x[3] = fused_row_bias(qr1, q0, q1, q2, q3, sk1, fc[m][ct].x[3], j0 + 1);
                                fc[m][ct].x[6] = fused_row_bias(qr1, q0, q1, q2, q3, sk1, fc[m][ct].x[6], j1);
                                fc[m][ct].x[7] = fused_row_bias(qr1, q0, q1, q2, q3, sk1, fc[m][ct].x[7], j1 + 1);
                                mx1 = fmaxf(mx1, fmaxf(fmaxf(fc[m][ct].x[2], fc[m][ct].x[3]), fmaxf(fc[m][ct].x[6], fc[m][ct].x[7])));
                            }
                        }
                        mx0 = wmma_row_max_16(mx0);
                        mx1 = wmma_row_max_16(mx1);
                        if (is_leader) {
                            *s_hmax_ptr(qr0, cg) = mx0;
                            *s_hmax_ptr(qr1, cg) = mx1;
                        }
                    }
                }
            } else {
#pragma unroll
                for (int m = 0; m < 3; ++m) {
                    const int qr0 = qr_base + m * 16 + acc_r0;
                    const int qr1 = qr_base + m * 16 + acc_r1;
                    float mx0 = -1e30f, mx1 = -1e30f;
#pragma unroll
                    for (int ct = 0; ct < 2; ++ct) {
                        const int j0 = jb + kv_cbase + ct * 16 + acc_c0;
                        const int j1 = jb + kv_cbase + ct * 16 + acc_c1;
                        fc[m][ct].x[0] = mask_bias(fc[m][ct].x[0], qr0, j0);
                        fc[m][ct].x[1] = mask_bias(fc[m][ct].x[1], qr0, j0 + 1);
                        fc[m][ct].x[4] = mask_bias(fc[m][ct].x[4], qr0, j1);
                        fc[m][ct].x[5] = mask_bias(fc[m][ct].x[5], qr0, j1 + 1);
                        mx0 = fmaxf(mx0, fmaxf(fmaxf(fc[m][ct].x[0], fc[m][ct].x[1]), fmaxf(fc[m][ct].x[4], fc[m][ct].x[5])));

                        fc[m][ct].x[2] = mask_bias(fc[m][ct].x[2], qr1, j0);
                        fc[m][ct].x[3] = mask_bias(fc[m][ct].x[3], qr1, j0 + 1);
                        fc[m][ct].x[6] = mask_bias(fc[m][ct].x[6], qr1, j1);
                        fc[m][ct].x[7] = mask_bias(fc[m][ct].x[7], qr1, j1 + 1);
                        mx1 = fmaxf(mx1, fmaxf(fmaxf(fc[m][ct].x[2], fc[m][ct].x[3]), fmaxf(fc[m][ct].x[6], fc[m][ct].x[7])));
                    }
                    mx0 = wmma_row_max_16(mx0);
                    mx1 = wmma_row_max_16(mx1);
                    if (is_leader) {
                        *s_hmax_ptr(qr0, cg) = mx0;
                        *s_hmax_ptr(qr1, cg) = mx1;
                    }
                }
            }
            asm volatile("bar.sync %0, 64;" :: "r"(rg + 1) : "memory");

#pragma unroll
            for (int m = 0; m < 3; ++m) {
                const int qr0 = qr_base + m * 16 + acc_r0;
                const int qr1 = qr_base + m * 16 + acc_r1;
                const float mx0 = fmaxf(*s_hmax_ptr(qr0, 0), *s_hmax_ptr(qr0, 1));
                const float mx1 = fmaxf(*s_hmax_ptr(qr1, 0), *s_hmax_ptr(qr1, 1));
                const float m_new0 = fmaxf(rm0[m], mx0);
                const float m_new1 = fmaxf(rm1[m], mx1);
                const float a0 = (rm0[m] > -1e20f) ? ((m_new0 == rm0[m]) ? 1.0f : fast_exp2f_hw(rm0[m] - m_new0)) : 0.0f;
                const float a1 = (rm1[m] > -1e20f) ? ((m_new1 == rm1[m]) ? 1.0f : fast_exp2f_hw(rm1[m] - m_new1)) : 0.0f;
                if (__any_sync(0xffffffffu, a0 != 1.0f || a1 != 1.0f)) {
#pragma unroll
                    for (int n = 0; n < 4; ++n) wmma_rescale_acc(acc[m][n], a0, a1);
                }
                float psum0 = 0.0f, psum1 = 0.0f;
#pragma unroll
                for (int ct = 0; ct < 2; ++ct) {
                    const float p00 = (all_valid_swa || fc[m][ct].x[0] > -1e20f) ? fast_exp2f_hw(fc[m][ct].x[0] - m_new0) : 0.0f;
                    const float p01 = (all_valid_swa || fc[m][ct].x[1] > -1e20f) ? fast_exp2f_hw(fc[m][ct].x[1] - m_new0) : 0.0f;
                    const float p04 = (all_valid_swa || fc[m][ct].x[4] > -1e20f) ? fast_exp2f_hw(fc[m][ct].x[4] - m_new0) : 0.0f;
                    const float p05 = (all_valid_swa || fc[m][ct].x[5] > -1e20f) ? fast_exp2f_hw(fc[m][ct].x[5] - m_new0) : 0.0f;
                    psum0 += (p00 + p01) + (p04 + p05);
                    *reinterpret_cast<__half2*>(sP + qr0 * 72 + kv_cbase + ct * 16 + acc_c0) = __halves2half2(__float2half(p00), __float2half(p01));
                    *reinterpret_cast<__half2*>(sP + qr0 * 72 + kv_cbase + ct * 16 + acc_c1) = __halves2half2(__float2half(p04), __float2half(p05));

                    const float p12 = (all_valid_swa || fc[m][ct].x[2] > -1e20f) ? fast_exp2f_hw(fc[m][ct].x[2] - m_new1) : 0.0f;
                    const float p13 = (all_valid_swa || fc[m][ct].x[3] > -1e20f) ? fast_exp2f_hw(fc[m][ct].x[3] - m_new1) : 0.0f;
                    const float p16 = (all_valid_swa || fc[m][ct].x[6] > -1e20f) ? fast_exp2f_hw(fc[m][ct].x[6] - m_new1) : 0.0f;
                    const float p17 = (all_valid_swa || fc[m][ct].x[7] > -1e20f) ? fast_exp2f_hw(fc[m][ct].x[7] - m_new1) : 0.0f;
                    psum1 += (p12 + p13) + (p16 + p17);
                    *reinterpret_cast<__half2*>(sP + qr1 * 72 + kv_cbase + ct * 16 + acc_c0) = __halves2half2(__float2half(p12), __float2half(p13));
                    *reinterpret_cast<__half2*>(sP + qr1 * 72 + kv_cbase + ct * 16 + acc_c1) = __halves2half2(__float2half(p16), __float2half(p17));
                }
                psum0 = wmma_row_sum_16(psum0);
                psum1 = wmma_row_sum_16(psum1);
                rm0[m] = m_new0; rl0[m] = rl0[m] * a0 + psum0;
                rm1[m] = m_new1; rl1[m] = rl1[m] * a1 + psum1;
            }
            __syncthreads();

            {
                __half* dst = sKV + kv_r * 136 + kv_c0;
                if (valid_cur) {
                    const __half2 vbias = __hmul2(neg1152, dv2_t);
                    __half2 h2[8];
                    i8x4_to_h2x2_fma(vraw0.x, dv2_t, vbias, h2[0], h2[1]); i8x4_to_h2x2_fma(vraw0.y, dv2_t, vbias, h2[2], h2[3]);
                    i8x4_to_h2x2_fma(vraw0.z, dv2_t, vbias, h2[4], h2[5]); i8x4_to_h2x2_fma(vraw0.w, dv2_t, vbias, h2[6], h2[7]);
                    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(h2);
                    *reinterpret_cast<uint4*>(dst + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                    i8x4_to_h2x2_fma(vraw1.x, dv2_t, vbias, h2[0], h2[1]); i8x4_to_h2x2_fma(vraw1.y, dv2_t, vbias, h2[2], h2[3]);
                    i8x4_to_h2x2_fma(vraw1.z, dv2_t, vbias, h2[4], h2[5]); i8x4_to_h2x2_fma(vraw1.w, dv2_t, vbias, h2[6], h2[7]);
                    *reinterpret_cast<uint4*>(dst + 16) = *reinterpret_cast<const uint4*>(h2);
                    *reinterpret_cast<uint4*>(dst + 24) = *reinterpret_cast<const uint4*>(h2 + 4);
                } else {
                    const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
                    for (int i = 0; i < 4; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
                }
            }
            __syncthreads();

            // Advance slot_cur incrementally (zero % cap!)
            slot_cur += 64;
            if constexpr (IS_SWA) {
                if (slot_cur >= cap) slot_cur -= cap;
            }

            // D11: Hoist fa[3] out of n=0..3 loop (eliminates 75% redundant sP loads, zero stack spills!).
#pragma unroll 1
            for (int k = 0; k < 64; k += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[3];
#pragma unroll
                for (int m = 0; m < 3; ++m) {
                    wmma::load_matrix_sync(fa[m], sP + (qr_base + m * 16) * 72 + k, 72);
                }
#pragma unroll
                for (int n = 0; n < 4; ++n) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
                    wmma::load_matrix_sync(fb, sKV + k * 136 + pv_pc0 + n * 16, 136);
#pragma unroll
                    for (int m = 0; m < 3; ++m) {
                        wmma::mma_sync(acc[m][n], fa[m], fb, acc[m][n]);
                    }
                }
            }
            __syncthreads();
        }
    }

    if (is_leader) {
#pragma unroll
        for (int m = 0; m < 3; ++m) {
            const int qr0 = qr_base + m * 16 + acc_r0;
            const int qr1 = qr_base + m * 16 + acc_r1;
            if (cg == 0) {
                *s_l0_ptr(qr0) = rl0[m]; *s_l0_ptr(qr1) = rl1[m];
                if constexpr (SPLIT_KV) { *s_m_ptr(qr0) = rm0[m]; *s_m_ptr(qr1) = rm1[m]; }
            } else {
                *s_l1_ptr(qr0) = rl0[m]; *s_l1_ptr(qr1) = rl1[m];
            }
        }
    }
    __syncthreads();

    // Epilogue: 2 passes x 96 rows (96*128*4 = 49,152 B <= 52,224 B sQ), vectorized float4 reads/writes.
    float* s_out = reinterpret_cast<float*>(sQ);
    for (int g_rg = 0; g_rg < 4; g_rg += 2) {
        if (rg == g_rg || rg == g_rg + 1) {
            const int local_r0 = (rg - g_rg) * 48;
#pragma unroll
            for (int m = 0; m < 3; ++m) {
#pragma unroll
                for (int n = 0; n < 4; ++n) {
                    wmma::store_matrix_sync(s_out + (local_r0 + m * 16) * 128 + pv_pc0 + n * 16,
                                            acc[m][n], 128, wmma::mem_row_major);
                }
            }
        }
        __syncthreads();
        for (int idx4 = tid; idx4 < 96 * 32; idx4 += 256) {
            const int r = idx4 >> 5, col4 = idx4 & 31, col = col4 << 2;
            const int qr = g_rg * 48 + r;
            const int head_g = qr >> 6, tok_r = qr & 63, t = t0 + tok_r;
            if (t < T) {
                const float4 v = reinterpret_cast<const float4*>(s_out)[idx4];
                if constexpr (SPLIT_KV) {
                    float* dst = split_out + (((size_t) t * split_n + sp) * n_head + (h0 + head_g)) * 130;
                    *reinterpret_cast<float2*>(dst + col)     = make_float2(v.x, v.y);
                    *reinterpret_cast<float2*>(dst + col + 2) = make_float2(v.z, v.w);
                    if (col == 0) {
                        dst[128] = *s_m_ptr(qr);
                        dst[129] = *s_l0_ptr(qr) + *s_l1_ptr(qr);
                    }
                } else {
                    const float inv_l = __fdividef(1.0f, *s_l0_ptr(qr) + *s_l1_ptr(qr));
                    const size_t oi = ((size_t) t * n_head + (h0 + head_g)) * 128 + col;
                    float4 o = make_float4(v.x * inv_l, v.y * inv_l, v.z * inv_l, v.w * inv_l);
                    if (gate) {
                        const float4 g4 = *reinterpret_cast<const float4*>(gate + oi);
                        o.x *= sigmf(g4.x); o.y *= sigmf(g4.y); o.z *= sigmf(g4.z); o.w *= sigmf(g4.w);
                    }
                    if (out_f16) {
                        const __half2 p0 = __halves2half2(__float2half_rn(o.x), __float2half_rn(o.y));
                        const __half2 p1 = __halves2half2(__float2half_rn(o.z), __float2half_rn(o.w));
                        uint2 u;
                        memcpy(&u.x, &p0, 4);
                        memcpy(&u.y, &p1, 4);
                        *reinterpret_cast<uint2*>(out_f16 + oi) = u;
                    } else {
                        *reinterpret_cast<float4*>(out + oi) = o;
                    }
                }
            }
        }
        __syncthreads();
    }
#endif
}

/// Zero-sP, Zero-Spill (STACK:0), 12-warp (384-thr) Native HMMA.884 (Volta sm_70) / m16n8k16+ldmatrix (Ampere/Ada sm_80+)
/// FlashAttention-2 SWA & Full-Attn kernel (Wave 8A):
/// - 2-BAR split-phase per-warp-segmented 2KB cp.async.cg pipeline (3 BAR -> 2 BAR/tile, 0 s_kv_sc SMEM, fast no-wrap ptrs)
/// - Algebraic sink folding into initial rm0/rm1 (eliminates per-tile sink LDG + 32 FADD)
/// - Pre-split rel_bias_f16_hi/lo direct XOR-swizzled cp.async.cg (eliminates 112-insn split_ozaki_h2 per tile; clean FP32 fallback)
/// - Fused ct=0..3 exp2(ct+1) + P*V(ct) + overlapped acc rescale (24/32 MUFU.EX2 hidden under HMMA.16816.F32)
/// - De-clumped QK^T (double-buffered Q+K, unroll 2) & interleaved Ozaki HMMA/SHFL
/// - Single-BAR Prologue (early tile-0 K+rb cp.async, per-warp R_lo in sV) & 0-BAR per-warp Epilogue (55.3 KB <= 96.3 KB)
/// - Wave 8.5 Attention: 2-Half overlapped KV dequant (ki=4,6 / ct=2,3), 0 CALL/BRA.DIV, pinned HMMA+SHFL Ozaki,
///   de-clumped 1-LDSM/2-HMMA QK^T (peeled ki=7), unconditional per-n acc rescale in ct=0 + 8+8+8 ex2_pin,
///   precomputed A1/B1 pointers + PRESPLIT_RB constexpr purge, compact unroll-2 epilogue + hardware tanh sigmoid!
/// Grid: dim3(n_head_kv, (T + 63) / 64) [hkv-fast, reverse-qt], Block: 384, SMEM: 97,280 B (sm_80+) / 73,728 B (sm_70).
template <bool IS_SWA = true, bool PRESPLIT_RB = true, bool SWA_LINEAR_HM = false>
__global__ void __launch_bounds__(384, 1) k_attn_q8_gqa3_pp_fa2_192_swa_fast(
        const float* __restrict__ Q,
        const int8_t* __restrict__ kq, const __half* __restrict__ ks,
        const int8_t* __restrict__ vq, const __half* __restrict__ vs,
        const float* __restrict__ sink, const float* __restrict__ gate,
        float* __restrict__ out,
        int n_head, int n_head_kv, int n_ext, int cap, int n_swa,
        const int32_t* __restrict__ pos0, float kq_scale, int T,
        float* __restrict__ R_ptr, const float* __restrict__ rel_bias_ptr,
        uint16_t* __restrict__ out_f16,
        const float* __restrict__ q_scale = nullptr, float qk_clip = 0.0f,
        const float* __restrict__ r_scale = nullptr, float rel_clip = 0.0f,
        float tau_floor_n = 0.0f, float tau_alpha = 0.0f,
        int ld_q = 0, int ld_r = 0, int ld_gate = 0,
        const __half* __restrict__ rel_bias_f16_hi = nullptr,
        const __half* __restrict__ rel_bias_f16_lo = nullptr,
        const __half* __restrict__ gate_f16 = nullptr,
        int ld_sink = 0, int swa_hm_base = 0) {
#if defined(__CUDA_ARCH__) && ((__CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750) || __CUDA_ARCH__ >= 800)
#if __CUDA_ARCH__ < 750
    constexpr int STRIDE = 144;  // 73,728 B <= 96 KB on Volta sm_70 (sQ + sKV)
#else
    constexpr int STRIDE = 128;  // 97,024 B <= 97,280 B on Ampere/Ada sm_80+ (XOR-swizzled sQ + sK + sV + s_rb_hi + s_rb_lo + sR_hi + s_rl)
    constexpr bool kRelOzaki = true;
#endif
    extern __shared__ __align__(16) char smem_raw[];
    __half* sQ = reinterpret_cast<__half*>(smem_raw);
#if __CUDA_ARCH__ < 750
    __half* sKV = sQ + 192 * STRIDE;
#else
    __half* sK            = sQ + 192 * STRIDE;                         // +49,152 B (16,384 B: 8 per-warp 2KB slices for K)
    __half* sV            = sK + 64 * STRIDE;                          // +65,536 B (16,384 B: 8 per-warp 2KB slices for V; temp sR_lo in Prologue)
    __half* s_rb_hi       = sV + 64 * STRIDE;                          // +81,920 B (4,096 B: 128 x 16 FP16 hi)
    __half* s_rb_lo       = s_rb_hi + 128 * 16;                        // +86,016 B (4,096 B: 128 x 16 FP16 lo)
    __half* sR_hi_persist = s_rb_lo + 128 * 16;                        // +90,112 B (6,144 B: 192 x 16 FP16 hi)
    float*  s_rl          = reinterpret_cast<float*>(sR_hi_persist + 192 * 16); // +96,256 B (768 B -> 97,024 B <= 97,280 B!)
#endif

    const int hkv = (int) blockIdx.x;
    const int t0  = (int) (gridDim.y - 1 - blockIdx.y) * 64;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    constexpr float kLog2e = 1.4426950408889634f;
    const float qscale_all = kq_scale * kLog2e;

    // Precompute CTA-uniform p0_t0 in a uniform register (UR on sm_80+) so it costs 0 general regs across Phase 1 & Phase 2!
    const int p0_t0   = __shfl_sync(0xffffffffu, pos0[0] + t0, 0);
    const int j_start = IS_SWA ? max(0, p0_t0 - n_swa + 1) : 0;
    const int j_end   = p0_t0 + min(63, T - 1 - t0) + 1;
    const bool pow2_cap = IS_SWA && ((cap & (cap - 1)) == 0);
    const int cap_mask = cap - 1;
    const int q_stride = (ld_q > 0) ? ld_q : (n_head * 128);
    const int r_stride = (ld_r > 0) ? ld_r : (n_head * 16);
    const int sink_stride = ld_sink ? ld_sink : n_head;

    auto compute_tau_r = [&](int r_loc) -> float {
        if constexpr (!IS_SWA) {
            if (tau_alpha != 0.0f && tau_floor_n > 0.0f) {
                const float ratio = fmaxf(1.0f, (float) (p0_t0 + r_loc + 1) / tau_floor_n);
                return 1.0f + tau_alpha * logf(ratio);
            }
        }
        return 1.0f;
    };

#if __CUDA_ARCH__ < 750
    (void) rel_bias_f16_hi; (void) rel_bias_f16_lo; (void) swa_hm_base;
    // Prologue (sm_70): 384 threads load 192x128 Q -> sQ (with optional fused clip+tau) + 192 threads clip+tau R in-place.
#pragma unroll 1
    for (int base_r = 0; base_r < 192; base_r += 96) {
        const int qr = base_r + (tid >> 2);
        const int qc0 = (tid & 3) * 32;
        const int g = qr >> 6, r = qr & 63, t = t0 + r, h = 3 * hkv + g;
        __half2* dst2 = reinterpret_cast<__half2*>(sQ + qr * STRIDE + qc0);
        if (t < T) {
            const float* qp = Q + (size_t) t * q_stride + (size_t) h * 128 + qc0;
            if (q_scale != nullptr) {
                const float* qsp = q_scale + (size_t) h * 128 + qc0;
                float4 qv[8];
                float sq = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float4 v = *reinterpret_cast<const float4*>(qp + i * 4);
                    const float4 sc = __ldg(reinterpret_cast<const float4*>(qsp + i * 4));
                    qv[i] = make_float4(v.x * sc.x, v.y * sc.y, v.z * sc.z, v.w * sc.w);
                    sq += qv[i].x * qv[i].x + qv[i].y * qv[i].y + qv[i].z * qv[i].z + qv[i].w * qv[i].w;
                }
                const unsigned qmask = __activemask();
                sq += __shfl_xor_sync(qmask, sq, 1);
                sq += __shfl_xor_sync(qmask, sq, 2);
                const float fac = fminf(1.0f, qk_clip / (sqrtf(sq) + 1e-6f)) * compute_tau_r(r) * qscale_all;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    dst2[i * 2 + 0] = __halves2half2(__float2half(qv[i].x * fac), __float2half(qv[i].y * fac));
                    dst2[i * 2 + 1] = __halves2half2(__float2half(qv[i].z * fac), __float2half(qv[i].w * fac));
                }
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float4 v = *reinterpret_cast<const float4*>(qp + i * 4);
                    dst2[i * 2 + 0] = __halves2half2(__float2half(v.x * qscale_all), __float2half(v.y * qscale_all));
                    dst2[i * 2 + 1] = __halves2half2(__float2half(v.z * qscale_all), __float2half(v.w * qscale_all));
                }
            }
        } else {
            const __half2 z2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));
#pragma unroll
            for (int i = 0; i < 16; ++i) dst2[i] = z2;
        }
        if ((tid & 3) == 0) {
            float* s_ml = reinterpret_cast<float*>(sQ + qr * STRIDE + 128);
            s_ml[0] = 0.0f;
            s_ml[1] = 1.0f;
        }
    }
    if (r_scale != nullptr && R_ptr != nullptr && tid < 192) {
        const int g = tid >> 6, r = tid & 63, t = t0 + r, h = 3 * hkv + g;
        if (t < T) {
            float4* rp = reinterpret_cast<float4*>(R_ptr + (size_t) t * r_stride + (size_t) h * 16);
            const float4* rsp = reinterpret_cast<const float4*>(r_scale + (size_t) h * 16);
            float4 r0 = rp[0], r1 = rp[1], r2 = rp[2], r3 = rp[3];
            const float4 s0 = __ldg(rsp + 0), s1 = __ldg(rsp + 1), s2 = __ldg(rsp + 2), s3 = __ldg(rsp + 3);
            r0.x *= s0.x; r0.y *= s0.y; r0.z *= s0.z; r0.w *= s0.w;
            r1.x *= s1.x; r1.y *= s1.y; r1.z *= s1.z; r1.w *= s1.w;
            r2.x *= s2.x; r2.y *= s2.y; r2.z *= s2.z; r2.w *= s2.w;
            r3.x *= s3.x; r3.y *= s3.y; r3.z *= s3.z; r3.w *= s3.w;
            const float rsq = (r0.x * r0.x + r0.y * r0.y + r0.z * r0.z + r0.w * r0.w)
                            + (r1.x * r1.x + r1.y * r1.y + r1.z * r1.z + r1.w * r1.w)
                            + (r2.x * r2.x + r2.y * r2.y + r2.z * r2.z + r2.w * r2.w)
                            + (r3.x * r3.x + r3.y * r3.y + r3.z * r3.z + r3.w * r3.w);
            const float rfac = fminf(1.0f, rel_clip / (sqrtf(rsq) + 1e-6f)) * compute_tau_r(r) * kLog2e;
            rp[0] = make_float4(r0.x * rfac, r0.y * rfac, r0.z * rfac, r0.w * rfac);
            rp[1] = make_float4(r1.x * rfac, r1.y * rfac, r1.z * rfac, r1.w * rfac);
            rp[2] = make_float4(r2.x * rfac, r2.y * rfac, r2.z * rfac, r2.w * rfac);
            rp[3] = make_float4(r3.x * rfac, r3.y * rfac, r3.z * rfac, r3.w * rfac);
        }
    }
    __syncthreads();

    auto mma_rc = [](float d[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, {%0, %1, %2, %3, %4, %5, %6, %7};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
            : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
    };
    auto mma_rr = [](float d[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
            "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, {%0, %1, %2, %3, %4, %5, %6, %7};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
            : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
    };

    float acc[8][8];
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[n][i] = 0.0f;

    int slot_base = IS_SWA ? (pow2_cap ? (j_start & cap_mask) : (j_start % cap)) : 0;
    const int8_t* kq_full_h = IS_SWA ? nullptr : (kq + (size_t) hkv * cap * 128);
    const __half* ks_full_h = IS_SWA ? nullptr : (ks + (size_t) hkv * cap);
    const int8_t* vq_full_h = IS_SWA ? nullptr : (vq + (size_t) hkv * cap * 128);
    const __half* vs_full_h = IS_SWA ? nullptr : (vs + (size_t) hkv * cap);
    for (int jb = j_start; jb < j_end; jb += 64) {
        // 1. Load & dequant K (64x128) -> sKV (single-shot 256 threads!).
        if (tid < 256) {
            const int kr = tid >> 2, kc0 = (tid & 3) * 32;
            const int j = jb + kr;
            __half* dst = sKV + kr * STRIDE + kc0;
            if (j < j_end) {
                __half2 dk2;
                const int8_t* kp;
                if constexpr (IS_SWA) {
                    int sl = slot_base + kr;
                    if (sl >= cap) sl -= cap;
                    dk2 = __half2half2(__ldg(&ks[(size_t) sl * n_head_kv + hkv]));
                    kp = kq + ((size_t) sl * n_head_kv + hkv) * 128 + kc0;
                } else {
                    dk2 = __half2half2(__ldg(&ks_full_h[j]));
                    kp = kq_full_h + (size_t) j * 128 + kc0;
                }
                const __half2 kb = __hmul2(__half2half2(__float2half(-1152.0f)), dk2);
#pragma unroll
                for (int blk = 0; blk < 2; ++blk) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(kp + blk * 16);
                    __half2 h0, h1, h2v, h3;
                    i8x4_to_h2x2_fma(raw.x, dk2, kb, h0, h1);
                    i8x4_to_h2x2_fma(raw.y, dk2, kb, h2v, h3);
                    __half2* d2 = reinterpret_cast<__half2*>(dst + blk * 16);
                    d2[0] = h0; d2[1] = h1; d2[2] = h2v; d2[3] = h3;
                    i8x4_to_h2x2_fma(raw.z, dk2, kb, h0, h1);
                    i8x4_to_h2x2_fma(raw.w, dk2, kb, h2v, h3);
                    d2[4] = h0; d2[5] = h1; d2[6] = h2v; d2[7] = h3;
                }
            } else {
                const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int i = 0; i < 4; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
            }
        }
        __syncthreads();

        // 2. Native QK^T (1x4 = 16x64 per warp).
        float fc[4][8];
#pragma unroll
        for (int ct = 0; ct < 4; ++ct)
#pragma unroll
            for (int i = 0; i < 8; ++i) fc[ct][i] = 0.0f;

        const int sq_off = (warp * 16 + ((lane & 16) >> 2) + ((lane & 4) << 1) + (lane & 3)) * STRIDE;
        const int sk_off = (((lane & 16) >> 2) + (lane & 8) + (lane & 3)) * STRIDE;
#pragma unroll 1
        for (int k = 0; k < 128; k += 8) {
            const uint4 q0 = *reinterpret_cast<const uint4*>(sQ + sq_off + k);
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) {
                const uint4 kb = *reinterpret_cast<const uint4*>(sKV + sk_off + (ct * 16) * STRIDE + k);
                mma_rc(fc[ct], q0.x, q0.y, kb.x, kb.y);
                mma_rc(fc[ct], q0.z, q0.w, kb.z, kb.w);
            }
        }
        __syncthreads();

        // 3. Stage rel_bias[d_lo..d_hi][16] into idle sKV (2-step unrolled, skipped for bulk Full-Attn tiles).
        const int d_lo = max(0, p0_t0 - (jb + 63));
        const int d_hi = min(n_ext - 1, (p0_t0 + 63) - jb);
        const int d_cnt = max(0, d_hi - d_lo + 1);
        const bool all_bulk_no_rel = (!IS_SWA) && (t0 + 63 < T) && (jb + 63 <= p0_t0 - n_ext);
        float* s_rb = reinterpret_cast<float*>(sKV);
        if (!all_bulk_no_rel && d_cnt > 0) {
            const int lim = d_cnt * 4;
            if (tid < lim) {
                const int di = tid >> 2, c4 = tid & 3;
                *reinterpret_cast<float4*>(s_rb + di * 20 + c4 * 4) =
                    reinterpret_cast<const float4*>(rel_bias_ptr + (size_t) (d_lo + di) * 16)[c4];
            }
            if (tid + 384 < lim) {
                const int idx = tid + 384;
                const int di = idx >> 2, c4 = idx & 3;
                *reinterpret_cast<float4*>(s_rb + di * 20 + c4 * 4) =
                    reinterpret_cast<const float4*>(rel_bias_ptr + (size_t) (d_lo + di) * 16)[c4];
            }
            __syncthreads();
        }

        // 4. Register-resident bias + online softmax + in-place packed P in fc[ct][0..3] (ZERO ph[], ZERO sP, STACK:0!).
        {
            int acc_r0, acc_r1, acc_c0, acc_c1;
            wmma_acc_layout_16x16(lane, acc_r0, acc_r1, acc_c0, acc_c1);
            const bool all_valid_swa = (t0 + 63 < T) &&
                                       (!IS_SWA || jb >= (p0_t0 + 63) - n_swa + 1) && (jb + 63 <= p0_t0) &&
                                       ((p0_t0 + 63) - jb < n_ext);
            const bool all_unmasked = all_valid_swa || all_bulk_no_rel;
            auto eval_row = [&](int qr, int i0, int i1, int i4, int i5) {
                const int r_loc = qr & 63;
                const int tr = t0 + r_loc, hr = 3 * hkv + (qr >> 6);
                const int pos_q = p0_t0 + r_loc;
                const float neg_sk = (all_unmasked || tr < T) ? (-__ldg(&sink[(size_t) tr * sink_stride + hr]) * kLog2e) : 0.0f;
                if (all_bulk_no_rel) {
#pragma unroll
                    for (int ct = 0; ct < 4; ++ct) {
                        fc[ct][i0] += neg_sk; fc[ct][i1] += neg_sk;
                        fc[ct][i4] += neg_sk; fc[ct][i5] += neg_sk;
                    }
                    return;
                }
                auto init_one = [&](float s, int j) -> float {
                    if (!all_valid_swa) {
                        const int d = pos_q - j;
                        if (tr >= T || j > pos_q || (IS_SWA && d >= n_swa)) return -1e30f;
                    }
                    return s + neg_sk;
                };
#pragma unroll
                for (int ct = 0; ct < 4; ++ct) {
                    const int j0 = jb + ct * 16 + acc_c0, j1 = jb + ct * 16 + acc_c1;
                    fc[ct][i0] = init_one(fc[ct][i0], j0);
                    fc[ct][i1] = init_one(fc[ct][i1], j0 + 1);
                    fc[ct][i4] = init_one(fc[ct][i4], j1);
                    fc[ct][i5] = init_one(fc[ct][i5], j1 + 1);
                }
                if (d_cnt > 0 && (all_valid_swa || tr < T)) {
                    const float4* rp = reinterpret_cast<const float4*>(R_ptr + (size_t) tr * r_stride + (size_t) hr * 16);
                    const float q_fac = (r_scale == nullptr) ? kLog2e : 1.0f;
                    auto add_dot16 = [&](float s, int j) -> float {
                        const int d = pos_q - j;
                        if (!all_valid_swa && (j > pos_q || (IS_SWA && d >= n_swa) || d >= n_ext)) return s;
                        const float4* bp = reinterpret_cast<const float4*>(s_rb + (d - d_lo) * 20);
                        float dot = 0.0f;
                        {
                            const float4 q0 = rp[0], b0 = bp[0];
                            dot = (q0.x * q_fac) * b0.x + (q0.y * q_fac) * b0.y + (q0.z * q_fac) * b0.z + (q0.w * q_fac) * b0.w;
                        }
#pragma unroll
                        for (int rk = 1; rk < 4; ++rk) {
                            const float4 qk = rp[rk], bk = bp[rk];
                            dot = fmaf(qk.x * q_fac, bk.x, fmaf(qk.y * q_fac, bk.y, fmaf(qk.z * q_fac, bk.z, fmaf(qk.w * q_fac, bk.w, dot))));
                        }
                        return s + dot;
                    };
#pragma unroll
                    for (int ct = 0; ct < 4; ++ct) {
                        const int j0 = jb + ct * 16 + acc_c0, j1 = jb + ct * 16 + acc_c1;
                        fc[ct][i0] = add_dot16(fc[ct][i0], j0);
                        fc[ct][i1] = add_dot16(fc[ct][i1], j0 + 1);
                        fc[ct][i4] = add_dot16(fc[ct][i4], j1);
                        fc[ct][i5] = add_dot16(fc[ct][i5], j1 + 1);
                    }
                }
            };
#pragma unroll 1
            for (int row_i = 0; row_i < 2; ++row_i) {
                if (row_i == 0) eval_row(warp * 16 + acc_r0, 0, 1, 4, 5);
                else            eval_row(warp * 16 + acc_r1, 2, 3, 6, 7);
            }
            float mx0 = -1e30f, mx1 = -1e30f;
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) {
                mx0 = fmaxf(mx0, fmaxf(fmaxf(fc[ct][0], fc[ct][1]), fmaxf(fc[ct][4], fc[ct][5])));
                mx1 = fmaxf(mx1, fmaxf(fmaxf(fc[ct][2], fc[ct][3]), fmaxf(fc[ct][6], fc[ct][7])));
            }
            float* s_ml0 = reinterpret_cast<float*>(sQ + (warp * 16 + acc_r0) * STRIDE + 128);
            float* s_ml1 = reinterpret_cast<float*>(sQ + (warp * 16 + acc_r1) * STRIDE + 128);
            const float rm0 = s_ml0[0], rm1 = s_ml1[0];
            const float m_new0 = fmaxf(rm0, wmma_row_max_16(mx0));
            const float m_new1 = fmaxf(rm1, wmma_row_max_16(mx1));
            const float a0 = (m_new0 == rm0) ? 1.0f : fast_exp2f_hw(rm0 - m_new0);
            const float a1 = (m_new1 == rm1) ? 1.0f : fast_exp2f_hw(rm1 - m_new1);
            if (__any_sync(0xffffffffu, a0 != 1.0f || a1 != 1.0f)) {
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    acc[n][0] *= a0; acc[n][1] *= a0; acc[n][4] *= a0; acc[n][5] *= a0;
                    acc[n][2] *= a1; acc[n][3] *= a1; acc[n][6] *= a1; acc[n][7] *= a1;
                }
            }
            float ps0 = 0.0f, ps1 = 0.0f;
            const bool want_r1 = (lane & 2) != 0;
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) {
                {
                    const float p00 = (all_unmasked || fc[ct][0] > -1e20f) ? fast_exp2f_hw(fc[ct][0] - m_new0) : 0.0f;
                    const float p01 = (all_unmasked || fc[ct][1] > -1e20f) ? fast_exp2f_hw(fc[ct][1] - m_new0) : 0.0f;
                    const float p12 = (all_unmasked || fc[ct][2] > -1e20f) ? fast_exp2f_hw(fc[ct][2] - m_new1) : 0.0f;
                    const float p13 = (all_unmasked || fc[ct][3] > -1e20f) ? fast_exp2f_hw(fc[ct][3] - m_new1) : 0.0f;
                    ps0 += p00 + p01;
                    ps1 += p12 + p13;
                    const uint32_t u0 = f2_to_h2u(p00, p01), v0 = f2_to_h2u(p12, p13);
                    const uint32_t r0 = __shfl_xor_sync(0xffffffffu, want_r1 ? u0 : v0, 2);
                    fc[ct][0] = __uint_as_float(want_r1 ? r0 : u0);
                    fc[ct][1] = __uint_as_float(want_r1 ? v0 : r0);
                }
                {
                    const float p04 = (all_unmasked || fc[ct][4] > -1e20f) ? fast_exp2f_hw(fc[ct][4] - m_new0) : 0.0f;
                    const float p05 = (all_unmasked || fc[ct][5] > -1e20f) ? fast_exp2f_hw(fc[ct][5] - m_new0) : 0.0f;
                    const float p16 = (all_unmasked || fc[ct][6] > -1e20f) ? fast_exp2f_hw(fc[ct][6] - m_new1) : 0.0f;
                    const float p17 = (all_unmasked || fc[ct][7] > -1e20f) ? fast_exp2f_hw(fc[ct][7] - m_new1) : 0.0f;
                    ps0 += p04 + p05;
                    ps1 += p16 + p17;
                    const uint32_t u1 = f2_to_h2u(p04, p05), v1 = f2_to_h2u(p16, p17);
                    const uint32_t r1 = __shfl_xor_sync(0xffffffffu, want_r1 ? u1 : v1, 2);
                    fc[ct][2] = __uint_as_float(want_r1 ? r1 : u1);
                    fc[ct][3] = __uint_as_float(want_r1 ? v1 : r1);
                }
            }
            const float rl_new0 = s_ml0[1] * a0 + wmma_row_sum_16(ps0);
            const float rl_new1 = s_ml1[1] * a1 + wmma_row_sum_16(ps1);
            __syncwarp();
            if (wmma_row_leader_16(lane)) {
                s_ml0[0] = m_new0; s_ml0[1] = rl_new0;
                s_ml1[0] = m_new1; s_ml1[1] = rl_new1;
            }
        }
        __syncthreads();

        // 5. Load & dequant V (64x128) -> sKV (single-shot 256 threads!).
        if (tid < 256) {
            const int kr = tid >> 2, kc0 = (tid & 3) * 32;
            const int j = jb + kr;
            __half* dst = sKV + kr * STRIDE + kc0;
            if (j < j_end) {
                __half2 dv2;
                const int8_t* vp;
                if constexpr (IS_SWA) {
                    int sl = slot_base + kr;
                    if (sl >= cap) sl -= cap;
                    dv2 = __half2half2(__ldg(&vs[(size_t) sl * n_head_kv + hkv]));
                    vp = vq + ((size_t) sl * n_head_kv + hkv) * 128 + kc0;
                } else {
                    dv2 = __half2half2(__ldg(&vs_full_h[j]));
                    vp = vq_full_h + (size_t) j * 128 + kc0;
                }
                const __half2 vb = __hmul2(__half2half2(__float2half(-1152.0f)), dv2);
#pragma unroll
                for (int blk = 0; blk < 2; ++blk) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(vp + blk * 16);
                    __half2 h0, h1, h2v, h3;
                    i8x4_to_h2x2_fma(raw.x, dv2, vb, h0, h1);
                    i8x4_to_h2x2_fma(raw.y, dv2, vb, h2v, h3);
                    __half2* d2 = reinterpret_cast<__half2*>(dst + blk * 16);
                    d2[0] = h0; d2[1] = h1; d2[2] = h2v; d2[3] = h3;
                    i8x4_to_h2x2_fma(raw.z, dv2, vb, h0, h1);
                    i8x4_to_h2x2_fma(raw.w, dv2, vb, h2v, h3);
                    d2[4] = h0; d2[5] = h1; d2[6] = h2v; d2[7] = h3;
                }
            } else {
                const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int i = 0; i < 4; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
            }
        }
        __syncthreads();

        if constexpr (IS_SWA && !SWA_LINEAR_HM) {
            slot_base += 64;
            if (slot_base >= cap) slot_base -= cap;
        }

        // 6. Native P*V (1x8 = 16x128 per warp).
        const bool hi8 = (lane & 8) != 0;
        const int sv_off = (lane & 3) * STRIDE + (lane & 8) + ((lane & 16) >> 2);
#pragma unroll
        for (int ct = 0; ct < 4; ++ct) {
            const __half* sv_base = sKV + sv_off + (ct * 16) * STRIDE;
            {
                const uint32_t p0 = __float_as_uint(fc[ct][0]), p1 = __float_as_uint(fc[ct][1]);
                const uint32_t o0 = __shfl_xor_sync(0xffffffffu, p0, 8);
                const uint32_t o1 = __shfl_xor_sync(0xffffffffu, p1, 8);
                const uint32_t a0 = hi8 ? o0 : p0, a1 = hi8 ? o1 : p1;
                const uint32_t a4 = hi8 ? p0 : o0, a5 = hi8 ? p1 : o1;
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    const uint2 b = *reinterpret_cast<const uint2*>(sv_base + 0 * STRIDE + n * 16);
                    mma_rr(acc[n], a0, a1, b.x, b.y);
                }
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    const uint2 b = *reinterpret_cast<const uint2*>(sv_base + 8 * STRIDE + n * 16);
                    mma_rr(acc[n], a4, a5, b.x, b.y);
                }
            }
            {
                const uint32_t p2 = __float_as_uint(fc[ct][2]), p3 = __float_as_uint(fc[ct][3]);
                const uint32_t o2 = __shfl_xor_sync(0xffffffffu, p2, 8);
                const uint32_t o3 = __shfl_xor_sync(0xffffffffu, p3, 8);
                const uint32_t a2 = hi8 ? o2 : p2, a3 = hi8 ? o3 : p3;
                const uint32_t a6 = hi8 ? p2 : o2, a7 = hi8 ? p3 : o3;
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    const uint2 b = *reinterpret_cast<const uint2*>(sv_base + 4 * STRIDE + n * 16);
                    mma_rr(acc[n], a2, a3, b.x, b.y);
                }
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    const uint2 b = *reinterpret_cast<const uint2*>(sv_base + 12 * STRIDE + n * 16);
                    mma_rr(acc[n], a6, a7, b.x, b.y);
                }
            }
        }
        __syncthreads();
    }

    // Epilogue (sm_70): copy s_rl from sQ padding into sKV (768 B), reuse sQ for 2 passes x 96 rows.
    int acc_r0, acc_r1, acc_c0, acc_c1;
    wmma_acc_layout_16x16(lane, acc_r0, acc_r1, acc_c0, acc_c1);
    float* s_rl = reinterpret_cast<float*>(sKV);
    if (tid < 192) {
        s_rl[tid] = reinterpret_cast<const float*>(sQ + tid * STRIDE + 128)[1];
    }
    __syncthreads();

    const int gate_stride = (ld_gate > 0) ? ld_gate : (n_head * 128);
    float* s_out = reinterpret_cast<float*>(sQ);
    for (int w_base = 0; w_base < 12; w_base += 6) {
        if (warp >= w_base && warp < w_base + 6) {
            const int lr0 = (warp - w_base) * 16;
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                float* dst0 = s_out + (lr0 + acc_r0) * 128 + n * 16;
                float* dst1 = s_out + (lr0 + acc_r1) * 128 + n * 16;
                *reinterpret_cast<float2*>(dst0 + acc_c0) = make_float2(acc[n][0], acc[n][1]);
                *reinterpret_cast<float2*>(dst0 + acc_c1) = make_float2(acc[n][4], acc[n][5]);
                *reinterpret_cast<float2*>(dst1 + acc_c0) = make_float2(acc[n][2], acc[n][3]);
                *reinterpret_cast<float2*>(dst1 + acc_c1) = make_float2(acc[n][6], acc[n][7]);
            }
        }
        __syncthreads();
        for (int idx4 = tid; idx4 < 96 * 32; idx4 += 384) {
            const int r = idx4 >> 5, col = (idx4 & 31) << 2;
            const int qr = w_base * 16 + r;
            const int head_g = qr >> 6, tok_r = qr & 63, t = t0 + tok_r;
            if (t < T) {
                const float4 v = reinterpret_cast<const float4*>(s_out)[idx4];
                const float inv_l = __fdividef(1.0f, s_rl[qr]);
                const size_t oi = ((size_t) t * n_head + (3 * hkv + head_g)) * 128 + col;
                float4 o = make_float4(v.x * inv_l, v.y * inv_l, v.z * inv_l, v.w * inv_l);
                if (gate_f16 != nullptr || gate != nullptr) {
                    const size_t gi = (size_t) t * gate_stride + (size_t) (3 * hkv + head_g) * 128 + col;
                    float4 g4;
                    if (gate_f16 != nullptr) {
                        const uint2 g2u = *reinterpret_cast<const uint2*>(gate_f16 + gi);
                        __half2 gh0, gh1;
                        memcpy(&gh0, &g2u.x, 4);
                        memcpy(&gh1, &g2u.y, 4);
                        const float2 gf0 = __half22float2(gh0);
                        const float2 gf1 = __half22float2(gh1);
                        g4 = make_float4(gf0.x, gf0.y, gf1.x, gf1.y);
                    } else {
                        g4 = *reinterpret_cast<const float4*>(gate + gi);
                    }
                    o.x *= sigmf(g4.x); o.y *= sigmf(g4.y); o.z *= sigmf(g4.z); o.w *= sigmf(g4.w);
                }
                if (out_f16) {
                    const __half2 p0h = __halves2half2(__float2half_rn(o.x), __float2half_rn(o.y));
                    const __half2 p1h = __halves2half2(__float2half_rn(o.z), __float2half_rn(o.w));
                    uint2 u;
                    memcpy(&u.x, &p0h, 4);
                    memcpy(&u.y, &p1h, 4);
                    *reinterpret_cast<uint2*>(out_f16 + oi) = u;
                } else {
                    *reinterpret_cast<float4*>(out + oi) = o;
                }
            }
        }
        __syncthreads();
    }
#else
    // Ampere / Ada (sm_80+): Wave 8.5 — 97,024 B SMEM, 2-Half overlapped KV dequant (0 CALL/BRA.DIV),
    // pinned HMMA+SHFL Ozaki, de-clumped 1-LDSM/2-HMMA QK^T (peeled ki=7), unconditional per-n acc rescale in ct=0
    // + 8+8+8 ex2_pin, precomputed A1/B1 pointers + PRESPLIT_RB constexpr purge, compact unroll-2 epilogue + tanh sigmoid!
    auto ldmat_x4 = [](uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
    };
    auto ldmat_x4_pin = [](uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr) : "memory");
    };
    auto ldmat_x4_t_pin = [](uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
        asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr) : "memory");
    };
    auto mma_16816_pin = [](float d[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1) : "memory");
    };
    auto shfl_idx_u32_pin = [](uint32_t v, int src_lane) -> uint32_t {
        uint32_t r;
        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;\n"
                     : "=r"(r) : "r"(v), "r"(src_lane) : "memory");
        return r;
    };
    auto ex2_pin = [](float x) -> float {
        float r;
        asm volatile("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x) : "memory");
        return r;
    };
    auto cp_async_cg_16 = [](uint32_t sa, const void* ga, int src_bytes) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :: "r"(sa), "l"(ga), "r"(src_bytes) : "memory");
    };
    auto cp_async_cg_16_full = [](uint32_t sa, const void* ga) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :: "r"(sa), "l"(ga) : "memory");
    };
    auto cp_async_commit = []() {
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    };
    auto cp_async_wait0 = []() {
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    };
    auto lds_u4 = [](uint32_t sa) -> uint4 {
        uint4 v;
        asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(sa) : "memory");
        return v;
    };
    auto sts_u4 = [](uint32_t sa, uint4 v) {
        asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n"
                     :: "r"(sa), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
    };
    auto fadd_pin = [](float a, float b) -> float {
        float r;
        asm volatile("add.f32 %0, %1, %2;\n" : "=f"(r) : "f"(a), "f"(b) : "memory");
        return r;
    };
    auto split_ozaki_h2 = [](float a, float b, __half2& hi, __half2& lo) {
        const __half ah = __float2half_rn(a), bh = __float2half_rn(b);
        hi = __halves2half2(ah, bh);
        lo = __halves2half2(__float2half_rn((a - __half2float(ah)) * 2048.0f),
                            __float2half_rn((b - __half2float(bh)) * 2048.0f));
    };
    auto pack_h2x4 = [](__half2 h0, __half2 h1, __half2 h2v, __half2 h3) -> uint4 {
        uint4 u;
        memcpy(&u.x, &h0, 4); memcpy(&u.y, &h1, 4);
        memcpy(&u.z, &h2v, 4); memcpy(&u.w, &h3, 4);
        return u;
    };

    const bool has_rel = (R_ptr != nullptr) && (PRESPLIT_RB ? (rel_bias_f16_hi != nullptr && rel_bias_f16_lo != nullptr)
                                                            : (rel_bias_ptr != nullptr));
    int slot_base = (IS_SWA && !SWA_LINEAR_HM) ? (pow2_cap ? (j_start & cap_mask) : (j_start % cap)) : 0;

    // Precompute warp-uniform/thread-invariant SMEM addresses outside jb loop (saves ~50 IMAD/LOP3/SHF/tile, STACK: 0!).
    const bool is_rb_warp = (__shfl_sync(0xffffffffu, warp, 0) >= 8);
    const bool is_kv_warp = !is_rb_warp;
    const int kv_r_sub = lane >> 3;
    const uint32_t sa_k0 = is_kv_warp
        ? static_cast<uint32_t>(__cvta_generic_to_shared(
              reinterpret_cast<int8_t*>(sK + (warp * 8 + 4) * STRIDE) + kv_r_sub * 128 + (lane & 7) * 16))
        : 0u;
    const uint32_t sa_kdst0 = is_kv_warp
        ? (static_cast<uint32_t>(__cvta_generic_to_shared(
               sK + (warp * 8 + kv_r_sub) * STRIDE + ((((lane & 7) * 2) ^ kv_r_sub) * 8))) ^ ((lane & 4) ? 16u : 0u))
        : 0u;
    const uint32_t sa_kdst1 = is_kv_warp
        ? (static_cast<uint32_t>(__cvta_generic_to_shared(
               sK + (warp * 8 + 4 + kv_r_sub) * STRIDE + ((((lane & 7) * 2) ^ (4 + kv_r_sub)) * 8))) ^ ((lane & 4) ? 16u : 0u))
        : 0u;

    // Branchless 8-warp segmented cp.async.cg launcher into SECOND HALF of each warp's exclusive 2KB slice (+1024B).
    auto launch_kv_async = [&](bool is_v, int jb_pf, int sb_pf, __half& sc_out) {
        if (is_kv_warp) {
            const int kv_r0 = warp * 8;
            const int r_sub = lane >> 3;
            const int c16   = (lane & 7) * 16;
            const uint32_t sa0 = is_v ? (sa_k0 + (uint32_t) (64 * STRIDE * sizeof(__half))) : sa_k0;
            const uint32_t sa1 = sa0 + 512u;
            const int8_t* q_raw = is_v ? vq : kq;
            const __half* s_raw = is_v ? vs : ks;
            const int j0 = jb_pf + kv_r0 + r_sub;
            const int j1 = j0 + 4;
            if constexpr (IS_SWA && !SWA_LINEAR_HM) {
                const uint32_t row_st = (uint32_t) n_head_kv * 128u;
                const int8_t* p_b = q_raw + (size_t) hkv * 128u + c16;
                const __half* s_b = s_raw + hkv;
                const bool ok0 = (j0 < j_end);
                const bool ok1 = (j1 < j_end);
                int sl0 = sb_pf + kv_r0 + r_sub;
                if (sl0 >= cap) sl0 -= cap;
                int sl1 = sl0 + 4;
                if (sl1 >= cap) sl1 -= cap;
                cp_async_cg_16(sa0, p_b + (size_t) (ok0 ? sl0 : 0) * row_st, ok0 ? 16 : 0);
                cp_async_cg_16(sa1, p_b + (size_t) (ok1 ? sl1 : 0) * row_st, ok1 ? 16 : 0);
                if (lane < 8) {
                    const int js = jb_pf + kv_r0 + lane;
                    int sls = sb_pf + kv_r0 + lane;
                    if (sls >= cap) sls -= cap;
                    const bool js_ok = (js < j_end);
                    const __half sv = __ldg(&s_b[(size_t) (js_ok ? sls : 0) * (size_t) n_head_kv]);
                    sc_out = js_ok ? sv : __float2half(0.0f);
                }
            } else {
                const int j_bias = SWA_LINEAR_HM ? swa_hm_base : 0;
                const size_t full_off = (size_t) hkv * (size_t) cap;
                const int8_t* p_b = q_raw + full_off * 128u + c16;
                if (jb_pf + 64 <= j_end) {
                    // CTA-uniform branchless bulk fast-path (all 64 KV rows strictly < j_end).
                    const int8_t* gp0 = p_b + (size_t) (j0 - j_bias) * 128u;
                    cp_async_cg_16_full(sa0, gp0);
                    cp_async_cg_16_full(sa1, gp0 + 512u);
                    if (lane < 8) {
                        sc_out = __ldg(&s_raw[full_off + (size_t) (jb_pf - j_bias + kv_r0 + lane)]);
                    }
                } else {
                    const bool ok0 = (j0 < j_end);
                    const bool ok1 = (j1 < j_end);
                    cp_async_cg_16(sa0, p_b + (size_t) (ok0 ? (j0 - j_bias) : 0) * 128u, ok0 ? 16 : 0);
                    cp_async_cg_16(sa1, p_b + (size_t) (ok1 ? (j1 - j_bias) : 0) * 128u, ok1 ? 16 : 0);
                    if (lane < 8) {
                        const int js = jb_pf + kv_r0 + lane;
                        const bool js_ok = (js < j_end);
                        const __half sv = __ldg(&s_raw[full_off + (size_t) (js_ok ? (js - j_bias) : 0)]);
                        sc_out = js_ok ? sv : __float2half(0.0f);
                    }
                }
            }
            cp_async_commit();
        }
    };

    // Warps 8..11 rel_bias launcher: PRESPLIT_RB constexpr + compact unroll-1 (<=32KB I-cache, 0 live loop regs!).
    auto launch_rb_async = [&](int jb_pf) {
        if (is_rb_warp) {
            const int rb_i = tid - 256;
            const int rb_r = PRESPLIT_RB ? ((rb_i & 63) >> 1) : (rb_i >> 2);
            const int d_top = p0_t0 + 63 - jb_pf - rb_r;
            if constexpr (PRESPLIT_RB) {
                const uint32_t rb_sa0 = static_cast<uint32_t>(__cvta_generic_to_shared(
                    ((rb_i >= 64) ? s_rb_lo : s_rb_hi) + rb_r * 16 + ((rb_i & 1) ^ ((rb_r >> 2) & 1)) * 8));
                const __half* rb_src = ((rb_i >= 64) ? rel_bias_f16_lo : rel_bias_f16_hi) + (rb_i & 1) * 8;
#pragma unroll 1
                for (int st = 0; st < 4; ++st) {
                    const int d = d_top - st * 32;
                    const bool d_ok = ((unsigned) d < (unsigned) n_ext);
                    const void* ga = d_ok ? (const void*) (rb_src + (size_t) d * 16)
                                          : (const void*) rb_src;
                    cp_async_cg_16(rb_sa0 + (uint32_t) (st * 1024u), ga, d_ok ? 16 : 0);
                }
            } else {
                const int rb_c = rb_i & 3;
                const uint32_t rb_sa0 = static_cast<uint32_t>(__cvta_generic_to_shared(
                    reinterpret_cast<float*>(s_rb_hi) + rb_r * 16 + rb_c * 4));
#pragma unroll 1
                for (int st = 0; st < 4; ++st) {
                    const int d = d_top - st * 32;
                    const bool d_ok = ((unsigned) d < (unsigned) n_ext);
                    const void* ga = d_ok ? (const void*) (rel_bias_ptr + (size_t) d * 16 + rb_c * 4)
                                          : (const void*) rel_bias_ptr;
                    cp_async_cg_16(rb_sa0 + (uint32_t) (st * 32 * 64), ga, d_ok ? 16 : 0);
                }
            }
            cp_async_commit();
        }
    };

    auto finish_rb_async = [&]() {
        if (is_rb_warp) {
            cp_async_wait0();
            if constexpr (!PRESPLIT_RB) {
                const int rb_i = tid - 256;
                const int rb_r = rb_i >> 2;
                const int rb_c = rb_i & 3;
                const float* s_rb_raw = reinterpret_cast<const float*>(s_rb_hi);
                uint4 dq_buf[4];
#pragma unroll
                for (int st = 0; st < 4; ++st) {
                    dq_buf[st] = *reinterpret_cast<const uint4*>(s_rb_raw + (st * 32 + rb_r) * 16 + rb_c * 4);
                }
                asm volatile("bar.sync 1, 128;\n" ::: "memory");
                const int c_hi = rb_c >> 1;
                const int c_off = (rb_c & 1) * 4;
#pragma unroll
                for (int st = 0; st < 4; ++st) {
                    const int row = st * 32 + rb_r;
                    const int chunk = c_hi ^ ((row >> 2) & 1);
                    __half2* dhi = reinterpret_cast<__half2*>(s_rb_hi + row * 16 + chunk * 8 + c_off);
                    __half2* dlo = reinterpret_cast<__half2*>(s_rb_lo + row * 16 + chunk * 8 + c_off);
                    split_ozaki_h2(__uint_as_float(dq_buf[st].x), __uint_as_float(dq_buf[st].y), dhi[0], dlo[0]);
                    split_ozaki_h2(__uint_as_float(dq_buf[st].z), __uint_as_float(dq_buf[st].w), dhi[1], dlo[1]);
                }
            }
        }
    };

    // Prologue 0: Launch tile-0 K_raw[0] + rb[0] cp.async.cg BEFORE Prologue Q/R loads (hides tile-0 latency & drops 1 BAR!).
    const bool have_tile0 = (j_start < j_end);
    const bool do_rb_0 = have_tile0 && has_rel && (n_ext > 0) && (p0_t0 - (j_start + 63) < n_ext);
    __half sc_k0 = __float2half(0.0f);
    if (have_tile0) {
        launch_kv_async(false, j_start, slot_base, sc_k0);
        if (do_rb_0) launch_rb_async(j_start);
    }

    // Prologue 1: Load 192x128 Q -> XOR-swizzled sQ (col8 ^ (qr & 7), STRIDE=128).
    // Wave 11: Single DRAM read (qv[8] in regs before acc[8][8] is live -> 0 spill, bit-identical, 128-bit STS!).
#pragma unroll 1
    for (int base_r = 0; base_r < 192; base_r += 96) {
        const int qr = base_r + (tid >> 2);
        const int qc0 = (tid & 3) * 32;
        const int col8_base = (tid & 3) * 4;
        const int row_swz = qr & 7;
        const int g = qr >> 6, r = qr & 63, t = t0 + r, h = 3 * hkv + g;
        const bool t_ok = (t < T);
        const float* qp = t_ok ? (Q + (size_t) t * q_stride + (size_t) h * 128 + qc0) : nullptr;
        const float* qsp = (t_ok && q_scale != nullptr) ? (q_scale + (size_t) h * 128 + qc0) : nullptr;
        float4 qv[8];
        float fac = t_ok ? qscale_all : 0.0f;
        if (q_scale != nullptr) {
            float sq = 0.0f;
            if (t_ok) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float4 v = *reinterpret_cast<const float4*>(qp + i * 4);
                    const float4 sc = __ldg(reinterpret_cast<const float4*>(qsp + i * 4));
                    const float vx = v.x * sc.x, vy = v.y * sc.y, vz = v.z * sc.z, vw = v.w * sc.w;
                    qv[i] = make_float4(vx, vy, vz, vw);
                    sq += vx * vx + vy * vy + vz * vz + vw * vw;
                }
            }
            sq += __shfl_xor_sync(0xffffffffu, sq, 1);
            sq += __shfl_xor_sync(0xffffffffu, sq, 2);
            if (t_ok) {
                fac = fminf(1.0f, qk_clip / (sqrtf(sq) + 1e-6f)) * compute_tau_r(r) * qscale_all;
            }
        } else if (t_ok) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                qv[i] = *reinterpret_cast<const float4*>(qp + i * 4);
            }
        }
#pragma unroll
        for (int c2 = 0; c2 < 4; ++c2) {
            uint4* dst4 = reinterpret_cast<uint4*>(sQ + qr * STRIDE + ((col8_base + c2) ^ row_swz) * 8);
            if (t_ok) {
                const float4 v0 = qv[c2 * 2 + 0];
                const float4 v1 = qv[c2 * 2 + 1];
                const __half2 h0 = __halves2half2(__float2half(v0.x * fac), __float2half(v0.y * fac));
                const __half2 h1 = __halves2half2(__float2half(v0.z * fac), __float2half(v0.w * fac));
                const __half2 h2 = __halves2half2(__float2half(v1.x * fac), __float2half(v1.y * fac));
                const __half2 h3 = __halves2half2(__float2half(v1.z * fac), __float2half(v1.w * fac));
                *dst4 = pack_h2x4(h0, h1, h2, h3);
            } else {
                *dst4 = make_uint4(0, 0, 0, 0);
            }
        }
    }

    // Prologue 2: Per-warp R_hi (into sR_hi_persist) + R_lo (temporarily in sV, loaded to r_lo0..3 via __syncwarp() BEFORE single BAR!),
    // initialize s_rl[0..191] = 1.0f, and fold sink ONCE into initial rm0/rm1!
    if (tid < 192) {
        s_rl[tid] = 1.0f;
    }
    const int hr_init = 3 * hkv + (warp >> 2);
    const int tr0_init = t0 + ((warp & 3) * 16) + (lane >> 2);
    float rm0 = (tr0_init < T)     ? (__ldg(&sink[(size_t) tr0_init * sink_stride + hr_init]) * kLog2e) : 0.0f;
    float rm1 = (tr0_init + 8 < T) ? (__ldg(&sink[(size_t) (tr0_init + 8) * sink_stride + hr_init]) * kLog2e) : 0.0f;

    uint32_t r_lo0 = 0, r_lo1 = 0, r_lo2 = 0, r_lo3 = 0;
    if (R_ptr != nullptr) {
        const int qr = warp * 16 + (lane >> 1);
        const int half_id = lane & 1;
        const int g = qr >> 6, r = qr & 63, t = t0 + r, h = 3 * hkv + g;
        const bool t_ok = (t < T);
        const int c_swz = half_id ^ ((qr >> 2) & 1);
        __half2* dhi = reinterpret_cast<__half2*>(sR_hi_persist + qr * 16 + c_swz * 8);
        __half2* dlo = reinterpret_cast<__half2*>(sV + qr * 16 + c_swz * 8);
        const float4* rp = t_ok ? reinterpret_cast<const float4*>(R_ptr + (size_t) t * r_stride + (size_t) h * 16 + half_id * 8) : nullptr;
        const float4* rsp = (t_ok && r_scale != nullptr) ? reinterpret_cast<const float4*>(r_scale + (size_t) h * 16 + half_id * 8) : nullptr;
        float4 rv[2];
        float rfac = t_ok ? kLog2e : 0.0f;
        if (r_scale != nullptr) {
            float rsq = 0.0f;
            if (t_ok) {
#pragma unroll
                for (int rk = 0; rk < 2; ++rk) {
                    const float4 v = rp[rk], sv = __ldg(rsp + rk);
                    const float rx = v.x * sv.x, ry = v.y * sv.y, rz = v.z * sv.z, rw = v.w * sv.w;
                    rv[rk] = make_float4(rx, ry, rz, rw);
                    rsq += rx * rx + ry * ry + rz * rz + rw * rw;
                }
            }
            rsq += __shfl_xor_sync(0xffffffffu, rsq, 1);
            if (t_ok) {
                rfac = fminf(1.0f, rel_clip / (sqrtf(rsq) + 1e-6f)) * compute_tau_r(r) * kLog2e;
            }
        } else if (t_ok) {
            rv[0] = rp[0];
            rv[1] = rp[1];
        }
        if (!t_ok) {
            rv[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            rv[1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
#pragma unroll
        for (int rk = 0; rk < 2; ++rk) {
            split_ozaki_h2(rv[rk].x * rfac, rv[rk].y * rfac, dhi[rk * 2 + 0], dlo[rk * 2 + 0]);
            split_ozaki_h2(rv[rk].z * rfac, rv[rk].w * rfac, dhi[rk * 2 + 1], dlo[rk * 2 + 1]);
        }
        __syncwarp();
        if constexpr (kRelOzaki) {
            const int r_row = warp * 16 + (lane & 15);
            const int r_c8 = (lane >> 4) ^ ((r_row >> 2) & 1);
            ldmat_x4(r_lo0, r_lo1, r_lo2, r_lo3, static_cast<uint32_t>(__cvta_generic_to_shared(sV + r_row * 16 + r_c8 * 8)));
        }
    }

    // Prologue 3: Compact 2-half 8-warp tile-0 K dequant + Warps 8..11 rb wait -> SINGLE Prologue BAR!
    if (have_tile0) {
        if (is_kv_warp) {
            cp_async_wait0();
            const uint32_t sc_u = (uint32_t) __half_as_ushort(sc_k0);
#pragma unroll 1
            for (int hh = 0; hh < 2; ++hh) {
                const int rswz = hh * 4 + kv_r_sub;
                uint4 rv = lds_u4(sa_k0 + (uint32_t) (hh * 512));
                if (lane & 4) {
                    const uint32_t tx = rv.x, ty = rv.y;
                    rv.x = rv.z; rv.y = rv.w; rv.z = tx; rv.w = ty;
                }
                if (hh == 1) asm volatile("bar.warp.sync 0xffffffff;\n" ::: "memory");
                const uint32_t sc_sh = shfl_idx_u32_pin(sc_u, rswz);
                const __half2 dsc = __half2half2(__ushort_as_half((uint16_t) sc_sh));
                const __half2 bias = __hmul2(__half2half2(__float2half(-1152.0f)), dsc);
                const uint32_t dst_sa = (hh == 0) ? sa_kdst0 : sa_kdst1;
                __half2 h0, h1, h2v, h3;
                i8x4_to_h2x2_fma(rv.x, dsc, bias, h0, h1);
                i8x4_to_h2x2_fma(rv.y, dsc, bias, h2v, h3);
                sts_u4(dst_sa, pack_h2x4(h0, h1, h2v, h3));
                i8x4_to_h2x2_fma(rv.z, dsc, bias, h0, h1);
                i8x4_to_h2x2_fma(rv.w, dsc, bias, h2v, h3);
                sts_u4(dst_sa ^ 16u, pack_h2x4(h0, h1, h2v, h3));
            }
        }
        if (do_rb_0) finish_rb_async();
    }
    __syncthreads();  // Publishes sQ, sR_hi_persist, s_rl, sK[0], s_rb_hi/lo[0], and frees sV!

    // Hoist Toeplitz lane-invariant constants & Ozaki SMEM base addresses inside Step A3 so Step A2 & B2 keep 0 extra regs!

    float acc[8][8];
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[n][i] = 0.0f;

    // Single compact jb loop (0 CALL, ~39 KB SASS, STACK: 0 on all 4 instantiations!).
    for (int jb = j_start; jb < j_end; jb += 64) {
        // Step A1: Launch V_raw[jb] cp.async.cg into second half of per-warp sV slices.
        __half sc_v = __float2half(0.0f);
        launch_kv_async(true, jb, slot_base, sc_v);

        // Step A2: De-clumped QK^T (1 LDSM -> 2 HMMA, peeled ki=7) + 2-Half 8-warp sV dequant (ki=4, 6; precomputed sa_kdst0/1).
        float fc[4][8];
#pragma unroll
        for (int ct = 0; ct < 4; ++ct)
#pragma unroll
            for (int i = 0; i < 8; ++i) fc[ct][i] = 0.0f;

        {
            const uint32_t sq_row_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sQ + (warp * 16 + (lane & 15)) * STRIDE));
            const uint32_t sq_c0 = (uint32_t) ((lane >> 4) ^ (lane & 7));
            const int k_row = (((lane >> 4) << 3) | (lane & 7));
            const uint32_t sk_row_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sK + k_row * STRIDE));
            const uint32_t sk_c0 = (uint32_t) (((lane >> 3) & 1) ^ (lane & 7));

            constexpr uint32_t kSkTileB = 16u * STRIDE * 2u;  // 4,096 B
            constexpr uint32_t kSvOffB  = 64u * STRIDE * 2u;  // 16,384 B (sV - sK)
            const uint32_t sa_v0 = sa_k0 + kSvOffB;
            uint32_t q0_0, q0_1, q0_2, q0_3;
            uint32_t q1_0, q1_1, q1_2, q1_3;
            uint32_t k0_0, k0_1, k0_2, k0_3;
            uint32_t k1_0, k1_1, k1_2, k1_3;
            uint4 v_rv{};
            __half2 v_dsc{}, v_bias{};

            ldmat_x4_pin(q0_0, q0_1, q0_2, q0_3, sq_row_addr + (sq_c0 << 4));
            ldmat_x4_pin(k0_0, k0_1, k0_2, k0_3, sk_row_addr + (sk_c0 << 4));
#pragma unroll
            for (int ki = 0; ki < 7; ++ki) {
                const uint32_t k_off = ((((uint32_t) ki << 1) ^ sk_c0) << 4);
                const uint32_t k_off_next = ((((uint32_t) (ki + 1) << 1) ^ sk_c0) << 4);
                const uint32_t q_off_next = ((((uint32_t) (ki + 1) << 1) ^ sq_c0) << 4);
                const bool odd = (ki & 1) != 0;
                const uint32_t qa0 = odd ? q1_0 : q0_0, qa1 = odd ? q1_1 : q0_1;
                const uint32_t qa2 = odd ? q1_2 : q0_2, qa3 = odd ? q1_3 : q0_3;
                ldmat_x4_pin(k1_0, k1_1, k1_2, k1_3, sk_row_addr + 1u * kSkTileB + k_off);
                mma_16816_pin(fc[0],     qa0, qa1, qa2, qa3, k0_0, k0_1);
                mma_16816_pin(fc[0] + 4, qa0, qa1, qa2, qa3, k0_2, k0_3);
                ldmat_x4_pin(k0_0, k0_1, k0_2, k0_3, sk_row_addr + 2u * kSkTileB + k_off);
                mma_16816_pin(fc[1],     qa0, qa1, qa2, qa3, k1_0, k1_1);
                mma_16816_pin(fc[1] + 4, qa0, qa1, qa2, qa3, k1_2, k1_3);
                ldmat_x4_pin(k1_0, k1_1, k1_2, k1_3, sk_row_addr + 3u * kSkTileB + k_off);
                mma_16816_pin(fc[2],     qa0, qa1, qa2, qa3, k0_0, k0_1);
                mma_16816_pin(fc[2] + 4, qa0, qa1, qa2, qa3, k0_2, k0_3);
                ldmat_x4_pin(k0_0, k0_1, k0_2, k0_3, sk_row_addr + k_off_next);
                if (!odd) {
                    ldmat_x4_pin(q1_0, q1_1, q1_2, q1_3, sq_row_addr + q_off_next);
                } else {
                    ldmat_x4_pin(q0_0, q0_1, q0_2, q0_3, sq_row_addr + q_off_next);
                }
                mma_16816_pin(fc[3],     qa0, qa1, qa2, qa3, k1_0, k1_1);
                mma_16816_pin(fc[3] + 4, qa0, qa1, qa2, qa3, k1_2, k1_3);

                // Overlapped 4-way 8-warp sV dequant spread across ki=5,6,7 (doubles V_raw HMMA shadow to 48 HMMAs, 0 bank conflicts!):
                if (ki == 5) {
                    if (is_kv_warp) {
                        const uint32_t sc_u = (uint32_t) __half_as_ushort(sc_v);
                        const uint32_t sc_sh = shfl_idx_u32_pin(sc_u, kv_r_sub);
                        cp_async_wait0();
                        v_rv = lds_u4(sa_v0);
                        if (lane & 4) {
                            const uint32_t tx = v_rv.x, ty = v_rv.y;
                            v_rv.x = v_rv.z; v_rv.y = v_rv.w; v_rv.z = tx; v_rv.w = ty;
                        }
                        v_dsc  = __half2half2(__ushort_as_half((uint16_t) sc_sh));
                        v_bias = __hmul2(__half2half2(__float2half(-1152.0f)), v_dsc);
                        __half2 h0, h1, h2v, h3;
                        i8x4_to_h2x2_fma(v_rv.x, v_dsc, v_bias, h0, h1);
                        i8x4_to_h2x2_fma(v_rv.y, v_dsc, v_bias, h2v, h3);
                        sts_u4(sa_kdst0 + kSvOffB, pack_h2x4(h0, h1, h2v, h3));
                    }
                } else if (ki == 6) {
                    if (is_kv_warp) {
                        __half2 h0, h1, h2v, h3;
                        i8x4_to_h2x2_fma(v_rv.z, v_dsc, v_bias, h0, h1);
                        i8x4_to_h2x2_fma(v_rv.w, v_dsc, v_bias, h2v, h3);
                        sts_u4((sa_kdst0 ^ 16u) + kSvOffB, pack_h2x4(h0, h1, h2v, h3));
                        v_rv = lds_u4(sa_v0 + 512u);
                        if (lane & 4) {
                            const uint32_t tx = v_rv.x, ty = v_rv.y;
                            v_rv.x = v_rv.z; v_rv.y = v_rv.w; v_rv.z = tx; v_rv.w = ty;
                        }
                        asm volatile("bar.warp.sync 0xffffffff;\n" ::: "memory");
                        const uint32_t sc_u = (uint32_t) __half_as_ushort(sc_v);
                        const uint32_t sc_sh = shfl_idx_u32_pin(sc_u, 4 + kv_r_sub);
                        v_dsc  = __half2half2(__ushort_as_half((uint16_t) sc_sh));
                        v_bias = __hmul2(__half2half2(__float2half(-1152.0f)), v_dsc);
                        i8x4_to_h2x2_fma(v_rv.x, v_dsc, v_bias, h0, h1);
                        i8x4_to_h2x2_fma(v_rv.y, v_dsc, v_bias, h2v, h3);
                        sts_u4(sa_kdst1 + kSvOffB, pack_h2x4(h0, h1, h2v, h3));
                    }
                }
            }
            // Peeled ki = 7: final sV STS before 8-HMMA cushion!
            {
                if (is_kv_warp) {
                    __half2 h0, h1, h2v, h3;
                    i8x4_to_h2x2_fma(v_rv.z, v_dsc, v_bias, h0, h1);
                    i8x4_to_h2x2_fma(v_rv.w, v_dsc, v_bias, h2v, h3);
                    sts_u4((sa_kdst1 ^ 16u) + kSvOffB, pack_h2x4(h0, h1, h2v, h3));
                }
                const uint32_t k_off = ((14u ^ sk_c0) << 4);
                ldmat_x4_pin(k1_0, k1_1, k1_2, k1_3, sk_row_addr + 1u * kSkTileB + k_off);
                mma_16816_pin(fc[0],     q1_0, q1_1, q1_2, q1_3, k0_0, k0_1);
                mma_16816_pin(fc[0] + 4, q1_0, q1_1, q1_2, q1_3, k0_2, k0_3);
                ldmat_x4_pin(k0_0, k0_1, k0_2, k0_3, sk_row_addr + 2u * kSkTileB + k_off);
                mma_16816_pin(fc[1],     q1_0, q1_1, q1_2, q1_3, k1_0, k1_1);
                mma_16816_pin(fc[1] + 4, q1_0, q1_1, q1_2, q1_3, k1_2, k1_3);
                ldmat_x4_pin(k1_0, k1_1, k1_2, k1_3, sk_row_addr + 3u * kSkTileB + k_off);
                mma_16816_pin(fc[2],     q1_0, q1_1, q1_2, q1_3, k0_0, k0_1);
                mma_16816_pin(fc[2] + 4, q1_0, q1_1, q1_2, q1_3, k0_2, k0_3);
                mma_16816_pin(fc[3],     q1_0, q1_1, q1_2, q1_3, k1_0, k1_1);
                mma_16816_pin(fc[3] + 4, q1_0, q1_1, q1_2, q1_3, k1_2, k1_3);
            }
        }

        // Step A3: Exact 22-bit Ozaki R * rel_bias strictly AFTER QK^T completes (2-block overlapped scheduling,
        // interleaved biject_add between HMMAs, strict BLK=0->1->2->3->4 biject_add FP32 order -> 100% bit-exact!) -> BAR #1!
        if (has_rel && (n_ext > 0) && (p0_t0 - (jb + 63) < n_ext)) {
            constexpr float kInv2048 = 1.0f / 2048.0f;
            const int r_row_oz = warp * 16 + (lane & 15);
            const int r_c8_oz = (lane >> 4) ^ ((r_row_oz >> 2) & 1);
            const uint32_t sr_hi_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sR_hi_persist + r_row_oz * 16 + r_c8_oz * 8));
            const int k_row_oz = (((lane >> 4) << 3) | (lane & 7));
            const int rb_row_oz = (3 - (warp & 3)) * 16 + k_row_oz;
            const int rb_c8_oz = ((lane >> 3) & 1) ^ ((rb_row_oz >> 2) & 1);
            const uint32_t srb_hi_addr0 = static_cast<uint32_t>(__cvta_generic_to_shared(s_rb_hi + rb_row_oz * 16 + rb_c8_oz * 8));
            const int t_lane_oz = lane & 3;
            const int s_toe_oz  = 7 - (lane >> 2);
            const int sh_a_oz   = s_toe_oz >> 1;
            const int sh_b_oz   = (s_toe_oz + 1) >> 1;
            const int lane_a_oz = (lane & ~3) | ((t_lane_oz + sh_a_oz) & 3);
            const int lane_b_oz = (lane & ~3) | ((t_lane_oz + sh_b_oz) & 3);
            const bool send_hi_a_oz = (t_lane_oz < sh_a_oz);
            const bool send_hi_b_oz = (t_lane_oz < sh_b_oz);
            const bool odd_s_oz     = (s_toe_oz & 1) != 0;
            auto biject_add = [&](float lo0, float lo1, float hi0, float hi1, float& dst0, float& dst1) {
                float e_even, e_odd;
                asm("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;\n"
                    : "=f"(e_even) : "f"(send_hi_b_oz ? hi0 : lo0), "r"(lane_b_oz));
                asm("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;\n"
                    : "=f"(e_odd) : "f"(send_hi_a_oz ? hi1 : lo1), "r"(lane_a_oz));
                dst0 += odd_s_oz ? e_odd  : e_even;
                dst1 += odd_s_oz ? e_even : e_odd;
            };
            uint32_t r_hi0, r_hi1, r_hi2, r_hi3;
            ldmat_x4_pin(r_hi0, r_hi1, r_hi2, r_hi3, sr_hi_addr);
            float prev_hi[4];
            float z0a[4] = {0.0f, 0.0f, 0.0f, 0.0f}, z1a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float z0b[4] = {0.0f, 0.0f, 0.0f, 0.0f}, z1b[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            uint32_t ba0, ba1, ba2, ba3, bb0, bb1, bb2, bb3;

            // Paired BLK=0 (a) + BLK=1 (b)
            if constexpr (kRelOzaki) {
                ldmat_x4_pin(ba0, ba1, ba2, ba3, srb_hi_addr0 + (4096u + 0u * 512u));
                ldmat_x4_pin(bb0, bb1, bb2, bb3, srb_hi_addr0 + (4096u + 1u * 512u));
                mma_16816_pin(z0a, r_hi0, r_hi1, r_hi2, r_hi3, ba0, ba1);
                mma_16816_pin(z1a, r_hi0, r_hi1, r_hi2, r_hi3, ba2, ba3);
                mma_16816_pin(z0b, r_hi0, r_hi1, r_hi2, r_hi3, bb0, bb1);
                mma_16816_pin(z1b, r_hi0, r_hi1, r_hi2, r_hi3, bb2, bb3);
                ldmat_x4_pin(ba0, ba1, ba2, ba3, srb_hi_addr0 + 0u * 512u);
                ldmat_x4_pin(bb0, bb1, bb2, bb3, srb_hi_addr0 + 1u * 512u);
                mma_16816_pin(z0a, r_lo0, r_lo1, r_lo2, r_lo3, ba0, ba1);
                mma_16816_pin(z1a, r_lo0, r_lo1, r_lo2, r_lo3, ba2, ba3);
                mma_16816_pin(z0b, r_lo0, r_lo1, r_lo2, r_lo3, bb0, bb1);
                mma_16816_pin(z1b, r_lo0, r_lo1, r_lo2, r_lo3, bb2, bb3);
                z0a[0] *= kInv2048; z0a[1] *= kInv2048; z0a[2] *= kInv2048; z0a[3] *= kInv2048;
                z1a[0] *= kInv2048; z1a[1] *= kInv2048; z1a[2] *= kInv2048; z1a[3] *= kInv2048;
                z0b[0] *= kInv2048; z0b[1] *= kInv2048; z0b[2] *= kInv2048; z0b[3] *= kInv2048;
                z1b[0] *= kInv2048; z1b[1] *= kInv2048; z1b[2] *= kInv2048; z1b[3] *= kInv2048;
                mma_16816_pin(z0a, r_hi0, r_hi1, r_hi2, r_hi3, ba0, ba1);
                mma_16816_pin(z1a, r_hi0, r_hi1, r_hi2, r_hi3, ba2, ba3);
                mma_16816_pin(z0b, r_hi0, r_hi1, r_hi2, r_hi3, bb0, bb1);
                mma_16816_pin(z1b, r_hi0, r_hi1, r_hi2, r_hi3, bb2, bb3);
            } else {
                ldmat_x4_pin(ba0, ba1, ba2, ba3, srb_hi_addr0 + 0u * 512u);
                ldmat_x4_pin(bb0, bb1, bb2, bb3, srb_hi_addr0 + 1u * 512u);
                mma_16816_pin(z0a, r_hi0, r_hi1, r_hi2, r_hi3, ba0, ba1);
                mma_16816_pin(z1a, r_hi0, r_hi1, r_hi2, r_hi3, ba2, ba3);
                mma_16816_pin(z0b, r_hi0, r_hi1, r_hi2, r_hi3, bb0, bb1);
                mma_16816_pin(z1b, r_hi0, r_hi1, r_hi2, r_hi3, bb2, bb3);
            }
            biject_add(z0a[2], z0a[3], z1a[2], z1a[3], fc[0][2], fc[0][3]);
            prev_hi[0] = z1a[0]; prev_hi[1] = z1a[1]; prev_hi[2] = z1a[2]; prev_hi[3] = z1a[3];

            auto step_oz_blk = [&]<int BLK>(float (&z0c)[4], float (&z1c)[4], float (&z0p)[4], float (&z1p)[4]) {
                constexpr uint32_t kBlkOff = (uint32_t) BLK * 512u;
                uint32_t b0, b1, b2, b3;
                z0c[0] = 0.0f; z0c[1] = 0.0f; z0c[2] = 0.0f; z0c[3] = 0.0f;
                z1c[0] = 0.0f; z1c[1] = 0.0f; z1c[2] = 0.0f; z1c[3] = 0.0f;
                if constexpr (kRelOzaki) {
                    ldmat_x4_pin(b0, b1, b2, b3, srb_hi_addr0 + (4096u + kBlkOff));
                    mma_16816_pin(z0c, r_hi0, r_hi1, r_hi2, r_hi3, b0, b1);
                    biject_add(prev_hi[0], prev_hi[1], z0p[0], z0p[1], fc[BLK - 2][0], fc[BLK - 2][1]);
                    mma_16816_pin(z1c, r_hi0, r_hi1, r_hi2, r_hi3, b2, b3);
                    ldmat_x4_pin(b0, b1, b2, b3, srb_hi_addr0 + kBlkOff);
                    biject_add(prev_hi[2], prev_hi[3], z0p[2], z0p[3], fc[BLK - 2][6], fc[BLK - 2][7]);
                    mma_16816_pin(z0c, r_lo0, r_lo1, r_lo2, r_lo3, b0, b1);
                    biject_add(z0p[0], z0p[1], z1p[0], z1p[1], fc[BLK - 2][4], fc[BLK - 2][5]);
                    mma_16816_pin(z1c, r_lo0, r_lo1, r_lo2, r_lo3, b2, b3);
                    z0c[0] *= kInv2048; z0c[1] *= kInv2048; z0c[2] *= kInv2048; z0c[3] *= kInv2048;
                    z1c[0] *= kInv2048; z1c[1] *= kInv2048; z1c[2] *= kInv2048; z1c[3] *= kInv2048;
                    mma_16816_pin(z0c, r_hi0, r_hi1, r_hi2, r_hi3, b0, b1);
                    biject_add(z0p[2], z0p[3], z1p[2], z1p[3], fc[BLK - 1][2], fc[BLK - 1][3]);
                    mma_16816_pin(z1c, r_hi0, r_hi1, r_hi2, r_hi3, b2, b3);
                } else {
                    ldmat_x4_pin(b0, b1, b2, b3, srb_hi_addr0 + kBlkOff);
                    mma_16816_pin(z0c, r_hi0, r_hi1, r_hi2, r_hi3, b0, b1);
                    biject_add(prev_hi[0], prev_hi[1], z0p[0], z0p[1], fc[BLK - 2][0], fc[BLK - 2][1]);
                    mma_16816_pin(z1c, r_hi0, r_hi1, r_hi2, r_hi3, b2, b3);
                    biject_add(prev_hi[2], prev_hi[3], z0p[2], z0p[3], fc[BLK - 2][6], fc[BLK - 2][7]);
                    biject_add(z0p[0], z0p[1], z1p[0], z1p[1], fc[BLK - 2][4], fc[BLK - 2][5]);
                    biject_add(z0p[2], z0p[3], z1p[2], z1p[3], fc[BLK - 1][2], fc[BLK - 1][3]);
                }
                prev_hi[0] = z1p[0]; prev_hi[1] = z1p[1]; prev_hi[2] = z1p[2]; prev_hi[3] = z1p[3];
            };
            step_oz_blk.template operator()<2>(z0a, z1a, z0b, z1b);
            step_oz_blk.template operator()<3>(z0b, z1b, z0a, z1a);
            step_oz_blk.template operator()<4>(z0a, z1a, z0b, z1b);
            biject_add(prev_hi[0], prev_hi[1], z0a[0], z0a[1], fc[3][0], fc[3][1]);
            biject_add(prev_hi[2], prev_hi[3], z0a[2], z0a[3], fc[3][6], fc[3][7]);
            biject_add(z0a[0], z0a[1], z1a[0], z1a[1], fc[3][4], fc[3][5]);
        }
        __syncthreads();  // BAR.SYNC #1: publishes sV[jb] AND releases sK + s_rb_hi/lo!

        if constexpr (IS_SWA && !SWA_LINEAR_HM) {
            slot_base += 64;
            if (slot_base >= cap) slot_base -= cap;
        }

        // Step B1: Launch K_raw[jb+64] (Warps 0..7) + rb[jb+64] (Warps 8..11) cp.async.cg in parallel!
        const bool have_next = (jb + 64 < j_end);
        __half sc_k = __float2half(0.0f);
        if (have_next) {
            launch_kv_async(false, jb + 64, slot_base, sc_k);
            if (has_rel && (n_ext > 0) && (p0_t0 - (jb + 127) < n_ext)) launch_rb_async(jb + 64);
        }

        // Step B2: Masking (skipped when all_valid_unmasked) + Softmax + P*V + Overlapped 8-warp sK Dequant!
        {
            const bool all_valid_unmasked = (t0 + 63 < T) &&
                                            (!IS_SWA || jb >= (p0_t0 + 63) - n_swa + 1) &&
                                            (jb + 63 <= p0_t0);
            if (!all_valid_unmasked) {
                const int r0_sm = lane >> 2;
                const int c0_sm = (lane & 3) * 2;
                const int r_loc0 = ((warp & 3) * 16) + r0_sm;
                const bool tr0_ok = (t0 + r_loc0 < T);
                const bool tr1_ok = (t0 + r_loc0 + 8 < T);
                const int pos_q0 = p0_t0 + r_loc0;
                const unsigned d_mask_lim = (unsigned) (IS_SWA ? n_swa : 0x7fffffff);
                const int d_base = pos_q0 - (jb + c0_sm);
#pragma unroll
                for (int ct = 0; ct < 4; ++ct) {
                    const int d00 = d_base - ct * 16;
                    if (!tr0_ok || (unsigned) d00       >= d_mask_lim) fc[ct][0] = -1e30f;
                    if (!tr0_ok || (unsigned) (d00 - 1) >= d_mask_lim) fc[ct][1] = -1e30f;
                    if (!tr0_ok || (unsigned) (d00 - 8) >= d_mask_lim) fc[ct][4] = -1e30f;
                    if (!tr0_ok || (unsigned) (d00 - 9) >= d_mask_lim) fc[ct][5] = -1e30f;
                    if (!tr1_ok || (unsigned) (d00 + 8) >= d_mask_lim) fc[ct][2] = -1e30f;
                    if (!tr1_ok || (unsigned) (d00 + 7) >= d_mask_lim) fc[ct][3] = -1e30f;
                    if (!tr1_ok || (unsigned) d00       >= d_mask_lim) fc[ct][6] = -1e30f;
                    if (!tr1_ok || (unsigned) (d00 - 1) >= d_mask_lim) fc[ct][7] = -1e30f;
                }
            }

            float mx0 = -1e30f, mx1 = -1e30f;
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) {
                mx0 = fmaxf(mx0, fmaxf(fmaxf(fc[ct][0], fc[ct][1]), fmaxf(fc[ct][4], fc[ct][5])));
                mx1 = fmaxf(mx1, fmaxf(fmaxf(fc[ct][2], fc[ct][3]), fmaxf(fc[ct][6], fc[ct][7])));
            }
            const float m_new0 = fmaxf(rm0, wmma_row_max_16(mx0));
            const float m_new1 = fmaxf(rm1, wmma_row_max_16(mx1));
            const bool need_rescale = (__any_sync(0xffffffffu, m_new0 != rm0 || m_new1 != rm1) != 0);
            float a0 = 1.0f, a1 = 1.0f;
            if (need_rescale) {
                a0 = (m_new0 == rm0) ? 1.0f : fast_exp2f_hw(rm0 - m_new0);
                a1 = (m_new1 == rm1) ? 1.0f : fast_exp2f_hw(rm1 - m_new1);
            }
            rm0 = m_new0;
            rm1 = m_new1;

            constexpr uint32_t kSvTileB = 16u * STRIDE * 2u;  // 4,096 B
            const uint32_t sv_row_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sV + (lane & 15) * STRIDE));
            const uint32_t sv_c0 = (uint32_t) ((lane >> 4) ^ (lane & 7));
            uint32_t b0_0, b0_1, b0_2, b0_3;
            uint32_t b1_0, b1_1, b1_2, b1_3;

            // Pre-issue ct=0 b0 LDSM + branchless acc[0] rescale (1.0f * x == x bit-exact!) overlapped with ct=0 ex2_pin!
            ldmat_x4_t_pin(b0_0, b0_1, b0_2, b0_3, sv_row_addr + (sv_c0 << 4));
            acc[0][0] *= a0; acc[0][1] *= a0; acc[0][2] *= a1; acc[0][3] *= a1;
            acc[0][4] *= a0; acc[0][5] *= a0; acc[0][6] *= a1; acc[0][7] *= a1;
            float ps0 = 0.0f, ps1 = 0.0f;
            float ps0_01, ps0_45, ps1_23, ps1_67;
            uint32_t pa0, pa1, pa2, pa3;
            {
                const float p00 = ex2_pin(fc[0][0] - m_new0);
                const float p01 = ex2_pin(fc[0][1] - m_new0);
                const float p12 = ex2_pin(fc[0][2] - m_new1);
                const float p13 = ex2_pin(fc[0][3] - m_new1);
                const float p04 = ex2_pin(fc[0][4] - m_new0);
                const float p05 = ex2_pin(fc[0][5] - m_new0);
                const float p16 = ex2_pin(fc[0][6] - m_new1);
                const float p17 = ex2_pin(fc[0][7] - m_new1);
                pa0 = f2_to_h2u(p00, p01);
                pa1 = f2_to_h2u(p12, p13);
                pa2 = f2_to_h2u(p04, p05);
                pa3 = f2_to_h2u(p16, p17);
                ps0_01 = fadd_pin(p00, p01);
                ps0_45 = fadd_pin(p04, p05);
                ps1_23 = fadd_pin(p12, p13);
                ps1_67 = fadd_pin(p16, p17);
            }

            uint4 k_rv{};
            __half2 k_dsc{}, k_bias{};

            // Fused ct=0..3: P*V(ct) + branchless pipelined acc[n+1] rescale in ct=0 + HMMA-overlapped ps0/ps1 + precomputed sa_kdst0/1 sK dequant!
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) {
                const uint32_t sv_ct = sv_row_addr + (uint32_t) ct * kSvTileB;
                uint32_t npa0, npa1, npa2, npa3;
                if (ct > 0) {
                    ldmat_x4_t_pin(b0_0, b0_1, b0_2, b0_3, sv_ct + (sv_c0 << 4));
                }
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    if (n < 7) {
                        const uint32_t n_next = (uint32_t) (n + 1);
                        if ((n & 1) == 0) {
                            ldmat_x4_t_pin(b1_0, b1_1, b1_2, b1_3, sv_ct + (((n_next << 1) ^ sv_c0) << 4));
                        } else {
                            ldmat_x4_t_pin(b0_0, b0_1, b0_2, b0_3, sv_ct + (((n_next << 1) ^ sv_c0) << 4));
                        }
                    }
                    if ((n & 1) == 0) {
                        mma_16816_pin(acc[n], pa0, pa1, pa2, pa3, b0_0, b0_1);
                        if (ct == 0 && n < 7) {
                            acc[n + 1][0] *= a0; acc[n + 1][1] *= a0; acc[n + 1][2] *= a1; acc[n + 1][3] *= a1;
                        }
                        mma_16816_pin(acc[n] + 4, pa0, pa1, pa2, pa3, b0_2, b0_3);
                        if (ct == 0 && n < 7) {
                            acc[n + 1][4] *= a0; acc[n + 1][5] *= a0; acc[n + 1][6] *= a1; acc[n + 1][7] *= a1;
                        }
                    } else {
                        mma_16816_pin(acc[n], pa0, pa1, pa2, pa3, b1_0, b1_1);
                        if (ct == 0 && n < 7) {
                            acc[n + 1][0] *= a0; acc[n + 1][1] *= a0; acc[n + 1][2] *= a1; acc[n + 1][3] *= a1;
                        }
                        mma_16816_pin(acc[n] + 4, pa0, pa1, pa2, pa3, b1_2, b1_3);
                        if (ct == 0 && n < 7) {
                            acc[n + 1][4] *= a0; acc[n + 1][5] *= a0; acc[n + 1][6] *= a1; acc[n + 1][7] *= a1;
                        }
                    }
                    if (ct == 0) {
                        if (n == 0) {
                            ps0 = fadd_pin(ps0_01, ps0_45);
                        } else if (n == 1) {
                            ps1 = fadd_pin(ps1_23, ps1_67);
                        } else if (n == 2) {
                            const float p0 = ex2_pin(fc[1][0] - m_new0);
                            const float p1 = ex2_pin(fc[1][1] - m_new0);
                            npa0 = f2_to_h2u(p0, p1);
                            ps0 = fadd_pin(ps0, fadd_pin(p0, p1));
                        } else if (n == 3) {
                            const float p0 = ex2_pin(fc[1][2] - m_new1);
                            const float p1 = ex2_pin(fc[1][3] - m_new1);
                            npa1 = f2_to_h2u(p0, p1);
                            ps1 = fadd_pin(ps1, fadd_pin(p0, p1));
                        } else if (n == 4) {
                            const float p0 = ex2_pin(fc[1][4] - m_new0);
                            const float p1 = ex2_pin(fc[1][5] - m_new0);
                            npa2 = f2_to_h2u(p0, p1);
                            ps0 = fadd_pin(ps0, fadd_pin(p0, p1));
                        } else if (n == 6) {
                            const float p0 = ex2_pin(fc[1][6] - m_new1);
                            const float p1 = ex2_pin(fc[1][7] - m_new1);
                            npa3 = f2_to_h2u(p0, p1);
                            ps1 = fadd_pin(ps1, fadd_pin(p0, p1));
                        }
                    } else if (ct < 3) {
                        if (n == 0) {
                            const float p0 = ex2_pin(fc[ct + 1][0] - m_new0);
                            const float p1 = ex2_pin(fc[ct + 1][1] - m_new0);
                            npa0 = f2_to_h2u(p0, p1);
                            ps0 = fadd_pin(ps0, fadd_pin(p0, p1));
                        } else if (n == 2) {
                            const float p0 = ex2_pin(fc[ct + 1][2] - m_new1);
                            const float p1 = ex2_pin(fc[ct + 1][3] - m_new1);
                            npa1 = f2_to_h2u(p0, p1);
                            ps1 = fadd_pin(ps1, fadd_pin(p0, p1));
                        } else if (n == 4) {
                            const float p0 = ex2_pin(fc[ct + 1][4] - m_new0);
                            const float p1 = ex2_pin(fc[ct + 1][5] - m_new0);
                            npa2 = f2_to_h2u(p0, p1);
                            ps0 = fadd_pin(ps0, fadd_pin(p0, p1));
                        } else if (n == 6) {
                            const float p0 = ex2_pin(fc[ct + 1][6] - m_new1);
                            const float p1 = ex2_pin(fc[ct + 1][7] - m_new1);
                            npa3 = f2_to_h2u(p0, p1);
                            ps1 = fadd_pin(ps1, fadd_pin(p0, p1));
                        }
                    }
                    // Overlapped 2-Half 8-warp sK dequant inside ct=2 (n=5,7) and ct=3 (n=1,3,5) using precomputed sa_kdst0/1!
                    if (ct == 2 && n == 5) {
                        if (have_next && is_kv_warp) {
                            const uint32_t sc_u = (uint32_t) __half_as_ushort(sc_k);
                            const uint32_t sc_sh = shfl_idx_u32_pin(sc_u, kv_r_sub);
                            cp_async_wait0();
                            k_rv = lds_u4(sa_k0);
                            if (lane & 4) {
                                const uint32_t tx = k_rv.x, ty = k_rv.y;
                                k_rv.x = k_rv.z; k_rv.y = k_rv.w; k_rv.z = tx; k_rv.w = ty;
                            }
                            k_dsc  = __half2half2(__ushort_as_half((uint16_t) sc_sh));
                            k_bias = __hmul2(__half2half2(__float2half(-1152.0f)), k_dsc);
                            __half2 h0, h1, h2v, h3;
                            i8x4_to_h2x2_fma(k_rv.x, k_dsc, k_bias, h0, h1);
                            i8x4_to_h2x2_fma(k_rv.y, k_dsc, k_bias, h2v, h3);
                            sts_u4(sa_kdst0, pack_h2x4(h0, h1, h2v, h3));
                        }
                    } else if (ct == 2 && n == 7) {
                        if (have_next && is_kv_warp) {
                            __half2 h0, h1, h2v, h3;
                            i8x4_to_h2x2_fma(k_rv.z, k_dsc, k_bias, h0, h1);
                            i8x4_to_h2x2_fma(k_rv.w, k_dsc, k_bias, h2v, h3);
                            sts_u4(sa_kdst0 ^ 16u, pack_h2x4(h0, h1, h2v, h3));
                        }
                    } else if (ct == 3 && n == 1) {
                        if (have_next && is_kv_warp) {
                            k_rv = lds_u4(sa_k0 + 512u);
                            if (lane & 4) {
                                const uint32_t tx = k_rv.x, ty = k_rv.y;
                                k_rv.x = k_rv.z; k_rv.y = k_rv.w; k_rv.z = tx; k_rv.w = ty;
                            }
                            asm volatile("bar.warp.sync 0xffffffff;\n" ::: "memory");
                            const uint32_t sc_u = (uint32_t) __half_as_ushort(sc_k);
                            const uint32_t sc_sh = shfl_idx_u32_pin(sc_u, 4 + kv_r_sub);
                            k_dsc  = __half2half2(__ushort_as_half((uint16_t) sc_sh));
                            k_bias = __hmul2(__half2half2(__float2half(-1152.0f)), k_dsc);
                        }
                    } else if (ct == 3 && n == 3) {
                        if (have_next && is_kv_warp) {
                            __half2 h0, h1, h2v, h3;
                            i8x4_to_h2x2_fma(k_rv.x, k_dsc, k_bias, h0, h1);
                            i8x4_to_h2x2_fma(k_rv.y, k_dsc, k_bias, h2v, h3);
                            sts_u4(sa_kdst1, pack_h2x4(h0, h1, h2v, h3));
                        }
                    } else if (ct == 3 && n == 5) {
                        if (have_next && is_kv_warp) {
                            __half2 h0, h1, h2v, h3;
                            i8x4_to_h2x2_fma(k_rv.z, k_dsc, k_bias, h0, h1);
                            i8x4_to_h2x2_fma(k_rv.w, k_dsc, k_bias, h2v, h3);
                            sts_u4(sa_kdst1 ^ 16u, pack_h2x4(h0, h1, h2v, h3));
                        }
                    }
                }
                if (ct < 3) {
                    pa0 = npa0; pa1 = npa1; pa2 = npa2; pa3 = npa3;
                }
            }

            const float sum0 = wmma_row_sum_16(ps0);
            const float sum1 = wmma_row_sum_16(ps1);
            if (wmma_row_leader_16(lane)) {
                const int qr0 = warp * 16 + (lane >> 2);
                s_rl[qr0]     = s_rl[qr0]     * a0 + sum0;
                s_rl[qr0 + 8] = s_rl[qr0 + 8] * a1 + sum1;
            }
        }

        // Step B3: sK[jb+64] dequant already completed inside ct=2,3; Warps 8..11 wait rb -> BAR #2!
        if (have_next) {
            if (has_rel && (n_ext > 0) && (p0_t0 - (jb + 127) < n_ext)) finish_rb_async();
            __syncthreads();  // BAR.SYNC #2: publishes sK[jb+64] + s_rb_hi/lo[jb+64] AND releases sV!
        }
    }

    // Epilogue (sm_80+): 0-BAR compact unroll-1 per-warp staging + single shared rp loop (<=32KB binary, STACK: 0)!
    {
        const int acc_r0 = lane >> 2;
        const int acc_r1 = acc_r0 + 8;
        const int acc_c0 = (lane & 3) * 2;
        const int acc_c1 = acc_c0 + 8;
        __syncwarp();
        const float inv_l0 = __fdividef(1.0f, s_rl[warp * 16 + acc_r0]);
        const float inv_l1 = __fdividef(1.0f, s_rl[warp * 16 + acc_r1]);
        const int gate_stride = (ld_gate > 0) ? ld_gate : (n_head * 128);
        constexpr int S_OUT_LD = 72;  // 0 bank conflicts on STS.64 (half-warp) AND LDS.128!
        float* s_warp = reinterpret_cast<float*>(smem_raw) + warp * (16 * S_OUT_LD);
        const int head_g = warp >> 2;
        const int h_out = 3 * hkv + head_g;
        const int tok_base = t0 + (warp & 3) * 16;
        const bool full_rows = (tok_base + 15 < T);

#pragma unroll 1
        for (int n_half = 0; n_half < 2; ++n_half) {
            if (n_half == 0) {
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    float* dst0 = s_warp + acc_r0 * S_OUT_LD + ni * 16;
                    float* dst1 = s_warp + acc_r1 * S_OUT_LD + ni * 16;
                    *reinterpret_cast<float2*>(dst0 + acc_c0) = make_float2(acc[ni][0] * inv_l0, acc[ni][1] * inv_l0);
                    *reinterpret_cast<float2*>(dst0 + acc_c1) = make_float2(acc[ni][4] * inv_l0, acc[ni][5] * inv_l0);
                    *reinterpret_cast<float2*>(dst1 + acc_c0) = make_float2(acc[ni][2] * inv_l1, acc[ni][3] * inv_l1);
                    *reinterpret_cast<float2*>(dst1 + acc_c1) = make_float2(acc[ni][6] * inv_l1, acc[ni][7] * inv_l1);
                }
            } else {
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    float* dst0 = s_warp + acc_r0 * S_OUT_LD + ni * 16;
                    float* dst1 = s_warp + acc_r1 * S_OUT_LD + ni * 16;
                    *reinterpret_cast<float2*>(dst0 + acc_c0) = make_float2(acc[4 + ni][0] * inv_l0, acc[4 + ni][1] * inv_l0);
                    *reinterpret_cast<float2*>(dst0 + acc_c1) = make_float2(acc[4 + ni][4] * inv_l0, acc[4 + ni][5] * inv_l0);
                    *reinterpret_cast<float2*>(dst1 + acc_c0) = make_float2(acc[4 + ni][2] * inv_l1, acc[4 + ni][3] * inv_l1);
                    *reinterpret_cast<float2*>(dst1 + acc_c1) = make_float2(acc[4 + ni][6] * inv_l1, acc[4 + ni][7] * inv_l1);
                }
            }
            __syncwarp();
            const int r_lane = lane >> 4;   // 0 or 1 (2 rows/iter, 16 lanes = 64 floats = 256 B coalesced!)
            const int col4   = lane & 15;   // 0..15
            const int col    = n_half * 64 + col4 * 4;
            if (full_rows && gate_f16 != nullptr && out_f16 != nullptr) {
                const size_t o_row_step = (size_t) n_head * 256u;
                const size_t g_row_step = (size_t) gate_stride * 2u;
                size_t oi = ((size_t) (tok_base + r_lane) * n_head + (size_t) h_out) * 128u + (size_t) col;
                size_t gi = (size_t) (tok_base + r_lane) * gate_stride + (size_t) h_out * 128u + (size_t) col;
                uint2 g_cur = __ldcs(reinterpret_cast<const uint2*>(gate_f16 + gi));
#pragma unroll 1
                for (int rp = 0; rp < 8; ++rp) {
                    const uint2 g_nxt = (rp + 1 < 8)
                        ? __ldcs(reinterpret_cast<const uint2*>(gate_f16 + gi + g_row_step))
                        : g_cur;
                    const int r = rp * 2 + r_lane;
                    float4 o = *reinterpret_cast<const float4*>(s_warp + r * S_OUT_LD + col4 * 4);
                    __half2 gh0, gh1;
                    memcpy(&gh0, &g_cur.x, 4);
                    memcpy(&gh1, &g_cur.y, 4);
                    const float2 gf0 = __half22float2(gh0);
                    const float2 gf1 = __half22float2(gh1);
                    o.x *= sigmf_fast_sm75(gf0.x); o.y *= sigmf_fast_sm75(gf0.y);
                    o.z *= sigmf_fast_sm75(gf1.x); o.w *= sigmf_fast_sm75(gf1.y);
                    const __half2 p0h = __halves2half2(__float2half_rn(o.x), __float2half_rn(o.y));
                    const __half2 p1h = __halves2half2(__float2half_rn(o.z), __float2half_rn(o.w));
                    uint2 u;
                    memcpy(&u.x, &p0h, 4);
                    memcpy(&u.y, &p1h, 4);
                    stcs_u2(reinterpret_cast<uint2*>(out_f16 + oi), u);
                    g_cur = g_nxt;
                    oi += o_row_step;
                    gi += g_row_step;
                }
            } else {
#pragma unroll 1
                for (int rp = 0; rp < 8; ++rp) {
                    const int r = rp * 2 + r_lane;
                    const int t = tok_base + r;
                    if (full_rows || t < T) {
                        float4 o = *reinterpret_cast<const float4*>(s_warp + r * S_OUT_LD + col4 * 4);
                        const size_t oi = ((size_t) t * n_head + (size_t) h_out) * 128u + (size_t) col;
                        if (gate_f16 != nullptr || gate != nullptr) {
                            const size_t gi = (size_t) t * gate_stride + (size_t) h_out * 128u + (size_t) col;
                            float4 g4;
                            if (gate_f16 != nullptr) {
                                const uint2 g2u = *reinterpret_cast<const uint2*>(gate_f16 + gi);
                                __half2 gh0, gh1;
                                memcpy(&gh0, &g2u.x, 4);
                                memcpy(&gh1, &g2u.y, 4);
                                const float2 gf0 = __half22float2(gh0);
                                const float2 gf1 = __half22float2(gh1);
                                g4 = make_float4(gf0.x, gf0.y, gf1.x, gf1.y);
                            } else {
                                g4 = *reinterpret_cast<const float4*>(gate + gi);
                            }
                            o.x *= sigmf_fast_sm75(g4.x); o.y *= sigmf_fast_sm75(g4.y);
                            o.z *= sigmf_fast_sm75(g4.z); o.w *= sigmf_fast_sm75(g4.w);
                        }
                        if (out_f16) {
                            const __half2 p0h = __halves2half2(__float2half_rn(o.x), __float2half_rn(o.y));
                            const __half2 p1h = __halves2half2(__float2half_rn(o.z), __float2half_rn(o.w));
                            uint2 u;
                            memcpy(&u.x, &p0h, 4);
                            memcpy(&u.y, &p1h, 4);
                            *reinterpret_cast<uint2*>(out_f16 + oi) = u;
                        } else {
                            *reinterpret_cast<float4*>(out + oi) = o;
                        }
                    }
                }
            }
            if (n_half == 0) __syncwarp();
        }
    }
#endif
#else
    (void) Q; (void) kq; (void) ks; (void) vq; (void) vs; (void) sink; (void) gate; (void) out;
    (void) n_head; (void) n_head_kv; (void) n_ext; (void) cap; (void) n_swa; (void) pos0; (void) kq_scale; (void) T;
    (void) R_ptr; (void) rel_bias_ptr; (void) out_f16;
    (void) q_scale; (void) qk_clip; (void) r_scale; (void) rel_clip; (void) tau_floor_n; (void) tau_alpha;
    (void) ld_q; (void) ld_r; (void) ld_gate; (void) rel_bias_f16_hi; (void) rel_bias_f16_lo;
    (void) gate_f16; (void) ld_sink;
#endif
}

/// Zero-sP, Zero-Spill FlashAttention-2 256-row BULK Split-KV kernel for deep global prefill (sm_70+):
/// Covers [0, bulk_aligned) across sp = 0..bulk_splits-1 (writes into split_out stride total_splits = bulk_splits + 1).
/// On Volta sm_70: uses native PTX HMMA.884 (mma.sync.aligned.m8n8k4) with 128-bit LDS.128 QK^T, 8-way batched
/// LDS.64 P*V, conflict-free stride-144 SMEM (92,160 B <= 96 KB), and compact packed4 softmax (74.9 TFLOPS = 89% cuBLAS).
/// On sm_75+: keeps WMMA 2x4 Q*K^T -> 0-shuffle wmma_acc_to_matA_16x16 -> 2x8 P*V (87,040 B SMEM).
__global__ void __launch_bounds__(256, 1) k_attn_q8_gqa3_pp_fa2_256_bulk(
        const float* __restrict__ Q,
        const int8_t* __restrict__ kq, const __half* __restrict__ ks,
        const int8_t* __restrict__ vq, const __half* __restrict__ vs,
        const float* __restrict__ sink, float* __restrict__ split_out,
        int n_head, int n_head_kv, int cap, int bulk_aligned,
        float kq_scale, int T, int bulk_splits, int total_splits) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
    // Volta sm_70 native HMMA.884 path (stride 144 -> 92,160 B SMEM, bit-exact with WMMA path, +19% faster!).
    constexpr int STRIDE = 144;
    extern __shared__ __align__(16) char smem_raw[];
    __half* sQ = reinterpret_cast<__half*>(smem_raw);
    __half* sKV = sQ + 256 * STRIDE;

    const int hkv = blockIdx.y;
    const int g = blockIdx.x % 3;
    const int qt = blockIdx.x / 3;
    const int h = 3 * hkv + g;
    const int t0 = qt * 256;
    const int sp = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    constexpr float kLog2e = 1.4426950408889634f;
    const float qscale_all = kq_scale * kLog2e;

    // E8: Per-qt bulk boundary (bulk_aligned + t0) shifts qt=1..7 bulk boundary forward by qt*256 tokens (2.4x less tail)!
    const int bulk_qt = bulk_aligned + t0;
    const int n_tiles = bulk_qt >> 6;
    const int t_per_sp = (n_tiles + bulk_splits - 1) / bulk_splits;
    const int j_start = sp * t_per_sp * 64;
    const int j_end = min(bulk_qt, j_start + t_per_sp * 64);
    if (j_start >= bulk_qt) {
        const int t = t0 + tid;
        if (t < T) {
            float* dst = split_out + (((size_t) t * total_splits + sp) * n_head + h) * 130;
            dst[128] = -1e30f;
            dst[129] = 0.0f;
        }
        return;
    }

    for (int base_r = 0; base_r < 256; base_r += 32) {
        const int qr = base_r + (tid >> 3), qc0 = (tid & 7) * 16;
        const int t = t0 + qr;
        __half2* dst2 = reinterpret_cast<__half2*>(sQ + qr * STRIDE + qc0);
        if (t < T) {
            const float* qp = Q + ((size_t) t * n_head + h) * 128 + qc0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = *reinterpret_cast<const float4*>(qp + i * 4);
                dst2[i * 2 + 0] = __halves2half2(__float2half(v.x * qscale_all), __float2half(v.y * qscale_all));
                dst2[i * 2 + 1] = __halves2half2(__float2half(v.z * qscale_all), __float2half(v.w * qscale_all));
            }
        } else {
            const __half2 z2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));
#pragma unroll
            for (int i = 0; i < 8; ++i) dst2[i] = z2;
        }
    }
    __syncthreads();

    auto mma_rc = [](float d[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, {%0, %1, %2, %3, %4, %5, %6, %7};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
            : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
    };
    auto mma_rr = [](float d[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
            "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, {%0, %1, %2, %3, %4, %5, %6, %7};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
            : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
    };

    const int qr_base = warp * 32;
    float rm0[2] = {-1e30f, -1e30f}, rm1[2] = {-1e30f, -1e30f};
    float rl0[2] = {0.0f, 0.0f}, rl1[2] = {0.0f, 0.0f};
    float acc[2][8][8];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int n = 0; n < 8; ++n)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[m][n][i] = 0.0f;

    const int kv_r = tid >> 2, kv_c0 = (tid & 3) * 32;
    const __half2 neg1152 = __half2half2(__float2half(-1152.0f));
    const int8_t* kq_base = kq + (size_t) hkv * cap * 128 + kv_c0;
    const int8_t* vq_base = vq + (size_t) hkv * cap * 128 + kv_c0;
    const __half* ks_h = ks + (size_t) hkv * cap;
    const __half* vs_h = vs + (size_t) hkv * cap;

    const int qa_row = ((lane & 16) >> 2) + ((lane & 4) << 1) + (lane & 3);
    const int kb_row = ((lane & 16) >> 2) + (lane & 8) + (lane & 3);
    const int vb_row = lane & 3;
    const int vb_col = (lane & 8) + ((lane & 16) >> 2);
    const bool hi8 = (lane & 8) != 0;
    const bool want_r1 = (lane & 2) != 0;

    const __half* sq_r0 = sQ + (qr_base + 0  + qa_row) * STRIDE;
    const __half* sq_r1 = sQ + (qr_base + 16 + qa_row) * STRIDE;
    const __half* sk_r  = sKV + kb_row * STRIDE;
    const __half* sv_r  = sKV + vb_row * STRIDE + vb_col;

    uint4 kraw0, kraw1;
    __half2 dk2;
    {
        const int j = j_start + kv_r;
        dk2 = __half2half2(__ldg(&ks_h[j]));
        const int8_t* kp = kq_base + (size_t) j * 128;
        kraw0 = *reinterpret_cast<const uint4*>(kp);
        kraw1 = *reinterpret_cast<const uint4*>(kp + 16);
    }

    for (int jb = j_start; jb < j_end; jb += 64) {
        {
            __half* dst = sKV + kv_r * STRIDE + kv_c0;
            const __half2 kb = __hmul2(neg1152, dk2);
            __half2 h2[8];
            i8x4_to_h2x2_fma(kraw0.x, dk2, kb, h2[0], h2[1]); i8x4_to_h2x2_fma(kraw0.y, dk2, kb, h2[2], h2[3]);
            i8x4_to_h2x2_fma(kraw0.z, dk2, kb, h2[4], h2[5]); i8x4_to_h2x2_fma(kraw0.w, dk2, kb, h2[6], h2[7]);
            *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(h2);
            *reinterpret_cast<uint4*>(dst + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
            i8x4_to_h2x2_fma(kraw1.x, dk2, kb, h2[0], h2[1]); i8x4_to_h2x2_fma(kraw1.y, dk2, kb, h2[2], h2[3]);
            i8x4_to_h2x2_fma(kraw1.z, dk2, kb, h2[4], h2[5]); i8x4_to_h2x2_fma(kraw1.w, dk2, kb, h2[6], h2[7]);
            *reinterpret_cast<uint4*>(dst + 16) = *reinterpret_cast<const uint4*>(h2);
            *reinterpret_cast<uint4*>(dst + 24) = *reinterpret_cast<const uint4*>(h2 + 4);
        }
        __syncthreads();

        float fc[2][4][8];
#pragma unroll
        for (int m = 0; m < 2; ++m)
#pragma unroll
            for (int ct = 0; ct < 4; ++ct)
#pragma unroll
                for (int i = 0; i < 8; ++i) fc[m][ct][i] = 0.0f;

        // E8: Issue V HBM prefetch BEFORE the 128-K QK^T HMMA loop so V latency overlaps 256 cycles of HMMA!
        const int jv = jb + kv_r;
        const __half2 dv2 = __half2half2(__ldg(&vs_h[jv]));
        const int8_t* vp = vq_base + (size_t) jv * 128;
        const uint4 vraw0 = *reinterpret_cast<const uint4*>(vp);
        const uint4 vraw1 = *reinterpret_cast<const uint4*>(vp + 16);

#pragma unroll
        for (int k = 0; k < 128; k += 8) {
            const uint4 q0 = *reinterpret_cast<const uint4*>(sq_r0 + k);
            const uint4 q1 = *reinterpret_cast<const uint4*>(sq_r1 + k);
            const uint4 kb0 = *reinterpret_cast<const uint4*>(sk_r + 0  * STRIDE + k);
            const uint4 kb1 = *reinterpret_cast<const uint4*>(sk_r + 16 * STRIDE + k);
            const uint4 kb2 = *reinterpret_cast<const uint4*>(sk_r + 32 * STRIDE + k);
            const uint4 kb3 = *reinterpret_cast<const uint4*>(sk_r + 48 * STRIDE + k);
            mma_rc(fc[0][0], q0.x, q0.y, kb0.x, kb0.y); mma_rc(fc[1][0], q1.x, q1.y, kb0.x, kb0.y);
            mma_rc(fc[0][1], q0.x, q0.y, kb1.x, kb1.y); mma_rc(fc[1][1], q1.x, q1.y, kb1.x, kb1.y);
            mma_rc(fc[0][2], q0.x, q0.y, kb2.x, kb2.y); mma_rc(fc[1][2], q1.x, q1.y, kb2.x, kb2.y);
            mma_rc(fc[0][3], q0.x, q0.y, kb3.x, kb3.y); mma_rc(fc[1][3], q1.x, q1.y, kb3.x, kb3.y);
            mma_rc(fc[0][0], q0.z, q0.w, kb0.z, kb0.w); mma_rc(fc[1][0], q1.z, q1.w, kb0.z, kb0.w);
            mma_rc(fc[0][1], q0.z, q0.w, kb1.z, kb1.w); mma_rc(fc[1][1], q1.z, q1.w, kb1.z, kb1.w);
            mma_rc(fc[0][2], q0.z, q0.w, kb2.z, kb2.w); mma_rc(fc[1][2], q1.z, q1.w, kb2.z, kb2.w);
            mma_rc(fc[0][3], q0.z, q0.w, kb3.z, kb3.w); mma_rc(fc[1][3], q1.z, q1.w, kb3.z, kb3.w);
        }

        uint32_t ph[2][4][4];
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            float mx0 = -1e30f, mx1 = -1e30f;
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) {
                mx0 = fmaxf(mx0, fmaxf(fmaxf(fc[m][ct][0], fc[m][ct][1]), fmaxf(fc[m][ct][4], fc[m][ct][5])));
                mx1 = fmaxf(mx1, fmaxf(fmaxf(fc[m][ct][2], fc[m][ct][3]), fmaxf(fc[m][ct][6], fc[m][ct][7])));
            }
            const float m_new0 = fmaxf(rm0[m], wmma_row_max_16(mx0));
            const float m_new1 = fmaxf(rm1[m], wmma_row_max_16(mx1));
            const float a0 = (rm0[m] > -1e20f) ? ((m_new0 == rm0[m]) ? 1.0f : fast_exp2f_hw(rm0[m] - m_new0)) : 0.0f;
            const float a1 = (rm1[m] > -1e20f) ? ((m_new1 == rm1[m]) ? 1.0f : fast_exp2f_hw(rm1[m] - m_new1)) : 0.0f;
            if (__any_sync(0xffffffffu, a0 != 1.0f || a1 != 1.0f)) {
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    acc[m][n][0] *= a0; acc[m][n][1] *= a0; acc[m][n][4] *= a0; acc[m][n][5] *= a0;
                    acc[m][n][2] *= a1; acc[m][n][3] *= a1; acc[m][n][6] *= a1; acc[m][n][7] *= a1;
                }
            }
            float ps0 = 0.0f, ps1 = 0.0f;
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) {
                const float p00 = fast_exp2f_hw(fc[m][ct][0] - m_new0);
                const float p01 = fast_exp2f_hw(fc[m][ct][1] - m_new0);
                const float p04 = fast_exp2f_hw(fc[m][ct][4] - m_new0);
                const float p05 = fast_exp2f_hw(fc[m][ct][5] - m_new0);
                ps0 += (p00 + p01) + (p04 + p05);
                const float p12 = fast_exp2f_hw(fc[m][ct][2] - m_new1);
                const float p13 = fast_exp2f_hw(fc[m][ct][3] - m_new1);
                const float p16 = fast_exp2f_hw(fc[m][ct][6] - m_new1);
                const float p17 = fast_exp2f_hw(fc[m][ct][7] - m_new1);
                ps1 += (p12 + p13) + (p16 + p17);
                const uint32_t u0 = f2_to_h2u(p00, p01), u1 = f2_to_h2u(p04, p05);
                const uint32_t v0 = f2_to_h2u(p12, p13), v1 = f2_to_h2u(p16, p17);
                const uint32_t r0 = __shfl_xor_sync(0xffffffffu, want_r1 ? u0 : v0, 2);
                const uint32_t r1 = __shfl_xor_sync(0xffffffffu, want_r1 ? u1 : v1, 2);
                ph[m][ct][0] = want_r1 ? r0 : u0; ph[m][ct][1] = want_r1 ? v0 : r0;
                ph[m][ct][2] = want_r1 ? r1 : u1; ph[m][ct][3] = want_r1 ? v1 : r1;
            }
            rm0[m] = m_new0; rl0[m] = rl0[m] * a0 + wmma_row_sum_16(ps0);
            rm1[m] = m_new1; rl1[m] = rl1[m] * a1 + wmma_row_sum_16(ps1);
        }
        __syncthreads();

        {
            __half* dst = sKV + kv_r * STRIDE + kv_c0;
            const __half2 vb = __hmul2(neg1152, dv2);
            __half2 h2[8];
            i8x4_to_h2x2_fma(vraw0.x, dv2, vb, h2[0], h2[1]); i8x4_to_h2x2_fma(vraw0.y, dv2, vb, h2[2], h2[3]);
            i8x4_to_h2x2_fma(vraw0.z, dv2, vb, h2[4], h2[5]); i8x4_to_h2x2_fma(vraw0.w, dv2, vb, h2[6], h2[7]);
            *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(h2);
            *reinterpret_cast<uint4*>(dst + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
            i8x4_to_h2x2_fma(vraw1.x, dv2, vb, h2[0], h2[1]); i8x4_to_h2x2_fma(vraw1.y, dv2, vb, h2[2], h2[3]);
            i8x4_to_h2x2_fma(vraw1.z, dv2, vb, h2[4], h2[5]); i8x4_to_h2x2_fma(vraw1.w, dv2, vb, h2[6], h2[7]);
            *reinterpret_cast<uint4*>(dst + 16) = *reinterpret_cast<const uint4*>(h2);
            *reinterpret_cast<uint4*>(dst + 24) = *reinterpret_cast<const uint4*>(h2 + 4);
        }
        __syncthreads();

        if (jb + 64 < j_end) {
            const int j = jb + 64 + kv_r;
            dk2 = __half2half2(__ldg(&ks_h[j]));
            const int8_t* kp = kq_base + (size_t) j * 128;
            kraw0 = *reinterpret_cast<const uint4*>(kp);
            kraw1 = *reinterpret_cast<const uint4*>(kp + 16);
        }

#pragma unroll
        for (int ct = 0; ct < 4; ++ct) {
            uint32_t a0_0, a0_1, a0_2, a0_3, a0_4, a0_5, a0_6, a0_7;
            uint32_t a1_0, a1_1, a1_2, a1_3, a1_4, a1_5, a1_6, a1_7;
            {
                const uint32_t o0 = __shfl_xor_sync(0xffffffffu, ph[0][ct][0], 8);
                const uint32_t o1 = __shfl_xor_sync(0xffffffffu, ph[0][ct][1], 8);
                const uint32_t o2 = __shfl_xor_sync(0xffffffffu, ph[0][ct][2], 8);
                const uint32_t o3 = __shfl_xor_sync(0xffffffffu, ph[0][ct][3], 8);
                a0_0 = hi8 ? o0 : ph[0][ct][0]; a0_1 = hi8 ? o1 : ph[0][ct][1];
                a0_2 = hi8 ? o2 : ph[0][ct][2]; a0_3 = hi8 ? o3 : ph[0][ct][3];
                a0_4 = hi8 ? ph[0][ct][0] : o0; a0_5 = hi8 ? ph[0][ct][1] : o1;
                a0_6 = hi8 ? ph[0][ct][2] : o2; a0_7 = hi8 ? ph[0][ct][3] : o3;
            }
            {
                const uint32_t o0 = __shfl_xor_sync(0xffffffffu, ph[1][ct][0], 8);
                const uint32_t o1 = __shfl_xor_sync(0xffffffffu, ph[1][ct][1], 8);
                const uint32_t o2 = __shfl_xor_sync(0xffffffffu, ph[1][ct][2], 8);
                const uint32_t o3 = __shfl_xor_sync(0xffffffffu, ph[1][ct][3], 8);
                a1_0 = hi8 ? o0 : ph[1][ct][0]; a1_1 = hi8 ? o1 : ph[1][ct][1];
                a1_2 = hi8 ? o2 : ph[1][ct][2]; a1_3 = hi8 ? o3 : ph[1][ct][3];
                a1_4 = hi8 ? ph[1][ct][0] : o0; a1_5 = hi8 ? ph[1][ct][1] : o1;
                a1_6 = hi8 ? ph[1][ct][2] : o2; a1_7 = hi8 ? ph[1][ct][3] : o3;
            }
            const __half* sv_base = sv_r + (ct * 16) * STRIDE;
            uint2 b[8];
#pragma unroll
            for (int n = 0; n < 8; ++n) b[n] = *reinterpret_cast<const uint2*>(sv_base + 0 * STRIDE + n * 16);
#pragma unroll
            for (int n = 0; n < 8; ++n) { mma_rr(acc[0][n], a0_0, a0_1, b[n].x, b[n].y); mma_rr(acc[1][n], a1_0, a1_1, b[n].x, b[n].y); }
#pragma unroll
            for (int n = 0; n < 8; ++n) b[n] = *reinterpret_cast<const uint2*>(sv_base + 4 * STRIDE + n * 16);
#pragma unroll
            for (int n = 0; n < 8; ++n) { mma_rr(acc[0][n], a0_2, a0_3, b[n].x, b[n].y); mma_rr(acc[1][n], a1_2, a1_3, b[n].x, b[n].y); }
#pragma unroll
            for (int n = 0; n < 8; ++n) b[n] = *reinterpret_cast<const uint2*>(sv_base + 8 * STRIDE + n * 16);
#pragma unroll
            for (int n = 0; n < 8; ++n) { mma_rr(acc[0][n], a0_4, a0_5, b[n].x, b[n].y); mma_rr(acc[1][n], a1_4, a1_5, b[n].x, b[n].y); }
#pragma unroll
            for (int n = 0; n < 8; ++n) b[n] = *reinterpret_cast<const uint2*>(sv_base + 12 * STRIDE + n * 16);
#pragma unroll
            for (int n = 0; n < 8; ++n) { mma_rr(acc[0][n], a0_6, a0_7, b[n].x, b[n].y); mma_rr(acc[1][n], a1_6, a1_7, b[n].x, b[n].y); }
        }
        __syncthreads();
    }

    int acc_r0, acc_r1, acc_c0, acc_c1;
    wmma_acc_layout_16x16(lane, acc_r0, acc_r1, acc_c0, acc_c1);
    const bool is_leader = wmma_row_leader_16(lane);
    float* s_out = reinterpret_cast<float*>(smem_raw);
    float* s_rm = reinterpret_cast<float*>(smem_raw + (256 + 64) * STRIDE * 2 - 2048);
    float* s_rl = s_rm + 256;
    if (is_leader) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            const int qr0 = qr_base + m * 16 + acc_r0, qr1 = qr_base + m * 16 + acc_r1;
            const int tr0 = t0 + qr0, tr1 = t0 + qr1;
            const float ns0 = (tr0 < T) ? (-__ldg(&sink[(size_t) tr0 * n_head + h]) * kLog2e) : 0.0f;
            const float ns1 = (tr1 < T) ? (-__ldg(&sink[(size_t) tr1 * n_head + h]) * kLog2e) : 0.0f;
            s_rm[qr0] = rm0[m] + ns0; s_rl[qr0] = rl0[m];
            s_rm[qr1] = rm1[m] + ns1; s_rl[qr1] = rl1[m];
        }
    }
    __syncthreads();

    for (int w_base = 0; w_base < 8; w_base += 4) {
        if (warp >= w_base && warp < w_base + 4) {
            const int lr0 = (warp - w_base) * 32;
#pragma unroll
            for (int m = 0; m < 2; ++m) {
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    float* dst0 = s_out + (lr0 + m * 16 + acc_r0) * 128 + n * 16;
                    float* dst1 = s_out + (lr0 + m * 16 + acc_r1) * 128 + n * 16;
                    *reinterpret_cast<float2*>(dst0 + acc_c0) = make_float2(acc[m][n][0], acc[m][n][1]);
                    *reinterpret_cast<float2*>(dst0 + acc_c1) = make_float2(acc[m][n][4], acc[m][n][5]);
                    *reinterpret_cast<float2*>(dst1 + acc_c0) = make_float2(acc[m][n][2], acc[m][n][3]);
                    *reinterpret_cast<float2*>(dst1 + acc_c1) = make_float2(acc[m][n][6], acc[m][n][7]);
                }
            }
        }
        __syncthreads();
        for (int idx4 = tid; idx4 < 128 * 32; idx4 += 256) {
            const int r = idx4 >> 5, col = (idx4 & 31) << 2;
            const int qr = w_base * 32 + r, t = t0 + qr;
            if (t < T) {
                const float4 v = reinterpret_cast<const float4*>(s_out)[idx4];
                float* dst = split_out + (((size_t) t * total_splits + sp) * n_head + h) * 130;
                *reinterpret_cast<float2*>(dst + col)     = make_float2(v.x, v.y);
                *reinterpret_cast<float2*>(dst + col + 2) = make_float2(v.z, v.w);
                if (col == 0) { dst[128] = s_rm[qr]; dst[129] = s_rl[qr]; }
            }
        }
        __syncthreads();
    }
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // Ampere / Ada (sm_80+) native ldmatrix.x4 + m16n8k16 path (STACK:0, XOR-swizzled STRIDE=128):
    // sQ (256*128*2 = 65,536 B) | sKV (64*128*2 = 16,384 B) | s_rm/s_rl (2,048 B) = 83,968 B <= 87,040 B caller SMEM!
    constexpr int STRIDE = 128;
    extern __shared__ __align__(16) char smem_raw[];
    __half* sQ  = reinterpret_cast<__half*>(smem_raw);
    __half* sKV = sQ + 256 * STRIDE;  // +65,536 B (16,384 B -> 81,920 B)

    const int hkv = blockIdx.y;
    const int g = blockIdx.x % 3;
    const int qt = blockIdx.x / 3;
    const int h = 3 * hkv + g;
    const int t0 = qt * 256;
    const int sp = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    constexpr float kLog2e = 1.4426950408889634f;
    const float qscale_all = kq_scale * kLog2e;

    const int bulk_qt = bulk_aligned + t0;
    const int n_tiles = bulk_qt >> 6;
    const int t_per_sp = (n_tiles + bulk_splits - 1) / bulk_splits;
    const int j_start = sp * t_per_sp * 64;
    const int j_end = min(bulk_qt, j_start + t_per_sp * 64);
    if (j_start >= bulk_qt) {
        const int t = t0 + tid;
        if (t < T) {
            float* dst = split_out + (((size_t) t * total_splits + sp) * n_head + h) * 130;
            dst[128] = -1e30f;
            dst[129] = 0.0f;
        }
        return;
    }

    // Load 256x128 Q -> XOR-swizzled sQ (col8 ^ (qr & 7)).
    for (int base_r = 0; base_r < 256; base_r += 32) {
        const int qr = base_r + (tid >> 3);
        const int qc0 = (tid & 7) * 16;
        const int col8_0 = (tid & 7) * 2;
        const int col8_1 = col8_0 + 1;
        const int rswz = qr & 7;
        const int t = t0 + qr;
        __half2* d0 = reinterpret_cast<__half2*>(sQ + qr * STRIDE + (col8_0 ^ rswz) * 8);
        __half2* d1 = reinterpret_cast<__half2*>(sQ + qr * STRIDE + (col8_1 ^ rswz) * 8);
        if (t < T) {
            const float* qp = Q + ((size_t) t * n_head + h) * 128 + qc0;
            const float4 v0 = *reinterpret_cast<const float4*>(qp + 0);
            const float4 v1 = *reinterpret_cast<const float4*>(qp + 4);
            const float4 v2 = *reinterpret_cast<const float4*>(qp + 8);
            const float4 v3 = *reinterpret_cast<const float4*>(qp + 12);
            d0[0] = __halves2half2(__float2half(v0.x * qscale_all), __float2half(v0.y * qscale_all));
            d0[1] = __halves2half2(__float2half(v0.z * qscale_all), __float2half(v0.w * qscale_all));
            d0[2] = __halves2half2(__float2half(v1.x * qscale_all), __float2half(v1.y * qscale_all));
            d0[3] = __halves2half2(__float2half(v1.z * qscale_all), __float2half(v1.w * qscale_all));
            d1[0] = __halves2half2(__float2half(v2.x * qscale_all), __float2half(v2.y * qscale_all));
            d1[1] = __halves2half2(__float2half(v2.z * qscale_all), __float2half(v2.w * qscale_all));
            d1[2] = __halves2half2(__float2half(v3.x * qscale_all), __float2half(v3.y * qscale_all));
            d1[3] = __halves2half2(__float2half(v3.z * qscale_all), __float2half(v3.w * qscale_all));
        } else {
            const uint4 z = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(d0) = z;
            *reinterpret_cast<uint4*>(d1) = z;
        }
    }
    __syncthreads();

    auto ldmat_x4 = [](uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
    };
    auto ldmat_x4_t = [](uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
        asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
    };
    auto mma_16816 = [](float d[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    };

    const int qr_base = warp * 32;
    float rm0[2] = {-1e30f, -1e30f}, rm1[2] = {-1e30f, -1e30f};
    float rl0[2] = {0.0f, 0.0f}, rl1[2] = {0.0f, 0.0f};
    float acc[2][8][8];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int n = 0; n < 8; ++n)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[m][n][i] = 0.0f;

    const int kv_r = tid >> 2, kv_c0 = (tid & 3) * 32;
    const size_t hkv_off = (size_t) hkv * (size_t) cap;

    const uint32_t sq0_row_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sQ + (qr_base + 0  + (lane & 15)) * STRIDE));
    const uint32_t sq_c0 = (uint32_t) ((lane >> 4) ^ (lane & 7));
    const int k_row = (((lane >> 4) << 3) | (lane & 7));
    const uint32_t sk_row_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sKV + k_row * STRIDE));
    const uint32_t sk_c0 = (uint32_t) (((lane >> 3) & 1) ^ (lane & 7));
    const uint32_t sv_row_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sKV + (lane & 15) * STRIDE));

    auto store_kv_swz = [&](uint4 raw0, uint4 raw1, __half2 sc2) {
        const int kv_col8_base = (tid & 3) * 4;
        const int kv_rswz = kv_r & 7;
        const __half2 kb = __hmul2(__half2half2(__float2half(-1152.0f)), sc2);
        __half2 h0, h1, h2v, h3;
        i8x4_to_h2x2_fma(raw0.x, sc2, kb, h0, h1);
        i8x4_to_h2x2_fma(raw0.y, sc2, kb, h2v, h3);
        __half2* d0 = reinterpret_cast<__half2*>(sKV + kv_r * STRIDE + ((kv_col8_base + 0) ^ kv_rswz) * 8);
        d0[0] = h0; d0[1] = h1; d0[2] = h2v; d0[3] = h3;
        i8x4_to_h2x2_fma(raw0.z, sc2, kb, h0, h1);
        i8x4_to_h2x2_fma(raw0.w, sc2, kb, h2v, h3);
        __half2* d1 = reinterpret_cast<__half2*>(sKV + kv_r * STRIDE + ((kv_col8_base + 1) ^ kv_rswz) * 8);
        d1[0] = h0; d1[1] = h1; d1[2] = h2v; d1[3] = h3;
        i8x4_to_h2x2_fma(raw1.x, sc2, kb, h0, h1);
        i8x4_to_h2x2_fma(raw1.y, sc2, kb, h2v, h3);
        __half2* d2 = reinterpret_cast<__half2*>(sKV + kv_r * STRIDE + ((kv_col8_base + 2) ^ kv_rswz) * 8);
        d2[0] = h0; d2[1] = h1; d2[2] = h2v; d2[3] = h3;
        i8x4_to_h2x2_fma(raw1.z, sc2, kb, h0, h1);
        i8x4_to_h2x2_fma(raw1.w, sc2, kb, h2v, h3);
        __half2* d3 = reinterpret_cast<__half2*>(sKV + kv_r * STRIDE + ((kv_col8_base + 3) ^ kv_rswz) * 8);
        d3[0] = h0; d3[1] = h1; d3[2] = h2v; d3[3] = h3;
    };

    uint4 kraw0, kraw1;
    __half2 dk2;
    {
        const size_t j_idx = hkv_off + (size_t) (j_start + kv_r);
        dk2 = __half2half2(__ldg(&ks[j_idx]));
        const int8_t* kp = kq + j_idx * 128u + kv_c0;
        kraw0 = *reinterpret_cast<const uint4*>(kp);
        kraw1 = *reinterpret_cast<const uint4*>(kp + 16);
    }

    for (int jb = j_start; jb < j_end; jb += 64) {
        store_kv_swz(kraw0, kraw1, dk2);
        __syncthreads();

        // Prefetch V before QK^T HMMA loop so V HBM latency overlaps QK^T!
        const size_t jv_idx = hkv_off + (size_t) (jb + kv_r);
        const __half2 dv2 = __half2half2(__ldg(&vs[jv_idx]));
        const int8_t* vp = vq + jv_idx * 128u + kv_c0;
        const uint4 vraw0 = *reinterpret_cast<const uint4*>(vp);
        const uint4 vraw1 = *reinterpret_cast<const uint4*>(vp + 16);

        float fc[2][4][8];
#pragma unroll
        for (int m = 0; m < 2; ++m)
#pragma unroll
            for (int ct = 0; ct < 4; ++ct)
#pragma unroll
                for (int i = 0; i < 8; ++i) fc[m][ct][i] = 0.0f;

        {
            constexpr uint32_t kSkTileB = 16u * STRIDE * 2u;
#pragma unroll 1
            for (int ki = 0; ki < 8; ++ki) {
                const uint32_t q_addr0 = sq0_row_addr + ((((uint32_t) ki << 1) ^ sq_c0) << 4);
                const uint32_t k_off = ((((uint32_t) ki << 1) ^ sk_c0) << 4);
                uint32_t qa0, qa1, qa2, qa3, qb0, qb1, qb2, qb3;
                ldmat_x4(qa0, qa1, qa2, qa3, q_addr0);
                ldmat_x4(qb0, qb1, qb2, qb3, q_addr0 + 4096u);
#pragma unroll
                for (int ct = 0; ct < 4; ++ct) {
                    uint32_t k0, k1, k2, k3;
                    ldmat_x4(k0, k1, k2, k3, sk_row_addr + (uint32_t) ct * kSkTileB + k_off);
                    mma_16816(fc[0][ct],     qa0, qa1, qa2, qa3, k0, k1);
                    mma_16816(fc[0][ct] + 4, qa0, qa1, qa2, qa3, k2, k3);
                    mma_16816(fc[1][ct],     qb0, qb1, qb2, qb3, k0, k1);
                    mma_16816(fc[1][ct] + 4, qb0, qb1, qb2, qb3, k2, k3);
                }
            }
        }

        // In-place online softmax + FP16 pack directly into fc[m][ct][0..3] (ZERO fa_p, STACK:0!).
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            float mx0 = -1e30f, mx1 = -1e30f;
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) {
                mx0 = fmaxf(mx0, fmaxf(fmaxf(fc[m][ct][0], fc[m][ct][1]), fmaxf(fc[m][ct][4], fc[m][ct][5])));
                mx1 = fmaxf(mx1, fmaxf(fmaxf(fc[m][ct][2], fc[m][ct][3]), fmaxf(fc[m][ct][6], fc[m][ct][7])));
            }
            const float m_new0 = fmaxf(rm0[m], wmma_row_max_16(mx0));
            const float m_new1 = fmaxf(rm1[m], wmma_row_max_16(mx1));
            const float a0 = (rm0[m] > -1e20f) ? ((m_new0 == rm0[m]) ? 1.0f : fast_exp2f_hw(rm0[m] - m_new0)) : 0.0f;
            const float a1 = (rm1[m] > -1e20f) ? ((m_new1 == rm1[m]) ? 1.0f : fast_exp2f_hw(rm1[m] - m_new1)) : 0.0f;
            if (__any_sync(0xffffffffu, a0 != 1.0f || a1 != 1.0f)) {
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    acc[m][n][0] *= a0; acc[m][n][1] *= a0; acc[m][n][4] *= a0; acc[m][n][5] *= a0;
                    acc[m][n][2] *= a1; acc[m][n][3] *= a1; acc[m][n][6] *= a1; acc[m][n][7] *= a1;
                }
            }
            float ps0 = 0.0f, ps1 = 0.0f;
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) {
                const float p00 = fast_exp2f_hw(fc[m][ct][0] - m_new0);
                const float p01 = fast_exp2f_hw(fc[m][ct][1] - m_new0);
                const float p12 = fast_exp2f_hw(fc[m][ct][2] - m_new1);
                const float p13 = fast_exp2f_hw(fc[m][ct][3] - m_new1);
                const float p04 = fast_exp2f_hw(fc[m][ct][4] - m_new0);
                const float p05 = fast_exp2f_hw(fc[m][ct][5] - m_new0);
                const float p16 = fast_exp2f_hw(fc[m][ct][6] - m_new1);
                const float p17 = fast_exp2f_hw(fc[m][ct][7] - m_new1);
                ps0 += (p00 + p01) + (p04 + p05);
                ps1 += (p12 + p13) + (p16 + p17);
                fc[m][ct][0] = __uint_as_float(f2_to_h2u(p00, p01));
                fc[m][ct][1] = __uint_as_float(f2_to_h2u(p12, p13));
                fc[m][ct][2] = __uint_as_float(f2_to_h2u(p04, p05));
                fc[m][ct][3] = __uint_as_float(f2_to_h2u(p16, p17));
            }
            rm0[m] = m_new0; rl0[m] = rl0[m] * a0 + wmma_row_sum_16(ps0);
            rm1[m] = m_new1; rl1[m] = rl1[m] * a1 + wmma_row_sum_16(ps1);
        }
        __syncthreads();

        store_kv_swz(vraw0, vraw1, dv2);
        __syncthreads();

        if (jb + 64 < j_end) {
            const size_t j_idx = hkv_off + (size_t) (jb + 64 + kv_r);
            dk2 = __half2half2(__ldg(&ks[j_idx]));
            const int8_t* kp = kq + j_idx * 128u + kv_c0;
            kraw0 = *reinterpret_cast<const uint4*>(kp);
            kraw1 = *reinterpret_cast<const uint4*>(kp + 16);
        }

        // Native P*V with double-buffered LDSM.x4.T.
        {
            constexpr uint32_t kSvTileB = 16u * STRIDE * 2u;
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) {
                const uint32_t p0_0 = __float_as_uint(fc[0][ct][0]), p0_1 = __float_as_uint(fc[0][ct][1]);
                const uint32_t p0_2 = __float_as_uint(fc[0][ct][2]), p0_3 = __float_as_uint(fc[0][ct][3]);
                const uint32_t p1_0 = __float_as_uint(fc[1][ct][0]), p1_1 = __float_as_uint(fc[1][ct][1]);
                const uint32_t p1_2 = __float_as_uint(fc[1][ct][2]), p1_3 = __float_as_uint(fc[1][ct][3]);
                const uint32_t sv_ct = sv_row_addr + (uint32_t) ct * kSvTileB;
                uint32_t b0_0, b0_1, b0_2, b0_3;
                uint32_t b1_0, b1_1, b1_2, b1_3;
                ldmat_x4_t(b0_0, b0_1, b0_2, b0_3, sv_ct + (sq_c0 << 4));
#pragma unroll
                for (int n = 0; n < 8; n += 2) {
                    ldmat_x4_t(b1_0, b1_1, b1_2, b1_3, sv_ct + ((((uint32_t) (n + 1) << 1) ^ sq_c0) << 4));
                    mma_16816(acc[0][n],     p0_0, p0_1, p0_2, p0_3, b0_0, b0_1);
                    mma_16816(acc[0][n] + 4, p0_0, p0_1, p0_2, p0_3, b0_2, b0_3);
                    mma_16816(acc[1][n],     p1_0, p1_1, p1_2, p1_3, b0_0, b0_1);
                    mma_16816(acc[1][n] + 4, p1_0, p1_1, p1_2, p1_3, b0_2, b0_3);
                    if (n < 6) {
                        ldmat_x4_t(b0_0, b0_1, b0_2, b0_3, sv_ct + ((((uint32_t) (n + 2) << 1) ^ sq_c0) << 4));
                    }
                    mma_16816(acc[0][n + 1],     p0_0, p0_1, p0_2, p0_3, b1_0, b1_1);
                    mma_16816(acc[0][n + 1] + 4, p0_0, p0_1, p0_2, p0_3, b1_2, b1_3);
                    mma_16816(acc[1][n + 1],     p1_0, p1_1, p1_2, p1_3, b1_0, b1_1);
                    mma_16816(acc[1][n + 1] + 4, p1_0, p1_1, p1_2, p1_3, b1_2, b1_3);
                }
            }
        }
        __syncthreads();
    }

    const int acc_r0 = lane >> 2;
    const int acc_r1 = acc_r0 + 8;
    const int acc_c0 = (lane & 3) * 2;
    const int acc_c1 = acc_c0 + 8;
    const bool is_leader = wmma_row_leader_16(lane);
    float* s_out = reinterpret_cast<float*>(smem_raw);
    float* s_rm = reinterpret_cast<float*>(smem_raw + 81920);
    float* s_rl = s_rm + 256;
    if (is_leader) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            const int qr0 = qr_base + m * 16 + acc_r0, qr1 = qr_base + m * 16 + acc_r1;
            const int tr0 = t0 + qr0, tr1 = t0 + qr1;
            const float ns0 = (tr0 < T) ? (-__ldg(&sink[(size_t) tr0 * n_head + h]) * kLog2e) : 0.0f;
            const float ns1 = (tr1 < T) ? (-__ldg(&sink[(size_t) tr1 * n_head + h]) * kLog2e) : 0.0f;
            s_rm[qr0] = rm0[m] + ns0; s_rl[qr0] = rl0[m];
            s_rm[qr1] = rm1[m] + ns1; s_rl[qr1] = rl1[m];
        }
    }
    __syncthreads();

    for (int w_base = 0; w_base < 8; w_base += 4) {
        if (warp >= w_base && warp < w_base + 4) {
            const int lr0 = (warp - w_base) * 32;
#pragma unroll
            for (int m = 0; m < 2; ++m) {
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    float* dst0 = s_out + (lr0 + m * 16 + acc_r0) * 128 + n * 16;
                    float* dst1 = s_out + (lr0 + m * 16 + acc_r1) * 128 + n * 16;
                    *reinterpret_cast<float2*>(dst0 + acc_c0) = make_float2(acc[m][n][0], acc[m][n][1]);
                    *reinterpret_cast<float2*>(dst0 + acc_c1) = make_float2(acc[m][n][4], acc[m][n][5]);
                    *reinterpret_cast<float2*>(dst1 + acc_c0) = make_float2(acc[m][n][2], acc[m][n][3]);
                    *reinterpret_cast<float2*>(dst1 + acc_c1) = make_float2(acc[m][n][6], acc[m][n][7]);
                }
            }
        }
        __syncthreads();
        for (int idx4 = tid; idx4 < 128 * 32; idx4 += 256) {
            const int r = idx4 >> 5, col = (idx4 & 31) << 2;
            const int qr = w_base * 32 + r, t = t0 + qr;
            if (t < T) {
                const float4 v = reinterpret_cast<const float4*>(s_out)[idx4];
                float* dst = split_out + (((size_t) t * total_splits + sp) * n_head + h) * 130;
                *reinterpret_cast<float2*>(dst + col)     = make_float2(v.x, v.y);
                *reinterpret_cast<float2*>(dst + col + 2) = make_float2(v.z, v.w);
                if (col == 0) { dst[128] = s_rm[qr]; dst[129] = s_rl[qr]; }
            }
        }
        __syncthreads();
    }
#elif !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750
    using namespace nvcuda;
    extern __shared__ __align__(16) char smem_raw[];
    // 87,040 B <= 98,304 B: sQ (256*136*2 = 69,632 B) | sKV (64*136*2 = 17,408 B). ZERO sP!
    __half* sQ = reinterpret_cast<__half*>(smem_raw);
    __half* sKV = sQ + 256 * 136;

    const int hkv = blockIdx.y;
    const int g = blockIdx.x % 3;
    const int qt = blockIdx.x / 3;
    const int h = 3 * hkv + g;
    const int t0 = qt * 256;
    const int sp = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    constexpr float kLog2e = 1.4426950408889634f;
    const float qscale_all = kq_scale * kLog2e;

    // E8: Per-qt bulk boundary (bulk_aligned + t0) shifts qt=1..7 bulk boundary forward by qt*256 tokens (2.4x less tail)!
    const int bulk_qt = bulk_aligned + t0;
    const int n_tiles = bulk_qt >> 6;
    const int t_per_sp = (n_tiles + bulk_splits - 1) / bulk_splits;
    const int j_start = sp * t_per_sp * 64;
    const int j_end = min(bulk_qt, j_start + t_per_sp * 64);
    if (j_start >= bulk_qt) {
        const int t = t0 + tid;
        if (t < T) {
            float* dst = split_out + (((size_t) t * total_splits + sp) * n_head + h) * 130;
            dst[128] = -1e30f;
            dst[129] = 0.0f;
        }
        return;
    }

    for (int base_r = 0; base_r < 256; base_r += 32) {
        const int qr = base_r + (tid >> 3), qc0 = (tid & 7) * 16;
        const int t = t0 + qr;
        __half2* dst2 = reinterpret_cast<__half2*>(sQ + qr * 136 + qc0);
        if (t < T) {
            const float* qp = Q + ((size_t) t * n_head + h) * 128 + qc0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = *reinterpret_cast<const float4*>(qp + i * 4);
                dst2[i * 2 + 0] = __halves2half2(__float2half(v.x * qscale_all), __float2half(v.y * qscale_all));
                dst2[i * 2 + 1] = __halves2half2(__float2half(v.z * qscale_all), __float2half(v.w * qscale_all));
            }
        } else {
            const __half2 z2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));
#pragma unroll
            for (int i = 0; i < 8; ++i) dst2[i] = z2;
        }
    }
    __syncthreads();

    const int qr_base = warp * 32;
    float rm0[2] = {-1e30f, -1e30f}, rm1[2] = {-1e30f, -1e30f};
    float rl0[2] = {0.0f, 0.0f}, rl1[2] = {0.0f, 0.0f};
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][8];
#pragma unroll
    for (int m = 0; m < 2; ++m) {
#pragma unroll
        for (int n = 0; n < 8; ++n) wmma::fill_fragment(acc[m][n], 0.0f);
    }

    const int kv_r = tid >> 2, kv_c0 = (tid & 3) * 32;
    const __half2 neg1152 = __half2half2(__float2half(-1152.0f));
    const int8_t* kq_h = kq + (size_t) hkv * cap * 128 + kv_c0;
    const int8_t* vq_h = vq + (size_t) hkv * cap * 128 + kv_c0;
    const __half* ks_h = ks + (size_t) hkv * cap;
    const __half* vs_h = vs + (size_t) hkv * cap;

    uint4 kraw0, kraw1;
    __half2 dk2;
    {
        const int j = j_start + kv_r;
        dk2 = __half2half2(__ldg(&ks_h[j]));
        const int8_t* kp = kq_h + (size_t) j * 128;
        kraw0 = *reinterpret_cast<const uint4*>(kp);
        kraw1 = *reinterpret_cast<const uint4*>(kp + 16);
    }

    for (int jb = j_start; jb < j_end; jb += 64) {
        {
            __half* dst = sKV + kv_r * 136 + kv_c0;
            const __half2 kb = __hmul2(neg1152, dk2);
            __half2 h2[8];
            i8x4_to_h2x2_fma(kraw0.x, dk2, kb, h2[0], h2[1]); i8x4_to_h2x2_fma(kraw0.y, dk2, kb, h2[2], h2[3]);
            i8x4_to_h2x2_fma(kraw0.z, dk2, kb, h2[4], h2[5]); i8x4_to_h2x2_fma(kraw0.w, dk2, kb, h2[6], h2[7]);
            *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(h2);
            *reinterpret_cast<uint4*>(dst + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
            i8x4_to_h2x2_fma(kraw1.x, dk2, kb, h2[0], h2[1]); i8x4_to_h2x2_fma(kraw1.y, dk2, kb, h2[2], h2[3]);
            i8x4_to_h2x2_fma(kraw1.z, dk2, kb, h2[4], h2[5]); i8x4_to_h2x2_fma(kraw1.w, dk2, kb, h2[6], h2[7]);
            *reinterpret_cast<uint4*>(dst + 16) = *reinterpret_cast<const uint4*>(h2);
            *reinterpret_cast<uint4*>(dst + 24) = *reinterpret_cast<const uint4*>(h2 + 4);
        }
        __syncthreads();

        uint4 vraw0, vraw1;
        __half2 dv2;
        {
            const int j = jb + kv_r;
            dv2 = __half2half2(__ldg(&vs_h[j]));
            const int8_t* vp = vq_h + (size_t) j * 128;
            vraw0 = *reinterpret_cast<const uint4*>(vp);
            vraw1 = *reinterpret_cast<const uint4*>(vp + 16);
        }

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc[2][4];
#pragma unroll
        for (int m = 0; m < 2; ++m) {
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) wmma::fill_fragment(fc[m][ct], 0.0f);
        }
#pragma unroll
        for (int k = 0; k < 128; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa0, fa1;
            wmma::load_matrix_sync(fa0, sQ + (qr_base + 0) * 136 + k, 136);
            wmma::load_matrix_sync(fa1, sQ + (qr_base + 16) * 136 + k, 136);
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                wmma::load_matrix_sync(fb, sKV + (ct * 16) * 136 + k, 136);
                wmma::mma_sync(fc[0][ct], fa0, fb, fc[0][ct]);
                wmma::mma_sync(fc[1][ct], fa1, fb, fc[1][ct]);
            }
        }
        __syncthreads();

        {
            __half* dst = sKV + kv_r * 136 + kv_c0;
            const __half2 vb = __hmul2(neg1152, dv2);
            __half2 h2[8];
            i8x4_to_h2x2_fma(vraw0.x, dv2, vb, h2[0], h2[1]); i8x4_to_h2x2_fma(vraw0.y, dv2, vb, h2[2], h2[3]);
            i8x4_to_h2x2_fma(vraw0.z, dv2, vb, h2[4], h2[5]); i8x4_to_h2x2_fma(vraw0.w, dv2, vb, h2[6], h2[7]);
            *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(h2);
            *reinterpret_cast<uint4*>(dst + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
            i8x4_to_h2x2_fma(vraw1.x, dv2, vb, h2[0], h2[1]); i8x4_to_h2x2_fma(vraw1.y, dv2, vb, h2[2], h2[3]);
            i8x4_to_h2x2_fma(vraw1.z, dv2, vb, h2[4], h2[5]); i8x4_to_h2x2_fma(vraw1.w, dv2, vb, h2[6], h2[7]);
            *reinterpret_cast<uint4*>(dst + 16) = *reinterpret_cast<const uint4*>(h2);
            *reinterpret_cast<uint4*>(dst + 24) = *reinterpret_cast<const uint4*>(h2 + 4);
        }

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa_p[2][4];
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            float mx0 = -1e30f, mx1 = -1e30f;
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) {
                mx0 = fmaxf(mx0, fmaxf(fmaxf(fc[m][ct].x[0], fc[m][ct].x[1]), fmaxf(fc[m][ct].x[4], fc[m][ct].x[5])));
                mx1 = fmaxf(mx1, fmaxf(fmaxf(fc[m][ct].x[2], fc[m][ct].x[3]), fmaxf(fc[m][ct].x[6], fc[m][ct].x[7])));
            }
            const float m_new0 = fmaxf(rm0[m], wmma_row_max_16(mx0));
            const float m_new1 = fmaxf(rm1[m], wmma_row_max_16(mx1));
            const float a0 = (rm0[m] > -1e20f) ? ((m_new0 == rm0[m]) ? 1.0f : fast_exp2f_hw(rm0[m] - m_new0)) : 0.0f;
            const float a1 = (rm1[m] > -1e20f) ? ((m_new1 == rm1[m]) ? 1.0f : fast_exp2f_hw(rm1[m] - m_new1)) : 0.0f;
            if (__any_sync(0xffffffffu, a0 != 1.0f || a1 != 1.0f)) {
#pragma unroll
                for (int n = 0; n < 8; ++n) wmma_rescale_acc(acc[m][n], a0, a1);
            }
            float ps0 = 0.0f, ps1 = 0.0f;
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) wmma_acc_to_matA_16x16<false>(fc[m][ct], m_new0, m_new1, ps0, ps1, fa_p[m][ct], lane);
            rm0[m] = m_new0; rl0[m] = rl0[m] * a0 + wmma_row_sum_16(ps0);
            rm1[m] = m_new1; rl1[m] = rl1[m] * a1 + wmma_row_sum_16(ps1);
        }
        __syncthreads();

        if (jb + 64 < j_end) {
            const int j = jb + 64 + kv_r;
            dk2 = __half2half2(__ldg(&ks_h[j]));
            const int8_t* kp = kq_h + (size_t) j * 128;
            kraw0 = *reinterpret_cast<const uint4*>(kp);
            kraw1 = *reinterpret_cast<const uint4*>(kp + 16);
        }
#pragma unroll
        for (int ct = 0; ct < 4; ++ct) {
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
                wmma::load_matrix_sync(fb, sKV + (ct * 16) * 136 + n * 16, 136);
                wmma::mma_sync(acc[0][n], fa_p[0][ct], fb, acc[0][n]);
                wmma::mma_sync(acc[1][n], fa_p[1][ct], fb, acc[1][n]);
            }
        }
        __syncthreads();
    }

    int acc_r0, acc_r1, acc_c0, acc_c1;
    wmma_acc_layout_16x16(lane, acc_r0, acc_r1, acc_c0, acc_c1);
    const bool is_leader = wmma_row_leader_16(lane);
    float* s_out = reinterpret_cast<float*>(smem_raw);
    float* s_rm = reinterpret_cast<float*>(smem_raw + 87040 - 2048);
    float* s_rl = s_rm + 256;
    if (is_leader) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            const int qr0 = qr_base + m * 16 + acc_r0, qr1 = qr_base + m * 16 + acc_r1;
            const int tr0 = t0 + qr0, tr1 = t0 + qr1;
            const float ns0 = (tr0 < T) ? (-__ldg(&sink[(size_t) tr0 * n_head + h]) * kLog2e) : 0.0f;
            const float ns1 = (tr1 < T) ? (-__ldg(&sink[(size_t) tr1 * n_head + h]) * kLog2e) : 0.0f;
            s_rm[qr0] = rm0[m] + ns0; s_rl[qr0] = rl0[m];
            s_rm[qr1] = rm1[m] + ns1; s_rl[qr1] = rl1[m];
        }
    }
    __syncthreads();

    for (int w_base = 0; w_base < 8; w_base += 4) {
        if (warp >= w_base && warp < w_base + 4) {
            const int lr0 = (warp - w_base) * 32;
#pragma unroll
            for (int m = 0; m < 2; ++m) {
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    wmma::store_matrix_sync(s_out + (lr0 + m * 16) * 128 + n * 16, acc[m][n], 128, wmma::mem_row_major);
                }
            }
        }
        __syncthreads();
        for (int idx4 = tid; idx4 < 128 * 32; idx4 += 256) {
            const int r = idx4 >> 5, col = (idx4 & 31) << 2;
            const int qr = w_base * 32 + r, t = t0 + qr;
            if (t < T) {
                const float4 v = reinterpret_cast<const float4*>(s_out)[idx4];
                float* dst = split_out + (((size_t) t * total_splits + sp) * n_head + h) * 130;
                *reinterpret_cast<float2*>(dst + col)     = make_float2(v.x, v.y);
                *reinterpret_cast<float2*>(dst + col + 2) = make_float2(v.z, v.w);
                if (col == 0) { dst[128] = s_rm[qr]; dst[129] = s_rl[qr]; }
            }
        }
        __syncthreads();
    }
#endif
}

/// Companion tail split kernel for k_attn_q8_gqa3_pp_fa2_256_bulk:
/// Covers [bulk_aligned + (qt128>>1)*256, pos0[0] + min(t0+127, T-1) + 1) (causal + rel_tab + sink 1.0)
/// and writes to split index tail_sp = total_splits - 1.
/// 8-warp (256-thr, 16 Q-rows/warp = 128 Q-rows/CTA -> 255 regs/thr -> STACK:0 guaranteed!) with 2x finer causal tiles.
/// Native HMMA.884 on Volta (55,296 B) & m16n8k16+ldmatrix on Ampere/Ada (52,224 B).
__global__ void __launch_bounds__(256, 1) k_attn_q8_gqa3_pp_fa2_256_tail(
        const float* __restrict__ Q,
        const int8_t* __restrict__ kq, const __half* __restrict__ ks,
        const int8_t* __restrict__ vq, const __half* __restrict__ vs,
        const float* __restrict__ rel_tab, const float* __restrict__ sink,
        float* __restrict__ split_out,
        int n_head, int n_head_kv, int cap, int n_ext, int bulk_aligned,
        const int32_t* __restrict__ pos0, float kq_scale, int T, int tail_sp, int total_splits) {
#if defined(__CUDA_ARCH__) && ((__CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750) || __CUDA_ARCH__ >= 800)
#if __CUDA_ARCH__ < 750
    constexpr int STRIDE = 144;
    constexpr int SMEM_BYTES = (128 + 64) * 144 * 2;  // 55,296 B
#else
    constexpr int STRIDE = 136;
    constexpr int SMEM_BYTES = (128 + 64) * 136 * 2;  // 52,224 B
#endif
    extern __shared__ __align__(16) char smem_raw[];
    __half* sQ = reinterpret_cast<__half*>(smem_raw);
    __half* sKV = sQ + 128 * STRIDE;

    const int hkv = blockIdx.y;
    const int g = blockIdx.x % 3;
    const int qt128 = blockIdx.x / 3;
    const int h = 3 * hkv + g;
    const int t0 = qt128 * 128;
    const int jb_start = bulk_aligned + (qt128 >> 1) * 256;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    constexpr float kLog2e = 1.4426950408889634f;
    const float qscale_all = kq_scale * kLog2e;

    const int p0 = pos0[0];
    const int t_last = min(t0 + 127, T - 1);
    const int j_end = p0 + t_last + 1;

    // 256 threads load 128x128 Q -> sQ in 4 iterations (32 rows/iter, 8 threads/row, 16 elems/thread).
    for (int base_r = 0; base_r < 128; base_r += 32) {
        const int qr = base_r + (tid >> 3), qc0 = (tid & 7) * 16;
        const int t = t0 + qr;
        __half2* dst2 = reinterpret_cast<__half2*>(sQ + qr * STRIDE + qc0);
        if (t < T) {
            const float* qp = Q + ((size_t) t * n_head + h) * 128 + qc0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = *reinterpret_cast<const float4*>(qp + i * 4);
                dst2[i * 2 + 0] = __halves2half2(__float2half(v.x * qscale_all), __float2half(v.y * qscale_all));
                dst2[i * 2 + 1] = __halves2half2(__float2half(v.z * qscale_all), __float2half(v.w * qscale_all));
            }
        } else {
            const __half2 z2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));
#pragma unroll
            for (int i = 0; i < 8; ++i) dst2[i] = z2;
        }
    }
    __syncthreads();

#if __CUDA_ARCH__ < 750
    auto mma_rc = [](float d[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, {%0, %1, %2, %3, %4, %5, %6, %7};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
            : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
    };
    auto mma_rr = [](float d[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
            "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, {%0, %1, %2, %3, %4, %5, %6, %7};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
            : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
    };
    const int qa_row = ((lane & 16) >> 2) + ((lane & 4) << 1) + (lane & 3);
    const int kb_row = ((lane & 16) >> 2) + (lane & 8) + (lane & 3);
    const int vb_row = lane & 3;
    const int vb_col = (lane & 8) + ((lane & 16) >> 2);
    const bool hi8 = (lane & 8) != 0;
    const bool want_r1 = (lane & 2) != 0;
#else
    auto ldmat_x4 = [](uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
        const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(p));
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
    };
    auto ldmat_x4_t = [](uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
        const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(p));
        asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
    };
    auto mma_16816 = [](float d[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    };
    const int lda_r = lane & 15, lda_c = (lane >> 4) << 3;
    const int ldb_r = ((lane >> 4) << 3) | (lane & 7), ldb_c = ((lane >> 3) & 1) << 3;
#endif

    int acc_r0, acc_r1, acc_c0, acc_c1;
    wmma_acc_layout_16x16(lane, acc_r0, acc_r1, acc_c0, acc_c1);
    const bool is_leader = wmma_row_leader_16(lane);
    const int qr_base = warp * 16;
    const int qr0 = qr_base + acc_r0, qr1 = qr_base + acc_r1;
    float rm0 = 0.0f, rm1 = 0.0f;
    float rl0 = 1.0f, rl1 = 1.0f;
    float acc[8][8];
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[n][i] = 0.0f;

    const int kv_r = tid >> 2, kv_c0 = (tid & 3) * 32;
    const __half2 neg1152 = __half2half2(__float2half(-1152.0f));
    const int8_t* kq_h = kq + (size_t) hkv * cap * 128 + kv_c0;
    const int8_t* vq_h = vq + (size_t) hkv * cap * 128 + kv_c0;
    const __half* ks_h = ks + (size_t) hkv * cap;
    const __half* vs_h = vs + (size_t) hkv * cap;

#if __CUDA_ARCH__ < 750
    const int sq_off = (qr_base + qa_row) * STRIDE;
    const int sk_off = kb_row * STRIDE;
    const int sv_off = vb_row * STRIDE + vb_col;
#else
    const int sq_off = (qr_base + lda_r) * STRIDE + lda_c;
    const int sk_off = ldb_r * STRIDE + ldb_c;
    const int sv_off = lda_r * STRIDE + lda_c;
#endif

    auto mask_bias = [&](float s, int qr, int j) -> float {
        const int tr = t0 + qr;
        const int pos_q = p0 + tr;
        if (tr >= T || j > pos_q || j >= j_end) return -1e30f;
        const int d = pos_q - j;
        const float b = (d < n_ext)
            ? (__ldg(&rel_tab[((size_t) tr * n_head + h) * n_ext + d]) * kLog2e)
            : (-__ldg(&sink[(size_t) tr * n_head + h]) * kLog2e);
        return s + b;
    };

    for (int jb = jb_start; jb < j_end; jb += 64) {
        // 256 threads load 64x128 K (4 threads/row x 32 bytes/thread = 2 uint4).
        {
            const int j = jb + kv_r;
            __half* dst = sKV + kv_r * STRIDE + kv_c0;
            if (j < j_end) {
                const __half2 dk2 = __half2half2(__ldg(&ks_h[j]));
                const __half2 kb = __hmul2(neg1152, dk2);
                const int8_t* kp = kq_h + (size_t) j * 128;
                const uint4 raw0 = *reinterpret_cast<const uint4*>(kp);
                const uint4 raw1 = *reinterpret_cast<const uint4*>(kp + 16);
                __half2 h2[8];
                i8x4_to_h2x2_fma(raw0.x, dk2, kb, h2[0], h2[1]); i8x4_to_h2x2_fma(raw0.y, dk2, kb, h2[2], h2[3]);
                i8x4_to_h2x2_fma(raw0.z, dk2, kb, h2[4], h2[5]); i8x4_to_h2x2_fma(raw0.w, dk2, kb, h2[6], h2[7]);
                *reinterpret_cast<uint4*>(dst)      = *reinterpret_cast<const uint4*>(h2);
                *reinterpret_cast<uint4*>(dst + 8)  = *reinterpret_cast<const uint4*>(h2 + 4);
                i8x4_to_h2x2_fma(raw1.x, dk2, kb, h2[0], h2[1]); i8x4_to_h2x2_fma(raw1.y, dk2, kb, h2[2], h2[3]);
                i8x4_to_h2x2_fma(raw1.z, dk2, kb, h2[4], h2[5]); i8x4_to_h2x2_fma(raw1.w, dk2, kb, h2[6], h2[7]);
                *reinterpret_cast<uint4*>(dst + 16) = *reinterpret_cast<const uint4*>(h2);
                *reinterpret_cast<uint4*>(dst + 24) = *reinterpret_cast<const uint4*>(h2 + 4);
            } else {
                const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int i = 0; i < 4; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
            }
        }
        __syncthreads();

        float fc[4][8];
#pragma unroll
        for (int ct = 0; ct < 4; ++ct)
#pragma unroll
            for (int i = 0; i < 8; ++i) fc[ct][i] = 0.0f;

#if __CUDA_ARCH__ < 750
#pragma unroll 1
        for (int k = 0; k < 128; k += 8) {
            const uint4 q0 = *reinterpret_cast<const uint4*>(sQ + sq_off + k);
            const uint4 kb0 = *reinterpret_cast<const uint4*>(sKV + sk_off + 0  * STRIDE + k);
            const uint4 kb1 = *reinterpret_cast<const uint4*>(sKV + sk_off + 16 * STRIDE + k);
            const uint4 kb2 = *reinterpret_cast<const uint4*>(sKV + sk_off + 32 * STRIDE + k);
            const uint4 kb3 = *reinterpret_cast<const uint4*>(sKV + sk_off + 48 * STRIDE + k);
            mma_rc(fc[0], q0.x, q0.y, kb0.x, kb0.y); mma_rc(fc[1], q0.x, q0.y, kb1.x, kb1.y);
            mma_rc(fc[2], q0.x, q0.y, kb2.x, kb2.y); mma_rc(fc[3], q0.x, q0.y, kb3.x, kb3.y);
            mma_rc(fc[0], q0.z, q0.w, kb0.z, kb0.w); mma_rc(fc[1], q0.z, q0.w, kb1.z, kb1.w);
            mma_rc(fc[2], q0.z, q0.w, kb2.z, kb2.w); mma_rc(fc[3], q0.z, q0.w, kb3.z, kb3.w);
        }
#else
#pragma unroll 1
        for (int k = 0; k < 128; k += 16) {
            uint32_t q0_0, q0_1, q0_2, q0_3;
            ldmat_x4(q0_0, q0_1, q0_2, q0_3, sQ + sq_off + k);
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) {
                uint32_t b0, b1, b2, b3;
                ldmat_x4(b0, b1, b2, b3, sKV + sk_off + (ct * 16) * STRIDE + k);
                mma_16816(fc[ct],     q0_0, q0_1, q0_2, q0_3, b0, b1);
                mma_16816(fc[ct] + 4, q0_0, q0_1, q0_2, q0_3, b2, b3);
            }
        }
#endif
        __syncthreads();

        {
            const int j = jb + kv_r;
            __half* dst = sKV + kv_r * STRIDE + kv_c0;
            if (j < j_end) {
                const __half2 dv2 = __half2half2(__ldg(&vs_h[j]));
                const __half2 vb = __hmul2(neg1152, dv2);
                const int8_t* vp = vq_h + (size_t) j * 128;
                const uint4 raw0 = *reinterpret_cast<const uint4*>(vp);
                const uint4 raw1 = *reinterpret_cast<const uint4*>(vp + 16);
                __half2 h2[8];
                i8x4_to_h2x2_fma(raw0.x, dv2, vb, h2[0], h2[1]); i8x4_to_h2x2_fma(raw0.y, dv2, vb, h2[2], h2[3]);
                i8x4_to_h2x2_fma(raw0.z, dv2, vb, h2[4], h2[5]); i8x4_to_h2x2_fma(raw0.w, dv2, vb, h2[6], h2[7]);
                *reinterpret_cast<uint4*>(dst)      = *reinterpret_cast<const uint4*>(h2);
                *reinterpret_cast<uint4*>(dst + 8)  = *reinterpret_cast<const uint4*>(h2 + 4);
                i8x4_to_h2x2_fma(raw1.x, dv2, vb, h2[0], h2[1]); i8x4_to_h2x2_fma(raw1.y, dv2, vb, h2[2], h2[3]);
                i8x4_to_h2x2_fma(raw1.z, dv2, vb, h2[4], h2[5]); i8x4_to_h2x2_fma(raw1.w, dv2, vb, h2[6], h2[7]);
                *reinterpret_cast<uint4*>(dst + 16) = *reinterpret_cast<const uint4*>(h2);
                *reinterpret_cast<uint4*>(dst + 24) = *reinterpret_cast<const uint4*>(h2 + 4);
            } else {
                const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int i = 0; i < 4; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
            }
        }

        {
            float mx0 = -1e30f, mx1 = -1e30f;
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) {
                const int j0 = jb + ct * 16 + acc_c0, j1 = jb + ct * 16 + acc_c1;
                fc[ct][0] = mask_bias(fc[ct][0], qr0, j0);
                fc[ct][1] = mask_bias(fc[ct][1], qr0, j0 + 1);
                fc[ct][4] = mask_bias(fc[ct][4], qr0, j1);
                fc[ct][5] = mask_bias(fc[ct][5], qr0, j1 + 1);
                mx0 = fmaxf(mx0, fmaxf(fmaxf(fc[ct][0], fc[ct][1]), fmaxf(fc[ct][4], fc[ct][5])));
                fc[ct][2] = mask_bias(fc[ct][2], qr1, j0);
                fc[ct][3] = mask_bias(fc[ct][3], qr1, j0 + 1);
                fc[ct][6] = mask_bias(fc[ct][6], qr1, j1);
                fc[ct][7] = mask_bias(fc[ct][7], qr1, j1 + 1);
                mx1 = fmaxf(mx1, fmaxf(fmaxf(fc[ct][2], fc[ct][3]), fmaxf(fc[ct][6], fc[ct][7])));
            }
            const float m_new0 = fmaxf(rm0, wmma_row_max_16(mx0));
            const float m_new1 = fmaxf(rm1, wmma_row_max_16(mx1));
            const float a0 = (m_new0 == rm0) ? 1.0f : fast_exp2f_hw(rm0 - m_new0);
            const float a1 = (m_new1 == rm1) ? 1.0f : fast_exp2f_hw(rm1 - m_new1);
            if (__any_sync(0xffffffffu, a0 != 1.0f || a1 != 1.0f)) {
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    acc[n][0] *= a0; acc[n][1] *= a0; acc[n][4] *= a0; acc[n][5] *= a0;
                    acc[n][2] *= a1; acc[n][3] *= a1; acc[n][6] *= a1; acc[n][7] *= a1;
                }
            }
            float ps0 = 0.0f, ps1 = 0.0f;
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) {
                const float p00 = (fc[ct][0] > -1e20f) ? fast_exp2f_hw(fc[ct][0] - m_new0) : 0.0f;
                const float p01 = (fc[ct][1] > -1e20f) ? fast_exp2f_hw(fc[ct][1] - m_new0) : 0.0f;
                const float p04 = (fc[ct][4] > -1e20f) ? fast_exp2f_hw(fc[ct][4] - m_new0) : 0.0f;
                const float p05 = (fc[ct][5] > -1e20f) ? fast_exp2f_hw(fc[ct][5] - m_new0) : 0.0f;
                ps0 += (p00 + p01) + (p04 + p05);
                const float p12 = (fc[ct][2] > -1e20f) ? fast_exp2f_hw(fc[ct][2] - m_new1) : 0.0f;
                const float p13 = (fc[ct][3] > -1e20f) ? fast_exp2f_hw(fc[ct][3] - m_new1) : 0.0f;
                const float p16 = (fc[ct][6] > -1e20f) ? fast_exp2f_hw(fc[ct][6] - m_new1) : 0.0f;
                const float p17 = (fc[ct][7] > -1e20f) ? fast_exp2f_hw(fc[ct][7] - m_new1) : 0.0f;
                ps1 += (p12 + p13) + (p16 + p17);
#if __CUDA_ARCH__ < 750
                const uint32_t u0 = f2_to_h2u(p00, p01), u1 = f2_to_h2u(p04, p05);
                const uint32_t v0 = f2_to_h2u(p12, p13), v1 = f2_to_h2u(p16, p17);
                const uint32_t r0 = __shfl_xor_sync(0xffffffffu, want_r1 ? u0 : v0, 2);
                const uint32_t r1 = __shfl_xor_sync(0xffffffffu, want_r1 ? u1 : v1, 2);
                fc[ct][0] = __uint_as_float(want_r1 ? r0 : u0); fc[ct][1] = __uint_as_float(want_r1 ? v0 : r0);
                fc[ct][2] = __uint_as_float(want_r1 ? r1 : u1); fc[ct][3] = __uint_as_float(want_r1 ? v1 : r1);
#else
                fc[ct][0] = __uint_as_float(f2_to_h2u(p00, p01));
                fc[ct][1] = __uint_as_float(f2_to_h2u(p12, p13));
                fc[ct][2] = __uint_as_float(f2_to_h2u(p04, p05));
                fc[ct][3] = __uint_as_float(f2_to_h2u(p16, p17));
#endif
            }
            rm0 = m_new0; rl0 = rl0 * a0 + wmma_row_sum_16(ps0);
            rm1 = m_new1; rl1 = rl1 * a1 + wmma_row_sum_16(ps1);
        }
        __syncthreads();

#if __CUDA_ARCH__ < 750
#pragma unroll
        for (int ct = 0; ct < 4; ++ct) {
            const uint32_t p0 = __float_as_uint(fc[ct][0]), p1 = __float_as_uint(fc[ct][1]);
            const uint32_t p2 = __float_as_uint(fc[ct][2]), p3 = __float_as_uint(fc[ct][3]);
            const uint32_t o0 = __shfl_xor_sync(0xffffffffu, p0, 8);
            const uint32_t o1 = __shfl_xor_sync(0xffffffffu, p1, 8);
            const uint32_t o2 = __shfl_xor_sync(0xffffffffu, p2, 8);
            const uint32_t o3 = __shfl_xor_sync(0xffffffffu, p3, 8);
            const uint32_t a0 = hi8 ? o0 : p0, a1 = hi8 ? o1 : p1;
            const uint32_t a2 = hi8 ? o2 : p2, a3 = hi8 ? o3 : p3;
            const uint32_t a4 = hi8 ? p0 : o0, a5 = hi8 ? p1 : o1;
            const uint32_t a6 = hi8 ? p2 : o2, a7 = hi8 ? p3 : o3;
            const __half* sv_base = sKV + sv_off + (ct * 16) * STRIDE;
#pragma unroll
            for (int step = 0; step < 4; ++step) {
                const uint32_t aa0 = (step == 0) ? a0 : (step == 1) ? a2 : (step == 2) ? a4 : a6;
                const uint32_t aa1 = (step == 0) ? a1 : (step == 1) ? a3 : (step == 2) ? a5 : a7;
                const __half* sv_s = sv_base + (step * 4) * STRIDE;
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    const uint2 b = *reinterpret_cast<const uint2*>(sv_s + n * 16);
                    mma_rr(acc[n], aa0, aa1, b.x, b.y);
                }
            }
        }
#else
#pragma unroll
        for (int ct = 0; ct < 4; ++ct) {
            const uint32_t p0 = __float_as_uint(fc[ct][0]), p1 = __float_as_uint(fc[ct][1]);
            const uint32_t p2 = __float_as_uint(fc[ct][2]), p3 = __float_as_uint(fc[ct][3]);
            const __half* sv_base = sKV + sv_off + (ct * 16) * STRIDE;
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                uint32_t b0, b1, b2, b3;
                ldmat_x4_t(b0, b1, b2, b3, sv_base + n * 16);
                mma_16816(acc[n],     p0, p1, p2, p3, b0, b1);
                mma_16816(acc[n] + 4, p0, p1, p2, p3, b2, b3);
            }
        }
#endif
        __syncthreads();
    }

    float* s_out = reinterpret_cast<float*>(smem_raw);
    float* s_rm = reinterpret_cast<float*>(smem_raw + SMEM_BYTES - 1024);
    float* s_rl = s_rm + 128;
    if (is_leader) {
        s_rm[qr0] = rm0; s_rl[qr0] = rl0;
        s_rm[qr1] = rm1; s_rl[qr1] = rl1;
    }
    __syncthreads();

    for (int w_base = 0; w_base < 8; w_base += 4) {
        if (warp >= w_base && warp < w_base + 4) {
            const int lr0 = (warp - w_base) * 16;
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                float* dst0 = s_out + (lr0 + acc_r0) * 128 + n * 16;
                float* dst1 = s_out + (lr0 + acc_r1) * 128 + n * 16;
                *reinterpret_cast<float2*>(dst0 + acc_c0) = make_float2(acc[n][0], acc[n][1]);
                *reinterpret_cast<float2*>(dst0 + acc_c1) = make_float2(acc[n][4], acc[n][5]);
                *reinterpret_cast<float2*>(dst1 + acc_c0) = make_float2(acc[n][2], acc[n][3]);
                *reinterpret_cast<float2*>(dst1 + acc_c1) = make_float2(acc[n][6], acc[n][7]);
            }
        }
        __syncthreads();
        for (int idx4 = tid; idx4 < 64 * 32; idx4 += 256) {
            const int r = idx4 >> 5, col = (idx4 & 31) << 2;
            const int qr = w_base * 16 + r, t = t0 + qr;
            if (t < T) {
                const float4 v = reinterpret_cast<const float4*>(s_out)[idx4];
                float* dst = split_out + (((size_t) t * total_splits + tail_sp) * n_head + h) * 130;
                *reinterpret_cast<float2*>(dst + col)     = make_float2(v.x, v.y);
                *reinterpret_cast<float2*>(dst + col + 2) = make_float2(v.z, v.w);
                if (col == 0) { dst[128] = s_rm[qr]; dst[129] = s_rl[qr]; }
            }
        }
        __syncthreads();
    }
#elif !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    constexpr int SMEM_BYTES = (128 + 64) * 136 * 2;
    extern __shared__ __align__(16) char smem_raw[];
    __half* sQ = reinterpret_cast<__half*>(smem_raw);
    __half* sKV = sQ + 128 * 136;

    const int hkv = blockIdx.y;
    const int g = blockIdx.x % 3;
    const int qt128 = blockIdx.x / 3;
    const int h = 3 * hkv + g;
    const int t0 = qt128 * 128;
    const int jb_start = bulk_aligned + (qt128 >> 1) * 256;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    constexpr float kLog2e = 1.4426950408889634f;
    const float qscale_all = kq_scale * kLog2e;

    const int p0 = pos0[0];
    const int t_last = min(t0 + 127, T - 1);
    const int j_end = p0 + t_last + 1;

    for (int base_r = 0; base_r < 128; base_r += 32) {
        const int qr = base_r + (tid >> 3), qc0 = (tid & 7) * 16;
        const int t = t0 + qr;
        __half2* dst2 = reinterpret_cast<__half2*>(sQ + qr * 136 + qc0);
        if (t < T) {
            const float* qp = Q + ((size_t) t * n_head + h) * 128 + qc0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = *reinterpret_cast<const float4*>(qp + i * 4);
                dst2[i * 2 + 0] = __halves2half2(__float2half(v.x * qscale_all), __float2half(v.y * qscale_all));
                dst2[i * 2 + 1] = __halves2half2(__float2half(v.z * qscale_all), __float2half(v.w * qscale_all));
            }
        } else {
            const __half2 z2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));
#pragma unroll
            for (int i = 0; i < 8; ++i) dst2[i] = z2;
        }
    }
    __syncthreads();

    int acc_r0, acc_r1, acc_c0, acc_c1;
    wmma_acc_layout_16x16(lane, acc_r0, acc_r1, acc_c0, acc_c1);
    const bool is_leader = wmma_row_leader_16(lane);
    const int qr_base = warp * 16;
    const int qr0 = qr_base + acc_r0, qr1 = qr_base + acc_r1;
    float rm0 = 0.0f, rm1 = 0.0f;
    float rl0 = 1.0f, rl1 = 1.0f;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
#pragma unroll
    for (int n = 0; n < 8; ++n) wmma::fill_fragment(acc[n], 0.0f);

    const int kv_r = tid >> 2, kv_c0 = (tid & 3) * 32;
    const __half2 neg1152 = __half2half2(__float2half(-1152.0f));
    const int8_t* kq_h = kq + (size_t) hkv * cap * 128 + kv_c0;
    const int8_t* vq_h = vq + (size_t) hkv * cap * 128 + kv_c0;
    const __half* ks_h = ks + (size_t) hkv * cap;
    const __half* vs_h = vs + (size_t) hkv * cap;
    auto mask_bias = [&](float s, int qr, int j) -> float {
        const int tr = t0 + qr;
        const int pos_q = p0 + tr;
        if (tr >= T || j > pos_q || j >= j_end) return -1e30f;
        const int d = pos_q - j;
        const float b = (d < n_ext)
            ? (__ldg(&rel_tab[((size_t) tr * n_head + h) * n_ext + d]) * kLog2e)
            : (-__ldg(&sink[(size_t) tr * n_head + h]) * kLog2e);
        return s + b;
    };

    for (int jb = jb_start; jb < j_end; jb += 64) {
        {
            const int j = jb + kv_r;
            __half* dst = sKV + kv_r * 136 + kv_c0;
            if (j < j_end) {
                const __half2 dk2 = __half2half2(__ldg(&ks_h[j]));
                const __half2 kb = __hmul2(neg1152, dk2);
                const int8_t* kp = kq_h + (size_t) j * 128;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(kp + i * 16);
                    __half2 h2[8];
                    i8x4_to_h2x2_fma(raw.x, dk2, kb, h2[0], h2[1]); i8x4_to_h2x2_fma(raw.y, dk2, kb, h2[2], h2[3]);
                    i8x4_to_h2x2_fma(raw.z, dk2, kb, h2[4], h2[5]); i8x4_to_h2x2_fma(raw.w, dk2, kb, h2[6], h2[7]);
                    *reinterpret_cast<uint4*>(dst + i * 16) = *reinterpret_cast<const uint4*>(h2);
                    *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                }
            } else {
                const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int i = 0; i < 4; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
            }
        }
        __syncthreads();

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc[4];
#pragma unroll
        for (int ct = 0; ct < 4; ++ct) wmma::fill_fragment(fc[ct], 0.0f);
#pragma unroll 1
        for (int k = 0; k < 128; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa0;
            wmma::load_matrix_sync(fa0, sQ + qr_base * 136 + k, 136);
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                wmma::load_matrix_sync(fb, sKV + (ct * 16) * 136 + k, 136);
                wmma::mma_sync(fc[ct], fa0, fb, fc[ct]);
            }
        }
        __syncthreads();

        {
            const int j = jb + kv_r;
            __half* dst = sKV + kv_r * 136 + kv_c0;
            if (j < j_end) {
                const __half2 dv2 = __half2half2(__ldg(&vs_h[j]));
                const __half2 vb = __hmul2(neg1152, dv2);
                const int8_t* vp = vq_h + (size_t) j * 128;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(vp + i * 16);
                    __half2 h2[8];
                    i8x4_to_h2x2_fma(raw.x, dv2, vb, h2[0], h2[1]); i8x4_to_h2x2_fma(raw.y, dv2, vb, h2[2], h2[3]);
                    i8x4_to_h2x2_fma(raw.z, dv2, vb, h2[4], h2[5]); i8x4_to_h2x2_fma(raw.w, dv2, vb, h2[6], h2[7]);
                    *reinterpret_cast<uint4*>(dst + i * 16) = *reinterpret_cast<const uint4*>(h2);
                    *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                }
            } else {
                const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int i = 0; i < 4; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
            }
        }

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa_p[4];
        {
            float mx0 = -1e30f, mx1 = -1e30f;
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) {
                const int j0 = jb + ct * 16 + acc_c0, j1 = jb + ct * 16 + acc_c1;
                fc[ct].x[0] = mask_bias(fc[ct].x[0], qr0, j0);
                fc[ct].x[1] = mask_bias(fc[ct].x[1], qr0, j0 + 1);
                fc[ct].x[4] = mask_bias(fc[ct].x[4], qr0, j1);
                fc[ct].x[5] = mask_bias(fc[ct].x[5], qr0, j1 + 1);
                mx0 = fmaxf(mx0, fmaxf(fmaxf(fc[ct].x[0], fc[ct].x[1]), fmaxf(fc[ct].x[4], fc[ct].x[5])));
                fc[ct].x[2] = mask_bias(fc[ct].x[2], qr1, j0);
                fc[ct].x[3] = mask_bias(fc[ct].x[3], qr1, j0 + 1);
                fc[ct].x[6] = mask_bias(fc[ct].x[6], qr1, j1);
                fc[ct].x[7] = mask_bias(fc[ct].x[7], qr1, j1 + 1);
                mx1 = fmaxf(mx1, fmaxf(fmaxf(fc[ct].x[2], fc[ct].x[3]), fmaxf(fc[ct].x[6], fc[ct].x[7])));
            }
            const float m_new0 = fmaxf(rm0, wmma_row_max_16(mx0));
            const float m_new1 = fmaxf(rm1, wmma_row_max_16(mx1));
            const float a0 = (m_new0 == rm0) ? 1.0f : fast_exp2f_hw(rm0 - m_new0);
            const float a1 = (m_new1 == rm1) ? 1.0f : fast_exp2f_hw(rm1 - m_new1);
            if (__any_sync(0xffffffffu, a0 != 1.0f || a1 != 1.0f)) {
#pragma unroll
                for (int n = 0; n < 8; ++n) wmma_rescale_acc(acc[n], a0, a1);
            }
            float ps0 = 0.0f, ps1 = 0.0f;
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) wmma_acc_to_matA_16x16<true>(fc[ct], m_new0, m_new1, ps0, ps1, fa_p[ct], lane);
            rm0 = m_new0; rl0 = rl0 * a0 + wmma_row_sum_16(ps0);
            rm1 = m_new1; rl1 = rl1 * a1 + wmma_row_sum_16(ps1);
        }
        __syncthreads();

#pragma unroll
        for (int ct = 0; ct < 4; ++ct) {
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
                wmma::load_matrix_sync(fb, sKV + (ct * 16) * 136 + n * 16, 136);
                wmma::mma_sync(acc[n], fa_p[ct], fb, acc[n]);
            }
        }
        __syncthreads();
    }

    float* s_out = reinterpret_cast<float*>(smem_raw);
    float* s_rm = reinterpret_cast<float*>(smem_raw + SMEM_BYTES - 1024);
    float* s_rl = s_rm + 128;
    if (is_leader) {
        s_rm[qr0] = rm0; s_rl[qr0] = rl0;
        s_rm[qr1] = rm1; s_rl[qr1] = rl1;
    }
    __syncthreads();

    for (int w_base = 0; w_base < 8; w_base += 4) {
        if (warp >= w_base && warp < w_base + 4) {
            const int lr0 = (warp - w_base) * 16;
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                wmma::store_matrix_sync(s_out + lr0 * 128 + n * 16, acc[n], 128, wmma::mem_row_major);
            }
        }
        __syncthreads();
        for (int idx4 = tid; idx4 < 64 * 32; idx4 += 256) {
            const int r = idx4 >> 5, col = (idx4 & 31) << 2;
            const int qr = w_base * 16 + r, t = t0 + qr;
            if (t < T) {
                const float4 v = reinterpret_cast<const float4*>(s_out)[idx4];
                float* dst = split_out + (((size_t) t * total_splits + tail_sp) * n_head + h) * 130;
                *reinterpret_cast<float2*>(dst + col)     = make_float2(v.x, v.y);
                *reinterpret_cast<float2*>(dst + col + 2) = make_float2(v.z, v.w);
                if (col == 0) { dst[128] = s_rm[qr]; dst[129] = s_rl[qr]; }
            }
        }
        __syncthreads();
    }
#endif
}

/// Combine Split-KV prefill FA2 outputs (log2-base m/l) and apply optional sigmoid gate (+ optional direct FP16 output).
/// grid (n_head, T), block 128.
__global__ void k_attn_pp_split_combine(const float* __restrict__ split_out, const float* __restrict__ gate,
                                        float* __restrict__ out, int n_head, int split_n,
                                        uint16_t* __restrict__ out_f16 = nullptr) {
    const int h = blockIdx.x, t = blockIdx.y, col = threadIdx.x;
    float mg = -1e30f;
    for (int sp = 0; sp < split_n; ++sp) {
        const float* src = split_out + (((size_t) t * split_n + sp) * n_head + h) * 130;
        if (src[129] > 0.0f) mg = fmaxf(mg, src[128]);
    }
    float lg = 0.0f, og = 0.0f;
    for (int sp = 0; sp < split_n; ++sp) {
        const float* src = split_out + (((size_t) t * split_n + sp) * n_head + h) * 130;
        const float l = src[129];
        if (l > 0.0f) {
            const float c = exp2f(src[128] - mg);
            lg += l * c;
            og += src[col] * c;
        }
    }
    const size_t oi = ((size_t) t * n_head + h) * 128 + col;
    float val = __fdividef(og, lg);
    if (gate) val *= sigmf(gate[oi]);
    if (out_f16) out_f16[oi] = __half_as_ushort(__float2half_rn(val));
    else out[oi] = val;
}

/// GQA-3 Split-KV Q8 decode / verify attention (1 <= T < 16):
/// grid (n_head_kv, split_n, T), block 128 (8 half-warps of 16 threads).
/// Each half-warp handles 1 key token via uint2 KV loads, SIMD i8x4_to_h2x2 decode, __half2 V acc, and log2 exp2f.
/// Volta sm_70 uses 6 blk/SM (80 reg budget -> STACK:0 + software-pipelined prefetch); sm_75+ uses 8 blk/SM.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
__global__ void __launch_bounds__(128, 6) k_attn_q8_gqa3_split(
#else
__global__ void __launch_bounds__(128, 8) k_attn_q8_gqa3_split(
#endif
        const float* __restrict__ Q,
        const int8_t* __restrict__ kq, const __half* __restrict__ ks,
        const int8_t* __restrict__ vq, const __half* __restrict__ vs,
        const float* __restrict__ rel_tab, const float* __restrict__ sink,
        float* __restrict__ split_out,
        int n_head, int n_head_kv, int n_ext, int cap, int is_swa, int n_swa,
        const int32_t* __restrict__ pos0, float kq_scale, int split_n) {
    __shared__ __half2 sq2[3][64];
    __shared__ float s_m[3][8];
    __shared__ float s_l[3][8];
    __shared__ float s_acc[8][128];
    const int hkv = blockIdx.x, sp = blockIdx.y, t = blockIdx.z;
    const int tid = threadIdx.x;
    const int hw = tid >> 4;
    const int hlane = tid & 15;
    const int kv_dim = n_head_kv * 128;
    const int h0 = 3 * hkv;
    constexpr float kLog2e = 1.4426950408889634f;
    const float kq_l2 = kq_scale * kLog2e;

    const int pos_q = pos0[0] + t;
    const int j0 = is_swa ? max(0, pos_q - n_swa + 1) : 0;
    const int j1 = pos_q + 1;
    const int n_tok = j1 - j0;
    const int chunk_sz = is_swa ? max(32, (n_tok + split_n - 1) / split_n) : max(64, (n_tok + split_n - 1) / split_n);
    const int c_beg = j0 + sp * chunk_sz;
    if (c_beg >= j1) {
        if (tid < 3) {
            float* dst = split_out + (((size_t) t * split_n + sp) * n_head + (h0 + tid)) * 130;
            dst[128] = -1e30f;
            dst[129] = 0.0f;
        }
        return;
    }
    const int c_end = min(j1, c_beg + chunk_sz);

    if (tid < 64) {
#pragma unroll
        for (int g = 0; g < 3; ++g) {
            const float2 qf = *reinterpret_cast<const float2*>(Q + ((size_t) t * n_head + (h0 + g)) * 128 + tid * 2);
            sq2[g][tid] = __float22half2_rn(qf);
        }
    }
    __syncthreads();

    const int base8 = hlane * 8;
    __half2 q2[3][4];
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        q2[g][0] = sq2[g][hlane * 4 + 0];
        q2[g][1] = sq2[g][hlane * 4 + 1];
        q2[g][2] = sq2[g][hlane * 4 + 2];
        q2[g][3] = sq2[g][hlane * 4 + 3];
    }

    float neg_sink2[3];
    const float* rt[3];
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        const int h = h0 + g;
        rt[g] = rel_tab + ((size_t) t * n_head + h) * n_ext;
        neg_sink2[g] = -sink[(size_t) t * n_head + h] * kLog2e;
    }

    const bool has_sink = (sp == 0 && hw == 0);
    const __half2 z2 = __half2half2(__float2half(0.0f));
    float m[3], l[3];
    __half2 acc2[3][4];
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        m[g] = has_sink ? 0.0f : -1e30f;
        l[g] = has_sink ? 1.0f : 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) acc2[g][i] = z2;
    }

    // Bulk loop for !is_swa tokens with dist >= n_ext: linear slot == j, Head-Major [n_head_kv][cap][128].
    // Uses shifted-nsink (0 bulk adds), fast_exp2f_hw, lazy rescale (only when sc > m[g]), and 0xffffffffu warp shuffles.
    const int8_t* kq_h = kq + (size_t) hkv * cap * 128 + base8;
    const int8_t* vq_h = vq + (size_t) hkv * cap * 128 + base8;
    const __half* ks_h = ks + (size_t) hkv * cap;
    const __half* vs_h = vs + (size_t) hkv * cap;
    const int bulk_end = is_swa ? c_beg : min(c_end, max(c_beg, pos_q - n_ext + 1));
    if (c_beg < bulk_end) {
#pragma unroll
        for (int g = 0; g < 3; ++g) {
            if (m[g] > -1e20f) m[g] -= neg_sink2[g];
        }
        const int bulk_vec_end = c_beg + ((bulk_end - c_beg) & ~7);
        const unsigned hw_mask = (hw & 1) ? 0xffff0000u : 0x0000ffffu;
        int j = c_beg + hw;
        if (j < bulk_vec_end) {
            float dkkq2 = __half2float(__ldg(&ks_h[j])) * kq_l2;
            uint2 k8 = *reinterpret_cast<const uint2*>(kq_h + (size_t) j * 128);
            for (; j < bulk_vec_end; j += 8) {
                const int jn = j + 8;
                float next_dkkq2 = 0.0f;
                uint2 next_k8 = make_uint2(0, 0);
                if (jn < bulk_vec_end) {
                    next_dkkq2 = __half2float(__ldg(&ks_h[jn])) * kq_l2;
                    next_k8 = *reinterpret_cast<const uint2*>(kq_h + (size_t) jn * 128);
                }
                __half2 kh[4];
                i8x4_to_h2x2(k8.x, kh[0], kh[1]);
                i8x4_to_h2x2(k8.y, kh[2], kh[3]);
                float sc[3];
#pragma unroll
                for (int g = 0; g < 3; ++g) {
                    __half2 d2 = __hmul2(q2[g][0], kh[0]);
                    d2 = __hfma2(q2[g][1], kh[1], d2);
                    d2 = __hfma2(q2[g][2], kh[2], d2);
                    d2 = __hfma2(q2[g][3], kh[3], d2);
                    float dt = __half2float(__low2half(d2)) + __half2float(__high2half(d2));
#pragma unroll
                    for (int o = 8; o > 0; o >>= 1) dt += __shfl_xor_sync(hw_mask, dt, o, 16);
                    sc[g] = dt * dkkq2;
                }
                // Exact FP16-zero V-skip: 2^-28 = 3.7e-9 -> __float2half(p * d_v) == 0.0 bit-identically!
                if (sc[0] - m[0] >= -28.0f || sc[1] - m[1] >= -28.0f || sc[2] - m[2] >= -28.0f) {
                    const float d_v = __half2float(__ldg(&vs_h[j]));
                    const uint2 v8 = *reinterpret_cast<const uint2*>(vq_h + (size_t) j * 128);
                    __half2 vh[4];
                    i8x4_to_h2x2(v8.x, vh[0], vh[1]);
                    i8x4_to_h2x2(v8.y, vh[2], vh[3]);
#pragma unroll
                    for (int g = 0; g < 3; ++g) {
                        if (sc[g] - m[g] >= -28.0f) {
                            float p;
                            if (sc[g] > m[g]) {
                                const float alpha = l[g] > 0.0f ? fast_exp2f_hw(m[g] - sc[g]) : 0.0f;
                                const __half2 a2 = __half2half2(__float2half(alpha));
                                l[g] = l[g] * alpha + 1.0f;
#pragma unroll
                                for (int i = 0; i < 4; ++i) acc2[g][i] = __hmul2(acc2[g][i], a2);
                                m[g] = sc[g];
                                p = 1.0f;
                            } else {
                                p = fast_exp2f_hw(sc[g] - m[g]);
                                l[g] += p;
                            }
                            const __half2 pv2 = __half2half2(__float2half(p * d_v));
#pragma unroll
                            for (int i = 0; i < 4; ++i) acc2[g][i] = __hfma2(pv2, vh[i], acc2[g][i]);
                        }
                    }
                }
                dkkq2 = next_dkkq2;
                k8 = next_k8;
            }
        }
        if (j < bulk_end) {
            const float dkkq2 = __half2float(__ldg(&ks_h[j])) * kq_l2;
            const float d_v = __half2float(__ldg(&vs_h[j]));
            const uint2 k8 = *reinterpret_cast<const uint2*>(kq_h + (size_t) j * 128);
            const uint2 v8 = *reinterpret_cast<const uint2*>(vq_h + (size_t) j * 128);
            __half2 kh[4], vh[4];
            i8x4_to_h2x2(k8.x, kh[0], kh[1]);
            i8x4_to_h2x2(k8.y, kh[2], kh[3]);
            i8x4_to_h2x2(v8.x, vh[0], vh[1]);
            i8x4_to_h2x2(v8.y, vh[2], vh[3]);
            const unsigned am = __activemask();
#pragma unroll
            for (int g = 0; g < 3; ++g) {
                __half2 d2 = __hmul2(q2[g][0], kh[0]);
                d2 = __hfma2(q2[g][1], kh[1], d2);
                d2 = __hfma2(q2[g][2], kh[2], d2);
                d2 = __hfma2(q2[g][3], kh[3], d2);
                float dt = __half2float(__low2half(d2)) + __half2float(__high2half(d2));
#pragma unroll
                for (int o = 8; o > 0; o >>= 1) dt += __shfl_xor_sync(am, dt, o, 16);
                const float sc = dt * dkkq2;
                float p;
                if (sc > m[g]) {
                    const float alpha = l[g] > 0.0f ? fast_exp2f_hw(m[g] - sc) : 0.0f;
                    const __half2 a2 = __half2half2(__float2half(alpha));
                    l[g] = l[g] * alpha + 1.0f;
#pragma unroll
                    for (int i = 0; i < 4; ++i) acc2[g][i] = __hmul2(acc2[g][i], a2);
                    m[g] = sc;
                    p = 1.0f;
                } else {
                    p = fast_exp2f_hw(sc - m[g]);
                    l[g] += p;
                }
                const __half2 pv2 = __half2half2(__float2half(p * d_v));
#pragma unroll
                for (int i = 0; i < 4; ++i) acc2[g][i] = __hfma2(pv2, vh[i], acc2[g][i]);
            }
        }
#pragma unroll
        for (int g = 0; g < 3; ++g) {
            if (m[g] > -1e20f) m[g] += neg_sink2[g];
        }
    }

    const bool pow2_cap = (cap & (cap - 1)) == 0;
    const int cap_mask = cap - 1;
    const int j_rem_beg = max(c_beg, bulk_end);
    const unsigned hw_mask_rem = (hw & 1) ? 0xffff0000u : 0x0000ffffu;
    if (is_swa && j_rem_beg < c_end) {
        // Software-pipelined SWA main loop (multiple-of-8 tokens: zero branch divergence, hw_mask shuffles, overlapped next_k8 prefetch!).
        const int8_t* swa_kq_base = kq + (size_t) hkv * 128 + base8;
        const int8_t* swa_vq_base = vq + (size_t) hkv * 128 + base8;
        const __half* swa_ks_base = ks + hkv;
        const __half* swa_vs_base = vs + hkv;
        const int rem_vec_end = j_rem_beg + ((c_end - j_rem_beg) & ~7);
        int j = j_rem_beg + hw;
        int slot = pow2_cap ? (j & cap_mask) : (j % cap);
        if (j < rem_vec_end) {
            float dkkq2 = __half2float(__ldg(&swa_ks_base[(size_t) slot * n_head_kv])) * kq_l2;
            uint2 k8 = *reinterpret_cast<const uint2*>(swa_kq_base + (size_t) slot * kv_dim);
            for (; j < rem_vec_end; j += 8) {
                const int jn = j + 8;
                int next_slot = slot + 8;
                if (next_slot >= cap) next_slot -= cap;
                float next_dkkq2 = 0.0f;
                uint2 next_k8 = make_uint2(0, 0);
                if (jn < rem_vec_end) {
                    next_dkkq2 = __half2float(__ldg(&swa_ks_base[(size_t) next_slot * n_head_kv])) * kq_l2;
                    next_k8 = *reinterpret_cast<const uint2*>(swa_kq_base + (size_t) next_slot * kv_dim);
                }
                __half2 kh[4];
                i8x4_to_h2x2(k8.x, kh[0], kh[1]);
                i8x4_to_h2x2(k8.y, kh[2], kh[3]);
                const int dist = pos_q - j;
                const bool in_rel = (dist < n_ext);
                float sc[3];
#pragma unroll
                for (int g = 0; g < 3; ++g) {
                    __half2 d2 = __hmul2(q2[g][0], kh[0]);
                    d2 = __hfma2(q2[g][1], kh[1], d2);
                    d2 = __hfma2(q2[g][2], kh[2], d2);
                    d2 = __hfma2(q2[g][3], kh[3], d2);
                    float dt = __half2float(__low2half(d2)) + __half2float(__high2half(d2));
#pragma unroll
                    for (int o = 8; o > 0; o >>= 1) dt += __shfl_xor_sync(hw_mask_rem, dt, o, 16);
                    const float bias = in_rel ? (__ldg(&rt[g][dist]) * kLog2e) : neg_sink2[g];
                    sc[g] = dt * dkkq2 + bias;
                }
                if (sc[0] - m[0] >= -28.0f || sc[1] - m[1] >= -28.0f || sc[2] - m[2] >= -28.0f) {
                    const float d_v = __half2float(__ldg(&swa_vs_base[(size_t) slot * n_head_kv]));
                    const uint2 v8 = *reinterpret_cast<const uint2*>(swa_vq_base + (size_t) slot * kv_dim);
                    __half2 vh[4];
                    i8x4_to_h2x2(v8.x, vh[0], vh[1]);
                    i8x4_to_h2x2(v8.y, vh[2], vh[3]);
#pragma unroll
                    for (int g = 0; g < 3; ++g) {
                        if (sc[g] - m[g] >= -28.0f) {
                            float p;
                            if (sc[g] > m[g]) {
                                const float alpha = l[g] > 0.0f ? fast_exp2f_hw(m[g] - sc[g]) : 0.0f;
                                const __half2 a2 = __half2half2(__float2half(alpha));
                                l[g] = l[g] * alpha + 1.0f;
#pragma unroll
                                for (int i = 0; i < 4; ++i) acc2[g][i] = __hmul2(acc2[g][i], a2);
                                m[g] = sc[g];
                                p = 1.0f;
                            } else {
                                p = fast_exp2f_hw(sc[g] - m[g]);
                                l[g] += p;
                            }
                            const __half2 pv2 = __half2half2(__float2half(p * d_v));
#pragma unroll
                            for (int i = 0; i < 4; ++i) acc2[g][i] = __hfma2(pv2, vh[i], acc2[g][i]);
                        }
                    }
                }
                slot = next_slot;
                dkkq2 = next_dkkq2;
                k8 = next_k8;
            }
        }
        if (j < c_end) {
            const float dkkq2 = __half2float(__ldg(&swa_ks_base[(size_t) slot * n_head_kv])) * kq_l2;
            const uint2 k8 = *reinterpret_cast<const uint2*>(swa_kq_base + (size_t) slot * kv_dim);
            __half2 kh[4];
            i8x4_to_h2x2(k8.x, kh[0], kh[1]);
            i8x4_to_h2x2(k8.y, kh[2], kh[3]);
            const int dist = pos_q - j;
            const bool in_rel = (dist < n_ext);
            const unsigned am = __activemask();
            float sc[3];
#pragma unroll
            for (int g = 0; g < 3; ++g) {
                __half2 d2 = __hmul2(q2[g][0], kh[0]);
                d2 = __hfma2(q2[g][1], kh[1], d2);
                d2 = __hfma2(q2[g][2], kh[2], d2);
                d2 = __hfma2(q2[g][3], kh[3], d2);
                float dt = __half2float(__low2half(d2)) + __half2float(__high2half(d2));
#pragma unroll
                for (int o = 8; o > 0; o >>= 1) dt += __shfl_xor_sync(am, dt, o, 16);
                const float bias = in_rel ? (__ldg(&rt[g][dist]) * kLog2e) : neg_sink2[g];
                sc[g] = dt * dkkq2 + bias;
            }
            if (sc[0] - m[0] >= -28.0f || sc[1] - m[1] >= -28.0f || sc[2] - m[2] >= -28.0f) {
                const float d_v = __half2float(__ldg(&swa_vs_base[(size_t) slot * n_head_kv]));
                const uint2 v8 = *reinterpret_cast<const uint2*>(swa_vq_base + (size_t) slot * kv_dim);
                __half2 vh[4];
                i8x4_to_h2x2(v8.x, vh[0], vh[1]);
                i8x4_to_h2x2(v8.y, vh[2], vh[3]);
#pragma unroll
                for (int g = 0; g < 3; ++g) {
                    if (sc[g] - m[g] >= -28.0f) {
                        float p;
                        if (sc[g] > m[g]) {
                            const float alpha = l[g] > 0.0f ? fast_exp2f_hw(m[g] - sc[g]) : 0.0f;
                            const __half2 a2 = __half2half2(__float2half(alpha));
                            l[g] = l[g] * alpha + 1.0f;
#pragma unroll
                            for (int i = 0; i < 4; ++i) acc2[g][i] = __hmul2(acc2[g][i], a2);
                            m[g] = sc[g];
                            p = 1.0f;
                        } else {
                            p = fast_exp2f_hw(sc[g] - m[g]);
                            l[g] += p;
                        }
                        const __half2 pv2 = __half2half2(__float2half(p * d_v));
#pragma unroll
                        for (int i = 0; i < 4; ++i) acc2[g][i] = __hfma2(pv2, vh[i], acc2[g][i]);
                    }
                }
            }
        }
    } else {
    const int j_first_b = j_rem_beg + (hw & ~1);
    for (int jb = j_first_b; jb < c_end; jb += 8) {
        const int j = jb + (hw & 1);
        if (j < c_end) {
            const float dkkq2 = __half2float(__ldg(&ks_h[j])) * kq_l2;
            const uint2 k8 = *reinterpret_cast<const uint2*>(kq_h + (size_t) j * 128);
            __half2 kh[4];
            i8x4_to_h2x2(k8.x, kh[0], kh[1]);
            i8x4_to_h2x2(k8.y, kh[2], kh[3]);
            const int dist = pos_q - j;
            const unsigned am = __activemask();
            float sc[3];
#pragma unroll
            for (int g = 0; g < 3; ++g) {
                __half2 d2 = __hmul2(q2[g][0], kh[0]);
                d2 = __hfma2(q2[g][1], kh[1], d2);
                d2 = __hfma2(q2[g][2], kh[2], d2);
                d2 = __hfma2(q2[g][3], kh[3], d2);
                float dt = __half2float(__low2half(d2)) + __half2float(__high2half(d2));
#pragma unroll
                for (int o = 8; o > 0; o >>= 1) dt += __shfl_xor_sync(am, dt, o, 16);
                const float bias = dist < n_ext ? (__ldg(&rt[g][dist]) * kLog2e) : neg_sink2[g];
                sc[g] = dt * dkkq2 + bias;
            }
            if (sc[0] - m[0] >= -28.0f || sc[1] - m[1] >= -28.0f || sc[2] - m[2] >= -28.0f) {
                const float d_v = __half2float(__ldg(&vs_h[j]));
                const uint2 v8 = *reinterpret_cast<const uint2*>(vq_h + (size_t) j * 128);
                __half2 vh[4];
                i8x4_to_h2x2(v8.x, vh[0], vh[1]);
                i8x4_to_h2x2(v8.y, vh[2], vh[3]);
#pragma unroll
                for (int g = 0; g < 3; ++g) {
                    if (sc[g] - m[g] >= -28.0f) {
                        float p;
                        if (sc[g] > m[g]) {
                            const float alpha = l[g] > 0.0f ? fast_exp2f_hw(m[g] - sc[g]) : 0.0f;
                            const __half2 a2 = __half2half2(__float2half(alpha));
                            l[g] = l[g] * alpha + 1.0f;
#pragma unroll
                            for (int i = 0; i < 4; ++i) acc2[g][i] = __hmul2(acc2[g][i], a2);
                            m[g] = sc[g];
                            p = 1.0f;
                        } else {
                            p = fast_exp2f_hw(sc[g] - m[g]);
                            l[g] += p;
                        }
                        const __half2 pv2 = __half2half2(__float2half(p * d_v));
#pragma unroll
                        for (int i = 0; i < 4; ++i) acc2[g][i] = __hfma2(pv2, vh[i], acc2[g][i]);
                    }
                }
            }
        }
    }
    }

#pragma unroll
    for (int g = 0; g < 3; ++g) {
        if (hlane == 0) {
            s_m[g][hw] = m[g];
            s_l[g][hw] = l[g];
        }
    }
#pragma unroll
    for (int g = 0; g < 3; ++g) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f2 = __half22float2(acc2[g][i]);
            s_acc[hw][base8 + 2 * i + 0] = f2.x;
            s_acc[hw][base8 + 2 * i + 1] = f2.y;
        }
        __syncthreads();
        const int h = h0 + g;
        float mg = -1e30f;
        for (int w = 0; w < 8; ++w) if (s_l[g][w] > 0.0f && s_m[g][w] > mg) mg = s_m[g][w];
        float lg = 0.0f, og = 0.0f;
        for (int w = 0; w < 8; ++w) {
            if (s_l[g][w] > 0.0f) {
                const float c = exp2f(s_m[g][w] - mg);
                lg += s_l[g][w] * c;
                og += s_acc[w][tid] * c;
            }
        }
        float* dst = split_out + (((size_t) t * split_n + sp) * n_head + h) * 130;
        dst[tid] = og;
        if (tid == 0) {
            dst[128] = mg;
            dst[129] = lg;
        }
        __syncthreads();
    }
}

/// Shared-KV Split-KV Q8 verify attention for T == 4 (non-SWA, GQA-3, head_dim=128):
/// Loads K/V once per BK=16 tile into shared memory instead of 4x from HBM;
/// pairs 2 half-warps per query token (parity 0/1) for full 8-half-warp occupancy and CUDA-graph dynamic *pos0.
__global__ void __launch_bounds__(128, 6) k_attn_q8_gqa3_verify4_split(
        const float* __restrict__ Q,
        const int8_t* __restrict__ kq, const __half* __restrict__ ks,
        const int8_t* __restrict__ vq, const __half* __restrict__ vs,
        const float* __restrict__ rel_tab, const float* __restrict__ sink,
        float* __restrict__ split_out,
        int n_head, int n_head_kv, int n_ext,
        const int32_t* __restrict__ pos0, float kq_scale, int split_n) {
    constexpr int BK = 16;
    __shared__ uint2 sK[BK][16];
    __shared__ uint2 sV[BK][16];
    __shared__ float sDK[BK];
    __shared__ float sDV[BK];
    __shared__ float s_m[3][8];
    __shared__ float s_l[3][8];
    __shared__ float s_acc[4][128];

    const int hkv = blockIdx.x, sp = blockIdx.y;
    const int tid = threadIdx.x, hw = tid >> 4, hlane = tid & 15;
    const int t = hw & 3;
    const int parity = hw >> 2;
    const int h0 = 3 * hkv, kv_dim = n_head_kv * 128;
    const int base8 = hlane * 8;
    constexpr float kLog2e = 1.4426950408889634f;
    const float kq_l2 = kq_scale * kLog2e;

    const int p0 = pos0[0];
    const int total = p0 + 4;
    const int chunk_sz = max(64, (total + split_n - 1) / split_n);
    const int c_beg = sp * chunk_sz;
    if (c_beg >= total) {
        if (parity == 0 && hlane == 0) {
            for (int g = 0; g < 3; ++g) {
                float* dst = split_out + (((size_t) t * split_n + sp) * n_head + (h0 + g)) * 130;
                dst[128] = -1e30f;
                dst[129] = 0.0f;
            }
        }
        return;
    }
    const int c_end = min(total, c_beg + chunk_sz);

    __half2 q2[3][4];
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        const float* qp = Q + ((size_t) t * n_head + (h0 + g)) * 128 + base8;
        q2[g][0] = __float22half2_rn(*reinterpret_cast<const float2*>(qp + 0));
        q2[g][1] = __float22half2_rn(*reinterpret_cast<const float2*>(qp + 2));
        q2[g][2] = __float22half2_rn(*reinterpret_cast<const float2*>(qp + 4));
        q2[g][3] = __float22half2_rn(*reinterpret_cast<const float2*>(qp + 6));
    }

    const int pos_q = p0 + t;
    float neg_sink2[3];
    const float* rt[3];
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        const int h = h0 + g;
        rt[g] = rel_tab + ((size_t) t * n_head + h) * n_ext;
        neg_sink2[g] = -sink[(size_t) t * n_head + h] * kLog2e;
    }

    const bool has_sink = (sp == 0 && parity == 0);
    const __half2 z2 = __half2half2(__float2half(0.0f));
    float m[3], l[3];
    __half2 acc2[3][4];
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        m[g] = has_sink ? 0.0f : -1e30f;
        l[g] = has_sink ? 1.0f : 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) acc2[g][i] = z2;
    }

    const int bulk_limit = p0 - n_ext + 1;
    const int bulk_end = min(c_end, max(c_beg, bulk_limit));
    int jb = c_beg;
    for (; jb + BK <= bulk_end; jb += BK) {
        for (int r = hw; r < BK; r += 8) {
            const int j = jb + r;
            sK[r][hlane] = *reinterpret_cast<const uint2*>(kq + (size_t) j * kv_dim + hkv * 128 + base8);
            sV[r][hlane] = *reinterpret_cast<const uint2*>(vq + (size_t) j * kv_dim + hkv * 128 + base8);
            if (hlane == 0) {
                const size_t idx = (size_t) j * n_head_kv + hkv;
                sDK[r] = __half2float(__ldg(&ks[idx])) * kq_l2;
                sDV[r] = __half2float(__ldg(&vs[idx]));
            }
        }
        __syncthreads();

#pragma unroll
        for (int k = parity; k < BK; k += 2) {
            const uint2 k8 = sK[k][hlane];
            const uint2 v8 = sV[k][hlane];
            const float dkkq2 = sDK[k], d_v = sDV[k];
            __half2 kh[4], vh[4];
            i8x4_to_h2x2(k8.x, kh[0], kh[1]);
            i8x4_to_h2x2(k8.y, kh[2], kh[3]);
            i8x4_to_h2x2(v8.x, vh[0], vh[1]);
            i8x4_to_h2x2(v8.y, vh[2], vh[3]);
            float dots[3];
#pragma unroll
            for (int g = 0; g < 3; ++g) {
                __half2 d2 = __hmul2(q2[g][0], kh[0]);
                d2 = __hfma2(q2[g][1], kh[1], d2);
                d2 = __hfma2(q2[g][2], kh[2], d2);
                d2 = __hfma2(q2[g][3], kh[3], d2);
                float dt = __half2float(__low2half(d2)) + __half2float(__high2half(d2));
#pragma unroll
                for (int o = 8; o > 0; o >>= 1) dt += __shfl_xor_sync(__activemask(), dt, o, 16);
                dots[g] = dt;
            }
#pragma unroll
            for (int g = 0; g < 3; ++g) {
                const float sc = dots[g] * dkkq2 + neg_sink2[g];
                const float m_new = fmaxf(m[g], sc);
                const float alpha = l[g] > 0.0f ? exp2f(m[g] - m_new) : 0.0f;
                const float p = exp2f(sc - m_new);
                const __half2 a2 = __half2half2(__float2half(alpha));
                const __half2 pv2 = __half2half2(__float2half(p * d_v));
                l[g] = l[g] * alpha + p;
#pragma unroll
                for (int i = 0; i < 4; ++i) acc2[g][i] = __hfma2(acc2[g][i], a2, __hmul2(pv2, vh[i]));
                m[g] = m_new;
            }
        }
        __syncthreads();
    }

    for (; jb < c_end; jb += BK) {
        const int nk = min(BK, c_end - jb);
        for (int r = hw; r < nk; r += 8) {
            const int j = jb + r;
            sK[r][hlane] = *reinterpret_cast<const uint2*>(kq + (size_t) j * kv_dim + hkv * 128 + base8);
            sV[r][hlane] = *reinterpret_cast<const uint2*>(vq + (size_t) j * kv_dim + hkv * 128 + base8);
            if (hlane == 0) {
                const size_t idx = (size_t) j * n_head_kv + hkv;
                sDK[r] = __half2float(__ldg(&ks[idx])) * kq_l2;
                sDV[r] = __half2float(__ldg(&vs[idx]));
            }
        }
        __syncthreads();

        for (int k = parity; k < nk; k += 2) {
            const int j = jb + k;
            if (j <= pos_q) {
                const uint2 k8 = sK[k][hlane];
                const uint2 v8 = sV[k][hlane];
                const float dkkq2 = sDK[k], d_v = sDV[k];
                __half2 kh[4], vh[4];
                i8x4_to_h2x2(k8.x, kh[0], kh[1]);
                i8x4_to_h2x2(k8.y, kh[2], kh[3]);
                i8x4_to_h2x2(v8.x, vh[0], vh[1]);
                i8x4_to_h2x2(v8.y, vh[2], vh[3]);
                const int dist = pos_q - j;
                float dots[3];
#pragma unroll
                for (int g = 0; g < 3; ++g) {
                    __half2 d2 = __hmul2(q2[g][0], kh[0]);
                    d2 = __hfma2(q2[g][1], kh[1], d2);
                    d2 = __hfma2(q2[g][2], kh[2], d2);
                    d2 = __hfma2(q2[g][3], kh[3], d2);
                    float dt = __half2float(__low2half(d2)) + __half2float(__high2half(d2));
#pragma unroll
                    for (int o = 8; o > 0; o >>= 1) dt += __shfl_xor_sync(__activemask(), dt, o, 16);
                    dots[g] = dt;
                }
#pragma unroll
                for (int g = 0; g < 3; ++g) {
                    const float bias = dist < n_ext ? (rt[g][dist] * kLog2e) : neg_sink2[g];
                    const float sc = dots[g] * dkkq2 + bias;
                    const float m_new = fmaxf(m[g], sc);
                    const float alpha = l[g] > 0.0f ? exp2f(m[g] - m_new) : 0.0f;
                    const float p = exp2f(sc - m_new);
                    const __half2 a2 = __half2half2(__float2half(alpha));
                    const __half2 pv2 = __half2half2(__float2half(p * d_v));
                    l[g] = l[g] * alpha + p;
#pragma unroll
                    for (int i = 0; i < 4; ++i) acc2[g][i] = __hfma2(acc2[g][i], a2, __hmul2(pv2, vh[i]));
                    m[g] = m_new;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int g = 0; g < 3; ++g) {
        if (hlane == 0) {
            s_m[g][hw] = m[g];
            s_l[g][hw] = l[g];
        }
    }
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        if (parity == 1) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 f2 = __half22float2(acc2[g][i]);
                s_acc[t][base8 + 2 * i + 0] = f2.x;
                s_acc[t][base8 + 2 * i + 1] = f2.y;
            }
        }
        __syncthreads();
        if (parity == 0) {
            const float m0 = m[g], l0 = l[g];
            const float m1 = s_m[g][t + 4], l1 = s_l[g][t + 4];
            float mg = -1e30f;
            if (l0 > 0.0f && m0 > mg) mg = m0;
            if (l1 > 0.0f && m1 > mg) mg = m1;
            const float c0 = (l0 > 0.0f) ? exp2f(m0 - mg) : 0.0f;
            const float c1 = (l1 > 0.0f) ? exp2f(m1 - mg) : 0.0f;
            const float lg = l0 * c0 + l1 * c1;
            float* dst = split_out + (((size_t) t * split_n + sp) * n_head + (h0 + g)) * 130;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 f0 = __half22float2(acc2[g][i]);
                dst[base8 + 2 * i + 0] = f0.x * c0 + s_acc[t][base8 + 2 * i + 0] * c1;
                dst[base8 + 2 * i + 1] = f0.y * c0 + s_acc[t][base8 + 2 * i + 1] * c1;
            }
            if (hlane == 0) {
                dst[128] = mg;
                dst[129] = lg;
            }
        }
        __syncthreads();
    }
}

/// Zero-Sync Warp-Independent Tensor-Core (WMMA 16x16x16) Split-KV Q8 verify attention for T == 4 (non-SWA, GQA-3, D=128):
/// 4 warps (128 thr, 4 blk/SM, 21,760 B SMEM, 128 regs, 0 spills): each warp owns a private 16x136 sKV slice and steps
/// independently with ZERO __syncthreads() inside the entire 800K KV loop, ZERO sS/sP SMEM via wmma_acc_to_matA_16x16,
/// V prefetch overlapping QK^T, shifted-nsink bulk loop, lazy acc rescale, and single 4-warp epilogue combine.
template <int T_ACT = 4>
__global__ void __launch_bounds__(128, 4) k_attn_q8_gqa3_verify4_wmma_split(
        const float* __restrict__ Q,
        const int8_t* __restrict__ kq, const __half* __restrict__ ks,
        const int8_t* __restrict__ vq, const __half* __restrict__ vs,
        const float* __restrict__ rel_tab, const float* __restrict__ sink,
        float* __restrict__ split_out,
        int n_head, int n_head_kv, int cap, int n_ext,
        const int32_t* __restrict__ pos0, float kq_scale, int split_n) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
    // Volta sm_70 native PTX HMMA.884 path: STRIDE=144 (23,040 B = 4 blk/SM, 0-bank-conflict LDS.64),
    // K-next prefetch overlapping P*V + V prefetch overlapping QK^T, 0 spills (STACK:0), direct float2 epilogue.
    constexpr int STRIDE = 144;
    extern __shared__ __align__(16) char smem_raw[];
    __half* sQ = reinterpret_cast<__half*>(smem_raw);
    __half* sKV_all = sQ + 16 * STRIDE;

    const int hkv = blockIdx.x, sp = blockIdx.y;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int h0 = 3 * hkv;
    constexpr float kLog2e = 1.4426950408889634f;
    const float qscale_all = kq_scale * kLog2e;

    constexpr int kQRows = 3 * T_ACT;
    const int p0 = pos0[0];
    const int total = p0 + T_ACT;
    const int chunk_sz = max(64, (total + split_n - 1) / split_n);
    const int c_beg = sp * chunk_sz;
    if (c_beg >= total) {
        if (tid < kQRows) {
            float* dst = split_out + (((size_t) (tid / 3) * split_n + sp) * n_head + (h0 + (tid % 3))) * 130;
            dst[128] = -1e30f;
            dst[129] = 0.0f;
        }
        return;
    }
    const int c_end = min(total, c_beg + chunk_sz);

    {
        const int qr = tid >> 3, qc0 = (tid & 7) * 16;
        const float* qp = (qr < kQRows) ? (Q + ((size_t) (qr / 3) * n_head + (h0 + (qr % 3))) * 128 + qc0) : nullptr;
        __half2* dst2 = reinterpret_cast<__half2*>(sQ + qr * STRIDE + qc0);
        if (qp) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = *reinterpret_cast<const float4*>(qp + i * 4);
                dst2[i * 2 + 0] = __halves2half2(__float2half(v.x * qscale_all), __float2half(v.y * qscale_all));
                dst2[i * 2 + 1] = __halves2half2(__float2half(v.z * qscale_all), __float2half(v.w * qscale_all));
            }
        } else {
            const __half2 z2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));
#pragma unroll
            for (int i = 0; i < 8; ++i) dst2[i] = z2;
        }
    }
    __syncthreads();

    auto mma_rc = [](float d[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, {%0, %1, %2, %3, %4, %5, %6, %7};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
            : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
    };
    auto mma_rr = [](float d[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
            "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, {%0, %1, %2, %3, %4, %5, %6, %7};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
            : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
    };

    const int acc_r0 = ((lane & 16) >> 2) + ((lane & 4) << 1) + (lane & 1);
    const int acc_r1 = acc_r0 + 2;
    const int acc_c0 = (lane & 8) + (lane & 2);
    const int acc_c1 = acc_c0 + 4;
    const bool is_leader = (lane & 10) == 0;
    const int qa_row = ((lane & 16) >> 2) + ((lane & 4) << 1) + (lane & 3);
    const int kb_row = ((lane & 16) >> 2) + (lane & 8) + (lane & 3);
    const int vb_row = lane & 3;
    const int vb_col = (lane & 8) + ((lane & 16) >> 2);
    const bool hi8 = (lane & 8) != 0;
    const bool want_r1 = (lane & 2) != 0;

    const bool has_sink = (sp == 0 && warp == 0);
    float rm0 = has_sink ? 0.0f : -1e30f;
    float rm1 = rm0;
    float rl0 = has_sink ? 1.0f : 0.0f;
    float rl1 = rl0;

    auto load_nsink = [&](int r) -> float {
        return (r < kQRows) ? (-__ldg(&sink[(size_t) (r / 3) * n_head + (h0 + (r % 3))]) * kLog2e) : 0.0f;
    };

    float acc[8][8];
#pragma unroll
    for (int n = 0; n < 8; ++n) {
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[n][i] = 0.0f;
    }

    __half* sKV = sKV_all + warp * 16 * STRIDE;
    const int qa_off = qa_row * STRIDE;
    const int kb_off = kb_row * STRIDE;
    const int vb_off = vb_row * STRIDE + vb_col;

    const int wr = lane >> 1;
    const int wc0 = (lane & 1) * 64;
    const __half2 neg1152 = __half2half2(__float2half(-1152.0f));
    const int bulk_end = min(c_end, max(c_beg, p0 - n_ext + 1));

    int jb = c_beg + warp * 16;
    if (jb <= bulk_end - 16) {
        const float ns0 = load_nsink(acc_r0), ns1 = load_nsink(acc_r1);
        if (rm0 > -1e20f) rm0 -= ns0;
        if (rm1 > -1e20f) rm1 -= ns1;

        for (; jb <= bulk_end - 16; jb += 64) {
            const int hj = hkv * cap + (jb + wr);
            {
                const __half2 dk2 = __half2half2(__ldg(&ks[hj]));
                const __half2 kb = __hmul2(neg1152, dk2);
                const int8_t* kp = kq + (size_t) hj * 128 + wc0;
                __half* dst = sKV + wr * STRIDE + wc0;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(kp + i * 16);
                    __half2 h2[8];
                    i8x4_to_h2x2_fma(raw.x, dk2, kb, h2[0], h2[1]); i8x4_to_h2x2_fma(raw.y, dk2, kb, h2[2], h2[3]);
                    i8x4_to_h2x2_fma(raw.z, dk2, kb, h2[4], h2[5]); i8x4_to_h2x2_fma(raw.w, dk2, kb, h2[6], h2[7]);
                    *reinterpret_cast<uint4*>(dst + i * 16)     = *reinterpret_cast<const uint4*>(h2);
                    *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                }
            }
            __syncwarp();

            float fc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 1
            for (int k = 0; k < 128; k += 8) {
                const uint4 q0 = *reinterpret_cast<const uint4*>(sQ + qa_off + k);
                const uint4 k0 = *reinterpret_cast<const uint4*>(sKV + kb_off + k);
                mma_rc(fc, q0.x, q0.y, k0.x, k0.y);
                mma_rc(fc, q0.z, q0.w, k0.z, k0.w);
            }

            const float mx0 = wmma_row_max_16(fmaxf(fmaxf(fc[0], fc[1]), fmaxf(fc[4], fc[5])));
            const float mx1 = wmma_row_max_16(fmaxf(fmaxf(fc[2], fc[3]), fmaxf(fc[6], fc[7])));
            // Exact bit-identical FP16-zero V-skip: when (mx - rm) <= -26.0f for every row in the warp,
            // every exp2(fc - rm) <= 2^-26 rounds to exact FP16 0x0000 -> P*V is identically zero; skip V HBM load & HMMA!
            if (!__any_sync(0xffffffffu, rm0 <= -1e20f || rm1 <= -1e20f || (mx0 - rm0) > -26.0f || (mx1 - rm1) > -26.0f)) {
                const float p00 = fast_exp2f_hw(fc[0] - rm0), p01 = fast_exp2f_hw(fc[1] - rm0);
                const float p04 = fast_exp2f_hw(fc[4] - rm0), p05 = fast_exp2f_hw(fc[5] - rm0);
                const float p12 = fast_exp2f_hw(fc[2] - rm1), p13 = fast_exp2f_hw(fc[3] - rm1);
                const float p16 = fast_exp2f_hw(fc[6] - rm1), p17 = fast_exp2f_hw(fc[7] - rm1);
                rl0 += wmma_row_sum_16((p00 + p01) + (p04 + p05));
                rl1 += wmma_row_sum_16((p12 + p13) + (p16 + p17));
                __syncwarp();
                continue;
            }

            const __half2 dv2 = __half2half2(__ldg(&vs[hj]));
            const int8_t* vp = vq + (size_t) hj * 128 + wc0;
            __syncwarp();
            {
                const __half2 vb = __hmul2(neg1152, dv2);
                __half* dst = sKV + wr * STRIDE + wc0;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint4 vr = *reinterpret_cast<const uint4*>(vp + i * 16);
                    __half2 h2[8];
                    i8x4_to_h2x2_fma(vr.x, dv2, vb, h2[0], h2[1]); i8x4_to_h2x2_fma(vr.y, dv2, vb, h2[2], h2[3]);
                    i8x4_to_h2x2_fma(vr.z, dv2, vb, h2[4], h2[5]); i8x4_to_h2x2_fma(vr.w, dv2, vb, h2[6], h2[7]);
                    *reinterpret_cast<uint4*>(dst + i * 16)     = *reinterpret_cast<const uint4*>(h2);
                    *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                }
            }

            const float m_new0 = fmaxf(rm0, mx0), m_new1 = fmaxf(rm1, mx1);
            const float a0 = (rm0 > -1e20f) ? ((m_new0 == rm0) ? 1.0f : fast_exp2f_hw(rm0 - m_new0)) : 0.0f;
            const float a1 = (rm1 > -1e20f) ? ((m_new1 == rm1) ? 1.0f : fast_exp2f_hw(rm1 - m_new1)) : 0.0f;
            if (__any_sync(0xffffffffu, a0 != 1.0f || a1 != 1.0f)) {
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    acc[n][0] *= a0; acc[n][1] *= a0; acc[n][4] *= a0; acc[n][5] *= a0;
                    acc[n][2] *= a1; acc[n][3] *= a1; acc[n][6] *= a1; acc[n][7] *= a1;
                }
            }
            const float p00 = fast_exp2f_hw(fc[0] - m_new0), p01 = fast_exp2f_hw(fc[1] - m_new0);
            const float p04 = fast_exp2f_hw(fc[4] - m_new0), p05 = fast_exp2f_hw(fc[5] - m_new0);
            const float p12 = fast_exp2f_hw(fc[2] - m_new1), p13 = fast_exp2f_hw(fc[3] - m_new1);
            const float p16 = fast_exp2f_hw(fc[6] - m_new1), p17 = fast_exp2f_hw(fc[7] - m_new1);
            rm0 = m_new0; rl0 = rl0 * a0 + wmma_row_sum_16((p00 + p01) + (p04 + p05));
            rm1 = m_new1; rl1 = rl1 * a1 + wmma_row_sum_16((p12 + p13) + (p16 + p17));

            const uint32_t u0 = f2_to_h2u(p00, p01), u1 = f2_to_h2u(p04, p05);
            const uint32_t v0 = f2_to_h2u(p12, p13), v1 = f2_to_h2u(p16, p17);
            const uint32_t r0 = __shfl_xor_sync(0xffffffffu, want_r1 ? u0 : v0, 2);
            const uint32_t r1 = __shfl_xor_sync(0xffffffffu, want_r1 ? u1 : v1, 2);
            const uint32_t h0u = want_r1 ? r0 : u0, h1u = want_r1 ? v0 : r0;
            const uint32_t h2u = want_r1 ? r1 : u1, h3u = want_r1 ? v1 : r1;
            const uint32_t o0 = __shfl_xor_sync(0xffffffffu, h0u, 8), o1 = __shfl_xor_sync(0xffffffffu, h1u, 8);
            const uint32_t o2 = __shfl_xor_sync(0xffffffffu, h2u, 8), o3 = __shfl_xor_sync(0xffffffffu, h3u, 8);
            const uint32_t pa0 = hi8 ? o0 : h0u, pa1 = hi8 ? o1 : h1u;
            const uint32_t pa2 = hi8 ? o2 : h2u, pa3 = hi8 ? o3 : h3u;
            const uint32_t pa4 = hi8 ? h0u : o0, pa5 = hi8 ? h1u : o1;
            const uint32_t pa6 = hi8 ? h2u : o2, pa7 = hi8 ? h3u : o3;
            __syncwarp();

            uint2 b[2];
#pragma unroll
            for (int nb = 0; nb < 8; nb += 2) {
                b[0] = *reinterpret_cast<const uint2*>(sKV + vb_off + 0 * STRIDE + (nb + 0) * 16);
                b[1] = *reinterpret_cast<const uint2*>(sKV + vb_off + 0 * STRIDE + (nb + 1) * 16);
                mma_rr(acc[nb + 0], pa0, pa1, b[0].x, b[0].y);
                mma_rr(acc[nb + 1], pa0, pa1, b[1].x, b[1].y);
                b[0] = *reinterpret_cast<const uint2*>(sKV + vb_off + 4 * STRIDE + (nb + 0) * 16);
                b[1] = *reinterpret_cast<const uint2*>(sKV + vb_off + 4 * STRIDE + (nb + 1) * 16);
                mma_rr(acc[nb + 0], pa2, pa3, b[0].x, b[0].y);
                mma_rr(acc[nb + 1], pa2, pa3, b[1].x, b[1].y);
                b[0] = *reinterpret_cast<const uint2*>(sKV + vb_off + 8 * STRIDE + (nb + 0) * 16);
                b[1] = *reinterpret_cast<const uint2*>(sKV + vb_off + 8 * STRIDE + (nb + 1) * 16);
                mma_rr(acc[nb + 0], pa4, pa5, b[0].x, b[0].y);
                mma_rr(acc[nb + 1], pa4, pa5, b[1].x, b[1].y);
                b[0] = *reinterpret_cast<const uint2*>(sKV + vb_off + 12 * STRIDE + (nb + 0) * 16);
                b[1] = *reinterpret_cast<const uint2*>(sKV + vb_off + 12 * STRIDE + (nb + 1) * 16);
                mma_rr(acc[nb + 0], pa6, pa7, b[0].x, b[0].y);
                mma_rr(acc[nb + 1], pa6, pa7, b[1].x, b[1].y);
            }
            __syncwarp();
        }
        if (rm0 > -1e20f) rm0 += ns0;
        if (rm1 > -1e20f) rm1 += ns1;
    }

    if (jb < c_end) {
        auto mask_bias = [&](float s, int r, int j) -> float {
            if (r >= kQRows || j >= c_end) return -1e30f;
            const int rt = r / 3, h = h0 + (r % 3), pos_q = p0 + rt;
            if (j > pos_q) return -1e30f;
            const int d = pos_q - j;
            const float b = (d < n_ext)
                ? (__ldg(&rel_tab[((size_t) rt * n_head + h) * n_ext + d]) * kLog2e)
                : (-__ldg(&sink[(size_t) rt * n_head + h]) * kLog2e);
            return s + b;
        };
        for (; jb < c_end; jb += 64) {
            const int j = jb + wr;
            {
                __half* dst = sKV + wr * STRIDE + wc0;
                if (j < c_end) {
                    const int hj = hkv * cap + j;
                    const __half2 dk2 = __half2half2(__ldg(&ks[hj]));
                    const __half2 kb = __hmul2(neg1152, dk2);
                    const int8_t* kp = kq + (size_t) hj * 128 + wc0;
#pragma unroll 1
                    for (int i = 0; i < 4; ++i) {
                        const uint4 raw = *reinterpret_cast<const uint4*>(kp + i * 16);
                        __half2 h2[8];
                        i8x4_to_h2x2_fma(raw.x, dk2, kb, h2[0], h2[1]); i8x4_to_h2x2_fma(raw.y, dk2, kb, h2[2], h2[3]);
                        i8x4_to_h2x2_fma(raw.z, dk2, kb, h2[4], h2[5]); i8x4_to_h2x2_fma(raw.w, dk2, kb, h2[6], h2[7]);
                        *reinterpret_cast<uint4*>(dst + i * 16)     = *reinterpret_cast<const uint4*>(h2);
                        *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                    }
                } else {
                    const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll 1
                    for (int i = 0; i < 8; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
                }
            }
            __syncwarp();
            float fc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 1
            for (int k = 0; k < 128; k += 8) {
                const uint4 q0 = *reinterpret_cast<const uint4*>(sQ + qa_off + k);
                const uint4 k0 = *reinterpret_cast<const uint4*>(sKV + kb_off + k);
                mma_rc(fc, q0.x, q0.y, k0.x, k0.y);
                mma_rc(fc, q0.z, q0.w, k0.z, k0.w);
            }
            __syncwarp();
            {
                __half* dst = sKV + wr * STRIDE + wc0;
                if (j < c_end) {
                    const int hj = hkv * cap + j;
                    const __half2 dv2 = __half2half2(__ldg(&vs[hj]));
                    const __half2 vb = __hmul2(neg1152, dv2);
                    const int8_t* vp = vq + (size_t) hj * 128 + wc0;
#pragma unroll 1
                    for (int i = 0; i < 4; ++i) {
                        const uint4 raw = *reinterpret_cast<const uint4*>(vp + i * 16);
                        __half2 h2[8];
                        i8x4_to_h2x2_fma(raw.x, dv2, vb, h2[0], h2[1]); i8x4_to_h2x2_fma(raw.y, dv2, vb, h2[2], h2[3]);
                        i8x4_to_h2x2_fma(raw.z, dv2, vb, h2[4], h2[5]); i8x4_to_h2x2_fma(raw.w, dv2, vb, h2[6], h2[7]);
                        *reinterpret_cast<uint4*>(dst + i * 16)     = *reinterpret_cast<const uint4*>(h2);
                        *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                    }
                } else {
                    const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll 1
                    for (int i = 0; i < 8; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
                }
            }
            const int j0 = jb + acc_c0, j1 = jb + acc_c1;
            fc[0] = mask_bias(fc[0], acc_r0, j0);     fc[1] = mask_bias(fc[1], acc_r0, j0 + 1);
            fc[4] = mask_bias(fc[4], acc_r0, j1);     fc[5] = mask_bias(fc[5], acc_r0, j1 + 1);
            fc[2] = mask_bias(fc[2], acc_r1, j0);     fc[3] = mask_bias(fc[3], acc_r1, j0 + 1);
            fc[6] = mask_bias(fc[6], acc_r1, j1);     fc[7] = mask_bias(fc[7], acc_r1, j1 + 1);
            const float mx0 = wmma_row_max_16(fmaxf(fmaxf(fc[0], fc[1]), fmaxf(fc[4], fc[5])));
            const float mx1 = wmma_row_max_16(fmaxf(fmaxf(fc[2], fc[3]), fmaxf(fc[6], fc[7])));
            const float m_new0 = fmaxf(rm0, mx0), m_new1 = fmaxf(rm1, mx1);
            const float a0 = (rm0 > -1e20f) ? ((m_new0 == rm0) ? 1.0f : fast_exp2f_hw(rm0 - m_new0)) : 0.0f;
            const float a1 = (rm1 > -1e20f) ? ((m_new1 == rm1) ? 1.0f : fast_exp2f_hw(rm1 - m_new1)) : 0.0f;
            if (__any_sync(0xffffffffu, a0 != 1.0f || a1 != 1.0f)) {
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    acc[n][0] *= a0; acc[n][1] *= a0; acc[n][4] *= a0; acc[n][5] *= a0;
                    acc[n][2] *= a1; acc[n][3] *= a1; acc[n][6] *= a1; acc[n][7] *= a1;
                }
            }
            const float p00 = (fc[0] > -1e20f) ? fast_exp2f_hw(fc[0] - m_new0) : 0.0f;
            const float p01 = (fc[1] > -1e20f) ? fast_exp2f_hw(fc[1] - m_new0) : 0.0f;
            const float p04 = (fc[4] > -1e20f) ? fast_exp2f_hw(fc[4] - m_new0) : 0.0f;
            const float p05 = (fc[5] > -1e20f) ? fast_exp2f_hw(fc[5] - m_new0) : 0.0f;
            const float p12 = (fc[2] > -1e20f) ? fast_exp2f_hw(fc[2] - m_new1) : 0.0f;
            const float p13 = (fc[3] > -1e20f) ? fast_exp2f_hw(fc[3] - m_new1) : 0.0f;
            const float p16 = (fc[6] > -1e20f) ? fast_exp2f_hw(fc[6] - m_new1) : 0.0f;
            const float p17 = (fc[7] > -1e20f) ? fast_exp2f_hw(fc[7] - m_new1) : 0.0f;
            rm0 = m_new0; rl0 = rl0 * a0 + wmma_row_sum_16((p00 + p01) + (p04 + p05));
            rm1 = m_new1; rl1 = rl1 * a1 + wmma_row_sum_16((p12 + p13) + (p16 + p17));
            const uint32_t u0 = f2_to_h2u(p00, p01), u1 = f2_to_h2u(p04, p05);
            const uint32_t v0 = f2_to_h2u(p12, p13), v1 = f2_to_h2u(p16, p17);
            const uint32_t r0 = __shfl_xor_sync(0xffffffffu, want_r1 ? u0 : v0, 2);
            const uint32_t r1 = __shfl_xor_sync(0xffffffffu, want_r1 ? u1 : v1, 2);
            const uint32_t h0u = want_r1 ? r0 : u0, h1u = want_r1 ? v0 : r0;
            const uint32_t h2u = want_r1 ? r1 : u1, h3u = want_r1 ? v1 : r1;
            const uint32_t o0 = __shfl_xor_sync(0xffffffffu, h0u, 8), o1 = __shfl_xor_sync(0xffffffffu, h1u, 8);
            const uint32_t o2 = __shfl_xor_sync(0xffffffffu, h2u, 8), o3 = __shfl_xor_sync(0xffffffffu, h3u, 8);
            const uint32_t pa0 = hi8 ? o0 : h0u, pa1 = hi8 ? o1 : h1u;
            const uint32_t pa2 = hi8 ? o2 : h2u, pa3 = hi8 ? o3 : h3u;
            const uint32_t pa4 = hi8 ? h0u : o0, pa5 = hi8 ? h1u : o1;
            const uint32_t pa6 = hi8 ? h2u : o2, pa7 = hi8 ? h3u : o3;
            __syncwarp();
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                uint2 b0 = *reinterpret_cast<const uint2*>(sKV + vb_off + 0 * STRIDE + n * 16);
                mma_rr(acc[n], pa0, pa1, b0.x, b0.y);
                b0 = *reinterpret_cast<const uint2*>(sKV + vb_off + 4 * STRIDE + n * 16);
                mma_rr(acc[n], pa2, pa3, b0.x, b0.y);
                b0 = *reinterpret_cast<const uint2*>(sKV + vb_off + 8 * STRIDE + n * 16);
                mma_rr(acc[n], pa4, pa5, b0.x, b0.y);
                b0 = *reinterpret_cast<const uint2*>(sKV + vb_off + 12 * STRIDE + n * 16);
                mma_rr(acc[n], pa6, pa7, b0.x, b0.y);
            }
            __syncwarp();
        }
    }

    __syncthreads();
    float* s_wm = reinterpret_cast<float*>(smem_raw + 16384);
    float* s_wl = s_wm + 64;
    float* s_mg = s_wl + 64;
    float* s_lg = s_mg + 16;
    if (is_leader) {
        s_wm[warp * 16 + acc_r0] = rm0; s_wl[warp * 16 + acc_r0] = rl0;
        s_wm[warp * 16 + acc_r1] = rm1; s_wl[warp * 16 + acc_r1] = rl1;
    }
    __syncthreads();
    if (tid < kQRows) {
        float mg = -1e30f;
#pragma unroll
        for (int w = 0; w < 4; ++w) if (s_wl[w * 16 + tid] > 0.0f) mg = fmaxf(mg, s_wm[w * 16 + tid]);
        float lg = 0.0f;
#pragma unroll
        for (int w = 0; w < 4; ++w) if (s_wl[w * 16 + tid] > 0.0f) lg += s_wl[w * 16 + tid] * fast_exp2f_hw(s_wm[w * 16 + tid] - mg);
        s_mg[tid] = mg;
        s_lg[tid] = lg;
    }
    __syncthreads();
    {
        const float cw0 = (acc_r0 < kQRows && rl0 > 0.0f) ? fast_exp2f_hw(rm0 - s_mg[acc_r0]) : 0.0f;
        const float cw1 = (acc_r1 < kQRows && rl1 > 0.0f) ? fast_exp2f_hw(rm1 - s_mg[acc_r1]) : 0.0f;
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            acc[n][0] *= cw0; acc[n][1] *= cw0; acc[n][4] *= cw0; acc[n][5] *= cw0;
            acc[n][2] *= cw1; acc[n][3] *= cw1; acc[n][6] *= cw1; acc[n][7] *= cw1;
        }
    }
    auto store_acc_smem = [&](float* base_s) {
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            float* p0s = base_s + acc_r0 * 128 + n * 16;
            float* p1s = base_s + acc_r1 * 128 + n * 16;
            *reinterpret_cast<float2*>(p0s + acc_c0) = make_float2(acc[n][0], acc[n][1]);
            *reinterpret_cast<float2*>(p0s + acc_c1) = make_float2(acc[n][4], acc[n][5]);
            *reinterpret_cast<float2*>(p1s + acc_c0) = make_float2(acc[n][2], acc[n][3]);
            *reinterpret_cast<float2*>(p1s + acc_c1) = make_float2(acc[n][6], acc[n][7]);
        }
    };
    float* s_out = reinterpret_cast<float*>(smem_raw);
    float* s_tmp = s_out + 16 * 128;
    if (warp < 2) store_acc_smem((warp == 0) ? s_out : s_tmp);
    __syncthreads();
#pragma unroll
    for (int r = 0; r < kQRows; ++r) s_out[r * 128 + tid] += s_tmp[r * 128 + tid];
    __syncthreads();
    if (warp == 2) store_acc_smem(s_tmp);
    __syncthreads();
#pragma unroll
    for (int r = 0; r < kQRows; ++r) s_out[r * 128 + tid] += s_tmp[r * 128 + tid];
    __syncthreads();
    if (warp == 3) store_acc_smem(s_tmp);
    __syncthreads();
#pragma unroll
    for (int r = 0; r < kQRows; ++r) {
        float* dst = split_out + (((size_t) (r / 3) * split_n + sp) * n_head + (h0 + (r % 3))) * 130;
        dst[tid] = s_out[r * 128 + tid] + s_tmp[r * 128 + tid];
        if (tid == 0) {
            dst[128] = s_mg[r];
            dst[129] = s_lg[r];
        }
    }
#elif !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750
    using namespace nvcuda;
    extern __shared__ __align__(16) char smem_raw[];
    // 21,760 B <= 22,016 B (4 blk/SM on V100/Ada): sQ (16*136*2 = 4,352 B) | sKV_all (4*16*136*2 = 17,408 B).
    __half* sQ = reinterpret_cast<__half*>(smem_raw);
    __half* sKV_all = sQ + 16 * 136;

    const int hkv = blockIdx.x, sp = blockIdx.y;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int h0 = 3 * hkv;
    constexpr float kLog2e = 1.4426950408889634f;
    const float qscale_all = kq_scale * kLog2e;

    constexpr int kQRows = 3 * T_ACT;
    const int p0 = pos0[0];
    const int total = p0 + T_ACT;
    const int chunk_sz = max(64, (total + split_n - 1) / split_n);
    const int c_beg = sp * chunk_sz;
    if (c_beg >= total) {
        if (tid < kQRows) {
            float* dst = split_out + (((size_t) (tid / 3) * split_n + sp) * n_head + (h0 + (tid % 3))) * 130;
            dst[128] = -1e30f;
            dst[129] = 0.0f;
        }
        return;
    }
    const int c_end = min(total, c_beg + chunk_sz);

    {
        const int qr = tid >> 3, qc0 = (tid & 7) * 16;
        const float* qp = (qr < kQRows) ? (Q + ((size_t) (qr / 3) * n_head + (h0 + (qr % 3))) * 128 + qc0) : nullptr;
        __half2* dst2 = reinterpret_cast<__half2*>(sQ + qr * 136 + qc0);
        if (qp) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = *reinterpret_cast<const float4*>(qp + i * 4);
                dst2[i * 2 + 0] = __halves2half2(__float2half(v.x * qscale_all), __float2half(v.y * qscale_all));
                dst2[i * 2 + 1] = __halves2half2(__float2half(v.z * qscale_all), __float2half(v.w * qscale_all));
            }
        } else {
            const __half2 z2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));
#pragma unroll
            for (int i = 0; i < 8; ++i) dst2[i] = z2;
        }
    }
    __syncthreads();

    int acc_r0, acc_r1, acc_c0, acc_c1;
    wmma_acc_layout_16x16(lane, acc_r0, acc_r1, acc_c0, acc_c1);
    const bool is_leader = wmma_row_leader_16(lane);

    const bool has_sink = (sp == 0 && warp == 0);
    float rm0 = has_sink ? 0.0f : -1e30f;
    float rm1 = rm0;
    float rl0 = has_sink ? 1.0f : 0.0f;
    float rl1 = rl0;

    auto load_nsink = [&](int r) -> float {
        return (r < kQRows) ? (-__ldg(&sink[(size_t) (r / 3) * n_head + (h0 + (r % 3))]) * kLog2e) : 0.0f;
    };

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
#pragma unroll
    for (int n = 0; n < 8; ++n) wmma::fill_fragment(acc[n], 0.0f);

    __half* sKV = sKV_all + warp * 16 * 136;
    const int wr = lane >> 1;          // 0..15 (2 lanes/row, 64 bytes/lane = 4 uint4s)
    const int wc0 = (lane & 1) * 64;
    const __half2 neg1152 = __half2half2(__float2half(-1152.0f));
    const int bulk_end = min(c_end, max(c_beg, p0 - n_ext + 1));

    int jb = c_beg + warp * 16;
    // Bulk loop: ZERO __syncthreads(), shifted-nsink (0 bulk adds), V prefetch overlapping QK^T!
    if (jb <= bulk_end - 16) {
        const float ns0 = load_nsink(acc_r0), ns1 = load_nsink(acc_r1);
        if (rm0 > -1e20f) rm0 -= ns0;
        if (rm1 > -1e20f) rm1 -= ns1;
        for (; jb <= bulk_end - 16; jb += 64) {
            const int j = jb + wr;
            {
                const int hj = hkv * cap + j;
                const __half2 dk2 = __half2half2(__ldg(&ks[hj]));
                const __half2 kb = __hmul2(neg1152, dk2);
                const int8_t* kp = kq + (size_t) hj * 128 + wc0;
                __half* dst = sKV + wr * 136 + wc0;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(kp + i * 16);
                    __half2 h2[8];
                    i8x4_to_h2x2_fma(raw.x, dk2, kb, h2[0], h2[1]); i8x4_to_h2x2_fma(raw.y, dk2, kb, h2[2], h2[3]);
                    i8x4_to_h2x2_fma(raw.z, dk2, kb, h2[4], h2[5]); i8x4_to_h2x2_fma(raw.w, dk2, kb, h2[6], h2[7]);
                    *reinterpret_cast<uint4*>(dst + i * 16)     = *reinterpret_cast<const uint4*>(h2);
                    *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                }
            }
            __syncwarp();

            const int hjv = hkv * cap + j;
            const __half2 dv2 = __half2half2(__ldg(&vs[hjv]));
            const int8_t* vp = vq + (size_t) hjv * 128 + wc0;
            const uint4 vr0 = *reinterpret_cast<const uint4*>(vp);
            const uint4 vr1 = *reinterpret_cast<const uint4*>(vp + 16);
            const uint4 vr2 = *reinterpret_cast<const uint4*>(vp + 32);
            const uint4 vr3 = *reinterpret_cast<const uint4*>(vp + 48);

            wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
            wmma::fill_fragment(fc, 0.0f);
#pragma unroll
            for (int k = 0; k < 128; k += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                wmma::load_matrix_sync(fa, sQ + k, 136);
                wmma::load_matrix_sync(fb, sKV + k, 136);
                wmma::mma_sync(fc, fa, fb, fc);
            }
            __syncwarp();

            {
                const __half2 vb = __hmul2(neg1152, dv2);
                __half* dst = sKV + wr * 136 + wc0;
                __half2 h2[8];
                i8x4_to_h2x2_fma(vr0.x, dv2, vb, h2[0], h2[1]); i8x4_to_h2x2_fma(vr0.y, dv2, vb, h2[2], h2[3]);
                i8x4_to_h2x2_fma(vr0.z, dv2, vb, h2[4], h2[5]); i8x4_to_h2x2_fma(vr0.w, dv2, vb, h2[6], h2[7]);
                *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(h2); *reinterpret_cast<uint4*>(dst + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                i8x4_to_h2x2_fma(vr1.x, dv2, vb, h2[0], h2[1]); i8x4_to_h2x2_fma(vr1.y, dv2, vb, h2[2], h2[3]);
                i8x4_to_h2x2_fma(vr1.z, dv2, vb, h2[4], h2[5]); i8x4_to_h2x2_fma(vr1.w, dv2, vb, h2[6], h2[7]);
                *reinterpret_cast<uint4*>(dst + 16) = *reinterpret_cast<const uint4*>(h2); *reinterpret_cast<uint4*>(dst + 24) = *reinterpret_cast<const uint4*>(h2 + 4);
                i8x4_to_h2x2_fma(vr2.x, dv2, vb, h2[0], h2[1]); i8x4_to_h2x2_fma(vr2.y, dv2, vb, h2[2], h2[3]);
                i8x4_to_h2x2_fma(vr2.z, dv2, vb, h2[4], h2[5]); i8x4_to_h2x2_fma(vr2.w, dv2, vb, h2[6], h2[7]);
                *reinterpret_cast<uint4*>(dst + 32) = *reinterpret_cast<const uint4*>(h2); *reinterpret_cast<uint4*>(dst + 40) = *reinterpret_cast<const uint4*>(h2 + 4);
                i8x4_to_h2x2_fma(vr3.x, dv2, vb, h2[0], h2[1]); i8x4_to_h2x2_fma(vr3.y, dv2, vb, h2[2], h2[3]);
                i8x4_to_h2x2_fma(vr3.z, dv2, vb, h2[4], h2[5]); i8x4_to_h2x2_fma(vr3.w, dv2, vb, h2[6], h2[7]);
                *reinterpret_cast<uint4*>(dst + 48) = *reinterpret_cast<const uint4*>(h2); *reinterpret_cast<uint4*>(dst + 56) = *reinterpret_cast<const uint4*>(h2 + 4);
            }

            const float mx0 = wmma_row_max_16(fmaxf(fmaxf(fc.x[0], fc.x[1]), fmaxf(fc.x[4], fc.x[5])));
            const float mx1 = wmma_row_max_16(fmaxf(fmaxf(fc.x[2], fc.x[3]), fmaxf(fc.x[6], fc.x[7])));
            const float m_new0 = fmaxf(rm0, mx0), m_new1 = fmaxf(rm1, mx1);
            const float a0 = (rm0 > -1e20f) ? ((m_new0 == rm0) ? 1.0f : fast_exp2f_hw(rm0 - m_new0)) : 0.0f;
            const float a1 = (rm1 > -1e20f) ? ((m_new1 == rm1) ? 1.0f : fast_exp2f_hw(rm1 - m_new1)) : 0.0f;
            if (__any_sync(0xffffffffu, a0 != 1.0f || a1 != 1.0f)) {
#pragma unroll
                for (int n = 0; n < 8; ++n) wmma_rescale_acc(acc[n], a0, a1);
            }
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa_p;
            float ps0 = 0.0f, ps1 = 0.0f;
            wmma_acc_to_matA_16x16<false>(fc, m_new0, m_new1, ps0, ps1, fa_p, lane);
            rm0 = m_new0; rl0 = rl0 * a0 + wmma_row_sum_16(ps0);
            rm1 = m_new1; rl1 = rl1 * a1 + wmma_row_sum_16(ps1);
            __syncwarp();

#pragma unroll
            for (int n = 0; n < 8; ++n) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
                wmma::load_matrix_sync(fb, sKV + n * 16, 136);
                wmma::mma_sync(acc[n], fa_p, fb, acc[n]);
            }
            __syncwarp();
        }
        if (rm0 > -1e20f) rm0 += ns0;
        if (rm1 > -1e20f) rm1 += ns1;
    }

    // Tail loop (causal + rel_tab; #pragma unroll 1 keeps bulk loop at 128 regs / 0 spills).
    if (jb < c_end) {
        auto mask_bias = [&](float s, int r, int j) -> float {
            if (r >= kQRows || j >= c_end) return -1e30f;
            const int rt = r / 3, h = h0 + (r % 3), pos_q = p0 + rt;
            if (j > pos_q) return -1e30f;
            const int d = pos_q - j;
            const float b = (d < n_ext)
                ? (__ldg(&rel_tab[((size_t) rt * n_head + h) * n_ext + d]) * kLog2e)
                : (-__ldg(&sink[(size_t) rt * n_head + h]) * kLog2e);
            return s + b;
        };
        for (; jb < c_end; jb += 64) {
            const int j = jb + wr;
            {
                __half* dst = sKV + wr * 136 + wc0;
                if (j < c_end) {
                    const int hj = hkv * cap + j;
                    const __half2 dk2 = __half2half2(__ldg(&ks[hj]));
                    const __half2 kb = __hmul2(neg1152, dk2);
                    const int8_t* kp = kq + (size_t) hj * 128 + wc0;
#pragma unroll 1
                    for (int i = 0; i < 4; ++i) {
                        const uint4 raw = *reinterpret_cast<const uint4*>(kp + i * 16);
                        __half2 h2[8];
                        i8x4_to_h2x2_fma(raw.x, dk2, kb, h2[0], h2[1]); i8x4_to_h2x2_fma(raw.y, dk2, kb, h2[2], h2[3]);
                        i8x4_to_h2x2_fma(raw.z, dk2, kb, h2[4], h2[5]); i8x4_to_h2x2_fma(raw.w, dk2, kb, h2[6], h2[7]);
                        *reinterpret_cast<uint4*>(dst + i * 16)     = *reinterpret_cast<const uint4*>(h2);
                        *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                    }
                } else {
                    const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll 1
                    for (int i = 0; i < 8; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
                }
            }
            __syncwarp();
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
            wmma::fill_fragment(fc, 0.0f);
#pragma unroll 1
            for (int k = 0; k < 128; k += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                wmma::load_matrix_sync(fa, sQ + k, 136);
                wmma::load_matrix_sync(fb, sKV + k, 136);
                wmma::mma_sync(fc, fa, fb, fc);
            }
            __syncwarp();
            {
                __half* dst = sKV + wr * 136 + wc0;
                if (j < c_end) {
                    const int hj = hkv * cap + j;
                    const __half2 dv2 = __half2half2(__ldg(&vs[hj]));
                    const __half2 vb = __hmul2(neg1152, dv2);
                    const int8_t* vp = vq + (size_t) hj * 128 + wc0;
#pragma unroll 1
                    for (int i = 0; i < 4; ++i) {
                        const uint4 raw = *reinterpret_cast<const uint4*>(vp + i * 16);
                        __half2 h2[8];
                        i8x4_to_h2x2_fma(raw.x, dv2, vb, h2[0], h2[1]); i8x4_to_h2x2_fma(raw.y, dv2, vb, h2[2], h2[3]);
                        i8x4_to_h2x2_fma(raw.z, dv2, vb, h2[4], h2[5]); i8x4_to_h2x2_fma(raw.w, dv2, vb, h2[6], h2[7]);
                        *reinterpret_cast<uint4*>(dst + i * 16)     = *reinterpret_cast<const uint4*>(h2);
                        *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                    }
                } else {
                    const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll 1
                    for (int i = 0; i < 8; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
                }
            }
            const int j0 = jb + acc_c0, j1 = jb + acc_c1;
            fc.x[0] = mask_bias(fc.x[0], acc_r0, j0);     fc.x[1] = mask_bias(fc.x[1], acc_r0, j0 + 1);
            fc.x[4] = mask_bias(fc.x[4], acc_r0, j1);     fc.x[5] = mask_bias(fc.x[5], acc_r0, j1 + 1);
            fc.x[2] = mask_bias(fc.x[2], acc_r1, j0);     fc.x[3] = mask_bias(fc.x[3], acc_r1, j0 + 1);
            fc.x[6] = mask_bias(fc.x[6], acc_r1, j1);     fc.x[7] = mask_bias(fc.x[7], acc_r1, j1 + 1);
            const float mx0 = wmma_row_max_16(fmaxf(fmaxf(fc.x[0], fc.x[1]), fmaxf(fc.x[4], fc.x[5])));
            const float mx1 = wmma_row_max_16(fmaxf(fmaxf(fc.x[2], fc.x[3]), fmaxf(fc.x[6], fc.x[7])));
            const float m_new0 = fmaxf(rm0, mx0), m_new1 = fmaxf(rm1, mx1);
            const float a0 = (rm0 > -1e20f) ? ((m_new0 == rm0) ? 1.0f : fast_exp2f_hw(rm0 - m_new0)) : 0.0f;
            const float a1 = (rm1 > -1e20f) ? ((m_new1 == rm1) ? 1.0f : fast_exp2f_hw(rm1 - m_new1)) : 0.0f;
            if (__any_sync(0xffffffffu, a0 != 1.0f || a1 != 1.0f)) {
#pragma unroll
                for (int n = 0; n < 8; ++n) wmma_rescale_acc(acc[n], a0, a1);
            }
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa_p;
            float ps0 = 0.0f, ps1 = 0.0f;
            wmma_acc_to_matA_16x16<true>(fc, m_new0, m_new1, ps0, ps1, fa_p, lane);
            rm0 = m_new0; rl0 = rl0 * a0 + wmma_row_sum_16(ps0);
            rm1 = m_new1; rl1 = rl1 * a1 + wmma_row_sum_16(ps1);
            __syncwarp();

#pragma unroll
            for (int n = 0; n < 8; ++n) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
                wmma::load_matrix_sync(fb, sKV + n * 16, 136);
                wmma::mma_sync(acc[n], fa_p, fb, acc[n]);
            }
            __syncwarp();
        }
    }

    // Single 4-warp epilogue combine (reuses sQ for s_wm/s_wl and sKV_all for s_out + s_tmp = 16,384 B <= 17,408 B).
    __syncthreads();
    float* s_wm = reinterpret_cast<float*>(sQ);          // [4][16] = 256 B
    float* s_wl = s_wm + 64;                             // [4][16] = 256 B
    float* s_mg = s_wl + 64;                             // [16]    = 64 B
    float* s_lg = s_mg + 16;                             // [16]    = 64 B
    if (is_leader) {
        s_wm[warp * 16 + acc_r0] = rm0; s_wl[warp * 16 + acc_r0] = rl0;
        s_wm[warp * 16 + acc_r1] = rm1; s_wl[warp * 16 + acc_r1] = rl1;
    }
    __syncthreads();
    if (tid < kQRows) {
        float mg = -1e30f;
#pragma unroll
        for (int w = 0; w < 4; ++w) if (s_wl[w * 16 + tid] > 0.0f) mg = fmaxf(mg, s_wm[w * 16 + tid]);
        float lg = 0.0f;
#pragma unroll
        for (int w = 0; w < 4; ++w) if (s_wl[w * 16 + tid] > 0.0f) lg += s_wl[w * 16 + tid] * fast_exp2f_hw(s_wm[w * 16 + tid] - mg);
        s_mg[tid] = mg;
        s_lg[tid] = lg;
    }
    __syncthreads();
    {
        const float cw0 = (acc_r0 < kQRows && rl0 > 0.0f) ? fast_exp2f_hw(rm0 - s_mg[acc_r0]) : 0.0f;
        const float cw1 = (acc_r1 < kQRows && rl1 > 0.0f) ? fast_exp2f_hw(rm1 - s_mg[acc_r1]) : 0.0f;
#pragma unroll
        for (int n = 0; n < 8; ++n) wmma_rescale_acc(acc[n], cw0, cw1);
    }
    float* s_out = reinterpret_cast<float*>(sKV_all);
    float* s_tmp = s_out + 16 * 128;
    if (warp == 0) {
#pragma unroll
        for (int n = 0; n < 8; ++n) wmma::store_matrix_sync(s_out + n * 16, acc[n], 128, wmma::mem_row_major);
    } else if (warp == 1) {
#pragma unroll
        for (int n = 0; n < 8; ++n) wmma::store_matrix_sync(s_tmp + n * 16, acc[n], 128, wmma::mem_row_major);
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < kQRows; ++r) s_out[r * 128 + tid] += s_tmp[r * 128 + tid];
    __syncthreads();
    if (warp == 2) {
#pragma unroll
        for (int n = 0; n < 8; ++n) wmma::store_matrix_sync(s_tmp + n * 16, acc[n], 128, wmma::mem_row_major);
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < kQRows; ++r) s_out[r * 128 + tid] += s_tmp[r * 128 + tid];
    __syncthreads();
    if (warp == 3) {
#pragma unroll
        for (int n = 0; n < 8; ++n) wmma::store_matrix_sync(s_tmp + n * 16, acc[n], 128, wmma::mem_row_major);
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < kQRows; ++r) {
        float* dst = split_out + (((size_t) (r / 3) * split_n + sp) * n_head + (h0 + (r % 3))) * 130;
        dst[tid] = s_out[r * 128 + tid] + s_tmp[r * 128 + tid];
        if (tid == 0) {
            dst[128] = s_mg[r];
            dst[129] = s_lg[r];
        }
    }
#endif
}

/// Zero-Sync Warp-Independent Tensor-Core (WMMA 16x16x16) Split-KV Q8 SWA verify attention for T == 4 (is_swa=1, GQA-3, D=128):
/// Packs 12 query rows (3 GQA heads x 4 tokens) into one 16x16x16 WMMA tile (12x less KV HBM traffic than scalar SWA!),
/// 4 independent warps (ZERO __syncthreads() in KV loop, 21,760 B SMEM, 4 blk/SM), V prefetch overlapping QK^T, ring-buffer slot.
template <int T_ACT = 4>
__global__ void __launch_bounds__(128, 4) k_attn_q8_gqa3_verify4_swa_wmma_split(
        const float* __restrict__ Q,
        const int8_t* __restrict__ kq, const __half* __restrict__ ks,
        const int8_t* __restrict__ vq, const __half* __restrict__ vs,
        const float* __restrict__ rel_tab, const float* __restrict__ sink,
        float* __restrict__ split_out,
        int n_head, int n_head_kv, int cap, int n_swa, int n_ext,
        const int32_t* __restrict__ pos0, float kq_scale, int split_n) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
    // Volta sm_70 native PTX HMMA.884 SWA verify4: STRIDE=144 (23,040 B = 4 blk/SM, 0-bank-conflict LDS.64),
    // V prefetch overlapping QK^T + K-next prefetch overlapping P*V, STACK:0, direct float2 epilogue.
    constexpr int STRIDE = 144;
    extern __shared__ __align__(16) char smem_raw[];
    __half* sQ = reinterpret_cast<__half*>(smem_raw);
    __half* sKV_all = sQ + 16 * STRIDE;

    const int hkv = blockIdx.x, sp = blockIdx.y;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int h0 = 3 * hkv, kv_dim = n_head_kv * 128;
    constexpr float kLog2e = 1.4426950408889634f;
    const float qscale_all = kq_scale * kLog2e;

    constexpr int kQRows = 3 * T_ACT;
    const int p0 = pos0[0];
    const int j_min = max(0, p0 - n_swa + 1);
    const int j_max = p0 + T_ACT;
    const int n_tot = j_max - j_min;
    const int chunk_sz = max(64, (n_tot + split_n - 1) / split_n);
    const int c_beg = j_min + sp * chunk_sz;
    if (c_beg >= j_max) {
        if (tid < kQRows) {
            float* dst = split_out + (((size_t) (tid / 3) * split_n + sp) * n_head + (h0 + (tid % 3))) * 130;
            dst[128] = -1e30f;
            dst[129] = 0.0f;
        }
        return;
    }
    const int c_end = min(j_max, c_beg + chunk_sz);

    {
        const int qr = tid >> 3, qc0 = (tid & 7) * 16;
        const float* qp = (qr < kQRows) ? (Q + ((size_t) (qr / 3) * n_head + (h0 + (qr % 3))) * 128 + qc0) : nullptr;
        __half2* dst2 = reinterpret_cast<__half2*>(sQ + qr * STRIDE + qc0);
        if (qp) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = *reinterpret_cast<const float4*>(qp + i * 4);
                dst2[i * 2 + 0] = __halves2half2(__float2half(v.x * qscale_all), __float2half(v.y * qscale_all));
                dst2[i * 2 + 1] = __halves2half2(__float2half(v.z * qscale_all), __float2half(v.w * qscale_all));
            }
        } else {
            const __half2 z2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));
#pragma unroll
            for (int i = 0; i < 8; ++i) dst2[i] = z2;
        }
    }
    __syncthreads();

    auto mma_rc = [](float d[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, {%0, %1, %2, %3, %4, %5, %6, %7};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
            : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
    };
    auto mma_rr = [](float d[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
            "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, {%0, %1, %2, %3, %4, %5, %6, %7};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
            : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
    };

    const int acc_r0 = ((lane & 16) >> 2) + ((lane & 4) << 1) + (lane & 1);
    const int acc_r1 = acc_r0 + 2;
    const int acc_c0 = (lane & 8) + (lane & 2);
    const int acc_c1 = acc_c0 + 4;
    const bool is_leader = (lane & 10) == 0;
    const int qa_row = ((lane & 16) >> 2) + ((lane & 4) << 1) + (lane & 3);
    const int kb_row = ((lane & 16) >> 2) + (lane & 8) + (lane & 3);
    const int vb_row = lane & 3;
    const int vb_col = (lane & 8) + ((lane & 16) >> 2);
    const bool hi8 = (lane & 8) != 0;
    const bool want_r1 = (lane & 2) != 0;

    const bool has_sink = (sp == 0 && warp == 0);
    float rm0 = has_sink ? 0.0f : -1e30f;
    float rm1 = rm0;
    float rl0 = has_sink ? 1.0f : 0.0f;
    float rl1 = rl0;

    float acc[8][8];
#pragma unroll
    for (int n = 0; n < 8; ++n) {
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[n][i] = 0.0f;
    }

    __half* sKV = sKV_all + warp * 16 * STRIDE;
    const int qa_off = qa_row * STRIDE;
    const int kb_off = kb_row * STRIDE;
    const int vb_off = vb_row * STRIDE + vb_col;
    const int wr = lane >> 1;
    const int wc0 = (lane & 1) * 64;
    const __half2 neg1152 = __half2half2(__float2half(-1152.0f));
    const bool pow2_cap = (cap & (cap - 1)) == 0;
    const int cap_mask = cap - 1;

    auto mask_bias = [&](float s, int r, int j) -> float {
        if (r >= kQRows || j >= c_end) return -1e30f;
        const int rt = r / 3, h = h0 + (r % 3), pos_q = p0 + rt;
        const int d = pos_q - j;
        if (d < 0 || d >= n_swa) return -1e30f;
        const float b = (d < n_ext)
            ? (__ldg(&rel_tab[((size_t) rt * n_head + h) * n_ext + d]) * kLog2e)
            : (-__ldg(&sink[(size_t) rt * n_head + h]) * kLog2e);
        return s + b;
    };

    for (int jb = c_beg + warp * 16; jb < c_end; jb += 64) {
        const int j = jb + wr;
        const bool valid_j = (j < c_end);
        const int slot = valid_j ? (pow2_cap ? (j & cap_mask) : (j % cap)) : 0;
        {
            __half* dst = sKV + wr * STRIDE + wc0;
            if (valid_j) {
                const __half2 dk2 = __half2half2(__ldg(&ks[(size_t) slot * n_head_kv + hkv]));
                const __half2 kb = __hmul2(neg1152, dk2);
                const int8_t* kp = kq + (size_t) slot * kv_dim + hkv * 128 + wc0;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(kp + i * 16);
                    __half2 h2[8];
                    i8x4_to_h2x2_fma(raw.x, dk2, kb, h2[0], h2[1]); i8x4_to_h2x2_fma(raw.y, dk2, kb, h2[2], h2[3]);
                    i8x4_to_h2x2_fma(raw.z, dk2, kb, h2[4], h2[5]); i8x4_to_h2x2_fma(raw.w, dk2, kb, h2[6], h2[7]);
                    *reinterpret_cast<uint4*>(dst + i * 16)     = *reinterpret_cast<const uint4*>(h2);
                    *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                }
            } else {
                const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int i = 0; i < 8; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
            }
        }
        __syncwarp();

        uint4 vr0 = make_uint4(0, 0, 0, 0), vr1 = vr0, vr2 = vr0, vr3 = vr0;
        __half2 dv2 = __half2half2(__float2half(0.0f));
        if (valid_j) {
            dv2 = __half2half2(__ldg(&vs[(size_t) slot * n_head_kv + hkv]));
            const int8_t* vp = vq + (size_t) slot * kv_dim + hkv * 128 + wc0;
            vr0 = *reinterpret_cast<const uint4*>(vp);
            vr1 = *reinterpret_cast<const uint4*>(vp + 16);
            vr2 = *reinterpret_cast<const uint4*>(vp + 32);
            vr3 = *reinterpret_cast<const uint4*>(vp + 48);
        }

        float fc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 1
        for (int k = 0; k < 128; k += 8) {
            const uint4 q0 = *reinterpret_cast<const uint4*>(sQ + qa_off + k);
            const uint4 k0 = *reinterpret_cast<const uint4*>(sKV + kb_off + k);
            mma_rc(fc, q0.x, q0.y, k0.x, k0.y);
            mma_rc(fc, q0.z, q0.w, k0.z, k0.w);
        }
        __syncwarp();

        {
            __half* dst = sKV + wr * STRIDE + wc0;
            if (valid_j) {
                const __half2 vb = __hmul2(neg1152, dv2);
                __half2 h2[8];
                i8x4_to_h2x2_fma(vr0.x, dv2, vb, h2[0], h2[1]); i8x4_to_h2x2_fma(vr0.y, dv2, vb, h2[2], h2[3]);
                i8x4_to_h2x2_fma(vr0.z, dv2, vb, h2[4], h2[5]); i8x4_to_h2x2_fma(vr0.w, dv2, vb, h2[6], h2[7]);
                *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(h2); *reinterpret_cast<uint4*>(dst + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                i8x4_to_h2x2_fma(vr1.x, dv2, vb, h2[0], h2[1]); i8x4_to_h2x2_fma(vr1.y, dv2, vb, h2[2], h2[3]);
                i8x4_to_h2x2_fma(vr1.z, dv2, vb, h2[4], h2[5]); i8x4_to_h2x2_fma(vr1.w, dv2, vb, h2[6], h2[7]);
                *reinterpret_cast<uint4*>(dst + 16) = *reinterpret_cast<const uint4*>(h2); *reinterpret_cast<uint4*>(dst + 24) = *reinterpret_cast<const uint4*>(h2 + 4);
                i8x4_to_h2x2_fma(vr2.x, dv2, vb, h2[0], h2[1]); i8x4_to_h2x2_fma(vr2.y, dv2, vb, h2[2], h2[3]);
                i8x4_to_h2x2_fma(vr2.z, dv2, vb, h2[4], h2[5]); i8x4_to_h2x2_fma(vr2.w, dv2, vb, h2[6], h2[7]);
                *reinterpret_cast<uint4*>(dst + 32) = *reinterpret_cast<const uint4*>(h2); *reinterpret_cast<uint4*>(dst + 40) = *reinterpret_cast<const uint4*>(h2 + 4);
                i8x4_to_h2x2_fma(vr3.x, dv2, vb, h2[0], h2[1]); i8x4_to_h2x2_fma(vr3.y, dv2, vb, h2[2], h2[3]);
                i8x4_to_h2x2_fma(vr3.z, dv2, vb, h2[4], h2[5]); i8x4_to_h2x2_fma(vr3.w, dv2, vb, h2[6], h2[7]);
                *reinterpret_cast<uint4*>(dst + 48) = *reinterpret_cast<const uint4*>(h2); *reinterpret_cast<uint4*>(dst + 56) = *reinterpret_cast<const uint4*>(h2 + 4);
            } else {
                const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int i = 0; i < 8; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
            }
        }

        const int j0 = jb + acc_c0, j1 = jb + acc_c1;
        fc[0] = mask_bias(fc[0], acc_r0, j0);     fc[1] = mask_bias(fc[1], acc_r0, j0 + 1);
        fc[4] = mask_bias(fc[4], acc_r0, j1);     fc[5] = mask_bias(fc[5], acc_r0, j1 + 1);
        fc[2] = mask_bias(fc[2], acc_r1, j0);     fc[3] = mask_bias(fc[3], acc_r1, j0 + 1);
        fc[6] = mask_bias(fc[6], acc_r1, j1);     fc[7] = mask_bias(fc[7], acc_r1, j1 + 1);

        const float mx0 = wmma_row_max_16(fmaxf(fmaxf(fc[0], fc[1]), fmaxf(fc[4], fc[5])));
        const float mx1 = wmma_row_max_16(fmaxf(fmaxf(fc[2], fc[3]), fmaxf(fc[6], fc[7])));
        const float m_new0 = fmaxf(rm0, mx0), m_new1 = fmaxf(rm1, mx1);
        const float a0 = (rm0 > -1e20f) ? ((m_new0 == rm0) ? 1.0f : fast_exp2f_hw(rm0 - m_new0)) : 0.0f;
        const float a1 = (rm1 > -1e20f) ? ((m_new1 == rm1) ? 1.0f : fast_exp2f_hw(rm1 - m_new1)) : 0.0f;
        if (__any_sync(0xffffffffu, a0 != 1.0f || a1 != 1.0f)) {
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                acc[n][0] *= a0; acc[n][1] *= a0; acc[n][4] *= a0; acc[n][5] *= a0;
                acc[n][2] *= a1; acc[n][3] *= a1; acc[n][6] *= a1; acc[n][7] *= a1;
            }
        }
        const float p00 = (fc[0] > -1e20f) ? fast_exp2f_hw(fc[0] - m_new0) : 0.0f;
        const float p01 = (fc[1] > -1e20f) ? fast_exp2f_hw(fc[1] - m_new0) : 0.0f;
        const float p04 = (fc[4] > -1e20f) ? fast_exp2f_hw(fc[4] - m_new0) : 0.0f;
        const float p05 = (fc[5] > -1e20f) ? fast_exp2f_hw(fc[5] - m_new0) : 0.0f;
        const float p12 = (fc[2] > -1e20f) ? fast_exp2f_hw(fc[2] - m_new1) : 0.0f;
        const float p13 = (fc[3] > -1e20f) ? fast_exp2f_hw(fc[3] - m_new1) : 0.0f;
        const float p16 = (fc[6] > -1e20f) ? fast_exp2f_hw(fc[6] - m_new1) : 0.0f;
        const float p17 = (fc[7] > -1e20f) ? fast_exp2f_hw(fc[7] - m_new1) : 0.0f;
        rm0 = m_new0; rl0 = rl0 * a0 + wmma_row_sum_16((p00 + p01) + (p04 + p05));
        rm1 = m_new1; rl1 = rl1 * a1 + wmma_row_sum_16((p12 + p13) + (p16 + p17));

        const uint32_t u0 = f2_to_h2u(p00, p01), u1 = f2_to_h2u(p04, p05);
        const uint32_t v0 = f2_to_h2u(p12, p13), v1 = f2_to_h2u(p16, p17);
        const uint32_t r0 = __shfl_xor_sync(0xffffffffu, want_r1 ? u0 : v0, 2);
        const uint32_t r1 = __shfl_xor_sync(0xffffffffu, want_r1 ? u1 : v1, 2);
        const uint32_t h0u = want_r1 ? r0 : u0, h1u = want_r1 ? v0 : r0;
        const uint32_t h2u = want_r1 ? r1 : u1, h3u = want_r1 ? v1 : r1;
        const uint32_t o0 = __shfl_xor_sync(0xffffffffu, h0u, 8), o1 = __shfl_xor_sync(0xffffffffu, h1u, 8);
        const uint32_t o2 = __shfl_xor_sync(0xffffffffu, h2u, 8), o3 = __shfl_xor_sync(0xffffffffu, h3u, 8);
        const uint32_t pa0 = hi8 ? o0 : h0u, pa1 = hi8 ? o1 : h1u;
        const uint32_t pa2 = hi8 ? o2 : h2u, pa3 = hi8 ? o3 : h3u;
        const uint32_t pa4 = hi8 ? h0u : o0, pa5 = hi8 ? h1u : o1;
        const uint32_t pa6 = hi8 ? h2u : o2, pa7 = hi8 ? h3u : o3;
        __syncwarp();

        uint2 b[2];
#pragma unroll
        for (int nb = 0; nb < 8; nb += 2) {
            b[0] = *reinterpret_cast<const uint2*>(sKV + vb_off + 0 * STRIDE + (nb + 0) * 16);
            b[1] = *reinterpret_cast<const uint2*>(sKV + vb_off + 0 * STRIDE + (nb + 1) * 16);
            mma_rr(acc[nb + 0], pa0, pa1, b[0].x, b[0].y);
            mma_rr(acc[nb + 1], pa0, pa1, b[1].x, b[1].y);
            b[0] = *reinterpret_cast<const uint2*>(sKV + vb_off + 4 * STRIDE + (nb + 0) * 16);
            b[1] = *reinterpret_cast<const uint2*>(sKV + vb_off + 4 * STRIDE + (nb + 1) * 16);
            mma_rr(acc[nb + 0], pa2, pa3, b[0].x, b[0].y);
            mma_rr(acc[nb + 1], pa2, pa3, b[1].x, b[1].y);
            b[0] = *reinterpret_cast<const uint2*>(sKV + vb_off + 8 * STRIDE + (nb + 0) * 16);
            b[1] = *reinterpret_cast<const uint2*>(sKV + vb_off + 8 * STRIDE + (nb + 1) * 16);
            mma_rr(acc[nb + 0], pa4, pa5, b[0].x, b[0].y);
            mma_rr(acc[nb + 1], pa4, pa5, b[1].x, b[1].y);
            b[0] = *reinterpret_cast<const uint2*>(sKV + vb_off + 12 * STRIDE + (nb + 0) * 16);
            b[1] = *reinterpret_cast<const uint2*>(sKV + vb_off + 12 * STRIDE + (nb + 1) * 16);
            mma_rr(acc[nb + 0], pa6, pa7, b[0].x, b[0].y);
            mma_rr(acc[nb + 1], pa6, pa7, b[1].x, b[1].y);
        }
        __syncwarp();
    }

    __syncthreads();
    float* s_wm = reinterpret_cast<float*>(smem_raw + 16384);
    float* s_wl = s_wm + 64;
    float* s_mg = s_wl + 64;
    float* s_lg = s_mg + 16;
    if (is_leader) {
        s_wm[warp * 16 + acc_r0] = rm0; s_wl[warp * 16 + acc_r0] = rl0;
        s_wm[warp * 16 + acc_r1] = rm1; s_wl[warp * 16 + acc_r1] = rl1;
    }
    __syncthreads();
    if (tid < kQRows) {
        float mg = -1e30f;
#pragma unroll
        for (int w = 0; w < 4; ++w) if (s_wl[w * 16 + tid] > 0.0f) mg = fmaxf(mg, s_wm[w * 16 + tid]);
        float lg = 0.0f;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const float lw = s_wl[w * 16 + tid];
            if (lw > 0.0f) lg += lw * fast_exp2f_hw(s_wm[w * 16 + tid] - mg);
        }
        s_mg[tid] = mg;
        s_lg[tid] = lg;
    }
    __syncthreads();
    {
        const float cw0 = (acc_r0 < kQRows && rl0 > 0.0f) ? fast_exp2f_hw(rm0 - s_mg[acc_r0]) : 0.0f;
        const float cw1 = (acc_r1 < kQRows && rl1 > 0.0f) ? fast_exp2f_hw(rm1 - s_mg[acc_r1]) : 0.0f;
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            acc[n][0] *= cw0; acc[n][1] *= cw0; acc[n][4] *= cw0; acc[n][5] *= cw0;
            acc[n][2] *= cw1; acc[n][3] *= cw1; acc[n][6] *= cw1; acc[n][7] *= cw1;
        }
    }
    auto store_acc_smem = [&](float* base_s) {
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            float* p0s = base_s + acc_r0 * 128 + n * 16;
            float* p1s = base_s + acc_r1 * 128 + n * 16;
            *reinterpret_cast<float2*>(p0s + acc_c0) = make_float2(acc[n][0], acc[n][1]);
            *reinterpret_cast<float2*>(p0s + acc_c1) = make_float2(acc[n][4], acc[n][5]);
            *reinterpret_cast<float2*>(p1s + acc_c0) = make_float2(acc[n][2], acc[n][3]);
            *reinterpret_cast<float2*>(p1s + acc_c1) = make_float2(acc[n][6], acc[n][7]);
        }
    };
    float* s_out = reinterpret_cast<float*>(smem_raw);
    float* s_tmp = s_out + 16 * 128;
    if (warp < 2) store_acc_smem((warp == 0) ? s_out : s_tmp);
    __syncthreads();
#pragma unroll
    for (int r = 0; r < kQRows; ++r) s_out[r * 128 + tid] += s_tmp[r * 128 + tid];
    __syncthreads();
    if (warp == 2) store_acc_smem(s_tmp);
    __syncthreads();
#pragma unroll
    for (int r = 0; r < kQRows; ++r) s_out[r * 128 + tid] += s_tmp[r * 128 + tid];
    __syncthreads();
    if (warp == 3) store_acc_smem(s_tmp);
    __syncthreads();
#pragma unroll
    for (int r = 0; r < kQRows; ++r) {
        float* dst = split_out + (((size_t) (r / 3) * split_n + sp) * n_head + (h0 + (r % 3))) * 130;
        dst[tid] = s_out[r * 128 + tid] + s_tmp[r * 128 + tid];
        if (tid == 0) {
            dst[128] = s_mg[r];
            dst[129] = s_lg[r];
        }
    }
#elif !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750
    using namespace nvcuda;
    extern __shared__ __align__(16) char smem_raw[];
    __half* sQ = reinterpret_cast<__half*>(smem_raw);
    __half* sKV_all = sQ + 16 * 136;

    const int hkv = blockIdx.x, sp = blockIdx.y;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int h0 = 3 * hkv, kv_dim = n_head_kv * 128;
    constexpr float kLog2e = 1.4426950408889634f;
    const float qscale_all = kq_scale * kLog2e;

    constexpr int kQRows = 3 * T_ACT;
    const int p0 = pos0[0];
    const int j_min = max(0, p0 - n_swa + 1);
    const int j_max = p0 + T_ACT;
    const int n_tot = j_max - j_min;
    const int chunk_sz = max(64, (n_tot + split_n - 1) / split_n);
    const int c_beg = j_min + sp * chunk_sz;
    if (c_beg >= j_max) {
        if (tid < kQRows) {
            float* dst = split_out + (((size_t) (tid / 3) * split_n + sp) * n_head + (h0 + (tid % 3))) * 130;
            dst[128] = -1e30f;
            dst[129] = 0.0f;
        }
        return;
    }
    const int c_end = min(j_max, c_beg + chunk_sz);

    {
        const int qr = tid >> 3, qc0 = (tid & 7) * 16;
        const float* qp = (qr < kQRows) ? (Q + ((size_t) (qr / 3) * n_head + (h0 + (qr % 3))) * 128 + qc0) : nullptr;
        __half2* dst2 = reinterpret_cast<__half2*>(sQ + qr * 136 + qc0);
        if (qp) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = *reinterpret_cast<const float4*>(qp + i * 4);
                dst2[i * 2 + 0] = __halves2half2(__float2half(v.x * qscale_all), __float2half(v.y * qscale_all));
                dst2[i * 2 + 1] = __halves2half2(__float2half(v.z * qscale_all), __float2half(v.w * qscale_all));
            }
        } else {
            const __half2 z2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));
#pragma unroll
            for (int i = 0; i < 8; ++i) dst2[i] = z2;
        }
    }
    __syncthreads();

    int acc_r0, acc_r1, acc_c0, acc_c1;
    wmma_acc_layout_16x16(lane, acc_r0, acc_r1, acc_c0, acc_c1);
    const bool is_leader = wmma_row_leader_16(lane);

    const bool has_sink = (sp == 0 && warp == 0);
    float rm0 = has_sink ? 0.0f : -1e30f;
    float rm1 = rm0;
    float rl0 = has_sink ? 1.0f : 0.0f;
    float rl1 = rl0;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
#pragma unroll
    for (int n = 0; n < 8; ++n) wmma::fill_fragment(acc[n], 0.0f);

    __half* sKV = sKV_all + warp * 16 * 136;
    const int wr = lane >> 1;
    const int wc0 = (lane & 1) * 64;
    const __half2 neg1152 = __half2half2(__float2half(-1152.0f));
    const bool pow2_cap = (cap & (cap - 1)) == 0;
    const int cap_mask = cap - 1;

    auto mask_bias = [&](float s, int r, int j) -> float {
        if (r >= kQRows || j >= c_end) return -1e30f;
        const int rt = r / 3, h = h0 + (r % 3), pos_q = p0 + rt;
        const int d = pos_q - j;
        if (d < 0 || d >= n_swa) return -1e30f;
        const float b = (d < n_ext)
            ? (__ldg(&rel_tab[((size_t) rt * n_head + h) * n_ext + d]) * kLog2e)
            : (-__ldg(&sink[(size_t) rt * n_head + h]) * kLog2e);
        return s + b;
    };

    for (int jb = c_beg + warp * 16; jb < c_end; jb += 64) {
        const int j = jb + wr;
        const bool valid_j = (j < c_end);
        const int slot = valid_j ? (pow2_cap ? (j & cap_mask) : (j % cap)) : 0;
        {
            __half* dst = sKV + wr * 136 + wc0;
            if (valid_j) {
                const __half2 dk2 = __half2half2(__ldg(&ks[(size_t) slot * n_head_kv + hkv]));
                const __half2 kb = __hmul2(neg1152, dk2);
                const int8_t* kp = kq + (size_t) slot * kv_dim + hkv * 128 + wc0;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(kp + i * 16);
                    __half2 h2[8];
                    i8x4_to_h2x2_fma(raw.x, dk2, kb, h2[0], h2[1]); i8x4_to_h2x2_fma(raw.y, dk2, kb, h2[2], h2[3]);
                    i8x4_to_h2x2_fma(raw.z, dk2, kb, h2[4], h2[5]); i8x4_to_h2x2_fma(raw.w, dk2, kb, h2[6], h2[7]);
                    *reinterpret_cast<uint4*>(dst + i * 16)     = *reinterpret_cast<const uint4*>(h2);
                    *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                }
            } else {
                const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int i = 0; i < 8; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
            }
        }
        __syncwarp();

        // Prefetch V in registers overlapping QK^T WMMA!
        uint4 vr0 = make_uint4(0, 0, 0, 0), vr1 = vr0, vr2 = vr0, vr3 = vr0;
        __half2 dv2 = __half2half2(__float2half(0.0f));
        if (valid_j) {
            dv2 = __half2half2(__ldg(&vs[(size_t) slot * n_head_kv + hkv]));
            const int8_t* vp = vq + (size_t) slot * kv_dim + hkv * 128 + wc0;
            vr0 = *reinterpret_cast<const uint4*>(vp);
            vr1 = *reinterpret_cast<const uint4*>(vp + 16);
            vr2 = *reinterpret_cast<const uint4*>(vp + 32);
            vr3 = *reinterpret_cast<const uint4*>(vp + 48);
        }

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
        wmma::fill_fragment(fc, 0.0f);
#pragma unroll
        for (int k = 0; k < 128; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
            wmma::load_matrix_sync(fa, sQ + k, 136);
            wmma::load_matrix_sync(fb, sKV + k, 136);
            wmma::mma_sync(fc, fa, fb, fc);
        }
        __syncwarp();

        {
            __half* dst = sKV + wr * 136 + wc0;
            if (valid_j) {
                const __half2 vb = __hmul2(neg1152, dv2);
                __half2 h2[8];
                i8x4_to_h2x2_fma(vr0.x, dv2, vb, h2[0], h2[1]); i8x4_to_h2x2_fma(vr0.y, dv2, vb, h2[2], h2[3]);
                i8x4_to_h2x2_fma(vr0.z, dv2, vb, h2[4], h2[5]); i8x4_to_h2x2_fma(vr0.w, dv2, vb, h2[6], h2[7]);
                *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(h2); *reinterpret_cast<uint4*>(dst + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                i8x4_to_h2x2_fma(vr1.x, dv2, vb, h2[0], h2[1]); i8x4_to_h2x2_fma(vr1.y, dv2, vb, h2[2], h2[3]);
                i8x4_to_h2x2_fma(vr1.z, dv2, vb, h2[4], h2[5]); i8x4_to_h2x2_fma(vr1.w, dv2, vb, h2[6], h2[7]);
                *reinterpret_cast<uint4*>(dst + 16) = *reinterpret_cast<const uint4*>(h2); *reinterpret_cast<uint4*>(dst + 24) = *reinterpret_cast<const uint4*>(h2 + 4);
                i8x4_to_h2x2_fma(vr2.x, dv2, vb, h2[0], h2[1]); i8x4_to_h2x2_fma(vr2.y, dv2, vb, h2[2], h2[3]);
                i8x4_to_h2x2_fma(vr2.z, dv2, vb, h2[4], h2[5]); i8x4_to_h2x2_fma(vr2.w, dv2, vb, h2[6], h2[7]);
                *reinterpret_cast<uint4*>(dst + 32) = *reinterpret_cast<const uint4*>(h2); *reinterpret_cast<uint4*>(dst + 40) = *reinterpret_cast<const uint4*>(h2 + 4);
                i8x4_to_h2x2_fma(vr3.x, dv2, vb, h2[0], h2[1]); i8x4_to_h2x2_fma(vr3.y, dv2, vb, h2[2], h2[3]);
                i8x4_to_h2x2_fma(vr3.z, dv2, vb, h2[4], h2[5]); i8x4_to_h2x2_fma(vr3.w, dv2, vb, h2[6], h2[7]);
                *reinterpret_cast<uint4*>(dst + 48) = *reinterpret_cast<const uint4*>(h2); *reinterpret_cast<uint4*>(dst + 56) = *reinterpret_cast<const uint4*>(h2 + 4);
            } else {
                const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int i = 0; i < 8; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
            }
        }

        const int j0 = jb + acc_c0, j1 = jb + acc_c1;
        fc.x[0] = mask_bias(fc.x[0], acc_r0, j0);     fc.x[1] = mask_bias(fc.x[1], acc_r0, j0 + 1);
        fc.x[4] = mask_bias(fc.x[4], acc_r0, j1);     fc.x[5] = mask_bias(fc.x[5], acc_r0, j1 + 1);
        fc.x[2] = mask_bias(fc.x[2], acc_r1, j0);     fc.x[3] = mask_bias(fc.x[3], acc_r1, j0 + 1);
        fc.x[6] = mask_bias(fc.x[6], acc_r1, j1);     fc.x[7] = mask_bias(fc.x[7], acc_r1, j1 + 1);

        const float mx0 = wmma_row_max_16(fmaxf(fmaxf(fc.x[0], fc.x[1]), fmaxf(fc.x[4], fc.x[5])));
        const float mx1 = wmma_row_max_16(fmaxf(fmaxf(fc.x[2], fc.x[3]), fmaxf(fc.x[6], fc.x[7])));
        const float m_new0 = fmaxf(rm0, mx0), m_new1 = fmaxf(rm1, mx1);
        const float a0 = (rm0 > -1e20f) ? ((m_new0 == rm0) ? 1.0f : fast_exp2f_hw(rm0 - m_new0)) : 0.0f;
        const float a1 = (rm1 > -1e20f) ? ((m_new1 == rm1) ? 1.0f : fast_exp2f_hw(rm1 - m_new1)) : 0.0f;
        if (__any_sync(0xffffffffu, a0 != 1.0f || a1 != 1.0f)) {
#pragma unroll
            for (int n = 0; n < 8; ++n) wmma_rescale_acc(acc[n], a0, a1);
        }
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa_p;
        float ps0 = 0.0f, ps1 = 0.0f;
        wmma_acc_to_matA_16x16<true>(fc, m_new0, m_new1, ps0, ps1, fa_p, lane);
        rm0 = m_new0; rl0 = rl0 * a0 + wmma_row_sum_16(ps0);
        rm1 = m_new1; rl1 = rl1 * a1 + wmma_row_sum_16(ps1);
        __syncwarp();

#pragma unroll
        for (int n = 0; n < 8; ++n) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
            wmma::load_matrix_sync(fb, sKV + n * 16, 136);
            wmma::mma_sync(acc[n], fa_p, fb, acc[n]);
        }
        __syncwarp();
    }

    // Single 4-warp epilogue combine: s_out + s_tmp = 16,384 B at [0..16383], scalars at [16384..16991].
    __syncthreads();
    float* s_wm = reinterpret_cast<float*>(smem_raw + 16384);
    float* s_wl = s_wm + 64;
    float* s_mg = s_wl + 64;
    float* s_lg = s_mg + 16;
    if (is_leader) {
        s_wm[warp * 16 + acc_r0] = rm0; s_wl[warp * 16 + acc_r0] = rl0;
        s_wm[warp * 16 + acc_r1] = rm1; s_wl[warp * 16 + acc_r1] = rl1;
    }
    __syncthreads();
    if (tid < kQRows) {
        float mg = -1e30f;
#pragma unroll
        for (int w = 0; w < 4; ++w) if (s_wl[w * 16 + tid] > 0.0f) mg = fmaxf(mg, s_wm[w * 16 + tid]);
        float lg = 0.0f;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const float lw = s_wl[w * 16 + tid];
            if (lw > 0.0f) lg += lw * fast_exp2f_hw(s_wm[w * 16 + tid] - mg);
        }
        s_mg[tid] = mg;
        s_lg[tid] = lg;
    }
    __syncthreads();
    {
        const float cw0 = (acc_r0 < kQRows && rl0 > 0.0f) ? fast_exp2f_hw(rm0 - s_mg[acc_r0]) : 0.0f;
        const float cw1 = (acc_r1 < kQRows && rl1 > 0.0f) ? fast_exp2f_hw(rm1 - s_mg[acc_r1]) : 0.0f;
#pragma unroll
        for (int n = 0; n < 8; ++n) wmma_rescale_acc(acc[n], cw0, cw1);
    }
    float* s_out = reinterpret_cast<float*>(smem_raw);
    float* s_tmp = s_out + 16 * 128;
    if (warp < 2) {
        float* dst_s = (warp == 0) ? s_out : s_tmp;
#pragma unroll
        for (int n = 0; n < 8; ++n) wmma::store_matrix_sync(dst_s + n * 16, acc[n], 128, wmma::mem_row_major);
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < kQRows; ++r) s_out[r * 128 + tid] += s_tmp[r * 128 + tid];
    __syncthreads();
    if (warp == 2) {
#pragma unroll
        for (int n = 0; n < 8; ++n) wmma::store_matrix_sync(s_tmp + n * 16, acc[n], 128, wmma::mem_row_major);
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < kQRows; ++r) s_out[r * 128 + tid] += s_tmp[r * 128 + tid];
    __syncthreads();
    if (warp == 3) {
#pragma unroll
        for (int n = 0; n < 8; ++n) wmma::store_matrix_sync(s_tmp + n * 16, acc[n], 128, wmma::mem_row_major);
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < kQRows; ++r) {
        float* dst = split_out + (((size_t) (r / 3) * split_n + sp) * n_head + (h0 + (r % 3))) * 130;
        dst[tid] = s_out[r * 128 + tid] + s_tmp[r * 128 + tid];
        if (tid == 0) {
            dst[128] = s_mg[r];
            dst[129] = s_lg[r];
        }
    }
#endif
}

/// Combine Split-KV partials across active splits and apply fused sigmoid gate (+ optional direct Q8_1 store): grid (n_head, T), block 128.
/// Cooperative SMEM staging computes mg, exp2f(ms-mg), and lg ONCE per split (128x fewer exp2f + skips zero-weight splits).
__global__ void k_attn_split_combine(const float* __restrict__ split_out, const float* __restrict__ gate,
                                     float* __restrict__ out, int n_head, int split_n,
                                     const int32_t* __restrict__ pos0, int is_swa, int n_swa,
                                     void* __restrict__ xq_out = nullptr) {
    __shared__ float s_m[kAttnSplitMax];
    __shared__ float s_l[kAttnSplitMax];
    __shared__ float s_w[kAttnSplitMax];
    __shared__ float s_red[4];
    __shared__ float s_red_l[4];
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const int warp = d >> 5, lane = d & 31;
    const int pos_q = pos0[0] + t;
    const int n_tok = is_swa ? min(pos_q + 1, n_swa) : (pos_q + 1);
    const int chunk_sz = is_swa ? max(32, (n_tok + split_n - 1) / split_n) : max(64, (n_tok + split_n - 1) / split_n);
    const int active_splits = (is_swa && split_n <= 24) ? split_n : min(split_n, (n_tok + chunk_sz - 1) / chunk_sz);

    const size_t stride_s = (size_t) n_head * 130;
    const float* base_h = split_out + ((size_t) t * split_n * n_head + h) * 130;

    float local_mg = -1e30f;
    for (int s = d; s < active_splits; s += 128) {
        const float* src = base_h + (size_t) s * stride_s;
        const float ms = src[128];
        const float ls = src[129];
        s_m[s] = ms;
        s_l[s] = ls;
        if (ls > 0.0f && ms > local_mg) local_mg = ms;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) local_mg = fmaxf(local_mg, __shfl_xor_sync(0xffffffffu, local_mg, o));
    if (lane == 0) s_red[warp] = local_mg;
    __syncthreads();
    const float mg = fmaxf(fmaxf(s_red[0], s_red[1]), fmaxf(s_red[2], s_red[3]));

    float local_lg = 0.0f;
    for (int s = d; s < active_splits; s += 128) {
        const float ls = s_l[s];
        float w = 0.0f;
        if (ls > 0.0f) {
            const float diff = s_m[s] - mg;
            w = (diff > -126.0f) ? exp2f(diff) : 0.0f;
            local_lg += ls * w;
        }
        s_w[s] = w;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) local_lg += __shfl_xor_sync(0xffffffffu, local_lg, o);
    if (lane == 0) s_red_l[warp] = local_lg;
    __syncthreads();
    const float lg = (s_red_l[0] + s_red_l[1]) + (s_red_l[2] + s_red_l[3]);

    float og = 0.0f;
    const float* base_d = base_h + d;
    for (int s = 0; s < active_splits; ++s) {
        const float w = s_w[s];
        if (w != 0.0f) og += base_d[(size_t) s * stride_s] * w;
    }
    const size_t oi = ((size_t) t * n_head + h) * 128 + d;
    float val = __fdividef(og, lg);
    if (gate) val *= sigmf(gate[oi]);
    out[oi] = val;
    if (xq_out) {
        q8_1_warp_store(val, xq_out, (t * n_head + h) * 4 + (d >> 5), d & 31);
    }
}

/// 1-pass SWA Q8 attention (T <= 8, is_swa=1, head_dim=128, n_head=3*n_head_kv) with fused sigmoid gate + optional Q8_1 store:
/// Grid (n_head, T) = 96 CTAs at T=4, block 128 (8 half-warps). Eliminates split_out, k_attn_split_combine, AND native_quantize_q8_1 before O!
__global__ void __launch_bounds__(128, 8) k_attn_q8_gqa3_swa_1pass(
    const float* __restrict__ Q,
    const int8_t* __restrict__ kq, const __half* __restrict__ ks,
    const int8_t* __restrict__ vq, const __half* __restrict__ vs,
    const float* __restrict__ rel_tab, const float* __restrict__ sink,
    const float* __restrict__ gate, float* __restrict__ out, void* __restrict__ xq_out,
    int n_head, int n_head_kv, int n_ext, int cap, int n_swa,
    const int32_t* __restrict__ pos0, float kq_scale) {
    __shared__ __half2 sq2[64];
    __shared__ float s_m[8];
    __shared__ float s_l[8];
    __shared__ float s_acc[8][128];
    const int h = blockIdx.x, t = blockIdx.y;
    const int hkv = h / 3;
    const int tid = threadIdx.x;
    const int hw = tid >> 4;
    const int hlane = tid & 15;
    const int kv_dim = n_head_kv * 128;
    constexpr float kLog2e = 1.4426950408889634f;
    const float kq_l2 = kq_scale * kLog2e;

    if (tid < 64) {
        const float2 qf = *reinterpret_cast<const float2*>(Q + ((size_t) t * n_head + h) * 128 + tid * 2);
        sq2[tid] = __float22half2_rn(qf);
    }
    __syncthreads();

    const int base8 = hlane * 8;
    const __half2 q0 = sq2[hlane * 4 + 0];
    const __half2 q1 = sq2[hlane * 4 + 1];
    const __half2 q2 = sq2[hlane * 4 + 2];
    const __half2 q3 = sq2[hlane * 4 + 3];

    const int pos_q = pos0[0] + t;
    const int j0 = max(0, pos_q - n_swa + 1);
    const int j1 = pos_q + 1;
    const float* rt = rel_tab + ((size_t) t * n_head + h) * n_ext;
    const float neg_sink2 = -__ldg(&sink[(size_t) t * n_head + h]) * kLog2e;

    const __half2 z2 = __half2half2(__float2half(0.0f));
    float m = (hw == 0) ? 0.0f : -1e30f;
    float l = (hw == 0) ? 1.0f : 0.0f;
    __half2 acc0 = z2, acc1 = z2, acc2 = z2, acc3 = z2;

    const bool pow2_cap = (cap & (cap - 1)) == 0;
    const int cap_mask = cap - 1;
    const unsigned hw_mask = (hw & 1) ? 0xffff0000u : 0x0000ffffu;

    for (int j = j0 + hw; j < j1; j += 8) {
        const int slot = pow2_cap ? (j & cap_mask) : (j % cap);
        const size_t idx = (size_t) slot * n_head_kv + hkv;
        const float dkkq2 = __half2float(__ldg(&ks[idx])) * kq_l2;
        const float d_v = __half2float(__ldg(&vs[idx]));
        const uint2 k8 = *reinterpret_cast<const uint2*>(kq + (size_t) slot * kv_dim + hkv * 128 + base8);
        const uint2 v8 = *reinterpret_cast<const uint2*>(vq + (size_t) slot * kv_dim + hkv * 128 + base8);
        __half2 kh0, kh1, kh2, kh3, vh0, vh1, vh2, vh3;
        i8x4_to_h2x2(k8.x, kh0, kh1);
        i8x4_to_h2x2(k8.y, kh2, kh3);
        i8x4_to_h2x2(v8.x, vh0, vh1);
        i8x4_to_h2x2(v8.y, vh2, vh3);
        __half2 d2 = __hmul2(q0, kh0);
        d2 = __hfma2(q1, kh1, d2);
        d2 = __hfma2(q2, kh2, d2);
        d2 = __hfma2(q3, kh3, d2);
        float dt = __half2float(__low2half(d2)) + __half2float(__high2half(d2));
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) dt += __shfl_xor_sync(hw_mask, dt, o, 16);
        const int dist = pos_q - j;
        const float bias = dist < n_ext ? (__ldg(&rt[dist]) * kLog2e) : neg_sink2;
        const float sc = dt * dkkq2 + bias;
        float p;
        if (sc > m) {
            const float alpha = l > 0.0f ? fast_exp2f_hw(m - sc) : 0.0f;
            const __half2 a2 = __half2half2(__float2half(alpha));
            l = l * alpha + 1.0f;
            acc0 = __hmul2(acc0, a2);
            acc1 = __hmul2(acc1, a2);
            acc2 = __hmul2(acc2, a2);
            acc3 = __hmul2(acc3, a2);
            m = sc;
            p = 1.0f;
        } else {
            p = fast_exp2f_hw(sc - m);
            l += p;
        }
        const __half2 pv2 = __half2half2(__float2half(p * d_v));
        acc0 = __hfma2(pv2, vh0, acc0);
        acc1 = __hfma2(pv2, vh1, acc1);
        acc2 = __hfma2(pv2, vh2, acc2);
        acc3 = __hfma2(pv2, vh3, acc3);
    }

    if (hlane == 0) { s_m[hw] = m; s_l[hw] = l; }
    {
        const float2 f0 = __half22float2(acc0), f1 = __half22float2(acc1);
        const float2 f2 = __half22float2(acc2), f3 = __half22float2(acc3);
        s_acc[hw][base8 + 0] = f0.x; s_acc[hw][base8 + 1] = f0.y;
        s_acc[hw][base8 + 2] = f1.x; s_acc[hw][base8 + 3] = f1.y;
        s_acc[hw][base8 + 4] = f2.x; s_acc[hw][base8 + 5] = f2.y;
        s_acc[hw][base8 + 6] = f3.x; s_acc[hw][base8 + 7] = f3.y;
    }
    __syncthreads();

    float mg = -1e30f;
#pragma unroll
    for (int w = 0; w < 8; ++w) if (s_l[w] > 0.0f && s_m[w] > mg) mg = s_m[w];
    float lg = 0.0f, og = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        if (s_l[w] > 0.0f) {
            const float c = exp2f(s_m[w] - mg);
            lg += s_l[w] * c;
            og += s_acc[w][tid] * c;
        }
    }
    const size_t oi = ((size_t) t * n_head + h) * 128 + tid;
    float val = __fdividef(og, lg);
    if (gate) val *= sigmf(gate[oi]);
    out[oi] = val;
    if (xq_out) {
        q8_1_warp_store(val, xq_out, (t * n_head + h) * 4 + (tid >> 5), tid & 31);
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

/// Zero-spill MoE router for Inkling2: sigmoid(logits) + exp_probs_b -> top-k -> normalize weights -> * scale.
/// n_exp <= 128, top_k <= 16. 8 warps (8 tokens) per CTA: grid((T + 7) / 8), block 256. STACK: 0!
__global__ void k_route_inkling2(const float* __restrict__ logits, const float* __restrict__ bias,
                                 int32_t* __restrict__ ids, float* __restrict__ ewts,
                                 int n_exp, int top_k, float scale, int T,
                                 uint32_t* __restrict__ d_hits = nullptr) {
    __shared__ float s_wlocal[8][16];
    __shared__ uint32_t s_hits[128];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int t = (int) blockIdx.x * 8 + warp;
    if (d_hits && tid < 128) s_hits[tid] = 0u;
    if (d_hits) __syncthreads();
    if (t < T) {
        float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
        float s0 = -1e30f, s1 = -1e30f, s2 = -1e30f, s3 = -1e30f;
        const float* lg = logits + (size_t) t * n_exp;
        if (lane < n_exp) { p0 = sigmf(lg[lane]); s0 = p0 + (bias ? bias[lane] : 0.0f); }
        if (32 + lane < n_exp) { p1 = sigmf(lg[32 + lane]); s1 = p1 + (bias ? bias[32 + lane] : 0.0f); }
        if (64 + lane < n_exp) { p2 = sigmf(lg[64 + lane]); s2 = p2 + (bias ? bias[64 + lane] : 0.0f); }
        if (96 + lane < n_exp) { p3 = sigmf(lg[96 + lane]); s3 = p3 + (bias ? bias[96 + lane] : 0.0f); }

        float wsum = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            float bv = s0; int be = lane; float bp = p0;
            if (s1 > bv) { bv = s1; be = 32 + lane; bp = p1; }
            if (s2 > bv) { bv = s2; be = 64 + lane; bp = p2; }
            if (s3 > bv) { bv = s3; be = 96 + lane; bp = p3; }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const int oe = __shfl_xor_sync(0xffffffffu, be, o);
                const float op = __shfl_xor_sync(0xffffffffu, bp, o);
                if (ov > bv || (ov == bv && oe < be)) { bv = ov; be = oe; bp = op; }
            }
            if (lane == 0) {
                ids[(size_t) t * top_k + k] = be;
                if (d_hits) atomicAdd(&s_hits[be], 1u);
                s_wlocal[warp][k] = bp;
                wsum += bp;
            }
            if ((be & 31) == lane) {
                const int q = be >> 5;
                if (q == 0) s0 = -1e30f;
                else if (q == 1) s1 = -1e30f;
                else if (q == 2) s2 = -1e30f;
                else s3 = -1e30f;
            }
        }
        const float wsum_b = __shfl_sync(0xffffffffu, wsum, 0);
        const float fac = scale / fmaxf(wsum_b, 6.103515625e-5f);
        __syncwarp();
        if (lane < top_k) {
            ewts[(size_t) t * top_k + lane] = s_wlocal[warp][lane] * fac;
        }
    }
    if (d_hits) {
        __syncthreads();
        if (tid < n_exp && s_hits[tid] > 0u) atomicAdd(&d_hits[tid], s_hits[tid]);
    }
}

/// Fused T=1 Router GEMV (64 CTAs x 128 thr, 2 rows/CTA) + Warp Top-K Route + Group (1 single launch, STACK:0, CUDA-Graph safe!).
/// Exact same per-row dot + block_sum FP32 order as k_gemv_f32_cta (100% bit-exact), fits 1 single wave on V100 80 SMs,
/// halves x loads & d_done atomic contention (128 -> 64), and replaces 128x ld.acquire.gpu with single release-acquire fence.
__global__ void __launch_bounds__(128, 4) k_router_gemv_and_group_t1(
    const float* __restrict__ W, const float* __restrict__ x, float* __restrict__ logits,
    const float* __restrict__ bias, int32_t* __restrict__ ids, float* __restrict__ ewts,
    const int32_t* __restrict__ slot_of, const uint8_t* __restrict__ blobs,
    size_t stride, int n_in, int n_exp, int top_k, float scale,
    unsigned long long* __restrict__ grp_ptr, int32_t* __restrict__ grp_start,
    int32_t* __restrict__ n_groups, int32_t* __restrict__ ent_dst,
    int32_t* __restrict__ ent_tok, uint32_t* __restrict__ d_hits, uint32_t* __restrict__ d_done) {
    __shared__ float sh0[4], sh1[4];
    __shared__ int s_is_last;
    const int o0 = (int) blockIdx.x * 2;
    if (o0 >= n_exp) return;
    const int o1 = o0 + 1;
    const bool has1 = (o1 < n_exp);
    const float4* wr0 = reinterpret_cast<const float4*>(W + (size_t) o0 * n_in);
    const float4* wr1 = has1 ? reinterpret_cast<const float4*>(W + (size_t) o1 * n_in) : wr0;
    const float4* xr = reinterpret_cast<const float4*>(x);
    float acc0 = 0.0f, acc1 = 0.0f;
    const int n4 = n_in >> 2;
    for (int i = threadIdx.x; i < n4; i += 128) {
        const float4 b = xr[i];
        const float4 a0 = wr0[i];
        acc0 += a0.x * b.x + a0.y * b.y + a0.z * b.z + a0.w * b.w;
        if (has1) {
            const float4 a1 = wr1[i];
            acc1 += a1.x * b.x + a1.y * b.y + a1.z * b.z + a1.w * b.w;
        }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        acc0 += __shfl_xor_sync(0xffffffffu, acc0, o);
        acc1 += __shfl_xor_sync(0xffffffffu, acc1, o);
    }
    if (lane == 0) { sh0[warp] = acc0; sh1[warp] = acc1; }
    __syncthreads();
    if (warp == 0) {
        acc0 = (lane < 4) ? sh0[lane] : 0.0f;
        acc1 = (lane < 4) ? sh1[lane] : 0.0f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            acc0 += __shfl_xor_sync(0xffffffffu, acc0, o);
            acc1 += __shfl_xor_sync(0xffffffffu, acc1, o);
        }
        if (lane == 0) {
            const uint32_t expected_ctas = (uint32_t) ((n_exp + 1) >> 1);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
            asm volatile("st.relaxed.gpu.global.f32 [%0], %1;" :: "l"(logits + o0), "f"(acc0) : "memory");
            if (has1) asm volatile("st.relaxed.gpu.global.f32 [%0], %1;" :: "l"(logits + o1), "f"(acc1) : "memory");
            uint32_t prev = 0u;
            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(prev) : "l"(d_done) : "memory");
            const int last = ((prev + 1u) == expected_ctas) ? 1 : 0;
            if (last) {
                asm volatile("st.relaxed.gpu.global.u32 [%0], 0;" :: "l"(d_done) : "memory");
            }
#else
            *((volatile float*) (logits + o0)) = acc0;
            if (has1) *((volatile float*) (logits + o1)) = acc1;
            __threadfence();
            const uint32_t prev = atomicAdd(d_done, 1u);
            const int last = ((prev + 1u) == expected_ctas) ? 1 : 0;
            if (last) {
                *((volatile uint32_t*) d_done) = 0u;
                __threadfence();
            }
#endif
            s_is_last = last;
        }
    }
    __syncthreads();
    if (!s_is_last || warp != 0) return;

    __shared__ int32_t s_e[16];
    __shared__ float s_p[16];
    __shared__ float s_fac;
    float p[4], s[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int e = j * 32 + lane;
        if (e < n_exp) {
            float lg_val;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
            asm volatile("ld.relaxed.gpu.global.f32 %0, [%1];" : "=f"(lg_val) : "l"(logits + e) : "memory");
#else
            lg_val = __ldcg(logits + e);
#endif
            const float pr = sigmf(lg_val);
            p[j] = pr;
            s[j] = pr + (bias ? bias[e] : 0.0f);
        } else {
            p[j] = 0.0f;
            s[j] = -1e30f;
        }
    }
    float wsum = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        float bv = s[0];
        int be = lane;
        float bp = p[0];
#pragma unroll
        for (int j = 1; j < 4; ++j) {
            if (s[j] > bv) { bv = s[j]; be = j * 32 + lane; bp = p[j]; }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            const int oe = __shfl_xor_sync(0xffffffffu, be, off);
            const float op = __shfl_xor_sync(0xffffffffu, bp, off);
            if (ov > bv || (ov == bv && oe < be)) { bv = ov; be = oe; bp = op; }
        }
        if (lane == 0) {
            s_e[k] = be;
            s_p[k] = bp;
            ids[k] = be;
            wsum += bp;
        }
        if ((be & 31) == lane) {
            const int b = be >> 5;
            if (b == 0) s[0] = -1e30f;
            else if (b == 1) s[1] = -1e30f;
            else if (b == 2) s[2] = -1e30f;
            else s[3] = -1e30f;
        }
    }
    if (lane == 0) s_fac = scale / fmaxf(wsum, 6.103515625e-5f);
    __syncwarp();
    int sl = -1;
    if (lane < top_k) {
        const int be = s_e[lane];
        ewts[lane] = s_p[lane] * s_fac;
        if (d_hits) d_hits[be] += 1u;
        sl = slot_of ? slot_of[be] : -1;
    }
    if (slot_of) {
        const unsigned valid_mask = __ballot_sync(0xffffffffu, sl >= 0);
        if (sl >= 0) {
            const int g = __popc(valid_mask & ((1u << lane) - 1u));
            grp_ptr[g] = (unsigned long long) (uintptr_t) (blobs + (size_t) sl * stride);
            grp_start[g] = g;
            ent_dst[g] = lane;
            ent_tok[g] = 0;
        }
        if (lane == 0) {
            const int ng = __popc(valid_mask);
            grp_start[ng] = ng;
            *n_groups = ng;
        }
    }
}

/// Zero-spill, zero-barrier, parallel 8-lane 1-warp router + grouping for T == 1 (top_k <= 16, n_exp <= 128):
/// Uses tiny SMEM (132 B) so sel_e/sel_p never spill to local stack (STACK: 0), then lanes 0..top_k-1 look up
/// slot_of[be] and emit compacted grp_ptr/grp_start/ent_dst/ent_tok in parallel via __ballot_sync/__popc!
__global__ void k_route_and_group_t1_warp(const float* __restrict__ logits, const float* __restrict__ bias,
                                          int32_t* __restrict__ ids, float* __restrict__ ewts,
                                          const int32_t* __restrict__ slot_of, const uint8_t* __restrict__ blobs,
                                          size_t stride, int n_exp, int top_k, float scale,
                                          unsigned long long* __restrict__ grp_ptr, int32_t* __restrict__ grp_start,
                                          int32_t* __restrict__ n_groups, int32_t* __restrict__ ent_dst,
                                          int32_t* __restrict__ ent_tok, uint32_t* __restrict__ d_hits = nullptr) {
    __shared__ int32_t s_e[16];
    __shared__ float s_p[16];
    __shared__ float s_fac;
    const int lane = threadIdx.x & 31;
    float p[4], s[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int e = j * 32 + lane;
        if (e < n_exp) {
            const float pr = sigmf(logits[e]);
            p[j] = pr;
            s[j] = pr + (bias ? bias[e] : 0.0f);
        } else {
            p[j] = 0.0f;
            s[j] = -1e30f;
        }
    }
    float wsum = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        float bv = s[0];
        int be = lane;
        float bp = p[0];
#pragma unroll
        for (int j = 1; j < 4; ++j) {
            if (s[j] > bv) { bv = s[j]; be = j * 32 + lane; bp = p[j]; }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oe = __shfl_xor_sync(0xffffffffu, be, o);
            const float op = __shfl_xor_sync(0xffffffffu, bp, o);
            if (ov > bv || (ov == bv && oe < be)) { bv = ov; be = oe; bp = op; }
        }
        if (lane == 0) {
            s_e[k] = be;
            s_p[k] = bp;
            ids[k] = be;
            wsum += bp;
        }
        if ((be & 31) == lane) {
            const int b = be >> 5;
            if (b == 0) s[0] = -1e30f;
            else if (b == 1) s[1] = -1e30f;
            else if (b == 2) s[2] = -1e30f;
            else s[3] = -1e30f;
        }
    }
    if (lane == 0) s_fac = scale / fmaxf(wsum, 6.103515625e-5f);
    __syncwarp();
    int sl = -1;
    if (lane < top_k) {
        const int be = s_e[lane];
        ewts[lane] = s_p[lane] * s_fac;
        if (d_hits) d_hits[be] += 1u;
        sl = slot_of[be];
    }
    const unsigned valid_mask = __ballot_sync(0xffffffffu, sl >= 0);
    if (sl >= 0) {
        const int g = __popc(valid_mask & ((1u << lane) - 1u));
        grp_ptr[g] = (unsigned long long) (uintptr_t) (blobs + (size_t) sl * stride);
        grp_start[g] = g;
        ent_dst[g] = lane;
        ent_tok[g] = 0;
    }
    if (lane == 0) {
        const int ng = __popc(valid_mask);
        grp_start[ng] = ng;
        *n_groups = ng;
    }
}

/// Fused k_route_inkling2 + k_group_experts in 1 CTA for full-VRAM small-T (T <= 8, n_exp <= 128, top_k <= 16):
/// Block = 128 (T <= 4) or 256 (T <= 8). Zero global ids roundtrip, zero second kernel launch!
__global__ void k_route_and_group_smallT(const float* __restrict__ logits, const float* __restrict__ bias,
                                         int32_t* __restrict__ ids, float* __restrict__ ewts,
                                         const int32_t* __restrict__ slot_of, const uint8_t* __restrict__ blobs,
                                         size_t stride, int T, int n_exp, int top_k, float scale,
                                         unsigned long long* __restrict__ grp_ptr, int32_t* __restrict__ grp_start,
                                         int32_t* __restrict__ n_groups, int32_t* __restrict__ ent_dst,
                                         int32_t* __restrict__ ent_tok, uint32_t* __restrict__ d_hits = nullptr) {
    __shared__ int32_t s_ids[128];
    __shared__ float s_wlocal[8][16];
    __shared__ float s_fac[8];
    __shared__ int32_t cnt[128];
    __shared__ int32_t slot_id[128];
    __shared__ int32_t cur_off[128];
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    if (tid < n_exp) {
        cnt[tid] = 0;
        slot_id[tid] = slot_of[tid];
    }
    if (warp < T) {
        const int t = warp;
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
        float wsum = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            float bv = s[0];
            int be = lane;
            float bp = p[0];
#pragma unroll
            for (int j = 1; j < 4; ++j) {
                if (s[j] > bv) { bv = s[j]; be = j * 32 + lane; bp = p[j]; }
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const int oe = __shfl_xor_sync(0xffffffffu, be, o);
                const float op = __shfl_xor_sync(0xffffffffu, bp, o);
                if (ov > bv || (ov == bv && oe < be)) { bv = ov; be = oe; bp = op; }
            }
            if (lane == 0) {
                const int eidx = t * top_k + k;
                ids[eidx] = be;
                s_ids[eidx] = be;
                if (d_hits) atomicAdd(&d_hits[be], 1u);
                s_wlocal[t][k] = bp;
                wsum += bp;
            }
            if ((be & 31) == lane) {
                const int b = be >> 5;
                if (b == 0) s[0] = -1e30f;
                else if (b == 1) s[1] = -1e30f;
                else if (b == 2) s[2] = -1e30f;
                else s[3] = -1e30f;
            }
        }
        if (lane == 0) {
            const float denom = fmaxf(wsum, 6.103515625e-5f);
            s_fac[t] = scale / denom;
        }
        __syncwarp();
        if (lane < top_k) {
            ewts[(size_t) t * top_k + lane] = s_wlocal[t][lane] * s_fac[t];
        }
    }
    __syncthreads();
    const int n_ent = T * top_k;
    for (int e = tid; e < n_ent; e += blockDim.x) {
        const int x = s_ids[e];
        if (slot_id[x] >= 0) atomicAdd(&cnt[x], 1);
    }
    __syncthreads();
    // 128-thread parallel warp prefix scan (4 warps x 32 lanes): eliminates the 128-iter serial tid==0 bottleneck!
    __shared__ int32_t s_warp_ng[4];
    __shared__ int32_t s_warp_off[4];
    const int c_val = (tid < n_exp) ? cnt[tid] : 0;
    const int has_val = (c_val > 0) ? 1 : 0;
    int ng_inc = has_val;
    int off_inc = c_val;
    if (tid < 128) {
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const int n_up = __shfl_up_sync(0xffffffffu, ng_inc, d);
            const int o_up = __shfl_up_sync(0xffffffffu, off_inc, d);
            if (lane >= d) { ng_inc += n_up; off_inc += o_up; }
        }
        if (lane == 31) {
            s_warp_ng[warp] = ng_inc;
            s_warp_off[warp] = off_inc;
        }
    }
    __syncthreads();
    if (tid == 0) {
        int ng_acc = 0, off_acc = 0;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const int wn = s_warp_ng[w], wo = s_warp_off[w];
            s_warp_ng[w] = ng_acc;
            s_warp_off[w] = off_acc;
            ng_acc += wn;
            off_acc += wo;
        }
        grp_start[ng_acc] = off_acc;
        *n_groups = ng_acc;
    }
    __syncthreads();
    if (tid < n_exp && has_val) {
        const int ng_idx = s_warp_ng[warp] + ng_inc - 1;
        const int off_idx = s_warp_off[warp] + off_inc - c_val;
        grp_ptr[ng_idx] = (unsigned long long) (blobs + (size_t) slot_id[tid] * stride);
        grp_start[ng_idx] = off_idx;
        cur_off[tid] = off_idx;
    }
    __syncthreads();
    for (int e = tid; e < n_ent; e += blockDim.x) {
        const int x = s_ids[e];
        if (slot_id[x] >= 0) {
            const int pos = atomicAdd(&cur_off[x], 1);
            ent_dst[pos] = e;
            ent_tok[pos] = e / top_k;
        }
    }
}

/// Fused k_combine_experts + k_shortconv (post-FFN, full-VRAM small-T): eliminates moe_raw HBM roundtrip & 1 launch/layer!
__global__ void k_combine_experts_shortconv(const float* __restrict__ eparts, const float* __restrict__ ewts,
                                            const float* __restrict__ w_conv, float* __restrict__ hist,
                                            const float* __restrict__ res_add, float* __restrict__ out,
                                            int N, int top_k, int hist_offset, int T,
                                            float* __restrict__ hist_trace = nullptr, size_t hist_trace_stride = 0) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= N) return;
    float* hc = hist + (size_t) (hist_offset + c) * 3;
    float h0 = hc[0], h1 = hc[1], h2 = hc[2];
    const float4 w4 = *reinterpret_cast<const float4*>(w_conv + (size_t) c * 4);
    const float scale = rsqrtf(5.0f);
    for (int t = 0; t < T; ++t) {
        const size_t base = (size_t) t * top_k;
        float x = 0.0f;
        for (int k = 0; k < top_k; ++k) x += ewts[base + k] * eparts[(base + k) * N + c];
        const float s = h0 * w4.x + h1 * w4.y + h2 * w4.z + x * w4.w;
        const size_t idx = (size_t) t * N + c;
        float y = (x + s) * scale;
        if (res_add) y += res_add[idx];
        out[idx] = y;
        h0 = h1; h1 = h2; h2 = x;
        if (hist_trace) {
            float* ht = hist_trace + (size_t) t * hist_trace_stride + (size_t) (hist_offset + c) * 3;
            ht[0] = h0; ht[1] = h1; ht[2] = h2;
        }
    }
    hc[0] = h0; hc[1] = h1; hc[2] = h2;
}

/// Fused prefill k_combine_experts + k_shortconv_pp + k_shortconv_hist_push (T >= 32, (N&3)==0, !save_conv_trace):
/// Eliminates 100% of moe_raw HBM write+read (37.7 MB/layer = 29 GB on 32K PP!) and 2 kernel launches/layer!
/// Grid: dim3((N/4 + 63)/64, (T + 63)/64), Block: 64. Exact FP32 sum order -> 100% bit-exact!
__global__ void k_combine_experts_shortconv_pp(const float* __restrict__ eparts, const float* __restrict__ ewts,
                                               const float* __restrict__ w_conv, float* __restrict__ hist,
                                               const float* __restrict__ res_add, float* __restrict__ out,
                                               int N, int top_k, int hist_offset, int T) {
    const int n4 = N >> 2;
    const int i4 = (int) blockIdx.x * blockDim.x + threadIdx.x;
    if (i4 >= n4) return;
    const int c0 = i4 << 2;
    const int t0 = (int) blockIdx.y * 128;
    const int t1 = min(T, t0 + 128);

    auto combine_tok = [&](int t) -> float4 {
        const size_t base = (size_t) t * top_k;
        float sx = 0.0f, sy = 0.0f, sz = 0.0f, sw = 0.0f;
#pragma unroll
        for (int k = 0; k < top_k; ++k) {
            const float w = __ldg(&ewts[base + k]);
            const float4 v = reinterpret_cast<const float4*>(eparts + (base + k) * N)[i4];
            sx = fmaf(w, v.x, sx); sy = fmaf(w, v.y, sy); sz = fmaf(w, v.z, sz); sw = fmaf(w, v.w, sw);
        }
        return make_float4(sx, sy, sz, sw);
    };

    float4 h0, h1, h2;
    if (t0 == 0) {
        const float* hc0 = hist + (size_t) (hist_offset + c0 + 0) * 3;
        const float* hc1 = hist + (size_t) (hist_offset + c0 + 1) * 3;
        const float* hc2 = hist + (size_t) (hist_offset + c0 + 2) * 3;
        const float* hc3 = hist + (size_t) (hist_offset + c0 + 3) * 3;
        h0 = make_float4(hc0[0], hc1[0], hc2[0], hc3[0]);
        h1 = make_float4(hc0[1], hc1[1], hc2[1], hc3[1]);
        h2 = make_float4(hc0[2], hc1[2], hc2[2], hc3[2]);
    } else {
        h0 = combine_tok(t0 - 3);
        h1 = combine_tok(t0 - 2);
        h2 = combine_tok(t0 - 1);
    }
    const float4 wc0 = *reinterpret_cast<const float4*>(w_conv + (size_t) (c0 + 0) * 4);
    const float4 wc1 = *reinterpret_cast<const float4*>(w_conv + (size_t) (c0 + 1) * 4);
    const float4 wc2 = *reinterpret_cast<const float4*>(w_conv + (size_t) (c0 + 2) * 4);
    const float4 wc3 = *reinterpret_cast<const float4*>(w_conv + (size_t) (c0 + 3) * 4);
    const float scale = rsqrtf(5.0f);
    for (int t = t0; t < t1; ++t) {
        const float4 x = combine_tok(t);
        float4 y;
        y.x = (x.x + (h0.x * wc0.x + h1.x * wc0.y + h2.x * wc0.z + x.x * wc0.w)) * scale;
        y.y = (x.y + (h0.y * wc1.x + h1.y * wc1.y + h2.y * wc1.z + x.y * wc1.w)) * scale;
        y.z = (x.z + (h0.z * wc2.x + h1.z * wc2.y + h2.z * wc2.z + x.z * wc2.w)) * scale;
        y.w = (x.w + (h0.w * wc3.x + h1.w * wc3.y + h2.w * wc3.z + x.w * wc3.w)) * scale;
        if (res_add) {
            const float4 r = reinterpret_cast<const float4*>(res_add + (size_t) t * N)[i4];
            y.x += r.x; y.y += r.y; y.z += r.z; y.w += r.w;
        }
        reinterpret_cast<float4*>(out + (size_t) t * N)[i4] = y;
        h0 = h1; h1 = h2; h2 = x;
    }
}

/// Companion 1.2 us tail history push for k_combine_experts_shortconv_pp (eliminates inter-CTA hist race on t0==0 vs t1==T!).
__global__ void k_combine_experts_hist_push(const float* __restrict__ eparts, const float* __restrict__ ewts,
                                            float* __restrict__ hist, int N, int top_k, int hist_offset, int T) {
    const int n4 = N >> 2;
    const int i4 = (int) blockIdx.x * blockDim.x + threadIdx.x;
    if (i4 >= n4 || T < 3) return;
    const int c0 = i4 << 2;
    auto combine_tok = [&](int t) -> float4 {
        const size_t base = (size_t) t * top_k;
        float sx = 0.0f, sy = 0.0f, sz = 0.0f, sw = 0.0f;
#pragma unroll
        for (int k = 0; k < top_k; ++k) {
            const float w = __ldg(&ewts[base + k]);
            const float4 v = reinterpret_cast<const float4*>(eparts + (base + k) * N)[i4];
            sx = fmaf(w, v.x, sx); sy = fmaf(w, v.y, sy); sz = fmaf(w, v.z, sz); sw = fmaf(w, v.w, sw);
        }
        return make_float4(sx, sy, sz, sw);
    };
    const float4 h0 = combine_tok(T - 3);
    const float4 h1 = combine_tok(T - 2);
    const float4 h2 = combine_tok(T - 1);
    float* hc0 = hist + (size_t) (hist_offset + c0 + 0) * 3;
    float* hc1 = hist + (size_t) (hist_offset + c0 + 1) * 3;
    float* hc2 = hist + (size_t) (hist_offset + c0 + 2) * 3;
    float* hc3 = hist + (size_t) (hist_offset + c0 + 3) * 3;
    hc0[0] = h0.x; hc0[1] = h1.x; hc0[2] = h2.x;
    hc1[0] = h0.y; hc1[1] = h1.y; hc1[2] = h2.y;
    hc2[0] = h0.z; hc2[1] = h1.z; hc2[2] = h2.z;
    hc3[0] = h0.w; hc3[1] = h1.w; hc3[2] = h2.w;
}

__global__ void k_combine_experts(const float* __restrict__ eparts, const float* __restrict__ ewts,
                                  float* __restrict__ out, int N, int top_k) {
    const int t = blockIdx.y;
    const size_t base = (size_t) t * top_k;
    if ((N & 3) == 0) {
        const int n4 = N >> 2;
        const int i4 = blockIdx.x * blockDim.x + threadIdx.x;
        if (i4 >= n4) return;
        float sx = 0.0f, sy = 0.0f, sz = 0.0f, sw = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            const float w = ewts[base + k];
            const float4 v = reinterpret_cast<const float4*>(eparts + (base + k) * N)[i4];
            sx += w * v.x; sy += w * v.y; sz += w * v.z; sw += w * v.w;
        }
        reinterpret_cast<float4*>(out + (size_t) t * N)[i4] = make_float4(sx, sy, sz, sw);
        return;
    }
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = 0.0f;
    for (int k = 0; k < top_k; ++k) s += ewts[base + k] * eparts[(base + k) * N + i];
    out[(size_t) t * N + i] = s;
}

/// Fused k_combine_experts_hybrid + k_shortconv (post-FFN, Hybrid small-T): eliminates moe_raw HBM roundtrip & 1 launch/layer!
__global__ void k_combine_experts_hybrid_shortconv(const float* __restrict__ eparts, const float* __restrict__ ewts,
                                                   const int32_t* __restrict__ ids, const int32_t* __restrict__ slot_of,
                                                   const float* __restrict__ cpu_sum,
                                                   const float* __restrict__ w_conv, float* __restrict__ hist,
                                                   const float* __restrict__ res_add, float* __restrict__ out,
                                                   int N, int top_k, int hist_offset, int T,
                                                   float* __restrict__ hist_trace = nullptr, size_t hist_trace_stride = 0) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= N) return;
    float* hc = hist + (size_t) (hist_offset + c) * 3;
    float h0 = hc[0], h1 = hc[1], h2 = hc[2];
    const float4 w4 = *reinterpret_cast<const float4*>(w_conv + (size_t) c * 4);
    const float scale = rsqrtf(5.0f);
    for (int t = 0; t < T; ++t) {
        const size_t base = (size_t) t * top_k;
        float x = cpu_sum[(size_t) t * N + c];
        for (int k = 0; k < top_k; ++k) {
            if (slot_of[ids[base + k]] >= 0) x += ewts[base + k] * eparts[(base + k) * N + c];
        }
        const float s = h0 * w4.x + h1 * w4.y + h2 * w4.z + x * w4.w;
        const size_t idx = (size_t) t * N + c;
        float y = (x + s) * scale;
        if (res_add) y += res_add[idx];
        out[idx] = y;
        h0 = h1; h1 = h2; h2 = x;
        if (hist_trace) {
            float* ht = hist_trace + (size_t) t * hist_trace_stride + (size_t) (hist_offset + c) * 3;
            ht[0] = h0; ht[1] = h1; ht[2] = h2;
        }
    }
    hc[0] = h0; hc[1] = h1; hc[2] = h2;
}

__global__ void k_combine_experts_hybrid(const float* __restrict__ eparts, const float* __restrict__ ewts,
                                         const int32_t* __restrict__ ids, const int32_t* __restrict__ slot_of,
                                         const float* __restrict__ cpu_sum, float* __restrict__ out,
                                         int N, int top_k) {
    const int t = (int) blockIdx.y;
    const size_t base = (size_t) t * top_k;
    const float* cs = cpu_sum + (size_t) t * N;
    float* dst = out + (size_t) t * N;
    if ((N & 3) == 0) {
        const int n4 = N >> 2;
        const int i4 = blockIdx.x * blockDim.x + threadIdx.x;
        if (i4 >= n4) return;
        const float4 c = reinterpret_cast<const float4*>(cs)[i4];
        float sx = c.x, sy = c.y, sz = c.z, sw = c.w;
        for (int k = 0; k < top_k; ++k) {
            if (slot_of[ids[base + k]] >= 0) {
                const float w = ewts[base + k];
                const float4 v = reinterpret_cast<const float4*>(eparts + (base + k) * N)[i4];
                sx += w * v.x; sy += w * v.y; sz += w * v.z; sw += w * v.w;
            }
        }
        reinterpret_cast<float4*>(dst)[i4] = make_float4(sx, sy, sz, sw);
        return;
    }
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = cs[i];
    for (int k = 0; k < top_k; ++k) {
        if (slot_of[ids[base + k]] >= 0) s += ewts[base + k] * eparts[(base + k) * N + i];
    }
    dst[i] = s;
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
                                int32_t* __restrict__ ent_tok, int top_k,
                                uint32_t* __restrict__ d_hits = nullptr) {
    // 1 block of 512 threads + 4-warp parallel prefix scan (eliminates 128-iter serial tid==0 bottleneck!).
    __shared__ int32_t cnt[128];
    __shared__ int32_t slot_id[128];
    __shared__ int32_t cur_off[128];
    __shared__ int32_t s_warp_ng[4];
    __shared__ int32_t s_warp_off[4];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    if (tid < n_exp) {
        cnt[tid] = 0;
        slot_id[tid] = n_slots > 0 ? slot_of[tid] : -1;
    }
    __syncthreads();
    for (int e = tid; e < n_ent; e += blockDim.x) {
        const int x = ids[e];
        if (slot_id[x] >= 0) atomicAdd(&cnt[x], 1);
    }
    __syncthreads();
    if (d_hits && tid < n_exp && cnt[tid] > 0) {
        d_hits[tid] += (uint32_t) cnt[tid];
    }
    const int c_val = (tid < n_exp) ? cnt[tid] : 0;
    const int has_val = (c_val > 0) ? 1 : 0;
    int ng_inc = has_val, off_inc = c_val;
    if (tid < 128) {
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const int n_up = __shfl_up_sync(0xffffffffu, ng_inc, d);
            const int o_up = __shfl_up_sync(0xffffffffu, off_inc, d);
            if (lane >= d) { ng_inc += n_up; off_inc += o_up; }
        }
        if (lane == 31) { s_warp_ng[warp] = ng_inc; s_warp_off[warp] = off_inc; }
    }
    __syncthreads();
    if (tid == 0) {
        int ng_acc = 0, off_acc = 0;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const int wn = s_warp_ng[w], wo = s_warp_off[w];
            s_warp_ng[w] = ng_acc; s_warp_off[w] = off_acc;
            ng_acc += wn; off_acc += wo;
        }
        grp_start[ng_acc] = off_acc;
        *n_groups = ng_acc;
    }
    __syncthreads();
    if (tid < n_exp && has_val) {
        const int ng_idx = s_warp_ng[warp] + ng_inc - 1;
        const int off_idx = s_warp_off[warp] + off_inc - c_val;
        grp_ptr[ng_idx] = (unsigned long long) (blobs + (size_t) slot_id[tid] * stride);
        grp_start[ng_idx] = off_idx;
        cur_off[tid] = off_idx;
    }
    __syncthreads();
    for (int e = tid; e < n_ent; e += blockDim.x) {
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
/// 512 threads + 4-warp parallel prefix scan (eliminates 128-iter serial scan on RTX 4070 PP!).
__global__ void k_group_experts_miss_map(const int32_t* __restrict__ ids,
                                         const uint8_t* __restrict__ stage_blobs, size_t stride,
                                         int n_ent, int n_exp, PpStageMap map,
                                         unsigned long long* __restrict__ grp_ptr, int32_t* __restrict__ grp_start,
                                         int32_t* __restrict__ n_groups, int32_t* __restrict__ ent_dst,
                                         int32_t* __restrict__ ent_tok, int top_k) {
    __shared__ int32_t cnt[128];
    __shared__ int16_t spos[128];
    __shared__ int32_t cur_off[128];
    __shared__ int32_t s_warp_ng[4];
    __shared__ int32_t s_warp_off[4];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    if (tid < n_exp) {
        cnt[tid] = 0;
        spos[tid] = map.pos[tid];
    }
    __syncthreads();
    for (int e = tid; e < n_ent; e += blockDim.x) {
        const int x = ids[e];
        if (spos[x] >= 0) atomicAdd(&cnt[x], 1);
    }
    __syncthreads();
    const int c_val = (tid < n_exp) ? cnt[tid] : 0;
    const int has_val = (c_val > 0) ? 1 : 0;
    int ng_inc = has_val, off_inc = c_val;
    if (tid < 128) {
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const int n_up = __shfl_up_sync(0xffffffffu, ng_inc, d);
            const int o_up = __shfl_up_sync(0xffffffffu, off_inc, d);
            if (lane >= d) { ng_inc += n_up; off_inc += o_up; }
        }
        if (lane == 31) { s_warp_ng[warp] = ng_inc; s_warp_off[warp] = off_inc; }
    }
    __syncthreads();
    if (tid == 0) {
        int ng_acc = 0, off_acc = 0;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const int wn = s_warp_ng[w], wo = s_warp_off[w];
            s_warp_ng[w] = ng_acc; s_warp_off[w] = off_acc;
            ng_acc += wn; off_acc += wo;
        }
        grp_start[ng_acc] = off_acc;
        *n_groups = ng_acc;
    }
    __syncthreads();
    if (tid < n_exp && has_val) {
        const int ng_idx = s_warp_ng[warp] + ng_inc - 1;
        const int off_idx = s_warp_off[warp] + off_inc - c_val;
        grp_ptr[ng_idx] = (unsigned long long) (stage_blobs + (size_t) spos[tid] * stride);
        grp_start[ng_idx] = off_idx;
        cur_off[tid] = off_idx;
    }
    __syncthreads();
    for (int e = tid; e < n_ent; e += blockDim.x) {
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

/// Multi-block argmax stage 1: grid(n_blk, rows), block 256. Saturates all SMs on 200K+ vocab.
__global__ void k_argmax_stage1(const float* __restrict__ logits, float* __restrict__ p_val,
                                int32_t* __restrict__ p_idx, int V) {
    __shared__ float sv[8];
    __shared__ int32_t si[8];
    const int b = blockIdx.x, row_id = blockIdx.y, n_blk = gridDim.x;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const float* row = logits + (size_t) row_id * V;
    float mv = -1e30f;
    int32_t mi = 0;
    const int stride = n_blk * 256;
    if ((((uintptr_t) row) & 15u) == 0u) {
        const int v4 = V >> 2;
        const float4* row4 = reinterpret_cast<const float4*>(row);
        for (int i4 = b * 256 + tid; i4 < v4; i4 += stride) {
            const float4 v = row4[i4];
            const int base = i4 << 2;
            if (v.x > mv) { mv = v.x; mi = base; }
            if (v.y > mv) { mv = v.y; mi = base + 1; }
            if (v.z > mv) { mv = v.z; mi = base + 2; }
            if (v.w > mv) { mv = v.w; mi = base + 3; }
        }
        if (b == 0) {
            const int tail_start = v4 << 2;
            for (int i = tail_start + tid; i < V; i += 256) {
                const float v = row[i];
                if (v > mv) { mv = v; mi = i; }
            }
        }
    } else {
        for (int i = b * 256 + tid; i < V; i += stride) {
            const float v = row[i];
            if (v > mv) { mv = v; mi = i; }
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, mv, o);
        const int32_t oi = __shfl_xor_sync(0xffffffffu, mi, o);
        if (ov > mv || (ov == mv && oi < mi)) { mv = ov; mi = oi; }
    }
    if (lane == 0) { sv[warp] = mv; si[warp] = mi; }
    __syncthreads();
    if (warp == 0) {
        mv = (lane < 8) ? sv[lane] : -1e30f;
        mi = (lane < 8) ? si[lane] : 0;
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, mv, o);
            const int32_t oi = __shfl_xor_sync(0xffffffffu, mi, o);
            if (ov > mv || (ov == mv && oi < mi)) { mv = ov; mi = oi; }
        }
        if (lane == 0) {
            p_val[row_id * n_blk + b] = mv;
            p_idx[row_id * n_blk + b] = mi;
        }
    }
}

/// Multi-block argmax stage 2: grid(rows), block 32 (pure warp reduction, zero SMEM).
__global__ void k_argmax_stage2(const float* __restrict__ p_val, const int32_t* __restrict__ p_idx,
                                int32_t* __restrict__ pick, int n_blk) {
    const int row_id = blockIdx.x, lane = threadIdx.x;
    const float* pv = p_val + row_id * n_blk;
    const int32_t* pi = p_idx + row_id * n_blk;
    float mv = -1e30f;
    int32_t mi = 0;
    for (int i = lane; i < n_blk; i += 32) {
        const float v = pv[i];
        const int32_t idx = pi[i];
        if (v > mv || (v == mv && idx < mi)) { mv = v; mi = idx; }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, mv, o);
        const int32_t oi = __shfl_xor_sync(0xffffffffu, mi, o);
        if (ov > mv || (ov == mv && oi < mi)) { mv = ov; mi = oi; }
    }
    if (lane == 0) pick[row_id] = mi;
}

/// Apply repetition / frequency / presence penalties in-place on GPU logits for `rows` rows (<= 64 unique tokens).
/// Grid (1, rows), Block 64. 100% bit-exact to CPU `build_sampling_dist`.
__global__ void k_apply_penalties(float* __restrict__ logits, int V,
                                  const int32_t* __restrict__ pen_toks, const int32_t* __restrict__ pen_cnts,
                                  int n_pen, float rep_pen, float freq_pen, float pres_pen) {
    const int j = (int) blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_pen) return;
    const int32_t tok = pen_toks[j];
    if (tok < 0 || tok >= V) return;
    float* row = logits + (size_t) blockIdx.y * V;
    float l = row[tok];
    if (rep_pen != 1.0f) l = (l <= 0.0f) ? (l * rep_pen) : (l / rep_pen);
    l -= (float) pen_cnts[j] * freq_pen + pres_pen;
    row[tok] = l;
}

__device__ __forceinline__ uint32_t f32_to_ordered_u32(float v) {
    const uint32_t u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

/// Exact GPU Top-K (K <= 256) Stage 1: build 11-bit MSB histogram (2048 buckets) per row.
/// Grid (n_blk, rows), Block 256 (8 KB SMEM, STACK: 0).
__global__ void k_topk_hist(const float* __restrict__ logits, uint32_t* __restrict__ hist_out, int V) {
    __shared__ uint32_t sh[2048];
    const int tid = threadIdx.x;
    const int b = blockIdx.x, row_id = blockIdx.y, n_blk = gridDim.x;
#pragma unroll
    for (int i = tid; i < 2048; i += 256) sh[i] = 0u;
    __syncthreads();
    const float* row = logits + (size_t) row_id * V;
    const int stride = n_blk * 256;
    for (int i = b * 256 + tid; i < V; i += stride) {
        const uint32_t bucket = f32_to_ordered_u32(row[i]) >> 21;
        atomicAdd(&sh[bucket], 1u);
    }
    __syncthreads();
    uint32_t* dst = hist_out + (size_t) row_id * 2048;
#pragma unroll
    for (int i = tid; i < 2048; i += 256) {
        const uint32_t c = sh[i];
        if (c > 0u) atomicAdd(&dst[i], c);
    }
}

/// Exact GPU Top-K Stage 2: find highest 11-bit bucket `thr_bucket` whose cumulative count >= K.
/// Grid (rows), Block 256 (8 KB SMEM, STACK: 0). Also resets `cand_cnt[row] = 0`.
__global__ void k_topk_select_thresh(const uint32_t* __restrict__ hist, int32_t* __restrict__ thr_bucket,
                                     int32_t* __restrict__ cand_cnt, int K) {
    __shared__ uint32_t sh[2048];
    __shared__ uint32_t s_warp[8];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int row_id = blockIdx.x;
    const uint32_t* h = hist + (size_t) row_id * 2048;
    // Reverse order: idx 0 = bucket 2047 (largest floats), idx 2047 = bucket 0
    for (int i = tid; i < 2048; i += 256) sh[i] = h[2047 - i];
    __syncthreads();
    // 256 threads each sum 8 consecutive buckets in descending order
    uint32_t local_sum = 0u;
    const int base = tid * 8;
#pragma unroll
    for (int j = 0; j < 8; ++j) local_sum += sh[base + j];
    // Inclusive warp scan on local_sum
    uint32_t v = local_sum;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const uint32_t n = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += n;
    }
    if (lane == 31) s_warp[warp] = v;
    __syncthreads();
    if (warp == 0) {
        uint32_t wv = (lane < 8) ? s_warp[lane] : 0u;
#pragma unroll
        for (int o = 1; o < 8; o <<= 1) {
            const uint32_t n = __shfl_up_sync(0xffffffffu, wv, o);
            if (lane >= o) wv += n;
        }
        if (lane < 8) s_warp[lane] = wv;
    }
    __syncthreads();
    const uint32_t warp_off = (warp > 0) ? s_warp[warp - 1] : 0u;
    const uint32_t end_sum = warp_off + v;
    const uint32_t start_sum = end_sum - local_sum;
    if (start_sum < (uint32_t) K && end_sum >= (uint32_t) K) {
        uint32_t acc = start_sum;
        for (int j = 0; j < 8; ++j) {
            acc += sh[base + j];
            if (acc >= (uint32_t) K) {
                thr_bucket[row_id] = 2047 - (base + j);
                break;
            }
        }
    }
    if (tid == 0) cand_cnt[row_id] = 0;
}

/// Exact GPU Top-K Stage 3: collect all elements with `(key >> 21) >= thr_bucket` up to `kTopkMaxCand = 2048`.
/// Grid (n_blk, rows), Block 256. Sets `cand_cnt[row] > 2048` on overflow (triggers exact host fallback).
__global__ void k_topk_collect(const float* __restrict__ logits, const int32_t* __restrict__ thr_bucket,
                               float* __restrict__ cand_val, int32_t* __restrict__ cand_idx,
                               int32_t* __restrict__ cand_cnt, int V) {
    constexpr int kMaxCand = 2048;
    const int b = blockIdx.x, row_id = blockIdx.y, n_blk = gridDim.x;
    const int tid = threadIdx.x;
    const uint32_t thr = (uint32_t) thr_bucket[row_id];
    const float* row = logits + (size_t) row_id * V;
    float* cv = cand_val + (size_t) row_id * kMaxCand;
    int32_t* ci = cand_idx + (size_t) row_id * kMaxCand;
    const int stride = n_blk * 256;
    for (int i = b * 256 + tid; i < V; i += stride) {
        const float v = row[i];
        if ((f32_to_ordered_u32(v) >> 21) >= thr) {
            const int pos = atomicAdd(&cand_cnt[row_id], 1);
            if (pos < kMaxCand) {
                cv[pos] = v;
                ci[pos] = i;
            }
        }
    }
}

/// Exact GPU Top-K Stage 4: deterministic bitonic sort of `n_cand <= 2048` in SMEM -> exact top `K` pairs.
/// Grid (rows), Block 256 (16 KB SMEM, STACK: 0). Writes `out_cnt[row] = K` on success or `-1` on overflow.
__global__ void k_topk_finalize(const float* __restrict__ cand_val, const int32_t* __restrict__ cand_idx,
                                const int32_t* __restrict__ cand_cnt,
                                float* __restrict__ out_val, int32_t* __restrict__ out_idx,
                                int32_t* __restrict__ out_cnt, int K) {
    constexpr int kMaxCand = 2048;
    __shared__ float sv[kMaxCand];
    __shared__ int32_t si[kMaxCand];
    const int row_id = blockIdx.x, tid = threadIdx.x;
    const int n = cand_cnt[row_id];
    if (n <= 0 || n > kMaxCand) {
        if (tid == 0) out_cnt[row_id] = -1;
        return;
    }
    int pow2 = 256;
    while (pow2 < n) pow2 <<= 1;
    const float* cv = cand_val + (size_t) row_id * kMaxCand;
    const int32_t* ci = cand_idx + (size_t) row_id * kMaxCand;
    for (int i = tid; i < pow2; i += 256) {
        if (i < n) { sv[i] = cv[i]; si[i] = ci[i]; }
        else { sv[i] = -3.402823466e+38f; si[i] = 0x7fffffff; }
    }
    __syncthreads();
    // Bitonic sort descending by (val > other_val || (val == other_val && idx < other_idx))
    for (int size = 2; size <= pow2; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int pair = tid; pair < (pow2 >> 1); pair += 256) {
                const int blk = pair / stride;
                const int off = pair - blk * stride;
                const int i = blk * (stride << 1) + off;
                const int j = i + stride;
                const bool asc = ((i & size) != 0);
                const float vi = sv[i], vj = sv[j];
                const int32_t ii = si[i], ij = si[j];
                const bool i_better = (vi > vj) || (vi == vj && ii < ij);
                // If !asc (descending block): want i_better; swap if !i_better.
                // If asc (ascending block): want !i_better; swap if i_better.
                if (asc ? i_better : !i_better) {
                    sv[i] = vj; si[i] = ij;
                    sv[j] = vi; si[j] = ii;
                }
            }
            __syncthreads();
        }
    }
    float* ov = out_val + (size_t) row_id * K;
    int32_t* oi = out_idx + (size_t) row_id * K;
    for (int i = tid; i < K; i += 256) {
        ov[i] = sv[i];
        oi[i] = si[i];
    }
    if (tid == 0) out_cnt[row_id] = K;
}

struct SwaLayerPtrs {
    int n_swa_layers = 0;
    int8_t* kq[48] = {};
    __half* ks[48] = {};
    int8_t* vq[48] = {};
    __half* vs[48] = {};
    __half* kc[48] = {};
    __half* vc[48] = {};
};

/// Save (`restore == 0`) or restore (`restore == 1`) `n_slots` SWA KV ring entries (`pos_base + si`) across all SWA layers.
/// Grid: (n_slots, n_swa_layers), Block: 256. Each slot backup holds `[kq(kv_dim), vq(kv_dim), ks(nhkv), vs(nhkv)]` or `[kc, vc]`.
/// Device-pos0 variant of k_swa_q8_bak for CUDA Graph capture (reads pos_base = pos0_d[0] + pos_add).
__global__ void k_swa_q8_bak_d(SwaLayerPtrs swa, uint8_t* __restrict__ bak, int kv_dim, int n_head_kv, int cap,
                               const int32_t* __restrict__ pos0_d, int pos_add, int first_si, int n_slots, int restore) {
    const int si = first_si + blockIdx.x;
    const int s = blockIdx.y;
    if (s >= swa.n_swa_layers) return;
    const int p = pos0_d[0] + pos_add + si;
    if (restore && p < cap) return;
    const int slot = p % cap;
    const size_t slot_bytes = (size_t) kv_dim * 2 + (size_t) n_head_kv * 2 * sizeof(__half);
    uint8_t* b = bak + ((size_t) si * swa.n_swa_layers + s) * slot_bytes;
    int8_t* bkq = (int8_t*) b;
    int8_t* bvq = bkq + kv_dim;
    __half* bks = (__half*) (bvq + kv_dim);
    __half* bvs = bks + n_head_kv;
    int8_t* dkq = swa.kq[s] + (size_t) slot * kv_dim;
    int8_t* dvq = swa.vq[s] + (size_t) slot * kv_dim;
    __half* dks = swa.ks[s] + (size_t) slot * n_head_kv;
    __half* dvs = swa.vs[s] + (size_t) slot * n_head_kv;
    for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
        if (restore) { dkq[i] = bkq[i]; dvq[i] = bvq[i]; }
        else { bkq[i] = dkq[i]; bvq[i] = dvq[i]; }
    }
    if (threadIdx.x < n_head_kv) {
        if (restore) { dks[threadIdx.x] = bks[threadIdx.x]; dvs[threadIdx.x] = bvs[threadIdx.x]; }
        else { bks[threadIdx.x] = dks[threadIdx.x]; bvs[threadIdx.x] = dvs[threadIdx.x]; }
    }
}

__global__ void k_swa_f16_bak_d(SwaLayerPtrs swa, uint8_t* __restrict__ bak, int kv_dim, int cap,
                                const int32_t* __restrict__ pos0_d, int pos_add, int first_si, int n_slots, int restore) {
    const int si = first_si + blockIdx.x;
    const int s = blockIdx.y;
    if (s >= swa.n_swa_layers) return;
    const int p = pos0_d[0] + pos_add + si;
    if (restore && p < cap) return;
    const int slot = p % cap;
    const size_t slot_bytes = (size_t) kv_dim * 2 * sizeof(__half);
    __half* b = (__half*) (bak + ((size_t) si * swa.n_swa_layers + s) * slot_bytes);
    __half* dkc = swa.kc[s] + (size_t) slot * kv_dim;
    __half* dvc = swa.vc[s] + (size_t) slot * kv_dim;
    for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
        if (restore) { dkc[i] = b[i]; dvc[i] = b[kv_dim + i]; }
        else { b[i] = dkc[i]; b[kv_dim + i] = dvc[i]; }
    }
}

__global__ void k_swa_q8_bak(SwaLayerPtrs swa, uint8_t* __restrict__ bak, int kv_dim, int n_head_kv, int cap,
                             int pos_base, int first_si, int n_slots, int restore) {
    const int si = first_si + blockIdx.x;
    const int s = blockIdx.y;
    if (s >= swa.n_swa_layers) return;
    const int p = pos_base + si;
    if (restore && p < cap) return;   // slot was empty prior to overwrite
    const int slot = p % cap;
    const size_t slot_bytes = (size_t) kv_dim * 2 + (size_t) n_head_kv * 2 * sizeof(__half);
    uint8_t* b = bak + ((size_t) si * swa.n_swa_layers + s) * slot_bytes;
    int8_t* bkq = (int8_t*) b;
    int8_t* bvq = bkq + kv_dim;
    __half* bks = (__half*) (bvq + kv_dim);
    __half* bvs = bks + n_head_kv;
    int8_t* dkq = swa.kq[s] + (size_t) slot * kv_dim;
    int8_t* dvq = swa.vq[s] + (size_t) slot * kv_dim;
    __half* dks = swa.ks[s] + (size_t) slot * n_head_kv;
    __half* dvs = swa.vs[s] + (size_t) slot * n_head_kv;
    for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
        if (restore) { dkq[i] = bkq[i]; dvq[i] = bvq[i]; }
        else { bkq[i] = dkq[i]; bvq[i] = dvq[i]; }
    }
    if (threadIdx.x < n_head_kv) {
        if (restore) { dks[threadIdx.x] = bks[threadIdx.x]; dvs[threadIdx.x] = bvs[threadIdx.x]; }
        else { bks[threadIdx.x] = dks[threadIdx.x]; bvs[threadIdx.x] = dvs[threadIdx.x]; }
    }
}

__global__ void k_swa_f16_bak(SwaLayerPtrs swa, uint8_t* __restrict__ bak, int kv_dim, int cap,
                              int pos_base, int first_si, int n_slots, int restore) {
    const int si = first_si + blockIdx.x;
    const int s = blockIdx.y;
    if (s >= swa.n_swa_layers) return;
    const int p = pos_base + si;
    if (restore && p < cap) return;
    const int slot = p % cap;
    const size_t slot_bytes = (size_t) kv_dim * 2 * sizeof(__half);
    __half* b = (__half*) (bak + ((size_t) si * swa.n_swa_layers + s) * slot_bytes);
    __half* bkc = b;
    __half* bvc = b + kv_dim;
    __half* dkc = swa.kc[s] + (size_t) slot * kv_dim;
    __half* dvc = swa.vc[s] + (size_t) slot * kv_dim;
    for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
        if (restore) { dkc[i] = bkc[i]; dvc[i] = bvc[i]; }
        else { bkc[i] = dkc[i]; bvc[i] = dvc[i]; }
    }
}

/// EX-4A: Remap live SWA Q8 KV ring entries [pos0 .. pos0 + n_live - 1] from old_cap -> new_cap across all SWA layers.
/// Grid: (n_live, n_swa_layers), Block: 256. 16-byte vectorized int4 copy when kv_dim % 16 == 0.
__global__ void k_swa_ring_remap_q8(SwaLayerPtrs src, SwaLayerPtrs dst, int kv_dim, int n_head_kv,
                                    int old_cap, int new_cap, int64_t pos0, int n_live) {
    const int j = (int) blockIdx.x;
    const int s = (int) blockIdx.y;
    if (j >= n_live || s >= src.n_swa_layers) return;
    const int64_t p = pos0 + j;
    const int src_slot = (int) (p % old_cap);
    const int dst_slot = (int) (p % new_cap);
    if ((kv_dim & 15) == 0) {
        const int n4 = kv_dim >> 4;
        const int4* sk = reinterpret_cast<const int4*>(src.kq[s] + (size_t) src_slot * kv_dim);
        const int4* sv = reinterpret_cast<const int4*>(src.vq[s] + (size_t) src_slot * kv_dim);
        int4* dk = reinterpret_cast<int4*>(dst.kq[s] + (size_t) dst_slot * kv_dim);
        int4* dv = reinterpret_cast<int4*>(dst.vq[s] + (size_t) dst_slot * kv_dim);
        for (int i = threadIdx.x; i < n4; i += blockDim.x) {
            dk[i] = sk[i];
            dv[i] = sv[i];
        }
    } else {
        const int8_t* sk = src.kq[s] + (size_t) src_slot * kv_dim;
        const int8_t* sv = src.vq[s] + (size_t) src_slot * kv_dim;
        int8_t* dk = dst.kq[s] + (size_t) dst_slot * kv_dim;
        int8_t* dv = dst.vq[s] + (size_t) dst_slot * kv_dim;
        for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
            dk[i] = sk[i];
            dv[i] = sv[i];
        }
    }
    if (threadIdx.x < n_head_kv) {
        dst.ks[s][(size_t) dst_slot * n_head_kv + threadIdx.x] = src.ks[s][(size_t) src_slot * n_head_kv + threadIdx.x];
        dst.vs[s][(size_t) dst_slot * n_head_kv + threadIdx.x] = src.vs[s][(size_t) src_slot * n_head_kv + threadIdx.x];
    }
}

/// EX-4A: Remap live SWA FP16 KV ring entries [pos0 .. pos0 + n_live - 1] from old_cap -> new_cap across all SWA layers.
__global__ void k_swa_ring_remap_f16(SwaLayerPtrs src, SwaLayerPtrs dst, int kv_dim,
                                     int old_cap, int new_cap, int64_t pos0, int n_live) {
    const int j = (int) blockIdx.x;
    const int s = (int) blockIdx.y;
    if (j >= n_live || s >= src.n_swa_layers) return;
    const int64_t p = pos0 + j;
    const int src_slot = (int) (p % old_cap);
    const int dst_slot = (int) (p % new_cap);
    if ((kv_dim & 7) == 0) {
        const int n4 = kv_dim >> 3;
        const int4* sk = reinterpret_cast<const int4*>(src.kc[s] + (size_t) src_slot * kv_dim);
        const int4* sv = reinterpret_cast<const int4*>(src.vc[s] + (size_t) src_slot * kv_dim);
        int4* dk = reinterpret_cast<int4*>(dst.kc[s] + (size_t) dst_slot * kv_dim);
        int4* dv = reinterpret_cast<int4*>(dst.vc[s] + (size_t) dst_slot * kv_dim);
        for (int i = threadIdx.x; i < n4; i += blockDim.x) {
            dk[i] = sk[i];
            dv[i] = sv[i];
        }
    } else {
        const __half* sk = src.kc[s] + (size_t) src_slot * kv_dim;
        const __half* sv = src.vc[s] + (size_t) src_slot * kv_dim;
        __half* dk = dst.kc[s] + (size_t) dst_slot * kv_dim;
        __half* dv = dst.vc[s] + (size_t) dst_slot * kv_dim;
        for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
            dk[i] = sk[i];
            dv[i] = sv[i];
        }
    }
}

// ============================================================================
// Wave-5 Ready Kernels (Part B: Zero-Precision-Loss, STACK:0 across sm_70/89)
// ============================================================================

/// 7a. GPU Embedding Q8_0 Dequant + Fused RMSNorm (exact IEEE-754 __half2float(d) * (float)(int8_t)q).
/// Grid: (T), Block: (N == 2304) ? 576 : 256.
__global__ void __launch_bounds__(576, 2) k_dequant_embd_rmsnorm_q8_0(
        const uint8_t* __restrict__ raw_uniq, const int32_t* __restrict__ tok_idx,
        const float* __restrict__ norm_w, float* __restrict__ out_cur,
        int T, int N, int row_bytes, float eps) {
    __shared__ float sh[32];
    const int t = (int) blockIdx.x;
    if (t >= T) return;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int32_t uid = tok_idx[t];
    const uint8_t* row = raw_uniq + (size_t) uid * (size_t) row_bytes;
    float* dst = out_cur + (size_t) t * (size_t) N;

    if (N == 2304 && blockDim.x == 576) {
        const int blk = tid >> 3;        // 0..71 (72 Q8_0 blocks of 32 elems)
        const int sub = (tid & 7) << 2;  // 0, 4, ..., 28
        const uint8_t* bp = row + (size_t) blk * 34u;
        __half dh;
        memcpy(&dh, bp, 2);
        const float d = __half2float(dh);
        const int8_t* qp = reinterpret_cast<const int8_t*>(bp + 2 + sub);
        const float4 v = make_float4((float) qp[0] * d, (float) qp[1] * d, (float) qp[2] * d, (float) qp[3] * d);
        float s = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) sh[warp] = s;
        __syncthreads();
        if (warp == 0) {
            float tot = (lane < 18) ? sh[lane] : 0.0f;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, o);
            if (lane == 0) sh[0] = rsqrtf(tot / 2304.0f + eps);
        }
        __syncthreads();
        const float r = sh[0];
        float4 o;
        if (norm_w) {
            const float4 ww = __ldg(reinterpret_cast<const float4*>(norm_w) + tid);
            o = make_float4(v.x * r * ww.x, v.y * r * ww.y, v.z * r * ww.z, v.w * r * ww.w);
        } else {
            o = make_float4(v.x * r, v.y * r, v.z * r, v.w * r);
        }
        reinterpret_cast<float4*>(dst)[tid] = o;
        return;
    }

    float s = 0.0f;
    for (int c = tid; c < N; c += blockDim.x) {
        const int blk = c >> 5, j = c & 31;
        const uint8_t* bp = row + (size_t) blk * 34u;
        __half dh;
        memcpy(&dh, bp, 2);
        const float v = (float) reinterpret_cast<const int8_t*>(bp + 2)[j] * __half2float(dh);
        dst[c] = v;
        s += v * v;
    }
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) N + eps);
    for (int c = tid; c < N; c += blockDim.x) {
        const float w = norm_w ? norm_w[c] : 1.0f;
        dst[c] = dst[c] * r * w;
    }
}

/// 7b. GPU Embedding BF16 Dequant + Fused RMSNorm (exact IEEE-754).
/// Grid: (T), Block: (N == 2304) ? 576 : 256.
__global__ void __launch_bounds__(576, 2) k_dequant_embd_rmsnorm_bf16(
        const uint8_t* __restrict__ raw_uniq, const int32_t* __restrict__ tok_idx,
        const float* __restrict__ norm_w, float* __restrict__ out_cur,
        int T, int N, int row_bytes, float eps) {
    __shared__ float sh[32];
    const int t = (int) blockIdx.x;
    if (t >= T) return;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int32_t uid = tok_idx[t];
    const uint16_t* row = reinterpret_cast<const uint16_t*>(raw_uniq + (size_t) uid * (size_t) row_bytes);
    float* dst = out_cur + (size_t) t * (size_t) N;

    if (N == 2304 && blockDim.x == 576) {
        const uint2 raw = reinterpret_cast<const uint2*>(row)[tid];
        const float4 v = make_float4(
            __uint_as_float((raw.x & 0xFFFFu) << 16),
            __uint_as_float(raw.x & 0xFFFF0000u),
            __uint_as_float((raw.y & 0xFFFFu) << 16),
            __uint_as_float(raw.y & 0xFFFF0000u));
        float s = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) sh[warp] = s;
        __syncthreads();
        if (warp == 0) {
            float tot = (lane < 18) ? sh[lane] : 0.0f;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, o);
            if (lane == 0) sh[0] = rsqrtf(tot / 2304.0f + eps);
        }
        __syncthreads();
        const float r = sh[0];
        float4 o;
        if (norm_w) {
            const float4 ww = __ldg(reinterpret_cast<const float4*>(norm_w) + tid);
            o = make_float4(v.x * r * ww.x, v.y * r * ww.y, v.z * r * ww.z, v.w * r * ww.w);
        } else {
            o = make_float4(v.x * r, v.y * r, v.z * r, v.w * r);
        }
        reinterpret_cast<float4*>(dst)[tid] = o;
        return;
    }

    float s = 0.0f;
    for (int c = tid; c < N; c += blockDim.x) {
        const float v = __uint_as_float((uint32_t) row[c] << 16);
        dst[c] = v;
        s += v * v;
    }
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) N + eps);
    for (int c = tid; c < N; c += blockDim.x) {
        const float w = norm_w ? norm_w[c] : 1.0f;
        dst[c] = dst[c] * r * w;
    }
}

/// 7c. GPU Embedding FP16 Dequant + Fused RMSNorm (exact IEEE-754).
/// Grid: (T), Block: (N == 2304) ? 576 : 256.
__global__ void __launch_bounds__(576, 2) k_dequant_embd_rmsnorm_f16(
        const uint8_t* __restrict__ raw_uniq, const int32_t* __restrict__ tok_idx,
        const float* __restrict__ norm_w, float* __restrict__ out_cur,
        int T, int N, int row_bytes, float eps) {
    __shared__ float sh[32];
    const int t = (int) blockIdx.x;
    if (t >= T) return;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int32_t uid = tok_idx[t];
    const __half* row = reinterpret_cast<const __half*>(raw_uniq + (size_t) uid * (size_t) row_bytes);
    float* dst = out_cur + (size_t) t * (size_t) N;

    if (N == 2304 && blockDim.x == 576) {
        const uint2 raw = reinterpret_cast<const uint2*>(row)[tid];
        __half2 h0, h1;
        memcpy(&h0, &raw.x, 4);
        memcpy(&h1, &raw.y, 4);
        const float2 f0 = __half22float2(h0), f1 = __half22float2(h1);
        const float4 v = make_float4(f0.x, f0.y, f1.x, f1.y);
        float s = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) sh[warp] = s;
        __syncthreads();
        if (warp == 0) {
            float tot = (lane < 18) ? sh[lane] : 0.0f;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, o);
            if (lane == 0) sh[0] = rsqrtf(tot / 2304.0f + eps);
        }
        __syncthreads();
        const float r = sh[0];
        float4 o;
        if (norm_w) {
            const float4 ww = __ldg(reinterpret_cast<const float4*>(norm_w) + tid);
            o = make_float4(v.x * r * ww.x, v.y * r * ww.y, v.z * r * ww.z, v.w * r * ww.w);
        } else {
            o = make_float4(v.x * r, v.y * r, v.z * r, v.w * r);
        }
        reinterpret_cast<float4*>(dst)[tid] = o;
        return;
    }

    float s = 0.0f;
    for (int c = tid; c < N; c += blockDim.x) {
        const float v = __half2float(row[c]);
        dst[c] = v;
        s += v * v;
    }
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) N + eps);
    for (int c = tid; c < N; c += blockDim.x) {
        const float w = norm_w ? norm_w[c] : 1.0f;
        dst[c] = dst[c] * r * w;
    }
}

/// 8. Zero-Split-K strict IEEE-754 FP32 Router SGEMM (Wave 8 Cut D): Y[T, M] = X[T, K] * W[M, K]^T (NO TF32!).
/// BM_T=32 (along T), BN_E=32 (along M), BK=64, 128 threads (4 K-sliced warps, 8x4 = 32 FP32 accs/thr),
/// double-buffered XOR-swizzled float4 SMEM (32,768 B -> 3 CTAs/SM on Ada, 0 bank conflicts on STS.128 & LDS.128),
/// deterministic 4-warp SMEM epilogue reduction & coalesced __stcs stores, STACK:0.
/// Launch: dim3((M + 31) / 32, (T + 31) / 32), Block: 128.
__global__ void __launch_bounds__(128, 3) k_router_sgemm_f32_pp(
        const float* __restrict__ X, const float* __restrict__ W, float* __restrict__ Y,
        int T, int M, int K) {
    constexpr int BM_T = 32, BN_E = 32, BK = 64, BK4 = BK / 4;
    __shared__ __align__(16) float4 sX[2][BK4][BM_T];  // 16,384 B (XOR-swizzled: 0 bank conflicts!)
    __shared__ __align__(16) float4 sW[2][BK4][BN_E];  // 16,384 B -> 32,768 B total (3 CTAs/SM on 100 KB SMEM)

    const int m0 = (int) blockIdx.x * BN_E;
    const int t0 = (int) blockIdx.y * BM_T;
    const int tid = threadIdx.x;
    const int wk = tid >> 5, lane = tid & 31;  // wk = 0..3 K-sliced warps (16 K / warp)
    const int lt = lane >> 3;                  // 0..3 (owns 8 T-rows: t = lt + 4*ti)
    const int le = lane & 7;                   // 0..7 (owns 4 E-cols: e = le + 8*ei)

    // Global loader coordinates: 128 threads = 8 rows x 16 float4s (64 floats = 256 B coalesced per row, 4 passes = 32 rows).
    const int ld_r0 = tid >> 4;   // 0..7
    const int ld_k4 = tid & 15;   // 0..15 (float4 index within BK=64)
    const int k_swz = ld_k4 & 7;

    auto load_stage = [&](int stage, int k0) {
        const int k_abs = k0 + (ld_k4 << 2);
        const bool k_full = ((K & 3) == 0) && (k_abs + 4 <= K);
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int row = p * 8 + ld_r0;
            const int row_swz = row ^ k_swz;
            float4 xv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const int tr = t0 + row;
            if (tr < T) {
                if (k_full) {
                    xv = *reinterpret_cast<const float4*>(X + (size_t) tr * K + k_abs);
                } else if (k_abs < K) {
                    const float* xp = X + (size_t) tr * K + k_abs;
                    if (k_abs + 0 < K) xv.x = xp[0];
                    if (k_abs + 1 < K) xv.y = xp[1];
                    if (k_abs + 2 < K) xv.z = xp[2];
                    if (k_abs + 3 < K) xv.w = xp[3];
                }
            }
            sX[stage][ld_k4][row_swz] = xv;

            float4 wv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const int mr = m0 + row;
            if (mr < M) {
                if (k_full) {
                    wv = __ldg(reinterpret_cast<const float4*>(W + (size_t) mr * K + k_abs));
                } else if (k_abs < K) {
                    const float* wp = W + (size_t) mr * K + k_abs;
                    if (k_abs + 0 < K) wv.x = __ldg(wp + 0);
                    if (k_abs + 1 < K) wv.y = __ldg(wp + 1);
                    if (k_abs + 2 < K) wv.z = __ldg(wp + 2);
                    if (k_abs + 3 < K) wv.w = __ldg(wp + 3);
                }
            }
            sW[stage][ld_k4][row_swz] = wv;
        }
    };

    float acc[8][4];
#pragma unroll
    for (int ti = 0; ti < 8; ++ti)
#pragma unroll
        for (int ei = 0; ei < 4; ++ei) acc[ti][ei] = 0.0f;

    int stage = 0;
    if (K > 0) load_stage(0, 0);
    __syncthreads();

    const int k4_base = wk * 4;
    for (int k0 = 0; k0 < K; k0 += BK) {
        const int next_stage = stage ^ 1;
        if (k0 + BK < K) {
            load_stage(next_stage, k0 + BK);
        }
#pragma unroll
        for (int dk4 = 0; dk4 < 4; ++dk4) {
            const int k4 = k4_base + dk4;
            const int sw = k4 & 7;
            float4 wx[4];
#pragma unroll
            for (int ei = 0; ei < 4; ++ei) {
                wx[ei] = sW[stage][k4][(le + 8 * ei) ^ sw];
            }
#pragma unroll
            for (int ti = 0; ti < 8; ++ti) {
                const float4 xv = sX[stage][k4][(lt + 4 * ti) ^ sw];
#pragma unroll
                for (int ei = 0; ei < 4; ++ei) {
                    acc[ti][ei] = fmaf(xv.x, wx[ei].x, acc[ti][ei]);
                    acc[ti][ei] = fmaf(xv.y, wx[ei].y, acc[ti][ei]);
                    acc[ti][ei] = fmaf(xv.z, wx[ei].z, acc[ti][ei]);
                    acc[ti][ei] = fmaf(xv.w, wx[ei].w, acc[ti][ei]);
                }
            }
        }
        stage = next_stage;
        __syncthreads();
    }

    // Deterministic 4-warp K-slice reduction reusing sX (16,384 B = 4 x 32 x 32 floats) + coalesced stcs_f4 stores.
    float (*s_red)[BM_T][BN_E] = reinterpret_cast<float (*)[BM_T][BN_E]>(sX);
#pragma unroll
    for (int ti = 0; ti < 8; ++ti) {
        const int t_idx = lt + 4 * ti;
#pragma unroll
        for (int ei = 0; ei < 4; ++ei) {
            s_red[wk][t_idx][le + 8 * ei] = acc[ti][ei];
        }
    }
    __syncthreads();

    const int out_t0 = tid >> 3;         // 0..15 (16 rows per pass x 2 passes = 32 rows)
    const int out_e  = (tid & 7) << 2;   // 0, 4, ..., 28 (8 lanes = 32 E-cols = 128 B coalesced)
#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        const int tr_loc = pass * 16 + out_t0;
        const int tr = t0 + tr_loc;
        if (tr < T) {
            const float4 v0 = *reinterpret_cast<const float4*>(&s_red[0][tr_loc][out_e]);
            const float4 v1 = *reinterpret_cast<const float4*>(&s_red[1][tr_loc][out_e]);
            const float4 v2 = *reinterpret_cast<const float4*>(&s_red[2][tr_loc][out_e]);
            const float4 v3 = *reinterpret_cast<const float4*>(&s_red[3][tr_loc][out_e]);
            const float4 sum = make_float4(((v0.x + v1.x) + v2.x) + v3.x,
                                           ((v0.y + v1.y) + v2.y) + v3.y,
                                           ((v0.z + v1.z) + v2.z) + v3.z,
                                           ((v0.w + v1.w) + v2.w) + v3.w);
            float* yr = Y + (size_t) tr * M + m0 + out_e;
            if (((M & 3) == 0) && (m0 + out_e + 4 <= M)) {
                stcs_f4(reinterpret_cast<float4*>(yr), sum);
            } else {
                if (m0 + out_e + 0 < M) yr[0] = sum.x;
                if (m0 + out_e + 1 < M) yr[1] = sum.y;
                if (m0 + out_e + 2 < M) yr[2] = sum.z;
                if (m0 + out_e + 3 < M) yr[3] = sum.w;
            }
        }
    }
}

/// 8b. Wave 9: Strict IEEE-754 FP32 Split-K Router SGEMM (NO TF32, NO FP16!).
/// BM_T=64, BN_E=128, BK=16 (split-load 2x8), Block=128 (4 warps = 2x2, 8x8 = 64 FP32 accs/thr).
/// Transposed padded SMEM: sX[2][16][68] (8,704 B) + sW[2][16][132] (16,896 B) = 25,600 B (3 CTAs/SM <= 100 KB, 0 bank conflicts!).
/// Precomputed float4* pointers (dead scalar fallback deleted), split-load BK=16 (12 prefetch regs, HALF barriers, 1024 FFMA/BAR),
/// true register double-buffering a[2][8]/b[2][8], compact float4 fast epilogue -> REG <= 128, STACK: 0!
/// Caller guarantees (K % 16) == 0 and (K / 16) % split_k == 0.
/// Launch: dim3((M + 127) / 128, (T + 63) / 64, split_k), Block: 128.
__global__ void __launch_bounds__(128, 4) k_router_sgemm_f32_pp_splitk(
        const float* __restrict__ X, const float* __restrict__ W, float* __restrict__ Y_out,
        int T, int M, int K, int split_k) {
    constexpr int BM_T = 64, BN_E = 128, BK = 16;
    __shared__ __align__(16) float sX[2][BK][68];   //  8,704 B (0 bank conflicts on STS & LDS.128!)
    __shared__ __align__(16) float sW[2][BK][132];  // 16,896 B -> 25,600 B total

    const int m0 = (int) blockIdx.x * BN_E;
    const int t0 = (int) blockIdx.y * BM_T;
    const int sk = (int) blockIdx.z;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;

    const int k_blocks = K >> 4;
    const int kb0 = (sk * k_blocks) / split_k;
    const int kb1 = ((sk + 1) * k_blocks) / split_k;
    const int k_start = kb0 * BK;
    const int k_end   = kb1 * BK;

    // Global loader coords (128 thr: 64 rows x 2 float4s per 8-K half -> 3 LDG.E.128 per half, 12 prefetch regs!).
    const int ld_r0 = tid >> 1;          // 0..63
    const int ld_k0 = (tid & 1) << 2;    // 0 or 4
    const int tr_ld = t0 + ld_r0;
    const int mr_ld0 = m0 + ld_r0;
    const int mr_ld1 = m0 + ld_r0 + 64;
    const bool tr_ok  = (tr_ld < T);
    const bool mr0_ok = (mr_ld0 < M);
    const bool mr1_ok = (mr_ld1 < M);

    const int k4_off = (k_start + ld_k0) >> 2;
    const float4* px  = reinterpret_cast<const float4*>(X + (size_t) tr_ld  * K) + k4_off;
    const float4* pw0 = reinterpret_cast<const float4*>(W + (size_t) mr_ld0 * K) + k4_off;
    const float4* pw1 = reinterpret_cast<const float4*>(W + (size_t) mr_ld1 * K) + k4_off;

    auto store_s_half = [&](int stg, int k_off, const float4& gx, const float4& gw0, const float4& gw1) {
        const int kb = k_off + ld_k0;
        sX[stg][kb + 0][ld_r0] = gx.x;
        sX[stg][kb + 1][ld_r0] = gx.y;
        sX[stg][kb + 2][ld_r0] = gx.z;
        sX[stg][kb + 3][ld_r0] = gx.w;
        sW[stg][kb + 0][ld_r0] = gw0.x;
        sW[stg][kb + 1][ld_r0] = gw0.y;
        sW[stg][kb + 2][ld_r0] = gw0.z;
        sW[stg][kb + 3][ld_r0] = gw0.w;
        sW[stg][kb + 0][ld_r0 + 64] = gw1.x;
        sW[stg][kb + 1][ld_r0 + 64] = gw1.y;
        sW[stg][kb + 2][ld_r0 + 64] = gw1.z;
        sW[stg][kb + 3][ld_r0 + 64] = gw1.w;
    };

    float acc[8][8];
#pragma unroll
    for (int ti = 0; ti < 8; ++ti)
#pragma unroll
        for (int ei = 0; ei < 8; ++ei) acc[ti][ei] = 0.0f;

    // 4 warps = 2x2 (each warp owns 32x64; each thread owns 8x8 = 2 float4s x 2 float4s).
    const int wm = warp >> 1;            // 0..1
    const int wn = warp & 1;             // 0..1
    const int lt = lane >> 3;            // 0..3
    const int le = lane & 7;             // 0..7
    const int t_row0 = wm * 32 + lt * 4;
    const int t_row1 = t_row0 + 16;
    const int e_col0 = wn * 64 + le * 4;
    const int e_col1 = e_col0 + 32;

    auto load_ab = [&](int stg, int kk, float a[8], float b[8]) {
        const float4 ax0 = *reinterpret_cast<const float4*>(&sX[stg][kk][t_row0]);
        const float4 ax1 = *reinterpret_cast<const float4*>(&sX[stg][kk][t_row1]);
        const float4 bw0 = *reinterpret_cast<const float4*>(&sW[stg][kk][e_col0]);
        const float4 bw1 = *reinterpret_cast<const float4*>(&sW[stg][kk][e_col1]);
        a[0] = ax0.x; a[1] = ax0.y; a[2] = ax0.z; a[3] = ax0.w;
        a[4] = ax1.x; a[5] = ax1.y; a[6] = ax1.z; a[7] = ax1.w;
        b[0] = bw0.x; b[1] = bw0.y; b[2] = bw0.z; b[3] = bw0.w;
        b[4] = bw1.x; b[5] = bw1.y; b[6] = bw1.z; b[7] = bw1.w;
    };

    if (k_start < k_end) {
        const float4 z4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        {
            const float4 gx0  = tr_ok  ? __ldg(px)     : z4;
            const float4 gw00 = mr0_ok ? __ldg(pw0)    : z4;
            const float4 gw10 = mr1_ok ? __ldg(pw1)    : z4;
            const float4 gx1  = tr_ok  ? __ldg(px + 2)  : z4;
            const float4 gw01 = mr0_ok ? __ldg(pw0 + 2) : z4;
            const float4 gw11 = mr1_ok ? __ldg(pw1 + 2) : z4;
            store_s_half(0, 0, gx0, gw00, gw10);
            store_s_half(0, 8, gx1, gw01, gw11);
            px += 4; pw0 += 4; pw1 += 4;
        }
        __syncthreads();

        int stage = 0;
        float a[2][8], b[2][8];
        load_ab(0, 0, a[0], b[0]);

        for (int k0 = k_start; k0 < k_end; k0 += BK) {
            const bool has_next = (k0 + BK < k_end);
            const int next_stage = stage ^ 1;

            // Half 0 (kk = 0..7): overlap 3 LDG.E.128 (next_stage[0..7]) behind 512 FFMAs (compact 2x unroll -> ~9 KB binary)!
            float4 gx = z4, gw0 = z4, gw1 = z4;
            if (has_next) {
                if (tr_ok)  gx  = __ldg(px);
                if (mr0_ok) gw0 = __ldg(pw0);
                if (mr1_ok) gw1 = __ldg(pw1);
            }
#pragma unroll 1
            for (int kb = 0; kb < 8; kb += 2) {
                load_ab(stage, kb + 1, a[1], b[1]);
#pragma unroll
                for (int ti = 0; ti < 8; ++ti) {
#pragma unroll
                    for (int ei = 0; ei < 8; ++ei) acc[ti][ei] = fmaf(a[0][ti], b[0][ei], acc[ti][ei]);
                }
                load_ab(stage, kb + 2, a[0], b[0]);
#pragma unroll
                for (int ti = 0; ti < 8; ++ti) {
#pragma unroll
                    for (int ei = 0; ei < 8; ++ei) acc[ti][ei] = fmaf(a[1][ti], b[1][ei], acc[ti][ei]);
                }
            }
            if (has_next) {
                store_s_half(next_stage, 0, gx, gw0, gw1);
            }

            // Half 1 (kk = 8..15): overlap 3 LDG.E.128 (next_stage[8..15]) behind 512 FFMAs!
            if (has_next) {
                if (tr_ok)  gx  = __ldg(px + 2);
                if (mr0_ok) gw0 = __ldg(pw0 + 2);
                if (mr1_ok) gw1 = __ldg(pw1 + 2);
                px += 4; pw0 += 4; pw1 += 4;
            }
#pragma unroll 1
            for (int kb = 8; kb < 14; kb += 2) {
                load_ab(stage, kb + 1, a[1], b[1]);
#pragma unroll
                for (int ti = 0; ti < 8; ++ti) {
#pragma unroll
                    for (int ei = 0; ei < 8; ++ei) acc[ti][ei] = fmaf(a[0][ti], b[0][ei], acc[ti][ei]);
                }
                load_ab(stage, kb + 2, a[0], b[0]);
#pragma unroll
                for (int ti = 0; ti < 8; ++ti) {
#pragma unroll
                    for (int ei = 0; ei < 8; ++ei) acc[ti][ei] = fmaf(a[1][ti], b[1][ei], acc[ti][ei]);
                }
            }
            load_ab(stage, 15, a[1], b[1]);
#pragma unroll
            for (int ti = 0; ti < 8; ++ti) {
#pragma unroll
                for (int ei = 0; ei < 8; ++ei) acc[ti][ei] = fmaf(a[0][ti], b[0][ei], acc[ti][ei]);
            }
#pragma unroll
            for (int ti = 0; ti < 8; ++ti) {
#pragma unroll
                for (int ei = 0; ei < 8; ++ei) acc[ti][ei] = fmaf(a[1][ti], b[1][ei], acc[ti][ei]);
            }

            if (has_next) {
                store_s_half(next_stage, 8, gx, gw0, gw1);
                __syncthreads();
                stage = next_stage;
                load_ab(stage, 0, a[0], b[0]);
            }
        }
    }

    // Direct coalesced float4 fast-path epilogue ((M & 3) == 0); compact SMEM-staged fallback for non-mod-4 M!
    float* y_base = Y_out + (size_t) sk * (size_t) T * (size_t) M;
    const int mc0 = m0 + e_col0;
    const int mc1 = m0 + e_col1;
    if ((M & 3) == 0) {
        const bool mc0_ok = (mc0 < M);
        const bool mc1_ok = (mc1 < M);
#pragma unroll
        for (int ti = 0; ti < 4; ++ti) {
            const int tr0 = t0 + t_row0 + ti;
            if (tr0 < T) {
                float* dst = y_base + (size_t) tr0 * M;
                if (mc0_ok) *reinterpret_cast<float4*>(dst + mc0) = make_float4(acc[ti][0], acc[ti][1], acc[ti][2], acc[ti][3]);
                if (mc1_ok) *reinterpret_cast<float4*>(dst + mc1) = make_float4(acc[ti][4], acc[ti][5], acc[ti][6], acc[ti][7]);
            }
            const int tr1 = t0 + t_row1 + ti;
            if (tr1 < T) {
                float* dst = y_base + (size_t) tr1 * M;
                if (mc0_ok) *reinterpret_cast<float4*>(dst + mc0) = make_float4(acc[ti + 4][0], acc[ti + 4][1], acc[ti + 4][2], acc[ti + 4][3]);
                if (mc1_ok) *reinterpret_cast<float4*>(dst + mc1) = make_float4(acc[ti + 4][4], acc[ti + 4][5], acc[ti + 4][6], acc[ti + 4][7]);
            }
        }
    } else {
        __syncthreads();
        float* s_ep = &sW[0][0][0];  // 4,224 floats >= 32 * 128 = 4,096 floats
#pragma unroll
        for (int half = 0; half < 2; ++half) {
#pragma unroll
            for (int ti = 0; ti < 4; ++ti) {
                const int r_loc = wm * 16 + lt * 4 + ti;
                *reinterpret_cast<float4*>(s_ep + r_loc * 128 + e_col0) =
                    make_float4(acc[half * 4 + ti][0], acc[half * 4 + ti][1], acc[half * 4 + ti][2], acc[half * 4 + ti][3]);
                *reinterpret_cast<float4*>(s_ep + r_loc * 128 + e_col1) =
                    make_float4(acc[half * 4 + ti][4], acc[half * 4 + ti][5], acc[half * 4 + ti][6], acc[half * 4 + ti][7]);
            }
            __syncthreads();
#pragma unroll 1
            for (int i = tid; i < 32 * 128; i += 128) {
                const int r_loc = i >> 7;
                const int mc = m0 + (i & 127);
                const int tr = t0 + (r_loc >> 4) * 32 + (r_loc & 15) + half * 16;
                if (tr < T && mc < M) {
                    y_base[(size_t) tr * M + mc] = s_ep[i];
                }
            }
            if (half == 0) __syncthreads();
        }
    }
}

/// 8b2. Wave 11 (Path C1): Ozaki-2 FP16-pair Tensor Core Split-K Router GEMM (22-bit mantissa FP32-exact, STACK: 0!).
/// Computes partial[sk, T, N] = X[T, k_start..k_end] @ W[N, k_start..k_end]^T via 3x mma.m16n8k16.f32:
///   acc_hi += X_hi @ W_hi^T,  acc_lo += X_hi @ W_lo^T + X_lo @ W_hi^T,  out = fmaf(acc_lo, 1/2048, acc_hi).
/// Tile: BM=32, BN=64, BK=32, Block=128 (2x2 warps, 16x32/warp = 4 m16n8 tiles), double-buffered XOR-swizzled SMEM (24 KB -> 4 CTAs/SM).
/// Launch: dim3((N + 63) / 64, (T + 31) / 32, sk), Block: 128. (When sk == 1, writes directly to Y[T, N]!)
__global__ void __launch_bounds__(128, 4) k_router_ozaki2_tc_pp_splitk(
        const float* __restrict__ X,       // [T, K] row-major FP32 (K multiple of 32, e.g. 2304)
        const float* __restrict__ W,       // [N, K] row-major FP32 (N = 128, multiple of 64)
        float* __restrict__ partial,       // [sk, T, N] if sk > 1, else [T, N] direct output Y!
        int T, int N, int K, int k_per_split) {
    const int n0 = (int) blockIdx.x * 64;
    const int t0 = (int) blockIdx.y * 32;
    const int sk = (int) blockIdx.z;
    const int k_start = sk * k_per_split;
    const int k_end = min(K, k_start + k_per_split);
    float* out_base = partial + (size_t) sk * (size_t) T * (size_t) N;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // Double-buffered XOR-swizzled SMEM: 2 * (32*32*2*2 + 64*32*2*2) = 24,576 B (24 KB -> 4 CTAs/SM on sm_89!).
    // Swizzle: swz_c8 = c8 ^ ((row >> 1) & 3) -> 0-way bank conflicts on BOTH STS.128 and ldmatrix.x4 (A & B)!
    __shared__ __align__(16) __half sX_hi[2][32][32];  // 4,096 B
    __shared__ __align__(16) __half sX_lo[2][32][32];  // 4,096 B
    __shared__ __align__(16) __half sW_hi[2][64][32];  // 8,192 B
    __shared__ __align__(16) __half sW_lo[2][64][32];  // 8,192 B

    const int tid = (int) threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int wy = warp >> 1, wx = warp & 1;

    auto ldmat_x4 = [](uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
    };
    auto mma_16816 = [](float d[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    };
    auto split_ozaki_f4x2 = [](const float4& v0, const float4& v1, uint4& out_hi, uint4& out_lo) {
        auto pair_split = [](float a, float b, uint32_t& uhi, uint32_t& ulo) {
            const __half ah = __float2half_rn(a);
            const __half bh = __float2half_rn(b);
            const __half2 h2 = __halves2half2(ah, bh);
            const __half2 l2 = __halves2half2(__float2half_rn((a - __half2float(ah)) * 2048.0f),
                                              __float2half_rn((b - __half2float(bh)) * 2048.0f));
            memcpy(&uhi, &h2, 4);
            memcpy(&ulo, &l2, 4);
        };
        pair_split(v0.x, v0.y, out_hi.x, out_lo.x);
        pair_split(v0.z, v0.w, out_hi.y, out_lo.y);
        pair_split(v1.x, v1.y, out_hi.z, out_lo.z);
        pair_split(v1.z, v1.w, out_hi.w, out_lo.w);
    };

    float acc_hi[4][4];
    float acc_lo[4][4];
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc_hi[ni][j] = 0.0f;
            acc_lo[ni][j] = 0.0f;
        }
    }

    if (k_start < k_end) {
        // Global loader coords: 128 threads = 32 rows x 4 chunks of 8 floats (2 float4s per thread for X, 4 float4s for W).
        const int ld_r = tid >> 2;                   // 0..31
        const int ld_c8 = tid & 3;                   // 0..3
        const int ld_k = ld_c8 << 3;                 // 0, 8, 16, 24
        const int ld_swz = ld_c8 ^ ((ld_r >> 1) & 3);
        const int tr_ld = t0 + ld_r;
        const int nr_ld0 = n0 + ld_r;
        const int nr_ld1 = n0 + ld_r + 32;
        const bool tr_ok  = (tr_ld < T);
        const bool nr0_ok = (nr_ld0 < N);
        const bool nr1_ok = (nr_ld1 < N);

        const int k4_init = (k_start + ld_k) >> 2;
        const float4* px  = reinterpret_cast<const float4*>(X + (size_t) tr_ld  * K) + k4_init;
        const float4* pw0 = reinterpret_cast<const float4*>(W + (size_t) nr_ld0 * K) + k4_init;
        const float4* pw1 = reinterpret_cast<const float4*>(W + (size_t) nr_ld1 * K) + k4_init;

        const float4 z4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const int sc8 = ld_swz << 3;
        auto load_xw0 = [&](int k0, float4& xv0, float4& xv1, float4& w0v0, float4& w0v1) {
            const bool k0_ok = (k0 + ld_k < k_end);
            const bool k1_ok = (k0 + ld_k + 4 < k_end);
            xv0  = (tr_ok  && k0_ok) ? __ldg(px)      : z4;
            xv1  = (tr_ok  && k1_ok) ? __ldg(px + 1)  : z4;
            w0v0 = (nr0_ok && k0_ok) ? __ldg(pw0)     : z4;
            w0v1 = (nr0_ok && k1_ok) ? __ldg(pw0 + 1) : z4;
            px += 8; pw0 += 8;
        };
        auto load_w1 = [&](int k0, float4& w1v0, float4& w1v1) {
            const bool k0_ok = (k0 + ld_k < k_end);
            const bool k1_ok = (k0 + ld_k + 4 < k_end);
            w1v0 = (nr1_ok && k0_ok) ? __ldg(pw1)     : z4;
            w1v1 = (nr1_ok && k1_ok) ? __ldg(pw1 + 1) : z4;
            pw1 += 8;
        };
        auto store_xw0 = [&](int stg, const float4& xv0, const float4& xv1,
                             const float4& w0v0, const float4& w0v1) {
            uint4 xh, xl, w0h, w0l;
            split_ozaki_f4x2(xv0, xv1, xh, xl);
            split_ozaki_f4x2(w0v0, w0v1, w0h, w0l);
            *reinterpret_cast<uint4*>(&sX_hi[stg][ld_r][sc8]) = xh;
            *reinterpret_cast<uint4*>(&sX_lo[stg][ld_r][sc8]) = xl;
            *reinterpret_cast<uint4*>(&sW_hi[stg][ld_r][sc8]) = w0h;
            *reinterpret_cast<uint4*>(&sW_lo[stg][ld_r][sc8]) = w0l;
        };
        auto store_w1 = [&](int stg, const float4& w1v0, const float4& w1v1) {
            uint4 w1h, w1l;
            split_ozaki_f4x2(w1v0, w1v1, w1h, w1l);
            *reinterpret_cast<uint4*>(&sW_hi[stg][ld_r + 32][sc8]) = w1h;
            *reinterpret_cast<uint4*>(&sW_lo[stg][ld_r + 32][sc8]) = w1l;
        };

        // Precompute base shared addresses for ldmatrix.x4 (A: 16x16, B: 2x 16x16 strips).
        const int a_row = wy * 16 + (lane & 15);
        const uint32_t a_swz0 = (uint32_t) ((lane >> 4) ^ ((a_row >> 1) & 3));
        const uint32_t sx_hi_base = static_cast<uint32_t>(__cvta_generic_to_shared(&sX_hi[0][a_row][0]));
        const uint32_t sx_lo_base = static_cast<uint32_t>(__cvta_generic_to_shared(&sX_lo[0][a_row][0]));

        const int b_row = wx * 32 + (((lane >> 4) << 3) | (lane & 7));
        const uint32_t b_swz0 = (uint32_t) (((lane >> 3) & 1) ^ ((b_row >> 1) & 3));
        const uint32_t sw_hi_base = static_cast<uint32_t>(__cvta_generic_to_shared(&sW_hi[0][b_row][0]));
        const uint32_t sw_lo_base = static_cast<uint32_t>(__cvta_generic_to_shared(&sW_lo[0][b_row][0]));

        auto run_k16 = [&](int stg, uint32_t ki_xor) {
            const uint32_t sx_off = (uint32_t) stg * 2048u + ((a_swz0 ^ ki_xor) << 4);
            const uint32_t sw_off = (uint32_t) stg * 4096u + ((b_swz0 ^ ki_xor) << 4);
            uint32_t ah0, ah1, ah2, ah3, al0, al1, al2, al3;
            uint32_t bh0, bh1, bh2, bh3, bl0, bl1, bl2, bl3;
            ldmat_x4(ah0, ah1, ah2, ah3, sx_hi_base + sx_off);
            ldmat_x4(bh0, bh1, bh2, bh3, sw_hi_base + sw_off);
            ldmat_x4(al0, al1, al2, al3, sx_lo_base + sx_off);
            ldmat_x4(bl0, bl1, bl2, bl3, sw_lo_base + sw_off);

            mma_16816(acc_hi[0], ah0, ah1, ah2, ah3, bh0, bh1);
            mma_16816(acc_hi[1], ah0, ah1, ah2, ah3, bh2, bh3);
            mma_16816(acc_lo[0], al0, al1, al2, al3, bh0, bh1);
            mma_16816(acc_lo[1], al0, al1, al2, al3, bh2, bh3);

            // Overlap strip-1 b_hi load behind strip-0 a_hi * b_lo MMAs
            ldmat_x4(bh0, bh1, bh2, bh3, sw_hi_base + sw_off + 1024u);
            mma_16816(acc_lo[0], ah0, ah1, ah2, ah3, bl0, bl1);
            mma_16816(acc_lo[1], ah0, ah1, ah2, ah3, bl2, bl3);

            // Overlap strip-1 b_lo load behind strip-1 b_hi MMAs
            ldmat_x4(bl0, bl1, bl2, bl3, sw_lo_base + sw_off + 1024u);
            mma_16816(acc_hi[2], ah0, ah1, ah2, ah3, bh0, bh1);
            mma_16816(acc_hi[3], ah0, ah1, ah2, ah3, bh2, bh3);
            mma_16816(acc_lo[2], al0, al1, al2, al3, bh0, bh1);
            mma_16816(acc_lo[3], al0, al1, al2, al3, bh2, bh3);

            mma_16816(acc_lo[2], ah0, ah1, ah2, ah3, bl0, bl1);
            mma_16816(acc_lo[3], ah0, ah1, ah2, ah3, bl2, bl3);
        };

        // Prologue: load first K=32 tile into stage 0.
        {
            float4 xv0, xv1, w0v0, w0v1, w1v0, w1v1;
            load_xw0(k_start, xv0, xv1, w0v0, w0v1);
            load_w1(k_start, w1v0, w1v1);
            store_xw0(0, xv0, xv1, w0v0, w0v1);
            store_w1(0, w1v0, w1v1);
        }
        __syncthreads();

        int stage = 0;
#pragma unroll 1
        for (int k0 = k_start; k0 < k_end; k0 += 32) {
            const bool has_next = (k0 + 32 < k_end);
            const int next_stage = stage ^ 1;
            float4 xv0 = z4, xv1 = z4, w0v0 = z4, w0v1 = z4;
            if (has_next) {
                load_xw0(k0 + 32, xv0, xv1, w0v0, w0v1);
            }
            run_k16(stage, 0u);
            float4 w1v0 = z4, w1v1 = z4;
            if (has_next) {
                load_w1(k0 + 32, w1v0, w1v1);
                store_xw0(next_stage, xv0, xv1, w0v0, w0v1);
            }
            run_k16(stage, 2u);
            if (has_next) {
                store_w1(next_stage, w1v0, w1v1);
                __syncthreads();
                stage = next_stage;
            }
        }
    }

    // Epilogue: Combine 22-bit Ozaki-2 (fmaf(acc_lo, 1/2048, acc_hi)) -> coalesced float2 stores!
    constexpr float kInv2048 = 1.0f / 2048.0f;
    const int row0 = t0 + wy * 16 + (lane >> 2);
    const int row1 = row0 + 8;
    const int col_base = n0 + wx * 32 + (lane & 3) * 2;
    if ((N & 1) == 0) {
        if (row0 < T) {
            float* dst0 = out_base + (size_t) row0 * N;
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                const int c = col_base + ni * 8;
                if (c < N) {
                    const float v0 = fmaf(acc_lo[ni][0], kInv2048, acc_hi[ni][0]);
                    const float v1 = fmaf(acc_lo[ni][1], kInv2048, acc_hi[ni][1]);
                    *reinterpret_cast<float2*>(dst0 + c) = make_float2(v0, v1);
                }
            }
        }
        if (row1 < T) {
            float* dst1 = out_base + (size_t) row1 * N;
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                const int c = col_base + ni * 8;
                if (c < N) {
                    const float v2 = fmaf(acc_lo[ni][2], kInv2048, acc_hi[ni][2]);
                    const float v3 = fmaf(acc_lo[ni][3], kInv2048, acc_hi[ni][3]);
                    *reinterpret_cast<float2*>(dst1 + c) = make_float2(v2, v3);
                }
            }
        }
    } else {
        if (row0 < T) {
            float* dst0 = out_base + (size_t) row0 * N;
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                const int c = col_base + ni * 8;
                if (c < N)     dst0[c]     = fmaf(acc_lo[ni][0], kInv2048, acc_hi[ni][0]);
                if (c + 1 < N) dst0[c + 1] = fmaf(acc_lo[ni][1], kInv2048, acc_hi[ni][1]);
            }
        }
        if (row1 < T) {
            float* dst1 = out_base + (size_t) row1 * N;
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                const int c = col_base + ni * 8;
                if (c < N)     dst1[c]     = fmaf(acc_lo[ni][2], kInv2048, acc_hi[ni][2]);
                if (c + 1 < N) dst1[c + 1] = fmaf(acc_lo[ni][3], kInv2048, acc_hi[ni][3]);
            }
        }
    }
#else
    // Clean exact FP32 fallback for pre-Ampere (sm_70 / Volta) build compatibility.
    for (int idx = (int) threadIdx.x; idx < 32 * 64; idx += 128) {
        const int row = t0 + (idx >> 6);
        const int col = n0 + (idx & 63);
        if (row < T && col < N) {
            float s = 0.0f;
            const float* xr = X + (size_t) row * K;
            const float* wr = W + (size_t) col * K;
            for (int k = k_start; k < k_end; ++k) {
                s = fmaf(xr[k], wr[k], s);
            }
            out_base[(size_t) row * N + col] = s;
        }
    }
#endif
}

/// 8b3. Wave 12B: Fast Pre-Split Ozaki-2 FP16-pair Tensor Core Router GEMM (BM=64, BN=128, BK=32, optional Split-K, STACK: 0!).
/// Computes Y[sk, T, N] = (X_hi @ W_hi^T) + (X_hi @ W_lo^T + X_lo @ W_hi^T) / 2048.0f directly from pre-split FP16 pairs!
/// Grid: dim3(1, (T + 63) / 64, sk), Block: 256 (2x4 warps), Static SMEM: 49,152 B (2 CTAs/SM on sm_89!).
/// If k_per_split <= 0 (or gridDim.z == 1), sk=1 and writes directly to Y[T, N];
/// else writes partial[(size_t)blockIdx.z * T * N + row * N + col] for k_router_splitk_reduce!
__global__ void __launch_bounds__(256, 2) k_router_ozaki2_tc_presplit_pp(
    const __half* __restrict__ X_hi, const __half* __restrict__ X_lo, // [T, K]
    const __half* __restrict__ W_hi, const __half* __restrict__ W_lo, // [N, K] (N=128)
    float* __restrict__ Y,                                            // [T, N] or [sk, T, N]
    int T, int N, int K, int k_per_split = 0) {
    const int n0 = (int) blockIdx.x * 128;
    const int t0 = (int) blockIdx.y * 64;
    const int k_start = (k_per_split > 0) ? (int) blockIdx.z * k_per_split : 0;
    const int k_end = (k_per_split > 0) ? min(K, k_start + k_per_split) : K;
    float* out_base = Y + (size_t) blockIdx.z * (size_t) T * (size_t) N;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // Double-buffered XOR-swizzled SMEM: 2 * (64*32*2*2 + 128*32*2*2) = 49,152 B (48 KB -> 2 CTAs/SM = 96 KB <= 100 KB on sm_89!).
    struct __align__(16) SmemTiles {
        __half x_hi[2][64][32];   // +0      (8,192 B; stage stride = 4,096 B)
        __half x_lo[2][64][32];   // +8,192  (8,192 B; stage stride = 4,096 B)
        __half w_hi[2][128][32];  // +16,384 (16,384 B; stage stride = 8,192 B)
        __half w_lo[2][128][32];  // +32,768 (16,384 B; stage stride = 8,192 B)
    };
    __shared__ SmemTiles smem;

    const int tid = (int) threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int wy = warp >> 2, wx = warp & 3; // 2x4 warps: wy in [0,1] owns 32 M-rows, wx in [0,3] owns 32 N-cols

    auto cp_async_cg_16 = [](uint32_t sa, const void* ga, int src_bytes) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :: "r"(sa), "l"(ga), "r"(src_bytes) : "memory");
    };
    auto cp_async_cg_16_full = [](uint32_t sa, const void* ga) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :: "r"(sa), "l"(ga) : "memory");
    };
    auto cp_async_commit = []() { asm volatile("cp.async.commit_group;\n" ::: "memory"); };
    auto cp_async_wait0  = []() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); };
    auto ldmat_x4 = [](uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
    };
    auto mma_16816 = [](float d[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    };

    // 2 M-tiles x 4 N-tiles x 4 floats x 2 (hi+lo) = 64 accumulator regs per thread.
    float acc_hi[2][4][4];
    float acc_lo[2][4][4];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc_hi[mi][ni][j] = 0.0f;
                acc_lo[mi][ni][j] = 0.0f;
            }
        }
    }

    const bool fast_kn = ((K & 31) == 0) && ((k_start & 31) == 0) && ((k_end & 31) == 0) && (n0 + 128 <= N);

    if (k_start < k_end) {
        // Unified 256-thread loader: 64 rows x 4 chunks of 8 halves (16 B).
        // 2 cp.async for X_hi/X_lo + 4 cp.async for W_hi/W_lo = 6 cp.async/thread overlapped behind 48 HMMAs!
        const int ld_r = tid >> 2;                   // 0..63
        const int ld_c8 = tid & 3;                   // 0..3
        const int ld_k = ld_c8 << 3;                 // 0, 8, 16, 24
        const int ld_swz = ld_c8 ^ ((ld_r >> 1) & 3);
        const uint32_t sa_ld0 = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.x_hi[0][ld_r][ld_swz << 3]));

        const int tr = t0 + ld_r;
        const bool x_ok = (tr < T);
        const int xb_fast = x_ok ? 16 : 0;
        const __half* pxh = X_hi + (x_ok ? ((size_t) tr * (size_t) K + (size_t) ld_k) : (size_t) ld_k);
        const __half* pxl = X_lo + (x_ok ? ((size_t) tr * (size_t) K + (size_t) ld_k) : (size_t) ld_k);

        const int nr0 = n0 + ld_r;
        const bool w0_ok = (nr0 < N);
        const bool w1_ok = (nr0 + 64 < N);
        const size_t w_stride64 = (size_t) 64 * (size_t) K;
        const __half* pwh = W_hi + (w0_ok ? ((size_t) nr0 * (size_t) K + (size_t) ld_k) : 0u);
        const __half* pwl = W_lo + (w0_ok ? ((size_t) nr0 * (size_t) K + (size_t) ld_k) : 0u);

        // Zero-branch fast tile loader for K % 32 == 0 && n0 + 128 <= N (e.g., K == 2304, N == 128).
        auto issue_tile_fast = [&](int stg, int k0) {
            const uint32_t x_stg = (uint32_t) stg * 4096u;
            const uint32_t w_stg = (uint32_t) stg * 8192u;
            const size_t w1_off = w_stride64 + (size_t) k0;

            cp_async_cg_16(sa_ld0 + x_stg,         (const void*) (pxh + k0), xb_fast);
            cp_async_cg_16(sa_ld0 + 8192u + x_stg, (const void*) (pxl + k0), xb_fast);
            cp_async_cg_16_full(sa_ld0 + 16384u + w_stg, (const void*) (pwh + k0));
            cp_async_cg_16_full(sa_ld0 + 32768u + w_stg, (const void*) (pwl + k0));
            cp_async_cg_16_full(sa_ld0 + 20480u + w_stg, (const void*) (pwh + w1_off));
            cp_async_cg_16_full(sa_ld0 + 36864u + w_stg, (const void*) (pwl + w1_off));
            cp_async_commit();
        };

        auto issue_tile_gen = [&](int stg, int k0) {
            const int rem = k_end - (k0 + ld_k);
            const int bk = (rem > 0) ? ((rem >= 8) ? 16 : (rem * 2)) : 0;
            const uint32_t x_stg = (uint32_t) stg * 4096u;
            const uint32_t w_stg = (uint32_t) stg * 8192u;

            const int xb = x_ok ? bk : 0;
            cp_async_cg_16(sa_ld0 + x_stg,
                           (xb > 0) ? (const void*) (pxh + k0) : (const void*) X_hi, xb);
            cp_async_cg_16(sa_ld0 + 8192u + x_stg,
                           (xb > 0) ? (const void*) (pxl + k0) : (const void*) X_lo, xb);

            const int wb0 = w0_ok ? bk : 0;
            cp_async_cg_16(sa_ld0 + 16384u + w_stg,
                           (wb0 > 0) ? (const void*) (pwh + k0) : (const void*) W_hi, wb0);
            cp_async_cg_16(sa_ld0 + 32768u + w_stg,
                           (wb0 > 0) ? (const void*) (pwl + k0) : (const void*) W_lo, wb0);

            const int wb1 = w1_ok ? bk : 0;
            const size_t w1_off = w_stride64 + (size_t) k0;
            cp_async_cg_16(sa_ld0 + 20480u + w_stg,
                           (wb1 > 0) ? (const void*) (pwh + w1_off) : (const void*) W_hi, wb1);
            cp_async_cg_16(sa_ld0 + 36864u + w_stg,
                           (wb1 > 0) ? (const void*) (pwl + w1_off) : (const void*) W_lo, wb1);

            cp_async_commit();
        };

        // Precompute ldmatrix.x4 base SMEM addresses (A: 2x 16x16 strips, B: 2x 16x16 strips).
        // Note: ((row + 16) >> 1) & 3 == (row >> 1) & 3, so strip-1 has identical swizzle at +1,024 B!
        const int a_row = wy * 32 + (lane & 15);
        const uint32_t a_swz0 = (uint32_t) ((lane >> 4) ^ ((a_row >> 1) & 3));
        const uint32_t sx_base = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.x_hi[0][a_row][0]));

        const int b_row = wx * 32 + (((lane >> 4) << 3) | (lane & 7));
        const uint32_t b_swz0 = (uint32_t) (((lane >> 3) & 1) ^ ((b_row >> 1) & 3));
        const uint32_t sw_base = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w_hi[0][b_row][0]));

        auto run_k16 = [&](int stg, uint32_t ki_xor) {
            const uint32_t sx_addr = sx_base + (uint32_t) stg * 4096u + ((a_swz0 ^ ki_xor) << 4);
            const uint32_t sw_addr = sw_base + (uint32_t) stg * 8192u + ((b_swz0 ^ ki_xor) << 4);

            uint32_t ah00, ah01, ah02, ah03, al00, al01, al02, al03;
            uint32_t ah10, ah11, ah12, ah13, al10, al11, al12, al13;
            uint32_t bh0, bh1, bh2, bh3, bl0, bl1, bl2, bl3;

            // Load M-tile 0 (hi/lo) + N-strip 0 (hi/lo)
            ldmat_x4(ah00, ah01, ah02, ah03, sx_addr);
            ldmat_x4(bh0, bh1, bh2, bh3, sw_addr);
            ldmat_x4(al00, al01, al02, al03, sx_addr + 8192u);
            ldmat_x4(bl0, bl1, bl2, bl3, sw_addr + 16384u);

            // M-tile 0 x N-strip 0 (overlap M-tile 1 loads behind HMMAs)
            mma_16816(acc_hi[0][0], ah00, ah01, ah02, ah03, bh0, bh1);
            mma_16816(acc_hi[0][1], ah00, ah01, ah02, ah03, bh2, bh3);
            ldmat_x4(ah10, ah11, ah12, ah13, sx_addr + 1024u);
            mma_16816(acc_lo[0][0], al00, al01, al02, al03, bh0, bh1);
            mma_16816(acc_lo[0][1], al00, al01, al02, al03, bh2, bh3);
            ldmat_x4(al10, al11, al12, al13, sx_addr + 9216u);
            mma_16816(acc_lo[0][0], ah00, ah01, ah02, ah03, bl0, bl1);
            mma_16816(acc_lo[0][1], ah00, ah01, ah02, ah03, bl2, bl3);

            // M-tile 1 x N-strip 0 (overlap N-strip 1 loads behind HMMAs)
            mma_16816(acc_hi[1][0], ah10, ah11, ah12, ah13, bh0, bh1);
            mma_16816(acc_hi[1][1], ah10, ah11, ah12, ah13, bh2, bh3);
            mma_16816(acc_lo[1][0], al10, al11, al12, al13, bh0, bh1);
            mma_16816(acc_lo[1][1], al10, al11, al12, al13, bh2, bh3);

            ldmat_x4(bh0, bh1, bh2, bh3, sw_addr + 1024u);
            mma_16816(acc_lo[1][0], ah10, ah11, ah12, ah13, bl0, bl1);
            mma_16816(acc_lo[1][1], ah10, ah11, ah12, ah13, bl2, bl3);

            ldmat_x4(bl0, bl1, bl2, bl3, sw_addr + 17408u);

            // M-tile 0 & 1 x N-strip 1
            mma_16816(acc_hi[0][2], ah00, ah01, ah02, ah03, bh0, bh1);
            mma_16816(acc_hi[0][3], ah00, ah01, ah02, ah03, bh2, bh3);
            mma_16816(acc_lo[0][2], al00, al01, al02, al03, bh0, bh1);
            mma_16816(acc_lo[0][3], al00, al01, al02, al03, bh2, bh3);

            mma_16816(acc_hi[1][2], ah10, ah11, ah12, ah13, bh0, bh1);
            mma_16816(acc_hi[1][3], ah10, ah11, ah12, ah13, bh2, bh3);
            mma_16816(acc_lo[1][2], al10, al11, al12, al13, bh0, bh1);
            mma_16816(acc_lo[1][3], al10, al11, al12, al13, bh2, bh3);

            mma_16816(acc_lo[0][2], ah00, ah01, ah02, ah03, bl0, bl1);
            mma_16816(acc_lo[0][3], ah00, ah01, ah02, ah03, bl2, bl3);
            mma_16816(acc_lo[1][2], ah10, ah11, ah12, ah13, bl0, bl1);
            mma_16816(acc_lo[1][3], ah10, ah11, ah12, ah13, bl2, bl3);
        };

        // Prologue: load first K=32 tile into stage 0.
        if (fast_kn) {
            issue_tile_fast(0, k_start);
        } else {
            issue_tile_gen(0, k_start);
        }
        cp_async_wait0();
        __syncthreads();

        int stage = 0;
#pragma unroll 1
        for (int k0 = k_start; k0 < k_end; k0 += 32) {
            const bool has_next = (k0 + 32 < k_end);
            const int next_stage = stage ^ 1;
            if (has_next) {
                if (fast_kn) {
                    issue_tile_fast(next_stage, k0 + 32);
                } else {
                    issue_tile_gen(next_stage, k0 + 32);
                }
            }
            run_k16(stage, 0u);
            run_k16(stage, 2u);
            if (has_next) {
                cp_async_wait0();
                __syncthreads();
                stage = next_stage;
            }
        }
    }

    // Epilogue: 22-bit Scaled-Ozaki combine -> 0-SMEM, 0-BAR warp-paired STG.128 (float4) when N % 4 == 0!
    constexpr float kInv2048 = 1.0f / 2048.0f;
    if ((N & 3) == 0) {
        const bool is_odd = (lane & 1) != 0;
        const int col4_base = n0 + wx * 32 + ((lane & 2) << 1);
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            const int row = t0 + wy * 32 + mi * 16 + (lane >> 2) + (is_odd ? 8 : 0);
            const bool row_ok = (row < T);
            float* dst = out_base + (size_t) row * (size_t) N;
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                const float v0 = fmaf(acc_lo[mi][ni][0], kInv2048, acc_hi[mi][ni][0]);
                const float v1 = fmaf(acc_lo[mi][ni][1], kInv2048, acc_hi[mi][ni][1]);
                const float v2 = fmaf(acc_lo[mi][ni][2], kInv2048, acc_hi[mi][ni][2]);
                const float v3 = fmaf(acc_lo[mi][ni][3], kInv2048, acc_hi[mi][ni][3]);
                const float peer_a = __shfl_xor_sync(0xffffffffu, is_odd ? v0 : v2, 1);
                const float peer_b = __shfl_xor_sync(0xffffffffu, is_odd ? v1 : v3, 1);
                const int c = col4_base + ni * 8;
                if (row_ok && (fast_kn || c < N)) {
                    const float4 out4 = is_odd ? make_float4(peer_a, peer_b, v2, v3)
                                               : make_float4(v0, v1, peer_a, peer_b);
                    *reinterpret_cast<float4*>(dst + c) = out4;
                }
            }
        }
    } else if ((N & 1) == 0) {
        const int col_base = n0 + wx * 32 + (lane & 3) * 2;
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            const int row0 = t0 + wy * 32 + mi * 16 + (lane >> 2);
            const int row1 = row0 + 8;
            if (row0 < T) {
                float* dst0 = out_base + (size_t) row0 * (size_t) N;
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    const int c = col_base + ni * 8;
                    if (c < N) {
                        const float v0 = fmaf(acc_lo[mi][ni][0], kInv2048, acc_hi[mi][ni][0]);
                        const float v1 = fmaf(acc_lo[mi][ni][1], kInv2048, acc_hi[mi][ni][1]);
                        *reinterpret_cast<float2*>(dst0 + c) = make_float2(v0, v1);
                    }
                }
            }
            if (row1 < T) {
                float* dst1 = out_base + (size_t) row1 * (size_t) N;
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    const int c = col_base + ni * 8;
                    if (c < N) {
                        const float v2 = fmaf(acc_lo[mi][ni][2], kInv2048, acc_hi[mi][ni][2]);
                        const float v3 = fmaf(acc_lo[mi][ni][3], kInv2048, acc_hi[mi][ni][3]);
                        *reinterpret_cast<float2*>(dst1 + c) = make_float2(v2, v3);
                    }
                }
            }
        }
    } else {
        const int col_base = n0 + wx * 32 + (lane & 3) * 2;
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            const int row0 = t0 + wy * 32 + mi * 16 + (lane >> 2);
            const int row1 = row0 + 8;
            if (row0 < T) {
                float* dst0 = out_base + (size_t) row0 * (size_t) N;
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    const int c = col_base + ni * 8;
                    if (c < N)     dst0[c]     = fmaf(acc_lo[mi][ni][0], kInv2048, acc_hi[mi][ni][0]);
                    if (c + 1 < N) dst0[c + 1] = fmaf(acc_lo[mi][ni][1], kInv2048, acc_hi[mi][ni][1]);
                }
            }
            if (row1 < T) {
                float* dst1 = out_base + (size_t) row1 * (size_t) N;
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    const int c = col_base + ni * 8;
                    if (c < N)     dst1[c]     = fmaf(acc_lo[mi][ni][2], kInv2048, acc_hi[mi][ni][2]);
                    if (c + 1 < N) dst1[c + 1] = fmaf(acc_lo[mi][ni][3], kInv2048, acc_hi[mi][ni][3]);
                }
            }
        }
    }
#else
    // Clean exact Ozaki-2 fallback for pre-Ampere (sm_70 / Volta) build compatibility.
    constexpr float kInv2048 = 1.0f / 2048.0f;
    for (int idx = (int) threadIdx.x; idx < 64 * 128; idx += 256) {
        const int row = t0 + (idx >> 7);
        const int col = n0 + (idx & 127);
        if (row < T && col < N) {
            float sh = 0.0f, sl = 0.0f;
            const __half* xh = X_hi + (size_t) row * (size_t) K;
            const __half* xl = X_lo + (size_t) row * (size_t) K;
            const __half* wh = W_hi + (size_t) col * (size_t) K;
            const __half* wl = W_lo + (size_t) col * (size_t) K;
            for (int k = k_start; k < k_end; ++k) {
                const float xhv = __half2float(xh[k]);
                const float xlv = __half2float(xl[k]);
                const float whv = __half2float(wh[k]);
                const float wlv = __half2float(wl[k]);
                sh = fmaf(xhv, whv, sh);
                sl = fmaf(xhv, wlv, fmaf(xlv, whv, sl));
            }
            out_base[(size_t) row * (size_t) N + (size_t) col] = fmaf(sl, kInv2048, sh);
        }
    }
#endif
}

/// 8c. Wave 8.5 Cut D: Deterministic Split-K Router Reduction (binary tree for split_k==8, fixed-order general).
/// Launch: dim3(((total_elems + 3) / 4 + 255) / 256), Block: 256.
__global__ void __launch_bounds__(256, 4) k_router_splitk_reduce(
        const float* __restrict__ Y_part, float* __restrict__ Y_out,
        int total_elems, int split_k) {
    const int i4 = (int) blockIdx.x * 256 + (int) threadIdx.x;
    const int idx0 = i4 << 2;
    if (idx0 >= total_elems) return;

    auto add4 = [](float4 a, float4 b) -> float4 {
        return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    };

    if ((total_elems & 3) == 0) {
        const size_t stride4 = (size_t) (total_elems >> 2);
        const float4* yp4 = reinterpret_cast<const float4*>(Y_part);
        float4 sum;
        if (split_k == 8) {
            const float4 p0 = yp4[0u * stride4 + (size_t) i4];
            const float4 p1 = yp4[1u * stride4 + (size_t) i4];
            const float4 p2 = yp4[2u * stride4 + (size_t) i4];
            const float4 p3 = yp4[3u * stride4 + (size_t) i4];
            const float4 p4 = yp4[4u * stride4 + (size_t) i4];
            const float4 p5 = yp4[5u * stride4 + (size_t) i4];
            const float4 p6 = yp4[6u * stride4 + (size_t) i4];
            const float4 p7 = yp4[7u * stride4 + (size_t) i4];
            sum = add4(add4(add4(p0, p1), add4(p2, p3)), add4(add4(p4, p5), add4(p6, p7)));
        } else {
            sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int sk = 0; sk < split_k; ++sk) {
                sum = add4(sum, yp4[(size_t) sk * stride4 + (size_t) i4]);
            }
        }
        reinterpret_cast<float4*>(Y_out)[i4] = sum;
    } else {
        const size_t stride = (size_t) total_elems;
        for (int c = 0; c < 4 && idx0 + c < total_elems; ++c) {
            const size_t idx = (size_t) (idx0 + c);
            float s;
            if (split_k == 8) {
                s = ((Y_part[0u * stride + idx] + Y_part[1u * stride + idx])
                   + (Y_part[2u * stride + idx] + Y_part[3u * stride + idx]))
                  + ((Y_part[4u * stride + idx] + Y_part[5u * stride + idx])
                   + (Y_part[6u * stride + idx] + Y_part[7u * stride + idx]));
            } else {
                s = 0.0f;
                for (int sk = 0; sk < split_k; ++sk) s += Y_part[(size_t) sk * stride + idx];
            }
            Y_out[idx] = s;
        }
    }
}

/// 9a. DSpark Zero-PCIe Running VRAM Accumulator (exact left-to-right IEEE FP32).
/// cur: [T, N]; acc1 (Full): [T, N]; acc0/acc2 (SWA): [T - swa_start, N]; acc_last: [N] (for t == T - 1).
/// Grid: (T), Block: (N == 2304) ? 576 : 256.
__global__ void __launch_bounds__(576, 2) k_dspark_accum_cur(
        const float* __restrict__ cur,
        float* __restrict__ acc0, float* __restrict__ acc1, float* __restrict__ acc2,
        float* __restrict__ acc_last,
        float w0, float w1, float w2, float w3,
        int T, int swa_start, int N, bool first_k) {
    const int t = (int) blockIdx.x;
    if (t >= T) return;
    const int n4 = N >> 2;
    const float4* src4 = reinterpret_cast<const float4*>(cur + (size_t) t * N);
    const bool do_swa = (t >= swa_start);
    const size_t swa_off = do_swa ? ((size_t) (t - swa_start) * N) : 0u;
    const size_t full_off = (size_t) t * N;
    const bool do_last = (acc_last != nullptr) && (t == T - 1);

    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 x = src4[i];
        if (acc1) {
            float4* d1 = reinterpret_cast<float4*>(acc1 + full_off) + i;
            if (first_k) {
                stcs_f4(d1, make_float4(w1 * x.x, w1 * x.y, w1 * x.z, w1 * x.w));
            } else {
                const float4 a = *d1;
                stcs_f4(d1, make_float4(a.x + w1 * x.x, a.y + w1 * x.y, a.z + w1 * x.z, a.w + w1 * x.w));
            }
        }
        if (do_swa) {
            if (acc0) {
                float4* d0 = reinterpret_cast<float4*>(acc0 + swa_off) + i;
                if (first_k) {
                    stcs_f4(d0, make_float4(w0 * x.x, w0 * x.y, w0 * x.z, w0 * x.w));
                } else {
                    const float4 a = *d0;
                    stcs_f4(d0, make_float4(a.x + w0 * x.x, a.y + w0 * x.y, a.z + w0 * x.z, a.w + w0 * x.w));
                }
            }
            if (acc2) {
                float4* d2 = reinterpret_cast<float4*>(acc2 + swa_off) + i;
                if (first_k) {
                    stcs_f4(d2, make_float4(w2 * x.x, w2 * x.y, w2 * x.z, w2 * x.w));
                } else {
                    const float4 a = *d2;
                    stcs_f4(d2, make_float4(a.x + w2 * x.x, a.y + w2 * x.y, a.z + w2 * x.z, a.w + w2 * x.w));
                }
            }
        }
        if (do_last) {
            float4* dl = reinterpret_cast<float4*>(acc_last) + i;
            if (first_k) {
                *dl = make_float4(w3 * x.x, w3 * x.y, w3 * x.z, w3 * x.w);
            } else {
                const float4 a = *dl;
                *dl = make_float4(a.x + w3 * x.x, a.y + w3 * x.y, a.z + w3 * x.z, a.w + w3 * x.w);
            }
        }
    }
    const int tail0 = n4 << 2;
    for (int c = tail0 + threadIdx.x; c < N; c += blockDim.x) {
        const float x = cur[full_off + c];
        if (acc1) acc1[full_off + c] = first_k ? (w1 * x) : (acc1[full_off + c] + w1 * x);
        if (do_swa) {
            if (acc0) acc0[swa_off + c] = first_k ? (w0 * x) : (acc0[swa_off + c] + w0 * x);
            if (acc2) acc2[swa_off + c] = first_k ? (w2 * x) : (acc2[swa_off + c] + w2 * x);
        }
        if (do_last) acc_last[c] = first_k ? (w3 * x) : (acc_last[c] + w3 * x);
    }
}

/// 9b. DSpark Final Term (k=3) + Fused RMSNorm + FP16 Output (zero intermediate HBM roundtrip for N==2304!).
/// Grid: (T_eff), Block: (N == 2304) ? 576 : 256.
__global__ void __launch_bounds__(576, 2) k_dspark_accum_final_rmsnorm_f16(
        const float* __restrict__ cur, float* __restrict__ acc,
        const float* __restrict__ rms_w, __half* __restrict__ out_f16,
        float w, float eps, int T_eff, int t_offset, int N) {
    __shared__ float sh[32];
    const int ti = (int) blockIdx.x;
    if (ti >= T_eff) return;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const float* cr = cur + (size_t) (t_offset + ti) * N;
    const float* ar = acc + (size_t) ti * N;
    __half* dst16 = out_f16 + (size_t) ti * N;

    if (N == 2304 && blockDim.x == 576) {
        const float4 cv = reinterpret_cast<const float4*>(cr)[tid];
        const float4 av = reinterpret_cast<const float4*>(ar)[tid];
        const float4 x = make_float4(av.x + w * cv.x, av.y + w * cv.y, av.z + w * cv.z, av.w + w * cv.w);
        float s = x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) sh[warp] = s;
        __syncthreads();
        if (warp == 0) {
            float tot = (lane < 18) ? sh[lane] : 0.0f;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, o);
            if (lane == 0) sh[0] = rsqrtf(tot / 2304.0f + eps);
        }
        __syncthreads();
        const float r = sh[0];
        float4 o;
        if (rms_w) {
            const float4 ww = __ldg(reinterpret_cast<const float4*>(rms_w) + tid);
            o = make_float4(x.x * r * ww.x, x.y * r * ww.y, x.z * r * ww.z, x.w * r * ww.w);
        } else {
            o = make_float4(x.x * r, x.y * r, x.z * r, x.w * r);
        }
        const __half2 p0 = __halves2half2(__float2half_rn(o.x), __float2half_rn(o.y));
        const __half2 p1 = __halves2half2(__float2half_rn(o.z), __float2half_rn(o.w));
        uint2 u;
        memcpy(&u.x, &p0, 4);
        memcpy(&u.y, &p1, 4);
        reinterpret_cast<uint2*>(dst16)[tid] = u;
        return;
    }

    float s = 0.0f;
    for (int c = tid; c < N; c += blockDim.x) {
        const float x = ar[c] + w * cr[c];
        s += x * x;
    }
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) N + eps);
    for (int c = tid; c < N; c += blockDim.x) {
        const float x = ar[c] + w * cr[c];
        const float ww = rms_w ? rms_w[c] : 1.0f;
        dst16[c] = __float2half_rn(x * r * ww);
    }
}

/// 10. Tiled Virtual Loop-Bank Weight Computation (Wave 9: K_PAD=36 16B-aligned double-buffered SMEM,
/// split load_chunk_raw / finish_chunk_regs for 100% HBM overlap, flipped kk4-outer / qi-inner dot-product (6x less LDS.128),
/// unroll-1 tail -> ~8 KB binary, REG <= 64, STACK: 0, 3 CTAs/SM!).
/// Exact FP32 parity with k_loop_bank_weights_pp.
/// Grid: ((T + 31) / 32), Block: 256.
__global__ void __launch_bounds__(256, 4) k_loop_bank_weights_pp_tiled(
        const float* __restrict__ bank0, const float* __restrict__ bank1,
        LoopBankQueries12 qs, float* __restrict__ w_out, float* __restrict__ inv_rms_out,
        int T, int N, float eps,
        uint16_t* __restrict__ bank0_f16 = nullptr, uint16_t* __restrict__ bank1_f16 = nullptr) {
    constexpr int T_TILE = 32, K_CHUNK = 32, K_PAD = 36;
    __shared__ __align__(16) float s_b0[2][T_TILE][K_PAD];  // 9,216 B (16B aligned -> STS.128 & LDS.128, 0 bank conflicts!)
    __shared__ __align__(16) float s_b1[2][T_TILE][K_PAD];  // 9,216 B
    __shared__ float s_tot[3][T_TILE];                      //   384 B
    __shared__ const float* s_q[48];                        //   384 B -> 19,200 B total (3 CTAs/SM = 57.6 KB <= 100 KB!)

    const int t0 = (int) blockIdx.x * T_TILE;
    if (t0 >= T) return;
    const int tid = threadIdx.x;
    const int ld_t  = tid >> 3;         // 0..31
    const int ld_k  = (tid & 7) << 2;   // 0, 4, ..., 28
    const int t_ld  = t0 + ld_t;
    const bool t_ld_ok = (t_ld < T);

    const int t_loc = tid & 31;         // 0..31
    const int q_blk = tid >> 5;         // 0..7
    const int q_base = q_blk * 6;
    const bool full_n = ((N & 31) == 0);
    const bool emit_f16 = (bank0_f16 != nullptr && bank1_f16 != nullptr);

    if (tid < 48) {
        s_q[tid] = qs.q[tid >> 2][tid & 3];
    }

    float acc_ss0 = 0.0f, acc_ss1 = 0.0f, acc_d01 = 0.0f;
    // Phase 1 (TOP of k0 loop): issue ONLY bank0/bank1 ldcs_f4 loads -> 100% HBM latency overlap behind dot-product!
    auto load_chunk_raw = [&](int k0, float4& b0v, float4& b1v) {
        const int kc = k0 + ld_k;
        b0v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        b1v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (t_ld_ok && (full_n || kc + 4 <= N)) {
            const size_t goff = (size_t) t_ld * N + kc;
            b0v = ldcs_f4(reinterpret_cast<const float4*>(bank0 + goff));
            b1v = ldcs_f4(reinterpret_cast<const float4*>(bank1 + goff));
        } else if (t_ld_ok && kc < N) {
            const size_t goff = (size_t) t_ld * N + kc;
            const float* p0 = bank0 + goff;
            const float* p1 = bank1 + goff;
            if (kc + 0 < N) { b0v.x = p0[0]; b1v.x = p1[0]; }
            if (kc + 1 < N) { b0v.y = p0[1]; b1v.y = p1[1]; }
            if (kc + 2 < N) { b0v.z = p0[2]; b1v.z = p1[2]; }
            if (kc + 3 < N) { b0v.w = p0[3]; b1v.w = p1[3]; }
        }
    };
    // Phase 2 (AFTER dot-product loop): emit_f16 (F2FP + stcs_u2) + acc_ss0/acc_ss1/acc_d01 right before STS.128!
    auto finish_chunk_regs = [&](int k0, const float4& b0v, const float4& b1v) {
        if (emit_f16 && t_ld_ok) {
            const int kc = k0 + ld_k;
            const size_t goff = (size_t) t_ld * N + kc;
            if (full_n || kc + 4 <= N) {
                const __half2 h0a = __floats2half2_rn(b0v.x, b0v.y);
                const __half2 h0b = __floats2half2_rn(b0v.z, b0v.w);
                uint2 u0;
                memcpy(&u0.x, &h0a, 4); memcpy(&u0.y, &h0b, 4);
                stcs_u2(reinterpret_cast<uint2*>(bank0_f16 + goff), u0);

                const __half2 h1a = __floats2half2_rn(b1v.x, b1v.y);
                const __half2 h1b = __floats2half2_rn(b1v.z, b1v.w);
                uint2 u1;
                memcpy(&u1.x, &h1a, 4); memcpy(&u1.y, &h1b, 4);
                stcs_u2(reinterpret_cast<uint2*>(bank1_f16 + goff), u1);
            } else if (kc < N) {
                const float* p0 = bank0 + goff;
                const float* p1 = bank1 + goff;
#pragma unroll 1
                for (int i = 0; i < 4 && kc + i < N; ++i) {
                    const __half h0 = __float2half_rn(p0[i]);
                    const __half h1 = __float2half_rn(p1[i]);
                    bank0_f16[goff + i] = *reinterpret_cast<const uint16_t*>(&h0);
                    bank1_f16[goff + i] = *reinterpret_cast<const uint16_t*>(&h1);
                }
            }
        }
        acc_ss0 += b0v.x * b0v.x + b0v.y * b0v.y + b0v.z * b0v.z + b0v.w * b0v.w;
        acc_ss1 += b1v.x * b1v.x + b1v.y * b1v.y + b1v.z * b1v.z + b1v.w * b1v.w;
        acc_d01 += b0v.x * b1v.x + b0v.y * b1v.y + b0v.z * b1v.z + b0v.w * b1v.w;
    };

    float d0[6], d1[6];
#pragma unroll
    for (int qi = 0; qi < 6; ++qi) {
        d0[qi] = 0.0f;
        d1[qi] = 0.0f;
    }

    int stage = 0;
    if (N > 0) {
        float4 b0v, b1v;
        load_chunk_raw(0, b0v, b1v);
        finish_chunk_regs(0, b0v, b1v);
        *reinterpret_cast<float4*>(&s_b0[0][ld_t][ld_k]) = b0v;
        *reinterpret_cast<float4*>(&s_b1[0][ld_t][ld_k]) = b1v;
    }
    __syncthreads();

    for (int k0 = 0; k0 < N; k0 += K_CHUNK) {
        const bool has_next = (k0 + K_CHUNK < N);
        float4 nb0{}, nb1{};
        if (has_next) {
            load_chunk_raw(k0 + K_CHUNK, nb0, nb1);
        }

        if (full_n || k0 + K_CHUNK <= N) {
            // Flipped loop: kk4 outer (unroll-1 -> REG <= 64, STACK: 0, ~8 KB binary), qi inner -> loads v0, v1 ONCE per kk4 (6x less SMEM traffic!)
#pragma unroll 1
            for (int kk4 = 0; kk4 < 8; ++kk4) {
                const float4 v0 = *reinterpret_cast<const float4*>(&s_b0[stage][t_loc][kk4 << 2]);
                const float4 v1 = *reinterpret_cast<const float4*>(&s_b1[stage][t_loc][kk4 << 2]);
#pragma unroll
                for (int qi = 0; qi < 6; ++qi) {
                    const float* q_ptr = s_q[q_base + qi];
                    if (q_ptr != nullptr) {
                        const float4 q4 = __ldg(reinterpret_cast<const float4*>(q_ptr + k0) + kk4);
                        d0[qi] = fmaf(v0.x, q4.x, fmaf(v0.y, q4.y, fmaf(v0.z, q4.z, fmaf(v0.w, q4.w, d0[qi]))));
                        d1[qi] = fmaf(v1.x, q4.x, fmaf(v1.y, q4.y, fmaf(v1.z, q4.z, fmaf(v1.w, q4.w, d1[qi]))));
                    }
                }
            }
        } else {
            const int k_lim = N - k0;
#pragma unroll 1
            for (int kk = 0; kk < k_lim; ++kk) {
                const float bv0 = s_b0[stage][t_loc][kk];
                const float bv1 = s_b1[stage][t_loc][kk];
#pragma unroll
                for (int qi = 0; qi < 6; ++qi) {
                    const float* q_ptr = s_q[q_base + qi];
                    if (q_ptr != nullptr) {
                        const float qv = __ldg(q_ptr + k0 + kk);
                        d0[qi] = fmaf(bv0, qv, d0[qi]);
                        d1[qi] = fmaf(bv1, qv, d1[qi]);
                    }
                }
            }
        }

        if (has_next) {
            finish_chunk_regs(k0 + K_CHUNK, nb0, nb1);
            stage ^= 1;
            *reinterpret_cast<float4*>(&s_b0[stage][ld_t][ld_k]) = nb0;
            *reinterpret_cast<float4*>(&s_b1[stage][ld_t][ld_k]) = nb1;
            __syncthreads();
        }
    }

    acc_ss0 += __shfl_xor_sync(0xffffffffu, acc_ss0, 4);
    acc_ss1 += __shfl_xor_sync(0xffffffffu, acc_ss1, 4);
    acc_d01 += __shfl_xor_sync(0xffffffffu, acc_d01, 4);
    acc_ss0 += __shfl_xor_sync(0xffffffffu, acc_ss0, 2);
    acc_ss1 += __shfl_xor_sync(0xffffffffu, acc_ss1, 2);
    acc_d01 += __shfl_xor_sync(0xffffffffu, acc_d01, 2);
    acc_ss0 += __shfl_xor_sync(0xffffffffu, acc_ss0, 1);
    acc_ss1 += __shfl_xor_sync(0xffffffffu, acc_ss1, 1);
    acc_d01 += __shfl_xor_sync(0xffffffffu, acc_d01, 1);
    if ((tid & 7) == 0) {
        s_tot[0][ld_t] = acc_ss0;
        s_tot[1][ld_t] = acc_ss1;
        s_tot[2][ld_t] = acc_d01;
    }
    __syncthreads();

    const int tg = t0 + t_loc;
    if (tg < T) {
        const float ss0 = s_tot[0][t_loc];
        const float ss1 = s_tot[1][t_loc];
        const float d01 = s_tot[2][t_loc];
        const float inv_sqrt_n = (N == 2304) ? (1.0f / 48.0f) : (1.0f / sqrtf((float) N));
#pragma unroll
        for (int qi = 0; qi < 6; ++qi) {
            if (s_q[q_base + qi] != nullptr) {
                const int idx = q_base + qi;
                const float wv0 = d0[qi] * inv_sqrt_n;
                const float wv1 = d1[qi] * inv_sqrt_n;
                const float mx = fmaxf(wv0, wv1);
                const float e0 = __expf(wv0 - mx);
                const float e1 = __expf(wv1 - mx);
                const float inv_sum = __fdividef(1.0f, e0 + e1);
                const float w0 = e0 * inv_sum;
                const float w1 = e1 * inv_sum;
                const size_t base = (size_t) idx * (size_t) T + (size_t) tg;
                *reinterpret_cast<float2*>(w_out + base * 2u) = make_float2(w0, w1);
                if (inv_rms_out) {
                    const float tot_ss = fmaxf(0.0f, w0 * w0 * ss0
                                                   + 2.0f * w0 * w1 * d01
                                                   + w1 * w1 * ss1);
                    inv_rms_out[base] = rsqrtf(tot_ss / (float) N + eps);
                }
            }
        }
    }
}

/// 11a. Pass-Level Hoisted Virtual-Message Control-Down Precomputation:
/// Computes pre_out[m, t, 0..63] = (w0*inv_m)*(bank0[t] * W_msg[m]^T) + (w1*inv_m)*(bank1[t] * W_msg[m]^T)
/// using mma.m16n8k16.f32 (sm_80+) / WMMA (sm_70+). Loads W_msg ONCE for both bank0 and bank1!
/// Supports both split (bank_w0, bank_w1) AND interleaved (bank_w1 == nullptr -> bank_w0 is [n_mods, T, 2]).
/// Grid: dim3((T + 31) / 32, n_mods_total), Block: 256.
__global__ void __launch_bounds__(256, 2) k_vmsg_down_pre_pass(
        const float* __restrict__ bank0, const float* __restrict__ bank1,
        const float* const* __restrict__ W_down_ptrs,
        const float* __restrict__ bank_w0, const float* __restrict__ bank_w1,
        const float* __restrict__ inv_m, float* __restrict__ pre_out,
        int T, int N, int n_mods_total) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int BM = 32, BN = 64, WK = 64, LD = 72;
    __shared__ __align__(16) __half sx[BM * LD];
    __shared__ __align__(16) __half sw[BN * LD];
    __shared__ float s_sc0[BM];
    __shared__ float s_sc1[BM];

    const int m = (int) blockIdx.y;
    if (m >= n_mods_total) return;
    const float* w_base = W_down_ptrs[m];
    if (!w_base) return;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int t0 = (int) blockIdx.x * BM;
    const int rows = min(BM, T - t0);
    if (rows <= 0) return;

    if (tid < BM) {
        if (tid < rows) {
            const size_t idx = (size_t) m * (size_t) T + (size_t) (t0 + tid);
            const float im = inv_m ? inv_m[idx] : 1.0f;
            float w0, w1;
            if (bank_w1) {
                w0 = bank_w0[idx];
                w1 = bank_w1[idx];
            } else {
                const float2 wv = reinterpret_cast<const float2*>(bank_w0)[idx];
                w0 = wv.x; w1 = wv.y;
            }
            s_sc0[tid] = w0 * im;
            s_sc1[tid] = w1 * im;
        } else {
            s_sc0[tid] = 0.0f;
            s_sc1[tid] = 0.0f;
        }
    }

    const int xr = tid >> 3, xc = (tid & 7) << 3;
    const int wr = tid >> 2, wc = (tid & 3) << 4;
    const bool x_ok = (xr < rows);

    const int wm = warp >> 2, wn = warp & 3;
    const int a_row = lane & 15, a_col = (lane >> 4) << 3;
    const int b_row = ((lane >> 4) << 3) | (lane & 7), b_col = ((lane >> 3) & 1) << 3;
    const uint32_t sx_a_smem = static_cast<uint32_t>(__cvta_generic_to_shared(sx + (wm * 16 + a_row) * LD + a_col));
    const uint32_t sw_b_smem = static_cast<uint32_t>(__cvta_generic_to_shared(sw + (wn * 16 + b_row) * LD + b_col));

    auto ldmat4 = [](uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
    };
    auto mma16816 = [](float d[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    };
    auto load_bank = [&](const float* src, int k0) {
        __half2* dst2 = reinterpret_cast<__half2*>(sx + xr * LD + xc);
        if (x_ok) {
            const float4* sp = reinterpret_cast<const float4*>(src + (size_t) (t0 + xr) * N + k0 + xc);
            const float4 v0 = sp[0], v1 = sp[1];
            dst2[0] = __floats2half2_rn(v0.x, v0.y);
            dst2[1] = __floats2half2_rn(v0.z, v0.w);
            dst2[2] = __floats2half2_rn(v1.x, v1.y);
            dst2[3] = __floats2half2_rn(v1.z, v1.w);
        } else {
            *reinterpret_cast<uint4*>(dst2) = make_uint4(0, 0, 0, 0);
        }
    };
    auto run_mma = [&](float acc_v[8]) {
#pragma unroll
        for (int kk = 0; kk < WK; kk += 16) {
            uint32_t a0, a1, a2, a3, b0, b1, b2, b3;
            const uint32_t off_b = (uint32_t) kk * 2u;
            ldmat4(a0, a1, a2, a3, sx_a_smem + off_b);
            ldmat4(b0, b1, b2, b3, sw_b_smem + off_b);
            mma16816(&acc_v[0], a0, a1, a2, a3, b0, b1);
            mma16816(&acc_v[4], a0, a1, a2, a3, b2, b3);
        }
    };

    float acc0[8] = {}, acc1[8] = {};
    for (int k0 = 0; k0 < N; k0 += WK) {
        // Load W_msg (second half of W_down: column offset N + k0) ONCE for both bank0 and bank1!
        {
            __half2* dst2 = reinterpret_cast<__half2*>(sw + wr * LD + wc);
            const float4* wp = reinterpret_cast<const float4*>(w_base + (size_t) wr * (2 * N) + N + k0 + wc);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = __ldg(wp + i);
                dst2[i * 2 + 0] = __floats2half2_rn(v.x, v.y);
                dst2[i * 2 + 1] = __floats2half2_rn(v.z, v.w);
            }
        }
        load_bank(bank0, k0);
        __syncthreads();
        run_mma(acc0);
        __syncthreads();
        load_bank(bank1, k0);
        __syncthreads();
        run_mma(acc1);
        __syncthreads();
    }

    const int acc_r0 = wm * 16 + (lane >> 2);
    const int acc_r1 = acc_r0 + 8;
    const int acc_c0 = wn * 16 + (lane & 3) * 2;
    const int acc_c1 = acc_c0 + 8;
    float* out_m = pre_out + (size_t) m * (size_t) T * BN;
    if (acc_r0 < rows) {
        const float c0 = s_sc0[acc_r0], c1 = s_sc1[acc_r0];
        float* dst0 = out_m + (size_t) (t0 + acc_r0) * BN;
        *reinterpret_cast<float2*>(dst0 + acc_c0) = make_float2(c0 * acc0[0] + c1 * acc1[0], c0 * acc0[1] + c1 * acc1[1]);
        *reinterpret_cast<float2*>(dst0 + acc_c1) = make_float2(c0 * acc0[4] + c1 * acc1[4], c0 * acc0[5] + c1 * acc1[5]);
    }
    if (acc_r1 < rows) {
        const float c0 = s_sc0[acc_r1], c1 = s_sc1[acc_r1];
        float* dst1 = out_m + (size_t) (t0 + acc_r1) * BN;
        *reinterpret_cast<float2*>(dst1 + acc_c0) = make_float2(c0 * acc0[2] + c1 * acc1[2], c0 * acc0[3] + c1 * acc1[3]);
        *reinterpret_cast<float2*>(dst1 + acc_c1) = make_float2(c0 * acc0[6] + c1 * acc1[6], c0 * acc0[7] + c1 * acc1[7]);
    }
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    constexpr int BM = 32, BN = 64, WK = 64, LD = 72;
    __shared__ __align__(16) __half sx[BM * LD];
    __shared__ __align__(16) __half sw[BN * LD];
    __shared__ float s_sc0[BM];
    __shared__ float s_sc1[BM];

    const int m = (int) blockIdx.y;
    if (m >= n_mods_total) return;
    const float* w_base = W_down_ptrs[m];
    if (!w_base) return;

    const int tid = threadIdx.x, warp = tid >> 5;
    const int t0 = (int) blockIdx.x * BM;
    const int rows = min(BM, T - t0);
    if (rows <= 0) return;

    if (tid < BM) {
        if (tid < rows) {
            const size_t idx = (size_t) m * (size_t) T + (size_t) (t0 + tid);
            const float im = inv_m ? inv_m[idx] : 1.0f;
            float w0, w1;
            if (bank_w1) { w0 = bank_w0[idx]; w1 = bank_w1[idx]; }
            else { const float2 wv = reinterpret_cast<const float2*>(bank_w0)[idx]; w0 = wv.x; w1 = wv.y; }
            s_sc0[tid] = w0 * im;
            s_sc1[tid] = w1 * im;
        } else {
            s_sc0[tid] = 0.0f; s_sc1[tid] = 0.0f;
        }
    }
    const int xr = tid >> 3, xc = (tid & 7) << 3;
    const int wr = tid >> 2, wc = (tid & 3) << 4;
    const bool x_ok = (xr < rows);
    const int wm = warp >> 2, wn = warp & 3;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0, acc1;
    wmma::fill_fragment(acc0, 0.0f);
    wmma::fill_fragment(acc1, 0.0f);

    auto load_bank = [&](const float* src, int k0) {
        __half2* dst2 = reinterpret_cast<__half2*>(sx + xr * LD + xc);
        if (x_ok) {
            const float4* sp = reinterpret_cast<const float4*>(src + (size_t) (t0 + xr) * N + k0 + xc);
            const float4 v0 = sp[0], v1 = sp[1];
            dst2[0] = __floats2half2_rn(v0.x, v0.y); dst2[1] = __floats2half2_rn(v0.z, v0.w);
            dst2[2] = __floats2half2_rn(v1.x, v1.y); dst2[3] = __floats2half2_rn(v1.z, v1.w);
        } else {
            *reinterpret_cast<uint4*>(dst2) = make_uint4(0, 0, 0, 0);
        }
    };
    auto mma_v70 = [&](wmma::fragment<wmma::accumulator, 16, 16, 16, float>& ac) {
#pragma unroll
        for (int kk = 0; kk < WK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
            wmma::load_matrix_sync(fa, sx + (wm * 16) * LD + kk, LD);
            wmma::load_matrix_sync(fb, sw + (wn * 16) * LD + kk, LD);
            wmma::mma_sync(ac, fa, fb, ac);
        }
    };

    for (int k0 = 0; k0 < N; k0 += WK) {
        {
            __half2* dst2 = reinterpret_cast<__half2*>(sw + wr * LD + wc);
            const float4* wp = reinterpret_cast<const float4*>(w_base + (size_t) wr * (2 * N) + N + k0 + wc);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = __ldg(wp + i);
                dst2[i * 2 + 0] = __floats2half2_rn(v.x, v.y);
                dst2[i * 2 + 1] = __floats2half2_rn(v.z, v.w);
            }
        }
        load_bank(bank0, k0);
        __syncthreads();
        mma_v70(acc0);
        __syncthreads();
        load_bank(bank1, k0);
        __syncthreads();
        mma_v70(acc1);
        __syncthreads();
    }

    float* s_out = reinterpret_cast<float*>(sw);
    wmma::store_matrix_sync(s_out + (wm * 16) * BN + wn * 16, acc0, BN, wmma::mem_row_major);
    __syncthreads();
    float v0[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int idx = k * 256 + tid;
        const int r = idx / BN;
        v0[k] = (r < rows) ? (s_sc0[r] * s_out[idx]) : 0.0f;
    }
    __syncthreads();
    wmma::store_matrix_sync(s_out + (wm * 16) * BN + wn * 16, acc1, BN, wmma::mem_row_major);
    __syncthreads();
    float* out_m = pre_out + (size_t) m * (size_t) T * BN;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int idx = k * 256 + tid;
        const int r = idx / BN, c = idx % BN;
        if (r < rows) {
            out_m[(size_t) (t0 + r) * BN + c] = v0[k] + s_sc1[r] * s_out[idx];
        }
    }
#else
    (void) bank0; (void) bank1; (void) W_down_ptrs; (void) bank_w0; (void) bank_w1;
    (void) inv_m; (void) pre_out; (void) T; (void) N; (void) n_mods_total;
#endif
}

/// 11a-lt1. Pack second half (msg cols N..2*N-1) of 48 core loop_down weights [64, 2*N] FP32 -> [48*64, N] FP16
/// for cuBLASLt pass-level vmsg_down_pre GEMM. Grid: dim3(48, 64), Block: 256, STACK:0.
__global__ void __launch_bounds__(256) k_pack_core_wdown_msg_f16(
        const float* const* __restrict__ W_down_ptrs,
        uint16_t* __restrict__ out_f16,
        int N) {
    const int m = (int) blockIdx.x;
    const int r = (int) blockIdx.y;
    const int tid = threadIdx.x;
    const float* w_m = W_down_ptrs[m];
    uint16_t* dst = out_f16 + ((size_t) m * 64u + (size_t) r) * (size_t) N;
    const int n4 = N >> 2;
    if (!w_m) {
        uint2* dst2 = reinterpret_cast<uint2*>(dst);
        const uint2 z = make_uint2(0u, 0u);
        for (int i = tid; i < n4; i += blockDim.x) dst2[i] = z;
        for (int c = (n4 << 2) + tid; c < N; c += blockDim.x) dst[c] = 0u;
        return;
    }
    const float* src = w_m + (size_t) r * (size_t) (2 * N) + (size_t) N;
    const float4* src4 = reinterpret_cast<const float4*>(src);
    uint2* dst2 = reinterpret_cast<uint2*>(dst);
    for (int i = tid; i < n4; i += blockDim.x) {
        const float4 v = __ldg(src4 + i);
        const __half2 h0 = __floats2half2_rn(v.x, v.y);
        const __half2 h1 = __floats2half2_rn(v.z, v.w);
        uint2 u;
        memcpy(&u.x, &h0, 4);
        memcpy(&u.y, &h1, 4);
        dst2[i] = u;
    }
    for (int c = (n4 << 2) + tid; c < N; c += blockDim.x) {
        const __half h = __float2half_rn(__ldg(src + c));
        uint16_t u16;
        memcpy(&u16, &h, 2);
        dst[c] = u16;
    }
}

/// 11a-lt2. Combine cuBLASLt B0_pre / B1_pre [T, n_mods_half * 64] FP32 with (w0, w1) and inv_rms -> pre_out [n_mods_half, T, 64] FP32.
/// Bit-exact indexing with k_loop_bank_weights_pp_tiled & k_vmsg_down_pre_pass.
/// Grid: dim3((T + 7) / 8, n_mods_half), Block: 128 (16 threads x float4 = 64 cols x 8 rows), STACK:0.
__global__ void __launch_bounds__(128) k_vmsg_combine_pre_f32(
        const float* __restrict__ B0_pre,
        const float* __restrict__ B1_pre,
        const float* __restrict__ vmsg_w,
        const float* __restrict__ vmsg_inv_rms,
        float* __restrict__ pre_out,
        int T, int n_mods_half) {
    const int m = (int) blockIdx.y;
    const int t = (int) blockIdx.x * 8 + (threadIdx.x >> 4);
    if (m >= n_mods_half || t >= T) return;
    const int c4 = (threadIdx.x & 15) * 4;
    const size_t w_idx = (size_t) m * (size_t) T + (size_t) t;
    const float2 wv = __ldg(reinterpret_cast<const float2*>(vmsg_w) + w_idx);
    const float inv_m = vmsg_inv_rms ? __ldg(&vmsg_inv_rms[w_idx]) : 1.0f;
    const float c0 = wv.x * inv_m;
    const float c1 = wv.y * inv_m;
    const size_t in_off = (size_t) t * ((size_t) n_mods_half * 64u) + (size_t) m * 64u + (size_t) c4;
    const float4 b0 = *reinterpret_cast<const float4*>(B0_pre + in_off);
    const float4 b1 = *reinterpret_cast<const float4*>(B1_pre + in_off);
    const float4 out = make_float4(c0 * b0.x + c1 * b1.x,
                                   c0 * b0.y + c1 * b1.y,
                                   c0 * b0.z + c1 * b1.z,
                                   c0 * b0.w + c1 * b1.w);
    *reinterpret_cast<float4*>(pre_out + w_idx * 64u + (size_t) c4) = out;
}

struct CtrlDownPhaseAPtrs3 {
    const float* c[3];       // [T, N] FP32
    const float* inv_c[3];   // [T] FP32 precomputed inv_rms of c[m] (optional; computed if nullptr)
    const float* w[3];       // [R=64, 2*N] FP32 loop_down weights (uses first half 0..N-1)
    const float* pre[3];     // [T, R=64] FP32 hoisted vmsg_down_pre (optional; added if non-null)
    float* out[3];           // [T, R=64] FP32 output
    const __half* w_f16[3] = {nullptr, nullptr, nullptr};
    float* part_out[3]     = {nullptr, nullptr, nullptr};
};

/// 11b. High-Occupancy Phase-A Control-Down + Hoisted pre_out Add (Wave 8 Cut A: BM=16, BN=32, WK=128, 128 threads).
/// 13,248 B SMEM (7 CTAs/SM on Ada!), 18 K-steps (2x fewer barriers!), ldmatrix.x4(A)+ldmatrix.x2(B) -> 16x8/warp, STACK:0!
/// Grid: dim3(2, (T + 15) / 16), Block: 128.
template <int M_MODS>
__global__ void __launch_bounds__(128, 4) k_control_down_phaseA_pp_wmma(
        CtrlDownPhaseAPtrs3 p, int T, int N) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int BM = 16, BN = 32, WK = 128, LD = 136;
    __shared__ __align__(16) __half sx[BM * LD];   //  4,352 B
    __shared__ __align__(16) __half sw[BN * LD];   //  8,704 B
    __shared__ float s_irc[M_MODS][BM];            //    192 B -> 13,248 B total (7 CTAs/SM!)

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int col0 = (int) blockIdx.x * BN;        // 0 or 32 (total R=64 cols)
    const int t0   = (int) blockIdx.y * BM;
    const int rows = min(BM, T - t0);
    if (rows <= 0) return;

    const int xr = tid >> 3, xc = (tid & 7) << 4;  // 16 rows x 8 threads/row x 16 floats = 16 x 128
    const int wr = tid >> 2, wc = (tid & 3) << 5;  // 32 rows x 4 threads/row x 32 floats = 32 x 128
    const bool x_ok = (xr < rows);

#pragma unroll
    for (int m = 0; m < M_MODS; ++m) {
        if (p.inv_c[m] != nullptr) {
            if (tid < BM) s_irc[m][tid] = (tid < rows) ? p.inv_c[m][t0 + tid] : 0.0f;
        } else {
            float ss = 0.0f;
            if (x_ok) {
                const float4* cp = reinterpret_cast<const float4*>(p.c[m] + (size_t) (t0 + xr) * N);
                const int n4 = N >> 2;
                for (int i = (tid & 7); i < n4; i += 8) {
                    const float4 v = cp[i];
                    ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
                }
            }
            ss += __shfl_xor_sync(0xffffffffu, ss, 4);
            ss += __shfl_xor_sync(0xffffffffu, ss, 2);
            ss += __shfl_xor_sync(0xffffffffu, ss, 1);
            if ((tid & 7) == 0) s_irc[m][xr] = x_ok ? rsqrtf(ss / (float) N + 1e-6f) : 0.0f;
        }
    }
    __syncthreads();

    const int wn = warp;  // 0..3 (each warp owns 16 x 8 = 1x m16n8k16)
    const int a_row = lane & 15, a_col = (lane >> 4) << 3;
    const int b_row = lane & 7,  b_col = ((lane >> 3) & 1) << 3;
    const uint32_t sx_a_smem = static_cast<uint32_t>(__cvta_generic_to_shared(sx + a_row * LD + a_col));
    const uint32_t sw_b_smem = static_cast<uint32_t>(__cvta_generic_to_shared(sw + (wn * 8 + b_row) * LD + b_col));

    auto ldmat4 = [](uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
    };
    auto ldmat2 = [](uint32_t& r0, uint32_t& r1, uint32_t addr) {
        asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
                     : "=r"(r0), "=r"(r1) : "r"(addr));
    };
    auto mma16816 = [](float d[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    };
    auto run_mma = [&](float acc_m[4]) {
#pragma unroll
        for (int kk = 0; kk < WK; kk += 16) {
            uint32_t a0, a1, a2, a3, b0, b1;
            const uint32_t off_b = (uint32_t) kk * 2u;
            ldmat4(a0, a1, a2, a3, sx_a_smem + off_b);
            ldmat2(b0, b1, sw_b_smem + off_b);
            mma16816(acc_m, a0, a1, a2, a3, b0, b1);
        }
    };
    auto load_x_scaled = [&](const float* src, int k0, float sc) {
        __half2* dst2 = reinterpret_cast<__half2*>(sx + xr * LD + xc);
        if (x_ok && k0 + xc + 16 <= N) {
            const float4* sp = reinterpret_cast<const float4*>(src + (size_t) (t0 + xr) * N + k0 + xc);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = sp[i];
                dst2[i * 2 + 0] = __floats2half2_rn(v.x * sc, v.y * sc);
                dst2[i * 2 + 1] = __floats2half2_rn(v.z * sc, v.w * sc);
            }
        } else if (x_ok) {
            const float* sp = src + (size_t) (t0 + xr) * N + k0 + xc;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int k_i = k0 + xc + i * 2;
                const float v0 = (k_i + 0 < N) ? sp[i * 2 + 0] * sc : 0.0f;
                const float v1 = (k_i + 1 < N) ? sp[i * 2 + 1] * sc : 0.0f;
                dst2[i] = __floats2half2_rn(v0, v1);
            }
        } else {
            const uint4 z = make_uint4(0, 0, 0, 0);
            reinterpret_cast<uint4*>(dst2)[0] = z;
            reinterpret_cast<uint4*>(dst2)[1] = z;
        }
    };
    auto load_w = [&](const float* w_base, int k0) {
        __half2* dst2 = reinterpret_cast<__half2*>(sw + wr * LD + wc);
        if (k0 + wc + 32 <= N) {
            const float4* wp = reinterpret_cast<const float4*>(w_base + (size_t) (col0 + wr) * (2 * N) + k0 + wc);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float4 v = __ldg(wp + i);
                dst2[i * 2 + 0] = __floats2half2_rn(v.x, v.y);
                dst2[i * 2 + 1] = __floats2half2_rn(v.z, v.w);
            }
        } else {
            const float* wp = w_base + (size_t) (col0 + wr) * (2 * N) + k0 + wc;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int k_i = k0 + wc + i * 2;
                const float v0 = (k_i + 0 < N) ? __ldg(wp + i * 2 + 0) : 0.0f;
                const float v1 = (k_i + 1 < N) ? __ldg(wp + i * 2 + 1) : 0.0f;
                dst2[i] = __floats2half2_rn(v0, v1);
            }
        }
    };

    const bool shared_c = (M_MODS > 1) && (p.c[0] == p.c[1]) && (p.inv_c[0] == p.inv_c[1])
                          && (M_MODS < 3 || (p.c[0] == p.c[2] && p.inv_c[0] == p.inv_c[2]));

    float acc[M_MODS][4];
#pragma unroll
    for (int m = 0; m < M_MODS; ++m)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[m][i] = 0.0f;

    for (int k0 = 0; k0 < N; k0 += WK) {
        if constexpr (M_MODS > 1) {
            if (shared_c) {
                load_x_scaled(p.c[0], k0, s_irc[0][xr]);
#pragma unroll
                for (int m = 0; m < M_MODS; ++m) {
                    load_w(p.w[m], k0);
                    __syncthreads();
                    run_mma(acc[m]);
                    __syncthreads();
                }
            } else {
#pragma unroll
                for (int m = 0; m < M_MODS; ++m) {
                    load_x_scaled(p.c[m], k0, s_irc[m][xr]);
                    load_w(p.w[m], k0);
                    __syncthreads();
                    run_mma(acc[m]);
                    __syncthreads();
                }
            }
        } else {
            load_x_scaled(p.c[0], k0, s_irc[0][xr]);
            load_w(p.w[0], k0);
            __syncthreads();
            run_mma(acc[0]);
            __syncthreads();
        }
    }

    const int acc_r0 = lane >> 2;
    const int acc_r1 = acc_r0 + 8;
    const int acc_c  = col0 + wn * 8 + (lane & 3) * 2;
#pragma unroll
    for (int m = 0; m < M_MODS; ++m) {
        if (acc_r0 < rows) {
            const size_t roff = (size_t) (t0 + acc_r0) * 64u;
            float2 v0 = make_float2(acc[m][0], acc[m][1]);
            if (p.pre[m]) {
                const float2 pr0 = *reinterpret_cast<const float2*>(p.pre[m] + roff + acc_c);
                v0.x += pr0.x; v0.y += pr0.y;
            }
            *reinterpret_cast<float2*>(p.out[m] + roff + acc_c) = v0;
        }
        if (acc_r1 < rows) {
            const size_t roff = (size_t) (t0 + acc_r1) * 64u;
            float2 v1 = make_float2(acc[m][2], acc[m][3]);
            if (p.pre[m]) {
                const float2 pr1 = *reinterpret_cast<const float2*>(p.pre[m] + roff + acc_c);
                v1.x += pr1.x; v1.y += pr1.y;
            }
            *reinterpret_cast<float2*>(p.out[m] + roff + acc_c) = v1;
        }
    }
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    constexpr int BM = 16, BN = 32, WK = 128, LD = 136;
    __shared__ __align__(16) __half sx[BM * LD];
    __shared__ __align__(16) __half sw[BN * LD];
    __shared__ float s_irc[M_MODS][BM];
    const int tid = threadIdx.x, warp = tid >> 5;
    const int col0 = (int) blockIdx.x * BN;
    const int t0   = (int) blockIdx.y * BM;
    const int rows = min(BM, T - t0);
    if (rows <= 0) return;
    const int xr = tid >> 3, xc = (tid & 7) << 4;
    const int wr = tid >> 2, wc = (tid & 3) << 5;
    const bool x_ok = (xr < rows);
#pragma unroll
    for (int m = 0; m < M_MODS; ++m) {
        if (p.inv_c[m] != nullptr) {
            if (tid < BM) s_irc[m][tid] = (tid < rows) ? p.inv_c[m][t0 + tid] : 0.0f;
        } else {
            float ss = 0.0f;
            if (x_ok) {
                const float4* cp = reinterpret_cast<const float4*>(p.c[m] + (size_t) (t0 + xr) * N);
                const int n4 = N >> 2;
                for (int i = (tid & 7); i < n4; i += 8) {
                    const float4 v = cp[i];
                    ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
                }
            }
            ss += __shfl_xor_sync(0xffffffffu, ss, 4);
            ss += __shfl_xor_sync(0xffffffffu, ss, 2);
            ss += __shfl_xor_sync(0xffffffffu, ss, 1);
            if ((tid & 7) == 0) s_irc[m][xr] = x_ok ? rsqrtf(ss / (float) N + 1e-6f) : 0.0f;
        }
    }
    __syncthreads();
    // 4 warps: wn = warp & 1 (2 x 16-col tiles = 32 cols), wk = warp >> 1 (splits WK=128 into 2 x 64).
    const int wn = warp & 1, wk = warp >> 1;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[M_MODS];
#pragma unroll
    for (int m = 0; m < M_MODS; ++m) wmma::fill_fragment(acc[m], 0.0f);
    for (int k0 = 0; k0 < N; k0 += WK) {
#pragma unroll
        for (int m = 0; m < M_MODS; ++m) {
            __half2* dx = reinterpret_cast<__half2*>(sx + xr * LD + xc);
            if (x_ok && k0 + xc + 16 <= N) {
                const float sc = s_irc[m][xr];
                const float4* sp = reinterpret_cast<const float4*>(p.c[m] + (size_t) (t0 + xr) * N + k0 + xc);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float4 v = sp[i];
                    dx[i * 2 + 0] = __floats2half2_rn(v.x * sc, v.y * sc);
                    dx[i * 2 + 1] = __floats2half2_rn(v.z * sc, v.w * sc);
                }
            } else if (x_ok) {
                const float sc = s_irc[m][xr];
                const float* sp = p.c[m] + (size_t) (t0 + xr) * N + k0 + xc;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int k_i = k0 + xc + i * 2;
                    dx[i] = __floats2half2_rn((k_i + 0 < N) ? sp[i * 2 + 0] * sc : 0.0f,
                                              (k_i + 1 < N) ? sp[i * 2 + 1] * sc : 0.0f);
                }
            } else {
                const uint4 z = make_uint4(0, 0, 0, 0);
                reinterpret_cast<uint4*>(dx)[0] = z;
                reinterpret_cast<uint4*>(dx)[1] = z;
            }
            __half2* dw = reinterpret_cast<__half2*>(sw + wr * LD + wc);
            if (k0 + wc + 32 <= N) {
                const float4* wp = reinterpret_cast<const float4*>(p.w[m] + (size_t) (col0 + wr) * (2 * N) + k0 + wc);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float4 v = __ldg(wp + i);
                    dw[i * 2 + 0] = __floats2half2_rn(v.x, v.y);
                    dw[i * 2 + 1] = __floats2half2_rn(v.z, v.w);
                }
            } else {
                const float* wp = p.w[m] + (size_t) (col0 + wr) * (2 * N) + k0 + wc;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int k_i = k0 + wc + i * 2;
                    dw[i] = __floats2half2_rn((k_i + 0 < N) ? __ldg(wp + i * 2 + 0) : 0.0f,
                                              (k_i + 1 < N) ? __ldg(wp + i * 2 + 1) : 0.0f);
                }
            }
            __syncthreads();
#pragma unroll
            for (int kk = wk * 64; kk < (wk + 1) * 64; kk += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                wmma::load_matrix_sync(fa, sx + kk, LD);
                wmma::load_matrix_sync(fb, sw + (wn * 16) * LD + kk, LD);
                wmma::mma_sync(acc[m], fa, fb, acc[m]);
            }
            __syncthreads();
        }
    }
    float* s_out = reinterpret_cast<float*>(sw);  // 2 * 16 * 32 * 4 = 4,096 B <= 8,704 B
#pragma unroll
    for (int m = 0; m < M_MODS; ++m) {
        wmma::store_matrix_sync(s_out + wk * (BM * BN) + wn * 16, acc[m], BN, wmma::mem_row_major);
        __syncthreads();
        for (int idx = tid; idx < rows * BN; idx += 128) {
            const int r = idx / BN, c = idx % BN;
            const size_t goff = (size_t) (t0 + r) * 64u + (size_t) (col0 + c);
            p.out[m][goff] = (s_out[idx] + s_out[BM * BN + idx]) + (p.pre[m] ? p.pre[m][goff] : 0.0f);
        }
        __syncthreads();
    }
#else
    (void) p; (void) T; (void) N;
#endif
}

/// 11c. Wave 8.5 Cut A: Split-K High-Occupancy Phase-A Control-Down (BM=64, BN=64, WK=64, Block=256).
/// SMEM: sx[64*72] + sw[64*72] + s_irc[M_MODS][64] = 19,200 B (5 CTAs/SM!), pre-packed FP16 w_f16[m] (stride N, 0 F2FP),
/// outer-m loop keeps acc[4][4] at 16 regs -> REG <= 48, STACK: 0!
/// Launch: dim3((T + 63) / 64, split_k), Block: 256.
template <int M_MODS, bool HAS_INV_C = true>
__global__ void __launch_bounds__(256, 3) k_control_down_phaseA_pp_wmma_splitk(
        CtrlDownPhaseAPtrs3 p, int T, int N, int split_k) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int BM = 64, BN = 64, WK = 64, LD = 72;
    __shared__ __align__(64) __half smem_buf[(M_MODS == 3) ? (2 * 256 * 32) : ((BM + BN) * LD)];
    __shared__ float s_irc[M_MODS][BM];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int t0   = (int) blockIdx.x * BM;
    const int sk   = (int) blockIdx.y;
    const int rows = min(BM, T - t0);
    if (rows <= 0) return;

    const int k_blocks = (N + WK - 1) / WK;
    const int kb0 = (sk * k_blocks) / split_k;
    const int kb1 = ((sk + 1) * k_blocks) / split_k;
    const int k_start = kb0 * WK;
    const int k_end   = min(N, kb1 * WK);

    // 8 warps = 4x2: wm = 0..3 (16 rows/warp), wn = 0..1 (32 cols/warp = 4x m16n8k16).
    const int wm = warp >> 1;
    const int wn = warp & 1;

    auto ldmat4 = [](uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
    };
    auto mma16816 = [](float d[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    };

    // Hot SWA fast path (M_MODS == 3, 20/24 layers) AND M_MODS == 1 fast path (144 FFN calls!):
    // M_MODS == 3: Loads sx ONCE per k0=32 tile + overlaps 3x branchless cp.async.cg for w_f16[0..2].
    // M_MODS == 1: 2-stage ping-pong double-buffered cp.async.cg + sx32 (16,384 B <= 18,432 B) + double-buffered ldmatrix/MMA!
    if constexpr (M_MODS == 1 || M_MODS == 3) {
        const bool shared_c = (M_MODS == 1) ||
                              ((p.c[0] == p.c[1]) && (p.c[0] == p.c[2]) &&
                               (p.inv_c[0] == p.inv_c[1]) && (p.inv_c[0] == p.inv_c[2]));
        const bool has_wf16 = (M_MODS == 1)
                              ? (p.w_f16[0] != nullptr)
                              : ((p.w_f16[0] != nullptr) && (p.w_f16[1] != nullptr) && (p.w_f16[2] != nullptr));
        if (shared_c && has_wf16 && ((N & 31) == 0)) {
            if constexpr (HAS_INV_C) {
                if (tid < BM) s_irc[0][tid] = (tid < rows) ? p.inv_c[0][t0 + tid] : 0.0f;
            } else {
                if (p.inv_c[0] != nullptr) {
                    if (tid < BM) s_irc[0][tid] = (tid < rows) ? p.inv_c[0][t0 + tid] : 0.0f;
                } else {
                    const int rr = tid >> 2;
                    float ss = 0.0f;
                    if (rr < rows) {
                        const float4* cp = reinterpret_cast<const float4*>(p.c[0] + (size_t) (t0 + rr) * N);
                        const int n4 = N >> 2;
                        for (int i = (tid & 3); i < n4; i += 4) {
                            const float4 v = cp[i];
                            ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
                        }
                    }
                    ss += __shfl_xor_sync(0xffffffffu, ss, 2);
                    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
                    if ((tid & 3) == 0) s_irc[0][rr] = (rr < rows) ? rsqrtf(ss / (float) N + 1e-6f) : 0.0f;
                }
            }
            __syncthreads();

            __half* sx32 = smem_buf;                  // +0      (4,096 B: 64 x 32 XOR-swizzled, 64B-aligned)
            __half* sw32 = smem_buf + 64 * 32;        // +4,096  (M_MODS * 4,096 B, 64B-aligned)

            const int ld_r   = tid >> 2;              // 0..63
            const int ld_c8  = tid & 3;               // 0..3
            const int ld_k   = ld_c8 << 3;            // 0, 8, 16, 24
            const int ld_swz = ld_c8 ^ ((ld_r >> 1) & 3);
            const bool x_ok  = (ld_r < rows);
            const float sc_r = x_ok ? s_irc[0][ld_r] : 0.0f;
            const float* x_row_p = p.c[0] + (x_ok ? ((size_t) (t0 + ld_r) * (size_t) N + (size_t) ld_k) : 0u);
            const uint32_t sa_x0 = static_cast<uint32_t>(__cvta_generic_to_shared(sx32 + ld_r * 32 + (ld_swz << 3)));
            const size_t w_row_off = (size_t) ld_r * (size_t) N + (size_t) ld_k;

            const int a_row = wm * 16 + (lane & 15);
            const uint32_t a_swz0 = (uint32_t) ((lane >> 4) ^ ((a_row >> 1) & 3));
            const uint32_t sx_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(sx32 + a_row * 32)) + (a_swz0 << 4);

            const int b_row = wn * 32 + (((lane >> 4) << 3) | (lane & 7));
            const uint32_t b_swz0 = (uint32_t) (((lane >> 3) & 1) ^ ((b_row >> 1) & 3));
            const uint32_t sw_b0 = static_cast<uint32_t>(__cvta_generic_to_shared(sw32 + b_row * 32)) + (b_swz0 << 4);

            float acc[M_MODS][4][4];
#pragma unroll
            for (int m = 0; m < M_MODS; ++m)
#pragma unroll
                for (int ni = 0; ni < 4; ++ni)
#pragma unroll
                    for (int i = 0; i < 4; ++i) acc[m][ni][i] = 0.0f;

            auto stage_x = [&](int k0, uint32_t dst_sa) {
                if (x_ok) {
                    const float4* xp = reinterpret_cast<const float4*>(x_row_p + k0);
                    const float4 v0 = xp[0];
                    const float4 v1 = xp[1];
                    const __half2 h0 = __floats2half2_rn(v0.x * sc_r, v0.y * sc_r);
                    const __half2 h1 = __floats2half2_rn(v0.z * sc_r, v0.w * sc_r);
                    const __half2 h2 = __floats2half2_rn(v1.x * sc_r, v1.y * sc_r);
                    const __half2 h3 = __floats2half2_rn(v1.z * sc_r, v1.w * sc_r);
                    uint4 u;
                    memcpy(&u.x, &h0, 4); memcpy(&u.y, &h1, 4);
                    memcpy(&u.z, &h2, 4); memcpy(&u.w, &h3, 4);
                    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n"
                                 :: "r"(dst_sa), "r"(u.x), "r"(u.y), "r"(u.z), "r"(u.w) : "memory");
                } else {
                    asm volatile("st.shared.v4.u32 [%0], {0, 0, 0, 0};\n" :: "r"(dst_sa) : "memory");
                }
            };

            if constexpr (M_MODS == 1) {
                if (k_start < k_end) {
                    const size_t wk_off0 = w_row_off + (size_t) k_start;
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa_x0 + 4096u), "l"(p.w_f16[0] + wk_off0) : "memory");
                    asm volatile("cp.async.commit_group;\n" ::: "memory");
                    stage_x(k_start, sa_x0);
                    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
                    __syncthreads();
                }
                int stage = 0;
#pragma unroll 1
                for (int k0 = k_start; k0 < k_end; k0 += 32) {
                    const bool has_next = (k0 + 32 < k_end);
                    const uint32_t cur_off  = (uint32_t) stage * 8192u;
                    const uint32_t next_off = cur_off ^ 8192u;
                    if (has_next) {
                        const size_t wk_off_n = w_row_off + (size_t) (k0 + 32);
                        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                                     :: "r"(sa_x0 + 4096u + next_off), "l"(p.w_f16[0] + wk_off_n) : "memory");
                        asm volatile("cp.async.commit_group;\n" ::: "memory");
                    }
                    uint32_t a0_0, a0_1, a0_2, a0_3;
                    uint32_t a1_0, a1_1, a1_2, a1_3;
                    uint32_t b0_0, b0_1, b0_2, b0_3;
                    uint32_t b1_0, b1_1, b1_2, b1_3;
                    const uint32_t sx_k0 = sx_a0 + cur_off;
                    const uint32_t sw_k0 = sw_b0 + cur_off;
                    ldmat4(a0_0, a0_1, a0_2, a0_3, sx_k0);
                    ldmat4(b0_0, b0_1, b0_2, b0_3, sw_k0);
                    ldmat4(b1_0, b1_1, b1_2, b1_3, sw_k0 + 1024u);
                    ldmat4(a1_0, a1_1, a1_2, a1_3, sx_k0 ^ 32u);
                    mma16816(acc[0][0], a0_0, a0_1, a0_2, a0_3, b0_0, b0_1);
                    mma16816(acc[0][1], a0_0, a0_1, a0_2, a0_3, b0_2, b0_3);
                    ldmat4(b0_0, b0_1, b0_2, b0_3, sw_k0 ^ 32u);
                    mma16816(acc[0][2], a0_0, a0_1, a0_2, a0_3, b1_0, b1_1);
                    mma16816(acc[0][3], a0_0, a0_1, a0_2, a0_3, b1_2, b1_3);
                    ldmat4(b1_0, b1_1, b1_2, b1_3, (sw_k0 ^ 32u) + 1024u);
                    if (has_next) {
                        stage_x(k0 + 32, sa_x0 + next_off);
                    }
                    mma16816(acc[0][0], a1_0, a1_1, a1_2, a1_3, b0_0, b0_1);
                    mma16816(acc[0][1], a1_0, a1_1, a1_2, a1_3, b0_2, b0_3);
                    mma16816(acc[0][2], a1_0, a1_1, a1_2, a1_3, b1_0, b1_1);
                    mma16816(acc[0][3], a1_0, a1_1, a1_2, a1_3, b1_2, b1_3);
                    if (has_next) {
                        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
                        __syncthreads();
                        stage ^= 1;
                    }
                }
            } else {
                constexpr uint32_t kStageStride = 16384u;
                if (k_start < k_end) {
                    const size_t wk_off0 = w_row_off + (size_t) k_start;
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa_x0 + 4096u),  "l"(p.w_f16[0] + wk_off0) : "memory");
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa_x0 + 8192u),  "l"(p.w_f16[1] + wk_off0) : "memory");
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa_x0 + 12288u), "l"(p.w_f16[2] + wk_off0) : "memory");
                    asm volatile("cp.async.commit_group;\n" ::: "memory");
                    stage_x(k_start, sa_x0);
                    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
                    __syncthreads();
                }
                int stage = 0;
#pragma unroll 1
                for (int k0 = k_start; k0 < k_end; k0 += 32) {
                    const bool has_next = (k0 + 32 < k_end);
                    const uint32_t cur_off  = (uint32_t) stage * kStageStride;
                    const uint32_t next_off = cur_off ^ kStageStride;
                    if (has_next) {
                        const size_t wk_off_n = w_row_off + (size_t) (k0 + 32);
                        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa_x0 + next_off + 4096u),  "l"(p.w_f16[0] + wk_off_n) : "memory");
                        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa_x0 + next_off + 8192u),  "l"(p.w_f16[1] + wk_off_n) : "memory");
                        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa_x0 + next_off + 12288u), "l"(p.w_f16[2] + wk_off_n) : "memory");
                        asm volatile("cp.async.commit_group;\n" ::: "memory");
                    }

                    const uint32_t sx_k0 = sx_a0 + cur_off;
                    const uint32_t sw_k0 = sw_b0 + cur_off;
                    {
                        uint32_t a0, a1, a2, a3;
                        ldmat4(a0, a1, a2, a3, sx_k0);
#pragma unroll
                        for (int m = 0; m < 3; ++m) {
                            uint32_t b0, b1, b2, b3;
                            const uint32_t sw_m = sw_k0 + (uint32_t) m * 4096u;
                            ldmat4(b0, b1, b2, b3, sw_m);
                            mma16816(acc[m][0], a0, a1, a2, a3, b0, b1);
                            mma16816(acc[m][1], a0, a1, a2, a3, b2, b3);
                            ldmat4(b0, b1, b2, b3, sw_m + 1024u);
                            mma16816(acc[m][2], a0, a1, a2, a3, b0, b1);
                            mma16816(acc[m][3], a0, a1, a2, a3, b2, b3);
                        }
                    }
                    if (has_next) {
                        stage_x(k0 + 32, sa_x0 + next_off);
                    }
                    {
                        uint32_t a0, a1, a2, a3;
                        ldmat4(a0, a1, a2, a3, sx_k0 ^ 32u);
                        const uint32_t sw_k1 = sw_k0 ^ 32u;
#pragma unroll
                        for (int m = 0; m < 3; ++m) {
                            uint32_t b0, b1, b2, b3;
                            const uint32_t sw_m = sw_k1 + (uint32_t) m * 4096u;
                            ldmat4(b0, b1, b2, b3, sw_m);
                            mma16816(acc[m][0], a0, a1, a2, a3, b0, b1);
                            mma16816(acc[m][1], a0, a1, a2, a3, b2, b3);
                            ldmat4(b0, b1, b2, b3, sw_m + 1024u);
                            mma16816(acc[m][2], a0, a1, a2, a3, b0, b1);
                            mma16816(acc[m][3], a0, a1, a2, a3, b2, b3);
                        }
                    }
                    if (has_next) {
                        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
                        __syncthreads();
                        stage ^= 1;
                    }
                }
            }

            // Recompute epilogue coords after k0 loop so acc_r0/acc_r1/acc_c_base/part_plane/t0/rows stay out of loop regs!
            const int ep_warp = (int) threadIdx.x >> 5;
            const int ep_lane = (int) threadIdx.x & 31;
            const int ep_t0   = (int) blockIdx.x * BM;
            const int ep_rows = min(BM, T - ep_t0);
            const int acc_r0  = (ep_warp >> 1) * 16 + (ep_lane >> 2);
            const int acc_r1  = acc_r0 + 8;
            const int acc_c_base = (ep_warp & 1) * 32 + (ep_lane & 3) * 2;
            if (split_k > 1) {
                const size_t part_plane = (size_t) blockIdx.y * (size_t) T * 64u;
#pragma unroll
                for (int m = 0; m < M_MODS; ++m) {
                    float* dst_base = p.part_out[m] + part_plane;
                    if (acc_r0 < ep_rows) {
                        const size_t roff = (size_t) (ep_t0 + acc_r0) * 64u;
#pragma unroll
                        for (int ni = 0; ni < 4; ++ni) {
                            stg_f2(reinterpret_cast<float2*>(dst_base + roff + acc_c_base + ni * 8), acc[m][ni][0], acc[m][ni][1]);
                        }
                    }
                    if (acc_r1 < ep_rows) {
                        const size_t roff = (size_t) (ep_t0 + acc_r1) * 64u;
#pragma unroll
                        for (int ni = 0; ni < 4; ++ni) {
                            stg_f2(reinterpret_cast<float2*>(dst_base + roff + acc_c_base + ni * 8), acc[m][ni][2], acc[m][ni][3]);
                        }
                    }
                }
            } else {
#pragma unroll
                for (int m = 0; m < M_MODS; ++m) {
                    const float* pre_m = p.pre[m];
                    float* out_m = p.out[m];
                    if (acc_r0 < ep_rows) {
                        const size_t roff = (size_t) (ep_t0 + acc_r0) * 64u;
#pragma unroll
                        for (int ni = 0; ni < 4; ++ni) {
                            const int c = acc_c_base + ni * 8;
                            float v0 = acc[m][ni][0], v1 = acc[m][ni][1];
                            if (pre_m) {
                                const float2 pr = *reinterpret_cast<const float2*>(pre_m + roff + c);
                                v0 += pr.x; v1 += pr.y;
                            }
                            stg_f2(reinterpret_cast<float2*>(out_m + roff + c), v0, v1);
                        }
                    }
                    if (acc_r1 < ep_rows) {
                        const size_t roff = (size_t) (ep_t0 + acc_r1) * 64u;
#pragma unroll
                        for (int ni = 0; ni < 4; ++ni) {
                            const int c = acc_c_base + ni * 8;
                            float v0 = acc[m][ni][2], v1 = acc[m][ni][3];
                            if (pre_m) {
                                const float2 pr = *reinterpret_cast<const float2*>(pre_m + roff + c);
                                v0 += pr.x; v1 += pr.y;
                            }
                            stg_f2(reinterpret_cast<float2*>(out_m + roff + c), v0, v1);
                        }
                    }
                }
            }
            return;
        }
    }

    // General fallback path (M_MODS == 1, non-shared c[m], !w_f16, or N % 32 != 0): compact unroll-1 on m (0 SASS bloat!).
    const int acc_r0 = wm * 16 + (lane >> 2);
    const int acc_r1 = acc_r0 + 8;
    const int acc_c_base = wn * 32 + (lane & 3) * 2;
    const size_t part_plane = (size_t) sk * (size_t) T * 64u;
    __half* sx = smem_buf;
    __half* sw = smem_buf + BM * LD;

    // Load or compute inv_c (HAS_INV_C=true purges 1,777-insn inv_c==nullptr branch!).
#pragma unroll 1
    for (int m = 0; m < M_MODS; ++m) {
        if constexpr (HAS_INV_C) {
            if (tid < BM) s_irc[m][tid] = (tid < rows) ? p.inv_c[m][t0 + tid] : 0.0f;
        } else {
            if (p.inv_c[m] != nullptr) {
                if (tid < BM) s_irc[m][tid] = (tid < rows) ? p.inv_c[m][t0 + tid] : 0.0f;
            } else {
                const int rr = tid >> 2;
                float ss = 0.0f;
                if (rr < rows) {
                    const float4* cp = reinterpret_cast<const float4*>(p.c[m] + (size_t) (t0 + rr) * N);
                    const int n4 = N >> 2;
                    for (int i = (tid & 3); i < n4; i += 4) {
                        const float4 v = cp[i];
                        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
                    }
                }
                ss += __shfl_xor_sync(0xffffffffu, ss, 2);
                ss += __shfl_xor_sync(0xffffffffu, ss, 1);
                if ((tid & 3) == 0) s_irc[m][rr] = (rr < rows) ? rsqrtf(ss / (float) N + 1e-6f) : 0.0f;
            }
        }
    }
    __syncthreads();

    const int a_row = lane & 15, a_col = (lane >> 4) << 3;
    const int b_row = ((lane >> 4) << 3) | (lane & 7), b_col = ((lane >> 3) & 1) << 3;
    const uint32_t sx_a_smem  = static_cast<uint32_t>(__cvta_generic_to_shared(sx + (wm * 16 + a_row) * LD + a_col));
    const uint32_t sw_b0_smem = static_cast<uint32_t>(__cvta_generic_to_shared(sw + (wn * 32 + b_row) * LD + b_col));
    const uint32_t sw_b1_smem = static_cast<uint32_t>(__cvta_generic_to_shared(sw + (wn * 32 + 16 + b_row) * LD + b_col));

    // Staging coords:
    const int sr = tid >> 4;          // 0..15
    const int sc4 = tid & 15;         // 0..15
    const int sc = sc4 << 2;          // 0, 4, ..., 60
    const int w16_r = tid >> 3;       // 0..31
    const int w16_c = (tid & 7) << 3; // 0, 8, ..., 56
    const bool n_al8 = ((N & 7) == 0);
    const bool n_al4 = ((N & 3) == 0);

#pragma unroll 1
    for (int m = 0; m < M_MODS; ++m) {
        const float* x_base = p.c[m];
        const __half* wf16  = p.w_f16[m];
        const float* wf32   = p.w[m];
        float acc[4][4];
#pragma unroll
        for (int ni = 0; ni < 4; ++ni)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[ni][i] = 0.0f;

        for (int k0 = k_start; k0 < k_end; k0 += WK) {
            const bool k_full = (k0 + WK <= N);
            // Stage sx[64 x 64] scaled by s_irc[m][r]
#pragma unroll
            for (int pass = 0; pass < 4; ++pass) {
                const int r = sr + pass * 16;
                __half2* dst2 = reinterpret_cast<__half2*>(sx + r * LD + sc);
                if (r < rows) {
                    const float sc_r = s_irc[m][r];
                    if (k_full && n_al4) {
                        const float4 v = *reinterpret_cast<const float4*>(x_base + (size_t) (t0 + r) * N + k0 + sc);
                        dst2[0] = __floats2half2_rn(v.x * sc_r, v.y * sc_r);
                        dst2[1] = __floats2half2_rn(v.z * sc_r, v.w * sc_r);
                    } else {
                        const float* sp = x_base + (size_t) (t0 + r) * N + k0 + sc;
                        const float v0 = (k0 + sc + 0 < N) ? sp[0] * sc_r : 0.0f;
                        const float v1 = (k0 + sc + 1 < N) ? sp[1] * sc_r : 0.0f;
                        const float v2 = (k0 + sc + 2 < N) ? sp[2] * sc_r : 0.0f;
                        const float v3 = (k0 + sc + 3 < N) ? sp[3] * sc_r : 0.0f;
                        dst2[0] = __floats2half2_rn(v0, v1);
                        dst2[1] = __floats2half2_rn(v2, v3);
                    }
                } else {
                    *reinterpret_cast<uint2*>(dst2) = make_uint2(0, 0);
                }
            }

            // Stage sw[64 x 64]: direct 16B FP16 loads from w_f16[m] (stride N) if available, else FP32 w[m] (stride 2*N)
            if (wf16 != nullptr) {
#pragma unroll
                for (int pass = 0; pass < 2; ++pass) {
                    const int r = w16_r + pass * 32;
                    if (k_full && n_al8) {
                        *reinterpret_cast<uint4*>(sw + r * LD + w16_c) =
                            __ldg(reinterpret_cast<const uint4*>(wf16 + (size_t) r * N + k0 + w16_c));
                    } else {
                        __half* d = sw + r * LD + w16_c;
                        const __half* s = wf16 + (size_t) r * N + k0 + w16_c;
#pragma unroll 1
                        for (int i = 0; i < 8; ++i) {
                            d[i] = (k0 + w16_c + i < N) ? __ldg(s + i) : __float2half(0.0f);
                        }
                    }
                }
            } else {
#pragma unroll
                for (int pass = 0; pass < 4; ++pass) {
                    const int r = sr + pass * 16;
                    __half2* dst2 = reinterpret_cast<__half2*>(sw + r * LD + sc);
                    if (k_full && n_al4) {
                        const float4 v = __ldg(reinterpret_cast<const float4*>(wf32 + (size_t) r * (2 * N) + k0 + sc));
                        dst2[0] = __floats2half2_rn(v.x, v.y);
                        dst2[1] = __floats2half2_rn(v.z, v.w);
                    } else {
                        const float* wp = wf32 + (size_t) r * (2 * N) + k0 + sc;
                        const float v0 = (k0 + sc + 0 < N) ? __ldg(wp + 0) : 0.0f;
                        const float v1 = (k0 + sc + 1 < N) ? __ldg(wp + 1) : 0.0f;
                        const float v2 = (k0 + sc + 2 < N) ? __ldg(wp + 2) : 0.0f;
                        const float v3 = (k0 + sc + 3 < N) ? __ldg(wp + 3) : 0.0f;
                        dst2[0] = __floats2half2_rn(v0, v1);
                        dst2[1] = __floats2half2_rn(v2, v3);
                    }
                }
            }
            __syncthreads();

#pragma unroll
            for (int kk = 0; kk < WK; kk += 16) {
                uint32_t a0, a1, a2, a3, b0, b1, b2, b3;
                const uint32_t off_b = (uint32_t) kk * 2u;
                ldmat4(a0, a1, a2, a3, sx_a_smem + off_b);
                ldmat4(b0, b1, b2, b3, sw_b0_smem + off_b);
                mma16816(acc[0], a0, a1, a2, a3, b0, b1);
                mma16816(acc[1], a0, a1, a2, a3, b2, b3);
                ldmat4(b0, b1, b2, b3, sw_b1_smem + off_b);
                mma16816(acc[2], a0, a1, a2, a3, b0, b1);
                mma16816(acc[3], a0, a1, a2, a3, b2, b3);
            }
            __syncthreads();
        }

        if (split_k > 1) {
            float* dst_base = p.part_out[m] + part_plane;
            if (acc_r0 < rows) {
                const size_t roff = (size_t) (t0 + acc_r0) * 64u;
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    stg_f2(reinterpret_cast<float2*>(dst_base + roff + acc_c_base + ni * 8), acc[ni][0], acc[ni][1]);
                }
            }
            if (acc_r1 < rows) {
                const size_t roff = (size_t) (t0 + acc_r1) * 64u;
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    stg_f2(reinterpret_cast<float2*>(dst_base + roff + acc_c_base + ni * 8), acc[ni][2], acc[ni][3]);
                }
            }
        } else {
            const float* pre_m = p.pre[m];
            float* out_m = p.out[m];
            if (acc_r0 < rows) {
                const size_t roff = (size_t) (t0 + acc_r0) * 64u;
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    const int c = acc_c_base + ni * 8;
                    float v0 = acc[ni][0], v1 = acc[ni][1];
                    if (pre_m) {
                        const float2 pr = *reinterpret_cast<const float2*>(pre_m + roff + c);
                        v0 += pr.x; v1 += pr.y;
                    }
                    stg_f2(reinterpret_cast<float2*>(out_m + roff + c), v0, v1);
                }
            }
            if (acc_r1 < rows) {
                const size_t roff = (size_t) (t0 + acc_r1) * 64u;
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    const int c = acc_c_base + ni * 8;
                    float v0 = acc[ni][2], v1 = acc[ni][3];
                    if (pre_m) {
                        const float2 pr = *reinterpret_cast<const float2*>(pre_m + roff + c);
                        v0 += pr.x; v1 += pr.y;
                    }
                    stg_f2(reinterpret_cast<float2*>(out_m + roff + c), v0, v1);
                }
            }
        }
    }
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    constexpr int BM = 64, BN = 64, WK = 64, LD = 72;
    __shared__ __align__(16) __half smem[(BM + BN) * LD];  // 18,432 B (>= 64*64*4 = 16,384 B for s_out)
    __shared__ float s_irc[M_MODS][BM];
    __half* sx = smem;
    __half* sw = smem + BM * LD;

    const int tid = threadIdx.x, warp = tid >> 5;
    const int t0   = (int) blockIdx.x * BM;
    const int sk   = (int) blockIdx.y;
    const int rows = min(BM, T - t0);
    if (rows <= 0) return;

    const int k_blocks = (N + WK - 1) / WK;
    const int kb0 = (sk * k_blocks) / split_k;
    const int kb1 = ((sk + 1) * k_blocks) / split_k;
    const int k_start = kb0 * WK;
    const int k_end   = min(N, kb1 * WK);

#pragma unroll
    for (int m = 0; m < M_MODS; ++m) {
        if (p.inv_c[m] != nullptr) {
            if (tid < BM) s_irc[m][tid] = (tid < rows) ? p.inv_c[m][t0 + tid] : 0.0f;
        } else {
            const int rr = tid >> 2;
            float ss = 0.0f;
            if (rr < rows) {
                const float4* cp = reinterpret_cast<const float4*>(p.c[m] + (size_t) (t0 + rr) * N);
                const int n4 = N >> 2;
                for (int i = (tid & 3); i < n4; i += 4) {
                    const float4 v = cp[i];
                    ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
                }
            }
            ss += __shfl_xor_sync(0xffffffffu, ss, 2);
            ss += __shfl_xor_sync(0xffffffffu, ss, 1);
            if ((tid & 3) == 0) s_irc[m][rr] = (rr < rows) ? rsqrtf(ss / (float) N + 1e-6f) : 0.0f;
        }
    }
    __syncthreads();

    const int wm = warp >> 1, wn = warp & 1;
    const int sr = tid >> 4, sc = (tid & 15) << 2;
    const int w16_r = tid >> 3, w16_c = (tid & 7) << 3;
    const size_t part_plane = (size_t) sk * (size_t) T * 64u;

#pragma unroll
    for (int m = 0; m < M_MODS; ++m) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];
        wmma::fill_fragment(acc[0], 0.0f);
        wmma::fill_fragment(acc[1], 0.0f);
        for (int k0 = k_start; k0 < k_end; k0 += WK) {
#pragma unroll
            for (int pass = 0; pass < 4; ++pass) {
                const int r = sr + pass * 16;
                __half2* dx = reinterpret_cast<__half2*>(sx + r * LD + sc);
                if (r < rows) {
                    const float sc_r = s_irc[m][r];
                    const float* sp = p.c[m] + (size_t) (t0 + r) * N + k0 + sc;
                    dx[0] = __floats2half2_rn((k0 + sc + 0 < N) ? sp[0] * sc_r : 0.0f,
                                              (k0 + sc + 1 < N) ? sp[1] * sc_r : 0.0f);
                    dx[1] = __floats2half2_rn((k0 + sc + 2 < N) ? sp[2] * sc_r : 0.0f,
                                              (k0 + sc + 3 < N) ? sp[3] * sc_r : 0.0f);
                } else {
                    *reinterpret_cast<uint2*>(dx) = make_uint2(0, 0);
                }
            }
            if (p.w_f16[m] != nullptr) {
#pragma unroll
                for (int pass = 0; pass < 2; ++pass) {
                    const int r = w16_r + pass * 32;
                    __half* dw = sw + r * LD + w16_c;
                    const __half* s = p.w_f16[m] + (size_t) r * N + k0 + w16_c;
#pragma unroll
                    for (int i = 0; i < 8; ++i) dw[i] = (k0 + w16_c + i < N) ? __ldg(s + i) : __float2half(0.0f);
                }
            } else {
#pragma unroll
                for (int pass = 0; pass < 4; ++pass) {
                    const int r = sr + pass * 16;
                    __half2* dw = reinterpret_cast<__half2*>(sw + r * LD + sc);
                    const float* wp = p.w[m] + (size_t) r * (2 * N) + k0 + sc;
                    dw[0] = __floats2half2_rn((k0 + sc + 0 < N) ? __ldg(wp + 0) : 0.0f,
                                              (k0 + sc + 1 < N) ? __ldg(wp + 1) : 0.0f);
                    dw[1] = __floats2half2_rn((k0 + sc + 2 < N) ? __ldg(wp + 2) : 0.0f,
                                              (k0 + sc + 3 < N) ? __ldg(wp + 3) : 0.0f);
                }
            }
            __syncthreads();
#pragma unroll
            for (int kk = 0; kk < WK; kk += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb0, fb1;
                wmma::load_matrix_sync(fa, sx + (wm * 16) * LD + kk, LD);
                wmma::load_matrix_sync(fb0, sw + (wn * 32) * LD + kk, LD);
                wmma::load_matrix_sync(fb1, sw + (wn * 32 + 16) * LD + kk, LD);
                wmma::mma_sync(acc[0], fa, fb0, acc[0]);
                wmma::mma_sync(acc[1], fa, fb1, acc[1]);
            }
            __syncthreads();
        }
        float* s_out = reinterpret_cast<float*>(smem);
        wmma::store_matrix_sync(s_out + (wm * 16) * 64 + wn * 32, acc[0], 64, wmma::mem_row_major);
        wmma::store_matrix_sync(s_out + (wm * 16) * 64 + wn * 32 + 16, acc[1], 64, wmma::mem_row_major);
        __syncthreads();
        for (int idx = tid; idx < rows * 64; idx += 256) {
            const int r = idx >> 6, c = idx & 63;
            const size_t goff = (size_t) (t0 + r) * 64u + (size_t) c;
            if (split_k > 1) {
                p.part_out[m][part_plane + goff] = s_out[idx];
            } else {
                p.out[m][goff] = s_out[idx] + (p.pre[m] ? p.pre[m][goff] : 0.0f);
            }
        }
        __syncthreads();
    }
#else
    (void) p; (void) T; (void) N; (void) split_k;
#endif
}

/// 11d. Wave 8.5 Cut A: Deterministic Split-K Phase-A Reduction + Hoisted pre_out Add (0% dead warps, 4 tokens/CTA).
/// Launch: dim3((T * 64 + 255) / 256), Block: 256 (100% active threads: 4 tokens/CTA x 64 cols/token, 128B warp coalesced!).
template <int M_MODS>
__global__ void __launch_bounds__(256, 4) k_phaseA_splitk_reduce_add_pre(
        CtrlDownPhaseAPtrs3 p, int T, int split_k) {
    const int tid = (int) threadIdx.x;
    const int row = (int) blockIdx.x * 16 + (tid >> 4);
    if (row >= T) return;
    const int col4 = tid & 15;
    const size_t idx4 = (size_t) row * 16u + (size_t) col4;
    const size_t plane4 = (size_t) T * 16u;
    auto add4 = [](float4 a, float4 b) -> float4 {
        return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    };

#pragma unroll 1
    for (int m = 0; m < M_MODS; ++m) {
        const float4* __restrict__ part4 = reinterpret_cast<const float4*>(p.part_out[m]);
        float4 s;
        if (split_k == 4) {
            const float4 p0 = ldcs_f4(part4 + 0u * plane4 + idx4);
            const float4 p1 = ldcs_f4(part4 + 1u * plane4 + idx4);
            const float4 p2 = ldcs_f4(part4 + 2u * plane4 + idx4);
            const float4 p3 = ldcs_f4(part4 + 3u * plane4 + idx4);
            s = add4(add4(add4(p0, p1), p2), p3);
        } else if (split_k == 6) {
            const float4 p0 = ldcs_f4(part4 + 0u * plane4 + idx4);
            const float4 p1 = ldcs_f4(part4 + 1u * plane4 + idx4);
            const float4 p2 = ldcs_f4(part4 + 2u * plane4 + idx4);
            const float4 p3 = ldcs_f4(part4 + 3u * plane4 + idx4);
            const float4 p4 = ldcs_f4(part4 + 4u * plane4 + idx4);
            const float4 p5 = ldcs_f4(part4 + 5u * plane4 + idx4);
            s = add4(add4(add4(add4(add4(p0, p1), p2), p3), p4), p5);
        } else if (split_k == 8) {
            const float4 p0 = ldcs_f4(part4 + 0u * plane4 + idx4);
            const float4 p1 = ldcs_f4(part4 + 1u * plane4 + idx4);
            const float4 p2 = ldcs_f4(part4 + 2u * plane4 + idx4);
            const float4 p3 = ldcs_f4(part4 + 3u * plane4 + idx4);
            const float4 p4 = ldcs_f4(part4 + 4u * plane4 + idx4);
            const float4 p5 = ldcs_f4(part4 + 5u * plane4 + idx4);
            const float4 p6 = ldcs_f4(part4 + 6u * plane4 + idx4);
            const float4 p7 = ldcs_f4(part4 + 7u * plane4 + idx4);
            s = add4(add4(add4(add4(add4(add4(add4(p0, p1), p2), p3), p4), p5), p6), p7);
        } else if (split_k == 9) {
            const float4 p0 = ldcs_f4(part4 + 0u * plane4 + idx4);
            const float4 p1 = ldcs_f4(part4 + 1u * plane4 + idx4);
            const float4 p2 = ldcs_f4(part4 + 2u * plane4 + idx4);
            const float4 p3 = ldcs_f4(part4 + 3u * plane4 + idx4);
            const float4 p4 = ldcs_f4(part4 + 4u * plane4 + idx4);
            const float4 p5 = ldcs_f4(part4 + 5u * plane4 + idx4);
            const float4 p6 = ldcs_f4(part4 + 6u * plane4 + idx4);
            const float4 p7 = ldcs_f4(part4 + 7u * plane4 + idx4);
            const float4 p8 = ldcs_f4(part4 + 8u * plane4 + idx4);
            s = add4(add4(add4(add4(add4(add4(add4(add4(p0, p1), p2), p3), p4), p5), p6), p7), p8);
        } else if (split_k == 12) {
            const float4 p0  = ldcs_f4(part4 +  0u * plane4 + idx4);
            const float4 p1  = ldcs_f4(part4 +  1u * plane4 + idx4);
            const float4 p2  = ldcs_f4(part4 +  2u * plane4 + idx4);
            const float4 p3  = ldcs_f4(part4 +  3u * plane4 + idx4);
            const float4 p4  = ldcs_f4(part4 +  4u * plane4 + idx4);
            const float4 p5  = ldcs_f4(part4 +  5u * plane4 + idx4);
            const float4 p6  = ldcs_f4(part4 +  6u * plane4 + idx4);
            const float4 p7  = ldcs_f4(part4 +  7u * plane4 + idx4);
            const float4 p8  = ldcs_f4(part4 +  8u * plane4 + idx4);
            const float4 p9  = ldcs_f4(part4 +  9u * plane4 + idx4);
            const float4 p10 = ldcs_f4(part4 + 10u * plane4 + idx4);
            const float4 p11 = ldcs_f4(part4 + 11u * plane4 + idx4);
            s = add4(add4(add4(add4(add4(add4(add4(add4(add4(add4(add4(p0, p1), p2), p3), p4), p5), p6), p7), p8), p9), p10), p11);
        } else {
            s = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int sk = 0; sk < split_k; ++sk) {
                s = add4(s, ldcs_f4(part4 + (size_t) sk * plane4 + idx4));
            }
        }
        if (p.pre[m]) {
            s = add4(s, ldcs_f4(reinterpret_cast<const float4*>(p.pre[m]) + idx4));
        }
        reinterpret_cast<float4*>(p.out[m])[idx4] = s;
    }
}

/// 12. Vertical Layer-Boundary Fusion: Post-FFN shortconv(moe_raw[t-3..t]) + ffn_inp[t] -> optional cur_out
/// + optional ndst_out (RMSNorm, when DO_PUSH) + next-layer Pre-Attn mix_normed (NQ=1 SWA or NQ=3 Full)
/// + optional RMSNorm + FP16 output (when WRITE_F16) or optional inv_rms output (when !WRITE_F16).
/// Wave 8.5 Cut C-micro: __launch_bounds__(576, 2) (2 CTAs/SM on Full-Attn!), 1-BAR fused ir_out reduction, ldcs_f4 for h0!
/// Supports optional t_offset for L2 sub-chunking! Grid: (T_chunk), Block: 576 (N == 2304, STACK:0).
template <int NQ, bool WRITE_F16, bool DO_PUSH, int B_VAL = -1>
__global__ void __launch_bounds__(576, 2) k_postffn_shortconv_preattn_mix_pp(
        const float* __restrict__ moe_raw, const float* __restrict__ w_conv, const float* __restrict__ conv_hist,
        const float* __restrict__ ffn_inp,     // [T_total, N] residual add (optional if nullptr)
        float* __restrict__ cur_out,           // [T_total, N] FP32 cur output (optional)
        float* __restrict__ ndst_out,          // [T_total, N] FP32 RMSNorm(cur) push output (when DO_PUSH)
        NormMixPtrs ptrs, int B,               // B existing already-normed history blocks (excluding last = RMSNorm(cur))
        const float* __restrict__ q_mix,       // [NQ * N] next-layer pre-attn mix queries
        MixOuts4 out_f32,                      // [T_total, N] optional FP32 outputs (in_q / in_k / in_v)
        NormF16Out3 nf16,                      // RMSNorm weights & FP16 outputs (used when WRITE_F16)
        MixInvRms4 ir_out,                     // optional [T_total] inv_rms outputs (used when !WRITE_F16)
        int hist_offset, int t_offset, int T_chunk, int N, float eps, bool local_out = false) {
    __shared__ float s_dot[NQ][8][18];
    __shared__ float s_w[NQ][8];
    __shared__ float s_ss[18];
    __shared__ float s_inv_rms;

    const int t_loc = (int) blockIdx.x;
    if (t_loc >= T_chunk) return;
    const int t = t_offset + t_loc;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int c0 = tid << 2;

    // 1. Post-FFN shortconv(moe_raw[t-3..t]) + residual ffn_inp[t]
    const float4 x = reinterpret_cast<const float4*>(moe_raw + (size_t) t * N)[tid];
    float4 h0, h1, h2;
    if (t >= 3) {
        // Streaming ldcs_f4 on final tap h0 (t-3) so sliding window stops evicting L2-pinned in_ffn/in_q!
        h0 = ldcs_f4(reinterpret_cast<const float4*>(moe_raw + (size_t) (t - 3) * N) + tid);
        h1 = reinterpret_cast<const float4*>(moe_raw + (size_t) (t - 2) * N)[tid];
        h2 = reinterpret_cast<const float4*>(moe_raw + (size_t) (t - 1) * N)[tid];
    } else {
        const float* hc0 = conv_hist + (size_t) (hist_offset + c0 + 0) * 3;
        const float* hc1 = conv_hist + (size_t) (hist_offset + c0 + 1) * 3;
        const float* hc2 = conv_hist + (size_t) (hist_offset + c0 + 2) * 3;
        const float* hc3 = conv_hist + (size_t) (hist_offset + c0 + 3) * 3;
        if (t == 0) {
            h0 = make_float4(hc0[0], hc1[0], hc2[0], hc3[0]);
            h1 = make_float4(hc0[1], hc1[1], hc2[1], hc3[1]);
            h2 = make_float4(hc0[2], hc1[2], hc2[2], hc3[2]);
        } else if (t == 1) {
            h0 = make_float4(hc0[1], hc1[1], hc2[1], hc3[1]);
            h1 = make_float4(hc0[2], hc1[2], hc2[2], hc3[2]);
            h2 = reinterpret_cast<const float4*>(moe_raw)[tid];
        } else {
            h0 = make_float4(hc0[2], hc1[2], hc2[2], hc3[2]);
            h1 = reinterpret_cast<const float4*>(moe_raw)[tid];
            h2 = reinterpret_cast<const float4*>(moe_raw + (size_t) N)[tid];
        }
    }
    const float4 wc0 = __ldg(reinterpret_cast<const float4*>(w_conv + (size_t) (c0 + 0) * 4));
    const float4 wc1 = __ldg(reinterpret_cast<const float4*>(w_conv + (size_t) (c0 + 1) * 4));
    const float4 wc2 = __ldg(reinterpret_cast<const float4*>(w_conv + (size_t) (c0 + 2) * 4));
    const float4 wc3 = __ldg(reinterpret_cast<const float4*>(w_conv + (size_t) (c0 + 3) * 4));
    const float scale = rsqrtf(5.0f);
    float4 cur_v;
    cur_v.x = (x.x + (h0.x * wc0.x + h1.x * wc0.y + h2.x * wc0.z + x.x * wc0.w)) * scale;
    cur_v.y = (x.y + (h0.y * wc1.x + h1.y * wc1.y + h2.y * wc1.z + x.y * wc1.w)) * scale;
    cur_v.z = (x.z + (h0.z * wc2.x + h1.z * wc2.y + h2.z * wc2.z + x.z * wc2.w)) * scale;
    cur_v.w = (x.w + (h0.w * wc3.x + h1.w * wc3.y + h2.w * wc3.z + x.w * wc3.w)) * scale;
    if (ffn_inp) {
        const float4 r = ldcs_f4(reinterpret_cast<const float4*>(ffn_inp + (size_t) t * N) + tid);
        cur_v.x += r.x; cur_v.y += r.y; cur_v.z += r.z; cur_v.w += r.w;
    }
    if (cur_out) {
        stcs_f4(reinterpret_cast<float4*>(cur_out + (size_t) t * N) + tid, cur_v);
    }

    // 2. Compute sum-of-squares of cur_v + dot products for cur_v and B history blocks.
    const int b_cnt = (B_VAL >= 0) ? B_VAL : B;
    constexpr int CV_CAP = (B_VAL >= 0) ? ((B_VAL > 0) ? B_VAL : 1) : 7;
    float4 cv[CV_CAP];
    if constexpr (B_VAL > 0) {
#pragma unroll
        for (int d = 0; d < B_VAL; ++d) {
            cv[d] = ldcs_f4(reinterpret_cast<const float4*>(ptrs.p[d] + (size_t) t * N) + tid);
        }
    }

    {
        float ss_c = cur_v.x * cur_v.x + cur_v.y * cur_v.y + cur_v.z * cur_v.z + cur_v.w * cur_v.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss_c += __shfl_xor_sync(0xffffffffu, ss_c, o);
        if (lane == 0) s_ss[warp] = ss_c;
    }
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
        const float4 qv = __ldg(reinterpret_cast<const float4*>(q_mix + (size_t) q * N) + tid);
        float dot = cur_v.x * qv.x + cur_v.y * qv.y + cur_v.z * qv.z + cur_v.w * qv.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
        if (lane == 0) s_dot[q][b_cnt][warp] = dot;
    }

    if constexpr (B_VAL > 0) {
#pragma unroll
        for (int d = 0; d < B_VAL; ++d) {
            const float4 v = cv[d];
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                const float4 qv = __ldg(reinterpret_cast<const float4*>(q_mix + (size_t) q * N) + tid);
                float dot = v.x * qv.x + v.y * qv.y + v.z * qv.z + v.w * qv.w;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
                if (lane == 0) s_dot[q][d][warp] = dot;
            }
        }
    } else if constexpr (B_VAL < 0) {
#pragma unroll
        for (int d = 0; d < 7; ++d) {
            if (d < B) {
                const float4 v = ldcs_f4(reinterpret_cast<const float4*>(ptrs.p[d] + (size_t) t * N) + tid);
                cv[d] = v;
#pragma unroll
                for (int q = 0; q < NQ; ++q) {
                    const float4 qv = __ldg(reinterpret_cast<const float4*>(q_mix + (size_t) q * N) + tid);
                    float dot = v.x * qv.x + v.y * qv.y + v.z * qv.z + v.w * qv.w;
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
                    if (lane == 0) s_dot[q][d][warp] = dot;
                }
            }
        }
    }
    __syncthreads();

    const int D = b_cnt + 1;
    const float inv_sqrt_n = (N == 2304) ? (1.0f / 48.0f) : (1.0f / sqrtf((float) N));
    if (warp == 0) {
        if (lane < NQ * D) {
            const int q = lane / D, d = lane % D;
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < 18; ++w) s += s_dot[q][d][w];
            if (d == b_cnt) {
                float ss = 0.0f;
#pragma unroll
                for (int w = 0; w < 18; ++w) ss += s_ss[w];
                const float ir = rsqrtf(ss / (float) N + eps);
                if (q == 0) s_inv_rms = ir;
                s *= ir;
            }
            s_w[q][d] = s * inv_sqrt_n;
        }
        __syncwarp();
        if (lane < NQ) {
            const int q = lane;
            float m = -1e30f;
            for (int d = 0; d < D; ++d) m = fmaxf(m, s_w[q][d]);
            float sum = 0.0f;
            for (int d = 0; d < D; ++d) {
                s_w[q][d] = __expf(s_w[q][d] - m);
                sum += s_w[q][d];
            }
            const float inv = __fdividef(1.0f, sum);
            for (int d = 0; d < D; ++d) s_w[q][d] *= inv;
            s_w[q][b_cnt] *= s_inv_rms;
        }
    }
    __syncthreads();

    if constexpr (DO_PUSH) {
        if (ndst_out) {
            const float r_cur = s_inv_rms;
            stcs_f4(reinterpret_cast<float4*>(ndst_out + (size_t) t * N) + tid,
                    make_float4(cur_v.x * r_cur, cur_v.y * r_cur, cur_v.z * r_cur, cur_v.w * r_cur));
        }
    }

    if constexpr (WRITE_F16) {
        float ax[NQ], ay[NQ], az[NQ], aw[NQ];
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            const float wb = s_w[q][b_cnt];
            ax[q] = wb * cur_v.x; ay[q] = wb * cur_v.y; az[q] = wb * cur_v.z; aw[q] = wb * cur_v.w;
        }
        if constexpr (B_VAL > 0) {
#pragma unroll
            for (int d = 0; d < B_VAL; ++d) {
#pragma unroll
                for (int q = 0; q < NQ; ++q) {
                    const float wd = s_w[q][d];
                    ax[q] += wd * cv[d].x; ay[q] += wd * cv[d].y; az[q] += wd * cv[d].z; aw[q] += wd * cv[d].w;
                }
            }
        } else if constexpr (B_VAL < 0) {
#pragma unroll
            for (int d = 0; d < 7; ++d) {
                if (d < B) {
#pragma unroll
                    for (int q = 0; q < NQ; ++q) {
                        const float wd = s_w[q][d];
                        ax[q] += wd * cv[d].x; ay[q] += wd * cv[d].y; az[q] += wd * cv[d].z; aw[q] += wd * cv[d].w;
                    }
                }
            }
        }
        __shared__ float s_irq[NQ];
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            float ss = ax[q] * ax[q] + ay[q] * ay[q] + az[q] * az[q] + aw[q] * aw[q];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
            if (lane == 0) s_dot[q][0][warp] = ss;
        }
        __syncthreads();
        if (tid < NQ) {
            float tot = 0.0f;
#pragma unroll
            for (int w = 0; w < 18; ++w) tot += s_dot[tid][0][w];
            s_irq[tid] = rsqrtf(tot / (float) N + eps);
        }
        __syncthreads();
        auto store_f16 = [&](int q, const float* nw, float* y_dst, uint16_t* y16_dst) {
            const float ir = s_irq[q];
            float4 o;
            if (nw) {
                const float4 ww = __ldg(reinterpret_cast<const float4*>(nw) + tid);
                o = make_float4(ax[q] * ir * ww.x, ay[q] * ir * ww.y, az[q] * ir * ww.z, aw[q] * ir * ww.w);
            } else {
                o = make_float4(ax[q] * ir, ay[q] * ir, az[q] * ir, aw[q] * ir);
            }
            const int t_o = local_out ? t_loc : t;
            if (y_dst) reinterpret_cast<float4*>(y_dst + (size_t) t_o * N)[tid] = o;
            if (y16_dst) {
                const __half2 p0 = __halves2half2(__float2half_rn(o.x), __float2half_rn(o.y));
                const __half2 p1 = __halves2half2(__float2half_rn(o.z), __float2half_rn(o.w));
                uint2 u;
                memcpy(&u.x, &p0, 4);
                memcpy(&u.y, &p1, 4);
                uint2* dst_u2 = reinterpret_cast<uint2*>(y16_dst + (size_t) t * N) + tid;
                if (y_dst) stcs_u2(dst_u2, u);
                else *dst_u2 = u;
            }
        };
        store_f16(0, nf16.w0, out_f32.p[0], nf16.y16_0);
        if constexpr (NQ >= 2) store_f16(1, nf16.w1, out_f32.p[1], nf16.y16_1);
        if constexpr (NQ >= 3) store_f16(2, nf16.w2, out_f32.p[2], nf16.y16_2);
    } else {
        // Compute one q at a time (saves 8 live regs -> REG <= 56, 2 CTAs/SM!) and fuse all NQ ir_out reductions into 1 BAR!
        const int t_o = local_out ? t_loc : t;
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            const float wb = s_w[q][b_cnt];
            float ax = wb * cur_v.x, ay = wb * cur_v.y, az = wb * cur_v.z, aw = wb * cur_v.w;
            if constexpr (B_VAL > 0) {
#pragma unroll
                for (int d = 0; d < B_VAL; ++d) {
                    const float wd = s_w[q][d];
                    ax += wd * cv[d].x; ay += wd * cv[d].y; az += wd * cv[d].z; aw += wd * cv[d].w;
                }
            } else if constexpr (B_VAL < 0) {
#pragma unroll
                for (int d = 0; d < 7; ++d) {
                    if (d < B) {
                        const float wd = s_w[q][d];
                        ax += wd * cv[d].x; ay += wd * cv[d].y; az += wd * cv[d].z; aw += wd * cv[d].w;
                    }
                }
            }
            if (out_f32.p[q]) {
                reinterpret_cast<float4*>(out_f32.p[q] + (size_t) t_o * N)[tid] = make_float4(ax, ay, az, aw);
            }
            if (ir_out.p[q]) {
                float ss = ax * ax + ay * ay + az * az + aw * aw;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
                if (lane == 0) s_dot[q][0][warp] = ss;
            }
        }
        bool any_ir = (ir_out.p[0] != nullptr);
        if constexpr (NQ >= 2) any_ir = any_ir || (ir_out.p[1] != nullptr);
        if constexpr (NQ >= 3) any_ir = any_ir || (ir_out.p[2] != nullptr);
        if (any_ir) {
            __syncthreads();
            if (warp == 0 && lane < NQ) {
                float* dst_ir = (lane == 0) ? ir_out.p[0] : ((lane == 1) ? ir_out.p[1] : ir_out.p[2]);
                if (dst_ir != nullptr) {
                    float tot = 0.0f;
#pragma unroll
                    for (int w = 0; w < 18; ++w) tot += s_dot[lane][0][w];
                    dst_ir[t_o] = rsqrtf(tot / (float) N + eps);
                }
            }
        }
    }
}

/// 13a. Multi-CTA (16-CTA) Expert Grouping Stage 1: Parallel Histogram over ids[0..n_ent-1].
/// Grid: (n_cta) [e.g. 16], Block: 256. Writes part_cnt[b * 128 + x] (size n_cta * 128 int32s = 8 KB).
__global__ void __launch_bounds__(256) k_group_experts_hist16(
        const int32_t* __restrict__ ids, int n_ent, int n_exp, int32_t* __restrict__ part_cnt) {
    __shared__ int32_t sh_cnt[128];
    const int tid = threadIdx.x;
    const int b = (int) blockIdx.x, nb = (int) gridDim.x;
    if (tid < 128) sh_cnt[tid] = 0;
    __syncthreads();
    const int e_start = (int) (((int64_t) b * n_ent) / nb);
    const int e_end   = (int) (((int64_t) (b + 1) * n_ent) / nb);
    for (int e = e_start + tid; e < e_end; e += blockDim.x) {
        const int x = ids[e];
        if ((unsigned) x < (unsigned) n_exp) atomicAdd(&sh_cnt[x], 1);
    }
    __syncthreads();
    if (tid < 128) {
        part_cnt[b * 128 + tid] = sh_cnt[tid];
    }
}

/// 13b. Multi-CTA Expert Grouping Stage 2 (VRAM Resident): Prefix Scan & Per-CTA Scatter Offsets.
/// Grid: (1), Block: 128 (4 warps). Reads part_hist (preserved!) and writes part_off (-1 for inactive experts).
__global__ void __launch_bounds__(128) k_group_experts_prefix16(
        const int32_t* __restrict__ part_hist, int32_t* __restrict__ part_off, int n_cta,
        const int32_t* __restrict__ slot_of, const uint8_t* __restrict__ blobs, size_t stride,
        int n_exp, int n_slots,
        unsigned long long* __restrict__ grp_ptr, int32_t* __restrict__ grp_start,
        int32_t* __restrict__ n_groups, uint32_t* __restrict__ d_hits = nullptr) {
    __shared__ int32_t s_warp_ng[4];
    __shared__ int32_t s_warp_off[4];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

    const int sl = (tid < n_exp && n_slots > 0) ? slot_of[tid] : -1;
    int total_c = 0;
    if (tid < n_exp && sl >= 0) {
        for (int b = 0; b < n_cta; ++b) {
            const int c = part_hist[b * 128 + tid];
            part_off[b * 128 + tid] = total_c;  // local exclusive offset across CTAs
            total_c += c;
        }
    }
    if (d_hits && tid < n_exp && total_c > 0) {
        d_hits[tid] += (uint32_t) total_c;
    }

    const int has_val = (total_c > 0) ? 1 : 0;
    int ng_inc = has_val, off_inc = total_c;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const int n_up = __shfl_up_sync(0xffffffffu, ng_inc, d);
        const int o_up = __shfl_up_sync(0xffffffffu, off_inc, d);
        if (lane >= d) { ng_inc += n_up; off_inc += o_up; }
    }
    if (lane == 31) { s_warp_ng[warp] = ng_inc; s_warp_off[warp] = off_inc; }
    __syncthreads();
    if (tid == 0) {
        int ng_acc = 0, off_acc = 0;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const int wn = s_warp_ng[w], wo = s_warp_off[w];
            s_warp_ng[w] = ng_acc; s_warp_off[w] = off_acc;
            ng_acc += wn; off_acc += wo;
        }
        grp_start[ng_acc] = off_acc;
        *n_groups = ng_acc;
    }
    __syncthreads();

    if (tid < n_exp) {
        if (has_val) {
            const int ng_idx = s_warp_ng[warp] + ng_inc - 1;
            const int off_idx = s_warp_off[warp] + off_inc - total_c;
            grp_ptr[ng_idx] = (unsigned long long) (blobs + (size_t) sl * stride);
            grp_start[ng_idx] = off_idx;
            for (int b = 0; b < n_cta; ++b) {
                part_off[b * 128 + tid] += off_idx;
            }
        } else {
            for (int b = 0; b < n_cta; ++b) {
                part_off[b * 128 + tid] = -1;
            }
        }
    }
}

/// 13c. Multi-CTA Expert Grouping Stage 2 (PCIe-Staged Miss Map): Prefix Scan & Per-CTA Scatter Offsets.
/// Grid: (1), Block: 128 (4 warps). Reads part_hist (preserved!) and writes part_off (-1 for inactive experts).
__global__ void __launch_bounds__(128) k_group_experts_prefix16_miss_map(
        const int32_t* __restrict__ part_hist, int32_t* __restrict__ part_off, int n_cta,
        const uint8_t* __restrict__ stage_blobs, size_t stride,
        int n_exp, PpStageMap map,
        unsigned long long* __restrict__ grp_ptr, int32_t* __restrict__ grp_start,
        int32_t* __restrict__ n_groups) {
    __shared__ int32_t s_warp_ng[4];
    __shared__ int32_t s_warp_off[4];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

    const int sp = (tid < n_exp) ? (int) map.pos[tid] : -1;
    int total_c = 0;
    if (tid < n_exp && sp >= 0) {
        for (int b = 0; b < n_cta; ++b) {
            const int c = part_hist[b * 128 + tid];
            part_off[b * 128 + tid] = total_c;
            total_c += c;
        }
    }

    const int has_val = (total_c > 0) ? 1 : 0;
    int ng_inc = has_val, off_inc = total_c;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const int n_up = __shfl_up_sync(0xffffffffu, ng_inc, d);
        const int o_up = __shfl_up_sync(0xffffffffu, off_inc, d);
        if (lane >= d) { ng_inc += n_up; off_inc += o_up; }
    }
    if (lane == 31) { s_warp_ng[warp] = ng_inc; s_warp_off[warp] = off_inc; }
    __syncthreads();
    if (tid == 0) {
        int ng_acc = 0, off_acc = 0;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const int wn = s_warp_ng[w], wo = s_warp_off[w];
            s_warp_ng[w] = ng_acc; s_warp_off[w] = off_acc;
            ng_acc += wn; off_acc += wo;
        }
        grp_start[ng_acc] = off_acc;
        *n_groups = ng_acc;
    }
    __syncthreads();

    if (tid < n_exp) {
        if (has_val) {
            const int ng_idx = s_warp_ng[warp] + ng_inc - 1;
            const int off_idx = s_warp_off[warp] + off_inc - total_c;
            grp_ptr[ng_idx] = (unsigned long long) (stage_blobs + (size_t) sp * stride);
            grp_start[ng_idx] = off_idx;
            for (int b = 0; b < n_cta; ++b) {
                part_off[b * 128 + tid] += off_idx;
            }
        } else {
            for (int b = 0; b < n_cta; ++b) {
                part_off[b * 128 + tid] = -1;
            }
        }
    }
}

/// 13d. Multi-CTA (16-CTA) Expert Grouping Stage 3: Unified Parallel Scatter (SMEM atomics only, ZERO global atomics!).
/// Grid: (n_cta) [same as Stage 1], Block: 256. Works identically for both VRAM-resident and miss-map groupings!
__global__ void __launch_bounds__(256) k_group_experts_scatter16(
        const int32_t* __restrict__ ids, const int32_t* __restrict__ part_off,
        int n_ent, int n_exp, int32_t* __restrict__ ent_dst, int32_t* __restrict__ ent_tok, int top_k) {
    __shared__ int32_t sh_off[128];
    const int tid = threadIdx.x;
    const int b = (int) blockIdx.x, nb = (int) gridDim.x;
    if (tid < 128) sh_off[tid] = (tid < n_exp) ? part_off[b * 128 + tid] : -1;
    __syncthreads();
    const int e_start = (int) (((int64_t) b * n_ent) / nb);
    const int e_end   = (int) (((int64_t) (b + 1) * n_ent) / nb);
    for (int e = e_start + tid; e < e_end; e += blockDim.x) {
        const int x = ids[e];
        if ((unsigned) x < (unsigned) n_exp && sh_off[x] >= 0) {
            const int pos = atomicAdd(&sh_off[x], 1);
            ent_dst[pos] = e;
            ent_tok[pos] = e / top_k;
        }
    }
}

}  // namespace strata::inkling2
