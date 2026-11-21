// inkling2_dspark.cuh - Inkling-2-Mini DSpark speculative drafter (3-layer DFlare + recurrent Markov head).
// Included directly by inkling2_engine.cu inside namespace strata::inkling2.
#pragma once

namespace dspark_detail {

__device__ __forceinline__ float bf16_to_f32_d(uint16_t u) {
    return __uint_as_float((uint32_t) u << 16);
}

__device__ __forceinline__ float warp_sum_d(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max_d(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

/// Mix 4 target feature tensors via softmax(fuse_row[0..3]) and RMSNorm: grid (T), block 256.
__global__ void k_dspark_mix_features(const float* __restrict__ f0, const float* __restrict__ f1,
                                      const float* __restrict__ f2, const float* __restrict__ f3,
                                      const float* __restrict__ fuse_row, const float* __restrict__ norm_w,
                                      float* __restrict__ out, int N, float eps) {
    __shared__ float sw[4];
    __shared__ float ssum[8];
    const int t = blockIdx.x;
    if (threadIdx.x == 0) {
        float a[4] = {fuse_row[0], fuse_row[1], fuse_row[2], fuse_row[3]};
        float mx = fmaxf(fmaxf(a[0], a[1]), fmaxf(a[2], a[3]));
        float s = 0.0f;
        for (int k = 0; k < 4; ++k) { a[k] = expf(a[k] - mx); s += a[k]; }
        const float inv = 1.0f / s;
        for (int k = 0; k < 4; ++k) sw[k] = a[k] * inv;
    }
    __syncthreads();
    const float w0 = sw[0], w1 = sw[1], w2 = sw[2], w3 = sw[3];
    const size_t base = (size_t) t * N;
    float* dst = out + base;
    float acc = 0.0f;
    for (int c = threadIdx.x; c < N; c += blockDim.x) {
        const size_t i = base + c;
        const float x = w0 * f0[i] + w1 * f1[i] + w2 * f2[i] + w3 * f3[i];
        dst[c] = x;
        acc += x * x;
    }
    acc = warp_sum_d(acc);
    if ((threadIdx.x & 31) == 0) ssum[threadIdx.x >> 5] = acc;
    __syncthreads();
    float tot = 0.0f;
    if (threadIdx.x < 8) tot = ssum[threadIdx.x];
    if (threadIdx.x < 32) {
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, o);
        if (threadIdx.x == 0) ssum[0] = rsqrtf(tot / (float) N + eps);
    }
    __syncthreads();
    const float sc = ssum[0];
    for (int c = threadIdx.x; c < N; c += blockDim.x) {
        dst[c] = dst[c] * sc * norm_w[c];
    }
}

/// Fused per-head RMSNorm(k) + NeoX RoPE(k) + Q8 store for injected k_ctx, v_ctx.
/// Grid (n_head_kv, T), Block 64 (d = 0..63 pairs (d, d+64) for D=128).
__global__ void k_dspark_norm_rope_kv_store_q8(const float* __restrict__ K, const float* __restrict__ V,
                                               const float* __restrict__ k_norm_w,
                                               int8_t* __restrict__ kq, __half* __restrict__ ks,
                                               int8_t* __restrict__ vq, __half* __restrict__ vs,
                                               int n_head_kv, int pos0, int cap, int is_swa,
                                               float eps, float freq_base, int n_rot) {
    __shared__ float s_red[2];
    const int hkv = blockIdx.x, t = blockIdx.y;
    const int d = threadIdx.x;   // 0..63
    const int warp = d >> 5;
    const int kv_dim = n_head_kv * 128;
    const size_t src_off = (size_t) t * kv_dim + hkv * 128;
    const float k0_raw = K[src_off + d];
    const float k1_raw = K[src_off + d + 64];
    const float v0 = V[src_off + d];
    const float v1 = V[src_off + d + 64];

    float sq = warp_sum_d(k0_raw * k0_raw + k1_raw * k1_raw);
    if ((d & 31) == 0) s_red[warp] = sq;
    __syncthreads();
    const float rms = rsqrtf((s_red[0] + s_red[1]) / 128.0f + eps);
    float k0 = k0_raw * rms * k_norm_w[d];
    float k1 = k1_raw * rms * k_norm_w[d + 64];

    // NeoX RoPE on pairs (d, d + n_rot/2) when d < n_rot/2
    if (d * 2 < n_rot) {
        const float theta = (float) (pos0 + t) * powf(freq_base, -2.0f * (float) d / (float) n_rot);
        float sv, cv;
        __sincosf(theta, &sv, &cv);
        const float rk0 = k0 * cv - k1 * sv;
        const float rk1 = k0 * sv + k1 * cv;
        k0 = rk0;
        k1 = rk1;
    }

    float ak = warp_max_d(fmaxf(fabsf(k0), fabsf(k1)));
    float av = warp_max_d(fmaxf(fabsf(v0), fabsf(v1)));
    if ((d & 31) == 0) { s_red[warp] = ak; }
    __syncthreads();
    const float amax_k = fmaxf(s_red[0], s_red[1]);
    if ((d & 31) == 0) { s_red[warp] = av; }
    __syncthreads();
    const float amax_v = fmaxf(s_red[0], s_red[1]);

    const int pos = pos0 + t;
    const int slot = is_swa ? (pos % cap) : pos;
    const float dk = amax_k / 127.0f;
    const float dv = amax_v / 127.0f;
    const float idk = dk > 0.0f ? (127.0f / amax_k) : 0.0f;
    const float idv = dv > 0.0f ? (127.0f / amax_v) : 0.0f;

    const size_t dst_off = (size_t) slot * kv_dim + hkv * 128;
    kq[dst_off + d]      = (int8_t) __float2int_rn(k0 * idk);
    kq[dst_off + d + 64] = (int8_t) __float2int_rn(k1 * idk);
    vq[dst_off + d]      = (int8_t) __float2int_rn(v0 * idv);
    vq[dst_off + d + 64] = (int8_t) __float2int_rn(v1 * idv);
    if (d == 0) {
        ks[(size_t) slot * n_head_kv + hkv] = __float2half(dk);
        vs[(size_t) slot * n_head_kv + hkv] = __float2half(dv);
    }
}

/// Fused per-head RMSNorm + NeoX RoPE for Q (blocks 0..n_head-1) and K (blocks n_head..n_head+n_head_kv-1).
/// Reads *pos0_d dynamically from device memory so the DFlare block can be captured in a CUDA Graph.
/// Grid (n_head + n_head_kv, T), Block 64 (D = 128).
__global__ void k_dspark_qk_norm_rope(float* __restrict__ Q, float* __restrict__ K,
                                      const float* __restrict__ q_norm_w, const float* __restrict__ k_norm_w,
                                      int n_head, int n_head_kv, const int32_t* __restrict__ pos0_d,
                                      float eps, float freq_base, int n_rot) {
    __shared__ float s_sq[2];
    const int idx_h = blockIdx.x, t = blockIdx.y;
    const int d = threadIdx.x;
    const int warp = d >> 5;
    const bool is_k = (idx_h >= n_head);
    const int h = is_k ? (idx_h - n_head) : idx_h;
    const int nh = is_k ? n_head_kv : n_head;
    float* ptr = (is_k ? K : Q) + ((size_t) t * nh + h) * 128;
    const float* nw = is_k ? k_norm_w : q_norm_w;

    const float x0 = ptr[d];
    const float x1 = ptr[d + 64];
    float sq = warp_sum_d(x0 * x0 + x1 * x1);
    if ((d & 31) == 0) s_sq[warp] = sq;
    __syncthreads();
    const float rms = rsqrtf((s_sq[0] + s_sq[1]) / 128.0f + eps);
    float y0 = x0 * rms * nw[d];
    float y1 = x1 * rms * nw[d + 64];

    if (d * 2 < n_rot) {
        const float theta = (float) (pos0_d[0] + t) * powf(freq_base, -2.0f * (float) d / (float) n_rot);
        float sv, cv;
        __sincosf(theta, &sv, &cv);
        const float ry0 = y0 * cv - y1 * sv;
        const float ry1 = y0 * sv + y1 * cv;
        y0 = ry0;
        y1 = ry1;
    }
    ptr[d] = y0;
    ptr[d + 64] = y1;
}

/// Non-causal block-uniform attention over context Q8 KV [j0..P-1] + FP32 self KV [0..block_n-1].
/// Reads P = *pos_d dynamically from device memory for CUDA Graph capture.
/// Grid (n_head, block_n), Block 128 (4 warps).
__global__ void k_dspark_attn_q8(const float* __restrict__ Q,
                                 const int8_t* __restrict__ kq_ctx, const __half* __restrict__ ks_ctx,
                                 const int8_t* __restrict__ vq_ctx, const __half* __restrict__ vs_ctx,
                                 const float* __restrict__ K_self, const float* __restrict__ V_self,
                                 float* __restrict__ out,
                                 int n_head, int n_head_kv, int cap, int is_swa, int n_swa,
                                 const int32_t* __restrict__ pos_d, int block_n, float kq_scale) {
    __shared__ float sq[128];
    __shared__ float s_m[4];
    __shared__ float s_l[4];
    __shared__ float s_acc[4][128];

    const int h = blockIdx.x, t = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int hkv = h / (n_head / n_head_kv);
    const int kv_dim = n_head_kv * 128;
    const int P = pos_d[0];
    const float* qv = Q + ((size_t) t * n_head + h) * 128;
    sq[threadIdx.x] = qv[threadIdx.x];
    __syncthreads();

    const int j0 = is_swa ? max(0, P - n_swa + 1) : 0;
    float m = -1e30f;
    float l = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    // 1. Context KV entries j in [j0 .. P-1]
    for (int j = j0 + warp; j < P; j += 4) {
        const int slot = is_swa ? (j % cap) : j;
        const size_t k_off = (size_t) slot * kv_dim + hkv * 128;
        const int8_t* kj = kq_ctx + k_off;
        const int8_t* vj = vq_ctx + k_off;
        const float ksc = __half2float(ks_ctx[(size_t) slot * n_head_kv + hkv]);
        const float vsc = __half2float(vs_ctx[(size_t) slot * n_head_kv + hkv]);
        float dot = sq[lane] * (float) kj[lane]
                  + sq[32 + lane] * (float) kj[32 + lane]
                  + sq[64 + lane] * (float) kj[64 + lane]
                  + sq[96 + lane] * (float) kj[96 + lane];
        dot = warp_sum_d(dot) * ksc;
        const float sc = dot * kq_scale;
        const float m_new = fmaxf(m, sc);
        const float alpha = l > 0.0f ? expf(m - m_new) : 0.0f;
        const float p = expf(sc - m_new);
        l = l * alpha + p;
        const float pv = p * vsc;
        acc0 = acc0 * alpha + pv * (float) vj[lane];
        acc1 = acc1 * alpha + pv * (float) vj[32 + lane];
        acc2 = acc2 * alpha + pv * (float) vj[64 + lane];
        acc3 = acc3 * alpha + pv * (float) vj[96 + lane];
        m = m_new;
    }

    // 2. Self block KV entries s in [0 .. block_n - 1] (non-causal)
    for (int s = warp; s < block_n; s += 4) {
        const size_t s_off = ((size_t) s * n_head_kv + hkv) * 128;
        const float* ks = K_self + s_off;
        const float* vs = V_self + s_off;
        float dot = sq[lane] * ks[lane]
                  + sq[32 + lane] * ks[32 + lane]
                  + sq[64 + lane] * ks[64 + lane]
                  + sq[96 + lane] * ks[96 + lane];
        dot = warp_sum_d(dot);
        const float sc = dot * kq_scale;
        const float m_new = fmaxf(m, sc);
        const float alpha = l > 0.0f ? expf(m - m_new) : 0.0f;
        const float p = expf(sc - m_new);
        l = l * alpha + p;
        acc0 = acc0 * alpha + p * vs[lane];
        acc1 = acc1 * alpha + p * vs[32 + lane];
        acc2 = acc2 * alpha + p * vs[64 + lane];
        acc3 = acc3 * alpha + p * vs[96 + lane];
        m = m_new;
    }

    if (lane == 0) { s_m[warp] = m; s_l[warp] = l; }
    s_acc[warp][lane]      = acc0;
    s_acc[warp][32 + lane] = acc1;
    s_acc[warp][64 + lane] = acc2;
    s_acc[warp][96 + lane] = acc3;
    __syncthreads();

    float mg = -1e30f;
    for (int w = 0; w < 4; ++w) if (s_l[w] > 0.0f && s_m[w] > mg) mg = s_m[w];
    float lg = 0.0f, og = 0.0f;
    for (int w = 0; w < 4; ++w) {
        if (s_l[w] > 0.0f) {
            const float c = expf(s_m[w] - mg);
            lg += s_l[w] * c;
            og += s_acc[w][threadIdx.x] * c;
        }
    }
    out[((size_t) t * n_head + h) * 128 + threadIdx.x] = lg > 0.0f ? (og / lg) : 0.0f;
}

/// Fast SIMD 4x int8 -> 2x half2 decode for DSpark attention.
__device__ __forceinline__ void dspark_i8x4_to_h2x2(uint32_t packed_i8x4, __half2& out0, __half2& out1) {
    const uint32_t u = packed_i8x4 ^ 0x80808080u;
    const uint32_t lo = __byte_perm(u, 0x6400u, 0x5150u);
    const uint32_t hi = __byte_perm(u, 0x6400u, 0x5352u);
    const __half2 bias = __half2half2(__float2half(1152.0f));
    out0 = __hsub2(*reinterpret_cast<const __half2*>(&lo), bias);
    out1 = __hsub2(*reinterpret_cast<const __half2*>(&hi), bias);
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
__device__ __forceinline__ void dspark_i8x4_to_h2x2_fma(uint32_t packed_i8x4, __half2 sc2, __half2 neg_bias_sc2,
                                                        __half2& out0, __half2& out1) {
    const uint32_t u = packed_i8x4 ^ 0x80808080u;
    const uint32_t lo = __byte_perm(u, 0x6400u, 0x5150u);
    const uint32_t hi = __byte_perm(u, 0x6400u, 0x5352u);
    out0 = __hfma2(*reinterpret_cast<const __half2*>(&lo), sc2, neg_bias_sc2);
    out1 = __hfma2(*reinterpret_cast<const __half2*>(&hi), sc2, neg_bias_sc2);
}

__device__ __forceinline__ float dspark_fast_exp2f_hw(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
#endif

/// Shared-KV Split-KV non-causal attention for DSpark full-attention layer (GQA-3, block_n <= 8, D=128):
/// 128 threads (8 half-warps = 1 per block token) cooperatively stage BK=16 Q8 KV tokens into shared memory ONCE per split!
/// Uses SIMD i8x4_to_h2x2 decode, __half2 V accumulation, and log2 exp2f softmax. Split 0 also adds self-block [0..block_n-1].
__global__ void __launch_bounds__(128, 6) k_dspark_attn_q8_sharedkv_split(
        const float* __restrict__ Q,
        const int8_t* __restrict__ kq_ctx, const __half* __restrict__ ks_ctx,
        const int8_t* __restrict__ vq_ctx, const __half* __restrict__ vs_ctx,
        const float* __restrict__ K_self, const float* __restrict__ V_self,
        float* __restrict__ split_out,
        int n_head, int n_head_kv, int P, int block_n, float kq_scale, int split_n) {
    constexpr int BK = 16;
    __shared__ uint2 sK[BK][16];
    __shared__ uint2 sV[BK][16];
    __shared__ float sDK[BK];
    __shared__ float sDV[BK];

    const int hkv = blockIdx.x, sp = blockIdx.y;
    const int tid = threadIdx.x, hw = tid >> 4, hlane = tid & 15;
    const int h0 = 3 * hkv, kv_dim = n_head_kv * 128;
    const int base8 = hlane * 8;
    constexpr float kLog2e = 1.4426950408889634f;
    const float kq_l2 = kq_scale * kLog2e;

    const int chunk_sz = max(64, (P + split_n - 1) / split_n);
    const int c_beg = sp * chunk_sz;
    if (c_beg >= P) {
        if (hw < block_n && hlane == 0) {
            for (int g = 0; g < 3; ++g) {
                float* dst = split_out + (((size_t) hw * split_n + sp) * n_head + (h0 + g)) * 130;
                dst[128] = -1e30f;
                dst[129] = 0.0f;
            }
        }
        return;
    }
    const int c_end = min(P, c_beg + chunk_sz);

    __half2 q2[3][4];
    if (hw < block_n) {
#pragma unroll
        for (int g = 0; g < 3; ++g) {
            const float* qp = Q + ((size_t) hw * n_head + (h0 + g)) * 128 + base8;
            q2[g][0] = __float22half2_rn(*reinterpret_cast<const float2*>(qp + 0));
            q2[g][1] = __float22half2_rn(*reinterpret_cast<const float2*>(qp + 2));
            q2[g][2] = __float22half2_rn(*reinterpret_cast<const float2*>(qp + 4));
            q2[g][3] = __float22half2_rn(*reinterpret_cast<const float2*>(qp + 6));
        }
    }

    const __half2 z2 = __half2half2(__float2half(0.0f));
    float m[3] = {-1e30f, -1e30f, -1e30f};
    float l[3] = {0.0f, 0.0f, 0.0f};
    __half2 acc2[3][4];
#pragma unroll
    for (int g = 0; g < 3; ++g) {
#pragma unroll
        for (int i = 0; i < 4; ++i) acc2[g][i] = z2;
    }

    for (int jb = c_beg; jb < c_end; jb += BK) {
        const int nk = min(BK, c_end - jb);
        for (int r = hw; r < nk; r += 8) {
            const int j = jb + r;
            sK[r][hlane] = *reinterpret_cast<const uint2*>(kq_ctx + (size_t) j * kv_dim + hkv * 128 + base8);
            sV[r][hlane] = *reinterpret_cast<const uint2*>(vq_ctx + (size_t) j * kv_dim + hkv * 128 + base8);
            if (hlane == 0) {
                const size_t idx = (size_t) j * n_head_kv + hkv;
                sDK[r] = __half2float(__ldg(&ks_ctx[idx])) * kq_l2;
                sDV[r] = __half2float(__ldg(&vs_ctx[idx]));
            }
        }
        __syncthreads();

        if (hw < block_n) {
            for (int k = 0; k < nk; ++k) {
                const uint2 k8 = sK[k][hlane];
                const uint2 v8 = sV[k][hlane];
                const float dkkq2 = sDK[k], d_v = sDV[k];
                __half2 kh[4], vh[4];
                dspark_i8x4_to_h2x2(k8.x, kh[0], kh[1]);
                dspark_i8x4_to_h2x2(k8.y, kh[2], kh[3]);
                dspark_i8x4_to_h2x2(v8.x, vh[0], vh[1]);
                dspark_i8x4_to_h2x2(v8.y, vh[2], vh[3]);
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
                    const float sc = dots[g] * dkkq2;
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

    if (sp == 0 && hw < block_n) {
        for (int s = 0; s < block_n; ++s) {
            const size_t s_off = ((size_t) s * n_head_kv + hkv) * 128 + base8;
            const float* ks = K_self + s_off;
            const float* vs = V_self + s_off;
            __half2 kh[4], vh[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                kh[i] = __float22half2_rn(*reinterpret_cast<const float2*>(ks + 2 * i));
                vh[i] = __float22half2_rn(*reinterpret_cast<const float2*>(vs + 2 * i));
            }
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
                const float sc = dots[g] * kq_l2;
                const float m_new = fmaxf(m[g], sc);
                const float alpha = l[g] > 0.0f ? exp2f(m[g] - m_new) : 0.0f;
                const float p = exp2f(sc - m_new);
                const __half2 a2 = __half2half2(__float2half(alpha));
                const __half2 p2 = __half2half2(__float2half(p));
                l[g] = l[g] * alpha + p;
#pragma unroll
                for (int i = 0; i < 4; ++i) acc2[g][i] = __hfma2(acc2[g][i], a2, __hmul2(p2, vh[i]));
                m[g] = m_new;
            }
        }
    }

    if (hw < block_n) {
#pragma unroll
        for (int g = 0; g < 3; ++g) {
            const int h = h0 + g;
            float* dst = split_out + (((size_t) hw * split_n + sp) * n_head + h) * 130;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 f2 = __half22float2(acc2[g][i]);
                dst[base8 + 2 * i + 0] = f2.x;
                dst[base8 + 2 * i + 1] = f2.y;
            }
            if (hlane == 0) {
                dst[128] = m[g];
                dst[129] = l[g];
            }
        }
    }
}

/// Tensor-Core (WMMA 16x16x16) Split-KV non-causal attention for DSpark full-attention layer (GQA-3, block_n <= 8, D=128, sm_70+):
/// Packs up to 24 query rows (3 GQA heads x block_n) into a 32x128 FP16 WMMA tile and processes BK=64 KV keys
/// in 39,296 B SMEM (<= 40,960 B, 2 blk/SM): HFMA2 pre-scaled KV dequant, ex2.approx softmax, register-resident
/// WMMA O accumulators (zero per-tile s_tile bounce), lazy acc rescale, and direct FP16 self-block in sp == 0.
__global__ void __launch_bounds__(256, 2) k_dspark_attn_q8_wmma_split(
        const float* __restrict__ Q,
        const int8_t* __restrict__ kq_ctx, const __half* __restrict__ ks_ctx,
        const int8_t* __restrict__ vq_ctx, const __half* __restrict__ vs_ctx,
        const float* __restrict__ K_self, const float* __restrict__ V_self,
        float* __restrict__ split_out,
        int n_head, int n_head_kv, const int32_t* __restrict__ pos_d, int block_n, float kq_scale, int split_n) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    extern __shared__ __align__(16) char smem_raw[];
    // 39,296 B <= 40,960 B (2 blk/SM): sQ (8704) | sKV (17408) | sS (8192) | sP (4608) | scalars (384)
    __half* sQ = reinterpret_cast<__half*>(smem_raw);
    __half* sKV = sQ + 32 * 136;
    float* sS = reinterpret_cast<float*>(sKV + 64 * 136);
    __half* sP = reinterpret_cast<__half*>(sS + 32 * 64);
    float* s_m = reinterpret_cast<float*>(sP + 32 * 72);
    float* s_l = s_m + 32;
    float* s_alpha = s_l + 32;

    const int hkv = blockIdx.x, sp = blockIdx.y;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int h0 = 3 * hkv, kv_dim = n_head_kv * 128;
    const int n_rows = block_n * 3;
    const int P = pos_d[0];
    constexpr float kLog2e = 1.4426950408889634f;
    const float qscale_all = kq_scale * kLog2e;

    const int chunk_sz = max(64, (P + split_n - 1) / split_n);
    const int c_beg = sp * chunk_sz;
    if (c_beg >= P && sp > 0) {
        if (tid < n_rows) {
            float* dst = split_out + (((size_t) (tid / 3) * split_n + sp) * n_head + (h0 + (tid % 3))) * 130;
            dst[128] = -1e30f;
            dst[129] = 0.0f;
        }
        return;
    }
    const int c_end = min(P, c_beg + chunk_sz);

    {
        const int qr = tid >> 3, qc0 = (tid & 7) * 16;
        const bool active = (qr < n_rows);
        const float* qp = active ? (Q + ((size_t) (qr / 3) * n_head + (h0 + (qr % 3))) * 128 + qc0) : nullptr;
        if ((tid & 7) == 0) {
            s_m[qr] = -1e30f;
            s_l[qr] = 0.0f;
            s_alpha[qr] = 1.0f;
        }
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
        // Zero unused sP rows [n_rows..31] once before the loop.
        for (int r = n_rows + warp; r < 32; r += 8) {
            sP[r * 72 + lane]      = __float2half(0.0f);
            sP[r * 72 + lane + 32] = __float2half(0.0f);
        }
    }
    __syncthreads();

    int acc_r0, acc_r1, acc_c0, acc_c1;
    wmma_acc_layout_16x16(lane, acc_r0, acc_r1, acc_c0, acc_c1);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0, acc1;
    wmma::fill_fragment(acc0, 0.0f);
    wmma::fill_fragment(acc1, 0.0f);

    const int kv_r = tid >> 2, kv_c0 = (tid & 3) * 32;
    const __half2 neg1152 = __half2half2(__float2half(-1152.0f));

    for (int jb = c_beg; jb < c_end; jb += 64) {
        const bool is_full = (jb + 64 <= c_end);
        {
            const int j = jb + kv_r;
            const bool valid = is_full ? true : (j < c_end);
            __half* dst = sKV + kv_r * 136 + kv_c0;
            if (valid) {
                const __half2 dk2 = __half2half2(__ldg(&ks_ctx[(size_t) j * n_head_kv + hkv]));
                const __half2 kbias = __hmul2(neg1152, dk2);
                const int8_t* kp = kq_ctx + (size_t) j * kv_dim + hkv * 128 + kv_c0;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(kp + i * 16);
                    __half2 h2[8];
                    dspark_i8x4_to_h2x2_fma(raw.x, dk2, kbias, h2[0], h2[1]);
                    dspark_i8x4_to_h2x2_fma(raw.y, dk2, kbias, h2[2], h2[3]);
                    dspark_i8x4_to_h2x2_fma(raw.z, dk2, kbias, h2[4], h2[5]);
                    dspark_i8x4_to_h2x2_fma(raw.w, dk2, kbias, h2[6], h2[7]);
                    *reinterpret_cast<uint4*>(dst + i * 16)     = *reinterpret_cast<const uint4*>(h2);
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
            const int rt = warp >> 2, ct = warp & 3;
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
            const bool valid = is_full ? true : (j < c_end);
            __half* dst = sKV + kv_r * 136 + kv_c0;
            if (valid) {
                const __half2 dv2 = __half2half2(__ldg(&vs_ctx[(size_t) j * n_head_kv + hkv]));
                const __half2 vbias = __hmul2(neg1152, dv2);
                const int8_t* vp = vq_ctx + (size_t) j * kv_dim + hkv * 128 + kv_c0;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(vp + i * 16);
                    __half2 h2[8];
                    dspark_i8x4_to_h2x2_fma(raw.x, dv2, vbias, h2[0], h2[1]);
                    dspark_i8x4_to_h2x2_fma(raw.y, dv2, vbias, h2[2], h2[3]);
                    dspark_i8x4_to_h2x2_fma(raw.z, dv2, vbias, h2[4], h2[5]);
                    dspark_i8x4_to_h2x2_fma(raw.w, dv2, vbias, h2[6], h2[7]);
                    *reinterpret_cast<uint4*>(dst + i * 16)     = *reinterpret_cast<const uint4*>(h2);
                    *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                }
            } else {
                const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int i = 0; i < 4; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
            }

            const int col0 = lane, col1 = lane + 32, j_0 = jb + col0, j_1 = jb + col1;
            const bool v0 = is_full ? true : (j_0 < c_end);
            const bool v1 = is_full ? true : (j_1 < c_end);
#pragma unroll
            for (int rr = 0; rr < 3; ++rr) {
                const int r = warp * 3 + rr;
                if (r < n_rows) {
                    const float sc0 = v0 ? sS[r * 64 + col0] : -1e30f;
                    const float sc1 = v1 ? sS[r * 64 + col1] : -1e30f;
                    float rmax = fmaxf(sc0, sc1);
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1) rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffffu, rmax, o));
                    const float mo = s_m[r], lo = s_l[r], mn = fmaxf(mo, rmax);
                    const float al = (lo > 0.0f) ? ((mn == mo) ? 1.0f : dspark_fast_exp2f_hw(mo - mn)) : 0.0f;
                    const float p0 = v0 ? dspark_fast_exp2f_hw(sc0 - mn) : 0.0f;
                    const float p1 = v1 ? dspark_fast_exp2f_hw(sc1 - mn) : 0.0f;
                    float ps = p0 + p1;
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1) ps += __shfl_xor_sync(0xffffffffu, ps, o);
                    sP[r * 72 + col0] = __float2half(p0);
                    sP[r * 72 + col1] = __float2half(p1);
                    if (lane == 0) {
                        s_m[r] = mn;
                        s_l[r] = lo * al + ps;
                        s_alpha[r] = al;
                    }
                }
            }
        }
        __syncthreads();

        {
            const float a00 = (acc_r0 < n_rows) ? s_alpha[acc_r0] : 0.0f;
            const float a01 = (acc_r1 < n_rows) ? s_alpha[acc_r1] : 0.0f;
            if (__any_sync(0xffffffffu, a00 != 1.0f || a01 != 1.0f)) wmma_rescale_acc(acc0, a00, a01);
            if (n_rows > 16) {
                const float a10 = (16 + acc_r0 < n_rows) ? s_alpha[16 + acc_r0] : 0.0f;
                const float a11 = (16 + acc_r1 < n_rows) ? s_alpha[16 + acc_r1] : 0.0f;
                if (__any_sync(0xffffffffu, a10 != 1.0f || a11 != 1.0f)) wmma_rescale_acc(acc1, a10, a11);
            }
            const int cbase = warp * 16;
#pragma unroll
            for (int k = 0; k < 64; k += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa0, fa1;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
                wmma::load_matrix_sync(fa0, sP + k, 72);
                wmma::load_matrix_sync(fa1, sP + 16 * 72 + k, 72);
                wmma::load_matrix_sync(fb, sKV + k * 136 + cbase, 136);
                wmma::mma_sync(acc0, fa0, fb, acc0);
                wmma::mma_sync(acc1, fa1, fb, acc1);
            }
        }
        __syncthreads();
    }

    if (sp == 0) {
        if (tid < 128) {
            const int sr = tid >> 3, sc0 = (tid & 7) * 16;
            const float* kp = (sr < block_n) ? (K_self + ((size_t) sr * n_head_kv + hkv) * 128 + sc0) : nullptr;
            __half2* dst2 = reinterpret_cast<__half2*>(sKV + sr * 136 + sc0);
            if (kp) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float4 v = *reinterpret_cast<const float4*>(kp + i * 4);
                    dst2[i * 2 + 0] = __halves2half2(__float2half(v.x), __float2half(v.y));
                    dst2[i * 2 + 1] = __halves2half2(__float2half(v.z), __float2half(v.w));
                }
            } else {
                const __half2 z2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));
#pragma unroll
                for (int i = 0; i < 8; ++i) dst2[i] = z2;
            }
        }
        __syncthreads();

        if ((warp & 3) == 0) {
            const int rt = warp >> 2;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
            wmma::fill_fragment(fc, 0.0f);
#pragma unroll
            for (int k = 0; k < 128; k += 16) {
                wmma::load_matrix_sync(fa, sQ + (rt * 16) * 136 + k, 136);
                wmma::load_matrix_sync(fb, sKV + k, 136);
                wmma::mma_sync(fc, fa, fb, fc);
            }
            wmma::store_matrix_sync(sS + (rt * 16) * 64, fc, 64, wmma::mem_row_major);
        }
        __syncthreads();

        if (tid < 128) {
            const int sr = tid >> 3, sc0 = (tid & 7) * 16;
            const float* vp = (sr < block_n) ? (V_self + ((size_t) sr * n_head_kv + hkv) * 128 + sc0) : nullptr;
            __half2* dst2 = reinterpret_cast<__half2*>(sKV + sr * 136 + sc0);
            if (vp) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float4 v = *reinterpret_cast<const float4*>(vp + i * 4);
                    dst2[i * 2 + 0] = __halves2half2(__float2half(v.x), __float2half(v.y));
                    dst2[i * 2 + 1] = __halves2half2(__float2half(v.z), __float2half(v.w));
                }
            } else {
                const __half2 z2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));
#pragma unroll
                for (int i = 0; i < 8; ++i) dst2[i] = z2;
            }
        }
        {
            const bool v0 = (lane < block_n);
#pragma unroll
            for (int rr = 0; rr < 3; ++rr) {
                const int r = warp * 3 + rr;
                if (r < n_rows) {
                    const float sc0 = v0 ? sS[r * 64 + lane] : -1e30f;
                    float rmax = sc0;
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1) rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffffu, rmax, o));
                    const float mo = s_m[r], lo = s_l[r], mn = fmaxf(mo, rmax);
                    const float al = (lo > 0.0f) ? ((mn == mo) ? 1.0f : dspark_fast_exp2f_hw(mo - mn)) : 0.0f;
                    const float p0 = v0 ? dspark_fast_exp2f_hw(sc0 - mn) : 0.0f;
                    float ps = p0;
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1) ps += __shfl_xor_sync(0xffffffffu, ps, o);
                    if (lane < 16) sP[r * 72 + lane] = __float2half(p0);
                    if (lane == 0) {
                        s_m[r] = mn;
                        s_l[r] = lo * al + ps;
                        s_alpha[r] = al;
                    }
                }
            }
        }
        __syncthreads();

        {
            const float a00 = (acc_r0 < n_rows) ? s_alpha[acc_r0] : 0.0f;
            const float a01 = (acc_r1 < n_rows) ? s_alpha[acc_r1] : 0.0f;
            if (__any_sync(0xffffffffu, a00 != 1.0f || a01 != 1.0f)) wmma_rescale_acc(acc0, a00, a01);
            if (n_rows > 16) {
                const float a10 = (16 + acc_r0 < n_rows) ? s_alpha[16 + acc_r0] : 0.0f;
                const float a11 = (16 + acc_r1 < n_rows) ? s_alpha[16 + acc_r1] : 0.0f;
                if (__any_sync(0xffffffffu, a10 != 1.0f || a11 != 1.0f)) wmma_rescale_acc(acc1, a10, a11);
            }
            const int cbase = warp * 16;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa0, fa1;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
            wmma::load_matrix_sync(fa0, sP, 72);
            wmma::load_matrix_sync(fa1, sP + 16 * 72, 72);
            wmma::load_matrix_sync(fb, sKV + cbase, 136);
            wmma::mma_sync(acc0, fa0, fb, acc0);
            wmma::mma_sync(acc1, fa1, fb, acc1);
        }
        __syncthreads();
    }

    // Reuse sKV (17,408 B >= 32*128*4 = 16,384 B) for single epilogue store.
    float* s_out = reinterpret_cast<float*>(sKV);
    wmma::store_matrix_sync(s_out + warp * 16,            acc0, 128, wmma::mem_row_major);
    wmma::store_matrix_sync(s_out + 16 * 128 + warp * 16, acc1, 128, wmma::mem_row_major);
    __syncthreads();

    const int my_col = tid & 127, my_half = tid >> 7;
    const int row0 = my_half * 16;
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        const int rr = row0 + r;
        if (rr < n_rows) {
            float* dst = split_out + (((size_t) (rr / 3) * split_n + sp) * n_head + (h0 + (rr % 3))) * 130;
            dst[my_col] = s_out[rr * 128 + my_col];
            if (my_col == 0) {
                dst[128] = s_m[rr];
                dst[129] = s_l[rr];
            }
        }
    }
#endif
}

/// Combine Split-KV partials for DSpark: grid (n_head, block_n), block 128.
/// Reads P = *pos_d dynamically from device memory for CUDA Graph capture.
__global__ void k_dspark_split_combine(const float* __restrict__ split_out, float* __restrict__ out,
                                       int n_head, const int32_t* __restrict__ pos_d, int split_n) {
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const int P = pos_d[0];
    const int chunk_sz = max(64, (P + split_n - 1) / split_n);
    const int active_splits = max(1, min(split_n, (P + chunk_sz - 1) / chunk_sz));
    float mg = -1e30f;
    for (int s = 0; s < active_splits; ++s) {
        const float* src = split_out + (((size_t) t * split_n + s) * n_head + h) * 130;
        if (src[129] > 0.0f && src[128] > mg) mg = src[128];
    }
    float lg = 0.0f, og = 0.0f;
    for (int s = 0; s < active_splits; ++s) {
        const float* src = split_out + (((size_t) t * split_n + s) * n_head + h) * 130;
        if (src[129] > 0.0f) {
            const float c = exp2f(src[128] - mg);
            lg += src[129] * c;
            og += src[d] * c;
        }
    }
    out[((size_t) t * n_head + h) * 128 + d] = lg > 0.0f ? (og / lg) : 0.0f;
}

__global__ void k_dspark_broadcast_tail(float* __restrict__ dst, const float* __restrict__ row, int total_rows, int N) {
    const int t = blockIdx.y + 1;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < total_rows && c < N) {
        dst[(size_t) t * N + c] = row[c];
    }
}

__global__ void k_dspark_swiglu(const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

__global__ void k_dspark_add(float* __restrict__ dst, const float* __restrict__ a, const float* __restrict__ b, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = a[i] + b[i];
}

__global__ void k_dspark_add3(float* __restrict__ dst, const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ c, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = a[i] + b[i] + c[i];
}

__global__ void k_dspark_prep_joint(const float* __restrict__ state, const void* __restrict__ markov_w1, int w1_type,
                                    int prev_id, const float* __restrict__ hidden_i, const float* __restrict__ context,
                                    float* __restrict__ joint_inp, int rank, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = 2 * rank + 2 * N;
    if (i >= total) return;
    if (i < rank) {
        joint_inp[i] = state[i];
    } else if (i < 2 * rank) {
        const int c = i - rank;
        const size_t idx = (size_t) prev_id * rank + c;
        if (w1_type == 30) joint_inp[i] = bf16_to_f32_d(((const uint16_t*) markov_w1)[idx]);
        else if (w1_type == 1) joint_inp[i] = __half2float(((const __half*) markov_w1)[idx]);
        else joint_inp[i] = ((const float*) markov_w1)[idx];
    } else if (i < 2 * rank + N) {
        joint_inp[i] = hidden_i[i - 2 * rank];
    } else {
        joint_inp[i] = context[i - 2 * rank - N];
    }
}

__global__ void k_dspark_rnn_update(const float* __restrict__ joint_out, const float* __restrict__ joint_b,
                                    float* __restrict__ state, float* __restrict__ latent, int rank) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rank) return;
    const float g_raw = joint_out[i] + joint_b[i];
    const float c_raw = joint_out[rank + i] + joint_b[rank + i];
    const float l_raw = joint_out[2 * rank + i] + joint_b[2 * rank + i];
    const float gate = 1.0f / (1.0f + expf(-g_raw));
    const float cand = tanhf(c_raw);
    const float s = state[i];
    state[i] = cand + gate * (s - cand);
    latent[i] = tanhf(l_raw);
}

__global__ void k_dspark_copy_row(float* __restrict__ dst, const float* __restrict__ src, int row, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = src[(size_t) row * N + i];
}

__global__ void k_dspark_softmax_probs(const float* __restrict__ logits, const float* __restrict__ logits_buf_row,
                                       int V, float inv_temp, float max_val, float* __restrict__ tmp_sum) {
    // Wait: host-side sampling reuses Engine::sample_logits directly cleanly when probabilistic is active.
    (void) logits; (void) logits_buf_row; (void) V; (void) inv_temp; (void) max_val; (void) tmp_sum;
}

}  // namespace dspark_detail

struct DsparkHparams {
    int n_layer = 3;
    int block_size = 8;
    int n_embd = 2304;
    int n_ff = 9216;
    int n_head = 24;
    int n_head_kv = 8;
    int head_dim = 128;
    int n_rot = 128;
    int q_dim = 3072;
    int kv_dim = 1024;
    int n_swa = 1025;       // gguf sliding_window + 1
    int markov_rank = 256;
    int64_t vocab_padded = 201024;
    int64_t vocab_unpadded = 200058;
    float rms_eps = 1e-6f;
    float freq_base = 1e6f;
    float softcap = 30.0f;
    std::vector<uint8_t> swa_pattern;
    std::vector<int> target_layers = {5, 23, 41, 48};

    bool is_swa(int il) const {
        if ((size_t) il < swa_pattern.size()) return swa_pattern[(size_t) il] != 0;
        return il != 1;
    }
};

struct DsparkLayer {
    bool is_swa = true;
    float* dflare_norm = nullptr;
    float* attn_norm = nullptr;
    float* ffn_norm = nullptr;
    float* attn_q_norm = nullptr;
    float* attn_k_norm = nullptr;
    float* attn_k_ctx_norm = nullptr;

    DenseMat attn_k_ctx;
    DenseMat attn_v_ctx;
    DenseMat attn_q;
    DenseMat attn_k;
    DenseMat attn_v;
    DenseMat attn_output;
    DenseMat ffn_gate;
    DenseMat ffn_up;
    DenseMat ffn_down;
};

struct DsparkModel : GpuWeightStore {
    std::unique_ptr<strata::GgufModel> gguf;
    DsparkHparams hp;

    float* dspark_fuse = nullptr;       // [n_layer + 1, ntarget] = [4, 4]
    float* dspark_norm = nullptr;       // [n_embd]
    float* mask_embd = nullptr;         // [n_embd]
    float* output_norm = nullptr;       // [n_embd]
    void* markov_w1 = nullptr;          // [vocab_padded, markov_rank] on GPU
    int markov_w1_type = 30;
    DenseMat joint_proj;                // [2*rank + 2*N, 3*rank]
    float* joint_proj_b = nullptr;      // [3*rank]
    DenseMat hidden_out;                // [rank, N]
    float* hidden_out_b = nullptr;      // [N]
    std::vector<DsparkLayer> layers;

    ~DsparkModel() { free_store(); }

    const uint8_t* find_tensor(const char* name, const strata::TensorInfo** out = nullptr) const override {
        size_t shard = 0;
        const strata::TensorInfo* t = gguf->find(name, &shard);
        if (out) *out = t;
        return t ? gguf->shard(shard).tensor_data(*t) : nullptr;
    }
};

inline bool load_dspark_model(const std::string& path, const Hparams& target_hp, DsparkModel& dm, std::string& err) {
    const auto t0 = Clock::now();
    try {
        dm.gguf = std::make_unique<strata::GgufModel>(strata::gguf_split_paths(path));
    } catch (const std::exception& ex) {
        err = ex.what();
        return false;
    }
    const strata::GgufFile& meta = dm.gguf->meta();
    const auto* arch = meta.get("general.architecture");
    if (!arch || arch->s != "dflash") {
        err = "expected draft GGUF architecture dflash";
        return false;
    }
    auto get_i = [&](const char* k, int64_t def) -> int64_t {
        const auto* v = meta.get(k);
        return v && v->is_num() ? (int64_t) v->u : def;
    };
    auto get_f = [&](const char* k, float def) -> float {
        const auto* v = meta.get(k);
        return v ? (float) v->num() : def;
    };

    auto& hp = dm.hp;
    hp.n_layer = (int) get_i("dflash.block_count", 3);
    hp.block_size = (int) get_i("dflash.block_size", 8);
    hp.n_embd = (int) get_i("dflash.embedding_length", (int) target_hp.n_embd);
    hp.n_ff = (int) get_i("dflash.feed_forward_length", 9216);
    hp.n_head = (int) get_i("dflash.attention.head_count", 24);
    hp.n_head_kv = (int) get_i("dflash.attention.head_count_kv", 8);
    hp.head_dim = (int) get_i("dflash.attention.key_length", 128);
    hp.n_rot = (int) get_i("dflash.rope.dimension_count", hp.head_dim);
    hp.q_dim = hp.n_head * hp.head_dim;
    hp.kv_dim = hp.n_head_kv * hp.head_dim;
    hp.n_swa = (int) get_i("dflash.attention.sliding_window", 1024) + 1;   // anchor + window before it
    hp.markov_rank = (int) get_i("dspark.markov_rank", 256);
    hp.vocab_padded = get_i("dflash.vocab_size", target_hp.vocab_padded);
    hp.vocab_unpadded = get_i("dflash.vocab_size_unpadded", target_hp.vocab_unpadded);
    hp.rms_eps = get_f("dflash.attention.layer_norm_rms_epsilon", target_hp.rms_eps);
    hp.freq_base = get_f("dflash.rope.freq_base", 1e6f);
    hp.softcap = get_f("dflash.final_logit_softcapping", target_hp.softcap);

    if (hp.n_embd != target_hp.n_embd || hp.vocab_unpadded != target_hp.vocab_unpadded ||
        hp.head_dim != 128 || hp.block_size != 8) {
        err = "dspark hyperparameters incompatible with target model";
        return false;
    }

    if (const auto* v = meta.get("dflash.attention.sliding_window_pattern")) {
        hp.swa_pattern.resize(v->items.size());
        for (size_t i = 0; i < v->items.size(); ++i) hp.swa_pattern[i] = v->items[i].u ? 1 : 0;
    }
    if (const auto* v = meta.get("dflash.target_layer_ids")) {
        hp.target_layers.clear();
        for (const auto& item : v->items) hp.target_layers.push_back((int) item.num());
    }
    if (hp.target_layers.size() != 4) {
        err = "dspark expects 4 target_layer_ids";
        return false;
    }

    const int64_t N = hp.n_embd;
    const int64_t rank = hp.markov_rank;
    dm.dspark_fuse = dm.upload_f32("dspark_fuse.weight", (size_t) (hp.n_layer + 1) * 4);
    dm.dspark_norm = dm.upload_f32("dspark_norm.weight", (size_t) N);
    dm.mask_embd = dm.upload_f32("mask_embd.weight", (size_t) N);
    dm.output_norm = dm.upload_f32("output_norm.weight", (size_t) N);

    {
        const strata::TensorInfo* wt = nullptr;
        const uint8_t* raw = dm.find_tensor("markov_w1.weight", &wt);
        if (!raw || !wt || (wt->type != 0 && wt->type != 1 && wt->type != 30)) {
            err = "markov_w1.weight missing or unsupported type";
            return false;
        }
        dm.markov_w1_type = (int) wt->type;
        const size_t elem_sz = (wt->type == 0) ? sizeof(float) : sizeof(uint16_t);
        const size_t bytes = (size_t) wt->elements() * elem_sz;
        void* d = dm.gpu_alloc<uint8_t>(bytes);
        CK(cudaMemcpy(d, raw, bytes, cudaMemcpyHostToDevice));
        dm.markov_w1 = d;
    }

    dm.joint_proj = dm.upload_dense("joint_proj.weight", (size_t) (3 * rank), (size_t) (2 * rank + 2 * N));
    dm.joint_proj_b = dm.upload_f32("joint_proj.bias", (size_t) (3 * rank));
    dm.hidden_out = dm.upload_dense("dspark_hidden_out.weight", (size_t) N, (size_t) rank);
    dm.hidden_out_b = dm.upload_f32("dspark_hidden_out.bias", (size_t) N);

    size_t fr_vram = 0, tot_vram = 0;
    const bool pack_dspark_q8 = (cudaMemGetInfo(&fr_vram, &tot_vram) == cudaSuccess && tot_vram >= (20ull << 30));
    dm.layers.resize((size_t) hp.n_layer);
    char nm[128];
    for (int il = 0; il < hp.n_layer; ++il) {
        auto& L = dm.layers[(size_t) il];
        L.is_swa = hp.is_swa(il);
        auto tn = [&](const char* sfx) {
            std::snprintf(nm, sizeof nm, "blk.%d.%s", il, sfx);
            return nm;
        };
        L.dflare_norm = dm.upload_f32(tn("dflare_norm.weight"), (size_t) N);
        L.attn_norm = dm.upload_f32(tn("attn_norm.weight"), (size_t) N);
        L.ffn_norm = dm.upload_f32(tn("ffn_norm.weight"), (size_t) N);
        L.attn_q_norm = dm.upload_f32(tn("attn_q_norm.weight"), (size_t) hp.head_dim);
        L.attn_k_norm = dm.upload_f32(tn("attn_k_norm.weight"), (size_t) hp.head_dim);
        L.attn_k_ctx_norm = dm.upload_f32(tn("attn_k_ctx_norm.weight"), (size_t) hp.head_dim);

        L.attn_k_ctx = dm.upload_dense(tn("attn_k_ctx.weight"), (size_t) hp.kv_dim, (size_t) N, false, pack_dspark_q8);
        L.attn_v_ctx = dm.upload_dense(tn("attn_v_ctx.weight"), (size_t) hp.kv_dim, (size_t) N, false, pack_dspark_q8);
        L.attn_q = dm.upload_dense(tn("attn_q.weight"), (size_t) hp.q_dim, (size_t) N, false, pack_dspark_q8);
        L.attn_k = dm.upload_dense(tn("attn_k.weight"), (size_t) hp.kv_dim, (size_t) N, false, pack_dspark_q8);
        L.attn_v = dm.upload_dense(tn("attn_v.weight"), (size_t) hp.kv_dim, (size_t) N, false, pack_dspark_q8);
        L.attn_output = dm.upload_dense(tn("attn_output.weight"), (size_t) N, (size_t) hp.q_dim, false, pack_dspark_q8);
        L.ffn_gate = dm.upload_dense(tn("ffn_gate.weight"), (size_t) hp.n_ff, (size_t) N, false, pack_dspark_q8);
        L.ffn_up = dm.upload_dense(tn("ffn_up.weight"), (size_t) hp.n_ff, (size_t) N, false, pack_dspark_q8);
        L.ffn_down = dm.upload_dense(tn("ffn_down.weight"), (size_t) N, (size_t) hp.n_ff, false, pack_dspark_q8);
    }

    std::fprintf(stderr, "strata-inkling2: loaded dspark draft (%d layers, block=%d, rank=%d) in %.2f s\n",
                 hp.n_layer, hp.block_size, hp.markov_rank, ms_since(t0) / 1000.0);
    return true;
}

struct DsparkEngine {
    const DsparkModel* dm = nullptr;
    int ctx = 0;
    int tmax = 0;
    int pos = 0;

    // Context Q8 KV cache per layer
    std::vector<int> kv_cap;
    std::vector<int8_t*> kq, vq;
    std::vector<__half*> ks, vs;

    // Inject & draft scratch
    float* ctx_norm = nullptr;       // [tmax * N]
    float* ctx_k = nullptr;          // [tmax * kv_dim]
    float* ctx_v = nullptr;          // [tmax * kv_dim]
    float* f_rnn_buf = nullptr;      // [tmax * N]
    float* f_rnn_last = nullptr;     // [N]

    float* block_cur = nullptr;      // [block_size * N]
    float* x_norm = nullptr;         // [block_size * N]
    float* Q = nullptr;              // [block_size * q_dim]
    float* K = nullptr;              // [block_size * kv_dim]
    float* V = nullptr;              // [block_size * kv_dim]
    float* att = nullptr;            // [block_size * q_dim]
    float* split_out = nullptr;      // [block_size * split_full * n_head * 130]
    int split_full = 40;
    float* wo = nullptr;             // [block_size * N]
    float* gate = nullptr;           // [block_size * n_ff]
    float* up = nullptr;             // [block_size * n_ff]

    float* rnn_state = nullptr;      // [rank]
    float* joint_inp = nullptr;      // [2*rank + 2*N]
    float* joint_out = nullptr;      // [3*rank]
    float* rnn_latent = nullptr;     // [rank]
    float* rnn_corr = nullptr;       // [N]
    float* h_pred = nullptr;         // [N]
    int32_t* pos_d = nullptr;        // [1] device pos for CUDA Graph DFlare capture
    float* h_anchor_pin = nullptr;   // pinned [N] host buffer for async anchor upload
    cudaGraphExec_t dflare_graph_short = nullptr; // captured 3-layer DFlare block (pos < 256)
    cudaGraphExec_t dflare_graph_split = nullptr; // captured 3-layer DFlare block (pos >= 256, WMMA Split-KV)
    std::vector<float> h_anchor;     // [N]

    // Stats
    int64_t n_draft_calls = 0;
    int64_t n_drafted = 0;
    int64_t n_accepted = 0;
    double draft_ms = 0.0;
    double inject_ms = 0.0;

    ~DsparkEngine() {
        if (dflare_graph_short) cudaGraphExecDestroy(dflare_graph_short);
        if (dflare_graph_split) cudaGraphExecDestroy(dflare_graph_split);
        for (auto* p : kq) cudaFree(p);
        for (auto* p : vq) cudaFree(p);
        for (auto* p : ks) cudaFree(p);
        for (auto* p : vs) cudaFree(p);
        cudaFree(ctx_norm);
        cudaFree(ctx_k);
        cudaFree(ctx_v);
        cudaFree(f_rnn_buf);
        cudaFree(f_rnn_last);
        cudaFree(block_cur);
        cudaFree(x_norm);
        cudaFree(Q);
        cudaFree(K);
        cudaFree(V);
        cudaFree(att);
        cudaFree(split_out);
        cudaFree(wo);
        cudaFree(gate);
        cudaFree(up);
        cudaFree(rnn_state);
        cudaFree(joint_inp);
        cudaFree(joint_out);
        cudaFree(rnn_latent);
        cudaFree(rnn_corr);
        cudaFree(h_pred);
        cudaFree(pos_d);
        if (h_anchor_pin) cudaFreeHost(h_anchor_pin);
    }

    void init(const DsparkModel& model, int max_ctx, int max_t) {
        dm = &model;
        ctx = max_ctx;
        tmax = std::max(max_t, 16);
        const auto& hp = dm->hp;
        kq.resize((size_t) hp.n_layer, nullptr);
        vq.resize((size_t) hp.n_layer, nullptr);
        ks.resize((size_t) hp.n_layer, nullptr);
        vs.resize((size_t) hp.n_layer, nullptr);
        kv_cap.resize((size_t) hp.n_layer, 0);
        for (int il = 0; il < hp.n_layer; ++il) {
            const int cap = hp.is_swa(il) ? hp.n_swa : max_ctx;
            kv_cap[(size_t) il] = cap;
            CK(cudaMalloc(&kq[(size_t) il], (size_t) cap * hp.kv_dim * sizeof(int8_t)));
            CK(cudaMalloc(&vq[(size_t) il], (size_t) cap * hp.kv_dim * sizeof(int8_t)));
            CK(cudaMalloc(&ks[(size_t) il], (size_t) cap * hp.n_head_kv * sizeof(__half)));
            CK(cudaMalloc(&vs[(size_t) il], (size_t) cap * hp.n_head_kv * sizeof(__half)));
        }
        auto d_f32 = [](float** p, size_t n) { CK(cudaMalloc(p, n * sizeof(float))); };
        const int B = hp.block_size;
        d_f32(&ctx_norm, (size_t) tmax * hp.n_embd);
        d_f32(&ctx_k, (size_t) tmax * hp.kv_dim);
        d_f32(&ctx_v, (size_t) tmax * hp.kv_dim);
        d_f32(&f_rnn_buf, (size_t) tmax * hp.n_embd);
        d_f32(&f_rnn_last, (size_t) hp.n_embd);

        d_f32(&block_cur, (size_t) B * hp.n_embd);
        d_f32(&x_norm, (size_t) B * hp.n_embd);
        d_f32(&Q, (size_t) B * hp.q_dim);
        d_f32(&K, (size_t) B * hp.kv_dim);
        d_f32(&V, (size_t) B * hp.kv_dim);
        d_f32(&att, (size_t) B * hp.q_dim);
        int dev = 0, n_sm = 80;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, dev);
        split_full = std::clamp((n_sm * 24) / std::max(1, hp.n_head_kv), 40, 240);
        d_f32(&split_out, (size_t) B * split_full * hp.n_head * 130);
        cudaFuncSetAttribute(dspark_detail::k_dspark_attn_q8_wmma_split,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 40960);
        cudaFuncSetAttribute(dspark_detail::k_dspark_attn_q8_wmma_split,
                             cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        d_f32(&wo, (size_t) B * hp.n_embd);
        d_f32(&gate, (size_t) B * hp.n_ff);
        d_f32(&up, (size_t) B * hp.n_ff);

        d_f32(&rnn_state, (size_t) hp.markov_rank);
        d_f32(&joint_inp, (size_t) (2 * hp.markov_rank + 2 * hp.n_embd));
        d_f32(&joint_out, (size_t) (3 * hp.markov_rank));
        d_f32(&rnn_latent, (size_t) hp.markov_rank);
        d_f32(&rnn_corr, (size_t) hp.n_embd);
        d_f32(&h_pred, (size_t) hp.n_embd);
        CK(cudaMalloc(&pos_d, sizeof(int32_t)));
        if (cudaMallocHost(&h_anchor_pin, (size_t) hp.n_embd * sizeof(float)) != cudaSuccess) {
            (void) cudaGetLastError();
            h_anchor_pin = nullptr;
        }
        h_anchor.resize((size_t) hp.n_embd);
        pos = 0;
    }

    void reset(cudaStream_t st) {
        pos = 0;
        CK(cudaMemsetAsync(f_rnn_last, 0, (size_t) dm->hp.n_embd * sizeof(float), st));
    }

    size_t checkpoint_bytes() const {
        const auto& hp = dm->hp;
        size_t total = (size_t) hp.n_embd * sizeof(float);   // f_rnn_last
        for (int il = 0; il < hp.n_layer; ++il) {
            if (!hp.is_swa(il)) continue;
            const size_t cap = (size_t) kv_cap[(size_t) il];
            total += cap * (size_t) hp.kv_dim * 2 + cap * (size_t) hp.n_head_kv * sizeof(__half) * 2;
        }
        return total;
    }

    void save_checkpoint(uint8_t* dst, cudaStream_t st) const {
        const auto& hp = dm->hp;
        size_t off = 0;
        const size_t rnn_bytes = (size_t) hp.n_embd * sizeof(float);
        CK(cudaMemcpyAsync(dst + off, f_rnn_last, rnn_bytes, cudaMemcpyDeviceToHost, st));
        off += rnn_bytes;
        for (int il = 0; il < hp.n_layer; ++il) {
            if (!hp.is_swa(il)) continue;
            const size_t cap = (size_t) kv_cap[(size_t) il];
            const size_t qb = cap * (size_t) hp.kv_dim;
            const size_t sb = cap * (size_t) hp.n_head_kv * sizeof(__half);
            CK(cudaMemcpyAsync(dst + off, kq[(size_t) il], qb, cudaMemcpyDeviceToHost, st)); off += qb;
            CK(cudaMemcpyAsync(dst + off, vq[(size_t) il], qb, cudaMemcpyDeviceToHost, st)); off += qb;
            CK(cudaMemcpyAsync(dst + off, ks[(size_t) il], sb, cudaMemcpyDeviceToHost, st)); off += sb;
            CK(cudaMemcpyAsync(dst + off, vs[(size_t) il], sb, cudaMemcpyDeviceToHost, st)); off += sb;
        }
    }

    void restore_checkpoint(const uint8_t* src, int new_pos, cudaStream_t st) {
        const auto& hp = dm->hp;
        size_t off = 0;
        const size_t rnn_bytes = (size_t) hp.n_embd * sizeof(float);
        CK(cudaMemcpyAsync(f_rnn_last, src + off, rnn_bytes, cudaMemcpyHostToDevice, st));
        off += rnn_bytes;
        for (int il = 0; il < hp.n_layer; ++il) {
            if (!hp.is_swa(il)) continue;
            const size_t cap = (size_t) kv_cap[(size_t) il];
            const size_t qb = cap * (size_t) hp.kv_dim;
            const size_t sb = cap * (size_t) hp.n_head_kv * sizeof(__half);
            CK(cudaMemcpyAsync(kq[(size_t) il], src + off, qb, cudaMemcpyHostToDevice, st)); off += qb;
            CK(cudaMemcpyAsync(vq[(size_t) il], src + off, qb, cudaMemcpyHostToDevice, st)); off += qb;
            CK(cudaMemcpyAsync(ks[(size_t) il], src + off, sb, cudaMemcpyHostToDevice, st)); off += sb;
            CK(cudaMemcpyAsync(vs[(size_t) il], src + off, sb, cudaMemcpyHostToDevice, st)); off += sb;
        }
        pos = new_pos;
    }

    /// Inject `T` committed target feature vectors (from pos0 .. pos0+T-1) into DSpark KV + update `f_rnn_last`.
    template <typename EngineT>
    void inject(EngineT& E, int pos0, int T) {
        if (T <= 0) return;
        const auto t0 = Clock::now();
        const auto& hp = dm->hp;
        const float* f0 = E.d_tgt_feat[0];
        const float* f1 = E.d_tgt_feat[1];
        const float* f2 = E.d_tgt_feat[2];
        const float* f3 = E.d_tgt_feat[3];
        for (int il = 0; il < hp.n_layer; ++il) {
            const auto& L = dm->layers[(size_t) il];
            dspark_detail::k_dspark_mix_features<<<(unsigned) T, 256, 0, E.st>>>(
                f0, f1, f2, f3, dm->dspark_fuse + il * 4, L.dflare_norm, ctx_norm, hp.n_embd, hp.rms_eps);
            E.prep_q8(ctx_norm, E.xq, T, hp.n_embd);
            E.project_preq(ctx_norm, T <= 8 ? E.xq : nullptr, L.attn_k_ctx, ctx_k, T, hp.kv_dim, hp.n_embd);
            E.project_preq(ctx_norm, T <= 8 ? E.xq : nullptr, L.attn_v_ctx, ctx_v, T, hp.kv_dim, hp.n_embd);
            dspark_detail::k_dspark_norm_rope_kv_store_q8<<<dim3((unsigned) hp.n_head_kv, (unsigned) T), 64, 0, E.st>>>(
                ctx_k, ctx_v, L.attn_k_ctx_norm,
                kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il],
                hp.n_head_kv, pos0, kv_cap[(size_t) il], hp.is_swa(il) ? 1 : 0,
                hp.rms_eps, hp.freq_base, hp.n_rot);
        }
        dspark_detail::k_dspark_mix_features<<<(unsigned) T, 256, 0, E.st>>>(
            f0, f1, f2, f3, dm->dspark_fuse + hp.n_layer * 4, dm->dspark_norm, f_rnn_buf, hp.n_embd, hp.rms_eps);
        CK(cudaMemcpyAsync(f_rnn_last, f_rnn_buf + (size_t) (T - 1) * hp.n_embd,
                           (size_t) hp.n_embd * sizeof(float), cudaMemcpyDeviceToDevice, E.st));
        pos = pos0 + T;
        if (E.prof) {
            CK(cudaStreamSynchronize(E.st));
            inject_ms += ms_since(t0);
        }
    }

    /// Draft `n_draft` (1..7) speculative tokens from `anchor` (placed at `pos`, seeing context `0..pos-1`).
    /// When `out_logits != nullptr` (`n_draft` rows of `vocab_unpadded`), uses `sample_fn(row_vec)` for each draft step.
    template <typename EngineT, typename SampleFn>
    void draft(EngineT& E, int32_t anchor, int n_draft, std::vector<int32_t>& out_tokens,
               std::vector<std::vector<float>>* out_logits, SampleFn&& sample_fn) {
        const auto t0 = Clock::now();
        const auto& hp = dm->hp;
        const int block_n = hp.block_size;
        const int N = hp.n_embd;
        const int rank = hp.markov_rank;

        out_tokens.resize((size_t) n_draft);
        if (out_logits) out_logits->resize((size_t) n_draft);

        // 1. Anchor embedding (with target tok_embd_norm) + device pos upload
        float* anchor_src = h_anchor_pin ? h_anchor_pin : h_anchor.data();
        E.m->dequant_tok(anchor, anchor_src);
        CK(cudaMemcpyAsync(x_norm, anchor_src, (size_t) N * sizeof(float), cudaMemcpyHostToDevice, E.st));
        CK(cudaMemcpyAsync(pos_d, &pos, sizeof(int32_t), cudaMemcpyHostToDevice, E.st));

        // 2. 3 DFlare transformer layers over block_cur [block_n, N] + output_norm + rnn_state zeroing (CUDA Graph captured)
        const float kq_scale = 1.0f / std::sqrt((float) hp.head_dim);
        const bool use_split = (pos >= 256 && hp.n_head == 3 * hp.n_head_kv && hp.head_dim == 128 && block_n <= 8);
        auto emit_dflare = [&]() {
            k_rms_norm<<<1, kThreads, 0, E.st>>>(x_norm, E.m->tok_embd_norm, block_cur, N, E.m->hp.rms_eps);
            dspark_detail::k_dspark_broadcast_tail<<<dim3((unsigned) ((N + 255) / 256), (unsigned) (block_n - 1)), 256, 0, E.st>>>(
                block_cur, dm->mask_embd, block_n, N);
            for (int il = 0; il < hp.n_layer; ++il) {
                const auto& L = dm->layers[(size_t) il];
                k_rms_norm<<<(unsigned) block_n, kThreads, 0, E.st>>>(block_cur, L.attn_norm, x_norm, N, hp.rms_eps);
                E.prep_q8(x_norm, E.xq, block_n, N);
                E.project_preq(x_norm, E.xq, L.attn_q, Q, block_n, hp.q_dim, N);
                E.project_preq(x_norm, E.xq, L.attn_k, K, block_n, hp.kv_dim, N);
                E.project_preq(x_norm, E.xq, L.attn_v, V, block_n, hp.kv_dim, N);
                dspark_detail::k_dspark_qk_norm_rope<<<dim3((unsigned) (hp.n_head + hp.n_head_kv), (unsigned) block_n), 64, 0, E.st>>>(
                    Q, K, L.attn_q_norm, L.attn_k_norm, hp.n_head, hp.n_head_kv, pos_d, hp.rms_eps, hp.freq_base, hp.n_rot);
                if (!L.is_swa && use_split) {
                    dspark_detail::k_dspark_attn_q8_wmma_split<<<dim3((unsigned) hp.n_head_kv, (unsigned) split_full), 256, 40960, E.st>>>(
                        Q, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il], K, V, split_out,
                        hp.n_head, hp.n_head_kv, pos_d, block_n, kq_scale, split_full);
                    dspark_detail::k_dspark_split_combine<<<dim3((unsigned) hp.n_head, (unsigned) block_n), 128, 0, E.st>>>(
                        split_out, att, hp.n_head, pos_d, split_full);
                } else {
                    dspark_detail::k_dspark_attn_q8<<<dim3((unsigned) hp.n_head, (unsigned) block_n), 128, 0, E.st>>>(
                        Q, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il], K, V, att,
                        hp.n_head, hp.n_head_kv, kv_cap[(size_t) il], hp.is_swa(il) ? 1 : 0, hp.n_swa, pos_d, block_n, kq_scale);
                }
                E.project(att, L.attn_output, wo, block_n, N, hp.q_dim);
                dspark_detail::k_dspark_add<<<(unsigned) (((int64_t) block_n * N + 255) / 256), 256, 0, E.st>>>(
                    block_cur, block_cur, wo, (int64_t) block_n * N);

                k_rms_norm<<<(unsigned) block_n, kThreads, 0, E.st>>>(block_cur, L.ffn_norm, x_norm, N, hp.rms_eps);
                E.prep_q8(x_norm, E.xq, block_n, N);
                E.project_preq(x_norm, E.xq, L.ffn_gate, gate, block_n, hp.n_ff, N);
                E.project_preq(x_norm, E.xq, L.ffn_up, up, block_n, hp.n_ff, N);
                dspark_detail::k_dspark_swiglu<<<(unsigned) (((int64_t) block_n * hp.n_ff + 255) / 256), 256, 0, E.st>>>(
                    gate, up, gate, (int64_t) block_n * hp.n_ff);
                E.project(gate, L.ffn_down, wo, block_n, N, hp.n_ff);
                dspark_detail::k_dspark_add<<<(unsigned) (((int64_t) block_n * N + 255) / 256), 256, 0, E.st>>>(
                    block_cur, block_cur, wo, (int64_t) block_n * N);
            }
            k_rms_norm<<<(unsigned) block_n, kThreads, 0, E.st>>>(block_cur, dm->output_norm, block_cur, N, hp.rms_eps);
            CK(cudaMemsetAsync(rnn_state, 0, (size_t) rank * sizeof(float), E.st));
        };
        if (!E.prof) {
            cudaGraphExec_t& dg = use_split ? dflare_graph_split : dflare_graph_short;
            if (!dg) {
                cudaGraph_t g = nullptr;
                CK(cudaStreamBeginCapture(E.st, cudaStreamCaptureModeThreadLocal));
                emit_dflare();
                CK(cudaStreamEndCapture(E.st, &g));
                CK(cudaGraphInstantiate(&dg, g, nullptr, nullptr, 0));
                CK(cudaGraphDestroy(g));
            }
            CK(cudaGraphLaunch(dg, E.st));
        } else {
            emit_dflare();
        }

        // 3. Recurrent Markov chain for i = 1 .. n_draft
        int32_t prev_id = anchor;
        const int joint_tot = 2 * rank + 2 * N;
        for (int i = 0; i < n_draft; ++i) {
            const float* hi = block_cur + (size_t) (i + 1) * N;
            dspark_detail::k_dspark_prep_joint<<<(unsigned) ((joint_tot + 255) / 256), 256, 0, E.st>>>(
                rnn_state, dm->markov_w1, dm->markov_w1_type, prev_id, hi, f_rnn_last, joint_inp, rank, N);
            E.project(joint_inp, dm->joint_proj, joint_out, 1, 3 * rank, joint_tot);
            dspark_detail::k_dspark_rnn_update<<<(unsigned) ((rank + 255) / 256), 256, 0, E.st>>>(
                joint_out, dm->joint_proj_b, rnn_state, rnn_latent, rank);
            E.project(rnn_latent, dm->hidden_out, rnn_corr, 1, N, rank);
            dspark_detail::k_dspark_add3<<<(unsigned) ((N + 255) / 256), 256, 0, E.st>>>(
                h_pred, hi, rnn_corr, dm->hidden_out_b, N);
            E.project(h_pred, E.m->output_weight, E.logits, 1, hp.vocab_unpadded, N);
            if (out_logits) {
                k_softcap<<<(unsigned) ((hp.vocab_unpadded + 255) / 256), 256, 0, E.st>>>(
                    E.logits, hp.vocab_unpadded, hp.softcap);
                auto& row_lg = (*out_logits)[(size_t) i];
                row_lg.resize((size_t) hp.vocab_unpadded);
                CK(cudaMemcpyAsync(row_lg.data(), E.logits, (size_t) hp.vocab_unpadded * sizeof(float),
                                   cudaMemcpyDeviceToHost, E.st));
                CK(cudaStreamSynchronize(E.st));
                prev_id = sample_fn(i, row_lg);
            } else {
                k_argmax<<<1, 256, 0, E.st>>>(E.logits, E.pick_d, (int) hp.vocab_unpadded);
                int32_t next_id = 0;
                CK(cudaMemcpyAsync(&next_id, E.pick_d, sizeof(int32_t), cudaMemcpyDeviceToHost, E.st));
                CK(cudaStreamSynchronize(E.st));
                prev_id = next_id;
            }
            out_tokens[(size_t) i] = prev_id;
        }
        ++n_draft_calls;
        n_drafted += n_draft;
        if (E.prof) draft_ms += ms_since(t0);
    }
};
