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

/// Mix 4 target feature tensors via softmax(fuse_row[0..3]) and RMSNorm:
/// Single-pass 576-thr float4 register-cached when N == 2304 && blockDim.x == 576 (0 dst re-reads!).
__global__ void k_dspark_mix_features(const float* __restrict__ f0, const float* __restrict__ f1,
                                      const float* __restrict__ f2, const float* __restrict__ f3,
                                      const float* __restrict__ fuse_row, const float* __restrict__ norm_w,
                                      float* __restrict__ out, int N, float eps) {
    __shared__ float sw[4];
    __shared__ float ssum[32];
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
    if (N == 2304 && blockDim.x == 576) {
        const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
        const float4 v0 = reinterpret_cast<const float4*>(f0 + base)[tid];
        const float4 v1 = reinterpret_cast<const float4*>(f1 + base)[tid];
        const float4 v2 = reinterpret_cast<const float4*>(f2 + base)[tid];
        const float4 v3 = reinterpret_cast<const float4*>(f3 + base)[tid];
        const float4 x = make_float4(
            w0 * v0.x + w1 * v1.x + w2 * v2.x + w3 * v3.x,
            w0 * v0.y + w1 * v1.y + w2 * v2.y + w3 * v3.y,
            w0 * v0.z + w1 * v1.z + w2 * v2.z + w3 * v3.z,
            w0 * v0.w + w1 * v1.w + w2 * v2.w + w3 * v3.w);
        float acc = warp_sum_d(x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w);
        if (lane == 0) ssum[warp] = acc;
        __syncthreads();
        if (warp == 0) {
            float tot = (lane < 18) ? ssum[lane] : 0.0f;
            tot = warp_sum_d(tot);
            if (lane == 0) ssum[0] = rsqrtf(tot / 2304.0f + eps);
        }
        __syncthreads();
        const float sc = ssum[0];
        const float4 nw = __ldg(reinterpret_cast<const float4*>(norm_w) + tid);
        reinterpret_cast<float4*>(dst)[tid] = make_float4(x.x * sc * nw.x, x.y * sc * nw.y, x.z * sc * nw.z, x.w * sc * nw.w);
        return;
    }
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

/// Final term (k=3) + fused FP32 RMSNorm (exact bit-for-bit parity with k_dspark_mix_features).
/// Safe for in-place `out == acc` (used for `f_rnn_last` update and FP32/BF16-sm80+ fallback).
/// Grid: (T_eff), Block: (N == 2304) ? 576 : 256.
__global__ void k_dspark_accum_final_rmsnorm_f32(
        const float* __restrict__ cur, const float* acc,
        const float* __restrict__ norm_w, float* out,
        float w, float eps, int T_eff, int t_offset, int N) {
    __shared__ float ssum[32];
    const int ti = (int) blockIdx.x;
    if (ti >= T_eff) return;
    const float* cr = cur + (size_t) (t_offset + ti) * (size_t) N;
    const float* ar = acc + (size_t) ti * (size_t) N;
    float* dst = out + (size_t) ti * (size_t) N;
    if (N == 2304 && blockDim.x == 576) {
        const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
        const float4 cv = reinterpret_cast<const float4*>(cr)[tid];
        const float4 av = reinterpret_cast<const float4*>(ar)[tid];
        const float4 x = make_float4(av.x + w * cv.x, av.y + w * cv.y, av.z + w * cv.z, av.w + w * cv.w);
        float s = warp_sum_d(x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w);
        if (lane == 0) ssum[warp] = s;
        __syncthreads();
        if (warp == 0) {
            float tot = (lane < 18) ? ssum[lane] : 0.0f;
            tot = warp_sum_d(tot);
            if (lane == 0) ssum[0] = rsqrtf(tot / 2304.0f + eps);
        }
        __syncthreads();
        const float sc = ssum[0];
        if (norm_w) {
            const float4 nw = __ldg(reinterpret_cast<const float4*>(norm_w) + tid);
            reinterpret_cast<float4*>(dst)[tid] = make_float4(
                x.x * sc * nw.x, x.y * sc * nw.y, x.z * sc * nw.z, x.w * sc * nw.w);
        } else {
            reinterpret_cast<float4*>(dst)[tid] = make_float4(x.x * sc, x.y * sc, x.z * sc, x.w * sc);
        }
        return;
    }
    float s = 0.0f;
    for (int c = threadIdx.x; c < N; c += blockDim.x) {
        const float x = ar[c] + w * cr[c];
        dst[c] = x;
        s += x * x;
    }
    s = warp_sum_d(s);
    if ((threadIdx.x & 31) == 0) ssum[threadIdx.x >> 5] = s;
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
        dst[c] = dst[c] * sc * (norm_w ? norm_w[c] : 1.0f);
    }
}

/// Fused per-head RMSNorm(k) + NeoX RoPE(k) + Q8 store for injected k_ctx, v_ctx.
/// Reads pos0 from `pos0_d[0]` when non-null (enabling CUDA Graph capture of inject!), else uses `pos0_imm`.
/// Grid (n_head_kv, T), Block 64 (d = 0..63 pairs (d, d+64) for D=128).
__global__ void k_dspark_norm_rope_kv_store_q8(const float* __restrict__ K, const float* __restrict__ V,
                                               const float* __restrict__ k_norm_w,
                                               int8_t* __restrict__ kq, __half* __restrict__ ks,
                                               int8_t* __restrict__ vq, __half* __restrict__ vs,
                                               int n_head_kv, int pos0_imm, const int32_t* __restrict__ pos0_d,
                                               int cap, int is_swa, float eps, float freq_base, int n_rot) {
    __shared__ float s_red[2];
    __shared__ float s_red_k[2];
    __shared__ float s_red_v[2];
    const int pos0 = pos0_d ? pos0_d[0] : pos0_imm;
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
    if ((d & 31) == 0) { s_red_k[warp] = ak; s_red_v[warp] = av; }
    __syncthreads();
    const float amax_k = fmaxf(s_red_k[0], s_red_k[1]);
    const float amax_v = fmaxf(s_red_v[0], s_red_v[1]);

    const int pos = pos0 + t;
    const int slot = is_swa ? (pos % cap) : pos;
    const float dk = amax_k / 127.0f;
    const float dv = amax_v / 127.0f;
    const float idk = dk > 0.0f ? (127.0f / amax_k) : 0.0f;
    const float idv = dv > 0.0f ? (127.0f / amax_v) : 0.0f;

    const size_t dst_off = is_swa ? ((size_t) slot * kv_dim + hkv * 128) : (((size_t) hkv * cap + slot) * 128);
    kq[dst_off + d]      = (int8_t) __float2int_rn(k0 * idk);
    kq[dst_off + d + 64] = (int8_t) __float2int_rn(k1 * idk);
    vq[dst_off + d]      = (int8_t) __float2int_rn(v0 * idv);
    vq[dst_off + d + 64] = (int8_t) __float2int_rn(v1 * idv);
    if (d == 0) {
        const size_t s_off = is_swa ? ((size_t) slot * n_head_kv + hkv) : ((size_t) hkv * cap + slot);
        ks[s_off] = __float2half(dk);
        vs[s_off] = __float2half(dv);
    }
}

/// 4-token-per-CTA variant with precomputed inv_freq_d (zero powf, 256 thr = 4 tokens x 64 threads).
/// Grid (n_head_kv, (T + 3) / 4), Block 256. Bit-exact to k_dspark_norm_rope_kv_store_q8.
__global__ void __launch_bounds__(256) k_dspark_norm_rope_kv_store_q8_x4(
        const float* __restrict__ K, const float* __restrict__ V,
        const float* __restrict__ k_norm_w,
        int8_t* __restrict__ kq, __half* __restrict__ ks,
        int8_t* __restrict__ vq, __half* __restrict__ vs,
        int n_head_kv, int pos0, int T, int cap, int is_swa,
        float eps, int n_rot, const float* __restrict__ inv_freq_d) {
    __shared__ float s_red[4][2];
    __shared__ float s_red_k[4][2];
    __shared__ float s_red_v[4][2];
    const int hkv = (int) blockIdx.x;
    const int tid = threadIdx.x;
    const int sub = tid >> 6;          // 0..3 token within CTA
    const int d   = tid & 63;          // 0..63
    const int w2  = (tid >> 5) & 1;    // 0..1 warp within token's 64 threads
    const int t   = (int) blockIdx.y * 4 + sub;
    const bool valid = (t < T);
    const int kv_dim = n_head_kv * 128;

    float k0_raw = 0.0f, k1_raw = 0.0f, v0 = 0.0f, v1 = 0.0f;
    if (valid) {
        const size_t src_off = (size_t) t * kv_dim + hkv * 128;
        k0_raw = K[src_off + d];
        k1_raw = K[src_off + d + 64];
        v0 = V[src_off + d];
        v1 = V[src_off + d + 64];
    }

    const float sq = warp_sum_d(k0_raw * k0_raw + k1_raw * k1_raw);
    if ((d & 31) == 0) s_red[sub][w2] = sq;
    __syncthreads();

    float k0 = 0.0f, k1 = 0.0f;
    if (valid) {
        const float rms = rsqrtf((s_red[sub][0] + s_red[sub][1]) / 128.0f + eps);
        k0 = k0_raw * rms * k_norm_w[d];
        k1 = k1_raw * rms * k_norm_w[d + 64];
        if (d * 2 < n_rot) {
            const float theta = (float) (pos0 + t) * __ldg(inv_freq_d + d);
            float sv, cv;
            __sincosf(theta, &sv, &cv);
            const float rk0 = k0 * cv - k1 * sv;
            const float rk1 = k0 * sv + k1 * cv;
            k0 = rk0;
            k1 = rk1;
        }
    }

    const float ak = warp_max_d(fmaxf(fabsf(k0), fabsf(k1)));
    const float av = warp_max_d(fmaxf(fabsf(v0), fabsf(v1)));
    if ((d & 31) == 0) { s_red_k[sub][w2] = ak; s_red_v[sub][w2] = av; }
    __syncthreads();

    if (valid) {
        const float amax_k = fmaxf(s_red_k[sub][0], s_red_k[sub][1]);
        const float amax_v = fmaxf(s_red_v[sub][0], s_red_v[sub][1]);
        const int pos = pos0 + t;
        const int slot = is_swa ? (pos % cap) : pos;
        const float dk = amax_k / 127.0f;
        const float dv = amax_v / 127.0f;
        const float idk = dk > 0.0f ? (127.0f / amax_k) : 0.0f;
        const float idv = dv > 0.0f ? (127.0f / amax_v) : 0.0f;

        const size_t dst_off = is_swa ? ((size_t) slot * kv_dim + hkv * 128) : (((size_t) hkv * cap + slot) * 128);
        kq[dst_off + d]      = (int8_t) __float2int_rn(k0 * idk);
        kq[dst_off + d + 64] = (int8_t) __float2int_rn(k1 * idk);
        vq[dst_off + d]      = (int8_t) __float2int_rn(v0 * idv);
        vq[dst_off + d + 64] = (int8_t) __float2int_rn(v1 * idv);
        if (d == 0) {
            const size_t s_off = is_swa ? ((size_t) slot * n_head_kv + hkv) : ((size_t) hkv * cap + slot);
            ks[s_off] = __float2half(dk);
            vs[s_off] = __float2half(dv);
        }
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
        const size_t k_off = is_swa ? ((size_t) slot * kv_dim + hkv * 128) : (((size_t) hkv * cap + slot) * 128);
        const size_t s_off = is_swa ? ((size_t) slot * n_head_kv + hkv) : ((size_t) hkv * cap + slot);
        const int8_t* kj = kq_ctx + k_off;
        const int8_t* vj = vq_ctx + k_off;
        const float ksc = __half2float(ks_ctx[s_off]);
        const float vsc = __half2float(vs_ctx[s_off]);
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

/// Zero-Sync Tensor-Core (WMMA 16x16x16) Split-KV non-causal attention for DSpark full-attention (GQA-3, block_n <= 8, D=128):
/// 4 warps (128 thr, 4 blk/SM, 18,432 B SMEM, <= 128 regs, 0 spills): 2 row-tiles (rt=0..1) x 2 independent KV streams (kw=0..1),
/// sharing 16x136 sKV[kw] via 64-thr bar.sync with ZERO __syncthreads(), ZERO sS/sP SMEM via wmma_acc_to_matA_16x16,
/// V prefetch overlapping QK^T, direct FP16 self-block in sp == 0, and single 2-pass epilogue combine.
template <bool IS_SWA = false>
__global__ void __launch_bounds__(128, 4) k_dspark_attn_q8_wmma_split(
        const float* __restrict__ Q,
        const int8_t* __restrict__ kq_ctx, const __half* __restrict__ ks_ctx,
        const int8_t* __restrict__ vq_ctx, const __half* __restrict__ vs_ctx,
        const float* __restrict__ K_self, const float* __restrict__ V_self,
        float* __restrict__ split_out,
        int n_head, int n_head_kv, int cap, int n_swa, const int32_t* __restrict__ pos_d, int block_n, float kq_scale, int split_n) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
    // Volta sm_70 native PTX HMMA.884 path: STRIDE=144 (18,432 B = 4 blk/SM, 0-bank-conflict LDS.64),
    // V prefetch overlapping QK^T + K-next prefetch overlapping P*V, STACK:0, direct float2 epilogue.
    constexpr int STRIDE = 144;
    extern __shared__ __align__(16) char smem_raw[];
    __half* sQ = reinterpret_cast<__half*>(smem_raw);       // 32 * 144 * 2 = 9,216 B
    __half* sKV_all = sQ + 32 * STRIDE;                     // 2 * 16 * 144 * 2 = 9,216 B (total = 18,432 B)

    const int hkv = blockIdx.x, sp = blockIdx.y;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int rt = warp >> 1;
    const int kw = warp & 1;
    const int h0 = 3 * hkv;
    const int kv_dim = n_head_kv * 128;
    const int n_rows = block_n * 3;
    const int P = pos_d[0];
    constexpr float kLog2e = 1.4426950408889634f;
    const float qscale_all = kq_scale * kLog2e;

    const int j0 = IS_SWA ? max(0, P - n_swa + 1) : 0;
    const int n_tok = P - j0;
    const int chunk_sz = max(64, (n_tok + split_n - 1) / split_n);
    const int c_beg = j0 + sp * chunk_sz;
    if (c_beg >= P && sp > 0) {
        if (tid < n_rows) {
            float* dst = split_out + (((size_t) (tid / 3) * split_n + sp) * n_head + (h0 + (tid % 3))) * 130;
            dst[128] = -1e30f;
            dst[129] = 0.0f;
        }
        return;
    }
    const int c_end = min(P, c_beg + chunk_sz);

#pragma unroll
    for (int br = 0; br < 32; br += 16) {
        const int qr = br + (tid >> 3), qc0 = (tid & 7) * 16;
        const float* qp = (qr < n_rows) ? (Q + ((size_t) (qr / 3) * n_head + (h0 + (qr % 3))) * 128 + qc0) : nullptr;
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

    float rm0 = -1e30f, rm1 = -1e30f, rl0 = 0.0f, rl1 = 0.0f;
    float acc[8][8];
#pragma unroll
    for (int n = 0; n < 8; ++n) {
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[n][i] = 0.0f;
    }

    __half* sKV = sKV_all + kw * 16 * STRIDE;
    const int qa_off = (rt * 16 + qa_row) * STRIDE;
    const int kb_off = kb_row * STRIDE;
    const int vb_off = vb_row * STRIDE + vb_col;
    const int tid_kw = (rt << 5) | lane;
    const int kr = tid_kw >> 2;
    const int kc0 = (tid_kw & 3) * 32;
    const __half2 neg1152 = __half2half2(__float2half(-1152.0f));

    int jb = c_beg + kw * 16;
    for (; jb <= c_end - 16; jb += 32) {
        const int j = jb + kr;
        const int slot_s = IS_SWA ? (j % cap) : 0;
        {
            const __half2 dk2 = __half2half2(__ldg(IS_SWA ? &ks_ctx[(size_t) slot_s * n_head_kv + hkv] : &ks_ctx[hkv * cap + j]));
            const __half2 kb = __hmul2(neg1152, dk2);
            const int8_t* kp = IS_SWA ? (kq_ctx + (size_t) slot_s * kv_dim + hkv * 128 + kc0) : (kq_ctx + (size_t) (hkv * cap + j) * 128 + kc0);
            __half* dst = sKV + kr * STRIDE + kc0;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const uint4 raw = *reinterpret_cast<const uint4*>(kp + i * 16);
                __half2 h2[8];
                dspark_i8x4_to_h2x2_fma(raw.x, dk2, kb, h2[0], h2[1]); dspark_i8x4_to_h2x2_fma(raw.y, dk2, kb, h2[2], h2[3]);
                dspark_i8x4_to_h2x2_fma(raw.z, dk2, kb, h2[4], h2[5]); dspark_i8x4_to_h2x2_fma(raw.w, dk2, kb, h2[6], h2[7]);
                *reinterpret_cast<uint4*>(dst + i * 16)     = *reinterpret_cast<const uint4*>(h2);
                *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
            }
        }
        asm volatile("bar.sync %0, 64;" :: "r"(kw + 1) : "memory");

        const __half2 dv2 = __half2half2(__ldg(IS_SWA ? &vs_ctx[(size_t) slot_s * n_head_kv + hkv] : &vs_ctx[hkv * cap + j]));
        const int8_t* vp = IS_SWA ? (vq_ctx + (size_t) slot_s * kv_dim + hkv * 128 + kc0) : (vq_ctx + (size_t) (hkv * cap + j) * 128 + kc0);
        const uint4 vr0 = *reinterpret_cast<const uint4*>(vp);
        const uint4 vr1 = *reinterpret_cast<const uint4*>(vp + 16);

        float fc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 1
        for (int k = 0; k < 128; k += 8) {
            const uint4 q0 = *reinterpret_cast<const uint4*>(sQ + qa_off + k);
            const uint4 k0 = *reinterpret_cast<const uint4*>(sKV + kb_off + k);
            mma_rc(fc, q0.x, q0.y, k0.x, k0.y);
            mma_rc(fc, q0.z, q0.w, k0.z, k0.w);
        }
        asm volatile("bar.sync %0, 64;" :: "r"(kw + 1) : "memory");

        {
            const __half2 vb = __hmul2(neg1152, dv2);
            __half* dst = sKV + kr * STRIDE + kc0;
            __half2 h2[8];
            dspark_i8x4_to_h2x2_fma(vr0.x, dv2, vb, h2[0], h2[1]); dspark_i8x4_to_h2x2_fma(vr0.y, dv2, vb, h2[2], h2[3]);
            dspark_i8x4_to_h2x2_fma(vr0.z, dv2, vb, h2[4], h2[5]); dspark_i8x4_to_h2x2_fma(vr0.w, dv2, vb, h2[6], h2[7]);
            *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(h2); *reinterpret_cast<uint4*>(dst + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
            dspark_i8x4_to_h2x2_fma(vr1.x, dv2, vb, h2[0], h2[1]); dspark_i8x4_to_h2x2_fma(vr1.y, dv2, vb, h2[2], h2[3]);
            dspark_i8x4_to_h2x2_fma(vr1.z, dv2, vb, h2[4], h2[5]); dspark_i8x4_to_h2x2_fma(vr1.w, dv2, vb, h2[6], h2[7]);
            *reinterpret_cast<uint4*>(dst + 16) = *reinterpret_cast<const uint4*>(h2); *reinterpret_cast<uint4*>(dst + 24) = *reinterpret_cast<const uint4*>(h2 + 4);
        }

        const float mx0 = wmma_row_max_16(fmaxf(fmaxf(fc[0], fc[1]), fmaxf(fc[4], fc[5])));
        const float mx1 = wmma_row_max_16(fmaxf(fmaxf(fc[2], fc[3]), fmaxf(fc[6], fc[7])));
        const float m_new0 = fmaxf(rm0, mx0), m_new1 = fmaxf(rm1, mx1);
        const float a0 = (rm0 > -1e20f) ? ((m_new0 == rm0) ? 1.0f : dspark_fast_exp2f_hw(rm0 - m_new0)) : 0.0f;
        const float a1 = (rm1 > -1e20f) ? ((m_new1 == rm1) ? 1.0f : dspark_fast_exp2f_hw(rm1 - m_new1)) : 0.0f;
        if (__any_sync(0xffffffffu, a0 != 1.0f || a1 != 1.0f)) {
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                acc[n][0] *= a0; acc[n][1] *= a0; acc[n][4] *= a0; acc[n][5] *= a0;
                acc[n][2] *= a1; acc[n][3] *= a1; acc[n][6] *= a1; acc[n][7] *= a1;
            }
        }
        const float p00 = dspark_fast_exp2f_hw(fc[0] - m_new0), p01 = dspark_fast_exp2f_hw(fc[1] - m_new0);
        const float p04 = dspark_fast_exp2f_hw(fc[4] - m_new0), p05 = dspark_fast_exp2f_hw(fc[5] - m_new0);
        const float p12 = dspark_fast_exp2f_hw(fc[2] - m_new1), p13 = dspark_fast_exp2f_hw(fc[3] - m_new1);
        const float p16 = dspark_fast_exp2f_hw(fc[6] - m_new1), p17 = dspark_fast_exp2f_hw(fc[7] - m_new1);
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
        asm volatile("bar.sync %0, 64;" :: "r"(kw + 1) : "memory");

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
        asm volatile("bar.sync %0, 64;" :: "r"(kw + 1) : "memory");
    }

    if (jb < c_end) {
        const int j = jb + kr;
        const int slot_s = (IS_SWA && j < c_end) ? (j % cap) : 0;
        {
            __half* dst = sKV + kr * STRIDE + kc0;
            if (j < c_end) {
                const __half2 dk2 = __half2half2(__ldg(IS_SWA ? &ks_ctx[(size_t) slot_s * n_head_kv + hkv] : &ks_ctx[hkv * cap + j]));
                const __half2 kb = __hmul2(neg1152, dk2);
                const int8_t* kp = IS_SWA ? (kq_ctx + (size_t) slot_s * kv_dim + hkv * 128 + kc0) : (kq_ctx + (size_t) (hkv * cap + j) * 128 + kc0);
#pragma unroll 1
                for (int i = 0; i < 2; ++i) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(kp + i * 16);
                    __half2 h2[8];
                    dspark_i8x4_to_h2x2_fma(raw.x, dk2, kb, h2[0], h2[1]); dspark_i8x4_to_h2x2_fma(raw.y, dk2, kb, h2[2], h2[3]);
                    dspark_i8x4_to_h2x2_fma(raw.z, dk2, kb, h2[4], h2[5]); dspark_i8x4_to_h2x2_fma(raw.w, dk2, kb, h2[6], h2[7]);
                    *reinterpret_cast<uint4*>(dst + i * 16)     = *reinterpret_cast<const uint4*>(h2);
                    *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                }
            } else {
                const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll 1
                for (int i = 0; i < 4; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
            }
        }
        asm volatile("bar.sync %0, 64;" :: "r"(kw + 1) : "memory");

        float fc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 1
        for (int k = 0; k < 128; k += 8) {
            const uint4 q0 = *reinterpret_cast<const uint4*>(sQ + qa_off + k);
            const uint4 k0 = *reinterpret_cast<const uint4*>(sKV + kb_off + k);
            mma_rc(fc, q0.x, q0.y, k0.x, k0.y);
            mma_rc(fc, q0.z, q0.w, k0.z, k0.w);
        }
        asm volatile("bar.sync %0, 64;" :: "r"(kw + 1) : "memory");

        {
            __half* dst = sKV + kr * STRIDE + kc0;
            if (j < c_end) {
                const __half2 dv2 = __half2half2(__ldg(IS_SWA ? &vs_ctx[(size_t) slot_s * n_head_kv + hkv] : &vs_ctx[hkv * cap + j]));
                const __half2 vb = __hmul2(neg1152, dv2);
                const int8_t* vp = IS_SWA ? (vq_ctx + (size_t) slot_s * kv_dim + hkv * 128 + kc0) : (vq_ctx + (size_t) (hkv * cap + j) * 128 + kc0);
#pragma unroll 1
                for (int i = 0; i < 2; ++i) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(vp + i * 16);
                    __half2 h2[8];
                    dspark_i8x4_to_h2x2_fma(raw.x, dv2, vb, h2[0], h2[1]); dspark_i8x4_to_h2x2_fma(raw.y, dv2, vb, h2[2], h2[3]);
                    dspark_i8x4_to_h2x2_fma(raw.z, dv2, vb, h2[4], h2[5]); dspark_i8x4_to_h2x2_fma(raw.w, dv2, vb, h2[6], h2[7]);
                    *reinterpret_cast<uint4*>(dst + i * 16)     = *reinterpret_cast<const uint4*>(h2);
                    *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                }
            } else {
                const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll 1
                for (int i = 0; i < 4; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
            }
        }
        const int rem = c_end - jb;
        if (acc_c0     >= rem) { fc[0] = -1e30f; fc[2] = -1e30f; }
        if (acc_c0 + 1 >= rem) { fc[1] = -1e30f; fc[3] = -1e30f; }
        if (acc_c1     >= rem) { fc[4] = -1e30f; fc[6] = -1e30f; }
        if (acc_c1 + 1 >= rem) { fc[5] = -1e30f; fc[7] = -1e30f; }
        const float mx0 = wmma_row_max_16(fmaxf(fmaxf(fc[0], fc[1]), fmaxf(fc[4], fc[5])));
        const float mx1 = wmma_row_max_16(fmaxf(fmaxf(fc[2], fc[3]), fmaxf(fc[6], fc[7])));
        const float m_new0 = fmaxf(rm0, mx0), m_new1 = fmaxf(rm1, mx1);
        const float a0 = (rm0 > -1e20f) ? ((m_new0 == rm0) ? 1.0f : dspark_fast_exp2f_hw(rm0 - m_new0)) : 0.0f;
        const float a1 = (rm1 > -1e20f) ? ((m_new1 == rm1) ? 1.0f : dspark_fast_exp2f_hw(rm1 - m_new1)) : 0.0f;
        if (__any_sync(0xffffffffu, a0 != 1.0f || a1 != 1.0f)) {
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                acc[n][0] *= a0; acc[n][1] *= a0; acc[n][4] *= a0; acc[n][5] *= a0;
                acc[n][2] *= a1; acc[n][3] *= a1; acc[n][6] *= a1; acc[n][7] *= a1;
            }
        }
        const float p00 = (fc[0] > -1e20f) ? dspark_fast_exp2f_hw(fc[0] - m_new0) : 0.0f;
        const float p01 = (fc[1] > -1e20f) ? dspark_fast_exp2f_hw(fc[1] - m_new0) : 0.0f;
        const float p04 = (fc[4] > -1e20f) ? dspark_fast_exp2f_hw(fc[4] - m_new0) : 0.0f;
        const float p05 = (fc[5] > -1e20f) ? dspark_fast_exp2f_hw(fc[5] - m_new0) : 0.0f;
        const float p12 = (fc[2] > -1e20f) ? dspark_fast_exp2f_hw(fc[2] - m_new1) : 0.0f;
        const float p13 = (fc[3] > -1e20f) ? dspark_fast_exp2f_hw(fc[3] - m_new1) : 0.0f;
        const float p16 = (fc[6] > -1e20f) ? dspark_fast_exp2f_hw(fc[6] - m_new1) : 0.0f;
        const float p17 = (fc[7] > -1e20f) ? dspark_fast_exp2f_hw(fc[7] - m_new1) : 0.0f;
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
        asm volatile("bar.sync %0, 64;" :: "r"(kw + 1) : "memory");
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
        asm volatile("bar.sync %0, 64;" :: "r"(kw + 1) : "memory");
    }

    if (sp == 0 && kw == 0) {
        {
            const float* kp = (kr < block_n) ? (K_self + ((size_t) kr * n_head_kv + hkv) * 128 + kc0) : nullptr;
            __half2* dst2 = reinterpret_cast<__half2*>(sKV + kr * STRIDE + kc0);
            if (kp) {
#pragma unroll 1
                for (int i = 0; i < 8; ++i) {
                    const float4 v = *reinterpret_cast<const float4*>(kp + i * 4);
                    dst2[i * 2 + 0] = __halves2half2(__float2half(v.x), __float2half(v.y));
                    dst2[i * 2 + 1] = __halves2half2(__float2half(v.z), __float2half(v.w));
                }
            } else {
                const __half2 z2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));
#pragma unroll 1
                for (int i = 0; i < 16; ++i) dst2[i] = z2;
            }
        }
        asm volatile("bar.sync 1, 64;" ::: "memory");

        float fc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 1
        for (int k = 0; k < 128; k += 8) {
            const uint4 q0 = *reinterpret_cast<const uint4*>(sQ + qa_off + k);
            const uint4 k0 = *reinterpret_cast<const uint4*>(sKV + kb_off + k);
            mma_rc(fc, q0.x, q0.y, k0.x, k0.y);
            mma_rc(fc, q0.z, q0.w, k0.z, k0.w);
        }
        asm volatile("bar.sync 1, 64;" ::: "memory");

        {
            const float* vp = (kr < block_n) ? (V_self + ((size_t) kr * n_head_kv + hkv) * 128 + kc0) : nullptr;
            __half2* dst2 = reinterpret_cast<__half2*>(sKV + kr * STRIDE + kc0);
            if (vp) {
#pragma unroll 1
                for (int i = 0; i < 8; ++i) {
                    const float4 v = *reinterpret_cast<const float4*>(vp + i * 4);
                    dst2[i * 2 + 0] = __halves2half2(__float2half(v.x), __float2half(v.y));
                    dst2[i * 2 + 1] = __halves2half2(__float2half(v.z), __float2half(v.w));
                }
            } else {
                const __half2 z2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));
#pragma unroll 1
                for (int i = 0; i < 16; ++i) dst2[i] = z2;
            }
        }
        if (acc_c0     >= block_n) { fc[0] = -1e30f; fc[2] = -1e30f; }
        if (acc_c0 + 1 >= block_n) { fc[1] = -1e30f; fc[3] = -1e30f; }
        if (acc_c1     >= block_n) { fc[4] = -1e30f; fc[6] = -1e30f; }
        if (acc_c1 + 1 >= block_n) { fc[5] = -1e30f; fc[7] = -1e30f; }
        const float mx0 = wmma_row_max_16(fmaxf(fmaxf(fc[0], fc[1]), fmaxf(fc[4], fc[5])));
        const float mx1 = wmma_row_max_16(fmaxf(fmaxf(fc[2], fc[3]), fmaxf(fc[6], fc[7])));
        const float m_new0 = fmaxf(rm0, mx0), m_new1 = fmaxf(rm1, mx1);
        const float a0 = (rm0 > -1e20f) ? ((m_new0 == rm0) ? 1.0f : dspark_fast_exp2f_hw(rm0 - m_new0)) : 0.0f;
        const float a1 = (rm1 > -1e20f) ? ((m_new1 == rm1) ? 1.0f : dspark_fast_exp2f_hw(rm1 - m_new1)) : 0.0f;
        if (__any_sync(0xffffffffu, a0 != 1.0f || a1 != 1.0f)) {
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                acc[n][0] *= a0; acc[n][1] *= a0; acc[n][4] *= a0; acc[n][5] *= a0;
                acc[n][2] *= a1; acc[n][3] *= a1; acc[n][6] *= a1; acc[n][7] *= a1;
            }
        }
        const float p00 = (fc[0] > -1e20f) ? dspark_fast_exp2f_hw(fc[0] - m_new0) : 0.0f;
        const float p01 = (fc[1] > -1e20f) ? dspark_fast_exp2f_hw(fc[1] - m_new0) : 0.0f;
        const float p04 = (fc[4] > -1e20f) ? dspark_fast_exp2f_hw(fc[4] - m_new0) : 0.0f;
        const float p05 = (fc[5] > -1e20f) ? dspark_fast_exp2f_hw(fc[5] - m_new0) : 0.0f;
        const float p12 = (fc[2] > -1e20f) ? dspark_fast_exp2f_hw(fc[2] - m_new1) : 0.0f;
        const float p13 = (fc[3] > -1e20f) ? dspark_fast_exp2f_hw(fc[3] - m_new1) : 0.0f;
        const float p16 = (fc[6] > -1e20f) ? dspark_fast_exp2f_hw(fc[6] - m_new1) : 0.0f;
        const float p17 = (fc[7] > -1e20f) ? dspark_fast_exp2f_hw(fc[7] - m_new1) : 0.0f;
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
        asm volatile("bar.sync 1, 64;" ::: "memory");
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
    }

    __syncthreads();
    float* s_wm = reinterpret_cast<float*>(smem_raw + 17408);
    float* s_wl = s_wm + 64;
    float* s_mg = s_wl + 64;
    float* s_lg = s_mg + 32;
    if (is_leader) {
        s_wm[kw * 32 + rt * 16 + acc_r0] = rm0; s_wl[kw * 32 + rt * 16 + acc_r0] = rl0;
        s_wm[kw * 32 + rt * 16 + acc_r1] = rm1; s_wl[kw * 32 + rt * 16 + acc_r1] = rl1;
    }
    __syncthreads();
    if (tid < n_rows) {
        float mg = -1e30f;
        if (s_wl[tid]      > 0.0f) mg = fmaxf(mg, s_wm[tid]);
        if (s_wl[32 + tid] > 0.0f) mg = fmaxf(mg, s_wm[32 + tid]);
        float lg = 0.0f;
        if (s_wl[tid]      > 0.0f) lg += s_wl[tid]      * dspark_fast_exp2f_hw(s_wm[tid]      - mg);
        if (s_wl[32 + tid] > 0.0f) lg += s_wl[32 + tid] * dspark_fast_exp2f_hw(s_wm[32 + tid] - mg);
        s_mg[tid] = mg;
        s_lg[tid] = lg;
    }
    __syncthreads();
    {
        const int r0 = rt * 16 + acc_r0, r1 = rt * 16 + acc_r1;
        const float cw0 = (r0 < n_rows && rl0 > 0.0f) ? dspark_fast_exp2f_hw(rm0 - s_mg[r0]) : 0.0f;
        const float cw1 = (r1 < n_rows && rl1 > 0.0f) ? dspark_fast_exp2f_hw(rm1 - s_mg[r1]) : 0.0f;
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            acc[n][0] *= cw0; acc[n][1] *= cw0; acc[n][4] *= cw0; acc[n][5] *= cw0;
            acc[n][2] *= cw1; acc[n][3] *= cw1; acc[n][6] *= cw1; acc[n][7] *= cw1;
        }
    }
    float* s_out = reinterpret_cast<float*>(smem_raw);
    float* s_tmp = s_out + 16 * 128;
    for (int g_rt = 0; g_rt < 2; ++g_rt) {
        if (g_rt * 16 >= n_rows) break;
        if (rt == g_rt) {
            float* dst_s = (kw == 0) ? s_out : s_tmp;
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                float* p0s = dst_s + acc_r0 * 128 + n * 16;
                float* p1s = dst_s + acc_r1 * 128 + n * 16;
                *reinterpret_cast<float2*>(p0s + acc_c0) = make_float2(acc[n][0], acc[n][1]);
                *reinterpret_cast<float2*>(p0s + acc_c1) = make_float2(acc[n][4], acc[n][5]);
                *reinterpret_cast<float2*>(p1s + acc_c0) = make_float2(acc[n][2], acc[n][3]);
                *reinterpret_cast<float2*>(p1s + acc_c1) = make_float2(acc[n][6], acc[n][7]);
            }
        }
        __syncthreads();
        const int r_lim = min(16, n_rows - g_rt * 16);
        for (int r = 0; r < r_lim; ++r) {
            const int rr = g_rt * 16 + r;
            float* dst = split_out + (((size_t) (rr / 3) * split_n + sp) * n_head + (h0 + (rr % 3))) * 130;
            dst[tid] = s_out[r * 128 + tid] + s_tmp[r * 128 + tid];
            if (tid == 0) {
                dst[128] = s_mg[rr];
                dst[129] = s_lg[rr];
            }
        }
        __syncthreads();
    }
#elif !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750
    using namespace nvcuda;
    extern __shared__ __align__(16) char smem_raw[];
    // 18,432 B (4 blk/SM): sQ (32*136*2 = 8,704 B) | sKV[2] (2*16*136*2 = 8,704 B) | epilogue scalars (1,024 B).
    __half* sQ = reinterpret_cast<__half*>(smem_raw);
    __half* sKV_all = sQ + 32 * 136;

    const int hkv = blockIdx.x, sp = blockIdx.y;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int rt = warp >> 1;      // 0: rows 0..15, 1: rows 16..31
    const int kw = warp & 1;       // 0..1: 2 independent 64-thr KV streams stepping by 32
    const int h0 = 3 * hkv;
    const int kv_dim = n_head_kv * 128;
    const int n_rows = block_n * 3;
    const int P = pos_d[0];
    constexpr float kLog2e = 1.4426950408889634f;
    const float qscale_all = kq_scale * kLog2e;

    const int j0 = IS_SWA ? max(0, P - n_swa + 1) : 0;
    const int n_tok = P - j0;
    const int chunk_sz = max(64, (n_tok + split_n - 1) / split_n);
    const int c_beg = j0 + sp * chunk_sz;
    if (c_beg >= P && sp > 0) {
        if (tid < n_rows) {
            float* dst = split_out + (((size_t) (tid / 3) * split_n + sp) * n_head + (h0 + (tid % 3))) * 130;
            dst[128] = -1e30f;
            dst[129] = 0.0f;
        }
        return;
    }
    const int c_end = min(P, c_beg + chunk_sz);

    // 128 threads load 32x128 Q -> sQ in 2 iterations (16 rows/iter, 8 thr/row, 16 elems/thr).
#pragma unroll
    for (int br = 0; br < 32; br += 16) {
        const int qr = br + (tid >> 3), qc0 = (tid & 7) * 16;
        const float* qp = (qr < n_rows) ? (Q + ((size_t) (qr / 3) * n_head + (h0 + (qr % 3))) * 128 + qc0) : nullptr;
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

    float rm0 = -1e30f, rm1 = -1e30f, rl0 = 0.0f, rl1 = 0.0f;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
#pragma unroll
    for (int n = 0; n < 8; ++n) wmma::fill_fragment(acc[n], 0.0f);

    __half* sKV = sKV_all + kw * 16 * 136;
    __half* sQ_rt = sQ + (rt * 16) * 136;
    const int tid_kw = (rt << 5) | lane;  // 0..63 within the 2-warp group sharing sKV[kw]
    const int kr = tid_kw >> 2;           // 0..15 (4 threads/row, 32 bytes/thread = 2 uint4s!)
    const int kc0 = (tid_kw & 3) * 32;
    const __half2 neg1152 = __half2half2(__float2half(-1152.0f));

    // Bulk context loop (all 16 keys valid): ZERO __syncthreads(), V prefetch overlapping QK^T!
    int jb = c_beg + kw * 16;
    for (; jb <= c_end - 16; jb += 32) {
        const int j = jb + kr;
        const int slot_s = IS_SWA ? (j % cap) : 0;
        {
            const __half2 dk2 = __half2half2(__ldg(IS_SWA ? &ks_ctx[(size_t) slot_s * n_head_kv + hkv] : &ks_ctx[hkv * cap + j]));
            const __half2 kb = __hmul2(neg1152, dk2);
            const int8_t* kp = IS_SWA ? (kq_ctx + (size_t) slot_s * kv_dim + hkv * 128 + kc0) : (kq_ctx + (size_t) (hkv * cap + j) * 128 + kc0);
            __half* dst = sKV + kr * 136 + kc0;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const uint4 raw = *reinterpret_cast<const uint4*>(kp + i * 16);
                __half2 h2[8];
                dspark_i8x4_to_h2x2_fma(raw.x, dk2, kb, h2[0], h2[1]); dspark_i8x4_to_h2x2_fma(raw.y, dk2, kb, h2[2], h2[3]);
                dspark_i8x4_to_h2x2_fma(raw.z, dk2, kb, h2[4], h2[5]); dspark_i8x4_to_h2x2_fma(raw.w, dk2, kb, h2[6], h2[7]);
                *reinterpret_cast<uint4*>(dst + i * 16)     = *reinterpret_cast<const uint4*>(h2);
                *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
            }
        }
        asm volatile("bar.sync %0, 64;" :: "r"(kw + 1) : "memory");

        const __half2 dv2 = __half2half2(__ldg(IS_SWA ? &vs_ctx[(size_t) slot_s * n_head_kv + hkv] : &vs_ctx[hkv * cap + j]));
        const int8_t* vp = IS_SWA ? (vq_ctx + (size_t) slot_s * kv_dim + hkv * 128 + kc0) : (vq_ctx + (size_t) (hkv * cap + j) * 128 + kc0);
        const uint4 vr0 = *reinterpret_cast<const uint4*>(vp);
        const uint4 vr1 = *reinterpret_cast<const uint4*>(vp + 16);

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
        wmma::fill_fragment(fc, 0.0f);
#pragma unroll
        for (int k = 0; k < 128; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
            wmma::load_matrix_sync(fa, sQ_rt + k, 136);
            wmma::load_matrix_sync(fb, sKV + k, 136);
            wmma::mma_sync(fc, fa, fb, fc);
        }
        asm volatile("bar.sync %0, 64;" :: "r"(kw + 1) : "memory");

        {
            const __half2 vb = __hmul2(neg1152, dv2);
            __half* dst = sKV + kr * 136 + kc0;
            __half2 h2[8];
            dspark_i8x4_to_h2x2_fma(vr0.x, dv2, vb, h2[0], h2[1]); dspark_i8x4_to_h2x2_fma(vr0.y, dv2, vb, h2[2], h2[3]);
            dspark_i8x4_to_h2x2_fma(vr0.z, dv2, vb, h2[4], h2[5]); dspark_i8x4_to_h2x2_fma(vr0.w, dv2, vb, h2[6], h2[7]);
            *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(h2); *reinterpret_cast<uint4*>(dst + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
            dspark_i8x4_to_h2x2_fma(vr1.x, dv2, vb, h2[0], h2[1]); dspark_i8x4_to_h2x2_fma(vr1.y, dv2, vb, h2[2], h2[3]);
            dspark_i8x4_to_h2x2_fma(vr1.z, dv2, vb, h2[4], h2[5]); dspark_i8x4_to_h2x2_fma(vr1.w, dv2, vb, h2[6], h2[7]);
            *reinterpret_cast<uint4*>(dst + 16) = *reinterpret_cast<const uint4*>(h2); *reinterpret_cast<uint4*>(dst + 24) = *reinterpret_cast<const uint4*>(h2 + 4);
        }

        const float mx0 = wmma_row_max_16(fmaxf(fmaxf(fc.x[0], fc.x[1]), fmaxf(fc.x[4], fc.x[5])));
        const float mx1 = wmma_row_max_16(fmaxf(fmaxf(fc.x[2], fc.x[3]), fmaxf(fc.x[6], fc.x[7])));
        const float m_new0 = fmaxf(rm0, mx0), m_new1 = fmaxf(rm1, mx1);
        const float a0 = (rm0 > -1e20f) ? ((m_new0 == rm0) ? 1.0f : dspark_fast_exp2f_hw(rm0 - m_new0)) : 0.0f;
        const float a1 = (rm1 > -1e20f) ? ((m_new1 == rm1) ? 1.0f : dspark_fast_exp2f_hw(rm1 - m_new1)) : 0.0f;
        if (__any_sync(0xffffffffu, a0 != 1.0f || a1 != 1.0f)) {
#pragma unroll
            for (int n = 0; n < 8; ++n) wmma_rescale_acc(acc[n], a0, a1);
        }
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa_p;
        float ps0 = 0.0f, ps1 = 0.0f;
        wmma_acc_to_matA_16x16<false>(fc, m_new0, m_new1, ps0, ps1, fa_p, lane);
        rm0 = m_new0; rl0 = rl0 * a0 + wmma_row_sum_16(ps0);
        rm1 = m_new1; rl1 = rl1 * a1 + wmma_row_sum_16(ps1);
        asm volatile("bar.sync %0, 64;" :: "r"(kw + 1) : "memory");

#pragma unroll
        for (int n = 0; n < 8; ++n) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
            wmma::load_matrix_sync(fb, sKV + n * 16, 136);
            wmma::mma_sync(acc[n], fa_p, fb, acc[n]);
        }
        asm volatile("bar.sync %0, 64;" :: "r"(kw + 1) : "memory");
    }

    // Partial tail context tile (< 16 keys)
    if (jb < c_end) {
        const int j = jb + kr;
        const int slot_s = (IS_SWA && j < c_end) ? (j % cap) : 0;
        {
            __half* dst = sKV + kr * 136 + kc0;
            if (j < c_end) {
                const __half2 dk2 = __half2half2(__ldg(IS_SWA ? &ks_ctx[(size_t) slot_s * n_head_kv + hkv] : &ks_ctx[hkv * cap + j]));
                const __half2 kb = __hmul2(neg1152, dk2);
                const int8_t* kp = IS_SWA ? (kq_ctx + (size_t) slot_s * kv_dim + hkv * 128 + kc0) : (kq_ctx + (size_t) (hkv * cap + j) * 128 + kc0);
#pragma unroll 1
                for (int i = 0; i < 2; ++i) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(kp + i * 16);
                    __half2 h2[8];
                    dspark_i8x4_to_h2x2_fma(raw.x, dk2, kb, h2[0], h2[1]); dspark_i8x4_to_h2x2_fma(raw.y, dk2, kb, h2[2], h2[3]);
                    dspark_i8x4_to_h2x2_fma(raw.z, dk2, kb, h2[4], h2[5]); dspark_i8x4_to_h2x2_fma(raw.w, dk2, kb, h2[6], h2[7]);
                    *reinterpret_cast<uint4*>(dst + i * 16)     = *reinterpret_cast<const uint4*>(h2);
                    *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                }
            } else {
                const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll 1
                for (int i = 0; i < 4; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
            }
        }
        asm volatile("bar.sync %0, 64;" :: "r"(kw + 1) : "memory");

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
        wmma::fill_fragment(fc, 0.0f);
#pragma unroll 1
        for (int k = 0; k < 128; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
            wmma::load_matrix_sync(fa, sQ_rt + k, 136);
            wmma::load_matrix_sync(fb, sKV + k, 136);
            wmma::mma_sync(fc, fa, fb, fc);
        }
        asm volatile("bar.sync %0, 64;" :: "r"(kw + 1) : "memory");

        {
            __half* dst = sKV + kr * 136 + kc0;
            if (j < c_end) {
                const __half2 dv2 = __half2half2(__ldg(IS_SWA ? &vs_ctx[(size_t) slot_s * n_head_kv + hkv] : &vs_ctx[hkv * cap + j]));
                const __half2 vb = __hmul2(neg1152, dv2);
                const int8_t* vp = IS_SWA ? (vq_ctx + (size_t) slot_s * kv_dim + hkv * 128 + kc0) : (vq_ctx + (size_t) (hkv * cap + j) * 128 + kc0);
#pragma unroll 1
                for (int i = 0; i < 2; ++i) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(vp + i * 16);
                    __half2 h2[8];
                    dspark_i8x4_to_h2x2_fma(raw.x, dv2, vb, h2[0], h2[1]); dspark_i8x4_to_h2x2_fma(raw.y, dv2, vb, h2[2], h2[3]);
                    dspark_i8x4_to_h2x2_fma(raw.z, dv2, vb, h2[4], h2[5]); dspark_i8x4_to_h2x2_fma(raw.w, dv2, vb, h2[6], h2[7]);
                    *reinterpret_cast<uint4*>(dst + i * 16)     = *reinterpret_cast<const uint4*>(h2);
                    *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                }
            } else {
                const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll 1
                for (int i = 0; i < 4; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
            }
        }
        const int rem = c_end - jb;
        if (acc_c0     >= rem) { fc.x[0] = -1e30f; fc.x[2] = -1e30f; }
        if (acc_c0 + 1 >= rem) { fc.x[1] = -1e30f; fc.x[3] = -1e30f; }
        if (acc_c1     >= rem) { fc.x[4] = -1e30f; fc.x[6] = -1e30f; }
        if (acc_c1 + 1 >= rem) { fc.x[5] = -1e30f; fc.x[7] = -1e30f; }
        const float mx0 = wmma_row_max_16(fmaxf(fmaxf(fc.x[0], fc.x[1]), fmaxf(fc.x[4], fc.x[5])));
        const float mx1 = wmma_row_max_16(fmaxf(fmaxf(fc.x[2], fc.x[3]), fmaxf(fc.x[6], fc.x[7])));
        const float m_new0 = fmaxf(rm0, mx0), m_new1 = fmaxf(rm1, mx1);
        const float a0 = (rm0 > -1e20f) ? ((m_new0 == rm0) ? 1.0f : dspark_fast_exp2f_hw(rm0 - m_new0)) : 0.0f;
        const float a1 = (rm1 > -1e20f) ? ((m_new1 == rm1) ? 1.0f : dspark_fast_exp2f_hw(rm1 - m_new1)) : 0.0f;
        if (__any_sync(0xffffffffu, a0 != 1.0f || a1 != 1.0f)) {
#pragma unroll
            for (int n = 0; n < 8; ++n) wmma_rescale_acc(acc[n], a0, a1);
        }
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa_p;
        float ps0 = 0.0f, ps1 = 0.0f;
        wmma_acc_to_matA_16x16<true>(fc, m_new0, m_new1, ps0, ps1, fa_p, lane);
        rm0 = m_new0; rl0 = rl0 * a0 + wmma_row_sum_16(ps0);
        rm1 = m_new1; rl1 = rl1 * a1 + wmma_row_sum_16(ps1);
        asm volatile("bar.sync %0, 64;" :: "r"(kw + 1) : "memory");

#pragma unroll
        for (int n = 0; n < 8; ++n) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
            wmma::load_matrix_sync(fb, sKV + n * 16, 136);
            wmma::mma_sync(acc[n], fa_p, fb, acc[n]);
        }
        asm volatile("bar.sync %0, 64;" :: "r"(kw + 1) : "memory");
    }

    // Split 0 appends exact WMMA self-block [0..block_n-1] via kw == 0 (64 threads, sKV[0]).
    if (sp == 0 && kw == 0) {
        {
            const float* kp = (kr < block_n) ? (K_self + ((size_t) kr * n_head_kv + hkv) * 128 + kc0) : nullptr;
            __half2* dst2 = reinterpret_cast<__half2*>(sKV + kr * 136 + kc0);
            if (kp) {
#pragma unroll 1
                for (int i = 0; i < 8; ++i) {
                    const float4 v = *reinterpret_cast<const float4*>(kp + i * 4);
                    dst2[i * 2 + 0] = __halves2half2(__float2half(v.x), __float2half(v.y));
                    dst2[i * 2 + 1] = __halves2half2(__float2half(v.z), __float2half(v.w));
                }
            } else {
                const __half2 z2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));
#pragma unroll 1
                for (int i = 0; i < 16; ++i) dst2[i] = z2;
            }
        }
        asm volatile("bar.sync 1, 64;" ::: "memory");

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
        wmma::fill_fragment(fc, 0.0f);
#pragma unroll 1
        for (int k = 0; k < 128; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
            wmma::load_matrix_sync(fa, sQ_rt + k, 136);
            wmma::load_matrix_sync(fb, sKV + k, 136);
            wmma::mma_sync(fc, fa, fb, fc);
        }
        asm volatile("bar.sync 1, 64;" ::: "memory");

        {
            const float* vp = (kr < block_n) ? (V_self + ((size_t) kr * n_head_kv + hkv) * 128 + kc0) : nullptr;
            __half2* dst2 = reinterpret_cast<__half2*>(sKV + kr * 136 + kc0);
            if (vp) {
#pragma unroll 1
                for (int i = 0; i < 8; ++i) {
                    const float4 v = *reinterpret_cast<const float4*>(vp + i * 4);
                    dst2[i * 2 + 0] = __halves2half2(__float2half(v.x), __float2half(v.y));
                    dst2[i * 2 + 1] = __halves2half2(__float2half(v.z), __float2half(v.w));
                }
            } else {
                const __half2 z2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));
#pragma unroll 1
                for (int i = 0; i < 16; ++i) dst2[i] = z2;
            }
        }
        if (acc_c0     >= block_n) { fc.x[0] = -1e30f; fc.x[2] = -1e30f; }
        if (acc_c0 + 1 >= block_n) { fc.x[1] = -1e30f; fc.x[3] = -1e30f; }
        if (acc_c1     >= block_n) { fc.x[4] = -1e30f; fc.x[6] = -1e30f; }
        if (acc_c1 + 1 >= block_n) { fc.x[5] = -1e30f; fc.x[7] = -1e30f; }
        const float mx0 = wmma_row_max_16(fmaxf(fmaxf(fc.x[0], fc.x[1]), fmaxf(fc.x[4], fc.x[5])));
        const float mx1 = wmma_row_max_16(fmaxf(fmaxf(fc.x[2], fc.x[3]), fmaxf(fc.x[6], fc.x[7])));
        const float m_new0 = fmaxf(rm0, mx0), m_new1 = fmaxf(rm1, mx1);
        const float a0 = (rm0 > -1e20f) ? ((m_new0 == rm0) ? 1.0f : dspark_fast_exp2f_hw(rm0 - m_new0)) : 0.0f;
        const float a1 = (rm1 > -1e20f) ? ((m_new1 == rm1) ? 1.0f : dspark_fast_exp2f_hw(rm1 - m_new1)) : 0.0f;
        if (__any_sync(0xffffffffu, a0 != 1.0f || a1 != 1.0f)) {
#pragma unroll
            for (int n = 0; n < 8; ++n) wmma_rescale_acc(acc[n], a0, a1);
        }
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa_p;
        float ps0 = 0.0f, ps1 = 0.0f;
        wmma_acc_to_matA_16x16<true>(fc, m_new0, m_new1, ps0, ps1, fa_p, lane);
        rm0 = m_new0; rl0 = rl0 * a0 + wmma_row_sum_16(ps0);
        rm1 = m_new1; rl1 = rl1 * a1 + wmma_row_sum_16(ps1);
        asm volatile("bar.sync 1, 64;" ::: "memory");

#pragma unroll
        for (int n = 0; n < 8; ++n) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
            wmma::load_matrix_sync(fb, sKV + n * 16, 136);
            wmma::mma_sync(acc[n], fa_p, fb, acc[n]);
        }
    }

    // Epilogue combine kw=0 & kw=1 for rt=0 and rt=1: s_out + s_tmp = 16,384 B at [0..16383], scalars at [17408..18431].
    __syncthreads();
    float* s_wm = reinterpret_cast<float*>(smem_raw + 17408); // [2][32] = 256 B
    float* s_wl = s_wm + 64;                                  // [2][32] = 256 B
    float* s_mg = s_wl + 64;                                  // [32]    = 128 B
    float* s_lg = s_mg + 32;                                  // [32]    = 128 B
    if (is_leader) {
        s_wm[kw * 32 + rt * 16 + acc_r0] = rm0; s_wl[kw * 32 + rt * 16 + acc_r0] = rl0;
        s_wm[kw * 32 + rt * 16 + acc_r1] = rm1; s_wl[kw * 32 + rt * 16 + acc_r1] = rl1;
    }
    __syncthreads();
    if (tid < n_rows) {
        float mg = -1e30f;
        if (s_wl[tid]      > 0.0f) mg = fmaxf(mg, s_wm[tid]);
        if (s_wl[32 + tid] > 0.0f) mg = fmaxf(mg, s_wm[32 + tid]);
        float lg = 0.0f;
        if (s_wl[tid]      > 0.0f) lg += s_wl[tid]      * dspark_fast_exp2f_hw(s_wm[tid]      - mg);
        if (s_wl[32 + tid] > 0.0f) lg += s_wl[32 + tid] * dspark_fast_exp2f_hw(s_wm[32 + tid] - mg);
        s_mg[tid] = mg;
        s_lg[tid] = lg;
    }
    __syncthreads();
    {
        const int r0 = rt * 16 + acc_r0, r1 = rt * 16 + acc_r1;
        const float cw0 = (r0 < n_rows && rl0 > 0.0f) ? dspark_fast_exp2f_hw(rm0 - s_mg[r0]) : 0.0f;
        const float cw1 = (r1 < n_rows && rl1 > 0.0f) ? dspark_fast_exp2f_hw(rm1 - s_mg[r1]) : 0.0f;
#pragma unroll
        for (int n = 0; n < 8; ++n) wmma_rescale_acc(acc[n], cw0, cw1);
    }
    float* s_out = reinterpret_cast<float*>(smem_raw);
    float* s_tmp = s_out + 16 * 128;
    for (int g_rt = 0; g_rt < 2; ++g_rt) {
        if (g_rt * 16 >= n_rows) break;
        if (rt == g_rt) {
            float* dst_s = (kw == 0) ? s_out : s_tmp;
#pragma unroll
            for (int n = 0; n < 8; ++n) wmma::store_matrix_sync(dst_s + n * 16, acc[n], 128, wmma::mem_row_major);
        }
        __syncthreads();
        const int r_lim = min(16, n_rows - g_rt * 16);
        for (int r = 0; r < r_lim; ++r) {
            const int rr = g_rt * 16 + r;
            float* dst = split_out + (((size_t) (rr / 3) * split_n + sp) * n_head + (h0 + (rr % 3))) * 130;
            dst[tid] = s_out[r * 128 + tid] + s_tmp[r * 128 + tid];
            if (tid == 0) {
                dst[128] = s_mg[rr];
                dst[129] = s_lg[rr];
            }
        }
        __syncthreads();
    }
#endif
}

/// Combine Split-KV partials for DSpark (+ optional direct Q8_1 store into xq_out): grid (n_head, block_n), block 128.
/// Reads P = *pos_d dynamically from device memory for CUDA Graph capture.
__global__ void k_dspark_split_combine(const float* __restrict__ split_out, float* __restrict__ out,
                                       int n_head, const int32_t* __restrict__ pos_d, int split_n,
                                       int is_swa = 0, int n_swa = 0, void* __restrict__ xq_out = nullptr) {
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const int P = pos_d[0];
    const int n_tok = is_swa ? min(P, n_swa - 1) : P;
    const int chunk_sz = max(64, (n_tok + split_n - 1) / split_n);
    const int active_splits = max(1, min(split_n, (n_tok + chunk_sz - 1) / chunk_sz));
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
    const float val = lg > 0.0f ? (og / lg) : 0.0f;
    out[((size_t) t * n_head + h) * 128 + d] = val;
    if (xq_out) {
        q8_1_warp_store(val, xq_out, (t * n_head + h) * 4 + (d >> 5), d & 31);
    }
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

__global__ void k_dspark_prep_joint_d(const float* __restrict__ state, const void* __restrict__ markov_w1, int w1_type,
                                      const int32_t* __restrict__ prev_id_d, const float* __restrict__ hidden_i,
                                      const float* __restrict__ context, float* __restrict__ joint_inp, int rank, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = 2 * rank + 2 * N;
    if (i >= total) return;
    if (i < rank) {
        joint_inp[i] = state[i];
    } else if (i < 2 * rank) {
        const int c = i - rank;
        const size_t idx = (size_t) (*prev_id_d) * rank + c;
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

/// Single-pass online-softmax argmax + exact top-1 softmax probability with in-register softcapping (reads logits ONCE).
/// Grid: (n_blk), Block: 256 -> writes per-block partials into p_val/p_idx/p_sum.
__global__ void k_dspark_argmax_prob_stage1(const float* __restrict__ logits,
                                            float* __restrict__ p_val, int32_t* __restrict__ p_idx,
                                            float* __restrict__ p_sum, int V, float softcap) {
    __shared__ float sv_raw[8];
    __shared__ int32_t si[8];
    __shared__ float sv_sc[8];
    __shared__ float sl[8];
    const int b = (int) blockIdx.x, n_blk = (int) gridDim.x;
    const int tid = (int) threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const float inv_sc = (softcap > 0.0f) ? (1.0f / softcap) : 0.0f;
    float mv_raw = -1e30f;
    int32_t mi = 0;
    float mv_sc = -1e30f;
    float l = 0.0f;
    auto push_elem = [&](float raw, int32_t idx) {
        if (raw > mv_raw || (raw == mv_raw && idx < mi)) { mv_raw = raw; mi = idx; }
        const float v = (softcap > 0.0f) ? (softcap * tanhf(raw * inv_sc)) : raw;
        if (v > mv_sc) {
            l = (l > 0.0f) ? (l * expf(mv_sc - v) + 1.0f) : 1.0f;
            mv_sc = v;
        } else {
            l += expf(v - mv_sc);
        }
    };
    const int stride = n_blk * 256;
    if ((((uintptr_t) logits) & 15u) == 0u) {
        const int v4 = V >> 2;
        const float4* row4 = reinterpret_cast<const float4*>(logits);
        for (int i4 = b * 256 + tid; i4 < v4; i4 += stride) {
            const float4 v = row4[i4];
            const int base = i4 << 2;
            push_elem(v.x, base);
            push_elem(v.y, base + 1);
            push_elem(v.z, base + 2);
            push_elem(v.w, base + 3);
        }
        if (b == 0) {
            const int tail0 = v4 << 2;
            for (int i = tail0 + tid; i < V; i += 256) push_elem(logits[i], i);
        }
    } else {
        for (int i = b * 256 + tid; i < V; i += stride) push_elem(logits[i], i);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float oraw = __shfl_xor_sync(0xffffffffu, mv_raw, o);
        const int32_t oi = __shfl_xor_sync(0xffffffffu, mi, o);
        const float osc = __shfl_xor_sync(0xffffffffu, mv_sc, o);
        const float ol = __shfl_xor_sync(0xffffffffu, l, o);
        if (oraw > mv_raw || (oraw == mv_raw && oi < mi)) { mv_raw = oraw; mi = oi; }
        const float mx = fmaxf(mv_sc, osc);
        l = (l > 0.0f ? l * expf(mv_sc - mx) : 0.0f) + (ol > 0.0f ? ol * expf(osc - mx) : 0.0f);
        mv_sc = mx;
    }
    if (lane == 0) {
        sv_raw[warp] = mv_raw;
        si[warp] = mi;
        sv_sc[warp] = mv_sc;
        sl[warp] = l;
    }
    __syncthreads();
    if (warp == 0) {
        mv_raw = (lane < 8) ? sv_raw[lane] : -1e30f;
        mi = (lane < 8) ? si[lane] : 0;
        mv_sc = (lane < 8) ? sv_sc[lane] : -1e30f;
        l = (lane < 8) ? sl[lane] : 0.0f;
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) {
            const float oraw = __shfl_xor_sync(0xffffffffu, mv_raw, o);
            const int32_t oi = __shfl_xor_sync(0xffffffffu, mi, o);
            const float osc = __shfl_xor_sync(0xffffffffu, mv_sc, o);
            const float ol = __shfl_xor_sync(0xffffffffu, l, o);
            if (oraw > mv_raw || (oraw == mv_raw && oi < mi)) { mv_raw = oraw; mi = oi; }
            const float mx = fmaxf(mv_sc, osc);
            l = (l > 0.0f ? l * expf(mv_sc - mx) : 0.0f) + (ol > 0.0f ? ol * expf(osc - mx) : 0.0f);
            mv_sc = mx;
        }
        if (lane == 0) {
            p_val[b] = mv_sc;
            p_idx[b] = mi;
            p_sum[b] = l;
        }
    }
}

/// Stage 2: reduce n_blk partials to global (out_id[0], out_prob[0]). Grid: 1, Block: 32.
__global__ void k_dspark_argmax_prob_stage2(const float* __restrict__ p_val, const int32_t* __restrict__ p_idx,
                                            const float* __restrict__ p_sum,
                                            int32_t* __restrict__ out_id, float* __restrict__ out_prob, int n_blk) {
    const int lane = (int) threadIdx.x;
    float mv_sc = -1e30f;
    int32_t mi = 0;
    float l = 0.0f;
    for (int i = lane; i < n_blk; i += 32) {
        const float v = p_val[i];
        const int32_t idx = p_idx[i];
        const float ps = p_sum[i];
        const float mx = fmaxf(mv_sc, v);
        if (v > mv_sc || (v == mv_sc && idx < mi)) mi = idx;
        l = (l > 0.0f ? l * expf(mv_sc - mx) : 0.0f) + (ps > 0.0f ? ps * expf(v - mx) : 0.0f);
        mv_sc = mx;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float osc = __shfl_xor_sync(0xffffffffu, mv_sc, o);
        const int32_t oi = __shfl_xor_sync(0xffffffffu, mi, o);
        const float ol = __shfl_xor_sync(0xffffffffu, l, o);
        if (osc > mv_sc || (osc == mv_sc && oi < mi)) mi = oi;
        const float mx = fmaxf(mv_sc, osc);
        l = (l > 0.0f ? l * expf(mv_sc - mx) : 0.0f) + (ol > 0.0f ? ol * expf(osc - mx) : 0.0f);
        mv_sc = mx;
    }
    if (lane == 0) {
        out_id[0] = mi;
        out_prob[0] = (l > 0.0f) ? (1.0f / l) : 1.0f;
    }
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
    float fuse_w_host[4][4] = {};       // Host-precomputed exact FP32 softmax of dspark_fuse rows 0..3
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

    // EX-4A: Contiguous Decode-only weight arena (evicted to pinned RAM during Prefill on Hybrid GPUs -> +358 MiB!)
    struct DecodeWeightRef {
        void** ptr_slot = nullptr;
        size_t offset = 0;
    };
    uint8_t* decode_w_gpu = nullptr;
    uint8_t* decode_w_pin = nullptr;
    bool decode_w_pin_cuda = false;
    bool decode_w_from_pool = false;
    size_t decode_w_bytes = 0;
    bool decode_w_offloadable = false;
    bool decode_w_resident = true;
    cudaStream_t decode_w_alloc_st = nullptr;
    cudaEvent_t ev_decode_w_alloc = nullptr;
    cudaEvent_t ev_decode_w_ready = nullptr;
    bool decode_w_h2d_pending = false;
    std::vector<DecodeWeightRef> decode_w_refs;

    void bind_decode_weight_ptrs(uint8_t* base) {
        for (const auto& r : decode_w_refs) {
            *r.ptr_slot = base ? (void*) (base + r.offset) : nullptr;
        }
    }

    void evict_decode_weights(cudaMemPool_t pool = nullptr, cudaStream_t pool_st = nullptr) {
        if (!decode_w_offloadable || (!decode_w_resident && !decode_w_gpu)) return;
        if (decode_w_gpu) {
            if (decode_w_from_pool && pool != nullptr) {
                if (decode_w_h2d_pending && ev_decode_w_ready) {
                    CK(cudaStreamWaitEvent(pool_st, ev_decode_w_ready, 0));
                }
                CK(cudaFreeAsync(decode_w_gpu, pool_st));
            } else {
                cudaFree(decode_w_gpu);
            }
            decode_w_gpu = nullptr;
            decode_w_from_pool = false;
            decode_w_alloc_st = nullptr;
        }
        bind_decode_weight_ptrs(nullptr);
        decode_w_resident = false;
        decode_w_h2d_pending = false;
    }

    // Stream-ordered allocation from `pool` on `pool_st` when provided (falls back to cudaMalloc).
    // Records `ev_decode_w_alloc` on `pool_st` so a subsequent H2D on a different stream (e.g. `cst`)
    // automatically waits on the pool allocation before writing `decode_w_gpu`.
    bool alloc_decode_weights(cudaMemPool_t pool = nullptr, cudaStream_t pool_st = nullptr) {
        if (!decode_w_offloadable || decode_w_resident) return false;
        if (!decode_w_gpu) {
            void* p = nullptr;
            if (pool != nullptr &&
                cudaMallocFromPoolAsync(&p, decode_w_bytes, pool, pool_st) == cudaSuccess && p != nullptr) {
                decode_w_gpu = reinterpret_cast<uint8_t*>(p);
                decode_w_from_pool = true;
                decode_w_alloc_st = pool_st;
                if (!ev_decode_w_alloc) CK(cudaEventCreateWithFlags(&ev_decode_w_alloc, cudaEventDisableTiming));
                CK(cudaEventRecord(ev_decode_w_alloc, pool_st));
            } else {
                if (pool != nullptr) (void) cudaGetLastError();
                decode_w_gpu = nullptr;
                CK(cudaMalloc(&decode_w_gpu, decode_w_bytes));
                decode_w_from_pool = false;
                decode_w_alloc_st = nullptr;
            }
            bind_decode_weight_ptrs(decode_w_gpu);
        }
        return true;
    }

    void queue_decode_weights_h2d(cudaStream_t s, bool async_stream = false,
                                  cudaMemPool_t pool = nullptr, cudaStream_t pool_st = nullptr) {
        if (!decode_w_offloadable || decode_w_resident) return;
        if (!decode_w_gpu) alloc_decode_weights(pool, pool ? (pool_st ? pool_st : s) : nullptr);
        // Critical sync safety: when `decode_w_gpu` was stream-allocated on `pool_st` (e.g. `st`)
        // and H2D runs on another non-blocking stream `s` (e.g. `cst`), order `s` after `ev_decode_w_alloc`.
        if (decode_w_from_pool && s != decode_w_alloc_st && ev_decode_w_alloc) {
            CK(cudaStreamWaitEvent(s, ev_decode_w_alloc, 0));
        }
        const auto t0 = Clock::now();
        CK(cudaMemcpyAsync(decode_w_gpu, decode_w_pin, decode_w_bytes, cudaMemcpyHostToDevice, s));
        if (async_stream) {
            if (!ev_decode_w_ready) CK(cudaEventCreateWithFlags(&ev_decode_w_ready, cudaEventDisableTiming));
            CK(cudaEventRecord(ev_decode_w_ready, s));
            decode_w_h2d_pending = true;
        } else {
            decode_w_h2d_pending = false;
        }
        decode_w_resident = true;
        std::fprintf(stderr, "strata-inkling2: dspark decode weights restored (%.1f MiB, async=%d, pool=%d) queued in %.1f ms\n",
                     (double) decode_w_bytes / (1 << 20), async_stream ? 1 : 0, decode_w_from_pool ? 1 : 0, ms_since(t0));
    }

    void restore_decode_weights(cudaStream_t st, cudaMemPool_t pool = nullptr, cudaStream_t pool_st = nullptr) {
        queue_decode_weights_h2d(st, false, pool, pool ? (pool_st ? pool_st : st) : nullptr);
    }

    ~DsparkModel() {
        if (ev_decode_w_alloc) { cudaEventDestroy(ev_decode_w_alloc); ev_decode_w_alloc = nullptr; }
        if (ev_decode_w_ready) { cudaEventDestroy(ev_decode_w_ready); ev_decode_w_ready = nullptr; }
        if (decode_w_gpu) {
            // Pool-backed decode_w_gpu must be evicted by Engine::~Engine() before destroying phase_pool.
            if (!decode_w_from_pool) cudaFree(decode_w_gpu);
            decode_w_gpu = nullptr;
            decode_w_from_pool = false;
            decode_w_alloc_st = nullptr;
        }
        if (decode_w_pin) {
            if (decode_w_pin_cuda) cudaFreeHost(decode_w_pin);
            else std::free(decode_w_pin);
            decode_w_pin = nullptr;
        }
        free_store();
    }

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
    {
        const int n_fuse_rows = std::min(hp.n_layer + 1, 4);
        float raw_fuse[4 * 4] = {};
        CK(cudaMemcpy(raw_fuse, dm.dspark_fuse, (size_t) (n_fuse_rows * 4) * sizeof(float), cudaMemcpyDeviceToHost));
        for (int r = 0; r < n_fuse_rows; ++r) {
            float a[4] = {raw_fuse[r * 4 + 0], raw_fuse[r * 4 + 1], raw_fuse[r * 4 + 2], raw_fuse[r * 4 + 3]};
            const float mx = fmaxf(fmaxf(a[0], a[1]), fmaxf(a[2], a[3]));
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) { a[k] = expf(a[k] - mx); s += a[k]; }
            const float inv = 1.0f / s;
            for (int k = 0; k < 4; ++k) dm.fuse_w_host[r][k] = a[k] * inv;
        }
    }
    dm.dspark_norm = dm.upload_f32("dspark_norm.weight", (size_t) N);
    dm.mask_embd = dm.upload_f32("mask_embd.weight", (size_t) N);
    dm.output_norm = dm.upload_f32("output_norm.weight", (size_t) N);
    dm.joint_proj_b = dm.upload_f32("joint_proj.bias", (size_t) (3 * rank));
    dm.hidden_out_b = dm.upload_f32("dspark_hidden_out.bias", (size_t) N);

    size_t fr_vram = 0, tot_vram = 0;
    const bool pack_dspark_q8 = (cudaMemGetInfo(&fr_vram, &tot_vram) == cudaSuccess && tot_vram >= (20ull << 30));
    const bool use_decode_arena = !pack_dspark_q8 && (std::getenv("STRATA_NO_PHASE_SPLIT") == nullptr);

    struct StagedDecodeEntry {
        const uint8_t* raw = nullptr;
        size_t bytes = 0;
        size_t offset = 0;
        void** ptr_slot = nullptr;
    };
    std::vector<StagedDecodeEntry> staged;
    size_t staged_bytes = 0;
    auto align256 = [](size_t x) -> size_t { return (x + 255u) & ~size_t(255u); };

    auto stage_dense = [&](const char* name, size_t rows_override, size_t cols_expected, DenseMat& dst) {
        if (!use_decode_arena) {
            dst = dm.upload_dense(name, rows_override, cols_expected, false, pack_dspark_q8);
            return;
        }
        const strata::TensorInfo* t = nullptr;
        const uint8_t* raw = dm.find_tensor(name, &t);
        if (!raw || !t || (t->type != 0 && t->type != 1 && t->type != 30 && !K::native_mmvq_supported((int) t->type))) {
            std::fprintf(stderr, "strata-inkling2: missing/unsupported dspark dense tensor %s (type=%u)\n", name, t ? t->type : 999u);
            std::exit(1);
        }
        const int64_t ne0 = (int64_t) t->shape[0];
        const int64_t ne1 = rows_override ? (int64_t) rows_override : (t->shape.size() > 1 ? (int64_t) t->shape[1] : 1);
        if (cols_expected && (size_t) ne0 != cols_expected) {
            std::fprintf(stderr, "strata-inkling2: %s ne0 expected %zu got %lld\n", name, cols_expected, (long long) ne0);
            std::exit(1);
        }
        const size_t rb = (t->type == 0) ? (size_t) ne0 * sizeof(float)
                        : (t->type == 1 || t->type == 30) ? (size_t) ne0 * sizeof(uint16_t)
                        : strata::quant::row_bytes((int) t->type, ne0);
        const size_t bytes = rb * (size_t) ne1;
        if (t->type != 30 && t->type != 0 && t->type != 1) dm.has_quant_dense = true;
        dst = DenseMat{(int) t->type, ne0, ne1, nullptr, bytes, false};
        staged.push_back(StagedDecodeEntry{raw, bytes, staged_bytes, &dst.d});
        staged_bytes += align256(bytes);
    };

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
        if (use_decode_arena) {
            staged.push_back(StagedDecodeEntry{raw, bytes, staged_bytes, &dm.markov_w1});
            staged_bytes += align256(bytes);
        } else {
            void* d = dm.gpu_alloc<uint8_t>(bytes);
            CK(cudaMemcpy(d, raw, bytes, cudaMemcpyHostToDevice));
            dm.markov_w1 = d;
        }
    }

    stage_dense("joint_proj.weight", (size_t) (3 * rank), (size_t) (2 * rank + 2 * N), dm.joint_proj);
    stage_dense("dspark_hidden_out.weight", (size_t) N, (size_t) rank, dm.hidden_out);

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

        // inject()-required context projections stay ALWAYS resident on GPU (~14-27 MiB)
        L.attn_k_ctx = dm.upload_dense(tn("attn_k_ctx.weight"), (size_t) hp.kv_dim, (size_t) N, false, pack_dspark_q8);
        L.attn_v_ctx = dm.upload_dense(tn("attn_v_ctx.weight"), (size_t) hp.kv_dim, (size_t) N, false, pack_dspark_q8);

        // Decode-only DFlare weights -> packed into offloadable arena on Hybrid GPUs
        stage_dense(tn("attn_q.weight"), (size_t) hp.q_dim, (size_t) N, L.attn_q);
        stage_dense(tn("attn_k.weight"), (size_t) hp.kv_dim, (size_t) N, L.attn_k);
        stage_dense(tn("attn_v.weight"), (size_t) hp.kv_dim, (size_t) N, L.attn_v);
        stage_dense(tn("attn_output.weight"), (size_t) N, (size_t) hp.q_dim, L.attn_output);
        stage_dense(tn("ffn_gate.weight"), (size_t) hp.n_ff, (size_t) N, L.ffn_gate);
        stage_dense(tn("ffn_up.weight"), (size_t) hp.n_ff, (size_t) N, L.ffn_up);
        stage_dense(tn("ffn_down.weight"), (size_t) N, (size_t) hp.n_ff, L.ffn_down);
    }

    if (use_decode_arena && staged_bytes > 0) {
        dm.decode_w_bytes = staged_bytes;
        if (cudaMallocHost(&dm.decode_w_pin, staged_bytes) == cudaSuccess) {
            dm.decode_w_pin_cuda = true;
        } else {
            (void) cudaGetLastError();
            dm.decode_w_pin = (uint8_t*) std::malloc(staged_bytes);
            dm.decode_w_pin_cuda = false;
        }
        dm.decode_w_refs.reserve(staged.size());
        for (const auto& e : staged) {
            std::memcpy(dm.decode_w_pin + e.offset, e.raw, e.bytes);
            dm.decode_w_refs.push_back(DsparkModel::DecodeWeightRef{e.ptr_slot, e.offset});
        }
        // Upload initially so resolve_inkling2_profile() sees truthful Decode VRAM headroom;
        // Hybrid mode evicts before place_experts() and restores in set_phase_decode().
        CK(cudaMalloc(&dm.decode_w_gpu, staged_bytes));
        CK(cudaMemcpy(dm.decode_w_gpu, dm.decode_w_pin, staged_bytes, cudaMemcpyHostToDevice));
        dm.bind_decode_weight_ptrs(dm.decode_w_gpu);
        dm.decode_w_from_pool = false;
        dm.decode_w_offloadable = true;
        dm.decode_w_resident = true;
    }

    std::fprintf(stderr, "strata-inkling2: loaded dspark draft (%d layers, block=%d, rank=%d, decode_arena=%.1f MiB) in %.2f s\n",
                 hp.n_layer, hp.block_size, hp.markov_rank,
                 (double) dm.decode_w_bytes / (1 << 20), ms_since(t0) / 1000.0);
    return true;
}

struct DsparkEngine {
    DsparkModel* dm = nullptr;
    int ctx = 0;
    int tmax = 0;
    int verify_max_t = 8;
    int pos = 0;

    // Context Q8 KV cache per layer
    std::vector<int> kv_cap;
    std::vector<int8_t*> kq, vq;
    std::vector<__half*> ks, vs;

    // Inject & draft scratch
    float* ctx_norm = nullptr;       // [tmax * N]
    float* ctx_k = nullptr;          // [tmax * kv_dim]
    float* ctx_v = nullptr;          // [tmax * kv_dim]
    float* f_rnn_last = nullptr;     // [N]

    float* block_cur = nullptr;      // [block_size * N]
    float* x_norm = nullptr;         // [block_size * N]
    float* Q = nullptr;              // [block_size * q_dim]
    float* K = nullptr;              // [block_size * kv_dim]
    float* V = nullptr;              // [block_size * kv_dim]
    float* att = nullptr;            // [block_size * q_dim]
    float* split_out = nullptr;      // [block_size * split_full * n_head * 130]
    int split_full = 120;
    float* wo = nullptr;             // [block_size * N]
    float* gate = nullptr;           // [block_size * n_ff]
    float* up = nullptr;             // [block_size * n_ff]

    float* rnn_state = nullptr;      // [rank]
    float* joint_inp = nullptr;      // [2*rank + 2*N]
    float* joint_out = nullptr;      // [3*rank]
    float* rnn_latent = nullptr;     // [rank]
    float* rnn_corr = nullptr;       // [N]
    float* h_pred = nullptr;         // [N]
    int32_t* pos_d = nullptr;        // [1] device pos for CUDA Graph DFlare / inject capture
    int32_t* h_pos_pin = nullptr;    // [1] pinned host pos for zero-sync DFlare H2D
    int32_t* h_inj_pos_pin = nullptr;// [1] separate pinned host pos for inject H2D (no race with subsequent draft()!)
    int32_t* d_draft_ids = nullptr;  // [8] device token chain [anchor, d0, d1, ...] for zero-sync Markov graph
    int32_t* h_draft_ids_pin = nullptr; // pinned [8] host staging for d_draft_ids
    float* d_prob_sum = nullptr;        // [64] device softmax sum partials for per-step confidence
    float* d_draft_prob = nullptr;      // [8] device top-1 probability per drafted token
    float* inv_freq_d = nullptr;        // [64] precomputed NeoX RoPE inv_freq for fast KV store
    float* h_draft_prob_pin = nullptr;  // pinned [8] host staging for d_draft_prob
    float* h_anchor_pin = nullptr;   // pinned [N] host buffer for async anchor upload
    cudaGraphExec_t dflare_graph_short = nullptr; // captured 3-layer DFlare block (pos < 256)
    cudaGraphExec_t dflare_graph_split = nullptr; // captured 3-layer DFlare block (pos >= 256, WMMA Split-KV)
    cudaGraphExec_t markov_graphs[8] = {};        // captured greedy Markov chains indexed by n_draft (1..7)
    bool markov_graph_oom[8] = {};
    cudaGraphExec_t markov_step_graphs[7] = {};   // per-step captured Markov graph with argmax + prob
    bool markov_step_oom[7] = {};
    cudaGraphExec_t inject_graphs[9] = {};        // captured inject(T=1..8, feat_off=0) graphs
    bool inject_graph_oom[9] = {};
    std::vector<cudaGraphExec_t> deferred_graphs_destroy;
    std::vector<float> h_anchor;     // [N]
    bool ctx_aliased = false;
    bool split_aliased = false;

    // Wave 5: Zero-PCIe Running VRAM Accumulator state (prefill T > 8 only; decode/verify T <= 8 stays on d_tgt_feat)
    float fuse_w_host[4][4] = {};
    bool prefill_vram_acc_active = false;
    float* d_pp_acc[3] = {nullptr, nullptr, nullptr}; // [0]=SWA L0, [1]=Full L1 (aliases E.msg3), [2]=SWA L2
    int pp_acc_swa_start = 0;

    // Stats
    int64_t n_draft_calls = 0;
    int64_t n_drafted = 0;
    int64_t n_accepted = 0;
    int64_t pld_calls = 0;
    int64_t pld_drafted = 0;
    int64_t pld_accepted = 0;
    double draft_ms = 0.0;
    double inject_ms = 0.0;

    void destroy_kv_graphs() {
        if (dflare_graph_short) { cudaGraphExecDestroy(dflare_graph_short); dflare_graph_short = nullptr; }
        if (dflare_graph_split) { cudaGraphExecDestroy(dflare_graph_split); dflare_graph_split = nullptr; }
        for (int i = 0; i < 9; ++i) {
            if (inject_graphs[i]) { cudaGraphExecDestroy(inject_graphs[i]); inject_graphs[i] = nullptr; }
            inject_graph_oom[i] = false;
        }
    }

    void destroy_decode_graphs() {
        if (dflare_graph_short) { cudaGraphExecDestroy(dflare_graph_short); dflare_graph_short = nullptr; }
        if (dflare_graph_split) { cudaGraphExecDestroy(dflare_graph_split); dflare_graph_split = nullptr; }
        for (int i = 0; i < 8; ++i) {
            if (markov_graphs[i]) { cudaGraphExecDestroy(markov_graphs[i]); markov_graphs[i] = nullptr; }
            markov_graph_oom[i] = false;
        }
        for (int i = 0; i < 7; ++i) {
            if (markov_step_graphs[i]) { cudaGraphExecDestroy(markov_step_graphs[i]); markov_step_graphs[i] = nullptr; }
            markov_step_oom[i] = false;
        }
    }

    void defer_destroy_kv_graphs() {
        auto defer = [&](cudaGraphExec_t& g) {
            if (g) { deferred_graphs_destroy.push_back(g); g = nullptr; }
        };
        defer(dflare_graph_short);
        defer(dflare_graph_split);
        for (int i = 0; i < 9; ++i) {
            defer(inject_graphs[i]);
            inject_graph_oom[i] = false;
        }
    }

    void defer_destroy_decode_graphs() {
        auto defer = [&](cudaGraphExec_t& g) {
            if (g) { deferred_graphs_destroy.push_back(g); g = nullptr; }
        };
        defer(dflare_graph_short);
        defer(dflare_graph_split);
        for (int i = 0; i < 8; ++i) {
            defer(markov_graphs[i]);
            markov_graph_oom[i] = false;
        }
        for (int i = 0; i < 7; ++i) {
            defer(markov_step_graphs[i]);
            markov_step_oom[i] = false;
        }
    }

    void flush_deferred_graphs() {
        if (deferred_graphs_destroy.empty()) return;
        for (cudaGraphExec_t g : deferred_graphs_destroy) {
            if (g) cudaGraphExecDestroy(g);
        }
        deferred_graphs_destroy.clear();
    }

    void evict_decode_weights(cudaMemPool_t pool = nullptr, cudaStream_t pool_st = nullptr) {
        if (!dm || !dm->decode_w_offloadable || (!dm->decode_w_resident && !dm->decode_w_gpu)) return;
        defer_destroy_decode_graphs();
        dm->evict_decode_weights(pool, pool_st);
    }

    bool alloc_decode_weights(cudaMemPool_t pool = nullptr, cudaStream_t pool_st = nullptr) {
        if (!dm || !dm->decode_w_offloadable || dm->decode_w_resident) return false;
        defer_destroy_decode_graphs();
        return dm->alloc_decode_weights(pool, pool_st);
    }

    void queue_decode_weights_h2d(cudaStream_t s, bool async_stream = false,
                                  cudaMemPool_t pool = nullptr, cudaStream_t pool_st = nullptr) {
        if (!dm || !dm->decode_w_offloadable || dm->decode_w_resident) return;
        defer_destroy_decode_graphs();
        dm->queue_decode_weights_h2d(s, async_stream, pool, pool_st);
    }

    void wait_decode_weights_ready(cudaStream_t st) {
        if (dm && dm->decode_w_h2d_pending && dm->ev_decode_w_ready) {
            CK(cudaStreamWaitEvent(st, dm->ev_decode_w_ready, 0));
            dm->decode_w_h2d_pending = false;
        }
    }

    void restore_decode_weights(cudaStream_t st, cudaMemPool_t pool = nullptr, cudaStream_t pool_st = nullptr) {
        if (!dm || !dm->decode_w_offloadable || dm->decode_w_resident) return;
        defer_destroy_decode_graphs();
        dm->restore_decode_weights(st, pool, pool_st);
    }

    void ensure_decode_weights_resident(cudaStream_t st, cudaMemPool_t pool = nullptr, cudaStream_t pool_st = nullptr) {
        restore_decode_weights(st, pool, pool_st);
    }

    size_t arena_base_bytes() const {
        if (!dm) return (size_t) 16 << 20;
        const size_t raw = (size_t) std::max(1, verify_max_t) * (size_t) dm->hp.vocab_unpadded * sizeof(float);
        return (raw + 255u) & ~size_t(255u);
    }

    size_t tail_bytes(int cur_tmax) const {
        if (!dm) return (size_t) 12 << 20;
        const auto& hp = dm->hp;
        auto align256 = [](size_t x) -> size_t { return (x + 255u) & ~size_t(255u); };
        const int eff_t = std::max(cur_tmax, 16);
        const size_t sz_ctx = align256((size_t) eff_t * hp.n_embd * sizeof(float))
                            + 2 * align256((size_t) eff_t * hp.kv_dim * sizeof(float));
        const size_t sz_sp = align256((size_t) hp.block_size * split_full * hp.n_head * 130 * sizeof(float));
        return std::max(sz_ctx, sz_sp);
    }

    /// Rebind aliased arena buffers (ctx_norm/k/v and split_out) when Engine switches Prefill <-> Decode arena.
    void rebind_arena(uint8_t* ext_arena, size_t ext_arena_bytes, int cur_tmax) {
        defer_destroy_kv_graphs();
        defer_destroy_decode_graphs();
        if (!ctx_aliased && !split_aliased) return;
        const auto& hp = dm->hp;
        auto align256 = [](size_t x) -> size_t { return (x + 255u) & ~size_t(255u); };
        const size_t kArenaBase = arena_base_bytes();
        const int eff_t = std::max(cur_tmax, 16);
        const size_t sz_cn = align256((size_t) eff_t * hp.n_embd * sizeof(float));
        const size_t sz_ck = align256((size_t) eff_t * hp.kv_dim * sizeof(float));
        const size_t sz_cv = align256((size_t) eff_t * hp.kv_dim * sizeof(float));
        if (ctx_aliased) {
            if (!ext_arena || ext_arena_bytes < kArenaBase + sz_cn + sz_ck + sz_cv) {
                std::fprintf(stderr, "strata-inkling2: dspark rebind_arena ctx overflow (%zu < %zu)\n",
                             ext_arena_bytes, kArenaBase + sz_cn + sz_ck + sz_cv);
                std::exit(1);
            }
            ctx_norm = reinterpret_cast<float*>(ext_arena + kArenaBase);
            ctx_k    = reinterpret_cast<float*>(ext_arena + kArenaBase + sz_cn);
            ctx_v    = reinterpret_cast<float*>(ext_arena + kArenaBase + sz_cn + sz_ck);
        }
        const size_t sz_sp = align256((size_t) hp.block_size * split_full * hp.n_head * 130 * sizeof(float));
        if (split_aliased) {
            if (!ext_arena || ext_arena_bytes < kArenaBase + sz_sp) {
                std::fprintf(stderr, "strata-inkling2: dspark rebind_arena split overflow (%zu < %zu)\n",
                             ext_arena_bytes, kArenaBase + sz_sp);
                std::exit(1);
            }
            split_out = reinterpret_cast<float*>(ext_arena + kArenaBase);
        }
    }

    ~DsparkEngine() {
        destroy_kv_graphs();
        destroy_decode_graphs();
        flush_deferred_graphs();
        for (auto* p : kq) cudaFree(p);
        for (auto* p : vq) cudaFree(p);
        for (auto* p : ks) cudaFree(p);
        for (auto* p : vs) cudaFree(p);
        if (!ctx_aliased) {
            cudaFree(ctx_norm);
            cudaFree(ctx_k);
            cudaFree(ctx_v);
        }
        cudaFree(f_rnn_last);
        cudaFree(block_cur);
        cudaFree(x_norm);
        cudaFree(Q);
        cudaFree(K);
        cudaFree(V);
        cudaFree(att);
        if (!split_aliased) cudaFree(split_out);
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
        cudaFree(d_draft_ids);
        cudaFree(d_prob_sum);
        cudaFree(d_draft_prob);
        cudaFree(inv_freq_d);
        if (h_pos_pin) cudaFreeHost(h_pos_pin);
        if (h_inj_pos_pin) cudaFreeHost(h_inj_pos_pin);
        if (h_draft_ids_pin) cudaFreeHost(h_draft_ids_pin);
        if (h_draft_prob_pin) cudaFreeHost(h_draft_prob_pin);
        if (h_anchor_pin) cudaFreeHost(h_anchor_pin);
    }

    void init(DsparkModel& model, int max_ctx, int max_t,
              int init_full_cap = 0, uint8_t* ext_arena = nullptr, size_t ext_arena_bytes = 0,
              int max_verify_t = 8) {
        dm = &model;
        ctx = max_ctx;
        tmax = std::max(max_t, 16);
        verify_max_t = std::max(1, max_verify_t);
        const auto& hp = dm->hp;
        kq.resize((size_t) hp.n_layer, nullptr);
        vq.resize((size_t) hp.n_layer, nullptr);
        ks.resize((size_t) hp.n_layer, nullptr);
        vs.resize((size_t) hp.n_layer, nullptr);
        kv_cap.resize((size_t) hp.n_layer, 0);
        const int full_init = (init_full_cap > 0) ? std::min(max_ctx, init_full_cap) : max_ctx;
        for (int il = 0; il < hp.n_layer; ++il) {
            const int cap = hp.is_swa(il) ? hp.n_swa : full_init;
            kv_cap[(size_t) il] = cap;
            CK(cudaMalloc(&kq[(size_t) il], (size_t) cap * hp.kv_dim * sizeof(int8_t)));
            CK(cudaMalloc(&vq[(size_t) il], (size_t) cap * hp.kv_dim * sizeof(int8_t)));
            CK(cudaMalloc(&ks[(size_t) il], (size_t) cap * hp.n_head_kv * sizeof(__half)));
            CK(cudaMalloc(&vs[(size_t) il], (size_t) cap * hp.n_head_kv * sizeof(__half)));
        }
        auto d_f32 = [](float** p, size_t n) { CK(cudaMalloc(p, n * sizeof(float))); };
        auto align256 = [](size_t x) -> size_t { return (x + 255u) & ~size_t(255u); };
        const int B = hp.block_size;
        const size_t kArenaBase = arena_base_bytes();  // keep clear of E.logits at arena offset 0
        const size_t sz_cn = align256((size_t) tmax * hp.n_embd * sizeof(float));
        const size_t sz_ck = align256((size_t) tmax * hp.kv_dim * sizeof(float));
        const size_t sz_cv = align256((size_t) tmax * hp.kv_dim * sizeof(float));
        if (ext_arena && ext_arena_bytes >= kArenaBase + sz_cn + sz_ck + sz_cv) {
            ctx_norm = reinterpret_cast<float*>(ext_arena + kArenaBase);
            ctx_k    = reinterpret_cast<float*>(ext_arena + kArenaBase + sz_cn);
            ctx_v    = reinterpret_cast<float*>(ext_arena + kArenaBase + sz_cn + sz_ck);
            ctx_aliased = true;
        } else {
            d_f32(&ctx_norm, (size_t) tmax * hp.n_embd);
            d_f32(&ctx_k, (size_t) tmax * hp.kv_dim);
            d_f32(&ctx_v, (size_t) tmax * hp.kv_dim);
            ctx_aliased = false;
        }
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
        split_full = std::clamp((n_sm * 12) / std::max(1, hp.n_head_kv), 40, 120);
        const size_t sz_sp = align256((size_t) B * split_full * hp.n_head * 130 * sizeof(float));
        if (ext_arena && ext_arena_bytes >= kArenaBase + sz_sp) {
            split_out = reinterpret_cast<float*>(ext_arena + kArenaBase);
            split_aliased = true;
        } else {
            d_f32(&split_out, (size_t) B * split_full * hp.n_head * 130);
            split_aliased = false;
        }
        cudaFuncSetAttribute(dspark_detail::k_dspark_attn_q8_wmma_split<false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 24576);
        cudaFuncSetAttribute(dspark_detail::k_dspark_attn_q8_wmma_split<false>,
                             cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        cudaFuncSetAttribute(dspark_detail::k_dspark_attn_q8_wmma_split<true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 24576);
        cudaFuncSetAttribute(dspark_detail::k_dspark_attn_q8_wmma_split<true>,
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
        if (cudaMallocHost(&h_pos_pin, sizeof(int32_t)) != cudaSuccess) {
            (void) cudaGetLastError();
            h_pos_pin = nullptr;
        }
        if (cudaMallocHost(&h_inj_pos_pin, sizeof(int32_t)) != cudaSuccess) {
            (void) cudaGetLastError();
            h_inj_pos_pin = nullptr;
        }
        CK(cudaMalloc(&d_draft_ids, 8 * sizeof(int32_t)));
        if (cudaMallocHost(&h_draft_ids_pin, 8 * sizeof(int32_t)) != cudaSuccess) {
            (void) cudaGetLastError();
            h_draft_ids_pin = nullptr;
        }
        CK(cudaMalloc(&d_prob_sum, 64 * sizeof(float)));
        CK(cudaMalloc(&d_draft_prob, 8 * sizeof(float)));
        CK(cudaMalloc(&inv_freq_d, 64 * sizeof(float)));
        {
            const int n_half_rot = std::min(64, std::max(1, hp.n_rot / 2));
            std::vector<float> h_if(64, 0.0f);
            for (int d = 0; d < n_half_rot; ++d) {
                h_if[(size_t) d] = std::pow((float) hp.freq_base, -2.0f * (float) d / (float) hp.n_rot);
            }
            CK(cudaMemcpy(inv_freq_d, h_if.data(), 64 * sizeof(float), cudaMemcpyHostToDevice));
        }
        if (cudaMallocHost(&h_draft_prob_pin, 8 * sizeof(float)) != cudaSuccess) {
            (void) cudaGetLastError();
            h_draft_prob_pin = nullptr;
        }
        if (cudaMallocHost(&h_anchor_pin, (size_t) hp.n_embd * sizeof(float)) != cudaSuccess) {
            (void) cudaGetLastError();
            h_anchor_pin = nullptr;
        }
        h_anchor.resize((size_t) hp.n_embd);
        std::memcpy(fuse_w_host, dm->fuse_w_host, sizeof(fuse_w_host));
        prefill_vram_acc_active = false;
        d_pp_acc[0] = d_pp_acc[1] = d_pp_acc[2] = nullptr;
        pp_acc_swa_start = 0;
        pos = 0;
    }

    void reset(cudaStream_t st) {
        pos = 0;
        prefill_vram_acc_active = false;
        CK(cudaMemsetAsync(f_rnn_last, 0, (size_t) dm->hp.n_embd * sizeof(float), st));
    }

    int count_full_layers() const {
        int n = 0;
        for (int il = 0; il < dm->hp.n_layer; ++il) {
            if (!dm->hp.is_swa(il)) ++n;
        }
        return n;
    }

    void grow_full_kv(int new_cap, cudaStream_t st) {
        const auto& hp = dm->hp;
        const int nhkv = hp.n_head_kv;
        bool grew = false;
        for (int il = 0; il < hp.n_layer; ++il) {
            if (hp.is_swa(il)) continue;
            const int old_cap = kv_cap[(size_t) il];
            if (new_cap <= old_cap) continue;
            if (!grew) {
                destroy_kv_graphs();
                grew = true;
            }
            const size_t n_old = (size_t) std::min<int64_t>(pos, old_cap);
            const size_t b_kv_head = n_old * 128 * sizeof(int8_t);
            const size_t b_s_head = n_old * sizeof(__half);
            const size_t old_kv_stride = (size_t) old_cap * 128;
            const size_t new_kv_stride = (size_t) new_cap * 128;
            const size_t per_tok = (size_t) hp.kv_dim * 2 + (size_t) nhkv * sizeof(__half) * 2;
            size_t fr = 0, tot = 0;
            CK(cudaMemGetInfo(&fr, &tot));
            const bool direct_d2d = (fr >= (size_t) new_cap * per_tok + ((size_t) 64 << 20));
            if (direct_d2d) {
                int8_t *nkq = nullptr, *nvq = nullptr;
                __half *nks = nullptr, *nvs = nullptr;
                CK(cudaMalloc(&nkq, (size_t) new_cap * hp.kv_dim * sizeof(int8_t)));
                CK(cudaMalloc(&nvq, (size_t) new_cap * hp.kv_dim * sizeof(int8_t)));
                CK(cudaMalloc(&nks, (size_t) new_cap * nhkv * sizeof(__half)));
                CK(cudaMalloc(&nvs, (size_t) new_cap * nhkv * sizeof(__half)));
                if (n_old > 0) {
                    for (int h = 0; h < nhkv; ++h) {
                        CK(cudaMemcpyAsync(nkq + (size_t) h * new_kv_stride, kq[(size_t) il] + (size_t) h * old_kv_stride, b_kv_head, cudaMemcpyDeviceToDevice, st));
                        CK(cudaMemcpyAsync(nvq + (size_t) h * new_kv_stride, vq[(size_t) il] + (size_t) h * old_kv_stride, b_kv_head, cudaMemcpyDeviceToDevice, st));
                        CK(cudaMemcpyAsync(nks + (size_t) h * new_cap,       ks[(size_t) il] + (size_t) h * old_cap,       b_s_head,  cudaMemcpyDeviceToDevice, st));
                        CK(cudaMemcpyAsync(nvs + (size_t) h * new_cap,       vs[(size_t) il] + (size_t) h * old_cap,       b_s_head,  cudaMemcpyDeviceToDevice, st));
                    }
                    CK(cudaStreamSynchronize(st));
                }
                cudaFree(kq[(size_t) il]); cudaFree(vq[(size_t) il]);
                cudaFree(ks[(size_t) il]); cudaFree(vs[(size_t) il]);
                kq[(size_t) il] = nkq; vq[(size_t) il] = nvq;
                ks[(size_t) il] = nks; vs[(size_t) il] = nvs;
            } else {
                std::vector<uint8_t> stage;
                if (n_old > 0) {
                    stage.resize((size_t) nhkv * (2 * b_kv_head + 2 * b_s_head));
                    uint8_t* p = stage.data();
                    for (int h = 0; h < nhkv; ++h) {
                        CK(cudaMemcpyAsync(p, kq[(size_t) il] + (size_t) h * old_kv_stride, b_kv_head, cudaMemcpyDeviceToHost, st)); p += b_kv_head;
                        CK(cudaMemcpyAsync(p, vq[(size_t) il] + (size_t) h * old_kv_stride, b_kv_head, cudaMemcpyDeviceToHost, st)); p += b_kv_head;
                        CK(cudaMemcpyAsync(p, ks[(size_t) il] + (size_t) h * old_cap,       b_s_head,  cudaMemcpyDeviceToHost, st)); p += b_s_head;
                        CK(cudaMemcpyAsync(p, vs[(size_t) il] + (size_t) h * old_cap,       b_s_head,  cudaMemcpyDeviceToHost, st)); p += b_s_head;
                    }
                    CK(cudaStreamSynchronize(st));
                }
                cudaFree(kq[(size_t) il]); cudaFree(vq[(size_t) il]);
                cudaFree(ks[(size_t) il]); cudaFree(vs[(size_t) il]);
                kq[(size_t) il] = nullptr; vq[(size_t) il] = nullptr;
                ks[(size_t) il] = nullptr; vs[(size_t) il] = nullptr;
                CK(cudaMalloc(&kq[(size_t) il], (size_t) new_cap * hp.kv_dim * sizeof(int8_t)));
                CK(cudaMalloc(&vq[(size_t) il], (size_t) new_cap * hp.kv_dim * sizeof(int8_t)));
                CK(cudaMalloc(&ks[(size_t) il], (size_t) new_cap * nhkv * sizeof(__half)));
                CK(cudaMalloc(&vs[(size_t) il], (size_t) new_cap * nhkv * sizeof(__half)));
                if (n_old > 0) {
                    const uint8_t* p = stage.data();
                    for (int h = 0; h < nhkv; ++h) {
                        CK(cudaMemcpyAsync(kq[(size_t) il] + (size_t) h * new_kv_stride, p, b_kv_head, cudaMemcpyHostToDevice, st)); p += b_kv_head;
                        CK(cudaMemcpyAsync(vq[(size_t) il] + (size_t) h * new_kv_stride, p, b_kv_head, cudaMemcpyHostToDevice, st)); p += b_kv_head;
                        CK(cudaMemcpyAsync(ks[(size_t) il] + (size_t) h * new_cap,       p, b_s_head,  cudaMemcpyHostToDevice, st)); p += b_s_head;
                        CK(cudaMemcpyAsync(vs[(size_t) il] + (size_t) h * new_cap,       p, b_s_head,  cudaMemcpyHostToDevice, st)); p += b_s_head;
                    }
                    CK(cudaStreamSynchronize(st));
                }
            }
            kv_cap[(size_t) il] = new_cap;
        }
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

    /// Inject `T` committed target feature vectors (from pos0 .. pos0+T-1, starting at row `feat_off` of d_tgt_feat)
    /// into DSpark KV + update `f_rnn_last`. Captures `inject_graphs[T]` for T = 1..8 (feat_off == 0)!
    template <typename EngineT>
    void inject(EngineT& E, int pos0, int T, int feat_off = 0) {
        if (T <= 0) return;
        const auto t0 = Clock::now();
        const auto& hp = dm->hp;
        const size_t row_skip = (size_t) feat_off * (size_t) hp.n_embd;
        const float* f0 = E.d_tgt_feat[0] + row_skip;
        const float* f1 = E.d_tgt_feat[1] + row_skip;
        const float* f2 = E.d_tgt_feat[2] + row_skip;
        const float* f3 = E.d_tgt_feat[3] + row_skip;
        const int mix_blk = (hp.n_embd == 2304) ? 576 : 256;
        const bool can_graph = (feat_off == 0 && T >= 1 && T <= 8 && !E.prof && !inject_graph_oom[T]);
        if (can_graph) {
            if (h_inj_pos_pin) {
                *h_inj_pos_pin = pos0;
                CK(cudaMemcpyAsync(pos_d, h_inj_pos_pin, sizeof(int32_t), cudaMemcpyHostToDevice, E.st));
            } else {
                CK(cudaMemcpyAsync(pos_d, &pos0, sizeof(int32_t), cudaMemcpyHostToDevice, E.st));
            }
            cudaGraphExec_t& ig = inject_graphs[T];
            if (!ig) {
                cudaGraph_t g = nullptr;
                if (cudaStreamBeginCapture(E.st, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
                    for (int il = 0; il < hp.n_layer; ++il) {
                        const auto& L = dm->layers[(size_t) il];
                        dspark_detail::k_dspark_mix_features<<<(unsigned) T, mix_blk, 0, E.st>>>(
                            f0, f1, f2, f3, dm->dspark_fuse + il * 4, L.dflare_norm, ctx_norm, hp.n_embd, hp.rms_eps);
                        E.prep_q8(ctx_norm, E.xq, T, hp.n_embd);
                        if (!E.try_project_pair_q8_0(E.xq, L.attn_k_ctx, ctx_k, hp.kv_dim,
                                                     L.attn_v_ctx, ctx_v, hp.kv_dim, T, hp.n_embd)) {
                            E.project_preq(ctx_norm, E.xq, L.attn_k_ctx, ctx_k, T, hp.kv_dim, hp.n_embd);
                            E.project_preq(ctx_norm, E.xq, L.attn_v_ctx, ctx_v, T, hp.kv_dim, hp.n_embd);
                        }
                        dspark_detail::k_dspark_norm_rope_kv_store_q8<<<dim3((unsigned) hp.n_head_kv, (unsigned) T), 64, 0, E.st>>>(
                            ctx_k, ctx_v, L.attn_k_ctx_norm,
                            kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il],
                            hp.n_head_kv, 0, pos_d, kv_cap[(size_t) il], hp.is_swa(il) ? 1 : 0,
                            hp.rms_eps, hp.freq_base, hp.n_rot);
                    }
                    const size_t last_off = (size_t) (T - 1) * (size_t) hp.n_embd;
                    dspark_detail::k_dspark_mix_features<<<1, mix_blk, 0, E.st>>>(
                        f0 + last_off, f1 + last_off, f2 + last_off, f3 + last_off,
                        dm->dspark_fuse + hp.n_layer * 4, dm->dspark_norm, f_rnn_last, hp.n_embd, hp.rms_eps);
                    if (cudaStreamEndCapture(E.st, &g) == cudaSuccess && g) {
                        if (cudaGraphInstantiate(&ig, g, nullptr, nullptr, 0) != cudaSuccess) {
                            (void) cudaGetLastError();
                            ig = nullptr;
                            inject_graph_oom[T] = true;
                        }
                        cudaGraphDestroy(g);
                    } else {
                        (void) cudaGetLastError();
                        inject_graph_oom[T] = true;
                    }
                } else {
                    (void) cudaGetLastError();
                    inject_graph_oom[T] = true;
                }
            }
            if (ig) {
                CK(cudaGraphLaunch(ig, E.st));
                pos = pos0 + T;
                return;
            }
        }
        for (int il = 0; il < hp.n_layer; ++il) {
            const auto& L = dm->layers[(size_t) il];
            // EX-5: SWA ring only retains the last hp.n_swa (1024) tokens! When T > hp.n_swa,
            // injecting the first (T - hp.n_swa) tokens is 100% overwritten before decode ever reads them.
            // Skipping them saves 74% of zero-copy PCIe feature reads + K/V GEMM on both SWA draft layers (bit-exact!).
            const int swa_skip = (hp.is_swa(il) && T > (int) hp.n_swa) ? (T - (int) hp.n_swa) : 0;
            const int t_eff = T - swa_skip;
            const int pos_eff = pos0 + swa_skip;
            const size_t f_off = (size_t) swa_skip * (size_t) hp.n_embd;
            dspark_detail::k_dspark_mix_features<<<(unsigned) t_eff, mix_blk, 0, E.st>>>(
                f0 + f_off, f1 + f_off, f2 + f_off, f3 + f_off,
                dm->dspark_fuse + il * 4, L.dflare_norm, ctx_norm, hp.n_embd, hp.rms_eps);
            E.prep_q8(ctx_norm, E.xq, t_eff, hp.n_embd);
            if (!E.try_project_pair_q8_0(t_eff <= 8 ? E.xq : nullptr, L.attn_k_ctx, ctx_k, hp.kv_dim,
                                         L.attn_v_ctx, ctx_v, hp.kv_dim, t_eff, hp.n_embd)) {
                E.project_preq(ctx_norm, t_eff <= 8 ? E.xq : nullptr, L.attn_k_ctx, ctx_k, t_eff, hp.kv_dim, hp.n_embd);
                E.project_preq(ctx_norm, t_eff <= 8 ? E.xq : nullptr, L.attn_v_ctx, ctx_v, t_eff, hp.kv_dim, hp.n_embd);
            }
            if (t_eff > 8 && inv_freq_d) {
                dspark_detail::k_dspark_norm_rope_kv_store_q8_x4<<<dim3((unsigned) hp.n_head_kv, (unsigned) ((t_eff + 3) / 4)), 256, 0, E.st>>>(
                    ctx_k, ctx_v, L.attn_k_ctx_norm,
                    kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il],
                    hp.n_head_kv, pos_eff, t_eff, kv_cap[(size_t) il], hp.is_swa(il) ? 1 : 0,
                    hp.rms_eps, hp.n_rot, inv_freq_d);
            } else {
                dspark_detail::k_dspark_norm_rope_kv_store_q8<<<dim3((unsigned) hp.n_head_kv, (unsigned) t_eff), 64, 0, E.st>>>(
                    ctx_k, ctx_v, L.attn_k_ctx_norm,
                    kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il],
                    hp.n_head_kv, pos_eff, nullptr, kv_cap[(size_t) il], hp.is_swa(il) ? 1 : 0,
                    hp.rms_eps, hp.freq_base, hp.n_rot);
            }
        }
        const size_t last_off = (size_t) (T - 1) * (size_t) hp.n_embd;
        dspark_detail::k_dspark_mix_features<<<1, mix_blk, 0, E.st>>>(
            f0 + last_off, f1 + last_off, f2 + last_off, f3 + last_off,
            dm->dspark_fuse + hp.n_layer * 4, dm->dspark_norm, f_rnn_last, hp.n_embd, hp.rms_eps);
        pos = pos0 + T;
        if (E.prof) {
            CK(cudaStreamSynchronize(E.st));
            inject_ms += ms_since(t0);
        }
    }

    /// Wave 5: Arm zero-PCIe running VRAM feature accumulators for a prefill chunk of length `T` (`T > 8`).
    /// [0]=SWA L0 ([T - pp_acc_swa_start, N]), [1]=Full L1 ([T, N], aliases E.msg3), [2]=SWA L2 ([T - pp_acc_swa_start, N]).
    void begin_prefill_vram_acc(float* acc_swa0, float* acc_full1, float* acc_swa2, int T) {
        d_pp_acc[0] = acc_swa0;
        d_pp_acc[1] = acc_full1;
        d_pp_acc[2] = acc_swa2;
        if (!dm) {
            pp_acc_swa_start = 0;
            prefill_vram_acc_active = false;
            return;
        }
        const auto& hp = dm->hp;
        // Match Wave 3C (EX-5) exact swa_skip formula in inject(): retain last hp.n_swa tokens.
        pp_acc_swa_start = (T > (int) hp.n_swa) ? (T - (int) hp.n_swa) : 0;
        const bool layout_ok = (hp.n_layer == 3 && hp.is_swa(0) && !hp.is_swa(1) && hp.is_swa(2));
        prefill_vram_acc_active = (layout_ok && acc_swa0 != nullptr && acc_full1 != nullptr && acc_swa2 != nullptr && T > 8);
    }

    /// Wave 5: Accumulate target layer feature `ti` (0, 1, 2 at il = 5, 23, 41) directly into VRAM accumulators.
    void accum_prefill_feat(int ti, const float* cur, int T, cudaStream_t st) {
        if (!prefill_vram_acc_active || ti < 0 || ti >= 3 || T <= 0) return;
        const int N = dm->hp.n_embd;
        const int blk = (N == 2304) ? 576 : 256;
        k_dspark_accum_cur<<<(unsigned) T, blk, 0, st>>>(
            cur, d_pp_acc[0], d_pp_acc[1], d_pp_acc[2], f_rnn_last,
            fuse_w_host[0][ti], fuse_w_host[1][ti], fuse_w_host[2][ti], fuse_w_host[3][ti],
            T, pp_acc_swa_start, N, ti == 0);
    }

    template <typename EngineT>
    bool can_inject_all_f16(const EngineT& E, int T) const {
        if (!prefill_vram_acc_active || !dm || !E.act_bf16) return false;
        const auto& hp = dm->hp;
        for (int il = 0; il < hp.n_layer; ++il) {
            const auto& L = dm->layers[(size_t) il];
            const int t_off = L.is_swa ? pp_acc_swa_start : 0;
            const int T_eff = T - t_off;
            if (T_eff <= 8) return false;
            if (L.attn_k_ctx.type == 0 || (L.attn_k_ctx.type == 30 && !E.bf16_x_f16)) return false;
            if (L.attn_v_ctx.type == 0 || (L.attn_v_ctx.type == 30 && !E.bf16_x_f16)) return false;
        }
        return true;
    }

    /// Wave 5: Finalize `ti = 3` (`cur_final`) + fused RMSNorm + FP16 emit -> K/V ctx GEMM -> Q8 KV store (ZERO PCIe!).
    /// Falls back cleanly to `inject(E, pos0, T)` whenever `!prefill_vram_acc_active`.
    template <typename EngineT>
    void inject_from_vram_acc(EngineT& E, const float* cur_final, int T, int pos0, cudaStream_t st) {
        if (T <= 0) {
            prefill_vram_acc_active = false;
            return;
        }
        if (!prefill_vram_acc_active || !cur_final) {
            prefill_vram_acc_active = false;
            inject(E, pos0, T);
            return;
        }
        const auto t0 = Clock::now();
        const auto& hp = dm->hp;
        const int N = hp.n_embd;
        const int mix_blk = (N == 2304) ? 576 : 256;

        // 1. Finalize f_rnn_last for the last token (T - 1) + exact FP32 RMSNorm (matches inject() bit-for-bit).
        dspark_detail::k_dspark_accum_final_rmsnorm_f32<<<1, mix_blk, 0, st>>>(
            cur_final, f_rnn_last, dm->dspark_norm, f_rnn_last,
            fuse_w_host[hp.n_layer][3], hp.rms_eps, 1, T - 1, N);

        // 2. Finalize each draft layer il = 0..2: fused k=3 accum + RMSNorm + direct FP16 emit (zero k_f32_to_f16!).
        for (int il = 0; il < hp.n_layer; ++il) {
            const auto& L = dm->layers[(size_t) il];
            const int t_off = L.is_swa ? pp_acc_swa_start : 0;
            const int T_eff = T - t_off;
            const int pos_eff = pos0 + t_off;
            const bool kv_can_f16 = (T_eff > 8 &&
                                     L.attn_k_ctx.type != 0 && (L.attn_k_ctx.type != 30 || E.bf16_x_f16) &&
                                     L.attn_v_ctx.type != 0 && (L.attn_v_ctx.type != 30 || E.bf16_x_f16));
            uint16_t* ctx_f16_u16 = E.act_bf16 ? E.act_bf16 : reinterpret_cast<uint16_t*>(ctx_norm);
            if (kv_can_f16 && ctx_f16_u16) {
                k_dspark_accum_final_rmsnorm_f16<<<(unsigned) T_eff, mix_blk, 0, st>>>(
                    cur_final, d_pp_acc[il], L.dflare_norm,
                    reinterpret_cast<__half*>(ctx_f16_u16),
                    fuse_w_host[il][3], hp.rms_eps, T_eff, t_off, N);
                E.project_preq(nullptr, nullptr, L.attn_k_ctx, ctx_k, T_eff, hp.kv_dim, N, ctx_f16_u16);
                E.project_preq(nullptr, nullptr, L.attn_v_ctx, ctx_v, T_eff, hp.kv_dim, N, ctx_f16_u16);
            } else {
                dspark_detail::k_dspark_accum_final_rmsnorm_f32<<<(unsigned) T_eff, mix_blk, 0, st>>>(
                    cur_final, d_pp_acc[il], L.dflare_norm, ctx_norm,
                    fuse_w_host[il][3], hp.rms_eps, T_eff, t_off, N);
                E.prep_q8(ctx_norm, E.xq, T_eff, N);
                if (!E.try_project_pair_q8_0(T_eff <= 8 ? E.xq : nullptr, L.attn_k_ctx, ctx_k, hp.kv_dim,
                                             L.attn_v_ctx, ctx_v, hp.kv_dim, T_eff, N)) {
                    E.project_preq(ctx_norm, T_eff <= 8 ? E.xq : nullptr, L.attn_k_ctx, ctx_k, T_eff, hp.kv_dim, N);
                    E.project_preq(ctx_norm, T_eff <= 8 ? E.xq : nullptr, L.attn_v_ctx, ctx_v, T_eff, hp.kv_dim, N);
                }
            }
            if (T_eff > 8 && inv_freq_d) {
                dspark_detail::k_dspark_norm_rope_kv_store_q8_x4<<<dim3((unsigned) hp.n_head_kv, (unsigned) ((T_eff + 3) / 4)), 256, 0, st>>>(
                    ctx_k, ctx_v, L.attn_k_ctx_norm,
                    kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il],
                    hp.n_head_kv, pos_eff, T_eff, kv_cap[(size_t) il], hp.is_swa(il) ? 1 : 0,
                    hp.rms_eps, hp.n_rot, inv_freq_d);
            } else {
                dspark_detail::k_dspark_norm_rope_kv_store_q8<<<dim3((unsigned) hp.n_head_kv, (unsigned) T_eff), 64, 0, st>>>(
                    ctx_k, ctx_v, L.attn_k_ctx_norm,
                    kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il],
                    hp.n_head_kv, pos_eff, nullptr, kv_cap[(size_t) il], hp.is_swa(il) ? 1 : 0,
                    hp.rms_eps, hp.freq_base, hp.n_rot);
            }
        }

        pos = pos0 + T;
        prefill_vram_acc_active = false;
        if (E.prof) {
            CK(cudaStreamSynchronize(st));
            inject_ms += ms_since(t0);
        }
    }

    template <typename EngineT>
    void run_dflare(EngineT& E, int32_t anchor) {
        const auto& hp = dm->hp;
        const int block_n = hp.block_size;
        const int N = hp.n_embd;
        const int rank = hp.markov_rank;

        // 1. Anchor embedding (with target tok_embd_norm) + device pos upload
        float* anchor_src = h_anchor_pin ? h_anchor_pin : h_anchor.data();
        E.m->dequant_tok(anchor, anchor_src);
        CK(cudaMemcpyAsync(x_norm, anchor_src, (size_t) N * sizeof(float), cudaMemcpyHostToDevice, E.st));
        if (h_pos_pin) {
            *h_pos_pin = pos;
            CK(cudaMemcpyAsync(pos_d, h_pos_pin, sizeof(int32_t), cudaMemcpyHostToDevice, E.st));
        } else {
            CK(cudaMemcpyAsync(pos_d, &pos, sizeof(int32_t), cudaMemcpyHostToDevice, E.st));
        }

        // 2. 3 DFlare transformer layers over block_cur [block_n, N] + output_norm + rnn_state zeroing (CUDA Graph captured)
        const float kq_scale = 1.0f / std::sqrt((float) hp.head_dim);
        const bool use_split = (pos >= 256 && hp.n_head == 3 * hp.n_head_kv && hp.head_dim == 128 && block_n <= 8);
        auto emit_dflare = [&]() {
            k_rms_norm<<<1, kThreads, 0, E.st>>>(x_norm, E.m->tok_embd_norm, block_cur, N, E.m->hp.rms_eps);
            dspark_detail::k_dspark_broadcast_tail<<<dim3((unsigned) ((N + 255) / 256), (unsigned) (block_n - 1)), 256, 0, E.st>>>(
                block_cur, dm->mask_embd, block_n, N);
            for (int il = 0; il < hp.n_layer; ++il) {
                const auto& L = dm->layers[(size_t) il];
                const int rms_blk = (N == 2304) ? 576 : kThreads;
                k_rms_norm_q8_1<<<(unsigned) block_n, rms_blk, 0, E.st>>>(block_cur, L.attn_norm, x_norm, E.xq, N, hp.rms_eps);
                E.project_preq(x_norm, E.xq, L.attn_q, Q, block_n, hp.q_dim, N);
                if (!E.try_project_pair_q8_0(E.xq, L.attn_k, K, hp.kv_dim, L.attn_v, V, hp.kv_dim, block_n, N)) {
                    E.project_preq(x_norm, E.xq, L.attn_k, K, block_n, hp.kv_dim, N);
                    E.project_preq(x_norm, E.xq, L.attn_v, V, block_n, hp.kv_dim, N);
                }
                dspark_detail::k_dspark_qk_norm_rope<<<dim3((unsigned) (hp.n_head + hp.n_head_kv), (unsigned) block_n), 64, 0, E.st>>>(
                    Q, K, L.attn_q_norm, L.attn_k_norm, hp.n_head, hp.n_head_kv, pos_d, hp.rms_eps, hp.freq_base, hp.n_rot);
                const void* dspark_att_xq = nullptr;
                if (use_split) {
                    if (!L.is_swa) {
                        dspark_detail::k_dspark_attn_q8_wmma_split<false><<<dim3((unsigned) hp.n_head_kv, (unsigned) split_full), 128, 18432, E.st>>>(
                            Q, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il], K, V, split_out,
                            hp.n_head, hp.n_head_kv, kv_cap[(size_t) il], hp.n_swa, pos_d, block_n, kq_scale, split_full);
                        dspark_detail::k_dspark_split_combine<<<dim3((unsigned) hp.n_head, (unsigned) block_n), 128, 0, E.st>>>(
                            split_out, att, hp.n_head, pos_d, split_full, 0, hp.n_swa, E.xq);
                    } else {
                        const int sp_swa = std::min(split_full, 24);
                        dspark_detail::k_dspark_attn_q8_wmma_split<true><<<dim3((unsigned) hp.n_head_kv, (unsigned) sp_swa), 128, 18432, E.st>>>(
                            Q, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il], K, V, split_out,
                            hp.n_head, hp.n_head_kv, kv_cap[(size_t) il], hp.n_swa, pos_d, block_n, kq_scale, sp_swa);
                        dspark_detail::k_dspark_split_combine<<<dim3((unsigned) hp.n_head, (unsigned) block_n), 128, 0, E.st>>>(
                            split_out, att, hp.n_head, pos_d, sp_swa, 1, hp.n_swa, E.xq);
                    }
                    dspark_att_xq = E.xq;
                } else {
                    dspark_detail::k_dspark_attn_q8<<<dim3((unsigned) hp.n_head, (unsigned) block_n), 128, 0, E.st>>>(
                        Q, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il], K, V, att,
                        hp.n_head, hp.n_head_kv, kv_cap[(size_t) il], hp.is_swa(il) ? 1 : 0, hp.n_swa, pos_d, block_n, kq_scale);
                }
                E.project_preq(att, dspark_att_xq, L.attn_output, wo, block_n, N, hp.q_dim);
                dspark_detail::k_dspark_add<<<(unsigned) (((int64_t) block_n * N + 255) / 256), 256, 0, E.st>>>(
                    block_cur, block_cur, wo, (int64_t) block_n * N);

                k_rms_norm_q8_1<<<(unsigned) block_n, (N == 2304) ? 576 : kThreads, 0, E.st>>>(block_cur, L.ffn_norm, x_norm, E.xq, N, hp.rms_eps);
                if (!E.try_project_pair_q8_0(E.xq, L.ffn_gate, gate, hp.n_ff, L.ffn_up, up, hp.n_ff, block_n, N)) {
                    E.project_preq(x_norm, E.xq, L.ffn_gate, gate, block_n, hp.n_ff, N);
                    E.project_preq(x_norm, E.xq, L.ffn_up, up, block_n, hp.n_ff, N);
                }
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
    }

    /// Confidence-gated greedy DSpark draft (1..max_k tokens; returns 0 tokens if step 0 falls below tau_min0).
    template <typename EngineT>
    void draft_confident(EngineT& E, int32_t anchor, int base_k, int max_k,
                         float tau_min0, float tau_step, float tau_cum,
                         float tau_ext, float tau_cum_ext,
                         std::vector<int32_t>& out_tokens) {
        ensure_decode_weights_resident(E.st);
        wait_decode_weights_ready(E.st);
        const auto t0 = Clock::now();
        const auto& hp = dm->hp;
        const int N = hp.n_embd;
        const int rank = hp.markov_rank;
        const int eff_base = std::max(1, base_k);
        const int eff_max = std::clamp(max_k, 1, std::min(7, hp.block_size - 1));
        out_tokens.clear();

        run_dflare(E, anchor);

        if (h_draft_ids_pin) {
            h_draft_ids_pin[0] = anchor;
            CK(cudaMemcpyAsync(d_draft_ids, h_draft_ids_pin, sizeof(int32_t), cudaMemcpyHostToDevice, E.st));
        } else {
            CK(cudaMemcpyAsync(d_draft_ids, &anchor, sizeof(int32_t), cudaMemcpyHostToDevice, E.st));
        }

        const int joint_tot = 2 * rank + 2 * N;
        float p_cum = 1.0f;
        for (int i = 0; i < eff_max; ++i) {
            auto emit_step = [&]() {
                const float* hi = block_cur + (size_t) (i + 1) * N;
                dspark_detail::k_dspark_prep_joint_d<<<(unsigned) ((joint_tot + 255) / 256), 256, 0, E.st>>>(
                    rnn_state, dm->markov_w1, dm->markov_w1_type, d_draft_ids + i, hi, f_rnn_last, joint_inp, rank, N);
                E.project(joint_inp, dm->joint_proj, joint_out, 1, 3 * rank, joint_tot);
                dspark_detail::k_dspark_rnn_update<<<(unsigned) ((rank + 255) / 256), 256, 0, E.st>>>(
                    joint_out, dm->joint_proj_b, rnn_state, rnn_latent, rank);
                E.project(rnn_latent, dm->hidden_out, rnn_corr, 1, N, rank);
                dspark_detail::k_dspark_add3<<<(unsigned) ((N + 255) / 256), 256, 0, E.st>>>(
                    h_pred, hi, rnn_corr, dm->hidden_out_b, N);
                E.project(h_pred, E.m->output_weight, E.logits, 1, hp.vocab_unpadded, N);
                constexpr int kArgBlks = 64;
                dspark_detail::k_dspark_argmax_prob_stage1<<<kArgBlks, 256, 0, E.st>>>(
                    E.logits, E.argmax_val, E.argmax_idx, d_prob_sum, (int) hp.vocab_unpadded, hp.softcap);
                dspark_detail::k_dspark_argmax_prob_stage2<<<1, 32, 0, E.st>>>(
                    E.argmax_val, E.argmax_idx, d_prob_sum, d_draft_ids + i + 1, d_draft_prob + i, kArgBlks);
            };
            if (!E.prof && !markov_step_oom[i]) {
                cudaGraphExec_t& sg = markov_step_graphs[i];
                if (!sg) {
                    cudaGraph_t g = nullptr;
                    if (cudaStreamBeginCapture(E.st, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
                        emit_step();
                        if (cudaStreamEndCapture(E.st, &g) == cudaSuccess && g) {
                            if (cudaGraphInstantiate(&sg, g, nullptr, nullptr, 0) != cudaSuccess) {
                                (void) cudaGetLastError();
                                sg = nullptr;
                                markov_step_oom[i] = true;
                            }
                            cudaGraphDestroy(g);
                        } else {
                            (void) cudaGetLastError();
                            markov_step_oom[i] = true;
                        }
                    } else {
                        (void) cudaGetLastError();
                        markov_step_oom[i] = true;
                    }
                }
                if (sg) CK(cudaGraphLaunch(sg, E.st));
                else emit_step();
            } else {
                emit_step();
            }

            int32_t tok_i = 0;
            float p_i = 0.0f;
            if (h_draft_ids_pin && h_draft_prob_pin) {
                CK(cudaMemcpyAsync(h_draft_ids_pin + 1 + i, d_draft_ids + 1 + i, sizeof(int32_t), cudaMemcpyDeviceToHost, E.st));
                CK(cudaMemcpyAsync(h_draft_prob_pin + i, d_draft_prob + i, sizeof(float), cudaMemcpyDeviceToHost, E.st));
                CK(cudaStreamSynchronize(E.st));
                tok_i = h_draft_ids_pin[1 + i];
                p_i = h_draft_prob_pin[i];
            } else {
                CK(cudaMemcpyAsync(&tok_i, d_draft_ids + 1 + i, sizeof(int32_t), cudaMemcpyDeviceToHost, E.st));
                CK(cudaMemcpyAsync(&p_i, d_draft_prob + i, sizeof(float), cudaMemcpyDeviceToHost, E.st));
                CK(cudaStreamSynchronize(E.st));
            }
            p_cum *= p_i;
            if (i == 0 && p_i < tau_min0) {
                break;
            }
            out_tokens.push_back(tok_i);
            if (i + 1 >= eff_max) break;
            if (i + 1 < eff_base) {
                if (p_i < tau_step || p_cum < tau_cum) break;
            } else {
                if (p_i < tau_ext || p_cum < tau_cum_ext) break;
            }
        }
        if (!out_tokens.empty()) {
            ++n_draft_calls;
            n_drafted += (int64_t) out_tokens.size();
        }
        if (E.prof) draft_ms += ms_since(t0);
    }

    /// Confidence-gated sampled DSpark draft (0..max_k tokens) with exact distribution preservation.
    /// `sample_fn(step_i, row_lg, out_pmax)` builds q_dist[step_i], sets `out_pmax = max_v q_i(v)`
    /// (deterministic given prefix, independent of the random draw!), and returns:
    ///   -1 if `step_i == 0 && out_pmax < tau_min0` (caller aborts before drawing / pushing q_dist),
    ///   otherwise the sampled token id >= 0 (after pushing q_dist[step_i]).
    template <typename EngineT, typename SampleFn>
    void draft_sampled_confident(EngineT& E, int32_t anchor, int base_k, int max_k,
                                 float tau_step, float tau_cum,
                                 float tau_ext, float tau_cum_ext,
                                 std::vector<int32_t>& out_tokens,
                                 std::vector<std::vector<float>>* out_logits,
                                 SampleFn&& sample_fn, bool gpu_sampled = false) {
        ensure_decode_weights_resident(E.st);
        wait_decode_weights_ready(E.st);
        const auto t0 = Clock::now();
        const auto& hp = dm->hp;
        const int N = hp.n_embd;
        const int rank = hp.markov_rank;
        const int eff_base = std::max(1, base_k);
        const int eff_max = std::clamp(max_k, 1, std::min(7, hp.block_size - 1));
        out_tokens.clear();
        if (out_logits) out_logits->clear();

        run_dflare(E, anchor);

        const int joint_tot = 2 * rank + 2 * N;
        int32_t prev_id = anchor;
        float p_cum = 1.0f;
        for (int i = 0; i < eff_max; ++i) {
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
            if (hp.softcap > 0.0f) {
                k_softcap<<<(unsigned) ((hp.vocab_unpadded + 255) / 256), 256, 0, E.st>>>(
                    E.logits, hp.vocab_unpadded, hp.softcap);
            }
            float p_i = 0.0f;
            if (gpu_sampled) {
                std::vector<float> dummy_lg;
                prev_id = sample_fn(i, dummy_lg, p_i);
            } else {
                std::vector<float> row_lg((size_t) hp.vocab_unpadded);
                CK(cudaMemcpyAsync(row_lg.data(), E.logits, (size_t) hp.vocab_unpadded * sizeof(float),
                                   cudaMemcpyDeviceToHost, E.st));
                CK(cudaStreamSynchronize(E.st));
                prev_id = sample_fn(i, row_lg, p_i);
                if (out_logits && prev_id >= 0) out_logits->push_back(std::move(row_lg));
            }
            if (prev_id < 0) break;
            out_tokens.push_back(prev_id);
            p_cum *= p_i;
            if (i + 1 >= eff_max) break;
            if (i + 1 < eff_base) {
                if (p_i < tau_step || p_cum < tau_cum) break;
            } else {
                if (p_i < tau_ext || p_cum < tau_cum_ext) break;
            }
        }
        if (!out_tokens.empty()) {
            ++n_draft_calls;
            n_drafted += (int64_t) out_tokens.size();
        }
        if (E.prof) draft_ms += ms_since(t0);
    }

    /// Draft `n_draft` (1..7) speculative tokens from `anchor` (placed at `pos`, seeing context `0..pos-1`).
    /// When `out_logits != nullptr` (`n_draft` rows of `vocab_unpadded`), uses `sample_fn(row_vec)` for each draft step.
    template <typename EngineT, typename SampleFn>
    void draft(EngineT& E, int32_t anchor, int n_draft, std::vector<int32_t>& out_tokens,
               std::vector<std::vector<float>>* out_logits, SampleFn&& sample_fn, bool gpu_sampled = false) {
        ensure_decode_weights_resident(E.st);
        wait_decode_weights_ready(E.st);
        const auto t0 = Clock::now();
        const auto& hp = dm->hp;
        const int N = hp.n_embd;
        const int rank = hp.markov_rank;

        out_tokens.resize((size_t) n_draft);
        if (out_logits) out_logits->resize((size_t) n_draft);

        run_dflare(E, anchor);

        // 3. Recurrent Markov chain for i = 1 .. n_draft
        const int joint_tot = 2 * rank + 2 * N;
        if (!out_logits && !gpu_sampled && n_draft >= 1 && n_draft <= 7 && d_draft_ids) {
            // Zero-sync device-side greedy Markov chain (1 graph launch + 1 single D2H copy instead of n_draft syncs!)
            if (h_draft_ids_pin) {
                h_draft_ids_pin[0] = anchor;
                CK(cudaMemcpyAsync(d_draft_ids, h_draft_ids_pin, sizeof(int32_t), cudaMemcpyHostToDevice, E.st));
            } else {
                CK(cudaMemcpyAsync(d_draft_ids, &anchor, sizeof(int32_t), cudaMemcpyHostToDevice, E.st));
            }
            auto emit_markov = [&]() {
                for (int i = 0; i < n_draft; ++i) {
                    const float* hi = block_cur + (size_t) (i + 1) * N;
                    dspark_detail::k_dspark_prep_joint_d<<<(unsigned) ((joint_tot + 255) / 256), 256, 0, E.st>>>(
                        rnn_state, dm->markov_w1, dm->markov_w1_type, d_draft_ids + i, hi, f_rnn_last, joint_inp, rank, N);
                    E.project(joint_inp, dm->joint_proj, joint_out, 1, 3 * rank, joint_tot);
                    dspark_detail::k_dspark_rnn_update<<<(unsigned) ((rank + 255) / 256), 256, 0, E.st>>>(
                        joint_out, dm->joint_proj_b, rnn_state, rnn_latent, rank);
                    E.project(rnn_latent, dm->hidden_out, rnn_corr, 1, N, rank);
                    dspark_detail::k_dspark_add3<<<(unsigned) ((N + 255) / 256), 256, 0, E.st>>>(
                        h_pred, hi, rnn_corr, dm->hidden_out_b, N);
                    E.project(h_pred, E.m->output_weight, E.logits, 1, hp.vocab_unpadded, N);
                    if (hp.vocab_unpadded >= 4096 && E.argmax_val && E.argmax_idx) {
                        constexpr int kArgBlks = 64;
                        k_argmax_stage1<<<dim3(kArgBlks, 1), 256, 0, E.st>>>(
                            E.logits, E.argmax_val, E.argmax_idx, (int) hp.vocab_unpadded);
                        k_argmax_stage2<<<1, 32, 0, E.st>>>(
                            E.argmax_val, E.argmax_idx, d_draft_ids + i + 1, kArgBlks);
                    } else {
                        k_argmax<<<1, 256, 0, E.st>>>(E.logits, d_draft_ids + i + 1, (int) hp.vocab_unpadded);
                    }
                }
            };
            if (!E.prof && !markov_graph_oom[n_draft]) {
                cudaGraphExec_t& mg = markov_graphs[n_draft];
                if (!mg) {
                    cudaGraph_t g = nullptr;
                    if (cudaStreamBeginCapture(E.st, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
                        emit_markov();
                        if (cudaStreamEndCapture(E.st, &g) == cudaSuccess && g) {
                            if (cudaGraphInstantiate(&mg, g, nullptr, nullptr, 0) != cudaSuccess) {
                                (void) cudaGetLastError();
                                mg = nullptr;
                                markov_graph_oom[n_draft] = true;
                            }
                            cudaGraphDestroy(g);
                        } else {
                            (void) cudaGetLastError();
                            markov_graph_oom[n_draft] = true;
                        }
                    } else {
                        (void) cudaGetLastError();
                        markov_graph_oom[n_draft] = true;
                    }
                }
                if (mg) CK(cudaGraphLaunch(mg, E.st));
                else emit_markov();
            } else {
                emit_markov();
            }
            if (h_draft_ids_pin) {
                CK(cudaMemcpyAsync(h_draft_ids_pin + 1, d_draft_ids + 1, (size_t) n_draft * sizeof(int32_t),
                                   cudaMemcpyDeviceToHost, E.st));
                CK(cudaStreamSynchronize(E.st));
                for (int i = 0; i < n_draft; ++i) out_tokens[(size_t) i] = h_draft_ids_pin[1 + i];
            } else {
                CK(cudaMemcpyAsync(out_tokens.data(), d_draft_ids + 1, (size_t) n_draft * sizeof(int32_t),
                                   cudaMemcpyDeviceToHost, E.st));
                CK(cudaStreamSynchronize(E.st));
            }
        } else {
            int32_t prev_id = anchor;
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
                if (gpu_sampled) {
                    if (hp.softcap > 0.0f) {
                        k_softcap<<<(unsigned) ((hp.vocab_unpadded + 255) / 256), 256, 0, E.st>>>(
                            E.logits, hp.vocab_unpadded, hp.softcap);
                    }
                    std::vector<float> dummy_lg;
                    prev_id = sample_fn(i, dummy_lg);
                } else if (out_logits) {
                    k_softcap<<<(unsigned) ((hp.vocab_unpadded + 255) / 256), 256, 0, E.st>>>(
                        E.logits, hp.vocab_unpadded, hp.softcap);
                    auto& row_lg = (*out_logits)[(size_t) i];
                    row_lg.resize((size_t) hp.vocab_unpadded);
                    CK(cudaMemcpyAsync(row_lg.data(), E.logits, (size_t) hp.vocab_unpadded * sizeof(float),
                                       cudaMemcpyDeviceToHost, E.st));
                    CK(cudaStreamSynchronize(E.st));
                    prev_id = sample_fn(i, row_lg);
                } else {
                    if (hp.vocab_unpadded >= 4096 && E.argmax_val && E.argmax_idx) {
                        constexpr int kArgBlks = 64;
                        k_argmax_stage1<<<dim3(kArgBlks, 1), 256, 0, E.st>>>(
                            E.logits, E.argmax_val, E.argmax_idx, (int) hp.vocab_unpadded);
                        k_argmax_stage2<<<1, 32, 0, E.st>>>(
                            E.argmax_val, E.argmax_idx, E.pick_d, kArgBlks);
                    } else {
                        k_argmax<<<1, 256, 0, E.st>>>(E.logits, E.pick_d, (int) hp.vocab_unpadded);
                    }
                    int32_t next_id = 0;
                    CK(cudaMemcpyAsync(&next_id, E.pick_d, sizeof(int32_t), cudaMemcpyDeviceToHost, E.st));
                    CK(cudaStreamSynchronize(E.st));
                    prev_id = next_id;
                }
                out_tokens[(size_t) i] = prev_id;
            }
        }
        ++n_draft_calls;
        n_drafted += n_draft;
        if (E.prof) draft_ms += ms_since(t0);
    }
};
