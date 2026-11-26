// src/inkling2/inkling2_engine.cu - Standalone / integrated CUDA + AVX2 hybrid engine for Inkling-2-Mini (arch inkling2).
#include "strata/inkling2/engine.hpp"
#include "strata/inkling2/profile_rules.hpp"
#include "inkling2_expert_prior.cuh"
#include "inkling2_kernels.cuh"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/gguf_split.hpp"
#include "strata/core/emulate.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/quant/dequant.hpp"
#include "strata/quant/types.hpp"
#if defined(STRATA_INKLING2_CPU_POOL)
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#endif

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#if !defined(_WIN32)
#include <unistd.h>
#endif
#if defined(__linux__)
#include <sys/mman.h>
#endif
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace strata::inkling2 {

__global__ void k_split_ozaki_f32_to_f16_pair(const float* __restrict__ src,
                                              __half* __restrict__ dst_hi,
                                              __half* __restrict__ dst_lo,
                                              size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float a = src[i];
        const __half ah = __float2half_rn(a);
        const __half al = __float2half_rn((a - __half2float(ah)) * 2048.0f);
        dst_hi[i] = ah;
        dst_lo[i] = al;
    }
}

namespace {

using Clock = std::chrono::steady_clock;
namespace K = strata::kernels;

inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

#define CK(call)                                                                                                  \
    do {                                                                                                          \
        cudaError_t err_ = (call);                                                                                \
        if (err_ != cudaSuccess) {                                                                                \
            std::fprintf(stderr, "strata-inkling2 CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                 \
                         cudaGetErrorString(err_));                                                               \
            std::exit(1);                                                                                         \
        }                                                                                                         \
    } while (0)

#define CKB(call)                                                                                                 \
    do {                                                                                                          \
        cublasStatus_t err_ = (call);                                                                             \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                                      \
            std::fprintf(stderr, "strata-inkling2 cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, (int) err_);  \
            std::exit(1);                                                                                         \
        }                                                                                                         \
    } while (0)

struct Hparams {
    int64_t n_layer = 48;       // virtual layers
    int64_t n_phys = 24;        // physical layers
    int64_t num_loops = 3;
    int64_t core_start = 6;
    int64_t core_end = 18;
    int64_t n_embd = 2304;
    int64_t n_ff = 1152;
    int64_t n_head = 24;
    int64_t n_head_kv = 8;
    int64_t head_dim = 128;
    int64_t q_dim = 3072;
    int64_t kv_dim = 1024;
    int64_t n_expert = 128;
    int64_t n_expert_used = 8;
    int64_t n_rel = 16;
    int64_t n_rel_ext = 1024;
    int64_t n_swa = 1024;
    int64_t attn_res_block_size = 6;
    int64_t n_conv = 4;
    int64_t loop_gate_rank = 64;
    int64_t vocab_padded = 201024;
    int64_t vocab_unpadded = 200058;
    int64_t log_scale_floor = 8192;
    float log_scale_alpha = 0.1f;
    float rms_eps = 1e-6f;
    float qk_clip = 22.6274166f;
    float rel_clip = 8.0f;
    float situ_beta = 4.0f;
    float situ_linear_beta = 25.0f;
    float expert_weights_scale = 8.0f;
    float softcap = 30.0f;
    std::vector<uint8_t> swa_pattern;

    int64_t n_sconv_channels() const { return 2 * kv_dim + 2 * n_embd; }
    bool is_swa(int64_t il) const {
        if ((size_t) il < swa_pattern.size()) return swa_pattern[(size_t) il] != 0;
        return (il % 6) != 5;
    }
    int loop_layer(int il) const {
        if (num_loops <= 1) return il;
        const int64_t n_core = core_end - core_start;
        if (il < core_start) return il;
        if (il < core_start + num_loops * n_core) return (int) (core_start + ((il - core_start) % n_core));
        return (int) (il - (num_loops - 1) * n_core);
    }
};

struct DenseMat {
    int type = -1;
    int64_t n_in = 0;
    int64_t n_out = 0;
    void* d = nullptr;
    size_t bytes = 0;
    bool q8_in_place = false;
};

struct GpuWeightStore {
    std::vector<void*> gpu_allocs;
    std::vector<void*> host_allocs;
    std::vector<void*> host_pinned_allocs;
    std::vector<void*> q8_packed_keys;
    bool has_quant_dense = false;

    virtual ~GpuWeightStore() = default;

    void free_store() {
        for (void* p : q8_packed_keys) K::native_q8_0_packed_unregister(p);
        for (void* p : gpu_allocs) cudaFree(p);
        for (void* p : host_pinned_allocs) cudaFreeHost(p);
        for (void* p : host_allocs) std::free(p);
        q8_packed_keys.clear();
        gpu_allocs.clear();
        host_pinned_allocs.clear();
        host_allocs.clear();
    }

    template <typename T>
    T* gpu_alloc(size_t count) {
        T* p = nullptr;
        CK(cudaMalloc(&p, count * sizeof(T)));
        gpu_allocs.push_back(p);
        return p;
    }

    virtual const uint8_t* find_tensor(const char* name, const strata::TensorInfo** out = nullptr) const = 0;

    float* upload_f32(const char* name, size_t expected_elems = 0) {
        const strata::TensorInfo* t = nullptr;
        const uint8_t* raw = find_tensor(name, &t);
        if (!raw || !t) {
            std::fprintf(stderr, "strata-inkling2: missing tensor %s\n", name);
            std::exit(1);
        }
        const int64_t ne = (int64_t) t->elements();
        if (expected_elems && (size_t) ne != expected_elems) {
            std::fprintf(stderr, "strata-inkling2: tensor %s expected %zu elems got %lld\n", name, expected_elems, (long long) ne);
            std::exit(1);
        }
        float* d = gpu_alloc<float>((size_t) ne);
        if (t->type == 0) {
            CK(cudaMemcpy(d, raw, (size_t) ne * sizeof(float), cudaMemcpyHostToDevice));
        } else if (t->type == 1) {
            std::vector<float> tmp((size_t) ne);
            const uint16_t* h = (const uint16_t*) raw;
            for (int64_t i = 0; i < ne; ++i) {
                const uint16_t bits = h[i];
                const uint32_t s = (uint32_t) (bits & 0x8000u) << 16;
                const uint32_t e = (bits >> 10) & 0x1Fu;
                const uint32_t m = bits & 0x03FFu;
                uint32_t u = s;
                if (e == 0) {
                    if (m != 0) {
                        float v = std::ldexp((float) m, -24);
                        std::memcpy(&u, &v, 4);
                        u |= s;
                    }
                } else if (e == 31) {
                    u |= 0x7F800000u | (m << 13);
                } else {
                    u |= ((e + 112u) << 23) | (m << 13);
                }
                std::memcpy(&tmp[(size_t) i], &u, 4);
            }
            CK(cudaMemcpy(d, tmp.data(), (size_t) ne * sizeof(float), cudaMemcpyHostToDevice));
        } else if (t->type == 30) {
            std::vector<float> tmp((size_t) ne);
            const uint16_t* b = (const uint16_t*) raw;
            for (int64_t i = 0; i < ne; ++i) {
                uint32_t u = (uint32_t) b[i] << 16;
                std::memcpy(&tmp[(size_t) i], &u, 4);
            }
            CK(cudaMemcpy(d, tmp.data(), (size_t) ne * sizeof(float), cudaMemcpyHostToDevice));
        } else {
            std::fprintf(stderr, "strata-inkling2: unsupported F32 param type %u on %s\n", t->type, name);
            std::exit(1);
        }
        return d;
    }

    DenseMat upload_dense(const char* name, size_t rows_override = 0, size_t cols_expected = 0,
                          bool pack_q8_in_place = false, bool pack_q8_extra = false) {
        const strata::TensorInfo* t = nullptr;
        const uint8_t* raw = find_tensor(name, &t);
        if (!raw || !t || (t->type != 0 && t->type != 1 && t->type != 30 && !K::native_mmvq_supported((int) t->type))) {
            std::fprintf(stderr, "strata-inkling2: missing/unsupported dense tensor %s (type=%u)\n", name, t ? t->type : 999u);
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
        if (rb == 0) {
            std::fprintf(stderr, "strata-inkling2: %s invalid row bytes (type=%u ne0=%lld)\n", name, t->type, (long long) ne0);
            std::exit(1);
        }
        const size_t bytes = rb * (size_t) ne1;
        uint8_t* d = gpu_alloc<uint8_t>(bytes);
        bool packed_in_place = false;
        if (t->type == 8 && pack_q8_in_place && K::native_q8_0_packed_eligible((int) ne0, (int) ne1)) {
            std::vector<uint8_t> packed(bytes);
            K::native_q8_0_pack_host(raw, packed.data(), (int) ne0, (int) ne1);
            CK(cudaMemcpy(d, packed.data(), bytes, cudaMemcpyHostToDevice));
            K::native_q8_0_packed_register(d, d, (int) ne0, (int) ne1);
            q8_packed_keys.push_back(d);
            packed_in_place = true;
        } else {
            CK(cudaMemcpy(d, raw, bytes, cudaMemcpyHostToDevice));
            if (t->type == 8 && pack_q8_extra && K::native_q8_0_packed_enabled() && K::native_q8_0_packed_eligible((int) ne0, (int) ne1)) {
                std::vector<uint8_t> packed(bytes);
                K::native_q8_0_pack_host(raw, packed.data(), (int) ne0, (int) ne1);
                uint8_t* pd = gpu_alloc<uint8_t>(bytes);
                CK(cudaMemcpy(pd, packed.data(), bytes, cudaMemcpyHostToDevice));
                K::native_q8_0_packed_register(d, pd, (int) ne0, (int) ne1);
                q8_packed_keys.push_back(d);
            }
        }
        if (t->type != 30 && t->type != 0 && t->type != 1) has_quant_dense = true;
        return DenseMat{(int) t->type, ne0, ne1, d, bytes, packed_in_place};
    }
};

struct PhysLayer {
    bool is_swa = true;
    bool has_loop = false;

    // Dense projections on GPU (BF16 or native GGUF quantized)
    DenseMat attn_q;          // [n_embd, q_dim]
    DenseMat attn_k;          // [n_embd, kv_dim]
    DenseMat attn_v;          // [n_embd, kv_dim]
    DenseMat attn_output;     // [q_dim, n_embd]
    DenseMat attn_gate;       // [n_embd, q_dim]
    DenseMat attn_sink_proj;  // [n_embd, n_head]
    DenseMat attn_rel;        // [n_embd, n_rel * n_head]
    DenseMat attn_qr;         // concatenated [attn_q; attn_rel] for FP32 QR (+ optional sink) GEMM
    DenseMat attn_gkv;        // concatenated [attn_gate; attn_k; attn_v] for SWA FP16-out GEMM
    DenseMat attn_qgr;        // concatenated [attn_q; attn_rel; attn_gate] for T==1 single MMVQ / FP32 fallback
    DenseMat attn_qgrkv;      // concatenated [attn_q; attn_rel; attn_gate; attn_k; attn_v] for SWA !has_msg T==1 / FP32 fallback
    DenseMat attn_kkc;        // concatenated [attn_k; attn_k_conv_proj] for !is_swa T==1
    DenseMat attn_vvc;        // concatenated [attn_v; attn_v_conv_proj] for !is_swa T==1
    DenseMat attn_k_conv_proj; // [n_embd, n_conv * n_head_kv] (!is_swa)
    DenseMat attn_v_conv_proj; // [n_embd, n_conv * n_head_kv] (!is_swa)

    // Loop controllers [4]: 0=attn, 1=k, 2=v, 3=mlp
    DenseMat loop_down[4] = {};   // [2*n_embd, 64]
    DenseMat loop_gain[4] = {};   // [64, n_embd]
    DenseMat loop_forget[4] = {}; // [64, n_embd]

    // F32 params on GPU
    float* attn_norm = nullptr;          // SWA
    float* attn_q_inp_norm = nullptr;    // !is_swa
    float* attn_k_inp_norm = nullptr;    // !is_swa
    float* attn_v_inp_norm = nullptr;    // !is_swa
    float* attn_q_clip = nullptr;        // [head_dim, n_head]
    float* attn_rel_clip = nullptr;      // [n_rel, n_head]
    float* attn_rel_bias = nullptr;      // [n_rel_ext, n_rel] transposed -> wait: GGUF is [n_rel, n_ext], i.e. d*n_rel + r!
    __half* attn_rel_bias_f16_hi = nullptr; // [n_rel_ext, n_rel] Ozaki hi (pre-split at model load)
    __half* attn_rel_bias_f16_lo = nullptr; // [n_rel_ext, n_rel] Ozaki lo (pre-split at model load)
    float* attn_k_conv = nullptr;        // [4, kv_dim] -> c*4 + lag
    float* attn_v_conv = nullptr;
    float* attn_post_conv = nullptr;     // [4, n_embd]
    float* ffn_post_conv = nullptr;      // [4, n_embd]
    float* attn_res_query = nullptr;     // [n_embd, 1 or 3] (null on phys 0)
    float* ffn_res_query = nullptr;      // [n_embd]
    float* ffn_norm = nullptr;           // [n_embd]
    float* ffn_gate_inp = nullptr;       // [n_embd, n_expert]
    __half* ffn_gate_inp_hi = nullptr;   // [n_expert, n_embd] Ozaki-2 hi (pre-split at model load, sm_80+)
    __half* ffn_gate_inp_lo = nullptr;   // [n_expert, n_embd] Ozaki-2 lo (pre-split at model load, sm_80+)
    float* exp_probs_b = nullptr;        // [n_expert]

    // MoE experts (host mmap source + packed blobs + VRAM cache)
    int gu_type = -1;
    int d_type = -1;
    size_t gu_row = 0;
    size_t d_row = 0;
    const uint8_t* h_gate_up = nullptr;
    const uint8_t* h_down = nullptr;
    const float* h_gu_scale = nullptr;   // optional (NVFP4)
    const float* h_d_scale = nullptr;    // optional (NVFP4)

    K::NativeExpertLayout L{};
#if defined(STRATA_INKLING2_CPU_POOL)
    strata::kernels::cpu::NativeFmt cpu_fmt{};
#endif
    uint8_t* hblobs = nullptr;           // [n_expert][L.bytes] on host (lazy or pre-built)
    bool hblobs_pinned = false;          // true if hblobs was allocated with cudaHostAlloc
    std::vector<uint8_t*> sparse_hblobs; // [n_expert] lazy per-expert pageable buffers when hblobs == nullptr
    uint8_t* d_blobs = nullptr;          // [n_slots][stride] in VRAM
    bool d_blobs_from_pool = false;      // Wave 6: true if d_blobs was allocated from Engine::phase_pool
    size_t stride = 0;
    int n_slots = 0;
    int32_t* d_slot_of = nullptr;        // [n_expert] in VRAM
    uint32_t* d_hits = nullptr;          // [n_expert] GPU routing frequency counter (even when n_slots == ne)
    std::vector<int32_t> slot_of;        // [n_expert] on host (-1 = CPU)
    std::vector<int32_t> exp_in_slot;    // [n_slots]
    std::vector<double> score;           // [n_expert]
    std::vector<double> prior_score;     // [n_expert] static decode frequency prior
};

struct Model : GpuWeightStore {
    std::unique_ptr<strata::GgufModel> gguf;
    Hparams hp;

    const uint8_t* host_tok_embd = nullptr;    // [vocab_padded, n_embd] on host (mmap)
    int tok_embd_type = -1;
    size_t tok_embd_row_bytes = 0;
    float* tok_embd_norm = nullptr;
    float* loop_res_query = nullptr;
    float* loop_kv_res_query = nullptr;
    float* output_res_query = nullptr;
    float* output_norm = nullptr;
    DenseMat output_weight;                    // [vocab_unpadded, n_embd] in VRAM

    std::vector<PhysLayer> layers;

    ~Model() override {
        for (auto& ly : layers) {
            for (uint8_t* p : ly.sparse_hblobs) std::free(p);
            ly.sparse_hblobs.clear();
        }
        free_store();
    }

    const uint8_t* find_tensor(const char* name, const strata::TensorInfo** out = nullptr) const override {
        size_t shard = 0;
        const strata::TensorInfo* t = gguf->find(name, &shard);
        if (out) *out = t;
        return t ? gguf->shard(shard).tensor_data(*t) : nullptr;
    }

    void dequant_tok(int32_t tok, float* dst) const {
        const int N = (int) hp.n_embd;
        const uint8_t* row = host_tok_embd + (size_t) tok * tok_embd_row_bytes;
        if (tok_embd_type == 30) {
            const uint16_t* bf = (const uint16_t*) row;
            for (int i = 0; i < N; ++i) {
                uint32_t u = (uint32_t) bf[i] << 16;
                std::memcpy(&dst[i], &u, 4);
            }
        } else if (tok_embd_type == 1) {
            const uint16_t* h = (const uint16_t*) row;
            for (int i = 0; i < N; ++i) {
                const uint16_t bits = h[i];
                const uint32_t s = (uint32_t) (bits & 0x8000u) << 16;
                const uint32_t e = (bits >> 10) & 0x1Fu;
                const uint32_t m = bits & 0x03FFu;
                uint32_t u = s;
                if (e == 0) {
                    if (m != 0) {
                        float v = std::ldexp((float) m, -24);
                        std::memcpy(&u, &v, 4);
                        u |= s;
                    }
                } else if (e == 31) {
                    u |= 0x7F800000u | (m << 13);
                } else {
                    u |= ((e + 112u) << 23) | (m << 13);
                }
                std::memcpy(&dst[i], &u, 4);
            }
        } else if (tok_embd_type == 0) {
            std::memcpy(dst, row, (size_t) N * sizeof(float));
        } else {
            strata::quant::dequant_row((strata::quant::Type) tok_embd_type, row, dst, N);
        }
    }
};

#include "inkling2_dspark.cuh"

// Bijective 0x8E/0x71 nibble bit permutation LUT for NVFP4 K-Tile packing.
static constexpr std::array<uint8_t, 256> kNvfp4KtilePermLut = [] {
    std::array<uint8_t, 256> t{};
    for (int b = 0; b < 256; ++b) {
        t[(size_t) b] = (uint8_t) (
            ((b & 0x08) << 4) |   // l3 -> bit 7
            ( b & 0x70)       |   // h2..h0 -> bits 6..4
            ((b & 0x07) << 1) |   // l2..l0 -> bits 3..1
            ((b & 0x80) >> 7));   // h3 -> bit 0
    }
    return t;
}();

inline void pack_nvfp4_ktile_blk(uint8_t* kt, size_t s_off, size_t q0_off, size_t q1_off, const uint8_t* blk) {
    std::memcpy(kt + s_off, blk, 4u);
    uint8_t* p0 = kt + q0_off;
    const uint8_t* s0 = blk + 4u;
    for (int i = 0; i < 16; ++i) p0[i] = kNvfp4KtilePermLut[s0[i]];
    uint8_t* p1 = kt + q1_off;
    const uint8_t* s1 = blk + 20u;
    for (int i = 0; i < 16; ++i) p1[i] = kNvfp4KtilePermLut[s1[i]];
}

void build_expert_blob(const PhysLayer& L, int64_t e, uint8_t* dst) {
    const int64_t n_ff = L.L.n_ff, n_embd = L.L.n_embd;
    const size_t gu_expert_bytes = (size_t) (2 * n_ff) * L.gu_row;
    const size_t d_expert_bytes = (size_t) n_embd * L.d_row;
    const uint8_t* gu_src = L.h_gate_up + (size_t) e * gu_expert_bytes;
    const uint8_t* d_src = L.h_down + (size_t) e * d_expert_bytes;
    // Gate+Up packing (decoupled from Down so nvfp4_ktile, q5k_ktile, q8_0_ktile, and raw row-major compose cleanly).
    if (L.L.nvfp4_ktile) {
        // Wave 13B: Coalesced NVFP4 K-Tile layout (0 extra bytes; exact same down_off / tail_off / bytes / stride).
        // Loop nt outer, wr middle, kb inner so mmap GGUF reads are sequential while scattering into L2-resident tiles.
        const int64_t n_gu_tiles = n_ff / 64;
        const int64_t k_gu_blks = n_embd / 64;
        for (int64_t nt = 0; nt < n_gu_tiles; ++nt) {
            const int64_t col0 = nt * 64;
            uint8_t* const nt_base = dst + (size_t) nt * (size_t) k_gu_blks * 4608u;
            for (int64_t wr = 0; wr < 128; ++wr) {
                const uint8_t* const row_ptr = (wr < 64)
                    ? (gu_src + (size_t) (col0 + wr) * L.gu_row)
                    : (gu_src + (size_t) (n_ff + col0 + (wr - 64)) * L.gu_row);
                const size_t s_off = (size_t) wr * 4u;
                const size_t q0_off = 512u + (size_t) wr * 16u;
                const size_t q1_off = 2560u + (size_t) wr * 16u;
                for (int64_t kb = 0; kb < k_gu_blks; ++kb) {
                    const uint8_t* const blk = row_ptr + (size_t) kb * 36u;
                    uint8_t* const kt = nt_base + (size_t) kb * 4608u;
                    pack_nvfp4_ktile_blk(kt, s_off, q0_off, q1_off, blk);
                }
            }
        }
    } else if (L.L.q5k_ktile) {
        // Coalesced Q5_K Gate+Up K-Tile layout: [n_ff/64, n_embd/256, 22528 B], wr=0..63 gate, wr=64..127 up.
        const int64_t n_gu_tiles = n_ff / 64;
        const int64_t k_gu_blks = n_embd / 256;
        for (int64_t nt = 0; nt < n_gu_tiles; ++nt) {
            const int64_t col0 = nt * 64;
            uint8_t* const nt_base = dst + (size_t) nt * (size_t) k_gu_blks * 22528u;
            for (int64_t wr = 0; wr < 128; ++wr) {
                const uint8_t* const row_ptr = (wr < 64)
                    ? (gu_src + (size_t) (col0 + wr) * L.gu_row)
                    : (gu_src + (size_t) (n_ff + col0 + (wr - 64)) * L.gu_row);
                const size_t h_off  = (size_t) wr * 16u;
                const size_t q0_off = 2048u  + (size_t) wr * 32u;
                const size_t q1_off = 6144u  + (size_t) wr * 32u;
                const size_t q2_off = 10240u + (size_t) wr * 32u;
                const size_t q3_off = 14336u + (size_t) wr * 32u;
                const size_t q4_off = 18432u + (size_t) wr * 32u;
                for (int64_t kb = 0; kb < k_gu_blks; ++kb) {
                    const uint8_t* const blk = row_ptr + (size_t) kb * 176u;
                    uint8_t* const kt = nt_base + (size_t) kb * 22528u;
                    std::memcpy(kt + h_off,  blk + 0u,   16u);
                    std::memcpy(kt + q0_off, blk + 16u,  32u);
                    std::memcpy(kt + q1_off, blk + 48u,  32u);
                    std::memcpy(kt + q2_off, blk + 80u,  32u);
                    std::memcpy(kt + q3_off, blk + 112u, 32u);
                    std::memcpy(kt + q4_off, blk + 144u, 32u);
                }
            }
        }
    } else {
        // [gate (n_ff rows) | up (n_ff rows)]
        std::memcpy(dst, gu_src, (size_t) n_ff * L.gu_row);
        std::memcpy(dst + L.L.up_off, gu_src + (size_t) n_ff * L.gu_row, (size_t) n_ff * L.gu_row);
    }

    // Down packing
    if (L.L.nvfp4_ktile) {
        const int64_t n_d_tiles = n_embd / 128;
        const int64_t k_d_blks = n_ff / 64;
        for (int64_t nt = 0; nt < n_d_tiles; ++nt) {
            const int64_t col0 = nt * 128;
            uint8_t* const nt_base = dst + L.L.down_off + (size_t) nt * (size_t) k_d_blks * 4608u;
            for (int64_t wr = 0; wr < 128; ++wr) {
                const uint8_t* const row_ptr = d_src + (size_t) (col0 + wr) * L.d_row;
                const size_t s_off = (size_t) wr * 4u;
                const size_t q0_off = 512u + (size_t) wr * 16u;
                const size_t q1_off = 2560u + (size_t) wr * 16u;
                for (int64_t kb = 0; kb < k_d_blks; ++kb) {
                    const uint8_t* const blk = row_ptr + (size_t) kb * 36u;
                    uint8_t* const kt = nt_base + (size_t) kb * 4608u;
                    pack_nvfp4_ktile_blk(kt, s_off, q0_off, q1_off, blk);
                }
            }
        }
    } else if (L.L.q8_0_ktile) {
        // Coalesced Q8_0 Down K-Tile layout: [n_embd/128, n_ff/64, 8704 B], wr=0..127 down.
        const int64_t n_d_tiles = n_embd / 128;
        const int64_t k_d_blks = n_ff / 64;
        for (int64_t nt = 0; nt < n_d_tiles; ++nt) {
            const int64_t col0 = nt * 128;
            uint8_t* const nt_base = dst + L.L.down_off + (size_t) nt * (size_t) k_d_blks * 8704u;
            for (int64_t wr = 0; wr < 128; ++wr) {
                const uint8_t* const row_ptr = d_src + (size_t) (col0 + wr) * L.d_row;
                const size_t d_off  = (size_t) wr * 4u;
                const size_t q0_off = 512u  + (size_t) wr * 32u;
                const size_t q1_off = 4608u + (size_t) wr * 32u;
                for (int64_t kb = 0; kb < k_d_blks; ++kb) {
                    const uint8_t* const blk = row_ptr + (size_t) kb * 68u;
                    uint8_t* const kt = nt_base + (size_t) kb * 8704u;
                    std::memcpy(kt + d_off + 0u, blk + 0u,  2u);
                    std::memcpy(kt + d_off + 2u, blk + 34u, 2u);
                    std::memcpy(kt + q0_off,     blk + 2u,  32u);
                    std::memcpy(kt + q1_off,     blk + 36u, 32u);
                }
            }
        }
    } else {
        std::memcpy(dst + L.L.down_off, d_src, d_expert_bytes);
    }
    if (L.L.tail_off) {
        const float sg = L.h_gu_scale ? L.h_gu_scale[e] : 1.0f;
        const float sd = L.h_d_scale ? L.h_d_scale[e] : 1.0f;
        const float tail[4] = {sg, sg, sd, 0.0f};
        std::memcpy(dst + L.L.tail_off, tail, sizeof tail);
    }
}

void build_all_expert_blobs(const PhysLayer& L, int ne, uint8_t* dst_base) {
    const int hw = (int) std::thread::hardware_concurrency();
    const int nt = std::clamp(hw > 0 ? hw : 4, 1, std::min(8, ne));
    if (nt <= 1) {
        for (int e = 0; e < ne; ++e) build_expert_blob(L, e, dst_base + (size_t) e * L.stride);
        return;
    }
    std::vector<std::thread> ths;
    ths.reserve((size_t) nt);
    for (int t = 0; t < nt; ++t) {
        const int e0 = (ne * t) / nt;
        const int e1 = (ne * (t + 1)) / nt;
        ths.emplace_back([&, e0, e1]() {
            for (int e = e0; e < e1; ++e) build_expert_blob(L, e, dst_base + (size_t) e * L.stride);
        });
    }
    for (auto& th : ths) th.join();
}

inline void drop_mmap_pages_interior(const void* ptr, size_t bytes) {
#if defined(__linux__)
    if (!ptr || bytes == 0) return;
    const long pgsz = sysconf(_SC_PAGESIZE);
    if (pgsz <= 0) return;
    const uintptr_t p0 = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t p_start = (p0 + (uintptr_t) pgsz - 1u) & ~(uintptr_t) ((size_t) pgsz - 1u);
    const uintptr_t p_end = (p0 + bytes) & ~(uintptr_t) ((size_t) pgsz - 1u);
    if (p_end > p_start) {
        (void) madvise(reinterpret_cast<void*>(p_start), (size_t) (p_end - p_start), MADV_DONTNEED);
    }
#else
    (void) ptr; (void) bytes;
#endif
}

inline void drop_layer_mmap_pages(const PhysLayer& L, int n_expert) {
    const size_t gu_bytes = (size_t) n_expert * (size_t) (2 * L.L.n_ff) * L.gu_row;
    const size_t d_bytes = (size_t) n_expert * (size_t) L.L.n_embd * L.d_row;
    drop_mmap_pages_interior(L.h_gate_up, gu_bytes);
    drop_mmap_pages_interior(L.h_down, d_bytes);
}

bool load_model(const std::string& path, Model& m, std::string& err) {
    const auto t0 = Clock::now();
    try {
        m.gguf = std::make_unique<strata::GgufModel>(strata::gguf_split_paths(path));
    } catch (const std::exception& ex) {
        err = ex.what();
        return false;
    }
    const strata::GgufFile& meta = m.gguf->meta();
    err = strata::check_architecture(meta);
    if (!err.empty()) return false;
    if (strata::model_family(meta) != strata::ModelFamily::Inkling2) {
        err = "expected architecture inkling2";
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
    Hparams& hp = m.hp;
    hp.n_layer = get_i("inkling2.block_count", 48);
    hp.num_loops = get_i("inkling2.num_loops", 3);
    hp.core_start = get_i("inkling2.loop.core_start", 6);
    hp.core_end = get_i("inkling2.loop.core_end", 18);
    const int64_t n_core = hp.num_loops > 1 ? (hp.core_end - hp.core_start) : 0;
    hp.n_phys = hp.n_layer - (hp.num_loops - 1) * n_core;
    hp.n_embd = get_i("inkling2.embedding_length", 2304);
    hp.n_ff = get_i("inkling2.expert_feed_forward_length", get_i("inkling2.feed_forward_length", 1152));
    if (const auto* v = meta.get("inkling2.attention.head_count")) {
        hp.n_head = !v->items.empty() ? (int64_t) v->items[0].u : (int64_t) v->u;
    }
    if (const auto* v = meta.get("inkling2.attention.head_count_kv")) {
        hp.n_head_kv = !v->items.empty() ? (int64_t) v->items[0].u : (int64_t) v->u;
    }
    hp.head_dim = get_i("inkling2.attention.key_length", 128);
    hp.q_dim = hp.n_head * hp.head_dim;
    hp.kv_dim = hp.n_head_kv * hp.head_dim;
    hp.n_expert = get_i("inkling2.expert_count", 128);
    hp.n_expert_used = get_i("inkling2.expert_used_count", 8);
    hp.n_rel = get_i("inkling2.attention.relative_length", 16);
    hp.n_rel_ext = get_i("inkling2.attention.relative_extent", 1024);
    hp.n_swa = get_i("inkling2.attention.sliding_window", 1024);
    hp.attn_res_block_size = get_i("inkling2.attn_res.block_size", 6);
    hp.n_conv = get_i("inkling2.shortconv.l_cache", 4);
    hp.loop_gate_rank = get_i("inkling2.loop.gate_rank", 64);
    hp.vocab_padded = get_i("inkling2.vocab_size", 201024);
    hp.vocab_unpadded = get_i("inkling2.vocab_size_unpadded", 200058);
    hp.log_scale_floor = get_i("inkling2.attention.log_scale_floor", 8192);
    hp.log_scale_alpha = get_f("inkling2.attention.log_scale_alpha", 0.1f);
    hp.rms_eps = get_f("inkling2.attention.layer_norm_rms_epsilon", 1e-6f);
    hp.qk_clip = get_f("inkling2.attention.qk_clip", 22.6274166f);
    hp.rel_clip = get_f("inkling2.attention.relative_clip", 8.0f);
    hp.situ_beta = get_f("inkling2.activation.situ_beta", 4.0f);
    hp.situ_linear_beta = get_f("inkling2.activation.situ_linear_beta", 25.0f);
    hp.expert_weights_scale = get_f("inkling2.expert_weights_scale", 8.0f);
    hp.softcap = get_f("inkling2.final_logit_softcapping", 30.0f);
    if (const auto* v = meta.get("inkling2.attention.sliding_window_pattern")) {
        hp.swa_pattern.resize(v->items.size());
        for (size_t i = 0; i < v->items.size(); ++i) hp.swa_pattern[i] = v->items[i].u ? 1 : 0;
    }

    // Global tensors (token_embd kept in host mmap)
    const strata::TensorInfo* te_info = nullptr;
    m.host_tok_embd = m.find_tensor("token_embd.weight", &te_info);
    if (!m.host_tok_embd || !te_info) {
        err = "token_embd.weight missing";
        return false;
    }
    m.tok_embd_type = (int) te_info->type;
    if (m.tok_embd_type == 30 || m.tok_embd_type == 1) m.tok_embd_row_bytes = (size_t) hp.n_embd * sizeof(uint16_t);
    else if (m.tok_embd_type == 0) m.tok_embd_row_bytes = (size_t) hp.n_embd * sizeof(float);
    else if (strata::quant::can_dequant((strata::quant::Type) m.tok_embd_type)) {
        m.tok_embd_row_bytes = strata::quant::row_bytes(m.tok_embd_type, hp.n_embd);
    }
    if (m.tok_embd_row_bytes == 0) {
        err = "token_embd.weight unsupported type " + std::to_string(m.tok_embd_type);
        return false;
    }
#if defined(__linux__)
    if (m.host_tok_embd && m.tok_embd_row_bytes > 0) {
        const size_t n_rows_embd = (te_info->shape.size() > 1) ? (size_t) te_info->shape[1] : (size_t) hp.vocab_padded;
        const size_t tok_embd_bytes = n_rows_embd * m.tok_embd_row_bytes;
        const long pgsz = sysconf(_SC_PAGESIZE);
        if (pgsz > 0 && tok_embd_bytes > 0) {
            const uintptr_t p0 = reinterpret_cast<uintptr_t>(m.host_tok_embd);
            const uintptr_t p_aligned = p0 & ~(uintptr_t) ((size_t) pgsz - 1u);
            const size_t len_aligned = tok_embd_bytes + (size_t) (p0 - p_aligned);
            (void) madvise(reinterpret_cast<void*>(p_aligned), len_aligned, MADV_WILLNEED);
        }
    }
#endif
    m.tok_embd_norm = m.upload_f32("token_embd_norm.weight", (size_t) hp.n_embd);
    if (hp.num_loops > 1) {
        m.loop_res_query = m.upload_f32("loop_res_query.weight", (size_t) hp.n_embd * 2 * n_core);
        m.loop_kv_res_query = m.upload_f32("loop_kv_res_query.weight", (size_t) hp.n_embd * 2 * n_core);
    }
    m.output_res_query = m.upload_f32("output_res_query.weight", (size_t) hp.n_embd);
    m.output_norm = m.upload_f32("output_norm.weight", (size_t) hp.n_embd);
    const bool pack_head = std::getenv("STRATA_Q8_HEAD_PACK") != nullptr && std::getenv("STRATA_Q8_HEAD_PACK")[0] == '1';
    m.output_weight = m.upload_dense("output.weight", (size_t) hp.vocab_unpadded, (size_t) hp.n_embd, pack_head, false);

    int lm_dev = 0, lm_cc_maj = 0;
    if (cudaGetDevice(&lm_dev) == cudaSuccess) {
        (void) cudaDeviceGetAttribute(&lm_cc_maj, cudaDevAttrComputeCapabilityMajor, lm_dev);
    }
    (void) cudaGetLastError();

    m.layers.resize((size_t) hp.n_phys);
    char nm[128];
    for (int ip = 0; ip < hp.n_phys; ++ip) {
        PhysLayer& L = m.layers[(size_t) ip];
        L.is_swa = hp.is_swa(ip);
        L.has_loop = (hp.num_loops > 1 && ip >= hp.core_start && ip < hp.core_end);
        const bool pack_core = L.has_loop;

        auto tn = [&](const char* suffix) -> const char* {
            std::snprintf(nm, sizeof nm, "blk.%d.%s", ip, suffix);
            return nm;
        };
        const char* nm_q = tn("attn_q.weight");
        const strata::TensorInfo* tq = nullptr;
        const uint8_t* raw_q = m.find_tensor(nm_q, &tq);
        const strata::TensorInfo *tg = nullptr, *tr = nullptr, *tk = nullptr, *tv = nullptr;
        std::string sn_g = "blk." + std::to_string(ip) + ".attn_gate.weight";
        std::string sn_r = "blk." + std::to_string(ip) + ".attn_rel.weight";
        std::string sn_k = "blk." + std::to_string(ip) + ".attn_k.weight";
        std::string sn_v = "blk." + std::to_string(ip) + ".attn_v.weight";
        const uint8_t* raw_g = m.find_tensor(sn_g.c_str(), &tg);
        const uint8_t* raw_r = m.find_tensor(sn_r.c_str(), &tr);
        const uint8_t* raw_k = m.find_tensor(sn_k.c_str(), &tk);
        const uint8_t* raw_v = m.find_tensor(sn_v.c_str(), &tv);
        const bool qgr_ok = raw_q && tq && raw_g && tg && raw_r && tr &&
                            tq->type == tg->type && tq->type == tr->type &&
                            tq->shape[0] == tg->shape[0] && tq->shape[0] == tr->shape[0] &&
                            tq->type != 0 && tq->type != 1 && tq->type != 30 && K::native_mmvq_supported((int) tq->type);
        const bool kv_ok_in_qgr = qgr_ok && L.is_swa && raw_k && tk && raw_v && tv &&
                                  tq->type == tk->type && tq->type == tv->type &&
                                  tq->shape[0] == tk->shape[0] && tq->shape[0] == tv->shape[0];
        if (kv_ok_in_qgr) {
            const int64_t nin = (int64_t) tq->shape[0];
            const size_t rb = strata::quant::row_bytes((int) tq->type, nin);
            const int64_t nq_out = tq->shape.size() > 1 ? (int64_t) tq->shape[1] : 1;
            const int64_t ng_out = tg->shape.size() > 1 ? (int64_t) tg->shape[1] : 1;
            const int64_t nr_out = tr->shape.size() > 1 ? (int64_t) tr->shape[1] : 1;
            const int64_t nk_out = tk->shape.size() > 1 ? (int64_t) tk->shape[1] : 1;
            const int64_t nv_out = tv->shape.size() > 1 ? (int64_t) tv->shape[1] : 1;
            const size_t bq = rb * (size_t) nq_out;
            const size_t bg = rb * (size_t) ng_out;
            const size_t br = rb * (size_t) nr_out;
            const size_t bk = rb * (size_t) nk_out;
            const size_t bv = rb * (size_t) nv_out;
            uint8_t* dall = m.gpu_alloc<uint8_t>(bq + br + bg + bk + bv);
            CK(cudaMemcpy(dall, raw_q, bq, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dall + bq, raw_r, br, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dall + bq + br, raw_g, bg, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dall + bq + br + bg, raw_k, bk, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dall + bq + br + bg + bk, raw_v, bv, cudaMemcpyHostToDevice));
            m.has_quant_dense = true;
            L.attn_q = DenseMat{(int) tq->type, nin, nq_out, dall, bq};
            L.attn_rel = DenseMat{(int) tr->type, nin, nr_out, dall + bq, br};
            L.attn_gate = DenseMat{(int) tg->type, nin, ng_out, dall + bq + br, bg};
            L.attn_k = DenseMat{(int) tk->type, nin, nk_out, dall + bq + br + bg, bk};
            L.attn_v = DenseMat{(int) tv->type, nin, nv_out, dall + bq + br + bg + bk, bv};
            L.attn_qr = DenseMat{(int) tq->type, nin, nq_out + nr_out, dall, bq + br};
            L.attn_gkv = DenseMat{(int) tg->type, nin, ng_out + nk_out + nv_out, dall + bq + br, bg + bk + bv};
            L.attn_qgr = DenseMat{(int) tq->type, nin, nq_out + nr_out + ng_out, dall, bq + br + bg};
            L.attn_qgrkv = DenseMat{(int) tq->type, nin, nq_out + nr_out + ng_out + nk_out + nv_out, dall, bq + br + bg + bk + bv};
        } else if (qgr_ok) {
            const int64_t nin = (int64_t) tq->shape[0];
            const size_t rb = strata::quant::row_bytes((int) tq->type, nin);
            const int64_t nq_out = tq->shape.size() > 1 ? (int64_t) tq->shape[1] : 1;
            const int64_t ng_out = tg->shape.size() > 1 ? (int64_t) tg->shape[1] : 1;
            const int64_t nr_out = tr->shape.size() > 1 ? (int64_t) tr->shape[1] : 1;
            const size_t bq = rb * (size_t) nq_out;
            const size_t bg = rb * (size_t) ng_out;
            const size_t br = rb * (size_t) nr_out;
            uint8_t* dqgr = m.gpu_alloc<uint8_t>(bq + br + bg);
            CK(cudaMemcpy(dqgr, raw_q, bq, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dqgr + bq, raw_r, br, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dqgr + bq + br, raw_g, bg, cudaMemcpyHostToDevice));
            m.has_quant_dense = true;
            L.attn_q = DenseMat{(int) tq->type, nin, nq_out, dqgr, bq};
            L.attn_rel = DenseMat{(int) tr->type, nin, nr_out, dqgr + bq, br};
            L.attn_gate = DenseMat{(int) tg->type, nin, ng_out, dqgr + bq + br, bg};
            L.attn_qr = DenseMat{(int) tq->type, nin, nq_out + nr_out, dqgr, bq + br};
            L.attn_qgr = DenseMat{(int) tq->type, nin, nq_out + nr_out + ng_out, dqgr, bq + br + bg};
        } else {
            L.attn_q = m.upload_dense(tn("attn_q.weight"), 0, 0, false, pack_core);
            L.attn_gate = m.upload_dense(tn("attn_gate.weight"), 0, 0, false, pack_core);
            L.attn_rel = m.upload_dense(tn("attn_rel.weight"));
        }
        if (!kv_ok_in_qgr) {
            if (!L.is_swa) {
                const strata::TensorInfo *tkc = nullptr, *tvc = nullptr;
                std::string sn_kc = "blk." + std::to_string(ip) + ".attn_k_conv_proj.weight";
                std::string sn_vc = "blk." + std::to_string(ip) + ".attn_v_conv_proj.weight";
                const uint8_t* raw_kc = m.find_tensor(sn_kc.c_str(), &tkc);
                const uint8_t* raw_vc = m.find_tensor(sn_vc.c_str(), &tvc);
                auto upload_pair_k = [&](const uint8_t* ra, const strata::TensorInfo* ta, const uint8_t* rb_raw, const strata::TensorInfo* tb,
                                         DenseMat& out_a, DenseMat& out_b, DenseMat& out_pair, const char* nm_a, const char* nm_b) {
                    if (ra && ta && rb_raw && tb && ta->type == tb->type && ta->shape[0] == tb->shape[0] &&
                        ta->type != 0 && ta->type != 1 && ta->type != 30 && K::native_mmvq_supported((int) ta->type)) {
                        const int64_t nin = (int64_t) ta->shape[0];
                        const size_t rbytes = strata::quant::row_bytes((int) ta->type, nin);
                        const int64_t na = ta->shape.size() > 1 ? (int64_t) ta->shape[1] : 1;
                        const int64_t nb = tb->shape.size() > 1 ? (int64_t) tb->shape[1] : 1;
                        const size_t ba = rbytes * (size_t) na, bb = rbytes * (size_t) nb;
                        uint8_t* dpair = m.gpu_alloc<uint8_t>(ba + bb);
                        CK(cudaMemcpy(dpair, ra, ba, cudaMemcpyHostToDevice));
                        CK(cudaMemcpy(dpair + ba, rb_raw, bb, cudaMemcpyHostToDevice));
                        m.has_quant_dense = true;
                        out_a = DenseMat{(int) ta->type, nin, na, dpair, ba};
                        out_b = DenseMat{(int) tb->type, nin, nb, dpair + ba, bb};
                        out_pair = DenseMat{(int) ta->type, nin, na + nb, dpair, ba + bb};
                    } else {
                        out_a = m.upload_dense(nm_a, 0, 0, false, pack_core);
                        out_b = m.upload_dense(nm_b);
                    }
                };
                upload_pair_k(raw_k, tk, raw_kc, tkc, L.attn_k, L.attn_k_conv_proj, L.attn_kkc, tn("attn_k.weight"), sn_kc.c_str());
                upload_pair_k(raw_v, tv, raw_vc, tvc, L.attn_v, L.attn_v_conv_proj, L.attn_vvc, tn("attn_v.weight"), sn_vc.c_str());
            } else {
                L.attn_k = m.upload_dense(tn("attn_k.weight"), 0, 0, false, pack_core);
                L.attn_v = m.upload_dense(tn("attn_v.weight"), 0, 0, false, pack_core);
            }
        }
        L.attn_output = m.upload_dense(tn("attn_output.weight"), 0, 0, false, pack_core);
        L.attn_sink_proj = m.upload_dense(tn("attn_sink_proj.weight"));

        if (L.is_swa) {
            L.attn_norm = m.upload_f32(tn("attn_norm.weight"), (size_t) hp.n_embd);
        } else {
            L.attn_q_inp_norm = m.upload_f32(tn("attn_q_inp_norm.weight"), (size_t) hp.n_embd);
            L.attn_k_inp_norm = m.upload_f32(tn("attn_k_inp_norm.weight"), (size_t) hp.n_embd);
            L.attn_v_inp_norm = m.upload_f32(tn("attn_v_inp_norm.weight"), (size_t) hp.n_embd);
        }

        if (L.has_loop) {
            static const char* mods[4] = {"attn", "k", "v", "mlp"};
            for (int j = 0; j < 4; ++j) {
                char s0[64], s1[64], s2[64];
                std::snprintf(s0, sizeof s0, "loop_%s_down.weight", mods[j]);
                std::snprintf(s1, sizeof s1, "loop_%s_gain.weight", mods[j]);
                std::snprintf(s2, sizeof s2, "loop_%s_forget.weight", mods[j]);
                L.loop_down[j] = m.upload_dense(tn(s0));
                L.loop_gain[j] = m.upload_dense(tn(s1));
                L.loop_forget[j] = m.upload_dense(tn(s2));
            }
        }

        L.attn_q_clip = m.upload_f32(tn("attn_q_clip.weight"), (size_t) hp.head_dim * hp.n_head);
        L.attn_rel_clip = m.upload_f32(tn("attn_rel_clip.weight"), (size_t) hp.n_rel * hp.n_head);
        const size_t rel_bias_elems = (size_t) hp.n_rel * (size_t) hp.n_rel_ext;
        L.attn_rel_bias = m.upload_f32(tn("attn_rel_bias.weight"), rel_bias_elems);
        L.attn_rel_bias_f16_hi = m.gpu_alloc<__half>(rel_bias_elems);
        L.attn_rel_bias_f16_lo = m.gpu_alloc<__half>(rel_bias_elems);
        k_split_ozaki_f32_to_f16_pair<<<(unsigned) ((rel_bias_elems + 255) / 256), 256>>>(
            L.attn_rel_bias, L.attn_rel_bias_f16_hi, L.attn_rel_bias_f16_lo, rel_bias_elems);
        CK(cudaGetLastError());
        L.attn_k_conv = m.upload_f32(tn("attn_k_conv.weight"), (size_t) hp.n_conv * hp.kv_dim);
        L.attn_v_conv = m.upload_f32(tn("attn_v_conv.weight"), (size_t) hp.n_conv * hp.kv_dim);
        L.attn_post_conv = m.upload_f32(tn("attn_post_conv.weight"), (size_t) hp.n_conv * hp.n_embd);
        L.ffn_post_conv = m.upload_f32(tn("ffn_post_conv.weight"), (size_t) hp.n_conv * hp.n_embd);
        if (ip > 0) L.attn_res_query = m.upload_f32(tn("attn_res_query.weight"), (size_t) hp.n_embd * (L.is_swa ? 1 : 3));
        L.ffn_res_query = m.upload_f32(tn("ffn_res_query.weight"), (size_t) hp.n_embd);
        L.ffn_norm = m.upload_f32(tn("ffn_norm.weight"), (size_t) hp.n_embd);
        const size_t router_elems = (size_t) hp.n_embd * (size_t) hp.n_expert;
        L.ffn_gate_inp = m.upload_f32(tn("ffn_gate_inp.weight"), router_elems);
        if (lm_cc_maj >= 8) {
            L.ffn_gate_inp_hi = m.gpu_alloc<__half>(router_elems);
            L.ffn_gate_inp_lo = m.gpu_alloc<__half>(router_elems);
            k_split_ozaki_f32_to_f16_pair<<<(unsigned) ((router_elems + 255) / 256), 256>>>(
                L.ffn_gate_inp, L.ffn_gate_inp_hi, L.ffn_gate_inp_lo, router_elems);
            CK(cudaGetLastError());
        }
        L.exp_probs_b = m.upload_f32(tn("exp_probs_b.bias"), (size_t) hp.n_expert);

        // Experts
        const strata::TensorInfo *ti_gu = nullptr, *ti_d = nullptr, *ti_gus = nullptr, *ti_ds = nullptr;
        L.h_gate_up = m.find_tensor(tn("ffn_gate_up_exps.weight"), &ti_gu);
        L.h_down = m.find_tensor(tn("ffn_down_exps.weight"), &ti_d);
        if (!L.h_gate_up || !L.h_down || !ti_gu || !ti_d) {
            err = std::string("missing expert tensors in layer ") + std::to_string(ip);
            return false;
        }
        L.gu_type = (int) ti_gu->type;
        L.d_type = (int) ti_d->type;
        L.h_gu_scale = (const float*) m.find_tensor(tn("ffn_gate_up_exps.scale"), &ti_gus);
        L.h_d_scale = (const float*) m.find_tensor(tn("ffn_down_exps.scale"), &ti_ds);

        if (L.gu_type == 30 && L.d_type == 30) {
            L.L.gu_type = 30;
            L.L.d_type = 30;
            L.L.n_embd = hp.n_embd;
            L.L.n_ff = hp.n_ff;
            L.L.gu_row = (size_t) hp.n_embd * 2;
            L.L.d_row = (size_t) hp.n_ff * 2;
            L.L.up_off = (size_t) hp.n_ff * L.L.gu_row;
            L.L.down_off = 2 * L.L.up_off;
            L.L.bytes = L.L.down_off + (size_t) hp.n_embd * L.L.d_row;
            L.L.tail_off = 0;
        } else {
            if (!K::native_expert_supported(L.gu_type, L.d_type, hp.n_embd, hp.n_ff)) {
                err = "unsupported expert format in layer " + std::to_string(ip);
                return false;
            }
            L.L = K::native_expert_layout(L.gu_type, L.d_type, hp.n_embd, hp.n_ff);
        }
        L.L.situ = true;
        L.L.situ_beta = hp.situ_beta;
        L.L.situ_linear_beta = hp.situ_linear_beta;
        L.gu_row = L.L.gu_row;
        L.d_row = L.L.d_row;
        L.stride = (L.L.bytes + 255) & ~(size_t) 255;

        static const bool no_nvfp4_ktile = [] {
            const char* v = std::getenv("STRATA_NO_NVFP4_KTILE");
            return v != nullptr && (v[0] == '\0' || std::atoi(v) != 0);
        }();
        const bool use_ktile = (!no_nvfp4_ktile && L.gu_type == 40 && L.d_type == 40 &&
                                (hp.n_ff % 64) == 0 && (hp.n_embd % 128) == 0 &&
                                L.gu_row == (size_t) (hp.n_embd / 64) * 36u &&
                                L.d_row == (size_t) (hp.n_ff / 64) * 36u &&
                                L.L.down_off == (size_t) (2 * hp.n_ff) * L.gu_row &&
                                L.L.tail_off == L.L.down_off + (size_t) hp.n_embd * L.d_row);
        L.L.nvfp4_ktile = use_ktile;
        static const bool no_epilogue_ktile = [] {
            const char* v = std::getenv("STRATA_NO_EPILOGUE_KTILE");
            return v != nullptr && (v[0] == '\0' || std::atoi(v) != 0);
        }();
        const bool use_q5k_ktile = (!no_epilogue_ktile && L.gu_type == 13 &&
                                    (hp.n_ff % 64) == 0 && (hp.n_embd % 256) == 0 &&
                                    L.gu_row == (size_t) (hp.n_embd / 256) * 176u &&
                                    L.L.down_off == (size_t) (2 * hp.n_ff) * L.gu_row);
        const bool use_q8_0_ktile = (!no_epilogue_ktile && L.d_type == 8 &&
                                     (hp.n_embd % 128) == 0 && (hp.n_ff % 64) == 0 &&
                                     L.d_row == (size_t) (hp.n_ff / 32) * 34u &&
                                     (L.L.down_off % 16u) == 0 &&
                                     L.L.bytes >= L.L.down_off + (size_t) hp.n_embd * L.d_row);
        L.L.q5k_ktile = use_q5k_ktile;
        L.L.q8_0_ktile = use_q8_0_ktile;

#if defined(STRATA_INKLING2_CPU_POOL)
        if (!strata::kernels::cpu::native_fmt(L.gu_type, L.d_type, hp.n_embd, hp.n_ff, L.cpu_fmt, err)) {
            return false;
        }
        L.cpu_fmt.situ = true;
        L.cpu_fmt.situ_beta = hp.situ_beta;
        L.cpu_fmt.situ_linear_beta = hp.situ_linear_beta;
        L.cpu_fmt.nvfp4_ktile = use_ktile;
        L.cpu_fmt.q5k_ktile = use_q5k_ktile;
        L.cpu_fmt.q8_0_ktile = use_q8_0_ktile;
#endif
    }
    std::fprintf(stderr, "strata-inkling2: dense + metadata loaded in %.2f s (n_phys=%lld, n_virt=%lld)\n",
                 ms_since(t0) / 1000.0, (long long) hp.n_phys, (long long) hp.n_layer);
    return true;
}

/// Wave 8.5: Pre-pack first half (cur cols 0..N-1) of 48 core loop_down [64, 2*N] FP32 -> [48*64, N] FP16 (bit-exact).
/// Grid: dim3(48, 64), Block: 256.
__global__ void __launch_bounds__(256) k_pack_core_wdown_cur_f16(
        const float* const* __restrict__ W_down_ptrs,
        __half* __restrict__ out_f16,
        int N) {
    const int m = (int) blockIdx.x;
    const int r = (int) blockIdx.y;
    const int tid = (int) threadIdx.x;
    const float* w_m = W_down_ptrs[m];
    __half* dst = out_f16 + ((size_t) m * 64u + (size_t) r) * (size_t) N;
    const int n4 = N >> 2;
    if (!w_m) {
        uint2* dst2 = reinterpret_cast<uint2*>(dst);
        const uint2 z = make_uint2(0u, 0u);
        for (int i = tid; i < n4; i += blockDim.x) dst2[i] = z;
        for (int c = (n4 << 2) + tid; c < N; c += blockDim.x) dst[c] = __float2half_rn(0.0f);
        return;
    }
    const float* src = w_m + (size_t) r * (size_t) (2 * N);
    const float4* src4 = reinterpret_cast<const float4*>(src);
    __half2* dst2 = reinterpret_cast<__half2*>(dst);
    for (int i = tid; i < n4; i += blockDim.x) {
        const float4 v = __ldg(src4 + i);
        dst2[i * 2 + 0] = __floats2half2_rn(v.x, v.y);
        dst2[i * 2 + 1] = __floats2half2_rn(v.z, v.w);
    }
    for (int c = (n4 << 2) + tid; c < N; c += blockDim.x) {
        dst[c] = __float2half_rn(__ldg(src + c));
    }
}

/// Wave 8.5: Bit-exact FP32 -> FP16 packer for [N, 64] core loop_gain / loop_forget weights.
__global__ void __launch_bounds__(256) k_pack_f32_to_f16_exact(
        const float* __restrict__ src,
        __half* __restrict__ dst,
        int n_elems) {
    const int tid = (int) blockIdx.x * (int) blockDim.x + (int) threadIdx.x;
    const int n4 = n_elems >> 2;
    if (tid < n4) {
        const float4 v = __ldg(reinterpret_cast<const float4*>(src) + tid);
        __half2* d2 = reinterpret_cast<__half2*>(dst) + tid * 2;
        d2[0] = __floats2half2_rn(v.x, v.y);
        d2[1] = __floats2half2_rn(v.z, v.w);
    }
    const int c = (n4 << 2) + tid;
    if (c < n_elems) {
        dst[c] = __float2half_rn(__ldg(src + c));
    }
}

struct PromptSuffixDrafter {
    static constexpr size_t kBuckets = 1u << 17;
    struct Bucket {
        int32_t p[4] = {-1, -1, -1, -1};
    };
    std::vector<Bucket> buckets;
    size_t indexed_len = 0;
    int32_t sig0 = 0, sig_mid = 0, sig_last = 0;

    static inline uint32_t hash4(int32_t a, int32_t b, int32_t c, int32_t d) {
        uint64_t h = (uint64_t) (uint32_t) a * 0x9e3779b97f4a7c15ULL;
        h = (h ^ ((uint64_t) (uint32_t) b * 0xbf58476d1ce4e5b9ULL)) * 0x94d049bb133111ebULL;
        h = (h ^ ((uint64_t) (uint32_t) c * 0x9e3779b97f4a7c15ULL)) * 0xbf58476d1ce4e5b9ULL;
        h ^= (uint64_t) (uint32_t) d * 0x94d049bb133111ebULL;
        h ^= h >> 31;
        return (uint32_t) (h & (kBuckets - 1));
    }

    void reset() {
        buckets.clear();
        buckets.resize(kBuckets);
        indexed_len = 0;
        sig0 = sig_mid = sig_last = 0;
    }

    void sync(const std::vector<int32_t>& toks) {
        const size_t n = toks.size();
        if (n < 4) {
            indexed_len = 0;
            return;
        }
        if (buckets.empty()) buckets.resize(kBuckets);
        size_t start = 3;
        if (indexed_len >= 4 && n >= indexed_len &&
            toks[0] == sig0 && toks[indexed_len >> 1] == sig_mid && toks[indexed_len - 1] == sig_last) {
            start = indexed_len;
        } else {
            std::fill(buckets.begin(), buckets.end(), Bucket{});
        }
        for (size_t i = start; i < n; ++i) {
            Bucket& bk = buckets[hash4(toks[i - 3], toks[i - 2], toks[i - 1], toks[i])];
            bk.p[3] = bk.p[2];
            bk.p[2] = bk.p[1];
            bk.p[1] = bk.p[0];
            bk.p[0] = (int32_t) i;
        }
        indexed_len = n;
        sig0 = toks[0];
        sig_mid = toks[n >> 1];
        sig_last = toks[n - 1];
    }

    bool try_draft(const std::vector<int32_t>& toks, int32_t anchor, int max_k, int min_match,
                   std::vector<int32_t>& out) const {
        const size_t n = toks.size();
        if (buckets.empty() || max_k < 1 || n < 3) return false;
        const int32_t w0 = toks[n - 3], w1 = toks[n - 2], w2 = toks[n - 1], w3 = anchor;
        const Bucket& bk = buckets[hash4(w0, w1, w2, w3)];
        int best_c = -1;
        int best_len = 0;
        for (int s = 0; s < 4; ++s) {
            const int c = bk.p[s];
            if (c < 3 || (size_t) (c + 1) >= n) continue;
            if (toks[(size_t) c] != w3 || toks[(size_t) (c - 1)] != w2 ||
                toks[(size_t) (c - 2)] != w1 || toks[(size_t) (c - 3)] != w0) {
                continue;
            }
            int mlen = 4;
            while (mlen < 16 && (size_t) mlen <= n && c >= mlen &&
                   toks[(size_t) (c - mlen)] == toks[n - (size_t) mlen]) {
                ++mlen;
            }
            if (mlen > best_len) {
                best_len = mlen;
                best_c = c;
                if (best_len >= 12) break;
            }
        }
        if (best_c < 0 || best_len < min_match) return false;
        const int avail = (int) (n - (size_t) (best_c + 1));
        const int len_cap = (best_len >= 8) ? max_k
                            : ((best_len >= 7) ? std::min(max_k, 4) : std::min(max_k, 3));
        const int k = std::min(avail, len_cap);
        if (k <= 0) return false;
        out.resize((size_t) k);
        for (int i = 0; i < k; ++i) out[(size_t) i] = toks[(size_t) (best_c + 1 + i)];
        return true;
    }
};

struct Engine {
    Model* m = nullptr;
    cudaStream_t st = nullptr;
    cudaStream_t aux_st = nullptr;
    cudaEvent_t ev_attn_done = nullptr;
    cudaEvent_t ev_moe_zeroed = nullptr;
    cudaEvent_t ev_pool_ready = nullptr; // Wave 6: orders st pool_alloc before cst H2D (0 host stall)
    cudaEvent_t ev_dspark_acc_src = nullptr;
    cudaEvent_t ev_dspark_acc_done = nullptr;
    bool moe_raw_zero_pending = false;
    cudaMemPool_t phase_pool = nullptr;
    bool use_phase_pool = false;
    strata::prefill::Gemm gemm;

    int ctx = 16384;
    int full_kv_cap = 16384;
    int swa_kv_cap = 1024;
    int prefill_swa_kv_cap = 1024;
    uint8_t* swa_kv_pool = nullptr;
    size_t swa_kv_pool_bytes = 0;
    // Wave 9: Shared-Scratch SWA KV Pool (persistent 40-layer ring fixed at swa_ring_cap;
    // single 1-layer scratch inside ephemeral_pool handles prefill chunks T > ring_extra).
    bool use_shared_swa_scratch = true;
    int swa_pp_cap = 0;
    int8_t* swa_pp_kq = nullptr;
    int8_t* swa_pp_vq = nullptr;
    __half* swa_pp_ks = nullptr;
    __half* swa_pp_vs = nullptr;
    __half* swa_pp_kc = nullptr;
    __half* swa_pp_vc = nullptr;
    int last_pp_log_slots = -1;
    int last_pp_log_cap = -1;
    int tmax = 512;
    int meta_tmax = 512;
    int active_pp_cap = 512;
    int64_t pos = 0;
    strata::inkling2::ResolvedProfile profile{};

    // KV caches & shortconv state (per virtual layer il = 0..47)
    bool kv_q8 = true;
    std::vector<__half*> kc, vc;
    std::vector<int8_t*> kq, vq;
    std::vector<__half*> ks, vs;
    std::vector<int> kv_cap;
    float* conv_hist_pool = nullptr;
    size_t layer_conv_floats = 0;
    std::vector<float*> conv_hist;   // [48][n_sconv_channels * 3]

    // Optional DSpark speculative decoding & verify rollback buffers (allocated only when --draft is active)
    DsparkEngine* dspark = nullptr;
    int verify_max_t = 1;
    int swa_bak_slots = 0;
    SwaLayerPtrs swa_ptrs{};
    uint8_t* swa_bak = nullptr;
    float* d_conv_trace = nullptr;   // [verify_max_t][n_layer][layer_conv_floats]
    bool d_conv_trace_from_pool = false;
    float* d_tgt_feat[4] = {};       // [tmax * N] captured at dspark.target_layers
    float* h_tgt_feat_pin[4] = {};   // non-null when d_tgt_feat is zero-copy mapped pinned RAM (Hybrid VRAM save)
    float* ds_acc_swa0 = nullptr;    // Wave 5: [min(cap_t, n_swa+1) * N] running VRAM acc for DSpark SWA L0
    float* ds_acc_swa2 = nullptr;    // Wave 5: [min(cap_t, n_swa+1) * N] running VRAM acc for DSpark SWA L2
    bool dspark_injected_in_fwd = false;
    double p_draft_ms = 0.0;

    // Activation scratch (all allocated up to tmax rows; phase-disjoint buffers share layer_arena)
    uint8_t* ephemeral_pool = nullptr;
    uint8_t* layer_arena = nullptr;
    size_t layer_arena_bytes = 0;
    float* pp_sp_buf = nullptr;
    size_t pp_sp_bytes = 0;
    uint16_t* ctrl_w16 = nullptr;
    size_t ctrl_w16_bytes = 0;
    float *cur = nullptr, *ffn_inp = nullptr;
    float *in_q = nullptr, *in_k = nullptr, *in_v = nullptr, *in_ffn = nullptr;
    float *msg0 = nullptr, *msg1 = nullptr, *msg2 = nullptr, *msg3 = nullptr;
    float *ctrl2N = nullptr, *ctrl64 = nullptr, *ctrl_g = nullptr, *ctrl_f = nullptr;
    float *ctrl_inv_rms = nullptr;   // [8 * cap_t] per-token inv_rms for msg0..3 & c (fused k_control_down_pp_wmma)
    float *vmsg_w = nullptr;         // [48][cap_t][2] virtual loop-bank softmax weights (D == 2)
    float *vmsg_inv_rms = nullptr;   // [48][cap_t] virtual loop-bank per-token inv_rms(msg)
    float *vmsg_down_pre = nullptr;  // Wave 5: [24][cap_t][64] hoisted half-pass vmsg control-down precomp
    const float** d_vmsg_wdown_ptrs = nullptr; // [48] device table of core loop_down[0..3].d pointers
    uint16_t* d_core_wdown_msg_f16 = nullptr;  // Wave 6: [48 * 64 * N] packed FP16 msg-half of core loop_down
    __half* d_core_wdown_cur_f16 = nullptr;    // Wave 8.5: [48 * 64 * N] packed FP16 cur-half of core loop_down (13.5 MiB)
    __half* d_core_wup_f16 = nullptr;          // Wave 8.5: [2 * 48 * N * 64] packed FP16 core loop_gain+forget (27.0 MiB)
    size_t reg1_bytes_cur = 0;                 // Wave 6: active reg1 size in layer_arena for vmsg cuBLAS scratch
    float* ffn_splitk_scratch = nullptr;       // Wave 8.5: Region-1 scratch strictly disjoint from moe_raw/in_ffn/reg2
    size_t ffn_splitk_scratch_floats = 0;      // Wave 8.5: capacity (in floats) of ffn_splitk_scratch
    bool can_vmsg_all_core = false;
    bool can_ctrl_down_wmma = false;
    float *norm_cur = nullptr;       // [tmax * N]
    float *prelude_cur_buf = nullptr;
    float *norm_blk_buf[9] = {};     // cached RMSNorm of pushed block residuals (5 ping-pong slots for standard Inkling-2)
    int norm_blk_cnt = 9;

    float *Q = nullptr, *K = nullptr, *V = nullptr, *R = nullptr;
    float *qgr_t1 = nullptr;         // [2 * q_dim + n_rel * n_head] contiguous T=1 fused output
    float *K_conv = nullptr, *V_conv = nullptr, *dyn_k = nullptr, *dyn_v = nullptr;
    float *sink = nullptr, *rel_tab = nullptr, *att = nullptr, *att_gate = nullptr, *wo = nullptr;
    int rel_tab_cap_t = 0;
    float *attn_split = nullptr;     // [verify_max_t * kAttnSplitMax * n_head * 130] for split GQA-3 decode/verify
    int attn_split_slots = 0;        // total (t, sp) slots in attn_split; every launch MUST satisfy T * sp <= attn_split_slots
    int sm_count = 80;
    int cc_maj = 0;
    int fa2_256_bulk_smem = 87040;
    int fa2_256_tail_smem = 52224;
    int fa2_192_swa_smem = 97280;
    bool can_fa2_192 = false;
    bool can_fa2_192_swa_fast = false;
    bool can_ctrl_wmma = false;
    bool can_ctrl_rms_3 = false;
    bool can_ctrl_rms_1 = false;
    bool can_fa2_256 = false;
    size_t l2_persist_bytes = 0;
    size_t l2_max_window_bytes = 0;
    bool bf16_x_f16 = false;
    bool has_bf16_dense = false;
    bool has_bf16_large_dense = false;
    bool has_quant_gemm_scratch = false;
    uint16_t* dequant_scratch2 = nullptr;
    static constexpr int kDecodePhaseCap = 24;
    bool in_decode_phase = false;
    size_t prefill_ephemeral_bytes = 0;
    size_t decode_ephemeral_bytes = 0;
    int saved_pp_stage_cnt = 0;
    size_t saved_pp_stage_bytes = 0;
    int attn_split_t1_full = 160;
    int attn_split_t1_swa = 20;
    float *router_logits = nullptr, *ewts = nullptr, *eparts = nullptr, *moe_raw = nullptr;
    size_t eparts_bytes = 0;
    int eparts_cap_t = 0;
    bool eparts_force_full = false;
    float *logits = nullptr;
    float* h_tok_buf = nullptr;      // pinned [meta_tmax * N] FP32 for token embedding upload
    std::vector<int32_t> tok_seen, tok_uniq; // Wave 5: O(T) unique-token deduplication scratch
    int32_t* h_tok_idx_pin = nullptr;  // pinned [meta_tmax] unique-row index per token
    uint8_t* h_tok_raw_pin = nullptr;  // pinned [meta_tmax * tok_embd_row_bytes] raw unique rows for GPU dequant
    size_t h_tok_raw_bytes = 0;
    uint8_t* h_is_img_pin = nullptr; // pinned [meta_tmax] flag (1 = vision embedding, skip tok_embd_norm)
    uint8_t* d_is_img = nullptr;     // device [meta_tmax] flag for k_rms_norm_masked
    size_t vram_reserve_bytes = 0;   // --vram-reserve-mib headroom (e.g. for GPU strata-vision)
    bool vram_elastic = false;       // --vram-elastic: dynamic expert cache shrink/grow via VRAM command
    size_t full_cache_bytes = 0;     // full expert cache size (for VRAM command reporting)
    uint8_t* ckpt_pin = nullptr;     // reusable pinned staging buffer for fast SWA/conv checkpoint D2H
    size_t ckpt_pin_cap = 0;
    uint16_t* act_bf16 = nullptr;    // FP16 scratch [max(tmax*2*N, 16*4*N)]
    uint16_t* act_bf16_only = nullptr; // Dedicated BF16 scratch on sm_75+ (!bf16_x_f16) so BF16 projections never clobber FP16 act_bf16
    uint16_t* pre_attn_f16_v = nullptr; // Pre-attn FP16 V activation buffer inside layer_arena Region 2

    void *xq = nullptr, *xq_k = nullptr, *xq_v = nullptr, *escr = nullptr;
    size_t escr_bytes = 0;
    int32_t *ids = nullptr, *grp_start = nullptr, *n_groups = nullptr, *ent_dst = nullptr, *ent_tok = nullptr;
    int32_t* d_grp_part_cnt = nullptr; // Wave 5: [2 * 16 * 128] multi-CTA expert grouping scratch (hist + off)
    int32_t* d_grp_part_off = nullptr;
    uint32_t* d_hits_pool = nullptr;   // Wave 5: contiguous [n_phys * n_expert] GPU hit counters (12 KiB)
    uint32_t* d_router_done = nullptr;
    unsigned long long* grp_ptr = nullptr;
    int32_t *pos0_d = nullptr, *pick_d = nullptr;
    int pos0_sub_cap = 1;
    float* argmax_val = nullptr;
    int32_t* argmax_idx = nullptr;
    // GPU Top-K + penalty scratch (< 128 KiB VRAM, avoids 200K D2H & CPU partial_sort on temp>0 chat!)
    uint32_t* d_topk_hist = nullptr;     // [verify_max_t * 2048]
    int32_t* d_topk_thr = nullptr;       // [verify_max_t]
    int32_t* d_topk_cnt = nullptr;       // [verify_max_t]
    int32_t* d_topk_out_cnt = nullptr;   // [verify_max_t]
    float* d_topk_cand_val = nullptr;    // [verify_max_t * 2048]
    int32_t* d_topk_cand_idx = nullptr;  // [verify_max_t * 2048]
    float* d_topk_out_val = nullptr;     // [verify_max_t * 256]
    int32_t* d_topk_out_idx = nullptr;   // [verify_max_t * 256]
    int32_t* d_pen_toks = nullptr;       // [256]
    int32_t* d_pen_cnts = nullptr;       // [256]
    float* h_topk_val_pin = nullptr;     // pinned [verify_max_t * 256]
    int32_t* h_topk_idx_pin = nullptr;   // pinned [verify_max_t * 256]
    int32_t* h_topk_cnt_pin = nullptr;   // pinned [verify_max_t]
    int32_t* h_pen_toks_pin = nullptr;   // pinned [256]
    int32_t* h_pen_cnts_pin = nullptr;   // pinned [256]
    int32_t* h_slot_pin = nullptr;       // pinned [n_expert] for async d_slot_of upload in rebalance
    int32_t* h_pick_pin = nullptr;       // pinned [verify_max_t] for zero-alloc D2H argmax readback
    int32_t* h_pos_pin = nullptr;        // pinned [1] for pos0_d upload

    // Hybrid CPU pool staging
    int32_t* h_ids = nullptr;        // [tmax * top_k]
    float* h_ewts = nullptr;         // [tmax * top_k] pinned for T==1 CPU pre-combine
    float* h_cpu_sum = nullptr;      // [N] pinned pre-combined CPU miss sum for T==1
    float* h_h = nullptr;            // [cpu_buf_cap * N]
    float* cpu_out = nullptr;        // [cpu_buf_cap * top_k * N]
    int32_t* h_miss = nullptr;       // [cpu_buf_cap * top_k]
    int32_t* d_miss = nullptr;
    float* ecpu = nullptr;           // [cpu_buf_cap * top_k * N]
    uint8_t* acts = nullptr;         // [cpu_buf_cap * kNativeActBytes]
    int cpu_buf_cap = 0;
    cudaEvent_t ev_route = nullptr;
    bool hybrid = false;
    bool knapsack_decode_active = false;
    int rebal_burst = 2;

    double effective_score(const PhysLayer& L, int e) const {
        static const double prior_w = [] {
            const char* v = std::getenv("STRATA_EXPERT_PRIOR_W");
            return (v && v[0]) ? std::atof(v) : 2.5;
        }();
        const double p = (e < (int) L.prior_score.size()) ? L.prior_score[(size_t) e] : 0.0;
        return L.score[(size_t) e] + prior_w * p;
    }
    int pp_chunk_cnt = 0;
    size_t pinned_used = 0;
    size_t max_pin_bytes = (size_t) 7168 << 20;
    int cpu_threads_req = 0;
    static constexpr size_t kH2dStageSlotBytes = (size_t) 8 << 20; // 2 x 8 MiB = 16 MiB reusable H2D staging
    uint8_t* h2d_stage_pin[2] = {nullptr, nullptr};
    cudaEvent_t ev_h2d_stage[2] = {nullptr, nullptr};
    bool h2d_stage_busy[2] = {false, false};
    int h2d_stage_next = 0;

    // Double-buffered PCIe GPU streaming for prefill on hybrid layers
    struct PpRun {
        int16_t e0 = 0;
        int16_t cnt = 0;
        int16_t dst_off = 0;
    };
    struct PpTask {
        int ip = -1;
        int il = -1;
        int n_staged = 0;
        int stage_slot = -1;
        int kicked_exps = 0;
        PpStageMap map{};
        std::vector<PpRun> runs;
    };
    static constexpr int kMaxPpStages = 24;
    cudaStream_t cst = nullptr;
    uint8_t* pp_stage_pool = nullptr;
    uint8_t* pp_stage[kMaxPpStages] = {};
    int pp_stage_cnt = 0;
    size_t pp_stage_bytes = 0;
    uint8_t* transient_pp_pool = nullptr; // Wave 8.5: transient Epilogue pre-buffering stages from phase_pool
    int transient_pp_cnt = 0;
    cudaEvent_t ev_pp_copy[kMaxPpStages] = {};
    cudaEvent_t ev_pp_used[kMaxPpStages] = {};
    // EX-5 Idea A & B: unified permanent + transient Prelude-borrowed staging ring
    uint8_t* pp_buf_ptr[kMaxPpStages] = {};
    size_t pp_buf_cap[kMaxPpStages] = {};
    int pp_buf_owner_ip[kMaxPpStages] = {};
    bool pp_slot_needs_wait[kMaxPpStages] = {};
    std::deque<int> pp_free_slots;
    int pp_active_slots = 0;
    static constexpr int kMaxRebalLayers = 48;
    cudaEvent_t ev_rebal_layer[kMaxRebalLayers] = {};
    bool layer_pending_rebal[kMaxRebalLayers] = {};
    bool donor_clobbered[kMaxRebalLayers] = {};
    static constexpr int kMaxCoreBorrow = 4;
    int core_borrowed_ips[kMaxCoreBorrow] = {};
    size_t core_borrowed_bytes[kMaxCoreBorrow] = {};
    int core_borrowed_cnt = 0;
    std::vector<PpTask> pp_tasks;
    std::vector<int> pp_layer_cnt;
    size_t pp_take_idx = 0;
    size_t pp_kick_idx = 0;
    int pp_stream_t = 16;
    bool pp_streaming = false;

#if defined(STRATA_INKLING2_CPU_POOL)
    std::unique_ptr<strata::kernels::cpu::ExpertPool> pool;
    std::vector<strata::kernels::cpu::ExpertJobMulti> jobs;
    strata::kernels::cpu::ThreadAffinity host_prev_aff{};
    bool host_pinned = false;
    void pin_host_for_pool() {
        if (host_pinned || !pool) return;
        // Opt-in only: pinning the host thread to core 0 shares GPU IRQ/timer interrupts and can propagate to CUDA
        // runtime threads, regressing Ryzen 7 5700X Hybrid decode by ~1.5-2.5%. Enable with STRATA_PIN_HOST=1.
        const char* env_pin = std::getenv("STRATA_PIN_HOST");
        if (!env_pin || std::atoi(env_pin) == 0) return;
        const int hc = strata::kernels::cpu::planned_host_core();
        if (hc >= 0) {
            host_prev_aff = strata::kernels::cpu::pin_current_thread(hc);
            host_pinned = host_prev_aff.valid;
        }
    }
#endif

    // Stats
    int64_t hy_gpu = 0, hy_cpu = 0;
    double hy_cpu_ms = 0.0;
    int placed_slots = 0;
    size_t placed_bytes = 0;
    size_t vram_free_after_place = 0;
    int pool_workers = 0;
    bool decode_scores_normalized = false;
    PromptSuffixDrafter pld;

    bool is_core_full_resident() const {
        if (!core_all_gpu || profile.regime == strata::inkling2::VramRegime::FullVram) return false;
        const auto& hp = m->hp;
        const int ne = (int) hp.n_expert;
        for (int ip = (int) hp.core_start; ip < (int) hp.core_end && ip < hp.n_phys; ++ip) {
            if (m->layers[(size_t) ip].n_slots < ne) return false;
        }
        return true;
    }

    bool prof = false;
    bool prof_detail = false;
    double p_embd_ms = 0, p_attn_ms = 0, p_moe_gpu_ms = 0, p_moe_wait_ms = 0, p_moe_cpu_ms = 0;
    double p_moe_up_ms = 0, p_moe_post_ms = 0, p_core_ms = 0, p_head_ms = 0, p_rebal_ms = 0;
    double p_d_mix = 0, p_d_qkv = 0, p_d_attn = 0, p_d_out = 0, p_d_route = 0, p_d_moe_gpu_core = 0, p_d_moe_gpu_nv = 0, p_d_moe_gpu_q5 = 0;
    double p_d_cpu_nv = 0, p_d_cpu_q5 = 0;
    int64_t p_miss_nv = 0, p_miss_q5 = 0, p_hit_nv = 0, p_hit_q5 = 0;
    int p_calls = 0;
    void reset_prof() {
        p_embd_ms = p_attn_ms = p_moe_gpu_ms = p_moe_wait_ms = p_moe_cpu_ms = 0;
        p_moe_up_ms = p_moe_post_ms = p_core_ms = p_head_ms = p_rebal_ms = 0;
        p_d_mix = p_d_qkv = p_d_attn = p_d_out = p_d_route = 0;
        p_d_moe_gpu_core = p_d_moe_gpu_nv = p_d_moe_gpu_q5 = 0;
        p_d_cpu_nv = p_d_cpu_q5 = 0;
        p_miss_nv = p_miss_q5 = p_hit_nv = p_hit_q5 = 0;
        p_calls = 0;
    }
    void print_prof(const char* tag) const {
        const double outer_moe = p_moe_gpu_ms + p_moe_wait_ms + p_moe_cpu_ms + p_moe_up_ms + p_moe_post_ms;
        const double pure_outer_attn = p_attn_ms;
        const double sum = p_embd_ms + p_core_ms + pure_outer_attn + outer_moe + p_head_ms + p_rebal_ms;
        std::fprintf(stderr,
                     "PROFILE %-4s (%d calls, sum=%.1f ms): embd=%.1f core=%.1f attn(outer)=%.1f "
                     "moe_gpu=%.1f moe_wait=%.1f moe_cpu=%.1f moe_up=%.1f moe_post=%.1f head=%.1f rebal=%.1f\n",
                     tag, p_calls, sum, p_embd_ms, p_core_ms, pure_outer_attn,
                     p_moe_gpu_ms, p_moe_wait_ms, p_moe_cpu_ms, p_moe_up_ms, p_moe_post_ms, p_head_ms, p_rebal_ms);
        if (prof_detail) {
            std::fprintf(stderr,
                         "  DETAIL %-4s: mix_ctrl=%.1f qkv_proj=%.1f kv_attn=%.1f out_conv=%.1f route=%.1f "
                         "moe_gpu(core=%.1f nv=%.1f q5=%.1f) moe_cpu(nv=%.1f/%lldmiss_%lldhit q5=%.1f/%lldmiss_%lldhit)\n",
                         tag, p_d_mix, p_d_qkv, p_d_attn, p_d_out, p_d_route,
                         p_d_moe_gpu_core, p_d_moe_gpu_nv, p_d_moe_gpu_q5,
                         p_d_cpu_nv, (long long) p_miss_nv, (long long) p_hit_nv,
                         p_d_cpu_q5, (long long) p_miss_q5, (long long) p_hit_q5);
        }
    }

    // Cached decode CUDA Graph for the contiguous 36-layer core (il = 6..41) when blk.6..17 are 100% in VRAM,
    // plus per-layer pre-route graphs (attn + mix + ffn_norm + route).
    cudaGraphExec_t core_graph = nullptr;
    cudaGraphExec_t verify_core_graph[10] = {};
    // Full 48-layer + swa_bak + head + argmax/softcap graphs for Full-VRAM (T=1..9, capture_dspark, want_softcap, want_argmax)
    cudaGraphExec_t full_decode_graph[10][2][2][2] = {};
    std::vector<cudaGraphExec_t> layer_pre_graph;
    std::vector<cudaGraphExec_t> verify_pre_graph[10];
    std::vector<cudaGraphExec_t> deferred_graphs_destroy;
    uint32_t* h_hits_pin = nullptr;
    bool core_all_gpu = false;
    bool graph_oom = false;

    // Prefix cache checkpoint (stores conv_hist + LIVE circular SWA KV window <= n_swa in host RAM; full-attn KV stays append-only on GPU)
    struct StateCheckpoint {
        std::vector<int32_t> tokens;
        std::vector<uint64_t> img_sigs;
        std::unique_ptr<uint8_t[]> data;
        size_t data_bytes = 0;
        std::unique_ptr<uint8_t[]> dspark_data;
        size_t dspark_bytes = 0;
        int swa_len = 0;
        int saved_pos = 0;
    };
    std::thread ckpt_bg_thread;

    void flush_ckpt_bg() {
        if (ckpt_bg_thread.joinable()) ckpt_bg_thread.join();
    }

    void prealloc_ckpt_pin() {
        const auto& hp = m->hp;
        const size_t conv_bytes = (size_t) hp.n_layer * layer_conv_floats * sizeof(float);
        size_t total = conv_bytes;
        for (int il = 0; il < hp.n_layer; ++il) {
            if (!hp.is_swa(il)) continue;
            const size_t n = (size_t) hp.n_swa;
            if (kv_q8) total += n * (size_t) hp.kv_dim * 2 + n * (size_t) hp.n_head_kv * sizeof(__half) * 2;
            else total += n * (size_t) hp.kv_dim * sizeof(__half) * 2;
        }
        const size_t ds_bytes = dspark ? dspark->checkpoint_bytes() : 0;
        const size_t stage_bytes = total + ds_bytes;
        if (stage_bytes > ckpt_pin_cap) {
            flush_ckpt_bg();
            if (ckpt_pin) { cudaFreeHost(ckpt_pin); ckpt_pin = nullptr; ckpt_pin_cap = 0; }
            if (cudaMallocHost(&ckpt_pin, stage_bytes) == cudaSuccess) ckpt_pin_cap = stage_bytes;
            else (void) cudaGetLastError();
        }
    }

    // Async checkpoint save: D2H into pre-allocated pinned staging (~3.6 ms), then background CPU copy into
    // uninitialized heap buffer while GPU immediately resumes the next prefill chunk / decode token!
    void save_checkpoint_into(StateCheckpoint& cp, const std::vector<int32_t>& tokens,
                              const std::vector<uint64_t>& img_sigs = {}) {
        flush_ckpt_bg();
        const auto& hp = m->hp;
        const int cur_pos = (int) tokens.size();
        const int swa_len = std::min<int>(cur_pos, (int) hp.n_swa);
        const size_t conv_bytes = (size_t) hp.n_layer * layer_conv_floats * sizeof(float);
        size_t total = conv_bytes;
        for (int il = 0; il < hp.n_layer; ++il) {
            if (!hp.is_swa(il)) continue;
            const size_t n = (size_t) swa_len;
            if (kv_q8) total += n * (size_t) hp.kv_dim * 2 + n * (size_t) hp.n_head_kv * sizeof(__half) * 2;
            else total += n * (size_t) hp.kv_dim * sizeof(__half) * 2;
        }
        cp.tokens = tokens;
        cp.img_sigs = img_sigs;
        cp.swa_len = swa_len;
        cp.saved_pos = cur_pos;
        cp.data_bytes = total;
        const size_t ds_bytes = dspark ? dspark->checkpoint_bytes() : 0;
        cp.dspark_bytes = ds_bytes;
        const size_t stage_bytes = total + ds_bytes;
        if (stage_bytes > ckpt_pin_cap) {
            if (ckpt_pin) { cudaFreeHost(ckpt_pin); ckpt_pin = nullptr; ckpt_pin_cap = 0; }
            if (cudaMallocHost(&ckpt_pin, stage_bytes) == cudaSuccess) ckpt_pin_cap = stage_bytes;
            else (void) cudaGetLastError();
        }
        if (!ckpt_pin) {
            cp.data.reset(new uint8_t[total]);
            if (dspark && ds_bytes > 0) cp.dspark_data.reset(new uint8_t[ds_bytes]);
        }
        uint8_t* dst_base = ckpt_pin ? ckpt_pin : cp.data.get();
        size_t off = 0;
        auto pull = [&](const void* src, size_t n) {
            if (n == 0) return;
            CK(cudaMemcpyAsync(dst_base + off, src, n, cudaMemcpyDeviceToHost, st));
            off += n;
        };
        pull(conv_hist_pool, conv_bytes);
        if (swa_len > 0) {
            for (int il = 0; il < hp.n_layer; ++il) {
                if (!hp.is_swa(il)) continue;
                const int cap = kv_cap[(size_t) il];
                const int start = (cur_pos - swa_len) % cap;
                const int n1 = std::min(swa_len, cap - start);
                const int n2 = swa_len - n1;
                if (kv_q8) {
                    const size_t row_q = (size_t) hp.kv_dim;
                    const size_t row_s = (size_t) hp.n_head_kv * sizeof(__half);
                    pull(kq[(size_t) il] + (size_t) start * row_q, (size_t) n1 * row_q);
                    if (n2 > 0) pull(kq[(size_t) il], (size_t) n2 * row_q);
                    pull(vq[(size_t) il] + (size_t) start * row_q, (size_t) n1 * row_q);
                    if (n2 > 0) pull(vq[(size_t) il], (size_t) n2 * row_q);
                    pull(ks[(size_t) il] + (size_t) start * hp.n_head_kv, (size_t) n1 * row_s);
                    if (n2 > 0) pull(ks[(size_t) il], (size_t) n2 * row_s);
                    pull(vs[(size_t) il] + (size_t) start * hp.n_head_kv, (size_t) n1 * row_s);
                    if (n2 > 0) pull(vs[(size_t) il], (size_t) n2 * row_s);
                } else {
                    const size_t row_h = (size_t) hp.kv_dim * sizeof(__half);
                    pull(kc[(size_t) il] + (size_t) start * hp.kv_dim, (size_t) n1 * row_h);
                    if (n2 > 0) pull(kc[(size_t) il], (size_t) n2 * row_h);
                    pull(vc[(size_t) il] + (size_t) start * hp.kv_dim, (size_t) n1 * row_h);
                    if (n2 > 0) pull(vc[(size_t) il], (size_t) n2 * row_h);
                }
            }
        }
        if (dspark) {
            dspark->save_checkpoint(ckpt_pin ? (ckpt_pin + total) : cp.dspark_data.get(), st);
        }
        CK(cudaStreamSynchronize(st));
        if (ckpt_pin) {
            const uint8_t* pin = ckpt_pin;
            const bool has_ds = (dspark != nullptr && ds_bytes > 0);
            ckpt_bg_thread = std::thread([&cp, pin, total, ds_bytes, has_ds]() {
                cp.data.reset(new uint8_t[total]);
                std::memcpy(cp.data.get(), pin, total);
                if (has_ds) {
                    cp.dspark_data.reset(new uint8_t[ds_bytes]);
                    std::memcpy(cp.dspark_data.get(), pin + total, ds_bytes);
                }
            });
        }
    }

    void restore_checkpoint(const StateCheckpoint& cp) {
        flush_ckpt_bg();
        const auto& hp = m->hp;
        const size_t conv_bytes = (size_t) hp.n_layer * layer_conv_floats * sizeof(float);
        const size_t total = cp.data_bytes;
        const size_t ds_bytes = (dspark && cp.dspark_data && cp.dspark_bytes > 0) ? cp.dspark_bytes : 0;
        const size_t stage_bytes = total + ds_bytes;
        if (stage_bytes > ckpt_pin_cap) {
            if (ckpt_pin) { cudaFreeHost(ckpt_pin); ckpt_pin = nullptr; ckpt_pin_cap = 0; }
            if (cudaMallocHost(&ckpt_pin, stage_bytes) == cudaSuccess) ckpt_pin_cap = stage_bytes;
            else (void) cudaGetLastError();
        }
        const uint8_t* src_base = cp.data.get();
        if (ckpt_pin && total > 0) {
            std::memcpy(ckpt_pin, cp.data.get(), total);
            if (ds_bytes > 0) std::memcpy(ckpt_pin + total, cp.dspark_data.get(), ds_bytes);
            src_base = ckpt_pin;
        }
        size_t off = 0;
        auto push = [&](void* dst, size_t n) {
            if (n == 0) return;
            CK(cudaMemcpyAsync(dst, src_base + off, n, cudaMemcpyHostToDevice, st));
            off += n;
        };
        push(conv_hist_pool, conv_bytes);
        const int swa_len = cp.swa_len;
        const int cur_pos = cp.saved_pos;
        if (swa_len > 0) {
            for (int il = 0; il < hp.n_layer; ++il) {
                if (!hp.is_swa(il)) continue;
                const int cap = kv_cap[(size_t) il];
                const int start = (cur_pos - swa_len) % cap;
                const int n1 = std::min(swa_len, cap - start);
                const int n2 = swa_len - n1;
                if (kv_q8) {
                    const size_t row_q = (size_t) hp.kv_dim;
                    const size_t row_s = (size_t) hp.n_head_kv * sizeof(__half);
                    push(kq[(size_t) il] + (size_t) start * row_q, (size_t) n1 * row_q);
                    if (n2 > 0) push(kq[(size_t) il], (size_t) n2 * row_q);
                    push(vq[(size_t) il] + (size_t) start * row_q, (size_t) n1 * row_q);
                    if (n2 > 0) push(vq[(size_t) il], (size_t) n2 * row_q);
                    push(ks[(size_t) il] + (size_t) start * hp.n_head_kv, (size_t) n1 * row_s);
                    if (n2 > 0) push(ks[(size_t) il], (size_t) n2 * row_s);
                    push(vs[(size_t) il] + (size_t) start * hp.n_head_kv, (size_t) n1 * row_s);
                    if (n2 > 0) push(vs[(size_t) il], (size_t) n2 * row_s);
                } else {
                    const size_t row_h = (size_t) hp.kv_dim * sizeof(__half);
                    push(kc[(size_t) il] + (size_t) start * hp.kv_dim, (size_t) n1 * row_h);
                    if (n2 > 0) push(kc[(size_t) il], (size_t) n2 * row_h);
                    push(vc[(size_t) il] + (size_t) start * hp.kv_dim, (size_t) n1 * row_h);
                    if (n2 > 0) push(vc[(size_t) il], (size_t) n2 * row_h);
                }
            }
        }
        pos = (int64_t) cp.tokens.size();
        if (dspark && ds_bytes > 0) {
            dspark->restore_checkpoint(ckpt_pin ? (ckpt_pin + total) : cp.dspark_data.get(), (int) cp.tokens.size(), st);
        }
        if (ckpt_pin) CK(cudaStreamSynchronize(st));
    }

    cudaError_t pool_alloc(void** ptr, size_t bytes, bool allow_graph_flush = true) {
        cudaError_t err = (use_phase_pool && phase_pool)
            ? cudaMallocFromPoolAsync(ptr, bytes, phase_pool, st)
            : cudaMalloc(ptr, bytes);
        if (err != cudaSuccess && allow_graph_flush &&
            (!deferred_graphs_destroy.empty() || (dspark && !dspark->deferred_graphs_destroy.empty()))) {
            (void) cudaGetLastError();
            flush_deferred_graphs();
            err = (use_phase_pool && phase_pool)
                ? cudaMallocFromPoolAsync(ptr, bytes, phase_pool, st)
                : cudaMalloc(ptr, bytes);
        }
        return err;
    }

    void pool_free(void* ptr) {
        if (!ptr) return;
        if (use_phase_pool && phase_pool) {
            CK(cudaFreeAsync(ptr, st));
        } else {
            CK(cudaFree(ptr));
        }
    }

    size_t pool_cached_free_bytes() const {
        if (!use_phase_pool || !phase_pool) return 0;
        uint64_t resv = 0, used = 0;
        if (cudaMemPoolGetAttribute(phase_pool, cudaMemPoolAttrReservedMemCurrent, &resv) != cudaSuccess ||
            cudaMemPoolGetAttribute(phase_pool, cudaMemPoolAttrUsedMemCurrent, &used) != cudaSuccess) {
            (void) cudaGetLastError();
            return 0;
        }
        return (resv > used) ? (size_t) (resv - used) : 0;
    }

    void trim_phase_pool(size_t min_bytes = 0) {
        if (use_phase_pool && phase_pool) {
            CK(cudaStreamSynchronize(st));
            CK(cudaMemPoolTrimTo(phase_pool, min_bytes));
        }
    }

    void alloc_conv_trace() {
        if (d_conv_trace || !m || verify_max_t <= 1) return;
        const size_t total_conv = (size_t) m->hp.n_layer * layer_conv_floats;
        const size_t bytes = (size_t) verify_max_t * total_conv * sizeof(float);
        if (use_phase_pool && phase_pool) {
            if (pool_alloc((void**) &d_conv_trace, bytes) == cudaSuccess && d_conv_trace) {
                d_conv_trace_from_pool = true;
                return;
            }
            (void) cudaGetLastError();
            d_conv_trace = nullptr;
        }
        CK(cudaMalloc(&d_conv_trace, bytes));
        d_conv_trace_from_pool = false;
    }

    void free_conv_trace() {
        if (!d_conv_trace) return;
        if (d_conv_trace_from_pool) {
            pool_free(d_conv_trace);
        } else {
            cudaFree(d_conv_trace);
        }
        d_conv_trace = nullptr;
        d_conv_trace_from_pool = false;
    }

    static void free_sparse_hblobs(PhysLayer& L) {
        for (uint8_t* p : L.sparse_hblobs) std::free(p);
        L.sparse_hblobs.clear();
    }

    static uint8_t* get_or_build_expert_hblob(PhysLayer& L, int e, int ne) {
        if (L.hblobs) return L.hblobs + (size_t) e * L.stride;
        if ((int) L.sparse_hblobs.size() < ne) L.sparse_hblobs.assign((size_t) ne, nullptr);
        uint8_t*& ptr = L.sparse_hblobs[(size_t) e];
        if (!ptr) {
            ptr = (uint8_t*) std::malloc(L.stride);
            if (!ptr) {
                std::fprintf(stderr, "strata-inkling2: sparse_hblobs malloc failed for expert %d\n", e);
                std::exit(1);
            }
            build_expert_blob(L, e, ptr);
        }
        return ptr;
    }

    void snapshot_evicted_sparse_hblobs(PhysLayer& L, const std::vector<int>& keep, int ne, bool only_evicted) {
        if (L.hblobs || L.d_blobs == nullptr || L.n_slots <= 0) return;
        std::vector<uint8_t> kept((size_t) ne, 0);
        if (only_evicted) {
            for (int e : keep) kept[(size_t) e] = 1;
        }
        if ((int) L.sparse_hblobs.size() < ne) L.sparse_hblobs.assign((size_t) ne, nullptr);
        struct D2hItem { int e; int s; uint8_t* dst; };
        std::vector<D2hItem> todo;
        todo.reserve((size_t) L.n_slots);
        for (int e = 0; e < ne; ++e) {
            if (only_evicted && kept[(size_t) e]) continue;
            const int s = L.slot_of[(size_t) e];
            if (s < 0 || L.sparse_hblobs[(size_t) e] != nullptr) continue;
            uint8_t* ptr = (uint8_t*) std::malloc(L.stride);
            if (!ptr) {
                std::fprintf(stderr, "strata-inkling2: sparse_hblobs malloc failed during snapshot\n");
                std::exit(1);
            }
            L.sparse_hblobs[(size_t) e] = ptr;
            todo.push_back(D2hItem{e, s, ptr});
        }
        if (todo.empty()) return;
        ensure_h2d_stage_bufs();
        if (h2d_stage_pin[0] && h2d_stage_pin[1] && L.L.bytes <= kH2dStageSlotBytes) {
            for (int bi = 0; bi < 2; ++bi) {
                if (h2d_stage_busy[bi]) {
                    CK(cudaEventSynchronize(ev_h2d_stage[bi]));
                    h2d_stage_busy[bi] = false;
                }
            }
            const int n_todo = (int) todo.size();
            for (int i = 0; i < n_todo; ++i) {
                const int bi = i & 1;
                if (i == 0) {
                    CK(cudaMemcpyAsync(h2d_stage_pin[0], L.d_blobs + (size_t) todo[0].s * L.stride,
                                       L.L.bytes, cudaMemcpyDeviceToHost, st));
                    CK(cudaEventRecord(ev_h2d_stage[0], st));
                }
                if (i + 1 < n_todo) {
                    const int nbi = (i + 1) & 1;
                    CK(cudaMemcpyAsync(h2d_stage_pin[nbi], L.d_blobs + (size_t) todo[(size_t) (i + 1)].s * L.stride,
                                       L.L.bytes, cudaMemcpyDeviceToHost, st));
                    CK(cudaEventRecord(ev_h2d_stage[nbi], st));
                }
                CK(cudaEventSynchronize(ev_h2d_stage[bi]));
                std::memcpy(todo[(size_t) i].dst, h2d_stage_pin[bi], L.L.bytes);
            }
        } else {
            for (const D2hItem& it : todo) {
                CK(cudaMemcpyAsync(it.dst, L.d_blobs + (size_t) it.s * L.stride,
                                   L.L.bytes, cudaMemcpyDeviceToHost, st));
            }
            CK(cudaStreamSynchronize(st));
        }
    }

    static bool layer_allow_pin_hblobs(const PhysLayer& L) {
        if (std::getenv("STRATA_NO_PIN_EXPERTS") != nullptr) return false;
        // Keep BF16 tail layers (gu_type == 30) unpinned; pin Q5_K (gu_type == 13) when within max_pin_bytes unless opted out.
        static const bool no_pin_q5k = (std::getenv("STRATA_NO_PIN_Q5K_HBLOBS") != nullptr);
        if (L.gu_type == 30 || (no_pin_q5k && L.gu_type == 13)) return false;
        return true;
    }

    void ensure_h2d_stage_bufs() {
        if (h2d_stage_pin[0] && h2d_stage_pin[1]) return;
        if (cudaMallocHost(&h2d_stage_pin[0], kH2dStageSlotBytes) == cudaSuccess &&
            cudaMallocHost(&h2d_stage_pin[1], kH2dStageSlotBytes) == cudaSuccess) {
            CK(cudaEventCreateWithFlags(&ev_h2d_stage[0], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&ev_h2d_stage[1], cudaEventDisableTiming));
        } else {
            (void) cudaGetLastError();
            if (h2d_stage_pin[0]) { cudaFreeHost(h2d_stage_pin[0]); h2d_stage_pin[0] = nullptr; }
            if (h2d_stage_pin[1]) { cudaFreeHost(h2d_stage_pin[1]); h2d_stage_pin[1] = nullptr; }
        }
    }

    void copy_h2d_staged(void* dst_dev, const uint8_t* src_host, size_t bytes, bool src_pinned, cudaStream_t stream) {
        if (bytes == 0) return;
        if (src_pinned) {
            CK(cudaMemcpyAsync(dst_dev, src_host, bytes, cudaMemcpyHostToDevice, stream));
            return;
        }
        ensure_h2d_stage_bufs();
        if (!h2d_stage_pin[0] || !h2d_stage_pin[1]) {
            CK(cudaMemcpyAsync(dst_dev, src_host, bytes, cudaMemcpyHostToDevice, stream));
            return;
        }
        size_t off = 0;
        while (off < bytes) {
            const int bi = h2d_stage_next;
            h2d_stage_next ^= 1;
            if (h2d_stage_busy[bi]) {
                CK(cudaEventSynchronize(ev_h2d_stage[bi]));
                h2d_stage_busy[bi] = false;
            }
            const size_t chunk = std::min(bytes - off, kH2dStageSlotBytes);
            std::memcpy(h2d_stage_pin[bi], src_host + off, chunk);
            CK(cudaMemcpyAsync(reinterpret_cast<uint8_t*>(dst_dev) + off, h2d_stage_pin[bi], chunk,
                               cudaMemcpyHostToDevice, stream));
            CK(cudaEventRecord(ev_h2d_stage[bi], stream));
            h2d_stage_busy[bi] = true;
            off += chunk;
        }
    }

    size_t non_expert_pinned_bytes() const {
        if (!m) return 0;
        const auto& hp = m->hp;
        const int64_t N = hp.n_embd;
        size_t b = 0;
        if (dspark && dspark->dm && dspark->dm->decode_w_pin_cuda) b += dspark->dm->decode_w_bytes;
        for (int k = 0; k < 4; ++k) {
            if (h_tgt_feat_pin[k]) b += (size_t) tmax * (size_t) N * sizeof(float);
        }
        if (h_tok_buf) b += (size_t) meta_tmax * (size_t) N * sizeof(float);
        if (h_tok_raw_pin) b += h_tok_raw_bytes;
        if (h_tok_idx_pin) b += (size_t) meta_tmax * sizeof(int32_t);
        if (h_is_img_pin) b += (size_t) meta_tmax * sizeof(uint8_t);
        if (h_slot_pin) b += (size_t) hp.n_phys * (size_t) hp.n_expert * sizeof(int32_t);
        if (h_hits_pin) b += (size_t) hp.n_phys * (size_t) hp.n_expert * sizeof(uint32_t);
        if (h_ids) b += (size_t) meta_tmax * (size_t) hp.n_expert_used * sizeof(int32_t);
        if (h_ewts) b += (size_t) meta_tmax * (size_t) hp.n_expert_used * sizeof(float);
        if (h_cpu_sum) b += (size_t) std::max(verify_max_t, 8) * (size_t) N * sizeof(float);
        if (h_h) b += (size_t) cpu_buf_cap * (size_t) N * sizeof(float);
        if (cpu_out) b += (size_t) cpu_buf_cap * (size_t) hp.n_expert_used * (size_t) N * sizeof(float);
        if (h_miss) b += (size_t) cpu_buf_cap * (size_t) hp.n_expert_used * sizeof(int32_t);
        if (h_topk_val_pin) b += (size_t) verify_max_t * 256 * sizeof(float);
        if (h_topk_idx_pin) b += (size_t) verify_max_t * 256 * sizeof(int32_t);
        if (h_topk_cnt_pin) b += (size_t) verify_max_t * sizeof(int32_t);
        if (h_pick_pin) b += (size_t) verify_max_t * sizeof(int32_t);
        if (h_pos_pin) b += (size_t) pos0_sub_cap * sizeof(int32_t);
        b += (size_t) 2 * kH2dStageSlotBytes;
        if (ckpt_pin_cap > 0) {
            b += ckpt_pin_cap;
        } else {
            size_t ck = (size_t) hp.n_layer * layer_conv_floats * sizeof(float);
            for (int il = 0; il < hp.n_layer; ++il) {
                if (!hp.is_swa(il)) continue;
                const size_t n = (size_t) hp.n_swa;
                ck += kv_q8 ? (n * (size_t) hp.kv_dim * 2 + n * (size_t) hp.n_head_kv * sizeof(__half) * 2)
                            : (n * (size_t) hp.kv_dim * sizeof(__half) * 2);
            }
            if (dspark) ck += dspark->checkpoint_bytes();
            b += ck;
        }
        return b;
    }

    void free_layer_blobs(PhysLayer& L) {
        if (!L.d_blobs) return;
        if (L.d_blobs_from_pool) {
            pool_free(L.d_blobs);
            L.d_blobs_from_pool = false;
        } else {
            auto it = std::find(m->gpu_allocs.begin(), m->gpu_allocs.end(), (void*) L.d_blobs);
            if (it != m->gpu_allocs.end()) m->gpu_allocs.erase(it);
            CK(cudaFree(L.d_blobs));
        }
        L.d_blobs = nullptr;
    }

    void free_pp_stages() {
        if (pp_stage_pool) {
            pool_free(pp_stage_pool);
            pp_stage_pool = nullptr;
        }
        for (int b = 0; b < kMaxPpStages; ++b) pp_stage[b] = nullptr;
        pp_stage_cnt = 0;
        pp_stage_bytes = 0;
    }

    bool alloc_pp_stages(int cnt, size_t slot_bytes) {
        free_pp_stages();
        if (cnt < 2 || slot_bytes == 0) return false;
        cnt = std::min(cnt, kMaxPpStages);
        const size_t stride_b = (slot_bytes + 255u) & ~size_t(255u);
        for (int try_cnt = cnt; try_cnt >= 2; --try_cnt) {
            uint8_t* pool_p = nullptr;
            if (pool_alloc((void**) &pool_p, (size_t) try_cnt * stride_b, /*allow_graph_flush=*/(try_cnt == 2)) == cudaSuccess && pool_p) {
                pp_stage_pool = pool_p;
                pp_stage_cnt = try_cnt;
                pp_stage_bytes = slot_bytes;
                for (int b = 0; b < try_cnt; ++b) pp_stage[b] = pool_p + (size_t) b * stride_b;
                return true;
            }
            (void) cudaGetLastError();
        }
        return false;
    }

    void upload_slot_of(int ip, cudaStream_t stream) {
        PhysLayer& L = m->layers[(size_t) ip];
        if (!L.d_slot_of) return;
        const size_t ne = (size_t) m->hp.n_expert;
        if (h_slot_pin) {
            int32_t* sp = h_slot_pin + (size_t) ip * ne;
            std::memcpy(sp, L.slot_of.data(), ne * sizeof(int32_t));
            CK(cudaMemcpyAsync(L.d_slot_of, sp, ne * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        } else {
            CK(cudaMemcpyAsync(L.d_slot_of, L.slot_of.data(), ne * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        }
    }

    bool will_phase_decode(int limit, bool has_dspark, int next_chunk_t = 0) const {
        static const bool disable_ps = (std::getenv("STRATA_NO_PHASE_SPLIT") != nullptr);
        const bool eff_in_decode = in_decode_phase && (next_chunk_t <= kDecodePhaseCap);
        if (disable_ps || !hybrid || eff_in_decode || tmax <= kDecodePhaseCap || tmax < 1024) return false;
        return limit >= 16 || (has_dspark && dspark && dspark->dm && (!dspark->dm->decode_w_resident || in_decode_phase) && limit >= 2);
    }

    void free_ephemeral_bufs() {
        if (dequant_scratch2) {
            std::string side_err;
            gemm.set_dequant_side(nullptr, side_err);
            dequant_scratch2 = nullptr;
        }
        if (ephemeral_pool) { pool_free(ephemeral_pool); ephemeral_pool = nullptr; }
        for (int b = 0; b < 9; ++b) norm_blk_buf[b] = nullptr;
        prelude_cur_buf = nullptr;
        cur = nullptr;
        ffn_inp = nullptr;
        msg3 = nullptr;
        ds_acc_swa0 = nullptr;
        ds_acc_swa2 = nullptr;
        sink = nullptr;
        router_logits = nullptr;
        ewts = nullptr;
        ctrl_inv_rms = nullptr;
        vmsg_w = nullptr;
        vmsg_inv_rms = nullptr;
        vmsg_down_pre = nullptr;
        act_bf16 = nullptr;
        act_bf16_only = nullptr;
        pre_attn_f16_v = nullptr;
        swa_pp_cap = 0;
        swa_pp_kq = swa_pp_vq = nullptr;
        swa_pp_ks = swa_pp_vs = nullptr;
        swa_pp_kc = swa_pp_vc = nullptr;
        rel_tab_cap_t = 0;
        eparts_cap_t = 0;
        if (layer_arena) { pool_free(layer_arena); layer_arena = nullptr; }
        layer_arena_bytes = 0;
        reg1_bytes_cur = 0;
        ffn_splitk_scratch = nullptr;
        ffn_splitk_scratch_floats = 0;
    }

    struct EphemeralLayout {
        size_t sz_N = 0, sz_msg3 = 0, sz_ds_acc = 0, sz_sink = 0, sz_rl = 0, sz_ew = 0;
        size_t sz_cir = 0, sz_vmsg_w = 0, sz_vmsg_ir = 0, sz_vmsg_pre = 0, sz_act = 0;
        int swa_pp_cap = 0;
        size_t sz_swa_pp_q = 0, sz_swa_pp_s = 0, sz_swa_pp_h = 0, sz_swa_pp_total = 0;
        size_t eph_bytes = 0;
        size_t sz_2N = 0, sz_q = 0, sz_kv = 0, sz_kv_conv = 0, sz_rel = 0, sz_dyn = 0, sz_c64 = 0, ctrl_w16_b = 0;
        int rel_tab_cap = 0;
        size_t sz_reltab = 0, sz_ctrl2N = 0, sz_ctrl_g = 0, sz_ctrl_f = 0, sz_norm_cur = 0;
        bool need_act_bf16_only = false;
        size_t sz_f16_row = 0, sz_pre_attn_f16 = 0;
        size_t reg0_bytes = 0, reg1_bytes = 0, reg2_bytes = 0, reg1_off = 0, reg2_off = 0;
        int eparts_cap = 0;
        size_t eparts_b = 0, escr_b = 0, moe_raw_off = 0;
        bool in_ffn_below_reg2 = false;
        size_t arena_bytes = 0;
    };

    EphemeralLayout compute_ephemeral_layout(int cap_t) const {
        EphemeralLayout L{};
        const auto& hp = m->hp;
        const int N = (int) hp.n_embd;
        const int64_t max_ent = (int64_t) cap_t * hp.n_expert_used;
        auto a256 = [](size_t n) -> size_t { return (n + 255) & ~(size_t) 255; };

        static const bool no_vmsg = (std::getenv("STRATA_NO_VMSG") != nullptr);
        static const bool no_vmsg_hoist = (std::getenv("STRATA_NO_VMSG_HOIST") != nullptr);
        const bool has_dspark_cfg = (dspark != nullptr || verify_max_t > 1);
        const bool shrink_msg3 = (!has_dspark_cfg && !no_vmsg && cc_maj >= 8 && N == 2304 && can_vmsg_all_core);

        L.sz_N = a256((size_t) cap_t * (size_t) N * sizeof(float));
        L.sz_msg3 = shrink_msg3 ? a256((size_t) std::min(cap_t, 32) * (size_t) N * sizeof(float)) : L.sz_N;
        L.sz_ds_acc = (has_dspark_cfg && cap_t > 8)
            ? a256((size_t) std::min(cap_t, (int) hp.n_swa + 1) * (size_t) N * sizeof(float))
            : 0;
        L.sz_sink = a256((size_t) cap_t * (size_t) hp.n_head * sizeof(float));
        L.sz_rl = a256((size_t) cap_t * (size_t) hp.n_expert * sizeof(float));
        L.sz_ew = a256((size_t) max_ent * sizeof(float));
        L.sz_cir = a256((size_t) cap_t * 8 * sizeof(float));
        L.sz_vmsg_w = a256((size_t) 48 * (size_t) cap_t * 2 * sizeof(float));
        L.sz_vmsg_ir = a256((size_t) 48 * (size_t) cap_t * sizeof(float));
        L.sz_vmsg_pre = (!no_vmsg && !no_vmsg_hoist && can_vmsg_all_core && cap_t >= 32)
            ? a256((size_t) 24 * (size_t) cap_t * 64 * sizeof(float))
            : 0;
        const size_t act_elems = std::max<size_t>((size_t) cap_t * 2 * (size_t) N, (size_t) 16 * 4 * (size_t) N);
        L.sz_act = a256(act_elems * sizeof(uint16_t));
        const int ring_extra = std::max(kDecodePhaseCap, verify_max_t);
        if (use_shared_swa_scratch && swa_ptrs.n_swa_layers > 0 && cap_t > ring_extra) {
            L.swa_pp_cap = (int) hp.n_swa + cap_t;
            if (kv_q8) {
                L.sz_swa_pp_q = a256((size_t) L.swa_pp_cap * (size_t) hp.kv_dim * sizeof(int8_t));
                L.sz_swa_pp_s = a256((size_t) L.swa_pp_cap * (size_t) hp.n_head_kv * sizeof(__half));
                L.sz_swa_pp_total = 2 * L.sz_swa_pp_q + 2 * L.sz_swa_pp_s;
            } else {
                L.sz_swa_pp_h = a256((size_t) L.swa_pp_cap * (size_t) hp.kv_dim * sizeof(__half));
                L.sz_swa_pp_total = 2 * L.sz_swa_pp_h;
            }
        }
        L.eph_bytes = (size_t) (norm_blk_cnt + 3) * L.sz_N + L.sz_msg3 + 2 * L.sz_ds_acc
                    + L.sz_sink + L.sz_rl + L.sz_ew + L.sz_cir + L.sz_vmsg_w + L.sz_vmsg_ir + L.sz_vmsg_pre
                    + L.sz_act + L.sz_swa_pp_total;

        L.sz_2N = a256((size_t) cap_t * 2 * (size_t) N * sizeof(float));
        L.sz_q = a256((size_t) cap_t * (size_t) hp.q_dim * sizeof(float));
        L.sz_kv = a256((size_t) cap_t * (size_t) hp.kv_dim * sizeof(float));
        // Wave 14: On kv_q8 + head_dim == 128 with verify_max_t < 32, T < 32 always uses fused_pre_attn_t1 / fused_kv_q8
        // (both handle save_conv_trace) and T >= 32 always uses fused_kv_q8_pp, so K_conv/V_conv are never read or written.
        // Reclaiming 2 * sz_kv saves 64 MiB @ 8K / 96 MiB @ 12K while keeping moe_raw ([216,288) @ 8K) < reg1_off (294 MiB)
        // and preserving pp_sp_buf >= 3 * per_sp_bytes (294 >= 292.5 MiB @ 8K, 441 >= 438.75 MiB @ 12K).
        const bool drop_kv_conv = (kv_q8 && hp.head_dim == 128 && verify_max_t < 32);
        L.sz_kv_conv = drop_kv_conv ? 0 : L.sz_kv;
        L.sz_rel = a256((size_t) cap_t * (size_t) (hp.n_rel * hp.n_head) * sizeof(float));
        L.sz_dyn = a256((size_t) cap_t * (size_t) (hp.n_conv * hp.n_head_kv) * sizeof(float));
        L.sz_c64 = a256((size_t) cap_t * (size_t) hp.loop_gate_rank * sizeof(float));
        L.ctrl_w16_b = a256((size_t) (hp.loop_gate_rank * 2 * N + 2 * N * hp.loop_gate_rank) * sizeof(uint16_t));

        const bool compact_reltab = (kv_q8 && can_fa2_192 && can_fa2_192_swa_fast && can_fa2_256 &&
                                     hp.n_head == 3 * hp.n_head_kv && hp.head_dim == 128 && hp.n_rel == 16);
        L.rel_tab_cap = compact_reltab ? std::min(cap_t, 256) : cap_t;
        L.sz_reltab = a256((size_t) L.rel_tab_cap * (size_t) hp.n_head * (size_t) hp.n_rel_ext * sizeof(float));

        bool all_ctrl_f32 = true;
        for (const auto& ly : m->layers) {
            for (int mod = 0; mod < 4; ++mod) {
                if ((ly.loop_down[mod].d && ly.loop_down[mod].type != 0) ||
                    (ly.loop_gain[mod].d && ly.loop_gain[mod].type != 0) ||
                    (ly.loop_forget[mod].d && ly.loop_forget[mod].type != 0)) {
                    all_ctrl_f32 = false;
                    break;
                }
            }
            if (!all_ctrl_f32) break;
        }
        const bool compact_ctrl = (can_ctrl_wmma && hp.loop_gate_rank == 64 && (N % 64) == 0 && all_ctrl_f32);
        const bool skip_norm_cur = (N == 2304 && (hp.n_phys / std::max<int64_t>(1, hp.attn_res_block_size) + 2) <= 8);
        const int ctrl_small_t = std::min(cap_t, 32);
        L.sz_ctrl2N = compact_ctrl ? a256((size_t) ctrl_small_t * 2 * (size_t) N * sizeof(float)) : L.sz_2N;
        L.sz_ctrl_g = compact_ctrl ? a256((size_t) ctrl_small_t * (size_t) N * sizeof(float)) : L.sz_N;
        L.sz_ctrl_f = compact_ctrl ? std::max(3 * L.sz_c64, L.sz_ctrl_g) : std::max(3 * L.sz_c64, L.sz_N);
        L.sz_norm_cur = skip_norm_cur ? 0 : L.sz_N;
        L.need_act_bf16_only = (!bf16_x_f16 && has_bf16_large_dense);
        L.sz_f16_row = a256((size_t) cap_t * (size_t) N * sizeof(uint16_t));
        L.sz_pre_attn_f16 = L.need_act_bf16_only
            ? std::max(2 * L.sz_f16_row, a256((size_t) cap_t * (size_t) hp.q_dim * sizeof(uint16_t)))
            : L.sz_f16_row;

        const size_t reg0_qkv = 3 * L.sz_N + 2 * L.sz_kv + 2 * L.sz_kv_conv + L.sz_rel + 2 * L.sz_dyn;
        const size_t per_sp_bytes = (size_t) cap_t * (size_t) hp.n_head * 130 * sizeof(float);
        const size_t attn_split_bytes = (size_t) attn_split_slots * (size_t) hp.n_head * 130 * sizeof(float);
        const int eff_want_sp = (cap_t >= 8192 && profile.regime != strata::inkling2::VramRegime::FullVram)
            ? std::max(2, profile.arch.hybrid_want_sp_8192)
            : std::max(2, profile.want_sp);
        const size_t want_sp = (size_t) eff_want_sp;
        L.reg0_bytes = a256(std::max({reg0_qkv, want_sp * per_sp_bytes, attn_split_bytes}));
        // Wave 12B: On kv_q8 + FA2-192 + all-F16 attn_output, T >= 128 writes act_bf16 (never FP32 att),
        // so reg1 only needs max(3*sz_N, 2*sz_q + 128*q_dim*4B) -> saves 36/72/108 MiB at 4K/8K/12K with 0 aliasing!
        bool all_attn_out_f16_ok = (hp.q_dim <= 2 * N);
        for (const auto& ly : m->layers) {
            if (ly.attn_output.type == 0 || (ly.attn_output.type == 30 && !bf16_x_f16)) {
                all_attn_out_f16_ok = false;
                break;
            }
        }
        const bool compact_att_reg1 = (kv_q8 && can_fa2_192 && hp.n_head == 3 * hp.n_head_kv &&
                                       hp.head_dim == 128 && all_attn_out_f16_ok);
        const size_t sz_att_compact = a256((size_t) std::min(cap_t, 128) * (size_t) hp.q_dim * sizeof(float));
        L.reg1_bytes = a256(std::max(3 * L.sz_N, compact_att_reg1 ? (2 * L.sz_q + sz_att_compact) : (3 * L.sz_q)));
        // Wave 5: Overlap pre_attn_f16_v with reg2_ctrl_dyn (ctrl2N/ctrl_g/ctrl_f/ctrl64/norm_cur), keeping ctrl_w16 disjoint!
        const size_t reg2_ctrl_dyn = L.sz_ctrl2N + L.sz_ctrl_g + L.sz_ctrl_f + L.sz_c64 + L.sz_norm_cur;
        const size_t reg2_ctrl_core = L.ctrl_w16_b + reg2_ctrl_dyn;
        const size_t reg2_pre_attn = L.ctrl_w16_b + std::max(reg2_ctrl_dyn, L.sz_pre_attn_f16);
        L.reg2_bytes = a256(std::max(reg2_pre_attn, L.sz_reltab));

        bool all_situ_fused = (hp.n_ff % 64 == 0 && hp.n_expert <= 256);
        bool has_bf16_exp = false;
        for (const auto& ly : m->layers) {
            if (ly.gu_type == 30 || ly.d_type == 30) has_bf16_exp = true;
            if (!ly.L.situ || !K::native_expert_wmma_supported(ly.L)) all_situ_fused = false;
        }
        static const bool no_moe_atomic = (std::getenv("STRATA_NO_MOE_ATOMIC") != nullptr);
        static const bool env_stage_zero = [] {
            const char* v = std::getenv("STRATA_PP_STAGE_MB");
            return v != nullptr && std::atoi(v) <= 0;
        }();
        const bool compact_eparts = (!eparts_force_full && !no_moe_atomic && !env_stage_zero &&
                                     cc_maj >= 8 && (N & 3) == 0 && all_situ_fused && !has_bf16_exp &&
                                     pp_stream_t <= 32);
        L.eparts_cap = compact_eparts ? std::min(cap_t, 32) : cap_t;
        L.eparts_b = a256((size_t) L.eparts_cap * (size_t) hp.n_expert_used * (size_t) N * sizeof(float));
        if (all_situ_fused) {
            const size_t escr_wmma = K::native_expert_wmma_fused_scratch_bytes(max_ent, hp.n_expert, hp.n_ff);
            const size_t escr_small = K::native_expert_scratch_bytes(std::min<int64_t>(max_ent, 16 * hp.n_expert_used), hp.n_ff);
            L.escr_b = a256(std::max(escr_wmma, escr_small));
        } else {
            L.escr_b = a256(K::native_expert_scratch_bytes(max_ent, hp.n_ff));
        }
        L.moe_raw_off = std::max(L.eparts_b + L.escr_b, 3 * L.sz_N);
        const size_t overlay_b_bytes = L.moe_raw_off + 2 * L.sz_N;
        L.reg1_off = L.reg0_bytes;
        L.reg2_off = L.reg1_off + L.reg1_bytes;
        L.in_ffn_below_reg2 = (L.moe_raw_off + 2 * L.sz_N <= L.reg2_off);
        const size_t overlay_a_bytes = L.reg0_bytes + L.reg1_bytes + L.reg2_bytes;
        const size_t overlay_a_ctrl_inffn = L.in_ffn_below_reg2
            ? overlay_a_bytes
            : (L.reg0_bytes + L.reg1_bytes + reg2_ctrl_core + L.sz_N);
        const size_t logits_bytes = a256((size_t) verify_max_t * (size_t) hp.vocab_unpadded * sizeof(float));
        const size_t dspark_arena_min = (verify_max_t > 1)
            ? (logits_bytes + (dspark ? dspark->tail_bytes(kDecodePhaseCap) : ((size_t) 12 << 20)))
            : 0;

        L.arena_bytes = std::max({overlay_a_bytes, overlay_b_bytes, overlay_a_ctrl_inffn, logits_bytes, dspark_arena_min});
        return L;
    }

    size_t estimate_ephemeral_total_bytes(int cap_t) const {
        const EphemeralLayout L = compute_ephemeral_layout(cap_t);
        return L.eph_bytes + L.arena_bytes;
    }

    size_t alloc_ephemeral_bufs(int cap_t) {
        const EphemeralLayout L = compute_ephemeral_layout(cap_t);
        size_t total_b = 0;

        // EX-5 Idea C + Wave 5: Single-slab ephemeral_pool via stream-ordered phase_pool
        CK(pool_alloc((void**) &ephemeral_pool, L.eph_bytes));
        total_b += L.eph_bytes;
        size_t eo = 0;
        auto eptr = [&](size_t sz) -> float* {
            float* p = reinterpret_cast<float*>(ephemeral_pool + eo);
            eo += sz;
            return p;
        };
        for (int b = 0; b < norm_blk_cnt; ++b) norm_blk_buf[b] = eptr(L.sz_N);
        prelude_cur_buf = eptr(L.sz_N);
        cur             = eptr(L.sz_N);
        ffn_inp         = eptr(L.sz_N);
        msg3            = eptr(L.sz_msg3);
        ds_acc_swa0     = (L.sz_ds_acc > 0) ? eptr(L.sz_ds_acc) : nullptr;
        ds_acc_swa2     = (L.sz_ds_acc > 0) ? eptr(L.sz_ds_acc) : nullptr;
        sink            = eptr(L.sz_sink);
        router_logits   = eptr(L.sz_rl);
        ewts            = eptr(L.sz_ew);
        ctrl_inv_rms    = eptr(L.sz_cir);
        vmsg_w          = eptr(L.sz_vmsg_w);
        vmsg_inv_rms    = eptr(L.sz_vmsg_ir);
        vmsg_down_pre   = (L.sz_vmsg_pre > 0) ? eptr(L.sz_vmsg_pre) : nullptr;
        act_bf16        = reinterpret_cast<uint16_t*>(ephemeral_pool + eo);
        eo += L.sz_act;
        swa_pp_cap = L.swa_pp_cap;
        swa_pp_kq = swa_pp_vq = nullptr;
        swa_pp_ks = swa_pp_vs = nullptr;
        swa_pp_kc = swa_pp_vc = nullptr;
        if (L.sz_swa_pp_total > 0) {
            if (kv_q8) {
                swa_pp_kq = reinterpret_cast<int8_t*>(ephemeral_pool + eo); eo += L.sz_swa_pp_q;
                swa_pp_vq = reinterpret_cast<int8_t*>(ephemeral_pool + eo); eo += L.sz_swa_pp_q;
                swa_pp_ks = reinterpret_cast<__half*>(ephemeral_pool + eo); eo += L.sz_swa_pp_s;
                swa_pp_vs = reinterpret_cast<__half*>(ephemeral_pool + eo); eo += L.sz_swa_pp_s;
            } else {
                swa_pp_kc = reinterpret_cast<__half*>(ephemeral_pool + eo); eo += L.sz_swa_pp_h;
                swa_pp_vc = reinterpret_cast<__half*>(ephemeral_pool + eo); eo += L.sz_swa_pp_h;
            }
        }

        ctrl_w16_bytes = L.ctrl_w16_b;
        rel_tab_cap_t = L.rel_tab_cap;
        eparts_cap_t = L.eparts_cap;
        eparts_bytes = L.eparts_b;
        escr_bytes = L.escr_b;
        layer_arena_bytes = L.arena_bytes;
        CK(pool_alloc((void**) &layer_arena, layer_arena_bytes));
        total_b += layer_arena_bytes;
        auto aptr = [&](size_t off) -> float* { return reinterpret_cast<float*>(layer_arena + off); };

        size_t o = 0;
        in_q = aptr(o); o += L.sz_N;
        in_k = aptr(o); o += L.sz_N;
        in_v = aptr(o); o += L.sz_N;
        K = aptr(o); o += L.sz_kv;
        V = aptr(o); o += L.sz_kv;
        R = aptr(o); o += L.sz_rel;
        K_conv = (L.sz_kv_conv > 0) ? aptr(o) : nullptr; o += L.sz_kv_conv;
        V_conv = (L.sz_kv_conv > 0) ? aptr(o) : nullptr; o += L.sz_kv_conv;
        dyn_k = aptr(o); o += L.sz_dyn;
        dyn_v = aptr(o); o += L.sz_dyn;
        pp_sp_buf = aptr(0);
        pp_sp_bytes = L.reg0_bytes;
        attn_split = aptr(0);
        wo = aptr(0);

        msg0 = aptr(L.reg1_off + 0 * L.sz_N);
        msg1 = aptr(L.reg1_off + 1 * L.sz_N);
        msg2 = aptr(L.reg1_off + 2 * L.sz_N);
        Q = aptr(L.reg1_off + 0 * L.sz_q);
        att_gate = aptr(L.reg1_off + 1 * L.sz_q);
        att = aptr(L.reg1_off + 2 * L.sz_q);

        o = L.reg2_off;
        ctrl_w16 = reinterpret_cast<uint16_t*>(layer_arena + o); o += L.ctrl_w16_b;
        const size_t reg2_dyn_off = o;
        ctrl2N = aptr(o); o += L.sz_ctrl2N;
        ctrl_g = aptr(o); o += L.sz_ctrl_g;
        ctrl_f = aptr(o); o += L.sz_ctrl_f;
        ctrl64 = aptr(o); o += L.sz_c64;
        norm_cur = (L.sz_norm_cur > 0) ? aptr(o) : nullptr; o += L.sz_norm_cur;
        uint16_t* pre_attn_f16_base = reinterpret_cast<uint16_t*>(layer_arena + reg2_dyn_off);
        act_bf16_only = L.need_act_bf16_only ? pre_attn_f16_base : nullptr;
        pre_attn_f16_v = L.need_act_bf16_only
            ? reinterpret_cast<uint16_t*>(layer_arena + reg2_dyn_off + L.sz_f16_row)
            : pre_attn_f16_base;
        rel_tab = aptr(L.reg2_off);

        // EX-5: Optional 16 MiB idle slice of layer_arena (STRATA_DEQUANT_SIDE=1, default OFF) strictly past reg2_bytes.
        const size_t dequant_side_need = (size_t) (8 * 1024 * 1024) * sizeof(uint16_t);
        const size_t side_off = L.reg2_off + L.reg2_bytes;
        dequant_scratch2 = (has_quant_gemm_scratch && cap_t >= 256 &&
                            layer_arena_bytes >= side_off + dequant_side_need)
                           ? reinterpret_cast<uint16_t*>(layer_arena + side_off)
                           : nullptr;

        eparts = aptr(0);
        escr = layer_arena + eparts_bytes;
        moe_raw = aptr(L.moe_raw_off);
        in_ffn = L.in_ffn_below_reg2 ? aptr(L.moe_raw_off + L.sz_N) : aptr(layer_arena_bytes - L.sz_N);
        logits = aptr(0);
        reg1_bytes_cur = L.reg1_bytes;
        // Wave 8.5: Compute Region-1 scratch strictly disjoint from moe_raw, in_ffn, and reg2
        // (at T=8192, in_ffn extends past reg1_off into Q; starting past max(moe_raw_end, in_ffn_end) guarantees zero aliasing!).
        {
            size_t ffn_scr_lo = L.reg1_off;
            const size_t ffn_scr_hi = L.reg1_off + L.reg1_bytes;
            if (L.moe_raw_off + L.sz_N > ffn_scr_lo && L.moe_raw_off < ffn_scr_hi) {
                ffn_scr_lo = std::max(ffn_scr_lo, L.moe_raw_off + L.sz_N);
            }
            const size_t in_ffn_off = L.in_ffn_below_reg2 ? (L.moe_raw_off + L.sz_N) : (L.arena_bytes - L.sz_N);
            if (in_ffn_off + L.sz_N > ffn_scr_lo && in_ffn_off < ffn_scr_hi) {
                ffn_scr_lo = std::max(ffn_scr_lo, in_ffn_off + L.sz_N);
            }
            ffn_scr_lo = (ffn_scr_lo + 255u) & ~size_t(255u);
            ffn_splitk_scratch = (ffn_scr_hi > ffn_scr_lo) ? aptr(ffn_scr_lo) : nullptr;
            ffn_splitk_scratch_floats = (ffn_scr_hi > ffn_scr_lo) ? ((ffn_scr_hi - ffn_scr_lo) / sizeof(float)) : 0;
        }
        return total_b;
    }

    int swa_ring_cap() const {
        return (int) m->hp.n_swa + std::max(kDecodePhaseCap, verify_max_t);
    }

    size_t swa_kv_pool_size_for(int cap) const {
        const auto& hp = m->hp;
        auto a256 = [](size_t n) -> size_t { return (n + 255u) & ~size_t(255u); };
        int n_swa_l = 0;
        for (int il = 0; il < hp.n_layer; ++il) {
            if (hp.is_swa(il)) ++n_swa_l;
        }
        if (n_swa_l == 0) return 0;
        const size_t per_layer = kv_q8
            ? (2 * a256((size_t) cap * hp.kv_dim * sizeof(int8_t)) +
               2 * a256((size_t) cap * hp.n_head_kv * sizeof(__half)))
            : (2 * a256((size_t) cap * hp.kv_dim * sizeof(__half)));
        return (size_t) n_swa_l * per_layer;
    }

    void bind_swa_kv_pool(uint8_t* pool, int cap, SwaLayerPtrs& out_ptrs, bool update_engine_vecs) {
        const auto& hp = m->hp;
        auto a256 = [](size_t n) -> size_t { return (n + 255u) & ~size_t(255u); };
        out_ptrs = SwaLayerPtrs{};
        size_t off = 0;
        const size_t q_bytes = a256((size_t) cap * hp.kv_dim * sizeof(int8_t));
        const size_t s_bytes = a256((size_t) cap * hp.n_head_kv * sizeof(__half));
        const size_t h_bytes = a256((size_t) cap * hp.kv_dim * sizeof(__half));
        for (int il = 0; il < hp.n_layer; ++il) {
            if (!hp.is_swa(il)) continue;
            const int s = out_ptrs.n_swa_layers++;
            if (kv_q8) {
                int8_t* pkq = reinterpret_cast<int8_t*>(pool + off); off += q_bytes;
                int8_t* pvq = reinterpret_cast<int8_t*>(pool + off); off += q_bytes;
                __half* pks = reinterpret_cast<__half*>(pool + off); off += s_bytes;
                __half* pvs = reinterpret_cast<__half*>(pool + off); off += s_bytes;
                if (s < 48) { out_ptrs.kq[s] = pkq; out_ptrs.vq[s] = pvq; out_ptrs.ks[s] = pks; out_ptrs.vs[s] = pvs; }
                if (update_engine_vecs) {
                    kq[(size_t) il] = pkq; vq[(size_t) il] = pvq;
                    ks[(size_t) il] = pks; vs[(size_t) il] = pvs;
                    kv_cap[(size_t) il] = cap;
                }
            } else {
                __half* pkc = reinterpret_cast<__half*>(pool + off); off += h_bytes;
                __half* pvc = reinterpret_cast<__half*>(pool + off); off += h_bytes;
                if (s < 48) { out_ptrs.kc[s] = pkc; out_ptrs.vc[s] = pvc; }
                if (update_engine_vecs) {
                    kc[(size_t) il] = pkc; vc[(size_t) il] = pvc;
                    kv_cap[(size_t) il] = cap;
                }
            }
        }
    }

    void resize_swa_kv_pool(int new_cap) {
        if (new_cap == swa_kv_cap || swa_ptrs.n_swa_layers == 0) return;
        const auto t0 = Clock::now();
        const auto& hp = m->hp;
        const size_t new_bytes = swa_kv_pool_size_for(new_cap);
        const int n_live = (int) std::min<int64_t>(pos, hp.n_swa);
        const int old_cap = swa_kv_cap;
        const size_t old_bytes = swa_kv_pool_bytes;
        uint8_t* new_pool = nullptr;
        if (n_live == 0) {
            if (swa_kv_pool) { pool_free(swa_kv_pool); swa_kv_pool = nullptr; }
            CK(pool_alloc((void**) &new_pool, new_bytes));
        } else {
            CK(pool_alloc((void**) &new_pool, new_bytes));
            SwaLayerPtrs new_ptrs{};
            bind_swa_kv_pool(new_pool, new_cap, new_ptrs, false);
            const int64_t pos0 = pos - n_live;
            if (kv_q8) {
                k_swa_ring_remap_q8<<<dim3((unsigned) n_live, (unsigned) swa_ptrs.n_swa_layers), 256, 0, st>>>(
                    swa_ptrs, new_ptrs, (int) hp.kv_dim, (int) hp.n_head_kv, swa_kv_cap, new_cap, pos0, n_live);
            } else {
                k_swa_ring_remap_f16<<<dim3((unsigned) n_live, (unsigned) swa_ptrs.n_swa_layers), 256, 0, st>>>(
                    swa_ptrs, new_ptrs, (int) hp.kv_dim, swa_kv_cap, new_cap, pos0, n_live);
            }
            if (!use_phase_pool) CK(cudaStreamSynchronize(st));
            if (swa_kv_pool) pool_free(swa_kv_pool);
        }
        swa_kv_pool = new_pool;
        swa_kv_pool_bytes = new_bytes;
        swa_kv_cap = new_cap;
        bind_swa_kv_pool(swa_kv_pool, swa_kv_cap, swa_ptrs, true);
        std::fprintf(stderr, "strata-inkling2: SWA KV pool %d -> %d (%.1f -> %.1f MiB, live=%d) in %.2f ms\n",
                     old_cap, new_cap, (double) old_bytes / (1 << 20), (double) new_bytes / (1 << 20),
                     n_live, ms_since(t0));
    }

    ~Engine() {
        if (aux_st) cudaStreamSynchronize(aux_st);
        if (cst) cudaStreamSynchronize(cst);
        if (st) cudaStreamSynchronize(st);
        flush_ckpt_bg();
        destroy_graphs();
        if (h_hits_pin) cudaFreeHost(h_hits_pin);
        for (int il = 0; m && il < m->hp.n_layer; ++il) {
            if (m->hp.is_swa(il)) continue;
            if ((size_t) il < kc.size() && kc[(size_t) il]) cudaFree(kc[(size_t) il]);
            if ((size_t) il < vc.size() && vc[(size_t) il]) cudaFree(vc[(size_t) il]);
            if ((size_t) il < kq.size() && kq[(size_t) il]) cudaFree(kq[(size_t) il]);
            if ((size_t) il < vq.size() && vq[(size_t) il]) cudaFree(vq[(size_t) il]);
            if ((size_t) il < ks.size() && ks[(size_t) il]) cudaFree(ks[(size_t) il]);
            if ((size_t) il < vs.size() && vs[(size_t) il]) cudaFree(vs[(size_t) il]);
        }
        if (swa_kv_pool) { pool_free(swa_kv_pool); swa_kv_pool = nullptr; }
        cudaFree(conv_hist_pool);
        cudaFree(swa_bak);
        free_conv_trace();
        for (int k = 0; k < 4; ++k) {
            if (h_tgt_feat_pin[k]) cudaFreeHost(h_tgt_feat_pin[k]);
            else cudaFree(d_tgt_feat[k]);
        }
        free_ephemeral_bufs();
        cudaFree(qgr_t1);
        cudaFree(xq); cudaFree(xq_k); cudaFree(xq_v);
        cudaFree(ids); cudaFree(grp_start); cudaFree(n_groups); cudaFree(ent_dst); cudaFree(ent_tok); cudaFree(grp_ptr);
        cudaFree(d_grp_part_cnt);
        cudaFree(d_hits_pool);
        cudaFree(d_vmsg_wdown_ptrs);
        cudaFree(d_core_wdown_msg_f16);
        cudaFree(d_core_wdown_cur_f16);
        cudaFree(d_core_wup_f16);
        cudaFree(d_router_done);
        cudaFree(pos0_d); cudaFree(pick_d);
        cudaFree(argmax_val); cudaFree(argmax_idx);
        cudaFree(d_topk_hist); cudaFree(d_topk_thr); cudaFree(d_topk_cnt); cudaFree(d_topk_out_cnt);
        cudaFree(d_topk_cand_val); cudaFree(d_topk_cand_idx);
        cudaFree(d_topk_out_val); cudaFree(d_topk_out_idx);
        cudaFree(d_pen_toks); cudaFree(d_pen_cnts);
        cudaFree(d_miss); cudaFree(ecpu);
        cudaFreeHost(h_tok_buf);
        if (h_tok_idx_pin) cudaFreeHost(h_tok_idx_pin);
        if (h_tok_raw_pin) cudaFreeHost(h_tok_raw_pin);
        cudaFreeHost(h_is_img_pin);
        cudaFree(d_is_img);
        if (ckpt_pin) cudaFreeHost(ckpt_pin);
        if (h_topk_val_pin) cudaFreeHost(h_topk_val_pin);
        if (h_topk_idx_pin) cudaFreeHost(h_topk_idx_pin);
        if (h_topk_cnt_pin) cudaFreeHost(h_topk_cnt_pin);
        if (h_pen_toks_pin) cudaFreeHost(h_pen_toks_pin);
        if (h_pen_cnts_pin) cudaFreeHost(h_pen_cnts_pin);
        if (h_slot_pin) cudaFreeHost(h_slot_pin);
        if (h_pick_pin) cudaFreeHost(h_pick_pin);
        if (h_pos_pin) cudaFreeHost(h_pos_pin);
        cudaFreeHost(h_ids); cudaFreeHost(h_ewts); cudaFreeHost(h_cpu_sum); cudaFreeHost(h_h); cudaFreeHost(cpu_out); cudaFreeHost(h_miss);
        std::free(acts);
        if (ev_route) cudaEventDestroy(ev_route);
        if (ev_attn_done) cudaEventDestroy(ev_attn_done);
        if (ev_moe_zeroed) cudaEventDestroy(ev_moe_zeroed);
        if (ev_pool_ready) cudaEventDestroy(ev_pool_ready);
        if (ev_dspark_acc_src) cudaEventDestroy(ev_dspark_acc_src);
        if (ev_dspark_acc_done) cudaEventDestroy(ev_dspark_acc_done);
        if (transient_pp_pool) { pool_free(transient_pp_pool); transient_pp_pool = nullptr; }
        transient_pp_cnt = 0;
        free_pp_stages();
        for (int b = 0; b < kMaxPpStages; ++b) {
            if (ev_pp_copy[b]) cudaEventDestroy(ev_pp_copy[b]);
            if (ev_pp_used[b]) cudaEventDestroy(ev_pp_used[b]);
        }
        for (int i = 0; i < kMaxRebalLayers; ++i) {
            if (ev_rebal_layer[i]) cudaEventDestroy(ev_rebal_layer[i]);
        }
        if (m) {
            for (auto& L : m->layers) {
                if (L.d_blobs && L.d_blobs_from_pool) {
                    pool_free(L.d_blobs);
                    L.d_blobs = nullptr;
                    L.d_blobs_from_pool = false;
                    L.n_slots = 0;
                }
            }
        }
        if (dspark && dspark->dm && dspark->dm->decode_w_from_pool) {
            dspark->evict_decode_weights(phase_pool, st);
        }
        if (st) cudaStreamSynchronize(st);
        if (cst) cudaStreamSynchronize(cst);
        for (int bi = 0; bi < 2; ++bi) {
            if (ev_h2d_stage[bi]) cudaEventDestroy(ev_h2d_stage[bi]);
            if (h2d_stage_pin[bi]) cudaFreeHost(h2d_stage_pin[bi]);
        }
        if (phase_pool) {
            cudaMemPoolTrimTo(phase_pool, 0);
            cudaMemPoolDestroy(phase_pool);
            phase_pool = nullptr;
        }
        if (aux_st) cudaStreamDestroy(aux_st);
        if (cst) cudaStreamDestroy(cst);
        if (st) cudaStreamDestroy(st);
#if defined(STRATA_INKLING2_CPU_POOL)
        if (host_pinned) {
            strata::kernels::cpu::restore_thread_affinity(host_prev_aff);
            host_pinned = false;
        }
#endif
    }

    bool init(Model& model, int max_ctx, int max_t, bool use_kv_q8, std::string& err,
              bool enable_gemm_quant = false, int max_verify_t = 1,
              const strata::inkling2::ResolvedProfile* prof_cfg = nullptr) {
        m = &model;
        ctx = max_ctx;
        size_t vfree_init = 0, vtot_init = 0;
        CK(cudaMemGetInfo(&vfree_init, &vtot_init));
        if (prof_cfg) {
            profile = *prof_cfg;
            full_kv_cap = std::clamp(profile.full_kv_init, 1, max_ctx);
        } else {
            full_kv_cap = (vtot_init < (20ULL << 30) && max_ctx > 32768) ? 32768 : max_ctx;
        }
        tmax = std::max(max_t, max_verify_t);
        meta_tmax = std::max(tmax, prof_cfg ? prof_cfg->max_adaptive_chunk : tmax);
        active_pp_cap = tmax;
        verify_max_t = std::max(1, max_verify_t);
        kv_q8 = use_kv_q8;
        const auto& hp = m->hp;
        const int64_t N = hp.n_embd;

        CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        CK(cudaStreamCreateWithFlags(&aux_st, cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&ev_attn_done, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&ev_moe_zeroed, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&ev_pool_ready, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&ev_dspark_acc_src, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&ev_dspark_acc_done, cudaEventDisableTiming));
        int cur_dev = 0, cc_min = 0, max_smem_optin = 49152;
        cc_maj = 0;
        cudaGetDevice(&cur_dev);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, cur_dev);
        cudaDeviceGetAttribute(&cc_maj, cudaDevAttrComputeCapabilityMajor, cur_dev);
        cudaDeviceGetAttribute(&cc_min, cudaDevAttrComputeCapabilityMinor, cur_dev);
        cudaDeviceGetAttribute(&max_smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, cur_dev);
        static const bool enable_l2_persist = [] {
            const char* v = std::getenv("STRATA_L2_PERSIST");
            return v != nullptr && (v[0] == '\0' || std::atoi(v) != 0);
        }();
        l2_persist_bytes = 0;
        l2_max_window_bytes = 0;
        if (enable_l2_persist && cc_maj >= 8) {
            cudaDeviceProp prop{};
            if (cudaGetDeviceProperties(&prop, cur_dev) == cudaSuccess &&
                prop.persistingL2CacheMaxSize > 0 && prop.accessPolicyMaxWindowSize > 0) {
                const size_t want_persist = std::min<size_t>(
                    (size_t) prop.persistingL2CacheMaxSize, (size_t) 32u * 1024u * 1024u);
                if (cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, want_persist) == cudaSuccess) {
                    l2_persist_bytes = want_persist;
                    l2_max_window_bytes = (size_t) prop.accessPolicyMaxWindowSize;
                } else {
                    (void) cudaGetLastError();
                }
            } else {
                (void) cudaGetLastError();
            }
        }
        static const bool no_phase_pool = (std::getenv("STRATA_NO_PHASE_POOL") != nullptr);
        int pool_ok = 0;
        if (!no_phase_pool && cc_maj >= 8 &&
            cudaDeviceGetAttribute(&pool_ok, cudaDevAttrMemoryPoolsSupported, cur_dev) == cudaSuccess && pool_ok) {
            cudaMemPoolProps props{};
            props.allocType = cudaMemAllocationTypePinned;
            props.handleTypes = cudaMemHandleTypeNone;
            props.location.type = cudaMemLocationTypeDevice;
            props.location.id = cur_dev;
            if (cudaMemPoolCreate(&phase_pool, &props) == cudaSuccess && phase_pool) {
                uint64_t thr = UINT64_MAX;
                (void) cudaMemPoolSetAttribute(phase_pool, cudaMemPoolAttrReleaseThreshold, &thr);
                use_phase_pool = true;
            } else {
                (void) cudaGetLastError();
                phase_pool = nullptr;
                use_phase_pool = false;
            }
        }
        has_quant_gemm_scratch = (m->has_quant_dense || enable_gemm_quant);
        if (!gemm.init(st, has_quant_gemm_scratch ? (size_t) (8 * 1024 * 1024) : 4096, err)) return false;

        // Allocate KV caches & contiguous recurrent shortconv state per virtual layer
        layer_pre_graph.assign((size_t) hp.n_layer, nullptr);
        pp_layer_cnt.assign((size_t) hp.n_layer, 0);
        kc.resize((size_t) hp.n_layer, nullptr);
        vc.resize((size_t) hp.n_layer, nullptr);
        kq.resize((size_t) hp.n_layer, nullptr);
        vq.resize((size_t) hp.n_layer, nullptr);
        ks.resize((size_t) hp.n_layer, nullptr);
        vs.resize((size_t) hp.n_layer, nullptr);
        kv_cap.resize((size_t) hp.n_layer);
        conv_hist.resize((size_t) hp.n_layer);
        layer_conv_floats = (size_t) hp.n_sconv_channels() * 3;
        CK(cudaMalloc(&conv_hist_pool, (size_t) hp.n_layer * layer_conv_floats * sizeof(float)));
        use_shared_swa_scratch = (std::getenv("STRATA_NO_SHARED_SWA_SCRATCH") == nullptr);
        const int ring_cap_init = swa_ring_cap();
        prefill_swa_kv_cap = use_shared_swa_scratch
            ? ring_cap_init
            : ((active_pp_cap <= 1) ? (int) hp.n_swa : (int) (hp.n_swa + active_pp_cap));
        swa_kv_cap = prefill_swa_kv_cap;
        swa_kv_pool_bytes = swa_kv_pool_size_for(swa_kv_cap);
        if (swa_kv_pool_bytes > 0) {
            CK(pool_alloc((void**) &swa_kv_pool, swa_kv_pool_bytes));
            bind_swa_kv_pool(swa_kv_pool, swa_kv_cap, swa_ptrs, true);
        }
        for (int il = 0; il < hp.n_layer; ++il) {
            if (!hp.is_swa(il)) {
                const int cap = full_kv_cap;
                kv_cap[(size_t) il] = cap;
                if (kv_q8) {
                    CK(cudaMalloc(&kq[(size_t) il], (size_t) cap * hp.kv_dim * sizeof(int8_t)));
                    CK(cudaMalloc(&vq[(size_t) il], (size_t) cap * hp.kv_dim * sizeof(int8_t)));
                    CK(cudaMalloc(&ks[(size_t) il], (size_t) cap * hp.n_head_kv * sizeof(__half)));
                    CK(cudaMalloc(&vs[(size_t) il], (size_t) cap * hp.n_head_kv * sizeof(__half)));
                } else {
                    CK(cudaMalloc(&kc[(size_t) il], (size_t) cap * hp.kv_dim * sizeof(__half)));
                    CK(cudaMalloc(&vc[(size_t) il], (size_t) cap * hp.kv_dim * sizeof(__half)));
                }
            }
            conv_hist[(size_t) il] = conv_hist_pool + (size_t) il * layer_conv_floats;
        }

        auto d_f32 = [](float** p, size_t n) { CK(cudaMalloc(p, n * sizeof(float))); };
        norm_blk_cnt = (hp.core_start == 6 && hp.core_end == 18 && hp.attn_res_block_size == 6 && hp.num_loops == 3) ? 5 : 9;
        d_f32(&qgr_t1, (size_t) (2 * hp.q_dim + hp.n_rel * hp.n_head + 2 * hp.kv_dim + 2 * hp.n_conv * hp.n_head_kv));
        bf16_x_f16 = gemm.bf16_converts_x() || profile.force_bf16_x_f16;
        int64_t max_bf16_w = 0;
        {
            auto note_bf16 = [&](const DenseMat& W) {
                if (W.type == 30 && W.d) {
                    max_bf16_w = std::max(max_bf16_w, W.n_out * W.n_in);
                    if (W.n_out > 64) has_bf16_large_dense = true;
                }
            };
            for (const auto& L : m->layers) {
                note_bf16(L.attn_q); note_bf16(L.attn_k); note_bf16(L.attn_v);
                note_bf16(L.attn_gate); note_bf16(L.attn_rel); note_bf16(L.attn_sink_proj);
                note_bf16(L.attn_k_conv_proj); note_bf16(L.attn_v_conv_proj); note_bf16(L.attn_output);
                for (int mod = 0; mod < 4; ++mod) {
                    note_bf16(L.loop_down[mod]); note_bf16(L.loop_gain[mod]); note_bf16(L.loop_forget[mod]);
                }
            }
            note_bf16(m->output_weight);
            if (max_bf16_w > 0) {
                has_bf16_dense = true;
                gemm.reserve_bf16_tc(max_bf16_w, bf16_x_f16 ? 0 : (int64_t) tmax * 2 * N);
            }
        }
        attn_split_t1_full = std::clamp((sm_count * profile.t1_full_split_mult) / std::max(1, (int) hp.n_head_kv), 40, kAttnSplitMax);
        attn_split_t1_swa = std::clamp((sm_count * profile.t1_swa_split_mult) / std::max(1, (int) hp.n_head_kv), 8, 32);
        {
            const int n_hkv = std::max(1, (int) hp.n_head_kv);
            const int max_t_split = std::min(std::max(verify_max_t, 1), 15);
            int max_slots = std::max(attn_split_t1_full, attn_split_t1_swa);
            for (int t = 2; t <= max_t_split; ++t) {
                int sp_t = 1;
                if (t <= 5) {
                    const int sp_swa = std::clamp((sm_count + n_hkv - 1) / n_hkv, 4, 16);
                    const int sp_full = std::clamp((sm_count * 24) / n_hkv, 40, kAttnSplitMax);
                    sp_t = std::max(sp_swa, sp_full);
                } else {
                    const int sp_swa = std::clamp((sm_count * 2) / (n_hkv * t), 4, 24);
                    const int sp_full = std::clamp((sm_count * 16) / (n_hkv * t), 16, kAttnSplitMax);
                    sp_t = std::max(sp_swa, sp_full);
                }
                max_slots = std::max(max_slots, t * sp_t);
            }
            attn_split_slots = max_slots;
        }
        // Configure dynamic shared memory for WMMA kernels upfront (required before CUDA Graph capture).
        // Gate high-SMEM FA2 kernels on cudaDevAttrMaxSharedMemoryPerBlockOptin so Turing (sm_75, 64 KB cap) cleanly falls back to pp_wmma (29 KB).
        auto try_set_smem = [&](const void* fn, int bytes) -> bool {
            if (max_smem_optin < bytes) return false;
            const cudaError_t e1 = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
            if (e1 != cudaSuccess) { (void) cudaGetLastError(); return false; }
            (void) cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
            (void) cudaGetLastError();
            return true;
        };
        try_set_smem((const void*) k_attn_q8_gqa3_verify4_wmma_split<5>, 24576);
        try_set_smem((const void*) k_attn_q8_gqa3_verify4_wmma_split<4>, 24576);
        try_set_smem((const void*) k_attn_q8_gqa3_verify4_wmma_split<3>, 24576);
        try_set_smem((const void*) k_attn_q8_gqa3_verify4_wmma_split<2>, 24576);
        try_set_smem((const void*) k_attn_q8_gqa3_verify4_swa_wmma_split<5>, 24576);
        try_set_smem((const void*) k_attn_q8_gqa3_verify4_swa_wmma_split<4>, 24576);
        try_set_smem((const void*) k_attn_q8_gqa3_verify4_swa_wmma_split<3>, 24576);
        try_set_smem((const void*) k_attn_q8_gqa3_verify4_swa_wmma_split<2>, 24576);
        can_ctrl_wmma = (cc_maj >= 7);
        can_ctrl_down_wmma = (cc_maj >= 8);
        cudaFuncSetAttribute((const void*) k_control_up_combine_pp_wmma, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        cudaFuncSetAttribute((const void*) k_control_down_phaseA_pp_wmma_splitk<3, true>,  cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        cudaFuncSetAttribute((const void*) k_control_down_phaseA_pp_wmma_splitk<3, false>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        cudaFuncSetAttribute((const void*) k_control_down_phaseA_pp_wmma_splitk<1, true>,  cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        cudaFuncSetAttribute((const void*) k_control_down_phaseA_pp_wmma_splitk<1, false>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        cudaFuncSetAttribute((const void*) k_control_up_combine_pp_wmma_vmsg<3>, cudaFuncAttributePreferredSharedMemoryCarveout, 50);
        cudaFuncSetAttribute((const void*) k_control_up_combine_pp_wmma_vmsg<1>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        cudaFuncSetAttribute((const void*) k_router_sgemm_f32_pp_splitk, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        if (cc_maj >= 8) {
            cudaFuncSetAttribute((const void*) k_router_ozaki2_tc_presplit_pp,
                                 cudaFuncAttributePreferredSharedMemoryCarveout,
                                 cudaSharedmemCarveoutMaxShared);
        }
        (void) cudaGetLastError();
        can_ctrl_rms_3 = (cc_maj >= 8) &&
            try_set_smem((const void*) k_control_up_combine_rms_f16_vmsg<3>, 29696);
        can_ctrl_rms_1 = (cc_maj >= 8) &&
            try_set_smem((const void*) k_control_up_combine_rms_f16_vmsg<1>, 29696);
        (void) cudaGetLastError();
        can_fa2_192 =
            try_set_smem((const void*) k_attn_q8_gqa3_pp_fa2_192<false, false, false>, 98304) &&
            try_set_smem((const void*) k_attn_q8_gqa3_pp_fa2_192<true, false, false>, 98304) &&
            try_set_smem((const void*) k_attn_q8_gqa3_pp_fa2_192<false, true, false>, 98304) &&
            try_set_smem((const void*) k_attn_q8_gqa3_pp_fa2_192<false, true, true>, 98304);
        fa2_192_swa_smem = (cc_maj >= 8) ? 97280 : 73728;
        can_fa2_192_swa_fast = ((cc_maj == 7 && cc_min < 5) || cc_maj >= 8) &&
            try_set_smem((const void*) k_attn_q8_gqa3_pp_fa2_192_swa_fast<true, true>, fa2_192_swa_smem) &&
            try_set_smem((const void*) k_attn_q8_gqa3_pp_fa2_192_swa_fast<false, true>, fa2_192_swa_smem) &&
            try_set_smem((const void*) k_attn_q8_gqa3_pp_fa2_192_swa_fast<true, false>, fa2_192_swa_smem) &&
            try_set_smem((const void*) k_attn_q8_gqa3_pp_fa2_192_swa_fast<false, false>, fa2_192_swa_smem) &&
            try_set_smem((const void*) k_attn_q8_gqa3_pp_fa2_192_swa_fast<true, true, true>, fa2_192_swa_smem) &&
            try_set_smem((const void*) k_attn_q8_gqa3_pp_fa2_192_swa_fast<true, false, true>, fa2_192_swa_smem);
        fa2_256_bulk_smem = (cc_maj == 7 && cc_min < 5) ? 92160 : 87040;
        fa2_256_tail_smem = (cc_maj == 7 && cc_min < 5) ? 55296 : 52224;
        can_fa2_256 =
            try_set_smem((const void*) k_attn_q8_gqa3_pp_fa2_256_bulk, fa2_256_bulk_smem) &&
            try_set_smem((const void*) k_attn_q8_gqa3_pp_fa2_256_tail, fa2_256_tail_smem);
        try_set_smem((const void*) k_attn_q8_gqa3_pp_wmma_gqa3, 49152);
        try_set_smem((const void*) k_attn_q8_gqa3_pp_wmma_t32, 49152);
        try_set_smem((const void*) k_attn_q8_gqa3_pp_wmma, 29184);

        const int64_t max_ent = (int64_t) meta_tmax * hp.n_expert_used;

        CK(cudaMallocHost(&h_tok_buf, (size_t) meta_tmax * N * sizeof(float)));
        CK(cudaMallocHost(&h_is_img_pin, (size_t) meta_tmax * sizeof(uint8_t)));
        CK(cudaMalloc(&d_is_img, (size_t) meta_tmax * sizeof(uint8_t)));
        tok_seen.assign((size_t) std::max<int64_t>(hp.vocab_padded, hp.vocab_unpadded), -1);
        tok_uniq.reserve((size_t) meta_tmax);
        CK(cudaMallocHost(&h_tok_idx_pin, (size_t) meta_tmax * sizeof(int32_t)));
        h_tok_raw_bytes = (size_t) meta_tmax * m->tok_embd_row_bytes;
        if (h_tok_raw_bytes > 0) {
            if (cudaMallocHost(&h_tok_raw_pin, h_tok_raw_bytes) != cudaSuccess) {
                (void) cudaGetLastError();
                h_tok_raw_pin = nullptr;
                h_tok_raw_bytes = 0;
            }
        }

        const int q8_kv_rows = std::min(tmax, 8);
        // xq is only ever used for T <= 8 (decode/verify) and DSpark block_n=8 (4*N).
        const size_t xq_blocks = std::max<size_t>((size_t) q8_kv_rows * ((2 * (size_t) N) / 32), (size_t) 16 * ((4 * (size_t) N) / 32));
        CK(cudaMalloc(&xq, xq_blocks * 36));
        CK(cudaMalloc(&xq_k, (size_t) q8_kv_rows * ((2 * N) / 32) * 36));
        CK(cudaMalloc(&xq_v, (size_t) q8_kv_rows * ((2 * N) / 32) * 36));

        // Wave 5: Check if all 12 core layers support virtual-message control-down & upload static 48-ptr W_down table once.
        can_vmsg_all_core = false;
        if (cc_maj >= 8 && N == 2304 && hp.loop_gate_rank == 64 && hp.num_loops > 1 &&
            (hp.core_end - hp.core_start) == 12 && norm_blk_cnt == 5) {
            bool ok = true;
            const float* h_wdown[48] = {};
            for (int l = 0; l < 12; ++l) {
                const PhysLayer& Lc = m->layers[(size_t) (hp.core_start + l)];
                if (!Lc.attn_res_query || Lc.attn_q.type == 0 || Lc.attn_k.type == 0 || Lc.attn_v.type == 0 ||
                    Lc.attn_gate.type == 0 || Lc.attn_rel.type == 0 || Lc.attn_sink_proj.type == 0) {
                    ok = false; break;
                }
                if (!bf16_x_f16 && (Lc.attn_q.type == 30 || Lc.attn_k.type == 30 || Lc.attn_v.type == 30 ||
                                    Lc.attn_gate.type == 30 || Lc.attn_rel.type == 30)) {
                    ok = false; break;
                }
                if (!Lc.is_swa && (Lc.attn_k_conv_proj.type == 0 || Lc.attn_v_conv_proj.type == 0)) {
                    ok = false; break;
                }
                for (int mod = 0; mod < 4; ++mod) {
                    if (Lc.loop_down[mod].type != 0 || !Lc.loop_down[mod].d ||
                        Lc.loop_gain[mod].type != 0 || !Lc.loop_gain[mod].d ||
                        Lc.loop_forget[mod].type != 0 || !Lc.loop_forget[mod].d) {
                        ok = false; break;
                    }
                    h_wdown[l * 4 + mod] = (const float*) Lc.loop_down[mod].d;
                }
                if (!ok) break;
            }
            if (ok) {
                CK(cudaMalloc(&d_vmsg_wdown_ptrs, 48 * sizeof(const float*)));
                CK(cudaMemcpy(d_vmsg_wdown_ptrs, h_wdown, 48 * sizeof(const float*), cudaMemcpyHostToDevice));
                if (cc_maj >= 8 && N == 2304) {
                    CK(cudaMalloc(&d_core_wdown_msg_f16, (size_t) 48 * 64 * (size_t) N * sizeof(uint16_t)));
                    k_pack_core_wdown_msg_f16<<<dim3(48, 64), 256, 0, st>>>(d_vmsg_wdown_ptrs, d_core_wdown_msg_f16, (int) N);
                    // Wave 8.5: Pre-pack bit-exact FP16 cur-half loop_down (13.5 MiB) & loop_gain+forget (27.0 MiB).
                    // Strictly skipped on TightHybrid (262K ctx) or tight free VRAM so 262K loses ZERO VRAM!
                    static const bool no_core_f16_pack = (std::getenv("STRATA_NO_CORE_F16_PACK") != nullptr);
                    size_t fr_pack = 0, tot_pack = 0;
                    CK(cudaMemGetInfo(&fr_pack, &tot_pack));
                    if (!no_core_f16_pack && profile.regime != strata::inkling2::VramRegime::FullVram &&
                        fr_pack > vram_reserve_bytes + ((size_t) 512 << 20)) {
                        const size_t wdown_cur_elems = (size_t) 48 * 64u * (size_t) N;
                        const size_t wup_mat_elems = (size_t) N * 64u;
                        const size_t wup_tot_elems = (size_t) 96 * wup_mat_elems;
                        if (cudaMalloc(&d_core_wdown_cur_f16, wdown_cur_elems * sizeof(__half)) == cudaSuccess && d_core_wdown_cur_f16) {
                            k_pack_core_wdown_cur_f16<<<dim3(48, 64), 256, 0, st>>>(d_vmsg_wdown_ptrs, d_core_wdown_cur_f16, (int) N);
                        } else {
                            (void) cudaGetLastError();
                            d_core_wdown_cur_f16 = nullptr;
                        }
                        if (cudaMalloc(&d_core_wup_f16, wup_tot_elems * sizeof(__half)) == cudaSuccess && d_core_wup_f16) {
                            const unsigned pack_blks = (unsigned) (((wup_mat_elems >> 2) + 255u) / 256u);
                            for (int l = 0; l < 12; ++l) {
                                const PhysLayer& Lc = m->layers[(size_t) (hp.core_start + l)];
                                for (int mod = 0; mod < 4; ++mod) {
                                    const size_t idx = (size_t) (l * 4 + mod);
                                    k_pack_f32_to_f16_exact<<<pack_blks, 256, 0, st>>>(
                                        (const float*) Lc.loop_gain[mod].d,
                                        d_core_wup_f16 + idx * wup_mat_elems,
                                        (int) wup_mat_elems);
                                    k_pack_f32_to_f16_exact<<<pack_blks, 256, 0, st>>>(
                                        (const float*) Lc.loop_forget[mod].d,
                                        d_core_wup_f16 + (48u + idx) * wup_mat_elems,
                                        (int) wup_mat_elems);
                                }
                            }
                        } else {
                            (void) cudaGetLastError();
                            d_core_wup_f16 = nullptr;
                        }
                    }
                }
                can_vmsg_all_core = true;
            }
        }

        if (const char* v = std::getenv("STRATA_PP_STREAM_T")) pp_stream_t = std::max(1, std::atoi(v));
        prefill_ephemeral_bytes = alloc_ephemeral_bufs(active_pp_cap);
        in_decode_phase = false;

        CK(cudaMalloc(&ids, (size_t) max_ent * sizeof(int32_t)));
        CK(cudaMalloc(&grp_start, (size_t) (hp.n_expert + 1) * sizeof(int32_t)));
        CK(cudaMalloc(&n_groups, sizeof(int32_t)));
        CK(cudaMalloc(&ent_dst, (size_t) max_ent * sizeof(int32_t)));
        CK(cudaMalloc(&ent_tok, (size_t) max_ent * sizeof(int32_t)));
        CK(cudaMalloc(&grp_ptr, (size_t) hp.n_expert * sizeof(unsigned long long)));
        CK(cudaMalloc(&d_grp_part_cnt, 2 * 16 * 128 * sizeof(int32_t)));
        d_grp_part_off = d_grp_part_cnt + 16 * 128;
        {
            const size_t hits_tot = (size_t) hp.n_phys * (size_t) hp.n_expert;
            CK(cudaMalloc(&d_hits_pool, hits_tot * sizeof(uint32_t)));
            CK(cudaMemset(d_hits_pool, 0, hits_tot * sizeof(uint32_t)));
            for (int ip = 0; ip < hp.n_phys; ++ip) {
                m->layers[(size_t) ip].d_hits = d_hits_pool + (size_t) ip * (size_t) hp.n_expert;
            }
        }
        CK(cudaMalloc(&d_router_done, sizeof(uint32_t)));
        CK(cudaMemset(d_router_done, 0, sizeof(uint32_t)));
        pos0_sub_cap = 1 + (meta_tmax + 255) / 256;
        CK(cudaMalloc(&pos0_d, (size_t) pos0_sub_cap * sizeof(int32_t)));
        CK(cudaMalloc(&pick_d, (size_t) meta_tmax * sizeof(int32_t)));
        CK(cudaMalloc(&argmax_val, (size_t) verify_max_t * 64 * sizeof(float)));
        CK(cudaMalloc(&argmax_idx, (size_t) verify_max_t * 64 * sizeof(int32_t)));
        // GPU Top-K + penalty buffers (< 128 KiB VRAM)
        CK(cudaMalloc(&d_topk_hist, (size_t) verify_max_t * 2048 * sizeof(uint32_t)));
        CK(cudaMalloc(&d_topk_thr, (size_t) verify_max_t * sizeof(int32_t)));
        CK(cudaMalloc(&d_topk_cnt, (size_t) verify_max_t * sizeof(int32_t)));
        CK(cudaMalloc(&d_topk_out_cnt, (size_t) verify_max_t * sizeof(int32_t)));
        CK(cudaMalloc(&d_topk_cand_val, (size_t) verify_max_t * 2048 * sizeof(float)));
        CK(cudaMalloc(&d_topk_cand_idx, (size_t) verify_max_t * 2048 * sizeof(int32_t)));
        CK(cudaMalloc(&d_topk_out_val, (size_t) verify_max_t * 256 * sizeof(float)));
        CK(cudaMalloc(&d_topk_out_idx, (size_t) verify_max_t * 256 * sizeof(int32_t)));
        CK(cudaMalloc(&d_pen_toks, 256 * sizeof(int32_t)));
        CK(cudaMalloc(&d_pen_cnts, 256 * sizeof(int32_t)));
        CK(cudaMallocHost(&h_topk_val_pin, (size_t) verify_max_t * 256 * sizeof(float)));
        CK(cudaMallocHost(&h_topk_idx_pin, (size_t) verify_max_t * 256 * sizeof(int32_t)));
        CK(cudaMallocHost(&h_topk_cnt_pin, (size_t) verify_max_t * sizeof(int32_t)));
        CK(cudaMallocHost(&h_pen_toks_pin, 256 * sizeof(int32_t)));
        CK(cudaMallocHost(&h_pen_cnts_pin, 256 * sizeof(int32_t)));
        CK(cudaMallocHost(&h_slot_pin, (size_t) hp.n_phys * (size_t) hp.n_expert * sizeof(int32_t)));
        CK(cudaMallocHost(&h_pick_pin, (size_t) verify_max_t * sizeof(int32_t)));
        CK(cudaMallocHost(&h_pos_pin, (size_t) pos0_sub_cap * sizeof(int32_t)));

        CK(cudaMallocHost(&h_ids, (size_t) max_ent * sizeof(int32_t)));
        CK(cudaMallocHost(&h_ewts, (size_t) max_ent * sizeof(float)));
        CK(cudaMallocHost(&h_cpu_sum, (size_t) std::max(verify_max_t, 8) * (size_t) N * sizeof(float)));
        CK(cudaMallocHost(&h_hits_pin, (size_t) hp.n_phys * (size_t) hp.n_expert * sizeof(uint32_t)));
        for (int t = 0; t < 10; ++t) verify_pre_graph[t].assign((size_t) hp.n_layer, nullptr);
        CK(cudaEventCreateWithFlags(&ev_route, cudaEventDisableTiming));
        for (int i = 0; i < kMaxRebalLayers; ++i) {
            CK(cudaEventCreateWithFlags(&ev_rebal_layer[i], cudaEventDisableTiming));
            layer_pending_rebal[i] = false;
        }
        CK(cudaStreamCreateWithFlags(&cst, cudaStreamNonBlocking));
        for (int b = 0; b < kMaxPpStages; ++b) {
            CK(cudaEventCreateWithFlags(&ev_pp_copy[b], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&ev_pp_used[b], cudaEventDisableTiming));
        }
        // EX-4C Wave 3 + EX-5: Pre-warm iq_kernels.cu AND inkling2_engine.cu cubins on st at startup
        // (eliminates ~55-58 ms cold-start cubin-behind-H2D stall on Run 1!).
        if (!m->layers.empty()) {
            (void) K::native_expert_grouped_wmma(m->layers[0].L, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, nullptr, nullptr, nullptr, st, 0);
        }
        if (cur && in_q && act_bf16 && layer_arena && layer_arena_bytes >= (size_t) 3 * (size_t) N * sizeof(float)) {
            CK(cudaMemsetAsync(cur, 0, (size_t) N * sizeof(float), st));
            CK(cudaMemsetAsync(layer_arena, 0, (size_t) 3 * (size_t) N * sizeof(float), st));
            k_rms_norm_f16<<<1, (N == 2304) ? 576 : kThreads, 0, st>>>(cur, m->tok_embd_norm, in_q, act_bf16, (int) N, hp.rms_eps);
            if (N == 2304) {
                NormMixPtrs wp{};
                wp.p[0] = cur;
                MixOuts4 mo4{{nullptr, nullptr, nullptr, nullptr}};
                NormF16Out3 nf16{m->tok_embd_norm, m->tok_embd_norm, m->tok_embd_norm, act_bf16, nullptr, nullptr};
                k_mix_normed_fused_1pass_f16<3><<<1, 576, 0, st>>>(wp, 1, reinterpret_cast<const float*>(layer_arena),
                                                                   mo4, nf16, (int) N, 1.0f / std::sqrt((float) N), hp.rms_eps);
            }
            CK(cudaStreamSynchronize(st));
        }
        reset_state();
        return true;
    }


    inline void launch_f32_to_f16(const float* src, uint16_t* dst, int64_t tot) {
        const int64_t nt = ((tot & 3) == 0) ? (tot >> 2) : tot;
        k_f32_to_f16<<<(unsigned) ((nt + 255) / 256), 256, 0, st>>>(src, dst, tot);
    }

    inline void launch_f32_to_bf16(const float* src, uint16_t* dst, int64_t tot) {
        const int64_t nt = ((tot & 3) == 0) ? (tot >> 2) : tot;
        k_f32_to_bf16<<<(unsigned) ((nt + 255) / 256), 256, 0, st>>>(src, dst, tot);
    }

    /// Exact GPU Top-K (K <= 256) + optional in-place penalties over `rows` rows of `logits + row_off*V` [rows, V].
    /// Returns true on success (writing sorted descending top-K vals/idxs to out_vals/out_idxs); false on bucket overflow.
    bool gpu_topk_rows(int rows, int K, const int32_t* pen_toks, const int32_t* pen_cnts, int n_pen,
                       float rep_pen, float freq_pen, float pres_pen, float* out_vals, int32_t* out_idxs, int row_off = 0) {
        const int V = (int) m->hp.vocab_unpadded;
        if (rows <= 0 || row_off < 0 || row_off + rows > verify_max_t || K <= 0 || K > 256 || K > V) return false;
        float* lgt = logits + (size_t) row_off * (size_t) V;
        if (n_pen > 0 && n_pen <= 256 && (rep_pen != 1.0f || freq_pen != 0.0f || pres_pen != 0.0f)) {
            std::memcpy(h_pen_toks_pin, pen_toks, (size_t) n_pen * sizeof(int32_t));
            std::memcpy(h_pen_cnts_pin, pen_cnts, (size_t) n_pen * sizeof(int32_t));
            CK(cudaMemcpyAsync(d_pen_toks, h_pen_toks_pin, (size_t) n_pen * sizeof(int32_t), cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(d_pen_cnts, h_pen_cnts_pin, (size_t) n_pen * sizeof(int32_t), cudaMemcpyHostToDevice, st));
            k_apply_penalties<<<dim3((unsigned) ((n_pen + 63) / 64), (unsigned) rows), 64, 0, st>>>(
                lgt, V, d_pen_toks, d_pen_cnts, n_pen, rep_pen, freq_pen, pres_pen);
        }
        constexpr int kHistBlks = 64;
        CK(cudaMemsetAsync(d_topk_hist, 0, (size_t) rows * 2048 * sizeof(uint32_t), st));
        k_topk_hist<<<dim3(kHistBlks, (unsigned) rows), 256, 0, st>>>(lgt, d_topk_hist, V);
        k_topk_select_thresh<<<(unsigned) rows, 256, 0, st>>>(d_topk_hist, d_topk_thr, d_topk_cnt, K);
        k_topk_collect<<<dim3(kHistBlks, (unsigned) rows), 256, 0, st>>>(
            lgt, d_topk_thr, d_topk_cand_val, d_topk_cand_idx, d_topk_cnt, V);
        k_topk_finalize<<<(unsigned) rows, 256, 0, st>>>(
            d_topk_cand_val, d_topk_cand_idx, d_topk_cnt, d_topk_out_val, d_topk_out_idx, d_topk_out_cnt, K);
        CK(cudaMemcpyAsync(h_topk_val_pin, d_topk_out_val, (size_t) rows * (size_t) K * sizeof(float), cudaMemcpyDeviceToHost, st));
        CK(cudaMemcpyAsync(h_topk_idx_pin, d_topk_out_idx, (size_t) rows * (size_t) K * sizeof(int32_t), cudaMemcpyDeviceToHost, st));
        CK(cudaMemcpyAsync(h_topk_cnt_pin, d_topk_out_cnt, (size_t) rows * sizeof(int32_t), cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        for (int r = 0; r < rows; ++r) {
            if (h_topk_cnt_pin[r] != K) return false;
        }
        std::memcpy(out_vals, h_topk_val_pin, (size_t) rows * (size_t) K * sizeof(float));
        std::memcpy(out_idxs, h_topk_idx_pin, (size_t) rows * (size_t) K * sizeof(int32_t));
        return true;
    }

    void init_cpu_bufs(int cpu_t) {
        if (h_h != nullptr && cpu_t <= cpu_buf_cap) return;
        if (h_h != nullptr) {
            if (st) CK(cudaStreamSynchronize(st));
            cudaFreeHost(h_h); h_h = nullptr;
            cudaFreeHost(cpu_out); cpu_out = nullptr;
            cudaFreeHost(h_miss); h_miss = nullptr;
            if (d_miss) { cudaFree(d_miss); d_miss = nullptr; }
            if (ecpu) { cudaFree(ecpu); ecpu = nullptr; }
#if defined(STRATA_INKLING2_CPU_POOL)
            std::free(acts); acts = nullptr;
#endif
            cpu_buf_cap = 0;
        }
        const int64_t N = m->hp.n_embd;
        const int64_t cpu_ent = (int64_t) cpu_t * m->hp.n_expert_used;
        CK(cudaMallocHost(&h_h, (size_t) cpu_t * N * sizeof(float)));
        CK(cudaMallocHost(&cpu_out, (size_t) cpu_ent * N * sizeof(float)));
        CK(cudaMallocHost(&h_miss, (size_t) cpu_ent * sizeof(int32_t)));
        CK(cudaMalloc(&d_miss, (size_t) cpu_ent * sizeof(int32_t)));
        CK(cudaMalloc(&ecpu, (size_t) cpu_ent * N * sizeof(float)));
#if defined(STRATA_INKLING2_CPU_POOL)
        acts = (uint8_t*) std::malloc((size_t) cpu_t * strata::kernels::cpu::kNativeActBytes);
#endif
        cpu_buf_cap = cpu_t;
    }

    void reset_state() {
        pos = 0;
        decode_scores_normalized = false;
        pld.reset();
        CK(cudaMemsetAsync(conv_hist_pool, 0, (size_t) m->hp.n_layer * layer_conv_floats * sizeof(float), st));
        if (dspark) dspark->reset(st);
    }

    void enable_dspark(DsparkEngine* ds) {
        dspark = ds;
        const auto& hp = m->hp;
        const bool map_tgt_feat = (profile.regime != strata::inkling2::VramRegime::FullVram);
        const size_t feat_bytes = (size_t) tmax * (size_t) hp.n_embd * sizeof(float);
        for (int k = 0; k < 4; ++k) {
            if (!d_tgt_feat[k]) {
                if (map_tgt_feat &&
                    cudaHostAlloc(&h_tgt_feat_pin[k], feat_bytes, cudaHostAllocMapped) == cudaSuccess) {
                    CK(cudaHostGetDevicePointer(&d_tgt_feat[k], h_tgt_feat_pin[k], 0));
                } else {
                    (void) cudaGetLastError();
                    h_tgt_feat_pin[k] = nullptr;
                    CK(cudaMalloc(&d_tgt_feat[k], feat_bytes));
                }
            }
        }
        const bool defer_conv_trace = (use_phase_pool && phase_pool != nullptr &&
                                       profile.regime != strata::inkling2::VramRegime::FullVram &&
                                       tmax >= 1024 && std::getenv("STRATA_NO_PHASE_SPLIT") == nullptr);
        if (!defer_conv_trace) alloc_conv_trace();
        swa_bak_slots = std::max(1, verify_max_t - 1);
        const size_t slot_bytes = kv_q8
            ? ((size_t) hp.kv_dim * 2 + (size_t) hp.n_head_kv * 2 * sizeof(__half))
            : ((size_t) hp.kv_dim * 2 * sizeof(__half));
        if (!swa_bak) CK(cudaMalloc(&swa_bak, (size_t) swa_bak_slots * swa_ptrs.n_swa_layers * slot_bytes));
    }

    /// Commit `1 + m` tokens `[anchor, d_0..d_{m-1}]` out of `1 + K` verified tokens starting at `pos_pre`.
    void commit_verify(int m_accepted, int K, int pos_pre) {
        const auto& hp = m->hp;
        if (m_accepted < K) {
            const size_t conv_bytes = (size_t) hp.n_layer * layer_conv_floats * sizeof(float);
            const size_t total_conv = (size_t) hp.n_layer * layer_conv_floats;
            CK(cudaMemcpyAsync(conv_hist_pool, d_conv_trace + (size_t) m_accepted * total_conv,
                               conv_bytes, cudaMemcpyDeviceToDevice, st));
            if (swa_ptrs.n_swa_layers > 0) {
                const int n_restore = K - m_accepted;
                if (kv_q8) {
                    k_swa_q8_bak<<<dim3((unsigned) n_restore, (unsigned) swa_ptrs.n_swa_layers), 256, 0, st>>>(
                        swa_ptrs, swa_bak, (int) hp.kv_dim, (int) hp.n_head_kv, swa_kv_cap,
                        pos_pre + 1, m_accepted, n_restore, 1);
                } else {
                    k_swa_f16_bak<<<dim3((unsigned) n_restore, (unsigned) swa_ptrs.n_swa_layers), 256, 0, st>>>(
                        swa_ptrs, swa_bak, (int) hp.kv_dim, swa_kv_cap,
                        pos_pre + 1, m_accepted, n_restore, 1);
                }
            }
        }
        pos = pos_pre + 1 + m_accepted;
    }

    void prep_q8(const float* X, void* q8_buf, int T, int64_t K_in) {
        if (T <= 8 && (K_in % 32) == 0) {
            K::native_quantize_q8_1(X, q8_buf, (int) K_in, T, st);
        }
    }

    void project_preq(const float* X, const void* q8_ready, const DenseMat& W, float* Y, int T, int64_t N_out, int64_t K_in,
                      const uint16_t* f16_ready = nullptr, const uint16_t* bf16_ready = nullptr) {
        if (W.type == 0) {
            project_f32(X, (const float*) W.d, Y, T, N_out, K_in);
            return;
        }
        if (W.type == 1) {
            const uint16_t* wp = (const uint16_t*) W.d;
            if (T >= 1 && T <= 8 && (K_in % 4) == 0 && ((((uintptr_t) wp) & 7u) == 0u) && ((((uintptr_t) X) & 15u) == 0u)) {
                if (N_out <= 128) {
                    if (T == 1) {
                        k_gemm_f16_smallT_cta<1, true><<<(unsigned) N_out, 256, 0, st>>>(wp, X, Y, (int) K_in, (int) N_out, 1);
                    } else if (T == 4) {
                        k_gemm_f16_smallT_cta<4, true><<<(unsigned) N_out, 256, 0, st>>>(wp, X, Y, (int) K_in, (int) N_out, 4);
                    } else if (T == 2) {
                        k_gemm_f16_smallT_cta<2, true><<<(unsigned) N_out, 256, 0, st>>>(wp, X, Y, (int) K_in, (int) N_out, 2);
                    } else if (T == 3) {
                        k_gemm_f16_smallT_cta<3, true><<<(unsigned) N_out, 256, 0, st>>>(wp, X, Y, (int) K_in, (int) N_out, 3);
                    } else {
                        k_gemm_f16_smallT_cta<8, false><<<(unsigned) N_out, 256, 0, st>>>(wp, X, Y, (int) K_in, (int) N_out, T);
                    }
                } else {
                    const unsigned nb = (unsigned) ((N_out + 7) / 8);
                    if (T == 1) {
                        k_gemm_f16_smallT_warp<1, true><<<nb, 256, 0, st>>>(wp, X, Y, (int) K_in, (int) N_out, 1);
                    } else if (T == 4) {
                        k_gemm_f16_smallT_warp<4, true><<<nb, 256, 0, st>>>(wp, X, Y, (int) K_in, (int) N_out, 4);
                    } else if (T == 2) {
                        k_gemm_f16_smallT_warp<2, true><<<nb, 256, 0, st>>>(wp, X, Y, (int) K_in, (int) N_out, 2);
                    } else if (T == 3) {
                        k_gemm_f16_smallT_warp<3, true><<<nb, 256, 0, st>>>(wp, X, Y, (int) K_in, (int) N_out, 3);
                    } else {
                        k_gemm_f16_smallT_warp<8, false><<<nb, 256, 0, st>>>(wp, X, Y, (int) K_in, (int) N_out, T);
                    }
                }
                return;
            }
            const uint16_t* xf16 = f16_ready;
            if (!xf16) {
                launch_f32_to_f16(X, act_bf16, (int64_t) T * K_in);
                xf16 = act_bf16;
            }
            gemm.f16(xf16, wp, Y, T, N_out, K_in);
            return;
        }
        if (W.type == 30) {
            const uint16_t* wp = (const uint16_t*) W.d;
            if (T == 1) {
                K::bf16_gemv_fp32_mmvf(X, wp, Y, K_in, N_out, st);
            } else if (T >= 2 && T <= 8 && (K_in % 4) == 0 && ((((uintptr_t) wp) & 7u) == 0u) && ((((uintptr_t) X) & 15u) == 0u)) {
                if (N_out <= 128) {
                    if (T == 4) {
                        k_gemm_bf16_smallT_cta<4, true><<<(unsigned) N_out, 256, 0, st>>>(wp, X, Y, (int) K_in, (int) N_out, 4);
                    } else if (T == 2) {
                        k_gemm_bf16_smallT_cta<2, true><<<(unsigned) N_out, 256, 0, st>>>(wp, X, Y, (int) K_in, (int) N_out, 2);
                    } else if (T == 3) {
                        k_gemm_bf16_smallT_cta<3, true><<<(unsigned) N_out, 256, 0, st>>>(wp, X, Y, (int) K_in, (int) N_out, 3);
                    } else {
                        k_gemm_bf16_smallT_cta<8, false><<<(unsigned) N_out, 256, 0, st>>>(wp, X, Y, (int) K_in, (int) N_out, T);
                    }
                } else {
                    const unsigned nb = (unsigned) ((N_out + 7) / 8);
                    if (T == 4) {
                        k_gemm_bf16_smallT_warp<4, true><<<nb, 256, 0, st>>>(wp, X, Y, (int) K_in, (int) N_out, 4);
                    } else if (T == 2) {
                        k_gemm_bf16_smallT_warp<2, true><<<nb, 256, 0, st>>>(wp, X, Y, (int) K_in, (int) N_out, 2);
                    } else if (T == 3) {
                        k_gemm_bf16_smallT_warp<3, true><<<nb, 256, 0, st>>>(wp, X, Y, (int) K_in, (int) N_out, 3);
                    } else {
                        k_gemm_bf16_smallT_warp<8, false><<<nb, 256, 0, st>>>(wp, X, Y, (int) K_in, (int) N_out, T);
                    }
                }
            } else if (T > 8 && N_out <= 64 && f16_ready != nullptr) {
                const int64_t n_w = N_out * K_in;
                if (ctrl_w16 && (size_t) n_w * sizeof(uint16_t) <= ctrl_w16_bytes) {
                    const int64_t n4 = (n_w + 3) >> 2;
                    k_bf16_to_f16<<<(unsigned) ((n4 + 255) / 256), 256, 0, st>>>(wp, ctrl_w16, n_w);
                    gemm.f16(f16_ready, ctrl_w16, Y, T, N_out, K_in);
                } else {
                    gemm.bf16(f16_ready, wp, Y, T, N_out, K_in, 0, 0.0f, 0, true);
                }
            } else if (bf16_x_f16) {
                const uint16_t* xf16 = f16_ready;
                if (!xf16) {
                    launch_f32_to_f16(X, act_bf16, (int64_t) T * K_in);
                    xf16 = act_bf16;
                }
                gemm.bf16(xf16, wp, Y, T, N_out, K_in, 0, 0.0f, 0, true);
            } else {
                const uint16_t* xbf16 = bf16_ready;
                if (!xbf16) {
                    uint16_t* dst_bf16 = (act_bf16_only && (const void*) X != (const void*) act_bf16_only &&
                                          (const void*) X != (const void*) in_ffn)
                                             ? act_bf16_only : act_bf16;
                    launch_f32_to_bf16(X, dst_bf16, (int64_t) T * K_in);
                    xbf16 = dst_bf16;
                }
                gemm.bf16(xbf16, wp, Y, T, N_out, K_in);
            }
            return;
        }
        const auto* ti = strata::quant::type_info(W.type);
        const int blck = ti ? ti->blck : 256;
        if (T <= 8 && (K_in % blck) == 0) {
            if (!q8_ready) K::native_quantize_q8_1(X, xq, (int) K_in, T, st);
            const void* xq_ptr = q8_ready ? q8_ready : xq;
            if (W.type == 8 && !W.q8_in_place && T >= 1 && (K_in % 256) == 0 && ((((uintptr_t) W.d) & 3u) == 0u)) {
                const uint8_t* w8 = (const uint8_t*) W.d;
                const uint32_t* xu = (const uint32_t*) xq_ptr;
                if (T == 1) {
                    if (N_out <= 1024) {
                        k_mmvq_q8_0_smallT<1, true, false, 1><<<(unsigned) N_out, 32, 0, st>>>(w8, xu, Y, (int) K_in, (int) N_out, 1);
                    } else if (N_out <= 3072) {
                        k_mmvq_q8_0_smallT<1, true, false, 2><<<(unsigned) ((N_out + 1) / 2), 64, 0, st>>>(w8, xu, Y, (int) K_in, (int) N_out, 1);
                    } else {
                        k_mmvq_q8_0_smallT<1, true, false, 4><<<(unsigned) ((N_out + 3) / 4), 128, 0, st>>>(w8, xu, Y, (int) K_in, (int) N_out, 1);
                    }
                } else {
                    const unsigned nb = (unsigned) ((N_out + 3) / 4);
                    if (T == 4) {
                        k_mmvq_q8_0_smallT<4, true, false><<<nb, 128, 0, st>>>(w8, xu, Y, (int) K_in, (int) N_out, 4);
                    } else if (T == 8) {
                        k_mmvq_q8_0_smallT<8, true, false><<<nb, 128, 0, st>>>(w8, xu, Y, (int) K_in, (int) N_out, 8);
                    } else if (T == 2) {
                        k_mmvq_q8_0_smallT<2, true, false><<<nb, 128, 0, st>>>(w8, xu, Y, (int) K_in, (int) N_out, 2);
                    } else if (T == 3) {
                        k_mmvq_q8_0_smallT<3, true, false><<<nb, 128, 0, st>>>(w8, xu, Y, (int) K_in, (int) N_out, 3);
                    } else {
                        k_mmvq_q8_0_smallT<8, false, false><<<nb, 128, 0, st>>>(w8, xu, Y, (int) K_in, (int) N_out, T);
                    }
                }
                return;
            }
            K::native_mmvq(W.type, W.d, xq_ptr, Y, (int) K_in, (int) N_out, T, st);
            return;
        }
        const uint16_t* xf16 = f16_ready;
        if (!xf16) {
            launch_f32_to_f16(X, act_bf16, (int64_t) T * K_in);
            xf16 = act_bf16;
        }
        gemm.native(xf16, W.type, W.d, Y, T, N_out, K_in);
    }

    void project(const float* X, const DenseMat& W, float* Y, int T, int64_t N_out, int64_t K_in) {
        project_preq(X, nullptr, W, Y, T, N_out, K_in);
    }

    bool try_project_pair_bf16(const float* X, const DenseMat& W1, float* Y1, int64_t N1,
                               const DenseMat& W2, float* Y2, int64_t N2, int T, int64_t K_in) {
        if (W1.type != 30 || W2.type != 30 || T < 2 || T > 8 || (K_in % 8) != 0) return false;
        const unsigned nb = (unsigned) ((N1 + N2 + 7) / 8);
        const uint16_t* w1 = (const uint16_t*) W1.d;
        const uint16_t* w2 = (const uint16_t*) W2.d;
        if (T == 4) {
            k_gemm_bf16_smallT_warp<4, true, true><<<nb, 256, 0, st>>>(w1, X, Y1, (int) K_in, (int) N1, 4, w2, Y2, (int) N2);
        } else if (T == 2) {
            k_gemm_bf16_smallT_warp<2, true, true><<<nb, 256, 0, st>>>(w1, X, Y1, (int) K_in, (int) N1, 2, w2, Y2, (int) N2);
        } else if (T == 3) {
            k_gemm_bf16_smallT_warp<3, true, true><<<nb, 256, 0, st>>>(w1, X, Y1, (int) K_in, (int) N1, 3, w2, Y2, (int) N2);
        } else {
            k_gemm_bf16_smallT_warp<8, false, true><<<nb, 256, 0, st>>>(w1, X, Y1, (int) K_in, (int) N1, T, w2, Y2, (int) N2);
        }
        return true;
    }


    bool try_project_pair_q8_0(const void* q8_ready, const DenseMat& W1, float* Y1, int64_t N1,
                               const DenseMat& W2, float* Y2, int64_t N2, int T, int64_t K_in) {
        if (!q8_ready || W1.type != 8 || W2.type != 8 || W1.q8_in_place || W2.q8_in_place || T < 1 || T > 8) return false;
        if ((K_in % 256) != 0 || (N1 & 3) != 0) return false;
        if (((((uintptr_t) W1.d) | ((uintptr_t) W2.d)) & 3u) != 0u) return false;
        const unsigned nb = (unsigned) ((N1 + N2 + 3) / 4);
        const uint8_t* w1 = (const uint8_t*) W1.d;
        const uint8_t* w2 = (const uint8_t*) W2.d;
        const uint32_t* xu = (const uint32_t*) q8_ready;
        if (T == 4) {
            k_mmvq_q8_0_smallT<4, true, true><<<nb, 128, 0, st>>>(w1, xu, Y1, (int) K_in, (int) N1, 4, w2, Y2, (int) N2);
        } else if (T == 1) {
            k_mmvq_q8_0_smallT<1, true, true><<<nb, 128, 0, st>>>(w1, xu, Y1, (int) K_in, (int) N1, 1, w2, Y2, (int) N2);
        } else if (T == 8) {
            k_mmvq_q8_0_smallT<8, true, true><<<nb, 128, 0, st>>>(w1, xu, Y1, (int) K_in, (int) N1, 8, w2, Y2, (int) N2);
        } else if (T == 2) {
            k_mmvq_q8_0_smallT<2, true, true><<<nb, 128, 0, st>>>(w1, xu, Y1, (int) K_in, (int) N1, 2, w2, Y2, (int) N2);
        } else if (T == 3) {
            k_mmvq_q8_0_smallT<3, true, true><<<nb, 128, 0, st>>>(w1, xu, Y1, (int) K_in, (int) N1, 3, w2, Y2, (int) N2);
        } else {
            k_mmvq_q8_0_smallT<8, false, true><<<nb, 128, 0, st>>>(w1, xu, Y1, (int) K_in, (int) N1, T, w2, Y2, (int) N2);
        }
        return true;
    }

    bool try_project_multi_q8_0(const void* q8_0, const void* q8_1, const void* q8_2,
                                const void* q8_3, const void* q8_4, int nmat,
                                const DenseMat& W0, float* Y0, int64_t N0,
                                const DenseMat& W1, float* Y1, int64_t N1,
                                const DenseMat& W2, float* Y2, int64_t N2,
                                const DenseMat* W3, float* Y3, int64_t N3,
                                const DenseMat* W4, float* Y4, int64_t N4,
                                int T, int64_t K_in) {
        if (!q8_0 || !q8_1 || !q8_2 || T < 1 || T > 8 || (K_in % 256) != 0) return false;
        if (W0.type != 8 || W1.type != 8 || W2.type != 8 || W0.q8_in_place || W1.q8_in_place || W2.q8_in_place) return false;
        if ((N0 & 3) != 0 || (N1 & 3) != 0) return false;
        if (((((uintptr_t) W0.d) | ((uintptr_t) W1.d) | ((uintptr_t) W2.d)) & 3u) != 0u) return false;
        int64_t tot_n = N0 + N1 + N2;
        if (nmat == 5) {
            if (!q8_3 || !q8_4 || !W3 || !W4 || W3->type != 8 || W4->type != 8 || W3->q8_in_place || W4->q8_in_place) return false;
            if ((N2 & 3) != 0 || (N3 & 3) != 0) return false;
            if (((((uintptr_t) W3->d) | ((uintptr_t) W4->d)) & 3u) != 0u) return false;
            tot_n += N3 + N4;
        }
        Q8MultiSpec sp{(const uint8_t*) W0.d, (const uint8_t*) W1.d, (const uint8_t*) W2.d,
                       W3 ? (const uint8_t*) W3->d : nullptr, W4 ? (const uint8_t*) W4->d : nullptr,
                       (const uint32_t*) q8_0, (const uint32_t*) q8_1, (const uint32_t*) q8_2,
                       (const uint32_t*) q8_3, (const uint32_t*) q8_4,
                       Y0, Y1, Y2, Y3, Y4, (int) N0, (int) N1, (int) N2, (int) N3, (int) N4};
        const unsigned nb = (unsigned) ((tot_n + 3) / 4);
        if (nmat == 3) {
            if (T == 4) k_mmvq_q8_0_smallT_multi<4, true, 3><<<nb, 128, 0, st>>>(sp, (int) K_in, 4);
            else if (T == 3) k_mmvq_q8_0_smallT_multi<3, true, 3><<<nb, 128, 0, st>>>(sp, (int) K_in, 3);
            else if (T == 2) k_mmvq_q8_0_smallT_multi<2, true, 3><<<nb, 128, 0, st>>>(sp, (int) K_in, 2);
            else if (T == 1) k_mmvq_q8_0_smallT_multi<1, true, 3><<<nb, 128, 0, st>>>(sp, (int) K_in, 1);
            else if (T == 8) k_mmvq_q8_0_smallT_multi<8, true, 3><<<nb, 128, 0, st>>>(sp, (int) K_in, 8);
            else k_mmvq_q8_0_smallT_multi<8, false, 3><<<nb, 128, 0, st>>>(sp, (int) K_in, T);
        } else {
            if (T == 4) k_mmvq_q8_0_smallT_multi<4, true, 5><<<nb, 128, 0, st>>>(sp, (int) K_in, 4);
            else if (T == 3) k_mmvq_q8_0_smallT_multi<3, true, 5><<<nb, 128, 0, st>>>(sp, (int) K_in, 3);
            else if (T == 2) k_mmvq_q8_0_smallT_multi<2, true, 5><<<nb, 128, 0, st>>>(sp, (int) K_in, 2);
            else if (T == 1) k_mmvq_q8_0_smallT_multi<1, true, 5><<<nb, 128, 0, st>>>(sp, (int) K_in, 1);
            else if (T == 8) k_mmvq_q8_0_smallT_multi<8, true, 5><<<nb, 128, 0, st>>>(sp, (int) K_in, 8);
            else k_mmvq_q8_0_smallT_multi<8, false, 5><<<nb, 128, 0, st>>>(sp, (int) K_in, T);
        }
        return true;
    }

    void project_f32(const float* X, const float* W, float* Y, int T, int64_t N_out, int64_t K_in) {
        if (T == 1 && (K_in % 4) == 0) {
            if (N_out <= 128) {
                k_gemv_f32_cta<<<(unsigned) N_out, 128, 0, st>>>(W, X, Y, (int) K_in, (int) N_out);
            } else {
                k_gemv_f32<<<(unsigned) ((N_out + 7) / 8), 256, 0, st>>>(W, X, Y, (int) K_in, (int) N_out);
            }
            return;
        }
        if (T >= 2 && T <= 8 && (K_in % 4) == 0) {
            if (N_out <= 128) {
                if (T == 4) {
                    k_gemm_f32_smallT_cta<4, true><<<(unsigned) N_out, 256, 0, st>>>(W, X, Y, (int) K_in, (int) N_out, 4);
                } else if (T == 2) {
                    k_gemm_f32_smallT_cta<2, true><<<(unsigned) N_out, 256, 0, st>>>(W, X, Y, (int) K_in, (int) N_out, 2);
                } else if (T == 3) {
                    k_gemm_f32_smallT_cta<3, true><<<(unsigned) N_out, 256, 0, st>>>(W, X, Y, (int) K_in, (int) N_out, 3);
                } else {
                    k_gemm_f32_smallT_cta<8, false><<<(unsigned) N_out, 256, 0, st>>>(W, X, Y, (int) K_in, (int) N_out, T);
                }
            } else {
                const unsigned nb = (unsigned) ((N_out + 7) / 8);
                if (T == 4) {
                    k_gemm_f32_smallT_warp<4, true><<<nb, 256, 0, st>>>(W, X, Y, (int) K_in, (int) N_out, 4);
                } else if (T == 2) {
                    k_gemm_f32_smallT_warp<2, true><<<nb, 256, 0, st>>>(W, X, Y, (int) K_in, (int) N_out, 2);
                } else if (T == 3) {
                    k_gemm_f32_smallT_warp<3, true><<<nb, 256, 0, st>>>(W, X, Y, (int) K_in, (int) N_out, 3);
                } else {
                    k_gemm_f32_smallT_warp<8, false><<<nb, 256, 0, st>>>(W, X, Y, (int) K_in, (int) N_out, T);
                }
            }
            return;
        }
        gemm.sgemm(X, W, Y, T, N_out, K_in);
    }

    void run_mix_normed(const std::vector<float*>& n_blocks, const float* c, bool c_already_normed, const float* query, float* const* outs, int n_q, int T, int swa_bcast, const MixInvRms4* ir_out = nullptr) {
        const auto& hp = m->hp;
        const int N = (int) hp.n_embd;
        const int B = (int) n_blocks.size();
        const int D = B + 1;
        const float inv_sqrt_n = 1.0f / std::sqrt((float) N);
        const bool fuse_last_norm = (!c_already_normed && T >= 1 && N == 2304 && D <= 8 && (n_q == 1 || n_q == 3));
        if (!c_already_normed && !fuse_last_norm) {
            k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(c, nullptr, norm_cur, N, hp.rms_eps);
        }
        NormMixPtrs np{};
        for (int i = 0; i < B; ++i) np.p[i] = n_blocks[(size_t) (B - 1 - i)];
        np.p[B] = (c_already_normed || fuse_last_norm) ? c : norm_cur;
        if (T >= 1 && (N & 3) == 0 && D <= 8 && (n_q == 1 || n_q == 3)) {
            MixOuts4 mo4{};
            for (int i = 0; i < (swa_bcast ? 3 : n_q); ++i) mo4.p[i] = outs[i];
            const MixInvRms4 ir4 = ir_out ? *ir_out : MixInvRms4{{nullptr, nullptr, nullptr, nullptr}};
            if (N == 2304) {
                if (fuse_last_norm) {
                    if (n_q == 1) {
                        k_mix_normed_fused_1pass<1, true><<<(unsigned) T, 576, 0, st>>>(np, D, query, nullptr, mo4, N, inv_sqrt_n, swa_bcast, hp.rms_eps, ir4);
                    } else {
                        k_mix_normed_fused_1pass<3, true><<<(unsigned) T, 576, 0, st>>>(np, D, query, nullptr, mo4, N, inv_sqrt_n, 0, hp.rms_eps, ir4);
                    }
                } else {
                    if (n_q == 1) {
                        k_mix_normed_fused_1pass<1, false><<<(unsigned) T, 576, 0, st>>>(np, D, query, nullptr, mo4, N, inv_sqrt_n, swa_bcast, hp.rms_eps, ir4);
                    } else {
                        k_mix_normed_fused_1pass<3, false><<<(unsigned) T, 576, 0, st>>>(np, D, query, nullptr, mo4, N, inv_sqrt_n, 0, hp.rms_eps, ir4);
                    }
                }
            } else if (n_q == 1) {
                k_mix_normed_fused<1><<<(unsigned) T, kThreads, 0, st>>>(np, D, query, nullptr, mo4, N, inv_sqrt_n, swa_bcast);
            } else {
                k_mix_normed_fused<3><<<(unsigned) T, kThreads, 0, st>>>(np, D, query, nullptr, mo4, N, inv_sqrt_n, 0);
            }
            return;
        }
        MixOuts mo{};
        for (int i = 0; i < (swa_bcast ? 3 : n_q); ++i) mo.p[i] = outs[i];
        k_mix_normed<<<dim3((unsigned) n_q, (unsigned) T), kThreads, 0, st>>>(np, D, query, mo, n_q, N, inv_sqrt_n, swa_bcast);
    }

    bool try_run_mix_normed_q8(const std::vector<float*>& n_blocks, const float* c, const float* query,
                               float* const* outs, const NormQ8Out3& nq8, int n_q, int T) {
        const auto& hp = m->hp;
        const int N = (int) hp.n_embd;
        const int B = (int) n_blocks.size();
        const int D = B + 1;
        if (T < 1 || N != 2304 || D > 8 || (n_q != 1 && n_q != 3)) return false;
        NormMixPtrs np{};
        for (int i = 0; i < B; ++i) np.p[i] = n_blocks[(size_t) (B - 1 - i)];
        np.p[B] = c;
        MixOuts4 mo4{};
        for (int i = 0; i < n_q; ++i) mo4.p[i] = outs[i];
        const float inv_sqrt_n = 1.0f / std::sqrt((float) N);
        if (n_q == 1) {
            k_mix_normed_fused_1pass_q8<1><<<(unsigned) T, 576, 0, st>>>(np, D, query, mo4, nq8, N, inv_sqrt_n, hp.rms_eps);
        } else {
            k_mix_normed_fused_1pass_q8<3><<<(unsigned) T, 576, 0, st>>>(np, D, query, mo4, nq8, N, inv_sqrt_n, hp.rms_eps);
        }
        return true;
    }

    bool try_run_mix_normed_f16(const std::vector<float*>& n_blocks, const float* c, const float* query,
                                float* const* outs, const NormF16Out3& nf16, int n_q, int T) {
        const auto& hp = m->hp;
        const int N = (int) hp.n_embd;
        const int B = (int) n_blocks.size();
        const int D = B + 1;
        if (T < 1 || N != 2304 || D > 8 || (n_q != 1 && n_q != 3)) return false;
        NormMixPtrs np{};
        for (int i = 0; i < B; ++i) np.p[i] = n_blocks[(size_t) (B - 1 - i)];
        np.p[B] = c;
        MixOuts4 mo4{};
        for (int i = 0; i < n_q; ++i) mo4.p[i] = outs[i];
        const float inv_sqrt_n = 1.0f / std::sqrt((float) N);
        if (n_q == 1) {
            k_mix_normed_fused_1pass_f16<1><<<(unsigned) T, 576, 0, st>>>(np, D, query, mo4, nf16, N, inv_sqrt_n, hp.rms_eps);
        } else {
            k_mix_normed_fused_1pass_f16<3><<<(unsigned) T, 576, 0, st>>>(np, D, query, mo4, nf16, N, inv_sqrt_n, hp.rms_eps);
        }
        return true;
    }

    void run_control_3_swa(const float* c_in, const float* ir_c,
                           const float* m0, const float* m1, const float* m2,
                           const float* irm0, const float* irm1, const float* irm2,
                           const PhysLayer& L, float* out0, float* out1, float* out2, int T) {
        const auto& hp = m->hp;
        const int N = (int) hp.n_embd;
        // ctrl_f (size T*N floats >> 3*T*64 floats) holds batched ctrl64[0..2] (zero extra VRAM, unused by WMMA up_combine!).
        float* c64_0 = ctrl_f;
        float* c64_1 = ctrl_f + (size_t) T * 64;
        float* c64_2 = ctrl_f + (size_t) T * 128;
        CtrlDownPtrs3 dp{
            {(const float*) L.loop_down[0].d, (const float*) L.loop_down[1].d, (const float*) L.loop_down[2].d},
            {m0, m1, m2},
            {irm0, irm1, irm2},
            {c64_0, c64_1, c64_2}
        };
        k_control_down_pp_wmma<3><<<(unsigned) ((T + 31) / 32), 256, 0, st>>>(c_in, ir_c, dp, T, N);
        // Write out1 (in_k) & out2 (in_v) BEFORE out0 (in_q == c_in) so c_in stays intact for all 3!
        const dim3 gup((unsigned) (N / 64), (unsigned) ((T + 63) / 64));
        k_control_up_combine_pp_wmma<<<gup, 256, 0, st>>>(c_in, m1, c64_1, (const float*) L.loop_gain[1].d, (const float*) L.loop_forget[1].d, out1, T, N);
        k_control_up_combine_pp_wmma<<<gup, 256, 0, st>>>(c_in, m2, c64_2, (const float*) L.loop_gain[2].d, (const float*) L.loop_forget[2].d, out2, T, N);
        k_control_up_combine_pp_wmma<<<gup, 256, 0, st>>>(c_in, m0, c64_0, (const float*) L.loop_gain[0].d, (const float*) L.loop_forget[0].d, out0, T, N);
    }

    void run_control(const float* c, const float* msg, const PhysLayer& L, int mod, float* out, int T,
                     const float* ir_c = nullptr, const float* ir_msg = nullptr) {
        const auto& hp = m->hp;
        const int N = (int) hp.n_embd;
        if (T >= 32 && (N & 3) == 0 && ctrl_w16 && L.loop_down[mod].type == 0 && L.loop_gain[mod].type == 0 && L.loop_forget[mod].type == 0) {
            // FP16 Tensor Core path for prefill loop controllers (zero extra VRAM: uses ctrl_w16 slice in layer_arena + act_bf16).
            const int64_t R = hp.loop_gate_rank;
            const int64_t n_wd = R * 2 * N;
            const int64_t n_wu = (int64_t) N * R;
            const int64_t n_x64 = (int64_t) T * R;
            if (R == 64 && (N % 64) == 0 && can_ctrl_wmma && can_ctrl_down_wmma && ir_c && ir_msg) {
                CtrlDownPtrs3 dp{{(const float*) L.loop_down[mod].d, nullptr, nullptr},
                                 {msg, nullptr, nullptr}, {ir_msg, nullptr, nullptr}, {ctrl64, nullptr, nullptr}};
                k_control_down_pp_wmma<1><<<(unsigned) ((T + 31) / 32), 256, 0, st>>>(c, ir_c, dp, T, N);
                k_control_up_combine_pp_wmma<<<dim3((unsigned) (N / 64), (unsigned) ((T + 63) / 64)), 256, 0, st>>>(
                    c, msg, ctrl64, (const float*) L.loop_gain[mod].d, (const float*) L.loop_forget[mod].d, out, T, N);
                return;
            }
            if ((size_t) (n_wd + 2 * n_wu) * sizeof(uint16_t) <= ctrl_w16_bytes) {
                uint16_t* w_down16 = ctrl_w16;
                k_control_prep_f16<<<dim3(2, (unsigned) T), (N == 2304) ? 576 : kThreads, 0, st>>>(c, msg, act_bf16, N, hp.rms_eps);
                if (R == 64 && (N % 64) == 0 && can_ctrl_wmma) {
                    // Convert ONLY W_down -> FP16, run loop_down GEMM, then fuse gain+forget+combine in 1 WMMA kernel!
                    launch_f32_to_f16((const float*) L.loop_down[mod].d, w_down16, n_wd);
                    gemm.f16(act_bf16, w_down16, ctrl64, T, R, 2 * N);
                    k_control_up_combine_pp_wmma<<<dim3((unsigned) (N / 64), (unsigned) ((T + 63) / 64)), 256, 0, st>>>(
                        c, msg, ctrl64, (const float*) L.loop_gain[mod].d, (const float*) L.loop_forget[mod].d, out, T, N);
                    return;
                }
                uint16_t* w_gain16 = w_down16 + n_wd;
                uint16_t* w_fgt16  = w_gain16 + n_wu;
                const int64_t tot_w4 = (n_wd + 2 * n_wu) >> 2;
                k_f32_to_f16_ctrl_w<<<(unsigned) ((tot_w4 + 255) / 256), 256, 0, st>>>(
                    (const float*) L.loop_down[mod].d, (const float*) L.loop_gain[mod].d,
                    (const float*) L.loop_forget[mod].d, w_down16, n_wd, n_wu);
                gemm.f16(act_bf16, w_down16, ctrl64, T, R, 2 * N);
                launch_f32_to_f16(ctrl64, act_bf16, n_x64);
                gemm.f16(act_bf16, w_gain16, ctrl_g, T, N, R);
                gemm.f16(act_bf16, w_fgt16, ctrl_f, T, N, R);
                const int64_t tot = (int64_t) T * N;
                const int64_t n4 = ((tot & 3) == 0) ? (tot >> 2) : tot;
                k_control_combine<<<(unsigned) ((n4 + 255) / 256), 256, 0, st>>>(c, msg, ctrl_g, ctrl_f, out, tot);
                return;
            }
        }
        if (T >= 1 && T <= 8 && L.loop_down[mod].type == 0 && (N & 3) == 0) {
            const float* wd = (const float*) L.loop_down[mod].d;
            const int R = (int) hp.loop_gate_rank;
            if (T == 4) {
                k_loop_down_fused_smallT<4, true><<<(unsigned) R, 256, 0, st>>>(wd, c, msg, ctrl64, N, R, 4, hp.rms_eps);
            } else if (T == 1) {
                k_loop_down_fused_smallT<1, true><<<(unsigned) R, 256, 0, st>>>(wd, c, msg, ctrl64, N, R, 1, hp.rms_eps);
            } else if (T == 2) {
                k_loop_down_fused_smallT<2, true><<<(unsigned) R, 256, 0, st>>>(wd, c, msg, ctrl64, N, R, 2, hp.rms_eps);
            } else if (T == 3) {
                k_loop_down_fused_smallT<3, true><<<(unsigned) R, 256, 0, st>>>(wd, c, msg, ctrl64, N, R, 3, hp.rms_eps);
            } else {
                k_loop_down_fused_smallT<8, false><<<(unsigned) R, 256, 0, st>>>(wd, c, msg, ctrl64, N, R, T, hp.rms_eps);
            }
        } else {
            k_control_prep<<<dim3(2, (unsigned) T), kThreads, 0, st>>>(c, msg, ctrl2N, N, hp.rms_eps);
            project(ctrl2N, L.loop_down[mod], ctrl64, T, hp.loop_gate_rank, 2 * N);
        }
        if (hp.loop_gate_rank == 64 && L.loop_gain[mod].type == 0 && L.loop_forget[mod].type == 0) {
            if (T == 1) {
                k_control_up_combine_t1<<<(unsigned) ((N + 7) / 8), 128, 0, st>>>(
                    c, msg, (const float*) L.loop_gain[mod].d, (const float*) L.loop_forget[mod].d,
                    ctrl64, out, (int) hp.loop_gate_rank, N);
                return;
            }
            if (T >= 2 && T <= 8) {
                const unsigned nb = (unsigned) ((N + 15) / 16);
                const float* wg = (const float*) L.loop_gain[mod].d;
                const float* wf = (const float*) L.loop_forget[mod].d;
                if (T == 4) {
                    k_control_up_combine_smallT<4, true><<<nb, 256, 0, st>>>(c, msg, wg, wf, ctrl64, out, 4, N);
                } else if (T == 2) {
                    k_control_up_combine_smallT<2, true><<<nb, 256, 0, st>>>(c, msg, wg, wf, ctrl64, out, 2, N);
                } else if (T == 3) {
                    k_control_up_combine_smallT<3, true><<<nb, 256, 0, st>>>(c, msg, wg, wf, ctrl64, out, 3, N);
                } else {
                    k_control_up_combine_smallT<8, false><<<nb, 256, 0, st>>>(c, msg, wg, wf, ctrl64, out, T, N);
                }
                return;
            }
        }
        project(ctrl64, L.loop_gain[mod], ctrl_g, T, N, hp.loop_gate_rank);
        project(ctrl64, L.loop_forget[mod], ctrl_f, T, N, hp.loop_gate_rank);
        const int64_t tot = (int64_t) T * N;
        const int64_t n4 = ((tot & 3) == 0) ? (tot >> 2) : tot;
        k_control_combine<<<(unsigned) ((n4 + 255) / 256), 256, 0, st>>>(c, msg, ctrl_g, ctrl_f, out, tot);
    }

    bool queue_sync_d_hits() {
        const auto& hp = m->hp;
        const int ne = (int) hp.n_expert;
        const int n_phys = (int) m->layers.size();
        if (d_hits_pool && h_hits_pin) {
            const size_t tot_bytes = (size_t) n_phys * (size_t) ne * sizeof(uint32_t);
            CK(cudaMemcpyAsync(h_hits_pin, d_hits_pool, tot_bytes, cudaMemcpyDeviceToHost, st));
            CK(cudaMemsetAsync(d_hits_pool, 0, tot_bytes, st));
            return true;
        }
        bool any = false;
        for (int ip = 0; ip < n_phys; ++ip) {
            PhysLayer& L = m->layers[(size_t) ip];
            if (!L.d_hits) continue;
            CK(cudaMemcpyAsync(h_hits_pin + (size_t) ip * ne, L.d_hits, (size_t) ne * sizeof(uint32_t), cudaMemcpyDeviceToHost, st));
            CK(cudaMemsetAsync(L.d_hits, 0, (size_t) ne * sizeof(uint32_t), st));
            any = true;
        }
        return any;
    }

    void finish_sync_d_hits(bool accumulate_hy = false) {
        const auto& hp = m->hp;
        const int ne = (int) hp.n_expert;
        const int n_phys = (int) m->layers.size();
        for (int ip = 0; ip < n_phys; ++ip) {
            PhysLayer& L = m->layers[(size_t) ip];
            if (!L.d_hits) continue;
            const uint32_t* tmp = h_hits_pin + (size_t) ip * ne;
            for (int e = 0; e < ne; ++e) {
                const uint32_t c = tmp[e];
                if (c == 0) continue;
                L.score[(size_t) e] += (double) c;
                if (accumulate_hy) {
                    if (L.slot_of[(size_t) e] >= 0) hy_gpu += c;
                    else hy_cpu += c;
                }
            }
        }
    }

    void sync_d_hits(bool accumulate_hy = false) {
        if (!queue_sync_d_hits()) return;
        CK(cudaStreamSynchronize(st));
        finish_sync_d_hits(accumulate_hy);
    }

    void pp_kick_next() {
        const int core_start_il = (int) m->hp.core_start;
        while (pp_kick_idx < pp_tasks.size() && !pp_free_slots.empty()) {
            PpTask& tk = pp_tasks[pp_kick_idx];
            const PhysLayer& L = m->layers[(size_t) tk.ip];
            const size_t need_b = (size_t) tk.n_staged * L.stride;
            // Pick the SMALLEST-CAPACITY free slot b with pp_buf_cap[b] >= need_b.
            size_t best_pos = pp_free_slots.size();
            size_t best_cap = ~size_t(0);
            for (size_t i = 0; i < pp_free_slots.size(); ++i) {
                const int cand = pp_free_slots[i];
                const size_t cap = pp_buf_cap[cand];
                if (cap >= need_b && cap < best_cap) {
                    best_cap = cap;
                    best_pos = i;
                }
            }
            if (best_pos >= pp_free_slots.size()) break; // Wait until a large-enough slot frees up
            // When kicking an Epilogue small task ahead of the Core gap into a large slot while Prelude
            // tasks still hold enough small slots to absorb all remaining small tasks, hold back so the
            // large slots stay reserved for upcoming large (Q5_K/Q8_0) Epilogue tasks!
            // Deadlock-free: requires pp_kick_idx > pp_take_idx and >= 1 already-kicked Prelude task holding a small slot.
            if (pp_kick_idx > pp_take_idx && best_cap >= pp_stage_bytes && need_b < pp_stage_bytes &&
                tk.il >= core_start_il) {
                int inflight_pre_small = 0;
                size_t max_small_cap = 0;
                for (size_t ti = pp_take_idx; ti < pp_kick_idx; ++ti) {
                    const PpTask& itk = pp_tasks[ti];
                    const int s = itk.stage_slot;
                    if (itk.il < core_start_il && s >= 0 && s < kMaxPpStages &&
                        pp_buf_cap[s] >= need_b && pp_buf_cap[s] < pp_stage_bytes) {
                        ++inflight_pre_small;
                        if (pp_buf_cap[s] > max_small_cap) max_small_cap = pp_buf_cap[s];
                    }
                }
                if (inflight_pre_small > 0) {
                    int rem_small = 0;
                    bool has_rem_large = false;
                    for (size_t ti = pp_kick_idx; ti < pp_tasks.size(); ++ti) {
                        const PpTask& rt = pp_tasks[ti];
                        const size_t rtb = (size_t) rt.n_staged * m->layers[(size_t) rt.ip].stride;
                        if (rtb <= max_small_cap) ++rem_small;
                        else has_rem_large = true;
                    }
                    if (has_rem_large && inflight_pre_small >= rem_small) break;
                }
            }
            const int b = pp_free_slots[best_pos];
            pp_free_slots.erase(pp_free_slots.begin() + (ptrdiff_t) best_pos);
            tk.stage_slot = b;
            if (pp_slot_needs_wait[b]) {
                CK(cudaStreamWaitEvent(cst, ev_pp_used[b], 0));
                pp_slot_needs_wait[b] = false;
            }
            if (pp_buf_owner_ip[b] >= 0 && pp_buf_owner_ip[b] < kMaxRebalLayers) {
                donor_clobbered[pp_buf_owner_ip[b]] = true;
            }
            for (const PpRun& r : tk.runs) {
                const size_t nbytes = (size_t) (r.cnt - 1) * L.stride + L.L.bytes;
                copy_h2d_staged(pp_buf_ptr[b] + (size_t) r.dst_off * L.stride,
                                L.hblobs + (size_t) r.e0 * L.stride,
                                nbytes, L.hblobs_pinned, cst);
            }
            CK(cudaEventRecord(ev_pp_copy[b], cst));
            ++pp_kick_idx;
        }
    }

    // EX-5 Idea B: Prepare all pp_tasks & permanent slots before CPU embedding dequant (zero initial ev_pp_used wait!).
    void pp_prepare() {
        const auto& hp = m->hp;
        const int ne = (int) hp.n_expert;
        pp_tasks.clear();
        std::fill(pp_layer_cnt.begin(), pp_layer_cnt.end(), 0);
        pp_take_idx = 0;
        pp_kick_idx = 0;
        pp_free_slots.clear();
        pp_active_slots = 0;
        for (int b = 0; b < kMaxPpStages; ++b) {
            pp_buf_ptr[b] = nullptr;
            pp_buf_cap[b] = 0;
            pp_buf_owner_ip[b] = -1;
            pp_slot_needs_wait[b] = false;
        }
        for (int i = 0; i < kMaxRebalLayers; ++i) donor_clobbered[i] = false;
        core_borrowed_cnt = 0;
        if (pp_stage_cnt <= 0 || !pp_stage[0] || pp_stage_bytes == 0) return;
        for (int b = 0; b < pp_stage_cnt; ++b) {
            pp_buf_ptr[b] = pp_stage[b];
            pp_buf_cap[b] = pp_stage_bytes;
            pp_buf_owner_ip[b] = -1;
            pp_slot_needs_wait[b] = false; // previous forward() synced st -> permanent slots are immediately idle
            pp_free_slots.push_back(b);
        }
        pp_active_slots = pp_stage_cnt;
        int missing[128];
        for (int il = 0; il < hp.n_layer; ++il) {
            const int ip = hp.loop_layer(il);
            const PhysLayer& L = m->layers[(size_t) ip];
            if (L.n_slots >= ne || !L.hblobs) continue;
            const int cap = (int) std::min<size_t>((size_t) ne, pp_stage_bytes / L.stride);
            if (cap <= 0) continue;
            int n_miss = 0;
            for (int x = 0; x < ne; ++x) {
                if (L.slot_of[(size_t) x] < 0) missing[n_miss++] = x;
            }
            // Balanced splitting across n_chunks eliminates 1-2 expert tail waves (e.g. 43 -> 15,14,14 instead of 14,14,14,1).
            const int n_chunks = (n_miss + cap - 1) / cap;
            for (int ci = 0, k0 = 0; ci < n_chunks; ++ci) {
                const int sz = (n_miss - k0 + (n_chunks - ci) - 1) / (n_chunks - ci);
                const int k1 = k0 + sz;
                PpTask tk;
                tk.ip = ip;
                tk.il = il;
                tk.n_staged = sz;
                tk.stage_slot = -1;
                tk.kicked_exps = 0;
                std::fill(std::begin(tk.map.pos), std::end(tk.map.pos), int16_t(-1));
                for (int i = k0; i < k1; ++i) {
                    const int x = missing[i];
                    const int dst = i - k0;
                    tk.map.pos[x] = (int16_t) dst;
                    if (!tk.runs.empty() && (int) tk.runs.back().e0 + (int) tk.runs.back().cnt == x) {
                        ++tk.runs.back().cnt;
                    } else {
                        tk.runs.push_back(PpRun{(int16_t) x, 1, (int16_t) dst});
                    }
                }
                pp_tasks.push_back(std::move(tk));
                ++pp_layer_cnt[(size_t) il];
                k0 = k1;
            }
        }
    }

    // EX-5 Idea B: Kick only the first ~48 MiB (<= 10 experts) of pp_tasks[0] onto cst BEFORE CPU embedding dequant,
    // overlapping DMA with 8-thread embedding work without head-of-line blocking ffn_inp H2D on st!
    void pp_kick_partial_first(size_t budget_bytes = ((size_t) 48 << 20)) {
        if (pp_tasks.empty() || pp_free_slots.empty() || pp_kick_idx != 0) return;
        PpTask& tk = pp_tasks[0];
        const PhysLayer& L = m->layers[(size_t) tk.ip];
        const size_t need_b = (size_t) tk.n_staged * L.stride;
        size_t best_pos = pp_free_slots.size();
        size_t best_cap = ~size_t(0);
        for (size_t i = 0; i < pp_free_slots.size(); ++i) {
            const int cand = pp_free_slots[i];
            const size_t cap = pp_buf_cap[cand];
            if (cap >= need_b && cap < best_cap) {
                best_cap = cap;
                best_pos = i;
            }
        }
        if (best_pos >= pp_free_slots.size()) return;
        const int b = pp_free_slots[best_pos];
        pp_free_slots.erase(pp_free_slots.begin() + (ptrdiff_t) best_pos);
        tk.stage_slot = b;
        if (pp_slot_needs_wait[b]) {
            CK(cudaStreamWaitEvent(cst, ev_pp_used[b], 0));
            pp_slot_needs_wait[b] = false;
        }
        if (pp_buf_owner_ip[b] >= 0 && pp_buf_owner_ip[b] < kMaxRebalLayers) {
            donor_clobbered[pp_buf_owner_ip[b]] = true;
        }
        const int max_exps = std::clamp((int) (budget_bytes / std::max<size_t>(1, L.stride)), 1, 10);
        const int split_exps = std::min(max_exps, tk.n_staged);
        int sent = 0;
        for (const PpRun& r : tk.runs) {
            if (sent >= split_exps) break;
            const int take = std::min<int>((int) r.cnt, split_exps - sent);
            const size_t nbytes = (size_t) (take - 1) * L.stride + L.L.bytes;
            copy_h2d_staged(pp_buf_ptr[b] + (size_t) r.dst_off * L.stride,
                            L.hblobs + (size_t) r.e0 * L.stride,
                            nbytes, L.hblobs_pinned, cst);
            sent += take;
        }
        tk.kicked_exps = sent;
        if (sent >= tk.n_staged) {
            CK(cudaEventRecord(ev_pp_copy[b], cst));
            ++pp_kick_idx;
        }
    }

    // EX-5 Idea B: Right after pos0_d H2D on st, finish remainder of pp_tasks[0] and fill remaining staging slots!
    void pp_kick_finish_and_fill() {
        if (pp_kick_idx == 0 && !pp_tasks.empty() && pp_tasks[0].stage_slot >= 0) {
            PpTask& tk = pp_tasks[0];
            const int b = tk.stage_slot;
            const PhysLayer& L = m->layers[(size_t) tk.ip];
            int skipped = 0;
            for (const PpRun& r : tk.runs) {
                const int r_end = skipped + (int) r.cnt;
                if (r_end <= tk.kicked_exps) {
                    skipped = r_end;
                    continue;
                }
                const int off_in_run = (tk.kicked_exps > skipped) ? (tk.kicked_exps - skipped) : 0;
                const int rem_cnt = (int) r.cnt - off_in_run;
                const size_t nbytes = (size_t) (rem_cnt - 1) * L.stride + L.L.bytes;
                copy_h2d_staged(pp_buf_ptr[b] + (size_t) ((int) r.dst_off + off_in_run) * L.stride,
                                L.hblobs + (size_t) ((int) r.e0 + off_in_run) * L.stride,
                                nbytes, L.hblobs_pinned, cst);
                skipped = r_end;
            }
            tk.kicked_exps = tk.n_staged;
            CK(cudaEventRecord(ev_pp_copy[b], cst));
            ++pp_kick_idx;
        }
        pp_kick_next();
    }

    // Wave 8.5 + Wave 9: Epilogue PCIe Pre-Buffering — opportunistically expand staging slots (+1..5 x stage_bytes)
    // from stream-ordered phase_pool during Prefill (e.g. using Shared-Scratch SWA / DSpark evicted decode_w_gpu / Tier-2 pool slack),
    // and free back to phase_pool right after Prefill finishes (ZERO permanent VRAM cost, 262K-safe!).
    void maybe_alloc_transient_pp_stages(size_t avail_known_min = 0) {
        static const bool no_transient_pp = (std::getenv("STRATA_NO_TRANSIENT_PP_STAGE") != nullptr);
        if (no_transient_pp || !use_phase_pool || !phase_pool || !is_core_full_resident()) return;
        if (pp_stage_cnt < 2 || pp_stage_bytes < ((size_t) 64 << 20)) return;
        if (transient_pp_pool != nullptr || pp_tasks.size() <= (size_t) pp_stage_cnt) return;
        const size_t stride_b = (pp_stage_bytes + 255u) & ~size_t(255u);
        const size_t pool_free = pool_cached_free_bytes();
        size_t avail = std::max(avail_known_min, pool_free);
        if (avail_known_min == 0) {
            size_t fr = 0, tot = 0;
            CK(cudaMemGetInfo(&fr, &tot));
            avail = std::max(avail, pool_free + fr);
        }
        const size_t guard = vram_reserve_bytes + ((size_t) 16 << 20);
        const int max_extra = std::min({10, kMaxPpStages - pp_active_slots, (int) pp_tasks.size() - pp_stage_cnt});
        for (int extra = max_extra; extra >= 1; --extra) {
            if (pp_active_slots + extra > kMaxPpStages) continue;
            const size_t need = (size_t) extra * stride_b;
            if (avail < need + guard) continue;
            uint8_t* p = nullptr;
            if (pool_alloc((void**) &p, need) == cudaSuccess && p != nullptr) {
                transient_pp_pool = p;
                transient_pp_cnt = extra;
                for (int i = 0; i < extra; ++i) {
                    const int b = pp_active_slots++;
                    pp_buf_ptr[b] = p + (size_t) i * stride_b;
                    pp_buf_cap[b] = pp_stage_bytes;
                    pp_buf_owner_ip[b] = -1;
                    // Order stream-ordered pool_alloc on st before cst H2D!
                    CK(cudaEventRecord(ev_pp_used[b], st));
                    pp_slot_needs_wait[b] = true;
                    pp_free_slots.push_back(b);
                }
                return;
            }
            (void) cudaGetLastError();
        }
    }

    void free_transient_pp_stages() {
        if (!transient_pp_pool) return;
        if (cst) CK(cudaStreamSynchronize(cst));
        pool_free(transient_pp_pool);
        transient_pp_pool = nullptr;
        transient_pp_cnt = 0;
    }

    // Wave 8.5 Cut A: Split-K Phase-A Control-Down + conditional reduction/pre_out add.
    template <int M_MODS>
    void launch_phaseA_splitk(CtrlDownPhaseAPtrs3& p, int rows, int N, float* part_base, size_t part_cap_floats) {
        int split_k = 1;
        if (N >= 64 && (N % 64) == 0) {
            const int k_tiles = N / 64;
            if (rows <= 512 && (k_tiles % 12) == 0) split_k = 12;
            else if (rows <= 768 && (k_tiles % 9) == 0) split_k = 9;
            else if (rows <= 1024 && (k_tiles % 6) == 0) split_k = 6;
            else if (rows <= 1536 && (k_tiles % 4) == 0) split_k = 4;
        }
        if (split_k > 1 && (!part_base || (size_t) M_MODS * (size_t) split_k * (size_t) rows * 64u > part_cap_floats)) {
            split_k = 1;
        }
        for (int m = 0; m < M_MODS; ++m) {
            p.part_out[m] = (split_k > 1) ? (part_base + (size_t) m * (size_t) split_k * (size_t) rows * 64u) : nullptr;
        }
        const bool has_inv_c = (p.inv_c[0] != nullptr);
        if (has_inv_c) {
            k_control_down_phaseA_pp_wmma_splitk<M_MODS, true><<<dim3((unsigned) ((rows + 63) / 64), (unsigned) split_k), 256, 0, st>>>(
                p, rows, N, split_k);
        } else {
            k_control_down_phaseA_pp_wmma_splitk<M_MODS, false><<<dim3((unsigned) ((rows + 63) / 64), (unsigned) split_k), 256, 0, st>>>(
                p, rows, N, split_k);
        }
        if (split_k > 1) {
            k_phaseA_splitk_reduce_add_pre<M_MODS><<<(unsigned) ((rows * 16 + 255) / 256), 256, 0, st>>>(
                p, rows, split_k);
        }
    }

    // Wave 8.5 Cut D + Wave 11 Path C1: 22-bit FP32-exact Ozaki-2 TC Router (sm_80+, 24 KB static SMEM, no carveout needed)
    // with opt-out STRATA_ROUTER_FP32_EXACT=1 -> scalar Split-K strict IEEE-754 FP32 Router SGEMM (uses disjoint ffn_splitk_scratch).
    void launch_router_gemm_pp(const float* x_ptr, const float* w_ptr, float* y_ptr, int rows, int n_expert, int N) {
        static const bool no_custom_router = [] {
            const char* v = std::getenv("STRATA_NO_CUSTOM_ROUTER");
            return v != nullptr && (v[0] == '\0' || std::atoi(v) != 0);
        }();
        static const bool router_fp32_exact = [] {
            const char* v = std::getenv("STRATA_ROUTER_FP32_EXACT");
            return v != nullptr && (v[0] == '\0' || std::atoi(v) != 0);
        }();
        if (!no_custom_router && !router_fp32_exact && cc_maj >= 8 &&
            rows >= 64 && (n_expert % 64) == 0 && (N % 32) == 0) {
            int sk = 1;
            if (rows <= 1024 && N >= 64 &&
                ffn_splitk_scratch != nullptr &&
                (size_t) 2 * (size_t) rows * (size_t) n_expert <= ffn_splitk_scratch_floats) {
                sk = 2;
            }
            const int k_per_split = (sk > 1) ? (((N / sk + 31) / 32) * 32) : N;
            float* y_dst = (sk > 1) ? ffn_splitk_scratch : y_ptr;
            k_router_ozaki2_tc_pp_splitk<<<dim3((unsigned) ((n_expert + 63) / 64), (unsigned) ((rows + 31) / 32), (unsigned) sk), 128, 0, st>>>(
                x_ptr, w_ptr, y_dst, rows, n_expert, N, k_per_split);
            if (sk > 1) {
                const int tot = rows * n_expert;
                k_router_splitk_reduce<<<(unsigned) (((tot + 3) / 4 + 255) / 256), 256, 0, st>>>(y_dst, y_ptr, tot, sk);
            }
            return;
        }
        if (!no_custom_router && rows >= 32 && (N % 16) == 0) {
            const int tiles_mn = ((n_expert + 127) / 128) * ((rows + 63) / 64);
            const int k16 = N / 16;
            int sk = 1;
            static constexpr int kCandidates[] = {8, 6, 4, 3, 2, 1};
            for (int cand : kCandidates) {
                if (cand == 1) { sk = 1; break; }
                if ((k16 % cand) == 0 && tiles_mn * cand <= 160 &&
                    ffn_splitk_scratch != nullptr &&
                    (size_t) cand * (size_t) rows * (size_t) n_expert <= ffn_splitk_scratch_floats) {
                    sk = cand;
                    break;
                }
            }
            float* y_dst = (sk > 1) ? ffn_splitk_scratch : y_ptr;
            k_router_sgemm_f32_pp_splitk<<<dim3((unsigned) ((n_expert + 127) / 128), (unsigned) ((rows + 63) / 64), (unsigned) sk), 128, 0, st>>>(
                x_ptr, w_ptr, y_dst, rows, n_expert, N, sk);
            if (sk > 1) {
                const int tot = rows * n_expert;
                k_router_splitk_reduce<<<(unsigned) (((tot + 3) / 4 + 255) / 256), 256, 0, st>>>(y_dst, y_ptr, tot, sk);
            }
        } else {
            project_f32(x_ptr, w_ptr, y_ptr, rows, n_expert, N);
        }
    }

    struct PostConvSpec {
        const float* w_conv = nullptr;
        float* hist = nullptr;
        const float* res_add = nullptr;
        float* out = nullptr;
        int hist_offset = 0;
        float* hist_trace = nullptr;
        size_t hist_trace_stride = 0;
    };

    static bool env_no_fuse_moe_zero() {
        static const bool v = []() {
            const char* s = std::getenv("STRATA_NO_FUSE_MOE_ZERO");
            return s != nullptr && (s[0] == '\0' || std::atoi(s) != 0);
        }();
        return v;
    }

    bool run_moe(PhysLayer& L, int il, const float* x_norm, int T, bool pre_routed_t1 = false,
                 bool xq_ready_in = false, bool skip_combine = false, bool f16_ready_in = false,
                 const PostConvSpec* post_conv = nullptr, bool pre_routed_pp = false,
                 bool capturing_graph = false) {
        const auto& hp = m->hp;
        const int N = (int) hp.n_embd, Kp = (int) hp.n_expert_used;
        const int n_ent = T * Kp;
        const bool has_gpu = L.n_slots > 0;
        const bool need_cpu = L.n_slots < hp.n_expert;
        const bool is_q5 = (L.L.gu_type == 13);
        const bool can_fused_router_t1 = (T == 1 && !pp_streaming && hp.n_expert <= 128 && Kp <= 16 && (N & 3) == 0 && d_router_done != nullptr);
        const bool pre_grouped_t1 = (pre_routed_t1 && T == 1 && has_gpu && hp.n_expert <= 128 && Kp <= 16 && (N & 3) == 0 && d_router_done != nullptr);
        const bool fused_rg = (has_gpu && !pre_routed_t1 && !pp_streaming && T >= 1 && T <= 8 &&
                               hp.n_expert <= 128 && Kp <= 16 && (!need_cpu || T == 1));

        const auto tp_moe_r0 = Clock::now();
        if (!pre_routed_t1) {
            if (need_cpu && T >= 1 && T <= std::max(verify_max_t, 8) && !pp_streaming) {
                if (T > cpu_buf_cap) init_cpu_bufs(T);
                CK(cudaMemcpyAsync(h_h, x_norm, (size_t) T * (size_t) N * sizeof(float), cudaMemcpyDeviceToHost, st));
            }
            if (can_fused_router_t1) {
                k_router_gemv_and_group_t1<<<(unsigned) ((hp.n_expert + 1) / 2), 128, 0, st>>>(
                    L.ffn_gate_inp, x_norm, router_logits, L.exp_probs_b, ids, ewts,
                    has_gpu ? L.d_slot_of : nullptr, has_gpu ? L.d_blobs : nullptr, L.stride,
                    N, (int) hp.n_expert, Kp, hp.expert_weights_scale,
                    grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                    (profile.use_d_hits || hybrid || full_kv_cap < ctx) ? L.d_hits : nullptr, d_router_done);
            } else {
                if (!pre_routed_pp) {
                    launch_router_gemm_pp(x_norm, L.ffn_gate_inp, router_logits, T, (int) hp.n_expert, N);
                }
                if (fused_rg) {
                    if (T == 1) {
                        k_route_and_group_t1_warp<<<1, 32, 0, st>>>(
                            router_logits, L.exp_probs_b, ids, ewts, L.d_slot_of, L.d_blobs, L.stride,
                            (int) hp.n_expert, Kp, hp.expert_weights_scale,
                            grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                            (profile.use_d_hits || hybrid || full_kv_cap < ctx) ? L.d_hits : nullptr);
                    } else {
                        const int blk = (T <= 4) ? 128 : 256;
                        k_route_and_group_smallT<<<1, blk, 0, st>>>(
                            router_logits, L.exp_probs_b, ids, ewts, L.d_slot_of, L.d_blobs, L.stride,
                            T, (int) hp.n_expert, Kp, hp.expert_weights_scale,
                            grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                            (profile.use_d_hits || hybrid || full_kv_cap < ctx) ? L.d_hits : nullptr);
                    }
                } else {
                    k_route_inkling2<<<(unsigned) ((T + 7) / 8), 256, 0, st>>>(
                        router_logits, L.exp_probs_b, ids, ewts, (int) hp.n_expert, Kp, hp.expert_weights_scale, T,
                        (profile.use_d_hits || hybrid || full_kv_cap < ctx) ? L.d_hits : nullptr);
                }
            }
            if (prof_detail) { CK(cudaStreamSynchronize(st)); p_d_route += ms_since(tp_moe_r0); }
        }

        const int ip_phys = hp.loop_layer(il);
        if (has_gpu && ip_phys < kMaxRebalLayers && layer_pending_rebal[ip_phys]) {
            CK(cudaStreamWaitEvent(st, ev_rebal_layer[ip_phys], 0));
            layer_pending_rebal[ip_phys] = false;
        }

        bool q8_ready = xq_ready_in;
        auto ensure_q8 = [&]() {
            if (!q8_ready) {
                K::quantize_q8_1_rows(x_norm, T, N, xq, st);
                q8_ready = true;
            }
        };
        bool wmma_ready = f16_ready_in;
        auto ensure_f16 = [&]() {
            if (!wmma_ready) {
                launch_f32_to_f16(x_norm, act_bf16, (int64_t) T * N);
                wmma_ready = true;
            }
        };
        static const bool no_moe_atomic = (std::getenv("STRATA_NO_MOE_ATOMIC") != nullptr);
        static const bool no_group16 = (std::getenv("STRATA_NO_GROUP16") != nullptr);
        const bool try_wmma = (T >= 16);
        const bool use_pp_stream = (pp_streaming && need_cpu && L.hblobs && pp_stage_bytes >= L.stride && pp_layer_cnt[(size_t) il] > 0);
        const bool use_atomic_moe = (!no_moe_atomic && cc_maj >= 8 && T >= 32 && (N & 3) == 0 && !skip_combine &&
                                     L.gu_type != 30 && L.d_type != 30 && K::native_expert_wmma_supported(L.L) &&
                                     (!need_cpu || use_pp_stream));
        const bool fuse_moe_zero = (!env_no_fuse_moe_zero() && use_atomic_moe && !capturing_graph && cc_maj >= 8 && try_wmma);
        const bool use_group16 = (!no_group16 && T >= 256 && hp.n_expert <= 128 && d_grp_part_cnt != nullptr);
        // Safety drain: if an async moe_raw zero was queued on aux_st but this layer does not use atomic MoE, wait now!
        if (moe_raw_zero_pending && !use_atomic_moe) {
            CK(cudaStreamWaitEvent(st, ev_moe_zeroed, 0));
            moe_raw_zero_pending = false;
        }
        bool moe_raw_ready = false;
        auto ensure_moe_raw_zeroed = [&]() {
            if (!use_atomic_moe || moe_raw_ready) return;
            if (moe_raw_zero_pending) {
                CK(cudaStreamWaitEvent(st, ev_moe_zeroed, 0));
                moe_raw_zero_pending = false;
            } else {
                CK(cudaMemsetAsync(moe_raw, 0, (size_t) T * (size_t) N * sizeof(float), st));
            }
            moe_raw_ready = true;
        };
        struct L2PersistGuard {
            cudaStream_t stream = nullptr;
            bool active = false;
            void set(void* base_ptr, size_t num_bytes, float hit_ratio,
                     cudaAccessProperty hit_prop, cudaAccessProperty miss_prop) {
                if (!stream || !base_ptr || num_bytes == 0) return;
                cudaStreamAttrValue attr{};
                attr.accessPolicyWindow.base_ptr = base_ptr;
                attr.accessPolicyWindow.num_bytes = num_bytes;
                attr.accessPolicyWindow.hitRatio = hit_ratio;
                attr.accessPolicyWindow.hitProp = hit_prop;
                attr.accessPolicyWindow.missProp = miss_prop;
                if (cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr) == cudaSuccess) {
                    active = true;
                } else {
                    (void) cudaGetLastError();
                }
            }
            void reset() {
                if (!active || !stream) return;
                cudaStreamAttrValue attr{};
                attr.accessPolicyWindow.num_bytes = 0;
                (void) cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
                (void) cudaGetLastError();
                active = false;
            }
            ~L2PersistGuard() { reset(); }
        } l2_guard{st, false};
        const bool use_l2_persist_moe = (T >= 2048 && cc_maj >= 8 &&
                                         l2_persist_bytes > 0 && l2_max_window_bytes > 0);
        auto run_grouped = [&](const unsigned long long* gptr, int64_t cap_g, int64_t est_ent = 0) {
            if (try_wmma) {
                ensure_f16();
                float* z_ptr = nullptr;
                size_t z_floats = 0;
                if (use_atomic_moe && !moe_raw_ready) {
                    if (fuse_moe_zero && !moe_raw_zero_pending) {
                        z_ptr = moe_raw;
                        z_floats = (size_t) T * (size_t) N;
                    } else {
                        ensure_moe_raw_zeroed();
                    }
                }
                if (use_l2_persist_moe && !l2_guard.active) {
                    if (use_atomic_moe) {
                        const size_t moe_b = (size_t) T * (size_t) N * sizeof(float);
                        const size_t cap_b = std::min<size_t>({l2_persist_bytes, l2_max_window_bytes, (size_t) 28u * 1024u * 1024u});
                        const size_t win_b = std::min(moe_b, cap_b);
                        l2_guard.set(moe_raw, win_b, 1.0f, cudaAccessPropertyPersisting, cudaAccessPropertyNormal);
                    } else {
                        const size_t act_b = (size_t) T * (size_t) N * sizeof(uint16_t);
                        const size_t cap_b = std::min<size_t>({l2_persist_bytes, l2_max_window_bytes, (size_t) 32u * 1024u * 1024u});
                        const size_t win_b = std::min(act_b, cap_b);
                        l2_guard.set(act_bf16, win_b, 1.0f, cudaAccessPropertyPersisting, cudaAccessPropertyStreaming);
                    }
                }
                if (K::native_expert_grouped_wmma(L.L, gptr, grp_start, n_groups, ent_dst, ent_tok,
                                                  cap_g, n_ent, act_bf16, escr,
                                                  use_atomic_moe ? moe_raw : eparts, st, est_ent,
                                                  use_atomic_moe ? ewts : nullptr,
                                                  z_ptr, z_floats)) {
                    if (z_ptr != nullptr) moe_raw_ready = true;
                    return;
                }
            }
            if (L.gu_type == 30 && L.d_type == 30) {
                K::native_expert_grouped_bf16_f32(L.L, gptr, grp_start, n_groups, ent_dst, ent_tok,
                                                  cap_g, n_ent, x_norm, escr, eparts, st);
                return;
            }
            ensure_q8();
            K::native_expert_grouped(L.L, gptr, grp_start, n_groups, ent_dst, ent_tok,
                                     cap_g, n_ent, xq, escr, eparts, st);
        };

        if (use_pp_stream) {
            // Zero host-GPU stalls across all 48 layers: k_route_inkling2 tracks exact routing counts in L.d_hits on GPU!
            const int ne = (int) hp.n_expert;
            const int n_chunks = pp_layer_cnt[(size_t) il];
            if (use_group16 && (has_gpu || n_chunks > 0)) {
                k_group_experts_hist16<<<16, 256, 0, st>>>(ids, n_ent, ne, d_grp_part_cnt);
            }
            auto run_gpu_res = [&]() {
                if (!has_gpu) return;
                if (use_group16) {
                    k_group_experts_prefix16<<<1, 128, 0, st>>>(d_grp_part_cnt, d_grp_part_off, 16, L.d_slot_of, L.d_blobs, L.stride,
                                                                ne, L.n_slots, grp_ptr, grp_start, n_groups);
                    k_group_experts_scatter16<<<16, 256, 0, st>>>(ids, d_grp_part_off, n_ent, ne, ent_dst, ent_tok, Kp);
                } else {
                    k_group_experts<<<1, 512, 0, st>>>(ids, L.d_slot_of, L.d_blobs, L.stride, n_ent, ne, L.n_slots,
                                                       grp_ptr, grp_start, n_groups, ent_dst, ent_tok, Kp);
                }
                const int64_t est_gpu = ((int64_t) n_ent * L.n_slots + ne - 1) / ne;
                run_grouped(grp_ptr, std::min<int64_t>(n_ent, L.n_slots), est_gpu);
            };
            auto run_staged = [&]() {
                for (int ci = 0; ci < n_chunks; ++ci) {
                    if (pp_take_idx >= pp_tasks.size()) {
                        std::fprintf(stderr, "strata-inkling2: pp stream underflow\n");
                        std::exit(1);
                    }
                    if (pp_tasks[pp_take_idx].stage_slot < 0) pp_kick_next();
                    const PpTask& tk = pp_tasks[pp_take_idx];
                    const int b = tk.stage_slot;
                    if (b < 0 || b >= kMaxPpStages || !pp_buf_ptr[b]) {
                        std::fprintf(stderr, "strata-inkling2: pp stage slot unassigned (idx=%zu)\n", pp_take_idx);
                        std::exit(1);
                    }
                    // k_group_experts_miss_map computes pointer arithmetic only (never dereferences pp_buf_ptr[b]);
                    // running it before waiting on ev_pp_copy[b] hides grouping latency behind PCIe DMA!
                    if (use_group16) {
                        k_group_experts_prefix16_miss_map<<<1, 128, 0, st>>>(d_grp_part_cnt, d_grp_part_off, 16, pp_buf_ptr[b], L.stride,
                                                                             ne, tk.map, grp_ptr, grp_start, n_groups);
                        k_group_experts_scatter16<<<16, 256, 0, st>>>(ids, d_grp_part_off, n_ent, ne, ent_dst, ent_tok, Kp);
                    } else {
                        k_group_experts_miss_map<<<1, 512, 0, st>>>(ids, pp_buf_ptr[b], L.stride, n_ent, ne, tk.map,
                                                                    grp_ptr, grp_start, n_groups, ent_dst, ent_tok, Kp);
                    }
                    CK(cudaStreamWaitEvent(st, ev_pp_copy[b], 0));
                    const int64_t est_st = ((int64_t) n_ent * tk.n_staged + ne - 1) / ne;
                    run_grouped(grp_ptr, std::min<int64_t>(n_ent, (int64_t) tk.n_staged), est_st);
                    CK(cudaEventRecord(ev_pp_used[b], st));
                    pp_slot_needs_wait[b] = true;
                    pp_free_slots.push_back(b);
                    ++pp_take_idx;
                    pp_kick_next();
                }
            };
            // On il > 0 with both resident & staged experts, drain staged buffers FIRST so pp_stage[0] is released
            // immediately for cst to keep streaming while run_gpu_res() overlaps with next layer's PCIe H2D!
            if (il > 0 && has_gpu && n_chunks > 0) {
                run_staged();
                run_gpu_res();
            } else {
                run_gpu_res();
                run_staged();
            }
            l2_guard.reset();
            if (use_atomic_moe) {
                ensure_moe_raw_zeroed();
                if (post_conv && post_conv->w_conv && T >= 32 && (N & 3) == 0 && !post_conv->hist_trace) {
                    const int n4 = N >> 2;
                    k_shortconv_pp_f4<<<dim3((unsigned) ((n4 + 63) / 64), (unsigned) ((T + 63) / 64)), 64, 0, st>>>(
                        moe_raw, post_conv->w_conv, post_conv->hist, post_conv->out, N, post_conv->hist_offset, T, post_conv->res_add);
                    k_shortconv_hist_push<<<(unsigned) ((N + 255) / 256), 256, 0, st>>>(
                        moe_raw, post_conv->hist, N, post_conv->hist_offset, T);
                    if (prof) { CK(cudaStreamSynchronize(st)); p_moe_gpu_ms += ms_since(tp_moe_r0); }
                    return true;
                }
                if (prof) { CK(cudaStreamSynchronize(st)); p_moe_gpu_ms += ms_since(tp_moe_r0); }
                return false;
            }
            if (post_conv && post_conv->w_conv && T >= 32 && (N & 3) == 0 && !post_conv->hist_trace) {
                const int n4 = N >> 2;
                const unsigned cb = ((n4 % 192) == 0) ? 192u : 64u;
                k_combine_experts_shortconv_pp<<<dim3((unsigned) ((n4 + (int) cb - 1) / (int) cb), (unsigned) ((T + 127) / 128)), cb, 0, st>>>(
                    eparts, ewts, post_conv->w_conv, post_conv->hist, post_conv->res_add, post_conv->out,
                    N, Kp, post_conv->hist_offset, T);
                k_combine_experts_hist_push<<<(unsigned) ((n4 + (int) cb - 1) / (int) cb), cb, 0, st>>>(
                    eparts, ewts, post_conv->hist, N, Kp, post_conv->hist_offset, T);
                if (prof) { CK(cudaStreamSynchronize(st)); p_moe_gpu_ms += ms_since(tp_moe_r0); }
                return true;
            }
            const int n4 = ((N & 3) == 0) ? (N >> 2) : N;
            const unsigned c_blk = ((n4 % 192) == 0) ? 192u : 256u;
            k_combine_experts<<<dim3((unsigned) ((n4 + (int) c_blk - 1) / (int) c_blk), (unsigned) T), c_blk, 0, st>>>(eparts, ewts, moe_raw, N, Kp);
            if (prof) { CK(cudaStreamSynchronize(st)); p_moe_gpu_ms += ms_since(tp_moe_r0); }
            return false;
        }

        if (need_cpu) {
            if (T > cpu_buf_cap) init_cpu_bufs(T);
            if (!pre_routed_t1) {
                CK(cudaMemcpyAsync(h_ids, ids, (size_t) n_ent * sizeof(int32_t), cudaMemcpyDeviceToHost, st));
                if (T >= 1 && T <= std::max(verify_max_t, 8)) {
                    CK(cudaMemcpyAsync(h_ewts, ewts, (size_t) n_ent * sizeof(float), cudaMemcpyDeviceToHost, st));
                } else {
                    CK(cudaMemcpyAsync(h_h, x_norm, (size_t) T * N * sizeof(float), cudaMemcpyDeviceToHost, st));
                }
            }
            CK(cudaEventRecord(ev_route, st));
        }

        const auto tp_mg0 = Clock::now();
        if (has_gpu) {
            if (!fused_rg && !pre_grouped_t1 && !can_fused_router_t1) {
                if (use_group16) {
                    k_group_experts_hist16<<<16, 256, 0, st>>>(ids, n_ent, (int) hp.n_expert, d_grp_part_cnt);
                    k_group_experts_prefix16<<<1, 128, 0, st>>>(d_grp_part_cnt, d_grp_part_off, 16, L.d_slot_of, L.d_blobs, L.stride,
                                                                (int) hp.n_expert, L.n_slots, grp_ptr, grp_start, n_groups);
                    k_group_experts_scatter16<<<16, 256, 0, st>>>(ids, d_grp_part_off, n_ent, (int) hp.n_expert, ent_dst, ent_tok, Kp);
                } else {
                    k_group_experts<<<1, 512, 0, st>>>(ids, L.d_slot_of, L.d_blobs, L.stride, n_ent, (int) hp.n_expert, L.n_slots,
                                                       grp_ptr, grp_start, n_groups, ent_dst, ent_tok, Kp);
                }
            }
            run_grouped(grp_ptr, std::min<int64_t>(n_ent, hp.n_expert));
            l2_guard.reset();
        } else if (use_atomic_moe) {
            ensure_moe_raw_zeroed();
        }
        if (prof_detail) {
            CK(cudaStreamSynchronize(st));
            const double mg_dt = ms_since(tp_mg0);
            if (!need_cpu) p_d_moe_gpu_core += mg_dt;
            else if (is_q5) p_d_moe_gpu_q5 += mg_dt;
            else p_d_moe_gpu_nv += mg_dt;
        }
        if (prof && need_cpu) { CK(cudaStreamSynchronize(st)); p_moe_gpu_ms += ms_since(tp_moe_r0); }

#if defined(STRATA_INKLING2_CPU_POOL)
        if (need_cpu) {
            namespace C = strata::kernels::cpu;
            const auto tp_wait0 = Clock::now();
            CK(cudaEventSynchronize(ev_route));
            if (prof) p_moe_wait_ms += ms_since(tp_wait0);
            const auto t0 = Clock::now();
            jobs.clear();
            int n_miss = 0;
            if (T == 1) {
                const void* act0 = (L.cpu_fmt.gu_act == 0) ? (const void*) h_h : (const void*) acts;
                for (int e = 0; e < Kp; ++e) {
                    const int x = h_ids[e];
                    L.score[(size_t) x] += 1.0;
                    h_miss[e] = L.slot_of[(size_t) x] < 0 ? 1 : 0;
                    if (!h_miss[e]) continue;
                    ++n_miss;
                    C::ExpertJobMulti j;
                    j.blob = get_or_build_expert_hblob(L, x, (int) hp.n_expert);
                    j.nact[0] = act0;
                    j.out[0] = cpu_out + (size_t) e * N;
                    j.nt = 1;
                    jobs.push_back(j);
                }
                if (n_miss > 0 && L.cpu_fmt.gu_act != 0) {
                    C::native_quant_act(L.cpu_fmt, h_h, acts);
                }
            } else {
                std::vector<char> quantized((size_t) T, 0);
                int open_job[128];
                std::fill_n(open_job, (int) hp.n_expert, -1);
                for (int e = 0; e < n_ent; ++e) {
                    const int x = h_ids[e];
                    L.score[(size_t) x] += 1.0;
                    h_miss[e] = L.slot_of[(size_t) x] < 0 ? 1 : 0;
                    if (!h_miss[e]) continue;
                    ++n_miss;
                    const int t = e / Kp;
                    if (L.cpu_fmt.gu_act != 0 && !quantized[(size_t) t]) {
                        C::native_quant_act(L.cpu_fmt, h_h + (size_t) t * N, acts + (size_t) t * C::kNativeActBytes);
                        quantized[(size_t) t] = 1;
                    }
                    int jidx = open_job[x];
                    if (jidx < 0 || jobs[(size_t) jidx].nt == C::MAXT) {
                        C::ExpertJobMulti j;
                        j.blob = get_or_build_expert_hblob(L, x, (int) hp.n_expert);
                        jidx = (int) jobs.size();
                        jobs.push_back(j);
                        open_job[x] = jidx;
                    }
                    C::ExpertJobMulti& j = jobs[(size_t) jidx];
                    j.nact[j.nt] = (L.cpu_fmt.gu_act == 0)
                                       ? (const void*) (h_h + (size_t) t * N)
                                       : (const void*) (acts + (size_t) t * C::kNativeActBytes);
                    j.out[j.nt] = cpu_out + (size_t) e * N;
                    ++j.nt;
                }
            }
            hy_cpu += n_miss;
            hy_gpu += n_ent - n_miss;
            if (prof_detail) {
                if (is_q5) { p_miss_q5 += n_miss; p_hit_q5 += (n_ent - n_miss); }
                else { p_miss_nv += n_miss; p_hit_nv += (n_ent - n_miss); }
            }
            if (n_miss > 0) {
                static const int l2_pf_min = [] { const char* v = std::getenv("STRATA_L2_PF_MIN"); return v ? std::atoi(v) : 0; }();
                if (l2_pf_min > 0 && n_miss >= l2_pf_min && hybrid && T <= std::max(verify_max_t, 8) && il + 1 < hp.n_layer) {
                    // F2: Free L2 prefetch of the next layer's dense weights on GPU while CPU computes missed experts!
                    const int ip_next = hp.loop_layer(il + 1);
                    const PhysLayer& Ln = m->layers[(size_t) ip_next];
                    K::L2PrefetchSet pf{};
                    pf.add(Ln.attn_q.d, Ln.attn_q.bytes);
                    pf.add(Ln.attn_gate.d, Ln.attn_gate.bytes);
                    pf.add(Ln.attn_rel.d, Ln.attn_rel.bytes);
                    pf.add(Ln.attn_k.d, Ln.attn_k.bytes);
                    pf.add(Ln.attn_v.d, Ln.attn_v.bytes);
                    pf.add(Ln.attn_output.d, Ln.attn_output.bytes);
                    if (pf.n > 0) K::l2_prefetch_ranges(pf, st);
                }
                const auto tp_c0 = Clock::now();
                pool->run_split_multi_native(L.cpu_fmt, jobs.data(), (int) jobs.size());
                const double c_ms = ms_since(tp_c0);
                hy_cpu_ms += c_ms;
                if (prof) p_moe_cpu_ms += c_ms;
                if (prof_detail) {
                    if (is_q5) p_d_cpu_q5 += c_ms;
                    else p_d_cpu_nv += c_ms;
                }
                const auto tp_u0 = Clock::now();
                if (T >= 1 && T <= std::max(verify_max_t, 8) && !skip_combine) {
                    // Pre-combine all CPU-missed experts per token into T x N floats (8x smaller H2D, 0 k_merge_cpu!).
                    for (int t = 0; t < T; ++t) {
                        float* dst = h_cpu_sum + (size_t) t * N;
                        const int base = t * Kp;
                        bool first = true;
                        for (int k = 0; k < Kp; ++k) {
                            if (!h_miss[base + k]) continue;
                            const float w = h_ewts[base + k];
                            const float* src = cpu_out + (size_t) (base + k) * N;
                            if (first) {
                                for (int i = 0; i < N; ++i) dst[i] = w * src[i];
                                first = false;
                            } else {
                                for (int i = 0; i < N; ++i) dst[i] += w * src[i];
                            }
                        }
                        if (first) std::memset(dst, 0, (size_t) N * sizeof(float));
                    }
                    const size_t sum_bytes = (size_t) T * (size_t) N * sizeof(float);
                    if (has_gpu) {
                        CK(cudaMemcpyAsync(ecpu, h_cpu_sum, sum_bytes, cudaMemcpyHostToDevice, st));
                        if (prof) { CK(cudaStreamSynchronize(st)); p_moe_up_ms += ms_since(tp_u0); }
                        const auto tp_p0 = Clock::now();
                        if (post_conv && post_conv->w_conv) {
                            k_combine_experts_hybrid_shortconv<<<(unsigned) ((N + 31) / 32), 32, 0, st>>>(
                                eparts, ewts, ids, L.d_slot_of, ecpu,
                                post_conv->w_conv, post_conv->hist, post_conv->res_add, post_conv->out,
                                N, Kp, post_conv->hist_offset, T, post_conv->hist_trace, post_conv->hist_trace_stride);
                            if (prof) { CK(cudaStreamSynchronize(st)); p_moe_post_ms += ms_since(tp_p0); }
                            return true;
                        }
                        const int n4 = ((N & 3) == 0) ? (N >> 2) : N;
                        k_combine_experts_hybrid<<<dim3((unsigned) ((n4 + 255) / 256), (unsigned) T), 256, 0, st>>>(
                            eparts, ewts, ids, L.d_slot_of, ecpu, moe_raw, N, Kp);
                        if (prof) { CK(cudaStreamSynchronize(st)); p_moe_post_ms += ms_since(tp_p0); }
                    } else {
                        CK(cudaMemcpyAsync(moe_raw, h_cpu_sum, sum_bytes, cudaMemcpyHostToDevice, st));
                        if (prof) { CK(cudaStreamSynchronize(st)); p_moe_up_ms += ms_since(tp_u0); }
                    }
                    return false;
                }
                if (has_gpu) {
                    CK(cudaMemcpyAsync(ecpu, cpu_out, (size_t) n_ent * N * sizeof(float), cudaMemcpyHostToDevice, st));
                    CK(cudaMemcpyAsync(d_miss, h_miss, (size_t) n_ent * sizeof(int32_t), cudaMemcpyHostToDevice, st));
                    k_merge_cpu<<<(unsigned) n_ent, kThreads, 0, st>>>(eparts, ecpu, d_miss, N);
                } else {
                    CK(cudaMemcpyAsync(eparts, cpu_out, (size_t) n_ent * N * sizeof(float), cudaMemcpyHostToDevice, st));
                }
                if (prof) { CK(cudaStreamSynchronize(st)); p_moe_up_ms += ms_since(tp_u0); }
            }
        } else {
            hy_gpu += n_ent;
        }
#else
        if (need_cpu) {
            std::fprintf(stderr, "strata-inkling2: layer needs CPU pool but STRATA_NATIVE_EXPERTS is OFF\n");
            std::exit(1);
        }
        hy_gpu += n_ent;
#endif
        if (skip_combine) return false;
        const auto tp_p0 = Clock::now();
        if (use_atomic_moe) {
            if (post_conv && post_conv->w_conv && T >= 32 && (N & 3) == 0 && !post_conv->hist_trace) {
                const int n4 = N >> 2;
                k_shortconv_pp_f4<<<dim3((unsigned) ((n4 + 63) / 64), (unsigned) ((T + 63) / 64)), 64, 0, st>>>(
                    moe_raw, post_conv->w_conv, post_conv->hist, post_conv->out, N, post_conv->hist_offset, T, post_conv->res_add);
                k_shortconv_hist_push<<<(unsigned) ((N + 255) / 256), 256, 0, st>>>(
                    moe_raw, post_conv->hist, N, post_conv->hist_offset, T);
                if (prof && need_cpu) { CK(cudaStreamSynchronize(st)); p_moe_post_ms += ms_since(tp_p0); }
                return true;
            }
            if (prof && need_cpu) { CK(cudaStreamSynchronize(st)); p_moe_post_ms += ms_since(tp_p0); }
            return false;
        }
        if (post_conv && post_conv->w_conv && T <= std::max(verify_max_t, 8)) {
            k_combine_experts_shortconv<<<(unsigned) ((N + 31) / 32), 32, 0, st>>>(
                eparts, ewts, post_conv->w_conv, post_conv->hist, post_conv->res_add, post_conv->out,
                N, Kp, post_conv->hist_offset, T, post_conv->hist_trace, post_conv->hist_trace_stride);
            if (prof && need_cpu) { CK(cudaStreamSynchronize(st)); p_moe_post_ms += ms_since(tp_p0); }
            return true;
        }
        if (post_conv && post_conv->w_conv && T >= 32 && (N & 3) == 0 && !post_conv->hist_trace) {
            const int n4 = N >> 2;
            const unsigned cb = ((n4 % 192) == 0) ? 192u : 64u;
            k_combine_experts_shortconv_pp<<<dim3((unsigned) ((n4 + (int) cb - 1) / (int) cb), (unsigned) ((T + 127) / 128)), cb, 0, st>>>(
                eparts, ewts, post_conv->w_conv, post_conv->hist, post_conv->res_add, post_conv->out,
                N, Kp, post_conv->hist_offset, T);
            k_combine_experts_hist_push<<<(unsigned) ((n4 + (int) cb - 1) / (int) cb), cb, 0, st>>>(
                eparts, ewts, post_conv->hist, N, Kp, post_conv->hist_offset, T);
            if (prof && need_cpu) { CK(cudaStreamSynchronize(st)); p_moe_post_ms += ms_since(tp_p0); }
            return true;
        }
        const int n4 = ((N & 3) == 0) ? (N >> 2) : N;
        k_combine_experts<<<dim3((unsigned) ((n4 + 255) / 256), (unsigned) T), 256, 0, st>>>(eparts, ewts, moe_raw, N, Kp);
        if (prof && need_cpu) { CK(cudaStreamSynchronize(st)); p_moe_post_ms += ms_since(tp_p0); }
        return false;
    }

    void rebalance(int max_swaps, int max_per_layer = 1, bool need_sync_hits = true) {
        if (!hybrid) return;
        if (need_sync_hits) {
            CK(cudaStreamSynchronize(st));
            sync_d_hits(false);
        }
        const auto& hp = m->hp;
        const int64_t ne = hp.n_expert;
        struct Cand {
            double gain;
            size_t li;
            int slot;
            int hot_x;
            int cold_y;
        };
        std::vector<Cand> cands;
        cands.reserve(m->layers.size() * (size_t) std::max(1, max_per_layer));
        std::vector<int> order((size_t) ne);
        std::vector<int> victims;
        for (size_t li = 0; li < m->layers.size(); ++li) {
            PhysLayer& L = m->layers[li];
            const int S = L.n_slots;
            if (S <= 0 || S >= ne) continue;
            const bool is_core = ((int) li >= hp.core_start && (int) li < hp.core_end);
            const double weight = is_core ? 3.0 : ((L.L.gu_type == 13) ? 1.43 : 1.0);
            for (int x = 0; x < ne; ++x) order[(size_t) x] = x;
            std::sort(order.begin(), order.end(),
                      [&](int a, int b) { return effective_score(L, a) > effective_score(L, b); });
            victims.resize((size_t) S);
            for (int s = 0; s < S; ++s) victims[(size_t) s] = s;
            const int k_lim = std::min(S, std::max(1, max_per_layer));
            std::partial_sort(victims.begin(), victims.begin() + k_lim, victims.end(),
                              [&](int a, int b) {
                                  return effective_score(L, L.exp_in_slot[(size_t) a]) < effective_score(L, L.exp_in_slot[(size_t) b]);
                              });
            int v_idx = 0;
            for (int i = 0; i < S && v_idx < k_lim; ++i) {
                const int hot_cand = order[(size_t) i];
                if (L.slot_of[(size_t) hot_cand] >= 0) continue;
                const int s = victims[(size_t) v_idx];
                const int y = L.exp_in_slot[(size_t) s];
                const double diff = effective_score(L, hot_cand) - effective_score(L, y);
                // Hysteresis avoids thrashing 4.5 MiB H2D swaps on near-tied scores during decode.
                const double min_diff = (max_swaps >= 48) ? 0.0 : std::max(0.25, 0.15 * effective_score(L, y));
                if (diff <= min_diff) break;
                cands.push_back(Cand{diff * weight, li, s, hot_cand, y});
                ++v_idx;
            }
            const double decay = in_decode_phase ? 0.92 : 0.80;
            for (int x = 0; x < ne; ++x) L.score[(size_t) x] *= decay;
        }
        if (cands.empty()) return;
        std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.gain > b.gain; });
        const int take = std::min<int>(max_swaps, (int) cands.size());
        // Sort chosen swaps by ascending physical layer index so earlier layers run while later layers copy on cst!
        std::sort(cands.begin(), cands.begin() + take, [](const Cand& a, const Cand& b) { return a.li < b.li; });
        int idx = 0;
        while (idx < take) {
            const size_t li = cands[(size_t) idx].li;
            PhysLayer& L = m->layers[li];
            while (idx < take && cands[(size_t) idx].li == li) {
                const Cand& c = cands[(size_t) idx++];
                L.slot_of[(size_t) c.cold_y] = -1;
                L.slot_of[(size_t) c.hot_x] = c.slot;
                L.exp_in_slot[(size_t) c.slot] = c.hot_x;
                const uint8_t* src_hot = get_or_build_expert_hblob(L, c.hot_x, (int) ne);
                copy_h2d_staged(L.d_blobs + (size_t) c.slot * L.stride, src_hot,
                                L.L.bytes, L.hblobs_pinned, cst);
            }
            if (h_slot_pin) {
                int32_t* sp = h_slot_pin + li * (size_t) ne;
                std::memcpy(sp, L.slot_of.data(), (size_t) ne * sizeof(int32_t));
                CK(cudaMemcpyAsync(L.d_slot_of, sp, (size_t) ne * sizeof(int32_t), cudaMemcpyHostToDevice, cst));
            } else {
                CK(cudaMemcpyAsync(L.d_slot_of, L.slot_of.data(), (size_t) ne * sizeof(int32_t), cudaMemcpyHostToDevice, cst));
            }
            if ((int) li < kMaxRebalLayers) {
                CK(cudaEventRecord(ev_rebal_layer[li], cst));
                layer_pending_rebal[li] = true;
            }
        }
    }

    void defer_destroy_graphs() {
        auto defer = [&](cudaGraphExec_t& g) {
            if (g) { deferred_graphs_destroy.push_back(g); g = nullptr; }
        };
        defer(core_graph);
        for (int t = 0; t < 10; ++t) defer(verify_core_graph[t]);
        for (int t = 0; t < 10; ++t)
            for (int c = 0; c < 2; ++c)
                for (int s = 0; s < 2; ++s)
                    for (int a = 0; a < 2; ++a)
                        defer(full_decode_graph[t][c][s][a]);
        for (auto& g : layer_pre_graph) defer(g);
        for (int t = 0; t < 10; ++t) {
            for (auto& g : verify_pre_graph[t]) defer(g);
        }
        graph_oom = false;
    }

    void flush_deferred_graphs() {
        for (cudaGraphExec_t g : deferred_graphs_destroy) {
            if (g) cudaGraphExecDestroy(g);
        }
        deferred_graphs_destroy.clear();
        if (dspark) dspark->flush_deferred_graphs();
    }

    void destroy_graphs() {
        defer_destroy_graphs();
        flush_deferred_graphs();
    }

    // Dynamic VRAM elasticity (--vram-elastic & Prefill<->Decode phase split):
    // EX-5 Idea C & D: 3-pass lock-free resize (Pass 1: plan + D2D shrink + cudaFree; Pass 2: batched cudaMalloc +
    // pinned d_slot_of upload on st; Pass 3: priority-ordered coalesced H2D on cst with optional async return!).
    void resize_experts_vram(size_t target_free_bytes, size_t guard_override = 0, bool equalize_tier2 = false,
                             bool async_cst = false, bool whole_layer_evict = false, bool restore_dspark_decode_w = false,
                             bool skip_sync = false, size_t known_fr = 0, size_t known_tot = 0,
                             bool knapsack_decode = false, bool tier1_only = false) {
        const auto t0 = Clock::now();
        const auto& hp = m->hp;
        const int ne = (int) hp.n_expert;

        if (!skip_sync) {
            CK(cudaStreamSynchronize(st));
            if (cst) CK(cudaStreamSynchronize(cst));
            if (aux_st) CK(cudaStreamSynchronize(aux_st));
        }
        for (int i = 0; i < kMaxRebalLayers; ++i) layer_pending_rebal[i] = false;
        if (dspark && dspark->dm) dspark->dm->decode_w_h2d_pending = false;
        defer_destroy_graphs();
        if (!(skip_sync && whole_layer_evict)) sync_d_hits(false);

        const bool core_res_hy = (profile.regime == strata::inkling2::VramRegime::CoreResidentHybrid);
        const size_t base_guard = (guard_override > 0)
            ? guard_override
            : (core_res_hy
                ? (verify_max_t > 1 ? ((size_t) 176 << 20) : ((size_t) 160 << 20))
                : ((size_t) 256 << 20));

        auto demote_tier2_pin = [&](size_t need_bytes) {
            for (int j = (int) m->layers.size() - 1; j >= 0 && pinned_used + need_bytes > max_pin_bytes; --j) {
                if (j >= hp.core_start && j < hp.core_end) continue;
                PhysLayer& T2 = m->layers[(size_t) j];
                if (!T2.hblobs || !T2.hblobs_pinned) continue;
                const size_t tb = (size_t) ne * T2.stride;
                uint8_t* pag = (uint8_t*) std::malloc(tb);
                if (!pag) continue;
                std::memcpy(pag, T2.hblobs, tb);
                auto it = std::find(m->host_pinned_allocs.begin(), m->host_pinned_allocs.end(), (void*) T2.hblobs);
                if (it != m->host_pinned_allocs.end()) m->host_pinned_allocs.erase(it);
                cudaFreeHost(T2.hblobs);
                T2.hblobs = pag;
                T2.hblobs_pinned = false;
                m->host_allocs.push_back(pag);
                pinned_used = (pinned_used >= tb) ? (pinned_used - tb) : 0;
            }
        };

        auto ensure_hblobs = [&](PhysLayer& L, bool is_tier1) {
            if (L.hblobs) return;
            const size_t hb_bytes = (size_t) ne * L.stride;
            const bool want_pin = layer_allow_pin_hblobs(L);
            if (want_pin && pinned_used + hb_bytes > max_pin_bytes && is_tier1) demote_tier2_pin(hb_bytes);
            if (want_pin && pinned_used + hb_bytes <= max_pin_bytes &&
                cudaHostAlloc(&L.hblobs, hb_bytes, cudaHostAllocDefault) == cudaSuccess) {
                m->host_pinned_allocs.push_back(L.hblobs);
                L.hblobs_pinned = true;
                pinned_used += hb_bytes;
            } else {
                (void) cudaGetLastError();
                L.hblobs = (uint8_t*) std::malloc(hb_bytes);
                if (!L.hblobs) { std::fprintf(stderr, "strata-inkling2: host malloc failed during VRAM resize\n"); std::exit(1); }
                L.hblobs_pinned = false;
                m->host_allocs.push_back(L.hblobs);
            }
            bool can_d2h = (L.n_slots == ne && L.d_blobs != nullptr);
            if (can_d2h) {
                for (int e = 0; e < ne; ++e) if (L.slot_of[(size_t) e] != e) { can_d2h = false; break; }
            }
            if (can_d2h) {
                CK(cudaMemcpyAsync(L.hblobs, L.d_blobs, hb_bytes, cudaMemcpyDeviceToHost, st));
                CK(cudaStreamSynchronize(st));
            } else {
                build_all_expert_blobs(L, ne, L.hblobs);
                drop_layer_mmap_pages(L, ne);
            }
        };

        std::vector<int> tier1, tier2, tier3;
        for (int ip = 0; ip < hp.n_phys; ++ip) {
            if (m->layers[(size_t) ip].gu_type == 30) tier3.push_back(ip);
            else if (ip >= hp.core_start && ip < hp.core_end) tier1.push_back(ip);
            else tier2.push_back(ip);
        }

        const size_t dspark_w_need = (restore_dspark_decode_w && dspark && dspark->dm &&
                                      dspark->dm->decode_w_offloadable && !dspark->dm->decode_w_resident &&
                                      !dspark->dm->decode_w_gpu)
                                     ? dspark->dm->decode_w_bytes : 0;

        std::vector<int> target_slots((size_t) hp.n_phys);
        for (int ip = 0; ip < hp.n_phys; ++ip) target_slots[(size_t) ip] = m->layers[(size_t) ip].n_slots;

        size_t fr = known_fr, tot = known_tot;
        if (tot == 0) CK(cudaMemGetInfo(&fr, &tot));
        bool any_raw_vram_op = false;
        const size_t pool_free0 = pool_cached_free_bytes();
        const size_t eff_fr = fr + pool_free0;

        // ---- Plan target_slots[0..n_phys-1] ----
        static const bool allow_core_spill = [] {
            const char* v = std::getenv("STRATA_KNAP_ALLOW_CORE_SPILL");
            return (v && v[0]) ? (std::atoi(v) != 0) : false;
        }();
        const float* prior_ptr = strata::inkling2::inkling2_mini_decode_prior(hp.n_phys, ne);
        const bool use_knapsack = (knapsack_decode && prior_ptr != nullptr && !whole_layer_evict);
        std::vector<double> blended_scores;
        if (use_knapsack) {
            blended_scores.resize((size_t) hp.n_phys * (size_t) ne, 0.0);
            for (int ip = 0; ip < hp.n_phys; ++ip) {
                const PhysLayer& L = m->layers[(size_t) ip];
                const double mult = (ip >= hp.core_start && ip < hp.core_end) ? 24.0 : 8.0;
                double sum_live = 0.0;
                for (int e = 0; e < ne; ++e) sum_live += L.score[(size_t) e];
                for (int e = 0; e < ne; ++e) {
                    const double pe = (e < (int) L.prior_score.size()) ? L.prior_score[(size_t) e] : (double) prior_ptr[(size_t) ip * ne + e];
                    const double le = (sum_live > 1e-6) ? (L.score[(size_t) e] / sum_live) * mult : pe;
                    blended_scores[(size_t) ip * ne + e] = 0.65 * pe + 0.35 * le;
                }
            }
            const size_t keep_free = std::max(target_free_bytes, base_guard) + dspark_w_need;
            size_t cur_all_b = 0;
            size_t tier1_full_b = 0;
            size_t t1_t2_full_b = 0;
            for (int ip = 0; ip < hp.n_phys; ++ip) {
                const PhysLayer& L = m->layers[(size_t) ip];
                cur_all_b += (size_t) L.n_slots * L.stride;
                if (L.gu_type == 30) continue;
                const size_t lb = (size_t) ne * L.stride;
                if (ip >= hp.core_start && ip < hp.core_end) tier1_full_b += lb;
                t1_t2_full_b += lb;
            }
            const size_t avail_total = (cur_all_b + eff_fr > keep_free) ? (cur_all_b + eff_fr - keep_free) : 0;
            struct KnapItem {
                double val;
                int ip;
                int e;
                size_t stride;
            };
            const bool knap_concave = [&] {
                const char* v = std::getenv("STRATA_KNAP_CONCAVE");
                return (v && v[0]) ? (std::atoi(v) != 0) : allow_core_spill;
            }();
            const double T_knap = [&] {
                const char* v = std::getenv("STRATA_KNAP_T");
                if (v && v[0]) return (double) std::clamp(std::atoi(v), 1, 16);
                return (double) std::clamp(verify_max_t >= 2 ? verify_max_t : 1, 4, 8);
            }();
            auto knap_val = [&](int ip, double score_e) -> double {
                const PhysLayer& L = m->layers[(size_t) ip];
                if (!knap_concave || L.stride == 0) return score_e;
                const bool is_core = (ip >= hp.core_start && ip < hp.core_end);
                const double loops = is_core ? (double) hp.num_loops : 1.0;
                const double q_e = std::clamp(score_e / loops, 0.0, 1.0);
                return loops * (1.0 - std::pow(1.0 - q_e, T_knap)) * ((double) L.L.bytes / (double) L.stride);
            };
            if (avail_total >= t1_t2_full_b) {
                for (int ip : tier1) target_slots[(size_t) ip] = ne;
                for (int ip : tier2) target_slots[(size_t) ip] = ne;
                knapsack_decode_active = false;
            } else if (!allow_core_spill && avail_total >= tier1_full_b) {
                for (int ip : tier1) target_slots[(size_t) ip] = ne;
                for (int ip : tier2) target_slots[(size_t) ip] = 0;
                for (int ip : tier3) target_slots[(size_t) ip] = 0;
                size_t rem = avail_total - tier1_full_b;
                std::vector<KnapItem> items;
                items.reserve(tier2.size() * (size_t) ne);
                for (int ip : tier2) {
                    const size_t st_b = m->layers[(size_t) ip].stride;
                    for (int e = 0; e < ne; ++e) {
                        items.push_back(KnapItem{knap_val(ip, blended_scores[(size_t) ip * ne + e]), ip, e, st_b});
                    }
                }
                std::sort(items.begin(), items.end(), [](const KnapItem& a, const KnapItem& b) {
                    if (a.val != b.val) return a.val > b.val;
                    if (a.ip != b.ip) return a.ip < b.ip;
                    return a.e < b.e;
                });
                for (const KnapItem& it : items) {
                    if (rem >= it.stride && target_slots[(size_t) it.ip] < ne) {
                        ++target_slots[(size_t) it.ip];
                        rem -= it.stride;
                    }
                }
                knapsack_decode_active = false;
            } else {
                for (int ip : tier1) target_slots[(size_t) ip] = 0;
                for (int ip : tier2) target_slots[(size_t) ip] = 0;
                for (int ip : tier3) target_slots[(size_t) ip] = 0;
                size_t rem = avail_total;
                std::vector<KnapItem> items;
                items.reserve((tier1.size() + tier2.size()) * (size_t) ne);
                for (int ip = 0; ip < hp.n_phys; ++ip) {
                    const PhysLayer& L = m->layers[(size_t) ip];
                    if (L.gu_type == 30) continue;
                    for (int e = 0; e < ne; ++e) {
                        items.push_back(KnapItem{knap_val(ip, blended_scores[(size_t) ip * ne + e]), ip, e, L.stride});
                    }
                }
                std::sort(items.begin(), items.end(), [](const KnapItem& a, const KnapItem& b) {
                    if (a.val != b.val) return a.val > b.val;
                    if (a.ip != b.ip) return a.ip < b.ip;
                    return a.e < b.e;
                });
                for (const KnapItem& it : items) {
                    if (rem >= it.stride && target_slots[(size_t) it.ip] < ne) {
                        ++target_slots[(size_t) it.ip];
                        rem -= it.stride;
                    }
                }
                knapsack_decode_active = true;
            }
        } else if (eff_fr < target_free_bytes + dspark_w_need) {
            const size_t eff_target = target_free_bytes + dspark_w_need;
            size_t freed_est = 0;
            auto plan_shrink_tier = [&](const std::vector<int>& tier, bool whole_evict) {
                for (int idx = (int) tier.size() - 1; idx >= 0 && eff_fr + freed_est < eff_target; --idx) {
                    const int ip = tier[(size_t) idx];
                    const PhysLayer& L = m->layers[(size_t) ip];
                    const int cur_s = target_slots[(size_t) ip];
                    if (cur_s <= 0) continue;
                    const size_t deficit = eff_target - (eff_fr + freed_est);
                    const int drop = (L.stride > 0) ? (int) ((deficit + L.stride - 1) / L.stride) : cur_s;
                    const int tgt = (whole_evict || drop >= cur_s) ? 0 : (cur_s - drop);
                    freed_est += (size_t) (cur_s - tgt) * L.stride;
                    target_slots[(size_t) ip] = tgt;
                }
            };
            plan_shrink_tier(tier3, whole_layer_evict);
            plan_shrink_tier(tier2, whole_layer_evict);
            plan_shrink_tier(tier1, false);
        } else {
            const size_t keep_free = std::max(target_free_bytes, base_guard) + dspark_w_need;
            size_t virt_fr = eff_fr;
            // Wave 14: During equalize_tier2 on CoreResidentHybrid (core_all_gpu, !TightHybrid), if total Tier-2
            // budget fits >= ne (128) slots for blk.0, pin blk.0 at 128/128 and equalize the remaining Tier-2 layers.
            // Eliminates Turn-2 L0 multi-chunk H2D and lets cst start prefetching L1 immediately at t=0.
            bool pin_blk0 = false;
            std::vector<int> tier2_pin, tier2_rest;
            if (equalize_tier2 && !tier2.empty()) {
                const size_t avail_eq = (virt_fr > keep_free) ? (virt_fr - keep_free) : 0;
                bool tier1_full = core_all_gpu;
                for (int ip : tier1) {
                    if (target_slots[(size_t) ip] < ne) { tier1_full = false; break; }
                }
                if (tier1_full && tier2[0] == 0) {
                    size_t sum_t2_b = 0;
                    for (int ip : tier2) sum_t2_b += (size_t) target_slots[(size_t) ip] * m->layers[(size_t) ip].stride;
                    const size_t blk0_full_b = (size_t) ne * m->layers[0].stride;
                    if (blk0_full_b > 0 && sum_t2_b + avail_eq >= blk0_full_b) {
                        pin_blk0 = true;
                        tier2_pin.push_back(0);
                        tier2_rest.assign(tier2.begin() + 1, tier2.end());
                        const size_t blk0_need_b = (target_slots[0] < ne)
                            ? (size_t) (ne - target_slots[0]) * m->layers[0].stride : 0;
                        const size_t rest_budget_b = sum_t2_b + avail_eq - blk0_full_b;
                        size_t sum_rest_st = 0;
                        int min_s = ne, max_s = 0;
                        for (int ip : tier2_rest) {
                            sum_rest_st += m->layers[(size_t) ip].stride;
                            min_s = std::min(min_s, target_slots[(size_t) ip]);
                            max_s = std::max(max_s, target_slots[(size_t) ip]);
                        }
                        if (!tier2_rest.empty() && sum_rest_st > 0 && (blk0_need_b > avail_eq || max_s - min_s > 16)) {
                            const int eq_s = (int) std::min<size_t>((size_t) ne, rest_budget_b / sum_rest_st);
                            for (int ip : tier2_rest) {
                                if (target_slots[(size_t) ip] > eq_s) {
                                    virt_fr += (size_t) (target_slots[(size_t) ip] - eq_s) * m->layers[(size_t) ip].stride;
                                    target_slots[(size_t) ip] = eq_s;
                                }
                            }
                        }
                    }
                }
                if (!pin_blk0) {
                    size_t sum_cur_b = 0, sum_st = 0;
                    int min_s = ne, max_s = 0;
                    for (int ip : tier2) {
                        const PhysLayer& L = m->layers[(size_t) ip];
                        sum_cur_b += (size_t) target_slots[(size_t) ip] * L.stride;
                        sum_st += L.stride;
                        min_s = std::min(min_s, target_slots[(size_t) ip]);
                        max_s = std::max(max_s, target_slots[(size_t) ip]);
                    }
                    if (max_s - min_s > 16 && sum_st > 0) {
                        const int eq_s = (int) std::min<size_t>((size_t) ne, (sum_cur_b + avail_eq) / sum_st);
                        for (int ip : tier2) {
                            if (target_slots[(size_t) ip] > eq_s) {
                                virt_fr += (size_t) (target_slots[(size_t) ip] - eq_s) * m->layers[(size_t) ip].stride;
                                target_slots[(size_t) ip] = eq_s;
                            }
                        }
                    }
                }
            }
            auto plan_grow_tier = [&](const std::vector<int>& tier) {
                if (tier.empty() || virt_fr <= keep_free) return;
                size_t avail = virt_fr - keep_free;
                size_t need_full = 0;
                for (int ip : tier) {
                    const PhysLayer& L = m->layers[(size_t) ip];
                    if (target_slots[(size_t) ip] < ne) need_full += (size_t) (ne - target_slots[(size_t) ip]) * L.stride;
                }
                if (need_full == 0) return;
                if (avail >= need_full) {
                    for (int ip : tier) target_slots[(size_t) ip] = ne;
                    virt_fr -= need_full;
                    return;
                }
                std::vector<int> targets(tier.size());
                size_t sum_cur_bytes = 0, sum_stride = 0;
                for (size_t i = 0; i < tier.size(); ++i) {
                    const PhysLayer& L = m->layers[(size_t) tier[i]];
                    targets[i] = target_slots[(size_t) tier[i]];
                    sum_cur_bytes += (size_t) targets[i] * L.stride;
                    sum_stride += L.stride;
                }
                const size_t total_b = sum_cur_bytes + avail;
                const int uniform_s = sum_stride > 0 ? (int) std::min<size_t>((size_t) ne, total_b / sum_stride) : 0;
                size_t used_b = 0;
                for (size_t i = 0; i < tier.size(); ++i) {
                    const int cur_s = target_slots[(size_t) tier[i]];
                    const int ns = std::max(cur_s, uniform_s);
                    targets[i] = std::min(ne, ns);
                    if (targets[i] > cur_s) used_b += (size_t) (targets[i] - cur_s) * m->layers[(size_t) tier[i]].stride;
                }
                bool added = true;
                while (added && used_b < avail) {
                    added = false;
                    for (size_t i = 0; i < tier.size(); ++i) {
                        const size_t sb = m->layers[(size_t) tier[i]].stride;
                        if (targets[i] < ne && used_b + sb <= avail) {
                            targets[i] += 1;
                            used_b += sb;
                            added = true;
                        }
                    }
                }
                for (size_t i = 0; i < tier.size(); ++i) target_slots[(size_t) tier[i]] = targets[i];
                virt_fr = (virt_fr > used_b) ? (virt_fr - used_b) : 0;
            };
            plan_grow_tier(tier1);
            if (!tier1_only) {
                if (pin_blk0) {
                    plan_grow_tier(tier2_pin);
                    plan_grow_tier(tier2_rest);
                } else {
                    plan_grow_tier(tier2);
                }
                plan_grow_tier(tier3);
            }
        }

        struct H2dRun { int e0; int cnt; int dst_off; };
        std::vector<std::vector<int>> keep_list((size_t) hp.n_phys);
        std::vector<std::vector<H2dRun>> pending_h2d((size_t) hp.n_phys);
        std::vector<bool> layer_resized((size_t) hp.n_phys, false);
        std::vector<bool> layer_done_alloc((size_t) hp.n_phys, false);

        // ---- Pass 1: D2D shrink (when free VRAM allows) & free all other resized layers before any H2D ----
        bool any_pool_alloc = false;
        const bool pool_active = (use_phase_pool && phase_pool != nullptr);
        for (int ip = 0; ip < hp.n_phys; ++ip) {
            PhysLayer& L = m->layers[(size_t) ip];
            const int tgt = std::clamp(target_slots[(size_t) ip], 0, ne);
            target_slots[(size_t) ip] = tgt;
            if (tgt == L.n_slots) continue;
            layer_resized[(size_t) ip] = true;
            const bool is_tier1 = (ip >= hp.core_start && ip < hp.core_end);
            const bool use_pool_for_layer = (!is_tier1 && pool_active);
            const bool use_sparse = (allow_core_spill && is_tier1 && !L.hblobs);
            if (!use_sparse) ensure_hblobs(L, is_tier1);

            if (tgt > 0) {
                std::vector<int> order((size_t) ne);
                for (int e = 0; e < ne; ++e) order[(size_t) e] = e;
                std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
                    const double sa = use_knapsack ? blended_scores[(size_t) ip * ne + a] : effective_score(L, a);
                    const double sb = use_knapsack ? blended_scores[(size_t) ip * ne + b] : effective_score(L, b);
                    if (sa != sb) return sa > sb;
                    return (L.slot_of[(size_t) a] >= 0) > (L.slot_of[(size_t) b] >= 0);
                });
                keep_list[(size_t) ip].assign(order.begin(), order.begin() + tgt);
                std::sort(keep_list[(size_t) ip].begin(), keep_list[(size_t) ip].end());
            }

            // Fast GPU-to-GPU D2D shrink: keeps resident experts at 500 GB/s (~0.8 ms) instead of 35 ms PCIe H2D!
            if (tgt > 0 && tgt < L.n_slots && L.d_blobs != nullptr) {
                CK(cudaMemGetInfo(&fr, &tot));
                size_t avail_d2d = fr + (use_pool_for_layer ? pool_cached_free_bytes() : 0);
                const size_t need_d2d = (size_t) tgt * L.stride;
                if (use_sparse && !use_pool_for_layer && avail_d2d < need_d2d + ((size_t) 64 << 20) &&
                    pool_cached_free_bytes() > ((size_t) 8 << 20)) {
                    trim_phase_pool(0);
                    CK(cudaMemGetInfo(&fr, &tot));
                    avail_d2d = fr;
                }
                if (use_sparse) {
                    const bool will_d2d = (avail_d2d >= need_d2d + ((size_t) 64 << 20));
                    snapshot_evicted_sparse_hblobs(L, keep_list[(size_t) ip], ne, /*only_evicted=*/will_d2d);
                }
                if (avail_d2d >= need_d2d + ((size_t) 64 << 20)) {
                    void* ndb = nullptr;
                    bool ndb_from_pool = false;
                    cudaError_t ae = cudaErrorMemoryAllocation;
                    if (use_pool_for_layer) {
                        ae = pool_alloc(&ndb, need_d2d);
                        if (ae == cudaSuccess && ndb) {
                            ndb_from_pool = true;
                            any_pool_alloc = true;
                        } else {
                            (void) cudaGetLastError();
                            ndb = nullptr;
                        }
                    }
                    if (!ndb) {
                        ae = cudaMalloc(&ndb, need_d2d);
                        if (ae != cudaSuccess && !deferred_graphs_destroy.empty()) {
                            (void) cudaGetLastError();
                            flush_deferred_graphs();
                            ae = cudaMalloc(&ndb, need_d2d);
                        }
                    }
                    if (ae == cudaSuccess && ndb) {
                        uint8_t* new_blobs = (uint8_t*) ndb;
                        const auto& keep = keep_list[(size_t) ip];
                        auto& h2d = pending_h2d[(size_t) ip];
                        bool any_d2d = false;
                        for (int k = 0; k < tgt; ) {
                            const int e0 = keep[(size_t) k];
                            const int s0 = L.slot_of[(size_t) e0];
                            if (s0 >= 0) {
                                int k1 = k + 1;
                                while (k1 < tgt && L.slot_of[(size_t) keep[(size_t) k1]] == s0 + (k1 - k)) ++k1;
                                const int cnt = k1 - k;
                                const size_t bytes = (size_t) (cnt - 1) * L.stride + L.L.bytes;
                                CK(cudaMemcpyAsync(new_blobs + (size_t) k * L.stride,
                                                   L.d_blobs + (size_t) s0 * L.stride,
                                                   bytes, cudaMemcpyDeviceToDevice, st));
                                any_d2d = true;
                                k = k1;
                            } else {
                                int k1 = k + 1;
                                while (k1 < tgt && L.slot_of[(size_t) keep[(size_t) k1]] < 0 &&
                                       keep[(size_t) k1] == keep[(size_t) (k1 - 1)] + 1) ++k1;
                                h2d.push_back(H2dRun{e0, k1 - k, k});
                                k = k1;
                            }
                        }
                        if (any_d2d && !L.d_blobs_from_pool) CK(cudaStreamSynchronize(st));
                        if (use_sparse && tgt == 0 && L.d_blobs != nullptr) {
                snapshot_evicted_sparse_hblobs(L, keep_list[(size_t) ip], ne, /*only_evicted=*/false);
            }
            if (!L.d_blobs_from_pool && L.d_blobs != nullptr) any_raw_vram_op = true;
                        free_layer_blobs(L);
                        L.d_blobs = new_blobs;
                        L.d_blobs_from_pool = ndb_from_pool;
                        if (!ndb_from_pool) m->gpu_allocs.push_back(L.d_blobs);
                        L.n_slots = tgt;
                        std::fill(L.slot_of.begin(), L.slot_of.end(), -1);
                        L.exp_in_slot.assign((size_t) tgt, -1);
                        for (int k = 0; k < tgt; ++k) {
                            const int e = keep[(size_t) k];
                            L.slot_of[(size_t) e] = k;
                            L.exp_in_slot[(size_t) k] = e;
                        }
                        layer_done_alloc[(size_t) ip] = true;
                        continue;
                    }
                    (void) cudaGetLastError();
                }
            }

            if (use_sparse && tgt == 0 && L.d_blobs != nullptr) {
                snapshot_evicted_sparse_hblobs(L, keep_list[(size_t) ip], ne, /*only_evicted=*/false);
            }
            if (!L.d_blobs_from_pool && L.d_blobs != nullptr) any_raw_vram_op = true;
            free_layer_blobs(L);
            L.n_slots = 0;
            std::fill(L.slot_of.begin(), L.slot_of.end(), -1);
            L.exp_in_slot.clear();
        }

        // ---- Pass 2: Batched alloc (pool-backed DSpark decode_w & Tier-2/3, raw Tier-1) + pinned d_slot_of uploads on st ----
        {
            if (!pool_active) any_raw_vram_op = true;
            size_t need_raw_pass2_bytes = pool_active ? 0 : dspark_w_need;
            for (int ip = 0; ip < hp.n_phys; ++ip) {
                const bool is_tier1 = (ip >= hp.core_start && ip < hp.core_end);
                if (layer_resized[(size_t) ip] && !layer_done_alloc[(size_t) ip] && target_slots[(size_t) ip] > 0) {
                    if (is_tier1 || !pool_active) {
                        need_raw_pass2_bytes += (size_t) target_slots[(size_t) ip] * m->layers[(size_t) ip].stride;
                    }
                }
            }
            if (need_raw_pass2_bytes > 0 && pool_cached_free_bytes() > 0) {
                CK(cudaMemGetInfo(&fr, &tot));
                if (fr < need_raw_pass2_bytes + ((size_t) 32 << 20)) {
                    trim_phase_pool(0);
                }
            }
        }
        if (restore_dspark_decode_w && dspark) {
            dspark->alloc_decode_weights(pool_active ? phase_pool : nullptr, st);
            if (dspark->dm && dspark->dm->decode_w_from_pool) any_pool_alloc = true;
        }
        std::vector<int> alloc_order;
        alloc_order.reserve((size_t) hp.n_phys);
        for (int ip : tier1) alloc_order.push_back(ip);
        for (int ip : tier2) alloc_order.push_back(ip);
        for (int ip : tier3) alloc_order.push_back(ip);

        for (int ip : alloc_order) {
            if (!layer_resized[(size_t) ip] || layer_done_alloc[(size_t) ip]) continue;
            const int tgt = target_slots[(size_t) ip];
            if (tgt <= 0) continue;
            PhysLayer& L = m->layers[(size_t) ip];
            const bool is_tier1 = (ip >= hp.core_start && ip < hp.core_end);
            const bool use_pool_for_layer = (!is_tier1 && pool_active);
            void* ndb = nullptr;
            bool ndb_from_pool = false;
            int s = tgt;
            bool trimmed_retry = false;
            while (s > 0) {
                const size_t req_b = (size_t) s * L.stride;
                if (use_pool_for_layer) {
                    if (pool_alloc(&ndb, req_b) == cudaSuccess && ndb) {
                        ndb_from_pool = true;
                        any_pool_alloc = true;
                        break;
                    }
                    (void) cudaGetLastError();
                    ndb = nullptr;
                    --s;
                } else {
                    if (cudaMalloc(&ndb, req_b) == cudaSuccess && ndb) {
                        ndb_from_pool = false;
                        break;
                    }
                    (void) cudaGetLastError();
                    ndb = nullptr;
                    if (!deferred_graphs_destroy.empty()) {
                        flush_deferred_graphs();
                        continue;
                    }
                    if (!trimmed_retry && pool_cached_free_bytes() > 0) {
                        trim_phase_pool(0);
                        trimmed_retry = true;
                        continue;
                    }
                    --s;
                }
            }
            if (s > 0 && ndb) {
                L.d_blobs = (uint8_t*) ndb;
                L.d_blobs_from_pool = ndb_from_pool;
                if (!ndb_from_pool) m->gpu_allocs.push_back(L.d_blobs);
                if (!L.d_slot_of) {
                    CK(cudaMalloc(&L.d_slot_of, (size_t) ne * sizeof(int32_t)));
                    m->gpu_allocs.push_back(L.d_slot_of);
                }
                L.n_slots = s;
                L.exp_in_slot.assign((size_t) s, -1);
                const auto& keep = keep_list[(size_t) ip];
                auto& h2d = pending_h2d[(size_t) ip];
                for (int k = 0; k < s; ) {
                    int k1 = k + 1;
                    while (k1 < s && keep[(size_t) k1] == keep[(size_t) (k1 - 1)] + 1) ++k1;
                    h2d.push_back(H2dRun{keep[(size_t) k], k1 - k, k});
                    for (int j = k; j < k1; ++j) {
                        const int e = keep[(size_t) j];
                        L.slot_of[(size_t) e] = j;
                        L.exp_in_slot[(size_t) j] = e;
                    }
                    k = k1;
                }
            }
        }
        for (int ip = 0; ip < hp.n_phys; ++ip) {
            if (!layer_resized[(size_t) ip]) continue;
            upload_slot_of(ip, st);
            rebal_burst = 2;
        }

        // ---- Pass 3: Priority-ordered coalesced H2D on cst (dspark decode_w -> blk.0..5 -> core -> blk.18..23) ----
        const bool do_async_cst = (async_cst && cst != nullptr);
        cudaStream_t h2d_stream = cst ? cst : st;
        if (any_pool_alloc && h2d_stream != st && ev_pool_ready) {
            CK(cudaEventRecord(ev_pool_ready, st));
            CK(cudaStreamWaitEvent(h2d_stream, ev_pool_ready, 0));
        }
        if (restore_dspark_decode_w && dspark) {
            dspark->queue_decode_weights_h2d(h2d_stream, do_async_cst, pool_active ? phase_pool : nullptr, st);
        }
        for (int ip = 0; ip < hp.n_phys; ++ip) {
            const auto& h2d = pending_h2d[(size_t) ip];
            if (h2d.empty()) continue;
            PhysLayer& L = m->layers[(size_t) ip];
            for (const H2dRun& r : h2d) {
                if (L.hblobs) {
                    const size_t bytes = (size_t) (r.cnt - 1) * L.stride + L.L.bytes;
                    copy_h2d_staged(L.d_blobs + (size_t) r.dst_off * L.stride,
                                    L.hblobs + (size_t) r.e0 * L.stride,
                                    bytes, L.hblobs_pinned, h2d_stream);
                } else {
                    for (int j = 0; j < r.cnt; ++j) {
                        const uint8_t* src_e = get_or_build_expert_hblob(L, r.e0 + j, ne);
                        copy_h2d_staged(L.d_blobs + (size_t) (r.dst_off + j) * L.stride,
                                        src_e, L.L.bytes, false, h2d_stream);
                    }
                }
            }
            if (do_async_cst && ip < kMaxRebalLayers) {
                CK(cudaEventRecord(ev_rebal_layer[ip], cst));
                layer_pending_rebal[ip] = true;
            }
        }
        if (!do_async_cst) {
            if (cst) CK(cudaStreamSynchronize(cst));
            CK(cudaStreamSynchronize(st));
            for (int i = 0; i < kMaxRebalLayers; ++i) layer_pending_rebal[i] = false;
            if (dspark && dspark->dm) dspark->dm->decode_w_h2d_pending = false;
        }

        // Recompute residency state & lazily initialize CPU pool/buffers if transitioning from Full-VRAM -> Hybrid
        int tot_slots = 0;
        size_t exp_bytes = 0;
        bool any_missing = false;
        bool all_core = !tier1.empty();
        for (int ip = 0; ip < hp.n_phys; ++ip) {
            const PhysLayer& L = m->layers[(size_t) ip];
            tot_slots += L.n_slots;
            exp_bytes += (size_t) L.n_slots * L.stride;
            if (L.n_slots < ne) any_missing = true;
            if (ip >= hp.core_start && ip < hp.core_end && L.n_slots < ne) all_core = false;
        }
        hybrid = any_missing;
        core_all_gpu = all_core;
#if defined(STRATA_INKLING2_CPU_POOL)
        if (hybrid && !pool) {
            int nw = cpu_threads_req > 0 ? cpu_threads_req : 0;
            pool = std::make_unique<strata::kernels::cpu::ExpertPool>(nw, true, true);
            pool_workers = pool->workers();
            pin_host_for_pool();
        }
#endif
        if (hybrid && !h_h) {
            const int cpu_t = (pp_stage[0] && pp_stream_t > 1) ? std::min(tmax, std::max(pp_stream_t - 1, kDecodePhaseCap)) : tmax;
            init_cpu_bufs(cpu_t);
        }
        placed_slots = tot_slots;
        placed_bytes = exp_bytes;
        if (exp_bytes > full_cache_bytes) full_cache_bytes = exp_bytes;
        if (any_pool_alloc || !deferred_graphs_destroy.empty()) any_raw_vram_op = any_raw_vram_op || !deferred_graphs_destroy.empty();
        if (!(skip_sync && known_tot > 0 && use_phase_pool && !any_raw_vram_op)) {
            CK(cudaMemGetInfo(&fr, &tot));
        }
        vram_free_after_place = fr;
        std::fprintf(stderr, "strata-inkling2: VRAM resize(target=%.0f MiB, async=%d) -> slots=%d/%lld (%.0f/%.0f MiB, free=%.0f MiB, hybrid=%d) in %.1f ms\n",
                     (double) target_free_bytes / (1 << 20), do_async_cst ? 1 : 0, tot_slots, (long long) hp.n_phys * ne,
                     (double) exp_bytes / (1 << 20), (double) full_cache_bytes / (1 << 20),
                     (double) fr / (1 << 20), hybrid ? 1 : 0, ms_since(t0));
    }

    void normalize_decode_scores() {
        static const double prior = [] {
            const char* v = std::getenv("STRATA_DECODE_SCORE_PRIOR");
            return v ? std::atof(v) : 4.0;
        }();
        decode_scores_normalized = true;
        if (!hybrid || prior < 0.0) return;
        const int ne = (int) m->hp.n_expert;
        for (PhysLayer& L : m->layers) {
            if (L.score.empty()) continue;
            if (prior == 0.0) {
                std::fill(L.score.begin(), L.score.end(), 0.0);
                continue;
            }
            double mx = 0.0;
            for (int e = 0; e < ne; ++e) {
                if (L.score[(size_t) e] > mx) mx = L.score[(size_t) e];
            }
            if (mx > prior) {
                const double scale = prior / mx;
                for (int e = 0; e < ne; ++e) L.score[(size_t) e] *= scale;
            }
        }
    }

    /// Phase-split Decode transition: free Prefill-only scratch (chunk arena, ephemeral_pool, pp_stage_pool),
    /// compact SWA KV pool (5,120 -> 1,048 -> +323 MiB!), batch-alloc & async-stream DSpark decode weights +
    /// hottest Tier-1/Tier-2 experts on cst while Decode starts immediately!
    void set_phase_decode() {
        static const bool disable_ps = (std::getenv("STRATA_NO_PHASE_SPLIT") != nullptr);
        if (disable_ps || !hybrid || in_decode_phase || tmax <= kDecodePhaseCap || tmax < 1024) return;
        CK(cudaStreamSynchronize(st));
        if (cst) CK(cudaStreamSynchronize(cst));
        if (aux_st) CK(cudaStreamSynchronize(aux_st));
        for (int i = 0; i < kMaxRebalLayers; ++i) layer_pending_rebal[i] = false;
        destroy_graphs();

        // 1. Free pp_stage_pool (single cudaFreeAsync!)
        saved_pp_stage_cnt = pp_stage_cnt;
        saved_pp_stage_bytes = pp_stage_bytes;
        if (transient_pp_pool) { pool_free(transient_pp_pool); transient_pp_pool = nullptr; }
        transient_pp_cnt = 0;
        free_pp_stages();

        // 2. Shrink ephemeral activation scratch (active_pp_cap -> kDecodePhaseCap)
        free_ephemeral_bufs();
        decode_ephemeral_bytes = alloc_ephemeral_bufs(kDecodePhaseCap);
        alloc_conv_trace();

        // 3. EX-4A / Wave 9: Compact SWA KV ring only when shared-scratch SWA is disabled (otherwise ring stays fixed at swa_ring_cap!)
        if (!use_shared_swa_scratch) {
            const int dec_swa_cap = (int) m->hp.n_swa + kDecodePhaseCap;
            if (swa_kv_cap > dec_swa_cap) {
                resize_swa_kv_pool(dec_swa_cap);
            }
        }

        // 4. EX-4A / EX-5 Idea D: Rebind DSpark arena; defer decode_w alloc & priority H2D to resize_experts_vram Pass 2/3!
        if (dspark) {
            dspark->rebind_arena(layer_arena, layer_arena_bytes, kDecodePhaseCap);
        }
        in_decode_phase = true;

        // 5. D2D-shrink frontloaded layers, batch-alloc, and stream DSpark decode_w + knapsack-ranked experts async on cst!
        const size_t dec_guard = (verify_max_t > 1) ? ((size_t) 176 << 20) : ((size_t) 144 << 20);
        static const bool no_dec_knapsack = (std::getenv("STRATA_NO_DECODE_KNAPSACK") != nullptr);
        resize_experts_vram(vram_reserve_bytes, dec_guard, /*equalize_tier2=*/true, /*async_cst=*/true,
                            /*whole_layer_evict=*/false, /*restore_dspark_decode_w=*/(dspark != nullptr),
                            /*skip_sync=*/false, 0, 0, /*knapsack_decode=*/!no_dec_knapsack);
        if (verify_max_t >= 6 && pool_cached_free_bytes() > ((size_t) 8 << 20)) trim_phase_pool(0);
        normalize_decode_scores();
    }

    /// Phase-split Prefill transition: evict DSpark decode-only weights (+358 MiB!), zero-H2D whole-layer evict
    /// decode-bonus experts (~1.5 ms), re-expand SWA KV pool, and restore active_pp_cap prefill scratch + pp_stage_pool.
    void set_phase_prefill(bool streams_already_synced = false, size_t known_fr = 0, size_t known_tot = 0) {
        if (!in_decode_phase) return;
        const bool was_knapsack_dec = knapsack_decode_active;
        knapsack_decode_active = false;
        if (!streams_already_synced) {
            CK(cudaStreamSynchronize(st));
            if (cst) CK(cudaStreamSynchronize(cst));
            if (aux_st) CK(cudaStreamSynchronize(aux_st));
            defer_destroy_graphs();
        }
        for (int i = 0; i < kMaxRebalLayers; ++i) layer_pending_rebal[i] = false;

        // 1. EX-4A + Wave 6: Evict DSpark decode-only weights FIRST (+358 MiB immediately back into phase_pool!)
        if (dspark) dspark->evict_decode_weights(use_phase_pool ? phase_pool : nullptr, st);
        free_conv_trace();

        // 2. EX-5 Idea C + Wave 6: Zero-H2D whole-layer evict of tail Tier-2 experts enough to fit peak(SWA, eph + net_swa + stage)
        int tgt_cap = active_pp_cap;
        const size_t target_eph = estimate_ephemeral_total_bytes(tgt_cap);
        const size_t extra_eph = (target_eph > decode_ephemeral_bytes)
            ? (target_eph - decode_ephemeral_bytes) : 0;
        prefill_swa_kv_cap = use_shared_swa_scratch
            ? swa_ring_cap()
            : ((tgt_cap <= 1) ? (int) m->hp.n_swa : (int) (m->hp.n_swa + tgt_cap));
        const size_t target_swa_bytes = use_shared_swa_scratch ? 0 : swa_kv_pool_size_for(prefill_swa_kv_cap);
        const size_t net_swa = (!use_shared_swa_scratch && target_swa_bytes > swa_kv_pool_bytes)
            ? (target_swa_bytes - swa_kv_pool_bytes) : 0;
        size_t core_restore_b = 0;
        if (was_knapsack_dec && profile.regime == strata::inkling2::VramRegime::CoreResidentHybrid) {
            const int ne_c = (int) m->hp.n_expert;
            for (int ip = (int) m->hp.core_start; ip < (int) m->hp.core_end && ip < (int) m->hp.n_phys; ++ip) {
                const PhysLayer& L = m->layers[(size_t) ip];
                if (L.gu_type != 30 && L.n_slots < ne_c) {
                    core_restore_b += (size_t) (ne_c - L.n_slots) * L.stride;
                }
            }
        }
        const bool core_will_be_full = is_core_full_resident() ||
            (was_knapsack_dec && profile.regime == strata::inkling2::VramRegime::CoreResidentHybrid);
        if (!core_will_be_full && (saved_pp_stage_bytes > ((size_t) 66 << 20) || saved_pp_stage_cnt > 2)) {
            size_t compact_bytes = (size_t) 64 << 20;
            for (int ip = 0; ip < (int) m->layers.size(); ++ip) {
                const auto& L = m->layers[(size_t) ip];
                if (L.gu_type == 40 && L.stride > 0) {
                    const size_t nm = ((compact_bytes + L.stride - 1) / L.stride) * L.stride;
                    if (nm > compact_bytes && nm - compact_bytes <= ((size_t) 256 << 10)) compact_bytes = nm;
                    break;
                }
            }
            if (saved_pp_stage_bytes > 0) saved_pp_stage_bytes = std::min(saved_pp_stage_bytes, compact_bytes);
            if (saved_pp_stage_cnt > 0) saved_pp_stage_cnt = std::min(saved_pp_stage_cnt, 2);
        }
        size_t stage_need = (size_t) saved_pp_stage_cnt * saved_pp_stage_bytes;
        const size_t min_stage_need = (size_t) std::min(saved_pp_stage_cnt, 2) * saved_pp_stage_bytes;
        int want_stage_cnt = saved_pp_stage_cnt;
        bool have_mem_info = (known_tot > 0 && use_phase_pool && phase_pool != nullptr);
        size_t fr_sp = have_mem_info ? known_fr : 0;
        size_t tot_sp = have_mem_info ? known_tot : 0;
        size_t cached_pool_sp = have_mem_info ? pool_cached_free_bytes() : 0;
        if (saved_pp_stage_cnt > 2 && saved_pp_stage_bytes > 0) {
            if (!have_mem_info) {
                CK(cudaMemGetInfo(&fr_sp, &tot_sp));
                cached_pool_sp = pool_cached_free_bytes();
                have_mem_info = true;
            }
            size_t t2_b = 0;
            const auto& hp_sp = m->hp;
            for (int ip = 0; ip < hp_sp.n_phys; ++ip) {
                if (ip >= hp_sp.core_start && ip < hp_sp.core_end) continue;
                t2_b += (size_t) m->layers[(size_t) ip].n_slots * m->layers[(size_t) ip].stride;
            }
            const size_t avail_sp = fr_sp + cached_pool_sp + t2_b;
            want_stage_cnt = 2;
            stage_need = min_stage_need;
            for (int sc = saved_pp_stage_cnt; sc >= 2; --sc) {
                const size_t sn = (size_t) sc * saved_pp_stage_bytes;
                const size_t full_peak = std::max(target_swa_bytes, extra_eph + net_swa + sn);
                if (avail_sp >= vram_reserve_bytes + full_peak + ((size_t) 64 << 20)) {
                    want_stage_cnt = sc;
                    stage_need = sn;
                    break;
                }
            }
        }
        const size_t peak_growth = std::max(target_swa_bytes, extra_eph + net_swa + stage_need) + core_restore_b;
        size_t need_free = vram_reserve_bytes + peak_growth + ((size_t) 64 << 20);
        // Reclaim scattered non-blk.0 Tier-2 slots on CoreResidentHybrid (>=4K prefill) so Turn 2
        // maybe_alloc_transient_pp_stages() gets trans = 3 (NEVER touches blk.0 128/128, Core 1536/1536, or TightHybrid/262K).
        if (core_will_be_full &&
            tgt_cap >= 4096 && use_phase_pool && phase_pool != nullptr && saved_pp_stage_bytes > 0) {
            if (!have_mem_info) {
                CK(cudaMemGetInfo(&fr_sp, &tot_sp));
                cached_pool_sp = pool_cached_free_bytes();
                have_mem_info = true;
            }
            const size_t eff_fr_t2 = fr_sp + cached_pool_sp;
            size_t non_blk0_t2_b = 0;
            const auto& hp_t2 = m->hp;
            for (int ip = 1; ip < hp_t2.n_phys; ++ip) {
                if (ip >= hp_t2.core_start && ip < hp_t2.core_end) continue;
                non_blk0_t2_b += (size_t) m->layers[(size_t) ip].n_slots * m->layers[(size_t) ip].stride;
            }
            if (non_blk0_t2_b > ((size_t) 1 << 20)) {
                const size_t extra_trans_need = std::min(non_blk0_t2_b, (size_t) 3 * saved_pp_stage_bytes);
                const size_t safe_cap = eff_fr_t2 + non_blk0_t2_b - ((size_t) 1 << 20);
                if (safe_cap > need_free) {
                    need_free = std::min(need_free + extra_trans_need, safe_cap);
                }
            }
        }
        resize_experts_vram(need_free, 0, /*equalize_tier2=*/false, /*async_cst=*/false, /*whole_layer_evict=*/true,
                            /*restore_dspark_decode_w=*/false, /*skip_sync=*/true,
                            have_mem_info ? fr_sp : 0, have_mem_info ? tot_sp : 0);

        // 3. Re-expand SWA KV ring only when shared-scratch SWA is disabled
        if (!use_shared_swa_scratch && swa_kv_cap < prefill_swa_kv_cap) {
            resize_swa_kv_pool(prefill_swa_kv_cap);
        }

        // 4. Restore active_pp_cap prefill scratch & DSpark arena via stream-ordered phase_pool
        free_ephemeral_bufs();
        const bool did_trim_post = (tgt_cap > 8192 || (!core_will_be_full && tgt_cap >= 4096));
        if (did_trim_post) trim_phase_pool(0);
        if (tgt_cap > tmax) {
            const size_t slab_pad = (size_t) 80 << 20;
            size_t avail_post = (have_mem_info && use_phase_pool && use_shared_swa_scratch &&
                                 !did_trim_post && swa_kv_cap >= prefill_swa_kv_cap)
                ? (fr_sp + pool_cached_free_bytes()) : 0;
            if (avail_post == 0 || avail_post < estimate_ephemeral_total_bytes(tgt_cap) + slab_pad) {
                size_t fr_post = 0, tot_post = 0;
                CK(cudaMemGetInfo(&fr_post, &tot_post));
                avail_post = fr_post + pool_cached_free_bytes();
            }
            const int orig_cap = tgt_cap;
            while (tgt_cap > tmax) {
                const size_t need_post = estimate_ephemeral_total_bytes(tgt_cap) + slab_pad;
                if (avail_post >= need_post) break;
                const int next_cap = (tgt_cap > 8192 && 8192 > tmax)
                    ? 8192
                    : ((tgt_cap > 4096 && 4096 > tmax) ? 4096 : tmax);
                std::fprintf(stderr, "strata-inkling2: set_phase_prefill post-trim VRAM %.1f MiB < %.1f MiB needed for cap=%d; stepping down to %d\n",
                             (double) avail_post / (1 << 20), (double) need_post / (1 << 20), tgt_cap, next_cap);
                tgt_cap = next_cap;
            }
            if (tgt_cap != orig_cap) {
                active_pp_cap = tgt_cap;
                const int new_swa_cap = use_shared_swa_scratch
                    ? swa_ring_cap()
                    : ((tgt_cap <= 1) ? (int) m->hp.n_swa : (int) (m->hp.n_swa + tgt_cap));
                if (!use_shared_swa_scratch && swa_kv_cap != new_swa_cap) {
                    resize_swa_kv_pool(new_swa_cap);
                }
                prefill_swa_kv_cap = new_swa_cap;
            }
        }
        prefill_ephemeral_bytes = alloc_ephemeral_bufs(tgt_cap);
        if (dspark) dspark->rebind_arena(layer_arena, layer_arena_bytes, tgt_cap);

        // 5. Restore pp_stage_pool in ONE pool_alloc (retry after tier shrink on tight 262K VRAM)
        if (want_stage_cnt > 0 && saved_pp_stage_bytes > 0) {
            if (!alloc_pp_stages(want_stage_cnt, saved_pp_stage_bytes)) {
                trim_phase_pool(0);
                if (d_core_wdown_cur_f16) { cudaFree(d_core_wdown_cur_f16); d_core_wdown_cur_f16 = nullptr; }
                if (d_core_wup_f16) { cudaFree(d_core_wup_f16); d_core_wup_f16 = nullptr; }
                size_t fr_st = 0, tot_st = 0;
                CK(cudaMemGetInfo(&fr_st, &tot_st));
                (void) fr_st; (void) tot_st;
                const size_t pool_st = pool_cached_free_bytes();
                const size_t min_st_need = pool_st + vram_reserve_bytes + (size_t) 2 * saved_pp_stage_bytes + ((size_t) 128 << 20);
                resize_experts_vram(min_st_need, 0, /*equalize_tier2=*/false, /*async_cst=*/false,
                                    /*whole_layer_evict=*/false, /*restore_dspark_decode_w=*/false, /*skip_sync=*/false);
                trim_phase_pool(0);
                if (!alloc_pp_stages(2, saved_pp_stage_bytes)) {
                    size_t compact_bytes = (size_t) 64 << 20;
                    for (int ip = 0; ip < (int) m->layers.size(); ++ip) {
                        const auto& L = m->layers[(size_t) ip];
                        if (L.gu_type == 40 && L.stride > 0) {
                            const size_t nm = ((compact_bytes + L.stride - 1) / L.stride) * L.stride;
                            if (nm > compact_bytes && nm - compact_bytes <= ((size_t) 256 << 10)) compact_bytes = nm;
                            break;
                        }
                    }
                    trim_phase_pool(0);
                    alloc_pp_stages(2, compact_bytes);
                }
                if (pp_stage_cnt > 0) {
                    saved_pp_stage_cnt = pp_stage_cnt;
                    saved_pp_stage_bytes = pp_stage_bytes;
                }
            }
        }
        in_decode_phase = false;
        if (was_knapsack_dec && !is_core_full_resident()) {
            // Restore Core 3x-loop layers first from pinned/sparse hblobs before Prefill pp_streaming begins.
            resize_experts_vram(vram_reserve_bytes + ((size_t) 64 << 20), 0, /*equalize_tier2=*/false,
                                /*async_cst=*/false, /*whole_layer_evict=*/false,
                                /*restore_dspark_decode_w=*/false, /*skip_sync=*/true,
                                0, 0, /*knapsack_decode=*/false, /*tier1_only=*/true);
        }
        for (PhysLayer& L : m->layers) {
            if (L.n_slots >= (int) m->hp.n_expert) free_sparse_hblobs(L);
        }
        if (!is_core_full_resident() && pool_cached_free_bytes() > ((size_t) 8 << 20)) trim_phase_pool(0);
    }

    /// Wave 5 + Wave 11 Path B: 4-tier prompt-adaptive prefill slab capacity {24, 4096, 8192, 12288}.
    /// Promotes to 8192 / 12288 ONLY on CoreResidentHybrid when Tier-1 core stays 100% resident (never touches core or 262K TightHybrid),
    /// with automatic safe step-down 12288 -> 8192 -> tmax as kv_cur grows or staging slots tighten.
    int ensure_pp_cap(int want_cap, int prompt_total_len = 0) {
        static const bool no_adaptive_chunk = (std::getenv("STRATA_NO_ADAPTIVE_CHUNK") != nullptr);
        static const bool no_vmsg = (std::getenv("STRATA_NO_VMSG") != nullptr);
        static const bool no_dspark_vram_acc = (std::getenv("STRATA_NO_DSPARK_VRAM_ACC") != nullptr);
        const auto& hp = m->hp;
        const int core_experts_total = (hp.num_loops > 1 && hp.core_end > hp.core_start)
            ? (int) ((hp.core_end - hp.core_start) * hp.n_expert) : 0;
        int core_experts_res = 0;
        for (int ip = (int) hp.core_start; ip < (int) hp.core_end && ip < hp.n_phys; ++ip) {
            core_experts_res += m->layers[(size_t) ip].n_slots;
        }
        const bool core_res_hy_knap = (in_decode_phase && knapsack_decode_active &&
                                       profile.regime == strata::inkling2::VramRegime::CoreResidentHybrid);
        const bool core_full_ok = (core_all_gpu && (core_experts_total <= 0 || core_experts_res >= core_experts_total)) ||
                                  core_res_hy_knap;
        int target_cap = std::clamp(want_cap, tmax, meta_tmax);
        if (!core_full_ok) {
            target_cap = std::max(tmax, std::min(target_cap, 4096));
        }
        if (no_adaptive_chunk || profile.max_adaptive_chunk <= tmax || !hybrid ||
            (dspark && (no_dspark_vram_acc || no_vmsg || !can_vmsg_all_core))) {
            target_cap = tmax;
        }
        if (!in_decode_phase && target_cap == active_pp_cap && prompt_total_len <= full_kv_cap) return active_pp_cap;
        size_t fr_cap = 0, tot_cap = 0;
        bool synced_in_cap = false;
        if (target_cap > tmax) {
            if (in_decode_phase) {
                CK(cudaStreamSynchronize(st));
                if (cst) CK(cudaStreamSynchronize(cst));
                if (aux_st) CK(cudaStreamSynchronize(aux_st));
                defer_destroy_graphs();
                if (dspark) dspark->defer_destroy_decode_graphs();
                synced_in_cap = true;
            }
            const size_t cur_eph = in_decode_phase ? decode_ephemeral_bytes : prefill_ephemeral_bytes;
            size_t kv_delta = 0;
            if (prompt_total_len > full_kv_cap) {
                int n_full = 0;
                for (int il = 0; il < hp.n_layer; ++il) if (!hp.is_swa(il)) ++n_full;
                if (dspark) n_full += dspark->count_full_layers();
                const size_t per_tok = kv_q8
                    ? ((size_t) hp.kv_dim * 2 + (size_t) hp.n_head_kv * sizeof(__half) * 2)
                    : ((size_t) hp.kv_dim * sizeof(__half) * 2);
                const bool use_headroom = (profile.full_kv_init <= 16384 && ctx <= 262144);
                const int target_kv = next_kv_ladder_cap(full_kv_cap, prompt_total_len, ctx, use_headroom);
                kv_delta = (size_t) n_full * (size_t) std::max(0, target_kv - full_kv_cap) * per_tok;
            }
            CK(cudaMemGetInfo(&fr_cap, &tot_cap));
            const size_t fr = fr_cap;
            size_t tier2_bytes = 0;
            for (int ip = 0; ip < hp.n_phys; ++ip) {
                if (ip >= hp.core_start && ip < hp.core_end) continue;
                tier2_bytes += (size_t) m->layers[(size_t) ip].n_slots * m->layers[(size_t) ip].stride;
            }
            constexpr int kMinCoreSlotsFor4K = 64;
            size_t tier1_borrow_bytes = 0;
            for (int ip = (int) hp.core_start; ip < (int) hp.core_end && ip < hp.n_phys; ++ip) {
                const PhysLayer& L = m->layers[(size_t) ip];
                if (L.n_slots > kMinCoreSlotsFor4K) {
                    tier1_borrow_bytes += (size_t) (L.n_slots - kMinCoreSlotsFor4K) * L.stride;
                }
            }
            const size_t dspark_evict_b = (dspark && dspark->dm &&
                                           dspark->dm->decode_w_offloadable && dspark->dm->decode_w_resident)
                                          ? dspark->dm->decode_w_bytes : 0;
            size_t core_restore_b = 0;
            if (core_res_hy_knap) {
                const int ne_c = (int) hp.n_expert;
                for (int ip = (int) hp.core_start; ip < (int) hp.core_end && ip < hp.n_phys; ++ip) {
                    const PhysLayer& L = m->layers[(size_t) ip];
                    if (L.gu_type != 30 && L.n_slots < ne_c) {
                        core_restore_b += (size_t) (ne_c - L.n_slots) * L.stride;
                    }
                }
            }
            const size_t avail_no_core = (fr + pool_cached_free_bytes() + tier2_bytes + dspark_evict_b > core_restore_b)
                ? (fr + pool_cached_free_bytes() + tier2_bytes + dspark_evict_b - core_restore_b) : 0;
            const size_t extra_margin = (kv_delta > 0)
                ? ((size_t) std::max(64, profile.kv_grow_reserve_mib) << 20)
                : ((size_t) 80 << 20);
            const int avail_stage_cnt = in_decode_phase ? saved_pp_stage_cnt : pp_stage_cnt;
            const size_t avail_stage_bytes = in_decode_phase ? saved_pp_stage_bytes : pp_stage_bytes;

            int cands[4];
            int n_cands = 0;
            cands[n_cands++] = target_cap;
            if (target_cap > 8192 && 8192 > tmax) cands[n_cands++] = 8192;
            if (target_cap > 4096 && 4096 > tmax) cands[n_cands++] = 4096;
            cands[n_cands++] = tmax;

            int chosen_cap = tmax;
            for (int ci = 0; ci < n_cands - 1; ++ci) {
                const int cand_cap = cands[ci];
                if (cand_cap > 4096 && !core_full_ok) continue;
                if (cand_cap > 8192 && avail_stage_bytes > 0 && avail_stage_cnt < 3) continue;
                const size_t tgt_eph = estimate_ephemeral_total_bytes(cand_cap);
                const size_t extra_eph = (tgt_eph > cur_eph) ? (tgt_eph - cur_eph) : 0;
                const size_t freed_eph = (cur_eph > tgt_eph) ? (cur_eph - tgt_eph) : 0;
                const int tgt_swa_cap = use_shared_swa_scratch ? swa_ring_cap() : ((int) hp.n_swa + cand_cap);
                const size_t tgt_swa_b = use_shared_swa_scratch ? 0 : swa_kv_pool_size_for(tgt_swa_cap);
                const size_t net_swa = (!use_shared_swa_scratch && tgt_swa_b > swa_kv_pool_bytes)
                    ? (tgt_swa_b - swa_kv_pool_bytes) : 0;
                const size_t freed_swa = (!use_shared_swa_scratch && swa_kv_pool_bytes > tgt_swa_b)
                    ? (swa_kv_pool_bytes - tgt_swa_b) : 0;
                const int req_stages_dec = (cand_cap > 8192)
                    ? std::min(saved_pp_stage_cnt, 3)
                    : std::min(saved_pp_stage_cnt, 2);
                const size_t min_stage_need = in_decode_phase
                    ? ((size_t) req_stages_dec * saved_pp_stage_bytes) : 0;
                const size_t swa_peak = (use_shared_swa_scratch || tgt_swa_cap == swa_kv_cap) ? 0 : tgt_swa_b;
                const size_t peak_growth = std::max(swa_peak, extra_eph + net_swa + min_stage_need);
                const size_t cand_margin = (cand_cap > 8192)
                    ? std::max(extra_margin, (size_t) 96 << 20)
                    : extra_margin;
                const size_t need_tot = peak_growth + kv_delta + vram_reserve_bytes + cand_margin;
                if (avail_no_core + freed_eph + freed_swa >= need_tot) {
                    chosen_cap = cand_cap;
                    break;
                }
                if (cand_cap <= 4096 && cand_cap > tmax && core_experts_total > 0) {
                    const size_t borrow_margin = std::max(cand_margin, (size_t) 112 << 20);
                    const size_t need_borrow = peak_growth + kv_delta + vram_reserve_bytes + borrow_margin;
                    if (avail_no_core + tier1_borrow_bytes + freed_eph + freed_swa >= need_borrow) {
                        chosen_cap = cand_cap;
                        break;
                    }
                }
            }
            target_cap = chosen_cap;
        }
        if (in_decode_phase) {
            active_pp_cap = target_cap;
            set_phase_prefill(synced_in_cap, fr_cap, tot_cap);
            return active_pp_cap;
        }
        if (target_cap == active_pp_cap) return active_pp_cap;

        CK(cudaStreamSynchronize(st));
        if (cst) CK(cudaStreamSynchronize(cst));
        if (aux_st) CK(cudaStreamSynchronize(aux_st));
        for (int i = 0; i < kMaxRebalLayers; ++i) layer_pending_rebal[i] = false;
        defer_destroy_graphs();

        int tgt_swa_cap = use_shared_swa_scratch
            ? swa_ring_cap()
            : ((target_cap <= 1) ? (int) hp.n_swa : (int) (hp.n_swa + target_cap));
        if (target_cap > active_pp_cap) {
            // Evict idle DSpark decode weights before resize_experts_vram so Tier-1 Core is never touched!
            if (dspark && dspark->dm && dspark->dm->decode_w_offloadable && dspark->dm->decode_w_resident) {
                dspark->evict_decode_weights(use_phase_pool ? phase_pool : nullptr, st);
            }
            const size_t tgt_eph = estimate_ephemeral_total_bytes(target_cap);
            const size_t extra_eph = (tgt_eph > prefill_ephemeral_bytes) ? (tgt_eph - prefill_ephemeral_bytes) : 0;
            const size_t tgt_swa_b = use_shared_swa_scratch ? 0 : swa_kv_pool_size_for(tgt_swa_cap);
            const size_t net_swa = (!use_shared_swa_scratch && tgt_swa_b > swa_kv_pool_bytes)
                ? (tgt_swa_b - swa_kv_pool_bytes) : 0;
            // Free old ephemeral BEFORE resizing SWA KV pool so peak growth is max(tgt_swa - cur_eph, extra_eph + net_swa)!
            const size_t swa_peak = (use_shared_swa_scratch || tgt_swa_cap == swa_kv_cap || tgt_swa_b <= prefill_ephemeral_bytes)
                ? 0 : (tgt_swa_b - prefill_ephemeral_bytes);
            const size_t peak_growth = std::max(swa_peak, extra_eph + net_swa);
            const size_t grow_pad = (target_cap > 8192) ? ((size_t) 128 << 20) : ((size_t) 96 << 20);
            const size_t need_free = vram_reserve_bytes + peak_growth + grow_pad;
            resize_experts_vram(need_free, 0, /*equalize_tier2=*/false, /*async_cst=*/false, /*whole_layer_evict=*/true,
                                /*restore_dspark_decode_w=*/false, /*skip_sync=*/true);
            free_ephemeral_bufs();
            trim_phase_pool(0);
            // Wave 12B: Verify actual post-trim free VRAM before committing large slabs; step down cleanly instead of OOMing.
            if (target_cap > tmax) {
                size_t fr_post = 0, tot_post = 0;
                CK(cudaMemGetInfo(&fr_post, &tot_post));
                const size_t avail_post = fr_post + pool_cached_free_bytes();
                const size_t slab_pad = (size_t) 80 << 20;
                while (target_cap > tmax) {
                    const int cur_swa_c = use_shared_swa_scratch ? swa_ring_cap() : ((int) hp.n_swa + target_cap);
                    const size_t cur_swa_b = use_shared_swa_scratch ? 0 : swa_kv_pool_size_for(cur_swa_c);
                    const size_t cur_swa_need = (!use_shared_swa_scratch && cur_swa_b > swa_kv_pool_bytes)
                        ? (cur_swa_b - swa_kv_pool_bytes) : 0;
                    const size_t need_post = estimate_ephemeral_total_bytes(target_cap) + cur_swa_need + slab_pad;
                    if (avail_post >= need_post) break;
                    const int next_cap = (target_cap > 8192 && 8192 > tmax)
                        ? 8192
                        : ((target_cap > 4096 && 4096 > tmax) ? 4096 : tmax);
                    std::fprintf(stderr, "strata-inkling2: post-trim VRAM %.1f MiB < %.1f MiB needed for cap=%d; stepping down to %d\n",
                                 (double) avail_post / (1 << 20), (double) need_post / (1 << 20), target_cap, next_cap);
                    target_cap = next_cap;
                }
                tgt_swa_cap = use_shared_swa_scratch
                    ? swa_ring_cap()
                    : ((target_cap <= 1) ? (int) hp.n_swa : (int) (hp.n_swa + target_cap));
            }
            if (!use_shared_swa_scratch && swa_kv_cap != tgt_swa_cap) resize_swa_kv_pool(tgt_swa_cap);
            prefill_ephemeral_bytes = alloc_ephemeral_bufs(target_cap);
            // Release leftover pool cache back to the CUDA driver so first-use cuBLAS/graph internal
            // allocations do not fail with CUBLAS_STATUS_INTERNAL_ERROR when raw driver free is tight.
            if (verify_max_t >= 6 && pool_cached_free_bytes() > ((size_t) 8 << 20)) trim_phase_pool(0);
        } else {
            free_ephemeral_bufs();
            prefill_ephemeral_bytes = alloc_ephemeral_bufs(target_cap);
            if (!use_shared_swa_scratch && swa_kv_cap != tgt_swa_cap) resize_swa_kv_pool(tgt_swa_cap);
        }
        if (dspark) dspark->rebind_arena(layer_arena, layer_arena_bytes, target_cap);
        active_pp_cap = target_cap;
        prefill_swa_kv_cap = tgt_swa_cap;
        std::fprintf(stderr, "strata-inkling2: adaptive prefill cap -> %d (swa_ring=%d, swa_pp=%d, pp_stages=%d, free_pool=%.1f MiB)\n",
                     active_pp_cap, swa_kv_cap, swa_pp_cap, pp_stage_cnt, (double) pool_cached_free_bytes() / (1 << 20));
        return active_pp_cap;
    }

    static int next_kv_ladder_cap(int cur_cap, int needed, int max_ctx, bool use_headroom_ladder) {
        if (needed <= cur_cap || cur_cap >= max_ctx) return cur_cap;
        if (use_headroom_ladder) {
            static constexpr int kLadder[] = {36864, 69632, 102400, 139264, 172032, 204800, 262144};
            for (int c : kLadder) {
                const int clamped = std::min(max_ctx, c);
                if (clamped > cur_cap && clamped >= needed) return clamped;
            }
        }
        return std::min(max_ctx, std::max(needed, cur_cap * 2));
    }

    void ensure_kv_cap(int needed_pos, int cur_t = 1) {
        if (needed_pos <= full_kv_cap || full_kv_cap >= ctx) return;
        const auto& hp = m->hp;
        const bool use_headroom = (profile.full_kv_init <= 16384 && ctx <= 262144);
        const int new_cap = use_headroom
            ? next_kv_ladder_cap(full_kv_cap, needed_pos, ctx, true)
            : std::min(ctx, std::max(needed_pos,
                                     (profile.regime == strata::inkling2::VramRegime::TightHybrid &&
                                      profile.full_kv_init > 16384 && full_kv_cap < 65536)
                                     ? (full_kv_cap + 16384)
                                     : (full_kv_cap * 2)));
        if (new_cap <= full_kv_cap) return;

        CK(cudaStreamSynchronize(st));
        if (cst) CK(cudaStreamSynchronize(cst));
        if (aux_st) CK(cudaStreamSynchronize(aux_st));
        trim_phase_pool(0);
        for (int i = 0; i < kMaxRebalLayers; ++i) layer_pending_rebal[i] = false;

        // Full-attention KV buffers are reallocated below; invalidate all captured decode/verify graphs
        // and reset graph_oom so graphs cleanly re-capture at the new capacity.
        destroy_graphs();

        const int ne = (int) hp.n_expert;
        // Pull GPU routing hit counts into L.score for ALL layers so shrinking keeps the true hottest experts!
        sync_d_hits(false);

        int n_full = 0;
        for (int il = 0; il < hp.n_layer; ++il) {
            if (!hp.is_swa(il)) ++n_full;
        }
        const int n_dspark_full = dspark ? dspark->count_full_layers() : 0;
        const size_t per_tok = kv_q8
            ? ((size_t) hp.kv_dim * 2 + (size_t) hp.n_head_kv * sizeof(__half) * 2)
            : ((size_t) hp.kv_dim * sizeof(__half) * 2);
        const size_t extra_total = (size_t) (n_full + n_dspark_full) * (size_t) (new_cap - full_kv_cap) * per_tok;
        // Host-staged per-layer KV growth needs ZERO extra full-layer temp headroom (saves ~520 MiB at 262K = +122 core experts!).
        const size_t need_free = extra_total + ((size_t) std::max(64, profile.kv_grow_reserve_mib) << 20);

        auto demote_tier2_pin = [&](size_t need_bytes) -> bool {
            for (int j = (int) m->layers.size() - 1; j >= 0 && pinned_used + need_bytes > max_pin_bytes; --j) {
                if (j >= hp.core_start && j < hp.core_end) continue;
                PhysLayer& T2 = m->layers[(size_t) j];
                if (!T2.hblobs || !T2.hblobs_pinned) continue;
                const size_t tb = (size_t) ne * T2.stride;
                uint8_t* pag = (uint8_t*) std::malloc(tb);
                if (!pag) continue;
                std::memcpy(pag, T2.hblobs, tb);
                auto it = std::find(m->host_pinned_allocs.begin(), m->host_pinned_allocs.end(), (void*) T2.hblobs);
                if (it != m->host_pinned_allocs.end()) m->host_pinned_allocs.erase(it);
                cudaFreeHost(T2.hblobs);
                T2.hblobs = pag;
                T2.hblobs_pinned = false;
                m->host_allocs.push_back(pag);
                pinned_used = (pinned_used >= tb) ? (pinned_used - tb) : 0;
            }
            return pinned_used + need_bytes <= max_pin_bytes;
        };

        auto ensure_hblobs = [&](PhysLayer& L, bool is_tier1) {
            if (L.hblobs) return;
            const size_t hb_bytes = (size_t) ne * L.stride;
            bool want_pin = layer_allow_pin_hblobs(L);
            if (want_pin && pinned_used + hb_bytes > max_pin_bytes && is_tier1) {
                demote_tier2_pin(hb_bytes);
            }
            if (want_pin && pinned_used + hb_bytes <= max_pin_bytes &&
                cudaHostAlloc(&L.hblobs, hb_bytes, cudaHostAllocDefault) == cudaSuccess) {
                m->host_pinned_allocs.push_back(L.hblobs);
                L.hblobs_pinned = true;
                pinned_used += hb_bytes;
            } else {
                (void) cudaGetLastError();
                L.hblobs = (uint8_t*) std::malloc(hb_bytes);
                if (!L.hblobs) {
                    std::fprintf(stderr, "strata-inkling2: host malloc failed during KV growth\n");
                    std::exit(1);
                }
                L.hblobs_pinned = false;
                m->host_allocs.push_back(L.hblobs);
            }
            // Fast D2H clone: a 128/128 VRAM layer already holds exact packed blobs in L.d_blobs (slot_of[e] == e)!
            // Avoids 6.3 GiB cold disk mmap rebuild during 131K->262K KV expansion (~16 s -> ~1.5 s).
            bool can_d2h_clone = (L.n_slots == ne && L.d_blobs != nullptr);
            if (can_d2h_clone) {
                for (int e = 0; e < ne; ++e) {
                    if (L.slot_of[(size_t) e] != e) { can_d2h_clone = false; break; }
                }
            }
            if (can_d2h_clone) {
                CK(cudaMemcpyAsync(L.hblobs, L.d_blobs, hb_bytes, cudaMemcpyDeviceToHost, st));
                CK(cudaStreamSynchronize(st));
            } else {
                build_all_expert_blobs(L, ne, L.hblobs);
                drop_layer_mmap_pages(L, ne);
            }
        };

        auto shrink_layer = [&](int ip, int target_slots) -> bool {
            PhysLayer& L = m->layers[(size_t) ip];
            target_slots = std::clamp(target_slots, 0, ne);
            if (target_slots >= L.n_slots || !L.d_blobs) return false;
            const bool is_tier1 = (ip >= hp.core_start && ip < hp.core_end);
            ensure_hblobs(L, is_tier1);

            std::vector<int> keep;
            if (target_slots > 0) {
                std::vector<int> order((size_t) ne);
                for (int e = 0; e < ne; ++e) order[(size_t) e] = e;
                std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
                    const double sa = effective_score(L, a);
                    const double sb = effective_score(L, b);
                    if (sa != sb) return sa > sb;
                    return (L.slot_of[(size_t) a] >= 0) > (L.slot_of[(size_t) b] >= 0);
                });
                keep.assign(order.begin(), order.begin() + target_slots);
            }

            const bool was_from_pool = L.d_blobs_from_pool;
            free_layer_blobs(L);
            if (was_from_pool) trim_phase_pool(0);
            L.n_slots = 0;
            std::fill(L.slot_of.begin(), L.slot_of.end(), -1);
            L.exp_in_slot.clear();

            if (target_slots > 0) {
                void* ndb = nullptr;
                int s = target_slots;
                while (s > 0 && cudaMalloc(&ndb, (size_t) s * L.stride) != cudaSuccess) {
                    (void) cudaGetLastError();
                    --s;
                }
                if (s > 0 && ndb) {
                    L.d_blobs = (uint8_t*) ndb;
                    m->gpu_allocs.push_back(L.d_blobs);
                    L.n_slots = s;
                    L.exp_in_slot.assign((size_t) s, -1);
                    for (int k = 0; k < s; ++k) {
                        const int e = keep[(size_t) k];
                        copy_h2d_staged(L.d_blobs + (size_t) k * L.stride, L.hblobs + (size_t) e * L.stride,
                                        L.L.bytes, L.hblobs_pinned, st);
                        L.slot_of[(size_t) e] = k;
                        L.exp_in_slot[(size_t) k] = e;
                    }
                    CK(cudaStreamSynchronize(st));
                }
            }
            if (L.d_slot_of) {
                upload_slot_of(ip, st);
                CK(cudaStreamSynchronize(st));
            }
            if (is_tier1 && L.n_slots < ne) core_all_gpu = false;
            hybrid = true;
            rebal_burst = 2;
            return true;
        };

        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        if (fr < need_free) {
            // 1. First shrink/evict tier2 (outer 1x/tok layers) proportionally (or to 0 if needed).
            std::vector<int> t2_layers, t1_layers;
            for (int ip = 0; ip < (int) m->layers.size(); ++ip) {
                if (m->layers[(size_t) ip].n_slots <= 0) continue;
                if (ip >= hp.core_start && ip < hp.core_end) t1_layers.push_back(ip);
                else t2_layers.push_back(ip);
            }
            if (!t2_layers.empty() && fr < need_free) {
                size_t t2_total_bytes = 0;
                for (int ip : t2_layers) t2_total_bytes += (size_t) m->layers[(size_t) ip].n_slots * m->layers[(size_t) ip].stride;
                if (fr + t2_total_bytes <= need_free) {
                    for (int ip : t2_layers) shrink_layer(ip, 0);
                } else {
                    const size_t deficit = need_free - fr;
                    size_t sum_stride = 0;
                    for (int ip : t2_layers) sum_stride += m->layers[(size_t) ip].stride;
                    const int drop = sum_stride > 0 ? (int) ((deficit + sum_stride - 1) / sum_stride) : ne;
                    for (int ip : t2_layers) shrink_layer(ip, std::max(0, m->layers[(size_t) ip].n_slots - drop));
                }
                CK(cudaMemGetInfo(&fr, &tot));
                for (int ip = (int) m->layers.size() - 1; ip >= 0 && fr < need_free; --ip) {
                    if (ip >= hp.core_start && ip < hp.core_end) continue;
                    if (shrink_layer(ip, 0)) CK(cudaMemGetInfo(&fr, &tot));
                }
            }
            // 1.25. Shrink oversized adaptive prefill slabs down to safe_cap = max(tmax, cur_t) and release optional FP16 packs before Tier-1 eviction.
            if (fr < need_free && !in_decode_phase && active_pp_cap > std::max(tmax, cur_t)) {
                const int safe_cap = std::max(tmax, cur_t);
                const int tgt_swa_cap = use_shared_swa_scratch
                    ? swa_ring_cap()
                    : ((safe_cap <= 1) ? (int) hp.n_swa : (int) (hp.n_swa + safe_cap));
                free_ephemeral_bufs();
                trim_phase_pool(0);
                prefill_ephemeral_bytes = alloc_ephemeral_bufs(safe_cap);
                if (!use_shared_swa_scratch && swa_kv_cap != tgt_swa_cap) resize_swa_kv_pool(tgt_swa_cap);
                if (dspark) dspark->rebind_arena(layer_arena, layer_arena_bytes, safe_cap);
                active_pp_cap = safe_cap;
                prefill_swa_kv_cap = tgt_swa_cap;
                trim_phase_pool(0);
                CK(cudaMemGetInfo(&fr, &tot));
            }
            if (fr < need_free && (d_core_wdown_cur_f16 || d_core_wup_f16)) {
                if (d_core_wdown_cur_f16) { cudaFree(d_core_wdown_cur_f16); d_core_wdown_cur_f16 = nullptr; }
                if (d_core_wup_f16) { cudaFree(d_core_wup_f16); d_core_wup_f16 = nullptr; }
                CK(cudaMemGetInfo(&fr, &tot));
            }
            // 1.5. EX-4: Before ever evicting 3x/tok Tier-1 core experts (which also forces 6.4 GiB host blob rebuilds!),
            //      shrink oversized pp_stage (e.g. 4x192 MiB = 768 MiB) down to compact 2x64 MiB (128 MiB -> +640 MiB in <1 ms!).
            if (fr < need_free && pp_stage_cnt > 0) {
                const size_t compact_mb = 64;
                size_t compact_bytes = compact_mb << 20;
                for (int ip = 0; ip < (int) m->layers.size(); ++ip) {
                    const auto& L = m->layers[(size_t) ip];
                    if (L.gu_type == 40 && L.stride > 0) {
                        const size_t nm = ((compact_bytes + L.stride - 1) / L.stride) * L.stride;
                        if (nm > compact_bytes && nm - compact_bytes <= ((size_t) 256 << 10)) compact_bytes = nm;
                        break;
                    }
                }
                const size_t cur_stage_tot = (size_t) pp_stage_cnt * pp_stage_bytes;
                const size_t tgt_stage_tot = (size_t) 2 * compact_bytes;
                if (cur_stage_tot > tgt_stage_tot) {
                    alloc_pp_stages(2, compact_bytes);
                    saved_pp_stage_cnt = pp_stage_cnt;
                    saved_pp_stage_bytes = pp_stage_bytes;
                    CK(cudaMemGetInfo(&fr, &tot));
                    std::fprintf(stderr, "strata-inkling2: shrunk pp_stage -> %dx%.0f MiB before KV grow (free=%.0f MiB)\n",
                                 pp_stage_cnt, (double) pp_stage_bytes / (1 << 20), (double) fr / (1 << 20));
                }
            }
            // 2. If still short, shrink tier1 (3x/tok core layers):
            //    - If deficit is small (<= 48 experts total), trim only from the tail core layer(s) to avoid building 6.4 GiB
            //      of host blobs across all 12 core layers just to shed 1-4 slots/layer!
            //    - Otherwise (e.g. 131K->262K), shrink uniformly across all 12 core layers so EVERY core layer preserves its top-K hot experts.
            if (!t1_layers.empty() && fr < need_free) {
                const size_t t1_stride = m->layers[(size_t) t1_layers[0]].stride;
                const size_t total_drop_exps = (t1_stride > 0) ? ((need_free - fr + t1_stride - 1) / t1_stride) : (size_t) ne;
                const bool tail_only = (total_drop_exps <= 48);
                if (tail_only) {
                    for (int idx = (int) t1_layers.size() - 1; idx >= 0 && fr < need_free; --idx) {
                        const int ip = t1_layers[(size_t) idx];
                        const PhysLayer& L = m->layers[(size_t) ip];
                        const size_t deficit = need_free - fr;
                        const int drop = (int) ((deficit + L.stride - 1) / L.stride);
                        shrink_layer(ip, std::max(0, L.n_slots - drop));
                        CK(cudaMemGetInfo(&fr, &tot));
                    }
                } else {
                    // Level all tier1 layers to the same uniform target slot count (avoids uneven 48..94/128 after prior tail trim).
                    size_t sum_stride = 0;
                    int sum_slots = 0;
                    for (int ip : t1_layers) {
                        sum_stride += m->layers[(size_t) ip].stride;
                        sum_slots += m->layers[(size_t) ip].n_slots;
                    }
                    while (fr < need_free && sum_stride > 0) {
                        const size_t deficit = need_free - fr;
                        const int total_drop = std::max(1, (int) ((deficit + t1_stride - 1) / t1_stride));
                        const int target_s = std::max(0, (sum_slots - total_drop) / (int) t1_layers.size());
                        bool any = false;
                        sum_slots = 0;
                        for (int ip : t1_layers) {
                            const int cur_s = m->layers[(size_t) ip].n_slots;
                            if (cur_s > target_s && shrink_layer(ip, target_s)) any = true;
                            sum_slots += m->layers[(size_t) ip].n_slots;
                        }
                        if (!any) {
                            for (int ip : t1_layers) {
                                const int cur_s = m->layers[(size_t) ip].n_slots;
                                if (cur_s > 0 && shrink_layer(ip, cur_s - 1)) any = true;
                            }
                            if (!any) break;
                        }
                        CK(cudaMemGetInfo(&fr, &tot));
                    }
                }
            }
        }

#if defined(STRATA_INKLING2_CPU_POOL)
        if (hybrid && !pool) {
            int nw = cpu_threads_req > 0 ? cpu_threads_req : 0;
            pool = std::make_unique<strata::kernels::cpu::ExpertPool>(nw, true, true);
            pool_workers = pool->workers();
            pin_host_for_pool();
            std::fprintf(stderr, "strata-inkling2: %s\n", pool->placement_line().c_str());
        }
#endif
        if (hybrid && !h_h) {
            bool has_bf16_experts = false;
            for (int ip = 0; ip < hp.n_phys; ++ip) {
                if (m->layers[(size_t) ip].gu_type == 30) { has_bf16_experts = true; break; }
            }
            const int cpu_t = (!has_bf16_experts && pp_stage[0] && pp_stream_t > 1) ? std::min(tmax, std::max(pp_stream_t - 1, kDecodePhaseCap)) : tmax;
            init_cpu_bufs(cpu_t);
        }

        // Reallocate full-attention KV buffers one layer at a time; use pinned host staging when tight on VRAM.
        struct PinnedStage {
            uint8_t* ptr = nullptr;
            size_t cap = 0;
            bool pinned = false;
            void ensure(size_t need) {
                if (need <= cap) return;
                release();
                if (cudaHostAlloc(&ptr, need, cudaHostAllocDefault) == cudaSuccess) {
                    cap = need; pinned = true;
                } else {
                    (void) cudaGetLastError();
                    ptr = (uint8_t*) std::malloc(need);
                    cap = ptr ? need : 0;
                    pinned = false;
                }
            }
            void release() {
                if (!ptr) return;
                if (pinned) cudaFreeHost(ptr);
                else std::free(ptr);
                ptr = nullptr; cap = 0; pinned = false;
            }
            ~PinnedStage() { release(); }
        } kv_stage;
        for (int il = 0; il < hp.n_layer; ++il) {
            if (hp.is_swa(il)) continue;
            const size_t n_old = (size_t) std::min<int64_t>(pos, full_kv_cap);
            CK(cudaMemGetInfo(&fr, &tot));
            const bool direct_d2d = (fr >= (size_t) new_cap * per_tok + ((size_t) 128 << 20));
            if (kv_q8) {
                // Full-Attn Q8 KV is Head-Major: [n_head_kv][cap][128] for kq/vq and [n_head_kv][cap] for ks/vs.
                const int nhkv = (int) hp.n_head_kv;
                const size_t b_kv_head = n_old * 128 * sizeof(int8_t);
                const size_t b_s_head = n_old * sizeof(__half);
                const size_t old_kv_stride = (size_t) full_kv_cap * 128;
                const size_t new_kv_stride = (size_t) new_cap * 128;
                if (direct_d2d) {
                    int8_t *nkq = nullptr, *nvq = nullptr;
                    __half *nks = nullptr, *nvs = nullptr;
                    CK(cudaMalloc(&nkq, (size_t) new_cap * hp.kv_dim * sizeof(int8_t)));
                    CK(cudaMalloc(&nvq, (size_t) new_cap * hp.kv_dim * sizeof(int8_t)));
                    CK(cudaMalloc(&nks, (size_t) new_cap * hp.n_head_kv * sizeof(__half)));
                    CK(cudaMalloc(&nvs, (size_t) new_cap * hp.n_head_kv * sizeof(__half)));
                    if (n_old > 0) {
                        for (int h = 0; h < nhkv; ++h) {
                            CK(cudaMemcpyAsync(nkq + (size_t) h * new_kv_stride, kq[(size_t) il] + (size_t) h * old_kv_stride, b_kv_head, cudaMemcpyDeviceToDevice, st));
                            CK(cudaMemcpyAsync(nvq + (size_t) h * new_kv_stride, vq[(size_t) il] + (size_t) h * old_kv_stride, b_kv_head, cudaMemcpyDeviceToDevice, st));
                            CK(cudaMemcpyAsync(nks + (size_t) h * new_cap,       ks[(size_t) il] + (size_t) h * full_kv_cap,   b_s_head,  cudaMemcpyDeviceToDevice, st));
                            CK(cudaMemcpyAsync(nvs + (size_t) h * new_cap,       vs[(size_t) il] + (size_t) h * full_kv_cap,   b_s_head,  cudaMemcpyDeviceToDevice, st));
                        }
                        CK(cudaStreamSynchronize(st));
                    }
                    cudaFree(kq[(size_t) il]); cudaFree(vq[(size_t) il]);
                    cudaFree(ks[(size_t) il]); cudaFree(vs[(size_t) il]);
                    kq[(size_t) il] = nkq; vq[(size_t) il] = nvq;
                    ks[(size_t) il] = nks; vs[(size_t) il] = nvs;
                } else {
                    if (n_old > 0) {
                        kv_stage.ensure((size_t) nhkv * (2 * b_kv_head + 2 * b_s_head));
                        uint8_t* p = kv_stage.ptr;
                        for (int h = 0; h < nhkv; ++h) {
                            CK(cudaMemcpyAsync(p, kq[(size_t) il] + (size_t) h * old_kv_stride, b_kv_head, cudaMemcpyDeviceToHost, st)); p += b_kv_head;
                            CK(cudaMemcpyAsync(p, vq[(size_t) il] + (size_t) h * old_kv_stride, b_kv_head, cudaMemcpyDeviceToHost, st)); p += b_kv_head;
                            CK(cudaMemcpyAsync(p, ks[(size_t) il] + (size_t) h * full_kv_cap,   b_s_head,  cudaMemcpyDeviceToHost, st)); p += b_s_head;
                            CK(cudaMemcpyAsync(p, vs[(size_t) il] + (size_t) h * full_kv_cap,   b_s_head,  cudaMemcpyDeviceToHost, st)); p += b_s_head;
                        }
                        CK(cudaStreamSynchronize(st));
                    }
                    cudaFree(kq[(size_t) il]); cudaFree(vq[(size_t) il]);
                    cudaFree(ks[(size_t) il]); cudaFree(vs[(size_t) il]);
                    kq[(size_t) il] = nullptr; vq[(size_t) il] = nullptr;
                    ks[(size_t) il] = nullptr; vs[(size_t) il] = nullptr;
                    CK(cudaMalloc(&kq[(size_t) il], (size_t) new_cap * hp.kv_dim * sizeof(int8_t)));
                    CK(cudaMalloc(&vq[(size_t) il], (size_t) new_cap * hp.kv_dim * sizeof(int8_t)));
                    CK(cudaMalloc(&ks[(size_t) il], (size_t) new_cap * hp.n_head_kv * sizeof(__half)));
                    CK(cudaMalloc(&vs[(size_t) il], (size_t) new_cap * hp.n_head_kv * sizeof(__half)));
                    if (n_old > 0) {
                        const uint8_t* p = kv_stage.ptr;
                        for (int h = 0; h < nhkv; ++h) {
                            CK(cudaMemcpyAsync(kq[(size_t) il] + (size_t) h * new_kv_stride, p, b_kv_head, cudaMemcpyHostToDevice, st)); p += b_kv_head;
                            CK(cudaMemcpyAsync(vq[(size_t) il] + (size_t) h * new_kv_stride, p, b_kv_head, cudaMemcpyHostToDevice, st)); p += b_kv_head;
                            CK(cudaMemcpyAsync(ks[(size_t) il] + (size_t) h * new_cap,       p, b_s_head,  cudaMemcpyHostToDevice, st)); p += b_s_head;
                            CK(cudaMemcpyAsync(vs[(size_t) il] + (size_t) h * new_cap,       p, b_s_head,  cudaMemcpyHostToDevice, st)); p += b_s_head;
                        }
                        CK(cudaStreamSynchronize(st));
                    }
                }
            } else {
                const size_t b_old = n_old * (size_t) hp.kv_dim * sizeof(__half);
                if (direct_d2d) {
                    __half *nkc = nullptr, *nvc = nullptr;
                    CK(cudaMalloc(&nkc, (size_t) new_cap * hp.kv_dim * sizeof(__half)));
                    CK(cudaMalloc(&nvc, (size_t) new_cap * hp.kv_dim * sizeof(__half)));
                    if (n_old > 0) {
                        CK(cudaMemcpyAsync(nkc, kc[(size_t) il], b_old, cudaMemcpyDeviceToDevice, st));
                        CK(cudaMemcpyAsync(nvc, vc[(size_t) il], b_old, cudaMemcpyDeviceToDevice, st));
                        CK(cudaStreamSynchronize(st));
                    }
                    cudaFree(kc[(size_t) il]); cudaFree(vc[(size_t) il]);
                    kc[(size_t) il] = nkc; vc[(size_t) il] = nvc;
                } else {
                    if (n_old > 0) {
                        kv_stage.ensure(2 * b_old);
                        CK(cudaMemcpyAsync(kv_stage.ptr, kc[(size_t) il], b_old, cudaMemcpyDeviceToHost, st));
                        CK(cudaMemcpyAsync(kv_stage.ptr + b_old, vc[(size_t) il], b_old, cudaMemcpyDeviceToHost, st));
                        CK(cudaStreamSynchronize(st));
                    }
                    cudaFree(kc[(size_t) il]); cudaFree(vc[(size_t) il]);
                    kc[(size_t) il] = nullptr; vc[(size_t) il] = nullptr;
                    CK(cudaMalloc(&kc[(size_t) il], (size_t) new_cap * hp.kv_dim * sizeof(__half)));
                    CK(cudaMalloc(&vc[(size_t) il], (size_t) new_cap * hp.kv_dim * sizeof(__half)));
                    if (n_old > 0) {
                        CK(cudaMemcpyAsync(kc[(size_t) il], kv_stage.ptr, b_old, cudaMemcpyHostToDevice, st));
                        CK(cudaMemcpyAsync(vc[(size_t) il], kv_stage.ptr + b_old, b_old, cudaMemcpyHostToDevice, st));
                        CK(cudaStreamSynchronize(st));
                    }
                }
            }
            kv_cap[(size_t) il] = new_cap;
        }
        if (dspark) dspark->grow_full_kv(new_cap, st);
        full_kv_cap = new_cap;
        int t1_min = ne, t1_max = 0, t2_min = ne, t2_max = 0, tot_slots = 0, t1_pin = 0, t2_pin = 0;
        size_t exp_bytes = 0;
        for (int ip = 0; ip < (int) m->layers.size(); ++ip) {
            const PhysLayer& L = m->layers[(size_t) ip];
            tot_slots += L.n_slots;
            exp_bytes += (size_t) L.n_slots * L.stride;
            if (ip >= hp.core_start && ip < hp.core_end) {
                t1_min = std::min(t1_min, L.n_slots); t1_max = std::max(t1_max, L.n_slots);
                if (L.hblobs_pinned) ++t1_pin;
            } else {
                t2_min = std::min(t2_min, L.n_slots); t2_max = std::max(t2_max, L.n_slots);
                if (L.hblobs_pinned) ++t2_pin;
            }
        }
        placed_slots = tot_slots;
        placed_bytes = exp_bytes;
        CK(cudaMemGetInfo(&fr, &tot));
        vram_free_after_place = fr;
        std::fprintf(stderr,
                     "strata-inkling2: expanded full-attn KV cache to %d slots (experts=%d/%lld, %.2f GiB, tier1=%d..%d/128 [pin=%d], tier2=%d..%d/128 [pin=%d], free=%.0f MiB)\n",
                     full_kv_cap, tot_slots, (long long) hp.n_phys * ne, (double) exp_bytes / (1 << 30),
                     t1_min, t1_max, t1_pin, t2_min, t2_max, t2_pin, (double) fr / (1 << 20));
    }

    int32_t forward(const int32_t* tokens, int T, bool copy_logits = false, std::vector<float>* out_logits = nullptr,
                    bool all_row_logits = false, bool capture_dspark = false, int verify_k_save = 0,
                    std::vector<int32_t>* out_picks = nullptr, bool need_logits = true,
                    bool defer_logits_d2h = false,
                    const float* img_embs = nullptr, size_t img_embs_n = 0, int32_t img_pad_id = -1,
                    size_t* img_cursor = nullptr, bool allow_prelude_borrow = false) {
        dspark_injected_in_fwd = false;
        moe_raw_zero_pending = false;
        if (T > kDecodePhaseCap) {
            decode_scores_normalized = false;
            if (in_decode_phase) set_phase_prefill();
        }
        if (T > active_pp_cap) ensure_pp_cap(T, (int) (pos + T));
        ensure_kv_cap((int) (pos + T), T);
        // EX-5 Idea A & B + Wave 17: RAII scope guard guarantees any clobbered Prelude donor d_blobs are cleanly freed
        // and invalidated on ALL return paths, while any in-place borrowed Core donor (blk.6) is NEVER freed/invalidated
        // and is asynchronously restored from pinned hblobs on cst after free_transient_pp_stages() (0 ms prefill penalty).
        struct BorrowCleanup {
            Engine* E = nullptr;
            ~BorrowCleanup() {
                if (!E) return;
                const int n_phys = (int) E->m->layers.size();
                const int core_start_ip = (int) E->m->hp.core_start;
                bool any_prelude_clobbered = false;
                for (int ip = 0; ip < n_phys && ip < kMaxRebalLayers; ++ip) {
                    if (ip < core_start_ip && E->donor_clobbered[ip]) { any_prelude_clobbered = true; break; }
                }
                if (any_prelude_clobbered) {
                    if (E->aux_st) CK(cudaStreamSynchronize(E->aux_st));
                    if (E->cst) CK(cudaStreamSynchronize(E->cst));
                    CK(cudaStreamSynchronize(E->st));
                    const int ne = (int) E->m->hp.n_expert;
                    for (int ip = 0; ip < n_phys && ip < kMaxRebalLayers && ip < core_start_ip; ++ip) {
                        if (!E->donor_clobbered[ip]) continue;
                        E->donor_clobbered[ip] = false;
                        PhysLayer& L = E->m->layers[(size_t) ip];
                        E->free_layer_blobs(L);
                        L.n_slots = 0;
                        L.exp_in_slot.clear();
                        std::fill(L.slot_of.begin(), L.slot_of.end(), -1);
                        if (L.d_slot_of) {
                            CK(cudaMemsetAsync(L.d_slot_of, 0xFF, (size_t) ne * sizeof(int32_t), E->st));
                        }
                    }
                    int tot_slots = 0;
                    size_t exp_bytes = 0;
                    for (int ip = 0; ip < n_phys; ++ip) {
                        const PhysLayer& L = E->m->layers[(size_t) ip];
                        tot_slots += L.n_slots;
                        exp_bytes += (size_t) L.n_slots * L.stride;
                    }
                    E->placed_slots = tot_slots;
                    E->placed_bytes = exp_bytes;
                    E->hybrid = true;
                }
                E->free_transient_pp_stages();
                const int n_cb = E->core_borrowed_cnt;
                E->core_borrowed_cnt = 0;
                for (int ci = 0; ci < n_cb; ++ci) {
                    const int core_ip = E->core_borrowed_ips[ci];
                    const size_t core_bytes = E->core_borrowed_bytes[ci];
                    if (core_ip >= core_start_ip && core_ip < n_phys && core_ip < kMaxRebalLayers &&
                        E->donor_clobbered[core_ip]) {
                        E->donor_clobbered[core_ip] = false;
                        PhysLayer& Lc = E->m->layers[(size_t) core_ip];
                        if (Lc.d_blobs && Lc.hblobs && core_bytes > 0 && E->cst) {
                            for (int b = 0; b < E->pp_active_slots && b < kMaxPpStages; ++b) {
                                if (E->pp_buf_owner_ip[b] == core_ip) {
                                    CK(cudaStreamWaitEvent(E->cst, E->ev_pp_used[b], 0));
                                }
                            }
                            CK(cudaMemcpyAsync(Lc.d_blobs, Lc.hblobs, core_bytes, cudaMemcpyHostToDevice, E->cst));
                            CK(cudaEventRecord(E->ev_rebal_layer[core_ip], E->cst));
                            E->layer_pending_rebal[core_ip] = true;
                        }
                    }
                }
                E->pp_free_slots.clear();
                E->pp_active_slots = 0;
                for (int b = 0; b < kMaxPpStages; ++b) {
                    E->pp_buf_ptr[b] = nullptr;
                    E->pp_buf_cap[b] = 0;
                    E->pp_buf_owner_ip[b] = -1;
                    E->pp_slot_needs_wait[b] = false;
                }
            }
        } borrow_cleanup{this};

        const auto& hp = m->hp;
        const int N = (int) hp.n_embd;
        const int64_t kv_dim = hp.kv_dim;
        const int64_t q_dim = hp.q_dim;
        const int64_t n_core = hp.num_loops > 1 ? (hp.core_end - hp.core_start) : 0;
        bool chunk_has_img = false;
        std::vector<int64_t> img_slot;
        if (img_embs != nullptr && img_cursor != nullptr && img_pad_id >= 0) {
            img_slot.assign((size_t) T, -1);
            for (int t = 0; t < T; ++t) {
                if (tokens[t] == img_pad_id && *img_cursor < img_embs_n) {
                    img_slot[(size_t) t] = (int64_t) (*img_cursor)++;
                    h_is_img_pin[t] = 1;
                    chunk_has_img = true;
                } else {
                    h_is_img_pin[t] = 0;
                }
            }
        }
        static const bool no_gpu_embd = (std::getenv("STRATA_NO_GPU_EMBD") != nullptr);
        const bool use_gpu_embd = (!no_gpu_embd && T >= 64 && !chunk_has_img &&
                                   m->host_tok_embd != nullptr && h_tok_raw_pin != nullptr && h_tok_idx_pin != nullptr &&
                                   (m->tok_embd_type == 8 || m->tok_embd_type == 30 || m->tok_embd_type == 1) &&
                                   (size_t) T * m->tok_embd_row_bytes <= h_tok_raw_bytes &&
                                   (size_t) T * m->tok_embd_row_bytes <= (size_t) T * (size_t) N * sizeof(float));

        pp_streaming = (hybrid && T >= pp_stream_t && pp_stage_cnt > 0 && pp_stage[0] != nullptr);
        if (pp_streaming) {
            // Wave 8.5: Free idle DSpark decode weights (+357.8 MiB) back into phase_pool BEFORE transient Epilogue stage expansion!
            if (T >= 256 && use_phase_pool && phase_pool && dspark && dspark->dm &&
                dspark->dm->decode_w_offloadable && dspark->dm->decode_w_resident) {
                dspark->evict_decode_weights(phase_pool, st);
            }
            // Wave 11 Path C2: On T >= 4096 when borrow==0 (e.g. GEN 1 or intermediate chunks) and pp_stage_cnt < 6,
            // cap Tier-2 tail experts (strictly keeping 100% Core 1536/1536 AND blk.0 128/128!) so transient
            // pre-buffering reaches pp_slots >= 6 (perm=4 trans=2) and eliminates Epilogue PCIe stall!
            size_t avail_known_min = 0;
            static const bool no_transient_pp_cap = (std::getenv("STRATA_NO_TRANSIENT_PP_STAGE") != nullptr);
            if (!no_transient_pp_cap && !allow_prelude_borrow && T >= 4096 &&
                use_phase_pool && phase_pool && is_core_full_resident() &&
                pp_stage_cnt >= 2 && pp_stage_cnt < 6 && pp_stage_bytes >= ((size_t) 64 << 20)) {
                const int core_tot = (hp.num_loops > 1 && hp.core_end > hp.core_start)
                    ? (int) ((hp.core_end - hp.core_start) * hp.n_expert) : 0;
                int core_res = 0;
                for (int ip = (int) hp.core_start; ip < (int) hp.core_end && ip < hp.n_phys; ++ip) {
                    core_res += m->layers[(size_t) ip].n_slots;
                }
                const size_t stride_b = (pp_stage_bytes + 255u) & ~size_t(255u);
                const size_t guard = vram_reserve_bytes + ((size_t) 64 << 20);
                size_t fr_t = 0, tot_t = 0;
                CK(cudaMemGetInfo(&fr_t, &tot_t));
                const size_t avail_t = pool_cached_free_bytes() + fr_t;
                avail_known_min = avail_t;
                const int max_need_extra = 6 - pp_stage_cnt;
                const size_t need_6 = (size_t) max_need_extra * stride_b + guard;
                if (core_res >= core_tot && avail_t < need_6) {
                    std::vector<int> evict_order;
                    evict_order.reserve((size_t) hp.n_phys);
                    size_t evictable_b = 0;
                    // Tail Tier-2 first (Epilogue blk.23..18, then Prelude blk.5..1); NEVER touch Core or blk.0!
                    for (int ip = hp.n_phys - 1; ip >= (int) hp.core_end; --ip) {
                        const PhysLayer& L = m->layers[(size_t) ip];
                        if (L.hblobs && L.d_blobs && L.n_slots > 0) {
                            evict_order.push_back(ip);
                            evictable_b += (size_t) L.n_slots * L.stride;
                        }
                    }
                    for (int ip = (int) hp.core_start - 1; ip >= 1; --ip) {
                        const PhysLayer& L = m->layers[(size_t) ip];
                        if (L.hblobs && L.d_blobs && L.n_slots > 0) {
                            evict_order.push_back(ip);
                            evictable_b += (size_t) L.n_slots * L.stride;
                        }
                    }
                    int tgt_extra = 0;
                    for (int ex = max_need_extra; ex >= 1; --ex) {
                        if (avail_t + evictable_b >= (size_t) ex * stride_b + guard) {
                            tgt_extra = ex;
                            break;
                        }
                    }
                    const size_t need_tgt = (size_t) tgt_extra * stride_b + guard;
                    if (tgt_extra > 0 && avail_t < need_tgt && !evict_order.empty()) {
                        bool synced_streams = false;
                        size_t freed_b = 0;
                        const int ne = (int) hp.n_expert;
                        defer_destroy_graphs();
                        for (int ip : evict_order) {
                            if (avail_t + freed_b >= need_tgt) break;
                            PhysLayer& L = m->layers[(size_t) ip];
                            if (!synced_streams && (!L.d_blobs_from_pool || (ip < kMaxRebalLayers && layer_pending_rebal[ip]))) {
                                if (cst) CK(cudaStreamSynchronize(cst));
                                CK(cudaStreamSynchronize(st));
                                for (int ri = 0; ri < kMaxRebalLayers; ++ri) layer_pending_rebal[ri] = false;
                                synced_streams = true;
                            }
                            freed_b += (size_t) L.n_slots * L.stride;
                            free_layer_blobs(L);
                            L.n_slots = 0;
                            L.exp_in_slot.clear();
                            std::fill(L.slot_of.begin(), L.slot_of.end(), -1);
                            if (L.d_slot_of) {
                                CK(cudaMemsetAsync(L.d_slot_of, 0xFF, (size_t) ne * sizeof(int32_t), st));
                            }
                        }
                        if (freed_b > 0) {
                            int tot_slots = 0;
                            size_t exp_bytes = 0;
                            for (int ip = 0; ip < hp.n_phys; ++ip) {
                                const PhysLayer& L = m->layers[(size_t) ip];
                                tot_slots += L.n_slots;
                                exp_bytes += (size_t) L.n_slots * L.stride;
                            }
                            placed_slots = tot_slots;
                            placed_bytes = exp_bytes;
                            hybrid = true;
                            avail_known_min = avail_t + freed_b;
                        }
                    }
                }
            }
            pp_prepare();
            maybe_alloc_transient_pp_stages(avail_known_min);
            pp_kick_partial_first(use_gpu_embd ? ((size_t) 8 << 20) : ((size_t) 48 << 20));
        }
        if (verify_k_save > 0 && !d_conv_trace) alloc_conv_trace();
        const bool compute_logits = need_logits || copy_logits || all_row_logits || out_picks != nullptr;
        const int rows = (all_row_logits || out_picks != nullptr) ? T : 1;
        const bool want_softcap = (copy_logits && hp.softcap > 0.0f);
        const bool want_argmax = !copy_logits || (out_picks != nullptr);
        const bool can_full_graph = (!hybrid && core_all_gpu && full_kv_cap == ctx && !prof && !prof_detail && !graph_oom &&
                                     img_embs == nullptr &&
                                     T >= 1 && T <= std::min(verify_max_t, 9) && compute_logits && rows == T &&
                                     h_tok_buf != nullptr && h_pos_pin != nullptr && h_pick_pin != nullptr &&
                                     (!copy_logits || defer_logits_d2h) &&
                                     ((T == 1 && verify_k_save == 0) || (T >= 2 && verify_k_save == T - 1)));
        bool capturing_full_graph = false;
        cudaGraphExec_t* fg_slot = nullptr;
        if (can_full_graph) {
            for (int ip = 0; ip < (int) m->layers.size() && ip < kMaxRebalLayers; ++ip) {
                if (layer_pending_rebal[ip]) {
                    CK(cudaStreamWaitEvent(st, ev_rebal_layer[ip], 0));
                    layer_pending_rebal[ip] = false;
                }
            }
            for (int t = 0; t < T; ++t) m->dequant_tok(tokens[t], h_tok_buf + (size_t) t * N);
            *h_pos_pin = (int32_t) pos;
            fg_slot = &full_decode_graph[T][capture_dspark ? 1 : 0][want_softcap ? 1 : 0][want_argmax ? 1 : 0];
            if (*fg_slot) {
                CK(cudaGraphLaunch(*fg_slot, st));
                int32_t last_pick = 0;
                if (want_argmax) {
                    CK(cudaStreamSynchronize(st));
                    if (out_picks) out_picks->assign(h_pick_pin, h_pick_pin + rows);
                    last_pick = h_pick_pin[rows - 1];
                }
                pos += T;
                return last_pick;
            }
            if (cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
                capturing_full_graph = true;
            } else {
                (void) cudaGetLastError();
                graph_oom = true;
            }
        }

        const bool save_conv_trace = (verify_k_save > 0 && d_conv_trace != nullptr);
        const size_t total_conv_floats = (size_t) hp.n_layer * layer_conv_floats;

        // 1. Token embedding from host mmap (+ injected vision embeddings) + masked tok_embd_norm
        if (prof) ++p_calls;
        const auto tp_e0 = Clock::now();
        if (use_gpu_embd) {
            // Wave 5: O(T) unique-token deduplication + single-thread raw row memcpy (<0.5 ms) -> GPU fused dequant+RMSNorm!
            const size_t seen_sz = tok_seen.size();
            const size_t rb = m->tok_embd_row_bytes;
            tok_uniq.clear();
            int32_t U = 0;
            for (int t = 0; t < T; ++t) {
                const int32_t tid = tokens[t];
                if ((size_t) tid < seen_sz) {
                    int32_t uid = tok_seen[(size_t) tid];
                    if (uid < 0) {
                        uid = U++;
                        tok_seen[(size_t) tid] = uid;
                        tok_uniq.push_back(tid);
                        std::memcpy(h_tok_raw_pin + (size_t) uid * rb,
                                    m->host_tok_embd + (size_t) tid * rb, rb);
                    }
                    h_tok_idx_pin[t] = uid;
                } else {
                    const int32_t uid = U++;
                    std::memcpy(h_tok_raw_pin + (size_t) uid * rb,
                                m->host_tok_embd + (size_t) tid * rb, rb);
                    h_tok_idx_pin[t] = uid;
                }
            }
            for (int32_t tid : tok_uniq) tok_seen[(size_t) tid] = -1;
            uint8_t* raw_d = reinterpret_cast<uint8_t*>(ffn_inp);
            CK(cudaMemcpyAsync(raw_d, h_tok_raw_pin, (size_t) U * rb, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(pick_d, h_tok_idx_pin, (size_t) T * sizeof(int32_t), cudaMemcpyHostToDevice, st));
        } else {
            if (!can_full_graph) {
                auto fill_row = [&](int t) {
                    if (chunk_has_img && img_slot[(size_t) t] >= 0) {
                        std::memcpy(h_tok_buf + (size_t) t * N, img_embs + (size_t) img_slot[(size_t) t] * N,
                                    (size_t) N * sizeof(float));
                    } else {
                        m->dequant_tok(tokens[t], h_tok_buf + (size_t) t * N);
                    }
                };
                if (T >= 128) {
#if defined(STRATA_INKLING2_CPU_POOL)
                    const bool unpinned_for_embd = host_pinned;
                    if (unpinned_for_embd) {
                        strata::kernels::cpu::restore_thread_affinity(host_prev_aff);
                        host_pinned = false;
                    }
#endif
                    const int hw = (int) std::thread::hardware_concurrency();
                    const int nt = std::clamp(hw > 0 ? hw : 4, 1, std::min(8, T / 64));
                    std::vector<std::thread> ths;
                    ths.reserve((size_t) (nt - 1));
                    const int per = (T + nt - 1) / nt;
                    for (int wi = 1; wi < nt; ++wi) {
                        const int t0 = wi * per;
                        const int t1 = std::min(T, t0 + per);
                        if (t0 >= t1) break;
                        ths.emplace_back([&, t0, t1]() {
                            for (int t = t0; t < t1; ++t) fill_row(t);
                        });
                    }
                    const int t1_0 = std::min(T, per);
                    for (int t = 0; t < t1_0; ++t) fill_row(t);
                    for (auto& th : ths) th.join();
#if defined(STRATA_INKLING2_CPU_POOL)
                    if (unpinned_for_embd) pin_host_for_pool();
#endif
                } else {
                    for (int t = 0; t < T; ++t) fill_row(t);
                }
            }
            CK(cudaMemcpyAsync(ffn_inp, h_tok_buf, (size_t) T * N * sizeof(float), cudaMemcpyHostToDevice, st));
            if (chunk_has_img) {
                CK(cudaMemcpyAsync(d_is_img, h_is_img_pin, (size_t) T * sizeof(uint8_t), cudaMemcpyHostToDevice, st));
            }
        }
        const int32_t pos32 = (int32_t) pos;
        if (T >= 256 && pos >= 8192 && T > rel_tab_cap_t && rel_tab_cap_t >= 256) {
            const int n_sub = (T + 255) / 256;
            if (h_pos_pin && 1 + n_sub <= pos0_sub_cap) {
                h_pos_pin[0] = pos32;
                for (int s = 0; s < n_sub; ++s) h_pos_pin[1 + s] = pos32 + s * 256;
                CK(cudaMemcpyAsync(pos0_d, h_pos_pin, (size_t) (1 + n_sub) * sizeof(int32_t), cudaMemcpyHostToDevice, st));
            } else {
                std::vector<int32_t> pos_tmp((size_t) (1 + n_sub));
                pos_tmp[0] = pos32;
                for (int s = 0; s < n_sub; ++s) pos_tmp[(size_t) (1 + s)] = pos32 + s * 256;
                CK(cudaMemcpyAsync(pos0_d, pos_tmp.data(), pos_tmp.size() * sizeof(int32_t), cudaMemcpyHostToDevice, st));
            }
        } else if (h_pos_pin) {
            if (!can_full_graph) *h_pos_pin = pos32;
            CK(cudaMemcpyAsync(pos0_d, h_pos_pin, sizeof(int32_t), cudaMemcpyHostToDevice, st));
        } else {
            CK(cudaMemcpyAsync(pos0_d, &pos32, sizeof(int32_t), cudaMemcpyHostToDevice, st));
        }
        if (pp_streaming) pp_kick_finish_and_fill();
        if (verify_k_save > 0 && swa_bak && swa_ptrs.n_swa_layers > 0) {
            if (kv_q8) {
                k_swa_q8_bak_d<<<dim3((unsigned) verify_k_save, (unsigned) swa_ptrs.n_swa_layers), 256, 0, st>>>(
                    swa_ptrs, swa_bak, (int) kv_dim, (int) hp.n_head_kv, swa_kv_cap,
                    pos0_d, 1, 0, verify_k_save, 0);
            } else {
                k_swa_f16_bak_d<<<dim3((unsigned) verify_k_save, (unsigned) swa_ptrs.n_swa_layers), 256, 0, st>>>(
                    swa_ptrs, swa_bak, (int) kv_dim, swa_kv_cap,
                    pos0_d, 1, 0, verify_k_save, 0);
            }
        }
        if (use_gpu_embd) {
            const uint8_t* raw_d = reinterpret_cast<const uint8_t*>(ffn_inp);
            const int emb_blk = (N == 2304) ? 576 : 256;
            const int rb = (int) m->tok_embd_row_bytes;
            if (m->tok_embd_type == 8) {
                k_dequant_embd_rmsnorm_q8_0<<<(unsigned) T, emb_blk, 0, st>>>(
                    raw_d, pick_d, m->tok_embd_norm, cur, T, N, rb, hp.rms_eps);
            } else if (m->tok_embd_type == 30) {
                k_dequant_embd_rmsnorm_bf16<<<(unsigned) T, emb_blk, 0, st>>>(
                    raw_d, pick_d, m->tok_embd_norm, cur, T, N, rb, hp.rms_eps);
            } else {
                k_dequant_embd_rmsnorm_f16<<<(unsigned) T, emb_blk, 0, st>>>(
                    raw_d, pick_d, m->tok_embd_norm, cur, T, N, rb, hp.rms_eps);
            }
        } else if (chunk_has_img) {
            k_rms_norm_masked<<<(unsigned) T, kThreads, 0, st>>>(ffn_inp, m->tok_embd_norm, d_is_img, cur, N, hp.rms_eps);
        } else {
            k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(ffn_inp, m->tok_embd_norm, cur, N, hp.rms_eps);
        }
        if (prof) { CK(cudaStreamSynchronize(st)); p_embd_ms += ms_since(tp_e0); }

        static const bool no_dspark_vram_acc = (std::getenv("STRATA_NO_DSPARK_VRAM_ACC") != nullptr);
        static const bool no_vmsg_fwd = (std::getenv("STRATA_NO_VMSG") != nullptr);
        if (capture_dspark && dspark) {
            const bool can_vram_acc = (!no_dspark_vram_acc && !no_vmsg_fwd && T >= 32 && cc_maj >= 8 &&
                                       N == 2304 && hp.loop_gate_rank == 64 && n_core == 12 && can_vmsg_all_core &&
                                       ds_acc_swa0 != nullptr && ds_acc_swa2 != nullptr && msg3 != nullptr);
            if (can_vram_acc) {
                dspark->begin_prefill_vram_acc(ds_acc_swa0, msg3, ds_acc_swa2, T);
            } else {
                dspark->prefill_vram_acc_active = false;
            }
        }

        std::vector<float*> norm_blocks;
        int next_blk_buf = 0;
        auto pick_blk_slot = [&](int il, int ip, int pass, bool& skip_recompute) -> int {
            skip_recompute = false;
            if (norm_blk_cnt == 5) {
                if (il == 0) return 0;
                if (ip == hp.core_start) {
                    if (pass > 0) skip_recompute = true; // norm_blk_buf[1] already holds rms_norm(prelude_cur)!
                    return 1;
                }
                if (n_core > 0 && ip >= hp.core_start && ip < hp.core_end) {
                    return (pass & 1) ? 3 : 2;
                }
                return 3; // epilogue push at il=42 (norm_bank & bank_last are dead; live = {0, 1, 2, 3})
            }
            return next_blk_buf++;
        };
        auto push_block = [&](const float* src, int il, int ip, int pass) {
            bool skip_recompute = false;
            const int idx = pick_blk_slot(il, ip, pass, skip_recompute);
            float* ndst = norm_blk_buf[idx];
            if (!skip_recompute) {
                k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(src, nullptr, ndst, N, hp.rms_eps);
            }
            norm_blocks.push_back(ndst);
        };

        std::vector<float*> prelude_norm_blocks;
        float* prelude_cur = prelude_cur_buf;
        std::vector<float*> norm_bank;
        float* bank_last_norm = norm_blk_buf[norm_blk_cnt - 1];     // cached normalized bank_last
        bool vmsg_pass_ready = false;
        bool vmsg_bank_f16_ready = false;
        int vmsg_hoist_base_l = -1;
        const float* vmsg_bank0 = nullptr;
        const float* vmsg_bank1 = nullptr;
        struct DeferredPostFfn {
            bool active = false;
            int src_il = -1;
            const float* w_conv = nullptr;
            float* hst = nullptr;
            int hist_offset = 0;
        };
        DeferredPostFfn deferred_postffn{};
        int pre_push_done_il = -1;
        float* pre_push_ndst = nullptr;
        bool pre_routed_pp_cur = false;
        bool dspark_acc_pending = false;
        auto wait_dspark_acc = [&]() {
            if (dspark_acc_pending) {
                CK(cudaStreamWaitEvent(st, ev_dspark_acc_done, 0));
                dspark_acc_pending = false;
            }
        };

        auto run_layer_pre = [&](int il, bool include_route_t1 = false) {
            const int ip = hp.loop_layer(il);
            PhysLayer& L = m->layers[(size_t) ip];
            const bool is_swa = hp.is_swa(il);
            pre_routed_pp_cur = false;

            const auto tp_d0 = Clock::now();
            int dspark_ti = -1;
            if (capture_dspark && dspark) {
                const auto& tlayers = dspark->dm->hp.target_layers;
                for (int ti = 0; ti < 3; ++ti) {
                    if (il == tlayers[(size_t) ti]) { dspark_ti = ti; break; }
                }
            }
            const bool is_dspark_tgt = (dspark_ti >= 0);
            const float* cur_in = cur;
            auto emit_dspark_capture = [&]() {
                if (!is_dspark_tgt) return;
                if (dspark->prefill_vram_acc_active) {
                    if (aux_st != nullptr && ev_dspark_acc_src && ev_dspark_acc_done &&
                        !capturing_full_graph && !prof && !prof_detail) {
                        CK(cudaEventRecord(ev_dspark_acc_src, st));
                        CK(cudaStreamWaitEvent(aux_st, ev_dspark_acc_src, 0));
                        dspark->accum_prefill_feat(dspark_ti, cur_in, T, aux_st);
                        CK(cudaEventRecord(ev_dspark_acc_done, aux_st));
                        dspark_acc_pending = true;
                    } else {
                        dspark->accum_prefill_feat(dspark_ti, cur_in, T, st);
                    }
                } else if (d_tgt_feat[dspark_ti]) {
                    CK(cudaMemcpyAsync(d_tgt_feat[dspark_ti], cur_in, (size_t) T * N * sizeof(float), cudaMemcpyDeviceToDevice, st));
                }
            };
            auto flush_deferred_postffn_to_cur = [&]() {
                if (!deferred_postffn.active) return;
                const int n4 = N >> 2;
                k_shortconv_pp_f4<<<dim3((unsigned) ((n4 + 63) / 64), (unsigned) ((T + 63) / 64)), 64, 0, st>>>(
                    moe_raw, deferred_postffn.w_conv, deferred_postffn.hst, cur, N, deferred_postffn.hist_offset, T, ffn_inp);
                k_shortconv_hist_push<<<(unsigned) ((N + 255) / 256), 256, 0, st>>>(
                    moe_raw, deferred_postffn.hst, N, deferred_postffn.hist_offset, T);
                deferred_postffn.active = false;
            };
            bool has_msg = false;
            float* ir_m0 = ctrl_inv_rms ? (ctrl_inv_rms + (size_t) 0 * T) : nullptr;
            float* ir_m1 = ctrl_inv_rms ? (ctrl_inv_rms + (size_t) 1 * T) : nullptr;
            float* ir_m2 = ctrl_inv_rms ? (ctrl_inv_rms + (size_t) 2 * T) : nullptr;
            float* ir_m3 = ctrl_inv_rms ? (ctrl_inv_rms + (size_t) 3 * T) : nullptr;
            float* ir_c0 = ctrl_inv_rms ? (ctrl_inv_rms + (size_t) 4 * T) : nullptr;
            float* ir_c1 = ctrl_inv_rms ? (ctrl_inv_rms + (size_t) 5 * T) : nullptr;
            float* ir_c2 = ctrl_inv_rms ? (ctrl_inv_rms + (size_t) 6 * T) : nullptr;
            bool has_msg_ir = false;
            bool use_vmsg_layer = false;
            // EX-4C Wave 2: When T > 8 and all attention projections consume FP16, skip dead FP32 in_q/in_k/in_v writes
            // AND emit f16_v directly into disjoint pre_attn_f16_v (eliminates 5.2 GB HBM writes + 24x launch_f32_to_f16(in_v)!).
            const bool attn_f16_only_ok = (T > 8 && act_bf16 != nullptr && pre_attn_f16_v != nullptr &&
                                           L.attn_q.type != 0 && L.attn_k.type != 0 && L.attn_v.type != 0 &&
                                           L.attn_gate.type != 0 && L.attn_rel.type != 0 &&
                                           (bf16_x_f16 || (L.attn_q.type != 30 && L.attn_k.type != 30 && L.attn_v.type != 30 &&
                                                           L.attn_gate.type != 30 && L.attn_rel.type != 30)));
            uint16_t* const f16_v_buf = attn_f16_only_ok ? pre_attn_f16_v : nullptr;
            if (n_core > 0 && ip >= hp.core_start && ip < hp.core_end) {
                const int pass = (il - (int) hp.core_start) / (int) n_core;
                static const bool no_vmsg_cublas = (std::getenv("STRATA_NO_VMSG_CUBLAS") != nullptr);
                static const bool no_vmsg_hoist = (std::getenv("STRATA_NO_VMSG_HOIST") != nullptr);
                auto a256_v = [](size_t n) -> size_t { return (n + 255u) & ~size_t(255u); };
                const size_t sz_bank_f16 = a256_v((size_t) T * (size_t) N * sizeof(uint16_t));
                const size_t sz_bpre_f32 = a256_v((size_t) T * (size_t) 24 * 64 * sizeof(float));
                const bool can_vmsg_cublas = (!no_vmsg_cublas && d_core_wdown_msg_f16 != nullptr &&
                                              cc_maj >= 8 && N == 2304 && Q != nullptr &&
                                              (2 * sz_bank_f16 + 2 * sz_bpre_f32) <= reg1_bytes_cur);
                uint8_t* reg1_u8 = reinterpret_cast<uint8_t*>(Q);
                uint16_t* bank0_f16 = reinterpret_cast<uint16_t*>(reg1_u8);
                uint16_t* bank1_f16 = reinterpret_cast<uint16_t*>(reg1_u8 + sz_bank_f16);
                float* B0_pre       = reinterpret_cast<float*>(reg1_u8 + 2 * sz_bank_f16);
                float* B1_pre       = reinterpret_cast<float*>(reg1_u8 + 2 * sz_bank_f16 + sz_bpre_f32);
                if (ip == hp.core_start) {
                    if (pass == 0) {
                        prelude_norm_blocks = norm_blocks;
                        if (!deferred_postffn.active) {
                            CK(cudaMemcpyAsync(prelude_cur, cur, (size_t) T * N * sizeof(float), cudaMemcpyDeviceToDevice, st));
                        }
                        vmsg_pass_ready = false;
                        vmsg_bank_f16_ready = false;
                        vmsg_hoist_base_l = -1;
                    } else {
                        flush_deferred_postffn_to_cur();
                        norm_bank.clear();
                        for (size_t bi = prelude_norm_blocks.size() + 1; bi < norm_blocks.size(); ++bi) norm_bank.push_back(norm_blocks[bi]);
                        k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(cur, nullptr, bank_last_norm, N, hp.rms_eps);
                        norm_blocks = prelude_norm_blocks;
                        // Wave 12: Alias cur_in directly to prelude_cur instead of 2x 75.5 MB D2D copy;
                        // cur is never mutated in-place before run_layer_post overwrites it!
                        cur_in = prelude_cur;
                        static const bool no_vmsg = (std::getenv("STRATA_NO_VMSG") != nullptr);
                        vmsg_hoist_base_l = -1;
                        vmsg_bank_f16_ready = false;
                        if (!no_vmsg && cc_maj >= 8 && T >= 32 && N == 2304 && hp.loop_gate_rank == 64 &&
                            n_core == 12 && norm_bank.size() == 1 && vmsg_w && vmsg_inv_rms && ctrl_inv_rms) {
                            vmsg_bank0 = norm_bank[0];
                            vmsg_bank1 = bank_last_norm;
                            LoopBankQueries12 qs{};
                            for (int l = 0; l < 12; ++l) {
                                const int64_t r = 2 * l;
                                qs.q[l][0] = m->loop_res_query + (r + 0) * N;
                                qs.q[l][1] = m->loop_kv_res_query + (r + 0) * N;
                                qs.q[l][2] = m->loop_kv_res_query + (r + 1) * N;
                                qs.q[l][3] = m->loop_res_query + (r + 1) * N;
                            }
                            const bool emit_bank_f16 = (can_vmsg_cublas && !no_vmsg_hoist &&
                                                        vmsg_down_pre != nullptr && can_vmsg_all_core);
                            k_loop_bank_weights_pp_tiled<<<(unsigned) ((T + 31) / 32), 256, 0, st>>>(
                                vmsg_bank0, vmsg_bank1, qs, vmsg_w, vmsg_inv_rms, T, N, hp.rms_eps,
                                emit_bank_f16 ? bank0_f16 : nullptr,
                                emit_bank_f16 ? bank1_f16 : nullptr);
                            vmsg_bank_f16_ready = emit_bank_f16;
                            vmsg_pass_ready = true;
                        } else {
                            vmsg_pass_ready = false;
                        }
                    }
                }
                if (pass > 0) {
                    has_msg = true;
                    const bool vmsg_ctrl_types_ok =
                        L.loop_down[0].type == 0 && L.loop_down[1].type == 0 && L.loop_down[2].type == 0 && L.loop_down[3].type == 0 &&
                        L.loop_gain[0].type == 0 && L.loop_gain[1].type == 0 && L.loop_gain[2].type == 0 && L.loop_gain[3].type == 0 &&
                        L.loop_forget[0].type == 0 && L.loop_forget[1].type == 0 && L.loop_forget[2].type == 0 && L.loop_forget[3].type == 0;
                    const bool vmsg_attn_f16_ok =
                        attn_f16_only_ok && f16_v_buf != nullptr && L.attn_sink_proj.type != 0 &&
                        (is_swa || (L.attn_k_conv_proj.type != 0 && L.attn_v_conv_proj.type != 0));
                    use_vmsg_layer = (vmsg_pass_ready && L.attn_res_query != nullptr && vmsg_ctrl_types_ok && vmsg_attn_f16_ok);
                    if (use_vmsg_layer) {
                        const int l_idx_h = ip - (int) hp.core_start;
                        if (!no_vmsg_hoist && vmsg_down_pre != nullptr && d_vmsg_wdown_ptrs != nullptr &&
                            can_vmsg_all_core && (l_idx_h == 0 || l_idx_h == 6)) {
                            const int l_start = (l_idx_h < 6) ? 0 : 6;
                            const int n_mods_half = std::min(6, (int) n_core - l_start) * 4;
                            const bool moe_raw_disjoint_reg1 =
                                (reinterpret_cast<uint8_t*>(moe_raw) + (size_t) T * (size_t) N * sizeof(float) <= reg1_u8);
                            const bool use_vmsg_cublas = (can_vmsg_cublas && (!deferred_postffn.active || moe_raw_disjoint_reg1));
                            if (use_vmsg_cublas) {
                                if (l_idx_h == 0 && vmsg_bank_f16_ready) {
                                    // bank0_f16 & bank1_f16 already emitted by k_loop_bank_weights_pp_tiled (0.00 ms cast!)
                                } else {
                                    launch_f32_to_f16(vmsg_bank0, bank0_f16, (int64_t) T * N);
                                    launch_f32_to_f16(vmsg_bank1, bank1_f16, (int64_t) T * N);
                                }
                                vmsg_bank_f16_ready = false;
                                const uint16_t* w_half = d_core_wdown_msg_f16 + (size_t) (l_start * 4) * 64 * (size_t) N;
                                const float* vmsg_w_half = vmsg_w + (size_t) (l_start * 4) * (size_t) T * 2;
                                const float* vmsg_ir_half = vmsg_inv_rms + (size_t) (l_start * 4) * (size_t) T;
                                const size_t bpre_half_elems = (size_t) T * (size_t) n_mods_half * 64u;
                                if ((size_t) T * (size_t) N * sizeof(uint16_t) == sz_bank_f16 &&
                                    2u * bpre_half_elems <= 2u * sz_bpre_f32 / sizeof(float)) {
                                    // bank0_f16 & bank1_f16 form a gapless [2*T, N] row-major FP16 matrix; fuse into 1x 2*T GEMM.
                                    float* const B1_merged = B0_pre + bpre_half_elems;
                                    gemm.f16(bank0_f16, w_half, B0_pre, (int64_t) 2 * T, (int64_t) n_mods_half * 64, N);
                                    k_vmsg_combine_pre_f32<<<dim3((unsigned) ((T + 7) / 8), (unsigned) n_mods_half), 128, 0, st>>>(
                                        B0_pre, B1_merged, vmsg_w_half, vmsg_ir_half, vmsg_down_pre, T, n_mods_half);
                                } else {
                                    gemm.f16(bank0_f16, w_half, B0_pre, T, (int64_t) n_mods_half * 64, N);
                                    gemm.f16(bank1_f16, w_half, B1_pre, T, (int64_t) n_mods_half * 64, N);
                                    k_vmsg_combine_pre_f32<<<dim3((unsigned) ((T + 7) / 8), (unsigned) n_mods_half), 128, 0, st>>>(
                                        B0_pre, B1_pre, vmsg_w_half, vmsg_ir_half, vmsg_down_pre, T, n_mods_half);
                                }
                            } else {
                                k_vmsg_down_pre_pass<<<dim3((unsigned) ((T + 31) / 32), (unsigned) n_mods_half), 256, 0, st>>>(
                                    vmsg_bank0, vmsg_bank1, d_vmsg_wdown_ptrs + l_start * 4,
                                    vmsg_w + (size_t) (l_start * 4) * (size_t) T * 2, nullptr,
                                    vmsg_inv_rms + (size_t) (l_start * 4) * (size_t) T,
                                    vmsg_down_pre, T, N, n_mods_half);
                            }
                            vmsg_hoist_base_l = l_start;
                        }
                    } else {
                        const int64_t row = 2 * (ip - hp.core_start);
                        const float* q_attn = m->loop_res_query + row * N;
                        const float* q_kv = m->loop_kv_res_query + row * N;
                        NormMixPtrs np{};
                        const int B = (int) norm_bank.size();
                        for (int i = 0; i < B; ++i) np.p[i] = norm_bank[(size_t) (B - 1 - i)];
                        np.p[B] = bank_last_norm;
                        MixOuts4 mo4{};
                        mo4.p[0] = msg0; mo4.p[1] = msg3; mo4.p[2] = msg1; mo4.p[3] = msg2;
                        if (T >= 1 && N == 2304 && B + 1 <= 8) {
                            MixInvRms4 ir_m4{{nullptr, nullptr, nullptr, nullptr}};
                            if (T >= 32 && can_ctrl_down_wmma && ir_m0) {
                                ir_m4 = MixInvRms4{{ir_m0, ir_m3, ir_m1, ir_m2}};
                                has_msg_ir = true;
                            }
                            k_mix_normed_fused_1pass<4><<<(unsigned) T, 576, 0, st>>>(
                                np, B + 1, q_attn, q_kv, mo4, N, 1.0f / std::sqrt((float) N), 0, hp.rms_eps, ir_m4);
                        } else if (T >= 1 && (N & 3) == 0 && B + 1 <= 8) {
                            k_mix_normed_fused<4><<<(unsigned) T, kThreads, 0, st>>>(
                                np, B + 1, q_attn, q_kv, mo4, N, 1.0f / std::sqrt((float) N), 0);
                        } else {
                            k_mix_normed_4<<<dim3(4, (unsigned) T), kThreads, 0, st>>>(
                                np, B + 1, q_attn, q_kv, mo4, N, 1.0f / std::sqrt((float) N));
                        }
                    }
                }
            }

            const float* ink_src = in_k;
            const float* inv_src = in_v;
            if (!deferred_postffn.active) {
                emit_dspark_capture();
            }
            const bool fuse_q8_norm = (T <= 8 && (N & 31) == 0);
            const int rms_q8_blk = (N == 2304) ? 576 : kThreads;
            const int rms_f16_blk = (N == 2304) ? 576 : kThreads;
            bool q8_pre_done = false;
            bool f16_v_pre_done = false;
            if (L.attn_res_query) {
                float* outs[3] = {in_q, in_k, in_v};
                float* outs_opt[3] = {attn_f16_only_ok ? nullptr : in_q,
                                      attn_f16_only_ok ? nullptr : in_k,
                                      attn_f16_only_ok ? nullptr : in_v};
                bool fused_mix_norm = false;
                if (!has_msg && deferred_postffn.active && T > 8 && N == 2304 && (int) norm_blocks.size() <= 7) {
                    const bool do_push_cur = (ip % hp.attn_res_block_size == 0);
                    const int pass_cur = (n_core > 0 && ip >= hp.core_start && ip < hp.core_end)
                                       ? ((il - (int) hp.core_start) / (int) n_core) : 0;
                    const bool is_prelude_save = (n_core > 0 && ip == hp.core_start && pass_cur == 0);
                    bool skip_push_recomp = false;
                    float* ndst_cur = nullptr;
                    if (do_push_cur) {
                        const int slot_idx = pick_blk_slot(il, ip, pass_cur, skip_push_recomp);
                        ndst_cur = norm_blk_buf[slot_idx];
                        pre_push_done_il = il;
                        pre_push_ndst = ndst_cur;
                    }
                    float* cur_out_ptr = (is_prelude_save && !is_dspark_tgt)
                        ? prelude_cur
                        : ((do_push_cur && !is_dspark_tgt) ? nullptr : cur);
                    float* ndst_out_ptr = (do_push_cur && !skip_push_recomp) ? ndst_cur : nullptr;
                    const int B = (int) norm_blocks.size();
                    NormMixPtrs np{};
                    for (int i = 0; i < B; ++i) np.p[i] = norm_blocks[(size_t) (B - 1 - i)];
                    MixInvRms4 ir_empty{{nullptr, nullptr, nullptr, nullptr}};
                    auto launch_postffn_b = [&](auto nq_tag, auto dp_tag, float* ndst_p, MixOuts4& mo4_arg, NormF16Out3& nf16_arg) {
                        constexpr int NQ_T = decltype(nq_tag)::value;
                        constexpr bool DP_T = decltype(dp_tag)::value;
                        switch (B) {
                        case 0: k_postffn_shortconv_preattn_mix_pp<NQ_T, true, DP_T, 0><<<(unsigned) T, 576, 0, st>>>(moe_raw, deferred_postffn.w_conv, deferred_postffn.hst, ffn_inp, cur_out_ptr, ndst_p, np, B, L.attn_res_query, mo4_arg, nf16_arg, ir_empty, deferred_postffn.hist_offset, 0, T, N, hp.rms_eps, false); break;
                        case 1: k_postffn_shortconv_preattn_mix_pp<NQ_T, true, DP_T, 1><<<(unsigned) T, 576, 0, st>>>(moe_raw, deferred_postffn.w_conv, deferred_postffn.hst, ffn_inp, cur_out_ptr, ndst_p, np, B, L.attn_res_query, mo4_arg, nf16_arg, ir_empty, deferred_postffn.hist_offset, 0, T, N, hp.rms_eps, false); break;
                        case 2: k_postffn_shortconv_preattn_mix_pp<NQ_T, true, DP_T, 2><<<(unsigned) T, 576, 0, st>>>(moe_raw, deferred_postffn.w_conv, deferred_postffn.hst, ffn_inp, cur_out_ptr, ndst_p, np, B, L.attn_res_query, mo4_arg, nf16_arg, ir_empty, deferred_postffn.hist_offset, 0, T, N, hp.rms_eps, false); break;
                        case 3: k_postffn_shortconv_preattn_mix_pp<NQ_T, true, DP_T, 3><<<(unsigned) T, 576, 0, st>>>(moe_raw, deferred_postffn.w_conv, deferred_postffn.hst, ffn_inp, cur_out_ptr, ndst_p, np, B, L.attn_res_query, mo4_arg, nf16_arg, ir_empty, deferred_postffn.hist_offset, 0, T, N, hp.rms_eps, false); break;
                        case 4: k_postffn_shortconv_preattn_mix_pp<NQ_T, true, DP_T, 4><<<(unsigned) T, 576, 0, st>>>(moe_raw, deferred_postffn.w_conv, deferred_postffn.hst, ffn_inp, cur_out_ptr, ndst_p, np, B, L.attn_res_query, mo4_arg, nf16_arg, ir_empty, deferred_postffn.hist_offset, 0, T, N, hp.rms_eps, false); break;
                        default: k_postffn_shortconv_preattn_mix_pp<NQ_T, true, DP_T, -1><<<(unsigned) T, 576, 0, st>>>(moe_raw, deferred_postffn.w_conv, deferred_postffn.hst, ffn_inp, cur_out_ptr, ndst_p, np, B, L.attn_res_query, mo4_arg, nf16_arg, ir_empty, deferred_postffn.hist_offset, 0, T, N, hp.rms_eps, false); break;
                        }
                    };
                    if (is_swa) {
                        MixOuts4 mo4{{outs_opt[0], nullptr, nullptr, nullptr}};
                        NormF16Out3 nf16{L.attn_norm, nullptr, nullptr, act_bf16, nullptr, nullptr};
                        if (do_push_cur && ndst_out_ptr != nullptr) {
                            launch_postffn_b(std::integral_constant<int, 1>{}, std::true_type{}, ndst_out_ptr, mo4, nf16);
                        } else {
                            launch_postffn_b(std::integral_constant<int, 1>{}, std::false_type{}, nullptr, mo4, nf16);
                        }
                        ink_src = in_q;
                        inv_src = in_q;
                    } else {
                        MixOuts4 mo4{{outs_opt[0], outs_opt[1], outs_opt[2], nullptr}};
                        NormF16Out3 nf16{L.attn_q_inp_norm, L.attn_k_inp_norm, L.attn_v_inp_norm,
                                         act_bf16, act_bf16 + (size_t) T * N, f16_v_buf};
                        if (do_push_cur && ndst_out_ptr != nullptr) {
                            launch_postffn_b(std::integral_constant<int, 3>{}, std::true_type{}, ndst_out_ptr, mo4, nf16);
                        } else {
                            launch_postffn_b(std::integral_constant<int, 3>{}, std::false_type{}, nullptr, mo4, nf16);
                        }
                        f16_v_pre_done = (f16_v_buf != nullptr);
                    }
                    k_shortconv_hist_push<<<(unsigned) ((N + 255) / 256), 256, 0, st>>>(
                        moe_raw, deferred_postffn.hst, N, deferred_postffn.hist_offset, T);
                    deferred_postffn.active = false;
                    if (is_prelude_save && is_dspark_tgt) {
                        CK(cudaMemcpyAsync(prelude_cur, cur, (size_t) T * N * sizeof(float), cudaMemcpyDeviceToDevice, st));
                    }
                    emit_dspark_capture();
                    fused_mix_norm = true;
                } else if (!has_msg && fuse_q8_norm) {
                    if (is_swa) {
                        NormQ8Out3 nq8{L.attn_norm, nullptr, nullptr, xq, nullptr, nullptr};
                        if (try_run_mix_normed_q8(norm_blocks, cur_in, L.attn_res_query, outs, nq8, 1, T)) {
                            fused_mix_norm = true;
                            q8_pre_done = true;
                            ink_src = in_q;
                            inv_src = in_q;
                        }
                    } else {
                        NormQ8Out3 nq8{L.attn_q_inp_norm, L.attn_k_inp_norm, L.attn_v_inp_norm, xq, xq_k, xq_v};
                        if (try_run_mix_normed_q8(norm_blocks, cur_in, L.attn_res_query, outs, nq8, 3, T)) {
                            fused_mix_norm = true;
                            q8_pre_done = true;
                        }
                    }
                } else if (!has_msg && T > 8 && N == 2304) {
                    if (is_swa) {
                        NormF16Out3 nf16{L.attn_norm, nullptr, nullptr, act_bf16, nullptr, nullptr};
                        if (try_run_mix_normed_f16(norm_blocks, cur_in, L.attn_res_query, outs_opt, nf16, 1, T)) {
                            fused_mix_norm = true;
                            ink_src = in_q;
                            inv_src = in_q;
                        }
                    } else {
                        NormF16Out3 nf16{L.attn_q_inp_norm, L.attn_k_inp_norm, L.attn_v_inp_norm,
                                         act_bf16, act_bf16 + (size_t) T * N, f16_v_buf};
                        if (try_run_mix_normed_f16(norm_blocks, cur_in, L.attn_res_query, outs_opt, nf16, 3, T)) {
                            fused_mix_norm = true;
                            f16_v_pre_done = (f16_v_buf != nullptr);
                        }
                    }
                }
                if (!fused_mix_norm) {
                if (use_vmsg_layer) {
                    static const bool no_l2_chunk = (std::getenv("STRATA_NO_L2_CHUNK") != nullptr);
                    static const bool fuse_ctrl_rms_attn = [] {
                        const char* v = std::getenv("STRATA_FUSE_CTRL_RMS");
                        return v != nullptr && (v[0] == '\0' || std::atoi(v) != 0);
                    }();
                    const bool use_fuse_ctrl_rms_3 = (attn_f16_only_ok && can_ctrl_rms_3 && fuse_ctrl_rms_attn &&
                                                      cc_maj >= 8 && T >= 128 && N == 2304 && !save_conv_trace &&
                                                      f16_v_buf != nullptr && L.attn_sink_proj.type != 0 &&
                                                      (bf16_x_f16 || (L.attn_q.type != 30 && L.attn_gate.type != 30 && L.attn_rel.type != 30)));
                    const int l_idx = ip - (int) hp.core_start;
                    const float* w_m0 = vmsg_w + (size_t) (l_idx * 4 + 0) * (size_t) T * 2;
                    const float* w_m1 = vmsg_w + (size_t) (l_idx * 4 + 1) * (size_t) T * 2;
                    const float* w_m2 = vmsg_w + (size_t) (l_idx * 4 + 2) * (size_t) T * 2;
                    const float* im_m0 = vmsg_inv_rms + (size_t) (l_idx * 4 + 0) * (size_t) T;
                    const float* im_m1 = vmsg_inv_rms + (size_t) (l_idx * 4 + 1) * (size_t) T;
                    const float* im_m2 = vmsg_inv_rms + (size_t) (l_idx * 4 + 2) * (size_t) T;
                    const float* nw0 = is_swa ? L.attn_norm : L.attn_q_inp_norm;
                    const float* nw1 = is_swa ? L.attn_norm : L.attn_k_inp_norm;
                    const float* nw2 = is_swa ? L.attn_norm : L.attn_v_inp_norm;
                    const bool has_vmsg_hoist = (vmsg_hoist_base_l >= 0 && l_idx >= vmsg_hoist_base_l &&
                                                 l_idx < vmsg_hoist_base_l + 6 && vmsg_down_pre != nullptr);
                    const bool use_l2_chunk_attn = (!no_l2_chunk && has_vmsg_hoist && Q != nullptr && T > 64 && cc_maj >= 8 &&
                                                    N == 2304 && (int) norm_blocks.size() <= 7);
                    const bool do_push_cur = (ip % hp.attn_res_block_size == 0);
                    const int pass_cur = (il - (int) hp.core_start) / (int) n_core;
                    bool skip_push_recomp = false;
                    float* ndst_cur = nullptr;
                    if (deferred_postffn.active && do_push_cur) {
                        const int slot_idx = pick_blk_slot(il, ip, pass_cur, skip_push_recomp);
                        ndst_cur = norm_blk_buf[slot_idx];
                        pre_push_done_il = il;
                        pre_push_ndst = ndst_cur;
                    }
                    float* cur_out_ptr = (do_push_cur && !is_dspark_tgt) ? nullptr : cur;
                    float* ndst_out_ptr = (do_push_cur && !skip_push_recomp) ? ndst_cur : nullptr;
                    MixInvRms4 ir_c4 = is_swa ? MixInvRms4{{ir_c0, nullptr, nullptr, nullptr}}
                                              : MixInvRms4{{ir_c0, ir_c1, ir_c2, nullptr}};
                    if (use_l2_chunk_attn) {
                        static const int env_ctrl_up3_chunk = [] {
                            const char* v = std::getenv("STRATA_CTRL_UP3_CHUNK");
                            return (v != nullptr && v[0] != '\0') ? std::atoi(v) : 0;
                        }();
                        const int default_ctrl_up3_chunk = 640;
                        const int T_CHUNK = (env_ctrl_up3_chunk > 0) ? env_ctrl_up3_chunk : default_ctrl_up3_chunk;
                        const int local_m0 = (l_idx - vmsg_hoist_base_l) * 4;
                        const int B = (int) norm_blocks.size();
                        const float inv_sqrt_n = 1.0f / std::sqrt((float) N);
                        NormMixPtrs np{};
                        for (int i = 0; i < B; ++i) np.p[i] = norm_blocks[(size_t) (B - 1 - i)];
                        MixOuts4 mo4{{in_q, is_swa ? nullptr : in_k, is_swa ? nullptr : in_v, nullptr}};
                        NormF16Out3 nf16_empty{};
                        for (int tc0 = 0; tc0 < T; tc0 += T_CHUNK) {
                            const int tc_len = std::min(T_CHUNK, T - tc0);
                            // Wave 7: Use Q (reg1 base, 100% idle during Pre-Attn) for c64_0/1/2 so they never
                            // alias f16_v_buf (pre_attn_f16_v inside reg2_ctrl_dyn, which caused NaN on tc0 >= T_CHUNK).
                            float* c64_0 = Q;
                            float* c64_1 = Q + (size_t) tc_len * 64;
                            float* c64_2 = Q + (size_t) tc_len * 128;
                            if (deferred_postffn.active) {
                                auto launch_postffn_msg_b = [&](auto nq_tag, auto dp_tag, float* ndst_p) {
                                    constexpr int NQ_T = decltype(nq_tag)::value;
                                    constexpr bool DP_T = decltype(dp_tag)::value;
                                    switch (B) {
                                    case 0: k_postffn_shortconv_preattn_mix_pp<NQ_T, false, DP_T, 0><<<(unsigned) tc_len, 576, 0, st>>>(moe_raw, deferred_postffn.w_conv, deferred_postffn.hst, ffn_inp, cur_out_ptr, ndst_p, np, B, L.attn_res_query, mo4, nf16_empty, ir_c4, deferred_postffn.hist_offset, tc0, tc_len, N, hp.rms_eps, true); break;
                                    case 1: k_postffn_shortconv_preattn_mix_pp<NQ_T, false, DP_T, 1><<<(unsigned) tc_len, 576, 0, st>>>(moe_raw, deferred_postffn.w_conv, deferred_postffn.hst, ffn_inp, cur_out_ptr, ndst_p, np, B, L.attn_res_query, mo4, nf16_empty, ir_c4, deferred_postffn.hist_offset, tc0, tc_len, N, hp.rms_eps, true); break;
                                    case 2: k_postffn_shortconv_preattn_mix_pp<NQ_T, false, DP_T, 2><<<(unsigned) tc_len, 576, 0, st>>>(moe_raw, deferred_postffn.w_conv, deferred_postffn.hst, ffn_inp, cur_out_ptr, ndst_p, np, B, L.attn_res_query, mo4, nf16_empty, ir_c4, deferred_postffn.hist_offset, tc0, tc_len, N, hp.rms_eps, true); break;
                                    case 3: k_postffn_shortconv_preattn_mix_pp<NQ_T, false, DP_T, 3><<<(unsigned) tc_len, 576, 0, st>>>(moe_raw, deferred_postffn.w_conv, deferred_postffn.hst, ffn_inp, cur_out_ptr, ndst_p, np, B, L.attn_res_query, mo4, nf16_empty, ir_c4, deferred_postffn.hist_offset, tc0, tc_len, N, hp.rms_eps, true); break;
                                    case 4: k_postffn_shortconv_preattn_mix_pp<NQ_T, false, DP_T, 4><<<(unsigned) tc_len, 576, 0, st>>>(moe_raw, deferred_postffn.w_conv, deferred_postffn.hst, ffn_inp, cur_out_ptr, ndst_p, np, B, L.attn_res_query, mo4, nf16_empty, ir_c4, deferred_postffn.hist_offset, tc0, tc_len, N, hp.rms_eps, true); break;
                                    default: k_postffn_shortconv_preattn_mix_pp<NQ_T, false, DP_T, -1><<<(unsigned) tc_len, 576, 0, st>>>(moe_raw, deferred_postffn.w_conv, deferred_postffn.hst, ffn_inp, cur_out_ptr, ndst_p, np, B, L.attn_res_query, mo4, nf16_empty, ir_c4, deferred_postffn.hist_offset, tc0, tc_len, N, hp.rms_eps, true); break;
                                    }
                                };
                                if (is_swa) {
                                    if (do_push_cur && ndst_out_ptr != nullptr) {
                                        launch_postffn_msg_b(std::integral_constant<int, 1>{}, std::true_type{}, ndst_out_ptr);
                                    } else {
                                        launch_postffn_msg_b(std::integral_constant<int, 1>{}, std::false_type{}, nullptr);
                                    }
                                } else {
                                    if (do_push_cur && ndst_out_ptr != nullptr) {
                                        launch_postffn_msg_b(std::integral_constant<int, 3>{}, std::true_type{}, ndst_out_ptr);
                                    } else {
                                        launch_postffn_msg_b(std::integral_constant<int, 3>{}, std::false_type{}, nullptr);
                                    }
                                }
                            } else {
                                NormMixPtrs np_off{};
                                for (int i = 0; i < B; ++i) np_off.p[i] = norm_blocks[(size_t) (B - 1 - i)] + (size_t) tc0 * N;
                                np_off.p[B] = cur_in + (size_t) tc0 * N;
                                if (is_swa) {
                                    k_mix_normed_fused_1pass<1, true><<<(unsigned) tc_len, 576, 0, st>>>(
                                        np_off, B + 1, L.attn_res_query, nullptr, mo4, N, inv_sqrt_n, 0, hp.rms_eps, ir_c4);
                                } else {
                                    k_mix_normed_fused_1pass<3, true><<<(unsigned) tc_len, 576, 0, st>>>(
                                        np_off, B + 1, L.attn_res_query, nullptr, mo4, N, inv_sqrt_n, 0, hp.rms_eps, ir_c4);
                                }
                            }
                            CtrlDownPhaseAPtrs3 dp_a{};
                            dp_a.c[0] = in_q; dp_a.c[1] = is_swa ? in_q : in_k; dp_a.c[2] = is_swa ? in_q : in_v;
                            dp_a.inv_c[0] = ir_c0; dp_a.inv_c[1] = is_swa ? ir_c0 : ir_c1; dp_a.inv_c[2] = is_swa ? ir_c0 : ir_c2;
                            dp_a.w[0] = (const float*) L.loop_down[0].d;
                            dp_a.w[1] = (const float*) L.loop_down[1].d;
                            dp_a.w[2] = (const float*) L.loop_down[2].d;
                            dp_a.w_f16[0] = d_core_wdown_cur_f16 ? (d_core_wdown_cur_f16 + (size_t) (l_idx * 4 + 0) * 64u * (size_t) N) : nullptr;
                            dp_a.w_f16[1] = d_core_wdown_cur_f16 ? (d_core_wdown_cur_f16 + (size_t) (l_idx * 4 + 1) * 64u * (size_t) N) : nullptr;
                            dp_a.w_f16[2] = d_core_wdown_cur_f16 ? (d_core_wdown_cur_f16 + (size_t) (l_idx * 4 + 2) * 64u * (size_t) N) : nullptr;
                            dp_a.pre[0] = vmsg_down_pre + (size_t) (local_m0 + 0) * (size_t) T * 64 + (size_t) tc0 * 64;
                            dp_a.pre[1] = vmsg_down_pre + (size_t) (local_m0 + 1) * (size_t) T * 64 + (size_t) tc0 * 64;
                            dp_a.pre[2] = vmsg_down_pre + (size_t) (local_m0 + 2) * (size_t) T * 64 + (size_t) tc0 * 64;
                            dp_a.out[0] = c64_0; dp_a.out[1] = c64_1; dp_a.out[2] = c64_2;
                            const size_t c64_used_floats = (size_t) 3 * (size_t) tc_len * 64u;
                            const size_t reg1_floats = reg1_bytes_cur / sizeof(float);
                            const size_t part_cap = (reg1_floats > c64_used_floats) ? (reg1_floats - c64_used_floats) : 0;
                            launch_phaseA_splitk<3>(dp_a, tc_len, N, Q + c64_used_floats, part_cap);

                            CtrlUpVmsgPtrs3 up{};
                            up.c[0] = in_q; up.c[1] = is_swa ? in_q : in_k; up.c[2] = is_swa ? in_q : in_v;
                            up.w_vmsg[0] = w_m0 + (size_t) tc0 * 2;
                            up.w_vmsg[1] = w_m1 + (size_t) tc0 * 2;
                            up.w_vmsg[2] = w_m2 + (size_t) tc0 * 2;
                            up.ctrl_in[0] = c64_0; up.ctrl_in[1] = c64_1; up.ctrl_in[2] = c64_2;
                            up.w_gain[0] = (const float*) L.loop_gain[0].d;
                            up.w_gain[1] = (const float*) L.loop_gain[1].d;
                            up.w_gain[2] = (const float*) L.loop_gain[2].d;
                            up.w_forget[0] = (const float*) L.loop_forget[0].d;
                            up.w_forget[1] = (const float*) L.loop_forget[1].d;
                            up.w_forget[2] = (const float*) L.loop_forget[2].d;
                            if (d_core_wup_f16) {
                                const size_t wup_m = (size_t) N * 64u;
                                up.wg_f16[0] = d_core_wup_f16 + (size_t) (l_idx * 4 + 0) * wup_m;
                                up.wg_f16[1] = d_core_wup_f16 + (size_t) (l_idx * 4 + 1) * wup_m;
                                up.wg_f16[2] = d_core_wup_f16 + (size_t) (l_idx * 4 + 2) * wup_m;
                                up.wf_f16[0] = d_core_wup_f16 + (size_t) (48 + l_idx * 4 + 0) * wup_m;
                                up.wf_f16[1] = d_core_wup_f16 + (size_t) (48 + l_idx * 4 + 1) * wup_m;
                                up.wf_f16[2] = d_core_wup_f16 + (size_t) (48 + l_idx * 4 + 2) * wup_m;
                            }
                            up.out[0] = in_q; up.out[1] = in_k; up.out[2] = in_v;
                            if (use_fuse_ctrl_rms_3) {
                                up.out[0] = nullptr; up.out[1] = nullptr; up.out[2] = nullptr;
                                CtrlUpRmsOuts ro{};
                                ro.rms_w[0] = nw0; ro.rms_w[1] = nw1; ro.rms_w[2] = nw2;
                                ro.out_f16[0] = act_bf16 + (size_t) tc0 * N;
                                ro.out_f16[1] = act_bf16 + (size_t) (T + tc0) * N;
                                ro.out_f16[2] = f16_v_buf + (size_t) tc0 * N;
                                ro.out_lo_f16 = nullptr;
                                ro.eps = hp.rms_eps;
                                k_control_up_combine_rms_f16_vmsg<3><<<dim3((unsigned) ((tc_len + 15) / 16), 1, 1), 576, 29696, st>>>(
                                    vmsg_bank0 + (size_t) tc0 * N, vmsg_bank1 + (size_t) tc0 * N, up, ro, tc_len, N);
                            } else {
                                k_control_up_combine_pp_wmma_vmsg<3><<<dim3((unsigned) ((N + 63) / 64), (unsigned) ((tc_len + 63) / 64)), 512, 0, st>>>(
                                    vmsg_bank0 + (size_t) tc0 * N, vmsg_bank1 + (size_t) tc0 * N, up, tc_len, N);
                                if (N == 2304 && tc_len >= 64) {
                                    k_rms_norm_3_f16_x4<<<dim3(3, (unsigned) ((tc_len + 3) / 4)), 256, 0, st>>>(
                                        in_q, in_k, in_v, nullptr, nullptr, nullptr, nw0, nw1, nw2,
                                        act_bf16 + (size_t) tc0 * N, act_bf16 + (size_t) (T + tc0) * N, f16_v_buf + (size_t) tc0 * N,
                                        tc_len, hp.rms_eps);
                                } else {
                                    k_rms_norm_3_f16<<<dim3(3, (unsigned) tc_len), rms_f16_blk, 0, st>>>(
                                        in_q, in_k, in_v, nullptr, nullptr, nullptr, nw0, nw1, nw2,
                                        act_bf16 + (size_t) tc0 * N, act_bf16 + (size_t) (T + tc0) * N, f16_v_buf + (size_t) tc0 * N,
                                        N, hp.rms_eps);
                                }
                            }
                        }
                        if (deferred_postffn.active) {
                            k_shortconv_hist_push<<<(unsigned) ((N + 255) / 256), 256, 0, st>>>(
                                moe_raw, deferred_postffn.hst, N, deferred_postffn.hist_offset, T);
                            deferred_postffn.active = false;
                            emit_dspark_capture();
                        }
                        f16_v_pre_done = true;
                    } else {
                        if (deferred_postffn.active && N == 2304 && (int) norm_blocks.size() <= 7) {
                            const int B = (int) norm_blocks.size();
                            NormMixPtrs np{};
                            for (int i = 0; i < B; ++i) np.p[i] = norm_blocks[(size_t) (B - 1 - i)];
                            MixOuts4 mo4{{in_q, is_swa ? nullptr : in_k, is_swa ? nullptr : in_v, nullptr}};
                            NormF16Out3 nf16_empty{};
                            if (is_swa) {
                                if (do_push_cur && ndst_out_ptr != nullptr) {
                                    k_postffn_shortconv_preattn_mix_pp<1, false, true><<<(unsigned) T, 576, 0, st>>>(
                                        moe_raw, deferred_postffn.w_conv, deferred_postffn.hst, ffn_inp,
                                        cur_out_ptr, ndst_out_ptr, np, B, L.attn_res_query, mo4, nf16_empty, ir_c4,
                                        deferred_postffn.hist_offset, 0, T, N, hp.rms_eps, false);
                                } else {
                                    k_postffn_shortconv_preattn_mix_pp<1, false, false><<<(unsigned) T, 576, 0, st>>>(
                                        moe_raw, deferred_postffn.w_conv, deferred_postffn.hst, ffn_inp,
                                        cur_out_ptr, nullptr, np, B, L.attn_res_query, mo4, nf16_empty, ir_c4,
                                        deferred_postffn.hist_offset, 0, T, N, hp.rms_eps, false);
                                }
                            } else {
                                if (do_push_cur && ndst_out_ptr != nullptr) {
                                    k_postffn_shortconv_preattn_mix_pp<3, false, true><<<(unsigned) T, 576, 0, st>>>(
                                        moe_raw, deferred_postffn.w_conv, deferred_postffn.hst, ffn_inp,
                                        cur_out_ptr, ndst_out_ptr, np, B, L.attn_res_query, mo4, nf16_empty, ir_c4,
                                        deferred_postffn.hist_offset, 0, T, N, hp.rms_eps, false);
                                } else {
                                    k_postffn_shortconv_preattn_mix_pp<3, false, false><<<(unsigned) T, 576, 0, st>>>(
                                        moe_raw, deferred_postffn.w_conv, deferred_postffn.hst, ffn_inp,
                                        cur_out_ptr, nullptr, np, B, L.attn_res_query, mo4, nf16_empty, ir_c4,
                                        deferred_postffn.hist_offset, 0, T, N, hp.rms_eps, false);
                                }
                            }
                            k_shortconv_hist_push<<<(unsigned) ((N + 255) / 256), 256, 0, st>>>(
                                moe_raw, deferred_postffn.hst, N, deferred_postffn.hist_offset, T);
                            deferred_postffn.active = false;
                            emit_dspark_capture();
                        } else {
                            if (deferred_postffn.active) {
                                flush_deferred_postffn_to_cur();
                                emit_dspark_capture();
                            }
                            // For SWA: swa_bcast = 0 (writes ONLY in_q), then vmsg reads in_q for all 3 modules!
                            run_mix_normed(norm_blocks, cur_in, false, L.attn_res_query, outs, is_swa ? 1 : 3, T, 0, &ir_c4);
                        }
                        float* c64_0 = ctrl_f;
                        float* c64_1 = ctrl_f + (size_t) T * 64;
                        float* c64_2 = ctrl_f + (size_t) T * 128;
                        if (has_vmsg_hoist) {
                            const int local_m0 = (l_idx - vmsg_hoist_base_l) * 4;
                            CtrlDownPhaseAPtrs3 dp_a{};
                            dp_a.c[0] = in_q; dp_a.c[1] = is_swa ? in_q : in_k; dp_a.c[2] = is_swa ? in_q : in_v;
                            dp_a.inv_c[0] = ir_c0; dp_a.inv_c[1] = is_swa ? ir_c0 : ir_c1; dp_a.inv_c[2] = is_swa ? ir_c0 : ir_c2;
                            dp_a.w[0] = (const float*) L.loop_down[0].d;
                            dp_a.w[1] = (const float*) L.loop_down[1].d;
                            dp_a.w[2] = (const float*) L.loop_down[2].d;
                            dp_a.w_f16[0] = d_core_wdown_cur_f16 ? (d_core_wdown_cur_f16 + (size_t) (l_idx * 4 + 0) * 64u * (size_t) N) : nullptr;
                            dp_a.w_f16[1] = d_core_wdown_cur_f16 ? (d_core_wdown_cur_f16 + (size_t) (l_idx * 4 + 1) * 64u * (size_t) N) : nullptr;
                            dp_a.w_f16[2] = d_core_wdown_cur_f16 ? (d_core_wdown_cur_f16 + (size_t) (l_idx * 4 + 2) * 64u * (size_t) N) : nullptr;
                            dp_a.pre[0] = vmsg_down_pre + (size_t) (local_m0 + 0) * (size_t) T * 64;
                            dp_a.pre[1] = vmsg_down_pre + (size_t) (local_m0 + 1) * (size_t) T * 64;
                            dp_a.pre[2] = vmsg_down_pre + (size_t) (local_m0 + 2) * (size_t) T * 64;
                            dp_a.out[0] = c64_0; dp_a.out[1] = c64_1; dp_a.out[2] = c64_2;
                            launch_phaseA_splitk<3>(dp_a, T, N, Q, reg1_bytes_cur / sizeof(float));
                        } else {
                            CtrlDownVmsgPtrs3 dp{
                                {in_q, is_swa ? in_q : in_k, is_swa ? in_q : in_v},
                                {ir_c0, is_swa ? ir_c0 : ir_c1, is_swa ? ir_c0 : ir_c2},
                                {w_m0, w_m1, w_m2},
                                {im_m0, im_m1, im_m2},
                                {(const float*) L.loop_down[0].d, (const float*) L.loop_down[1].d, (const float*) L.loop_down[2].d},
                                {c64_0, c64_1, c64_2}
                            };
                            k_control_down_pp_wmma_vmsg<3><<<(unsigned) ((T + 31) / 32), 256, 0, st>>>(vmsg_bank0, vmsg_bank1, dp, T, N);
                        }
                        CtrlUpVmsgPtrs3 up{};
                        up.c[0] = in_q; up.c[1] = is_swa ? in_q : in_k; up.c[2] = is_swa ? in_q : in_v;
                        up.w_vmsg[0] = w_m0; up.w_vmsg[1] = w_m1; up.w_vmsg[2] = w_m2;
                        up.ctrl_in[0] = c64_0; up.ctrl_in[1] = c64_1; up.ctrl_in[2] = c64_2;
                        up.w_gain[0] = (const float*) L.loop_gain[0].d;
                        up.w_gain[1] = (const float*) L.loop_gain[1].d;
                        up.w_gain[2] = (const float*) L.loop_gain[2].d;
                        up.w_forget[0] = (const float*) L.loop_forget[0].d;
                        up.w_forget[1] = (const float*) L.loop_forget[1].d;
                        up.w_forget[2] = (const float*) L.loop_forget[2].d;
                        if (d_core_wup_f16) {
                            const size_t wup_m = (size_t) N * 64u;
                            up.wg_f16[0] = d_core_wup_f16 + (size_t) (l_idx * 4 + 0) * wup_m;
                            up.wg_f16[1] = d_core_wup_f16 + (size_t) (l_idx * 4 + 1) * wup_m;
                            up.wg_f16[2] = d_core_wup_f16 + (size_t) (l_idx * 4 + 2) * wup_m;
                            up.wf_f16[0] = d_core_wup_f16 + (size_t) (48 + l_idx * 4 + 0) * wup_m;
                            up.wf_f16[1] = d_core_wup_f16 + (size_t) (48 + l_idx * 4 + 1) * wup_m;
                            up.wf_f16[2] = d_core_wup_f16 + (size_t) (48 + l_idx * 4 + 2) * wup_m;
                        }
                        up.out[0] = in_q; up.out[1] = in_k; up.out[2] = in_v;
                        if (use_fuse_ctrl_rms_3) {
                            up.out[0] = nullptr; up.out[1] = nullptr; up.out[2] = nullptr;
                            CtrlUpRmsOuts ro{};
                            ro.rms_w[0] = nw0; ro.rms_w[1] = nw1; ro.rms_w[2] = nw2;
                            ro.out_f16[0] = act_bf16;
                            ro.out_f16[1] = act_bf16 + (size_t) T * N;
                            ro.out_f16[2] = f16_v_buf;
                            ro.out_lo_f16 = nullptr;
                            ro.eps = hp.rms_eps;
                            k_control_up_combine_rms_f16_vmsg<3><<<dim3((unsigned) ((T + 15) / 16), 1, 1), 576, 29696, st>>>(
                                vmsg_bank0, vmsg_bank1, up, ro, T, N);
                        } else {
                            k_control_up_combine_pp_wmma_vmsg<3><<<dim3((unsigned) ((N + 63) / 64), (unsigned) ((T + 63) / 64)), 512, 0, st>>>(
                                vmsg_bank0, vmsg_bank1, up, T, N);
                            k_rms_norm_3_f16<<<dim3(3, (unsigned) T), rms_f16_blk, 0, st>>>(
                                in_q, in_k, in_v, nullptr, nullptr, nullptr, nw0, nw1, nw2,
                                act_bf16, act_bf16 + (size_t) T * N, f16_v_buf, N, hp.rms_eps);
                        }
                        f16_v_pre_done = true;
                    }
                } else {
                if (deferred_postffn.active) {
                    flush_deferred_postffn_to_cur();
                    emit_dspark_capture();
                }
                const bool use_fused_ctrl_down = (has_msg && has_msg_ir && T >= 32 && N == 2304 &&
                                                  hp.loop_gate_rank == 64 && can_ctrl_wmma && can_ctrl_down_wmma &&
                                                  L.loop_down[0].type == 0 && L.loop_down[1].type == 0 && L.loop_down[2].type == 0 &&
                                                  L.loop_gain[0].type == 0 && L.loop_gain[1].type == 0 && L.loop_gain[2].type == 0 &&
                                                  L.loop_forget[0].type == 0 && L.loop_forget[1].type == 0 && L.loop_forget[2].type == 0);
                if (use_fused_ctrl_down) {
                    MixInvRms4 ir_c4 = is_swa ? MixInvRms4{{ir_c0, nullptr, nullptr, nullptr}}
                                              : MixInvRms4{{ir_c0, ir_c1, ir_c2, nullptr}};
                    // For SWA: swa_bcast = 0 (writes ONLY in_q, saving 72 MB/layer!), then run_control_3_swa reads in_q ONCE!
                    run_mix_normed(norm_blocks, cur_in, false, L.attn_res_query, outs, is_swa ? 1 : 3, T, 0, &ir_c4);
                    if (is_swa) {
                        run_control_3_swa(in_q, ir_c0, msg0, msg1, msg2, ir_m0, ir_m1, ir_m2, L, in_q, in_k, in_v, T);
                    } else {
                        run_control(in_q, msg0, L, 0, in_q, T, ir_c0, ir_m0);
                        run_control(in_k, msg1, L, 1, in_k, T, ir_c1, ir_m1);
                        run_control(in_v, msg2, L, 2, in_v, T, ir_c2, ir_m2);
                    }
                } else {
                run_mix_normed(norm_blocks, cur_in, false, L.attn_res_query, outs, is_swa ? 1 : 3, T, is_swa ? 1 : 0);
                if (has_msg) {
                    const bool can_ctrl3 = (T >= 1 && T <= 8 && (N & 3) == 0 && hp.loop_gate_rank == 64 &&
                                            L.loop_down[0].type == 0 && L.loop_down[1].type == 0 && L.loop_down[2].type == 0 &&
                                            L.loop_gain[0].type == 0 && L.loop_gain[1].type == 0 && L.loop_gain[2].type == 0 &&
                                            L.loop_forget[0].type == 0 && L.loop_forget[1].type == 0 && L.loop_forget[2].type == 0);
                    if (can_ctrl3) {
                        CtrlPtrs3 cp{(const float*) L.loop_down[0].d, (const float*) L.loop_down[1].d, (const float*) L.loop_down[2].d,
                                     in_q, in_k, in_v, msg0, msg1, msg2};
                        CtrlUpPtrs3 up{(const float*) L.loop_gain[0].d, (const float*) L.loop_gain[1].d, (const float*) L.loop_gain[2].d,
                                       (const float*) L.loop_forget[0].d, (const float*) L.loop_forget[1].d, (const float*) L.loop_forget[2].d,
                                       in_q, in_k, in_v, msg0, msg1, msg2, in_q, in_k, in_v};
                        const unsigned nb = (unsigned) ((N + 15) / 16);
                        // Reuse ctrl2N (size T*2*N floats >> 3*T*64 floats) for batched 3-module ctrl64 (0 extra VRAM).
                        if (T == 1) {
                            k_loop_down_fused_smallT_3<1, true><<<dim3(64, 3), 256, 0, st>>>(cp, ctrl2N, N, 64, 1, hp.rms_eps);
                            k_control_up_combine_smallT_3<1, true><<<dim3(nb, 3), 256, 0, st>>>(up, ctrl2N, 1, N);
                        } else if (T == 4) {
                            k_loop_down_fused_smallT_3<4, true><<<dim3(64, 3), 256, 0, st>>>(cp, ctrl2N, N, 64, 4, hp.rms_eps);
                            k_control_up_combine_smallT_3<4, true><<<dim3(nb, 3), 256, 0, st>>>(up, ctrl2N, 4, N);
                        } else if (T == 2) {
                            k_loop_down_fused_smallT_3<2, true><<<dim3(64, 3), 256, 0, st>>>(cp, ctrl2N, N, 64, 2, hp.rms_eps);
                            k_control_up_combine_smallT_3<2, true><<<dim3(nb, 3), 256, 0, st>>>(up, ctrl2N, 2, N);
                        } else if (T == 3) {
                            k_loop_down_fused_smallT_3<3, true><<<dim3(64, 3), 256, 0, st>>>(cp, ctrl2N, N, 64, 3, hp.rms_eps);
                            k_control_up_combine_smallT_3<3, true><<<dim3(nb, 3), 256, 0, st>>>(up, ctrl2N, 3, N);
                        } else {
                            k_loop_down_fused_smallT_3<8, false><<<dim3(64, 3), 256, 0, st>>>(cp, ctrl2N, N, 64, T, hp.rms_eps);
                            k_control_up_combine_smallT_3<8, false><<<dim3(nb, 3), 256, 0, st>>>(up, ctrl2N, T, N);
                        }
                    } else {
                        run_control(in_q, msg0, L, 0, in_q, T);
                        run_control(in_k, msg1, L, 1, in_k, T);
                        run_control(in_v, msg2, L, 2, in_v, T);
                    }
                }
                }
                if (is_swa && has_msg) {
                    if (fuse_q8_norm) {
                        k_rms_norm_3_q8_1<<<dim3(3, (unsigned) T), rms_q8_blk, 0, st>>>(
                            in_q, in_k, in_v, in_q, in_k, in_v, L.attn_norm, L.attn_norm, L.attn_norm,
                            xq, xq_k, xq_v, N, hp.rms_eps);
                        q8_pre_done = true;
                    } else if (T > 8) {
                        k_rms_norm_3_f16<<<dim3(3, (unsigned) T), rms_f16_blk, 0, st>>>(
                            in_q, in_k, in_v, outs_opt[0], outs_opt[1], outs_opt[2], L.attn_norm, L.attn_norm, L.attn_norm,
                            act_bf16, act_bf16 + (size_t) T * N, f16_v_buf, N, hp.rms_eps);
                        f16_v_pre_done = (f16_v_buf != nullptr);
                    } else {
                        k_rms_norm_3<<<dim3(3, (unsigned) T), kThreads, 0, st>>>(in_q, in_k, in_v, L.attn_norm, N, hp.rms_eps);
                    }
                } else if (is_swa) {
                    if (fuse_q8_norm) {
                        k_rms_norm_q8_1<<<(unsigned) T, rms_q8_blk, 0, st>>>(in_q, L.attn_norm, in_q, xq, N, hp.rms_eps);
                        q8_pre_done = true;
                    } else if (T > 8) {
                        k_rms_norm_f16<<<(unsigned) T, rms_f16_blk, 0, st>>>(in_q, L.attn_norm, in_q, act_bf16, N, hp.rms_eps);
                    } else {
                        k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(in_q, L.attn_norm, in_q, N, hp.rms_eps);
                    }
                    ink_src = in_q;
                    inv_src = in_q;
                } else {
                    if (fuse_q8_norm) {
                        k_rms_norm_3_q8_1<<<dim3(3, (unsigned) T), rms_q8_blk, 0, st>>>(
                            in_q, in_k, in_v, in_q, in_k, in_v, L.attn_q_inp_norm, L.attn_k_inp_norm, L.attn_v_inp_norm,
                            xq, xq_k, xq_v, N, hp.rms_eps);
                        q8_pre_done = true;
                    } else if (T > 8) {
                        k_rms_norm_3_f16<<<dim3(3, (unsigned) T), rms_f16_blk, 0, st>>>(
                            in_q, in_k, in_v, outs_opt[0], outs_opt[1], outs_opt[2], L.attn_q_inp_norm, L.attn_k_inp_norm, L.attn_v_inp_norm,
                            act_bf16, act_bf16 + (size_t) T * N, f16_v_buf, N, hp.rms_eps);
                        f16_v_pre_done = (f16_v_buf != nullptr);
                    } else {
                        k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(in_q, L.attn_q_inp_norm, in_q, N, hp.rms_eps);
                        k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(in_k, L.attn_k_inp_norm, in_k, N, hp.rms_eps);
                        k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(in_v, L.attn_v_inp_norm, in_v, N, hp.rms_eps);
                    }
                }
                }
                }
            } else {
                if (deferred_postffn.active) {
                    flush_deferred_postffn_to_cur();
                    emit_dspark_capture();
                }
                if (has_msg) {
                    run_control(cur_in, msg0, L, 0, in_q, T);
                    run_control(cur_in, msg1, L, 1, in_k, T);
                    run_control(cur_in, msg2, L, 2, in_v, T);
                    if (is_swa) {
                        if (fuse_q8_norm) {
                            k_rms_norm_3_q8_1<<<dim3(3, (unsigned) T), rms_q8_blk, 0, st>>>(
                                in_q, in_k, in_v, in_q, in_k, in_v, L.attn_norm, L.attn_norm, L.attn_norm,
                                xq, xq_k, xq_v, N, hp.rms_eps);
                            q8_pre_done = true;
                        } else if (T > 8) {
                            k_rms_norm_3_f16<<<dim3(3, (unsigned) T), rms_f16_blk, 0, st>>>(
                                in_q, in_k, in_v, in_q, in_k, in_v, L.attn_norm, L.attn_norm, L.attn_norm,
                                act_bf16, act_bf16 + (size_t) T * N, nullptr, N, hp.rms_eps);
                        } else {
                            k_rms_norm_3<<<dim3(3, (unsigned) T), kThreads, 0, st>>>(in_q, in_k, in_v, L.attn_norm, N, hp.rms_eps);
                        }
                    } else {
                        if (fuse_q8_norm) {
                            k_rms_norm_3_q8_1<<<dim3(3, (unsigned) T), rms_q8_blk, 0, st>>>(
                                in_q, in_k, in_v, in_q, in_k, in_v, L.attn_q_inp_norm, L.attn_k_inp_norm, L.attn_v_inp_norm,
                                xq, xq_k, xq_v, N, hp.rms_eps);
                            q8_pre_done = true;
                        } else if (T > 8) {
                            k_rms_norm_3_f16<<<dim3(3, (unsigned) T), rms_f16_blk, 0, st>>>(
                                in_q, in_k, in_v, in_q, in_k, in_v, L.attn_q_inp_norm, L.attn_k_inp_norm, L.attn_v_inp_norm,
                                act_bf16, act_bf16 + (size_t) T * N, nullptr, N, hp.rms_eps);
                        } else {
                            k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(in_q, L.attn_q_inp_norm, in_q, N, hp.rms_eps);
                            k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(in_k, L.attn_k_inp_norm, in_k, N, hp.rms_eps);
                            k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(in_v, L.attn_v_inp_norm, in_v, N, hp.rms_eps);
                        }
                    }
                } else if (is_swa) {
                    if (fuse_q8_norm) {
                        k_rms_norm_q8_1<<<(unsigned) T, rms_q8_blk, 0, st>>>(cur_in, L.attn_norm, in_q, xq, N, hp.rms_eps);
                        q8_pre_done = true;
                    } else if (T > 8) {
                        k_rms_norm_f16<<<(unsigned) T, rms_f16_blk, 0, st>>>(cur_in, L.attn_norm, in_q, act_bf16, N, hp.rms_eps);
                    } else {
                        k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(cur_in, L.attn_norm, in_q, N, hp.rms_eps);
                    }
                    ink_src = in_q;
                    inv_src = in_q;
                } else {
                    if (fuse_q8_norm) {
                        k_rms_norm_3_q8_1<<<dim3(3, (unsigned) T), rms_q8_blk, 0, st>>>(
                            cur_in, cur_in, cur_in, in_q, in_k, in_v, L.attn_q_inp_norm, L.attn_k_inp_norm, L.attn_v_inp_norm,
                            xq, xq_k, xq_v, N, hp.rms_eps);
                        q8_pre_done = true;
                    } else if (T > 8) {
                        k_rms_norm_3_f16<<<dim3(3, (unsigned) T), rms_f16_blk, 0, st>>>(
                            cur_in, cur_in, cur_in, in_q, in_k, in_v, L.attn_q_inp_norm, L.attn_k_inp_norm, L.attn_v_inp_norm,
                            act_bf16, act_bf16 + (size_t) T * N, nullptr, N, hp.rms_eps);
                    } else {
                        k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(cur_in, L.attn_q_inp_norm, in_q, N, hp.rms_eps);
                        k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(cur_in, L.attn_k_inp_norm, in_k, N, hp.rms_eps);
                        k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(cur_in, L.attn_v_inp_norm, in_v, N, hp.rms_eps);
                    }
                }
            }
            if (prof_detail) { CK(cudaStreamSynchronize(st)); p_d_mix += ms_since(tp_d0); }

            // Attention projections (quantize/convert unique input vectors ONCE per layer)
            const auto tp_d1 = Clock::now();
            const void* xq_k_src = (ink_src == in_q) ? xq : xq_k;
            const void* xq_v_src = (inv_src == in_q) ? xq : (inv_src == ink_src ? xq_k_src : xq_v);
            if (!q8_pre_done) {
                prep_q8(in_q, xq, T, N);
                if (ink_src != in_q) prep_q8(ink_src, xq_k, T, N);
                if (inv_src != in_q && inv_src != ink_src) prep_q8(inv_src, xq_v, T, N);
            }
            const uint16_t* f16_q = (T > 8) ? act_bf16 : nullptr;
            const uint16_t* f16_k_pre = (T > 8) ? ((ink_src == in_q) ? act_bf16 : (act_bf16 + (size_t) T * N)) : nullptr;
            // C4: On sm_80+ (!bf16_x_f16), convert in_q -> BF16 ONCE ONLY if large projections (q/gate/rel) are BF16;
            // narrow BF16 attn_sink_proj (N_out<=64) runs directly on f16_q via k_gemm_f16x_bf16w_smallN!
            const uint16_t* bf16_q = nullptr;
            if (!bf16_x_f16 && T > 8 && act_bf16_only &&
                (L.attn_q.type == 30 || L.attn_gate.type == 30 || L.attn_rel.type == 30)) {
                launch_f32_to_bf16(in_q, act_bf16_only, (int64_t) T * N);
                bf16_q = act_bf16_only;
            }
            const bool fused_pre_attn_t1 = (T == 1 && kv_q8 && hp.head_dim == 128 && hp.n_rel == 16 && !save_conv_trace);
            const bool fuse_sink_pre_attn = (fused_pre_attn_t1 && L.attn_sink_proj.type == 30 && (N & 1) == 0 && (N == 2304));
            // EX-5 Wave 2: Opt-in only (STRATA_DEQUANT_SIDE=1, default OFF) — Wave 1 nsys proved overlapped
            // dequant_q8_0_kernel contends for SM/L2 with cuBLAS GEMM (+4.1 ms).
            static const bool enable_dequant_side = [] {
                const char* v = std::getenv("STRATA_DEQUANT_SIDE");
                return v != nullptr && std::atoi(v) != 0;
            }();
            const bool use_dequant_side = (enable_dequant_side && hybrid && T >= 256 && dequant_scratch2 != nullptr);
            if (use_dequant_side) {
                std::string side_err;
                gemm.set_dequant_side(dequant_scratch2, side_err);
            }
            const int64_t rel_dim = hp.n_rel * hp.n_head;
            const int64_t dyn_dim = hp.n_conv * hp.n_head_kv;
            const bool fused_kv_q8_pp = (T >= 32 && kv_q8 && hp.head_dim == 128 && !save_conv_trace);
            // Hoist FA2-192-SWA-fast & want_att_f16 dispatch flags BEFORE sink/QKV so fused QKV / Wave 10 split
            // [Q,R,(sink)] FP32 + [gate,K,V] FP16 ONLY activate when k_attn_q8_gqa3_pp_fa2_192_swa_fast actually runs
            // (best_sp == 1 is guaranteed because fuse_swa_rel_pp requires is_swa and fuse_full_rel_pp requires pos < 8192)
            // and writes act_bf16 (leaving reg1's att slice untouched!).
            const bool fuse_swa_rel_pp = kv_q8 && is_swa && can_fa2_192 && T >= 128 &&
                                         hp.n_rel == 16 && hp.n_head == 3 * hp.n_head_kv && hp.head_dim == 128;
            const bool fuse_full_rel_pp = kv_q8 && !is_swa && can_fa2_192 && can_fa2_192_swa_fast && T >= 128 && pos < 8192 &&
                                          hp.n_rel == 16 && hp.n_head == 3 * hp.n_head_kv && hp.head_dim == 128;
            const bool use_fa2_192_swa_fast = can_fa2_192_swa_fast && (fuse_swa_rel_pp || fuse_full_rel_pp);
            const bool use_fa2_256_path = (kv_q8 && hp.n_head == 3 * hp.n_head_kv && hp.head_dim == 128 &&
                                           can_fa2_256 && !is_swa && T >= 256 && pos >= 8192 &&
                                           pp_sp_buf && pp_sp_bytes >= (size_t) T * (size_t) hp.n_head * 130 * sizeof(float) * 2);
            const bool sub_tile_fa2_256 = (use_fa2_256_path && T > rel_tab_cap_t && rel_tab_cap_t >= 256 && hp.n_rel == 16);
            const bool want_att_f16 = (T >= 128 && act_bf16 != nullptr && q_dim <= 2 * N &&
                                       L.attn_output.type != 0 && (L.attn_output.type != 30 || bf16_x_f16));
            static const bool no_fused_qgr = (std::getenv("STRATA_NO_FUSED_QGR") != nullptr);
            static const bool no_gkv_f16 = [] {
                const char* v = std::getenv("STRATA_NO_GKV_F16");
                return v != nullptr && std::atoi(v) != 0;
            }();
            static const bool no_fuse_sink = [] {
                const char* v = std::getenv("STRATA_NO_FUSE_SINK");
                return v != nullptr && std::atoi(v) != 0;
            }();
            const size_t reg0_tmp_off = (size_t) (reinterpret_cast<uint8_t*>(K) - layer_arena);
            const size_t reg0_tmp_bytes = (pp_sp_bytes > reg0_tmp_off) ? (pp_sp_bytes - reg0_tmp_off) : 0;
            const size_t reg0_tmp_elems = reg0_tmp_bytes / sizeof(uint16_t);
            const size_t reg1_cap_floats = reg1_bytes_cur / sizeof(float);
            const bool can_fused_qgr_base = (!no_fused_qgr && !use_dequant_side && !save_conv_trace &&
                                             T >= 128 && cc_maj >= 8 && kv_q8 && want_att_f16 &&
                                             use_fa2_192_swa_fast && f16_q != nullptr &&
                                             L.attn_qgr.d != nullptr && L.attn_qgr.type == 8);
            static const bool qr_pad128 = [] {
                const char* v = std::getenv("STRATA_QR_PAD128");
                return v != nullptr && std::atoi(v) != 0;
            }();
            const int64_t qr_rows_base = q_dim + rel_dim;
            const int64_t qr_rows_raw = qr_rows_base + hp.n_head;
            const int64_t qr_rows_fused = qr_pad128
                ? (((qr_rows_raw + 127) / 128) * 128)
                : (qr_rows_base + ((hp.n_head + 31) / 32) * 32);
            const int64_t n_sink_pad = qr_rows_fused - qr_rows_raw;
            const size_t gkv_f16_as_floats = ((size_t) T * (size_t) (q_dim + 2 * kv_dim) + 1u) / 2u;
            const bool use_split_qr_gkv_f16 = (!no_gkv_f16 && T >= 4096 && can_fused_qgr_base && fused_kv_q8_pp &&
                                               L.attn_qr.d != nullptr && L.attn_qr.type == 8 &&
                                               L.attn_qr.n_out == qr_rows_base &&
                                               L.attn_gate.type == 8 && L.attn_k.type == 8 && L.attn_v.type == 8 &&
                                               ((size_t) T * (size_t) qr_rows_base + gkv_f16_as_floats <= reg1_cap_floats) &&
                                               (std::max(reg0_tmp_elems, (size_t) gemm.scratch_elems()) >= (size_t) qr_rows_base * (size_t) N));
            const bool fuse_sink_in_qr = (use_split_qr_gkv_f16 && !no_fuse_sink && !fuse_sink_pre_attn &&
                                          L.attn_sink_proj.d != nullptr && L.attn_sink_proj.type == 30 &&
                                          L.attn_sink_proj.n_out == hp.n_head && hp.n_head <= 32 &&
                                          (N % 4) == 0 &&
                                          reg0_tmp_elems >= (size_t) qr_rows_fused * (size_t) N &&
                                          ((size_t) T * (size_t) qr_rows_fused + gkv_f16_as_floats <= reg1_cap_floats));
            const bool use_fused_qgrkv = (!use_split_qr_gkv_f16 && can_fused_qgr_base && is_swa && !has_msg &&
                                          ink_src == in_q && inv_src == in_q &&
                                          L.attn_qgrkv.d != nullptr && L.attn_qgrkv.type == 8 &&
                                          L.attn_qgrkv.n_out == (2 * q_dim + rel_dim + 2 * kv_dim) &&
                                          (size_t) T * (size_t) L.attn_qgrkv.n_out <= reg1_cap_floats &&
                                          reg0_tmp_elems >= (size_t) L.attn_qgrkv.n_out * (size_t) N);
            const bool use_fused_qgr = (!use_split_qr_gkv_f16 && !use_fused_qgrkv && can_fused_qgr_base &&
                                        L.attn_qgr.n_out == (2 * q_dim + rel_dim) &&
                                        (size_t) T * (size_t) L.attn_qgr.n_out <= reg1_cap_floats &&
                                        reg0_tmp_elems >= (size_t) L.attn_qgr.n_out * (size_t) N);
            if (!fuse_sink_pre_attn && !fuse_sink_in_qr) {
                project_preq(in_q, xq, L.attn_sink_proj, sink, T, hp.n_head, N, f16_q, bf16_q);
            }
            float* pQ = Q;
            float* pGate = att_gate;
            float* pR = R;
            float* pK = K;
            float* pV = V;
            float* pSink = sink;
            float* pDynK = dyn_k;
            float* pDynV = dyn_v;
            const __half* pGate_f16 = nullptr;
            const __half* pK_f16 = nullptr;
            const __half* pV_f16 = nullptr;
            int ld_q_attn = 0, ld_r_attn = 0, ld_gate_attn = 0, ld_sink_attn = 0, ld_kv_sconv = 0;
            if (use_split_qr_gkv_f16) {
                // Wave 8B-2: [Q, R, (sink)] in strict FP32 (CUBLAS_COMPUTE_32F), [gate, K, V] in FP16 out (strict CUBLAS_COMPUTE_32F).
                float* reg1_base = Q;
                const int64_t qr_rows = fuse_sink_in_qr ? qr_rows_fused : qr_rows_base;
                if (fuse_sink_in_qr) {
                    uint16_t* w_tmp = reinterpret_cast<uint16_t*>(K);
                    gemm.dequant_to_f16(L.attn_qr.type, L.attn_qr.d, w_tmp, L.attn_qr.n_out, N);
                    const int64_t sink_pad_n4 = (hp.n_head + n_sink_pad) * N / 4;
                    k_bf16_to_f16_pad<<<(unsigned) ((sink_pad_n4 + 255) / 256), 256, 0, st>>>(
                        (const uint16_t*) L.attn_sink_proj.d,
                        w_tmp + (size_t) L.attn_qr.n_out * (size_t) N,
                        (int) hp.n_head, (int) n_sink_pad, (int) N);
                    gemm.f16(f16_q, w_tmp, reg1_base, T, qr_rows, N);
                    pQ = reg1_base;
                    pR = reg1_base + q_dim;
                    pSink = reg1_base + q_dim + rel_dim;
                    ld_q_attn = ld_r_attn = ld_sink_attn = (int) qr_rows;
                } else {
                    const bool use_k_tmp_qr = (reg0_tmp_elems >= (size_t) L.attn_qr.n_out * (size_t) N);
                    if (use_k_tmp_qr) gemm.set_temp_scratch(K, reg0_tmp_elems);
                    project_preq(in_q, xq, L.attn_qr, reg1_base, T, L.attn_qr.n_out, N, f16_q, bf16_q);
                    if (use_k_tmp_qr) gemm.set_temp_scratch(nullptr, 0);
                    pQ = reg1_base;
                    pR = reg1_base + q_dim;
                    pSink = sink;
                    ld_q_attn = ld_r_attn = (int) qr_rows;
                    ld_sink_attn = 0;
                }
                uint16_t* gkv_base = reinterpret_cast<uint16_t*>(reg1_base + (size_t) T * (size_t) qr_rows);
                const bool use_fused_gkv_f16 = (is_swa && !has_msg &&
                                                ink_src == in_q && inv_src == in_q &&
                                                L.attn_gkv.d != nullptr && L.attn_gkv.type == 8 &&
                                                L.attn_gkv.n_out == (q_dim + 2 * kv_dim) &&
                                                reg0_tmp_elems >= (size_t) L.attn_gkv.n_out * (size_t) N);
                if (use_fused_gkv_f16) {
                    gemm.set_temp_scratch(K, reg0_tmp_elems);
                    gemm.native_f16out(f16_q, L.attn_gkv.type, L.attn_gkv.d, gkv_base, T, L.attn_gkv.n_out, N);
                    gemm.set_temp_scratch(nullptr, 0);
                    pGate_f16 = reinterpret_cast<const __half*>(gkv_base);
                    pK_f16 = reinterpret_cast<const __half*>(gkv_base + q_dim);
                    pV_f16 = reinterpret_cast<const __half*>(gkv_base + q_dim + kv_dim);
                    ld_gate_attn = ld_kv_sconv = (int) L.attn_gkv.n_out;
                } else {
                    const uint16_t* f16_v = (inv_src == in_q) ? f16_q : (f16_v_pre_done ? f16_v_buf : nullptr);
                    if (f16_v == nullptr && pre_attn_f16_v != nullptr) {
                        launch_f32_to_f16(inv_src, pre_attn_f16_v, (int64_t) T * N);
                        f16_v = pre_attn_f16_v;
                    }
                    const bool use_k_tmp_gkv = (reg0_tmp_elems >= (size_t) std::max(q_dim, kv_dim) * (size_t) N);
                    if (use_k_tmp_gkv) gemm.set_temp_scratch(K, reg0_tmp_elems);
                    gemm.native_f16out(f16_q, L.attn_gate.type, L.attn_gate.d, gkv_base, T, q_dim, N);
                    pGate_f16 = reinterpret_cast<const __half*>(gkv_base);
                    ld_gate_attn = (int) q_dim;
                    if (f16_k_pre != nullptr && f16_v != nullptr) {
                        uint16_t* k16 = gkv_base + (size_t) T * (size_t) q_dim;
                        uint16_t* v16 = k16 + (size_t) T * (size_t) kv_dim;
                        gemm.native_f16out(f16_k_pre, L.attn_k.type, L.attn_k.d, k16, T, kv_dim, N);
                        gemm.native_f16out(f16_v, L.attn_v.type, L.attn_v.d, v16, T, kv_dim, N);
                        if (use_k_tmp_gkv) gemm.set_temp_scratch(nullptr, 0);
                        pK_f16 = reinterpret_cast<const __half*>(k16);
                        pV_f16 = reinterpret_cast<const __half*>(v16);
                        ld_kv_sconv = 0;
                    } else {
                        if (use_k_tmp_gkv) gemm.set_temp_scratch(nullptr, 0);
                        project_preq(ink_src, xq_k_src, L.attn_k, K, T, kv_dim, N, f16_k_pre);
                        project_preq(inv_src, xq_v_src, L.attn_v, V, T, kv_dim, N, f16_v);
                    }
                    if (!is_swa) {
                        project_preq(ink_src, xq_k_src, L.attn_k_conv_proj, dyn_k, T, dyn_dim, N, f16_k_pre);
                        const uint16_t* f16_v_reuse = (f16_v != nullptr) ? f16_v
                                                    : ((inv_src == in_q || L.attn_v.type != 0) ? act_bf16 : nullptr);
                        project_preq(inv_src, xq_v_src, L.attn_v_conv_proj, dyn_v, T, dyn_dim, N, f16_v_reuse);
                    }
                }
            } else if (use_fused_qgrkv) {
                float* reg1_base = Q;
                gemm.set_temp_scratch(K, reg0_tmp_elems);
                project_preq(in_q, xq, L.attn_qgrkv, reg1_base, T, L.attn_qgrkv.n_out, N, f16_q, bf16_q);
                gemm.set_temp_scratch(nullptr, 0);
                pQ = reg1_base;
                pR = reg1_base + q_dim;
                pGate = reg1_base + q_dim + rel_dim;
                pK = reg1_base + 2 * q_dim + rel_dim;
                pV = reg1_base + 2 * q_dim + rel_dim + kv_dim;
                ld_q_attn = ld_gate_attn = ld_r_attn = ld_kv_sconv = (int) L.attn_qgrkv.n_out;
            } else if (T == 1 && !has_msg && is_swa && L.attn_qgrkv.d != nullptr) {
                pQ = qgr_t1;
                pR = qgr_t1 + q_dim;
                pGate = qgr_t1 + q_dim + rel_dim;
                pK = qgr_t1 + 2 * q_dim + rel_dim;
                pV = pK + kv_dim;
                project_preq(in_q, xq, L.attn_qgrkv, qgr_t1, 1, L.attn_qgrkv.n_out, N);
            } else if (T >= 1 && T <= 8 &&
                       try_project_multi_q8_0(xq, xq, xq, xq_k_src, xq_v_src, 5,
                                              L.attn_q, Q, q_dim, L.attn_gate, att_gate, q_dim,
                                              L.attn_rel, R, rel_dim, &L.attn_k, K, kv_dim, &L.attn_v, V, kv_dim, T, N)) {
                // Fused 5-way Q+Gate+R+K+V in 1 launch (per-matrix xq_k/xq_v covers all 48 layers)!
                if (!is_swa) {
                    project_preq(ink_src, xq_k_src, L.attn_k_conv_proj, dyn_k, T, dyn_dim, N, nullptr);
                    project_preq(inv_src, xq_v_src, L.attn_v_conv_proj, dyn_v, T, dyn_dim, N, nullptr);
                }
            } else {
                if (use_fused_qgr) {
                    float* reg1_base = Q;
                    gemm.set_temp_scratch(K, reg0_tmp_elems);
                    project_preq(in_q, xq, L.attn_qgr, reg1_base, T, L.attn_qgr.n_out, N, f16_q, bf16_q);
                    gemm.set_temp_scratch(nullptr, 0);
                    pQ = reg1_base;
                    pR = reg1_base + q_dim;
                    pGate = reg1_base + q_dim + rel_dim;
                    ld_q_attn = ld_gate_attn = ld_r_attn = (int) L.attn_qgr.n_out;
                } else if (T == 1 && L.attn_qgr.d != nullptr) {
                    pQ = qgr_t1;
                    pR = qgr_t1 + q_dim;
                    pGate = qgr_t1 + q_dim + rel_dim;
                    project_preq(in_q, xq, L.attn_qgr, qgr_t1, 1, L.attn_qgr.n_out, N);
                } else {
                    if (!(T >= 1 && T <= 8 &&
                          try_project_multi_q8_0(xq, xq, xq, nullptr, nullptr, 3,
                                                 L.attn_q, Q, q_dim, L.attn_gate, att_gate, q_dim,
                                                 L.attn_rel, R, rel_dim, nullptr, nullptr, 0, nullptr, nullptr, 0, T, N))) {
                        if (!try_project_pair_bf16(in_q, L.attn_q, Q, q_dim, L.attn_gate, att_gate, q_dim, T, N) &&
                            !try_project_pair_q8_0(xq, L.attn_q, Q, q_dim, L.attn_gate, att_gate, q_dim, T, N)) {
                            project_preq(in_q, xq, L.attn_q, Q, T, q_dim, N, f16_q, bf16_q);
                            project_preq(in_q, xq, L.attn_gate, att_gate, T, q_dim, N, f16_q, bf16_q);
                        }
                        project_preq(in_q, xq, L.attn_rel, R, T, rel_dim, N, f16_q, bf16_q);
                    }
                }
                const bool paired_kv = (T >= 2 && T <= 8 && inv_src == ink_src &&
                                        (try_project_pair_bf16(ink_src, L.attn_k, K, kv_dim, L.attn_v, V, kv_dim, T, N) ||
                                         try_project_pair_q8_0(xq_k_src, L.attn_k, K, kv_dim, L.attn_v, V, kv_dim, T, N)));
                if (!paired_kv) {
                    if (!is_swa && T == 1 && L.attn_kkc.d != nullptr) {
                        float* kbuf = qgr_t1 + 2 * q_dim + rel_dim;
                        pK = kbuf;
                        pDynK = kbuf + kv_dim;
                        project_preq(ink_src, xq_k_src, L.attn_kkc, kbuf, 1, L.attn_kkc.n_out, N);
                    } else {
                        project_preq(ink_src, xq_k_src, L.attn_k, K, T, kv_dim, N, f16_k_pre);
                    }
                    if (!is_swa && T == 1 && L.attn_vvc.d != nullptr) {
                        float* vbuf = qgr_t1 + 2 * q_dim + rel_dim + kv_dim + dyn_dim;
                        pV = vbuf;
                        pDynV = vbuf + kv_dim;
                        project_preq(inv_src, xq_v_src, L.attn_vvc, vbuf, 1, L.attn_vvc.n_out, N);
                    } else {
                        const uint16_t* f16_v = (T > 8 && inv_src == in_q) ? f16_q : (f16_v_pre_done ? f16_v_buf : nullptr);
                        project_preq(inv_src, xq_v_src, L.attn_v, V, T, kv_dim, N, f16_v);
                    }
                }
                // Dynamic conv projections (BF16 on !is_swa, N_out=32): reuse f16_k_pre & f16_v directly via gemm.bf16(x_is_f16=true)!
                if (!is_swa && !(T == 1 && L.attn_kkc.d != nullptr)) {
                    project_preq(ink_src, xq_k_src, L.attn_k_conv_proj, dyn_k, T, dyn_dim, N, f16_k_pre);
                }
                if (!is_swa && !(T == 1 && L.attn_vvc.d != nullptr)) {
                    const uint16_t* f16_v_reuse = f16_v_pre_done ? f16_v_buf
                                                : ((T > 8 && (inv_src == in_q || L.attn_v.type != 0)) ? act_bf16 : nullptr);
                    project_preq(inv_src, xq_v_src, L.attn_v_conv_proj, dyn_v, T, dyn_dim, N, f16_v_reuse);
                }
            }

            // Re-arm dequant_side right after QKV (resets side_cur_=0 & records side_free_[0] on st) so attn_output
            // dequants into dedicated gemm.scratch_ overlapped with Attention without touching layer_arena/rel_tab!
            // Wave 17: Opt-in only via STRATA_WO_DEQUANT_SIDE=1 (default OFF: +127 ms regression on 8K prefill).
            static const bool opt_wo_dequant_side = [] {
                const char* v = std::getenv("STRATA_WO_DEQUANT_SIDE");
                return v != nullptr && (v[0] == '\0' || std::atoi(v) != 0);
            }();
            const bool use_wo_dequant_side = (opt_wo_dequant_side && !use_dequant_side &&
                                              hybrid && T >= 256 && !capturing_full_graph &&
                                              has_quant_gemm_scratch &&
                                              L.attn_output.type != 0 && L.attn_output.type != 1 && L.attn_output.type != 30 &&
                                              (size_t) N * (size_t) q_dim <= (size_t) gemm.scratch_elems() &&
                                              (dequant_scratch2 != nullptr || act_bf16 != nullptr));
            if (use_dequant_side || use_wo_dequant_side) {
                std::string side_err;
                uint16_t* side_tok = dequant_scratch2 ? dequant_scratch2 : act_bf16;
                gemm.set_dequant_side(side_tok, side_err);
            }
            float* hst = conv_hist[(size_t) il];
            float* h_tr = save_conv_trace ? (d_conv_trace + (size_t) il * layer_conv_floats) : nullptr;
            if (prof_detail) { CK(cudaStreamSynchronize(st)); p_d_qkv += ms_since(tp_d1); }
            const auto tp_d2 = Clock::now();
            const int apply_tau = is_swa ? 0 : 1;
            const int max_rel = is_swa ? std::min<int>((int) hp.n_swa, (int) hp.n_rel_ext) : (int) hp.n_rel_ext;
            // Wave 9: Shared-Scratch SWA KV Pool — prefill chunks T > (ring_cap - n_swa) execute on single-layer
            // swa_pp_scratch (cap = n_swa + active_pp_cap) and export only the last min(pos+T, n_swa) tokens to ring!
            const int ring_cap = kv_cap[(size_t) il];
            const bool use_swa_pp = (is_swa && use_shared_swa_scratch && !save_conv_trace && verify_k_save == 0 &&
                                     T > (ring_cap - (int) hp.n_swa) &&
                                     swa_pp_cap >= (int) hp.n_swa + T &&
                                     (kv_q8 ? (swa_pp_kq != nullptr && swa_pp_vq != nullptr && swa_pp_ks != nullptr && swa_pp_vs != nullptr)
                                            : (swa_pp_kc != nullptr && swa_pp_vc != nullptr)));
            const bool use_swa_pp_hm = (use_swa_pp && kv_q8 && cc_maj >= 8 && is_swa && T >= 128 &&
                                        use_fa2_192_swa_fast && fused_kv_q8_pp &&
                                        pK_f16 != nullptr && pV_f16 != nullptr);
            const int swa_hm_j_hist = (use_swa_pp_hm && pos > 0)
                                      ? (int) (pos - std::min<int64_t>(pos, hp.n_swa)) : 0;
            int8_t* const cur_kq = use_swa_pp ? swa_pp_kq : kq[(size_t) il];
            int8_t* const cur_vq = use_swa_pp ? swa_pp_vq : vq[(size_t) il];
            __half* const cur_ks = use_swa_pp ? swa_pp_ks : ks[(size_t) il];
            __half* const cur_vs = use_swa_pp ? swa_pp_vs : vs[(size_t) il];
            __half* const cur_kc = use_swa_pp ? swa_pp_kc : kc[(size_t) il];
            __half* const cur_vc = use_swa_pp ? swa_pp_vc : vc[(size_t) il];
            const int cur_kv_cap = use_swa_pp ? swa_pp_cap : ring_cap;
            if (use_swa_pp && pos > 0) {
                const int hist = (int) std::min<int64_t>(pos, hp.n_swa);
                const int j_hist = (int) (pos - hist);
                if (use_swa_pp_hm) {
                    k_swa_kv_ring_to_hm_q8<<<(unsigned) hist, 256, 0, st>>>(
                        kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il],
                        swa_pp_kq, swa_pp_ks, swa_pp_vq, swa_pp_vs,
                        j_hist, hist, ring_cap, swa_pp_cap, (int) hp.n_head_kv);
                } else if (kv_q8) {
                    k_swa_kv_ring_copy_q8<<<(unsigned) hist, 256, 0, st>>>(
                        kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il],
                        swa_pp_kq, swa_pp_ks, swa_pp_vq, swa_pp_vs,
                        j_hist, hist, ring_cap, swa_pp_cap, (int) kv_dim, (int) hp.n_head_kv);
                } else {
                    k_swa_kv_ring_copy_f16<<<(unsigned) hist, 256, 0, st>>>(
                        kc[(size_t) il], vc[(size_t) il],
                        swa_pp_kc, swa_pp_vc,
                        j_hist, hist, ring_cap, swa_pp_cap, (int) kv_dim);
                }
            }
            // Wave 5 + Wave 12B + Wave 13B: Overlap streaming k_zero_f4_cs(moe_raw) on aux_st during Attention
            // (when moe_raw is strictly disjoint from Region 1 Q/Gate/att, active pR, pp_sp_buf, and wo) so zeroing
            // completes before W_o / shortconv / router with zero contention; fall back to post-W_o otherwise.
            static const bool no_async_moe_zero = [] {
                const char* v = std::getenv("STRATA_NO_ASYNC_MOE_ZERO");
                return v != nullptr && (v[0] == '\0' || std::atoi(v) != 0);
            }();
            static const bool no_zero_f4_cs = [] {
                const char* v = std::getenv("STRATA_NO_ZERO_F4_CS");
                return v != nullptr && (v[0] == '\0' || std::atoi(v) != 0);
            }();
            static const int moe_zero_ctas = [] {
                const char* v = std::getenv("STRATA_MOE_ZERO_CTAS");
                return (v != nullptr && v[0] != '\0') ? std::clamp(std::atoi(v), 1, 128) : 32;
            }();
            static const bool no_moe_atomic_pre = (std::getenv("STRATA_NO_MOE_ATOMIC") != nullptr);
            const uint8_t* moe_lo_u8 = reinterpret_cast<const uint8_t*>(moe_raw);
            const uint8_t* moe_hi_u8 = moe_lo_u8 + (size_t) T * (size_t) N * sizeof(float);
            const bool moe_raw_disjoint_ctrl = (ctrl_w16 != nullptr) &&
                (moe_hi_u8 <= reinterpret_cast<const uint8_t*>(ctrl_w16));
            const bool moe_raw_disjoint_wo =
                (moe_lo_u8 >= reinterpret_cast<const uint8_t*>(wo) + (size_t) T * (size_t) N * sizeof(float));
            const bool moe_raw_disjoint_reg1 = (Q != nullptr) &&
                (moe_hi_u8 <= reinterpret_cast<const uint8_t*>(Q));
            const bool attn_reads_pR = (kv_q8 && hp.n_head == 3 * hp.n_head_kv && hp.head_dim == 128 &&
                                        (sub_tile_fa2_256 ||
                                         (can_fa2_192 && T >= 128 && (use_fa2_192_swa_fast || (is_swa && fuse_swa_rel_pp)))));
            const uint8_t* pr_lo_u8 = reinterpret_cast<const uint8_t*>(pR);
            const size_t pr_span = (size_t) T * (size_t) (ld_r_attn > 0 ? ld_r_attn : rel_dim) * sizeof(float);
            const bool moe_raw_disjoint_pR = (!attn_reads_pR) ||
                (moe_hi_u8 <= pr_lo_u8) || (moe_lo_u8 >= pr_lo_u8 + pr_span);
            const bool attn_may_use_pp_sp = (kv_q8 && hp.n_head == 3 * hp.n_head_kv && hp.head_dim == 128 &&
                                             (use_fa2_256_path || (can_fa2_192 && T >= 128 && !is_swa && pos >= 8192)));
            const bool moe_raw_disjoint_attn = (moe_raw_disjoint_wo && moe_raw_disjoint_reg1 &&
                                                moe_raw_disjoint_pR && !attn_may_use_pp_sp);
            const bool fuse_moe_zero_pre = (!env_no_fuse_moe_zero() && !no_moe_atomic_pre && !capturing_full_graph &&
                                            !include_route_t1 && T >= 32 && cc_maj >= 8 && (N & 3) == 0 &&
                                            L.gu_type != 30 && L.d_type != 30 && K::native_expert_wmma_supported(L.L) &&
                                            (L.n_slots >= hp.n_expert ||
                                             (pp_streaming && L.hblobs && pp_stage_bytes >= L.stride && pp_layer_cnt[(size_t) il] > 0)));
            const bool want_async_moe_zero = (!fuse_moe_zero_pre && !no_async_moe_zero && !no_moe_atomic_pre && !capturing_full_graph &&
                                              !include_route_t1 && aux_st != nullptr && T >= 32 && cc_maj >= 8 &&
                                              (N & 3) == 0 && !save_conv_trace && moe_raw_disjoint_ctrl &&
                                              L.gu_type != 30 && L.d_type != 30 && K::native_expert_wmma_supported(L.L) &&
                                              (L.n_slots >= hp.n_expert ||
                                               (pp_streaming && L.hblobs && pp_stage_bytes >= L.stride && pp_layer_cnt[(size_t) il] > 0)));
            auto kick_async_moe_zero = [&]() {
                if (!want_async_moe_zero || moe_raw_zero_pending) return;
                CK(cudaEventRecord(ev_attn_done, st));
                CK(cudaStreamWaitEvent(aux_st, ev_attn_done, 0));
                if (!no_zero_f4_cs && T >= 128 && cc_maj >= 8) {
                    k_zero_f4_cs<<<(unsigned) moe_zero_ctas, 256, 0, aux_st>>>(moe_raw, (size_t) T * (size_t) N);
                } else {
                    CK(cudaMemsetAsync(moe_raw, 0, (size_t) T * (size_t) N * sizeof(float), aux_st));
                }
                CK(cudaEventRecord(ev_moe_zeroed, aux_st));
                moe_raw_zero_pending = true;
            };
            if (fused_pre_attn_t1) {
                const int d_splits = (max_rel > 512) ? 4 : 2;
                const unsigned n_ctas = (unsigned) (2 * (int) hp.n_head_kv + (int) hp.n_head * d_splits);
                k_pre_attn_t1_fused<<<n_ctas, 128, 0, st>>>(
                    pK, pV, L.attn_k_conv, L.attn_v_conv, is_swa ? nullptr : pDynK, is_swa ? nullptr : pDynV,
                    hst, cur_kq, cur_ks, cur_vq, cur_vs,
                    pQ, L.attn_q_clip, hp.qk_clip, pR, L.attn_rel_clip, hp.rel_clip,
                    L.attn_rel_bias, sink, rel_tab,
                    (int) hp.n_head, (int) hp.n_head_kv, (int) hp.n_rel_ext, max_rel,
                    cur_kv_cap, is_swa ? 1 : 0, hp.rms_eps, apply_tau, pos0_d,
                    (int) hp.log_scale_floor, hp.log_scale_alpha, d_splits,
                    fuse_sink_pre_attn ? (const uint16_t*) L.attn_sink_proj.d : nullptr,
                    fuse_sink_pre_attn ? in_q : nullptr, fuse_sink_pre_attn ? N : 0);
            } else {
            const bool fused_kv_q8 = (T < 32 && kv_q8 && hp.head_dim == 128);
            if (fused_kv_q8) {
                k_shortconv_kv_clip_norm_store_q8<<<dim3((unsigned) hp.n_head_kv, 2), 128, 0, st>>>(
                    pK, pV, L.attn_k_conv, L.attn_v_conv, is_swa ? nullptr : pDynK, is_swa ? nullptr : pDynV,
                    hst, cur_kq, cur_ks, cur_vq, cur_vs,
                    (int) hp.n_head_kv, hp.qk_clip, hp.rms_eps, pos0_d, cur_kv_cap, is_swa ? 1 : 0,
                    T, h_tr, total_conv_floats);
            } else if (fused_kv_q8_pp) {
                if (pK_f16 != nullptr && pV_f16 != nullptr) {
                    const dim3 kv_grid_32((unsigned) hp.n_head_kv, (unsigned) ((T + 31) / 32));
                    if (is_swa || (pDynK == nullptr && pDynV == nullptr)) {
                        k_shortconv_kv_clip_norm_store_q8_pp_f16<false, 32><<<kv_grid_32, 64, 0, st>>>(
                            pK_f16, pV_f16, L.attn_k_conv, L.attn_v_conv, nullptr, nullptr,
                            hst, cur_kq, cur_ks, cur_vq, cur_vs,
                            (int) hp.n_head_kv, hp.qk_clip, hp.rms_eps, pos0_d, cur_kv_cap,
                            use_swa_pp_hm ? 0 : (is_swa ? 1 : 0), T, ld_kv_sconv, use_swa_pp_hm ? swa_hm_j_hist : 0);
                    } else {
                        k_shortconv_kv_clip_norm_store_q8_pp_f16<true, 32><<<kv_grid_32, 64, 0, st>>>(
                            pK_f16, pV_f16, L.attn_k_conv, L.attn_v_conv, pDynK, pDynV,
                            hst, cur_kq, cur_ks, cur_vq, cur_vs,
                            (int) hp.n_head_kv, hp.qk_clip, hp.rms_eps, pos0_d, cur_kv_cap, is_swa ? 1 : 0, T, ld_kv_sconv);
                    }
                    k_shortconv_kv_hist_push_f16<<<(unsigned) ((2 * kv_dim + 255) / 256), 256, 0, st>>>(
                        pK_f16, pV_f16, hst, (int) kv_dim, T, ld_kv_sconv);
                } else {
                    k_shortconv_kv_clip_norm_store_q8_pp<<<dim3((unsigned) hp.n_head_kv, (unsigned) ((T + 63) / 64)), 64, 0, st>>>(
                        pK, pV, L.attn_k_conv, L.attn_v_conv, is_swa ? nullptr : pDynK, is_swa ? nullptr : pDynV,
                        hst, cur_kq, cur_ks, cur_vq, cur_vs,
                        (int) hp.n_head_kv, hp.qk_clip, hp.rms_eps, pos0_d, cur_kv_cap, is_swa ? 1 : 0, T, ld_kv_sconv);
                    k_shortconv_kv_hist_push<<<(unsigned) ((2 * kv_dim + 255) / 256), 256, 0, st>>>(
                        pK, pV, hst, (int) kv_dim, T, ld_kv_sconv);
                }
            } else if (T >= 32 && !save_conv_trace) {
                k_shortconv_kv_pp<<<dim3((unsigned) ((2 * kv_dim + 255) / 256), (unsigned) ((T + 15) / 16)), 256, 0, st>>>(
                    pK, pV, L.attn_k_conv, L.attn_v_conv, is_swa ? nullptr : pDynK, is_swa ? nullptr : pDynV,
                    hst, K_conv, V_conv, (int) kv_dim, (int) hp.n_head_kv, T);
                k_shortconv_kv_hist_push<<<(unsigned) ((2 * kv_dim + 255) / 256), 256, 0, st>>>(
                    pK, pV, hst, (int) kv_dim, T);
            } else {
                k_shortconv_kv<<<(unsigned) ((2 * kv_dim + 255) / 256), 256, 0, st>>>(
                    pK, pV, L.attn_k_conv, L.attn_v_conv, is_swa ? nullptr : pDynK, is_swa ? nullptr : pDynV,
                    hst, K_conv, V_conv, (int) kv_dim, (int) hp.n_head_kv, T, h_tr, total_conv_floats);
            }

            if (!use_fa2_192_swa_fast) {
                if (hp.head_dim == 128 && hp.n_rel <= 32) {
                    k_clip_and_tau_qr<<<dim3((unsigned) ((hp.n_head + 3) / 4), (unsigned) T), 128, 0, st>>>(
                        pQ, L.attn_q_clip, hp.qk_clip, pR, L.attn_rel_clip, hp.rel_clip,
                        (int) hp.n_head, (int) hp.n_rel, apply_tau, pos0_d, (int) hp.log_scale_floor, hp.log_scale_alpha);
                } else {
                    k_clip_and_tau<<<dim3((unsigned) hp.n_head, (unsigned) T), 32, 0, st>>>(
                        pQ, L.attn_q_clip, hp.qk_clip, (int) hp.head_dim, (int) hp.n_head, apply_tau, pos0_d, (int) hp.log_scale_floor, hp.log_scale_alpha);
                    k_clip_and_tau<<<dim3((unsigned) hp.n_head, (unsigned) T), 32, 0, st>>>(
                        pR, L.attn_rel_clip, hp.rel_clip, (int) hp.n_rel, (int) hp.n_head, apply_tau, pos0_d, (int) hp.log_scale_floor, hp.log_scale_alpha);
                }
            }

            if (!fused_kv_q8 && !fused_kv_q8_pp) {
                if (kv_q8) {
                    k_clip_norm_kv_store_q8<<<dim3((unsigned) hp.n_head_kv, (unsigned) T), 32, 0, st>>>(
                        K_conv, V_conv, cur_kq, cur_ks, cur_vq, cur_vs,
                        (int) hp.n_head_kv, hp.qk_clip, hp.rms_eps, pos0_d, cur_kv_cap, is_swa ? 1 : 0);
                } else {
                    k_clip_and_tau<<<dim3((unsigned) hp.n_head_kv, (unsigned) T), 32, 0, st>>>(
                        K_conv, nullptr, hp.qk_clip, (int) hp.head_dim, (int) hp.n_head_kv, 0, pos0_d, (int) hp.log_scale_floor, hp.log_scale_alpha);
                    k_v_rms_norm<<<dim3((unsigned) hp.n_head_kv, (unsigned) T), 32, 0, st>>>(V_conv, (int) hp.head_dim, (int) hp.n_head_kv, hp.rms_eps);
                    k_kv_store<<<dim3((unsigned) ((kv_dim + 255) / 256), (unsigned) T), 256, 0, st>>>(
                        K_conv, V_conv, cur_kc, cur_vc, (int) kv_dim, pos0_d, cur_kv_cap, is_swa ? 1 : 0);
                }
            }

            if (!fuse_swa_rel_pp && !fuse_full_rel_pp && !sub_tile_fa2_256) {
                if (T >= 32 && hp.n_rel == 16) {
                    k_rel_table_t32<<<dim3((unsigned) hp.n_head, (unsigned) ((T + 31) / 32)), kThreads, 0, st>>>(
                        pR, L.attn_rel_bias, sink, rel_tab, (int) hp.n_head, (int) hp.n_rel, (int) hp.n_rel_ext, max_rel, pos0_d, T);
                } else if (T >= 1 && T <= 8 && hp.n_rel == 16) {
                    const int d_splits = (max_rel > 512) ? 4 : 2;
                    k_rel_table_split<<<dim3((unsigned) hp.n_head, (unsigned) d_splits), 128, 0, st>>>(
                        pR, L.attn_rel_bias, sink, rel_tab, (int) hp.n_head, (int) hp.n_rel, (int) hp.n_rel_ext, max_rel, pos0_d, T, d_splits);
                } else {
                    k_rel_table<<<dim3((unsigned) hp.n_head, (unsigned) ((T + 7) / 8)), kThreads, 0, st>>>(
                        pR, L.attn_rel_bias, sink, rel_tab, (int) hp.n_head, (int) hp.n_rel, (int) hp.n_rel_ext, max_rel, pos0_d, T);
                }
            }
            }
            if (moe_raw_disjoint_attn) kick_async_moe_zero();
            const float kq_scale = 1.0f / std::sqrt((float) hp.head_dim);
            const void* att_xq_ready = nullptr;
            const uint16_t* att_f16_ready = nullptr;
            if (kv_q8) {
                if (hp.n_head == 3 * hp.n_head_kv && hp.head_dim == 128) {
                    const int sp_cap = std::max(1, attn_split_slots / std::max(1, T));
                    if (is_swa && T >= 2 && T <= 5) {
                        const int sp = std::min(sp_cap, std::clamp((sm_count + (int) hp.n_head_kv - 1) / std::max(1, (int) hp.n_head_kv), 4, 16));
                        if (T == 5) {
                            k_attn_q8_gqa3_verify4_swa_wmma_split<5><<<dim3((unsigned) hp.n_head_kv, (unsigned) sp), 128, 23040, st>>>(
                                pQ, cur_kq, cur_ks, cur_vq, cur_vs, rel_tab, sink, attn_split,
                                (int) hp.n_head, (int) hp.n_head_kv, cur_kv_cap, (int) hp.n_swa, (int) hp.n_rel_ext, pos0_d, kq_scale, sp);
                        } else if (T == 4) {
                            k_attn_q8_gqa3_verify4_swa_wmma_split<4><<<dim3((unsigned) hp.n_head_kv, (unsigned) sp), 128, 23040, st>>>(
                                pQ, cur_kq, cur_ks, cur_vq, cur_vs, rel_tab, sink, attn_split,
                                (int) hp.n_head, (int) hp.n_head_kv, cur_kv_cap, (int) hp.n_swa, (int) hp.n_rel_ext, pos0_d, kq_scale, sp);
                        } else if (T == 3) {
                            k_attn_q8_gqa3_verify4_swa_wmma_split<3><<<dim3((unsigned) hp.n_head_kv, (unsigned) sp), 128, 23040, st>>>(
                                pQ, cur_kq, cur_ks, cur_vq, cur_vs, rel_tab, sink, attn_split,
                                (int) hp.n_head, (int) hp.n_head_kv, cur_kv_cap, (int) hp.n_swa, (int) hp.n_rel_ext, pos0_d, kq_scale, sp);
                        } else {
                            k_attn_q8_gqa3_verify4_swa_wmma_split<2><<<dim3((unsigned) hp.n_head_kv, (unsigned) sp), 128, 23040, st>>>(
                                pQ, cur_kq, cur_ks, cur_vq, cur_vs, rel_tab, sink, attn_split,
                                (int) hp.n_head, (int) hp.n_head_kv, cur_kv_cap, (int) hp.n_swa, (int) hp.n_rel_ext, pos0_d, kq_scale, sp);
                        }
                        k_attn_split_combine<<<dim3((unsigned) hp.n_head, (unsigned) T), 128, 0, st>>>(
                            attn_split, pGate, att, (int) hp.n_head, sp, pos0_d, 1, (int) hp.n_swa, xq);
                        att_xq_ready = xq;
                    } else if (is_swa && T >= 6 && T <= 8) {
                        k_attn_q8_gqa3_swa_1pass<<<dim3((unsigned) hp.n_head, (unsigned) T), 128, 0, st>>>(
                            pQ, cur_kq, cur_ks, cur_vq, cur_vs, rel_tab, sink,
                            pGate, att, xq, (int) hp.n_head, (int) hp.n_head_kv, (int) hp.n_rel_ext,
                            cur_kv_cap, (int) hp.n_swa, pos0_d, kq_scale);
                        att_xq_ready = xq;
                    } else if (T >= 2 && T <= 5 && !is_swa) {
                        const int sp = std::min(sp_cap, std::clamp((sm_count * 24) / std::max(1, (int) hp.n_head_kv), 40, kAttnSplitMax));
                        if (T == 5) {
                            k_attn_q8_gqa3_verify4_wmma_split<5><<<dim3((unsigned) hp.n_head_kv, (unsigned) sp), 128, 23040, st>>>(
                                pQ, cur_kq, cur_ks, cur_vq, cur_vs, rel_tab, sink, attn_split,
                                (int) hp.n_head, (int) hp.n_head_kv, cur_kv_cap, (int) hp.n_rel_ext, pos0_d, kq_scale, sp);
                        } else if (T == 4) {
                            k_attn_q8_gqa3_verify4_wmma_split<4><<<dim3((unsigned) hp.n_head_kv, (unsigned) sp), 128, 23040, st>>>(
                                pQ, cur_kq, cur_ks, cur_vq, cur_vs, rel_tab, sink, attn_split,
                                (int) hp.n_head, (int) hp.n_head_kv, cur_kv_cap, (int) hp.n_rel_ext, pos0_d, kq_scale, sp);
                        } else if (T == 3) {
                            k_attn_q8_gqa3_verify4_wmma_split<3><<<dim3((unsigned) hp.n_head_kv, (unsigned) sp), 128, 23040, st>>>(
                                pQ, cur_kq, cur_ks, cur_vq, cur_vs, rel_tab, sink, attn_split,
                                (int) hp.n_head, (int) hp.n_head_kv, cur_kv_cap, (int) hp.n_rel_ext, pos0_d, kq_scale, sp);
                        } else {
                            k_attn_q8_gqa3_verify4_wmma_split<2><<<dim3((unsigned) hp.n_head_kv, (unsigned) sp), 128, 23040, st>>>(
                                pQ, cur_kq, cur_ks, cur_vq, cur_vs, rel_tab, sink, attn_split,
                                (int) hp.n_head, (int) hp.n_head_kv, cur_kv_cap, (int) hp.n_rel_ext, pos0_d, kq_scale, sp);
                        }
                        k_attn_split_combine<<<dim3((unsigned) hp.n_head, (unsigned) T), 128, 0, st>>>(
                            attn_split, pGate, att, (int) hp.n_head, sp, pos0_d, 0, (int) hp.n_swa, xq);
                        att_xq_ready = xq;
                    } else if (T < 16) {
                        const int sp = std::min(sp_cap, (T == 1)
                            ? (is_swa ? attn_split_t1_swa : attn_split_t1_full)
                            : (is_swa ? std::clamp((sm_count * 2) / std::max(1, (int) hp.n_head_kv * T), 4, 24)
                                      : std::clamp((sm_count * 16) / std::max(1, (int) hp.n_head_kv * T), 16, kAttnSplitMax)));
                        k_attn_q8_gqa3_split<<<dim3((unsigned) hp.n_head_kv, (unsigned) sp, (unsigned) T), 128, 0, st>>>(
                            pQ, cur_kq, cur_ks, cur_vq, cur_vs, rel_tab, sink, attn_split,
                            (int) hp.n_head, (int) hp.n_head_kv, (int) hp.n_rel_ext, cur_kv_cap, is_swa ? 1 : 0, (int) hp.n_swa, pos0_d, kq_scale, sp);
                        k_attn_split_combine<<<dim3((unsigned) hp.n_head, (unsigned) T), 128, 0, st>>>(
                            attn_split, pGate, att, (int) hp.n_head, sp, pos0_d, is_swa ? 1 : 0, (int) hp.n_swa, (T <= 8) ? xq : nullptr);
                        if (T <= 8) att_xq_ready = xq;
                    } else if (use_fa2_256_path) {
                        // Zero-sP, Zero-Spill 256-row FA2: pure bulk Split-KV [0, bulk_aligned) + 1 tail split [bulk_aligned, pos+T).
                        // Uses Region 0 pp_sp_buf inside layer_arena (zero extra VRAM; disjoint from Q/att_gate/att & rel_tab).
                        const int q_tiles = (T + 255) / 256;
                        const int base_ctas = (int) hp.n_head * q_tiles;
                        const size_t per_sp_bytes = (size_t) T * (size_t) hp.n_head * 130 * sizeof(float);
                        const int max_total_sp = (int) std::min<size_t>(9, pp_sp_bytes / per_sp_bytes);
                        const int max_bulk_sp = std::max(1, max_total_sp - 1);
                        const int bulk_end_min = (int) pos - (int) hp.n_rel_ext + 1;
                        const int bulk_aligned = (bulk_end_min > 0) ? ((bulk_end_min / 64) * 64) : 0;
                        int best_sp = 1;
                        if (bulk_aligned >= 64 * 64 && max_bulk_sp >= 2 && sm_count > 0) {
                            const int n_kv_tiles = bulk_aligned / 64;
                            const int max_sp = std::min(max_bulk_sp, std::max(1, n_kv_tiles / 64));
                            double best_cost = (double) ((base_ctas + sm_count - 1) / sm_count);
                            for (int sp = 2; sp <= max_sp; ++sp) {
                                const double cost = (double) ((base_ctas * sp + sm_count - 1) / sm_count) / (double) sp;
                                if (cost + 1e-4 < best_cost) {
                                    best_cost = cost;
                                    best_sp = sp;
                                }
                            }
                        }
                        const int total_sp = best_sp + 1;
                        k_attn_q8_gqa3_pp_fa2_256_bulk<<<dim3((unsigned) (3 * q_tiles), (unsigned) hp.n_head_kv, (unsigned) best_sp), 256, (size_t) fa2_256_bulk_smem, st>>>(
                            pQ, cur_kq, cur_ks, cur_vq, cur_vs, sink, pp_sp_buf,
                            (int) hp.n_head, (int) hp.n_head_kv, cur_kv_cap, bulk_aligned, kq_scale, T, best_sp, total_sp);
                        if (sub_tile_fa2_256) {
                            const int sub_step = (rel_tab_cap_t / 256) * 256;
                            for (int t_off = 0; t_off < T; t_off += sub_step) {
                                const int cur_t = std::min(sub_step, T - t_off);
                                const int32_t* pos_sub_d = pos0_d + 1 + (t_off / 256);
                                k_rel_table_t32<<<dim3((unsigned) hp.n_head, (unsigned) ((cur_t + 31) / 32)), kThreads, 0, st>>>(
                                    pR + (size_t) t_off * rel_dim, L.attn_rel_bias, sink + (size_t) t_off * hp.n_head, rel_tab,
                                    (int) hp.n_head, (int) hp.n_rel, (int) hp.n_rel_ext, max_rel, pos_sub_d, cur_t);
                                const int cur_qt128 = (cur_t + 127) / 128;
                                k_attn_q8_gqa3_pp_fa2_256_tail<<<dim3((unsigned) (3 * cur_qt128), (unsigned) hp.n_head_kv), 256, (size_t) fa2_256_tail_smem, st>>>(
                                    pQ + (size_t) t_off * hp.q_dim, cur_kq, cur_ks, cur_vq, cur_vs,
                                    rel_tab, sink + (size_t) t_off * hp.n_head,
                                    pp_sp_buf + (size_t) t_off * (size_t) total_sp * (size_t) hp.n_head * 130,
                                    (int) hp.n_head, (int) hp.n_head_kv, cur_kv_cap, (int) hp.n_rel_ext,
                                    bulk_aligned + t_off, pos_sub_d, kq_scale, cur_t, best_sp, total_sp);
                            }
                        } else {
                            const int q_tiles_128 = (T + 127) / 128;
                            k_attn_q8_gqa3_pp_fa2_256_tail<<<dim3((unsigned) (3 * q_tiles_128), (unsigned) hp.n_head_kv), 256, (size_t) fa2_256_tail_smem, st>>>(
                                pQ, cur_kq, cur_ks, cur_vq, cur_vs, rel_tab, sink, pp_sp_buf,
                                (int) hp.n_head, (int) hp.n_head_kv, cur_kv_cap, (int) hp.n_rel_ext, bulk_aligned, pos0_d, kq_scale, T, best_sp, total_sp);
                        }
                        k_attn_pp_split_combine<<<dim3((unsigned) hp.n_head, (unsigned) T), 128, 0, st>>>(
                            pp_sp_buf, pGate, att, (int) hp.n_head, total_sp, want_att_f16 ? act_bf16 : nullptr);
                        if (want_att_f16) att_f16_ready = act_bf16;
                    } else if (can_fa2_192 && T >= 128) {
                        const int q_tiles = (T + 63) / 64;
                        const int base_ctas = (int) hp.n_head_kv * q_tiles;
                        const size_t per_sp_bytes = (size_t) T * (size_t) hp.n_head * 130 * sizeof(float);
                        const int max_sp_mem = (pp_sp_buf && per_sp_bytes > 0) ? (int) std::min<size_t>(9, pp_sp_bytes / per_sp_bytes) : 1;
                        int best_sp = 1;
                        if (!is_swa && pos >= 8192 && max_sp_mem >= 2 && sm_count > 0) {
                            // Choose split factor that minimizes tail-wave waste (waves / sp) while keeping >= 64 tiles/split.
                            const int n_kv_tiles = (int) ((pos + T + 63) / 64);
                            const int max_sp = std::min(max_sp_mem, std::max(1, n_kv_tiles / 64));
                            double best_cost = (double) ((base_ctas + sm_count - 1) / sm_count);
                            for (int sp = 2; sp <= max_sp; ++sp) {
                                const double cost = (double) ((base_ctas * sp + sm_count - 1) / sm_count) / (double) sp;
                                if (cost + 1e-4 < best_cost) {
                                    best_cost = cost;
                                    best_sp = sp;
                                }
                            }
                        }
                        if (best_sp > 1) {
                            k_attn_q8_gqa3_pp_fa2_192<true, false, false><<<dim3((unsigned) q_tiles, (unsigned) hp.n_head_kv, (unsigned) best_sp), 256, 97280, st>>>(
                                pQ, cur_kq, cur_ks, cur_vq, cur_vs, rel_tab, sink, pGate, att, pp_sp_buf,
                                (int) hp.n_head, (int) hp.n_head_kv, (int) hp.n_rel_ext, cur_kv_cap, (int) hp.n_swa, pos0_d, kq_scale, T, best_sp);
                            k_attn_pp_split_combine<<<dim3((unsigned) hp.n_head, (unsigned) T), 128, 0, st>>>(
                                pp_sp_buf, pGate, att, (int) hp.n_head, best_sp, want_att_f16 ? act_bf16 : nullptr);
                            if (want_att_f16) att_f16_ready = act_bf16;
                        } else if (is_swa && fuse_swa_rel_pp && can_fa2_192_swa_fast) {
                            const bool has_presplit_rb = (L.attn_rel_bias_f16_hi != nullptr && L.attn_rel_bias_f16_lo != nullptr);
                            if (use_swa_pp_hm) {
                                if (has_presplit_rb) {
                                    k_attn_q8_gqa3_pp_fa2_192_swa_fast<true, true, true><<<dim3((unsigned) hp.n_head_kv, (unsigned) q_tiles), 384, (size_t) fa2_192_swa_smem, st>>>(
                                        pQ, cur_kq, cur_ks, cur_vq, cur_vs, pSink, pGate, att,
                                        (int) hp.n_head, (int) hp.n_head_kv, (int) hp.n_rel_ext, cur_kv_cap, (int) hp.n_swa, pos0_d, kq_scale, T,
                                        pR, L.attn_rel_bias, want_att_f16 ? act_bf16 : nullptr,
                                        L.attn_q_clip, hp.qk_clip, L.attn_rel_clip, hp.rel_clip, (float) hp.log_scale_floor, 0.0f,
                                        ld_q_attn, ld_r_attn, ld_gate_attn,
                                        L.attn_rel_bias_f16_hi, L.attn_rel_bias_f16_lo, pGate_f16, ld_sink_attn, swa_hm_j_hist);
                                } else {
                                    k_attn_q8_gqa3_pp_fa2_192_swa_fast<true, false, true><<<dim3((unsigned) hp.n_head_kv, (unsigned) q_tiles), 384, (size_t) fa2_192_swa_smem, st>>>(
                                        pQ, cur_kq, cur_ks, cur_vq, cur_vs, pSink, pGate, att,
                                        (int) hp.n_head, (int) hp.n_head_kv, (int) hp.n_rel_ext, cur_kv_cap, (int) hp.n_swa, pos0_d, kq_scale, T,
                                        pR, L.attn_rel_bias, want_att_f16 ? act_bf16 : nullptr,
                                        L.attn_q_clip, hp.qk_clip, L.attn_rel_clip, hp.rel_clip, (float) hp.log_scale_floor, 0.0f,
                                        ld_q_attn, ld_r_attn, ld_gate_attn,
                                        nullptr, nullptr, pGate_f16, ld_sink_attn, swa_hm_j_hist);
                                }
                            } else if (has_presplit_rb) {
                                k_attn_q8_gqa3_pp_fa2_192_swa_fast<true, true><<<dim3((unsigned) hp.n_head_kv, (unsigned) q_tiles), 384, (size_t) fa2_192_swa_smem, st>>>(
                                    pQ, cur_kq, cur_ks, cur_vq, cur_vs, pSink, pGate, att,
                                    (int) hp.n_head, (int) hp.n_head_kv, (int) hp.n_rel_ext, cur_kv_cap, (int) hp.n_swa, pos0_d, kq_scale, T,
                                    pR, L.attn_rel_bias, want_att_f16 ? act_bf16 : nullptr,
                                    L.attn_q_clip, hp.qk_clip, L.attn_rel_clip, hp.rel_clip, (float) hp.log_scale_floor, 0.0f,
                                    ld_q_attn, ld_r_attn, ld_gate_attn,
                                    L.attn_rel_bias_f16_hi, L.attn_rel_bias_f16_lo, pGate_f16, ld_sink_attn);
                            } else {
                                k_attn_q8_gqa3_pp_fa2_192_swa_fast<true, false><<<dim3((unsigned) hp.n_head_kv, (unsigned) q_tiles), 384, (size_t) fa2_192_swa_smem, st>>>(
                                    pQ, cur_kq, cur_ks, cur_vq, cur_vs, pSink, pGate, att,
                                    (int) hp.n_head, (int) hp.n_head_kv, (int) hp.n_rel_ext, cur_kv_cap, (int) hp.n_swa, pos0_d, kq_scale, T,
                                    pR, L.attn_rel_bias, want_att_f16 ? act_bf16 : nullptr,
                                    L.attn_q_clip, hp.qk_clip, L.attn_rel_clip, hp.rel_clip, (float) hp.log_scale_floor, 0.0f,
                                    ld_q_attn, ld_r_attn, ld_gate_attn,
                                    nullptr, nullptr, pGate_f16, ld_sink_attn);
                            }
                            if (want_att_f16) att_f16_ready = act_bf16;
                        } else if (!is_swa && fuse_full_rel_pp && can_fa2_192_swa_fast) {
                            const bool has_presplit_rb = (L.attn_rel_bias_f16_hi != nullptr && L.attn_rel_bias_f16_lo != nullptr);
                            const float tau_eff = ((pos + T) <= (int) hp.log_scale_floor) ? 0.0f : hp.log_scale_alpha;
                            if (has_presplit_rb) {
                                k_attn_q8_gqa3_pp_fa2_192_swa_fast<false, true><<<dim3((unsigned) hp.n_head_kv, (unsigned) q_tiles), 384, (size_t) fa2_192_swa_smem, st>>>(
                                    pQ, cur_kq, cur_ks, cur_vq, cur_vs, pSink, pGate, att,
                                    (int) hp.n_head, (int) hp.n_head_kv, (int) hp.n_rel_ext, cur_kv_cap, (int) hp.n_swa, pos0_d, kq_scale, T,
                                    pR, L.attn_rel_bias, want_att_f16 ? act_bf16 : nullptr,
                                    L.attn_q_clip, hp.qk_clip, L.attn_rel_clip, hp.rel_clip, (float) hp.log_scale_floor, tau_eff,
                                    ld_q_attn, ld_r_attn, ld_gate_attn,
                                    L.attn_rel_bias_f16_hi, L.attn_rel_bias_f16_lo, pGate_f16, ld_sink_attn);
                            } else {
                                k_attn_q8_gqa3_pp_fa2_192_swa_fast<false, false><<<dim3((unsigned) hp.n_head_kv, (unsigned) q_tiles), 384, (size_t) fa2_192_swa_smem, st>>>(
                                    pQ, cur_kq, cur_ks, cur_vq, cur_vs, pSink, pGate, att,
                                    (int) hp.n_head, (int) hp.n_head_kv, (int) hp.n_rel_ext, cur_kv_cap, (int) hp.n_swa, pos0_d, kq_scale, T,
                                    pR, L.attn_rel_bias, want_att_f16 ? act_bf16 : nullptr,
                                    L.attn_q_clip, hp.qk_clip, L.attn_rel_clip, hp.rel_clip, (float) hp.log_scale_floor, tau_eff,
                                    ld_q_attn, ld_r_attn, ld_gate_attn,
                                    nullptr, nullptr, pGate_f16, ld_sink_attn);
                            }
                            if (want_att_f16) att_f16_ready = act_bf16;
                        } else if (is_swa && fuse_swa_rel_pp) {
                            k_attn_q8_gqa3_pp_fa2_192<false, true, true><<<dim3((unsigned) q_tiles, (unsigned) hp.n_head_kv), 256, 97280, st>>>(
                                pQ, cur_kq, cur_ks, cur_vq, cur_vs, nullptr, sink, pGate, att, nullptr,
                                (int) hp.n_head, (int) hp.n_head_kv, (int) hp.n_rel_ext, cur_kv_cap, (int) hp.n_swa, pos0_d, kq_scale, T, 1,
                                pR, L.attn_rel_bias, want_att_f16 ? act_bf16 : nullptr);
                            if (want_att_f16) att_f16_ready = act_bf16;
                        } else if (is_swa) {
                            k_attn_q8_gqa3_pp_fa2_192<false, true, false><<<dim3((unsigned) q_tiles, (unsigned) hp.n_head_kv), 256, 97280, st>>>(
                                pQ, cur_kq, cur_ks, cur_vq, cur_vs, rel_tab, sink, pGate, att, nullptr,
                                (int) hp.n_head, (int) hp.n_head_kv, (int) hp.n_rel_ext, cur_kv_cap, (int) hp.n_swa, pos0_d, kq_scale, T, 1,
                                nullptr, nullptr, want_att_f16 ? act_bf16 : nullptr);
                            if (want_att_f16) att_f16_ready = act_bf16;
                        } else {
                            k_attn_q8_gqa3_pp_fa2_192<false, false, false><<<dim3((unsigned) q_tiles, (unsigned) hp.n_head_kv), 256, 97280, st>>>(
                                pQ, cur_kq, cur_ks, cur_vq, cur_vs, rel_tab, sink, pGate, att, nullptr,
                                (int) hp.n_head, (int) hp.n_head_kv, (int) hp.n_rel_ext, cur_kv_cap, (int) hp.n_swa, pos0_d, kq_scale, T, 1,
                                nullptr, nullptr, want_att_f16 ? act_bf16 : nullptr);
                            if (want_att_f16) att_f16_ready = act_bf16;
                        }
                    } else {
                        k_attn_q8_gqa3_pp_wmma<<<dim3((unsigned) hp.n_head, (unsigned) ((T + 15) / 16)), 128, 29184, st>>>(
                            pQ, cur_kq, cur_ks, cur_vq, cur_vs, rel_tab, sink, pGate, att,
                            (int) hp.n_head, (int) hp.n_head_kv, (int) hp.n_rel_ext, cur_kv_cap, is_swa ? 1 : 0, (int) hp.n_swa, pos0_d, kq_scale, T);
                    }
                } else {
                    k_attn_q8<<<dim3((unsigned) hp.n_head, (unsigned) T), 128, 0, st>>>(
                        pQ, cur_kq, cur_ks, cur_vq, cur_vs, rel_tab, sink, pGate, att,
                        (int) hp.n_head, (int) hp.n_head_kv, (int) hp.n_rel_ext, cur_kv_cap, is_swa ? 1 : 0, (int) hp.n_swa, pos0_d, kq_scale);
                }
            } else {
                k_attn<<<dim3((unsigned) hp.n_head, (unsigned) T), 128, 0, st>>>(
                    pQ, cur_kc, cur_vc, rel_tab, sink, att,
                    (int) hp.n_head, (int) hp.n_head_kv, (int) hp.n_rel_ext, cur_kv_cap, is_swa ? 1 : 0, (int) hp.n_swa, pos0_d, kq_scale);
            }
            if (use_swa_pp) {
                const int keep = (int) std::min<int64_t>(pos + T, hp.n_swa);
                const int j_keep = (int) (pos + T - keep);
                if (use_swa_pp_hm) {
                    k_swa_kv_hm_to_ring_q8<<<(unsigned) keep, 256, 0, st>>>(
                        swa_pp_kq, swa_pp_ks, swa_pp_vq, swa_pp_vs,
                        kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il],
                        j_keep, swa_hm_j_hist, keep, swa_pp_cap, ring_cap, (int) hp.n_head_kv);
                } else if (kv_q8) {
                    k_swa_kv_ring_copy_q8<<<(unsigned) keep, 256, 0, st>>>(
                        swa_pp_kq, swa_pp_ks, swa_pp_vq, swa_pp_vs,
                        kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il],
                        j_keep, keep, swa_pp_cap, ring_cap, (int) kv_dim, (int) hp.n_head_kv);
                } else {
                    k_swa_kv_ring_copy_f16<<<(unsigned) keep, 256, 0, st>>>(
                        swa_pp_kc, swa_pp_vc,
                        kc[(size_t) il], vc[(size_t) il],
                        j_keep, keep, swa_pp_cap, ring_cap, (int) kv_dim);
                }
            }
            if (prof_detail) { CK(cudaStreamSynchronize(st)); p_d_attn += ms_since(tp_d2); }

            const auto tp_d3 = Clock::now();
            if (!kv_q8) {
                k_mul_sigmoid<<<(unsigned) (((int64_t) T * q_dim + 255) / 256), 256, 0, st>>>(att, pGate, (int64_t) T * q_dim);
            }
            project_preq(att, att_xq_ready, L.attn_output, wo, T, N, q_dim, att_f16_ready);
            if (use_dequant_side || use_wo_dequant_side) {
                std::string side_err;
                gemm.set_dequant_side(nullptr, side_err);
            }
            if (moe_raw_disjoint_wo) kick_async_moe_zero();
            const bool do_push = (ip % hp.attn_res_block_size == 0);
            const bool fuse_post_attn_ffn_t1 = (T == 1 && !do_push && !has_msg && !save_conv_trace &&
                                                N == 2304 && (int) norm_blocks.size() + 1 <= 8);
            static const bool no_sconv_mix = (std::getenv("STRATA_NO_SCONV_MIX") != nullptr);
            const int B_after_push = (int) norm_blocks.size() + (do_push ? 1 : 0);
            const bool fuse_post_attn_ffn_pp = (!no_sconv_mix && T >= 32 && N == 2304 && !save_conv_trace &&
                                                B_after_push <= 7 && L.ffn_res_query != nullptr && act_bf16 != nullptr);
            auto launch_vmsg_ffn_ctrl = [&]() {
                const int l_idx = ip - (int) hp.core_start;
                const float* w_m3 = vmsg_w + (size_t) (l_idx * 4 + 3) * (size_t) T * 2;
                const float* im_m3 = vmsg_inv_rms + (size_t) (l_idx * 4 + 3) * (size_t) T;
                if (vmsg_hoist_base_l >= 0 && l_idx >= vmsg_hoist_base_l && l_idx < vmsg_hoist_base_l + 6 && vmsg_down_pre != nullptr) {
                    const int local_m3 = (l_idx - vmsg_hoist_base_l) * 4 + 3;
                    CtrlDownPhaseAPtrs3 dp_fa{};
                    dp_fa.c[0] = in_ffn;
                    dp_fa.inv_c[0] = ir_c0;
                    dp_fa.w[0] = (const float*) L.loop_down[3].d;
                    dp_fa.w_f16[0] = d_core_wdown_cur_f16 ? (d_core_wdown_cur_f16 + (size_t) (l_idx * 4 + 3) * 64u * (size_t) N) : nullptr;
                    dp_fa.pre[0] = vmsg_down_pre + (size_t) local_m3 * (size_t) T * 64;
                    dp_fa.out[0] = ctrl64;
                    launch_phaseA_splitk<1>(dp_fa, T, N, ffn_splitk_scratch, ffn_splitk_scratch_floats);
                } else {
                    CtrlDownVmsgPtrs3 dp_f{
                        {in_ffn, nullptr, nullptr},
                        {ir_c0, nullptr, nullptr},
                        {w_m3, nullptr, nullptr},
                        {im_m3, nullptr, nullptr},
                        {(const float*) L.loop_down[3].d, nullptr, nullptr},
                        {ctrl64, nullptr, nullptr}
                    };
                    k_control_down_pp_wmma_vmsg<1><<<(unsigned) ((T + 31) / 32), 256, 0, st>>>(vmsg_bank0, vmsg_bank1, dp_f, T, N);
                }
                CtrlUpVmsgPtrs3 up_f{};
                up_f.c[0] = in_ffn;
                up_f.w_vmsg[0] = w_m3;
                up_f.ctrl_in[0] = ctrl64;
                up_f.w_gain[0] = (const float*) L.loop_gain[3].d;
                up_f.w_forget[0] = (const float*) L.loop_forget[3].d;
                if (d_core_wup_f16) {
                    const size_t wup_m = (size_t) N * 64u;
                    up_f.wg_f16[0] = d_core_wup_f16 + (size_t) (l_idx * 4 + 3) * wup_m;
                    up_f.wf_f16[0] = d_core_wup_f16 + (size_t) (48 + l_idx * 4 + 3) * wup_m;
                }
                up_f.out[0] = in_ffn;
                k_control_up_combine_pp_wmma_vmsg<1><<<dim3((unsigned) ((N + 63) / 64), (unsigned) ((T + 127) / 128)), 256, 0, st>>>(
                    vmsg_bank0, vmsg_bank1, up_f, T, N);
                k_rms_norm_f16<<<(unsigned) T, rms_f16_blk, 0, st>>>(in_ffn, L.ffn_norm, in_ffn, act_bf16, N, hp.rms_eps);
            };
            if (fuse_post_attn_ffn_t1) {
                const int B = (int) norm_blocks.size();
                NormMixPtrs np{};
                for (int i = 0; i < B; ++i) np.p[i] = norm_blocks[(size_t) (B - 1 - i)];
                np.p[B] = ffn_inp;
                const float inv_sqrt_n = 1.0f / std::sqrt((float) N);
                k_shortconv_mix_normed_fused_1pass_q8_t1<<<1, 576, 0, st>>>(
                    wo, L.attn_post_conv, hst, cur_in, ffn_inp, 2 * (int) kv_dim,
                    np, B + 1, L.ffn_res_query, in_ffn, L.ffn_norm, xq, N, inv_sqrt_n, hp.rms_eps);
            } else if (fuse_post_attn_ffn_pp) {
                // Note: push_block(cur_in) pushes rms_norm(cur_in) (independent of wo->ffn_inp!), so run it first and
                // keep ndst_out = nullptr in the fused kernel so norm_blocks matches bit-for-bit on do_push layers.
                if (do_push) {
                    if (pre_push_done_il == il) {
                        norm_blocks.push_back(pre_push_ndst);
                        pre_push_done_il = -1;
                    } else {
                        const int pass_cur = (n_core > 0 && ip >= hp.core_start && ip < hp.core_end)
                                           ? ((il - (int) hp.core_start) / (int) n_core) : 0;
                        push_block(cur_in, il, ip, pass_cur);
                    }
                }
                const int B = (int) norm_blocks.size();
                NormMixPtrs np{};
                for (int i = 0; i < B; ++i) np.p[i] = norm_blocks[(size_t) (B - 1 - i)];
                static const bool no_l2_chunk = (std::getenv("STRATA_NO_L2_CHUNK") != nullptr);
                static const bool no_l2_router = (std::getenv("STRATA_NO_L2_ROUTER") != nullptr);
                static const bool no_custom_router_ffn = [] {
                    const char* v = std::getenv("STRATA_NO_CUSTOM_ROUTER");
                    return v != nullptr && (v[0] == '\0' || std::atoi(v) != 0);
                }();
                static const bool router_fp32_exact_ffn = [] {
                    const char* v = std::getenv("STRATA_ROUTER_FP32_EXACT");
                    return v != nullptr && (v[0] == '\0' || std::atoi(v) != 0);
                }();
                static const bool fuse_ctrl_rms_ffn = [] {
                    const char* v = std::getenv("STRATA_FUSE_CTRL_RMS");
                    return v != nullptr && (v[0] == '\0' || std::atoi(v) != 0);
                }();
                const bool need_cpu_ffn = (L.n_slots < hp.n_expert);
                const bool use_pp_stream_ffn = (pp_streaming && need_cpu_ffn && L.hblobs &&
                                                pp_stage_bytes >= L.stride && pp_layer_cnt[(size_t) il] > 0);
                const bool can_l2_router_ffn = (!no_l2_router && !include_route_t1 &&
                                                L.gu_type != 30 && L.d_type != 30 &&
                                                K::native_expert_wmma_supported(L.L) &&
                                                (!need_cpu_ffn || use_pp_stream_ffn));
                const bool f16_ffn_ready_pre = (T >= 16 && (N & 3) == 0);
                const bool use_presplit_router = (!no_custom_router_ffn && !router_fp32_exact_ffn &&
                                                  cc_maj >= 8 && T >= 128 && N == 2304 && hp.n_expert == 128 &&
                                                  L.ffn_gate_inp_hi != nullptr && L.ffn_gate_inp_lo != nullptr &&
                                                  can_l2_router_ffn && f16_ffn_ready_pre);
                auto launch_router_pp = [&](const float* x_ptr, int t_off, int rows) {
                    float* y_ptr = router_logits + (size_t) t_off * (size_t) hp.n_expert;
                    launch_router_gemm_pp(x_ptr, L.ffn_gate_inp, y_ptr, rows, (int) hp.n_expert, N);
                };
                // Wave 12B + Wave 14: BM=64, BN=128, BK=32, Block=256 pre-split Ozaki-2 TC Router with Split-K
                // (with cudaSharedmemCarveoutMaxShared -> 2 CTAs/SM = 1 wave: m_tiles <= 14 -> sk=3 e.g. 13*3=39 CTAs;
                //  15 <= m_tiles && rows <= 1536 -> sk=4 e.g. 23*4=92 CTAs = 2.00 CTAs/SM on 46 SMs; rows > 1536 -> sk=2).
                auto launch_presplit_router_pp = [&](const __half* x_hi, const __half* x_lo, int t_off, int rows) {
                    float* y_ptr = router_logits + (size_t) t_off * (size_t) hp.n_expert;
                    const int m_tiles = (rows + 63) / 64;
                    int sk = (m_tiles <= 14) ? 3 : ((rows <= 1536) ? 4 : 2);
                    while (sk > 1) {
                        const int kps = ((N / sk + 31) / 32) * 32;
                        if (kps * sk >= N && kps * (sk - 1) < N &&
                            ffn_splitk_scratch != nullptr &&
                            (size_t) sk * (size_t) rows * (size_t) hp.n_expert <= ffn_splitk_scratch_floats) {
                            break;
                        }
                        sk = (sk == 3) ? 2 : (sk / 2);
                    }
                    const int k_per_split = (sk > 1) ? (((N / sk + 31) / 32) * 32) : 0;
                    float* y_dst = (sk > 1) ? ffn_splitk_scratch : y_ptr;
                    k_router_ozaki2_tc_presplit_pp<<<dim3(1, (unsigned) m_tiles, (unsigned) sk), 256, 0, st>>>(
                        x_hi, x_lo,
                        L.ffn_gate_inp_hi, L.ffn_gate_inp_lo,
                        y_dst, rows, (int) hp.n_expert, N, k_per_split);
                    if (sk > 1) {
                        const int tot = rows * (int) hp.n_expert;
                        k_router_splitk_reduce<<<(unsigned) (((tot + 3) / 4 + 255) / 256), 256, 0, st>>>(y_dst, y_ptr, tot, sk);
                    }
                };
                if (!has_msg) {
                    if (!no_l2_chunk && can_l2_router_ffn && (use_presplit_router || T >= 1024)) {
                        // Wave 8 Cut C+D + Wave 12B: T_CHUNK=1472 (23 M-tiles x sk=4 = 92 CTAs = 2.00 CTAs/SM on 46 SMs)
                        // with pre-split Ozaki-2 X_hi/X_lo (X_lo reuses in_ffn as __half*, 6.47 MiB <= 0.5*sz_N).
                        const int ffn_chunk = use_presplit_router ? 1472 : 1024;
                        for (int tc0 = 0; tc0 < T; tc0 += ffn_chunk) {
                            const int tc_len = std::min(ffn_chunk, T - tc0);
                            auto launch_sconv_t_b = [&](int len, float* o32, const float* rw, uint16_t* o16, float* irp, int toff, bool loc, __half* xlo) {
                                switch (B) {
                                case 0: k_shortconv_mix_normed_fused_1pass_pp<true, true, 0><<<(unsigned) len, 576, 0, st>>>(wo, L.attn_post_conv, hst, do_push ? nullptr : cur_in, ffn_inp, nullptr, np, B, L.ffn_res_query, o32, rw, o16, irp, 2 * (int) kv_dim, len, N, hp.rms_eps, toff, loc, xlo); break;
                                case 1: k_shortconv_mix_normed_fused_1pass_pp<true, true, 1><<<(unsigned) len, 576, 0, st>>>(wo, L.attn_post_conv, hst, do_push ? nullptr : cur_in, ffn_inp, nullptr, np, B, L.ffn_res_query, o32, rw, o16, irp, 2 * (int) kv_dim, len, N, hp.rms_eps, toff, loc, xlo); break;
                                case 2: k_shortconv_mix_normed_fused_1pass_pp<true, true, 2><<<(unsigned) len, 576, 0, st>>>(wo, L.attn_post_conv, hst, do_push ? nullptr : cur_in, ffn_inp, nullptr, np, B, L.ffn_res_query, o32, rw, o16, irp, 2 * (int) kv_dim, len, N, hp.rms_eps, toff, loc, xlo); break;
                                case 3: k_shortconv_mix_normed_fused_1pass_pp<true, true, 3><<<(unsigned) len, 576, 0, st>>>(wo, L.attn_post_conv, hst, do_push ? nullptr : cur_in, ffn_inp, nullptr, np, B, L.ffn_res_query, o32, rw, o16, irp, 2 * (int) kv_dim, len, N, hp.rms_eps, toff, loc, xlo); break;
                                case 4: k_shortconv_mix_normed_fused_1pass_pp<true, true, 4><<<(unsigned) len, 576, 0, st>>>(wo, L.attn_post_conv, hst, do_push ? nullptr : cur_in, ffn_inp, nullptr, np, B, L.ffn_res_query, o32, rw, o16, irp, 2 * (int) kv_dim, len, N, hp.rms_eps, toff, loc, xlo); break;
                                default: k_shortconv_mix_normed_fused_1pass_pp<true, true, -1><<<(unsigned) len, 576, 0, st>>>(wo, L.attn_post_conv, hst, do_push ? nullptr : cur_in, ffn_inp, nullptr, np, B, L.ffn_res_query, o32, rw, o16, irp, 2 * (int) kv_dim, len, N, hp.rms_eps, toff, loc, xlo); break;
                                }
                            };
                            if (use_presplit_router) {
                                __half* x_lo_buf = reinterpret_cast<__half*>(in_ffn);
                                launch_sconv_t_b(tc_len, nullptr, L.ffn_norm, act_bf16, nullptr, tc0, true, x_lo_buf);
                                launch_presplit_router_pp(
                                    reinterpret_cast<const __half*>(act_bf16 + (size_t) tc0 * N), x_lo_buf, tc0, tc_len);
                            } else {
                                launch_sconv_t_b(tc_len, in_ffn, L.ffn_norm, act_bf16, nullptr, tc0, true, nullptr);
                                launch_router_pp(in_ffn, tc0, tc_len);
                            }
                        }
                        pre_routed_pp_cur = true;
                    } else {
                        k_shortconv_mix_normed_fused_1pass_pp<true, true><<<(unsigned) T, 576, 0, st>>>(
                            wo, L.attn_post_conv, hst, do_push ? nullptr : cur_in,
                            ffn_inp, nullptr, np, B, L.ffn_res_query,
                            in_ffn, L.ffn_norm, act_bf16, nullptr,
                            2 * (int) kv_dim, T, N, hp.rms_eps);
                    }
                    k_shortconv_hist_push<<<(unsigned) ((N + 255) / 256), 256, 0, st>>>(
                        wo, hst, N, 2 * (int) kv_dim, T);
                    kick_async_moe_zero();
                } else if (use_vmsg_layer) {
                    const int l_idx_ffn = ip - (int) hp.core_start;
                    const bool use_l2_chunk_ffn = (!no_l2_chunk && in_k != nullptr && vmsg_hoist_base_l >= 0 &&
                                                   l_idx_ffn >= vmsg_hoist_base_l && l_idx_ffn < vmsg_hoist_base_l + 6 &&
                                                   vmsg_down_pre != nullptr && T > 64 && cc_maj >= 8);
                    if (use_l2_chunk_ffn) {
                        const bool use_l2_router = can_l2_router_ffn;
                        const bool use_fuse_ctrl_rms_1 = (use_l2_router && use_presplit_router &&
                                                          can_ctrl_rms_1 && fuse_ctrl_rms_ffn);
                        const int local_m3 = (l_idx_ffn - vmsg_hoist_base_l) * 4 + 3;
                        const float* w_m3 = vmsg_w + (size_t) (l_idx_ffn * 4 + 3) * (size_t) T * 2;
                        // Wave 7: When !use_l2_router, use in_k ([sz_N..2*sz_N), 100% disjoint from wo/moe_raw/in_ffn/ctrl64)
                        // as the local chunk scratch so tc0 >= ffn_msg_chunk never overwrites normalized in_ffn[0..tc0-1].
                        static const int env_ffn_msg_chunk = [] {
                            const char* v = std::getenv("STRATA_FFN_MSG_CHUNK");
                            return (v != nullptr && v[0] != '\0') ? std::atoi(v) : 0;
                        }();
                        const int ffn_msg_chunk = (env_ffn_msg_chunk > 0) ? env_ffn_msg_chunk
                                                                          : ((use_l2_router && use_presplit_router) ? 1408 : 1024);
                        float* ffn_tmp = use_l2_router ? in_ffn : in_k;
                        for (int tc0 = 0; tc0 < T; tc0 += ffn_msg_chunk) {
                            const int tc_len = std::min(ffn_msg_chunk, T - tc0);
                            switch (B) {
                            case 0: k_shortconv_mix_normed_fused_1pass_pp<false, true, 0><<<(unsigned) tc_len, 576, 0, st>>>(wo, L.attn_post_conv, hst, do_push ? nullptr : cur_in, ffn_inp, nullptr, np, B, L.ffn_res_query, ffn_tmp, nullptr, nullptr, ir_c0, 2 * (int) kv_dim, tc_len, N, hp.rms_eps, tc0, true); break;
                            case 1: k_shortconv_mix_normed_fused_1pass_pp<false, true, 1><<<(unsigned) tc_len, 576, 0, st>>>(wo, L.attn_post_conv, hst, do_push ? nullptr : cur_in, ffn_inp, nullptr, np, B, L.ffn_res_query, ffn_tmp, nullptr, nullptr, ir_c0, 2 * (int) kv_dim, tc_len, N, hp.rms_eps, tc0, true); break;
                            case 2: k_shortconv_mix_normed_fused_1pass_pp<false, true, 2><<<(unsigned) tc_len, 576, 0, st>>>(wo, L.attn_post_conv, hst, do_push ? nullptr : cur_in, ffn_inp, nullptr, np, B, L.ffn_res_query, ffn_tmp, nullptr, nullptr, ir_c0, 2 * (int) kv_dim, tc_len, N, hp.rms_eps, tc0, true); break;
                            case 3: k_shortconv_mix_normed_fused_1pass_pp<false, true, 3><<<(unsigned) tc_len, 576, 0, st>>>(wo, L.attn_post_conv, hst, do_push ? nullptr : cur_in, ffn_inp, nullptr, np, B, L.ffn_res_query, ffn_tmp, nullptr, nullptr, ir_c0, 2 * (int) kv_dim, tc_len, N, hp.rms_eps, tc0, true); break;
                            case 4: k_shortconv_mix_normed_fused_1pass_pp<false, true, 4><<<(unsigned) tc_len, 576, 0, st>>>(wo, L.attn_post_conv, hst, do_push ? nullptr : cur_in, ffn_inp, nullptr, np, B, L.ffn_res_query, ffn_tmp, nullptr, nullptr, ir_c0, 2 * (int) kv_dim, tc_len, N, hp.rms_eps, tc0, true); break;
                            default: k_shortconv_mix_normed_fused_1pass_pp<false, true, -1><<<(unsigned) tc_len, 576, 0, st>>>(wo, L.attn_post_conv, hst, do_push ? nullptr : cur_in, ffn_inp, nullptr, np, B, L.ffn_res_query, ffn_tmp, nullptr, nullptr, ir_c0, 2 * (int) kv_dim, tc_len, N, hp.rms_eps, tc0, true); break;
                            }
                            CtrlDownPhaseAPtrs3 dp_fa{};
                            dp_fa.c[0] = ffn_tmp;
                            dp_fa.inv_c[0] = ir_c0;
                            dp_fa.w[0] = (const float*) L.loop_down[3].d;
                            dp_fa.w_f16[0] = d_core_wdown_cur_f16 ? (d_core_wdown_cur_f16 + (size_t) (l_idx_ffn * 4 + 3) * 64u * (size_t) N) : nullptr;
                            dp_fa.pre[0] = vmsg_down_pre + (size_t) local_m3 * (size_t) T * 64 + (size_t) tc0 * 64;
                            dp_fa.out[0] = ctrl64;
                            launch_phaseA_splitk<1>(dp_fa, tc_len, N, ffn_splitk_scratch, ffn_splitk_scratch_floats);

                            CtrlUpVmsgPtrs3 up_f{};
                            up_f.c[0] = ffn_tmp;
                            up_f.w_vmsg[0] = w_m3 + (size_t) tc0 * 2;
                            up_f.ctrl_in[0] = ctrl64;
                            up_f.w_gain[0] = (const float*) L.loop_gain[3].d;
                            up_f.w_forget[0] = (const float*) L.loop_forget[3].d;
                            if (d_core_wup_f16) {
                                const size_t wup_m = (size_t) N * 64u;
                                up_f.wg_f16[0] = d_core_wup_f16 + (size_t) (l_idx_ffn * 4 + 3) * wup_m;
                                up_f.wf_f16[0] = d_core_wup_f16 + (size_t) (48 + l_idx_ffn * 4 + 3) * wup_m;
                            }
                            up_f.out[0] = ffn_tmp;
                            if (use_fuse_ctrl_rms_1) {
                                // in_k ([sz_N..2*sz_N)) is 100% disjoint from wo, moe_raw, in_ffn (up_f.c[0]),
                                // ffn_splitk_scratch, and ctrl64, so writing X_lo there has zero read/write race!
                                up_f.out[0] = nullptr;
                                __half* x_lo_buf = reinterpret_cast<__half*>(in_k);
                                CtrlUpRmsOuts ro{};
                                ro.rms_w[0] = L.ffn_norm;
                                ro.out_f16[0] = act_bf16 + (size_t) tc0 * N;
                                ro.out_lo_f16 = x_lo_buf;
                                ro.eps = hp.rms_eps;
                                k_control_up_combine_rms_f16_vmsg<1><<<dim3((unsigned) ((tc_len + 15) / 16), 1, 1), 576, 29696, st>>>(
                                    vmsg_bank0 + (size_t) tc0 * N, vmsg_bank1 + (size_t) tc0 * N, up_f, ro, tc_len, N);
                                launch_presplit_router_pp(
                                    reinterpret_cast<const __half*>(act_bf16 + (size_t) tc0 * N), x_lo_buf, tc0, tc_len);
                            } else {
                                k_control_up_combine_pp_wmma_vmsg<1><<<dim3((unsigned) ((N + 63) / 64), (unsigned) ((tc_len + 127) / 128)), 256, 0, st>>>(
                                    vmsg_bank0 + (size_t) tc0 * N, vmsg_bank1 + (size_t) tc0 * N, up_f, tc_len, N);
                                float* ffn_y_dst = use_l2_router ? in_ffn : (in_ffn + (size_t) tc0 * N);
                                if (use_l2_router && use_presplit_router) {
                                    // Wave 13A + Wave 14: Emit Ozaki-2 X_lo into disjoint in_k ([sz_N..2*sz_N)), skip dead
                                    // FP32 in_ffn write (y = nullptr -> zero consumers since pre_routed_pp=true & f16_ffn_ready=true),
                                    // and run pre-split Ozaki-2 TC Router on all 24 has_msg FFN layers.
                                    __half* x_lo_buf = reinterpret_cast<__half*>(in_k);
                                    k_rms_norm_f16<<<(unsigned) tc_len, rms_f16_blk, 0, st>>>(
                                        ffn_tmp, L.ffn_norm, nullptr, act_bf16 + (size_t) tc0 * N, N, hp.rms_eps, x_lo_buf);
                                    launch_presplit_router_pp(
                                        reinterpret_cast<const __half*>(act_bf16 + (size_t) tc0 * N), x_lo_buf, tc0, tc_len);
                                } else {
                                    k_rms_norm_f16<<<(unsigned) tc_len, rms_f16_blk, 0, st>>>(
                                        ffn_tmp, L.ffn_norm, ffn_y_dst, act_bf16 + (size_t) tc0 * N, N, hp.rms_eps, nullptr);
                                    if (use_l2_router) {
                                        launch_router_pp(in_ffn, tc0, tc_len);
                                    }
                                }
                            }
                        }
                        k_shortconv_hist_push<<<(unsigned) ((N + 255) / 256), 256, 0, st>>>(
                            wo, hst, N, 2 * (int) kv_dim, T);
                        kick_async_moe_zero();
                        if (use_l2_router) pre_routed_pp_cur = true;
                    } else {
                        k_shortconv_mix_normed_fused_1pass_pp<false, true><<<(unsigned) T, 576, 0, st>>>(
                            wo, L.attn_post_conv, hst, do_push ? nullptr : cur_in,
                            ffn_inp, nullptr, np, B, L.ffn_res_query,
                            in_ffn, nullptr, nullptr, ir_c0,
                            2 * (int) kv_dim, T, N, hp.rms_eps);
                        k_shortconv_hist_push<<<(unsigned) ((N + 255) / 256), 256, 0, st>>>(
                            wo, hst, N, 2 * (int) kv_dim, T);
                        kick_async_moe_zero();
                        launch_vmsg_ffn_ctrl();
                    }
                } else {
                    const bool ffn_ctrl_ir = (has_msg_ir && T >= 32 && N == 2304 &&
                                              hp.loop_gate_rank == 64 && can_ctrl_wmma && can_ctrl_down_wmma && ir_c0 != nullptr);
                    k_shortconv_mix_normed_fused_1pass_pp<false, true><<<(unsigned) T, 576, 0, st>>>(
                        wo, L.attn_post_conv, hst, do_push ? nullptr : cur_in,
                        ffn_inp, nullptr, np, B, L.ffn_res_query,
                        in_ffn, nullptr, nullptr, ffn_ctrl_ir ? ir_c0 : nullptr,
                        2 * (int) kv_dim, T, N, hp.rms_eps);
                    k_shortconv_hist_push<<<(unsigned) ((N + 255) / 256), 256, 0, st>>>(
                        wo, hst, N, 2 * (int) kv_dim, T);
                    kick_async_moe_zero();
                    run_control(in_ffn, msg3, L, 3, in_ffn, T, ffn_ctrl_ir ? ir_c0 : nullptr, ffn_ctrl_ir ? ir_m3 : nullptr);
                    k_rms_norm_f16<<<(unsigned) T, rms_f16_blk, 0, st>>>(in_ffn, L.ffn_norm, in_ffn, act_bf16, N, hp.rms_eps);
                }
            } else {
                if (T >= 32 && !save_conv_trace) {
                    if ((N & 3) == 0) {
                        const int n4 = N >> 2;
                        const unsigned sb = 64u;
                        k_shortconv_pp_f4<<<dim3((unsigned) ((n4 + (int) sb - 1) / (int) sb), (unsigned) ((T + 63) / 64)), sb, 0, st>>>(
                            wo, L.attn_post_conv, hst, ffn_inp, N, 2 * (int) kv_dim, T, do_push ? nullptr : cur_in);
                    } else {
                        k_shortconv_pp<<<dim3((unsigned) ((N + 255) / 256), (unsigned) ((T + 15) / 16)), 256, 0, st>>>(
                            wo, L.attn_post_conv, hst, ffn_inp, N, 2 * (int) kv_dim, T, do_push ? nullptr : cur_in);
                    }
                    k_shortconv_hist_push<<<(unsigned) ((N + 255) / 256), 256, 0, st>>>(
                        wo, hst, N, 2 * (int) kv_dim, T);
                    kick_async_moe_zero();
                } else {
                    k_shortconv<<<(unsigned) ((N + 63) / 64), 64, 0, st>>>(wo, L.attn_post_conv, nullptr, hst, ffn_inp, N, 2 * (int) kv_dim, 0, T, do_push ? nullptr : cur_in, h_tr, total_conv_floats);
                }
                if (do_push) {
                    if (pre_push_done_il == il) {
                        norm_blocks.push_back(pre_push_ndst);
                        pre_push_done_il = -1;
                    } else {
                        const int pass_cur = (n_core > 0 && ip >= hp.core_start && ip < hp.core_end)
                                           ? ((il - (int) hp.core_start) / (int) n_core) : 0;
                        push_block(cur_in, il, ip, pass_cur);
                    }
                }

                // MoE
                float* ffn_mix_out[1] = {in_ffn};
                const bool ffn_want_q8 = (T <= 8 && (N & 31) == 0);
                bool ffn_fused_norm = false;
                if (!has_msg && ffn_want_q8) {
                    NormQ8Out3 nq8{L.ffn_norm, nullptr, nullptr, xq, nullptr, nullptr};
                    ffn_fused_norm = try_run_mix_normed_q8(norm_blocks, ffn_inp, L.ffn_res_query, ffn_mix_out, nq8, 1, T);
                } else if (!has_msg && T >= 16 && N == 2304) {
                    NormF16Out3 nf16{L.ffn_norm, nullptr, nullptr, act_bf16, nullptr, nullptr};
                    ffn_fused_norm = try_run_mix_normed_f16(norm_blocks, ffn_inp, L.ffn_res_query, ffn_mix_out, nf16, 1, T);
                }
                if (!ffn_fused_norm) {
                    if (use_vmsg_layer) {
                        MixInvRms4 ir_ffn{{ir_c0, nullptr, nullptr, nullptr}};
                        run_mix_normed(norm_blocks, ffn_inp, false, L.ffn_res_query, ffn_mix_out, 1, T, 0, &ir_ffn);
                        launch_vmsg_ffn_ctrl();
                    } else {
                        const bool ffn_ctrl_ir = (has_msg && has_msg_ir && T >= 32 && N == 2304 &&
                                                  hp.loop_gate_rank == 64 && can_ctrl_wmma && can_ctrl_down_wmma && ir_c0 != nullptr);
                        MixInvRms4 ir_ffn{{ir_c0, nullptr, nullptr, nullptr}};
                        run_mix_normed(norm_blocks, ffn_inp, false, L.ffn_res_query, ffn_mix_out, 1, T, 0, ffn_ctrl_ir ? &ir_ffn : nullptr);
                        if (has_msg) run_control(in_ffn, msg3, L, 3, in_ffn, T, ffn_ctrl_ir ? ir_c0 : nullptr, ffn_ctrl_ir ? ir_m3 : nullptr);
                        if (ffn_want_q8) {
                            k_rms_norm_q8_1<<<(unsigned) T, rms_q8_blk, 0, st>>>(in_ffn, L.ffn_norm, in_ffn, xq, N, hp.rms_eps);
                        } else if (T >= 16 && (N & 3) == 0) {
                            k_rms_norm_f16<<<(unsigned) T, rms_f16_blk, 0, st>>>(in_ffn, L.ffn_norm, in_ffn, act_bf16, N, hp.rms_eps);
                        } else {
                            k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(in_ffn, L.ffn_norm, in_ffn, N, hp.rms_eps);
                        }
                    }
                }
            }
            if (include_route_t1) {
                // Capture h_h D2H + router + route(+group on T==1) + h_ids/h_ewts D2H inside layer_pre_graph/verify_pre_graph!
                const int Kp = (int) hp.n_expert_used;
                const int n_ent = T * Kp;
                CK(cudaMemcpyAsync(h_h, in_ffn, (size_t) T * (size_t) N * sizeof(float), cudaMemcpyDeviceToHost, st));
                if (T == 1 && hp.n_expert <= 128 && Kp <= 16 && (N & 3) == 0 && d_router_done != nullptr) {
                    const bool has_g = L.n_slots > 0;
                    k_router_gemv_and_group_t1<<<(unsigned) ((hp.n_expert + 1) / 2), 128, 0, st>>>(
                        L.ffn_gate_inp, in_ffn, router_logits, L.exp_probs_b, ids, ewts,
                        has_g ? L.d_slot_of : nullptr, has_g ? L.d_blobs : nullptr, L.stride,
                        N, (int) hp.n_expert, Kp, hp.expert_weights_scale,
                        grp_ptr, grp_start, n_groups, ent_dst, ent_tok, L.d_hits, d_router_done);
                } else {
                    project_f32(in_ffn, L.ffn_gate_inp, router_logits, T, hp.n_expert, N);
                    k_route_inkling2<<<(unsigned) ((T + 7) / 8), 256, 0, st>>>(
                        router_logits, L.exp_probs_b, ids, ewts, (int) hp.n_expert, Kp, hp.expert_weights_scale, T, L.d_hits);
                }
                CK(cudaMemcpyAsync(h_ids, ids, (size_t) n_ent * sizeof(int32_t), cudaMemcpyDeviceToHost, st));
                CK(cudaMemcpyAsync(h_ewts, ewts, (size_t) n_ent * sizeof(float), cudaMemcpyDeviceToHost, st));
            }
            if (prof_detail) { CK(cudaStreamSynchronize(st)); p_d_out += ms_since(tp_d3); }
        };

        auto advance_pre_meta = [&](int il) {
            const int ip = hp.loop_layer(il);
            int pass_cur = 0;
            if (n_core > 0 && ip >= hp.core_start && ip < hp.core_end) {
                pass_cur = (il - (int) hp.core_start) / (int) n_core;
                if (ip == hp.core_start) {
                    if (pass_cur == 0) {
                        prelude_norm_blocks = norm_blocks;
                    } else {
                        norm_bank.clear();
                        for (size_t bi = prelude_norm_blocks.size() + 1; bi < norm_blocks.size(); ++bi) norm_bank.push_back(norm_blocks[bi]);
                        norm_blocks = prelude_norm_blocks;
                    }
                }
            }
            if (ip % hp.attn_res_block_size == 0) {
                bool skip_recompute = false;
                const int idx = pick_blk_slot(il, ip, pass_cur, skip_recompute);
                norm_blocks.push_back(norm_blk_buf[idx]);
            }
        };

        auto run_layer_post = [&](int il, bool pre_routed_t1 = false) {
            const int ip = hp.loop_layer(il);
            PhysLayer& L = m->layers[(size_t) ip];
            float* hst = conv_hist[(size_t) il];
            float* h_tr = save_conv_trace ? (d_conv_trace + (size_t) il * layer_conv_floats) : nullptr;
            const bool is_hy = (L.n_slots < hp.n_expert);
            const bool xq_ffn_ready = (T <= 8 && (N & 31) == 0);
            const bool f16_ffn_ready = (T >= 16 && (N & 3) == 0);
            const bool fuse_comb_conv = (!is_hy && T <= std::max(verify_max_t, 8));
            const PostConvSpec hy_conv{L.ffn_post_conv, hst, ffn_inp, cur, 2 * (int) kv_dim + N, h_tr, total_conv_floats};
            const bool want_pp_comb_conv = (T >= 32 && !save_conv_trace && (N & 3) == 0 && (!is_hy || pp_streaming));
            static const bool no_vert_fuse = (std::getenv("STRATA_NO_VERT_FUSE") != nullptr);
            static const bool no_moe_atomic_post = (std::getenv("STRATA_NO_MOE_ATOMIC") != nullptr);
            const bool will_use_atomic_moe = (!no_moe_atomic_post && cc_maj >= 8 && T >= 32 && (N & 3) == 0 && !fuse_comb_conv &&
                                              L.gu_type != 30 && L.d_type != 30 && K::native_expert_wmma_supported(L.L) &&
                                              (!is_hy || (pp_streaming && L.hblobs && pp_stage_bytes >= L.stride && pp_layer_cnt[(size_t) il] > 0)));
            bool can_defer_postffn = false;
            if (!no_vert_fuse && will_use_atomic_moe && N == 2304 && !save_conv_trace &&
                il + 1 < hp.n_layer && (int) norm_blocks.size() <= 7 &&
                reinterpret_cast<uint8_t*>(moe_raw) >= reinterpret_cast<uint8_t*>(in_q) + 3 * (size_t) T * (size_t) N * sizeof(float) &&
                reinterpret_cast<uint8_t*>(moe_raw) + (size_t) T * (size_t) N * sizeof(float) <= reinterpret_cast<uint8_t*>(Q)) {
                const int ip_next = hp.loop_layer(il + 1);
                const int pass_next = (n_core > 0 && ip_next >= hp.core_start && ip_next < hp.core_end)
                                    ? ((il + 1 - (int) hp.core_start) / (int) n_core) : 0;
                const bool is_pass_reset_next = (n_core > 0 && ip_next == hp.core_start && pass_next > 0);
                const PhysLayer& L_next = m->layers[(size_t) ip_next];
                if (!is_pass_reset_next && L_next.attn_res_query != nullptr && act_bf16 != nullptr) {
                    const bool has_msg_next = (n_core > 0 && ip_next >= hp.core_start && ip_next < hp.core_end && pass_next > 0);
                    if (!has_msg_next) {
                        can_defer_postffn = true;
                    } else if (vmsg_pass_ready) {
                        const bool is_swa_next = hp.is_swa(il + 1);
                        const bool attn_f16_ok_next = (pre_attn_f16_v != nullptr &&
                            L_next.attn_q.type != 0 && L_next.attn_k.type != 0 && L_next.attn_v.type != 0 &&
                            L_next.attn_gate.type != 0 && L_next.attn_rel.type != 0 && L_next.attn_sink_proj.type != 0 &&
                            (bf16_x_f16 || (L_next.attn_q.type != 30 && L_next.attn_k.type != 30 && L_next.attn_v.type != 30 &&
                                            L_next.attn_gate.type != 30 && L_next.attn_rel.type != 30)) &&
                            (is_swa_next || (L_next.attn_k_conv_proj.type != 0 && L_next.attn_v_conv_proj.type != 0)));
                        const bool ctrl_ok_next =
                            L_next.loop_down[0].type == 0 && L_next.loop_down[1].type == 0 && L_next.loop_down[2].type == 0 && L_next.loop_down[3].type == 0 &&
                            L_next.loop_gain[0].type == 0 && L_next.loop_gain[1].type == 0 && L_next.loop_gain[2].type == 0 && L_next.loop_gain[3].type == 0 &&
                            L_next.loop_forget[0].type == 0 && L_next.loop_forget[1].type == 0 && L_next.loop_forget[2].type == 0 && L_next.loop_forget[3].type == 0;
                        can_defer_postffn = (attn_f16_ok_next && ctrl_ok_next);
                    }
                }
            }
            const bool post_conv_done = run_moe(L, il, in_ffn, T, pre_routed_t1, xq_ffn_ready, fuse_comb_conv,
                                                f16_ffn_ready,
                                                can_defer_postffn ? nullptr : (((is_hy && T <= std::max(verify_max_t, 8) && !pp_streaming) || want_pp_comb_conv) ? &hy_conv : nullptr),
                                                pre_routed_pp_cur, capturing_full_graph);
            pre_routed_pp_cur = false;
            const auto tp_mp0 = Clock::now();
            if (post_conv_done) {
                // Already fused k_combine_experts[_hybrid]_shortconv directly into cur!
            } else if (can_defer_postffn) {
                deferred_postffn = DeferredPostFfn{true, il, L.ffn_post_conv, hst, 2 * (int) kv_dim + N};
            } else if (fuse_comb_conv) {
                k_combine_experts_shortconv<<<(unsigned) ((N + 31) / 32), 32, 0, st>>>(
                    eparts, ewts, L.ffn_post_conv, hst, ffn_inp, cur, N, (int) hp.n_expert_used,
                    2 * (int) kv_dim + N, T, h_tr, total_conv_floats);
            } else if (T >= 32 && !save_conv_trace) {
                k_shortconv_pp<<<dim3((unsigned) ((N + 255) / 256), (unsigned) ((T + 15) / 16)), 256, 0, st>>>(
                    moe_raw, L.ffn_post_conv, hst, cur, N, 2 * (int) kv_dim + N, T, ffn_inp);
                k_shortconv_hist_push<<<(unsigned) ((N + 255) / 256), 256, 0, st>>>(
                    moe_raw, hst, N, 2 * (int) kv_dim + N, T);
            } else {
                k_shortconv<<<(unsigned) ((N + 255) / 256), 256, 0, st>>>(moe_raw, L.ffn_post_conv, nullptr, hst, cur, N, 2 * (int) kv_dim + N, 0, T, ffn_inp, h_tr, total_conv_floats);
            }
            if (prof) {
                CK(cudaStreamSynchronize(st));
                if (is_hy) p_moe_post_ms += ms_since(tp_mp0);
                else if (ip >= hp.core_start && ip < hp.core_end) p_core_ms += ms_since(tp_mp0);
            }
        };

        const int core_end_virt = (int) (hp.core_start + hp.num_loops * n_core);
        for (int il = 0; il < hp.n_layer; ++il) {
            const bool use_t1_core = (!capturing_full_graph && !prof && T == 1 && !capture_dspark);
            const bool use_vf_core = (!capturing_full_graph && !prof && T >= 1 && T <= std::min(verify_max_t, 9) &&
                                      capture_dspark && ((T == 1 && !save_conv_trace) || (T >= 2 && save_conv_trace)));
            if ((use_t1_core || use_vf_core) && core_all_gpu && !prof_detail && il == hp.core_start && n_core > 0) {
                // EX-5 Idea D: Drain any pending async cst rebal events on core layers BEFORE entering stream capture!
                for (int cip = (int) hp.core_start; cip < (int) hp.core_end && cip < kMaxRebalLayers; ++cip) {
                    if (layer_pending_rebal[cip]) {
                        CK(cudaStreamWaitEvent(st, ev_rebal_layer[cip], 0));
                        layer_pending_rebal[cip] = false;
                    }
                }
                cudaGraphExec_t& cg = use_t1_core ? core_graph : verify_core_graph[T];
                if (!cg && !graph_oom) {
                    const auto saved_nb = norm_blocks;
                    const int saved_next = next_blk_buf;
                    const auto saved_pnb = prelude_norm_blocks;
                    const auto saved_bank = norm_bank;
                    cudaGraph_t g = nullptr;
                    cudaError_t ce = cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
                    if (ce == cudaSuccess) {
                        for (int cl = il; cl < core_end_virt; ++cl) {
                            run_layer_pre(cl, false);
                            run_layer_post(cl, false);
                        }
                        ce = cudaStreamEndCapture(st, &g);
                    }
                    cudaGraphExec_t ge = nullptr;
                    if (ce == cudaSuccess && g) ce = cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0);
                    if (g) cudaGraphDestroy(g);
                    if (ce == cudaSuccess && ge) {
                        cg = ge;
                    } else {
                        (void) cudaGetLastError();
                        graph_oom = true;
                        norm_blocks = saved_nb;
                        next_blk_buf = saved_next;
                        prelude_norm_blocks = saved_pnb;
                        norm_bank = saved_bank;
                    }
                } else if (cg) {
                    if (norm_blk_cnt == 5) {
                        norm_blocks = {norm_blk_buf[0], norm_blk_buf[1], norm_blk_buf[2]};
                        next_blk_buf = 3;
                    } else {
                        norm_blocks = {norm_blk_buf[0], norm_blk_buf[5], norm_blk_buf[6]};
                        next_blk_buf = 7;
                    }
                }
                if (cg) {
                    const auto tp_cg0 = Clock::now();
                    CK(cudaGraphLaunch(cg, st));
                    if (prof) { CK(cudaStreamSynchronize(st)); p_core_ms += ms_since(tp_cg0); }
                    il = core_end_virt - 1;
                    continue;
                }
            }
            const bool is_outer = (il < hp.core_start || il >= core_end_virt);
            const bool use_t1_pre_graph = (T == 1 && hybrid && !prof_detail && !pp_streaming && !capture_dspark);
            // Capture verify_pre_graph[T] up to T=9 by default (override via STRATA_VERIFY_PRE_GRAPH_MAX_T).
            static const int vf_pre_max_t = [] {
                const char* v = std::getenv("STRATA_VERIFY_PRE_GRAPH_MAX_T");
                return (v && v[0]) ? std::clamp(std::atoi(v), 1, 9) : 9;
            }();
            const bool use_vf_pre_graph = (T >= 1 && T <= vf_pre_max_t && T <= verify_max_t && hybrid &&
                                           (is_outer || !core_all_gpu) && !prof_detail && !pp_streaming &&
                                           capture_dspark && ((T == 1 && !save_conv_trace) || (T >= 2 && save_conv_trace)));
            const bool use_pre_graph = use_t1_pre_graph || use_vf_pre_graph;
            const int ip_cur = hp.loop_layer(il);
            const bool include_route_t1 = use_pre_graph && (m->layers[(size_t) ip_cur].n_slots < hp.n_expert);
            // EX-5 Idea D: Wait on any pending async cst rebal for this layer BEFORE cudaStreamBeginCapture!
            if (m->layers[(size_t) ip_cur].n_slots > 0 &&
                ip_cur < kMaxRebalLayers && layer_pending_rebal[ip_cur]) {
                CK(cudaStreamWaitEvent(st, ev_rebal_layer[ip_cur], 0));
                layer_pending_rebal[ip_cur] = false;
            }
            auto& pre_vec = use_t1_pre_graph ? layer_pre_graph : verify_pre_graph[T];
            const auto tp_ly0 = Clock::now();
            if (use_pre_graph && (pre_vec[(size_t) il] || !graph_oom)) {
                if (!pre_vec[(size_t) il]) {
                    const auto saved_nb = norm_blocks;
                    const int saved_next = next_blk_buf;
                    const auto saved_pnb = prelude_norm_blocks;
                    const auto saved_bank = norm_bank;
                    cudaGraph_t g = nullptr;
                    cudaError_t ce = cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
                    if (ce == cudaSuccess) {
                        run_layer_pre(il, include_route_t1);
                        ce = cudaStreamEndCapture(st, &g);
                    }
                    cudaGraphExec_t ge = nullptr;
                    if (ce == cudaSuccess && g) ce = cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0);
                    if (g) cudaGraphDestroy(g);
                    if (ce == cudaSuccess && ge) {
                        pre_vec[(size_t) il] = ge;
                        CK(cudaGraphLaunch(pre_vec[(size_t) il], st));
                    } else {
                        (void) cudaGetLastError();
                        graph_oom = true;
                        norm_blocks = saved_nb;
                        next_blk_buf = saved_next;
                        prelude_norm_blocks = saved_pnb;
                        norm_bank = saved_bank;
                        run_layer_pre(il, include_route_t1);
                    }
                } else {
                    advance_pre_meta(il);
                    CK(cudaGraphLaunch(pre_vec[(size_t) il], st));
                }
            } else {
                run_layer_pre(il, include_route_t1);
            }
            if (prof) {
                CK(cudaStreamSynchronize(st));
                if (is_outer) p_attn_ms += ms_since(tp_ly0);
                else p_core_ms += ms_since(tp_ly0);
            }
            wait_dspark_acc();
            run_layer_post(il, include_route_t1);
            // EX-5 Idea A: On the final prefill chunk before set_phase_decode(), reclaim finished Prelude
            // d_blobs as transient pp_stage slots so cst pre-buffers Epilogue experts during the 36-layer Core!
            if (allow_prelude_borrow && pp_streaming && core_all_gpu && il < hp.core_start && pp_stage_bytes > 0) {
                const int ip_don = hp.loop_layer(il);
                PhysLayer& Ld = m->layers[(size_t) ip_don];
                const size_t slot_stride = (pp_stage_bytes + 255u) & ~size_t(255u);
                if (Ld.hblobs && Ld.d_blobs && Ld.n_slots > 0 && ip_don < kMaxRebalLayers && slot_stride > 0) {
                    const size_t donor_bytes = (size_t) Ld.n_slots * Ld.stride;
                    const int n_full = (int) (donor_bytes / slot_stride);
                    size_t min_rem_task_b = 0;
                    for (size_t ti = pp_kick_idx; ti < pp_tasks.size(); ++ti) {
                        const PpTask& rt = pp_tasks[ti];
                        const size_t tb = (size_t) rt.n_staged * m->layers[(size_t) rt.ip].stride;
                        if (tb > 0 && (min_rem_task_b == 0 || tb < min_rem_task_b)) min_rem_task_b = tb;
                    }
                    size_t carve_stride = slot_stride;
                    int n_carve = n_full;
                    if (min_rem_task_b > 0) {
                        const size_t small_stride = (min_rem_task_b + 255u) & ~size_t(255u);
                        if (small_stride > 0 && small_stride < slot_stride) {
                            const int n_small = (int) (donor_bytes / small_stride);
                            if (n_small > n_full) {
                                int n_fit_small = 0;
                                int n_need_large = 0;
                                for (size_t ti = pp_kick_idx; ti < pp_tasks.size(); ++ti) {
                                    const PpTask& rt = pp_tasks[ti];
                                    const size_t tb = (size_t) rt.n_staged * m->layers[(size_t) rt.ip].stride;
                                    if (tb > 0 && tb <= small_stride) ++n_fit_small;
                                    else if (tb > small_stride) ++n_need_large;
                                }
                                if (n_fit_small >= n_small && (n_full == 0 || n_need_large <= 8)) {
                                    carve_stride = small_stride;
                                    n_carve = n_small;
                                }
                            }
                        }
                    }
                    for (int si = 0; si < n_carve && pp_active_slots < kMaxPpStages; ++si) {
                        const int b = pp_active_slots++;
                        pp_buf_ptr[b] = Ld.d_blobs + (size_t) si * carve_stride;
                        pp_buf_cap[b] = carve_stride;
                        pp_buf_owner_ip[b] = ip_don;
                        CK(cudaEventRecord(ev_pp_used[b], st));
                        pp_slot_needs_wait[b] = true;
                        pp_free_slots.push_back(b);
                    }
                    if (n_carve > 0) pp_kick_next();
                }
            }
            // Wave 17: After blk.6's final loop pass (il == 30 on Inkling-2), borrow 3 x 162.3 MiB full slots in-place
            // from blk.6.d_blobs on CoreResidentHybrid (both borrow=0 and borrow=1) so cst pre-buffers Epilogue L44..47
            // with zero PCIe bubbles; BorrowCleanup restores blk.6 asynchronously on cst with 0 ms prefill stall.
            static const bool no_core_borrow = [] {
                const char* v = std::getenv("STRATA_NO_CORE_BORROW");
                return v != nullptr && (v[0] == '\0' || std::atoi(v) != 0);
            }();
            const int il_core_final0 = (int) (hp.core_start + (hp.num_loops - 1) * n_core);
            if (!no_core_borrow && pp_streaming && core_all_gpu &&
                profile.regime != strata::inkling2::VramRegime::FullVram &&
                hp.num_loops > 1 && n_core > 0 && pp_stage_bytes > 0 &&
                il >= il_core_final0 && il < il_core_final0 + 2 &&
                core_borrowed_cnt < kMaxCoreBorrow &&
                pp_kick_idx < pp_tasks.size()) {
                const int ip_core = (int) hp.core_start + (il - il_core_final0);
                PhysLayer& Lc = m->layers[(size_t) ip_core];
                const size_t slot_stride = (pp_stage_bytes + 255u) & ~size_t(255u);
                bool id_map = (Lc.hblobs != nullptr && Lc.hblobs_pinned && Lc.d_blobs != nullptr &&
                               Lc.n_slots >= hp.n_expert && ip_core < kMaxRebalLayers &&
                               slot_stride > 0 && (int) Lc.slot_of.size() >= hp.n_expert);
                for (int e = 0; id_map && e < hp.n_expert; ++e) {
                    if (Lc.slot_of[(size_t) e] != e) id_map = false;
                }
                if (id_map) {
                    const size_t donor_bytes = (size_t) Lc.n_slots * Lc.stride;
                    const int n_carve = std::min((int) (donor_bytes / slot_stride), kMaxPpStages - pp_active_slots);
                    if (n_carve > 0) {
                        core_borrowed_ips[core_borrowed_cnt] = ip_core;
                        core_borrowed_bytes[core_borrowed_cnt] = std::min(donor_bytes, (size_t) n_carve * slot_stride);
                        ++core_borrowed_cnt;
                        for (int si = 0; si < n_carve; ++si) {
                            const int b = pp_active_slots++;
                            pp_buf_ptr[b] = Lc.d_blobs + (size_t) si * slot_stride;
                            pp_buf_cap[b] = pp_stage_bytes;
                            pp_buf_owner_ip[b] = ip_core;
                            CK(cudaEventRecord(ev_pp_used[b], st));
                            pp_slot_needs_wait[b] = true;
                            pp_free_slots.push_back(b);
                        }
                        pp_kick_next();
                    }
                }
            }
            if (pp_streaming && T >= 256 && il == hp.core_start - 1) {
                const int borrowed_cnt = std::max(0, pp_active_slots - pp_stage_cnt - transient_pp_cnt);
                if (pos == 0 || last_pp_log_slots != pp_active_slots || last_pp_log_cap != swa_pp_cap) {
                    last_pp_log_slots = pp_active_slots;
                    last_pp_log_cap = swa_pp_cap;
                    std::fprintf(stderr, "strata-inkling2: pp_stream T=%d swa_ring=%d swa_pp=%d pp_slots=%d (perm=%d trans=%d borrow=%d x %.1f MiB)\n",
                                 T, swa_kv_cap, swa_pp_cap, pp_active_slots, pp_stage_cnt, transient_pp_cnt, borrowed_cnt,
                                 (double) pp_stage_bytes / (1 << 20));
                }
            }
        }

        // Output head (skip 200K vocab matmul + sync on intermediate prefill chunks where only KV/DSpark states are needed)
        const auto tp_h0 = Clock::now();
        const bool dspark_vram_acc_on = (capture_dspark && dspark && dspark->prefill_vram_acc_active);
        const bool need_all_head_rows = (rows == T) || (capture_dspark && dspark && (dspark_vram_acc_on || d_tgt_feat[3]));
        const bool last_row_only = (compute_logits && !need_all_head_rows && T > 1);
        const bool fuse_head_q8 = (compute_logits && (rows == T || last_row_only) && rows <= 8 && (N & 31) == 0);
        const float* head_h = (rows == T) ? cur : (cur + (size_t) (T - 1) * N);
        if (compute_logits || (capture_dspark && dspark && (dspark_vram_acc_on || d_tgt_feat[3]))) {
            if (last_row_only) {
                // Only mix+norm the single final token on the last prefill chunk (avoids ~132 MB dead HBM reads!)
                const size_t row_off = (size_t) (T - 1) * (size_t) N;
                std::vector<float*> nb_last(norm_blocks.size());
                for (size_t bi = 0; bi < norm_blocks.size(); ++bi) nb_last[bi] = norm_blocks[bi] + row_off;
                float* cur_last = cur + row_off;
                float* out_head[1] = {cur_last};
                NormQ8Out3 nq8_h{m->output_norm, nullptr, nullptr, fuse_head_q8 ? xq : nullptr, nullptr, nullptr};
                if (!try_run_mix_normed_q8(nb_last, cur_last, m->output_res_query, out_head, nq8_h, 1, 1)) {
                    float* out_tmp[1] = {in_ffn};
                    run_mix_normed(nb_last, cur_last, false, m->output_res_query, out_tmp, 1, 1, 0);
                    if (fuse_head_q8) {
                        k_rms_norm_q8_1<<<1, (N == 2304) ? 576 : kThreads, 0, st>>>(in_ffn, m->output_norm, cur_last, xq, N, hp.rms_eps);
                    } else {
                        k_rms_norm<<<1, kThreads, 0, st>>>(in_ffn, m->output_norm, cur_last, N, hp.rms_eps);
                    }
                }
                head_h = cur_last;
            } else {
                float* out_head[1] = {cur};
                NormQ8Out3 nq8_h{m->output_norm, nullptr, nullptr, fuse_head_q8 ? xq : nullptr, nullptr, nullptr};
                if (!try_run_mix_normed_q8(norm_blocks, cur, m->output_res_query, out_head, nq8_h, 1, T)) {
                    float* out_tmp[1] = {in_ffn};
                    run_mix_normed(norm_blocks, cur, false, m->output_res_query, out_tmp, 1, T, 0);
                    if (fuse_head_q8) {
                        k_rms_norm_q8_1<<<(unsigned) T, (N == 2304) ? 576 : kThreads, 0, st>>>(in_ffn, m->output_norm, cur, xq, N, hp.rms_eps);
                    } else {
                        k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(in_ffn, m->output_norm, cur, N, hp.rms_eps);
                    }
                }
                if (capture_dspark && dspark && !dspark_vram_acc_on && d_tgt_feat[3]) {
                    CK(cudaMemcpyAsync(d_tgt_feat[3], cur, (size_t) T * N * sizeof(float), cudaMemcpyDeviceToDevice, st));
                }
            }
        }
        wait_dspark_acc();
        // Wave 5: Overlap DSpark inject_from_vram_acc() directly on st BEFORE forward()'s final sync!
        const int64_t tot_lg = (int64_t) rows * hp.vocab_unpadded;
        const bool overlap_head_dspark = (dspark_vram_acc_on && compute_logits && rows == 1 &&
                                          !capturing_full_graph && !prof && !prof_detail &&
                                          aux_st != nullptr && ev_dspark_acc_src && ev_dspark_acc_done &&
                                          dspark->can_inject_all_f16(*this, T) &&
                                          m->output_weight.type == 8 && !m->output_weight.q8_in_place &&
                                          hp.vocab_unpadded > 3072 && (N % 256) == 0 &&
                                          ((((uintptr_t) m->output_weight.d) & 3u) == 0u));
        if (overlap_head_dspark) {
            CK(cudaEventRecord(ev_dspark_acc_src, st));
            CK(cudaStreamWaitEvent(aux_st, ev_dspark_acc_src, 0));
            K::native_quantize_q8_1(head_h, xq, (int) N, 1, aux_st);
            k_mmvq_q8_0_smallT<1, true, false, 4><<<(unsigned) ((hp.vocab_unpadded + 3) / 4), 128, 0, aux_st>>>(
                (const uint8_t*) m->output_weight.d, (const uint32_t*) xq, logits, (int) N, (int) hp.vocab_unpadded, 1);
            if (want_softcap) {
                k_softcap<<<(unsigned) ((tot_lg + 255) / 256), 256, 0, aux_st>>>(logits, tot_lg, hp.softcap);
            }
            if (want_argmax) {
                if (hp.vocab_unpadded >= 4096 && argmax_val && argmax_idx && rows <= verify_max_t) {
                    constexpr int kArgBlks = 64;
                    k_argmax_stage1<<<dim3(kArgBlks, 1u), 256, 0, aux_st>>>(
                        logits, argmax_val, argmax_idx, (int) hp.vocab_unpadded);
                    k_argmax_stage2<<<1u, 32, 0, aux_st>>>(
                        argmax_val, argmax_idx, pick_d, kArgBlks);
                } else {
                    k_argmax<<<1u, 256, 0, aux_st>>>(logits, pick_d, (int) hp.vocab_unpadded);
                }
            }
            CK(cudaEventRecord(ev_dspark_acc_done, aux_st));
            dspark->inject_from_vram_acc(*this, cur, T, (int) pos, st);
            dspark_injected_in_fwd = true;
            CK(cudaStreamWaitEvent(st, ev_dspark_acc_done, 0));
        } else if (dspark_vram_acc_on) {
            dspark->inject_from_vram_acc(*this, cur, T, (int) pos, st);
            dspark_injected_in_fwd = true;
        }
        if (!compute_logits) {
            const bool queued_hits = pp_streaming ? queue_sync_d_hits() : false;
            if (T <= kDecodePhaseCap) flush_deferred_graphs();
            CK(cudaStreamSynchronize(st));
            if (queued_hits) finish_sync_d_hits(true);
            if (prof) p_head_ms += ms_since(tp_h0);
            pos += T;
            return 0;
        }
        if (!overlap_head_dspark) {
            project_preq(head_h, fuse_head_q8 ? xq : nullptr, m->output_weight, logits, rows, hp.vocab_unpadded, N);
            if (want_softcap) {
                k_softcap<<<(unsigned) ((tot_lg + 255) / 256), 256, 0, st>>>(logits, tot_lg, hp.softcap);
            }
            if (want_argmax) {
                if (hp.vocab_unpadded >= 4096 && argmax_val && argmax_idx && rows <= verify_max_t) {
                    constexpr int kArgBlks = 64;
                    k_argmax_stage1<<<dim3(kArgBlks, (unsigned) rows), 256, 0, st>>>(
                        logits, argmax_val, argmax_idx, (int) hp.vocab_unpadded);
                    k_argmax_stage2<<<(unsigned) rows, 32, 0, st>>>(
                        argmax_val, argmax_idx, pick_d, kArgBlks);
                } else {
                    k_argmax<<<(unsigned) rows, 256, 0, st>>>(logits, pick_d, (int) hp.vocab_unpadded);
                }
            }
        }
        if (capturing_full_graph) {
            if (want_argmax) {
                CK(cudaMemcpyAsync(h_pick_pin, pick_d, (size_t) rows * sizeof(int32_t), cudaMemcpyDeviceToHost, st));
            }
            cudaGraph_t g = nullptr;
            cudaError_t ce = cudaStreamEndCapture(st, &g);
            cudaGraphExec_t ge = nullptr;
            if (ce == cudaSuccess && g) ce = cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0);
            if (g) cudaGraphDestroy(g);
            if (ce == cudaSuccess && ge) {
                *fg_slot = ge;
                CK(cudaGraphLaunch(ge, st));
                int32_t last_pick = 0;
                if (want_argmax) {
                    CK(cudaStreamSynchronize(st));
                    if (out_picks) out_picks->assign(h_pick_pin, h_pick_pin + rows);
                    last_pick = h_pick_pin[rows - 1];
                }
                pos += T;
                return last_pick;
            }
            (void) cudaGetLastError();
            graph_oom = true;
            // Fallback: re-run uncaptured cleanly for this call
            return forward(tokens, T, copy_logits, out_logits, all_row_logits, capture_dspark, verify_k_save,
                           out_picks, need_logits, defer_logits_d2h);
        }
        if (copy_logits && defer_logits_d2h && !want_argmax) {
            if (T <= kDecodePhaseCap) flush_deferred_graphs();
            if (pp_streaming) {
                const bool queued_hits = queue_sync_d_hits();
                CK(cudaStreamSynchronize(st));
                if (queued_hits) finish_sync_d_hits(true);
            }
            if (prof) p_head_ms += ms_since(tp_h0);
            pos += T;
            return 0;
        }
        if (copy_logits && out_logits && !defer_logits_d2h) {
            out_logits->resize((size_t) tot_lg);
            CK(cudaMemcpyAsync(out_logits->data(), logits, (size_t) tot_lg * sizeof(float), cudaMemcpyDeviceToHost, st));
        }
        const bool queued_hits = pp_streaming ? queue_sync_d_hits() : false;
        int32_t last_pick = 0;
        if (want_argmax) {
            int32_t* pbuf = (h_pick_pin && rows <= verify_max_t) ? h_pick_pin : nullptr;
            std::vector<int32_t> tmp_picks;
            if (!pbuf) { tmp_picks.assign((size_t) rows, 0); pbuf = tmp_picks.data(); }
            CK(cudaMemcpyAsync(pbuf, pick_d, (size_t) rows * sizeof(int32_t), cudaMemcpyDeviceToHost, st));
            if (T <= kDecodePhaseCap) flush_deferred_graphs();
            CK(cudaStreamSynchronize(st));
            if (out_picks) out_picks->assign(pbuf, pbuf + rows);
            last_pick = pbuf[rows - 1];
        } else {
            if (T <= kDecodePhaseCap) flush_deferred_graphs();
            CK(cudaStreamSynchronize(st));
        }
        if (queued_hits) finish_sync_d_hits(true);
        if (prof) p_head_ms += ms_since(tp_h0);
        pos += T;
        return last_pick;
    }
};

bool place_experts(Model& m, Engine& E, double vram_gib, int threads, std::string& err) {
    const auto t0 = Clock::now();
    const auto& hp = m.hp;
    const int ne = (int) hp.n_expert;
    size_t fr_pre = 0, tot = 0;
    CK(cudaMemGetInfo(&fr_pre, &tot));
    // Wave 8 Fix 0 (F5): Warm up cuBLAS (f16, bf16, sgemm) BEFORE budgeting VRAM so its lazy ~80 MiB
    // cubin load is measured upfront (prevents CUBLAS_STATUS_NOT_SUPPORTED (15) on explicit --chunk 8192
    // while keeping exact 4K Tier-2 expert count by deducting cublas_used from base_reserve down to 64 MiB).
    if (E.act_bf16 && E.cur) {
        CK(cudaMemsetAsync(E.act_bf16, 0, 16 * 64 * sizeof(uint16_t), E.st));
        E.gemm.f16(E.act_bf16, E.act_bf16, E.cur, 16, 64, 64);
        if (!E.bf16_x_f16) E.gemm.bf16(E.act_bf16, E.act_bf16, E.cur, 16, 64, 64);
        if (E.in_q) {
            CK(cudaMemsetAsync(E.in_q, 0, 16 * 64 * sizeof(float), E.st));
            E.gemm.sgemm(E.in_q, E.in_q, E.cur, 16, 64, 64);
        }
        CK(cudaStreamSynchronize(E.st));
    }
    size_t fr = 0;
    CK(cudaMemGetInfo(&fr, &tot));
    const size_t cublas_used = (fr_pre > fr) ? (fr_pre - fr) : 0;
    const bool core_res_hy = (E.profile.regime == strata::inkling2::VramRegime::CoreResidentHybrid);
    // DSpark adds dflare_graph + markov_graphs + verify_core_graph; keep 288 MiB headroom when enabled.
    const size_t min_resv = (size_t) 64 << 20;
    const size_t raw_base_reserve = core_res_hy
        ? (E.verify_max_t > 1 ? ((size_t) 176 << 20) : ((size_t) 144 << 20))
        : ((size_t) 256 << 20);
    const size_t base_reserve = (raw_base_reserve > cublas_used + min_resv)
        ? (raw_base_reserve - cublas_used)
        : min_resv;
    const size_t reserve = base_reserve + E.vram_reserve_bytes;
    size_t budget = vram_gib >= 0.0 ? (size_t) (vram_gib * (double) (1ull << 30))
                                    : (fr > reserve ? fr - reserve : 0);

    size_t stage_mb = (size_t) (E.profile.pp_stage_mb > 0 ? E.profile.pp_stage_mb : 64);
    const bool env_stage_mb = (std::getenv("STRATA_PP_STAGE_MB") != nullptr);
    const bool env_stage_cnt = (std::getenv("STRATA_PP_STAGE_CNT") != nullptr);
    if (const char* v = std::getenv("STRATA_PP_STAGE_MB")) stage_mb = (size_t) std::max(0, std::atoi(v));
    size_t max_exp_stride = 0;
    size_t all_exps_bytes = 0;
    size_t tier1_full_pre = 0;
    for (int ip = 0; ip < hp.n_phys; ++ip) {
        max_exp_stride = std::max(max_exp_stride, m.layers[(size_t) ip].stride);
        const size_t lb = (size_t) ne * m.layers[(size_t) ip].stride;
        all_exps_bytes += lb;
        if (m.layers[(size_t) ip].gu_type != 30 && ip >= hp.core_start && ip < hp.core_end) tier1_full_pre += lb;
    }
    // Keep extra headroom when hybrid-offloading so CUDA graphs + desktop fit OOM-free
    // (CoreResidentHybrid pre-allocates full KV & uses 1 compact core_graph, so 160 MiB reserve leaves ~135 MiB live free).
    if (vram_gib < 0.0 && budget < all_exps_bytes) {
        const size_t extra_resv = core_res_hy ? ((size_t) 16 << 20) : ((size_t) 32 << 20);
        budget = budget > extra_resv ? (budget - extra_resv) : 0;
    }
    if (budget < all_exps_bytes && stage_mb > 0 && max_exp_stride > 0) {
        int pref_cnt = std::clamp(E.profile.pp_stage_cnt_pref > 0 ? E.profile.pp_stage_cnt_pref : 3, 2, Engine::kMaxPpStages);
        if (const char* v = std::getenv("STRATA_PP_STAGE_CNT")) pref_cnt = std::clamp(std::atoi(v), 2, Engine::kMaxPpStages);
        // Wave 7 + Wave 8.5: At chunk >= 8192 on sm_89 CoreResidentHybrid, pp_stage is freed before Decode CUDA graphs
        // and shrunk before KV growth, so reclaim reserve above the 64 MiB Prefill safety margin (+ cached phase_pool free)
        // for 4-stage Epilogue pre-buffering (up to 4x162 MiB matching adapt_best, never dropping to 2x64 MiB!).
        size_t stage_slack = 0;
        const size_t min_stage_resv_pad = (E.tmax >= 8192) ? ((size_t) 112 << 20) : ((size_t) 64 << 20);
        if (vram_gib < 0.0 && core_res_hy && !env_stage_mb && !env_stage_cnt &&
            E.profile.arch.id == strata::inkling2::ArchId::Sm89Ada && E.tmax >= 8192 && tier1_full_pre > 0) {
            const size_t min_prefill_resv = E.vram_reserve_bytes + min_stage_resv_pad;
            if (fr > budget + min_prefill_resv) {
                stage_slack = fr - budget - min_prefill_resv;
            }
            stage_slack += E.pool_cached_free_bytes();
        }
        const size_t stage_budget = budget + stage_slack;
        // EX-4 + Wave 8.5: On sm_89 Hybrid with chunk >= 2048, opportunistically upgrade staging to 4x162.3 MiB
        // (cuts outer-layer MoE chunk splits 5->3 and pre-buffers 1.4 full Epilogue layers during Core!).
        // At chunk >= 8192 CoreResidentHybrid, check actual 4x162.3 MiB (649 MiB) / 4x128 / 4x96 / 4x64 budgets first
        // so explicit --chunk 8192 never falls back to 2x64 MiB while keeping Tier-1 128/128 resident!
        if (!env_stage_mb && !env_stage_cnt &&
            E.profile.arch.id == strata::inkling2::ArchId::Sm89Ada && E.tmax >= 2048 && tier1_full_pre > 0) {
            if (E.tmax >= 8192 && core_res_hy) {
                if (stage_budget > tier1_full_pre + ((size_t) 656 << 20) + ((size_t) 32 << 20)) {
                    stage_mb = 192; // snaps to 38 * nvfp4_stride (~162.3 MiB, 4x = 649 MiB, 5 chunks/Q5_K layer!)
                    pref_cnt = 4;
                } else if (stage_budget > tier1_full_pre + ((size_t) 512 << 20) + ((size_t) 32 << 20)) {
                    stage_mb = 128;
                    pref_cnt = 4;
                } else if (stage_budget > tier1_full_pre + ((size_t) 384 << 20) + ((size_t) 32 << 20)) {
                    stage_mb = 96;
                    pref_cnt = 4;
                } else if (stage_budget > tier1_full_pre + ((size_t) 256 << 20) + ((size_t) 32 << 20)) {
                    stage_mb = 64;
                    pref_cnt = 4;
                } else if (stage_budget > tier1_full_pre + ((size_t) 192 << 20) + ((size_t) 32 << 20)) {
                    stage_mb = 64;
                    pref_cnt = 3;
                }
            } else {
                if (stage_budget > tier1_full_pre + ((size_t) 768 << 20) + ((size_t) 512 << 20)) {
                    stage_mb = 192;
                    pref_cnt = 4;
                } else if (stage_budget > tier1_full_pre + ((size_t) 512 << 20) + ((size_t) 256 << 20)) {
                    stage_mb = 128;
                    pref_cnt = 4;
                } else if (budget > tier1_full_pre + ((size_t) 256 << 20) + ((size_t) 192 << 20)) {
                    stage_mb = 128;
                    pref_cnt = 2;
                }
            }
        }
        size_t req_stage = stage_mb << 20;
        if (req_stage < max_exp_stride) req_stage = max_exp_stride;
        for (int ip = 0; ip < hp.n_phys; ++ip) {
            const auto& L = m.layers[(size_t) ip];
            if (L.gu_type == 40 && L.stride > 0) {
                if (stage_mb == 192 && !env_stage_mb) {
                    // Wave 9: With +630 MiB freed by Shared-Scratch SWA KV Pool, restore 38 * nvfp4_stride (162.3 MiB -> 4x = 649 MiB):
                    // cuts Q5_K streaming waves 6->5 per layer while still leaving +529 MiB for resident Prelude + BorrowCleanup Epilogue prebuf!
                    req_stage = std::max(max_exp_stride, (size_t) 38 * L.stride);
                } else {
                    // Snap up by <= 256 KiB to the next exact multiple of NVFP4 L.stride (e.g. 64 MiB -> 15 * stride).
                    const size_t next_mult = ((req_stage + L.stride - 1) / L.stride) * L.stride;
                    if (next_mult > req_stage && next_mult - req_stage <= ((size_t) 256 << 10)) {
                        req_stage = next_mult;
                    }
                }
                break;
            }
        }
        const size_t min_prefill_resv_st = E.vram_reserve_bytes + std::max(min_resv, min_stage_resv_pad);
        const int want_cnt = (stage_budget > req_stage * (size_t) pref_cnt + ((size_t) 128 << 20)) ? pref_cnt : 2;
        if (stage_budget > req_stage * (size_t) want_cnt + ((size_t) 256 << 20)) {
            size_t fr_before_st = 0, fr_after_st = 0;
            CK(cudaMemGetInfo(&fr_before_st, &tot));
            if (E.alloc_pp_stages(want_cnt, req_stage)) {
                CK(cudaStreamSynchronize(E.st));
                CK(cudaMemGetInfo(&fr_after_st, &tot));
                const size_t stage_tot = E.pp_stage_bytes * (size_t) E.pp_stage_cnt;
                const size_t actual_delta = (fr_before_st > fr_after_st) ? (fr_before_st - fr_after_st) : 0;
                const size_t charged_stage = std::max(stage_tot, actual_delta);
                const size_t from_budget = (charged_stage > stage_slack) ? (charged_stage - stage_slack) : 0;
                budget = (budget > from_budget) ? (budget - from_budget) : 0;
                if (vram_gib < 0.0) {
                    const size_t max_budget_after = (fr_after_st > min_prefill_resv_st) ? (fr_after_st - min_prefill_resv_st) : 0;
                    if (budget > max_budget_after) budget = max_budget_after;
                }
            }
        }
    }
    const int cpu_t = (E.pp_stage[0] && E.pp_stream_t > 1) ? std::min(E.tmax, std::max(E.pp_stream_t - 1, Engine::kDecodePhaseCap)) : E.tmax;
    // Defer init_cpu_bufs() until hybrid/CPU experts are actually needed (saves 72 MiB VRAM + 81 MiB pinned RAM on V100 Full-VRAM!).
    const bool will_need_cpu = (budget < all_exps_bytes);
    if (will_need_cpu && E.pp_stage_cnt == 0 && E.eparts_cap_t < E.tmax) {
        // Hybrid CPU fallback without pp_stage cannot use atomic MoE for T >= 32; restore full eparts backing safely.
        const size_t old_eph = E.prefill_ephemeral_bytes;
        E.free_ephemeral_bufs();
        E.eparts_force_full = true;
        E.prefill_ephemeral_bytes = E.alloc_ephemeral_bufs(E.active_pp_cap);
        E.trim_phase_pool(0);
        const size_t delta = (E.prefill_ephemeral_bytes > old_eph) ? (E.prefill_ephemeral_bytes - old_eph) : 0;
        budget = (budget > delta) ? (budget - delta) : 0;
    }
    if (will_need_cpu) {
        size_t fr_before_cpu = 0;
        CK(cudaMemGetInfo(&fr_before_cpu, &tot));
        E.init_cpu_bufs(cpu_t);
        size_t fr_cpu = 0;
        CK(cudaMemGetInfo(&fr_cpu, &tot));
        const size_t used_cpu = (fr_before_cpu > fr_cpu) ? (fr_before_cpu - fr_cpu) : 0;
        if (budget > used_cpu) budget -= used_cpu;
        else budget = 0;
    }

    size_t used = 0;
    std::vector<int> slots(m.layers.size(), 0);

    // Priority 1: Looped core NVFP4 layers (blk.6..17, executed 3x/token!)
    // Priority 2: Compact quantized outer layers (NVFP4/Q5_K)
    // Priority 3: BF16 outer layers (gu_type == 30, e.g. blk.20..23 in UD-Q8_K_XL) placed in remaining VRAM
    std::vector<int> tier1, tier2, tier3;
    for (int ip = 0; ip < hp.n_phys; ++ip) {
        if (m.layers[(size_t) ip].gu_type == 30) tier3.push_back(ip);
        else if (ip >= hp.core_start && ip < hp.core_end) tier1.push_back(ip);
        else tier2.push_back(ip);
    }

    size_t tier1_full = 0;
    for (int ip : tier1) tier1_full += (size_t) ne * m.layers[(size_t) ip].stride;
    if (!tier1.empty()) {
        if (used + tier1_full <= budget) {
            for (int ip : tier1) slots[(size_t) ip] = ne;
            used += tier1_full;
        } else {
            // Priority 1 strictly maximizes tier1 (core runs 3x/tok & uses compact NVFP4);
            // optional STRATA_TIER2_FLOOR can reserve outer slots when desired.
            int t2_floor = 0;
            if (const char* v = std::getenv("STRATA_TIER2_FLOOR")) t2_floor = std::max(0, std::atoi(v));
            size_t sum_tier2_stride = 0;
            for (int ip : tier2) sum_tier2_stride += m.layers[(size_t) ip].stride;
            size_t tier2_resv = (size_t) t2_floor * sum_tier2_stride;
            size_t sum_tier1_stride = 0;
            for (int ip : tier1) sum_tier1_stride += m.layers[(size_t) ip].stride;
            const size_t min_tier1_bytes = (size_t) std::max(1, ne / 2) * sum_tier1_stride;
            if (tier2.empty() || budget <= used + min_tier1_bytes) {
                tier2_resv = 0;
            } else if (budget - used - min_tier1_bytes < tier2_resv) {
                tier2_resv = budget - used - min_tier1_bytes;
            }
            const size_t t1_budget = (budget > used + tier2_resv) ? (budget - used - tier2_resv) : (budget - used);
            const int s = sum_tier1_stride > 0 ? (int) std::min<size_t>((size_t) ne, t1_budget / sum_tier1_stride) : 0;
            for (int ip : tier1) slots[(size_t) ip] = s;
            used += (size_t) s * sum_tier1_stride;
        }
    }

    // Priority 2: Distribute remaining VRAM across tier2 (sm_89 Hybrid >=2048 defaults to frontloading blk.0..5
    // so blk.0 runs with minimal PCIe wait while cst streams blk.1..5; STRATA_PP_FRONTLOAD=0/1 overrides).
    const char* fl_env = std::getenv("STRATA_PP_FRONTLOAD");
    const bool frontload_en = fl_env ? (std::atoi(fl_env) != 0)
                                     : (E.profile.arch.id == strata::inkling2::ArchId::Sm89Ada && E.tmax >= 2048 && E.pp_stage_cnt > 0);
    if (!tier2.empty() && budget > used) {
        if (frontload_en) {
            for (int ip : tier2) {
                const size_t st_b = m.layers[(size_t) ip].stride;
                if (st_b == 0 || budget <= used) break;
                const int s = (int) std::min<size_t>((size_t) ne, (budget - used) / st_b);
                slots[(size_t) ip] = s;
                used += (size_t) s * st_b;
            }
        } else {
            size_t sum_tier2_stride = 0;
            for (int ip : tier2) sum_tier2_stride += m.layers[(size_t) ip].stride;
            const int s = sum_tier2_stride > 0 ? (int) std::min<size_t>((size_t) ne, (budget - used) / sum_tier2_stride) : 0;
            if (s > 0) {
                for (int ip : tier2) slots[(size_t) ip] = s;
                used += (size_t) s * sum_tier2_stride;
            }
            bool added = true;
            while (added && used < budget) {
                added = false;
                for (int ip : tier2) {
                    const size_t st_b = m.layers[(size_t) ip].stride;
                    if (slots[(size_t) ip] < ne && used + st_b <= budget) {
                        slots[(size_t) ip] += 1;
                        used += st_b;
                        added = true;
                    }
                }
            }
        }
    }

    // Priority 3: Distribute remaining VRAM across BF16 outer layers (e.g. blk.20..23 in UD-Q8_K_XL)
    if (!tier3.empty() && budget > used) {
        size_t sum_tier3_stride = 0;
        for (int ip : tier3) sum_tier3_stride += m.layers[(size_t) ip].stride;
        const int s = sum_tier3_stride > 0 ? (int) std::min<size_t>((size_t) ne, (budget - used) / sum_tier3_stride) : 0;
        if (s > 0) {
            for (int ip : tier3) slots[(size_t) ip] = s;
            used += (size_t) s * sum_tier3_stride;
        }
        bool added = true;
        while (added && used < budget) {
            added = false;
            for (int ip : tier3) {
                const size_t st_b = m.layers[(size_t) ip].stride;
                if (slots[(size_t) ip] < ne && used + st_b <= budget) {
                    slots[(size_t) ip] += 1;
                    used += st_b;
                    added = true;
                }
            }
        }
    }

    std::vector<uint8_t> tmp;
    bool need_cpu = false;
    int total_vram_experts = 0;
    size_t actual_used = 0;
    size_t pinned_used = E.non_expert_pinned_bytes();
    size_t max_pin_bytes = (size_t) 7168 << 20;
#if !defined(_WIN32)
    {
        const long pages = sysconf(_SC_PHYS_PAGES);
        const long psize = sysconf(_SC_PAGESIZE);
        if (pages > 0 && psize > 0) {
            const size_t phys_ram = (size_t) pages * (size_t) psize;
            max_pin_bytes = std::clamp<size_t>((phys_ram * 2) / 5, (size_t) 2048 << 20, (size_t) 12288 << 20);
        }
        if (std::FILE* mf = std::fopen("/proc/meminfo", "r")) {
            char line[256];
            while (std::fgets(line, sizeof line, mf)) {
                unsigned long long avail_kb = 0;
                if (std::sscanf(line, "MemAvailable: %llu kB", &avail_kb) == 1) {
                    const size_t avail_b = (size_t) avail_kb * 1024ULL;
                    const size_t head_b = (size_t) 5120 << 20;
                    if (avail_b > head_b) {
                        max_pin_bytes = std::min<size_t>(avail_b - head_b, ((size_t) 14336 << 20) + pinned_used);
                    } else {
                        max_pin_bytes = (avail_b > ((size_t) 1024 << 20)) ? (avail_b / 2) : 0;
                    }
                    break;
                }
            }
            std::fclose(mf);
        }
    }
#endif
    if (const char* v = std::getenv("STRATA_MAX_PIN_MB")) max_pin_bytes = (size_t) std::max(0, std::atoi(v)) << 20;
    E.max_pin_bytes = max_pin_bytes;
    E.cpu_threads_req = threads;
    std::vector<int> alloc_order;
    alloc_order.reserve((size_t) hp.n_phys);
    for (int ip : tier1) alloc_order.push_back(ip);
    for (int ip : tier2) alloc_order.push_back(ip);
    for (int ip = 0; ip < hp.n_phys; ++ip) {
        if (m.layers[(size_t) ip].gu_type == 30) alloc_order.push_back(ip);
    }
    // F5: Allocate ONE reusable 2-slot pinned staging buffer outside the layer loop (eliminates 24x mlock/munlock stalls!).
    constexpr int kPlaceBatch = 8;
    size_t max_no_hblobs_stride = 0;
    for (int ip : alloc_order) {
        if (slots[(size_t) ip] >= ne) max_no_hblobs_stride = std::max(max_no_hblobs_stride, m.layers[(size_t) ip].stride);
    }
    uint8_t* pbuf[2] = {nullptr, nullptr};
    cudaEvent_t ev_b[2] = {nullptr, nullptr};
    bool has_place_pin = false;
    if (max_no_hblobs_stride > 0) {
        const size_t pb_bytes = (size_t) kPlaceBatch * max_no_hblobs_stride;
        if (cudaMallocHost(&pbuf[0], pb_bytes) == cudaSuccess && cudaMallocHost(&pbuf[1], pb_bytes) == cudaSuccess) {
            CK(cudaEventCreateWithFlags(&ev_b[0], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&ev_b[1], cudaEventDisableTiming));
            has_place_pin = true;
        } else {
            (void) cudaGetLastError();
            if (pbuf[0]) { cudaFreeHost(pbuf[0]); pbuf[0] = nullptr; }
            if (pbuf[1]) { cudaFreeHost(pbuf[1]); pbuf[1] = nullptr; }
        }
    }
    const int place_nw = std::clamp((int) std::thread::hardware_concurrency(), 1, 8);
    static const bool no_core_borrow_place = [] {
        const char* v = std::getenv("STRATA_NO_CORE_BORROW");
        return v != nullptr && (v[0] == '\0' || std::atoi(v) != 0);
    }();
    bool tier1_planned_all_gpu = !tier1.empty();
    for (int tip : tier1) if (slots[(size_t) tip] < ne) tier1_planned_all_gpu = false;
    size_t tier2_hblobs_need = 0;
    for (int tip : tier2) {
        const PhysLayer& TL = m.layers[(size_t) tip];
        if ((slots[(size_t) tip] < ne || (frontload_en && will_need_cpu)) && Engine::layer_allow_pin_hblobs(TL)) {
            tier2_hblobs_need += (size_t) ne * TL.stride;
        }
    }
    size_t tier1_hblobs_need = 0;
    for (int tip : tier1) {
        const PhysLayer& TL = m.layers[(size_t) tip];
        if (Engine::layer_allow_pin_hblobs(TL)) tier1_hblobs_need += (size_t) ne * TL.stride;
    }
    static const bool no_prebuild_tier1 = [] {
        const char* v = std::getenv("STRATA_NO_PREBUILD_TIER1");
        return v != nullptr && (v[0] == '\0' || std::atoi(v) != 0);
    }();
    const bool prebuild_tier1_hblobs = (!no_prebuild_tier1 &&
                                        frontload_en && will_need_cpu &&
                                        E.profile.regime != strata::inkling2::VramRegime::FullVram &&
                                        E.ctx >= 131072 &&
                                        tier1_hblobs_need > 0 &&
                                        (pinned_used + tier1_hblobs_need + tier2_hblobs_need <= max_pin_bytes));

    const float* place_prior_ptr = strata::inkling2::inkling2_mini_decode_prior(hp.n_phys, ne);
    for (int ip : alloc_order) {
        PhysLayer& L = m.layers[(size_t) ip];
        int s = slots[(size_t) ip];
        L.slot_of.assign((size_t) ne, -1);
        L.score.assign((size_t) ne, 0.0);
        if (place_prior_ptr) {
            L.prior_score.assign(place_prior_ptr + (size_t) ip * ne, place_prior_ptr + (size_t) (ip + 1) * ne);
        } else {
            L.prior_score.clear();
        }
        if (!L.d_hits) {
            CK(cudaMalloc(&L.d_hits, (size_t) ne * sizeof(uint32_t)));
            CK(cudaMemset(L.d_hits, 0, (size_t) ne * sizeof(uint32_t)));
            m.gpu_allocs.push_back(L.d_hits);
        }
        // EX-5 Idea D: Pre-allocate L.d_slot_of once for all physical layers (24 * 512 B = 12 KiB) so
        // resize_experts_vram() never calls cudaMalloc for d_slot_of during phase transitions!
        if (!L.d_slot_of) {
            CK(cudaMalloc(&L.d_slot_of, (size_t) ne * sizeof(int32_t)));
            CK(cudaMemset(L.d_slot_of, 0xFF, (size_t) ne * sizeof(int32_t)));
            m.gpu_allocs.push_back(L.d_slot_of);
        }

        if (s > 0) {
            void* dbl = nullptr;
            while (s > 0 && cudaMalloc(&dbl, (size_t) s * L.stride) != cudaSuccess) {
                (void) cudaGetLastError();
                dbl = nullptr;
                --s;
            }
            if (s > 0 && dbl) {
                L.d_blobs = (uint8_t*) dbl;
                L.d_blobs_from_pool = false;
                m.gpu_allocs.push_back(L.d_blobs);
                actual_used += (size_t) s * L.stride;
            } else {
                s = 0;
            }
        }
        slots[(size_t) ip] = s;
        L.n_slots = s;
        L.exp_in_slot.assign((size_t) s, -1);

        const bool is_tier1_ip = (ip >= hp.core_start && ip < hp.core_end) && L.gu_type != 30;
        const bool is_tier2_ip = (ip < hp.core_start || ip >= hp.core_end) && L.gu_type != 30;
        const bool want_core_borrow_pin = (!no_core_borrow_place &&
                                           E.profile.regime != strata::inkling2::VramRegime::FullVram &&
                                           tier1_planned_all_gpu && will_need_cpu &&
                                           hp.num_loops > 1 && hp.core_start < hp.core_end &&
                                           ip >= (int) hp.core_start && ip < (int) hp.core_start + 2 &&
                                           s == ne && L.gu_type != 30);
        if (s < ne || (frontload_en && will_need_cpu && is_tier2_ip) || (prebuild_tier1_hblobs && is_tier1_ip)) {
            if (s < ne) need_cpu = true;
            const size_t hb_bytes = (size_t) ne * L.stride;
            const bool want_pin = Engine::layer_allow_pin_hblobs(L) && (pinned_used + hb_bytes <= max_pin_bytes);
            if (want_pin && cudaHostAlloc(&L.hblobs, hb_bytes, cudaHostAllocDefault) == cudaSuccess) {
                m.host_pinned_allocs.push_back(L.hblobs);
                L.hblobs_pinned = true;
                pinned_used += hb_bytes;
            } else {
                (void) cudaGetLastError();
                L.hblobs = (uint8_t*) std::malloc(hb_bytes);
                if (!L.hblobs) {
                    err = "host malloc failed for layer " + std::to_string(ip);
                    return false;
                }
                L.hblobs_pinned = false;
                m.host_allocs.push_back(L.hblobs);
            }
            build_all_expert_blobs(L, ne, L.hblobs);
            drop_layer_mmap_pages(L, ne);
        } else if (want_core_borrow_pin && !L.hblobs) {
            const size_t hb_bytes = (size_t) ne * L.stride;
            const bool want_pin = Engine::layer_allow_pin_hblobs(L) &&
                                  (pinned_used + tier2_hblobs_need + hb_bytes <= max_pin_bytes);
            if (want_pin && cudaHostAlloc(&L.hblobs, hb_bytes, cudaHostAllocDefault) == cudaSuccess) {
                m.host_pinned_allocs.push_back(L.hblobs);
                L.hblobs_pinned = true;
                pinned_used += hb_bytes;
                build_all_expert_blobs(L, ne, L.hblobs);
                drop_layer_mmap_pages(L, ne);
            } else {
                (void) cudaGetLastError();
                L.hblobs = nullptr;
                L.hblobs_pinned = false;
            }
        }

        if (s > 0) {
            std::vector<int> init_exp((size_t) s);
            if (s < ne && !L.prior_score.empty()) {
                std::vector<int> ord((size_t) ne);
                for (int e = 0; e < ne; ++e) ord[(size_t) e] = e;
                std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) {
                    return L.prior_score[(size_t) a] > L.prior_score[(size_t) b];
                });
                init_exp.assign(ord.begin(), ord.begin() + s);
                std::sort(init_exp.begin(), init_exp.end());
            } else {
                for (int k = 0; k < s; ++k) init_exp[(size_t) k] = k;
            }
            if (L.hblobs) {
                if (s == ne || L.prior_score.empty()) {
                    CK(cudaMemcpyAsync(L.d_blobs, L.hblobs, (size_t) s * L.stride, cudaMemcpyHostToDevice, E.st));
                } else {
                    for (int k = 0; k < s; ) {
                        int k1 = k + 1;
                        while (k1 < s && init_exp[(size_t) k1] == init_exp[(size_t) k] + (k1 - k)) ++k1;
                        const int cnt = k1 - k;
                        const size_t bytes = (size_t) (cnt - 1) * L.stride + L.L.bytes;
                        CK(cudaMemcpyAsync(L.d_blobs + (size_t) k * L.stride,
                                           L.hblobs + (size_t) init_exp[(size_t) k] * L.stride,
                                           bytes, cudaMemcpyHostToDevice, E.st));
                        k = k1;
                    }
                }
            } else if (has_place_pin && L.stride <= max_no_hblobs_stride) {
                int bi = 0;
                for (int k0 = 0; k0 < s; k0 += kPlaceBatch, bi ^= 1) {
                    const int kn = std::min(kPlaceBatch, s - k0);
                    if (k0 >= 2 * kPlaceBatch) CK(cudaEventSynchronize(ev_b[bi]));
                    const int nt = std::min(place_nw, kn);
                    if (nt > 1) {
                        std::vector<std::thread> ths;
                        ths.reserve((size_t) (nt - 1));
                        const int per = (kn + nt - 1) / nt;
                        for (int wi = 1; wi < nt; ++wi) {
                            const int a = wi * per, b = std::min(kn, a + per);
                            if (a >= b) break;
                            ths.emplace_back([&, a, b, k0, bi]() {
                                for (int j = a; j < b; ++j) build_expert_blob(L, init_exp[(size_t) (k0 + j)], pbuf[bi] + (size_t) j * L.stride);
                            });
                        }
                        const int b0 = std::min(kn, per);
                        for (int j = 0; j < b0; ++j) build_expert_blob(L, init_exp[(size_t) (k0 + j)], pbuf[bi] + (size_t) j * L.stride);
                        for (auto& th : ths) th.join();
                    } else {
                        for (int j = 0; j < kn; ++j) build_expert_blob(L, init_exp[(size_t) (k0 + j)], pbuf[bi] + (size_t) j * L.stride);
                    }
                    CK(cudaMemcpyAsync(L.d_blobs + (size_t) k0 * L.stride, pbuf[bi],
                                       (size_t) kn * L.stride, cudaMemcpyHostToDevice, E.st));
                    CK(cudaEventRecord(ev_b[bi], E.st));
                }
                CK(cudaStreamSynchronize(E.st));
            } else {
                tmp.resize(L.L.bytes);
                for (int k = 0; k < s; ++k) {
                    build_expert_blob(L, init_exp[(size_t) k], tmp.data());
                    CK(cudaMemcpy(L.d_blobs + (size_t) k * L.stride, tmp.data(), L.L.bytes, cudaMemcpyHostToDevice));
                }
            }
            for (int k = 0; k < s; ++k) {
                const int e = init_exp[(size_t) k];
                L.slot_of[(size_t) e] = k;
                L.exp_in_slot[(size_t) k] = e;
            }
            E.upload_slot_of(ip, E.st);
            CK(cudaStreamSynchronize(E.st));
            if (s == ne && !L.hblobs) drop_layer_mmap_pages(L, ne);
            total_vram_experts += s;
        }
        if (s < ne) E.hybrid = true;
    }
    if (has_place_pin) {
        cudaEventDestroy(ev_b[0]);
        cudaEventDestroy(ev_b[1]);
        cudaFreeHost(pbuf[0]);
        cudaFreeHost(pbuf[1]);
    }
    if (!E.hybrid && m.has_quant_dense) strata::prefill::Gemm::set_w16_auto_default(true);
    E.pinned_used = pinned_used;
    used = actual_used;

#if defined(STRATA_INKLING2_CPU_POOL)
    if (need_cpu) {
        if (!E.h_h) E.init_cpu_bufs(cpu_t);
        int nw = threads > 0 ? threads : 0;
        E.pool = std::make_unique<strata::kernels::cpu::ExpertPool>(nw, true, true);
        E.pool_workers = E.pool->workers();
        E.pin_host_for_pool();
        std::fprintf(stderr, "strata-inkling2: %s\n", E.pool->placement_line().c_str());
    }
#else
    if (need_cpu) {
        err = "model experts exceed VRAM and STRATA_NATIVE_EXPERTS is OFF";
        return false;
    }
#endif

    E.core_all_gpu = !tier1.empty() && slots[(size_t) tier1[0]] == ne;
    for (int ip : tier1) if (slots[(size_t) ip] != ne) E.core_all_gpu = false;

    size_t fr_after = 0, tot_after = 0;
    CK(cudaMemGetInfo(&fr_after, &tot_after));
    E.placed_slots = total_vram_experts;
    E.placed_bytes = used;
    E.full_cache_bytes = used;
    E.vram_free_after_place = fr_after;

    int tier2_min = ne, tier2_max = 0;
    for (int ip : tier2) {
        tier2_min = std::min(tier2_min, slots[(size_t) ip]);
        tier2_max = std::max(tier2_max, slots[(size_t) ip]);
    }
    std::fprintf(stderr,
                 "strata-inkling2: placed %d/%lld experts in VRAM (%.2f GiB, tier1=%d/128/layer, tier2=%d..%d/128/layer, stage=%dx%.0fMiB, free=%.0f MiB) in %.2f s\n",
                 total_vram_experts, (long long) hp.n_phys * ne, (double) used / (1 << 30),
                 tier1.empty() ? 0 : slots[(size_t) tier1[0]],
                 tier2.empty() ? 0 : tier2_min, tier2.empty() ? 0 : tier2_max,
                 E.pp_stage_cnt, (double) E.pp_stage_bytes / (1 << 20),
                 (double) fr_after / (1 << 20),
                 ms_since(t0) / 1000.0);
    return true;
}

std::vector<int32_t> parse_id_list(const std::string& s) {
    std::vector<int32_t> out;
    out.reserve(s.size() / 4 + 1);
    const char* p = s.c_str();
    const char* const end = p + s.size();
    while (p < end) {
        while (p < end && (*p == ',' || *p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) ++p;
        if (p >= end) break;
        bool neg = false;
        if (*p == '-') { neg = true; ++p; }
        else if (*p == '+') { ++p; }
        if (p >= end || *p < '0' || *p > '9') {
            while (p < end && *p != ',' && *p != ' ' && *p != '\t') ++p;
            continue;
        }
        int64_t val = 0;
        while (p < end && *p >= '0' && *p <= '9') {
            val = val * 10 + (*p - '0');
            ++p;
        }
        out.push_back((int32_t) (neg ? -val : val));
    }
    return out;
}

struct SamplingParams {
    float temp = 0.0f;
    float top_p = 1.0f;
    int top_k = 0;
    float min_p = 0.0f;
    float rep_pen = 1.0f;
    float freq_pen = 0.0f;
    float pres_pen = 0.0f;
    int pen_last_n = 64;
    uint64_t seed = 0;
    bool needs_logits() const {
        return temp > 0.0f || rep_pen != 1.0f || freq_pen != 0.0f || pres_pen != 0.0f;
    }
};

struct SamplingDist {
    std::vector<int32_t> idx;
    std::vector<float> probs;
    float prob_of(int32_t tok) const {
        for (size_t i = 0; i < idx.size(); ++i) if (idx[i] == tok) return probs[i];
        return 0.0f;
    }
};

SamplingDist build_sampling_dist(const float* raw_logits, int V, const std::vector<int32_t>& history, const SamplingParams& smp) {
    std::vector<float> adj(raw_logits, raw_logits + V);
    if (smp.pen_last_n > 0 && (smp.rep_pen != 1.0f || smp.freq_pen != 0.0f || smp.pres_pen != 0.0f)) {
        std::vector<int> counts;
        std::vector<int32_t> uniq;
        const size_t start = history.size() > (size_t) smp.pen_last_n ? history.size() - (size_t) smp.pen_last_n : 0;
        for (size_t i = start; i < history.size(); ++i) {
            const int32_t tok = history[i];
            if (tok < 0 || tok >= V) continue;
            bool found = false;
            for (size_t j = 0; j < uniq.size(); ++j) {
                if (uniq[j] == tok) { ++counts[j]; found = true; break; }
            }
            if (!found) { uniq.push_back(tok); counts.push_back(1); }
        }
        for (size_t j = 0; j < uniq.size(); ++j) {
            float& l = adj[(size_t) uniq[j]];
            if (smp.rep_pen != 1.0f) l = (l <= 0.0f) ? (l * smp.rep_pen) : (l / smp.rep_pen);
            l -= (float) counts[j] * smp.freq_pen + smp.pres_pen;
        }
    }
    SamplingDist d;
    if (smp.temp <= 0.0f) {
        int best = 0;
        for (int i = 1; i < V; ++i) if (adj[(size_t) i] > adj[(size_t) best]) best = i;
        d.idx = {best};
        d.probs = {1.0f};
        return d;
    }
    const int K = (smp.top_k > 0 && smp.top_k < V) ? smp.top_k : std::min(V, 256);
    std::vector<int32_t> idx((size_t) V);
    for (int i = 0; i < V; ++i) idx[(size_t) i] = i;
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int32_t a, int32_t b) {
        return adj[(size_t) a] > adj[(size_t) b];
    });
    const float max_l = adj[(size_t) idx[0]];
    const float inv_t = 1.0f / smp.temp;
    std::vector<float> probs((size_t) K);
    double sum = 0.0;
    for (int i = 0; i < K; ++i) {
        const float p = std::exp((adj[(size_t) idx[(size_t) i]] - max_l) * inv_t);
        probs[(size_t) i] = p;
        sum += p;
    }
    int keep = K;
    if (smp.min_p > 0.0f) {
        const float thr = probs[0] * smp.min_p;
        while (keep > 1 && probs[(size_t) (keep - 1)] < thr) --keep;
    }
    if (smp.top_p > 0.0f && smp.top_p < 1.0f) {
        double csum = 0.0;
        const double target = sum * (double) smp.top_p;
        for (int i = 0; i < keep; ++i) {
            csum += probs[(size_t) i];
            if (csum >= target) { keep = i + 1; break; }
        }
    }
    double ksum = 0.0;
    for (int i = 0; i < keep; ++i) ksum += probs[(size_t) i];
    const float inv_ksum = (ksum > 0.0) ? (float) (1.0 / ksum) : 1.0f;
    d.idx.assign(idx.begin(), idx.begin() + keep);
    d.probs.resize((size_t) keep);
    for (int i = 0; i < keep; ++i) d.probs[(size_t) i] = probs[(size_t) i] * inv_ksum;
    return d;
}

int32_t sample_from_dist(const SamplingDist& d, std::mt19937_64& rng) {
    if (d.idx.size() <= 1) return d.idx[0];
    const double r = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    double acc = 0.0;
    for (size_t i = 0; i < d.idx.size(); ++i) {
        acc += d.probs[i];
        if (r <= acc) return d.idx[i];
    }
    return d.idx[0];
}

SamplingDist build_sampling_dist_from_topk(const float* vals, const int32_t* idxs, int K, const SamplingParams& smp) {
    SamplingDist d;
    if (K <= 0) return d;
    const float max_l = vals[0];
    const float inv_t = 1.0f / smp.temp;
    std::vector<float> probs((size_t) K);
    double sum = 0.0;
    for (int i = 0; i < K; ++i) {
        const float p = std::exp((vals[(size_t) i] - max_l) * inv_t);
        probs[(size_t) i] = p;
        sum += p;
    }
    int keep = K;
    if (smp.min_p > 0.0f) {
        const float thr = probs[0] * smp.min_p;
        while (keep > 1 && probs[(size_t) (keep - 1)] < thr) --keep;
    }
    if (smp.top_p > 0.0f && smp.top_p < 1.0f) {
        double csum = 0.0;
        const double target = sum * (double) smp.top_p;
        for (int i = 0; i < keep; ++i) {
            csum += probs[(size_t) i];
            if (csum >= target) { keep = i + 1; break; }
        }
    }
    double ksum = 0.0;
    for (int i = 0; i < keep; ++i) ksum += probs[(size_t) i];
    const float inv_ksum = (ksum > 0.0) ? (float) (1.0 / ksum) : 1.0f;
    d.idx.assign(idxs, idxs + keep);
    d.probs.resize((size_t) keep);
    for (int i = 0; i < keep; ++i) d.probs[(size_t) i] = probs[(size_t) i] * inv_ksum;
    return d;
}

inline bool can_use_gpu_topk(const SamplingParams& smp, int V) {
    if (smp.temp <= 0.0f) return false;
    const int K = (smp.top_k > 0 && smp.top_k < V) ? smp.top_k : std::min(V, 256);
    return K >= 1 && K <= 256 && smp.pen_last_n <= 256;
}

inline void collect_penalty_counts(const std::vector<int32_t>& history, const int32_t* extra, int n_extra,
                                   int V, const SamplingParams& smp,
                                   std::vector<int32_t>& uniq, std::vector<int32_t>& cnts) {
    uniq.clear();
    cnts.clear();
    if (smp.pen_last_n <= 0 || (smp.rep_pen == 1.0f && smp.freq_pen == 0.0f && smp.pres_pen == 0.0f)) return;
    const size_t total = history.size() + (size_t) n_extra;
    const size_t start = total > (size_t) smp.pen_last_n ? total - (size_t) smp.pen_last_n : 0;
    auto push_tok = [&](int32_t tok) {
        if (tok < 0 || tok >= V) return;
        for (size_t j = 0; j < uniq.size(); ++j) {
            if (uniq[j] == tok) { ++cnts[j]; return; }
        }
        uniq.push_back(tok);
        cnts.push_back(1);
    };
    for (size_t i = start; i < history.size(); ++i) push_tok(history[i]);
    const size_t ex_start = start > history.size() ? (start - history.size()) : 0;
    for (size_t i = ex_start; i < (size_t) n_extra; ++i) push_tok(extra[i]);
}

bool build_sampling_dists_gpu(Engine& E, int rows, const std::vector<int32_t>& history,
                              const int32_t* extra, int n_extra, const SamplingParams& smp,
                              std::vector<SamplingDist>& out_dists, int row_off = 0) {
    const int V = (int) E.m->hp.vocab_unpadded;
    if (!can_use_gpu_topk(smp, V) || rows <= 0 || row_off < 0 || row_off + rows > E.verify_max_t) return false;
    const int K = (smp.top_k > 0 && smp.top_k < V) ? smp.top_k : std::min(V, 256);
    std::vector<int32_t> p_toks, p_cnts;
    collect_penalty_counts(history, extra, n_extra, V, smp, p_toks, p_cnts);
    std::vector<float> vals((size_t) rows * (size_t) K);
    std::vector<int32_t> idxs((size_t) rows * (size_t) K);
    if (!E.gpu_topk_rows(rows, K, p_toks.data(), p_cnts.data(), (int) p_toks.size(),
                         smp.rep_pen, smp.freq_pen, smp.pres_pen, vals.data(), idxs.data(), row_off)) {
        // Fallback: copy full logits D2H (note penalties were already applied in-place on GPU if n_pen > 0).
        std::vector<float> full((size_t) rows * (size_t) V);
        CK(cudaMemcpyAsync(full.data(), E.logits + (size_t) row_off * (size_t) V, full.size() * sizeof(float), cudaMemcpyDeviceToHost, E.st));
        CK(cudaStreamSynchronize(E.st));
        SamplingParams smp_nopen = smp;
        smp_nopen.rep_pen = 1.0f; smp_nopen.freq_pen = 0.0f; smp_nopen.pres_pen = 0.0f;
        out_dists.resize((size_t) rows);
        for (int r = 0; r < rows; ++r) {
            out_dists[(size_t) r] = build_sampling_dist(full.data() + (size_t) r * V, V, history, smp_nopen);
        }
        return true;
    }
    out_dists.resize((size_t) rows);
    for (int r = 0; r < rows; ++r) {
        out_dists[(size_t) r] = build_sampling_dist_from_topk(
            vals.data() + (size_t) r * K, idxs.data() + (size_t) r * K, K, smp);
    }
    return true;
}

int32_t sample_logits(const std::vector<float>& logits, const std::vector<int32_t>& history, const SamplingParams& smp, std::mt19937_64& rng) {
    return sample_from_dist(build_sampling_dist(logits.data(), (int) logits.size(), history, smp), rng);
}

int32_t sample_residual_dist(const SamplingDist& p, const SamplingDist& q, std::mt19937_64& rng) {
    std::vector<int32_t> r_idx;
    std::vector<float> r_w;
    r_idx.reserve(p.idx.size());
    r_w.reserve(p.idx.size());
    double sum = 0.0;
    for (size_t i = 0; i < p.idx.size(); ++i) {
        const float diff = p.probs[i] - q.prob_of(p.idx[i]);
        if (diff > 0.0f) {
            r_idx.push_back(p.idx[i]);
            r_w.push_back(diff);
            sum += diff;
        }
    }
    if (sum <= 0.0) return sample_from_dist(p, rng);
    const double u = std::uniform_real_distribution<double>(0.0, sum)(rng);
    double acc = 0.0;
    for (size_t i = 0; i < r_idx.size(); ++i) {
        acc += r_w[i];
        if (u <= acc) return r_idx[i];
    }
    return r_idx[0];
}

/// Shared autoregressive / DSpark speculative decode loop for both `--serve` and `--bench`/CLI.
template <typename EmitFn, typename StopCheckFn>
void decode_tokens_loop(Engine& E, DsparkEngine* dspark, int draft_max, int32_t first_tok, int limit,
                        const SamplingParams& smp, std::mt19937_64& rng,
                        std::vector<int32_t>& hist, std::vector<int32_t>& live_tokens,
                        EmitFn&& on_emit, StopCheckFn&& is_stopped, bool* out_hit_eos = nullptr) {
    if (E.hybrid && !E.decode_scores_normalized) E.normalize_decode_scores();

    const int V = (int) E.m->hp.vocab_unpadded;
    const bool sample_on = smp.needs_logits();
    int32_t next_tok = first_tok;
    int emitted = 0;
    std::vector<float> lg;
    std::vector<int32_t> d_toks, vtoks, t_picks;
    std::vector<std::vector<float>> d_logits;
    std::vector<SamplingDist> q_dists;
    if (out_hit_eos) *out_hit_eos = false;

    const bool use_gpu_topk = sample_on && can_use_gpu_topk(smp, V);
    std::vector<SamplingDist> tmp_dists;
    std::vector<SamplingDist> p_dists;

    auto feed_single = [&](int32_t tok) {
        const int pos_pre = (int) E.pos;
        next_tok = E.forward(&tok, 1, sample_on, use_gpu_topk ? nullptr : &lg, false, dspark != nullptr, 0, nullptr, true, use_gpu_topk);
        if (use_gpu_topk) {
            build_sampling_dists_gpu(E, 1, hist, nullptr, 0, smp, tmp_dists);
            next_tok = sample_from_dist(tmp_dists[0], rng);
        } else if (sample_on) {
            next_tok = sample_logits(lg, hist, smp, rng);
        }
        live_tokens.push_back(tok);
        if (dspark) dspark->inject(E, pos_pre, 1);
    };

    const bool adaptive_dspark = [] {
        const char* v = std::getenv("STRATA_DSPARK_ADAPTIVE");
        return !(v && v[0] == '0');
    }();
    const bool conf_gate = [] {
        const char* v = std::getenv("STRATA_DSPARK_CONF_GATE");
        return !(v && v[0] == '0');
    }();
    const bool pld_enabled = [] {
        const char* v = std::getenv("STRATA_DSPARK_PLD");
        return !(v && v[0] == '0');
    }();
    auto env_f = [](const char* k, float def) -> float {
        const char* v = std::getenv(k);
        return (v && v[0]) ? (float) std::atof(v) : def;
    };
    auto env_i = [](const char* k, int def) -> int {
        const char* v = std::getenv(k);
        return (v && v[0]) ? std::atoi(v) : def;
    };
    const bool tight_hy = (E.hybrid && !E.core_all_gpu);
    const float tau_min0 = env_f("STRATA_DSPARK_TAU0", tight_hy ? 0.15f : 0.0f);
    const float tau_step = env_f("STRATA_DSPARK_TAU_STEP", tight_hy ? 0.50f : 0.45f);
    const float tau_cum  = env_f("STRATA_DSPARK_TAU_CUM", tight_hy ? 0.30f : 0.25f);
    const float tau_ext  = env_f("STRATA_DSPARK_TAU_EXT", E.hybrid ? 0.78f : 0.90f);
    const float tau_cum_ext = env_f("STRATA_DSPARK_TAU_CUM_EXT", E.hybrid ? 0.58f : 0.72f);
    const int base_k_cfg = std::clamp(env_i("STRATA_DSPARK_BASE_K", (E.hybrid && draft_max >= 4) ? 3 : draft_max), 1, 7);
    const int max_ext_cfg = [&] {
        const char* v = std::getenv("STRATA_DSPARK_MAX_EXT");
        return (v && v[0]) ? std::clamp(std::atoi(v), 1, 7) : draft_max;
    }();
    const bool sample_pld = [] {
        const char* v = std::getenv("STRATA_DSPARK_SAMPLE_PLD");
        return !(v && v[0] == '0');
    }();
    const float tau_min0_s = env_f("STRATA_DSPARK_SAMPLE_TAU0", E.hybrid ? (tight_hy ? 0.30f : 0.25f) : 0.15f);
    const float tau_step_s = env_f("STRATA_DSPARK_SAMPLE_TAU_STEP", E.hybrid ? (tight_hy ? 0.50f : 0.48f) : 0.45f);
    const float tau_cum_s  = env_f("STRATA_DSPARK_SAMPLE_TAU_CUM",  E.hybrid ? (tight_hy ? 0.30f : 0.25f) : 0.25f);
    const float sample_alpha_min = env_f("STRATA_DSPARK_SAMPLE_ALPHA_MIN", E.hybrid ? 0.25f : 0.18f);
    const bool use_conf_cd = conf_gate && (tight_hy || (sample_on && E.hybrid));
    int consec_abort = 0;
    const int pld_base_min = [] (size_t init_n, bool th) {
        const char* v = std::getenv("STRATA_DSPARK_PLD_MIN_MATCH");
        if (v && v[0]) return std::clamp(std::atoi(v), 4, 12);
        return (init_n >= 65536 || th) ? 7 : 6;
    }(live_tokens.size(), tight_hy);
    int pld_min_match = pld_base_min;
    int pld_cooldown = 0;
    // D7: Unbiased truncated-geometric MLE for per-token acceptance probability alpha + batched cooldown inject!
    // Every verify step observes `m` successes out of `min(K, m + 1)` Bernoulli trials (unbiased for all K=1..7).
    float alpha_num = 3.25f;
    float alpha_den = 5.0f;
    int cur_dm = draft_max;
    int skip_cd = 0;
    int cd_backoff = 12;
    int cd_cnt = 0;
    int cd_pos0 = 0;
    constexpr int kCdFeatBase = 16;
    constexpr int kCdMaxBatch = 48;
    const bool can_batch_cd = (dspark != nullptr && E.tmax >= kCdFeatBase + kCdMaxBatch &&
                               E.d_tgt_feat[0] && E.d_tgt_feat[1] && E.d_tgt_feat[2] && E.d_tgt_feat[3]);
    auto flush_cd_inject = [&]() {
        if (dspark && cd_cnt > 0) {
            dspark->inject(E, cd_pos0, cd_cnt, kCdFeatBase);
            cd_cnt = 0;
        }
    };
    auto feed_single_cd = [&](int32_t tok) {
        const int pos_pre = (int) E.pos;
        if (!sample_on) {
            next_tok = E.forward(&tok, 1, false, nullptr, false, dspark != nullptr);
        } else if (use_gpu_topk) {
            E.forward(&tok, 1, true, nullptr, false, dspark != nullptr, 0, nullptr, true, true);
            build_sampling_dists_gpu(E, 1, hist, nullptr, 0, smp, tmp_dists);
            next_tok = sample_from_dist(tmp_dists[0], rng);
        } else {
            lg.clear();
            E.forward(&tok, 1, true, &lg, false, dspark != nullptr);
            next_tok = sample_logits(lg, hist, smp, rng);
        }
        live_tokens.push_back(tok);
        if (dspark) {
            if (can_batch_cd) {
                if (cd_cnt == 0) cd_pos0 = pos_pre;
                const size_t nbytes = (size_t) E.m->hp.n_embd * sizeof(float);
                const size_t dst_off = (size_t) (kCdFeatBase + cd_cnt) * (size_t) E.m->hp.n_embd;
                for (int fi = 0; fi < 4; ++fi) {
                    CK(cudaMemcpyAsync(E.d_tgt_feat[fi] + dst_off, E.d_tgt_feat[fi], nbytes,
                                       cudaMemcpyDeviceToDevice, E.st));
                }
                ++cd_cnt;
                if (cd_cnt >= kCdMaxBatch) flush_cd_inject();
            } else {
                dspark->inject(E, pos_pre, 1);
            }
        }
    };

    while (emitted < limit && !is_stopped()) {
        if (!on_emit(next_tok)) {
            if (out_hit_eos) *out_hit_eos = true;
            ++emitted;
            break;
        }
        ++emitted;
        const int rem = limit - emitted;
        if (rem <= 0 || is_stopped()) break;

        if (!dspark || rem < 2 || draft_max < 1) {
            flush_cd_inject();
            feed_single(next_tok);
            if (E.hybrid && (emitted % 8 == 0)) E.rebalance(E.core_all_gpu ? 4 : 8, 2, false);
            continue;
        }

        const int32_t anchor = next_tok;
        bool used_pld = false;
        d_toks.clear();
        if (pld_enabled && (!sample_on || sample_pld)) {
            E.pld.sync(live_tokens);
            const int pld_max_k = std::min(std::min(E.verify_max_t - 1, std::max(draft_max, tight_hy ? 4 : 5)), rem - 1);
            if (pld_max_k >= 1 && E.pld.try_draft(live_tokens, anchor, pld_max_k, pld_min_match, d_toks)) {
                used_pld = true;
                if (sample_on) {
                    q_dists.resize(d_toks.size());
                    for (size_t r = 0; r < d_toks.size(); ++r) {
                        q_dists[r].idx = {d_toks[r]};
                        q_dists[r].probs = {1.0f};
                    }
                }
            }
        }

        if (!used_pld) {
            if (adaptive_dspark && ((!conf_gate && cur_dm == 0) || use_conf_cd) && skip_cd > 0) {
                feed_single_cd(next_tok);
                --skip_cd;
                if (E.hybrid && (emitted % 8 == 0)) E.rebalance(E.core_all_gpu ? 4 : 8, 2, false);
                continue;
            }
            flush_cd_inject();

            if (!sample_on && adaptive_dspark && conf_gate) {
                const int base_k = std::min(draft_max, rem - 1);
                const int eff_base_k = std::min(base_k, base_k_cfg);
                const int ext_lim = (draft_max >= 3) ? std::max(draft_max, max_ext_cfg) : draft_max;
                const int max_k = std::min(std::min(E.verify_max_t - 1, ext_lim), rem - 1);
                dspark->draft_confident(E, anchor, eff_base_k, max_k,
                                        tau_min0, tau_step, tau_cum, tau_ext, tau_cum_ext, d_toks);
                if (d_toks.empty()) {
                    feed_single(anchor);
                    if (pld_cooldown > 0 && --pld_cooldown == 0) pld_min_match = std::max(pld_base_min, pld_min_match - 1);
                    if (E.hybrid && (emitted % 8 == 0)) E.rebalance(E.core_all_gpu ? 4 : 8, 2, false);
                    continue;
                }
            } else if (sample_on && adaptive_dspark && conf_gate) {
                const int base_k = std::min(draft_max, rem - 1);
                const int eff_base_k = std::min(base_k, base_k_cfg);
                const int ext_lim = (draft_max >= 3) ? std::max(draft_max, max_ext_cfg) : draft_max;
                const int max_k = std::min(std::min(E.verify_max_t - 1, ext_lim), rem - 1);
                const size_t tail_keep = (!use_gpu_topk && smp.pen_last_n > 0) ? (size_t) smp.pen_last_n : 0;
                const size_t h_start = hist.size() > tail_keep ? (hist.size() - tail_keep) : 0;
                q_dists.clear();
                dspark->draft_sampled_confident(
                    E, anchor, eff_base_k, max_k,
                    tau_step_s, tau_cum_s, tau_ext, tau_cum_ext,
                    d_toks, use_gpu_topk ? nullptr : &d_logits,
                    [&](int step_i, std::vector<float>& row_lg, float& out_pmax) -> int32_t {
                        SamplingDist q;
                        if (use_gpu_topk) {
                            build_sampling_dists_gpu(E, 1, hist, d_toks.data(), step_i, smp, tmp_dists);
                            q = std::move(tmp_dists[0]);
                        } else {
                            std::vector<int32_t> tmp_hist(hist.begin() + (ptrdiff_t) h_start, hist.end());
                            tmp_hist.insert(tmp_hist.end(), d_toks.begin(), d_toks.begin() + step_i);
                            q = build_sampling_dist(row_lg.data(), V, tmp_hist, smp);
                        }
                        out_pmax = q.probs.empty() ? 0.0f : q.probs[0];
                        if (step_i == 0 && out_pmax < tau_min0_s) return -1;
                        const int32_t tok = sample_from_dist(q, rng);
                        q_dists.push_back(std::move(q));
                        return tok;
                    },
                    use_gpu_topk);
                if (d_toks.empty()) {
                    ++consec_abort;
                    if (use_conf_cd && consec_abort >= 3) {
                        skip_cd = 8;
                        consec_abort = 0;
                    }
                    feed_single(anchor);
                    if (pld_cooldown > 0 && --pld_cooldown == 0) pld_min_match = std::max(pld_base_min, pld_min_match - 1);
                    if (E.hybrid && (emitted % 8 == 0)) E.rebalance(E.core_all_gpu ? 4 : 8, 2, false);
                    continue;
                }
                consec_abort = 0;
            } else {
                const int use_dm = (adaptive_dspark && !conf_gate) ? std::max(1, cur_dm) : draft_max;
                const int max_k = std::min(use_dm, rem - 1);
                if (use_gpu_topk) {
                    q_dists.resize((size_t) max_k);
                    dspark->draft(E, anchor, max_k, d_toks, nullptr, [&](int step_i, std::vector<float>&) {
                        build_sampling_dists_gpu(E, 1, hist, d_toks.data(), step_i, smp, tmp_dists);
                        q_dists[(size_t) step_i] = std::move(tmp_dists[0]);
                        return sample_from_dist(q_dists[(size_t) step_i], rng);
                    }, true);
                } else if (sample_on) {
                    q_dists.resize((size_t) max_k);
                    const size_t tail_keep = smp.pen_last_n > 0 ? (size_t) smp.pen_last_n : 0;
                    const size_t h_start = hist.size() > tail_keep ? (hist.size() - tail_keep) : 0;
                    dspark->draft(E, anchor, max_k, d_toks, &d_logits, [&](int step_i, std::vector<float>& row_lg) {
                        std::vector<int32_t> tmp_hist(hist.begin() + (ptrdiff_t) h_start, hist.end());
                        tmp_hist.insert(tmp_hist.end(), d_toks.begin(), d_toks.begin() + step_i);
                        q_dists[(size_t) step_i] = build_sampling_dist(row_lg.data(), V, tmp_hist, smp);
                        return sample_from_dist(q_dists[(size_t) step_i], rng);
                    });
                } else {
                    dspark->draft(E, anchor, max_k, d_toks, nullptr, [&](int, std::vector<float>&) -> int32_t { return 0; });
                }
            }
        } else {
            flush_cd_inject();
        }

        const int K = (int) d_toks.size();
        vtoks.resize((size_t) (1 + K));
        vtoks[0] = anchor;
        for (int i = 0; i < K; ++i) vtoks[(size_t) (1 + i)] = d_toks[(size_t) i];

        const int pos_pre = (int) E.pos;
        lg.clear();
        E.forward(vtoks.data(), 1 + K, sample_on, use_gpu_topk ? nullptr : &lg, sample_on, true, K,
                  sample_on ? nullptr : &t_picks, true, use_gpu_topk);
        if (use_gpu_topk) {
            const bool has_pen = (smp.pen_last_n > 0 && (smp.rep_pen != 1.0f || smp.freq_pen != 0.0f || smp.pres_pen != 0.0f));
            if (!has_pen || K == 0) {
                build_sampling_dists_gpu(E, 1 + K, hist, nullptr, 0, smp, p_dists, 0);
            } else {
                p_dists.resize((size_t) (1 + K));
                for (int r = 0; r <= K; ++r) {
                    build_sampling_dists_gpu(E, 1, hist, d_toks.data(), r, smp, tmp_dists, r);
                    p_dists[(size_t) r] = std::move(tmp_dists[0]);
                }
            }
        }

        int m = 0;
        bool hit_eos = false;
        if (!sample_on) {
            while (m < K && t_picks[(size_t) m] == d_toks[(size_t) m]) ++m;
            for (int j = 0; j < m; ++j) {
                if (!on_emit(d_toks[(size_t) j])) {
                    m = j + 1;
                    hit_eos = true;
                    break;
                }
            }
            if (!hit_eos) next_tok = t_picks[(size_t) m];
        } else {
            bool rejected = false;
            for (int r = 0; r < K; ++r) {
                const SamplingDist p_r_local = use_gpu_topk ? SamplingDist{} : build_sampling_dist(lg.data() + (size_t) r * V, V, hist, smp);
                const SamplingDist& p_r = use_gpu_topk ? p_dists[(size_t) r] : p_r_local;
                const float p_x = p_r.prob_of(d_toks[(size_t) r]);
                const float q_x = q_dists[(size_t) r].prob_of(d_toks[(size_t) r]);
                const double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
                if (u * (double) q_x <= (double) p_x) {
                    ++m;
                    if (!on_emit(d_toks[(size_t) r])) {
                        hit_eos = true;
                        break;
                    }
                } else {
                    next_tok = sample_residual_dist(p_r, q_dists[(size_t) r], rng);
                    rejected = true;
                    break;
                }
            }
            if (!rejected && !hit_eos) {
                if (use_gpu_topk) {
                    next_tok = sample_from_dist(p_dists[(size_t) K], rng);
                } else {
                    const SamplingDist p_last = build_sampling_dist(lg.data() + (size_t) K * V, V, hist, smp);
                    next_tok = sample_from_dist(p_last, rng);
                }
            }
        }

        emitted += m;
        E.commit_verify(m, K, pos_pre);
        for (int i = 0; i < 1 + m; ++i) live_tokens.push_back(vtoks[(size_t) i]);
        if (used_pld) {
            ++dspark->pld_calls;
            dspark->pld_drafted += K;
            dspark->pld_accepted += m;
            if (m == 0) {
                pld_min_match = std::min(10, pld_min_match + 1);
                pld_cooldown = 24;
            } else if (m >= 2) {
                pld_cooldown = 0;
                pld_min_match = std::max(pld_base_min, pld_min_match - 1);
            }
        } else {
            dspark->n_accepted += m;
            if (pld_cooldown > 0 && --pld_cooldown == 0) pld_min_match = std::max(pld_base_min, pld_min_match - 1);
        }
        dspark->inject(E, pos_pre, 1 + m);
        if (adaptive_dspark && !used_pld && K >= 1 && (!conf_gate || use_conf_cd)) {
            const int trials = std::min(K, m + 1);
            alpha_num = 0.85f * alpha_num + (float) m;
            alpha_den = 0.85f * alpha_den + (float) trials;
            const float alpha = alpha_num / std::max(1e-3f, alpha_den);
            if (!conf_gate) {
                // Hybrid mode pays incremental CPU DDR4 bandwidth per extra draft token, so raise K-promotion thresholds
                // to match the exact mathematical break-even (keeps dm=4 for high-acceptance code, dm=2 for prose).
                const float th_max = E.hybrid ? 0.76f : 0.55f;
                const float th_3   = E.hybrid ? 0.56f : 0.38f;
                const float th_2   = E.hybrid ? 0.38f : 0.24f;
                const float th_1   = E.hybrid ? 0.22f : 0.16f;
                if (cur_dm == 0) {
                    if (m >= 1) {
                        cd_backoff = 12;
                        cur_dm = (alpha >= th_3) ? std::min(draft_max, 3) : std::min(draft_max, 2);
                    } else {
                        skip_cd = cd_backoff;
                        cd_backoff = std::min(48, cd_backoff * 2);
                    }
                } else {
                    if (alpha >= th_max) { cur_dm = draft_max; cd_backoff = 12; }
                    else if (alpha >= th_3) { cur_dm = std::min(draft_max, 3); cd_backoff = 12; }
                    else if (alpha >= th_2) { cur_dm = std::min(draft_max, 2); cd_backoff = 12; }
                    else if (alpha >= th_1) { cur_dm = 1; cd_backoff = 12; }
                    else { cur_dm = 0; skip_cd = cd_backoff; cd_backoff = std::min(48, cd_backoff * 2); }
                }
            } else {
                const float a_thr = sample_on ? sample_alpha_min : 0.18f;
                if (m >= 1) {
                    cd_backoff = 12;
                } else if (alpha_den >= 4.0f && alpha < a_thr) {
                    skip_cd = cd_backoff;
                    cd_backoff = std::min(48, cd_backoff * 2);
                }
            }
        }
        if (hit_eos) {
            if (out_hit_eos) *out_hit_eos = true;
            break;
        }
        if (E.hybrid && (emitted % 8 < (1 + m))) E.rebalance(E.core_all_gpu ? 8 : 12, 2, false);
    }
    flush_cd_inject();
}

}  // namespace

int run(int argc, char** argv) {
    std::string model_path;
    std::string draft_path;
    std::string tokens_str = "1,2";
    std::string dump_logits_path;
    int max_ctx = 65536;
    int chunk = 0;
    int max_new = 32;
    int draft_max = 3;
    int cpu_threads = 0;
    double expert_vram = -1.0;
    bool serve = false;
    bool bench = false;
    bool kv_q8 = true;
    bool vision_enabled = false;
    bool vram_elastic = false;
    int vram_reserve_mib = 0;
    int32_t img_pad_id = 200054;
    bool profile = false;
    bool profile_detail = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next_arg = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "strata-inkling2: missing value for %s\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--native" || a == "--model") model_path = next_arg();
        else if (a == "--draft" || a == "--draft-model") draft_path = next_arg();
        else if (a == "--draft-max" || a == "--spec") draft_max = std::clamp(std::atoi(next_arg()), 1, 7);
        else if (a == "--max-context") max_ctx = std::atoi(next_arg());
        else if (a == "--chunk" || a == "--prefill") chunk = std::atoi(next_arg());
        else if (a == "--max-new" || a == "-n") max_new = std::atoi(next_arg());
        else if (a == "--tokens") tokens_str = next_arg();
        else if (a == "--expert-vram") expert_vram = std::atof(next_arg());
        else if (a == "--cpu-threads" || a == "--pool-workers") cpu_threads = std::atoi(next_arg());
        else if (a == "--dump-logits") dump_logits_path = next_arg();
        else if (a == "--kv-f16") kv_q8 = false;
        else if (a == "--kv-q8") kv_q8 = true;
        else if (a == "--vision") vision_enabled = true;
        else if (a == "--vram-elastic") vram_elastic = true;
        else if (a == "--vram-segment-mib") (void) next_arg();
        else if (a == "--vram-reserve-mib") vram_reserve_mib = std::max(0, std::atoi(next_arg()));
        else if (a == "--image-pad-id") img_pad_id = std::atoi(next_arg());
        else if (a == "--serve") serve = true;
        else if (a == "--bench") bench = true;
        else if (a == "--profile") profile = true;
        else if (a == "--profile-detail") { profile = true; profile_detail = true; }
    }

    if (model_path.empty()) {
        std::fprintf(stderr, "usage: strata-inkling2 --model <file.gguf> [--draft <dspark.gguf>] [--draft-max N] [--max-context N] [--kv-q8|--kv-f16] [--tokens IDS] [--max-new N] [--dump-logits PATH] [--bench] [--serve]\n");
        return 2;
    }

    Model m;
    std::string err;
    if (!load_model(model_path, m, err)) {
        std::fprintf(stderr, "strata-inkling2: load failed: %s\n", err.c_str());
        return 1;
    }
    std::unique_ptr<DsparkModel> dm;
    std::unique_ptr<DsparkEngine> dspark;
    if (!draft_path.empty()) {
        dm = std::make_unique<DsparkModel>();
        if (!load_dspark_model(draft_path, m.hp, *dm, err)) {
            std::fprintf(stderr, "strata-inkling2: draft load failed: %s\n", err.c_str());
            return 1;
        }
    }

    // Resolve two-axis (SM arch x VRAM residency regime) profile after dense (+ optional draft) weights load.
    strata::inkling2::ProfileInput pin;
    {
        int cur_dev = 0, raw_maj = 0, raw_min = 0, sm_cnt = 0;
        cudaGetDevice(&cur_dev);
        cudaDeviceGetAttribute(&raw_maj, cudaDevAttrComputeCapabilityMajor, cur_dev);
        cudaDeviceGetAttribute(&raw_min, cudaDevAttrComputeCapabilityMinor, cur_dev);
        cudaDeviceGetAttribute(&sm_cnt, cudaDevAttrMultiProcessorCount, cur_dev);
        size_t vfree = 0, vtot = 0;
        cudaMemGetInfo(&vfree, &vtot);

        pin.cc_major = strata::cc_major_of(raw_maj);
        pin.cc_minor = strata::cc_minor_of(raw_min);
        pin.sm_count = sm_cnt;
        pin.vram_total_bytes = vtot;
        const size_t vram_res_b = (size_t) vram_reserve_mib << 20;
        pin.vram_free_bytes = (vfree > vram_res_b) ? (vfree - vram_res_b) : 0;
        pin.max_ctx = max_ctx;
        pin.requested_chunk = chunk;
        pin.kv_q8 = kv_q8;
        pin.has_draft = (dm != nullptr);
        pin.n_swa = (int) m.hp.n_swa;
        pin.n_full_layers = 0;
        pin.n_swa_layers = 0;
        for (int il = 0; il < m.hp.n_layer; ++il) {
            if (m.hp.is_swa(il)) ++pin.n_swa_layers;
            else ++pin.n_full_layers;
        }
        pin.kv_bytes_per_tok = kv_q8
            ? ((size_t) m.hp.kv_dim * 2 + (size_t) m.hp.n_head_kv * sizeof(__half) * 2)
            : ((size_t) m.hp.kv_dim * sizeof(__half) * 2);
        for (int ip = 0; ip < m.hp.n_phys; ++ip) {
            const size_t lb = (size_t) m.hp.n_expert * m.layers[(size_t) ip].stride;
            pin.all_experts_bytes += lb;
            if (ip >= m.hp.core_start && ip < m.hp.core_end) pin.tier1_experts_bytes += lb;
        }
        if (expert_vram >= 0.0) {
            pin.expert_vram_forced = true;
            pin.expert_vram_cap_bytes = (size_t) (expert_vram * (double) (1ull << 30));
        }
    }
    const auto prof_cfg = strata::inkling2::resolve_inkling2_profile(pin);
    chunk = prof_cfg.chunk;
    std::fprintf(stderr, "strata-inkling2: %s\n", strata::inkling2::format_profile_line(pin, prof_cfg).c_str());

    Engine E;
    E.vram_reserve_bytes = (size_t) vram_reserve_mib << 20;
    E.vram_elastic = vram_elastic;
    if (!E.init(m, max_ctx, chunk, kv_q8, err, dm && dm->has_quant_dense, dm ? (draft_max + 1) : 1, &prof_cfg)) {
        std::fprintf(stderr, "strata-inkling2: engine init failed: %s\n", err.c_str());
        return 1;
    }
    if (dm) {
        dspark = std::make_unique<DsparkEngine>();
        dspark->init(*dm, max_ctx, chunk, E.full_kv_cap, E.layer_arena, E.layer_arena_bytes, E.verify_max_t);
        E.enable_dspark(dspark.get());
        E.reset_state();
        // EX-4A: On Hybrid GPUs with chunk >= 1024, evict DSpark Decode-only weights (~358 MiB) during Prefill
        // so place_experts() gets +80 Tier-2 experts (-25% PCIe traffic). Restored in ~14 ms at set_phase_decode()!
        if (prof_cfg.regime != strata::inkling2::VramRegime::FullVram && chunk >= 1024 &&
            std::getenv("STRATA_NO_PHASE_SPLIT") == nullptr) {
            dspark->evict_decode_weights(E.use_phase_pool ? E.phase_pool : nullptr, E.st);
        }
    }
    if (!place_experts(m, E, expert_vram, cpu_threads, err)) {
        std::fprintf(stderr, "strata-inkling2: expert placement failed: %s\n", err.c_str());
        return 1;
    }
    // Safety: if all experts actually fit in VRAM (non-hybrid), restore DSpark decode weights immediately.
    if (dspark && !E.hybrid) {
        dspark->ensure_decode_weights_resident(E.st, E.use_phase_pool ? E.phase_pool : nullptr, E.st);
        CK(cudaStreamSynchronize(E.st));
    }

    if (serve) {
        int32_t eos_id = 200006;
        if (const auto* ev = m.gguf->meta().get("tokenizer.ggml.eos_token_id")) {
            if (ev->is_num()) eos_id = (int32_t) ev->u;
        }
        std::mutex q_mu;
        std::condition_variable q_cv;
        std::deque<std::string> q_lines;
        bool stdin_eof = false;
        std::atomic<bool> stop_req{false};

        std::thread reader([&] {
            std::string line;
            char chunk_buf[65536];
            while (std::fgets(chunk_buf, sizeof chunk_buf, stdin)) {
                line += chunk_buf;
                if (!line.empty() && line.back() != '\n' && !std::feof(stdin)) continue;
                while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
                if (line == "STOP") {
                    stop_req.store(true, std::memory_order_release);
                    line.clear();
                    continue;
                }
                {
                    std::lock_guard<std::mutex> lk(q_mu);
                    q_lines.push_back(std::move(line));
                }
                q_cv.notify_one();
                line.clear();
            }
            {
                std::lock_guard<std::mutex> lk(q_mu);
                stdin_eof = true;
            }
            q_cv.notify_one();
        });

        std::printf("INFO engine=0.2.17-inkling2 context=%d kv=%s kv_resident=%d expert_slots=%d experts=%lld expert_cache_mib=%lld vram_free_mib=%lld pool_workers=%d dspark=%s draft_max=%d vision=%s vram_elastic=%d\n",
                    max_ctx, kv_q8 ? "int8" : "fp16", max_ctx,
                    E.placed_slots, (long long) m.hp.n_phys * m.hp.n_expert,
                    (long long) (E.placed_bytes >> 20), (long long) (E.vram_free_after_place >> 20),
                    E.pool_workers, dspark ? "on" : "off", dspark ? draft_max : 0,
                    vision_enabled ? "on" : "off", E.vram_elastic ? 1 : 0);
        E.prealloc_ckpt_pin();
        std::printf("READY %d stop\n", max_ctx);
        std::fflush(stdout);

        auto fnv1a_bytes = [](const void* data, size_t n) -> uint64_t {
            uint64_t h = 14695981039346656037ULL;
            const auto* p = (const uint8_t*) data;
            for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
            return h ? h : 1ULL;
        };
        auto read_sve1_file = [&](const std::string& path, int expected_embd,
                                  std::vector<float>& out_embs,
                                  std::vector<int>& rec_ntok,
                                  std::vector<uint64_t>& rec_hashes,
                                  std::string& rerr) -> bool {
            out_embs.clear();
            rec_ntok.clear();
            rec_hashes.clear();
            FILE* f = std::fopen(path.c_str(), "rb");
            if (!f) { rerr = "cannot open embeddings file: " + path; return false; }
            int32_t hdr[5];
            while (true) {
                const size_t got = std::fread(hdr, sizeof(int32_t), 5, f);
                if (got == 0 && std::feof(f)) break;
                if (got != 5) { std::fclose(f); rerr = "truncated SVE1 header"; return false; }
                if ((uint32_t) hdr[0] != 0x31455653u) { std::fclose(f); rerr = "bad SVE1 magic"; return false; }
                const int n_tok = hdr[1], n_embd = hdr[4];
                if (n_tok <= 0 || n_embd != expected_embd) {
                    std::fclose(f);
                    rerr = "SVE1 shape mismatch (n_tok=" + std::to_string(n_tok) +
                           ", n_embd=" + std::to_string(n_embd) + ", expected=" + std::to_string(expected_embd) + ")";
                    return false;
                }
                const size_t off = out_embs.size();
                const size_t cnt = (size_t) n_tok * (size_t) n_embd;
                out_embs.resize(off + cnt);
                if (std::fread(out_embs.data() + off, sizeof(float), cnt, f) != cnt) {
                    std::fclose(f);
                    rerr = "truncated SVE1 float payload";
                    return false;
                }
                rec_ntok.push_back(n_tok);
                rec_hashes.push_back(fnv1a_bytes(out_embs.data() + off, cnt * sizeof(float)));
            }
            std::fclose(f);
            if (rec_ntok.empty()) { rerr = "empty SVE1 embeddings file"; return false; }
            return true;
        };

        std::vector<float> lg;
        std::vector<int32_t> hist;
        std::vector<int32_t> live_tokens;
        std::vector<uint64_t> live_img_sigs;
        std::deque<Engine::StateCheckpoint> snaps;
        constexpr size_t kMaxSnaps = 8;
        std::mt19937_64 rng(0x9e3779b97f4a7c15ULL);
        bool got_quit = false;

        for (;;) {
            std::string line;
            {
                std::unique_lock<std::mutex> lk(q_mu);
                q_cv.wait(lk, [&] { return !q_lines.empty() || stdin_eof; });
                if (q_lines.empty()) break;
                line = std::move(q_lines.front());
                q_lines.pop_front();
            }
            if (line == "QUIT" || line == "EXIT") { got_quit = true; break; }
            if (line == "VRAM" || line.rfind("VRAM ", 0) == 0) {
                if (!E.vram_elastic) {
                    std::printf("ERR this engine was started without --vram-elastic\n");
                    std::fflush(stdout);
                    continue;
                }
                size_t req_resv = E.vram_reserve_bytes;
                if (line.size() > 5) {
                    const long long mib = std::atoll(line.c_str() + 5);
                    req_resv = (mib > 0) ? ((size_t) mib << 20) : 0;
                }
                E.trim_phase_pool(0);
                E.resize_experts_vram(req_resv);
                std::printf("VRAM expert_slots=%d experts=%lld expert_cache_mib=%lld expert_cache_full_mib=%lld vram_free_mib=%lld\n",
                            E.placed_slots, (long long) m.hp.n_phys * m.hp.n_expert,
                            (long long) (E.placed_bytes >> 20), (long long) (E.full_cache_bytes >> 20),
                            (long long) (E.vram_free_after_place >> 20));
                std::fflush(stdout);
                continue;
            }
            const bool is_geni = (line.rfind("GENI ", 0) == 0);
            const bool is_gen = (!is_geni && (line.rfind("GEN ", 0) == 0 || line == "GEN"));
            if (is_gen || is_geni) {
                stop_req.store(false, std::memory_order_release);
                const size_t skip = is_geni ? 5 : (line.size() > 4 ? 4 : line.size());
                std::istringstream iss(line.substr(skip));
                int req_new = 0;
                iss >> req_new;
                SamplingParams smp;
                std::vector<int32_t> prompt;
                std::vector<std::string> pos_args;
                std::vector<std::string> img_keys;
                std::string tok;
                while (iss >> tok) {
                    const size_t eq = tok.find('=');
                    if (eq != std::string::npos) {
                        const std::string k = tok.substr(0, eq);
                        const std::string v = tok.substr(eq + 1);
                        if (k == "temperature") smp.temp = std::strtof(v.c_str(), nullptr);
                        else if (k == "top_p") smp.top_p = std::strtof(v.c_str(), nullptr);
                        else if (k == "top_k") smp.top_k = std::atoi(v.c_str());
                        else if (k == "min_p") smp.min_p = std::strtof(v.c_str(), nullptr);
                        else if (k == "penalty_repeat") smp.rep_pen = std::strtof(v.c_str(), nullptr);
                        else if (k == "penalty_freq") smp.freq_pen = std::strtof(v.c_str(), nullptr);
                        else if (k == "penalty_present") smp.pres_pen = std::strtof(v.c_str(), nullptr);
                        else if (k == "penalty_last_n") smp.pen_last_n = std::atoi(v.c_str());
                        else if (k == "seed") smp.seed = (uint64_t) std::strtoull(v.c_str(), nullptr, 10);
                        else if (k == "img_keys") {
                            std::stringstream ks(v);
                            std::string item;
                            while (std::getline(ks, item, ',')) {
                                if (!item.empty()) img_keys.push_back(item);
                            }
                        }
                        continue;
                    }
                    pos_args.push_back(tok);
                }
                std::vector<float> img_embs;
                std::vector<uint64_t> prompt_img_sigs;
                size_t img_embs_n = 0;
                if (is_geni) {
                    if (pos_args.size() < 2) {
                        std::printf("ERR GENI requires embeddings path and token ids\n");
                        std::fflush(stdout);
                        continue;
                    }
                    std::vector<int> rec_ntok;
                    std::vector<uint64_t> rec_hashes;
                    std::string rerr;
                    if (!read_sve1_file(pos_args[0], (int) m.hp.n_embd, img_embs, rec_ntok, rec_hashes, rerr)) {
                        std::printf("ERR %s\n", rerr.c_str());
                        std::fflush(stdout);
                        continue;
                    }
                    for (size_t ai = 1; ai < pos_args.size(); ++ai) {
                        for (int32_t id : parse_id_list(pos_args[ai])) prompt.push_back(id);
                    }
                    const bool use_keys = (img_keys.size() == rec_ntok.size());
                    for (size_t r = 0; r < rec_ntok.size(); ++r) {
                        const uint64_t sig = use_keys
                            ? fnv1a_bytes(img_keys[r].data(), img_keys[r].size())
                            : rec_hashes[r];
                        prompt_img_sigs.insert(prompt_img_sigs.end(), (size_t) rec_ntok[r], sig);
                    }
                    img_embs_n = prompt_img_sigs.size();
                    size_t n_pads = 0;
                    for (int32_t id : prompt) if (id == img_pad_id) ++n_pads;
                    if (n_pads != img_embs_n) {
                        std::printf("ERR image token count mismatch: prompt has %zu pad tokens (id=%d), embeddings has %zu\n",
                                    n_pads, (int) img_pad_id, img_embs_n);
                        std::fflush(stdout);
                        continue;
                    }
                } else {
                    for (const auto& pa : pos_args) {
                        for (int32_t id : parse_id_list(pa)) prompt.push_back(id);
                    }
                }
                if (smp.seed > 0) rng.seed(smp.seed);
                if (prompt.empty()) {
                    std::printf("DONE 0 0 0 0 stop\n");
                    std::fflush(stdout);
                    continue;
                }
                if ((int) prompt.size() >= max_ctx) {
                    std::printf("ERR prompt (%zu) exceeds max_context (%d)\n", prompt.size(), max_ctx);
                    std::fflush(stdout);
                    continue;
                }

                // Prefix / KV cache reuse along live_tokens & saved SWA/conv checkpoints (image-signature aware)
                const size_t max_possible = prompt.size() - 1;
                size_t lcp_live = 0;
                size_t pad_seen = 0;
                const size_t lim_live = std::min(live_tokens.size(), max_possible);
                while (lcp_live < lim_live && live_tokens[lcp_live] == prompt[lcp_live]) {
                    if (prompt[lcp_live] == img_pad_id) {
                        if (pad_seen >= live_img_sigs.size() || pad_seen >= prompt_img_sigs.size() ||
                            live_img_sigs[pad_seen] != prompt_img_sigs[pad_seen]) {
                            break;
                        }
                        ++pad_seen;
                    }
                    ++lcp_live;
                }

                size_t reuse = 0;
                if (lcp_live == live_tokens.size()) {
                    reuse = lcp_live;
                } else {
                    int best_idx = -1;
                    for (int si = (int) snaps.size() - 1; si >= 0; --si) {
                        const auto& stoks = snaps[(size_t) si].tokens;
                        const auto& ssigs = snaps[(size_t) si].img_sigs;
                        if (stoks.size() <= lcp_live &&
                            std::equal(stoks.begin(), stoks.end(), prompt.begin()) &&
                            ssigs.size() <= prompt_img_sigs.size() &&
                            std::equal(ssigs.begin(), ssigs.end(), prompt_img_sigs.begin())) {
                            best_idx = si;
                            break;
                        }
                    }
                    if (best_idx >= 0) {
                        E.restore_checkpoint(snaps[(size_t) best_idx]);
                        live_tokens = snaps[(size_t) best_idx].tokens;
                        live_img_sigs = snaps[(size_t) best_idx].img_sigs;
                        snaps.resize((size_t) best_idx + 1);
                        reuse = live_tokens.size();
                    } else {
                        E.flush_ckpt_bg();
                        E.reset_state();
                        live_tokens.clear();
                        live_img_sigs.clear();
                        snaps.clear();
                        reuse = 0;
                    }
                }
                std::printf("RESUME %zu\n", reuse);
                std::fflush(stdout);

                size_t img_cursor = 0;
                if (is_geni) {
                    for (size_t i = 0; i < reuse; ++i) {
                        if (prompt[i] == img_pad_id) ++img_cursor;
                    }
                }

                const auto t_pp0 = Clock::now();
                E.hy_gpu = 0;
                E.hy_cpu = 0;
                E.hy_cpu_ms = 0.0;
                hist = prompt;
                const bool sample_on = smp.needs_logits();
                const bool use_gpu_topk_pp = sample_on && can_use_gpu_topk(smp, (int) m.hp.vocab_unpadded);
                int32_t next_tok = 0;
                const size_t fresh = prompt.size() - reuse;
                const int avail = std::max(0, max_ctx - (int) prompt.size());
                const int limit = req_new > 0 ? std::min(req_new, avail) : avail;
                int eff_chunk = chunk;
                if (E.in_decode_phase && fresh <= 64) {
                    eff_chunk = Engine::kDecodePhaseCap;
                } else {
                    int want_cap = chunk;
                    if (fresh > (size_t) chunk && E.profile.max_adaptive_chunk > chunk) {
                        if (fresh <= 4096u) {
                            want_cap = std::max(chunk, std::min(4096, E.profile.max_adaptive_chunk));
                        } else if (fresh <= 8192u) {
                            want_cap = std::max(chunk, std::min(8192, E.profile.max_adaptive_chunk));
                        } else if (reuse == 0 && fresh <= 12288u && E.profile.max_adaptive_chunk >= 12288) {
                            want_cap = 12288;
                        } else {
                            want_cap = std::max(chunk, std::min(8192, E.profile.max_adaptive_chunk));
                        }
                    }
                    eff_chunk = E.ensure_pp_cap(want_cap, (int) prompt.size());
                }
                for (size_t off = reuse; off < prompt.size() && !stop_req.load(std::memory_order_acquire); off += (size_t) eff_chunk) {
                    const int tc = (int) std::min<size_t>((size_t) eff_chunk, prompt.size() - off);
                    const bool is_last = (off + (size_t) tc == prompt.size());
                    const int pos_pre = (int) E.pos;
                    const size_t img_cur_pre = img_cursor;
                    const bool will_dec_last = is_last && E.will_phase_decode(limit, dspark != nullptr, tc);
                    next_tok = E.forward(prompt.data() + off, tc, is_last && sample_on,
                                         (is_last && use_gpu_topk_pp) ? nullptr : &lg, false, dspark != nullptr, 0, nullptr,
                                         is_last, is_last && use_gpu_topk_pp,
                                         is_geni ? img_embs.data() : nullptr, img_embs_n,
                                         is_geni ? img_pad_id : -1, is_geni ? &img_cursor : nullptr,
                                         will_dec_last);
                    if (is_last && use_gpu_topk_pp) {
                        std::vector<SamplingDist> d0;
                        build_sampling_dists_gpu(E, 1, hist, nullptr, 0, smp, d0);
                        next_tok = sample_from_dist(d0[0], rng);
                    }
                    if (dspark && !E.dspark_injected_in_fwd) dspark->inject(E, pos_pre, tc);
                    live_tokens.insert(live_tokens.end(), prompt.begin() + (ptrdiff_t) off, prompt.begin() + (ptrdiff_t) (off + (size_t) tc));
                    if (is_geni && img_cursor > img_cur_pre) {
                        live_img_sigs.insert(live_img_sigs.end(),
                                             prompt_img_sigs.begin() + (ptrdiff_t) img_cur_pre,
                                             prompt_img_sigs.begin() + (ptrdiff_t) img_cursor);
                    }
                    if (E.hybrid && !will_dec_last) {
                        if (E.rebal_burst > 0) { E.rebalance(192, 16, !E.pp_streaming); --E.rebal_burst; }
                        else if ((++E.pp_chunk_cnt & 3) == 0) E.rebalance(16, 2, !E.pp_streaming);
                    }
                    const double el_ms = ms_since(t_pp0);
                    const double done_fresh = (double) (off + (size_t) tc - reuse);
                    const double pp_tps_now = done_fresh / std::max(1e-6, el_ms / 1000.0);
                    std::printf("PP %zu %zu %.0f %.1f\n", off + (size_t) tc, prompt.size(), el_ms, pp_tps_now);
                    std::fflush(stdout);
                    if (!is_last && !stop_req.load(std::memory_order_acquire)) {
                        const size_t last_cp = snaps.empty() ? 0 : snaps.back().tokens.size();
                        if (live_tokens.size() - last_cp >= 8192) {
                            E.flush_ckpt_bg();
                            if (snaps.size() >= kMaxSnaps) snaps.pop_front();
                            snaps.emplace_back();
                            E.save_checkpoint_into(snaps.back(), live_tokens, live_img_sigs);
                        }
                    }
                }
                const double pp_ms = ms_since(t_pp0);
                E.flush_deferred_graphs();
                if (sample_on && !use_gpu_topk_pp && !stop_req.load(std::memory_order_acquire)) {
                    next_tok = sample_logits(lg, hist, smp, rng);
                }
                bool saved_prompt_cp = !(!stop_req.load(std::memory_order_acquire) && live_tokens.size() == prompt.size() &&
                                         (snaps.empty() || snaps.back().tokens.size() != live_tokens.size()));
                auto maybe_save_prompt_cp = [&]() {
                    if (saved_prompt_cp) return;
                    saved_prompt_cp = true;
                    E.flush_ckpt_bg();
                    if (snaps.size() >= kMaxSnaps) snaps.pop_front();
                    snaps.emplace_back();
                    E.save_checkpoint_into(snaps.back(), live_tokens, live_img_sigs);
                };

                const auto t_tg0 = Clock::now();
                int emitted = 0;
                const char* finish = "length";
                if (stop_req.load(std::memory_order_acquire)) {
                    finish = "cancel";
                } else {
                    if (E.will_phase_decode(limit, dspark != nullptr)) E.set_phase_decode();
                    bool hit_eos = false;
                    decode_tokens_loop(
                        E, dspark.get(), draft_max, next_tok, limit, smp, rng, hist, live_tokens,
                        [&](int32_t tok) -> bool {
                            std::printf("T %d\n", (int) tok);
                            std::fflush(stdout);
                            // Save end-of-prompt checkpoint AFTER flushing token 0 so TTFT never waits on checkpoint D2H!
                            if (emitted == 0) maybe_save_prompt_cp();
                            ++emitted;
                            hist.push_back(tok);
                            if (tok == eos_id || tok == 199999) return false;
                            return true;
                        },
                        [&]() -> bool { return stop_req.load(std::memory_order_acquire); },
                        &hit_eos);
                    maybe_save_prompt_cp();
                    if (stop_req.load(std::memory_order_acquire)) finish = "cancel";
                    else if (hit_eos) finish = "stop";
                }
                const double tg_ms = ms_since(t_tg0);
                const long long lookups = (long long) E.hy_gpu + (long long) E.hy_cpu;
                std::printf("DONE %d %zu %.2f %.2f %s 0 0 %zu %lld %lld 0 0 0.0 %zu\n",
                            emitted, prompt.size(), pp_ms, tg_ms, finish,
                            reuse, (long long) E.hy_gpu, lookups, fresh);
                std::fflush(stdout);
            }
        }
        E.flush_ckpt_bg();
        if (reader.joinable()) {
            if (got_quit) reader.detach();
            else reader.join();
        }
        return 0;
    }

    std::vector<int32_t> prompt = parse_id_list(tokens_str);
    if (prompt.empty()) prompt = {1};

    int want_cap_cli = chunk;
    if (prompt.size() > (size_t) chunk && E.profile.max_adaptive_chunk > chunk) {
        if (prompt.size() <= 8192u && E.profile.max_adaptive_chunk >= 8192) {
            want_cap_cli = std::max(chunk, 8192);
        } else if (prompt.size() > 8192u && E.profile.max_adaptive_chunk > 8192) {
            const size_t rounded = ((prompt.size() + 4095u) / 4096u) * 4096u;
            want_cap_cli = std::min<int>(E.profile.max_adaptive_chunk,
                                         (int) std::min<size_t>(rounded, (size_t) E.profile.max_adaptive_chunk));
        } else {
            want_cap_cli = E.profile.max_adaptive_chunk;
        }
    }
    const int eff_chunk_cli = E.ensure_pp_cap(want_cap_cli, (int) prompt.size());

    if (bench) {
        // Warmup 1 pass
        E.reset_state();
        int32_t w = 0;
        for (size_t off = 0; off < prompt.size(); off += (size_t) eff_chunk_cli) {
            const int tc = (int) std::min<size_t>((size_t) eff_chunk_cli, prompt.size() - off);
            const int p0 = (int) E.pos;
            const bool is_last_w = (off + (size_t) tc == prompt.size());
            w = E.forward(prompt.data() + off, tc, false, nullptr, false, dspark != nullptr, 0, nullptr, is_last_w);
            if (dspark && !E.dspark_injected_in_fwd) dspark->inject(E, p0, tc);
        }
        if (E.hybrid) E.rebalance(192, 16, !E.pp_streaming);
        if (max_new > 0) {
            if (dspark && max_new >= 2) {
                if (E.hybrid) E.set_phase_decode();
                std::vector<int32_t> wd;
                dspark->draft(E, w, std::min(draft_max, max_new - 1), wd, nullptr, [&](int, std::vector<float>&) -> int32_t { return 0; });
                if (E.in_decode_phase) E.set_phase_prefill();
            } else {
                E.forward(&w, 1);
            }
        }
        E.reset_state();
        E.hy_gpu = E.hy_cpu = 0;
        E.hy_cpu_ms = 0.0;
        if (dspark) {
            dspark->n_draft_calls = dspark->n_drafted = dspark->n_accepted = 0;
            dspark->pld_calls = dspark->pld_drafted = dspark->pld_accepted = 0;
            dspark->draft_ms = dspark->inject_ms = 0.0;
        }
    }

    E.prof = profile;
    E.prof_detail = profile_detail;
    E.reset_prof();
    std::vector<float> dumped;
    const auto t_pp0 = Clock::now();
    int32_t next_tok = 0;
    for (size_t off = 0; off < prompt.size(); off += (size_t) eff_chunk_cli) {
        const int tc = (int) std::min<size_t>((size_t) eff_chunk_cli, prompt.size() - off);
        const bool is_last = (off + (size_t) tc == prompt.size());
        const int p0 = (int) E.pos;
        const bool will_dec_last = is_last && E.will_phase_decode(max_new, dspark != nullptr, tc);
        next_tok = E.forward(prompt.data() + off, tc, is_last && !dump_logits_path.empty(), &dumped,
                             false, dspark != nullptr, 0, nullptr, is_last, false,
                             nullptr, 0, -1, nullptr, will_dec_last);
        if (dspark && !E.dspark_injected_in_fwd) dspark->inject(E, p0, tc);
        const auto tr0 = Clock::now();
        if (E.hybrid && !will_dec_last) E.rebalance(8, 1, !E.pp_streaming);
        if (profile) { CK(cudaStreamSynchronize(E.st)); E.p_rebal_ms += ms_since(tr0); }
    }
    const double pp_ms = ms_since(t_pp0);
    E.flush_deferred_graphs();
    if (profile) { E.print_prof("pp"); E.reset_prof(); }

    if (!dump_logits_path.empty() && !dumped.empty()) {
        std::FILE* f = std::fopen(dump_logits_path.c_str(), "wb");
        if (f) {
            std::fwrite(dumped.data(), sizeof(float), dumped.size(), f);
            std::fclose(f);
            std::fprintf(stderr, "strata-inkling2: wrote %zu logits to %s\n", dumped.size(), dump_logits_path.c_str());
        }
    }

    std::vector<int32_t> gen;
    gen.reserve((size_t) max_new);
    std::vector<int32_t> cli_hist = prompt;
    std::vector<int32_t> cli_live = prompt;
    SamplingParams cli_smp;
    std::mt19937_64 cli_rng(0x9e3779b97f4a7c15ULL);

    const auto t_tg0 = Clock::now();
    if (max_new > 0) {
        if (E.will_phase_decode(max_new, dspark != nullptr)) E.set_phase_decode();
        decode_tokens_loop(
            E, dspark.get(), draft_max, next_tok, max_new, cli_smp, cli_rng, cli_hist, cli_live,
            [&](int32_t tok) -> bool {
                gen.push_back(tok);
                return true;
            },
            [&]() -> bool { return false; });
    }
    const double tg_ms = ms_since(t_tg0);
    if (profile) {
        E.print_prof("tg");
        if (dspark) {
            std::fprintf(stderr, "PROFILE dspark: draft_calls=%lld drafted=%lld acc=%lld draft=%.1f ms inject=%.1f ms out_type=%d\n",
                         (long long) dspark->n_draft_calls, (long long) dspark->n_drafted, (long long) dspark->n_accepted,
                         dspark->draft_ms, dspark->inject_ms, m.output_weight.type);
        }
    }

    std::printf("tokens:");
    for (int32_t id : gen) std::printf(" %d", (int) id);
    std::printf("\n");

    const double pp_tps = (double) prompt.size() / (pp_ms / 1000.0);
    const double tg_tps = gen.size() > 1 ? (double) (gen.size() - 1) / (tg_ms / 1000.0) : 0.0;
    if (dspark) {
        const int64_t tot_acc = dspark->n_accepted + dspark->pld_accepted;
        const int64_t tot_drf = dspark->n_drafted + dspark->pld_drafted;
        const int64_t tot_cls = dspark->n_draft_calls + dspark->pld_calls;
        const double acc_rate = tot_drf > 0 ? (100.0 * (double) tot_acc / (double) tot_drf) : 0.0;
        const double mean_len = tot_cls > 0 ? (1.0 + (double) tot_acc / (double) tot_cls) : 1.0;
        const double ds_rate = dspark->n_drafted > 0 ? (100.0 * (double) dspark->n_accepted / (double) dspark->n_drafted) : 0.0;
        std::fprintf(stderr,
                     "strata-inkling2: pp %zu tok in %.1f ms (%.2f t/s) | tg %d tok in %.1f ms (%.2f t/s) | dspark acc=%lld/%lld (%.1f%%, %.2f tok/step; neural=%lld/%lld %.1f%% pld=%lld/%lld) | hits=%lld misses=%lld cpu_ms=%.1f\n",
                     prompt.size(), pp_ms, pp_tps, std::max(0, (int) gen.size() - 1), tg_ms, tg_tps,
                     (long long) tot_acc, (long long) tot_drf, acc_rate, mean_len,
                     (long long) dspark->n_accepted, (long long) dspark->n_drafted, ds_rate,
                     (long long) dspark->pld_accepted, (long long) dspark->pld_drafted,
                     (long long) E.hy_gpu, (long long) E.hy_cpu, E.hy_cpu_ms);
    } else {
        std::fprintf(stderr, "strata-inkling2: pp %zu tok in %.1f ms (%.2f t/s) | tg %d tok in %.1f ms (%.2f t/s) | hits=%lld misses=%lld cpu_ms=%.1f\n",
                     prompt.size(), pp_ms, pp_tps, std::max(0, (int) gen.size() - 1), tg_ms, tg_tps,
                     (long long) E.hy_gpu, (long long) E.hy_cpu, E.hy_cpu_ms);
    }
    return 0;
}

}  // namespace strata::inkling2

