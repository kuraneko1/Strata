// src/inkling2/inkling2_engine.cu - Standalone / integrated CUDA + AVX2 hybrid engine for Inkling-2-Mini (arch inkling2).
#include "strata/inkling2/engine.hpp"
#include "strata/inkling2/profile_rules.hpp"
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
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace strata::inkling2 {
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
    DenseMat attn_qgr;        // concatenated [attn_q; attn_gate; attn_rel] for T==1 single MMVQ
    DenseMat attn_qgrkv;      // concatenated [attn_q; attn_gate; attn_rel; attn_k; attn_v] for SWA !has_msg T==1
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
    float* attn_k_conv = nullptr;        // [4, kv_dim] -> c*4 + lag
    float* attn_v_conv = nullptr;
    float* attn_post_conv = nullptr;     // [4, n_embd]
    float* ffn_post_conv = nullptr;      // [4, n_embd]
    float* attn_res_query = nullptr;     // [n_embd, 1 or 3] (null on phys 0)
    float* ffn_res_query = nullptr;      // [n_embd]
    float* ffn_norm = nullptr;           // [n_embd]
    float* ffn_gate_inp = nullptr;       // [n_embd, n_expert]
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
    uint8_t* d_blobs = nullptr;          // [n_slots][stride] in VRAM
    size_t stride = 0;
    int n_slots = 0;
    int32_t* d_slot_of = nullptr;        // [n_expert] in VRAM
    uint32_t* d_hits = nullptr;          // [n_expert] GPU routing frequency counter (even when n_slots == ne)
    std::vector<int32_t> slot_of;        // [n_expert] on host (-1 = CPU)
    std::vector<int32_t> exp_in_slot;    // [n_slots]
    std::vector<double> score;           // [n_expert]
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

    ~Model() override { free_store(); }

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

void build_expert_blob(const PhysLayer& L, int64_t e, uint8_t* dst) {
    const int64_t n_ff = L.L.n_ff, n_embd = L.L.n_embd;
    const size_t gu_expert_bytes = (size_t) (2 * n_ff) * L.gu_row;
    const size_t d_expert_bytes = (size_t) n_embd * L.d_row;
    const uint8_t* gu_src = L.h_gate_up + (size_t) e * gu_expert_bytes;
    const uint8_t* d_src = L.h_down + (size_t) e * d_expert_bytes;
    // [gate (n_ff rows) | up (n_ff rows) | down (n_embd rows)]
    std::memcpy(dst, gu_src, (size_t) n_ff * L.gu_row);
    std::memcpy(dst + L.L.up_off, gu_src + (size_t) n_ff * L.gu_row, (size_t) n_ff * L.gu_row);
    std::memcpy(dst + L.L.down_off, d_src, d_expert_bytes);
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
    m.tok_embd_norm = m.upload_f32("token_embd_norm.weight", (size_t) hp.n_embd);
    if (hp.num_loops > 1) {
        m.loop_res_query = m.upload_f32("loop_res_query.weight", (size_t) hp.n_embd * 2 * n_core);
        m.loop_kv_res_query = m.upload_f32("loop_kv_res_query.weight", (size_t) hp.n_embd * 2 * n_core);
    }
    m.output_res_query = m.upload_f32("output_res_query.weight", (size_t) hp.n_embd);
    m.output_norm = m.upload_f32("output_norm.weight", (size_t) hp.n_embd);
    const bool pack_head = std::getenv("STRATA_Q8_HEAD_PACK") != nullptr && std::getenv("STRATA_Q8_HEAD_PACK")[0] == '1';
    m.output_weight = m.upload_dense("output.weight", (size_t) hp.vocab_unpadded, (size_t) hp.n_embd, pack_head, false);

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
            uint8_t* dall = m.gpu_alloc<uint8_t>(bq + bg + br + bk + bv);
            CK(cudaMemcpy(dall, raw_q, bq, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dall + bq, raw_g, bg, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dall + bq + bg, raw_r, br, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dall + bq + bg + br, raw_k, bk, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dall + bq + bg + br + bk, raw_v, bv, cudaMemcpyHostToDevice));
            m.has_quant_dense = true;
            L.attn_q = DenseMat{(int) tq->type, nin, nq_out, dall, bq};
            L.attn_gate = DenseMat{(int) tg->type, nin, ng_out, dall + bq, bg};
            L.attn_rel = DenseMat{(int) tr->type, nin, nr_out, dall + bq + bg, br};
            L.attn_k = DenseMat{(int) tk->type, nin, nk_out, dall + bq + bg + br, bk};
            L.attn_v = DenseMat{(int) tv->type, nin, nv_out, dall + bq + bg + br + bk, bv};
            L.attn_qgr = DenseMat{(int) tq->type, nin, nq_out + ng_out + nr_out, dall, bq + bg + br};
            L.attn_qgrkv = DenseMat{(int) tq->type, nin, nq_out + ng_out + nr_out + nk_out + nv_out, dall, bq + bg + br + bk + bv};
        } else if (qgr_ok) {
            const int64_t nin = (int64_t) tq->shape[0];
            const size_t rb = strata::quant::row_bytes((int) tq->type, nin);
            const int64_t nq_out = tq->shape.size() > 1 ? (int64_t) tq->shape[1] : 1;
            const int64_t ng_out = tg->shape.size() > 1 ? (int64_t) tg->shape[1] : 1;
            const int64_t nr_out = tr->shape.size() > 1 ? (int64_t) tr->shape[1] : 1;
            const size_t bq = rb * (size_t) nq_out;
            const size_t bg = rb * (size_t) ng_out;
            const size_t br = rb * (size_t) nr_out;
            uint8_t* dqgr = m.gpu_alloc<uint8_t>(bq + bg + br);
            CK(cudaMemcpy(dqgr, raw_q, bq, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dqgr + bq, raw_g, bg, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dqgr + bq + bg, raw_r, br, cudaMemcpyHostToDevice));
            m.has_quant_dense = true;
            L.attn_q = DenseMat{(int) tq->type, nin, nq_out, dqgr, bq};
            L.attn_gate = DenseMat{(int) tg->type, nin, ng_out, dqgr + bq, bg};
            L.attn_rel = DenseMat{(int) tr->type, nin, nr_out, dqgr + bq + bg, br};
            L.attn_qgr = DenseMat{(int) tq->type, nin, nq_out + ng_out + nr_out, dqgr, bq + bg + br};
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
        L.attn_rel_bias = m.upload_f32(tn("attn_rel_bias.weight"), (size_t) hp.n_rel * hp.n_rel_ext);
        L.attn_k_conv = m.upload_f32(tn("attn_k_conv.weight"), (size_t) hp.n_conv * hp.kv_dim);
        L.attn_v_conv = m.upload_f32(tn("attn_v_conv.weight"), (size_t) hp.n_conv * hp.kv_dim);
        L.attn_post_conv = m.upload_f32(tn("attn_post_conv.weight"), (size_t) hp.n_conv * hp.n_embd);
        L.ffn_post_conv = m.upload_f32(tn("ffn_post_conv.weight"), (size_t) hp.n_conv * hp.n_embd);
        if (ip > 0) L.attn_res_query = m.upload_f32(tn("attn_res_query.weight"), (size_t) hp.n_embd * (L.is_swa ? 1 : 3));
        L.ffn_res_query = m.upload_f32(tn("ffn_res_query.weight"), (size_t) hp.n_embd);
        L.ffn_norm = m.upload_f32(tn("ffn_norm.weight"), (size_t) hp.n_embd);
        L.ffn_gate_inp = m.upload_f32(tn("ffn_gate_inp.weight"), (size_t) hp.n_embd * hp.n_expert);
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

#if defined(STRATA_INKLING2_CPU_POOL)
        if (!strata::kernels::cpu::native_fmt(L.gu_type, L.d_type, hp.n_embd, hp.n_ff, L.cpu_fmt, err)) {
            return false;
        }
        L.cpu_fmt.situ = true;
        L.cpu_fmt.situ_beta = hp.situ_beta;
        L.cpu_fmt.situ_linear_beta = hp.situ_linear_beta;
#endif
    }
    std::fprintf(stderr, "strata-inkling2: dense + metadata loaded in %.2f s (n_phys=%lld, n_virt=%lld)\n",
                 ms_since(t0) / 1000.0, (long long) hp.n_phys, (long long) hp.n_layer);
    return true;
}

struct Engine {
    Model* m = nullptr;
    cudaStream_t st = nullptr;
    strata::prefill::Gemm gemm;

    int ctx = 16384;
    int full_kv_cap = 16384;
    int swa_kv_cap = 1024;
    int tmax = 512;
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
    float* d_tgt_feat[4] = {};       // [tmax * N] captured at dspark.target_layers
    float* h_tgt_feat_pin[4] = {};   // non-null when d_tgt_feat is zero-copy mapped pinned RAM (Hybrid VRAM save)
    double p_draft_ms = 0.0;

    // Activation scratch (all allocated up to tmax rows; phase-disjoint buffers share layer_arena)
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
    float *norm_cur = nullptr;       // [tmax * N]
    float *prelude_cur_buf = nullptr;
    float *norm_blk_buf[9] = {};     // cached RMSNorm of pushed block residuals (5 ping-pong slots for standard Inkling-2)
    int norm_blk_cnt = 9;

    float *Q = nullptr, *K = nullptr, *V = nullptr, *R = nullptr;
    float *qgr_t1 = nullptr;         // [2 * q_dim + n_rel * n_head] contiguous T=1 fused output
    float *K_conv = nullptr, *V_conv = nullptr, *dyn_k = nullptr, *dyn_v = nullptr;
    float *sink = nullptr, *rel_tab = nullptr, *att = nullptr, *att_gate = nullptr, *wo = nullptr;
    float *attn_split = nullptr;     // [verify_max_t * kAttnSplitMax * n_head * 130] for split GQA-3 decode/verify
    int attn_split_slots = 0;        // total (t, sp) slots in attn_split; every launch MUST satisfy T * sp <= attn_split_slots
    int sm_count = 80;
    int fa2_256_bulk_smem = 87040;
    int fa2_256_tail_smem = 52224;
    int fa2_192_swa_smem = 69632;
    bool can_fa2_192 = false;
    bool can_fa2_192_swa_fast = false;
    bool can_ctrl_wmma = false;
    bool can_fa2_256 = false;
    bool bf16_x_f16 = false;
    bool has_bf16_dense = false;
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
    float *logits = nullptr;
    float* h_tok_buf = nullptr;      // pinned [tmax * N] FP32 for token embedding upload
    uint8_t* h_is_img_pin = nullptr; // pinned [tmax] flag (1 = vision embedding, skip tok_embd_norm)
    uint8_t* d_is_img = nullptr;     // device [tmax] flag for k_rms_norm_masked
    size_t vram_reserve_bytes = 0;   // --vram-reserve-mib headroom (e.g. for GPU strata-vision)
    bool vram_elastic = false;       // --vram-elastic: dynamic expert cache shrink/grow via VRAM command
    size_t full_cache_bytes = 0;     // full expert cache size (for VRAM command reporting)
    uint8_t* ckpt_pin = nullptr;     // reusable pinned staging buffer for fast SWA/conv checkpoint D2H
    size_t ckpt_pin_cap = 0;
    uint16_t* act_bf16 = nullptr;    // FP16 scratch [max(tmax*2*N, 16*4*N)]
    uint16_t* act_bf16_only = nullptr; // Dedicated BF16 scratch on sm_75+ (!bf16_x_f16) so BF16 projections never clobber FP16 act_bf16

    void *xq = nullptr, *xq_k = nullptr, *xq_v = nullptr, *escr = nullptr;
    size_t escr_bytes = 0;
    int32_t *ids = nullptr, *grp_start = nullptr, *n_groups = nullptr, *ent_dst = nullptr, *ent_tok = nullptr;
    uint32_t* d_router_done = nullptr;
    unsigned long long* grp_ptr = nullptr;
    int32_t *pos0_d = nullptr, *pick_d = nullptr;
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
    float* h_h = nullptr;            // [tmax * N]
    float* cpu_out = nullptr;        // [tmax * top_k * N]
    int32_t* h_miss = nullptr;       // [tmax * top_k]
    int32_t* d_miss = nullptr;
    float* ecpu = nullptr;           // [tmax * top_k * N]
    uint8_t* acts = nullptr;         // [tmax * kNativeActBytes]
    cudaEvent_t ev_route = nullptr;
    bool hybrid = false;
    int rebal_burst = 2;
    int pp_chunk_cnt = 0;
    size_t pinned_used = 0;
    size_t max_pin_bytes = (size_t) 7168 << 20;
    int cpu_threads_req = 0;

    // Double-buffered PCIe GPU streaming for prefill on hybrid layers
    struct PpRun {
        int16_t e0 = 0;
        int16_t cnt = 0;
        int16_t dst_off = 0;
    };
    struct PpTask {
        int ip = -1;
        int n_staged = 0;
        PpStageMap map{};
        std::vector<PpRun> runs;
    };
    static constexpr int kMaxPpStages = 8;
    cudaStream_t cst = nullptr;
    uint8_t* pp_stage[kMaxPpStages] = {};
    int pp_stage_cnt = 0;
    size_t pp_stage_bytes = 0;
    cudaEvent_t ev_pp_copy[kMaxPpStages] = {};
    cudaEvent_t ev_pp_used[kMaxPpStages] = {};
    static constexpr int kMaxRebalLayers = 48;
    cudaEvent_t ev_rebal_layer[kMaxRebalLayers] = {};
    bool layer_pending_rebal[kMaxRebalLayers] = {};
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
    cudaGraphExec_t verify_core_graph[9] = {};
    // Full 48-layer + swa_bak + head + argmax/softcap graphs for Full-VRAM (T=1..8, capture_dspark, want_softcap, want_argmax)
    cudaGraphExec_t full_decode_graph[9][2][2][2] = {};
    std::vector<cudaGraphExec_t> layer_pre_graph;
    std::vector<cudaGraphExec_t> verify_pre_graph[9];
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

    void free_ephemeral_bufs() {
        for (int b = 0; b < 9; ++b) {
            if (norm_blk_buf[b]) { cudaFree(norm_blk_buf[b]); norm_blk_buf[b] = nullptr; }
        }
        if (prelude_cur_buf) { cudaFree(prelude_cur_buf); prelude_cur_buf = nullptr; }
        if (cur) { cudaFree(cur); cur = nullptr; }
        if (ffn_inp) { cudaFree(ffn_inp); ffn_inp = nullptr; }
        if (msg3) { cudaFree(msg3); msg3 = nullptr; }
        if (sink) { cudaFree(sink); sink = nullptr; }
        if (router_logits) { cudaFree(router_logits); router_logits = nullptr; }
        if (ewts) { cudaFree(ewts); ewts = nullptr; }
        if (act_bf16) { cudaFree(act_bf16); act_bf16 = nullptr; }
        act_bf16_only = nullptr;
        if (layer_arena) { cudaFree(layer_arena); layer_arena = nullptr; }
        layer_arena_bytes = 0;
    }

    size_t alloc_ephemeral_bufs(int cap_t) {
        const auto& hp = m->hp;
        const int N = (int) hp.n_embd;
        const int64_t max_ent = (int64_t) cap_t * hp.n_expert_used;
        auto d_f32 = [](float** p, size_t n) { CK(cudaMalloc(p, n * sizeof(float))); };
        auto a256 = [](size_t n) -> size_t { return (n + 255) & ~(size_t) 255; };
        size_t total_b = 0;

        for (int b = 0; b < norm_blk_cnt; ++b) {
            d_f32(&norm_blk_buf[b], (size_t) cap_t * N);
            total_b += (size_t) cap_t * N * sizeof(float);
        }
        d_f32(&prelude_cur_buf, (size_t) cap_t * N); total_b += (size_t) cap_t * N * sizeof(float);
        d_f32(&cur, (size_t) cap_t * N);             total_b += (size_t) cap_t * N * sizeof(float);
        d_f32(&ffn_inp, (size_t) cap_t * N);         total_b += (size_t) cap_t * N * sizeof(float);
        d_f32(&msg3, (size_t) cap_t * N);            total_b += (size_t) cap_t * N * sizeof(float);
        d_f32(&sink, (size_t) cap_t * hp.n_head);    total_b += (size_t) cap_t * hp.n_head * sizeof(float);
        d_f32(&router_logits, (size_t) cap_t * hp.n_expert); total_b += (size_t) cap_t * hp.n_expert * sizeof(float);
        d_f32(&ewts, (size_t) max_ent);              total_b += (size_t) max_ent * sizeof(float);

        const size_t act_elems = std::max<size_t>((size_t) cap_t * 2 * (size_t) N, (size_t) 16 * 4 * (size_t) N);
        CK(cudaMalloc(&act_bf16, act_elems * sizeof(uint16_t)));
        total_b += act_elems * sizeof(uint16_t);

        const size_t sz_N = a256((size_t) cap_t * (size_t) N * sizeof(float));
        const size_t sz_2N = a256((size_t) cap_t * 2 * (size_t) N * sizeof(float));
        const size_t sz_q = a256((size_t) cap_t * (size_t) hp.q_dim * sizeof(float));
        const size_t sz_kv = a256((size_t) cap_t * (size_t) hp.kv_dim * sizeof(float));
        const size_t sz_rel = a256((size_t) cap_t * (size_t) (hp.n_rel * hp.n_head) * sizeof(float));
        const size_t sz_dyn = a256((size_t) cap_t * (size_t) (hp.n_conv * hp.n_head_kv) * sizeof(float));
        const size_t sz_c64 = a256((size_t) cap_t * (size_t) hp.loop_gate_rank * sizeof(float));
        ctrl_w16_bytes = a256((size_t) (hp.loop_gate_rank * 2 * N + 2 * N * hp.loop_gate_rank) * sizeof(uint16_t));
        const size_t sz_reltab = a256((size_t) cap_t * (size_t) hp.n_head * (size_t) hp.n_rel_ext * sizeof(float));

        const size_t reg0_qkv = 3 * sz_N + 4 * sz_kv + sz_rel + 2 * sz_dyn;
        const size_t per_sp_bytes = (size_t) cap_t * (size_t) hp.n_head * 130 * sizeof(float);
        const size_t attn_split_bytes = (size_t) attn_split_slots * (size_t) hp.n_head * 130 * sizeof(float);
        const size_t want_sp = (size_t) std::max(2, profile.want_sp);
        const size_t reg0_bytes = a256(std::max({reg0_qkv, want_sp * per_sp_bytes, attn_split_bytes}));
        const size_t reg1_bytes = a256(std::max(3 * sz_N, 3 * sz_q));
        const size_t reg2_ctrl = sz_2N + 3 * sz_N + sz_c64 + ctrl_w16_bytes;
        const size_t reg2_bytes = a256(std::max(reg2_ctrl, sz_reltab));
        const size_t overlay_a_bytes = reg0_bytes + reg1_bytes + reg2_bytes;
        const size_t overlay_a_ctrl_inffn = reg0_bytes + reg1_bytes + reg2_ctrl + sz_N;

        eparts_bytes = a256((size_t) max_ent * (size_t) N * sizeof(float));
        bool all_situ_fused = (hp.n_ff % 64 == 0 && hp.n_expert <= 256);
        for (const auto& L : m->layers) {
            if (!L.L.situ || !K::native_expert_wmma_supported(L.L)) { all_situ_fused = false; break; }
        }
        if (all_situ_fused) {
            const size_t escr_wmma = K::native_expert_wmma_fused_scratch_bytes(max_ent, hp.n_expert, hp.n_ff);
            const size_t escr_small = K::native_expert_scratch_bytes(std::min<int64_t>(max_ent, 16 * hp.n_expert_used), hp.n_ff);
            escr_bytes = a256(std::max(escr_wmma, escr_small));
        } else {
            escr_bytes = a256(K::native_expert_scratch_bytes(max_ent, hp.n_ff));
        }
        const size_t overlay_b_bytes = eparts_bytes + escr_bytes + sz_N + sz_N;
        const size_t logits_bytes = a256((size_t) verify_max_t * (size_t) hp.vocab_unpadded * sizeof(float));
        const size_t dspark_arena_min = (verify_max_t > 1) ? ((size_t) 28 << 20) : 0;

        layer_arena_bytes = std::max({overlay_a_bytes, overlay_b_bytes, overlay_a_ctrl_inffn, logits_bytes, dspark_arena_min});
        CK(cudaMalloc(&layer_arena, layer_arena_bytes));
        total_b += layer_arena_bytes;
        auto aptr = [&](size_t off) -> float* { return reinterpret_cast<float*>(layer_arena + off); };

        size_t o = 0;
        in_q = aptr(o); o += sz_N;
        in_k = aptr(o); o += sz_N;
        in_v = aptr(o); o += sz_N;
        K = aptr(o); o += sz_kv;
        V = aptr(o); o += sz_kv;
        R = aptr(o); o += sz_rel;
        K_conv = aptr(o); o += sz_kv;
        V_conv = aptr(o); o += sz_kv;
        dyn_k = aptr(o); o += sz_dyn;
        dyn_v = aptr(o); o += sz_dyn;
        pp_sp_buf = aptr(0);
        pp_sp_bytes = reg0_bytes;
        attn_split = aptr(0);
        wo = aptr(0);

        const size_t reg1_off = reg0_bytes;
        msg0 = aptr(reg1_off + 0 * sz_N);
        msg1 = aptr(reg1_off + 1 * sz_N);
        msg2 = aptr(reg1_off + 2 * sz_N);
        Q = aptr(reg1_off + 0 * sz_q);
        att_gate = aptr(reg1_off + 1 * sz_q);
        att = aptr(reg1_off + 2 * sz_q);

        const size_t reg2_off = reg1_off + reg1_bytes;
        o = reg2_off;
        ctrl2N = aptr(o); o += sz_2N;
        act_bf16_only = (!bf16_x_f16 && has_bf16_dense) ? reinterpret_cast<uint16_t*>(ctrl2N) : nullptr;
        ctrl_g = aptr(o); o += sz_N;
        ctrl_f = aptr(o); o += sz_N;
        ctrl64 = aptr(o); o += sz_c64;
        norm_cur = aptr(o); o += sz_N;
        ctrl_w16 = reinterpret_cast<uint16_t*>(layer_arena + o);
        rel_tab = aptr(reg2_off);

        eparts = aptr(0);
        escr = layer_arena + eparts_bytes;
        moe_raw = aptr(eparts_bytes + escr_bytes);
        in_ffn = aptr(layer_arena_bytes - sz_N);
        logits = aptr(0);
        return total_b;
    }

    ~Engine() {
        flush_ckpt_bg();
        if (core_graph) cudaGraphExecDestroy(core_graph);
        for (int t = 0; t < 9; ++t) {
            if (verify_core_graph[t]) cudaGraphExecDestroy(verify_core_graph[t]);
        }
        for (int t = 0; t < 9; ++t)
            for (int c = 0; c < 2; ++c)
                for (int s = 0; s < 2; ++s)
                    for (int a = 0; a < 2; ++a)
                        if (full_decode_graph[t][c][s][a]) cudaGraphExecDestroy(full_decode_graph[t][c][s][a]);
        for (auto g : layer_pre_graph) if (g) cudaGraphExecDestroy(g);
        for (int t = 0; t < 9; ++t) {
            for (auto g : verify_pre_graph[t]) if (g) cudaGraphExecDestroy(g);
        }
        if (h_hits_pin) cudaFreeHost(h_hits_pin);
        for (auto* p : kc) cudaFree(p);
        for (auto* p : vc) cudaFree(p);
        for (auto* p : kq) cudaFree(p);
        for (auto* p : vq) cudaFree(p);
        for (auto* p : ks) cudaFree(p);
        for (auto* p : vs) cudaFree(p);
        cudaFree(conv_hist_pool);
        cudaFree(swa_bak);
        cudaFree(d_conv_trace);
        for (int k = 0; k < 4; ++k) {
            if (h_tgt_feat_pin[k]) cudaFreeHost(h_tgt_feat_pin[k]);
            else cudaFree(d_tgt_feat[k]);
        }
        free_ephemeral_bufs();
        cudaFree(qgr_t1);
        cudaFree(xq); cudaFree(xq_k); cudaFree(xq_v);
        cudaFree(ids); cudaFree(grp_start); cudaFree(n_groups); cudaFree(ent_dst); cudaFree(ent_tok); cudaFree(grp_ptr);
        cudaFree(d_router_done);
        cudaFree(pos0_d); cudaFree(pick_d);
        cudaFree(argmax_val); cudaFree(argmax_idx);
        cudaFree(d_topk_hist); cudaFree(d_topk_thr); cudaFree(d_topk_cnt); cudaFree(d_topk_out_cnt);
        cudaFree(d_topk_cand_val); cudaFree(d_topk_cand_idx);
        cudaFree(d_topk_out_val); cudaFree(d_topk_out_idx);
        cudaFree(d_pen_toks); cudaFree(d_pen_cnts);
        cudaFree(d_miss); cudaFree(ecpu);
        cudaFreeHost(h_tok_buf);
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
        for (int b = 0; b < kMaxPpStages; ++b) {
            if (pp_stage[b]) cudaFree(pp_stage[b]);
            if (ev_pp_copy[b]) cudaEventDestroy(ev_pp_copy[b]);
            if (ev_pp_used[b]) cudaEventDestroy(ev_pp_used[b]);
        }
        for (int i = 0; i < kMaxRebalLayers; ++i) {
            if (ev_rebal_layer[i]) cudaEventDestroy(ev_rebal_layer[i]);
        }
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
        verify_max_t = std::max(1, max_verify_t);
        kv_q8 = use_kv_q8;
        const auto& hp = m->hp;
        const int64_t N = hp.n_embd;

        CK(cudaStreamCreate(&st));
        if (!gemm.init(st, (m->has_quant_dense || enable_gemm_quant) ? (size_t) (8 * 1024 * 1024) : 4096, err)) return false;

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
        swa_kv_cap = (tmax <= hp.n_swa) ? (int) hp.n_swa : (int) (hp.n_swa + tmax);
        swa_ptrs.n_swa_layers = 0;
        for (int il = 0; il < hp.n_layer; ++il) {
            const int cap = hp.is_swa(il) ? swa_kv_cap : full_kv_cap;
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
            if (hp.is_swa(il) && swa_ptrs.n_swa_layers < 48) {
                const int s = swa_ptrs.n_swa_layers++;
                swa_ptrs.kq[s] = kq[(size_t) il];
                swa_ptrs.vq[s] = vq[(size_t) il];
                swa_ptrs.ks[s] = ks[(size_t) il];
                swa_ptrs.vs[s] = vs[(size_t) il];
                swa_ptrs.kc[s] = kc[(size_t) il];
                swa_ptrs.vc[s] = vc[(size_t) il];
            }
            conv_hist[(size_t) il] = conv_hist_pool + (size_t) il * layer_conv_floats;
        }

        auto d_f32 = [](float** p, size_t n) { CK(cudaMalloc(p, n * sizeof(float))); };
        norm_blk_cnt = (hp.core_start == 6 && hp.core_end == 18 && hp.attn_res_block_size == 6 && hp.num_loops == 3) ? 5 : 9;
        d_f32(&qgr_t1, (size_t) (2 * hp.q_dim + hp.n_rel * hp.n_head + 2 * hp.kv_dim + 2 * hp.n_conv * hp.n_head_kv));
        attn_split_slots = verify_max_t * kAttnSplitMax;
        int cur_dev = 0, cc_maj = 0, cc_min = 0, max_smem_optin = 49152;
        cudaGetDevice(&cur_dev);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, cur_dev);
        cudaDeviceGetAttribute(&cc_maj, cudaDevAttrComputeCapabilityMajor, cur_dev);
        cudaDeviceGetAttribute(&cc_min, cudaDevAttrComputeCapabilityMinor, cur_dev);
        cudaDeviceGetAttribute(&max_smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, cur_dev);
        bf16_x_f16 = gemm.bf16_converts_x() || profile.force_bf16_x_f16;
        int64_t max_bf16_w = 0;
        {
            auto note_bf16 = [&](const DenseMat& W) {
                if (W.type == 30 && W.d) max_bf16_w = std::max(max_bf16_w, W.n_out * W.n_in);
            };
            for (const auto& L : m->layers) {
                note_bf16(L.attn_q); note_bf16(L.attn_k); note_bf16(L.attn_v);
                note_bf16(L.attn_gate); note_bf16(L.attn_rel); note_bf16(L.attn_sink_proj);
                note_bf16(L.attn_k_conv_proj); note_bf16(L.attn_v_conv_proj); note_bf16(L.attn_output);
                for (int mod = 0; mod < 4; ++mod) {
                    note_bf16(L.loop_down[mod]); note_bf16(L.loop_gain[mod]); note_bf16(L.loop_forget[mod]);
                }
            }
            if (max_bf16_w > 0) {
                has_bf16_dense = true;
                gemm.reserve_bf16_tc(max_bf16_w, bf16_x_f16 ? 0 : (int64_t) tmax * 2 * N);
            }
        }
        attn_split_t1_full = std::clamp((sm_count * profile.t1_full_split_mult) / std::max(1, (int) hp.n_head_kv), 40, kAttnSplitMax);
        attn_split_t1_swa = std::clamp((sm_count * profile.t1_swa_split_mult) / std::max(1, (int) hp.n_head_kv), 8, 32);
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
        cudaFuncSetAttribute((const void*) k_control_up_combine_pp_wmma, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        can_fa2_192 =
            try_set_smem((const void*) k_attn_q8_gqa3_pp_fa2_192<false, false, false>, 98304) &&
            try_set_smem((const void*) k_attn_q8_gqa3_pp_fa2_192<true, false, false>, 98304) &&
            try_set_smem((const void*) k_attn_q8_gqa3_pp_fa2_192<false, true, false>, 98304) &&
            try_set_smem((const void*) k_attn_q8_gqa3_pp_fa2_192<false, true, true>, 98304);
        fa2_192_swa_smem = (cc_maj == 7 && cc_min < 5) ? 73728 : 69632;
        can_fa2_192_swa_fast = ((cc_maj == 7 && cc_min < 5) || cc_maj >= 8) &&
            try_set_smem((const void*) k_attn_q8_gqa3_pp_fa2_192_swa_fast, fa2_192_swa_smem);
        fa2_256_bulk_smem = (cc_maj == 7 && cc_min < 5) ? 92160 : 87040;
        fa2_256_tail_smem = (cc_maj == 7 && cc_min < 5) ? 55296 : 52224;
        can_fa2_256 =
            try_set_smem((const void*) k_attn_q8_gqa3_pp_fa2_256_bulk, fa2_256_bulk_smem) &&
            try_set_smem((const void*) k_attn_q8_gqa3_pp_fa2_256_tail, fa2_256_tail_smem);
        try_set_smem((const void*) k_attn_q8_gqa3_pp_wmma_gqa3, 49152);
        try_set_smem((const void*) k_attn_q8_gqa3_pp_wmma_t32, 49152);
        try_set_smem((const void*) k_attn_q8_gqa3_pp_wmma, 29184);

        const int64_t max_ent = (int64_t) tmax * hp.n_expert_used;

        CK(cudaMallocHost(&h_tok_buf, (size_t) tmax * N * sizeof(float)));
        CK(cudaMallocHost(&h_is_img_pin, (size_t) tmax * sizeof(uint8_t)));
        CK(cudaMalloc(&d_is_img, (size_t) tmax * sizeof(uint8_t)));

        const int q8_kv_rows = std::min(tmax, 8);
        // xq is only ever used for T <= 8 (decode/verify) and DSpark block_n=8 (4*N).
        const size_t xq_blocks = std::max<size_t>((size_t) q8_kv_rows * ((2 * (size_t) N) / 32), (size_t) 16 * ((4 * (size_t) N) / 32));
        CK(cudaMalloc(&xq, xq_blocks * 36));
        CK(cudaMalloc(&xq_k, (size_t) q8_kv_rows * ((2 * N) / 32) * 36));
        CK(cudaMalloc(&xq_v, (size_t) q8_kv_rows * ((2 * N) / 32) * 36));

        prefill_ephemeral_bytes = alloc_ephemeral_bufs(tmax);
        in_decode_phase = false;

        CK(cudaMalloc(&ids, (size_t) max_ent * sizeof(int32_t)));
        CK(cudaMalloc(&grp_start, (size_t) (hp.n_expert + 1) * sizeof(int32_t)));
        CK(cudaMalloc(&n_groups, sizeof(int32_t)));
        CK(cudaMalloc(&ent_dst, (size_t) max_ent * sizeof(int32_t)));
        CK(cudaMalloc(&ent_tok, (size_t) max_ent * sizeof(int32_t)));
        CK(cudaMalloc(&grp_ptr, (size_t) hp.n_expert * sizeof(unsigned long long)));
        CK(cudaMalloc(&d_router_done, sizeof(uint32_t)));
        CK(cudaMemset(d_router_done, 0, sizeof(uint32_t)));
        CK(cudaMalloc(&pos0_d, sizeof(int32_t)));
        CK(cudaMalloc(&pick_d, (size_t) tmax * sizeof(int32_t)));
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
        CK(cudaMallocHost(&h_pos_pin, sizeof(int32_t)));

        CK(cudaMallocHost(&h_ids, (size_t) max_ent * sizeof(int32_t)));
        CK(cudaMallocHost(&h_ewts, (size_t) max_ent * sizeof(float)));
        CK(cudaMallocHost(&h_cpu_sum, (size_t) std::max(verify_max_t, 8) * (size_t) N * sizeof(float)));
        CK(cudaMallocHost(&h_hits_pin, (size_t) hp.n_phys * (size_t) hp.n_expert * sizeof(uint32_t)));
        for (int t = 0; t < 9; ++t) verify_pre_graph[t].assign((size_t) hp.n_layer, nullptr);
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
        if (const char* v = std::getenv("STRATA_PP_STREAM_T")) pp_stream_t = std::max(1, std::atoi(v));
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

    /// Exact GPU Top-K (K <= 256) + optional in-place penalties over `rows` rows of `logits` [rows, V].
    /// Returns true on success (writing sorted descending top-K vals/idxs to out_vals/out_idxs); false on bucket overflow.
    bool gpu_topk_rows(int rows, int K, const int32_t* pen_toks, const int32_t* pen_cnts, int n_pen,
                       float rep_pen, float freq_pen, float pres_pen, float* out_vals, int32_t* out_idxs) {
        const int V = (int) m->hp.vocab_unpadded;
        if (rows <= 0 || rows > verify_max_t || K <= 0 || K > 256 || K > V) return false;
        if (n_pen > 0 && n_pen <= 256 && (rep_pen != 1.0f || freq_pen != 0.0f || pres_pen != 0.0f)) {
            std::memcpy(h_pen_toks_pin, pen_toks, (size_t) n_pen * sizeof(int32_t));
            std::memcpy(h_pen_cnts_pin, pen_cnts, (size_t) n_pen * sizeof(int32_t));
            CK(cudaMemcpyAsync(d_pen_toks, h_pen_toks_pin, (size_t) n_pen * sizeof(int32_t), cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(d_pen_cnts, h_pen_cnts_pin, (size_t) n_pen * sizeof(int32_t), cudaMemcpyHostToDevice, st));
            k_apply_penalties<<<dim3((unsigned) ((n_pen + 63) / 64), (unsigned) rows), 64, 0, st>>>(
                logits, V, d_pen_toks, d_pen_cnts, n_pen, rep_pen, freq_pen, pres_pen);
        }
        constexpr int kHistBlks = 64;
        CK(cudaMemsetAsync(d_topk_hist, 0, (size_t) rows * 2048 * sizeof(uint32_t), st));
        k_topk_hist<<<dim3(kHistBlks, (unsigned) rows), 256, 0, st>>>(logits, d_topk_hist, V);
        k_topk_select_thresh<<<(unsigned) rows, 256, 0, st>>>(d_topk_hist, d_topk_thr, d_topk_cnt, K);
        k_topk_collect<<<dim3(kHistBlks, (unsigned) rows), 256, 0, st>>>(
            logits, d_topk_thr, d_topk_cand_val, d_topk_cand_idx, d_topk_cnt, V);
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
        if (h_h != nullptr) return;
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
    }

    void reset_state() {
        pos = 0;
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
        const size_t total_conv = (size_t) hp.n_layer * layer_conv_floats;
        if (!d_conv_trace) CK(cudaMalloc(&d_conv_trace, (size_t) verify_max_t * total_conv * sizeof(float)));
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
                    uint16_t* dst_bf16 = (act_bf16_only && X != ctrl2N) ? act_bf16_only : act_bf16;
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

    void run_mix_normed(const std::vector<float*>& n_blocks, const float* c, bool c_already_normed, const float* query, float* const* outs, int n_q, int T, int swa_bcast) {
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
            if (N == 2304) {
                if (fuse_last_norm) {
                    if (n_q == 1) {
                        k_mix_normed_fused_1pass<1, true><<<(unsigned) T, 576, 0, st>>>(np, D, query, nullptr, mo4, N, inv_sqrt_n, swa_bcast, hp.rms_eps);
                    } else {
                        k_mix_normed_fused_1pass<3, true><<<(unsigned) T, 576, 0, st>>>(np, D, query, nullptr, mo4, N, inv_sqrt_n, 0, hp.rms_eps);
                    }
                } else {
                    if (n_q == 1) {
                        k_mix_normed_fused_1pass<1, false><<<(unsigned) T, 576, 0, st>>>(np, D, query, nullptr, mo4, N, inv_sqrt_n, swa_bcast, 0.0f);
                    } else {
                        k_mix_normed_fused_1pass<3, false><<<(unsigned) T, 576, 0, st>>>(np, D, query, nullptr, mo4, N, inv_sqrt_n, 0, 0.0f);
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

    void run_control(const float* c, const float* msg, const PhysLayer& L, int mod, float* out, int T) {
        const auto& hp = m->hp;
        const int N = (int) hp.n_embd;
        if (T >= 32 && (N & 3) == 0 && ctrl_w16 && L.loop_down[mod].type == 0 && L.loop_gain[mod].type == 0 && L.loop_forget[mod].type == 0) {
            // FP16 Tensor Core path for prefill loop controllers (zero extra VRAM: uses ctrl_w16 slice in layer_arena + act_bf16).
            const int64_t R = hp.loop_gate_rank;
            const int64_t n_wd = R * 2 * N;
            const int64_t n_wu = (int64_t) N * R;
            const int64_t n_x64 = (int64_t) T * R;
            if ((size_t) (n_wd + 2 * n_wu) * sizeof(uint16_t) <= ctrl_w16_bytes) {
                uint16_t* w_down16 = ctrl_w16;
                k_control_prep_f16<<<dim3(2, (unsigned) T), kThreads, 0, st>>>(c, msg, act_bf16, N, hp.rms_eps);
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

    void sync_d_hits(bool accumulate_hy = false) {
        const auto& hp = m->hp;
        const int ne = (int) hp.n_expert;
        const int n_phys = (int) m->layers.size();
        bool any = false;
        for (int ip = 0; ip < n_phys; ++ip) {
            PhysLayer& L = m->layers[(size_t) ip];
            if (!L.d_hits) continue;
            CK(cudaMemcpyAsync(h_hits_pin + (size_t) ip * ne, L.d_hits, (size_t) ne * sizeof(uint32_t), cudaMemcpyDeviceToHost, st));
            CK(cudaMemsetAsync(L.d_hits, 0, (size_t) ne * sizeof(uint32_t), st));
            any = true;
        }
        if (!any) return;
        CK(cudaStreamSynchronize(st));
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

    void pp_kick_next() {
        if (pp_kick_idx >= pp_tasks.size() || pp_stage_cnt <= 0) return;
        const PpTask& tk = pp_tasks[pp_kick_idx];
        const int b = (int) (pp_kick_idx % (size_t) pp_stage_cnt);
        const PhysLayer& L = m->layers[(size_t) tk.ip];
        CK(cudaStreamWaitEvent(cst, ev_pp_used[b], 0));
        for (const PpRun& r : tk.runs) {
            const size_t nbytes = (size_t) (r.cnt - 1) * L.stride + L.L.bytes;
            CK(cudaMemcpyAsync(pp_stage[b] + (size_t) r.dst_off * L.stride,
                               L.hblobs + (size_t) r.e0 * L.stride,
                               nbytes, cudaMemcpyHostToDevice, cst));
        }
        CK(cudaEventRecord(ev_pp_copy[b], cst));
        ++pp_kick_idx;
    }

    void pp_begin(int core_end_virt, int core_all_gpu_flag) {
        const auto& hp = m->hp;
        const int ne = (int) hp.n_expert;
        pp_tasks.clear();
        std::fill(pp_layer_cnt.begin(), pp_layer_cnt.end(), 0);
        pp_take_idx = 0;
        pp_kick_idx = 0;
        if (pp_stage_cnt <= 0 || !pp_stage[0] || pp_stage_bytes == 0) return;
        (void) core_end_virt;
        (void) core_all_gpu_flag;
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
                tk.n_staged = sz;
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
        for (int b = 0; b < pp_stage_cnt; ++b) CK(cudaEventRecord(ev_pp_used[b], st));
        for (int k = 0; k < pp_stage_cnt && pp_kick_idx < pp_tasks.size(); ++k) pp_kick_next();
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

    bool run_moe(PhysLayer& L, int il, const float* x_norm, int T, bool pre_routed_t1 = false,
                 bool xq_ready_in = false, bool skip_combine = false, bool f16_ready_in = false,
                 const PostConvSpec* post_conv = nullptr) {
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
            if (need_cpu && T >= 1 && T <= 8 && !pp_streaming) {
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
                project_f32(x_norm, L.ffn_gate_inp, router_logits, T, hp.n_expert, N);
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
        const bool try_wmma = (T >= 16);
        auto run_grouped = [&](const unsigned long long* gptr, int64_t cap_g, int64_t est_ent = 0) {
            if (try_wmma) {
                ensure_f16();
                if (K::native_expert_grouped_wmma(L.L, gptr, grp_start, n_groups, ent_dst, ent_tok,
                                                  cap_g, n_ent, act_bf16, escr, eparts, st, est_ent)) {
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

        if (pp_streaming && need_cpu && L.hblobs && pp_stage_bytes >= L.stride && pp_layer_cnt[(size_t) il] > 0) {
            // Zero host-GPU stalls across all 48 layers: k_route_inkling2 tracks exact routing counts in L.d_hits on GPU!
            const int ne = (int) hp.n_expert;
            const int n_chunks = pp_layer_cnt[(size_t) il];
            auto run_gpu_res = [&]() {
                if (!has_gpu) return;
                k_group_experts<<<1, 512, 0, st>>>(ids, L.d_slot_of, L.d_blobs, L.stride, n_ent, ne, L.n_slots,
                                                   grp_ptr, grp_start, n_groups, ent_dst, ent_tok, Kp);
                const int64_t est_gpu = ((int64_t) n_ent * L.n_slots + ne - 1) / ne;
                run_grouped(grp_ptr, std::min<int64_t>(n_ent, L.n_slots), est_gpu);
            };
            auto run_staged = [&]() {
                for (int ci = 0; ci < n_chunks; ++ci) {
                    if (pp_take_idx >= pp_tasks.size()) {
                        std::fprintf(stderr, "strata-inkling2: pp stream underflow\n");
                        std::exit(1);
                    }
                    const PpTask& tk = pp_tasks[pp_take_idx];
                    const int b = (int) (pp_take_idx % (size_t) pp_stage_cnt);
                    // k_group_experts_miss_map computes pointer arithmetic only (never dereferences pp_stage[b]);
                    // running it before waiting on ev_pp_copy[b] hides grouping latency behind PCIe DMA!
                    k_group_experts_miss_map<<<1, 512, 0, st>>>(ids, pp_stage[b], L.stride, n_ent, ne, tk.map,
                                                                grp_ptr, grp_start, n_groups, ent_dst, ent_tok, Kp);
                    CK(cudaStreamWaitEvent(st, ev_pp_copy[b], 0));
                    const int64_t est_st = ((int64_t) n_ent * tk.n_staged + ne - 1) / ne;
                    run_grouped(grp_ptr, std::min<int64_t>(n_ent, (int64_t) tk.n_staged), est_st);
                    CK(cudaEventRecord(ev_pp_used[b], st));
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
            if (post_conv && post_conv->w_conv && T >= 32 && (N & 3) == 0 && !post_conv->hist_trace) {
                const int n4 = N >> 2;
                k_combine_experts_shortconv_pp<<<dim3((unsigned) ((n4 + 63) / 64), (unsigned) ((T + 63) / 64)), 64, 0, st>>>(
                    eparts, ewts, post_conv->w_conv, post_conv->hist, post_conv->res_add, post_conv->out,
                    N, Kp, post_conv->hist_offset, T);
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
            if (!pre_routed_t1) {
                CK(cudaMemcpyAsync(h_ids, ids, (size_t) n_ent * sizeof(int32_t), cudaMemcpyDeviceToHost, st));
                if (T >= 1 && T <= 8) {
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
                k_group_experts<<<1, 512, 0, st>>>(ids, L.d_slot_of, L.d_blobs, L.stride, n_ent, (int) hp.n_expert, L.n_slots,
                                                   grp_ptr, grp_start, n_groups, ent_dst, ent_tok, Kp);
            }
            run_grouped(grp_ptr, std::min<int64_t>(n_ent, hp.n_expert));
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
                    j.blob = L.hblobs + (size_t) x * L.stride;
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
                        j.blob = L.hblobs + (size_t) x * L.stride;
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
                if (l2_pf_min > 0 && n_miss >= l2_pf_min && hybrid && T <= 8 && il + 1 < hp.n_layer) {
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
        if (post_conv && post_conv->w_conv && T <= 8) {
            k_combine_experts_shortconv<<<(unsigned) ((N + 31) / 32), 32, 0, st>>>(
                eparts, ewts, post_conv->w_conv, post_conv->hist, post_conv->res_add, post_conv->out,
                N, Kp, post_conv->hist_offset, T, post_conv->hist_trace, post_conv->hist_trace_stride);
            if (prof && need_cpu) { CK(cudaStreamSynchronize(st)); p_moe_post_ms += ms_since(tp_p0); }
            return true;
        }
        if (post_conv && post_conv->w_conv && T >= 32 && (N & 3) == 0 && !post_conv->hist_trace) {
            const int n4 = N >> 2;
            k_combine_experts_shortconv_pp<<<dim3((unsigned) ((n4 + 63) / 64), (unsigned) ((T + 63) / 64)), 64, 0, st>>>(
                eparts, ewts, post_conv->w_conv, post_conv->hist, post_conv->res_add, post_conv->out,
                N, Kp, post_conv->hist_offset, T);
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
                      [&](int a, int b) { return L.score[(size_t) a] > L.score[(size_t) b]; });
            victims.resize((size_t) S);
            for (int s = 0; s < S; ++s) victims[(size_t) s] = s;
            const int k_lim = std::min(S, std::max(1, max_per_layer));
            std::partial_sort(victims.begin(), victims.begin() + k_lim, victims.end(),
                              [&](int a, int b) {
                                  return L.score[(size_t) L.exp_in_slot[(size_t) a]] < L.score[(size_t) L.exp_in_slot[(size_t) b]];
                              });
            int v_idx = 0;
            for (int i = 0; i < S && v_idx < k_lim; ++i) {
                const int hot_cand = order[(size_t) i];
                if (L.slot_of[(size_t) hot_cand] >= 0) continue;
                const int s = victims[(size_t) v_idx];
                const int y = L.exp_in_slot[(size_t) s];
                const double diff = L.score[(size_t) hot_cand] - L.score[(size_t) y];
                // Hysteresis avoids thrashing 4.5 MiB H2D swaps on near-tied scores during decode.
                const double min_diff = (max_swaps >= 48) ? 0.0 : std::max(0.25, 0.15 * L.score[(size_t) y]);
                if (diff <= min_diff) break;
                cands.push_back(Cand{diff * weight, li, s, hot_cand, y});
                ++v_idx;
            }
            for (int x = 0; x < ne; ++x) L.score[(size_t) x] *= 0.8;
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
                CK(cudaMemcpyAsync(L.d_blobs + (size_t) c.slot * L.stride, L.hblobs + (size_t) c.hot_x * L.stride,
                                   L.L.bytes, cudaMemcpyHostToDevice, cst));
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

    void destroy_graphs() {
        if (core_graph) { cudaGraphExecDestroy(core_graph); core_graph = nullptr; }
        for (int t = 0; t < 9; ++t) {
            if (verify_core_graph[t]) { cudaGraphExecDestroy(verify_core_graph[t]); verify_core_graph[t] = nullptr; }
        }
        for (int t = 0; t < 9; ++t)
            for (int c = 0; c < 2; ++c)
                for (int s = 0; s < 2; ++s)
                    for (int a = 0; a < 2; ++a)
                        if (full_decode_graph[t][c][s][a]) {
                            cudaGraphExecDestroy(full_decode_graph[t][c][s][a]);
                            full_decode_graph[t][c][s][a] = nullptr;
                        }
        for (auto& g : layer_pre_graph) {
            if (g) { cudaGraphExecDestroy(g); g = nullptr; }
        }
        for (int t = 0; t < 9; ++t) {
            for (auto& g : verify_pre_graph[t]) {
                if (g) { cudaGraphExecDestroy(g); g = nullptr; }
            }
        }
        graph_oom = false;
    }

    // Dynamic VRAM elasticity (--vram-elastic & Prefill<->Decode phase split):
    // shrink expert cache when free < target_free_bytes, or grow toward full residency when free > guard.
    void resize_experts_vram(size_t target_free_bytes, size_t guard_override = 0) {
        const auto t0 = Clock::now();
        const auto& hp = m->hp;
        const int ne = (int) hp.n_expert;

        CK(cudaStreamSynchronize(st));
        if (cst) CK(cudaStreamSynchronize(cst));
        for (int i = 0; i < kMaxRebalLayers; ++i) layer_pending_rebal[i] = false;
        destroy_graphs();
        sync_d_hits(false);

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
            const bool want_pin = (std::getenv("STRATA_NO_PIN_EXPERTS") == nullptr);
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
            }
        };

        auto set_layer_slots = [&](int ip, int target_slots) -> bool {
            PhysLayer& L = m->layers[(size_t) ip];
            target_slots = std::clamp(target_slots, 0, ne);
            if (target_slots == L.n_slots) return false;
            const bool is_tier1 = (ip >= hp.core_start && ip < hp.core_end);
            ensure_hblobs(L, is_tier1);

            std::vector<int> keep;
            if (target_slots > 0) {
                std::vector<int> order((size_t) ne);
                for (int e = 0; e < ne; ++e) order[(size_t) e] = e;
                std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
                    if (L.score[(size_t) a] != L.score[(size_t) b]) return L.score[(size_t) a] > L.score[(size_t) b];
                    return (L.slot_of[(size_t) a] >= 0) > (L.slot_of[(size_t) b] >= 0);
                });
                keep.assign(order.begin(), order.begin() + target_slots);
            }

            if (L.d_blobs) {
                auto it = std::find(m->gpu_allocs.begin(), m->gpu_allocs.end(), (void*) L.d_blobs);
                if (it != m->gpu_allocs.end()) m->gpu_allocs.erase(it);
                CK(cudaFree(L.d_blobs));
                L.d_blobs = nullptr;
            }
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
                    if (!L.d_slot_of) {
                        CK(cudaMalloc(&L.d_slot_of, (size_t) ne * sizeof(int32_t)));
                        m->gpu_allocs.push_back(L.d_slot_of);
                    }
                    L.n_slots = s;
                    L.exp_in_slot.assign((size_t) s, -1);
                    for (int k = 0; k < s; ++k) {
                        const int e = keep[(size_t) k];
                        CK(cudaMemcpyAsync(L.d_blobs + (size_t) k * L.stride, L.hblobs + (size_t) e * L.stride,
                                           L.L.bytes, cudaMemcpyHostToDevice, st));
                        L.slot_of[(size_t) e] = k;
                        L.exp_in_slot[(size_t) k] = e;
                    }
                    CK(cudaStreamSynchronize(st));
                }
            }
            if (L.d_slot_of) {
                CK(cudaMemcpy(L.d_slot_of, L.slot_of.data(), (size_t) ne * sizeof(int32_t), cudaMemcpyHostToDevice));
            }
            rebal_burst = 2;
            return true;
        };

        std::vector<int> tier1, tier2, tier3;
        for (int ip = 0; ip < hp.n_phys; ++ip) {
            if (m->layers[(size_t) ip].gu_type == 30) tier3.push_back(ip);
            else if (ip >= hp.core_start && ip < hp.core_end) tier1.push_back(ip);
            else tier2.push_back(ip);
        }

        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));

        if (fr < target_free_bytes) {
            // SHRINK: evict Tier 3 -> Tier 2 -> Tier 1 (fast: full-layer cudaFree first, minimal re-upload)
            auto shrink_tier = [&](const std::vector<int>& tier) {
                for (int idx = (int) tier.size() - 1; idx >= 0 && fr < target_free_bytes; --idx) {
                    const int ip = tier[(size_t) idx];
                    const PhysLayer& L = m->layers[(size_t) ip];
                    if (L.n_slots <= 0) continue;
                    const size_t deficit = target_free_bytes - fr;
                    const int drop = (L.stride > 0) ? (int) ((deficit + L.stride - 1) / L.stride) : L.n_slots;
                    const int tgt = (drop >= L.n_slots) ? 0 : (L.n_slots - drop);
                    set_layer_slots(ip, tgt);
                    CK(cudaMemGetInfo(&fr, &tot));
                }
            };
            shrink_tier(tier3);
            shrink_tier(tier2);
            shrink_tier(tier1);
        } else {
            // GROW: restore Tier 1 -> Tier 2 -> Tier 3 back toward full residency while keeping (target_free + base_guard) free
            const size_t keep_free = std::max(target_free_bytes, base_guard);
            auto grow_tier = [&](const std::vector<int>& tier) {
                if (tier.empty()) return;
                CK(cudaMemGetInfo(&fr, &tot));
                if (fr <= keep_free) return;
                size_t avail = fr - keep_free;
                size_t need_full = 0;
                for (int ip : tier) {
                    const PhysLayer& L = m->layers[(size_t) ip];
                    if (L.n_slots < ne) need_full += (size_t) (ne - L.n_slots) * L.stride;
                }
                if (need_full == 0) return;
                if (avail >= need_full) {
                    for (int ip : tier) {
                        if (m->layers[(size_t) ip].n_slots < ne) set_layer_slots(ip, ne);
                    }
                    return;
                }
                // Distribute available VRAM evenly across layers in this tier that are below full
                std::vector<int> targets(tier.size());
                size_t sum_cur_bytes = 0, sum_stride = 0;
                for (size_t i = 0; i < tier.size(); ++i) {
                    const PhysLayer& L = m->layers[(size_t) tier[i]];
                    targets[i] = L.n_slots;
                    sum_cur_bytes += (size_t) L.n_slots * L.stride;
                    sum_stride += L.stride;
                }
                const size_t total_b = sum_cur_bytes + avail;
                const int uniform_s = sum_stride > 0 ? (int) std::min<size_t>((size_t) ne, total_b / sum_stride) : 0;
                size_t used_b = 0;
                for (size_t i = 0; i < tier.size(); ++i) {
                    const int cur_s = m->layers[(size_t) tier[i]].n_slots;
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
                for (size_t i = 0; i < tier.size(); ++i) {
                    const int ip = tier[i];
                    if (targets[i] > m->layers[(size_t) ip].n_slots) {
                        CK(cudaMemGetInfo(&fr, &tot));
                        const size_t cur_b = (size_t) m->layers[(size_t) ip].n_slots * m->layers[(size_t) ip].stride;
                        const size_t max_b = (fr + cur_b > keep_free) ? (fr + cur_b - keep_free) : 0;
                        const int max_s = m->layers[(size_t) ip].stride > 0
                            ? (int) std::min<size_t>((size_t) ne, max_b / m->layers[(size_t) ip].stride) : 0;
                        const int ns = std::min(targets[i], max_s);
                        if (ns > m->layers[(size_t) ip].n_slots) set_layer_slots(ip, ns);
                    }
                }
            };
            grow_tier(tier1);
            grow_tier(tier2);
            grow_tier(tier3);
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
            const int cpu_t = (pp_stage[0] && pp_stream_t > 1) ? std::min(tmax, pp_stream_t - 1) : tmax;
            init_cpu_bufs(cpu_t);
        }
        placed_slots = tot_slots;
        placed_bytes = exp_bytes;
        if (exp_bytes > full_cache_bytes) full_cache_bytes = exp_bytes;
        CK(cudaMemGetInfo(&fr, &tot));
        vram_free_after_place = fr;
        std::fprintf(stderr, "strata-inkling2: VRAM resize(target=%.0f MiB) -> slots=%d/%lld (%.0f/%.0f MiB, free=%.0f MiB, hybrid=%d) in %.1f ms\n",
                     (double) target_free_bytes / (1 << 20), tot_slots, (long long) hp.n_phys * ne,
                     (double) exp_bytes / (1 << 20), (double) full_cache_bytes / (1 << 20),
                     (double) fr / (1 << 20), hybrid ? 1 : 0, ms_since(t0));
    }

    /// Phase-split Decode transition: free Prefill-only scratch (chunk=2048 arena, norm_blk_buf, pp_stage -> ~650 MiB)
    /// and immediately grow hot tier1/tier2 experts from pinned RAM before decoding.
    void set_phase_decode() {
        static const bool disable_ps = (std::getenv("STRATA_NO_PHASE_SPLIT") != nullptr);
        if (disable_ps || !hybrid || in_decode_phase || tmax <= kDecodePhaseCap || tmax < 1024) return;
        CK(cudaStreamSynchronize(st));
        if (cst) CK(cudaStreamSynchronize(cst));
        for (int i = 0; i < kMaxRebalLayers; ++i) layer_pending_rebal[i] = false;
        destroy_graphs();

        // Free pp_stage ring buffers (unused during decode)
        saved_pp_stage_cnt = pp_stage_cnt;
        saved_pp_stage_bytes = pp_stage_bytes;
        for (int b = 0; b < pp_stage_cnt; ++b) {
            if (pp_stage[b]) { cudaFree(pp_stage[b]); pp_stage[b] = nullptr; }
        }
        pp_stage_cnt = 0;
        pp_stage_bytes = 0;

        free_ephemeral_bufs();
        decode_ephemeral_bytes = alloc_ephemeral_bufs(kDecodePhaseCap);
        if (dspark) dspark->rebind_arena(layer_arena, layer_arena_bytes, kDecodePhaseCap);
        in_decode_phase = true;

        const size_t dec_guard = (verify_max_t > 1) ? ((size_t) 176 << 20) : ((size_t) 144 << 20);
        resize_experts_vram(vram_reserve_bytes, dec_guard);
    }

    /// Phase-split Prefill transition: shrink temporary decode-bonus experts and restore full tmax prefill scratch + pp_stage.
    void set_phase_prefill() {
        if (!in_decode_phase) return;
        CK(cudaStreamSynchronize(st));
        if (cst) CK(cudaStreamSynchronize(cst));
        for (int i = 0; i < kMaxRebalLayers; ++i) layer_pending_rebal[i] = false;
        destroy_graphs();

        const size_t extra_eph = (prefill_ephemeral_bytes > decode_ephemeral_bytes)
            ? (prefill_ephemeral_bytes - decode_ephemeral_bytes) : 0;
        const size_t stage_tot = (size_t) saved_pp_stage_cnt * saved_pp_stage_bytes;
        const size_t need_free = vram_reserve_bytes + extra_eph + stage_tot + ((size_t) 128 << 20);
        resize_experts_vram(need_free);

        free_ephemeral_bufs();
        prefill_ephemeral_bytes = alloc_ephemeral_bufs(tmax);
        if (dspark) dspark->rebind_arena(layer_arena, layer_arena_bytes, tmax);

        if (saved_pp_stage_cnt > 0 && saved_pp_stage_bytes > 0) {
            int ok = 0;
            for (int b = 0; b < saved_pp_stage_cnt; ++b) {
                if (cudaMalloc(&pp_stage[b], saved_pp_stage_bytes) == cudaSuccess) {
                    ++ok;
                } else {
                    (void) cudaGetLastError();
                    break;
                }
            }
            if (ok >= 2) {
                pp_stage_cnt = ok;
                pp_stage_bytes = saved_pp_stage_bytes;
            } else {
                for (int b = 0; b < ok; ++b) { cudaFree(pp_stage[b]); pp_stage[b] = nullptr; }
                pp_stage_cnt = 0;
                pp_stage_bytes = 0;
            }
        }
        in_decode_phase = false;
    }

    void ensure_kv_cap(int needed_pos) {
        if (needed_pos <= full_kv_cap || full_kv_cap >= ctx) return;
        const auto& hp = m->hp;
        const int step_cap = (profile.regime == strata::inkling2::VramRegime::TightHybrid && full_kv_cap < 65536)
                           ? (full_kv_cap + 16384)
                           : (full_kv_cap * 2);
        const int new_cap = std::min(ctx, std::max(needed_pos, step_cap));
        if (new_cap <= full_kv_cap) return;

        CK(cudaStreamSynchronize(st));
        if (cst) CK(cudaStreamSynchronize(cst));
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
            bool want_pin = (std::getenv("STRATA_NO_PIN_EXPERTS") == nullptr);
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
                    if (L.score[(size_t) a] != L.score[(size_t) b]) return L.score[(size_t) a] > L.score[(size_t) b];
                    return (L.slot_of[(size_t) a] >= 0) > (L.slot_of[(size_t) b] >= 0);
                });
                keep.assign(order.begin(), order.begin() + target_slots);
            }

            auto it = std::find(m->gpu_allocs.begin(), m->gpu_allocs.end(), (void*) L.d_blobs);
            if (it != m->gpu_allocs.end()) m->gpu_allocs.erase(it);
            CK(cudaFree(L.d_blobs));
            L.d_blobs = nullptr;
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
                        CK(cudaMemcpyAsync(L.d_blobs + (size_t) k * L.stride, L.hblobs + (size_t) e * L.stride,
                                           L.L.bytes, cudaMemcpyHostToDevice, st));
                        L.slot_of[(size_t) e] = k;
                        L.exp_in_slot[(size_t) k] = e;
                    }
                    CK(cudaStreamSynchronize(st));
                }
            }
            if (L.d_slot_of) {
                CK(cudaMemcpy(L.d_slot_of, L.slot_of.data(), (size_t) ne * sizeof(int32_t), cudaMemcpyHostToDevice));
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
            const int cpu_t = (!has_bf16_experts && pp_stage[0] && pp_stream_t > 1) ? std::min(tmax, pp_stream_t - 1) : tmax;
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
                    size_t* img_cursor = nullptr) {
        if (in_decode_phase && T > kDecodePhaseCap) set_phase_prefill();
        ensure_kv_cap((int) (pos + T));
        const auto& hp = m->hp;
        const int N = (int) hp.n_embd;
        const int64_t kv_dim = hp.kv_dim;
        const int64_t q_dim = hp.q_dim;
        const int64_t n_core = hp.num_loops > 1 ? (hp.core_end - hp.core_start) : 0;
        const bool compute_logits = need_logits || copy_logits || all_row_logits || out_picks != nullptr;
        const int rows = (all_row_logits || out_picks != nullptr) ? T : 1;
        const bool want_softcap = (copy_logits && hp.softcap > 0.0f);
        const bool want_argmax = !copy_logits || (out_picks != nullptr);
        const bool can_full_graph = (!hybrid && core_all_gpu && full_kv_cap == ctx && !prof && !prof_detail && !graph_oom &&
                                     img_embs == nullptr &&
                                     T >= 1 && T <= 8 && T <= verify_max_t && compute_logits && rows == T &&
                                     h_tok_buf != nullptr && h_pos_pin != nullptr && h_pick_pin != nullptr &&
                                     (!copy_logits || defer_logits_d2h) &&
                                     ((T == 1 && verify_k_save == 0) || (T >= 2 && verify_k_save == T - 1)));
        bool capturing_full_graph = false;
        cudaGraphExec_t* fg_slot = nullptr;
        if (can_full_graph) {
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
        bool chunk_has_img = false;
        std::vector<int64_t> img_slot;
        if (!can_full_graph) {
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
        const int32_t pos32 = (int32_t) pos;
        if (h_pos_pin) {
            if (!can_full_graph) *h_pos_pin = pos32;
            CK(cudaMemcpyAsync(pos0_d, h_pos_pin, sizeof(int32_t), cudaMemcpyHostToDevice, st));
        } else {
            CK(cudaMemcpyAsync(pos0_d, &pos32, sizeof(int32_t), cudaMemcpyHostToDevice, st));
        }
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
        if (chunk_has_img) {
            k_rms_norm_masked<<<(unsigned) T, kThreads, 0, st>>>(ffn_inp, m->tok_embd_norm, d_is_img, cur, N, hp.rms_eps);
        } else {
            k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(ffn_inp, m->tok_embd_norm, cur, N, hp.rms_eps);
        }
        if (prof) { CK(cudaStreamSynchronize(st)); p_embd_ms += ms_since(tp_e0); }

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

        auto run_layer_pre = [&](int il, bool include_route_t1 = false) {
            const int ip = hp.loop_layer(il);
            PhysLayer& L = m->layers[(size_t) ip];
            const bool is_swa = hp.is_swa(il);

            const auto tp_d0 = Clock::now();
            bool has_msg = false;
            if (n_core > 0 && ip >= hp.core_start && ip < hp.core_end) {
                const int pass = (il - (int) hp.core_start) / (int) n_core;
                if (ip == hp.core_start) {
                    if (pass == 0) {
                        prelude_norm_blocks = norm_blocks;
                        CK(cudaMemcpyAsync(prelude_cur, cur, (size_t) T * N * sizeof(float), cudaMemcpyDeviceToDevice, st));
                    } else {
                        norm_bank.clear();
                        for (size_t bi = prelude_norm_blocks.size() + 1; bi < norm_blocks.size(); ++bi) norm_bank.push_back(norm_blocks[bi]);
                        k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(cur, nullptr, bank_last_norm, N, hp.rms_eps);
                        norm_blocks = prelude_norm_blocks;
                        CK(cudaMemcpyAsync(cur, prelude_cur, (size_t) T * N * sizeof(float), cudaMemcpyDeviceToDevice, st));
                    }
                }
                if (pass > 0) {
                    has_msg = true;
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
                        k_mix_normed_fused_1pass<4><<<(unsigned) T, 576, 0, st>>>(
                            np, B + 1, q_attn, q_kv, mo4, N, 1.0f / std::sqrt((float) N), 0);
                    } else if (T >= 1 && (N & 3) == 0 && B + 1 <= 8) {
                        k_mix_normed_fused<4><<<(unsigned) T, kThreads, 0, st>>>(
                            np, B + 1, q_attn, q_kv, mo4, N, 1.0f / std::sqrt((float) N), 0);
                    } else {
                        k_mix_normed_4<<<dim3(4, (unsigned) T), kThreads, 0, st>>>(
                            np, B + 1, q_attn, q_kv, mo4, N, 1.0f / std::sqrt((float) N));
                    }
                }
            }

            const float* ink_src = in_k;
            const float* inv_src = in_v;
            if (capture_dspark && dspark) {
                const auto& tlayers = dspark->dm->hp.target_layers;
                for (int ti = 0; ti < 3; ++ti) {
                    if (il == tlayers[(size_t) ti] && d_tgt_feat[ti]) {
                        CK(cudaMemcpyAsync(d_tgt_feat[ti], cur, (size_t) T * N * sizeof(float), cudaMemcpyDeviceToDevice, st));
                    }
                }
            }
            const bool fuse_q8_norm = (T <= 8 && (N & 31) == 0);
            const int rms_q8_blk = (N == 2304) ? 576 : kThreads;
            const int rms_f16_blk = (N == 2304) ? 576 : kThreads;
            bool q8_pre_done = false;
            if (L.attn_res_query) {
                float* outs[3] = {in_q, in_k, in_v};
                bool fused_mix_norm = false;
                if (!has_msg && fuse_q8_norm) {
                    if (is_swa) {
                        NormQ8Out3 nq8{L.attn_norm, nullptr, nullptr, xq, nullptr, nullptr};
                        if (try_run_mix_normed_q8(norm_blocks, cur, L.attn_res_query, outs, nq8, 1, T)) {
                            fused_mix_norm = true;
                            q8_pre_done = true;
                            ink_src = in_q;
                            inv_src = in_q;
                        }
                    } else {
                        NormQ8Out3 nq8{L.attn_q_inp_norm, L.attn_k_inp_norm, L.attn_v_inp_norm, xq, xq_k, xq_v};
                        if (try_run_mix_normed_q8(norm_blocks, cur, L.attn_res_query, outs, nq8, 3, T)) {
                            fused_mix_norm = true;
                            q8_pre_done = true;
                        }
                    }
                } else if (!has_msg && T > 8 && N == 2304) {
                    if (is_swa) {
                        NormF16Out3 nf16{L.attn_norm, nullptr, nullptr, act_bf16, nullptr, nullptr};
                        if (try_run_mix_normed_f16(norm_blocks, cur, L.attn_res_query, outs, nf16, 1, T)) {
                            fused_mix_norm = true;
                            ink_src = in_q;
                            inv_src = in_q;
                        }
                    } else {
                        NormF16Out3 nf16{L.attn_q_inp_norm, L.attn_k_inp_norm, L.attn_v_inp_norm,
                                         act_bf16, act_bf16 + (size_t) T * N, nullptr};
                        if (try_run_mix_normed_f16(norm_blocks, cur, L.attn_res_query, outs, nf16, 3, T)) {
                            fused_mix_norm = true;
                        }
                    }
                }
                if (!fused_mix_norm) {
                run_mix_normed(norm_blocks, cur, false, L.attn_res_query, outs, is_swa ? 1 : 3, T, is_swa ? 1 : 0);
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
                if (is_swa && has_msg) {
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
                            in_q, in_k, in_v, in_q, in_k, in_v, L.attn_q_inp_norm, L.attn_k_inp_norm, L.attn_v_inp_norm,
                            act_bf16, act_bf16 + (size_t) T * N, nullptr, N, hp.rms_eps);
                    } else {
                        k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(in_q, L.attn_q_inp_norm, in_q, N, hp.rms_eps);
                        k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(in_k, L.attn_k_inp_norm, in_k, N, hp.rms_eps);
                        k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(in_v, L.attn_v_inp_norm, in_v, N, hp.rms_eps);
                    }
                }
                }
            } else {
                if (has_msg) {
                    run_control(cur, msg0, L, 0, in_q, T);
                    run_control(cur, msg1, L, 1, in_k, T);
                    run_control(cur, msg2, L, 2, in_v, T);
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
                        k_rms_norm_q8_1<<<(unsigned) T, rms_q8_blk, 0, st>>>(cur, L.attn_norm, in_q, xq, N, hp.rms_eps);
                        q8_pre_done = true;
                    } else if (T > 8) {
                        k_rms_norm_f16<<<(unsigned) T, rms_f16_blk, 0, st>>>(cur, L.attn_norm, in_q, act_bf16, N, hp.rms_eps);
                    } else {
                        k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(cur, L.attn_norm, in_q, N, hp.rms_eps);
                    }
                    ink_src = in_q;
                    inv_src = in_q;
                } else {
                    if (fuse_q8_norm) {
                        k_rms_norm_3_q8_1<<<dim3(3, (unsigned) T), rms_q8_blk, 0, st>>>(
                            cur, cur, cur, in_q, in_k, in_v, L.attn_q_inp_norm, L.attn_k_inp_norm, L.attn_v_inp_norm,
                            xq, xq_k, xq_v, N, hp.rms_eps);
                        q8_pre_done = true;
                    } else if (T > 8) {
                        k_rms_norm_3_f16<<<dim3(3, (unsigned) T), rms_f16_blk, 0, st>>>(
                            cur, cur, cur, in_q, in_k, in_v, L.attn_q_inp_norm, L.attn_k_inp_norm, L.attn_v_inp_norm,
                            act_bf16, act_bf16 + (size_t) T * N, nullptr, N, hp.rms_eps);
                    } else {
                        k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(cur, L.attn_q_inp_norm, in_q, N, hp.rms_eps);
                        k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(cur, L.attn_k_inp_norm, in_k, N, hp.rms_eps);
                        k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(cur, L.attn_v_inp_norm, in_v, N, hp.rms_eps);
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
            // C4: On sm_80+ (!bf16_x_f16), convert in_q -> BF16 ONCE and share across attn_sink_proj, attn_q, attn_gate, attn_rel!
            const uint16_t* bf16_q = nullptr;
            if (!bf16_x_f16 && T > 8 && act_bf16_only &&
                (L.attn_sink_proj.type == 30 || L.attn_q.type == 30 || L.attn_gate.type == 30 || L.attn_rel.type == 30)) {
                launch_f32_to_bf16(in_q, act_bf16_only, (int64_t) T * N);
                bf16_q = act_bf16_only;
            }
            const bool fused_pre_attn_t1 = (T == 1 && kv_q8 && hp.head_dim == 128 && hp.n_rel == 16 && !save_conv_trace);
            const bool fuse_sink_pre_attn = (fused_pre_attn_t1 && L.attn_sink_proj.type == 30 && (N & 1) == 0 && (N == 2304));
            if (!fuse_sink_pre_attn) {
                project_preq(in_q, xq, L.attn_sink_proj, sink, T, hp.n_head, N, f16_q, bf16_q);
            }
            float* pQ = Q;
            float* pGate = att_gate;
            float* pR = R;
            float* pK = K;
            float* pV = V;
            float* pDynK = dyn_k;
            float* pDynV = dyn_v;
            const int64_t rel_dim = hp.n_rel * hp.n_head;
            const int64_t dyn_dim = hp.n_conv * hp.n_head_kv;
            if (T == 1 && !has_msg && is_swa && L.attn_qgrkv.d != nullptr) {
                pQ = qgr_t1;
                pGate = qgr_t1 + q_dim;
                pR = qgr_t1 + 2 * q_dim;
                pK = pR + rel_dim;
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
                if (T == 1 && L.attn_qgr.d != nullptr) {
                    pQ = qgr_t1;
                    pGate = qgr_t1 + q_dim;
                    pR = qgr_t1 + 2 * q_dim;
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
                        const uint16_t* f16_v = (T > 8 && inv_src == in_q) ? f16_q : nullptr;
                        project_preq(inv_src, xq_v_src, L.attn_v, V, T, kv_dim, N, f16_v);
                    }
                }
                // Dynamic conv projections (BF16 on !is_swa): reuse f16_k_pre (act_bf16 + T*N) and f16_v (act_bf16 when attn_v wrote FP16)!
                if (!is_swa && !(T == 1 && L.attn_kkc.d != nullptr)) {
                    project_preq(ink_src, xq_k_src, L.attn_k_conv_proj, dyn_k, T, dyn_dim, N, bf16_x_f16 ? f16_k_pre : nullptr);
                }
                if (!is_swa && !(T == 1 && L.attn_vvc.d != nullptr)) {
                    const uint16_t* f16_v_reuse = (T > 8 && bf16_x_f16 && (inv_src == in_q || L.attn_v.type != 0)) ? act_bf16 : nullptr;
                    project_preq(inv_src, xq_v_src, L.attn_v_conv_proj, dyn_v, T, dyn_dim, N, f16_v_reuse);
                }
            }

            float* hst = conv_hist[(size_t) il];
            float* h_tr = save_conv_trace ? (d_conv_trace + (size_t) il * layer_conv_floats) : nullptr;
            if (prof_detail) { CK(cudaStreamSynchronize(st)); p_d_qkv += ms_since(tp_d1); }
            const auto tp_d2 = Clock::now();
            const int apply_tau = is_swa ? 0 : 1;
            const int max_rel = is_swa ? std::min<int>((int) hp.n_swa, (int) hp.n_rel_ext) : (int) hp.n_rel_ext;
            if (fused_pre_attn_t1) {
                const int d_splits = (max_rel > 512) ? 4 : 2;
                const unsigned n_ctas = (unsigned) (2 * (int) hp.n_head_kv + (int) hp.n_head * d_splits);
                k_pre_attn_t1_fused<<<n_ctas, 128, 0, st>>>(
                    pK, pV, L.attn_k_conv, L.attn_v_conv, is_swa ? nullptr : pDynK, is_swa ? nullptr : pDynV,
                    hst, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il],
                    pQ, L.attn_q_clip, hp.qk_clip, pR, L.attn_rel_clip, hp.rel_clip,
                    L.attn_rel_bias, sink, rel_tab,
                    (int) hp.n_head, (int) hp.n_head_kv, (int) hp.n_rel_ext, max_rel,
                    kv_cap[(size_t) il], is_swa ? 1 : 0, hp.rms_eps, apply_tau, pos0_d,
                    (int) hp.log_scale_floor, hp.log_scale_alpha, d_splits,
                    fuse_sink_pre_attn ? (const uint16_t*) L.attn_sink_proj.d : nullptr,
                    fuse_sink_pre_attn ? in_q : nullptr, fuse_sink_pre_attn ? N : 0);
            } else {
            const bool fused_kv_q8 = (T < 32 && kv_q8 && hp.head_dim == 128);
            if (fused_kv_q8) {
                k_shortconv_kv_clip_norm_store_q8<<<dim3((unsigned) hp.n_head_kv, 2), 128, 0, st>>>(
                    pK, pV, L.attn_k_conv, L.attn_v_conv, is_swa ? nullptr : pDynK, is_swa ? nullptr : pDynV,
                    hst, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il],
                    (int) hp.n_head_kv, hp.qk_clip, hp.rms_eps, pos0_d, kv_cap[(size_t) il], is_swa ? 1 : 0,
                    T, h_tr, total_conv_floats);
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

            if (!fused_kv_q8) {
                if (kv_q8) {
                    k_clip_norm_kv_store_q8<<<dim3((unsigned) hp.n_head_kv, (unsigned) T), 32, 0, st>>>(
                        K_conv, V_conv, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il],
                        (int) hp.n_head_kv, hp.qk_clip, hp.rms_eps, pos0_d, kv_cap[(size_t) il], is_swa ? 1 : 0);
                } else {
                    k_clip_and_tau<<<dim3((unsigned) hp.n_head_kv, (unsigned) T), 32, 0, st>>>(
                        K_conv, nullptr, hp.qk_clip, (int) hp.head_dim, (int) hp.n_head_kv, 0, pos0_d, (int) hp.log_scale_floor, hp.log_scale_alpha);
                    k_v_rms_norm<<<dim3((unsigned) hp.n_head_kv, (unsigned) T), 32, 0, st>>>(V_conv, (int) hp.head_dim, (int) hp.n_head_kv, hp.rms_eps);
                    k_kv_store<<<dim3((unsigned) ((kv_dim + 255) / 256), (unsigned) T), 256, 0, st>>>(
                        K_conv, V_conv, kc[(size_t) il], vc[(size_t) il], (int) kv_dim, pos0_d, kv_cap[(size_t) il], is_swa ? 1 : 0);
                }
            }

            const bool fuse_swa_rel_pp = kv_q8 && is_swa && can_fa2_192 && T >= 128 &&
                                         hp.n_rel == 16 && hp.n_head == 3 * hp.n_head_kv && hp.head_dim == 128;
            if (!fuse_swa_rel_pp) {
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
            const bool fuse_swa_rel_pp = kv_q8 && is_swa && can_fa2_192 && T >= 128 &&
                                         hp.n_rel == 16 && hp.n_head == 3 * hp.n_head_kv && hp.head_dim == 128;
            const float kq_scale = 1.0f / std::sqrt((float) hp.head_dim);
            const void* att_xq_ready = nullptr;
            const uint16_t* att_f16_ready = nullptr;
            const bool want_att_f16 = (T >= 128 && act_bf16 != nullptr && q_dim <= 2 * N &&
                                       L.attn_output.type != 0 && (L.attn_output.type != 30 || bf16_x_f16));
            if (kv_q8) {
                if (hp.n_head == 3 * hp.n_head_kv && hp.head_dim == 128) {
                    const int sp_cap = std::max(1, attn_split_slots / std::max(1, T));
                    if (is_swa && T >= 2 && T <= 5) {
                        const int sp = std::min(sp_cap, std::clamp((sm_count + (int) hp.n_head_kv - 1) / std::max(1, (int) hp.n_head_kv), 4, 16));
                        if (T == 5) {
                            k_attn_q8_gqa3_verify4_swa_wmma_split<5><<<dim3((unsigned) hp.n_head_kv, (unsigned) sp), 128, 23040, st>>>(
                                pQ, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il], rel_tab, sink, attn_split,
                                (int) hp.n_head, (int) hp.n_head_kv, kv_cap[(size_t) il], (int) hp.n_swa, (int) hp.n_rel_ext, pos0_d, kq_scale, sp);
                        } else if (T == 4) {
                            k_attn_q8_gqa3_verify4_swa_wmma_split<4><<<dim3((unsigned) hp.n_head_kv, (unsigned) sp), 128, 23040, st>>>(
                                pQ, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il], rel_tab, sink, attn_split,
                                (int) hp.n_head, (int) hp.n_head_kv, kv_cap[(size_t) il], (int) hp.n_swa, (int) hp.n_rel_ext, pos0_d, kq_scale, sp);
                        } else if (T == 3) {
                            k_attn_q8_gqa3_verify4_swa_wmma_split<3><<<dim3((unsigned) hp.n_head_kv, (unsigned) sp), 128, 23040, st>>>(
                                pQ, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il], rel_tab, sink, attn_split,
                                (int) hp.n_head, (int) hp.n_head_kv, kv_cap[(size_t) il], (int) hp.n_swa, (int) hp.n_rel_ext, pos0_d, kq_scale, sp);
                        } else {
                            k_attn_q8_gqa3_verify4_swa_wmma_split<2><<<dim3((unsigned) hp.n_head_kv, (unsigned) sp), 128, 23040, st>>>(
                                pQ, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il], rel_tab, sink, attn_split,
                                (int) hp.n_head, (int) hp.n_head_kv, kv_cap[(size_t) il], (int) hp.n_swa, (int) hp.n_rel_ext, pos0_d, kq_scale, sp);
                        }
                        k_attn_split_combine<<<dim3((unsigned) hp.n_head, (unsigned) T), 128, 0, st>>>(
                            attn_split, pGate, att, (int) hp.n_head, sp, pos0_d, 1, (int) hp.n_swa, xq);
                        att_xq_ready = xq;
                    } else if (is_swa && T >= 6 && T <= 8) {
                        k_attn_q8_gqa3_swa_1pass<<<dim3((unsigned) hp.n_head, (unsigned) T), 128, 0, st>>>(
                            pQ, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il], rel_tab, sink,
                            pGate, att, xq, (int) hp.n_head, (int) hp.n_head_kv, (int) hp.n_rel_ext,
                            kv_cap[(size_t) il], (int) hp.n_swa, pos0_d, kq_scale);
                        att_xq_ready = xq;
                    } else if (T >= 2 && T <= 5 && !is_swa) {
                        const int sp = std::min(sp_cap, std::clamp((sm_count * 24) / std::max(1, (int) hp.n_head_kv), 40, kAttnSplitMax));
                        if (T == 5) {
                            k_attn_q8_gqa3_verify4_wmma_split<5><<<dim3((unsigned) hp.n_head_kv, (unsigned) sp), 128, 23040, st>>>(
                                pQ, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il], rel_tab, sink, attn_split,
                                (int) hp.n_head, (int) hp.n_head_kv, kv_cap[(size_t) il], (int) hp.n_rel_ext, pos0_d, kq_scale, sp);
                        } else if (T == 4) {
                            k_attn_q8_gqa3_verify4_wmma_split<4><<<dim3((unsigned) hp.n_head_kv, (unsigned) sp), 128, 23040, st>>>(
                                pQ, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il], rel_tab, sink, attn_split,
                                (int) hp.n_head, (int) hp.n_head_kv, kv_cap[(size_t) il], (int) hp.n_rel_ext, pos0_d, kq_scale, sp);
                        } else if (T == 3) {
                            k_attn_q8_gqa3_verify4_wmma_split<3><<<dim3((unsigned) hp.n_head_kv, (unsigned) sp), 128, 23040, st>>>(
                                pQ, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il], rel_tab, sink, attn_split,
                                (int) hp.n_head, (int) hp.n_head_kv, kv_cap[(size_t) il], (int) hp.n_rel_ext, pos0_d, kq_scale, sp);
                        } else {
                            k_attn_q8_gqa3_verify4_wmma_split<2><<<dim3((unsigned) hp.n_head_kv, (unsigned) sp), 128, 23040, st>>>(
                                pQ, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il], rel_tab, sink, attn_split,
                                (int) hp.n_head, (int) hp.n_head_kv, kv_cap[(size_t) il], (int) hp.n_rel_ext, pos0_d, kq_scale, sp);
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
                            pQ, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il], rel_tab, sink, attn_split,
                            (int) hp.n_head, (int) hp.n_head_kv, (int) hp.n_rel_ext, kv_cap[(size_t) il], is_swa ? 1 : 0, (int) hp.n_swa, pos0_d, kq_scale, sp);
                        k_attn_split_combine<<<dim3((unsigned) hp.n_head, (unsigned) T), 128, 0, st>>>(
                            attn_split, pGate, att, (int) hp.n_head, sp, pos0_d, is_swa ? 1 : 0, (int) hp.n_swa, (T <= 8) ? xq : nullptr);
                        if (T <= 8) att_xq_ready = xq;
                    } else if (can_fa2_256 && !is_swa && T >= 256 && pos >= 8192 &&
                               pp_sp_buf && pp_sp_bytes >= (size_t) T * (size_t) hp.n_head * 130 * sizeof(float) * 2) {
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
                        const int q_tiles_128 = (T + 127) / 128;
                        k_attn_q8_gqa3_pp_fa2_256_bulk<<<dim3((unsigned) (3 * q_tiles), (unsigned) hp.n_head_kv, (unsigned) best_sp), 256, (size_t) fa2_256_bulk_smem, st>>>(
                            pQ, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il], sink, pp_sp_buf,
                            (int) hp.n_head, (int) hp.n_head_kv, kv_cap[(size_t) il], bulk_aligned, kq_scale, T, best_sp, total_sp);
                        k_attn_q8_gqa3_pp_fa2_256_tail<<<dim3((unsigned) (3 * q_tiles_128), (unsigned) hp.n_head_kv), 256, (size_t) fa2_256_tail_smem, st>>>(
                            pQ, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il], rel_tab, sink, pp_sp_buf,
                            (int) hp.n_head, (int) hp.n_head_kv, kv_cap[(size_t) il], (int) hp.n_rel_ext, bulk_aligned, pos0_d, kq_scale, T, best_sp, total_sp);
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
                                pQ, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il], rel_tab, sink, pGate, att, pp_sp_buf,
                                (int) hp.n_head, (int) hp.n_head_kv, (int) hp.n_rel_ext, kv_cap[(size_t) il], (int) hp.n_swa, pos0_d, kq_scale, T, best_sp);
                            k_attn_pp_split_combine<<<dim3((unsigned) hp.n_head, (unsigned) T), 128, 0, st>>>(
                                pp_sp_buf, pGate, att, (int) hp.n_head, best_sp, want_att_f16 ? act_bf16 : nullptr);
                            if (want_att_f16) att_f16_ready = act_bf16;
                        } else if (is_swa && fuse_swa_rel_pp && can_fa2_192_swa_fast) {
                            k_attn_q8_gqa3_pp_fa2_192_swa_fast<<<dim3((unsigned) q_tiles, (unsigned) hp.n_head_kv), 384, (size_t) fa2_192_swa_smem, st>>>(
                                pQ, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il], sink, pGate, att,
                                (int) hp.n_head, (int) hp.n_head_kv, (int) hp.n_rel_ext, kv_cap[(size_t) il], (int) hp.n_swa, pos0_d, kq_scale, T,
                                pR, L.attn_rel_bias, want_att_f16 ? act_bf16 : nullptr);
                            if (want_att_f16) att_f16_ready = act_bf16;
                        } else if (is_swa && fuse_swa_rel_pp) {
                            k_attn_q8_gqa3_pp_fa2_192<false, true, true><<<dim3((unsigned) q_tiles, (unsigned) hp.n_head_kv), 256, 97280, st>>>(
                                pQ, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il], nullptr, sink, pGate, att, nullptr,
                                (int) hp.n_head, (int) hp.n_head_kv, (int) hp.n_rel_ext, kv_cap[(size_t) il], (int) hp.n_swa, pos0_d, kq_scale, T, 1,
                                pR, L.attn_rel_bias, want_att_f16 ? act_bf16 : nullptr);
                            if (want_att_f16) att_f16_ready = act_bf16;
                        } else if (is_swa) {
                            k_attn_q8_gqa3_pp_fa2_192<false, true, false><<<dim3((unsigned) q_tiles, (unsigned) hp.n_head_kv), 256, 97280, st>>>(
                                pQ, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il], rel_tab, sink, pGate, att, nullptr,
                                (int) hp.n_head, (int) hp.n_head_kv, (int) hp.n_rel_ext, kv_cap[(size_t) il], (int) hp.n_swa, pos0_d, kq_scale, T, 1,
                                nullptr, nullptr, want_att_f16 ? act_bf16 : nullptr);
                            if (want_att_f16) att_f16_ready = act_bf16;
                        } else {
                            k_attn_q8_gqa3_pp_fa2_192<false, false, false><<<dim3((unsigned) q_tiles, (unsigned) hp.n_head_kv), 256, 97280, st>>>(
                                pQ, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il], rel_tab, sink, pGate, att, nullptr,
                                (int) hp.n_head, (int) hp.n_head_kv, (int) hp.n_rel_ext, kv_cap[(size_t) il], (int) hp.n_swa, pos0_d, kq_scale, T, 1,
                                nullptr, nullptr, want_att_f16 ? act_bf16 : nullptr);
                            if (want_att_f16) att_f16_ready = act_bf16;
                        }
                    } else {
                        k_attn_q8_gqa3_pp_wmma<<<dim3((unsigned) hp.n_head, (unsigned) ((T + 15) / 16)), 128, 29184, st>>>(
                            pQ, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il], rel_tab, sink, pGate, att,
                            (int) hp.n_head, (int) hp.n_head_kv, (int) hp.n_rel_ext, kv_cap[(size_t) il], is_swa ? 1 : 0, (int) hp.n_swa, pos0_d, kq_scale, T);
                    }
                } else {
                    k_attn_q8<<<dim3((unsigned) hp.n_head, (unsigned) T), 128, 0, st>>>(
                        pQ, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il], rel_tab, sink, pGate, att,
                        (int) hp.n_head, (int) hp.n_head_kv, (int) hp.n_rel_ext, kv_cap[(size_t) il], is_swa ? 1 : 0, (int) hp.n_swa, pos0_d, kq_scale);
                }
            } else {
                k_attn<<<dim3((unsigned) hp.n_head, (unsigned) T), 128, 0, st>>>(
                    pQ, kc[(size_t) il], vc[(size_t) il], rel_tab, sink, att,
                    (int) hp.n_head, (int) hp.n_head_kv, (int) hp.n_rel_ext, kv_cap[(size_t) il], is_swa ? 1 : 0, (int) hp.n_swa, pos0_d, kq_scale);
            }
            if (prof_detail) { CK(cudaStreamSynchronize(st)); p_d_attn += ms_since(tp_d2); }

            const auto tp_d3 = Clock::now();
            if (!kv_q8) {
                k_mul_sigmoid<<<(unsigned) (((int64_t) T * q_dim + 255) / 256), 256, 0, st>>>(att, pGate, (int64_t) T * q_dim);
            }
            project_preq(att, att_xq_ready, L.attn_output, wo, T, N, q_dim, att_f16_ready);
            const bool do_push = (ip % hp.attn_res_block_size == 0);
            const bool fuse_post_attn_ffn_t1 = (T == 1 && !do_push && !has_msg && !save_conv_trace &&
                                                N == 2304 && (int) norm_blocks.size() + 1 <= 8);
            if (fuse_post_attn_ffn_t1) {
                const int B = (int) norm_blocks.size();
                NormMixPtrs np{};
                for (int i = 0; i < B; ++i) np.p[i] = norm_blocks[(size_t) (B - 1 - i)];
                np.p[B] = ffn_inp;
                const float inv_sqrt_n = 1.0f / std::sqrt((float) N);
                k_shortconv_mix_normed_fused_1pass_q8_t1<<<1, 576, 0, st>>>(
                    wo, L.attn_post_conv, hst, cur, ffn_inp, 2 * (int) kv_dim,
                    np, B + 1, L.ffn_res_query, in_ffn, L.ffn_norm, xq, N, inv_sqrt_n, hp.rms_eps);
            } else {
                if (T >= 32 && !save_conv_trace) {
                    k_shortconv_pp<<<dim3((unsigned) ((N + 255) / 256), (unsigned) ((T + 15) / 16)), 256, 0, st>>>(
                        wo, L.attn_post_conv, hst, ffn_inp, N, 2 * (int) kv_dim, T, do_push ? nullptr : cur);
                    k_shortconv_hist_push<<<(unsigned) ((N + 255) / 256), 256, 0, st>>>(
                        wo, hst, N, 2 * (int) kv_dim, T);
                } else {
                    k_shortconv<<<(unsigned) ((N + 63) / 64), 64, 0, st>>>(wo, L.attn_post_conv, nullptr, hst, ffn_inp, N, 2 * (int) kv_dim, 0, T, do_push ? nullptr : cur, h_tr, total_conv_floats);
                }
                if (do_push) {
                    const int pass_cur = (n_core > 0 && ip >= hp.core_start && ip < hp.core_end)
                                       ? ((il - (int) hp.core_start) / (int) n_core) : 0;
                    push_block(cur, il, ip, pass_cur);
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
                    run_mix_normed(norm_blocks, ffn_inp, false, L.ffn_res_query, ffn_mix_out, 1, T, 0);
                    if (has_msg) run_control(in_ffn, msg3, L, 3, in_ffn, T);
                    if (ffn_want_q8) {
                        k_rms_norm_q8_1<<<(unsigned) T, rms_q8_blk, 0, st>>>(in_ffn, L.ffn_norm, in_ffn, xq, N, hp.rms_eps);
                    } else if (T >= 16 && (N & 3) == 0) {
                        k_rms_norm_f16<<<(unsigned) T, rms_f16_blk, 0, st>>>(in_ffn, L.ffn_norm, in_ffn, act_bf16, N, hp.rms_eps);
                    } else {
                        k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(in_ffn, L.ffn_norm, in_ffn, N, hp.rms_eps);
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
            const bool fuse_comb_conv = (!is_hy && T <= 8);
            const PostConvSpec hy_conv{L.ffn_post_conv, hst, ffn_inp, cur, 2 * (int) kv_dim + N, h_tr, total_conv_floats};
            const bool want_pp_comb_conv = (T >= 32 && !save_conv_trace && (N & 3) == 0 && (!is_hy || pp_streaming));
            const bool post_conv_done = run_moe(L, il, in_ffn, T, pre_routed_t1, xq_ffn_ready, fuse_comb_conv,
                                                f16_ffn_ready, ((is_hy && T <= 8 && !pp_streaming) || want_pp_comb_conv) ? &hy_conv : nullptr);
            const auto tp_mp0 = Clock::now();
            if (post_conv_done) {
                // Already fused k_combine_experts[_hybrid]_shortconv directly into cur!
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
        pp_streaming = (hybrid && T >= pp_stream_t && pp_stage_cnt > 0 && pp_stage[0] != nullptr);
        if (pp_streaming) pp_begin(core_end_virt, core_all_gpu ? 1 : 0);
        for (int il = 0; il < hp.n_layer; ++il) {
            const bool use_t1_core = (!capturing_full_graph && !prof && T == 1 && !capture_dspark);
            const bool use_vf_core = (!capturing_full_graph && !prof && T >= 1 && T <= 8 && T <= verify_max_t &&
                                      capture_dspark && ((T == 1 && !save_conv_trace) || (T >= 2 && save_conv_trace)));
            if ((use_t1_core || use_vf_core) && core_all_gpu && !prof_detail && il == hp.core_start && n_core > 0) {
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
            const bool use_vf_pre_graph = (T >= 1 && T <= 8 && T <= verify_max_t && hybrid &&
                                           (is_outer || !core_all_gpu) && !prof_detail && !pp_streaming &&
                                           capture_dspark && ((T == 1 && !save_conv_trace) || (T >= 2 && save_conv_trace)));
            const bool use_pre_graph = use_t1_pre_graph || use_vf_pre_graph;
            const int ip_cur = hp.loop_layer(il);
            const bool include_route_t1 = use_pre_graph && (m->layers[(size_t) ip_cur].n_slots < hp.n_expert);
            if (include_route_t1 && m->layers[(size_t) ip_cur].n_slots > 0 &&
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
            run_layer_post(il, include_route_t1);
        }

        // Output head (skip 200K vocab matmul + sync on intermediate prefill chunks where only KV/DSpark states are needed)
        const auto tp_h0 = Clock::now();
        const bool need_all_head_rows = (rows == T) || (capture_dspark && dspark && d_tgt_feat[3]);
        const bool last_row_only = (compute_logits && !need_all_head_rows && T > 1);
        const bool fuse_head_q8 = (compute_logits && (rows == T || last_row_only) && rows <= 8 && (N & 31) == 0);
        const float* head_h = (rows == T) ? cur : (cur + (size_t) (T - 1) * N);
        if (compute_logits || (capture_dspark && dspark && d_tgt_feat[3])) {
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
                if (capture_dspark && dspark && d_tgt_feat[3]) {
                    CK(cudaMemcpyAsync(d_tgt_feat[3], cur, (size_t) T * N * sizeof(float), cudaMemcpyDeviceToDevice, st));
                }
            }
        }
        if (!compute_logits) {
            CK(cudaStreamSynchronize(st));
            if (pp_streaming) sync_d_hits(true);
            if (prof) p_head_ms += ms_since(tp_h0);
            pos += T;
            return 0;
        }
        project_preq(head_h, fuse_head_q8 ? xq : nullptr, m->output_weight, logits, rows, hp.vocab_unpadded, N);
        const int64_t tot_lg = (int64_t) rows * hp.vocab_unpadded;
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
            if (pp_streaming) {
                CK(cudaStreamSynchronize(st));
                sync_d_hits(true);
            }
            if (prof) p_head_ms += ms_since(tp_h0);
            pos += T;
            return 0;
        }
        if (copy_logits && out_logits && !defer_logits_d2h) {
            out_logits->resize((size_t) tot_lg);
            CK(cudaMemcpyAsync(out_logits->data(), logits, (size_t) tot_lg * sizeof(float), cudaMemcpyDeviceToHost, st));
        }
        int32_t last_pick = 0;
        if (want_argmax) {
            int32_t* pbuf = (h_pick_pin && rows <= verify_max_t) ? h_pick_pin : nullptr;
            std::vector<int32_t> tmp_picks;
            if (!pbuf) { tmp_picks.assign((size_t) rows, 0); pbuf = tmp_picks.data(); }
            CK(cudaMemcpyAsync(pbuf, pick_d, (size_t) rows * sizeof(int32_t), cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            if (out_picks) out_picks->assign(pbuf, pbuf + rows);
            last_pick = pbuf[rows - 1];
        } else {
            CK(cudaStreamSynchronize(st));
        }
        if (pp_streaming) sync_d_hits(true);
        if (prof) p_head_ms += ms_since(tp_h0);
        pos += T;
        return last_pick;
    }
};

bool place_experts(Model& m, Engine& E, double vram_gib, int threads, std::string& err) {
    const auto t0 = Clock::now();
    const auto& hp = m.hp;
    const int ne = (int) hp.n_expert;
    size_t fr = 0, tot = 0;
    CK(cudaMemGetInfo(&fr, &tot));
    const bool core_res_hy = (E.profile.regime == strata::inkling2::VramRegime::CoreResidentHybrid);
    // DSpark adds dflare_graph + markov_graphs + verify_core_graph; keep 288 MiB headroom when enabled.
    const size_t base_reserve = core_res_hy
        ? (E.verify_max_t > 1 ? ((size_t) 176 << 20) : ((size_t) 144 << 20))
        : ((size_t) 256 << 20);
    const size_t reserve = base_reserve + E.vram_reserve_bytes;
    size_t budget = vram_gib >= 0.0 ? (size_t) (vram_gib * (double) (1ull << 30))
                                    : (fr > reserve ? fr - reserve : 0);

    size_t stage_mb = (size_t) (E.profile.pp_stage_mb > 0 ? E.profile.pp_stage_mb : 64);
    if (const char* v = std::getenv("STRATA_PP_STAGE_MB")) stage_mb = (size_t) std::max(0, std::atoi(v));
    size_t max_exp_stride = 0;
    size_t all_exps_bytes = 0;
    for (int ip = 0; ip < hp.n_phys; ++ip) {
        max_exp_stride = std::max(max_exp_stride, m.layers[(size_t) ip].stride);
        all_exps_bytes += (size_t) ne * m.layers[(size_t) ip].stride;
    }
    // Keep extra headroom when hybrid-offloading so CUDA graphs + desktop fit OOM-free
    // (CoreResidentHybrid pre-allocates full KV & uses 1 compact core_graph, so 160 MiB reserve leaves ~135 MiB live free).
    if (vram_gib < 0.0 && budget < all_exps_bytes) {
        const size_t extra_resv = core_res_hy ? ((size_t) 16 << 20) : ((size_t) 32 << 20);
        budget = budget > extra_resv ? (budget - extra_resv) : 0;
    }
    if (budget < all_exps_bytes && stage_mb > 0 && max_exp_stride > 0) {
        size_t req_stage = stage_mb << 20;
        if (req_stage < max_exp_stride) req_stage = max_exp_stride;
        // Snap up by <= 256 KiB to the next exact multiple of NVFP4 L.stride (e.g. 64 MiB -> 15 * 4,479,232 = 67,188,480 B)
        // to avoid wasting ~4.2 MiB/slot and 1-2 expert tail chunks!
        for (int ip = 0; ip < hp.n_phys; ++ip) {
            const auto& L = m.layers[(size_t) ip];
            if (L.gu_type == 40 && L.stride > 0) {
                const size_t next_mult = ((req_stage + L.stride - 1) / L.stride) * L.stride;
                if (next_mult > req_stage && next_mult - req_stage <= ((size_t) 256 << 10)) {
                    req_stage = next_mult;
                }
                break;
            }
        }
        const int pref_cnt = std::clamp(E.profile.pp_stage_cnt_pref > 0 ? E.profile.pp_stage_cnt_pref : 3, 2, Engine::kMaxPpStages);
        const int want_cnt = (budget > req_stage * (size_t) pref_cnt + ((size_t) 512 << 20)) ? pref_cnt : 2;
        if (budget > req_stage * (size_t) want_cnt + ((size_t) 256 << 20)) {
            int ok_cnt = 0;
            for (int b = 0; b < want_cnt; ++b) {
                if (cudaMalloc(&E.pp_stage[b], req_stage) == cudaSuccess) {
                    ++ok_cnt;
                } else {
                    (void) cudaGetLastError();
                    break;
                }
            }
            if (ok_cnt >= 2) {
                E.pp_stage_cnt = ok_cnt;
                E.pp_stage_bytes = req_stage;
                budget -= req_stage * (size_t) ok_cnt;
            } else {
                for (int b = 0; b < ok_cnt; ++b) { cudaFree(E.pp_stage[b]); E.pp_stage[b] = nullptr; }
            }
        }
    }
    const int cpu_t = (E.pp_stage[0] && E.pp_stream_t > 1) ? std::min(E.tmax, E.pp_stream_t - 1) : E.tmax;
    // Defer init_cpu_bufs() until hybrid/CPU experts are actually needed (saves 72 MiB VRAM + 81 MiB pinned RAM on V100 Full-VRAM!).
    const bool will_need_cpu = (budget < all_exps_bytes);
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

    // Priority 2: Distribute remaining VRAM equally across tier2, then greedily fill leftover budget
    if (!tier2.empty() && budget > used) {
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
    size_t pinned_used = 0;
    size_t max_pin_bytes = (size_t) 7168 << 20;
#if !defined(_WIN32)
    {
        const long pages = sysconf(_SC_PHYS_PAGES);
        const long psize = sysconf(_SC_PAGESIZE);
        if (pages > 0 && psize > 0) {
            const size_t phys_ram = (size_t) pages * (size_t) psize;
            max_pin_bytes = std::clamp<size_t>((phys_ram * 2) / 5, (size_t) 7168 << 20, (size_t) 12288 << 20);
        }
        if (std::FILE* mf = std::fopen("/proc/meminfo", "r")) {
            char line[256];
            while (std::fgets(line, sizeof line, mf)) {
                unsigned long long avail_kb = 0;
                if (std::sscanf(line, "MemAvailable: %llu kB", &avail_kb) == 1) {
                    const size_t avail_b = (size_t) avail_kb * 1024ULL;
                    const size_t head_b = (size_t) 5120 << 20;
                    if (avail_b > head_b) {
                        const size_t safe_pin = avail_b - head_b;
                        max_pin_bytes = std::clamp<size_t>(safe_pin, (size_t) 4096 << 20, (size_t) 14336 << 20);
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

    for (int ip : alloc_order) {
        PhysLayer& L = m.layers[(size_t) ip];
        int s = slots[(size_t) ip];
        L.slot_of.assign((size_t) ne, -1);
        L.score.assign((size_t) ne, 0.0);
        if (!L.d_hits) {
            CK(cudaMalloc(&L.d_hits, (size_t) ne * sizeof(uint32_t)));
            CK(cudaMemset(L.d_hits, 0, (size_t) ne * sizeof(uint32_t)));
            m.gpu_allocs.push_back(L.d_hits);
        }

        if (s > 0) {
            void* dbl = nullptr;
            while (s > 0 && cudaMalloc(&dbl, (size_t) s * L.stride) != cudaSuccess) {
                (void) cudaGetLastError();
                --s;
            }
            if (s > 0 && dbl) {
                L.d_blobs = (uint8_t*) dbl;
                m.gpu_allocs.push_back(L.d_blobs);
                CK(cudaMalloc(&L.d_slot_of, (size_t) ne * sizeof(int32_t)));
                m.gpu_allocs.push_back(L.d_slot_of);
                actual_used += (size_t) s * L.stride;
            } else {
                s = 0;
            }
        }
        slots[(size_t) ip] = s;
        L.n_slots = s;
        L.exp_in_slot.assign((size_t) s, -1);

        if (s < ne) {
            need_cpu = true;
            const size_t hb_bytes = (size_t) ne * L.stride;
            const bool want_pin = (std::getenv("STRATA_NO_PIN_EXPERTS") == nullptr) && (pinned_used + hb_bytes <= max_pin_bytes);
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
        }

        if (s > 0) {
            if (L.hblobs) {
                CK(cudaMemcpyAsync(L.d_blobs, L.hblobs, (size_t) s * L.stride, cudaMemcpyHostToDevice, E.st));
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
                                for (int j = a; j < b; ++j) build_expert_blob(L, k0 + j, pbuf[bi] + (size_t) j * L.stride);
                            });
                        }
                        const int b0 = std::min(kn, per);
                        for (int j = 0; j < b0; ++j) build_expert_blob(L, k0 + j, pbuf[bi] + (size_t) j * L.stride);
                        for (auto& th : ths) th.join();
                    } else {
                        for (int j = 0; j < kn; ++j) build_expert_blob(L, k0 + j, pbuf[bi] + (size_t) j * L.stride);
                    }
                    CK(cudaMemcpyAsync(L.d_blobs + (size_t) k0 * L.stride, pbuf[bi],
                                       (size_t) kn * L.stride, cudaMemcpyHostToDevice, E.st));
                    CK(cudaEventRecord(ev_b[bi], E.st));
                }
                CK(cudaStreamSynchronize(E.st));
            } else {
                tmp.resize(L.L.bytes);
                for (int k = 0; k < s; ++k) {
                    build_expert_blob(L, k, tmp.data());
                    CK(cudaMemcpy(L.d_blobs + (size_t) k * L.stride, tmp.data(), L.L.bytes, cudaMemcpyHostToDevice));
                }
            }
            for (int k = 0; k < s; ++k) {
                L.slot_of[(size_t) k] = k;
                L.exp_in_slot[(size_t) k] = k;
            }
            CK(cudaMemcpyAsync(L.d_slot_of, L.slot_of.data(), (size_t) ne * sizeof(int32_t), cudaMemcpyHostToDevice, E.st));
            CK(cudaStreamSynchronize(E.st));
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
    // F5: Warm up cuBLAS (f16, bf16, sgemm) AFTER place_experts so zero VRAM is taken from Tier-2 experts while eliminating chunk-0 cold start!
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
                 "strata-inkling2: placed %d/%lld experts in VRAM (%.2f GiB, tier1=%d/128/layer, tier2=%d..%d/128/layer, free=%.0f MiB) in %.2f s\n",
                 total_vram_experts, (long long) hp.n_phys * ne, (double) used / (1 << 30),
                 tier1.empty() ? 0 : slots[(size_t) tier1[0]],
                 tier2.empty() ? 0 : tier2_min, tier2.empty() ? 0 : tier2_max,
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
                              std::vector<SamplingDist>& out_dists) {
    const int V = (int) E.m->hp.vocab_unpadded;
    if (!can_use_gpu_topk(smp, V) || rows <= 0 || rows > E.verify_max_t) return false;
    const int K = (smp.top_k > 0 && smp.top_k < V) ? smp.top_k : std::min(V, 256);
    std::vector<int32_t> p_toks, p_cnts;
    collect_penalty_counts(history, extra, n_extra, V, smp, p_toks, p_cnts);
    std::vector<float> vals((size_t) rows * (size_t) K);
    std::vector<int32_t> idxs((size_t) rows * (size_t) K);
    if (!E.gpu_topk_rows(rows, K, p_toks.data(), p_cnts.data(), (int) p_toks.size(),
                         smp.rep_pen, smp.freq_pen, smp.pres_pen, vals.data(), idxs.data())) {
        // Fallback: copy full logits D2H (note penalties were already applied in-place on GPU if n_pen > 0).
        std::vector<float> full((size_t) rows * (size_t) V);
        CK(cudaMemcpyAsync(full.data(), E.logits, full.size() * sizeof(float), cudaMemcpyDeviceToHost, E.st));
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

        if (adaptive_dspark && cur_dm == 0 && skip_cd > 0) {
            feed_single_cd(next_tok);
            --skip_cd;
            if (E.hybrid && (emitted % 8 == 0)) E.rebalance(E.core_all_gpu ? 4 : 8, 2, false);
            continue;
        }
        flush_cd_inject();

        const int use_dm = adaptive_dspark ? std::max(1, cur_dm) : draft_max;
        const int K = std::min(use_dm, rem - 1);
        const int32_t anchor = next_tok;
        if (use_gpu_topk) {
            q_dists.resize((size_t) K);
            dspark->draft(E, anchor, K, d_toks, nullptr, [&](int step_i, std::vector<float>&) {
                build_sampling_dists_gpu(E, 1, hist, d_toks.data(), step_i, smp, tmp_dists);
                q_dists[(size_t) step_i] = std::move(tmp_dists[0]);
                return sample_from_dist(q_dists[(size_t) step_i], rng);
            }, true);
        } else if (sample_on) {
            q_dists.resize((size_t) K);
            const size_t tail_keep = smp.pen_last_n > 0 ? (size_t) smp.pen_last_n : 0;
            const size_t h_start = hist.size() > tail_keep ? (hist.size() - tail_keep) : 0;
            dspark->draft(E, anchor, K, d_toks, &d_logits, [&](int step_i, std::vector<float>& row_lg) {
                std::vector<int32_t> tmp_hist(hist.begin() + (ptrdiff_t) h_start, hist.end());
                tmp_hist.insert(tmp_hist.end(), d_toks.begin(), d_toks.begin() + step_i);
                q_dists[(size_t) step_i] = build_sampling_dist(row_lg.data(), V, tmp_hist, smp);
                return sample_from_dist(q_dists[(size_t) step_i], rng);
            });
        } else {
            dspark->draft(E, anchor, K, d_toks, nullptr, [&](int, std::vector<float>&) -> int32_t { return 0; });
        }

        vtoks.resize((size_t) (1 + K));
        vtoks[0] = anchor;
        for (int i = 0; i < K; ++i) vtoks[(size_t) (1 + i)] = d_toks[(size_t) i];

        const int pos_pre = (int) E.pos;
        lg.clear();
        E.forward(vtoks.data(), 1 + K, sample_on, use_gpu_topk ? nullptr : &lg, sample_on, true, K,
                  sample_on ? nullptr : &t_picks, true, use_gpu_topk);
        if (use_gpu_topk) {
            build_sampling_dists_gpu(E, 1 + K, hist, nullptr, 0, smp, p_dists);
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
        dspark->n_accepted += m;
        dspark->inject(E, pos_pre, 1 + m);
        if (adaptive_dspark && K >= 1) {
            const int trials = std::min(K, m + 1);
            alpha_num = 0.85f * alpha_num + (float) m;
            alpha_den = 0.85f * alpha_den + (float) trials;
            const float alpha = alpha_num / std::max(1e-3f, alpha_den);
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
        dspark->init(*dm, max_ctx, chunk, E.full_kv_cap, E.layer_arena, E.layer_arena_bytes);
        E.enable_dspark(dspark.get());
        E.reset_state();
    }
    if (!place_experts(m, E, expert_vram, cpu_threads, err)) {
        std::fprintf(stderr, "strata-inkling2: expert placement failed: %s\n", err.c_str());
        return 1;
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
                const int eff_chunk = (E.in_decode_phase && fresh <= 64) ? Engine::kDecodePhaseCap : chunk;
                for (size_t off = reuse; off < prompt.size() && !stop_req.load(std::memory_order_acquire); off += (size_t) eff_chunk) {
                    const int tc = (int) std::min<size_t>((size_t) eff_chunk, prompt.size() - off);
                    const bool is_last = (off + (size_t) tc == prompt.size());
                    const int pos_pre = (int) E.pos;
                    const size_t img_cur_pre = img_cursor;
                    next_tok = E.forward(prompt.data() + off, tc, is_last && sample_on,
                                         (is_last && use_gpu_topk_pp) ? nullptr : &lg, false, dspark != nullptr, 0, nullptr,
                                         is_last, is_last && use_gpu_topk_pp,
                                         is_geni ? img_embs.data() : nullptr, img_embs_n,
                                         is_geni ? img_pad_id : -1, is_geni ? &img_cursor : nullptr);
                    if (is_last && use_gpu_topk_pp) {
                        std::vector<SamplingDist> d0;
                        build_sampling_dists_gpu(E, 1, hist, nullptr, 0, smp, d0);
                        next_tok = sample_from_dist(d0[0], rng);
                    }
                    if (dspark) dspark->inject(E, pos_pre, tc);
                    live_tokens.insert(live_tokens.end(), prompt.begin() + (ptrdiff_t) off, prompt.begin() + (ptrdiff_t) (off + (size_t) tc));
                    if (is_geni && img_cursor > img_cur_pre) {
                        live_img_sigs.insert(live_img_sigs.end(),
                                             prompt_img_sigs.begin() + (ptrdiff_t) img_cur_pre,
                                             prompt_img_sigs.begin() + (ptrdiff_t) img_cursor);
                    }
                    if (E.hybrid) {
                        if (E.rebal_burst > 0) { E.rebalance(192, 16); --E.rebal_burst; }
                        else if ((++E.pp_chunk_cnt & 3) == 0) E.rebalance(16, 2);
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
                const int avail = std::max(0, max_ctx - (int) prompt.size());
                const int limit = req_new > 0 ? std::min(req_new, avail) : avail;
                if (stop_req.load(std::memory_order_acquire)) {
                    finish = "cancel";
                } else {
                    if (E.hybrid && limit >= 16) E.set_phase_decode();
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

    if (bench) {
        // Warmup 1 pass
        E.reset_state();
        int32_t w = 0;
        for (size_t off = 0; off < prompt.size(); off += (size_t) chunk) {
            const int tc = (int) std::min<size_t>((size_t) chunk, prompt.size() - off);
            const int p0 = (int) E.pos;
            const bool is_last_w = (off + (size_t) tc == prompt.size());
            w = E.forward(prompt.data() + off, tc, false, nullptr, false, dspark != nullptr, 0, nullptr, is_last_w);
            if (dspark) dspark->inject(E, p0, tc);
        }
        if (E.hybrid) E.rebalance(192, 16);
        if (max_new > 0) {
            if (dspark && max_new >= 2) {
                std::vector<int32_t> wd;
                dspark->draft(E, w, std::min(draft_max, max_new - 1), wd, nullptr, [&](int, std::vector<float>&) -> int32_t { return 0; });
            } else {
                E.forward(&w, 1);
            }
        }
        E.reset_state();
        E.hy_gpu = E.hy_cpu = 0;
        E.hy_cpu_ms = 0.0;
        if (dspark) {
            dspark->n_draft_calls = dspark->n_drafted = dspark->n_accepted = 0;
            dspark->draft_ms = dspark->inject_ms = 0.0;
        }
    }

    E.prof = profile;
    E.prof_detail = profile_detail;
    E.reset_prof();
    std::vector<float> dumped;
    const auto t_pp0 = Clock::now();
    int32_t next_tok = 0;
    for (size_t off = 0; off < prompt.size(); off += (size_t) chunk) {
        const int tc = (int) std::min<size_t>((size_t) chunk, prompt.size() - off);
        const bool is_last = (off + (size_t) tc == prompt.size());
        const int p0 = (int) E.pos;
        next_tok = E.forward(prompt.data() + off, tc, is_last && !dump_logits_path.empty(), &dumped,
                             false, dspark != nullptr, 0, nullptr, is_last);
        if (dspark) dspark->inject(E, p0, tc);
        const auto tr0 = Clock::now();
        if (E.hybrid) E.rebalance(8);
        if (profile) { CK(cudaStreamSynchronize(E.st)); E.p_rebal_ms += ms_since(tr0); }
    }
    const double pp_ms = ms_since(t_pp0);
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
        if (E.hybrid && max_new >= 16) E.set_phase_decode();
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
        const double acc_rate = dspark->n_drafted > 0 ? (100.0 * (double) dspark->n_accepted / (double) dspark->n_drafted) : 0.0;
        const double mean_len = dspark->n_draft_calls > 0 ? (1.0 + (double) dspark->n_accepted / (double) dspark->n_draft_calls) : 1.0;
        std::fprintf(stderr,
                     "strata-inkling2: pp %zu tok in %.1f ms (%.2f t/s) | tg %d tok in %.1f ms (%.2f t/s) | dspark acc=%lld/%lld (%.1f%%, %.2f tok/step) | hits=%lld misses=%lld cpu_ms=%.1f\n",
                     prompt.size(), pp_ms, pp_tps, std::max(0, (int) gen.size() - 1), tg_ms, tg_tps,
                     (long long) dspark->n_accepted, (long long) dspark->n_drafted, acc_rate, mean_len,
                     (long long) E.hy_gpu, (long long) E.hy_cpu, E.hy_cpu_ms);
    } else {
        std::fprintf(stderr, "strata-inkling2: pp %zu tok in %.1f ms (%.2f t/s) | tg %d tok in %.1f ms (%.2f t/s) | hits=%lld misses=%lld cpu_ms=%.1f\n",
                     prompt.size(), pp_ms, pp_tps, std::max(0, (int) gen.size() - 1), tg_ms, tg_tps,
                     (long long) E.hy_gpu, (long long) E.hy_cpu, E.hy_cpu_ms);
    }
    return 0;
}

}  // namespace strata::inkling2

