// src/inkling2/inkling2_engine.cu - Standalone / integrated CUDA + AVX2 hybrid engine for Inkling-2-Mini (arch inkling2).
#include "strata/inkling2/engine.hpp"
#include "inkling2_kernels.cuh"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/gguf_split.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
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
    uint8_t* d_blobs = nullptr;          // [n_slots][stride] in VRAM
    size_t stride = 0;
    int n_slots = 0;
    int32_t* d_slot_of = nullptr;        // [n_expert] in VRAM
    std::vector<int32_t> slot_of;        // [n_expert] on host (-1 = CPU)
    std::vector<int32_t> exp_in_slot;    // [n_slots]
    std::vector<double> score;           // [n_expert]
};

struct Model {
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
    bool has_quant_dense = false;

    std::vector<PhysLayer> layers;
    std::vector<void*> gpu_allocs;
    std::vector<void*> host_allocs;

    ~Model() {
        for (void* p : gpu_allocs) cudaFree(p);
        for (void* p : host_allocs) std::free(p);
    }

    template <typename T>
    T* gpu_alloc(size_t count) {
        T* p = nullptr;
        CK(cudaMalloc(&p, count * sizeof(T)));
        gpu_allocs.push_back(p);
        return p;
    }

    const uint8_t* find_tensor(const char* name, const strata::TensorInfo** out = nullptr) const {
        size_t shard = 0;
        const strata::TensorInfo* t = gguf->find(name, &shard);
        if (out) *out = t;
        return t ? gguf->shard(shard).tensor_data(*t) : nullptr;
    }

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

    DenseMat upload_dense(const char* name, size_t rows_override = 0, size_t cols_expected = 0) {
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
        CK(cudaMemcpy(d, raw, bytes, cudaMemcpyHostToDevice));
        if (t->type != 30 && t->type != 0 && t->type != 1) has_quant_dense = true;
        return DenseMat{(int) t->type, ne0, ne1, d, bytes};
    }
};

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
    if (m.tok_embd_type == 30) m.tok_embd_row_bytes = (size_t) hp.n_embd * sizeof(uint16_t);
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
    m.output_weight = m.upload_dense("output.weight", (size_t) hp.vocab_unpadded, (size_t) hp.n_embd);

    m.layers.resize((size_t) hp.n_phys);
    char nm[128];
    for (int ip = 0; ip < hp.n_phys; ++ip) {
        PhysLayer& L = m.layers[(size_t) ip];
        L.is_swa = hp.is_swa(ip);
        L.has_loop = (hp.num_loops > 1 && ip >= hp.core_start && ip < hp.core_end);

        auto tn = [&](const char* suffix) -> const char* {
            std::snprintf(nm, sizeof nm, "blk.%d.%s", ip, suffix);
            return nm;
        };
        L.attn_q = m.upload_dense(tn("attn_q.weight"));
        L.attn_k = m.upload_dense(tn("attn_k.weight"));
        L.attn_v = m.upload_dense(tn("attn_v.weight"));
        L.attn_output = m.upload_dense(tn("attn_output.weight"));
        L.attn_gate = m.upload_dense(tn("attn_gate.weight"));
        L.attn_sink_proj = m.upload_dense(tn("attn_sink_proj.weight"));
        L.attn_rel = m.upload_dense(tn("attn_rel.weight"));

        if (L.is_swa) {
            L.attn_norm = m.upload_f32(tn("attn_norm.weight"), (size_t) hp.n_embd);
        } else {
            L.attn_q_inp_norm = m.upload_f32(tn("attn_q_inp_norm.weight"), (size_t) hp.n_embd);
            L.attn_k_inp_norm = m.upload_f32(tn("attn_k_inp_norm.weight"), (size_t) hp.n_embd);
            L.attn_v_inp_norm = m.upload_f32(tn("attn_v_inp_norm.weight"), (size_t) hp.n_embd);
            L.attn_k_conv_proj = m.upload_dense(tn("attn_k_conv_proj.weight"));
            L.attn_v_conv_proj = m.upload_dense(tn("attn_v_conv_proj.weight"));
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
    cublasHandle_t blas = nullptr;
    strata::prefill::Gemm gemm;

    int ctx = 16384;
    int tmax = 512;
    int64_t pos = 0;

    // KV caches & shortconv state (per virtual layer il = 0..47)
    bool kv_q8 = true;
    std::vector<__half*> kc, vc;
    std::vector<int8_t*> kq, vq;
    std::vector<__half*> ks, vs;
    std::vector<int> kv_cap;
    std::vector<float*> conv_hist;   // [48][n_sconv_channels * 3]

    // Activation scratch (all allocated up to tmax rows)
    float *cur = nullptr, *inpSA = nullptr, *ffn_inp = nullptr;
    float *in_q = nullptr, *in_k = nullptr, *in_v = nullptr, *in_ffn = nullptr;
    float *msg0 = nullptr, *msg1 = nullptr, *msg2 = nullptr, *msg3 = nullptr;
    float *ctrl2N = nullptr, *ctrl64 = nullptr, *ctrl_g = nullptr, *ctrl_f = nullptr;
    float *stack_norm = nullptr;     // [tmax * 8 * N]
    float *blk_buf[10] = {};         // intra-forward block residuals (non-aliasing across loop passes)

    float *Q = nullptr, *K = nullptr, *V = nullptr, *R = nullptr;
    float *K_conv = nullptr, *V_conv = nullptr, *dyn_k = nullptr, *dyn_v = nullptr;
    float *sink = nullptr, *rel_tab = nullptr, *att = nullptr, *att_gate = nullptr, *wo = nullptr;
    float *router_logits = nullptr, *ewts = nullptr, *eparts = nullptr, *moe_raw = nullptr, *moe_conv = nullptr;
    float *logits = nullptr;
    float* h_tok_buf = nullptr;      // pinned [tmax * N] FP32 for token embedding upload
    uint16_t* act_bf16 = nullptr;    // [tmax * 2*N] (reused as BF16 or FP16 scratch)

    void *xq = nullptr, *escr = nullptr;
    int32_t *ids = nullptr, *grp_start = nullptr, *n_groups = nullptr, *ent_dst = nullptr, *ent_tok = nullptr;
    unsigned long long* grp_ptr = nullptr;
    int32_t *tok_d = nullptr, *pos0_d = nullptr, *pick_d = nullptr;

    // Hybrid CPU pool staging
    int32_t* h_ids = nullptr;        // [tmax * top_k]
    float* h_h = nullptr;            // [tmax * N]
    float* cpu_out = nullptr;        // [tmax * top_k * N]
    int32_t* h_miss = nullptr;       // [tmax * top_k]
    int32_t* d_miss = nullptr;
    float* ecpu = nullptr;           // [tmax * top_k * N]
    uint8_t* acts = nullptr;         // [tmax * kNativeActBytes]
    cudaEvent_t ev_route = nullptr;
    bool hybrid = false;

#if defined(STRATA_INKLING2_CPU_POOL)
    std::unique_ptr<strata::kernels::cpu::ExpertPool> pool;
    std::vector<strata::kernels::cpu::ExpertJobMulti> jobs;
#endif

    // Stats
    int64_t hy_gpu = 0, hy_cpu = 0;
    double hy_cpu_ms = 0.0;
    int placed_slots = 0;
    size_t placed_bytes = 0;
    size_t vram_free_after_place = 0;
    int pool_workers = 0;

    // Cached decode CUDA Graph for the contiguous 36-layer core (il = 6..41) when blk.6..17 are 100% in VRAM
    cudaGraphExec_t core_graph = nullptr;
    bool core_all_gpu = false;

    // Prefix cache checkpoint (stores conv_hist + circular SWA KV state in host RAM; full-attn KV stays append-only on GPU)
    struct StateCheckpoint {
        std::vector<int32_t> tokens;
        std::vector<uint8_t> data;
    };

    StateCheckpoint save_checkpoint(const std::vector<int32_t>& tokens) {
        const auto& hp = m->hp;
        const size_t conv_bytes = (size_t) hp.n_sconv_channels() * 3 * sizeof(float);
        size_t total = (size_t) hp.n_layer * conv_bytes;
        for (int il = 0; il < hp.n_layer; ++il) {
            if (!hp.is_swa(il)) continue;
            const size_t cap = (size_t) kv_cap[(size_t) il];
            if (kv_q8) total += cap * (size_t) hp.kv_dim * 2 + cap * (size_t) hp.n_head_kv * sizeof(__half) * 2;
            else total += cap * (size_t) hp.kv_dim * sizeof(__half) * 2;
        }
        StateCheckpoint cp;
        cp.tokens = tokens;
        cp.data.resize(total);
        size_t off = 0;
        auto pull = [&](const void* src, size_t n) {
            CK(cudaMemcpyAsync(cp.data.data() + off, src, n, cudaMemcpyDeviceToHost, st));
            off += n;
        };
        for (int il = 0; il < hp.n_layer; ++il) {
            pull(conv_hist[(size_t) il], conv_bytes);
            if (!hp.is_swa(il)) continue;
            const size_t cap = (size_t) kv_cap[(size_t) il];
            if (kv_q8) {
                pull(kq[(size_t) il], cap * (size_t) hp.kv_dim);
                pull(vq[(size_t) il], cap * (size_t) hp.kv_dim);
                pull(ks[(size_t) il], cap * (size_t) hp.n_head_kv * sizeof(__half));
                pull(vs[(size_t) il], cap * (size_t) hp.n_head_kv * sizeof(__half));
            } else {
                pull(kc[(size_t) il], cap * (size_t) hp.kv_dim * sizeof(__half));
                pull(vc[(size_t) il], cap * (size_t) hp.kv_dim * sizeof(__half));
            }
        }
        CK(cudaStreamSynchronize(st));
        return cp;
    }

    void restore_checkpoint(const StateCheckpoint& cp) {
        const auto& hp = m->hp;
        const size_t conv_bytes = (size_t) hp.n_sconv_channels() * 3 * sizeof(float);
        size_t off = 0;
        auto push = [&](void* dst, size_t n) {
            CK(cudaMemcpyAsync(dst, cp.data.data() + off, n, cudaMemcpyHostToDevice, st));
            off += n;
        };
        for (int il = 0; il < hp.n_layer; ++il) {
            push(conv_hist[(size_t) il], conv_bytes);
            if (!hp.is_swa(il)) continue;
            const size_t cap = (size_t) kv_cap[(size_t) il];
            if (kv_q8) {
                push(kq[(size_t) il], cap * (size_t) hp.kv_dim);
                push(vq[(size_t) il], cap * (size_t) hp.kv_dim);
                push(ks[(size_t) il], cap * (size_t) hp.n_head_kv * sizeof(__half));
                push(vs[(size_t) il], cap * (size_t) hp.n_head_kv * sizeof(__half));
            } else {
                push(kc[(size_t) il], cap * (size_t) hp.kv_dim * sizeof(__half));
                push(vc[(size_t) il], cap * (size_t) hp.kv_dim * sizeof(__half));
            }
        }
        pos = (int64_t) cp.tokens.size();
    }

    ~Engine() {
        if (core_graph) cudaGraphExecDestroy(core_graph);
        for (auto* p : kc) cudaFree(p);
        for (auto* p : vc) cudaFree(p);
        for (auto* p : kq) cudaFree(p);
        for (auto* p : vq) cudaFree(p);
        for (auto* p : ks) cudaFree(p);
        for (auto* p : vs) cudaFree(p);
        for (auto* p : conv_hist) cudaFree(p);
        for (auto* p : blk_buf) cudaFree(p);
        cudaFree(cur); cudaFree(inpSA); cudaFree(ffn_inp);
        cudaFree(in_q); cudaFree(in_k); cudaFree(in_v); cudaFree(in_ffn);
        cudaFree(msg0); cudaFree(msg1); cudaFree(msg2); cudaFree(msg3);
        cudaFree(ctrl2N); cudaFree(ctrl64); cudaFree(ctrl_g); cudaFree(ctrl_f);
        cudaFree(stack_norm);
        cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(R);
        cudaFree(K_conv); cudaFree(V_conv); cudaFree(dyn_k); cudaFree(dyn_v);
        cudaFree(sink); cudaFree(rel_tab); cudaFree(att); cudaFree(att_gate); cudaFree(wo);
        cudaFree(router_logits); cudaFree(ewts); cudaFree(eparts); cudaFree(moe_raw); cudaFree(moe_conv);
        cudaFree(logits); cudaFree(act_bf16);
        cudaFree(xq); cudaFree(escr);
        cudaFree(ids); cudaFree(grp_start); cudaFree(n_groups); cudaFree(ent_dst); cudaFree(ent_tok); cudaFree(grp_ptr);
        cudaFree(tok_d); cudaFree(pos0_d); cudaFree(pick_d);
        cudaFree(d_miss); cudaFree(ecpu);
        cudaFreeHost(h_tok_buf);
        cudaFreeHost(h_ids); cudaFreeHost(h_h); cudaFreeHost(cpu_out); cudaFreeHost(h_miss);
        std::free(acts);
        if (ev_route) cudaEventDestroy(ev_route);
        if (blas) cublasDestroy(blas);
        if (st) cudaStreamDestroy(st);
    }

    bool init(Model& model, int max_ctx, int max_t, bool use_kv_q8, std::string& err) {
        m = &model;
        ctx = max_ctx;
        tmax = max_t;
        kv_q8 = use_kv_q8;
        const auto& hp = m->hp;
        const int64_t N = hp.n_embd;

        CK(cudaStreamCreate(&st));
        CKB(cublasCreate(&blas));
        CKB(cublasSetStream(blas, st));
        if (!gemm.init(st, m->has_quant_dense ? (size_t) (8 * 1024 * 1024) : 4096, err)) return false;

        // Allocate KV caches & recurrent history per virtual layer
        kc.resize((size_t) hp.n_layer, nullptr);
        vc.resize((size_t) hp.n_layer, nullptr);
        kq.resize((size_t) hp.n_layer, nullptr);
        vq.resize((size_t) hp.n_layer, nullptr);
        ks.resize((size_t) hp.n_layer, nullptr);
        vs.resize((size_t) hp.n_layer, nullptr);
        kv_cap.resize((size_t) hp.n_layer);
        conv_hist.resize((size_t) hp.n_layer);
        for (int il = 0; il < hp.n_layer; ++il) {
            const int cap = hp.is_swa(il) ? (int) hp.n_swa : ctx;
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
            CK(cudaMalloc(&conv_hist[(size_t) il], (size_t) hp.n_sconv_channels() * 3 * sizeof(float)));
        }

        auto d_f32 = [](float** p, size_t n) { CK(cudaMalloc(p, n * sizeof(float))); };
        for (int b = 0; b < 10; ++b) d_f32(&blk_buf[b], (size_t) tmax * N);
        d_f32(&cur, (size_t) tmax * N);
        d_f32(&inpSA, (size_t) tmax * N);
        d_f32(&ffn_inp, (size_t) tmax * N);
        d_f32(&in_q, (size_t) tmax * N);
        d_f32(&in_k, (size_t) tmax * N);
        d_f32(&in_v, (size_t) tmax * N);
        d_f32(&in_ffn, (size_t) tmax * N);
        d_f32(&msg0, (size_t) tmax * N);
        d_f32(&msg1, (size_t) tmax * N);
        d_f32(&msg2, (size_t) tmax * N);
        d_f32(&msg3, (size_t) tmax * N);
        d_f32(&ctrl2N, (size_t) tmax * 2 * N);
        d_f32(&ctrl64, (size_t) tmax * hp.loop_gate_rank);
        d_f32(&ctrl_g, (size_t) tmax * N);
        d_f32(&ctrl_f, (size_t) tmax * N);
        d_f32(&stack_norm, (size_t) tmax * 8 * N);

        d_f32(&Q, (size_t) tmax * hp.q_dim);
        d_f32(&K, (size_t) tmax * hp.kv_dim);
        d_f32(&V, (size_t) tmax * hp.kv_dim);
        d_f32(&R, (size_t) tmax * hp.n_rel * hp.n_head);
        d_f32(&K_conv, (size_t) tmax * hp.kv_dim);
        d_f32(&V_conv, (size_t) tmax * hp.kv_dim);
        d_f32(&dyn_k, (size_t) tmax * hp.n_conv * hp.n_head_kv);
        d_f32(&dyn_v, (size_t) tmax * hp.n_conv * hp.n_head_kv);
        d_f32(&sink, (size_t) tmax * hp.n_head);
        d_f32(&rel_tab, (size_t) tmax * hp.n_head * hp.n_rel_ext);
        d_f32(&att, (size_t) tmax * hp.q_dim);
        d_f32(&att_gate, (size_t) tmax * hp.q_dim);
        d_f32(&wo, (size_t) tmax * N);

        const int64_t max_ent = (int64_t) tmax * hp.n_expert_used;
        d_f32(&router_logits, (size_t) tmax * hp.n_expert);
        d_f32(&ewts, (size_t) max_ent);
        d_f32(&eparts, (size_t) max_ent * N);
        d_f32(&moe_raw, (size_t) tmax * N);
        d_f32(&moe_conv, (size_t) tmax * N);
        d_f32(&logits, (size_t) hp.vocab_unpadded);

        CK(cudaMallocHost(&h_tok_buf, (size_t) tmax * N * sizeof(float)));
        CK(cudaMalloc(&act_bf16, (size_t) tmax * 2 * N * sizeof(uint16_t)));

        CK(cudaMalloc(&xq, (size_t) tmax * ((2 * N) / 32) * 36));
        CK(cudaMalloc(&escr, K::native_expert_scratch_bytes(max_ent, hp.n_ff)));
        CK(cudaMalloc(&ids, (size_t) max_ent * sizeof(int32_t)));
        CK(cudaMalloc(&grp_start, (size_t) (hp.n_expert + 1) * sizeof(int32_t)));
        CK(cudaMalloc(&n_groups, sizeof(int32_t)));
        CK(cudaMalloc(&ent_dst, (size_t) max_ent * sizeof(int32_t)));
        CK(cudaMalloc(&ent_tok, (size_t) max_ent * sizeof(int32_t)));
        CK(cudaMalloc(&grp_ptr, (size_t) hp.n_expert * sizeof(unsigned long long)));
        CK(cudaMalloc(&tok_d, (size_t) tmax * sizeof(int32_t)));
        CK(cudaMalloc(&pos0_d, sizeof(int32_t)));
        CK(cudaMalloc(&pick_d, (size_t) tmax * sizeof(int32_t)));

        CK(cudaMallocHost(&h_ids, (size_t) max_ent * sizeof(int32_t)));
        CK(cudaMallocHost(&h_h, (size_t) tmax * N * sizeof(float)));
        CK(cudaMallocHost(&cpu_out, (size_t) max_ent * N * sizeof(float)));
        CK(cudaMallocHost(&h_miss, (size_t) max_ent * sizeof(int32_t)));
        CK(cudaMalloc(&d_miss, (size_t) max_ent * sizeof(int32_t)));
        CK(cudaMalloc(&ecpu, (size_t) max_ent * N * sizeof(float)));
#if defined(STRATA_INKLING2_CPU_POOL)
        acts = (uint8_t*) std::malloc((size_t) tmax * strata::kernels::cpu::kNativeActBytes);
#endif
        CK(cudaEventCreateWithFlags(&ev_route, cudaEventDisableTiming));
        reset_state();
        return true;
    }

    void reset_state() {
        pos = 0;
        for (int il = 0; il < m->hp.n_layer; ++il) {
            CK(cudaMemsetAsync(conv_hist[(size_t) il], 0, (size_t) m->hp.n_sconv_channels() * 3 * sizeof(float), st));
        }
    }

    void project(const float* X, const DenseMat& W, float* Y, int T, int64_t N_out, int64_t K_in) {
        if (W.type == 0) {
            project_f32(X, (const float*) W.d, Y, T, N_out, K_in);
            return;
        }
        if (W.type == 1) {
            const int64_t tot = (int64_t) T * K_in;
            k_f32_to_f16<<<(unsigned) ((tot + 255) / 256), 256, 0, st>>>(X, act_bf16, tot);
            gemm.f16(act_bf16, (const uint16_t*) W.d, Y, T, N_out, K_in);
            return;
        }
        if (W.type == 30) {
            const uint16_t* wp = (const uint16_t*) W.d;
            if (T == 1) {
                K::bf16_gemv_fp32_mmvf(X, wp, Y, K_in, N_out, st);
            } else {
                const int64_t tot = (int64_t) T * K_in;
                k_f32_to_bf16<<<(unsigned) ((tot + 255) / 256), 256, 0, st>>>(X, act_bf16, tot);
                gemm.bf16(act_bf16, wp, Y, T, N_out, K_in);
            }
            return;
        }
        const auto* ti = strata::quant::type_info(W.type);
        const int blck = ti ? ti->blck : 256;
        if (T <= 8 && (K_in % blck) == 0) {
            K::native_quantize_q8_1(X, xq, (int) K_in, T, st);
            K::native_mmvq(W.type, W.d, xq, Y, (int) K_in, (int) N_out, T, st);
            return;
        }
        const int64_t tot = (int64_t) T * K_in;
        k_f32_to_f16<<<(unsigned) ((tot + 255) / 256), 256, 0, st>>>(X, act_bf16, tot);
        gemm.native(act_bf16, W.type, W.d, Y, T, N_out, K_in);
    }

    void project_f32(const float* X, const float* W, float* Y, int T, int64_t N_out, int64_t K_in) {
        if (T == 1 && (K_in % 4) == 0) {
            k_gemv_f32<<<(unsigned) ((N_out + 7) / 8), 256, 0, st>>>(W, X, Y, (int) K_in, (int) N_out);
            return;
        }
        const float alpha = 1.0f, beta = 0.0f;
        CKB(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, (int) N_out, T, (int) K_in, &alpha, W, (int) K_in, X, (int) K_in, &beta, Y, (int) N_out));
    }

    void run_mix(const std::vector<float*>& blocks, float* c, const float* query, float* const* outs, int n_q, int T, int swa_bcast) {
        const auto& hp = m->hp;
        const int B = (int) blocks.size();
        const int D = B + 1;
        BlkPtrs bp{};
        for (int i = 0; i < B; ++i) bp.p[i] = blocks[(size_t) (B - 1 - i)];
        MixOuts mo{};
        for (int i = 0; i < (swa_bcast ? 3 : n_q); ++i) mo.p[i] = outs[i];
        k_stack_rms_norm<<<dim3((unsigned) D, (unsigned) T), kThreads, 0, st>>>(bp, c, stack_norm, D, (int) hp.n_embd, T, hp.rms_eps);
        const float inv_sqrt_n = 1.0f / std::sqrt((float) hp.n_embd);
        k_mix<<<dim3((unsigned) n_q, (unsigned) T), kThreads, 0, st>>>(stack_norm, query, mo, D, (int) hp.n_embd, n_q, inv_sqrt_n, swa_bcast);
    }

    void run_control(const float* c, const float* msg, const PhysLayer& L, int mod, float* out, int T) {
        const auto& hp = m->hp;
        const int N = (int) hp.n_embd;
        k_control_prep<<<dim3(2, (unsigned) T), kThreads, 0, st>>>(c, msg, ctrl2N, N, hp.rms_eps);
        project(ctrl2N, L.loop_down[mod], ctrl64, T, hp.loop_gate_rank, 2 * N);
        project(ctrl64, L.loop_gain[mod], ctrl_g, T, N, hp.loop_gate_rank);
        project(ctrl64, L.loop_forget[mod], ctrl_f, T, N, hp.loop_gate_rank);
        const int64_t tot = (int64_t) T * N;
        k_control_combine<<<(unsigned) ((tot + 255) / 256), 256, 0, st>>>(c, msg, ctrl_g, ctrl_f, out, tot);
    }

    void run_moe(PhysLayer& L, const float* x_norm, int T) {
        const auto& hp = m->hp;
        const int N = (int) hp.n_embd, Kp = (int) hp.n_expert_used;
        const int n_ent = T * Kp;
        project_f32(x_norm, L.ffn_gate_inp, router_logits, T, hp.n_expert, N);
        k_route_inkling2<<<(unsigned) T, (unsigned) hp.n_expert, 0, st>>>(router_logits, L.exp_probs_b, ids, ewts, (int) hp.n_expert, Kp, hp.expert_weights_scale);

        const bool has_gpu = L.n_slots > 0;
        const bool need_cpu = L.n_slots < hp.n_expert;

        if (need_cpu) {
            CK(cudaMemcpyAsync(h_ids, ids, (size_t) n_ent * sizeof(int32_t), cudaMemcpyDeviceToHost, st));
            CK(cudaMemcpyAsync(h_h, x_norm, (size_t) T * N * sizeof(float), cudaMemcpyDeviceToHost, st));
            CK(cudaEventRecord(ev_route, st));
        }

        if (has_gpu) {
            k_group_experts<<<1, 128, 0, st>>>(ids, L.d_slot_of, L.d_blobs, L.stride, n_ent, (int) hp.n_expert, L.n_slots,
                                               grp_ptr, grp_start, n_groups, ent_dst, ent_tok, Kp);
            K::quantize_q8_1_rows(x_norm, T, N, xq, st);
            K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                                     std::min<int64_t>(n_ent, hp.n_expert), n_ent, xq, escr, eparts, st);
        }

#if defined(STRATA_INKLING2_CPU_POOL)
        if (need_cpu) {
            namespace C = strata::kernels::cpu;
            while (cudaEventQuery(ev_route) == cudaErrorNotReady) {
            }
            const auto t0 = Clock::now();
            jobs.clear();
            int n_miss = 0;
            if (T == 1) {
                if (L.cpu_fmt.gu_act != 0) {
                    C::native_quant_act(L.cpu_fmt, h_h, acts);
                }
                const void* act0 = (L.cpu_fmt.gu_act == 0) ? (const void*) h_h : (const void*) acts;
                for (int e = 0; e < Kp; ++e) {
                    const int x = h_ids[e];
                    L.score[(size_t) x] += 1.0;
                    h_miss[e] = L.slot_of[(size_t) x] < 0 ? 1 : 0;
                    if (!h_miss[e]) continue;
                    ++n_miss;
                    C::ExpertJobMulti j;
                    j.blob = L.hblobs + (size_t) x * L.L.bytes;
                    j.nact[0] = act0;
                    j.out[0] = cpu_out + (size_t) e * N;
                    j.nt = 1;
                    jobs.push_back(j);
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
                        j.blob = L.hblobs + (size_t) x * L.L.bytes;
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
            if (n_miss > 0) {
                pool->run_split_multi_native(L.cpu_fmt, jobs.data(), (int) jobs.size());
                hy_cpu_ms += ms_since(t0);
                if (has_gpu) {
                    CK(cudaMemcpyAsync(ecpu, cpu_out, (size_t) n_ent * N * sizeof(float), cudaMemcpyHostToDevice, st));
                    CK(cudaMemcpyAsync(d_miss, h_miss, (size_t) n_ent * sizeof(int32_t), cudaMemcpyHostToDevice, st));
                    k_merge_cpu<<<(unsigned) n_ent, kThreads, 0, st>>>(eparts, ecpu, d_miss, N);
                } else {
                    CK(cudaMemcpyAsync(eparts, cpu_out, (size_t) n_ent * N * sizeof(float), cudaMemcpyHostToDevice, st));
                }
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
        k_combine_experts<<<dim3((unsigned) ((N + 255) / 256), (unsigned) T), 256, 0, st>>>(eparts, ewts, moe_raw, N, Kp);
    }

    void rebalance(int max_swaps) {
        if (!hybrid) return;
        const int64_t ne = m->hp.n_expert;
        int left = max_swaps;
        for (size_t li = 0; li < m->layers.size() && left > 0; ++li) {
            PhysLayer& L = m->layers[li];
            const int S = L.n_slots;
            if (S <= 0 || S >= ne) continue;
            std::vector<int> order((size_t) ne);
            for (int x = 0; x < ne; ++x) order[(size_t) x] = x;
            std::sort(order.begin(), order.end(), [&](int a, int b) { return L.score[(size_t) a] > L.score[(size_t) b]; });
            std::vector<int> victims;
            for (int s = 0; s < S; ++s) victims.push_back(s);
            std::sort(victims.begin(), victims.end(), [&](int a, int b) {
                return L.score[(size_t) L.exp_in_slot[(size_t) a]] < L.score[(size_t) L.exp_in_slot[(size_t) b]];
            });
            size_t vi = 0;
            bool changed = false;
            for (int i = 0; i < S && left > 0; ++i) {
                const int x = order[(size_t) i];
                if (L.slot_of[(size_t) x] >= 0) continue;
                while (vi < victims.size()) {
                    const int s = victims[vi++];
                    const int y = L.exp_in_slot[(size_t) s];
                    if (L.score[(size_t) y] >= L.score[(size_t) x]) { vi = victims.size(); break; }
                    bool y_top = false;
                    for (int j = 0; j < S; ++j) if (order[(size_t) j] == y) { y_top = true; break; }
                    if (y_top) continue;
                    L.slot_of[(size_t) y] = -1;
                    L.slot_of[(size_t) x] = s;
                    L.exp_in_slot[(size_t) s] = x;
                    CK(cudaMemcpyAsync(L.d_blobs + (size_t) s * L.stride, L.hblobs + (size_t) x * L.L.bytes,
                                       L.L.bytes, cudaMemcpyHostToDevice, st));
                    changed = true;
                    --left;
                    break;
                }
            }
            if (changed) {
                CK(cudaMemcpyAsync(L.d_slot_of, L.slot_of.data(), (size_t) ne * sizeof(int32_t), cudaMemcpyHostToDevice, st));
            }
            for (int x = 0; x < ne; ++x) L.score[(size_t) x] *= 0.5;
        }
    }

    int32_t forward(const int32_t* tokens, int T, bool copy_logits = false, std::vector<float>* out_logits = nullptr) {
        const auto& hp = m->hp;
        const int N = (int) hp.n_embd;
        const int64_t kv_dim = hp.kv_dim;
        const int64_t q_dim = hp.q_dim;
        const int64_t n_core = hp.num_loops > 1 ? (hp.core_end - hp.core_start) : 0;

        // 1. Token embedding from host mmap + tok_embd_norm
        for (int t = 0; t < T; ++t) {
            const uint8_t* row = m->host_tok_embd + (size_t) tokens[t] * m->tok_embd_row_bytes;
            float* dst = h_tok_buf + (size_t) t * N;
            if (m->tok_embd_type == 30) {
                const uint16_t* bf = (const uint16_t*) row;
                for (int i = 0; i < N; ++i) {
                    uint32_t u = (uint32_t) bf[i] << 16;
                    std::memcpy(&dst[i], &u, 4);
                }
            } else if (m->tok_embd_type == 0) {
                std::memcpy(dst, row, (size_t) N * sizeof(float));
            } else {
                strata::quant::dequant_row((strata::quant::Type) m->tok_embd_type, row, dst, N);
            }
        }
        CK(cudaMemcpyAsync(inpSA, h_tok_buf, (size_t) T * N * sizeof(float), cudaMemcpyHostToDevice, st));
        const int32_t pos32 = (int32_t) pos;
        CK(cudaMemcpyAsync(pos0_d, &pos32, sizeof(int32_t), cudaMemcpyHostToDevice, st));
        k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(inpSA, m->tok_embd_norm, cur, N, hp.rms_eps);

        std::vector<float*> blocks;
        int next_blk_buf = 0;
        auto push_block = [&](const float* src) {
            float* dst = blk_buf[next_blk_buf++];
            CK(cudaMemcpyAsync(dst, src, (size_t) T * N * sizeof(float), cudaMemcpyDeviceToDevice, st));
            blocks.push_back(dst);
        };

        std::vector<float*> prelude_blocks;
        float* prelude_cur = blk_buf[9];   // dedicated saved buffer for prelude_cur
        std::vector<float*> bank;
        float* bank_last = blk_buf[8];     // dedicated saved buffer for bank_last

        auto run_layer = [&](int il) {
            const int ip = hp.loop_layer(il);
            PhysLayer& L = m->layers[(size_t) ip];
            const bool is_swa = hp.is_swa(il);

            bool has_msg = false;
            if (n_core > 0 && ip >= hp.core_start && ip < hp.core_end) {
                const int pass = (il - (int) hp.core_start) / (int) n_core;
                if (ip == hp.core_start) {
                    if (pass == 0) {
                        prelude_blocks = blocks;
                        CK(cudaMemcpyAsync(prelude_cur, cur, (size_t) T * N * sizeof(float), cudaMemcpyDeviceToDevice, st));
                    } else {
                        bank.clear();
                        for (size_t bi = prelude_blocks.size() + 1; bi < blocks.size(); ++bi) bank.push_back(blocks[bi]);
                        CK(cudaMemcpyAsync(bank_last, cur, (size_t) T * N * sizeof(float), cudaMemcpyDeviceToDevice, st));
                        blocks = prelude_blocks;
                        CK(cudaMemcpyAsync(cur, prelude_cur, (size_t) T * N * sizeof(float), cudaMemcpyDeviceToDevice, st));
                    }
                }
                if (pass > 0) {
                    has_msg = true;
                    const int64_t row = 2 * (ip - hp.core_start);
                    const float* q_attn = m->loop_res_query + row * N;
                    const float* q_kv = m->loop_kv_res_query + row * N;
                    float* out_attn[2] = {msg0, msg3};
                    float* out_kv[2] = {msg1, msg2};
                    run_mix(bank, bank_last, q_attn, out_attn, 2, T, 0);
                    run_mix(bank, bank_last, q_kv, out_kv, 2, T, 0);
                }
            }

            CK(cudaMemcpyAsync(inpSA, cur, (size_t) T * N * sizeof(float), cudaMemcpyDeviceToDevice, st));

            if (L.attn_res_query) {
                float* outs[3] = {in_q, in_k, in_v};
                run_mix(blocks, cur, L.attn_res_query, outs, is_swa ? 1 : 3, T, is_swa ? 1 : 0);
            } else {
                CK(cudaMemcpyAsync(in_q, cur, (size_t) T * N * sizeof(float), cudaMemcpyDeviceToDevice, st));
                if (!is_swa || has_msg) {
                    CK(cudaMemcpyAsync(in_k, cur, (size_t) T * N * sizeof(float), cudaMemcpyDeviceToDevice, st));
                    CK(cudaMemcpyAsync(in_v, cur, (size_t) T * N * sizeof(float), cudaMemcpyDeviceToDevice, st));
                }
            }

            if (has_msg) {
                run_control(in_q, msg0, L, 0, in_q, T);
                run_control(in_k, msg1, L, 1, in_k, T);
                run_control(in_v, msg2, L, 2, in_v, T);
            }

            const float* ink_src = in_k;
            const float* inv_src = in_v;
            if (is_swa && has_msg) {
                k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(in_q, L.attn_norm, in_q, N, hp.rms_eps);
                k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(in_k, L.attn_norm, in_k, N, hp.rms_eps);
                k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(in_v, L.attn_norm, in_v, N, hp.rms_eps);
            } else if (is_swa) {
                k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(in_q, L.attn_norm, in_q, N, hp.rms_eps);
                ink_src = in_q;
                inv_src = in_q;
            } else {
                k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(in_q, L.attn_q_inp_norm, in_q, N, hp.rms_eps);
                k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(in_k, L.attn_k_inp_norm, in_k, N, hp.rms_eps);
                k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(in_v, L.attn_v_inp_norm, in_v, N, hp.rms_eps);
            }

            // Attention projections
            project(in_q, L.attn_q, Q, T, q_dim, N);
            project(ink_src, L.attn_k, K, T, kv_dim, N);
            project(inv_src, L.attn_v, V, T, kv_dim, N);
            project(in_q, L.attn_rel, R, T, hp.n_rel * hp.n_head, N);
            project(in_q, L.attn_sink_proj, sink, T, hp.n_head, N);
            project(in_q, L.attn_gate, att_gate, T, q_dim, N);

            float* hst = conv_hist[(size_t) il];
            if (!is_swa) {
                project(ink_src, L.attn_k_conv_proj, dyn_k, T, hp.n_conv * hp.n_head_kv, N);
                project(inv_src, L.attn_v_conv_proj, dyn_v, T, hp.n_conv * hp.n_head_kv, N);
            }
            k_shortconv<<<(unsigned) ((kv_dim + 255) / 256), 256, 0, st>>>(K, L.attn_k_conv, is_swa ? nullptr : dyn_k, hst, K_conv, (int) kv_dim, 0, (int) hp.n_head_kv, T);
            k_shortconv<<<(unsigned) ((kv_dim + 255) / 256), 256, 0, st>>>(V, L.attn_v_conv, is_swa ? nullptr : dyn_v, hst, V_conv, (int) kv_dim, (int) kv_dim, (int) hp.n_head_kv, T);

            const int apply_tau = is_swa ? 0 : 1;
            k_clip_and_tau<<<dim3((unsigned) hp.n_head, (unsigned) T), 32, 0, st>>>(
                Q, L.attn_q_clip, hp.qk_clip, (int) hp.head_dim, (int) hp.n_head, apply_tau, pos0_d, (int) hp.log_scale_floor, hp.log_scale_alpha);
            k_clip_and_tau<<<dim3((unsigned) hp.n_head_kv, (unsigned) T), 32, 0, st>>>(
                K_conv, nullptr, hp.qk_clip, (int) hp.head_dim, (int) hp.n_head_kv, 0, pos0_d, (int) hp.log_scale_floor, hp.log_scale_alpha);
            k_v_rms_norm<<<dim3((unsigned) hp.n_head_kv, (unsigned) T), 32, 0, st>>>(V_conv, (int) hp.head_dim, (int) hp.n_head_kv, hp.rms_eps);
            k_clip_and_tau<<<dim3((unsigned) hp.n_head, (unsigned) T), 32, 0, st>>>(
                R, L.attn_rel_clip, hp.rel_clip, (int) hp.n_rel, (int) hp.n_head, apply_tau, pos0_d, (int) hp.log_scale_floor, hp.log_scale_alpha);

            if (kv_q8) {
                k_kv_store_q8<<<dim3((unsigned) hp.n_head_kv, (unsigned) T), 32, 0, st>>>(
                    K_conv, V_conv, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il],
                    (int) hp.n_head_kv, pos0_d, kv_cap[(size_t) il], is_swa ? 1 : 0);
            } else {
                k_kv_store<<<dim3((unsigned) ((kv_dim + 255) / 256), (unsigned) T), 256, 0, st>>>(
                    K_conv, V_conv, kc[(size_t) il], vc[(size_t) il], (int) kv_dim, pos0_d, kv_cap[(size_t) il], is_swa ? 1 : 0);
            }

            k_rel_table<<<dim3((unsigned) hp.n_head, (unsigned) T), kThreads, 0, st>>>(
                R, L.attn_rel_bias, sink, rel_tab, (int) hp.n_head, (int) hp.n_rel, (int) hp.n_rel_ext, pos0_d, T);
            const float kq_scale = 1.0f / std::sqrt((float) hp.head_dim);
            if (kv_q8) {
                k_attn_q8<<<dim3((unsigned) hp.n_head, (unsigned) T), 128, 0, st>>>(
                    Q, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il], rel_tab, sink, att,
                    (int) hp.n_head, (int) hp.n_head_kv, (int) hp.n_rel_ext, kv_cap[(size_t) il], is_swa ? 1 : 0, (int) hp.n_swa, pos0_d, kq_scale);
            } else {
                k_attn<<<dim3((unsigned) hp.n_head, (unsigned) T), 128, 0, st>>>(
                    Q, kc[(size_t) il], vc[(size_t) il], rel_tab, sink, att,
                    (int) hp.n_head, (int) hp.n_head_kv, (int) hp.n_rel_ext, kv_cap[(size_t) il], is_swa ? 1 : 0, (int) hp.n_swa, pos0_d, kq_scale);
            }

            k_mul_sigmoid<<<(unsigned) (((int64_t) T * q_dim + 255) / 256), 256, 0, st>>>(att, att_gate, (int64_t) T * q_dim);
            project(att, L.attn_output, wo, T, N, q_dim);
            k_shortconv<<<(unsigned) ((N + 255) / 256), 256, 0, st>>>(wo, L.attn_post_conv, nullptr, hst, cur, N, 2 * (int) kv_dim, 0, T);

            if (ip % hp.attn_res_block_size == 0) {
                push_block(inpSA);
            } else {
                k_add<<<(unsigned) (((int64_t) T * N + 255) / 256), 256, 0, st>>>(cur, inpSA, (int64_t) T * N);
            }

            CK(cudaMemcpyAsync(ffn_inp, cur, (size_t) T * N * sizeof(float), cudaMemcpyDeviceToDevice, st));

            // MoE
            float* ffn_mix_out[1] = {in_ffn};
            run_mix(blocks, cur, L.ffn_res_query, ffn_mix_out, 1, T, 0);
            if (has_msg) run_control(in_ffn, msg3, L, 3, in_ffn, T);
            k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(in_ffn, L.ffn_norm, in_ffn, N, hp.rms_eps);
            run_moe(L, in_ffn, T);
            k_shortconv<<<(unsigned) ((N + 255) / 256), 256, 0, st>>>(moe_raw, L.ffn_post_conv, nullptr, hst, cur, N, 2 * (int) kv_dim + N, 0, T);
            k_add<<<(unsigned) (((int64_t) T * N + 255) / 256), 256, 0, st>>>(cur, ffn_inp, (int64_t) T * N);
        };

        const int core_end_virt = (int) (hp.core_start + hp.num_loops * n_core);
        for (int il = 0; il < hp.n_layer; ++il) {
            if (T == 1 && core_all_gpu && il == hp.core_start && n_core > 0) {
                if (!core_graph) {
                    cudaGraph_t g = nullptr;
                    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
                    for (int cl = il; cl < core_end_virt; ++cl) run_layer(cl);
                    CK(cudaStreamEndCapture(st, &g));
                    CK(cudaGraphInstantiate(&core_graph, g, nullptr, nullptr, 0));
                    CK(cudaGraphDestroy(g));
                } else {
                    blocks = {blk_buf[0], blk_buf[5], blk_buf[6]};
                    next_blk_buf = 7;
                }
                CK(cudaGraphLaunch(core_graph, st));
                il = core_end_virt - 1;
                continue;
            }
            run_layer(il);
        }

        // Output head
        float* out_head[1] = {in_ffn};
        run_mix(blocks, cur, m->output_res_query, out_head, 1, T, 0);
        k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(in_ffn, m->output_norm, cur, N, hp.rms_eps);
        // Only compute logits for the last row unless needed? Actually compute all if small or last row:
        const float* last_h = cur + (size_t) (T - 1) * N;
        project(last_h, m->output_weight, logits, 1, hp.vocab_unpadded, N);
        k_softcap<<<(unsigned) ((hp.vocab_unpadded + 255) / 256), 256, 0, st>>>(logits, hp.vocab_unpadded, hp.softcap);
        k_argmax<<<1, 256, 0, st>>>(logits, pick_d, (int) hp.vocab_unpadded);
        int32_t next_id = 0;
        if (copy_logits && out_logits) {
            out_logits->resize((size_t) hp.vocab_unpadded);
            CK(cudaMemcpyAsync(out_logits->data(), logits, (size_t) hp.vocab_unpadded * sizeof(float), cudaMemcpyDeviceToHost, st));
        }
        CK(cudaMemcpyAsync(&next_id, pick_d, sizeof(int32_t), cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        pos += T;
        return next_id;
    }
};

bool place_experts(Model& m, Engine& E, double vram_gib, int threads, std::string& err) {
    const auto t0 = Clock::now();
    const auto& hp = m.hp;
    const int ne = (int) hp.n_expert;
    size_t fr = 0, tot = 0;
    CK(cudaMemGetInfo(&fr, &tot));
    const size_t reserve = (size_t) 256 << 20;
    const size_t budget = vram_gib >= 0.0 ? (size_t) (vram_gib * (double) (1ull << 30))
                                          : (fr > reserve ? fr - reserve : 0);

    size_t used = 0;
    std::vector<int> slots(m.layers.size(), 0);

    // Priority 1: Looped core NVFP4 layers (blk.6..17, executed 3x/token!)
    std::vector<int> tier1, tier2;
    for (int ip = 0; ip < hp.n_phys; ++ip) {
        if (m.layers[(size_t) ip].gu_type == 30) continue;   // BF16 stays CPU-resident
        if (ip >= hp.core_start && ip < hp.core_end) tier1.push_back(ip);
        else tier2.push_back(ip);
    }

    size_t tier1_full = 0;
    for (int ip : tier1) tier1_full += (size_t) ne * m.layers[(size_t) ip].stride;
    if (!tier1.empty()) {
        if (used + tier1_full <= budget) {
            for (int ip : tier1) slots[(size_t) ip] = ne;
            used += tier1_full;
        } else {
            const size_t per_layer_stride = m.layers[(size_t) tier1[0]].stride;
            const int s = (int) std::min<size_t>((size_t) ne, (budget - used) / (tier1.size() * per_layer_stride));
            for (int ip : tier1) slots[(size_t) ip] = s;
            used += (size_t) s * tier1.size() * per_layer_stride;
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

    std::vector<uint8_t> tmp;
    bool need_cpu = false;
    int total_vram_experts = 0;
    for (int ip = 0; ip < hp.n_phys; ++ip) {
        PhysLayer& L = m.layers[(size_t) ip];
        const int s = slots[(size_t) ip];
        L.n_slots = s;
        L.slot_of.assign((size_t) ne, -1);
        L.exp_in_slot.assign((size_t) s, -1);
        L.score.assign((size_t) ne, 0.0);

        if (s < ne) {
            need_cpu = true;
            L.hblobs = (uint8_t*) std::malloc((size_t) ne * L.L.bytes);
            if (!L.hblobs) {
                err = "host malloc failed for layer " + std::to_string(ip);
                return false;
            }
            m.host_allocs.push_back(L.hblobs);
            for (int e = 0; e < ne; ++e) build_expert_blob(L, e, L.hblobs + (size_t) e * L.L.bytes);
        }

        if (s > 0) {
            CK(cudaMalloc(&L.d_blobs, (size_t) s * L.stride));
            m.gpu_allocs.push_back(L.d_blobs);
            CK(cudaMalloc(&L.d_slot_of, (size_t) ne * sizeof(int32_t)));
            m.gpu_allocs.push_back(L.d_slot_of);

            tmp.resize(L.L.bytes);
            for (int k = 0; k < s; ++k) {
                const uint8_t* src = L.hblobs ? (L.hblobs + (size_t) k * L.L.bytes) : (build_expert_blob(L, k, tmp.data()), tmp.data());
                CK(cudaMemcpy(L.d_blobs + (size_t) k * L.stride, src, L.L.bytes, cudaMemcpyHostToDevice));
                L.slot_of[(size_t) k] = k;
                L.exp_in_slot[(size_t) k] = k;
            }
            CK(cudaMemcpy(L.d_slot_of, L.slot_of.data(), (size_t) ne * sizeof(int32_t), cudaMemcpyHostToDevice));
            total_vram_experts += s;
        }
        if (s > 0 && s < ne) E.hybrid = true;
    }

#if defined(STRATA_INKLING2_CPU_POOL)
    if (need_cpu) {
        int nw = threads > 0 ? threads : 0;
        E.pool = std::make_unique<strata::kernels::cpu::ExpertPool>(nw, true, true);
        E.pool_workers = E.pool->workers();
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
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        std::stringstream ws(item);
        int32_t id;
        while (ws >> id) out.push_back(id);
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

int32_t sample_logits(std::vector<float>& logits, const std::vector<int32_t>& history, const SamplingParams& smp, std::mt19937_64& rng) {
    const int V = (int) logits.size();
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
            float& l = logits[(size_t) uniq[j]];
            if (smp.rep_pen != 1.0f) l = (l <= 0.0f) ? (l * smp.rep_pen) : (l / smp.rep_pen);
            l -= (float) counts[j] * smp.freq_pen + smp.pres_pen;
        }
    }
    if (smp.temp <= 0.0f) {
        int best = 0;
        for (int i = 1; i < V; ++i) if (logits[(size_t) i] > logits[(size_t) best]) best = i;
        return best;
    }
    const int K = (smp.top_k > 0 && smp.top_k < V) ? smp.top_k : std::min(V, 256);
    std::vector<int32_t> idx((size_t) V);
    for (int i = 0; i < V; ++i) idx[(size_t) i] = i;
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int32_t a, int32_t b) {
        return logits[(size_t) a] > logits[(size_t) b];
    });
    const float max_l = logits[(size_t) idx[0]];
    const float inv_t = 1.0f / smp.temp;
    std::vector<float> probs((size_t) K);
    double sum = 0.0;
    for (int i = 0; i < K; ++i) {
        const float p = std::exp((logits[(size_t) idx[(size_t) i]] - max_l) * inv_t);
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
    const double r = std::uniform_real_distribution<double>(0.0, ksum)(rng);
    double acc = 0.0;
    for (int i = 0; i < keep; ++i) {
        acc += probs[(size_t) i];
        if (r <= acc) return idx[(size_t) i];
    }
    return idx[0];
}

}  // namespace

int run(int argc, char** argv) {
    std::string model_path;
    std::string tokens_str = "1,2";
    std::string dump_logits_path;
    int max_ctx = 65536;
    int chunk = 256;
    int max_new = 32;
    int cpu_threads = 0;
    double expert_vram = -1.0;
    bool serve = false;
    bool bench = false;
    bool kv_q8 = true;

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
        else if (a == "--max-context") max_ctx = std::atoi(next_arg());
        else if (a == "--chunk" || a == "--prefill") chunk = std::atoi(next_arg());
        else if (a == "--max-new" || a == "-n") max_new = std::atoi(next_arg());
        else if (a == "--tokens") tokens_str = next_arg();
        else if (a == "--expert-vram") expert_vram = std::atof(next_arg());
        else if (a == "--cpu-threads" || a == "--pool-workers") cpu_threads = std::atoi(next_arg());
        else if (a == "--dump-logits") dump_logits_path = next_arg();
        else if (a == "--kv-f16") kv_q8 = false;
        else if (a == "--kv-q8") kv_q8 = true;
        else if (a == "--serve") serve = true;
        else if (a == "--bench") bench = true;
    }

    if (model_path.empty()) {
        std::fprintf(stderr, "usage: strata-inkling2 --model <file.gguf> [--max-context N] [--kv-q8|--kv-f16] [--tokens IDS] [--max-new N] [--dump-logits PATH] [--bench] [--serve]\n");
        return 2;
    }

    Model m;
    std::string err;
    if (!load_model(model_path, m, err)) {
        std::fprintf(stderr, "strata-inkling2: load failed: %s\n", err.c_str());
        return 1;
    }

    Engine E;
    if (!E.init(m, max_ctx, chunk, kv_q8, err)) {
        std::fprintf(stderr, "strata-inkling2: engine init failed: %s\n", err.c_str());
        return 1;
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

        std::printf("INFO engine=0.2.17-inkling2 context=%d kv=%s kv_resident=%d expert_slots=%d experts=%lld expert_cache_mib=%lld vram_free_mib=%lld pool_workers=%d\n",
                    max_ctx, kv_q8 ? "int8" : "fp16", max_ctx,
                    E.placed_slots, (long long) m.hp.n_phys * m.hp.n_expert,
                    (long long) (E.placed_bytes >> 20), (long long) (E.vram_free_after_place >> 20),
                    E.pool_workers);
        std::printf("READY %d stop\n", max_ctx);
        std::fflush(stdout);

        std::vector<float> lg;
        std::vector<int32_t> hist;
        std::vector<int32_t> live_tokens;
        std::vector<Engine::StateCheckpoint> snaps;
        constexpr size_t kMaxSnaps = 8;
        std::mt19937_64 rng(0x9e3779b97f4a7c15ULL);

        for (;;) {
            std::string line;
            {
                std::unique_lock<std::mutex> lk(q_mu);
                q_cv.wait(lk, [&] { return !q_lines.empty() || stdin_eof; });
                if (q_lines.empty()) break;
                line = std::move(q_lines.front());
                q_lines.pop_front();
            }
            if (line == "QUIT" || line == "EXIT") break;
            if (line.rfind("GEN ", 0) == 0 || line == "GEN") {
                stop_req.store(false, std::memory_order_release);
                std::istringstream iss(line.size() > 4 ? line.substr(4) : "");
                int req_new = 0;
                iss >> req_new;
                SamplingParams smp;
                std::vector<int32_t> prompt;
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
                        continue;
                    }
                    for (int32_t id : parse_id_list(tok)) prompt.push_back(id);
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

                // Prefix / KV cache reuse along live_tokens & saved SWA/conv checkpoints
                const size_t max_possible = prompt.size() - 1;
                size_t lcp_live = 0;
                const size_t lim_live = std::min(live_tokens.size(), max_possible);
                while (lcp_live < lim_live && live_tokens[lcp_live] == prompt[lcp_live]) ++lcp_live;

                size_t reuse = 0;
                if (lcp_live == live_tokens.size()) {
                    reuse = lcp_live;
                } else {
                    int best_idx = -1;
                    for (int si = (int) snaps.size() - 1; si >= 0; --si) {
                        const auto& stoks = snaps[(size_t) si].tokens;
                        if (stoks.size() <= lcp_live &&
                            std::equal(stoks.begin(), stoks.end(), prompt.begin())) {
                            best_idx = si;
                            break;
                        }
                    }
                    if (best_idx >= 0) {
                        E.restore_checkpoint(snaps[(size_t) best_idx]);
                        live_tokens = snaps[(size_t) best_idx].tokens;
                        snaps.resize((size_t) best_idx + 1);
                        reuse = live_tokens.size();
                    } else {
                        E.reset_state();
                        live_tokens.clear();
                        snaps.clear();
                        reuse = 0;
                    }
                }
                std::printf("RESUME %zu\n", reuse);
                std::fflush(stdout);

                const auto t_pp0 = Clock::now();
                E.hy_gpu = 0;
                E.hy_cpu = 0;
                E.hy_cpu_ms = 0.0;
                hist = prompt;
                const bool sample_on = smp.needs_logits();
                int32_t next_tok = 0;
                const size_t fresh = prompt.size() - reuse;
                for (size_t off = reuse; off < prompt.size() && !stop_req.load(std::memory_order_acquire); off += (size_t) chunk) {
                    const int tc = (int) std::min<size_t>((size_t) chunk, prompt.size() - off);
                    const bool is_last = (off + (size_t) tc == prompt.size());
                    next_tok = E.forward(prompt.data() + off, tc, is_last && sample_on, &lg);
                    live_tokens.insert(live_tokens.end(), prompt.begin() + (ptrdiff_t) off, prompt.begin() + (ptrdiff_t) (off + (size_t) tc));
                    if (E.hybrid) E.rebalance(8);
                    const double el_ms = ms_since(t_pp0);
                    const double done_fresh = (double) (off + (size_t) tc - reuse);
                    const double pp_tps_now = done_fresh / std::max(1e-6, el_ms / 1000.0);
                    std::printf("PP %zu %zu %.0f %.1f\n", off + (size_t) tc, prompt.size(), el_ms, pp_tps_now);
                    std::fflush(stdout);
                    if (!is_last && !stop_req.load(std::memory_order_acquire)) {
                        const size_t last_cp = snaps.empty() ? 0 : snaps.back().tokens.size();
                        if (live_tokens.size() - last_cp >= 2048) {
                            snaps.push_back(E.save_checkpoint(live_tokens));
                            if (snaps.size() > kMaxSnaps) snaps.erase(snaps.begin());
                        }
                    }
                }
                if (!stop_req.load(std::memory_order_acquire) && live_tokens.size() == prompt.size()) {
                    if (snaps.empty() || snaps.back().tokens.size() != live_tokens.size()) {
                        snaps.push_back(E.save_checkpoint(live_tokens));
                        if (snaps.size() > kMaxSnaps) snaps.erase(snaps.begin());
                    }
                }
                const double pp_ms = ms_since(t_pp0);
                if (sample_on && !stop_req.load(std::memory_order_acquire)) {
                    next_tok = sample_logits(lg, hist, smp, rng);
                }

                const auto t_tg0 = Clock::now();
                int emitted = 0;
                const char* finish = "length";
                const int avail = std::max(0, max_ctx - (int) prompt.size());
                const int limit = req_new > 0 ? std::min(req_new, avail) : avail;
                if (stop_req.load(std::memory_order_acquire)) {
                    finish = "cancel";
                } else {
                    for (int step = 0; step < limit; ++step) {
                        if (stop_req.load(std::memory_order_acquire)) {
                            finish = "cancel";
                            break;
                        }
                        std::printf("T %d\n", (int) next_tok);
                        std::fflush(stdout);
                        ++emitted;
                        hist.push_back(next_tok);
                        if (next_tok == eos_id || next_tok == 199999) {
                            finish = "stop";
                            break;
                        }
                        if (step + 1 < limit) {
                            const int32_t fed = next_tok;
                            next_tok = E.forward(&fed, 1, sample_on, &lg);
                            live_tokens.push_back(fed);
                            if (sample_on) next_tok = sample_logits(lg, hist, smp, rng);
                            if (E.hybrid && (step % 4 == 3)) E.rebalance(4);
                        }
                    }
                }
                const double tg_ms = ms_since(t_tg0);
                const long long lookups = (long long) E.hy_gpu + (long long) E.hy_cpu;
                std::printf("DONE %d %zu %.2f %.2f %s 0 0 %zu %lld %lld 0 0 0.0 %zu\n",
                            emitted, prompt.size(), pp_ms, tg_ms, finish,
                            reuse, (long long) E.hy_gpu, lookups, fresh);
                std::fflush(stdout);
            }
        }
        if (reader.joinable()) reader.join();
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
            w = E.forward(prompt.data() + off, tc);
        }
        if (max_new > 0) E.forward(&w, 1);
        E.reset_state();
        E.hy_gpu = E.hy_cpu = 0;
        E.hy_cpu_ms = 0.0;
    }

    std::vector<float> dumped;
    const auto t_pp0 = Clock::now();
    int32_t next_tok = 0;
    for (size_t off = 0; off < prompt.size(); off += (size_t) chunk) {
        const int tc = (int) std::min<size_t>((size_t) chunk, prompt.size() - off);
        const bool is_last = (off + (size_t) tc == prompt.size());
        next_tok = E.forward(prompt.data() + off, tc, is_last && !dump_logits_path.empty(), &dumped);
        if (E.hybrid) E.rebalance(8);
    }
    const double pp_ms = ms_since(t_pp0);

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
    if (max_new > 0) gen.push_back(next_tok);

    const auto t_tg0 = Clock::now();
    for (int step = 1; step < max_new; ++step) {
        next_tok = E.forward(&next_tok, 1);
        gen.push_back(next_tok);
        if (E.hybrid && (step % 4 == 0)) E.rebalance(4);
    }
    const double tg_ms = ms_since(t_tg0);

    std::printf("tokens:");
    for (int32_t id : gen) std::printf(" %d", (int) id);
    std::printf("\n");

    const double pp_tps = (double) prompt.size() / (pp_ms / 1000.0);
    const double tg_tps = max_new > 1 ? (double) (max_new - 1) / (tg_ms / 1000.0) : 0.0;
    std::fprintf(stderr, "strata-inkling2: pp %zu tok in %.1f ms (%.2f t/s) | tg %d tok in %.1f ms (%.2f t/s) | hits=%lld misses=%lld cpu_ms=%.1f\n",
                 prompt.size(), pp_ms, pp_tps, std::max(0, max_new - 1), tg_ms, tg_tps,
                 (long long) E.hy_gpu, (long long) E.hy_cpu, E.hy_cpu_ms);
    return 0;
}

}  // namespace strata::inkling2

