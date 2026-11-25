// src/prefill/gemm.cu - see include/strata/prefill/gemm.hpp.
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/core/arch_defaults.hpp"   // prompt_f16_resolve (inline)
#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)
#include "strata/prefill/mmq_ids_margin.hpp"
#include "strata/prefill/moe_mmq.hpp"
#endif

#ifdef STRATA_USE_HIP
#include "wmma_gemm.h"
#endif

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#if !defined(__HIPCC__) && !defined(STRATA_HIP_GFX906)
#include <cublasLt.h>
#include <mutex>
#include <unordered_map>
#endif

#include <algorithm>
#include <atomic>

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
// The HIP compatibility shim maps CUDA shuffle spellings to Strata helpers.
// hipBLASLt's public headers declare native HIP shuffle functions, so keep
// those declarations from being macro-expanded in this translation unit.
#undef __shfl_xor_sync
#undef __shfl_down_sync
#undef __shfl_up_sync
#undef __shfl_sync
#undef __ballot_sync
#endif

#include <algorithm>
#include <cctype>
#include <initializer_list>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
#include "hipblaslt_tuning.hpp"
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <map>
#include <set>
#include <tuple>
#endif
#if defined(__HIPCC__) && defined(STRATA_ROCBLAS_AVAILABLE)
#include "rocblas_tuning.hpp"
#include "strata/core/amd_community.hpp"
#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>
#include <filesystem>
#include <map>
#include <string>
#include <system_error>
#include <tuple>
#endif

#if defined(__HIPCC__)
#include <hip/hip_fp16.h>
#endif

namespace strata::prefill {
namespace {
#if defined(__HIPCC__)
// STRATA_F16_RANGE (gemm_set_f16_range): the largest |value| and the count beyond 65504 / non-finite, for the BF16
// weights converted to FP16 (slot 0, bf16_to_f16_rows) and the FP16 GEMM outputs widened (slot 1, widen_rows_f16 -
// an output rocBLAS wrote as FP16 is Inf there, not saturated).  As the activation counter in kernels.cu.
__device__ int g_f16_range_on_g = 0;
__device__ unsigned g_f16_range_max_g[2] = {0, 0};
__device__ unsigned long long g_f16_range_over_g[2] = {0, 0};
__device__ __forceinline__ void f16_range_note_g(int k, float f) {
    const float a = fabsf(f);
    const unsigned bits = __float_as_uint(a);
    if (bits > g_f16_range_max_g[k]) atomicMax(&g_f16_range_max_g[k], bits);
    if (a > 65504.0f || isnan(f)) atomicAdd(&g_f16_range_over_g[k], 1ull);
}
// Y's rows, written by an FP16-out GEMM as FP16 at the start of each FP32 row (ldc = 2 ldy halves), widened in place.
// Float c overwrites halves 2c and 2c+1, so a row is walked from its end in blocks: a block's halves are read into
// registers, the block syncs, then writes its floats - which only cover halves of blocks already read.
// STRATA_DBG_NAN: the FP16 outputs that are not finite (an FP32-accumulated sum past 65504 becomes inf in FP16).
__device__ unsigned long long g_f16_nonfinite = 0;
__global__ void widen_rows_f16(float* __restrict__ Y, int64_t n, int64_t ldy, int count_nonfinite) {
    float* y = Y + (int64_t) blockIdx.x * ldy;
    const __half* h = reinterpret_cast<const __half*>(y);
    const int64_t nb = (n + blockDim.x - 1) / blockDim.x;
    for (int64_t b = nb - 1; b >= 0; --b) {
        const int64_t c = b * blockDim.x + threadIdx.x;
        const float v = c < n ? __half2float(h[c]) : 0.0f;
        if (count_nonfinite && c < n && !isfinite(v)) atomicAdd(&g_f16_nonfinite, 1ull);
        if (c < n && g_f16_range_on_g) f16_range_note_g(1, v);
        __syncthreads();
        if (c < n) y[c] = v;
        __syncthreads();
    }
}
// BF16 weight rows -> FP16, saturated (bench/results/2026-10-04-rdna2-fp16-prompt: none leaves FP16's range)
__global__ void bf16_to_f16_rows(const uint16_t* __restrict__ s, __half* __restrict__ d, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const float f = __uint_as_float((uint32_t) s[i] << 16);
        if (g_f16_range_on_g) f16_range_note_g(0, f);
        d[i] = __float2half(isnan(f) ? f : fminf(fmaxf(f, -65504.0f), 65504.0f));   // as hf_sat: a NaN stays NaN
    }
}
#endif
}  // namespace

void gemm_set_f16_range(bool on) {
#if defined(__HIPCC__)
    const int v = on ? 1 : 0;
    if (cudaMemcpyToSymbol(g_f16_range_on_g, &v, sizeof v) != cudaSuccess) {
        std::fprintf(stderr, "prefill gemm: setting the FP16 range check failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        std::exit(1);
    }
#else
    (void) on;
#endif
}
bool gemm_f16_range_read(float max_abs[2], unsigned long long over[2], bool reset) {
#if defined(__HIPCC__)
    unsigned bits[2] = {0, 0}; unsigned long long o[2] = {0, 0};
    if (cudaMemcpyFromSymbol(bits, g_f16_range_max_g, sizeof bits) != cudaSuccess ||
        cudaMemcpyFromSymbol(o, g_f16_range_over_g, sizeof o) != cudaSuccess) { cudaGetLastError(); return false; }
    for (int k = 0; k < 2; ++k) { union { unsigned u; float f; } c; c.u = bits[k]; max_abs[k] = c.f; over[k] = o[k]; }
    if (reset) {
        const unsigned z[2] = {0, 0}; const unsigned long long zz[2] = {0, 0};
        cudaMemcpyToSymbol(g_f16_range_max_g, z, sizeof z); cudaMemcpyToSymbol(g_f16_range_over_g, zz, sizeof zz);
    }
    return true;
#else
    (void) max_abs; (void) over; (void) reset;
    return false;
#endif
}

namespace {

void ck(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: %s: cuBLAS status %d\n", what, (int) s);
        std::exit(1);
    }
}

// #247/#325: on Windows (seen on gfx1201), hipBLAS can return success with the correct BF16/FP16 product for some
// shapes (hc up once T >= 96, the router) and still leave hipErrorInvalidValue set, which the next kernel's error
// check turns into an exit. The multiply has finished, so that one stale error is cleared after a GEMM that succeeded;
// any other error still stops the engine. Windows only: on Linux a stale hipErrorInvalidValue is a real error from
// an earlier call and keeps being reported. A no-op everywhere else (CUDA compiles none of it).
#if defined(__HIPCC__) && defined(_WIN32)
void absorb_hipblas_sticky(const char* what) {
    const hipError_t sticky = hipGetLastError();
    if (sticky == hipSuccess || sticky == hipErrorInvalidValue) return;
    std::fprintf(stderr, "prefill gemm: %s left %s\n", what, hipGetErrorString(sticky));
    std::exit(1);
}
#define STRATA_ABSORB_HIPBLAS_STICKY(what) absorb_hipblas_sticky(what)
#else
#define STRATA_ABSORB_HIPBLAS_STICKY(what) ((void) 0)
#endif

// A setup call whose failure the engine survives (the handle keeps its defaults), as before #240 - but said.
void note(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) std::fprintf(stderr, "prefill gemm: %s: cuBLAS status %d (continuing)\n", what, (int) s);
}

#if !defined(__HIPCC__) && !defined(STRATA_HIP_GFX906)
// CUDA sm_80+: shape-keyed cublasLtMatmul heuristic autotuner cache for prefill Gemm::f16, Gemm::f16_f16out, and Gemm::sgemm.
// Strictly CUBLAS_COMPUTE_32F (FP16->FP32 accum->FP32/FP16 out for f16/f16_f16out; strict IEEE FP32->FP32 out, NO TF32 for sgemm).
struct CudaLtKey {
    int dev;
    int M;          // col-major rows (N_out)
    int N;          // col-major cols (T)
    int K;          // inner dimension (K_in)
    int ldc;        // col-major leading dim of C/D (ldy)
    int dtype_ab;   // 0: CUDA_R_16F, 1: CUDA_R_32F
    int out_f16;    // 0: CUDA_R_32F output, 1: CUDA_R_16F output (STRICT CUBLAS_COMPUTE_32F in all cases)
    size_t max_ws;  // workspace budget in bytes

    bool operator==(const CudaLtKey& o) const {
        return dev == o.dev && M == o.M && N == o.N && K == o.K &&
               ldc == o.ldc && dtype_ab == o.dtype_ab && out_f16 == o.out_f16 && max_ws == o.max_ws;
    }
};

struct CudaLtKeyHash {
    size_t operator()(const CudaLtKey& k) const noexcept {
        uint64_t h = 14695981039346656037ull;
        const auto mix = [&](uint64_t v) {
            h ^= v;
            h *= 1099511628211ull;
        };
        mix((uint64_t) (uint32_t) k.dev);
        mix((uint64_t) (uint32_t) k.M);
        mix((uint64_t) (uint32_t) k.N);
        mix((uint64_t) (uint32_t) k.K);
        mix((uint64_t) (uint32_t) k.ldc);
        mix((uint64_t) (uint32_t) k.dtype_ab);
        mix((uint64_t) (uint32_t) k.out_f16);
        mix((uint64_t) k.max_ws);
        return (size_t) h;
    }
};

struct CudaLtGlobalEntry {
    cublasLtMatmulAlgo_t algo{};
    size_t workspace_bytes = 0;
    bool valid = false;
};

struct CudaLtEntry {
    cublasLtMatmulAlgo_t algo{};
    cublasLtMatmulDescOpaque_t op_desc{};
    cublasLtMatrixLayoutOpaque_t a_desc{};
    cublasLtMatrixLayoutOpaque_t b_desc{};
    cublasLtMatrixLayoutOpaque_t c_desc{};
    size_t workspace_bytes = 0;
    bool valid = false;
    bool descs_ready = false;
};

struct CudaLtCache {
    std::unordered_map<CudaLtKey, CudaLtEntry, CudaLtKeyHash> map;
};

// Process-wide per-device backing cache so separate Gemm instances on the same GPU select the exact same
// tuned algorithm (bitwise deterministic across instances, e.g. gemm_w16_parity) while each Gemm keeps a
// lock-free hot-path map in lt_cache_ with pre-initialized opaque descriptors.
std::mutex g_cuda_lt_global_mu;
std::unordered_map<CudaLtKey, CudaLtGlobalEntry, CudaLtKeyHash> g_cuda_lt_global_cache;
std::unordered_map<CudaLtKey, CudaLtGlobalEntry, CudaLtKeyHash> g_cuda_lt_bucket_cache;

inline int cublaslt_t_bucket(int T) {
    return (T >= 4096) ? 4 : (T >= 2048) ? 3 : (T >= 1024) ? 2 : (T >= 512) ? 1 : 0;
}

bool init_cuda_lt_descs(CudaLtEntry& e, const CudaLtKey& key) {
    if (e.descs_ready) return true;
    const cudaDataType_t dtype_ab = (key.dtype_ab != 0) ? CUDA_R_32F : CUDA_R_16F;
    const cudaDataType_t dtype_cd = (key.out_f16 != 0) ? CUDA_R_16F : CUDA_R_32F;
    // HARD PRECISION INVARIANT: strictly CUBLAS_COMPUTE_32F!
    // - is_f32=false, out_f16=false: FP16 inputs -> FP32 accumulate -> FP32 output (NEVER CUBLAS_COMPUTE_16F or FAST_16F).
    // - is_f32=false, out_f16=true : FP16 inputs -> FP32 accumulate -> FP16 output (STRICT CUBLAS_COMPUTE_32F, NEVER CUBLAS_COMPUTE_16F).
    // - is_f32=true , out_f16=false: strict IEEE FP32 inputs -> FP32 accumulate -> FP32 output (NEVER FAST_TF32).
    const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    const cudaDataType_t scale_type = CUDA_R_32F;
    const cublasOperation_t trans_a = CUBLAS_OP_T;
    const cublasOperation_t trans_b = CUBLAS_OP_N;
    // Column-major view: Y^T[M, N] (ldc=ldy) = W[K, M]^T (lda = K) * X[K, N] (ldb = K).
    if (cublasLtMatmulDescInit(&e.op_desc, compute_type, scale_type) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(&e.op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(&e.op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutInit(&e.a_desc, dtype_ab, (uint64_t) key.K, (uint64_t) key.M, (int64_t) key.K) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutInit(&e.b_desc, dtype_ab, (uint64_t) key.K, (uint64_t) key.N, (int64_t) key.K) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutInit(&e.c_desc, dtype_cd, (uint64_t) key.M, (uint64_t) key.N, (int64_t) key.ldc) != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    e.descs_ready = true;
    return true;
}

bool cublaslt_tune_disabled() {
    static const bool off = [] {
        const char* v = std::getenv("STRATA_NO_CUBLASLT_TUNE");
        return v != nullptr && v[0] != '\0' && std::atoi(v) != 0;
    }();
    return off;
}

bool cublaslt_tune_verbose() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_CUBLASLT_VERBOSE");
        return v != nullptr && v[0] != '\0' && std::atoi(v) != 0;
    }();
    return on;
}

bool fused_q8_gemm_disabled() {
    static const bool off = [] {
        const char* no = std::getenv("STRATA_NO_FUSED_Q8_GEMM");
        if (no != nullptr && no[0] != '\0' && std::atoi(no) != 0) return true;
        const char* v = std::getenv("STRATA_FUSED_Q8_GEMM");
        return !(v != nullptr && v[0] != '\0' && std::atoi(v) != 0);
    }();
    return off;
}

bool fused_q8_f16out_enabled() {
    static const bool on = [] {
        const char* no = std::getenv("STRATA_NO_FUSED_Q8_GEMM");
        if (no != nullptr && no[0] != '\0' && std::atoi(no) != 0) return false;
        const char* v = std::getenv("STRATA_FUSED_Q8_F16OUT");
        if (v != nullptr && v[0] != '\0') return std::atoi(v) != 0;
        return false;
    }();
    return on;
}

int current_sm_count() {
    static std::atomic<int> sms[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return 0; }
    int v = sms[dev].load(std::memory_order_relaxed);
    if (v == 0) {
        int count = 0;
        if (cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || count <= 0) {
            cudaGetLastError();
            return 0;
        }
        v = count;
        sms[dev].store(v, std::memory_order_relaxed);
    }
    return v;
}

// ---- Persistent L2-swizzled fused Q8_0 x FP16 -> FP32-accum Tensor Core GEMM (CUDA sm_80+) --------------------------
// Tile: BT=128 (tokens, MMA M), BM=128 (output features, MMA N), BK=32 (1 Q8_0 block = 34 B/row).
// 256 threads = 8 warps in 2x4 layout (64 tokens x 32 weight rows per warp -> 64 FP32 acc regs, STACK:0, 2 CTAs/SM).
// SMEM: 3-stage sX[3][128][32] (24 KB) + 2-stage sW[2][128][32] (16 KB) = 40.0 KB (0-bank-conflict XOR swizzle).
// Q8_0 dequant matches dequant_q8_0_kernel<__half> bit-for-bit via exact 0x6400/1152.0f magic-bias HMUL2.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ int q8g_swz32(int row, int col8) {
    return (row << 5) + ((col8 ^ ((row >> 1) & 3)) << 3);
}
__device__ __forceinline__ void q8g_cp_async_cg_16(uint32_t smem_addr, const void* glob_ptr, bool pred) {
    const int src_bytes = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(smem_addr), "l"(glob_ptr), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void q8g_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
__device__ __forceinline__ void q8g_cp_async_wait0() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}
__device__ __forceinline__ void q8g_cp_async_wait1() {
    asm volatile("cp.async.wait_group 1;\n" ::: "memory");
}
__device__ __forceinline__ void q8g_ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void q8g_st_shared_v4(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n"
                 :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
}
__device__ __forceinline__ void q8g_mma_m16n8k16_f32(float d[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                     uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

struct Q8HalfBlk {
    uint32_t q0, q1, q2, q3;
    uint16_t d_bits;
};

// Alignment-safe & strictly in-bounds loader for 16 int8_t weights + FP16 scale from a 2B-aligned 34-byte block_q8_0.
__device__ __forceinline__ Q8HalfBlk q8g_load_half_blk(const uint8_t* bp, int half_blk) {
    Q8HalfBlk r;
    const uint8_t* qp = bp + 2 + (half_blk << 4);
    if ((((uintptr_t) bp) & 2u) != 0u) {
        // bp is 2 mod 4 -> qp (bp + 2 or bp + 18) is 4-byte aligned, strictly within [bp + 2, bp + 33].
        r.d_bits = __ldg(reinterpret_cast<const uint16_t*>(bp));
        const uint32_t* p32 = reinterpret_cast<const uint32_t*>(qp);
        r.q0 = __ldg(p32 + 0);
        r.q1 = __ldg(p32 + 1);
        r.q2 = __ldg(p32 + 2);
        r.q3 = __ldg(p32 + 3);
    } else {
        // bp is 0 mod 4 -> qp is 2 mod 4; qp+2, qp+6, qp+10 are 4-byte aligned, strictly within [bp, bp + 33].
        uint16_t h0;
        if (half_blk == 0) {
            const uint32_t v0 = __ldg(reinterpret_cast<const uint32_t*>(bp));
            r.d_bits = static_cast<uint16_t>(v0 & 0xFFFFu);
            h0 = static_cast<uint16_t>(v0 >> 16);
        } else {
            r.d_bits = __ldg(reinterpret_cast<const uint16_t*>(bp));
            h0 = __ldg(reinterpret_cast<const uint16_t*>(qp));
        }
        const uint32_t* p32 = reinterpret_cast<const uint32_t*>(qp + 2);
        const uint32_t m0 = __ldg(p32 + 0);
        const uint32_t m1 = __ldg(p32 + 1);
        const uint32_t m2 = __ldg(p32 + 2);
        const uint16_t h1 = __ldg(reinterpret_cast<const uint16_t*>(qp + 14));
        r.q0 = static_cast<uint32_t>(h0) | (m0 << 16);
        r.q1 = __funnelshift_r(m0, m1, 16);
        r.q2 = __funnelshift_r(m1, m2, 16);
        r.q3 = (m2 >> 16) | (static_cast<uint32_t>(h1) << 16);
    }
    return r;
}

__device__ __forceinline__ void q8g_i8x4_to_h2x2_scaled(uint32_t packed_i8x4, __half2 scale2,
                                                        uint32_t& out0, uint32_t& out1) {
    const uint32_t u = packed_i8x4 ^ 0x80808080u;
    const uint32_t lo = __byte_perm(u, 0x6400u, 0x5150u);
    const uint32_t hi = __byte_perm(u, 0x6400u, 0x5352u);
    const uint32_t bias_u = 0x64806480u;   // __half2(1152.0f, 1152.0f): (1024 + (q ^ 0x80)) - 1152 == q exact
    const __half2 bias = *reinterpret_cast<const __half2*>(&bias_u);
    const __half2 h0 = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&lo), bias), scale2);
    const __half2 h1 = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&hi), bias), scale2);
    out0 = *reinterpret_cast<const uint32_t*>(&h0);
    out1 = *reinterpret_cast<const uint32_t*>(&h1);
}

__device__ __forceinline__ void q8g_dequant_store_sw(const Q8HalfBlk& blk, uint32_t dst_lo, uint32_t dst_hi) {
    const __half2 d2 = __half2half2(__ushort_as_half(blk.d_bits));
    uint32_t v0, v1, v2, v3;
    q8g_i8x4_to_h2x2_scaled(blk.q0, d2, v0, v1);
    q8g_i8x4_to_h2x2_scaled(blk.q1, d2, v2, v3);
    q8g_st_shared_v4(dst_lo, v0, v1, v2, v3);
    q8g_i8x4_to_h2x2_scaled(blk.q2, d2, v0, v1);
    q8g_i8x4_to_h2x2_scaled(blk.q3, d2, v2, v3);
    q8g_st_shared_v4(dst_hi, v0, v1, v2, v3);
}
#endif

template <int N_SPLITS, bool OUT_F16 = false>
__global__ void __launch_bounds__(256, 2)
k_gemm_q8_0_f16_f32_sm80(const uint8_t* __restrict__ W_q8, const __half* __restrict__ X_f16,
                         void* __restrict__ Y_single, int ldy_single, MultiOut5 outs,
                         int T, int M_total, int K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(128) __half smem[5 * 128 * 32];   // 40,960 B: sX[3][128][32] + sW[2][128][32]

    const int tid = (int) threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wt = warp >> 2;   // 0..1 (64 tokens / warp = 4 x m16)
    const int wm = warp & 3;    // 0..3 (32 weight rows / warp = 4 x n8)

    // X loader coordinates: 64 rows x 4 (16B) chunks, 2 rows per thread (xr and xr + 64)
    const int xr = tid >> 2;
    const int c16 = tid & 3;
    const int xc = c16 << 3;

    // W loader coordinates: 128 rows x 2 half-blocks (16 int8s each)
    const int wr = tid >> 1;
    const int half_blk = tid & 1;

    // Precomputed SMEM byte addresses (128B aligned base -> pure XOR/add stepping)
    const uint32_t sx_wr_base = static_cast<uint32_t>(__cvta_generic_to_shared(smem + q8g_swz32(xr, c16)));
    const uint32_t sw_wr_base = static_cast<uint32_t>(__cvta_generic_to_shared(smem + 3 * 4096 + q8g_swz32(wr, half_blk << 1)));

    const int a_lane_row = lane & 15;
    const int a_lane_c8 = lane >> 4;
    const int b_lane_row = ((lane >> 4) << 3) | (lane & 7);
    const int b_lane_c8 = (lane >> 3) & 1;
    const uint32_t sx_rd_base = static_cast<uint32_t>(__cvta_generic_to_shared(
        smem + q8g_swz32(wt * 64 + a_lane_row, a_lane_c8)));
    const uint32_t sw_rd_base = static_cast<uint32_t>(__cvta_generic_to_shared(
        smem + 3 * 4096 + q8g_swz32(wm * 32 + b_lane_row, b_lane_c8)));

    const int nb_k = K >> 5;
    const size_t w_row_bytes = (size_t) nb_k * 34u;
    const int n_tiles_t = (T + 127) >> 7;
    const int n_tiles_m = M_total >> 7;
    const int total_tiles = n_tiles_t * n_tiles_m;

    constexpr int kGroupT = 8;
    const int tiles_per_group = kGroupT * n_tiles_m;
    const int acc_r0_off = lane >> 2;
    const int acc_c0_off = (lane & 3) << 1;

    for (int tile_idx = (int) blockIdx.x; tile_idx < total_tiles; tile_idx += (int) gridDim.x) {
        if (tile_idx != (int) blockIdx.x) {
            __syncthreads();   // ensure previous tile's HMMAs finished reading SMEM before reusing sX/sW
        }

        // L2-swizzled tile coordinates: GROUP_T=8 T-tiles x all M-tiles (caches 1024-token X slice in L2)
        const int group_id = tile_idx / tiles_per_group;
        const int first_t = group_id * kGroupT;
        const int group_size_t = min(kGroupT, n_tiles_t - first_t);
        const int in_group = tile_idx - group_id * tiles_per_group;
        const int tile_m = in_group / group_size_t;
        const int tile_t = first_t + (in_group - tile_m * group_size_t);

        const int t0 = tile_t << 7;
        const int m0 = tile_m << 7;

        const bool x0_ok = (t0 + xr) < T;
        const bool x1_ok = (t0 + xr + 64) < T;
        const __half* const xp0 = X_f16 + (size_t) (x0_ok ? (t0 + xr) : 0) * (size_t) K + xc;
        const __half* const xp1 = X_f16 + (size_t) (x1_ok ? (t0 + xr + 64) : 0) * (size_t) K + xc;
        const uint8_t* w_cur = W_q8 + (size_t) (m0 + wr) * w_row_bytes;

        float acc[4][4][4];
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
#pragma unroll
            for (int nm = 0; nm < 4; ++nm) {
                acc[mt][nm][0] = 0.0f;
                acc[mt][nm][1] = 0.0f;
                acc[mt][nm][2] = 0.0f;
                acc[mt][nm][3] = 0.0f;
            }
        }

        // Prologue: issue cp.async for sX[0] and (if nb_k > 1) sX[1], overlap with W[0] load & dequant into sW[0]
        q8g_cp_async_cg_16(sx_wr_base,           xp0, x0_ok);
        q8g_cp_async_cg_16(sx_wr_base + 0x1000u, xp1, x1_ok);
        q8g_cp_async_commit();

        if (nb_k > 1) {
            q8g_cp_async_cg_16(sx_wr_base + 0x2000u, xp0 + 32, x0_ok);
            q8g_cp_async_cg_16(sx_wr_base + 0x3000u, xp1 + 32, x1_ok);
            q8g_cp_async_commit();
        }

        {
            const Q8HalfBlk w0 = q8g_load_half_blk(w_cur, half_blk);
            q8g_dequant_store_sw(w0, sw_wr_base, sw_wr_base ^ 0x10u);
        }

        if (nb_k > 1) {
            q8g_cp_async_wait1();
        } else {
            q8g_cp_async_wait0();
        }
        __syncthreads();

        int sx_rd = 0;
        int sx_wr = 2;
        int sw_rd = 0;

        auto do_mma_k16 = [&](uint32_t a_base, uint32_t b_base) {
            uint32_t b00, b01, b02, b03, b10, b11, b12, b13;
            q8g_ldmatrix_x4(b00, b01, b02, b03, b_base);
            q8g_ldmatrix_x4(b10, b11, b12, b13, b_base + 0x400u);
#pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                uint32_t a0, a1, a2, a3;
                q8g_ldmatrix_x4(a0, a1, a2, a3, a_base + (uint32_t) (mt << 10));
                q8g_mma_m16n8k16_f32(acc[mt][0], a0, a1, a2, a3, b00, b01);
                q8g_mma_m16n8k16_f32(acc[mt][1], a0, a1, a2, a3, b02, b03);
                q8g_mma_m16n8k16_f32(acc[mt][2], a0, a1, a2, a3, b10, b11);
                q8g_mma_m16n8k16_f32(acc[mt][3], a0, a1, a2, a3, b12, b13);
            }
        };

        for (int bk = 0; bk < nb_k; ++bk) {
            const bool has_next = (bk + 1 < nb_k);
            const bool has_next2 = (bk + 2 < nb_k);

            if (has_next2) {
                const uint32_t dst_x = sx_wr_base + (uint32_t) (sx_wr << 13);
                const int k_off = (bk + 2) << 5;
                q8g_cp_async_cg_16(dst_x,           xp0 + k_off, x0_ok);
                q8g_cp_async_cg_16(dst_x + 0x1000u, xp1 + k_off, x1_ok);
                q8g_cp_async_commit();
            }

            Q8HalfBlk next_w;
            if (has_next) {
                w_cur += 34;
                next_w = q8g_load_half_blk(w_cur, half_blk);
            }

            const uint32_t a_base0 = sx_rd_base + (uint32_t) (sx_rd << 13);
            const uint32_t b_base0 = sw_rd_base + (uint32_t) (sw_rd << 13);
            do_mma_k16(a_base0, b_base0);

            if (has_next) {
                const uint32_t dst_w_lo = sw_wr_base + (uint32_t) ((sw_rd ^ 1) << 13);
                q8g_dequant_store_sw(next_w, dst_w_lo, dst_w_lo ^ 0x10u);
            }

            do_mma_k16(a_base0 ^ 0x20u, b_base0 ^ 0x20u);

            if (has_next) {
                if (has_next2) {
                    q8g_cp_async_wait1();
                } else {
                    q8g_cp_async_wait0();
                }
                __syncthreads();
                sx_rd = (sx_rd == 2) ? 0 : (sx_rd + 1);
                sx_wr = (sx_wr == 2) ? 0 : (sx_wr + 1);
                sw_rd ^= 1;
            }
        }

        // Epilogue: resolve output pointer & stride once per BM=128 tile (zero per-element branching)
        void* out_base_raw = Y_single;
        int out_ld = ldy_single;
        int m_local = m0;
        if constexpr (N_SPLITS > 1) {
            out_base_raw = nullptr;
            out_ld = 0;
            m_local = 0;
            int prev_end = 0;
#pragma unroll
            for (int s = 0; s < 5; ++s) {
                if (s < outs.n_outs) {
                    const int cur_end = outs.ends[s];
                    if (m0 >= prev_end && m0 < cur_end) {
                        out_base_raw = outs.ptrs[s];
                        out_ld = cur_end - prev_end;
                        m_local = m0 - prev_end;
                    }
                    prev_end = cur_end;
                }
            }
        }

        const int col_base = m_local + wm * 32 + acc_c0_off;
        const int row_base = t0 + wt * 64 + acc_r0_off;
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
            const int r0 = row_base + mt * 16;
            const int r1 = r0 + 8;
            if constexpr (OUT_F16) {
                __half* const out_f16 = static_cast<__half*>(out_base_raw);
                if (r0 < T) {
                    __half2* dst0 = reinterpret_cast<__half2*>(out_f16 + (size_t) r0 * (size_t) out_ld + col_base);
#pragma unroll
                    for (int nm = 0; nm < 4; ++nm) {
                        dst0[nm * 4] = __floats2half2_rn(acc[mt][nm][0], acc[mt][nm][1]);
                    }
                }
                if (r1 < T) {
                    __half2* dst1 = reinterpret_cast<__half2*>(out_f16 + (size_t) r1 * (size_t) out_ld + col_base);
#pragma unroll
                    for (int nm = 0; nm < 4; ++nm) {
                        dst1[nm * 4] = __floats2half2_rn(acc[mt][nm][2], acc[mt][nm][3]);
                    }
                }
            } else {
                float* const out_f32 = static_cast<float*>(out_base_raw);
                if (r0 < T) {
                    float* dst0 = out_f32 + (size_t) r0 * (size_t) out_ld + col_base;
#pragma unroll
                    for (int nm = 0; nm < 4; ++nm) {
                        *reinterpret_cast<float2*>(dst0 + nm * 8) = make_float2(acc[mt][nm][0], acc[mt][nm][1]);
                    }
                }
                if (r1 < T) {
                    float* dst1 = out_f32 + (size_t) r1 * (size_t) out_ld + col_base;
#pragma unroll
                    for (int nm = 0; nm < 4; ++nm) {
                        *reinterpret_cast<float2*>(dst1 + nm * 8) = make_float2(acc[mt][nm][2], acc[mt][nm][3]);
                    }
                }
            }
        }
    }
#else
    (void) W_q8; (void) X_f16; (void) Y_single; (void) ldy_single; (void) outs;
    (void) T; (void) M_total; (void) K;
#endif
}

template <int N_SPLITS, bool OUT_F16 = false>
void ensure_q8_gemm_carveout() {
    static std::atomic<uint64_t> configured_mask{0};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return; }
    const uint64_t bit = 1ull << dev;
    if ((configured_mask.load(std::memory_order_relaxed) & bit) == 0) {
        if (cudaFuncSetAttribute(
                k_gemm_q8_0_f16_f32_sm80<N_SPLITS, OUT_F16>,
                cudaFuncAttributePreferredSharedMemoryCarveout,
                cudaSharedmemCarveoutMaxShared) != cudaSuccess) {
            (void) cudaGetLastError();
        }
        configured_mask.fetch_or(bit, std::memory_order_relaxed);
    }
}

template <int N_SPLITS, bool OUT_F16 = false>
void launch_gemm_q8_0_f16_f32_impl(const void* w_q8_0, const __half* x_f16, void* Y,
                                   int ldy, const MultiOut5& outs, int T, int M_total, int K, cudaStream_t st) {
    ensure_q8_gemm_carveout<N_SPLITS, OUT_F16>();
    const int n_tiles_t = (T + 127) >> 7;
    const int n_tiles_m = M_total >> 7;
    const int total_tiles = n_tiles_t * n_tiles_m;
    const int n_sms = current_sm_count();
    const int max_ctas = (n_sms > 0) ? (n_sms * 2) : total_tiles;
    const int grid = (total_tiles < max_ctas) ? total_tiles : max_ctas;
    k_gemm_q8_0_f16_f32_sm80<N_SPLITS, OUT_F16><<<(unsigned) grid, 256, 0, st>>>(
        static_cast<const uint8_t*>(w_q8_0), x_f16, Y, ldy, outs, T, M_total, K);
}

void launch_gemm_q8_0_f16_f32_single(const void* w_q8_0, const __half* x_f16, float* Y,
                                     int T, int M_total, int K, int ldy, cudaStream_t st) {
    const MultiOut5 dummy{};
    launch_gemm_q8_0_f16_f32_impl<1, false>(w_q8_0, x_f16, Y, ldy, dummy, T, M_total, K, st);
}

void launch_gemm_q8_0_f16_f32_single_f16out(const void* w_q8_0, const __half* x_f16, __half* Y,
                                            int T, int M_total, int K, int ldy, cudaStream_t st) {
    const MultiOut5 dummy{};
    launch_gemm_q8_0_f16_f32_impl<1, true>(w_q8_0, x_f16, Y, ldy, dummy, T, M_total, K, st);
}

void launch_gemm_q8_0_f16_f32_multi(const void* w_q8_0, const __half* x_f16, const MultiOut5& outs,
                                    int T, int M_total, int K, cudaStream_t st) {
    launch_gemm_q8_0_f16_f32_impl<5, false>(w_q8_0, x_f16, nullptr, 0, outs, T, M_total, K, st);
}
#endif

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
struct HipLtCallKey {
    strata::prefill::hipblaslt::InputType type;
    int t;
    int n;
    int k;
    int ldy;
    uint32_t beta_bits;

    bool operator<(const HipLtCallKey& other) const {
        return std::tie(type, n, k, ldy, t, beta_bits) <
               std::tie(other.type, other.n, other.k, other.ldy, other.t, other.beta_bits);
    }
};

struct HipLtCachedAlgo {
    bool supported = false;
    hipblasLtMatmulAlgo_t algo{};
    size_t workspace_bytes = 0;
};

struct HipLtState {
    hipblasLtHandle_t handle = nullptr;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    strata::prefill::hipblaslt::TuningTable table;
    std::map<HipLtCallKey, HipLtCachedAlgo> cache;
    uint64_t lt_launches = 0;
    uint64_t fallbacks = 0;
    std::set<std::tuple<strata::prefill::hipblaslt::InputType, int, int, int, int>> fallback_shapes;

    ~HipLtState() {
        if (std::getenv("STRATA_HIPBLASLT_VERBOSE")) {
            std::fprintf(stderr, "prefill gemm: hipBLASLt summary launches=%llu fallbacks=%llu unique_fallback_shapes=%zu\n",
                         (unsigned long long) lt_launches, (unsigned long long) fallbacks, fallback_shapes.size());
            for (const auto& shape : fallback_shapes) {
                const auto type = std::get<0>(shape);
                std::fprintf(stderr, "prefill gemm: fallback shape dtype=%s T=%d N=%d K=%d ldy=%d\n",
                             type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16",
                             std::get<1>(shape), std::get<2>(shape), std::get<3>(shape), std::get<4>(shape));
            }
        }
        if (handle) hipblasLtDestroy(handle);
    }
};

struct HipLtDescriptors {
    hipblasLtMatmulDesc_t op = nullptr;
    hipblasLtMatrixLayout_t a = nullptr;
    hipblasLtMatrixLayout_t b = nullptr;
    hipblasLtMatrixLayout_t c = nullptr;

    ~HipLtDescriptors() {
        if (op) hipblasLtMatmulDescDestroy(op);
        if (a) hipblasLtMatrixLayoutDestroy(a);
        if (b) hipblasLtMatrixLayoutDestroy(b);
        if (c) hipblasLtMatrixLayoutDestroy(c);
    }

    bool init(hipDataType type, int t, int n, int k, int ldy, int ldb = 0) {
        const hipblasOperation_t trans_a = HIPBLAS_OP_T;
        const hipblasOperation_t trans_b = HIPBLAS_OP_N;
        if (hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)) !=
                HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)) !=
                HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&a, type, k, n, k) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&b, type, k, t, ldb > k ? ldb : k) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&c, HIP_R_32F, n, t, ldy) != HIPBLAS_STATUS_SUCCESS) {
            return false;
        }
        return true;
    }
};

// The engine package's own tuning folder (<engine folder>/tuning), for a start that did not come through the launcher (a hand-made
// config, a CLI run): without a table the prompt GEMMs fall back to plain hipBLAS and a Strix Halo reads prompts at half speed
// (340 against 690 tok/s, IQ3_S 4K).  The launcher sets STRATA_HIPBLASLT_TUNING itself; this is the same table found by the engine.
std::string own_tuning_dir() {
#if defined(_WIN32)
    char p[4096] = {};
    const DWORD n = GetModuleFileNameA(nullptr, p, (DWORD) sizeof(p));
    std::string s(p, n);
    const size_t k = s.find_last_of("\\/");
    return k == std::string::npos ? std::string() : s.substr(0, k + 1) + "tuning/";
#else
    char p[4096] = {};
    const ssize_t n = readlink("/proc/self/exe", p, sizeof(p) - 1);
    std::string s(p, n > 0 ? (size_t) n : 0);
    const size_t k = s.find_last_of('/');
    return k == std::string::npos ? std::string() : s.substr(0, k + 1) + "tuning/";
#endif
}

std::unique_ptr<HipLtState> create_hipblaslt_state(void* workspace, size_t workspace_bytes) {
    std::string path_own;
    const char* path = std::getenv("STRATA_HIPBLASLT_TUNING");
    bool from_package = false, table_forced = false;
    if (!path || !*path) {
        const char* sw = std::getenv("STRATA_HIPBLASLT_TABLE");     // the launcher's switch (strata_control/hipblaslt_tuning.py): off / on / auto
        std::string low = sw != nullptr ? sw : "";                   // trimmed and lower case, as the launcher reads it
        while (!low.empty() && std::isspace((unsigned char) low.back())) low.pop_back();
        while (!low.empty() && std::isspace((unsigned char) low.front())) low.erase(low.begin());
        for (char& ch : low) ch = (char) std::tolower((unsigned char) ch);
        const auto word = [&](std::initializer_list<const char*> ws) {
            for (const char* w : ws) if (low == w) return true;
            return false;
        };
        if (word({"off", "0", "no", "false", "none", "disable", "disabled"})) return nullptr;   // never the packaged table
        table_forced = word({"on", "1", "yes", "true", "force", "enable", "enabled"});          // any card that has a table beside the engine
        from_package = true;                                          // the file is found below, once the card and the version are known
    }

    auto state = std::make_unique<HipLtState>();
    state->workspace = workspace;
    state->workspace_bytes = workspace_bytes;
    if (hipblasLtCreate(&state->handle) != HIPBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: hipBLASLt handle creation failed; using hipBLASEx\n");
        return nullptr;
    }

    int version = 0;
    if (hipblasLtGetVersion(state->handle, &version) != HIPBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: hipBLASLt version query failed; using hipBLASEx\n");
        return nullptr;
    }
    int device = 0;
    hipDeviceProp_t properties{};
    if (hipGetDevice(&device) != hipSuccess || hipGetDeviceProperties(&properties, device) != hipSuccess) {
        std::fprintf(stderr, "prefill gemm: HIP device query failed; using hipBLASEx\n");
        return nullptr;
    }
    std::string arch(properties.gcnArchName);
    const auto suffix = arch.find(':');
    if (suffix != std::string::npos) arch.resize(suffix);

    if (from_package) {
        // Unset or "auto": only Strix Halo (gfx1151) takes the packaged table by itself, the owner's decision of 2026-10-19 (the same
        // rule as AUTO_ARCHS in strata_control/hipblaslt_tuning.py): on other cards the table changes the answers' last digits for ~1%
        // (gfx1201), so they take it with STRATA_HIPBLASLT_TABLE=on only.
        if (!table_forced && arch != "gfx1151") return nullptr;
        const std::string dir = own_tuning_dir();
        if (dir.empty()) return nullptr;
        path_own = dir + arch + "-hipblaslt-" + std::to_string(version) + ".txt";
        std::FILE* f = std::fopen(path_own.c_str(), "rb");
        if (f == nullptr) return nullptr;       // no table for this card and hipBLASLt version beside the engine: plain hipBLAS as before
        std::fclose(f);
        path = path_own.c_str();
    }
    std::string error;
    if (!state->table.load(path, arch, version, error)) {
        std::fprintf(stderr, "prefill gemm: %s; using hipBLASEx\n", error.c_str());
        return nullptr;
    }
    std::fprintf(stderr, "prefill gemm: hipBLASLt tuning enabled (%zu rows, %s, version %d)\n",
                 state->table.rows().size(), arch.c_str(), version);
    return state;
}

HipLtCachedAlgo resolve_hipblaslt_algo(HipLtState& state, strata::prefill::hipblaslt::InputType type, int t,
                                       int n, int k, int ldy, float beta) {
    uint32_t beta_bits = 0;
    static_assert(sizeof(beta_bits) == sizeof(beta));
    std::memcpy(&beta_bits, &beta, sizeof(beta));
    const HipLtCallKey key{type, t, n, k, ldy, beta_bits};
    const auto cached = state.cache.find(key);
    if (cached != state.cache.end()) return cached->second;

    HipLtCachedAlgo resolved;
    const bool verbose = std::getenv("STRATA_HIPBLASLT_VERBOSE") != nullptr;
    const auto* row = state.table.closest(type, n, k, ldy, t);
    if (!row) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; no calibration for dtype=%s T=%d N=%d K=%d ldy=%d\n",
                         type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16", t, n, k, ldy);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    HipLtDescriptors desc;
    const hipDataType input_type = type == strata::prefill::hipblaslt::InputType::bf16 ? HIP_R_16BF : HIP_R_16F;
    if (!desc.init(input_type, t, n, k, ldy)) {
        return state.cache.emplace(key, resolved).first->second;
    }

    std::vector<int> solution_ids{row->solution_id};
    std::vector<hipblasLtMatmulHeuristicResult_t> candidates;
    if (hipblaslt_ext::getAlgosFromIndex(state.handle, solution_ids, candidates) != HIPBLAS_STATUS_SUCCESS ||
        candidates.empty() || candidates.front().state != HIPBLAS_STATUS_SUCCESS ||
        hipblaslt_ext::getIndexFromAlgo(candidates.front().algo) != row->solution_id) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution %d unavailable for T=%d N=%d K=%d ldy=%d\n",
                         row->solution_id, t, n, k, ldy);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    const float alpha = 1.0f;
    size_t required_workspace = 0;
    auto algo = candidates.front().algo;
    if (hipblaslt_ext::matmulIsAlgoSupported(state.handle, desc.op, &alpha, desc.a, desc.b, &beta, desc.c, desc.c,
                                             algo, required_workspace) != HIPBLAS_STATUS_SUCCESS) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution %d rejects actual T=%d N=%d K=%d ldy=%d beta=%.9g\n",
                         row->solution_id, t, n, k, ldy, beta);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    resolved.supported = true;
    resolved.algo = algo;
    resolved.workspace_bytes = required_workspace;
    if (verbose) {
        std::fprintf(stderr,
                     "prefill gemm: Lt solution=%d dtype=%s T=%d N=%d K=%d ldy=%d beta=%.9g workspace=%zu\n",
                     row->solution_id, type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16", t, n,
                     k, ldy, beta, required_workspace);
    }
    return state.cache.emplace(key, resolved).first->second;
}

// the hipBLASLt solution index the tuning table gives this call (-1: none / unsupported)
int hipblaslt_solution_for(void* opaque_state, strata::prefill::hipblaslt::InputType type, int64_t t, int64_t n, int64_t k,
                           int64_t ldy, float beta) {
    auto* state = static_cast<HipLtState*>(opaque_state);
    if (!state || t <= 0 || n <= 0 || k <= 0 || t > INT_MAX || n > INT_MAX || k > INT_MAX || ldy > INT_MAX || ldy < n) return -1;
    const auto resolved = resolve_hipblaslt_algo(*state, type, (int) t, (int) n, (int) k, (int) ldy, beta);
    if (!resolved.supported || resolved.workspace_bytes > state->workspace_bytes) return -1;
    auto algo = resolved.algo;
    return hipblaslt_ext::getIndexFromAlgo(algo);
}

bool try_hipblaslt(void* opaque_state, strata::prefill::hipblaslt::InputType type, const uint16_t* x,
                   const uint16_t* w, float* y, int64_t t, int64_t n, int64_t k, int64_t ldy, float beta,
                   void* stream, int64_t ldx = 0) {
    auto* state = static_cast<HipLtState*>(opaque_state);
    if (!state || t <= 0 || n <= 0 || k <= 0 || t > INT_MAX || n > INT_MAX || k > INT_MAX || ldy > INT_MAX ||
        ldy < n) {
        return false;
    }
    const auto resolved = resolve_hipblaslt_algo(*state, type, (int) t, (int) n, (int) k, (int) ldy, beta);
    if (!resolved.supported) {
        ++state->fallbacks;
        state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
        return false;
    }
    if (resolved.workspace_bytes > state->workspace_bytes) {
        ++state->fallbacks;
        state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
        if (std::getenv("STRATA_HIPBLASLT_VERBOSE")) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution needs %zu workspace bytes, have %zu\n",
                         resolved.workspace_bytes, state->workspace_bytes);
        }
        return false;
    }

    HipLtDescriptors desc;
    const hipDataType input_type = type == strata::prefill::hipblaslt::InputType::bf16 ? HIP_R_16BF : HIP_R_16F;
    if (!desc.init(input_type, (int) t, (int) n, (int) k, (int) ldy, (int) ldx)) return false;
    const float alpha = 1.0f;
    const hipblasStatus_t status = hipblasLtMatmul(state->handle, desc.op, &alpha, w, desc.a, x, desc.b, &beta, y,
                                                   desc.c, y, desc.c, &resolved.algo, state->workspace,
                                                   state->workspace_bytes, (hipStream_t) stream);
    if (status == HIPBLAS_STATUS_SUCCESS) {
        ++state->lt_launches;
        return true;
    }

    std::fprintf(stderr, "prefill gemm: hipBLASLt launch failed with status %d\n", (int) status);
    if (beta != 0.0f) {
        std::fprintf(stderr, "prefill gemm: refusing a fallback after hipBLASLt failed with nonzero beta\n");
        std::exit(1);
    }
    auto* mutable_state = static_cast<HipLtState*>(opaque_state);
    uint32_t beta_bits = 0;
    std::memcpy(&beta_bits, &beta, sizeof(beta_bits));
    auto cached = mutable_state->cache.find(HipLtCallKey{type, (int) t, (int) n, (int) k, (int) ldy, beta_bits});
    if (cached != mutable_state->cache.end()) cached->second.supported = false;
    ++mutable_state->fallbacks;
    mutable_state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
    return false;
}
#endif

#if defined(__HIPCC__) && defined(STRATA_ROCBLAS_AVAILABLE)
// The rocBLAS solution table of f16_inplace (STRATA_ROCBLAS_TUNING: a table file, or a directory holding
// <arch>-rocblas-<version>.txt for this card and rocBLAS build; src/prefill/rocblas_tuning.hpp).  rocBLAS's own
// pick for some shapes is up to 6x slower than the best kernel it holds for them (gfx1030: tools/hip/tune_rocblas.cpp).
struct RocBlasState {
    rocblas_handle handle = nullptr;
    strata::prefill::rocblas_tuning::Table table;
    struct Pick { bool tuned = false; int solution = 0; int refusals = 0; };
    std::map<std::tuple<int, int, int, int>, Pick> cache;   // (T, N, K, ldy) -> the solution to run, or the default
    uint64_t tuned_launches = 0;
    uint64_t fallbacks = 0;

    ~RocBlasState() {
        if (std::getenv("STRATA_ROCBLAS_VERBOSE")) {
            std::fprintf(stderr, "prefill gemm: rocBLAS tuning summary tuned_launches=%llu fallbacks=%llu\n",
                         (unsigned long long) tuned_launches, (unsigned long long) fallbacks);
        }
        if (handle) rocblas_destroy_handle(handle);
    }
};

std::unique_ptr<RocBlasState> create_rocblas_state(void* stream) {
    const char* path = std::getenv("STRATA_ROCBLAS_TUNING");
    if (!path || !*path) return nullptr;
    auto state = std::make_unique<RocBlasState>();
    if (rocblas_create_handle(&state->handle) != rocblas_status_success) {
        std::fprintf(stderr, "prefill gemm: rocBLAS handle creation failed; its default kernels run\n");
        return nullptr;
    }
    rocblas_set_stream(state->handle, (hipStream_t) stream);
    char version[128] = {0};
    if (rocblas_get_version_string(version, sizeof version) != rocblas_status_success) {
        std::fprintf(stderr, "prefill gemm: rocBLAS version query failed; its default kernels run\n");
        return nullptr;
    }
    int device = 0;
    hipDeviceProp_t properties{};
    if (hipGetDevice(&device) != hipSuccess || hipGetDeviceProperties(&properties, device) != hipSuccess) {
        std::fprintf(stderr, "prefill gemm: HIP device query failed; rocBLAS's default kernels run\n");
        return nullptr;
    }
    std::string arch(properties.gcnArchName);
    if (const auto suffix = arch.find(':'); suffix != std::string::npos) arch.resize(suffix);
    std::filesystem::path file(path);
    std::error_code ec;
    if (std::filesystem::is_directory(file, ec)) {
        // the table of this card and this rocBLAS build (its version string, trailing dots dropped)
        std::string tag(version);
        while (!tag.empty() && tag.back() == '.') tag.pop_back();
        file /= arch + "-rocblas-" + tag + ".txt";
    }
    // The community-made tables (strata/core/amd_community.hpp) load only with STRATA_AMD_COMMUNITY_TUNING on (the default).
    if (!strata::amd_community::rocblas_table_allowed(arch, version)) {
        std::fprintf(stderr, "prefill gemm: the rocBLAS table for %s / %s is community-made and STRATA_AMD_COMMUNITY_TUNING=0; "
                             "rocBLAS's default kernels run\n", arch.c_str(), version);
        return nullptr;
    }
    std::string error;
    if (!state->table.load(file.string(), arch, version, error)) {
        std::fprintf(stderr, "prefill gemm: %s (%s); rocBLAS's default kernels run\n", error.c_str(),
                     file.string().c_str());
        return nullptr;
    }
    // rocBLAS loads its kernel library on the first product through a handle; a solution-index call before that
    // fails (rocblas_status_internal_error) although the index is valid.  One tiny default product opens it.
    {
        uint16_t* tiny = nullptr;
        if (hipMalloc(&tiny, 4 * 64 * 64 * 2) == hipSuccess) {
            const float one = 1.0f, zero = 0.0f;
            (void) rocblas_gemm_ex(state->handle, rocblas_operation_transpose, rocblas_operation_none, 64, 64, 64, &one,
                                   tiny, rocblas_datatype_f16_r, 64, tiny + 64 * 64, rocblas_datatype_f16_r, 64, &zero,
                                   tiny + 2 * 64 * 64, rocblas_datatype_f16_r, 128, tiny + 2 * 64 * 64,
                                   rocblas_datatype_f16_r, 128, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
            (void) hipStreamSynchronize((hipStream_t) stream);
            (void) hipFree(tiny);
        }
    }
    size_t tuned_rows = 0;
    for (const auto& row : state->table.rows()) tuned_rows += row.tuned ? 1 : 0;
    std::fprintf(stderr, "prefill gemm: rocBLAS tuning enabled (%zu rows, %zu with a solution, %s, rocBLAS %s)\n",
                 state->table.rows().size(), tuned_rows, arch.c_str(), version);
    return state;
}

// Gemm::f16_inplace's product through the table's solution for its shape.  False when the shape has no row, or the
// launch was refused (then the default kernel runs, and for that shape from now on).
bool try_rocblas_f16o(void* opaque_state, const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N,
                      int64_t K, int64_t ldy) {
    auto* state = static_cast<RocBlasState*>(opaque_state);
    if (!state || T <= 0 || N <= 0 || K <= 0 || T > INT_MAX || N > INT_MAX || K > INT_MAX || ldy > INT_MAX / 2) return false;
    const auto key = std::make_tuple((int) T, (int) N, (int) K, (int) ldy);
    auto it = state->cache.find(key);
    if (it == state->cache.end()) {
        RocBlasState::Pick pick;
        if (const auto* row = state->table.closest((int) N, (int) K, (int) ldy, (int) T); row && row->tuned) {
            pick.tuned = true;
            pick.solution = row->solution;
        }
        if (std::getenv("STRATA_ROCBLAS_VERBOSE")) {
            if (pick.tuned)
                std::fprintf(stderr, "prefill gemm: rocBLAS solution %d for T=%lld N=%lld K=%lld ldy=%lld\n",
                             pick.solution, (long long) T, (long long) N, (long long) K, (long long) ldy);
            else
                std::fprintf(stderr, "prefill gemm: rocBLAS default kernel; no row for T=%lld N=%lld K=%lld ldy=%lld\n",
                             (long long) T, (long long) N, (long long) K, (long long) ldy);
        }
        it = state->cache.emplace(key, pick).first;
    }
    if (!it->second.tuned) return false;
    const float one = 1.0f, zero = 0.0f;
    const rocblas_status status = rocblas_gemm_ex(
        state->handle, rocblas_operation_transpose, rocblas_operation_none, (int) N, (int) T, (int) K, &one, W,
        rocblas_datatype_f16_r, (int) K, X, rocblas_datatype_f16_r, (int) K, &zero, Y, rocblas_datatype_f16_r,
        (int) (2 * ldy), Y, rocblas_datatype_f16_r, (int) (2 * ldy), rocblas_datatype_f32_r,
        rocblas_gemm_algo_solution_index, it->second.solution, 0);
    if (status == rocblas_status_success) {
        ++state->tuned_launches;
        return true;
    }
    // A refusal writes nothing (the default kernel runs now).  The first one can be rocBLAS still setting up for the
    // kernel (its device workspace: seen as invalid_value once, success after); a solution refused three times is
    // dropped for this shape.
    ++state->fallbacks;
    if (++it->second.refusals >= 3) {
        std::fprintf(stderr, "prefill gemm: rocBLAS solution %d refused (status %d) for T=%lld N=%lld K=%lld ldy=%lld "
                             "three times; the default kernel runs for this shape\n", it->second.solution, (int) status,
                     (long long) T, (long long) N, (long long) K, (long long) ldy);
        it->second.tuned = false;
    } else if (std::getenv("STRATA_ROCBLAS_VERBOSE")) {
        std::fprintf(stderr, "prefill gemm: rocBLAS solution %d refused (status %d) for T=%lld N=%lld K=%lld ldy=%lld; "
                             "the default kernel this time\n", it->second.solution, (int) status, (long long) T,
                     (long long) N, (long long) K, (long long) ldy);
    }
    return false;
}
#endif

}  // namespace

bool prompt_f16() {
#if defined(__HIPCC__)
    constexpr int kMaxDev = 64;
    static std::atomic<int8_t> cached[kMaxDev] = {};   // 0 unknown, 1 off, 2 on
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= kMaxDev) { (void) hipGetLastError(); return false; }
    if (const int8_t c = cached[dev].load(std::memory_order_relaxed)) return c == 2;
    // (#835) STRATA_HIP_PROMPT_F16=1 turns it on.  It changes the prompt path's numbers (an FP16 rounding of each 16-bit GEMM's output), so a card
    // gets it unasked only where it was measured: the Radeon 780M (gfx1103, integrated, Linux) and the discrete RX 6000 cards (gfx1030 / 1031 / 1032, #5151;
    // strata::core::prompt_f16_resolve), where =0 is the way back.
    const char* e = std::getenv("STRATA_HIP_PROMPT_F16");
    hipDeviceProp_t p{};
    if (hipGetDeviceProperties(&p, dev) != hipSuccess) { (void) hipGetLastError(); p.gcnArchName[0] = 0; p.integrated = 0; }
#if defined(_WIN32)
    constexpr bool kWin = true;
#else
    constexpr bool kWin = false;
#endif
    const bool on = strata::core::prompt_f16_resolve(e, p.gcnArchName, p.integrated != 0, kWin);
    if (on && e != nullptr) std::fprintf(stderr, "strata prefill: HIP device %d%s%s - STRATA_HIP_PROMPT_F16=1: the prompt's 16-bit GEMMs run FP16 in and out "
                                 "(rocBLAS is tuned only for that on gfx103x; not bit-identical to the default)%s",
                         dev, p.gcnArchName[0] ? " " : "", p.gcnArchName, "\n");
    else if (on && std::strncmp(p.gcnArchName, "gfx103", 6) == 0)
        std::fprintf(stderr, "strata prefill: HIP device %d %s - the prompt's 16-bit GEMMs run FP16 in and out by default here (rocBLAS is tuned for that on "
                             "gfx103x; STRATA_HIP_PROMPT_F16=0 keeps the old route)%s", dev, p.gcnArchName, "\n");
    // (#5151: the start-up tip that asked gfx103x owners to set STRATA_HIP_PROMPT_F16=1 is gone: users never set variables.  On a discrete RX 6000 card the
    // FP16 route is the default now (prompt_f16_resolve): against the exact FP32 products it is no further from exact than the old route and 11-14 % faster.)
    cached[dev].store(on ? 2 : 1, std::memory_order_relaxed);
    return on;
#else
    return false;
#endif
}

// A padded X (ldx > K) no kernel took: its rows copied to a contiguous FP16 buffer of stride K.  Grown, never shrunk
// (one buffer per Gemm; a padded X only arrives from STRATA_PF_PAD's producers).  One 2-D copy: the same values, and
// the ordinary paths below then read them exactly as if the producer had written them unpadded.
const uint16_t* Gemm::unpad_x(const uint16_t* X, int64_t T, int64_t K, int64_t ldx) {
    if (X == nullptr || T <= 0 || K <= 0 || ldx <= K) return nullptr;
    const int64_t need = T * K;
    if (need > unpadded_x_elems_) {
        if (unpadded_x_ != nullptr) { cudaFree(unpadded_x_); unpadded_x_ = nullptr; unpadded_x_elems_ = 0; }
        void* p = nullptr;
        if (cudaMalloc(&p, (size_t) need * sizeof(uint16_t)) != cudaSuccess) { cudaGetLastError(); return nullptr; }
        unpadded_x_ = (uint16_t*) p;
        unpadded_x_elems_ = need;
    }
    if (cudaMemcpy2DAsync(unpadded_x_, (size_t) K * sizeof(uint16_t), X, (size_t) ldx * sizeof(uint16_t),
                          (size_t) K * sizeof(uint16_t), (size_t) T, cudaMemcpyDeviceToDevice,
                          (cudaStream_t) stream_) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    return unpadded_x_;
}

bool gemm_dequant_side() {
#if defined(__HIPCC__)
    return false;
#else
    static const bool on = [] { const char* v = std::getenv("STRATA_DEQUANT_SIDE"); return v && v[0] == '1'; }();
    return on;
#endif
}

Gemm::~Gemm() {
    for (W16Entry& e : w16_) if (e.d) (void) cudaFree(e.d);
    for (void* e : side_ready_)
        if (e) (void) cudaEventDestroy((cudaEvent_t) e);
    for (void* e : side_free_)
        if (e) (void) cudaEventDestroy((cudaEvent_t) e);
    if (side_stream_) (void) cudaStreamDestroy((cudaStream_t) side_stream_);
#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)
    delete static_cast<strata::prefill::mmq::Context*>(mmq_ctx_);
    if (mmq_buf_) cudaFree(mmq_buf_);
#endif
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    delete static_cast<HipLtState*>(hipblaslt_state_);
#endif
#if defined(__HIPCC__) && defined(STRATA_ROCBLAS_AVAILABLE)
    delete static_cast<RocBlasState*>(rocblas_state_);
#endif
#if !defined(__HIPCC__) && !defined(STRATA_HIP_GFX906)
    delete static_cast<CudaLtCache*>(lt_cache_);
    lt_cache_ = nullptr;
    if (lt_) {
        (void) cublasLtDestroy((cublasLtHandle_t) lt_);
        lt_ = nullptr;
    }
#endif
    if (handle_) cublasDestroy((cublasHandle_t) handle_);
    if (tc_w_) cudaFree(tc_w_);
    if (tc_x_) cudaFree(tc_x_);
    if (unpadded_x_) cudaFree(unpadded_x_);
    if (!external_) {
        if (scratch_) cudaFree(scratch_);
        if (workspace_) cudaFree(workspace_);
    }
}

bool Gemm::init_external(void* stream, uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes,
                         std::string& err) {
    cublasHandle_t h = nullptr;
    if (const cublasStatus_t s = cublasCreate(&h); s != CUBLAS_STATUS_SUCCESS) {
        err = "prefill gemm: cublasCreate: cuBLAS status " + std::to_string((int) s);
        return false;
    }
    handle_ = h;
    stream_ = stream;
    external_ = true;
    note(cublasSetStream(h, (cudaStream_t) stream), "cublasSetStream");
    workspace_ = workspace;
    ws_bytes_ = ws_bytes;
    note(cublasSetWorkspace(h, workspace_, ws_bytes), "cublasSetWorkspace");
    note(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
#if !defined(__HIPCC__) && !defined(STRATA_HIP_GFX906)
    {
        cublasLtHandle_t lth = nullptr;
        const cublasStatus_t lts = cublasLtCreate(&lth);
        if (lts == CUBLAS_STATUS_SUCCESS) {
            lt_ = lth;
            lt_cache_ = new CudaLtCache();
        } else {
            note(lts, "cublasLtCreate");
        }
    }
#endif
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    hipblaslt_state_ = create_hipblaslt_state(workspace_, ws_bytes).release();
#endif
#if defined(__HIPCC__) && defined(STRATA_ROCBLAS_AVAILABLE)
    rocblas_state_ = create_rocblas_state(stream).release();
#endif
    return true;
}

void Gemm::rebind(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes) {
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    workspace_ = workspace;
    ws_bytes_ = ws_bytes;
    cublasSetWorkspace((cublasHandle_t) handle_, workspace_, ws_bytes);
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (hipblaslt_state_) {
        auto* state = static_cast<HipLtState*>(hipblaslt_state_);
        state->workspace = workspace_;
        state->workspace_bytes = ws_bytes;
    }
#endif
}

void Gemm::set_temp_scratch(void* ptr, size_t elems) {
    if (ptr == nullptr || elems == 0) {
        temp_scratch_ = nullptr;
        temp_scratch_elems_ = 0;
    } else {
        temp_scratch_ = ptr;
        temp_scratch_elems_ = elems;
    }
}

bool Gemm::set_dequant_side(uint16_t* scratch2, std::string& err) {
    side_scratch_ = scratch2;
    side_cur_ = 0;
    if (scratch2 == nullptr) return true;
    if (side_stream_ == nullptr) {   // once: the stream and the four events outlive every rebind
        cudaStream_t s = nullptr;
        if (const cudaError_t e = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking); e != cudaSuccess) {
            err = std::string("prefill gemm: dequant side stream: ") + cudaGetErrorString(e);
            side_scratch_ = nullptr;
            return false;
        }
        side_stream_ = s;
    }
    for (int i = 0; i < 2; ++i) {
        if (side_ready_[i] == nullptr) {
            cudaEvent_t r = nullptr;
            if (cudaEventCreateWithFlags(&r, cudaEventDisableTiming) != cudaSuccess) {
                err = "prefill gemm: dequant side events";
                side_scratch_ = nullptr;
                return false;
            }
            side_ready_[i] = r;   // kept even on a later failure: the destructor frees what was made
        }
        if (side_free_[i] == nullptr) {
            cudaEvent_t f = nullptr;
            if (cudaEventCreateWithFlags(&f, cudaEventDisableTiming) != cudaSuccess) {
                err = "prefill gemm: dequant side events";
                side_scratch_ = nullptr;
                return false;
            }
            side_free_[i] = f;
        }
    }
    // both scratches are free once what the compute stream holds now is done (a rebind follows a stream sync, so
    // this is belt and braces; a wait on an event never recorded would pass anyway)
    for (int i = 0; i < 2; ++i)
        (void) cudaEventRecord((cudaEvent_t) side_free_[i], (cudaStream_t) stream_);
    return true;
}

bool Gemm::init(void* stream, int64_t scratch_elems, std::string& err) {
    // #240: every failure names the call and the real status, so "no VRAM" can be told from a broken install
    cublasHandle_t h = nullptr;
    if (const cublasStatus_t s = cublasCreate(&h); s != CUBLAS_STATUS_SUCCESS) {
        err = "prefill gemm: cublasCreate: cuBLAS status " + std::to_string((int) s);
        return false;
    }
    handle_ = h;
    stream_ = stream;
    note(cublasSetStream(h, (cudaStream_t) stream), "cublasSetStream");
    // A fixed workspace so the handle never allocates on the way (and graphs could capture it later).
    const size_t ws = 32u << 20;
    if (const cudaError_t e = cudaMalloc(&workspace_, ws); e != cudaSuccess) {
        err = std::string("prefill gemm: workspace of 32 MiB: ") + cudaGetErrorString(e);
        return false;
    }
    ws_bytes_ = ws;
    note(cublasSetWorkspace(h, workspace_, ws), "cublasSetWorkspace");
    note(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
#if !defined(__HIPCC__) && !defined(STRATA_HIP_GFX906)
    {
        cublasLtHandle_t lth = nullptr;
        const cublasStatus_t lts = cublasLtCreate(&lth);
        if (lts == CUBLAS_STATUS_SUCCESS) {
            lt_ = lth;
            lt_cache_ = new CudaLtCache();
        } else {
            note(lts, "cublasLtCreate");
        }
    }
#endif
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    hipblaslt_state_ = create_hipblaslt_state(workspace_, ws).release();
#endif
#if defined(__HIPCC__) && defined(STRATA_ROCBLAS_AVAILABLE)
    rocblas_state_ = create_rocblas_state(stream).release();
#endif
    if (scratch_elems > 0) {
        if (const cudaError_t e = cudaMalloc((void**) &scratch_, (size_t) scratch_elems * 2); e != cudaSuccess) {
            err = "prefill gemm: dequant scratch of " + std::to_string(scratch_elems * 2 >> 20) + " MiB: " +
                  cudaGetErrorString(e);
            return false;
        }
    }
    scratch_elems_ = scratch_elems;
    return true;
}

#if !defined(__HIPCC__)
// ---- below sm_80: BF16 GEMMs without BF16 tensor cores (PR #655, #540, #395) ----------------------------------------
// Turing and Volta have no BF16 tensor cores: cublasGemmEx on CUDA_R_16BF inputs falls back to a SIMT fp32 kernel
// (magma_sgemmEx), a fifth of a 4K prompt on an RTX 2080 Ti (#655) and ~10% of a V100's prompt (#540).  BF16 -> FP16
// is exact for every value inside FP16's normal range (the 7-bit mantissa fits in 10 bits); weights and normalized
// activations sit there.  With the FP16 path, the weight is converted once per call and the activations in row
// slices, into the instance's own buffers, and the product runs as the FP16 GEMM (fp32 accumulate) on the tensor
// cores.  Finite values beyond FP16's range are clamped to +-65504 (#540) instead of becoming Inf.  A beta = 1
// product (STRATA_PREFILL_BF16X2's remainder, ~2^-9 of the original, deep in FP16's subnormal band) keeps cuBLAS.
//   Default: on for compute capability 7.0 - 7.4 (Volta: only the experimental STRATA_EXPERIMENTAL_SM60 build runs
//   there), OFF for 7.5 (RTX 20 in the ready-made engine: the sums round differently, so it is opt-in until it has
//   been gated).  STRATA_BF16_TC=1 / =0 turns it on / off on any 7.x card (=2, a test mode: on any card).
// Pascal (6.x) has no tensor cores and cuBLAS has no BF16 GEMM for it (#395, measured NOT_SUPPORTED on a P40): both
// operands are widened to fp32 by an exact shift and the product is cublasSgemm with the same fp32 accumulator.
namespace {
__global__ void bf16_to_f16_kernel(const uint16_t* __restrict__ in, __half* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float f = __uint_as_float((uint32_t) in[i] << 16);
        if (f > 65504.0f && !isinf(f)) f = 65504.0f;
        else if (f < -65504.0f && !isinf(f)) f = -65504.0f;
        out[i] = __float2half_rn(f);
    }
}
void bf16_to_f16(const uint16_t* in, uint16_t* out, int64_t n, cudaStream_t st) {
    if (n <= 0) return;
    bf16_to_f16_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, st>>>(in, reinterpret_cast<__half*>(out), n);
}
#if defined(STRATA_EXPERIMENTAL_SM60)   // Pascal runs only the experimental build
__global__ void bf16_to_f32_kernel(const uint16_t* __restrict__ in, float* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __uint_as_float((uint32_t) in[i] << 16);
}
void bf16_to_f32(const uint16_t* in, float* out, int64_t n, cudaStream_t st) {
    if (n <= 0) return;
    bf16_to_f32_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, st>>>(in, out, n);
}
#endif
// The current device's compute capability as 10 * major + minor, per device (a layer split can mix cards); 0: not
// known, read as "not an old card" so a failed query keeps the cuBLAS BF16 call.
int current_cc() {
    static std::atomic<int> cc[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return 0; }
    int v = cc[dev].load(std::memory_order_relaxed);
    if (v == 0) {
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
            cudaGetLastError();
            return 0;
        }
        v = 10 * major + minor;
        cc[dev].store(v, std::memory_order_relaxed);
    }
    return v;
}
// 0: cuBLAS's BF16 GEMM, 1: through FP16 (tensor cores), 2: through FP32 (Pascal)
int bf16_path() {
    static const int forced = [] {
        const char* v = std::getenv("STRATA_BF16_TC");
        return v != nullptr && v[0] != '\0' ? std::atoi(v) : -1;
    }();
    const int cc = current_cc();
    if (forced == 2 && cc > 0) return 1;   // a test mode: the FP16 path on any card (gemm_bf16_parity on sm_80+)
    if (cc <= 0 || cc >= 80) return 0;
    if (cc < 70) return 2;
    if (forced >= 0) return forced != 0 ? 1 : 0;
    return cc < 75 ? 1 : 0;
}
bool grow(uint16_t*& p, int64_t& have, int64_t want) {   // `have`, `want`: 2-byte elements
    if (have >= want) return true;
    if (p) cudaFree(p);
    p = nullptr;
    have = 0;
    if (cudaMalloc((void**) &p, (size_t) want * 2) != cudaSuccess) { cudaGetLastError(); p = nullptr; return false; }
    have = want;
    return true;
}
constexpr int64_t kXSliceElems = 16ll << 20;   // 32 MiB of FP16 activations per slice (64 MiB as fp32)

#if defined(STRATA_EXPERIMENTAL_SM60)
// ---- Pascal, opt-in (STRATA_DENSE_F32=1): the dense projections' FP16-operand GEMMs as cublasSgemm ----------------
// Measured on a Tesla P40 (sm_61) with cuBLAS 12.4.5.8 (bench/results/2026-10-12-p40-sm61-kernels).  For FP16
// operands with FP32 sums cuBLAS has two Maxwell kernels there, maxwell_sgemm_fp16_64x64 and _32x128, and picks the
// slow one for most chunk sizes: the 10240 x 2560 qkv projection runs at 7.6 TFLOPS for T 7030-7040 and at 4.3 from
// T 7041 (49 -> 85 ms), and at 4.3-4.6 for two thirds of the T between 256 and 8192; the 12288-row query projection
// likewise.  A 7K-token prompt reads in 16.3 or 19.5 s by its token count.  cublasSgemm on FP32 copies
// (maxwell_sgemm_128x64) runs at 8.4-9.8 TFLOPS for every T from 512 (6.7+ from 256).  The copies are exact
// (FP16 -> FP32), and for the shapes and sizes taken here the two kernels add every output's products in the same
// order: the result is bitwise the cuBLAS FP16 call's (gemm_f32_parity: 5,201 chunk sizes on a P40 with cuBLAS 12.4, a
// measurement of this library, not a guarantee of cuBLAS's; below the sizes taken here - T under
// ~144 at 6144+ rows, under ~408 at 2560 rows - cuBLAS switches one of them to another kernel family and the bits
// differ, so those keep the FP16 call).  The weight goes through the BF16 path's conversion buffer (126 MiB for the
// largest, 21 MiB more than that path's own), the activations through its slices.
// Opt-in, and on Pascal (compute capability 6.x) only: the equality belongs to the cuBLAS build, so run
// gemm_f32_parity with yours before turning it on; newer cards have tensor cores for the FP16 call.
__global__ void f16_to_f32_kernel(const uint16_t* __restrict__ in, float* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(__ushort_as_half(in[i]));
}
void f16_to_f32(const uint16_t* in, float* out, int64_t n, cudaStream_t st) {
    if (n <= 0) return;
    f16_to_f32_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, st>>>(in, out, n);
}
int g_dense_f32 = -1;   // gemm_set_dense_f32 (the parity test: 0 / 1 on any card); -1 = STRATA_DENSE_F32
bool dense_f32_on() {
    static const bool env = [] {
        const char* v = std::getenv("STRATA_DENSE_F32");
        return v != nullptr && std::atoi(v) != 0;
    }();
    if (g_dense_f32 >= 0) return g_dense_f32 != 0;
    if (!env) return false;
    const int cc = current_cc();
    return cc >= 60 && cc < 70;
}
// The smallest T taken for a weight of N rows and K columns (0: never): the measured shapes only - the 2560-column
// qkv (10240 rows), query (12288) and gate (6144) projections and the 2560-row output projections (6144 and 4096
// columns), each from a T where the FP32 route (its conversions included) is the faster one or within ~7% of the FP16
// call (the gate projection below ~1300 tokens, the 4096-column output projection near 4096) and well above the T
// where the two stop agreeing bitwise.  Every other shape (a row slice of a weight too) keeps the FP16 call.
int64_t dense_f32_min_t(int64_t N, int64_t K) {
    if (K == 2560 && (N == 10240 || N == 12288)) return 256;
    if (K == 2560 && N == 6144) return 512;
    if (N == 2560 && (K == 6144 || K == 4096)) return 4096;
    return 0;
}
#endif
}  // namespace
#endif
#if !defined(__HIPCC__) && defined(STRATA_EXPERIMENTAL_SM60)
void gemm_set_dense_f32(int mode) { g_dense_f32 = mode; }
#else
void gemm_set_dense_f32(int) {}
#endif
bool Gemm::bf16_hcd_exact(const uint16_t* X, int64_t ldx, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K) {
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    static std::atomic<bool> told{false};
    if (!hipblaslt_state_ || N != 320 || K != 10240 || T < 1) {
        if (N == 320 && K == 10240 && !told.exchange(true))
            std::fprintf(stderr, "strata: STRATA_HCD_EXACT: no hipBLASLt tuning table is loaded, so the exact kernel has nothing to match: "
                                 "the hyper-connection down projection stays on hipBLAS\n");
        return false;
    }
    // The kernel copies hipBLASLt solution 1176 / 1177's summation order, which is only true of the library build the
    // table was calibrated with: take it only when the table makes hipBLASLt pick one of those two for this T.
    const int id = hipblaslt_solution_for(hipblaslt_state_, strata::prefill::hipblaslt::InputType::bf16, T, N, K, N, 0.0f);
    if (id != 1176 && id != 1177) {
        if (T >= 4096 && !told.exchange(true))   // (a short chunk may legitimately get another solution: not worth a line)
            std::fprintf(stderr, "strata: STRATA_HCD_EXACT: the tuning table gives the hyper-connection down projection (N 320, K 10240) "
                                 "solution %d at T %lld, not 1176 / 1177: hipBLASLt runs it (a changed table or library; "
                                 "tools/hip/gfx1151-hipblaslt-100401.txt is the one measured)\n",
                         id, (long long) T);
        return false;
    }
    return strata_pf_hcdown_exact_bf16(X, ldx, W, Y, T, N, K, stream_);
#else
    (void) X; (void) ldx; (void) W; (void) Y; (void) T; (void) N; (void) K;
    return false;
#endif
}

#if defined(__HIPCC__)
// ---- RDNA2 (gfx103x): the prompt path's GEMMs as FP32 SGEMMs ------------------------------------------------------
// The GEMMs here take 16-bit inputs (FP16 or BF16) and write FP32 (rocBLAS's "HS" / "BS" types).  The rocBLAS the engine
// ships has tuned gfx1030 kernels for FP16 -> FP16 ("HH"), int8 and FP32 -> FP32 ("SS") only: every 16-bit-input,
// FP32-output GEMM runs a "fallback" kernel at ~5 TFLOPS.  Measured on an RX 6800 (ROCm 10.2.0a20260930) with the
// engine's exact calls (opA = T, opB = N, fp32 compute): at the Coder's prompt shapes the FP16 GEMMs of one fresh
// 12K-token prompt take 14.1 s, the same GEMMs as SGEMM on widened copies 4.7 s including the widening (e.g.
// N 10240 x T 8192 x K 2560: 86.7 -> 27.7 ms); the BF16 hyper-connection GEMMs 10.9 -> 7.5 ms (N 320, K 10240).
// Widening FP16 / BF16 to FP32 is exact and SGEMM accumulates in FP32 like the native call, so the products are the
// same up to summation order.  The weight is widened once per call, the activations in slices of up to 128 MiB of
// FP32 (an 8,192-token chunk at K = 2560 is one slice).  Small N (< 64: routers, gates) keeps the native call:
// there the widening costs more than the GEMM gains.  If the FP32 buffers cannot be allocated, the native call runs.
//   OPT-IN here: STRATA_RDNA2_SGEMM=1 turns it on for gfx103x devices (other cards keep the native call, also in a
//   mixed layer split); =all turns it on for every AMD card (A/B arms, tests); unset or 0 is the native path.  The
//   author ran it as the gfx103x default (RX 6800 343 -> 534 tok/s); it is not bit-identical to the native call.
namespace {
__global__ void rdna2_bf16_to_f32_kernel(const uint16_t* __restrict__ in, float* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __uint_as_float((uint32_t) in[i] << 16);
}
__global__ void rdna2_f16_to_f32_kernel(const __half* __restrict__ in, float* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}
void rdna2_widen(const uint16_t* in, float* out, int64_t n, bool bf16, cudaStream_t st) {
    if (n <= 0) return;
    const unsigned blocks = (unsigned) ((n + 255) / 256);
    if (bf16) rdna2_bf16_to_f32_kernel<<<blocks, 256, 0, st>>>(in, out, n);
    else rdna2_f16_to_f32_kernel<<<blocks, 256, 0, st>>>(reinterpret_cast<const __half*>(in), out, n);
}
// whether this GEMM runs as an SGEMM (see above)
bool rdna2_sgemm_on(int64_t N) {
    static const int mode = [] {   // 0 off (unset), 1 gfx103x devices, 2 every AMD device
        const char* v = std::getenv("STRATA_RDNA2_SGEMM");
        if (v == nullptr || v[0] == '\0') return 0;
        if (std::strcmp(v, "all") == 0) return 2;
        return std::atoi(v) > 0 ? 1 : 0;
    }();
    if (mode == 0 || N < 64) return false;
    if (mode == 2) return true;
    static std::atomic<int> arch[64] = {};   // per device: 0 not known, 1 gfx103x, 2 another architecture
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 64) { (void) hipGetLastError(); return false; }
    int v = arch[dev].load(std::memory_order_relaxed);
    if (v == 0) {
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, dev) != hipSuccess) { (void) hipGetLastError(); return false; }
        v = std::strncmp(prop.gcnArchName, "gfx103", 6) == 0 ? 1 : 2;
        arch[dev].store(v, std::memory_order_relaxed);
    }
    return v == 1;
}
// `have`, `want`: FP32 elements (the buffers are the instance's 2-byte tc_ buffers, sized in 2-byte elements)
bool rdna2_grow(uint16_t*& p, int64_t& have_u16, int64_t want_f32) {
    const int64_t want = 2 * want_f32;
    if (have_u16 >= want) return true;
    if (p) cudaFree(p);
    p = nullptr;
    have_u16 = 0;
    if (cudaMalloc((void**) &p, (size_t) want * 2) != cudaSuccess) { (void) cudaGetLastError(); p = nullptr; return false; }
    have_u16 = want;
    return true;
}
constexpr int64_t kRdna2SliceF32 = 32ll << 20;   // 128 MiB of FP32 activations per slice
}  // namespace

// Y^T[N, T] (+)= W[N, K] . X^T[K, T] as SGEMMs on FP32 copies; false: not this card / shape, or no buffers (the caller
// runs its native GEMM, nothing was written)
bool Gemm::rdna2_sgemm(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                       float beta, bool bf16) {
    if (K <= 0 || !rdna2_sgemm_on(N)) return false;
    const int64_t rows = std::max<int64_t>(1, std::min<int64_t>(T, kRdna2SliceF32 / K));
    // growing frees a buffer an earlier GEMM on this stream may still read (as the sm_60 FP32 route does)
    if (tc_w_elems_ < 2 * N * K || tc_x_elems_ < 2 * rows * K) (void) hipStreamSynchronize((hipStream_t) stream_);
    if (!rdna2_grow(tc_w_, tc_w_elems_, N * K) || !rdna2_grow(tc_x_, tc_x_elems_, rows * K)) return false;
    float* const wf = reinterpret_cast<float*>(tc_w_);
    float* const xf = reinterpret_cast<float*>(tc_x_);
    const float alpha = 1.0f;
    rdna2_widen(W, wf, N * K, bf16, (cudaStream_t) stream_);
    for (int64_t t0 = 0; t0 < T; t0 += rows) {
        const int64_t n = std::min<int64_t>(rows, T - t0);
        rdna2_widen(X + t0 * K, xf, n * K, bf16, (cudaStream_t) stream_);
        // every slice is a disjoint block of Y, so each gets the caller's beta
        ck(hipblasSgemm((hipblasHandle_t) handle_, HIPBLAS_OP_T, HIPBLAS_OP_N, (int) N, (int) n, (int) K, &alpha, wf,
                        (int) K, xf, (int) K, &beta, Y + t0 * ldy, (int) ldy),
           "hipblasSgemm (RDNA2)");
    }
    return true;
}
#endif

bool Gemm::bf16_converts_x() const {
#if defined(__HIPCC__) || defined(STRATA_EXPERIMENTAL_SM60)
    return false;
#else
    return bf16_path() == 1;
#endif
}
void Gemm::reserve_bf16_tc(int64_t w_elems, int64_t x_elems) {
#if !defined(__HIPCC__)
    const int path = bf16_path();
    if (path != 1 && path != 2) return;
    const int mult = (path == 2) ? 2 : 1;
    if (w_elems > 0) grow(tc_w_, tc_w_elems_, mult * w_elems);
    if (x_elems > 0) grow(tc_x_, tc_x_elems_, mult * std::min<int64_t>(x_elems, kXSliceElems));
#else
    (void) w_elems; (void) x_elems;
#endif
}
__global__ void f16_to_bf16_kernel(const __half* __restrict__ in, uint16_t* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (uint16_t) (__float_as_uint(__half2float(in[i])) >> 16);   // exact: the value was a BF16
}
void Gemm::bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                float beta, int64_t ldx, bool x_f16) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    if (ldx <= K) ldx = 0;
#ifdef STRATA_USE_HIP
    // #313 (opt-in STRATA_WMMA_GEMM=1; STRATA_WMMA_BF16=0 excludes bf16): RDNA3 / RDNA3.5 WMMA dense GEMM, before
    // hipBLASLt; it returns false for shapes it does not take (and off gfx11) and this falls through
    static const bool wmma_on = [] { const char* v = std::getenv("STRATA_WMMA_GEMM"); return v && v[0] != 0 && v[0] != '0'; }();
    static const bool bf16_on = [] { const char* v = std::getenv("STRATA_WMMA_BF16"); return !v || (v[0] != 0 && v[0] != '0'); }();
    if (wmma_on && bf16_on && ldx == 0 && T >= 16 && (beta == 0.0f || beta == 1.0f) &&
        strata_wmma_gemm_bf16(X, W, Y, T, N, K, ldy, beta, stream_)) {
        return;
    }
#endif
#if defined(__HIPCC__)
    if (f16_io_) {   // X is the FP16 image (set_act_f16); W goes through the dequantization scratch as FP16
        // the only accumulating BF16 product is the BF16X2 low part, which bf16x2_mode() turns off with this path
        if (beta != 0.0f) { std::fprintf(stderr, "prefill gemm: an accumulating BF16 GEMM on the FP16 prompt path\n"); std::exit(1); }
        const int64_t rows = std::min<int64_t>(N, scratch_elems_ / K);
        if (rows <= 0) { std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long) K); std::exit(1); }
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = std::min(rows, N - r0);
            bf16_to_f16_rows<<<1024, 256, 0, (cudaStream_t) stream_>>>(W + r0 * K, (__half*) scratch_, n * K);
            f16_inplace(X, scratch_, Y + r0, T, n, K, ldy);
        }
        return;
    }
#endif
    const float alpha = 1.0f;
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (try_hipblaslt(hipblaslt_state_, strata::prefill::hipblaslt::InputType::bf16, X, W, Y, T, N, K, ldy,
                      beta, stream_, ldx)) {
        STRATA_ABSORB_HIPBLAS_STICKY("hipBLASLt bf16");
        return;
    }
#endif
#if defined(__HIPCC__)
    // a padded X (STRATA_PF_PAD, ldx > K) keeps the native call: the widening copies contiguous rows
    if (ldx == 0 && rdna2_sgemm(X, W, Y, T, N, K, ldy, beta, true)) return;
#endif
#if !defined(__HIPCC__)
#if defined(STRATA_EXPERIMENTAL_SM60)
    constexpr bool kPascalPath = true;
#else
    constexpr bool kPascalPath = false;
#endif
    if (const int path = K > 0 ? bf16_path() : 0; (path == 1 && N > 1 && beta == 0.0f) || (path == 2 && kPascalPath)) {
        // FP16 (path 1) or FP32 (path 2, Pascal) copies of the weight and of X, converted in tiles.  Issue #1757: when
        // the buffers did not fit in the VRAM that was left (a stage of a layer split keeps ~100 MiB), this fell through
        // to the raw BF16 cuBLAS call, which a Pascal card answers with CUBLAS_STATUS_INTERNAL_ERROR (14).  Now the
        // weight is converted in tiles of N rows (the whole weight when it fits, as before) and X in slices; a card with
        // no room even for a tile says so.  A single output row stays cuBLAS's GEMV for path 1 (faster than the
        // conversions).  Every tile is a disjoint block of Y, so each gets the caller's beta.
        const int mult = path == 2 ? 2 : 1;   // 2-byte elements per converted value
        int64_t w_rows = N;
        int64_t x_rows = std::max<int64_t>(1, std::min<int64_t>(T, kXSliceElems / K));
        bool fit = false;
        for (;;) {
            if (grow(tc_w_, tc_w_elems_, mult * w_rows * K) && (x_f16 || grow(tc_x_, tc_x_elems_, mult * x_rows * K))) { fit = true; break; }
            if (w_rows <= 64 && x_rows <= 64) break;
            w_rows = std::max<int64_t>(std::min<int64_t>(w_rows, 64), w_rows / 2);
            x_rows = std::max<int64_t>(std::min<int64_t>(x_rows, 64), x_rows / 2);
        }
        if (!fit && path == 2) {   // Pascal has no BF16 cuBLAS call to fall back to; Volta keeps its raw one below
            std::fprintf(stderr, "prefill gemm: no GPU memory left for the %s conversion buffers of a %lld x %lld BF16 weight "
                                 "(Pascal cards run the BF16 products through them); free VRAM on this card: raise "
                                 "--vram-reserve-mib, lower --max-context or --kv-resident, or give it fewer layers\n",
                         "FP32", (long long) N, (long long) K);
            std::exit(1);
        }
        if (fit) {
            for (int64_t n0 = 0; n0 < N; n0 += w_rows) {
                const int64_t nn = std::min<int64_t>(w_rows, N - n0);
                if (path == 2) {
#if defined(STRATA_EXPERIMENTAL_SM60)
                    float* const wf = reinterpret_cast<float*>(tc_w_);
                    float* const xf = reinterpret_cast<float*>(tc_x_);
                    bf16_to_f32(W + n0 * K, wf, nn * K, (cudaStream_t) stream_);
                    for (int64_t t0 = 0; t0 < T; t0 += x_rows) {
                        const int64_t n = std::min<int64_t>(x_rows, T - t0);
                        bf16_to_f32(X + t0 * K, xf, n * K, (cudaStream_t) stream_);
                        ck(cublasSgemm((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) nn, (int) n, (int) K, &alpha,
                                       wf, (int) K, xf, (int) K, &beta, Y + t0 * ldy + n0, (int) ldy),
                           "cublasSgemm (bf16 on Pascal)");
                    }
#endif
                } else {
                    bf16_to_f16(W + n0 * K, tc_w_, nn * K, (cudaStream_t) stream_);
                    for (int64_t t0 = 0; t0 < T; t0 += x_rows) {
                        const int64_t n = std::min<int64_t>(x_rows, T - t0);
                        if (x_f16) {   // already the converted values (the same slices: the same GEMMs)
                            f16(X + t0 * K, tc_w_, Y + t0 * ldy + n0, n, nn, K, ldy, beta);
                        } else {
                            bf16_to_f16(X + t0 * K, tc_x_, n * K, (cudaStream_t) stream_);
                            f16(tc_x_, tc_w_, Y + t0 * ldy + n0, n, nn, K, ldy, beta);
                        }
                    }
                }
            }
            return;
        }
    }
#endif
#if !defined(__HIPCC__)
    if (x_f16) {   // not the tensor-core path after all (one output row, or no room): X back to BF16, a slice at a time
        const int64_t x_rows = std::max<int64_t>(1, std::min<int64_t>(T, kXSliceElems / K));
        if (!grow(tc_x_, tc_x_elems_, x_rows * K)) {
            std::fprintf(stderr, "prefill gemm: no GPU memory for converting an FP16 image back to BF16\n");
            std::exit(1);
        }
        for (int64_t t0 = 0; t0 < T; t0 += x_rows) {
            const int64_t n = std::min<int64_t>(x_rows, T - t0);
            f16_to_bf16_kernel<<<(unsigned) ((n * K + 255) / 256), 256, 0, (cudaStream_t) stream_>>>(
                reinterpret_cast<const __half*>(X + t0 * K), tc_x_, n * K);
            bf16(tc_x_, W, Y + t0 * ldy, n, N, K, ldy, beta, 0, false);
        }
        return;
    }
#endif
    // Column-major view: Y^T[N, T] = W[N, K] (stored K x N col-major, transposed) . X^T[K, T].
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16BF, (int) K, X, CUDA_R_16BF, (int) (ldx ? ldx : K), &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx");
    STRATA_ABSORB_HIPBLAS_STICKY("cublasGemmEx");
}

bool Gemm::try_cublaslt_tune(const void* X, const void* W, void* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                             float beta, bool is_f32, bool out_f16) {
#if !defined(__HIPCC__) && !defined(STRATA_HIP_GFX906)
    if (!lt_ || !lt_cache_ || cublaslt_tune_disabled()) return false;
    // Prefill path on Ampere / Ada / Hopper / Blackwell (sm_80+); Volta sm_70 / Turing sm_75 / Pascal sm_60 keep cublasGemmEx / cublasSgemm.
    if (T < 256 || N <= 0 || K <= 0 || T > INT_MAX || N > INT_MAX || K > INT_MAX || ldy < N || ldy > INT_MAX) return false;
    if (current_cc() < 80) return false;
    // Require 16-byte pointer alignment so any cached 128-bit vectorized kernel works across all layers / scratch buffers.
    if ((((uintptr_t) W | (uintptr_t) X | (uintptr_t) Y) & 15u) != 0u) return false;

    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }

    const cublasLtHandle_t lt_handle = (cublasLtHandle_t) lt_;
    const cudaStream_t st = (cudaStream_t) stream_;
    const size_t avail_ws = workspace_ ? ws_bytes_ : 0;
    const uint64_t max_ws_u64 = (uint64_t) avail_ws;
    void* const ws_ptr = avail_ws ? workspace_ : nullptr;

    auto* const local_cache = static_cast<CudaLtCache*>(lt_cache_);
    const CudaLtKey key{dev, (int) N, (int) T, (int) K, (int) ldy, is_f32 ? 1 : 0, out_f16 ? 1 : 0, avail_ws};
    auto it = local_cache->map.find(key);

    if (it == local_cache->map.end()) {
        bool found_global = false;
        CudaLtGlobalEntry g_entry{};
        {
            std::lock_guard<std::mutex> lk(g_cuda_lt_global_mu);
            const auto git = g_cuda_lt_global_cache.find(key);
            if (git != g_cuda_lt_global_cache.end()) {
                g_entry = git->second;
                found_global = true;
            }
        }
        if (found_global) {
            CudaLtEntry entry{};
            entry.algo = g_entry.algo;
            entry.workspace_bytes = g_entry.workspace_bytes;
            entry.valid = g_entry.valid;
            it = local_cache->map.emplace(key, entry).first;
            if (it->second.valid && !init_cuda_lt_descs(it->second, key)) {
                it->second.valid = false;
                return false;
            }
        } else {
            cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
            const bool capturing = (cudaStreamIsCapturing(st, &cap_status) == cudaSuccess &&
                                    cap_status != cudaStreamCaptureStatusNone);
            if (capturing) return false;   // never sync/create events during CUDA graph capture

            CudaLtEntry tune_descs{};
            if (!init_cuda_lt_descs(tune_descs, key)) return false;
            cublasLtMatmulDesc_t op_desc = &tune_descs.op_desc;
            cublasLtMatrixLayout_t a_desc = &tune_descs.a_desc;
            cublasLtMatrixLayout_t b_desc = &tune_descs.b_desc;
            cublasLtMatrixLayout_t c_desc = &tune_descs.c_desc;

            CudaLtKey bucket_key = key;
            bucket_key.N = cublaslt_t_bucket((int) T);
            if (beta == 0.0f) {
                bool found_bucket = false;
                CudaLtGlobalEntry b_entry{};
                {
                    std::lock_guard<std::mutex> lk(g_cuda_lt_global_mu);
                    const auto bit = g_cuda_lt_bucket_cache.find(bucket_key);
                    if (bit != g_cuda_lt_bucket_cache.end() && bit->second.valid) {
                        b_entry = bit->second;
                        found_bucket = true;
                    }
                }
                if (found_bucket) {
                    cublasLtMatmulHeuristicResult_t chk{};
                    if (cublasLtMatmulAlgoCheck(lt_handle, op_desc, a_desc, b_desc, c_desc, c_desc,
                                                &b_entry.algo, &chk) == CUBLAS_STATUS_SUCCESS &&
                        chk.state == CUBLAS_STATUS_SUCCESS && chk.workspaceSize <= avail_ws) {
                        CudaLtEntry entry = tune_descs;
                        entry.algo = b_entry.algo;
                        entry.workspace_bytes = chk.workspaceSize;
                        entry.valid = true;
                        it = local_cache->map.emplace(key, entry).first;
                        CudaLtGlobalEntry ge{};
                        ge.algo = b_entry.algo;
                        ge.workspace_bytes = chk.workspaceSize;
                        ge.valid = true;
                        {
                            std::lock_guard<std::mutex> lk(g_cuda_lt_global_mu);
                            g_cuda_lt_global_cache.emplace(key, ge);
                        }
                        goto have_cached_lt;
                    }
                }
            }

            cublasLtMatmulPreferenceOpaque_t pref_storage{};
            cublasLtMatmulPreference_t pref = &pref_storage;
            constexpr uint32_t kMinAlign = 16;
            if (cublasLtMatmulPreferenceInit(pref) != CUBLAS_STATUS_SUCCESS ||
                cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws_u64, sizeof(max_ws_u64)) != CUBLAS_STATUS_SUCCESS ||
                cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &kMinAlign, sizeof(kMinAlign)) != CUBLAS_STATUS_SUCCESS ||
                cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &kMinAlign, sizeof(kMinAlign)) != CUBLAS_STATUS_SUCCESS ||
                cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &kMinAlign, sizeof(kMinAlign)) != CUBLAS_STATUS_SUCCESS ||
                cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &kMinAlign, sizeof(kMinAlign)) != CUBLAS_STATUS_SUCCESS) {
                return false;
            }

            constexpr int kMaxHeur = 32;
            cublasLtMatmulHeuristicResult_t heurs[kMaxHeur];
            int n_heur = 0;
            const cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(
                lt_handle, op_desc, a_desc, b_desc, c_desc, c_desc, pref, kMaxHeur, heurs, &n_heur);
            if (hs != CUBLAS_STATUS_SUCCESS || n_heur <= 0) {
                if (beta == 0.0f) {
                    CudaLtEntry entry{};
                    entry.valid = false;
                    local_cache->map.emplace(key, entry);
                    std::lock_guard<std::mutex> lk(g_cuda_lt_global_mu);
                    g_cuda_lt_global_cache.emplace(key, g_entry);
                }
                return false;
            }

            // Safety: never overwrite Y with micro-benchmark runs when beta != 0.0f; just execute heur[0] cleanly.
            if (beta != 0.0f) {
                const float alpha = 1.0f;
                for (int i = 0; i < n_heur; ++i) {
                    if (heurs[i].state == CUBLAS_STATUS_SUCCESS && heurs[i].workspaceSize <= avail_ws) {
                        const cublasStatus_t st_run = cublasLtMatmul(
                            lt_handle, op_desc, &alpha, W, a_desc, X, b_desc, &beta,
                            Y, c_desc, Y, c_desc, &heurs[i].algo, ws_ptr, avail_ws, st);
                        return st_run == CUBLAS_STATUS_SUCCESS;
                    }
                }
                return false;
            }

            constexpr int kMaxCands = 40;
            cublasLtMatmulHeuristicResult_t cands[kMaxCands];
            int n_cands = 0;
            for (int i = 0; i < n_heur && n_cands < kMaxCands; ++i) {
                if (heurs[i].state == CUBLAS_STATUS_SUCCESS && heurs[i].workspaceSize <= avail_ws) {
                    cands[n_cands++] = heurs[i];
                }
            }

            int best_idx = -1;
            float best_ms = 1e30f;
            float heur0_ms = -1.0f;
            int tested = 0;

            const bool skip_bench = (N <= 64 && T < 1024);
            static const bool no_exh_lt = []() {
                const char* v = std::getenv("STRATA_CUBLASLT_NO_EXHAUSTIVE");
                return v && v[0] == '1';
            }();
            if (!skip_bench && !is_f32 && !out_f16 && T >= 1024 && N >= 512 && K >= 512 && !no_exh_lt && n_cands < kMaxCands) {
                int algo_ids[64];
                int n_ids = 0;
                if (cublasLtMatmulAlgoGetIds(lt_handle, CUBLAS_COMPUTE_32F, CUDA_R_32F,
                                             CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, CUDA_R_32F,
                                             64, algo_ids, &n_ids) == CUBLAS_STATUS_SUCCESS) {
                    uint32_t tile_want = CUBLASLT_MATMUL_TILE_256x128;
                    uint32_t stage_want = CUBLASLT_MATMUL_STAGES_32x3;
                    uint32_t sw_want = 0u;
                    for (int ai = 0; ai < n_ids && n_cands < kMaxCands; ++ai) {
                        cublasLtMatmulAlgo_t a;
                        if (cublasLtMatmulAlgoInit(lt_handle, CUBLAS_COMPUTE_32F, CUDA_R_32F,
                                                   CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, CUDA_R_32F,
                                                   algo_ids[ai], &a) != CUBLAS_STATUS_SUCCESS) {
                            continue;
                        }
                        if (cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile_want, sizeof(tile_want)) != CUBLAS_STATUS_SUCCESS ||
                            cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stage_want, sizeof(stage_want)) != CUBLAS_STATUS_SUCCESS ||
                            cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &sw_want, sizeof(sw_want)) != CUBLAS_STATUS_SUCCESS) {
                            continue;
                        }
                        cublasLtMatmulHeuristicResult_t chk{};
                        if (cublasLtMatmulAlgoCheck(lt_handle, op_desc, a_desc, b_desc, c_desc, c_desc, &a, &chk) != CUBLAS_STATUS_SUCCESS ||
                            chk.state != CUBLAS_STATUS_SUCCESS || chk.workspaceSize > avail_ws) {
                            continue;
                        }
                        chk.algo = a;
                        bool dup = false;
                        for (int ci = 0; ci < n_cands; ++ci) {
                            if (std::memcmp(&cands[ci].algo, &a, sizeof(a)) == 0) {
                                dup = true;
                                break;
                            }
                        }
                        if (!dup) {
                            cands[n_cands++] = chk;
                        }
                    }
                }
            }

            if (skip_bench) {
                // Skip micro-timing for tiny N_out (<= 64) at small batch (T < 1024); directly cache and use heur[0].
                if (n_cands > 0) best_idx = 0;
            } else {
                // For small-M (N_out <= 64, e.g. conv_proj M=32, N=8192, K=2304), use 5 warmups + 7 median iters
                // so boost clock and L2 residency settle and tiny-kernel timing jitter cannot flip best_idx.
                const bool small_m = (N <= 64);
                const int n_warmup = small_m ? 5 : 4;
                const int n_runs = small_m ? 7 : 5;
                cudaEvent_t ev[8] = {};
                bool ev_ok = true;
                for (int e = 0; e <= n_runs; ++e) {
                    if (cudaEventCreate(&ev[e]) != cudaSuccess) {
                        ev_ok = false;
                        break;
                    }
                }
                if (!ev_ok) cudaGetLastError();

                const float alpha = 1.0f;
                const float zero = 0.0f;

                for (int i = 0; i < n_cands; ++i) {
                    bool warm_ok = true;
                    for (int w = 0; w < n_warmup; ++w) {
                        const cublasStatus_t w_st = cublasLtMatmul(
                            lt_handle, op_desc, &alpha, W, a_desc, X, b_desc, &zero,
                            Y, c_desc, Y, c_desc, &cands[i].algo, ws_ptr, avail_ws, st);
                        if (w_st != CUBLAS_STATUS_SUCCESS) {
                            (void) cudaGetLastError();
                            warm_ok = false;
                            break;
                        }
                    }
                    if (!warm_ok) continue;
                    if (!ev_ok) {
                        best_idx = i;
                        break;
                    }
                    if (cudaEventRecord(ev[0], st) != cudaSuccess) {
                        (void) cudaGetLastError();
                        continue;
                    }
                    bool run_ok = true;
                    for (int r = 0; r < n_runs; ++r) {
                        const cublasStatus_t r_st = cublasLtMatmul(
                            lt_handle, op_desc, &alpha, W, a_desc, X, b_desc, &zero,
                            Y, c_desc, Y, c_desc, &cands[i].algo, ws_ptr, avail_ws, st);
                        if (r_st != CUBLAS_STATUS_SUCCESS || cudaEventRecord(ev[r + 1], st) != cudaSuccess) {
                            (void) cudaGetLastError();
                            run_ok = false;
                            break;
                        }
                    }
                    if (!run_ok) continue;
                    if (cudaEventSynchronize(ev[n_runs]) != cudaSuccess) {
                        cudaGetLastError();
                        continue;
                    }
                    float dts[7] = {};
                    bool el_ok = true;
                    for (int r = 0; r < n_runs; ++r) {
                        if (cudaEventElapsedTime(&dts[r], ev[r], ev[r + 1]) != cudaSuccess) {
                            cudaGetLastError();
                            el_ok = false;
                            break;
                        }
                    }
                    if (!el_ok) continue;
                    std::sort(dts, dts + n_runs);
                    const float ms = dts[n_runs / 2];
                    if (tested == 0) heur0_ms = ms;
                    ++tested;
                    const float threshold = small_m ? (best_ms * 0.995f) : best_ms;
                    if (best_idx < 0 || ms < threshold) {
                        best_ms = ms;
                        best_idx = i;
                    }
                }
                for (int e = 0; e <= n_runs; ++e) {
                    if (ev[e]) (void) cudaEventDestroy(ev[e]);
                }
            }

            if (best_idx >= 0) {
                g_entry.valid = true;
                g_entry.algo = cands[best_idx].algo;
                g_entry.workspace_bytes = cands[best_idx].workspaceSize;
                if (cublaslt_tune_verbose()) {
                    const char* const tag = is_f32 ? "f32" : (out_f16 ? "f16_f16out" : "f16");
                    if (skip_bench) {
                        std::fprintf(stderr,
                                     "prefill gemm: cublasLt fast-pick %s M=%d N=%d K=%d ldc=%d ws=%zu: winner=#%d (M<=64, N<1024 skip bench, n_heur=%d)\n",
                                     tag, (int) N, (int) T, (int) K, (int) ldy,
                                     g_entry.workspace_bytes, best_idx, n_heur);
                    } else {
                        std::fprintf(stderr,
                                     "prefill gemm: cublasLt tuned %s M=%d N=%d K=%d ldc=%d ws=%zu: winner=#%d %.3f ms (heur0=%.3f ms, tested=%d/%d)\n",
                                     tag, (int) N, (int) T, (int) K, (int) ldy,
                                     g_entry.workspace_bytes, best_idx, best_ms, heur0_ms, tested, n_cands);
                    }
                }
            } else {
                g_entry.valid = false;
            }
            CudaLtEntry entry{};
            entry.algo = g_entry.algo;
            entry.workspace_bytes = g_entry.workspace_bytes;
            entry.valid = g_entry.valid;
            it = local_cache->map.emplace(key, entry).first;
            if (it->second.valid && !init_cuda_lt_descs(it->second, key)) {
                it->second.valid = false;
            }
            {
                std::lock_guard<std::mutex> lk(g_cuda_lt_global_mu);
                g_cuda_lt_global_cache.emplace(key, g_entry);
                if (g_entry.valid) {
                    g_cuda_lt_bucket_cache.emplace(bucket_key, g_entry);
                }
            }
        }
    }

have_cached_lt:
    CudaLtEntry& chosen = it->second;
    if (!chosen.valid || chosen.workspace_bytes > avail_ws) return false;
    if (!chosen.descs_ready && !init_cuda_lt_descs(chosen, key)) {
        chosen.valid = false;
        return false;
    }

    // Always execute the winning algorithm explicitly so Y is guaranteed exact from chosen.algo.
    const float alpha = 1.0f;
    const cublasStatus_t run_st = cublasLtMatmul(
        lt_handle, &chosen.op_desc, &alpha, W, &chosen.a_desc, X, &chosen.b_desc, &beta,
        Y, &chosen.c_desc, Y, &chosen.c_desc, &chosen.algo, ws_ptr, avail_ws, st);
    if (run_st == CUBLAS_STATUS_SUCCESS) return true;
    if (beta != 0.0f) {
        ck(run_st, is_f32 ? "cublasLtMatmul f32" : (out_f16 ? "cublasLtMatmul f16_f16out" : "cublasLtMatmul f16"));
    }
    it->second.valid = false;
    return false;
#else
    (void) X; (void) W; (void) Y; (void) T; (void) N; (void) K; (void) ldy; (void) beta; (void) is_f32; (void) out_f16;
    return false;
#endif
}

void Gemm::f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
#ifdef STRATA_USE_HIP
    // #313 (opt-in STRATA_WMMA_GEMM=1): RDNA3 / RDNA3.5 WMMA dense GEMM, before hipBLASLt (falls through when false)
    static const bool wmma_on = [] { const char* v = std::getenv("STRATA_WMMA_GEMM"); return v && v[0] != 0 && v[0] != '0'; }();
    static const bool pf_on = [] { const char* v = std::getenv("STRATA_PF_GEMM"); return v && v[0] == '1'; }();
    static const int64_t pf_min_t = [] { const char* v = std::getenv("STRATA_PF_SWITCH_MIN_T"); return v ? (int64_t) std::atoll(v) : (int64_t) 0; }();
    if (pf_on && T >= pf_min_t && strata_pf_gemm_f16(X, W, Y, T, N, K, ldy, beta, stream_)) return;   // S23: opt-in
    if (wmma_on && T >= 16 && (beta == 0.0f || beta == 1.0f) &&
        strata_wmma_gemm_f16(X, W, Y, T, N, K, ldy, beta, stream_)) {
        return;
    }
#endif
#if defined(__HIPCC__)
    if (f16_io_ && beta == 0.0f) { f16_inplace(X, W, Y, T, N, K, ldy); return; }
#endif
    const float alpha = 1.0f;
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (try_hipblaslt(hipblaslt_state_, strata::prefill::hipblaslt::InputType::f16, X, W, Y, T, N, K, ldy,
                      beta, stream_)) {
        STRATA_ABSORB_HIPBLAS_STICKY("hipBLASLt f16");
        return;
    }
#endif
#if !defined(__HIPCC__) && defined(STRATA_EXPERIMENTAL_SM60)
    // Pascal, STRATA_DENSE_F32=1: cublasSgemm on FP32 copies (dense_f32_on above: where it is bitwise the FP16 call)
    if (const int64_t min_t = beta == 0.0f ? dense_f32_min_t(N, K) : 0; min_t > 0 && T >= min_t && dense_f32_on()) {
        const int64_t n_sl = (T * K + kXSliceElems - 1) / kXSliceElems;   // even slices of at most 64 MiB as fp32
        const int64_t x_rows = (T + n_sl - 1) / n_sl;
        // grow() frees a buffer an earlier GEMM on this stream may still read
        if (tc_w_elems_ < 2 * N * K || tc_x_elems_ < 2 * x_rows * K) cudaStreamSynchronize((cudaStream_t) stream_);
        if (grow(tc_w_, tc_w_elems_, 2 * N * K) && grow(tc_x_, tc_x_elems_, 2 * x_rows * K)) {
            static bool said = false;
            if (!said) {
                said = true;
                std::fprintf(stderr, "prefill gemm: the dense projections' FP16 GEMMs run as cublasSgemm on FP32 copies "
                                     "(STRATA_DENSE_F32=1; gemm_f32_parity checks the bits against this cuBLAS)\n");
            }
            float* const wf = reinterpret_cast<float*>(tc_w_);
            float* const xf = reinterpret_cast<float*>(tc_x_);
            f16_to_f32(W, wf, N * K, (cudaStream_t) stream_);
            for (int64_t t0 = 0; t0 < T; t0 += x_rows) {
                const int64_t n = std::min<int64_t>(x_rows, T - t0);
                f16_to_f32(X + t0 * K, xf, n * K, (cudaStream_t) stream_);
                ck(cublasSgemm((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) n, (int) K, &alpha, wf, (int) K,
                               xf, (int) K, &beta, Y + t0 * ldy, (int) ldy),
                   "cublasSgemm (FP16 operands, STRATA_DENSE_F32)");
            }
            return;
        }
        cudaGetLastError();   // no room for the copies: the FP16 call below
    }
#endif
#if defined(__HIPCC__)
    if (rdna2_sgemm(X, W, Y, T, N, K, ldy, beta, false)) return;
#endif
#if !defined(__HIPCC__)
#if !defined(STRATA_HIP_GFX906)
    if (try_cublaslt_tune(X, W, Y, T, N, K, ldy, beta, /*is_f32=*/false)) return;
#endif
    const auto f16_gemm = [&](cublasGemmAlgo_t algo) {
        return cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha,
                            W, CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                            CUBLAS_COMPUTE_32F, algo);
    };
    // Issue #1650: with a second CUDA device initialized in the process, cuBLAS 12 answers the default algorithm of a
    // few shapes - narrow ones, and the set moves with K - with CUBLAS_STATUS_INTERNAL_ERROR; a fixed algorithm keeps
    // a kernel that state does not break.  Only the shapes that fail are moved (the failing call itself is harmless);
    // the wide shapes, for which the fixed algorithms are markedly slower, stay on the default.
    const int64_t shape = ((int64_t) K << 32) | (uint32_t) N;
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
    for (int i = 0; i < f16_algo_n_; ++i)
        if (f16_algo_shape_[i] == shape) { algo = CUBLAS_GEMM_ALGO2; break; }
    cublasStatus_t st = f16_gemm(algo);
    if (st == CUBLAS_STATUS_INTERNAL_ERROR && algo == CUBLAS_GEMM_DEFAULT) {
        if (f16_algo_n_ < kF16AlgoSlots) f16_algo_shape_[f16_algo_n_++] = shape;
        st = f16_gemm(CUBLAS_GEMM_ALGO2);
    }
    ck(st, "cublasGemmEx f16");
#else
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx f16");
#endif
    STRATA_ABSORB_HIPBLAS_STICKY("cublasGemmEx f16");
}

void Gemm::f16_f16out(const uint16_t* X, const uint16_t* W, uint16_t* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                      float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    const float alpha = 1.0f;
#if !defined(__HIPCC__)
#if !defined(STRATA_HIP_GFX906)
    if (try_cublaslt_tune(X, W, Y, T, N, K, ldy, beta, /*is_f32=*/false, /*out_f16=*/true)) return;
#endif
    const auto f16o_gemm = [&](cublasGemmAlgo_t algo) {
        return cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha,
                            W, CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) K, &beta, Y, CUDA_R_16F, (int) ldy,
                            CUBLAS_COMPUTE_32F, algo);
    };
    cublasStatus_t st = f16o_gemm(CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st == CUBLAS_STATUS_INTERNAL_ERROR) {
        st = f16o_gemm(CUBLAS_GEMM_ALGO2);
    }
    ck(st, "cublasGemmEx f16_f16out");
#else
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) K, &beta, Y, CUDA_R_16F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx f16_f16out");
#endif
    STRATA_ABSORB_HIPBLAS_STICKY("cublasGemmEx f16_f16out");
}

void Gemm::sgemm(const float* X, const float* W, float* Y, int64_t T, int64_t N, int64_t K) {
    if (T <= 0 || N <= 0 || K <= 0) return;
    const float alpha = 1.0f, beta = 0.0f;
#if defined(__HIPCC__)
    ck(hipblasSgemm((hipblasHandle_t) handle_, HIPBLAS_OP_T, HIPBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    (int) K, X, (int) K, &beta, Y, (int) N),
       "hipblasSgemm");
    STRATA_ABSORB_HIPBLAS_STICKY("hipblasSgemm");
#else
#if !defined(STRATA_HIP_GFX906)
    if (try_cublaslt_tune(X, W, Y, T, N, K, N, 0.0f, /*is_f32=*/true)) return;
#endif
    ck(cublasSgemm((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                   (int) K, X, (int) K, &beta, Y, (int) N),
       "cublasSgemm");
#endif
}

#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)
bool Gemm::native_mmq(const uint16_t* X, int type, const void* W, float* Y, int64_t T, int64_t N, int64_t K,
                      int64_t ldy) {
    namespace mmq = strata::prefill::mmq;
    constexpr int64_t kRows = 1024, kMaxK = 8192;
    static const bool enabled = [] {
        const char* e = std::getenv("STRATA_DENSE_MMQ");
        return e && e[0] == '1';
    }();
    if (!enabled || mmq_failed_ || !mmq::built() || !mmq::fits(type, N)) return false;
    // K must be a multiple of 256: llama.cpp's MMQ loads the weights in 256-value K chunks, and the
    // chunk past a partial row reads past the row (the next row's bytes - or, for the last weight
    // row, bytes past the tensor, which no caller guarantees to be zeros). The MoE path is safe
    // because its gather buffers carry a zeroed tail; a dense tensor is passed as-is, so a K that
    // is not a full chunk multiple (e.g. the 640-value shared-expert down: 20 blocks of 32) stays on
    // the dequantize + cuBLAS path.
    if (K > kMaxK || K % 256 != 0) return false;
    int64_t R = scratch_elems_ / (2 * K);   // FP32 activations for a chunk fit in the FP16 dequant scratch
    if (R > kRows) R = kRows;
    if (R > T) R = T;
    if (R < 1) return false;
    if (ldy <= 0) ldy = N;

    const auto up = [](size_t v) { return (v + 255) & ~(size_t) 255; };
    // The row table is read past its end by llama.cpp's MMQ (mul_mat_q: a whole J-column tile from the start of each tile, J up to 128; the stream-k fixup:
    // up to roundup(max_rows, J) from the expert's first row): kRows + the margin of mmq_ids_margin.hpp (max_rows <= kRows), the identity written over all of it.
    // Exact, with only the 256 bytes of the bounds after it, a tile of 112 columns on a full chunk read 384 bytes past the table, 128 past the allocation.
    const size_t ids_rows = (size_t) mmq::mmq_ids_table_rows((uint64_t) kRows, (uint64_t) kRows);
    const size_t ident_off = up(mmq::q8_bytes(kRows, kMaxK));
    const size_t bounds_off = ident_off + up(ids_rows * 4);
    if (!mmq_buf_) {
        if (cudaMalloc(&mmq_buf_, bounds_off + 256) != cudaSuccess) {
            cudaGetLastError();
            mmq_failed_ = true;
            return false;
        }
        mmq_ctx_ = new mmq::Context();
        mmq::iota((int32_t*) ((uint8_t*) mmq_buf_ + ident_off), (int64_t) ids_rows, stream_);
        std::fprintf(stderr, "prefill gemm: dense MMQ on (STRATA_DENSE_MMQ=1)\n");
    }
    void* xq = mmq_buf_;
    const int32_t* ident = (const int32_t*) ((uint8_t*) mmq_buf_ + ident_off);
    int32_t* bounds = (int32_t*) ((uint8_t*) mmq_buf_ + bounds_off);
    float* xf = (float*) scratch_;
    auto* ctx = static_cast<mmq::Context*>(mmq_ctx_);

    for (int64_t r0 = 0; r0 < T; r0 += R) {
        const int64_t rows = (T - r0 < R) ? T - r0 : R;
        mmq::f16_to_f32(X + r0 * K, xf, rows * K, stream_);
        mmq::quantize(xf, nullptr, xq, type, K, K, rows, stream_);
        mmq::set_bounds(bounds, (int32_t) rows, stream_);
        mmq::Product p;
        p.w = W;
        p.type = type;
        p.w_rows = N;
        p.w_cols = K;
        p.expert_bytes = mmq::matrix_bytes(type, N, K);
        p.n = 1;
        p.xq = xq;
        p.bounds = bounds;
        p.ids = ident;
        p.total_rows = rows;
        p.max_rows = rows;
        p.dst = Y + r0 * ldy;
        p.ld_dst = ldy;
        ctx->run(p, stream_);
    }
    return true;
}
#endif
void Gemm::f16_inplace(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy) {
#if defined(__HIPCC__)
    const float one = 1.0f, zero = 0.0f;
    bool done = false;
#if defined(STRATA_ROCBLAS_AVAILABLE)
    done = try_rocblas_f16o(rocblas_state_, X, W, Y, T, N, K, ldy);   // the table's kernel for this shape, if any
#endif
    if (!done) {
        ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &one, W,
                        CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) K, &zero, Y, CUDA_R_16F, (int) (2 * ldy),
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
           "cublasGemmEx f16 out");
    }
    static const bool dbg_nan = std::getenv("STRATA_DBG_NAN") != nullptr;
    widen_rows_f16<<<(unsigned) T, 256, 0, (cudaStream_t) stream_>>>(Y, N, ldy, dbg_nan ? 1 : 0);
    if (dbg_nan) {   // debug only: a sync per GEMM
        unsigned long long bad = 0, zero_count = 0;
        (void) hipStreamSynchronize((hipStream_t) stream_);
        if (hipMemcpyFromSymbol(&bad, HIP_SYMBOL(g_f16_nonfinite), sizeof bad) == hipSuccess && bad != 0) {
            std::fprintf(stderr, "strata prefill: STRATA_DBG_NAN: %llu non-finite FP16 GEMM outputs (T=%lld N=%lld K=%lld; FP16 ends at 65504)%s",
                         bad, (long long) T, (long long) N, (long long) K, "\n");
            (void) hipMemcpyToSymbol(HIP_SYMBOL(g_f16_nonfinite), &zero_count, sizeof zero_count);
        }
    }
#else
    (void) X; (void) W; (void) Y; (void) T; (void) N; (void) K; (void) ldy;
#endif
}

void Gemm::native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                  int64_t ldy, float beta, int64_t ldx) {
    uint16_t* const cur_scratch = temp_scratch_ ? static_cast<uint16_t*>(temp_scratch_) : scratch_;
    const int64_t cur_scratch_elems = temp_scratch_ ? static_cast<int64_t>(temp_scratch_elems_) : scratch_elems_;
#ifdef STRATA_USE_HIP
    // S23 (opt-in STRATA_PF_PAD=1, with STRATA_PF_GEMM=1 on chunks of STRATA_PF_SWITCH_MIN_T+ tokens): the weight
    // dequantized into the scratch with row stride K + 64 halves, so no 4 KB-multiple stride camps on the memory
    // channels (gemm_probe8: K 6144 19 -> 29 TFLOPS from W alone, 35 with X padded too).  Bitwise the same products.
    static const bool pad_on = [] { const char* v = std::getenv("STRATA_PF_PAD"); return v && v[0] == '1'; }();
    static const bool pf_on = [] { const char* v = std::getenv("STRATA_PF_GEMM"); return v && v[0] == '1'; }();
    static const int64_t pf_min_t = [] { const char* v = std::getenv("STRATA_PF_SWITCH_MIN_T"); return v ? (int64_t) std::atoll(v) : (int64_t) 0; }();
    // (strata_pf_padded_gemm_available(): the CURRENT device runs the padded gfx11 product.  On a gfx1151 + gfx1201 split the process-wide switch
    // is on for both stages; the gfx12 stage would dequantize the weight padded, find no product, and do it again unpadded below.)
    if (pad_on && pf_on && T >= pf_min_t && T >= 64 && N >= 512 && K % 32 == 0 && N * (K + 64) <= cur_scratch_elems &&
        strata_pf_padded_gemm_available()) {
        const int64_t ldw = K + 64;
        if (strata::kernels::dequant_f16_ld(ggml_type, W_blocks, 0, N, K, ldw, cur_scratch, stream_) &&
            strata_pf_gemm_f16_ld(X, ldx > 0 ? ldx : K, cur_scratch, ldw, Y, T, N, K, ldy, beta, stream_)) {
            return;
        }
    }
#endif
    if (ldx > 0 && ldx != K) {
#ifdef STRATA_USE_HIP
        // X arrived padded but the weight cannot be (an i-quant without a strided dequant, or the scratch is too
        // small for N x (K + 64)): the weight unpadded, X read at its stride - the same products, the same bits
        const int64_t rows = cur_scratch_elems / K;
        bool ok = rows > 0 && strata_pf_padded_gemm_available();
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; ok && r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, cur_scratch, stream_);
            ok = strata_pf_gemm_f16_ld(X, ldx, cur_scratch, K, Y + r0, T, n, K, ldy, beta, stream_);
        }
        if (ok) return;
#endif
        // Nothing took a padded X: the padded product is the gfx11 matrix cores' (STRATA_PF_PAD asks
        // strata_pf_padded_gemm_available() before padding now, so this is a stale producer or another card's
        // activation).  Copy the rows unpadded and run the ordinary paths - the same products, the same bits - rather
        // than ending the engine.
        const uint16_t* Xu = unpad_x(X, T, K, ldx);
        if (Xu != nullptr) {
            native(Xu, ggml_type, W_blocks, Y, T, N, K, ldy, beta, 0);
            return;
        }
        std::fprintf(stderr, "prefill gemm: a padded X (ldx %lld, K %lld) has no path on this card and could not be "
                             "copied unpadded (no device memory)\n", (long long) ldx, (long long) K);
        std::exit(1);
    }
#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)
    if (beta == 0.0f && (ldx == 0 || ldx == K) && native_mmq(X, ggml_type, W_blocks, Y, T, N, K, ldy)) return;
#endif
#if !defined(__HIPCC__) && !defined(STRATA_HIP_GFX906)
    // Persistent L2-swizzled fused Q8_0 x FP16 -> FP32-accum Tensor Core GEMM (sm_80+):
    // skips dequant_q8_0_kernel and cublasLt completely with bit-exact Q8_0 dequant and FP32 accumulation.
    if (ggml_type == 8 && beta == 0.0f && (ldx == 0 || ldx == K) &&
        X != nullptr && W_blocks != nullptr && Y != nullptr &&
        T >= 256 && N > 0 && (N & 127) == 0 && K > 0 && (K & 31) == 0 &&
        T <= INT_MAX && N <= INT_MAX && K <= INT_MAX &&
        !fused_q8_gemm_disabled() && current_cc() >= 80) {
        const int64_t out_ldy = (ldy > 0) ? ldy : N;
        if (out_ldy >= N && out_ldy <= INT_MAX && (out_ldy & 1) == 0 &&
            ((((uintptr_t) X) & 15u) == 0u) && ((((uintptr_t) W_blocks) & 1u) == 0u) &&
            ((((uintptr_t) Y) & 7u) == 0u)) {
            launch_gemm_q8_0_f16_f32_single(W_blocks, reinterpret_cast<const __half*>(X), Y,
                                            (int) T, (int) N, (int) K, (int) out_ldy, (cudaStream_t) stream_);
            return;
        }
    }
#endif
#ifndef STRATA_USE_HIP
    // The FP16 weight kept on the device after its first dequant.  NOT bit-exact against the uncached sliced form on every
    // card (see the default below): off unless STRATA_PF_W16_CACHE_MIB is set.  When on it is sized by the card, never by the user.  It takes only VRAM nobody else holds: a weight is
    // cached while the card keeps kW16AutoReserve free afterwards (the MTP prompt capture, the batch windows and a long
    // prompt's buffers are allocated later, out of that room; a replica with 309 MiB free ran out of memory at 14K tokens),
    // so a card whose expert cache fills it caches nothing and a card with room (all experts resident) caches what the
    // prompt path uses (~6 GB for Qwen3.8).  STRATA_PF_W16_CACHE_MIB=N is an explicit budget (2 GiB kept free, as before),
    // =0 turns it off.
    if (w16_budget_ < 0) {
        const char* v = std::getenv("STRATA_PF_W16_CACHE_MIB");
        if (v != nullptr && std::strcmp(v, "auto") == 0) {
            // STRATA_PF_W16_CACHE_MIB=auto (tester switch, round0212 item 7): the automatic sizing - weights are cached while the card
            // keeps kW16AutoReserve free afterwards, so a card whose expert cache fills it caches nothing.  Not bit-exact (see below),
            // so it is not the default until the owner says yes.
            w16_budget_ = 16ll << 30;
            w16_reserve_ = kW16AutoReserve;
        } else if (v != nullptr) {
            w16_budget_ = (int64_t) std::atoll(v) * (1ll << 20);
            w16_reserve_ = 2ll << 30;
        } else {
            // INT3 turned the default off (11-01) because gemm_dequant_side_parity failed with it on sm_86 / sm_120 / sm_121.  That test builds its Gemm
            // with a 10.5 M-element scratch, smaller than four of its weights, so uncached those take the sliced path (several GEMMs) and cached they
            // take one full-N GEMM: not the engine.  The engine's scratch is GEMM_SCRATCH = 32 M elements and every dense projection of Qwen3.8
            // (largest attn_q, 31.5 M) fits it, so cached and uncached are the same GEMM on the same FP16 values (gemm_w16_parity: the engine's scratch,
            // shapes and types, bitwise).  Round0212 item 7: on when the engine says so (set_w16_auto_default: Linux CUDA with batch slots), else off
            // as in v0.2.9.  It takes only VRAM beyond kW16AutoReserve, so a card whose expert cache fills it caches nothing.  =0 turns it off.
            w16_budget_ = w16_auto_default_().load() ? (16ll << 30) : 0;
            w16_reserve_ = kW16AutoReserve;
        }
    }
    if (w16_budget_ > 0 && !side_scratch_) {
        const int64_t bytes = N * K * 2;
        uint16_t* cw = nullptr;
        for (const W16Entry& e : w16_)
            if (e.p == W_blocks && e.type == ggml_type && e.n == N && e.k == K) { cw = e.d; break; }
        if (cw == nullptr && w16_used_ + bytes <= w16_budget_) {
            size_t fb = 0, tb = 0;
            if (cudaMemGetInfo(&fb, &tb) == cudaSuccess && fb > (size_t) bytes + (size_t) w16_reserve_ &&
                cudaMalloc((void**) &cw, (size_t) bytes) == cudaSuccess) {
                strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, cw, stream_);
                w16_.push_back({W_blocks, ggml_type, N, K, cw});
                w16_used_ += bytes;
            } else {
                (void) cudaGetLastError();
                cw = nullptr;
                if (w16_budget_ > 0 && !w16_logged_) {
                    w16_logged_ = true;
                    std::fprintf(stderr, "strata prefill: FP16 weight cache %lld MiB (the card keeps %lld MiB free; STRATA_PF_W16_CACHE_MIB=0 turns it off)\n",
                                 (long long) (w16_used_ >> 20), (long long) (w16_reserve_ >> 20));
                }
                w16_budget_ = w16_used_ > 0 ? w16_used_ : 0;   // out of room: stop trying
            }
        }
        if (cw != nullptr) {
            f16(X, cw, Y, T, N, K, ldy, beta);
            return;
        }
    }
#endif
    if (N * K > cur_scratch_elems) {
        // Too large for the scratch at once: in row slices.
        const int64_t rows = cur_scratch_elems / K;
        if (rows <= 0) { std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long) K); std::exit(1); }
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, cur_scratch, stream_);
            f16(X, cur_scratch, Y + r0, T, n, K, ldy, beta);
        }
        // the side stream's next dequant into scratch_ waits for these slices' reads and writes of it
        if (side_scratch_ && !temp_scratch_) (void) cudaEventRecord((cudaEvent_t) side_free_[0], (cudaStream_t) stream_);
        return;
    }
    if (side_scratch_ && !temp_scratch_) {
        // STRATA_DEQUANT_SIDE: the dequant on the side stream, into the scratch the compute stream read last but one.
        // free[cur] -> dequant -> ready[cur] -> GEMM -> free[cur] orders the dequant after the GEMM that last read
        // this scratch and the GEMM after the dequant that filled it, so every read and write of a scratch is where
        // the serial form had it - the same kernels, the same bits - and the dequant runs while the kernels queued
        // ahead of this GEMM do.
        uint16_t* const s = side_cur_ == 0 ? scratch_ : side_scratch_;
        (void) cudaStreamWaitEvent((cudaStream_t) side_stream_, (cudaEvent_t) side_free_[side_cur_], 0);
        strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, s, side_stream_);
        (void) cudaEventRecord((cudaEvent_t) side_ready_[side_cur_], (cudaStream_t) side_stream_);
        (void) cudaStreamWaitEvent((cudaStream_t) stream_, (cudaEvent_t) side_ready_[side_cur_], 0);
        f16(X, s, Y, T, N, K, ldy, beta);
        (void) cudaEventRecord((cudaEvent_t) side_free_[side_cur_], (cudaStream_t) stream_);
        side_cur_ ^= 1;
        return;
    }
    strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, cur_scratch, stream_);
    f16(X, cur_scratch, Y, T, N, K, ldy, beta);
}

void Gemm::dequant_to_f16(int ggml_type, const void* W_blocks, uint16_t* dst, int64_t N, int64_t K) {
    if (N <= 0 || K <= 0 || W_blocks == nullptr || dst == nullptr) return;
    strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, dst, stream_);
}

void Gemm::native_f16out(const uint16_t* X, int ggml_type, const void* W_blocks, uint16_t* Y, int64_t T, int64_t N,
                         int64_t K, int64_t ldy, float beta, int64_t ldx) {
    if (T <= 0 || N <= 0) return;
    if (ldx > 0 && ldx != K) {
        const uint16_t* Xu = unpad_x(X, T, K, ldx);
        if (Xu != nullptr) {
            native_f16out(Xu, ggml_type, W_blocks, Y, T, N, K, ldy, beta, 0);
            return;
        }
        std::fprintf(stderr, "prefill gemm: a padded X (ldx %lld, K %lld) has no f16out path on this card and could not be "
                             "copied unpadded (no device memory)\n", (long long) ldx, (long long) K);
        std::exit(1);
    }
#if !defined(__HIPCC__) && !defined(STRATA_HIP_GFX906)
    // Persistent L2-swizzled fused Q8_0 x FP16 -> FP32-accum -> FP16-out Tensor Core GEMM (sm_80+):
    // skips dequant_q8_0_kernel and cublasLt completely with bit-exact Q8_0 dequant and FP32 accumulation.
    if (ggml_type == 8 && beta == 0.0f && (ldx == 0 || ldx == K) &&
        X != nullptr && W_blocks != nullptr && Y != nullptr &&
        T >= 256 && N > 0 && (N & 127) == 0 && K > 0 && (K & 31) == 0 &&
        T <= INT_MAX && N <= INT_MAX && K <= INT_MAX &&
        fused_q8_f16out_enabled() && current_cc() >= 80) {
        const int64_t out_ldy = (ldy > 0) ? ldy : N;
        if (out_ldy >= N && out_ldy <= INT_MAX && (out_ldy & 1) == 0 &&
            ((((uintptr_t) X) & 15u) == 0u) && ((((uintptr_t) W_blocks) & 1u) == 0u) &&
            ((((uintptr_t) Y) & 3u) == 0u)) {
            launch_gemm_q8_0_f16_f32_single_f16out(W_blocks, reinterpret_cast<const __half*>(X),
                                                   reinterpret_cast<__half*>(Y),
                                                   (int) T, (int) N, (int) K, (int) out_ldy,
                                                   (cudaStream_t) stream_);
            return;
        }
    }
#endif
    uint16_t* const cur_scratch = temp_scratch_ ? static_cast<uint16_t*>(temp_scratch_) : scratch_;
    const int64_t cur_scratch_elems = temp_scratch_ ? static_cast<int64_t>(temp_scratch_elems_) : scratch_elems_;
    if (N * K > cur_scratch_elems) {
        // Too large for the scratch at once: in row slices.
        const int64_t rows = cur_scratch_elems / K;
        if (rows <= 0) { std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long) K); std::exit(1); }
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, cur_scratch, stream_);
            f16_f16out(X, cur_scratch, Y + r0, T, n, K, ldy, beta);
        }
        if (side_scratch_ && !temp_scratch_) (void) cudaEventRecord((cudaEvent_t) side_free_[0], (cudaStream_t) stream_);
        return;
    }
    if (side_scratch_ && !temp_scratch_) {
        uint16_t* const s = side_cur_ == 0 ? scratch_ : side_scratch_;
        (void) cudaStreamWaitEvent((cudaStream_t) side_stream_, (cudaEvent_t) side_free_[side_cur_], 0);
        strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, s, side_stream_);
        (void) cudaEventRecord((cudaEvent_t) side_ready_[side_cur_], (cudaStream_t) side_stream_);
        (void) cudaStreamWaitEvent((cudaStream_t) stream_, (cudaEvent_t) side_ready_[side_cur_], 0);
        f16_f16out(X, s, Y, T, N, K, ldy, beta);
        (void) cudaEventRecord((cudaEvent_t) side_free_[side_cur_], (cudaStream_t) stream_);
        side_cur_ ^= 1;
        return;
    }
    dequant_to_f16(ggml_type, W_blocks, cur_scratch, N, K);
    f16_f16out(X, cur_scratch, Y, T, N, K, ldy, beta);
}

void Gemm::native_q8_0_f16_multi(const void* w_q8_0, const void* x_f16, const MultiOut5& outs, int T, int M_total, int K,
                                 void* st) {
    if (T <= 0 || M_total <= 0 || K <= 0 || outs.n_outs <= 0) return;
#if !defined(__HIPCC__) && !defined(STRATA_HIP_GFX906)
    if (w_q8_0 != nullptr && x_f16 != nullptr && outs.n_outs <= 5 &&
        T >= 32 && (M_total & 127) == 0 && (K & 31) == 0 &&
        !fused_q8_gemm_disabled() && current_cc() >= 80 &&
        ((((uintptr_t) x_f16) & 15u) == 0u) && ((((uintptr_t) w_q8_0) & 1u) == 0u)) {
        bool valid_outs = (outs.ends[outs.n_outs - 1] == M_total);
        int prev_end = 0;
        for (int s = 0; s < outs.n_outs && valid_outs; ++s) {
            const int end = outs.ends[s];
            if (end <= prev_end || (end & 127) != 0 || outs.ptrs[s] == nullptr ||
                ((((uintptr_t) outs.ptrs[s]) & 7u) != 0u)) {
                valid_outs = false;
            }
            prev_end = end;
        }
        if (valid_outs) {
            cudaStream_t cst = st ? (cudaStream_t) st : (cudaStream_t) stream_;
            launch_gemm_q8_0_f16_f32_multi(w_q8_0, static_cast<const __half*>(x_f16), outs,
                                           T, M_total, K, cst);
            return;
        }
    }
#endif
    int start = 0;
    const int n_outs = outs.n_outs < 5 ? outs.n_outs : 5;
    for (int s = 0; s < n_outs; ++s) {
        const int end = outs.ends[s];
        const int n_s = end - start;
        if (n_s > 0 && outs.ptrs[s] != nullptr) {
            const uint8_t* w_slice = static_cast<const uint8_t*>(w_q8_0) +
                                     (size_t) start * (size_t) (K / 32) * 34u;
            native(static_cast<const uint16_t*>(x_f16), 8, w_slice, outs.ptrs[s], T, n_s, K);
        }
        start = end;
    }
}

}  // namespace strata::prefill
