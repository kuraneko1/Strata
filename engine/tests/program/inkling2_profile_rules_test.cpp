// tests/program/inkling2_profile_rules_test.cpp - CPU-only unit tests for strata::inkling2::resolve_inkling2_profile.
// Locks down V100 (sm_70 32GB), RTX 4070 (sm_89 12GB @ 65K/131K/262K), unmeasured Generic GPUs, and explicit CLI overrides.
#include "strata/inkling2/profile_rules.hpp"

#include <cstdio>

namespace {
int fails = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++fails;
    }
}

constexpr size_t GiB = (size_t) 1 << 30;
constexpr size_t MiB = (size_t) 1 << 20;

// Inkling-2-Mini-UD-Q4_K_M exact expert footprint: 12 core layers = 6.45 GiB, all 24 layers = 13.76 GiB.
constexpr size_t kTier1Bytes = (size_t) (6.45 * (double) GiB);
constexpr size_t kAllExpBytes = (size_t) (13.76 * (double) GiB);

strata::inkling2::ProfileInput v100_32g(int max_ctx, bool draft = true) {
    strata::inkling2::ProfileInput in;
    in.cc_major = 7;
    in.cc_minor = 0;
    in.sm_count = 80;
    in.vram_total_bytes = 32 * GiB;
    in.vram_free_bytes = draft ? (size_t) (29.1 * (double) GiB) : (size_t) (29.8 * (double) GiB);
    in.max_ctx = max_ctx;
    in.kv_q8 = true;
    in.has_draft = draft;
    in.tier1_experts_bytes = kTier1Bytes;
    in.all_experts_bytes = kAllExpBytes;
    return in;
}

strata::inkling2::ProfileInput rtx4070_12g(int max_ctx, bool draft = false) {
    strata::inkling2::ProfileInput in;
    in.cc_major = 8;
    in.cc_minor = 9;
    in.sm_count = 46;
    in.vram_total_bytes = 12 * GiB;
    // Measured free VRAM on RTX 4070 after dense model (~2.0 GiB) + desktop (~0.2 GiB) load: ~9.85 GiB.
    in.vram_free_bytes = (size_t) (9.85 * (double) GiB);
    in.max_ctx = max_ctx;
    in.kv_q8 = true;
    in.has_draft = draft;
    in.tier1_experts_bytes = kTier1Bytes;
    in.all_experts_bytes = kAllExpBytes;
    return in;
}
}  // namespace

int main() {
    using namespace strata::inkling2;

    // 1. V100 32GB @ 819,200 ctx (DSpark ON): measured real V100 (29.7 GiB free) promotes to chunk=2048 (+21-24% PP,
    //    want_sp=4, ~558 MiB free after placement), while tighter headroom (29.1 GiB) falls back safely to chunk=1024 / want_sp=6!
    {
        auto in = v100_32g(819200, true);
        in.vram_free_bytes = (size_t) (29.7 * (double) GiB);
        const auto p = resolve_inkling2_profile(in);
        std::printf("%s\n", format_profile_line(in, p).c_str());
        check(p.arch.id == ArchId::Sm70Volta, "V100 selects Sm70Volta arch");
        check(p.regime == VramRegime::FullVram, "V100 819K selects FullVram");
        check(p.chunk == 2048, "V100 819K measured 29.7 GiB promotes to chunk=2048");
        check(p.max_adaptive_chunk == p.chunk, "V100 819K keeps max_adaptive_chunk == chunk");
        check(p.full_kv_init == 819200, "V100 819K allocates full 819K KV upfront");
        check(p.want_sp == 4, "V100 819K chunk=2048 uses want_sp=4");
        check(!p.use_d_hits, "V100 FullVram skips d_hits atomics");
        check(p.pp_stage_cnt_pref == 0, "V100 FullVram allocates no pp_stage");
        check(p.force_bf16_x_f16, "V100 enables force_bf16_x_f16");

        const auto in_tight = v100_32g(819200, true);
        const auto p_tight = resolve_inkling2_profile(in_tight);
        check(p_tight.regime == VramRegime::FullVram, "V100 819K tight stays FullVram");
        check(p_tight.chunk == 1024, "V100 819K tight falls back cleanly to chunk=1024");
        check(p_tight.max_adaptive_chunk == p_tight.chunk, "V100 819K tight keeps max_adaptive_chunk == chunk");
        check(p_tight.want_sp == 6, "V100 819K chunk=1024 uses exact want_sp=6 (best_sp=5 + 1 tail on 80 SMs)");
    }

    // 2. V100 32GB @ 1,048,576 ctx (1M, DSpark OFF): overflows FullVram -> drops cleanly to CoreResidentHybrid
    {
        const auto in = v100_32g(1048576, false);
        const auto p = resolve_inkling2_profile(in);
        std::printf("%s\n", format_profile_line(in, p).c_str());
        check(p.regime == VramRegime::CoreResidentHybrid, "V100 1M selects CoreResidentHybrid");
        check(p.chunk == 1024, "V100 1M keeps chunk=1024");
        check(p.max_adaptive_chunk == p.chunk, "V100 1M keeps max_adaptive_chunk == chunk");
        check(p.full_kv_init == 1048576, "V100 1M allocates 1M KV upfront while keeping core 128/128");
        check(p.use_d_hits, "V100 1M hybrid enables d_hits");
    }

    // 3. RTX 4070 12GB @ 65,536 ctx: CoreResidentHybrid (core 128/128 resident, dynamic 16K->65K KV, chunk=4096 + SWA compaction)
    {
        const auto in = rtx4070_12g(65536);
        const auto p = resolve_inkling2_profile(in);
        std::printf("%s\n", format_profile_line(in, p).c_str());
        check(p.arch.id == ArchId::Sm89Ada, "RTX 4070 selects Sm89Ada arch");
        check(p.regime == VramRegime::CoreResidentHybrid, "RTX 4070 65K selects CoreResidentHybrid");
        check(p.chunk == 4096, "RTX 4070 65K promotes to chunk=4096 (EX-4: SWA compaction + dynamic 16K KV init)");
        check(p.max_adaptive_chunk == 12288, "RTX 4070 65K exposes max_adaptive_chunk=12288 while keeping default chunk=4096");
        check(p.full_kv_init == 16384, "RTX 4070 65K starts KV at 16K (+780 MiB for staging & blk.0 residency on 4K-16K)");
        check(p.want_sp == 4, "RTX 4070 65K reserves 4 splits at T>=2048");
        check(p.pp_stage_cnt_pref == 2 && p.pp_stage_mb == 64, "RTX 4070 65K baseline stage config");
        check(p.use_d_hits, "RTX 4070 65K enables d_hits");
        check(!p.force_bf16_x_f16, "RTX 4070 uses native BF16 TC");
    }

    // 4. RTX 4070 12GB @ 131,072 ctx: CoreResidentHybrid (promotes to chunk=4096 with dynamic 16K KV init!)
    {
        const auto in = rtx4070_12g(131072);
        const auto p = resolve_inkling2_profile(in);
        std::printf("%s\n", format_profile_line(in, p).c_str());
        check(p.regime == VramRegime::CoreResidentHybrid, "RTX 4070 131K selects CoreResidentHybrid");
        check(p.chunk == 4096, "RTX 4070 131K promotes to chunk=4096 with 16K initial KV");
        check(p.max_adaptive_chunk == 12288, "RTX 4070 131K exposes max_adaptive_chunk=12288 while keeping default chunk=4096");
        check(p.full_kv_init == 16384, "RTX 4070 131K starts KV at 16K (+1.8 GiB free on short/medium prompts)");
        check(p.want_sp == 4, "RTX 4070 131K reserves 4 splits at T>=2048");
        check(p.pp_stage_cnt_pref == 2, "RTX 4070 131K baseline stage config");

        auto in_tight_desk = rtx4070_12g(131072);
        in_tight_desk.vram_free_bytes = (size_t) (9.50 * (double) GiB);
        const auto p_fb = resolve_inkling2_profile(in_tight_desk);
        check(p_fb.regime == VramRegime::CoreResidentHybrid, "RTX 4070 131K tight desktop stays CoreResidentHybrid");
        check(p_fb.chunk == 4096 && p_fb.want_sp == 4, "RTX 4070 131K tight desktop still fits chunk=4096 with 16K initial KV");
        check(p_fb.max_adaptive_chunk == 12288, "RTX 4070 131K tight desktop exposes max_adaptive_chunk=12288");
    }

    // 5. RTX 4070 12GB @ 262,144 ctx: TightHybrid (auto floor chunk=2048, adaptive 8K/12K enabled while Core fits, 16K->262K KV)
    {
        const auto in = rtx4070_12g(262144);
        const auto p = resolve_inkling2_profile(in);
        std::printf("%s\n", format_profile_line(in, p).c_str());
        check(p.regime == VramRegime::TightHybrid, "RTX 4070 262K selects TightHybrid");
        check(p.chunk == 2048, "RTX 4070 262K auto-selects chunk=2048");
        check(p.max_adaptive_chunk == 12288, "RTX 4070 262K TightHybrid unlocks max_adaptive_chunk=12288 while keeping chunk=2048 floor");
        check(p.full_kv_init == 16384, "RTX 4070 262K starts KV at 16K (+328 MiB VRAM for adaptive 8K/12K short/medium prefill)");
        check(p.want_sp == 4, "RTX 4070 262K reserves 4 splits at T=2048");
        check(p.pp_stage_cnt_pref == 3 && p.pp_stage_mb == 64, "RTX 4070 262K uses 3x64 MiB ring");
    }

    // 6. Explicit CLI --chunk override ALWAYS wins (e.g. user wants Prefill-focused --chunk 2048 at 65K, or --chunk 1024 at 262K)
    {
        auto in65 = rtx4070_12g(65536);
        in65.requested_chunk = 2048;
        const auto p65 = resolve_inkling2_profile(in65);
        check(p65.chunk == 2048 && p65.want_sp == 4, "explicit --chunk 2048 at 65K wins and adjusts want_sp=4");
        check(p65.max_adaptive_chunk == 2048, "explicit --chunk 2048 pins max_adaptive_chunk=2048");

        auto in262 = rtx4070_12g(262144);
        in262.requested_chunk = 1024;
        const auto p262 = resolve_inkling2_profile(in262);
        check(p262.chunk == 1024 && p262.want_sp == 5, "explicit --chunk 1024 at 262K wins and adjusts want_sp=5");
        check(p262.max_adaptive_chunk == 1024, "explicit --chunk 1024 pins max_adaptive_chunk=1024");
    }

    // 7. Unmeasured GPU (e.g. RTX 3090 24GB sm_86 @ 262K) uses conservative Generic profile (never guesses unmeasured T=2048)
    {
        ProfileInput in;
        in.cc_major = 8;
        in.cc_minor = 6;
        in.sm_count = 82;
        in.vram_total_bytes = 24 * GiB;
        in.vram_free_bytes = 21 * GiB;
        in.max_ctx = 262144;
        in.tier1_experts_bytes = kTier1Bytes;
        in.all_experts_bytes = kAllExpBytes;
        const auto p = resolve_inkling2_profile(in);
        std::printf("%s\n", format_profile_line(in, p).c_str());
        check(p.arch.id == ArchId::Generic, "sm_86 selects Generic arch profile");
        check(p.regime == VramRegime::FullVram, "24GB @ 262K fits FullVram");
        check(p.chunk == 1024 && p.want_sp == 7 && !p.use_d_hits, "24GB FullVram uses clean 1024/7/off");
        check(p.max_adaptive_chunk == p.chunk, "Generic FullVram keeps max_adaptive_chunk == chunk");

        in.vram_total_bytes = 12 * GiB;
        in.vram_free_bytes = (size_t) (9.5 * (double) GiB);
        const auto p_tight = resolve_inkling2_profile(in);
        check(p_tight.regime == VramRegime::TightHybrid, "unmeasured 12GB @ 262K selects TightHybrid");
        check(p_tight.chunk == 1024, "unmeasured sm_86 TightHybrid stays conservatively on chunk=1024");
        check(p_tight.max_adaptive_chunk == p_tight.chunk, "Generic TightHybrid keeps max_adaptive_chunk == chunk");

        in.max_ctx = 65536;
        const auto p_core = resolve_inkling2_profile(in);
        check(p_core.regime == VramRegime::CoreResidentHybrid, "unmeasured 12GB @ 65K selects CoreResidentHybrid");
        check(p_core.chunk == 1024, "unmeasured sm_86 CoreResidentHybrid stays conservatively on chunk=1024");
        check(p_core.max_adaptive_chunk == p_core.chunk, "Generic CoreResidentHybrid keeps max_adaptive_chunk == chunk");
    }

    // 8. Forced --expert-vram below all_experts_bytes demotes FullVram -> Hybrid cleanly (for parity testing)
    {
        auto in = rtx4070_12g(16384);
        in.expert_vram_forced = true;
        in.expert_vram_cap_bytes = (size_t) (4.46 * (double) GiB);
        const auto p = resolve_inkling2_profile(in);
        check(p.regime == VramRegime::TightHybrid, "forced --expert-vram 4.46 GiB selects TightHybrid");
        check(p.max_adaptive_chunk == 2048, "forced --expert-vram 4.46 GiB keeps max_adaptive_chunk=2048");
        check(p.use_d_hits && p.pp_stage_cnt_pref == 3, "forced hybrid keeps d_hits & staging active");
        check(p.full_kv_init == 16384, "16K ctx still allocates 16K KV upfront");
    }

    // 9. RTX 4070 12GB with DSpark enabled uses chunk=2048 + full 64 MiB pp_stage (freed during decode via phase-split)
    {
        auto in131 = rtx4070_12g(131072, /*draft=*/true);
        in131.vram_free_bytes = (size_t) (9.50 * (double) GiB);
        const auto p131 = resolve_inkling2_profile(in131);
        std::printf("%s\n", format_profile_line(in131, p131).c_str());
        check(p131.regime == VramRegime::TightHybrid, "RTX 4070 131K + DSpark demotes cleanly to TightHybrid");
        check(p131.chunk == 2048, "RTX 4070 131K + DSpark uses chunk=2048");
        check(p131.max_adaptive_chunk == 12288, "RTX 4070 131K + DSpark TightHybrid exposes max_adaptive_chunk=12288");
        check(p131.full_kv_init == 16384, "RTX 4070 131K + DSpark TightHybrid starts KV at 16K");
        check(p131.pp_stage_mb == 64, "RTX 4070 12GB + DSpark TightHybrid uses 64 MiB pp_stage");
        check(p131.kv_grow_reserve_mib == 80, "RTX 4070 12GB + DSpark TightHybrid uses 80 MiB kv grow reserve");

        auto in65 = rtx4070_12g(65536, /*draft=*/true);
        in65.vram_free_bytes = (size_t) (9.50 * (double) GiB);
        const auto p65 = resolve_inkling2_profile(in65);
        check(p65.regime == VramRegime::CoreResidentHybrid, "RTX 4070 65K + DSpark keeps CoreResidentHybrid");
        check(p65.chunk == 4096, "RTX 4070 65K + DSpark promotes to chunk=4096 with SWA compaction + DSpark phase offload");
        check(p65.max_adaptive_chunk == 12288, "RTX 4070 65K + DSpark CoreResidentHybrid exposes max_adaptive_chunk=12288");
        check(p65.pp_stage_mb == 64, "RTX 4070 65K + DSpark uses 64 MiB baseline pp_stage");
    }

    // 10. Explicit --chunk 8192 / 16384 accepted, --chunk 32768 clamped to 16384, and want_sp == 2 for chunk >= 8192
    {
        auto in8k = rtx4070_12g(65536);
        in8k.requested_chunk = 8192;
        const auto p8k = resolve_inkling2_profile(in8k);
        std::printf("%s\n", format_profile_line(in8k, p8k).c_str());
        check(p8k.chunk == 8192, "explicit --chunk 8192 is accepted without clamping to 4096");
        check(p8k.max_adaptive_chunk == 8192, "explicit --chunk 8192 sets max_adaptive_chunk=8192");
        check(p8k.want_sp == 2, "chunk=8192 sets want_sp=2 (saves 32 MiB @ 8K)");

        auto in16k = rtx4070_12g(65536);
        in16k.requested_chunk = 16384;
        const auto p16k = resolve_inkling2_profile(in16k);
        check(p16k.chunk == 16384, "explicit --chunk 16384 is accepted without clamping");
        check(p16k.max_adaptive_chunk == 16384, "explicit --chunk 16384 sets max_adaptive_chunk=16384");
        check(p16k.want_sp == 2, "chunk=16384 sets want_sp=2 (saves 64 MiB @ 16K)");

        auto in32k = rtx4070_12g(65536);
        in32k.requested_chunk = 32768;
        const auto p32k = resolve_inkling2_profile(in32k);
        check(p32k.chunk == 16384, "explicit --chunk 32768 clamps cleanly to 16384");
        check(p32k.max_adaptive_chunk == 16384, "clamped --chunk 32768 sets max_adaptive_chunk=16384");
        check(p32k.want_sp == 2, "clamped chunk=16384 sets want_sp=2");

        // Also verify want_sp == 2 when chunk >= 8192 is explicitly requested on V100 and Generic
        auto v100_8k = v100_32g(819200, true);
        v100_8k.requested_chunk = 8192;
        const auto pv8k = resolve_inkling2_profile(v100_8k);
        check(pv8k.chunk == 8192 && pv8k.want_sp == 2 && pv8k.max_adaptive_chunk == 8192,
              "V100 explicit --chunk 8192 is accepted with want_sp=2");

        auto gen_8k = rtx4070_12g(65536);
        gen_8k.cc_major = 8;
        gen_8k.cc_minor = 6;
        gen_8k.requested_chunk = 8192;
        const auto pg8k = resolve_inkling2_profile(gen_8k);
        check(pg8k.arch.id == ArchId::Generic && pg8k.chunk == 8192 && pg8k.want_sp == 2 && pg8k.max_adaptive_chunk == 8192,
              "Generic explicit --chunk 8192 is accepted with want_sp=2");
    }

    if (fails > 0) {
        std::printf("FAILED: %d check(s)\n", fails);
        return 1;
    }
    std::printf("ALL PASSED\n");
    return 0;
}
