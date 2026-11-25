<div align="center">

# Strata — Inkling-2-Mini Edition

**High-throughput, long-context CUDA + AVX2 hybrid inference for Inkling-2-Mini.**

[![Repository](https://img.shields.io/badge/Repo-kuraneko1%2FStrata-0ab887)](https://github.com/kuraneko1/Strata)
[![Engine](https://img.shields.io/badge/Engine-strata--inkling2-2563eb)](docs/INKLING2.md)
[![Verified GPUs](https://img.shields.io/badge/Verified-RTX%204070%2012GB%20%C2%B7%20V100%2032GB-6d28d9)](#supported-hardware--auto-profiles)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Highlights](#highlights) · [Profiles](#supported-hardware--auto-profiles) · [Benchmarks](#benchmarks) · [Build](#build-from-source) · [Quick Start](#quick-start) · [Architecture](#architecture--optimizations)

</div>

---

**`kuraneko1/Strata`** is a fork of [Niko1221/Strata](https://github.com/Niko1221/Strata) focused on **`strata-inkling2`**: a dedicated CUDA + AVX2 hybrid inference engine for the **Inkling-2-Mini** (`inkling2`) looped-MoE architecture.

It runs **65K–262K** context on a single **12 GB consumer GPU (RTX 4070)** and **819K–1M** context on a **32 GB Tesla V100** with **zero quantization or precision changes**, served behind an OpenAI- and Anthropic-compatible HTTP API and built-in Web UI (`serve.server`).

## Highlights

- **Zero-knob automatic tuning (`SM arch × VRAM regime`)** — Choose `--max-context` (and optionally `--draft` for DSpark). The engine inspects the GPU architecture and live free VRAM to select the verified chunk size, KV allocation policy, staging budget, FA2 split-K width, and CUDA Graph mode via [profile_rules.hpp](engine/include/strata/inkling2/profile_rules.hpp).
- **Zero quantization / precision compromises** — Native `UD-Q4_K_M` weights and `Q8` KV cache (`--kv-q8`), verified for 64/64 greedy token parity across optimization rounds.
- **Fast on both 12 GB hybrid and 32 GB full-VRAM GPUs** — On RTX 4070 12 GB, a single `--max-context 262144` run on real C++/CUDA code reaches **`8,054–8,072 t/s` prefill** at `8K` and sustains **`50.4 t/s`** (`16K`), **`43.4 t/s`** (`64K`), **`39.5 t/s`** (`128K`), and **`28.7 t/s`** (`256K`) steady decode with **DSpark ON** + dynamic offload/reload (`+4.5%` to `+15.4%` faster than DSpark OFF from `16K`–`256K`). On V100 32 GB (GPU 0, ECC Off), all 3,072 experts fit in VRAM up to 819K with DSpark (`135–167 t/s` decode at 64K–256K, `109.5 t/s` at 512K, `89.9 t/s` at 800K, `192.8 t/s` at 512).
- **Drop-in OpenAI & Anthropic API** — Works directly with Claude Code, Codex CLI, OpenCode, and other local coding agents, plus upstream Strata's Qwen3.8-Flash-Next and Qwen3.6-35B-A3B engines.

## Supported Hardware & Auto-Profiles

The profile selector separates **SM architecture** (`sm70`, `sm89`, `generic`) from **VRAM regime** (`full_vram`, `core_resident_hybrid`, `tight_hybrid`) so tuning for 12 GB Ada GPUs never regresses 32 GB Volta GPUs (and vice versa).

| Verified GPU | `--max-context` | Auto-Resolved Profile | Prefill Chunk | KV Cache Strategy | Expert Placement |
| --- | ---: | --- | ---: | --- | --- |
| **RTX 4070 12 GB** (`sm_89`) | `65536` | `sm89 + core_resident_hybrid` | `4096` (adapt. `<= 8192`) | Dynamic `16K → 36K → 65K` | `tier1` 100%, Phase-Split compaction |
| **RTX 4070 12 GB** (`sm_89`) | `131072` | `sm89 + core_resident_hybrid` | `4096` (adapt. `<= 8192`) | Dynamic `16K → 36K → 69K → 131K` | `tier1` 100%, Phase-Split compaction |
| **RTX 4070 12 GB** (`sm_89`) | `262144` | `sm89 + tight_hybrid` | `2048` (adapt. `<= 12288`) | Dynamic `16K → 36K → 69K → 139K → 262K` | Adaptive eviction (`75.0%` hit @ 262K) |
| **Tesla V100 32 GB** (`sm_70`) | `819200` (+ DSpark) | `sm70 + full_vram` | `1024` | Upfront `819K` | `3,072 / 3,072` (`100%` VRAM, `916 MiB` free) |
| **Tesla V100 32 GB** (`sm_70`) | `1048576` | `sm70 + core_resident_hybrid` | `1024` | Dynamic | `2,897 / 3,072` (`tier1` 100%) |

> **Other NVIDIA GPUs (`sm_75+`)**: Turing (`sm_75`), Ampere (`sm_86`), other Ada (`sm_89`), and Blackwell (`sm_120`, CUDA 13+) GPUs build and run via the safe `generic` profile and automatic VRAM sizing. Hardware verification and dedicated profiles currently cover **RTX 4070 12 GB** and **Tesla V100 32 GB**.

## Benchmarks

All numbers measured on real hardware with `Inkling-2-Mini-UD-Q4_K_M.gguf` (`--kv-q8`, zero quantization change).

### RTX 4070 12 GB (`sm_89`, Ryzen 7 5700X, 262K Single Profile)

Apples-to-apples real tokenized C++/CUDA code context ladder (`--max-context 262144 --kv-q8 --serve`, identical prompts & `n_gen=128` for both DSpark ON and DSpark OFF; `8K` = fresh empty-KV turn, `16K–256K` = continuous multi-turn sweep cumulative effective prefill & steady-state decode):

<p align="center">
  <img src="docs/media/bench-rtx4070.png" width="960" alt="RTX 4070 12 GB 262K Prefill and Decode throughput vs context depth (DSpark ON vs DSpark OFF)">
</p>

| Context Depth | DSpark ON (`--draft-max 3`) Prefill | DSpark ON Decode | DSpark OFF Prefill | DSpark OFF Decode | GPU Hit (ON / OFF) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `8K` (fresh) | **`8,053.6 t/s`** | `47.3 t/s` | **`8,071.9 t/s`** | **`48.6 t/s`** | `85.4%` / `85.5%` |
| `16K` | **`6,869.8 t/s`** | **`50.4 t/s`** | `6,706.3 t/s` | `45.1 t/s` | `85.4%` / `85.5%` |
| `32K` | `5,222.7 t/s` | **`47.6 t/s`** | **`5,241.2 t/s`** | `45.4 t/s` | `85.4%` / `85.5%` |
| `64K` | `4,674.8 t/s` | **`43.4 t/s`** | `4,699.6 t/s` | `40.4 t/s` | `85.3%` / `85.5%` |
| `128K` | `3,297.7 t/s` | **`39.5 t/s`** | `3,691.3 t/s` | `35.1 t/s` | `86.8%` / `85.5%` |
| `160K` | `2,811.8 t/s` | **`31.6 t/s`** | `2,766.0 t/s` | `28.9 t/s` | `72.3%` / `72.3%` |
| `256K` (`261.5K`) | `2,359.2 t/s` | **`28.7 t/s`** | `2,370.6 t/s` | `25.6 t/s` | `72.7%` / `71.7%` |

> **Why DSpark ON preserves Prefill and wins `16K–256K` Decode**: With Phase-Split dynamic offload/reload (`357.8 MiB` async pool transfer), DSpark weights are evicted before `pp_streaming` so initial `8K` prefill stays **`>8,050 t/s`**, then restored asynchronously for decode where speculative verification beats DSpark OFF across the entire `16K–256K` sweep (**`28.7` vs `25.6 t/s` at 256K**, **`+12.0%`**; on `131K`, **`39.2` vs `32.4 t/s`**, **`+21.2%`**). Zero precision degradation (`64/64` parity on both DSpark OFF & ON).

Full RTX 4070 progression & CSV data: [2026-11-22 RTX 4070 262K Report](docs/bench-results/2026-11-22-rtx4070-inkling2-262k/README.md).

### Tesla V100-PCIE-32GB (`sm_70`, CUDA 12.9)

<p align="center">
  <img src="docs/media/bench-v100.png" width="960" alt="Tesla V100 32 GB Prefill and Decode throughput vs context depth">
</p>

| Mode (`--max-context`) | Context Depth | Prefill | Decode | Notes |
| --- | ---: | ---: | ---: | --- |
| **`819200`** (DSpark ON, `full_vram`) | `64K` | **`2,692.0 t/s`** | **`167.3 t/s`** | `3,072/3,072` experts, `916 MiB` free (`192.8 t/s` @ 512) |
| **`819200`** (DSpark ON, `full_vram`) | `256K` | **`1,710.1 t/s`** | **`135.4 t/s`** | Volta `HMMA.884` Decode + 12-row SWA WMMA + `T=1` Q8_0 (97% STREAM) |
| **`819200`** (DSpark ON, `full_vram`) | `512K` | **`1,075.9 t/s`** | **`109.5 t/s`** | `>1,075 t/s` PP & `2.34×` decode past half-million ctx |
| **`819200`** (DSpark ON, `full_vram`) | `800K` | **`722.8 t/s`** | **`89.9 t/s`** | `3.47×` PP / `2.61×` decode vs baseline; `0→819K` in `11.9 min` |
| **`1048576`** (DSpark OFF, `core_resident_hybrid`) | `1M` | — | **`54.4 t/s`** | `2,897/3,072` experts (`+90` vs pre-arena) |

Full V100 800K progression, Roofline analysis & chart: [2026-11-22 V100 800K Report](docs/bench-results/2026-11-22-v100-inkling2-800k/README.md).

### Practical & Coding Multimodal Benchmarks (`mmproj-F16` + `--vram-elastic`)

End-to-end HTTP SSE practical benchmarks (`1,536` completion tokens, real 4-bug C++/Python debugging & Japanese cross-referencing, Text-only vs +1,024-token Image via native CUDA WMMA FA2 `strata-vision`):

<p align="center">
  <img src="docs/media/bench-rtx4070-v100-practical.png" width="960" alt="RTX 4070 and Tesla V100 Practical & Coding Multimodal Prefill and Decode vs Context Depth">
</p>

<p align="center">
  <img src="docs/media/bench-multimodal-breakdown.png" width="960" alt="Multimodal Text-only vs +Vision Decode Overhead and Turn 1 vs Turn 2 KV Reuse TTFT">
</p>

| Hardware & Mode | Depth | Code PP / Dec (Text) | Code +Vision Dec | Practical JP PP / Dec (+Vision) | Turn 2 KV Reuse TTFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| **RTX 4070 12 GB** (`65K`, Elastic Vision `965 MiB`) | `62K` | `2,864` / **`48.5 t/s`** | **`48.2 t/s`** | `2,832` / **`49.3 t/s`** (`Q=5/5`) | **`0.56 s`** |
| **RTX 4070 12 GB** (`131K`, Elastic Vision `965 MiB`) | `126K` | `2,505` / **`37.3 t/s`** | **`36.7 t/s`** | `2,512` / **`37.0 t/s`** (`Q=5/5`) | **`0.69 s`** |
| **RTX 4070 12 GB** (`262K`, Elastic Vision `965 MiB`) | `254K` | `2,059` / **`27.3 t/s`** | **`27.2 t/s`** | `2,126` / **`27.8 t/s`** (`Q=5/5`) | **`1.03 s`** |
| **Tesla V100 32 GB** (`262K`, Resident Vision + DSpark) | `62K` | `2,981` / **`126.0 t/s`** | **`118.3 t/s`** | `2,952` / **`112.7 t/s`** (`Q=5/5`) | **`0.46 s`** |
| **Tesla V100 32 GB** (`262K`, Resident Vision + DSpark) | `254K` | `2,144` / **`96.3 t/s`** | **`96.3 t/s`** | `2,144` / **`77.0 t/s`** (`Q=5/5`) | **`0.78 s`** |
| **Tesla V100 32 GB** (`800K`, Elastic Vision + DSpark) | `520K` | `1,540` / **`76.9 t/s`** | **`70.1 t/s`** | `1,537` / **`57.0 t/s`** (`Q=5/5`) | **`1.31 s`** |
| **Tesla V100 32 GB** (`800K`, Elastic Vision + DSpark) | `785K` | `1,203` / **`61.2 t/s`** | **`63.0 t/s`** | `1,205` / **`46.7 t/s`** (`Q=5/5`) | **`1.79 s`** |

Full multimodal tables, methodology & raw CSV: [2026-11-23 Multimodal 262K–800K Report](docs/bench-results/2026-11-23-rtx4070-v100-multimodal-262k-800k/README.md).

<a id="build-from-source"></a>

## Build from Source

Requires **Linux x86_64** (AVX2 CPU), **CMake 3.24+**, a **C++20** compiler, and the **CUDA Toolkit** (CUDA 12.x/13.x for `sm_75+`; CUDA 12.x required for `sm_70` V100).

```bash
git clone https://github.com/kuraneko1/Strata.git
cd Strata
```

### 1. Local NVIDIA GPU (`sm_75+`, e.g. RTX 4070)

```bash
cmake -S engine -B engine/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native

cmake --build engine/build --target strata-inkling2 -j
```

### 2. Portable Multi-GPU Binary (`sm_75+`, distributable)

```bash
cmake -S engine -B engine/build-dist \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_PORTABLE=ON \
  -DCMAKE_CUDA_ARCHITECTURES="75;86;89"   # add ;120 on CUDA 13.0+

cmake --build engine/build-dist --target strata-inkling2 -j
```

### 3. Tesla V100 (`sm_70`, CUDA 12.x)

```bash
cmake -S engine -B engine/build-sm70 \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_EXPERIMENTAL_SM60=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70

cmake --build engine/build-sm70 --target strata-inkling2 -j
```

### Profile Rules Unit Test (CPU-only)

```bash
g++ -std=c++20 -O2 -Wall -Wextra -Iengine/include \
  engine/tests/program/inkling2_profile_rules_test.cpp \
  -o /tmp/inkling2_profile_rules_test && /tmp/inkling2_profile_rules_test
```

## Quick Start

### 1. Server Mode (OpenAI / Anthropic API + Web UI)

Create a JSON config (e.g. `inkling2.json`). Simply change `--max-context` to switch tiers — the profile, chunk size, and KV policy adapt automatically:

```json
{
  "arch": "inkling2",
  "model_name": "Inkling-2-Mini-UD-Q4_K_M",
  "exe": "/path/to/Strata/engine/build/strata-inkling2",
  "args": [
    "--model", "/path/to/Inkling-2-Mini-UD-Q4_K_M.gguf",
    "--max-context", "262144",
    "--serve"
  ],
  "tokenizer": "/path/to/strata_tokenizer",
  "sampling": {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 40
  },
  "fit_max_tokens": true,
  "no_api_key": true
}
```

Recommended `--max-context` presets:
- **RTX 4070 12 GB — fastest daily coding (`65K`)**: `"--max-context", "65536"`
- **RTX 4070 12 GB — balanced long context (`131K`)**: `"--max-context", "131072"`
- **RTX 4070 12 GB — max context (`262K`)**: `"--max-context", "262144"`
- **Tesla V100 32 GB — 819K + DSpark**: `"--max-context", "819200", "--draft", "/path/to/Inkling-2-Mini-DSpark.gguf", "--draft-max", "3"`

Start the server:

```bash
PYTHONPATH=./terminal python3 -m serve.server \
  --engine strata \
  --config /path/to/inkling2.json \
  --host 0.0.0.0 \
  --port 8085 \
  --no-api-key
```

Startup logs print the active profile line so you can verify what was selected:

```text
[inkling2] profile: arch=sm89 regime=tight_hybrid (cc=8.9, sm=46, free=10.4 GiB) -> chunk=2048 kv_grow=dynamic stage_mb=64 need_free=112MiB want_sp=5 d_hits=on
```

Point any OpenAI- or Anthropic-compatible client (Claude Code, Codex CLI, OpenCode, etc.) at `http://localhost:8085`.

### 2. Launch Command Examples

Change only `--max-context` (and optionally `--draft` on 32 GB GPUs) — the engine auto-selects the matching profile, chunk size, and KV strategy.

```bash
# RTX 4070 12 GB — 65K daily coding (~8,055–8,074 PP @ 8K, ~43.7–51.5 t/s real-code Dec; DSpark ON)
./engine/build/strata-inkling2 \
  --model /path/to/Inkling-2-Mini-UD-Q4_K_M.gguf \
  --draft /path/to/dspark-Inkling-2-Mini-Q8_0.gguf --draft-max 3 \
  --max-context 65536 --bench --max-new 128

# RTX 4070 12 GB — 131K balanced (~8,040–8,068 PP @ 8K, ~39.2–50.5 t/s real-code Dec; DSpark ON)
./engine/build/strata-inkling2 \
  --model /path/to/Inkling-2-Mini-UD-Q4_K_M.gguf \
  --draft /path/to/dspark-Inkling-2-Mini-Q8_0.gguf --draft-max 3 \
  --max-context 131072 --bench --max-new 128

# RTX 4070 12 GB — 262K max context (~8,054 PP @ 8K -> 2,359 @ 256K / 50.4 -> 28.7 t/s real-code Dec; DSpark ON)
./engine/build/strata-inkling2 \
  --model /path/to/Inkling-2-Mini-UD-Q4_K_M.gguf \
  --draft /path/to/dspark-Inkling-2-Mini-Q8_0.gguf --draft-max 3 \
  --max-context 262144 --bench --max-new 128

# RTX 4070 12 GB — 262K API Server + Elastic Vision (OpenAI/Anthropic on :8085)
# Add "--vision", "--vram-elastic", "--kv-q8" and a "vision" block in your config JSON

# Tesla V100 32 GB — 819K + DSpark (auto: sm70 + full_vram, chunk=1024, 135–167 t/s @ 64K–256K, 89.9 t/s @ 800K)
./engine/build-sm70/strata-inkling2 \
  --model /path/to/Inkling-2-Mini-UD-Q4_K_M.gguf \
  --draft /path/to/Inkling-2-Mini-DSpark.gguf --draft-max 3 \
  --max-context 819200 --bench --max-new 128
```

Quick API smoke test once `serve.server` is running on port `8085`:

```bash
curl -s http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Inkling-2-Mini-UD-Q4_K_M",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 64
  }'
```

### 3. Optional Overrides

Explicit CLI flags and environment variables always override auto-profile defaults:

| Override | Purpose |
| --- | --- |
| `--chunk <N>` | Override auto-selected prefill chunk size (`1024` / `2048` / etc.) |
| `--expert-vram <GiB>` | Override automatic MoE expert VRAM budget |
| `--kv-q8` / `--kv-f16` | Quantized Q8 (default) or FP16 KV cache |
| `STRATA_PP_STAGE_MB` | PCIe double-buffered staging size per buffer in MiB (`0` disables) |
| `STRATA_PP_STREAM_T` | Minimum batch size `T` to engage PCIe prefill streaming |
| `STRATA_TIER2_FLOOR` | Optional `tier2` expert slots reserved per layer before maximizing `tier1` |
| `STRATA_MAX_PIN_MB` | Cap (`MiB`) on pinned host RAM for offloaded experts (`7168` default) |
| `STRATA_NO_PIN_EXPERTS` | Disable `cudaHostAlloc` pinned host memory for offloaded experts |

See [docs/INKLING2.md](docs/INKLING2.md) for the complete CLI flag list and wire-format details.

<a id="how-does-it-work"></a>

## Architecture & Optimizations

`strata-inkling2` is engineered around a Roofline analysis of looped-MoE (`tier1` 3×-looped core layers + `tier2` outer layers) with Sliding Window Attention + Full Attention:

- **Two-Axis Profile Table** ([profile_rules.hpp](engine/include/strata/inkling2/profile_rules.hpp)): Pure, CPU-testable rules keyed by `ArchProfile × VramRegime`. Keeps V100 `full_vram` on `want_sp=7` (`6` exact waves on 80 SMs) and skips `d_hits` atomics when all experts are VRAM-resident, while giving 12 GB `tight_hybrid` a tighter arena and `T=2048` prefill.
- **Unified Phase-Disjoint `layer_arena`**: Reuses a single 256B-aligned scratch arena across disjoint layer phases (attention, message passing, MoE routing/grouped WMMA), reclaiming `~400 MiB` on 12 GB GPUs and `+264 MiB` on V100.
- **Compact-Miss PCIe Double-Buffered Prefill (`pp_streaming`)**: Packs only VRAM-missing experts in contiguous host-to-device DMA runs into double-buffered staging (`pp_stage[2]`) with `expected_entries`-aware grouped WMMA launch sizing. Prefill↔Decode Phase-Split (`set_phase_prefill()` / `ensure_pp_cap()`), adaptive `T=8192/12288` chunking, and the cuBLASLt `g_cuda_lt_bucket_cache` push `262K` initial prefill past **`8,000+ t/s`** while keeping `DSpark` decode weights dynamically offloaded during prefill and asynchronously restored for decode.
- **Zero-Copy Direct AVX2 `nvfp4_256_ktile_*` CPU MoE** ([nvfp4_avx2.cpp](engine/src/kernels/cpu/nvfp4_avx2.cpp)): Unpermutes GPU K-tile NVFP4 weights directly in YMM registers via K-outer streaming, bit-identical to `ggml-cpu` (`0` differing bits), eliminating scalar LUT unpacking and cutting hybrid CPU MoE time nearly in half while keeping GPU `nvfp4_ktile` enabled.
- **Decode CUDA Graphs & Prefix Checkpoints**: Uses a single 36-layer `core_graph` when `tier1` is 100% resident (`full_vram` / `core_resident_hybrid`), per-layer `layer_pre_graph[48]` during partial offload (`tight_hybrid`), and prefix checkpoint caching for multi-turn coding sessions.
- **Multi-Threaded `dequant_tok`**: Parallelizes CPU-side Q4_K/Q6_K token dequantization for `T >= 128`, cutting 4K-token host dequant latency by `3.4×`.

## Upstream Models & Documentation

Everything from upstream Strata remains available in this repository:

- **Inkling-2-Mini engine guide**: [docs/INKLING2.md](docs/INKLING2.md)
- **Tesla V100 (`sm_70`) notes**: [docs/NVIDIA_V100.md](docs/NVIDIA_V100.md)
- **Upstream models**: [Qwen3.8-Flash-Next](docs/MODELS.md) · [Qwen3.6-35B-A3B](docs/QWEN36.md)
- **General documentation**: [How Strata works](docs/HOW_IT_WORKS.md) · [Troubleshooting](docs/TROUBLESHOOTING.md) · [Privacy & Safety](docs/SAFETY.md) · [Technical reference](AGENTS.md)

## Credits & License

- Built on [Niko1221/Strata](https://github.com/Niko1221/Strata) — see [Credits and model licenses](docs/HOW_IT_WORKS.md#credits).
- Released under the [MIT License](LICENSE).
