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
- **Fast on both 12 GB hybrid and 32 GB full-VRAM GPUs** — On RTX 4070 12 GB, 65K/131K runs with 100% of looped `tier1` core experts resident in VRAM (`50.4 t/s` at 65K, `38.2 t/s` at 131K), while 262K switches to `chunk=2048` compact-miss PCIe streaming (`1,506 t/s` prefill, `27.3 t/s` decode at 262K). On V100 32 GB, all 3,072 experts fit in VRAM up to 819K with DSpark (`99–118 t/s` decode).
- **Drop-in OpenAI & Anthropic API** — Works directly with Claude Code, Codex CLI, OpenCode, and other local coding agents, plus upstream Strata's Qwen3.8-Flash-Next and Qwen3.6-35B-A3B engines.

## Supported Hardware & Auto-Profiles

The profile selector separates **SM architecture** (`sm70`, `sm89`, `generic`) from **VRAM regime** (`full_vram`, `core_resident_hybrid`, `tight_hybrid`) so tuning for 12 GB Ada GPUs never regresses 32 GB Volta GPUs (and vice versa).

| Verified GPU | `--max-context` | Auto-Resolved Profile | Prefill Chunk | KV Cache Strategy | Expert Placement |
| --- | ---: | --- | ---: | --- | --- |
| **RTX 4070 12 GB** (`sm_89`) | `65536` | `sm89 + core_resident_hybrid` | `2048` | Upfront `65K` (0s growth) | `1,764 / 3,072` (`tier1` 100%, `tier2` 18–20/128) |
| **RTX 4070 12 GB** (`sm_89`) | `131072` | `sm89 + core_resident_hybrid` | `2048` | Upfront `131K` (0s growth) | `1,554 / 3,072` (`tier1` 100%, `tier2` 1–2/128) |
| **RTX 4070 12 GB** (`sm_89`) | `262144` | `sm89 + tight_hybrid` | `2048` | Dynamic `32K → 262K` | Adaptive eviction (`74.9%` hit @ 262K) |
| **Tesla V100 32 GB** (`sm_70`) | `819200` (+ DSpark) | `sm70 + full_vram` | `1024` | Upfront `819K` | `3,072 / 3,072` (`100%` VRAM, `916 MiB` free) |
| **Tesla V100 32 GB** (`sm_70`) | `1048576` | `sm70 + core_resident_hybrid` | `1024` | Dynamic | `2,897 / 3,072` (`tier1` 100%) |

> **Other NVIDIA GPUs (`sm_75+`)**: Turing (`sm_75`), Ampere (`sm_86`), other Ada (`sm_89`), and Blackwell (`sm_120`, CUDA 13+) GPUs build and run via the safe `generic` profile and automatic VRAM sizing. Hardware verification and dedicated profiles currently cover **RTX 4070 12 GB** and **Tesla V100 32 GB**.

## Benchmarks

All numbers measured on real hardware with `Inkling-2-Mini-UD-Q4_K_M.gguf` (`--kv-q8`, zero quantization change).

### RTX 4070 12 GB (`sm_89`, Ryzen 7 5700X, DSpark OFF)

<p align="center">
  <img src="docs/media/bench-rtx4070.png" width="960" alt="RTX 4070 12 GB Prefill and Decode throughput vs context depth">
</p>

| Mode (`--max-context`) | Context Depth | Prefill | Decode | Notes |
| --- | ---: | ---: | ---: | --- |
| **`65536`** (`core_resident_hybrid`, `T=2048`) | `32K` | **`2,250.4 t/s`** | **`57.1 t/s`** | `tier1` 100%, 0s mid-run KV growth |
| **`65536`** (`core_resident_hybrid`, `T=2048`) | `65K` | **`2,014.1 t/s`** | **`45.0 t/s`** | `90.4%` hit (`--chunk 1024`: `1,671 / 50.1 t/s`) |
| **`131072`** (`core_resident_hybrid`, `T=2048`) | `64K` | **`1,987.6 t/s`** | `35.4 t/s` | `tier1` 100%, 0s mid-run KV growth |
| **`131072`** (`core_resident_hybrid`, `T=2048`) | `130.5K` | **`1,771.9 t/s`** | **`31.5 t/s`** | `--chunk 1024`: `1,467 / 38.2 t/s` |
| **`262144`** (`tight_hybrid`, `T=2048`) | `32K` | `2,330.4 t/s` | `58.0 t/s` | Dynamic KV (`32K → 262K`) |
| **`262144`** (`tight_hybrid`, `T=2048`) | `128K` | `1,851.2 t/s` | `30.3 t/s` | Post-expansion steady decode |
| **`262144`** (`tight_hybrid`, `T=2048`) | `256K` | **`1,437.0–1,506.4 t/s`** | **`26.1–27.3 t/s`** | `3.3×` PP / `2.3×` decode vs baseline |

Full RTX 4070 progression & CSV data: [2026-11-22 RTX 4070 262K Report](docs/bench-results/2026-11-22-rtx4070-inkling2-262k/README.md).

### Tesla V100-PCIE-32GB (`sm_70`, CUDA 12.9)

<p align="center">
  <img src="docs/media/bench-v100.png" width="960" alt="Tesla V100 32 GB Prefill and Decode throughput vs context depth">
</p>

| Mode (`--max-context`) | Context Depth | Prefill | Decode | Notes |
| --- | ---: | ---: | ---: | --- |
| **`819200`** (DSpark ON, `full_vram`) | `65K` | `2,428.7 t/s` | **`117.6 t/s`** | `3,072/3,072` experts, `916 MiB` free |
| **`819200`** (DSpark ON, `full_vram`) | `131K` | `1,935.4 t/s` | **`110.8 t/s`** | Zero regression vs pre-R4 baseline |
| **`819200`** (DSpark ON, `full_vram`) | `262K` | `1,483.7 t/s` | **`99.3 t/s`** | Verified up to `819,200` tokens |
| **`1048576`** (DSpark OFF, `core_resident_hybrid`) | `1M` | — | **`54.4 t/s`** | `2,897/3,072` experts (`+90` vs pre-arena) |

Full V100 800K progression, Roofline analysis & chart: [2026-11-22 V100 800K Report](docs/bench-results/2026-11-22-v100-inkling2-800k/README.md).

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
# RTX 4070 12 GB — 65K fastest (auto: sm89 + core_resident_hybrid, chunk=2048, ~2,014 PP / ~45–57 t/s Dec)
./engine/build/strata-inkling2 \
  --model /path/to/Inkling-2-Mini-UD-Q4_K_M.gguf \
  --max-context 65536 --bench --max-new 128

# RTX 4070 12 GB — 131K balanced (auto: sm89 + core_resident_hybrid, chunk=2048, ~1,772 PP / ~32–37 t/s Dec)
./engine/build/strata-inkling2 \
  --model /path/to/Inkling-2-Mini-UD-Q4_K_M.gguf \
  --max-context 131072 --bench --max-new 128

# RTX 4070 12 GB — 262K max context (auto: sm89 + tight_hybrid, chunk=2048, ~1,437–1,506 PP / ~26–27 t/s Dec)
./engine/build/strata-inkling2 \
  --model /path/to/Inkling-2-Mini-UD-Q4_K_M.gguf \
  --max-context 262144 --bench --max-new 128

# Tesla V100 32 GB — 819K + DSpark (auto: sm70 + full_vram, chunk=1024, 99–118 t/s)
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
- **Compact-Miss PCIe Double-Buffered Prefill (`pp_streaming`)**: Packs only VRAM-missing experts in contiguous host-to-device DMA runs into double-buffered staging (`pp_stage[2]`) with `expected_entries`-aware grouped WMMA launch sizing. Expanding the SWA circular ring to `n_swa + tmax` (`3072`) unlocks `chunk=2048` on hybrid `sm_89`, halving PCIe weight transfer bytes per token.
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
