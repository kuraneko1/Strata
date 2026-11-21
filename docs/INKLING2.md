# Inkling-2-Mini (`inkling2`) Support

This fork (`kuraneko1/Strata`, branched from [Niko1221/Strata](https://github.com/Niko1221/Strata)) adds the **`strata-inkling2`** CUDA + CPU (AVX2) hybrid inference engine for the `inkling2` architecture (**Inkling-2-Mini**), along with wire-format parsing (`InklingOutputParser` in `terminal/serve/frontend.py`) and host/PCIe/disk telemetry support in the serve stack.

> **Status & tested hardware:** Developed and benchmarked on **Linux x86_64 with NVIDIA GeForce RTX 4070 12 GB (`sm_89`, CUDA 12.9) and AMD Ryzen 7 5700X (8C/16T)**. Code paths are not tied to RTX 4070 and auto-scale to available VRAM via `cudaMemGetInfo`, though other GPUs/OSes have not been tested on hardware here.

---

## Build from source

Requires CMake 3.24+, a C++20 compiler, and CUDA Toolkit (CUDA 12.x/13.x for `sm_75+`; CUDA 12.x only for experimental `sm_70`).

```bash
git clone https://github.com/kuraneko1/Strata.git
cd Strata

# Auto-detect local NVIDIA GPU architecture:
cmake -S engine -B engine/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native

cmake --build engine/build --target strata-inkling2 -j
```

### Target GPU architecture notes

| GPU family | `-DCMAKE_CUDA_ARCHITECTURES` | Notes |
| --- | --- | --- |
| Current machine | `native` | Recommended when building locally |
| RTX 20 series (Turing) | `75` | Standard support floor (`sm_75+`) |
| RTX 30 series (Ampere) | `86` | Standard (`sm_86`) |
| RTX 40 series (Ada) | `89` | Tested (`sm_89`, CUDA 12.9) |
| RTX 50 series (Blackwell) | `120` | Requires CUDA 13.0+ (`engine/cmake/backends/cuda.cmake`) |
| Multi-GPU distributable build (`sm_75+`) | `"75;86;89"` (or `"75;86;89;120"` on CUDA 13+) | Consider `-DSTRATA_PORTABLE=ON` for portable x86 AVX2 CPU code |
| Tesla V100 / Titan V (Volta, `sm_70`) | `70` + `-DSTRATA_EXPERIMENTAL_SM60=ON` | **Experimental / untested on `strata-inkling2`**. Requires **CUDA 12.x** (CUDA 13 dropped `sm_70`). See [NVIDIA_V100.md](NVIDIA_V100.md). |

---

## Running the server (`serve.server`)

Prepare a JSON config pointing to your `strata-inkling2` binary, `.gguf` model, and tokenizer directory (`strata_tokenizer`):

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

Start the OpenAI / Anthropic / Web UI server:

```bash
PYTHONPATH=./terminal python3 -m serve.server \
  --engine strata \
  --config /path/to/strata-inkling-2-mini.json \
  --host 0.0.0.0 \
  --port 8085 \
  --no-api-key
```

---

## Standalone CLI (`strata-inkling2`)

```text
strata-inkling2 --model <file.gguf> [options]
```

| Flag | Default | Description |
| --- | --- | --- |
| `--model` / `--native <path>` | (required) | Path to GGUF model |
| `--draft` / `--draft-model <path>` | off | Optional DSpark draft GGUF (`arch=dflash`) for speculative decoding |
| `--draft-max` / `--spec <N>` | `3` | Draft block steps per verify cycle (`1..7`) |
| `--max-context <N>` | `65536` | Context window size (`kv_q8` full-attn KV cache scales with this) |
| `--chunk` / `--prefill <N>` | `256` | Prefill chunk size (e.g. `512` for higher prefill throughput) |
| `--max-new` / `-n <N>` | `32` | Tokens to generate in CLI / `--bench` mode |
| `--tokens <id1,id2,...>` | `"1,2"` | Comma-separated input token IDs for CLI / `--bench` |
| `--expert-vram <GiB>` | auto | Override VRAM budget for MoE experts (`-1` = auto with headroom) |
| `--cpu-threads <N>` | auto | CPU worker thread count for offloaded experts |
| `--kv-q8` / `--kv-f16` | `--kv-q8` | Quantized Q8 or FP16 KV cache |
| `--bench` | off | Run 1 warmup pass, reset state, and print `pp` / `tg` tok/s |
| `--serve` | off | Enter stdin/stdout protocol mode for `serve.server` |
| `--profile` / `--profile-detail` | off | Print per-stage timing breakdown (`--profile` adds syncs) |

### Environment tuning knobs

| Variable | Default | Purpose |
| --- | --- | --- |
| `STRATA_PP_STAGE_MB` | `96` | Double-buffered PCIe staging size per buffer in MiB during prefill (`0` disables) |
| `STRATA_PP_STREAM_T` | `16` | Minimum batch size `T` to engage PCIe double-buffered prefill streaming |
| `STRATA_TIER2_FLOOR` | `0` | Optional expert slots per `tier2` layer reserved before maximizing `tier1` |
| `STRATA_MAX_PIN_MB` | `7168` | Cap (`MiB`) on `cudaHostAlloc` pinned host memory for offloaded experts |
| `STRATA_NO_PIN_EXPERTS` | unset | Disable `cudaHostAlloc` for offloaded experts |

---

## Key Engine Optimizations

- **Automatic VRAM placement & OOM headroom (`place_experts`)**: Prioritizes 3x-looped core experts (`tier1`: `blk.6..17`), greedily packs `tier2` with any remaining VRAM, reserves extra headroom (`~400 MiB` when hybrid-offloading) for CUDA Graphs and desktop compositor safety, and caps pinned host RAM (`7.0 GiB` default) to prevent swap thrashing when all 24 layers offload.
- **Compact-miss PCIe prefill streaming (`pp_streaming`)**: Packs only missing experts (`slot_of[x] < 0`) in contiguous host-to-device runs into double-buffered staging (`pp_stage[2]`) with by-value `PpStageMap` (`k_group_experts_miss_map`), cutting PCIe DMA volume when `tier1` is partially cached (e.g. `262144` ctx).
- **Decode CUDA Graphs with fallback**: Uses a single 36-layer `core_graph` when `tier1` fits 100% in VRAM, or per-layer `layer_pre_graph[48]` when `tier1` is partially offloaded (`!core_all_gpu`), with graceful fallback to eager execution on graph allocation failure.
- **Fused dense & normalization kernels**: Fused SWA (`[Q;Gate;Rel;K;V]`) and full-attention (`[K;k_conv]`, `[V;v_conv]`) MMVQs, cached normalized residual blocks (`norm_blk_buf`), fused `k_shortconv` + residual add, `k_clip_norm_kv_store_q8`, `k_control_up_combine_t1`, `k_rms_norm_3`, `k_mix_normed_4`, and prefix checkpoint cache reuse (`save_checkpoint` / `restore_checkpoint`).

---

## Observed Benchmarks (RTX 4070 12 GB + Ryzen 7 5700X)

Measured on `Inkling-2-Mini-UD-Q4_K_M.gguf` (`--kv-q8 --bench --chunk 512 --max-new 128`, synthetic varied 512-token prompt):

| Context | VRAM expert placement | Prefill (`pp512`) | Decode (`tg127`) |
| ---: | --- | ---: | ---: |
| `65536` | `1676/3072` (`tier1=128/128`, `tier2=11..12/128`, `free=368 MiB`) | `546.95 t/s` | `45.30 t/s` |
| `262144` | `965/3072` (`tier1=80/128`, `tier2=0..1/128`, `free=376 MiB`) | `360.57 t/s` | `31.79 t/s` |
