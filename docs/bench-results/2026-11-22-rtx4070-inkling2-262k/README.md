# Inkling-2-Mini on RTX 4070 (12 GB): 262K Q8KV (DSpark OFF) Optimization & Roofline

- **Hardware**: NVIDIA GeForce RTX 4070 (12 GB GDDR6X, `sm_89`, 46 SMs, 36 MiB L2, PCIe Gen4 x16) + AMD Ryzen 7 5700X (8C/16T, 31 GiB DDR4)
- **Model**: `Inkling-2-Mini-UD-Q4_K_M.gguf` (zero quantization change; INT8 KV `--kv-q8`, DSpark OFF, `--max-context 262144`, `chunk=1024`)
- **Hardware Roofline (measured)**:
  - GDDR6X Read-only: **`476.1 GB/s`** (`504 GB/s` theoretical) | Copy: `443.0 GB/s`
  - PCIe Gen4 x16 H2D: **`25.31 GB/s` pinned** vs **`11.90 GB/s` pageable** (`2.13×`)
  - CPU 8-thread DDR4 Read: **`32.53 GB/s`** (concurrent CPU + PCIe combined: `32.7 GB/s`, shared DDR4 bus)
  - Standalone `FA2_256_bulk` (`T=1024`, 260K, `sp=4`): **`35.98 ms/L`** (`287.9 ms` for 8 layers, **`91.0 TFLOPS`**)
  - Standalone `k_attn_q8_gqa3_split` (`T=1`, 262K, `sp=46`): **`1.258 ms/L`** (`10.06 ms` for 8 layers, **`433.4 GB/s` = `91.0%`** of GDDR6X read peak)

## Key Optimizations (Zero Quantization Change)

1. **Zero-Waste Host-Staged KV Expansion (`ensure_kv_cap`)**: Reallocating full-attention KV layers via a temporary 1-layer host staging buffer eliminates the `+520 MiB` double-buffer VRAM reserve during `131K → 262K` expansion (`+184` core experts retained permanently: `1,080` vs `896`).
2. **Uniform Level-Target Hot-Expert Shrinking + GPU Routing Tracking (`k_route_inkling2` `d_hits`)**: Replaced whole-layer `128 → 0` eviction with uniform level-target partial shrinking across `tier1` (3×-looped core layers), preserving the top-90 hottest experts tracked on GPU across all 12 core layers (`90/128` per layer = `1,080` VRAM experts vs `7×128 + 5×0 = 896` in baseline; GPU hit rate `43.8% → 74.9%`). Small-deficit guard trims tail layer(s) only when `<= 48` total experts need to be freed.
3. **Zero-Stall 3-Slot Ring `pp_streaming` (`3 × 64 MiB = 192 MiB`)**: Tracking all routing counts in `k_route_inkling2` (`d_hits`) eliminates all 48 per-layer `h_ids` D2H copies and `cudaEventSynchronize(ev_route)` pipeline stalls inside `pp_streaming` (replaced by a single 12 KB `sync_d_hits()` at chunk end), while a 3-slot ring buffer keeps PCIe Gen4 continuously fed across 36 ms `FA2_256` layers.
4. **All-24-Layer Swap-Safe Pinned Host Blobs (`cudaHostAlloc`) + 8-Thread Parallel Build**: Caps pinned RAM safely at `clamp(MemAvailable - 5 GiB, 4 GiB, 14 GiB)` with `tier1` priority demotion, keeping all 12 core + all 12 outer layers (`13.76 GiB`) pinned at `25.31 GB/s` async PCIe Gen4 without swapping.
5. **Unlock `FA2_256_bulk` `best_sp=4` via Dead `eparts` Buffer**: Reuses `eparts` (`75.5 MB` at `T=1024`, dead during attention) instead of `escr` (`37.7 MB`), cutting 260K FA2 on 46-SM Ada from `395.6 ms` (`sp=1`) to `287.9 ms` (`sp=4`, `91.0 TFLOPS`) with zero extra VRAM.
6. **Fused `layer_pre_graph` + Fast `T=1` Q8KV Attention & CPU Pre-Combine**:
   - Captures `h_h` D2H + `router` + `k_route_inkling2` + `h_ids`/`h_ewts` D2H inside `layer_pre_graph[il]` (`-240` CUDA API calls/tok) and skips CPU `native_quant_act` when `n_miss == 0`.
   - `k_attn_q8_gqa3_split`: `fast_exp2f_hw`, shifted-`neg_sink2`, lazy rescale, `sp = sm_count` (`46`), and exact FP16-zero V-skip (`2^-28`, `0` spills / 64 regs).
   - `k_combine_experts_hybrid`: CPU pre-combines missed experts into one `9 KB` vector (`8×` smaller H2D, `moe_up = 0.43 ms/tok`, `moe_post = 0.61 ms/tok`, `moe_wait = 0.01 ms/tok`).
7. **Core-Weighted Burst+Throttled `rebalance()` + `--serve` `QUIT` Fix**: Burst convergence (`48/8` for 2 chunks after shrink), throttled steady-state (`8/2` every 4 chunks), plus `reader.detach()` on `QUIT`.
8. **Unified Phase-Disjoint `layer_arena` + SWA-3072 Ring `chunk=2048` + Parallel `dequant_tok` + WMMA `avg_rows` Fix (`a6268454`)**:
   - Aliases phase-disjoint scratch buffers (`rel_tab`, `pp_sp_buf`, `Q/K/V/att/att_gate/wo`, `msg0..2`, `ctrl`, `eparts`, `escr`, `moe_raw`) into one 256B-aligned `layer_arena`, shrinks `xq_k/xq_v` to `min(tmax, 8)`, and tightens `ensure_kv_cap` reserve (`256 → 112 MiB`), reclaiming `~400 MiB` (`90 → 96/128` core at `T=1024`: `27.41 t/s` 262K Decode).
   - Expands SWA circular ring to `n_swa + tmax` (`3072`) and auto-selects `chunk = 2048` on `<20 GiB` hybrid Q8KV (without DSpark), cutting PCIe expert streaming bytes/token in **half** (`560 ms → 280 ms / 1024 tok`) while keeping `86/128` core (`27.28 t/s` Decode).
   - Parallelizes `dequant_tok` across up to 8 CPU threads (`144 ms → 42 ms` over 4K Zipf, `3.43×`) and passes true `expected_entries` to `native_expert_grouped_wmma` to eliminate 4× excess CTA launches during `pp_streaming`.

## Benchmark Results (`512 → 262,000`, Q8KV, DSpark OFF, Idle RTX 4070)

| Context Depth | Base (`9056d598`) PP / Dec | R3 (`9e7429de`) PP / Dec | **R4 (`a6268454`, `T=2048`) PP / Dec** | PP Gain (vs Base) | Decode Gain (vs Base) | GPU Hit |
|---:|---:|---:|---:|---:|---:|---:|
| `512` | `564.0` / `45.10` | `554.9` / `55.31` | `404.7` / `54.55` | `0.72×` | **`1.21×`** | `87.3%` |
| `32,768` | `1,895.8` / `51.40` | `1,923.6` / `50.65` | **`2,404.3`** / `47.85` | **`1.27×`** | `0.93×` | `94.8%` |
| `65,536` | `1,759.4` / `25.33` | `1,752.4` / `35.52` | **`2,187.5`** / `25.22`* | **`1.24×`** | — | `95.0%` |
| `131,072` | `903.2` / `9.70` | `1,512.7` / `5.27`* | **`1,937.7`** / `3.40`* | **`2.15×`** | — | `85.2%` |
| `196,608` | `492.6` / `11.98` | `1,328.7` / `28.61` | **`1,679.1`** / **`28.82`** | **`3.41×`** | **`2.41×`** | `74.9%` |
| **`262,000`** | **`456.2` / `12.03`** | **`1,216.7` / `26.79`** | **`1,506.4` / `27.28`** | **`3.30×`** | **`2.27×`** | **`74.9%`** |
| **Cum `0→262K`** | **`385.42 s`** | **`182.74 s`** | **`145.92 s` (`2.43 min`)** | **`2.64×`** | — | — |

*\*At `65,536` and `131,072`, `tg_ms` includes one-time KV expansion + host blob builds; steady-state decode at `196K–262K` is `28.82–27.28 t/s`. With `STRATA_PP_STAGE_MB=96`, 262K PP reaches **`1,520.5 t/s` (`3.33×`)** and cumulative `0→262K` drops to **`143.95 s` (`2.40 min`, `2.68×`)**. With `--chunk 1024`, R4 keeps `96/128` core (`1,152` experts) for **`27.41 t/s`** 262K Decode (`57.38 t/s` at 512, `36.68 t/s` at 65K).*

## Tier1 vs Tier2 Allocation Proof (Idle RTX 4070, `ev=4.51 GiB`, Zipf 35K Vocab, 2048 PP + 64 TG)

| `STRATA_TIER2_FLOOR` | `tier1 / tier2` slots | PP (`2048` Zipf) | Decode (`63` tok) | `cpu_ms` | Total Hits |
|---:|---|---:|---:|---:|---:|
| **`0` (Optimal)** | **`86 / 0`** | **`1,720.7 t/s`** | **`27.95 t/s`** | **`1,364.6 ms`** | **`420,820`** |
| `8` | `77 / 8` | `1,574.5 t/s` | `25.28 t/s` | `1,583.9 ms` | `388,115` |
| `24` | `64 / 20` | `1,464.3 t/s` | `22.72 t/s` | `1,863.4 ms` | `349,812` |

- Greedy 64-token parity (`ev=4.46` vs `ev=7.8`): **`64/64` exact match**.
