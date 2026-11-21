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

1. **Zero-Waste Host-Staged KV Expansion (`ensure_kv_cap`)**: Reallocating full-attention KV layers via a temporary 1-layer host staging buffer eliminates the `+520 MiB` double-buffer VRAM reserve during `131K → 262K` expansion (`+122` core experts retained permanently).
2. **Proportional Hot-Expert Shrinking + GPU Hit Tracking (`d_hits`)**: Replaced whole-layer `128 → 0` eviction with uniform partial-slot shrinking across `tier1` (3×-looped core layers), preserving the top-K hottest experts tracked by `k_group_experts` (`d_hits`) across all 12 core layers (`89/128` per layer = `1,068` VRAM experts vs `7×128 + 5×0 = 896` in baseline; GPU hit rate `43.8% → 74.9%`). Small-deficit guard trims tail layer(s) only when `<= 48` total experts need to be freed.
3. **Tier-1 Pinned Host Blobs (`cudaHostAlloc`) + Parallel Build**: Evicted/shrunk `tier1` core layers now use pinned host memory (prioritized over `tier2` via demotion when needed, swap-safe cap `MemAvailable - 5 GiB`, max `12 GiB`) and 8-thread parallel `build_all_expert_blobs()`, restoring full `25.31 GB/s` async PCIe Gen4 overlap during `pp_streaming`.
4. **Unlock `FA2_256_bulk` `best_sp=4` via Dead `eparts` Buffer**: Reuses `eparts` (`75.5 MB` at `T=1024`, dead during attention) instead of `escr` (`37.7 MB`), cutting 260K FA2 on 46-SM Ada from `395.6 ms` (`sp=1`) to `287.9 ms` (`sp=4`, `91.0 TFLOPS`) with zero extra VRAM.
5. **Fast `T=1` Q8KV Attention (`k_attn_q8_gqa3_split`)**: Upgraded to `fast_exp2f_hw`, shifted-`neg_sink2`, lazy rescale, `sp = sm_count` (`46` on RTX 4070), and exact FP16-zero V-skip (`sc - m < -28.0f`, `2^-28 = 3.7e-9`, `0` spills / 64 regs; verified 64/64 greedy parity).
6. **`T=1` Hybrid MoE CPU Pre-Combine (`k_combine_experts_hybrid`)**: CPU sums missed experts into one `9 KB` vector (8× smaller H2D upload than `72 KB`), eliminating `d_miss` upload and `k_merge_cpu` (and eliminating `k_combine_experts` entirely on `0`-slot outer layers).
7. **Core-Weighted Multi-Swap `rebalance()` + `--serve` `QUIT` Fix**: Prioritizes 3×-looped core layers (`weight=3.0`) with burst convergence after shrink (`48/8`) and steady-state throttling (`8/2` every 4 chunks), plus `reader.detach()` on `QUIT` to prevent stdin-EOF hang.

## Benchmark Results (`512 → 262,000`, Q8KV, DSpark OFF, Idle RTX 4070)

| Context Depth | Base (`9056d598`) PP | **Opt PP** | PP Gain | Base Decode | **Opt Decode** | Decode Gain | GPU Hit (Base → Opt) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `512` | `564.0 t/s` | **`584.0 t/s`** | `1.04×` | `45.10 t/s` | **`54.17 t/s`** | **`1.20×`** | `78.2% → 79.4%` |
| `32,768` | `1,895.8 t/s` | `1,687.8 t/s` | `0.89×` | `51.40 t/s` | **`51.68 t/s`** | `1.01×` | `91.3% → 93.8%` |
| `65,536` | `1,759.4 t/s` | `1,668.7 t/s` | `0.95×` | `25.33 t/s` | **`32.94 t/s`** | **`1.30×`** | `90.2% → 91.7%` |
| `131,072` | `903.2 t/s` | **`1,472.6 t/s`** | **`1.63×`** | `9.70 t/s` | `3.38 t/s`* | — | `68.7% → 77.0%` |
| `196,608` | `492.6 t/s` | **`1,163.0 t/s`** | **`2.36×`** | `11.98 t/s` | **`28.58 t/s`** | **`2.39×`** | `43.8% → 74.9%` |
| **`262,000`** | **`456.2 t/s`** | **`1,076.6 t/s`** | **`2.36×`** | **`12.03 t/s`** | **`26.60 t/s`** | **`2.21×`** | **`43.8% → 74.9%`** |
| **Cum `0→262K`** | **`385.42 s`** | **`201.18 s`** | **`1.92×`** | — | — | — | — |

*\*At `131,072`, `tg_ms` included the one-time `131K → 262K` KV expansion + 12-layer `hblobs` build (`~16.4 s` single-threaded before the 8-thread parallel build follow-up; steady-state decode is `~28.5 t/s`).*

### Follow-up Stage-Size & Varied-Token Verification (Idle RTX 4070, `ev=4.46 GiB`, Zipf-varied 35K vocab)
- `stage=48 MiB` (`tier1=87/128`): PP `1,481.4 t/s`, Decode `29.71 t/s`
- `stage=64 MiB` (`tier1=86/128`, new default): PP **`1,587.4 t/s` (`+7.2%`)**, Decode **`30.02 t/s`** (also preserves `tier1=128/128` full GPU through `131K`)
- `stage=96 MiB` (`tier1=85/128`): PP **`1,709.3 t/s`**, Decode `28.89 t/s`
- Greedy 64-token parity (`ev=4.46` vs `ev=7.8`): **`64/64` exact match**.
