# Inkling-2-Mini on V100 (32GB): 800K (~819,000) Sweep, Kernel Optimizations & 1M Native Capacity

Measured on `z170-desktop` (`CUDA 13.0` / `sm_70`) using commit `6ff2c009` (`Inkling-2-Mini-UD-Q4_K_M.gguf`, `dspark-Inkling-2-Mini-Q8_0.gguf`, `kv=int8`, `chunk=1024`).

Machine-readable CSV data: [v100-pcie-800k.csv](v100-pcie-800k.csv).

---

## 1. GPU 0 (`Tesla V100-PCIE-32GB`, ECC Off) `512 -> 819,000` (`800K`) Full Sweep (`6ff2c009` -> `8a30df9a`)

Config: `--max-context 819200 --draft ... --draft-max 3` (`3072/3072` experts resident in VRAM, `13.76 GiB` experts, `652 -> 660 MiB` free VRAM).  
Each depth checkpoint appended tokens up to target depth (`temperature=0`) and generated 64 tokens.

| Depth | Added (`RESUME`) | `6ff2c009` PP | **`8a30df9a` PP** | PP Gain | Last 1024-tok Chunk | `6ff2c009` Decode | **`8a30df9a` Decode** | Decode Gain |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `512` | `+512` (`0`) | `0.40 s` (`1,291.2 t/s`) | `0.38 s` (**`1,333.2 t/s`**) | `+3.3%` | `333 -> 292 ms` | `130.5 t/s` | **`131.3 t/s`** | `+0.6%` |
| `65,536` (`64K`) | `+64,961` (`575`) | `40.03 s` (`1,622.9 t/s`) | `26.92 s` (**`2,413.0 t/s`**) | **`+48.7%`** | `837 -> 480 ms` | `105.5 t/s` | **`119.9 t/s`** | **`+13.6%`** |
| `131,072` (`128K`) | `+65,473` (`65,599`) | `63.41 s` (`1,032.5 t/s`) | `35.26 s` (**`1,856.7 t/s`**) | **`+79.8%`** | `1,196 -> 604 ms` | `89.6 t/s` | **`110.8 t/s`** | **`+23.7%`** |
| `262,144` (`256K`) | `+131,009` (`131,135`) | `196.60 s` (`666.4 t/s`) | `95.52 s` (**`1,371.5 t/s`**) | **`2.06x`** | `1,924 -> 864 ms` | `68.7 t/s` | **`96.3 t/s`** | **`+40.2%`** |
| `393,216` (`384K`) | `+131,009` (`262,207`) | `289.67 s` (`452.3 t/s`) | `128.81 s` (**`1,017.1 t/s`**) | **`2.25x`** | `2,653 -> 1,127 ms` | `55.5 t/s` | **`85.7 t/s`** | **`+54.4%`** |
| `524,288` (`512K`) | `+131,009` (`393,279`) | `383.18 s` (`341.9 t/s`) | `161.96 s` (**`808.9 t/s`**) | **`2.37x`** | `3,380 -> 1,383 ms` | `46.7 t/s` | **`77.0 t/s`** | **`+64.9%`** |
| `655,360` (`640K`) | `+131,009` (`524,351`) | `476.69 s` (`274.8 t/s`) | `195.24 s` (**`671.0 t/s`**) | **`2.44x`** | `4,114 -> 1,644 ms` | `40.4 t/s` | **`69.7 t/s`** | **`+72.5%`** |
| `786,432` (`768K`) | `+131,009` (`655,423`) | `570.23 s` (`229.7 t/s`) | `228.52 s` (**`573.3 t/s`**) | **`2.50x`** | `4,840 -> 1,902 ms` | `35.5 t/s` | **`64.1 t/s`** | **`+80.6%`** |
| **`819,000` (`~800K`)** | `+32,505` (`786,495`) | `156.22 s` (`208.1 t/s`) | `61.82 s` (**`525.8 t/s`**) | **`2.53x`** | **`5,028 -> 1,970 ms`** | `34.5 t/s` | **`62.8 t/s`** | **`+82.0%`** |

Cumulative `0 -> 819,000` wall time: `2,188.1 s` (`36.5 min`) in `6ff2c009` -> **`942.4 s` (`15.7 min`, `2.32x` faster overall, `877 t/s` avg PP)** in `8a30df9a`. Zero OOMs, zero thermal/power throttling.

---

## 2. GPU 1 (`Tesla PG500-216`, ECC On) `512 -> 258,048` Progression (`a825d998` -> `6ff2c009` -> `2a9c9f5f` -> `8a30df9a`)

Key `6ff2c009` changes:
- `k_attn_q8_gqa3_pp_wmma_t32`: 4 barriers/tile (`48,768 B` shared mem <= `49,152 B`), `exp2f` + `kLog2e`
- `k_attn_q8_gqa3_split` & `k_dspark_attn_q8_sharedkv_split`: `kAttnSplitMax` / `split_full` raised `160 -> 240`, `i8x4_to_h2x2` K/V decode, `__half2` V accumulation (`__hfma2`), `exp2f`
- `k_attn_q8_gqa3_verify4_split`: Shared-KV (`BK=16`) + 8 half-warp parity split for `T=4` non-SWA verify (`sp=160`)

Key `2a9c9f5f` (Round 2 Tensor Core WMMA & Graph) changes:
- `k_attn_q8_gqa3_verify4_wmma_split`: Packs 12 query rows (`3 GQA heads x 4 tokens`) into one `16x16x16` WMMA tile in `30,976 B` (`<= 32 KB`, `3 blk/SM`, conflict-free strides), `sp=150` (`3.173 -> 1.430 ms/L`, **2.22x faster**).
- `k_dspark_attn_q8_wmma_split`: Packs 24 query rows (`3 GQA heads x 8 tokens`) into `32x128` WMMA (`48,768 B`, `2 blk/SM`, `sp=160`) + exact WMMA `K_self/V_self` (`4.425 -> 2.344 ms/L`, **1.89x faster**).
- `k_attn_q8_gqa3_pp_wmma_gqa3`: Shared-`sKV` 48-row (`16 tok x 3 heads`) WMMA prefill (`44,224 B`, `2 blk/SM`), sharing K/V HBM loads & `i8x4_to_h2x2` decode across all 3 GQA heads (`215.3 -> 195.5 ms/L`, **+10.1%**).
- `DsparkEngine::draft`: CUDA Graph capture (`dflare_graph_short` / `dflare_graph_split`) over all 3 DFlare layers using dynamic device `*pos_d` + pinned `h_anchor_pin`.
- `Engine::save_checkpoint`: Reusable pinned host staging buffer + `8,192`-tok prefill snapshot stride (`65,536`-tok rollback window).

Key `8a30df9a` (Round 3 Prefill FlashAttention-2 + MoE/Mix/Ctrl) changes:
- `k_attn_q8_gqa3_pp_fa2_192<SPLIT_KV>` + `k_attn_pp_split_combine`: True zero-`sS` FlashAttention-2 (`192` rows = `64 tok x 3 GQA`, `384` threads, `97,280 B` SMEM, `1 blk/SM`) with in-register `1x4` QK^T -> `ex2.approx` softmax -> register-resident `4x2` P*V, bulk KV register prefetch, and zero-VRAM Split-KV reusing `escr` (`195.22 -> 71.99 ms/L` at `258K`, **`2.71x` faster**, `45.1 TFLOPS` on 72-SM V100 / `81.8 TFLOPS` on RTX 4070).
- `iq_kernels.cu`: 1-instruction `__byte_perm` PRMT E2M1 decode + SMEM UE4M3 LUT in `wmma_grouped_64_kernel<40>` (`-12%` NVFP4 MoE), adaptive `BM=128/BN=128` `wmma_grouped_128_kernel` for `Q5_K`/`Q4_K`/`Q8_0` (`1.75x` faster `Q5_K` GU), vectorized `situ_f16_valid_kernel` (`1.42x`).
- `k_mix_normed_fused<1,3,4>` (`1.40x`), `k_rel_table_t32` (`1.69x`), FP16 Tensor Core `run_control` (`T >= 32`), cached FP16 Q/K/V activations, skip 200K vocab matmul on non-last prefill chunks, and Volta `Gemm::bf16` `x_f16=true` + `reserve_bf16_tc()` CUDA Graph safety fix.

| Depth | `a825d998` PP | `6ff2c009` PP | `2a9c9f5f` PP | **`8a30df9a` PP** | PP Gain vs Base | `2a9c9f5f` Decode | **`8a30df9a` Decode** | Total Decode Gain |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `512` | `0.37 s` (`1,390.1 t/s`) | `0.40 s` (`1,268.8 t/s`) | `0.43 s` (`1,186.4 t/s`) | `0.39 s` (**`1,301.0 t/s`**, chunk `304 ms`) | `-6.4%` (`+9.7%` vs R2) | `502.03 ms` (`127.5 t/s`) | `498.88 ms` (**`128.3 t/s`**) | **`+8.6%`** |
| `32,768` | `19.78 s` (`1,633.1 t/s`) | `17.74 s` (`1,814.5 t/s`) | `16.51 s` (`1,949.6 t/s`) | `12.46 s` (**`2,584.5 t/s`**, chunk `420 ms`) | **`+58.3%`** | `530.53 ms` (`120.6 t/s`) | `529.32 ms` (**`120.9 t/s`**) | **`+16.9%`** |
| `65,536` | `27.23 s` (`1,204.1 t/s`) | `24.90 s` (`1,313.7 t/s`) | `23.00 s` (`1,422.0 t/s`) | `14.93 s` (**`2,191.0 t/s`**, chunk `492 ms`) | **`+82.0%`** | `556.45 ms` (`115.0 t/s`) | `555.14 ms` (**`115.3 t/s`**) | **`+26.6%`** |
| `131,072` | `76.32 s` (`858.6 t/s`) | `70.64 s` (`926.9 t/s`) | `64.92 s` (`1,008.5 t/s`) | `36.68 s` (**`1,784.8 t/s`**, chunk `634 ms`) | **`+107.9%` (`2.08x`)** | `605.31 ms` (`105.7 t/s`) | `603.86 ms` (**`106.0 t/s`**) | **`+46.2%`** |
| `196,608` | `105.52 s` (`620.9 t/s`) | `98.67 s` (`663.6 t/s`) | `90.25 s` (`725.4 t/s`) | `45.85 s` (**`1,427.8 t/s`**, chunk `777 ms`) | **`+130.0%` (`2.30x`)** | `651.74 ms` (`98.2 t/s`) | `651.16 ms` (**`98.3 t/s`**) | **`+62.2%`** |
| **`258,048`** | `125.58 s` (`489.0 t/s`) | `117.83 s` (`520.9 t/s`) | `107.75 s` (`569.6 t/s`) | `51.53 s` (**`1,191.2 t/s`**, chunk `911 ms`) | **`+143.6%` (`2.44x`)** | `696.24 ms` (`91.9 t/s`) | `693.49 ms` (**`92.3 t/s`**) | **`+74.8%`** |

Cumulative `0 -> 258,048` PP time on GPU 1 (`PG500-216`, 72 SMs, ECC On): `354.80 s` (`a825d998`) -> `330.18 s` (`6ff2c009`) -> `302.86 s` (`2a9c9f5f`) -> **`161.84 s` (`8a30df9a`, `2.19x` faster end-to-end, `0 MiB` extra VRAM)**.

---

## 3. `0 -> 200K` One-Shot vs `0 -> 100K -> 200K` Split & GPU 0 vs GPU 1 (`6ff2c009`)

| GPU | Condition | Added (`RESUME`) | PP Time | PP Speed | Decode (64 tok) |
|---|---|---:|---:|---:|---:|
| GPU 0 (`V100-PCIE`, ECC Off) | One-shot `0 -> 200,000` | `+200,000` (`0`) | `195.77 s` | **`1,021.6 t/s`** | `75.0 t/s` (`853.2 ms`) |
| GPU 0 (`V100-PCIE`, ECC Off) | Split step 1 `0 -> 100,000` | `+100,000` (`0`) | `70.88 s` | `1,410.9 t/s` | `92.8 t/s` (`689.5 ms`) |
| GPU 0 (`V100-PCIE`, ECC Off) | Split step 2 `100K -> 200K` (cont) | `+99,937` (`100,063`) | `124.76 s` | **`801.0 t/s`** | `77.2 t/s` (`828.8 ms`) |
| GPU 1 (`PG500-216`, ECC On) | One-shot `0 -> 200,000` | `+200,000` (`0`) | `218.33 s` | `916.1 t/s` | `70.1 t/s` (`912.4 ms`) |
| GPU 1 (`PG500-216`, ECC On) | Split step 1 `0 -> 100,000` | `+100,000` (`0`) | `76.76 s` | `1,302.7 t/s` | `88.5 t/s` (`722.9 ms`) |
| GPU 1 (`PG500-216`, ECC On) | Split step 2 `100K -> 200K` (`omit_gen`) | `+100,000` (`100,000`) | `141.82 s` | `705.1 t/s` | `72.0 t/s` (`888.4 ms`) |

- Total wall time is identical (`70.88 + 124.76 = 195.64 s` vs `195.77 s` on GPU 0). The apparent drop (`1,021.6 -> 801.0 t/s`) reflects `1.5x` larger mean KV depth (`150K` vs `100K`).
- GPU 0 (`ECC Disabled`) outperforms GPU 1 (`ECC Enabled`, `1107 MHz` MCLK) by `+11.5%` PP and `+7%` Decode at 200K.

---

## 4. 32GB V100 VRAM Context Capacity Boundaries

| Configuration | `--max-context` | VRAM Experts | Free VRAM | Short Decode |
|---|---:|---|---:|---:|
| 100% VRAM experts (DSpark ON) | **`819,200`** (`800K`) | `3,072 / 3,072` (`13.76 GiB`) | `680 MiB` | `34.31 t/s` (`3 tok`) |
| Over 100%-VRAM (DSpark ON) | `851,968` (`832K`) | `2,992 / 3,072` (`Tier2 121..122`) | `360 MiB` | `31.44 t/s` (`15 tok`) |
| 100% VRAM experts (DSpark OFF) | `917,504` (`896K`) | `3,072 / 3,072` (`13.76 GiB`) | `1,302 MiB` | `69.53 t/s` (`15 tok`) |
| 100% VRAM experts max (DSpark OFF) | **`974,848`** (`952K`) | `3,072 / 3,072` (`13.76 GiB`) | `348 MiB` | `68.58 t/s` (`3 tok`) |
| 1M Native (DSpark OFF) | **`1,048,576`** (`1M`) | `2,807 / 3,072` (`Tier1 128/128`, `Tier2 105..106/128`) | `356 MiB` | `38.70 t/s` (`1 tok`) |
| 1M Native (DSpark ON) | `1,048,576` (`1M`) | `2,275 / 3,072` (`Tier1 128/128`, `Tier2 61..62/128`) | `356 MiB` | `23.20 t/s` (`31 tok`) |
| 1.25M check (DSpark OFF) | `1,310,720` | `1,959 / 3,072` (`Tier1 128/128`, `Tier2 35..36/128`) | `368 MiB` | `33.77 t/s` (`3 tok`) |
