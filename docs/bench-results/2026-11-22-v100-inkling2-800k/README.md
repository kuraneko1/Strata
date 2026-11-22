# Inkling-2-Mini on V100 (32GB): 800K (~819,000) Sweep, Kernel Optimizations & 1M Native Capacity

Measured on `z170-desktop` (`CUDA 13.0` / `sm_70`) using commit `6ff2c009` (`Inkling-2-Mini-UD-Q4_K_M.gguf`, `dspark-Inkling-2-Mini-Q8_0.gguf`, `kv=int8`, `chunk=1024`).

Machine-readable CSV data: [v100-pcie-800k.csv](v100-pcie-800k.csv).

---

## 1. GPU 0 (`Tesla V100-PCIE-32GB`, ECC Off) `512 -> 819,000` (`800K`) Full Sweep (`6ff2c009` -> `8a30df9a` -> `2f09e21e` -> `9056d598`)

Config: `--max-context 819200 --draft ... --draft-max 3` (`3072/3072` experts resident in VRAM, `13.76 GiB` experts, `652 MiB` free VRAM, exact INT8 KV — zero quantization changes).  
Each depth checkpoint appended tokens up to target depth (`temperature=0`) and generated 64 tokens.

Key `2f09e21e` (Round 4) & `9056d598` (Round 5 — zero quantization change) changes:
- `wmma_acc_to_matA_16x16`: Bit-exact in-register `16x16` accumulator -> `matrix_a` converter (6 warp shuffles on Volta `sm_70`, 0 shuffles on `sm_75+`, verified `max_err = 0.00000000`), completely eliminating `sP` SMEM and 2-way `sP` bank conflicts.
- `k_attn_q8_gqa3_pp_fa2_256_bulk` + `_tail` (`9056d598`): Zero-`sP`, zero-`bar.sync`, L2-swizzled `dim3(3*q_tiles, n_head_kv, best_sp)` `256`-row global prefill (`2x4` QK^T + `2x8` P*V, `0.500` SMEM loads/WMMA, `87,040 B` SMEM), reaching **`169.1 ms/L` sustained (`61.0 TFLOPS` = `72.5%` of real cuBLAS `4096^3` peak)**.
- `k_attn_q8_gqa3_verify4_wmma_split` & `k_dspark_attn_q8_wmma_split` (`9056d598`): Zero-Sync warp-independent V3 (`128` thr, `4 blk/SM`, `21,760 B` / `18,432 B` SMEM, `128` regs, ZERO `__syncthreads()` inside the entire 800K KV loop, ZERO `sS`/`sP`), hitting **`2.363 ms/L` (`721.1 GB/s` = `83.3%` of real HBM2 STREAM peak)** standalone and breaking **`70.1 t/s` (`913.11 ms`)** at `819K` (`100.2 t/s` at `262K`)!

| Depth | Added (`RESUME`) | `6ff2c009` PP | `8a30df9a` (R3) PP | `2f09e21e` (R4) PP | **`9056d598` (R5) PP** | Total PP Gain | Last Chunk (`1024`) | `6ff2c009` Decode | `2f09e21e` (R4) Decode | **`9056d598` (R5) Decode** | Total Decode Gain |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `512` | `+512` (`0`) | `1,291.2 t/s` | `1,333.2 t/s` | `1,339.7 t/s` | **`1,349.2 t/s`** | `+4.5%` | `333 -> 289 ms` | `130.5 t/s` | `130.1 t/s` | `130.2 t/s` (`491.7 ms`) | `-0.2%` |
| `65,536` | `+64,961` (`575`) | `1,622.9 t/s` | `2,413.0 t/s` | `2,438.8 t/s` | **`2,427.2 t/s`** | **`+49.6%`** | `837 -> 464 ms` | `105.5 t/s` | `118.0 t/s` | **`118.6 t/s`** (`539.5 ms`) | **`+12.4%`** |
| `131,072` | `+65,473` (`65,599`) | `1,032.5 t/s` | `1,856.7 t/s` | `1,916.1 t/s` | **`1,935.3 t/s`** | **`+87.4%`** | `1,196 -> 571 ms` | `89.6 t/s` | `110.2 t/s` | **`111.6 t/s`** (`573.3 ms`) | **`+24.6%`** |
| `262,144` | `+131,009` (`131,135`) | `666.4 t/s` | `1,371.5 t/s` | `1,446.8 t/s` | **`1,483.0 t/s`** | **`2.23x`** | `1,924 -> 787 ms` | `68.7 t/s` | `97.2 t/s` | **`100.2 t/s`** (`638.6 ms`) | **`+45.9%`** |
| `393,216` | `+131,009` (`262,207`) | `452.3 t/s` | `1,017.1 t/s` | `1,088.2 t/s` | **`1,127.8 t/s`** | **`2.49x`** | `2,653 -> 1,005 ms` | `55.5 t/s` | `87.4 t/s` | **`91.2 t/s`** (`701.5 ms`) | **`+64.3%`** |
| `524,288` | `+131,009` (`393,279`) | `341.9 t/s` | `808.9 t/s` | `872.6 t/s` | **`910.4 t/s`** | **`2.66x`** | `3,380 -> 1,221 ms` | `46.7 t/s` | `78.9 t/s` | **`83.5 t/s`** (`766.9 ms`) | **`+78.8%`** |
| `655,360` | `+131,009` (`524,351`) | `274.8 t/s` | `671.0 t/s` | `728.5 t/s` | **`763.4 t/s`** | **`2.78x`** | `4,114 -> 1,442 ms` | `40.4 t/s` | `72.3 t/s` | **`77.0 t/s`** (`831.0 ms`) | **`+90.6%`** |
| `786,432` | `+131,009` (`655,423`) | `229.7 t/s` | `573.3 t/s` | `625.5 t/s` | **`657.2 t/s`** | **`2.86x`** | `4,840 -> 1,654 ms` | `35.5 t/s` | `66.5 t/s` | **`71.8 t/s`** (`891.4 ms`) | **`+102.3%`** |
| **`819,000`** | `+32,505` (`786,495`) | `208.1 t/s` | `525.8 t/s` | `574.3 t/s` | **`603.7 t/s`** | **`2.90x`** | **`5,028 -> 1,709 ms`** | `34.5 t/s` | `65.3 t/s` | **`70.1 t/s`** (`913.1 ms`) | **`+103.2% (2.03x)`** |

Cumulative `0 -> 819,000` wall time: `2,188.1 s` (`36.5 min`) in `6ff2c009` -> `942.4 s` (`15.7 min`) in `8a30df9a` -> `875.9 s` (`14.6 min`) in `2f09e21e` -> `841.7 s` (`14.0 min`, `984 t/s` avg PP) in `9056d598` -> `741.2 s` (`12.35 min`) in **Round 6** -> `720.3 s` (`12.00 min`) in **Round 7** -> **`713.1 s` (`11.88 min`, `3.07x` faster overall, `1,160 t/s` avg PP)** in **Round 8**. Zero OOMs, zero thermal/power throttling, zero quantization changes.

### Round 6–9 Update (GPU 0: `Tesla V100-PCIE-32GB`, 80 SM, ECC Off)

> Note: All Round 6–9 sweeps were measured strictly on **GPU 0 (`Tesla V100-PCIE-32GB`, 80 SM, ECC Off)** and are kept separate from Section 2's **GPU 1 (`Tesla PG500-216`, 72 SM, ECC On, higher-bandwidth SKU)**.

- **Round 6 (`940132c6`)**: Replacing C++ WMMA inside `k_attn_q8_gqa3_pp_fa2_256_bulk` on `sm_70` with direct PTX `HMMA.884` (`LDS.128`, 8-way batched `LDS.64`, conflict-free `STRIDE=144` SMEM `92,160 B`, compact `packed4` softmax) boosted Global Attention from `61.0` to **`74.8 TFLOPS` (`89.0%` of real `cuBLAS 4096^3` ceiling, `137.8 ms/L`)**.
- **Round 7 (`a2e79622`, Decode & MoE Floor Boost)**: 2-stage `k_argmax_stage1/stage2`, fused small-`T` F32/BF16 GEMMs, standalone zero-spill NVFP4 decode kernels, `k_mix_normed_fused` for `T >= 1`, and Volta native `HMMA.884` `BM=128` MoE prefill.
- **Round 8 (`23928eb7`, Full Roofline Sweep)**: Zero-SMEM `Q8_0` small-`T` MMVQ + pair fusion, zero-spill fused NVFP4 Gate+Up+SiTU+Q8_1, 1-pass SWA, 768-thread `k_rms_norm_q8_1`, 1-pass `k_mix_normed_fused_1pass`, fused `loop_down`/`route`/`shortconv`, and Head-Major Full-Attn KV (`[n_head_kv][cap][128]`), breaking `80.9 t/s` at 819K and `182.8 t/s` at 512.
- **Round 9 (Volta `HMMA.884` Decode Attention + 12-Row SWA WMMA + `T=1` Q8_0 + 4-blk/SM NVFP4 LUT — zero quantization change, `256/256` exact greedy parity vs R8, `0 MiB` extra VRAM `free=916 MiB`)**:
  - **D1 (`k_attn_q8_gqa3_verify4_swa_wmma_split` + Direct `xq` Combine)**: Packs `3 GQA × 4 tok = 12` SWA query rows into one `16×16×16` Tensor Core tile (`sp=10`, 4 independent warps, ZERO `__syncthreads()` in the KV loop, V-prefetch overlapping `QK^T`, fused sigmoid gate + direct `q8_1_warp_store` in `k_attn_split_combine`), cutting SWA verify from `59.3 µs -> 20.0 µs` (`3.0x` faster at 512 and saving **`-55.1 ms`** at full `n_swa=1025` window!).
  - **D2 (`k_dspark_attn_q8_wmma_split<IS_SWA>` + Direct `E.xq` Combine)**: Extended Tensor Core Split-KV + direct Q8_1 store to DSpark SWA layers 0 & 2 (`sp_swa=10`, ring-buffer `slot = j % cap`), eliminating scalar `k_dspark_attn_q8` bottlenecks at `P >= 256`.
  - **D3 (`k_mmvq_q8_0_smallT<1>` for `T=1` Q8_0)**: Enabled zero-SMEM `k_mmvq_q8_0_smallT<1>` for DSpark's `200K × 2304` LM head (`645.9 -> 581.5 µs = 841 GB/s = 97.1%` of V100 `865.9 GB/s` STREAM!).
  - **D4 (Volta Native `HMMA.884` Decode Attention)**: Upgraded `verify4_wmma_split`, `verify4_swa_wmma_split`, and `dspark_attn_q8_wmma_split<true/false>` on `sm_70` to direct PTX `mma.sync.aligned.m8n8k4` + `LDS.128` + conflict-free `STRIDE=144` `LDS.64` + direct `float2` epilogue (`STACK: 0` on DSpark and `0` inner-loop spills on target `verify4`), raising effective `64K -> 819K` Full-Attn KV scan bandwidth from `667.0 -> 716.7 GB/s` (`82.8%` of STREAM) and cutting 819K Decode from `791.31 -> 711.62 ms` (**`89.9 t/s`**, **`+11.1%` vs R8**, **`2.61x` vs Base**).
  - **D5 & P1/P2**: Raised `native_gu_nvfp4_72_situ_q8_1_kernel` occupancy to `4 blk/SM` (`REG: 58, STACK: 0`) with 1-cycle SMEM `s_ue4m3[256]` LUT (also in `native_down_nvfp4_36_kernel`), added 6-reg NVFP4 weight prefetch in `wmma_grouped_128_kernel<40>`, and fused `k_rms_norm_3_f16` / `k_rms_norm_f16` for prefill `attn_norm` (`div_64k` PP `-190 ms`).

| Depth | R7 (`a2e79622`) PP | R8 (`23928eb7`) PP | **R9 PP** | R9 vs Base PP | R7 Decode | R8 Decode | **R9 Decode** | R9 vs R8 Decode | R9 vs Base Decode |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `512` | `1,403.3 t/s` | `1,428.7 t/s` | **`1,416.8 t/s`** | `+9.7%` | `166.4 t/s` | `182.8 t/s` | **`192.8 t/s`** (`332.0 ms`) | **`+5.5%`** | `+47.7%` |
| `65,536` | `2,643.2 t/s` | `2,713.7 t/s` | **`2,692.0 t/s`** | `+65.9%` | `142.0 t/s` | `146.3 t/s` | **`167.3 t/s`** (`382.5 ms`) | **`+14.4%`** | `+58.6%` |
| `131,072` | `2,153.9 t/s` | `2,195.4 t/s` | **`2,181.3 t/s`** | `2.11x` | `131.8 t/s` | `136.2 t/s` | **`154.8 t/s`** (`413.5 ms`) | **`+13.7%`** | `+72.8%` |
| `262,144` | `1,693.2 t/s` | `1,718.5 t/s` | **`1,710.1 t/s`** | `2.57x` | `116.3 t/s` | `120.1 t/s` | **`135.4 t/s`** (`472.6 ms`) | **`+12.7%`** | `+97.1%` |
| `393,216` | `1,304.0 t/s` | `1,323.4 t/s` | **`1,318.5 t/s`** | `2.92x` | `104.3 t/s` | `107.5 t/s` | **`120.5 t/s`** (`531.3 ms`) | **`+12.1%`** | **`2.17x`** |
| `524,288` | `1,065.7 t/s` | `1,078.9 t/s` | **`1,075.9 t/s`** | `3.15x` | `93.9 t/s` | `97.6 t/s` | **`109.5 t/s`** (`584.5 ms`) | **`+12.2%`** | **`2.34x`** |
| `655,360` | `901.8 t/s` | `911.1 t/s` | **`909.0 t/s`** | `3.31x` | `86.0 t/s` | `89.1 t/s` | **`99.5 t/s`** (`643.0 ms`) | **`+11.7%`** | **`2.46x`** |
| `786,432` | `782.2 t/s` | `785.4 t/s` | **`783.6 t/s`** | `3.41x` | `79.3 t/s` | `82.3 t/s` | **`91.7 t/s`** (`697.7 ms`) | **`+11.4%`** | **`2.58x`** |
| **`819,000`** | `719.8 t/s` | `724.0 t/s` | **`722.8 t/s`** | **`3.47x`** | `77.0 t/s` | `80.9 t/s` | **`89.9 t/s`** (`711.6 ms`) | **`+11.1%`** | **`2.61x`** |

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

---

## 5. GPU 0 (`V100-PCIE-32GB`, 80 SM, ECC Off) Kernel Breakdown & Silicon Roofline

### Hardware Ceilings Measured on GPU 0 (`1380 MHz` SM / `877 MHz` HBM2 / `250 W` TDP)

| Hardware Metric | Spec Peak | Measured Ceiling | Efficiency |
|---|---:|---:|---:|
| HBM2 Memory Bandwidth (1 GB STREAM read) | `898.0 GB/s` | **`865.9 GB/s`** | `96.4%` |
| Pure INT8 KV Read (`819K`, `[8][pos][128]` / `[pos][8][128]` strided) | `898.0 GB/s` | `865.0` / `786.7 GB/s` (`1.940` / `2.132 ms/L`) | `96.3%` / `87.6%` |
| `verify4` Strided KV Read + `HFMA2` Dequant -> `sKV` | `898.0 GB/s` | **`849.9 GB/s`** (`2.005 ms/L`) | `98.2%` of STREAM |
| FP16 Tensor Core (`cuBLAS 4096^3` / `attn_o` / `moe` / `attn_q`) | `113.0 TFLOPS` | **`84.1`** / `82.3` / `68.5` / `66.5 TFLOPS` | `74.4%` (`4096^3`) |
| Pure 8-Warp WMMA Loop (`3x2` QK^T + `3x4` P*V, `0` spills) | `113.0 TFLOPS` | **`84.8 TFLOPS`** (`121.55 ms/L` at `819K`) | `100.8%` of cuBLAS |

### Kernel Section Breakdown (`1024`-tok Chunk) & Achieved Roofline Efficiency

| Regime / Kernel Section | Time (`1024`-tok chunk) | Share | Achieved Rate | % of Real Silicon Ceiling |
|---|---:|---:|---:|---:|
| **800K Prefill (`1,709 ms/chunk`) — Global Attention (`8L`, R5 `FA2_256_bulk` Zero-`sP`)** | **`1,353 ms`** (`169.1 ms/L`) | **`79.2%`** | **`61.0 TFLOPS`** (incl. INT8 dequant + softmax + Split-KV) | **`72.5%`** of `84.1` TC ceiling (`163.3 ms/L` = `63.1 TFLOPS` standalone) |
| **800K Prefill — All Non-Global (`40L` SWA + `48L` MoE/Dense/Mix/Ctrl)** | **`356 ms`** | `20.8%` | — | — |
| **Short Diverse Prefill (`416.5 ms/chunk` GPU, `2048`-tok `nvprof`)**: | | | | |
| ├ NVFP4 MoE Gate+Up + Down (`44L`, `wmma_grouped_64`) | `133.8 + 72.1 = 205.9 ms` | `49.4%` | `~21–30 TFLOPS` (Zipf routed `128` exp) | Memory/dequant + routing-imbalance bound |
| ├ Dense `Q8_0` GEMMs (`Q/K/V/O` + `dequant_q8_0`) | `47.7 ms` | `11.5%` | `66–82 TFLOPS` (`cuBLAS` / `cutlass`) | `80–97%` of `cuBLAS` ceiling |
| ├ `FA2_192` SWA (`40L`) + Global (`8L`) | `41.8 ms` | `10.0%` | — | `0` spills (`222` / `255` regs) |
| ├ `Q5_K` GU (`4L`) + `Q8_0` Down (`4L`) (`wmma_grouped_128`) | `16.7 + 8.8 = 25.5 ms` | `6.1%` | — | Vectorized `float4` epilogue |
| └ Elementwise / Norm / Mix / Rel / Ctrl / ShortConv / Combine | `95.6 ms` | `23.0%` | Memory-bound | Fused `mix`/`rel`/`ctrl` + `float4` `f32->f16`/`combine` |
| **800K Decode (`verify4_wmma_v3` Zero-Sync, `8L` Global KV scan, `1.70 GB/L`)** | **`18.9 ms/step`** (`2.363 ms/L`) | `~44%` of step | **`721.1 GB/s`** (incl. `HFMA2` + WMMA + softmax) | **`83.3%`** of `865.9` STREAM (**`97.5%`** of `MODE1` `2.305 ms/L` floor!) |
| **800K Decode (`dspark` full-attn `v3` Zero-Sync, `1L` + combine)** | **`~7.4 ms/step`** | `~17%` of step | — | `4 blk/SM`, `128` regs, `0` `__syncthreads()` in KV loop |

