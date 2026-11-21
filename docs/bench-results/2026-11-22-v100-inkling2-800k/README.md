# Inkling-2-Mini on V100 (32GB): 800K (~819,000) Sweep, Kernel Optimizations & 1M Native Capacity

Measured on `z170-desktop` (`CUDA 13.0` / `sm_70`) using commit `6ff2c009` (`Inkling-2-Mini-UD-Q4_K_M.gguf`, `dspark-Inkling-2-Mini-Q8_0.gguf`, `kv=int8`, `chunk=1024`).

Machine-readable CSV data: [v100-pcie-800k.csv](v100-pcie-800k.csv).

---

## 1. GPU 0 (`Tesla V100-PCIE-32GB`, ECC Off) `512 -> 819,000` (`800K`) Full Sweep

Config: `--max-context 819200 --draft ... --draft-max 3` (`3072/3072` experts resident in VRAM, `13.76 GiB` experts, `652 MiB` free VRAM).  
Each depth checkpoint appended tokens up to target depth (`temperature=0`) and generated 64 tokens.

| Depth | Added (`RESUME`) | Segment PP Time | Segment Avg PP | Last 1024-tok Chunk | Decode (64 tok) | Decode Speed | Total Elapsed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `512` | `+512` (`0`) | `0.40 s` | `1,291.2 t/s` | `333 ms` | `490.49 ms` | **`130.5 t/s`** | `0.9 s` |
| `65,536` (`64K`) | `+64,961` (`575`) | `40.03 s` | `1,622.9 t/s` | `837 ms` | `606.84 ms` | **`105.5 t/s`** | `41.5 s` |
| `131,072` (`128K`) | `+65,473` (`65,599`) | `63.41 s` | `1,032.5 t/s` | `1,196 ms` | `714.11 ms` | **`89.6 t/s`** | `105.7 s` |
| `262,144` (`256K`) | `+131,009` (`131,135`) | `196.60 s` | `666.4 t/s` | `1,924 ms` | `931.59 ms` | **`68.7 t/s`** | `303.3 s` |
| `393,216` (`384K`) | `+131,009` (`262,207`) | `289.67 s` | `452.3 t/s` | `2,653 ms` | `1,153.73 ms` | **`55.5 t/s`** | `594.3 s` |
| `524,288` (`512K`) | `+131,009` (`393,279`) | `383.18 s` | `341.9 t/s` | `3,380 ms` | `1,371.36 ms` | **`46.7 t/s`** | `979.0 s` |
| `655,360` (`640K`) | `+131,009` (`524,351`) | `476.69 s` | `274.8 t/s` | `4,114 ms` | `1,584.68 ms` | **`40.4 t/s`** | `1,457.5 s` |
| `786,432` (`768K`) | `+131,009` (`655,423`) | `570.23 s` | `229.7 t/s` | `4,840 ms` | `1,803.06 ms` | **`35.5 t/s`** | `2,029.7 s` |
| **`819,000` (`~800K`)** | `+32,505` (`786,495`) | `156.22 s` | `208.1 t/s` | **`5,028 ms`** | `1,857.53 ms` | **`34.5 t/s`** | `2,188.1 s` |

Cumulative PP (`0 -> 819,000`): **`2,176.43 s` (~36.3 min, overall `~376 t/s`)**. Zero OOMs, zero thermal/power throttling.

---

## 2. GPU 1 (`Tesla PG500-216`, ECC On) `512 -> 258,048` Progression (`a825d998` -> `6ff2c009` -> `2a9c9f5f`)

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

| Depth | `a825d998` PP | `6ff2c009` PP | **`2a9c9f5f` PP** | `a825d998` Decode | `6ff2c009` Decode | **`2a9c9f5f` Decode** | Total Decode Gain |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `512` | `0.37 s` (`1,390.1 t/s`) | `0.40 s` (`1,268.8 t/s`) | `0.43 s` (`1,186.4 t/s`) | `541.77 ms` (`118.1 t/s`) | `503.94 ms` (`127.0 t/s`) | `502.03 ms` (**`127.5 t/s`**) | **`+8.0%`** |
| `32,768` | `19.78 s` (`1,633.1 t/s`) | `17.74 s` (`1,814.5 t/s`) | `16.51 s` (**`1,949.6 t/s`**) | `618.71 ms` (`103.4 t/s`) | `571.64 ms` (`112.0 t/s`) | `530.53 ms` (**`120.6 t/s`**) | **`+16.6%`** |
| `65,536` | `27.23 s` (`1,204.1 t/s`) | `24.90 s` (`1,313.7 t/s`) | `23.00 s` (**`1,422.0 t/s`**) | `702.30 ms` (`91.1 t/s`) | `636.58 ms` (`100.5 t/s`) | `556.45 ms` (**`115.0 t/s`**) | **`+26.2%`** |
| `131,072` | `76.32 s` (`858.6 t/s`) | `70.64 s` (`926.9 t/s`) | `64.92 s` (**`1,008.5 t/s`**) | `882.22 ms` (`72.5 t/s`) | `759.08 ms` (`84.3 t/s`) | `605.31 ms` (**`105.7 t/s`**) | **`+45.8%`** |
| `196,608` | `105.52 s` (`620.9 t/s`) | `98.67 s` (`663.6 t/s`) | `90.25 s` (**`725.4 t/s`**) | `1,055.37 ms` (`60.6 t/s`) | `882.29 ms` (`72.5 t/s`) | `651.74 ms` (**`98.2 t/s`**) | **`+62.0%`** |
| **`258,048`** | `125.58 s` (`489.0 t/s`) | `117.83 s` (`520.9 t/s`) | `107.75 s` (**`569.6 t/s`**) | `1,211.86 ms` (`52.8 t/s`) | `988.37 ms` (`64.8 t/s`) | `696.24 ms` (**`91.9 t/s`**) | **`+74.1%`** |

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
