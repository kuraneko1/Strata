# NVIDIA V100 / PG500 (Volta, `sm_70`) ボトルネック分析・最適化ロードマップ

作成日: 2026-11-27  
基準コミット: `ae9c00c72` / `6fb0765ab` (`v100-opt`, `--kv-q8`, `chunk=4096`, インクリング2系エンジン)  
対象ハードウェア・分離測定環境 (`z170-desktop1729`):
- **GPU 0 (`Tesla V100-PCIE-32GB`, 80 SM, ECC Off)**: 実測上限 **HBM2 Read `860.9 GB/s`** (Copy `781.5 GB/s`) / **FP16 Tensor Core `98.36 TFLOPS`** (`8192x8192x4096`)
- **GPU 1 (`Tesla PG500-216`, 72 SM, ECC On)**: 実測上限 **HBM2 Read `1,026.6 GB/s`** (Copy `988.4 GB/s`) / **FP16 Tensor Core `88.47 TFLOPS`**

モデル物理下限 (`chunk=4096`, `--kv-q8`):
- Active 重み転送量/tok: `4.060 GB` (Dense `1.784 GB` + MoE 8/128 `1.784 GB` + LM head `0.492 GB`)
- KV Q8_0 容量/tok: `16,640 B/tok` (Full-Attn 8層 × 2,080 B) → Decode 理論転送量: `4-WL ~4.22 GB/tok` / `65K = 5.227 GB/tok` / `262K = 8.496 GB/tok`
- Active 演算量/tok: Decode `9.829 GFLOP/tok`, Prefill `10.56` (`4-WL D=4595`) / `13.53` (`65K`) / `23.19 GFLOP/tok` (`262K`)

---

## 1. 現在の実効帯域・実効計算力 解放率サマリー

### GPU 0 (`V100-PCIE-32GB`: `860.9 GB/s`, `98.36 TFLOPS`)

| ワークロード | フェーズ (`dm`) | 実測速度 | 実効帯域 (上限比) | 実効計算力 (上限比) |
|---|---|---:|---:|---:|
| **4-WL** (`~4.6K`) | **Prefill** (`OFF`) | `4,610.7 t/s` | *計算律速* | **`48.68 TF` (`49.5%`)** |
| **Deep 65K** | **Prefill** (`OFF`) | `3,996.0 t/s` | *計算律速* | **`54.05 TF` (`55.0%`)** |
| **Deep 262K** | **Prefill** (`OFF`) | `2,611.8 t/s` | *計算律速* | **`60.56 TF` (`61.6%`)** |
| **4-WL** (`~4.6K`) | **Decode** (`OFF`) | `103.52 t/s` | **`437.0 GB/s` (`50.8%`)** | `1.017 TF` (`1.03%`) |
| **Deep 65K** | **Decode** (`OFF`) | `85.60 t/s` | **`447.2 GB/s` (`51.9%`)** | `0.841 TF` (`0.86%`) |
| **Deep 262K** | **Decode** (`OFF`) | `59.59 t/s` | **`506.4 GB/s` (`58.8%`)** | `0.586 TF` (`0.60%`) |
| **Deep 65K** | **Decode** (`dm=3 / dm=6`) | `125.83 / 124.07 t/s` | *(投機検証 M=4..8)* | `1.237 / 1.219 TF` (`1.26% / 1.24%`) |
| **Deep 262K** | **Decode** (`dm=3 / dm=6`) | `100.78 / 95.91 t/s` | *(投機検証 M=4..8)* | `0.991 / 0.943 TF` (`1.01% / 0.96%`) |

### GPU 1 (`PG500-216`: `1,026.6 GB/s`, `88.47 TFLOPS`)

| ワークロード | フェーズ (`dm`) | 実測速度 | 実効帯域 (上限比) | 実効計算力 (上限比) |
|---|---|---:|---:|---:|
| **4-WL** (`~4.6K`) | **Prefill** (`OFF`) | `4,334.4 t/s` | *計算律速* | **`45.76 TF` (`51.7%`)** |
| **Deep 65K** | **Prefill** (`OFF`) | `3,691.1 t/s` | *計算律速* | **`49.93 TF` (`56.4%`)** |
| **Deep 262K** | **Prefill** (`OFF`) | `2,409.2 t/s` | *計算律速* | **`55.86 TF` (`63.1%`)** |
| **4-WL** (`~4.6K`) | **Decode** (`OFF`) | `114.13 t/s` | **`482.2 GB/s` (`47.0%`)** | `1.122 TF` (`1.27%`) |
| **Deep 65K** | **Decode** (`OFF`) | `93.02 t/s` | **`485.9 GB/s` (`47.3%`)** | `0.914 TF` (`1.03%`) |
| **Deep 262K** | **Decode** (`OFF`) | `63.05 t/s` | **`536.0 GB/s` (`52.2%`)** | `0.620 TF` (`0.70%`) |
| **Deep 65K** | **Decode** (`dm=3 / dm=6`) | `132.49 / 121.87 t/s` | *(投機検証 M=4..8)* | `1.302 / 1.198 TF` (`1.47% / 1.35%`) |
| **Deep 262K** | **Decode** (`dm=3 / dm=6`) | `95.30 / 88.63 t/s` | *(投機検証 M=4..8)* | `0.937 / 0.871 TF` (`1.06% / 0.98%`) |

> **注記**: Prefill は `chunk=4096` で重み再利用・KV タイル共有が効く **Tensor Core 計算律速領域** (`49.5〜63.2%`)、Decode は TC 利用率 **約 `0.6〜1.7%`** の **完全メモリ帯域＋起動・同期レイテンシ律速** です（`nvprof` で PP の `dram_read_throughput` は直接未取得）。

---

## 2. 時間内訳の実測プロファイルデータ (GPU 0 `65K`)

### 2.1 Decode OFF (`w15`: `g64 - g1 = 63 tok`)

- `done` 計測 Wall 差分: `809.74 - 14.20 = 795.54 ms` (`12.63 ms/tok` `nvprof` 負荷込み / 非計測クリーン `85.60 t/s = 11.68 ms/tok`)
- 全 29 Decode カーネル純計: **`763.86 ms`** (`43,659 calls` = **`693.0 calls/tok`**、PP 微少差分 `8.44 ms` 含む `g64-g1` 差分は `772.30 ms`)
- `g64` 合計非カーネル HostGap: `45.88 ms` (`5.67%` of `809.74 ms`)

| # | カーネル名 | 合計時間 | 比率 | 呼出回数 | 平均時間 | 分類 |
|---:|---|---:|---:|---:|---:|---|
| 1 | `k_attn_q8_gqa3_split` | `187.37 ms` | 24.53% | 3,024 | `61.96 us` | Attention (KV理論下限 ~79ms) |
| 2 | `native_gu_nvfp4_72_situ_q8_1_t1_kernel<1>` | `103.42 ms` | 13.54% | 2,772 | `37.31 us` | MoE NVFP4 GU+SiTU |
| 3 | `k_mmvq_q8_0_smallT<1,1,0,4>` (Target LM Head) | `72.53 ms` | 9.50% | 1,323 | `54.82 us` | Dense Q8 (`98.1% HBM2` 限界) |
| 4 | `native_down_nvfp4_36_t1_kernel<1>` | `53.98 ms` | 7.07% | 2,772 | `19.47 us` | MoE NVFP4 Down |
| 5 | `k_mmvq_q8_0_smallT_multi<1,1,5,4>` (QKVGR) | `53.53 ms` | 7.01% | 1,764 | `30.35 us` | Dense Q8 |
| 6 | `k_mmvq_q8_0_smallT<1,1,0,2>` (O-proj) | `45.43 ms` | 5.95% | 3,024 | `15.02 us` | Dense Q8 |
| 7 | `k_pre_attn_t1_fused` | `33.02 ms` | 4.32% | 3,024 | `10.92 us` | 非GEMV グルー |
| 8 | `k_router_gemv_and_group_t1` | `31.25 ms` | 4.09% | 3,024 | `10.33 us` | 非GEMV / 小行列Router |
| 9 | `k_attn_split_combine` | `22.49 ms` | 2.94% | 3,024 | `7.44 us` | Attention Combine |
| 10 | `k_control_up_combine_rms_q8_t1<3>` | `19.78 ms` | 2.59% | 1,512 | `13.08 us` | 非GEMV LoopCtrl+RMS+Q8 |
| 11 | `k_mix_normed_fused_1pass_q8<1>` | `14.62 ms` | 1.91% | 1,512 | `9.67 us` | 非GEMV Mix+Norm+Q8 |
| 12 | `k_control_up_combine_rms_q8_t1<1>` | `14.40 ms` | 1.88% | 1,512 | `9.52 us` | 非GEMV LoopCtrl+RMS+Q8 |
| 13 | `k_mix_normed_q8_phase1<1>` | `14.02 ms` | 1.83% | 1,260 | `11.12 us` | 非GEMV Mix+Norm (2パス目無) |
| 14 | `k_mix_normed_fused_1pass_dual<1>` | `13.21 ms` | 1.73% | 1,260 | `10.48 us` | 非GEMV Mix |
| 15 | `k_loop_down_fused_smallT_3<1,1>` | `12.81 ms` | 1.68% | 1,512 | `8.47 us` | 非GEMV LoopCtrl Down |
| 16 | `k_mix_normed_fused_1pass<1,1>` | `11.72 ms` | 1.53% | 1,512 | `7.75 us` | 非GEMV Mix |
| 17 | `native_gu_q5k_72_situ_q8_1_t1_ktile_kernel` | `10.84 ms` | 1.42% | 252 | `43.00 us` | Dense MoE Q5K GU |
| 18 | `k_combine_experts_shortconv_k8<0,1>` | `8.50 ms` | 1.11% | 3,024 | `2.81 us` | 非GEMV Combine+Conv |
| 19 | `k_loop_down_fused_smallT<1,1>` | `8.14 ms` | 1.07% | 1,512 | `5.38 us` | 非GEMV LoopCtrl Down |
| 20 | `native_down_q8_0_36_t1_ktile_kernel` | `7.67 ms` | 1.00% | 252 | `30.46 us` | Dense MoE Q8_0 Down |
| 21 | `k_shortconv` | `6.36 ms` | 0.83% | 1,764 | `3.61 us` | 非GEMV ShortConv |
| 22 | `k_mix_normed_q8_phase2` | `5.84 ms` | 0.76% | 1,260 | `4.64 us` | 非GEMV Mix phase2 |
| 23 | `k_mix_normed_fused_1pass_q8<3>` | `4.76 ms` | 0.62% | 252 | `18.90 us` | 非GEMV Mix+Norm+Q8 |
| 24 | `k_mix_normed_fused_1pass_dual<3>` | `2.69 ms` | 0.35% | 252 | `10.69 us` | 非GEMV Mix |
| 25 | `k_rms_norm` | `2.62 ms` | 0.34% | 567 | `4.62 us` | 非GEMV RMSNorm |
| 26 | `k_bf16_conv_kv_smallT<1,1>` | `1.97 ms` | 0.26% | 504 | `3.91 us` | 非GEMV KV Conv |
| 27 | `k_argmax_stage1` | `0.36 ms` | 0.05% | 63 | `5.66 us` | Argmax |
| 28 | `k_rms_norm_q8_1` | `0.33 ms` | 0.04% | 63 | `5.17 us` | 非GEMV RMSNorm+Q8 |
| 29 | `k_argmax_stage2` | `0.20 ms` | 0.03% | 63 | `3.21 us` | Argmax |

### 2.2 Decode OFF カーネル間ギャップ詳細 (`w6f`: `trace65k_off_g16.csv`)

- **定常ステップ (Graph Replay, `14 tok` 平均)**:
  - Wall **`12.725 ms/tok`** = Kernel正味 **`12.297 ms/tok`** + `memcpy` `0.009 ms/tok` + **ギャップ合計 `0.420 ms/tok` (`63 tok` 換算約 `26.5 ms`)**
  - バケット内訳 (`1 tok` あたり):
    - `<= 2 us` (GPU カーネル間起動ギャップ): **`741.1回/tok` → `382.3 us/tok`** (平均 `0.52 us/回`)
    - `2〜10 us`: `2.1回/tok` → `11.8 us/tok`
    - `10〜100 us`: `1.4回/tok` → `26.1 us/tok` (`DtoH -> HtoD` 境界待ち **`20.08 us/tok`** 含む)
  - 主要遷移ギャップ TOP (`us/tok`):
    1. `k_pre_attn_t1_fused` → `k_attn_q8_gqa3_split`: **`35.53 us/t`** (`48c × 0.74 us`)
    2. `k_attn_q8_gqa3_split` → `k_attn_split_combine`: **`26.62 us/t`** (`48c × 0.55 us`)
    3. `k_attn_split_combine` → `k_mmvq_q8_0_smallT` (O-proj): **`23.49 us/t`** (`48c × 0.49 us`)
    4. `k_router_gemv_and_group_t1` → `native_gu_nvfp4`: **`22.38 us/t`** (`44c × 0.51 us`)
    5. `native_down_nvfp4` → `k_combine_experts_shortconv`: **`22.17 us/t`** (`44c × 0.50 us`)
    6. `DtoH` → `HtoD`: **`20.08 us/t`** (`1.0c × 20.08 us`)
    7. `native_gu_nvfp4` → `native_down_nvfp4`: **`19.94 us/t`** (`44c × 0.45 us`)
    8. `k_mix_normed_fused_1pass` → `k_loop_down_fused_smallT`: **`15.57 us/t`** (`24c × 0.65 us`)
    9. `k_mmvq_q8_0_smallT` → `k_shortconv`: **`14.03 us/t`** (`28c × 0.50 us`)
    10. `k_mix_normed_fused_1pass_q8` → `k_mmvq_q8_0_smallT`: **`13.47 us/t`** (`20c × 0.67 us`)
    11. `k_mix_normed_q8_phase1` → `k_mix_normed_q8_phase2`: **`11.94 us/t`** (`20c × 0.60 us`)
    12. `k_control_up_combine_t1` → `k_rms_norm_q8_1`: **`11.79 us/t`** (`24c × 0.49 us`)
    13. `k_mix_normed_fused_1pass_dual` → `k_loop_down_fused_smallT_3`: **`11.49 us/t`** (`24c × 0.48 us`)
    14. `k_control_up_combine_smallT_3` → `k_rms_norm_3_q8_1`: **`11.42 us/t`** (`23.9c × 0.48 us`)
    15. `k_loop_down_fused_smallT` → `k_control_up_combine_t1`: **`11.19 us/t`** (`23.9c × 0.47 us`)
- **初回トークン (`Step1 -> Step2` コールドキャプチャ時)**:
  - Wall `33.124 ms` = Kernel `12.357 ms` + Mem `13.909 ms` + Gaps **`6.861 ms`** (`DtoH->HtoD` `5.34 ms`, `DtoH->DtoH` `1.08 ms` ※後続 `f43ab8d0c` でD2H回数は大幅削減済み)

### 2.3 Decode `dm=6` 65K (`w14`: `g64-g1 = 63 tok` & `w6f`: `trace65k_dm6_g32`)

- `w14` 差分: `done` Decode差分 **`497.04 ms`** (`7.89 ms/tok`), カーネル計 **`464.82 ms`** (`18,844 calls`), 非カーネル差分 **`32.22 ms`** (`HtoD +18.64 ms`, API差分: `cudaStreamSynchronize 382.04ms / 23c`, `cudaEventSynchronize 327.76ms / 21c`, `cudaLaunchKernel 90.93ms / 13,725c`, `cudaGraphLaunch 9.26ms / 33c`, `cudaGraphInstantiate 6.50ms / 5c`)
- `w14` 主なカーネル上位:
  - NVFP4 MoE: `gu` `82.07 ms (17.66%, 1012c)` + `down` `46.61 ms (10.03%, 1012c)` = `128.68 ms (27.7%)`
  - `k_dspark_lmhead`: `35.85 ms (7.71%, 126c)` + `dspark_detail` 他 `6.15 ms`
  - `volta_s884gemm_fp16_256x128_ldg8_tn`: `24.57 ms (5.29%, 98c)` *(参考: `w13` では `128x128` が `44.05 ms / 64c`)*
  - `k_pre_attn_smallT_fused`: `21.53 ms (4.63%, 1104c, 19.50us)`
  - Verify Split-KV (Full `verify4/verify8` + SWA) + `k_attn_split_combine` (`8.02 ms`, `1104c`)
  - Q8_0 `smallT_multi` / `smallT` 各 `T` 分岐群
- `w6f` トレースギャップ (`dm=6` 約 `31 tok` 全体 `218.88 ms`, Kernel `173.03 ms`, Mem `14.52 ms`, Gaps `31.34 ms`):
  - `<= 100 us` 小ギャップ合計: **`5.21 ms`** (`<=2us` `6,078c = 3.75ms`, `2-10us` `298c = 1.00ms`, `10-100us` `18c = 0.46ms`)
  - **`> 100 us` 大ギャップ合計: `26.13 ms` (7回)** — 主に `mix_normed -> native_quantize` (`5.70ms`), `DtoH -> HtoD` (`5.94ms`, `4.97ms`, `4.51ms`), `k_dspark_norm_rope -> HtoD` (`4.60ms`) のコールドキャプチャ・ホスト待ち。

### 2.4 Prefill OFF 65K (`w7bfix`: `dec65k_off_g1.csv`, `KernSum = 16,291.57 ms`, `47,838 calls / 94 kernels`, `Wall = 16,550.98 ms`)

| カテゴリ | 主なカーネル | 時間 | 比率 |
|---|---|---:|---:|
| **① MoE Grouped WMMA** | `wmma_grouped_gu_situ_128_kernel<40>` (`4,275.46ms`, `704c`) + `<13>` (`446.15ms`) + `wmma_grouped_128<40>` (`2,379.59ms`, `704c`) + `<8>` (`242.81ms`) | **`7,344.01 ms`** | **45.1%** |
| **② FlashAttention-2** | `k_attn_q8_gqa3_pp_fa2_256_bulk` (`2,768.20ms`, `112c`) + `pp_fa2_192_swa_fast` 3バリアント計 (`1,762.03ms`) + `pp_fa2_256_tail` (`309.86ms`) + `pp_split_combine` (`59.60ms`) | **`4,899.69 ms`** | **30.1%** |
| **③ Dense TC GEMM** | `cutlass_70_tensorop_s884gemm_f16_128x128` (`1,324.13ms`) + `cutlass_relu` (`545.19ms`) + `volta_*s884gemm*` 計 (`458.27ms`: `128x128` `188.03`, `64x64_tn` `179.27`, `256x128` `61.54`, `64x64_f2f` `29.43`) + `cutlass_wmma_32x32` (`16.67ms`) | **`2,344.26 ms`** | **14.4%** |
| **④ ループ制御 + Router** | `k_control_up_combine_pp_wmma_vmsg<3>/<1>` (`333.37ms`) + `k_control_down_phaseA_pp_wmma_splitk<3>/<1>` + reduce (`204.38ms`) + `k_router_sgemm_f32_pp_splitk` + reduce (`234.76ms`) | **`~772.51 ms`** | **4.7%** |
| **⑤ Norm / Mix / Conv / Dequant / 他** | `k_shortconv_mix_normed_fused_1pass_pp` 計 (`~261ms`) + `k_postffn_shortconv_preattn_mix_pp` 計 (`~197ms`) + `k_rms_norm_3_f16_x4` (`96.06ms`) + `dequant_q8_0_h16_fast` (`54.25ms`) + `k_rms_norm_f16` (`48.89ms`) + 他50超 | **`~931.10 ms`** | **5.7%** |
| **⑥ PP カーネル間ギャップ (`w6f` 実PP窓)** | 小ギャップ `<=100us`: `50,806c = 52.62 ms` / 大ギャップ `>100us`: `22c = 22.24 ms` (`cutlass -> splitKreduce` `14.39ms / 48c`, `hist_push -> HtoD` `3.59ms`) | **`74.85 ms`** | (~0.4%) |

---

## 3. 物理限界でほぼ頭打ちな箇所（深追い不要）

1. **Target LM Head (`k_mmvq_q8_0_smallT<1,1,0,4>`, `72.53 ms` @ Decode OFF 65K)**:
   - 実測 **`98.1%` HBM2 帯域達成**。ここからの改善幅はほぼゼロ。
2. **Dense Q8 Linear & Dense MoE (`QKVGR` `53.5ms` / `O-proj` `45.4ms` / `Q5K+Q80` MoE `18.5ms`)**:
   - 帯域律速領域の高効率域に到達済み。
3. **Prefill `cutlass_70_tensorop_*` (`1.87s`) と `k_attn_q8_gqa3_pp_fa2_256_bulk` (`2.77s`)**:
   - `pp_fa2_256_bulk` は SASS の **`64.6%` が `HMMA`**。残余は Volta の `exp2f` (`MUFU`) と `int8 -> fp16` デクオントという命令セット物理制約。

---

## 4. 改善候補リスト（チェックリスト）

### A. Decode OFF / 共通パス（主戦場：`11.68 ms/tok (65K Steady 85.60 t/s)` → `11.56 ms/tok (86.47 t/s)`）

- [~] **OPT-D1 | SWA 層の Split-KV 撤廃・単一カーネル完結** *(計測のうえ T=1 退行によりリバート)*
  - **現状コスト**: SWA 層でも `k_attn_q8_gqa3_split` → `k_attn_split_combine` の 2 段階を起動（`k_attn_split_combine` `22.49ms` の大部分と前後のギャップ `0.55 + 0.49 us/回`）。
  - **箇所**: [inkling2_engine.cu:8308-8317](file:///home/ryu/Strata-v100/engine/src/inkling2/inkling2_engine.cu#L8308-L8317), [inkling2_kernels.cuh:13081-13160](file:///home/ryu/Strata-v100/engine/src/inkling2/inkling2_kernels.cuh#L13081-L13160)
  - **結果 (`4e900b216` でリバート)**: SWA 1-pass 化 + OPT-D2 の合算で `65K nvprof` 上 `+0.487 ms/tok` の退行を確認し、`k_attn_q8_gqa3_split` の単一パスへ戻した（`T=1` の Split-KV 並列読込のほうが V100 上で高速）。

- [~] **OPT-D2 | Full-Attn `k_attn_q8_gqa3_split` の `sm_70` `6 -> 8 blk/SM` 解放** *(計測のうえ T=1 退行によりリバート)*
  - **現状コスト**: `187.37 ms (24.53%)`、`sm_70` ではレジスタ予算 `80` 制約のため `__launch_bounds__(128, 6)` (`inkling2_kernels.cuh:13086`)。理論 KV 下限 (`~79 ms`) との差 約 `108 ms` がここに含まれる。
  - **箇所**: [inkling2_kernels.cuh:13081-13160](file:///home/ryu/Strata-v100/engine/src/inkling2/inkling2_kernels.cuh#L13081-L13160), [inkling2_engine.cu:8303-8314](file:///home/ryu/Strata-v100/engine/src/inkling2/inkling2_engine.cu#L8303-L8314)
  - **結果 (`4e900b216` でリバート)**: OPT-D1 と合わせて `+0.487 ms/tok` の退行実測により `__launch_bounds__(128, 6)` および元の `sp=12/6` 配分へ戻した。

- [x] **OPT-D3 | Full-Attn `k_attn_split_combine` 出力と O-proj Q8_1 入力の整流** *(実装・採用)*
  - **現状コスト**: Full 層でも `split_combine` (`22.49 ms`) が `attn_split` を読み、`att` / `xq` を書いてから `O-proj` (`45.43 ms`) が読んでいる（間に `0.49 us × 48 = 23.49 us/tok` ギャップ）。
  - **箇所**: [inkling2_engine.cu:8315-8317](file:///home/ryu/Strata-v100/engine/src/inkling2/inkling2_engine.cu#L8315-L8317)
  - **結果**: `T=1` の `xq` 専用経路で不要な `att` F32 出力 (`att_f32_out`) 書出しを省略し採用。

- [~] **OPT-D4 | `k_mix_normed_q8_phase1` + `phase2` の 1 カーネル統合** *(計測のうえ T=1 退行によりリバート)*
  - **現状コスト**: `phase1` (`14.02 ms`, `1260c`) + `phase2` (`5.84 ms`, `1260c`) + ギャップ (`11.94 us/t`) = 計 **`~20.6 ms`**。
  - **箇所**: [inkling2_kernels.cuh:1888](file:///home/ryu/Strata-v100/engine/src/inkling2/inkling2_kernels.cuh#L1888), [inkling2_engine.cu:8584](file:///home/ryu/Strata-v100/engine/src/inkling2/inkling2_engine.cu#L8584)
  - **結果 (`4e900b216` でリバート)**: OPT-D6 側の `deferred_postffn_t1` と含めた融合で `+0.428 ms/tok` の退行を確認し、元の multi-CTA `phase1`/`phase2` (`ae9c00c72`) へ戻した。

- [x] **OPT-D5 | `k_loop_down_fused_smallT` + `k_control_up_combine_rms_q8_t1` 縦融合 (`ctrl64` GMEM 往復撤廃)** *(実装・採用: `-0.124 ms/tok`)*
  - **現状コスト**: `k_loop_down_fused_smallT` (`8.14 ms`) + `_3` (`12.81 ms`) + `control_up_combine_rms_q8_t1<1>` (`14.40 ms`) + `<3>` (`19.78 ms`) + 遷移ギャップ (`11.19 + ~11.4 us/t`) = 計 **`~56.5 ms`**。
  - **箇所**: [inkling2_kernels.cuh:2491, 2651, 5728](file:///home/ryu/Strata-v100/engine/src/inkling2/inkling2_kernels.cuh#L2491), [inkling2_engine.cu:7513-7555, 8870-8886](file:///home/ryu/Strata-v100/engine/src/inkling2/inkling2_engine.cu#L8870-L8886)
  - **結果**: `k_loop_ctrl_fused_rms_q8_t1<1,64>/<3,32>` により `0.875 -> 0.751 ms/tok` (`-0.124 ms/tok`, `-48 launches/tok`) を実測達成。

- [~] **OPT-D6 | `k_combine_experts_shortconv` + 次段 `k_mix_normed` / `k_shortconv` の縦融合** *(計測のうえ T=1 退行によりリバート)*
  - **現状コスト**: `k_combine_experts_shortconv_k8` (`8.50 ms`, `3024c`) + `k_shortconv` (`6.36 ms`, `1764c`) + 各前後ギャップ（`down -> combine` `22.17 us/t`、`mmvq -> shortconv` `14.03 us/t`）。
  - **箇所**: [inkling2_engine.cu:8836](file:///home/ryu/Strata-v100/engine/src/inkling2/inkling2_engine.cu#L8836)
  - **結果 (`4e900b216` でリバート)**: OPT-D4 と合わせて `+0.428 ms/tok` 退行のため `deferred_postffn_t1` / 1-pass 融合を無効化・リバート。

- [~] **OPT-D7 | `k_pre_attn_t1_fused` の構造・SM 配分見直し** *(計測のうえ T=1 退行によりリバート)*
  - **現状コスト**: `33.02 ms` (`3024c`, `10.92 us/回`) ＋ 後続 `k_attn_q8_gqa3_split` との間が最重ギャップ (**`0.74 us × 48 = 35.53 us/t`**)。
  - **箇所**: [inkling2_kernels.cuh:7362](file:///home/ryu/Strata-v100/engine/src/inkling2/inkling2_kernels.cuh#L7362), [inkling2_engine.cu:8101-8113](file:///home/ryu/Strata-v100/engine/src/inkling2/inkling2_engine.cu#L8101-L8113)
  - **結果 (`4e900b216` でリバート)**: Split 変更が `+0.040 ms/tok` の退行となったため `ae9c00c72` の実装へ戻した（結果 `0.524 -> 0.516 ms/tok`）。

- [x] **OPT-D8 | `k_router_gemv_and_group_t1` の HBM2/SM 飽和最適化** *(実装・採用: `-0.019 ms/tok`)*
  - **現状コスト**: `31.25 ms` (`3024c`, `10.33 us/回`) ＋ 後続 `native_gu_nvfp4` ギャップ (`22.38 us/t`)。
  - **箇所**: [inkling2_engine.cu:8904-8911](file:///home/ryu/Strata-v100/engine/src/inkling2/inkling2_engine.cu#L8904-L8911)
  - **結果**: `k_router_gemv_and_group_t1<2>` 化により `0.496 -> 0.477 ms/tok` (`-0.019 ms/tok`) を達成し採用。

- [x] **OPT-D9 | NVFP4 MoE Decode (`157.40 ms`) の中間往復・展開コスト圧縮** *(一部採用: GU `-0.080 ms/tok` 採用 / Down `+0.038 ms/tok` リバート)*
  - **現状コスト**: Active 重み `1.784 GB` の物理下限 `~130 ms` に対し `157.40 ms`（差分 約 `27 ms` ＋ `gu -> down` ギャップ `19.94 us/t`）。
  - **箇所**: `engine/src/kernels/cuda/iq_kernels.cu` (`native_gu_nvfp4_72_situ_q8_1_t1_kernel`, `native_down_nvfp4_36_t1_kernel`)
  - **結果**: `native_gu_nvfp4_72_situ_q8_1_t1_kernel<bool=1>` の改善 (`1.642 -> 1.561 ms/tok`, `-0.080 ms/tok`) は採用。`native_down_nvfp4_36_t1_kernel` 改変は `+0.038 ms/tok` の退行実測により `ae9c00c72` へリバート。

---

### B. Decode DSpark (`dm=3..6`) 専用パス

- [x] **OPT-S1 | コールド Graph キャプチャ / `>100us` ホスト同期ギャップの解消** *(実装・採用)*
  - **現状コスト**: `w14` にて `g64-g1` 非カーネル差分 **`32.22 ms`** (`HtoD +18.64ms`, `GraphInstantiate 6.50ms/5c`, `GraphLaunch 9.26ms/33c`, `StreamSync 382ms/23c`)。`w6f` では `>100us` ギャップ 7回だけで **`26.13 ms`**。
  - **方針**: 実運用・測定時に発生しやすい特定 `T` パターンの事前ウォームアップまたは定常ステップ間の不要な同期ポイント排除。

- [x] **OPT-S2 | 小M (`T=6..8`) 検証用 GEMM と SWA/Pre-Attn の整理** *(実装・採用)*
  - **現状コスト**: `volta_s884gemm_fp16_256x128` (`24.57 ms` @ `w14`, `128x128` `44.05 ms` @ `w13`)、`k_pre_attn_smallT_fused` (`21.53 ms` @ `w14`)、多数の `verify4/verify8` SWA Split-KV + `split_combine` (`8.02 ms`)。
  - **箇所**: [inkling2_engine.cu:8114-8126, 8283-8301](file:///home/ryu/Strata-v100/engine/src/inkling2/inkling2_engine.cu#L8283-L8301)
  - **結果**: 65K `dm=6` 4WL のステップあたり処理時間 (`ms/step`) が全4負荷平均 `-0.28 ms/step` 短縮 (`edit_rag 20.00->19.40`, `code 18.42->18.14`, `prose 19.62->19.63`, `japanese 18.45->18.22 ms/step`)。

---

### C. Prefill (`OFF`) パス（現在 `54.05 TF = 55.0%` → **`55.80 TF = 56.7%`** @ 65K / `63.1%` @ 262K）

- [x] **OPT-P1 | `wmma_grouped_gu_situ_128<40>` / `grouped_128<40>` の小Mグループ効率改善** *(実装・採用: 計 `-544.33 ms` @ 65K PP)*
  - **現状コスト**: 計 **`7,344.01 ms` (`45.1%`)** (`gu_situ<40>` `4,275.5ms` + `down<40>` `2,379.6ms` 等)。
  - **箇所**: [iq_kernels.cu:10995, 11308, 16864-16916](file:///home/ryu/Strata-v100/engine/src/kernels/cuda/iq_kernels.cu#L11308)
  - **結果**: `wmma_grouped_*_small_sm70_kernel<32/64>` 動的ディスパッチにより `65K PP g1` 上で GU `-358.86 ms` + Down `-185.47 ms` = **`-544.33 ms`** 削減を達成。

- [x] **OPT-P2 | PP ループ制御・Router SplitK 補助カーネルの統合** *(実装・採用: 計 `-99.22 ms` @ 65K PP)*
  - **現状コスト**: `k_control_up_combine_pp_wmma_vmsg<3>/<1>` (`333.37ms`) + `k_control_down_phaseA` + reduce (`204.38ms`) + `k_router_sgemm_f32_pp_splitk` + reduce (`234.76ms`) = 計 **`772.51 ms` (`4.7%`)**。
  - **結果**: `vmsg<3>` `-37.95 ms` + `phaseA` `-31.45 ms` + `reduce` `-21.88 ms` + `ozaki2` router `-7.94 ms` = **`-99.22 ms`** 削減を達成。

- [x] **OPT-P3 | PP 非TC Norm / Mix / Conv / Dequant 群の縦融合** *(実装・採用・次波課題あり)*
  - **現状コスト**: 合計 **`~931 ms` (`5.7%`)**（`shortconv_mix_normed`、`postffn_shortconv`、`dequant_q8_0_h16_fast` `54.25ms`、`k_rms_norm_f16` `48.89ms` 等）。
  - **結果・補足**: `k_rms_norm_f16` (`48.89 -> 1.97 ms`) は削減できた一方、`k_control_up_combine_rms_f16_vmsg<1>` (`222.99 ms`) は元の `vmsg<1>` (`110.61 ms`) + `k_rms_norm_f16` 合算より `+65.46 ms` 増加しており、次波で非融合化または WMMA 化することで追加 `-65 ms` 回収可能。

- [x] **OPT-P4 | `cutlass` → `splitKreduce_kernel` 間の大ギャップ圧縮** *(実装・採用)*
  - **現状コスト**: `w6f` PP 窓で **`14.39 ms`** (`48c × ~300 us/回`)。
  - **方針**: cuBLAS/CUTLASS SplitK が不要な形状での 1-pass GEMM 選択、または並列ストリーム化検討。

---

## 5. Wave 16 (`4e900b216`) 実装・実測結果 (GPU 0 `V100-PCIE-32GB`)

基準 (`ae9c00c72`) から Wave 16 実装後に T=1 退行4項目 (`OPT-D1/D2`, `D4/D6`, `D7`, `D9-down`) を外科的リバートし、勝ち筋 (`OPT-D3`, `D5`, `D8`, `D9-GU`, `S1/S2`, `P1..P4`) のみを残した確定ビルド `4e900b216` の GPU 0 単独 (`CUDA_VISIBLE_DEVICES=0`、GPU 1 未接触) 実測結果。

### 5.1 GPU 0 速度 & Roofline 更新 (`ae9c00c72` → `4e900b216`)

| ワークロード | フェーズ (`dm`) | `ae9c00c72` | `4e900b216` | 変化率 | 実効帯域 / 計算力 (上限比) |
|---|---|---:|---:|---:|---|
| **4-WL** (`~4.6K`, 8-run平均) | **Prefill** (`OFF`) | `4,610.7 t/s` | **`4,851.3 t/s`** | **+5.2%** | **`51.23 TF` (`52.1%`)** *(greedy平均 `4,355.6→4,672.9`, samp平均 `4,865.8→5,029.7`)* |
| **Deep 65K** | **Prefill** (`OFF`) | `3,996.0 t/s` (`16.27s`) | **`4,125.7 t/s`** (`15.75s`) | **+3.2%** | **`55.80 TF` (`56.7%`)** *(from `54.05 TF / 55.0%`)* |
| **Deep 262K** | **Prefill** (`OFF`) | `2,611.8 t/s` (`100.12s`) | **`2,675.2 t/s`** (`97.75s`) | **+2.4%** | **`62.03 TF` (`63.1%`)** *(from `60.56 TF / 61.6%`)* |
| **4-WL** (`~4.6K`, 8-run平均) | **Decode** (`OFF`) | `103.52 t/s` | **`104.89 t/s`** | **+1.3%** | **`442.8 GB/s` (`51.4%`)** *(from `437.0 GB/s / 50.8%`)* |
| **Deep 65K** | **Decode** (`OFF` Steady) | `85.60 t/s` | **`86.47 t/s`** | **+1.0%** | **`451.7 GB/s` (`52.5%`)** *(Greedy1 `85.20→86.27`, Samp `85.56→86.36`)* |
| **Deep 262K** | **Decode** (`OFF` Steady) | `59.59 t/s` | **`59.96 t/s`** | **+0.6%** | **`509.5 GB/s` (`59.2%`)** *(Greedy1 `59.45→59.79`, Samp `59.60→60.01`)* |

- **DSpark (`dm=3 / dm=6`) 実測メモ**:
  - **Prefill**: 全ワークロード・全 `dm` で `+2.3%〜+5.1%` 向上（例: 65K `dm=3` `3967.5→4102.6 t/s`, `dm=6` `3958.5→4096.6 t/s`; 262K `dm=3` `2606.1→2667.7 t/s`, `dm=6` `2606.4→2667.4 t/s`）。
  - **Decode CLI 4WL (`ms/step` 実測)**: `dm=6` は 4WL 平均 **`-0.28 ms/step`** 短縮（`edit_rag 20.00→19.40`, `code 18.42→18.14`, `prose 19.62→19.63`, `japanese 18.45→18.22 ms/step`）、`dm=3` は平均 `0.00 ms/step`（誤差範囲）。シーケンスごとの `t/s` 上下（例: 65K `dm=3` Samp `119.40→143.35 (+20.1%)`, 4WL `dm=6 prose greedy 141.44→168.47 (+19.1%)`, 262K `dm=3 steady 100.78→83.39 (-17.3%)`）は FP 累積順序変化によるドラフト受理長 (`tok/step`) の揺らぎに起因する。

### 5.2 `nvprof` 純 GPU カーネル差分 (GPU 0 `65K OFF`)

- **Decode (`w15 -> 4e900b216`, `dc>0` 純カーネル計)**: `12.125 -> 11.903 ms/tok` (**`-0.222 ms/tok`**; `nvprof` Wall `12.853 -> 12.600 ms/tok`, `-0.253 ms/tok`)
  - `OPT-D5` (`k_loop_ctrl_fused_rms_q8_t1<3,32>/<1,64>`): `0.875 -> 0.751 ms/tok` (**`-0.124 ms/tok`**, `-48 launches/tok`)
  - `OPT-D9-GU` (`native_gu_nvfp4_72_situ_q8_1_t1_kernel<1>`): `1.642 -> 1.561 ms/tok` (**`-0.080 ms/tok`**)
  - `OPT-D8` (`k_router_gemv_and_group_t1<2>`): `0.496 -> 0.477 ms/tok` (**`-0.019 ms/tok`**)
- **Prefill 65K `g1` (`w7bfix -> 4e900b216`, 純カーネル計)**: `16,291.57 -> 15,742.17 ms` (**`-549.40 ms` / `-3.37%`**; 全 GPU 活動合計では `base` `25,061.80 -> 18,683.70 ms`, `-6,378.10 ms`)
  - `OPT-P1` (`wmma_grouped_*_small_sm70_kernel<32/64>`): GU **`-358.86 ms`** + Down **`-185.47 ms`** = **計 `-544.33 ms`**
  - `OPT-P2` (`vmsg<3>` `-37.95 ms` + `phaseA` `-31.45 ms` + `reduce` `-21.88 ms` + `ozaki2` router `-7.94 ms`): **計 `-99.22 ms`**
  - **次波の伸びしろ発見**: `OPT-P3` の `k_control_up_combine_rms_f16_vmsg<1>` (`222.99 ms`) が元の `vmsg<1>` (`110.61 ms`) + `k_rms_norm_f16` (`48.89 -> 1.97 ms`) 比で `+65.46 ms` 増加しており、ここを非融合または WMMA 化することで追加約 `-65 ms` を無償で回収可能。

### 5.3 品質・検証ゲート (`4e900b216`)

- **ビルド & スピル**: `engine/build-sm70` / `engine/build-sm89` ともにビルド成功、`libstrata_inkling2.a` および変更 `nvfp4`/`ktile` カーネルのスタックスピル `0`、`settings_register.py --check == 76`（新規 env var 0）。
- **`compute-sanitizer --tool memcheck`** (GPU 0): Case A (`OFF T=1`) / Case B (`DSpark dm=6`) / Case C (`PP T=128`) すべて **`ERROR SUMMARY: 0 errors`**。
- **Logit パリティ** (`STRATA_NO_CUBLASLT_TUNE=1 STRATA_NO_MOE_ATOMIC=1`, vs `ae9c00c72`): `L1` `max_abs=0.108644 cos=0.999996755 arg=12813==12813`, `L4` `0.000000`, `L7` `0.000000`, `L512` `max_abs=3.574849 cos=0.999564616 arg=200008==200008`。
