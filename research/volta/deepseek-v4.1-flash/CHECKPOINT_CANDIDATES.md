# DeepSeek-V4.1-Flash — checkpoint candidates for the 1Cat SM70 port

> **Origin.** PLI Labs research archive, file `v100-research/deepseek-v4.1-flash/CHECKPOINT_CANDIDATES.md`. Published 2026-11-18 in the Volta V100 research
> repository.
> **Edited for publication.** Machine names, private file paths, network addresses, session names and the
> operator's name were replaced with generic terms (`the V100 server`, `[private path]`, `[address]`,
> `the owner`). Links to files that are not published were turned into plain text. No figure and no finding
> was changed. Dates inside the note are its own; later results are in the build pages of this repository.

Lane L-MODEL, 2026-10-06. For every safetensors candidate the **headers of all 48 shards were range-read** and
censused per class (`[private path]>/census.json`, script `scripts/cand_census.py`; each
header set verified against the HF LFS file sizes). Configs, quantization configs, READMEs and indexes are in
`cand/<repo>/files/`. Shard identity vs the official checkpoint was checked by **LFS sha256 oid** per shard.
GGUF candidates: tree listings, READMEs, and (kernelpool) a parsed GGUF header (`l-model/gguf/kernelpool.json`).
Abliterated/uncensored variants skipped as non-reference.

Reference: `deepseek-ai/DeepSeek-V4.1-Flash` @ `2cba9e42aa026125f3ed06c6d98c1db82f7ca027` — 510,286,023,000 tensor
bytes (475.24 GiB); weights unchanged since the first upload (identical shard oids at `517ef625`, `fb2764a5`,
`dba1be0a`). Now downloading to `[...]/DeepSeek-V4.1-Flash`.

## 1. Safetensors candidates — per-class bytes (GiB, header-exact)

"GPU-resident" = everything except Engram tables, MTP/DSpark and vision (i.e. backbone layers + embed + head).

| Repo | Revision | Total | Routed experts | Engram tables | Dense (non-expert) | MTP | **GPU-resident** | Shards identical to official |
|-|-|-:|-:|-:|-:|-:|-:|-:|
| **deepseek-ai (official)** | `2cba9e42aa02` | 475.24 | 268.95 MXFP4 | 188.83 FP8 | 6.70 | 7.39 | **278.12** (298,624,579,008 B) | 48/48 |
| RedHatAI/DeepSeek-V4.1-Flash | `37119b250584` | 475.24 | = official | | | | | 48/48 (mirror) |
| nvidia/DeepSeek-V4.1-Flash-NVFP4 | `3431dde3247c` | 491.06 | 284.77 NVFP4 | 188.83 FP8 | 6.70 | 7.39 (MXFP4) | 293.94 | 6/48 |
| s-zaizen/DeepSeek-V4.1-Flash-NVFP4 | `179b7cda2548` | 491.06 | 284.77 NVFP4 | 188.83 FP8 | 6.70 | 7.39 (MXFP4) | 293.94 | 8/48 (6/48 identical to NVIDIA's) |
| LibertAIDAI/DeepSeek-V4.1-Flash-NVFP4 | `dfce15b92ed1` | 399.90 | 284.77 NVFP4 | 97.28 MXFP4 | 6.70 | 7.78 (NVFP4) | 293.94 | 0/48 |
| aidendle94/DeepSeek-V4.1-Flash-NVFP4-Engram | `efd61f3ac263` | 389.41 | 268.95 MXFP4 | **103.00 NVFP4** | 6.70 | 7.39 | 278.12 | **46/48** (only Engram shards differ) |
| amd/DeepSeek-V4.1-Flash-Quark-MXFP4 | `0d56db2133ce` | 474.58 | 268.95 MXFP4 | 188.83 FP8 | 6.09 | 7.34 | 277.50 | 3/48 |
| intel-ai/DeepSeek-V4.1-Flash-MXFP4-Mixed-CT-AutoRound | `3fdb0aa7c3e8` | 475.45 | 268.95 MXFP4 | 188.83 FP8 | 6.90 | 7.40 | 278.31 | 1/48 (renamed metadata) |
| INCModel3/DeepSeek-V4.1-Flash-MXFP4-Engram-AutoRound | `e610b50533d6` | 383.89 | 268.95 MXFP4 | 97.28 MXFP4 | 6.90 | 7.40 | 278.31 | 3/48 |
| lvkaokao/DeepSeek-V4.1-Flash-MXFP4-Engram-AutoRound | `8e136b119adf` | 383.69 | 268.95 MXFP4 | **97.28 MXFP4** | 6.70 | 7.39 | 278.12 | **46/48** (only Engram shards differ) |
| INCModel3/DeepSeek-V4.1-Flash-W4A16-Engram-AutoRound | `04f286e80267` | 420.62 | 292.68 INT4 | 103.00 INT4 | 13.07 (BF16) | 8.51 | 308.21 | 0/48 |
| lvkaokao/DeepSeek-V4.1-Flash-W4A16-Engram-AutoRound | `1429632e8d96` | 420.62 | = INCModel3 W4A16 (48/48 identical oids) | | | | 308.21 | 0/48 |

Per-GPU weight bytes (GPU-resident ÷ GPUs, even split; Engram/MTP/vision excluded): official **23.18 GiB on 12,
34.76 GiB on 8**; NVFP4 variants 24.50 / 36.74; W4A16 25.68 / 38.53. Usable budget ≈ 29.5 GiB/GPU.

## 2. Format details per candidate

| Repo | Experts | Attention / dense | Shared expert | Engram table | embed / head | MTP experts | Scales / groups | Tooling to run | Quality claims |
|-|-|-|-|-|-|-|-|-|-|
| **official** | E2M1 packed `I8 [N,K/2]` + `F8_E8M0 [N,K/32]` (MXFP4, **4.25 bpw**) | FP8 E4M3 + E8M0 per **32×32** block | FP8 32×32 | FP8 E4M3 `[R,256]` + E8M0 `[R,8]` | BF16 | MXFP4 | UE8M0 pow2 | reference TileLang (SM89+), vLLM ≥ 0.30, SGLang ≥ 0.5.21 | DeepSeek's own evals |
| nvidia NVFP4 | E2M1 `U8` + `weight_scale` E4M3 per **16** + `weight_scale_2` FP32 per tensor + calibrated `input_scale` (W4A4), **4.5 bpw**; ModelOpt `dsv4-nvfp4-experts` (v0.47.0rc0), source rev `dba1be0a` | unchanged FP8 (`exclude *.attn.*`) | unchanged | unchanged FP8 | unchanged | unchanged MXFP4 | E4M3/16 + FP32 global | vLLM `deepseekv41-flash-0909` image / SGLang `dev-cu13-dsv41`, GB300 (Blackwell only) | weights converted **losslessly** ("all 16,986,931,200 weight blocks"); vs MXFP4 source: GPQA 91.29 vs 91.04, AA-LCR 78.44 vs 78.56, SciCode 55.84 vs 54.40, IFBench 77.27 vs 76.67, MMMU-Pro 73.70 vs 74.05, TB2.1 82.16 vs 81.60 |
| s-zaizen NVFP4 | as NVIDIA (ModelOpt MXFP4→NVFP4 cast, `input_scale = 1.0`, no calibration), source rev `fb2764a5` | unchanged | unchanged | unchanged | unchanged | unchanged | | none claimed | none run |
| LibertAIDAI NVFP4 | NVFP4 bit-exact transcode (0 mismatches over 2.36e9 elements), **no input_scale** | unchanged | unchanged | **E2M1 `U8 [R,128]` + E8M0/32**, scale = pow2 ≥ amax/6 (no clipping) | unchanged | NVFP4 | | custom (`dsv41_fp4_stream.py`); "no engine can run it" at release | Engram: mean cosine 0.99342/0.99345 (1.5 M rows); no end-to-end eval |
| aidendle94 NVFP4-Engram | official MXFP4 (byte-identical) | official | official | **NVFP4: E2M1 `U8 [R,128]` + E4M3 `[R,16]` + FP32 global per table** (`config.json: engram_quant_global_scale {1: 0.004092, 14: 0.008929}`); GPTQ error feedback through `wkv` + per-block MSE scale search | official | official | E4M3/16 | patched vLLM for 4× DGX Spark, rows read from shards on disk (`DSV41_ENGRAM_DISK=1`) | rel RMS 8.99 % / 9.06 %; teacher-forced logprob −0.111 vs −0.112, argmax agreement 97.5 % = FP8; needle 32K/128K found; tool call pass; decode 87.3/66.4/27.1 vs 90.7/63.5/31.3 tok/s (NFS rows) |
| amd Quark MXFP4 | MXFP4 (renamed `weight`/`weight_scale` U8), W4A4 OCP MX | FP8 E4M3 + E8M0 kept (template excluded `*attn*`) | **re-quantized FP8 → MXFP4 (lossy)** | unchanged FP8 | unchanged | MXFP4 | E8M0/32 | AMD Quark 0.13, vLLM ROCm image, MI355 | GSM8K 92.34 vs 92.87; GPQA 90.40 vs 89.39 (±2.33 % CI) |
| intel-ai Mixed-CT | bit-identical MXFP4 (`weight_packed`) | bit-identical FP8, scales expanded 32×32 → OCP 1×32 (`compressed-tensors` mixed) | same | unchanged FP8 | | | E8M0/32 rowwise | one vLLM patch on Blackwell | bit-identical weights (claims) |
| INCModel3 MXFP4-Engram | official MXFP4 bytes | MXFP8 rowwise 1×32 (lossless scale expansion) | same | **MXFP4 `I8 [R,128]` + E8M0/32** (RTN, rel err ≈ 0.12) | | | | vLLM main ≥ #56201 + `dsv41-quant-plugin`, `DSV41_ENGRAM_DTYPE=fp4` | gsm8k 92.19/92.27 vs 92.87 (sibling) |
| lvkaokao MXFP4-Engram | **official (byte-identical shards)** | official | official | **MXFP4 `I8 [R,128]` + E8M0 `[R,8]` under the original names** (auto-round RTN, rel err ≈ 0.12) | official | official | E8M0/32 | same plugin | gsm8k 92.19/92.27 vs 92.87 (twin) |
| INCModel3 / lvkaokao W4A16 | **INT4 sym g32, auto_gptq packing** (`qweight I32`, `qzeros I32`, `scales F16`) — lossy re-quant of MXFP4 (E2M1 grid ⊄ INT4 grid) | **BF16** (dequantized, 2× bytes) | BF16 | INT4 sym g32 `U8 [R,128]` + **FP16** scale `[R,8]` | BF16 | INT4 | fp16/32 | vLLM MoeWNA16/Marlin + plugin | gsm8k 93.93/94.01 vs 92.87 (single benchmark) |

## 3. GGUF and DSpark artifacts (fallback / comparison only)

| Repo | Revision | Files | Experts | Engram | Notes |
|-|-|-|-|-|-|
| smalinin/DeepSeek-V4.1-Flash-GGUF | `fb2ce0313f74` | 5 variants × 10 shards (MXFP4 507.95 GB; Q2_K-Q5 335.38 GB; IQ2_XXS/Q2_K-Q8 371.44 GB; Protected Q2_K-Q8 413.39 GB; IQ3_XS-Q8 434.02 GB) + DSpark-AUTO 7.97 GB + mmproj 0.97 GB + imatrix (262,144 tok) | MXFP4 (120 tensors) in the MXFP4 variant; dense Q8_0 | **Q8_0** (or Q5_K) — re-quantized from FP8 | runtime: `smalinin/llama.cpp` branch `my_build_deepseek41`, CUDA only; Protected Q2_K tg32 35.7 tok/s on 6 GPUs (their claim) |
| JigSawPT/DeepSeek-V4.1-Flash-GGUF ("MXFP4-engram") | `be6653aeb554` | 11 shards, 501.81 GB | lossless MXFP4 repack (480/480 blocks byte-identical) | **raw FP8 bytes + scales** (separate tensors, shards 2–4) | runtime `JigSawPT/llama.cpp` branch `dsv41-porte`; RTX 5090 + NVMe streaming: 5.1 tok/s new content, 21.4 resident; logit corr 0.9967 vs reference at 1,401 tokens; report repo `JigSawPT/deepseek-v41-flash-on-5090` @ `dbbdb887` |
| JigSawPT/DeepSeek-V4.1-Flash-DSpark-GGUF | `0fe521efb6e3` | 1 × 7.97 GB | DSpark MXFP4 experts, dense Q8_0/BF16 | — | `dflash.dsv41_semantics` key (no q-norm, shifted mHC, collapse with last FFN mix); acceptance 51 %/79 %/97 % (prose/code/verbatim); best draft length 2 on an I/O-bound box |
| kernelpool/DeepSeek-V4.1-Flash-MXFP4-GGUF | `22a073e51781` | `.gguf.part1` 480.0 GB + `.part2` 21.6 GB, DSpark-MXFP4 7.97 GB; no README | MXFP4 (120 tensors), dense Q8_0, hc/indexer/compressor F16 | **FP8 interleaved 264-B rows**: I8 `[264, rows]`, `deepseek41.engram.encoding = "e4m3_e8m0_32_row264"`; metadata carries the 48 primes, 8 multipliers and the 129,280-entry token map | arch `deepseek41`; useful design reference for a host-gather row layout and as a cross-check of hash constants (validated here) |
| antirez/deepseek-v4.1-flash-gguf | `dd8a266f7145` | Q2 365.71 GB; Q4 518.60 GB (2 parts); vision 0.97 GB | Q2: IQ2_XXS gate/up + Q2_K down; Q4: Q4_K (re-quantized, lossy), 8,192-token imatrix | **native FP8** | DwarfStar (Metal) only; source rev `df42c109` |
| vcruz305/DeepSeek-V4.1-Flash-GGUF | — | Q4_K_M 415 GB | FP4 → Q4_K (lossy) | quantized `engram_q/k` | rejected as reference in the plan |

## 4. Other notable artifacts

- **Tokha233 "deepseek-v4.1-flash-a100-turbo"** (HF mirror `StellarVoyager/DeepSeek-V4.1-Flash-A100-A800-SM80` @
  `f4dfddab1d7d`; GitHub `Tokha233/deepseek-v4.1-flash-a100-turbo`): official weights served on **8× A800 (SM80, no
  FP8 tensor cores)** with `wtdcode/vllm-backport` (`master-v013`) + patches: "FP4 Marlin experts · BF16 dense dispatch ·
  FP8 MLA KV (`--kv-cache-dtype fp8_ds_mla`)", `--engram-config {"cpu_offload": true}`, TP4×DP2, EP8, DSpark 5, CUDA
  graphs FULL_DECODE_ONLY, 1M context; patches include `vllm/v1/attention/ops/candidate_mqa_sm80.py` (Triton
  candidate-only index logits via an E4M3→BF16 LUT) and `model_executor/kernels/linear/mxfp8/marlin.py`. Claims 3,040
  tok/s (C128 short), SWE-bench Verified 87/100 subset. **This is the closest prior art for a pre-FP8 GPU** — the
  dequant-to-16-bit strategy maps directly onto V100 (FP16 instead of BF16); Triton pieces need SM70 review.
- Weights mirror: `RedHatAI/DeepSeek-V4.1-Flash` (`37119b250584`) is byte-identical (48/48 oids) — an alternative
  download source.

## 5. Mapping onto 1Cat's V4 MXFP4-experts + FP8 path on V100 (fused dequant → FP16)

Facts that decide it:
1. **Experts.** The official experts use exactly V4-Flash's expert encoding (I8-packed E2M1, E8M0 per 32 along K;
   V4-Flash header census: `w1 [2048,2048] I8 + [2048,128] E8M0`). Measured exponents −12…−2 ⇒ every expert weight
   dequantizes **exactly** into FP16. NVFP4 adds a second-level FP32 scale and E4M3 sub-scales for **zero quality gain
   here** (all NVFP4 variants were lossless transcodes of these same MXFP4 values) and +15.82 GiB (+5.9 %) of HBM;
   their W4A4 activation path is Blackwell-only. INT4 W4A16 is a lossy re-quantization that also doubles dense bytes.
2. **Dense.** FP8 32×32 blocks with exponents −13…−6 ⇒ exact FP16 dequant (only `engram.wkv` needs a scale bias).
   1Cat's V4 FP8 path was built for 128×128 blocks — it needs 32-block support, nothing format-wise new.
3. **HBM.** Official GPU-resident 278.12 GiB: 12 GPUs → 23.2 GiB/GPU average (PP3×TP4 13/14/13: 22.7/24.1/22.7),
   leaving ≈ 5–7 GiB/GPU for KV (tiny: 890 B/token model-native), indexer, workspace, graphs. 8 GPUs → 34.8 GiB/GPU
   = ≥ 42 GiB of expert spill required. Any NVFP4/INT4 variant makes both cases worse.
4. **Engram.** Only the 4-bit-Engram variants change what fits in host RAM: 97.3 GiB (MXFP4) / 103.0 GiB (NVFP4,
   INT4) vs 188.8 GiB (FP8) on a 196 GiB host. The two "drop-in" ones keep all 46 other shards byte-identical
   to the official download: **lvkaokao MXFP4-Engram** (RTN, rel err ≈ 0.12, official tensor names/layout) and
   **aidendle94 NVFP4-Engram** (GPTQ+MSE, rel RMS ≈ 9 %, the only one with a teacher-forced end-to-end check).
5. **Quality / long context.** No candidate changes attention, indexer or KV formats (KV QAT is model code), so
   long-context behavior is identical across them; the only quality variables are the Engram precision (4-bit
   variants) and expert re-quantization (INT4, AMD shared experts).

### Recommendation

1. **Primary and production checkpoint: the official `deepseek-ai/DeepSeek-V4.1-Flash` @ `2cba9e42`** (already
   downloading). It maps 1:1 onto 1Cat's V4 MXFP4-expert + FP8-dense path (exact FP16 dequant, smallest HBM footprint
   of all candidates, the correctness reference by construction). Load-skip `vision.*`/`aligner.*`/`image_*`
   (0.90 GiB) and `bias_vl` for text-only.
2. **Engram:** serve the official FP8 tables from the **root E18 NVMe** (repacked once into interleaved 264-B rows,
   202.8 GB; nvme2 measured only 7.3 k random-read IOPS at QD32) with host-side async gather + hot-row cache
   (ENGRAM.md §5, §9). In parallel, evaluate a 4-bit Engram for the 12-GPU production layout — it becomes the
   practical default if the root disk cannot hold the FP8 row file:
   fetch only **shards 47 and 48 of `aidendle94/DeepSeek-V4.1-Flash-NVFP4-Engram` @ `efd61f3ac263`** (55.45 GB +
   55.46 GB; best documented quality) and/or of **`lvkaokao/DeepSeek-V4.1-Flash-MXFP4-Engram-AutoRound` @
   `8e136b119adf`** (52.38 GB each; simplest format: E2M1 + E8M0/32, same decode as the experts) — or quantize locally
   from the FP8 tables. Adopt only if a teacher-forced logprob/argmax comparison vs FP8 Engram on our prompts shows
   parity (aidendle94's 40-completion test is the bar). Note: each of these shards is > 200 MB — that download is a
   decision for the coordinator, not this lane.
3. **Do not use** NVIDIA/s-zaizen/LibertAIDAI NVFP4 (lossless re-encodes of the same experts, bigger, Blackwell-only
   benefits; LibertAIDAI's MTP experts are also NVFP4), AMD Quark (lossy shared experts), or W4A16 (lossy experts,
   BF16 dense, +30 GiB GPU-resident).
4. **GGUFs** only as numerics cross-checks (JigSawPT's port has a published 0.9967 logit correlation vs the reference);
   kernelpool's `row264` Engram layout and metadata constants are a useful design reference. DSpark needs no separate
   artifact — `mtp.*` in the official checkpoint is the draft model.
5. **SM80 prior art** (Tokha233/StellarVoyager) is the most relevant engine reference for "no FP8 tensor cores":
   Marlin FP4 experts + 16-bit dense + CPU-offloaded Engram + FP8-stored MLA KV — the same shape as our V100 plan.
