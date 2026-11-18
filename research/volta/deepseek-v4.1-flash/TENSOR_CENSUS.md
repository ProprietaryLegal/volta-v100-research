# DeepSeek-V4.1-Flash — exact tensor byte census

> **Origin.** PLI Labs research archive, file `v100-research/deepseek-v4.1-flash/TENSOR_CENSUS.md`. Published 2026-11-18 in the Volta V100 research
> repository.
> **Edited for publication.** Machine names, private file paths, network addresses, session names and the
> operator's name were replaced with generic terms (`the V100 server`, `[private path]`, `[address]`,
> `the owner`). Links to files that are not published were turned into plain text. No figure and no finding
> was changed. Dates inside the note are its own; later results are in the build pages of this repository.

Lane L-MODEL, 2026-10-06. Source: `deepseek-ai/DeepSeek-V4.1-Flash` @ `2cba9e42aa026125f3ed06c6d98c1db82f7ca027`.
Machine-readable: `tensor_census.json` (aggregates, per-layer, stage splits, every
non-routed-expert tensor) and `tensor_census_full.tsv.gz` (all 96,085 tensors:
name, dtype, shape, bytes, shard, class, layer, params).

## Method

For each of the 48 shards: HTTP range read `bytes=0-7` (little-endian u64 header length), then
`bytes=8-(8+len-1)` (header JSON), from `https://huggingface.co/<repo>/resolve/<rev>/<shard>`. Every tensor's
`data_offsets` span was checked against `numel × dtype size` (no mismatch), and `8 + header_len + data_end`
was checked against the HF LFS file size for every shard: **48/48 match**. Scripts:
`[...]/{fetch_headers.sh,census.py,census_report.py}`; raw headers in
`[private path]`.

| Quantity | Value |
|-|-|
| Tensors | 96,085 |
| Tensor bytes (sum of data spans) | **510,286,023,000 B = 475.241 GiB** (= index `total_size`, exact) |
| File bytes (incl. 48 headers) | 510,296,708,312 B |
| Largest shard | `model-00047` = layer-1 Engram (101,535,150,272 data bytes in one file); `model-00048` = layer-14 Engram |

## Dtype codes in the headers

| Header dtype | Count | Meaning | Used for |
|-|-|-|-|
| `I8` | 47,232 | **packed FP4 E2M1, two values per byte**, packed along K (in-features). Low nibble = even K index, high nibble = odd (decode table in `inference/convert.py:13-15,31-34`); reference views it as `torch.float4_e2m1fn_x2` (`convert.py:181`) | routed experts only (backbone + MTP) |
| `F8_E8M0` | 47,589 | UE8M0 power-of-two scale (`torch.float8_e8m0fnu`) | every scale: experts (1 per 32 along K), FP8 dense (1 per 32×32 block), Engram rows (1 per 32 channels) |
| `F8_E4M3` | 357 | FP8 E4M3FN | dense FP8 linears, shared experts, Engram tables, Engram `wkv`, DSpark `main_proj` |
| `BF16` | 520 | bf16 | embed, head, norms, gate.weight, compressor wkv/wgate, indexer wk/weights_proj, engram q/k_weight, markov head, vision |
| `F32` | 387 | fp32 | `hc_*`, `attn_sink`, `gate.bias`, `gate.bias_vl` |

There is no `F4`/`U8`/`F8_E5M2` dtype in the official checkpoint. Quantization config (`config.json`):
`quant_method=fp8, weight_block_size=[32,32], scale_fmt=ue8m0, activation_scheme=dynamic, expert_dtype=fp4`.

## Bytes by class (official checkpoint)

| Group : class | Bytes | GiB | Tensors | Params | dtypes |
|-|-:|-:|-:|-:|-|
| backbone : routed_experts.weight (FP4) | 271,790,899,200 | 253.1250 | 46,080 | 543,581,798,400 | I8 |
| backbone : routed_experts.scale | 16,986,931,200 | 15.8203 | 46,080 | — | E8M0 |
| **backbone routed experts total** | **288,777,830,400** | **268.9453** | | | |
| backbone : engram.embed.weight | 196,613,849,600 | 183.1109 | 2 | 196,613,849,600 | E4M3 |
| backbone : engram.embed.scale | 6,144,182,800 | 5.7222 | 2 | — | E8M0 |
| **Engram tables total** | **202,758,032,400** | **188.8331** | | | |
| backbone : engram.wkv (+scale) | 314,880,000 | 0.2933 | 4 | 314,572,800 | E4M3+E8M0 |
| backbone : engram.q_weight/k_weight | 163,840 | 0.0002 | 4 | 81,920 | BF16 |
| backbone : attn.wq_a (+scale) | 262,400,000 | 0.2444 | 80 | 262,144,000 | E4M3+E8M0 |
| backbone : attn.wq_b (+scale) | 1,679,360,000 | 1.5640 | 80 | 1,677,721,600 | E4M3+E8M0 |
| backbone : attn.wkv (+scale) | 104,960,000 | 0.0978 | 80 | 104,857,600 | E4M3+E8M0 |
| backbone : attn.wo_a (+scale) | 1,343,488,000 | 1.2512 | 80 | 1,342,177,280 | E4M3+E8M0 |
| backbone : attn.wo_b (+scale) | 1,679,360,000 | 1.5640 | 80 | 1,677,721,600 | E4M3+E8M0 |
| backbone : attn q_norm/kv_norm/attn_sink | 153,600 | 0.0001 | 120 | 74,240 | BF16/F32 |
| backbone : attn.compressor (wkv, wgate, norm) | 36,704,256 | 0.0342 | 11 | 18,352,128 | BF16 |
| backbone : attn.indexer (wq_b+scale, weights_proj, wk, k_norm) | 45,130,752 | 0.0420 | 32 | 43,516,416 | E4M3/E8M0/BF16 |
| backbone : shared_expert.weight | 1,415,577,600 | 1.3184 | 120 | 1,415,577,600 | E4M3 |
| backbone : shared_expert.scale | 1,382,400 | 0.0013 | 120 | — | E8M0 |
| backbone : gate.weight | 157,286,400 | 0.1465 | 40 | 78,643,200 | BF16 |
| backbone : gate.bias + gate.bias_vl | 122,880 | 0.0001 | 80 | 30,720 | F32 |
| backbone : hc_* | 157,295,040 | 0.1465 | 240 | 39,323,760 | F32 |
| backbone : attn_norm/ffn_norm | 819,200 | 0.0008 | 80 | 409,600 | BF16 |
| embed.weight | 1,323,827,200 | 1.2329 | 1 | 661,913,600 | BF16 |
| head.weight | 1,323,827,200 | 1.2329 | 1 | 661,913,600 | BF16 |
| norm.weight | 10,240 | 0.0000 | 1 | 5,120 | BF16 |
| **mtp.* (DSpark, 3 blocks) total** | **7,932,874,632** | **7.3881** | | 14,225,362,530 | |
| — of which mtp routed experts (128/blk, weight+scale) | 7,219,445,760 | 6.7236 | 2,304 | 13,589,544,960 | I8+E8M0 |
| — of which mtp dense (attn, shared, gate, hc, main_proj, markov, confidence) | 713,428,872 | 0.6644 | | | |
| **vision.* + aligner.* + image_{start,end,newline}** | **970,536,960** | **0.9039** | 266 | 485,268,480 | BF16 |

Group totals: backbone 498,734,946,768 B (464.483 GiB) · embed/head/norm 2,647,664,640 B (2.466 GiB) ·
mtp 7,932,874,632 B (7.388 GiB) · vision 970,536,960 B (0.904 GiB).

### Community census check (JigSawPT discussion figures)

| Claim | Header-exact | Verdict |
|-|-|-|
| routed experts MXFP4 268.9 GiB | 288,777,830,400 B = **268.945 GiB** (backbone; +6.724 GiB more in MTP) | **verified** |
| Engram tables 188.8 GiB | 202,758,032,400 B = **188.833 GiB** | **verified** |
| "rest" 9.6 GiB | backbone non-expert non-table + embed/head/norm = **9.171 GiB** (9,847,742,808 B); +MTP-dense 0.664 → 9.835; +vision 0.904 | **approximately; partition differs** (their 9.6 is not reproducible to the byte from any natural partition; use 9.171 GiB) |

## Parameter counts (header-derived; FP4 counted as 2 params per I8 byte)

| Quantity | Params | Note |
|-|-:|-|
| All tensors | 763,205,315,794 | |
| "Backbone" (all text weights excl. Engram tables, MTP, vision; incl. embed+head) | **551,880,835,184** | = README/report "552B backbone" |
| Engram tables | **196,613,849,600** | = report "196B Engram parameters" |
| Routed experts (backbone) | 543,581,798,400 | 35,389,440 per expert × 384 × 40 |
| DSpark/MTP | 14,225,362,530 | |
| Vision + aligner | 485,268,480 | |
| **Active per decoded token** (6/384 routed × 40 + all dense of 40 layers + head + 48 Engram rows) | **≈ 16.13 B** | report: 16B |
| **Active per prefill token (encoder only, layers 0–19)** | **≈ 7.89 B** | report: 8B (CED: decoder runs only on the last n_win=128 tokens) |

## Per-tensor shapes (one layer; all 40 layers identical except where noted)

| Tensor | dtype | shape | bytes |
|-|-|-|-:|
| `attn.wq_a` / `.scale` | E4M3 / E8M0 | [1280, 5120] / [40, 160] | 6,553,600 / 6,400 |
| `attn.wq_b` / `.scale` | E4M3 / E8M0 | [32768, 1280] / [1024, 40] | 41,943,040 / 40,960 |
| `attn.wkv` / `.scale` | E4M3 / E8M0 | [512, 5120] / [16, 160] | 2,621,440 / 2,560 |
| `attn.wo_a` / `.scale` | E4M3 / E8M0 | [8192, 4096] / [256, 128] | 33,554,432 / 32,768 |
| `attn.wo_b` / `.scale` | E4M3 / E8M0 | [5120, 8192] / [160, 256] | 41,943,040 / 40,960 |
| `attn.q_norm` / `kv_norm` / `attn_sink` | BF16 / BF16 / F32 | [1280] / [512] / [64] | 2,560 / 1,024 / 256 |
| `attn.compressor.wkv` (layers 2,8,14,20) | BF16 | [512, 5120] | 5,242,880 |
| `attn.compressor.wgate` (layers 2,8,14 only; ratio-1 layer 20 has none) | BF16 | [512, 5120] | 5,242,880 |
| `attn.compressor.norm` (2,8,14,20) | BF16 | [512] | 1,024 |
| `attn.indexer.wq_b` / `.scale` (2,8,14,20,24,28,32,36) | E4M3 / E8M0 | [4096, 1280] / [128, 40] | 5,242,880 / 5,120 |
| `attn.indexer.weights_proj` (same 8 layers) | BF16 | [32, 5120] | 327,680 |
| `attn.indexer.wk` / `k_norm` (2,8,14,20 only) | BF16 | [128, 512] / [128] | 131,072 / 256 |
| `ffn.gate.weight` / `bias` / `bias_vl` | BF16 / F32 / F32 | [384, 5120] / [384] / [384] | 3,932,160 / 1,536 / 1,536 |
| `ffn.shared_experts.w1,w3` / `.scale` | E4M3 / E8M0 | [2304, 5120] / [72, 160] | 11,796,480 / 11,520 each |
| `ffn.shared_experts.w2` / `.scale` | E4M3 / E8M0 | [5120, 2304] / [160, 72] | 11,796,480 / 11,520 |
| `ffn.experts.{0..383}.w1,w3` / `.scale` | I8(FP4×2) / E8M0 | [2304, 2560] / [2304, 160] | 5,898,240 / 368,640 each |
| `ffn.experts.{0..383}.w2` / `.scale` | I8(FP4×2) / E8M0 | [5120, 1152] / [5120, 72] | 5,898,240 / 368,640 |
| `hc_{attn,ffn}_fn` / `_base` / `_scale` | F32 | [24, 20480] / [24] / [3] | 1,966,080 / 96 / 12 each |
| `attn_norm`, `ffn_norm` | BF16 | [5120] | 10,240 each |
| `engram.embed.weight` (L1 / L14) | E4M3 | [384006168, 256] / [384016682, 256] | 98,305,579,008 / 98,308,270,592 |
| `engram.embed.scale` (L1 / L14) | E8M0 | [384006168, 8] / [384016682, 8] | 3,072,049,344 / 3,072,133,456 |
| `engram.wkv` / `.scale` | E4M3 / E8M0 | [25600, 6144] / [800, 192] | 157,286,400 / 153,600 |
| `engram.q_weight`, `k_weight` | BF16 | [4, 5120] | 40,960 each |
| `embed.weight`, `head.weight` | BF16 | [129280, 5120] | 1,323,827,200 each |
| `mtp.{0,1,2}` attn (wq_a, wq_b, wkv, wo_a, wo_b, norms, sink) | as backbone | as backbone | 126,758,656 each (no compressor/indexer) |
| `mtp.{0,1,2}.ffn.experts.{0..127}.*` | I8 / E8M0 | as backbone | 2,406,481,920 per block |
| `mtp.{i}.ffn.gate.weight` / bias / bias_vl | BF16 / F32 | [128, 5120] / [128] | 1,310,720 / 512 |
| `mtp.0.main_proj` / `.scale`, `mtp.0.main_norm` | E4M3 / E8M0, BF16 | [5120, 15360] / [160, 480], [5120] | 78,643,200 / 76,800, 10,240 |
| `mtp.2.markov_head.embed` / `.head` | BF16 | [129280, 256] each | 66,191,360 each |
| `mtp.2.confidence_head.proj` | BF16 | [1, 5376] | 10,752 |
| `mtp.2.norm` | BF16 | [5120] | 10,240 |

MTP has **no** `embed.weight`/`head.weight` (tied to the backbone's, `convert.py:108-110`, `model.py:1212-1213`).

Bits per weight: routed experts = (17,694,720 + 1,105,920) B × 8 / 35,389,440 = **4.25 bpw** (MXFP4: E2M1 +
one E8M0 per 32). Dense FP8 = 8 + 8/1024 = **8.008 bpw**. Engram rows = (256 + 8) B / 256 = **8.25 bpw**.

## Per-layer bytes (checkpoint-native; "GPU" = everything except the Engram tables)

`dense_fp16` = the same layer if every F8_E4M3 tensor were expanded to FP16 (scales dropped); BF16 counted as
FP16, F32 kept. Experts stay MXFP4 in both columns.

| Layer | Attention mode | Routed experts B | Dense B | **GPU native B** | GiB | GPU w/ dense FP16 B | Engram table B (host) |
|-|-|-:|-:|-:|-:|-:|-:|
| 0 | SWA only | 7,219,445,760 | 170,055,128 | 7,389,500,888 | 6.882 | 7,551,347,672 | 0 |
| 1 | SWA only +Engram | 7,219,445,760 | 327,577,048 | 7,547,022,808 | 7.029 | 7,866,002,392 | 101,377,628,352 |
| 2 | CSA2(r=2,Full) | 7,219,445,760 | 186,248,920 | 7,405,694,680 | 6.897 | 7,572,779,224 | 0 |
| 3–7 | CSA2(r=2,Reuse) | 7,219,445,760 | 170,055,128 | 7,389,500,888 | 6.882 | 7,551,347,672 | 0 |
| 8 | CSA2(r=2,Full) | 7,219,445,760 | 186,248,920 | 7,405,694,680 | 6.897 | 7,572,779,224 | 0 |
| 9–13 | CSA2(r=2,Reuse) | 7,219,445,760 | 170,055,128 | 7,389,500,888 | 6.882 | 7,551,347,672 | 0 |
| 14 | CSA2(r=2,Full) +Engram | 7,219,445,760 | 343,770,840 | 7,563,216,600 | 7.044 | 7,887,433,944 | 101,380,404,048 |
| 15–19 | CSA2(r=2,Reuse) | 7,219,445,760 | 170,055,128 | 7,389,500,888 | 6.882 | 7,551,347,672 | 0 |
| 20 | CSA2(r=1,Full) + candidate source | 7,219,445,760 | 181,006,040 | 7,400,451,800 | 6.892 | 7,567,536,344 | 0 |
| 21–23, 25–27, 29–31, 33–35, 37–39 | CSA2(r=1,Reuse) | 7,219,445,760 | 170,055,128 | 7,389,500,888 | 6.882 | 7,551,347,672 | 0 |
| 24, 28, 32, 36 | CSA2(r=1,Reindex) | 7,219,445,760 | 175,630,808 | 7,395,076,568 | 6.887 | 7,562,161,112 | 0 |
| 40 (mtp.0) | DSpark SWA + main_proj | 2,406,481,920 | 246,161,880 | 2,652,643,800 | 2.470 | 2,893,056,984 | 0 |
| 41 (mtp.1) | DSpark SWA | 2,406,481,920 | 167,431,640 | 2,573,913,560 | 2.397 | 2,735,760,344 | 0 |
| 42 (mtp.2) | DSpark SWA + Markov/confidence heads | 2,406,481,920 | 299,835,352 | 2,706,317,272 | 2.520 | 2,868,164,056 | 0 |

(Per-layer rows for every layer individually are in `tensor_census.json → derived.per_layer`.)

| Aggregate | Bytes | GiB |
|-|-:|-:|
| **Backbone GPU-resident, native** (layers 0–39 excl. Engram tables + embed + head + final norm) | **298,624,579,008** | **278.116** |
| same with all FP8 dense expanded to FP16 | 305,454,618,048 | 284.477 |
| MTP / DSpark (3 blocks, native; embed+head shared with backbone) | 7,932,874,632 | 7.388 |
| Engram tables (host-resident candidate) | 202,758,032,400 | 188.833 |
| Vision + aligner (text-only bring-up: load-skip) | 970,536,960 | 0.904 |

The reference `convert.py:157-173` dequantizes `wo_a` to BF16 (+33.5 MB per layer, +1.25 GiB total); the
numbers above are checkpoint-native.

## Pipeline-stage splits (native bytes, weights only — no KV, workspace, CUDA context)

Embed (1.233 GiB) on stage 0, head + final norm (1.233 GiB) on the last stage. TP4 per-GPU = stage / 4
(experts split evenly by EP or TP; Engram tables excluded, MTP excluded).

| Layout | Stage layers | Stage GiB | per GPU (TP4) GiB | with FP16 dense, per GPU GiB |
|-|-|-:|-:|-:|
| 8 GPU PP2×TP4 | 0–19 | 139.21 | **34.80** | 35.63 |
| | 20–39 | 138.90 | **34.73** | 35.49 |
| 12 GPU PP3×TP4 (13/14/13) | 0–12 | 90.88 | **22.72** | 23.25 |
| | 13–26 | 96.53 | **24.13** | 24.70 |
| | 27–39 | 90.71 | **22.68** | 23.17 |
| 12 GPU PP3×TP4 (14/13/13) | 0–13 | 97.76 | 24.44 | 25.01 |
| | 14–26 | 89.64 | 22.41 | 22.94 |
| | 27–39 | 90.71 | 22.68 | 23.17 |
| 12 GPU, (13/13/14) | 0–12 / 13–25 / 26–39 | 90.88 / 89.64 / 97.60 | 22.72 / 22.41 / 24.40 | |

Consequences (arithmetic, not measurement):
- **8 GPUs:** 34.7–34.8 GiB/GPU of weights alone vs ~29.5 GiB usable → ≥ 5.3 GiB/GPU (≈ 42.4 GiB total) of
  routed experts must spill to host *before* any KV/workspace: ≈ 1.06 GiB per layer ≈ 15.8 % of expert
  bytes ≈ 61 of 384 experts per layer (18,800,640 B per expert). With ~2–3 GiB/GPU of KV + workspace the
  spill grows to ≈ 80–90 experts/layer. The 8-GPU phase needs a deliberate expert-offload path (cold experts
  in pinned host RAM), sized from measured free HBM.
- **12 GPUs:** 22.4–24.4 GiB/GPU native leaves ~5–7 GiB/GPU for KV, indexer, workspace, NCCL, graphs.
  Adding DSpark (7.39 GiB) to the last stage (13/14/13 split) gives 98.1 GiB → 24.5 GiB/GPU; DSpark needs
  `embed` too (tied) — either duplicate 1.23 GiB on the last stage or ship the draft embeddings from stage 0.
- Global KV is tiny (890 B/token in the model's FP4/FP4/FP8 layout; see MODEL_ARCHAEOLOGY §4.7) — HBM, not KV,
  is spent on weights.

## TP/kernel alignment facts for the port

- Expert intermediate 2304: TP4 shard = 576 (18 × 32 groups; 9 × 64; **not** a multiple of 128). TP8 = 288
  (9 × 32). V4-Flash's 2048 sharded to 512. Any 1Cat MXFP4/Marlin tile that assumes K or N multiples of 128
  per rank must be checked; EP (96 whole experts per rank at 4-way) avoids the split entirely.
- Hidden 5120 = 40 × 128 = 160 × 32 (V4: 4096). wq_b out 32768 (64 heads × 512); wo_a is block-diagonal over
  8 groups of [1024 × 4096]; TP4 → 16 heads, 2 o-groups, 8 indexer heads per rank.
- FP8 dense block is **32×32** (V4-Flash: 128×128) — 1Cat's FP8-block dequant/GEMM must accept 32-blocks;
  the activation-quant block in the reference also changed from 128 to 32 (`model.py:27,187,197`).
- Vocab 129,280 / 4 = 32,320 rows per rank (embed/head shard cleanly).
