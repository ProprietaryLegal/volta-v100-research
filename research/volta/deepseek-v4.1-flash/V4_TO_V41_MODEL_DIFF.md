# DeepSeek-V4-Flash → DeepSeek-V4.1-Flash: model-side diff

> **Origin.** PLI Labs research archive, file `v100-research/deepseek-v4.1-flash/V4_TO_V41_MODEL_DIFF.md`. Published 2026-11-18 in the Volta V100 research
> repository.
> **Edited for publication.** Machine names, private file paths, network addresses, session names and the
> operator's name were replaced with generic terms (`the V100 server`, `[private path]`, `[address]`,
> `the owner`). Links to files that are not published were turned into plain text. No figure and no finding
> was changed. Dates inside the note are its own; later results are in the build pages of this repository.

Lane L-MODEL, 2026-10-06. Compared trees (both official, read in full):
- **V4**: `deepseek-ai/DeepSeek-V4-Flash` @ `60d8d70770c6776ff598c94bb586a859a38244f1` (2026-06-22) — `config.json`,
  `inference/{config.json, model.py (827 lines), kernel.py (536), convert.py, generate.py, README.md}`; header census
  of all 46 shards (`[...]/census_v4flash{.json,_full.tsv}`; 159,609,485,896 B; sizes match HF 46/46).
- **V4.1**: `deepseek-ai/DeepSeek-V4.1-Flash` @ `2cba9e42aa026125f3ed06c6d98c1db82f7ca027` (MODEL_ARCHAEOLOGY.md).
Refs: `V4 m.py:N` = V4 `inference/model.py`, `m.py:N` = V4.1 `inference/model.py`.
(Which V4 checkpoint 1Cat validated is L-1CAT's finding; this table is checkpoint-vs-checkpoint.)

## Totals

| | V4-Flash | V4.1-Flash |
|-|-|-|
| Checkpoint bytes | 159,609,485,896 (148.65 GiB), 46 shards | 510,286,023,000 (475.24 GiB), 48 shards |
| Total params (all tensors, FP4 = 2/byte) | 290.94 B | 763.21 B (incl. 196.61 B Engram, 0.49 B vision) |
| "Backbone" params (README definition) | 284.33 B (README: 284B) | 551.88 B (README: 552B) + 196.61 B Engram |
| Active / token | ≈ 13.27 B decode (README: 13B) | ≈ 16.13 B decode, ≈ 7.89 B prefill (README: 16B / 8B) |
| Routed experts (weight+scale) | 137.06 GiB (43 × 256 experts) | 268.95 GiB (40 × 384) |
| Engram tables | — | 188.83 GiB |
| Everything else (backbone, excl. experts) + embed/head | 6.27 + 1.97 GiB | 9.17 GiB |
| MTP | 1 layer, 3.35 GiB | DSpark 3 blocks, 7.39 GiB |
| Global KV per token (model-native formats) | ≈ 3.5 KB (computed from V4 code: 21 CSA r4 + 20 HCA r128 private caches, FP8 non-RoPE + bf16 RoPE, FP4 indexer K; inference) | 890 B (report + computed, MODEL_ARCHAEOLOGY §4.7) |

## Component table

| Component | V4-Flash behavior | V4.1-Flash behavior | Notes for the port |
|-|-|-|-|
| Layers / width | 43 layers, dim 4096, 64 heads, vocab 129,280 (V4 config) | 40 layers (20 enc + 20 dec), dim 5120, 64 heads, vocab 129,280 | shape change; CED split is new semantics (row "CED") |
| Overall layout | uniform decoder stack | **Causal Encoder–Decoder**: decoder global KV only from the encoder output (layer 20 is the only decoder KV source) (m.py:653-661, report §2.2) | exact compute unchanged; enables encoder-only prefill + decoder bounded replay (approximate, opt-in) |
| Attention core | MQA latent 512 (rope 64), q_lora 1024, o_lora 1024 × 8 groups, sink, SWA 128 (V4 m.py:436-543) | same structure, q_lora **1280** (m.py:613-789) | reusable kernels; shapes change |
| `sparse_attn` kernel | running max initialized to −inf | initialized to **−1e30** so a query with no valid index yields an all-zero output instead of NaN (V4.1 kernel.py:352-355); otherwise identical | tiny kernel patch |
| q normalization | **per-head RMS norm of q** after `wq_b` (`q *= rsqrt(mean(q²)+eps)`, V4 m.py:498) | **none** (m.py:771-772) | must be gated off for V4.1 (JigSawPT's "q-norm defect") |
| Window-KV QAT | FP8 on the 448 non-RoPE dims, block **64**, RoPE dims bf16 (V4 m.py:505-506) | FP8 E4M3 on the **whole** 512 incl. RoPE, block **32**, UE8M0 (m.py:707) | numerics change |
| Compressed-attention types | hybrid **CSA (ratio 4, with indexer) + HCA (ratio 128, no indexer — attends to all compressed positions, `get_compress_topk_idxs` V4 m.py:268-276)**, alternating layers 2…42 | pure **CSA2**: ratio 2 (layers 2–19) and ratio 1 (20–39), all indexed | HCA path not needed; ratio-1 path new |
| Compressor | per compressed layer, private; ratio 4 uses **overlapping 2m windows** (`overlap_transform`, coff=2) and **absolute position bias `ape`** (V4 m.py:279-377); compresses + RoPEs + quantizes inside | only on KV sources (2, 8, 14, 20); **no overlap, no `ape`**; ratio 2 = softmax-gated pair pooling in fp32, ratio 1 = `norm(wkv(x))`; returns pre-RoPE latent, RoPE/quant done by Attention (m.py:429-485, 739-763) | simpler; different state machine for decode |
| Compressed-KV storage | per layer; FP8 non-RoPE (block 64) + bf16 RoPE (V4 m.py:372) | shared per KV group; **FP4 E2M1 + E4M3 scale per 16**, RoPE included (m.py:760) | new QAT format; KV group sharing |
| Cross-layer sharing | none (every layer owns its KV and indexer) | **Full / Reindex / Reuse** modes: KV sources [2,8,14,20], index sources [+24,28,32,36], Reuse layers reuse top-k (`SharedAttentionRuntime` m.py:1166-1180) | new; PP stages must forward shared state |
| Indexer K | separate indexer **compressor from hidden states** (head 128) with **Hadamard rotation** + FP4 (V4 m.py:380-433, `rotate_activation` 247-251) | **projected from the main-KV latent**: `k_norm(wk(latent))`, RoPE, FP4 E8M0/32, **no Hadamard** (m.py:517-548) | drop `fast_hadamard_transform`; new `wk` [128×512] |
| Indexer Q | 64 heads × 128 from `qr`, RoPE, Hadamard, FP4 (V4 m.py:411-416) | **32** heads × 128 from `qr`, RoPE, FP4, no Hadamard (m.py:550-552) | |
| Index scoring | relu(q·k)·weights_proj, top-512 over all compressed positions (V4 m.py:418-433) | same scoring; plus **hierarchical candidate pool**: layer 20 picks top-2048 blocks of 8, layers 24/28/32/36 score only inside the pool (m.py:569-610); effective only > 16K tokens | new mask logic, post-training-aware (must match) |
| mHC | multi-pass: each sublayer computes and applies its own pre/post/comb (V4 m.py:673-700); final collapse with separate `hc_head_fn/base/scale` (V4 m.py:715-735) | **Single-Pass**: pre-mix shifted one sublayer later (m.py:968-994); final collapse with last `ffn_pre`, **no `hc_head_*` tensors** | `hc_split_sinkhorn` kernel body identical (diffed); dataflow change |
| Routing | `sqrtsoftplus`, noaux bias, top-6/256; **first 3 layers hash-routed** by `tid2eid[token]` (I64 [129280,6]) (V4 m.py:546-584); weights `/= sum` | top-6/**384**, no hash layers, **`gate_temp`** (1.0), normalize with `+1e-20`, **`bias_vl`** for image tokens (m.py:792-827) | drop hash-route path; route_scale 1.5 unchanged |
| Experts | MXFP4 (E2M1 + E8M0/32), inter **2048**, SwiGLU clamp 10, routing weight applied before `w2` (V4 m.py:587-606) | identical format and math, inter **2304** (m.py:830-851) | MXFP4 expert kernels reusable; 2304/TP4 = 576 is not a multiple of 128 (TENSOR_CENSUS) |
| Shared expert | 1, FP8 | 1, FP8 | same |
| Dense FP8 blocks | **128×128**, act-quant block 128 (V4 config `weight_block_size [128,128]`, V4 m.py:17) | **32×32**, act-quant block 32 (m.py:27-28) | 1Cat FP8 dequant must accept 32-blocks (16× more scales) |
| Engram | — | n-gram hash memory at layers 1 and 14, 2 × ~384 M rows FP8, `wkv` 6144→25600, per-hc gate (m.py:296-365; engram.py) | **new**: host-resident table service + GPU gate (ENGRAM.md) |
| MTP / spec | 1 MTP layer: `e_proj/h_proj/enorm/hnorm`, own `hc_head`, standard next-token MTP (V4 m.py:738-766) | **DSpark**: 3 blocks, SWA-only, 128 experts top-3, `main_proj` from layers 37–39 block inputs, block of 5 with noise token, Markov head (rank 256), confidence head (m.py:1020-1157, 1274-1282) | new draft model + verification policy |
| Vision | — | DeepSeek-ViT 32 layers + aligner, image delimiters, `bias_vl` routing (vision.py, m.py:1215-1239) | out of scope (text only) |
| RoPE / YaRN | θ 10000 (SWA-only layers), compressed layers θ 160000 + YaRN ×16 over 65,536 (V4 m.py:466-482) | **same rule and same numbers** (m.py:680-698); YaRN implemented identically (no mscale) | reusable as is |
| Norm eps | 1e-6 | **1e-20** | config change |
| Activation quant in `linear` | FP8 per 128 (`scale_fmt` ue8m0 when `scale_dtype=fp8`) | FP8 per 32, UE8M0 | V100 uses FP16 activations either way |
| FP4 QAT kernel | E8M0 scale only | adds E4M3-scale mode (amax floor 6·2⁻⁹) for compressed KV (kernel.py:160-166) | |
| HF tensor names | `layers.N.attn.compressor.ape`, `attn.indexer.compressor.*`, `ffn.gate.tid2eid`, `hc_head_*`, `mtp.0.{e_proj,h_proj,enorm,hnorm,hc_head_*}` | `attn.indexer.{wk,k_norm}`, `ffn.gate.bias_vl`, `layers.{1,14}.engram.*`, `mtp.{0,1,2}.*` + `main_proj/main_norm/markov_head/confidence_head`, `vision.*`, `aligner.*`, `image_*` | loader mapping changes |
| Reference runtime deps | TileLang + `fast_hadamard_transform` | TileLang (no Hadamard), sympy (prime search), tokenizers (Engram vocab map) | |
| Prompt format | V4 encoding (`<｜DSML｜tool_calls>`, verbal effort) | DSML with leading space, numeric effort 1–100, `<｜System｜>` mid-conversation (encoding/README.md) | server-side change |

## What carries over unchanged (reuse candidates for 1Cat's V4 SM70 path)

MQA latent attention with sink and SWA ring buffer; sparse gather attention (`sparse_attn` identical except the −1e30 init); grouped
low-rank output projection; MXFP4 expert format (byte-identical encoding: I8 packed E2M1, E8M0 per 32 along K);
shared expert FP8; `sqrtsoftplus`/noaux routing core; SwiGLU clamp; Sinkhorn mHC kernel; YaRN; compressed-latent RoPE
convention (group's first token); indexer scoring formula and top-k; vocab/tokenizer family.

## What is new or semantically different (must be written or modified)

1. Remove per-head q-norm. 2. Single-pass mHC dataflow + head collapse without `hc_head`. 3. CSA2 (ratio 2/1
compressor without overlap/ape; KV-group sharing; Full/Reindex/Reuse; indexer K from latent; no Hadamard).
4. Candidate-block pool (layer 20 → 24/28/32/36). 5. New QAT formats (window KV FP8 incl. RoPE @32; compressed KV
FP4 + E4M3/16). 6. FP8 32×32 dense blocks. 7. No hash routing; `bias_vl` (vision only). 8. Engram (host table +
GPU gate). 9. DSpark draft model (optional for first light). 10. eps 1e-20. 11. CED encoder-only prefill / decoder
bounded replay (optional optimization; the exact path needs nothing new).
