# DeepSeek-V4.1-Flash — model archaeology (official reference implementation)

> **Origin.** PLI Labs research archive, file `v100-research/deepseek-v4.1-flash/MODEL_ARCHAEOLOGY.md`. Published 2026-11-18 in the Volta V100 research
> repository.
> **Edited for publication.** Machine names, private file paths, network addresses, session names and the
> operator's name were replaced with generic terms (`the V100 server`, `[private path]`, `[address]`,
> `the owner`). Links to files that are not published were turned into plain text. No figure and no finding
> was changed. Dates inside the note are its own; later results are in the build pages of this repository.

Lane L-MODEL, 2026-10-06. Everything here was read from the official repository
`deepseek-ai/DeepSeek-V4.1-Flash` @ **`2cba9e42aa026125f3ed06c6d98c1db82f7ca027`** (2026-10-01 "Adding chat template").
Local copies: `[private path]` (README, config.json, inference/*, encoding/*, chat_template.jinja,
tokenizer_config.json, model.safetensors.index.json, `DeepSeek_V41_Tech_Report.pdf` 51 pp; text extract at
`[...]/techreport.txt`). `file:line` references are to those copies.

| File | sha256 (prefix) | Lines |
|-|-|-|
| `inference/model.py` | 4e9ae236… | 1310 |
| `inference/engram.py` | 11f35ecb… | 184 |
| `inference/kernel.py` | 1236c350… | 591 |
| `inference/convert.py` | 03502834… | 205 |
| `inference/generate.py` | 8668d67f… | 218 |
| `inference/vision.py` / `image_processor.py` | 5d49edc1… / 482759e3… | 119 / — |
| `config.json` / `inference/config.json` | 8be45ce0… / 2e84f45c… | — |
| `DeepSeek_V41_Tech_Report.pdf` | ba68e2e4… | 51 pp |

**Weights have not changed since the first upload.** Shard LFS oids at `517ef625` (init, 2026-09-10), `fb2764a5`,
`dba1be0a` and `2cba9e42` are identical (48/48); every later commit touched docs/encoding/chat template only. So
quantized derivatives made from any of those revisions (e.g. NVIDIA from `dba1be0a`, s-zaizen from `fb2764a5`,
antirez from `df42c109`) share our source weights.

Inference deps (`inference/requirements.txt`): `torch>=2.10.0, transformers, tokenizers, safetensors>=0.7.0, numpy,
sympy, Pillow, tilelang==0.1.8, tqdm`. All GEMM/attention kernels are TileLang (`kernel.py`), with TMA and warp
specialization disabled (`kernel.py:9-12`) — but they use FP8 tensor-core GEMMs, so they need SM89+.

---

## 1. Shape of the model (config.json → inference/config.json)

| Key | Value | Notes |
|-|-|-|
| architecture | `DeepseekV41ForCausalLM`, `model_type deepseek_v41` (text: `deepseek_v41_text`) | `transformers_version "5.6.0"` is only a stamp — no released transformers has `deepseek_v41` (§12) |
| layers | 40 backbone (`n_layers`) + 3 DSpark (`n_mtp_layers`/`num_nextn_predict_layers`) | layer ids 40–42 = `mtp.0..2` |
| hidden `dim` | 5120 | V4-Flash: 4096 |
| vocab | 129,280 | bos 0, eos 1, pad 2; `image_token_id` 129264; DSpark noise token 128799 |
| attention | 64 heads, **1 KV head (MQA, shared K=V latent)**, head_dim 512 (last 64 = RoPE), q_lora 1280, o_lora 1024 × o_groups 8, `attn_sink`, sliding window 128 | |
| compress_ratios (43 entries) | `[0,0, 2×18, 1×20, 0,0,0]` | 0 = SWA only; 2 = 2:1 pooled latent; 1 = one latent per token |
| kv_source_layer_ids | [2, 8, 14, 20] | own a Compressor + compressed KV cache + index-K |
| index_source_layer_ids | [2, 8, 14, 20, 24, 28, 32, 36] | own an Indexer query |
| indexer | 32 heads × 128, top-512 | V4-Flash: 64 heads |
| candidate | source layer 20, block 8, top-2048 blocks | hierarchical indexer, decoder only |
| MoE | 384 routed (top-6) + 1 shared, inter 2304, `sqrtsoftplus`, `noaux_tc`, `norm_topk_prob`, routed_scaling 1.5, `swiglu_limit` 10 | V4-Flash: 256 routed, inter 2048, 3 hash-routed layers |
| mHC | `hc_mult` 4, Sinkhorn 20 iters, `hc_eps` 1e-6 | |
| norm eps | **1e-20** (`rms_norm_eps`) | V4-Flash 1e-6 |
| Engram | layers [1, 14], `num_embeddings` [384,006,168; 384,016,682], n-gram ≤ 4, `engram_vocab_size` 16,000,000, 8 heads × 256, pad id 2, compressed vocab 99,092 | ENGRAM.md |
| DSpark | 3 blocks, block 5, Markov rank 256, 128 experts top-3, target layers [37,38,39] | §8 |
| RoPE | θ 10000; compressed layers θ 160000; YaRN factor 16 over 65,536 → 1,048,576, β_fast 32, β_slow 1 | §9 |
| vision | 32-layer ViT, 1024 wide, 16 heads, patch 14, 3×3 unshuffle, ≤1024 image tokens | out of scope for text bring-up |
| quantization | `fp8`, `weight_block_size [32,32]`, `scale_fmt ue8m0`, `activation_scheme dynamic`, `expert_dtype fp4` | TENSOR_CENSUS.md |

Parameter counts (header-exact, TENSOR_CENSUS.md): 551.88 B "backbone" (incl. embed/head), 196.61 B Engram, 14.23 B
DSpark, 0.49 B vision; active ≈ 16.13 B/token decode, ≈ 7.89 B/token prefill (encoder only).

## 2. Top-level dataflow — `Transformer.forward` (model.py:1241-1272)

```
engram_hashes = NgramHashState(input_ids, start_pos, mask)            # model.py:1252, engram.py:160-184 (token ids only)
h = embed(input_ids)                                                    # 1253 (ParallelEmbedding, vocab-sharded + all_reduce, 168-178)
h = h.unsqueeze(2).repeat(1,1,4,1)                                      # 1258: 4 identical residual copies [b,s,4,5120]
pre_mix = one-hot [1,0,0,0]                                             # 1260, make_identity_pre_mix 1159-1163
for i, layer in enumerate(layers):                                      # 1261
    if layer.engram: h = layer.engram(h, engram_hashes[:,:,k,:], mask)  # 1262-1263: BEFORE the block (layers 1, 14)
    if i in (37,38,39): main_hiddens.append(h.mean(dim=2))              # 1265-1266: DSpark taps the block INPUT
    h, pre_mix = layer(h, start_pos, pre_mix, image_mask)               # 1267
h = layer.hc_pre(h, pre_mix)          # 1268: collapse with the LAST block's ffn_pre — there are no hc_head_* weights in V4.1
logits = head(norm(h))                # 1269: head weight held fp32 (ParallelHead 997-1017), last position only by default
```

`generate.py` (28-87) does plain autoregressive decoding: one prefill pass over `[0:min_prompt_len]`, then one token per
step; sampling is Gumbel-max (`model.py:1285-1292`); **it never calls `forward_spec`** (the DSpark path is implemented
but unused, `model.py:129-130`). Interactive mode sets `max_seq_len = 64*1024`, `max_batch_size = 1` (`generate.py:124-126`).

## 3. Causal Encoder–Decoder (CED)

Tech report §2.2 (txt 379-413), §3.2.2 (990-1026); README "Architecture".
- Layers 0–19 = causal **encoder**, 20–39 = **decoder**. Layers 0–1 are SWA only; 2–19 CSA2 ratio 2; 20–39 CSA2 ratio 1.
- Decoder **global** KV is projected only from the encoder's final hidden state `H_{L/2}`: in the code this is simply
  that layer 20 is the only decoder KV source (`kv_source_layers` contains 20) and its Compressor reads layer 20's own
  attention input (`attn_norm(hc_pre(h))`, `model.py:983-985,747`), i.e. the residual stream right after layer 19.
  Layers 21–39 never project their own global KV (Reuse/Reindex, §4.4).
- Decoder **SWA** KV is per-layer from each decoder layer's own hidden state (`_window_kv`, `model.py:700-720`).
- Consequence: for prompt tokens older than the window, only layers 0–19 + layer 20's `wkv`/compressor/indexer-K
  projections are needed → "8B active during prefill" (computed: 7.89 B). DeepSeek's deployment ("Decoder SWA
  Bounded Replay") runs the decoder only on the last `n_win = 128` prompt tokens with SWA truncated to the replay
  segment, and post-trains with the same replay (txt 1011-1026); it is **approximate by design**.
- **The reference `model.py` does NOT implement either optimization**: prefill runs all 40 layers on every token,
  which is the exact computation. JigSawPT measured that exact encoder-only prefill needs the decoder on the last
  19 × 128 = 2,432 positions (receptive field of the stacked windows) — report README line 340.
  For the V100 port: implement exact full prefill first (reference parity), then Decoder-SWA-Bounded-Replay as an
  opt-in prefill optimization (≈ halves prefill FLOPs and expert traffic at long context).

## 4. Attention — CSA2 (`Attention`, model.py:613-789)

### 4.1 Projections and per-layer state

| Tensor | Shape (out × in) | dtype | Code |
|-|-|-|-|
| `wq_a` | 1280 × 5120 | FP8 32×32 | 640, `qr = q_norm(wq_a(x))` 770 |
| `q_norm` | 1280 | bf16 RMSNorm eps 1e-20 | 641 |
| `wq_b` | 32768 × 1280 (64 heads × 512), ColumnParallel | FP8 | 642, 771 |
| `wkv` | 512 × 5120 (one shared K=V latent) | FP8 | 643, 705 |
| `kv_norm` | 512 | bf16 | 644 |
| `wo_a` | 8192 × 4096 (8 groups × 1024 out, each group sees 8 heads × 512 = 4096), ColumnParallel, **block-diagonal** | FP8 in ckpt → bf16 after `convert.py:157-173` | 645-649, einsum 785-787 |
| `wo_b` | 5120 × 8192, RowParallel (fp32 all-reduce) | FP8 | 650, 788 |
| `attn_sink` | 64 (per head) | fp32 | 639 |
| `softmax_scale` | 512^-0.5 (no YaRN mscale anywhere) | | 651 |

**No per-head q RMS-normalization in V4.1.** V4 did `q *= rsqrt(mean(q²))` after `wq_b` (V4 model.py:498); V4.1
goes straight from `wq_b` to RoPE (`model.py:771-772`). JigSawPT found exactly this as "the q-norm defect" when
porting from a V4 engine — a V4-derived path (1Cat) must gate the q-norm off for V4.1.

### 4.2 Forward (model.py:765-789)

1. `qr = q_norm(wq_a(x))`, `q = wq_b(qr)` → [b,s,h_local,512]; RoPE on the last 64 dims (interleaved pairs, complex view,
   `apply_rotary_emb` 392-406).
2. Window KV (`_window_kv` 700-720): `kv = kv_norm(wkv(x))`, RoPE on last 64, then **QAT fake-quant to FP8 E4M3 with
   UE8M0 scale per 32 channels over the whole vector, RoPE tail included** (`act_quant(..., inplace=True)` 707). Ring
   buffer of 128 slots per layer (`window_kv_cache`, 663-668). Window indices: `get_window_topk_idxs` 409-426
   (prefill: causal window per query; decode: whole ring, -1 for unfilled).
3. If `compress_ratio > 0` (layers 2–39): `_compress_kv` (739-763) — source layers run the Compressor, everyone gets
   top-k compressed indices (own indexer or the shared ones), the latent is RoPE'd and **QAT fake-quantized to FP4
   E2M1 with an E4M3 scale per 16 channels** (`fp4_act_quant(latent, 16, True, scale_dtype=float8_e4m3fn)` 760) and
   written to the source layer's `compress_kv_cache`. KV = concat(window KV, compressed KV); index lists concatenated
   with the compressed indices offset by the window length (`offset = kv.size(1)` 776).
4. `o = sparse_attn(q, kv, attn_sink, topk_idxs, scale)` (kernel.py:310-403): per query, gather the listed KV rows
   (-1 = skip), online-softmax over them, **the sink adds `exp(attn_sink[h] - max)` to the denominator only**
   (kernel.py:382-385) — a learned "attend to nothing" mass; K and V are the same 512-vector (MQA latent). Heads padded
   to 16 if fewer (TP16+).
5. Inverse RoPE on `o[..., -64:]` (781) so the output is position-free again.
6. `wo_a` block-diagonal grouped projection: `o.view(b,s,8_local,4096) · wo_a.view(8,1024,4096)` → [b,s,8,1024]
   (785-787, kept bf16 by convert.py; the comment says an FP8 grouped GEMM would halve its memory), then `wo_b`.

Per query the attention reads ≤ 128 window + ≤ 512 compressed entries = **≤ 640 KV rows of 512** — context-length
independent once the indexer has chosen.

### 4.3 Compressor (model.py:429-485)

- Only on KV-source layers (2, 8, 14, 20) (`model.py:658-659`).
- ratio 2 (layers 2, 8, 14): `wkv`, `wgate` both 512 × 5120 (bf16 in ckpt, promoted to fp32), every 2 consecutive
  tokens are pooled: `kv = Σ softmax(score over the 2 tokens, per channel) · kv` (473-475), carrying an incomplete
  group in `kv_state/score_state` across decode steps (476-482). **No overlap and no absolute position embedding**
  (V4's ratio-4 CSA used overlapping 2m windows + `ape`; V4.1 removes both, report txt 481-487).
- ratio 1 (layer 20): plain `norm(wkv(x))`, no gate, bf16 (461-462) — the decoder's uncompressed global KV.
- Output is the **pre-RoPE** latent (the indexer needs it unrotated, 432-434); `norm` = RMSNorm(512).
- A latent represents the first token of its group: RoPE position `j·ratio` (538-543, 752-757), with the
  compressed-layer frequencies (θ 160000 + YaRN).

### 4.4 CSA2 modes and the shared-state runtime

`SharedAttentionRuntime` (model.py:1166-1180) is one global slot each for `compress_kv`, `index_k`, `topk_idxs`,
`candidates`; sources write before consumers read because layers run in order.

| Mode (report Fig. 4) | Layers | Computes | Reuses |
|-|-|-|-|
| SWA only | 0, 1 (+ DSpark 40–42) | q, window KV | — |
| Full | 2, 8, 14 (enc, r=2); 20 (dec, r=1) | q, window KV, compressor latent → main KV, index-K, indexer q, top-512 | — |
| Reuse | 3–7, 9–13, 15–19, 21–23, 25–27, 29–31, 33–35, 37–39 | q, window KV | main KV of the latest Full layer + top-k of the latest Full/Reindex layer (`_compress_topk_idxs` 725-726) |
| Reindex | 24, 28, 32, 36 | q, window KV, own indexer q (`wq_b`, `weights_proj`) | main KV + index-K of layer 20, candidate pool of layer 20 |

Groups (encoder 3 × [Full + 5 Reuse], decoder [Full + 3 Reuse] + 4 × [Reindex + 3 Reuse]) match report §4.2.1
(txt 1097-1104).

**Pipeline-parallel consequence (inference):** a stage that holds a Reuse/Reindex layer whose source sits on an
earlier stage needs, per step, the source's new latent rows (to mirror the compressed KV + index-K cache:
≤ 512+128 values per new latent), the top-512 indices (`[b,s,512]` int32 = 2 KB/token) and, for Reindex layers,
layer 20's candidate pool (the reference materializes it as a `[b,s,T]` bool mask; ship the 2048 block ids instead,
8 KB/token). The hidden state crossing a PP boundary is already `[b,s,4,5120]` (+ `pre_mix [b,s,4]`) because of
mHC — 40 KB/token in 16-bit, 4× a normal model. The report's training stack calls this "pipeline payload
extensions" (txt 892-896). Aligning stage boundaries with group boundaries (14 and 20) avoids KV mirroring for the
encoder, but the decoder is one KV group (20–39) so a PP3 split always crosses it.

### 4.5 Indexer (model.py:488-580)

- Present on index sources (`model.py:660-661`). Query: `wq_b` 4096 × 1280 (32 heads × 128) applied to the attention
  `qr` (shared low-rank query), RoPE on the last 64 dims, **FP4 QAT (E2M1, E8M0 per 32)** (550-552).
- Keys: only KV-source layers own `wk` (128 × 512, bf16) + `k_norm`; `k = k_norm(wk(latent))` from the **pre-RoPE
  compressor latent**, RoPE (at the group's first position), FP4 QAT, stored in `k_cache` [b, max_seq/ratio, 128]
  (517-548). CSA2 "obtains indexer K by projecting main KV entries" (report txt 485-486) — V4 had a separate indexer
  compressor with Hadamard rotation; V4.1 has neither.
- Score: `index_score[b,s,t] = Σ_h relu(q_h · k_t) · w_h` with `w = weights_proj(x) · 128^-0.5 · 32^-0.5`
  (`weights_proj` 32 × 5120 bf16, applied to the attention input x) (555-557); TP partial sums all-reduced (558-559).
  Causal mask in prefill: position t visible once the query passed the group's last token (563-567).
- Top-k: `min(512, end_pos//ratio)` best, re-sorted by position, -1 for unreachable, +offset (577-580).
- **Cost note:** the reference materializes `index_score` densely, `[b, s, h_local, t]` before the head reduction
  (556) — at prefill this is O(s·t·h) memory, so long prompts must be chunked; engines replace this with fused
  "sparse MQA logits" kernels (DeepGEMM in vLLM/SGLang). 1Cat's V4 SM70 path has its own fused indexer (see
  `documents/plans/2026-07-07_fused-indexer-design.md`).

### 4.6 Candidate block mask — the hierarchical sparse indexer (model.py:569-575, 583-610)

- **Computed by layer 20 only** (`is_candidate_source`, 502): from its full-range index scores, each block of 8
  consecutive compressed positions (= 8 tokens at ratio 1) gets the max score of its positions; the newest
  (partially filled) block is pinned to +inf; the top **2048** blocks are kept; the boolean pool is expanded back to
  positions (`select_candidate_blocks`, 596-610). Pool = 2048 × 8 = **16,384 positions**.
- **Used by** every later index source (`uses_candidates = 0 <= candidate_source_layer < layer_id`, 503): layers
  24, 28, 32, 36 mask their own scores to the pool before their own top-512 (573-575). Encoder indexers (2, 8, 14)
  have id < 20 so they never use it. Reuse layers inherit whatever top-k was last published.
- **When it matters:** only when the reachable blocks exceed 2048, i.e. **context > 16,384 tokens** (decoder ratio
  1). At ≤ 16K every reachable block is selected and the mask is a no-op. Above that it bounds the Reindex layers'
  scoring to 16,384 positions regardless of context (report §2.3.2, txt 548-579; "training-aware, introduced in
  post-training", so it must be reproduced exactly, not treated as an optimization).
- Note: layer 20 itself still scores the entire causal context (O(T) per query).

### 4.7 Global-KV / indexer state per token (what must persist)

Model-native formats (report §2.4.4, txt 678-706; code 546, 552, 707, 760):

| State | Format | Bytes per latent | Latents per token | **Bytes / token** |
|-|-|-:|-:|-:|
| encoder main KV, layers 2/8/14 | E2M1 + E4M3/16 | 256 + 32 = 288 | 3 × ½ | 432 |
| encoder index-K, layers 2/8/14 | E2M1 + E8M0/32 | 64 + 4 = 68 | 3 × ½ | 102 |
| decoder main KV, layer 20 | E2M1 + E4M3/16 | 288 | 1 | 288 |
| decoder index-K, layer 20 | E2M1 + E8M0/32 | 68 | 1 | 68 |
| **global total** | | | | **890** (= report's figure) |
| SWA ring, every layer | E4M3 + UE8M0/32 | 512 + 16 = 528 | 128 slots × 43 layers (incl. DSpark) | 2.9 MB per sequence, constant |

At 256K tokens the global KV is 233 MB; at 1M, 933 MB. The **reference stores these QAT-rounded values in bf16**
(e.g. `compress_kv_cache` bf16 [b, T/ratio, 512], 669-679) = 1024 B per main latent and 256 B per index-K; all those
values are exactly representable in FP16 too (E2M1 × E4M3 ⊂ FP16 range; see §11), so a V100 port can choose FP16
storage (3,200 B/token: 0.84 GB at 256K, 3.4 GB at 1M) or packed FP4 with in-kernel decode.

## 5. Hyper-connections — Single-Pass mHC (Block, model.py:907-994; kernel.py:406-474)

- Residual stream = 4 copies `[b,s,4,5120]`. Each sublayer (attention, FFN) has `hc_*_fn` [24, 20480] fp32,
  `hc_*_base` [24], `hc_*_scale` [3] (938-946); `mix_hc = (2+4)·4 = 24` = 4 pre + 4 post + 16 comb.
- `hc_mixes` (948-955): flatten the 4 copies to 20,480, **one RMS statistic over all 20,480** (`rsqrt(mean(x²)+eps)`),
  `mixes = (x @ hc_fn^T) · rsqrt` in fp32, then `hc_split_sinkhorn`:
  `pre = sigmoid(m[0:4]·scale0 + base) + eps`, `post = 2·sigmoid(m[4:8]·scale1 + base)`,
  `comb = softmax_rows(m[8:24]·scale2 + base) + eps`, column-normalize, then 19 more alternating row/column
  normalizations (20 total) → doubly stochastic 4×4 (kernel.py:426-460).
- `hc_pre(x, pre)` = Σ_c pre_c · x_c (957-960); `hc_post` = `post_c · F(x) + Σ_j comb[j,c]·residual_j`
  (962-966; note the `comb.unsqueeze(-1) * residual.unsqueeze(-2)` sum over dim 2 contracts the *first* comb index).
- **Single-pass shift (V4.1 change):** the coefficients a sublayer computes are used by the *next* sublayer
  (`Block.forward` 968-994): attention input uses the `pre` computed by the previous block's FFN hc_mixes; the FFN
  input uses the `pre` computed by this block's attention hc_mixes (`x = self.hc_pre(x, attn_pre)` 990); the block
  returns `ffn_pre` for the next block. First block: one-hot pre_mix. Final collapse uses the last `ffn_pre`
  (1268). Report §2.4.1 (txt 584-636); training uses the same shift ("empirically negligible degradation").
- V4 by contrast applied each sublayer's own `pre` immediately and had a separate `hc_head_fn/base/scale` for the
  final collapse (V4 model.py:673-681, 728-735).
- HC precision: all hc math in fp32 in the reference; `hc_post` output cast back to the activation dtype (bf16) (966).

## 6. MoE (model.py:792-904)

- **Gate** (792-827): `scores = linear(x.float(), W.float()) / gate_temp` (fp32 GEMV, W bf16 [384,5120]; gate_temp 1.0),
  `scores = sqrt(softplus(scores))`, `idx = topk(scores + bias, 6)`; weights = unbiased `scores.gather(idx)`,
  normalized by `sum + 1e-20` (literal, "matches training", 825), × `route_scale` 1.5.
- **`gate.bias_vl`**: a second expert-selection bias used *instead of* `bias` for tokens inside an image span
  (`torch.where(image_mask, bias_vl, bias)` 818-820). Training "maintains separate expert-wise correction biases for
  text and image tokens" (report txt 367-376, "Multimodal Auxiliary-loss-free Load Balancing"). For text-only
  serving `image_mask` is None and `bias_vl` is never read (it exists only if `vision_n_layers > 0`, 807).
  LibertAIDAI measured `bias` vs `bias_vl` as essentially uncorrelated (mean corr −0.05) and the text `bias` nearly
  uniform (std 0.063) — no cold experts to prune or to statically spill.
- **No hash-routed layers** (V4-Flash routed its first 3 layers by `tid2eid[token]`).
- **Expert** (830-851): SwiGLU with clamps from training: `up = clamp(w3x, -10, 10)`, `gate = clamp(w1x, max=10)`,
  `h = silu(gate)·up`; **the routing weight multiplies `h` before `w2`** (849-850), then `w2`. Routed experts are
  FP4 (`float4_e2m1fn_x2` [out, in/2] + E8M0 [out, in/32], 219-224); the shared expert is the same class with FP8
  weights (887).
- **MoE.forward** (889-904): fp32 accumulation `y[idx] += expert(x[idx], w)`, all-reduce over TP/EP ranks, add the
  shared expert, cast back. Experts are split across ranks by index range (EP-style, 863-885); the shared expert is
  replicated per rank in this reference (it is not TP-sharded).

## 7. Engram (model.py:296-365, 1197-1200, 1252, 1262-1263; engram.py)

Summary — details, byte math and the placement analysis are in **ENGRAM.md**.
- Hash ids are a pure function of token ids (via a tokenizer-derived compressed vocab of 99,092 ids), computed once per
  forward for all positions, before the embedding (`model.py:1252`).
- Per position and Engram layer: (4−1) n-gram orders × 8 heads = **24 rows** of 256 FP8 values + 8 UE8M0 scales
  (264 B) → 6,336 B/token/layer, **12,672 B/token** for both layers.
- `wkv` (FP8, 25,600 × 6,144) turns the concatenated 6,144-vector into 4 keys (one per hc copy) + 1 value; a per-copy
  gate `sigmoid(signed_sqrt(⟨h_c · q_w·k_w, key_c⟩ · rstd_h · rstd_k / √5120))` adds `gate_c · value` to each copy
  (350-365). Applied to the residual stream *before* blocks 1 and 14.
- Reference placement: tables row-sharded across TP ranks **on GPU**, gathered with `F.embedding` + all-reduce
  (296-325). DeepSeek's production stack prefetches rows from host memory over RDMA (report txt 652-654); vLLM keeps
  them in pinned host memory read via UVA by default.

## 8. DSpark / MTP (model.py:1020-1157, 1207-1213, 1274-1282)

- Three `DSparkBlock`s stored under `mtp.*`; each is a full block with SWA-only attention (`compress_ratio 0`,
  `DSparkAttention` 1032-1074), MoE with **128 experts, top-3** (`get_moe_config` 142-149), its own mHC params; no
  Engram, no compressor/indexer; embed and head **tied to the backbone** (1212-1213; `convert.py:108-110` drops any
  `mtp.*.embed/head`).
- Input: `main_hidden` = concat over target layers 37, 38, 39 of the **block input averaged over the 4 hc copies**
  (1265-1266) → `main_proj` (FP8 5120 × 15360) → `main_norm` → `main_x`, used as **extra KV** in the draft's
  attention: the window cache of each DSpark layer is filled with `kv_norm(wkv(main_x))` of the committed tokens
  (1039-1052, 1065).
- Draft tokens: block of 5 positions `[last_token, noise×4]` (noise id 128799) embedded with the backbone embed and
  expanded to 4 hc copies (`forward_embed` 1128-1135). The 5 queries attend to the 128-slot window of main KV plus
  the 5 draft KVs (`get_dspark_topk_idxs` 1020-1029; no causal mask among the 5 — semi-autoregressive).
- Head (`forward_head` 1137-1156): base logits from the backbone head for all 5 positions; a **Markov head**
  (`markov_head.embed/head`, 129,280 × 256 bf16 each, rank 256) adds a bigram bias from the previously sampled draft
  token, sequentially over the 5 positions; a **confidence head** (`proj` 1 × 5376 over [hidden, markov_embed])
  predicts per-position acceptance (used by the scheduler to choose verification length, report txt 664-670).
- Prefill only seeds the DSpark window caches (`DSparkBlock.forward` 1122-1126).
- Reference status: "Only the forward pass is implemented here — nothing calls forward_spec" (129-130); the self-test
  calls it (1303-1309). Acceptance/verification logic lives in the engines (vLLM `vllm/models/deepseek_v41/*/dspark.py`,
  SGLang `kernels/ops/speculative/dspark/`). Community measurements (JigSawPT, RTX 5090 offload-bound): acceptance
  51 % prose / 79 % code / 97 % verbatim; best draft length 2 there because verification is I/O-bound.

## 9. RoPE and YaRN (model.py:368-406, 680-698)

- Interleaved pairs (`view_as_complex(x.unflatten(-1,(-1,2)))`), applied to the last 64 of 512 dims of q, window KV,
  compressed latent, indexer q/k; inverse rotation on the attention output.
- Layers with `compress_ratio > 0` (2–39): **θ = 160,000 with YaRN** (factor 16, original 65,536, β_fast 32,
  β_slow 1) for *all* their RoPE, window KV included. Layers with ratio 0 (0, 1, DSpark 40–42): θ = 10,000,
  **no YaRN** (686-687).
- YaRN as implemented (376-386): frequency pairs 0–15 unchanged, pairs ≥ 25 divided by 16, linear ramp between
  (for θ 160000, d 64). **No attention-temperature (mscale) factor** is applied anywhere (softmax scale is plain
  512^-0.5) — a porting trap for vLLM-style `YaRNScalingRotaryEmbedding`/DeepSeek-YaRN classes that fold mscale into
  cos/sin.
- Trained at 64K with sparse attention from scratch; extended to 1M at 34T tokens (report txt 1137-1138).

## 10. Norms, embedding, head

- RMSNorm (281-293): fp32 compute, weight in the default dtype; eps **1e-20** for every norm (attn/ffn/q/kv/
  compressor/k_norm/final). Engram and mHC use `norm_eps` 1e-20 too.
- `ParallelEmbedding` (152-178) vocab-sharded; `ParallelHead` (997-1017) weight held **fp32** (bf16 in ckpt), logits
  fp32, all-gather over vocab shards.
- `tie_word_embeddings: false` for the backbone; DSpark ties to backbone embed/head.

## 11. Kernels and numerics (kernel.py) — and what changes on V100

| Function | What it does | Reference numerics | V100 path |
|-|-|-|-|
| `linear()` (model.py:181-207) | weight dtype dispatch | FP4/FP8 weights ⇒ **activation quantized to FP8 E4M3, block 32, UE8M0 scale** (`act_quant`), then FP8×FP8 GEMM | FP16 activations, weights dequantized to FP16 in-kernel (no FP8/FP4 MMA on SM70) |
| `act_quant` (kernel.py:40-124) | per-32 block amax → scale = 2^ceil(log2(amax/448)) (`fast_round_scale`), clamp ±448, cast E4M3 | also used *in place* (quant→dequant) as QAT on window KV | software E4M3 round-trip in FP16/FP32 where it is model semantics (window KV) |
| `fp4_act_quant` (128-204) | per-32 E8M0 (indexer q/k) or per-16 E4M3 (compressed KV; amax floor 6·2^-9) FP4 round-trip | QAT, output-visible (changes top-k) | must be reproduced bit-exactly (LUT round to E2M1 grid) |
| `fp8_gemm` (207-307) | A_fp8 · B_fp8ᵀ, per-K-block (32) partial sums in fp32 rescaled by `s_a[i]·s_b[j]` and accumulated in a second fp32 accumulator | out bf16 | FP16 HMMA, fp32 accumulate; per-32 weight scales are per (32×32 block) → can be folded into dequant (scale exponents measured −13…−6 ⇒ exact in FP16) |
| `fp4_gemm` (477-591) | FP4 → (via fp32) FP8 cast (lossless), then as `fp8_gemm` with weight scale per (row, 32-K) | out bf16 | dequant E2M1 × 2^e directly to FP16 (measured expert exponents −12…−2 ⇒ every value exact in FP16) |
| `sparse_attn` (310-403) | gather-by-index flash attention, bf16 inputs, fp32 softmax, sink in denominator | bf16 | FP16 MMA (V100 has no bf16) — overflow check needed on q·k (scores are fp32-accumulated) |
| `hc_split_sinkhorn` (406-474) | per-token 4×4 Sinkhorn | fp32 | fp32, trivially portable |

Measured scale exponents (E8M0 = 2^(byte−127)), from HTTP range reads of real scale tensors
(`[...]/scale_sample.json`; 30.97 M expert scales across layers 0/1/10/20/30/39/mtp.0,
643 K dense scales of layers 0/2/20/39, both `engram.wkv`, 8.4 M Engram row scales):

| Weights | exponent range | FP16-exact dequant? |
|-|-|-|
| routed experts (E2M1 × 2^e) | −12 … −2 | **yes** (min nonzero 2^-13, max 1.5) |
| dense FP8 (attn, shared expert, indexer) | −13 … −6 | **yes** (E4M3 × 2^e ≥ 2^-22, ≤ 7; FP16 subnormals cover it) |
| `engram.wkv` FP8 | −18 … −8; **22.9 % of blocks below −14** | **no** — small elements fall below FP16 subnormal resolution; dequant with a 2^k pre-bias (e.g. scales × 2^10, divide the fp32 output) or keep FP32 dequant |
| Engram table rows | −10 … −6 | **yes** (row values ≤ 7, ≥ 2^-19) |

Numerics implications for the FP16-activation V100 path (inference, not measured yet):
1. Activations are *more* precise than the reference (FP16 vs per-32 FP8). Expect small, benign logit deltas; for
   bit-level validation add an optional "emulate act_quant" debug mode (FP8 round-trip of GEMM inputs).
2. The QAT fake-quants (window KV FP8, compressed KV FP4/E4M3-16, indexer q/k FP4/E8M0-32) are **model semantics**
   and must be applied (transformers PR #48721 makes the same point; vLLM/SGLang implement them).
3. Range: the reference keeps activations bf16 (8-bit exponent). FP16 tops out at 65,504. The 4-copy residual stream,
   MoE fp32 sums and attention logits need an overflow audit; a safe default is an fp32 residual stream (4 × 5120 ×
   4 B = 80 KB/token) with FP16 only at GEMM inputs. The training clamps (`swiglu_limit` 10) bound expert
   intermediates.

## 12. convert.py — HF layout vs inference layout

- HF names are **already** inference names (`layers.N.attn.*`, `ffn.experts.E.w{1,2,3}.{weight,scale}`, `.scale`
  not `weight_scale_inv`, no `model.` prefix) — the rename rules (`convert.py:105-126`) are no-ops for this
  checkpoint (verified against the index: no `self_attn`, `mlp`, `weight_scale_inv`, `e_score_correction_bias`,
  `tid2eid`).
- What it actually does: per-rank sharding into `model{i}-mp{MP}.safetensors` — routed experts by index range
  (128-134; backbone 384 and DSpark 128 must both divide MP), Engram tables row-sharded with ceil(rows/MP) padding
  (pad 0 for weight, value **1.0** for the E8M0 scale) (135-141), `embed/wq_b/wo_a/head/attn_sink/weights_proj`
  dim 0, `wo_b` dim 1 (56-64); everything else replicated (incl. shared experts, `wq_a`, `wkv`, compressor, Engram
  `wkv`).
- `wo_a` is dequantized FP8→bf16 (157-173). Experts: `int8 → float4_e2m1fn_x2` view (181), or `--expert-dtype fp8`
  casts E2M1 → E4M3 losslessly by absorbing up to 2^6 of the per-32 E8M0 scales into the values and keeping one E8M0
  per 32×32 block (`cast_e2m1fn_to_e4m3fn` 18-53).
- DSpark `embed/head` skipped (tied) (108-110). Index completeness asserted (148-152).

## 13. Prompt format, tokenizer, chat template

- `tokenizer.json` (6.37 MB) + `tokenizer_config.json` (`PreTrainedTokenizerFast`, no BOS/EOS auto-add,
  `model_max_length` 1,048,576). The **Engram compressed-vocab map is derived from the tokenizer** at load time
  (`build_compressed_token_map`, engram.py:17-61) and must equal 99,092 — tokenizer revision matters (vLLM PR #57897
  "Honor tokenizer revision in DeepSeek V4.1 Engram").
- Official prompt encoder: `encoding/encoding.py` (V4.1 changes vs V4: DSML tags with a leading space, numeric
  reasoning effort 1–100 rendered only in thinking mode at index 0 — `low` 50 / `high` 75 (default) / `max` 100
  (+`medium` 62 in the jinja), mid-conversation `<｜System｜>` messages; `encoding/README.md`). A
  `chat_template.jinja` ported from `encoding.py` was added 2026-10-01 (#68).
- Recommended sampling: temperature 1.0, top_p 0.95–1.0 (README).

## 14. Engine support (porting sources) — status 2026-10-06

| Engine | Status | Where | Notes |
|-|-|-|-|
| **HF transformers** | **No `deepseek_v41` in any release** (latest v5.19.0, 2026-10-06; `src/transformers/models/` has `deepseek_v4`, not v41). Tracking issue [#48755](https://github.com/huggingface/transformers/issues/48755) (open; maintainer remi-or "working on it"). | Community PRs (closed, unmerged): [#48721](https://github.com/huggingface/transformers/pull/48721) text model, branch `malaiwah/transformers:add-deepseek-v41` @ `62d7ebd7de4938e072b7aaeb881593b79dc56835` → `src/transformers/models/deepseek_v41/{configuration,modeling,modular}_deepseek_v41.py` (108 KB modeling) + `tests/models/deepseek_v41/test_modeling_deepseek_v41.py`; [#48768](https://github.com/huggingface/transformers/pull/48768) vision (`@a290b39`) | #48721 claims bit-exact QAT fake-quant ports and CPU parity with the reference; the cleanest pure-PyTorch porting reference after `inference/model.py` |
| **vLLM** | Supported since **v0.30.0** (2026-09-22); main @ `b6d8e8afd985f5711eee68e343d2ce908d166488` (2026-10-06) | `vllm/models/deepseek_v41/` — `attention.py`, `compressor.py`, `sparse_mla.py`, `quant_config.py`, `decoder_replay_layers.py`, `common/{engram.py,ops/*,rope.py,mm_preprocess.py}`, `nvidia/{model.py,model_state.py,dspark.py,flashmla.py,flashinfer_sparse.py,ops/*}`, `amd/*`; `vllm/config/engram.py`; `docs/features/engram.md`; tests `tests/kernels/test_engram.py`, `tests/distributed/test_engram_dp_shard.py` | Key PRs: [#56228](https://github.com/vllm-project/vllm/pull/56228) model definitions (merged 09-10), [#56214](https://github.com/vllm-project/vllm/pull/56214) support (09-11), [#56512](https://github.com/vllm-project/vllm/pull/56512) Engram UVA prefetch + DP sharding (09-13), [#56227](https://github.com/vllm-project/vllm/pull/56227) encoder SWA bounded replay (09-18); open: [#56757](https://github.com/vllm-project/vllm/pull/56757) disk-backed (mmap) Engram (closed unmerged — throughput parity with pinned UVA on B200), [#58856](https://github.com/vllm-project/vllm/pull/58856) FP8 Engram rows as int8 for SM86 lookup (pre-SM89 Triton FP8 issue — relevant to SM70), [#59655](https://github.com/vllm-project/vllm/pull/59655) keep DSpark experts MXFP4 in NVFP4 ckpts |
| **SGLang** | Supported since **v0.5.21** (2026-10-02; absent in v0.5.20); main @ `90aa1cb0fbc0b99ba74a18a0df44151da40a59d5` | V4.1 lives inside `python/sglang/srt/models/deepseek_v4.py` (225 KB) + `models/deepseek_v4_dspark.py`, `models/deepseek_v41_vit.py`, `configs/deepseek_v41.py`, `layers/engram.py` (host table: memfd/anon mmap + `cudaHostRegister`, env `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE`), `layers/attention/dsv4/{dsv41_sparse.py,v41_indexer/}`, `kernels/ops/embeddings/engram_{hash,gather,gate}.py`, `kernels/ops/speculative/dspark/*` | Main integration PR [#38798](https://github.com/sgl-project/sglang/pull/38798) (merged 09-18) |
| llama.cpp | Upstream: conversion/runtime PR [ggml-org/llama.cpp#28696](https://github.com/ggml-org/llama.cpp/pull/28696) (open). Forks: `JigSawPT/llama.cpp` branch `dsv41-porte`; `smalinin/llama.cpp` branch `my_build_deepseek41` (CUDA); antirez DwarfStar (Metal, `ds4.1flash`) | | Useful as independent numerics cross-checks (JigSawPT: logit corr 0.9967 vs reference at 1,401 tokens) |
| vLLM SM80 backport (prior art for no-FP8 GPUs) | Tokha233 `deepseek-v4.1-flash-a100-turbo` (HF mirror `StellarVoyager/DeepSeek-V4.1-Flash-A100-A800-SM80` @ `f4dfddab1d7d`) on `wtdcode/vllm-backport` `master-v013` | `patches/files/vllm/v1/attention/ops/candidate_mqa_sm80.py`, `patches/files/vllm/model_executor/kernels/linear/mxfp8/marlin.py`, `patches/files/vllm/models/deepseek_v4_1/nvidia/model.py`, `patches/files/vllm/model_executor/layers/sparse_attn_indexer.py`, `config/serve.json` | 8× A800: FP4 Marlin experts, BF16 dense, FP8 MLA KV, Engram `cpu_offload`, DSpark 5 — the same strategy class as the V100 plan (FP16 instead of BF16) |
| Reference | `inference/` (TileLang, FP8 tensor cores ⇒ SM89+) | | Ground truth for unit tests; runs nowhere on V100 as-is |

## 15. Things the V100 port must get right (checklist distilled from the code)

1. No per-head q RMS-norm (V4 had one). 2. Single-pass mHC coefficient shift + final collapse with last `ffn_pre`
(no `hc_head_*`). 3. Engram applied to the residual *before* blocks 1 and 14, gated per hc copy. 4. CSA2 shared state
(KV sources 2/8/14/20, index sources +24/28/32/36, Reuse layers reuse top-k), across PP stages. 5. Candidate pool
(layer 20 → 24/28/32/36), active > 16K tokens. 6. QAT fake-quants: window KV FP8 (whole vector, block 32, UE8M0),
compressed KV FP4 + E4M3/16, indexer q/k FP4 + E8M0/32. 7. Compressor ratio 2 (fp32 softmax pooling, no overlap/ape),
ratio 1 plain. 8. RoPE interleaved on last 64 dims, θ 160000 + YaRN (no mscale) on layers 2–39, θ 10000 no YaRN on
0/1/DSpark; inverse RoPE on attention output. 9. Attention sink in the softmax denominator. 10. Gate: sqrtsoftplus,
bias for selection only, normalize with +1e-20, ×1.5; routing weight applied before `w2`; SwiGLU clamps.
11. FP8 32×32 blocks (not 128×128 as V4-Flash). 12. eps 1e-20 everywhere. 13. `engram.wkv` FP16 dequant needs a
scale bias. 14. Tokenizer-derived Engram hash constants (compressed vocab 99,092; per-layer RNG seeds 10007·layer).
