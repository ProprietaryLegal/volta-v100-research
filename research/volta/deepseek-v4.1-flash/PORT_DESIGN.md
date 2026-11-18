# DeepSeek-V4.1-Flash on 1Cat-vLLM SM70 — port design (lane L-ARCH)

> **Origin.** PLI Labs research archive, file `v100-research/deepseek-v4.1-flash/PORT_DESIGN.md`. Published 2026-11-18 in the Volta V100 research
> repository.
> **Edited for publication.** Machine names, private file paths, network addresses, session names and the
> operator's name were replaced with generic terms (`the V100 server`, `[private path]`, `[address]`,
> `the owner`). Links to files that are not published were turned into plain text. No figure and no finding
> was changed. Dates inside the note are its own; later results are in the build pages of this repository.

Lane L-ARCH, 2026-10-07 (UTC). Status: **COMPLETE** (all 8 sections); progress log and scratch (calculator,
upstream fetches) not published (`memplan.py` reproduces every byte figure in §6).

Pinned sources (all `file:line` citations are at these revisions):

| Source | Revision | Local path |
|-|-|-|
| 1Cat-vLLM | `origin/main` **`2d5078372d7f`** (branch `ds41/sm70-v41` = that + L-BUILD build commits) | `[private path]` (read-only for design lanes) |
| upstream vLLM | main **`b6d8e8afd985`** (2026-10-06) | `[private path]`, `[private path]` |
| official reference | `deepseek-ai/DeepSeek-V4.1-Flash` **`2cba9e42`** `inference/{model.py,kernel.py,engram.py,convert.py}` | `[private path]` |

Citation prefixes: `1C:` = 1Cat path at 2d507837; `up:` = upstream vLLM at b6d8e8af; `ref:` = official
`inference/` at 2cba9e42 (`m.py` = model.py, `k.py` = kernel.py, `e.py` = engram.py). Inputs used: the plan,
DECISIONS.md D1–D3, HARDWARE.md, 1CAT_ARCHAEOLOGY.md, MODEL_ARCHAEOLOGY.md, TENSOR_CENSUS.md
(+json), ENGRAM.md, V4_TO_V41_MODEL_DIFF.md, CHECKPOINT_CANDIDATES.md.

## 0. Decisions taken in this design (each with its basis; lanes implement these, they do not re-decide)

| # | Decision | Basis |
|-|-|-|
| A1 | V4.1 is a **new package** `vllm/models/deepseek_v41/` (SM70 implementation under `sm70/`). V4 files under `vllm/models/deepseek_v4/` are **not modified**; V4.1 imports only the V4/1Cat pieces listed in §2.1 rule 3. | V4 code keys layer type on `compress_ratio <= 1` (1C:`vllm/models/deepseek_v4/sm70/sparse.py:142,182`; `nvidia/model.py:650,721-747`), applies per-head q-norm (1C:`attention.py:253`) and asserts ratios {4,128} (1C:`compressor.py:198-213`): editing it in place risks silent-wrong V4.1 *and* V4 regressions. Upstream made the same split (up:`vllm/models/deepseek_v41/`). |
| A2 | Dense FP8 (32×32, UE8M0) weights are **dequantized to FP16 at load** through 1Cat's existing SM70 dequant fallback; TurboMind group-32 FP8 kernels are a P5 optimisation. | Exact: dense scale exponents −13…−6 ⇒ every E4M3×2^e is an FP16 value (MODEL_ARCHAEOLOGY §11). Fallback exists and is shape-generic: 1C:`vllm/model_executor/layers/quantization/fp8.py:534-546` (block path), `:608-620` (`_dequantize_block_weight` reads `weight_block_size`), `:646-647` (`F.linear`), `:494-498` (`dequantized_bmm` for wo_a). Cost +0.66 GiB/GPU (§6). |
| A3 | **KV records are FP16 rows holding the model's QAT values exactly**: window KV 512×FP16 (1024 B), compressed KV 512×FP16 (1024 B/latent), index-K 128×FP16 (256 B/latent). The V4 584-B `fp8_ds_mla` record is **not** used. | The 584-B record (448 E4M3 per-64 + 64 BF16 RoPE) cannot hold V4.1's FP8-per-32-incl.-RoPE window QAT nor FP4×E4M3/16 compressed QAT without a second rounding; upstream keeps it on non-SM100 as an approximation (up:`vllm/models/deepseek_v41/attention.py:122-131`). FP16 holds both QAT grids exactly (E2M1×E4M3 ≤ 6 significant bits, range 2^-10…2688; E4M3×2^s exact for s ≥ −15). KV is small for V4.1 (≤ 1.9 KB/token/GPU, §6), so the 2–3.5× record growth costs < 0.5 GiB at 256K. It removes software-FP8 decode from the attention gathers. Packed MXFP8-528/NVFP4-288 records stay a ≥512K-context option. |
| A4 | **HC residual stream stored as BF16** (software type on SM70), all HC math in FP32, sublayer inputs FP16, sublayer outputs (wo_b, MoE) **FP32** until a measured range audit allows FP16. | The reference stores the stream in BF16 (ref:`m.py:960,966`); 1Cat's FP16 stream needed saturation patches for V4 (#658 merged, #715 open; 1C:`vllm/model_executor/kernels/mhc/tilelang.py`). BF16 storage reproduces reference rounding exactly with BF16 range at FP16 cost. |
| A5 | **Engram: CPU-authoritative hashing, per-TP-rank sub-table sharding (rank r gathers sub-tables {r, r+4, …, r+20}), row-parallel `wkv` with an FP32 all-reduce**; tables in a page-packed row file on the root NVMe (15 × 264-B rows per 4 KiB page), page cache = warm tier, admission prefetch. | D2. Sharding removes 4× duplicated gathers/caches, balances n-gram orders across ranks (2-gram rows hit 78.6 %, 4-gram 37.4 %: ENGRAM.md §5) and cuts `wkv` HBM/FLOPs by 4. Page packing makes every row exactly one aligned 4 KiB read (+7.0 GB file vs unpacked). |
| A6 | **Phase A = PP2×TP4 cut at layer 20** (boards A→C), deliberate **UVA expert spill of 116 experts/layer** (10.16 GiB/GPU, 81.2 GiB pinned host). | Layer 20 is the only kv-sharing-legal 2-stage cut (1CAT §9; up:`attention.py:594-601`). Spill size from §6 arithmetic at a 29.5 GiB/GPU target. |
| A7 | **Phase B = PP3×TP4 split [0–13][14–27][28–39]** (boards A→C→B) with a **stage-3 mirror of kv-source 20** (compressed KV + index-K records + candidate blocks shipped in the PP payload). | Layer 14 is a Full kv/index source (nothing crosses 1→2 but the stream); layer 28 is a Reindex layer (no top-k crosses 2→3). The coordinator's 13/14/13 (D3) would need source 8 mirrored into stage 2 plus the top-k of layers 8 and 24 shipped (stages start at Reuse layers 13 and 27). Weights 25.29/24.99/21.66 GiB/GPU (FP16 dense). Board B (180 W) gets the 12-layer stage. |
| A8 | Hetero-TP and TP8 are **benchmarks, not build targets**: TP8 only as an 8-GPU comparator to PP2×TP4 (same spill). | 12 GPUs cannot run TP8 uniformly and TP12 is illegal (64 heads); `[0–19]TP4 + [20–39]TP8` still needs 36 GiB/GPU on stage 1; `[0–13]TP4 + [14–39]TP8` fits but needs non-uniform TP across PP, which vLLM does not support (§5.4). |
| A9 | Bring-up runs **synchronous scheduling**; async scheduling under PP is a P5 item. | Under async PP the non-last ranks only get sampled tokens as a GPU broadcast and keep −1 placeholders on CPU (1C:`vllm/v1/worker/gpu_model_runner.py:9809-9840`, `:2618-2640`); the Engram hash needs token ids on CPU. The async path is specified (§3.6) but not on the critical path. |

---

## 1. V4 → V4.1 component table

"V4 behavior" = how 1Cat runs DeepSeek-V4-Flash on SM70 today. "V4.1 behavior" = the official reference.
Owner tags: [CORE] [ATTN] [MOE] [ENGRAM] [REF] = lanes of §2. Phase tags: P2 = needed for first correct token,
P5 = optimisation after correctness.

| Component | V4 behavior (1Cat SM70) | V4.1 behavior (official) | Reusable from 1Cat? | Modification | New implementation |
|-|-|-|-|-|-|
| **Dense FP8 weights** (wq_a, wq_b, wkv, wo_a, wo_b, shared expert, indexer wq_b, Engram wkv, DSpark main_proj) | E4M3, 128×128 UE8M0 blocks → TurboMind W8A16 group-128 HMMA; only `[128,128]` admitted (1C:`fp8.py:365-370`, `sm70_fp8.py:445-446`, kernels `sm70_884_8.cu:110-140`) | E4M3, **32×32** UE8M0 blocks (`weight_block_size [32,32]`); exponents −13…−6 (Engram `wkv` −18…−8) | **Yes**: the SM70 dequant-to-FP16 fallback (1C:`fp8.py:534-546,608-620,646-647`, `dequantized_bmm` `:494-498`) | V4.1 quant config routes every dense FP8 linear to the fallback (no env); Engram `wkv` gets a 2^10 scale bias (§4) | P5: TurboMind E4M3 `GroupSizeV=32` instantiations + `fp8_sm70_prepare(group=32)` with UE8M0 byte scales as `Config_MXF4` does (1C:`sm70_884_4.cu:303-311`) [MOE, P5] |
| **MXFP4 routed experts** | E2M1 + E8M0/32, 256 experts, inter 2048 → TP4 512; TurboMind grouped MoE, compact/direct-order decode, QPN-M1 kernel; allow-list hidden 4096/inter 2048/256/top-6 (1C:`mxfp4_sm70_moe.py:40-43,246-286`); QPN-M1 shapes pinned (1C:`csrc/sm70_turbomind/ops/mxfp4_qpn_m1_sm70.cu:1182-1203`) | identical byte format; **384 experts, inter 2304 → TP4 576** (= 18×32), hidden 5120; DSpark 128 experts top-3 | **Yes** (format, repack `_prepare_mxfp4_sm70_experts`, TurboMind grouped family) | contract generalised to {hidden 5120, inter 2304, E ∈ {384,128}, top-k ∈ {6,3} }; tile check for W13 N=1152/K=5120 and W2 N=5120/K=576; QPN-M1 off until re-templated; skinny split-K list += {4,6,9,12} | resident/UVA **spill partition** for Phase A (§5.2) [MOE] |
| **BF16 tensors** (embed, head, norms, gate.weight, compressor, indexer wk/weights_proj, Engram q/k) | BF16 → FP16 at load (`--dtype half`, 1C:`nvidia/model.py:1349-1356`) | same set plus new tensors; head kept FP32 in ref (ref:`m.py:1006`) | Yes | head: FP16 weights, **FP32 logits** (FP16×FP16 products exact in FP32); gate: FP16 weight, FP32 accumulate; compressor: FP16 operands, FP32 accumulate | — |
| **Checkpoint names / loader** | `_make_deepseek_v4_weights_mapper` (1C:`nvidia/model.py:1301-1337`), hash-route `tid2eid` | HF names = inference names (`.scale`, no `model.` prefix); new `attn.indexer.{wk,k_norm}`, `ffn.gate.bias_vl`, `layers.{1,14}.engram.*`, `mtp.{0..2}.*`, `vision.*` | pattern yes | — | V4.1 mapper + `skip_checkpoint_weight()` (hook at 1C:`model_loader/default_loader.py:385`) skipping `*.engram.embed.*` (2 × 101 GB in shards 47/48, which also hold the needed `wkv`/`q_weight`/`k_weight`), `vision.*`, `aligner.*`, `image_*`, `*.bias_vl`, `mtp.*` (until DSpark) [CORE] |
| **Embeddings** | `VocabParallelEmbedding` FP16, broadcast to 4 HC copies in layer 0 (`mhc_pre_broadcast_tilelang`) | vocab 129,280 × 5120, ×4 copies, identity pre-mix (ref:`m.py:1253-1260`) | Yes | output cast to the BF16 stream | — |
| **Hyper-connections** | each sublayer applies its *own* pre immediately (`mhc_fused_post_pre_tilelang`, 1C:`nvidia/model.py:892-981`); final collapse through learned `hc_head_*` (1C:`tilelang.py:1111`) | **single-pass shift**: sublayer k's `pre` collapses sublayer k+1's input; block returns `ffn_pre`; final collapse = `hc_pre(h, last ffn_pre)`, **no `hc_head_*`** (ref:`m.py:968-994,1268`) | Sinkhorn/split body identical (ref:`k.py:406-474` = V4); fused kernels **no** (they apply the wrong pre and need FP16 streams) | — | `common/hc.py`: `hc_mixes/hc_pre/hc_post/hc_collapse` on a BF16 stream with FP32 math [CORE, P2]; fused Triton shifted post+pre (semantics of up:`nvidia/model.py:564-611` `mhc_shifted_post_pre`) [CORE, P5] |
| **HC stream / PP hidden dtype** | FP16 `[T,4,4096]` | BF16 `[T,4,5120]` (+ `pre_mix [T,4]` FP32) | — | — | decision A4 [CORE] |
| **Attention q path** | `q = wq_b(q_norm(wq_a x))`, **per-head RMS q-norm**, RoPE; fused SM70 kernel (1C:`sm70/qnorm_rope_kv_fp8_insert.py:23-269`) | q_lora **1280**, **no per-head q-norm** (ref:`m.py:770-772`) | projections yes; fused kernel no (q-norm hard-wired) | — | `sm70/q_rope_kv_insert.py`: q RoPE (last 64 dims, interleaved) + window-KV QAT + FP16 insert [ATTN] |
| **wkv / MQA latent 512** | `kv_norm(wkv x)`; `fused_wqa_wkv` replicated (`disable_tp`, 1C:`nvidia/model.py:664-670`) | same (wkv 512×5120, one K=V latent, 448 NoPE + 64 RoPE) | Yes | — | — |
| **Window KV (FP8 q-dq)** | QAT = E4M3 per-64 on NoPE + BF16 RoPE = the 584-B record exactly | QAT = **E4M3 + UE8M0 per 32 over all 512 incl. RoPE** (ref:`m.py:707`, `act_quant(...,inplace=True)`) | SWA paged cache + metadata (`DeepseekV4SWACache`, `DeepseekSparseSWAMetadata`, 1C:`vllm/v1/attention/backends/mla/sparse_swa.py:161-199`) | SWA spec with `model_version="deepseek_v41"`, 1024-B FP16 record | bit-exact `fp8_block32_qdq` encoder (§4.2) [ATTN] |
| **Compressor ratios 0/1/2** | per compressed layer; ratio 4 (overlap `coff=2` + `ape`) or 128; `assert compress_ratio in [4,128]` (1C:`compressor.py:198-213`); FP32 state ring as `SlidingWindowMLASpec` (1C:`compressor.py:164-232`) | only on kv sources; **ratio 2** = per-channel softmax pooling of 2 tokens in FP32, no overlap/ape; **ratio 1** = `norm(wkv x)`, no gate/state; returns **pre-RoPE** latent; RoPE at the group's first position; cache write = FP4 E2M1 + E4M3/16 QAT (ref:`m.py:429-485,739-763`) | state-cache spec mechanism, Triton store structure | — | `deepseek_v41/compressor.py` (port of up:`deepseek_v41/compressor.py:174-251` semantics on FP16/FP32) [ATTN] |
| **Modes Full / Reindex / Reuse** | every C4 layer owns compressor + indexer + cache | Full = 2, 8, 14, 20; Reindex = 24, 28, 32, 36; Reuse = all other layers ≥ 2; SWA-only = 0, 1 (ref:`m.py:653-661,725-726`) | — | — | `LayerTopology` (§3.1) + consumer wiring [ATTN] |
| **Shared KV sources** | none | `kv_source_layer_ids [2,8,14,20]`; consumers read `max(s ≤ layer)`; upstream resolves via the static forward context and **refuses PP cuts inside a group** (up:`attention.py:322-351,573-601`) | static-forward-context lookup pattern | — | same lookup + **stage mirror** of source 20 for PP3 (§3.5) [ATTN] |
| **Indexer** | 64 heads, own ratio-4 compressor with Hadamard, FP8 K 132 B, SM70 cuBLAS + Triton epilogue + `persistent_topk` (1C:`sm70/indexer.py`, `layers/sparse_attn_indexer.py:170-447`) | **32 heads × 128**; q = `wq_b(qr)` + RoPE + **FP4 E2M1/E8M0-32 QAT**; **K = `k_norm(wk(pre-RoPE latent))`** + RoPE + FP4 QAT, only on kv sources; **no Hadamard**; score = Σₕ relu(q·k)·w, w = `weights_proj(x)`·128^-½·32^-½; top-512 re-sorted by position (ref:`m.py:488-580`) | scoring/top-k kernels (head count is a parameter), `persistent_topk` k=512 | indexer is **replicated** per TP rank (as 1Cat V4, 1C:`attention.py:939-951`) — no score all-reduce | `deepseek_v41/indexer.py`, `sm70/indexer_kernels.py` (FP16 K records) [ATTN] |
| **Candidate block mask (>16K)** | none | layer 20 keeps top-2048 blocks of 8 (newest block pinned +inf); 24/28/32/36 mask to it; no-op at ≤ 16,384 tokens (ref:`m.py:569-575,583-610`) | — | — | port up:`kernels/attention/dsa/candidate_blocks.py` (pure Triton, no `tl.dot`) [ATTN] |
| **Sparse attention kernel** | `DeepseekV4SM70SparseImpl`: cuBLAS gather+BMM (#816) or paged Triton FP8 decode; C4/C128/SWA branches, `ValueError` otherwise (1C:`sm70/sparse.py:115-260`) | per query ≤ 128 window + ≤ 512 compressed rows, online softmax, **running max init −1e30** (all-invalid row → 0 output) (ref:`k.py:310-403`) | gather+BMM structure, online-softmax Triton | FP16 record gather (no software-FP8 decode); index→slot via the **source's** block table | `sm70/sparse.py` V4.1 impl [ATTN] |
| **attn_sink** | per-head FP32, added to the softmax denominator | identical (ref:`k.py:382-385`) | Yes | — | — |
| **Grouped o-proj** (o_lora 1024 × 8) | inverse RoPE (Triton) + grouped TurboMind FP8 BMM wo_a + wo_b (1C:`sm70/projection.py:84-100`) | identical geometry (TP4: 2 groups of 1024×4096 per rank), wo_a FP8 32×32 | inverse-RoPE kernel yes; TurboMind BMM no (block 128) | wo_a via `dequantized_bmm` einsum FP16 (1C:`fp8.py:649-656`) | P5 grouped g32 BMM [MOE, P5] |
| **RoPE / YaRN** | θ 160000 + YaRN for compressed layers, base θ with factor 1 when `compress_ratio <= 1` (1C:`nvidia/model.py:712-747`) | ratio > 0 (layers 2–39) θ = `compress_rope_theta` 160000 + YaRN ×16 over 65,536, β 32/1, **no mscale**; ratio 0 (0, 1, DSpark) θ 10000, no YaRN (ref:`m.py:680-698`) | rotary class (`deepseek_yarn`, `mscale=0`) | selection on **`> 0`** (port up:`deepseek_v41/common/rope.py`) | `common/rope.py` [ATTN] |
| **MoE router** | `GateLinear` FP32 out, sqrtsoftplus, noaux bias, **hash-routed layers** via `tid2eid` (1C:`nvidia/model.py:443-466`); top-k CUDA kernel has `case 384` (1C:`csrc/moe/topk_softplus_sqrt_kernels.cu:588,611`) | sqrtsoftplus, bias for selection only, weights /(Σ+1e-20) ×1.5, `gate_temp` 1.0, **no hash layers**; `bias_vl` for image tokens only (ref:`m.py:792-827`) | top-k kernel yes | `num_hash_layers=0` explicitly (V4 reads it unguarded, 1C:`nvidia/model.py:453`); `bias_vl` not loaded | `sm70/moe.py` wrapper [MOE] |
| **Routed experts** | TP-sharded inter dim; routing weight applied after w2 | routing weight applied to `h` **before** w2 (ref:`m.py:849-850`) | Yes | either order allowed (linear; §4.3) | spill partition [MOE] |
| **Shared expert** | FP8 gated MLP, TP-sharded, swiglu clamp (`DeepseekV4MLP`, 1C:`nvidia/model.py:70-118`) | 1 shared expert 5120↔2304, FP8 32×32, same clamp (ref:`m.py:887`) | Yes (with FP16 weights) | added inside the FP32 combine before the single rounding | — |
| **swiglu_limit** | 10, `silu_and_mul_with_clamp` (1C:`mxfp4_sm70_moe.py:752-760`) | up clamped ±10, gate ≤ 10 (ref:`m.py:840-848`) | Yes | — | — |
| **Engram** | none (closest: Qwen4Exp PLE pinned/disk tiers, 1C:`models/qwen4_exp/nvidia/ple_layer.py`, `csrc/ple_disk_rows.cpp`, `v1/ple_offload/`) | layers 1, 14; 24 rows/token/layer (3 n-gram orders × 8 heads) of 256 FP8 + 8 UE8M0; `wkv` 6144→25600; per-HC sigmoid(signed-sqrt) gate; applied before the block (ref:`m.py:296-365,1262-1263`; ref:`e.py`) | patterns (disk-row C++ gather, pinned staging, UVA view `get_accelerator_view_from_cpu_tensor`) | — | `common/engram.py`, `common/engram_host.py`, `sm70/engram_kernels.py`, repack tool [ENGRAM] |
| **DSpark / MTP** | `nvidia/dspark.py` (Markov/confidence heads, N=7, target layers 40–42, `main_proj` 2^-6 pre-scale, 1C:`nvidia/dspark.py:140-146`) | 3 SWA-only blocks, **128 experts top-3**, target layers **37–39** (block inputs averaged over HC), block 5 with noise token 128799, Markov rank 256, confidence head (ref:`m.py:1020-1157`) | structure, verifier infra | target ids, MoE shape, tied embed (embed replica on last stage) | `sm70/dspark.py` [CORE, P5] |
| **Norms (eps 1e-20)** | eps 1e-6 | **1e-20** for every RMSNorm, HC mixes and Engram rstd (ref:`m.py:281-293`) | vLLM `rms_norm` (FP32 variance, float eps) yes | **FP32 variance + FP32 eps mandatory**; zero (padding) rows must give 0, not NaN | — |
| **TP collectives** | TP4 push all-reduce for exact FP16 payloads 8 KiB (= 4096 halves) / 80 KiB (1C:`csrc/custom_all_reduce.cuh:98,186-240`) | — | generic custom-AR / NCCL yes | FP32 `[T,5120]` (20 KiB at B1) all-reduces until the range audit (§4) | new: Engram partial-`wkv` FP32 all-reduce (100 KiB/token); P5 exact-size admission for 5120-half payloads [CORE, P5] |
| **PP** | static metadata-free transfer for exactly `{"hidden_states": [1,4,4096] FP16}` (1C:`vllm/v1/worker/gpu_worker.py:1085-1146`) | payload `hidden_states [T,4,5120]` BF16 + `pre_mix [T,4]` FP32 (as up:`nvidia/model.py:841-866`); PP3 2→3 adds kv20 records + candidates | tensor-dict PP path yes | schema-driven static path | `StagePlan`, payload keys (§3.1) [CORE] |
| **KV cache manager** | `MLAAttentionSpec` 584 B for `model_version="deepseek_v4"` (1C:`vllm/v1/kv_cache_interface.py:358-427`); per-layer compressed specs; V4 tensor bucketing (1C:`vllm/v1/core/kv_cache_utils.py:1379-1433`); specs merged by layer name across PP workers (`:3010-3031`) | specs only on kv sources (consumers return none); SWA on every layer | merging-by-name across PP workers is exactly what the mirror needs | `model_version="deepseek_v41"` page sizes; V4.1 grouping | mirror registration (§3.5) [ATTN] |
| **CUDA graphs** | breakable graphs; FULL decode B1–B8; decode context buckets = `index_topk × min(ratio>0) × {1,2,8,32,64}` from hf_config (1C:`vllm/v1/cudagraph_dispatcher.py:44-80`) | — | **Yes, unchanged**: the generic derivation yields 512 × {1,2,8,32,64} for V4.1 | attention, Engram wait, mirror ingest are eager breaks | — |
| **Tokenizer / chat template / reasoning parser** | `tokenizers/deepseek_v4{,_encoding}.py`; reasoning parser = deepseek_v3 (1C:`vllm/reasoning/__init__.py:31-34`) | DSML with leading space, numeric reasoning effort 1–100 (low 50 / medium 62 / high 75 / max 100), mid-conversation `<｜System｜>`; `chat_template.jinja` (2026-10-01) | — | — | port up:`vllm/tokenizers/deepseek_v41{,_encoding}.py`, `vllm/reasoning/deepseek_v41_engine_reasoning_parser.py` [CORE, P5] |
| **Tool parser** | `tool_parsers/deepseekv4_tool_parser.py` (DSML corruption on SM70 open, #597) | V4.1 DSML | — | — | port up:`vllm/tool_parsers/deepseekv41_engine_tool_parser.py` + test against #597's repro [CORE, P5] |
| **Config / registry / quant claim** | registry 1C:`model_executor/models/registry.py:102`; quant claim only for `model_type == "deepseek_v4"` (1C:`models/deepseek_v4/quant_config.py:120-132`) | `DeepseekV41ForCausalLM`, `model_type deepseek_v41` / `deepseek_v41_text` (nested `text_config`) | — | — | port up:`transformers_utils/configs/deepseek_v41.py` (flattening); registry; `DeepseekV41FP8Config` [CORE] |
| **Activation quant in `linear()`** | n/a (V4 ref used per-128 FP8 act quant) | every FP8/FP4 GEMM input is FP8-quantised per 32 (ref:`m.py:181-207`) | — | **skipped** on V100 (FP16 activations are more precise); emulated only in the L-REF `ref-exact` mode | — |
| **Vision tower, `bias_vl`, CED replay** | — | ViT 32 layers; decoder SWA bounded replay is an approximate deployment optimisation, not in the reference | — | — | out of scope (load-skip); bounded replay = P5+ opt-in |

---

## 2. Package layout and lane ownership

### 2.1 Rules

1. **One owner per file.** A lane edits only files it owns (tables below). A change needed in another lane's file is
   requested through the coordinator as a design update to this document (section + exact signature), then made by
   the owner. Cross-lane types live in one frozen file, `common/contracts.py` (§3.1), owned by L-CORE and written
   verbatim from §3 in L-CORE's first commit **C0**; every other lane branches from C0.
2. **Worktrees:** after L-BUILD finishes, each lane works in its own worktree on branch
   `ds41/<lane>` cut from `ds41/sm70-v41` + C0, commits per completed sub-item, parks WIP under `refs/checkpoints/`.
   The coordinator merges in the order C0 → {ATTN, MOE, ENGRAM in any order} → CORE integration. Python-only lanes
   reuse the main tree's compiled extensions (copy `vllm/*.so` and `vllm/**/_*.so` from `[private path]`
   into the worktree; never rebuild for Python-only changes). L-MOE (and L-CORE at P5) change `csrc/` and build in
   their own worktree with `TORCH_CUDA_ARCH_LIST=7.0`, ccache, `MAX_JOBS=10`.
3. **V4 is frozen.** No lane edits `vllm/models/deepseek_v4/**`. V4.1 may *import* these V4/1Cat pieces read-only:
   `deepseek_v4.common.ops.fp8_software` (software E4M3 helpers), `deepseek_v4.sm70.projection` (inverse-RoPE
   Triton kernel), `deepseek_v4.sm70.gemv` (FP16/FP13 GEMV — shape-gated, falls back), `deepseek_v4.nvidia.model.DeepseekV4MLP`
   (shared-expert MLP), `model_executor.layers.fused_moe` + the SM70 MXFP4 method (through L-MOE's factory),
   `v1.attention.backends.mla.sparse_swa` SWA metadata (through L-ATTN), mHC kernels are **not** imported (A4).
   A CI grep gate (owned by L-CORE: `tests/models/deepseek_v41/test_import_boundary.py`) fails if `deepseek_v41/`
   imports `deepseek_v4.attention`, `deepseek_v4.compressor`, `deepseek_v4.sm70.sparse`, `deepseek_v4.sm70.indexer`
   or `deepseek_v4.sm70.qnorm_rope_kv_fp8_insert`.
4. **Shared hot files** (more than one lane needs a change) have exactly one owner; the needed changes are fully
   specified here so the owner can make them once:

| Shared file (1Cat tree) | Owner | Changes other lanes need (specified here) |
|-|-|-|
| `vllm/v1/kv_cache_interface.py` | L-ATTN | `MLAAttentionSpec.real_page_size_bytes` / `SlidingWindowMLASpec`: branch `model_version == "deepseek_v41"` → `storage_block_size × head_size × 2` (FP16 rows); no other model touched |
| `vllm/v1/core/kv_cache_utils.py` | L-ATTN | V4.1 grouping: generalise `_get_kv_cache_config_deepseek_v4` (1C:`:1379-1433`) to `model_version in {"deepseek_v4","deepseek_v41"}`; mirror layers need nothing (merged by name, `:3010-3031`) |
| `vllm/model_executor/models/registry.py` | L-CORE | `"DeepseekV41ForCausalLM": ("vllm.models.deepseek_v41", "DeepseekV41ForCausalLM")` |
| `vllm/transformers_utils/configs/__init__.py`, `vllm/transformers_utils/config.py` | L-CORE | register `deepseek_v41` / `deepseek_v41_text` → `DeepseekV41Config` |
| `vllm/model_executor/layers/quantization/fp8.py` | L-CORE (P2); hands to L-MOE at P5 | none at P2 (the V4.1 quant config forces the fallback per layer); P5: admit `[32,32]` to TurboMind once L-MOE's g32 kernels pass |
| `vllm/model_executor/layers/quantization/mxfp4_sm70_moe.py`, `mxfp4.py` (factory), `fused_moe/experts/skinny_sm70_moe.py`, `csrc/sm70_turbomind/**` | L-MOE | — |
| `vllm/v1/worker/gpu_worker.py` | L-CORE | (a) schema-driven static PP path (replace the `[1,4,4096]` check at 1C:`:1085-1146` with the model's declared schema); (b) call `engram_service.begin_step(plan)` **before** the PP receive (§3.6) |
| `vllm/v1/worker/gpu_model_runner.py` | L-CORE | (a) `engram_service.bind_batch(layout)` after `_prepare_inputs`, before the forward; (b) P5 async-PP: pinned D2H of `prev_sampled_token_ids` + event inside `_pp_receive_prev_sampled_token_ids_to_input_batch` (1C:`:9809-9840`), handed to `bind_batch` |
| `csrc/custom_all_reduce.cuh`, `vllm/distributed/device_communicators/custom_all_reduce.py` | L-CORE (P5) | exact-size admission for 5120-half (10 KiB) and FP32 20 KiB payloads |
| `vllm/v1/cudagraph_dispatcher.py` | nobody | unchanged: the generic bucket rule already gives 512 × {1,2,8,32,64} (1C:`:44-80`) |
| `vllm/envs.py` | nobody | V4.1 knobs live in `deepseek_v41/knobs.py` (L-CORE helper); each lane declares its own `VLLM_DS41_<LANE>_*` flags in its own module through `knobs.env_bool/env_int/env_str` |

### 2.2 Package tree (mirrors up:`vllm/models/deepseek_v41/` where sensible; `sm70/` replaces upstream's `nvidia/`)

```
vllm/models/deepseek_v41/
  __init__.py                 CORE    exports DeepseekV41ForCausalLM (imports sm70.model on CUDA cc 7.x; raises elsewhere)
  knobs.py                    CORE    env helpers (no flags of other lanes)
  quant_config.py             CORE    DeepseekV41FP8Config: claims model_type deepseek_v41/_text; dense FP8 -> SM70 FP16
                                      dequant fallback; FusedMoE -> moe.make_v41_moe_method (MOE's factory)
  attention.py                ATTN    DeepseekV41Attention (q path, window KV, source/consumer wiring, sink, o-proj)
  compressor.py               ATTN    DeepseekV41Compressor (ratio 1 / 2, FP32 state cache for ratio 2)
  indexer.py                  ATTN    DeepseekV41Indexer (+ index-K writer on kv sources, candidate source/consumer)
  kv_mirror.py                ATTN    DeepseekV41KVSourceMirror (stage-local replica of source 20)
  common/
    __init__.py               CORE
    contracts.py              CORE    FROZEN cross-lane types/constants (verbatim from §3.1)
    topology.py               CORE    layer_topology(), make_stage_plan(), legal-cut check
    hc.py                     CORE    hc_mixes / hc_pre / hc_post / hc_collapse (BF16 stream, FP32 math)
    rope.py                   ATTN    build_v41_rope(config, compress_ratio)  (port of up common/rope.py)
    qat.py                    ATTN    bit-exact QAT q-dq: fp8_block32_qdq, fp4_e8m0_qdq, fp4_e4m3_qdq (Triton + torch twin)
    candidate_blocks.py       ATTN    port of up kernels/attention/dsa/candidate_blocks.py
    engram.py                 ENGRAM  EngramLayout, token map, hash (numpy), DeepseekV41Engram module
    engram_host.py            ENGRAM  EngramHostService (row file, thread-pool pread, pinned staging, async H2D, prefetch)
  sm70/
    __init__.py               CORE
    model.py                  CORE    DeepseekV41DecoderLayer, DeepseekV41Model, DeepseekV41ForCausalLM, load_weights, PP
    hc_kernels.py             CORE    (P5) fused Triton shifted post+pre on the BF16 stream
    dspark.py                 CORE    (P5) DSpark draft (3 blocks, 128x top-3, Markov + confidence heads)
    moe.py                    MOE     DeepseekV41MoE (router, routed experts via FusedMoE, shared expert, spill partition)
    gemv.py                   MOE     decode GEMV fast paths for K=5120 (gate N=384, shared expert, compressor/indexer small
                                      projections), copied from 1C:deepseek_v4/sm70/gemv.py (K=4096-only); ATTN may import it
    sparse.py                 ATTN    DeepseekV41SM70SparseImpl + backend (FP16 records; decode + prefill)
    sparse_kernels.py         ATTN    Triton/cuBLAS gather kernels for FP16 records
    q_rope_kv_insert.py       ATTN    q RoPE + window-KV QAT + FP16 paged insert
    indexer_kernels.py        ATTN    FP16 index-K scoring (prefill tiles, decode), candidate mask application, top-k glue
    engram_kernels.py         ENGRAM  Triton: row decode (E4M3 x UE8M0 -> FP16), fused post-wkv gate into the BF16 stream
csrc/engram_rows.cpp          ENGRAM  pread/O_DIRECT thread-pool row gather (pattern of 1C:csrc/ple_disk_rows.cpp); built as a
                                      torch extension loaded at runtime (no CMakeLists edit at P2)
tools/ds41_engram_repack.py   ENGRAM  offline: shards 47/48 -> page-packed row files on / (root NVMe)
vllm/transformers_utils/configs/deepseek_v41.py   CORE  port of up file (flattens text_config, load-skips vision)
vllm/tokenizers/deepseek_v41.py, deepseek_v41_encoding.py,
vllm/reasoning/deepseek_v41_engine_reasoning_parser.py,
vllm/tool_parsers/deepseekv41_engine_tool_parser.py              CORE (P5)  ports of the up files
tests/models/deepseek_v41/   each lane owns test_<lane>_*.py; CORE owns conftest.py + test_import_boundary.py
```

L-REF lives **outside** 1Cat: `[private path]` (owner L-REF): `ref_model.py` (pure-torch FP32
port of ref `m.py` semantics, no TileLang), `ref_qat.py` (bit-exact QAT twins), `ref_engram.py` (hash + rows from the
official shards), `weights.py` (lazy per-layer safetensors reader; MXFP4/FP8 → FP32 on demand), `golden.py` (capture
CLI), `compare.py` (metrics + gates of §4.5), `prompts/` (deterministic smoke set), `xcheck.py` (llama.cpp
cross-check driver).

### 2.3 Lane scopes (what each lane owns end to end)

| Lane | Owns | Delivers (P2) | Depends on |
|-|-|-|-|
| **L-CORE** | contracts/topology/hc/knobs, config + registry + quant claim, `sm70/model.py` (decoder layer, model, causal LM, `load_weights` incl. dense FP8→FP16 through the fallback, `skip_checkpoint_weight`), PP stage plan + payload + static PP path, worker/runner hooks, server integration, launch flags, frontend ports (P5), DSpark (P5), custom-AR exact sizes (P5), fused HC kernels (P5) | C0 within the first session; layer-subset and full-model forward on PP2 and PP3 | ATTN/MOE/ENGRAM modules behind §3 signatures |
| **L-ATTN** | attention, compressor, indexer, candidate mask, rope, QAT ops, KV specs + page sizes + grouping, SWA/compressed/index-K records, sparse MLA impl, mirror, decode/prefill workspaces | `DeepseekV41Attention` for all four layer modes passing §4.5 per-op gates; mirror ingest/export | C0; L-REF golden vectors |
| **L-MOE** | router, routed experts (TurboMind MXFP4 shape generalisation, decode path, QPN-M1 re-template later), shared expert, spill partition, decode GEMV fast paths (`sm70/gemv.py`, K=5120), P5 TurboMind g32 dense FP8 | `DeepseekV41MoE` (E=384/top-6 and E=128/top-3) passing bitwise A/B vs dequantised reference; spill partition | C0; L-REF |
| **L-ENGRAM** | hash + layout, repack tool, row file format, host service (pread pool, pinned staging, async H2D, admission prefetch), Engram module + kernels, Engram `wkv` loading (biased FP16, row-parallel) | repacked row files on `/`; `DeepseekV41Engram` + service passing exact hash/row tests and §4.5 module gates | C0; L-REF (hash/row golden) |
| **L-REF** | FP32 golden reference, QAT twins, golden capture + compare tooling, prompt set, the llama.cpp cross-check driver, range audit (§4.4) | golden vectors for every op in §4.5 at layers {0,1,2,3,14,20,21,24,28} + full-model short goldens; range audit table | official weights on disk (ds41-pull) |

---

## 3. Exact interfaces between lanes

Conventions: TP = 4 inside a stage; `T` = tokens in the step, flattened in runner order (decodes first, as the V4
MLA backends require); `T_pad` = CUDA-graph padded count (rows `T:T_pad` are zero and must stay finite).
Shapes are **per TP rank**. Every signature below is implemented verbatim (names, argument order, return types);
bodies are the owner's business.

### 3.1 `common/contracts.py` (L-CORE, frozen; written in C0 exactly as below)

```python
# vllm/models/deepseek_v41/common/contracts.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# ---- model constants: asserted against hf_config in DeepseekV41Model.__init__ ----
HIDDEN = 5120
HC = 4
N_HEADS = 64
HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = 448
Q_LORA = 1280
O_GROUPS = 8
O_LORA = 1024
WINDOW = 128
IDX_HEADS = 32
IDX_DIM = 128
IDX_TOPK = 512
CAND_SOURCE = 20
CAND_BLOCK = 8
CAND_TOPK_BLOCKS = 2048
KV_SOURCES = (2, 8, 14, 20)
INDEX_SOURCES = (2, 8, 14, 20, 24, 28, 32, 36)
ENGRAM_LAYERS = (1, 14)
ENGRAM_SUBTABLES = 24          # (max_ngram 4 - 1) orders x 8 heads, order-major: s = (n - 2) * 8 + head
ENGRAM_ROW_BYTES = 264         # 256 x E4M3 + 8 x UE8M0 (one per 32 channels)
ENGRAM_HEAD_DIM = 256
N_EXPERTS = 384
TOP_K = 6
MOE_INTER = 2304
ROUTED_SCALE = 1.5
SWIGLU_LIMIT = 10.0
NORM_EPS = 1e-20
HC_EPS = 1e-6
HC_SINKHORN_ITERS = 20
COMPRESS_ROPE_THETA = 160000.0
ROPE_THETA = 10000.0

# ---- dtypes (numerics contract, PORT_DESIGN §4) ----
STREAM_DTYPE = torch.bfloat16        # HC residual stream [T, HC, HIDDEN]; software BF16 on SM70
ACT_DTYPE = torch.float16            # sublayer inputs and GEMM operands
SUBLAYER_OUT_DTYPE = torch.float32   # attention (after wo_b) and MoE outputs, TP-all-reduced
MIX_DTYPE = torch.float32            # pre / post / comb

# ---- KV records: FP16 rows holding QAT values exactly (PORT_DESIGN A3) ----
KV_RECORD_DTYPE = torch.float16
SWA_RECORD_DIM = 512                 # window K=V: E4M3 x UE8M0/32 QAT over all 512 dims      -> 1024 B/token/layer
CKV_RECORD_DIM = 512                 # compressed latent: E2M1 x E4M3/16 QAT, RoPE applied   -> 1024 B/latent
IK_RECORD_DIM = 128                  # index key: E2M1 x UE8M0/32 QAT, RoPE applied          ->  256 B/latent
KV_MODEL_VERSION = "deepseek_v41"
SWA_CACHE_DTYPE_STR = "v41_fp16_qat_swa"
CKV_CACHE_DTYPE_STR = "v41_fp16_qat_ckv"
IK_CACHE_DTYPE_STR = "v41_fp16_qat_ik"
CAND_ALL = -2                        # cand row[0] sentinel: <= CAND_TOPK_BLOCKS blocks reachable, no masking


@dataclass(frozen=True)
class LayerTopology:
    layer_id: int
    compress_ratio: int              # 0 = SWA only, 1 = full-length compressed, 2 = 2:1 pooled
    mode: str                        # "swa" | "full" | "reindex" | "reuse"
    kv_source: int | None            # max(s in KV_SOURCES if s <= layer_id) when compress_ratio > 0
    index_source: int | None         # max(s in INDEX_SOURCES if s <= layer_id) when compress_ratio > 0
    owns_compressor: bool            # layer_id in KV_SOURCES
    owns_indexer: bool               # layer_id in INDEX_SOURCES
    is_candidate_source: bool        # layer_id == CAND_SOURCE
    uses_candidates: bool            # owns_indexer and CAND_SOURCE < layer_id
    has_engram: bool                 # layer_id in ENGRAM_LAYERS
    rope_theta: float                # COMPRESS_ROPE_THETA if compress_ratio > 0 else ROPE_THETA
    yarn: bool                       # compress_ratio > 0


@dataclass(frozen=True)
class StagePlan:
    pp_rank: int
    pp_size: int
    first_layer: int                 # inclusive
    last_layer: int                  # inclusive (backbone layers only; DSpark rides on the last stage)
    mirrored_kv_sources: tuple[int, ...]   # kv sources on an earlier stage replicated here (v1: () or (20,))
    exports_kv_sources: tuple[int, ...]    # kv sources on this stage that a later stage mirrors (v1: () or (20,))
    engram_layers: tuple[int, ...]


# legal first layers of a stage (v1): SWA-only layer 1, index sources; any source referenced from
# before the cut must be 20 (mirrored). PP2 -> cut 20; PP3 -> cuts (14, 28) is the only split with <= 17 layers/stage.
LEGAL_STAGE_STARTS = (1, 2, 8, 14, 20, 24, 28, 32, 36)


@dataclass
class SharedAttnBuffers:
    """Allocated once per stage by DeepseekV41Model, handed to every attention layer and the mirror."""
    topk_indices: torch.Tensor           # [max_num_batched_tokens, IDX_TOPK] int32: logical compressed positions
                                         #   (ascending, -1 pad) published by the latest index source, read by Reuse layers
    candidate_blocks: torch.Tensor       # [max_num_batched_tokens, CAND_TOPK_BLOCKS] int32: block ids (ascending, -1 pad);
                                         #   row[0] == CAND_ALL => no mask. Written by layer 20 or the mirror.
    export_ckv: torch.Tensor | None      # [max_num_batched_tokens, CKV_RECORD_DIM] fp16 (stage exporting source 20 only)
    export_ik: torch.Tensor | None       # [max_num_batched_tokens, IK_RECORD_DIM] fp16 (stage exporting source 20 only)


def allocate_shared_attn_buffers(max_num_batched_tokens: int, stage: StagePlan,
                                 device: torch.device) -> SharedAttnBuffers:
    exports = CAND_SOURCE in stage.exports_kv_sources
    return SharedAttnBuffers(
        topk_indices=torch.full((max_num_batched_tokens, IDX_TOPK), -1, dtype=torch.int32, device=device),
        candidate_blocks=torch.full((max_num_batched_tokens, CAND_TOPK_BLOCKS), -1, dtype=torch.int32, device=device),
        export_ckv=(torch.zeros((max_num_batched_tokens, CKV_RECORD_DIM), dtype=KV_RECORD_DTYPE, device=device)
                    if exports else None),
        export_ik=(torch.zeros((max_num_batched_tokens, IK_RECORD_DIM), dtype=KV_RECORD_DTYPE, device=device)
                   if exports else None),
    )


# ---- PP payload keys (IntermediateTensors) ----
PP_KEY_HIDDEN = "hidden_states"      # [T, HC, HIDDEN] STREAM_DTYPE        every boundary
PP_KEY_PRE_MIX = "pre_mix"           # [T, HC] float32                     every boundary
PP_KEY_KV20_CKV = "kv20_ckv"         # [T, CKV_RECORD_DIM] float16         boundary into a stage mirroring 20
PP_KEY_KV20_IK = "kv20_ik"           # [T, IK_RECORD_DIM] float16          idem
PP_KEY_CAND20 = "cand20"             # [T, CAND_TOPK_BLOCKS] int32         idem


# ---- Engram step plan (worker/runner -> EngramHostService) ----
@dataclass(frozen=True)
class EngramReqStep:
    req_id: str
    start_pos: int                       # absolute position of the first scheduled token (= num_computed_tokens)
    num_tokens: int                      # scheduled tokens of this request this step (incl. spec tokens)
    token_ids: np.ndarray | None         # [num_tokens] int32 raw token ids; None => unknown until bind_batch (async PP)
    prompt_token_ids: np.ndarray | None  # full prompt; set when the request is new or resumed this step


@dataclass(frozen=True)
class EngramStepPlan:
    step_id: int
    reqs: tuple[EngramReqStep, ...]      # scheduler order
    finished_req_ids: frozenset[str]


@dataclass(frozen=True)
class EngramBatchLayout:
    step_id: int
    req_order: tuple[str, ...]           # runner's flattened order (decodes first)
    query_start_loc: np.ndarray          # [num_reqs + 1] int32, CPU
    num_tokens: int                      # T
    num_tokens_padded: int               # T_pad
    sampled_fill: tuple[torch.Tensor, torch.cuda.Event] | None   # async PP, or pp_size == 1: pinned int32 [num_reqs] ids for
                                                                 #   the EngramReqStep entries with token_ids None (valid after event)
```

`common/topology.py` (L-CORE) implements:

```python
def layer_topology(hf_config, layer_id: int) -> LayerTopology: ...
    # mode: ratio 0 -> "swa"; layer in KV_SOURCES -> "full"; layer in INDEX_SOURCES -> "reindex"; else "reuse".
    # DSpark layers 40..42 -> ratio 0, "swa", no compressor/indexer/engram.
def make_stage_plan(hf_config, pp_rank: int, pp_size: int, partition: list[int]) -> StagePlan: ...
    # partition = VLLM_PP_LAYER_PARTITION counts over the 40 backbone layers. Raises ValueError naming the
    # offending layer if a stage start is not in LEGAL_STAGE_STARTS, or if a Reuse layer's index source or a
    # kv source other than 20 lies on an earlier stage.
```

Topology of the 40 backbone layers (ref config; every lane's tests enumerate this table):

| Layers | ratio | mode | kv source | index source | extras |
|-|-|-|-|-|-|
| 0 | 0 | swa | — | — | θ 10000, no YaRN |
| 1 | 0 | swa | — | — | **Engram** before the block |
| 2, 8 | 2 | full | self | self | compressor (wkv+wgate), index-K writer |
| 3–7, 9–13, 15–19 | 2 | reuse | 2 / 8 / 14 | 2 / 8 / 14 | — |
| 14 | 2 | full | self | self | **Engram** before the block |
| 20 | 1 | full | self | self | compressor (wkv only), index-K writer, **candidate source** |
| 21–23 | 1 | reuse | 20 | 20 | — |
| 24, 28, 32, 36 | 1 | reindex | 20 | self | own indexer q, uses candidates |
| 25–27, 29–31, 33–35, 37–39 | 1 | reuse | 20 | 24 / 28 / 32 / 36 | 37–39 inputs feed DSpark (P5) |

### 3.2 Model skeleton (L-CORE): `sm70/model.py`, `common/hc.py`

```python
# common/hc.py  -- reference semantics ref:m.py:948-966, k.py:406-474 (all math FP32, stream BF16)
def hc_mixes(stream: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """stream [T,4,5120] bf16; fn [24,20480] f32; scale [3] f32; base [24] f32.
    Returns pre [T,4] f32 (sigmoid+eps), post [T,4] f32 (2*sigmoid), comb [T,4,4] f32 (Sinkhorn, 20 iters).
    rsqrt(mean(x^2) + 1e-20) over the flattened 20480 values, FP32 GEMM (no HMMA: fn is FP32-only)."""
def hc_pre(stream: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """sum_c pre[:,c] * stream[:,c,:] in FP32, rounded to bf16 -> [T,5120] bf16 (ref:m.py:957-960)."""
def hc_post(x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    """x [T,5120] f32 (sublayer output); residual [T,4,5120] bf16 -> [T,4,5120] bf16:
    y[:,c] = post[:,c]*x + sum_j comb[:,j,c]*residual[:,j]   (FIRST comb index contracts, ref:m.py:965)."""
def rmsnorm_to_act(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    """bf16/f32 in, FP32 variance + FP32 eps, * weight, -> fp16 [T,5120]. Zero rows -> zeros."""

class DeepseekV41DecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str, topo: LayerTopology, stage: StagePlan,
                 shared: SharedAttnBuffers, engram_service: "EngramHostService | None",
                 spill: "ExpertSpillPlan | None") -> None:
        # children: self.engram (DeepseekV41Engram | None), self.attn (DeepseekV41Attention),
        #           self.ffn (DeepseekV41MoE); params: attn_norm, ffn_norm [5120] fp16;
        #           hc_attn_fn, hc_ffn_fn [24,20480] f32; hc_attn_base, hc_ffn_base [24] f32; hc_attn_scale, hc_ffn_scale [3] f32
    def forward(self, stream: torch.Tensor, pre_in: torch.Tensor, positions: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        # stream [T,4,5120] bf16, pre_in [T,4] f32, positions [T] int64  ->  (stream' bf16, ffn_pre f32)
        if self.engram is not None:
            stream = self.engram(stream, positions)                    # before the block's mixes (ref:m.py:1262-1263)
        attn_pre, attn_post, attn_comb = hc_mixes(stream, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = rmsnorm_to_act(hc_pre(stream, pre_in), self.attn_norm.weight)          # uses the PREVIOUS pre
        stream = hc_post(self.attn(positions, x), stream, attn_post, attn_comb)
        ffn_pre, ffn_post, ffn_comb = hc_mixes(stream, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = rmsnorm_to_act(hc_pre(stream, attn_pre), self.ffn_norm.weight)         # uses this block's attn pre
        stream = hc_post(self.ffn(x), stream, ffn_post, ffn_comb)
        return stream, ffn_pre

class DeepseekV41Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # builds StagePlan, SharedAttnBuffers (contracts.allocate_shared_attn_buffers), EngramHostService (if
        # stage.engram_layers), ExpertSpillPlan (from launch flag, §5.2), embed_tokens (first stage; also last stage
        # when DSpark is on), layers (make_layers with PPMissingLayer outside the stage), mirror (if
        # stage.mirrored_kv_sources), final norm (last stage)
    def make_empty_intermediate_tensors(self, batch_size: int, dtype: torch.dtype,
                                        device: torch.device) -> IntermediateTensors:
        # keys for the boundary INTO this stage: HIDDEN [B,4,5120] bf16 (ignores `dtype`), PRE_MIX [B,4] f32,
        # + KV20_CKV / KV20_IK / CAND20 when 20 in self.stage.mirrored_kv_sources
    def forward(self, input_ids: torch.Tensor | None, positions: torch.Tensor,
                intermediate_tensors: IntermediateTensors | None,
                inputs_embeds: torch.Tensor | None = None) -> torch.Tensor | IntermediateTensors:
        # first stage: stream = embed(input_ids)[:, None, :].expand(T,4,5120).to(bf16); pre = one-hot [1,0,0,0]
        # later stage: stream, pre = it[HIDDEN], it[PRE_MIX]; if mirror: self.mirror.ingest(positions,
        #   it[KV20_CKV], it[KV20_IK], it[CAND20])  (eager break, before the first layer)
        # loop layers; not last -> IntermediateTensors({HIDDEN, PRE_MIX} + {KV20_CKV: shared.export_ckv[:T],
        #   KV20_IK: shared.export_ik[:T], CAND20: shared.candidate_blocks[:T]} if exporting)
        # last -> rmsnorm_to_act(hc_pre(stream, pre), norm.weight)  [T,5120] fp16   (ref:m.py:1268-1269)

class DeepseekV41ForCausalLM(nn.Module, SupportsPP):
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:   # [T,5120] fp16 -> [T,129280] f32
    def skip_checkpoint_weight(self, name: str) -> bool:                     # §3.7 skip list
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    @property
    def engram_service(self) -> "EngramHostService | None":                 # read by worker/runner hooks (§3.6)
```

### 3.3 Attention family (L-ATTN)

```python
# attention.py
class DeepseekV41Attention(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str, topo: LayerTopology, stage: StagePlan,
                 shared: SharedAttnBuffers, aux_streams: list[torch.cuda.Stream] | None = None) -> None: ...
    def forward(self, positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """positions [T] int64; x [T,5120] fp16 (= attn_norm(hc_pre(...))).
        Returns [T,5120] float32 = wo_b(...) all-reduced over TP (identical on every TP rank)."""

# compressor.py  (only constructed on kv sources)
class DeepseekV41Compressor(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str, compress_ratio: int) -> None: ...
    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x [T,5120] fp16. Returns (latent [N,512] fp16 PRE-RoPE, normed; latent_pos [N] int64 = position of
        each latent's FIRST token). ratio 1: N = T. ratio 2: N = completed pairs this step; the pending odd token's
        FP32 (kv, score) lives in the paged state cache `{prefix}.state_cache`."""

# indexer.py  (only constructed on index sources)
class DeepseekV41Indexer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str, topo: LayerTopology,
                 shared: SharedAttnBuffers) -> None: ...
    def write_keys(self, latent: torch.Tensor, latent_pos: torch.Tensor) -> None:
        """kv sources only: k = fp4_e8m0_qdq(rope(k_norm(wk(latent)))) -> paged index-K cache (and, on source 20 of an
        exporting stage, shared.export_ik[:T]). Must run before the compressed-KV write (ref:m.py:516-548)."""
    def forward(self, x: torch.Tensor, qr: torch.Tensor, positions: torch.Tensor) -> None:
        """x [T,5120] fp16 (weights_proj input), qr [T,1280] fp16 (= q_norm(wq_a x)). Scores against the kv
        source's index-K cache; layer 20 writes shared.candidate_blocks[:T]; 24/28/32/36 mask with it; writes the
        top-512 logical compressed positions to shared.topk_indices[:T]."""

# kv_mirror.py
class DeepseekV41KVSourceMirror(nn.Module):
    def __init__(self, vllm_config: VllmConfig, source_layer: int, shared: SharedAttnBuffers) -> None:
        """Registers in the static forward context, under the SOURCE's names, two AttentionLayerBase children:
        f"model.layers.{source_layer}.attn" (compressed-KV spec) and
        f"model.layers.{source_layer}.attn.indexer.k_cache" (index-K spec), whose get_kv_cache_spec() returns
        specs equal (dataclass ==) to the source's on the earlier stage."""
    def ingest(self, positions: torch.Tensor, ckv: torch.Tensor, ik: torch.Tensor, cand: torch.Tensor) -> None:
        """ckv [T,512] fp16, ik [T,128] fp16 (exact records as written by the source), cand [T,2048] int32.
        Scatter ckv/ik to the slots of attn_metadata[source prefix].slot_mapping; copy cand to
        shared.candidate_blocks[:T]. Eager-break op."""
```

Inside `DeepseekV41Attention.forward` (per TP rank; TP4 ⇒ 16 local heads, 2 local o-groups):

| Step | Tensor | Shape / dtype | Reference |
|-|-|-|-|
| fused `wq_a`/`wkv` (replicated) | `qr_kv` | [T, 1280+512] fp16 | ref:`m.py:770,705` |
| `q_norm`, `kv_norm` (FP32 var, eps 1e-20) | `qr`, `kv` | [T,1280], [T,512] fp16 | |
| `wq_b` (column) → RoPE on dims 448:512 (interleaved pairs, θ/YaRN per topo) — **no q-norm** | `q` | [T,16,512] fp16 | ref:`m.py:771-772` |
| window KV: RoPE(kv) → `fp8_block32_qdq` → SWA paged insert | record | [T,512] fp16 | ref:`m.py:700-720` |
| kv source: compressor → `indexer.write_keys(latent)` → RoPE(latent at first-token pos) → `fp4_e4m3_qdq(16)` → compressed paged insert (+ `export_ckv` on 20) | `latent` | [N,512] fp16 | ref:`m.py:739-763` |
| index source: `indexer.forward(x, qr, positions)` → `shared.topk_indices[:T]` | top-k | [T,512] int32 | ref:`m.py:725-737` |
| sparse attention over ≤128 window + ≤512 compressed rows, sink, −1e30 init | `o` | [T,16,512] fp16 | ref:`k.py:310-403` |
| inverse RoPE on dims 448:512; `wo_a` grouped (2 local groups, [1024×4096] each); `wo_b` row-parallel FP32 out + TP all-reduce | out | [T,5120] f32 | ref:`m.py:781-788` |

KV-cache spec declarations (L-ATTN; page sizes via `kv_cache_interface.py` branch for `model_version="deepseek_v41"`):

```python
# every layer: f"{prefix}.swa_cache"
SlidingWindowMLASpec(block_size=swa_block, num_kv_heads=1, head_size=512, dtype=torch.float16,
                     sliding_window=128, compress_ratio=1, model_version="deepseek_v41")         # 1024 B/token
# kv sources (and the mirror) only: f"{prefix}"  (consumers return None)
MLAAttentionSpec(block_size=B, num_kv_heads=1, head_size=512, dtype=torch.float16, compress_ratio=r,
                 cache_dtype_str="v41_fp16_qat_ckv", model_version="deepseek_v41")               # (B/r) x 1024 B
# kv sources (and the mirror) only: f"{prefix}.indexer.k_cache"
MLAAttentionSpec(block_size=B, num_kv_heads=1, head_size=128, dtype=torch.float16, compress_ratio=r,
                 cache_dtype_str="v41_fp16_qat_ik", model_version="deepseek_v41")                # (B/r) x 256 B
# ratio-2 sources only: f"{prefix}.compressor.state_cache"  (pending odd token: FP32 kv[512] + score[512])
SlidingWindowMLASpec(..., head_size=1024, dtype=torch.float32, sliding_window=2, compress_ratio=1,
                     model_version="deepseek_v41")   # block size chosen by L-ATTN for page-size unification, as V4 does
```

Source discovery: a consumer resolves its sources by name in `vllm_config.compilation_config.static_forward_context`
(`model.layers.{kv_source}.attn`, `model.layers.{kv_source}.attn.indexer.k_cache`) exactly as
up:`attention.py:573-601`; on a mirrored stage the mirror owns those names, so consumer code is identical on every
stage. Logical compressed positions → physical slots use the **source's** block table/metadata
(`attn_metadata[source_prefix]`), as 1C:`sm70/sparse.py:225-236` does for C4.

### 3.4 MoE (L-MOE): `sm70/moe.py`

```python
@dataclass(frozen=True)
class ExpertSpillPlan:
    spilled_expert_ids: tuple[int, ...]   # global expert ids kept in pinned host memory; identical on all TP ranks
    mode: str = "uva"                      # zero-copy UVA view of cudaHostAlloc memory (1C:get_accelerator_view_from_cpu_tensor)

def make_spill_plan(num_spilled_per_layer: int, n_routed_experts: int = 384,
                    ranking: "np.ndarray | None" = None) -> ExpertSpillPlan: ...
    # ranking None -> highest ids spilled; else the least-routed ids (L-REF routing histogram, §5.2)

class DeepseekV41MoE(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str, layer_id: int, *, n_routed_experts: int = 384,
                 top_k: int = 6, spill: ExpertSpillPlan | None = None) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [T,5120] fp16 (= ffn_norm(hc_pre(stream, attn_pre))). Returns [T,5120] float32 =
        sum_k w_k * expert_k(x) + shared(x), TP-all-reduced. Router: logits FP32 (FP16 weight, FP32 accumulate),
        sqrt(softplus), top-6 of (score + bias), weights = score / (sum + 1e-20) * 1.5."""
    @staticmethod
    def expert_params_mapping(n_routed_experts: int) -> list[tuple[str, str, int, str]]: ...  # used by L-CORE load_weights

def make_v41_moe_method(moe: FusedMoEConfig) -> FusedMoEMethodBase: ...   # factory used by DeepseekV41FP8Config
```

Per-rank expert tensors (TP4 splits the 2304 intermediate rows into 576): w13 = E2M1 [384, 1152, 2560 B] + E8M0
[384, 1152, 160]; w2 = E2M1 [384, 5120, 288 B] + E8M0 [384, 5120, 18]; 4,700,160 B per expert shard; TurboMind
repacks in place (layout is L-MOE's; byte count must stay within +1 %).

### 3.5 PP payload and the kv-source mirror (L-CORE packs/unpacks, L-ATTN produces/consumes)

| Boundary | Keys | Bytes / token |
|-|-|-|
| every stage boundary | `hidden_states` [T,4,5120] bf16 + `pre_mix` [T,4] f32 | 40,960 + 16 = **40,976** |
| into a stage mirroring source 20 (PP3 2→3 only) | + `kv20_ckv` [T,512] f16 + `kv20_ik` [T,128] f16 + `cand20` [T,2048] i32 | + 1,024 + 256 + 8,192 = **50,448** total |

Why it is exact and slot-consistent:
1. Layer 20 has `compress_ratio = 1`, so it emits exactly one compressed latent and one index key per scheduled
   token, in token order: rows `0:T` of `export_ckv` / `export_ik` are the records the source wrote for this step
   (the same FP16 values, copied in the same kernel that writes the cache). No recomputation on stage 3.
2. The mirror registers its two cache layers under the **source's layer names** with **equal specs**.
   `get_kv_cache_configs` merges specs by layer name across workers and asserts equality
   (1C:`vllm/v1/core/kv_cache_utils.py:3010-3031`), so source and mirror are the *same* layer in the *same* KV
   group of the single scheduler-side KV manager: both stages receive identical block ids, identical slot mappings,
   and prefix-cache hits/evictions apply to both copies at once. The mirror writes `attn_metadata[name].slot_mapping`
   slots, exactly the slots the source wrote.
3. Candidates: layer 20 writes `shared.candidate_blocks[:T]`; the packer sends that slice; the mirror copies it into
   stage 3's `shared.candidate_blocks[:T]`, read by Reindex layers 28/32/36. Layer-20 top-k is **not** shipped:
   stage 3 starts at Reindex layer 28, whose own top-k serves 29–31.
4. Debug knob `VLLM_DS41_ATTN_MIRROR_CHECK=1`: the exporter adds `kv20_crc` [T] int32 (sum of the int16 bit
   patterns per row) and the mirror re-reads the slots after ingest and asserts equality.

Static-transfer schema (L-CORE, `gpu_worker.py`): the model declares `pp_static_schema(num_tokens) ->
dict[str, (shape, dtype)]` from `make_empty_intermediate_tensors`; the worker's metadata-free path admits any schema
the model declares (not only `[1,4,4096]`), for T ∈ {1..8} decode steps.

### 3.6 Engram: host service, module, runner/worker hooks (L-ENGRAM + hooks by L-CORE)

**Row file (format v1, `tools/ds41_engram_repack.py`).** One file per Engram layer on the root NVMe,
`$DS41_ENGRAM_DIR` (default `[private path]`, same LVM as `/`): `engram_L01_v1.rows`,
`engram_L14_v1.rows`.

```
[0, 4096)          header: magic "DS41ENGR", version=1, layer_id, num_rows, row_bytes=264, rows_per_page=15,
                   page_bytes=4096, value_dtype="e4m3fn", scale_dtype="ue8m0", scale_block=32,
                   24 x (prime u64, offset u64) in order-major order (s = (n-2)*8 + head),
                   token_map_sha256 (compressed vocab, must be 99,092 ids), source shard name + sha256 of the
                   weight/scale byte ranges, checkpoint rev "2cba9e42...", header_sha256
[4096 + 4096*p, +4096)  page p: rows 15p .. 15p+14 at byte 264*i; bytes 3960..4095 zero
row:               256 x E4M3 (channel 0..255) then 8 x UE8M0 (scale j covers channels 32j..32j+31)
row r lives at:    4096 + (r // 15) * 4096 + (r % 15) * 264        -> exactly one aligned 4 KiB page per row
sizes:             L1  4096 + 25,600,412 * 4096 = 104,857,291,648 B
                   L14 4096 + 25,601,113 * 4096 = 104,860,162,944 B      total 209.72 GB (195.3 GiB)
```

The tool refuses to start unless `/` has ≥ 209.72 GB + 30 GB free (D2 guard), writes pages sequentially, then
verifies 2 M random rows byte-for-byte against the safetensors tensors and writes the header last (a file without a
valid header is never opened).

**Hashing (CPU, numpy, bit-exact with ref:`e.py:129-184`).** Compressed map from the tokenizer (99,092 ids, asserted);
multipliers `np.random.default_rng(10007*layer).integers(0, bound, 4)*2+1` with `bound = (int64_max // 99092)//2`;
per position `t`: `tok[s] = cmap[ids[t-s]]` (pad id `cmap[2]` for `t-s < 0` or after a dead token);
`h_n = XOR_{s<n}(tok[s]*mult[s])` for n = 2,3,4 (int64; no overflow by construction); row id for sub-table
`s=(n-2)*8+k` = `h_n % prime[s] + offset[s]`. Rank r gathers only sub-tables `S_r = {r, r+4, r+8, r+12, r+16, r+20}`
(two per n-gram order). The service keeps a per-request position-indexed history of compressed ids (rollback-safe
for spec decode, rebuilt from `prompt_token_ids` on resume). Reads use buffered `pread` with `POSIX_FADV_RANDOM` from
a C++ thread pool rather than an mmap (1Cat's PLE disk tier uses mmap + `MADV_RANDOM`): same page cache, but explicit
queue depth, no page-fault serialization in the main thread and I/O errors as return codes instead of SIGBUS.

```python
# common/engram_host.py
class EngramHostService:
    def __init__(self, hf_config, layers: tuple[int, ...], tp_rank: int, tp_size: int, row_dir: str,
                 tokenizer_path: str, max_num_batched_tokens: int, device: torch.device,
                 io_threads: int = 4) -> None: ...
    def begin_step(self, plan: EngramStepPlan) -> None:
        """Worker thread, called BEFORE the PP receive. Hash every request whose token ids are known; start
        high-priority gathers of this step's rows (buffered pread, POSIX_FADV_RANDOM, dedup) into a pinned host
        slot keyed by (req_id, pos). New/resumed requests: hash the whole prompt and queue low-priority warm-up reads
        of rows beyond this step (admission prefetch). Never blocks on I/O."""
    def bind_batch(self, layout: EngramBatchLayout) -> None:
        """Runner thread, after _prepare_inputs and before the forward. Resolves async-PP unknown tokens from
        layout.sampled_fill (waits its event in the I/O thread), then assembles rows in runner order into pinned
        slot k [T_pad, n_layers, 6, 264] u8 (zero rows for padding) and enqueues the H2D copy into the static device
        buffer on the service's copy stream; records ready_event[k]. Returns without waiting for I/O."""
    def wait_rows(self, layer_id: int) -> torch.Tensor:
        """Inside the Engram eager-break op: block the CPU until slot k's H2D is enqueued, make the current stream
        wait on ready_event[k], return the static device view [T_pad, 6, 264] u8 for layer_id."""
    def end_step(self, step_id: int) -> None: ...
    def stats(self) -> dict[str, float]: ...   # rows, misses, p50/p99 gather latency, warm-up backlog
    def shutdown(self) -> None: ...

# common/engram.py
class DeepseekV41Engram(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str, layer_id: int,
                 service: EngramHostService) -> None: ...
        # params: wkv_r [25600, 1536] fp16 = dequant(cols of S_r) * 2^10;  qk [4, 5120] f32 = q_weight * k_weight
    def forward(self, stream: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """stream [T_pad,4,5120] bf16 -> bf16. rows = service.wait_rows(layer) (eager break); decode rows to FP16
        [T,1536] (exact); partial = (rows @ wkv_r^T) * 2^-10 in FP32 (FP16 operands, FP32 accumulate, alpha
        applied in FP32); all-reduce FP32 over TP; key = partial[:, :20480].view(T,4,5120), value = partial[:,
        20480:]; per copy c: dot = sum(h_c * qk_c * key_c) * rsqrt(mean(h_c^2)+1e-20) * rsqrt(mean(key_c^2)+1e-20)
        / sqrt(5120); gate = sigmoid(sign(dot) * sqrt(max(|dot|, 1e-6))); h_c += gate * value (FP32) -> bf16
        (ref:m.py:350-365). Prefill is processed in 1024-token sub-chunks (bounds the FP32 partial to 100 MiB)."""
```

**Where token ids come from inside vLLM v1** (both hooks owned by L-CORE):
1. `Worker.execute_model` (1C:`vllm/v1/worker/gpu_worker.py:1172`), first statement for ranks whose model has an
   `engram_service`: build `EngramStepPlan` from `scheduler_output` alone — new requests carry `prompt_token_ids`
   and `num_computed_tokens`; cached requests carry `num_computed_tokens` and, under synchronous scheduling with
   PP > 1, `new_token_ids` (1C:`gpu_model_runner.py:2618-2640` consumes the same field); `num_scheduled_tokens[req]`
   and `scheduled_spec_decode_tokens` give the step's tokens. Call `begin_step(plan)` **before** the PP receive at
   `:1221-1238` so that stage 2's layer-14 gather overlaps stage 1's compute (layer 14 is the first layer of
   stage 2 in PP3).
2. `GPUModelRunner.execute_model`, after `_prepare_inputs` (batch order, `query_start_loc`, padding known) and
   before the model call: `bind_batch(EngramBatchLayout(...))`. P5 async PP: in
   `_pp_receive_prev_sampled_token_ids_to_input_batch` (1C:`:9809-9840`) add a non-blocking copy of `recv`
   [num_reqs,1] int32 into a pinned buffer + event, passed as `sampled_fill`.
3. Engram-owning ranks: PP2 stage 1 (layers 1 and 14); PP3 stage 1 (layer 1) and stage 2 (layer 14). The last stage
   never owns Engram under PP. Without PP (TP8 benchmark) the owner is the sampling rank: `scheduler_output` carries
   no `new_token_ids`, so `begin_step` leaves decode `token_ids` None and the runner passes the ids it just sampled
   (already on CPU under synchronous scheduling) as `sampled_fill` with a completed event.

### 3.7 Checkpoint name → module parameter (load ownership)

Rank r of TP4. "FP16" for FP8 sources = dequantised at load through the fallback (exact); BF16 sources become FP16
(exact for |w| ≥ 2^-14) except where FP32 is listed.

| Checkpoint tensor (HF name) | Lane | Parameter (vLLM) | Rank-r slice | Dtype after load |
|-|-|-|-|-|
| `embed.weight` [129280,5120] | CORE | `model.embed_tokens.weight` (first stage; last stage too with DSpark) | rows [32320r, 32320(r+1)) | FP16 |
| `head.weight` | CORE | `lm_head.weight` (last stage) | rows [32320r, …) | FP16, FP32 logits |
| `norm.weight`, `layers.N.{attn,ffn}_norm.weight` | CORE | `model.norm.weight`, `model.layers.N.{attn,ffn}_norm.weight` | replicated | FP32 |
| `layers.N.hc_{attn,ffn}_{fn,base,scale}` | CORE | `model.layers.N.hc_*` | replicated | FP32 |
| `layers.N.attn.wq_a.{weight,scale}` / `attn.wkv.{weight,scale}` | ATTN | `…attn.fused_wqa_wkv.weight` shard 0 (rows 0:1280) / shard 1 (rows 1280:1792) | replicated | FP16 |
| `layers.N.attn.{q_norm,kv_norm}.weight` | ATTN | `…attn.{q_norm,kv_norm}.weight` | replicated | FP32 |
| `layers.N.attn.wq_b.*` [32768,1280] | ATTN | `…attn.wq_b.weight` | rows [8192r, 8192(r+1)) = heads 16r…16r+15 | FP16 |
| `layers.N.attn.wo_a.*` [8192,4096] | ATTN | `…attn.wo_a.weight` (`dequantized_bmm`, 2 groups) | rows [2048r, …) = groups 2r, 2r+1 | FP16 |
| `layers.N.attn.wo_b.*` [5120,8192] | ATTN | `…attn.wo_b.weight` | cols [2048r, …) | FP16 |
| `layers.N.attn.attn_sink` [64] | ATTN | `…attn.attn_sink` | [16r, 16r+16) | FP32 |
| `layers.{2,8,14}.attn.compressor.{wkv,wgate}.weight` [512,5120] | ATTN | `…compressor.fused_wkv_wgate.weight` [1024,5120] | replicated | FP16 (FP32 accumulate) |
| `layers.20.attn.compressor.wkv.weight` | ATTN | `…compressor.wkv.weight` [512,5120] | replicated | FP16 |
| `layers.{2,8,14,20}.attn.compressor.norm.weight` | ATTN | `…compressor.norm.weight` | replicated | FP32 |
| `layers.{2,8,14,20,24,28,32,36}.attn.indexer.wq_b.*` [4096,1280] | ATTN | `…indexer.wq_b.weight` | replicated (all 32 heads) | FP16 |
| `layers.{…}.attn.indexer.weights_proj.weight` [32,5120] | ATTN | `…indexer.weights_proj.weight` | replicated | FP16 |
| `layers.{2,8,14,20}.attn.indexer.wk.weight` [128,512], `k_norm.weight` | ATTN | `…indexer.wk.weight`, `…indexer.k_norm.weight` | replicated | FP16 / FP32 |
| `layers.N.ffn.gate.weight` [384,5120] | MOE | `…ffn.gate.weight` | replicated | FP16 (FP32 accumulate) |
| `layers.N.ffn.gate.bias` [384] | MOE | `…ffn.gate.e_score_correction_bias` | replicated | FP32 |
| `layers.N.ffn.gate.bias_vl` | MOE | skipped (text-only) | — | — |
| `layers.N.ffn.experts.E.w{1,3}.{weight,scale}` | MOE | `…ffn.experts.w13_weight[E]`, `w13_weight_scale[E]` | rows [576r, 576(r+1)) of w1 and of w3 | u8 E2M1 pairs / u8 E8M0 |
| `layers.N.ffn.experts.E.w2.{weight,scale}` | MOE | `…ffn.experts.w2_weight[E]`, `w2_weight_scale[E]` | packed cols [288r, …), scale cols [18r, …) | u8 / u8 |
| `layers.N.ffn.shared_experts.w{1,3}.*` | MOE | `…ffn.shared_experts.gate_up_proj.weight` | rows [576r, …) of each | FP16 |
| `layers.N.ffn.shared_experts.w2.*` | MOE | `…ffn.shared_experts.down_proj.weight` | cols [576r, …) | FP16 |
| `layers.{1,14}.engram.wkv.{weight,scale}` [25600,6144] | ENGRAM | `…engram.wkv_r` [25600,1536] | cols of sub-tables S_r | FP16 × 2^10 |
| `layers.{1,14}.engram.{q_weight,k_weight}` [4,5120] | ENGRAM | `…engram.qk` = q·k | replicated | FP32 |
| `layers.{1,14}.engram.embed.{weight,scale}` | ENGRAM (offline) | **not loaded** (`skip_checkpoint_weight`); repack tool input | — | row file |
| `mtp.{0,1,2}.*` | CORE (P5) | `dspark.*` | as backbone, 128 experts | as backbone |
| `vision.*`, `aligner.*`, `image_{start,end,newline}` | — | skipped | — | — |

### 3.8 Golden tensors (L-REF ↔ every lane)

Layout: `<golden dir>/manifest.json` + `L{nn}.safetensors` per captured layer +
`final.safetensors`. `manifest.json`: checkpoint rev, mode (`v100-semantic` = no activation quant, all QAT kept, FP32
elsewhere — the **primary comparator**; `ref-exact` = also FP8 per-32 activation quant on every quantised GEMM input),
token ids, positions, layer subset, chunking. Tensor names (FP32 unless noted; per-head tensors only for layers
listed in the manifest):

| Name | Shape | Name | Shape |
|-|-|-|-|
| `stream_in`, `stream_out` | [T,4,5120] | `pre_in`, `pre_out` | [T,4] |
| `hc.attn_pre/post`, `hc.ffn_pre/post` | [T,4] | `hc.attn_comb`, `hc.ffn_comb` | [T,4,4] |
| `engram.hash` (i64) | [T,24] | `engram.rows` (dequantised) | [T,24,256] |
| `engram.key`, `engram.value`, `engram.gate` | [T,4,5120], [T,5120], [T,4] | `attn.x` | [T,5120] |
| `attn.qr`, `attn.q` (post-RoPE) | [T,1280], [T,64,512] | `attn.kv_win` (post-QAT) | [T,512] |
| `attn.latent` (pre-RoPE), `attn.ckv` (post-RoPE+QAT) | [N,512] | `idx.q` (post-QAT), `idx.k` | [T,32,128], [N,128] |
| `idx.score` (rows listed) | [t,N] | `idx.topk` (i32, sorted, −1 pad) | [T,512] |
| `cand.blocks` (layer 20, i32) | [T,2048] | `attn.o` | [T,64,512] |
| `attn.out` | [T,5120] | `moe.x`, `moe.logits` | [T,5120], [T,384] |
| `moe.topk_ids` (i32), `moe.weights` | [T,6] | `moe.out` | [T,5120] |
| `final.h`, `logits` (positions listed) | [T,5120], [t,129280] | | |

The port dumps the same names under `VLLM_DS41_DUMP_DIR` for `VLLM_DS41_DUMP_LAYERS` (rank 0 gathers TP-sharded
tensors; L-CORE owns the hook, each lane names its tensors as above). `compare.py` applies §4.5.

---

## 4. Numerics contract (FP16-on-Volta)

### 4.1 Precision map — MUST rules are acceptance criteria for code review

Evidence that FP16 range is a real hazard for this family: on DeepSeek-V4-Flash the BOS/attention-sink row's
residual reaches ~2k at layer 27, 20k at 33, 63.5k at 40 and passes 65,504 from layer 41, while every other token
stays below ~200; FP16 squares overflowed beyond |x| = 255.9 and inf residuals turned into NaN attention for every
token with BOS in its window (1Cat PR #658 body; `hc_head` follow-up #715 still open).
V4.1 has the same mHC/sink design (40 layers, hidden 5120), so the contract assumes the same profile until L-REF's
audit (§4.4) measures it.

| Quantity | Storage | Compute | Rule | Basis |
|-|-|-|-|-|
| HC residual stream | **BF16** (software on SM70) | FP32 | MUST (never FP16) | ref keeps BF16 (ref:`m.py:960,966`); #658 range data |
| `hc_mixes` (prenorm GEMM ×`hc_fn` [24,20480]) | FP32 weights | FP32 GEMM, squares in FP32 | MUST | `hc_fn` is FP32-only; #658 item 1 (FP16 squares) |
| Sinkhorn split (20 iters) | — | FP32 | MUST | ref:`k.py:406-474` |
| `hc_pre` sum, `hc_post` | BF16 out | FP32 | MUST | #658 item 3 (pre-mix sum ~157k) |
| every RMSNorm (eps **1e-20**) incl. q/kv/compressor/k_norm/final, HC rsqrt, Engram rstd | weight FP16/FP32 | FP32 variance + FP32 eps; zero rows → 0 | MUST | 1e-20 underflows to 0 in FP16 → rsqrt(0) = inf → NaN on padding rows |
| sublayer inputs (after norm) | FP16 | — | MUST | V100 has no BF16 MMA; post-norm values are bounded |
| dense GEMMs (FP16 weights) | FP16 operands | FP32 accumulate; `torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False` set at model init | MUST | split-K FP16 reductions otherwise allowed by torch |
| wo_b output (all-reduced), MoE combine (routed + shared) | **FP32** | FP32 | MUST at P2; MAY drop to FP16 at P5 only if §4.4 max < 8,192 | ref all-reduces wo_b in FP32 (ref:`m.py:271-278`) and sums experts in FP32 (ref:`m.py:893-903`) |
| per-expert w13 / activation / w2 outputs | FP16 | FP32 accumulate, clamp + SiLU in FP32 | MAY (bounded by `swiglu_limit` 10) — FP32 epilogue required if §4.4 per-expert max ≥ 16,384 | ref rounds per-expert outputs to BF16 |
| attention: q·k, softmax, sink | scores FP32 | FP32; probabilities FP16 into PV | MUST FP32 scores; FP16 P allowed (ref uses BF16 P) | ref:`k.py:311-389` |
| router logits, softplus/sqrt, bias add, top-6, normalisation | FP32 | FP32 (FP16 gate weight, FP32 accumulate) | MUST | discrete decision |
| indexer q·k, ReLU·w, head sum, top-512; candidate block max/top-2048 | FP32 | FP32 | MUST | discrete decisions |
| compressor ratio 2 (wkv, wgate, softmax pooling, state) | FP32 state | FP32 accumulate (FP16 operands OK: inputs FP16, weights BF16-exact) | MUST | ref promotes to FP32 (ref:`m.py:445-448,464-475`) |
| Engram rows → values | FP16 | exact decode | MUST exact | row exponents −10…−6 |
| Engram `wkv` | FP16 weights × **2^10 bias** | FP32 accumulate, α = 2^-10 applied in FP32, FP32 output | MUST | `wkv` exponents −18…−8; 22.9 % of blocks below 2^-14 (MODEL_ARCHAEOLOGY §11) |
| Engram gate | — | FP32 | MUST | ref:`m.py:357-365` |
| logits | FP16 head weight | FP32 accumulate and **FP32 output** | MUST | ref head is FP32 (ref:`m.py:1006-1012`) |
| embedding | FP16 (BF16-exact) | — | — | — |

Invariant for TP-replicated decisions: router top-k, indexer top-k and candidate blocks are computed redundantly on
the four TP ranks from all-reduced inputs, so every rank must see **bitwise-identical** inputs (NCCL and 1Cat's
fixed-order custom all-reduce both broadcast one result). Debug knob `VLLM_DS41_MOE_TOPK_CHECK=1` all-gathers a
per-step checksum of the router ids and asserts equality. Golden comparisons run with a pinned TurboMind tactic
cache (per-process autotune changes numerics across restarts: 1Cat #299/#349).

### 4.2 Quantise–dequantise steps: which are model semantics (emulate bit-faithfully) and which are skipped

| Step in the reference | Where | Treatment | Exact algorithm (all FP32 IEEE, RNE casts) |
|-|-|-|-|
| window KV FP8 QAT | ref:`m.py:707` | **bit-faithful** `fp8_block32_qdq` | per 32-block: `amax = max(max|x|, 1e-4)`; `e = ceil_log2(amax * fp32(1/448))` with `ceil_log2(v) = exp_bits(v) - 127 + (mant_bits(v) != 0)`; `s = 2^e`; `y = e4m3_rne(clamp(x/s, ±448)) * s` (ref:`k.py:22-37,72-87`) |
| compressed KV FP4 QAT | ref:`m.py:760` | **bit-faithful** `fp4_e4m3_qdq` (block 16) | `amax = max(max|x|, 6·2^-9)`; `s = e4m3_rne(min(amax / 6.0, 448))` (true IEEE division; saturating); `y = e2m1_rne(clamp(x / s, ±6)) * s` (ref:`k.py:157-174`) |
| indexer q and index-K FP4 QAT | ref:`m.py:546,552` | **bit-faithful** `fp4_e8m0_qdq` (block 32) | `amax = max(max|x|, 6·2^-126)`; `e = ceil_log2(amax * fp32(1/6))`; `s = 2^e`; `y = e2m1_rne(clamp(x/s, ±6)) * s` |
| activation FP8 quant inside `linear()` | ref:`m.py:187,197` | **skipped** (FP16 activations); emulated only in the L-REF `ref-exact` golden | per 32: as `fp8_block32_qdq` without the in-place flag |
| `wo_a` dequant to BF16 in `convert.py` | ref:`convert.py:157-173` | n/a (we dequantise all dense to FP16) | — |

`e2m1_rne` grid {0, 0.5, 1, 1.5, 2, 3, 4, 6}, ties to the even mantissa: 0.25→0, 0.75→1, 1.25→1, 1.75→2, 2.5→2,
3.5→4, 5→4 (sign kept). `e4m3_rne` = round-to-nearest-even with saturation to ±448 (what the TileLang `T.Cast(FP8)` → `__nv_fp8_e4m3` / `cvt.rn.satfinite` does); `torch.float8_e4m3fn` matches it only after an explicit clamp to ±448 (it returns NaN on overflow). FP16 holds every result exactly:
window KV for block exponents e ≥ −15 (block amax ≥ 0.0068; smaller blocks may flush sub-2^-24 values — counted by a
debug counter), compressed KV always (|y| ∈ {0} ∪ [2^-10, 2688], ≤ 6 significant bits), index q/k for e ≥ −23.
Triton implementations use `tl.div_rn` (IEEE) — never approximate division — because `x/s` with a non-power-of-two
E4M3 scale sits on rounding boundaries. The reference TileLang kernels cannot run on SM70/SM86 (FP8/FP4 `cvt`
instructions), so L-REF validates its torch twins exhaustively over all 65,536 FP16 scalars for the element casts
and over 10^7 random blocks (incl. constructed tie cases) for the scale logic, and the port's Triton twins are tested
bit-for-bit against the L-REF twins.

### 4.3 Deviations from the reference that are allowed (and measured, not argued)

FP16 activations instead of BF16 / FP8-quantised GEMM inputs; FP32 sublayer outputs instead of BF16-rounded ones;
HMMA accumulation order; routing weight applied after `w2` (linear, ref applies before); indexer replicated instead of
head-sharded with an all-reduce; Engram `wkv` output kept FP32 (ref rounds to BF16); index scores FP32 (ref
materialises them in BF16, ref:`m.py:555-557`, which creates exact ties the port will break differently). Each is
covered by the §4.5 gates; none may be "fixed" by adding clamps or saturation (a saturating store is a model change).

### 4.4 Range audit (L-REF deliverable, gates two MAY rules)

On the FP32 golden over the smoke corpus (§7.6, every prompt starting with BOS, plus 3 long documents): per layer the
max |·| and its position for: stream, `hc_pre` sum, attention output (pre/post all-reduce), per-expert w2 output,
shared-expert output, MoE combine, q·k logits, Engram `wkv` output, indexer scores. Output: `RANGE_AUDIT.md` +
json. Rules: sublayer outputs may go FP16 only if every max < 8,192; per-expert outputs may stay FP16 only if every
max < 16,384. The stream stays BF16 regardless.

### 4.5 Tolerances (unit tests vs golden; port fed the golden's inputs for per-op tests)

| Test | Metric | Gate (vs `v100-semantic` golden) |
|-|-|-|
| QAT twins (`fp8_block32_qdq`, `fp4_e4m3_qdq`, `fp4_e8m0_qdq`) | bitwise | 0 mismatches |
| Engram hash ids, row bytes, decoded rows | exact | 0 mismatches |
| Engram module output stream | rel-RMS / max-abs | ≤ 1e-3 / ≤ 1e-2 × max |
| HC mixes, Sinkhorn, pre/post | rel error | ≤ 1e-6 (FP32 both sides) |
| dense linear (FP16 HMMA, FP32 acc) | rel-RMS | ≤ 2e-3 |
| router | logits rel-RMS; top-6 sets | ≤ 1e-5; identical except pairs with relative score gap < 1e-6 |
| MoE layer output | rel-RMS | ≤ 3e-3 |
| indexer | score rel-RMS; top-512 set overlap per query | ≤ 1e-3; ≥ 99.5 % (mean), ≥ 98 % (min) |
| candidate blocks (layer 20) | set equality per query given identical scores | 100 % |
| compressor latent (pre-RoPE) | rel-RMS | ≤ 1e-3 |
| sparse attention output | rel-RMS | ≤ 2e-3 |
| full block (stream out) | rel-RMS | ≤ 5e-3 per layer |
| end-to-end, teacher-forced ≤ 2K tokens | final-logit rel-RMS; cosine; top-1 agreement; mean \|ΔNLL\| | ≤ 0.02 (GLM TP4 failed at 0.041, passed at 0.0047: 1CAT §8.2); ≥ 0.999 median / ≥ 0.99 p1; ≥ 98 %; ≤ 0.02 nats/token |
| end-to-end vs `ref-exact` golden | top-1 agreement; mean \|ΔNLL\| | ≥ 96 %; ≤ 0.04 nats/token |
| PP3 vs PP2 on the same prompts | top-1 agreement; logit rel-RMS | ≥ 99.5 %; ≤ 5e-3 (same kernels; only the mirror and the spill's FP32 summation order differ) |
| any tensor | NaN/Inf | none (padding rows included) |

---

## 5. Pipeline design

### 5.1 Legal stage cuts

A stage may start at layer b only if no layer in it reuses a top-k published on an earlier stage and every kv source
it reads from an earlier stage is mirrorable (v1 mirrors only source 20, whose ratio-1 records map 1:1 onto tokens).
That gives `LEGAL_STAGE_STARTS = {1, 2, 8, 14, 20, 24, 28, 32, 36}` (§3.1). With ≤ 17 layers per 4-GPU stage
(6.88 GiB/layer native): **PP2 → cut 20 (20/20)**; **PP3 → cuts (14, 28) = 14/14/12 is the only legal split**
((14,24) puts 16 layers = 28.3 GiB of weights on stage 3; (8,24) and (14,32) are worse). The coordinator's 13/14/13
(D3) starts stages at Reuse layers 13 and 27, so it would need source 8 mirrored and the top-k of layers 8 and 24
shipped; it is superseded by A7.

### 5.2 Phase A — 8 GPUs, PP2×TP4, cut at 20, deliberate expert spill

| | Stage 1 (layers 0–19, board A, idx 0–3) | Stage 2 (layers 20–39, board C, idx 8–11) |
|-|-|-|
| weights/GPU, FP16 dense, all experts resident | 36.05 GiB (dense 2.12, experts 33.62, embed 0.31) | 35.89 GiB (dense 1.97, experts 33.62, head 0.31) |
| runtime/GPU (4K chunk, 32K ctx, §6.2 lines) | 3.39 GiB (owns both Engram layers) | 3.07 GiB |
| weight budget at 29.5 GiB | 26.11 GiB → spill ≥ 9.94 GiB = 113.5 experts/layer | 26.43 GiB → spill ≥ 9.46 GiB = 108.1 experts/layer |
| **chosen spill** | **116 experts/layer** (30.2 %), 10.16 GiB/GPU, 40.6 GiB/stage | same, 40.6 GiB/stage |
| crossing tensors | — | in: `hidden_states` + `pre_mix` = 40,976 B/token |

Total pinned host memory 81.2 GiB (cudaHostAlloc is not limited by the 8 MiB memlock, HARDWARE §A.6).
Which experts: the same 116 global ids in every layer of both stages; bring-up spills ids 268–383; once L-REF has a
routing histogram over the smoke corpus, the 116 least-routed ids per layer (gain expected small: the text bias is
near-uniform, std 0.063, MODEL_ARCHAEOLOGY §6). Spill size is a launch flag computed from the profiled free memory
(`--ds41-spill-experts-per-layer`, default from this table), never vLLM's universal `cpu_offload_gb`.

Mechanism (L-MOE): each layer holds two `FusedMoE` partitions behind one router — *resident* (268 experts, HBM) and
*spilled* (116 experts, MXFP4 bytes in pinned host memory exposed to kernels through the UVA view
`get_accelerator_view_from_cpu_tensor`, 1C:`vllm/utils/torch_utils.py:850`). Both run every step with the same
top-k ids and an `expert_map` that masks the other partition's experts; outputs add into the FP32 combine.
UVA zero-copy is chosen over a staged H2D copy because the router picks experts only at that layer (nothing to
prefetch), the PCIe bytes are identical, and zero-copy needs no extra sync or HBM staging slot; the pointers are
static, so decode graphs capture it.

Cost (arithmetic, to be measured): expected spilled experts touched per token per layer = 6 × 116/384 = 1.81;
1.81 × 4,700,160 B = 8.52 MB per GPU per layer; × 20 layers = **170 MB per GPU per stage per token**. The four GPUs
of a board read through one PEX 8749 Gen3 x16 uplink (12.87 GB/s measured for 4 concurrent GPUs = 3.2 GB/s each;
6.5 GB/s for a single GPU is the lower bound): **53 ms per stage, ≈ 106 ms per token for two stages** (26 ms/stage at
6.5 GB/s). Phase A decode is therefore expected at ≈ 7–8 tok/s. Prefill: with 4K-token chunks every spilled expert is
touched in every layer (≈ 64 tokens/expert), so each chunk streams 10.16 GiB per GPU per stage ≈ 3.4 s/stage/chunk;
a 32K prompt adds ≈ 54 s. Phase A is the correctness reference, not the speed target.

Host RAM in Phase A: 196.4 GiB − 81.2 pinned − ~24 (8 workers) − ~2 (engine/API) − ~15 (OS and other host
services) ≈ 75 GiB for the page cache, shared by the Engram warm tier and other memory-mapped services.

### 5.3 Phase B — 12 GPUs, PP3×TP4 [0–13][14–27][28–39], mirror of source 20 on stage 3

| | Stage 1: layers 0–13, board A (idx 0–3) | Stage 2: layers 14–27, board C (idx 8–11) | Stage 3: layers 28–39, board B (idx 4–7, 180 W) |
|-|-|-|-|
| layer modes | SWA 0–1, Full 2/8, Reuse | Full 14 (Engram) + Reuse 15–19, **Full 20 (r=1, candidate source)**, Reuse 21–23, Reindex 24, Reuse 25–27 | Reindex 28/32/36 + Reuse; DSpark targets 37–39 |
| weights/GPU (FP16 dense) | 25.29 GiB | 24.99 GiB | 21.66 GiB (+2.0 DSpark +0.31 embed replica at P5) |
| Engram | layer 1 | layer 14 | — |
| KV sources held | 2, 8 (r=2) | 14 (r=2), 20 (r=1) | **mirror of 20** |
| into the stage per step | — | `hidden_states` + `pre_mix`: 40,976 B/token | + `kv20_ckv` 1,024 + `kv20_ik` 256 + `cand20` 8,192 = **50,448 B/token** |

Board B (180 W power cap, slowest under sustained GEMM load, HARDWARE §A.1) gets the 12-layer stage. Engram owners
sit on boards A and C, which have separate host uplinks. Decode latency of the PP payload: 41–50 KB per hop at
6.4 GB/s ≈ 7–8 µs plus ≈ 35 µs per-transfer latency (HARDWARE §A.6) ≈ 0.09 ms/token for both hops. Prefill
(4K-token chunk): 168 MB (1→2) and 207 MB (2→3) ≈ 26 ms and 32 ms per chunk.

How the replica is registered and written in vLLM v1 (details §3.5): stage 3 builds `DeepseekV41KVSourceMirror(20)`
whose two cache layers carry the source's names `model.layers.20.attn` and `model.layers.20.attn.indexer.k_cache` and
equal specs; `get_kv_cache_configs` merges them by name with stage 2's, so the scheduler's one KV manager hands both
stages the same block ids; stage 2's layer 20 copies the records it writes into `export_ckv/export_ik` (one row per
token because r = 1) and the candidate rows into the payload; stage 3's `mirror.ingest` scatters them into its copy
with the same `slot_mapping` before layer 28 runs. Prefix caching and preemption act on one block table, so both
copies stay consistent.

Expected decode (arithmetic, P5 targets set from it): per GPU per layer a token reads ≈ 103 MB of FP16 dense
weights (98 MiB, incl. replicated `wq_a/wkv`, gate, HC) + 6 × 4.70 MB experts ≈ 131 MB → 5.2 GB over 40 layers
≈ 8 ms at ~650 GB/s, + LM head 0.5 ms + 80 intra-board all-reduces ≈ 1.2 ms + PP hops 0.1 ms → **≈ 10 ms/token
bandwidth ceiling (≈ 100 tok/s)**; with FP8 dense (P5) ≈ 6.5 ms. 1Cat reached 73.5 tok/s on V4 PP2×TP4 (1CAT §2.3),
so 30–50 tok/s target-only is a realistic first Phase-B goal.

### 5.4 Alternatives evaluated: TP8 and hetero TP

| Layout | Weights/GPU | Fits? | Pros | Cons | Verdict |
|-|-|-|-|-|-|
| TP8 on 8 GPUs (A+C) | 36.52 GiB (replicated parts do not shrink) | spill ≈ 10.6 GiB/GPU | no PP cut, no mirror; all 8 GPUs read spilled experts concurrently → spill ≈ 53 ms/token (half of PP2); GLM-5.3 decoded 20–28 % faster as TP8 than TP4×PP2 on the V100 server (1CAT §8.2) | 80 all-reduces/token cross PCIe (≈ 50–80 µs each ⇒ 4–6 ms/token); expert inter per rank 288 breaks the skinny `K%64` rule (1CAT §9) | **benchmark for Phase A** once L-MOE confirms K=288 on the TurboMind grouped path; same code as PP2 |
| TP8 on 12 GPUs | — | not uniform; TP12 illegal (64 heads) | — | — | impossible |
| PP2 hetero [0–19 TP4][20–39 TP8] | 36.05 GiB on stage 1 | stage 1 needs spill | decoder unsplit | spill remains; non-uniform TP | rejected |
| PP2 hetero [0–13 TP4][14–39 TP8] (cut 14 is legal) | 25.3 / ≈ 23 GiB | yes | no mirror; bandwidth ceiling 6.75 L vs 10 L for PP3×TP4 (L = per-layer bytes) | 52 PCIe-crossing all-reduces ≈ 3–4 ms/token eat the gain (≈ 8.9 vs 8.5 ms estimated); vLLM and 1Cat assume one TP size across PP stages | not first; revisit only if PP3 shows > 15 % of step time in PP hops/bubbles |
| **PP3×TP4 + mirror** | 25.29 / 24.99 / 21.66 GiB | yes, no spill | all collectives inside NVLink boards; cross-board traffic = 41–50 KB/token | mirror code (§3.5) | **build first for 12 GPUs** |

### 5.5 Launch shape (P3/P4; L-CORE writes the scripts at P5)

```
# Phase A (correctness reference)
CUDA_VISIBLE_DEVICES=<A0,A1,A2,A3,C8,C9,C10,C11 UUIDs>  VLLM_PP_LAYER_PARTITION=20,20
  --tensor-parallel-size 4 --pipeline-parallel-size 2 --dtype half --max-model-len 32768
  --max-num-seqs 1 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.92 --no-async-scheduling
  --ds41-spill-experts-per-layer 116 --ds41-engram-dir [private path]
# Phase B (production)
CUDA_VISIBLE_DEVICES=<A0..A3, C8..C11, B4..B7 UUIDs>  VLLM_PP_LAYER_PARTITION=14,14,12
  --tensor-parallel-size 4 --pipeline-parallel-size 3 --dtype half --max-model-len 262144
  --max-num-seqs 4 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.92 --no-async-scheduling
```

(`--ds41-*` flags are model knobs read by L-CORE's `knobs.py`; final spelling is L-CORE's. Every launch refuses to
start with < 30 GB free on `/`, D2.)

---

## 6. Byte-level memory plan (8 and 12 GPUs)

All figures: `python3 [...]/memplan.py` (header-exact census, placement rules of §3.7).
GiB = 2^30 B. Target: ≤ 29.5 GiB per GPU of everything (torch reports 31.73 GiB total per V100).

### 6.1 Weights per GPU (TP4)

Placement rules: replicated on every TP rank = `wq_a`+`wkv` (fused), q/kv norms, compressor, indexer (all 32 heads),
gate weight + bias, `hc_*`, attn/ffn norms, Engram q·k; sharded /4 = `wq_b`, `wo_a`, `wo_b`, `attn_sink`, shared
expert, routed experts (inter dim), embed/head (vocab), Engram `wkv` (K columns). Dense FP8 → FP16 (2 B/param, scales
dropped); BF16 → FP16; F32 stays.

| Item (per GPU) | Bytes | Arithmetic |
|-|-:|-|
| one expert shard (w1+w3+w2 incl. E8M0) | 4,700,160 | 18,800,640 / 4 |
| routed experts per layer | 1,804,861,440 (1.681 GiB) | 384 × 4,700,160 |
| dense, Reuse layer (FP16) | 102.65 MB (97.90 MiB) | wq_a+wkv 18.35 MB (repl.) + wq_b 20.97 + wo_a 16.78 + wo_b 20.97 + shared 17.69 + gate 3.93 (repl.) + hc 3.93 (repl.) + norms |
| dense, layer 2/8 (Full r=2) | 118.34 MiB | + compressor wkv+wgate 10.49 MB + indexer wq_b 10.49 MB + weights_proj/wk |
| dense, layer 20 (Full r=1) | 113.34 MiB | compressor wkv only |
| dense, layers 24/28/32/36 (Reindex) | 108.21 MiB | indexer wq_b + weights_proj |
| dense, layer 1 (Engram) | 172.98 MiB | + Engram `wkv_r` 78.64 MB (25600×1536×2) |
| dense, layer 14 (Full r=2 + Engram) | 193.42 MiB | |
| embed shard / head shard | 315.6 MiB each | 32,320 × 5,120 × 2 |
| same dense, native FP8 (P5 TurboMind g32) | 52.76 MiB (Reuse layer) | −46 % |

| Layout (FP16 dense) | Stage | Dense | Experts | Embed/head | **Total/GPU** | FP8-dense (P5) |
|-|-|-:|-:|-:|-:|-:|
| PP2×TP4 | 0–19 | 2.119 | 33.618 | 0.308 | **36.045 GiB** | 35.075 |
| | 20–39 | 1.967 | 33.618 | 0.308 | **35.894 GiB** | 34.988 |
| **PP3×TP4 [0–13][14–27][28–39]** | 0–13 | 1.452 | 23.533 | 0.308 | **25.293 GiB** | 24.629 |
| | 14–27 | 1.457 | 23.533 | 0 | **24.990 GiB** | 24.321 |
| | 28–39 | 1.177 | 20.171 | 0.308 | **21.657 GiB** | 21.113 |
| PP3 13/14/13 (D3, illegal cut, for comparison) | 0–12 / 13–26 / 27–39 | | | | 23.52 / 24.99 / 23.43 | |
| TP8 (8 GPUs, one stage) | 0–39 | | 33.618 | 0.154 + 0.154 | **36.52 GiB** | 35.39 |

(TENSOR_CENSUS's 22.7/24.1/22.7 GiB divided every byte by 4 and kept FP8; the difference is replicated tensors
(≈ 26 MB/layer) and FP16 dense (+46–52 MiB/layer).)

### 6.2 Runtime lines per GPU (estimates to be replaced by L-CORE's first-launch measurement)

| Line | Size | Basis |
|-|-:|-|
| CUDA context + cuBLAS/cuBLASLt workspaces | 0.45 GiB | torch 2.10 + 4 streams × 32 MiB |
| NCCL (TP4 ring + PP p2p, `NCCL_BUFFSIZE=4M`, `NCCL_MAX_NCHANNELS=4`) + custom-AR IPC buffers | 0.40 GiB | |
| CUDA graphs (FULL decode B1–B8, breakable) | 0.30 GiB | |
| activations at `max_num_batched_tokens` N | 192 KiB × N = 0.75 GiB (N = 4096) / 1.50 GiB (8192) | per token: BF16 stream in+out 81,920 B + FP16 sublayer input 10,240 + MoE w13 out 6×1152×2 = 13,824 + act 6,912 + w2 out 6×5120×2 = 61,440 + FP32 combine 20,480 = 194,816 B |
| attention workspace | 0.55 GiB | indexer prefill tile 192 MiB (1C:`sm70/indexer.py:61` `_PREFILL_TILE_MB`) + top-k [N,512] i32 8 MiB + candidates [N,2048] i32 32 MiB + candidate block-score tile ≤ 64 MiB + sparse-prefill gather tile ≤ 256 MiB (query-tiled; L-ATTN must cap it — V4's per-request gather workspace scales with context: upstream's 32-request chunk reserves "about 8.3 GiB at 1M", 1Cat cut it to 8, 1C:`sm70/sparse.py:119-121`) |
| Engram device buffers (per Engram layer on the stage) | 0.15 GiB | rows [N, 6, 264] u8 6.2 MiB + FP16 rows [1024,1536] + FP32 partial [1024, 25600] 100 MiB (1024-token sub-chunks) |
| fragmentation / Triton autotune headroom | 0.50 GiB | 1Cat GLM PP2×TP4 OOM'd at 0.92 inside a Triton autotuner (1CAT §8.2) |
| SWA KV | (128 + N + 64) × 1,024 B × layers on stage | FP16 records, ring + current chunk |
| compressed KV + index-K | context × (bytes/token below) | FP16 records |
| DSpark (P5, stage 3 only) | 3 draft SWA caches 3 × 128 × 1,024 B per sequence + `main_hidden` [N, 15360] FP16 = 120 MiB at N = 4096 | weights counted in §6.3 (+2.0 GiB: experts 6.72/4 = 1.68 GiB + dense/Markov ≈ 0.35 GiB) |

KV bytes per token of context per GPU (replicated on the 4 TP ranks; MQA):

| Stage | Sources held | Compressed KV | Index-K | **Total B/token** | 8K | 32K | 128K | 256K | 1M |
|-|-|-:|-:|-:|-:|-:|-:|-:|-:|
| PP2 s1 (0–19) | 2, 8, 14 (r=2) | 1.5 × 1024 | 1.5 × 256 | 1,920 | 15 MiB | 60 MiB | 240 MiB | 480 MiB | 1.88 GiB |
| PP2 s2 (20–39) | 20 (r=1) | 1,024 | 256 | 1,280 | 10 MiB | 40 MiB | 160 MiB | 320 MiB | 1.25 GiB |
| PP3 s1 (0–13) | 2, 8 | 1,024 | 256 | 1,280 | 10 MiB | 40 MiB | 160 MiB | 320 MiB | 1.25 GiB |
| PP3 s2 (14–27) | 14 (r=2), 20 (r=1) | 1,536 | 384 | 1,920 | 15 MiB | 60 MiB | 240 MiB | 480 MiB | 1.88 GiB |
| PP3 s3 (28–39) | mirror of 20 | 1,024 | 256 | 1,280 | 10 MiB | 40 MiB | 160 MiB | 320 MiB | 1.25 GiB |

Window KV is per sequence, not per token of context: 128 slots × 1,024 B × 40 layers = 5.2 MB per sequence across
all stages (the V4 584-B record would be 3.0 MB but cannot hold V4.1's QAT values, A3). Whole model: 3,200 B/token per TP rank in FP16 records (3.6× the model-native 890 B, MODEL_ARCHAEOLOGY §4.7) +
1,280 B for the PP3 mirror; SWA ≈ 58.6 MiB per 14-layer stage at N = 4096. HC residual in the PP buffers:
`hidden_states` 4 × 5120 × 2 = 40,960 B/token (BF16) + `pre_mix` 16 B; the static PP receive buffers are
N × 40,976 B = 160 MiB at N = 4096 (inside the activations line).

### 6.3 Totals per GPU

`max_num_batched_tokens` = 4096 (P2–P4 default):

| Layout / stage | Weights | Runtime (incl. KV) at 8K | at 256K | at 1M | Total at 8K | **Total at 256K** | Total at 1M |
|-|-:|-:|-:|-:|-:|-:|-:|
| PP2 s1 (all experts resident) | 36.05 | 3.35 | 3.80 | 5.21 | 39.39 | 39.85 | 41.25 |
| PP2 s1 **with 116-expert spill** | 25.89 | 3.35 | 3.80 | — | 29.24 | 29.69 | — |
| PP2 s2 **with spill** | 25.73 | 3.04 | 3.34 | — | 28.77 | 29.07 | — |
| PP3 s1 | 25.29 | 3.17 | 3.47 | 4.41 | 28.46 | **28.76** | 29.70 |
| PP3 s2 | 24.99 | 3.17 | 3.63 | 5.03 | 28.16 | **28.62** | 30.02 ✗ |
| PP3 s3 (+mirror) | 21.66 | 3.01 | 3.31 | 4.25 | 24.67 | **24.97** | 25.91 |
| PP3 s3 + DSpark (P5: +2.0 weights, +0.31 embed replica) | 23.97 | 3.01 | 3.31 | 4.25 | 26.98 | 27.28 | 28.22 |

`max_num_batched_tokens` = 8192 adds 0.75 GiB activations + 0.06 GiB SWA: PP3 s1 at 256K = 29.57 GiB (over by
0.07) → 8K prefill chunks require FP8 dense (P5, −0.66 GiB/GPU).

Conclusions: (1) Phase A fits at 32K only with the 116-expert spill (29.2–29.7 GiB). (2) Phase B fits 256K on every
stage with ≥ 0.7 GiB to spare; 512K fits (stage 2: 29.10 GiB); 1M needs FP8 dense or packed KV records.
(3) `--gpu-memory-utilization 0.92` (29.19 GiB managed by vLLM: weights + profiled activations + non-torch + KV;
the 0.5 GiB fragmentation line and Triton transients live in the remaining 2.54 GiB). KV capacity at 0.92: stage 2
binds at ≈ (29.19 − 24.99 − 2.60 − 0.06) GiB / 1,920 B ≈ 860K tokens; stage 1 ≈ 1.04M; Phase A stage 1 ≈ 262K.

### 6.4 Host RAM (196.4 GiB)

| | Phase A (PP2, spill) | Phase B (PP3) |
|-|-:|-:|
| pinned expert spill | 81.2 GiB | 0 |
| worker processes (≈ 3 GiB anon each) | 24 GiB (8) | 36 GiB (12) |
| engine core + API server | 2 GiB | 2 GiB |
| OS and other host services | ≈ 15 GiB | ≈ 15 GiB |
| Engram pinned staging (3 slots × 4096 × n_layers × 6 × 264 B per process) | < 0.1 GiB | < 0.1 GiB |
| **left for page cache** (Engram warm tier; other memory-mapped services compete) | **≈ 74 GiB** | **≈ 143 GiB** |

At 4 KiB of page cache per cached row, 143 GiB holds ≈ 37 M rows (≈ 5 % of the 768 M rows; the locality simulation
touched 3.06 % of each table over 1.13 M tokens, ENGRAM §5). The FP4 Engram variant (97 GiB fully RAM-resident) fits
only in Phase B and stays quality-gated (D2).

---

## 7. Bring-up order and test plan

### 7.1 Two implementations per module

Every lane ships each module with `impl ∈ {"torch", "sm70"}` (knob `VLLM_DS41_<LANE>_IMPL`): `torch` is a slow,
obviously-correct on-GPU path at real shapes and real TP sharding (gather + torch softmax for attention, per-expert
dequantised loops for MoE, service + torch math for Engram); `sm70` is the fast path. The torch path is validated
against L-REF first and then serves as the in-tree oracle for the sm70 path (same inputs, same process), so
integration (L-CORE) can start on `torch` impls while kernels are still being built.

### 7.2 Schedule and dependencies

| When | L-CORE | L-ATTN | L-MOE | L-ENGRAM | L-REF |
|-|-|-|-|-|-|
| day 0 (now; weights still downloading) | **C0**: contracts, topology, knobs, config + registry, quant claim, test scaffolding, import-boundary test | QAT twins (Triton + torch) | MXFP4 contract generalisation + shape tests on synthetic weights | hash + layout (validated against the GGUF metadata constants in ENGRAM §2), repack tool | `ref_qat.py` (exhaustive tests), `ref_engram.py`, `weights.py`, `ref_model.py` |
| shards land | loader + `skip_checkpoint_weight`, HC, model skeleton on `torch` impls | rope, compressor, window KV, indexer, candidates (`torch` then `sm70`) | router, spill partition, layer tests | run repack → `[private path]`, service, module | golden captures: per-op layers {0,1,2,3,14,20,21,24,25,28}, full-model short goldens, range audit |
| I0–I2 | PP payload, worker/runner hooks, dump hook | sparse impl, KV specs + grouping, mirror | sm70 decode path | prefetch + latency tests | compare.py runs for every lane |
| I3–I4 | launch scripts (dev), Phase A/B runs | long-context ladder fixes | QPN-M1 re-template (P5) | Engram stats under load | quality smoke + the llama.cpp cross-check |
| P5 | frontend ports, async PP, DSpark, fused HC, custom-AR sizes | packed KV records (≥ 512K) | TurboMind g32 dense FP8 | FP4 Engram variant (quality-gated) | matched GSM8K-64 / HumanEval-32 / LongBench gates |

### 7.3 Unit tests per lane (all vs L-REF golden, gates §4.5; files `tests/models/deepseek_v41/test_<lane>_*.py`)

| Lane | Tests |
|-|-|
| L-REF | QAT twins exhaustive (65,536 FP16 scalars per cast; 10^7 random + constructed-tie blocks per scale rule); hash multipliers/primes/offsets equal the GGUF constants (ENGRAM §2); compressed vocab = 99,092; golden manifest integrity |
| L-CORE | topology table (§3.1) for all 43 layer ids; `make_stage_plan` accepts (20) and (14,28), rejects 13/14/13 with a message naming layer 13; HC ops vs golden; loader: every non-skipped tensor consumed exactly once, rank slices per §3.7, dtypes, zero bytes read from `*.engram.embed.*`; PP schema sender == receiver; logits FP32; NaN-free on zero padding rows |
| L-ATTN | QAT bitwise vs L-REF twins; RoPE θ/YaRN per layer type (0, 2, 20); window KV records; compressor r=1 and r=2 incl. a chunk ending at an odd position, decode completing a pair, preemption + recompute, prefix-cache hit at a block boundary; indexer at layers 2, 20, 24 (scores, top-512 overlap); candidate blocks at 20K and 25K context from golden op inputs (set equality); sparse attention decode + prefill; one full layer per mode (0, 2, 3, 20, 21, 24, 25); mirror: ingest bytes == source cache bytes at the same slots; all-invalid row → zero output |
| L-MOE | router parity; per-shape bitwise A/B of the TurboMind path vs a dequantised FP16 reference GEMM (W13 1152×5120, W2 5120×576; E 384/top-6 and 128/top-3; T ∈ {1..8, 64, 512, 4096}); MoE layer vs golden; resident+spilled == all-resident (within FP32 summation-order tolerance 1e-6 rel); CUDA-graph capture/replay B1–B8 bitwise vs eager; TP top-k consistency check |
| L-ENGRAM | hash ids vs L-REF over 100K tokens incl. sequence start, chunk boundaries, resumed requests, dead-token handling; row file: 2 M random rows byte-equal to safetensors, header checks; service: runner-order assembly, padding rows zero, async fill, admission prefetch hit rate; module output vs golden at layers 1 and 14; gather latency p50/p99 at T = 1 and T = 4096, cold and warm, on the root NVMe |

### 7.4 Integration order

1. **I0 single layers** (TP4, board A, eager, golden inputs): layers 0, 1, 2, 3, 14, 20, 21, 24, 25 (consumers run after
   their source to populate its cache). Gate: per-layer stream rel-RMS ≤ 5e-3.
2. **I1 layer-subset model** on one board: layers {0, 1, 2, 3, 14, 15, 20, 21, 24, 25, 28, 29} with their original
   ids (closed under the source relation) + embed + final collapse + head (≈ 22 GiB/GPU, no spill); L-REF runs the
   same subset (`--layers`). Prompts ≤ 2K teacher-forced, plus one 20K prompt for the candidate mask. Gate: §4.5
   end-to-end rows.
3. **I2 pipeline plumbing on the subset**: PP2 cut at 20 (boards A, C), then PP3 cuts at 14 and 28 (A, C, B).
   Gate: outputs **bitwise identical** to I1 (PP moves tensors only; same TP layout, same kernels — run with the pinned
   TurboMind tactic cache and `CUBLAS_WORKSPACE_CONFIG=:4096:8`); mirror check knob on.
4. **I3 Phase A full model**: PP2×TP4, 8 GPUs, 116-expert spill. Smallest end-to-end smoke (§7.5), then the quality
   smoke at ≤ 8K.
5. **I4 Phase B full model**: PP3×TP4, 12 GPUs, no spill. Gate vs I3: top-1 ≥ 99.5 %, logit rel-RMS ≤ 5e-3 (the
   spill changes FP32 summation order only). Context ladder with real decode: 8K → 16K → 32K (candidate mask
   active above 16,384) → 64K → 128K → 256K; record KV capacity, prefill peak HBM, TTFT, prefill/decode tok/s.
6. **I5** each P5 optimisation A/B'd against I4 outputs with the same gates before it becomes default.

### 7.5 Smallest end-to-end smoke (Phase A)

`<｜begin▁of▁sentence｜>The capital of France is` (7 tokens), greedy, 16 new tokens, `--max-num-seqs 1`.
Pass = no NaN/Inf anywhere; Engram hash ids of the 7 positions equal the golden; teacher-forced top-1 at all 7
positions equals the full-model FP32 golden (computed on CPU, layer-streamed, minutes) and final-logit cosine
≥ 0.999; first generated token " Paris"; per-layer stream rel-RMS ≤ 5e-3 in dump mode.

### 7.6 Focused quality smoke (deterministic, both phases)

12 fixed prompts in `tools/ds41_ref/prompts/` (chat template, thinking mode where noted): short factual (EN),
Chinese factual, Python completion, three GSM8K-style problems, multi-turn chat with a mid-conversation system
message, DSML tool call (parser check, cf. 1Cat #597), verbatim paragraph copy (Engram-sensitive), 8K document
summary, needles at 16K / 32K / 128K / 256K. Metrics: teacher-forced top-1 agreement, mean |ΔNLL| and logit cosine
on the golden continuation's first 256 tokens (vs `v100-semantic`, gates §4.5); greedy first-32-token exact match
(informational); needle retrieval exact. At P4/P5 the 1Cat discipline applies to every optimisation: matched
GSM8K-64 / HumanEval-32 / LongBench subsets with no aggregate regression (1Cat #283/#344).

Independent cross-check with llama.cpp: feed the identical token ids (prompt + our greedy continuation) to a llama.cpp build
with the smalinin MXFP4 GGUF once its pull completes, and to `llama-deepseek41-long` (`GGML_CUDA_REFERENCE_F32=1` oracle mode)
on the Q4_K_M; compare per-position top-1 / top-5 overlap and PPL (informational: their Engram and expert quantisation
differ from the official FP8/MXFP4). Exact cross-checks that do not depend on quantisation: Engram row ids vs
`llama-ds41-dump`, candidate blocks at 25K vs `examples/ds41-dump/check_candidates.py` (which matched the official
two-level top-k on 917/917 queries).

---

## 8. Risk register (top 10)

| # | Risk | Early signal | Mitigation | Owner |
|-|-|-|-|-|
| 1 | **FP16 range overflow** (sink/BOS rows; V4 measured 63.5k at layer 40) → inf/NaN or silent clipping | NaN on prompts ≥ 1 token after BOS; §4.4 audit maxima | BF16 stream + FP32 math (A4), FP32 sublayer outputs until the audit clears FP16; no saturating stores; NaN sentinel in dump mode | CORE, REF |
| 2 | **Silent-wrong V4 semantics** carried into V4.1 (ratio ≤ 1 treated as SWA-only incl. RoPE/YaRN, per-head q-norm, hash routing, `hc_head`) | layers 20–39 stream drift; I0 per-layer gates | separate package (A1) + import-boundary test; per-mode layer tests (0, 2, 3, 20, 21, 24, 25); topology table test | CORE, ATTN |
| 3 | **Discrete-decision flips** (router top-6, indexer top-512, candidate top-2048) from FP16 activations vs the reference's BF16/FP8 → drift at long context | top-k overlap below gate; long-context needle failures with short-context passes | FP32 scoring everywhere (§4.1); per-op overlap gates; TP-consistency checksum; candidate set equality from golden inputs; the llama.cpp candidate cross-check at 25K | ATTN, MOE, REF |
| 4 | **KV mirror inconsistency** in PP3 (slots, prefix cache, preemption) → wrong attention only on stage 3, only at long context | PP3 ≠ PP2 on the I2 subset; mirror CRC knob | mirror registered under the source's names so one KV manager drives both copies (§3.5); bitwise I2 gate; CRC debug check; preemption + prefix-cache tests | ATTN, CORE |
| 5 | **MXFP4 kernel generalisation** (TP4 inter 576 not ×128; QPN-M1 pinned to V4 shapes; TP8 288 breaks skinny `K%64`) → wrong output or slow fallback | per-shape bitwise A/B failures; decode MoE time ≫ V4 | generic TurboMind grouped path first, QPN-M1 off until re-templated; per-shape A/B suite; TP8 benchmark gated on K=288 | MOE |
| 6 | **Engram miss latency / disk** (root NVMe shared with OS; page cache squeezed by the 81 GiB spill and other services' mmaps; row file leaves ~44 GB on `/`) → decode stalls, cold prefill seconds | `stats()` p99 gather > 1 ms at T=1; prefill stalls at layer 1/14 | page-packed one-read rows, sub-table sharding, admission prefetch, gathers started before the PP receive; launch guard ≥ 30 GB free on `/`; FP4 RAM-resident variant (quality-gated, Phase B only) | ENGRAM |
| 7 | **Engram hash/tokenizer mismatch** → every row wrong, output plausible but degraded | row-id mismatch vs golden / `llama-ds41-dump` | assert 99,092 compressed ids + multiplier/prime checksums at startup; tokenizer sha in the row-file header; exact row-id tests | ENGRAM, REF |
| 8 | **1Cat PP/TP instability** under V4.1 (open #539 custom-AR illegal access at PP2×TP4 startup, #531 stalls with experimental split-K flags, #597 16 tok/s spin-wait and DSML corruption) | startup crash in custom AR; GPUs at 100 % util and < 150 W | sync scheduling (A9), experimental split-K flags off, `--disable-custom-all-reduce` fallback for bring-up, enable optimisations one at a time with A/B; GPU power/util recorded per run | CORE |
| 9 | **HBM over-commit** (Triton autotune allocations outside the torch allocator — the GLM 0.92 OOM; untiled prefill gathers; 8K-chunk activations) | OOM at first long request or during autotune | 4K chunks until FP8 dense lands; util 0.92 with ≥ 2.5 GiB outside vLLM's budget; every workspace capped by the §6.2 lines; profile-run measurement replaces the estimates | CORE, ATTN |
| 10 | **Host-RAM pressure in Phase A** (81 GiB pinned + 8 workers + page cache) → swap, page-cache thrash, slow load | `free -g` available < 20 GiB; load time > 15 min | Phase A only at ≤ 32K and `--max-num-seqs 1`; spill sized from measured free HBM; TP8 benchmark halves spill time; Phase B is the production target | CORE, MOE |

Also tracked (below the top 10): DSpark verification under PP + async scheduling (1Cat #836) — P5 only; reference
QAT kernels cannot run on SM70/SM86, so bit-faithfulness rests on exhaustive twin tests (§4.2); per-process TurboMind
tactic autotuning makes runs non-bitwise across restarts (pin the tactic cache for every golden comparison).

---

## 9. Amendments (coordinator; binding on all lanes)

**AM-1 (2026-10-07, from L-CORE DCR-1) — Engram weight loading.** L-CORE routes every checkpoint tensor
`layers.{1,14}.engram.*` that is not skipped (i.e. everything except `engram.embed.*`) to the owning module:
```python
class DeepseekV41Engram(nn.Module):
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Names are module-relative: 'wkv.weight', 'wkv.scale', 'q_weight', 'k_weight'.
        Returns the set of names consumed; L-CORE raises on any name not consumed."""
```
`wkv` keeps its FP8 blocks + UE8M0 scales through this path; L-ENGRAM applies the 2^10 bias and row-parallel slice itself.

**AM-2 (from L-CORE DCR-2) — unbound steps.** Memory profiling and CUDA-graph capture call the model with no
`begin_step`/`bind_batch`. In that state `DeepseekV41Engram.forward` contributes exactly zero to the stream (same
shapes/dtypes, no host I/O) and the service records that the step was a dummy. A REAL scheduled step that reaches
Engram without bound rows must raise (fail loud), never silently contribute zeros. The runner marks dummy runs explicitly
(L-CORE passes `is_dummy=True` through the forward context); Engram decides on that flag, not on the absence of rows.

**AM-3 (from L-CORE DCR-3) — HC golden precision.** L-REF captures the hyper-connection goldens (hc_mixes/pre/post/
collapse inputs and outputs) in FP64; the §4.5 HC gate stays ≤ 1e-6 measured against FP64.

**AM-4 (D5) — Engram rows read in place** from the original shards 47/48 on `/` (no page-packed repack; scales in pinned
RAM). §0 A5's row-file layout is superseded; A5's sharding, hashing and prefetch design stand.

**AM-5 (from L-ATTN DCR-1)** — the compressor returns its latent in **FP32** (FP16 rounding before the FP4 QAT changed
0.4 % of compressed records).
**AM-6 (from L-ATTN DCR-2)** — the indexer's `qr` input is **FP32** (L-ATTN computes it with a two-pass split-FP16
GEMM); FP16 `qr` dropped layer-20 exact top-512 agreement to 60 % and the min overlap below the 98 % gate.
**AM-7 (from L-ATTN DCR-3)** — §6 memory plan adds the vLLM single block-pool overhead: at 256K, 509/764/626 MiB/GPU
(PP3) and 764/844 MiB/GPU (PP2). L-CORE's launchers size `--gpu-memory-utilization` with it.
**AM-8 (from L-ATTN DCR-4)** — `vllm/config/vllm.py` (owner L-CORE) adds `DeepseekV41ForCausalLM` to the breakable-CUDA-graph
architecture list. V4.1 attention builders declare `AttentionCGSupport.NEVER` until the P5 full-decode-graph work.
**AM-9 (L-ATTN finding)** — the official `Indexer.forward` scores index sources 2/8/14 against the last-published
index-K cache on decode steps that complete no ratio-2 pair (stale state). The port keeps each source's OWN K (design +
upstream vLLM). L-REF adds an `own-k` reference mode; decode comparisons for layers 2–19 use own-k goldens; the
stale-state behaviour is documented as a reference-implementation artefact, and both modes' end-to-end agreement is reported.
**AM-10 (L-ENGRAM DCR-E1)** — §3.6 row-file format replaced by in-place reads: `row_dir` = the shard directory or the
checkpoint dir with its index (as D5). **AM-11 (DCR-E2, binding on L-CORE)** — `sampled_fill` is read in runner order,
one id per request; a resumed/preempted request passes ALL committed tokens as its prompt to the Engram service; FULL
CUDA-graph mode is refused on Engram stages (breakable graphs only). **AM-12 (DCR-E3)** — the §4.5 Engram module-stream
gate is measured against a reference fed the same BF16 stream input (A4 storage alone costs 1.8–2.5e-3 vs the FP32-stream
golden); the raw golden delta is reported, not gated. Launch defaults: `VLLM_DS41_ENGRAM_IO_THREADS=2` (PP2), `=1` (PP3).
**AM-13 (L-REF DCR-1)** — end-to-end gates vs the `v100-semantic` reference: top-1 agreement ≥ 98 %, |ΔNLL| ≤ 0.02,
KL ≤ 0.01; logit rel-RMS and cosine are diagnostics only (the reference fails them against itself: QAT steps amplify FP32
noise ~5–6×/layer; prefill-vs-decode of the same tokens gives rel-RMS 0.016–0.20 with 100 % top-1).
**AM-14 (DCR-2)** — vs `ref-exact`: |ΔNLL| ≤ 0.06 (the ideal v100-semantic target itself sits at 0.045–0.049).
**AM-15 (DCR-3)** — the 5e-3 per-layer gate applies only to single layers fed the golden's inputs; multi-layer chains use
the end-to-end metrics. Range audit: stream peaks 3.6e5 on BOS (BF16 storage, never FP16); MoE combine 2.28e4 → FP32;
attention out ≤ 484 (FP16 allowed at P5); per-expert outputs 1.5e4 → keep the FP32 epilogue.
**AM-13′ (REF-BF16, supersedes AM-13's absolute |ΔNLL|/top-1/KL thresholds)** — pooled over p0–p3 and measured against a
reference run at the PORT'S precision (`v100-semantic+ownk --stream-dtype bf16 --fp16-sublayer-inputs`) scored the same
way as the port (one teacher-forced pass): |ΔNLL| ≤ ref + 0.01, top-1 ≥ max(ref − 0.01, 0.97), KL ≤ ref + 0.005; per-prompt
values reported, not gated (same-precision runs scatter 0.01–0.04 per prompt). **AM-14′** — same form vs `ref-exact`
(|ΔNLL| ≤ ref + 0.01). End-to-end gates only catch gross errors (planted Engram-skip: |ΔNLL| 0.094, caught; dropped
attention sink at layer 20: +0.009, NOT caught) — the per-layer golden tests remain mandatory for every change.
