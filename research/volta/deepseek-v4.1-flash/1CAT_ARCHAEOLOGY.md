# 1Cat-vLLM source archaeology for DeepSeek-V4.1-Flash on SM70 (lane L-1CAT)

> **Origin.** PLI Labs research archive, file `v100-research/deepseek-v4.1-flash/1CAT_ARCHAEOLOGY.md`. Published 2026-11-18 in the Volta V100 research
> repository.
> **Edited for publication.** Machine names, private file paths, network addresses, session names and the
> operator's name were replaced with generic terms (`the V100 server`, `[private path]`, `[address]`,
> `the owner`). Links to files that are not published were turned into plain text. No figure and no finding
> was changed. Dates inside the note are its own; later results are in the build pages of this repository.

Status: COMPLETE (2026-10-06, lane L-1CAT; progress file `documents/plans/exec-logs/2026-10-06-ds41/L-1CAT.the progress log`).
Sections are numbered as in the brief and appear in the order they were written (1, 7, 8, 2, 4, 5, 6, 3, 9).

**Key findings for the port (details in the cited sections):**

1. Base: 1Cat `origin/main@2d507837` = upstream vLLM `4ff865c38` (2026-05-29, `v0.21.1rc0`+438) + 2,987 1Cat
   commits, no later upstream merge; DS-V4 SM70 support was hand-backported from vLLM v0.26.0 (#159) (§1).
2. 1Cat has **no** V4.1/Engram code anywhere; upstream vLLM main **does** (`vllm/models/deepseek_v41/`, merged
   vllm#56208/#56228/#56214, 2026-09-10/11, ~55 follow-ups) — but it is a rewrite on infrastructure 1Cat lacks,
   so it is a semantic porting guide, not a cherry-pick (§7).
3. Reusable as-is: TurboMind HMMA GEMM family, MXFP4 expert *format*, software-FP8, the 584-B `fp8_ds_mla` KV
   record (upstream V4.1 keeps it on non-SM100), SM70 sparse-MLA kernels (index-list driven), indexer scoring +
   top-k, Q-norm/RoPE/KV insert (64 heads × 512 unchanged), grouped WO-A geometry, mHC TileLang FP16, router top-k
   (has a 384-expert case), TP4 push all-reduce, CUDA-graph/workspace infra (§2, §4, §9).
4. Hard gaps: FP8 linears are `[32,32]` UE8M0 and 1Cat's SM70 FP8 path is `[128,128]`-only with no SM70 MXFP8
   kernel; MXFP4 MoE allow-list/QPN-M1 kernel pinned to V4-Flash shapes; V4 code treats `compress_ratio <= 1` as
   SWA-only (V4.1 ratio-1 layers 20-39 are full-length compressed — silent-wrong-result hazard incl. RoPE/YaRN);
   kv-source/index-source sharing, candidate mask and Engram are new (§5, §9).
5. Topology constraint: upstream refuses PP splits inside a kv-sharing group; with `kv_source_layer_ids
   [2,8,14,20]` the only legal stage starts are {2,8,14,20}, so PP2 cuts at 20 (20/20) and a balanced PP3 is
   impossible without new cross-stage KV forwarding (§9).
6. Engram analogue already in 1Cat: Qwen4Exp PLE pinned-host/disk-mmap tiers with software-E4M3 UVA gathers —
   the pattern that fits 188.9 GiB of Engram rows on a 196 GiB host (§4, §9).
7. Evidence caveats: 1Cat's 73.5 tok/s PP2×TP4 / 65.1 tok/s TP8 DS-V4 numbers come from DGX-1 hosts with
   cross-island NVLink; an outside user reproduced only 16.3 tok/s with the 1.5.0 wheel (issue #597, open), and
   on the V100 server GLM-5.3 ran faster as TP8 than TP4×PP2 (§2.3, §8.2).
Date: 2026-10-06. Host: the V100 server. Read-only research; no code changed anywhere.

Evidence conventions:

- `1Cat@<sha>:<path>:<line>` cites 1CatAI/1Cat-vLLM source. Bare `path:line` references to 1Cat files
  are at **`1Cat@2d507837`** (the L-BUILD clone's `origin/main`). Files were first read from the API
  snapshot `d2a0cc20d0d4` (PR #1011, 2026-10-06 23:26 UTC) in `[private path]`
  while the clone downloaded; see the pinning note below for the equivalence.
- `vLLM@<sha>:<path>:<line>` cites vllm-project/vllm (pinned main `b6d8e8afd985`, 2026-10-06
  22:02 UTC, copied to `[private path]`).
- `#NNN` without a repo means a 1CatAI/1Cat-vLLM PR/issue; upstream ones are written `vllm#NNNNN`
  and `sglang#NNNNN`. PR/issue bodies are saved under `[private path]`.
- "Inference:" marks a conclusion not directly stated by a source.

**Pinning note (added after the clone landed).** The L-BUILD clone `[private path]`
(`.clone-complete` 23:34 UTC) has `origin/main` = **`2d5078372d7f`** (merge of PR #1010,
2026-10-06 21:00 UTC). The API snapshot `d2a0cc20d0d4` is 13 commits later (PR #1011) and those
13 commits touch only GGUF files plus `vllm/config/kernel.py` (GitHub compare API). Every
DeepSeek-V4, FP8, MXFP4, mHC, all-reduce, worker and envs file cited below was checked
byte-identical between the two (sha1 of `git show origin/main:<path>` = snapshot file), so all
`1Cat@…` line numbers below are valid at **`1Cat@2d507837`** (the commit L-BUILD branches from).

---

## 1. Identity

| Item | Value | Evidence |
|-|-|-|
| Repo | `1CatAI/1Cat-vLLM` ("V100 / SM70-focused vLLM engineering fork"), 1,238 stars, pushed 2026-10-06 | `gh api repos/1CatAI/1Cat-vLLM` |
| HEAD analysed | `origin/main` **`2d5078372d7f1de8fc66515b7b758a1a09aedd62`**, 2026-10-07 05:00 +0800 (= 2026-10-06 21:00 UTC), "Merge pull request #1010 …dflash-page832" | `git -C [private path] log -1 origin/main` |
| Upstream vLLM base | **`vllm-project/vllm@4ff865c38eac`** (2026-05-29, "[Bugfix] Disable allreduce_rms_fusion when pipeline_parallel_size > 1 (#43616)") = `v0.21.1rc0` + 438 commits; `v0.26.0` is 1,815 commits ahead of it | `git merge-base --is-ancestor 4ff865c38 origin/main` = true; compare API `v0.21.1rc0...4ff865c38` "ahead 438", `v0.26.0...4ff865c38` "behind 1815"; `1Cat@2d507837:docs/design/sm70_v100_migration_control.md:862-883` names this HEAD as the migration target, and source builds report `vllm-0.21.1rc1.dev438+g4ff865c38…` (`:4856`, `:24315`); `docs/design/sm70_deepseek_v4_flash_bringup.md:15-27` ("diverges near v0.21.1rc0 … v0.26.0 is 1,815 commits ahead") |
| History shape | 20,078 commits on `origin/main`; 2,987 are 1Cat commits after the base (first: `ee0bc6a87` 2026-06-14 "[Core] Add SM70 V100 migration paths"); **no later upstream merge** (no commit in `4ff865c38..origin/main` predates 2026-06); upstream fixes arrive as hand backports (e.g. #159 backported the dependency-closed V4 subset of vLLM `v0.26.0` + main `dc818c198d3f`) | `git rev-list`, `git log` date filter; PR #159 body |
| Authors since base | yangzhuxinyzx 2,654 · Leonccaa 111 · Peuqui 107 · Leon 37 · 1CatTCat 13 · others ≤9 | `git log --format=%an 4ff865c38..origin/main` |
| Releases (tags) | v0.0.2 (2026-03-21), v0.0.3 (05-01), v1.0.0 (05-13, *pre*-rebase lineage — does not contain `4ff865c38`), v1.1.0 (05-27), v1.2.0 (06-20), v1.2.1 (06-23), v1.2.2 `644d8a7c` (07-19), v1.3.0 `6ada86ed` (08-17), v1.5.0 `d8f42b39` (09-02), **v1.5.1 `589a5d1e` (10-03, latest)**; `origin/main` is 1,122 commits past v1.5.1 | `gh release list`; `git tag`; release notes in scratch `release-v1.5.*.md` |
| Release wheels | v1.5.1 `1cat_vllm-1.5.1-cp312-cp312-linux_x86_64.whl` (181,116,240 B) SHA256 `757f2200ba6bd20f7e3e5c8fc68c7e05461df6d284baf676ac0c16ce19434959`; v1.5.0 wheel SHA256 `2a4d6bee4e19d315b142f2c563059f3064ddeeca563a6bdc828c33e1073c825b` (Python 3.12, CUDA 12.8, torch 2.10, SM70) | release notes / assets |
| DeepSeek-V4 history | `vllm/models/deepseek_v4/` arrived with the upstream base (Woosuk Kwon refactor #43004…#43077, 2026-05-18/19; NVFP4 MoE #42209 05-22); SM70 route added by **#159** (merged 2026-08-03, `e8646577`), 41 1Cat commits on that directory since the base; last: `cc60d5392` 2026-10-05 (#874) | `git log -- vllm/models/deepseek_v4/` |
| Other 1CatAI repos | 1cat-studio, 1Cat-vLLM-Gaudi, 1CatDL, 1Cat-vLLM-Intel (archived), 1Cat-Tunnel, CloudGamingDemo, 1CatQA, 1cat-tunnel-client, 1Cat-LinkStat, 1cat-llmtest, `1CatAI/lmdeploy` (fork). **No "1Cat-LLM" repo exists.** TurboMind is vendored in-tree (`csrc/sm70_turbomind/lmdeploy/…`), not consumed from the lmdeploy fork at build time | `gh repo list 1CatAI`; `1Cat@2d507837:CMakeLists.txt:366-444` |

### Branches relevant to DeepSeek / V4 / SM70 (444 remote refs)

None mention V4.1, `deepseek_v41` or Engram (searched all ref names, all commit messages
`git log --all -i --grep 'v4.1|v41|engram'`, and `git grep -i 'deepseek_v41|DeepseekV41|engram'`
across every `refs/remotes/origin/*` tree: the only hit, `vllm/model_executor/kernels/ple/ngram.py`,
is a false positive on the identifier `PLENgram`). DeepSeek-V4 branches (merge status into
`origin/main`):

| Branch | Last commit | Status |
|-|-|-|
| `agent/v100-dsv4-pp2tp4-{fp8-prescaled-decode,fp13-gemv,mxfp4-direct-order,mxfp4-prescaled-decode,qpn8-accuracy}-20260826` | 2026-08-26/28 | merged |
| `agent/private-v100-dsv4-{pp2tp4-followup,fused-fp16-aux,prescale-spec-match}-2026082x` | 2026-08-27/28 | merged |
| `agent/v100-dsv4-pp2tp4-fused-fp16-aux-20260826` (superseded by #348) | 2026-08-26 | open |
| `agent/private-v100-dsv4-deterministic-fp8-tactics-20260827` (#349, closed: stabilise TurboMind FP8 tactic selection) | 2026-08-27 | open/unmerged |
| `codex/pr-dsv4-bmm-capability-20261003` (#816), `codex/pr-mxfp4-repack-20261002` (#767), `codex/v100-skinny-moe-kernel-20261003-074912` (#742) | 2026-10-02/03 | merged |
| `perf/sm70-tp4-allreduce-defaults` | 2026-09-12 | merged |


---

## 7. Existing DeepSeek-V4.1 / Engram work (1Cat, upstream vLLM, SGLang, community)

### 7.1 1Cat-vLLM: none

No branch, PR, issue, commit message or source file in 1CatAI/1Cat-vLLM mentions DeepSeek-V4.1,
`deepseek_v41`, `DeepseekV41ForCausalLM` or Engram (§1 search; PR list of all 849 PRs and 162
issues filtered by title — nothing). `vllm/config/engram.py`, `candidate_blocks.py`,
`sparse_mqa_indexer.py`, `transformers_utils/configs/deepseek_v41.py`, the V4.1 tokenizer/encoding
and parser files are all **absent** from `1Cat@2d507837`. The closest existing 1Cat code is the
Qwen4Exp/Qwen3.8-Flash-Next **PLE** n-gram embedding stack (§4, §9), which upstream itself treats
as "the same mechanism" as Engram (`vLLM@b6d8e8af:docs/features/engram.md`, "Supported models"
table: `Qwen4ExpForCausalLM … ple_layer_ids — same mechanism, called PLE").

Hard V4.1 blockers inside 1Cat's V4 path (each verified in source; detail in §2/§5/§9):

| Blocker | Where | Why V4.1 trips it |
|-|-|-|
| Quant-method override keyed on `model_type == "deepseek_v4"` | `1Cat@2d507837:vllm/models/deepseek_v4/quant_config.py:120-132` | V4.1 `model_type` is `deepseek_v41`; the FP8/MXFP4 config would not be claimed |
| SM70 FP8 TurboMind admits only `weight_block_size == [128, 128]` | `vllm/model_executor/layers/quantization/fp8.py:365-370`; `vllm/model_executor/kernels/linear/scaled_mm/sm70_fp8.py:445-446` ("requires 128 x 128 block weight scales"); E4M3 TurboMind kernels instantiated with `GroupSizeV=128` only (`csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_8.cu:112-124`) | V4.1 checkpoint is `weight_block_size [32, 32]`, `scale_fmt ue8m0` (`[...]/config.json` `quantization_config`) |
| SM70 MXFP4 MoE allow-list = V4-Flash geometry | `vllm/model_executor/layers/quantization/mxfp4_sm70_moe.py:40-43,246-286` (hidden 4096, inter 2048, 256 experts, top-6) | V4.1: hidden 5120, `moe_intermediate_size` 2304, 384 experts, top-6 (config) |
| MXFP4 QPN M1 decode kernel hard-codes V4 TP4 shapes | `csrc/sm70_turbomind/ops/mxfp4_qpn_m1_sm70.cu:1182-1203` (`{1,4096}`, `{256,4096,128}`, `{6,512}`…) | V4.1 TP4 W13 K=5120 N=1152, W2 K=576 N=5120, 384 experts |
| SM70 sparse MLA / compressor accept only ratios 4/128 (+SWA) | `vllm/models/deepseek_v4/sm70/sparse.py:226-243` (C4/C128 branches, `ValueError` otherwise); `vllm/models/deepseek_v4/compressor.py:198-213` (`assert compress_ratio in [4, 128]`) | V4.1 `compress_ratios` ∈ {0,1,2} (config: layers 0-1 → 0, 2-19 → 2, 20-39 → 1, 3 DSpark layers → 0) |
| No KV-source / index-source sharing, no candidate-block mask, no Engram | whole V4 package | V4.1 `kv_source_layer_ids [2,8,14,20]`, `index_source_layer_ids [2,8,14,20,24,28,32,36]`, `candidate_source_layer_id 20` (block 8, top 2048), `engram_layer_ids [1,14]` |
| Hidden-size–specific fast paths | mHC FP32 staging requires `hidden_size == 4096` (`vllm/model_executor/kernels/mhc/tilelang.py:783-796`); FP16/FP13 GEMV `_K = 4096` (`vllm/models/deepseek_v4/sm70/gemv.py:13-23`); static PP transfer schema `(1, 4, 4096)` (`vllm/v1/worker/gpu_worker.py:1118-1146`); TP4 push all-reduce exact 8-KiB payload = 4096 halves (`csrc/custom_all_reduce.cuh:98,206-208`) | V4.1 hidden 5120 → these fall back to slower generic routes (not errors), except where noted in §5 |

### 7.2 Upstream vLLM: full DeepSeek-V4.1 implementation on `main` (merged)

`vllm-project/vllm` main (`b6d8e8afd985`, 2026-10-06) ships `vllm/models/deepseek_v41/`
(nvidia + amd + common), `vllm/config/engram.py`, `docs/features/engram.md`,
`vllm/model_executor/kernels/attention/dsa/candidate_blocks.py`, tokenizer/parser/renderer files and
tests. Introduction and key follow-ups (all merged; `gh pr view`, saved in `scratch/upstream/pr-*.md`):

| PR | Merged | What |
|-|-|-|
| vllm#56208 | 2026-09-10 (`b47b01cf`) | Frontend: `DeepseekV41Config`, `tokenizers/deepseek_v41{,_encoding}.py`, `parser/deepseek_v41.py`, DSML tool parser, reasoning parser, Rust renderer/parser |
| vllm#56228 | 2026-09-10 (`9b959b86`) | Model definitions: new `vllm/models/deepseek_v4_1/` (attention 1,225 lines, `common/engram.py` 1,033, cache_utils 1,254, compressor 323, nvidia/model 1,111, dspark 558, flashmla/flashinfer_sparse, quant_config 229) + `kernels/attention/dsa/candidate_blocks.py`, `kernels/linear/mxfp8/deep_gemm.py`, mHC kernel updates, `sparse_attn_indexer.py` (+97) |
| vllm#56214 | 2026-09-11 (`e77daef8`) | "Support DeepSeek-V4.1-Flash": registry, `config/engram.py`, `mxfp8_utils`, FP8 tests (+378), `test_engram.py` (+778), compressor/indexer/mHC tests, KV offload/connectors |
| vllm#56512 | 2026-09-13 | Engram DP sharding, async **prefetch of CPU-offloaded rows on a side stream**, `/dev/shm` shared host tables |
| vllm#56741 | 2026-09-14 (`cf1584f3`) | Rename `deepseek_v4_1/` → `deepseek_v41/` |
| vllm#56893 / #56935 | 2026-09-15/16 | MXFP8 whole-KV record (SM100 FlashMLA only) / NVFP4 compressed KV |
| vllm#56254, #56266, #56962, #57604, #54049 | 09-15…10-02 | DeepGEMM sparse MQA logits, Mega-Gate, Mega-mHC, MegaMoE staging (Hopper/Blackwell-only perf) |
| vllm#56926 | 2026-09-25 | Engram: serialize offloaded lookups, pack host tables into huge pages (`use_thp`) |
| vllm#58678 | 2026-09-26 | Shard the Engram `wkv` projection across TP |
| vllm#58132, #56227, #59532 | 09-18…10-06 | SWA-bounded / decoder replay layers (memory/perf) |
| vllm#59327, #59639 | 2026-10-01 | Faster Engram host lookups (sorted rows); Triton 3.8 lookup-kernel crash fix |
| vllm#58560 | 2026-10-05 | Keep the compressor ring out of the null block |
| vllm#60064 | 2026-10-06 | Permute mega-attention `wq_b`/`wo_a` while loading |

55 commits touch `vllm/models/deepseek_v41/` (plus 9 under the old `deepseek_v4_1/` path);
list in `scratch/upstream/v41-commits.txt`, `v4_1-commits.txt`.

#### What upstream V4.1 changes relative to upstream V4 (the porting guide)

Upstream V4.1 is a **rewrite, not a patch** of V4: `attention.py` 1,157→1,494 lines with 1,079
changed lines, `nvidia/model.py` 2,029→1,641 (2,088 changed), `compressor.py` 499→327, cache_utils
1,749→1,583 (1,960 changed) (`diff` of `upstream/src/vllm/models/deepseek_v4*/`). V4.1's NVIDIA
model still imports V4 pieces (`from vllm.models.deepseek_v4.nvidia.model import …` for the MoE
base/router, `vLLM@b6d8e8af:vllm/models/deepseek_v41/nvidia/model.py:1-118`), and keeps the class
name `DeepseekV4Attention` / `DeepseekCompressor`. Semantic deltas, with upstream locations:

1. **Compression ratios {0,1,2}** instead of {SWA, 4, 128}. Ratio 1 = "full-length compressed
   cache", no gate, no pooling, no state cache; ratio 2 = gated softmax pooling with an FP32
   state cache (`vLLM@b6d8e8af:vllm/models/deepseek_v41/compressor.py:174-251`,
   `attention.py:304-321`). The V4 ratio-4 overlap/`coff` logic and ratio-128 dense path disappear.
2. **KV-source sharing**: compressors and compressed-KV caches exist only on
   `kv_source_layer_ids`; every compressed layer reads `max(s ≤ layer)`'s cache via the forward
   context (`attention.py:322-351,573-601,1111-1144`; consumers return no `KVCacheSpec`).
   **Upstream refuses PP splits inside a kv-sharing group** (`attention.py:594-601`
   "PP splits inside a v4.1 kv-sharing group are not supported").
3. **Index-source sharing + K from the KV latent**: indexers exist only on
   `index_source_layer_ids`; others reuse the shared `topk_indices_buffer`; the index key is
   `k_norm(wk(latent))` of the kv-source compressor latent, and non-owning index sources share
   the kv source's paged K cache — no indexer-local compressor or hidden-state K GEMM
   (`attention.py:1226-1240`).
4. **Candidate-block mask**: the indexer at layer 20 publishes its top 2048 blocks of 8 compressed
   positions; later indexers mask their scores to those blocks before top-k
   (`attention.py:1237-1240,1329-1375`; kernels `vllm/model_executor/kernels/attention/dsa/candidate_blocks.py`
   — pure Triton max-reduce/flag/mask, no `tl.dot`, no FP8).
5. **Engram** on layers 1 and 14 (`common/engram.py`, 1,537 lines at main): tokenizer-compressed
   vocab map (99,092 ids), multiplicative-XOR multi-head n-gram hashing into prime-sized buckets,
   FP8-E4M3 rows of 256 with UE8M0 scales per 32, `wkv` projection, per-hyper-connection-branch
   sigmoid gate, residual add into the HC stream; `hash_cache` keyed by the first local layer's
   SWA slots for lookbacks across chunks (`engram.py:1-35`). Lookup kernel
   `_engram_lookup_kernel` reads FP8 rows (optionally from **pinned host memory over UVA**) and
   writes **bf16** (`engram.py:617-692`); post-`wkv` gating `_fused_engram_post_wkv_kernel`
   (`engram.py:1155`). `EngramConfig.cpu_offload` defaults **True** (pinned host tables, side-stream
   prefetch), plus `use_thp`, `embedding_across_dp`, `dp_shared_memory` (`vLLM@b6d8e8af:vllm/config/engram.py:33-60`).
6. **Linear quantization**: `weight_block_size == [32,32]` + UE8M0 → `ModelOptLinearMethod`
   with `kMxfp8Static` weights / `kMxfp8Dynamic` activations (`deepseek_v41/quant_config.py:193-216`);
   routed experts stay **MXFP4** (`Mxfp4MoEMethod`) unless `moe_quant_algo == NVFP4`.
   `override_quantization_method` also claims `deepseek_v41{,_text}` (`quant_config.py:155-165`).
7. **KV record**: V4.1 writes DeepSeek's 528-B all-FP8 record only on SM100; **every other arch
   keeps the V4 584-B `fp8_ds_mla` record** (448 FP8 NoPE + 64 BF16 RoPE + 7 UE8M0/64 + pad)
   (`deepseek_v41/attention.py:122-131`, `common/ops/cache_utils.py:38-55`). Hence 1Cat's packed
   584-B SM70 decode kernels remain format-compatible.
8. **MoE**: same `sqrtsoftplus`/`noaux_tc` router base reused from V4, `num_hash_layers=0`
   (no hash-routed layers), `gate.bias_vl` applied to image tokens only
   (`nvidia/model.py:124-151`, `amd/model.py:119-127`, `nvidia/vl_model.py:15,137`).
9. **DSpark**: 3 draft layers with their own expert count/top-k
   (`dspark_n_routed_experts`=128, `dspark_num_experts_per_tok`=3; `nvidia/model.py:134-141`,
   `nvidia/dspark.py:381-392`), Markov head, confidence head default-on
   (`dspark.py:136-160`); target layers `[37,38,39]`.
10. Vision tower/CED, decoder replay, SWA-bounded replay, Mega-* fusions, sequence parallel —
    performance or out-of-scope features; not needed for a first SM70 text port.

**Upstream files 1Cat would need (port, not cherry-pick).** A literal cherry-pick is not viable:
V4.1 depends on upstream infrastructure that is absent from 1Cat's 2026-05-29 base, e.g.
`vllm/model_executor/layers/sparse_mqa_indexer.py`, `vllm/v1/attention/backends/mla/sparse_indexer.py`,
`vllm/model_executor/layers/fusion/quant_activation.py`, `vllm/model_executor/layers/fused_moe/moe_output.py`,
`vllm/model_executor/kernels/linear/mxfp8/deep_gemm.py`, `vllm/config/engram.py`,
`candidate_blocks.py` (presence check of 25 dependency paths against both trees, scratch log).
1Cat ported V4 the same way ("dependency-closed backport", PR #159). The port set is:
`transformers_utils/configs/deepseek_v41.py` (flattening text/vision config), `config/engram.py`,
`models/deepseek_v41/common/engram.py` (hash + lookup + gating; replace bf16/FP8-native Triton
with FP16 + software E4M3 decode), `kernels/attention/dsa/candidate_blocks.py` (as is),
the V4.1 deltas of `attention.py`/`compressor.py`/`nvidia/model.py`/`quant_config.py`/`dspark.py`
re-expressed on top of 1Cat's `vllm/models/deepseek_v4/sm70/*`, and the frontend
`tokenizers/deepseek_v41{,_encoding}.py` + `parser/deepseek_v41.py` + tool/reasoning parsers.

### 7.3 SGLang

SGLang main (`90aa1cb0fbc0`, 2026-10-06) has V4.1: `python/sglang/srt/configs/deepseek_v41.py`,
`srt/layers/engram.py` (38,932 B), `srt/layers/attention/dsv4/dsv41_sparse.py`,
`srt/mem_cache/dsv41_request_window.py`, Engram kernels `kernels/ops/embeddings/engram_{hash,gather,gate}.py`
(tree API). Merged integration PRs include sglang#38798 ("dsv4.1: remaining model and runtime
integration", 2026-09-18), sglang#38949, sglang#39677; dozens remain open (e.g. sglang#40106
"numa_shared Engram host table layout", sglang#42787 "shard the engram tables on hash-column
boundaries", sglang#42604 "native block32 DeepGEMM", sglang#39914 "SM120 candidate fallback").
All target SM90/SM100/SM120/AMD/NPU; none target SM70.

### 7.4 Community ports outside 1Cat (reference only; none on Volta)

- `yhfgyyf/vllm-deepseek-v4-sm89` (pushed 2026-10-06, 179★): DeepSeek-V4.1-Flash incl. Engram and
  DSpark on **SM89 (8× RTX 4090 48GB) and SM120**; has `tests/kernels/attention/test_dsv4_sm89_portable_ops.py`
  — the nearest existing non-Hopper V4.1 port (SM89 has native FP8, so its linear path is not
  Volta-applicable).
- `guqiong96/Lvllm` (NUMA CPU+GPU hybrid MoE, "requires … NVIDIA GPU (SM75+)"): lists
  DeepSeek-V4.1-Flash support and a `dsv41_serve_tp2_3090_dspark.sh`; "DeepSeek-V4.x peak ≈ 590 GB
  resident (≥ 640 GB advised)" host RAM — not viable on the V100 server's 196 GiB.
- Code search also finds V4.1 package copies in `weicj/vLLM-2080Ti-Definitive` (SM75 fork; its
  README claims no V4.1 result), `ttdxq/gfx906-vllm`, `wtdcode/vllm-backport`. No SM70 V4.1 port
  was found anywhere (gh code search `DeepseekV41ForCausalLM`, 2026-10-06).


---

## 8. Local context on the V100 server

### 8.1 `[private path]` (production GLM tree; read-only inspection)

Branch `codex/glm53-sm70-source-audit`, HEAD `00193c04` (2026-09-30); `origin` is a blob-less
partial clone (only `git log`/working-tree reads were used, so no lazy fetch touched it).
`d743d0c9` merged 1Cat main `2c8b9baa` (2026-09-30 00:24 +0800) for the 09-29 venv upgrade.
Local-only commits (`git log origin/main..HEAD`, 12 non-merge) and whether 1Cat
`origin/main@2d507837` has absorbed them:

| Local commit | What | In 1Cat main? |
|-|-|-|
| `73f332ec` [Model][SM70] Accept DFlash embed replica across split `load_weights` calls (GLM5Next) | loader fix | **Yes** — upstreamed as **#651** (merged 2026-09-26, `f472bb0e9`/`5b43a285b`); `vllm/models/glm5next/nvidia/model.py:844-863` carries the replica tracking |
| `95bc5ef4` Gate drafts for structured output until reasoning ends; **tolerate faulted NVML devices** | scheduler + `platforms/cuda.py` | **No** — `vllm/platforms/cuda.py:937` on main still does `[cls._get_physical_device_name(i) for i in range(device_ids)]` (a drained/Xid-74 card raises); no reasoning-gated draft logic in `v1/core/sched/scheduler.py` |
| `0004fc44` Gate async-path drafts on structured output until reasoning ends | scheduler | **No** |
| `ff20de68` Fill worker drafts before the grammar bitmask on the non-deferred path (`EngineCore._fill_structured_output_drafts`) — fixes 1Cat issue #442 class failure under PP>1 + async scheduling + spec decode | `v1/engine/core.py` | **No** — symbol absent from `v1/engine/core.py` on main. Clean PR branch exists at `ProprietaryLegal/1Cat-vLLM:fix/structured-drafts-nondeferred-bitmask` (`924d3dc7`); not filed upstream (PR list has no matching title) |
| `00193c04` SM70 D256 GQA: Q8001..Q8191 prefill with KV < 8192 takes the dense path | `flash_attn_v100.py` guard | **Superseded (inference)** — main replaced the Q8192-kernel padding with a "Q8000 core + leading fringe" dispatcher (`vllm/v1/attention/backends/flash_attn_v100.py:1620-1640,1740-1760`), so the local guard's precondition (padding to the Q8192 kernel with `kMinTotalKV` 8192) no longer exists; re-test the 8032-token crash case before dropping it. Irrelevant to DeepSeek MLA. |
| `d2edf9b3`, `097183fa`, `815b41f0`, `d10c2873`, `409baf03`, `8f96ac4d`, `05b33b79` | GLM SM70 source-audit docs, guarded NVFP4/grouped/router probes, sampler-logit capture tests | No (local diagnostics; not needed upstream) |

**Consequence for the DS-V4.1 port:** if the V4.1 server will use DSpark/MTP with PP>1 and
structured output (tool calls with `response_format`/`json_schema`) or run while a drained card
is present, carry `95bc5ef4`+`0004fc44`+`ff20de68` onto `ds41/sm70-v41` — they are model-agnostic
engine fixes that 1Cat main still lacks.

### 8.2 Lessons from prior the V100 server research that apply to this port

| Lesson | Source | Bearing on DS-V4.1 |
|-|-|-|
| On the V100 server (cross-board PCIe ~5 GB/s), GLM-5.3 B1 decode was **faster as TP8×PP1 than TP4×PP2** (+20–28% decode, ≈4× prefill; legal-prose 67.6 vs 56.9 tok/s) because the single-sequence PP bubble cost more than the PCIe all-reduce hop | `docs/models/glm-5.3-flash/ONECAT_LAUNCH_20260917.md:207-219` (experiment D) | Do not assume PP×TP4 beats TP8 on the V100 server; 1Cat's own DS-V4 numbers (TP8 65.1 vs PP2×TP4 73.6 tok/s) were taken on a DGX-1 hybrid cube-mesh where the cross-island pairs also have NVLink (issue #597 describes that topology) — the V100 server's boards have **no** NVLink between them. Benchmark both. But see §9: MLA KV is replicated per TP rank (issue #597 measured TP8 could not hold 16K context at 0.85 util), and V4.1's kv-sharing groups restrict PP cuts. |
| 1Cat GLM PP2×TP4 first request OOM'd at `--gpu-memory-utilization 0.92` inside a Triton autotuner that allocates outside the torch allocator; `0.90` + `VLLM_PP_LAYER_PARTITION=24,21` fixed it | `ONECAT_LAUNCH_20260917.md:63-90` | V4.1 Triton kernels (indexer, sparse MLA, Engram lookup) autotune/allocate similarly; budget ≥2 GiB/GPU headroom; use `VLLM_PP_LAYER_PARTITION` (respecting kv-sharing cut points, §9) |
| GLM TP4 (v100-skinny overlay) failed a fixed endpoint budget (final-logit rel-RMS 0.041 vs 0.0047 for TP2) and FP32 collective accumulation alone did not fix it; a matched CPU TP4 oracle passed, so the failure was partition-matched GPU numerics upstream of the head | `docs/models/glm-5.3-flash/TP4_NUMERICAL_DIAGNOSTICS.md:1-60`, `the progress log:1020-1030` | Expect TP-layout-dependent FP16 drift; validate V4.1 against a **partition-matched** reference (same TP slicing) and use token-local, not only aggregate, metrics. 1Cat's own policy is the same: greedy identity is not a gate, matched GSM8K/HumanEval/LongBench are (#283, #344) |
| Ordered grouped-MoE combine (ascending expert id, separate FP32 mul/add, shared expert added before the single FP16 rounding) was built and passed exact-oracle tests | `the progress log:1054-1118`; `docs/runtime/v100-skinny/ordered-combine/probe-20260917T054458Z/` | 1Cat's MXFP4 path already keeps "routing, quantization, accumulation order" bitwise (#160/#164/#181/#308: top-6 direct-order + half2 reducer, bitwise vs reference); V4.1's top-6 keeps that contract, but the 384-expert/2304-inter shapes need the same bitwise A/B before admitting the fast paths |
| Marlin shared-workspace use is safe only with per-call synchronization and foreign-stream rejection (regression PASS with private vs shared workspace bit-exact) | `docs/runtime/v100-skinny/marlin-workspace/probe-20260917T061300Z/RESULT.md` | 1Cat rejects Marlin for DS-V4 experts on SM70 (`mxfp4.py:851-872`); relevant only if a Marlin fallback is used for the DSpark draft experts or SM75 cards |
| Native FP16 GEMM reduction precision was **not** the TP4 culprit (strict vs reduced FP16 bit-identical; CPU-oracle rel-RMS 1.9e-5) | `docs/runtime/v100-skinny/native-gemm-precision/probe-20260917T055500Z/RESULT.md` | Look elsewhere (activation range, FP16 overflow in mHC/sink rows — cf. 1Cat #658/#715) when V4.1 drifts |
| KDA fused-gate API silently ignored GLM's `safe_gate/lower_bound` kwargs in 1Cat 1.2.2 (wrong numerics without an error) | `VOLTA_COMPATIBILITY.md` KDA chunked-prefill row | Same hazard class for V4.1: reused V4 kernels must be checked to honour V4.1-only arguments (compress ratio 1/2, candidate mask, kv-source prefix), not just accept them |
| llama.cpp DS-V4 on sm_70 ran (≈12 tok/s, 8 cards) once quantized-K rotation was disabled; `>8` GPUs in one llama.cpp process hit CUDA's 8-peer VMM mapping limit | `roadmap.md` Phase 1/3.5/3.6 | The GGUF fallback for V4.1 (if ever used) inherits both; vLLM's one-process-per-GPU model avoids the peer-mapping limit as long as custom all-reduce IPC stays within each TP4 group |
| the V100 server GPUs 0/1 threw NVLink Xid 74 (2026-09-22) and were drained; 1Cat needed the NVML tolerance patch | `ONECAT_LAUNCH_20260917.md:221-231` | Address GPUs by UUID; keep `95bc5ef4` (above) |
| v100-skinny QPN NVFP4/MXFP4 kernels (MIT, dnv2003) were evaluated here for GLM; 1Cat has since merged them for MoE as `--moe-backend sm70_skinny` (#742) | `docs/runtime/v100-skinny/FOUNDATION.md` | Candidate V4.1 expert backend with fewer shape hard-codes than the TurboMind QPN-M1 path (§2, §9) |


---

## 2. DeepSeek-V4 support in 1Cat — how each component executes on SM70

All paths at `1Cat@2d507837`. "HMMA" = Volta tensor cores via `mma.sync.aligned.m8n8k4`
(TurboMind `SM70_MMA_884`, `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/arch/mma_sm70.h:32-34`;
QPN kernels `fp8_qpn8_sm70.cu`, `mxfp4_qpn_m1_sm70.cu`, `skinny_moe_qpn_sm70.cu` each contain an
`mma.sync … m8n8k4` asm block). cuBLAS FP16 GEMM also uses HMMA. **Triton `tl.dot` on Volta does
not reach tensor cores in this stack** (Triton 3.6.0 from torch 2.10): PR #250 measured 3.3 TFLOP/s
for Triton vs 54 TFLOP/s cuBLAS FP16 and 13 TFLOP/s plain FP32 CUDA cores on the indexer shape, and
the code comment repeats it (`vllm/models/deepseek_v4/sm70/indexer.py:369-376,590-597`) — so every
Triton kernel below is CUDA-core FMA arithmetic (inference from that measurement; consistent with
Triton having dropped MMAv1/Volta lowering). Software FP8 = E4M3 bytes held as `uint8` and decoded
with integer/FP32 arithmetic (`vllm/models/deepseek_v4/common/ops/fp8_software.py:21-78`;
`needs_software_fp8()` is true below SM89, `common/ops/cache_utils.py:26-30`). The AOT gate in #159
confirmed "17 Triton kernels emitted sm_70 cubins; no PTX contained native E4M3 instructions".

### 2.1 Entry, dispatch and model-level guards

| Item | Location | SM70 behaviour |
|-|-|-|
| Package entry | `vllm/models/deepseek_v4/__init__.py:10-26` | NVIDIA path imports `nvidia/model.py` (`DeepseekV4ForCausalLM`), `nvidia/mtp.py`, `nvidia/dspark.py`; registry `vllm/model_executor/models/registry.py:102,618-620` |
| dtype guard | `nvidia/model.py:1349-1356` | SM70/SM75 requires `--dtype half` (no BF16) |
| Sparse-MLA impl selection ("sole platform check") | `attention.py:104-135` | `is_device_capability_family(70)` → `DeepseekV4SM70SparseImpl` ("SM70 FP16 Triton sparse MLA with packed FP8 KV"); Turing (SM75) takes the same route since #798 |
| Per-layer SM70 switch | `attention.py:198` `_use_sm70_path` | selects grouped O-proj, SM70 qnorm/RoPE/KV insert, FP16/FP13 GEMV |
| KV dtype | `attention.py:811-821` | any `fp8*` KV dtype is rewritten to `fp8_ds_mla` (584-B packed record; `vllm/v1/kv_cache_interface.py:382-386`) — `--kv-cache-dtype fp8_e5m2` has no effect (confirmed by issue #597) |
| PP intermediate tensors | `nvidia/model.py:1080-1095` | PP carries the 4 mHC streams `[tokens, 4, hidden]` (bring-up doc: collapsing to 2-D would change semantics) |

### 2.2 Component map

| Component | 1Cat file / function | SM70 kernel & arithmetic | Tensor cores? | Dequant / formats | Alignment, workspace, defaults |
|-|-|-|-|-|-|
| **Sparse MLA decode** (SWA-only, C4, C128) | `sm70/sparse.py:115-384` `DeepseekV4SM70SparseImpl._forward_decode`; kernels `sm70/sparse_kernels.py:118,321,469,589,669,782` (`_sm70_sparse_paged_fp8*`) and `common/ops/sparse_decode_bmm.py` | Route order: (1) **gather + 2 cuBLAS BMMs** (`sparse_attn_decode_bmm`, #816) when `_bmm_blocker` (`sparse.py:52-91`) admits it — FP32 QK scores through softmax, FP16 probabilities into HMMA PV; (2) else paged Triton kernel reading the packed FP8 cache directly with online softmax (optionally split-K FP32 attention-state reduction #163 and QK-D split #171, env-gated, default off); C4/C128 top-k lists combined with SWA indices (`common/ops/cache_utils.py:500-555`) | BMM route: yes (cuBLAS). Triton route: no (FMA) | 584-B `fp8_ds_mla`: 448 E4M3 NoPE in 7 groups of 64 + 64 **BF16** RoPE + 7 UE8M0 scales (+pad); software decode `fp8_e4m3fn_bits_to_fp32` × `exp2(ue8m0-127)` (`sparse_kernels.py:198-213`); BF16 RoPE loaded and cast to FP16 | head_dim 512 = 448+64 asserted (`sparse.py:73-74`); C128 uses a fixed physical row stride on SM70 for FULL-graph replay (`flashmla_sparse.py:674-677`); device-side `topk_lens` loop bound; workspace via `current_workspace_manager()`; `VLLM_SM70_DSV4_SPARSE_MLA_SPLITK_{SWA,C4,C128}` and `_QK_DSPLIT` default **False** (`envs.py:484-487`); `KernelConfig.sm70_sparse.decode_bmm` policy |
| **Sparse MLA prefill** | `sm70/sparse.py:405-533` `_forward_prefill` → `common/ops/sparse_prefill_bmm.py` or `sm70_sparse_attention_gathered` (`sparse_kernels.py:838`) | dequant-gather of selected keys to FP16 workspace, then cuBLAS BMM (#816) or gathered Triton kernel | BMM: yes | same | `PREFILL_CHUNK_SIZE = 8` (upstream 32; "reserves about 8.3 GiB at 1M context", `sparse.py:117-121`) |
| **Q-norm + RoPE + KV FP8 insert** | `sm70/qnorm_rope_kv_fp8_insert.py:23-269`, called from `attention.py:694` | Triton; TP4 B1 fused variant ("TP4 Q transform with eight exact packed-KV writer CTAs") when `num_tokens == 1 and num_heads == 16` (`:224`); fixed race: NoPE copy restricted to dims 0-447 so RoPE 448-511 has one writer (#162) | no | software FP32→E4M3 encode (`fp8_software.py:fp32_to_fp8_e4m3fn_bits`), UE8M0 per 64 | `VLLM_SM70_DSV4_QNORM_KV_FUSED_TP4=True` (`envs.py:382`) |
| **Compressor** (C4 overlap / C128) | `compressor.py:233-545`; Triton `compress_norm_rope_store_triton` (`common/ops/fused_compress_quant_cache.py:170-256`) | NVIDIA non-SM70 uses CuTe-DSL; SM70 is excluded (`compressor.py:468-480`) and runs the Triton kernel with `USE_SOFTWARE_FP8=needs_software_fp8()`; `fused_wkv_wgate` marked for FP13 GEMV (`:307`); optional private FP32 state rings (`VLLM_SM70_DSV4_PRIVATE_COMPRESSOR_STATE`, `:45-52`) | no | E4M3 + UE8M0 (main KV, block 64) or FP8 + FP32 scale (indexer K, 128+4 B) | **ratio ∈ {4,128} asserted** (`:198-213`) |
| **Lightning indexer** (C4 layers, 64 heads × 128, top-512) | `vllm/model_executor/layers/sparse_attn_indexer.py:170-447` (SM70 branch `sm70_fp16_indexer`), `sm70/indexer.py` | Prefill: `torch.mm` `[tokens·heads,128]×[128,keys]` (cuBLAS) + fused Triton `relu→weight→head-sum→key-scale` epilogue (`indexer.py:590-646`); decode: fused paged Triton logits with per-head ReLU (default) or gather + cuBLAS (`_decode_logits_cublas`, `:705-803`, inside graphs only with context buckets #874); top-k: `torch.ops._C.persistent_topk` for k∈{512,1024,2048} else `top_k_per_row_decode/prefill` (`sparse_attn_indexer.py:285,416,425`; generic CUDA in `csrc/libtorch_stable/topk.cu`, `sampler.cu`) | prefill & cuBLAS decode: yes; Triton decode: no | indexer K cache **block-major** `[block_size×128 FP8][block_size×FP32 scale]` (layout bug fixed in #250); software decode | `VLLM_SM70_INDEXER_{FUSED_LOGITS,RELU,PREFILL_CUBLAS,DECODE_CUBLAS}` default "1", `PREFILL_TILE_MB=192`, `DECODE_CUBLAS_MIN_KEYS=1024` (`indexer.py:29-73`); FP4 indexer cache rejected on SM70 (`common/ops/fused_indexer_q.py:391-396`) |
| **O projection** (8 groups, LoRA 1024) | `attention.py:361-374` → `sm70/projection.py:84-100` `sm70_grouped_output_projection` | Triton inverse GPT-J RoPE (`projection.py:12-81`), then `wo_a` as a **grouped TurboMind block-FP8 BMM** (`bmm_batch_size` groups, `kernels/linear/scaled_mm/sm70_fp8.py:498-516`), then `wo_b` FP8 linear; exact B1 two-group WO-A descriptor (8×128×64, split-K 7) default (#300) | yes (TurboMind HMMA) | FP8 E4M3 block-128 + UE8M0 → FP16 in registers | K=4096,N=1024 per group at TP4 |
| **Dense FP8 linears** (q/kv LoRA, wq_b, wkv, wo_a/b, shared expert, compressor) | `quantization/fp8.py:351-370,520` → `TurboMindFp8LinearKernel` (`sm70_fp8.py:425-…`, `fp8_sm70_prepare`), QPN8 (`qpn8_blk.py:112-218`, `csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu`) | W8A16: E4M3 weights repacked `[K,N]`, FP16 scales `[K/128, N]` (`_sm70_ops.py:505-535`); TurboMind `Config_E4M3` kernels with `Transform_HMMA_SIMT_B` (SIMT dequant in registers → HMMA) instantiated for GroupSizeV 128 (`sm70_884_8.cu:110-140`); M1 "prescaled" variants absorb the E4M3 exponent bias into UE8M0 scales (#309/#316/#347); QPN8 M=1..8 for block-128 (#801); grouped BMM for WO-A | yes | E4M3→FP16 + per-(N-row, K/128) FP16 scale | **admits only `[128,128]` blocks** (`fp8.py:365-370`, `sm70_fp8.py:445-446`, `qpn8_blk.py:152-158`); `VLLM_SM70_FP8_TURBOMIND=True`, `_PRESCALED_M1_DECODE=True`, `_PRESCALED_M1_SHARED_GATE=True`, `VLLM_SM70_FP8_QPN8_PP2_TP4=False` (quality regression, #344) (`envs.py:302-336,463-465`); autotuned tactics are per-process → independent servers can pick different stable token streams (#299/#300/#349) |
| **Routed experts MXFP4** (256 × top-6, inter 2048) | `quantization/mxfp4.py:846-879` `make_deepseek_v4_mxfp4_moe_method` → `mxfp4_sm70_moe.py:372-1023` `Mxfp4SM70MoEMethod` | Load: per-expert nibble/scale repack into TurboMind layout (`_prepare_mxfp4_sm70_experts`, `:342-369`, peak-bounded by #767); decode B1: QPN M1 kernel (`mxfp4_moe_qpn_m1_sm70_out`, `:815-830`, `mxfp4_qpn_m1_sm70.cu`) or compact/direct-top6/direct-order grouped TurboMind launches (one W13 + one W2 launch for six one-row experts, #164/#308); M≤8 graph-safe buffers (`_GRAPH_SAFE_MAX_TOKENS=8`, `:44,551-600`); prefill/large M: TurboMind grouped MoE (`kMxfp4Moe`, `awq_sm70_gemm.cu:914-916,1058`); SwiGLU with clamp 10 (`silu_and_mul_with_clamp`, `:752-760`); explicit alternative `--moe-backend sm70_skinny` (#742, `fused_moe/experts/skinny_sm70_moe.py`) | yes (HMMA) | E2M1 nibbles × E8M0 scale per 32 → FP16 in registers; skinny path rebases E8M0 per expert into the FP16-decodable window (`skinny_sm70_moe.py:192-260`) | allow-list hidden 4096 / inter 2048 / 256 experts / top-6 (`:40-43,246-286`); Marlin and full dequant rejected (`mxfp4.py:851-872`); QPN-M1 kernel shape-pinned (`mxfp4_qpn_m1_sm70.cu:1182-1203`); skinny needs N%32, K%64 and a split-K in {16,8,10} dividing K/16 (`skinny_sm70_moe.py:42-47,85-88`) |
| **Router / gate** (`sqrtsoftplus`, `noaux_tc`, hash layers) | `nvidia/model.py:401-617` `DeepseekV4MoE`; router GEMV `sm70/gemv.py` | Gate GEMV: Triton FP16→FP32 GEMV (default off) or packed **FP13** GEMV (default on, `VLLM_SM70_DSV4_FP13_GEMV=True`, `envs.py:380`) for K=4096, N∈{64,256,512,1024,2048} (`gemv.py:15-23`), else cuBLAS; top-k via `csrc/moe/topk_softplus_sqrt_kernels.cu` (generic CUDA) | no (CUDA cores, memory-bound) | FP16 weights (FP13 = upper 13 bits, value-gated lossless for normal FP16, #325) | hash MoE (`tid2eid`, layers < `num_hash_layers`) needs `input_ids` (`model.py:452-466,554`) |
| **mHC** (hc_mult 4, Sinkhorn 20) | `vllm/model_executor/kernels/mhc/tilelang.py:164,640,671,1111`; `triton.py:61,183,373,562` | TileLang FP16 kernels (TileLang/TVM-FFI **0.1.10**; 0.1.9 fails on SM70 BF16 fallback, bring-up log); SM70 **FP32 staging** decode path for M=1..8 (`tilelang.py:783-878`, `VLLM_SM70_DSV4_MHC_FP32_STAGE=True`); FP16 saturation of attention-sink rows (#658; `hc_head` follow-up #715 still open) | no (TileLang FP16 SIMT; GEMM part tiny) | n/a | FP32-stage path requires `hidden_size == 4096`, `hc_mult == 4`, `num_tokens ≤ 8` (`tilelang.py:783-796`); BF16 rejected below SM80 (`:44-49`) |
| **Embedding / LM head** | `nvidia/model.py:1369-1388` | `ParallelLMHead` FP16; SM70 LM-head layout logged at startup (bring-up) | yes (cuBLAS) | FP16 | — |
| **TP collectives** | `vllm/distributed/device_communicators/custom_all_reduce.py:217-385,423-447`; `csrc/custom_all_reduce.cuh:84-240` | TP4: SGLang-style two-epoch **push** all-reduce for exact payloads (8 KiB = B1×4096 FP16 → 4 CTAs; 80 KiB M8 rows) else 1-stage/2-stage pull; TP8: optional hierarchical (`VLLM_SM70_TP8_HIERARCHICAL_CUSTOM_AR`, #170) | n/a | FP16 with FP32 accumulation (bitwise vs fixed-order oracle, #299) | admission = SM70 + fully-connected TP4 + CUDA-graph capture + FP16 + exact size (`custom_all_reduce.cuh:186-240`); generic small sizes ≤80 KiB if `VLLM_SM70_TP4_PUSH_ALLREDUCE_SMALL_MESSAGES` (envs default "1", `envs.py:6486-6489`) |
| **Pipeline parallel** | `vllm/v1/worker/gpu_worker.py:1085-1270`; `vllm/distributed/parallel_state.py:1308-1361` | metadata-free static `isend/irecv` for the exact B1 schema `{"hidden_states": [1,4,4096] FP16}` (#283, `VLLM_SM70_PP_STATIC_HIDDEN_TRANSFER=True`), else ordinary tensor-dict PP | n/a | FP16 | DSpark drafter lives only on the last PP rank; draft-local PP size 1 (#260) |
| **CUDA graphs** | `vllm/v1/cudagraph_dispatcher.py`; breakable graphs | V4 auto-enables `VLLM_USE_BREAKABLE_CUDAGRAPH=1` and decode **context buckets** (2048, 4096, …) so the indexer can use cuBLAS inside bucketed graphs (#181/#274/#874; startup log in issue #597) | — | — | FULL decode graphs B1–B8; FULL_AND_PIECEWISE typical |
| **DSpark** (V4-Flash 3 draft layers, Markov head) | `nvidia/dspark.py:49-375` (`DSparkMarkovHead`, `DSparkConfidenceHead`, `DSparkDeepseekV4ForCausalLM`); `vllm/v1/spec_decode/dspark.py`; `vllm/config/speculative.py` | N=7 drafts, M=8 verifier; anchor-first expansion, non-causal sparse SWA metadata; `main_proj` input pre-scaled by 2^-6 on SM70 to avoid FP16 overflow (`dspark.py:140-146`); MXFP4 verifier experts slot-grouped (#181, `VLLM_SM70_MXFP4_MOE_GROUPED_M8`, default off) | per component | per component | target layers 40/41/42 must sit on the last PP stage |
| **MTP** (non-DSpark) | `nvidia/mtp.py` | standard vLLM MTP; "Ordinary MTP is not compatible with this checkpoint schema" for V4-Flash DSpark (#165) | — | — | — |
| **Chat / tools** | `vllm/tokenizers/deepseek_v4{,_encoding}.py`, `vllm/renderers/deepseek_v4.py`, `vllm/tool_parsers/deepseekv4_tool_parser.py`, `--reasoning-parser deepseek_v4` | CPU | — | — | DSML tool calls observed corrupted on SM70 in issue #597 (open) |

### 2.3 Demonstrated DeepSeek-V4 runs (1Cat evidence)

Unless a row says otherwise: 8× V100-SXM2-32GB DGX-1-class hosts (two 4-GPU NVLink islands; issue #597 notes the
cross-island pairs 0↔7,1↔6,2↔5,3↔4 also have NVLink), CUDA 12.8–12.9, torch 2.10, Python 3.12,
checkpoint `deepseek-ai/DeepSeek-V4-Flash` @ `60d8d70770c6` (bring-up) or the later
"DeepSeek-V4-Flash-0731" revision (#250/#251/#274/#325), FP8 dense + MXFP4 experts, `fp8_ds_mla` KV,
`--dtype half`, CUDA graphs.

| Topology | Workload / sampling | Result | Evidence |
|-|-|-|-|
| TP8, no spec (bring-up) | 1024/256, T=1.0, top-p 1.0 | 134.1 ms TPOT = 7.46 tok/s | #159 doc |
| TP8, no spec | same | 76.3 → 33.9 → ~29 ms TPOT after active-expert, sparse split-K, compact MXFP4 | #160, #163, #164 |
| TP8, no spec, accepted baseline | same | **15.357 ms TPOT ≈ 65.1 tok/s** | #181, README |
| TP8, DSpark N7 | same | 30.42 tok/s (AL ≈1.57) | #181 |
| TP8, DSpark N7, latest main | 120-token prompt, 3,500 out, T=1.0 | **116.288 tok/s** median (acceptance 0.589, 5.12 tokens/round, verifier 35.1 ms/round); matched no-spec 66.014 tok/s | #274; `docs/design/sm70_deepseek_v4_dspark_verifier_20ms_long_context.md:30-52` |
| PP2×TP4, no DSpark | 1024/256 | QPN8 59.25→64.36 tok/s; static PP +0.22% | #283 |
| PP2×TP4, no DSpark, push AR / WO-A grouped | 1024/256 | +3.6% / +3.1% | #299, #300 |
| **PP2×TP4, no DSpark, strict quality control** | 1024/256; GSM8K-64 64/64, HumanEval 29/32, LongBench 44.740 | **73.539 tok/s** (combined endpoint 73.613–73.646) | #344, README |
| TP8 long context, `max_model_len 262144` | needle 4K–207K | 15/15 exact after indexer layout+ReLU fix; 207K prefill 768→350 s, 128K 268→148 s | #250 |
| PP5 mixed rig (3× V100-PCIe + 2× RTX 8000), DeepSeek-V4-Flash-NVFP4 / MXFP4 | 35.8K prefill / decode | skinny MoE 18.9 s / 15.5 tok/s vs default 31.9 s / 6.8 tok/s; DSpark k=5 69–76 ms/step | #742, #874 |
| TP8, AWQ W4A16 gs32 community checkpoint, DSpark 5, 48K, 32 seqs | production RAG | crashes fixed by bounded indexer workspace + `combine_topk_swa_indices` fallback (still open as issue) | issue #208 |

Launch shape used by outside reproducers (issue #597, 1.5.0 wheel, PP2×TP4):
`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,7,6,5,4 vllm serve … --trust-remote-code
--dtype half --tensor-parallel-size 4 --pipeline-parallel-size 2 --kv-cache-dtype fp8_e5m2
--max-model-len 16384 --max-num-seqs 1 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.85
--compilation-config '{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1]}'
--enable-auto-tool-choice --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4`; startup logs
showed: SM70 FP16 Triton sparse MLA, `fp8_ds_mla`, `expert_dtype 'fp4'`, TurboMind MXFP4 MoE
(256 local experts, graph-safe B1–B8), MXFP4 QPN M1 route, FP8 grouped-BMM, TP4 push all-reduce,
auto `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, decode context buckets (2048, 4096); KV ≈ 101 KB/token/rank
(3.1 GiB → 32,098 tokens at 0.85).

**Counter-evidence to weigh (all open):** issue #597 — an outside 8× V100 user with the official
1.5.0 wheel measured **16.3 tok/s** (vs 73.6 documented), with both PP stages at 100% utilisation
but 73–148 W (spin-waiting), plus 0/10 correct DSML tool-call arguments while GSM8K-64 = 64/64;
issue #539 — PP2×TP4 startup dies with `custom_all_reduce.cuh:1735 illegal memory access`;
issue #531 — the six experimental MXFP4/sparse split-K flags stall the engine every 2–4 h;
issue #82 (closed 2026-08-01, before #159) — maintainers then called DS-V4-Flash on V100
unsupported because "MXFP4 path … minimum capability is SM80". None of 1Cat's PRs reports
v100server-like topology (boards joined only by PCIe).


---

## 4. Generic SM70 infrastructure vs DeepSeek-V4-specific code

"Generic" = selected by device/dtype/shape/topology, used by several model families (Qwen3.x,
Qwen4Exp/Flash-Next, GLM-5.3, DS-V4). "V4-specific" = lives under `vllm/models/deepseek_v4/` or is
admitted only for V4-Flash tensor contracts. Maturity = default state + strongest evidence.

| Component | File(s) @ `1Cat@2d507837` | SM70 implementation | Generic / V4-specific | Maturity / evidence |
|-|-|-|-|-|
| SM70 build | `CMakeLists.txt:107-115` (arch lists incl. 7.0 for CUDA ≥12.8), `:366-444` (TurboMind SM70 sources, `ENABLE_SM70_TURBOMIND`), `:750-790` (`_sm70_sampler_C`); `flash-attention-v100/` in-tree (`setup.py` accepts only 7.0/7.2); `cmake/external_projects/vllm_flash_attn.cmake:39-45,112` (`zhinianqin/flash-attention-v100@c2eda5e6`, `-gencode=arch=compute_70,code=sm_70`) | nvcc sm_70 cubins; TileLang JIT at runtime | generic | release wheels 1.5.0/1.5.1 are SM70-only; #344 "all 38 cubins are sm_70 only" |
| TurboMind SM70 GEMM (vendored lmdeploy 0.12.1 subset) | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/{arch/config_sm70_s884.h,kernel/sm70_884_{4,8,16}.cu,gemm.cu,tuner/*}`; front-end `csrc/sm70_turbomind/ops/awq_sm70_gemm.cu`; Python `vllm/_sm70_ops.py`, `quantization/sm70_turbomind.py` | `Sm70_s884` mainloop, HMMA m8n8k4, weights (INT4/U4, E4M3, E2M1 MXFP4/NVFP4, GGUF lattices) dequantized in registers (`Transform_HMMA_SIMT_B`), FP16 group scales (`Operand_V_Pack`), split-K, runtime autotuner with tune cache | generic | default for AWQ/FP8/NVFP4/MXFP4 on SM70; tactic autotuning is per-process (non-identical streams across restarts, #299/#300) |
| FP8 block W8A16 linear | `quantization/fp8.py:351-370,458-560`; `kernels/linear/scaled_mm/sm70_fp8.py` | TurboMind E4M3 (group 128 along K, scale per N-row); exact/prescaled M1 kernels; grouped BMM | generic (Qwen3.8-FP8, GLM, V4) but **block-128 only** | default; many PRs (#212, #271, #300, #316, #347, #794) |
| QPN8 / QPN2 / QPN4 skinny decode kernels (v100-skinny lineage, MIT) | `csrc/sm70_turbomind/ops/{fp8_qpn8_sm70.cu,nvfp4_qpn2_sm70.cu,nvfp4_qpn4_sm70.cu}`; `kernels/linear/scaled_mm/qpn8_blk.py` | M≤8 (QPN8 also M≤16/32 chunked) HMMA weight-streaming GEMV-like kernels | generic | QPN8 block-128 default via capability (#801); PP2×TP4 QPN8 for V4 **off** (quality) |
| MXFP4 MoE (TurboMind) | `quantization/mxfp4_sm70_moe.py`; `mxfp4.py:846-879` | grouped TurboMind MXFP4 MoE; graph-safe B1–B8 buffers; compact/direct-order/broadcast-input decode | **V4-specific** (allow-list 4096/2048/256/top-6) | default for DS-V4 MXFP4; MiMo-V2.6 (top-8) rejected (issue #679) |
| MXFP4 QPN M1 MoE decode | `csrc/sm70_turbomind/ops/mxfp4_qpn_m1_sm70.cu:1182-1203` | HMMA m8n8k4 6-route kernel | **V4-specific** (exact TP4 V4-Flash shapes) | default (`VLLM_SM70_MXFP4_MOE_QPN_M1_DECODE=True`) |
| Skinny QPN MoE (`--moe-backend sm70_skinny`) | `fused_moe/experts/skinny_sm70_moe.py`; `csrc/sm70_turbomind/ops/skinny_moe_qpn_sm70.cu` | grouped NVFP4/MXFP4 MoE with device-side routing, in-place fragment permutation (no second copy), per-expert E8M0 rebase | generic (SM70+SM75) | opt-in since #742 (2026-10-03); measured on DS-V4-Flash PP5 mixed rig |
| NVFP4 MoE / FP8 MoE (TurboMind) | `quantization/nvfp4_sm70_moe.py`, `fp8_sm70_moe.py`, `sm70_moe_router.py` | packed TurboMind grouped MoE, route planner (batched / per-expert / dense / active-dense) | generic | default for Qwen3.6-35B NVFP4, Flash-Next, GLM-5.3 |
| Custom all-reduce | `distributed/device_communicators/custom_all_reduce.py`; `csrc/custom_all_reduce.{cu,cuh}`; `csrc/sm70_ring_collectives.cu`; `model_executor/layers/sm70_collectives.py` | TP4 two-epoch push all-reduce keyed by **payload size** (8 KiB=4096 halves, 5 KiB=2560, 80 KiB=8×5120 …), TP8 hierarchical, TP4 long-prefill fused norm, ring collectives | generic | TP4 push default (#299); TP2 SM70 hang/collapse issues #430/#652 (workaround `--disable-custom-all-reduce`, doc #759); PP2×TP4 startup illegal access issue #539 open |
| Pipeline parallel | `v1/worker/gpu_worker.py:1085-1270`; `distributed/parallel_state.py:1308-1361`; PP profiling repair #295; DSpark final-stage ownership #260 | static B1 hidden transfer; standard tensor-dict PP | static schema **V4-specific** (`[1,4,4096]`); rest generic | default on |
| CUDA graphs | `vllm/compilation/breakable_cudagraph.py`; `v1/cudagraph_dispatcher.py`; `v1/worker/workspace.py` | Breakable graphs; FULL decode B1–B8; context buckets for sparse indexer | generic infra; V4 bucket derivation (index_topk × smallest compress ratio) | default for V4 (#181, #874) |
| Software FP8 helpers | `vllm/models/deepseek_v4/common/ops/fp8_software.py`; `cache_utils.py:26-30`; #777 capability-based selection | integer/FP32 E4M3 encode/decode in Triton | generic helper (lives in V4 tree) | default below SM89 |
| Sparse MLA on packed FP8 KV | `vllm/models/deepseek_v4/sm70/{sparse,sparse_kernels}.py`; `common/ops/sparse_{decode,prefill}_bmm.py` | Triton paged FP8 + cuBLAS BMM routes | **V4-specific** (584-B record, 512=448+64 head, C4/C128/SWA metadata); GLM has a separate `models/glm5next/sm70/sparse.py` | default (BMM via `KernelConfig.sm70_sparse`, #816/#857) |
| Lightning indexer SM70 | `layers/sparse_attn_indexer.py` (SM70 branch), `models/deepseek_v4/sm70/indexer.py` | cuBLAS + Triton epilogue; paged Triton decode | V4-specific (DSA-style; GLM uses `sparse_attn_indexer_kpool.py`) | default; correctness fixes #250 (layout + ReLU) |
| Compressor (C4/C128) | `models/deepseek_v4/compressor.py`; `common/ops/fused_compress_quant_cache.py` | Triton + software FP8 | V4-specific | default |
| mHC kernels | `model_executor/kernels/mhc/{tilelang,tilelang_kernels,triton,torch}.py` | TileLang FP16 + SM70 Triton FP32 staging | shared GLM-5.3 / V4 / Qwen4Exp(HC) | default; FP16-range fixes #658 (merged), #715 (open) |
| FP16/FP13 GEMV | `models/deepseek_v4/sm70/gemv.py` | Triton, exact FP32 accumulation | V4-specific (K=4096) | FP13 default on (#325) |
| Grouped O projection | `models/deepseek_v4/sm70/projection.py` + TurboMind grouped FP8 BMM | inverse RoPE (Triton) + HMMA | V4-specific | default (#300) |
| KV-cache spec / memory | `v1/kv_cache_interface.py:358-427` (`MLAAttentionSpec` 584-B for `model_version="deepseek_v4"`, alignment padding); prefill workspace factor 8 (`sm70/sparse.py:117-121`); KV zeroing race fix #406; SWA cache-dead block recycling #765/#774; MXFP4 repack peak #767; PP memory lifetime #344 (post-load pointer tables) | — | mixed: spec V4-specific, allocator fixes generic | merged |
| YaRN / RoPE | `nvidia/model.py` (#251: YaRN off on SWA-only layers via factor 1); MTP YaRN context inheritance #410/#540 | — | #251 V4-specific; MTP YaRN generic | merged |
| DSpark speculative decoding | `models/deepseek_v4/nvidia/dspark.py`; `v1/spec_decode/dspark.py`; `config/speculative.py` | Markov/confidence heads, anchor-first N-token block, strict rejection sampler | V4-family specific (also used by other DSpark drafters) | merged (#165/#181/#183/#274), verifier fast paths opt-in |
| Host-resident n-gram tables (PLE) | `models/qwen4_exp/nvidia/ple_layer.py` (pinned-host FP8 gather `_gather_ple_fp8_from_pinned_kernel`, software E4M3 decode `_e4m3fn_byte_to_float`, disk `mmap`+`MADV_RANDOM` offload), `models/qwen4_exp/common/ple.py` (tier planner VRAM/host/disk), `v1/ple_offload/{connector,worker}.py` (separate CPU process), `csrc/ple_disk_rows.cpp`, `kernels/ple/ngram.py` | Triton gathers from pinned memory (UVA) + CPU thread-pool disk gathers | Qwen4Exp-specific today, but the closest analogue to V4.1 Engram | PLE cpu offload default for Flash-Next; disk offload opt-in, measured (`docs/design/sm70_qwen38_ple_disk_offload.md`) |
| Attention for dense models (Flash-V100, XQA, D256 GQA prefill) | `v1/attention/backends/flash_attn_v100.py`, `flash-attention-v100/` | HMMA FlashAttention | generic, **not used by DS-V4** (MLA sparse path) except the DSpark draft SWA | default for Qwen/GLM |

**Reading of the table.** The DS-V4 SM70 port is roughly half generic machinery (TurboMind GEMM
families, collectives, graphs, software FP8, mHC, workspace manager) and half V4-shaped glue
(sparse MLA/indexer/compressor for ratios 4/128, MXFP4 allow-list, GEMV/PP/mHC fast paths keyed
to hidden 4096). V4.1 can reuse the first half unchanged and must re-shape or generalise the second.


---

## 5. Architecture guards on the V4 path and their SM70 routing

Grep of the V4 path and its dependencies for `has_device_capability`, `get_device_capability`,
`is_device_capability(_family)`, `supports_fp8/fp4`, `cutlass_fp8_supported`, `sm_80/sm_90`,
`compute_80`, `< 80`, `>= 89`, `is_arch_support_pdl`, `__CUDA_ARCH__` (48 Python hits saved in
`scratch/l-1cat/guards.txt`). Classification: **I** = real instruction/hardware dependency,
**P** = support/tuning policy (a portable implementation exists or could), **R** = routing switch
that already sends SM70 elsewhere.

| Guard | Location | Class | What it protects | SM70 route in 1Cat |
|-|-|-|-|-|
| FlashMLA sparse backend `capability.major in [9, 10]` | `v1/attention/backends/mla/flashmla_sparse.py:135-137` | I | FlashMLA kernels (wgmma/TMA, SM90/SM100) | `DeepseekV4SM70SparseImpl` (Triton paged FP8 / cuBLAS BMM), `attention.py:113-135` |
| FlashMLA `get_mla_metadata` skipped on SM70 | `v1/attention/backends/mla/sparse_swa.py:498-505` | I | FlashMLA metadata op not built for SM70 | SM70 metadata path (device-side lengths) |
| C128 fixed physical row stride on SM70 | `flashmla_sparse.py:672-677` | P (correctness choice) | FULL-graph replay stability of adaptive C128 width | SM70-only stable stride |
| DeepGEMM (`fp8_einsum`, paged MQA logits, MegaMoE) needs SM90/SM100 | `vllm/utils/deep_gemm.py:44,78`; `nvidia/model.py:261-262` ("DeepGEMM MegaMoE requires SM100"); `v1/attention/backends/mla/indexer.py:438-458` | I (wgmma/tcgen05/TMA) | O-proj einsum, indexer logits, fused MoE | grouped TurboMind FP8 BMM for `wo_a` (`sm70/projection.py`), cuBLAS/Triton indexer (`sm70/indexer.py`), TurboMind MXFP4 MoE |
| `fp8_einsum` recipe by `cap.major` | `attention.py:238-244` | R | SM90 vs SM100 scale layout | never reached on SM70 (returns at `:361-374`) |
| `needs_software_fp8()` = CUDA and not ≥ SM89 | `common/ops/cache_utils.py:26-30`; #777 | I | native `cvt…e4m3` instructions (SM89+) | software E4M3 encode/decode in Triton (`fp8_software.py`) for KV insert, gather, compressor |
| FP4 indexer cache requires SM100 | `common/ops/fused_indexer_q.py:391-396`; `indexer.py:438-446` | I (FP4 paged MQA kernels) | MXFP4 indexer K cache | FP8 (128 B + FP32 scale) indexer cache on SM70 |
| Compressor CuTe-DSL only on non-SM70 CUDA | `compressor.py:468-480` | I (CuTe-DSL kernels target SM90+) | fused compress/norm/RoPE/store | Triton `compress_norm_rope_store_triton` + software FP8 |
| Fused qnorm/RoPE/KV-insert CUDA kernel: BF16 path compiled as a no-op below SM80 | `csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:381-386,441,466,535` | I (no BF16 `_typeConvert` pre-Ampere) | BF16 inputs | Triton `sm70/qnorm_rope_kv_fp8_insert.py` (FP16) |
| PDL (`griddepcontrol.wait`) only ≥ SM90 | `csrc/moe/topk_softplus_sqrt_kernels.cu:170,297,422`; `fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:408,438,478,532`; `kernels/mhc/tilelang_kernels.py:27` | I (optional) | programmatic dependent launch | compiled out; same kernels run on SM70 |
| mHC BF16 requires SM80 | `kernels/mhc/tilelang.py:44-49` | I | BF16 math; TileLang 0.1.9 BF16 fallback did not compile on SM70 | FP16 TileLang kernels + SM70 FP32 staging (`tilelang.py:783-878`) |
| `--dtype half` required on SM70/SM75 | `nvidia/model.py:1349-1356` | I | no BF16 tensor cores | FP16 everywhere; FP16-range hazards handled by #658 (mHC sink rows) and DSpark 2^-6 pre-scale (`nvidia/dspark.py:140-146`, P) |
| SM70 TurboMind "exact SM70/SM72" | `quantization/sm70_turbomind.py:106-129,177-187`; `kernels/linear/scaled_mm/sm70_fp8.py:429-438` | P/build | s884 kernels are compiled only for 7.0/7.2 | — (SM75 cards take Marlin or skinny) |
| `Fp8Config.get_min_capability` lowered to 70 when SM70 TurboMind/QPN8/dequant fallback is available | `quantization/fp8.py:159-170` | P | upstream FP8 linear min capability 89 (native FP8 GEMM) | W8A16 TurboMind/QPN8 (weights FP8, math FP16 on HMMA) |
| FP8 TurboMind admits only `[128,128]` blocks | `fp8.py:365-370`; `sm70_fp8.py:445-446`; `qpn8_blk.py:152-158,195-198` | P (layout/instantiation, not hardware) | prepared scale layout `[K/128, N]`; E4M3 kernels instantiated with GroupSizeV=128 | **none for V4.1's `[32,32]`** — needs new instantiations (see §9) |
| DS-V4 MXFP4 on SM70 must use TurboMind (Marlin / emulation rejected) | `quantization/mxfp4.py:851-872` | P | upstream `Mxfp4MoEMethod` backends need SM80+ (issue #82 reply) | `Mxfp4SM70MoEMethod` or `--moe-backend sm70_skinny` |
| MXFP4 MoE shape allow-list | `mxfp4_sm70_moe.py:246-286` | P ("currently supports … got") | untested shapes; only the QPN-M1 kernel truly hard-codes six routes/256 experts (issue #679 analysis) | **blocks V4.1 (5120/2304/384)** |
| Skinny QPN kernels SM70/SM75 only | `fused_moe/experts/skinny_sm70_moe.py:443-451`; QPN8 `qpn8_blk.py:119-130` | I (inline `mma.sync.m8n8k4` asm) | — | these *are* the SM70 kernels |
| SM70 sparse BMM admission requires 7.x | `sm70/sparse.py:52-91` (`_bmm_blocker`), `:110-112` | P | cuBLAS route is portable | SM70/SM75 only by policy |
| FP16/FP13 GEMV exact SM70 | `sm70/gemv.py:277,336,369-371,419` | P (tuned shapes) | — | cuBLAS fallback elsewhere |
| Indexer cuBLAS decode on SM70 family | `sm70/indexer.py:804-840` | P | — | paged Triton fallback |
| mHC FP32 staging exact SM70 + hidden 4096 | `kernels/mhc/tilelang.py:783-796,974-982` | P | — | generic TileLang FP16 path for other shapes |
| Turing on the SM70 route | `attention.py:104-110` (`is_device_capability_family(70)` covers 7.x), #798 | P | — | SM75 shares Triton/cuBLAS pieces; TurboMind pieces keep exact-SM70 checks |

**Conclusion.** Every hard (I) guard on the V4 path already has a working SM70 alternative in 1Cat.
The guards V4.1 will newly trip are all **P** (layout/shape allow-lists: `[128,128]` FP8 blocks,
V4-Flash MoE geometry, hidden-4096 fast paths, compress ratios 4/128) plus V4.1-only upstream
kernels that are BF16/FP8-native Triton (Engram lookup/gate) or Hopper-only (Mega-*, DeepGEMM
sparse MQA logits, FP4/MXFP8 KV records) — the latter have no SM70 route today and must be
re-expressed with FP16 + software E4M3 decode, never enabled by deleting a guard.

---

## 6. Dependencies 1Cat actually uses on this path (pins at `1Cat@2d507837`)

| Dependency | Pin | Where | Role on the V4 path |
|-|-|-|-|
| Python | 3.12 (wheels `cp312`) | release notes | runtime |
| PyTorch | `torch==2.10.0` (CUDA 12.8 wheels; torchvision 0.25.0, torchaudio 2.10.0) | `requirements/cuda.txt:7-11`, `pyproject.toml:11`, `requirements/build/cuda.txt` | cuBLAS BMM/mm routes, allocator, NCCL |
| CUDA toolkit | 12.8 recommended (CMake keeps 7.0 in `CUDA_SUPPORTED_ARCHS` for CUDA 12.8 "while CUDA 13 warns that pre-7.5…") | `CMakeLists.txt:107-115`; v1.5.0 notes "Python 3.12 / CUDA 12.8 / PyTorch 2.10 / SM70" | nvcc sm_70 |
| Triton | not pinned by 1Cat; comes with torch 2.10 → **triton 3.6.0** (the V100 server glm53 venv `triton-3.6.0.dist-info`; issue #836 env) ; `triton_kernels` external project tag `v3.5.1` (`cmake/external_projects/triton_kernels.cmake:3`) | — | all SM70 Triton kernels (FMA, no tensor cores) |
| TileLang / TVM-FFI | `tilelang==0.1.10`, `apache-tvm-ffi==0.1.10` (0.1.9 fails on SM70) | `requirements/cuda.txt:14-15`; bring-up log | mHC kernels |
| TurboMind | vendored from **lmdeploy 0.12.1** (`92c6efbfe` 2026-03-21 "Vendor lmdeploy source", `lmdeploy/version.py: __version__ = '0.12.1'`), now a 1Cat-modified subset under `csrc/sm70_turbomind/lmdeploy/src/turbomind/` (42 commits since the base); fork `1CatAI/lmdeploy` (parent InternLM/lmdeploy) not used at build time | `CMakeLists.txt:366-444` | FP8/MXFP4/NVFP4/AWQ HMMA GEMMs, grouped MoE |
| v100-skinny kernels (dnv2003, MIT) | in-tree `csrc/sm70_turbomind/ops/{nvfp4_qpn2,nvfp4_qpn4,fp8_qpn8,mxfp4_qpn_m1,skinny_moe_qpn}_sm70.cu` + `LICENSE.v100-skinny` | #443, #742 | QPN decode kernels |
| CUTLASS | `CUTLASS_REVISION v4.4.2` (headers-only FetchContent; `VLLM_CUTLASS_SRC_DIR` override) | `CMakeLists.txt:327-360` | SM70 prefill CUTLASS pieces (`qwen38_prefill_cutlass.cu`), Flash-V100 H3 (`flash-attention-v100/kernel/h3/UPSTREAM.md`: CUTLASS `da5e086d` = v4.4.2); **not** used by the DS-V4 SM70 kernels |
| Flash-Attention-V100 | in-tree `flash-attention-v100/` (`setup.py` arch 7.0/7.2 only); CMake fetch `zhinianqin/flash-attention-v100@c2eda5e6115b` | `cmake/external_projects/vllm_flash_attn.cmake:39-45` | dense attention for other models; DSpark draft SWA (#165) |
| Marlin | upstream vLLM Marlin kernels present; **rejected for DS-V4 experts on SM70** (`mxfp4.py:851-872`); Marlin serves the SM75 stages in mixed rigs (#742 discussion) | — | not on the V100 V4 path |
| ModelOpt | no runtime dependency; ModelOpt *checkpoint formats* handled by `quantization/modelopt.py` (+ SM70 W4A16 NVFP4 via TurboMind, #228); V4 NVFP4 MoE branch `quant_config.py:142-151` | — | only for NVFP4 checkpoints |
| FlashInfer | `flashinfer-python==0.6.11.post2`, `flashinfer-cubin==0.6.11.post2` | `requirements/cuda.txt:12-13` | not on the SM70 V4 path |
| FlashMLA / DeepGEMM | external projects `vllm-project/FlashMLA@a6ec2ba7bd0a` (archs 9.0a/10.0), `deepseek-ai/DeepGEMM@891d57b4db10` (9.0a/10.0) | `cmake/external_projects/{flashmla,deepgemm}.cmake` | **not built/used on SM70** |
| NCCL | from the torch wheel: `nvidia-nccl-cu12 2.27.5` (the V100 server glm53 venv) | — | PP p2p, non-custom all-reduce |
| Transformers / safetensors / hub | `transformers >= 5.15.1` (the V100 server venv 5.17.0), `safetensors >= 0.8.0` (F8_E8M0/F4 dtypes), `huggingface_hub` 1.31.0 (venv) | `requirements/common.txt:10-12` | config/weights |
| CuTe-DSL / quack / humming | `nvidia-cutlass-dsl[cu13]==4.7.0`, `quack-kernels>=0.3.3`, `humming-kernels[cu13]==0.1.2` | `requirements/cuda.txt:24-31` | installed, unused on SM70 (CuTe-DSL compressor excluded, `compressor.py:468-480`) |
| Release wheel | `1cat_vllm-1.5.1-cp312-cp312-linux_x86_64.whl` SHA256 `757f2200…4959`; bundles `_C`, `_moe_C`, `_sm70_sampler_C`, Flash-V100, TurboMind, FlashQLA | v1.5.1 assets | reference build for A/B against the source build |


---

## 3. Keyword sweep (tree, history, PRs/issues)

Counts at `1Cat@2d507837`: files under `vllm/`+`csrc/` matching (case-insensitive) · commit
subjects in `4ff865c38..origin/main` · PR titles (849 PRs). Meaningful hits only; raw lists are
reproducible with the commands in the scratch log.

| Keyword | Files · commits · PRs | What the hits are |
|-|-|-|
| deepseek | 198 · 94 · 36 | V4 model package (upstream-inherited + `sm70/`), DS-V3.2/OCR/V2 models, V4 tokenizer/renderer/tool parser, ~30 V4 SM70 PRs (#159…#874, listed in §2.3/§1) |
| v4.1 / v41 / engram | 0 · 0 · 0 (one false-positive file: `kernels/ple/ngram.py`, identifier `PLENgram`) | **no V4.1 or Engram work** anywhere in 1Cat |
| nvfp4 | 178 · 124 · 39 | ModelOpt NVFP4 W4A16 on SM70 via TurboMind/QPN2/QPN4 (#228, #445), grouped NVFP4 MoE (#270), DS-V4-Flash-NVFP4 on the PP5 mixed rig (#669, #742) |
| mxfp4 | 143 · 43 · 13 | DS-V4 MXFP4 MoE (`mxfp4_sm70_moe.py`, #160/#164/#308/#326/#767), E8M0 subnormal scale preservation (#222), Bailing-V3 MXFP4 (#468/#551), skinny MXFP4 (#742) |
| mxfp8 | 56 · 0 · 0 | upstream-inherited only: `kernels/linear/mxfp8/{emulation,marlin,flashinfer,xpu}.py`, MoE oracles. On SM70 the MXFP8 linear choices are **Marlin (requires SM75, `mxfp8/marlin.py:15-20`) or BF16 emulation** (`mxfp8/emulation.py:16-58`, full per-call dequant to BF16 — unusable on Volta). No SM70 MXFP8 kernel exists |
| fp8 | 515 · 120 · 42 | SM70 W8A16 FP8 (TurboMind/QPN8/prescaled), FP8 KV E5M2/E4M3 for dense attention (Flash-V100 bridge), DS-V4 FP8 dense paths, FP8-resident MTP experts (#595), software FP8 (#777) |
| sm70 / volta | 374+69 · 895+43 · 399+11 | the fork's whole purpose; `sm70_*` modules listed in §4 |
| marlin | 123 · 13 · 4 | upstream Marlin (AWQ/GPTQ/FP4/MXFP8) kept for SM75+; Marlin split-K FP32 reduce + AWQ non-TurboMind SM70 route (#204/#221); **explicitly rejected for DS-V4 experts** (`mxfp4.py:851-872`) |
| turbomind | 223 · 44 · 11 | vendored TurboMind SM70 GEMM (lmdeploy 0.12.1 lineage), MXFP4/FP8/NVFP4/AWQ/GGUF kernels, tuner fixes (#165 recursive `tune_mutex` deadlock, #641 tuner illegal access with FP8 MTP experts, open) |
| modelopt | 35 · 19 · 7 | ModelOpt NVFP4/FP8 loaders with SM70 routing; DS-V4 `moe_quant_algo == NVFP4` branch |
| mla / sparse / indexer / kpool | 264 / — / 105 / 26 files | DS-V4 sparse MLA + C4 indexer (§2), GLM-5.3 DSA `sparse_attn_indexer_kpool.py` + `models/glm5next/sm70/sparse.py`, Qwen4Exp QSA sparse attention; TRITON_MLA smem clamp (#211/#218); MLA prefill Flash-V100 route (#200) |
| PP2 / TP4 / pipeline | 7 / 79 files · 28 / 137 commits | PP2×TP4 DS-V4 campaign (#283, #295, #299–#326, #344), GLM TP4/PP2 (#341, #456, #503), static PP transfer, PP×spec-decode fixes (#574/#636/#639/#662 per issue #836); open PP bugs #539 (DS-V4 PP2×TP4 custom-AR illegal access), #836 (MTP4 over PP2 device assert), #479 |
| all_reduce / custom allreduce | 104 · 10 · 3 | TP4 push all-reduce (8 KiB/80 KiB/Qwen sizes), TP8 hierarchical (#170), ring collectives, TP4+RMSNorm sync reduction (#957); TP2 hang/collapse issues #430/#652 |
| gemv | 26 · 22 · 10 | DS-V4 FP16/FP13 GEMV (#168, #322, #325, #348), KDA/GLM GEMV (#392/#396), AWQ M5 batched GEMV, `glm53_fp16_gemv_sm70.cu` |
| speculative / mtp / dspark | 156 (mtp) / 14 (dspark) files | DFlash/DFlash2/DDTree/EAGLE/MTP4 on SM70 (most of the fork's 2026-08..10 work), DS-V4 DSpark N7 (#165/#181/#183/#274/#874), DSpark weight-skip (#669), DSpark draft config inheritance (issue #298) |
| kv cache | — | `fp8_ds_mla` 584-B spec (`kv_cache_interface.py:382-386`), PLE/KV offload connectors, KV zeroing race (#406), SWA block recycling (#765) |
| yarn | 29 · 8 · 3 | DS-V4 SWA-layer YaRN fix (#251), MTP YaRN context validation (#410/#540) |


---

## 9. First-cut V4 → V4.1 reuse table (1Cat side)

V4.1 facts are from `[...]/config.json` (`model_type
deepseek_v41_text`, 40 layers, hidden 5120, 64 heads × 512 (RoPE 64), `num_key_value_heads` 1,
q_lora 1280, o_groups 8 × o_lora 1024, 384 experts top-6 + 1 shared, `moe_intermediate_size`
2304, `routed_scaling_factor` 1.5, `sqrtsoftplus`/`noaux_tc`, `swiglu_limit` 10, `hc_mult` 4,
`hc_sinkhorn_iters` 20, sliding window 128, index 32 heads × 128 top-512, `compress_ratios`
`[0,0, 2×18, 1×20, 0,0,0]`, `compress_rope_theta` 160000, YaRN ×16 over 65,536,
`quantization_config {fp8, dynamic, weight_block_size [32,32], scale_fmt ue8m0, expert_dtype fp4}`,
Engram rows `[384,006,168, 384,016,682]` × 256, DSpark target layers `[37,38,39]`, 128 draft experts
top-3, block 5, Markov rank 256). V4-Flash facts are from 1Cat's bring-up doc (43 layers, hidden
4096, q_lora 1024, 256 experts, inter 2048, 64 index heads, ratios {SWA, 4, 128}, FP8 block 128).
"Reuse" means the 1Cat SM70 code runs V4.1 unchanged; "adapt" = same kernels/structure with
shape/contract generalisation; "new" = no 1Cat counterpart. Model-side semantics are L-MODEL's
remit; this table only says what the 1Cat code must do.

| Component | 1Cat V4 implementation (SM70) | Reusable for V4.1? | What must change |
|-|-|-|-|
| Config / registry / quant claim | `DeepseekV4ForCausalLM`; `DeepseekV4FP8Config.override_quantization_method` claims only `model_type == "deepseek_v4"` (`quant_config.py:120-132`) | adapt | add `DeepseekV41ForCausalLM` + a flattening `DeepseekV41Config` (port `vLLM@b6d8e8af:vllm/transformers_utils/configs/deepseek_v41.py`); claim `deepseek_v41{,_text}`; load-skip vision tower; keep `--dtype half` guard |
| **Dense FP8 linears** (wq_a 5120→1280, wq_b 1280→64·512, wkv 5120→512, wo_a 8 groups 4096→1024, wo_b 8192→5120, shared expert 5120↔2304, compressor, indexer wq_b/weights_proj, DSpark main_proj) | TurboMind W8A16 E4M3, **group 128** (`fp8.py:365-370`, `sm70_fp8.py:445`, `sm70_884_8.cu:110-140`), QPN8 block-128 M≤8, prescaled M1, grouped WO-A BMM | **adapt (largest kernel delta)** | V4.1 is `[32,32]` blocks with UE8M0 (upstream treats it as MXFP8, `deepseek_v41/quant_config.py:193-216`); 1Cat has **no SM70 MXFP8 path** (`mxfp8/marlin.py` needs SM75; `mxfp8/emulation.py` dequantises to BF16 every call). Needed: E4M3 TurboMind instantiations with GroupSizeV 32 (the MXFP4 kernels already run GroupSizeV 32 with E8M0 `uint8` scales: `Config_MXF4` at `sm70_884_4.cu:303-311`, `arch/config_sm70_s884.h:159-168`), `fp8_sm70_prepare(group_size=32)` with scales expanded along N (FP16 scales cost 1/16 of the FP8 bytes vs 1/64 at block 128 — or store E8M0 bytes as `Config_MXF4` does), relax the `[128,128]` gates, re-tune/requalify prescaled-M1 and grouped WO-A (WO-A geometry K 4096, N 1024, 2 groups at TP4 is *identical* to V4), extend QPN8 to per-32 K scales or leave it off. Exact power-of-two scales keep the prescale trick valid |
| **Routed experts MXFP4** | `Mxfp4SM70MoEMethod` (TurboMind grouped MXFP4; direct-top6/direct-order/broadcast decode; graph-safe B1–B8), QPN-M1 kernel, optional skinny backend | **adapt** | format identical (E2M1 + E8M0/32). Lift the allow-list (`mxfp4_sm70_moe.py:40-43,246-286`) to hidden 5120 / inter 2304 / 384 experts (top-6 unchanged so the 6-slot decode logic and buffers carry over). TP4 local inter = 576: W13 N=1152 K=5120, W2 N=5120 K=576 — all multiples of 32/64/128 except 1152 vs 256-wide CTAs (inference: needs a tile check). QPN-M1 kernel (`mxfp4_qpn_m1_sm70.cu:1182-1203`) must be re-templated or disabled. Skinny: `grouped_splitk` has no split for W2 K=576 (K/16=36; offers 16/8/10, `skinny_sm70_moe.py:42-47`) → add split 4/6/9/12. TP8 (inter 288) breaks the skinny K%64 rule. Expert bytes grow ~1.7× per layer-set (384×2304×5120 vs 256×2048×4096) |
| Router / gate | V4 router base (`sqrtsoftplus`, `noaux_tc`, e_score bias, hash layers); `csrc/moe/topk_softplus_sqrt_kernels.cu` (has a `case 384` and `case 128`, `:591-614`); FP13/FP16 gate GEMV only K=4096 | reuse (kernel) / adapt (GEMV) | no hash layers — V4.1's config has no `num_hash_layers` key while the V4 MoE reads `config.num_hash_layers` unguarded (`nvidia/model.py:452`; upstream V4.1 passes `num_hash_layers=0`, `deepseek_v41/nvidia/model.py:148`); 384 experts supported by the CUDA top-k; gate GEMV K=5120 N=384 falls back to cuBLAS (extend `gemv.py:15-23` later for speed); `gate.bias_vl` affects image tokens only — load, ignore for text |
| Shared expert | FP8 dense (gated SiLU), shared-gate prescale (#347) | adapt | block-32 FP8 (above); shape 5120↔2304 |
| **Attention-layer taxonomy** | everything keys on `compress_ratio <= 1` = SWA-only, `== 4` = C4+indexer, `== 128` = C128 (`sm70/sparse.py:142,155,182,226-251,428-432`, `attention.py:313,839`, `nvidia/model.py:709-728`, `nvidia/flashmla.py:153-337`) | **adapt — silent-wrong-result hazard** | V4.1 ratios are 0 (SWA-only: layers 0,1 + DSpark), **1 (full-length compressed KV: layers 20-39)** and 2 (2:1 gated pooling: layers 2-19). Under V4 logic ratio-1 layers would be treated as SWA-only and get base `rope_theta` with YaRN off; upstream V4.1 uses `compress_ratio > 0` for both compressed-KV and YaRN/`compress_rope_theta` (`vLLM@b6d8e8af:vllm/models/deepseek_v41/common/rope.py`, "YaRN applies to all compressed-KV layers (compress_ratio > 0)"). Every `<= 1` / `> 1` test must be re-audited |
| Sparse MLA decode/prefill kernels | Triton paged-FP8 (+split-K, QK-D split), cuBLAS gather+BMM; 584-B record; SWA+top-k index combine | **reuse kernels**, adapt metadata | the kernels consume index lists and the 584-B record, which upstream V4.1 keeps on non-SM100 GPUs (`deepseek_v41/attention.py:122-131`). Metadata builders need ratio-1/2 variants (V4 has C4 top-k and C128 dense-compressed lists); compressed cache read through the **kv-source** layer, not `layer.kv_cache` |
| Compressor | Triton `compress_norm_rope_store_triton` + software FP8; asserts ratio ∈ {4,128}, V4 C4 overlap (`compressor.py:198-213,274`) | adapt | ratio 1: no gate, no pooling, no state cache; ratio 2: gate + FP32 state, and no V4-style C4 window overlap (`overlap`/`coff` do not occur in `vLLM@b6d8e8af:…/deepseek_v41/compressor.py`; see `:174-251`); only on kv-source layers (2, 8, 14, 20); emitted latent also feeds the indexer K |
| KV-source sharing (`kv_source_layer_ids [2,8,14,20]`) | none (each compressed layer owns its cache) | **new** | consumers return no `KVCacheSpec` and read the source's cache via the forward context (`deepseek_v41/attention.py:573-601,1111-1144`); check 1Cat's hybrid KV allocator accepts spec-less attention layers. **PP cut points**: upstream rejects PP splits inside a kv-sharing group (`:594-601`); with ratios 2 on 2-19 and 1 on 20-39 the legal stage starts are only {2, 8, 14, 20} — all of layers 20-39 must sit on one stage. **PP2 at 20 is natural (20/20); a balanced PP3 is impossible without new cross-stage KV forwarding** (e.g. 0-7/8-19/20-39 = 8/12/20). This directly affects the plan's Phase-B PP3×TP4 |
| Indexer | SM70 FP8 indexer cache + cuBLAS/Triton scoring with per-head ReLU + `persistent_topk`; indexer has its own compressor/K (`compressor.py`, `sparse_attn_indexer.py:170-447`) | reuse scoring/top-k, adapt K path | V4.1: indexers only on 8 index sources (2,8,14,20,24,28,32,36), consumers reuse `topk_indices_buffer`; **K = k_norm(wk(kv-source latent))**, cache shared with the kv source (`deepseek_v41/attention.py:1226-1240`); 32 heads (V4 64) — SM70 kernels take `num_heads` as a parameter; FP8 (not MXFP4) indexer cache is the SM70-legal choice (`fused_indexer_q.py:391-396`) |
| Candidate-block mask (src layer 20, block 8, top 2048) | none | **new (easy)** | port `vLLM@b6d8e8af:vllm/model_executor/kernels/attention/dsa/candidate_blocks.py` (pure Triton max/flag/mask kernels; no `tl.dot`, no FP8) and the dense-indexer masking args; skip the "sparse logits" opt-in (requires MXFP4 indexer cache) |
| Q-norm/RoPE/KV insert | Triton, TP4 fused when 16 local heads (`qnorm_rope_kv_fp8_insert.py:224`) | reuse | 64 heads, head 512, RoPE 64 unchanged; only the rope selection above changes |
| O projection | inverse RoPE + grouped TurboMind FP8 WO-A + WO-B | reuse (after block-32 FP8) | identical group geometry; wo_b N 5120 |
| mHC | TileLang FP16 + SM70 FP32 staging (hidden == 4096 only), FP16 sink-row saturation #658 | reuse (generic path) / adapt (fast path) | extend staging to hidden 5120; apply open #715 (`hc_head` saturation) — FP16 overflow risk grows with depth. Upstream V4.1 restructured mHC (post folded into delayed pre, vllm#56633; Mega-mHC) for speed only |
| **Engram** (layers 1, 14; ≈2 × 101 GB FP8 rows + UE8M0) | none for DeepSeek; Qwen4Exp **PLE** stack is the analogue: pinned-host FP8 table + Triton UVA gather with software E4M3 decode (`qwen4_exp/nvidia/ple_layer.py:157-200`), VRAM/host/disk tier planner (`qwen4_exp/common/ple.py:157-308`), disk mode = retained safetensors mmaps + `MADV_RANDOM` + 32-thread deduplicated gathers (`docs/design/sm70_qwen38_ple_disk_offload.md`), separate CPU worker process (`v1/ple_offload/`) | **new model code on reusable infrastructure** | port upstream `common/engram.py` semantics (tokenizer-compressed vocab 99,092, multiplicative-XOR hashing into prime buckets, `hash_cache` on SWA slots, `wkv`, per-branch sigmoid gate into the HC stream); replace its bf16/FP8-native Triton (`engram.py:617-692`) with FP16 output + software E4M3 decode; reuse 1Cat's host-tier/disk-tier machinery. Sizing (inference): 384,006,168+384,016,682 rows × (256 + 8) B ≈ 202.8 GB = 188.9 GiB > what the V100 server can pin (196 GiB total RAM) → the 1Cat **disk-mmap tier** is the pattern that fits, measured on V100 hosts at 3.5–9.3 ms per 80-row lookup (cached/cold, 32 workers) and 0.31–0.43 s per 8K-token prefill chunk; per decode token Engram reads ≈ 2 layers × 8 heads × 3 n-gram orders × 264 B ≈ 12.7 KB (inference from config) |
| DSpark (3 draft layers) | `nvidia/dspark.py` (Markov head, optional confidence head, 2^-6 `main_proj` pre-scale), N=7 target-verifier graphs, slot-grouped M=8 MXFP4 verifier | adapt | target layers 37-39 (V4: 40-42); draft MoE is **128 experts top-3** (`dspark_n_routed_experts`), outside every SM70 MoE allow-list; block size 5; confidence head default-on upstream; layers have ratio 0 |
| KV-cache spec | `MLAAttentionSpec` 584 B/token for `model_version="deepseek_v4"` (`kv_cache_interface.py:382-386`) | reuse | only kv sources allocate; ratio-2 sources 292 B/token, ratio-1 source 584 B/token (inference) |
| Pipeline parallel | static B1 transfer schema `[1,4,4096]` (`gpu_worker.py:1121,1146`); mHC 4-stream intermediate tensors | adapt | schema `[1,4,5120]`; stage boundaries restricted as above; Engram hash state lives on PP ranks owning an Engram layer (upstream `nvidia/model.py:796-802`); DSpark on last stage |
| TP collectives | TP4 push AR exact 8 KiB (4096 halves) / 80 KiB | reuse | B1 payload 10 KiB (5120 halves): admitted only via the generic small-message path (`custom_all_reduce.cuh:192-198`) or a new exact constant; TP8 hierarchical sizes are 4096-based (`custom_all_reduce.py:32-33`) |
| CUDA graphs / buckets | breakable graphs; decode context buckets = `index_topk × smallest ratio` × {1,2,8,32,64} (#874) | adapt | smallest V4.1 ratio is 1 → re-derive buckets |
| FP16/FP13 GEMV, mHC staging, fused WQA/WKV prescale | V4-shape fast paths (K=4096) | fall back | not needed for correctness; re-tune after bring-up |
| Software FP8, TurboMind core, sampler, workspace manager, Flash-V100 (draft SWA) | generic | reuse | — |
| Tokenizer / chat template / DSML tool + reasoning parsers | `tokenizers/deepseek_v4*`, `tool_parsers/deepseekv4_tool_parser.py` | adapt | port upstream `tokenizers/deepseek_v41{,_encoding}.py`, `parser/deepseek_v41.py`, `tool_parsers/deepseekv41_engine_tool_parser.py`; note DSML corruption report on SM70 (#597) |
| Vision tower / `bias_vl` / CED | none | out of scope | load-skip |

### 9.1 Port-order implications (from the 1Cat side)

1. **Hard prerequisites before any V4.1 token:** block-32 FP8 W8A16 (TurboMind group-32 E4M3),
   generalized MXFP4 MoE contract, ratio-{0,1,2} attention taxonomy + rope selection, kv-source and
   index-source sharing, Engram (even a slow reference first), candidate mask.
2. **Topology:** PP2×TP4 with the cut at layer 20 is the only kv-sharing-legal 2-stage split and is
   balanced by layer count; PP3×TP4 on 12 GPUs needs either an unbalanced 8/12/20 split or new
   cross-stage forwarding of the layer-20 compressed KV/indexer state. TP8 avoids PP cuts but
   replicates MLA KV per rank (issue #597: TP8 could not hold 16K context at 0.85 util for V4) and
   crosses the V100 server's PCIe-only board boundary in every all-reduce (yet won for GLM on the V100 server, §8.2).
3. **Quality gates to copy:** 1Cat's own discipline — bitwise operator screens under CUDA-graph
   replay, then paired GSM8K-64 / HumanEval-32 / LongBench / needle with no aggregate regression
   (#283/#344) — and its known V4 traps: indexer cache layout + per-head ReLU (#250), KV RoPE write
   race (#162), YaRN on SWA layers (#251), FP16 sink-row overflow (#658/#715), SWA vs sparse
   threshold mismatch under DSpark (#603), per-process TurboMind tactic autotuning (#299/#349).

