# v100-skinny foundation review for GLM-5.3-Flash NVFP4

> **Origin.** PLI Labs research archive, file `v100-research/v100-skinny/FOUNDATION.md`. Published 2026-11-18 in the Volta V100 research
> repository.
> **Edited for publication.** Machine names, private file paths, network addresses, session names and the
> operator's name were replaced with generic terms (`the V100 server`, `[private path]`, `[address]`,
> `the owner`). Links to files that are not published were turned into plain text. No figure and no finding
> was changed. Dates inside the note are its own; later results are in the build pages of this repository.

Status: historical source/reproducibility review from initial discovery on
2026-09-16. The observations, gaps, recommendations, and commands below describe
that inspection slice, not the current campaign state. No model weights were
downloaded, no GPU serving was started, and no build or install was run **by this
review task**.

Subsequent work established the isolated Qwen TP2 smoke baseline; its
raw receipt records API
completion, QPN/Marlin routes, and memory. It is not the published TP4 benchmark
and is not a GLM result. GLM checkpoint acquisition also completed; see the
storage verification.
The current GLM implementation and acceptance ledger are maintained in
the progress log and the adjacent component
screening reports. Initial recommendations here to wait for those now-completed
steps are retained as historical analysis, not active launch instructions.

## TASK

Determine what the upstream `dnv2003/v100-skinny` release actually provides,
which exact engine/checkpoint it reproduces, and whether that contract can be
carried to GLM-5.3 NVFP4 on 10 and 12 Tesla V100s. Record the QPN2/QPN8/Marlin
format contracts, KV and attention state behavior, memory duplication, model
assumptions, and an honest pinned baseline for the fleet bootstrap.

The target used for the GLM analysis is NVIDIA's current
[`nvidia/GLM-5.3-Flash-NVFP4`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4),
not the much larger generic `RadixArk/GLM-5.3-NVFP4` artifact. These are
different architectures and different storage footprints; both distinctions
are called out below.

## WHAT I INSPECTED

Local external checkouts (kept outside the research repo):

- `[private path]`, cloned from
  [`dnv2003/v100-skinny`](https://github.com/dnv2003/v100-skinny), exact tag
  `v1.1.0`.
- `[...]/1cat-vllm-1.2.2`, shallow checkout of the exact
  [`1Cat-vLLM` `v1.2.2` tag](https://github.com/1CatAI/1Cat-vLLM/tree/v1.2.2),
  including its source and environment declarations.

Upstream v100-skinny source reviewed:

- `README.md`, including Quick start, Reproduction, QPN description, mixed
  precision route table, hardware limitations, and results provenance.
- `scripts/bootstrap-sm70.sh` and `scripts/serve-qwen38-native.sh`.
- `fork_patches/README.md` and the complete patch sources:
  `marlin.py`, `modelopt.py`, `gdn_attn.py`, `torch_utils.py`,
  `attention.py`, `gpu_model_runner.py`, and `custom_all_reduce.py`.
- `kernels/skinny_kernels.cu`, `docs/qpn_race_notes.md`, and the committed
  benchmark/result files linked by the README.

Current primary model metadata was read without downloading weights:

- NVIDIA model revision/API SHA:
  `09b04e5e74bca08ca8549fc736d4cdd8624bfde3`.
- `config.json`, `hf_quant_config.json`, `.quant_summary.txt`,
  `model.safetensors.index.json`, README, and the safetensors shard tree were
  read from the Hugging Face `resolve/main` endpoints.
- For comparison, `RadixArk/GLM-5.3-NVFP4` currently reports an architecture
  of `GlmMoeDsaForCausalLM` and approximately 465 GB storage, while NVIDIA's
  Flash artifact is `Glm5NextForConditionalGeneration` and approximately 204
  GB. The latter is the relevant target for the existing
  `docs/models/glm-5.3-flash` work.

## FILES / COMMITS

### v100-skinny

- Tag/commit: `v1.1.0` /
  `5b589c0dc81223e0ba65bcb3e755874723f8b515`
- Commit message: `v1.1: native mixed NVFP4/FP8 LLM inference on NVIDIA
  Volta`.
- Tag metadata describes Qwen3.8-27B unchanged mixed NVFP4/FP8 execution on
  four V100-SXM2-16GB cards.
- There is **no `docs/REPRODUCE.md`** in the tag. The README's in-page
  `## Reproduction` section (around lines 585 onward) is the supported
  reproduction document; its linked harnesses and result files are present.

### 1Cat-vLLM

- Tag/commit: `v1.2.2` /
  `644d8a7cd05ed4ecd1cd188e3c05b4bbd074f504`.
- `pyproject.toml` and `requirements/cuda.txt` declare `torch==2.10.0`,
  CUDA 12.8 wheels, `tilelang==0.1.9`, and
  `apache-tvm-ffi==0.1.9`; v100-skinny's bootstrap deliberately overrides
  the latter two to 0.1.10.
- The pinned source has Qwen3Next/GDN and broad SM70 plumbing, but no
  `glm5_next` model implementation. Its registry has
  `GlmMoeDsaForCausalLM` mapped to `deepseek_v2.py`; that is not the
  `Glm5NextForConditionalGeneration` architecture in the NVIDIA Flash
  checkpoint.

### Pinned published Qwen baseline

- Model: `RadixArk/Qwen3.8-27B-NVFP4`.
- Immutable revision:
  `554ebba9b5f1b79dc11246341960360e6ef05ef4`.
- Engine wheel URL:
  `https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.2.2/1cat_vllm-1.2.2-cp312-cp312-linux_x86_64.whl`.
- Wheel SHA256:
  `8a628983ad9d675559910372643220c418b307ddc7fd52ac65a7f5fbcb104bc6`.
- Required bootstrap override: `tilelang==0.1.10` and
  `apache-tvm-ffi==0.1.10` together.

### GLM-5.3-Flash metadata facts

From NVIDIA model revision `09b04e5e74bca08ca8549fc736d4cdd8624bfde3`:

- 33 safetensors shards, total file bytes `204,439,103,396` (190.3988
  GiB / 204.4391 decimal GB), including safetensors headers; the indexed
  tensor payload is `204,419,110,596` bytes (190.380132 GiB), which the
  metadata-only census matches exactly. The repository's `usedStorage` is
  `204,475,612,312` bytes.
- 320B total parameters / 18B activated according to the model card; 45 base
  language-model layers (`0..44`) plus an MTP/next-token layer stored under
  layer index `45`.
- `hidden_size=4096`, `num_routed_experts=288`, `num_experts_per_tok=8`,
  `num_hidden_layers=45`, `num_nextn_predict_layers=1`, vocabulary 154,880.
- Layers `0..2` are dense MLP; layers `3..44` are sparse MoE. Layer 45's
  288-expert MTP weights are BF16 in the published index (no
  `input_scale`, `weight_scale`, or `weight_scale_2` tensors).
- Attention is not Qwen GDN: 34 KDA/linear-attention layers and 11
  `deepseek_sparse_attention` layers at
  `[3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]`, with mHC, DSA indexer, and
  one-million-token declared context.
- Published ModelOpt recipe is `NVFP4` with group size 16 and `FP8` KV-cache
  metadata. `lm_head`, embeddings, all attention, routers, shared experts,
  visual modules, and the MTP layer are excluded from quantization; the dense
  MLP and routed experts carry NVFP4 tensors.
- Safetensors index counts are 36,297 quantized linear triples: 36,297 U8
  packed-code tensors, 36,297 FP8 per-group scale tensors, 36,297 F32 global
  weight scales, and 36,297 F32 activation/input scales. This is 9 dense MLP
  linears plus 42 x 288 x 3 routed-expert linears.
- Header samples confirm the W4A4 storage contract: a dense layer-0
  `gate_proj.weight` is U8 `[12288, 2048]`, its
  `weight_scale` is FP8-E4M3 `[12288, 256]`, and both global and activation
  scales are scalar F32; a layer-3 expert `gate_proj.weight` is U8
  `[2048, 2048]` with FP8 `[2048, 256]` group scales. The MTP expert sample
  is BF16 `[2048, 4096]`.

## FINDINGS

### 1. What v100-skinny actually supports

The release is a Qwen3.8-specific serving stack, not a general GLM port. Its
headline contract is mixed **W4A16 NVFP4** plus **W8A16 FP8**, with FP16
activations and FP16 KV. It assumes Qwen3.8's fused projection names,
Qwen3Next/GDN recurrent-state metadata, chain-MTP layout, chat template, and
MTP proposer behavior. The README's model-specific count is 64 layers: 48
GDN/linear-attention layers and 16 full-attention layers; only the latter
hold ordinary token KV.

The target NVIDIA Flash checkpoint instead exposes `glm5_next`, mHC, KDA,
DeepSeek sparse attention, 288 routed experts, W4A4 expert/dense-MLP
quantization, BF16 MTP, and FP8-KV metadata. None of that is an accidental
rename. The 1Cat-vLLM v1.2.2 source has no `glm5_next` implementation, and
the skinny GDN patch explicitly describes graph-visible metadata tensors for
Qwen GDN operations. Reusing `serve-qwen38-native.sh` for GLM would therefore
be a false baseline.

### 2. Quantization math and scale contracts

#### Qwen/v100-skinny NVFP4 W4A16 path

The checkpoint-native codes are two FP4 E2M1 nibbles per byte along K:

```text
codes:       uint8 [N, K/2]
group scales: FP8 E4M3 [N, K/16]
global scale: FP32 scalar, ModelOpt amax / (6 * 448) = amax / 2688
activation:   FP16 (no activation quantization)
```

The intended numerical value is the E2M1 code multiplied by its FP8 group
scale and the multiplicative global scale. `kernels/skinny_kernels.cu` folds
the decoder's `2^-14` representation correction by using `gscale * 16384`;
there is no FP8-to-FP4 requantization. `_qpn_prepack` in `marlin.py` is a
byte/nibble address permutation only: it expands nibbles, applies the
Volta-fragment K order, and repacks. It requires `N % 32 == 0` and `K % 64 ==
0`; chunking limits temporary gather intermediates to roughly 300 MB. The
one-shot lm_head gather would be about 2.4 GiB.

#### QPN2

- Owns NVFP4 decode/verification `M=1..8` when the shape is eligible.
- Uses Volta `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32`, four
  quadpairs on one N=32 tile, split-K 8/16/32, and accumulator count 1/2.
- Shared reduction scratch is FP32. The kernel checks contiguous FP16 input,
  contiguous U8 code/scale buffers, K divisibility, N divisibility, and exact
  payload sizes; ineligible shapes must go to another route.
- There is no `gemm_qpn2_mt2`. NVFP4 `M=9..16` uses legacy `gemm_qpn` and
  NVFP4 `M>=17` goes to Marlin in the shipping memory profile.

#### QPN8

- Owns the Qwen protected FP8 W8A16 regions, not NVFP4.
- Input bytes are FP8 E4M3 `[N,K]`; the prepack uses the same fragment
  permutation. The kernel decodes each byte as a value divided by 256, so
  `_sm70_qpn8_tile_scales` stores one FP32 scale per N=32 tile after
  multiplying the source scale by 256.
- QPN8 supports M<=8; QPN8-MT2 supports M<=16; packed/chunked calls continue
  through M=96; above M=96 the path reconstructs transient FP16 for prefill.
  There is no persistent FP16 copy. QPN8 prepack is inverted and asserted
  byte-identical before the source FP8 storage is released.
- The old reference fallback in `modelopt.py` uses `scales.max()` when logical
  widths and scale counts do not match. That is bring-up-only behavior and
  must **not** be silently reused by a GLM adapter: GLM logical widths,
  expert packing, and scale ownership must be validated exactly; a mismatch
  should fail closed rather than max-scale requantize.

#### Target GLM NVFP4 stores activation-scale metadata; the POC may declare W4A16

NVIDIA's GLM checkpoint serializes 4-bit E2M1 weights, FP8-E4M3 weight scales
per 16-element group, an F32 `weight_scale_2`, and an F32 `input_scale` for
each quantized linear. The native metadata is therefore W4A4-oriented, but a
Volta POC may explicitly choose W4A16 semantics. Under that choice, the
adapter must state how FP16 activations and the serialized `input_scale` are
handled and must pass a real-tensor reference comparison; W4A4 equivalence is
not the acceptance condition.

v100-skinny's SM70 linear hook turns serialized `NVFP4` into its W4A16
Marlin adapter: it keeps the codes and weight scales but discards
`input_scale`. That behavior cannot be silently reused for GLM; the GLM
adapter must make the choice explicit and preserve enough metadata to audit
the numerical contract.

The regular `ModelOptNvFp4Config` still reports minimum capability 75, while
the SM70 lowering is wired for the Qwen mixed/W4A16 route. The custom QPN2
linear op does not process routed expert batches. The GLM target's MoE path is
delegated to vLLM's `select_nvfp4_moe_backend` / `make_nvfp4_moe_kernel`; no
v100-skinny QPN2/Marlin expert implementation is present.

### 3. Resident memory and duplication

For a Qwen W4A16 linear that reaches skinny, load-time state is:

1. checkpoint-native U8 codes and FP8 group scales;
2. a second QPN fragment-order code/scale copy; and
3. a Marlin-repacked weight/scale copy for prefill and `M>=17`.

`VLLM_SKINNY_DROP_CT=1` frees (1) after the QPN prepack, but (2) and (3)
remain. The release therefore has two packed low-bit copies per NVFP4 layer;
it does not have a persistent FP16 model copy. QPN8 keeps one packed FP8 copy
and frees its source only after the invertibility check. Large-M FP8 prefill
uses transient reconstruction; QPN2/Marlin path selection is based on runtime
M, not a static model-wide choice.

For GLM, the 204.44 GB checkpoint alone is not the runtime footprint. Any
attempt to route all 36,297 W4A4 expert/dense linears through the W4A16 skinny
adapter would require activation-contract changes and potentially QPN + Marlin
duplicates over the 190.4 GiB binary weight payload. The BF16 MTP layer,
unquantized attention/routers/shared experts/embeddings/lm_head, expert
dispatch buffers, recurrent KDA state, sparse-attention state, NCCL buffers,
CUDA graph pools, and transient prepack workspaces all consume additional
memory. A 10-GPU 320 GiB or 12-GPU 384 GiB raw HBM total therefore cannot be
called feasible from checkpoint size alone.

The target's model card recommends Blackwell-only execution and a Blackwell
vLLM image; this is a support statement, not a Volta result. A Volta POC
needs a model-loader census and per-rank peak-memory trace before any full
weight load is approved.

### 4. KV, prefill, decode, and state contracts

The skinny patch deliberately ignores a checkpoint FP8-KV directive below
SM80, resolving Qwen's FP8-KV metadata to FP16 KV. README results attribute
about +4.82 ms/round to scalar paged attention when FP8 KV is honored on SM70;
the published Qwen path therefore gates for FP16 KV, XQA/tensor-core decode,
and zero `scalar_paged` calls.

That policy is not enough to make GLM attention work. Qwen's fast metadata
patch is for GDN chain-MTP: it builds `mamba_state_idx`, speculative state
slot selectors, query starts, and state indices for Qwen's recurrent state
contract. GLM's KDA layers and DSA indexer have different state and attention
layouts; its 11 sparse layers use compressed MLA/DSA state, and its model
config requests FP8 KV. The v100-skinny patch has no GLM5Next state adapter,
no DSA Volta validation, and no proof that the 1Cat FlashAttention V100
backend accepts GLM's sparse/KDA metadata.

Prefill and decode also differ materially. QPN2/QPN8 are decode/small-M
weight-stream kernels. Qwen NVFP4 M>=17 uses Marlin, while FP8 M>96 uses a
transient FP16 reconstruction. GLM's W4A4 routed-expert kernels must process
expert-token batches after routing, activation scales, and grouped GEMM; the
existing skinny linear custom op cannot be substituted for that MoE path.
The GLM MTP layer is BF16 and its MTP state/proposer path is not the Qwen GDN
chain path.

### 5. Ten/twelve-GPU topology constraints

The published skinny result is TP4, not TP10 or TP12. The GLM target has
`hidden_size=4096`, 288 routed experts, and 8 experts per token. A blind
`--tensor-parallel-size 10` or `12` is not a safe launch assumption: hidden
and projection partition divisibility, expert parallel partitioning (288 is
divisible by 12 but not 10), and pipeline/expert parallel interactions need
to be checked against the exact runtime/model implementation. The existing
NVIDIA recipe is TP4 + expert parallel on Blackwell, not a recipe for either
V100 topology.

The fleet's Qwen reference smoke test may use the approved idle pair (TP2),
but it must be labeled **non-published** because the pinned launcher and
published numbers are TP4. The GLM POC's “exactly 10 GPUs, then 12” condition
should be applied only after the architecture/parallelism adapter exists and
the coordinator approves UUID visibility. Do not infer a successful 2-GPU
Qwen smoke test as GLM feasibility.

## CHANGES MADE

- Added this research report only:
  `docs/runtime/v100-skinny/FOUNDATION.md`.
- Cloned the upstream and engine sources under `[private path]` for read-only
  inspection; no files in those external checkouts were edited.
- Preserved all pre-existing dirty work in `[private path]`, including
  the existing `docs/runtime/v100-skinny/baseline/bootstrap-20260916.log`,
  `scripts/glm-5.3-flash/bootstrap_baseline.sh`, model review edits, and wiki
  changes. No commit was created.

## TESTS RUN

Read-only checks performed:

- Disk preflight: `[private path]` had approximately 702 GiB free when the clone
  was made; `[private path]` had approximately 1.8 TiB free.
- Git status, exact tag/commit metadata, and clean status for both external
  checkouts.
- `git ls-tree`/source inspection of all v100-skinny runtime, kernel, patch,
  script, README, result, and reproduction files.
- Hugging Face API/model-tree inspection for shard count/size, current model
  revision, safetensors index counts, and safetensors-header range reads for
  representative quantized and BF16 tensors.
- Static `rg` search of 1Cat-vLLM v1.2.2 for `glm5_next`, `Glm5Next`, and
  `GLM-5`: no `glm5_next` implementation exists in the pinned source; the
  generic `GlmMoeDsaForCausalLM` path is separate.

Not run in this source-analysis slice:

- `bootstrap-sm70.sh`, pip/conda installs, CUDA JIT compilation, model
  downloads, `nvidia-smi` allocation, or any GPU kernel/serving benchmark.
- Any GLM load or launch. There is no claim that GLM currently boots on V100.

## RESULT

The published v100-skinny result has an identified **Qwen3.8-27B TP4
reference recipe** (not locally reproduced in this source-analysis slice),
subject to an isolated Python 3.12 environment, a torch build
whose runtime reports `sm_70`, CUDA 12.8 `nvcc`, the digest-pinned 1Cat wheel,
and the paired 0.1.10 TileLang/tvm-ffi override. It is not a GLM-5.3-Flash
runtime. The exact upstream reproduction is:

```bash
cd [private path]
export ENV_PREFIX=[private path]
# Use a fresh isolated Python 3.12 environment. The upstream script installs
# the digest-pinned wheel and the required tilelang/tvm-ffi overrides.
bash scripts/bootstrap-sm70.sh

hf download RadixArk/Qwen3.8-27B-NVFP4 \
  --revision 554ebba9b5f1b79dc11246341960360e6ef05ef4 \
  --local-dir [...]/Qwen3.8-27B-NVFP4

# Published configuration is TP4. Override CUDA_VISIBLE_DEVICES only after
# fleet approval; TP2 is a smoke test and not a published result.
bash scripts/serve-qwen38-native.sh \
  [...]/Qwen3.8-27B-NVFP4
```

Before allowing the bootstrap to JIT-build, the isolated environment should
be gated with:

```bash
ENV_PREFIX=[private path]
"$ENV_PREFIX/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available()
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('arch', torch.cuda.get_arch_list())
assert 'sm_70' in torch.cuda.get_arch_list()
x = torch.ones((8, 8), device='cuda', dtype=torch.float16)
assert torch.isfinite(x @ x).all().item()
print('sm70 fp16 smoke: PASS')
PY
```

The exact GLM checkpoint has no known-good minimal V100 model recipe in the
inspected sources. The smallest honest GLM prerequisite is metadata-only
validation followed by a loader/architecture smoke test against a single
representative dense NVFP4 linear and a routed-expert grouped GEMM under the
declared W4A16 semantics; a full 204 GB download should wait for that adapter
and a per-rank memory plan.

## OPEN ISSUES

1. **Architecture gap (blocking):** pinned 1Cat-vLLM has no `glm5_next` model
   class, while NVIDIA Flash is `Glm5NextForConditionalGeneration`.
2. **Quantization gap (blocking):** the checkpoint carries W4A4-oriented
   activation-scale metadata, while skinny QPN2/Marlin is W4A16 and
   intentionally discards `input_scale`. A W4A16 POC is allowed, but only with
   an explicit scale treatment and a real-tensor accuracy reference; it must
   not claim W4A4 equivalence.
3. **MoE gap (blocking):** 36,288 routed-expert linears are the dominant
   target weight set; skinny QPN2/QPN8 custom linear dispatch does not cover
   routed expert grouped GEMM. Existing ModelOpt MoE selection is not a
   validated SM70 path for this target.
4. **Attention/state gap (blocking):** Qwen GDN metadata and FlashAttention
   V100 assumptions do not establish GLM KDA/DSA support. FP8-KV metadata must
   be deliberately converted to a validated Volta storage policy, not merely
   inherited from Qwen.
5. **MTP accounting/deferment:** target layer 45 is BF16 and has a different
   model/state contract; Qwen chain-MTP fast metadata cannot be assumed. MTP
   is optional for the initial text POC and may be explicitly deferred, but
   its tensors and the deferment must be recorded in the loader census.
6. **Scale-contract gap:** any GLM adapter must reject logical-width/scale
   mismatches. It must not use the existing reference `scales.max()` fallback
   or silently requantize expert weights.
7. **Parallelism gap:** TP10/TP12 divisibility and EP/PP layout have not been
   proven; raw HBM arithmetic is insufficient. The 10-GPU requirement is a
   future POC constraint, not a current launch recipe.
8. **Baseline environment gate:** the fleet bootstrap must verify the isolated
   torch build's actual `sm_70` arch list and tiny FP16 CUDA op before JIT or
   model work. Host Python 3.13 / torch 2.12.1+cu130 is not an accepted
   baseline.
9. **Storage/download:** the NVIDIA Flash artifact needs about 204.44 GB of
   shard storage before cache/temp overhead. The generic RadixArk artifact is
   about 464.87 GB and should not be selected accidentally.

## RECOMMENDATION

1. Complete and gate the isolated Qwen TP4 reference environment. If only the
   idle TP2 pair is available, run a clearly labeled smoke test and preserve
   the published TP4 claim as untested on this host.
2. Treat GLM-5.3-Flash on V100 as **blocked pending a separate GLM5Next +
   validated W4A16-semantics MoE/attention adapter** (or a separately chosen
   native W4A4 path), not as a checkpoint swap. Acceptance is a real-tensor
   reference under the declared semantics, not W4A4 equivalence. Keep the
   exact NVIDIA revision pinned before any 204 GB download.
3. Do not enable lm_head requantization or borrow lm_head data from another
   checkpoint. The target lm_head is excluded from quantization and must stay
   on its own published storage/loader path.
4. Require a loader census that proves, per module and per expert: tensor
   shape, activation-scale use, weight-scale ownership, route/backend, and
   peak resident bytes. Fail closed on missing scales, width mismatch, or
   fallback backend.
5. Only after the single-module and attention-state gates pass should the
   coordinator evaluate 10-GPU and 12-GPU parallel layouts and full model
   storage. A successful Qwen baseline is evidence that the Volta kernel stack
   works, not evidence that GLM fits or computes correctly.
