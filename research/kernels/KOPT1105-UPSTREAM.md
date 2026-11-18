# KOPT1105 upstream comparison

> **Origin.** PLI Labs research archive, file `kernel-research/KOPT1105-UPSTREAM.md`. Published 2026-11-18 in the Volta V100 research
> repository.
> **Edited for publication.** Machine names, private file paths, network addresses, session names and the
> operator's name were replaced with generic terms (`the V100 server`, `[private path]`, `[address]`,
> `the owner`). Links to files that are not published were turned into plain text. No figure and no finding
> was changed. Dates inside the note are its own; later results are in the build pages of this repository.

Inspected 2026-11-05 UTC, CPU-only. No GPU work, server calls, heldout reads or production edits. This is source analysis, not performance qualification. The sources below support mechanisms; none supplies a measured incremental D51 speedup.

## Pins and source scope

| Source | Exact revision / evidence |
|---|---|
| D51R5 packaged kernel | `evidence/commit.txt` says `8f8e60223`; actual `kit/csrc/prefill_iq.cu` SHA256 `2da1b35180a279424a43224cebad53dccd909fdbfa257c3c37b7c2a33e998810`. Headers are the packaged dependency closure; do not assume an unmodified upstream checkout. |
| philbert local read-only clone | `23cec6a9fcd1d8face54c178fced8b4300628ae5`; the two inspected kernel/adapter files have no diff against HEAD. Root `[private path]`. |
| 1Cat PR1277 | API verifies merged revision `899e9ed3319a776416fc7042c6a9ece188a67b64`; PR head `a472a3277e229f13e7f080c0234de093d0d3e955`. |
| llama.cpp | GitHub `master` resolved to `3de52ac38818978d6da8f386eda43a856e53714a`; raw files read at this SHA. |
| ik_llama.cpp | GitHub `main` resolved to `009dce19afdd3f1e4b5039bcd8853256726c322b`; raw files read at this SHA. |
| v100-skinny | GitHub `main` resolved to `5b589c0dc81223e0ba65bcb3e755874723f8b515`; raw kernel read at this SHA. |

Production `src-pp2` is a source snapshot without Git metadata available through its NFS path. This report does not assign it an invented upstream HEAD.

## What D51 already does

Relative paths in this paragraph start at `[private path]`.

- `prefill_iq.cu:23` builds compact member tiles; `:77` selects canonical lattice decode and BK32, two stages, four/eight warps; `:88` executes fixed-grid or flattened scheduling; `:120` launches the planner separately. This already replaces the original coarse grouped scheduler.
- `upstream/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/mainloop_sm70.h:339–376` decodes a B fragment **outside** the inner M loop and reuses it across that warp's M iterations. A claim that baseline decodes once for every scalar output row would be false. The remaining possibility is to share decoded B across different M warps or CTAs.
- `mainloop_sm70.h:230` initializes the codebook each mainloop invocation. Persistent iteration calls the kernel again, so initialization repeats. Hoisting is plausible only if codebook lifetime is separated from epilogue/mainloop union reuse; removing the initialization without proving lifetime is unsafe.
- `lattice_transform.h:18–23` initializes shared codebook words. `:43–84` loads aligned grid words, unpacks half2 pairs, applies signs/delta and FP16 scale. Already optimized word-level decode should be the comparison baseline. Replacing FP16 steps with one FP32 multiply followed by rounding can change operands.

## Transferable designs and boundaries

### Philbert EXL3 prefill

[Kernel at the inspected pin](https://github.com/philbert440/deepseek-v4.1-v100/blob/23cec6a9fcd1d8face54c178fced8b4300628ae5/csrc/sm70_turbomind/ops/exl3_moe_sm70.cu#L768): lines 768–925 describe and implement row tiers 16/32/64; decode one B fragment per K tile, reuse over row groups; configurable P-slot register load ring; direct m8n8k4 FP32 accumulation. Its loop refills a packed-weight slot immediately after decode, before the MMA consuming decoded registers. This is useful evidence for software load/decode/MMA pipelining on SM70 without cp.async. D51 already has a staged mainloop, so compare instruction ordering and dependency chains before adding buffers.

The EXL3 trellis/window decoder, sm80 storage order and Hadamard-related preprocessing are not IQ lattice-compatible. Do not copy its decoder or claim its published absolute timings predict D51 gains. Its M-tier scheduler and scheduling of packed loads are transferable design patterns. K-split or independent accumulator chains alter summation order even with FP32 accumulators.

[Python adapter](https://github.com/philbert440/deepseek-v4.1-v100/blob/23cec6a9fcd1d8face54c178fced8b4300628ae5/vllm/models/deepseek_v41/sm70/moe_exl3.py#L355): lines 355–424 construct expert-sorted tiles, choose tiers, and reconstruct dense weights for sufficiently hot experts before cuBLAS. `:384` calls `info.cpu()` outside capture; captured execution uses capacity bounds. Do not transplant this host synchronization into D51. A hot-expert FP16 cache requires a device-side policy or pre-frozen shape policy, explicit memory budget and separate measurement of preparation cost. The microbench defaults dense threshold 160 (`scripts/ds41_moe_prefill_microbench.py:81`); that is an EXL3 benchmark setting, not a justified IQ crossover threshold.

### llama.cpp IQ2_XXS MMQ

[MMQ kernel](https://github.com/ggml-org/llama.cpp/blob/3de52ac38818978d6da8f386eda43a856e53714a/ggml/src/ggml-cuda/mmq.cuh#L1000) uses expert boundaries and indexed destination rows with regular tiling or stream-K/fixup. Transfer these ideas for a small-M/tail workload only after showing wave imbalance; D51's 8K hot shapes may already have ample CTAs. Stream-K adds partial buffers and a reduction, changes summation order and needs the numerical gate.

The latest fused gate/up/GLU support at `:438–439` includes Q4_K/Q5_K/Q6_K/Q8_0/IQ4_XS, **not IQ2_XXS**. `:1010–1018` requires Turing MMA, and `:923` disallows fused split-K. It is architectural evidence for paired gate/up fragments, not an available Volta IQ2 kernel.

[MMQ dispatch](https://github.com/ggml-org/llama.cpp/blob/3de52ac38818978d6da8f386eda43a856e53714a/ggml/src/ggml-cuda/mmq.cu#L311) quantizes broadcast gate/up activations once per token and scatters to expert slots (`:315–359`). D51 uses canonical FP16 activations; an analogous FP16 gather-once scheme can trade one copy for fewer indexed loads, but copying Q8_1 arithmetic would introduce activation quantization and require KL/near-tie checks. It is not a bitwise refactor.

### ik_llama.cpp indexed MMQ

[Indexed loader and dispatcher](https://github.com/ikawrakow/ik_llama.cpp/blob/009dce19afdd3f1e4b5039bcd8853256726c322b/ggml/src/ggml-cuda/mmq_id_common.cuh#L2861) unpacks IQ2_XXS grid/signs into a shared integer tile; lines 2861–2920 include padded layouts, and `:3731–3736` bind it to Q8_1 products. SM70 takes the DP4A branch; this is not a Volta FP16 Tensor Core replacement. Shared decoded-B reuse is relevant, while integer activation arithmetic is a numerically different path.

Its dispatch at `:4421–4453` contrasts ordinary tiling with a stream-K grid selected using an estimated wave efficiency and allocates fixup space when needed. This is a usable scheduler design reference, not evidence stream-K wins the D51 shapes.

[ID construction](https://github.com/ikawrakow/ik_llama.cpp/blob/009dce19afdd3f1e4b5039bcd8853256726c322b/ggml/src/ggml-cuda/mmq_id.cu#L28) creates source/destination permutations and expert bounds; specialized top-k layouts include six experts (`:69`). A D51 planning fusion can share expert counts/member offsets without changing route ordering, but must preserve duplicate token routes and distinguish GU indexed inputs from DN contiguous rows. Test all real histograms, empty experts, tails and duplicate tokens.

### v100-skinny QPN2/QPN8

[QPN2 kernel](https://github.com/dnv2003/v100-skinny/blob/5b589c0dc81223e0ba65bcb3e755874723f8b515/kernels/skinny_kernels.cu#L1385) uses prepacked Volta fragment order, compile-time K partitioning and one/two accumulator chains. Lines 1421–1439 load packed weights and activation vectors directly into registers before HMMA; `:1443–1472` combine chains/warp partials. These mechanisms justify testing layout simplification and independent accumulator scheduling, with the numerical gate mandatory for reordered sums. QPN2's wrapper accepts only M1–8 (`:1485`); QPN8 MT2 allows M1–16 (`:1783`). Their measured bandwidth and their NVFP4/FP8 codecs do not transfer to D51 IQ2_XXS prefill. A tiny-expert specialization could reuse the geometry concept while retaining D51's global M>6 dispatch boundary.

## 1Cat overlap check

Queried primary GitHub PR API for [1021](https://github.com/1CatAI/1Cat-vLLM/pull/1021), [1030](https://github.com/1CatAI/1Cat-vLLM/pull/1030), [1031](https://github.com/1CatAI/1Cat-vLLM/pull/1031), [1243](https://github.com/1CatAI/1Cat-vLLM/pull/1243), [1277](https://github.com/1CatAI/1Cat-vLLM/pull/1277), [344](https://github.com/1CatAI/1Cat-vLLM/pull/344). The first four are open at inspection; #1277 and #344 are closed. #1021 is FP8-resident block-FP8 dense, #1030 geometry/generalized skinny split-K, #1031 DS-V4.1 PP deployments, #1243 EXL3/graph/DSpark, #1277 merged DS-V4.1 package. #344's actual title is “Gate PP2 TP4 fast paths on quality”; older memory's shorthand is not its title. No independent IQ2_XXS prefill speed claim is inferred from these PR titles.

## Priority implications

1. First improve existing lattice-mainloop scheduling/layout and adaptive M-tail scheduling; retain operands and reduction order wherever possible.
2. Investigate codebook initialization lifetime and duplicate cross-warp decode with SASS/resource evidence. D51 already does intra-warp B reuse.
3. Trial a hot-expert decode-once FP16 path only with preparation, memory and synchronization costs included.
4. Treat stream-K, independent accumulators and fused activation as numerical changes, even if each uses FP32 internally.
5. Upstream integer MMQ, NVFP4/QPN and Turing GLU fusion are comparison designs; none is a drop-in D51 replacement. No GPU tests were performed here. GPU work runs after the benches and GitHub updates complete.
