# The two new architectures, in plain terms (MiniMax-M3 vs DeepSeek-V4-Flash)

> **Origin.** PLI Labs research archive, file `v100-research/wiki/05-new-archs-msa-vs-csa.md`. Published 2026-11-18 in the Volta V100 research
> repository.
> **Edited for publication.** Machine names, private file paths, network addresses, session names and the
> operator's name were replaced with generic terms (`the V100 server`, `[private path]`, `[address]`,
> `the owner`). Links to files that are not published were turned into plain text. No figure and no finding
> was changed. Dates inside the note are its own; later results are in the build pages of this repository.

*Added 2026-07-07. Why one of these is easy on Volta and the other is a research project.*

## MiniMax-M3 (~428B total / ~23B active)
Think "DeepSeek-V3-style MoE body with a new attention hat." The MoE part is ordinary:
128 experts, pick 4 per token (sigmoid gate + bias correction), plus one always-on shared
expert — llama.cpp already knows how to run this shape on any CUDA GPU, including sm_70.

The hat is **MSA — MiniMax Sparse Attention**. Every 128 consecutive tokens form a "block."
A lightweight **index branch** (the "router tech") scores which 16 blocks matter for the
current query, and the main branch then runs *normal, exact* softmax attention over just those
16×128 = 2,048 tokens instead of the whole million-token history. Key property: the main
branch is standard attention. If you skip the router and attend to *everything* (dense
fallback), you get the same kind of answer — you just pay O(n²) compute. That's why
llama.cpp PR #24523 can serve M3 correctly on a V100 today with MSA simply turned off, and
why "make V100s compatible with M3's router tech" = writing one new fp16 top-k-blocks +
block-gather kernel family, not re-architecting attention.

What Volta loses: the 1M-context economics (dense at 1M is hopeless anywhere), MTP/EAGLE3
speculative decode (dropped in GGUF), and the vision tower. What it keeps: full answer quality
at practical contexts (≤64K on our KV budget: dense f16 KV ≈ 120 KB/token).

## DeepSeek-V4-Flash (284B total / 13B active)
Here the compression **is** the architecture — there is no "just run it dense" escape hatch.
Three mechanisms stack: (1) MLA-style latent KV (one 512-dim latent per token instead of
64 heads — the KV cache is tiny, ~50 KB/token upper bound); (2) **CSA** — a lightning indexer
(64 small heads) picks the top-512 relevant tokens from a 4:1-compressed history; (3) **HCA** —
a 128:1 heavily-compressed summary stream plus a 128-token sliding window for local detail.
Residuals flow through "manifold-constrained hyper-connections" (mHC) instead of plain adds.
Roughly five custom GPU operators implement this; upstream llama.cpp merged them (PR #24162,
June 29 2026) as generic CUDA, and the cchuter fork carries hand-fused versions (including
explicit SM_70 software-fallback paths, in our `[...]/llama.cpp-v4port` checkout).

Nobody on record has run V4-Flash on a Volta card. Every "fast" stack (vLLM, SGLang,
ktransformers, TensorRT-LLM, NVFP4/W4A16 quants) hard-requires sm_80+ — even A100 fails on
some paths. Our experiment: the merged generic-op path compiled fine for sm_70 on 07-02; the
first post-reboot smoke (runbook step 1) decides whether V100 serving is real. Upside if it
works: MLA's tiny KV means 8×32 GB holds the Q4 weights (163 GB) *and* very long contexts —
the opposite of M3's KV economics.

## Cheat sheet
| | M3 | V4-Flash |
|---|---|---|
| Sparse attn | MSA (block router + exact softmax) | CSA/HCA (compression built-in) |
| Dense fallback | ✅ correct, merged, V100-tested | ❌ doesn't exist |
| KV per token | ~120 KB (GQA, dense) | ~≤50 KB (MLA latent) |
| 4-bit weights | ~257 GB (12+ cards) — Q2 = 143 GB (8 cards) | ~163 GB (8 cards, roomy) |
| V100 status | works today (dense) | unproven; we are the test |
| Patch project | MSA index branch on sm_70 (moderate) | port fork's fused kernels (only if generic path works-but-slow) |
| License | MiniMax Community (not MIT) | MIT |
