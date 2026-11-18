# The source research

The notes the builds were planned from. They are working documents: dated, written in the middle of the work, and
sometimes superseded by a later measurement. Where a note and a build page disagree, the build page is later and
governs. Start with the three files marked **start here**.

Every file below is PLI Labs' own note, copied from our private research archive. Each says at its top where it
came from and what was changed for publication: machine names, private file paths, network addresses, session names
and the operator's name were replaced with generic terms; links to unpublished files became plain text; a few
lines that named the author of a note or quoted an instruction were shortened. No figure and no finding was
changed. Third-party work is summarised and linked in these notes, never copied: follow the
links to the originals.

## Volta architecture and the kernel rules

| File | What it holds | Date |
|---|---|---|
| [kernels/KERNEL-RESEARCH-EXPLAINER.md](kernels/KERNEL-RESEARCH-EXPLAINER.md) | **Start here.** What a GPU kernel is, why decode is limited by memory and hand-offs, how to build one well, a glossary. Plain terms | 2026-11-02 |
| [kernels/KERNEL-RESEARCH-CLAUDE-NOTES.md](kernels/KERNEL-RESEARCH-CLAUDE-NOTES.md) | **Start here.** Twelve binding rules for anyone (or any coding agent) writing a kernel for sm_70 or the DGX Spark, each with its source | 2026-11-02 |
| [kernels/KERNEL-RESEARCH-REPORT.md](kernels/KERNEL-RESEARCH-REPORT.md) | **Start here.** The full survey: the roofline, memory hierarchy, landmark kernels, hardware generations from Volta to Blackwell, AI-written kernels, and its reference list | 2026-11-02 |
| [kernels/CUSTOM-KERNEL-OPPORTUNITIES-20261109.md](kernels/CUSTOM-KERNEL-OPPORTUNITIES-20261109.md) | The ranked list of kernel opportunities written for the eight-card NVFP4 build | 2026-11-09 |
| [kernels/KOPT1105-ROOFLINE.md](kernels/KOPT1105-ROOFLINE.md) | A CPU-only roofline and census analysis of the prefill expert kernel: where padded work comes from | 2026-11-05 |
| [kernels/KOPT1105-UPSTREAM.md](kernels/KOPT1105-UPSTREAM.md) | What upstream offered for the same kernel | 2026-11-05 |
| [kernels/UPSWEEP-ATTN-PRIOR-ART.md](kernels/UPSWEEP-ATTN-PRIOR-ART.md) | Prior art for attention kernels on sm_70: what exists upstream and what it assumes | 2026-11-01 |
| [volta/v100-skinny/FOUNDATION.md](volta/v100-skinny/FOUNDATION.md) | Our review of [v100-skinny](https://github.com/dnv2003/v100-skinny) (Dennis Vertzyas' kernels) as a foundation: what it proves and what it leaves open | 2026-09-16 |
| [volta/wiki/05-new-archs-msa-vs-csa.md](volta/wiki/05-new-archs-msa-vs-csa.md) | Two newer attention designs in plain terms, and why one is easy on Volta | 2026-07-07 |

## Where the time goes (measured)

| File | What it holds | Date |
|---|---|---|
| [kernels/PROF1103-REPORT.md](kernels/PROF1103-REPORT.md) | The first profile of a decode round on eight cards: 32 ms split by stage and kind of work, and why one upstream pull request did not transfer | 2026-11-03 |
| [kernels/PROF1103b-REPORT.md](kernels/PROF1103b-REPORT.md) | The second profile, after the first levers | 2026-11-03 |
| [kernels/PROF1-report.md](kernels/PROF1-report.md) | The earlier profile of the EXL3 engine's round | 2026-11-01 |
| [kernels/TOK1103-REPORT.md](kernels/TOK1103-REPORT.md) | Tokens-per-second levers found in v100-skinny and 1Cat-vLLM, with upstream numbers and our estimates | 2026-11-03 |
| [LANE-RESEARCH.md in Volta Max uPLIft](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4/blob/main/docs/research/LANE-RESEARCH.md) | Where the time of a decode round goes on the eight-card NVFP4 build. Published in that build's repository; not copied here | 2026-11-09 |
| [eight-card/topk-full-20261110.md](eight-card/topk-full-20261110.md) | The full cost of the index top-k kernels per decode round, measured on one card of a four-card test machine | 2026-11-10 |

The twelve-card time budget is [in that build's repository](https://github.com/ProprietaryLegal/Volta-12X-AmPLIfy-DeepSeek-V4.1-Flash-1017-NVFP4-12xV100-SXM2-PP3xTP4/blob/main/research/TWELVE-CARD-TIME-BUDGET.md) and
summarised in [../optimizations/volta-12x-amplify.md](../optimizations/volta-12x-amplify.md#where-the-time-goes).

## Upstream sweeps

| File | What it holds | Date |
|---|---|---|
| [kernels/UPSCAN1105-REPORT.md](kernels/UPSCAN1105-REPORT.md) | A read-only sweep of public work for ideas that port to this build, ranked | 2026-11-05 |
| [kernels/UPSCAN1105-ONECAT.md](kernels/UPSCAN1105-ONECAT.md) | The same for 1Cat-vLLM alone | 2026-11-05 |
| [kernels/UPSCAN1105-SUPPLEMENT.md](kernels/UPSCAN1105-SUPPLEMENT.md) | Its supplement | 2026-11-05 |
| [volta/findings/V-C-speedups-untried.md](volta/findings/V-C-speedups-untried.md) | Speed-ups not yet tried on V100, June 2026: an early survey, written before the DeepSeek work. Read it for the options, not for the numbers | 2026-06 |
| [volta/findings/V-D-community-china.md](volta/findings/V-D-community-china.md) | What the wider community had measured on V100s, including Chinese-language sources, June 2026 | 2026-06 |

## The first DeepSeek-V4.1 port (October 2026, behind 1Cat-vLLM #1031)

The record of the first twelve-card port: three stages of four cards, the official checkpoint, no CUDA graphs. It
ended at about 3 tok/s and an overall FAIL of its exactness gates, and it is direct prior art for
[Volta 12X AmPLIfy](../optimizations/volta-12x-amplify.md).

| File | What it holds |
|---|---|
| [volta/deepseek-v4.1-flash/PORT_DESIGN.md](volta/deepseek-v4.1-flash/PORT_DESIGN.md) | The design of the port: package layout, contracts between parts, the stage cuts, the test plan |
| [volta/deepseek-v4.1-flash/MODEL_ARCHAEOLOGY.md](volta/deepseek-v4.1-flash/MODEL_ARCHAEOLOGY.md) | The model read from its reference code: what every part does |
| [volta/deepseek-v4.1-flash/V4_TO_V41_MODEL_DIFF.md](volta/deepseek-v4.1-flash/V4_TO_V41_MODEL_DIFF.md) | What changed from DeepSeek-V4 to V4.1 |
| [volta/deepseek-v4.1-flash/1CAT_ARCHAEOLOGY.md](volta/deepseek-v4.1-flash/1CAT_ARCHAEOLOGY.md) | 1Cat-vLLM read as a codebase: what it had for DeepSeek-V4 and what V4.1 needed |
| [volta/deepseek-v4.1-flash/TENSOR_CENSUS.md](volta/deepseek-v4.1-flash/TENSOR_CENSUS.md) | Every tensor of the checkpoint by kind, format and size |
| [volta/deepseek-v4.1-flash/ENGRAM.md](volta/deepseek-v4.1-flash/ENGRAM.md) | The Engram n-gram memory: tables, hashing, how it is served |
| [volta/deepseek-v4.1-flash/CHECKPOINT_CANDIDATES.md](volta/deepseek-v4.1-flash/CHECKPOINT_CANDIDATES.md) | Which published checkpoints could be served on Volta, and why |
| [volta/deepseek-v4.1-flash/RANGE_AUDIT.md](volta/deepseek-v4.1-flash/RANGE_AUDIT.md) | The numeric ranges of activations, for FP16 safety |
| [volta/deepseek-v4.1-flash/NLL_GAP.md](volta/deepseek-v4.1-flash/NLL_GAP.md) | How much of a quality gap came from the port's precision choices (answer: almost none) |
| [volta/deepseek-v4.1-flash/E2E_FINAL.md](volta/deepseek-v4.1-flash/E2E_FINAL.md) | The final end-to-end campaign and its FAIL. Three raw API responses were removed: generated model text is not published here |

## Not copied here, and where to find it

| Material | Why not | Where |
|---|---|---|
| The v100-skinny kernels, benchmarks and results | Dennis Vertzyas' work | <https://github.com/dnv2003/v100-skinny> |
| The 1Cat-vLLM SM70 kernels and pull-request texts | their authors' work | <https://github.com/1CatAI/1Cat-vLLM>; every pull request we used is linked in [../CREDITS.md](../CREDITS.md) |
| Our production kernels (`v41k`, the index scorers, the prefill expert GEMM) | already public as patches | [Volta sPLIt, engine/production](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/tree/main/engine/production) |
| The RB 4 prompt-reading kernel | already public as source | [Volta Max uPLIft, engine/kernels/prefill-rb4](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4/tree/main/engine/kernels/prefill-rb4) |
| The plans of the eight-card NVFP4 build | already public | [Volta Max uPLIft, docs/plans](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4/tree/main/docs/plans) |
| The lever log of the eight-card build | already public | [Volta sPLIt, docs/levers.md](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/blob/main/docs/levers.md) |
| The June 2026 notes on the serving stack and on serving larger models | they describe one machine's inventory in detail; their conclusions were published at the time | [pli-labs, research/v100](https://github.com/ProprietaryLegal/pli-labs/tree/main/research/v100) |
| The long working logs of individual kernel experiments, the kernel campaign on the test machine, the hardware inventory | working logs tied to one machine. Their results are in the build pages | [../optimizations/](../optimizations/README.md) |
| The twelve-card build's measurements, history, kernel plan and time budget | published in the build's own repository | [Volta 12X AmPLIfy](https://github.com/ProprietaryLegal/Volta-12X-AmPLIfy-DeepSeek-V4.1-Flash-1017-NVFP4-12xV100-SXM2-PP3xTP4); summarised in [../optimizations/volta-12x-amplify.md](../optimizations/volta-12x-amplify.md) |
