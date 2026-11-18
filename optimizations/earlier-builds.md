# The earlier V100 builds: what each one found

These came before Volta sPLIt. Each is a public repository with its own measurements; this page lists what each
contributed to the later builds, with the number and the link. Nothing here was measured for this page, and each
row's conditions are those of its source.

## DeepSeek-V4-Flash on eight V100s with llama.cpp (July 2026)

Repository: <https://github.com/ProprietaryLegal/deepseek-v4-flash-v100>

| Finding | Number (MEASURED, eight V100-SXM2-32GB, 245 W per card, llama.cpp layer split) | Source |
|---|---|---|
| The model runs, slowly | 12.2 tok/s generation, 209 tok/s prompt reading at 8K, 4-bit GGUF | [README, headline numbers](https://github.com/ProprietaryLegal/deepseek-v4-flash-v100#headline-numbers) |
| Batch settings `-b 2048 -ub 1024` | +5 to 6 % generation over the initial settings | same |
| **A quantized key cache silently corrupts output** | With `--cache-type-k q8_0` the model loads, reports healthy and writes confident nonsense, with no error. First blamed on a Volta kernel; that was disproved (it reproduces on a CPU). The cause is a rotation that llama.cpp enables for quantized key caches, which diverts this architecture off its sparse attention path. Fix: an FP16 key cache, or the published patches | [README, "The bug you need to know about"](https://github.com/ProprietaryLegal/deepseek-v4-flash-v100#the-bug-you-need-to-know-about--updated-it-affects-every-backend-not-just-volta); upstream issue [llama.cpp #25382](https://github.com/ggml-org/llama.cpp/issues/25382) |
| More than eight cards does not help with a layer split | ten cards about equal to eight (11.5 against 12.2 tok/s); fourteen cards **halve** throughput (7.8 tok/s) once the split spans four NVLink islands joined by PCIe | [README, "Quickstart"](https://github.com/ProprietaryLegal/deepseek-v4-flash-v100#quickstart) |

What it taught the later builds: a model can look healthy while being wrong, so every later change carries an
output check; and adding boards without a layout that respects PCIe makes things slower.

## GLM-5.3-Flash NVFP4 on eight V100s with 1Cat-vLLM (September 2026)

The first fast large-model build on these boards, and the one that set the working rule for everything after it.
Record: PLI Labs' research archive (the launch note is not published; the
[1Cat-vLLM archaeology note](../research/volta/deepseek-v4.1-flash/1CAT_ARCHAEOLOGY.md) cites it). Engine and
recipe: 1Cat-vLLM's own GLM route, by its authors. MEASURED 2026-09-17 to 09-23 on eight V100-SXM2-32GB.

| Finding | Number | Note |
|---|---|---|
| A runtime written from scratch against the upstream recipe | 1.13 tok/s from our own native runtime; about 49 tok/s from the upstream 1Cat-vLLM route on the same eight cards, with no speculative drafter | **Launch the proven upstream recipe first.** Days had gone into testing the hand-built path. UPSTREAM reference for the same model: 53.0 tok/s ([1Cat-vLLM #402](https://github.com/1CatAI/1Cat-vLLM/pull/402)) |
| With the DFlash2 speculative drafter, seven drafts per round, two stages of four cards | 78.8 to 113.8 tok/s on short prompts; 75.0 tok/s after a 147K-token prompt, 67.9 after 186K; prefill 300 to 434 tok/s | context 204,800 tokens, about 197K usable; needles 5 of 5 at 147K |
| The same model as one tensor-parallel group of eight, no pipeline | 60.5 to 141.2 tok/s across five workloads: 20 to 28 % more decode and about four times the prefill of two stages of four | For this model the single group won, because a pipeline with one request leaves a stage idle half the time. For DeepSeek-V4.1 the same layout lost ([measured later](volta-12x-amplify.md#rejected)). **A layout result belongs to a model**; measure it again |
| A card fault | an NVLink error (Xid 74) took two cards out of service on 2026-09-22 | the build moved to the remaining boards |

## Qwen3.8-27B on four-card boards (September and October 2026)

A 27B model fits one board, so it runs as plain tensor parallel 4 with a speculative drafter on the stock 1Cat-vLLM
wheel. Variants run in these weeks: the NVFP4 build with multi-token prediction, two servers on one four-card
board (two cards each), and Swift 1.5, a fine-tune in the same format. The published measurement is
[swift-1.5-v100](https://github.com/ProprietaryLegal/swift-1.5-v100): 247.7 tok/s decode at a 12K prompt on four
cards. In the founder's words at the time
([comment, 2026-10-31](https://www.reddit.com/r/LocalLLaMA/comments/1xk20x5/lawyer_with_too_many_gpus_part_3_deepseekv41flash/19ry1nqtiytqb/)):
"I am able to get like 200+ tok/s decode on qwen 3.8 across 2 cards at full context, and 260 tok/s across 4 cards."

## DeepSeek-V4.1-Flash, the first twelve-card port (October 2026)

Pull request: [1Cat-vLLM #1031](https://github.com/1CatAI/1Cat-vLLM/pull/1031) · record:
<https://huggingface.co/ProprietaryLegal/deepseek-v4.1-flash-sm70>

| Finding | Number | Source |
|---|---|---|
| The official checkpoint (FP8 dense layers, MXFP4 experts, Engram) runs on twelve V100s in three stages of four, or on eight with 120 experts per layer in pinned host RAM | UPSTREAM (the pull request): 2.8 to 3.9 tok/s. No CUDA graphs: limited by the CPU and by kernel launches | [#1031](https://github.com/1CatAI/1Cat-vLLM/pull/1031) |
| Only some stage cuts are legal: layer 20 for two stages; layers 14 and 28 for three | - | [#1031](https://github.com/1CatAI/1Cat-vLLM/pull/1031) |
| Block-FP8 dense weights can stay FP8 on Volta | UPSTREAM: dense bytes 63.5 to 33.7 MiB per layer set; 115.2 to 74.2 microseconds at one row (-36 %); a loss above 16 rows (+13 % at 32, +23 % at 64) | [#1021](https://github.com/1CatAI/1Cat-vLLM/pull/1021), merged 2026-11-16 |
| The MXFP4 expert kernel generalised to this model's geometry (384 experts, 6 active; split sizes 4, 6, 9, 12, 18) | - | [#1030](https://github.com/1CatAI/1Cat-vLLM/pull/1030), open |

What it taught: the layout was right and the engine was not yet fast, because it had no CUDA graphs. Other
people's engine work on the same fork supplied what was missing: CUDA graphs and the speculative drafter for this
model ([#1243](https://github.com/1CatAI/1Cat-vLLM/pull/1243), rivetphilbot) and the complete four-card package
with packed key-value storage ([#1277](https://github.com/1CatAI/1Cat-vLLM/pull/1277), yangzhuxinyzx).

## DeepSeek-V4.1-Flash on the EXL3 engine (2026-11-01)

Repository: [Volta sPLIt, `previous-build/`](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/tree/main/previous-build)
and the interim report in [deepseek-v4-flash-v100](https://github.com/ProprietaryLegal/deepseek-v4-flash-v100#deepseek-v41-flash-1017-on-8-v100-with-1cat-vllm-results-so-far-2026-11-01).
The engine and the EXL3 pack are Phil Tompkins' ([philbert440/deepseek-v4.1-v100](https://github.com/philbert440/deepseek-v4.1-v100)).

| Finding | Number (MEASURED, eight V100-SXM2-32GB, 180 W, 1,425 MHz ceiling, its own method) | Source |
|---|---|---|
| Decode | 42 to 46 tok/s, 31.4 with thinking on; 71.2 to 71.6 ms per verify round | [Volta sPLIt README, the progression](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4#the-progression-in-toks) |
| Engram tables resident in RAM instead of read from disk | cold first token on a 75K-token prompt 46.00 to 26.94 s (-41.4 %); warm first token unchanged; three concurrent 300K prompts 476 to 368 s; 16 of 16 greedy outputs byte-identical | [interim report](https://github.com/ProprietaryLegal/deepseek-v4-flash-v100#deepseek-v41-flash-1017-on-8-v100-with-1cat-vllm-results-so-far-2026-11-01) |
| A shared-memory-staged prefill kernel for the EXL3 experts | -7.5 to -13.2 % per tiled pass on one card, bit-identical. Kernel level only; no server measurement | same |
| Hardware faults found on the way | PCIe completion-timeout resets at a higher power limit with uncapped clocks; ECC errors that stopped when memory speed was lowered | same |

What it taught: the verify round, not tokens per second, is the unit to measure an engine in. Tokens per second
moved with the test text; the round did not.

## The 1Cat package on one four-card board (2026-11-02)

Record: [Volta sPLIt, docs/four-card.md](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/blob/main/docs/four-card.md).
The package is [1Cat-vLLM #1277](https://github.com/1CatAI/1Cat-vLLM/pull/1277) by yangzhuxinyzx.

| Finding | Number (MEASURED on one board, #1277 with no changes) | Source |
|---|---|---|
| Warm decode | 27.8 ms per round; warm real text 30.8 ms per round, 97.1 tok/s | docs/four-card.md |
| Fresh text | 116.7 ms per round, 25.0 tok/s | same |
| Why fresh text is slow | Four cards cannot hold every expert (the routed experts alone are 132 GB), so a third of them are fetched from host memory over a shared PCIe uplink | same |
| Our four-card series of the later patches | **failed to boot** in its four-card smoke test (2026-11-04). Not current | same |

What it taught: the engine was already fast when the experts were on the cards. The whole Volta sPLIt port follows
from that one observation: put the second half of the model on a second board so that no expert is ever fetched.

## Swift 1.5 on four V100s (October 2026)

Repository: <https://github.com/ProprietaryLegal/swift-1.5-v100>

A 27B model, not DeepSeek. It runs on the stock 1Cat-vLLM 1.5.1 wheel with no engine change: NVFP4 MLP weights and
FP8 attention weights load through 1Cat's Volta paths. MEASURED: 247.7 tok/s decode at a 12K-token prompt on four
cards, the same as the base model's NVFP4 build
([README](https://github.com/ProprietaryLegal/swift-1.5-v100#what-we-got)). It is here as the reference point for
what one board does with a model that fits it: tensor parallel 4 inside one board, no pipeline, no offload.

## The V100 research notes of June and July 2026

Folder: <https://github.com/ProprietaryLegal/pli-labs/tree/main/research/v100>. The July 2026 serving sprint
measured, on the same class of hardware: prefix-cache reuse cutting time to first token 36.7 to 86.6 times; CUDA
graphs giving 2.56 to 4.5 times decode on sm_70; a 122B mixture-of-experts model going from 39.7 tok/s on stock
Ollama to 61.8 on llama.cpp and 67.6 on vLLM
([case study](https://github.com/ProprietaryLegal/pli-labs/blob/main/research/v100/serving-optimization-2026-07/README.md)).
The fuller research archive behind those notes is in this repository's [research/](../research/README.md) folder.
