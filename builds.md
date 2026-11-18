# The builds

Every final build of PLI Labs on Tesla V100 (Volta, sm_70) cards, with its link, what it runs on and its headline
numbers. All serve one model, DeepSeek-V4.1-Flash-1017 (40 layers, 384 routed experts per layer, the DSpark
speculative drafter, the Engram n-gram memory), unless the row says otherwise.

Labels, as everywhere in this repository:

- **MEASURED**: observed on our hardware. The row gives the date, the power limit and the clock ceiling.
- **UPSTREAM**: someone else's published figure, which we have not reproduced. The row links it.
- **ESTIMATE**: arithmetic or a model built on measurements. Never a result.

A configuration that was not accepted by its own rule says "quality not established". Speeds measured on different
days at different power limits are not a controlled comparison, and the tables say so where it matters.

## At a glance

| Build | Cards and layout | Routed experts | One request, decode | Prompt reading | Status |
|---|---|---|---|---|---|
| [Volta sPLIt](#volta-split) | 8, PP2 x TP4 | all from compressed IQ2-class records, all on the cards | code 215 tok/s, legal 118 at a 1K prompt | 2,200 tok/s at 128K | public, final (checkpoint of 2026-11-05) |
| [Volta Max uPLIft](#volta-max-uplift) | 8, PP2 x TP4 | the most-used at NVIDIA's NVFP4 precision, the rest compressed, all on the cards | general 117.7 tok/s, legal-style 94.7 (short sets) | 2,876 to 3,132 tok/s at 256K | public, final (close-out of 2026-11-13) |
| [Volta Full NVFP4](#volta-full-nvfp4) | 8, PP2 x TP4 | all at NVFP4 precision, the overflow in host RAM | legal-style 28 to 30 tok/s, general 19 to 20 | 1,419 to 1,595 tok/s | public, final (a reference build) |
| [Volta 12X AmPLIfy](#volta-12x-amplify) | 12, PP3 x TP4 | all at NVFP4 precision, all on the cards | 128.1 tok/s on fresh text | 3,284 tok/s on a 62K prompt (150 W) | measured; repository not yet published |

"PP2 x TP4" means two pipeline stages of four tensor-parallel cards: the model's layers are cut into two runs, each
run lives on one four-card NVLink board, and the four cards of a board each hold a quarter of every matrix. The
front page explains [why the model is cut that way](index.md#part-3-how-it-was-made-to-work-under-1cat-vllm).

## Volta sPLIt

Repository: <https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4> ·
Hugging Face card: <https://huggingface.co/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-GGUF>

- **What it is.** The 1Cat-vLLM DeepSeek-V4.1 package for one four-card board
  ([1Cat-vLLM #1277](https://github.com/1CatAI/1Cat-vLLM/pull/1277), by yangzhuxinyzx) ported to two boards, then made
  fast one switch at a time. Experts and backbone are ISTA-DASLab's 2-bit
  [GSQ-RCO GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) (138.24 GB), every expert
  resident in card memory. The Engram tables (110.60 GB) stay in host RAM.
- **Runs on.** Eight V100-SXM2-32GB on two four-card NVLink boards; a 12-core workstation CPU; 251 GB of host RAM.
- **Headline, MEASURED 2026-11-05** (one request at a time, thinking off, greedy, 1,024 tokens out, 150 W per
  card, clock ceiling 1,522 MHz;
  [source](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4#tokens-per-second)):

  | Prompt | First token | Prompt reading | Code | Structured | Legal | Prose | Chat |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | 1K | 1.22 s | 820 to 948 tok/s | 215.0 | 202.5 | 117.6 | 89.2 | 95.1 |
  | 32K | 17.2 s | 1,894 to 1,913 tok/s | 200.5 | 206.6 | 116.5 | 90.1 | 106.9 |
  | 128K | 59.2 s | 2,198 to 2,220 tok/s | 182.8 | 196.5 | 110.9 | 91.1 | 95.8 |
  | 256K | 118.7 s | 2,197 to 2,222 tok/s | 160.0 | 179.7 | 103.1 | 84.1 | 80.5 |

- **The progression on one fixed test set, MEASURED** (the same source): the plain two-board port 99.2 tok/s
  (31.1 ms per round, 2026-11-03); with Model Runner V2 and the kernel work 134.2 to 135.0 tok/s (2026-11-04). The
  checkpoint figure of about 135.2 is an ESTIMATE (that row plus one measured change of -0.180 ms per round).
- **Quality, MEASURED 2026-11-05**
  ([source](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4#quality)):
  LegalBench 0.716 balanced accuracy over 22 tasks; MMLU-Pro 79.9 % on 700 items; HumanEval+ 89.6 %; the served
  build picks the original model's most likely next token 89.0 % of the time on a general panel and 92.5 % on a
  legal panel. That gap is the 2-bit quantization, not the speed work.
- **Context.** 524,288 tokens per request. Needle retrieval at about 450K and 498K tokens: 6 of 6 needles in 4 of
  4 runs (MEASURED, same source).
- **What it is not.** One request runs at a time. Three chat slots keep three chats' prompt caches (adopted
  2026-11-11); they are not a throughput claim.
- Optimizations, won and lost: [optimizations/volta-split.md](optimizations/volta-split.md).

## Volta Max uPLIft

Repository: <https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4> ·
Hugging Face package (556 GB): <https://huggingface.co/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-hot_iq-8xV100-SXM2-PP2xTP4>

- **What it is.** The same eight cards and engine, with as much of the routed-expert work as fits at the precision
  of NVIDIA's [NVFP4 pack](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4). Every expert's compressed
  record goes on the cards first; then experts are promoted to their NVFP4 weights, most-used first, until the next
  one does not fit. In the first release that put 6,087 of the 15,360 routed experts at NVFP4, and 79.0 to 85.0 %
  of decode routes ran on them (MEASURED, in-sample, on the texts the choice was made from).
- **Runs on.** The same eight V100-SXM2-32GB. Since 2026-11-12 the host has 1 TB of RAM.
- **Headline, MEASURED 2026-11-13, the close-out bench** (two-slot default, one request at a time, cards at the
  default 300 W limit and a 1,530 MHz ceiling: the power cap was not in force;
  [source](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4#speed-tokens-per-second)):
  - short sets: general 117.7 tok/s (26.36 ms per round), legal-style 94.7 tok/s (25.98 ms per round);
  - code 143 to 175 tok/s from a 1K to a 256K prompt, legal 79 to 99;
  - first token 13.7 s after 32K, 43.8 s after 128K, 86.7 s after 256K; prompt reading 2,876 to 3,132 tok/s at
    256K;
  - three requests together on the three-slot option: 176.2 tok/s in total, 1.48 times one stream.
- **Closeness to the NVFP4 pack, MEASURED 2026-11-11** on the build of the first release (one slot, native prompt
  reading), rules fixed before any capture
  ([source](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4#accuracy-how-close-to-nvfp4)):

  | Measure | Volta Max uPLIft | Volta sPLIt | The reference run twice |
  |---|---|---|---|
  | Same next token as the NVFP4 reference | 95.1 % | 92.2 % | 100.0 % |
  | Mean KL divergence from the reference (top 20, nats) | 0.079 | 0.196 | 0.000 |
  | Multiple-choice answers that differ from the reference's | 1.6 % | 15.4 % | |

- **Quality not established** for the default since 2026-11-12 (two slots, fast prompt reading) and for the one-
  and three-slot options: no closeness test has run on them, and they hold fewer NVFP4 experts.
- **Slower than Volta sPLIt** in decode in every code and legal cell of the two benches; faster to the first
  token. The two benches ran at different power limits.
- Optimizations, won and lost: [optimizations/volta-max-uplift.md](optimizations/volta-max-uplift.md).

## Volta Full NVFP4

Repository: <https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100>

- **What it is.** Every routed expert at exactly NVIDIA's NVFP4 precision on eight cards. At that precision the
  experts need about 269 GiB (ESTIMATE, arithmetic in the repository) and eight cards have 256 GiB, so the
  most-used experts stay on the cards and the rest are read from pinned host RAM over PCIe. It is the reference of
  Volta Max uPLIft's closeness test.
- **Headline, MEASURED 2026-11-09** (one request at a time, greedy, thinking off, clock ceiling 1,522 MHz;
  [source](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100#tokens-per-second)):
  legal-style short set 28.2 and 30.2 tok/s (two runs), general 18.8 and 20.1; prompt reading 1,419 to 1,595
  tok/s; a 498,099-token prompt read in 274.9 and 326.6 s (two reserve settings) with no out-of-memory error.
- **Exactness, MEASURED 2026-11-09:** folding NVFP4 to MXFP4 is exact. 44,696,862,720 folded values were checked
  bit-equal to the NVFP4 formula
  ([source](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100#what-this-is)).
- **Why it is slow, MEASURED:** one expert read from host RAM adds about 975 microseconds to a layer at one row.
  No miss path came within about 10 ms per round of having every expert on the cards (ESTIMATE from the measured
  miss costs). That finding is why Volta Max uPLIft keeps every expert on the cards and why the twelve-card build
  exists.
- Optimizations, won and lost: [optimizations/volta-full-nvfp4.md](optimizations/volta-full-nvfp4.md).

## Volta 12X AmPLIfy

Repository: **not yet published.** The build is measured and described here from PLI Labs' private working record.
A public repository under the name `Volta-12X-AmPLIfy-DeepSeek-V4.1-Flash-1017-NVFP4-12xV100-SXM2-PP3xTP4` does not
exist yet; this page will link it when it does.

- **What it is.** Twelve V100-SXM2-32GB in three NVLink boards, three pipeline stages of four tensor-parallel
  cards (layers 0 to 13, 14 to 27, 28 to 39). All 15,360 routed experts are held in card memory at NVFP4
  precision: none is compressed and none is read from host memory. One expert in NVFP4 is 4,874,240 bytes on each
  of its four cards, so all forty layers are 278.9 GiB of expert records: more than eight cards hold, less than
  twelve (ARITHMETIC).
- **Where the layout comes from.** The three-stage cut at layers 14 and 28 is that of
  [1Cat-vLLM #1031](https://github.com/1CatAI/1Cat-vLLM/pull/1031) (PLI Labs, 2026-10-07), which ran the model on
  twelve cards at about 3 tok/s on an engine without CUDA graphs (UPSTREAM, the pull request's own figure). The
  work of this build was teaching the fast eight-card engine a third stage.
- **Headline, MEASURED 2026-11-17** (a twelve-V100 server; four prompt slots; 4,092-row prompt chunks; 150 W per
  card, clock ceiling 1,477 MHz unless the row says 200 W; source: the build's measurement record, private until
  the repository is published):

  | What | Figure | Conditions |
  |---|---|---|
  | One request, decode | 128.1 tok/s on fresh text; 176.6 tok/s on a copying task | 150 W |
  | Four requests together, decode | 306.6 tok/s on fresh text, summed over the four requests (60.5 to 85.5 each) | 150 W. A sum over requests is not elapsed throughput: by wall clock the same run gave 242.6 tok/s |
  | Three requests together, decode, against the eight-card default measured in the same window with the same tool | 288.0 tok/s against 198.2 (sums) | 150 W, three slots |
  | Prompt reading, one 62,263-token prompt | 3,284 tok/s at 150 W; 3,733 tok/s at 200 W | one run each |
  | Prompt reading by length | 60,706 tokens 3,850 tok/s (first token 15.8 s); 93,025 tokens 3,833 (24.3 s); 247,953 tokens 3,316 (74.8 s); 425,777 tokens 2,454 (173.5 s) | 200 W, one run each |
  | First token of a turn that adds about 880 tokens to a 128,600-token chat | 1.24 s at the client | 200 W |
  | The same, adding about 4,200 tokens | 3.75 s at the client | 200 W, median of two |
  | Load, launch to ready | 134 s | |
  | Card memory, highest sample with four grown chats at once | 30,398 MiB of 32,510 | |

- **Closeness to an all-NVFP4 reference, MEASURED 2026-11-17** (the reference is a two-stage arm with every
  expert at NVFP4 and the overflow read from host memory; short prompts): this build's answers first diverge from
  the reference after a median of 208 tokens, the eight-card default's after a median of 7; the median of the mean
  absolute log-probability difference is 0.0090 against 0.0656. These figures say how closely the build tracks the
  all-NVFP4 model. They do not by themselves say its answers are better for a given kind of work.
- **What it is not.** It is not a faster engine for one request: one request decodes at about the eight-card
  default's speed (no measurable change in one window, -3.4 % and -1.2 % in another, inside the night's spread).
  More requests at once than stages gain little, because a pipeline step carries one request.
- **Open.** No figure exists for memory over hours of serving at four slots. Six and eight slots have no verdict.
  Answer quality in use is not established.
- Optimizations, won and lost: [optimizations/volta-12x-amplify.md](optimizations/volta-12x-amplify.md).

## Earlier V100 builds

| Repository | What it is | Headline (its own conditions) |
|---|---|---|
| [deepseek-v4-flash-v100](https://github.com/ProprietaryLegal/deepseek-v4-flash-v100) | DeepSeek-V4-Flash on eight V100s with llama.cpp (July 2026), with the root cause of a silent quantized-KV corruption and the fix; then the interim DeepSeek-V4.1 results on the EXL3 engine (2026-11-01) | MEASURED: V4-Flash 12.2 tok/s generation, 209 tok/s prompt reading at 8K, llama.cpp layer split, 245 W per card. V4.1 interim: 42 to 46 tok/s decode |
| [deepseek-v4.1-v100](https://github.com/ProprietaryLegal/deepseek-v4.1-v100) | A fork of [philbert440's engine snapshot](https://github.com/philbert440/deepseek-v4.1-v100) for the EXL3 pack at 1M context on eight V100s. Phil Tompkins' work, mirrored | UPSTREAM: see the repository |
| [Volta sPLIt, `previous-build/`](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/tree/main/previous-build) | Our first build on these eight cards: the EXL3 pack on that engine (2026-11-01) | MEASURED, its own method: 42 to 46 tok/s decode, 31.4 with thinking on; 71.2 to 71.6 ms per verify round |
| [swift-1.5-v100](https://github.com/ProprietaryLegal/swift-1.5-v100) | Swift 1.5 (a Qwen3.8-27B fine-tune) on four V100s with the stock 1Cat-vLLM 1.5.1 wheel: launch profile, power settings, a measured comparison | MEASURED: 247.7 tok/s decode at a 12K prompt on four cards |
| [pli-labs, `research/v100`](https://github.com/ProprietaryLegal/pli-labs/tree/main/research/v100) | The earlier V100 research notes: the July 2026 serving sprint, launch profiles, benchmark bands | see the folder |
| Hugging Face [deepseek-v4.1-flash-sm70](https://huggingface.co/ProprietaryLegal/deepseek-v4.1-flash-sm70) | Patches, launch scripts and results of the first twelve-card port (1Cat-vLLM #1031, October 2026) | UPSTREAM in the pull request: 2.8 to 3.9 tok/s without CUDA graphs |
| Hugging Face [sm70-fp8-block32-kernels](https://huggingface.co/ProprietaryLegal/sm70-fp8-block32-kernels) | The block-FP8 dense kernels of [1Cat-vLLM #1021](https://github.com/1CatAI/1Cat-vLLM/pull/1021) | UPSTREAM in the pull request: 115.2 to 74.2 microseconds at one row |

## Mirrors we keep (other people's work)

| Mirror | Original | Licence |
|---|---|---|
| [ProprietaryLegal/1Cat-vLLM](https://github.com/ProprietaryLegal/1Cat-vLLM) | [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM), the Volta and Turing vLLM fork every build here runs on | Apache-2.0 |
| [ProprietaryLegal/v100-skinny](https://github.com/ProprietaryLegal/v100-skinny) | [dnv2003/v100-skinny](https://github.com/dnv2003/v100-skinny), Dennis Vertzyas' hand-written NVFP4 and FP8 kernels for Volta | MIT (kernels, docs), Apache-2.0 (fork patches) |

## Sister work (not Volta)

The DGX sPLIce family runs the same model on two NVIDIA DGX Sparks with an x86 machine that holds the Engram tables
and runs the drafter on two RTX 3090s. It is a different chip (GB10) with a different bottleneck. Start at
[DGX sPLIce](https://github.com/ProprietaryLegal/DGX-sPLIce-DeepSeek-V4.1-Flash-2xDGX-Spark-2xRTX3090-drafter-x86-offload);
its measurement method is published as
[dsv41-window-harness](https://github.com/ProprietaryLegal/dsv41-window-harness).

## The builder skill

The method that produced these builds is packaged as a skill for coding agents:
<https://github.com/ProprietaryLegal/pli-volta-builder>. See [the front page](index.md#part-9-the-builder-skill).
