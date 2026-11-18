# The builds

Every final build of PLI Labs on Tesla V100 (Volta, sm_70) cards: its link, what it runs on, and five lines of
numbers. Each build is described on its own; depth is on its own page under [optimizations/](optimizations/README.md)
and in its repository. All serve DeepSeek-V4.1-Flash-1017 (40 layers, 384 routed experts per layer, the DSpark
drafter, the Engram memory).

Every number is **MEASURED** on our hardware with its date, power limit and clock ceiling, unless it says
**UPSTREAM** (someone else's, linked) or **ESTIMATE** (arithmetic). "Quality not established" means no accuracy test
has run on that configuration. tok/s is tokens per second; "prefill" is prompt reading; the KV cache is what lets a
request hold a long context.

## Volta 12X AmPLIfy

[![Volta 12X AmPLIfy: 12x V100-SXM2-32GB; decode 161 tok/s for one code request at 128K tokens of context, 87 on legal or prose; prefill 4,444 tok/s on a 128K code prompt, 4,152 at 256K; three code requests together 364 tok/s summed; KV cache: 524K tokens of context per request, four slots](https://raw.githubusercontent.com/ProprietaryLegal/ProprietaryLegal/main/assets/volta-12x-amplify.svg)](https://github.com/ProprietaryLegal/Volta-12X-AmPLIfy-DeepSeek-V4.1-Flash-1017-NVFP4-12xV100-SXM2-PP3xTP4)

Repository: <https://github.com/ProprietaryLegal/Volta-12X-AmPLIfy-DeepSeek-V4.1-Flash-1017-NVFP4-12xV100-SXM2-PP3xTP4> ·
[its optimizations](optimizations/volta-12x-amplify.md)

Every routed expert at NVIDIA's NVFP4 precision, all 15,360 in card memory. MEASURED 2026-11-18, the closing bench of
the first optimization campaign on the serving set of that day (kit 9N with nine engine switches, see
[its optimizations](optimizations/volta-12x-amplify.md)): 175 W per card, clock ceiling 1,452 MHz, four slots, cold
reads, thinking off, 1,024 tokens out, mean of two runs; code, legal and prose prompts of 1K to 256K tokens.

| What | Figures |
|---|---|
| **Hardware** | 12x V100-SXM2-32GB on three NVLink boards; three pipeline stages of four tensor-parallel cards (layers 0-13, 14-27, 28-39) |
| **Speed, one request** | Decode at 128K tokens of context: code 161 tok/s, legal 87, prose 87 (at 1K: 178 / 99 / 84; at 256K: 168 / 94 / 79). Prefill, prompt tokens over the time to the first token: code 4,414 tok/s at 64K, 4,444 at 128K, 4,152 at 256K; legal 3,746 / 3,794 / 4,076; prose 3,670 / 4,033 / 4,159. First token on a 128K prompt: 29.5 s code, 35.4 s legal, 32.8 s prose; on a 256K prompt 63 to 64 s |
| **Several requests together** | Three streams released together on 128K prompts, each stream's own decode rate while all three stream, summed: code 364 tok/s, legal 205, prose 200 (at 1K: 432 / 248 / 213). Counted over the joint window, from the first request's start to the last one's end: 139 / 109 / 108 tok/s at 128K and 415 / 242 / 209 at 1K. Long prompts are read one after another, so the later streams wait for the earlier reads; a pipeline step carries one request. With the document already in the prefix cache (MEASURED 2026-11-18, same conditions): a request's first token comes in 0.4 to 0.5 s at 128K and 0.7 s at 256K at any kind, and three code streams on one cached 128K document decode 390 tok/s over the joint window (legal 254, prose 228), the summed and joint figures within 2 %. The first time a second or a third stream shares a document the server reads it once more; after that every stream hits |
| **Context and KV cache** | 524,288 tokens per request, four slots (four chats kept warm). On a chat of about 128,600 tokens with every prefix a hit, a turn that adds 880 tokens reaches its first token in 1.13 s; adding 4,200 tokens, 2.32 s (window p5, 175 W). Decode round 24.2 ms on a short chat, 27.0 ms at 128K of context (window p3, 175 W) |
| **Accuracy** | The experts are NVIDIA's NVFP4 pack, value-equal to DeepSeek's official experts (UPSTREAM). Against an all-native reference: first divergence after a median of 208 tokens; mean absolute log-probability difference 0.0090. Answer quality in use: not established |
## Volta Max uPLIft

[![Volta Max uPLIft: 8x V100-SXM2-32GB; at 128K context: decode 148 tok/s on code, 146 on JSON, 79 on legal text, 77 on prose; prefill 2,727 to 3,145 tok/s, first token after 43.8 s; three requests together 176 tok/s, same tokens as each alone; KV cache: 524K tokens of context per request, two slots, a follow-up resumes in 0.69 s](https://raw.githubusercontent.com/ProprietaryLegal/ProprietaryLegal/main/assets/volta-max-uplift.svg)](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4)

Repository: <https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4> ·
[Hugging Face package, 556 GB](https://huggingface.co/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-hot_iq-8xV100-SXM2-PP2xTP4) ·
[its optimizations](optimizations/volta-max-uplift.md)

As much of the routed work at NVIDIA's NVFP4 precision as fits on eight cards, every routed expert on the cards.
MEASURED 2026-11-13 (close-out bench, two-slot default, cards at the default 300 W limit, clock ceiling 1,530 MHz)
unless the row gives another date.

| What | Figures |
|---|---|
| **Hardware** | 8x V100-SXM2-32GB on two NVLink boards; two pipeline stages of four tensor-parallel cards. Host RAM 1 TB |
| **Speed** | At 128K of context: decode 147.9 tok/s on code, 145.5 on JSON, 79.1 on legal text, 76.9 on prose. Short sets: 117.7 tok/s general, 94.7 legal-style. Prefill 2,727 to 3,145 tok/s at 128K and 2,876 to 3,132 at 256K; first token after 43.8 s at 128K |
| **Context and KV cache** | 524,288 tokens per request. A follow-up after a changed reply resumes from a snapshot: first token 0.69 s where a cold read takes 31.0 s at 100K tokens |
| **Concurrency** | Two slots by default, one and three as options. Two requests together 158.7 tok/s in total; three together on three slots 176.2. Every request gave the same token ids as the same prompt alone |
| **Accuracy** | First release (one slot, native prompt reading), 2026-11-11, rules fixed before any capture: the same next token as NVIDIA's NVFP4 pack on 95.1 % of compared tokens; KL divergence from it 0.079 nats; 83.6 % on 513 MMLU questions. The default since 2026-11-12 (two slots, fast prompt reading): **quality not established** |

## Volta sPLIt

[![Volta sPLIt: 8x V100-SXM2-32GB; at 128K context: decode 183 tok/s on code, 197 on JSON, 111 on legal text, 91 on prose; prefill 2,200 tok/s, first token after 59 s; one request at a time; KV cache: 524K tokens of context per request](https://raw.githubusercontent.com/ProprietaryLegal/ProprietaryLegal/main/assets/volta-split.svg)](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4)

Repository: <https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4> ·
[Hugging Face card](https://huggingface.co/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-GGUF) ·
[its optimizations](optimizations/volta-split.md)

The 1Cat-vLLM DeepSeek-V4.1 package ([#1277](https://github.com/1CatAI/1Cat-vLLM/pull/1277), yangzhuxinyzx) ported to
two boards, with ISTA-DASLab's 2-bit GGUF and every expert resident on the cards. MEASURED 2026-11-05 (checkpoint
build, one request at a time, thinking off, greedy, 150 W per card, clock ceiling 1,522 MHz).

| What | Figures |
|---|---|
| **Hardware** | 8x V100-SXM2-32GB on two NVLink boards; two pipeline stages of four tensor-parallel cards; 251 GB of host RAM for the Engram tables |
| **Speed** | At 128K of context: decode 182.8 tok/s on code, 196.5 on JSON, 110.9 on legal text, 91.1 on prose. At 1K: 215.0 on code, 117.6 on legal text. Prefill 2,198 to 2,220 tok/s at 128K; first token after 59.2 s at 128K, 17.2 s at 32K |
| **Context and KV cache** | 524,288 tokens per request; KV pool 2,076,870 tokens. Needles found at about 450K and 498K tokens: 6 of 6 in 4 of 4 runs |
| **Concurrency** | One request at a time. Three chat slots (since 2026-11-11) keep three chats' caches: a return to a chat reaches its first token in 0.57 to 0.88 s where a re-read took 20.6 to 49.2 s |
| **Accuracy** | LegalBench 0.716 balanced accuracy (22 tasks); MMLU-Pro 79.9 % (700 items); HumanEval+ 89.6 %. The same next token as the original model 89.0 % of the time on a general panel, 92.5 % on a legal panel |

## Volta Full NVFP4

[![Volta Full NVFP4: 8x V100-SXM2-32GB plus host RAM; decode 28 to 30 tok/s on legal-style text; prefill 1,419 to 1,595 tok/s, a 498K-token prompt read in 326.6 s; one request at a time; KV pool 2.08M tokens, 524K context per request](https://raw.githubusercontent.com/ProprietaryLegal/ProprietaryLegal/main/assets/volta-full-nvfp4.svg)](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100)

Repository: <https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100> ·
[its optimizations](optimizations/volta-full-nvfp4.md)

Every routed expert at exactly NVIDIA's NVFP4 precision on eight cards; the experts that do not fit are read from
pinned host RAM over PCIe. Built as an all-NVFP4 output to check other work against. MEASURED 2026-11-09, clock
ceiling 1,522 MHz.

| What | Figures |
|---|---|
| **Hardware** | 8x V100-SXM2-32GB on two NVLink boards; two pipeline stages of four; about 101 GB of experts in host RAM (ESTIMATE) |
| **Speed** | Decode 28.2 and 30.2 tok/s on a legal-style short set (two runs), 18.8 and 20.1 on a general set; 31.5 tok/s after 106,531 tokens of context. Prefill 1,419 to 1,595 tok/s |
| **Context and KV cache** | 524,288 tokens per request; KV pool 2,076,870 tokens. A 498,099-token prompt read in 326.6 s with no out-of-memory error |
| **Concurrency** | One request at a time |
| **Accuracy** | 44,696,862,720 folded expert values checked bit-equal to the NVFP4 formula. 83.8 % on 513 MMLU questions (2026-11-11) |

## Earlier V100 builds

| Repository | What it is |
|---|---|
| [deepseek-v4-flash-v100](https://github.com/ProprietaryLegal/deepseek-v4-flash-v100) | DeepSeek-V4-Flash on eight V100s with llama.cpp (July 2026): 12.2 tok/s generation, 209 tok/s prefill at 8K (MEASURED, 245 W per card); the root cause and fix of a silent quantized-KV corruption; and the interim DeepSeek-V4.1 results of 2026-11-01 on the EXL3 engine |
| [Volta sPLIt, `previous-build/`](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/tree/main/previous-build) | The first DeepSeek-V4.1 build on eight cards, on Phil Tompkins' EXL3 engine (2026-11-01): 42 to 46 tok/s decode by its own method |
| [deepseek-v4.1-v100](https://github.com/ProprietaryLegal/deepseek-v4.1-v100) | A fork of [philbert440's engine snapshot](https://github.com/philbert440/deepseek-v4.1-v100). Phil Tompkins' work |
| GLM-5.3-Flash NVFP4 on eight V100s with 1Cat-vLLM (September 2026; no repository of its own) | The first fast large-model build on these boards: about 49 tok/s without a drafter, 78.8 to 113.8 tok/s with the DFlash2 drafter on short prompts, 204,800 tokens of context (MEASURED; [earlier builds](optimizations/earlier-builds.md#glm-53-flash-nvfp4-on-eight-v100s-with-1cat-vllm-september-2026)) |
| [swift-1.5-v100](https://github.com/ProprietaryLegal/swift-1.5-v100) | Swift 1.5 (a 27B model) on four V100s with the stock 1Cat-vLLM wheel: 247.7 tok/s decode at a 12K prompt (MEASURED) |
| [pli-labs, `research/v100`](https://github.com/ProprietaryLegal/pli-labs/tree/main/research/v100) | The V100 research notes of June and July 2026 |
| Hugging Face [deepseek-v4.1-flash-sm70](https://huggingface.co/ProprietaryLegal/deepseek-v4.1-flash-sm70), [sm70-fp8-block32-kernels](https://huggingface.co/ProprietaryLegal/sm70-fp8-block32-kernels) | The first twelve-card port ([1Cat-vLLM #1031](https://github.com/1CatAI/1Cat-vLLM/pull/1031), October 2026; UPSTREAM in the pull request: 2.8 to 3.9 tok/s without CUDA graphs) and its FP8 dense kernels ([#1021](https://github.com/1CatAI/1Cat-vLLM/pull/1021)) |

What each of these found: [optimizations/earlier-builds.md](optimizations/earlier-builds.md).

## Mirrors we keep (other people's work)

| Mirror | Original | Licence |
|---|---|---|
| [ProprietaryLegal/1Cat-vLLM](https://github.com/ProprietaryLegal/1Cat-vLLM) | [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM), the Volta and Turing vLLM fork every build here runs on | Apache-2.0 |
| [ProprietaryLegal/v100-skinny](https://github.com/ProprietaryLegal/v100-skinny) | [dnv2003/v100-skinny](https://github.com/dnv2003/v100-skinny), Dennis Vertzyas' hand-written NVFP4 and FP8 kernels for Volta | MIT (kernels, docs), Apache-2.0 (fork patches) |

## Sister work (not Volta)

[DGX sPLIce](https://github.com/ProprietaryLegal/DGX-sPLIce-DeepSeek-V4.1-Flash-2xDGX-Spark-2xRTX3090-drafter-x86-offload)
runs the same model on two NVIDIA DGX Sparks with an x86 machine that holds the Engram tables and runs the drafter.
A different chip (GB10) with its own repository; its measurement harness is
[dsv41-window-harness](https://github.com/ProprietaryLegal/dsv41-window-harness).

## The hardware and the skill

What to buy to build one of these boards: [hardware.md](hardware.md). The method that produced the builds, packaged
for coding agents: [PLI Volta Builder](https://github.com/ProprietaryLegal/pli-volta-builder)
([front page, part 9](index.md#part-9-the-builder-skill)).
