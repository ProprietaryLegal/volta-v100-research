# When an optimization is likely to apply

A decision table keyed on your situation, not on ours. Each row says what was measured on the Volta builds, on
which build, and the condition under which it should transfer. "Likely" is the honest word: every gain below was
measured on V100-SXM2-32GB cards with DeepSeek-V4.1-Flash-1017, a speculative drafter that proposes five tokens,
and the 1Cat-vLLM engine. On other hardware or another model, treat each row as a hypothesis with a number
attached, and measure.

This is the long form of [part 6 of the front page](../index.md#part-6-when-an-optimization-is-likely-to-apply).
Build names link to their optimization pages, which link every figure to its source.

## 1. How many cards and boards do you have?

| Your situation | What applies | Measured basis |
|---|---|---|
| One four-card NVLink board, model fits | Tensor parallel 4 inside the board. No pipeline, no offload. | Swift 1.5: 247.7 tok/s at a 12K prompt ([earlier builds](../optimizations/earlier-builds.md#swift-15-on-four-v100s-october-2026)) |
| One board, model does not fit | Expect the offload to set your speed, not the kernels. Fresh text ran at 25.0 tok/s against 97.1 warm, because a third of the experts were fetched from host memory | [the 1Cat package on one board](../optimizations/earlier-builds.md#the-1cat-package-on-one-four-card-board-2026-11-02) |
| Two boards | **Cut the model into two pipeline stages, one per board.** Tensor parallelism stays inside a board; one hand-off per step crosses PCIe. Do not span both boards with one tensor-parallel group: the eight-way sum took 30.8 microseconds against 5.28 inside a board, and the full build of that idea lost 3 % on one request, 41 % on three and 26 % on long reads | [Volta sPLIt](../optimizations/volta-split.md), [what did not work](../optimizations/what-did-not-work.md#1-it-crossed-pcie) |
| Two boards, and the hand-off between them is slow | Check which transport the hand-off takes. One setting (`NCCL_P2P_LEVEL=NVL`) cut time to first token by 13 to 14 % | [Volta sPLIt](../optimizations/volta-split.md#adopted-long-context-and-prompt-reading) |
| Three boards | A third stage buys **capacity and kept-warm chats, not single-request speed**: a request's round still carries the same forty layers of work. Check that your engine spaces a request's decode steps by the pipeline depth: without that the three-stage build answered wrongly | [Volta 12X AmPLIfy](../optimizations/volta-12x-amplify.md) |
| More cards than fit behind one PCIe root | Give each PCIe switch its own x16 root port if you can. It was worth about 2 % on long-prompt first token and nothing on decode: tidy, not decisive | [Volta sPLIt](../optimizations/volta-split.md#hardware-changes) |
| A layer split across three or more boards (llama.cpp style) | Do not expect speed. Fourteen cards halved the throughput of eight | [earlier builds](../optimizations/earlier-builds.md#deepseek-v4-flash-on-eight-v100s-with-llamacpp-july-2026) |

## 2. How much of the model fits in card memory?

| Your situation | What applies | Measured basis |
|---|---|---|
| Everything a decode round touches fits at the precision you want | Stop here. Nothing is fetched during decode | Volta sPLIt (2-bit experts on 8 cards), Volta 12X AmPLIfy (NVFP4 experts on 12) |
| It fits only at a lower precision | Serve the lower precision everywhere, or promote the **most-used** weights to the higher precision until the cards are full. Promotion by use per byte put 6,087 of 15,360 experts at native precision and 79 to 85 % of decode routes on them. The result picked the same next token as NVIDIA's NVFP4 pack on 95.1 % of compared tokens. It costs decode speed: the promoted records are larger | [Volta Max uPLIft](../optimizations/volta-max-uplift.md) |
| You are tempted to leave the rarely used weights in host RAM | Size it first: about 975 microseconds per host-RAM expert per layer at one row over PCIe Gen3. The all-native eight-card build ran at 28 to 30 tok/s. Every miss path tried lost | [Volta Full NVFP4](../optimizations/volta-full-nvfp4.md) |
| Large lookup tables that a round reads only a few rows of | Host RAM is fine. The Engram tables (110 GB) stay in host RAM and cost about 0.12 ms per round. Keep them resident: reading them from disk cost 19 s on a cold 75K prompt | [Volta sPLIt](../optimizations/volta-split.md), [earlier builds](../optimizations/earlier-builds.md#deepseek-v41-flash-on-the-exl3-engine-2026-11-01) |
| Memory is within a few hundred MiB of full | Set the reserve per stage from sampled peaks during a long read, not one reserve for all. Sample memory through a whole boot. Add a hard limit below the card's capacity so that an overrun fails loudly in one worker | Volta Full NVFP4, Volta 12X AmPLIfy |
| You need room for another slot | Halve the prompt chunk. It roughly halved the working memory of a prompt read on the fullest stage and cost 4 % of reading speed | [Volta 12X AmPLIfy](../optimizations/volta-12x-amplify.md#adopted-configuration-the-builds-own-settings-against-each-other) |

## 3. One request at a time, or many?

| Your situation | What applies | Measured basis |
|---|---|---|
| One user, one request at a time | Everything in the decode tables applies. Model Runner V2 was the largest single decode gain (-2.76 ms of a 25 ms round). Keep scheduling synchronous: asynchronous scheduling with a pipeline and a drafter was 1.7 ms per round slower | [Volta sPLIt](../optimizations/volta-split.md#adopted-decode-the-round-a-token-is-written-in) |
| Several long chats that take turns (an agent and its helpers) | **Slots, for their caches.** With one slot every return to a chat read its whole prompt again (20 to 49 s); with a slot per chat, 0.6 to 0.9 s. This is the largest latency gain in the whole record and needs no kernel | [Volta sPLIt](../optimizations/volta-split.md#adopted-caches-and-correctness), [Volta 12X AmPLIfy](../optimizations/volta-12x-amplify.md#adopted-configuration-the-builds-own-settings-against-each-other) |
| Several requests truly at once | With this engine a pipeline step carries one request, so throughput rises only up to the number of stages: two together 1.33 times one stream on two stages; three together 1.48 times; on three stages a fourth request added 6 %. Batched decode inside a step exists upstream ([1Cat-vLLM #1482](https://github.com/1CatAI/1Cat-vLLM/pull/1482), UPSTREAM: 628 tok/s for four requests on sixteen cards) and is not in these builds | [Volta Max uPLIft](../optimizations/volta-max-uplift.md#adopted-speed) |
| A short request must not wait behind a long read | Slots do not solve this: the short request starts in 4 s and not 57 s, but decodes at 15 tok/s until the read ends, and both end later | [what did not work](../optimizations/what-did-not-work.md#5-it-was-a-scheduling-or-host-change-that-cost-more-than-it-saved) |
| Each slot costs memory you were using for precision | It does. Three slots held 5,650 experts at native precision against 6,203 with one | [Volta Max uPLIft](../optimizations/volta-max-uplift.md#adopted-speed) |

## 4. Short context or long?

| Your situation | What applies | Measured basis |
|---|---|---|
| Prompts under about 8K tokens | The decode-round levers are all that matter. The long-context scorers showed no detectable change here | Volta sPLIt |
| Prompts of 32K to 128K | Keep the request in the batch across empty pipeline steps (-9.7 % of a round at 32K, -24.8 % at 128K). Skip, for early prompt chunks, the layers whose state those chunks cannot reach (-8.8 % to first token at 32K). Check the transport of the stage hand-off (-13 to -14 %) | [Volta sPLIt](../optimizations/volta-split.md#adopted-long-context-and-prompt-reading) |
| Prompts of 150K to 500K | Add: score only the live part of the attention index (decode round -7.0 % at 150K, -13.7 % at 300K); the fused prefill index scorer (first token -4.9 % at 150K, -9.5 % at 300K); a larger on-card ring for the key-value cache (-1.09 ms per round at 300K). The prefill skip grows to -20.6 % at 150K and -31.5 % at 300K | same |
| Decode slows as the chat grows | Look at the attention index first. On twelve cards it grew from 0.77 ms of a round on a short chat to 2.78 ms at 128K and 4.47 ms at 249K, and it was the whole growth | [Volta 12X AmPLIfy](../optimizations/volta-12x-amplify.md#where-the-time-goes) |
| Prompt reading is slower than expected and the cards show idle time inside each chunk | Check whether the allocator's cache is released after every chunk. Releasing only above the chunk size read 10 to 13 % faster at 60K to 93K tokens with no higher memory peak | [Volta 12X AmPLIfy](../optimizations/volta-12x-amplify.md#tested-not-yet-in-what-serves) |
| You want a faster prompt read and can spend watts | A higher power limit helped prompt reading (+13.7 % from 150 W to 200 W on twelve cards) and did nothing for decode (neutral on eight cards, under 2 % on twelve) | Volta 12X AmPLIfy, Volta sPLIt |

## 5. How much of the prompt repeats?

| Your situation | What applies | Measured basis |
|---|---|---|
| Every request continues the last one exactly | The ordinary prefix cache already serves you. Give each concurrent chat a slot | section 3 |
| A client inserts a system message in the middle of a chat | Keep it where the client sent it. Moving it to the front shifted every later token and voided the cache: 11.2 s against 0.51 s for the next turn | [Volta sPLIt](../optimizations/volta-split.md#adopted-caches-and-correctness) |
| A tool-calling chat whose previous reply comes back slightly changed | Keep a snapshot of the sliding-window state at the end of each prompt and resume there: 0.69 s against 31 s at 100K tokens, with no card memory | [Volta Max uPLIft](../optimizations/volta-max-uplift.md#adopted-the-prompt-cache) |
| The client drops the model's reasoning when it replays the chat | Expect a cold read after any long answer: 36 to 44 s in place of about a second on a 130K-token chat. The cure is on the client: replay what the model produced | [Volta 12X AmPLIfy](../optimizations/volta-12x-amplify.md) |
| Cache misses you cannot explain | Log one line per request with the cache decision and its reason, numbers only | [Volta Max uPLIft](../optimizations/volta-max-uplift.md#adopted-the-prompt-cache) |
| Nothing repeats | No cache helps. Spend on prompt-reading speed (section 4) | |

## 6. Must answers stay bit-identical?

| Your requirement | What you may use | What you must give up |
|---|---|---|
| Bit-identical to the engine before the change | Every lever marked "bitwise", "exact" or "identical output" in the build tables: the index reduce, the double-buffered mat-vec, the fast expert kernels, the plan fusion, the attention fusions, the drafter sharding, Model Runner V2, the tiled index scorer, the prefill skip, the transport setting, slots, the snapshot hit | The instruction diet (-0.87 ms), the fused prefill index scorer and the prefill expert GEMM: they passed a near-tie gate but are not bitwise |
| The same answers except at genuine near-ties | Add the three above | Anything that failed the gate: the fused prefill attention (-11 % to first token), tensor-core decode attention, other prompt chunk sizes |
| Close to a reference model, measurably | Then change precision deliberately and measure closeness: promote the most-used weights, or add cards | Identity as a test. Use the closeness judge ([testing.md](testing.md#7-two-judges-and-what-each-can-show)) |
| Reproducible across restarts | Sampled output was not, until an all-reduce's order was fixed. Greedy output on one test machine differed between two boots of the same bytes. Test inside one boot | |
| Identical whether a turn was cached or read cold | Not available on this engine at near-tied tokens, with or without any of these changes | |

## 7. What kind of text?

Tokens per second depends on the text more than on any lever here. On the same engine and the same round time:

| Text | Accepted drafts of 5 | Tokens per round | Decode at a 1K prompt (Volta sPLIt, MEASURED 2026-11-05) |
|---|---|---|---|
| Code | 3.6 to 3.8 | about 4.7 | 215 tok/s |
| Structured output (JSON) | 3.6 to 3.8 | about 4.6 | 203 tok/s |
| Legal writing | about 1.6 | about 2.6 | 118 tok/s |
| Prose, chat | 1.1 to 1.2 | about 2.1 | 89 to 95 tok/s |

Source: [Volta sPLIt, "Why the kinds differ"](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4#why-the-kinds-differ).
If your work is prose, the drafter is where the next gain is, and three attempts to tune it lost
([what did not work](../optimizations/what-did-not-work.md#3-the-round-got-shorter-and-the-tokens-got-fewer)).
Coupling the drafter's random draws to the verifier's did win at temperatures above 0: +14 % tok/s.

## Before you port any of it

1. Read what upstream has already built and measured. Most of what is fast here is other people's work
   ([CREDITS](../CREDITS.md)).
2. Size the idea: bytes over bandwidth, and the 0.3 ms floor ([kernel-building.md](kernel-building.md)).
3. Write the rule before the data ([testing.md](testing.md)).
