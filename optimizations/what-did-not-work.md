# What did not work

Every idea below was built, or sized far enough to be closed, and lost. They are grouped by the reason they lost,
because the reason is what transfers to another machine. Each row names the build it was tested on and links its
record.

All figures are MEASURED on the build named, under that build's conditions
([builds.md](../builds.md) gives them), unless the row says ESTIMATE.

## 1. It crossed PCIe

The cards read their own memory at hundreds of GB/s and reach anything else at a few GB/s.

| Idea | Build | Result | Record |
|---|---|---|---|
| Keep the less-used experts in host RAM and read them when routed | Volta Full NVFP4 | about 975 microseconds per host-RAM expert per layer at one row; the build decodes at 28 to 30 tok/s on legal-style text. No miss path (zero-copy mapping, copy-engine DMA, a gather on the card, an AVX2 host kernel) came within about 10 ms per round (ESTIMATE from the measured miss costs) | [README](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100#what-did-not-work-or-is-not-yet-good) |
| One tensor-parallel group of eight cards across two boards (probe) | Volta sPLIt | the eight-way sum across the boards took 30.8 microseconds against 5.28 inside one board; best modelled round 24.65 ms against a 22.0 ms bar. No-go before a build | [levers](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/blob/main/docs/levers.md#did-not-work) |
| The same idea built in full: one stage on eight cards, a per-layer exchange between the boards | eight cards, one test window during the twelve-card work | against the same eight cards as two stages of four: one request -3 %, three at once -41 %, long prompt reads -26 % | [the twelve-card page](volta-12x-amplify.md#rejected) |
| A draft model on a separate GPU over the network | Volta Max uPLIft | 42.6 ms per draft round at the median against a 4.5 ms bar; the draft head itself runs in 1.7 ms there | [README](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4#what-did-not-work-or-is-not-yet-good) |
| A layer split over more than two boards with llama.cpp | DeepSeek-V4-Flash, July 2026 | fourteen cards halve the throughput of eight (7.8 against 12.2 tok/s) | [README](https://github.com/ProprietaryLegal/deepseek-v4-flash-v100#quickstart) |
| The Engram rows cached in spare card memory | Volta sPLIt | not built: all host Engram work is about 0.12 ms per round, and a round needs all 288 rows, so a partial cache removes nothing | [levers](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/blob/main/docs/levers.md#did-not-work) |

## 2. It was faster and changed the answers

These are the expensive ones: real speed, refused.

| Idea | Build | Speed | Why refused | Record |
|---|---|---|---|---|
| Fused prefill attention with an exact softmax | Volta sPLIt | first token -11.4 % at 32K, -10.4 % at 128K | 1 of 8 greedy prompts identical; a confident token moved. 99.4 % of attention elements were bit-equal: the order of the score sums differed | [levers](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/blob/main/docs/levers.md#did-not-work) |
| Expert top-p pruning | Volta sPLIt | none: 78 tok/s against 99 | greedy output changed; the method drops experts the model would use | same |
| Tensor-core decode attention with a split sum | Volta sPLIt | -0.143 ms per round, not detectable | a confident token moved in 6 of 8 rows | same |
| An FP16-packed shared-expert kernel | Volta sPLIt | 25.30 to 25.15 ms per round | KL divergence up to 0.17 | same |
| Prompt chunks of 4,096 or 16,384 tokens in place of 8,192 | Volta sPLIt | 4,096 saved 5.6 % of first-token time at 32K; 16,384 was slower | the chunk size changes how a long prompt rounds; a confident token moved | same |
| The suggested tuning configuration of the dense kernel | Volta sPLIt | no gain | 2 of 8 identical. An autotuned configuration is a change of rounding | same |
| Fast prompt reading from compressed records | Volta Max uPLIft | first read near 200K tokens 86 to 119 s against 155 to 165 s | **adopted as the default anyway**, with the label "quality not established": it changes output from the first token and no closeness test has run on it | [README](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4#what-worked) |

The last row is not a refusal. It is listed here so that a reader sees the one place where speed was taken without a
quality verdict, and that the page says so.

## 3. The round got shorter and the tokens got fewer

Speculative decoding makes tokens per second the product of two things. Several ideas improved one and lost more on
the other.

| Idea | Build | Result | Record |
|---|---|---|---|
| Verify at most 3 drafts per round | Volta sPLIt | round -2.09 ms, accepted drafts -0.877 per round: -14.5 % tok/s | [levers](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/blob/main/docs/levers.md#did-not-work) |
| Probabilistic sharded drafts for thinking mode | Volta sPLIt | +0.05 to +0.15 accepted drafts, +0.5 to +1.55 ms per round: -3.1 to +0.6 % tok/s | same |
| Draft temperature 1.2 or 0.85 | Volta sPLIt | -2.9 % tok/s, or about 0 | same |
| 3 or 4 drafts per round | Volta sPLIt | refused at boot: the drafter checkpoint's block size is 5 | same |

## 4. The kernel was not limited by what the idea fixed

| Idea | Build | Result | What the kernel was limited by | Record |
|---|---|---|---|---|
| Activation prefetch and evict-first weight hints | Volta sPLIt | -0.04 ms: neutral | the expert kernels run at 24 % occupancy and stall on long scoreboard waits, not on memory | [levers](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/blob/main/docs/levers.md#did-not-work) |
| Warming the L2 cache with the next weight matrix on a side branch | Volta sPLIt | lost | the warm-up stretched to 21 to 28 microseconds beside the main kernel | same |
| A deeper prefetch in the dense kernel | Volta sPLIt | 1.2 to 5 microseconds slower in every variant | the existing double buffer is enough | same |
| 19 variants of the expert kernel | Volta sPLIt | none beat production; half-warp variants 25 to 30 % slower | instruction issue and latency. This led to the instruction diet, which won 0.87 ms | same |
| A lookup-free repack of the IQ1_M expert format | Volta sPLIt | 360 to 344 instructions per 256 weights (-4.4 % against a 20 % bar), +35.7 % bytes | not worth the memory. No-go before a build | same |
| 64 rows per weight load in the prompt-reading expert kernel | Volta Max uPLIft | the same bits, slower: 204 registers per thread, up to 1.39 times the old time per cycle | registers and shared memory on this chip | [README](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4#what-did-not-work-or-is-not-yet-good) |
| A tensor-core prompt-reading kernel that is not bit-equal | Volta Max uPLIft | 21.8 % slower on one card than the kernel it would replace | - | same |
| A new decode kernel for the NVFP4 experts | Volta Max uPLIft | closed without a build on eight cards: on one card the variants predicted 0.000 ms per round by a rule fixed before the data | the "1.45 ms" that started the work was an upper bound on headroom, not a prediction | same |
| 200 W in place of 150 W per card | Volta sPLIt | -0.06 ms per round: neutral | decode is not power-bound | [levers](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/blob/main/docs/levers.md#did-not-work) |

## 5. It was a scheduling or host change that cost more than it saved

| Idea | Build | Result | Record |
|---|---|---|---|
| Asynchronous scheduling with a pipeline and speculative decoding | Volta sPLIt | +1.678 ms per round slower once its crash was fixed. 1Cat's own [issue #852](https://github.com/1CatAI/1Cat-vLLM/issues/852) (zambalee) reports +55 % decode from turning it off on a two-stage V100 build (UPSTREAM) | [levers](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/blob/main/docs/levers.md#did-not-work) |
| CPU pinning, a spin-wait in the broadcast queue, 8 router warps | Volta sPLIt | +0.079 ms; +0.04 to +0.09 ms; -0.027 ms (neutral) | same |
| A fused layer transition, three variants | Volta sPLIt | slower or neutral | same |
| Releasing the allocator's cache more often (at 512 rows) | Volta 12X AmPLIfy | the same memory peak; larger turns' first token 10.9 s against 5.2 s | [README](https://github.com/ProprietaryLegal/Volta-12X-AmPLIfy-DeepSeek-V4.1-Flash-1017-NVFP4-12xV100-SXM2-PP3xTP4#what-did-not-work-or-is-not-yet-good) |
| Awaiting each hand-off between stages | Volta 12X AmPLIfy | no memory gain (31,664 against 31,628 MiB) | [README](https://github.com/ProprietaryLegal/Volta-12X-AmPLIfy-DeepSeek-V4.1-Flash-1017-NVFP4-12xV100-SXM2-PP3xTP4#what-did-not-work-or-is-not-yet-good) |
| Prompt chunk size chosen by prompt length | Volta Max uPLIft | the same output on only 25 of 31 cases and 522 MiB more card memory | [README](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4#what-did-not-work-or-is-not-yet-good) |
| Prompt slots as a way to serve a short request during a long read | Volta Max uPLIft | the short request starts in 4.1 to 4.3 s and not 56.8 s, but decodes at 15.3 tok/s until the read ends, and both requests end later than with one slot | [README](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4#requests-at-once) |
| More requests at once than pipeline stages | Volta 12X AmPLIfy | four together gave 6 % more than three (306.6 against 288.0 tok/s, summed) | [README](https://github.com/ProprietaryLegal/Volta-12X-AmPLIfy-DeepSeek-V4.1-Flash-1017-NVFP4-12xV100-SXM2-PP3xTP4#measured) |

## 6. It did not survive first contact with the hardware

| What | Build | What happened | Record |
|---|---|---|---|
| The four-card package of the later patches | Volta sPLIt | failed to boot in its smoke test; marked not current | [four-card.md](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/blob/main/docs/four-card.md) |
| The first three-stage boot | Volta 12X AmPLIfy | 15 of 15 answers wrong: the scheduler did not space decode steps by the pipeline depth | [README](https://github.com/ProprietaryLegal/Volta-12X-AmPLIfy-DeepSeek-V4.1-Flash-1017-NVFP4-12xV100-SXM2-PP3xTP4#what-did-not-work-or-is-not-yet-good); [the build's page](volta-12x-amplify.md#what-had-to-be-built) |
| Two engine changes that had passed CPU tests | Volta 12X AmPLIfy | one died on a table sized for four slots, one refused to boot in the model loader. Their CPU tests had not gone through the server's real entry point | [README](https://github.com/ProprietaryLegal/Volta-12X-AmPLIfy-DeepSeek-V4.1-Flash-1017-NVFP4-12xV100-SXM2-PP3xTP4#what-did-not-work-or-is-not-yet-good) |
| The first fast-reading variant | Volta Max uPLIft | stopped the host or a card on one long chat, three times of three, until the host's power delivery was changed | [README](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4#what-did-not-work-or-is-not-yet-good) |
| A quantized key cache with llama.cpp | DeepSeek-V4-Flash | healthy status, confident nonsense, no error | [README](https://github.com/ProprietaryLegal/deepseek-v4-flash-v100#the-bug-you-need-to-know-about--updated-it-affects-every-backend-not-just-volta) |

## What the losses have in common

1. **An isolated kernel benchmark was wrong in both directions.** It predicted 0.03 ms for a change that gave 0.42,
   and it showed gains for changes that lost in the server. No lever was adopted or dropped on a microbenchmark.
2. **"Not detectable" is not "no effect".** With a run-to-run spread of about 0.12 ms, a true 0.15 ms gain passes
   a 0.15 ms rule about 40 % of the time
   ([method](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/blob/main/docs/method.md#detection-power-and-drift)).
   Several rows above are honest unknowns, not proofs of zero.
3. **The largest gains were not kernels.** Keeping a request in the batch (-24.8 % at 128K), a transport setting
   (-14 % first token), a prompt cache that survives a changed reply (0.69 s against 31 s), not releasing memory
   that is asked for again (+10 to 13 % reading). Each was found by timing where the request waited.
