# Volta Max uPLIft: every optimization

Build: DeepSeek-V4.1-Flash-1017 on eight V100-SXM2-32GB, two pipeline stages of four tensor-parallel cards. Every
routed expert is on the cards in decode; the most-used run at exactly the precision of NVIDIA's NVFP4 pack, the rest
from compressed IQ2-class records.
Repository: <https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4>.

This page is the long form of [part 4 of the front page](../index.md#part-4-every-optimization-build-by-build).
Sources: the build's [README](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4#readme),
its [measurements](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4/blob/main/docs/measurements.md),
its [history](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4/blob/main/docs/history.md)
and the [front-page archive of 2026-11-14](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4/blob/main/docs/front-page-archive-2026-11-14.md).
Nothing here was measured for this page.

## How to read the tables

- This build changed its measuring method from Volta sPLIt's. A change is tested in a **window**: the serving build
  is stopped, test arms are booted, a rule written before the data decides, and a server is restored. The rules are
  published in the repository's
  [docs/acceptance/](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4/tree/main/docs/acceptance).
- **Power limits changed between windows** (1,425, 1,475, 1,522 and 1,530 MHz clock ceilings; 150, 175 and 300 W).
  Compare inside a row, not across rows.
- "Token for token the same" means the new path was compared with the old one on the same boot or build and gave
  identical token ids. It says the change does not move answers. It does not say the build's answers equal the
  NVFP4 pack's.
- "Quality not established" means no closeness test has run on that configuration.

## The idea of the build

Everything except the routed-expert path is Volta sPLIt's engine and switch set. The new work answers one question:
how much of the routed work can run at the official precision with every expert still on the cards?

| Step | What it is | Result (MEASURED) | Changes answers? | Source |
|---|---|---|---|---|
| The exact fold | NVIDIA's NVFP4 pack stores an expert with one FP8 scale per 16 values. Folding it to MXFP4 (one power-of-two scale per 32 values, 4.25 bits per weight) loses nothing | 44,696,862,720 folded values bit-equal to the NVFP4 formula, relative error 0 (2026-11-09) | exact by proof | [Volta Full NVFP4, "What this is"](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100#what-this-is) |
| The native expert path | The folded experts run through the QPN kernel of [v100-skinny](https://github.com/dnv2003/v100-skinny) (Dennis Vertzyas) as packaged in 1Cat-vLLM by [#742](https://github.com/1CatAI/1Cat-vLLM/pull/742) (Peuqui): 4-bit weights, FP16 activations, FP16 tensor cores | one card: worst relative error 4.78e-4 against an FP32 reference; CUDA graphs replay bit-equal; 131 microseconds per layer at one row (one card of a four-card test machine, 150 W) | - | [Volta Full NVFP4, "What worked"](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100#what-worked) |
| Promotion by use per byte (`hot_iq`) | Every expert's compressed record goes on the cards first. Experts are then promoted to NVFP4 in order of use per byte until the next does not fit | first release: 6,087 of 15,360 experts at NVFP4; 79.0 to 85.0 % of decode routes ran on them (in-sample, on the four kinds of text the choice was made from) | the promoted experts are value-exact; the others are the compressed records | [README, "What this is"](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4#what-this-is) |
| Dropping the host tier | An earlier step kept part of the experts in host RAM (three tiers). General text was slow there because of host-RAM reads | general text 33.4 ms per round without a host tier against 58.0 to 59.4 ms with it (2026-11-09; speed only, that profile was not accepted) | - | [README, "What worked"](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4#what-worked) |
| Per-stage card reserve | One reserve for both stages ran out of memory on a 100K prompt. Reserves are now set per stage from memory peaks sampled during long reads | a 498,099-token prompt reads without an out-of-memory error | - | same |

### How the experts were chosen, and how the choice was judged

1. **Count routes.** The model's router picks 6 of 384 experts per layer for every token. Counters in the engine
   record which experts a stream of text actually uses. The counts were taken on four kinds of text: three public
   prompt sets, and the owner's own private workload (in his words, in
   [a public post](https://www.reddit.com/r/homelab/comments/1xusbar/lawyers_v100_homelab_update_2_the_8x_v100_box_now/):
   "'Used the most' was counted on a mix of four kinds of text. One is my own legal harness traffic, which is
   private and not published. The other three are public prompt sets in the repo."). The counts themselves are not
   published; to reproduce the build you count routes on your own text.
2. **Promote by use per byte.** Every expert's compressed record goes on the cards first. Experts are then swapped
   up to their NVFP4 records in order of use per byte until the next one does not fit. Packing the cheapest
   promotions first would have fitted more experts (6,377 to 6,379 against 6,087) but less-used ones, so it could
   not carry more of the work at NVFP4.
3. **Judge by closeness, properly.** The claim "closer to the NVFP4 pack" was tested, not assumed. The test was
   preregistered: items, scoring code and verdict rules fixed and hashed before any output was captured
   ([preregistration](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4/blob/main/quality/closeness-test/PREREGISTRATION.txt)).
   320 contexts and 513 MMLU questions, greedy, thinking off; the reference served on the same cards and run twice,
   so its own repeat is the noise floor (100 % identical). Results for the build of the first release, MEASURED
   2026-11-11: the same next token as NVIDIA's NVFP4 pack on **95.1 %** of compared tokens (95 % interval 94.4 to
   95.7); mean KL divergence from it 0.079 nats (a lower bound, over the top 20 tokens); the first token of every
   answer identical (native prompt reading); 1.6 % of multiple-choice answers different from the reference's;
   accuracy 83.6 %, not distinguishable from the reference's at this sample size.
4. **Say what the test does not show.** It measures closeness to NVIDIA's pack, which is itself a quantization of
   the original model. It is greedy, without thinking, at most 32 tokens per item. And it ran on the first release:
   the later default (two slots, fast prompt reading) has not been through it, which is why that default says
   "quality not established" everywhere.

## Adopted: speed

| Optimization | What it is | Gave (MEASURED) | Cost | Changes answers? | Source |
|---|---|---|---|---|---|
| Smaller stage-1 reserve (7.50 to 7.00 GiB, 2026-11-11) | Half a GiB more per card for NVFP4 experts on the second stage | 205 more experts at NVFP4 (6,292 of 15,360); decode routes on NVFP4 experts 80.3 to 86.4 % against 79.0 to 85.0 %; speeds unchanged within noise | less memory margin | passed the 12-prompt closeness check; the full closeness test was not rerun | [archive, update of 2026-11-11](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4/blob/main/docs/front-page-archive-2026-11-14.md#update-2026-11-11-a-faster-route-plan-kernel-and-a-larger-nvfp4-tier-map) |
| **Group-plan kernel** (windows 11 and 11c) | One launch builds the plan of which expert launches run, in place of a chain of tensor operations | decode round 33.9 to 29.2 and 28.5 ms: mean -4.94 ms against a noise floor of 1.14 (window 11c, 1,475 MHz). The first window showed -4.12 ms against a floor of 3.43 and was "no measurable change" by its rule; one replication had been declared in advance | none | token for token the same | same; rule: [WINDOW11-accept-rule.md](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4/blob/main/docs/acceptance/WINDOW11-accept-rule.md) |
| Fused tail for the compressed experts (windows 14, 14c) | One launch does the weighted sum of the compressed experts' step | about 1.3 ms less per decode round on second runs; by the first-run rule: no measurable change (first runs of a boot carry 2 to 3 ms of noise) | none | token for token the same | [archive, update of 2026-11-12](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4/blob/main/docs/front-page-archive-2026-11-14.md#update-2026-11-12-two-prompt-slots-by-default-fast-prompt-reading-and-four-kernels) |
| SwiGLU kernel in decode (window 12) | One launch computes the activation of the NVFP4 experts | about 1.3 ms less per decode round on second runs; by the first-run rule: no measurable change | none | token for token the same | same |
| Two prompt-reading kernels (window 17) | A one-launch SwiGLU, and an expert kernel that handles 32 token rows per block of weights it loads instead of 16 (RB 4) | second reads of a 215,847-token prompt 4.43 s shorter (89.3 to 89.8 s against 93.7 to 94.2 s); first reads: no measurable change (noise floor 11.6 s) | 48 MiB more peak memory on the fullest card | the same output bits (one card, then token-exact on eight) | same |
| **Fast prompt reading** (window 13b; default since 2026-11-12) | While a prompt is read, each compressed expert is computed from its compressed record on the card, not from its NVFP4 record staged from host RAM | near 200K tokens a first read took 86 to 119 s against 155 to 165 s for native reading (another day and build) | **it changes the output from the first token: quality not established.** Its window made no verdict: the native arm was not run and the memory check failed by 64 MiB | changes output | same |
| **Prompt slots** (windows 15 and 18; two slots the default since 2026-11-12) | The server keeps two or three requests' prompts and decodes them together | 2026-11-13, 300 W: two together 158.7 tok/s in total (1.33 times one stream); three together on three slots 176.2 (1.48 times) | each slot costs NVFP4 experts: 6,203, 5,936 and 5,650 of 15,360 at one, two and three slots. A short request sent during a long read starts sooner but decodes at 15.3 tok/s until the read ends, and both end later than with one slot | every request gave the same token ids as the same prompt alone. Quality of the slot profiles: not established | [README, "Requests at once"](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4#requests-at-once) |

## Adopted: the prompt cache

| Optimization | What it is | Gave (MEASURED) | Changes answers? | Source |
|---|---|---|---|---|
| **Snapshot hit** (window rh, 2026-11-13) | In a chat with tools, the previous reply does not come back token for token (the tool-call parser and the chat template each add two line feeds), so the server read the whole chat again. The server now keeps a copy of each slot's sliding-window rows in host memory at the end of each prompt (81.8 MB for two slots, no card memory) and resumes there | first token against a cold read: **0.690 s against 30.987 s at 100,340 tokens**; 0.706 against 13.430 s at 40,550 tokens | the same output as an ordinary cache hit at the same position in the same boot (193 readings, 0 differences) | [README, "Prompt cache"](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4#prompt-cache) |
| One log line per request on the cache decision (window px) | The scheduler writes numbers and reason words for each request: whether the cache was used, how much, and if not, why | makes cache misses diagnosable; no text or token ids are logged | none | same |
| System notes keep their place; top-k segments of 1,024 (window 16) | The two request-path switches of Volta sPLIt | a follow-up turn that adds a system message to a 30,779-token chat reused 30,752 tokens: 0.45 s against 14.84 s | token for token the same | same |

A finding of this work that applies to any engine with a prefix cache: at near-tied tokens a greedy reply can differ
between a cold read of a prompt and a read that resumes from a cache hit, and between hits at different positions.
A token-exact test of cache reuse must compare a hit with a hit at the same position in the same boot.

## Did not work, or is not yet good

| Item | Result (MEASURED) | Why, or what follows |
|---|---|---|
| A decode kernel for the NVFP4 path (closed without a change) | An earlier "1.45 ms per round" was an upper bound on headroom, not a prediction. On one card the candidate launch variants predicted 0.000 ms per round by a rule fixed before the data, and the remaining fold 0.066 ms, below what an eight-card run can resolve | No eight-card window was spent on it |
| An expert kernel with 64 rows per weight load (RB 8) | the same bits, but slower: 204 registers per thread and 41,728 bytes of shared memory per block; up to 1.39 times the old kernel's time per cycle | Not shipped |
| A tensor-core prompt-reading kernel whose output is not bit-equal | on one card 21.8 % slower than the RB 4 kernel it would replace; its rule asked for 10 % faster | No go |
| Prompt chunk size chosen by prompt length | the same output on 25 of 31 cases, and 522 MiB more on the stage-1 cards | Rejected: its timing effect showed only at about 6.5K to 9.5K new tokens |
| A draft model on a separate GPU over the network | one draft round took 42.6 ms at the median against a bar of 4.5 ms; the draft head alone runs in 1.7 ms there | Not admitted: the network round trip is the cost, not the head. On these cards a decode round is about 25 ms and the local drafter is 0.64 ms of it, so even a 2.8 ms remote step would cost about 8 % of decode speed. What moving the drafter off the V100s would buy is card memory, not time. This row is the network case only; it says nothing about a drafter on another card in the same host |
| The first prompt-reading variant | its first read of one long chat stopped the host or a card three times out of three | After a change to the host's power delivery it read nine of nine long chats; cause not proven |
| Compressed records are not official precision | on synthetic inputs an expert's compressed record is far from its official record (relative error 0.60 to 0.69 in three layers) | This is why the build reports the share of decode routes that run at NVFP4 |
| The promoted set is fixed | It is chosen once, from four kinds of text | Text unlike those may route more often to compressed records |
| The long tool loop's last request | ran to its 32,000-token limit inside thinking in 5 of 8 runs across the builds of this repository | Open |
| Memory on one bench boot | the cards of stage 1 reached 32,000 MiB, 2 MiB over the build's own line, with no out-of-memory error | 32 MiB more reserve on stage 1 (window m32) answered it |
| The first-tally profiles, the three-tier mode, four slots | measured, not accepted | [history](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4/blob/main/docs/history.md) |

Source of the rows: [README, "What did not work or is not yet good"](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4#what-did-not-work-or-is-not-yet-good).

## What this build teaches

1. On eight cards the routed experts do not all fit at native precision. Choosing which experts get that
   precision, and keeping every expert on the cards, decided more than any kernel: without a host tier, general
   text went from 58.0 to 59.4 ms per round down to 33.4.
2. First runs after a boot carry 2 to 3 ms of noise per round on this engine. Three kernels here were "no
   measurable change" by a first-run rule and about 1.3 ms faster on second runs. A rule must say which run is the
   reading before the data exists.
3. A prompt cache is worth more than any kernel on a long chat (0.69 s against 31 s), and it breaks on details far
   from the GPU: two line feeds, a system message, the order of chats.
