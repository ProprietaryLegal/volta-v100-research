# Volta 12X AmPLIfy: every optimization

Build: DeepSeek-V4.1-Flash-1017 on twelve V100-SXM2-32GB in three NVLink boards, three pipeline stages of four
tensor-parallel cards (layers 0 to 13, 14 to 27, 28 to 39). All 15,360 routed experts are in card memory at the
precision of NVIDIA's NVFP4 pack: none is compressed and none is read from host memory.
Repository: <https://github.com/ProprietaryLegal/Volta-12X-AmPLIfy-DeepSeek-V4.1-Flash-1017-NVFP4-12xV100-SXM2-PP3xTP4>.

This page is the long form of [part 4 of the front page](../index.md#part-4-every-optimization-build-by-build).
Its sources are the build's
[README](https://github.com/ProprietaryLegal/Volta-12X-AmPLIfy-DeepSeek-V4.1-Flash-1017-NVFP4-12xV100-SXM2-PP3xTP4#readme),
[docs/measurements.md](https://github.com/ProprietaryLegal/Volta-12X-AmPLIfy-DeepSeek-V4.1-Flash-1017-NVFP4-12xV100-SXM2-PP3xTP4/blob/main/docs/measurements.md),
[docs/history.md](https://github.com/ProprietaryLegal/Volta-12X-AmPLIfy-DeepSeek-V4.1-Flash-1017-NVFP4-12xV100-SXM2-PP3xTP4/blob/main/docs/history.md)
and [its time budget](https://github.com/ProprietaryLegal/Volta-12X-AmPLIfy-DeepSeek-V4.1-Flash-1017-NVFP4-12xV100-SXM2-PP3xTP4/blob/main/research/TWELVE-CARD-TIME-BUDGET.md).
Nothing here was measured for this page.

## How to read the tables

- Every figure is MEASURED on 2026-11-17 or early 2026-11-18 on one twelve-V100 server, clock ceiling 1,477 MHz,
  unless a row says ESTIMATE or UPSTREAM. The power limit was 150 W per card until late on the 17th and 200 W
  after; each row says which. Most 200 W figures are one run.
- A change is tested in a **window**: what serves is stopped, test arms boot one change at a time, a rule written
  before the data decides, and the serving build is restored and checked.
- The build is tuned for long-context agent work: one request at a time, prompts of 60,000 to 200,000 tokens that
  grow by small turns.

## Why twelve cards and three stages

One expert at NVFP4 is 4,874,240 bytes on each of its four cards. Forty layers of 384 experts are 278.9 GiB of
expert records. Eight 32 GB cards hold 254 GiB, so eight cards must compress some experts or read some from host
memory; twelve cards hold them all (ARITHMETIC).

Twenty layers of experts are 34.86 GiB per card, more than one card, so two stages do not fit. A stage must start
at a layer where the sparse attention index has a source, and the last stage also carries the drafter and the
output head. That leaves one cut that is legal and fits: layers 0 to 13, 14 to 27, 28 to 39 (ARITHMETIC). It is the
layout of [1Cat-vLLM #1031](https://github.com/1CatAI/1Cat-vLLM/pull/1031) (PLI Labs, October 2026), which ran at
about 3 tok/s on an engine without CUDA graphs (UPSTREAM, the pull request's figure).

## What had to be built

The engine change is published as install kits in the repository's
[kit/](https://github.com/ProprietaryLegal/Volta-12X-AmPLIfy-DeepSeek-V4.1-Flash-1017-NVFP4-12xV100-SXM2-PP3xTP4/tree/main/kit)
folder; the design is its
[docs/ARCHITECTURE.md](https://github.com/ProprietaryLegal/Volta-12X-AmPLIfy-DeepSeek-V4.1-Flash-1017-NVFP4-12xV100-SXM2-PP3xTP4/blob/main/docs/ARCHITECTURE.md).

| Change | What it is | Result | Changes answers? | Status |
|---|---|---|---|---|
| A third pipeline stage | The engine accepted two stages cut at layer 20 and nothing else. The kit teaches it three stages, a copy of one attention source on the last stage, and the Engram hand-over | with a quantized expert mix, the three-stage engine reads clean against a two-stage reference boot of the same experts (0 of 12 early divergences) | no, once the next row is in | adopted |
| **Scheduler spacing** | Under synchronous scheduling nothing spaced a request's decode steps by the pipeline depth. At two stages the call order happens to match. At three, the first decode step after every prompt ran one call before its data arrived, and the first stage embedded another request's tokens | without it: 15 of 15 answers wrong | **without it the build is wrong** | adopted. Found by running the real engine loop on a CPU |
| Stall guard for three stages | The slot scheduler counted two empty held steps as a stall, which three stages produce in normal work | found by a test before any server boot | - | adopted |
| No embedding on the middle stage; one Engram layer applied a stage earlier | The middle stage does not need the embedding table. The same idea as [1Cat-vLLM #1362](https://github.com/1CatAI/1Cat-vLLM/pull/1362) (Peuqui); the change is our own code | the fullest card's highest sample fell from 31,808 to 31,336 MiB at two slots | no | adopted |
| A card memory limit | A fuse on every worker at a line below the card's capacity | on 12 of 12 workers; no token and no memory mark changed | no | adopted |

## Adopted: configuration (the build's own settings against each other)

| Change | Gave | Cost | Changes answers? |
|---|---|---|---|
| Prompt chunks of 4,092 rows in place of 8,188 | The working memory of a prompt read on the fullest stage roughly halves. That made room for a third and a fourth slot: three slots at the small chunk peak at 30,052 MiB, two slots at the large chunk at 31,336 MiB | prefill 4 % slower on one read each (3,286 against 3,410 tok/s, 150 W) | a different chunk cut rounds a long prompt differently; judged by closeness to the reference, not by identity |
| Three slots, then four | A turn on a long chat after another chat was read: first token 2.1 to 2.6 s at three slots, 46.7 s at two (the chat's slot had been taken and 138,824 tokens were read again). Three together 288.0 tok/s summed; four together 306.6 (242.6 by wall clock) | about 345 MiB on the fullest card for the fourth slot. Four gave 6 % more than three: a pipeline step carries one request, so a slot does not batch | a slot changes no token (seven long prompts identical, 7 of 7) |
| 200 W in place of 150 W per card | prefill 3,284 to 3,733 tok/s on a 62,263-token prompt (+13.7 %, one run) | power and heat; decode moved by under 2 % | no |
| Checks on a test port before serving | Every check runs on a test port and writes a record of exactly which code and configuration passed. The serving boot sends no request and starts only on a matching record | a move or a return takes about 3 minutes without a server | - |

## Tested, not yet in what serves

| Change | Result | Changes answers? | Status |
|---|---|---|---|
| **Release the allocator's cache only above the chunk size** (`VLLM_V41_EMPTY_CACHE_TOKENS=8192`) | The engine released cached card memory after every prompt step of 2,048 rows or more, so each 4,092-row chunk gave its work memory back and asked for it again. With the higher threshold: prefill +12.6 % at 60,706 tokens (3,850 to 4,334 tok/s), +9.9 % at 93,027 tokens, +2.8 % at 247,953 tokens (200 W, one boot). Card memory in a four-chat rehearsal was not higher, although a higher mark had been predicted | identical on seven long prompts by the strict judge | passed every line of its rule; waits for the adopt boot |
| **The conversation store** (a chat's state saved to disk) | Every server test passed at four slots: the state check on twelve workers, a 130,000-token chat restored from disk after a restart with its state equal to the saved state and an identical answer, 0 waits over 2,976 saves. A 131,466-token turn restored after a restart reached its first token in 1.23 s, where a cold read took 43.84 s. Card memory with it on was within the sampler's resolution of off | a restored turn is not token-identical to a cold read of the same text (true of any cached turn on this engine) | not switched on: a paired on/off comparison of the first token comes first |
| Skip unused layers on the middle stage during early prompt chunks | not yet measured: its test arm was stopped by its own witness, which misread a warning line of the API server | - | open |
| A fixed split count in the dense multiply library (`TM_GEMM_TUNE=max_splits=1`) | first divergence from the reference at 96 tokens where the rule asked for 104 | changes rounding | INCONCLUSIVE at its quality line; stays out |
| A repaired token cache; the drafter's faster kernel | built and tested on CPU | - | not yet on the cards |

## Rejected

| Idea | Result | Why |
|---|---|---|
| Release the allocator's cache at 512 rows (more often) | the same memory peak; larger turns' first token 10.9 s against 5.2 s | It bought no memory and made turns twice as late |
| Await each hand-off between stages instead of overlapping it | 31,664 MiB against 31,628 for the plain arm | No memory gain |
| Three slots at the large prompt chunk | 31,628 MiB, 370 MiB under the limit, against 30,052 at the small chunk | Not chosen |
| One tensor-parallel group of eight cards with no pipeline (the same eight cards as two stages of four, one test window) | one request -3 %, three at once -41 %, long prompt reads -26 % | One stage has one decode step in flight, so requests take turns; every layer's exchange between the two boards crosses PCIe. It also cannot hold every expert at NVFP4 |
| Upstream's prompt pipeline across stages ([1Cat-vLLM #1482](https://github.com/1CatAI/1Cat-vLLM/pull/1482)) | nothing to port | This engine already keeps one prompt chunk per stage in flight |
| Six and eight slots | no verdict | The six-slot arm died on a table sized for four slots; fixed, then not re-run by choice |

## Where the time goes

MEASURED once, at 200 W, with the engine's own timers and a profiler that cost 5.67 % of a round
([the time budget](https://github.com/ProprietaryLegal/Volta-12X-AmPLIfy-DeepSeek-V4.1-Flash-1017-NVFP4-12xV100-SXM2-PP3xTP4/blob/main/research/TWELVE-CARD-TIME-BUDGET.md)).

**A decode round for one request, short chat: 26.25 ms with the profiler, 25.1 ms without.** The three stages work
one after another.

| Part of the round | ms | Share |
|---|---:|---:|
| Routed and shared experts | 7.70 | 29 % |
| Dense multiplies (projections) | 5.39 | 21 % |
| Drafter | 2.74 | 10 % |
| Attention | 2.24 | 9 % |
| Normalisation and mixing between sublayers | 1.71 | 7 % |
| Router | 1.25 | 5 % |
| Sparse attention index | 0.77 | 3 % |
| Four-card sums | 0.64 | 2 % |
| Output head and sampling | 0.64 | 2 % |
| Outside any kernel (launch gaps, host work, hand-off) | 3.78 | 14 % |

With context the round grows, and the growth is the index: 28.07 ms at about 128,000 tokens of context (index 2.78
ms), 30.98 ms at about 249,000 (index 4.47 ms).

**A 4,092-row prompt chunk: 793 ms on the first stage, 891 ms on the middle stage, 844 ms on the last** (median
forward time). On the first stage: experts, router and their sums 318 ms; attention 119 ms; output projection 104 ms;
normalisation 92 ms; query and key projections 65 ms; index 53 ms. The index is what makes the later stages slower
(142 and 222 ms), and it grows with context.

**A turn that adds about 900 tokens to a 128,617-token chat: 1.24 s to the first token.** 386 ms of it pass before
the scheduler sees the request (chat template, tokenization, hashing): about 3 microseconds per token of the whole
prompt, on every request, cache hit or not.

## Rounding: two findings about this engine family

A long prompt read in differently cut chunks does not give bit-identical state. Two sources were found, each proven
with a probe that needs no model:

1. **Large four-card sums go through NCCL in half precision**, in an order that depends on the buffer's size and the
   row's place in it. Small sums take the engine's own all-reduce, which adds in single precision and rounds once.
2. **The dense multiply library picks its split count from the chunk's row count.** A prompt's last chunk has a
   different row count from the others, so it is computed with a different split, and rows computed under different
   splits round differently. A one-card probe confirmed it in 25 seconds at all sixteen row counts a CPU model of
   the library's dispatch had predicted.

No token is shown wrong by either. They explain why "token-identical after a change of chunk size" is the wrong
test, and why a cached turn is not token-identical to a cold read of the same text.

## What went wrong, and what the independent review found

- **The first three-stage boot answered wrongly** (the scheduler spacing above). A diagnostic window run with the
  fault still in ended with four card errors (Xid 43) on the first stage.
- **Two kits passed CPU tests that did not go through the server's real entry point.** One died on its first
  requests on a label table sized for four slots; one refused to boot because the model loader dropped the tensors
  of a moved layer. Both now have tests through the real entry point.
- **Five test arms were stopped by the build's own checks, each meeting a real log of its kind for the first
  time:** an out-of-memory pattern that matched the memory limit's own announcement line (a good boot refused, a
  second boot paid for); a worker count doubled by a kit's extra log line; a witness that read an environment a
  renamed worker process blanks; a turn probe that replayed answers without their reasoning, so its turns were read
  cold; a witness that failed a boot on the API server's warning about an unregistered switch. The rule since: every
  pattern a check applies to a server log is first run on a real log of that kind of boot.
- **Several predictions were wrong** and are recorded as wrong.
- **An independent review by a second model, read-only, after the build was serving**, found:
  1. **The equality judge accepted cut-off answers**: it compared two answers only through the shorter one. The
     judges were made strict (equal length, equal ids, a missing key INCONCLUSIVE), and the three historical
     identity results were re-checked on whole sequences and stand.
  2. **A token cache returned wrong token ids** on a hit when an edit removed a special token at its cut, before it
     ever served. It was held, repaired and re-proven by two independent fuzz tests (one of 3,200 edits, 0
     sequences different from a full encode).
  3. A sum over requests is not elapsed throughput: four together were 306.6 tok/s summed and 242.6 by wall clock.
  4. The first reading of the time budget overstated one item: of the host's time freeing and allocating card
     memory per chunk, only about half coincided with an idle card. The prediction was cut to about 9 % before the
     test ran; the test then measured +9.9 % at 93,000 tokens.
  5. The estimate that ranked the work items had under-weighted decode. The order of work changed: decode for one
     request first.

## Not established

- Memory over hours of serving at four slots (at two slots the serving arm crept 58 MiB in half an hour).
- Answer quality in use.
- Anything at six or eight slots.
- More than one run of most 200 W figures.
