# Volta 12X AmPLIfy: every optimization

Build: DeepSeek-V4.1-Flash-1017 on twelve V100-SXM2-32GB in three NVLink boards, three pipeline stages of four
tensor-parallel cards (layers 0 to 13, 14 to 27, 28 to 39). All 15,360 routed experts are in card memory at the
precision of NVIDIA's NVFP4 pack.

**Repository: not yet published.** Every figure on this page comes from PLI Labs' private working record of the
build (its plan, with each test's rule, prediction and result; its measurement file; its time-budget analysis). The
figures are MEASURED on 2026-11-17 on one twelve-V100 server unless a row says ESTIMATE or UPSTREAM. The power limit
was 150 W per card until late that day and 200 W after, with a 1,477 MHz clock ceiling throughout; each row says
which. Most 200 W figures are one run. When the build's repository is published this page will link each row to its
record.

## Why this build exists

One expert at NVFP4 is 4,874,240 bytes on each of its four cards. Forty layers of 384 experts are 278.9 GiB of
expert records. Eight cards hold 254 GiB (ARITHMETIC). So an eight-card build must compress some experts
([Volta Max uPLIft](volta-max-uplift.md)) or read some from host memory
([Volta Full NVFP4](volta-full-nvfp4.md), at a cost of about a millisecond per expert per layer). Twelve cards hold
them all.

Twenty layers of experts are 34.86 GiB per card, more than one card, so two stages do not fit either. A stage must
start at a layer where the sparse attention index has a source, and the last stage also carries the drafter and the
output head. That leaves one cut that is legal and fits: layers 0 to 13, 14 to 27, 28 to 39 (ARITHMETIC). It is the
layout of [1Cat-vLLM #1031](https://github.com/1CatAI/1Cat-vLLM/pull/1031) (PLI Labs, October 2026), which ran at
about 3 tok/s on an engine without CUDA graphs (UPSTREAM, the pull request's figure).

## What had to be built

| Change | What it is | Result | Changes answers? | Status |
|---|---|---|---|---|
| A third pipeline stage | The fast eight-card engine accepted two stages cut at layer 20 and nothing else. The change teaches it three stages, a copy of one attention source on the last stage, and the Engram hand-over | with the eight-card build's own experts, the three-stage engine matches the eight-card build (0 of 12 early divergences) | no, once the next row is in | adopted |
| **Scheduler spacing** | Under synchronous scheduling nothing spaced a request's decode steps by the pipeline depth. At two stages the call order happens to match. At three, the first decode step after every prompt ran one call before its data arrived, and the first stage embedded another request's tokens | without it: 15 of 15 answers wrong (token 0 right, junk for three to five tokens, then the right answer). With it: clean | **without it the build is wrong** | adopted. Found by running the real engine loop on a CPU, not by reading code |
| Stall guard for three stages | The slot scheduler counted two empty held steps as a stall, which three stages produce in normal work | found by a test before any server boot | - | adopted |
| No embedding on the middle stage; one Engram layer applied a stage earlier | The middle stage does not need the embedding table. Same idea as [1Cat-vLLM #1362](https://github.com/1CatAI/1Cat-vLLM/pull/1362) (Peuqui); our change is our own code | the fullest card's highest memory sample fell from 31,808 to 31,336 MiB | no | adopted |
| A card memory limit | A fuse on every worker at a line below the card's capacity | on 12 of 12 workers; no token and no memory mark changed | no | adopted |

## Adopted: configuration

| Change | Gave | Cost | Changes answers? |
|---|---|---|---|
| Prompt chunks of 4,092 rows in place of 8,188 | The working memory of a prompt read on the fullest stage roughly halves (2,675 MiB above the warm level against 1,385). That makes room for a third and fourth slot: three slots at the small chunk peak lower (30,052 MiB) than two slots at the large chunk (31,336 MiB) | prompt reading 4 % slower on one read each (3,286 against 3,410 tok/s, 150 W) | a different chunk cut rounds a long prompt differently; judged by closeness to the reference, not by identity |
| Three slots, then four | Three requests together: 288.0 tok/s summed, against 198.2 for the eight-card build in the same window with the same tool. A turn on a long chat after another chat was read: first token 2.1 to 2.6 s at three slots against 46.7 s at two (the chat's slot had been taken and 138,824 tokens were read again). Four together: 306.6 tok/s summed | about 345 MiB on the fullest card for the fourth slot. Four requests gave only 6 % more than three: a pipeline step carries one request, so a slot does not batch. By wall clock the four-together run was 242.6 tok/s, not 306.6 | a slot changes no token (seven long prompts identical, 7 of 7) |
| 200 W in place of 150 W per card | prompt reading 3,284 to 3,733 tok/s on a 62,263-token prompt (+13.7 %, one run) | power and heat; decode moved by under 2 % | no |
| Checks on a test port before serving | Every check runs on a test port and writes a record of exactly which code and configuration passed. The serving boot sends no request and starts only on a matching record | a move to the serving port took 3 minutes without a server | - |

## Tested, 2026-11-17 evening (200 W, one boot per variant, one run)

| Change | Result | Changes answers? | Status |
|---|---|---|---|
| Release the allocator's cache only above the chunk size | The engine released cached card memory after every prompt step of 2,048 rows or more, so each 4,092-row chunk gave its work memory back and asked for it again. Raising the threshold to 8,192 rows: reading +12.6 % at 60,706 tokens (3,850 to 4,334 tok/s), +9.9 % at 93,027 tokens, +2.8 % at 247,953 tokens. Card memory in a four-chat rehearsal was not higher (30,394 against 30,446 MiB), although a higher mark had been predicted | identical on seven long prompts by a strict judge | passed every line of its rule; to be installed with the next accepted set |
| Skip unused layers on the middle stage during early prompt chunks | The last stage already skips prompt chunks whose rows cannot reach it; the middle stage still computed its last seven layers for them | not measured | - | open: its test arm was stopped by its own witness, which misread a warning line of the API server. The switch itself was active |
| A fixed split count in the dense multiply library | See "rounding" below | first divergence from the reference at 96 tokens where the rule asked for 104 | changes rounding | INCONCLUSIVE at its quality line; stays out |

## Rejected

| Idea | Result | Why |
|---|---|---|
| Release the allocator's cache at 512 rows (more often) | the same memory peak (31,334 against 31,334 MiB); larger turns' first token 10.9 s against 5.2 s | It bought no memory and made turns twice as late |
| Await each hand-off between stages instead of overlapping it | 31,664 MiB against 31,628 for the plain arm | No memory gain |
| Three slots at the large prompt chunk | 31,628 MiB, 370 MiB under the limit, against 30,052 at the small chunk | Not chosen |
| One tensor-parallel group of eight cards, no pipeline (built and measured on eight cards) | against two stages of four on the same cards: one request -3 %, three at once -41 %, long reads -26 % | One stage has one decode step in flight, so requests take turns; every layer's exchange between the two boards crosses PCIe |
| Upstream's prompt pipeline across stages ([1Cat-vLLM #1482](https://github.com/1CatAI/1Cat-vLLM/pull/1482)) | not needed | This engine already keeps one prompt chunk per stage in flight (read in the code and confirmed with a CPU model of the scheduler) |
| Six and eight slots | no verdict | The six-slot arm died on a table sized for four slots; fixed, then not re-run by choice |

## Where the time goes

MEASURED once, at 200 W, with the engine's own timers and a profiler that cost 5.67 % of a round.

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

## Rounding: two findings that apply to the eight-card builds too

A long prompt read in differently cut chunks does not give bit-identical state. Two sources were found, each proven
with a probe that needs no model:

1. **Large four-card sums go through NCCL in half precision**, in an order that depends on the buffer's size and the
   row's place in it. Small sums take the engine's own all-reduce, which adds in single precision and rounds once.
   The NCCL sum differed from the single-precision sum in 32.89 % of elements in the probe.
2. **The dense multiply library picks its split count from the chunk's row count.** A prompt's last chunk has a
   different row count from the others, so it is computed with a different split, and rows computed under different
   splits round differently. A one-card probe confirmed it at all sixteen row counts a CPU model of the library's
   dispatch had predicted.

No token is shown wrong by either. They explain why "token-identical after a change of chunk size" is the wrong
test, and why a cached turn is not token-identical to a cold read of the same text.

## What went wrong, and what the independent review found

- **The first three-stage boot answered wrongly** (the scheduler spacing above). A diagnostic window run with the
  fault still in ended with four card errors (Xid 43) on the first stage.
- **Two engine changes passed CPU tests that did not go through the server's real entry point.** One died on its
  first requests on a label table sized for four slots; one refused to boot because the model loader dropped the
  tensors of a moved layer. Both now have tests through the real entry point.
- **A rule was mis-scaled.** One window closed a configuration on a long-prompt limit taken from short prompts.
  Later measurement showed that a 64-row shift of the chunk cut moves long-prompt answers as far as halving the
  chunk does. The configuration was accepted in the next window.
- **The test driver's own checks spoiled test arms five times.** Each time a check met a kind of log line or
  request it had never been run against: an out-of-memory pattern that matched the memory limit's own announcement
  line (a good boot refused, a second boot paid for); a worker count doubled by an engine change's extra log line
  (the window stopped at its first boot); a witness that read each worker's process environment, which a renamed
  process blanks; a turn probe that replayed answers without their reasoning, so its turns missed the cache and
  measured cold reads; a witness that failed a boot on the API server's warning about an unregistered switch.
  The rule since: every pattern a check applies to a server log is first run on a real log of that kind of boot,
  and a new change's new log lines go through every pattern before its first window.
- **Two memory predictions were wrong** and are recorded as wrong in the plan.
- **An independent review by a second model, read-only, after the build was serving**, found:
  1. A token cache under construction returned wrong token ids on a hit when an edit removed the special token at
     its cut. The check in use compared token counts and could not have seen it. The cache was repaired and
     re-proven on complete id sequences.
  2. **The equality judges accepted cut-off answers**: answers were compared only through the shorter one. Removing
     one token from each of twelve real references still gave "identical 12 of 12". Both judges were made strict
     (equal length, equal key set, a missing or unstable key INCONCLUSIVE). The three historical identity results
     were re-checked on whole sequences and stand.
  3. The first reading of the time budget overstated one item: of the host's time freeing and allocating card
     memory per chunk, only about half coincided with an idle card. The prediction was cut from 16 to 19 % to
     about 9 % before the test ran; the test then measured +9.9 % at 93,000 tokens.
  4. A sum over requests is not elapsed throughput (306.6 against 242.6 tok/s), and the 200 W figures are one run.
  5. The estimate that ranked the work items had under-weighted decode. The order of work changed.

## Not established

- Memory over hours of serving at four slots.
- Answer quality in use.
- Anything at six or eight slots.
- More than one run of most 200 W figures.
