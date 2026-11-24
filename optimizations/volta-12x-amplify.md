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

**Update 2026-11-24.** The tables below are the record as of 2026-11-19. That evening (window x2) the owner had everything built switched on, the conversation store included, and the server has served with it since. It serves `PP3KIT11-over-cut` from 2026-11-24 with twenty adopted lines: the eleven switches below plus the store, the lean stage hand-off, padding-free expert records, the mHC kernel and their settings. The repository's [fifth release](https://github.com/ProprietaryLegal/Volta-12X-AmPLIfy-DeepSeek-V4.1-Flash-1017-NVFP4-12xV100-SXM2-PP3xTP4#readme) gives what serves now, its measured speeds and the levers closed since.

## How to read the tables

- Every figure is MEASURED on 2026-11-17 or early 2026-11-18 on one twelve-V100 server, clock ceiling 1,477 MHz,
  unless a row says ESTIMATE or UPSTREAM. The power limit was 150 W per card until late on the 17th and 200 W
  after; each row says which. Most 200 W figures are one run. Rows dated 2026-11-19 were measured at 175 W per
  card and a 1,452 MHz clock ceiling, the setting the server has run at since the afternoon of 2026-11-18.
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

## Adopted: engine changes (nine serving since 2026-11-18, the tenth and the eleventh since 2026-11-19)

Every row below was switched on by its own test window: the rule written before the data, one change per boot,
a reference boot of the same window, the seven accepted long prompts answered token for token, and a rehearsal of
four grown chats under the memory line. MEASURED on 2026-11-18 at 200 W per card unless the row says otherwise;
the server has run at 175 W per card and a clock ceiling of 1,452 MHz since the afternoon (the owner's setting after
a power trip at 200 W). The order is the order they went in; the tenth switch went in on 2026-11-19 and closed the
campaign's list; the eleventh, the fixed split count, went in the same afternoon as the first item of the plan's next
tier.

| Change | What it is | Result | Changes answers? |
|---|---|---|---|
| **Release the allocator's cache only above the chunk size** (`VLLM_V41_EMPTY_CACHE_TOKENS=8192`) | The engine released cached card memory after every prompt step of 2,048 rows or more, so each 4,092-row chunk gave its work memory back and asked for it again. Now only steps above 8,192 rows release it | prompt reads +12.6 % at 60,706 tokens (3,850 tok/s) and +9.9 % at 93,025, same memory peak (one boot pair, 200 W); the first switch adopted through the serving record | no (7 of 7) |
| **Skip the middle stage's unused layers during early prompt chunks** (`VLLM_V41_PP_MID_PREFILL_SKIP=1`) | While the first stage still reads a prompt's early chunks, the middle stage has nothing of that prompt to compute and skipped nothing; now it skips | alone: reads +10.4 % at 60,704 tokens, +16.9 % at 93,027, +31.0 % at 247,952, +49.6 % at 425,776 (4,206 / 4,322 / 4,301 / 4,125 tok/s against 3,810 / 3,696 / 3,284 / 2,757; one window, 200 W); the gain grows with length, and the prediction (+3 to +8 % at 94,000) was low by half and more. With the release above: +19.7 % / +22.1 % / +34.1 % / +53.2 % (4,562 / 4,515 / 4,405 / 4,223 tok/s; first token 13.3 / 20.6 / 56.3 / 100.8 s against 15.9 / 25.2 / 75.5 / 154.4). A turn beside a 248,000-token read: whole turn 10.43 s against 12.75 | no (7 of 7; 5 of 5 at the long turns) |
| **Submit the next step ahead of an empty one; no sleep after an answer** (`VLLM_V41_PP_EMPTY_AHEAD=1`, `VLLM_V41_PP_ANSWER_NO_SLEEP=1`) | The pipeline's loop waited on an empty step and slept between an answer and the next submission | the decode round 24.235 ms against 25.392 on a short chat (-4.55 %) and 26.985 against 28.164 at 128,000 tokens of context (-4.19 %); answer to submission 1.40 ms against 2.84; 0 empty steps in 64 of 64 steady steps | no (7 of 7) |
| **The drafter's faster kernel** (`VLLM_V41_DRAFT_GEMV2=1`) | The speculative drafter's matrix-vector product reads its weights once | with the two loop fixes: round -5.04 % short, -4.75 % at 128K, against the plain boot; after a one-step turn at 128K 170.0 tok/s (5.88 ms a token), after a cold read 137.9 tok/s (one request, median of 4) | no |
| **The token cache in the front end** (`VLLM_V41_PREFIX_TOKEN_CACHE=1`) | A chat is tokenized once; a later turn re-uses the ids of its unchanged prefix instead of tokenizing the whole chat again | the front end of a turn on a chat of about 137,000 tokens: 47.2 ms against 308.0 at the median (-256.8 ms; 0 of 20 turns not faster); a cold read's front not slower; the reply's text equal 27 of 27 cells | no |
| **The prompt-end hold** (`VLLM_V41_MULTI_SEQ_PROMPT_END_HOLD=128`) | While a long document is read in one slot, a short turn in another slot gets the pipeline at the end of the read's chunks instead of waiting for the whole read | a turn of 880 tokens beside a 248,000-token read: whole turn 5.12 s against 11.99 (-57 %; the line asked -50 %); the read's own first token 60.99 s against 62.66 (not slower); medians of 3 | no (7 of 7; 3 of 3) |
| **Turn chunks** (`VLLM_V41_PP_TURN_CHUNKS=3`) | A turn's new tokens are read as a head and three pieces so the three stages overlap them, instead of one block through the pipeline | first token of a turn alone on a 128,000-token chat (medians of 2): +880 tokens 1.131 s against 1.184; +2,048 1.471 against 2.061; +4,200 2.319 against 3.687 (63 %; the line asked at most 75 %); +10,000 3.623 against 4.963 | no (7 of 7; 5 of 5) |
| **The sparse-attention index on the tensor cores** (`VLLM_V41_INDEX_TC=1`) | The scorer that picks which compressed keys a token attends to, as a tensor-core kernel (two indexed layers on the first stage, three on each of the others) | the decode round at 128,000 tokens 25.829 ms against 26.633 (-0.804 ms, -3.0 %) and 25.719 against 26.677 after many steps (-0.958 ms, -3.6 %); short chats within 0.07 ms; measured in two windows (the first window's rule for its counter line was wrong and was corrected before the second); the reference's round varies about 0.4 ms boot to boot. The 128K read 4,587 tok/s against 4,582; four requests together, fresh text, 337.9 tok/s in sum against 316.6 | no: streamed text equal 22 of 22, 7 of 7 |
| **Two boots without the drafter answer alike** (the quality window) | Not a change: a check that the served configuration is deterministic across boots with the drafter off | eight forced replies (prompts near 4,200, 94,000, 128,000 and 252,000 tokens, each followed by a turn; 205 tokens at temperature zero) equal in text 8 of 8 across two independent boots | - |
| **The batched `wo_a` projection at the build's chunk size** (`VLLM_V41_WO_A_BATCHED_4K=1`; MEASURED 2026-11-19, 175 W) | The batched path of one attention projection in prompt reads admitted exactly 8,188-row chunks, so on this build's 4,092-row chunks it never ran; now it admits them | cold reads of 60,705 / 93,025 / 247,948 / 425,779 tokens: 4,663 / 4,741 / 4,531 / 4,309 tok/s against 4,476 / 4,559 / 4,358 / 4,028 in the same window without it (+4.2 / +4.0 / +4.0 / +7.0 %); memory peak 30,832 MiB, the same as without. In the final bench of 2026-11-19 a cold code read at 32K to 256K tokens was 4.8 to 5.0 % faster than in the bench of 2026-11-18 on the same build (4,657 tok/s at 128K) | rounding: against the accepted capture the first divergence came after a median of 58 tokens and the mean absolute log-probability difference was 0.0354 nats, inside the closeness line; the greedy text of a long answer can differ, and with it the drafter's acceptance and the decode rate of that answer, by up to about 9 % either way |
| **A fixed split count in the dense multiply library** (`TM_GEMM_TUNE=max_splits=1`; MEASURED 2026-11-19, 175 W) | The library that does the dense multiplies picks how many pieces to split a multiply into from the chunk's row count, so a prompt's last chunk (a different row count) was computed under a different split from the rest and rounded differently; now every chunk takes one split | cold reads of 60,706 / 93,028 / 247,953 / 425,783 tokens: 4,659 / 4,756 / 4,552 / 4,314 tok/s against 4,664 / 4,747 / 4,515 / 4,327 without it in the same window (-0.1 / +0.2 / +0.8 / -0.3 %); one fresh request's decode +1.3 %; memory peak 30,832 MiB, the same. In the bench that followed (the afternoon of 2026-11-19) the first and the second cold read of every document agreed within 0.15 s, where the two benches before had read the first legal or prose document 1 to 11 s slower than the second (a hypothesis, not a proof: the split search on a shape not yet seen was the cost, and the fixed count removes it) | rounding: against the accepted capture the answers were identical on 1 of 7 long prompts, the first divergence came after a median of 45 to 48 tokens and the mean absolute log-probability difference was 0.035 to 0.037 nats, inside the closeness line (50 and 0.0307 are the yardstick's); a cache hit is not token-identical to the cold read of the same text in general (identical 4 of 9; see "Rounding" below) |

What 175 W costs against 200 W (MEASURED in two windows, not one boot pair): a read of about 426,000 tokens took
152.0 s at 150 W against 100.8 s at 200 W; decode moved little. The 175 W figures of the whole set are the bench of the afternoon of 2026-11-19 (the eleven
switches), in [builds.md](../builds.md#volta-12x-amplify).

## Tested, not yet in what serves

| Change | Result | Changes answers? | Status |
|---|---|---|---|
| A fixed split count in the dense multiply library (`TM_GEMM_TUNE=max_splits=1`), first window (2026-11-17) | first divergence from the reference at 96 tokens where the rule asked for 104 | changes rounding | INCONCLUSIVE at its quality line then; adopted on 2026-11-19 in a window of its own with a scaled yardstick (the row in the adopted table) |

## Rejected

Every row below ran in its own window with a rule written before the data, or was closed by a measurement before a
window was spent on it. The four rows dated 2026-11-19 closed the campaign's list; the build serves without them.

| Idea | Result | Why |
|---|---|---|
| **The conversation store** (a chat's state saved to disk; MEASURED 2026-11-19, 175 W, the serving set of the day) | Every state check passed again at four slots: the dump, a restart on the saved set with exact answers after it, the memory equal to the store-off boot, 0 waits over 1,704 and 624 saves. Its timing line failed: in a four-chat rehearsal (30 requests, largest prompt about 153,600 tokens, four boots off / on / on / off) the median first token of the larger turns was 4.718 and 3.722 s with the store on against 3.596 and 3.590 s off, limit 3.696; the small turns 3.029 and 0.999 s against 2.586 and 2.302 (INCONCLUSIVE, the two on-boots straddle the limit) | Correct in every state check, slower at the first token with it on; the cause is not established (the two on-boots differ from each other more than the off-boots do). Stays off |
| **A leaner hand-off between stages** (fewer copies per step; MEASURED 2026-11-19, 175 W) | A first kit did not boot: it assumed a hidden size of 4,096 where the model's is 5,120. The fixed kit booted with twelve active hand-offs and answered token for token; the decode round after a cold read of about 128,000 tokens was 25.822 ms against 25.954 without it, 0.133 ms shorter where the rule asked at least 0.3; after a one-step turn 25.947 against 26.253, 0.306 ms shorter but with the two boots' spreads overlapping, unreadable | Under its line; the short-chat cells differed by 12 to 21 % only because the reference boot's short cells spread 24.7 to 44.1 ms, which is noise, not a gain |
| **The zero padding out of each native expert record** (576 of 640 columns are data; re-cut packs and a sidecar kernel; MEASURED 2026-11-19, 175 W) | Answers identical 7 of 7; 756 to 878 MiB less card memory per card by stage; but the decode round LONGER: 25.990 ms against 25.895 after a one-step turn at 128K (+0.095) and 25.743 against 25.588 after a cold read (+0.155), where the rule asked at least 0.3 ms shorter; short chats +0.27 and +0.90 % | The prediction (0.33 to 0.41 ms shorter) was wrong in sign: the smaller records did not shorten the round. The memory they free is real and recorded |
| **An exact four-card sum** (the large sums between the four cards of a stage added in one fixed order in single precision instead of the communication library's size-dependent order; MEASURED 2026-11-19, 175 W) | On the board, without a model, the exact sum cost 2.2 times the library's call on a large tile (2.632 against 1.180 ms, the four cards bit-identical). Served, it answered the seven accepted long prompts identically on 0 of 7, clean of early wrong tokens, but with the first divergence after a median of 13 tokens where the line asked at least 25 (the yardstick's 50 halved): its answers are farther from the reference than the fixed split count's, not nearer | An exact sum re-rounds every four-card sum of every layer at every row, and the reference was recorded under the library's arithmetic, so the distance measured is between two roundings of the whole network, not the chunk-dependence it was meant to remove. The speed line never ran. Whether an exact, chunking-independent answer is worth a new reference is the owner's decision; closed as built |
| Fused kernels for the native expert layer (five launches and a cast) | Not built: timed per layer on one card at six rows, this build's path 231.9 microseconds against upstream's fused kernels 241.7; the best width of a fused version 0.205 ms a round, under the 0.300 ms gate the plan set for building it | Closed at its timing gate; the existing path is already the faster |
| Whole 64-row passes in every prompt chunk (a 4,092-row chunk leaves a 60-row remainder on a slower path) | Measured from the profile at under 0.37 % of a chunk | Closed without a window: below the band a window can read |
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

Both sources were taken into windows on 2026-11-19. The split count was fixed and adopted (the eleventh switch);
the exact four-card sum was tried and rejected (the tables above). A cache hit was then measured against its own
cold read on nine prompts of 32K to 128K tokens (greedy, 32 tokens): identical on 4 of 9 (every prose prompt and
one code prompt), the first difference after a median of 25 tokens on the rest. The hit computes its tail rows in
another chunking over the cached blocks, through the four-card sums and the expert path, so the remaining
chunk-dependence lies outside the dense multiplies; whether it is worth a lever is an open item.

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
