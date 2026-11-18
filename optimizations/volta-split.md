# Volta sPLIt: every optimization

Build: DeepSeek-V4.1-Flash-1017 on eight V100-SXM2-32GB, two pipeline stages of four tensor-parallel cards, every
routed expert from a compressed IQ2-class record on the cards.
Repository: <https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4>.

This page is the long form of [part 4 of the front page](../index.md#part-4-every-optimization-build-by-build). Its
source is the build's own lever log,
[docs/levers.md](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/blob/main/docs/levers.md),
which holds every pair of every test. Nothing here was measured for this page.

## How to read the tables

- **Conditions of every figure**, unless the row says otherwise: MEASURED 2026-11-03 to 2026-11-05 on eight
  V100-SXM2-32GB at 150 W per card and a 1,522 MHz clock ceiling; one request at a time; the A/B harness of the
  build (each arm a fresh server boot, arms in the order OFF ON ON OFF, small levers in eight arms).
- **The unit is milliseconds per verify round**, decode only. A round is one pass of the target model that checks
  the drafter's five proposed tokens and emits one token plus every accepted draft. Tokens per second moves with
  the text (code accepts more drafts than prose); the round time is the engine's own speed. Negative is faster.
- **The baseline**, the plain two-board port with no lever: 31.1 ms per round at temperature 0 (99.2 tok/s on the
  fixed test set), 31.7 ms at temperature 1.
- **"Changes answers?"** has three values. *Bitwise*: the new code computes the same bits. *Near-tie gate*: the
  output may differ in its last bits, and the change was admitted only because eight greedy prompts came back
  identical or differed only where two candidate tokens were within a small margin fixed in advance. *Changes
  output*: the change failed that gate and was not installed.
- **"Ours"** means PLI Labs wrote it. Everything else names its source.

## Adopted: decode (the round a token is written in)

| Optimization (switch) | What it is | Gave | Cost | Changes answers? | Source and code |
|---|---|---|---|---|---|
| Index reduce on the live prefix (`INDEX_REDUCE`) | The sparse-attention index adds score rows across the four cards every round. It added a table sized for the maximum context. Now it adds only the part in use, through the engine's own all-reduce instead of NCCL | about -0.85 ms per round | none found | exact; it also made sampled output identical across restarts | ours; merged upstream as [1Cat-vLLM #1386](https://github.com/1CatAI/1Cat-vLLM/pull/1386) |
| Double-buffered dense mat-vec (`DMV_X`) | The dense matrix-times-vector kernel loads the next block of weights while it multiplies the current one | -0.42 ms, 6 of 6 pairs faster. The isolated benchmark had predicted 0.03 to 0.07 ms | none | bitwise | ours, built on #1277's kernel; merged as [#1390](https://github.com/1CatAI/1Cat-vLLM/pull/1390) |
| Coupled sampled drafts (`COUPLED_DRAFTS`) | At a temperature above 0 the drafter and the verifier share their random draws, so drafts agree with the target far more often | accepted drafts per round 1.85 to 2.25 at temperature 1: +14 % tok/s | about 0.3 ms of overhead per round at temperature 1, none at 0 | exact (the text keeps the target's distribution) | the idea is [TensorFold #25](https://github.com/Elevatormusic/TensorFold/pull/25); our port merged as [#1375](https://github.com/1CatAI/1Cat-vLLM/pull/1375) |
| Draft head and sampler on vocabulary shards (`MARKOV_SHARD`, `COUPLED_SHARD`, `COUPLED_LOCAL`) | Each card works on its quarter of the vocabulary inside the captured graph | -0.2 ms at temperature 0; -0.23 and -0.12 to -0.24 ms at temperature 1 | none | exact | ours; open as [#1377](https://github.com/1CatAI/1Cat-vLLM/pull/1377) |
| Expert plan inside the quantize launch (`PLAN_FUSE`) | The list of which experts run is built by the kernel that already runs before them, not by a launch of its own | -0.36 ms (temperature 0), -0.27 (temperature 1). The isolated benchmark saw no gain | none | bitwise | ours; merged as [#1383](https://github.com/1CatAI/1Cat-vLLM/pull/1383) |
| Fast routed-expert kernels (`EXPERT_FAST`) | New kernels for the experts' down projection and gate/up projection | down -0.68 to -0.84 ms; gate/up a further -0.53 to -0.82 ms | none | bitwise against the reference kernel; sanitizers clean | ours; merged as [#1384](https://github.com/1CatAI/1Cat-vLLM/pull/1384) |
| Fused attention prologue and epilogue (`FUSE_ATTN_PREP`, `FUSE_ATTN_POST`) | The small steps before and after decode attention become one launch each | -0.19 ms | none | bitwise | ours; open as [#1389](https://github.com/1CatAI/1Cat-vLLM/pull/1389) |
| Expert-kernel instruction diet (`EXPERT_DIET=3`) | Profiling showed the expert kernels were limited by instructions issued per weight byte, not by memory. The diet removes instructions | **-0.871 ms** | none | near-tie gate (7 of 8 identical plus one known near-tie); sanitizers 0 errors | ours; patch [0050](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/blob/main/engine/production/0050-expert-diet.patch); merged as [#1385](https://github.com/1CatAI/1Cat-vLLM/pull/1385) |
| Router floor (`ROUTE_FAST`, 4 warps) | A faster expert router | -0.165 ms, inside that session's drift: **not a significant win**. Kept under the rule for upstream-backed changes that are correct and inconclusive | none | greedy output bitwise | from our DGX Spark work |
| Drafter projection on shards (`DRAFT_MP_SHARD`) | Every card of the drafter's board read the drafter's full 78.6 MB projection every round. Now each computes its quarter | -0.130 ms, 8 of 8 pairs faster. The automatic rule called it "not detectable" and rolled it back; it was adopted by hand because it is bitwise and every pair was faster | none | bitwise | ours; patch [0054](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/blob/main/engine/production/0054-draft-mp-shard.patch); open as [#1374](https://github.com/1CatAI/1Cat-vLLM/pull/1374) |
| **Model Runner V2** (`VLLM_USE_V2_MODEL_RUNNER=1`) | vLLM's newer model runner as carried by 1Cat-vLLM, with the DSpark path of [#1221](https://github.com/1CatAI/1Cat-vLLM/pull/1221) (Peuqui); our part is the DeepSeek-V4.1 port | **-2.759 ms per round** (120.5 to 134.2 tok/s at temperature 0) | at one prompt length, 32K, it is 1.4 ms per round slower than the old runner; cause unknown. A sampled request that asks for log-probabilities draws a different (equally valid) sample | bitwise in every arm | patch [0055](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/blob/main/engine/production/0055-mrv2.patch); open as [#1379](https://github.com/1CatAI/1Cat-vLLM/pull/1379) |
| All-reduce push for this model's message size (`SUM2_PUSH_ANY`) | The sum of the shared and routed experts is folded into the four-card all-reduce. As a setting alone it lost 0.28 ms, because the model's 61,440-byte message was not admitted to the fast path. The rebuilt admission lets it in | -0.180 ms (eight arms, 95 % interval -0.278 to -0.081) | none | sampled texts identical | ours, on 1Cat's custom all-reduce; patch [0056](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/blob/main/engine/production/0056-allreduce-sum2-push.patch); merged as [#1382](https://github.com/1CatAI/1Cat-vLLM/pull/1382) |
| Top-k segments of 1,024 (`TOPK_SEG=1024`, 2026-11-11) | The index's top-k selection cut its score row into segments of 8,192; segments of 1,024 run more of it in parallel | -0.17 ms per round at 8K, -0.31 at 32K, -0.33 at 64K, -0.40 at 128K (clock ceiling 1,477 MHz). It passed its rule by 0.01 ms on a declared repeat window after missing by 0.01 ms; both windows are published | none | byte-identical | ours; patch [0059](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/blob/main/engine/production/0059-topk-segment-width.patch) |

## Adopted: long context and prompt reading

| Optimization (switch) | What it is | Gave | Cost | Changes answers? | Source and code |
|---|---|---|---|---|---|
| Keep the request in the batch (`PP_KEEP_BATCH`) | With two pipeline stages every other step is empty for a given stage. The request's token history was resent and rebuilt after each empty step. Now it stays | decode round -9.7 % at 32K, -17.4 % at 64K, **-24.8 % at 128K** (84 to 111 tok/s) | none | texts identical | our port of [1Cat-vLLM #1263](https://github.com/1CatAI/1Cat-vLLM/pull/1263) (Peuqui) to the old runner; merged as [#1373](https://github.com/1CatAI/1Cat-vLLM/pull/1373) |
| **Prefill skip** (`PREFILL_SKIP_ROWS=3072`) | Early prompt chunks skip the second-stage layers that hold only short sliding windows, for rows those windows can never reach | first token **-8.8 % at 32K, -20.6 % at 150K, -31.5 % at 300K** | none | identical output at every length | ours; design reference: the replay rows of [philbert440/deepseek-v4.1-v100](https://github.com/philbert440/deepseek-v4.1-v100); patch [0051](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/blob/main/engine/production/0051-prefill-skip.patch); open as [#1376](https://github.com/1CatAI/1Cat-vLLM/pull/1376) |
| Larger hot ring for the key-value cache (`KV_HOT_POSITIONS=524288`) | More of the cache stays on the cards (the default is 32,768 positions) | -1.09 ms per round at 300K, -0.25 at 150K | card memory | identical output | our setting of a switch that #1277 ships |
| **Tiled decode index scorer** (`INDEX_TILED`) | The kernel that scores which earlier tokens to attend to walks only the live blocks of keys and stages the query once per block | decode round **-7.0 % at 150K, -13.7 % at 300K**; not detectable at short context | none | bitwise by construction; 51,090 test cases, 0 mismatches | ours; patch [0052](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/blob/main/engine/production/0052-index-tiled.patch); merged as [#1372](https://github.com/1CatAI/1Cat-vLLM/pull/1372) |
| **Fused prefill index scorer** (`INDEX_FUSE=ilp4`) | During prompt reading the index scored each tile with four library calls and read its score tile three times. One kernel now keeps the scores in registers | first token **-4.9 % at 150K, -9.5 % at 300K**; nothing at 32K | none | near-tie gate PASS; outputs identical. No summation order was found that is bitwise-equal to the old path | ours; ideas from [#1280](https://github.com/1CatAI/1Cat-vLLM/pull/1280) and [#1282](https://github.com/1CatAI/1Cat-vLLM/pull/1282) (Peuqui) and philbert440's fused indexer; patch [0053](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/blob/main/engine/production/0053-index-fuse.patch); open as [#1388](https://github.com/1CatAI/1Cat-vLLM/pull/1388) |
| **Host hand-off between the boards** (`NCCL_P2P_LEVEL=NVL`) | NCCL then uses card-to-card transfers only inside a board, and the hand-off between the two boards takes the host path. Found by a step timer that showed later prompt chunks waiting on the hand-off | first token **-13.4 % at 32K, -14.2 % at 128K**; decode unchanged | none. One environment variable | bitwise | an NCCL setting, found by our stage-split profile |
| **Prefill expert GEMM** (`PREFILL_IQ_GEMM`) | 44 % of prompt-reading time on the first stage was the expert layers. The kernel keeps 1Cat's operand decoders and replaces the scheduling: compact per-expert tiles, one tile choice frozen per codec | first token -4.1 % at 32K, -4.3 % at 150K, -2.4 % at 300K; the kernel alone -16.6 % | none | near-tie gate PASS; not bitwise (at most 48 units in the last place) | ours; patch [0057](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/blob/main/engine/production/0057-prefill-iq-gemm.patch); merged as [#1387](https://github.com/1CatAI/1Cat-vLLM/pull/1387) |

## Adopted: caches and correctness

| Change (switch) | What it is | Gave | Cost | Changes answers? | Source and code |
|---|---|---|---|---|---|
| Hot-ring refill after a cache hit (`HOT_REFILL`) | On a prefix-cache hit the on-card part of the key-value cache is refilled from the host copy. Without it a hit past 32K could read stale rows | a correctness fix | none | fixes wrong output | the bug report is [1Cat-vLLM issue #1341](https://github.com/1CatAI/1Cat-vLLM/issues/1341) (squibloads); our fix merged as [#1381](https://github.com/1CatAI/1Cat-vLLM/pull/1381) |
| Three chat slots (`max-num-seqs 3`, 2026-11-11) | With one slot the server kept one chat's prompt cache, so long chats taking turns each read their whole prompt again | three chats of 60K, 100K and 150K tokens taking turns: cache hits 0 of 6 to **6 of 6**; first token 20.6 to 49.2 s down to **0.57 to 0.88 s**. A lone request is not slower | card memory per slot. Requests still run one at a time: not a throughput claim | near-tie gate PASS, every row identical. On chat-template turns a cached reply can differ from a cold read at a near-tie (seen on the one-slot server too) | ours; the multi-request engine work under it is not in the published patches |
| System notes keep their place (`SYSTEM_IN_PLACE=1`, 2026-11-11) | A system message added in the middle of a chat was moved to the front, which shifted every later token and voided the cache | cached tokens 0 to 31,248 of 31,300; next first token **11.2 s to 0.51 s** | none | replies without such a note identical | ours; patch [0058](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/blob/main/engine/production/0058-system-notes-in-place.patch) |

## Hardware changes

| Change | Gave | Source |
|---|---|---|
| Each PCIe switch on its own x16 root port (one switch card moved) | about 2 % on long-prompt first token, nothing on decode | [README, the progression table](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4#the-progression-in-toks) |
| 200 W instead of 150 W per card | -0.06 ms per round: neutral. The cards run at 150 W | [docs/levers.md](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/blob/main/docs/levers.md#did-not-work) |

## Did not work

Every one of these was built and tested. Each row saves a reader the same work.

| Idea | Result | Why |
|---|---|---|
| Expert top-p pruning (skip the least likely experts) | 78 tok/s against 99; greedy output changed | The loss of answers is built into the method: it drops experts the model would use |
| Asynchronous scheduling with the old runner | +1.678 ms per round slower (95 % interval +1.539 to +1.817) after a crash at request end was fixed | Detectably slower. Model Runner V2 replaced the attempt |
| Asynchronous scheduling on Model Runner V2 | -0.085 to -0.117 ms against synchronous | About 0.4 %; it needs its own serving profile. Off |
| One tensor-parallel group of eight across both boards (probe) | no-go before a build: best modelled round 24.65 ms against a 22.0 ms bar | The eight-way sum across the boards took 30.8 microseconds against 5.28 inside one board |
| Prompt chunks of 16,384 tokens | first token +17.5 % at 32K, +5.6 % at 150K; near-tie gate FAIL | Slower, and the chunk size changes long-prompt rounding |
| Prompt chunks of 4,096 tokens | saved 5.6 % of first-token time at 32K and nothing at 128K; near-tie gate FAIL on its re-test (a confident token moved, gaps of 0.31 and 0.78 nats against a bar of 0.1) | The chunk size changes long-prompt rounding |
| Fused prefill attention, exact-softmax mode | first token **-11.4 % at 32K, -10.4 % at 128K**; near-tie gate FAIL: 1 of 8 identical | 99.4 % of attention elements were bit-equal; the rest moved confident tokens. Not installed. A version that keeps the summation order was then prepared (bit-identical on the test machine's synthetic tensors); its window on real weights was still pending at the checkpoint |
| Tensor-core decode attention, split-K (`ATTN_HMMA=2`) | -0.143 ms, not detectable; near-tie gate FAIL (a confident token moved in 6 of 8 rows) | Rejected on quality |
| FP16-packed shared-expert kernel (`SHARED_DOWN`) | 25.30 to 25.15 ms per round; near-tie gate FAIL (KL up to 0.17) | Rejected on quality |
| Expert "wave" kernel | +0.216 ms per round slower | Detectably slower. Four earlier runs of this test stopped in their own build and check steps before one ran |
| Fused layer transition, three variants | slower or neutral (26.8 against 16.53 microseconds for the pair it replaces; a later variant +0.020 ms) | The mixing step in the middle is 11.4 of the 16.5 microseconds |
| 19 expert-kernel variants | all bit-clean, none beat production; half-warp variants 25 to 30 % slower | The expert kernels are limited by instruction issue and latency. This is what led to the instruction diet |
| Activation prefetch, weight evict-first hints | -0.04 ms: neutral | The expert kernels run at 24 % occupancy and stall on long scoreboard waits, not on memory |
| Warming the L2 cache with the next weight matrix on a side branch | lost | The warm-up stretched to 21 to 28 microseconds beside the main kernel; the best case was 2.4 to 4 |
| Deeper prefetch in the dense kernel | every variant 1.2 to 5 microseconds slower, bitwise over 3,648 runs | The existing double buffer is enough |
| Bitwise FP8 drafter mat-vec (`DRAFT_GEMV2`) | 780.1 microseconds against 627.8 with the sharding alone | Slower than the sharding alone. Off |
| Probabilistic sharded drafts, draft temperatures 1.2 and 0.85 | +0.05 to +0.15 accepted drafts per round but +0.5 to +1.55 ms per round: -3.1 to +0.6 % tok/s | The extra drafter work eats the acceptance gain |
| At most 3 verified drafts per round | round -2.09 ms but 0.877 fewer accepted drafts: **-14.5 % tok/s** | Shorter rounds do not pay for the lost drafts |
| 3 or 4 drafts per round | refused at boot | The drafter checkpoint's block size is 5 |
| Engram rows cached in spare card memory | not built | Sized first: all host Engram work is about 0.12 ms per round, and a round needs all 288 rows, so a partial cache removes nothing |
| A lookup-free repack of the IQ1_M expert format | no-go before a build: 360 to 344 instructions per 256 weights (-4.4 % against a 20 % bar), +35.7 % bytes | Not worth the memory |
| CPU pinning to physical cores | +0.079 ms (95 % interval -0.020 to +0.177) | Rules out a gain of 0.05 ms or more |
| A spin-wait in the broadcast queue | +0.04 to +0.09 ms | - |
| Router with 8 warps instead of 4 | -0.027 ms (95 % interval -0.125 to +0.071) | Neutral on this machine |
| A side stream for the mixing step | numerics passed; below the win bar in isolation; the captured graph showed no real fork and join | Parked until a new design |
| The suggested tiling configuration of the dense kernel | no gain; near-tie gate FAIL (2 of 8 identical) | An autotuned configuration is a change of rounding |

## Two lessons the build's page states

1. Two of the winners (`DMV_X`, `PLAN_FUSE`) were predicted near zero by isolated kernel benchmarks, and one loser
   looked good alone. Only the whole server, measured in mirrored arms, decides.
2. The biggest prompt-reading gain of the last day was a one-line transport setting, found by timing where the
   prompt waited between the two boards, not a kernel.

## What was sent upstream

On 2026-11-05 PLI Labs opened 20 pull requests against 1Cat-vLLM (#1370 to #1390) with the port and the pieces
that won, each new switch off by default. On 2026-11-18 (read with `gh pr list`), 12 of the 20 are merged and 8 are
open; a later documentation pull request (#1488) is merged too. The full list with links is in
[../CREDITS.md](../CREDITS.md#what-pli-labs-sent-upstream).
