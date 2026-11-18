# Custom kernel work that could speed up the NVFP4-8R build

> **Origin.** PLI Labs research archive, file `kernel-research/CUSTOM-KERNEL-OPPORTUNITIES-20261109.md`. Published 2026-11-18 in the Volta V100 research
> repository.
> **Edited for publication.** Machine names, private file paths, network addresses, session names and the
> operator's name were replaced with generic terms (`the V100 server`, `[private path]`, `[address]`,
> `the owner`). Links to files that are not published were turned into plain text. No figure and no finding
> was changed. Dates inside the note are its own; later results are in the build pages of this repository.

Date: 2026-11-09. Build: NVIDIA's DeepSeek-V4.1-Flash-1017 NVFP4 pack on the V100 server's eight V100 cards, port 18043.

A kernel is a small program that runs on the graphics card. One pass of the model runs about two thousand of
them. A custom kernel is one written for our cards and our weight format instead of a general one.

## How to read this

Labels on every number:

- MEASURED: taken from a run on our hardware. The file that holds it is named. Paths are under
  `[private path]` unless they start with `research/kernels/`, which is in the repo under `nvfp4-8r/`.
- UPSTREAM: someone else's number, with the pull request (PR) or repository.
- ESTIMATED: my arithmetic, shown.
- Hypothesis: my reasoning, not yet tested.

What I ran: file reads, checksums, CPU arithmetic, and a fetch of the public PR texts. I ran nothing on a card.
Every statement about what a kernel does is a reading of its code. The source material is collected in
`research/kernels/README.md`.

## Where the time is today

Two phases matter. Prefill is the model reading the prompt. Decode is the model writing the answer, in rounds:
each round checks up to six tokens at once and keeps about three.

| Quantity | This build | Label |
|---|---|---|
| Decode, document-style text | 82 to 88 ms per round | MEASURED, `logs/baseline/nvfp4-final-l1-tput.out` |
| Prefill speed, about 105,000 tokens | 1,441 tokens per second | MEASURED, `nvfp4-final-long100k.out` |
| A 216,000-token prompt (thinking on, 300 tokens out): wait before the first token | 134 to 140 s | MEASURED |
| Same request: time spent writing | 7 to 10 s | MEASURED, same files |

So for a long prompt, reading is about 94 percent of the wait whenever the kept prompt is not reused. In that
test it was not reused, because the test dropped the earlier reasoning text. That is a cache matter, not a kernel
matter. When reuse works, reading falls away and writing speed is what you feel. Kernels cannot change reuse.

Prefill works through the prompt in chunks of 8,188 tokens. This build takes 5.15 s per chunk (ESTIMATED: the
wait above divided by 26.4 chunks).

## Ranked list

| Rank | Opportunity | Helps | Expected gain on this build | Effort | Risk | Code to start from |
|---|---|---|---|---|---|---|
| 1 | Run the warm experts from their small records during prefill | Prefill | Up to 2.2 to 3.0 s of the 5.15 s per chunk (ESTIMATED upper bound) | Large | Medium | Yes: production's own prefill kernels |
| 2 | Tensor-core prefill kernel for the native experts | Prefill | 3 to 20 percent of the wait, perhaps double (ESTIMATED) | Medium | Medium | Yes: 1Cat PR #1282 |
| 3 | One fused kernel for the expert bookkeeping and activation | Decode | 3 to 7 ms per round (ESTIMATED) | Small to medium | Low | Partly: production's plan and activation kernels |
| 4 | Small-record expert kernel sized for the few routes it serves | Decode | Up to about 6 ms per round (ESTIMATED upper bound); measure first | Medium | Medium | Yes: production's v41k |
| 5 | Remove the zero padding from each native expert record | Both | 3.6 percent fewer expert bytes; 0.3 to 0.4 ms per round; 170 to 200 more experts on each card | Medium | Low | Yes: your PR #1030 |
| 6 | Fetch-and-keep kernel for experts in host memory | Decode | Unknown; only if you reject mixed precision | Large | High | No |
| 7 | Fused kernel for the Engram table lookup in MXFP4 | Decode | None today; removes a 0.24 ms per round penalty if you switch tables (ESTIMATED) | Small | Low | No |
| 8 to 12 | Top-k launch sizing; long-prefill attention ports; FP8 dense kernels; drafter kernels; two-tile expert kernel for decode | | Not worth doing now | | | |

## 1. Run the warm experts from their small records during prefill

What it is. An expert is one of 384 small sub-networks in each layer. Each token is sent to six of them; each
sending is a route. The native record of an expert is 4.87 MB per card. Not all of them fit in card memory, so
the rest sit in host memory, and every byte fetched from there crosses PCIe, the slow link between the main board
and a card. In prefill a chunk touches almost every expert, so every off-card expert is copied to the card for
every chunk of every layer. That copying is called staging.

Where the time is. One copy costs 1,014.8 microseconds (MEASURED on one card,
`logs/testbox/p3-bench-20261109T185349Z.json`). Each card has 111 off-card experts per layer on the first half
of the model and 149 on the second (MEASURED counts, `plans/00-WHERE-TIME-GOES.md`). If a chunk touches all of
them, that is 111 x 20 layers x 1.015 ms = 2.25 s, and 3.02 s for the second half (ESTIMATED upper bound), against
5.15 s per chunk. I read that as: staging is most of the prefill time. That is a hypothesis until a per-chunk timing
splits it.

The opportunity. The three-tier plan now being built keeps a small, lower-precision copy (the IQ record, about
2 MB) of the warm experts on the card. Production already has kernels that run prefill from those records
(`SharedOriginalPrefill` and the transcode kernels in `gguf_prefill_transcode_sm70.cu`). Using them for the warm
tier in prefill means those experts never cross PCIe. This is plan 04 Part B, today only an outline.

A warning that follows from the same arithmetic. Three-tier decode alone makes prefill slower, not faster: more
experts leave the native set, so 207 to 245 per layer would be staged instead of 111 to 149 (ESTIMATED in plan
04). Since reading dominates your requests, three-tier should not ship without this item or an equivalent.

Effort and risk. Large. The risk is whether production's prefill code can run on a chosen subset of experts
without its own cache object (plan 04, step B1). Warm experts would be at IQ precision in prefill as in decode.

Proof. Acceptance: one layer at 512 and 8,188 rows matches a reference built from the real weights. Paired
measurement: the 105,000-token and 216,000-token prompts, old, new, new, old.

## 2. Tensor-core prefill kernel for the native experts

The concept. Tensor cores are the part of the card that multiplies matrices fast, about 116 trillion operations
per second on a V100 (ESTIMATED in AGENTS.md section 0.4). They only accept 16-bit numbers, so 4-bit weights
must be decoded first. The kernel this build uses, `skinny_moe_qpn_sm70`, was shaped for decode, where an expert
sees at most six rows (a row is one token's data). In prefill it decodes each block of weights again for every
16 rows. A chunk of 8,188 tokens gives each expert about 128 rows (ESTIMATED: 8,188 x 6 / 384), so the same
weights are read and decoded eight times.

Where the time is. One layer with 512 rows, all experts on the card, takes 5.2 to 5.5 ms (MEASURED on one card,
a progress log of a private working repository). Scaled to a full chunk that is 43 to 86 ms per layer, or 0.86 to
1.7 s per half of the model per chunk (ESTIMATED: 8 to 16 times the measurement, times 20 layers).

Upstream already has the kernel. 1Cat PR #1282 adds `skinny_moe_e8m0_m8n8k4_triton`, written in Triton, a
language for writing kernels in Python. It decodes a block once and uses it for up to 64 rows. It takes exactly
our record format. The file is already in this build's engine and is not used. UPSTREAM numbers: 16 to 23
percent faster at 24 rows per expert; with the modified Triton it needs, a plain matrix multiply reaches 82.5
instead of 8.6 trillion operations per second on a V100; whole-prompt prefill 7.8 percent faster on another
model. At 128 rows per expert I expect more than the 24-row figure (hypothesis).

Expected gain. If the expert math gets 1.2 to 3 times faster, that saves 0.14 to 1.15 s per chunk in each half
of the model (ESTIMATED). That is 3 to 20 percent of today's wait if the two halves work on different chunks at
the same time, and up to twice that if they do not. I did not check which. The share grows once staging is fixed.

Effort and risk. Medium. It needs the modified Triton build from PR #1282 in this build's own environment, which
changes how every Triton kernel in the engine is compiled. Upstream reports that answers were worded differently
with it on. So it needs the quality check, not only a speed check. Production is not touched.

Proof. Acceptance: the existing prefill check `accept_p3_layer.py` at 64 and 512 rows, extended to 8,188, against
the high-precision references. Then a paired per-layer timing on a test card, then the two long prompts in ABBA order.

## 3. One fused kernel for the expert bookkeeping and activation

The concept. Launching a kernel has a fixed cost. In decode the cards replay a recorded list of launches (a CUDA
graph), and every small operation in that list is its own launch. Before the expert kernel runs, the code sorts
the 36 routes of a round into groups, using about ten small operations, and about twelve more when a layer has
experts both on and off the card. Between the two halves of the expert math, the activation (SwiGLU) is another
handful. Each works on a few dozen numbers yet costs a full launch.

Where the time is. A layer with a split costs 101.4 microseconds more than one without, at six rows, even when
nothing is off-card: 4.06 ms per round over 40 layers (MEASURED on one card, same bench file). The all-on-card
layer takes 347 microseconds, and the weight bytes alone account for about 230 to 285 of that at the measured
read rate (ESTIMATED: 30 to 36 experts x 4.87 MB at 620 to 640 GB/s). The remaining 60 to 115 microseconds per
layer is bookkeeping, activation and the final sum: 2.4 to 4.6 ms per round.

Expected gain. 3 to 7 ms per round if one kernel replaces the chains (ESTIMATED). That is 4 to 8 percent of
today's round and more of a three-tier round. It applies with or without three-tier. This is plan 03 Part B,
widened to include the activation.

Upstream code. Production fused the same step for its own experts (`q8_plan` in `v41k.cu`) and has a fused
SwiGLU kernel (`deepseek_v41_swiglu`). The earlier campaign's lesson applies: a fusion saves the time of the
kernels it removes, and a bench of the new kernel alone misleads (`research/kernels/notes/KERNEL-RESEARCH-CLAUDE-NOTES.md`,
rule 3).

Effort and risk. Small to medium, low risk. The grouping is integer work, so the result can be required to
match bit for bit.

Proof. Acceptance: the fused plan equals the old chain on every routing in the lane's dispatch tests. The
token-exact greedy check must pass. Paired: ABBA on the three public prompts, decode ms per round.

## 4. Small-record expert kernel sized for the few routes it serves

In the three-tier build the warm experts run on production's IQ kernel. It was built to serve all 36 routes of a
round. In the new build it serves only the warm ones, a few per layer (ESTIMATED from the lane's route shares:
about 4 to 9 percent of 36, so one to three routes). It still launches its full sequence every layer. On production that sequence costs
185 microseconds per layer, 7.4 ms per round (MEASURED, `research/kernels/notes/PROF1103b-REPORT.md`).

Nobody has measured what it costs with few live routes; plan 01 lists that as its first measurement. If the cost
stays near full, up to about 6 ms per round is at stake (ESTIMATED upper bound: 7.4 ms minus a tenth). If it
already shrinks with the route count, there is nothing to build. So: measure, then decide. A version would loop
only over live routes, as the index scorer `v41idx.cu` does for live keys.

Effort and risk. Medium. The IQ kernels share work across the four cards, which makes changes delicate.

Proof. The plan 01 arm difference, then a per-layer bench with real routing on a test card.

## 5. Remove the zero padding from each native expert record

Each native record carries padding. The last matrix of an expert has 576 columns per card, but the kernel
accepted only widths it could split evenly, so the records were padded to 640. That is 174,080 bytes of zeros in
every 4,874,240-byte record, 3.57 percent (arithmetic from the constants in `nvfp4_8r_native.py`).

PR #1030 (ours) added the missing splits (9 and 18) to this kernel for exactly this width. It is open upstream
and not in this engine.

Expected gain. Every expert read, every fetch and every staging copy shrinks 3.57 percent. That is 0.3 to 0.4 ms
per round on the expert math (ESTIMATED: 3.57 percent of the 230 to 285 microseconds of weight reading per
layer in item 3, times 40 layers). It also
frees 0.8 to 0.95 GB per card, room for 170 to 200 more native experts (ESTIMATED: hot experts per card x
174,080 bytes), which cuts fetches further.

Effort and risk. Medium, low risk: port the kernel cases, rebuild all packs, rerun the pack and layer checks.
One conflict: the Triton kernel of item 2 wants this width in multiples of 128 for its largest tile, so the two
must be designed together.

Proof. The existing value-exact pack check and layer check, then ABBA.

## 6. Fetch-and-keep kernel for experts in host memory

Today a fetch of an off-card expert costs 0.9 to 1.35 ms on one card (MEASURED) and about 2.3 ms on the full
server (ESTIMATED from two boots). The alternatives measured so far: a driver copy at 743.7 microseconds, which
cannot run inside the recorded launch list, and a kernel-side gather at 1,014.8. None can beat the link, about
12 GB/s at best (UPSTREAM, PR #1277), which is 0.4 ms per expert.

A kernel could read an expert once and also leave a copy on the card, so the next use is free (hypothesis). Its
value depends on how often the same off-card expert returns within a few rounds, which has not been measured.
With three-tier, decode on document text sends almost nothing to host memory, so this item only matters if you
decide every route must be native. Not recommended now.

## 7. Fused kernel for the Engram table lookup in MXFP4

The Engram is a large lookup table the model consults at two layers. The build serves it today in production's
format (IQ4_NL), read from the serve profile `run/serve-nvfp4-pp2.yaml`. NVIDIA's pack ships the same table in
the 4-bit MXFP4 format, and the lane built and checked that table as an option. Decoding it runs as plain small
operations: 0.91 to 1.01 million rows per second against 4.2 to 4.6 million for production's format, about 120
microseconds more per lookup (MEASURED on one card, `the progress log`, step P4). That would be about 0.24 ms per
round (ESTIMATED: two lookups). One small kernel would close it. It gains nothing on today's profile. Build it
only if you decide to serve NVIDIA's table.

## Not worth doing now

- Top-k launch sizing. The check found that four small kernels in the attention index are sized by the maximum
  context, not the live one. Its own estimate is tens of microseconds per round. Decode is flat from 46 to
  303,004 tokens of context (MEASURED, `nvfp4-final-ctx.jsonl`), so nothing grows with context here.
- Long-prefill attention and indexer ports (1Cat #1280 and the indexer part of #1282). Those PRs target the
  mainline DeepSeek-V4 path. This engine has its own fused prefill attention and index kernels from the earlier
  campaign. Prefill speed does not fall with length (MEASURED: 1,441 and 1,525 tokens per second at 104,000 and
  498,000 tokens).
- FP8 dense kernels (#1021, #1157). They serve 8-bit dense weights. This build's dense weights are production's
  5-bit and 8-bit GGUF formats on the `dmv13_x` kernel. Moving to FP8 would add bytes and slow decode. It would
  be a precision choice, not a speed one.
- Drafter kernels. The drafter is the small model that guesses the next five tokens. It takes about 1.35 ms of
  a round on production (MEASURED, PROF1103b) and has no host wait (read-only check, `logs/checks/pin-and-drafter-wait-20261109.md`).
  That is under 2 percent of this build's round.
- Two-tile expert kernel for decode (the open gap v100-skinny names). It pays when an expert sees more than
  eight rows. With six rows per round that never happens in decode. Prefill already uses the two-tile form.

The campaign of 2026-11-05 polished single kernels on production and found gains of 0.1 to 0.9 percent each
(`research/kernels/kcamp1106/`, S5, S6, S8). The same will be true here until the fetches are gone.

## Recommendation

Build item 3 first: it is small, exact, helps every round on either residency plan, and has production code to
copy from. Alongside it, take one timing of a single 8,188-token chunk that separates staging, expert math and
the rest; that one measurement decides between items 1 and 2. If staging dominates, as the arithmetic says, tie
item 1 to the three-tier build so that reading speed does not get worse when writing speed gets better. If
expert math is a fifth or more of the chunk, try item 2 on a test card, since the kernel already exists upstream.
