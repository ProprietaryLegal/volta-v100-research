# PROF1103b: decode round on the S8-D + PLAN_FUSE stack (the V100 server 1Cat PP2xTP4 + DSpark 5)

> **Origin.** PLI Labs research archive, file `kernel-research/PROF1103b-REPORT.md`. Published 2026-11-18 in the Volta V100 research
> repository.
> **Edited for publication.** Machine names, private file paths, network addresses, session names and the
> operator's name were replaced with generic terms (`the V100 server`, `[private path]`, `[address]`,
> `the owner`). Links to files that are not published were turned into plain text. No figure and no finding
> was changed. Dates inside the note are its own; later results are in the build pages of this repository.

Measured 2026-11-03 06:38-06:50Z. Server: combo f54308f + dmvx v41k.so, env S8-D + VLLM_V41_PLAN_FUSE=1,
serve-v41pkg-pp2-prof2.yaml (torch profiler, stack off), 150 W cap. Request: 13-token public prompt, 96 tokens, T=0,
thinking off: 51 decode rounds (1.9 tokens/round on this prompt). Traces: the V100 server bench5/prof-20261103b (8 ranks); rank 0
and rank 4 here. Tables: kernels51.txt (categories), top-kernels.txt (top 40 per stage), experts.txt (per call),
one-layer-s1.txt (one stage-1 layer). Scripts: prof_kernels2.py (rounds as argv[3]; prof_kernels.py
hard-codes 35).

## Round (MEASURED)
- Period about 29.5 ms (stage-1 step starts 29.2-30.0 ms apart), the same as the unprofiled 28.9-29.5.
- Stage 1 kernel work 10.89 ms/round (850 kernels), annotation span 12.6-13.4 ms.
- Stage 2 kernel work 14.05 ms/round (1,167 kernels) excluding the 9.4 ms/round PP receive wait (SendRecv).

## Where the kernel time goes (ms/round, MEASURED)
| Family | Stage 1 | Stage 2 | Note |
|---|---|---|---|
| Experts (gate_up_em5, down_rows, peer_gather, q8_plan) | 3.75 | 3.65 | 40 % of a layer; gtypes mixed, mostly IQ1_M (29) |
| Dense GEMV dmv_x (qkv, q_b, wo_a, wo_b, shared, head) | 2.05 | 2.27 | 102 calls/stage, 16-22 us each, 390-490 GB/s (dmvx-bench) vs ~735 floor |
| mHC transitions (_transition_parts3/_finish3) | 0.71 | 0.75 | 74-80 calls, ~9.5 us |
| cuBLAS/CUTLASS GEMM (attention, shared expert) | 1.19 | 1.23 | |
| Custom all-reduce (1-stage push) | 0.41 | 0.54 | |
| Router (_route, _row_projection) | 0.63 | 0.66 | |
| DSpark drafter (fp8_gemv, turbomind, _markov_block) | - | ~1.35 | |
| NCCL all-gather | 0.05 | 0.45 | 3/round on stage 2: PP-receive TP all-gather 350-400 us (absorbs stage-1 TP skew), logits gather ~50 us, one ~15 us |

Expert gtypes per round: stage 1 gate/up 13x gtype 29 (101.7 us), 5x 16 (82.5), 2x 17 (106.3); down 16x 29 (78.0),
4x 16 (68.4). Stage 2 gate/up 12x 16 (80.4), 5x 17 (96.3), 3x 29 (90.6); down 12x 29 (73.9), 4x 17 (70.9), 4x 16 (63.9);
peer_gather 14.1 us (stage 1) vs 25.3 us (stage 2).

One stage-1 layer: 33 kernels, 483 us of kernel time, 31 us of gaps (about 1 us per kernel), span 488 us. Launch gaps
are therefore about 0.6 ms per stage; the time is in the kernels.

## What this says (ESTIMATED)
1. Experts are the target (expertk1103 lane): 7.4 ms/round; reaching ~450 GB/s on down and ~650 GB/s on gate/up is the
   review's ~2.6 ms. IQ1_M gate/up (stage 1) and down are the largest single items.
2. Dense GEMV is second: 4.3 ms/round at 390-490 GB/s; at the 735 GB/s floor it would be ~1.3 ms less; a realistic
   ~0.5 ms. The calls are dependent (qkv -> q_b, wo_a -> wo_b), so the gain must come from per-call efficiency
   (wave tails on 80 SMs at 4-10 MB), not overlap.
3. A local greedy-verify argmax (avoid the logits all-gather at T=0) is worth only ~0.05 ms: dropped.
4. Small-kernel fusion (rotate/qdq/short_rms, ~155 kernels/round at 2-4 us) is worth ~0.3-0.5 ms/stage including gaps.
