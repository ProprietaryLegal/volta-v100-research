# PROF1103: where the 8-GPU decode round goes (the V100 server 1Cat PP2xTP4 + DSpark 5)

> **Origin.** PLI Labs research archive, file `kernel-research/PROF1103-REPORT.md`. Published 2026-11-18 in the Volta V100 research
> repository.
> **Edited for publication.** Machine names, private file paths, network addresses, session names and the
> operator's name were replaced with generic terms (`the V100 server`, `[private path]`, `[address]`,
> `the owner`). Links to files that are not published were turned into plain text. No figure and no finding
> was changed. Dates inside the note are its own; later results are in the build pages of this repository.

Measured 2026-11-03 01:10-01:25Z. Server: v41pkg/pp2-up 947041b, serve-v41pkg-pp2-prof.yaml (= serve-v41pkg-pp2.yaml plus a
torch profiler, stack off), 1522 MHz / 150-200 W as deployed. Request: 16-token public prompt, 96 tokens out, T=0,
thinking off. Traces: the V100 server bench5/prof-20261103 (8 workers); stage-1 rank 0 and stage-2 rank 0 copied here.
Scripts: prof_rounds.py, prof_timeline.py, prof_kernels.py, prof_ctx.py.

## Round timeline (MEASURED, 35 decode rounds, 96 tokens = 2.7 tokens per round)
Every round has the same shape:

| Part | ms | Share |
|---|---|---|
| Stage 1 (Board B, layers 0-19 + Engram), annotation span | 13.6 | 42 % |
| Stage 2 (Board C, layers 20-39 + head + DSpark drafter), after stage 1's hand-off arrives | 13.1 | 41 % |
| Gap after stage 2's forward annotation -> next stage-1 start: 3.4 ms of stage-2 GPU work (DSpark drafter: fp8_gemv 0.81, _markov_block 0.43, turbomind gemm 0.37, ...), then ~2.1 ms of host (D2H sync, engine IPC, scheduler, broadcast, stage-1 input prep) | 5.6 | 17 % |
| **Round period** | **32.3** | |

While one board works the other waits (stage 2's recv kernel spins 10.1 ms per round).

## GPU kernel time per round (MEASURED; sums can exceed wall time where streams overlap)

| Category | Stage 1 ms | Stage 2 ms | Note |
|---|---|---|---|
| Experts (v41k gate_up_em5, down_rows, peer_gather; IQ2_XXS) | 4.21 | 4.07 | ~57 % of read roofline (ESTIMATED from ~30 distinct experts x 35.4M x 2.06 bpw / 4 ranks) |
| Dense GEMV (dmvns dense_mv, GGUF dense) | 2.23 | 2.46 | |
| Other small kernels | 1.56 | 1.78 | 302 / 379 kernels |
| cuBLAS / CUTLASS GEMM | 1.19 | 1.23 | |
| mHC transitions / DSpark Markov | 0.70 | 1.20 | |
| Indexer score all-reduce (NCCL ring LL) | 0.39 | 1.12 | 3 + 5 per round, 130 / 223 us each |
| Hidden all-reduces (custom AR 1-stage push) | 0.30 | 0.51 | |
| DSpark FP8 GEMV | - | 0.81 | |
| NCCL all-gather | 0.05 | 0.52 | |
| Elementwise / copies | 0.46 | 0.90 | |
| **Kernels per round** | **890** | **1,199** | |
| Kernel sum | 11.6 | 15.9 | stage-1 span 13.6 => ~2 ms of launch bubbles |

## Findings, ranked by ESTIMATED tok/s gain
1. **Host gap: CORRECTED 01:25Z to ~2.1 ms (6.5 %)**; the other 3.4 ms of the 5.6 ms gap is the DSpark drafter on stage 2 (CPU trace: stage 2's D2H copy completes 3.4 ms after the forward annotation; 342 drafter kernels sit in that window). Original text: PP2 runs synchronous scheduling (`no-async-scheduling`, VLLM_V41_ASYNC unset; the
   1Cat #1277 TP4 path sets VLLM_V41_ASYNC=1). Async PP removes most of the ~2.1 ms: ESTIMATED +5-7 %. Owner: main. Also cheap: a num_speculative_tokens sweep (3/4/5): later draft positions accept rarely and each costs ~0.65 ms of drafter plus verify rows.
2. **Indexer score all-reduce over the full 524,288-key store every step.** After `index_scores_decode`, the decode
   path all-reduces `scores[tokens, keys.shape[0]]`, the whole packed store width (6 MiB ratio 1, 3 MiB ratio 2), on
   NCCL, whatever the live context. 1.5 ms per round at short context (4.7 %). Fix: reduce only the live prefix
   (graph-safe bounded reduce), or gather q/weights and score all heads locally. ESTIMATED +4-5 % at short context.
3. **Launch bubbles ~2 ms per stage**, 2,089 kernels per round: fusion work (KFUS1 P1/N1 direction). +5-10 %.
4. **Experts 8.3 ms (26 %)**, already near the memory limit for IQ2_XXS: only fewer bytes help (our DGX Spark work's
   lever 2, expert top-p prune, lossy, +3-6 %).
5. **Dense GEMV 4.7 ms (15 %)**: check achieved bandwidth vs the QPN8 FP8 GEMV (v100-skinny 718.6 GB/s at M=1).

## Why PR #344 doesn't transfer directly
- #344's speed comes from kernels for the native MXFP4 experts and FP8 dense layers, on DS-V4 Flash (256 experts,
  hidden 4096, 43 layers): 13.3 ms per token without drafting. Its fast paths are hard-coded to V4-Flash shapes.
- V4.1 Flash 1017 in native MXFP4 needs ~269-289 GiB for the experts alone (ESTIMATED), more than the 256 GiB of HBM
  on 8 GPUs, so the native route needs host spill (#1031) or 12 GPUs.
- Our IQ2_XXS expert kernels already run near the memory limit, and #344's #308-style MXFP4 hot paths don't touch
  GGUF experts (fork-sparks-upstream TOK1103 facts, 01:0xZ).
- What does transfer: the dense-layer FP8-QPN GEMV idea (finding 5) and the host-side discipline (finding 1).
