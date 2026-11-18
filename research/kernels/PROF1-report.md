# PROF1: where the V100 server's decode step goes (MEASURED 2026-11-01, FINAL-resident-mp1 + torch profiler)

> **Origin.** PLI Labs research archive, file `kernel-research/PROF1-report.md`. Published 2026-11-18 in the Volta V100 research
> repository.
> **Edited for publication.** Machine names, private file paths, network addresses, session names and the
> operator's name were replaced with generic terms (`the V100 server`, `[private path]`, `[address]`,
> `the owner`). Links to files that are not published were turned into plain text. No figure and no finding
> was changed. Dates inside the note are its own; later results are in the build pages of this repository.

One profiled boot
(the V100 server top10/PROF1; traces top10/PROF1/traces/<cell>/, 8 ranks each; parser and tables in a private campaign folder). Clocks 1425 MHz cap, 180 W. Greedy, short synthetic essay prompt (39 tokens), 512 output tokens.

## Cell c1-short (one stream)
| | profiled | plain (same boot) |
|---|---:|---:|
| wall for 512 tokens | 18.26 s | 17.09 s |
| decode steps (spec drafts delta) | 244 | 244 |
| ms per step | 74.8 | **70.0** |
| accepted per step | 1.10 (low-acceptance prose prompt) | 1.10 |
Profiler overhead at one stream: ~7 %.

Per step, profiled (ms; rank 0 = stage 0 TP0, rank 4 = stage 1 TP0; other TP ranks similar):
| | stage 0 | stage 1 |
|---|---:|---:|
| kernel window | 74.5 | 74.3 |
| GPU busy (kernels) | 29.1 | 56.7, of which 24.3 is the NCCL recv kernel spinning while stage 0 works -> **~32.4 real** |
| GPU idle | 45.4 | 17.6 |
| routed MoE (exl3_gemv_mixed_grouped + gating) | 5.8 | 5.1 |
| dense GEMM (cutlass wmma, s884 fp16, cuBLAS, fp32 sgemm 0.7, gemv) | 6.3 | 8.4 |
| norm / elementwise / copy / reduce (torch small ops) | 6.0 | 7.3 |
| attention + indexer (_fused_attn_partial/merge, rope/kv insert) | 3.8 | 4.3 |
| TP4 all-reduce over NVLink (~40 per stage per step, ~57-75 us each) | 3.7 | 3.5 |
| other (mHC sinkhorn/hc_pre/hc_post, gate_up_swiglu, down_combine, radix sort) | 3.7 | 4.1 |
Kernel launches per step (top-98 rows of the key-averages table, so a lower bound): ~2,530 on stage 0, ~3,510 on stage 1.

### What it says (DERIVED from the table)
- A single stream runs the two stages one after the other: stage 0 ~29 ms of kernels, then stage 1 ~32 ms, then the
  sampled tokens go back. Each board sits idle while the other works. The remaining ~9-13 ms per step are launch and host
  gaps and the PP hand-off.
- Memory bandwidth is not the limit: the bandwidth-heavy routed MoE is ~11 ms of ~70 (16 %).
- The "many small kernels" bucket is large: torch small ops ~13 ms + gaps ~9-13 ms = roughly 22-26 ms (about a third).
- Dense GEMMs at verify width (M = 6) ~15 ms; all-reduce ~7 ms; attention ~8 ms; mHC/other ~8 ms.

## Update 09:30Z: 3-stream and 32K tables (rank 0 = stage 0, rank 4 = stage 1; ms per step, profiled)

| cell / rank | window | busy | idle | pp_hop/nccl | allreduce | moe | attn/indexer | dense GEMM | norm/elementwise | memcpy | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c3-short r0 | 139.1 | 69.0 | 70.2 | 0.1 | 29.2 | 16.0 | 2.6 | 9.3 | 7.4 | 0.4 | 3.9 |
| c3-short r4 | 138.8 | 98.9 | 39.9 | 47.5 | 6.9 | 14.7 | 4.4 | 12.3 | 9.0 | 0.4 | 4.2 |
| c1-32k r0 | 191.3 | 138.9 | 52.3 | 67.1 | 5.8 | 14.8 | 12.1 | 19.0 | 13.0 | 1.1 | 6.3 |
| c1-32k r4 | 186.4 | 75.8 | 110.7 | 33.0 | 3.1 | 6.1 | 7.7 | 11.8 | 9.2 | 0.5 | 4.8 |

Reading (owner): at 3 streams, stage 0's all-reduce bucket grows from 3.7 to 29.2 ms/step while its MoE grows 5.8 -> 16.0.
The all-reduce time is most likely spin-wait inside the custom push all-reduce (ranks arriving at different times), not
link time; KDEC should check per-rank arrival skew. The c1-32k windows include the 26K prefill, so they are not clean
decode numbers (pp_hop 67 ms on stage 0 is prefill hand-off). Parser and tables in a private campaign folder.
