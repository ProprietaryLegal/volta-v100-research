# TOK1103: tok/s levers from v100-skinny and 1CatAI/1Cat-vLLM

> **Origin.** PLI Labs research archive, file `kernel-research/TOK1103-REPORT.md`. Published 2026-11-18 in the Volta V100 research
> repository.
> **Edited for publication.** Machine names, private file paths, network addresses, session names and the
> operator's name were replaced with generic terms (`the V100 server`, `[private path]`, `[address]`,
> `the owner`). Links to files that are not published were turned into plain text. No figure and no finding
> was changed. Dates inside the note are its own; later results are in the build pages of this repository.

Research note of 2026-11-03: a sweep of the forks, comments and pull requests of v100-skinny and 1Cat-vLLM for
anything that raises tokens per second on the eight-card build.
Raw data: TOK1103/raw/. Labels: MEASURED (our hardware), UPSTREAM (PR author's numbers), ESTIMATED (projection for
the V100 server PP2xTP4 V4.1 + DSpark).

Baseline (MEASURED, PROF1103): 32.3 ms per round = stage 1 13.6 + stage 2 13.1 + gap 5.6 (~3.4 drafter + ~2.1 host);
prose ~2.15 tokens per round, ~70.5 tok/s. (A/B baseline 01:18-01:35Z: 99.1 / 99.6 tok/s over the 3-prompt mix,
31.1 ms per round.)

Why most V4-Flash kernel PRs don't transfer: V4.1 here is a custom path (deepseek_v41/* + v41k): hidden 5120, 40
layers, 384 experts top-6, IQ1_M/IQ2_XXS/IQ2_XS GGUF experts, Q5_K/Q8_0 dense via dmv13. V4-Flash is hidden 4096, 43
layers, 256 experts, native MXFP4, FP8 GEMMs. FP13 GEMV, MXFP4 direct-order, sparse-MLA split-K, QNORM_KV and the
#308/#344 MXFP4 hot paths target code we don't run.

## Ranked
1. **Live-prefix decode index reduce.** BUILT: c6d459c on v41pkg/tok1103 (off 947041b), knob VLLM_V41_INDEX_REDUCE
   (default 0, a compile factor).
   - Problem (MEASURED): 8 NCCL all-reduces per round over the whole store width, 130/223 us each, ~1.5 ms per round.
     Custom AR refuses them: the store has max_len//ratio+1 rows, so T x rows x 2 B is never a multiple of 16
     (collective_provider.reduce size%16).
   - Fix: V41IndexBank (CustomAllreduce.create_shared_buffer over IPC; 1 KiB epoch flags + FP16 partial
     [8, round_up(rows, 8)], ~8.4 MB per rank at 524,289 rows); kernel v41k index_reduce_live (64 CTAs x 256 threads,
     per-CTA epochs st.release.sys / ld.acquire.sys, sums columns [0, (pos[T-1]+1)/ratio) of the 4 partials in rank
     order in FP32, one rounding). Graph-safe. Built only on TP4 with a fully connected ca_comm and a v41k that has the
     op; otherwise a warning and today's path.
   - Exactness: identical on every rank, custom-AR numerics; NOT bitwise equal to NCCL's FP16 ring sum -> near-tie gate.
   - Gain (ESTIMATED): -1.2 to -1.4 ms/round at short context, +4-5 %; at 131K ~60-70 us vs 223 us per call.
   - CPU checks (MEASURED): 10 new tests pass; V4.1 suite shows the base's 11 CPU-environment failures, none new; nvcc
     12.4 sm_70 compile OK (40 regs); ruff/clang-format clean; data boundary PASS.
   - GPU plan: rebuild v41k; log line "V4.1 index reduce bank (TOK1103)" on all 8 workers; A knob 0 vs B knob 1, T=0,
     short + 32K + 131K; greedy identical or near-tie; acceptance +-1 %; index_reduce_live replaces the NCCL kernels.
2. **Host gap via 1Cat Model Runner V2.** Not built, effort L. #1221 (open): DSpark on MRV2 + context-bucket graphs,
   UPSTREAM V4-Flash PP5 prose 42.9 -> 48.2 tok/s (+12 %), step 70 -> 63 ms. #1263 (merged): no full-history resend
   under PP (UPSTREAM GLM-5.3 PP5 516K 114.5 -> 96.7 ms/step), V2-runner only. #1267 (merged): device-side top-k/top-p
   row classification. Our PP2 code is in the V1 runner. ESTIMATED +6-10 % once ported.
   Lane's async note: under PP the AsyncScheduler schedules a request only every pp_size-th step (coordinator: that gate is
   `use_v2_model_runner` only in async_scheduler.py; the V1-runner async arm is being measured).
3. **Launch bubbles / fusion.** Not built. 2,089 kernels per round (MEASURED). Upstream #884/#957/#1292 (TP4 AR+norm),
   #1172 mHC (UPSTREAM -5-14 % per op), #1147 router+Q8; KFUS1 P1-P6 measured these on the old engine. ESTIMATED
   +5-10 % total (coordinator: size from removed kernel time; graphed gaps are ~0.3-1 us).
4. **Confidence-scheduled DSpark verification.** Config only: dspark_confidence_threshold /
   dspark_max_verification_tokens exist; the drafter has mtp.2.confidence_head. Fewer verify rows -> fewer distinct
   experts. ESTIMATED +0-4 %. Requires synchronous scheduling.
5. **FP8 small-shape tuning.** Env only. #1187: VLLM_SM70_FP8_TUNE_SMALL_SHAPES defaults to 1 in pp2-up; UPSTREAM
   V4-Flash PP5 DSpark 30.4-38.9 tok/s on vs 50.3-51.7 off. #1273 freeze_tuning_after_warmup ~-2 of 59 ms. V4.1 dense
   doesn't use this tuner; the FP8 drafter may. ESTIMATED 0 to +5 %.
6. **Drafter tail.** ~3.4 ms of the gap (MEASURED). VLLM_V41_MARKOV_SHARD=1 exists (KFUS1 d5a, MEASURED -0.55 ms on
   the old engine). Drafter ARs NCCL -> custom AR ESTIMATED -0.2 ms. #1218 reduced draft vocabulary: MRV2 only,
   UPSTREAM +9 %.
7. **Not applicable or already present:** #1127 and Peuqui 9bc45c6630 fused scorer (we have v41k
   index_scores_decode); #1142 (M20), #963 (>32 verify rows); #1290/#1298/#1287; #1245 small; Peuqui 772cbe8032
   (memory only), 569d11c562 (Flash only).
8. **Rejected on upstream evidence:** TP8 (V4-Flash TP8 66 vs PP2xTP4 75 tok/s; the 116 tok/s TP8 N7 run is
   acceptance-driven); native MXFP4 V4.1 (~269-289 GB > 256 GB); EXL3 #1243 (30.5 tok/s plain).
9. **v100-skinny:** main only, 7 commits (Qwen NVFP4); PR #8 grouped NVFP4/MXFP4 MoE N/A for GGUF; fp8_e5m2 KV costs
   +4.82 ms/round on Qwen (unused here); Peuqui lessons already present (FP8 MTP head, capture sizes k+1, IndexShare
   rejected). Forks are Qwen-oriented; no V4.1 decode win.

Suggested GPU order: item 1 (~20 min); items 5 and 6 (~25 min); item 4 (~25 min); items 2 and 3 are code ports.
