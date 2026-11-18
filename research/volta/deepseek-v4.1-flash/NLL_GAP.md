# DS-V4.1 port: the AM-13 |ΔNLL| gap and the BF16 stream (lane REF-BF16, 2026-10-07)

> **Origin.** PLI Labs research archive, file `v100-research/deepseek-v4.1-flash/NLL_GAP.md`. Published 2026-11-18 in the Volta V100 research
> repository.
> **Edited for publication.** Machine names, private file paths, network addresses, session names and the
> operator's name were replaced with generic terms (`the V100 server`, `[private path]`, `[address]`,
> `the owner`). Links to files that are not published were turned into plain text. No figure and no finding
> was changed. Dates inside the note are its own; later results are in the build pages of this repository.

Question (D16 item 2): the 8-GPU PP2×TP4 port matches the `ref-exact` golden within AM-14, but misses AM-13 vs the
`v100-semantic` golden (|ΔNLL| 0.022–0.039 vs gate 0.02; top-1 0.973–0.995; KL p3 0.0113). The port stores the HC
stream in BF16 (A4); the v100-semantic reference keeps it FP32. How much of the gap does that explain?

Status: DONE 2026-10-07 ~10:45Z. **Short answer: the BF16 stream explains ≈ 0.005 of the port's pooled |ΔNLL| 0.032;
FP16 sublayer inputs ≈ 0.002; the remaining 0.026 is the floor of the scoring method itself (the FP32 reference scored
the same way). The port matches the precision-matched reference; no residual port error. Proposed baseline-relative
AM-13′ in §Conclusion.**

## Method

* `tools/ds41_ref/ref_model.py` — new `Sequence(stream_dtype="fp32"|"bf16", fp16_sublayer_inputs=bool)` (CLI
  `--stream-dtype bf16`, `--fp16-sublayer-inputs`, via `add_precision_args`), orthogonal to the numerics mode and to
  own-k. Rounding points copied from the port (`ds41/sm70-v41` 15b6127ea, `common/hc.py`, `sm70/model.py`,
  `common/engram.py`):
  * `bf16`: the stream is rounded to BF16 after `hc_expand` (exact), after Engram (`stream.copy_((h + g·v).to(bf16))`),
    after every `hc_post`; every `hc_pre` collapse (attention input, FFN input, final collapse) is rounded to BF16
    (its output dtype is `STREAM_DTYPE`). HC mixes / Sinkhorn / Engram gate read the BF16 values in FP32.
  * `fp16in`: `rmsnorm_to_act` outputs FP16 — `attn.x`, `moe.x`, `final.h` — and the head weight is FP16
    (BF16 → FP16 cast, FP32 accumulate and output).
  * CPU check against the port's own `hc.py` functions (random BF16 streams incl. a 1e4-scaled BOS-like row):
    `hc_pre`, `rmsnorm_to_act`, `hc_expand` bitwise equal; `hc_post` 1.9e-5 of elements one BF16 step apart (FP32
    summation order `bmm` vs elementwise at rounding boundaries — not a semantic difference).
* `tools/ds41_ref/tf_variants.py` — teacher-forced logits: ONE prefill over the golden's prompt + 32-token greedy
  continuation, the same token sequence the port was scored on (`compare.py endpoint` → vLLM `prompt_logprobs`).
  Variants on v100-semantic numerics with own-k index semantics.
* `tools/ds41_ref/nll_gap.py` — AM-13/14 metrics (top-1, mean |ΔNLL|, exact KL and the port-comparable top-20 KL lower
  bound) of every variant vs the goldens and vs the FP32 teacher-forced run; port numbers from L-INTEG run 2.

Port data: the port's teacher-forced logits were **not saved as tensors** — only the endpoint summaries
(`[private path]>_<mode>.json`: top-1, top-5, |ΔNLL|, KL lower bound from the
top-20 prompt logprobs). Per-layer port dumps exist for the p0_smoke run-1 steps (`integ/bringup/dump_pp2/`).

## Result 1 — teacher-forced variants vs the v100-semantic golden (run A, 10:07Z, 507 s on c76e1399)

Gating golden `<p>__v100-semantic+ownk` (the one the port was gated on). Port = L-INTEG run 2 endpoint reports, own-k
(gating) entries. KLlb = KL lower bound from the candidate's top-20 (the metric the port was measured with; the exact
KL of the reference variants is within 0.0001–0.003 of it, `nll_gap_A.json`). dNLL split: prompt positions (golden =
one prefill) / continuation positions (golden = 32 greedy decode steps).

| prompt (pos) | metric | **ref FP32** (v100-sem numerics, TF) | ref +bf16 stream | ref +bf16 +fp16 inputs | **port** (8×V100) |
|-|-|-|-|-|-|
| p0_smoke (37) | top-1 / \|ΔNLL\| / KLlb | 0.946 / 0.022 / 0.0027 | 0.973 / 0.021 / 0.0037 | 0.946 / 0.064 / 0.0133 | 0.973 / 0.039 / 0.0057 |
| p1_legal (116) | top-1 / \|ΔNLL\| / KLlb | 0.974 / 0.014 / 0.0006 | 0.974 / 0.022 / 0.0016 | 0.983 / 0.018 / 0.0013 | 0.983 / 0.026 / 0.0013 |
| p2_code (189) | top-1 / \|ΔNLL\| / KLlb | 0.984 / 0.027 / 0.0101 | 0.989 / 0.028 / 0.0056 | 0.989 / 0.024 / 0.0080 | 0.995 / 0.022 / 0.0063 |
| p3_doc (2077) | top-1 / \|ΔNLL\| / KLlb | 0.990 / 0.027 / 0.0027 | 0.985 / 0.032 / 0.0050 | 0.987 / 0.035 / 0.0036 | 0.984 / 0.033 / 0.0045 |
| **pooled (2419)** | top-1 / \|ΔNLL\| / KLlb | 0.988 / **0.0263** / 0.0032 | 0.984 / **0.0313** / 0.0049 | 0.987 / **0.0335** / 0.0040 | 0.985 / **0.0320** / 0.0045 |

dNLL prompt / continuation split (ref FP32 → +bf16 → +bf16+fp16in): p0 0.0001/0.026 → 0.008/0.023 → 0.039/0.068;
p1 0.016/0.009 → 0.024/0.016 → 0.019/0.017; p2 0.005/**0.137** → 0.008/0.126 → 0.008/0.104; p3 0.027/0.023 →
0.033/0.024 → 0.035/0.018. (p2's continuation contains a generated `<eos><bos>`; the BOS row's stream reaches
3.6e5, RANGE_AUDIT.md — prefill-vs-decode differences there are large in every variant.)

Pure effect of the precision change on the teacher-forced logits (variant vs ref FP32 on the same ids, top-1 / |ΔNLL| /
exact KL): +bf16: p0 0.973/0.030/0.0051, p1 0.983/0.022/0.0013, p2 0.989/0.026/0.0102, p3 0.988/0.030/0.0027;
+bf16+fp16in: p0 1.000/0.051/0.0074, p1 0.991/0.023/0.0014, p2 0.989/0.025/0.0096, p3 0.984/0.034/0.0028.

Correction to the L-INTEG table: its p3 row (0.983 / 0.035 / KLlb **0.0113**) is the informational entry vs the
OFFICIAL (stale index-K) golden; vs the gating own-k golden the port has top-1 0.984, |ΔNLL| 0.033, KLlb **0.0045**
(passes the KL gate). Likewise p3 vs ref-exact: own-k 0.980 / 0.044 / 0.0076 (official 0.977 / 0.053 / 0.0567).

Readings:
1. **The reference itself fails AM-13 |ΔNLL| ≤ 0.02 when scored the way the port is scored.** The FP32
   v100-semantic reference, run as one teacher-forced prefill over prompt + continuation and compared to its own
   golden (prefill + greedy decode), gives 0.022 / 0.014 / 0.027 / 0.027 (pooled 0.026), top-1 0.946 on p0. This is
   the FP32-noise × QAT-amplification floor (L-REF 01:50Z / 03:30Z) plus prefill-vs-decode on the 32 continuation
   positions. L-REF's earlier floor (0.005) came from 8 decode steps on p1 only and under-estimated it.
2. **BF16 stream storage adds ≈ +0.005 pooled (0.026 → 0.031); FP16 sublayer inputs another ≈ +0.002 (→ 0.0335).**
   The precision contract alone (A4 + §4.1) moves the teacher-forced NLL by 0.022–0.051 per prompt vs the FP32 run.
3. **The port (pooled 0.0320, top-1 0.985, KLlb 0.0045) sits inside the ideal precision-matched band**
   (bf16: 0.0313, bf16+fp16in: 0.0335). Per prompt the port is within ±0.012 of the bf16+fp16in variant on p1–p3 and
   better than it on p0 (37 positions, one 1.49-nat outlier in the variant). No residual port error is visible at the
   end-to-end level.

## Result 2 — per-layer localisation: port p0 prefill dump vs the reference variants (run C, 10:08Z, 75 s)

Port: L-INTEG run 1 dump `integ/bringup/dump_pp2/step00000` (p0_smoke prefill, 6 tokens, all 40 layers chained, no
golden feeding). Reference: `tf_variants.py --prompts p0_smoke --prompt-only --capture-layers 0-39` (same 6 tokens)
in fp32 / bf16 / bf16+fp16in. Rel-RMS over the whole [6,4,5120] stream (`localise_p0.json`):

| layer | stream_out port vs ref FP32 | port vs ref bf16 | port vs ref bf16+fp16in | ref bf16 vs ref FP32 |
|-|-|-|-|-|
| 0 | 1.70e-3 | 1.30e-3 | 1.28e-3 | 1.72e-3 |
| 2 | 3.39e-3 | 1.55e-3 | 1.81e-3 | 3.40e-3 |
| 7 | 6.49e-3 | 2.03e-3 | 2.08e-3 | 6.11e-3 |
| 14 | 4.34e-3 | 1.77e-3 | 2.30e-3 | 4.31e-3 |
| 20 | 2.50e-3 | 0.98e-3 | 0.87e-3 | 2.43e-3 |
| 24 | 2.49e-3 | 2.50e-3 | 1.33e-3 | 2.69e-3 |
| 28 | 3.12e-3 | 1.96e-3 | 1.59e-3 | 3.16e-3 |
| 38 | 5.72e-3 | 2.08e-3 | 1.77e-3 | 5.61e-3 |
| 39 | 4.60e-3 | 2.04e-3 | 2.07e-3 | 4.47e-3 |
| range L0–39 | 1.7–6.5e-3 | 1.0–2.7e-3 | 0.8–2.4e-3 | 1.7–6.1e-3 |

Final position (the next-token distribution after the prompt): port logits vs ref FP32 (= the golden, identical
computation) rel 0.138, KL 0.0048; vs ref bf16 rel 0.027, KL 0.00074; vs ref bf16+fp16in rel 0.0205, KL 0.00074.
Last-row `final.h` rel: port vs FP32 0.119, vs bf16 0.037, vs bf16+fp16in 0.030 (ref bf16 vs FP32: 0.118).

Readings:
4. **Layer by layer, the port's distance to the FP32 golden equals the BF16-storage distance** (columns 2 and 5 agree
   to ±10 % on every layer), and the port sits 2–3× closer to the BF16-stream reference: 0.8–2.4e-3, i.e. about one
   BF16 rounding of the stored stream (BF16 half-ulp ≈ 2^-9; a uniform rounding error is ≈ 1.1e-3 rel-RMS). There is
   no layer where the port departs from the precision-matched reference — no localisable port error at stream level.
5. On the final next-token distribution the precision-matched reference cuts the port's distance by 5–6× (logit rel
   0.138 → 0.02–0.027, KL 0.0048 → 0.0007).

## Result 3 — attribution and the ref-exact side (run B, 10:25Z, 986 s; metrics `nll_gap.json`)

Pooled over the 4 prompts (2,419 teacher-forced positions, token-weighted; p3 dominates with 2,077):

| variant (v100-semantic numerics, own-k) | vs v100-semantic: top-1 / \|ΔNLL\| / KLlb | vs ref-exact: top-1 / \|ΔNLL\| / KLlb |
|-|-|-|
| ref FP32 (= the golden's precision, scored like the port) | 0.988 / **0.0263** / 0.0032 | 0.981 / 0.0466 / 0.0117 |
| + FP16 sublayer inputs only | 0.989 / 0.0299 / 0.0032 | — |
| + BF16 stream only | 0.984 / 0.0313 / 0.0049 | 0.979 / 0.0468 / 0.0080 |
| + BF16 stream + FP16 sublayer inputs (**the port's precision contract**) | 0.987 / **0.0335** / 0.0040 | 0.977 / 0.0449 / 0.0087 |
| + … + every FP8/FP4-weight GEMM operand FP16 (approximate upper bound) | 0.987 / 0.0341 / 0.0041 | — |
| **port, 8×V100 PP2×TP4 eager** | 0.985 / **0.0320** / 0.0045 | 0.979 / **0.0438** / 0.0081 |

Per prompt vs ref-exact (own-k; ref variants on the ref-exact golden's own continuation ids, p1 shares the ids):
| prompt | ref FP32 | +bf16 | +bf16+fp16in | port |
|-|-|-|-|-|
| p0_smoke | 1.000 / 0.013 | 1.000 / 0.012 | 1.000 / 0.018 | 1.000 / 0.017 |
| p1_legal | 0.983 / 0.033 | 0.983 / 0.039 | 0.991 / 0.033 | 0.991 / 0.033 |
| p2_code | 0.979 / 0.049 | 0.963 / 0.042 | 0.979 / 0.030 | 0.963 / 0.052 |
| p3_doc | 0.980 / 0.048 | 0.980 / 0.048 | 0.976 / 0.047 | 0.980 / 0.044 |

Per-prompt vs v100-semantic, the two single-factor variants: fp16in only p0 0.023, p1 0.021, p2 0.016, p3 0.032;
approx-fp16gemm p0 0.031, p1 0.023, p2 0.032, p3 0.035.

Readings:
6. Attribution of the pooled |ΔNLL| 0.032 the port shows vs the v100-semantic golden: **0.026 is the scoring floor**
   (the FP32 reference itself, scored as one teacher-forced prefill against its prefill+decode golden); the BF16 stream
   alone adds +0.005, FP16 sublayer inputs alone +0.004, together +0.007 (0.0335); the remaining FP16 GEMM operand
   roundings add nothing measurable (+0.0006, approximate). **Port residual: −0.0015 (none).** The BF16 stream is
   therefore responsible for ~0.005 of the ~0.012 by which the port exceeds the 0.02 gate; the rest is the floor.
7. Per prompt, precision-equivalent variants scatter by ±0.01–0.04 (p0: 0.021 vs 0.064 for two variants that differ
   only in FP16 input rounding) — QAT-boundary chaos makes single-prompt |ΔNLL| a noisy statistic; only the pooled
   number is stable to ~0.003.
8. vs ref-exact the port (0.0438) is again inside the band (0.0449–0.0468) and better than the FP32 reference itself;
   L-INTEG's relative-form AM-14 failure on p2 (0.052 vs "ideal" 0.030) is the same chaos: the FP32 reference scores
   0.049 on p2.

## Result 4 — does a baseline-relative gate still catch bugs? (run BUG, 10:34Z, 420 s)

Two planted errors on top of the port's precision contract (bf16 stream + fp16 inputs), `tf_variants.py` variants
`+bugE14` (layer-14 Engram skipped) and `+bugSink20` (layer-20 attention sink removed), scored like the port:

| variant | pooled vs v100-semantic: top-1 / \|ΔNLL\| / KLlb | per prompt \|ΔNLL\| (p0/p1/p2/p3) | vs the clean bf16+fp16in run (same ids): top-1 / \|ΔNLL\| / KL |
|-|-|-|-|
| bf16+fp16in (clean) | 0.987 / 0.0335 / 0.0040 | 0.064 / 0.018 / 0.024 / 0.035 | — |
| + Engram L14 skipped | **0.961 / 0.0940 / 0.0380** | 1.502 / 0.106 / 0.082 / 0.069 | 0.961 / 0.090 / 0.037 |
| + L20 sink removed | 0.987 / 0.0329 / 0.0041 | 0.049 / 0.021 / 0.023 / 0.034 | 0.997 / 0.009 / 0.0003 |

Reading 9: an error the size of a missing Engram layer moves pooled |ΔNLL| by +0.06 and is caught by any of the
gates; an error whose own end-to-end footprint is < 0.01 nats (a removed sink at one late layer) is **invisible
end to end**, with or without the baseline — it is below the floor of the scoring itself. End-to-end gates detect
gross errors only; subtle ones are the per-op golden-fed tests' job (§4.5 rows, AM-15) and the chained-stream check of
Result 2.

## Conclusion and recommendation

| pooled over p0–p3 (2,419 pos.), vs v100-semantic+ownk | top-1 | \|ΔNLL\| | KLlb |
|-|-|-|-|
| AM-13 gate | ≥ 0.98 | ≤ 0.020 | ≤ 0.010 |
| ref FP32 = the golden's own precision, scored like the port (floor) | 0.988 | 0.0263 | 0.0032 |
| ref + BF16 stream (A4) | 0.984 | 0.0313 | 0.0049 |
| ref + FP16 sublayer inputs | 0.989 | 0.0299 | 0.0032 |
| ref + BF16 stream + FP16 inputs = port precision contract ("ideal") | 0.987 | 0.0335 | 0.0040 |
| **port (8×V100 PP2×TP4, eager)** | **0.985** | **0.0320** | **0.0045** |
| ideal + planted bug: Engram L14 skipped | 0.961 | 0.0940 | 0.0380 |

**Answer.** The BF16 stream explains ≈ 0.005 of the port's pooled |ΔNLL| 0.032 (FP32 0.026 → 0.031); FP16 sublayer
inputs add ≈ 0.002 more (0.0335). The largest share, 0.026, is not the port at all: the FP32 reference scored the way
the port is scored (one teacher-forced prefill vs the golden's prefill + greedy decode) already misses the 0.02 gate
on p0/p2/p3. The port sits inside the precision-matched band (0.0320 vs 0.0313–0.0335; vs ref-exact 0.0438 vs
0.0449–0.0468) and, layer by layer on p0, within about one BF16 rounding (0.8–2.7e-3 rel-RMS) of the BF16-stream
reference while 1.7–6.5e-3 from the FP32 golden — the same distance as BF16 storage alone. **The gap is explained by
the precision contract plus the scoring floor; no residual port error is detectable.** The p3 "KL 0.0113" in the
L-INTEG table is the informational vs-official-golden entry; vs the gating own-k golden it is 0.0045 (passes).

**Proposed AM-13′ (replaces the absolute AM-13 thresholds).** The end-to-end check scores the
port AND the precision-matched reference (`tf_variants.py --variants bf16+fp16in`, i.e. `--stream-dtype bf16
--fp16-sublayer-inputs`, own-k, one teacher-forced prefill over the same ids) against the `v100-semantic+ownk`
golden on p0–p3 and gates the **pooled, token-weighted** statistics:
* |ΔNLL|_port ≤ |ΔNLL|_ideal + 0.01 (today 0.0335 + 0.01 = 0.0435: port 0.0320 PASS; Engram-bug 0.094 FAIL);
* top-1_port ≥ top-1_ideal − 0.01 and ≥ 0.97 (port 0.985 PASS; Engram-bug 0.961 FAIL);
* KLlb_port ≤ KLlb_ideal + 0.005 (0.0090: port 0.0045 PASS; Engram-bug 0.038 FAIL).
Per-prompt numbers are reported, not gated (precision-equivalent runs scatter by 0.01–0.04 per prompt). The ideal
numbers are fixed for these goldens (`nll_gap.json`) and need re-running only when the goldens change.
**AM-14′** the same form vs `ref-exact+ownk`: pooled |ΔNLL| ≤ ideal 0.0449 + 0.01 = 0.0549 (port 0.0438 PASS), top-1 ≥
ideal 0.977 − 0.01 (port 0.979 PASS). The absolute AM-14 (≤ 0.06) can stay as a backstop.

Further recommendations:
1. Sharp detection lives below the end-to-end level: add a precision-matched golden mode to `golden.py` (wire
   `R.add_precision_args`; case suffix e.g. `+port`) so L-INTEG's chained dumps can be checked per layer against it
   (p0 today: ≤ 2.7e-3 vs the BF16-stream reference; a gate of 5e-3 on short prompts would have margin) — not done in
   this lane.
2. L-INTEG should save the port's teacher-forced logits as tensors (all positions, FP32) instead of only the top-20
   `prompt_logprobs`; that gives the exact KL and allows direct port-vs-ideal comparisons.
3. The scoring floor (0.026) comes from scoring a prefill against a prefill+decode golden; a teacher-forced FP32
   golden (prefill over prompt + continuation, `tf_variants.py --variants fp32`, already on disk) would remove the
   decode part of that floor for the continuation positions if the coordinator prefers an absolute gate.

## Artifacts

Code (not committed; coordinator commits): `tools/ds41_ref/ref_model.py` (Sequence `stream_dtype`,
`fp16_sublayer_inputs`, `Sequence.st/act`, `add_precision_args`, FP16 head option), `tools/ds41_ref/tf_variants.py`
(teacher-forced variant runner, capture + planted-bug options), `tools/ds41_ref/nll_gap.py` (metrics + dump
localisation). Data: `[private path]` — `tf/` (run A), `tf_B/`, `tf_bug/`, `dump_ref/` (run C),
`nll_gap_all.json` (every row), `localise_p0.json`, `check_hc.py` (CPU equivalence check vs the port's hc.py),
`run_tf.sh`, `logs/`. GPU c76e1399 used 09:58–10:34Z only (4 runs: 507 s, 75 s, 986 s, 420 s).
