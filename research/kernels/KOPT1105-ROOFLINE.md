# D51R5 prefill MoE: CPU roofline and census analysis

> **Origin.** PLI Labs research archive, file `kernel-research/KOPT1105-ROOFLINE.md`. Published 2026-11-18 in the Volta V100 research
> repository.
> **Edited for publication.** Machine names, private file paths, network addresses, session names and the
> operator's name were replaced with generic terms (`the V100 server`, `[private path]`, `[address]`,
> `the owner`). Links to files that are not published were turned into plain text. No figure and no finding
> was changed. Dates inside the note are its own; later results are in the build pages of this repository.

Evidence window: `[private path]`. Analysis performed without GPU access. **Bottleneck labels below are hypotheses; hardware counters are unavailable.** This is an optimization plan input, not performance qualification.

The strongest new finding is a tuning-coverage gap. The selected tile is M64/N128/K32, grid160, for every codec/role. Across the actual PP0/rank0 8188-token census, this tile performs about35.75% padded row work. A mixed full64/tail32-or64 schedule removes19.56% of issued row-FMAs without increasing the number of output CTAs. The historical micro suite never selected these distributed histograms: its18 selections are4096-aligned hot experts or a single routed row.

## Reproducible artifacts and scope

- `derive_roofline.py`: CPU-only derivation; source path is explicit, no recursive directory traversal.
- `roofline-shapes.csv`: all **36,547 distinct** `(codec,role,tokens,descriptor_m,N,K,32 expert counts)` shapes among **153,600** records, with occurrence weights, PP0/rank0 occurrence weights, classification hypotheses and arithmetic/traffic floors. These are all shapes, not merely the18 micro selections.
- `roofline-summary.json`: source SHA256, assumptions,24 codec/role/token aggregate cells, PP0/rank0 padding and18 micro rows.
- Derivation asserts nonnegative32-expert histograms, sum=high−low, N divisible128, K divisible32, and exact candidate/baseline micro histogram alignment. It does not execute or import a GPU implementation.

Run from this directory:

```bash
python3 derive_roofline.py
```

The census contains tokens7,2616,8188,8192; layers0–39; ranks0–3. GU is `(N,K)=(1152,5120)`; DN is `(1280,2304)`. Descriptor M is the larger output allocation, **not the sum of the32 bank-local expert counts**. Use the latter for useful FLOPs. Source `check/capture.py:25–31` establishes geometry and validates the bank-local counts.

## Hardware ceilings and uncertainty

At the brief's observed1290MHz and150W operating point, theoretical FP16-input/FP32-accumulator HMMA capacity is:

`80 SM × 8 tensor cores/SM × 64 FMA/core/cycle × 2 FLOP/FMA × 1.290e9 cycle/s = 105.6768 TFLOP/s`.

This is an arithmetic ceiling at an assumed clock, **not a measured sustained rate at150W**. Power-limited clock variability and instruction issue reduce realizable throughput. V100 has80 enabled SMs,640 Tensor Cores,6MiB L2 and nominal900GB/s HBM bandwidth; a Tensor Core performs64 FMAs per cycle. [NVIDIA V100 architecture whitepaper, hardware table and Tensor Cores section](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)

Register/shared capacity is65,53632-bit registers/SM,96KiB shared/SM,64 warps/SM. The nominal HBM figure is a ceiling; additional in-flight work is needed to approach it at low occupancy. [NVIDIA Volta tuning guide, occupancy and memory-throughput sections](https://docs.nvidia.com/cuda/volta-tuning-guide/)

For an **optimistic shared-memory service model**,32 banks ×4 bytes/bank/cycle ×80SM ×1.290GHz gives13.2096TB/s, before bank conflicts, issue limitations, barriers and address-generation work. The per-bank premise is the conventional conflict-free CUDA bank model; the result is an estimate, not a V100 benchmark. [NVIDIA shared-memory bank explanation](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)

No sustained L2 bandwidth was measured. CSV columns at2TB/s and4TB/s are **sensitivity scenarios**, never asserted V100 specifications. A one-expert canonical GU bank occupies2.21MB for IQ1_M/IQ2_XXS or2.95MB for IQ2_XS; DN is1.11/1.47MB. One or two weights can fit within6MiB L2, but the full32-expert bank and activation streams cannot. Cache reuse depends on task order and competing traffic. N-fastest traversal sweeps N tiles before the next member tile, so the source comment about weight reuse is not proof of ideal L2 reuse.

## Canonical storage and formulas

The live canonical format differs from original GGUF bits per weight:

| Codec | Packed code bytes/weight | Metadata | Total bytes/weight | Shared codebook |
|---|---:|---:|---:|---:|
|29 IQ1_M|0.25|2B/16 weights|0.375|16384B|
|16 IQ2_XXS|0.25|4B/32 weights|0.375|2048B|
|17 IQ2_XS|0.25|4B/16 weights|0.500|4096B|

Source: `csrc/prefill_iq.cu:77–81`; packaged `arch/config_sm70_s884.h:140–145`; `lattice_codebooks.h:75–77,213–215,777–779,1125`. Do not use IQ1_M's original1.75bpw or IQ2_XXS's2.0625bpw for this kernel's bank traffic.

For each histogram, let `R=sum(m_e)`, `E=count(m_e>0)`, `T=sum(ceil(m_e/64))`, `J=N/128`, and `b` equal canonical bytes/weight above.

- Useful FLOPs: `F=2RNK`.
- Padded issued FLOPs model: `Fpad=2×64T×N×K` (M-masked loads/stores do not remove all HMMA on padded rows).
- Tile count: `TJ`; persistent grid160 launches160 CTAs, each processing its stride of these tasks. With fewer160 tasks, some CTAs immediately return.
- Compulsory weight bytes if each active expert is read once: `W=ENKb`.
- Route-ideal traffic reference: `Bideal=W+2RK+2RN`, charging one A read per routed row and one output write. A token may repeat across experts, so this is **not a strict minimum unique-address count**. No route-index trace is invented.
- Logical tiled global traffic: `Blogical=T×N×K×b+2RKJ+2RN`, ignoring pointer/plan traffic, line overfetch and cache reuse. `Blogical/900GB/s` is a no-cache model, **not measured HBM traffic and not a rigorous bandwidth lower bound when caches hit**.
- Tensor floor: `Fpad/105.6768TF/s`; route-ideal HBM reference: `Bideal/900GB/s`.
- Optimistic shared service bytes: `2×(2×64T×K×J+T×N×K×b)` for packed stage stores+one read; `T×N×K` for one codebook byte per decoded weight; `8×64T×N` for FP32 epilogue rearrange store+read; `TJ×codebook_bytes` for table initialization. Divide by13.2096TB/s. This deliberately excludes replicated warp fragment reads, repeated metadata/lookup reads, bank conflicts and barriers, so is only a low reference cost.

These floors overlap in execution; **do not add them**. Take their maximum only as a coarse roofline reference, then account for decode, occupancy, synchronization and instruction issue. The API/event duration also includes planner and host-dispatch effects; it is not a pure HMMA kernel duration.

## Recorded resource use

`dispatch.json.resources` contains the actual window's CUDA attribute/occupancy results. `resources.json` independently records ptxas registers,64B stack frame and zero spill stores/loads for all18 specializations. A64B stack frame must not be called a spill. One discrepancy is IQ1_M/M32N64: runtime attributes122 registers versus ptxas120; use the runtime record for occupancy and preserve the discrepancy rather than silently normalize it.

| Tile | M×N | Threads | IQ1_M regs/shared/blocks | IQ2_XXS regs/shared/blocks | IQ2_XS regs/shared/blocks |
|---|---|---:|---|---|---|
|0|32×64|128|122 /22048B /4|120 /8208B /4|120 /10272B /4|
|1|64×64|128|128 /26144B /3|149 /16400B /3|149 /16400B /3|
|2|128×64|128|211 /34336B /2|208 /32784B /2|214 /32784B /2|
|3|32×128|128|127 /23584B /4|135 /16400B /3|135 /16400B /3|
|4, selected|64×128|128|176 /32784B /2|212 /32784B /2|218 /32784B /2|
|5|128×128|256|176 /65552B /1|209 /65552B /1|212 /65552B /1|

Selected tile4 permits8 resident warps/SM,12.5% theoretical warp occupancy. Grid160 corresponds to two CTAs per80SM when all tasks are live; it cannot exploit a3–4CTA/SM small-tile configuration. Grid80 would halve active CTAs and is not an obvious large-shape improvement. Large grid is still not a guarantee of balanced final waves.

Tile4 shared memory is **32KiB+16B**, which itself prevents three resident CTAs at96KiB. Registers also prevent three: trimming16 shared bytes alone does not fix occupancy. Three128-thread blocks need shared≤32768B and approximately≤168registers/thread after allocation granularity. The epilogue FP32 scratch is a candidate to shrink (smaller epilogue subtiles), paired with live-register reduction. Occupancy is a latency-hiding resource, not an objective to maximize regardless of reuse.

## Measured micro rows against model floors

All numbers below are microseconds except TF/s. These are selected tile4/grid160 cold-cache medians from the fixed R5 window. HBM column is the **no-cache logical-load model**, not measured HBM time; shared column is the optimistic service model above.

| Codec/role/histogram | R | Candidate | Tensor floor | HBM no-cache | Shared optimistic | Useful TF/s |
|---|---:|---:|---:|---:|---:|---:|
|29/gu/longest|8192|2721.8|914.5|1174.4|221.5|35.50|
|29/gu/shortest|1|195.6|7.1|2.6|1.7|0.06|
|29/gu/skewed|4096|1251.3|457.2|587.2|110.7|38.61|
|29/dn/longest|24576|3900.4|1371.7|1800.1|345.3|37.16|
|29/dn/shortest|1|120.8|3.6|1.3|0.9|0.05|
|29/dn/skewed|4096|619.5|228.6|300.0|57.6|39.00|
|16/gu/longest|24576|8146.4|2743.4|3523.2|660.7|35.59|
|16/gu/shortest|1|191.0|7.1|2.6|1.7|0.06|
|16/gu/skewed|4096|1434.1|457.2|587.2|110.1|33.69|
|16/dn/longest|12288|1979.4|685.8|900.0|170.6|36.62|
|16/dn/shortest|1|101.4|3.6|1.3|0.9|0.06|
|16/dn/skewed|4096|634.9|228.6|300.0|56.9|38.05|
|17/gu/longest|12288|4203.0|1371.7|1918.9|352.0|34.49|
|17/gu/shortest|1|206.8|7.1|3.4|1.8|0.06|
|17/gu/skewed|4096|1463.8|457.2|639.6|117.3|33.01|
|17/dn/longest|12288|2009.1|685.8|978.7|181.6|36.07|
|17/dn/shortest|1|108.0|3.6|1.7|0.9|0.05|
|17/dn/skewed|4096|636.4|228.6|326.2|60.5|37.96|

Large selected histograms deliver33–39 useful TF/s,31–37% of the105.7TF/s ceiling. Their measured durations exceed even the no-cache logical HBM model by about2× or more. This is consistent with instruction/decode/shared-memory and occupancy costs; it does **not** uniquely prove decode-bound execution. Small one-route rows have1/64 row utilization and only9GU or10DN live tiles; latency, underfill and padding dominate the useful-work ratio. `counters.json` reports `ncu_permission / ERR_NVGPUCTRPERM`; `dispatch.json` reports `no_cuda_events`. Missing measurements remain INCONCLUSIVE.

For the remaining census shapes, CSV classification is deliberately explicit: inactive; latency/tail/underfill hypothesis when fewer160 tasks; otherwise decode/shared/occupancy unresolved hypothesis. There is no factual basis to label every unmeasured shape tensor-core-bound or shared-bound. New event timings and counters are the discriminating tests.

## Offline SASS: what it establishes

Parent lane compiled the pinned source on CPU using CUDA12.4; artifacts are `evidence/compile/baseline/prefill_iq.sass` and `evidence/baseline-sass.json`. Production R5 used CUDA12.8, so compiler resource differences are preserved.

| Codec, tile4 | Offline registers | Static instructions | HMMA instructions | LDS | STS | PRMT | BAR.SYNC |
|---|---:|---:|---:|---:|---:|---:|---:|
|29|176|5487|256|66|49|198|15|
|16|219|5646|256|66|49|202|15|
|17|210|5694|256|66|49|206|15|

All18 offline specializations report zero spill stores/loads. Static counts include initialization, epilogue, error paths and loops; they are not dynamic instruction counts or cycle shares. For example,64static atomic-CAS instructions and32CALL instructions occur within the generic emitted code, but the candidate has no split-K task reduction. Their presence does not prove active atomic execution. The hot branch for alpha1/beta0 and plain FP16 output should be inspected before specializing away generic epilogue paths.

The tensor instruction family is HMMA.884 with FP32 accumulation. Packed B operands are staged in shared memory, then `lattice_transform.h:34–85` decodes eight-weight packets to half fragments in registers. For each pair the source performs byte permutation, half-pair bias subtraction, sign handling (or IQ1_M delta addition), and half multiplication. That fragment is reused across the within-warp M accumulator loop; it is **not decoded afresh for every individual output row**. Different M-warp partitions duplicate B decode, and each subsequent M CTA repeats it. The selected tile partitions warps2×2; a dequant-once shared-B prototype should target that duplication and larger-M reuse, accounting for the extra FP16 shared footprint.

`mainloop_sm70.h:228` initializes the codebook for each tile invocation, including each iteration of `compact_gemm`'s persistent traversal. The same mainloop already uses two shared stages and register-prefetched global loads. A new proposal called merely “double buffer” duplicates existing behavior; worthwhile variants are deeper lookahead, reordered loads/decode/HMMA, or retaining codebook across tiles with separate storage lifetime. SM70 does not gain SM80 `cp.async` merely by including CUDA pipeline headers.

## Why the micro selection misses ordinary tails

`check/microbench.py:50–61` restricts to rank0, then selects maximum route count, minimum positive route count, and maximum skew across **both PP stages and all token sizes**. It does not stratify by PP stage, chunk size, frequency or tail utilization. In this exact window:

| Codec/role | Longest selection: tokens/layer/R/active experts | Shortest | Most skewed |
|---|---|---|---|
|29/GU|8192/20/8192/2|7/0/1/1|8192/20/4096/1|
|29/DN|8192/37/24576/6|7/0/1/1|8192/20/4096/1|
|16/GU|8192/37/24576/6|7/10/1/1|8192/22/4096/1|
|16/DN|8192/22/12288/3|7/7/1/1|8192/22/4096/1|
|17/GU|8192/39/12288/3|7/17/1/1|8192/27/4096/1|
|17/DN|8192/39/12288/3|7/23/1/1|8192/23/4096/1|

Thus every longest/skewed exemplar is from PP1, whereas the requested budget is PP0. All selected longest/skewed active experts have4096rows, so M32/64/128 have essentially no row-tail difference. Every shortest exemplar is a one-route call. None of the138240 census records at8188tokens or3840 records at2616tokens is selected. The census itself is an instrumentation fixture, not a measured probability distribution of future user workloads; repeated ranks and repeated request/chunk fixtures must not be treated as independent traffic observations.

The candidate's frozen18-cell aggregate is29923.84µs versus35291.15baseline and34906.11warm-tuned:15.21% and14.27% lower respectively. It is an unweighted sum of selected micro cells, **not elapsed time for an8192-token PP0 chunk**. Re-selecting the best of12existing tile/grid choices separately for each of those18histograms yields29372.93µs, only1.84% further aggregate reduction in hindsight. This is selected on the same data, not validated performance. Shape-stratified tuning has a stronger justification in the missing tail distributions than this1.84% suggests.

## Tail opportunity arithmetic

The table uses only rank0, PP0/layers0–19 to avoid TP rank replication and PP1 hardware-budget confusion. Occurrences reflect this census capture, not general serving frequency.

| Tokens | Records | Live rows | Padded M64 rows | Mixed64/tail32-or64 rows | Issued-row work removed |
|---|---:|---:|---:|---:|---:|
|7|480|1680|70912|35456|50.00%|
|2616|480|627840|1171840|871040|25.669%|
|8188|17280|35372160|55055872|44287552|19.559%|
|8192|960|1966080|1966336|1966208|0.0065%|

For each expert retain full64-row CTAs. For the last partial tile, useM32 when remainder1–32, otherwiseM64 when remainder33–63. This achieves the same padded-row count as all-M32 tiling **without doubling the full-tile CTA count**. A task must carry an explicit row base or a tile-id interpretation consistent with its M variant; changing BM without that adjustment overlaps or skips rows. There is no numerical K-reduction change, but altered mapping still requires byte comparison and the approved near-tie policy if byte identity fails. Additional tail launches, resource union size and worse warp reuse can consume the arithmetic saving.

For8188tokens, utilization improves `35372160/55055872 =64.248%` to `35372160/44287552 =79.869%`. The25.669% and19.559% numbers are **issued-FMA savings**, not wall-time forecasts. Since GU and DN have differentK/N products, a combined arithmetic FLOP-weighted result must weight each row by its role's2NK; `roofline-shapes.csv` supplies these values. An independent role-weighted recomputation on this capture gives the same19.559%/25.669% reductions because the GU/DN aggregate counts balance; this coincidence must not be presumed for a new capture. A mixed M16/32/64 schedule can reduce padding further (PP0 issued-row savings28.386% at8188tokens), but M16 is not currently supported and the smaller-fragment/launch tradeoff is larger.

A reasonable PLAN sensitivity calculation is `saved_chunk_time = affected_GEMM_seconds × affected_fraction × realized_fractional_speedup`. For illustration only, if0.8s/chunk is affected, half of it benefits, and realized speedup corresponds to10–20% duration reduction, the saving is0.04–0.08s/chunk. Re-measure the affected duration before forecasting more precisely; do not multiply1.40sFFN by19.56% as though everyFFN operation were padded HMMA.

## Required next measurements (queued, not executed here)

1. Preserve the exact old18cell replay so historical micro results remain comparable. Add separate PP0/PP1 × token-size × codec/role strata, with representative route-count/tail/entropy bins. `roofline-summary.json.stage_groups` supplies record and live-route counts for all ranks and rank0 separately. Dedupe geometries within strata, keep multiplicity as metadata, and report both fixture-weighted and equal-stratum results; measure real request frequency separately.
2. Freeze tuning selections on a public/non-heldout calibration capture, then evaluate on a separate allowed public capture. This lane never accesses a heldout directory. Do not choose and claim a gain from the same eight timing samples.
3. Measure per-call GEMM/planner duration and actual kernel duration separately. For representative large/tail/tiny shapes collect tensor active cycles, eligible warps, register occupancy, local-memory bytes, shared bank conflicts/wavefronts, L2 hit rate/bytes, HBM bytes, integer/decode issue and barrier stalls where permissions allow. Unavailable counters stay unavailable; event-only performance decisions remain possible.
4. Compare fixed160 against0 and any new grid per shape; include active-task count and final-wave balance. Measure full+tail scheduling versus all-M32 as well as currentM64. Retain original arithmetic, full-width checkpoint oracle and untouched-output guards.
5. Only after independent numerical and sanitizer qualification compare PP0 per8192token chunk and public long-prompt TTFT under the approved full gates. GPU work **runs after the benches and GitHub updates complete**.
