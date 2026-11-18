# UPSCAN1105 — ranked upstream opportunities

> **Origin.** PLI Labs research archive, file `kernel-research/UPSCAN1105-REPORT.md`. Published 2026-11-18 in the Volta V100 research
> repository.
> **Edited for publication.** Machine names, private file paths, network addresses, session names and the
> operator's name were replaced with generic terms (`the V100 server`, `[private path]`, `[address]`,
> `the owner`). Links to files that are not published were turned into plain text. No figure and no finding
> was changed. Dates inside the note are its own; later results are in the build pages of this repository.

| Rank | Opportunity / primary evidence | KCAMP | Expected effect on OUR build; evidence status | SM70 / effort |
|---|---|---|---|---|
| 1 | Full multi-request DSpark verify, expert/dense weight reuse, batched drafter: [Maya API](https://github.com/mw00/project-maya/blob/c1670044bbc09310f35f9d00f7596c50ae8a38f1/src/ds4/ds4_model.hpp#L281-L311), [1Cat #1369](https://github.com/1CatAI/1Cat-vLLM/pull/1369), DSX row-budget design | C0/C1/C2 | KCAMP FORECAST: four 6-row windows sharing a forward costing 1.3–1.6 single windows gives 4/(1.3–1.6)=2.5–3.08× aggregate. No source demonstrates this result on our configuration. Verify 24-row paths end to end. | Algorithms portable; IQ2 kernels need FP16 HMMA/FP32 order; large, ~2–4 weeks |
| 2 | Two request microbatches in flight across PP2: prior philbert analysis, [MRV2 DBO design](https://github.com/vllm-project/vllm/pull/61464) | C3/C2 | Local trace shows ~9.4 ms stage wait in older profile. Balanced pipeline ceiling approaches 2×; KCAMP's 1.7–1.9× is an ESTIMATE. DBO is a design reference, not implemented PP2 support. Do not multiply by rank1 without joint measurement. | Host/stream ownership portable; large, ~2–4 weeks |
| 3 | Decode-priority cadence between original prefill chunks: [Maya scheduler](https://github.com/mw00/project-maya/blob/c1670044bbc09310f35f9d00f7596c50ae8a38f1/src/ds4/ds4_main.cpp#L1970-L2034), DSX AMD runner | C4 | Local mixed cohort decoder fell to ~9.8 tok/s. SOURCE CLAIMS: Maya p95 TTFT29.1→13.5s, wall unchanged; DSX background aggregate70.12→146.80 tok/s with TTFT22.18→24.34s. Local magnitude unknown; preserve 8,188 numerical boundaries. | Policy portable; medium/large, ~1–2 weeks |
| 4 | Exact-order fused sparse prefill attention: [SM70 Maya reference](https://github.com/mw00/project-maya/blob/c1670044bbc09310f35f9d00f7596c50ae8a38f1/src/ds4/ds4_kernels.cu#L4208-L4438), prior philbert port | S1 | Already F1 candidate-ready. Our mode2 MEASURED TTFT−10.4..−11.4% but failed numerics; mode3 must demonstrate both exactness and retained speed. This is corroboration, not a new discovered win. | Genuine SM70 kernels; remaining validation/integration ~1–3 days if candidate holds |
| 5 | Overlap/cut residual prefill handoff, retaining stored values and dtype: [1Cat #1294](https://github.com/1CatAI/1Cat-vLLM/pull/1294), Maya chunk pipeline | S2 | Residual~0.6–0.7s of3.64s/8K chunk =>16.5–19.2% removable-time ceiling; KCAMP conservatively forecasts up to~15% TTFT. #1294 already carried: find residual rather than reapply it. | PP transport portable; medium/large, ~1–2 weeks |
| 6 | Shape-specific grouped IQ expert prefill tiles: [Maya Volta IQ WMMA](https://github.com/mw00/project-maya/blob/c1670044bbc09310f35f9d00f7596c50ae8a38f1/src/ds4/ds4_iq.cu#L1075-L1165), 1Cat batched paths | S3/C1 | Our adopted D51 MEASURED real-route kernels−16.6%, whole TTFT−2.4..−4.3%. Incremental KCAMP FORECAST0.12–0.25s/3.64s=3.3–6.9% TTFT ceiling, not another16.6% whole-model gain. | Adapt codec/tile/reduction order; medium, ~3–10 days; F2 already running |
| 7 | Producer writes next consumer's layout; remove repack/quantization launches: [1Cat #1367](https://github.com/1CatAI/1Cat-vLLM/pull/1367), Maya HC/quant fusions | S6/C1 | SOURCE CLAIM #1367 removes~180 c4 launches, saves~0.2–0.5ms on Qwen. If we removed0.2–0.5ms from22.05ms, conditional gain0.9–2.3%. First count equivalent V4.1 launches; they may not exist/already be fused. | SM70 pattern; QPN/K16 code is not IQ2 drop-in; medium, ~3–7 days |
| 8 | Reclaim prefill temporaries by row-block Engram gather and disjoint-lifetime scratch reuse: [1Cat issue #1342](https://github.com/1CatAI/1Cat-vLLM/issues/1342), [DSX allocator](https://github.com/lxe/dsx/blob/60e9870b74509825f2f8de088306d82260f0fd87/nvidia/src/gpumodel/prefill.rs#L329-L365) | New/C0/C5/S4 | SOURCE CLAIMS: Engram 800→100 MiB per stage-0 GPU at 8K; DSX scratch~1.91GB at8K. Different allocations; neither verified free on our runtime, so do not add them. Capacity benefit first, no automatic tok/s gain. | Format-independent lifetimes; small/medium audit+patch, ~2–5 days |
| 9 | Fused multi-request sparse attention/indexer; avoid extra gather/materialization: [Maya implementation](https://github.com/mw00/project-maya/blob/c1670044bbc09310f35f9d00f7596c50ae8a38f1/src/ds4/ds4_model.cu#L21072-L21179), [1Cat #1347](https://github.com/1CatAI/1Cat-vLLM/pull/1347) | C1/C5/S10 | Maya SOURCE CLAIM aggregate43.9→49.3 no drafts;52.0→56.3 with drafts. Our S10 FORECAST1–3ms/round at128–256K only if remaining gather/top-k is exposed. #1347 explicitly excludes V4.1 and overlaps ideas already ported into our scorers. | Generic CUDA algorithm portable; extend source16-row cap to24; large, ~1–3 weeks |
| 10 | Target-verified prompt/suffix lookup proposals for repetitive code edits: [Maya suffix drafter](https://github.com/mw00/project-maya/blob/c1670044bbc09310f35f9d00f7596c50ae8a38f1/include/strata/spec/suffix_drafter.hpp#L1-L53), DSX AMD speculator | New/C1 | SOURCE CLAIMS Maya84→111/96→144 tok/s on whole-file edits; DSX lookup91.5 vs64.2/request historically. No general V100 forecast; requires actual repetition and cheaper accepted proposals than DSpark. Plain prose may gain nothing. | Hardware independent policy; target verification/row limits and sampled coupling need care; medium, ~4–8 days |

Completed 2026-11-05, read-only. Ranking weighs existing measured bottlenecks, incremental value beyond our deployed stack, portability and effort. Days/weeks are engineering judgments, not commitments or measured runtime. **No GPU, server, production configuration or code was changed.** No build or benchmark ran in this lane.

## What changed in the upstream picture

The scan covers all states of 1Cat PRs created **or updated** since 2026-10-28T00:00:00Z, every main commit returned since that timestamp, and issue-topic matches across all dates. Latest observed main is `3c051b52ed873475b01da07be3e9e1fe9ab8ae7e`, dated2026-11-05. A PR updated recently can contain code merged months ago; the census separates that from new main changes.

The new November5 batching work #1367/#1368/#1369 is valuable because it targets real SM70 multi-request verification shapes, but it is Qwen/QPN8/native-FP4/GDN work. It does not supply DeepSeek-V4.1 GGUF IQ2_XXS batched kernels or PP2/MRV2 request ownership. Maya and DSX demonstrate independent batched request-state designs; Maya has a16-row cap, while our four*(five drafts+one target)=24 rows requires extending every relevant owner, not only the scheduler.

The public evidence supports C1/C2/C4 as useful engineering directions. It does **not** establish the proposed combined4–5× aggregate result: DSX's reported four streams are408–432 tok/s against246 alone (1.66–1.76×), and Maya's RAM-heavy V100 example reports1.26× at four streams. Different residency, quantization and bottlenecks prevent a direct comparison. C1 and C3 gains share resources and must be measured together.

## 1Cat detail and complete census

- ONECAT.md: relevant changes, source claims, source-file/diff evidence, stack relationship, applicability, effort and KCAMP mapping.
- ONECAT-CENSUS.md: **110 PRs** with number/title, author, state, created/updated/merged dates, published-series presence and applicability.
- ONECAT-COMMITS.md: **68 main commits**, full SHA links, authors and author/committer dates, baseline relationship.
- ONECAT-ISSUES.md: **117 topic-title issue matches across all dates**, with author/state/dates. ONECAT-ISSUES-BODY.md adds **57 body/spelling matches** after searching all **207 non-PR issues**: **174 candidates screened, 33 nonmatches**. Substantive issue bodies are assessed in ONECAT.md.
- Raw evidence: exhaustive API pages, PR/issue bodies and selected changed-file patches. Commit and PR metadata were read live, not inferred from the frozen series.

Published base is #1277 merge`899e9ed33`, updated main`6db9c81`, plus25 named PR patches. Membership is checked against `forks/1cat-4xv100-tp4/tools/series.in` and `SERIES.tsv`. This is **source inclusion**, not proof that every generic kernel reaches V4.1/MRV2. New merges of already-carried PRs do not supply another speed gain. The four-card series also predates some eight-card adopted levers; the separate production levers record supplies that comparison.

Already covered: #1263 keep-batch (our port), #1294 eager-send overlap, #1280/#1282 prefill design, #1221 MRV2/DSpark port, issue#1341 hot-refill repair, coupled drafts/shards, routed expert instruction diet, tiled decode scorer and fused prefill scorer. Do not rank these again as new patches. #1347 is a design reference only for this model. Issue#1361's+28% MTP synchronization observation applies to the Flash-V100 EAGLE/MTP metadata path; no proof that our DSpark path pays it.

Additional issue-body screening found request-local drafter-length mismatches (#668/#424), speculative c4 corruption despite normal acceptance (#656), dummy-capture metadata freezing the no-history branch (#728), and pending-transfer/failed-load/per-group watermark ownership (#1249/#1238/#632). These reinforce C2 and resource ownership in C0/C3/C5; they establish no new defect in our deployed paths and do not change the ranking.

1Cat root [licence at the observed head](https://github.com/1CatAI/1Cat-vLLM/blob/3c051b52ed873475b01da07be3e9e1fe9ab8ae7e/LICENSE) is Apache-2.0; borrowed kernel/subtree notices must remain with any eventual port. No port was performed.

## Maya detail

Pinned clone: `upstream/project-maya` at`c1670044bbc09310f35f9d00f7596c50ae8a38f1`. Root engine licence MIT. MAYA.md inventories **28 technique families**, each mapped to KCAMP or new, with SHA-pinned file:line citations, portability, effort and source/arithmetic effects.

Most useful beyond the ranking: exact native-grid KV storage; output-head8/16-row weight reuse; BF16 router **storage** widened toFP32 without BF16 tensor cores; async deduplicated Engram prefetch; partial restore points for edited turns; incremental prompt tokenization; cost-aware draft budgeting. Implementation evidence and source timing claims are separated in that inventory. Dynamic drafting objectives are portable, but its hidden cost clamps/periodic probing are not instructions to transplant those policies into our fail-fast code.

Restore points are a potentially useful **new cache-state feature**, not a new prefix-cache multiplier. Our existing repeat500K prompt already reused499,616 tokens in2.2s. Maya's edited18K TTFT15.0→0.2s costs first-read16.7→17.7s, and applies to different cache semantics. A distinct benefit must be demonstrated for edits/partial compressed-state restoration our current cache misses; retained state must include window rings, partial compressor groups, compressed/index KV and drafter state on both PP stages. Conditional saving is reused_tokens/current_prefill_rate minus save/restore overhead, not the source's50× label.

Do not mix model benchmarks: Maya's4×128K/80–84 aggregate tok/s record is **GLM-5.3-Flash**. Its DeepSeek4×V100 Maya-S64K source claim44 decode/592 prefill tok/s uses different model quant/residency and does not suggest replacing our engine. NVIDIA SM80 INT8 tensor cores/cp.async, SM90 PDL, AMD MFMA/WMMA and Metal kernels cannot run unchanged on Volta. Maya's actual SM70 FP16 WMMA code is portable in principle, but its different score/reduction order must pass our exactness rules.

## DSX detail

Pinned clone: `upstream/dsx` at`60e9870b74509825f2f8de088306d82260f0fd87`. Root licence MIT. DSX.md inventories **27 technique families** with exact source locators, claims versus code evidence, KCAMP mapping, effort and exclusions. Local source links can be resolved under the pinned clone; permalink base is recorded there.

Its NVIDIA implementation uses sequential conversation parking; **continuous batching is in the separate AMD/HIP backend**. NVIDIA SM70 attention selects serial CUDA-core code while newer tensor-core functions are trap stubs, so that attention is not a replacement for our SM70 tensor-core work. T16/MXFP4/CB3 weights and expert deferral are distinct numerical/model choices, not IQ2_XXS optimization patches.

Useful C0/C5 accounting: four streams need24 verify rows; source's31-row budget is an AMD shape boundary, not ours. Packed compressed512-wide rows1024→288 bytes and128-wide index rows256→72 are3.56× storage reductions **only when the uncompressed values already lie on that exact FP4+scale grid**. Ours already packs KV, so audit the actual retained representation before crediting capacity. More compressed history is not automatically a larger524K hot ring; four256K requests require~1,048,576 hot positions, twice current capacity, and current KV-pool2.08M-token capacity alone does not prove fit.

DSX's decoder-tail refinement mostly duplicates adopted PREFILL_SKIP_ROWS3072. The remaining per-layer taper would reduce19*3072=58,368 tail token-layers to128*(19+18+...+1)=24,320, a58.3% reduction in **that tail only**. Our PP0 prefill remains limiting; no whole-TTFT gain justified. Keep lower priority than C1/C4/S2.

## Portability, expected effects and exclusions

Volta constraint: FP16 tensor-core MMA.m8n8k4,80SMs,32GB per GPU, about879GB/s ceiling used by the brief. Software decoding of BF16/FP8/FP4 storage can be portable; native BF16/FP8/INT8 tensor-core instruction paths and cp.async are not. Bandwidth limits are only ceilings: prior profiling found expert kernels issue/latency-bound. Reading half as many bytes does not guarantee twice the throughput.

At22.05ms/round, a verified0.5ms saving yields22.05/21.55−1=2.32%;1ms yields4.75%. These arithmetic effects require unchanged acceptance and token work. Reducing drafts can shorten rounds while reducing tok/s: our max-three verification test lost14.5% throughput. Preserve seeded sampling/coupled-draw identity, row-level reduction order, request-local speculative rollback and graph buffer lifetime. A new source's exactness claim does not prove batch invariance on our imported kernel.

Do not reopen unchanged rejected work: lossy expert top-p/deferral,16K/4K numerical chunk changes, TP8 cross-board all-reduce, generic V1 async, non-exact shared-down/split-K attention, deeper prefetch and idle Engram-HBM cache proposals. A distinct implementation that removes the documented failure is a new experiment; another source reporting a faster different format is not that evidence.

## Three most actionable next steps

1. **Build C1 with C2 as one acceptance unit.** Fix the24-row contract across router, plan, dense projections, experts, drafter, sampler, attention/indexer, all-reduce and MRV2 graphs. Test real routing and weights at6/12/18/24 rows against each request alone. Measure aggregate output and per-request latency with unchanged emitted-token work. Use #1369/Maya/DSX as designs, not drop-in kernels.
2. **Implement C4 cadence at existing numerical chunk boundaries.** Separate each request's decode/prefill state and give ready decodes turns between8,188-row pieces. Report maximum decode gap, ongoing decoder tok/s and new-request TTFT together. If a whole chunk still causes unacceptable stalls, use the already studied layer-slice approach rather than silently resizing it.
3. **Finish S1's exact candidate and investigate S2 residual from existing traces.** F1 is already built; establish its exact mode3 win. Then determine whether remaining0.6–0.7s/chunk is transfer, synchronization or another owner before coding overlap. In parallel planning, audit issue#1342/DSX scratch lifetimes for actual conc4 memory headroom.

## Coverage and verification

Public GitHub reads and local source inspection only. Required snapshots, census and source inventories are preserved for resumption. Prior research read: CONCBATCHOP1104/RESEARCH.md, research-notes.md, philbert-prefill-inventory.md; LANES.md upstream history; public build README, docs/levers.md and patch series; KCAMP1106/PLAN.md. SUPPLEMENT.md separately records standing 0rand/TensorFold/jayleaton/exllamav3/HeNryous/vLLM metadata watch and selected primary PRs. It is not represented as an exhaustive implementation review of those additional repositories.

No inference host was contacted. No GPU-GRANT, pp2-go.state, server port, production checkout or GPU was touched; no GitHub write, email or publication occurred. Performance estimates remain estimates until an authorized owner executes the campaign.
