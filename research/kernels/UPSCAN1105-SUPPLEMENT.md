# Supplementary standing upstream watch

> **Origin.** PLI Labs research archive, file `kernel-research/UPSCAN1105-SUPPLEMENT.md`. Published 2026-11-18 in the Volta V100 research
> repository.
> **Edited for publication.** Machine names, private file paths, network addresses, session names and the
> operator's name were replaced with generic terms (`the V100 server`, `[private path]`, `[address]`,
> `the owner`). Links to files that are not published were turned into plain text. No figure and no finding
> was changed. Dates inside the note are its own; later results are in the build pages of this repository.

This is a metadata watch alongside the three sources required by BRIEF.md, not a replacement for their full inventories. Public GitHub reads only, 2026-11-05. Raw evidence is in `evidence/`.

- 0rand: public repository list captured; all 15 branches of `deepseek-v41-tensorfold-spark` and all 15 branches of `DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark` examined for commits since 2026-10-28. Only the former's `main-0rand` has new commits in this date window (2), head `a903cb5fe3b4c91b2b17e2a1aaf5dc80a045ced0`; this is the head already reviewed in the owner's SPARKS-RECIPE/REPORT-20261102.md. Recipe default is `pr65-flash17-serving` at `305659970ae63729b0753b27dea85a7ac4b5a93b`. No unreviewed branch update after the last reviewed head was found.
- `jayleaton/deepseek-v41-tensorfold-spark`: 10 main commits since date; source of earlier reviewed batch/row-invariance design, not new Volta kernels. Saved metadata.
- `ashhart/TensorFold`: 1,450 main commits returned across all pages since date, latest `cde4fe7d9959`; saved complete metadata. Title inventory only here, not a full 1,450-change implementation review.
- `turboderp-org/exllamav3`: 20 master commits since date, saved metadata.
- `HeNryous/deepseek-v41-1017-dgx-spark`: zero default-branch commits since date. Initial guessed repo `HeNryous/DeepSeek-V4.1-Flash` returned 404; resolved actual name from the owner's upstream-watch script and checked successfully. The 404 is not an access blocker.
- `vllm-project/vllm`: 726 main commits since date, saved complete metadata and inspected selected primary PR descriptions below. Hardware-specific numbers remain author claims on their stated equipment.

| Primary change | Potential use / KCAMP mapping | Scope and exclusion |
|---|---|---|
| [#61464 MRV2 DBO with speculative decoding](https://github.com/vllm-project/vllm/pull/61464) | C3: request-boundary microbatch split, full-batch metadata rebuild after target forward so speculator FULL graphs see valid buffers | Engine design reference, not proof of PP2/DSpark/SM70 support or measured benefit here. DBO communication overlap is not automatically PP-stage interleaving. |
| [#64458 MRV2 PP broadcast fixes](https://github.com/vllm-project/vllm/pull/64458) | C0/C2/C3/S6: avoid non-last-stage host sync; gather on main stream before broadcast stream reads; skip absent drafts; size broadcast per actual step | Correctness/ownership audit before greater concurrency; compare with our already ported PP hooks. No extra speed credit for duplicate code. |
| [#60618 fused prefill score + top-k](https://github.com/vllm-project/vllm/pull/60618) | New/S10-related prefill selection design: remove logits materialization, one persistent launch for <=16,384 query rows / <=1M keys; 296 MiB workspace stated vs 2.6 GiB LiteTopK | Their gated 24-bit ranking permits boundary ties and has a rescanning path; our exactness must be preserved. Modern FP8 scoring implementation needs a Volta FP16/FP32 replacement. No portable whole-model multiplier. |
| [#59131 gather compressed KV once](https://github.com/vllm-project/vllm/pull/59131) | S1/S2/S4 context/memory traffic: 38 compressed-layer gathers become 4 source-group gathers per chunk on the described path | Mega-attention B200 path; our packed ring and fused sparse attention do not necessarily gather whole history. Audit actual calls first; no forecast without traffic count. |
| [#63956 token-sharded sparse MLA prefill](https://github.com/vllm-project/vllm/pull/63956) | New/S1/S4: sequence-parallel prefill, preserve head-sharded decode in mixed steps | Author claims GLM-5.3 NVFP4 TP4 GB200 TTFT 1.24x/1.29x/1.40x at 32K/64K/128K. Those are neither V4.1 nor Volta data. Would change communication and GEMM shapes; no exactness assumption. |
| [#64442 DSpark inverse-CDF Markov fusion](https://github.com/vllm-project/vllm/pull/64442) | C1/S6 implementation pattern for fused sampler; author GB200 one-request Markov 111.9→71.6 µs, not whole round | Our coupled keyed-Gumbel sampler is already fused/sharded. Switching random transforms would lose seeded byte identity/coupling unless redesigned, so do not adopt as a drop-in speed win. |

All six selected PR descriptions and merge SHAs are captured as `evidence/vllm-pr-*.json`. None of these supplementary reads authorize deployment. The ranked report is grounded in the requested 1Cat/Maya/DSX inventories and our measured baseline.
