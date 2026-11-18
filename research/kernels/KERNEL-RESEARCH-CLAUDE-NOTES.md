# Kernel-building rules for the DSV41 repository

> **Origin.** PLI Labs research archive, file `kernel-research/KERNEL-RESEARCH-CLAUDE-NOTES.md`. Published 2026-11-18 in the Volta V100 research
> repository.
> **Edited for publication.** Machine names, private file paths, network addresses, session names and the
> operator's name were replaced with generic terms (`the V100 server`, `[private path]`, `[address]`,
> `the owner`). Links to files that are not published were turned into plain text. No figure and no finding
> was changed. Dates inside the note are its own; later results are in the build pages of this repository.

Terse and binding for kernel lanes on the V100 server (8x V100-SXM2, sm_70, PP2 x TP4, 1Cat vLLM fork, v41k / sm70_turbomind ops); rule 12 covers NVIDIA's DGX Spark (GB10, sm_121). Each rule names its source. L-sources are local: `PROF1103/REPORT.md` [L1], `TOK1103/REPORT.md` [L2], `LANES.md` by UTC stamp [L3], code in `V41PKG/wt/combo1103` [L4]. The full report is `KERNEL-RESEARCH/REPORT.md`.

1. **Size it before you build it.** Compute the speed-of-light time: bytes ÷ ~720 GB/s achievable per V100 (900 GB/s nameplate) for decode, or FLOPs ÷ 125 TFLOPS FP16 for compute-bound work. Convert to ms per round against the 31.1 ms baseline. A saving under ~0.3 ms per round (~1 %) won't be measurable; don't spend a GPU window on it unless it unblocks something larger.
   *Source:* roofline (Williams et al., https://doi.org/10.1145/1498765.1498785); 718.6 GB/s best observed GEMV and the round breakdown [L1].

2. **Remove work before speeding it up.** Look for kernels sized by *capacity* (max context, full store width, padded vocab, max verify rows) instead of live data. The live-prefix index reduce saved ~0.85 ms per round (first reported as 1.3 against an outlier baseline; REVIEW1103 F6), the campaign's biggest single kernel win.
   *Source:* [L2]; IDX-A/B 98.0 → 102.9 tok/s [L3 2026-11-03T02:24Z].

3. **Every launch must pay for itself.** The decode path already runs 2,089 kernels per round with ~2 ms of bubbles per stage. Never add chains of small torch ops per layer: lever 2 as ~50 ops per layer cost +6.9 ms. Fuse into a neighbour, and value a fusion by the kernel time and gaps it removes (~0.3-1 µs per graphed gap), not by its microbenchmark.
   *Source:* [L1]; [L2] item 3; lever 2 78.2 vs 99.1 tok/s [L3 01:29Z]; CUDA graphs on V100 (https://developer.nvidia.com/blog/cuda-graphs/).

4. **Default-off knob, untouched old path.** New kernels go behind a `VLLM_V41_*` knob that defaults to 0. Prove the knob-off path is unchanged (AST/`knob_off_check`, not only an equality test) before landing.
   *Source:* V41PKG landing conditions [L3 2026-11-02 00:32Z]; every lever this campaign followed it [L3].

5. **Declare the numerics contract up front.** If you keep the reduction order (same warps/split; split N, not K), the result must be **bitwise** identical. Otherwise state a tolerance and pass the end-to-end gate: greedy identical, or flips only at traced near-ties. Autotuned or "suggested" configs are numerics changes too.
   *Source:* `gguf_dmv13_x.cuh` header [L4]; wo_a N-split [L3 2026-11-01 15:54Z]; DMVX-CFG 2/8 identical, dropped [L3 03:28Z]; lever 1 trace proof [L3 03:08Z]; batch invariance (https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/).

6. **Test like an adversary.** Cover held-out shapes: M = 1 … k+1 verify rows, both PP stages, every layer role, ragged K, size 0/1/max. Write into fresh, NaN-poisoned output buffers and never reuse reference outputs. Run `compute-sanitizer` (memcheck, racecheck, synccheck). Mutation tests must be killed. This applies doubly to AI-written kernels.
   *Source:* Sakana's walk-back (https://techcrunch.com/2025/02/21/sakana-walks-back-claims-that-its-ai-can-dramatically-speed-up-model-training/); KernelBench-Verified (https://arxiv.org/abs/2607.16241); Kernel Contracts (https://arxiv.org/abs/2604.22032); KFUS1 mutants [L3 2026-11-02T00:28Z].

7. **Benchmark protocol.**
   - Microbenchmarks run with a cold L2, CUDA-graph replay, warm-up and clocks locked at 1522 MHz, and report % of speed of light.
   - Nsight Compute's defaults (base clocks, cache flush) are not production timings.
   - **Only the end-to-end ABBA decides.** Use mirror order, an A/A noise pair and a quiet host CPU (no nvcc builds during arms), and discard or flag the first arm after a deploy.

   *Source:* Triton `do_bench` (https://github.com/triton-lang/triton/blob/main/python/triton/testing.py); NCU defaults (https://forums.developer.nvidia.com/t/is-it-acceptable-to-measure-kernel-performance-using-nsight-compute/338212); DMVX bench 0.03 vs end-to-end 0.45 ms [L3 03:24Z]; ASYNC-B1 confound [L3 01:58Z]; first-boot divergence [L3 03:00Z].

8. **The sm_70 ISA is fixed.** You have `mma.sync.aligned.m8n8k4` FP16 with **f32 accumulate** (HMMA.884), the WMMA 16x16x16 API, `dp4a`, `lop3` and warp shuffles. You do **not** have `cp.async`, `ldmatrix`, TMA, `wgmma`, clusters, BF16, FP8/FP4 MMA or INT8 MMA. Pipelines are register-staged. Never port code that needs the missing pieces; port the idea.
   *Source:* Volta tuning guide (https://docs.nvidia.com/cuda/volta-tuning-guide/index.html); Ampere/Hopper tuning guides for what came later; `x_mma` in [L4].

9. **The sm_70 resource budget.**
   - ≤ 96 KB shared per SM (dynamic shared above 48 KB needs an opt-in); 64K registers per SM; ≤ 255 registers per thread.
   - Check `-Xptxas -v`: 0 spills, and registers low enough for the blocks per SM you planned.
   - Keep hot loops under ~12 KiB of instructions.
   - Launch no empty blocks, and size grids for 80 SMs.
   - Put read-only tables in L1-cached global memory, not per-CTA shared copies.
   - Set the L1/shared carveout on purpose (`dmv13_x` requests max-shared so blocks per SM aren't capped).

   *Source:* Volta tuning guide (above); KMOE1 i-cache/regs/empty-block findings [L3 2026-11-01T18:49Z]; v41k codebook table and `dmv13_x` carveout notes [L4].

10. **Pin the toolchain.** Stay on CUDA 12.x: CUDA 13.0 removed Volta offline compilation. PyTorch keeps sm_70 only in cu126 builds, through 2.15. Triton officially supports CC 8.0+, so compile-check every Triton kernel for sm_70, verify it on the GPU, and prefer CUDA C++ for hot paths.
    *Source:* https://docs.nvidia.com/cuda/cuda-features-archive/index.html ; https://dev-discuss.pytorch.org/t/dropping-volta-support-from-cuda-12-8-binaries-for-release-2-11/3290 ; https://github.com/triton-lang/triton

11. **Communication kernels have hard rules.** Our custom all-reduce path needs size % 16 B == 0 (pad rows) and a fully connected TP group. Use one-shot for small messages and reduce only live data. Cross-GPU flags use `st.release.sys` / `ld.acquire.sys` with per-CTA epochs, and the reduce order is fixed by rank (deterministic, but not NCCL-bitwise).
    *Source:* TOK1103 `index_reduce_live` [L2]; vLLM one-shot vs two-shot (https://github.com/vllm-project/vllm/files/13574639/Latency-optimal.allreduce.and.cuda.graph.optimization.pdf).

12. **DGX Spark (GB10) is sm_121a, not sm_100a.**
    - Build with `-gencode=arch=compute_121a,code=sm_121a`; the shorthand reportedly drops the "a".
    - There is no `tcgen05`/TMEM/2-SM MMA: FP4/FP8 run through warp-level block-scaled `mma.sync`.
    - Limits are 99 KB shared per block and 48 warps per SM.
    - At 273 GB/s, bytes per token decide decode speed. FA4, DeepGEMM-SM100 and FlashMLA-SM100 won't run as-is.

    *Source:* https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html ; https://docs.nvidia.com/dgx/dgx-spark/hardware.html ; community reports (https://forums.developer.nvidia.com/t/dearest-cutlass-team-when-the-hell-are-you-going-to-properly-fix-tcgen05-fp4-support-for-dgx-spark-gb10-sm121/359598); not confirmed in NVIDIA ISA docs.
