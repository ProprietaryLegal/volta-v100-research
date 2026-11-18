# Building a custom kernel for sm_70

A guide to writing a CUDA kernel for the Tesla V100 (Volta, compute capability 7.0) that survives contact with a
real model server. It is the long form of
[part 7 of the front page](../index.md#part-7-how-to-build-a-custom-kernel-for-sm_70).

The rules come from the kernel notes kept during these builds. The full texts, with their sources, are in this
repository:

- [research/kernels/KERNEL-RESEARCH-CLAUDE-NOTES.md](../research/kernels/KERNEL-RESEARCH-CLAUDE-NOTES.md): the
  twelve binding rules, each with its source;
- [research/kernels/KERNEL-RESEARCH-EXPLAINER.md](../research/kernels/KERNEL-RESEARCH-EXPLAINER.md): the same
  ground in plain terms, with a glossary;
- [research/kernels/KERNEL-RESEARCH-REPORT.md](../research/kernels/KERNEL-RESEARCH-REPORT.md): the survey, with
  its reference list.

A kernel here means a program launched onto the card. A decode round of DeepSeek-V4.1-Flash on the eight-card build
ran about 2,089 of them (MEASURED by profile, 2026-11-03, the kernel notes).

## Step 1. Size it before you build it

Two numbers decide whether an idea is worth a day.

**Bytes over bandwidth.** A V100-SXM2 has 900 GB/s of HBM2 on its data sheet. Measured ceilings are lower:
879 GB/s read and 822 GB/s copy, and about 720 GB/s for the best dense matrix-times-vector kernel (all UPSTREAM,
[v100-skinny](https://github.com/dnv2003/v100-skinny)). A kernel that must read B bytes cannot finish in less than
B / 720 GB/s. Compute that first.

**FLOPs over tensor throughput.** The FP16 tensor cores give about 125 TFLOPS at the full 1,530 MHz clock
(published). Divide: the card needs about 139 floating-point operations per byte delivered to keep its tensor
cores busy. A matrix-times-M-rows multiply does about `2 x M / (bytes per weight)` operations per byte:

| Weight format | Bytes per weight | Operations per byte at 1 row | At 6 rows (1 token + 5 drafts) |
|---|---|---|---|
| FP16 | 2 | 1 | 6 |
| 4-bit | 0.5 | 4 | 24 |
| about 2-bit | about 0.26 | about 8 | about 47 |

Every decode shape is far below 139. **Decode on this card is never limited by arithmetic.** It is limited by
bytes, and below that by fixed costs. Prompt reading, with thousands of rows, is the opposite: it is
arithmetic-bound, and tensor-core kernels pay there.

**The 0.3 ms floor.** Convert the saving to milliseconds per decode round. On these builds a round was 22 to 31 ms
and the best measurement resolved about 0.05 ms per round when texts paired, about 0.12 ms otherwise. A saving
under about 0.3 ms per round (about 1 %) will not be measurable in the server. Do not spend a test window on it
unless it unblocks something larger.

Three caveats, each measured the hard way:

1. **Achievable bandwidth is below nameplate.** Measure a copy kernel on the actual card at the actual clock cap.
2. **Decoding compressed weights costs instructions per weight, not per bit.** Codebook and lattice formats spend
   several instructions per weight to unpack. One analysis in these notes counted 7.6 to 8.8 issue slots per
   weight for a 2-to-5-bit format, of which 6.0 were irreducible: going from 4 bits to 3 saved no time. The expert
   kernels of Volta sPLIt read at 290 to 415 GB/s, well under the ceiling, because they were limited by
   instruction issue. That is why an "instruction diet" won 0.87 ms per round and nineteen memory-oriented variants
   won nothing.
3. **Small kernels are latency-bound.** A kernel that reads a few hundred kilobytes never reaches the bandwidth
   ceiling; launch, ramp-up and tail dominate.

## Step 2. Attack in this order

1. **Remove work that is sized by capacity instead of by live data.** Look for anything sized by the maximum
   context, the full store, the padded vocabulary or the maximum number of rows. The biggest single kernel win of
   the first build was adding only the live part of an index table across the cards: about 0.85 ms per round.
   The second example: scoring only the live blocks of keys cut a decode round by 13.7 % at 300K tokens.
2. **Cut launches and the gaps between them.** Fuse a small step into its neighbour. Value a fusion by the kernel
   time and gaps it removes (about 0.3 to 1 microsecond per gap inside a captured graph), not by its isolated
   benchmark. Never add chains of small library operations per layer: one attempt that added about 50 small
   operations per layer cost 6.9 ms per round.
3. **Then cut bytes.** Only now is a new weight kernel worth writing.

On the twelve-card build the same order held one level up: 3.78 ms of a 26.25 ms round was outside any kernel, and
the largest prompt-reading gain was not releasing memory that the next chunk asked for again.

## Step 3. What sm_70 has and does not have

| You have | You do not have |
|---|---|
| `mma.sync.aligned.m8n8k4`: FP16 inputs, FP16 or FP32 accumulate | `cp.async` (asynchronous copies into shared memory): Ampere and later |
| The WMMA 16x16x16 API on top of it | `ldmatrix` (loading tensor-core fragments from shared memory): Turing and later |
| `dp4a` (8-bit dot products on ordinary cores), `lop3`, warp shuffles | BF16, TF32, INT8, FP8 or FP4 tensor-core modes |
| 80 SMs; up to 96 KB of shared memory per SM (above 48 KB needs an opt-in); 64K registers per SM, at most 255 per thread | Tile copies (TMA), warp-group multiplies (`wgmma`), thread-block clusters |
| A 6 MB L2 cache: weights stream from HBM on every step | A cache big enough to hold a layer |

Sources: NVIDIA's [Volta tuning guide](https://docs.nvidia.com/cuda/volta-tuning-guide/index.html) and the
[Ampere](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html) and
[Hopper](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) guides for what came later.

Consequences:

- **Port the idea, not the code.** A kernel written for Ampere or Hopper assumes instructions this chip lacks. It
  needs a rewrite: pipelines staged through registers, fragments loaded by hand.
- **Every low-bit format is decoded in software into FP16 operands.** A 4-bit weight format does not make the
  multiply faster here. It makes the read smaller. v100-skinny's NVFP4 kernel reads at 77 % of the measured
  ceiling at one row (UPSTREAM): that is the whole gain, and it is a real one.
- **Budget the kernel.** Check the compiler's report: zero register spills; registers low enough for the blocks per
  SM you planned; hot loops under about 12 KiB of instructions; no empty blocks; grids sized for 80 SMs; read-only
  tables in global memory, not copied into each block's shared memory.
- **Pin the toolchain.** CUDA 13.0 removed offline compilation for Volta
  ([CUDA features archive](https://docs.nvidia.com/cuda/cuda-features-archive/index.html)). PyTorch keeps sm_70
  only in its CUDA 12.6 builds
  ([announcement](https://dev-discuss.pytorch.org/t/dropping-volta-support-from-cuda-12-8-binaries-for-release-2-11/3290)).
  Triton officially supports compute capability 8.0 and later, so check every Triton kernel on the card and prefer
  CUDA C++ for hot paths.

## Step 4. State the numerics contract first

Before writing the kernel, write one sentence: *bitwise* or *tolerance*.

- **Bitwise.** If the kernel keeps the order of every sum (the same split, the same warps), its output must equal
  the old kernel's bit for bit, and the test is equality over every shape.
- **Tolerance.** If it reorders sums (a fused epilogue, a different split, an autotuned configuration), say so, give
  the bound, and expect to pass the server's near-tie gate: greedy outputs identical, or changed only where the top
  two tokens were nearly tied.

An autotuned or "suggested" configuration is a numerics change. A split count that depends on the batch size is a
numerics change that depends on the batch size: on the twelve-card build that is exactly why a prompt's last chunk
rounds differently from the others.

Know in advance that the contract can cost you the kernel. The fastest prompt-reading kernel of the first build
(-11.4 % to first token) was bit-equal on 99.4 % of elements and was not installed.

## Step 5. The switch

Every kernel goes in behind an environment switch that defaults off. With the switch off, the old code path must be
provably unchanged: a CPU test plus a comparison of the syntax tree, not only an equality test of outputs. The
replaced library is kept for rollback.

This is what makes the rest possible. A/B arms differ by one variable. A lost lever is removed by unsetting it.
Twenty pull requests could be offered upstream with no risk to anyone who did not opt in.

## Step 6. Test like an adversary

Kernels written with AI help reward weak tests. The literature the notes cite is blunt about it (a claimed
10 to 100 times speedup [walked back](https://techcrunch.com/2025/02/21/sakana-walks-back-claims-that-its-ai-can-dramatically-speed-up-model-training/)
after the system was found skipping the correctness check). So:

- Cover the shapes the server will send: 1 to k+1 rows for a drafter that proposes k tokens; every pipeline stage;
  every layer role; ragged sizes; sizes 0, 1 and maximum.
- Write outputs into fresh buffers pre-filled with NaN. Never reuse a reference's output buffer.
- Run `compute-sanitizer` memcheck, racecheck and synccheck, on one card and on each tensor-parallel worker.
- Mutation-test the test: break the kernel on purpose and confirm the test fails.
- Use real routing histograms and real shapes for benchmarks, not uniform synthetic ones.

## Step 7. Only the end-to-end A/B decides

Isolated benchmarks misled in both directions on these builds:

| Kernel | Isolated benchmark said | Server A/B measured |
|---|---|---|
| Double-buffered dense mat-vec | 0.03 to 0.07 ms per round | -0.42 ms |
| Expert plan fused into the quantize launch | slightly slower | -0.36 ms |
| A candidate decode kernel for the NVFP4 experts | "1.45 ms of headroom" | 0.000 ms predicted on one card under a rule fixed in advance; no server window spent |

So: mirrored arms (OFF, ON, ON, OFF), milliseconds per round, an A-against-A noise measurement, a quiet machine,
untimed warm-up after each boot. The protocol is in [testing.md](testing.md). Never drop a lever, and never adopt
one, on a microbenchmark alone.

When a microbenchmark is all you can afford, write the rule for it first too. The last row above is that case: the
rule said what one-card result would justify an eight-card window, the result did not, and the work was closed
honestly instead of being "probably a small win".

## What does not port from Ampere or Hopper

| Technique on newer chips | Why it does not port | What was done here instead |
|---|---|---|
| FlashAttention-2 and later, FlashInfer attention | written for Turing or Ampere and later | the fork's hand-written Volta attention kernels; our fused prologue and epilogue |
| Marlin (fast 4-bit GEMM) | written for Ampere and later; no Volta path | v100-skinny's QPN kernels: quad pairs on N, weights pre-packed into fragment order, no shared memory in the main loop |
| FP8 or FP4 tensor-core GEMM (DeepGEMM and others) | no FP8 or FP4 tensor modes | decode to FP16 in registers, multiply with `mma.sync.m8n8k4` |
| Warp-specialised producer and consumer pipelines | no asynchronous copy: the loading warps cannot also compute | register-staged double buffering |
| Megakernels over a whole layer | not attempted here | fusing two or three neighbours at a time, each proven bitwise |

## Where to read more

- The measured time budget of a decode round on eight cards:
  [Volta Max uPLIft, docs/research/LANE-RESEARCH.md](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4/blob/main/docs/research/LANE-RESEARCH.md)
  and its copy in [research/eight-card/](../research/README.md).
- The same on twelve cards: [optimizations/volta-12x-amplify.md](../optimizations/volta-12x-amplify.md#where-the-time-goes).
- The published kernels: the production patches of
  [Volta sPLIt](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/tree/main/engine/production)
  and the prompt-reading kernel source of
  [Volta Max uPLIft](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4/tree/main/engine/kernels/prefill-rb4).
- The kernels these builds stand on: [v100-skinny](https://github.com/dnv2003/v100-skinny) and the SM70 kernels of
  [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM).
