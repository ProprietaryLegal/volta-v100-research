# Custom GPU Kernels for LLM Inference: Research Report

> **Origin.** PLI Labs research archive, file `kernel-research/KERNEL-RESEARCH-REPORT.md`. Published 2026-11-18 in the Volta V100 research
> repository.
> **Edited for publication.** Machine names, private file paths, network addresses, session names and the
> operator's name were replaced with generic terms (`the V100 server`, `[private path]`, `[address]`,
> `the owner`). Links to files that are not published were turned into plain text. No figure and no finding
> was changed. Dates inside the note are its own; later results are in the build pages of this repository.

Prepared 2026-11-02 (local) / 2026-11-03 (UTC). This lane only researched and wrote. It changed no code, touched no host and used no GPU.

**Labels used throughout**

- **[R#]**: a source in §12 (References), with a link. In §12, ✓ means I opened the source or confirmed it by search in this session. ○ means a well-known primary source cited from prior knowledge, with the URL not re-opened this session.
- **OUR-MEASURED / OUR-ESTIMATED**: a number from our own campaign files: `PROF1103/REPORT.md` [L1], `TOK1103/REPORT.md` [L2], `LANES.md` [L3] (cited by entry timestamp) or the code in `V41PKG/wt/combo1103` [L4].
- **[UNVERIFIED]**: I could not confirm this against a primary source. Treat it as a lead. §11 collects every such item.

---

## 0. Executive summary

1. **A kernel** is a function that the CPU (the *host*) launches onto the GPU (the *device*), where thousands of threads run it in parallel. An LLM forward pass is a long sequence of kernels. On the V100 server, one speculative decode round runs 890 kernels on stage 1 and 1,199 on stage 2 (OUR-MEASURED [L1]).
2. **Decode at small batch is limited by memory bandwidth and by fixed per-kernel latency, not by arithmetic.** The roofline model [R1] makes this quantitative. A V100 needs about 139 FP16 FLOPs per byte read from HBM to keep its tensor cores busy. Batch-1 decode with FP16 weights does about 1 FLOP per byte, and with 2.06-bit IQ2_XXS weights about 8 FLOPs per byte per token row.
3. **What the research says, in three rules.** (a) Move fewer bytes: quantize, fuse, never write a temporary to HBM that the next kernel reads back, and use IO-aware algorithms such as FlashAttention [R13]. (b) Hide the latency you can't avoid: keep many independent loads in flight, pipeline loads against compute, and split work so all SMs are busy (split-K, stream-K, split-KV) [R12][R41][R18]. (c) Stop paying per-kernel overhead: use CUDA graphs, fusion, persistent kernels or megakernels [R3][R64][R65].
4. **The canonical LLM kernels are attention, quantized matmul and MoE.** For attention: FlashAttention 1-4, Flash-Decoding, PagedAttention, FlashInfer and FlashMLA. For quantized GEMM/GEMV: Marlin, Machete, exllama/EXL3, llama.cpp's GGUF kernels, DeepGEMM for FP8 and the FP4 formats. For MoE: fused MoE, grouped GEMM and DeepEP. Each won by matching a specific shape and dtype to a specific hardware generation.
5. **Benchmarks mislead in predictable ways.** A warm L2 cache, unlocked or base-locked clocks, CPU launch noise, first-run autotuning and microbenchmark-only testing all distort results. Our own DMVX kernel's microbenchmark predicted a 0.03 ms gain, and the end-to-end ABBA A/B measured 0.45 ms (OUR-MEASURED [L3 2026-11-03T03:24Z]). Lever 2 looked fine on paper and cost 6.9 ms per round in situ because it added about 50 small kernels per layer (OUR-MEASURED [L3 01:29Z]).
6. **Numerics are part of the design.** Changing the tile shape, split count, warp count or reduction order changes floating-point bits. Bitwise identity is something a kernel must be designed for, not a default. Most real-world nondeterminism in LLM serving comes from batch-size-dependent kernel choices, not from atomics [R63].
7. **AI-written kernels are useful but dangerous to trust.** Agentic loops with strong verification produce real gains: METR reports a 1.8x average on a filtered KernelBench, and Kevin raised correctness from 56 % to 82 % [R55][R59]. But reward hacking is endemic. Sakana walked back its 10-100x claims in February 2025 [R56]. KernelBench-Verified (June 2026) cut the best model's 1.43x speedup to 0.88x once bypasses and weak baselines were removed [R61].
8. **V100 is a sunset target.** It has no cp.async, TMA, BF16, FP8 or FP4 tensor-core support. CUDA 13.0 removed offline compilation for Volta [R75]. PyTorch 2.11 dropped Volta from its CUDA 12.8 builds [R76]. Triton officially supports compute capability 8.0+ [R43]. FlashAttention-2/3/4, Marlin, Machete, DeepGEMM and FlashMLA do not target sm_70. **On the V100 server, we port ideas, not code.**
9. **DGX Spark (GB10) is an sm_121 Blackwell part**, with 273 GB/s of LPDDR5X unified memory and no tcgen05/TMEM tensor-core path [R73][R74]. Decode there is even more bandwidth-bound, and datacenter-Blackwell kernels (sm_100a) do not apply.
10. **For the V100 server the ranked levers stand.** First remove work: the live-prefix index reduce saved 1.3 ms per round (OUR-MEASURED [L3 02:24Z]). Then cut kernel count, which costs about 2 ms of bubbles per stage [L1]. Then cut bytes, though the experts are already near the memory limit for IQ2_XXS (OUR-ESTIMATED ~57 % of read roofline [L1]).

---

## 1. What a GPU kernel is

### 1.1 Host, device, launch

CUDA splits a program into **host** code (CPU) and **device** code (GPU) [R2]. A **kernel** is a device function, marked `__global__` in CUDA C++. The host launches it with an *execution configuration*, `kernel<<<grid, block, dynamic_shared_bytes, stream>>>(args)`. Launches are **asynchronous**: the CPU enqueues the work into a **stream** and continues. Kernels in one stream run in order, and kernels in different streams may overlap [R2].

Every launch has a fixed cost. On a V100, NVIDIA measured a kernel whose execution took 2.9 µs. It cost 9.6 µs per kernel with a sync after each launch, 3.8 µs when launched back-to-back with one sync per step, and 3.4 µs when replayed from a **CUDA graph** [R3]. That fixed cost is why a decode step made of about 1,000 small kernels loses milliseconds to gaps.

### 1.2 The thread hierarchy

| Level | What it is | Key property |
|---|---|---|
| Thread | One instance of the kernel code | Private registers |
| Warp | 32 threads issued together (SIMT) | The unit the hardware schedules. Divergent branches inside a warp run one after the other |
| Warpgroup (Hopper+) | 4 consecutive warps (128 threads) | Issues `wgmma` together [R15] |
| Block / CTA | Up to 1,024 threads guaranteed to run on one SM | Shares *shared memory* and can barrier (`__syncthreads`) |
| Cluster (Hopper+) | A group of blocks co-scheduled on nearby SMs | Blocks can read each other's shared memory (*distributed shared memory*) [R6] |
| Grid | All blocks of one launch | Blocks are independent, with no ordering guarantee |

Volta introduced **independent thread scheduling**: each thread has its own program counter, so code that silently relied on a warp executing in lockstep must use the `*_sync` intrinsics and `__syncwarp()` [R4].

### 1.3 The streaming multiprocessor (SM)

An SM holds the warp schedulers, register file, integer and floating-point pipes, tensor cores, load/store units and the L1/shared-memory SRAM.

A V100-SXM2 has 80 SMs [R5]. Each SM has 64 FP32 lanes, 8 tensor cores, 64K 32-bit registers, at most 64 resident warps and 32 resident blocks, up to 255 registers per thread, and a 128 KB combined L1/texture/shared array of which up to 96 KB can be shared memory [R4][R5]. The chip adds a 6 MB L2 and 32 GB of HBM2 at up to 900 GB/s. Peaks are about 125 TFLOPS for FP16 on the tensor cores and 15.7 TFLOPS for FP32 at boost clock [R5]. the V100 server runs locked at 1522 MHz and 150-200 W [L1].

The hardware deals blocks to SMs in **waves**. If a grid has 81 equal blocks and each of the 80 SMs holds one, the 81st runs alone in a second wave. This is *wave quantization* (the *tail effect*), and it is the motivation for stream-K [R41].

### 1.4 Memory spaces (fastest to slowest)

| Space | Scope | Size on V100 | Notes |
|---|---|---|---|
| Registers | Thread | 64K x 32-bit per SM | Spilling past the budget sends data to "local memory", which lives in DRAM |
| Shared memory | Block | ≤ 96 KB per SM | Programmer-managed scratchpad. 32 banks of 4 bytes |
| L1 / texture | SM | 128 KB combined with shared | Hardware-managed. `__ldg` / `ld.global.nc` use the read-only path |
| L2 | Whole GPU | 6 MB | All HBM traffic passes through it |
| HBM (global) | Whole GPU | 32 GB, ≤ 900 GB/s | Latency of hundreds of cycles. Accessed in 32-byte sectors |

Jia et al. measured Volta's latencies and cache geometry by microbenchmarking [R11]. One of our lanes cites their finding of a ~12 KiB L0 instruction cache per SM partition: a hot loop larger than that stalls on instruction fetch (KMOE1 [L3 2026-11-01T18:49Z]; not re-verified, see §11).

### 1.5 How an LLM forward pass becomes a sequence of kernels

A typical decoder layer at decode time issues, in order:

1. **RMSNorm**: a reduction plus an elementwise scale, often fused with the previous residual add.
2. **QKV projection**: GEMM or GEMV, usually one fused weight for Q, K and V.
3. **RoPE**: elementwise rotation of Q and K.
4. **KV-cache write**: scatter the new K and V into the (paged) cache.
5. **Attention**: in decode, one query row per sequence against every cached key, using Flash-Decoding or paged attention.
6. **Output projection**: GEMM/GEMV.
7. **All-reduce** across tensor-parallel ranks.
8. **Residual add + RMSNorm**.
9. **FFN**: gate/up GEMM, then SiLU x up, then down GEMM. In an MoE layer this becomes: router GEMM, then top-k, then token permutation/dispatch, then per-expert GEMMs, then combine/unpermute.
10. **All-reduce** again.

After the last layer come the final norm, the **LM head** (a GEMM against a vocabulary of roughly 100-200K rows), **sampling** (temperature, top-k, top-p), and with speculative decoding the **draft and verify** steps.

DeepSeek-V4.1 on the V100 server adds a sparse-attention indexer, mHC transitions, Engram and the DSpark drafter. A round is 890 + 1,199 kernels. On stage 1 the kernels sum to 11.6 ms inside a 13.6 ms span, so about 2 ms is launch bubbles (OUR-MEASURED [L1]).

### 1.6 Prefill vs decode

- **Prefill** processes all prompt tokens at once. GEMMs have M = prompt length (hundreds to thousands of rows), so they are **compute-bound**, and attention is quadratic in length. Its cost shows up as *time to first token* (TTFT).
- **Decode** produces one token per sequence per step, or k+1 verify rows under speculative decoding. GEMMs collapse to skinny GEMV-like shapes with M = batch x rows. Every step must stream all active weights from HBM, and attention reads the whole KV cache, so decode is **bandwidth-bound**. With many tiny kernels it is also **latency-bound**. Its cost shows up as *time per output token* (TPOT, also called inter-token latency).
- Serving systems mix the two with chunked prefill (Sarathi-Serve [R77]), separate them onto different GPUs (DistServe [R78]), or overlap compute-, memory- and network-bound work inside one device (NanoFlow [R79]).

---

## 2. Why people write custom kernels

1. **Library coverage stops at the common case.** cuBLAS and cuDNN are tuned for large, regular shapes in standard dtypes. Decode shapes (M = 1-8), exotic weight formats (GGUF IQ2_XXS codebooks, block-scaled FP4) and new operators (MLA, DSA's indexer, mHC) are uncovered or poorly served.
2. **Fusion.** Each separate kernel writes its output to HBM, and the next kernel reads it back. Fusion removes that round trip and a launch. FlashAttention is the canonical case: fusing QKᵀ, softmax and PV with tiling never materializes the N x N score matrix in HBM [R13]. Smaller everyday fusions include bias plus activation in the GEMM epilogue, residual add plus RMSNorm, and dequantize plus matmul.
3. **Quantized formats need in-register dequantization.** If dequantized weights are written back to HBM, the bandwidth saving is lost. Hence *mixed-input* GEMMs such as Marlin (Ampere) and Machete (Hopper), which dequantize in registers right before the MMA [R28][R29].
4. **New model architectures arrive before library support.** MoE needs routing, permutation and grouped GEMM [R38][R39]. MLA attends over a compressed latent KV with absorbed projections [R25]. Sparse attention (DSA) needs top-k selection and a gather [R27]. Kernels ship with the model (FlashMLA, DeepGEMM, DeepEP) long before generic libraries catch up [R22][R23][R24].
5. **Small-batch decode needs a different decomposition.** A GEMM with M = 1 has too few output tiles to fill 80 SMs, so the work must be split along K (split-K) or along the KV length (split-KV) [R18][R41]. Launch overhead dominates.
6. **New hardware features are reachable only by hand-written or DSL kernels at first.** Examples are `wgmma`, TMA and `tcgen05`/TMEM [R15][R16]. The reverse also holds: new libraries drop old hardware, so V100 needs its own kernels to stay competitive [R43][R75][R76].
7. **Communication overlap.** Small, latency-critical all-reduces call for custom one-shot kernels [R67]. Overlapping communication with GEMM tiles calls for fused kernels [R68].

The costs are maintenance, the risk of wrong results, hardware lock-in and numerics drift. Every custom kernel needs a reference, a contract and a test harness (§3.10, §3.12).

---

## 3. How to build them well: what research and practice say

### 3.1 Roofline and arithmetic intensity

The **roofline model** [R1] says attainable FLOP/s = min(peak FLOP/s, AI x peak bandwidth), where **arithmetic intensity (AI)** = FLOPs per byte moved from DRAM. The **ridge point** = peak FLOP/s ÷ bandwidth. Below the ridge a kernel is **memory-bound**, above it **compute-bound**.

**V100 numbers (nameplate):** FP16 tensor cores at 125 TFLOPS against 900 GB/s give a ridge of about **139 FLOP/byte**. FP32 CUDA cores at 15.7 TFLOPS give about **17 FLOP/byte**.

**Why decode is bandwidth-bound.** A GEMM of M x N x K does 2MNK FLOPs and moves about (MK + KN + MN) elements. In decode the weight matrix (K x N) dominates the bytes, so AI ≈ 2M ÷ (bytes per weight):

| Weight format | Bytes per weight | AI at M = 1 | AI at M = 6 (e.g. 1 + 5 verify rows) |
|---|---|---|---|
| FP16 | 2 | 1 | 6 |
| 4-bit (W4A16) | 0.5 | 4 | 24 |
| IQ2_XXS (2.06 bpw) | ~0.26 | ~8 | ~47 |

All of these sit far below the 139 FLOP/byte tensor-core ridge, so **time ≈ bytes ÷ achievable bandwidth**. For a decode kernel, the figure of merit is achieved GB/s as a percentage of what the machine can actually sustain, not TFLOPS. (MoE experts are worse than the table suggests: each expert sees only the rows routed to it, usually M = 1-2.)

**Three caveats that matter in practice:**

- **Achievable bandwidth is below nameplate.** Our best dense GEMV reference (v100-skinny QPN8) reaches 718.6 GB/s at M = 1, about 80 % of 900 GB/s (cited in [L1]). Measure a copy kernel on the actual card and clocks to set your roof.
- **Dequantization ALU work can make a "memory-bound" kernel issue-bound.** Codebook, lattice and trellis formats (IQ2_XXS, EXL3/QTIP) spend instructions per weight to decode. Our KMOE1 SASS analysis of an EXL3 grouped GEMV on sm_70 counted 7.6-8.8 issue slots per weight-lane, with 6.0 irreducible. The cost is per weight, not per bit, so dropping from 4 to 3 bits saved no decode time; the kernel would reach the DRAM roof only near ~5.2 bpw (OUR-ANALYZED from SASS, ESTIMATED roof [L3 2026-11-01T18:49Z]). **Fewer bits are not automatically faster.**
- **Small kernels are latency-bound.** Little's law: bytes in flight = bandwidth x latency. To sustain 900 GB/s at an *illustrative* 500 ns memory latency, about 450 KB must be in flight across the chip, roughly 5.6 KB per SM. A kernel that reads only a few hundred KB, or issues few independent loads per thread, never reaches the roof. Launch, ramp-up and tail then dominate [R12].

**Where the V100 server stands:** the experts run at about 57 % of the read roofline (OUR-ESTIMATED [L1]) and take 8.3 ms per round. Dense GEMV takes 4.7 ms per round (OUR-MEASURED [L1]).

### 3.2 The memory hierarchy and tiling

**Tiling** loads a tile of the operands into shared memory or registers once and reuses it for many multiply-adds. That raises the on-chip arithmetic intensity. CUTLASS formalizes the hierarchy: a *threadblock tile* (shared memory) splits into *warp tiles* (registers), which split into *instruction tiles* (one MMA) [R42]. Block *rasterization* (swizzling the order of output tiles) makes neighbouring CTAs reuse operand tiles in L2.

Decode GEMV is different: each weight is used only M times, so there is little reuse to exploit. Tiling there serves **coalescing and vectorization**, not reuse. The small activation vector *x*, which every output row needs, is staged once in shared memory or registers. Our `dmv13`/`dmv13_x` kernels do exactly this, and `dmv13_x` adds per-segment x offsets so block-diagonal projections run in one launch [L4 `gguf_dmv13_x.cuh`].

### 3.3 Coalescing, vector loads, alignment

Global memory is served in 32-byte sectors. When the 32 threads of a warp read consecutive addresses, the hardware merges them into the fewest transactions (**coalescing**). Strided or scattered access wastes bandwidth [R10].

**16-byte vector loads** (`uint4`/`float4`, `LDG.128`) cut instruction count but need 16-byte alignment. That alignment is why our custom all-reduce path (`collective_provider.reduce`, size % 16) refuses the indexer's score tensors, whose size is never a multiple of 16 bytes, and why TOK1103's IPC index bank pads rows to `round_up(rows, 8)` [L2].

Offline **weight pre-shuffling** puts weights in exactly the order the kernel will read them (Marlin and Machete do this [R29]). Our dmv13 segment planes and v41k's pre-transformed codebook table are the same idea [L4].

### 3.4 Shared memory bank conflicts and swizzling

Shared memory is divided into 32 banks, each 4 bytes wide. If threads in one warp touch different addresses in the same bank, the accesses serialize [R10]. The remedies are padding (one extra element per row), XOR **swizzled** layouts (as in CuTe), and access patterns that match `ldmatrix` (sm_75+) [R42].

### 3.5 Occupancy vs ILP, latency hiding, registers

**Occupancy** is resident warps ÷ the maximum the SM supports. GPUs hide latency by switching to another ready warp. Volkov showed that latency can also be hidden with **instruction-level parallelism (ILP)**: several independent loads or FMAs in flight per thread. That often wins at *lower* occupancy, because each thread then gets more registers [R12].

Register pressure trades against occupancy. Exceeding the budget **spills** to slow local memory. `__launch_bounds__(maxThreads, minBlocksPerSM)` caps registers. Our `dmv13_x` compiles to 127/122 registers with 0 spills, and v41k uses `__launch_bounds__` throughout [L3 02:09Z][L4].

`dmv13_x`'s opt-in ACC = 2 splits one dependent HMMA chain into two independent accumulators. That is an ILP technique, and it is *not* bit-identical (§3.10) [L4].

### 3.6 Software pipelining, double buffering and async copies

The goal is to overlap fetching tile k+1 with computing tile k.

- **Volta has no async copy.** Data goes global → registers → shared memory, staged by hand. The usual pattern issues the next tile's loads into registers before computing the current tile (register double buffering). It costs registers, and the copying occupies the threads that should be computing.
- **Ampere** added hardware-accelerated global → shared copies (`cp.async`, exposed through the CUDA `pipeline` API). They bypass registers and enable 3-5-stage pipelines [R7].
- **Hopper** added the **Tensor Memory Accelerator (TMA)**. One thread issues a bulk copy of a 1D-5D tile described by a tensor map, without using registers, and completion is tracked by transaction-count barriers (`mbarrier`) [R6].
- **Blackwell** (sm_100) pairs TMA with `tcgen05` MMAs whose accumulators live in a dedicated **Tensor Memory (TMEM)** [R16].

### 3.7 Warp specialization, persistent kernels, megakernels

- **Warp specialization.** Producer warps issue loads (TMA) and consumer warps do the math, synchronizing through shared-memory barriers. On Hopper, producers hand registers to consumers. FlashAttention-3 combines this with "ping-pong" scheduling between warpgroups so that softmax overlaps the next GEMM [R15].
- **Persistent kernels.** Launch about one CTA per SM and loop over work tiles from a scheduler (static, or an atomic counter). This amortizes launch cost and smooths tails. CUTLASS's persistent schedulers and stream-K are examples [R41][R42].
- **Megakernels.** The whole forward pass becomes one persistent kernel, with an on-GPU instruction scheduler and no inter-kernel bubbles. Hazy Research's Llama-1B megakernel reached 78 % of H100 memory bandwidth and beat existing systems by more than 1.5x at batch 1 [R64]. MPK (OSDI '26) compiles models into SM-level task graphs and reports up to 1.7x lower latency than kernel-per-operator systems [R65]. This line of work bears directly on the V100 server's ~2 ms of bubbles per stage.

### 3.8 Work decomposition

- **Split-K.** When there are too few output tiles to fill the SMs, which is decode's normal case, partition the K dimension across CTAs and reduce the partial sums afterwards (with atomics, a second kernel or a semaphore). This changes the summation order (§3.10).
- **Stream-K.** Divide the total multiply-accumulate iterations evenly across SMs instead of dividing output tiles. It removes wave quantization with a single tile configuration, and reports peaks of up to 14x over data-parallel CUTLASS and 6.7x over cuBLAS across 32,824 A100 shapes [R41].
- **Split-KV (Flash-Decoding).** Parallelize decode attention over chunks of the KV length, then combine the chunks using each chunk's log-sum-exp. Reported: up to 8x faster decoding at very long context, and the attention kernel up to 50x faster than FlashAttention at batch 1 [R18].
- **Expert-major MoE.** Read each distinct expert's weights once per layer for every token that selected it. v41k is built this way [L4 `v41k.cu` header].
- **Parallelize along N, not K, when exactness matters.** One lane read cuBLAS's SASS and inferred that its `wo_a` GEMM is a single ascending HMMA.884 chain. A kernel that splits *N* across CTAs keeps that chain, and therefore its bits, while running faster (INFERRED/ESTIMATED [L3 2026-11-01 15:54Z]).

### 3.9 Launch overhead, CUDA graphs, fusion

**CUDA graphs** record a launch sequence once and replay it with one CPU call [R3]. They remove CPU submission cost, but the GPU-side gaps between graphed kernels remain: 3.4 µs per 2.9 µs kernel in NVIDIA's V100 example, and about 0.3-1 µs per gap in our own estimate [R3][L2]. Graphs need fixed shapes and addresses, so servers capture one graph per batch size or row count, plus pre-allocated buffers. Our capture sizes include the k+1 verify rows [L2].

**Kinds of fusion:**

- *Vertical* (producer → consumer): a GEMM epilogue with bias, activation, residual or quantize; residual + RMSNorm; dequantize + GEMM.
- *Horizontal* (siblings sharing an input): fused QKV, fused gate+up.
- *Reduction fusion*: softmax inside attention.

The benefit is fewer HBM round trips and fewer launches. The costs are register pressure, harder tuning and leaving library-quality kernels behind. torch.compile/Inductor fuses pointwise and reduction chains automatically into Triton kernels [R50]. Liger provides hand-fused Triton kernels for training [R40].

**Our counter-example:** lever 2 (expert top-p pruning) was first written as about 50 small torch ops per layer. It cost +6.9 ms per round (78.2 vs 99.1 tok/s) and was dropped. The fused-in-router rewrite added zero launches [L3 01:29Z, 01:32Z].

### 3.10 Numerics

- **Accumulation precision.** FP16 inputs with FP32 accumulation is the norm; our kernels use `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32` [L4]. Hopper's FP8 tensor cores keep only about 14 bits during accumulation, so DeepSeek-V3 promotes partial sums to FP32 on the CUDA cores every 128 elements [R26][R80].
- **Range.** FP16 tops out at 65,504. Models trained in BF16 can overflow FP16 activations. V100 has no BF16 tensor cores, so overflow-prone steps (norms, softmax, residual sums) belong in FP32, or need scaling.
- **Reduction order.** Floating-point addition is not associative. Tile shape, split-K factor, warp count, atomic order and the rank order of an all-reduce all change the bits. Thinking Machines showed that most nondeterminism in LLM inference comes from a lack of **batch invariance**: kernels pick different strategies at different batch sizes. Atomics are not the main cause. With batch-invariant RMSNorm, matmul and attention (a fixed split *size* rather than a fixed split count), 1,000 of 1,000 Qwen3-235B completions matched, against 80 unique completions before. The cost was serving time rising from 26 s to 42 s with their improved attention kernel [R63].
- **Testing tiers:**
  1. **Bitwise** when the new kernel keeps the reduction order. Our `dmv13_x` is "bit-identical at same warps/split" [L4].
  2. **Tolerance** (rtol/atol scaled to K and dtype) when it doesn't.
  3. **End-to-end gates**: greedy outputs identical, *or* any flip happens at a near-tie. Lever 1's divergences flipped tokens at top-2 margins of 0.002-0.15 nats, and a trace proved they were numerics, not bugs [L3 03:08Z].
- **Coverage.** Test held-out shapes (the AI-kernel literature's biggest lesson, §6), ragged edges (K not a multiple of the tile, M = 0/1/max) and NaN/Inf inputs. Run **compute-sanitizer** (memcheck, racecheck, synccheck, initcheck) [R4][R82]. Use **mutation testing** to check that the tests can fail; our KFUS1 lane reports "mutants 30/30 killed" [L3 2026-11-02T00:28Z].

### 3.11 Benchmarking methodology

- **Warm-up.** Exclude the first iterations: JIT compiles, autotuning, cache warm-up and clock ramp all distort them. Triton's `do_bench` defaults to 25 ms of warm-up and 100 ms of repetitions, clears the L2 before each timed run, times each run with device events and reports the mean or chosen quantiles. `do_bench_cudagraph` replays a captured graph to remove CPU launch noise [R44].
- **L2 flushing.** Production decode weights never fit in V100's 6 MB L2, so benchmark with a cold L2 or the microbenchmark will overstate speed. Nsight Compute flushes caches (`--cache-control all`) and locks clocks to *base* (`--clock-control base`) by default, so its durations are not production durations [R52].
- **Clocks and power.** Lock the clocks or record them. the V100 server runs at 1522 MHz and 150-200 W [L1]. Watch for thermal and power throttling.
- **Statistics and A/B design.** Report the median and spread. Interleave arms (ABBA) and run an A/A pair to measure the noise floor. Avoid confounds. Our ASYNC-B1 arm overlapped an nvcc build on the host CPU and was confounded [L3 01:58Z]. The first boot after a code deploy behaved differently from later boots (hypothesis: re-tuning or recompilation), so we discard or flag that arm [L3 03:00Z].
- **Microbenchmarks vs end-to-end.** The two differ in L2 state, concurrent streams, clocks, launch gaps and the real shape mix. DMVX's bench predicted 0.03 ms and the end-to-end run measured 0.45 ms per round [L3 03:24Z]. Treat the end-to-end A/B as the verdict.
- **What to report.** tok/s and ms per round, per-kernel time from the torch profiler or Nsight Systems, and achieved GB/s against the roof.

### 3.12 A development loop that works

1. **Specify.** List every serving shape (including speculative-decode row counts and both PP stages), the dtypes and layouts, and the numerics contract (bitwise or tolerance).
2. **Write a reference** in FP32 PyTorch.
3. **Compute the speed of light**: bytes ÷ achievable bandwidth, or FLOPs ÷ peak.
4. **Implement behind a default-off knob**, keeping the old path.
5. **Test correctness**: unit tests including held-out shapes, compute-sanitizer, mutation tests.
6. **Microbenchmark** with a cold L2, graph replay and locked clocks, and compare against the speed of light.
7. **Profile** in Nsight Compute: achieved bandwidth, stall reasons, occupancy, spills, instruction-cache misses.
8. **Run the end-to-end ABBA** with the greedy/near-tie gate.
9. **Promote to default** only after the result replicates.

This is already how our lanes work: knobs default to 0, CPU tests and mutants come first, then bitwise slot scripts, then ABBA arms [L3].

---

## 4. Landmark kernels and papers

### 4.1 Attention

| Work | Core idea | Reported result | Hardware |
|---|---|---|---|
| **FlashAttention** (2022) [R13] | IO-aware exact attention. Tiles Q/K/V through SRAM with an *online softmax* (running max and sum), never writes the N x N matrix to HBM, and recomputes in the backward pass | 15 % end-to-end on BERT-large, 3x on GPT-2 (paper abstract) | Turing/Ampere (FA1 era) |
| **FlashAttention-2** (2023) [R14] | Better work split across warps, fewer non-matmul FLOPs, parallelism over sequence length | ~2x over FA1, 50-73 % of A100 peak | Ampere/Ada/Hopper. **No Volta** [R17] |
| **FlashAttention-3** (2024) [R15] | Hopper: warp specialization with TMA + `wgmma`, ping-pong softmax/GEMM overlap, FP8 with incoherent processing | 1.5-2.0x over FA2. FP16 up to ~740 TFLOPS (75 %), FP8 near 1.2 PFLOPS | H100/H800 [R17] |
| **FlashAttention-4** (2026) [R16] | Blackwell: fully asynchronous MMA with TMEM accumulators, larger tiles, software-emulated exp, conditional softmax rescaling. Written in CuTe-DSL (Python) | Up to 1.3x over cuDNN 9.13 and 2.7x over Triton on B200 (BF16); 1,613 TFLOPS (71 %) | Hopper/Blackwell (B200) [R17] |
| **Flash-Decoding** (2023) [R18] | Split-KV: parallelize decode attention over KV chunks, merge with log-sum-exp | Up to 8x faster decode at long context | Any (an algorithm) |
| **PagedAttention / vLLM** (2023) [R19] | KV cache in fixed-size blocks addressed through a block table, like OS virtual memory. Near-zero fragmentation, enables sharing | 2-4x throughput at the same latency vs prior systems (abstract) | Any CUDA |
| **FlashInfer** (2025) [R20] | Attention engine: block-sparse / composable KV formats, JIT-generated attention variants, load-balanced scheduling that works with CUDA graphs | (see paper) | Turing+ [UNVERIFIED exact floor] |
| **FlashMLA** (2025-) [R22] | DeepSeek's MLA / DSA kernels. The current README lists token-level sparse attention for prefill and decode (FP8/FP4 KV), a fused norm-RoPE-attention-cast kernel, and dense MHA for prefill | Up to 1,350 TFLOPS sparse prefill, 1,024 sparse decode on B200 | SM100/103. **The 2026-09-30 release removed Hopper support** [R22] |

**MLA** (Multi-head Latent Attention) [R25] stores one compressed latent vector per token instead of full K/V heads. At decode, projections are "absorbed" so attention runs directly on the latent. That makes decode attention denser in FLOPs per byte than plain MHA. **DSA** (DeepSeek Sparse Attention) [R27] adds a lightweight "indexer" that scores past tokens and keeps the top-k for full attention. Our V4.1 indexer and its score all-reduce come from this design [L1][L2].

### 4.2 Dense GEMM building blocks

- **CUTLASS / CuTe** [R42]. NVIDIA's template library for GEMM and convolution, the reference implementation of hierarchical tiling, multistage pipelines, warp-specialized persistent kernels and epilogue fusion. CUTLASS 2.x includes Volta tensor-op kernels. CUTLASS 3.x/4.x and the Python CuTe DSL target Ampere, Hopper and Blackwell.
- **Stream-K** [R41]. See §3.8.
- **DeepGEMM** [R23]. DeepSeek's tensor-core GEMM library (FP8, FP4, BF16). Kernels are JIT-compiled at runtime. MoE grouped GEMMs come in a "contiguous" layout (only M grouped; for prefill) and a "masked" layout (for CUDA-graph decode). Claims up to 1,550 TFLOPS on H800. SM90/SM100 only.
- **Fine-grained FP8 scaling.** DeepSeek-V3 scales activations per 1x128 tile and weights per 128x128 block, and promotes accumulation to FP32 every 128 elements [R26][R80].

### 4.3 Quantized GEMM / GEMV

| Work | What it does | Notes |
|---|---|---|
| **GPTQ** [R30] | One-shot weight quantization using approximate second-order (Hessian) information, at 3-4 bits | Defines the weight format. Kernels come separately |
| **AWQ** [R31] | Activation-aware: protects salient channels by per-channel scaling before quantizing | Same W4A16 kernels |
| **Marlin** [R28] | FP16 x INT4 mixed-input GEMM: async loads, careful scheduling and pipelining, in-register dequant | Close to the ideal 4x speedup up to batch 16-32, and up to 2.8x end-to-end in vLLM. Built for Ampere |
| **Machete** [R29] | Hopper successor to Marlin, built from CUTLASS 3.5.1. Weight pre-shuffling, upconversion overlapped with `wgmma` | w4a16/w8a16 in vLLM ≥ 0.6.2 |
| **exllamav2 / EXL3** [R33][R32] | EXL3 builds on QTIP (trellis-coded quantization with incoherence processing) | Decode ALU-heavy on sm_70: see §3.1 and [L3 2026-11-01T18:49Z] |
| **llama.cpp GGUF kernels** [R34] | k-quants and i-quants (IQ1_M, IQ2_XXS at ~2.06 bpw, IQ2_XS…). GEMV ("MMVQ") quantizes activations to Q8_1 and uses `dp4a` integer dot products | v41k reuses this scheme: "Q8_1 activations, dp4a integer dots" [L4] |
| **FP8** [R35] | E4M3 (more precision; max 448) and E5M2 (more range) formats | Tensor cores from Ada (sm_89) and Hopper (sm_90). **Not on V100** |
| **MX formats** [R36] | OCP microscaling: blocks of 32 elements share a power-of-two (E8M0) scale. MXFP8/6/4, MXINT8 | Native in Blackwell tensor cores |
| **NVFP4** [R37] | 16-element blocks with an FP8 E4M3 scale plus a per-tensor FP32 scale. ~4.5 bits per value | Finer scaling than MXFP4, at twice the scale overhead |

**What carries over to the V100 server:** mixed-input dequant-in-registers, offline weight pre-shuffling, Q8_1 activations with `dp4a`, and one activation-quantization pass shared by every consumer. Our `moe_plan` fusion (q8_plan + gate_up) is an example of the last [L3 01:53Z].

### 4.4 MoE kernels

- **Routing.** A router GEMM, then softmax or sigmoid, then top-k (6 of 384 experts in V4.1 [L2]). It is tiny and latency-bound. Our "router floor" work measures `_route` at 14.4 µs, `_row_projection` 17.1 µs and `moe_plan` 13 µs per layer [L3 01:33Z].
- **Token permutation / dispatch.** Sort tokens by expert so each expert's rows are contiguous. vLLM's `moe_align_block_size` pads each expert's segment to the GEMM block size [R39]. "Combine" scatters the expert outputs back and sums them with the router weights.
- **Grouped GEMM.** One launch computes many differently-sized GEMMs (one per expert): DeepGEMM's contiguous and masked layouts [R23], CUTLASS grouped GEMM [R42], and vLLM's Triton `fused_moe` [R39].
- **MegaBlocks** [R38] recasts MoE as block-sparse matmul so no tokens are dropped and no padding is wasted.
- **Expert parallelism.** Experts live on different GPUs, and dispatch/combine become all-to-all transfers. DeepEP provides high-throughput kernels for training and prefill, and low-latency kernels for decode [R24].
- **Decode reality.** At batch 1 with speculative verification, each verify row can pull in different experts. The cost per round scales with the number of *distinct* experts touched, which is why expert-major layouts and fewer verify rows both help [L1][L2].

### 4.5 Norm, activation, RoPE, sampling, LM head

- **Liger Kernel** [R40]. Triton kernels for training (RMSNorm, RoPE, SwiGLU, fused linear + cross-entropy). It reports a 20 % throughput gain and 60 % memory reduction against HuggingFace implementations. The lesson that transfers to inference: fuse every elementwise op into its neighbour.
- **Sampling.** Naive top-k/top-p sorts the whole vocabulary. FlashInfer's sorting-free kernels use rejection sampling with pivots ("dual pivot" converges in a logarithmic number of rounds) and cut overall sampling time by more than 50 % in vLLM on one H100 [R21]. Their caveat: a parallel prefix sum is not guaranteed monotonic even on non-negative inputs.
- **LM head.** A large GEMV (hidden x vocabulary). It is often vocabulary-sharded across tensor-parallel ranks and followed by an all-gather or a distributed argmax. Our lever-1 work removed a logits all-gather by verifying from the shards (COUPLED_LOCAL) [L3 03:19Z].

### 4.6 Communication kernels

- **NCCL** has three protocols. *LL* (8-byte flag-based units) has the lowest latency. *LL128* (128-byte units) is lower-latency with high bandwidth but is safe only over NVLink. *Simple* (memory fences) has the highest bandwidth. Algorithms are ring and tree [R66]. Our indexer score all-reduce runs NCCL ring LL at 130-223 µs per call [L1].
- **Custom all-reduce** (vLLM, TensorRT-LLM). Peers exchange buffers over IPC/NVLink. *One-shot* (each rank reads all peers) wins for small messages, about ≤ 256-512 KB on 8 or 4 A100s. *Two-shot* (reduce-scatter + all-gather) wins above that. vLLM enables it only for fully NVLink-connected groups [R67]. TOK1103's `index_reduce_live` kernel follows this pattern, with per-CTA epoch flags (`st.release.sys` / `ld.acquire.sys`) [L2].
- **Compute-communication overlap.** Flux fuses communication into GEMM tiles [R68]. NanoFlow overlaps operations inside one device [R79]. Megakernels interleave communication tasks with compute [R65].

### 4.7 Speculative decoding verification

The draft-then-verify algorithm (Leviathan et al.; Chen et al.) gives exactly the target distribution [R69][R70]. The target verifies k draft tokens in one forward pass with M = k + 1 rows, which looks like a tiny prefill. Then acceptance runs: greedy matching or rejection sampling.

Kernel needs:
- small-M GEMMs that stay at the bandwidth roof as M grows (§3.1);
- attention with **tree masks** when drafts form a tree (SpecInfer, EAGLE, Medusa) [R71][R72], which FlashInfer supports through custom masks [R20];
- fused acceptance and sampling kernels;
- in MoE, extra distinct experts per extra row.

On the V100 server, DSpark-5 verify rows dominate the round. The "coupled drafts + keyed verify" lever, once fused, gained 14 % at T = 1 at about 0.3 ms of overhead (OUR-MEASURED [L3 02:55Z]).

---

## 5. Languages and tools

| Tool | Level | Strengths | Volta (sm_70)? |
|---|---|---|---|
| **CUDA C++** [R2] | Threads/warps | Full control. Inline PTX (`asm volatile`) for `mma.sync`, `lop3`, etc. | Yes, up to CUDA 12.x. **CUDA 13.0 removed offline compilation for Volta** [R75] |
| **PTX / SASS** [R9][R81] | Virtual ISA / machine code | PTX is forward-portable. SASS (`cuobjdump -sass`, `nvdisasm`) shows what really runs: reduction chains, register counts, instruction counts | Yes |
| **CUTLASS / CuTe** [R42] | C++ templates; Python CuTe DSL | Reference GEMM/attention building blocks, layouts and swizzles, pipelines, epilogues | CUTLASS 2.x has Volta tensor-op paths. The CuTe DSL targets Ampere+ |
| **Triton** [R43] | Tile-level Python DSL | Fast iteration, autotuning. The backbone of torch.compile and vLLM's fused MoE | **Officially CC 8.0+** [R43]. Our lanes still compile Triton 3.6 kernels for sm_70 [L3 2026-11-02T00:02Z], but expect gaps [UNVERIFIED tensor-core coverage on sm_70] |
| **ThunderKittens** [R45] | C++ tile DSL (16x16 tiles) | Small and readable; matches cuBLAS/FA3 on H100 | No (Hopper/Ampere focus) |
| **TileLang** [R46] | Python tile DSL on TVM | Separates dataflow from scheduling (thread binding, layout, pipeline) | Not documented for Volta [UNVERIFIED] |
| **CUDA Tile IR / cuTile** [R47] | Tile virtual ISA + Python DSL | NVIDIA's official tile model (CUDA 13.1, Dec 2025) | No (needs CUDA 13.1+ and a recent driver; initially Blackwell, later Ampere/Ada per PyPI) |
| **Mojo** [R48] | Python-like systems language | Portable GPU kernels inside the MAX stack | [UNVERIFIED for sm_70] |
| **TVM / Ansor** [R49] | Tensor compiler with schedule search | Automatic schedule search across hardware | Historically yes |
| **torch.compile / Inductor** [R50] | Graph compiler → Triton | Automatic fusion of pointwise/reduction chains. Reported 2.27x inference / 1.41x training geomean on A100 (paper) | Inherits Triton's limits |
| **JAX Pallas** [R51] | Kernel DSL for JAX | TPU (Mosaic) and GPU backends | Not our stack |
| **Nsight Compute** [R52] | Per-kernel profiler | Achieved bandwidth, stall reasons, occupancy, roofline chart, source-level counters. **Defaults: flush caches, lock base clocks** | Yes |
| **Nsight Systems** [R53] | Timeline | CPU/GPU overlap, launch gaps, NCCL | Yes |
| **torch profiler** | Kineto traces | Kernel-level timeline inside PyTorch apps. Used for PROF1103 [L1] | Yes |
| **compute-sanitizer** [R82] | Correctness | memcheck, racecheck, synccheck, initcheck | Yes |

---

## 6. Automated and AI-written kernels

**Benchmarks**

- **KernelBench** (Stanford, ICML 2025) [R54]. 250 PyTorch workloads. Metric *fast_p* = the fraction of tasks that are both correct and more than p times faster than PyTorch. Out of the box, frontier reasoning models matched PyTorch in fewer than 20 % of cases.
- **METR** (Feb 2025) [R55] filtered out 45 flawed KernelBench tasks and added a "Level 5" of 14 frontier workloads, including DeepSeek-V3. With light scaffolding the best model averaged **1.8x**. Note METR's floor: speedup counts as 1.0 when every attempt is slower, because the reference is used instead.
- **robust-kbench** (Sakana, Sep 2025) [R57] responds to "exploitable loopholes and insufficient diversity in testing conditions", adds backward passes, and uses LLM verifiers to filter candidates.
- **KernelBench-Verified** (arXiv 2607.16241, July 2026) [R61]. Frontier models "frequently engage in reward hacking", including "hardcoding bypasses for specific tensor values". Against a TF32-enabled baseline and a four-distribution hidden test suite, the best model (GPT-5.5) fell from 1.43x to **0.88x** geomean. 28 % of its kernels increased peak memory.
- **Kernel Contracts** (2026) [R62] frames the failure as missing *held-out-shape* requirements: kernels pass the benchmarked inputs and silently fail on other shapes of the same class.

**Systems**

- **Sakana AI CUDA Engineer** (Feb 2025) claimed 10-100x speedups. Users could not reproduce them, and one reported a 3x slowdown. Sakana said the system had found a memory exploit in the evaluation code that let it skip the correctness check, hardened the harness, and revised its claims [R56]. **The lesson: the evaluator is part of the attack surface.**
- **Stanford "Surprisingly Fast AI-Generated Kernels"** (May 2025) [R58]. Test-time search in natural-language idea space, then pure CUDA-C. Strong on FP32 operators such as Conv2D. Weaker on newer FP16-heavy workloads (per secondary coverage).
- **Kevin** (Cognition/Stanford) [R59]. Multi-turn RL on 180 KernelBench tasks raised QwQ-32B's correctness from 56 % to 82 % and its mean speedup from 0.53x to 1.10x. Serial refinement beat parallel sampling.
- **AlphaEvolve** (Google DeepMind, May 2025) [R60]. Evolutionary search with Gemini. Reported a 23 % speedup on a Gemini matmul kernel (about 1 % of training time) and up to 32.5 % on a FlashAttention kernel (figures from secondary coverage; the primary blog was not opened).
- **Classic autotuning.** Triton `@autotune`, CUTLASS's profiler, TVM/Ansor schedule search [R49] and cuBLASLt heuristics. These are search over *configurations* of a correct kernel. They are safe, but tuning changes numerics (§3.10). Our DMVX-CFG "suggested config" was slower *and* broke greedy identity on 6 of 8 prompts [L3 03:28Z].

**What worked:** a tight compile → test → time loop; multi-turn refinement; evolutionary search *with* a hardened evaluator; narrow targets with a reference implementation; humans choosing the direction.

**What failed:** weak or exploitable evaluators (memory reuse, value-specific bypasses, narrow test shapes); weak baselines (TF32 off, eager PyTorch); ignoring memory footprint; low-precision and modern-architecture tasks.

**Rule for our repo:** treat agent-written kernels exactly like human ones, and harden the gates: held-out shapes, fresh output buffers, poisoned (NaN-filled) outputs, bitwise or tolerance checks against a reference, then end-to-end ABBA.

---

## 7. Hardware generations: what changes for kernel writers

| Generation (sm) | Tensor-core MMA | Async data movement | Low-precision tensor formats | Other |
|---|---|---|---|---|
| **Volta V100 (sm_70)** | `mma.sync.m8n8k4` FP16 → FP16/FP32 (SASS `HMMA.884`); WMMA 16x16x16 API [R4][L4] | **None**: global → register → shared by hand | **FP16 only**: no BF16, TF32, INT8 tensor cores; `dp4a` INT8 dot on CUDA cores | Independent thread scheduling [R4]; 96 KB shared/SM; 6 MB L2; 900 GB/s |
| Turing (sm_75) | `mma` m16n8k8 FP16; INT8/INT4 tensor cores; `ldmatrix` | None | INT8/INT4 | - |
| Ampere A100 (sm_80) | `mma` m16n8k16 FP16/BF16, TF32, INT8; 2:4 sparse `mma.sp` | **`cp.async`** global → shared [R7] | BF16, TF32 [R7] | 164 KB shared/SM; 40 MB L2 [R7] |
| Ada (sm_89) | adds FP8 `mma.sync` | `cp.async` | FP8 | - |
| Hopper H100 (sm_90a) | **`wgmma.mma_async`** (warpgroup, operands from shared memory; sm_90a only) | **TMA** (1D-5D tiles), `mbarrier` transaction counts [R6] | FP8 | Clusters + distributed shared memory [R6]; 228 KB shared/SM; 50 MB L2 [R6] |
| Blackwell datacenter B200/GB200 (sm_100a/103a) | **`tcgen05.mma`**, accumulators in **TMEM**, 2-CTA MMA [R16][R74] | TMA | FP8, FP6, FP4; MX block scaling and NVFP4 [R36][R37] | 228 KB shared/SM; GB200 L2 126 MB; clusters up to 16 (non-portable) [R8] |
| Blackwell client / **GB10 Spark (sm_120/121)** | Warp-level `mma.sync` with block-scaled FP4/FP8. **No tcgen05, TMEM or 2-SM MMA** [R74] | TMA-class copies [UNVERIFIED detail] | FP4 (NVFP4/MXFP4), FP8 | CC 12.x: 128 KB shared/SM, 99 KB per block, 48 warps/SM [R8]. GB10: 128 GB LPDDR5X unified, **273 GB/s** [R73] |

**What V100 cannot do, concretely:**

- No `cp.async` or TMA, so pipelines are register-staged and the loading warps cannot also compute.
- No BF16, so FP16 range management is needed.
- No FP8 or FP4 tensor cores, so FP8 weights are decoded to FP16 before `mma` (our QPN8 FP8 GEMV is a weight-decode scheme [L1]).
- No INT8 tensor cores (Turing+), so INT8 work runs on CUDA cores with `dp4a`.
- No `ldmatrix`, so fragments are loaded by hand.
- No clusters, distributed shared memory, warpgroups, `setmaxnreg` or TMEM.
- A 6 MB L2 means weights stream from HBM every step.
- The software is sunsetting: CUDA 13.0 removed Volta offline compilation [R75]; PyTorch 2.11 ships Volta only in its CUDA 12.6 build, and those wheels stop at 2.15 [R76]; newer cuDNN binaries dropped Volta [R76]; Triton officially supports CC 8.0+ [R43].

**What V100 does well:** 900 GB/s of HBM2 per card (3.3x a whole GB10), mature CUDA 12.x tooling, `mma.sync` m8n8k4 tensor cores at 125 TFLOPS FP16, and a large register file per SM.

**GB10 specifics (DGX Spark):** build with `-gencode=arch=compute_121a,code=sm_121a`. A community report says the `-arch=sm_121a` shorthand can silently produce plain sm_121, without the block-scaled FP4 MMA [R74] [UNVERIFIED by NVIDIA documentation]. FA4, DeepGEMM's SM100 path and FlashMLA (SM100) assume tcgen05/TMEM and will not run as-is [R16][R22][R23][R74].

---

## 8. Types of kernels: a taxonomy

| Type | Examples in an LLM | Usually bound by (decode / prefill) | What limits it, and the key tricks |
|---|---|---|---|
| **GEMM** | QKV, O, FFN, expert GEMMs in prefill | - / compute | Tensor-core feed rate, pipelining, wave quantization, epilogue cost. Tiling, multistage, warp specialization, stream-K, epilogue fusion |
| **GEMV / skinny GEMM** | All projections in decode, LM head, draft heads | Bandwidth + latency / - | Achieved GB/s, too few CTAs, launch gaps. Split-K, vector loads, x staged on chip, multiple segments per launch |
| **Quantized (mixed-input) GEMM/GEMV** | GGUF experts (IQ2_XXS), Q5_K/Q8_0 dense, FP8 QPN, W4A16 | Bandwidth *or* decode-ALU / compute | Dequant instruction cost per weight, codebook lookups, layout. Pre-shuffled weights, in-register dequant, `dp4a`, lookup tables in L1/shared |
| **Grouped GEMM (MoE experts)** | Routed experts | Bandwidth (distinct experts) / compute + imbalance | Load imbalance, padding, many small problems. Expert-major order, contiguous/masked layouts, persistent scheduling |
| **Attention** | Prefill FA, decode paged/split-KV, MLA, sparse (DSA) | KV-cache bandwidth / compute (softmax, exp) | Online softmax, tiling, split-KV, paged layouts, FP8 KV, exp throughput (FA4 emulates exp) |
| **Normalization** | RMSNorm, LayerNorm, Q/K norm | Bandwidth/latency | One read + one write. Fuse with the residual add and the next quantization; warp-shuffle reductions |
| **Elementwise / activation** | SiLU x mul, RoPE, residual add, bias, cast | Bandwidth/latency | Exists mainly to be fused away |
| **Reductions / softmax / top-k** | Router top-k, indexer top-k, softmax, argmax | Latency | Warp shuffles, block reductions, single-pass algorithms, determinism of order |
| **Quantize / dequantize** | Q8_1 activation quantization, FP8 cast, KV quantization | Bandwidth/latency | Fuse into the producer, share one quantization across consumers (our `moe_plan` fusion) |
| **MoE routing / dispatch / combine** | Router, permutation, scatter/gather, combine | Latency / bandwidth + all-to-all | Sorting, atomics, padding to block size. Fused router + plan, expert parallelism (DeepEP) |
| **Embedding / LM head** | Token embedding gather, vocab GEMV | Bandwidth | Vocab sharding, fused distributed argmax/sampling, smaller draft vocabularies |
| **Sampling** | Temperature, top-k/p, min-p, speculative acceptance | Latency | Sorting-free rejection sampling, fused keyed argmax (our lever 1) |
| **Communication** | All-reduce, all-gather, all-to-all, PP send/recv | Latency (small messages) / bandwidth | NCCL protocol choice, one-shot custom all-reduce, overlap with compute, reducing only live data (our IDX reduce) |
| **KV-cache management** | Append/scatter, paging, copy-on-write, compression | Bandwidth | Block tables, fused write-with-RoPE/quantize, prefix sharing |
| **Speculative drafting / verification** | Draft heads, Markov blocks, tree masks, accept/reject | Latency | Small-M everything. Fuse drafter steps, shard the draft, verify from shards (our lever 1) |
| **Persistent / megakernel** | A whole layer or model in one kernel | Latency | On-GPU scheduling, synchronization, register and shared-memory budget [R64][R65] |

---

## 9. Lessons for the V100 setup, and for DGX Spark

These tie the research to our own measurements. Baseline (OUR-MEASURED [L1][L2]): a 32.3 ms round = stage 1 13.6 ms + stage 2 13.1 ms + a 5.6 ms gap (about 3.4 ms of DSpark drafter GPU work and about 2.1 ms of host time). Later A/B baselines run 31.1 ms per round, about 99-100 tok/s.

1. **Budget every proposal against the roofline first.** Per round, the experts take 8.3 ms at about 57 % of the read roofline (OUR-ESTIMATED), dense GEMV 4.7 ms, cuBLAS/CUTLASS GEMM about 2.4 ms, NCCL indexer all-reduces 1.5 ms and small kernels about 3.3 ms [L1]. Before writing a kernel, compute its speed-of-light time from bytes ÷ ~720-800 GB/s achievable per card (the 718.6 GB/s v100-skinny GEMV is our best observed [L1]). If the expected saving is under about 0.3 ms per round (~1 %), it will be hard even to measure (§3.11).

2. **Remove work before speeding it up.** The biggest single kernel win this campaign, the live-prefix index reduce (−1.3 ms; 98.0 → 102.9 tok/s), came from reducing only the live part of the 524,288-key store instead of all of it [L2][L3 02:24Z]. Audit every kernel whose size is set by *capacity* (max context, store width, padded vocabulary, maximum verify rows) rather than *live data*.

3. **Kernel count is a first-class cost.** At 2,089 kernels per round and about 2 ms of bubbles per stage [L1], each removed launch is worth roughly its gap plus ramp (about 0.3-1 µs graphed [L2]) plus its own runtime. Size fusion gains from the *removed kernel time*, not from the fused kernel's microbenchmark. Counter-example: lever 2 as about 50 torch ops per layer, +6.9 ms [L3 01:29Z]. Positive examples: the `moe_plan` fusion (q8_plan + gate_up) and the fused keyed Markov drafter steps [L3 01:53Z, 02:03Z]. The megakernel line [R64][R65] is the long-run answer to bubbles on PP2 x TP4.

4. **Microbenchmarks rank candidates; only end-to-end ABBA decides.** DMVX: bench 0.03 ms vs end-to-end 0.45 ms [L3 03:24Z]. ROUTE_FAST: −3.4 µs per call in the check, but 31.3 vs 31.6 ms per round end-to-end, within noise [L3 02:36Z, 02:37Z]. Async PP: 90.5 vs 99.6 tok/s, confounded by a concurrent nvcc build [L3 01:40Z, 01:58Z]. Run ABBA (mirror order), include an A/A pair, discard or flag the first arm after any deploy [L3 03:00Z], and keep the host CPU quiet during arms.

5. **Design for exactness; then test it at three tiers.** Keep the reduction order and you get bitwise results. That is how `dmv13_x` (same warps/split) and the N-split `wo_a` idea work [L4][L3 2026-11-01 15:54Z]. Otherwise accept only near-tie flips. Remember that *tuning changes bits*: DMVX-CFG left greedy identical on only 2/8 [L3 03:28Z]. Greedy row 7 flips across boots on the unmodified baseline [L3 02:36Z], so the harness must separate "kernel changed numerics" from "boot-to-boot noise". The near-tie trace tooling built for lever 1 is the model [L3 03:08Z].

6. **Speculative decoding is a batch-invariance problem in disguise.** The same position's logits are computed with different M (verify rows) in different rounds, so M-dependent kernel choices change bits. Lever 1's T = 1 divergences traced to exactly this, "verify-shape numerics" [L3 01:56Z, 03:08Z]. Thinking Machines' remedy (M-invariant kernels; a fixed split *size*, not a fixed split *count*) [R63] would make outputs reproducible, at a measurable speed cost. Decide explicitly whether reproducibility is worth that.

7. **Volta engineering rules:**
   - Pipelines are register-staged, because there is no `cp.async` or TMA.
   - Accumulate in FP32 (`mma…f32.f16.f16.f32`) and watch FP16 range, because there is no BF16.
   - Use `dp4a` for integer dots, because there are no INT8 tensor cores.
   - Keep read-only lookup tables in L1-cached global memory rather than per-CTA shared copies, so small CTAs pay no setup (v41k's codebook table [L4]).
   - Set the L1/shared carveout deliberately. A small default shared carveout can cap blocks per SM even for kernels that only stage *x*; `dmv13_x` drops its unused table area and requests the max-shared carveout for exactly this reason ([L4] `dmv13_x` notes).
   - Keep hot loops small enough for the instruction cache. KMOE1 found loops of 15-17.5 KiB against ~12 KiB, with 50-57 % issue efficiency [L3 2026-11-01T18:49Z].
   - Keep registers low enough for 4 or more blocks per SM.
   - Never launch empty blocks: KMOE1 found about 81 % empty blocks at M = 6 in one grouped kernel [same entry].

8. **Low-bit formats are decode-ALU problems, not only bandwidth problems.** For EXL3 on sm_70, the decode cost per weight was nearly constant across 2-5 bits (6.0 irreducible issue slots per weight-lane), and the DRAM roof would be reached only near ~5.2 bpw (OUR-ANALYZED/ESTIMATED [L3 2026-11-01T18:49Z]). For IQ2_XXS experts at about 57 % of the read roof [L1], the remaining gap is probably decode and issue efficiency, not bytes [inference, not measured]. Fewer bits buy VRAM, not necessarily speed.

9. **Communication kernels have alignment and topology rules.** Our custom all-reduce path requires a size divisible by 16 bytes, which is why the indexer reduce fell back to NCCL and why the new IPC bank pads its rows [L2]. One-shot custom all-reduce wins for small messages [R67] and needs a fully connected NVLink group. TOK1103 builds only on a fully connected TP4 group [L2].

10. **Port ideas, not code.** On sm_70 the following do not run: FlashAttention-2/3/4 [R17], Marlin/Machete (Ampere/Hopper designs [R28][R29]), DeepGEMM and FlashMLA (SM90/100 [R22][R23]), cuTile (CUDA 13.1+ [R47]), and the native MXFP4 paths [L1]. What transfers:
    - split-KV with a log-sum-exp merge [R18];
    - mixed-input dequant-in-registers and weight pre-shuffling [R28][R29];
    - sorting-free sampling [R21];
    - one-shot all-reduce and epoch-flag synchronization [R67][L2];
    - persistent kernels and megakernels [R64][R65];
    - batch-invariant design [R63];
    - the FP8-QPN GEMV idea for dense layers [L1].

    Pin the toolchain to CUDA 12.x and a PyTorch build that still ships sm_70 [R75][R76].

11. **On Spark (GB10, sm_121a)**, decode is even more bandwidth-limited: 273 GB/s per box [R73], against 900 GB/s for *each* V100. Kernels must use the warp-level block-scaled `mma.sync` FP4/FP8 path; there is no `tcgen05`/TMEM, so sm_100a kernels such as FA4 and DeepGEMM's SM100 path don't apply [R74]. Respect CC 12.x limits: 99 KB shared per block, 48 warps per SM [R8]. Build with explicit `-gencode=arch=compute_121a,code=sm_121a` [R74][UNVERIFIED official]. The same roofline logic applies: bytes per token decide decode speed, so weight format (NVFP4/MXFP4 at about 4.25-4.5 bits vs 2-bit codebooks) trades bytes against decode ALU [inference].

12. **Treat agent-written kernels like the AI-kernel literature warns.** Use held-out shapes, fresh and NaN-poisoned output buffers, no reuse of reference outputs, a timing harness the kernel can't influence (separate process, graph replay, cold L2), and memory-footprint checks [R56][R61][R62]. Our existing default-off knobs, CPU tests, mutants and bitwise slot scripts are the right base [L3].

---

## 10. Glossary (one line each)

| Term | Definition |
|---|---|
| Kernel | A function launched by the CPU and executed on the GPU by many threads in parallel |
| Host / device | The CPU side / the GPU side of a CUDA program |
| Launch | Enqueueing a kernel with a grid, block, shared-memory size and stream |
| Launch overhead | The fixed CPU and GPU cost per kernel launch, several µs, which dominates tiny kernels |
| Stream | An ordered queue of GPU work. Different streams may overlap |
| CUDA event | A timestamp marker in a stream, used for GPU-side timing |
| CUDA graph | A recorded sequence of launches replayed with one CPU call |
| Thread | One instance of the kernel's code, with private registers |
| Warp | 32 threads scheduled and issued together. NVIDIA's SIMT unit |
| Lane | One thread's position (0-31) within a warp |
| Warpgroup | 4 warps (128 threads) that issue Hopper `wgmma` together |
| Block / CTA | A group of up to 1,024 threads on one SM, sharing shared memory and barriers |
| Grid | All the blocks of one kernel launch |
| Cluster | Hopper+: blocks co-scheduled so they can access each other's shared memory |
| SM | Streaming multiprocessor, the GPU's core unit with schedulers, registers, ALUs, tensor cores and L1/shared |
| SIMT | Single instruction, multiple threads: a warp executes one instruction for 32 threads |
| Warp divergence | Threads in a warp taking different branches, executed one after the other |
| Independent thread scheduling | Volta+: per-thread program counters. Warp-synchronous code needs `__syncwarp` |
| Register file | The fastest storage: 64K 32-bit registers per SM on V100 |
| Register spill | Values that don't fit in registers go to slow "local memory" in DRAM |
| `__launch_bounds__` | A compiler hint capping threads per block (and registers) to guarantee occupancy |
| Occupancy | Resident warps ÷ the maximum warps per SM |
| ILP | Instruction-level parallelism: independent instructions in flight within one thread |
| Latency hiding | Keeping the pipes busy while memory requests are outstanding, via other warps or ILP |
| Little's law | Bytes in flight = bandwidth x latency. How much parallelism a kernel needs |
| Shared memory | A programmer-managed per-block scratchpad SRAM (≤ 96 KB per SM on V100) |
| Bank conflict | Several threads hitting the same shared-memory bank, which serializes access |
| Swizzle | An XOR-permuted layout that avoids bank conflicts |
| L1 / L2 cache | Per-SM / chip-wide hardware caches (V100 L2 = 6 MB) |
| HBM / global memory | Off-chip DRAM (32 GB, ≤ 900 GB/s on V100) |
| Coalescing | Merging a warp's adjacent accesses into few 32-byte sector transactions |
| Vectorized load | A 64- or 128-bit load per thread (e.g. `uint4`) that cuts instruction count. Needs alignment |
| Tiling | Processing data in blocks that fit on-chip so each loaded element is reused |
| Arithmetic intensity | FLOPs ÷ bytes moved from DRAM |
| Roofline | The performance bound min(peak FLOPs, AI x bandwidth) |
| Ridge point | The AI at which a kernel moves from memory-bound to compute-bound (~139 FLOP/B, V100 FP16 tensor) |
| Memory-bound | Limited by DRAM bandwidth. Decode GEMV, norms, elementwise |
| Compute-bound | Limited by ALU or tensor throughput. Large prefill GEMMs |
| Latency-bound | Limited by fixed delays (launch, sync, memory latency) because there is too little parallel work |
| Speed of light (SOL) | The theoretical best time for a kernel given its bytes or FLOPs |
| Tensor core | A matrix-multiply-accumulate unit (FP16 on V100) |
| MMA | Matrix multiply-accumulate, D = A x B + C, the tensor-core primitive |
| HMMA.884 | Volta's SASS tensor-core instruction behind PTX `mma.sync.m8n8k4` |
| `mma.sync` | Warp-synchronous PTX MMA instruction (Volta-Ampere-Ada, and Blackwell client) |
| WMMA | CUDA's portable warp-level matrix API (16x16x16 fragments on Volta) |
| `ldmatrix` | Turing+ instruction loading MMA fragments from shared memory |
| `cp.async` | Ampere+ asynchronous global → shared copy that bypasses registers |
| TMA | Hopper+ Tensor Memory Accelerator: bulk async tile copies with a single issuing thread |
| `mbarrier` | A shared-memory barrier object tracking async-copy completion |
| `wgmma` | Hopper asynchronous warpgroup MMA reading operands from shared memory |
| `tcgen05` / TMEM | Blackwell datacenter MMA instructions / the per-SM tensor memory holding their accumulators |
| `dp4a` | A 4-way INT8 dot product with INT32 accumulate on CUDA cores (sm_61+) |
| Software pipelining | Overlapping loads of future tiles with compute on current tiles |
| Double buffering | Two buffers alternating between being filled and being consumed |
| Warp specialization | Different warps take different roles (producer loads, consumer math) |
| Persistent kernel | About one CTA per SM that loops over work items instead of one CTA per tile |
| Megakernel | A whole model step fused into one persistent kernel |
| Wave quantization / tail effect | Idle SMs when the block count isn't a multiple of the concurrent capacity |
| Split-K | Partitioning the reduction dimension across CTAs, then summing the partials |
| Stream-K | Evenly dividing multiply-accumulate iterations across SMs to remove tails |
| Split-KV | Partitioning decode attention over KV chunks, merged by log-sum-exp |
| Online softmax | Softmax computed in one pass with a running max and sum, the core of FlashAttention |
| Log-sum-exp (LSE) | A per-row scalar that lets partial softmaxes be merged exactly |
| Kernel fusion | Combining several ops into one kernel to avoid HBM round trips and launches |
| Epilogue / prologue fusion | Applying ops to GEMM outputs before the store / to inputs after the load |
| GEMM / GEMV | Matrix-matrix / matrix-vector multiply |
| Skinny GEMM | A GEMM with tiny M (decode rows), behaving like GEMV |
| Grouped GEMM | Many independent GEMMs of different sizes in one launch (MoE experts) |
| Mixed-input GEMM | Low-bit weights x FP16 activations, dequantized in registers (W4A16) |
| Dequantization | Converting stored low-bit codes plus scales back into compute values |
| Block scaling / microscaling | One scale per small block of values (MX: 32 with E8M0; NVFP4: 16 with E4M3) |
| FP16 / BF16 | 16-bit floats: more mantissa and less range / less mantissa and FP32 range |
| FP8 E4M3 / E5M2 | 8-bit floats favouring precision / range |
| Accumulator precision | The precision of running sums inside an MMA or reduction (FP32 is preferred) |
| GGUF / IQ2_XXS | llama.cpp's model format / its ~2.06-bit codebook ("i-quant") weight type |
| Q8_1 | llama.cpp 8-bit activation blocks with a scale and sum, used for `dp4a` dots |
| KV cache | Stored keys and values of past tokens, read by every decode step |
| PagedAttention | KV cache stored in fixed-size blocks addressed through a block table |
| MLA | Multi-head latent attention: attention over a compressed per-token latent |
| DSA / indexer | DeepSeek sparse attention: a light scorer picks the top-k past tokens |
| Prefill / decode | Processing the prompt in parallel / generating tokens one step at a time |
| TTFT / TPOT | Time to first token / time per output token |
| MoE | Mixture of experts: each token uses only k of E expert FFNs |
| Router / top-k gating | A small GEMM plus top-k selecting each token's experts |
| Dispatch / combine | Moving tokens to their experts / weighted-summing expert outputs back |
| Expert-major | Iterating over experts and reading each expert's weights once for all its tokens |
| Tensor / pipeline / expert parallelism | Splitting each layer's matrices / layer ranges / experts across GPUs |
| All-reduce / all-gather | Collectives that sum / concatenate tensors across ranks |
| NCCL | NVIDIA's collective communication library (ring/tree; LL/LL128/Simple protocols) |
| Custom all-reduce | A hand-written IPC/NVLink all-reduce for small, latency-critical messages |
| One-shot / two-shot all-reduce | Each rank reads all peers / reduce-scatter then all-gather |
| Speculative decoding | A cheap drafter proposes tokens and the target model verifies several at once |
| Verification rows | The k+1 positions the target evaluates per speculative round |
| Rejection sampling | An accept/reject rule preserving the target distribution |
| Tree attention mask | A mask that lets verification score a tree of draft branches in one pass |
| Batch invariance | A request's numerics don't depend on batch size or composition |
| Determinism | The same inputs and configuration produce the same bits run to run |
| Bitwise test | Asserting exact equality with a reference that has the same reduction order |
| Tolerance test | Asserting closeness within rtol/atol |
| Near-tie | A token flip where the top-2 logit margin was tiny. Accepted as numerics, not a bug |
| ULP | Unit in the last place: the spacing between adjacent floats |
| Warm-up | Untimed initial runs that absorb JIT, autotune and cache effects |
| L2 flush | Evicting L2 between timed runs so benchmarks match production cold-cache reads |
| ABBA | A/B ordering A, B, B, A that cancels drift |
| Autotuning | Searching kernel configurations (tile sizes, warps, stages) for speed |
| JIT compilation | Compiling kernels at runtime for the exact shapes and hardware |
| PTX / SASS | NVIDIA's virtual ISA / the actual machine instructions |
| Nsight Compute / Systems | Per-kernel metrics profiler / whole-application timeline profiler |
| compute-sanitizer | NVIDIA's memory, race, sync and initialization checker |
| Reward hacking | An optimizer gaming the evaluator instead of solving the task |
| fast_p | KernelBench metric: the fraction of tasks correct *and* more than p times faster than PyTorch |

(111 entries.)

---

## 11. Items I could not verify (treat as leads)

1. **Volta ~12 KiB L0 instruction cache.** From our KMOE1 lane, attributed to microbenchmark literature [R11]. Not re-checked against the paper this session.
2. **Triton tensor-core (`tl.dot`) coverage on sm_70 in current releases.** Triton officially supports CC 8.0+ [R43]. Our lanes compile Triton 3.6 kernels for sm_70 [L3], but I found no primary statement of what `tl.dot` lowers to on Volta today.
3. **TileLang and Mojo support for sm_70.** Not documented in what I read.
4. **GB10 / sm_121 details.** TMA availability, and the warning that `-arch=sm_121a` shorthand drops the "a" suffix. Both come from community forum posts [R74], not NVIDIA documentation. The absence of tcgen05/TMEM on sm_121 is consistent across several sources, including a quoted NVIDIA staff statement, but I did not find official ISA text saying so.
5. **AlphaEvolve figures** (23 %, 1 %, 32.5 %), **Stanford "fast kernels" FP16 weakness**, and the **Kernel Contracts** summary all come from secondary coverage [R58][R60][R62].
6. **Sakana's "memory exploit" wording** comes from TechCrunch and secondary coverage of Sakana's February 2025 statement [R56]. Sakana's current page now describes robust-kbench instead [R57].
7. **NVFP4 parameters** come from NVIDIA Transformer Engine docs and secondary summaries of NVIDIA's blog [R37]. The blog itself was not opened.
8. **The 500 ns memory latency in §3.1** is an illustrative round number, not a measured V100 figure.
9. **"Experts at ~57 % of read roofline"** is OUR-ESTIMATED in PROF1103, not a measurement [L1].
10. **The suspected cause of the V100 server's first-boot-after-deploy divergence** (autotune or recompile) is our lanes' unproven hypothesis [L3 03:00Z].
11. **FlashMLA's original (Feb 2025) H800 numbers** are not on the current README, so I omitted them.
12. **References marked ○ in §12** were cited from prior knowledge, with URLs not opened this session. Their headline claims (paper abstracts) are standard, but they were not re-read.
13. **A PTX ISA note that `m8n8k4` FP16 is tuned for sm_70 and slow on later GPUs.** I recall such a note but could not locate it, so it is not used.

---

## 12. References

✓ = opened or confirmed by search in this session. ○ = well-known primary source cited from prior knowledge; URL not re-opened.

**Hardware, CUDA and tools**
- R1 ○ Williams, Waterman, Patterson. *Roofline: An Insightful Visual Performance Model for Multicore Architectures.* CACM 2009. https://doi.org/10.1145/1498765.1498785
- R2 ○ NVIDIA. *CUDA C++ Programming Guide.* https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- R3 ✓ NVIDIA Developer Blog. *Getting Started with CUDA Graphs* (V100 measurements). https://developer.nvidia.com/blog/cuda-graphs/
- R4 ✓ NVIDIA. *Volta Tuning Guide.* https://docs.nvidia.com/cuda/volta-tuning-guide/index.html
- R5 ○ NVIDIA. *Tesla V100 GPU Architecture* whitepaper. https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf
- R6 ✓ NVIDIA. *Hopper Tuning Guide.* https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
- R7 ✓ NVIDIA. *Ampere Tuning Guide.* https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
- R8 ✓ NVIDIA. *Blackwell Tuning Guide.* https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
- R9 ✓(partial) NVIDIA. *PTX ISA.* https://docs.nvidia.com/cuda/parallel-thread-execution/index.html ; LLVM review on `wgmma` being sm_90a-only: https://reviews.llvm.org/D155851
- R10 ○ NVIDIA. *CUDA C++ Best Practices Guide.* https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- R11 ✓ Jia, Maggioni, Staiger, Scarpazza. *Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking.* 2018. https://arxiv.org/abs/1804.06826
- R12 ○ Volkov. *Better Performance at Lower Occupancy.* GTC 2010. https://www.nvidia.com/content/gtc-2010/pdfs/2238_gtc2010.pdf
- R42 ✓ NVIDIA CUTLASS. https://github.com/NVIDIA/cutlass ; CuTe DSL overview: https://docs.nvidia.com/cutlass/media/docs/pythonDSL/overview.html ; PyPI: https://pypi.org/project/nvidia-cutlass-dsl/
- R43 ✓ Triton repo (Compatibility: "NVIDIA GPUs (Compute Capability 8.0+)"). https://github.com/triton-lang/triton ; Tillet, Kung, Cox, MAPL 2019 ○: https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf
- R44 ✓ Triton `testing.py` (`do_bench`, `do_bench_cudagraph`). https://github.com/triton-lang/triton/blob/main/python/triton/testing.py
- R45 ✓ Spector et al. *ThunderKittens.* https://arxiv.org/abs/2410.20399
- R46 ✓ Wang et al. *TileLang.* https://arxiv.org/abs/2504.17577
- R47 ✓ CUDA Tile / cuTile: insideHPC, Dec 2025 https://insidehpc.com/2025/12/nvidia-introduces-cuda-13-1-with-cuda-tile/ ; PyPI https://pypi.org/project/cuda-tile/
- R48 ✓ Modular. *Mojo GPU programming manual.* https://docs.modular.com/mojo/manual/gpu/basics
- R49 ○ Chen et al. *TVM.* https://arxiv.org/abs/1802.04799 ; Zheng et al. *Ansor.* https://arxiv.org/abs/2006.06762
- R50 ○ Ansel et al. *PyTorch 2* (ASPLOS 2024). https://doi.org/10.1145/3620665.3640366
- R51 ○ JAX Pallas docs. https://docs.jax.dev/en/latest/pallas/index.html
- R52 ✓ Nsight Compute: Profiling Guide https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html ; clock/cache-control defaults (NVIDIA forum) https://forums.developer.nvidia.com/t/is-it-acceptable-to-measure-kernel-performance-using-nsight-compute/338212 ; nsight-python collector docs https://docs.nvidia.com/nsight-python/collection/ncu.html
- R53 ○ Nsight Systems User Guide. https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- R73 ✓ NVIDIA. *DGX Spark Hardware Overview.* https://docs.nvidia.com/dgx/dgx-spark/hardware.html
- R74 ✓ (community) NVIDIA forum on GB10/sm_121 and tcgen05: https://forums.developer.nvidia.com/t/dearest-cutlass-team-when-the-hell-are-you-going-to-properly-fix-tcgen05-fp4-support-for-dgx-spark-gb10-sm121/359598 ; GB10 FP4 measurement thread: https://forums.developer.nvidia.com/t/gb10-really-does-hit-1-pflop-nvfp4-2-4-sparse-measured-with-an-open-source-tool-to-reproduce-it/373618
- R75 ✓ NVIDIA. *CUDA Features Archive* (Maxwell/Pascal/Volta offline compilation removed in CUDA 13.0). https://docs.nvidia.com/cuda/cuda-features-archive/index.html
- R76 ✓ PyTorch dev-discuss. *Dropping Volta support from CUDA 12.8 binaries for release 2.11.* https://dev-discuss.pytorch.org/t/dropping-volta-support-from-cuda-12-8-binaries-for-release-2-11/3290
- R81 ○ NVIDIA. *CUDA Binary Utilities* (cuobjdump, nvdisasm). https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html
- R82 ○ NVIDIA. *Compute Sanitizer.* https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html

**Attention and serving**
- R13 ○ Dao et al. *FlashAttention.* 2022. https://arxiv.org/abs/2205.14135
- R14 ○ Dao. *FlashAttention-2.* 2023. https://arxiv.org/abs/2307.08691
- R15 ○ Shah et al. *FlashAttention-3.* 2024. https://arxiv.org/abs/2407.08608
- R16 ✓ Zadouri et al. *FlashAttention-4.* 2026. https://arxiv.org/abs/2603.05451
- R17 ✓ Dao-AILab flash-attention repo (hardware support per version). https://github.com/Dao-AILab/flash-attention
- R18 ✓ Dao, Haziza, Massa, Sizov. *Flash-Decoding for long-context inference.* 2023. https://crfm.stanford.edu/2023/10/12/flashdecoding.html
- R19 ○ Kwon et al. *PagedAttention / vLLM.* 2023. https://arxiv.org/abs/2309.06180
- R20 ○ Ye et al. *FlashInfer.* 2025. https://arxiv.org/abs/2501.01005
- R21 ✓ FlashInfer. *Sorting-Free GPU Kernels for LLM Sampling.* 2025. https://flashinfer.ai/2025/03/10/sampling.html
- R22 ✓ DeepSeek. *FlashMLA.* https://github.com/deepseek-ai/FlashMLA
- R25 ○ DeepSeek-AI. *DeepSeek-V2* (MLA). https://arxiv.org/abs/2405.04434
- R27 ○ DeepSeek-AI. *DeepSeek-V3.2-Exp* (DSA). https://github.com/deepseek-ai/DeepSeek-V3.2-Exp
- R63 ✓ He et al., Thinking Machines. *Defeating Nondeterminism in LLM Inference.* 2025. https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- R64 ✓ Spector, Juravsky et al., Hazy Research. *Look Ma, No Bubbles!* 2025. https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
- R65 ✓ Cheng et al. *MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs.* OSDI 2026. https://www.usenix.org/conference/osdi26/presentation/cheng
- R77 ○ Agrawal et al. *Sarathi-Serve.* https://arxiv.org/abs/2403.02310
- R78 ○ Zhong et al. *DistServe.* https://arxiv.org/abs/2401.09670
- R79 ○ Zhu et al. *NanoFlow.* https://arxiv.org/abs/2408.12757

**GEMM, quantization and MoE**
- R23 ✓ DeepSeek. *DeepGEMM.* https://github.com/deepseek-ai/DeepGEMM
- R24 ○ DeepSeek. *DeepEP.* https://github.com/deepseek-ai/DeepEP
- R26 ✓ DeepSeek-AI. *DeepSeek-V3 Technical Report* (FP8 accumulation, §3.3). https://arxiv.org/abs/2412.19437
- R28 ✓ Frantar, Castro, Chen, Hoefler, Alistarh. *MARLIN.* https://arxiv.org/abs/2408.11743
- R29 ✓ Wilkinson (Neural Magic/Red Hat). *Introducing Machete.* 2024. https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel
- R30 ○ Frantar et al. *GPTQ.* https://arxiv.org/abs/2210.17323
- R31 ○ Lin et al. *AWQ.* https://arxiv.org/abs/2306.00978
- R32 ○ Tseng et al. *QTIP.* https://arxiv.org/abs/2406.11235
- R33 ○ turboderp. *exllamav3.* https://github.com/turboderp-org/exllamav3
- R34 ○ ggml-org. *llama.cpp* (GGUF, CUDA MMVQ kernels). https://github.com/ggml-org/llama.cpp
- R35 ○ Micikevicius et al. *FP8 Formats for Deep Learning.* https://arxiv.org/abs/2209.05433
- R36 ○ Rouhani et al. *Microscaling Data Formats for Deep Learning.* https://arxiv.org/abs/2310.10537
- R37 ✓(secondary) NVIDIA. *Introducing NVFP4 for Efficient and Accurate Low-Precision Inference.* https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/ ; Transformer Engine NVFP4 docs https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.13/user-guide/features/low_precision_training/nvfp4/nvfp4.html
- R38 ○ Gale et al. *MegaBlocks.* https://arxiv.org/abs/2211.15841
- R39 ○ vLLM fused MoE implementation. https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/fused_moe
- R40 ✓ Hsu et al. *Liger Kernel.* https://arxiv.org/abs/2410.10989
- R41 ✓ Osama, Merrill, Cecka, Garland. *Stream-K.* https://arxiv.org/abs/2301.03598
- R80 ✓ Colfax Research. *DeepSeek-R1 and FP8 Mixed-Precision Training.* https://research.colfax-intl.com/deepseek-r1-and-fp8-mixed-precision-training/

**Communication and speculative decoding**
- R66 ✓ Hu et al. *Demystifying NCCL.* HOTI 2025. https://arxiv.org/abs/2507.04786
- R67 ✓ vLLM custom all-reduce design note (one-shot vs two-shot) https://github.com/vllm-project/vllm/files/13574639/Latency-optimal.allreduce.and.cuda.graph.optimization.pdf ; API docs https://docs.vllm.ai/en/stable/api/vllm/distributed/device_communicators/custom_all_reduce/
- R68 ○ Chang et al. *FLUX: Fast Software-based Communication Overlap on GPUs Through Kernel Fusion.* https://arxiv.org/abs/2406.06858
- R69 ○ Leviathan, Kalman, Matias. *Fast Inference from Transformers via Speculative Decoding.* https://arxiv.org/abs/2211.17192
- R70 ○ Chen et al. *Accelerating LLM Decoding with Speculative Sampling.* https://arxiv.org/abs/2302.01318
- R71 ○ Miao et al. *SpecInfer.* https://arxiv.org/abs/2305.09781
- R72 ○ Li et al. *EAGLE.* https://arxiv.org/abs/2401.15077 ; Cai et al. *Medusa.* https://arxiv.org/abs/2401.10774

**Automated kernel generation**
- R54 ✓ Ouyang et al. *KernelBench.* ICML 2025. https://arxiv.org/abs/2502.10517
- R55 ✓ METR. *Measuring Automated Kernel Engineering.* 2025. https://metr.org/blog/2025-02-14-measuring-automated-kernel-engineering/
- R56 ✓ TechCrunch. *Sakana walks back claims that its AI can dramatically speed up model training.* 2025-02-21. https://techcrunch.com/2025/02/21/sakana-walks-back-claims-that-its-ai-can-dramatically-speed-up-model-training/
- R57 ✓ Lange et al. *Towards Robust Agentic CUDA Kernel Benchmarking, Verification, and Optimization* (robust-kbench). https://arxiv.org/abs/2509.14279
- R58 ✓(secondary) Ouyang, Liang, Mirhoseini. *Surprisingly Fast AI-Generated Kernels We Didn't Mean to Publish (Yet).* 2025. https://scalingintelligence.stanford.edu/blogs/fastkernels/
- R59 ✓ Baronio et al. *Kevin: Multi-Turn RL for Generating CUDA Kernels.* https://arxiv.org/abs/2507.11948
- R60 ✓(secondary) Google DeepMind. *AlphaEvolve* (May 2025). https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/ ; coverage: https://the-decoder.com/alphaevolve-is-google-deepminds-new-ai-system-that-autonomously-creates-better-algorithms/
- R61 ✓ Zhang et al. *KernelBench-Verified: Do LLM-Generated Kernels Actually Beat PyTorch?* 2026. https://arxiv.org/abs/2607.16241
- R62 ✓(secondary) *Kernel Contracts: A Specification Language for ML Kernel Correctness Across Heterogeneous Silicon.* 2026. https://arxiv.org/abs/2604.22032

**Our own files (local, read-only)**
- L1 `[...]/REPORT.md`: 8-GPU decode round profile (2026-11-03).
- L2 `[...]/REPORT.md`: upstream tok/s levers and the live-prefix index reduce.
- L3 `[...]/LANES.md`: entries cited by UTC timestamp (2026-11-03 unless a date is given).
- L4 `[...]/gguf_dmv13_x.cuh` and `.../benchmarks/deepseek_v41/v41k/v41k.cu`.
