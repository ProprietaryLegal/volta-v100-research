# GPU Kernels, in Plain Terms

> **Origin.** PLI Labs research archive, file `kernel-research/KERNEL-RESEARCH-EXPLAINER.md`. Published 2026-11-18 in the Volta V100 research
> repository.
> **Edited for publication.** Machine names, private file paths, network addresses, session names and the
> operator's name were replaced with generic terms (`the V100 server`, `[private path]`, `[address]`,
> `the owner`). Links to files that are not published were turned into plain text. No figure and no finding
> was changed. Dates inside the note are its own; later results are in the build pages of this repository.

*For the owner, 2026-11-02. A companion to the technical report (KERNEL-RESEARCH/REPORT.md), which carries the sources and the detail.*

---

## The one-paragraph version

A **kernel** is a small program that runs on the graphics card (the **GPU**), copied across thousands of workers at once. Running a language model means running a long chain of these kernels for every word it writes. On our V100 machine, the V100 server, that is about 2,000 kernels for every three or so **tokens** (a token is a word or a piece of a word). A **custom kernel** is one written for a specific job instead of taken from NVIDIA's standard library. People write them because the standard ones are built for big, regular jobs, and generating text one word at a time is neither. The research says the speed limit for text generation is almost never the GPU's arithmetic. It is how fast data can be hauled out of memory, and how much time is lost between kernels. Good kernel work therefore means moving fewer bytes, keeping the memory pipe full, and cutting the number of handoffs, then proving each change is correct and faster with honest measurement.

---

## 1. What a kernel is

Think of the GPU as a factory floor with 80 workstations. NVIDIA calls each workstation a **streaming multiprocessor (SM)**. Each station has dozens of workers who all do the same step at the same moment on different pieces of material. The CPU, the computer's ordinary processor, is the floor manager. It does not do the heavy work itself. It hands out instruction sheets.

A **kernel** is one of those instruction sheets. When the manager hands it out (a **launch**), every station gets a copy, and thousands of workers (**threads**) carry it out in parallel. Threads move in squads of 32, called **warps**. Squads are grouped into teams that share a workbench, called **blocks**, and all the blocks for one job make up the **grid**. You will see these words in our lane reports. They describe how a kernel divides its work.

Each handoff costs a few millionths of a second (microseconds). On a V100, NVIDIA measured a tiny kernel that took 2.9 microseconds to run but about 3.4 to 9.6 microseconds per kernel once the launch cost was included, depending on how the launches were issued. One kernel's overhead is trivial. Two thousand kernels per round adds up to milliseconds.

### Where the data lives

The factory has storage at several distances:

- **Registers**: in each worker's hands. Instant, but tiny.
- **Shared memory**: a small workbench each team shares. Very fast, a few dozen kilobytes.
- **L2 cache**: a chip-wide shelf. On a V100 it holds only 6 megabytes.
- **HBM (main GPU memory)**: the warehouse. 32 gigabytes per V100, but every trip takes hundreds of clock cycles, and the road from the warehouse carries at most about 900 gigabytes per second.

Nearly everything in kernel engineering is about using the near storage well and making the warehouse trips count.

### How a model becomes a chain of kernels

To produce one token, the model passes the current text through dozens of identical **layers**. Each layer runs a fixed recipe of kernels: normalize the numbers, multiply by large weight tables, run **attention** (the step where the model looks back over earlier words), multiply again, and add the results together. When the model is split across several GPUs, the GPUs also swap partial results. At the end, the model scores every word in its vocabulary and picks one (**sampling**).

There are two phases:

- **Prefill** reads your whole prompt at once. That is lots of arithmetic on lots of words together, which GPUs do well.
- **Decode** writes the answer one word at a time. For every single word, the GPU must haul the model's entire set of active weights out of the warehouse, use each number only once or a few times, and then do it all again for the next word.

Decode is the phase we spend our effort on. It is limited by the road, not by the workers.

---

## 2. What a custom kernel is, and why people write them

NVIDIA ships well-tuned standard kernels (libraries called cuBLAS and cuDNN). They are like factory-standard machine tools: excellent for big, common jobs. A **custom kernel** is a purpose-built jig for one specific job. People build them for six reasons.

1. **The job is an odd shape.** Generating one word at a time means multiplying a huge table by a single short list of numbers. Standard tools are tuned for big blocks of work and leave most of the factory idle on a job that small.

2. **Combining steps saves trips.** Every separate kernel writes its result back to the warehouse, and the next kernel fetches it again. Merging several steps into one kernel (**fusion**) skips those round trips and the handoffs. The most famous example, **FlashAttention**, made attention several times faster mainly by never writing its large intermediate table to main memory.

3. **Compressed weights need special handling.** We store the model's weights compressed, at about 2 bits per number instead of 16 (**quantization**). That cuts warehouse traffic roughly eightfold, but only if the kernel unpacks the numbers right at the workstation (**dequantization**) instead of writing an unpacked copy back to the warehouse. No standard library understands our particular compression formats.

4. **New model designs arrive first.** DeepSeek's models use a **mixture of experts** (only a few of hundreds of sub-networks run for each word), a compressed form of attention, and a "lightning indexer" that decides which earlier words are worth looking at. The kernels for these ship with the model, or not at all, long before standard libraries catch up.

5. **New hardware features are reachable only by hand at first.** Each GPU generation adds new instructions, and only hand-written kernels use them at first. The reverse also bites us: new libraries drop old hardware, so on our 2017-era V100s, we write our own or go without.

6. **Communication between GPUs.** When eight GPUs share a model, they constantly swap small pieces of data. Custom communication kernels can do the small, urgent swaps faster than general-purpose ones.

The price is real. Custom kernels must be maintained. They can be subtly wrong. They tie you to specific hardware. They can change the model's arithmetic in tiny ways that occasionally change a word of output. That is why every one of ours goes in behind an off switch with a test harness around it.

---

## 3. What the research says about building them well

### Know which wall you are hitting

The most useful single idea is the **roofline model**. Every kernel is limited either by arithmetic speed (**compute-bound**) or by memory speed (**memory-bound**). Which one depends on **arithmetic intensity**: how much math the kernel does for each byte it hauls from the warehouse.

A V100's math units (its **tensor cores**, specialized matrix-multiplying hardware) need about 139 operations per byte delivered to stay busy. Decoding one word with uncompressed weights does about 1 operation per byte. Even with our 2-bit compression it is roughly 8 per byte per word checked. Either way, the workers wait on the road almost all the time. So for decode, the right scorecard is "what share of the road's capacity did we use", not "how much math per second".

One subtlety matters a lot for us. Unpacking very compressed formats takes real work per number. Past a certain point, squeezing to fewer bits stops helping, because the workers become the bottleneck while they unpack. One of our lanes found exactly that with a 2-to-5-bit format on the V100: the unpacking cost per weight was almost the same no matter how many bits it used.

### Move fewer bytes

Use compression. Fuse steps so intermediate results never touch the warehouse. Load each piece of data once and reuse it as much as possible while it is on the workbench (**tiling**). Read memory in neat, consecutive stretches so the hardware can bundle the requests (**coalescing**).

### Keep the pipe full

Memory requests take a long time to come back, so a good kernel keeps many requests in flight at once, like a restaurant that keeps the next order cooking while serving the current one. Researchers call this **latency hiding** and **pipelining**. Newer GPUs have dedicated hardware for copying data in the background. The V100 does not, so on our machine the workers must do the copying themselves, which is part of why porting newer techniques is hard.

### Fill the whole factory

A one-word job can be too small to give all 80 workstations something to do. Standard tricks split the job along another direction so every station gets a slice, then add the slices together at the end: **split-K** for matrix work, **split-KV** for attention.

### Cut the handoffs

Every kernel launch costs time, and the gaps between kernels are dead time. **CUDA graphs** record a whole sequence of launches and replay it with one command, which removes most of the manager's paperwork. Fusion removes kernels entirely. The frontier idea is the **megakernel**: one giant kernel that runs an entire step of the model with no gaps. Stanford's Hazy Research group reported more than 1.5x faster single-user generation on a small model this way.

### Respect the arithmetic

Computers round every calculation slightly, and the result depends on the *order* in which numbers are added. A kernel that adds the same numbers in a different order gives a microscopically different answer. Usually nothing visible happens. Occasionally, when the model's top two word choices are nearly tied, the other word wins, and the text from that point on differs.

Research from Thinking Machines showed that this, not random hardware behaviour, is the main reason AI services give different answers to identical requests. Kernels pick different strategies depending on how busy the server is. Good practice is a three-tier test:

1. **Bit-for-bit identical** when the new kernel keeps the same addition order.
2. **Within a tight tolerance** when it doesn't.
3. In end-to-end tests, a changed word is accepted only if it happened at a **near-tie**.

### Measure honestly

Benchmarks mislead in predictable ways. The first few runs are slow while things warm up. Small test data sits in the fast cache when real data wouldn't. Profiling tools sometimes run the GPU at reduced clock speed by default. A kernel can look fast alone and slow in the full system, or the reverse.

The defence is to time many repetitions after a warm-up, flush the cache between runs, and treat the full end-to-end test as the verdict. Run test arms in an interleaved order (**ABBA**: baseline, change, change, baseline) so drift cancels out, and make sure nothing else is running on the machine while you test.

### Be careful with AI-written kernels

There is now a sizeable literature on AI models writing kernels. It shows real gains when the AI works in a loop (write, compile, test, time, revise) with strict checking. METR measured an average 1.8x speedup on a cleaned-up benchmark.

It also shows the AI "gaming the test". In February 2025, Sakana AI claimed 10-100x speedups and had to walk them back: its system had found a loophole in the evaluation code that let it skip the correctness check. A 2026 study (arXiv, July 2026) found that once loopholes and weak comparisons were removed, the best model's average 1.43x "speedup" became 0.88x, a slowdown. The lesson: the test harness is part of the attack surface, and it must check unseen input sizes and fresh output buffers.

---

## 4. Key terminology (about 25 terms)

- **Kernel**: a program the CPU launches onto the GPU, run by thousands of threads at once.
- **Host / device**: the CPU side / the GPU side.
- **Launch / launch overhead**: handing a kernel to the GPU / the fixed few-microsecond cost of each handoff.
- **Thread, warp, block, grid**: one worker; a squad of 32 moving in lockstep; a team sharing a workbench; all the teams on one job.
- **SM (streaming multiprocessor)**: one of the GPU's workstations. A V100 has 80.
- **Registers, shared memory, L2, HBM**: storage from nearest and smallest to farthest and largest.
- **Bandwidth**: how many bytes per second the memory road can deliver (about 900 GB/s on a V100).
- **Memory-bound / compute-bound / latency-bound**: limited by the road / by the math / by fixed delays and gaps.
- **Arithmetic intensity**: math operations per byte fetched. It tells you which wall you'll hit.
- **Roofline**: the chart and formula that turn arithmetic intensity into a speed limit.
- **Tensor core**: hardware specialized for multiplying small matrices very fast.
- **GEMM / GEMV**: matrix-times-matrix / matrix-times-vector, the core math of every layer. Decode is mostly GEMV-like.
- **Tiling**: working on blocks of data that fit on the workbench so each piece is reused.
- **Coalescing**: reading memory in neat consecutive runs the hardware can bundle.
- **Occupancy**: how many squads each workstation keeps active to cover waiting time.
- **Pipelining (double buffering)**: fetching the next batch of data while working on the current one.
- **Fusion**: merging several steps into one kernel to avoid trips and handoffs.
- **CUDA graph**: a recorded sequence of launches replayed with one command.
- **Split-K / split-KV**: dividing a small job along another direction so all workstations help.
- **Quantization / dequantization**: storing weights in fewer bits / unpacking them at the workstation.
- **Prefill / decode**: reading the prompt all at once / writing the answer word by word.
- **KV cache**: the model's stored notes about earlier words, re-read every decode step.
- **Attention / FlashAttention**: the step where the model looks back at earlier words / the landmark kernel that made it fast by never writing big temporary tables to memory.
- **Mixture of experts (MoE)**: a model where each word uses only a few of many sub-networks ("experts"). DeepSeek V4.1 uses 6 of 384.
- **All-reduce**: GPUs adding their partial results together and sharing the total.
- **Speculative decoding / verification**: a small fast model guesses several words ahead, and the big model checks them all in one pass.
- **Bitwise identical / near-tie**: exactly the same output bits / a changed word where the top two choices were nearly equal.
- **Microbenchmark vs end-to-end**: timing one kernel alone vs timing the whole system generating text.
- **Autotuning**: automatically trying kernel settings to find the fastest.
- **Reward hacking**: an automated optimizer gaming its test instead of solving the problem.

---

## 5. The types of kernels

Most of a model's work falls into a dozen families:

- **Matrix multiplication (GEMM).** The big multiplications in every layer. Compute-bound when processing a whole prompt, memory-bound when generating word by word.
- **Matrix-vector and quantized multiplication (GEMV).** The decode version, often reading compressed weights. Limited by the memory road and, for very compressed formats, by unpacking work.
- **Grouped / expert kernels (MoE).** Many small multiplications, one per chosen expert. Limited by how many *different* experts the current words need, and by uneven workloads.
- **Attention.** Reads the stored notes on every earlier word. Limited by memory at long context. The FlashAttention family, Flash-Decoding and PagedAttention live here.
- **Normalization.** Rescales the numbers between steps. Cheap, and the best kernel for it is one that has been fused into its neighbour.
- **Elementwise and activation.** Simple per-number operations: add, multiply, apply a curve, rotate positions (RoPE). They exist mainly to be fused away.
- **Reductions, softmax and top-k.** Find the largest values, add things up, turn scores into probabilities. Limited by latency and by the need for a careful addition order.
- **Quantize / dequantize.** Pack and unpack numbers. Best done inside the kernels that produce or consume them.
- **Routing, dispatch and combine.** Decide which experts each word uses, sort words to experts, then merge the answers. Tiny jobs, so limited by latency.
- **Embedding and LM head.** Look up each word's starting numbers / score every word in the vocabulary at the end. Big reads, and the scoring is often split across GPUs.
- **Sampling.** Pick the next word. The fastest methods avoid sorting the whole vocabulary.
- **Communication.** Move and combine data between GPUs. Small messages are latency-bound, large ones bandwidth-bound.
- **KV-cache management.** Store and organize the notes on earlier words.
- **Speculative drafting and verification.** The guess-and-check machinery: many tiny steps, so overhead dominates.

---

## 6. How this applies to our project

**Where the time goes.** On the V100 server, one decode round takes about 31-32 milliseconds and produces about three tokens, roughly 100 tokens per second. The round splits into the first group of four GPUs (about 13.6 ms), the second group (about 13.1 ms), and a gap of about 5.6 ms that is mostly the guessing model's work plus about 2 ms of ordinary bookkeeping.

**What the measurements tell us:**

- **The experts are already near the speed limit.** The expert kernels take about 8.3 ms per round and run at roughly 57 % of the theoretical memory limit for our compression format. That figure is an estimate, not a measurement. The remaining gap there is mostly unpacking work, not data volume.
- **Handoffs cost real time.** There are about 2,000 kernels per round, and the gaps between them cost about 2 ms on each group of four GPUs. Fusion and, eventually, megakernel-style designs are the cure.
- **The biggest recent win came from not doing work.** The model was combining a full-size scoring table across GPUs at every step, even when only a small part of it was in use. Combining only the live part saved about 0.85 ms per round, roughly 3 % (first reported as 1.3 ms; a later review found the comparison run was unusually slow).
- **Small isolated tests can mislead in both directions.** One new kernel's solo benchmark predicted a 0.03 ms saving, and the full test measured 0.45 ms. Another idea, written as about 50 small steps per layer, made every round 7 ms slower and was dropped. The full interleaved A/B test is the verdict.
- **Correctness is checked at the word level.** When an output word changes, we prove it happened at a near-tie by tracing the scores. That is what we did for the "coupled drafts" change, which made sampled generation (temperature 1, the more varied mode) about 14 % faster.

**The V100 is a sunset platform.** NVIDIA's CUDA 13 no longer compiles for it. PyTorch is phasing it out. Most of the famous modern kernels (FlashAttention 2, 3 and 4, Marlin, DeepSeek's own kernel libraries) never supported it. Our practice is to borrow the *ideas* and write our own V100 versions, and to keep the software versions pinned to ones that still support the card.

**The DGX Spark boxes are the opposite case.** They are new Blackwell-generation hardware, but their memory road is much narrower: about 273 GB/s per box, against 900 GB/s for *each* of the V100 server's eight V100s. They also lack some features of NVIDIA's datacenter Blackwell chips, so kernels built for those chips won't run on them as-is. On the Sparks, data volume per word matters even more, and the choice of compressed weight format is the main lever.

**Bottom line.** The research and our own numbers agree on the order of attack: first stop doing unnecessary work, then cut the number of kernels and the gaps between them, then cut the bytes. Prove each step with a correctness test and an honest end-to-end A/B.
