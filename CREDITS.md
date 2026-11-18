# Credits

Most of what makes a V100 serve a 2026 model is other people's work. This page says whose, by public handle, with a
link to the work itself. Authors and states of pull requests were read from GitHub with `gh pr view` on 2026-11-18.
If we have missed or misdescribed a contribution, please open an issue and we will correct it.

## The model and its weights

| Who | What | Licence |
|---|---|---|
| **DeepSeek** | [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017): the model, its official MXFP4 experts, the DSpark speculative drafter, the Engram design | MIT |
| **NVIDIA** | [DeepSeek-V4.1-Flash-1017-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4): the NVFP4 pack whose experts the NVFP4 builds serve, and its accuracy table | MIT |
| **ISTA-DASLab** | [DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF): the 2-bit quantization Volta sPLIt serves, and the non-expert tensors of every build | MIT |
| **ggml** | [ggml](https://github.com/ggml-org/ggml): the GGUF container and the IQ quantization formats | MIT |
| **Phil Tompkins** (`philbert440`) | The [EXL3 pack](https://huggingface.co/philbert440/DeepSeek-V4.1-Flash-1017-EXL3-franken-v100) and [engine](https://github.com/philbert440/deepseek-v4.1-v100) our first eight-card build ran on | see the repositories |

## The engine: 1Cat-vLLM

[1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) (Apache-2.0) is the vLLM fork that keeps Volta and Turing
cards serving current models. Every build here runs on it. The work of its authors that these builds use directly:

| Pull request or issue | Author | State (2026-11-18) | What it is | How these builds use it |
|---|---|---|---|---|
| [#1277](https://github.com/1CatAI/1Cat-vLLM/pull/1277) | yangzhuxinyzx | merged 2026-10-27 | DeepSeek-V4.1-Flash on four V100s: 512K packed key-value storage, faster prompt reading, the model's kernels | **The base of every build here.** Volta sPLIt is its port to two boards |
| [#1482](https://github.com/1CatAI/1Cat-vLLM/pull/1482) | yangzhuxinyzx | merged 2026-11-11 | The official checkpoint in four stages of four cards on sixteen V100s, with batched decode | Not in our engine. The upstream reference for several requests per step |
| [#274](https://github.com/1CatAI/1Cat-vLLM/pull/274), [#308](https://github.com/1CatAI/1Cat-vLLM/pull/308), [#344](https://github.com/1CatAI/1Cat-vLLM/pull/344) | yangzhuxinyzx | merged 2026-08 | The gated DSpark verifier; decode hot paths for two stages of four cards; quality gates on the fast paths (DeepSeek-V4) | The speculative decoding path and the two-stage precedent |
| [#183](https://github.com/1CatAI/1Cat-vLLM/pull/183) | yangzhuxinyzx | merged 2026-08-03 | Probabilistic DSpark drafting | Tested here as a variant; lost |
| [#1292](https://github.com/1CatAI/1Cat-vLLM/pull/1292) | yangzhuxinyzx | merged 2026-10-28 | The fused four-card push all-reduce | Our all-reduce change admits this model's message size to it |
| [#742](https://github.com/1CatAI/1Cat-vLLM/pull/742) | Peuqui | merged 2026-10-03 | The skinny QPN expert backend for NVFP4 and MXFP4 | Every NVFP4 expert in the NVFP4 builds runs through it |
| [#1263](https://github.com/1CatAI/1Cat-vLLM/pull/1263) | Peuqui | merged 2026-10-28 | Do not resend a request's token history every decode step | Our keep-batch lever is its port to the old runner (-24.8 % of a round at 128K) |
| [#1221](https://github.com/1CatAI/1Cat-vLLM/pull/1221) | Peuqui | open | The DSpark speculator for Model Runner V2 | The path our Model Runner V2 port builds on |
| [#1280](https://github.com/1CatAI/1Cat-vLLM/pull/1280), [#1282](https://github.com/1CatAI/1Cat-vLLM/pull/1282), [#1347](https://github.com/1CatAI/1Cat-vLLM/pull/1347) | Peuqui | closed; open; open | Long-prefill index and sparse attention work; tensor-core Triton prefill routes; a fused decode index scorer | Ideas behind our fused prefill index scorer |
| [#1362](https://github.com/1CatAI/1Cat-vLLM/pull/1362) | Peuqui | merged 2026-11-16 | Skip the embedding and head on pipeline stages that do not hold them | The precedent for the twelve-card build's middle-stage memory change |
| [#874](https://github.com/1CatAI/1Cat-vLLM/pull/874), [#1222](https://github.com/1CatAI/1Cat-vLLM/pull/1222), [#1273](https://github.com/1CatAI/1Cat-vLLM/pull/1273), [#1294](https://github.com/1CatAI/1Cat-vLLM/pull/1294), [#1218](https://github.com/1CatAI/1Cat-vLLM/pull/1218) | Peuqui | merged | Index, grouped GEMM, send overlap and draft-vocabulary work on the same path | Read before every plan. The Volta sPLIt patch series carries 1Cat's main branch and then-open pull requests by their authors, unchanged |
| [#1243](https://github.com/1CatAI/1Cat-vLLM/pull/1243) | rivetphilbot | open | DeepSeek-V4.1-Flash on eight V100s with EXL3 experts, CUDA graphs, DSpark and 1M context | The engine of our first eight-card build |
| [#1157](https://github.com/1CatAI/1Cat-vLLM/pull/1157) | kezboard233 | merged 2026-11-10 | A native block-scale FP8 GEMM for 17 to 32 rows | Covers the row range where our FP8 dense kernel loses |
| [#1325](https://github.com/1CatAI/1Cat-vLLM/pull/1325), [#1537](https://github.com/1CatAI/1Cat-vLLM/pull/1537), [issue #1341](https://github.com/1CatAI/1Cat-vLLM/issues/1341) | squibloads | merged; open; closed | Row staging for large prefill batches; a pipeline speculation fix; the bug report on stale rows after a cache hit | Issue #1341 is the bug our hot-ring refill fixes |
| [issue #852](https://github.com/1CatAI/1Cat-vLLM/issues/852) | zambalee | open | Turning off asynchronous scheduling gives +55 % single-stream decode on a two-stage V100 build | Why every build here schedules synchronously |

Peuqui also maintains the [Triton fork](https://github.com/Peuqui/triton) those prefill routes need.

## The kernels: v100-skinny

**Dennis Vertzyas** (`dnv2003`), [v100-skinny](https://github.com/dnv2003/v100-skinny) (MIT for kernels and
documents, Apache-2.0 for its fork patches): hand-written NVFP4 and FP8 kernels for Volta's `mma.sync.m8n8k4`, and
the measurements this whole site leans on: the 879 GB/s read ceiling, 77 % of it at one row for NVFP4, the
serving traps. His kernels reach these builds through 1Cat-vLLM #742. No v100-skinny source file is copied into
this repository. [ProprietaryLegal/v100-skinny](https://github.com/ProprietaryLegal/v100-skinny) is a mirror, not
our work.

## Ideas, tools and methods

| Who | What | Where it shows up |
|---|---|---|
| **The vLLM project** | [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0): the serving engine underneath, and Model Runner V2 | everywhere |
| **Elevatormusic** and the TensorFold authors | [TensorFold #25](https://github.com/Elevatormusic/TensorFold/pull/25): coupled sampled drafts | +14 % tok/s at temperature 1 on Volta sPLIt |
| **0rand** | [deepseek-v41-tensorfold-spark](https://github.com/0rand/deepseek-v41-tensorfold-spark): the same idea in a DGX Spark recipe, and expert top-p pruning | the pruning was tested here and lost |
| **sudoingX** | The `p4_kld.py` fidelity method and its general panel; the [two-Spark launcher](https://github.com/sudoingX/dgx-spark-deepseek-v41) | Volta sPLIt's fidelity figures |
| **Baseten** | [deepseek-ultrafast](https://github.com/basetenlabs/deepseek-ultrafast) | read and ported privately; the twelve-card build's conversation store comes out of that work |
| **Leonccaa** | [ReVolta](https://github.com/Leonccaa/ReVolta), a V100 inference server | reviewed for ideas |
| **mw00**, **lxe** | [project-maya](https://github.com/mw00/project-maya), [dsx](https://github.com/lxe/dsx) | read as research leads |
| **0xSero** | the hill-climbing scripts at <https://github.com/0xsero> | the builder skill's optimization loop is borrowed from them |
| **OpenMMLab** | TurboMind's sm_70 GEMM code, carried by 1Cat-vLLM | the operand decoders our prefill expert kernel keeps |
| **NVIDIA** | The Volta, Ampere, Hopper and Blackwell tuning guides; compute-sanitizer | the kernel rules |

## What PLI Labs sent upstream

Pull requests to 1Cat-vLLM from PLI Labs' two accounts, read on 2026-11-18.

| Pull request | State | What it does |
|---|---|---|
| [#651](https://github.com/1CatAI/1Cat-vLLM/pull/651) | merged 2026-09-26 | Accept a drafter embedding replica across split weight loads |
| [#1021](https://github.com/1CatAI/1Cat-vLLM/pull/1021) | merged 2026-11-16 | Keep block-FP8 dense weights in FP8 on Volta |
| [#1030](https://github.com/1CatAI/1Cat-vLLM/pull/1030) | open | Generalise the MXFP4 expert geometry; split sizes 4, 6, 9, 12, 18 |
| [#1031](https://github.com/1CatAI/1Cat-vLLM/pull/1031) | open | DeepSeek-V4.1-Flash on V100: three stages of four on twelve cards, two on eight |
| [#1370](https://github.com/1CatAI/1Cat-vLLM/pull/1370) | open | Two stages of four for the #1277 package (the Volta sPLIt port) |
| [#1371](https://github.com/1CatAI/1Cat-vLLM/pull/1371) | open | Preserve synchronous speculative step ownership in a pipeline |
| [#1372](https://github.com/1CatAI/1Cat-vLLM/pull/1372) | merged 2026-11-16 | Tile the live decode index scoring |
| [#1373](https://github.com/1CatAI/1Cat-vLLM/pull/1373) | merged 2026-11-10 | Retain persistent request rows across empty pipeline steps |
| [#1374](https://github.com/1CatAI/1Cat-vLLM/pull/1374) | open | Shard the drafter's main projection |
| [#1375](https://github.com/1CatAI/1Cat-vLLM/pull/1375) | merged 2026-11-16 | Coupled sampled drafts with keyed verification |
| [#1376](https://github.com/1CatAI/1Cat-vLLM/pull/1376) | open | Skip layers 21 to 39 in early prefill chunks |
| [#1377](https://github.com/1CatAI/1Cat-vLLM/pull/1377) | open | Coupled drafts and keyed verification on vocabulary shards |
| [#1379](https://github.com/1CatAI/1Cat-vLLM/pull/1379) | open | DeepSeek-V4.1 and its drafter on Model Runner V2 |
| [#1380](https://github.com/1CatAI/1Cat-vLLM/pull/1380) | merged 2026-11-09 | README: the eight-card result |
| [#1381](https://github.com/1CatAI/1Cat-vLLM/pull/1381) | merged 2026-11-10 | Restore packed key-value bytes after a cache hit; reserve the speculative rewind span |
| [#1382](https://github.com/1CatAI/1Cat-vLLM/pull/1382) | merged 2026-11-10 | Admit small aligned FP16 messages to the four-card push all-reduce |
| [#1383](https://github.com/1CatAI/1Cat-vLLM/pull/1383) | merged 2026-11-16 | Build the routed-expert plan inside the quantize launch |
| [#1384](https://github.com/1CatAI/1Cat-vLLM/pull/1384) | merged 2026-11-16 | Fast routed-expert decode kernels |
| [#1385](https://github.com/1CatAI/1Cat-vLLM/pull/1385) | merged 2026-11-16 | Instruction-diet routed-expert kernels |
| [#1386](https://github.com/1CatAI/1Cat-vLLM/pull/1386) | merged 2026-11-16 | Reduce decode index scores over the live prefix |
| [#1387](https://github.com/1CatAI/1Cat-vLLM/pull/1387) | merged 2026-11-16 | Grouped prefill GEMM for the compressed routed experts |
| [#1388](https://github.com/1CatAI/1Cat-vLLM/pull/1388) | open | Fused prefill index scores |
| [#1389](https://github.com/1CatAI/1Cat-vLLM/pull/1389) | open | Fuse the attention prep and post kernels |
| [#1390](https://github.com/1CatAI/1Cat-vLLM/pull/1390) | merged 2026-11-16 | Dense mat-vec on a double-buffered kernel |
| [#1488](https://github.com/1CatAI/1Cat-vLLM/pull/1488) | merged 2026-11-16 | README: NVFP4 hot experts on eight V100s |

Two earlier pull requests ([#96](https://github.com/1CatAI/1Cat-vLLM/pull/96), [#100](https://github.com/1CatAI/1Cat-vLLM/pull/100),
July 2026) were closed without merging. The maintainers corrected some of the merged ones before merging (#1381's
description lists their corrections); the merged code is theirs as well as ours.

The llama.cpp fix for DeepSeek-V4 with a quantized key cache is reported upstream as
[llama.cpp #25382](https://github.com/ggml-org/llama.cpp/issues/25382); the patches are in
[deepseek-v4-flash-v100](https://github.com/ProprietaryLegal/deepseek-v4-flash-v100/tree/main/patches).

## Press

Tom's Hardware, 2026-11-09, Shane Downing:
[Lawyer builds private AI server with eight used Nvidia V100 GPUs](https://www.tomshardware.com/tech-industry/artificial-intelligence/lawyer-builds-private-ai-server-with-eight-used-nvidia-v100-gpus-up-to-215-tokens-per-second-to-keep-legal-files-offline-costing-about-usd2-400-per-four-gpu-board).
