# Sources

Every external source this repository relies on, with its link and what it is used for. Pull requests and issues
of 1Cat-vLLM are listed with their authors in [CREDITS.md](CREDITS.md) and are not repeated here. The research
notes under [research/](research/README.md) carry their own reference lists, the longest in
[the kernel survey](research/kernels/KERNEL-RESEARCH-REPORT.md).

## PLI Labs' own repositories and packages (the records behind the MEASURED figures)

| Source | Used for |
|---|---|
| [Volta sPLIt](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4): README, [docs/levers.md](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/blob/main/docs/levers.md), [docs/method.md](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/blob/main/docs/method.md), [docs/four-card.md](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/blob/main/docs/four-card.md), [engine/production](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/tree/main/engine/production) | Every Volta sPLIt figure; the A/B method; the patches |
| [Volta sPLIt on Hugging Face](https://huggingface.co/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-GGUF) | The build's card |
| [Volta Max uPLIft](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4): README, [docs/measurements.md](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4/blob/main/docs/measurements.md), [docs/history.md](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4/blob/main/docs/history.md), [docs/acceptance](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4/tree/main/docs/acceptance), [the closeness test's preregistration](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4/blob/main/quality/closeness-test/PREREGISTRATION.txt) | Every Volta Max uPLIft figure; the window method; the closeness judge |
| [Volta Max uPLIft package on Hugging Face](https://huggingface.co/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-hot_iq-8xV100-SXM2-PP2xTP4) | The build as one download |
| [Volta Full NVFP4](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100) | Every Volta Full NVFP4 figure; the exact fold; the cost of host-RAM experts |
| [deepseek-v4-flash-v100](https://github.com/ProprietaryLegal/deepseek-v4-flash-v100) | The July 2026 llama.cpp build; the quantized key-cache bug; the interim EXL3 results |
| [swift-1.5-v100](https://github.com/ProprietaryLegal/swift-1.5-v100) | A model that fits one board |
| [deepseek-v4.1-v100](https://github.com/ProprietaryLegal/deepseek-v4.1-v100) | Our fork of Phil Tompkins' engine snapshot |
| [pli-labs, research/v100](https://github.com/ProprietaryLegal/pli-labs/tree/main/research/v100) and its [July 2026 case study](https://github.com/ProprietaryLegal/pli-labs/blob/main/research/v100/serving-optimization-2026-07/README.md) | The earlier V100 notes |
| Hugging Face [deepseek-v4.1-flash-sm70](https://huggingface.co/ProprietaryLegal/deepseek-v4.1-flash-sm70), [sm70-fp8-block32-kernels](https://huggingface.co/ProprietaryLegal/sm70-fp8-block32-kernels) | The record of the first twelve-card port and its FP8 kernels |
| [DGX sPLIce](https://github.com/ProprietaryLegal/DGX-sPLIce-DeepSeek-V4.1-Flash-2xDGX-Spark-2xRTX3090-drafter-x86-offload), [dsv41-window-harness](https://github.com/ProprietaryLegal/dsv41-window-harness) | The sister build on DGX Sparks and its fenced-window harness |
| The private working record of Volta 12X AmPLIfy | Every twelve-card figure. Not yet published; this repository says so wherever it is used |

## Engines and kernels

| Source | Used for |
|---|---|
| [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) (Apache-2.0) | The engine of every build. Pull requests: [CREDITS.md](CREDITS.md) |
| [dnv2003/v100-skinny](https://github.com/dnv2003/v100-skinny) (MIT; fork patches Apache-2.0) | The NVFP4 and FP8 kernels for Volta; the measured read ceiling of 879 GB/s; kernel throughput by row count; the comparison with an RTX 5090 |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) (Apache-2.0) | The serving engine underneath; Model Runner V2 |
| [philbert440/deepseek-v4.1-v100](https://github.com/philbert440/deepseek-v4.1-v100) and the [EXL3 pack](https://huggingface.co/philbert440/DeepSeek-V4.1-Flash-1017-EXL3-franken-v100) | The engine and weights of our first eight-card build; the design reference for the prefill skip |
| [Elevatormusic/TensorFold #25](https://github.com/Elevatormusic/TensorFold/pull/25) | Coupled sampled drafts |
| [0rand/deepseek-v41-tensorfold-spark](https://github.com/0rand/deepseek-v41-tensorfold-spark) | The same idea in a DGX Spark recipe; expert top-p pruning (tested here, lost) |
| [sudoingX/dgx-spark-deepseek-v41](https://github.com/sudoingX/dgx-spark-deepseek-v41) | The two-Spark launcher. sudoingX also wrote the `p4_kld.py` fidelity method behind Volta sPLIt's next-token agreement figures |
| [Peuqui/triton](https://github.com/Peuqui/triton) | The Triton fork behind 1Cat's tensor-core prefill routes |
| [ggml](https://github.com/ggml-org/ggml); [llama.cpp issue #25382](https://github.com/ggml-org/llama.cpp/issues/25382) | The GGUF and IQ formats; the quantized key-cache bug report |

## Models and weights

| Source | Used for |
|---|---|
| [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (MIT) | The model, its drafter, config and tokenizer |
| [nvidia/DeepSeek-V4.1-Flash-1017-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4) (MIT) | The NVFP4 experts; NVIDIA's accuracy table (UPSTREAM) |
| [ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) (MIT) | The 2-bit experts of Volta sPLIt; the non-expert tensors and Engram tables of every build |

## Vendor documents

| Source | Used for |
|---|---|
| NVIDIA, [Tesla V100 GPU architecture whitepaper](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf) | 80 SMs, 900 GB/s HBM2, about 125 TFLOPS FP16 tensor throughput, the 6 MB L2 cache |
| NVIDIA, [Volta tuning guide](https://docs.nvidia.com/cuda/volta-tuning-guide/index.html) | What `sm_70` has: `mma.sync.m8n8k4`, shared memory and register limits |
| NVIDIA, [Ampere](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html), [Hopper](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) and [Blackwell](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) tuning guides | What each later generation added |
| NVIDIA, [DGX Spark hardware overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html) | 128 GB of unified memory at 273 GB/s |
| NVIDIA, [CUDA features archive](https://docs.nvidia.com/cuda/cuda-features-archive/index.html) | CUDA 13.0 removed offline compilation for Volta |
| PyTorch, [dropping Volta from the CUDA 12.8 binaries](https://dev-discuss.pytorch.org/t/dropping-volta-support-from-cuda-12-8-binaries-for-release-2-11/3290) | Which PyTorch builds still support `sm_70` |

## Writing about the hardware

| Source | Used for |
|---|---|
| The founder's post to r/homelab, 2026-11-05: <https://www.reddit.com/r/homelab/comments/1xoumr9/> | The cost of the cards and boards; the hardware faults; the driver branch. Quoted in part 8 of the front page |
| Tom's Hardware, 2026-11-09: [Lawyer builds private AI server with eight used Nvidia V100 GPUs](https://www.tomshardware.com/tech-industry/artificial-intelligence/lawyer-builds-private-ai-server-with-eight-used-nvidia-v100-gpus-up-to-215-tokens-per-second-to-keep-legal-files-offline-costing-about-usd2-400-per-four-gpu-board) | Independent coverage of the eight-card build |
| TechCrunch, 2025-02-21: [Sakana walks back claims that its AI can dramatically speed up model training](https://techcrunch.com/2025/02/21/sakana-walks-back-claims-that-its-ai-can-dramatically-speed-up-model-training/) | Why AI-written kernels need adversarial tests |

## The skill

[PLI Volta Builder](https://github.com/ProprietaryLegal/pli-volta-builder): the method of this repository packaged
for coding agents.
