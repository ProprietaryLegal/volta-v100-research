# V-D: Community & Chinese-Source Knowledge — Multi-V100 Large-Model Serving

> **Origin.** PLI Labs research archive, file `v100-research/findings/V-D-community-china.md`. Published 2026-11-18 in the Volta V100 research
> repository.
> **Edited for publication.** Machine names, private file paths, network addresses, session names and the
> operator's name were replaced with generic terms (`the V100 server`, `[private path]`, `[address]`,
> `the owner`). Links to files that are not published were turned into plain text. No figure and no finding
> was changed. Dates inside the note are its own; later results are in the build pages of this repository.

**Research target:** `the V100 server` = a multi-board Tesla V100-SXM2-32GB server (sm_70). NVLink only
within 4-card boards; cross-board = PCIe Gen3 ×8. Goal: serve large models fast, and serve
BIGGER models spread across many V100s.

**Method:** Mine real-world community know-how (heavy emphasis on Chinese sources — China
runs the largest V100/V100-32GB fleets, so the deepest practical tuning is often in Chinese)
for exact configs, tok/s numbers, gotchas, and lesser-known sm_70 patches across vLLM,
llama.cpp, LMDeploy, SGLang, KTransformers, ik_llama, and quant kernels (AWQ/GPTQ Marlin).

**Status:** COMPLETE (first pass). Open gaps listed in §7.
**Date:** 2026-06-13.

---

## How to read this file

- Each finding tags **[CN]** (Chinese-language source) or **[EN]**.
- **Confidence:** High / Med / Low — based on corroboration + specificity of numbers.
- Every claim carries a URL. tok/s tables list model | stack | GPUs | quant | tok/s | source.
- "**MISSING FROM OUR STACK**" callouts flag things our current setup (vLLM AWQ-Marlin,
  llama.cpp kv-split, LMDeploy) does not yet exploit.

---

## TL;DR

See the full **TL;DR** block at the END of this file (§ after "7. Confidence & gaps") for the
7 headline findings + the recommended changes. Quick index: §1 = Chinese sources (incl. the
definitive liaoweiming SM70-dispatch deep-dive + the AWQ-Marlin 67-vs-741 gap), §2 = GitHub
issues/forks (1Cat-vLLM, ch2lab/vllm-v100, FP16-vs-Marlin bug, SGLang/KTransformers gating,
ik_llama TP), §3 = Reddit/aggregated benchmarks + the TP/PP/EP topology law, §4 = Volta
FlashAttention forks, §5 = consolidated tok/s tables + heuristics, §6 = what our stack is
missing (ranked), §7 = confidence & gaps.

---

## 1. Chinese sources (Zhihu / CSDN / juejin / bilibili / 公众号 / CN GitHub)

> NOTE on access: Zhihu (zhuanlan.zhihu.com) returns **HTTP 403**, CSDN blog returns
> **HTTP 521 / Cloudflare**, ascendai.csdn.net **404** to direct WebFetch — standard CN
> anti-scrape. Findings from these are reconstructed from search-result snippets +
> corroborating fetchable mirrors, and flagged **[CN, snippet-only]** with lower confidence
> where I could not read the full body.

### 1.1 vLLM on V100 — the core CN consensus (multiple Zhihu/CSDN posts)

**[CN] Tesla V100 + vLLM 0.10.2 踩坑记录 (Zhihu p/1956047359613044602)** — *snippet-only (403)*.
Title = "Tesla V100 install & use vLLM 0.10.2 — pitfall log." Confidence: Med (title +
corroboration; full body blocked).

**[CN] 8卡V100 + vLLM Qwen 部署实战 (ascendai.csdn.net)** — *snippet-only (404)*. Reported
finding: 8× V100 can tensor-parallel-serve Qwen; with **4× V100-16GB** TP, max non-quantized
model ≈ **14B params at ~30 tok/s**. V100 (sm 7.0) → **no bf16, must use fp16**. Confidence: Med.

**[CN→bilingual] vLLM Forum — "在V100显卡上，vLLM并发问题" (discuss.vllm.ai/t/1896)** — FETCHED.
This is the single most concrete V100 datapoint so far. Verbatim findings:
- Single request ≈ **15 tok/s**; 60 concurrent requests still ≈ 15 tok/s aggregate (NO
  throughput scaling) → **prefill blocks subsequent requests** on V100.
- **Attention backends on V100:** FlashAttention = NOT supported; FlexAttention = NOT
  supported; xformers = limited-version-compatible; fallback = **PyTorch SDPA**.
- Manual override: `VLLM_ATTENTION_BACKEND=XFORMERS` or `=TORCH_SDPA`.
- Quote: "TORCH_SDPA backend has best compatibility on V100, but slower speed."
- Root causes: prompt-prefill blocking; V100 mem/bandwidth limits (esp. multimodal Qwen3-VL).
  Same params on RTX 4090 → normal concurrency. Confidence: **High** (read in full).
- URL: https://discuss.vllm.ai/t/v100-vllm/1896

> **MISSING FROM OUR STACK? (flag)** The CN/forum consensus says on V100 vLLM you are stuck
> on XFORMERS or TORCH_SDPA (no FlashAttention). Our notes notes already capture
> TRITON_ATTN as the working V100 prefill backend (avoids the FLASH_ATTN_V100 96KB crash) —
> need to confirm whether the community's XFORMERS path or our TRITON_ATTN path is faster on
> sm_70. (See §2 GitHub for the Triton-attn angle.)

### 1.0 ⭐ THE definitive CN deep-dive: vLLM SM-architecture adaptation (liaoweiming.org)

**[CN] 廖维明 — "vLLM 对不同 GPU SM 架构的适配机制深度剖析"** — FETCHED IN FULL. This is the
single most useful Chinese source found: a code-level analysis of how vLLM dispatches kernels
by compute capability, with V100/SM70 called out explicitly. Confidence: **High**.
URL: https://www.liaoweiming.org/blog/vllm-sm-architecture-adaptation

**Complete SM70 feature-gating matrix in upstream vLLM (translated):**

| Component | SM70 (V100) | Note |
|---|---|---|
| FlashAttention (Dao) | ✗ | needs SM80+ |
| FlashInfer | ✗ | needs SM75+ |
| **Triton Attention** | ✓ | universal fallback — the V100 path |
| CUTLASS INT8 GEMM | ✗ | needs SM75+ |
| **cuBLAS FP16 GEMM** | ✓ | default dense path on V100 |
| **Marlin (AWQ/GPTQ)** | ✗ | needs SM75+ (no INT8 tensor core on SM70) |
| GPTQ (non-Marlin) | ✓ | but slow |
| BitsAndBytes | ✓ | minimum-capability path |
| BF16 | ✗ | FP16 only |
| native FP8 | ✗ | no hardware FP8 |

Root cause (translated): "**V100's SM70 has no INT8 Tensor Core → cannot use AWQ/GPTQ Marlin
kernel**; no BF16; no FP8 HW; no FlashAttention (needs Ampere)."

**How 1Cat-vLLM gets around it (translated, code-level):**
1. **SM70 AWQ via WMMA:** "use Volta's WMMA instructions (FP16 Tensor Core) to do weight-only
   INT4 dequantize + FP16 GEMM," integrating "lmdeploy TurboMind's SM70 WMMA kernel." (i.e.
   it does NOT need the missing INT8 tensor core — it dequantizes INT4→FP16 and runs FP16 MMA.)
2. **MLA prefill fallback:** "if cap[0] < 8 (i.e. SM70), use SDPA as the prefill backend."
3. **Recommended runtime config (VERBATIM from the blog — note: differs from 1Cat README!):**
   ```
   VLLM_DISABLE_PYNCCL=1 python -m vllm.entrypoints.openai.api_server \
       --attention-backend TRITON_ATTN \
       --disable-custom-all-reduce \
       --gpu-memory-utilization 0.80 \
       --tensor-parallel-size 4
   ```

> **⚠️ RECONCILIATION (important — two conflicting V100 configs in the wild):**
> - 1Cat's OWN README (§2.1) says: `--attention-backend FLASH_ATTN_V100`, and explicitly
>   "do NOT pass `--disable-custom-all-reduce` for 27B TP4 decode," gpu-util 0.88.
> - This CN blog says: `--attention-backend TRITON_ATTN`, `--disable-custom-all-reduce`,
>   `VLLM_DISABLE_PYNCCL=1`, gpu-util 0.80.
> - Our OWN memory says: TRITON_ATTN (to dodge the FLASH_ATTN_V100 96KB prefill crash).
> Two of three (our notes + the CN blog) favor **TRITON_ATTN + disable-custom-all-reduce**;
> 1Cat's README favors its newer FLASH_ATTN_V100 + paged-prefill. **Most likely truth:**
> FLASH_ATTN_V100 is newer and faster on long prompts but version-fragile (the 96KB crash);
> TRITON_ATTN + disable-custom-all-reduce + DISABLE_PYNCCL is the safe, broadly-recommended
> fallback. **Action: benchmark both on current 1Cat build; the custom all-reduce + pynccl
> flags interact with our cross-board PCIe Gen3 ×8 topology (see §6).**

### 1.05 ⭐ The AWQ-Marlin tok/s gap (why this all matters)

**[CN] devpress.csdn "vLLM + AWQ … 为什么 v100 默认不支持"** + corroborated across sources:
the headline number that justifies the entire V100-fork ecosystem —
- **AWQ WITHOUT Marlin ≈ 67 tok/s** vs **AWQ WITH Marlin ≈ 741 tok/s = 10.9× speedup.**
- Upstream vLLM gates Marlin to SM75+, so a *stock* vLLM on V100 either refuses AWQ or runs
  the ~67 tok/s slow dequant path. The forks restore the fast path with a **WMMA FP16** kernel
  (TurboMind-SM70) instead of the unavailable INT8-tensor-core Marlin kernel.
- (Numbers are illustrative of the kernel gap on a small model, not our exact models.)
  Confidence: Med-High (number appears in multiple CN sources; exact model/ctx unspecified).
- URL: https://devpress.csdn.net/v1/article/detail/161462608

### 1.2 LMDeploy / TurboMind on V100 (CN GitHub issue + CN docs)

**[CN] QwenLM/Qwen3 Issue #1023** — FETCHED. "On 4× V100-16GB, lmdeploy serving
Qwen2.5-32B-Instruct-GPTQ-Int4 tops out at 80 tok/s — is that normal?" Verbatim:
```
lmdeploy serve api_server ./Qwen2.5-32B-Instruct-GPTQ-Int4 --model-format gptq --tp 4 --quant-policy 8
```
- LMDeploy/TurboMind GPTQ-Int4, TP4, `--quant-policy 8` (= INT8 KV cache) → **~80 tok/s** single-stream.
- Same user on **vLLM** with the Int8 variant → peak **52.5 tok/s** (prompt 7.3, tail 14.6).
- **Takeaway: on V100, LMDeploy/TurboMind W4 ≈ 80 tok/s beat vLLM W8 ≈ 52 tok/s for a 32B-class model.**
- Issue closed "not planned" (no maintainer baseline). Confidence: **High** (read in full).
- URL: https://github.com/QwenLM/Qwen3/issues/1023

**[CN/EN] LMDeploy docs + search consensus:** LMDeploy advertises Volta (sm70) support;
TurboMind W4A16 (AWQ) is **~2.4× faster than FP16**; TurboMind supports **MXFP4 from V100
upward**. Caveat in docs: some W4 paths state a **sm80 minimum** while V100 is explicitly
called out as compatible for MXFP4 — i.e. read the kernel-support matrix carefully per release.
- URLs: https://github.com/InternLM/lmdeploy/blob/main/README_zh-CN.md ,
  https://github.com/AllentDan/lmdeploy/blob/main/docs/zh_cn/quantization/w4a16.md

> **MISSING FROM OUR STACK? (flag — important)** Our notes already use LMDeploy, but this CN
> issue is direct evidence that **LMDeploy TurboMind W4 (~80 tok/s) materially beats vLLM
> (~52 tok/s) on the SAME 32B-class model on V100.** If our LMDeploy lane isn't using
> `--quant-policy 8` (INT8 KV) it's leaving headroom on the table. Worth an A/B vs our
> 1Cat-vLLM AWQ 121 tok/s number (different model size — Qwen3.5-35B-A3B MoE is much cheaper
> per token than dense 32B, so not apples-to-apples).

---

## 2. GitHub issues/comments on serving repos, filtered to V100 / sm_70 / Volta

### 2.1 1Cat-vLLM — the V100 (SM70) vLLM fork (THE key resource) — FETCHED

`github.com/1CatAI/1Cat-vLLM` — "vLLM fork for Tesla V100 (SM70) with AWQ 4-bit, CUDA 12.8
build flow, validated Qwen3.5 27B/35B on multi-GPU V100." This is the same fork our notes
notes call "1Cat-vLLM" (the AWQ lane at 121 tok/s). The README has more than our notes captured:

**What it patches for sm_70:**
- **TurboMind-derived SM70 kernels** for AWQ inference (i.e. it ports LMDeploy's fast W4 GEMM into vLLM).
- A **V100 FlashAttention path** named `FLASH_ATTN_V100` — *recommended* backend (NOT xformers/SDPA).
  → This directly contradicts the broad CN claim "FlashAttention impossible on V100"; a Volta
  FA kernel exists and is the fork's default.
- **MTP (multi-token-prediction) speculative decoding** with SM70-safe defaults.
- Long-context default 256K (`--max-model-len 262144`).

**CUDA 12.8 build flow (verbatim):**
```bash
export CUDA_HOME=[...]/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="7.0"
export MAX_JOBS=12
export NVCC_THREADS=1
pushd flash-attention-v100
python -m build --wheel --no-isolation --outdir ../dist-cu128-sm70
popd
python -m build --wheel --no-isolation --outdir dist-cu128-sm70
```
Install prebuilt wheels:
```bash
python -m pip install --prefer-binary --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  ./flash_attn_v100-*.whl ./vllm-*.whl
```

**Validated launch configs (verbatim):**
```bash
# Qwen3.5-27B-AWQ, TP4
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen3.5-27B-AWQ \
  --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 262144 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 16384
```
- Qwen3.6-27B-AWQ (TP4 + MTP): MTP4 spec-decode, prefix cache, `mamba_cache_mode=align`, `max_num_seqs=4`, `max_num_batched_tokens=8192`.
- Qwen3.6-35B-A3B-AWQ (TP4 MoE): `max_num_batched_tokens=8192`.
- Qwen3.5-122B-A10B-AWQ (TP4): reduced prefill chunk budget, `max_num_batched_tokens=8096`.

**Gotchas (verbatim / paraphrased):**
- First request may spend **1–3 min compiling kernels** on V100 — not representative of steady state.
- Run from OUTSIDE the source checkout so installed wheels load (not the local tree).
- Do NOT pass `--disable-custom-all-reduce` for 27B TP4 decode.
- `VLLM_SM70_ENABLE_DENSE_F16_FASTPATH=1` is experimental — keep OFF for MoE.
- `VLLM_FLASH_V100_ENABLE_PAGED_PREFILL=1` forces a paged-prefill variant.
- Experimental FP8 KV: `--kv-cache-dtype fp8_e5m2` (not a stable default).
- Confidence: **High** (read in full).
- URL: https://github.com/1CatAI/1Cat-vLLM

> **MAJOR FLAG vs our stack:** Our notes describes the V100 vLLM lane via
> "FLASH_ATTN_V100 96KB prefill crash → use TRITON_ATTN" and "AWQ-Marlin." But THIS fork's
> author treats `FLASH_ATTN_V100` as the *working, recommended* backend, with paged-prefill
> + a 96KB-aware path (`VLLM_FLASH_V100_ENABLE_PAGED_PREFILL=1`). Two possibilities: (a) we
> hit the 96KB crash on an older build and the fork has since fixed it with the paged-prefill
> flag; (b) we're on a different fork/version. **Action: re-test `FLASH_ATTN_V100` +
> `VLLM_FLASH_V100_ENABLE_PAGED_PREFILL=1` on current 1Cat-vLLM before assuming TRITON_ATTN
> is the only option — the fork author gets a real Volta FlashAttention kernel working.**

### 2.2 FP16-vs-Marlin garbled-output bug on V100 (vLLM #34694) — directly relevant

`vllm-project/vllm#34694`: NVFP4 models give **garbled output with `--dtype bfloat16` on GPUs
without native FP4 (RTX 4090 SM89, V100 SM70)** — they fall back to the Marlin kernel, and
`dequant_fp8_scales<nv_bfloat162, kFE4M3fn>` in `csrc/quantization/marlin/dequant.h` widens
the exponent wrong (S0E5M3→BF16). **The FP16 path is UNAFFECTED** (5-bit exp maps directly to
FP16). Practical rule for us: **on V100, run FP4/FP8-derived Marlin paths in FP16, never BF16**
(V100 can't do bf16 anyway, but if an alias sneaks bf16 in, expect garble). Confidence: High.
- URL: https://github.com/vllm-project/vllm/issues/34694

### 2.3 GPTQ w4a16 slower than FP16 (vLLM #21266) — the "quant is slower" trap
`vllm-project/vllm#21266`: users see **GPTQ w4a16 SLOWER than FP16** — typically because the
GPTQ path isn't hitting Marlin (act-order/group-size/arch mismatch) and falls to a slow
dequant. Lesson for V100: a 4-bit model is only fast if it lands on the **Marlin** (or the
fork's TurboMind-SM70) kernel; verify the kernel actually selected, don't assume "4-bit = fast."
Confidence: Med-High.
- URL: https://github.com/vllm-project/vllm/issues/21266

### 2.5 vLLM PR #36026 — the SM70 MoE bugfix (Qwen3.5-35B-A3B on V100)
**[EN+CN] HF discussion Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 #5 (vllm SM70 V100 support)** — FETCHED.
User MaxwellLyu reports success running **Qwen3.5-35B-A3B-GPTQ-Int4 on 4× V100 32GB** by
(1) installing vLLM from source editable, (2) applying **PR #36026** bugfix, (3) launching:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000 \
vllm serve "/path/Qwen3.5-35B-A3B-GPTQ-Int4" \
  --host [address] --port 8018 \
  --served-model-name "Qwen3.5-35B-A3B-GPTQ-Int4" \
  --mm-encoder-attn-backend TORCH_SDPA \
  --max-model-len auto --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --gpu-memory-utilization 0.79 --tensor-parallel-size 4
```
- **`--mm-encoder-attn-backend TORCH_SDPA`** is called "very important for multimodal on V100"
  (vision encoder has no FA on sm70). Start gpu-util 0.5 then raise. Uses **GPTQ-Int4** (not AWQ).
- This shows a **stock-vLLM + one PR** path (no full fork) for a MoE model on V100 — useful if
  we ever can't use 1Cat. No tok/s given. Confidence: High (read in full).
- URLs: https://huggingface.co/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4/discussions/5 ,
  https://github.com/vllm-project/vllm/pull/36026

> **MISSING FROM OUR STACK? (flag)** If we serve any multimodal/VL model on V100 via vLLM,
> set `--mm-encoder-attn-backend TORCH_SDPA` or the vision tower silently tries an unsupported
> FA path. Also GPTQ-Int4 (not just AWQ) works on V100 MoE with PR #36026.

### 2.6 KTransformers on V100 — GPU expert kernel needs Marlin → BLOCKED (negative finding)
KTransformers (Tsinghua KVCache.AI; CPU/GPU **MoE offload** — runs DeepSeek-R1/V3 671B on a
single 24GB GPU at ~286 tok/s prefill / ~14 tok/s gen by keeping attention+shared-experts on
GPU and routing sparse experts to CPU/DRAM):
- Its **fast GPU expert operator is `KExpertsMarlin`** → **Marlin needs SM75+, so it will NOT
  run on V100.** Issues #434 / #1471 show Marlin-path errors and the GGUF weight-name mismatch
  when falling back. On V100 you'd be forced to `KExpertsTorch`/Llamafile (CPU) experts — much
  slower, partly defeating the point.
- BUT KTransformers' core idea (GPU = attention + dense/shared layers, CPU = sparse MoE experts)
  is exactly how you'd serve a **671B-class MoE** that will not fit in the server's card memory at usable quant.
  Feasible on our box IF the V100 expert kernel is swapped for a WMMA-FP16 INT4 kernel (same
  trick 1Cat uses) or experts run on CPU.
- Confidence: Med (Marlin gating well-documented; no direct V100 KTransformers success report).
- URLs: https://github.com/kvcache-ai/ktransformers/issues/434 ,
  https://github.com/kvcache-ai/ktransformers/issues/1471 ,
  https://developer.aliyun.com/article/1652385 (CN overview),
  https://madsys.cs.tsinghua.edu.cn/.../SOSP25-chen.pdf (SOSP'25 paper)

> **OPPORTUNITY (flag):** KTransformers is the most promising "serve a model BIGGER than the server's card memory"
> route (671B MoE via CPU expert offload), but needs a V100 (WMMA) expert kernel to be fast.
> This is a real engineering lane, not a config tweak. Roadmap candidate.

### 2.7 llama.cpp `-sm graph` (graph-level TP + NCCL) — newer multi-GPU mode
**[EN] Jagusztin (Medium) + llama.cpp docs:** a new split mode **`-sm graph`** does tensor
parallelism at the GGML graph level (distributes graph NODES, not just layers), uses **NCCL**
for topology-aware P2P, claims **3–4× over default layer-split** and keeps GPUs "pegged at
100%" (vs 20–30% idle on the bus). Demonstrated on **4× T4** (not V100), `sudo apt install
libnccl-dev`, build `-DGGML_CUDA=ON`, run `llama-cli -m model.gguf -sm graph -ngl 99`.
- Relevance to us: T4 is also a no-NVLink card; if `-sm graph`+NCCL helps T4, it may help our
  **cross-board PCIe Gen3 ×8** case (where layer-split currently dominates). Within an NVLink
  4-card board it should shine. Confidence: Med (claim is vendor-ish; no V100 number yet).
- URLs: https://medium.com/@jagusztinl/llama-cpp-performance-breakthrough-for-multi-gpu-setups-04c83a66feb2 ,
  https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md

> **MISSING FROM OUR STACK? (flag — high value):** Our notes's llama.cpp guidance is "-sm
> tensor (TP) for dense, -sm layer for MoE, -sm row deprecated." **`-sm graph` + NCCL is newer
> than that note** and explicitly targets the multi-GPU-idle problem that hurts us cross-board.
> Worth benchmarking `-sm graph` vs `-sm tensor`/`-sm layer` on (a) one NVLink board and (b)
> cross-board, with `libnccl-dev` installed.

### 2.8 ik_llama.cpp tensor-parallel status (Discussion #979)
`-sm row` in ik_llama **crashes**; with `-no-fug` it runs but gives wrong results → **ik_llama
has no working TP yet.** Maintainer's 2×3090 numbers (8B Q4_0): layer-split **170 tok/s** vs
row-split **72 tok/s** (~2.4×), calls mainline row-split "not proper TP and kinda shit." For
us: on ik_llama, stick to **layer-split**; don't expect TP. Confidence: High.
- URL: https://github.com/ikawrakow/ik_llama.cpp/discussions/979

---

## 3. Reddit r/LocalLLaMA multi-V100 builds

> Direct r/LocalLLaMA searches returned mostly noise (Reddit poorly indexed here); the most
> useful community datapoint surfaced via hardware-corner.net (which aggregates LocalLLaMA
> commenter experience). Captured below; flagged Med confidence as second-hand.

**[EN] hardware-corner.net "Tesla V100 32GB for LLM" (Aug 2025)** — FETCHED. Single V100-32GB,
**llama.cpp server**, Unsloth **Q4_K_XL** dynamic quants, Open WebUI, driver 575.57.08:

| Model | ctx | prompt-proc tok/s | gen tok/s | source |
|---|---|---|---|---|
| Qwen3-30B (MoE) | 10K | 787.56 | **55.03** | hardware-corner |
| Qwen3-30B (MoE) | 70K (max) | — | 12.32 | hardware-corner |
| Qwen3-32B (dense) | 10K | 396.43 | **16.87** | hardware-corner |
| Qwen3-32B (dense) | 28K (max) | — | ~11 | hardware-corner |
| Nemotron-49B (dense) | 7K | 402.56 | 17.66 | hardware-corner |

Cross-card context: RTX 3090 Qwen3-32B@10K = 24.52 t/s; RTX 5090@28K = 30.71 t/s. A LocalLLaMA
commenter planned 2× V100-16GB SXM2 + PCIe adapters (no benchmark). Confidence: High (article).
- URL: https://www.hardware-corner.net/guides/tesla-v100-32gb-for-llm/

> **KEY TAKEAWAY (confirms our notes):** On a single V100, **MoE crushes dense** — Qwen3-30B
> MoE = 55 t/s vs dense Qwen3-32B = 16.87 t/s at the same 10K ctx (3.3×). This is the strongest
> argument for our "V100 boards need MoE for ≥40 tok/s" policy, from an independent source.

### 2.4 SGLang on V100 (sm_70) — backend matrix
`sgl-project/sglang` attention-backend default selection (from docs/issue #5064):
**sm75 → Triton; sm80/86/89 → FlashInfer; sm90 → FA3/FlashInfer; sm100 → FlashInfer MLA.**
The published matrix starts at **sm75** — V100 (sm70) is NOT explicitly listed, so SGLang on
V100 would land on the **Triton** attention backend (the sm75 path) at best, with no FlashInfer.
Practically: SGLang is viable on V100 only via its Triton backend; expect it to be a notch
behind a purpose-built V100 vLLM/LMDeploy fork. Confidence: Med (inferred from the matrix; no
V100-specific SGLang success report found).
- URLs: https://github.com/sgl-project/sglang/issues/5064 ,
  https://docs.sglang.io/advanced_features/attention_backend.html

---

## 4. Forks / patches specifically enabling or optimizing V100 (sm_70)

There is a surprisingly deep ecosystem of V100-specific forks. Ranked by usefulness to us:

### 4.1 1Cat-vLLM (already covered in §2.1) — TurboMind-SM70 AWQ + FLASH_ATTN_V100 + MTP.
The most complete "serve big AWQ models on multi-V100" solution. Our existing lane.

### 4.2 ch2lab/vllm-v100 — second independent V100 vLLM fork — FETCHED (HAS tok/s numbers)
`github.com/ch2lab/vllm-v100` (branch `v100-support`). Independent of 1Cat but same ideas,
and it **publishes throughput numbers + specific bug fixes** 1Cat's README omits:
- **Patches:** custom SM70 prefill (`fused_mha_forward`) + paged-decode kernel with GQA;
  fixes a **softmax tail-max overflow** in long contexts; **TurboMind SM70 GEMM** for W4A16 AWQ;
  **FP8 KV** (e4m3 + e5m2) with per-token-head dynamic scaling; fixes FP8 E4M3 overflow
  threshold (exp>14 → exp>15) and an FP8 **E5M2-decoded-as-E4M3** dtype-mismatch bug.
- **Build (verbatim):**
  ```bash
  git clone github.com/ch2lab/vllm-v100.git && cd vllm-v100
  git checkout v100-support
  uv venv --python 3.12 && source .venv/bin/activate
  uv pip install torch==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128
  bash build_flash_v100.sh
  VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
  # full C++ rebuild: bash build_vllm.sh
  ```
  Requires torch **2.9.1+cu128** (note: 2.10+ raises the floor to **sm75**, dropping sm70 — a
  recurring trap; cf. our notes's "torch must be cu126 for FP8 MoE on Volta").
- **Throughput (Qwen3-0.6B-AWQ, V100-16GB, batch=8, short prompt):**

  | precision | tok/s | vs FP16 |
  |---|---|---|
  | FP16 KV | 206 | — |
  | FP8 E4M3 KV | 228 | +11% |
  | FP8 E5M2 KV | 239 | +16% |

  Caveat: "FP8 yields +11–16% decode throughput at ctx ≤ 2K; overhead beyond 8K." Tiny model,
  so absolute numbers aren't transferable, but the **FP8-KV decode win on V100 is real and
  free VRAM** — directly relevant to packing bigger models / longer context. Confidence: High.
- URL: https://github.com/ch2lab/vllm-v100

> **MISSING FROM OUR STACK? (flag)** FP8 KV cache (`fp8_e5m2`) on V100 gives +11–16% decode
> AND halves KV memory — lets us push context or batch on the same cards. 1Cat exposes it as
> experimental (`--kv-cache-dtype fp8_e5m2`); ch2lab fixed the E5M2 decode bug. Worth enabling
> + measuring on our drafting lane. Also: ch2lab's softmax tail-max overflow fix may be the
> same class of bug behind any long-context garble we've seen.

### 4.3 Standalone Volta FlashAttention kernel forks (plug into custom stacks)

**ai-bond/flash-attention-v100** — FETCHED. FA2 for sm70. Exposes `flash_attn_func`,
`flash_attn_varlen_func`, `flash_attn_with_kvcache` (so it DOES support a decode/KV path,
unlike sirCamp). Supports causal, sliding-window, softcap, ALiBi, dropout. Build: CUDA 12.9.1
(last Volta-supporting CUDA), `torch==2.10.0+cu129`, `pip install . --no-build-isolation -v`.
No perf numbers in README. This is the kernel most likely underneath the vLLM forks.
Confidence: Med-High. URL: https://github.com/ai-bond/flash-attention-v100

**sirCamp/flash-attention-legacy** — FETCHED. FA2 for Pascal+Volta with **native HF
Transformers v5.0+ integration** (`attn_implementation="flash_attn_legacy"`, no monkey-patch).
Hard limits (verbatim): **head dim only d=64/d=128; FP16 only (no BF16); no dropout in kernel;
PREFILL ONLY — decode (q_len=1) falls back to standard attention.** Uses WMMA 16×16×16 tensor
cores. Claim: "faster than eager for N≥2048." Measured at N=8192,B=2,H=8,d=64: only **1.15×**
faster than eager but **708 MB vs 16,882 MB memory** (the real win is MEMORY, enabling longer
ctx / avoiding OOM, not raw speed). Confidence: High (numbers in README).
- URL: https://github.com/sirCamp/flash-attention-legacy

**ZRayZzz/flash-attention-v100** — FA2 for V100 following CUTLASS tutorials + FA2 paper.
Educational/reference; less integrated. Confidence: Low (not fetched in depth).
- URL: https://github.com/ZRayZzz/flash-attention-v100

**FearL0rd/ComfyUI-Flash-Attention_v100** — FA**1** for V100/T4 as a ComfyUI node (image-gen
context, not LLM serving). Noted for completeness. URL:
https://github.com/FearL0rd/ComfyUI-ComfyUI-Flash-Attention_v100

**Dao-AILab/flash-attention#1760** — the canonical "can I use FA on V100?" thread; official
answer: **no** (sm_70 not supported upstream; raises `FlashAttention2 is not supported on this
GPU architecture (sm_70)`). Confirms why all the above are *forks*.
- URL: https://github.com/Dao-AILab/flash-attention/issues/1760

> **KEY SYNTHESIS on Volta FlashAttention:** The forks split into two camps —
> **prefill-only** (sirCamp: d=64/128, FP16, big MEMORY win, ~1.15× speed) vs **full
> prefill+decode** (ai-bond + the vLLM-integrated kernels in 1Cat/ch2lab, which add
> `with_kvcache`). For OUR goal (serve big models fast + bigger models), the value of Volta FA
> is mostly (a) **memory** — fit longer context / more KV on 32GB — and (b) prefill latency on
> long prompts; raw decode tok/s is dominated by the W4 GEMM (TurboMind-SM70/Marlin), not
> attention. So the AWQ GEMM kernel + FP8 KV matter more for tok/s than the FA kernel does.

### 3.1 databasemart 3×V100 vLLM benchmark — best multi-V100 throughput data found
**[EN] databasemart.com "3×V100 vLLM benchmark"** — FETCHED. 3× V100, vLLM, **float16** (no
quant), input 100 / output 600 tok, concurrency 50 & 100. The crucial methodological note:
**"TP=2 used wherever possible; PP=3 for models too large to fit under TP=2."** This is exactly
our NVLink-pair-vs-cross-board dilemma in miniature.

| Model | Size GB | TP | PP | conc | total tok/s | output tok/s | TTFT ms | source |
|---|---|---|---|---|---|---|---|---|
| gemma-3-4b | 8.1 | 2 | 1 | 50 | 1847 | 1583 | 689 | databasemart |
| gemma-3-12b | 23 | 1 | 3 | 50 | 449 | 385 | 1157 | databasemart |
| Qwen2.5-7B | 15 | 2 | 1 | 50 | 1384 | 1150 | 801 | databasemart |
| Qwen2.5-14B | 28 | 1 | 3 | 50 | 590 | 492 | 1279 | databasemart |
| R1-Distill-Llama-8B | 15 | 2 | 1 | 50 | 1449 | 1229 | 903 | databasemart |
| R1-Distill-Qwen-14B | 28 | 1 | 3 | 50 | 480 | 372 | 1239 | databasemart |
| gemma-3-4b | 8.1 | 2 | 1 | 100 | 2475 | 2121 | 1167 | databasemart |
| Qwen2.5-14B | 28 | 1 | 3 | 100 | 798 | 665 | 2224 | databasemart |

Confidence: High (full tables). URL: https://www.databasemart.com/blog/vllm-gpu-benchmark-v100-3

> **TAKEAWAY:** Note how the **14B-class (PP=3) models run ~3× SLOWER aggregate than 7B-class
> (TP=2)** — partly model size, but the PP path also serializes. This independently supports
> keeping a model on ONE NVLink board (TP within board) rather than spanning boards (PP) when
> it fits. For us: TP=4 within a 4-card NVLink board > TP=8 across two boards over PCIe.

### 3.2 Ollama single-V100 (16GB) baseline table
**[EN] databasemart Ollama V100 benchmark** (1× V100-16GB, 4-bit GGUF):

| Model | size | 4-bit gen tok/s |
|---|---|---|
| llama2-7b | 3.8 | 107.49 |
| mistral-7b | 4.1 | 107.31 |
| qwen2.5-7b | 4.7 | 86.00 |
| deepseek-r1-7b | 4.7 | 87.10 |
| llama3.1-8b | 4.9 | 84.07 |
| deepseek-coder-v2-16b (MoE) | 8.9 | 69.16 |
| llama2-13b | 7.4 | 67.51 |
| gemma2-9b | 5.4 | 59.90 |
| qwen2.5-14b | 9.0 | 49.38 |
| deepseek-r1-14b | 9 | 48.63 |
| **gemma2-27b (dense)** | 16 | **8.37** |

Note the dense-27B cliff (8.37 t/s) — Ollama's plain layer path on one card is much slower than
llama.cpp-tuned (hardware-corner got dense-32B = 16.87 on llama.cpp). Confidence: High.
- URL: https://www.databasemart.com/blog/ollama-gpu-benchmark-v100

### 3.3 LMDeploy "1.8× vLLM" throughput claim (multiple CN sources)
**[CN] 火山引擎 / CSDN gitblog:** LMDeploy advertises **throughput 1.8× vLLM**, via Persistent
(continuous) Batch, blocked KV cache, dynamic split-fuse, TP, and tuned kernels; **W4 inference
2.4× FP16**; AWQ cuts VRAM 75% with <1% precision loss; supports AWQ + KV-quant + auto-prefix
simultaneously. This corroborates the QwenLM#1023 real-world 80 vs 52 tok/s (~1.5×) gap.
Confidence: Med (vendor claim, but multiply-sourced + a matching real measurement).
- URLs: https://developer.volcengine.com/articles/7540544902736642091 ,
  https://blog.csdn.net/gitblog_00698/article/details/151703190 ,
  https://www.cnblogs.com/SVicen/p/19281160 (大模型推理加速全景技术指南)

### 3.4 Negative findings (tools that do NOT help V100)
- **ExLlamaV2 / EXL2:** optimized for **Ampere+**; no V100-specific path; community has no
  V100 EXL2 numbers. EXL2's speed advantage does not transfer to sm70. Skip on V100.
  URL: https://oobabooga.github.io/blog/posts/gptq-awq-exl2-llamacpp/
- **FlashRT** ("CUDA-first inference runtime"): a $200 V100 reportedly beats an RTX 3060, but
  the article gives **no V100/sm70 support confirmation and no numbers** — unverified, watch-list.
  URL: https://dev.to/soytuber/deepseek-v4-flash-benchmarks-flashrt-cuda-runtime-v100-llm-performance-58i2
- **SGLang:** no sm70 in its backend matrix (sm75 floor) → Triton-only on V100, no FlashInfer.
- **KTransformers GPU experts:** Marlin-gated (sm75+) → CPU expert fallback only on V100.

---

## 5. Consolidated perf tables

### 5.1 Master table — every V100 tok/s datapoint gathered (model | stack | GPUs | quant | tok/s | source)

| Model | Stack | GPUs | Quant | tok/s (gen) | Notes | Source |
|---|---|---|---|---|---|---|
| Qwen3-30B MoE | llama.cpp | 1×V100-32G | Q4_K_XL | **55.0** @10K ctx | 12.3 @70K | hardware-corner |
| Qwen3-32B dense | llama.cpp | 1×V100-32G | Q4_K_XL | 16.9 @10K | ~11 @28K | hardware-corner |
| Nemotron-49B dense | llama.cpp | 1×V100-32G | Q4_K_XL | 17.7 @7K | — | hardware-corner |
| Qwen2.5-32B dense | LMDeploy TurboMind | 4×V100-16G | GPTQ-W4 +INT8KV | **~80** | single-stream | QwenLM#1023 |
| Qwen2.5-32B dense | vLLM | 4×V100-16G | GPTQ-W8 | ~52 peak | 7.3 prompt/14.6 tail | QwenLM#1023 |
| Qwen3.5-35B-A3B MoE | 1Cat-vLLM | 4×V100-32G | AWQ-W4 (WMMA) | **~121** (our notes) | LF_DRAFTER=awq | our notes + 1Cat |
| Qwen3-0.6B | ch2lab vllm-v100 | 1×V100-16G | AWQ + FP8-E5M2 KV | 239 (vs 206 FP16) | +16% from FP8 KV | ch2lab |
| Qwen2.5-7B | vLLM | 3×V100 | fp16 | 1150 (50-conc agg) | TP=2 | databasemart |
| Qwen2.5-14B | vLLM | 3×V100 | fp16 | 492 (50-conc agg) | PP=3 (slower path) | databasemart |
| gemma2-27b dense | Ollama | 1×V100-16G | Q4 | 8.4 | untuned layer path | databasemart |
| (kernel demo) AWQ no-Marlin | vLLM | V100 | AWQ | 67 | vs 741 w/Marlin | liaoweiming/devpress |
| DeepSeek-R1 671B MoE | KTransformers | 1×24G+CPU | Q4 | 14 gen / 286 prefill | NOT V100 (4090D); Marlin-gated on V100 | aliyun |

(Aggregate throughput numbers are at 50–100 concurrency; single-stream numbers noted. Different
models/contexts — read the Notes column before comparing.)

### 5.2 Decision heuristics distilled from the community
1. **MoE >> dense on V100** for tok/s (55 vs 17 same ctx). Prefer MoE (A3B/A10B) drafters.
2. **W4 (AWQ/GPTQ) only pays off if it hits a fast kernel** — Marlin (sm75+, NOT V100) or the
   fork's **WMMA-FP16 INT4** kernel (TurboMind-SM70). Else 4-bit can be *slower* than FP16.
3. **LMDeploy TurboMind ≥ vLLM** on V100 for dense W4 (~1.5–1.8×); set **INT8 KV (`--quant-policy 8`)**.
4. **FP8 KV cache (e5m2)** = +11–16% decode AND half KV memory on V100 (ch2lab) — nearly free.
5. **Keep a model on ONE NVLink board** (TP within board). Cross-board PP/TP over PCIe Gen3 ×8
   serializes and loses badly (databasemart PP=3 cliff; our own topology note).
6. **Attention on V100:** TRITON_ATTN (safe, multiply-recommended) vs FLASH_ATTN_V100 (1Cat,
   faster long-prompt but version-fragile/96KB-crash). For VL models, mm-encoder = TORCH_SDPA.
7. **First request compiles kernels 1–3 min** — warm up before benchmarking; never trust req #1.
8. **torch/CUDA pinning is the #1 footgun:** sm70 dropped at torch 2.10 (sm75 floor) and CUDA
   13.0 (Volta offline-compile removed). Pin **CUDA ≤12.9, torch cu126/cu128/cu129** per fork.

### 3.5 TP/PP/EP parallel-strategy decision rules (CN deep-dive) — maps onto our topology
**[CN] cnblogs ljbguanli "分布式大模型推理实战：TP/PP/EP并行策略深度解析"** — FETCHED. The single
clearest statement of *when to use which parallelism*, and it lines up exactly with the V100 server:

| Strategy | Bandwidth needed | Latency | Network | Use for |
|---|---|---|---|---|
| TP (single node) | **600 GB/s+** | <5µs | NVLink/NVSwitch | 70B+ dense, latency-sensitive |
| TP (cross-node) | 400 Gbps+ | <10µs | InfiniBand NDR | — |
| PP | **50 Gbps+** | <50µs | RoCEv2/IB | cross-node 405B+, bandwidth-limited |
| EP (MoE) | 200 Gbps+ | <20µs | IB HDR/NDR | DeepSeek-V3-class 256-expert MoE |

Verbatim rule: **"TP must be confined to single nodes; cross-node deployments should use PP or
EP"** — TP comm overhead overwhelms compute below ~600 GB/s.
- (Minor source error: it says "V100 lacks NVLink." FALSE for our V100-**SXM2** — they have
  NVLink *within* the 4-card boards. That intra-board NVLink is precisely the asset to use for TP.)
- Confidence: High (clear, well-structured). URL: https://www.cnblogs.com/ljbguanli/p/19966993

> **DIRECT MAPPING TO the V100 server (the key architectural insight of this report):**
> - Our measured fabric (memory): intra-board NVLink {2,3}=145, {0,1}LS=107 GB/s; **cross-board
>   = PCIe Gen3 ×8 ≈ 6.5 GB/s host.** The TP threshold is ~600 GB/s.
> - Our best NVLink board (~145 GB/s) is **~4× under** the ideal-TP bandwidth, and cross-board
>   (6.5 GB/s) is **~90× under** it. So: **TP=4 within one board is acceptable-but-not-ideal;
>   TP across boards (6.5 GB/s) is catastrophic and must never be used.**
> - Therefore the community-correct layout for the V100 server: **TP=4 inside ONE 4-card NVLink board per
>   model** (exactly our current "one large model per board" policy), and to go BIGGER than one
>   board, use **PP across boards** (PP tolerates 50 Gbps ≈ 6.25 GB/s — right at our PCIe ×8!)
>   or **EP for MoE** (route experts to other boards). This independently validates our
>   gpu_pools policy and tells us the *correct* way to span boards is **PP/EP, never TP**.

### 3.6 Misc CN deployment-guide confirmations
- **[CN] nvidia.csdn "【V100显卡】用vllm本地部署 QWEN3"** — confirms the universal V100 vLLM rule:
  must pass `--dtype float16` (or `half`); vLLM defaults to bf16 which sm70 can't do.
  URL: https://nvidia.csdn.net/68e9d3734b11580edfa2c99c.html
- **[CN] 8-card-V100 OOM playbook (search consensus):** for big models on 8×V100, combine
  `--tensor-parallel-size` (within board) + `--pipeline-parallel-size` (across), lower
  `--gpu-memory-utilization`, reduce `--max-model-len`, raise `--swap-space`, and use
  `--enforce-eager` (skips CUDA-graph capture, saves VRAM at some speed cost). **V100 < 8.0 →
  no FP8 weight quant** (FP8 *KV cache* is a separate, fork-enabled thing — don't conflate).
- **[EN] jinzhen-lin vLLM PR #14138** — "optimize gptq/awq Marlin kernel when n is small"
  (barrier_acquire/release dominate at small batch). Relevant to *Marlin-capable* cards, not
  V100 directly, but explains why low-batch single-stream W4 underperforms — and our V100 WMMA
  kernel likely has the same small-n sensitivity. URL: https://github.com/vllm-project/vllm/pull/14138
- **[CN] Qwen3.6-27B vLLM guide (jermey.cn)** — note: targets **A100 FP8**, NOT V100 (the FP8
  weight path is unavailable on sm70). Useful only as a reminder that most CN "Qwen3.6 deploy"
  guides assume Ampere FP8 and DON'T transfer to V100 — you need the 1Cat/ch2lab AWQ-WMMA path.
  URL: https://www.jermey.cn/2026/05/02/vllm-deploy-qwen3-guide.html

---

## 6. What our stack is missing (synthesis & recommendations)

Our current V100 serving stack (per memory): **1Cat-vLLM AWQ (WMMA) drafting lane @121 tok/s**,
**llama.cpp kv-split for gemma/dense**, **LMDeploy** available, **TRITON_ATTN** to dodge the
FLASH_ATTN_V100 96KB crash, reranker/embedder sidecars. Against the community state of the art,
here is what is genuinely missing or under-exploited, ranked by expected payoff:

**(A) FP8 KV cache on V100 — free +11–16% decode + half the KV memory.** [ch2lab/vllm-v100]
We are not (per memory) running `--kv-cache-dtype fp8_e5m2`. ch2lab not only enables it on sm70
but *fixed the E5M2-decoded-as-E4M3 bug*. Half-size KV directly lets us push context length or
batch on the same 32 GB, and the decode speedup is nearly free at ctx ≤ ~8K. **This is the
single highest-ROI, lowest-risk change.** Test on the drafting lane; watch for the >8K overhead.

**(B) Re-examine the attention-backend decision (FLASH_ATTN_V100 vs TRITON_ATTN).** We
defaulted to TRITON_ATTN after a 96KB FLASH_ATTN_V100 prefill crash. But 1Cat's *current*
README treats FLASH_ATTN_V100 (with `VLLM_FLASH_V100_ENABLE_PAGED_PREFILL=1`) as the
recommended, working backend — implying the crash class we hit may now be fixed by the
paged-prefill path. The CN deep-dive still recommends TRITON_ATTN for safety. **Action:**
A/B both on the *current* 1Cat build with long prompts; FLASH_ATTN_V100 should win prefill
latency if stable. Keep TRITON_ATTN as the proven fallback.

**(C) Reconcile the all-reduce / pynccl flags with our topology.** The CN deep-dive's V100
config adds `VLLM_DISABLE_PYNCCL=1` + `--disable-custom-all-reduce`; 1Cat says do NOT disable
custom all-reduce for 27B TP4 decode. These flags govern the collective that runs over our
intra-board NVLink (or, badly, cross-board PCIe). Within one NVLink board, custom all-reduce
likely helps; the safe-fallback flags exist for flaky P2P. **Action:** measure both on a single
4-card board; the right answer is topology-specific and we have the better (NVLink) case.

**(D) `-sm graph` + NCCL in llama.cpp — newer than our "-sm tensor/layer" rule.** Our notes's
llama.cpp guidance predates the graph-split mode. `-sm graph` does graph-level TP with NCCL
topology awareness, claims 3–4× over layer-split and targets the multi-GPU-idle problem that
hurts us most cross-board. **Action:** `apt install libnccl-dev`, rebuild with `-DGGML_CUDA=ON`,
benchmark `-sm graph` vs our current `-sm tensor`(dense)/`-sm layer`(MoE) on (i) one NVLink
board and (ii) cross-board. Could change our gemma/dense lane defaults.

**(E) LMDeploy TurboMind as a first-class dense-W4 lane with INT8 KV.** Real-world CN evidence
(QwenLM#1023) shows LMDeploy W4 + `--quant-policy 8` ≈ 80 tok/s vs vLLM W8 ≈ 52 on the SAME
32B dense model, and CN sources claim ~1.5–1.8× vLLM throughput generally. Our notes have
LMDeploy "available" but the AWQ drafter runs on 1Cat-vLLM. For **dense** 30–70B models (where
1Cat's MoE-tuned path is less of an advantage), LMDeploy TurboMind W4 + INT8 KV may beat our
vLLM path. **Action:** bench LMDeploy vs 1Cat-vLLM on a dense 32B (apples-to-apples), with INT8
KV on. (Our 121 tok/s headline is a *MoE* model — not comparable to dense 32B.)

**(F) The "bigger than one board / bigger than the server's card memory" path is PP/EP, not TP — and KTransformers
is the 671B route.** The clearest new strategic finding: per the TP/PP/EP bandwidth table, to
serve a model larger than one 4-card board you must use **pipeline parallel across boards**
(PP tolerates ~6 GB/s ≈ our PCIe ×8) or **expert parallel** for MoE — *never* cross-board TP.
For a 671B-class MoE that won't fit even at Q4, **KTransformers' CPU/GPU expert-offload** is the
community's answer (286 prefill / 14 gen on a single 24 GB + CPU). The catch: its fast GPU
expert kernel (KExpertsMarlin) is sm75+, so on V100 it needs a WMMA-FP16 INT4 expert kernel
(the same trick 1Cat already implements) or CPU experts. **This is an engineering lane (roadmap),
not a config flag — but it is the only credible "serve a model bigger than the box" path found.**

**(G) Guardrails we should codify (already partly in memory, reinforced by sources):**
- Never run FP4/FP8-Marlin paths in **bf16** on V100 → garbled output (vLLM#34694); FP16 only.
- Verify the W4 kernel actually selected (Marlin-substitute/WMMA), or 4-bit can be *slower*
  than FP16 (vLLM#21266; the 67-vs-741 gap).
- Pin **CUDA ≤12.9 + torch cu126/128/129**; torch 2.10 and CUDA 13.0 drop sm70.
- For VL models: `--mm-encoder-attn-backend TORCH_SDPA` (vision tower has no sm70 FA).
- Warm up (first request compiles kernels 1–3 min); never benchmark request #1.

**What we are NOT missing (doing right):** one-large-model-per-NVLink-board (TP within board);
MoE-first for tok/s; AWQ-WMMA via 1Cat; TRITON_ATTN as a safe attention default; FP16 dtype.
The community broadly does exactly this — our gpu_pools "no cross-board job" rule is *validated*
by the TP/PP bandwidth math.

---

## 7. Confidence & gaps

- **High confidence:** the SM70 feature-gating matrix; AWQ-needs-WMMA-not-Marlin; MoE≫dense on
  V100; TP-within-board / PP-across-board; FP8-KV decode win; torch/CUDA version traps; the
  Volta FA forks' existence and limits. (All multiply-sourced or read in full.)
- **Medium confidence:** exact tok/s comparisons across stacks (different models/contexts/
  concurrency — the master table mixes single-stream and 50–100-conc aggregate; treat as
  *relative*, not absolute); LMDeploy "1.8× vLLM" (vendor claim + one real measurement);
  `-sm graph` magnitude (demoed on T4, not V100).
- **Gaps / could not verify:** (1) No first-hand **large-model (100B+) cross-board PP on V100**
  tok/s number found anywhere — community mostly serves ≤35B per box or uses MoE. (2) No public
  **1Cat-vLLM throughput table** (numbers are from our own memory). (3) Reddit r/LocalLLaMA
  is poorly indexed via this search path; deeper Reddit mining (direct) could add real
  homelab-V100 multi-card anecdotes. (4) Zhihu/CSDN bodies were blocked (403/521) — snippet-
  level only for several; a CN-locale fetch or browser would recover full configs.

---

## TL;DR (top of file mirror)

1. **Two community V100-vLLM forks** beyond what our notes detail: **1CatAI/1Cat-vLLM** (our
   lane; TurboMind-SM70 WMMA AWQ + FLASH_ATTN_V100 + MTP) and **ch2lab/vllm-v100** (publishes
   tok/s + **FP8-KV** +11–16% decode + bug-fixes). Both replace the missing Marlin path with a
   **WMMA-FP16 INT4** kernel — the central V100 trick.
2. **AWQ on stock vLLM is dead on V100** (no Marlin, sm75+): ~67 tok/s slow path vs ~741 with
   Marlin elsewhere (10.9×). The forks' WMMA kernel is what makes 4-bit fast on sm70.
3. **MoE ≫ dense on V100** (Qwen3-30B MoE 55 t/s vs dense-32B 17 t/s @10K, llama.cpp). Stay MoE.
4. **LMDeploy TurboMind W4 + INT8 KV ≈ 80 t/s beats vLLM W8 ≈ 52 t/s** on the same dense 32B
   (CN GitHub real measurement) — under-used for our dense lanes.
5. **Topology law:** TP needs ~600 GB/s → keep **TP=4 within one NVLink board**; go bigger only
   via **PP/EP across boards** (PCIe ×8 ≈ 6 GB/s is fine for PP, fatal for TP). Validates our
   "no cross-board job" policy and tells us PP/EP is the *correct* board-spanning method.
6. **Highest-ROI missing pieces:** (A) FP8 KV cache (free decode + half KV mem), (B) re-test
   FLASH_ATTN_V100 paged-prefill vs TRITON_ATTN, (D) llama.cpp `-sm graph`+NCCL, (E) LMDeploy
   dense-W4 lane, (F) KTransformers expert-offload for 671B (needs a V100 WMMA expert kernel).
7. **Footguns:** never bf16 on V100 (garble); pin CUDA≤12.9/torch≤2.9-sm70; verify the W4
   kernel actually engaged; `--dtype float16`; VL → `--mm-encoder-attn-backend TORCH_SDPA`;
   warm up (req#1 compiles 1–3 min).

## Source log (every URL)

### Chinese-language sources [CN]
- ⭐ https://www.liaoweiming.org/blog/vllm-sm-architecture-adaptation — vLLM SM-arch dispatch deep-dive (FETCHED FULL) [High]
- https://devpress.csdn.net/v1/article/detail/161462608 — why V100 doesn't support AWQ by default; 67-vs-741 Marlin gap [Med-Hi]
- https://github.com/QwenLM/Qwen3/issues/1023 — 4×V100 LMDeploy Qwen2.5-32B GPTQ-Int4 80 t/s vs vLLM 52 (FETCHED) [High]
- https://huggingface.co/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4/discussions/5 — 4×V100 vLLM + PR#36026 config (FETCHED) [High]
- https://www.cnblogs.com/ljbguanli/p/19966993 — TP/PP/EP parallel-strategy decision rules (FETCHED) [High]
- https://nvidia.csdn.net/68e9d3734b11580edfa2c99c.html — V100 vLLM QWEN3 must use --dtype float16 [Med]
- https://www.jermey.cn/2026/05/02/vllm-deploy-qwen3-guide.html — Qwen3.6-27B vLLM (A100 FP8, NOT V100) (FETCHED) [Med]
- https://developer.aliyun.com/article/1652385 — KTransformers 671B on 24G overview [Med]
- https://madsys.cs.tsinghua.edu.cn/publication/.../SOSP25-chen.pdf — KTransformers SOSP'25 paper [Med]
- https://developer.volcengine.com/articles/7540544902736642091 — Ollama vs vLLM vs LMDeploy compare [Med]
- https://blog.csdn.net/gitblog_00698/article/details/151703190 — LMDeploy 1.8× vLLM throughput claim [Med]
- https://www.cnblogs.com/SVicen/p/19281160 — 大模型推理加速全景技术指南 [Med]
- https://zhuanlan.zhihu.com/p/1956047359613044602 — Tesla V100 + vLLM 0.10.2 pitfall log (BLOCKED 403, snippet) [Med]
- https://ascendai.csdn.net/6986a94454b52172bc5a64cf.html — 8-card V100 vLLM Qwen (BLOCKED 404, snippet) [Med]
- https://blog.csdn.net/ai_hpc_trivia/article/details/155302212 — vLLM multi-card perf test (BLOCKED 521, snippet) [Low]
- https://github.com/AllentDan/lmdeploy/blob/main/docs/zh_cn/quantization/w4a16.md — LMDeploy W4A16 CN docs [Med]
- https://github.com/InternLM/lmdeploy/blob/main/README_zh-CN.md — LMDeploy CN readme [Med]

### English sources [EN]
- https://github.com/1CatAI/1Cat-vLLM — V100 vLLM fork (TurboMind-SM70 AWQ, FLASH_ATTN_V100, MTP) (FETCHED) [High]
- https://github.com/ch2lab/vllm-v100 — 2nd V100 vLLM fork, FP8-KV + tok/s + bugfixes (FETCHED) [High]
- https://github.com/vllm-project/vllm/issues/34694 — FP4 Marlin garble in bf16 on V100/4090; FP16 unaffected [High]
- https://github.com/vllm-project/vllm/issues/21266 — GPTQ w4a16 slower than FP16 (kernel-miss) [Med-Hi]
- https://github.com/vllm-project/vllm/pull/36026 — SM70 MoE bugfix referenced by HF#5 [Med]
- https://github.com/vllm-project/vllm/pull/14138 — Marlin small-n optimization (jinzhen-lin) [Med]
- https://discuss.vllm.ai/t/v100-vllm/1896 — V100 vLLM concurrency/prefill-block, backend matrix (FETCHED) [High]
- https://github.com/sgl-project/sglang/issues/5064 + docs — SGLang attn backend matrix (sm75 floor) [Med]
- https://github.com/ikawrakow/ik_llama.cpp/discussions/979 — ik_llama TP status; layer 170 vs row 72 (FETCHED) [High]
- https://github.com/kvcache-ai/ktransformers/issues/312 — KTransformers multi-GPU DeepSeek R1 [Med]
- https://github.com/kvcache-ai/ktransformers/issues/434 — KExpertsMarlin CUDA-graph error [Med]
- https://github.com/kvcache-ai/ktransformers/issues/1471 — KExpertsMarlin KeyError / GGUF name mismatch [Med]
- https://github.com/ai-bond/flash-attention-v100 — FA2 for sm70 w/ KV-cache path (FETCHED) [Med-Hi]
- https://github.com/sirCamp/flash-attention-legacy — FA2 Pascal/Volta, HF-integrated, prefill-only (FETCHED) [High]
- https://github.com/ZRayZzz/flash-attention-v100 — FA2 V100 reference impl [Low]
- https://github.com/FearL0rd/ComfyUI-ComfyUI-Flash-Attention_v100 — FA1 V100/T4 ComfyUI node [Low]
- https://github.com/Dao-AILab/flash-attention/issues/1760 — official "no V100 FA" thread [High]
- https://www.hardware-corner.net/guides/tesla-v100-32gb-for-llm/ — single-V100-32G llama.cpp tok/s (FETCHED) [High]
- https://www.databasemart.com/blog/vllm-gpu-benchmark-v100-3 — 3×V100 vLLM benchmark tables (FETCHED) [High]
- https://www.databasemart.com/blog/ollama-gpu-benchmark-v100 — 1×V100-16G Ollama table (FETCHED) [High]
- https://medium.com/@jagusztinl/llama-cpp-performance-breakthrough-for-multi-gpu-setups-04c83a66feb2 — `-sm graph`+NCCL (FETCHED) [Med]
- https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md — llama.cpp multi-GPU docs [Med]
- https://oobabooga.github.io/blog/posts/gptq-awq-exl2-llamacpp/ — quant format comparison (EXL2 = Ampere+) [Med]
- https://dev.to/soytuber/deepseek-v4-flash-benchmarks-flashrt-cuda-runtime-v100-llm-performance-58i2 — FlashRT (unverified on V100) [Low]
- https://knightli.com/en/2026/05/09/llama-cpp-multi-gpu-offload-performance/ — 2×V100-16G vs 1×32G methodology (FETCHED) [Med]

