# V-C — Speed-Ups & Untried Options for a V100 (sm_70) server

> **Origin.** PLI Labs research archive, file `v100-research/findings/V-C-speedups-untried.md`. Published 2026-11-18 in the Volta V100 research
> repository.
> **Edited for publication.** Machine names, private file paths, network addresses, session names and the
> operator's name were replaced with generic terms (`the V100 server`, `[private path]`, `[address]`,
> `the owner`). Links to files that are not published were turned into plain text. No figure and no finding
> was changed. Dates inside the note are its own; later results are in the build pages of this repository.

**Scope:** every concrete way to make *inference faster* on a multi-board Tesla V100-SXM2-32GB server
(sm_70 / Volta), with emphasis on **options NOT YET TRIED** on this server. Decode-tok/s
focused (the target workload is single-stream, `max_num_seqs=1`, long context).
Research-first and adversarial: marketing tok/s are distrusted, and every technique that
silently needs sm_80+ is flagged as a non-starter on Volta.

**Method / provenance.** Findings combine (a) **on-box inspection** of the installed
`1Cat-vLLM-1.0.0` conda env and a private benchmark setup (ground
truth), (b) two parallel deep-web research lanes (spec-decode/quant; attention/MoE/serving),
and (c) corroboration from internal notes. Per the eval-provenance rule, **no number
here is decision-grade until re-measured on our own box with `/v1/models` verified** — most
public numbers are A100/H100 (sm_80+) and are labelled **(NOT V100)**. Confidence is per
claim (high / med / low).

---

## 0. The two facts that reorganize the whole question

1. **The model architecture matters before the GPU.** Qwen3.5 / Qwen3.6 (the A3B-class
   targets) are **hybrid Gated-DeltaNet (linear-attention) + gated-attention MoE** models
   (Qwen3-Next lineage; 3.6 mixes the two ~4:1). That recurrent SSM state changes which
   speculative-decoding methods are even *correct*, **independent of Volta**. The 122B-A10B
   target is conventional full-attention and behaves differently. So "what speeds up decode"
   is driven first by model family, second by sm_70. (Confidence: high — corroborated on-box
   by `qwen3_5_mtp`/`qwen3_next_mtp` model types in the fork; see §1.)

2. **Our workload is the best case for latency tricks and the worst case for throughput
   marketing.** At `max_num_seqs=1`, decode is pure memory-bandwidth-bound weight streaming
   with the GPU under-utilized. Spec-decode multipliers are *larger* at batch=1 than the
   batched QPS numbers vendors quote; conversely CUDA-graph and kernel-launch overhead is a
   *larger* fraction of step time at batch=1, so removing it helps more here than the headline
   suggests. (Confidence: high — mechanistic.)

---

## 1. Speculative / assisted decoding on V100

**On-box reality (verified):** the installed fork ships the **full vLLM v1 spec-decode suite** —
`vllm/v1/spec_decode/{ngram_proposer,suffix_decoding,eagle,medusa,draft_model,dflash}.py` —
and `config/speculative.py` registers `ngram`, `suffix`, `eagle`/`eagle3`, and the MTP family
including **`qwen3_5_mtp`** and **`qwen3_next_mtp`**. There is **no sm_70 guard** anywhere in
the spec-decode path; the only `capability` checks are for multimodal/M-RoPE, not Volta. So the
machinery is present and arch-agnostic on the proposer side — the question is which method is
*correct* and *fast* for our models.

### MTP (built-in multi-token-prediction heads) — the winner for this server
- **Qwen3.5/3.6 ship MTP weights inside the checkpoint** (no separate draft file). vLLM's
  official Qwen3.5 recipe enables it with
  `--speculative-config '{"method":"mtp","num_speculative_tokens":N}'` and recommends **MTP-1
  at low concurrency** ("reduces TPOT with high acceptance"). Confirmed on-box:
  `qwen3_5_mtp`/`qwen3_next_mtp` are registered MTP model types. **Confidence: high.**
- **No sm_80 requirement** — the MTP head is one extra transformer layer over the verifier's
  final hidden state, runs bf16/fp16. The 1Cat-vLLM README already validates **MTP as its
  primary V100 spec-decode path** (ships `Qwen3.6-27B-AWQ + MTP` TP4 examples; `max_num_seqs`
  default 1 baseline / 4 for MTP). **Confidence: high.**
- **MTP is the spec method that is *correct* on Gated-DeltaNet models.** vLLM issue
  [#39273](https://github.com/vllm-project/vllm/issues/39273): ngram/suffix advance the GDN SSM
  recurrent state by all N proposed tokens with **no rollback** to the accepted count →
  corrupted/repeated output. **MTP avoids this** because draft tokens are generated *after* the
  base step completes (state evolution and proposal are decoupled). This is the decisive reason
  to default to MTP — and it is **not a Volta bug; it breaks on A100/H100 too.** **Confidence: high.**
- **Realistic gain:** Qwen3.6-27B@Q8 measured **2.24× @ γ=2 (accept 0.83)**, **2.40× @ γ=3
  (accept 0.72)** on DGX-Spark / RTX-3090-class (NOT V100); a 3090 run showed 38→65 tok/s
  (1.71×). Absolute tok/s will be lower on Volta, but **acceptance rate is model-intrinsic and
  transfers** → expect a real **~1.7–2.2× decode multiplier at batch=1.** **Confidence: med-high.**
  Sources: [vLLM Qwen3.5 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html),
  [zolotukhin.ai MTP](https://zolotukhin.ai/blog/2026-05-08-why-mtp-heads-are-the-speculative-decode-draft-qwen3-a3b-deserves/),
  [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM).

### n-gram / prompt-lookup (`method: ngram`)
- Draft-model-free; the proposer is table-lookup against the prompt/history, **no quant/attention
  tensor-core kernels → arch-independent, runs on sm_70.** **Confidence: high.**
- Canonical gain up to **~2.8× at QPS=1 on summarization** (Llama-3-70B/4×H100) — long-document
  drafting is similarly repetitive (boilerplate, repeated names, quoted source text), so a
  **1.5–2.5×** band is plausible; the *acceptance* transfers, the absolute tok/s does not.
  **Confidence: med.** [vLLM spec-decode blog](https://blog.vllm.ai/2024/10/17/spec-decode.html).
- **⚠️ Unsafe on Qwen3.5/3.6 (GDN) per #39273 — use ONLY on the 122B full-attention target.**
  **Confidence: high.**

### Suffix decoding (`method: suffix`, Arctic / SuffixDecoding)
- Strictly stronger draft-model-free proposer than ngram (matches prompt *and* prior
  generations, frequency-weighted, adaptive depth; on-box `suffix_decoding_max_tree_depth=24`).
  CPU proposer (~20 µs/token) → **arch-independent, runs on sm_70.** **Confidence: high.**
- Reported **2.3×–6.3× decode-time reduction** (1.4×–3.9× faster than vLLM ngram) on
  high-repetition workloads; document drafting likely lands at the lower end. **Confidence: med**
  (vendor numbers, mixed HW).
  [Snowflake/Arctic](https://www.snowflake.com/en/engineering-blog/suffixdecoding-arctic-inference-vllm/).
- **⚠️ Same GDN "propose-N / partial-accept" pattern as ngram → almost certainly hits the #39273
  rollback bug on 3.5/3.6** (no explicit safe-on-GDN confirmation found). Treat as **122B-only
  until verified.** Also requires the Arctic Inference package — verify it builds against the
  sm_70 fork's vLLM pin. **Confidence: med.**

### Draft-model speculative decoding (small Qwen draft + big Qwen target)
- The draft reuses the target's attention/quant kernels → would run on the fork's
  `FLASH_ATTN_V100`/Triton, **no FA2 required.** Historically vLLM wired the draft path tightest
  to FlashInfer (sm_75+), with known spec-decode metadata bugs ([#13264](https://github.com/vllm-project/vllm/issues/13264));
  **must force the V100 backend, not FlashInfer.** **Confidence: med.**
- Weakest modern option (extra KV + compute, lower acceptance than feature heads); a *dense* small
  draft mispredicts MoE routing-driven tokens, depressing acceptance. **Not recommended over MTP.**
  **Confidence: low-med.** [BentoML](https://www.bentoml.com/blog/3x-faster-llm-inference-with-speculative-decoding).

### EAGLE / EAGLE-3 / EAGLE-3.1
- **Public heads exist for Qwen3 but NOT Qwen3.5/3.6 (the GDN generation).** Closest matches:
  [AngelSlim/Qwen3-a3B_eagle3](https://huggingface.co/AngelSlim/Qwen3-a3B_eagle3) (Qwen3-30B-A3B),
  [lmsys/Qwen3-235B-A22B-EAGLE3](https://huggingface.co/lmsys/Qwen3-235B-A22B-EAGLE3),
  [nvidia/Qwen3-235B-A22B-Eagle3](https://huggingface.co/nvidia/Qwen3-235B-A22B-Eagle3). No head
  for our exact served variants. **Confidence: high.**
- **CC ≥ 7.0 in theory (Volta OK), but degraded in practice:** EAGLE-3's edge is *tree* attention
  via custom tree-mask kernels routed through FlashInfer (sm_75+); on the fork you'd fall back to a
  **chain (non-tree) EAGLE on FLASH_ATTN_V100, losing part of the speedup.** Realistic Qwen3 EAGLE3
  is **~1.4–1.9× (accept-len ~2)** even on supported HW. **Confidence: med.**
  [Red Hat EAGLE3](https://developers.redhat.com/articles/2025/07/01/fly-eagle3-fly-faster-inference-vllm-speculative-decoding).
- Net: **lower priority than MTP** (no native head for our models + tree-kernel degradation on Volta).

### Medusa
- Multi-head approach, **superseded by EAGLE** in current vLLM guidance; no Qwen3.5/3.6 heads.
  **Not recommended.** **Confidence: med-high.**

### DFlash (block-diffusion draft, Qwen3.5) — watch, don't deploy
- On-box `dflash.py` present; drafts a whole block via a lightweight block-diffusion model. Reported
  **5.2× HumanEval / 3.0× MT-Bench on B200/SGLang**, but **B200-demonstrated** and the fork marks it
  *experimental/research*. Promising long-term, **not production on V100 today.** **Confidence: med
  (low for V100).** [kaitchup DFlash](https://kaitchup.substack.com/p/dflash-for-qwen35-eagle-for-gemma).

### Ranking — single-stream (`max_num_seqs=1`) document drafting

| Rank | Method | sm_70? | Safe on 3.5/3.6-GDN? | Expected decode × (batch=1, V100) | Conf |
|---|---|---|---|---|---|
| **1** | **MTP** (built-in head) | **Yes — fork-validated** | **Yes (decoupled)** | **~1.7–2.2×** | high |
| 2 | ngram (prompt-lookup) | Yes (kernel-free) | **No** — ✅ 122B full-attn only | ~1.5–2.5× on repetitive text | med |
| 3 | suffix (Arctic) | Yes (CPU proposer) | Likely unsafe — ✅ 122B only | ~1.5–3× on repetitive | med |
| 4 | EAGLE3 | Partial (chain not tree) | No head for our models | ~1.4–1.9× (Qwen3 proxy) | med |
| 5 | draft-model | Yes (force non-FlashInfer) | n/a | ~1.2–2× (weak for MoE) | low-med |
| — | Medusa / DFlash | Yes / on-box | superseded / experimental | — | — |

**Bottom line:** **MTP is the default** for the A3B-class targets (only method that is *both*
Volta-validated *and* GDN-correct, weights are free in the checkpoint). For the **122B Qwen3
full-attention** target, **ngram/suffix become viable** and may beat MTP on heavily-quoted
boilerplate — A/B them. Start MTP-1; try 2–3 if acceptance holds.

---

## 2. Quantization for SPEED on Volta (decode tok/s, not just fit)

**On-box reality (verified):** the fork's AWQ path is **TurboMind-derived, not Marlin.** The warmup
module `model_executor/warmup/awq_sm70_warmup.py` literally "primes the **TurboMind** small-shape
autotuner," and there are dedicated sm_70 ops: `awq_sm70_moe.py`, a single-token compact MoE op
(`awq_moe_single_token_sm70_out`, gated by `VLLM_SM70_AWQ_ENABLE_SINGLE_TOKEN_COMPACT`), and a
persistent **GEMM-shape LUT cache** (`VLLM_SM70_GEMM_LUT_PATH`, `sm70_gemm_import_cache`/`export_cache`).
This is real Volta-specific speed engineering — and it means **if a log/flag says "marlin," that is a
naming/repack artifact; the kernel executing on Volta is the TurboMind sm_70 path.**

### Marlin / GPTQ-Marlin — DO NOT run on Volta
- Marlin's core MMA uses PTX `m16n8k16`, which **requires `.target sm_80`+** → **does not compile on
  sm_70 (or even sm_75).** The famous "10.9× AWQ / 2.6× GPTQ" speedups are **sm_80+ only.**
  **Confidence: high.** [llm-awq #204](https://github.com/mit-han-lab/llm-awq/issues/204),
  [MARLIN paper](https://arxiv.org/pdf/2408.11743).

### INT8 W8A8 — no Volta tensor-core path, unlikely to win
- V100 tensor cores are **FP16-accumulate only, no INT8 IMMA**; sm_70 INT8 is limited to **dp4a on
  the SIMT path**, not tensor cores. vLLM lists W8A8 as Hopper/Ada-official with Turing/Ampere getting
  only W8A16-via-Marlin → **no sm_70 W8A8 path**, and W8A8's whole point (INT8 tensor-core throughput)
  doesn't exist on Volta. **Confidence: high** (no INT8 TC on sm_70) / **med-high** (therefore not
  faster than AWQ-4bit). [vLLM INT8 docs](https://docs.vllm.ai/en/stable/features/quantization/int8/).

### FP8 — fit/KV only, not a faster decode-compute path
- No native sm_70 FP8. On Volta FP8 is an **fp8_e5m2 KV-cache** trick (saves KV memory → longer
  context / bigger batch), **not a faster decode compute path.** A tiny dense micro-bench shows FP8
  marginally above FP16 only because it's KV-bound on a 0.6B model — **no evidence of an FP8 decode
  win on our MoE targets.** **Confidence: high.**

### Winner = AWQ-4bit (TurboMind sm_70)
- For memory-bandwidth-bound batch=1 decode, **4-bit weight-only is the throughput king**: ¼ the
  weight bytes per token, dequantized into V100 FP16 tensor cores. Our own measured numbers
  corroborate the *relative* win: **35B-A3B-AWQ ≈ 121 tok/s vs ≈ 82 GGUF (+48%)**, **122B-A10B-AWQ
  ≈ 77 (+41%)**, **27B ≈ 81**. **Confidence: high** on our numbers;
  external V100 MoE-AWQ tok/s benchmarks are scarce (see gap note).

### KV-cache quant for speed/quality (llama.cpp lane)
- In llama.cpp, quantized KV **requires `-fa`**; **symmetric q8_0/q8_0 (or q4_0/q4_0)** stays on the
  fast fused path, **asymmetric K≠V drops to a slow path.** The mma kernel converts quantized KV to
  F16 internally → works on V100; batch-1 decode routes to the `vec` kernel so the penalty stays small.
  **Recommendation: q8_0/q8_0 symmetric + `-fa`** (quality≈f16, frees KV VRAM for context).
  **Confidence: high** on the rule; **med** on the exact V100 decode delta.
  [llama.cpp KV-quant #10378](https://github.com/ggml-org/llama.cpp/issues/10378).

**Quant bottom line:** AWQ-4bit (TurboMind sm_70) is the best single-stream decode quant on Volta.
Marlin/GPTQ-Marlin are sm_80-locked, INT8 W8A8 has no Volta TC path, FP8 helps fit not speed. The
biggest *additional* decode win on top of AWQ is **stacking MTP** (§1) — quant shrinks bytes/token,
MTP cuts sequential steps; they multiply.

---

## 3. Attention kernels on Volta

**On-box reality (verified):** `flash_attn_v100_cuda.cpython-312-x86_64-linux-gnu.so` is a **real
compiled CUDA extension**, loaded by `vllm/v1/attention/backends/flash_attn_v100.py`, which provides
**paged decode + dense/paged prefill** Volta kernels with a strict Triton/eager fallback policy
(`TritonAttentionImpl` base). Selecting it also flips a tuned `max_num_batched_tokens` default on sm_70
(`arg_utils.py`: cap (7,0) + ≥30 GiB + `VLLM_ATTENTION_BACKEND=FLASH_ATTN_V100`). The launch scripts
already use `--attention-backend FLASH_ATTN_V100` + `VLLM_SM70_ENABLE_LM_HEAD_FASTPATH=1`. So on the
vLLM lane we already have a purpose-built Volta FA — the untried work is on the **llama.cpp** lane and
in **not regressing** into the broken Triton path.

### llama.cpp `-fa` — the most important "it actually works now" finding
- **`-fa` is NOT a silent slow fallback on current mainline.** Since PR
  [#17505](https://github.com/ggml-org/llama.cpp/pull/17505) (merged 2025-12-03) the fused `mma`
  tensor-core FA kernel was extended to Volta (m8n8k4); before that V100 used the older but still-fused
  **WMMA** kernel. **Action: build llama.cpp from HEAD with `-DCMAKE_CUDA_ARCHITECTURES=70`** — a pinned
  pre-Dec-2025 build leaves V100 on the slower WMMA path. **Confidence: high.**
- **Measured V100 effect (PR thread, Tesla V100-SXM2, Q4): prefill clearly helped, decode neutral-to-
  slightly-slower at batch=1** (the dispatcher *intentionally* routes batch-1 decode away from tensor
  cores — "only faster for sufficiently large matrices"). e.g. Qwen3moe-30B-A3B Q4_K ≈ **1052 pp /
  139 tg**; PR [#16843](https://github.com/ggml-org/llama.cpp/pull/16843) showed up to **2.35× prefill**.
  **So `-fa` on V100 = big prefill/TTFT win, ~neutral decode.** **Confidence: high** (Q4-specific).

### What to AVOID on the vLLM lane (silent slow / hangs)
- **vLLM `TRITON_ATTN` HANGS on V100** (confirmed sm_70; v0.17.0 ViT path regression, PR #32183).
  Fix: stay on `FLASH_ATTN_V100`/XFORMERS/TORCH_SDPA, or `--skip-mm-profiling`. **`--enforce-eager`
  does NOT fix it.** **Confidence: high.** [vLLM #36357](https://github.com/vllm-project/vllm/issues/36357).
- **The "96 KB shared-memory prefill crash" is a *head_dim* problem, not a prompt-length problem** —
  Triton wants 98304 B vs Volta's 65536 B cap, triggered by large head_dim (Gemma 256/512). **Chunked
  prefill does NOT fix it.** Fix: smaller-head model or non-Triton backend. **Confidence: high.**
  [vLLM #38918](https://github.com/vllm-project/vllm/issues/38918). (Mirrors our notes'
  `FLASH_ATTN_V100 96KB prefill crash → use TRITON_ATTN` note — but note the *opposite* failure also
  exists: on our exact card `--enable-chunked-prefill --dtype float16` hits a Triton "mma layout only
  on Ampere" assertion, [#11352](https://github.com/vllm-project/vllm/issues/11352). The safe path is
  the custom `FLASH_ATTN_V100` backend, which is what the launch scripts already use.)
- **FlashAttention-2 and FlashInfer: not supported on sm_70** (FA2 hard-errors "Ampere or newer";
  FlashInfer is sm_75+). **Confidence: high.**
- **`flash-attention-legacy` (sirCamp)** is the only maintained PyTorch FA-v2 port for sm_70, but it's
  a **memory win, not throughput** (~1.13× prefill, **no decode** — falls back for q_len=1). Not worth
  wiring given the fork already has a better Volta FA. **Confidence: high.**

**Ranking — what actually accelerates attention on sm_70:** (1) `1Cat-vLLM FLASH_ATTN_V100` (already
in use); (2) llama.cpp `-fa` HEAD build (prefill win); (3) XFORMERS/TORCH_SDPA as safe vLLM baseline;
(✗) TRITON_ATTN / stock-Triton FA / FA2 / FlashInfer — avoid.

---

## 4. MoE-specific speedups

**On-box reality (verified):** the fork has sm_70 MoE quant kernels (`awq_sm70_moe.py`, single-token
compact MoE op). The untried levers are a **V100 fused-MoE tuning JSON**, the **TP-vs-EP / island
placement** decision, and **CPU expert-offload** to serve bigger MoEs than HBM allows.

### The single biggest *free* MoE win: a V100 fused-MoE tuning JSON
- vLLM's Triton `fused_moe` **does run on sm_70** (real grouped-GEMM kernel, not a Python per-expert
  loop), but it looks up a per-device config `E=…,N=…,device_name=…V100….json` which **does not exist
  upstream** → it warns *"Using default MoE config. Performance might be sub-optimal!"* and runs an
  untuned heuristic. **Generating one with `benchmark_moe.py` for our exact (E, N, dtype) per model is
  the biggest free MoE win on V100.** **Confidence: high.**
  [vLLM #17619](https://github.com/vllm-project/vllm/issues/17619),
  [vllm-moe-configs](https://github.com/MissionSquad/vllm-moe-configs).

### Expert parallelism (EP) — do NOT enable on this server
- vLLM's EP all-to-all without NVLink/DeepEP falls to a **"naive" backend the project calls "not
  suitable for production"**; optimized EP backends **require NVLink or InfiniBand**, and "systems
  without P2P connectivity will perform extremely poorly." Our cross-board fabric is **PCIe Gen3 x8
  (~6.5 GB/s)** — fatal for all-to-all. **Confidence: high.**
  [vLLM forum EP-without-NVLink](https://discuss.vllm.ai/t/expert-parallelism-all-to-all-communication-without-nvlink-and-deepep/2421).
- EP loses even on H100/NVLink at small scale (Qwen3.5-35B-A3B, 4×H100: `EP+TP=2` slower than plain
  `TP=2` at every concurrency). **(NOT V100.)** And EP's all-to-all only engages when DP>1, so on a
  single-board TP-only config `--enable-expert-parallel` is **a no-op or pure overhead.**
  **Confidence: high.** [jarvislabs EP](https://jarvislabs.ai/blog/expert-parallelism-mixed-strategies-vllm).
- **Practical: serve each MoE TP=4 inside one NVLink quad; never pass `--enable-expert-parallel`; for a
  too-big MoE use PP/layer-split across boards, never EP across PCIe.** **Confidence: high.**

### CPU/RAM offload of cold experts — the lever to serve a *bigger* MoE
- Governing math: decode tok/s ≈ memory-bandwidth ÷ active-bytes-per-token. Offloaded experts are
  gated by **CPU RAM bandwidth**, not the V100's 900 GB/s HBM2 (which only serves the GPU-resident
  attention/KV/router/shared-expert). **DDR5-12ch (~460 GB/s) vs DDR4-8ch (~200 GB/s) ≈ 2× — the
  decisive variable.** **Confidence: high.**
- **Mainline llama.cpp `-ot`/`--n-cpu-moe` is the most realistic V100 offload path** (builds and runs
  on Volta; our benchmark setup already runs sm_70 `-fa`/`-sm tensor` builds). Pattern:
  `-ot "\.ffn_.*_exps\.=CPU"` (keep attention+shared on GPU, push expert FFN to CPU) or `--n-cpu-moe N`.
  Proxies (NOT V100): gpt-oss-120B MXFP4 on RTX-4080 16 GB, 78% CPU offload **12.45 t/s**; EPYC-only
  DeepSeek 671B Q4 **3.4–9.7 t/s** (NUMA-pinned higher). **Confidence: high it works; med on numbers.**
  [llama.cpp -ot discussion](https://github.com/ggml-org/llama.cpp/discussions/15396).
- **ktransformers: poor V100 fit.** Peak path needs **Intel AMX** (Sapphire Rapids); **EPYC has no
  AMX** (AVX-512/AVX2 fallback, slower), and its GPU side leans on Marlin/FlashInfer (sm_80+). Famous
  "20 tok/s" is peak; measured DeepSeek-V3 671B decode **~13.7 t/s on 4090 + Xeon-AMX (NOT V100).**
  **Confidence: high (NOT V100) / med-high (EPYC penalty).** [LMSYS KTransformers](https://www.lmsys.org/blog/2025-10-22-KTransformers/).
- **ik_llama.cpp: officially Turing+, but works on Volta and is often FASTER for CPU+GPU MoE offload —
  if past PR #1785 (2026-05-15).** Real V100-SXM2 numbers from that PR thread: 2×V100 qwen3.5-9B Q8
  pp4096 **1418→1922 t/s (+35%)**, tg −6.8%; single-V100+CPU-offload **Qwen3.5-122B-A10B Q4_K_M ≈167
  PP / 20 TG**, **Qwen3.5-397B-A17B ≈66–71 PP / 9–10 TG**. Earlier Volta builds had a quantized
  large-batch MUL_MAT overflow (→ `-inf` logits → crash on long prompts), fixed by PRs #1638/#1776/#1785.
  Multi-V100 gotcha: `--split-mode layer` garbles on asymmetric 32G+16G cards → use `--split-mode graph`
  or explicit `--tensor-split`. **Mainline is the safer Volta target; ik is the speed option to A/B.**
  **Confidence: high (real V100 numbers).** [ik PR #1785](https://github.com/ikawrakow/ik_llama.cpp/pull/1785),
  [ik PR #875](https://github.com/ikawrakow/ik_llama.cpp/pull/875).

**Synthesized offload decode bands (V100 + fast EPYC DDR; derived from math + nearest proxies, NO direct
V100 data — re-measure):** ~120B-A5B (gpt-oss) **~12–22 t/s** (DDR5) / ~6–10 (DDR4); GLM-4.5-Air ~A12B
**~8–12** / ~6–10; Qwen3-235B-A22B **~9–14** / ~5–8; DeepSeek 671B-A37B **~9–15** / ~5–8. **Confidence: med.**

---

## 5. Serving-level tuning (verify Volta)

**Framing:** stock vLLM dropped sm_70 build targets at ~0.20 (we use the 1Cat fork). The **V1 engine**
(default ≥0.8.0) changes defaults for chunked prefill, prefix caching, max-num-seqs, and CUDA-graph mode
— **confirm whether the fork runs V1 or V0** (`VLLM_USE_V1`); it determines half the answers below.

### CUDA graphs on Volta — we are probably leaving 10–28% on the table
- **Graphs WORK on sm_70 in the versions that run on Volta; there is NO blanket "must use
  `--enforce-eager` on V100" rule.** The validated 1Cat-vLLM TP4/TP2 profiles run **graphs ON.**
  **Confidence: high.**
- **Expected ~10–28% single-stream decode speedup** (eliminates per-step kernel-launch + sync overhead,
  a larger fraction of step time at batch=1 — our regime). No V100 number exists; transfers
  mechanistically. **Confidence: med.**
- **Why people reflexively `--enforce-eager`:** it's the documented crash-bisection flag (copied as
  folklore); capture pre-allocates ~1.5–2 GB; old sm_70 graph bugs (since patched); AWQ-Marlin illegal-
  memory under graphs (real but **sm_90/H100**, not V100). **Action: drop reflexive `--enforce-eager`;
  use it only as a crash/OOM valve.** On Volta you get **PIECEWISE graphs** (attention eager, rest
  graphed — FULL mode needs FA3/sm_90); set **`cudagraph_capture_sizes=[1,2]`** for the single-stream
  lane to reclaim VRAM while keeping the speedup. **Confidence: high (works) / med (magnitude).**

### `--max-num-seqs` — set LOW on single-stream lanes
- Pure host-side scheduler knob (not sm_80-gated). At batch-1 it does **not** change decode tok/s, but
  setting it **low (1–4)** wins indirectly: V1 sizes KV budget as `max_num_seqs × max_model_len` (a high
  ceiling steals VRAM / shortens max context) and CUDA-graph capture defaults up to `max_num_seqs` (a
  high ceiling = more captured graphs + longer warmup). **Pour freed VRAM into context via higher
  `--gpu-memory-utilization`.** **PP gotcha:** on PP lanes keep it **above the PP stage count** or later
  stages starve. **Confidence: high.**

### Chunked prefill
- Scheduler feature (tiles the *token-budget* axis), not a memory fix. **V0 default OFF; V1 always ON
  and cannot be disabled.** For **single-stream long-context the benefit is small** (no concurrent
  decodes to interleave) and you pay KV re-read per chunk; it does **not** fix the 96 KB head_dim crash,
  and on our exact card `--enable-chunked-prefill --dtype float16` hits a Triton MMA assertion (#11352).
  **Prefer chunked-prefill-OFF (V0) for the single-stream lane; if forced to V1, set
  `--max-num-batched-tokens` high + `--max-num-seqs` low.** **Confidence: high.**

### Prefix caching — high ROI for repeated long prompts
- `--enable-prefix-caching` (default-ON in V1) skips prefill for shared prefixes; **absolute ms saved
  is larger on slow V100 prefill.** Skepticism flag: the old V0 impl cost −36.7% with no shared prefix
  (fixed v0.6.5) — only trust "free" on V1. **Verify a nonzero `gpu_prefix_cache_hit_rate` actually
  appears on the custom `FLASH_ATTN_V100`/XFORMERS backend** (class of silent-0 bugs). **Confidence:
  high / med (V100 hit-rate confirmation).**

### NCCL / topology
- **Biggest lever: TP inside each NVLink island, PP across the PCIe gap** (vLLM docs say this directly).
  TP all-reduces *every layer* (hammers 6.5 GB/s); PP passes *one activation per stage boundary*.
  **Never straddle a TP group across PCIe.** Decode all-reduce is tiny (~256 KiB) and latency-bound, so
  decode tuning is a latency problem → benchmark **`NCCL_PROTO=LL` vs `LL128`**; leave `NCCL_ALGO` and
  other vars default. vLLM custom all-reduce is NVLink-only and **auto-disables with >2 PCIe GPUs**.
  **Confidence: high** (placement) / **med** (no direct V100 NCCL-flip tok/s data).
- ✗ **`NCCL_NVLS_ENABLE`/NVLS** needs 3rd-gen NVSwitch + sm_90 — self-disables on SXM2 V100. ✗ NCCL
  symmetric-memory (2.27+) unsupported CC ≤ 7.5. **Confidence: high.**

### Multi-LoRA — the old wall is gone, but unverified on V100
- The "Punica needs sm_80, no LoRA on V100" wall was **removed** (PR #5036, Aug 2024 → Triton
  SGMV/BGMV, hardware-agnostic; v0.8.0 moved LoRA to V1 Triton). Asterisk: Triton runs the **FMA
  fallback (no tensor cores) on sm_70**, so per-token LoRA cost is higher than Ampere (painful case =
  batched multi-adapter; our single-stream single-adapter use minimizes it). **Genuine gap: no
  post-2024 report confirms multi-LoRA actually serving on V100; 1Cat doesn't mention `--enable-lora`.
  Smoke-test a rank-matched adapter before trusting it.** **Confidence: high (wall gone) / med (V100 LoRA).**

---

## 6. The "haven't-been-tried" experiment list for THIS server — ranked by upside ÷ effort

Anchors today (verified on-box, but **note provenance**): the `1cat-bench` numbers
(35B-A3B-AWQ **121 tok/s**, 122B-A10B-AWQ **77**, 27B **81**) were captured on **another machine,
TP2** (`launch_27b/35b.sh` → `CUDA_VISIBLE_DEVICES=1,2`, `--tensor-parallel-size 2`), *not* a 4-card board of the V100
server. The task brief calls these "the V100 server TP4" anchors. **Action E0 below: re-measure on an NVLink quad of the
V100 server with `/v1/models` verified to settle the discrepancy before attributing any speedup.**

| # | Experiment | Lane | Expected upside (batch=1 decode) | Effort | Conf | Risk / gate |
|---|---|---|---|---|---|---|
| **E1** | **MTP-1 spec-decode** on 35B-A3B-AWQ (then 122B): `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`, sweep 1→3 | vLLM (1Cat) | **~1.7–2.2×** | Low | high | watch acceptance; GDN-safe |
| **E2** | **Drop `--enforce-eager`, PIECEWISE CUDA graphs**, `cudagraph_capture_sizes=[1,2]` | vLLM | **~1.1–1.3×** + VRAM back | Low | high(works)/med(mag) | bisect if a big AWQ MoE crashes |
| **E3** | **Generate V100 `fused_moe` tuning JSON** (`benchmark_moe.py`) per model E,N,dtype | vLLM | kills "sub-optimal MoE" penalty (single-digit %–?) | Low-med | high | one-time per model/quant |
| **E4** | **ngram + suffix on the 122B full-attention target** (A/B vs MTP) | vLLM | **~1.5–2.5×** on boilerplate | Low | med | **122B ONLY — corrupts GDN** |
| **E5** | **llama.cpp HEAD `-fa` + q8_0/q8_0 KV** rebuild (`CUDA_ARCH=70`) | llama.cpp | big **prefill/TTFT** win, neutral decode | Low-med | high | decode unchanged at batch=1 |
| **E6** | **`--max-num-seqs 1–4` + raise `--gpu-memory-utilization`**, chunked-prefill OFF (V0) | vLLM | longer context / less warmup (not raw tok/s) | Low | high | keep > PP stages on PP lanes |
| **E7** | **Prefix caching ON + verify nonzero hit-rate** on FLASH_ATTN_V100 | vLLM | skips repeated-prompt prefill (TTFT) | Low | high/med | confirm hit-rate not silent-0 |
| **E8** | **TP-in-island + PP-across-PCIe** to serve a bigger MoE; `NCCL_PROTO=LL` vs `LL128` A/B | vLLM | enables bigger model; small decode-latency % | Med | high(placement)/med(NCCL) | PP bubbles at low concurrency |
| **E9** | **CPU expert-offload** of a 120B–235B MoE via mainline llama.cpp `-ot`/`--n-cpu-moe`, NUMA-pinned | llama.cpp | serve **bigger** MoE at ~9–15 t/s | Med | high(works)/med(num) | gated by EPYC DDR bandwidth |
| **E10** | **ik_llama.cpp ≥ PR #1785** for CPU+GPU MoE offload A/B vs mainline | ik_llama | often **faster** than mainline on V100 | Med | high(V100 data) | unofficial on Volta; `--split-mode graph` |
| **E11** | **EAGLE3 on 122B** using the lmsys/nvidia 235B-A22B head as a proxy | vLLM | ~1.4–1.9× *if* it loads | Med-high | low-med | no exact head; chain-not-tree on Volta |
| **E12** | **Multi-LoRA smoke test** on V100 (rank-matched adapter) | vLLM | enables per-task adapters | Low | med | Triton-FMA; may silent-ignore |
| ✗ | EP / DeepEP / Marlin / INT8-W8A8 / native-FP8 / FlashInfer / TRITON_ATTN / NVLS | — | **none on Volta** | — | high | see kill-list |

**Top 3 to run first (highest upside ÷ effort, all on the existing 1Cat-vLLM lane):** **E1 (MTP)**,
**E2 (CUDA graphs)**, **E3 (MoE tuning JSON)** — E1 and E2 *multiply* (MTP cuts steps; graphs cut
per-step overhead) on top of the AWQ-4bit byte savings already in place. Settle **E0** (provenance)
first so the multipliers are measured against the correct v100server-TP4 baseline.

---

## 7. Explicit "needs sm_80+ / sm_90 / NVSwitch — WON'T HELP VOLTA" kill-list

1. **FlashAttention-2** (`flash-attn`, vLLM `FLASH_ATTN`) — sm_80+ (hard error "Ampere or newer").
2. **FlashInfer** backend — sm_75+ fast paths; "remove on SM_70/SM_75."
3. **vLLM `TRITON_ATTN`** — not "needs sm_80" but **HANGS on sm_70**; use FLASH_ATTN_V100/XFORMERS.
4. **Stock Triton tutorial flash-attention** — hangs on Volta (mma layout); FMA fallback otherwise.
5. **FULL CUDA-graph mode** (attention in-graph) — needs FA3 = sm_90; V100 capped at PIECEWISE.
6. **Marlin / GPTQ-Marlin / FP8-Marlin** — sm_80+. **Machete** — sm_90. **CUTLASS-MoE / DeepGEMM /
   FlashInfer-TRTLLM-MoE** — sm_90/sm_100. V100 → Triton fallback. *(The fork's "AWQ-Marlin" is really
   TurboMind sm_70 — see §2.)*
7. **DeepEP** (+ DeepGEMM, NVSHMEM/IBGDA) — Hopper sm_90; won't build for sm_70.
8. **`--enable-expert-parallel` over PCIe / TP-only** — "naive" all-to-all "not suitable for production."
9. **`NCCL_NVLS_ENABLE` / NVLS (NVLink SHARP)** — 3rd-gen NVSwitch + sm_90; self-disables on SXM2.
10. **NCCL symmetric-memory kernels (2.27+) / vLLM SymmMemCommunicator** — unsupported CC ≤ 7.5.
11. **Native bfloat16 compute** — sm_80+; V100 casts bf16→fp16 (verify model tolerance).
12. **Native FP8 compute decode** — no sm_70 FP8; FP8 is fit/KV only, not faster than AWQ-4bit.
13. **INT8 W8A8 tensor-core path** — V100 has no INT8 IMMA (dp4a SIMT only); W8A8 is Hopper/Ada.
14. **Punica CUDA LoRA kernels** — required CC ≥ 8.0; obsolete (Triton replaced it) — gate gone, but a
    pre-Aug-2024 vLLM will still refuse LoRA on V100.
15. **ik_llama.cpp's *officially supported* GPU tier** — "Turing or newer" on paper (works in practice;
    DeepSeek-MLA FA is Turing+).
16. **ktransformers peak (AMX) kernels** — Intel Sapphire-Rapids AMX; EPYC has none (AVX fallback).
17. **Stock vLLM wheels ≥ 0.20** — sm_70 dropped from build targets; use the 1Cat fork / ≤0.18.x / source.

---

## 8. Honest evidence gaps (drive the experiments, don't trust the bands)

- **No public V100-specific tok/s** exists for CUDA-graph-vs-eager, NCCL-proto flips, batch-1
  max-num-seqs, MoE-with-offload, or MoE-AWQ generally. Every concrete multiplier above **transfers
  mechanistically from newer cards** — decision-grade numbers must come from A/B-ing on our own box
  (eval-provenance rule: verify `/v1/models`, record endpoint/SHA/params/seed).
- **Provenance discrepancy (E0):** the existing anchors are **another machine's TP2**, not **the V100 server TP4** — re-measure.
- **GDN-safety of `suffix`** is unconfirmed (likely shares the ngram #39273 bug on 3.5/3.6).
- **No EAGLE/MTP head for the exact 122B variant** we serve (235B-A22B head is the closest proxy).
- **No on-record multi-LoRA-on-V100 confirmation** post-Punica-removal — code wall gone, smoke test required.
- **Chinese-language V100-MoE tok/s** (Zhihu/CSDN/bilibili) did not surface in US-region search and may
  exist behind that wall — a CN-language pass is a worthwhile follow-up for the community-research lane.

---

### Sources (primary)
- 1Cat-vLLM fork: https://github.com/1CatAI/1Cat-vLLM • ch2lab/vllm-v100: https://github.com/ch2lab/vllm-v100
- GDN spec-decode rollback bug: https://github.com/vllm-project/vllm/issues/39273
- vLLM Qwen3.5 recipe (MTP): https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html
- MTP-as-draft analysis: https://zolotukhin.ai/blog/2026-05-08-why-mtp-heads-are-the-speculative-decode-draft-qwen3-a3b-deserves/
- vLLM spec-decode blog: https://blog.vllm.ai/2024/10/17/spec-decode.html • Suffix/Arctic: https://www.snowflake.com/en/engineering-blog/suffixdecoding-arctic-inference-vllm/
- EAGLE3 heads: https://huggingface.co/AngelSlim/Qwen3-a3B_eagle3 • https://huggingface.co/lmsys/Qwen3-235B-A22B-EAGLE3 • Red Hat EAGLE3: https://developers.redhat.com/articles/2025/07/01/fly-eagle3-fly-faster-inference-vllm-speculative-decoding
- Marlin sm_80 gate: https://github.com/mit-han-lab/llm-awq/issues/204 • https://arxiv.org/pdf/2408.11743
- vLLM INT8 hardware: https://docs.vllm.ai/en/stable/features/quantization/int8/
- llama.cpp Volta FA: https://github.com/ggml-org/llama.cpp/pull/17505 • https://github.com/ggml-org/llama.cpp/pull/16843 • KV-quant: https://github.com/ggml-org/llama.cpp/issues/10378
- vLLM TRITON_ATTN hang: https://github.com/vllm-project/vllm/issues/36357 • 96KB head_dim: https://github.com/vllm-project/vllm/issues/38918 • float16 chunked-prefill MMA: https://github.com/vllm-project/vllm/issues/11352
- fused_moe V100 config: https://github.com/vllm-project/vllm/issues/17619 • https://github.com/MissionSquad/vllm-moe-configs
- EP without NVLink: https://discuss.vllm.ai/t/expert-parallelism-all-to-all-communication-without-nvlink-and-deepep/2421 • https://jarvislabs.ai/blog/expert-parallelism-mixed-strategies-vllm
- llama.cpp expert offload: https://github.com/ggml-org/llama.cpp/discussions/15396 • KTransformers: https://www.lmsys.org/blog/2025-10-22-KTransformers/
- ik_llama V100: https://github.com/ikawrakow/ik_llama.cpp/pull/1785 • https://github.com/ikawrakow/ik_llama.cpp/pull/875
- DeepEP sm_90: https://github.com/deepseek-ai/DeepEP • vLLM sm_70 drop @0.20: https://discuss.vllm.ai/t/support-for-v100-sm-70-on-vllm-0-20/2605

*Compiled 2026-06-13. No system changes made; on-box facts inspected
read-only. Re-measure every multiplier on a TP4 NVLink quad of the V100 server before relying on it.*
