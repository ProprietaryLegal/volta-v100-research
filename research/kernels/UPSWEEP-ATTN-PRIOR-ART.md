# ATTN-PRIOR-ART: how sm70 tensor-core attention kernels handle extreme logit scales

> **Origin.** PLI Labs research archive, file `kernel-research/UPSWEEP-ATTN-PRIOR-ART.md`. Published 2026-11-18 in the Volta V100 research
> repository.
> **Edited for publication.** Machine names, private file paths, network addresses, session names and the
> operator's name were replaced with generic terms (`the V100 server`, `[private path]`, `[address]`,
> `the owner`). Links to files that are not published were turned into plain text. No figure and no finding
> was changed. Dates inside the note are its own; later results are in the build pages of this repository.

UPSWEEP, 2026-11-01 (about 17:30Z). Research only. No the V100 server GPU, no builds; the V100 server reads were read-only.

Question from main (16:57Z): KATT1's HMMA decode attention kernel (patch 0012) failed R6 numerics on three
q std 2000 cases. How do existing sm70 tensor-core attention kernels stay accurate at extreme logit scales
without losing speed, and what can KATT1 adopt, at what cost in microseconds?

Tags: **VERIFIED** = I or a helper read the code at the named commit. **INFERRED** = arithmetic or reasoning
from verified code, not measured. **UNVERIFIED** = from memory or a title, code not read in this pass.
External code was read through `gh api` at pinned short shas. Fetched copies are in
`[private path]`
and `.../scratchpad/pa-B/src/`. They were never executed.

Commits used:

| Repo | Commit | Notes |
|---|---|---|
| NVIDIA/cutlass | e13f5cd414 | examples/41 |
| facebookresearch/xformers | 3e62181f37 | Last commit before 3d947a67fa removed the cutlass FMHA backend; head b5410d3bb4 no longer has it. |
| Dao-AILab/flash-attention | v1.0.9 6d48e14a6c, v2.0.0 4f285b3547, main 4616f5116d | |
| zhinianqin/flash-attention-v100 | c2eda5e611 | This is exactly the engine's pin `VLLM_FLASH_ATTN_SM70_COMMIT` (the V100 server `[private path]` `cmake/external_projects/vllm_flash_attn.cmake:39`, engine 23cec6a). |
| ai-bond/flash-attention-v100 | 0ababf82e3 | |
| ZRayZzz/flash-attention-v100 | 05eace2700 | |
| xormal/V100-SM_70-Flash-Attn-2-v1 | 05c632aa8e | |
| InternLM/lmdeploy | dd4db0da27 | |
| ggml-org/llama.cpp | dde879bc4f | |
| Ootomo wmma_extension | a9e6b34e40 | |
| cuMpSGEMM | 42dd92ea1c | |
| 1Cat #1277 | 899e9ed331 | Local copies in scratchpad `up/v41src/`. |
| KATT1 worktree | head 96db5a1 | `[private path]`. The kernel file has an uncommitted "version 2" edit (mtime 17:17Z). |

---

## 0. Bottom line

1. **The R6 failure is in the per-block score path, not in the P precision or the split merge.**
   - All three failing cases are T16 v640 q std 2000 at split_keys 32, 64 and 128, and the only failing check is
     `matches_emulation` (kernel vs CPU emulation 4.5 ulps). VERIFIED (`slot.json`, below).
   - All three splits give identical numbers, and so does the no-P-split variant. So the error comes before the
     split merge and before P. INFERRED.
2. **No surveyed sm70 kernel does anything special for extreme logits.**
   - Every tensor-core kernel (cutlass/xformers, all three FA-V100 forks, llama.cpp MMA, TurboMind prefill, 1Cat)
     chains QK through the tensor-core C operand. VERIFIED.
   - The two production decode stacks avoid the problem by not using tensor cores for Volta decode QK. TurboMind
     runs FP32 SIMT for all sm70 decode, including the 576-dim MLA decode. llama.cpp routes Volta decode to its
     VEC or TILE kernels. VERIFIED.
3. **The one direct prior art for V100 accumulator rounding is Ootomo's TCEC `mma_rn_sync`.**
   - Each k-block is computed with C=0 and added outside the tensor core with FP32 round-to-nearest adds.
     VERIFIED, `tcec.hpp` L672-694.
   - That is what KATT1's uncommitted version 2 already does: `DS41_HMMA_QK_CHAIN` default 1, and
     `DS41_HMMA_PV_FRESH` default 1. VERIFIED, KATT1 worktree `ds41_decode_hmma_sm70.cu` L26-36, L254-282.
4. **The best cheap addition is TurboMind's exp2 FFMA fold, at about 0 µs.**
   - Keep the running max in raw units and compute `p = exp2f(fmaf(S_raw, c, -mc))`, with `c = scale*log2e` and
     `mc = fl(m*c)`. TurboMind `impl_simt.h` L478, L495; FA-V100 c2eda5e611 `softmax.h` L72-79. VERIFIED.
   - This removes the per-element rounding of `a * scale` at |s| about 8.4e3, where the FP32 ulp is 2^-10.
     INFERRED.
   - Doing the rescale and the merge as `exp2(mc_old - mc_new)` keeps the offsets exactly consistent. No surveyed
     kernel does this. INFERRED.
5. **Do not copy the prior art's shortcuts.** VERIFIED in the listed code.
   - fast-math builds (TurboMind, xformers, all FA-V100 forks).
   - Q pre-scaled in fp16 (llama.cpp MMA, xormal, ZRayZzz).
   - fp16 raw scores (ZRayZzz; engine sm70_79t).
   - fp16 O partials (1Cat `attn_split`).
   - fp16 output accumulators with max offsets (llama.cpp).
   - LSE-form merges at large LSE.

   KATT1's FP32 S, P hi/lo split and FP32 (m,l) merge already exceed every surveyed kernel.

---

## 1. What failed in R6

Sources: `R6-owner/run-20261101T163248Z/slot-KATT1-hmma.out` and the V100 server
`[...]/slot.json` (read-only). VERIFIED.

- `summary.numerics_problems` lists exactly `T16 v640 q2000.0 ks32/ks64/ks128: matches_emulation`.
- The `ulp_max`, `ulp_mean`, cos, KL and finiteness checks all passed for those cases. The extreme cases are gated
  more loosely; KATT1 report §3.2.
- Kernel vs FP64 at q std 2000, max/mean ulps:

| T | today | 0012 (all ks) | emulation diff | #1277 comparator |
|---|---|---|---|---|
| 2 | 0.00/0.000 | 0.00/0.000 | 0.00 | 0.00 |
| 6 | 0.85/0.004 | 1.28/0.005 | 0.001 | 2.63 |
| 12 | 1.33/0.005 | 0.53/0.005 | 0.125 | 2.67 |
| 16 | 0.64/0.002 | **2.75**/0.004 | **4.5** | 9.63 |
| 18 | 0.50/0.002 | 0.50/0.002 | 1.0 | 0.93 |

- `ks64_nosplit` (no P split) gives the same vs-FP64 numbers as ks64 at every q std 2000 T. That is why the m2
  mutation is MISSED on all five q std 2000 rows. VERIFIED.
- At normal scale (q std 1 and 4), 0012 equals today (0.50-0.51 max ulps) and m2 is caught 117/135. VERIFIED.

Reading (INFERRED):
- At q std 2000 the softmax is nearly one-hot. Each p is either about 1 or below fp16 resolution, so the hi/lo
  split cannot matter. The m2 teeth are structurally impossible on these rows, and the gate should not ask for
  them there.
- What moves the output is a near tie between the top keys, where the output is sensitive to the score difference.
  Since ks does not matter, the drift is in the per-block QK scores (or the emulation's model of them).
- The emulation disagrees with the hardware by 4.5 ulps only where the score error is largest. That fits an
  imperfect model of V100 HMMA internals (alignment and truncation inside each k=4 step), which a 32-step chain
  amplifies.

### 1.1 Error budget at q std 2000 (INFERRED arithmetic)

**Setup.**
- Raw |q·k| reaches 1.9e5, where the FP32 ulp is 2^-6 = 0.0156.
- The scaled logit reaches |s| = 1.9e5 × 512^-0.5 ≈ 8.4e3, where the FP32 ulp is 2^-10 ≈ 9.8e-4.
- Products of fp16 Q and fp16 K are exact inside the HMMA.
- The 0012 K rows (E2M1 × E4M3) are exact in fp16. VERIFIED: KATT1 report, and `pack_half2` of exact
  products, kernel L240-243.

**Version 1** (one chain per warp output):
- 8 WMMA k16 steps = 32 HMMA k=4 steps, each rounding toward zero, at partial magnitudes up to about |S|.
- Worst-case bias is tens of ulp(S), about 0.1-0.5 raw, which is 4e-3 to 2e-2 in scaled logit.
- Then 3 round-to-nearest cross-warp adds, and one rounding of `a * scale` (≤ 0.5 ulp(s) = 4.9e-4 logit).

**Version 2** (KATT1 worktree defaults: QK_CHAIN 1, PV_FRESH 1):
- Each k16 step starts from zero, so there are 4 RZ steps at partial magnitude about |S|/sqrt(32) for random
  data. The 8 chain results are summed with round-to-nearest, then the 3 cross-warp adds.
- Expected score error is about 1-2 ulp(S), so 0.7-1.4e-3 logit.
- The `a * scale` rounding (≤ 4.9e-4) is then about a quarter to a third of the remaining logit error.
- KATT1's own CPU model says about 4x smaller FP32 score error. VERIFIED that the comment says so, kernel L34-36.

**Output sensitivity.**
- For a top pair with weights near 0.5, d(out)/out is about 0.25 × δ × |v1 - v2| / |out|.
- So δ = 1e-3 gives about 0.25-1 fp16 ulp, and δ = 1e-2 gives several ulps. That matches 2.75 ulps at T16 for
  version 1.

---

## 2. Per-implementation survey

Columns: (a) where scale and log2e are applied; (b) QK accumulator on sm70 and how it chains; (c) running max;
(d) P precision before P·V; (e) split combine; (f) attention sink; (g) large-logit or Volta numerics notes.

### 2.1 NVIDIA cutlass examples/41 (fused MHA), e13f5cd414. All VERIFIED (helper read).

- **a.** Without bias, `iterative_softmax` multiplies the FP32 accumulator by `scaling*kLog2e` (`kernel_forward.h`
  L1186, L1191). It takes the max of the scaled values, then `exp2f(frag - mi)` (L1262). With bias,
  `accum *= p.scale` comes first (L816-819), so there are two roundings. Each element is rounded at the scaled
  magnitude, which is the same weakness as 0012's `a * scale`.
- **b.** Sm70 f16 uses TensorOp `GemmShape<8,8,4>` (`gemm_kernel_utils.h` L192-201), i.e. `mma.sync m8n8k4
  f32.f16.f16.f32` (`mma_sm70.h` L513), with `accum_t = float` (L137). QK clears the accumulator then chains
  through C (L793, L799). f32 and bf16 on Sm70 fall back to SIMT (L150-159).
- **c.** Starts at -inf (L670-671). When all keys are masked by the bias it sets `mi = 0` temporarily
  (L1232-1240). No max offset.
- **d.** Single fp16, `half_t(accum)` = `__float2half_rn` (`mma_from_smem.h` L1754; `half.h` L195-197). The sum
  uses unrounded FP32 p.
- **e.** No forward split-KV. With large head dim, each key block's P·V starts from a cleared accumulator and is
  merged in FP32 outside the tensor core (`epilogue_rescale_output.h` L166-172). This is a P·V analogue of
  "fresh chains". LSE = `mi/kLog2e + log(s_prime)` (L1141-1144).
- **f.** No sinks.
- **g.** Only a NaN guard.

### 2.2 xformers cutlass FMHA, 3e62181f37 (removed at 3d947a67fa). VERIFIED.

- **a-d.** Same code as cutlass 2.1: scale at L829-832 and L1246/L1251; Volta P at `mma_from_smem.h` L1755;
  m8n8k4 at `gemm_kernel_utils.h` L174-183.
- Sm70 kernels are generated for f16 and f32 only (`generate_kernels.py` L120-121), built with
  `--use_fast_math` (`setup.py` L456).
- **e.** `num_splits_key` is backward-only (`kernel_backward.h` L715, L2065-2121).
- The removed SIMT `decoder.cu` (74d817262e) scales each thread's partial dot before the reduce (L273), starts
  the max at a finite `lowest()` (L248), and runs a two-pass softmax with `__expf` (L328, L347).
- **f, g.** No sinks, no large-logit notes.

### 2.3 Dao-AILab flash-attention: no sm70 in any upstream version. VERIFIED.

- v1.0.9 compiles sm_75/80/90 and checks `is_sm90||is_sm8x||is_sm75` (`setup.py` L114-120;
  `fmha_api.cpp` L207-211).
- v2.0.0 targets sm_80/90 (`setup.py` L115-120).
- main defaults to "80;90;100;110;120" and says "Ampere GPUs or newer" (`flash_api.cpp` L393-394).
- Upstream `softmax.h` L76-89 documents the FFMA exponent form and an `UNFUSE_FMA` switch.

### 2.4 zhinianqin/flash-attention-v100, c2eda5e611. This is the engine's FA pin. VERIFIED.

- **a.** The max is taken over raw scores. `max_scaled = fl(max*scale_log2)`, then
  `exp2f(x*scale_log2 - max_scaled)` contracts to one FFMA (`softmax.h` L70-79; the `UNFUSE_FMA` variant uses
  `__fmul_rn`). `sm70_scale_apply_exp2` simply forwards to this (L140-141; I read it). The tile rescale is
  `exp2f((m_prev - m_cur)*scale_log2)` (L196), which is not offset-consistent. Fast-math build
  (`CMakeLists.txt` L70).
- **b.** `SM70_8x8x4_F32F16F16F32_TN` with an FP32 accumulator (`kernel_traits.h` L100-103), chained through C
  (`utils.h` L139-158).
- **c.** The first tile is zero-initialised; `Check_inf` maps a -inf max to 0. No offset.
- **d.** Single fp16 `Element(acc_s)`, rounding mode INFERRED to be nearest. row_sum uses unrounded FP32
  (`flash_fwd_kernel.h` L330-333).
- **e.** Split-KV combine is LSE form, FP32, with `expf`/`logf` (L1196-1205, L1221, L1266-1271). At
  LSE ≈ 8.4e3 each split weight carries about 1e-3 absolute error. INFERRED.
- **f.** No sinks.
- **Engine use.** The engine's `csrc/attention/sm70_v37/tail.cu` calls
  `FLASH_NAMESPACE::sm70_scale_apply_exp2(scores, max, softmax_scale_log2)` (L329-330), with
  `softmax_scale * M_LOG2E` passed in at L853 and a -inf-safe rescale at L284-287. VERIFIED (engine 23cec6a,
  read-only). So the engine's own V100 FA path already uses the FFMA form.

### 2.5 ai-bond/flash-attention-v100, 0ababf82e3. VERIFIED.

- **a.** `acc *= SOFTMAX_SCALE` after the MMA (`mat_mul.h` L110, L126, L171-173), then
  `__expf(fmaxf(s - max, -80))` (`softmax.h` L103-106). Fast-math.
- **b.** PTX `wmma.mma.sync m16n16k16 .f32.f32` (`mma_m16n16k16.h` L625-646), chained through C (`mat_mul.h`
  L80-98). The comment calls FP32 accumulation "for numerical stability" (L622). This is the closest design to
  0012 v1.
- **c.** `NEG_INF = -1e30f`, finite (`kernel.h` L20). The exp argument is clamped at -80.
- **d.** `__float22half2_rn` (L143); the sum uses unrounded FP32. O lives in FP32 shared memory and is fed back
  as C (`mat_mul.h` L235-256).
- **e.** No split: `num_splits <= 1` is enforced (`forward_kvcache.cu` L465).
- **f.** No sinks.

### 2.6 ZRayZzz/flash-attention-v100, 05eace2700. VERIFIED.

- m8n8k4 with an FP32 accumulator (`fused_mha_kernel.cu` L804, L147-156), but scores are copied to fp16 shared
  memory (L108-111, L157). Raw 1.9e5 would overflow to inf.
- Max taken on raw S then multiplied by scale (L166); `__expf(acc*scale - max)` (L173).
- P, the rescale factor and 1/sum are all fp16 (L175, L185, L306).
- No split, no sinks.

### 2.7 xormal/V100-SM_70-Flash-Attn-2-v1, 05c632aa8e (8 stars). VERIFIED.

- Pre-scales Q in fp16 (`decode_hmma.cuh` L215).

### 2.8 LMDeploy TurboMind, dd4db0da27. VERIFIED (helper read; I re-read the softmax, merge, reduce and sink lines and the 576 instantiation).

- **Which kernel runs.** sm70 decode is SIMT, not tensor cores.
  - `kernel/decoding_sm70_{64,128,256,576}.cu` instantiate `Impl<MMA_SIMT,...>`. For the 576-dim MLA decode:
    `Mainloop<arch::Sm70, Impl<MMA_SIMT, T, Tkv, 2, 1, 16, 1, 1, 8, kHeadDim, 2>>` (L19).
  - Only prefill uses `MMA_884` (`attention_sm70_128.cu` L20-25).
  - The registry matches on decode mode (`registry.cu` L64-67); `Sm70` covers compute capability 700-749
    (`arch.h` L17-19).
- **a.** Q is not prescaled; it is stored as half after RoPE (`attention_universal.h` L201-247, L340).
  - Host side sets `inv_sqrt_dh = scaling*log2(e)` and `scale_sinks = scaling`
    (`unified_attention_layer.cc` L541-551).
  - Softmax on raw scores: `p = exp2f(S*qk_scale - M*qk_scale)` (`impl_simt.h` L495). Rescale is
    `exp2f((prev_M - M)*qk_scale)` (L478). M stays in raw units.
  - Fast-math build (attention `CMakeLists.txt` L15).
- **b.** With fp16 KV, `Tqk = float`. K is converted to float (L318-319), the dot is an FP32 FMA (L349), then an
  8-lane shuffle tree (L367-369). With int8/int4 KV, `Tqk = half`. Prefill uses
  `mma.sync.m8n8k4...f32.f16.f16.f32` (`core/mma.h` L19), chained through C (`impl_884.h` L224).
- **c.** Starts at -inf (`attention_universal.h` L450). Masked scores get `score -= inf`
  (`mainloop_sm70.h` L149). When M is -inf, p and the rescale are forced to 0 (`impl_simt.h` L479-481,
  L496-498, L559-561).
- **d.** P is rounded once to fp16 then used in an FP32 FMA with float V (`impl_simt.h` L513, L448). L sums
  unrounded p (L499-502).
- **e.** Each split stores unnormalised FP32 O plus raw (M, L) (`attention_universal.h` L551, L616).
  `reduce.cu` combines in FP32 with `exp2f((M_i - M)*exp_scale)` (L104-106, L150-152). (m,l) form, not LSE.
- **f.** Sinks are supported. In the last split only, and only if M is finite:
  `L += expf(sink - M*scale_sinks)` (`attention_universal.h` L529-536). Denominator only; the max is not updated.
  Added in d0b6355b0a "Initial gpt-oss support for turbomind (#3839)".
- **g.** No comments on large logits. The robustness comes from the design: FP32 SIMT, the exp2 FFMA fold, and
  FP32 everywhere except one P rounding.

### 2.9 llama.cpp CUDA flash attention, dde879bc4f. VERIFIED (helper read; constants re-read).

**Volta dispatch** (`fattn.cu` L691-699):
- VEC if head dim ≤ 256 and query rows × effective GQA ≤ 2.
- TILE if that product ≤ 16 (comment at L696: "On Volta tensor cores are only faster for sufficiently large
  matrices").
- MMA otherwise.
- Head dim 512 decode always goes to TILE (L623-630).
- The WMMA m16n16k16 kernel is gone: 2e1c9cd814 "CUDA: generalized (mma) FA, add Volta support (#17505)" moved
  Volta to m8n8k4, and fa72aeccb2 deleted the file.

**Per kernel:**

| Kernel | Scale and exp | QK accumulation | P and output accumulator |
|---|---|---|---|
| VEC | Q in FP32, times scale in FP32 (`fattn-vec.cuh` L241-245); `expf` | FP32 SIMT | P FP32; float2 accumulator |
| TILE | q × scale in FP32 → half2 → × 0.25 in half; KQ × 4 after the dot (`fattn-tile.cuh` L925, L932, L936, L631) | each product rounded to fp16 (L546) | P half; half2 accumulator |
| MMA | Q rounded to fp16, then times scale in fp16 (`fattn-mma-f16.cuh` L1278, L1306) | m8n8k4 f32, chained through C (`mma.cuh` L1522-1529) | P half2; fp16-accumulating m8n8k4 for P·V (L1542-1549) |

**Shared:**
- Max starts at -FLT_MAX/2.
- Constants in `fattn-common.cuh`: `SOFTMAX_FTZ_THRESHOLD -20` (L11), `HALF_MAX_HALF` (L10), and
  `FATTN_KQ_MAX_OFFSET 3*ln2` (L19). The offset exists so the fp16 output accumulator does not overflow
  (comment L13-18). Commits: e95d0bc8fd (#17746) and df17a4c94f (#18614). 0cdce38a97 added the × 0.25 / × 4
  trick. 73955f7d2a made VEC pure FP32 on CUDA.
- Combine is FP32 (m,l) with `expf` (L963-977).
- Sinks are in all three kernels and also update the max (VEC L381-396, TILE L1052-1064, MMA L1426-1445).
  Added in fd1234cb46 and 1425f587a8.
- The CUDA FA code no longer reads `GGML_PREC` (75a3a6c2cd).
- The old WMMA kernel (`fattn-wmma-f16.cu` at 73955f7d2a) kept the QK accumulator in half unless
  `GGML_PREC_F32`, and the output fragment always in half.
- No comment anywhere about Volta tensor-core truncation.

### 2.10 1Cat #1277, 899e9ed331. I read all three paths. VERIFIED unless noted.

**Served decode** (`attention_ops.py` `_attend_fused` L925-941 with `_sink_softmax` L898-922;
`VLLM_V41_ATTN_FUSED=1`):
- `torch.bmm(q, keys^T, out_dtype=float32)` on fp16 inputs, i.e. cuBLAS. Tensor-core HGEMM with FP32
  accumulation chained over all 512 dims is INFERRED (cuBLAS default on sm70).
- Then the Triton kernel applies `* scale` in FP32 (L915), takes the max including the sink
  (`m = max(max(s), z)`, L919), computes `e = exp(s - m)` and `p = e / (sum(e) + exp(z - m))` (L920-921), and
  stores P as fp16 (L922).
- Then `torch.bmm(probs, keys)` with fp16 output (L941).
- No split; the whole row is one program.
- KATT1 measured this comparator at 1.2-2.7 max ulps at normal scale and 9.63 at T16 q std 2000. VERIFIED
  (slot output).

**`_sparse_attention_fused`** (L371-414; `VLLM_V41_FUSED_SPARSE`, listed in #1277 HANDOFF as a dead end):
- `tl.dot(fp16 q, fp16 kv)`, then `* scale` (L397).
- Online softmax with -inf and a safe max (L399-403).
- `prob.to(fp16)` before P·V (L404).
- Sink at the end: `top = max(m, z)`, `total = l*exp(m - top) + exp(z - top)` (L406-410).

**`v41k.cu` `attn_split` / `attn_combine`** (`benchmarks/deepseek_v41/v41k/v41k.cu` L2033-2165; compiled but
unused by the served path, per KATT1 report §7):
- **QK.** WMMA m16n16k16 FP32. Each warp takes one key tile and one half of d, so there are two independent
  chains of 16 k16 steps (64 HMMA steps each), summed round-to-nearest in FP32 (`ss[0] + ss[1]`,
  L2066-2081, L2093). This is a partial form of chain splitting.
- **Softmax.** Scale after (L2093), `expf`, P a single fp16 RN (L2102), l summed from unrounded e.
- **O partial.** P·V per 32-key chunk from a zeroed accumulator (fresh), then **stored as fp16**
  (L2129-2134). The unnormalised O can reach up to 32 × |v|. With |v| up to 2688 (E2M1 6 × E4M3 448) that is
  86k > 65504, so overflow is possible. INFERRED.
- **Combine.** The max starts at the sink (L2141), FP32 with `expf` (L2144-2160), and P·V partials are read
  back from fp16.

**#1280** (open, DS-V4 fused sparse-MLA softmax): my earlier sweep (candidates.tsv U04) recorded it as "the same
softmax for DS-V4". Code not read in this pass. UNVERIFIED.

### 2.11 Our engine's V100 attention (1Cat-vLLM port, the V100 server engine 23cec6a). VERIFIED (read-only grep).

- **`csrc/attention/sm70_v37/tail.cu`:** FA2-style. `sm70_scale_apply_exp2` with `softmax_scale_log2` (L329,
  L853), max in raw units, rescale `exp2f((row_max - safe_max)*softmax_scale_log2)` (L284-287). Exported state
  is converted back with `row_max * softmax_scale_log2 * M_LN2` (L761). L861 comment: "the caller supplies
  centered/scaled V and restores its output afterwards" (a V-centering trick for fp16 range).
- **`csrc/attention/sm70_79t/prefill.cu`:** raw QK from cuBLAS stored as **half** scores (L611), then
  `exp2f(score * 1.4426950408889634f)` (L613-615, L759-762); `#error "log2-domain scores require exact-exp
  cuBLAS raw QK"` (L123). fp16 scores would overflow at raw 1.9e5. Prefill only.
- **`csrc/ds41_sparse_prefill_sm70.cu`:** the production prefill WMMA that 0012's per-block code derives from
  (KATT1 report). Not re-read here.

### 2.12 Reference: KATT1 0012 (worktree, uncommitted version 2 over 329e92a). VERIFIED.

- **QK.** Q fragments loaded once (L184-187). `DS41_HMMA_QK_CHAIN` (default 1) WMMA k-steps per FP32 chain,
  with chain results summed round-to-nearest in k order (L254-282). `QK_CHAIN 8` reproduces version 1.
  Cross-warp FP32 sum (L294).
- **Softmax.** Scale after (`sv = a * scale`, L295), `expf(sv - m_new)` (L312), P hi/lo (L314-316), l from
  unrounded p (L313), alpha = `expf(m_old - m_new)` (L322). Running-max floor -1e30 (L77).
- **P·V.** `DS41_HMMA_PV_FRESH` (default 1): each 16-key step starts from zero, and `o = fmaf(o, alpha, pv)`
  (header L32-33; L331-362).
- **Merge.** `expf(ws_m - m)` (L404), with `l += expf(sink - m)` (L412) in the denominator only.
- **Build.** No fast math (L38).
- **Harness hook.** `kScores` writes every FP32 score (L123, L297-302).

---

## 3. Techniques aimed directly at tensor-core accumulator rounding

1. **Ootomo/Yokota TCEC `mma_rn_sync`** (wmma_extension a9e6b34e40). VERIFIED (helper read; I re-read
   L668-695).
   - `mma_sync` defaults to `mma_rn_sync` (`tcec.hpp` L898-917).
   - Each k-block's main term is computed with `zero_op(tmp); mma_op(tmp, a, b, tmp)` and then added with
     ordinary FP32 adds (`frag_d.x[i] += tmp.x[i]`, L672-694).
   - The error-correction terms use operands scaled by 2048 for half (`scale.hpp` L11-20) and are chained
     through C.
   - `mma_rz_sync` (L775) is the variant that chains everything through C.
   - sm_70 is supported. cuMpSGEMM 42dd92ea1c calls it (`device_tcec_wrapper.hpp` L105).
   - The correction terms only matter when inputs are inexact in fp16. 0012's Q and K are exact fp16, so only
     the RN-accumulation half applies. INFERRED.
2. **cutlass `MmaTensorOpFastF32`** (3xTF32 big/small split; `mma_tensor_op_fast_f32.h` L85-139, L364-373).
   VERIFIED. It requires sm80 or newer, so it is not usable on V100.
3. **1Cat `attn_split`:** two d-half chains, RN-summed (§2.10). VERIFIED.
4. **cutlass FMHA large-D path:** per-key-block P·V from a cleared accumulator, merged in FP32 outside the
   tensor core (§2.1e). VERIFIED.
5. **Avoid tensor cores for decode QK:** TurboMind SIMT FP32 (§2.8) and llama.cpp VEC (§2.9). VERIFIED. This is
   the expensive option (§4 item 9).

Background (UNVERIFIED citation details, not read in this pass):
- Fasi, Higham, Mikaitis, Pranesh, "Numerical behavior of NVIDIA tensor cores", PeerJ CS 2021: V100 products
  exact, alignment and normalisation by truncation inside each k=4 step. KATT1's report already cites it.
- Ootomo and Yokota, IJHPCA 2022, "Recovering single precision accuracy from Tensor Cores while surpassing the
  FP32 theoretical peak performance": the source of the RN-outside-the-tensor-core idea.

---

## 4. Ranked techniques for KATT1, with expected cost

Baseline: version 1 MEASURED 28.6 µs/layer at T6, 35.3 at T12 and 53.1 at T18 (ks64, v640). Costs are INFERRED
from instruction counts unless marked; none were measured.

Count used below: QK per warp per 32-key block = 2 n-tiles × 8 k16 steps, so one 8-float fragment add per step
is 128 FADD per thread per block. At T6, ks64 and v640 there are 60 CTAs, each with 2 blocks, so about 256
FADD/thread on the critical path, about 0.2-0.4 µs at 1.4 GHz before overlap with HMMA and loads.

| # | Technique | Prior art (pointer, tag) | What it fixes | Expected cost |
|---|---|---|---|---|
| 1 | **QK short chains (C=0 per k16) plus RN FP32 sum.** Already in KATT1 v2 as `QK_CHAIN=1`. | TCEC `mma_rn_sync` L672-694 (V); 1Cat `attn_split` 2 chains (V) | Removes the 32-step RZ chain bias, the dominant term (§1.1). This should also shrink the emulation mismatch, because the model's sensitivity to HMMA internals scales with chain length (I). | +0.2-1.0 µs at T6, +0.5-2 at T18 (I). If register pressure or SASS shows a regression, `QK_CHAIN=2` (2 × 4-step chains per partial) is the halfway point. |
| 2 | **P·V fresh per 16-key step, `o = fmaf(o, alpha, pv)`.** Already in v2 as `PV_FRESH=1`. | cutlass FMHA large-D epilogue merge (V); TCEC (V) | Matters most at normal scale (long P·V chains over 640 keys) for "accuracy beyond the bar". Little effect at q std 2000, where P is one-hot (I). | +0.2-1.0 µs (64 FFMA per thread per 16-key step) (I) |
| 3 | **exp2 FFMA fold with raw-unit max** (details below). | TurboMind `impl_simt.h` L478/L495, `reduce.cu` L104-106, sink L533 (V); FA-V100 c2eda5e611 `softmax.h` L70-79 (V); engine `sm70_v37/tail.cu` L329 (V) | Removes the per-element rounding of `a*scale` at |s| ≈ 8.4e3 (≤ 4.9e-4 logit, up to 4.9e-4 relative in p). After v2 that is about a quarter to a third of the remaining logit error at q std 2000 (I). | **About 0 µs.** exp2f is no dearer than expf without fast math; one FFMA replaces FMUL + FADD (I). |
| 4 | **Offset-consistent rescale and merge** (details below). No surveyed kernel does this. | none (idea from TurboMind's raw-M (m,l) form) | Removes the per-block ≤ 1 ulp(m·c) offset mismatch that TurboMind and FA2 accept (≤ 6.8e-4 relative in p_old at m·c ≈ 1.2e4) (I). | About 0 µs: one FMUL per row per block (I) |
| 5 | **Compensated (Neumaier) or two-level sum** of the 8 chain partials and the 4 warp partials. | none in surveyed code (I) | Cuts the remaining round-to-nearest summation error from about sqrt(11) roundings to about 1 (I). Worth it only if R6-HMMA2 still shows more than today's ulps at q std 2000. | +0.3-1.0 µs (3 extra FADD per add) (I) |
| 6 | **Keep FP32 scores; do not pre-scale Q in fp16.** | Negative examples: llama.cpp MMA L1278/L1306, xormal L215, ZRayZzz fp16 S (V) | Relative FP accumulation error does not depend on scale, so pre-scaling buys no accuracy. 512^-0.5 = 2^-4.5 is not a power of two, so `fp16(q*scale)` adds a rounding per element. A 2^-4 pre-scale is exact but only helps fp16 range, which is not a problem here (I). | 0 µs |
| 7 | **Keep the build free of fast math.** | Negative examples: TurboMind CMake L15, xformers `setup.py` L456, FA-V100 forks (V) | `__expf` = `ex2(x*log2e)`; its error grows with |x| (I). | 0 µs |
| 8 | **Adaptive FP32 recompute of near-max keys** (details below). | TurboMind and llama.cpp use FP32 SIMT for Volta decode outright (V) | Exact-ish scores where it matters, at no cost in normal rows (I). | About 0.1 µs for the always-on check (a warp-uniform compare on the block max). +2-5 µs only on flagged rows. Kernel complexity: medium (I). |
| 9 | **Full FP32 SIMT QK**, as TurboMind and llama.cpp VEC do. | TurboMind 576-dim MLA decode = MMA_SIMT (V) | Removes tensor-core rounding entirely. | About +15-25 µs at T6. QK is 2·16·640·512 = 10.5 MFLOP per token, 63 MFLOP at T6; 4.5 µs at the 14 TFLOPS peak, 3-5x that in practice (I). This loses most of the 151 µs/call gain. **Not recommended.** |
| 10 | **m8n8k4 PTX with C=0 per k=4 step** (finest TCEC granularity). | TCEC works at the k-block level (V) | Shortens each RZ chain from 4 steps to 1, but grows the round-to-nearest adds per warp from 8 to 32, so it is a net loss unless paired with #5. Needs the quad-pair fragment layout rewritten (I). | +1-3 µs (I). **Not recommended.** |

**Details for row 3.**
- Precompute `c = scale * log2(e)` on the host, in double then rounded to FP32.
- Keep `m` as the raw-unit max of the FP32 S. Ordering is unchanged because scale > 0.
- Compute `mc = m * c` once per row per block, and `p = exp2f(fmaf(a, c, -mc))`.
- The single FFMA rounds only near the small difference. The rounding of `mc` is shared by the whole row and
  cancels in p/l.
- The `kScores` hook and the CPU emulation must switch to raw units, or log both.

**Details for row 4.**
- Store `mc = fl(m*c)` per row.
- Compute `alpha = exp2f(mc_old - mc_new)`. The subtraction is exact (Sterbenz) whenever the two are within a
  factor of 2, which covers every case where alpha is not effectively 0.
- In the merge, use `a_s = exp2f(mc_s - mc_M)`.
- For the sink, use `exp2f(fmaf(sink, log2e, -mc_M))`. Denominator only, as now and as in TurboMind; llama.cpp
  and 1Cat also put the sink in the max, which only matters if the sink exceeds the row max.

**Details for row 8.**
- Make the branch warp-uniform: if the block's max |S| exceeds 2^13, recompute the FP32 dot (SIMT FMA over 512
  dims) for keys within about 16 logits of the running max.
- This is only needed if the real-scale bound (KATT1's job) shows logits anywhere near 1e3.

**Priority.**
- Rows 1 and 2 are already written; they need GPU timing and numerics in R7 row R6-HMMA2.
- Rows 3 and 4 are the best remaining accuracy per µs and should go in the same build, so one slot measures them.
- Row 5 only if needed. Row 8 only if the real-scale evidence demands it. Rows 9 and 10 are not recommended.

### 4.1 Test-side notes (INFERRED, for KATT1's harness; no change made by me)

- **m2 teeth at q std 2000 cannot exist.** With one-hot P the hi/lo split is invisible, and `ks64_nosplit` equals
  ks64 on all five q std 2000 rows. Require m2 on the normal-scale cases, where it is caught 117/135, and record
  the extreme rows as "not applicable" rather than MISSED. The same applies to T2 mixed q std 4, where the effect
  falls under the threshold. This is not a tolerance loosening; the mutation has no observable effect there.
- **Diagnose before tuning.** Run the existing `kScores` hook (L123, L297-302) at T16 q std 2000 and compare
  per-key FP32 scores with an FP64 dot, for v1 and for v2. Errors larger than a few ulp(S) (2^-6 each at 1.9e5)
  mean chain truncation (rows 1 and 5); otherwise it is the `a*scale` rounding (rows 3 and 4).
- **Emulation.** A model of HMMA internals (alignment and truncation inside each k=4 step) will always be
  imperfect. Shorter chains (row 1) and exact-offset softmax (rows 3 and 4) reduce how far that imperfection can
  move the output. This is a better route to `matches_emulation` than loosening its tolerance. main asked
  (17:08Z) that no tolerance be loosened without real-scale evidence and an an internal audit review.

---

## 5. Summary table

| Implementation | Scale / log2e | QK on sm70 | Running max | P before P·V | Split combine | Sink | fast-math |
|---|---|---|---|---|---|---|---|
| cutlass ex41 | ×(scale·log2e) after MMA, then exp2 | m8n8k4 f32, chained through C | -inf | fp16 RN | none (LSE out) | no | n/a |
| xformers (pre-removal) | same as cutlass | m8n8k4 f32, chained | -inf | fp16 RN | backward only | no | yes |
| FA upstream | no sm70 | - | - | - | - | - | - |
| FA-V100 zhinianqin (engine pin) | FFMA exp2, raw-unit max | m8n8k4 f32, chained | 0 init / Check_inf | fp16 | LSE, expf/logf | no | yes |
| FA-V100 ai-bond | ×scale after, `__expf` | wmma m16n16k16 f32, chained | -1e30 | fp16 RN | none | no | yes |
| FA-V100 ZRayZzz | ×scale in exp | m8n8k4 f32 → **fp16 S** | -inf | fp16 | none | no | n/a |
| TurboMind decode | FFMA exp2, raw-unit M | **FP32 SIMT** | -inf | fp16 RN, FP32 FMA | (m,l) exp2 raw M | yes, denominator | yes |
| llama.cpp VEC/TILE/MMA | FP32 Q·scale / fp16 / fp16 | FP32 / fp16 products / m8n8k4 | -FLT_MAX/2 + offset | FP32 / half / half | (m,l) expf | yes, also max | n/a |
| 1Cat #1277 served | ×scale after (Triton) | cuBLAS (I: tensor core, chained) | max incl. sink | fp16 | none | yes, max + denominator | n/a |
| 1Cat `attn_split` | ×scale after | wmma f32, 2 chains, RN-summed | -inf | fp16 RN, **fp16 O** | (m,l) expf | yes, max + denominator | n/a |
| Engine sm70_v37 tail | FFMA exp2 (FA-V100) | m8n8k4 f32 (FA-V100) | -inf, safe | fp16 | per FA-V100 | n/a | n/a |
| **KATT1 0012 v2** | ×scale after, expf | wmma f32, **C=0 per k16, RN sum** | -1e30 floor | **fp16 hi + lo** | (m,l) expf FP32 | yes, denominator | **no** |

All cells VERIFIED except "1Cat #1277 served QK" (INFERRED cuBLAS behaviour) and cells marked n/a (not checked).
