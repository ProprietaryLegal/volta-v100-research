# DeepSeek-V4.1-Flash — Engram conditional memory: exact format, mechanics, placement on the V100 server

> **Origin.** PLI Labs research archive, file `v100-research/deepseek-v4.1-flash/ENGRAM.md`. Published 2026-11-18 in the Volta V100 research
> repository.
> **Edited for publication.** Machine names, private file paths, network addresses, session names and the
> operator's name were replaced with generic terms (`the V100 server`, `[private path]`, `[address]`,
> `the owner`). Links to files that are not published were turned into plain text. No figure and no finding
> was changed. Dates inside the note are its own; later results are in the build pages of this repository.

Lane L-MODEL, 2026-10-06. Sources: official repo @ `2cba9e42aa026125f3ed06c6d98c1db82f7ca027`
(`inference/engram.py`, `inference/model.py`, `inference/convert.py`, tech report §2.4.2 / §3.1.3), header census
(TENSOR_CENSUS.md), measured E8M0 scale exponents (`[...]/scale_sample.json`), GGUF
metadata of `kernelpool/DeepSeek-V4.1-Flash-MXFP4-GGUF` (`l-model/gguf/kernelpool.json`), and a locality
simulation run here with the official tokenizer and a re-implementation of the reference hash
(`l-model/scripts/engram_locality.py` → `l-model/locality/locality_mixed.json`).

## 1. The tables (header-exact)

| Tensor | dtype | shape | bytes |
|-|-|-|-:|
| `layers.1.engram.embed.weight` | F8_E4M3 | [384,006,168, 256] | 98,305,579,008 |
| `layers.1.engram.embed.scale` | F8_E8M0 (UE8M0, 1 per 32 channels) | [384,006,168, 8] | 3,072,049,344 |
| `layers.14.engram.embed.weight` | F8_E4M3 | [384,016,682, 256] | 98,308,270,592 |
| `layers.14.engram.embed.scale` | F8_E8M0 | [384,016,682, 8] | 3,072,133,456 |
| **total tables** | | | **202,758,032,400 B = 188.833 GiB** |
| `layers.{1,14}.engram.wkv.weight` / `.scale` | F8_E4M3 / E8M0 (32×32 blocks) | [25,600, 6,144] / [800, 192] | 157,286,400 / 153,600 each |
| `layers.{1,14}.engram.q_weight`, `k_weight` | BF16 | [4, 5,120] | 40,960 each |

Each table is one shard file of the official checkpoint (`model-00047` = layer 1, `model-00048` = layer 14), with
the weight and its scale stored as **two separate tensors** (a row's 256 B of values and 8 B of scales live ~98 GB
apart in the file).

**Row = 256 FP8 E4M3 values + 8 UE8M0 scales = 264 B**; value = e4m3 × 2^(scale_byte − 127) per 32-channel block
(`model.py:309-320`). Measured row-scale exponents: −10 … −6 (layer 14), −9 … −6 (layer 1) over 8.4 M sampled
scales; the NVFP4-Engram author reports table amax 11.0 (L1) and 24.0 (L14). All row values are exactly
representable in FP16.

### Bucket structure (`engram.py:86-126`)

A table is the concatenation of **24 disjoint sub-tables** = 3 n-gram orders (2, 3, 4) × 8 hash heads. Sub-table
sizes are distinct primes drawn upward from `engram_vocab_size − 1 = 15,999,999`, never reused across layers
(order-major, head-minor; layer 1 takes the first 24 primes, layer 14 the next 24): 16,000,057 … 16,000,889.
Σ(first 24) = 384,006,168 and Σ(next 24) = 384,016,682 — exactly `engram_num_embeddings` (verified against the
GGUF metadata list). So "`engram_vocab_size` 16 M vs `num_embeddings` ≈ 384 M" = 24 heads/orders × ~16 M rows.
Report §2.4.2 (txt 647-651): "N-gram orders {2,3,4}, 8 hash heads, total embedding dimension 2048 per order
[8 × 256], each head indexes a table of approximately 16M entries … distinct primes … FP8 precision".

## 2. Hashing — depends only on token ids (`engram.py:17-61, 64-83, 129-184`)

1. **Compressed vocab** (built once from the tokenizer, 17-61): every token id is decoded alone, normalized
   (NFKC → NFD → strip accents → lowercase → collapse whitespace; a lone space survives via a sentinel), and
   tokens with the same normalized key share one id; partial-UTF-8 byte tokens keep their raw token string.
   Result must be **99,092** ids (asserted, 146). Re-implemented here with `tokenizers` 0.22.2 on the official
   `tokenizer.json`: 129,280 → **99,092** ✓.
2. **Multipliers** (64-83): per Engram layer, `np.random.default_rng(10007 * layer_id).integers(0, bound, 4) * 2 + 1`,
   `bound = (int64_max // 99092) // 2` — one odd multiplier per look-back position (0..3), **shared by all heads and
   orders**. Reproduced exactly (layer 1: 76632096046245, 4839876093313, 35959672319349, 73987337458391; layer 14:
   67716810739261, 51510806800915, 30921347202721, 82619226485591 — identical to the GGUF metadata).
3. **Per position t** (160-184): `tok[s] = compressed[t−s]` for s = 0..3, replaced by the compressed pad id
   (`token_map[2]`) when t−s < 0 or the token is "dead" (inside an image span, or the chain after one).
   `prod[s] = tok[s] * mult[layer][s]` (int64, wraps like torch); `h_n = prod[0] ^ prod[1] ^ … ^ prod[n−1]` for
   n = 2, 3, 4; row id for (order n, head k) = `h_n mod prime[layer][n−2][k] + offset[layer][n−2][k]`. The 1-gram is
   never looked up. Output `[B, L, 2 layers, 24]` (int64).
4. The per-sequence history (`cache`, 155-167) carries the last tokens across prefill/decode.

**Cost:** ≈ 4 mul + 3 xor + 24 mod per token per layer (~60 integer ops/token for both layers) — negligible on CPU
(< 0.1 s for a 1M-token prompt single-threaded) and also cheap on GPU (the reference computes it on GPU).
**The hash for position t is known the moment token t exists** — for decode, right after sampling; for prefill,
for the whole prompt before the first layer. Nothing depends on hidden states. DeepSeek: "deterministic addressing
enables embeddings to be prefetched from host memory via background RDMA transfers, with prefetching for the first
module overlapping computation in the first Transformer block" (txt 652-654); training prefetches "for the entire
local batch before each pipeline stage begins" (txt 913-915).

## 3. Lookup and combination (`model.py:296-365`, `1262-1263`)

Per position and Engram layer: **24 rows** (one per order × head) → dequantize (fp32 value × scale, cast bf16)
→ concatenate to a 24 × 256 = **6,144-vector** (order-major) → `wkv` (FP8 linear 6,144 → 25,600 = 5 × 5,120) →
split into 4 keys (one per hc residual copy) + 1 value. For each hc copy c:
`dot_c = Σ_d h_c[d] · (q_weight[c,d]·k_weight[c,d]) · key_c[d] · rsqrt(mean(h_c²)+ε) · rsqrt(mean(key_c²)+ε) / √5120`,
`gate_c = sigmoid(sign(dot_c)·sqrt(max(|dot_c|, 1e-6)))`, `h_c ← h_c + gate_c · value` (all fp32, cast back).
The module is applied to the residual stream **before** block 1 and before block 14 (`Transformer.forward`
1262-1263); masked (image) positions pass through with gate 0. Reference TP: tables row-sharded across ranks, each
rank zero-fills foreign rows and the 24×256 bf16 result is all-reduced (`ParallelEngramEmbedding.forward` 312-325);
`wkv` is replicated.

Note on numerics: in the reference the dequantized rows go through `linear()`, i.e. are re-quantized to FP8 per
32-block before the FP8 `wkv` GEMM; since the rows are already E4M3 × pow2 per 32 block the re-quantization is
(near-)identity (vLLM PR #58766 "feed Engram's FP8 rows to wkv without dequant + requant"). `engram.wkv` itself has
scale exponents down to 2^-18 (22.9 % of blocks below 2^-14) — **dequantizing `wkv` to FP16 needs a scale bias**
(e.g. ×2^10 on the weights, ÷2^10 on the fp32 output) or fp32 dequant.

## 4. Bytes per token

| Table format | B per row | rows / token / layer | **B / token / layer** | **B / token (2 layers)** | GPU-side after dequant (FP16) |
|-|-:|-:|-:|-:|-:|
| official FP8 E4M3 + E8M0/32 | 264 | 24 | 6,336 | **12,672** | 24,576 |
| MXFP4 E2M1 + E8M0/32 | 136 | 24 | 3,264 | 6,528 | 24,576 |
| NVFP4 E2M1 + E4M3/16 (+fp32 global/table) | 144 | 24 | 3,456 | 6,912 | 24,576 |
| INT4 g32 + fp16 scale (INC W4A16) | 144 | 24 | 3,456 | 6,912 | 24,576 |

Decode: 12.7 KB/token of packed rows is ~2.5 µs over a 5 GB/s PCIe link — bandwidth is irrelevant; **latency and
RAM capacity are the constraints**. Prefill: 12.7 KB × prompt tokens (64K → 0.83 GB of row bytes). The `wkv` GEMM
(157 M FP8 weights per Engram layer) is ordinary dense compute and belongs on the GPU (TP-shardable by output).

If rows are read through the OS page cache, every row costs a 4 KiB page (two pages with the official split
value/scale tensors). Interleaving value + scale into one 264-B row (as the kernelpool GGUF does: tensor
`blk.N.engram_embd.weight`, I8 `[264, rows]`, metadata `engram.encoding = "e4m3_e8m0_32_row264"`) makes it one page
per row (+6.4 % straddling a page boundary).

## 5. Locality (measured here, inference beyond the corpus)

Corpus: 1,125,756 tokens (133 v100-research Markdown files, 100 CPython stdlib modules, the V4.1 tech report text),
each file hashed as its own sequence, official tokenizer + reference hash (validated by the 99,092 compressed vocab).

| Per Engram layer (both layers ≈ identical) | Value |
|-|-|
| lookups | 27,018,144 (24/token) |
| distinct rows touched | 11,745,459 (**3.06 %** of the 384 M rows) = 3.10 GB of FP8 row bytes |
| distinct 4 KiB pages touched | 9,198,614 = 37.7 GB of page cache (12× the row bytes) |
| hit rate with an infinite row cache | **56.5 %** (43.5 % are first-touch, compulsory misses) |
| LRU row cache 100 K rows (26 MB FP8) | 41.4 % |
| LRU row cache 1 M rows (264 MB) | 50.3 % |
| LRU row cache 4 M rows (1.06 GB) | 55.2 % |
| reuse by order: 2-gram / 3-gram / 4-gram | 78.6 % / 53.6 % / 37.4 % of lookups hit a row seen before |
| share of lookups served by the top 1 % of distinct rows | 37.5 % (2-gram), 23.4 % (3-gram), 15.3 % (4-gram) |

Interpretation (inference): the hot set is small and saturates quickly — a ~1 M-row-per-layer cache (≈ 0.5 GB for
both layers in FP8) captures ~90 % of the achievable hits — but roughly **40–45 % of lookups on new text are
compulsory misses** that must be served from wherever the full table lives. Rows have no spatial locality (adjacent
rows are unrelated n-grams), so a row-granular user-space cache is ~12–15× more memory-efficient than relying on the
page cache. The tech report states no hot-row fraction. Re-sent agentic prefixes do not re-read rows when prefix KV
caching hits (no recomputation), so the steady state is dominated by new tokens.

**the V100 server's disks (measured by L-INV, HARDWARE.md §A.5):** `[private path]` (DRAM-less Realtek RTS5772DL, 126 media
errors) under the concurrent download: 4 KiB random reads QD1 p50 0.75 ms / mean 2.1 ms / p99 14 ms, **QD32 only
7.3 k IOPS**; root NVMe (Phison E18 with DRAM, 254 GB free): **QD1 p50 74 µs, QD32 219 k IOPS**. Page cache: 196.4 GiB
RAM, of which qdrant's mmap holds ~64 GiB.

At decode that is ~48 × 0.45 ≈ 22 row misses per token. JigSawPT (RTX 5090 + PCIe-5 NVMe, Windows, mmap'd tables)
measured 16.5 µs/row at queue depth 8 after making the 48 reads asynchronous (10.4 ms → 0.92 ms per token, "11.4×,
exact"); one batch of ~22 parallel 4 KiB reads at QD ≥ 16 is ≈ 0.1–0.2 ms on a modern NVMe (estimate; the V100 server's nvme2
random-read IOPS/latency to be measured by L-INV). Layer 1's rows have embed + block 0 to hide behind; layer 14's
have 13 blocks. Prefill 64K tokens ≈ 1.4 M misses ≈ 5.5 GB of 4 KiB random reads ≈ 2–4 s at 1.5–3 GB/s random —
small against V100 prefill compute at that length (estimate) **on a fast drive**. Applied to the V100 server (arithmetic,
not measured end-to-end): on the root E18 (219 k IOPS) 22 misses/token ≈ 0.1 ms per decode step and a 64K prefill
≈ 1.4 M misses ≈ 6–7 s; on nvme2 as measured (7.3 k IOPS) the same costs ≈ 3 ms/token and **≈ 190 s per 64K
prefill** — nvme2 is not a viable Engram miss path unless an idle re-measurement is far better.

## 6. Placements in the wild

| Who | Where the tables live | Mechanism |
|-|-|-|
| Reference `inference/` | GPU HBM, row-sharded over TP ranks (`convert.py:135-141`, `model.py:296-325`) | `F.embedding` + all-reduce; sized for 8 × 141 GB-class GPUs |
| DeepSeek production (report) | host memory | background RDMA prefetch overlapped with block 0 (txt 652-654); "sharded Engram embedding tables" (txt 202); RL rollouts keep them in GPU memory (txt 924-926) |
| vLLM ≥ v0.30.0 | **pinned host memory by default** (`EngramConfig.cpu_offload=True`, `vllm/config/engram.py:38`) | Triton UVA gather + side-stream prefetch (PR #56512); options `embedding_across_dp`, `dp_shared_memory` (/dev/shm), `use_thp`; disk-backed mmap variant PR #56757 (unmerged) measured **parity with pinned UVA on B200 TP4** (14,168 vs 13,632 tok/s, 8k/1k, conc 16) at 95 GB vs 352 GB host RAM, with host-side gather + dedup + on-device dequant |
| SGLang ≥ v0.5.21 | GPU by default; `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE` → host (memfd/anonymous mmap, MADV_HUGEPAGE/COLLAPSE, `cudaHostRegister`) | `srt/layers/engram.py` |
| llama.cpp forks / DwarfStar | mmap'd from the GGUF on SSD, rows read by the host | JigSawPT, smalinin ("mmap/lazy host-backed"), antirez ("reads the needed rows directly from disk") |
| aidendle94 (4 × DGX Spark) | rows read straight from checkpoint shards on NVMe (`DSV41_ENGRAM_DISK=1`), CPU decode of NVFP4 rows | |

## 7. FP4 storage — feasibility and evidence

| Variant (who) | Format | Scale granularity | Size (2 tables) | Measured error | End-to-end evidence |
|-|-|-|-:|-|-|
| official | FP8 E4M3 | UE8M0 / 32 | 202,758,032,400 B (188.83 GiB) | — | reference |
| lvkaokao / INCModel3 "MXFP4-Engram" (auto-round RTN) | E2M1 packed, I8 [R,128] | E8M0 / 32 | 104,451,107,600 B (97.28 GiB) | rel err ≈ 0.12 vs FP8 | gsm8k 92.19 / 92.27 vs 92.87 baseline (5-shot, n = 1319) — only test |
| LibertAIDAI (pow2 scale ≥ amax/6, no clipping) | E2M1 packed, U8 [R,128] | E8M0 / 32 | 104,451,107,600 B (97.28 GiB) | mean cosine 0.99342 / 0.99345 (1.5 M rows) | none ("not evaluated end-to-end") |
| aidendle94 NVFP4-Engram (GPTQ feedback through `wkv` + MSE scale search) | E2M1 packed, U8 [R,128] | E4M3 / 16 + fp32 global per table (0.004092 L1, 0.008929 L14) | 110,595,290,400 B (103.00 GiB) | rel RMS 8.99 % / 9.06 %; error through `wkv` 8.23 % | teacher-forced scoring of 40 FP8-model completions: mean logprob −0.111 vs −0.112, argmax agreement 97.5 % vs 97.5 %; needle 32K/128K found; tool call pass |
| INCModel3 / lvkaokao W4A16 | INT4 sym, U8 [R,128] | fp16 / 32 | 110,595,290,400 B (103.00 GiB) | — | gsm8k 93.93 / 94.01 (with INT4 experts) |
| smalinin GGUF | Q5_K or Q8_0 | — | 135.2 GB (Q5_K) / 208.9 GB (Q8_0) | — | — |

Decode cost of FP4 rows is a 16-entry LUT × scale(s) — trivial on CPU or GPU, no tensor cores involved. Both MXFP4
and NVFP4 decode to exact FP16. The wkv input is the dequantized row, so the format only changes row bytes and
the row values. Evidence that 4-bit Engram is harmless is thin (one teacher-forced test and gsm8k); it must be
re-checked here (teacher-forced logprob/argmax agreement of FP4 vs FP8 Engram on our own prompts, plus a long-
context needle) before it becomes a default.

## 8. RAM arithmetic on the V100 server (196 GiB visible, ONE NUMA node per L-INV)

| Engram storage | Table bytes | Fits pinned in RAM? | Comment |
|-|-:|-|-|
| FP8 (official) | 188.83 GiB | **No** (leaves ~7 GiB for OS + 8–12 worker processes + staging) | mmap from nvme2 + page cache + hot-row cache |
| MXFP4 | 97.28 GiB | Yes in the 12-GPU layout (no expert spill): leaves ~99 GiB | 8-GPU layout also needs ~42–60 GiB of expert spill + per-process overhead → tight |
| NVFP4 / INT4-g32-fp16 | 103.00 GiB | Yes (12-GPU); tight (8-GPU) | |
| hypothetical E2M1 + E8M0/64 (132 B/row) | 101,379,016,200 B = 94.42 GiB | Yes | not an existing artifact |
| floor: E2M1 values only (128 B/row) | 98,306,924,800 B = 91.56 GiB | — | no scales; not usable as is |

NUMA: the V100 server is a single NUMA node (TR PRO 3945WX), so there is no remote-NUMA penalty to manage; hugepages matter
only for a pinned/anonymous table (TLB reach on random 136–264 B reads: THP/hugetlbfs recommended; vLLM `use_thp`,
SGLang `MADV_HUGEPAGE`).

## 9. Placement verdict for the V100 port

1. **Never HBM-resident in full** (189 GiB ≈ 15.7 GiB per GPU across 12 GPUs — the whole KV/workspace budget and
   more). An HBM **hot-row cache** is optional: 1 M rows per layer = 264 MB FP8 per Engram layer per stage captures
   ~50 % of lookups; a host-RAM row cache captures the same with no HBM cost — measure before spending HBM.
2. **Exact default (Phase A/B bring-up):** official FP8 tables, repacked once into interleaved 264-B rows (one file per
   layer, ~101 GB each, 202.8 GB total, sequential write) on the **root E18 NVMe** (254 GB free — tight; nvme2's
   measured 7.3 k IOPS makes it unusable as the miss path, §5), opened read-only and mmap'd/pread with deep queue depth
   (io_uring or a thread pool) by a per-stage host Engram service. Hash on CPU from token ids as soon as they exist
   (decode: right after sampling; prefill: whole chunk up front; DSpark drafts: per draft token); gather 48 packed rows
   per token into a pinned staging buffer (dedup within the batch, user-space LRU row cache in RAM), async H2D
   (12.7 KB/token), dequant + concat on GPU, `wkv` + gate on GPU. The page cache uses whatever RAM is free as a warm
   tier automatically.
3. **Fast option (12-GPU layout only, quality-gated; becomes the practical default if 202.8 GB cannot be given to
   the root NVMe):** a 4-bit Engram (MXFP4 or NVFP4 rows, ~97–103 GiB) fully
   RAM-resident (pinned → V100 UVA zero-copy gather, or CPU gather) — removes NVMe from the decode path. Gate on a
   measured teacher-forced comparison against the FP8 path. The artifacts exist as two drop-in shard files
   (lvkaokao MXFP4-Engram shards 47/48, aidendle94 NVFP4-Engram shards 47/48 — all their other 46 shards are
   byte-identical to the official ones), or can be produced locally from the FP8 tables (RTN in one pass).
4. **Stage ownership:** Engram layer 1 always sits on PP stage 0; layer 14 sits on stage 0 (PP2 20/20) or stage 1
   (PP3 13/14/13). Each owning stage runs its own gather; both stages can start gathering at the same time (hashes
   are known before layer 0).
5. Keep the reference's hash semantics bit-exact (int64 wrap-around multiply, Python/torch floor-mod with positive
   modulus, pad/dead handling, per-sequence history) and the tokenizer revision fixed — any mismatch silently
   rehashes the whole table (engram.py:143-146).

## 10. Measured on the V100 server (lane L-ENGRAM, 2026-10-07 02:50–03:40Z)

Implementation measured: branch `ds41/engram` (1Cat worktree `[private path]`):
`common/engram.py` (hash, module), `common/engram_host.py` (EngramHostService), `csrc/engram_rows.cpp` (row reader),
`sm70/engram_kernels.py`. Rows are read **in place** from the official, sha256-verified shards 47/48 on the root
Phison E18 (`[private path]`, DECISIONS D4/D5 / PORT_DESIGN AM-4); each TP rank keeps only its own 6 of
24 sub-tables' UE8M0 scales page-locked in RAM. Harness: `[...]/engram_bench.py`, raw
results `results.jsonl` (same dir). "Cold" = `POSIX_FADV_DONTNEED` on the shard (no sudo; drops its page-cache pages)
plus never-seen text; "warm" = the same tokens again. "4 ranks" = four processes (one per TP4 rank, its own
sub-tables) gathering concurrently, as a TP4 stage does; all processes shared one V100 (GPU-<masked>), so end-to-end
step latencies come from 1-rank runs and multi-rank runs report I/O.

**Layout facts.** `embed.weight` starts at absolute byte 664 (shard 47) / 672 (shard 48), so rows are not 256-aligned:
1 row in 16 straddles a 4 KiB page. Measured scale exponents over the full tables: layer 1 −10…−5, layer 14 −11…−4
(wider than the 8.4 M-scale sample in §1) — every row value is still exact in FP16 with no bias (window −15…+7).

**Correctness on the real shards** (tests `test_engram_rows.py`, `test_engram_module.py`): rows gathered in place ==
safetensors reads for 4 ranks × 2 layers × {buffered 200K, O_DIRECT 20K} random ids incl. every bucket's ends; the
hash is bit-exact vs the official `engram.py` over 123K real tokens; module output (layers 1/14, real `wkv`/q/k, real
token ids) vs an FP32 transcription of `model.py:296-365`: stream rel-RMS 3.7e-5 / 2.8e-5 (gate 1e-3). For scale: the
reference's own BF16 rounding of `kv` moves its output by 3.5e-3 / 2.8e-3 rel-RMS.

| Decode, T = 1 (12 rows per rank: 2 layers × 6 sub-tables) | cold | warm |
|-|-|-|
| gather latency, 1 rank, back-to-back steps (p50 / p99) | 0.22 / 1.0 ms | 0.04 / 0.07 ms |
| gather latency, 1 rank, 30 ms idle between steps (≈ decode cadence) | 0.30 / 3.2 ms | 0.06 / 1.9 ms |
| gather latency, 4 ranks concurrent (p50 / p99) | 0.21 / 0.4–0.8 ms | 0.04 / 0.08–0.3 ms |
| misses per step per rank (fully cold text) | 7.5–9 of 12 | 0 |
| host CPU per step: begin_step + bind_batch + wait_rows (main thread) | ≈ 0.4 ms | 0.40 ms |
| reader-thread CPU per step | 0.12 ms | 0.04–0.06 ms |
| GPU per Engram layer per rank (row decode 26 µs + wkv GEMV 125 µs + gate 21 µs) | 0.17 ms | 0.17 ms |
| H2D per step (3.2 KB rows + index) | ≈ 10–20 µs at 6.5 GB/s | |

Against the ~30–50 ms per-token budget of the > 20 tok/s milestone, Engram costs ≈ 0.35 ms GPU (two layers) +
2 FP32 all-reduces of 100 KiB + ≈ 0.4 ms CPU on the step path, ≈ 1–2 %; the gather itself is started before the PP
receive / before layer 0 and overlaps compute. Its p99 (≈ 1–3 ms cold, mostly wake-up latency at idle) can stall
layer 1 on ≲ 1 % of tokens by ≈ 1–2 ms.

| Prefill, 4 ranks concurrent, cold, admission prefetch on, NO compute between chunks (8K-token chunks) | 1 thread/rank | 2 threads/rank | 4 threads/rank |
|-|-:|-:|-:|
| 8K tokens: gather (1 chunk) | — | 0.96 s | — |
| 32K tokens: total gather (4 chunks) | — | 2.5 s | — |
| 64K tokens: total gather (8 chunks; 786K lookups, 442K unique rows, ~350K misses per rank) | 6.2 s | 4.3 s | 3.5 s |
| root NVMe random 4 KiB reads achieved (device counter) | 237k IOPS | 342k IOPS | 423k IOPS |
| reader-thread CPU per rank during the burst | 0.53 cores | 0.79 cores | 1.13 cores |
| warm 8K chunk gather (81K unique rows) | 76 ms | 52 ms | 35 ms |

The deep queue comes from `POSIX_FADV_WILLNEED` issued for every miss before the blocking reads (L-INV measured
219k IOPS at QD32 with plain reads). **With compute between chunks** (2 s simulated per 8K chunk, 2 threads/rank,
64K tokens): admission prefetch off → every chunk waits ≈ 576 ms (4.7 s total); on → only the first chunk waits
(573 ms), chunks 2–8 gather in ≈ 48 ms (1.15 s total): the prefetch hides ≈ 85 % of a cold 64K prompt's Engram I/O.
1-rank 8K chunk: cold 0.54 s, warm 67 ms gather; its 21 MB H2D is ≈ 3.3 ms at 6.5 GB/s.

**In-place layout cost (D5's 5 % rule).** Straddling rows: O_DIRECT QD1 79 → 89 µs (+13 %), QD32 5.03 → 5.39 µs/row
(+7 %); buffered deep-queue throughput 444k → 263k rows/s for straddlers. Weighted by 1/16 of rows: ≈ +4 % on
cold-prefill gather throughput and ≈ 0 % on decode (gather overlapped, ≪ budget). The repack is **not** justified.

**Memory.** Page-locked scales 0.715 GiB per layer per rank (1.43 GiB for a rank owning layers 1 + 14; 5.7 GiB per PP2
stage, 2.9 GiB per PP3 Engram stage), exact-size via `cudaHostRegister` (RLIMIT_MEMLOCK 8 MiB does not apply).
Pinned staging 50 MiB per rank at `max_num_batched_tokens` 8192 with 2 layers (2 slots); device: staging 52 MB + dense
26 MB + `wkv_r` 78.6 MB per layer. Page cache: ≈ 1.8 GB per rank per 64K prompt (442K rows × 4 KiB). Service init
7–9 s per rank cold (1.43 GiB scale read + token map 1.0 s + extension load); first build of the extension ≈ 20 s.

**CPU-thread budget (≤ 6 for Engram on 12c/24t).** Decode needs ≈ 0.1 core per stage. Prefill bursts: 4 ranks ×
0.53 / 0.79 / 1.13 cores at 1 / 2 / 4 threads per rank. Recommended `VLLM_DS41_ENGRAM_IO_THREADS`: **2 for PP2**
(one Engram stage: ≈ 3.2 cores) and **1 for PP3** (two Engram stages: ≈ 4.3 cores); the ctor default 4 (§3.6) would
use ≈ 4.5 cores per stage.
