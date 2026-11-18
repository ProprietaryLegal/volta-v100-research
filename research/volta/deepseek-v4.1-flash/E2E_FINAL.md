# DeepSeek-V4.1-Flash V100 final end-to-end campaign

> **Origin.** PLI Labs research archive, file `v100-research/deepseek-v4.1-flash/E2E_FINAL.md`. Published 2026-11-18 in the Volta V100 research
> repository.
> **Edited for publication.** Machine names, private file paths, network addresses, session names and the
> operator's name were replaced with generic terms (`the V100 server`, `[private path]`, `[address]`,
> `the owner`). Links to files that are not published were turned into plain text. No figure and no finding
> was changed. Dates inside the note are its own; later results are in the build pages of this repository.

Updated 2026-10-07T17:23:28Z. Build **d73d1616b174aec48787f9cdafc5d349b294539a** on `ds41/sm70-v41`, host the V100 server. No model code edits or commits.

**Campaign complete; overall FAIL.** Teacher-forced gates PASS; exact greedy goldens FAIL; DSpark caps1–5 startup FAIL. Context, concurrency, async equality, PP2 inference and semantic API checks completed. Scope: attention decode OFF only.

Artifacts, full commands, raw SSE/token IDs, request bodies, metrics, full FP32 teacher-forced logits and memory traces: `[private path]`. Checkpoints: a progress log (not published).

## Method

Production PP3×TP4 maps boards A→C→B, partition 14/14/12, no spill. A: eager/synchronous, mirror CRC on, util .90, chunk 1024, dump layers none/steps all. B/C: launcher default decode graphs, synchronous, mirror off, util .92, chunk 4096. D: same speed settings with async scheduling and VLLM_DS41_CORE_ASYNC_PP=1. E: eager/synchronous, block64, one sequence, prefix caching disabled for both target and draft runs; fixed draft block5, verification cap1..5. F: PP2×TP4 boards A+C, spill120, eager. G: fresh PP3 server after F; default graphs/synchronous. Every server exports `VLLM_DS41_ATTN_DECODE_PATH=0`, `VLLM_USE_V2_MODEL_RUNNER=0`, `VLLM_DS41_HC_FUSED=1`. g32 dense and fused MoE retain merged defaults. Every server runs in its own terminal session through the provided launcher.

Reused PP3 tf_logits.py, pp3_gap.py (only output paths redirected), client.py, api_check.sh, memmon.sh, existing generated prompts. Stream benchmark extends bench.py methodology with return_token_ids, full raw events and before/after Prometheus metrics. TTFT ends at first returned token; prefill rate=prompt tokens/TTFT; decode rate excludes tokens in the first SSE chunk. Needle=answer in first80 characters. Context is actual prompt+completion tokens. C aggregate throughput is total output/wall across the four concurrent requests; per-stream decode also reported. Latencies include HTTP/client overhead. No target-only/speculative throughput mixing.

Quality gate AM-13′: pooled token-weighted |ΔNLL|≤.0435, top1≥.9768 and≥.97, top20 KL lower bound≤.0090. AM-14′: |ΔNLL|≤.0549, top1≥.9673. Own-k goldens, per D17/NLL_GAP. Greedy golden exactness uses own-k continuations, ignore_eos=true to enforce32 returned tokens (older PP3 client stopped p2 at12); legacy IDs retained. Greedy golden exactness reported separately; prior p2/p3 mismatches do not alter those pooled thresholds. Dump mode materializes fused HC posts per layer; its full-logit quality evidence applies to that eager dump configuration.
Live preflight power limits: board A265W, B180W, C238W. A later `clocks.csv` snapshot reports application graphics clock1290MHz on all cards; this is not a measurement of locked SM clock or per-run operating clocks. Host settings were not changed. `_C.abi3.so` SHA256 `9f8ca6ba54ef53cde8e5c3a535e3a4284dab9bd318f09dd34518237f68d4c7c7` verified before every launch. All48 checkpoint shards had verifier OK markers. UUIDs/topology are in preflight.log and resolved commands.

## A — quality

- v100-semantic+ownk: PASS, positions 2419; |ΔNLL| 0.030941, top1 0.984291, KLlb 0.003790; gates {'NLL': True, 'top1': True, 'KLlb': True}.
- ref-exact+ownk: PASS, positions 2419; |ΔNLL| 0.047175, top1 0.978090, KLlb 0.010226; gates {'NLL': True, 'top1': True}.
- p0_smoke: greedy 24/32 exact prefix; FAIL, first mismatch 24.
- p1_text: greedy 26/32 exact prefix; FAIL, first mismatch 26.
- p2_code: greedy 24/32 exact prefix; FAIL, first mismatch 24.
- p3_doc: greedy 18/32 exact prefix; FAIL, first mismatch 18.
- F p0: 24/32 golden exact prefix; PASS=False.

## B/D/F — performance and context

|Run|Prompt tokens|Achieved total|TTFT s|Prefill tok/s|Decode tok/s|Needle|Async token equality|
|---|---:|---:|---:|---:|---:|---|---|
|B 1024|1018|1274|1.618|629.2|10.104|True|—|
|B 8192|8185|8441|5.167|1584.2|10.694|True|—|
|B 32768|32745|33001|14.777|2216.0|10.228|True|—|
|B 65536|65499|65755|30.421|2153.1|10.003|True|—|
|B 131072|131065|131321|77.588|1689.2|9.109|True|—|
|B 262144|261465|261721|291.191|897.9|7.599|True|—|
|D 1024|1018|1274|1.974|515.6|11.989|True|True|
|D 32768|32745|33001|14.792|2213.6|11.957|True|True|
|F 1024|1018|1274|11.436|89.0|5.506|True|—|

## C — concurrency

- Stream 0: 7.961 tok/s, TTFT 0.948s, 256 tokens; needle True.
- Stream 1: 8.363 tok/s, TTFT 3.651s, 256 tokens; needle True.
- Stream 2: 8.362 tok/s, TTFT 3.640s, 256 tokens; needle True.
- Stream 3: 8.363 tok/s, TTFT 3.644s, 256 tokens; needle True.
- Aggregate: 29.106 tok/s over 35.181s wall; sum of per-stream decode rates 33.049 tok/s.

## E — DSpark (separate eager target-only baseline)

|Run|Prompt|Decode tok/s|Exact token equality|First mismatch|Acceptance|Accepted tokens/step|
|---|---|---:|---|---|---:|---:|
|E_target|p1_text 32 r0|6.252|baseline|—|None|None|
|E_target|p1_text 256 r0|7.767|baseline|—|None|None|
|E_target|p1_text 256 r1|7.853|baseline|—|None|None|
|E_target|p1_text 256 r2|7.737|baseline|—|None|None|
|E_target|p2_code 32 r0|7.500|baseline|—|None|None|
|E_target|p2_code 256 r0|7.807|baseline|—|None|None|
|E_target|p2_code 256 r1|7.349|baseline|—|None|None|
|E_target|p2_code 256 r2|7.574|baseline|—|None|None|
|E_target|p3_doc 32 r0|6.656|baseline|—|None|None|
|E_target|p3_doc 256 r0|7.144|baseline|—|None|None|
|E_target|p3_doc 256 r1|7.058|baseline|—|None|None|
|E_target|p3_doc 256 r2|7.288|baseline|—|None|None|
|E1|startup FAIL — see E1.serve.log||||||
|E2|startup FAIL — see E2.serve.log||||||
|E3|startup FAIL — see E3.serve.log||||||
|E4|startup FAIL — see E4.serve.log||||||
|E5|startup FAIL — see E5.serve.log||||||
- E_target_p1_text: 3 completed 256-token repeats, median 7.767 tok/s; repeat equality [{'equal': True, 'first_mismatch': None}, {'equal': True, 'first_mismatch': None}].
- E_target_p2_code: 3 completed 256-token repeats, median 7.574 tok/s; repeat equality [{'equal': True, 'first_mismatch': None}, {'equal': True, 'first_mismatch': None}].
- E_target_p3_doc: 3 completed 256-token repeats, median 7.144 tok/s; repeat equality [{'equal': True, 'first_mismatch': None}, {'equal': True, 'first_mismatch': None}].

## Memory (sampled every 5s; MiB by live index)

Host values are whole-host observations, including other CPU jobs and file cache; vLLM RSS is sampled process RSS and may count shared mappings repeatedly. Peaks are per-column maxima across the run, not one simultaneous snapshot.

|Run|GPU0|1|2|3|4|5|6|7|8|9|10|11|min MemAvailable GiB|max anon GiB|max shmem GiB|max vLLM RSS GiB|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|A_mem|29888|29888|29888|29888|28116|28116|28116|28116|30798|30798|30798|30798|90.4|50.3|49.0|38.3|
|B_mem|31454|31454|31454|31454|29082|29082|29082|29082|32168|32168|32168|32168|99.5|38.7|50.9|39.1|
|D_mem|31434|31434|31434|31434|29060|29060|29060|29060|32184|32184|32184|32184|105.0|35.5|48.4|37.9|
|E1_mem|28090|28090|28090|28090|27902|27902|27902|27902|26600|26600|26600|27982|106.8|34.3|48.5|35.4|
|E2_mem|27072|27072|27072|27072|27902|27902|27902|27902|26578|26578|26578|26578|105.8|35.5|48.5|35.6|
|E3_mem|27240|27208|27240|27640|27778|27778|27778|27778|28186|28186|28186|28186|105.8|35.8|48.5|35.3|
|E4_mem|28090|28090|28090|28090|27772|27770|27772|27450|27982|27982|27982|27982|105.8|35.7|48.2|35.1|
|E5_mem|27072|27072|27072|27072|25934|25934|25934|25934|26644|26600|26552|26600|108.0|33.7|48.2|34.5|
|E_target_mem|30170|30170|30170|30170|27434|27434|27434|27434|31208|31208|31208|31208|102.8|38.9|48.5|37.5|
|F_mem|30282|30282|30282|30282|0|0|0|0|30632|30632|30632|30632|27.4|31.3|131.6|111.8|
|G_mem|29260|29260|29260|29260|26518|26518|26518|26518|30230|30230|30230|30230|97.4|44.5|46.3|38.1|

## G — API

Health/models/chat/completions stream and non-stream: raw evidence `[...]/G_api.log`. HTTP200 health=True; choices replies=27. Full semantic inspection required; shell exit alone is not a PASS.
- (Three raw API responses that stood here were removed for publication: this repository does not publish generated model text.)

Full SSE reads to `[DONE]`, usage/max_tokens validation, model identity and semantic tool/reasoning checks:
- G_verify_health: PASS; request and full raw response in `G_verify_health.request.json` / `G_verify_health.raw`.
- G_verify_models: PASS; request and full raw response in `G_verify_models.request.json` / `G_verify_models.raw`.
- G_verify_chat_False: PASS; request and full raw response in `G_verify_chat_False.request.json` / `G_verify_chat_False.raw`.
- G_verify_completion_False: PASS; request and full raw response in `G_verify_completion_False.request.json` / `G_verify_completion_False.raw`.
- G_verify_chat_True: PASS; request and full raw response in `G_verify_chat_True.request.json` / `G_verify_chat_True.raw`.
- G_verify_completion_True: PASS; request and full raw response in `G_verify_completion_True.request.json` / `G_verify_completion_True.raw`.
- G_verify_reasoning_False: PASS; request and full raw response in `G_verify_reasoning_False.request.json` / `G_verify_reasoning_False.raw`.
- G_verify_reasoning_True: PASS; request and full raw response in `G_verify_reasoning_True.request.json` / `G_verify_reasoning_True.raw`.
- G_verify_tool: PASS; request and full raw response in `G_verify_tool.request.json` / `G_verify_tool.raw`.

## Failures and final state

Predecessor E2E-FINAL had attention-ON hang/startup failures. This report tests attention OFF only, as binding D28 requires. Prior evidence remains in e2e-final/.
- E1: server startup; see E1.serve.log; exact command/error in campaign.log and run serve log; no code fix attempted.
- E2: server startup; see E2.serve.log; exact command/error in campaign.log and run serve log; no code fix attempted.
- E3: server startup; see E3.serve.log; exact command/error in campaign.log and run serve log; no code fix attempted.
- E4: server startup; see E4.serve.log; exact command/error in campaign.log and run serve log; no code fix attempted.
- E5: server startup; see E5.serve.log; exact command/error in campaign.log and run serve log; no code fix attempted.
# Exact greedy failures, attention OFF
A eager/full-dump/mirror1/util.90/chunk1024. Pooled AM-13-prime/14-prime PASS. All32 tokens received; no hang.
p0_smoke: FAIL first mismatch zero-based24, target token5512, own-k golden token15225. Exact request [...]/A_p0_smoke.json.request.json, command campaign.log.
p1_text: FAIL first mismatch zero-based26, target token418, own-k golden token305. Exact request [...]/A_p1_text.json.request.json, command campaign.log.
p2_code: FAIL first mismatch zero-based24, target token0, own-k golden token38. Exact request [...]/A_p2_code.json.request.json, command campaign.log.
p3_doc: FAIL first mismatch zero-based18, target token1522, own-k golden token58782. Exact request [...]/A_p3_doc.json.request.json, command campaign.log.
Component: end-to-end greedy numeric path, before divergence sequences agree; no API/token extraction error (full returned IDs and raw dump exist). Changes since previous validated PP3 include g32 dense, fused HC/MoE, COMM. The pooled gate tolerates close-distribution token flips; no single kernel attribution is established without controlled ablations. Attention decode OFF disproves that new decode kernel as a necessary cause. No code fix permitted/performed.
I/O stall during seventh TF request cleared (196.7s); writer D/rq_qos_wait and peer waits captured in A_io_stall.log, not classified as model deadlock.

First mismatch logits saved in A_first_mismatch_logits.json. Every returned mismatching token equals the saved FP32 logit argmax, ruling out SSE token extraction/sampler error. For p0/1/2/3, port chosen-minus-golden logit margins .810902/.022987/.152424/.088457; reference opposing margins .066246/.217216/.151389/.161526. Under identical token prefix this is a distribution/top1 change in the model numeric forward path. It is not proof of a specific new op failure.

## DSpark PP3 startup failure (E1–E5)
Root assertion kv_cache_utils.py:3112, before KV initialization/health. Failing component: V4.1 DSpark draft cache namespace across PP workers. llm_base_proposer.py:_get_model calls get_model without a prefix; get_model defaults prefix empty; DSparkV41DraftModel makes model.layers.0/1/2.attn.swa_cache on the LAST stage (draft SWA window133). Those names already identify BACKBONE layers0/1/2 on the FIRST stage (window128). Global per-worker cache merge therefore compares unequal dataclasses for identical names. The local duplicate guard misses this because they are on different PP stages. V4 DSpark uses prefix layers.{num_hidden_layers+stage}, i.e.40..42, avoiding this collision.
Primary source anchors: sm70/dspark.py:185,380,426; llm_base_proposer.py:2137; model_loader/__init__.py:133; kv_cache_utils.py:3112. CPU dataclass merge reproduction (not a full-server run): CUDA_VISIBLE_DEVICES='' [private path] [...]/spec_cache_repro.py ; exit1 with SAME assertion, full evidence spec_cache_repro.log. Full E1 env/resolved command and runtime stack E1.serve.log. No code fixed; owner should make draft cache names distinct (as V4 does) before rerun. All speculative text/code parity, acceptance and throughput UNMEASURED, not0.

All five caps separately reproduced the identical root assertion before health: E1 16:40:03Z, E2 16:48:53Z, E3 16:54:23Z, E4 16:59:54Z, E5 17:05:04Z (controller observation times). Exact inherited env, UUID mapping, launcher command and stack are preserved separately in E1.serve.log through E5.serve.log. Subsequent NCCL connection-closed / worker-died messages are shutdown consequences. Every server was stopped and GPU idleness checked before the next. All speculative p1/p2/p3 parity, acceptance, accepted tokens/step and throughput are unmeasured for every cap.

## F PP2/spill120 greedy
F p0 returned32 tokens; golden FAIL at zero-based index24 (5512 vs15225), matching A p0. All32 F IDs equal all32 A IDs: True. No separate spill/PP2 token divergence was observed for this prompt. Exact command F.serve.log and request F_p0.json.request.json. Numeric-path failure attribution remains bounded as above.

Campaign A–G completed under attention decode OFF. Overall acceptance: FAIL because all four PP3 golden checks and the PP2 p0 golden check failed, and all five DSpark caps failed startup. AM-13′/14′, all six context runs, four concurrent streams, both async comparisons, PP2 inference/needle, and nine semantic API checks completed successfully. No attention-OFF generation hang occurred.

Final audit at 2026-10-07T17:21:39Z: clean frozen HEAD d73d1616b174aec48787f9cdafc5d349b294539a, original extension hash unchanged; no compute processes; no owned server, memmon, controller or API-guard terminal session sessions; ports8041/8042 free. Exact host/process evidence final_state.json. Artifact consistency108/108 PASS; seven consolidated FP32 logit files, 2,441,879,512 bytes, verified F32 headers and SHA256 in quality_artifacts.json. Lock release is recorded below after final report generation. No model code changed or committed.

Resume: fix owners should resolve the DSpark draft/backbone cache namespace collision, then rerun each cap1..5 with the exact E_target/E1..E5 commands in commands.json and the same eager/block64/one-sequence/prefix-cache-off settings, preserving a fresh matched target baseline. Investigate golden distribution changes with controlled numeric-path comparisons; no single kernel is isolated. Attention-ON defects remain outside this override campaign and require their own fixed-build validation.

E2E-LOCK removed at 2026-10-07T17:23:28Z after a second all-GPU idle check and confirmation that all owned servers/controllers/monitors/guard sessions were gone. Other lanes may now resume GPU work.


Exact per-run controller commands, inherited test env, resolved launcher flags and UUIDs: `commands.json`, with full evidence in each `<tag>.serve.log`. All HTTP request bodies and raw responses are retained alongside the measurements. Artifact consistency audit: `artifact_audit.json`; this checks measurement integrity, not model correctness.
