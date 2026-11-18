# Volta Full NVFP4: every optimization

Build: DeepSeek-V4.1-Flash-1017 on eight V100-SXM2-32GB with every routed expert at exactly the precision of
NVIDIA's NVFP4 pack. The most-used experts are on the cards; the rest are in pinned host RAM and are read over PCIe.
Repository: <https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100>.

This is a reference build: it exists to give an all-NVFP4 output to check other work against, and it accepts
being slow for that. Every figure below is MEASURED on 2026-11-09 unless the row says ESTIMATE, at a 1,522 MHz clock ceiling on the
eight-card server, or on one card of a four-card test machine at 150 W where the row says "one card". Source of every
row: the repository's [README](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100#readme)
and [docs/measurements.md](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100/blob/main/docs/measurements.md).

## What worked

| Item | What it is | Result | Changes answers? | Code |
|---|---|---|---|---|
| Exact NVFP4 to MXFP4 fold | NVIDIA's pack stores one FP8 scale per 16 values. Because the pack was converted losslessly from DeepSeek's MXFP4 experts, those scales fold back to one power-of-two scale per 32 values | 44,696,862,720 values bit-equal to the NVFP4 formula in FP32, relative error 0 | exact | [build_packs.py](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100/blob/main/harness/benchmarks/nvfp4_8r/build_packs.py), [accept_p1.py](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100/blob/main/harness/benchmarks/nvfp4_8r/accept_p1.py) |
| Scale range on Volta | The kernel rebases the scales into FP16. All 61,440 expert records had to fit | all accepted; widest exponent span 8 against a kernel limit of 29 | - | the same folder |
| The QPN expert layer | 4-bit weights, FP16 activations, FP16 tensor cores: the v100-skinny kernel as packaged by [1Cat-vLLM #742](https://github.com/1CatAI/1Cat-vLLM/pull/742) | one card: worst relative error 4.78e-4 against an FP32 reference; CUDA graphs replay bit-equal to eager; 131 microseconds per layer at 1 row, 280 to 348 at 6 rows | - | [engine/](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100/tree/main/engine) |
| Two tiers, bit-equal | The experts on the cards and the experts in host RAM give the same bits as all experts on one card, in decode and in prompt reading | bit-equal for natural and forced host routes | exact | the same |
| Zero-copy host experts | The host-RAM experts are packed once into a page-aligned arena, registered with CUDA and read in place by the kernel | works; its cost is the row "Reading experts from host RAM" below | exact | the same |
| Per-stage card reserve | One reserve for both stages ran out of memory on a 100K prompt. Reserves from sampled peaks (4.5 GiB on the first stage, 7.5 on the second) fixed it | a 498,099-token prompt read in 274.9 and 326.6 s without an out-of-memory error | - | [serve-nvfp4-pp2.yaml](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100/blob/main/engine/configs/serve-nvfp4-pp2.yaml) |
| Decode flat with context | - | 30.2 to 33.7 tok/s from 46 to 303,004 tokens of context | - | - |

## What did not work or is not good

| Item | Result | What follows |
|---|---|---|
| Speed | 28.2 and 30.2 tok/s on a legal-style short set, 18.8 and 20.1 on a general set | Use it as a reference, not as a server |
| **Reading experts from host RAM** | one host-RAM expert per token at 1 row adds about 975 microseconds to a layer; at 6 rows each costs about 1.0 to 1.4 ms per layer (about 3.5 to 4.9 GB/s over PCIe). Other miss paths on one card: copy-engine DMA 743.7 microseconds per expert, a gather on the card 1,014.8, an AVX2 host kernel 2,953.7 | ESTIMATE from those: no miss path comes within about 10 ms per round of all experts on the cards. **This number is why the later builds keep every expert in card memory** |
| General text | about half the legal-style speed (174.64 against 87.58 ms per round) | The most-used set was chosen from routing counts that are not published, and general text misses it more often. With no experts in host RAM the gap goes away |
| Prompt reading | 1,419 to 1,595 tok/s | Every expert not on the cards is staged from host RAM while reading |
| MXFP4 Engram tables from NVIDIA's pack | built and checked (rows bit-equal), but the 4-bit GGUF table decodes 4 to 5 times more rows per second | Not the default |
| Memory margin | highest card memory seen 32,086 MiB against a limit of about 32,494 | About 0.4 GiB of margin on the first stage |
| First boots | three failed fast: a host mapping requested on card 0 for every worker; staging buffers kept per layer instead of per card; one shared reserve for both stages | Each fixed at its cause |

## What this build teaches

PCIe is the wall. A V100 reads its own memory at up to 900 GB/s. The same card reads host memory at 3.5 to 4.9 GB/s.
Any design that leaves weights a decode round needs on the far side of PCIe pays about a millisecond per expert per
layer, and a round touches several experts in each of 40 layers. The measured answer was to stop trying to make the
miss path fast and to make the miss not happen: first by choosing which experts get native precision
([Volta Max uPLIft](volta-max-uplift.md)), then by adding a third board
([Volta 12X AmPLIfy](volta-12x-amplify.md)).
