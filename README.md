# Volta V100 research

**DeepSeek-V4.1-Flash on Tesla V100 cards from 2018: the architecture, the builds, every speed optimization that
won or lost, the testing method, and the source research.**

Built by **Nick Mermiges, Esq.**, founder of [PLI Labs](https://proprietarylegal.ai/labs.html) · [proprietarylegal.ai](https://proprietarylegal.ai)

This repository is the single home of PLI Labs' research on the Tesla V100 (Volta, sm_70). It holds documents, not
engines: the code of each build lives in that build's own repository, linked from [builds.md](builds.md).

**Start with the long read: [index.md](index.md)** (also the front page of this repository's GitHub Pages site).

## The map

| Path | What it is |
|---|---|
| [index.md](index.md) | The post: Volta's architecture, how it compares with newer chips, how it was made to work under 1Cat-vLLM, every optimization, the testing method, when each optimization applies, how to build a kernel, the hardware and its drawbacks |
| [builds.md](builds.md) | Every final build with its link, what it runs on and its headline measured numbers |
| [hardware.md](hardware.md) | The hardware list: the parts for a four-card SXM2 NVLink board and its host, rough prices, search links, cooling options, photos, what goes wrong |
| [optimizations/](optimizations/README.md) | One page per build with its full table of what was adopted and what lost; [what-did-not-work.md](optimizations/what-did-not-work.md) across all builds |
| [method/testing.md](method/testing.md) | The testing method as a reusable document, including what went wrong with it |
| [method/kernel-building.md](method/kernel-building.md) | How to build a custom kernel for sm_70 |
| [method/decision-table.md](method/decision-table.md) | When a given optimization is likely to apply, keyed on your situation |
| [research/](research/README.md) | The source research: the Volta architecture notes, the kernel research, the eight-card time budget; each file with its origin |
| [sources.md](sources.md) | Every external source with its link |
| [CREDITS.md](CREDITS.md) | Whose work this stands on, by public handle, with links |
| `tools/check_links.py` | The link checker this repository is verified with |

## The builds

| Build | Cards | Routed experts | Repository |
|---|---|---|---|
| Volta sPLIt | 8 | compressed (2-bit), all on the cards | [link](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4) |
| Volta Max uPLIft | 8 | the most-used at NVIDIA's NVFP4 precision, all on the cards | [link](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4) |
| Volta Full NVFP4 | 8 | all at NVFP4 precision, the overflow in host RAM | [link](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100) |
| Volta 12X AmPLIfy | 12 | all at NVFP4 precision, all on the cards | [link](https://github.com/ProprietaryLegal/Volta-12X-AmPLIfy-DeepSeek-V4.1-Flash-1017-NVFP4-12xV100-SXM2-PP3xTP4) |

## How to read the numbers

Every number is labelled **MEASURED** (on our hardware, with its date, power limit and clock ceiling), **UPSTREAM**
(someone else's, linked) or **ESTIMATE** (arithmetic, with its basis). An estimate or a plan is never written as a
result. A configuration that did not pass its own rule says "quality not established". The failures are told as
plainly as the wins.

## Checking this repository

```sh
python3 tools/check_links.py             # every internal link and heading anchor
python3 tools/check_links.py --external  # also request every external link once
```

The site is plain GitHub Pages with a stock theme (`_config.yml`, one small stylesheet in `assets/css/`). To build it
locally with the same toolchain GitHub uses:

```sh
mkdir -p _site
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD:/src:ro" -v "$PWD/_site:/out" -e PAGES_REPO_NWO=OWNER/REPO \
  --entrypoint /bin/sh ghcr.io/actions/jekyll-build-pages:v1.0.13 \
  -c 'cd /tmp && github-pages build -s /src -d /out'
```

## The builder skill

The method behind these builds is packaged as a skill for coding agents, **PLI Volta Builder**:
<https://github.com/ProprietaryLegal/pli-volta-builder>.

## Work with PLI Labs

[PLI Labs](https://proprietarylegal.ai/labs.html) is the open research arm of Proprietary Legal Intelligence. PLI
builds AI around an organization's own work product. If you want private, on-premises model serving of your own,
from choosing the hardware to the measured stack, we consult on exactly that:
[proprietarylegal.ai](https://proprietarylegal.ai).

## Licence

Original documentation and code are Apache-2.0 ([LICENSE](LICENSE)); see [NOTICE](NOTICE). Third-party work is
linked and credited, not copied ([CREDITS.md](CREDITS.md)). No model weights, compiled libraries, routing data or
prompt corpora are distributed here.
