# Optimizations, build by build

One page per build. Each page has one table of what was adopted (what it is, what it gave, what it cost, whether it
changes answers, the link to its code or record) and one table of what lost.

| Page | Build | Cards |
|---|---|---|
| [volta-split.md](volta-split.md) | Volta sPLIt: every expert from a compressed record, the fastest decode | 8 |
| [volta-max-uplift.md](volta-max-uplift.md) | Volta Max uPLIft: the most-used experts at NVFP4 precision, all on the cards | 8 |
| [volta-full-nvfp4.md](volta-full-nvfp4.md) | Volta Full NVFP4: every expert at NVFP4 precision, the overflow in host RAM (a reference build) | 8 |
| [volta-12x-amplify.md](volta-12x-amplify.md) | Volta 12X AmPLIfy: every expert at NVFP4 precision on the cards (repository not yet published) | 12 |
| [earlier-builds.md](earlier-builds.md) | The builds before these: llama.cpp, the first twelve-card port, the EXL3 engine, one board, Swift 1.5 | 4 to 12 |
| [what-did-not-work.md](what-did-not-work.md) | Every idea that lost, across all builds, grouped by why it lost | |

Headline numbers of each build: [../builds.md](../builds.md). When a given optimization is likely to apply to your
machine: [../method/decision-table.md](../method/decision-table.md).
