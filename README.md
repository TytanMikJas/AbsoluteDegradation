# AbsoluteDegradation: A Physics-Inspired Synthetic Film-Degradation Pipeline and Archival Film Restoration Benchmark

**NeurIPS 2026, Evaluations and Datasets Track** · [Mikołaj Jastrzębski](https://www.mikjas.com), [Dawid Glinkowski](https://www.linkedin.com/in/dawid-glinkowski-444354224/), [Dawid Zieliński](https://www.linkedin.com/in/dawziel/), [Daniel Borkowski](https://www.linkedin.com/in/daniel-borkowski-/?isSelfProfile=false), Wojciech Kozłowski, Kamil Adamczewski · Wrocław University of Science and Technology

[![arXiv](https://img.shields.io/badge/arXiv-2607.02131-b31b1b.svg)](https://arxiv.org/abs/2607.02131)
[![Project page](https://img.shields.io/badge/Project-page-d9a441.svg)](https://www.mikjas.com/absolute-degradation)
[![Model](https://img.shields.io/badge/Model-DART-2b6cb0.svg)](https://github.com/TytanMikJas/DART-Degradation-Aware-Recurrent-Transformer)

[![AbsoluteDegradation](https://www.mikjas.com/og-absolute-degradation.png)](https://www.mikjas.com/absolute-degradation)

> **Status:** the degradation pipeline and the benchmark will be released by the end of October 2026.
> Watch or star the repository to be notified.

## Overview

Archival film restoration has two missing pieces. There is no paired training data, because the pristine version of a deteriorated film no longer exists. And there is no standard benchmark, because existing real-world datasets are small, low quality or hard to access.

AbsoluteDegradation addresses both:

1. **A degradation pipeline.** A modular, physics-inspired synthesiser that turns clean video into realistic, temporally coherent film damage, so restoration models can be trained with supervision.
2. **An archival benchmark.** 81,576 high-resolution frames from real archival footage, for consistent evaluation under real-world conditions.

## Highlights

- **Models the analog-to-digital chain.** Signal-dependent film grain, mean-reverting gate weave driven by Ornstein–Uhlenbeck processes, and parametric mechanical scratches.
- **Temporally coherent.** Jointly models 7 analog artifact families across time, with a severity curriculum.
- **Real archival benchmark.** 81,576 frames from 30 public-domain films (1896–1918) sourced from the Library of Congress.
- **Better transfer to real footage.** Restoration models trained with AbsoluteDegradation generalise better to real film across RTN, MambaOFR, BasicVSR++ and RVRT, and the benchmark exposes systematic failure modes of current methods.

Numbers and comparisons are in the [paper](https://arxiv.org/abs/2607.02131).

## Release plan

- [ ] Degradation pipeline (code and configs), Archival benchmark (81,576 frames) with download instructions, Evaluation scripts (30.10.2026)

## Related

[DART](https://github.com/TytanMikJas/DART-Degradation-Aware-Recurrent-Transformer) (ACCV 2026) is our degradation-aware restoration model, trained with this pipeline and evaluated on this benchmark.

## Citation

```bibtex
@inproceedings{jastrzebski2026absolutedegradation,
  title     = {{AbsoluteDegradation}: A Physics-Inspired Synthetic Film-Degradation Pipeline and Archival Film Restoration Benchmark},
  author    = {Jastrz{\k{e}}bski, Miko{\l}aj and Glinkowski, Dawid and Zieli{\'n}ski, Dawid and Borkowski, Daniel and Koz{\l}owski, Wojciech and Adamczewski, Kamil},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},
  year      = {2026}
}
```
