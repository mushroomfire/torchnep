<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mushroomfire/torchnep/master/docs/assets/logo-mark-dark.png">
    <img src="https://raw.githubusercontent.com/mushroomfire/torchnep/master/docs/assets/logo-mark.png" alt="TorchNEP logo" width="110">
  </picture>
  <br>
  TorchNEP
</h1>

<p align="center">Train NEP machine-learned interatomic potentials in PyTorch — the models run directly in GPUMD.</p>

<p align="center">
  <a href="https://pypi.org/project/torchnep/"><img src="https://img.shields.io/pypi/v/torchnep?logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://pypi.org/project/torchnep/"><img src="https://img.shields.io/pypi/pyversions/torchnep?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://github.com/mushroomfire/torchnep/actions/workflows/test.yml"><img src="https://github.com/mushroomfire/torchnep/actions/workflows/test.yml/badge.svg" alt="Tests"></a>
  <a href="https://mushroomfire.github.io/torchnep/"><img src="https://img.shields.io/badge/docs-online-18202C" alt="Documentation"></a>
  <a href="https://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/license-GPLv3-blue" alt="License: GPL v3"></a>
  <a href="https://pypi.org/project/torchnep/"><img src="https://img.shields.io/pypi/dm/torchnep" alt="Downloads"></a>
</p>

<p align="center">
  <a href="https://mushroomfire.github.io/torchnep/"><b>Documentation</b></a> ·
  <a href="https://mushroomfire.github.io/torchnep/getting-started/quickstart/">Quick start</a> ·
  <a href="https://github.com/mushroomfire/TorchNEP_models">Examples</a> ·
  <a href="https://mushroomfire.github.io/torchnep/citation/">Citation</a>
</p>

TorchNEP is a from-scratch implementation of [NEP4](https://gpumd.org/theory/nep.html), the neuroevolution potential architecture.

- **GPUMD-compatible** — `nep.txt` files load directly into GPUMD for molecular dynamics
- **Two-stage training** — a force-focused stage, then an energy-focused stage
- **Multi-GPU and multi-node** — data-parallel training with near-linear scaling
- **Fast on NVIDIA and AMD** — `torch.compile` with automatic backend selection, tuned on CUDA and ROCm
- **Memory-friendly** — the dataset stays in host memory; GPU memory scales with the batch, not the dataset
- **Fine-tuning, ZBL, plots** — start from any `nep.txt`, add short-range repulsion, plot every run

<p align="center">
  <img src="https://raw.githubusercontent.com/mushroomfire/torchnep/master/docs/assets/speed_scaling.png" alt="Training speed and scaling" width="90%">
</p>

## Installation

Install the [PyTorch build](https://pytorch.org/get-started/locally/) for your hardware first, then:

```bash
pip install torchnep -U
```

Optional extras: `torchnep[ase]` (ASE calculator), `torchnep[plot]` (figures), `torchnep[all]`.

## Quick start

```python
from torchnep import train_nep

train_nep("nep.in", "train.xyz", output_dir="output", valid_ratio=0.1)
```

The best model is written to `output/nep_best.txt`. The [documentation](https://mushroomfire.github.io/torchnep/) covers `nep.in`, the training data format, multi-GPU training, restart and fine-tuning, prediction, the ASE calculator and plotting.

## Citation

```bibtex
@misc{wu2026torchne,
      title={TorchNEP: Ultra-Efficient and Accurate Training of Neuroevolution Potentials},
      author={Yong-Chao Wu and Xiaoya Chang and Tero Mäkinen and Amin Esfandiarpour and Jian-Li Shao and Tapio Ala-Nissila and Zheyong Fan and Mikko Alava},
      year={2026},
      eprint={2606.19557},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph},
      url={https://arxiv.org/abs/2606.19557},
}
```
