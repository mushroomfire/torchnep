# Installation

TorchNEP needs only `torch >= 2.0` and `numpy`, and installs neither of them itself: the right PyTorch build depends on your hardware.

## 1. Install PyTorch

Pick the build that matches your GPU (CUDA or ROCm) or CPU from the [PyTorch installation guide](https://pytorch.org/get-started/locally/). NumPy comes with it.

## 2. Install TorchNEP

=== "PyPI"

    ```bash
    pip install torchnep -U
    ```

=== "Development version"

    ```bash
    pip install git+https://github.com/mushroomfire/torchnep.git
    ```

=== "From source"

    ```bash
    git clone https://github.com/mushroomfire/torchnep.git
    cd torchnep
    pip install .
    ```

## Optional extras

| Extra | Adds | Needed for |
|---|---|---|
| `torchnep[ase]` | ASE | the [ASE calculator](../guide/ase.md) |
| `torchnep[plot]` | matplotlib, polars | [plotting](../guide/plotting.md) |
| `torchnep[all]` | everything above | |

```bash
pip install "torchnep[all]"
```

## Check the installation

```bash
python -c "import torch, torchnep; print(torchnep.__version__, torch.cuda.is_available())"
```

`torch.cuda.is_available()` is also `True` on AMD GPUs with a ROCm build of PyTorch.

!!! tip "torch.compile"
    On a GPU, training compiles its kernels with `torch.compile`, which needs Triton (shipped with the CUDA and ROCm builds of PyTorch) and a C compiler (below). Without them TorchNEP falls back to eager mode and says so in the log.

## A C compiler on the GPU machine

Triton builds a small C launcher the first time it runs on a machine and keeps it in `~/.triton/cache`. Since PyTorch 2.12, some built-in CUDA operations run through Triton even without `torch.compile`, so the GPU machine needs a C compiler at least once. Cluster compute nodes often have none. Load one in the job script before Python starts:

```bash
module load gcc          # or: export CC=/path/to/gcc CXX=/path/to/g++
```

Without one, TorchNEP warns at start-up, runs those operations with the regular CUDA kernels and keeps `torch.compile` off. Training and prediction still work, only without the compiled speed-up.

## Running the test suite

The tests need a source checkout and the development extras:

```bash
git clone https://github.com/mushroomfire/torchnep.git
cd torchnep
pip install -e ".[dev]"
```

**On any machine (CPU).** This is what CI runs. Tests that need a GPU are skipped.

```bash
TORCHNEP_TEST_DDP=1 pytest -n auto      # about a minute on an 8-core laptop
```

`TORCHNEP_TEST_DDP=1` adds the multi-process tests, which start two processes with `torchrun`. `-n auto` spreads the tests over all CPU cores.

**On a GPU machine: the full suite.** Some paths only run on a GPU: training with `torch.compile` (the default on GPUs) against eager mode, the CUDA-only safeguards and multi-GPU training. Run everything on a node with two GPUs and a C compiler:

```bash
module load gcc                          # if the node has no C compiler (see above)
TORCHNEP_TEST_DDP=1 pytest -rs           # one process: the GPU tests share the devices
```

On 2 × NVIDIA GH200 this takes about 25 minutes (measured with coverage on) and runs every test but one (it needs an Apple GPU). With a single GPU the two-GPU tests are skipped; `-rs` lists every skipped test and why.

**Coverage.** Add `--cov=torchnep --cov-report=term-missing` to either command to see which lines the run did not reach.
