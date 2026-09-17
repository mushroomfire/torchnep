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
    On a GPU, training compiles its kernels with `torch.compile`, which needs Triton (shipped with the CUDA and ROCm builds of PyTorch). Without it TorchNEP falls back to eager mode and says so in the log.
