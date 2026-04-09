#!/usr/bin/env python
"""Example training script. Run from the example/ directory."""
import sys
sys.path.insert(0, "..")
from torchnep import train_nep

# Pure-PyTorch baseline run (no handwritten CUDA kernels in the training loop).
# Fixed seed makes results reproducible — use the same seed to compare
# pure-torch vs CUDA-kernel implementations for correctness verification.
train_nep(
    config_file="nep.in",
    data_file="train.xyz",
    output_dir="output_cuda",
    device="cuda",          # or "cpu" if no GPU available
    precision="float32",
    num_epochs=1000,           # short run for pure-torch baseline verification
    batch_size=32,
    lr=1e-2,
    print_interval=1,
    checkpoint_interval=100,
    restart=True,          # always fresh start for reproducibility
    seed=1,                # fixed seed: enables direct comparison with CUDA path
    pytorch_only=True
)
