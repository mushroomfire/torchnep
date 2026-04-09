#!/usr/bin/env python
"""Example training script. Run from the example/ directory."""
import sys
sys.path.insert(0, "..")
from torchnep import train_nep

train_nep(
    config_file="nep.in",
    data_file="train.xyz",
    output_dir="output",
    device="cuda",          # "cpu" for no GPU
    precision="float32",
    num_epochs=2000,         # 200 epochs is usually enough for convergence
    batch_size=32,          # 32 structures/batch (good GPU utilization)
    lr=1e-2,                # Adam default, cosine decay to 1e-4
    print_interval=10,
)
