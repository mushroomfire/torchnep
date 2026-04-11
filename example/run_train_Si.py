#!/usr/bin/env python
"""Train NEP for silicon."""
import sys
sys.path.insert(0, "..")
from torchnep import train_nep

train_nep(
    config_file="nep_Si.in",
    data_file="train_Si.xyz",
    output_dir="output_Si",
    device="cuda",
    pytorch_only=True,
    use_autograd_forces=True
)
