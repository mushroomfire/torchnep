#!/usr/bin/env python
"""
Basic training script. All config from nep.in.
Run from the example/ directory.
"""
import sys
sys.path.insert(0, "..")
from torchnep import train_nep

train_nep(
    config_file="nep_large.in",
    data_file="train_large.xyz",
    output_dir="output_large",
    device="cuda",
    pytorch_only=False,
    print_interval=1
)
