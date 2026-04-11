#!/usr/bin/env python
"""
Basic training script. All config from nep.in.
Run from the example/ directory.
"""
import sys
sys.path.insert(0, "..")
from torchnep import train_nep

train_nep(
    config_file="nep.in",
    data_file="train.xyz",
    output_dir="output",
    device="cuda",
)
