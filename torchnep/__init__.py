# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Yongchao Wu.
# Part of the torchnep project — https://github.com/mushroomfire/torchnep.

__version__ = "1.0.1"

# Public API: only the three high-level entry points users actually call.
# Internal building blocks (NEPModel, NEPCalculator, data readers, ...) remain
# importable from their submodules but are intentionally not re-exported here.
from .predict import predict_dataset
from .train import train_nep
from .train_sharded import train_nep_sharded

__all__ = [
    "predict_dataset", "train_nep", "train_nep_sharded",
]
