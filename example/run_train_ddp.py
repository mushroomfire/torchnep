#!/usr/bin/env python
"""Multi-GPU DDP training. Run from the example/ directory.

Training parameters (epoch, batch, lr, etc.) are read from nep.in.

Usage 1 — Single node, multiple GPUs (most common):
    torchrun --standalone --nproc_per_node=2 run_train_ddp.py

Usage 2 — Single node, auto-spawn (no torchrun needed):
    python run_train_ddp.py

Usage 3 — Multi-node via SLURM (e.g. 2 nodes × 4 GPUs = 8 GPUs total):
    See example SLURM script: submit_ddp.sh
"""
import os
import sys
sys.path.insert(0, "..")

# Check if launched via torchrun (sets RANK, WORLD_SIZE, LOCAL_RANK)
if "RANK" in os.environ:
    # torchrun / SLURM mode
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    from torchnep.train import _ddp_worker

    _ddp_worker(
        rank=local_rank, world_size=world_size,
        config_file="nep.in", data_file="train.xyz",
        output_dir="output_ddp",
        pytorch_only=False,  # use CUDA kernels; set True for pure-PyTorch
    )
else:
    # Single-node spawn mode (python run_train_ddp.py)
    from torchnep import train_nep_ddp

    train_nep_ddp(
        config_file="nep.in",
        data_file="train.xyz",
        output_dir="output_ddp",
        pytorch_only=False,  # use CUDA kernels; set True for pure-PyTorch
    )
