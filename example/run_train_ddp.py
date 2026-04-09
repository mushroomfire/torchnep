#!/usr/bin/env python
"""Multi-GPU DDP training. Run from the example/ directory.

Single node, multiple GPUs:
    python run_train_ddp.py

Multi-node via torchrun (e.g. 2 nodes × 4 GPUs):
    # On node 0:
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
             --master_addr=node0_ip --master_port=29500 \
             run_train_ddp.py

    # On node 1:
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
             --master_addr=node0_ip --master_port=29500 \
             run_train_ddp.py
"""
import os
import sys
sys.path.insert(0, "..")

# Check if launched via torchrun (sets RANK, WORLD_SIZE, LOCAL_RANK)
if "RANK" in os.environ:
    # torchrun multi-node mode
    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    from torchnep.train import _ddp_worker
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)

    # Call worker directly (dist already initialized by torchrun)
    from torchnep.data import parse_nep_in, read_xyz
    from torchnep.model import NEPModel
    from torchnep.train import (preprocess_structures, compute_energy_shift,
                                GPUDataStore, compute_q_scaler)
    import numpy as np, time

    _ddp_worker(
        rank=local_rank, world_size=world_size,
        config_file="nep.in", data_file="train.xyz",
        output_dir="output_ddp", precision="float32",
        num_epochs=200, batch_size=32, lr=1e-2, print_interval=10,
    )
else:
    # Single-node spawn mode
    from torchnep import train_nep_ddp

    train_nep_ddp(
        config_file="nep.in",
        data_file="train.xyz",
        output_dir="output_ddp",
        precision="float32",
        num_epochs=200,
        batch_size=32,
        lr=1e-2,
        print_interval=10,
        num_gpus=None,  # auto-detect
    )
