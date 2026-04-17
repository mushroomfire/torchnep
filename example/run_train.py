#!/usr/bin/env python
"""
Single-GPU:  python run_train.py
Multi-GPU :  torchrun --nproc_per_node=N run_train.py
"""

from torchnep import train_nep, train_nep_sharded

# Single-GPU example — "auto" picks "cuda" (few types) or "fast" (many types).
# train_nep(
#     config_file="nep_AlO.in",
#     data_file="train_AlO.xyz",
#     output_dir="output_AlO",
#     device="cuda",
#     backend="auto",           # "auto" | "loop" | "fast" | "cuda"
#     print_interval=1,
#     use_compile=True,
# )

train_nep_sharded(
    config_file="nep_Si.in",
    data_file="train_Si.xyz",
    output_dir="output_Si",
    backend="auto",
    use_autograd_forces=True,
    print_interval=1,
    use_compile=True,
)
