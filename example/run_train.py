#!/usr/bin/env python
"""
Single-GPU:  python run_train.py
Multi-GPU :  torchrun --nproc_per_node=N run_train.py
"""

from torchnep import train_nep

# train_nep(
#     config_file="nep_AlO.in",
#     data_file="train_AlO.xyz",
#     output_dir="output_cuda_AlO",
#     device="cuda",
#     pytorch_only=False,
#     print_interval=1,
#     use_compile=True
# )

# train_nep(
#     config_file="nep_AlO.in",
#     data_file="train_AlO.xyz",
#     output_dir="output_analytical_AlO",
#     device="cuda",
#     pytorch_only=True,
#     use_autograd_forces=False,
#     print_interval=1,
#     use_compile=True
# )

train_nep(
    config_file="nep_AlO.in",
    data_file="train_AlO.xyz",
    output_dir="output_autograd_AlO",
    device="cuda",
    pytorch_only=True,
    use_autograd_forces=True,
    print_interval=1,
    use_compile=True
)