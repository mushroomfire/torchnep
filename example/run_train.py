#!/usr/bin/env python
"""Example training script. Run from the example/ directory."""
import sys
sys.path.insert(0, "..")
from torchnep import train_nep


# Pure autograd reference: forces via torch.autograd.grad (create_graph=True)
# This is the gold-standard for verifying gradient correctness.
train_nep(
    config_file="nep.in",
    data_file="train.xyz",
    output_dir="output_autograd",
    device="cuda",
    precision="float32",
    num_epochs=1000,
    batch_size=32,
    lr=1e-2,
    print_interval=1,
    checkpoint_interval=100,
    restart=False,
    pytorch_only=True,
    use_autograd_forces=True,
    # Stability defaults: max_grad_norm=5.0, adam_eps=1e-4, warmup_epochs=10
)

# # Analytical force path (faster, no create_graph)
# train_nep(
#     config_file="nep.in",
#     data_file="train.xyz",
#     output_dir="output_analytical",
#     device="cuda",
#     precision="float32",
#     num_epochs=1000,
#     batch_size=32,
#     lr=1e-2,
#     print_interval=10,
#     checkpoint_interval=100,
#     restart=False,
#     pytorch_only=True,
#     use_autograd_forces=False,
# )
