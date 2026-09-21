# Source layout

| File | Role |
|---|---|
| `__init__.py` | Public entry points: `train_nep`, `train_nep_sharded`, `predict_dataset`, `predict_dataset_sharded`, `export_valid_split`. |
| `data.py` | Reading extended-XYZ frames and `nep.in`; the NumPy neighbor builder used for training data. |
| `neighbor.py` | PyTorch linked-cell neighbor search, O(N), for large structures. |
| `model.py` | The trainable NEP4 model (`NEPModel`), per-element networks, ZBL, `slim_model`. |
| `ops.py` | Core kernels: Chebyshev and angular basis, descriptors, network evaluation, ZBL, analytical forces. |
| `nep.py` | `NEPCalculator`: load a `nep.txt`, compute energy, forces, virial and descriptors. |
| `predict.py` | Streamed, batched full-dataset prediction. |
| `train.py` | Single-device training: streaming data store, two-stage loop, schedulers, checkpoints. |
| `train_sharded.py` | Data-parallel multi-GPU / multi-node training. |
| `compiled_autograd.py` | `torch.compile` support for autograd forces. |
| `ase_calculator.py` | The ASE calculator `NEP`. |
| `plot.py` | `NEPPlotter` and the readers of the output files. |
| `constants.py` | Element table, covalent radii, NEP polynomial coefficients. |
