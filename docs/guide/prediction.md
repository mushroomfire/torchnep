# Prediction

## One structure

`NEPCalculator` loads a `nep.txt` and evaluates single structures:

```python
import numpy as np
from torchnep.nep import NEPCalculator

a = 3.515
positions = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]) * a
positions[0, 0] += 0.1                      # displace one atom by 0.1 A

calc = NEPCalculator("nep.txt", device="cuda")
result = calc.compute(species=["Ni"] * 4, positions=positions, cell=np.eye(3) * a)

print(f"total energy {float(result['energy'].sum()):.4f} eV")
print(np.round(result["forces"].cpu().numpy(), 4))
```

```text
total energy -21.8318 eV
[[-0.9239 -0.     -0.    ]
 [ 0.4405  0.     -0.    ]
 [ 0.4405 -0.      0.    ]
 [ 0.0428  0.      0.    ]]
```

`compute` returns `energy` (N,) per-atom energies in eV — sum them for the total — `forces` (N, 3) in eV/Å and `virial` (N, 9) per-atom virials in eV, all as torch tensors on the calculator's device.

`return_components=True` splits every quantity into the neural-network part and the ZBL part:

```python
result = calc.compute(species, positions, cell, return_components=True)
result["energy_nep"], result["energy_zbl"]    # sum to result["energy"]
```

`calc.get_descriptor(species, positions, cell)` returns the scaled per-atom descriptors, `(N, dim)` as NumPy. For large MD cells, `calc.compute_tiled(...)` evaluates the structure in blocks with bounded memory.

## Full-dataset prediction

`predict_dataset` predicts every frame of an `.xyz` file, writes GPUMD-format output files and prints the E/F/V RMSE and MAE when the file has reference labels.

```python
from torchnep import predict_dataset

predict_dataset(
    "nep.txt",
    "test.xyz",
    output_dir="results",
    output_descriptor=0,   # 0 off, 1 per-frame mean, 2 per-atom (as GPUMD)
    batch_size=None,       # auto from free GPU memory, or an int
    chunk_atoms=None,      # atoms per streamed chunk (default 200000)
)
```

It writes `energy_train.out`, `force_train.out`, `virial_train.out`, `stress_train.out` and, with `output_descriptor`, `descriptor.out` — the same layout as a training run, so the [plots](plotting.md) work on them too. A run over 623 frames with reference labels ends like this:

```text
  index_xyz:     0.0s   (623 frames, 67416 atoms, 1 chunk(s) of ~200000 atoms, energy label: energy)
  batch_size:  auto -> 67 (free 10.6 GiB, ~13761 KiB/frame est.)
  read:          0.1s
  neighbors:     1.2s
  compute:       1.0s   (first chunk incl. warm-up 1.0s)
  write:         0.1s
  TOTAL:         2.7s   -> pred_test/(energy|force|virial|stress)_train.out
  ----------------------------------------------------------
                             RMSE          MAE
  Energy (eV/atom)       0.004032     0.003075   (623 frames)
  Force  (eV/A)          0.122787     0.086898   (67416 atoms)
  Virial (eV/atom)       0.027611     0.016967   (623 frames)
  ----------------------------------------------------------
```

The file is processed in chunks of about `chunk_atoms` atoms: read, build neighbor lists, predict in device-sized batches, append the rows. Host memory is bounded by the chunk and GPU memory by the batch, so any dataset fits; a batch that runs out of memory is retried at half the size. `dtype` is `"float32"` by default.

## Multi-GPU prediction

`predict_dataset_sharded` takes the same arguments and writes the same files, with one process per GPU:

```python
from torchnep import predict_dataset_sharded

predict_dataset_sharded("nep.txt", "huge.xyz", output_dir="results")
```

```bash
torchrun --standalone --nproc_per_node=8 predict.py
```

Under SLURM, launch it like [multi-node training](distributed.md#several-nodes-with-slurm).
