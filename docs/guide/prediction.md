# Prediction

## One structure

`NEPCalculator` loads a `nep.txt` and evaluates single structures:

```python
import numpy as np
from torchnep.nep import NEPCalculator

calc = NEPCalculator("nep.txt", device="cuda")
result = calc.compute(
    species=["Cr", "Cr", "Ni"],
    positions=np.array([[0, 0, 0], [1.5, 0, 0], [3, 0, 0]]),
    cell=np.eye(3) * 6.0,
)
result["energy"]    # (N,)   per-atom energy, eV — sum for the total
result["forces"]    # (N, 3) forces, eV/Å
result["virial"]    # (N, 9) per-atom virial, eV
```

`return_components=True` splits every quantity into the neural-network part and the ZBL part:

```python
result = calc.compute(species, positions, cell, return_components=True)
result["energy_nep"], result["energy_zbl"]    # sum to result["energy"]
```

`calc.get_descriptor(species, positions, cell)` returns the scaled per-atom descriptors. For large MD cells, `calc.compute_tiled(...)` evaluates the structure in blocks with bounded memory.

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

It writes `energy_train.out`, `force_train.out`, `virial_train.out`, `stress_train.out` and, with `output_descriptor`, `descriptor.out` — the same layout as a training run, so the [plots](plotting.md) work on them too.

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
