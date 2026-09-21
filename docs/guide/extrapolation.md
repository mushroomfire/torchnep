# Extrapolation grade (experimental)

!!! warning "Experimental — only on the `feat/extrapolation-gamma` branch"
    This feature is not part of a release yet. Install the branch to try it:

    ```bash
    pip install "git+https://github.com/mushroomfire/torchnep.git@feat/extrapolation-gamma"
    ```

The extrapolation grade γ tells how far an atomic environment lies outside the training set of a model, **with one model** — no committee of models is needed. It is computed for every atom of any structure, for example the frames of an MD run, and can pick the frames worth labelling with DFT.

## What it measures

Every atom is represented by the gradient of its NEP energy with respect to its element's network weights, `b = dE_i / d(w0, b0, w1)` — the model linearised in its parameters. From the training set, TorchNEP keeps for every element the **active set**: the training atoms whose rows span the largest volume (MaxVol). A new atom is written in the basis of the active rows, `b = c A`, and its grade is

```text
gamma = max_j |c_j|
```

- **γ ≤ 1**: the environment lies inside the region spanned by the training set (interpolation). Every training atom has γ ≤ `tol` (1.01) by construction.
- **γ > 1**: it lies outside; putting it into the active set would multiply the spanned volume by γ.
- A structure's grade is the largest grade of its atoms.

The rows are numerically far from full rank — their singular values decay smoothly over many orders of magnitude — so γ is computed in the leading subspace of each element, the directions whose singular value is above `rcond` × the largest. What lies outside that subspace is reported separately as **γ_res**: the out-of-subspace norm of the atom's row divided by the largest one met in the training set (> 1: more weight outside the training subspace than any training atom).

## Quick start

Three steps: build the active set from the training set once per model, grade new structures, choose the ones to label.

```python
from torchnep.extrapolation import build_active_set, compute_gamma, select_structures

# 1. once per trained model
build_active_set("nep.txt", "train.xyz", "active_set.pt")

# 2. grades of any structures
g = compute_gamma("nep.txt", "active_set.pt", "md.xyz", output_file="gamma.npz")

# 3. choose up to 200 frames for DFT
res = select_structures("nep.txt", "active_set.pt", "md.xyz",
                        output_xyz="to_dft.xyz", max_frames=200)
```

Output of these three calls for a Cr-Co-Ni model and its training set (3030 frames), grading and choosing from a pool of 2175 structures of another dataset, on one GPU:

```text
  active set: 3030 frames, 257880 atoms, 3 elements, K = 80 x (63 + 2) = 5200, rcond 0.0001, tol 1.01
  pass 1 (Gram matrices, 3030 frames): 2.9s
    Cr  rows     82288  rank   345 / 5200
    Co  rows     85112  rank   308 / 5200
    Ni  rows     90480  rank   289 / 5200
  subspaces and seed active sets: 11.3s
  pass 2 (MaxVol, 3030 frames): 0.6s
  check 1: 201 atoms above 1.01 (largest 1.7668) -> 87 swaps (0.3s)
  check 2: 3 atoms above 1.01 (largest 1.0759) -> 18 swaps (0.1s)
  check 3: every training atom has gamma <= 1.01 (largest 1.0065; 0.1s)
  TOTAL: 16.3s -> active_set.pt
  gamma: 2175 frames in 1.5s; frames with gamma > 1.01: 786, median 0.882, max 14.2
  select: 786 candidate frames (89032 atoms)
  select: 200 frames chosen in 3.3s
```

`rank` is the dimension of the element's subspace (its number of active rows) out of K. Of the 786 pool frames above `tol`, the 200 chosen ones start with grades 14.2, 5.0, 10.6, 3.3, 7.6 — each recomputed after the frames before it were added, which is why they do not decrease monotonically.

## Build the active set

`build_active_set(model_file, xyz_file, output_file)` streams the training set:

1. on a random sample of frames, the Gram matrix of every element's rows — its leading eigenvectors give the subspace — and a random sample of rows that seeds the active set;
2. on all frames, MaxVol swaps of every atom whose grade exceeds `tol`;
3. check passes over all frames: the atoms still above `tol` (a row passed early can exceed it after later swaps) are swapped in, until a pass finds none — after 3 to 9 passes on the training sets we tried. The log reports the result and the largest training grade, e.g. `check 3: every training atom has gamma <= 1.01`; if `max_passes` runs out first, a last check reports how many atoms remain above `tol` (typically a handful, with grades just above it).

The descriptors are kept in host memory between passes when they fit in 25 % of the free memory, so the check passes cost only the projection (`TORCHNEP_GAMMA_CACHE_GB` sets another budget, `0` disables it).

| Argument | Default | Meaning |
|---|---|---|
| `rcond` | `1e-4` | Subspace: directions whose singular value is above `rcond` × the largest. Smaller keeps more directions (a larger active set). |
| `tol` | `1.01` | MaxVol threshold; every training atom ends with γ ≤ `tol`. |
| `sample_frames` | `50000` | Frames used for the Gram matrices and the seed (all frames if fewer). |
| `init_rows` | `4` | Seed: LU pivoting over `init_rows` × r sampled rows. |
| `max_passes` | `10` | Check passes (with swaps) at most. |
| `dtype` | `"float64"` | Descriptor precision; the subspace and MaxVol always run in float64. |
| `precision` | `"float64"` | `"float32"`: grade the rows of passes 2 and 3 in float32 first and regrade in float64 only those near or above `tol`. The swaps, and so the active set, are the same; much faster on GPUs with slow float64 (most consumer cards). |
| `chunk_atoms` | 200 000 | Atoms read per chunk (host memory). |
| `seed` | `0` | Random sample of frames and rows. |

The active set belongs to one model: it stores a fingerprint of the model's parameters, and loading it with another model raises an error. Rebuild it after every retraining.

## Grade structures

`compute_gamma(model_file, active_set, xyz_file)` returns a dict of NumPy arrays and, with `output_file`, saves it as `.npz`:

| Key | Shape | Meaning |
|---|---|---|
| `gamma` | frames | largest γ of the frame's atoms |
| `gamma_res` | frames | largest γ_res of the frame's atoms |
| `atom` | frames | index (in the frame) of the atom with the largest γ (of atoms with equal grades, e.g. symmetric sites, any may come out) |
| `natoms` | frames | atoms per frame |
| `gamma_atoms`, `gamma_res_atoms` | atoms | every atom, in file order (`per_atom=True`) |

Atoms of an element that has no atom in the training set get γ = ∞.

`precision="float32"` grades in float32: γ stays within 1e-4 and γ_res within a few 1e-3 (relative) of the float64 values, and grading runs several times faster on GPUs whose float64 throughput is low (most consumer cards); on GPUs with fast float64 it brings nothing. `dtype` sets the precision of the descriptors themselves.

## Choose new structures

`select_structures(model_file, active_set, xyz_file, output_xyz=..., max_frames=...)` makes a greedy D-optimal choice. The frames whose grade lies in (`gamma_min`, `gamma_max`] are visited from the highest grade down; a frame is taken only if its grade, **recomputed against the active set extended by the frames taken before it**, still exceeds `gamma_min` — so of a group of near-identical frames, only the first is taken. The chosen frames are copied verbatim to `output_xyz`.

| Argument | Default | Meaning |
|---|---|---|
| `max_frames` | all | Stop after this many frames. |
| `gamma_min` | `tol` of the active set | Frames at or below it are not candidates. |
| `gamma_max` | none | Skip frames above it, e.g. broken MD frames. |
| `output_active_set` | none | Save the active set extended by the chosen frames. |

It returns `index` (chosen frames, in order of choice), `gamma_at_choice` and `gamma` (grades of all frames against the original active set).

## Several GPUs

`build_active_set_sharded`, `compute_gamma_sharded` and `select_structures_sharded` take the same arguments (without `device`) and run one process per GPU, launched like [multi-GPU training](distributed.md):

```python title="active.py"
from torchnep.extrapolation import build_active_set_sharded, compute_gamma_sharded

build_active_set_sharded("nep.txt", "train.xyz", "active_set.pt")
compute_gamma_sharded("nep.txt", "active_set.pt", "md.xyz", output_file="gamma.npz")
```

```bash
torchrun --standalone --nproc_per_node=8 active.py
```

Any launcher that sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `LOCAL_WORLD_SIZE`, `MASTER_ADDR` and `MASTER_PORT` works as well, for example one `srun` task per GPU on one or several nodes:

```bash
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
srun --ntasks-per-node=8 --gpus-per-node=8 bash -c \
  'RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID WORLD_SIZE=$SLURM_NTASKS \
   LOCAL_WORLD_SIZE=$SLURM_NTASKS_PER_NODE python active.py'
```

Every process must see all GPUs of its node — it takes GPU number `LOCAL_RANK` — or the processes fall back to slower collectives through the CPU. The processes of a node share its CPU cores for reading the file (`TORCHNEP_PREPROC_WORKERS` sets the number of reader processes per GPU).

Every rank streams its own share of the frames. For the active set, the Gram matrices are summed on the rank that owns each element, the ranks run MaxVol on their shares, the owners merge the ranks' active rows and the check passes run as in the single-GPU version — every training atom still ends with γ ≤ `tol`. Grades equal the single-GPU ones to round-off and the choice of structures is identical. Rank 0 writes the output files.

## Performance

The file is read by worker processes one chunk ahead while the GPU computes the neighbor lists, the descriptors and the grades of the current chunk: about 2 × 10⁵ atoms per second per GPU. Measured with a 16-element model (K = 80 × (35 + 2) = 2960) and its training set of 105 464 frames (6.9 million atoms), on GPUs with fast float64:

| | 1 GPU | 8 GPUs |
|---|---|---|
| `build_active_set` (converged) | 212 s | 62 s |
| `compute_gamma`, all 105 464 frames | 29 – 36 s | 9 – 20 s |

The upper figures of `compute_gamma` include the start-up of a fresh run. On one GPU the build spends 13 s in pass 1 (50 000 sampled frames), 45 s on the subspaces and seeds — mostly the eigendecompositions of the K × K Gram matrices, about 2.5 s per element, which the sharded build divides among the GPUs — 83 s in pass 2 and 6 – 11 s per check pass (9 passes). Choosing 200 of the 2175 frames of the example above takes 3.3 s.

## Limitations

- **Experimental.** On one test (a Cr-Co-Ni model and a pool of structures from another dataset), choosing frames by γ reduced the error on held-out structures as much as choosing them by a committee of four models, and much more than choosing them at random. It has not been used in production active learning yet.
- **Memory while building:** the Gram matrices take K² × 8 bytes per element on the GPU (K = neurons × (descriptor size + 2); 70 MB for K = 2960). With the sharded build, each element's matrix is summed on one rank.
- **γ_res** is computed as `|b|² − |b V|²`, which costs digits: it is accurate to about 1e-8 relative, plenty for a grade.
- The active set depends on the model; rebuild it after retraining.
