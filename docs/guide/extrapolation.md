# Extrapolation grade (experimental)

!!! warning "Experimental — only on the `feat/extrapolation-gamma` branch"
    This feature is not part of a release yet. Install the branch to try it:

    ```bash
    pip install "git+https://github.com/mushroomfire/torchnep.git@feat/extrapolation-gamma"
    ```

With **one** trained model, the extrapolation grade tells how far an atomic environment lies outside the model's training set — no committee of models is needed. Its use is active learning: out of many structures — MD frames from GPUMD, LAMMPS or anything that writes extended XYZ — choose the few worth computing with DFT and adding to the training set. The grade can also be computed by GPUMD during MD.

## Choosing structures for DFT

Two calls: build the model's active set from its training set, once per model, then choose from the candidate structures.

```python
from torchnep.extrapolation import build_active_set, select_structures

build_active_set("nep.txt", "train.xyz", "active_set.pt")      # once per trained model

res = select_structures("nep.txt", "active_set.pt", "md.xyz",
                        output_xyz="to_dft.xyz", max_frames=200)
```

`to_dft.xyz` receives the chosen frames, copied verbatim from `md.xyz`. Compute them with DFT, add them to the training set and retrain; then rebuild the active set for the new model before choosing from its MD runs. `res["index"]` holds the positions of the chosen frames in `md.xyz`, in order of choice, and `res["grade"]` the grade of every frame of `md.xyz`.

Output of the two calls for a Cr-Co-Ni model and its training set (3030 frames), choosing from a pool of 2175 structures of another dataset, on one GPU:

```text
  active set: 3030 frames, 257880 atoms, 3 elements, K = 80 x (63 + 2) = 5200, rcond 0.0001, tol 1.01, float32
  pass 1 (Gram matrices, 3030 frames): 7.4s
    Cr  rows     82288  rank   345 / 5200
    Co  rows     85112  rank   308 / 5200
    Ni  rows     90480  rank   289 / 5200
  subspaces and seed active sets: 4.6s
  pass 2 (MaxVol, 3030 frames): 1.0s
  check 1: 336 atoms above 1.01 (largest 1.7697) -> 153 swaps (0.5s)
  check 2: 6 atoms above 1.01 (largest 1.0969) -> 56 swaps (0.3s)
  check 3: 3 atoms above 1.01 (largest 1.0747) -> 16 swaps (0.3s)
  check 4: every training atom has gamma <= 1.01 (largest 1.0084; 0.2s)
  TOTAL: 17.4s -> active_set.pt
  select: 845 candidate frames (95106 atoms)
  select: 200 frames chosen in 6.4s
```

### How the choice works

A frame's **grade** is the largest grade of its atoms; above 1, the frame lies outside the training set (see [What the grade measures](#what-the-grade-measures)). The frames whose grade lies in (`grade_min`, `grade_max`] are the candidates — 845 of the 2175 above. They are visited from the highest grade down, and a frame is taken only if its grade, **recomputed after the frames taken before it were added to the training data**, still exceeds `grade_min`. Of a group of near-identical frames only the first is taken: once it is in, the others no longer extrapolate. The grades at the time of choice (`res["grade_at_choice"]`, here 20.9, 14.8, 12.4, 7.8, 10.9, …) therefore do not decrease monotonically. A frame with an element that has no atom in the training set has grade ∞ and is always taken.

| Argument | Default | Meaning |
|---|---|---|
| `max_frames` | all | Stop after this many frames. |
| `grade_min` | `tol` of the active set (1.01) | Frames at or below it are not candidates. |
| `grade_max` | none | Frames above it are not candidates either. |
| `output_xyz` | none | Write the chosen frames here. |
| `output_active_set` | none | Save the active set extended by the chosen frames. |
| `precision` | `"float32"` | Descriptors and first grades in float32; the choice itself runs in float64. |

With `max_frames`, the most extrapolating frames are taken first, so the default `grade_min` is usually fine. The frames with the largest grades lie farthest from anything the model was trained on; they can be unphysical, e.g. MD frames after the simulation went wrong and atoms ran into each other. Look at them before sending them to DFT, and exclude such frames with `grade_max`.

### Do I need `compute_gamma`?

Not for choosing: `select_structures` grades the frames itself. `compute_gamma` only grades, which is useful to look at the grades before or instead of choosing:

- how much an MD run extrapolates, e.g. the fraction of frames with grade > 1 — it should shrink from one round of active learning to the next;
- the distribution of the grades, to set `grade_min` or `grade_max`;
- per-atom grades (`per_atom=True`) and the atom with the largest grade in each frame: where a large structure extrapolates (a surface, a defect, one species), e.g. to cut out a smaller cell around it for DFT.

```python
from torchnep.extrapolation import compute_gamma

g = compute_gamma("nep.txt", "active_set.pt", "md.xyz", output_file="grades.npz")
print((g["grade"] > 1).mean())          # fraction of extrapolating frames
```

For the pool above it prints 0.399, after the log line `grades: 2175 frames in 1.2s; frames with grade > 1.01: 845 (gamma > 1.01: 805), median grade 0.887, max 20.9`. See [Grade structures](#grade-structures) for the returned arrays.

## Grades during MD with GPUMD

GPUMD computes the same grade γ during MD with its [`compute_extrapolation`](https://gpumd.org/dev/gpumd/input_parameters/compute_extrapolation.html) keyword. Write the active set in GPUMD's format when building it, or later from the saved one:

```python
build_active_set("nep.txt", "train.xyz", "active_set.pt", asi_file="active_set.asi")

# or, for an active set built before
from torchnep.extrapolation import ActiveSet
ActiveSet.load("active_set.pt", "nep.txt").save_gpumd("active_set.asi")
```

and add to GPUMD's `run.in`, for example

```text
compute_extrapolation asi_file active_set.asi gamma_low 2 gamma_high 50 check_interval 100 dump_interval 100
```

Every `check_interval` steps GPUMD grades all atoms, writes the frames with γ ≥ `gamma_low` to `extrapolation_dump.xyz`, and stops the run when γ exceeds `gamma_high`.

The difference from TorchNEP's own choice: GPUMD grades by γ only, while `select_structures` ranks by the larger of γ and γ_res (see [What the grade measures](#what-the-grade-measures)).

## What the grade measures

The method is the D-optimality (MaxVol) active learning developed for moment tensor potentials [[1–3]](#references) and used for the atomic cluster expansion [[4]](#references), with the MaxVol algorithm of [[5]](#references); see these papers for the theory. In short:

Every atom is represented by the gradient of its NEP energy with respect to its element's network weights, `b = dE_i / d(w0, b0, w1)` (K = neurons × (descriptor size + 2) numbers) — the model linearised in its parameters. From the training set, TorchNEP keeps for every element the **active set**: the training atoms whose rows span the largest volume. A new atom is written in the basis of the active rows, `b = c A`, and

```text
gamma = max_j |c_j|
```

- **γ ≤ 1**: the environment lies inside the region spanned by the training set (interpolation). Every training atom has γ ≤ `tol` (1.01) by construction.
- **γ > 1**: it lies outside; putting it into the active set would multiply the spanned volume by γ.

Two choices are specific to this implementation. The rows are numerically far from full rank — their singular values decay smoothly over many orders of magnitude — so γ is computed in the leading subspace of each element, the directions whose singular value is above `rcond` × the largest; the rows are projected onto it and whitened, which leaves γ unchanged (γ does not depend on the basis) and keeps the matrices well conditioned. The part of `b` outside the subspace is measured by **γ_res**: its norm divided by the largest one met in the training set (> 1: more weight outside the training subspace than any training atom). An atom's **grade** is the larger of γ and γ_res — both scale with how far the atom lies outside, and both are ≤ 1 for every training atom. The choice of structures (above) is a greedy, re-graded variant of choosing by MaxVol, in which a chosen frame's atoms enter the active set and their directions outside the subspace are added to it.

## Build the active set

`build_active_set(model_file, xyz_file, output_file)` streams the training set:

1. on a random sample of frames, the Gram matrix of every element's rows — its leading eigenvectors give the subspace — and a random sample of rows that seeds the active set;
2. on all frames, MaxVol swaps of every atom whose grade exceeds `tol`;
3. check passes over all frames: the atoms still above `tol` (a row passed early can exceed it after later swaps) are swapped in, until a pass finds none — after 3 to 9 passes on the training sets we tried. The log reports the result and the largest training grade, e.g. `check 4: every training atom has gamma <= 1.01`; if `max_passes` runs out first, a last check reports how many atoms remain above `tol` (typically a handful, with grades just above it).

`rank` in the log is the dimension of the element's subspace, i.e. its number of active rows, out of K. The descriptors are kept in host memory between passes when they fit in 25 % of the free memory, so the check passes cost only the projection (`TORCHNEP_GAMMA_CACHE_GB` sets another budget, `0` disables it).

| Argument | Default | Meaning |
|---|---|---|
| `rcond` | `1e-4` | Subspace: directions whose singular value is above `rcond` × the largest. Smaller keeps more directions (a larger active set). |
| `tol` | `1.01` | MaxVol threshold; every training atom ends with γ ≤ `tol`. |
| `sample_frames` | `50000` | Frames used for the Gram matrices and the seed (all frames if fewer). |
| `init_rows` | `4` | Seed: LU pivoting over `init_rows` × r sampled rows. |
| `max_passes` | `10` | Check passes (with swaps) at most. |
| `precision` | `"float32"` | Descriptors and screening in float32 (see [float32 or float64](#float32-or-float64)); the Gram matrices, subspaces and MaxVol always run in float64. |
| `asi_file` | none | Also write the active set for GPUMD. |
| `chunk_atoms` | 200 000 | Atoms read per chunk (host memory). |
| `seed` | `0` | Random sample of frames and rows. |

The active set belongs to one model: it stores a fingerprint of the model's parameters, and loading it with another model raises an error. Rebuild it after every retraining.

## Grade structures

`compute_gamma(model_file, active_set, xyz_file)` returns a dict of NumPy arrays and, with `output_file`, saves it as `.npz`:

| Key | Shape | Meaning |
|---|---|---|
| `grade` | frames | largest grade of the frame's atoms, max(γ, γ_res) — what `select_structures` ranks by |
| `gamma` | frames | largest γ of the frame's atoms (what GPUMD computes) |
| `gamma_res` | frames | largest γ_res of the frame's atoms |
| `atom` | frames | index (in the frame) of the atom with the largest grade (of atoms with equal grades, e.g. symmetric sites, any may come out) |
| `natoms` | frames | atoms per frame |
| `gamma_atoms`, `gamma_res_atoms` | atoms | every atom, in file order (`per_atom=True`) |

Atoms of an element that has no atom in the training set get γ = ∞.

## float32 or float64

All functions take `precision="float32"` (default) or `"float64"`. In float32 the descriptors and the grades of the streamed frames are computed in float32; every decision that depends on the threshold is checked in float64, and the Gram matrices, subspaces and MaxVol always run in float64. Measured differences:

- **Grades** of the same structures with the same active set: γ within 7 × 10⁻⁵ (relative; median 1.4 × 10⁻⁶), γ_res within 2 × 10⁻³, over 105 464 frames.
- **Active sets** built in float32 and float64 differ, as any two MaxVol runs over slightly different numbers do, but both keep every training atom within `tol` (largest training γ 1.0094 and 1.0099, whichever precision grades them); the subspace dimensions were identical.
- **Choices**: with the same active set, float32 and float64 chose the same 200 frames of the Cr-Co-Ni pool; with the two active sets, 178 of the 200 chosen frames, and the first ten, were the same.
- **Speed**: float32 grading was 1.3 × faster on a GPU with fast float64 (V100); on GPUs with slow float64 (most consumer cards) the gain is several-fold. Building took the same time in both, since the float64 Gram matrices dominate.

## Several GPUs

`build_active_set_sharded`, `compute_gamma_sharded` and `select_structures_sharded` take the same arguments (without `device`) and run one process per GPU, launched like [multi-GPU training](distributed.md):

```python title="active.py"
from torchnep.extrapolation import build_active_set_sharded, select_structures_sharded

build_active_set_sharded("nep.txt", "train.xyz", "active_set.pt")
select_structures_sharded("nep.txt", "active_set.pt", "md.xyz",
                          output_xyz="to_dft.xyz", max_frames=200)
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

The file is read by worker processes one chunk ahead while the GPU computes the neighbor lists, the descriptors and the grades of the current chunk. Measured with a 16-element model (K = 80 × (35 + 2) = 2960) and its training set of 105 464 frames (6.9 million atoms), in float64:

| | 1 GPU | 8 GPUs |
|---|---|---|
| `build_active_set` (converged) | 212 s | 62 s |
| `compute_gamma`, all 105 464 frames | 29 – 36 s | 9 – 20 s |

The upper figures of `compute_gamma` include the start-up of a fresh run. On one GPU the build spends 13 s in pass 1 (50 000 sampled frames), 45 s on the subspaces and seeds — mostly the eigendecompositions of the K × K Gram matrices, about 2.5 s per element, which the sharded build divides among the GPUs — 83 s in pass 2 and 6 – 11 s per check pass (9 passes). On one V100, building took 244 s and grading all frames 32 s in float32. Choosing 200 of the 2175 frames of the Cr-Co-Ni example takes about 6 s.

## Limitations

- **Experimental.** On one test (a Cr-Co-Ni model and a pool of structures from another dataset), choosing frames by γ reduced the error on held-out structures as much as choosing them by a committee of four models, and much more than choosing them at random. It has not been used in production active learning yet.
- **Memory while building:** the Gram matrices take K² × 8 bytes per element on the GPU (70 MB for K = 2960). With the sharded build, each element's matrix is summed on one rank.
- **γ_res** in float64 is computed as `|b|² − |b V|²`, which costs digits: it is accurate to about 1e-8 relative, plenty for a grade.
- The active set depends on the model; rebuild it after retraining.

## References

1. E. V. Podryabinkin and A. V. Shapeev, *Active learning of linearly parametrized interatomic potentials*, Comput. Mater. Sci. **140**, 171–180 (2017). [doi:10.1016/j.commatsci.2017.08.031](https://doi.org/10.1016/j.commatsci.2017.08.031) — the extrapolation grade and the MaxVol active set.
2. K. Gubaev, E. V. Podryabinkin, G. L. W. Hart and A. V. Shapeev, *Accelerating high-throughput searches for new alloys with active learning of interatomic potentials*, Comput. Mater. Sci. **156**, 148–156 (2019). [doi:10.1016/j.commatsci.2018.09.031](https://doi.org/10.1016/j.commatsci.2018.09.031) — active learning of moment tensor potentials, whose energy is nonlinear in part of the parameters.
3. E. Podryabinkin, K. Garifullin, A. Shapeev and I. Novikov, *MLIP-3: Active learning on atomic environments with moment tensor potentials*, J. Chem. Phys. **159**, 084112 (2023). [doi:10.1063/5.0155887](https://doi.org/10.1063/5.0155887) — grades of single atomic environments.
4. Y. Lysogorskiy, A. Bochkarev, M. Mrovec and R. Drautz, *Active learning strategies for atomic cluster expansion models*, Phys. Rev. Materials **7**, 043801 (2023). [doi:10.1103/PhysRevMaterials.7.043801](https://doi.org/10.1103/PhysRevMaterials.7.043801) — per-element active sets for a many-element potential.
5. S. A. Goreinov, I. V. Oseledets, D. V. Savostyanov, E. E. Tyrtyshnikov and N. L. Zamarashkin, *How to find a good submatrix*, in *Matrix Methods: Theory, Algorithms and Applications*, World Scientific (2010), pp. 247–256. [doi:10.1142/9789812836021_0015](https://doi.org/10.1142/9789812836021_0015) — the MaxVol algorithm.
