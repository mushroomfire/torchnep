# Multi-GPU and multi-node

`train_nep_sharded` trains data-parallel: one process per GPU, gradients averaged across processes. Each process loads only its share of the structures, so the host and GPU memory of the data shrinks with the number of GPUs.

```python title="run_train.py"
from torchnep import train_nep_sharded

train_nep_sharded("nep.in", "train.xyz", output_dir="output")
```

It takes the same arguments as [`train_nep`](training.md#runtime-arguments) (without `device`) and writes the same output files.

!!! note "Batch size"
    `batch` in `nep.in` is the number of structures **per GPU**. The global batch is `batch` × the number of GPUs.

## One node

```bash
torchrun --standalone --nproc_per_node=4 run_train.py    # 4 GPUs
```

## Several nodes with SLURM

One `srun` task per node starts `torchrun`, which starts one process per GPU on that node:

```bash title="train.slurm"
#!/bin/bash
#SBATCH --nodes=2                  # nodes
#SBATCH --ntasks-per-node=1        # one torchrun per node
#SBATCH --gpus-per-node=4          # GPUs per node
#SBATCH --cpus-per-task=16         # CPU cores per node

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))

srun --nodes=$SLURM_NNODES --ntasks-per-node=1 bash -c "
  torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=\$SLURM_GPUS_ON_NODE \
    --node_rank=\$SLURM_NODEID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    run_train.py
"
```

Any launcher that sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR` and `MASTER_PORT` works as well.

## Collective timeout

A rank that stops answering a collective — a network hiccup, a sick node — leaves the others waiting until the
watchdog aborts the job, so the timeout decides how long a hung run keeps the whole allocation. It is 30 minutes,
which also covers the longest legitimate wait: rank 0 alone writes the end-of-training prediction files while the
other ranks already sit in the next collective (about 10 minutes for 13 million frames). `TORCHNEP_DIST_TIMEOUT_MIN`
sets another value, in minutes; it applies to sharded training, prediction and extrapolation alike.

## Restarting

A sharded run resumes from `checkpoint.pt` like a single-GPU run. Resubmitting the same job continues where it stopped, which also splits long trainings into several jobs that each fit a queue's time limit.

## Prediction on several GPUs

`predict_dataset_sharded` is the multi-GPU counterpart of `predict_dataset`, launched the same way; see [Prediction](prediction.md#multi-gpu-prediction).
