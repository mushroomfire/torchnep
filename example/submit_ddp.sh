#!/bin/bash
#SBATCH --job-name=torchnep-ddp
#SBATCH --account=<your_account>
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --nodes=2                # number of nodes
#SBATCH --ntasks-per-node=1      # one torchrun per node
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4             # GPUs per node
#SBATCH --output=slurm-%j.out

# ---- Activate environment ----
module load cuda
source activate myenv   # or: conda activate myenv

# ---- Network setup ----
# Use the first node as master
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

echo "MASTER_ADDR=$MASTER_ADDR"
echo "NODES=$SLURM_NNODES, GPUS_PER_NODE=$SLURM_GPUS_ON_NODE"

# ---- Launch ----
# srun launches one torchrun per node; torchrun spawns per-GPU workers.
srun torchrun \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    run_train_ddp.py
