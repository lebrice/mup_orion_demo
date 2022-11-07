#!/bin/bash
#SBATCH --job-name=mup_train
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --cpus-per-task=4
#SBATCH --output=/network/scratch/n/normandf/mup/logs/slurm-%j.out
#SBATCH --error=/network/scratch/n/normandf/mup/logs/slurm-%j.err

module load miniconda/3
conda activate $SCRATCH/conda/mup
EXP_NAME="gpt2_256"

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# Optional: Set some wandb-related environment variables.
# export WANDB_LOG_MODEL=1
# export WANDB_WATCH=all
export WANDB_PROJECT=mup_demo
export WANDB_TAGS=$EXP_NAME

accelerate launch \
    --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
    --num_machines $SLURM_NNODES --num_processes $WORLD_SIZE \
    mup_demo/train.py "$@"
