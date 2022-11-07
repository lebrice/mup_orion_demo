#!/bin/bash
#SBATCH --job-name=mup_sweep
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --cpus-per-task=4
#SBATCH --array=0-10%4
#SBATCH --output=/network/scratch/n/normandf/mup/logs/slurm-%A_%a.out
#SBATCH --error=/network/scratch/n/normandf/mup/logs/slurm-%A_%a.err

module load miniconda/3
conda activate $SCRATCH/conda/mup

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export OMP_NUM_THREADS=12

EXP_NAME="gpt2_256_4"

# Optional: Set some wandb-related environment variables.
# export WANDB_LOG_MODEL=1
# export WANDB_WATCH=all
export WANDB_PROJECT=mup_demo
export WANDB_TAGS=$EXP_NAME

orion hunt -n $EXP_NAME --config sweep_config.yaml \
    --exp-max-broken=999 --exp-max-trials=1000 --working-dir runs/$EXP_NAME \
    accelerate launch \
    --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
    --num_machines $SLURM_NNODES --num_processes $WORLD_SIZE \
    mup_demo/train.py \
    --output_dir {exp.working_dir}/{trial.id} \
    --run_name {exp.name}-{trial.id} \
    --per_device_train_batch_size=16 --auto_find_batch_size=False \
    --learning_rate~"loguniform(1e-7,1e-1)" \
    --n_embd=256 --n_head=8 --n_layer=4 \
    --num_train_epochs=3
