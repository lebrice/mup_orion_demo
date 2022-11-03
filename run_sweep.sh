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
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export OMP_NUM_THREADS=12

module load miniconda/3
conda activate $SCRATCH/conda/mup

EXP_NAME="gpt2_256"

# Optional: Set some wandb-related environment variables.
export WANDB_LOG_MODEL=1
export WANDB_WATCH=all
export WANDB_PROJECT=mup_demo
export WANDB_TAGS=$EXP_NAME

orion hunt -n $EXP_NAME --config sweep_config.yaml --working-dir runs/$EXP_NAME \
    accelerate launch mup_demo/train.py \
    --output_dir {exp.working_dir}/{trial.id} --overwrite_output_dir=True \
    --run_name {exp.name}-{trial.id} \
    --load_best_model_at_end=True --metric_for_best_model=eval_loss --greater_is_better=False \
    --evaluation_strategy=steps \
    --save_strategy=steps \
    --logging_steps=500 \
    --per_device_train_batch_size=32 --auto_find_batch_size=False \
    --readout_zero_init=True --query_zero_init=True \
    --dataset_name=wikitext --dataset_config_name=wikitext-2-raw-v1 \
    --learning_rate~"loguniform(1e-7,1e-2)" \
    --n_embd=256 --n_head=4 --n_layer=2 \
    --max_steps=5000 \
    --report_to=wandb
