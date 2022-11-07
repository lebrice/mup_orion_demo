#!/bin/bash
#SBATCH --job-name=mup_large
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --cpus-per-task=4
#SBATCH --output=/network/scratch/n/normandf/mup/logs/slurm-%j.out
#SBATCH --error=/network/scratch/n/normandf/mup/logs/slurm-%j.err

# module load miniconda/3
# conda activate $SCRATCH/conda/mup
EXP_NAME="gpt2_256"

# Optional: Set some wandb-related environment variables.
export WANDB_LOG_MODEL=1
export WANDB_WATCH=all
export WANDB_PROJECT=mup_demo
export WANDB_TAGS=$EXP_NAME

# accelerate launch mup_demo/train_big_model.py \
#     --output_dir runs/gpt2_1024 --overwrite_output_dir=True \
#     --run_name gpt2_1024 \
#     --load_best_model_at_end=True --metric_for_best_model=eval_loss --greater_is_better=False \
#     --evaluation_strategy=steps \
#     --save_strategy=steps \
#     --logging_steps=500 \
#     --per_device_train_batch_size=32 --auto_find_batch_size=True \
#     --readout_zero_init=True --query_zero_init=True \
#     --dataset_name=wikitext --dataset_config_name=wikitext-2-raw-v1 \
#     --n_embd=1024 --n_head=4 --n_layer=2 \
#     --max_steps=5000 \
#     --report_to=wandb

accelerate launch mup_demo/train.py \
    --output_dir runs/gpt2_1024_5000 --overwrite_output_dir=True \
    --run_name gpt2_1024_5000 \
    --load_best_model_at_end=True --metric_for_best_model=eval_loss --greater_is_better=False \
    --evaluation_strategy=steps \
    --save_strategy=steps \
    --logging_steps=500 \
    --per_device_train_batch_size=32 --auto_find_batch_size=True \
    --readout_zero_init=True --query_zero_init=True \
    --dataset_name=wikitext --dataset_config_name=wikitext-2-raw-v1 \
    --n_embd=1024 --n_head=4 --n_layer=2 \
    --max_steps=5000 \
    --learning_rate=0.0003504 \
    --report_to=wandb
