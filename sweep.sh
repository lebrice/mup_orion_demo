#!/bin/bash
#SBATCH --job-name=mup_sweep
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-20
#SBATCH --output=/network/scratch/n/normandf/mup/logs/slurm-%A_%a.out
#SBATCH --error=/network/scratch/n/normandf/mup/logs/slurm-%A_%a.err

module load miniconda/3
conda activate $SCRATCH/conda/mup

export EXP_NAME=${EXP_NAME:-"gpt2_repro"}

echo "Starting sweep with name $EXP_NAME"

orion hunt -n $EXP_NAME --config sweep_config.yaml \
    --exp-max-broken=999 --exp-max-trials=1000 --working-dir runs/$EXP_NAME \
    ./train.sh \
    --output_dir {exp.working_dir}/{trial.id} \
    --run_name {exp.name}-{trial.id} \
    --per_device_train_batch_size=20 --auto_find_batch_size=False \
    --learning_rate~"loguniform(1e-7,1e-1,default_value=5e-05)" \
    --n_embd~"choices(128,256,512,1024,2048,4096)" --n_head=2 --n_layer=2 \
    --lr_scheduler_type="constant" \
    --max_steps=10_000 --block_size=256 --save_steps=2500
