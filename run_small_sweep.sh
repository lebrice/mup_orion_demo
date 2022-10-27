#!/usr/bin/bash
export WANDB_LOG_MODEL=1
export WANDB_WATCH=all
export WANDB_PROJECT=mup_demo

orion hunt -n mup_debug --exp-max-trials=5 --working-dir runs \
    accelerate launch mup_demo/train.py \
    --output_dir {exp.working_dir}/{exp.name}/{trial.id} --run_name {exp.name}-{trial.id} \
    --logging_steps=10 \
    --load_best_model_at_end=True --metric_for_best_model=eval_loss --greater_is_better=False \
    --evaluation_strategy=steps \
    --learning_rate~"loguniform(1e-7,1e-4)" \
    --config_path=gpt2_config_small.json
