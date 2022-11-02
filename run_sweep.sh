EXP_NAME="gpt2_256"

# Optional: Set some wandb-related environment variables.
export WANDB_LOG_MODEL=1
export WANDB_WATCH=all
export WANDB_PROJECT=mup_demo
export WANDB_TAGS=$EXP_NAME

orion hunt -n $EXP_NAME --exp-max-trials=50 --working-dir runs/$EXP_NAME \
    accelerate launch mup_demo/train.py \
    --output_dir {exp.working_dir}/{trial.id} --overwrite_output_dir=True \
    --run_name {exp.name}-{trial.id} \
    --load_best_model_at_end=True --metric_for_best_model=eval_loss --greater_is_better=False \
    --evaluation_strategy=steps \
    --save_strategy=steps \
    --logging_steps=100 \
    --per_device_train_batch_size=32 --auto_find_batch_size=True \
    --readout_zero_init=True --query_zero_init=True \
    --dataset_name=wikitext --dataset_config_name=wikitext-2-raw-v1 \
    --learning_rate~"loguniform(1e-7,1e-2)" \
    --n_embd=256 --n_head=4 --n_layer=2 \
    --max_steps=5000 \
    --report_to wandb

# accelerate launch mup_demo/train.py \
#     --output_dir runs/long --run_name long \
#     --load_best_model_at_end=True \
#     --metric_for_best_model=eval_loss \
#     --greater_is_better=False --evaluation_strategy=steps \
#     --save_strategy=steps --logging_steps=500 \
#     --n_embd=200 --n_head=4 --n_layer=2 \
#     --num_train_epochs=100 \
#     --dataset_name c4 --dataset_config_name realnewslike
