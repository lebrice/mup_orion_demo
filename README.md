# mup_orion_demo

Demo of [mup](https://www.github.com/microsoft/mup) with [Orion](https://www.github.com/epistimio/orion).

[mup](https://www.github.com/microsoft/mup) is a brilliant research project, that makes it possible to tune the hyper-parameter of large-scale machine learning models by transferring the parameters found when tuning a smaller version of the same model.

This can dramatically reduce the amount of compute required to optimize very large models, potentially saving millions of dollars in training cost!

This is a demo of how to use mup to train a HuggingFace transformer from [mutransformers](https://www.github.com/microsoft/mutransformers) with Orion.

## Repo layout

The repo contains examples for two different APIs:

- One using HuggingFace's [Accelerate](https://huggingface.co/docs/accelerate/main/en/index) library;
- The second using the [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) API.

Here is what each file contains:

```console
mup_demo
├── model.py              # Model creation functions
└── utils.py              # Utility functions common to both examples
├── train.py              # Training script with HF's Trainer API  (runnable from the CLI)
├── train_big_model.py    # Fetches the best trial from the sweep, trains scaled-up model.
```

## Example commands:

### Train a MuP Transformer using HF's Trainer API:

```console
accelerate launch mup_demo/trainer_example/train.py \
    --per_device_train_batch_size 4 --output_dir runs/test_run
```

### Running a sweep with Orion's command-line API:

```console
orion hunt -n mup_debug \
    accelerate launch mup_demo/trainer_example/train.py \
        --per_device_train_batch_size~"uniform(1,4,discrete=True)" \
        --output_dir=runs/orion_cli/{exp.name}/{trial.id} \
        --max_train_samples=100
```

```console
orion hunt -n mup_debug --exp-max-trials=5 --working-dir runs \
    WANDB_LOG_MODEL=1 WANDB_WATCH=all WANDB_PROJECT=mup_demo accelerate launch mup_demo/train.py \
    --output_dir runs/{exp.name}/{trial.id} --run_name {trial.id} \
    --learning_rate~"loguniform(1e-7,1e-4)" \
    --config_name_or_path=small_gpt2_config.json



```

### Running a sweep with Orion's Python API:

```console
accelerate launch mup_demo/trainer_example/tune_orion.py \
    --per_device_train_batch_size 4 \
    --output_dir=runs/orion_python --max_train_samples=100
```

### Running a sweep with the HF Trainer API:

```console
accelerate launch mup_demo/trainer_example/tune_trainer.py \
    --per_device_train_batch_size 4 --output_dir=runs/trainer_plugin \
    --max_train_samples=100
```

## Other examples:

### Train a MuP Transformer for a few epochs on a simple task using HF's Accelerate API:

```console
accelerate launch mup_demo/manual_example/train.py
```

### Tune a MuP Transformer on a simple task using HF's Accelerate API:

```console
accelerate launch mup_demo/manual_example/tune.py
```
