# mup_orion_demo

Demo of [mup](https://www.github.com/microsoft/mup) with [Orion](https://www.github.com/epistimio/orion).

[mup](https://www.github.com/microsoft/mup) is a brilliant research project, that makes it possible to tune the hyper-parameter of large-scale machine learning models by transferring the parameters found when tuning a smaller version of the same model.

This can dramatically reduce the amount of compute required to optimize very large models, potentially saving millions of dollars in training cost!

This is a demo of how to use mup to train a transformer from [mutransformers](https://www.github.com/microsoft/mutransformers) with Orion.

## Repo layout

The repo contains two examples: One using HuggingFace's [Accelerate](https://huggingface.co/docs/accelerate/main/en/index) library, and another using the [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) API.

Here is what each file contains:

```console
mup_demo
├── manual_example             # Folder containing a 'manual' example using HF's Accelerate API
│   ├── data.py                  # Data loading and preprocessing
│   ├── train.py                 # Manual training script                 (runnable from the CLI)
│   └── tune.py                  # Tuning script with Orion's Python API  (runnable from the CLI)
├── trainer_example            # Folder containing a 'trainer' example using HF's Trainer API
│   ├── mup_trainer_plugin.py    # "Plugin" for the HF Trainer API for MuP
│   ├── orion_trainer_plugin.py  # "Plugin" for the HF Trainer API for Orion
│   ├── train.py                 # Training script with HF's Trainer API  (runnable from the CLI)
│   ├── tune_orion.py            # Tuning script using Orion's Python API (runnable from the CLI)
│   └── tune_trainer.py          # Tuning script using HF's Trainer API   (runnable from the CLI)
├── model.py                   # Model creation functions common to both examples
└── utils.py                   # Utility functions common to both examples
```

## Example commands:

### Train a MuP Transformer using HF's Trainer API:

```console
accelerate launch mup_demo/trainer_example/train.py \
    --model_name_or_path gpt2 --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 --ddp_find_unused_parameters=False \
    --do_train --do_eval --output_dir test_run
```

### Running a sweep with Orion's command-line API:

```console
orion hunt -n test \
    accelerate launch mup_demo/trainer_example/train.py \
        --model_name_or_path gpt2 --dataset_name wikitext \
        --dataset_config_name wikitext-2-raw-v1 \
        --per_device_train_batch_size~"uniform(1,4,discrete=True)" \
        --per_device_eval_batch_size 8 --ddp_find_unused_parameters=False \
        --do_train --do_eval --output_dir=runs/{exp.name}/{trial.id} \
        --max_train_samples=100
```

### Running a sweep with Orion's Python API:

```console
accelerate launch mup_demo/trainer_example/tune_orion.py \
    --model_name_or_path gpt2 --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 --ddp_find_unused_parameters=False \
    --do_train --do_eval --output_dir=runs \
    --max_train_samples=100
```

### Running a sweep with the HF Trainer API:

```console
accelerate launch mup_demo/trainer_example/tune_trainer.py \
    --model_name_or_path gpt2 --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 --ddp_find_unused_parameters=False \
    --do_train --do_eval --output_dir=runs \
    --max_train_samples=100
```

### Train a MuP Transformer for a few epochs on a simple task using HF's Accelerate API:

```console
accelerate launch mup_demo/manual_example/train.py
```

### Tune a MuP Transformer on a simple task using HF's Accelerate API:

```console
accelerate launch mup_demo/manual_example/tune.py
```
