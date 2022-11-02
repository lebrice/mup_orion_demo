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
├── utils.py              # Utility functions
├── train.py              # Training script with HF's Trainer API  (runnable from the CLI)
└── train_big_model.py    # Fetches the best trial from the sweep, trains scaled-up model.
```

## Running the example:

0. Make sure to run `accelerate config` first
```console
accelerate config
```

1. Tune a small version of a MuP-parametrized GPT2 model:

```console
./run_sweep.sh
```
