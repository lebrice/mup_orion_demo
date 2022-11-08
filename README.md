# mup_orion_demo

Demo of [mup](https://www.github.com/microsoft/mup) with [Orion](https://www.github.com/epistimio/orion).

[mup](https://www.github.com/microsoft/mup) is a brilliant research project, that makes it possible to tune the hyper-parameter of large-scale machine learning models by transferring the parameters found when tuning a smaller version of the same model.

This can dramatically reduce the amount of compute required to optimize very large models, potentially saving millions of dollars in training cost!

This is a demo of how to use mup to train a HuggingFace transformer from [mutransformers](https://www.github.com/microsoft/mutransformers) with Orion.

## Repo layout

```console
mup_demo
├── model.py              # MuP Model creation functions
├── utils.py              # Utility functions
├── mup_trainer_patch.py  # Patch for the Trainer from HF, to make it use optimizers from MuP.
├── train.py              # Training script with HF's Trainer API  (runnable from the CLI)
└── train_big_model.py    # Fetches the best trial from the sweep, trains scaled-up model.
```

## Running the example locally:

0. Install the dependencies.

```console
conda env create -n mup --file env.yaml
conda activate mup
pip install -e .
```

1. Run `accelerate config`, so you won't need to pass all the arguments to `accelerate launch` later.

```console
accelerate config
```

2. Launch a single training run, just to make sure everything works:

```console
accelerate launch mup_demo/train.py --n_embd=256 --n_head=4 --n_layer=2 --report_to=none
```

3. Tune a small version of a MuP-parametrized GPT2 model:

```console
./run_sweep.sh
```

4. Launch a single training run with the best parameters found by the sweep, overwriting what we
   want to change from that run:

```console
accelerate launch mup_demo/train_big_model.py --output_dir=runs/large --n_embd=1024
```
