"""Tune the hyper-parameters of a MuP Transformer using the Trainer API.

Example command:
```bash
accelerate launch mup_demo/trainer_example/tune.py \
    --model_name_or_path gpt2 --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 --per_device_eval_batch_size 8 \
    --ddp_find_unused_parameters=False --do_train --do_eval \
    --num_train_epochs=1 --max_train_samples=100 \
    --output_dir runs/tune_debug
```
"""
from __future__ import annotations

import logging

from mup_demo.trainer_example.orion_trainer_plugin import OrionTrainer
from mup_demo.trainer_example.train import _setup_logging, parse_args, setup_trainer

logger = logging.getLogger(__name__)

# TODO: Could try to add the mup-variants in these lists here?


def tune_using_trainer_api():
    """Tune the hyper-parameters using the `hyperparameter_search` API of the HuggingFace Trainer.

    This "works", but the output directory structure is weird (might be fixable though). Also, the
    code required to make this work is quite messy.
    """
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    model_args, data_args, training_args = parse_args()

    # Setup logging
    _setup_logging(training_args)

    trainer = setup_trainer(
        model_args=model_args, data_args=data_args, training_args=training_args
    )

    trainer: OrionTrainer
    best_run = trainer.hyperparameter_search(
        n_trials=10,
        direction="minimize",
        backend="orion",
        # NOTE: Passing `None` would use a default HPO space (not recommended).
        hp_space={
            "learning_rate": "loguniform(1e-6, 1e-4)",
            "num_train_epochs": "fidelity(1, 5)",
            "seed": "uniform(1, 100, discrete=True)",
        },
    )
    print(best_run)


if __name__ == "__main__":
    tune_using_trainer_api()
    # tune_using_orion()
