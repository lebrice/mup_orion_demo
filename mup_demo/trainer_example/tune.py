""" TODO: Tuning portion. """

# Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text
# file or a dataset.

# Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
# https://huggingface.co/models?filter=text-generation

# Example command:
# ```
# accelerate launch mup_demo/trainer_example/tune.py \
#     --model_name_or_path gpt2 --dataset_name wikitext \
#     --dataset_config_name wikitext-2-raw-v1 --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 8 --ddp_find_unused_parameters=False \
#     --do_train --do_eval --output_dir test_run
# ```
from __future__ import annotations

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain

import datasets
import evaluate
import mutransformers
import simple_parsing
import transformers
from datasets import Dataset, DatasetDict
from datasets.load import load_dataset
from simple_parsing.helpers import flag
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
    AutoModelForCausalLM,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version
import mutransformers
from mup_demo.model import get_gpt2_model

# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.24.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# TODO: Could try to add the mup-variants in these lists here?

from .train import (
    parse_args,
    main,
    ModelArguments,
    DataTrainingArguments,
    CustomTrainer,
    setup_trainer,
)


def tune():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    model_args, data_args, training_args = parse_args()
    trainer = setup_trainer(
        model_args=model_args, data_args=data_args, training_args=training_args
    )
    best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize", backend="orion")
    print(best_run)


if __name__ == "__main__":
    tune()
