#!/usr/bin/env python
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trains a GPT2 model with MuP parametrization on a causal language modeling task.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation


Example command:
```
accelerate launch mup_demo/trainer.py \
    --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 --per_device_eval_batch_size 8 \
    --output_dir test_run
```
"""
from __future__ import annotations

import dataclasses
import functools
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Literal

import datasets
import evaluate
import mutransformers
import simple_parsing
import transformers
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from simple_parsing.helpers import flag
from simple_parsing.helpers.serialization.serializable import save_yaml
from torch import Tensor
from torch.utils.data import Dataset
from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoTokenizer,
    EvalPrediction,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers import Trainer as _Trainer
from transformers import TrainingArguments as _TrainingArguments
from transformers import default_data_collator, is_torch_tpu_available, set_seed
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.training_args import ExplicitEnum
from transformers.utils.logging import get_logger

import wandb
from mup_demo.model import get_gpt2_model
from mup_demo.mup_trainer_plugin import patch_trainer_for_mup
from mup_demo.utils import is_main_process

logger = get_logger(__name__)

# Apply the 'patch' to the Trainer class so it uses the mup optimizers.
patch_trainer_for_mup()

# TODO: Could try to add the mup-variants in these lists here. There's an issue where the config
# files get parsed into the GPT2Config from HF rather than the GPT2Config from mutransformers.

MODEL_CONFIG_CLASSES: list[type[PretrainedConfig]] = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())  # type: ignore
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
# NOTE: Not sure this does anything..
MODEL_FOR_CAUSAL_LM_MAPPING[transformers.GPT2Config] = mutransformers.GPT2LMHeadModel
# assert False, MODEL_CONFIG_CLASSES
# assert False, MODEL_FOR_CAUSAL_LM_MAPPING[transformers.GPT2Config]


# TODO: Replace these ModelArguments with a dataclass version of the GPT2Config?


@dataclass
class GPT2ConfigArgs:
    """Dataclass containing the fields of the GPT2Config class, so we can change them from the CLI
    more easily than with the 'config_overrides' argument from the example.

    NOTE: Unsure why HuggingFace didn't opt to use a dataclass for the GPT2Config class. It would
    have made this unnecessary.
    """

    vocab_size: int = 50257
    """Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling [`GPT2Model`] or [`TFGPT2Model`].
    """

    n_positions: int = 1024
    """The maximum sequence length that this model might ever be used with. Typically set this to something large
    just in case (e.g., 512 or 1024 or 2048).
    """

    n_embd: int = 768
    """ Dimensionality of the embeddings and hidden states. """

    n_layer: int = 12
    """ Number of hidden layers in the Transformer encoder. """

    n_head: int = 12
    """ Number of attention heads for each attention layer in the Transformer encoder."""

    n_inner: int | None = None
    """Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd """

    activation_function: Literal["relu", "silu", "gelu", "tanh", "gelu_new"] = "gelu"
    """ Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
    """

    resid_pdrop: float = 0.1
    """The dropout probability for all fully connected layers in the embeddings, encoder, and
    pooler.
    """

    embd_pdrop: float = 0.1
    """The dropout ratio for the embeddings. """

    attn_pdrop: float = 0.1
    """ The dropout ratio for the attention. """

    layer_norm_epsilon: float = 1e-5
    """ The epsilon to use in the layer normalization layers. """

    initializer_range: float = 0.02
    """The standard deviation of the truncated_normal_initializer for initializing all weight
    matrices.
    """

    # NOTE: Omitting the summary_* fields, since they're not used in the model. They're only used
    # when doing a sentence summarization task.

    scale_attn_weights: bool = True
    """Scale attention weights by dividing by sqrt(hidden_size)."""

    use_cache: bool = True
    """ Whether or not the model should return the last key/values attentions (not used by all
    models).
    """

    scale_attn_by_inverse_layer_idx: bool = False
    """Whether to additionally scale attention weights by `1 / layer_idx + 1`."""

    reorder_and_upcast_attn: bool = False
    """Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
    dot-product/softmax to float() when training with mixed precision.
    """

    attn_mult: float | None = None
    """ "Attention key-projection weight multiplier alpha_{attn}" from the mup paper.
    If set to None, take the value of `(self.hidden_size / self.num_attention_heads)**0.5`.
    """

    readout_zero_init: bool = False
    query_zero_init: bool = False

    def to_config(self) -> mutransformers.GPT2Config:
        """Convert this dataclass to a GPT2Config instance."""
        config_kwargs = asdict(self)
        config_kwargs.pop("readout_zero_init")  # Not a valid argument for GPT2Config
        config_kwargs.pop("query_zero_init")  # Not a valid argument for GPT2Config
        return mutransformers.GPT2Config(**asdict(self))

    def make_model(self) -> mutransformers.GPT2LMHeadModel:
        # NOTE: unused. Currently creating the config and then using it
        # NOTE: This currently doesn't allow for finetuning a model.
        config = self.to_config()
        logger.info(f"Creating a MuP-parametrized GPT2 model with config: {config}")
        return get_gpt2_model(
            config,
            model_type=mutransformers.GPT2LMHeadModel,
            readout_zero_init=self.readout_zero_init,
            query_zero_init=self.query_zero_init,
        )


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train
    from scratch."""

    # model_name_or_path: str | None = None
    # """The model checkpoint for weights initialization.Don't set if you want to train a model from
    # scratch.
    # """

    model: GPT2ConfigArgs = field(default_factory=GPT2ConfigArgs)
    """Configuration options for the model. This is used to create the GPT2Config object, which is
    used to create the model.
    """

    # model_type: str | None = field(
    #     default=None,
    #     metadata={
    #         "help": "If training from scratch, pass a model type from the list: "
    #         + ", ".join(MODEL_TYPES)
    #     },
    # )

    # config_overrides: str | None = None
    # """Override some existing default config settings when a model is trained from scratch.
    # Example: "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
    # """

    # NOTE: Added this here.
    # config_path: Path | None = None
    # """ Pretrained config name or path if not the same as model_name """

    # config_name: str | None = None
    # """ Pretrained config name or path if not the same as model_name """

    tokenizer_name: str | None = None
    """ Pretrained tokenizer name or path if not the same as model_name """

    cache_dir: str | None = None
    """Where do you want to store the pretrained models downloaded from huggingface.co"""

    use_fast_tokenizer: bool = flag(True)
    """Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."""

    model_revision: str = "main"
    """ The specific model version to use (can be a branch name, tag name or commit id)."""

    use_auth_token: bool = False
    """Will use the token generated when running `huggingface-cli login` (necessary to use this
    script with private models).
    """

    def __post_init__(self):
        pass
        # if self.config_overrides is not None and (
        #     self.config_name is not None or self.model_name_or_path is not None
        # ):
        #     raise ValueError(
        #         "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
        #     )


@dataclass
class DataTrainingArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""

    dataset_name: str | None = None
    """The name of the dataset to use (via the datasets library)."""

    dataset_config_name: str | None = None
    """The configuration name of the dataset to use (via the datasets library)."""

    train_file: str | None = None
    """The input training data file (a text file)."""

    validation_file: str | None = None
    """An optional input evaluation data file to evaluate the perplexity on (a text file)."""

    max_train_samples: int | None = None
    """For debugging purposes or quicker training, truncate the number of training examples to
    this value if set."""

    max_eval_samples: int | None = None
    """For debugging purposes or quicker training, truncate the number of evaluation examples to
    this value if set.
    """

    block_size: int | None = None
    """Optional input sequence length after tokenization.
    The training dataset will be truncated in block of this size for training.
    Default to the model max input length for single sentence inputs (take into account special
    tokens).
    """

    overwrite_cache: bool = False
    """Overwrite the cached training and evaluation sets"""

    validation_split_percentage: int | None = 5
    """The percentage of the train set used as validation set in case there's no validation split.
    """

    preprocessing_num_workers: int | None = None
    """The number of processes to use for the preprocessing."""

    keep_linebreaks: bool = True
    """Whether to keep line breaks when using TXT files or not."""

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                _raise_if_bad_extension(self.train_file, "train_file")
            if self.validation_file is not None:
                _raise_if_bad_extension(self.validation_file, "validation_file")


@dataclass
class TrainingArguments(_TrainingArguments):
    """TrainingArguments is the subset of the arguments we use in our example scripts **which
    relate to the training loop itself**."""

    def __post_init__(self):
        # NOTE: Little temporary patch so these fields (which have a `Enum | str` annotation) are
        # parsed properly.
        # BUG: Fix a bug that occurs with ExplicitEnum, where it parses it by name into
        # a 'IntervalStrategy.NO' string, rather than into the 'IntervalStrategy.NO' enum value!
        def _to_str(v: ExplicitEnum | str) -> str:
            if "." in v:
                return v.split(".")[-1].lower()
            return v if not isinstance(v, ExplicitEnum) else v.value

        self.evaluation_strategy = _to_str(self.evaluation_strategy)  # "no"
        self.logging_strategy = _to_str(self.logging_strategy)  # "steps"
        self.save_strategy = _to_str(self.save_strategy)  # "steps"
        self.hub_strategy = _to_str(self.hub_strategy)  # "every_save"
        self.lr_scheduler_type = _to_str(self.lr_scheduler_type)  # "linear"
        self.optim = _to_str(self.optim)  # "adamw_hf"

        super().__post_init__()


# Apply a small patch to the Trainer class, to allow the auto_find_batch_size to work properly.
class Trainer(_Trainer):
    def _inner_training_loop(
        self,
        batch_size: int,
        args: TrainingArguments,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
    ):
        # NOTE: Fix a bug here in the base class:
        args.per_device_train_batch_size = batch_size
        return super()._inner_training_loop(
            batch_size=batch_size,
            args=args,
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
        )


def _raise_if_bad_extension(file_path: str, attr_name: str):
    extension = file_path.split(".")[-1]
    if extension not in ["csv", "json", "txt"]:
        raise ValueError(f"`{attr_name}` should be a csv, json or txt file.")


def parse_args(
    default_model_args: ModelArguments | None = None,
    default_data_args: DataTrainingArguments | None = None,
    default_training_args: TrainingArguments | None = None,
) -> tuple[ModelArguments, DataTrainingArguments, TrainingArguments]:
    """Parse the model, data, and training arguments from the command-line.

    Uses [SimpleParsing](https://www.github.com/lebrice/SimpleParsing), an alternative to the
    HFArgumentParser, which also uses dataclasses to create argparse arguments.
    """
    parser = simple_parsing.ArgumentParser(description=__doc__, add_config_path_arg=True)
    parser.add_arguments(ModelArguments, dest="model", default=default_model_args)
    parser.add_arguments(DataTrainingArguments, dest="data", default=default_data_args)
    parser.add_arguments(TrainingArguments, dest="training", default=default_training_args)

    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments

    # if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # # If we pass only one argument to the script and it's the path to a json file,
    # # let's parse it to get our arguments.
    # model_args, data_args, training_args = parser.parse_json_file(
    #     json_file=os.path.abspath(sys.argv[1])
    # )
    # else:
    args = parser.parse_args()
    # model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args = args.model
    data_args = args.data
    training_args = args.training

    return model_args, data_args, training_args


def parse_with_good_defaults() -> tuple[ModelArguments, DataTrainingArguments, TrainingArguments]:
    return parse_args(
        default_model_args=ModelArguments(
            model=GPT2ConfigArgs(),
            tokenizer_name="gpt2",
        ),
        default_data_args=DataTrainingArguments(
            dataset_name="wikitext",
            dataset_config_name="wikitext-2-raw-v1",
        ),
        default_training_args=TrainingArguments(
            output_dir="runs/debug",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=8,
            ddp_find_unused_parameters=False,
            do_train=True,
            do_eval=True,
            num_train_epochs=1,
            auto_find_batch_size=True,
            # output_dir="runs/tune_plugin_api",
            # evaluation_strategy="no",
            # logging_strategy="steps",
            # save_strategy="steps",
            # hub_strategy="every_save",
            # optim="adamw_hf",
        ),
    )


def main():
    # See all possible arguments in the TrainingArguments class, or by passing the --help flag to
    # this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    model_args, data_args, training_args = parse_with_good_defaults()

    # Setup logging
    _setup_logging(training_args)

    trainer = setup_trainer(
        model_args=model_args, data_args=data_args, training_args=training_args
    )

    if training_args.do_train:
        train(trainer=trainer, model_args=model_args, data_args=data_args)

    metrics = None
    if training_args.do_eval:
        metrics = evaluation_loop(trainer=trainer, model_args=model_args, data_args=data_args)

    # NOTE: Removed some Optional Post-run stuff: creating a model card, uploading the model to
    # the hub, etc.

    if training_args.do_eval:
        assert metrics is not None
        from orion.client import report_results

        objective = metrics["eval_loss"]
        report_results(
            [dict(name="objective", type="objective", value=objective)]
            + [dict(name=key, type="statistic", value=value) for key, value in metrics.items()]
        )

    return metrics


# Training
def train(trainer: Trainer, model_args: ModelArguments, data_args: DataTrainingArguments):
    train_dataset = trainer.train_dataset
    assert train_dataset is not None
    training_args = trainer.args

    checkpoint = None
    last_checkpoint = find_last_checkpoint(training_args)
    checkpoint = training_args.resume_from_checkpoint or last_checkpoint or None

    log_dir = Path(training_args.output_dir)
    log_dir.mkdir(exist_ok=True, parents=True)  # Make sure the log dir exists
    save_yaml(training_args, log_dir / "training_args.yaml")
    save_yaml(model_args, log_dir / "model_args.yaml")
    save_yaml(data_args, log_dir / "data_args.yaml")

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    max_train_samples = (
        data_args.max_train_samples
        if data_args.max_train_samples is not None
        else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    return metrics


# Evaluation
def evaluation_loop(
    trainer: Trainer, model_args: ModelArguments, data_args: DataTrainingArguments
):
    logger.info("*** Evaluate ***")
    eval_dataset = trainer.eval_dataset
    assert eval_dataset is not None

    metrics = trainer.evaluate()

    max_eval_samples = (
        data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    )
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    return metrics


def setup_trainer(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
) -> Trainer:
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_clm", model_args, data_args)

    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    # last_checkpoint = find_last_checkpoint(training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    raw_datasets = get_datasets(data_args=data_args, model_args=model_args)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    tokenizer = _get_tokenizer(model_args=model_args, tokenizer_kwargs=tokenizer_kwargs)

    # Preprocessing the datasets.

    train_dataset, eval_dataset = preprocess_datasets(
        raw_datasets=raw_datasets,
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
    )

    # logger.info("Non-default config values:\n" + yaml.dump(config.to_diff_dict()))
    model_init: Callable[[], PreTrainedModel] = functools.partial(
        make_model,
        tokenizer=tokenizer,
        model_args=model_args,
    )

    preprocess_logits_for_metrics: Callable[[Tensor, Tensor], Tensor] | None = None
    compute_metrics: Callable[[EvalPrediction], dict] | None = None
    if training_args.do_eval and not is_torch_tpu_available():

        def _preprocess_logits_for_metrics(logits: Tensor, labels: Tensor) -> Tensor:
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def _compute_metrics(eval_preds: EvalPrediction) -> dict:
            preds, labels = eval_preds
            # TODO: Switch to this syntax if it works.
            # preds = eval_preds.predictions
            # labels = eval_preds.label_ids

            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            metrics = metric.compute(predictions=preds, references=labels)
            assert isinstance(metrics, dict)
            return metrics

        preprocess_logits_for_metrics = _preprocess_logits_for_metrics
        compute_metrics = _compute_metrics

    assert train_dataset is not None

    # note: This isn't working, it's causing it to re-init.
    if is_main_process():
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "mup_debug"),
            name=training_args.run_name,
            config={
                "model": dataclasses.asdict(model_args),
                "data": dataclasses.asdict(data_args),
                "training_args": dataclasses.asdict(training_args),
            },
            # tags=
        )
    training_args.report_to = []

    # Initialize our Trainer
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        # NOTE: next arg has wrong annotation (should be marked as optional)
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,  # type: ignore
    )
    # TODO: Potentially initialize wandb manually, to get more fine-grained control over what is
    # logged.
    from transformers.integrations import WandbCallback

    if is_main_process():
        assert wandb.run is not None
        trainer.add_callback(WandbCallback())
    return trainer


def find_last_checkpoint(training_args: _TrainingArguments) -> str | None:
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        # NOTE: This is being a little bit annoying.
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return last_checkpoint


def get_datasets(data_args: DataTrainingArguments, model_args: ModelArguments) -> DatasetDict:
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        assert isinstance(raw_datasets, DatasetDict)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            **dataset_args,
        )
        assert isinstance(raw_datasets, DatasetDict)

        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    return raw_datasets


def _setup_logging(training_args: TrainingArguments):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )


def _get_tokenizer(
    model_args: ModelArguments, tokenizer_kwargs: dict[str, Any]
) -> PreTrainedTokenizerBase:
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    return tokenizer


def make_model(
    model_args: ModelArguments,
    tokenizer: PreTrainedTokenizerBase,
) -> PreTrainedModel:

    model: PreTrainedModel
    logger.info("Creating a MuP GPT2 model.")

    # NOTE: Hard-coding to GPT2 model for simplicity.
    # model = AutoModelForCausalLM.from_config(config)
    model = model_args.model.make_model()

    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"Model size={n_params/2**20:.2f}M params")

    model.resize_token_embeddings(len(tokenizer))
    return model


def preprocess_datasets(
    raw_datasets: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
) -> tuple[Dataset | None, Dataset | None]:

    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be "
                "chunked into smaller bits before being passed to the model."
            )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with training_args.main_process_first(desc="grouping texts together"):
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    train_dataset: Dataset | datasets.arrow_dataset.Dataset | None = None
    eval_dataset: Dataset | datasets.arrow_dataset.Dataset | None = None

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    return train_dataset, eval_dataset


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
