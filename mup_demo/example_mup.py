# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Literal, TypedDict
import warnings

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

import evaluate
import tqdm
from accelerate import Accelerator, DistributedType
from datasets.load import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)
from transformers import get_scheduler
from torch.optim import Optimizer
from mutransformers import BertConfig, BertForSequenceClassification
from mup import make_base_shapes, set_base_shapes, MuAdamW
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

########################################################################
# This is a fully working simple example to use Accelerate
#
# This example trains a Bert base model on GLUE MRPC
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - (multi) TPUs
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, follow the instructions
# in the readme for examples:
# https://github.com/huggingface/accelerate/tree/main/examples
#
########################################################################


MAX_GPU_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 32


def get_dataloaders(accelerator: Accelerator, batch_size: int = 16):
    """
    Creates a set of `DataLoader`s for the `glue` dataset,
    using "bert-base-cased" as the tokenizer.

    Args:
        accelerator (`Accelerator`):
            An `Accelerator` object
        batch_size (`int`, *optional*):
            The batch size for the train and validation DataLoaders.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    datasets = load_dataset("glue", "mrpc")

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            max_length=None,
        )
        return outputs

    # Apply the method we just defined to all the examples in all the splits of the dataset
    # starting with the main process first:
    with accelerator.main_process_first():
        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["idx", "sentence1", "sentence2"],
        )

    # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
    # transformers library
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    def collate_fn(examples):
        # On TPU it's best to pad everything to the same length or training will be very slow.
        if accelerator.distributed_type == DistributedType.TPU:
            return tokenizer.pad(
                examples, padding="max_length", max_length=128, return_tensors="pt"
            )
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    # Instantiate dataloaders.
    train_dataloader = DataLoader(
        tokenized_datasets["train"],  # type: ignore
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=batch_size,
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"],  # type: ignore
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=EVAL_BATCH_SIZE,
    )

    return train_dataloader, eval_dataloader


def replace(model_config: BertConfig, **kwargs) -> BertConfig:
    delta_config = model_config.to_dict()
    delta_config.update(**kwargs)
    return type(model_config).from_dict(delta_config)


def get_model(target_config: BertConfig) -> BertForSequenceClassification:
    model_class = BertForSequenceClassification
    # define a base model
    base_config = BertConfig(
        hidden_size=256,
        intermediate_size=256,
        num_attention_heads=16,
        # num_labels=5,
    )
    base_model = model_class(base_config)

    # OK seems like I'm making some progress.

    # base_model = BertForSequenceClassification.from_pretrained("bert-base-cased")
    # base_config = base_model.config
    # assert isinstance(base_config, BertConfig)
    # define a delta models where we vary all "widths" we want to vary
    delta_config = replace(
        base_config,
        hidden_size=200,
        intermediate_size=300,
        num_attention_heads=5,
    )
    # TODO: There seems to be a bug that happens only when using a BertForSequenceClassification
    # model directly, instead of using a pretrained model: The validation metrics are fixed, and
    # don't change at all, regardless of training. This is not the case when using a pretrained
    # model.
    delta_model = model_class(delta_config)

    # base_model = model_class(config=base_config)
    # delta_model = model_class(config=delta_config)
    # define a base shape object based on comparing delta_model against base_model
    base_shapes = make_base_shapes(base_model, delta_model, savefile="bert256.bsh")

    # FIXME: Trying this instead, but it doesn't work!
    # target_model = type(base_model)(target_config)
    target_model = model_class(config=target_config)
    # set base shapes
    set_base_shapes(target_model, base_shapes)
    # you can alternatively load base shape from file
    # set_base_shapes(target_model, 'bert256.bsh')

    # re-initialize
    target_model.apply(target_model._init_weights)
    print(f"Total parameters in the base model:   {base_model.num_parameters()}")
    print(f"Total parameters in the delta model:  {delta_model.num_parameters()}")
    print(f"Total parameters in the target model: {target_model.num_parameters()}")
    return target_model


def training_function(hparams: HParams, config: Config):
    # Initialize accelerator
    accelerator = Accelerator(cpu=config.cpu, mixed_precision=config.mixed_precision)
    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = hparams.lr
    num_epochs = hparams.num_epochs
    seed = hparams.seed
    batch_size = hparams.batch_size

    eval_metric = evaluate.load("glue", "mrpc")

    # If the batch size is too big we use gradient accumulation
    gradient_accumulation_steps = 1
    if (
        batch_size > MAX_GPU_BATCH_SIZE
        and accelerator.distributed_type != DistributedType.TPU
    ):
        gradient_accumulation_steps = batch_size // MAX_GPU_BATCH_SIZE
        batch_size = MAX_GPU_BATCH_SIZE

    set_seed(seed)
    train_dataloader, eval_dataloader = get_dataloaders(accelerator, batch_size)
    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     "bert-base-cased", return_dict=True
    # )
    from transformers import AutoConfig

    default_config = BertConfig.from_pretrained("bert-base-cased")
    big_config = replace(
        default_config,
        hidden_size=1024,
        intermediate_size=1024 * 4,
        num_attention_heads=32,
    )

    small_config = replace(
        default_config,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        # num_labels=2,  # TODO: This is specific to this particular dataset.
    )
    model = get_model(small_config)
    # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
    # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
    # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
    # model = model.to(accelerator.device)

    # Instantiate optimizer
    optimizer = MuAdamW(params=model.parameters(), lr=lr)

    # Instantiate scheduler
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(len(train_dataloader) * num_epochs)
        // gradient_accumulation_steps,
    )
    # lr_scheduler = get_linear_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=100,
    #     num_training_steps=(len(train_dataloader) * num_epochs)
    #     // gradient_accumulation_steps,
    # )

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order
    # we gave them to the prepare method.
    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )  # type: ignore
    from accelerate.accelerator import AcceleratedOptimizer, AcceleratedScheduler
    from torch.optim.lr_scheduler import _LRScheduler

    assert isinstance(optimizer, (AcceleratedOptimizer))
    assert isinstance(lr_scheduler, (_LRScheduler, AcceleratedScheduler))

    train_step_metric = evaluate.load("glue", "mrpc")
    train_epoch_metric = evaluate.load("glue", "mrpc")

    # Now we train the model
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            with accelerator.accumulate(model), warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                # We could avoid this line since we set the accelerator with `device_placement=True`.
                # batch.to(accelerator.device)
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            postfix = {"loss": loss.detach().item()}
            predictions = outputs.logits.detach().argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics(
                (predictions, batch["labels"])
            )
            train_epoch_metric.add_batch(
                predictions=predictions,
                references=references,
            )
            train_step_metric.add_batch(
                predictions=predictions,
                references=references,
            )
            postfix.update(train_step_metric.compute())
            pbar.set_postfix(postfix)
            # print(f"step {step}: {loss} {train_metric_result}")

        model.eval()
        eval_pbar = tqdm.tqdm(eval_dataloader, desc=f"Evaluation epoch {epoch}")
        for step, batch in enumerate(eval_pbar):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            # batch.to(accelerator.device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics(
                (predictions, batch["labels"])
            )
            eval_metric.add_batch(
                predictions=predictions,
                references=references,
            )

        # Use accelerator.print to print only on the main process.
        accelerator.print(
            f"epoch {epoch}:",
            f"\tTrain: {train_epoch_metric.compute()}",
            f"\tValid: {eval_metric.compute()}",
            sep="\n",
        )


@dataclass
class Config:
    mixed_precision: Literal["no", "fp16", "bf16"] = "no"
    """Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16).
    Bf16 requires PyTorch >= 1.10 and an Nvidia Ampere GPU.
    """

    cpu: bool = False
    """If passed, will train on the CPU."""


@dataclass
class HParams:
    lr: float = 2e-5
    num_epochs: int = 3
    seed: int = 42
    batch_size: int = 128


def main():
    import simple_parsing

    parser = simple_parsing.ArgumentParser(
        description="Simple example of training script."
    )
    parser.add_arguments(Config, dest="config")
    args = parser.parse_args()
    config: Config = args.config
    hparams = HParams()
    training_function(hparams, config)


if __name__ == "__main__":
    main()
