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
import argparse
import collections
from pkgutil import get_data

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

import evaluate
from accelerate import Accelerator, DistributedType
from datasets.load import load_dataset
from torchmetrics import BinnedRecallAtFixedPrecision
import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)


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
from collections import Counter


def get_dataloaders(accelerator: Accelerator, batch_size: int):
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
        tokenized_datasets["train"],
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=batch_size,
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"],
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=EVAL_BATCH_SIZE,
    )

    return train_dataloader, eval_dataloader


def training_function(config, args):
    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])

    valid_metric = evaluate.load("glue", "mrpc")

    # If the batch size is too big we use gradient accumulation
    gradient_accumulation_steps = 1
    if batch_size > MAX_GPU_BATCH_SIZE:
        gradient_accumulation_steps = batch_size // MAX_GPU_BATCH_SIZE
        batch_size = MAX_GPU_BATCH_SIZE

    # Initialize accelerator
    accelerator = Accelerator(
        cpu=args.cpu,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    set_seed(seed)
    train_dataloader, eval_dataloader = get_dataloaders(accelerator, batch_size)
    # Instantiate the model (we build the model here so that the seed also control new weights initialization)

    # BUG: This is a weird config bug happening that I don't understand!
    # BUG: The validation metric is always the same value, after every epoch, and doesn't change!
    # NOTE:
    # - When using a pretrained model, the accuracy goes from .3 to 0.68 over 3 epochs.
    # - When using a fresh model, the accuracy goes directly to 0.68, regardless of training.
    from transformers import BertForSequenceClassification, BertConfig, AutoConfig

    # model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")
    # model_config = AutoConfig.from_pretrained("bert-base-cased")
    model = BertForSequenceClassification(BertConfig())

    # from transformers import AutoConfig
    # model_config = AutoConfig.from_pretrained("bert-base-cased")
    # model = BertForSequenceClassification(model_config)

    # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
    # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
    # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
    # model = model.to(accelerator.device)

    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=lr)

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(len(train_dataloader) * num_epochs)
        // gradient_accumulation_steps,
    )

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    train_step_metric = evaluate.load("glue", "mrpc")
    train_epoch_metric = evaluate.load("glue", "mrpc")

    # Now we train the model
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm.tqdm(train_dataloader, desc=f"Train Epoch {epoch}")
        prediction_counter = collections.Counter()
        for step, batch in enumerate(pbar):
            with accelerator.accumulate(model):
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
            prediction_counter.update(predictions.tolist())
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
        accelerator.print(f"Training Predictions: {prediction_counter}")
        prediction_counter.clear()
        model.eval()
        eval_pbar = tqdm.tqdm(eval_dataloader, desc=f"Validation epoch {epoch}")
        for step, batch in enumerate(eval_pbar):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            # batch.to(accelerator.device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics(
                (predictions, batch["labels"])
            )
            prediction_counter.update(predictions.tolist())
            valid_metric.add_batch(
                predictions=predictions,
                references=references,
            )
        accelerator.print(f"Validation predictions: {prediction_counter}")
        # Use accelerator.print to print only on the main process.
        accelerator.print(
            f"epoch {epoch}:",
            f"\tTrain: {train_epoch_metric.compute()}",
            f"\tValid: {valid_metric.compute()}",
            sep="\n",
        )


def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="If passed, will train on the CPU."
    )
    args = parser.parse_args()
    config = {"lr": 2e-5, "num_epochs": 3, "seed": 12, "batch_size": 128}
    training_function(config, args)


if __name__ == "__main__":
    main()
