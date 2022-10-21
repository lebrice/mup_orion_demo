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
from dataclasses import dataclass
import yaml
import collections
from typing import Literal
import warnings

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import evaluate
import tqdm
from accelerate import Accelerator, DistributedType
from datasets.load import load_dataset
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)
from mutransformers import BertConfig, BertForSequenceClassification
from mup import MuAdamW
import warnings
from mup_demo.model import _replace, get_bert_model
from accelerate.accelerator import AcceleratedOptimizer, AcceleratedScheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from evaluate import EvaluationModule
from mup_demo.data import GlueDataModule
from pathlib import Path
from dataclasses import asdict

warnings.filterwarnings("ignore", category=FutureWarning)

MAX_GPU_BATCH_SIZE = 256
EVAL_BATCH_SIZE = 32

from mup_demo.model import HParams


@dataclass(frozen=True)
class Config:
    log_dir: Path = Path("logs")
    max_train_samples: int | None = 10_000
    max_test_samples: int | None = 1_000
    dataloader_num_workers: int = 4

    mixed_precision: Literal["no", "fp16", "bf16"] = "no"
    """Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16).
    Bf16 requires PyTorch >= 1.10 and an Nvidia Ampere GPU.
    """

    cpu: bool = False
    """If passed, will train on the CPU."""

    @property
    def sweep_dir(self) -> Path:
        return self.log_dir.parent


def training_function(hparams: HParams, config: Config):
    set_seed(hparams.random_seed)

    config.log_dir.mkdir(parents=True, exist_ok=True)

    # Initialize accelerator
    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = hparams.learning_rate
    num_epochs = hparams.num_epochs
    batch_size = hparams.batch_size

    valid_epoch_metric = evaluate.load("glue", "mrpc")

    # If the batch size is too big we use gradient accumulation
    gradient_accumulation_steps = 1
    if batch_size > MAX_GPU_BATCH_SIZE:
        gradient_accumulation_steps = batch_size // MAX_GPU_BATCH_SIZE
        batch_size = MAX_GPU_BATCH_SIZE

    accelerator = Accelerator(
        cpu=config.cpu,
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    if accelerator.is_main_process:
        print(f"Running in {config.log_dir}")
        with open(config.log_dir / "config.yaml", "w") as f:
            yaml.dump({"config": config, "hparams": hparams}, f)

    # train_dataloader, eval_dataloader = get_dataloaders(accelerator, batch_size)
    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     "bert-base-cased", return_dict=True
    # )
    datamodule = GlueDataModule(
        accelerator=accelerator, batch_size=batch_size, eval_batch_size=EVAL_BATCH_SIZE
    )
    train_dataloader = datamodule.train_dataloader()
    eval_dataloader = datamodule.val_dataloader()

    default_config = BertConfig.from_pretrained("bert-base-cased")
    assert isinstance(default_config, BertConfig)

    small_config = _replace(
        default_config,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        # num_labels=2,  # TODO: This is specific to this particular dataset.
    )
    model = get_bert_model(small_config, BertForSequenceClassification)
    # model = ScalableBertModel(small_config)
    # model = get_model(small_config)
    # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
    # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
    # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
    # model = model.to(accelerator.device)

    # Instantiate optimizer
    optimizer = MuAdamW(params=model.parameters(), lr=lr)

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(len(train_dataloader) * num_epochs)
        // gradient_accumulation_steps,
    )

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

    assert isinstance(optimizer, (AcceleratedOptimizer))
    assert isinstance(lr_scheduler, (_LRScheduler, AcceleratedScheduler))

    train_epoch_metric = datamodule.make_metric()
    train_step_metric = datamodule.make_metric()
    valid_epoch_metric = datamodule.make_metric()

    def save_dir_for_epoch(epoch: int) -> Path:
        return config.log_dir / f"epoch_{epoch:02d}"

    def epoch_has_checkpoints(epoch: int) -> bool:
        files = ["optimizer.bin", "pytorch_model.bin", "scheduler.bin"]
        return all((save_dir_for_epoch(epoch) / file).exists() for file in files)

    # Now we train the model
    for epoch in range(1, num_epochs + 1):
        epoch_save_dir = save_dir_for_epoch(epoch)

        train_epoch_metric_value = train_epoch(
            model=model,
            train_dataloader=train_dataloader,
            accelerator=accelerator,
            epoch=epoch,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch_metric=train_epoch_metric,
            step_metric=train_step_metric,
        )

        valid_epoch_metric_value = validation_epoch(
            model=model,
            valid_dataloader=eval_dataloader,
            accelerator=accelerator,
            epoch=epoch,
            epoch_metric=valid_epoch_metric,
        )

        # Use accelerator.print to print only on the main process.
        accelerator.print(
            f"epoch {epoch}:",
            f"\tTrain: {train_epoch_metric_value}",
            f"\tValid: {valid_epoch_metric_value}",
            sep="\n",
        )
        accelerator.save_state(str(epoch_save_dir))

    return valid_epoch_metric_value


def train_epoch(
    model: BertForSequenceClassification | DDP,
    train_dataloader: DataLoader,
    accelerator: Accelerator,
    optimizer: AcceleratedOptimizer | AdamW,
    lr_scheduler: _LRScheduler | AcceleratedScheduler,
    epoch_metric: EvaluationModule,
    step_metric: EvaluationModule,
    epoch: int,
):
    model.train()
    prediction_counter = collections.Counter()

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
        prediction_counter.update(predictions.tolist())
        epoch_metric.add_batch(
            predictions=predictions,
            references=references,
        )
        train_step_metric_value = step_metric.compute(
            predictions=predictions,
            references=references,
        )
        assert isinstance(train_step_metric_value, dict)
        postfix.update(train_step_metric_value)
        pbar.set_postfix(postfix)

    accelerator.print(f"Training predictions: {prediction_counter}")
    prediction_counter.clear()
    return epoch_metric.compute()


def validation_epoch(
    model: BertForSequenceClassification | DDP,
    valid_dataloader: DataLoader,
    epoch_metric: EvaluationModule,
    accelerator: Accelerator,
    epoch: int,
):
    model.eval()
    prediction_counter = collections.Counter()
    total_loss = 0.0
    eval_pbar = tqdm.tqdm(valid_dataloader, desc=f"Evaluation epoch {epoch}")
    for step, batch in enumerate(eval_pbar):
        # We could avoid this line since we set the accelerator with `device_placement=True`.
        # batch.to(accelerator.device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references, loss = accelerator.gather_for_metrics(
            (predictions, batch["labels"], outputs.loss)
        )
        prediction_counter.update(predictions.tolist())
        total_loss += loss.sum().item()
        epoch_metric.add_batch(
            predictions=predictions,
            references=references,
        )

    accelerator.print(f"Evaluation predictions: {prediction_counter}")
    prediction_counter.clear()

    epoch_metric_values = epoch_metric.compute()
    assert isinstance(epoch_metric_values, dict)
    epoch_metric_values.setdefault("loss", total_loss)
    return epoch_metric_values


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
