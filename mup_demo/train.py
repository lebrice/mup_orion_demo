from __future__ import annotations
from dataclasses import dataclass
import dataclasses
import random
from typing import TypedDict, Union
from mutransformers import BertConfig, BertForSequenceClassification
from mup import make_base_shapes, set_base_shapes, MuAdamW
from datasets import DatasetDict, load_dataset
from dataclasses import field
import functools
import os
from transformers import AutoTokenizer
import numpy as np
from transformers import BatchEncoding
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import get_scheduler
import torch
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.utils.data import Dataset
from torch import Tensor
import numpy as np
from orion.client import build_experiment
import yaml
from orion.core.worker.trial import Trial
from simple_parsing.helpers.hparams import HyperParameters, log_uniform
from pathlib import Path
from torch.optim import Optimizer
from torch.nn.parallel import DistributedDataParallel as DDP
import json


def get_model(target_config: BertConfig) -> BertForSequenceClassification:
    model_class = BertForSequenceClassification
    # define a base model
    base_config = BertConfig(
        hidden_size=256,
        intermediate_size=256,
        num_attention_heads=16,
        num_labels=5,
    )
    # define a delta models where we vary all "widths" we want to vary
    delta_config = BertConfig(
        hidden_size=200,
        intermediate_size=300,
        num_attention_heads=5,
        num_labels=5,
    )
    # TODO: Not sure I understand how HPO fits into this. Do we do HPO on the base config? or on
    # the target config, but with lower values than the base config?

    # define target model
    # NOTE: Original config:
    # target_config = BertConfig(
    #     hidden_size=1024,
    #     intermediate_size=1024 * 4,
    #     num_attention_heads=32,
    #     num_labels=5,
    # )
    # TODO: Scale this correctly based on some "scale factor"?
    # target_config = BertConfig(
    #     hidden_size=64 * scale_factor,
    #     intermediate_size=64 * scale_factor * 4,
    #     num_attention_heads=max(1, int(8 * np.log2(scale_factor))),
    #     num_labels=5,
    # )

    base_model = model_class(config=base_config)
    delta_model = model_class(config=delta_config)
    # define a base shape object based on comparing delta_model against base_model
    base_shapes = make_base_shapes(base_model, delta_model, savefile="bert256.bsh")

    target_model = model_class(config=target_config)
    # set base shapes
    set_base_shapes(target_model, base_shapes)
    # you can alternatively load base shape from file
    # set_base_shapes(target_model, 'bert256.bsh')

    # re-initialize
    target_model.apply(target_model._init_weights)

    return target_model


class Batch(TypedDict):
    labels: torch.Tensor
    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    attention_mask: torch.Tensor


# NOTE: Making the tokenizer global, so that the dataset can be cached.
# Doesn't seem to work, since the datasets are re-processed each time..
# If the tokenizer function is a method, then it seems to change the hash, which prevents caching.
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples: dict) -> BatchEncoding:
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# TODO: Can't get this to work.
# def compute_metrics(eval_preds):
#     metric = evaluate.load("accuracy")
#     logits, labels = eval_preds
#     # predictions = np.argmax(logits, axis=-1)
#     return metric.compute(
#         logits=logits,
#         references=labels,
#         # predictions=predictions,
#         average="macro",
#     )


@dataclass
class HParams(HyperParameters):
    learning_rate: float = log_uniform(1e-7, 1e-2, default=0.00005)
    # batch_size: int = log_uniform(4, 128, default=32, base=2, discrete=True)
    batch_size: int = 128
    model: BertConfig = field(
        default_factory=functools.partial(
            BertConfig,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_labels=5,  # TODO: This is specific to this particular dataset.
        )
    )


@dataclass(frozen=True)
class Config:
    log_dir: Path = Path("logs")
    num_epochs: int = 1
    max_train_samples: int | None = 10_000
    max_test_samples: int | None = 1_000
    dataloader_num_workers: int = 4

    random_seed: int = 42


def get_optimizer(model: BertForSequenceClassification, hparams: HParams):
    # make sure to use mup optimizers for training
    return MuAdamW(model.parameters(), lr=hparams.learning_rate)


def get_lr_scheduler(
    optimizer: Optimizer, train_dataset_length: int, num_epochs: int, batch_size: int
):
    num_training_steps = int(num_epochs) * (train_dataset_length // batch_size)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    assert isinstance(lr_scheduler, torch.optim.lr_scheduler._LRScheduler)
    return lr_scheduler


class YelpDataModule:
    def __init__(self, batch_size: int, config: Config):
        self.config = config
        self.batch_size = batch_size
        self.train_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    def prepare_data(self):
        # Runs only on the first worker.
        dataset = load_dataset("yelp_review_full")
        assert isinstance(dataset, DatasetDict)
        # self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        tokenized_datasets = (
            dataset.map(
                tokenize_function,
                batched=True,
                load_from_cache_file=True,
                # NOTE: This is hard-coded, because the caching isn't working.
                cache_file_names={"train": "train_cached", "test": "test_cached"},
            )
            .remove_columns(["text"])
            .rename_column("label", "labels")
        )
        tokenized_datasets.set_format("torch")

        train_dataset = tokenized_datasets["train"]
        test_dataset = tokenized_datasets["test"]
        if self.config.max_train_samples:
            train_dataset = train_dataset.select(range(self.config.max_train_samples))
        if self.config.max_test_samples:
            test_dataset = test_dataset.select(range(self.config.max_test_samples))
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def setup(self, stage: str):
        # Do it "again", but on all workers. This doesn't download anything the second time, but
        # it does set the `train_dataset` and `test_dataset` attributes on `self` for each worker.
        self.prepare_data()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.config.dataloader_num_workers,
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.config.dataloader_num_workers,
        )


def train(
    hparams: HParams,
    config: Config,
):

    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.random.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)

    config.log_dir.mkdir(parents=True, exist_ok=True)
    local_rank = int(os.environ.get("RANK", -1))
    if local_rank in [-1, 0]:
        print(f"Running in {config.log_dir}")
        with open(config.log_dir / "config.yaml", "w") as f:
            yaml.dump({"config": config, "hparams": hparams}, f)

    model = get_model(hparams.model)
    datamodule = YelpDataModule(batch_size=hparams.batch_size, config=config)

    # mimicking a PyTorch Lightning Trainer-like API.
    datamodule.prepare_data()
    datamodule.setup("fit")
    train_dataset_length = len(datamodule.train_dataset)

    train_dataloader = datamodule.train_dataloader()
    test_dataloader = datamodule.test_dataloader()

    optimizer = get_optimizer(model=model, hparams=hparams)
    lr_scheduler = get_lr_scheduler(
        optimizer=optimizer,
        train_dataset_length=train_dataset_length,
        num_epochs=config.num_epochs,
        batch_size=hparams.batch_size,
    )

    # TODO: Use the auto_batch_size stuff maybe?
    # Make sure not to actually change the model state though.
    # initial_model_state = model.save_pretrained("temp")
    from accelerate import Accelerator

    accelerator = Accelerator()
    # NOTE: set_base_shapes is causing issues with FSDP, since the weight names don't line up with
    # the model anymore.
    (
        train_dataloader,
        test_dataloader,
        model,
        optimizer,
        lr_scheduler,
    ) = accelerator.prepare(
        train_dataloader, test_dataloader, model, optimizer, lr_scheduler
    )
    model: BertForSequenceClassification | DDP
    train_dataloader: DataLoader
    test_dataloader: DataLoader

    # Register the LR scheduler
    accelerator.register_for_checkpointing(lr_scheduler)

    epoch_metrics = test_epoch(model=model, test_dataloader=test_dataloader)

    # TODO: Resume interrupted training.
    # if epoch_save_path.exists() and epoch_save_path.is_dir():

    for epoch in range(config.num_epochs):
        epoch_save_path = config.log_dir / f"epoch_{epoch:03d}"

        accelerator.save_state(str(epoch_save_path))

        progress_bar = tqdm(train_dataloader, desc=f"training epoch {epoch}")
        for batch in progress_bar:
            with accelerator.accumulate(model):
                outputs: SequenceClassifierOutput = model(**batch)
                loss = outputs.loss
                assert isinstance(loss, Tensor)
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        epoch_metrics = test_epoch(model=model, test_dataloader=test_dataloader)

        if local_rank in [-1, 0]:
            with open(epoch_save_path / "metrics.json", "w") as f:
                json.dump(epoch_metrics, f)

            print(f"Epoch {epoch}:")
            print(f"\tEval Loss: {epoch_metrics['loss']:.4f}")
            print(f"\tAccuracy: {epoch_metrics['accuracy']:.2%}")

            print(epoch_metrics)
            torch.save(model.state_dict(), config.log_dir / f"model_{epoch}.pt")
        # TODO: Would be nicer to be able to use this from HuggingFace:
        # model.save_pretrained(config.log_dir / f"epoch_{epoch}")

    return epoch_metrics


class EpochMetrics(TypedDict):
    loss: float
    accuracy: float
    n: int


def test_epoch(
    model: BertForSequenceClassification | DDP, test_dataloader: DataLoader
) -> EpochMetrics:
    accurate = 0
    total = 0
    total_loss = 0.0
    model.eval()
    test_progress_bar = tqdm(test_dataloader, desc=f"Testing...")
    for batch in test_progress_bar:
        with torch.no_grad():
            predictions: SequenceClassifierOutput = model(**batch)
            y = batch["labels"]
            accurate += predictions.logits.argmax(-1).eq(y).int().sum().item()
            total += y.shape[0]
            loss = predictions.loss
            assert isinstance(loss, Tensor)
            total_loss += loss.item()
    accuracy = accurate / total
    return {"accuracy": accuracy, "loss": total_loss, "n": total}


def main():
    """Perform an HPO sweep using smaller transformer models, and extract the best HPO parameters
    found. Then, use those parameters to train a very large model.
    """
    base_log_dir = Path("logs")
    sweep_log_dir = base_log_dir / "test_sweep"
    config = Config(
        num_epochs=1,
        max_train_samples=10_000,
        max_test_samples=1000,
        dataloader_num_workers=4,
    )

    experiment = build_experiment(
        name="mup",
        space=HParams.get_orion_space_dict(),
        # space={"learning_rate": "log_uniform(1e-7, 1e-2, default=1e-3)"},
        storage={
            "type": "legacy",
            "database": {"type": "pickleddb", "host": str(sweep_log_dir / "db.pkl")},
        },
        max_trials=20,
        working_dir=sweep_log_dir,
    )
    store = torch.distributed.FileStore("filestore", int(os.environ["WORLD_SIZE"]))

    local_rank = int(os.environ["RANK"])

    while not experiment.is_done:
        # BUG: When using DDP, I'm getting a different trial from experiment.suggest for each
        # worker!
        # TODO: Find the trial that was suggested on the master worker (the one with RANK 0)

        # TODO: Perhaps we could use this as part of a filename ?
        n_completed_trials = experiment.algorithms.n_observed
        if local_rank == 0:
            trial = experiment.suggest()
            trial_params = trial.params
            trial_id = trial.id
            log_dir = Path(trial.working_dir)

            # TODO: Somehow serialize the Trial and send it to the other workers.
            for key, value in trial_params.items():
                store.set(key, str(value))
            store.set("log_dir", str(log_dir))
            store.set("trial_id", trial_id)
        else:
            trial_params = {
                key: float(store.get(key).decode())
                for key in HParams.get_orion_space_dict()
            }
            log_dir = Path(store.get("log_dir").decode())
            trial_id = store.get("trial_id").decode()

        hparams = HParams(**trial_params)
        print(f"Worker {local_rank} got learning_rate of {hparams.learning_rate}")
        print(f"Running trial with hparams: {trial_params}")

        # BUG: STILL getting different values for different workers!!

        config_for_this_trial = dataclasses.replace(config, log_dir=log_dir)

        metrics = train(hparams, config_for_this_trial)
        print(f"Trial {trial_id} finished with metrics: {metrics}")

        if local_rank == 0:
            experiment.observe(
                trial,
                # NOTE: Put the loss as the first objective, so that Orion uses it. Also keep the
                # other metrics as additional objectives.
                [dict(name="test_loss", value=metrics["loss"], type="objective")]
                + [
                    dict(name=key, value=value, type="objective")
                    for key, value in metrics.items()
                    if key != "loss"
                ],
            )

    # Idea: Could we add something like a 'best_trial_so_far' property/method on the Experiment
    # object?
    # TODO: This isn't typed.
    trials = experiment.fetch_trials_by_status("completed")
    best_trial = min(trials, key=lambda trial: trial.objective.value)
    print(f"Best trial: {best_trial.id} with objective: {best_trial.objective.value}")
    print(f"Best trial params: {best_trial.params}")

    # IDEA: Make the HUGE model now!
    # TODO: Should we make it in this script here, or in a separate step?
    print(f"Best params:")
    for trial in sorted(trials, key=lambda trial: trial.objective.value):
        metrics = get_trial_metrics(trial)
        print(f"{trial.working_dir}:", trial.params, metrics)

    # TODO: Run some training on the bigger model.
    # train_big_model(best_trial)


def get_trial_metrics(trial: Trial) -> EpochMetrics:
    last_epoch_folder = sorted(Path(trial.working_dir).glob("epoch_*"))[-1]
    with open(last_epoch_folder / "metrics.json") as f:
        return json.load(f)


def train_big_model(best_trial: Trial):
    # Reuse the hparams that we found on the small model, to train a big model only once!
    # NOTE: Assuming that the hparams we were sweeping over don't have to do with the model width!
    big_model_hparams = HParams(
        **best_trial.params,
        model=BertConfig(
            hidden_size=1024,
            intermediate_size=1024 * 4,
            num_attention_heads=32,
            num_labels=5,
        ),
    )
    big_model_training_config = Config(
        log_dir=Path("logs") / "test_sweep" / "big_model",
        num_epochs=100,
        max_train_samples=None,
        max_test_samples=None,
        dataloader_num_workers=torch.multiprocessing.cpu_count(),
    )
    train(big_model_hparams, big_model_training_config)


if __name__ == "__main__":
    main()
