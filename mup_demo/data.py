from __future__ import annotations
import typing
from datasets import DatasetDict, DatasetInfo, Features, SplitInfo, load_dataset
from transformers import AutoTokenizer
import contextlib
import evaluate
from evaluate import EvaluationModule
from transformers import BatchEncoding
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from datasets.load import load_dataset_builder
from accelerate.accelerator import AcceleratorState
from accelerate.accelerator import prepare_data_loader
from accelerate import Accelerator, DistributedType
from datasets.builder import DatasetBuilder

if typing.TYPE_CHECKING:
    from .train import Config

# NOTE: Making the tokenizer global, so that the dataset can be cached.
# Doesn't seem to work, since the datasets are re-processed each time..
# If the tokenizer function is a method, then it seems to change the hash, which prevents caching.
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples: dict) -> BatchEncoding:
    return tokenizer(examples["text"], padding="max_length", truncation=True)


class TextDataModule:
    def __init__(
        self,
        dataset_name: str,
        task: str | None = None,
        batch_size: int = 16,
        accelerator: Accelerator | None = None,
    ):
        self.dataset_name = dataset_name
        self.task = task
        self.dataset_name_and_task = (dataset_name, task)

        self.batch_size = batch_size
        self.accelerator = accelerator

        self.dataset_builder: DatasetBuilder = load_dataset_builder(
            *self.dataset_name_and_task
        )
        self.train_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    @property
    def info(self) -> DatasetInfo:
        return self.dataset_builder.info

    @property
    def features(self) -> Features:
        features = self.dataset_builder.info.features
        assert features is not None
        return features

    @property
    def num_train_samples(self) -> int:
        assert self.dataset_builder.info.splits
        splits: dict[str, SplitInfo] = self.dataset_builder.info.splits
        return splits["train"].num_examples

    def make_metric(self) -> EvaluationModule:
        return evaluate.load(*self.dataset_name_and_task)


class YelpDataModule(TextDataModule):
    def __init__(
        self,
        config: Config,
        batch_size: int = 16,
        accelerator: Accelerator | None = None,
    ):
        super().__init__(
            dataset_name="yelp_review_full",
            batch_size=batch_size,
            accelerator=accelerator,
        )
        self.config = config

        self.train_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    @property
    def num_labels(self) -> int:
        return self.features["label"].num_classes

    def make_metric(self) -> EvaluationModule:
        return evaluate.load("glue", "mrpc")

    def prepare_data(self):
        # Runs only on the first worker.
        dataset = load_dataset("yelp_review_full")
        assert isinstance(dataset, DatasetDict)
        # self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        with (
            self.accelerator.main_process_first()
            if self.accelerator
            else contextlib.nullcontext()
        ):
            tokenized_datasets = dataset.map(
                # tokenizer,  # todo: look into using the tokenizer directly?
                tokenize_function,
                batched=True,
                load_from_cache_file=True,
                # NOTE: This is hard-coded, because the built-in caching of HuggingFace isn't
                # working for some reason!
                cache_file_names={
                    "train": str(self.config.sweep_dir / "train_cached"),
                    "test": str(self.config.sweep_dir / "test_cached"),
                },
                remove_columns=["text"],
            ).rename_column("label", "labels")
        tokenized_datasets.set_format("torch")

        train_dataset = tokenized_datasets["train"]
        test_dataset = tokenized_datasets["test"]
        if self.config.max_train_samples and self.config.max_train_samples < len(
            train_dataset
        ):
            train_dataset = train_dataset.select(range(self.config.max_train_samples))
        if self.config.max_test_samples and self.config.max_test_samples < len(
            test_dataset
        ):
            test_dataset = test_dataset.select(range(self.config.max_test_samples))
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def setup(self, stage: str | None = None):
        # Do it "again", but on all workers. This doesn't download anything the second time, but
        # it does set the `train_dataset` and `test_dataset` attributes on `self` for each worker.
        self.prepare_data()

    def train_dataloader(self):
        if self.train_dataset is None:
            self.setup()
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.config.dataloader_num_workers,
            shuffle=True,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            self.setup()
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.config.dataloader_num_workers,
        )


class GlueDataModule(TextDataModule):
    def __init__(
        self,
        accelerator: Accelerator,
        batch_size: int = 16,
        eval_batch_size: int | None = None,
    ):
        super().__init__(
            dataset_name="glue",
            task="mrpc",
            batch_size=batch_size,
            accelerator=accelerator,
        )
        self.eval_batch_size = eval_batch_size or batch_size

        self.tokenized_datasets: DatasetDict | None = None

        self.train_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    @property
    def num_labels(self) -> int:
        return self.features["label"].num_classes

    def prepare_data(self):
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
        with self.accelerator.main_process_first():
            tokenized_datasets = datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=["idx", "sentence1", "sentence2"],
            )

        # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
        # transformers library
        self.tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    def train_dataloader(self) -> DataLoader:
        """
        Creates a set of `DataLoader`s for the `glue` dataset,
        using "bert-base-cased" as the tokenizer.

        Args:
            accelerator (`Accelerator`):
                An `Accelerator` object
            batch_size (`int`, *optional*):
                The batch size for the train and validation DataLoaders.
        """
        if self.tokenized_datasets is None:
            self.prepare_data()

        # Instantiate dataloaders.
        return DataLoader(
            self.tokenized_datasets["train"],  # type: ignore
            shuffle=True,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
        )

    def collate_fn(self, examples):
        # On TPU it's best to pad everything to the same length or training will be very slow.
        if self.accelerator.distributed_type == DistributedType.TPU:
            return tokenizer.pad(
                examples, padding="max_length", max_length=128, return_tensors="pt"
            )
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.tokenized_datasets["validation"],  # type: ignore
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.eval_batch_size,
        )


def get_glue_dataloaders(accelerator: Accelerator, batch_size: int = 16):
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
