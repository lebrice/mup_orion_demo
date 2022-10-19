from __future__ import annotations
import typing
from datasets import DatasetDict, DatasetInfo, SplitInfo, load_dataset
from transformers import AutoTokenizer
import contextlib
from transformers import BatchEncoding
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from datasets.load import load_dataset_builder
from accelerate import Accelerator


if typing.TYPE_CHECKING:
    from .train import Config

# NOTE: Making the tokenizer global, so that the dataset can be cached.
# Doesn't seem to work, since the datasets are re-processed each time..
# If the tokenizer function is a method, then it seems to change the hash, which prevents caching.
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples: dict) -> BatchEncoding:
    return tokenizer(examples["text"], padding="max_length", truncation=True)


class YelpDataModule:
    def __init__(self, batch_size: int, config: Config):
        self.config = config
        self.batch_size = batch_size
        self.train_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

        self.dataset_builder = load_dataset_builder("yelp_review_full")

    @property
    def info(self) -> DatasetInfo:
        return self.dataset_builder.info

    @property
    def features(self):
        return self.dataset_builder.info.features

    @property
    def num_train_samples(self) -> int:
        assert self.dataset_builder.info.splits
        splits: dict[str, SplitInfo] = self.dataset_builder.info.splits
        return splits["train"].num_examples

    def prepare_data(self, accelerator: Accelerator | None = None):
        # Runs only on the first worker.
        dataset = load_dataset("yelp_review_full")
        assert isinstance(dataset, DatasetDict)
        # self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        with (
            accelerator.main_process_first()
            if accelerator
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
