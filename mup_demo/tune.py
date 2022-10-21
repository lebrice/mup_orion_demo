from __future__ import annotations
import dataclasses
from typing import TypedDict
from mutransformers import BertConfig, BertForSequenceClassification
from mup import MuAdamW
import os
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from transformers import get_scheduler
import torch
from transformers.modeling_outputs import SequenceClassifierOutput
from torch import Tensor
from orion.client import build_experiment
import yaml
from orion.core.worker.trial import Trial
from pathlib import Path
from torch.optim import Optimizer
from torch.nn.parallel import DistributedDataParallel as DDP
import json
from mup_demo.data import YelpDataModule
from accelerate import Accelerator
from accelerate.accelerator import AcceleratedScheduler
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from transformers import set_seed
from mup_demo.model import get_bert_model
from transformers import get_linear_schedule_with_warmup
import tqdm
from typing_extensions import NotRequired
from typing import Callable
from mup_demo.train import training_function, Config
from mup_demo.model import HParams
from mup_demo.utils import suggest_trial, is_main_process


class Batch(TypedDict):
    labels: torch.Tensor
    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    attention_mask: torch.Tensor


def train(
    hparams: HParams,
    config: Config,
):

    set_seed(config.random_seed)

    accelerator = Accelerator()

    config.log_dir.mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        print(f"Running in {config.log_dir}")
        with open(config.log_dir / "config.yaml", "w") as f:
            yaml.dump({"config": config, "hparams": hparams}, f)

    model = get_bert_model(hparams.model)
    datamodule = YelpDataModule(
        batch_size=hparams.batch_size, config=config, accelerator=accelerator
    )

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    datamodule.prepare_data()
    datamodule.setup("fit")

    train_dataloader = datamodule.train_dataloader()
    test_dataloader = datamodule.test_dataloader()

    optimizer = MuAdamW(model.parameters(), lr=hparams.learning_rate)
    num_training_steps = datamodule.num_train_samples * hparams.num_epochs

    lr_scheduler: AcceleratedScheduler | LRScheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=num_training_steps,
    )

    # TODO: Use the auto_batch_size stuff maybe?

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
    )  # type: ignore
    model: BertForSequenceClassification | DDP
    train_dataloader: DataLoader
    test_dataloader: DataLoader

    epoch_metrics = valid_epoch(model=model, test_dataloader=test_dataloader)

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
                progress_bar.set_postfix({"loss": loss.item()})

        epoch_metrics = test_epoch(model=model, test_dataloader=test_dataloader)

        if accelerator.is_main_process:
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


class TrainingFunctionOutput(TypedDict):
    loss: float
    accuracy: NotRequired[float]


def tune(
    training_function: Callable[
        [HParams, Config], TrainingFunctionOutput
    ] = training_function
):
    """Perform an HPO sweep using smaller transformer models, and extract the best HPO parameters
    found. Then, use those parameters to train a very large model.
    """

    # TODO: The sweep directory should actually be immutable, and by that I mean, every change to
    # the sweep parameters / space / config / etc should trigger a new sweep directory to be used.

    base_log_dir = Path("logs")
    sweep_log_dir = base_log_dir / "test_sweep"
    config = Config(
        max_train_samples=10_000,
        max_test_samples=1000,
        dataloader_num_workers=4,
    )

    experiment = build_experiment(
        name="mup",
        space=HParams.get_orion_space_dict(),
        storage={
            "type": "legacy",
            "database": {"type": "pickleddb", "host": str(sweep_log_dir / "db.pkl")},
        },
        max_trials=10,
        working_dir=sweep_log_dir,
    )

    while not experiment.is_done:
        trial = suggest_trial(experiment)
        print(f"Experiment suggested hparams: {trial.params}")
        hparams = HParams(**trial.params)

        # Use the 'base' config, but replace the log_dir with the trial's working_dir.
        config_for_this_trial = dataclasses.replace(
            config, log_dir=Path(trial.working_dir)
        )
        metrics = training_function(hparams, config_for_this_trial)
        # metrics = train(hparams, config_for_this_trial)

        if is_main_process():
            print(f"Trial {trial.id} finished with metrics: {metrics}")
            experiment.observe(
                trial,
                # NOTE: Put the loss as the first objective, so that Orion uses it. Also keep the
                # other metrics as additional objectives.
                [dict(name="valid_loss", value=metrics["loss"], type="objective")]
                # + [
                #     dict(name=key, value=value, type="objective")
                #     for key, value in metrics.items()
                #     if key != "loss"
                # ],
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
        # metrics = get_trial_metrics(trial)
        print(f"{trial.working_dir}:", trial.params, trial.results)

    # TODO: Run some training on the bigger model.
    # train_big_model(best_trial)


def train_big_model(best_trial: Trial):
    # Reuse the hparams that we found on the small model, to train a big model only once!
    # NOTE: Assuming that the hparams we were sweeping over don't have to do with the model width!
    with open(Path(best_trial.working_dir) / "hparams.yaml") as f:
        best_model_hparams: HParams = yaml.load(f, loader=yaml.FullLoader)

    big_model_hparams = dataclasses.replace(
        best_model_hparams,
        num_epochs=100,
        model=dataclasses.replace(
            best_model_hparams.model,
            hidden_size=1024,
            intermediate_size=1024 * 4,
            num_attention_heads=32,
            num_labels=5,
        ),
    )
    big_model_training_config = Config(
        log_dir=Path("logs") / "test_sweep" / "big_model",
        max_train_samples=None,
        max_test_samples=None,
        dataloader_num_workers=torch.multiprocessing.cpu_count(),
    )
    training_function(hparams=big_model_hparams, config=big_model_training_config)


if __name__ == "__main__":
    tune()
