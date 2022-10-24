from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable, TypedDict

import torch
import yaml
from orion.client import build_experiment
from orion.core.worker.trial import Trial
from typing_extensions import NotRequired

from mup_demo.model import HParams
from mup_demo.manual_example.train import Config, training_function
from mup_demo.utils import is_main_process, suggest_trial

# class Batch(TypedDict):
#     labels: torch.Tensor
#     input_ids: torch.Tensor
#     token_type_ids: torch.Tensor
#     attention_mask: torch.Tensor


class TrainingFunctionOutput(TypedDict):
    loss: float
    accuracy: NotRequired[float]


def tune(
    training_function: Callable[[HParams, Config], TrainingFunctionOutput] = training_function
):
    """Perform an HPO sweep using smaller transformer models, and extract the best HPO parameters
    found.

    Then, use those parameters to train a very large model.
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
        config_for_this_trial = dataclasses.replace(config, log_dir=Path(trial.working_dir))
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
    print("Best params:")
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
