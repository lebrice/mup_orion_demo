from __future__ import annotations

from pathlib import Path
from typing import Iterable

import orion
from orion.client import get_experiment
from orion.core.worker.trial import Trial
from simple_parsing.helpers.serialization.serializable import load_yaml

from mup_demo.train import (
    DataTrainingArguments,
    ModelArguments,
    TrainingArguments,
    evaluation_loop,
    setup_trainer,
    train,
)
from mup_demo.utils import load_training_args


def get_best_trial_configs(
    experiment_name: str,
) -> tuple[ModelArguments, DataTrainingArguments, TrainingArguments]:
    # if trials_are_available_locally(experiment_name):
    #     return get_best_trial_configs_orion(experiment_name)
    return get_best_trial_configs_wandb(experiment_name)


def trials_are_available_locally(experiment_name: str, min_trials: int = 25) -> bool:
    try:
        experiment = get_experiment(experiment_name)
    except orion.core.utils.exceptions.NoConfigurationError:
        return False
    return len(experiment.fetch_trials_by_status("completed")) > min_trials


def get_best_trial_configs_orion(
    experiment_name: str,
) -> tuple[ModelArguments, DataTrainingArguments, TrainingArguments]:
    experiment = get_experiment(experiment_name)
    trials: list[Trial] = experiment.fetch_trials_by_status("completed")
    best_trial = min(
        trials, key=lambda trial: trial.objective.value if trial.objective else float("inf")
    )
    print(f"Best trial: {best_trial}")
    print(f"Best trial working directory: {best_trial.working_dir}")

    log_dir = Path(best_trial.working_dir)
    # TODO: Load the model, data, and training args from the yaml config files.

    model_args = load_yaml(ModelArguments, log_dir / "model_args.yaml")
    data_args = load_yaml(DataTrainingArguments, log_dir / "data_args.yaml")
    training_args = load_training_args(log_dir / "training_args.yaml")
    return model_args, data_args, training_args


def get_best_trial_configs_wandb(
    experiment_name: str,
) -> tuple[ModelArguments, DataTrainingArguments, TrainingArguments]:
    """Fetches the best trials found on wandb, and reads the model_args.yaml, data_args.yaml and
    training_args.yaml files into configuration objects."""
    from wandb.apis.public import File, Run

    import wandb

    log_dir = Path("runs/large")
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs: Iterable[Run] = api.runs("lebrice/mup_demo", filters={"tags": experiment_name})

    best_run: Run = min(runs, key=lambda run: float(run.summary.get("eval/loss", 1e10) or 1e10))
    print(f"Best run: {best_run}")
    model_args_file: File = best_run.file("model_args.yaml")
    data_args_file: File = best_run.file("data_args.yaml")
    training_args_file: File = best_run.file("training_args.yaml")
    file = model_args_file.download(root=str(log_dir), replace=True)
    # FIXME: debugging stuff.
    assert False, file.read()

    model_args_file.download(str(log_dir))
    data_args_file.download(str(log_dir))
    training_args_file.download(str(log_dir))

    model_args = load_yaml(ModelArguments, log_dir / "model_args.yaml")
    assert False, model_args
    data_args = load_yaml(DataTrainingArguments, log_dir / "data_args.yaml")
    training_args = load_yaml(TrainingArguments, log_dir / "training_args.yaml")
    return model_args, data_args, training_args


def main():
    experiment_name = "gpt2_256"
    model_args, data_args, training_args = get_best_trial_configs(experiment_name)

    # Overwrite the entries that we want to change:
    training_args.output_dir = "runs/gpt2_large"
    model_args.model.n_embd = 1024

    trainer = setup_trainer(
        model_args=model_args, data_args=data_args, training_args=training_args
    )

    train_metrics = train(trainer, model_args=model_args, data_args=data_args)
    eval_metrics = evaluation_loop(trainer=trainer, model_args=model_args, data_args=data_args)
    print(f"Train metrics: {train_metrics}")
    print(f"Eval metrics: {eval_metrics}")


if __name__ == "__main__":
    main()
