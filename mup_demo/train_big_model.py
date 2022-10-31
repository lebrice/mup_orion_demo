from __future__ import annotations

from pathlib import Path

from orion.client import get_experiment
from orion.core.worker.trial import Trial

from mup_demo.train import (
    DataTrainingArguments,
    ModelArguments,
    TrainingArguments,
    evaluation_loop,
    setup_trainer,
    train,
)


def main():
    experiment_name = "gpt2_256"
    experiment = get_experiment(experiment_name)
    trials: list[Trial] = experiment.fetch_trials_by_status("completed")
    best_trial = min(
        trials, key=lambda trial: trial.objective.value if trial.objective else float("inf")
    )
    print(f"Best trial: {best_trial}")
    print(f"Best trial working directory: {best_trial.working_dir}")

    log_dir = Path(best_trial.working_dir)
    # TODO: Load the model, data, and training args from the yaml config files.
    from simple_parsing.helpers.serialization.serializable import load_yaml

    model_args = load_yaml(ModelArguments, log_dir / "model_args.yaml")
    data_args = load_yaml(DataTrainingArguments, log_dir / "data_args.yaml")
    training_args = load_yaml(TrainingArguments, log_dir / "training_args.yaml")

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
