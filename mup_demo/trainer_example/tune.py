""" TODO: Tuning portion.


Example command:
```bash
accelerate launch mup_demo/trainer_example/tune.py \
    --model_name_or_path gpt2 --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 --per_device_eval_batch_size 8 \
    --ddp_find_unused_parameters=False --do_train --do_eval \
    --num_train_epochs=1 --max_train_samples=100 \
    --output_dir runs/tune_debug
```
"""
from __future__ import annotations

import logging
from dataclasses import fields, replace
from pathlib import Path
from typing import TypeVar

from orion.core.worker.trial import Trial
from transformers import TrainingArguments

from mup_demo.trainer_example.orion_trainer_plugin import OrionTrainer
from mup_demo.trainer_example.train import (
    DataTrainingArguments,
    ModelArguments,
    _setup_logging,
    evaluation_loop,
    parse_args,
    setup_trainer,
    train,
)
from mup_demo.utils import is_main_process, suggest_trial

logger = logging.getLogger(__name__)

# TODO: Could try to add the mup-variants in these lists here?


def tune():
    """Tune the hyper-parameters using the `hyperparameter_search` API of the HuggingFace Trainer.

    This "works", but the output directory structure is weird (might be fixable though). Also, the
    code required to make this work is quite messy.
    """
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    model_args, data_args, training_args = parse_args()

    # Setup logging
    _setup_logging(training_args)

    trainer = setup_trainer(
        model_args=model_args, data_args=data_args, training_args=training_args
    )

    trainer: OrionTrainer
    best_run = trainer.hyperparameter_search(
        n_trials=10,
        direction="minimize",
        backend="orion",
        hp_space={
            "learning_rate": "loguniform(1e-7, 1e-3)",
        },
    )
    print(best_run)


def main():

    # TODO: DO something different (and better) here, much more like the manual example, with one
    # level of abstraction above the Trainer (and re-creating the Trainer each time).

    model_args, data_args, training_args = parse_args()

    _setup_logging(training_args)

    from orion.client import build_experiment
    from orion.executor.single_backend import SingleExecutor

    # TODO: Set the `sweep_dir` to the `output_dir` that is set on the command-line.
    sweep_dir = Path(training_args.output_dir)
    # sweep_dir = Path("runs") / "trainer_sweep"
    experiment = build_experiment(
        name="mup_demo",
        space={
            "learning_rate": "loguniform(1e-7, 1e-3)",
            "seed": "uniform(0, 100, discrete=True)",
        },
        algorithms="random",
        storage={
            "type": "legacy",
            "database": {"type": "pickleddb", "host": str(sweep_dir / "db.pkl")},
        },
        working_dir=str(sweep_dir),
        executor=SingleExecutor(n_workers=1),
        max_trials=10,
    )

    while not experiment.is_done:
        trial = suggest_trial(experiment)
        logger.info(f"Trial {trial.id} suggested, with params {trial.params}")
        assert trial.working_dir is not None
        assert Path(trial.working_dir).parent == sweep_dir
        # Create the configs for this run, by replacing some of the values in the original configs
        # with the ones suggested in the trial.
        run_training_args = _replace_fields_of(
            training_args, **trial.params, output_dir=trial.working_dir
        )
        run_model_args = _replace_fields_of(model_args, **trial.params)
        run_data_args = _replace_fields_of(data_args, **trial.params)
        run_training_args.output_dir = trial.working_dir

        if is_main_process():
            logger.info(f"Run model args: {run_model_args}")
            logger.info(f"Run data args: {run_data_args}")
            logger.info(f"Run training args: {run_training_args}")

        run_trainer = setup_trainer(
            model_args=run_model_args,
            data_args=run_data_args,
            training_args=run_training_args,
        )

        train(trainer=run_trainer, model_args=run_model_args, data_args=run_data_args)
        metrics = evaluation_loop(
            trainer=run_trainer, model_args=run_model_args, data_args=run_data_args
        )

        kwargs = {
            "finetuned_from": run_model_args.model_name_or_path,
            "tasks": "text-generation",
        }
        if run_data_args.dataset_name is not None:
            kwargs["dataset_tags"] = run_data_args.dataset_name
            if run_data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = run_data_args.dataset_config_name
                kwargs[
                    "dataset"
                ] = f"{run_data_args.dataset_name} {run_data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = run_data_args.dataset_name

        if run_training_args.push_to_hub:
            # NOTE: We don't really want to push to the hub when we're just doing HPO!
            run_trainer.push_to_hub(**kwargs)
        else:
            run_trainer.create_model_card(**kwargs)

        print(f"Trial {trial.id} finished with metrics: {metrics}")

        results = [dict(name="eval_loss", value=metrics["eval_loss"], type="objective")]
        if is_main_process():
            experiment.observe(trial, results)

    if is_main_process():
        completed_trials: list[Trial] = experiment.fetch_trials_by_status("completed")
        best_trial = min(completed_trials, key=lambda trial: trial.objective.value)
        print(f"Best trial: {best_trial.id} with objective: {best_trial.objective.value}")
        print(f"Best trial params: {best_trial.params}")

    # # IDEA: Make the HUGE model now!
    # # TODO: Should we make it in this script here, or in a separate step?
    # print("Best params:")
    # for trial in sorted(trials, key=lambda trial: trial.objective.value):
    #     # metrics = get_trial_metrics(trial)
    #     print(f"{trial.working_dir}:", trial.params, trial.results)


ConfigType = TypeVar("ConfigType", TrainingArguments, DataTrainingArguments, ModelArguments)


def _replace_fields_of(obj: ConfigType, **kwargs) -> ConfigType:
    overlapping_fields = {f.name for f in fields(obj)}.intersection(kwargs.keys())
    if overlapping_fields:
        logger.info(
            f"Replacing the following values in the {type(obj).__name__}: with values from the "
            f"Trial:\n" + str({k: v for k, v in kwargs.items() if k in overlapping_fields})
        )
    things_to_overwrite = {k: v for k, v in kwargs.items() if k in overlapping_fields}
    return replace(obj, **things_to_overwrite)


if __name__ == "__main__":
    # tune()
    main()
