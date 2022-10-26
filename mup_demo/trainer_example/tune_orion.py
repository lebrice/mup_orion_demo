"""Tunes the hyper-parameters of a MuP Transformer using Orion's Python API.

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
from pathlib import Path

from orion.core.worker.trial import Trial

from mup_demo.trainer_example.train import (
    _setup_logging,
    evaluation_loop,
    parse_args,
    setup_trainer,
    train,
)
from mup_demo.utils import is_main_process, replace_fields_of, suggest_trial

logger = logging.getLogger(__name__)


def tune_using_orion():
    """Tunes the hyper-parameters of a MuP Transformer using Orion's Python API."""

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

        # Suggest a new trial.
        trial = suggest_trial(experiment)
        logger.info(f"Trial {trial.id} suggested, with params {trial.params}")
        assert trial.working_dir is not None
        assert Path(trial.working_dir).parent == sweep_dir

        # Create the configs for this run, by replacing some of the values in the original configs
        # with the ones suggested in the trial.
        trial_hparams = trial.params.copy()
        run_training_args = replace_fields_of(
            training_args, **trial_hparams, output_dir=trial.working_dir
        )
        run_model_args = replace_fields_of(model_args, **trial_hparams)
        run_data_args = replace_fields_of(data_args, **trial_hparams)
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


if __name__ == "__main__":
    tune_using_orion()
