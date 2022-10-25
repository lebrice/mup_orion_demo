from __future__ import annotations

import dataclasses
import importlib.util
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, ClassVar

from orion.client import ExperimentClient, build_experiment
from orion.core.worker.trial import Trial
from orion.executor.single_backend import SingleExecutor
from transformers import TrainingArguments
from transformers.integrations import logger
from transformers.trainer import (
    BestRun,
    Trainer,
    default_hp_search_backend,
    get_last_checkpoint,
)
from transformers.trainer_utils import ExplicitEnum

from mup_demo.utils import suggest_trial


class HPSearchBackend(ExplicitEnum):
    OPTUNA = "optuna"
    RAY = "ray"
    SIGOPT = "sigopt"
    WANDB = "wandb"
    ORION = "orion"


def default_hp_space_orion() -> dict:
    """Return the default HPO space to use."""
    return {
        "learning_rate": "loguniform(1e-6, 1e-4)",
        "num_train_epochs": "fidelity(1, 5)",
        "seed": "uniform(1, 40, discrete=True)",
        "per_device_train_batch_size": "choices([4, 8, 16, 32, 64])",
    }


def is_orion_available() -> bool:
    return importlib.util.find_spec("orion") is not None


class HPSearchPlugin(ABC):
    """IDEA: Move the HPO logic into a 'plugin' class, to fix the weird mess in HuggingFace's
    `Trainer` class.

    There's a weird mix of functions (see above) and ugly backend-specific methods on the Trainer
    class, which really don't make sense.
    """

    name: ClassVar[str]
    url: ClassVar[str]
    requirements: ClassVar[list[str]] = []

    @classmethod
    def install_command(cls) -> str:
        return "pip install " + " ".join(cls.requirements)

    @abstractmethod
    def default_hpo_space(self) -> dict[str, Any]:
        """Returns the default Hyper-Parameter optimization space to use.

        Should be a dictionary where the keys correspond to the TrainingArguments fields. The
        values can be anything, depending on the HPO framework.
        """

    @abstractmethod
    def report_results(
        self,
        trial: dict[str, Any],
        step: int,
        metrics: dict[str, float],
    ):
        """Report the results of this Trial to the HPO framework."""

    @abstractmethod
    def run_hpo_sweep(self, trainer: Trainer, hpo_space: dict[str, Any], n_trials: int) -> BestRun:
        """Run the HPO search."""
        raise NotImplementedError

    def update_training_args(
        self, trial_params: dict[str, Any], training_args: TrainingArguments
    ) -> TrainingArguments:
        """Called at the start of a new run, so the Trainer can be updated using the values in the
        Trial."""
        for key, value in trial_params.items():
            if not hasattr(training_args, key):
                raise RuntimeError(
                    f"Trying to set attribute {key} to value of {value} in the hyperparameter "
                    f"search, but there is no corresponding field in `TrainingArguments`!"
                )
        return dataclasses.replace(training_args, **trial_params)

    def setup_before_run(self, trial: dict[str, Any]):
        """Called at the start of a new run."""


class OrionHPSearchPlugin(HPSearchPlugin):

    name: ClassVar[str] = "orion"
    url: ClassVar[str] = "https://www.github.com/epistimio/orion"
    requirements: ClassVar[list[str]] = ["orion"]

    def __init__(self, **experiment_kwargs):
        self.experiment_kwargs = experiment_kwargs
        self.experiment: ExperimentClient

    def default_hpo_space(self) -> dict[str, str]:
        return default_hp_space_orion()

    def run_hpo_sweep(self, trainer: Trainer, hpo_space: dict[str, Any], n_trials: int):
        """Run the HPO search."""
        hpo_space = hpo_space or self.default_hpo_space()
        sweep_dir = Path(trainer.args.output_dir)
        self.experiment = build_experiment(
            name="mup_demo",
            space=hpo_space,
            algorithms={"tpe": {"seed": 42}},
            max_trials=n_trials,
            executor=SingleExecutor(n_workers=1),
            working_dir=str(sweep_dir),
            # TODO: Figure out where/how to create a "sweep dir" in this context.
            storage={
                "type": "legacy",
                "database": {"type": "pickleddb", "host": str(sweep_dir / "db.pkl")},
            },
            **self.experiment_kwargs,
        )

        while not self.experiment.is_done:
            # NOTE: This only happens on the main process, and all worker processes receive the
            # same new Trial as a result.
            trial: Trial = suggest_trial(self.experiment)
            logger.info(f"Trial params: {trial.params}")
            training_args = self.update_training_args(trial.params, trainer.args)
            training_args.output_dir = trial.working_dir
            training_args.logging_dir = trial.working_dir

            print(training_args)

            # TODO: Perhaps the Trainer should be re-created here. It would make sense to have this
            # `sweep` function be one level of abstraction higher than the Trainer object, IMO,
            # instead of modyfing the Trainer object in-place.
            trainer.args = training_args

            try:
                checkpoint: str | None = None
                if Path(trial.working_dir).exists():
                    checkpoint = get_last_checkpoint(trial.working_dir)
                trainer.train(resume_from_checkpoint=checkpoint)
                metrics = trainer.evaluate()

            except RuntimeError as err:
                if "CUDA out of memory" in str(err):
                    self.report_bad_results(
                        trial=trial,
                        trainer=trainer,
                        step=trainer.state.global_step,
                        exception=err,
                    )
                    # TODO: Clear the CUDA state, etc?
                else:
                    raise
            else:
                self.report_results(
                    trial=trial,
                    step=trainer.state.global_step,
                    metrics=metrics,
                    trainer=trainer,
                )

        completed_trials: list[Trial] = self.experiment.fetch_trials_by_status("completed")
        best_trial = min(completed_trials, key=lambda trial: trial.objective.value)
        best_objective = best_trial.objective.value

        best_run = BestRun(
            run_id=best_trial.id, objective=best_objective, hyperparameters=best_trial.params
        )
        return best_run

    def report_results(
        self,
        trainer: Trainer,
        trial: Trial,
        step: int,
        metrics: dict[str, float],
    ):
        """Report the results of this Trial to the HPO framework."""

        # objective = trainer.compute_objective(metrics)
        assert "eval_loss" in metrics, metrics.keys()
        objective = metrics["eval_loss"]
        if trainer.args.process_index == 0:
            print(f"Reporting objective of {objective} for trial {trial}")
            results = [
                dict(name="eval_loss", value=objective, type="objective"),
                dict(name="step", value=step, type="statistic"),
            ]
            results += [
                dict(name=name, value=value, type="statistic")
                for name, value in metrics.items()
                if name != "eval_loss"
            ]
            self.experiment.observe(trial, results)

    def report_bad_results(
        self,
        trainer: Trainer,
        trial: Trial,
        step: int,
        exception: Exception,
    ):
        """Report the results of this Trial to the HPO framework."""
        # objective = trainer.compute_objective(metrics)
        if trainer.args.process_index == 0:
            print(f"Reporting bad trial {trial}")
            objective = 10_000_000
            if trainer.args.process_index == 0:
                print(f"Reporting fake objective of {objective} for trial {trial}")
                results = [
                    dict(name="eval_loss", value=objective, type="objective"),
                    dict(name="step", value=step, type="statistic"),
                ]
                self.experiment.observe(trial, results)


class OrionTrainer(Trainer):
    def hyperparameter_search(
        self,
        hp_space: dict[str, Any] | None = None,
        compute_objective: Callable[[dict[str, float]], float] | None = None,
        n_trials: int = 20,
        direction: str = "minimize",
        backend: str | HPSearchBackend | None = None,
    ) -> BestRun:

        # TODO: Replace their ugly, confused API below with a clean plugin system.

        if backend is None:
            backend = default_hp_search_backend()
        available_backends = HPSearchPlugin.__subclasses__()
        if backend is None:
            names = [b.name for b in available_backends]
            installation_instructions = [b.install_command() for b in available_backends]
            options_str = ", ".join(names[:-1]) + " or " + names[-1]
            raise RuntimeError(
                f"At least one of {options_str} should be installed. "
                + "\n".join(
                    f"To install {name} run `{install_command}`. "
                    for name, install_command in zip(names, installation_instructions)
                )
            )
        selected_hpo_plugins = [
            plugin_class for plugin_class in available_backends if plugin_class.name == backend
        ]
        if len(selected_hpo_plugins) == 0:
            raise RuntimeError(f"The backend you selected ({backend}) is not available.")
        elif len(selected_hpo_plugins) > 1:
            raise RuntimeError(
                f"Found multiple backends with the same name ({backend}): {selected_hpo_plugins}."
            )

        selected_hpo_plugin = selected_hpo_plugins[0]

        plugin = selected_hpo_plugin()
        hp_space = hp_space or plugin.default_hpo_space()

        best_run = plugin.run_hpo_sweep(
            trainer=self,
            hpo_space=hp_space,
            n_trials=n_trials,
        )

        return best_run
