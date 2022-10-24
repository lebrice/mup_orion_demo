from __future__ import annotations
from abc import ABC, abstractmethod
import copy

import dataclasses
import importlib.util
from pathlib import Path
from tempfile import TemporaryFile
from typing import Callable, Any, ClassVar, Protocol
import functools
from transformers.trainer import get_last_checkpoint
from transformers import TrainingArguments
from mup_demo.manual_example.train import Config, training_function
from mup_demo.model import HParams
from mup_demo.utils import is_main_process, suggest_trial
from orion.client import build_experiment
from transformers.trainer import (
    BestRun,
    Trainer,
    default_compute_objective,
    default_hp_search_backend,
    is_optuna_available,
    is_ray_tune_available,
    is_sigopt_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
    run_hp_search_sigopt,
    run_hp_search_wandb,
)
from orion.executor.single_backend import SingleExecutor

from transformers.trainer_utils import (
    BestRun,
    ExplicitEnum,
    default_hp_space_optuna,
    default_hp_space_ray,
    default_hp_space_sigopt,
    default_hp_space_wandb,
)
from torch import nn
from typing import Optional
from transformers import TrainingArguments, PreTrainedModel, DataCollator
from typing_extensions import ParamSpec
from typing import Callable, TypeVar


class HPSearchBackend(ExplicitEnum):
    OPTUNA = "optuna"
    RAY = "ray"
    SIGOPT = "sigopt"
    WANDB = "wandb"
    ORION = "orion"


def default_hp_space_orion() -> dict:
    """Return the default HPO space to use."""
    # from .integrations import is_optuna_available
    return {
        "learning_rate": "loguniform(1e-6, 1e-4)",
        "num_train_epochs": "uniform(1, 5, discrete=True)",
        "seed": "uniform(1, 40, discrete=True)",
        "per_device_train_batch_size": "choices([4, 8, 16, 32, 64])",
    }


def is_orion_available() -> bool:
    return importlib.util.find_spec("orion") is not None


from transformers.integrations import (
    PREFIX_CHECKPOINT_DIR,
    ParallelMode,
    logger,
)
import os
import pickle
import torch
from dataclasses import asdict
from orion.client import build_experiment
from orion.core.worker.trial import Trial


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
        return f"pip install " + " ".join(cls.requirements)

    @abstractmethod
    def default_hpo_space(self) -> dict[str, Any]:
        """Returns the default Hyper-Parameter optimization space to use.

        Should be a dictionary where the keys correspond to the TrainingArguments fields.
        The values can be anything, depending on the HPO framework.
        """
        pass

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
        Trial.
        """
        for key, value in trial_params.items():
            if not hasattr(training_args, key):
                raise RuntimeError(
                    f"Trying to set attribute {key} to value of {value} in the hyperparameter "
                    f"search, but there is no corresponding field in `TrainingArguments`!"
                )
        return dataclasses.replace(training_args, **trial_params)

    def setup_before_run(self, trial: dict[str, Any]):
        """Called at the start of a new run."""


from orion.client import report_objective, ExperimentClient


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

        # return super().hyperparameter_search(
        #     hp_space, compute_objective, n_trials, direction, backend, hp_name, **kwargs
        # )
        if backend is None:
            backend = default_hp_search_backend()
        if backend is None:
            raise RuntimeError(
                "At least one of orion, optuna, ray, or sigopt should be installed. "
                "To install orion run `pip install orion`. "
                "To install optuna run `pip install optuna`. "
                "To install ray run `pip install ray[tune]`. "
                "To install sigopt run `pip install sigopt`."
            )
        backend = HPSearchBackend(backend)

        if backend == HPSearchBackend.ORION and not is_orion_available():
            raise RuntimeError(
                "You picked the orion backend, but it is not installed. Use `pip install orion`."
            )
        if backend == HPSearchBackend.OPTUNA and not is_optuna_available():
            raise RuntimeError(
                "You picked the optuna backend, but it is not installed. Use `pip install optuna`."
            )
        if backend == HPSearchBackend.RAY and not is_ray_tune_available():
            raise RuntimeError(
                "You picked the Ray Tune backend, but it is not installed. Use `pip install 'ray[tune]'`."
            )
        if backend == HPSearchBackend.SIGOPT and not is_sigopt_available():
            raise RuntimeError(
                "You picked the sigopt backend, but it is not installed. Use `pip install sigopt`."
            )
        if backend == HPSearchBackend.WANDB and not is_wandb_available():
            raise RuntimeError(
                "You picked the wandb backend, but it is not installed. Use `pip install wandb`."
            )
        self.hp_search_backend = backend
        if self.model_init is None:
            raise RuntimeError(
                "To use hyperparameter search, you need to pass your model through a model_init function."
            )

        self.hp_space = default_hp_space[backend] if hp_space is None else hp_space
        self.hp_name = hp_name
        self.compute_objective = (
            default_compute_objective if compute_objective is None else compute_objective
        )

        backend_dict = {
            HPSearchBackend.OPTUNA: run_hp_search_optuna,
            HPSearchBackend.RAY: run_hp_search_ray,
            HPSearchBackend.SIGOPT: run_hp_search_sigopt,
            HPSearchBackend.WANDB: run_hp_search_wandb,
            # NEW:
            HPSearchBackend.ORION: run_hp_search_orion,
        }
        best_run = backend_dict[backend](self, n_trials, direction, **kwargs)

        self.hp_search_backend = None
        return best_run

    def _hp_search_setup(self, trial: dict[str, Any]):
        """HP search setup code

        TODO: This is really really ugly code from HuggingFace. They really should have a proper
        plugin system instead of this mess.
        """
        self._trial = trial

        # TODO: For Orion, perhaps we'd like to change the output dir on the Trainer to the
        # working_dir of the Trial?

        if self.hp_search_backend is None or trial is None:
            return
        params: dict[str, Any] = {}
        if self.hp_search_backend == HPSearchBackend.OPTUNA:
            params = self.hp_space(trial)
        elif self.hp_search_backend == HPSearchBackend.RAY:
            params = trial
            params.pop("wandb", None)
        elif self.hp_search_backend == HPSearchBackend.SIGOPT:
            params = {
                k: int(v) if isinstance(v, str) else v
                for k, v in trial.assignments.items()  # type: ignore
            }
        elif self.hp_search_backend == HPSearchBackend.WANDB:
            params = trial
        # ---- ADDED -----
        elif self.hp_search_backend == HPSearchBackend.ORION:
            params = self.hp_space(trial)
        # ----------------

        for key, value in params.items():
            if not hasattr(self.args, key):
                logger.warning(
                    f"Trying to set {key} in the hyperparameter search but there is no corresponding field in"
                    " `TrainingArguments`."
                )
                continue
            old_attr = getattr(self.args, key, None)
            # Casting value to the proper type
            if old_attr is not None:
                value = type(old_attr)(value)
            setattr(self.args, key, value)

        if self.hp_search_backend == HPSearchBackend.ORION:
            logger.info(f"Trial: {trial}")
        elif self.hp_search_backend == HPSearchBackend.OPTUNA:
            logger.info(f"Trial: {trial.params}")  # type: ignore
        elif self.hp_search_backend == HPSearchBackend.SIGOPT:
            logger.info(f"SigOpt Assignments: {trial.assignments}")  # type: ignore
        elif self.hp_search_backend == HPSearchBackend.WANDB:
            logger.info(f"W&B Sweep parameters: {trial}")

        if self.args.deepspeed:
            # Rebuild the deepspeed config to reflect the updated training parameters
            from transformers.deepspeed import HfTrainerDeepSpeedConfig

            self.args.hf_deepspeed_config = HfTrainerDeepSpeedConfig(self.args.deepspeed)
            self.args.hf_deepspeed_config.trainer_config_process(self.args)

    def _report_to_hp_search(
        self,
        trial: "optuna.Trial" | dict[str, Any],
        step: int,
        metrics: dict[str, float],
    ):
        if self.hp_search_backend is None or trial is None:
            return
        self.objective = self.compute_objective(metrics.copy())
        if self.hp_search_backend == HPSearchBackend.OPTUNA:
            import optuna

            trial.report(self.objective, step)
            if trial.should_prune():
                self.callback_handler.on_train_end(self.args, self.state, self.control)
                raise optuna.TrialPruned()
        elif self.hp_search_backend == HPSearchBackend.RAY:
            from ray import tune

            if self.control.should_save:
                self._tune_save_checkpoint()
            tune.report(objective=self.objective, **metrics)
        elif self.hp_search_backend == HPSearchBackend.ORION:
            from orion.client import report_objective

            if self.control.should_save:
                self._tune_save_checkpoint()
            tune.report(objective=self.objective, **metrics)
