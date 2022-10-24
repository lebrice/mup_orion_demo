from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path
from typing import Callable, Any

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
from transformers.trainer_utils import (
    BestRun,
    ExplicitEnum,
    default_hp_space_optuna,
    default_hp_space_ray,
    default_hp_space_sigopt,
    default_hp_space_wandb,
)


class HPSearchBackend(ExplicitEnum):
    OPTUNA = "optuna"
    RAY = "ray"
    SIGOPT = "sigopt"
    WANDB = "wandb"
    ORION = "orion"


def default_hp_space_orion(trial) -> dict:
    """Return the default HPO space to use.

    TODO: Not sure what the type is of this `trial` argument.
    """
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


def run_hp_search_orion(
    trainer: OrionTrainerPlugin, n_trials: int, direction: str, **kwargs
) -> BestRun:
    """TODO: Implement this to add Orion as a source for HParam tuning."""

    # NOTE: This is the code from `run_hp_search_optuna` that I copied and modified.
    # import optuna

    if trainer.args.process_index == 0:

        def _objective(trial, checkpoint_dir=None):
            checkpoint = None
            if checkpoint_dir:
                for subdir in os.listdir(checkpoint_dir):
                    if subdir.startswith(PREFIX_CHECKPOINT_DIR):
                        checkpoint = os.path.join(checkpoint_dir, subdir)
            trainer.objective = None
            if trainer.args.world_size > 1:
                if trainer.args.parallel_mode != ParallelMode.DISTRIBUTED:
                    raise RuntimeError(
                        "only support DDP optuna HPO for ParallelMode.DISTRIBUTED currently."
                    )
                trainer._hp_search_setup(trial)
                torch.distributed.broadcast_object_list(pickle.dumps(trainer.args), src=0)
                trainer.train(resume_from_checkpoint=checkpoint)
            else:
                trainer.train(resume_from_checkpoint=checkpoint, trial=trial)
            # If there hasn't been any evaluation during the training loop.
            if getattr(trainer, "objective", None) is None:
                metrics = trainer.evaluate()
                trainer.objective = trainer.compute_objective(metrics)
            return trainer.objective

        timeout = kwargs.pop("timeout", None)
        n_jobs = kwargs.pop("n_jobs", 1)
        study = optuna.create_study(direction=direction, **kwargs)
        study.optimize(_objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)
        best_trial = study.best_trial
        return BestRun(str(best_trial.number), best_trial.value, best_trial.params)
    else:
        for i in range(n_trials):
            trainer.objective = None
            args_main_rank = list(pickle.dumps(trainer.args))
            if trainer.args.parallel_mode != ParallelMode.DISTRIBUTED:
                raise RuntimeError(
                    "only support DDP optuna HPO for ParallelMode.DISTRIBUTED currently."
                )
            torch.distributed.broadcast_object_list(args_main_rank, src=0)
            args = pickle.loads(bytes(args_main_rank))
            for key, value in asdict(args).items():
                if key != "local_rank":
                    setattr(trainer.args, key, value)
            trainer.train(resume_from_checkpoint=None)
            # If there hasn't been any evaluation during the training loop.
            if getattr(trainer, "objective", None) is None:
                metrics = trainer.evaluate()
                trainer.objective = trainer.compute_objective(metrics)
        return None


default_hp_space = {
    HPSearchBackend.ORION: default_hp_space_orion,
    HPSearchBackend.OPTUNA: default_hp_space_optuna,
    HPSearchBackend.RAY: default_hp_space_ray,
    HPSearchBackend.SIGOPT: default_hp_space_sigopt,
    HPSearchBackend.WANDB: default_hp_space_wandb,
}


class OrionTrainerPlugin(Trainer):
    def hyperparameter_search(
        self,
        hp_space: Callable[["optuna.Trial"], dict[str, float]] | None = None,
        compute_objective: Callable[[dict[str, float]], float] | None = None,
        n_trials: int = 20,
        direction: str = "minimize",
        backend: str | HPSearchBackend | None = None,
        hp_name: Callable[["optuna.Trial"], str] | None = None,
        **kwargs,
    ) -> BestRun:
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

    def _hp_search_setup(self, trial: "optuna.Trial" | dict[str, Any]):
        """HP search setup code"""
        self._trial = trial

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
