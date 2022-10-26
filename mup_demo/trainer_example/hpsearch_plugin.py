from __future__ import annotations

import dataclasses
import importlib.util
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, ClassVar, Generic, Literal, TypeVar

from transformers import TrainingArguments
from transformers.integrations import hp_params, logger
from transformers.trainer import BestRun, Trainer, get_last_checkpoint
from transformers.trainer_utils import default_compute_objective

HPSearchBackend = Literal["optuna", "ray", "sigopt", "wandb", "orion"]


def default_hp_space_orion() -> dict:
    """Return the default HPO space to use."""
    return {
        "learning_rate": "loguniform(1e-6, 1e-4)",
        "num_train_epochs": "fidelity(1, 5)",
        "seed": "uniform(1, 40, discrete=True)",
        "per_device_train_batch_size": "choices([4, 8, 16, 32, 64])",
    }


def orion_is_available() -> bool:
    return importlib.util.find_spec("orion") is not None


TrialType = TypeVar("TrialType")


class HPSearchPlugin(Generic[TrialType], ABC):
    """A base class for a hyper-parameter optimization plugin for HuggingFace.

    There are several issues with the Hyper-Parameter Optimization (HPO) of the `Trainer` class.

    `Trainer.hyperparameter_search` method:

    1. Difficulty of adding new frameworks:
       Using globally accessible dictionaries to store callbacks for each framework is nice.
       However, the code actually relies on a mix of such global dictionaries, as well as
       inaccessible local dictionaries inside various functions, hard-coded if cascades, and enums.
       This makes it inconvenient to add support for other HPO libraries.

    2.  Messy implementation with unnecessary coupling:
        The main body of the HPO routine is contained in a function for each framework, which is
        stored in a global dictionary. This is fine so far. *However*, the issue is that these
        functions, in many cases, rely on additional, framework-specific methods of the Trainer
        class, which appear to have been added as "patches" to the Trainer class. There is therefore
        a lot of bad coupling between the Trainer class and these functions.

    2.  No clear separation of concerns:
        The entire HPO routine operates as a method on the Trainer class. The Trainer, which is
        normally used to perform a single training run, is now used to perform multiple runs in
        succession, using the same Trainer instance.

        This is problematic for a few reasons, one of which is that there is a possibility of one
        training run affecting another.

        One potential solution to this, in my mind, would be to perform HPO at one level of
        abstraction above the Trainer API. For instance, by passing some sort of callable which
        should create the Trainer, given a set of hyper-parameters.

        On the other hand, reusing some of the Trainer's components between runs is more efficient.
        There is probably a tradeoff to be made here, where each HPO "plugin" can decide what to
        share and what to recreate for each run.

    The idea is that each framework could implement and encapsulate its own logic for doing HPO
    inside a subclass of this `HPSearchPlugin` class. This plugin would then either be passed to
    a Trainer method (e.g. a `Trainer.hyperparameter_search`-like method, or (preferably IMO)
    used directly to run the sweep, via `HPSearchPlugin.tune_hyperparameters()`.

    Each HPO framework only needs to implement the abstract methods of the `HPSearchPlugin` class.
    """

    name: ClassVar[str]
    url: ClassVar[str]
    requirements: ClassVar[list[str]] = []

    def __init__(
        self,
        compute_objective: Callable[[dict[str, float]], float] = default_compute_objective,
        minimize: bool = True,
    ) -> None:
        super().__init__()
        self.compute_objective = compute_objective
        self.minimize = minimize

        self.training_args_list: list[TrainingArguments] = []
        self.trials: list[TrialType] = []
        self.train_metrics: list[dict[str, float]] = []
        self.eval_metrics: list[dict[str, float]] = []

    @abstractmethod
    def suggest_trial(self) -> TrialType:
        """Suggests a new trial (set of hyper-parameters to use)."""
        raise NotImplementedError

    @abstractmethod
    def is_done(self) -> bool:
        """Returns True if the HPO search is done."""
        raise NotImplementedError

    @abstractmethod
    def default_hp_space(self) -> dict[str, Any]:
        """Returns the default Hyper-Parameter optimization space to use.

        Should be a dictionary where the keys correspond to the TrainingArguments fields. The
        values can be anything, depending on the HPO framework.
        """
        raise NotImplementedError

    @abstractmethod
    def get_trainer_for_run(self, base_trainer: Trainer, trial: TrialType) -> Trainer:
        """Called at the start of a new run, to adapt the trainer with the information of the
        Trial.

        NOTE: This doesn't necessarily have to create a new Trainer, it can also modify the
        trainer in-place and return it.
        """
        # Not quite sure how to do this correctly.
        raise NotImplementedError

    @abstractmethod
    def report_results(
        self,
        trial: TrialType,
        run_trainer: Trainer,
        train_metrics: dict[str, float],
        eval_metrics: dict[str, float],
    ):
        """Report the results of this Trial to the HPO framework."""
        raise NotImplementedError

    def get_trial_hparam_dict(self, trial: TrialType) -> dict[str, Any]:
        """Returns a dictionary containing the hyper-parameters of the Trial."""
        return hp_params(trial)

    def report_error(
        self,
        trial: TrialType,
        run_trainer: Trainer,
        error: Exception,
    ):
        """Report that this run encountered an error to the HPO framework, if applicable.

        This hook is optional, and does nothing by default.
        """

    def save_state(self, sweep_dir: Path) -> list[TrialType]:
        """Saves the state of this plugin in the given directory."""
        raise NotImplementedError

    def load_state_dict(self, sweep_dir: Path) -> None:
        """Restores the state of the Plugin from the given directory."""

    def run_hpo_sweep(
        self,
        base_trainer: Trainer,
        hp_space: dict[str, Any] | None,
        n_trials: int,
    ) -> BestRun:
        """Runs the hyper-parameter optimization procedure."""
        sweep_dir = Path(base_trainer.args.output_dir)

        hp_space = hp_space or self.default_hp_space()
        self.on_sweep_setup(
            base_trainer=base_trainer, hp_space=hp_space, n_trials=n_trials, sweep_dir=sweep_dir
        )

        if sweep_dir.exists():
            self.load_state_dict(sweep_dir)

        while not self.is_done():
            trial = self.suggest_trial()
            logger.info(f"New Trial: {trial}")

            self.on_before_run_start(base_trainer=base_trainer, trial=trial)

            run_trainer = self.get_trainer_for_run(base_trainer, trial=trial)

            # NOTE: Instead of adding a new hook, we could also just assume that the run_trainer
            # has a different output_dir for the run.
            if (
                run_trainer is not base_trainer
                and run_trainer.args.output_dir == base_trainer.args.output_dir
            ):
                raise RuntimeError(
                    "The output_dir for the Trainer used in a run of HPO should be different than "
                    "the output_dir of the base trainer, which is used as the base output dir for "
                    "the runs of the HPO sweep.\n"
                    f"The current HPO plugin ({self}) should set a different "
                    f"value for `run_trainer.args.output_dir` than {run_trainer.args.output_dir}."
                )

            run_output_dir = Path(run_trainer.args.output_dir)
            run_id = run_output_dir.name

            # NOTE: An alternative, which would require an extra hook to identify each run:
            # run_id = self.get_run_id(run_trainer, trial=trial)
            # run_output_dir = Path(sweep_dir / run_id)

            run_last_checkpoint = (
                get_last_checkpoint(run_output_dir) if run_output_dir.exists() else None
            )

            try:
                train_output = run_trainer.train(resume_from_checkpoint=run_last_checkpoint)
                train_metrics = train_output.metrics
                eval_metrics = base_trainer.evaluate()
                # TODO: Might want to save some stuff here using the Trainer?
                self.trials.append(trial)
                self.train_metrics.append(train_metrics)
                self.eval_metrics.append(eval_metrics)
                self.training_args_list.append(deepcopy(run_trainer.args))

                self.report_results(
                    trial=trial,
                    run_trainer=run_trainer,
                    train_metrics=train_metrics,
                    eval_metrics=eval_metrics,
                )

            except Exception as error:
                self.report_error(
                    trial=trial,
                    run_trainer=run_trainer,
                    error=error,
                )
                raise

            self.on_run_end(run_trainer=run_trainer, trial=trial, metrics=eval_metrics)

        objectives = [self.compute_objective(metrics) for metrics in self.eval_metrics]
        best_objective = min(objectives) if self.minimize else max(objectives)
        best_index = objectives.index(best_objective)

        best_trial = self.trials[best_index]
        best_training_args = self.training_args_list[best_index]
        best_metrics = self.eval_metrics[best_index]

        best_run_id = Path(best_training_args.output_dir).name
        best_hyperparameters_dict = self.get_trial_hparam_dict(best_trial)

        best_run = BestRun(
            run_id=best_run_id, objective=best_objective, hyperparameters=best_hyperparameters_dict
        )
        return best_run

    def on_sweep_setup(
        self, base_trainer: Trainer, hp_space: dict[str, Any], sweep_dir: Path, n_trials: int
    ) -> None:
        """Called at the start of the sweep."""

    @classmethod
    def install_command(cls) -> str:
        return "pip install " + " ".join(cls.requirements)

    def _update_training_args(
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

    def on_before_run_start(self, base_trainer: Trainer, trial: TrialType):
        """Called at the start of a new run."""

    def on_run_start(self, run_trainer: Trainer, trial: TrialType):
        """Called at the start of a new run."""

    def on_run_end(self, run_trainer: Trainer, trial: TrialType, metrics: dict[str, float]):
        """Called at the start of a new run."""


class OrionTrainer(Trainer):
    def hyperparameter_search(
        self,
        hp_space: dict[str, Any] | None = None,
        minimize: bool = True,
        hpsearch_plugin: HPSearchPlugin | None = None,
        n_trials: int = 20,
    ) -> BestRun:
        if hpsearch_plugin is None:
            if orion_is_available():
                from mup_demo.trainer_example.orion_hpsearch_plugin import (
                    OrionHPSearchPlugin,
                )

                hpsearch_plugin = OrionHPSearchPlugin(
                    minimize=minimize,
                    name="tmp_sweep",
                    space=hp_space,
                    # orion experiment kwargs:
                )
        assert hpsearch_plugin is not None

        hpsearch_plugin = hpsearch_plugin

        best_run = hpsearch_plugin.run_hpo_sweep(
            base_trainer=self, hp_space=hp_space, n_trials=n_trials
        )

        return best_run


class NewHPSearchAPIMixin(Trainer):
    def hyperparameter_search(
        self,
        hp_space: dict[str, Any] | None = None,
        minimize: bool = True,
        hpsearch_plugin: HPSearchPlugin | None = None,
        n_trials: int = 20,
    ) -> BestRun:
        if hpsearch_plugin is None:
            if orion_is_available():
                from mup_demo.trainer_example.orion_hpsearch_plugin import (
                    OrionHPSearchPlugin,
                )

                hpsearch_plugin = OrionHPSearchPlugin(
                    minimize=minimize,
                    name="tmp_sweep",
                    space=hp_space,
                    # NOTE: orion experiment kwargs could be passed here.
                )
        assert hpsearch_plugin is not None

        hpsearch_plugin = hpsearch_plugin

        best_run = hpsearch_plugin.run_hpo_sweep(
            base_trainer=self, hp_space=hp_space, n_trials=n_trials
        )

        return best_run
