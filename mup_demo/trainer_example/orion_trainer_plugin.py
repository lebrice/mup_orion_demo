from __future__ import annotations

import dataclasses
import importlib.util
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, ClassVar, Generic, Literal, TypeVar

from orion.client import ExperimentClient, build_experiment
from orion.core.worker.trial import Trial
from transformers import TrainingArguments
from transformers.integrations import hp_params, logger
from transformers.trainer import BestRun, Trainer, get_last_checkpoint
from transformers.trainer_utils import default_compute_objective
from typing_extensions import ParamSpec

from mup_demo.utils import suggest_trial

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
    """IDEA: Move the HPO logic into a 'plugin' class, to fix the weird mess in HuggingFace's
    `Trainer` class.

    There's a weird mix of functions (see above) and ugly backend-specific methods on the Trainer
    class, which really don't make sense.
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
        hp_space = hp_space or self.default_hp_space()
        sweep_dir = Path(base_trainer.args.output_dir)

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

    @abstractmethod
    def get_trainer_for_run(self, base_trainer: Trainer, trial: TrialType) -> Trainer:
        """Called at the start of a new run, to adapt the trainer with the information of the
        Trial.

        NOTE: This doesn't necessarily have to create a new Trainer, it can also modify the
        trainer in-place and return it.
        """
        # Not quite sure how to do this correctly.
        raise NotImplementedError

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


P = ParamSpec("P")


class OrionHPSearchPlugin(HPSearchPlugin[Trial]):

    name: ClassVar[str] = "orion"
    url: ClassVar[str] = "https://www.github.com/epistimio/orion"
    requirements: ClassVar[list[str]] = ["orion"]

    def __init__(
        self,
        compute_objective: Callable[[dict[str, float]], float] = default_compute_objective,
        minimize: bool = True,
        _experiment_function: Callable[P, ExperimentClient] = build_experiment,
        *experiment_args: P.args,
        **experiment_kwargs: P.kwargs,
    ) -> None:
        super().__init__(compute_objective, minimize)
        self._experiment_function = _experiment_function
        self.experiment_args = experiment_args
        self.experiment_kwargs = experiment_kwargs
        self.experiment: ExperimentClient | None = None

    def default_hp_space(self) -> dict[str, str]:
        return default_hp_space_orion()

    def is_done(self) -> bool:
        return self.experiment and self.experiment.is_done

    def on_sweep_setup(
        self, base_trainer: Trainer, hp_space: dict[str, Any], sweep_dir: Path, n_trials: int
    ) -> None:
        super().on_sweep_setup(base_trainer, hp_space, sweep_dir, n_trials)

        # TODO: Distinction between n_trials and max_trials? (as in, n_trials in addition to
        # the current number of completed trials in the sweep, vs n_trials in total?)
        self.experiment_kwargs["space"] = hp_space
        self.experiment_kwargs.setdefault("max_trials", n_trials)
        self.experiment_kwargs.setdefault("working_dir", str(sweep_dir))
        self.experiment_kwargs.setdefault(
            "storage",
            {
                "type": "legacy",
                "database": {"type": "pickleddb", "host": str(sweep_dir / "db.pkl")},
            },
        )
        self.experiment = self._experiment_function(
            *self.experiment_args,
            **self.experiment_kwargs,
        )

    def suggest_trial(self) -> Trial:
        assert self.experiment is not None
        trial: Trial = suggest_trial(self.experiment)
        logger.info(f"Trial params: {trial.params}")
        return trial

    def get_trainer_for_run(self, base_trainer: Trainer, trial: Trial) -> Trainer:
        run_training_args = self._update_training_args(trial.params, base_trainer.args)

        # Important: Change some values that aren't hyper-parameters, but that need to be set
        # differently for each run.
        run_training_args.output_dir = trial.working_dir
        run_training_args.logging_dir = trial.working_dir

        # TODO: Make 100% sure that this is okay. Ideally, we'd create a new Trainer for each run.
        run_trainer = base_trainer
        run_trainer.args = run_training_args
        return run_trainer

    def report_results(
        self,
        trial: Trial,
        run_trainer: Trainer,
        train_metrics: dict[str, float],
        eval_metrics: dict[str, float],
    ):
        """Report the results of this Trial to the HPO framework."""
        assert self.experiment
        objective = self.compute_objective(eval_metrics)

        if run_trainer.args.process_index == 0:
            print(f"Reporting objective of {objective} for trial {trial}")
            results = [dict(name="objective", value=objective, type="objective")]
            results += [
                dict(name=name, value=value, type="statistic")
                for metrics_dict in [train_metrics, eval_metrics]
                for name, value in metrics_dict.items()
            ]
            self.experiment.observe(trial, results)

    def report_error(self, trial: Trial, run_trainer: Trainer, error: Exception):
        # NOTE: Could report a bad objective to Orion, but that's not a priority here.
        pass


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
                from mup_demo.trainer_example.orion_trainer_plugin import (
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
