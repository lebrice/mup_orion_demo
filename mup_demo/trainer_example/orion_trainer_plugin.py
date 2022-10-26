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
        self.metrics: list[dict[str, float]] = []

    @abstractmethod
    def suggest_trial(self) -> TrialType:
        """Suggests a new trial (set of hyper-parameters to use)."""
        raise NotImplementedError

    @abstractmethod
    def get_run_id(self, run_trainer: Trainer, trial: TrialType) -> str:
        """Returns a unique identifier to use for this run.

        This will be used to make the Trial's working directory.
        """
        raise NotImplementedError

    @property
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

    @abstractmethod
    def save_state(self, sweep_dir: Path) -> list[TrialType]:
        """Saves the state of this plugin in the given directory."""
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, sweep_dir: Path) -> None:
        """Restores the state of the Plugin from the given directory."""

    def run_hpo_sweep(
        self, base_trainer: Trainer, hp_space: dict[str, Any], n_trials: int
    ) -> BestRun:
        """Runs the hyper-parameter optimization procedure."""
        hp_space = hp_space or self.default_hp_space()
        sweep_dir = Path(base_trainer.args.output_dir)

        if sweep_dir.exists():
            self.load_state_dict(sweep_dir)

        while not self.is_done:
            trial = self.suggest_trial()
            logger.info(f"New Trial: {trial}")

            self.on_before_run_start(trial)

            run_trainer = self.get_trainer_for_run(base_trainer, trial=trial)

            # NOTE: Instead of adding a new hook, we could also just assume that the run_trainer
            # has a different output_dir for the run.
            if run_trainer.args.output_dir == base_trainer.args.output_dir:
                raise RuntimeError(
                    "The output_dir for the Trainer used in a run of HPO should be different than "
                    "the output_dir of the base trainer, which is used as the base output dir for "
                    "the runs of the HPO sweep. "
                    f"This HPO plugin {self} should set a different "
                    "value for `run_trainer.args.output_dir`."
                )

            run_output_dir = Path(run_trainer.args.output_dir)
            run_id = run_output_dir.name

            # Another alternative, which would require an extra hook to identify each run:
            # run_id = self.get_run_id(run_trainer, trial=trial)
            # run_output_dir = Path(sweep_dir / run_id)

            run_last_checkpoint = (
                get_last_checkpoint(run_output_dir) if run_output_dir.exists() else None
            )

            self.on_before_run_start(trial)

            try:
                train_metrics = run_trainer.train(resume_from_checkpoint=run_last_checkpoint)
                eval_metrics = base_trainer.evaluate()
                # TODO: Might want to save some stuff here using the Trainer?

                self.report_results(
                    trial=trial,
                    run_trainer=run_trainer,
                    train_metrics=train_metrics,
                    eval_metrics=eval_metrics,
                )
                self.trials.append(trial)
                self.metrics.append(eval_metrics)
                self.training_args_list.append(deepcopy(run_trainer.args))

            except Exception as error:
                self.report_error(
                    trial=trial,
                    run_trainer=run_trainer,
                    error=error,
                )
                raise error

            self.on_run_end(run_trainer=run_trainer, trial=trial, metrics=eval_metrics)

        objectives = [self.compute_objective(metrics) for metrics in self.metrics]
        best_objective = min(objectives) if self.minimize else max(objectives)
        best_index = objectives.index(best_objective)

        best_trial = self.trials[best_index]
        best_training_args = self.training_args_list[best_index]
        best_metrics = self.metrics[best_index]

        best_run_id = Path(best_training_args.output_dir).name
        best_hyperparameters_dict = self.get_trial_hparam_dict(best_trial)

        best_run = BestRun(
            run_id=best_run_id, objective=best_objective, hyperparameters=best_hyperparameters_dict
        )
        return best_run

    def get_trainer_for_run(self, base_trainer: Trainer, trial: TrialType) -> Trainer:
        """Called at the start of a new run, to adapt the trainer with the information of the
        Trial.

        NOTE: This doesn't necessarily have to create a new Trainer, it can also modify the
        trainer in-place and return it.
        """
        # Not quite sure how to do this correctly.
        raise NotImplementedError
        new_training_args = self.update_training_args(trial, base_trainer.args)
        run_trainer = base_trainer
        run_trainer.args = new_training_args
        return base_trainer

    @classmethod
    def install_command(cls) -> str:
        return "pip install " + " ".join(cls.requirements)

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

    def on_before_run_start(self, trial: TrialType):
        """Called at the start of a new run."""

    def on_run_start(self, run_trainer: Trainer, trial: TrialType):
        """Called at the start of a new run."""

    def on_run_end(self, run_trainer: Trainer, trial: TrialType, metrics: dict[str, float]):
        """Called at the start of a new run."""


class OrionHPSearchPlugin(HPSearchPlugin):

    name: ClassVar[str] = "orion"
    url: ClassVar[str] = "https://www.github.com/epistimio/orion"
    requirements: ClassVar[list[str]] = ["orion"]

    def __init__(
        self,
        compute_objective: Callable[[dict[str, float]], float] = default_compute_objective,
        minimize: bool = True,
        **experiment_kwargs,
    ) -> None:
        super().__init__(compute_objective, minimize)
        self.experiment_kwargs = experiment_kwargs
        self.experiment: ExperimentClient | None = None

    def default_hp_space(self) -> dict[str, str]:
        return default_hp_space_orion()

    def run_hpo_sweep(self, base_trainer: Trainer, hpo_space: dict[str, Any], n_trials: int):
        """Run the HPO search."""
        hpo_space = hpo_space or self.default_hp_space()
        sweep_dir = Path(base_trainer.args.output_dir)
        # TODO: Create the experiment using the `experiment_kwargs` passed to the constructor.

        experiment_kwargs = self.experiment_kwargs.copy()
        experiment_kwargs.setdefault("algorithms", {"tpe": {"seed": 42}})
        experiment_kwargs.update(
            max_trials=n_trials,
            working_dir=str(sweep_dir),
            # TODO: Figure out where/how to create a "sweep dir" in this context.
            storage={
                "type": "legacy",
                "database": {"type": "pickleddb", "host": str(sweep_dir / "db.pkl")},
            },
        )

        self.experiment = build_experiment(
            name="mup_demo",
            space=hpo_space,
            **self.experiment_kwargs,
        )

        while not self.experiment.is_done:
            # NOTE: This only happens on the main process, and all worker processes receive the
            # same new Trial as a result.
            trial: Trial = suggest_trial(self.experiment)
            logger.info(f"Trial params: {trial.params}")
            training_args = self.update_training_args(trial.params, base_trainer.args)
            training_args.output_dir = trial.working_dir
            training_args.logging_dir = trial.working_dir

            self.on_before_run_start(trial.params)

            print(training_args)

            # TODO: Perhaps the Trainer should be re-created here. It would make sense to have this
            # `sweep` function be one level of abstraction higher than the Trainer object, IMO,
            # instead of modyfing the Trainer object in-place.
            base_trainer.args = training_args

            try:
                checkpoint: str | None = None
                if Path(trial.working_dir).exists():
                    checkpoint = get_last_checkpoint(trial.working_dir)
                base_trainer.train(resume_from_checkpoint=checkpoint)
                metrics = base_trainer.evaluate()

            except RuntimeError as err:
                if "CUDA out of memory" in str(err):
                    self.report_bad_results(
                        trial=trial,
                        trainer=base_trainer,
                        step=base_trainer.state.global_step,
                        exception=err,
                    )
                    # TODO: Clear the CUDA state, etc?
                else:
                    raise
            else:
                self.report_results(
                    trial=trial,
                    step=base_trainer.state.global_step,
                    metrics=metrics,
                    trainer=base_trainer,
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
        assert self.experiment
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
        assert self.experiment
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


HPSearchPluginType = TypeVar("HPSearchPluginType", bound=HPSearchPlugin)
P = ParamSpec("P")


class OrionTrainer(Trainer):
    def hyperparameter_search(
        self,
        hp_space: dict[str, Any] | None = None,
        hpsearch_plugin: HPSearchPlugin | None = None,
        n_trials: int = 20,
        compute_objective: Callable[[dict[str, float]], float] = default_compute_objective,
        minimize: bool = True,
    ) -> BestRun:
        hp_space = hp_space or hpsearch_plugin.default_hp_space()
        if hpsearch_plugin is None:
            if orion_is_available():
                from mup_demo.trainer_example.orion_trainer_plugin import (
                    OrionHPSearchPlugin,
                )

                hpsearch_plugin = OrionHPSearchPlugin(
                    compute_objective=compute_objective,
                    minimize=minimize,
                    # orion experiment kwargs:
                    space=hp_space,
                )
        assert hpsearch_plugin is not None

        hpsearch_plugin = hpsearch_plugin

        best_run = hpsearch_plugin.run_hpo_sweep(
            base_trainer=self, hp_space=hp_space, n_trials=n_trials
        )

        return best_run
