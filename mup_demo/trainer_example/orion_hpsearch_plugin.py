from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable, ClassVar

from orion.client import ExperimentClient, build_experiment
from orion.core.worker.trial import Trial
from transformers.integrations import logger
from transformers.trainer import Trainer
from transformers.trainer_utils import default_compute_objective
from typing_extensions import ParamSpec

from mup_demo.trainer_example.hpsearch_plugin import (
    HPSearchPlugin,
    default_make_trainer_for_run,
)
from mup_demo.utils import suggest_trial

P = ParamSpec("P")


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


class OrionHPSearchPlugin(HPSearchPlugin[Trial]):
    """A plugin to be passed to Trainer.hyperparameter_search() to use Orion for HPO."""

    name: ClassVar[str] = "orion"
    url: ClassVar[str] = "https://www.github.com/epistimio/orion"
    requirements: ClassVar[list[str]] = ["orion"]

    def __init__(
        self,
        compute_objective: Callable[[dict[str, float]], float] = default_compute_objective,
        minimize: bool = True,
        make_trainer_for_run: Callable[[Trainer, Trial], Trainer] = default_make_trainer_for_run,
        _experiment_function: Callable[P, ExperimentClient] = build_experiment,
        *experiment_args: P.args,
        **experiment_kwargs: P.kwargs,
    ) -> None:
        super().__init__(compute_objective, minimize)
        self.make_trainer_for_run = make_trainer_for_run
        if not minimize:
            raise NotImplementedError(
                "Orion only supports minimization for now. "
                "(see https://github.com/Epistimio/orion/issues/873)"
            )
        self._experiment_function = _experiment_function
        self.experiment_args = experiment_args
        self.experiment_kwargs = experiment_kwargs
        self.experiment: ExperimentClient | None = None

    def default_hp_space(self) -> dict[str, str]:
        return default_hp_space_orion()

    def is_done(self) -> bool:
        return bool(self.experiment and self.experiment.is_done)

    def on_sweep_setup(
        self, base_trainer: Trainer, hp_space: dict[str, Any], sweep_dir: Path, n_trials: int
    ) -> None:
        super().on_sweep_setup(base_trainer, hp_space, sweep_dir, n_trials)

        # TODO: Distinction between n_trials and max_trials? (as in, n_trials in addition to
        # the current number of completed trials in the sweep, vs n_trials in total?)
        self.experiment_kwargs["space"] = hp_space
        self.experiment_kwargs.setdefault("max_trials", n_trials)
        self.experiment_kwargs.setdefault("working_dir", str(sweep_dir))
        # In not passed, setup a Pickleddb database in the sweep directory.
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

    def get_trial_hparam_dict(self, trial: Trial) -> dict[str, Any]:
        return trial.params

    def get_trainer_for_run(self, base_trainer: Trainer, trial: Trial) -> Trainer:
        starting_output_dir = base_trainer.args.output_dir
        run_trainer = super().get_trainer_for_run(base_trainer, trial)
        # If the output_dir wasn't already changed by the trial hyper-parameters, then change it.
        if run_trainer.args.output_dir == starting_output_dir:
            run_trainer.args.output_dir = trial.working_dir
            # TODO: Do we also need to change this `logging_dir` ?
            run_trainer.args.logging_dir = trial.working_dir

        # TODO: Make sure that there is no state leaking from one run to the next.
        # Ideally, create a new Trainer while reusing all the expensive objects. (model, datasets,
        # etc).
        return run_trainer

    def report_results(
        self,
        trial: Trial,
        run_trainer: Trainer,
        train_metrics: dict[str, float],
        eval_metrics: dict[str, float],
    ):
        """Report the results of this Trial to Orion."""
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
