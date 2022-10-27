from __future__ import annotations

from orion.client import get_experiment
from orion.core.worker.trial import Trial


def main():
    sweep_name = "mup_demo"
    experiment = get_experiment(sweep_name)
    trials: list[Trial] = experiment.fetch_trials_by_status("completed")
    best_trial = min(
        trials, key=lambda trial: trial.objective.value if trial.objective else float("inf")
    )
    print(f"Best trial: {best_trial}")