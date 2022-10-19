from threading import local
from orion.client import ExperimentClient
from orion.algo.base import BaseAlgorithm
import os
from orion.core.worker.trial import Trial
from accelerate import Accelerator
from orion.client.experiment import TrialCM
from pathlib import Path
import pickle
import torch
from accelerate.accelerator import AcceleratorState


def suggest_trial(experiment: ExperimentClient, accelerator: Accelerator) -> Trial:
    local_rank = int(os.environ.get("RANK", -1))
    if local_rank == -1:
        # Not in a DDP context.
        # todo: this isn't typed.
        trial = experiment.suggest()
        # TODO: https://github.com/Epistimio/orion/issues/1009
        assert isinstance(trial, (TrialCM, Trial))
        if isinstance(trial, TrialCM):
            return trial._cm_trial
        return trial

    master_addr = os.environ["MASTER_ADDR"]
    master_port = int(os.environ["MASTER_PORT"])
    filename = Path(experiment.working_dir) / f"_trial_{master_addr}_{master_port}.pkl"
    # print(f"WORKER {local_rank}: Filename: {filename}")

    with accelerator.main_process_first():
        if not filename.exists():
            # Main worker!
            assert local_rank == 0
            trial = experiment.suggest()
            assert isinstance(trial, (TrialCM, Trial))
            if isinstance(trial, TrialCM):
                trial = trial._cm_trial
            with open(filename, "wb") as f:
                pickle.dump(trial, f)
        else:
            with open(filename, "rb") as f:
                trial = pickle.load(f)
                assert isinstance(trial, Trial)

    # Wait for everyone before removing the file.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if filename.exists():
            filename.unlink()

    return trial
