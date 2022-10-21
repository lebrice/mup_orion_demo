import functools
from typing import TypeVar
from orion.client import ExperimentClient
import os
from orion.core.worker.trial import Trial
from accelerate import Accelerator
from orion.client.experiment import TrialCM
from pathlib import Path
import pickle
import torch
import contextlib
from typing import Callable
from typing_extensions import ParamSpec
import tempfile


def suggest_trial_manual(experiment: ExperimentClient) -> Trial:
    if not in_ddp_context():
        # Not in a DDP context.
        # todo: this isn't typed.
        trial = experiment.suggest()
        # TODO: https://github.com/Epistimio/orion/issues/1009
        assert isinstance(trial, (TrialCM, Trial))
        if isinstance(trial, TrialCM):
            return trial._cm_trial
        return trial

    local_rank = int(os.environ["RANK"])
    master_addr = os.environ["MASTER_ADDR"]
    master_port = int(os.environ["MASTER_PORT"])
    filename = Path(experiment.working_dir) / f"_trial_{master_addr}_{master_port}.pkl"
    # print(f"WORKER {local_rank}: Filename: {filename}")

    with main_process_first():
        if not filename.exists() and local_rank == 0:
            # Main worker!
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
    wait_for_everyone()

    if is_main_process:
        if filename.exists():
            filename.unlink()

    return trial


def suggest_trial_using_accelerator(
    experiment: ExperimentClient, accelerator: Accelerator
) -> Trial:
    local_rank = get_local_rank()
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
        if not filename.exists() and local_rank == 0:
            # Main worker!
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
    # accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        if filename.exists():
            filename.unlink()

    return trial


def in_ddp_context() -> bool:
    return (
        "RANK" in os.environ
        and os.environ["RANK"].isdigit()
        and "MASTER_ADDR" in os.environ
        and "MASTER_PORT" in os.environ
        and os.environ["MASTER_PORT"].isdigit()
        and "WORLD_SIZE" in os.environ
        and os.environ["WORLD_SIZE"].isdigit()
    )


O = TypeVar("O")
P = ParamSpec("P")


def runs_on_main_process(function: Callable[P, O]) -> Callable[P, O]:
    """Makes the decorated function or method run only on the main process when in a DDP context.

    The results are then pickled to other processes, to make it look as if all processes ran the
    function.

    TODO: Incorporate a hash of the parameters in the shared filename?
    TODO: What should we do if the process group isn't initialized yet, but the environment
    variables are present? (Can the process group be un-initialized even when the environment
    variables are present?)
    """
    if not in_ddp_context():
        # Not in a DDP context: Just return the output of the function call.
        return function

    @functools.wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs):
        local_rank = int(os.environ["RANK"])
        master_addr = os.environ["MASTER_ADDR"]
        master_port = int(os.environ["MASTER_PORT"])
        filename = Path(f"_{function.__qualname__}_{master_addr}_{master_port}.pkl")
        with main_process_first():
            output: O
            if not filename.exists() and local_rank == 0:
                # Main worker!
                output = function(*args, **kwargs)
                with open(filename, "wb") as f:
                    pickle.dump(output, f)
            else:
                with open(filename, "rb") as f:
                    output = pickle.load(f)

        # Wait for everyone before removing the file.
        wait_for_everyone()

        if is_main_process():
            if filename.exists():
                filename.unlink()

        return output

    return wrapper


@runs_on_main_process
def suggest_trial(experiment: ExperimentClient) -> Trial:
    trial = experiment.suggest()
    if isinstance(trial, TrialCM):
        return trial._cm_trial
    return trial


def wait_for_everyone():
    if not in_ddp_context():
        return
    try:
        torch.distributed.barrier()
    except ImportError:
        pass
    except RuntimeError:
        # We can get here if the process group isn't initialized yet.
        try:
            import accelerate

            accelerator = accelerate.Accelerator()
            accelerator.wait_for_everyone()
        except ImportError:
            pass


def get_local_rank() -> int:
    return int(os.environ["RANK"])


def is_main_process() -> bool:
    return get_local_rank() in (-1, 0)


@contextlib.contextmanager
def main_process_first():
    yield from _goes_first(is_main=is_main_process())


def _goes_first(is_main: bool):
    if not is_main:
        wait_for_everyone()
    yield
    if is_main:
        wait_for_everyone()
