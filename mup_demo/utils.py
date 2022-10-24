import contextlib
import functools
import os
from typing import Callable, TypeVar, cast

from orion.client import ExperimentClient
from orion.client.experiment import TrialCM
from orion.core.worker.trial import Trial
from typing_extensions import ParamSpec


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


OutputType = TypeVar("OutputType")
P = ParamSpec("P")


def runs_on_main_process(function: Callable[P, OutputType]) -> Callable[P, OutputType]:
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

    import torch.distributed as dist

    @functools.wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs):

        if dist.get_rank() == 0:
            # Assumes world_size of 3.
            output = function(*args, **kwargs)  # any picklable object
            output_buffer = [output]
        else:
            output_buffer = [None]
        dist.broadcast_object_list(output_buffer, src=0)
        outputs = cast(list[OutputType], output_buffer)
        output = outputs[0]
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
        import torch

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
