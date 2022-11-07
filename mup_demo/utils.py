from __future__ import annotations

import contextlib
import functools
import os
import typing
from dataclasses import fields, replace
from logging import getLogger as get_logger
from pathlib import Path
from typing import Callable, TypeVar, cast

import yaml
from orion.client import ExperimentClient
from orion.client.experiment import TrialCM
from orion.core.worker.trial import Trial
from typing_extensions import ParamSpec

if typing.TYPE_CHECKING:
    from mup_demo.train import TrainingArguments
logger = get_logger(__name__)


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


ConfigType = TypeVar("ConfigType")


def replace_fields_of(obj: ConfigType, **kwargs) -> ConfigType:
    """uses items from `kwargs` to replace matching fields of `obj`.

    Returns the new object.
    """
    overlapping_fields = {f.name for f in fields(obj)}.intersection(kwargs.keys())
    if overlapping_fields:
        logger.info(
            f"Replacing the following values in the {type(obj).__name__}: with values from the "
            f"Trial:\n" + str({k: v for k, v in kwargs.items() if k in overlapping_fields})
        )
    things_to_overwrite = {k: v for k, v in kwargs.items() if k in overlapping_fields}
    return replace(obj, **things_to_overwrite)


def save_yaml(config, path: Path) -> None:
    with open(path, "w") as f:
        yaml.dump(config, f)


def load_yaml(path: Path) -> None:
    with open(path) as f:
        return yaml.safe_load(f)


def load_training_args(path: Path) -> TrainingArguments:
    from mup_demo.train import TrainingArguments

    with open(path) as f:
        dict_or_object = yaml.safe_load(f)
        if isinstance(dict_or_object, dict):
            dict_or_object.pop("_n_gpu", "")
            for strategy_field in [
                "evaluation_strategy",
                "logging_strategy",
                "save_strategy",
                "hub_strategy",
                "lr_scheduler_type",
                "optim",
            ]:
                if strategy_field not in dict_or_object:
                    continue
                v = dict_or_object[strategy_field]
                if isinstance(v, str):
                    dict_or_object[strategy_field] = v.lower()

            training_args = TrainingArguments(**dict_or_object)
        else:
            assert isinstance(dict_or_object, TrainingArguments)
            training_args = dict_or_object
    return training_args
