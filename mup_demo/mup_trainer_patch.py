from __future__ import annotations

from typing import Callable

import torch.optim
import transformers.optimization
from torch.optim import Optimizer
from transformers.trainer import Trainer, TrainingArguments
from transformers.utils.logging import get_logger

import mup

logger = get_logger(__name__)


def patch_trainer_for_mup(Trainer=Trainer):
    """Patches the `get_optimizer_cls_and_kwargs` staticmethod of the Trainer class to return the
    MuP Variants of the SGD, Adam, or AdamW optimizers."""

    _get_optimizer_cls_and_kwargs = Trainer.get_optimizer_cls_and_kwargs

    def get_optimizer_cls_and_kwargs(
        args: TrainingArguments,
    ) -> tuple[Callable[..., Optimizer], dict]:
        # Use the base class to get the optimizer class and kwargs.
        optimizer_cls, kwargs = _get_optimizer_cls_and_kwargs(args)
        if optimizer_cls in [mup.MuAdam, mup.MuAdamW, mup.MuSGD]:
            pass  # do nothing, it's already a MuP optimizer.
        elif optimizer_cls is torch.optim.Adam:
            optimizer_cls = mup.MuAdam
        elif optimizer_cls in [torch.optim.AdamW, transformers.optimization.AdamW]:
            optimizer_cls = mup.MuAdamW
        elif optimizer_cls is torch.optim.SGD:
            optimizer_cls = mup.MuSGD
        else:
            raise NotImplementedError(
                f"To use the MuP Trainer plugin, the optimizer must be one of Adam, AdamW, or "
                f"SGD. Got {optimizer_cls} (from args.optim={args.optim})."
            )
        logger.info(f"Using MuP optimizer: {optimizer_cls} with kwargs: {kwargs}")
        return optimizer_cls, kwargs

    Trainer.get_optimizer_cls_and_kwargs = get_optimizer_cls_and_kwargs
