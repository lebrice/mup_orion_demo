from __future__ import annotations

import logging

import torch.optim
from torch import nn
from torch.optim import Optimizer
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    ShardedDDPOption,
    Trainer,
    TrainingArguments,
    get_parameter_names,
    is_sagemaker_mp_enabled,
)

import mup

logger = logging.getLogger(__name__)


class MupTrainerPlugin(Trainer):
    @classmethod
    def get_optimizer_cls_and_kwargs(cls, args: TrainingArguments) -> tuple[type[Optimizer], dict]:
        optimizer_cls, kwargs = super().get_optimizer_cls_and_kwargs(args)

        if optimizer_cls is torch.optim.Adam:
            optimizer_cls = mup.MuAdam
        elif optimizer_cls is torch.optim.AdamW:
            optimizer_cls = mup.MuAdamW
        elif optimizer_cls is torch.optim.SGD:
            optimizer_cls = mup.MuSGD
        else:
            raise NotImplementedError(
                f"To use the MuP Trainer plugin, the optimizer must be one of Adam, AdamW, or "
                f"SGD. Got {optimizer_cls}"
            )
        print(f"Using MuP optimizer: {optimizer_cls} with kwargs: {kwargs}")
        return optimizer_cls, kwargs

    def create_optimizer(self):
        # NOTE: Copying all of this just to be certain that it uses the MuAdamW optimizer.

        self.hp_search_backend
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if n in decay_parameters
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if n not in decay_parameters
                    ],
                    "weight_decay": 0.0,
                },
            ]

            # NEW CODE: #
            # NOTE: Change this here to use `self.get_optimizer_cls_and_kwargs`
            # optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            optimizer_cls, optimizer_kwargs = type(self).get_optimizer_cls_and_kwargs(self.args)
            # END NEW CODE #

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                from transformers.trainer import OSS  # type: ignore

                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes  # type: ignore

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")

        if is_sagemaker_mp_enabled():
            from transformers.trainer import smp  # type: ignore

            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
