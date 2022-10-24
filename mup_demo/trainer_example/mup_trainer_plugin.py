from __future__ import annotations
from transformers.trainer import Trainer

import logging

import mutransformers
import transformers
from torch import nn
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    Trainer,
)
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    ShardedDDPOption,
    get_parameter_names,
    is_sagemaker_mp_enabled,
)
from transformers.utils.versions import require_version
import mutransformers

# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.24.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# TODO: Could try to add the mup-variants in these lists here?


MODEL_CONFIG_CLASSES[
    MODEL_CONFIG_CLASSES.index(transformers.GPT2Config)
] = mutransformers.GPT2Config
CONFIG_MAPPING["gpt2"] = mutransformers.GPT2Config
CONFIG_MAPPING["bert"] = mutransformers.BertConfig
CONFIG_MAPPING["roberta"] = mutransformers.RobertaConfig


class MupTrainerPlugin(Trainer):
    def create_optimizer(self):
        # NOTE: Copying all of this just to be certain that it uses the MuAdamW optimizer.
        from mup import MuAdamW

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

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            # NEW CODE: #
            optimizer_cls = MuAdamW
            print(f"Using MuP optimizer: {optimizer_cls}")
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
