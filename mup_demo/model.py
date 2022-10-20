from __future__ import annotations
from dataclasses import dataclass
from mutransformers import BertConfig, BertForSequenceClassification
from mup import make_base_shapes, set_base_shapes
from dataclasses import field
import functools
from simple_parsing.helpers.hparams.hyperparameters import HyperParameters
from simple_parsing.helpers.hparams.hparam import log_uniform
from dataclasses import dataclass
from typing import TypeVar
from accelerate import init_empty_weights


from mutransformers import BertConfig, BertForSequenceClassification
from mup import make_base_shapes, set_base_shapes
from mutransformers import PretrainedConfig, PreTrainedModel, BertPreTrainedModel


ConfigType = TypeVar("ConfigType", bound=PretrainedConfig)


@dataclass
class HParams(HyperParameters):
    learning_rate: float = log_uniform(1e-6, 1e-3, default=0.00005)
    # batch_size: int = log_uniform(4, 128, default=32, base=2, discrete=True)
    batch_size: int = 128
    num_epochs: int = 3
    random_seed: int = 42

    model: BertConfig = field(
        default_factory=functools.partial(
            BertConfig,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_labels=5,  # TODO: This is specific to this particular dataset.
        )
    )


def _replace(model_config: ConfigType, **kwargs) -> ConfigType:
    delta_config = model_config.to_dict()
    delta_config.update(**kwargs)
    return type(model_config).from_dict(delta_config)


M = TypeVar("M", bound=PreTrainedModel)
BertModelType = TypeVar("BertModelType", bound=BertPreTrainedModel)


class ScalableBertModel(BertForSequenceClassification):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        base_config = _replace(
            config,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
        )
        delta_config = _replace(
            config,
            hidden_size=200,
            intermediate_size=300,
            num_attention_heads=5,
        )
        with init_empty_weights():
            base_model = type(self)(base_config)
            delta_model = type(self)(delta_config)
        base_shapes = make_base_shapes(base_model, delta_model, savefile="bert256.bsh")
        # set base shapes
        set_base_shapes(self, base_shapes)
        # re-initialize
        self.apply(self._init_weights)

        print(f"Total parameters in the base model:   {base_model.num_parameters()}")
        print(f"Total parameters in the delta model:  {delta_model.num_parameters()}")
        print(f"Total parameters in the target model: {self.num_parameters()}")


def get_bert_model(
    config: BertConfig, model_class: type[BertModelType] = BertForSequenceClassification
) -> BertModelType:
    base_config = _replace(
        config,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
    )
    delta_config = _replace(
        config,
        hidden_size=200,
        intermediate_size=300,
        num_attention_heads=5,
    )
    with init_empty_weights():
        base_model = model_class(base_config)
        delta_model = model_class(delta_config)

    # define a base model
    with init_empty_weights():
        base_model = model_class(base_config)
        delta_model = model_class(delta_config)
    target_model = model_class(config=config)

    base_shapes = make_base_shapes(base_model, delta_model, savefile="bert256.bsh")
    # set base shapes
    set_base_shapes(target_model, base_shapes)
    # re-initialize
    target_model.apply(target_model._init_weights)
    print(f"Total parameters in the base model:   {base_model.num_parameters()}")
    print(f"Total parameters in the delta model:  {delta_model.num_parameters()}")
    print(f"Total parameters in the target model: {target_model.num_parameters()}")
    return target_model
