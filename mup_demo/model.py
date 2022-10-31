from __future__ import annotations

import contextlib
from functools import partial
from typing import TypeVar

from accelerate import init_empty_weights
from mutransformers import (
    BertConfig,
    BertPreTrainedModel,
    GPT2Config,
    GPT2LMHeadModel,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.utils import logging

from mup import make_base_shapes, set_base_shapes

logger = logging.get_logger(__name__)

M = TypeVar("M", bound=PreTrainedModel)

ConfigType = TypeVar("ConfigType", bound=PretrainedConfig)
BertModelType = TypeVar("BertModelType", bound=BertPreTrainedModel)
GPT2ModelType = TypeVar("GPT2ModelType", bound=GPT2LMHeadModel)


def _replace(model_config: ConfigType, **kwargs) -> ConfigType:
    delta_config = model_config.to_dict()
    delta_config.update(**kwargs)
    # NOTE: would be nice to avoid all the logging that goes on when calling `from_dict`..
    # return type(model_config)(**delta_config)
    with temp_change_verbosity(logging.ERROR):
        config = type(model_config).from_dict(delta_config)
    return config


@contextlib.contextmanager
def temp_change_verbosity(verbosity: int = logging.ERROR):
    verbosity = logging.get_verbosity()
    logging.set_verbosity(logging.ERROR)
    yield
    logging.set_verbosity(verbosity)


def get_bert_model(config: BertConfig, model_type: type[BertModelType]) -> BertModelType:
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
    # Create the base and delta models using empty weights, so they don't take any memory.
    with init_empty_weights():
        base_model = model_type(base_config)
        delta_model = model_type(delta_config)

    target_model = model_type(config=config)

    base_shapes = make_base_shapes(base_model, delta_model, savefile="bert256.bsh")
    # set base shapes
    set_base_shapes(target_model, base_shapes)
    # re-initialize
    target_model.apply(target_model._init_weights)
    print(f"Total parameters in the base model:   {base_model.num_parameters()}")
    print(f"Total parameters in the delta model:  {delta_model.num_parameters()}")
    print(f"Total parameters in the target model: {target_model.num_parameters()}")
    return target_model


def get_gpt2_model(
    config: GPT2Config,
    model_type: type[GPT2ModelType],
    readout_zero_init=False,
    query_zero_init=False,
) -> GPT2ModelType:
    assert isinstance(config, GPT2Config)
    base_config = _replace(
        config,
        n_head=4,
        # activation_function="relu",
        n_embd=256,
        # n_layer=2,
    )
    delta_config = _replace(
        config,
        n_head=5,
        # activation_function="relu",
        n_embd=200,
        # n_layer=2,
    )
    # with init_empty_weights():
    base_model = model_type(config=base_config)
    delta_model = model_type(config=delta_config)

    filename = "gpt2.bsh"
    base_shapes = make_base_shapes(base_model, delta_model, savefile=filename)

    model = model_type(config=config)
    set_base_shapes(model, base_shapes)
    model.apply(
        partial(
            model._init_weights,
            readout_zero_init=readout_zero_init,
            query_zero_init=query_zero_init,
        )
    )
    logger.info(f"Total parameters in the base model:   {base_model.num_parameters()}")
    logger.info(f"Total parameters in the delta model:  {delta_model.num_parameters()}")
    logger.info(f"Total parameters in the target model: {model.num_parameters()}")

    return model
