"""SGLang-specific ATOM weight loading helpers."""

from __future__ import annotations

import re

from torch import nn
from transformers import AutoConfig

from atom.model_loader.loader import (
    WeightsMapper,
    load_model,
    load_model_in_plugin_mode,
)


def _packed_mapping_key_matches_weight_name(weight_name: str, key: str) -> bool:
    """Match a packed-mapping key as a full path segment."""
    if key.startswith(".") or key.endswith("."):
        return key in weight_name
    return (
        re.search(r"(?:^|\.)" + re.escape(key) + r"(?:\.|$)", weight_name) is not None
    )


def load_model_in_sglang_plugin_mode(
    model,
    config,
    prefix: str = "",
    weights_mapper: WeightsMapper | None = None,
    load_fused_expert_weights_fn=None,
) -> set[str]:
    return load_model_in_plugin_mode(
        model=model,
        config=config,
        prefix=prefix,
        weights_mapper=weights_mapper,
        load_fused_expert_weights_fn=load_fused_expert_weights_fn,
        packed_mapping_key_matcher=_packed_mapping_key_matches_weight_name,
    )


def load_model_for_sglang_plugin(
    model: nn.Module,
    model_name_or_path: str,
    hf_config: AutoConfig,
    load_dummy: bool = False,
    spec_decode: bool = False,
    prefix: str = "",
    is_plugin_mode: bool = False,
    weights_mapper: WeightsMapper | None = None,
    load_fused_expert_weights_fn=None,
):
    return load_model(
        model=model,
        model_name_or_path=model_name_or_path,
        hf_config=hf_config,
        load_dummy=load_dummy,
        spec_decode=spec_decode,
        prefix=prefix,
        is_plugin_mode=is_plugin_mode,
        weights_mapper=weights_mapper,
        load_fused_expert_weights_fn=load_fused_expert_weights_fn,
        packed_mapping_key_matcher=_packed_mapping_key_matches_weight_name,
    )
