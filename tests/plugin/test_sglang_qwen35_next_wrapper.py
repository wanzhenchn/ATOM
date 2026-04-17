# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Minimal Qwen3.5 plugin regression test under mocked SGLang deps."""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager, nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

_Obj = SimpleNamespace


def _mod(name: str, *, package: bool = False, **attrs) -> ModuleType:
    mod = ModuleType(name)
    if package:
        mod.__path__ = []
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


class _WeightsMapper:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __or__(self, other):
        merged = {}
        for name in ("orig_to_new_substr", "orig_to_new_prefix", "orig_to_new_suffix"):
            merged[name] = {**getattr(self, name, {}), **getattr(other, name, {})}
        return _WeightsMapper(**merged)


@contextmanager
def _patch_qwen35_plugin():
    def pp_group():
        return _Obj(is_first_rank=True, is_last_rank=True)

    def text_config(atom_config):
        return atom_config.hf_config.text_config

    def empty_factory(*_a, **_k):
        def factory(*_x, **_y):
            return {}

        return factory

    mods = {
        **{
            name: _mod(name, package=True)
            for name in (
                "sglang",
                "sglang.srt",
                "sglang.srt.layers",
                "sglang.srt.layers.quantization",
                "sglang.srt.model_executor",
                "sglang.srt.models",
                "aiter",
                "aiter.dist",
                "atom.models",
            )
        },
        "sglang.srt.distributed": _mod(
            "sglang.srt.distributed",
            get_pp_group=pp_group,
        ),
        "sglang.srt.layers.quantization.base_config": _mod(
            "sglang.srt.layers.quantization.base_config",
            QuantizationConfig=object,
        ),
        "sglang.srt.model_executor.forward_batch_info": _mod(
            "sglang.srt.model_executor.forward_batch_info",
            ForwardBatch=object,
            PPProxyTensors=object,
        ),
        "sglang.srt.models.qwen3_5": _mod(
            "sglang.srt.models.qwen3_5",
            Qwen3_5ForConditionalGeneration=object,
            Qwen3_5MoeForConditionalGeneration=object,
        ),
        "aiter.dist.parallel_state": _mod(
            "aiter.dist.parallel_state",
            get_pp_group=pp_group,
        ),
        "atom.model_loader.loader": _mod(
            "atom.model_loader.loader",
            WeightsMapper=_WeightsMapper,
        ),
        "atom.model_ops.embed_head": _mod(
            "atom.model_ops.embed_head",
            ParallelLMHead=object,
            VocabParallelEmbedding=object,
        ),
        "atom.model_ops.moe": _mod(
            "atom.model_ops.moe",
            FusedMoE=type(
                "FusedMoE",
                (),
                {"make_expert_params_mapping": staticmethod(lambda *_a, **_k: [])},
            ),
        ),
        "atom.model_ops.topK": _mod(
            "atom.model_ops.topK",
            is_rocm_aiter_fusion_shared_expert_enabled=lambda *_a, **_k: False,
        ),
        "atom.models.qwen3_5": _mod(
            "atom.models.qwen3_5",
            Qwen3_5ForCausalLMBase=object,
            Qwen3_5DecoderLayer=object,
            Qwen3_5Model=object,
            Qwen3_5RMSNorm=object,
            get_qwen3_5_text_config=text_config,
        ),
        "atom.models.utils": _mod(
            "atom.models.utils",
            IntermediateTensors=dict,
            extract_layer_index=lambda _prefix: 0,
            make_empty_intermediate_tensors_factory=empty_factory,
            make_layers=lambda *_a, **_k: (0, 0, []),
            maybe_prefix=lambda prefix, name: f"{prefix}.{name}" if prefix else name,
        ),
        "atom.plugin.sglang.utils.loader": _mod(
            "atom.plugin.sglang.utils.loader",
            load_model_in_sglang_plugin_mode=lambda *_a, **_k: set(),
        ),
        "atom.plugin.sglang.models.base_model_wrapper": _mod(
            "atom.plugin.sglang.models.base_model_wrapper",
            build_atom_model_for_sglang=lambda *_a, **_k: None,
        ),
        "atom.plugin.sglang.models.qwen3_next": _mod(
            "atom.plugin.sglang.models.qwen3_next",
            Qwen3NextSglangAttention=object,
        ),
        "atom.plugin.sglang.utils.gdn_forward_helper": _mod(
            "atom.plugin.sglang.utils.gdn_forward_helper",
            sglang_gdn_bridge=lambda _fb: nullcontext(),
        ),
    }
    for key in (*mods, "atom.plugin.sglang.models.qwen3_5"):
        sys.modules.pop(key, None)
    with patch.dict(sys.modules, mods):
        yield importlib.import_module("atom.plugin.sglang.models.qwen3_5")


def test_qwen35_prepare_adaptations_keep_core_symbol_unchanged():
    with _patch_qwen35_plugin() as q35:
        import atom.models.qwen3_5 as shim

        orig = shim.Qwen3_5Model
        atom_config = _Obj(
            hf_config=_Obj(
                text_config=_Obj(model_type="qwen3_5_moe_text", num_experts=8)
            )
        )
        q35.apply_prepare_model_adaptations(atom_config, "Qwen3_5MoeForCausalLM")

    assert shim.Qwen3_5Model is orig
    assert atom_config.hf_config.text_config.n_shared_experts == 1
    assert atom_config.hf_config.text_config.n_routed_experts == 8
