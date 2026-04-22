# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the Qwen3.5 SGLang wrapper under mocked plugin deps."""

import importlib
import sys
from types import ModuleType
from unittest.mock import patch

import torch

from atom.plugin import prepare as plugin_prepare


class _Obj:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _WeightsMapper:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __or__(self, other):
        merged = {}
        for name in ("orig_to_new_substr", "orig_to_new_prefix", "orig_to_new_suffix"):
            merged[name] = {**getattr(self, name, {}), **getattr(other, name, {})}
        return _WeightsMapper(**merged)


class _StubFusedMoE:
    @staticmethod
    def make_expert_params_mapping(*_args, **_kwargs):
        return []


def _package(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []
    return module


def _module(name: str, **attrs) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _packages(*names: str) -> dict[str, ModuleType]:
    return {name: _package(name) for name in names}


def _object_module(name: str, *symbols: str) -> ModuleType:
    return _module(name, **{symbol: object for symbol in symbols})


def _identity(x):
    return x


def _sglang_modules(pp_group) -> dict[str, ModuleType]:
    return {
        **_packages(
            "sglang",
            "sglang.srt",
            "sglang.srt.layers",
            "sglang.srt.layers.quantization",
            "sglang.srt.model_executor",
            "sglang.srt.models",
            "aiter",
            "aiter.dist",
        ),
        "sglang.srt.distributed": _module(
            "sglang.srt.distributed",
            get_pp_group=pp_group,
        ),
        "sglang.srt.layers.quantization.base_config": _module(
            "sglang.srt.layers.quantization.base_config",
            QuantizationConfig=object,
        ),
        "sglang.srt.model_executor.forward_batch_info": _module(
            "sglang.srt.model_executor.forward_batch_info",
            ForwardBatch=object,
            PPProxyTensors=object,
        ),
        "sglang.srt.models.qwen3_5": _module(
            "sglang.srt.models.qwen3_5",
            Qwen3_5ForConditionalGeneration=object,
            Qwen3_5MoeForConditionalGeneration=object,
        ),
        "aiter.dist.parallel_state": _module(
            "aiter.dist.parallel_state",
            get_pp_group=pp_group,
        ),
    }


def _atom_modules() -> dict[str, ModuleType]:
    return {
        "atom.config": _object_module(
            "atom.config",
            "Config",
            "QuantizationConfig",
        ),
        "atom.model_loader.loader": _module(
            "atom.model_loader.loader",
            WeightsMapper=_WeightsMapper,
        ),
        "atom.model_ops.topK": _module(
            "atom.model_ops.topK",
            is_rocm_aiter_fusion_shared_expert_enabled=lambda *_a, **_k: False,
        ),
        "atom.model_ops.utils": _module(
            "atom.model_ops.utils",
            atom_parameter=_identity,
        ),
        "atom.utils.decorators": _module(
            "atom.utils.decorators",
            support_torch_compile=lambda *args, **kwargs: (lambda cls: cls),
        ),
        "atom.model_ops.embed_head": _object_module(
            "atom.model_ops.embed_head",
            "VocabParallelEmbedding",
            "ParallelLMHead",
        ),
        "atom.model_config.qwen3_5": _object_module(
            "atom.model_config.qwen3_5",
            "Qwen3_5Config",
            "Qwen3_5TextConfig",
        ),
        "atom.model_config.qwen3_5_moe": _object_module(
            "atom.model_config.qwen3_5_moe",
            "Qwen3_5MoeConfig",
            "Qwen3_5MoeTextConfig",
        ),
        "atom.model_ops.moe": _module(
            "atom.model_ops.moe",
            FusedMoE=_StubFusedMoE,
        ),
        "atom.model_ops.linear": _object_module(
            "atom.model_ops.linear",
            "MergedColumnParallelLinear",
        ),
        "atom.model_ops.layernorm": _object_module(
            "atom.model_ops.layernorm",
            "GemmaRMSNorm",
        ),
        "atom.models.qwen3_next": _object_module(
            "atom.models.qwen3_next",
            "Qwen3NextAttention",
            "Qwen3NextGatedDeltaNet",
            "Qwen3NextModel",
            "Qwen3NextSparseMoeBlock",
            "Qwen3NextMLP",
            "Qwen3NextDecoderLayer",
        ),
        "atom.models.utils": _module(
            "atom.models.utils",
            IntermediateTensors=dict,
            PPMissingLayer=object,
            make_empty_intermediate_tensors_factory=lambda *_a, **_k: (
                lambda *_x, **_y: {}
            ),
            make_layers=lambda *_a, **_k: (0, 0, []),
            maybe_prefix=lambda prefix, name: f"{prefix}.{name}" if prefix else name,
            extract_layer_index=lambda _prefix: 0,
        ),
        "atom.model_ops.split_chunk": _module(
            "atom.model_ops.split_chunk",
            fused_split_chunk_zeros=lambda *_a, **_k: None,
            fused_split_chunk_zeros_qwen3_5_qkvzba=lambda *_a, **_k: None,
        ),
        "atom.plugin.config": _module(
            "atom.plugin.config",
            generate_atom_config_for_plugin_mode=_identity,
        ),
        "atom.plugin.sglang.utils.loader": _module(
            "atom.plugin.sglang.utils.loader",
            SGLangLoaderPatch=object,
            load_model_in_sglang_plugin_mode=lambda *_a, **_k: set(),
        ),
        "atom.plugin.sglang.utils.forward_context": _object_module(
            "atom.plugin.sglang.utils.forward_context",
            "SGLangForwardBatchMetadata",
        ),
        "atom.plugin.sglang.utils.gdn_context": _object_module(
            "atom.plugin.sglang.utils.gdn_context",
            "SGLangGDNForwardContext",
        ),
    }


def _make_fake_modules() -> dict[str, ModuleType]:
    def pp_group():
        return _Obj(is_first_rank=True, is_last_rank=True)

    return {
        **_sglang_modules(pp_group),
        **_atom_modules(),
    }


def _import_qwen35_module(monkeypatch):
    fake_modules = _make_fake_modules()
    patcher = patch.dict(sys.modules, fake_modules)
    old_framework = plugin_prepare._CURRENT_FRAMEWORK

    monkeypatch.setattr(
        plugin_prepare,
        "_CURRENT_FRAMEWORK",
        "sglang",
        raising=False,
    )

    patcher.start()
    sys.modules.pop("atom.models.qwen3_5", None)
    sys.modules.pop("atom.models", None)
    module = importlib.import_module("atom.models.qwen3_5")
    return module, patcher, old_framework


def test_qwen35_bf16_mapping_uses_fused_in_proj_layout(monkeypatch):
    module, patcher, old_framework = _import_qwen35_module(monkeypatch)
    try:
        atom_config = _Obj(
            quant_config=_Obj(global_quant_config=_Obj(quant_dtype=torch.bfloat16))
        )
        remapped = module._apply_bf16_in_proj_mapping(
            dict(module._PACKED_MODULES_MAPPING), atom_config
        )
    finally:
        patcher.stop()
        sys.modules.pop("atom.models.qwen3_5", None)
        sys.modules.pop("atom.models", None)
        plugin_prepare._set_framework_backbone(old_framework)

    assert "in_proj_qkvzba" in remapped
    assert remapped["in_proj_qkv"] == ("in_proj_qkvzba", (0, 1, 2))
    assert remapped["in_proj_z"] == ("in_proj_qkvzba", 3)
    assert remapped["in_proj_b"] == ("in_proj_qkvzba", 4)
    assert remapped["in_proj_a"] == ("in_proj_qkvzba", 5)
    assert "in_proj_qkvz" not in remapped
    assert "in_proj_ba" not in remapped
