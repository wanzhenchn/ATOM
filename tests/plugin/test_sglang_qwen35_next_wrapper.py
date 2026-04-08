# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Qwen3-next ``ForCausalLM`` and Qwen3.5 model shim under mocked ``sglang`` (see ``test_sglang_model_wrapper``)."""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


@contextmanager
def _qwen35_shim_import_context():
    """Real ``atom.config`` + ``mark_spliting_op`` stub (conftest no-ops custom op register)."""
    import atom.utils as au

    def _spliting_op(*_a, **_k):
        def _d(f):
            f.spliting_op = True
            return f

        return _d

    saved = sys.modules.get("atom.config")
    sys.modules.pop("atom.config", None)
    try:
        importlib.import_module("atom.config")
        with patch.object(au, "mark_spliting_op", _spliting_op):
            yield
    finally:
        if saved is not None:
            sys.modules["atom.config"] = saved


def _atom_models_pkg_if_missing(mods: dict[str, ModuleType]) -> None:
    """``qwen3_next`` does ``import atom.models.qwen3_next``; parent ``atom.models`` must exist."""
    if "atom.models" in sys.modules:
        return
    models_dir = Path(__file__).resolve().parents[2] / "atom" / "models"
    if not models_dir.is_dir():
        return
    pkg = ModuleType("atom.models")
    pkg.__path__ = [str(models_dir)]
    mods["atom.models"] = pkg


class _Obj:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeLogitsProcessor:
    def __init__(self, config):
        self.config = config

    def __call__(self, input_ids, hidden_states, lm_head, forward_batch):
        return _Obj(
            input_ids=input_ids,
            hidden_states=hidden_states,
            lm_head=lm_head,
            forward_batch=forward_batch,
        )


def _package(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []
    return module


def _make_fake_modules(*, is_last_rank: bool) -> dict[str, ModuleType]:
    sglang_pkg = _package("sglang")
    srt_pkg = _package("sglang.srt")
    layers_pkg = _package("sglang.srt.layers")
    quant_pkg = _package("sglang.srt.layers.quantization")
    model_executor_pkg = _package("sglang.srt.model_executor")

    distributed_mod = ModuleType("sglang.srt.distributed")
    distributed_mod.get_pp_group = lambda: _Obj(is_last_rank=is_last_rank)

    logits_mod = ModuleType("sglang.srt.layers.logits_processor")
    logits_mod.LogitsProcessor = _FakeLogitsProcessor
    logits_mod.LogitsProcessorOutput = object

    quant_base_mod = ModuleType("sglang.srt.layers.quantization.base_config")
    quant_base_mod.QuantizationConfig = object

    forward_batch_mod = ModuleType("sglang.srt.model_executor.forward_batch_info")
    forward_batch_mod.ForwardBatch = object
    forward_batch_mod.PPProxyTensors = object

    attn_backend_pkg = _package("atom.plugin.sglang.attention_backend")
    mla_mod = ModuleType("atom.plugin.sglang.attention_backend.sgl_attention_mla")
    mla_mod.setup_deepseek_for_sglang = lambda model: None

    return {
        "sglang": sglang_pkg,
        "sglang.srt": srt_pkg,
        "sglang.srt.distributed": distributed_mod,
        "sglang.srt.layers": layers_pkg,
        "sglang.srt.layers.logits_processor": logits_mod,
        "sglang.srt.layers.quantization": quant_pkg,
        "sglang.srt.layers.quantization.base_config": quant_base_mod,
        "sglang.srt.model_executor": model_executor_pkg,
        "sglang.srt.model_executor.forward_batch_info": forward_batch_mod,
        "atom.plugin.sglang.attention_backend": attn_backend_pkg,
        "atom.plugin.sglang.attention_backend.sgl_attention_mla": mla_mod,
    }


@contextmanager
def _noop_gdn_bridge(_fb):
    yield


def _import_qwen3_next_module(monkeypatch, fake_model, *, is_last_rank: bool):
    import atom
    import torch
    from torch import nn

    monkeypatch.setattr(
        atom,
        "prepare_model",
        MagicMock(return_value=fake_model),
        raising=False,
    )
    mods = dict(_make_fake_modules(is_last_rank=is_last_rank))
    gdn = ModuleType("atom.plugin.sglang.utils.gdn_forward_helper")
    gdn.sglang_gdn_bridge = _noop_gdn_bridge
    core = ModuleType("atom.models.qwen3_next")

    class _Core(nn.Module):
        def forward(self, *args, **kwargs):
            return torch.tensor([0.0])

    core.Qwen3NextModel = _Core
    mods["atom.plugin.sglang.utils.gdn_forward_helper"] = gdn
    mods["atom.models.qwen3_next"] = core
    _atom_models_pkg_if_missing(mods)
    for key in (
        "atom.plugin.sglang.models.qwen3_next",
        "atom.models.qwen3_next",
        "atom.plugin.sglang.utils.gdn_forward_helper",
    ):
        sys.modules.pop(key, None)
    patcher = patch.dict(sys.modules, mods)
    patcher.start()
    try:
        mod = importlib.import_module("atom.plugin.sglang.models.qwen3_next")
    except ImportError:
        patcher.stop()
        pytest.skip(
            "atom.plugin.sglang.models.qwen3_next not importable with sglang stubs "
            "(check torch / atom.models.utils / other deps)."
        )
    return mod, patcher


def test_qwen3_next_forwards_kwargs_to_prepare_model(monkeypatch):
    fake = MagicMock(return_value="hs")
    fake.lm_head = object()
    mod, patcher = _import_qwen3_next_module(monkeypatch, fake, is_last_rank=False)
    try:
        w = mod.Qwen3NextForCausalLM(
            _Obj(vocab_size=128, architectures=["Qwen3NextForCausalLM"])
        )
        fb = _Obj(tag="fb")
        pp = _Obj(hidden_states="h", residual="r")
        out = w.forward(
            input_ids="iid",
            positions="pos",
            forward_batch=fb,
            input_embeds="emb",
            pp_proxy_tensors=pp,
            get_embedding=False,
        )
        assert out == "hs"
        fake.assert_called_once_with(
            input_ids="iid",
            positions="pos",
            intermediate_tensors=pp,
            inputs_embeds="emb",
            forward_batch=fb,
            get_embedding=False,
            pp_proxy_tensors=pp,
        )
    finally:
        patcher.stop()


def test_qwen3_next_last_rank_returns_logits_output(monkeypatch):
    fake = MagicMock(return_value="hs")
    fake.lm_head = object()
    mod, patcher = _import_qwen3_next_module(monkeypatch, fake, is_last_rank=True)
    try:
        w = mod.Qwen3NextForCausalLM(
            _Obj(vocab_size=128, architectures=["Qwen3NextForCausalLM"])
        )
        fb = _Obj(tag="fb")
        out = w.forward(input_ids="iid", positions="pos", forward_batch=fb)
        assert out.input_ids == "iid"
        assert out.hidden_states == "hs"
        assert out.lm_head is fake.lm_head
        assert out.forward_batch is fb
    finally:
        patcher.stop()


def test_qwen35_sglang_model_shim_patch_is_idempotent():
    pytest.importorskip("sglang.srt.models.qwen3_5")
    try:
        with _qwen35_shim_import_context():
            import atom.models.qwen3_5 as shim
            from atom.plugin.sglang.models import qwen3_5 as q35
            orig = shim.Qwen3_5Model
            try:
                q35._apply_qwen35_sglang_model_patch()
                assert shim.Qwen3_5Model is q35.Qwen3_5SglangModel
                q35._apply_qwen35_sglang_model_patch()
                assert shim.Qwen3_5Model is q35.Qwen3_5SglangModel
            finally:
                shim.Qwen3_5Model = orig
    except ImportError as exc:
        pytest.skip(f"qwen3_5 plugin import chain unavailable: {exc}")
