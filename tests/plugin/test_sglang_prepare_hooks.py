# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for current SGLang prepare/register env gating behavior."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from atom.plugin import prepare as plugin_prepare
from atom.utils import envs

_ENV_KEY = "ATOM_SGLANG_USE_NATIVE_AITER_ATTN_BACKEND"


class _Obj:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _register_ops_gate(register_custom: MagicMock) -> None:
    """Mirror the env gate inside ``atom.plugin.register.register_ops_to_sglang``."""
    if envs.ATOM_SGLANG_USE_NATIVE_AITER_ATTN_BACKEND:
        return
    register_custom()


def _make_fake_register_module(model_arch: str):
    fake_model = MagicMock()
    fake_model.atom_config = None
    fake_model_cls = MagicMock(return_value=fake_model)
    fake_register = MagicMock()
    fake_register._ATOM_SUPPORTED_MODELS = {model_arch: fake_model_cls}
    fake_register.register_ops_to_sglang = MagicMock()
    fake_register.init_aiter_dist = MagicMock()
    fake_register.set_attn_cls = MagicMock()
    return fake_register, fake_model, fake_model_cls


@pytest.fixture(autouse=True)
def _reset_framework_state():
    plugin_prepare._set_framework_backbone("atom")
    yield
    plugin_prepare._set_framework_backbone("atom")


@pytest.fixture(autouse=True)
def _clear_native_aiter_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(_ENV_KEY, raising=False)
    yield
    monkeypatch.delenv(_ENV_KEY, raising=False)


@pytest.mark.parametrize(
    "model_arch,preset,expected_env",
    (
        ("Qwen3_5ForConditionalGeneration", None, "1"),
        ("Qwen3NextForCausalLM", None, "1"),
        ("DeepseekV3ForCausalLM", None, None),
        ("Qwen3MoeForCausalLM", None, None),
        ("Qwen3_5ForConditionalGeneration", "0", "0"),
        ("Qwen3_5ForConditionalGeneration", "1", "1"),
    ),
)
def test_prepare_model_native_aiter_env_behavior(
    monkeypatch: pytest.MonkeyPatch,
    model_arch: str,
    preset: str | None,
    expected_env: str | None,
):
    if preset is None:
        monkeypatch.delenv(_ENV_KEY, raising=False)
    else:
        monkeypatch.setenv(_ENV_KEY, preset)

    fake_atom_config = _Obj(plugin_config=_Obj(is_plugin_mode=True))
    fake_register, _fake_model, fake_model_cls = _make_fake_register_module(model_arch)
    fake_config_mod = MagicMock()
    fake_config_mod.generate_atom_config_for_plugin_mode = MagicMock(
        return_value=fake_atom_config
    )

    with patch.dict(
        sys.modules,
        {
            "atom.plugin.register": fake_register,
            "atom.plugin.config": fake_config_mod,
            "atom.plugin.sglang.graph_capture_patch": MagicMock(
                apply_graph_capture_patch=MagicMock()
            ),
        },
    ):
        plugin_prepare.prepare_model(
            config=_Obj(architectures=[model_arch]),
            engine="sglang",
        )

    assert os.getenv(_ENV_KEY) == expected_env
    assert envs.ATOM_SGLANG_USE_NATIVE_AITER_ATTN_BACKEND is (
        expected_env in {"1", "true", "yes"}
    )
    fake_register.register_ops_to_sglang.assert_called_once_with(
        atom_config=fake_atom_config
    )
    fake_model_cls.assert_called_once()


@pytest.mark.parametrize(
    "env_setup,expect_custom_registered",
    (
        ("native", False),
        ("unset", True),
        ("zero", True),
    ),
)
def test_register_ops_env_gate(
    monkeypatch: pytest.MonkeyPatch, env_setup: str, expect_custom_registered: bool
):
    if env_setup == "native":
        monkeypatch.setenv(_ENV_KEY, "1")
    elif env_setup == "unset":
        monkeypatch.delenv(_ENV_KEY, raising=False)
    else:
        monkeypatch.setenv(_ENV_KEY, "0")
    register_custom = MagicMock()
    _register_ops_gate(register_custom)

    if expect_custom_registered:
        register_custom.assert_called_once()
    else:
        register_custom.assert_not_called()
