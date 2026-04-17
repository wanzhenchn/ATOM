# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""SGLang prepare hooks: native-aiter env default (Qwen3.5 / Qwen3-next only) and
the ``register_ops_to_sglang`` env gate (PR #355). Qwen modules are stubbed in
``sys.modules``; register gate mirrors ``atom.plugin.register`` without importing it."""

from __future__ import annotations

import os
import sys
import types
from contextlib import contextmanager, nullcontext
from typing import Any
from unittest.mock import MagicMock

import pytest

from atom.plugin.sglang.utils import prepare_hooks
from atom.plugin.sglang.utils.prepare_hooks import run_sglang_prepare_hooks
from atom.utils import envs

_ENV_KEY = "ATOM_SGLANG_USE_NATIVE_AITER_ATTN_BACKEND"


@contextmanager
def _stub_qwen35_prepare_module():
    q35 = types.ModuleType("atom.plugin.sglang.models.qwen3_5")
    q35.apply_prepare_model_adaptations = MagicMock()
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "atom.plugin.sglang.models.qwen3_5", q35)
        yield q35.apply_prepare_model_adaptations


def _register_ops_gate(_atom_config: Any, register_custom: MagicMock) -> None:
    """Same branch as ``register_ops_to_sglang`` in ``atom.plugin.register``."""
    if envs.ATOM_SGLANG_USE_NATIVE_AITER_ATTN_BACKEND:
        return
    register_custom()


@pytest.fixture(autouse=True)
def _clear_native_aiter_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(_ENV_KEY, raising=False)
    yield
    monkeypatch.delenv(_ENV_KEY, raising=False)


@pytest.mark.parametrize(
    "model_arch,preset,expected_env,expected_native,expect_prepare_call",
    (
        ("Qwen3_5ForCausalLM", None, "1", True, True),
        ("Qwen3NextForCausalLM", None, "1", True, False),
        ("DeepseekV3ForCausalLM", None, None, False, False),
        ("Qwen3ForCausalLM", None, None, False, False),
        ("Qwen3_5ForCausalLM", "0", "0", False, True),
        ("Qwen3_5ForCausalLM", "1", "1", True, True),
    ),
)
def test_prepare_hooks_native_aiter_env_behavior(
    monkeypatch: pytest.MonkeyPatch,
    model_arch: str,
    preset: str | None,
    expected_env: str | None,
    expected_native: bool,
    expect_prepare_call: bool,
):
    if preset is None:
        monkeypatch.delenv(_ENV_KEY, raising=False)
    else:
        monkeypatch.setenv(_ENV_KEY, preset)

    ctx = (
        _stub_qwen35_prepare_module()
        if model_arch.startswith("Qwen3_5")
        else nullcontext()
    )
    with ctx as prepare_mock:
        run_sglang_prepare_hooks(model_arch, MagicMock())

    assert os.getenv(_ENV_KEY) == expected_env
    assert envs.ATOM_SGLANG_USE_NATIVE_AITER_ATTN_BACKEND is expected_native
    if expect_prepare_call:
        prepare_mock.assert_called_once()
    else:
        assert prepare_mock is None


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
    reg = MagicMock()
    _register_ops_gate(MagicMock(), reg)
    if expect_custom_registered:
        reg.assert_called_once()
    else:
        reg.assert_not_called()


def test_register_sglang_prepare_hook_no_duplicate_entries():
    calls: list[str] = []

    def hook_a(_m: str, _c: object) -> None:
        calls.append("a")

    def hook_b(_m: str, _c: object) -> None:
        calls.append("b")

    n0 = len(prepare_hooks._SGLANG_PREPARE_HOOKS)
    prepare_hooks.register_sglang_prepare_hook(hook_a)
    prepare_hooks.register_sglang_prepare_hook(hook_b)
    prepare_hooks.register_sglang_prepare_hook(hook_a)
    assert len(prepare_hooks._SGLANG_PREPARE_HOOKS) == n0 + 2
    try:
        run_sglang_prepare_hooks("DeepseekV3ForCausalLM", MagicMock())
        assert calls == ["a", "b"]
    finally:
        prepare_hooks._SGLANG_PREPARE_HOOKS.remove(hook_a)
        prepare_hooks._SGLANG_PREPARE_HOOKS.remove(hook_b)
        assert len(prepare_hooks._SGLANG_PREPARE_HOOKS) == n0
