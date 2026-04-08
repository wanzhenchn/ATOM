# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""SGLang prepare hooks: native-aiter env default (Qwen3.5 / Qwen3-next only) and
the ``register_ops_to_sglang`` env gate (PR #355). Qwen modules are stubbed in
``sys.modules``; register gate mirrors ``atom.plugin.register`` without importing it."""

from __future__ import annotations

import os
import sys
import types
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest

from atom.plugin.sglang.utils import prepare_hooks
from atom.plugin.sglang.utils.prepare_hooks import run_sglang_prepare_hooks
from atom.utils import envs

_ENV_KEY = "ATOM_SGLANG_USE_NATIVE_AITER_ATTN_BACKEND"
_MISSING = object()


@contextmanager
def _stub_sglang_qwen_prepare_modules():
    q35 = types.ModuleType("atom.plugin.sglang.models.qwen3_5")
    q35.apply_prepare_model_adaptations = MagicMock()
    qn = types.ModuleType("atom.plugin.sglang.models.qwen3_next")
    qn.apply_qwen3_next_sglang_model_patch = MagicMock()
    names = (
        "atom.plugin.sglang.models.qwen3_5",
        "atom.plugin.sglang.models.qwen3_next",
    )
    saved = {n: sys.modules.get(n, _MISSING) for n in names}
    sys.modules["atom.plugin.sglang.models.qwen3_5"] = q35
    sys.modules["atom.plugin.sglang.models.qwen3_next"] = qn
    try:
        yield
    finally:
        for n in names:
            prev = saved[n]
            if prev is _MISSING:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = prev


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
    "model_arch",
    ("Qwen3_5ForCausalLM", "Qwen3NextForCausalLM"),
)
def test_prepare_hooks_setdefault_native_aiter_for_qwen(model_arch: str):
    with _stub_sglang_qwen_prepare_modules():
        run_sglang_prepare_hooks(model_arch, MagicMock())
    assert os.environ.get(_ENV_KEY) == "1"


@pytest.mark.parametrize(
    "model_arch",
    ("DeepseekV3ForCausalLM", "Qwen3ForCausalLM"),
)
def test_prepare_hooks_no_setdefault_for_non_qwen_oot(model_arch: str):
    run_sglang_prepare_hooks(model_arch, MagicMock())
    assert os.getenv(_ENV_KEY) is None


@pytest.mark.parametrize(
    "preset,expect_false_via_envs",
    (("0", True), ("1", False)),
)
def test_prepare_hooks_setdefault_respects_existing_env(
    monkeypatch: pytest.MonkeyPatch, preset: str, expect_false_via_envs: bool
):
    monkeypatch.setenv(_ENV_KEY, preset)
    with _stub_sglang_qwen_prepare_modules():
        run_sglang_prepare_hooks("Qwen3_5ForCausalLM", MagicMock())
    assert os.environ[_ENV_KEY] == preset
    assert envs.ATOM_SGLANG_USE_NATIVE_AITER_ATTN_BACKEND is (not expect_false_via_envs)


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
