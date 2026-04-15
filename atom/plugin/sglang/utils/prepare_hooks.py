# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Prepare-time hooks for SGLang plugin mode.

``atom.plugin.prepare.prepare_model`` calls :func:`run_sglang_prepare_hooks` only.
Built-in handlers (Qwen3.5 / Qwen3-next model swaps and config tweaks) are
registered when this module is imported. For those arches only, the built-in
hook also defaults ``ATOM_SGLANG_USE_NATIVE_AITER_ATTN_BACKEND`` so
``register_ops_to_sglang`` sees native ``AiterAttnBackend`` unless the user set
the variable already. Additional hooks can be registered via
:func:`register_sglang_prepare_hook`.
"""

from __future__ import annotations

import os
from typing import Any, Callable

_SGLANG_PREPARE_HOOKS: list[Callable[[str, Any], None]] = []


def register_sglang_prepare_hook(fn: Callable[[str, Any], None]) -> None:
    """Register ``fn(model_arch, atom_config)``. Each hook decides whether to act."""
    if fn not in _SGLANG_PREPARE_HOOKS:
        _SGLANG_PREPARE_HOOKS.append(fn)


def run_sglang_prepare_hooks(model_arch: str, atom_config: Any) -> None:
    """Invoke all registered hooks in order (before ``register_ops_to_sglang``)."""
    for fn in _SGLANG_PREPARE_HOOKS:
        fn(model_arch, atom_config)


def _qwen3_prepare_hook(model_arch: str, atom_config: Any) -> None:
    """Default Qwen3.5 / Qwen3-next SGLang prepare logic (lazy-imports model modules)."""
    if model_arch.startswith("Qwen3_5") or model_arch == "Qwen3NextForCausalLM":
        os.environ.setdefault("ATOM_SGLANG_USE_NATIVE_AITER_ATTN_BACKEND", "1")
    if model_arch.startswith("Qwen3_5"):
        from atom.plugin.sglang.models.qwen3_5 import apply_prepare_model_adaptations

        apply_prepare_model_adaptations(atom_config, model_arch)
    elif model_arch == "Qwen3NextForCausalLM":
        from atom.plugin.sglang.models.qwen3_next import (
            apply_qwen3_next_sglang_model_patch,
        )

        apply_qwen3_next_sglang_model_patch()


register_sglang_prepare_hook(_qwen3_prepare_hook)
