"""SGLang-specific ATOM weight loading helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from atom.model_loader.loader import load_model_in_plugin_mode


@dataclass(frozen=True)
class SGLangLoaderPatch:
    weights_mapper: Any = None
    load_fused_expert_weights_fn: Optional[Callable[..., Any]] = None
    packed_mapping_key_matcher: Optional[Callable[[str, str], bool]] = None

    @classmethod
    def from_model(
        cls,
        model,
        *,
        packed_mapping_key_matcher: Optional[Callable[[str, str], bool]] = None,
    ) -> "SGLangLoaderPatch":
        """Read the model-provided loader patch and apply wrapper defaults."""
        get_patch = getattr(model, "get_sglang_loader_patch", None)
        loader_patch = (
            get_patch()
            if callable(get_patch)
            else getattr(model, "sglang_loader_patch", None)
        )
        if loader_patch is None:
            loader_patch = cls()
        return cls(
            weights_mapper=loader_patch.weights_mapper,
            load_fused_expert_weights_fn=loader_patch.load_fused_expert_weights_fn,
            packed_mapping_key_matcher=(
                packed_mapping_key_matcher
                if packed_mapping_key_matcher is not None
                else loader_patch.packed_mapping_key_matcher
            ),
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
) -> set[str]:
    """Load an ATOM model through the shared plugin-mode loader for SGLang.

    SGLang plugin models may expose a small model-specific loader patch via
    `get_sglang_loader_patch()` or `sglang_loader_patch`. This helper resolves
    that patch once, then forwards the merged loading options into ATOM's
    generic `load_model_in_plugin_mode()` implementation.

    The only SGLang-specific default added here is the packed-module key matcher:
    SGLang packed mappings should match whole weight-name path segments instead
    of using the broader default substring match.
    """
    # Read optional per-model loading customizations and inject the stricter
    # packed-weight matcher expected by SGLang plugin models.
    loader_patch = SGLangLoaderPatch.from_model(
        model,
        packed_mapping_key_matcher=_packed_mapping_key_matches_weight_name,
    )
    # Delegate the actual weight iteration/loading to the shared ATOM loader so
    # SGLang stays aligned with the common plugin-mode loading path.
    return load_model_in_plugin_mode(
        model=model,
        config=config,
        prefix=prefix,
        weights_mapper=loader_patch.weights_mapper,
        load_fused_expert_weights_fn=loader_patch.load_fused_expert_weights_fn,
        packed_mapping_key_matcher=loader_patch.packed_mapping_key_matcher,
    )
