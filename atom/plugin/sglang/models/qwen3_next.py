"""ATOM Qwen3-next model wrappers for SGLang external model loading."""

import logging
from typing import Any

import torch
import atom.models.qwen3_next as _qwen3_next_shim

from atom.models.qwen3_next import Qwen3NextModel as _Qwen3NextModelCore
from atom.models.utils import IntermediateTensors
from atom.plugin.sglang.models.base_model_wrapper import _AtomCausalLMBaseForSglang
from atom.plugin.sglang.utils.gdn_forward_helper import sglang_gdn_bridge

logger = logging.getLogger("atom.plugin.sglang.models.qwen3_next")


class Qwen3NextSglangModel(_Qwen3NextModelCore):
    """``Qwen3NextModel`` with SGLang GDN bridge (keeps ``atom.models`` plugin-free)."""

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        fb = model_kwargs.get("forward_batch")
        with sglang_gdn_bridge(fb):
            return _Qwen3NextModelCore.forward(
                self,
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
                **model_kwargs,
            )


def apply_qwen3_next_sglang_model_patch() -> None:
    if _qwen3_next_shim.Qwen3NextModel is Qwen3NextSglangModel:
        return
    _qwen3_next_shim.Qwen3NextModel = Qwen3NextSglangModel


class Qwen3NextForCausalLM(_AtomCausalLMBaseForSglang):
    """ATOM-backed Qwen3Next for SGLang (shared init / forward / load_weights with base)."""


EntryClass = Qwen3NextForCausalLM
