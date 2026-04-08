"""ATOM Qwen3.5 VLM for SGLang external model inference (out-of-tree).

Subclasses SGLang in-tree ``Qwen3_5ForConditionalGeneration`` /
``Qwen3_5MoeForConditionalGeneration``; language stack is built via
``atom.prepare_model(..., engine="sglang")``.

Set ``SGLANG_EXTERNAL_MODEL_PACKAGE`` to the parent of this package (e.g.
``atom.plugin.sglang.oot``'s parent is ``atom.plugin.sglang`` — use the value
that matches your SGLang external package convention).

**Weight loading parity (FP8 / packed tensors)**

- Language weights are loaded by ``atom.model_loader.loader.load_model`` (plugin
  mode), not by SGLang's in-tree ``load_weights``. ``packed_modules_mapping``
  must stay aligned with SGLang's ``QWEN3_5_PACKED_MODULES_MAPPING`` intent
  (inverse mapping: HF shard name → fused param + shard id).

- On HIP, SGLang's ``Qwen3_5ForCausalLM.load_weights`` applies
  ``qwen3_5_gdn_fused_proj.FUSED_PROJ_WEIGHT_MAPPING`` before stacked QKV/MLP
  mapping. ATOM GDN uses ``in_proj_qkvz`` / ``in_proj_ba`` only; the subset of
  that table for ``in_proj_qkv*`` / ``in_proj_z`` / ``in_proj_b`` /
  ``in_proj_a`` is already covered by ``_QWEN35_OOT_PACKED_MODULES_MAPPING``.
  Checkpoints that expose a single ``in_proj_fused`` tensor (full BF16 fusion)
  are not supported here—use HF-style split projection keys.

- MoE fused expert slabs: SGLang chunks ``gate_up_proj`` on ``dim=-2``; this
  module uses ``load_fused_expert_weights`` with orientation detection so both
  ``[2*half, hidden]`` and ``[hidden, 2*half]`` slabs load correctly (relevant
  for some FP8 checkpoints).

- After ``super().__init__``, we assign ``quant_config.packed_modules_mapping``
  when the config supports it, matching ``Qwen3_5ForCausalLM`` so FP8 (and other)
  quant paths that read ``packed_modules_mapping`` see the same mapping as the
  in-tree model class.

- ``atom.plugin.prepare.prepare_model`` dispatches to
  ``apply_prepare_model_adaptations`` (this module) for ``Qwen3_5*`` arches:
  MoE text-config patch for ``Qwen3NextSparseMoeBlock``, then quant remap
  (parity with vLLM ``model_wrapper`` / ROCm/ATOM#448).
"""

from __future__ import annotations

import logging
import os

# Before ``register_ops_to_sglang`` runs, prefer SGLang built-in ``AiterAttnBackend``
# over ``ATOMAttnBackendForSgl`` for this OOT path (PR #355). Override with
# ``ATOM_SGLANG_USE_NATIVE_AITER_ATTN_BACKEND=0`` in the environment if needed.
os.environ.setdefault("ATOM_SGLANG_USE_NATIVE_AITER_ATTN_BACKEND", "1")
from collections.abc import Iterable
from typing import Any, Optional

import torch
from torch import nn

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    PPProxyTensors,
)
from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration as _SglQwen35VL,
    Qwen3_5MoeForConditionalGeneration as _SglQwen35MoeVL,
)

from atom.model_loader.loader import WeightsMapper, load_model_in_plugin_mode
from atom.models.qwen3_5 import Qwen3_5Model
from atom.models.utils import IntermediateTensors
from transformers.configuration_utils import layer_type_validation

logger = logging.getLogger("atom.plugin.sglang.oot")


# Inverse of SGLang ``QWEN3_5_PACKED_MODULES_MAPPING`` (HF split names → fused module).
# Extra keys ``.gate.`` / ``shared_expert_gate`` support ROCm shared-expert fusion naming.
#
# Loader uses path-segment matching (``atom/model_loader/loader.py``) so ``v_proj``
# does not match inside ``qkv_proj``, ``in_proj_qkv`` not inside ``in_proj_qkvz``, etc.
# Dynamic FP8 checkpoints often use fused tensor names; ``(name, None)`` rows use
# ``MergedColumnParallelLinear.weight_loader`` fused split. Without this, ATOM plugin
# load corrupts weights vs native SGLang ``CompressedTensorsW8A8Fp8MoE`` and
# ``shuffle_weight`` fails (e.g. ``1 % 16 == 1``).
_QWEN35_OOT_PACKED_MODULES_MAPPING = {
    "qkv_proj": ("qkv_proj", None),
    "in_proj_qkvz": ("in_proj_qkvz", None),
    "in_proj_ba": ("in_proj_ba", None),
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_up_proj": ["gate_proj", "up_proj"],
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
    "in_proj_qkv": ("in_proj_qkvz", (0, 1, 2)),
    "in_proj_z": ("in_proj_qkvz", 3),
    "in_proj_b": ("in_proj_ba", 0),
    "in_proj_a": ("in_proj_ba", 1),
    ".gate.": (".gate.", 0),
    "shared_expert_gate": ("gate", 1),
}

_QWEN35_SGLANG_VL_HF_MAPPER = WeightsMapper(
    orig_to_new_substr={
        "attn.qkv.": "attn.qkv_proj.",
    },
    orig_to_new_prefix={
        "model.visual.": "visual.",
        "model.language_model.model.": "model.",
        "model.language_model.lm_head.": "lm_head.",
        "model.language_model.": "model.",
        "lm_head.": "lm_head.",
    },
)


def _patch_qwen35_moe_text_for_sparse_moe_block(hf_config: Any) -> None:
    """HF Qwen3.5 MoE text config often omits fields that Qwen3NextSparseMoeBlock uses."""
    tc = getattr(hf_config, "text_config", None)
    if tc is None:
        tc = hf_config
    mt = getattr(tc, "model_type", "") or ""
    if "qwen3_5" not in mt or "moe" not in mt:
        return
    tc.n_shared_experts = 1
    if hasattr(tc, "num_experts"):
        tc.n_routed_experts = tc.num_experts


def remap_qwen35_quant_config_for_sglang_plugin(atom_config: Any) -> None:
    """Align ``QuantizationConfig`` layer patterns with OOT weight names (vLLM PR #448).

    vLLM plugin calls ``quant_config.remap_layer_name`` in ``model_wrapper`` before
    building the ATOM model; SGLang OOT must do the same so FP8 layer patterns /
    ``exclude_layers`` match parameter names after HF→SGLang prefix mapping.
    """
    atom_config.quant_config.remap_layer_name(
        atom_config.hf_config,
        packed_modules_mapping=dict(_QWEN35_OOT_PACKED_MODULES_MAPPING),
        weights_mapper=_QWEN35_SGLANG_VL_HF_MAPPER,
    )


_QWEN35_PREPARE_MOE_ARCHES = frozenset(
    ("Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeForCausalLM")
)
_QWEN35_PREPARE_REMAP_ARCHES = frozenset(
    ("Qwen3_5MoeForConditionalGeneration", "Qwen3_5ForConditionalGeneration")
)


def apply_prepare_model_adaptations(atom_config: Any, model_arch: str) -> None:
    """Qwen3.5-only hooks for ``atom.plugin.prepare.prepare_model`` (before ATOM LM build)."""
    if model_arch in _QWEN35_PREPARE_MOE_ARCHES:
        _patch_qwen35_moe_text_for_sparse_moe_block(atom_config.hf_config)
    if model_arch in _QWEN35_PREPARE_REMAP_ARCHES:
        remap_qwen35_quant_config_for_sglang_plugin(atom_config)


def _patch_qwen35_moe_text_config_for_atom(config: Any) -> None:
    """VLM ``__init__``: same MoE text patch as prepare_model (multimodal root config)."""
    _patch_qwen35_moe_text_for_sparse_moe_block(config)


class _AtomQwen35FlatLanguageStack(nn.Module):
    """Expose ``embed_tokens`` / ``layers`` / ``norm`` for SGLang VLM."""

    _pending_vlm_root_config: Any = None

    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        del config
        root = _AtomQwen35FlatLanguageStack._pending_vlm_root_config
        if root is None:
            raise RuntimeError(
                "ATOM Qwen3.5 VLM language stack: missing pending VLM root config"
            )

        import atom

        self.pp_group = get_pp_group()
        self.quant_config = quant_config

        atom_lm = atom.prepare_model(config=root, engine="sglang")
        if atom_lm is None:
            arch = getattr(root, "architectures", ["unknown"])[0]
            raise ValueError(f"ATOM failed to build language model for {arch}")

        dec = atom_lm.model
        self.config = dec.config
        self.vocab_size = dec.vocab_size
        self.start_layer = dec.start_layer
        self.end_layer = dec.end_layer
        self.make_empty_intermediate_tensors = dec.make_empty_intermediate_tensors
        self.atom_config = atom_lm.atom_config

        if hasattr(atom_lm, "lm_head"):
            atom_lm._modules.pop("lm_head", None)

        for name, child in dec.named_children():
            self.add_module(name, child)

        atom_lm._modules.pop("model", None)
        self._atom_lm_ref = atom_lm

        if hasattr(atom_lm, "get_expert_mapping"):
            self.get_expert_mapping = atom_lm.get_expert_mapping  # type: ignore[method-assign]

    def get_input_embeddings(self):
        return self.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        input_deepstack_embeds: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        model_kwargs: dict[str, Any] = dict(kwargs)
        if forward_batch is not None:
            model_kwargs["forward_batch"] = forward_batch
        if pp_proxy_tensors is not None:
            model_kwargs["pp_proxy_tensors"] = pp_proxy_tensors
        if input_deepstack_embeds is not None:
            model_kwargs["input_deepstack_embeds"] = input_deepstack_embeds

        out = Qwen3_5Model.forward(
            self,
            input_ids,
            positions,
            None,
            input_embeds,
            **model_kwargs,
        )
        if isinstance(out, IntermediateTensors):
            return PPProxyTensors(dict(out.tensors))
        return out


class Qwen3_5ForConditionalGeneration(_SglQwen35VL):
    packed_modules_mapping = _QWEN35_OOT_PACKED_MODULES_MAPPING

    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        _AtomQwen35FlatLanguageStack._pending_vlm_root_config = config
        try:
            super().__init__(
                config,
                quant_config,
                prefix,
                language_model_cls=_AtomQwen35FlatLanguageStack,
            )
        finally:
            _AtomQwen35FlatLanguageStack._pending_vlm_root_config = None
        self.atom_config = self.model.atom_config
        if quant_config is not None and hasattr(
            quant_config, "packed_modules_mapping"
        ):
            quant_config.packed_modules_mapping = self.packed_modules_mapping
        logger.info("Initialized ATOM-backed %s", self.__class__.__name__)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return load_model_in_plugin_mode(
            model=self,
            config=self.atom_config,
            prefix="",
            weights_mapper=_QWEN35_SGLANG_VL_HF_MAPPER,
        )


class Qwen3_5MoeForConditionalGeneration(_SglQwen35MoeVL):
    packed_modules_mapping = _QWEN35_OOT_PACKED_MODULES_MAPPING

    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        _patch_qwen35_moe_text_config_for_atom(config)
        _AtomQwen35FlatLanguageStack._pending_vlm_root_config = config
        try:
            super().__init__(
                config,
                quant_config,
                prefix,
                language_model_cls=_AtomQwen35FlatLanguageStack,
            )
        finally:
            _AtomQwen35FlatLanguageStack._pending_vlm_root_config = None
        self.atom_config = self.model.atom_config
        if quant_config is not None and hasattr(
            quant_config, "packed_modules_mapping"
        ):
            quant_config.packed_modules_mapping = self.packed_modules_mapping
        logger.info("Initialized ATOM-backed %s", self.__class__.__name__)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

    def detect_fused_expert_format(self, weight_name: str) -> bool:
        return "experts.gate_up_proj" in weight_name or (
            "experts.down_proj" in weight_name
            and ".experts." in weight_name
            and weight_name.count(".experts.") == 1
        )

    def get_fused_expert_mapping(self) -> list[tuple[str, str, str]]:
        return [
            ("experts.w13_weight", "experts.gate_up_proj", "w1"),
            ("experts.w2_weight", "experts.down_proj", "w2"),
        ]

    @staticmethod
    def _expert_gate_up_to_w1_w3(
        fused: torch.Tensor, half: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One expert's fused gate+up → ``w1`` / ``w3`` rows expected by ``FusedMoE``.

        ATOM stores ``w13_weight[e]`` as ``[2*half, hidden]`` (gate then up on dim 0).
        Checkpoints may use the same layout or ``[hidden, 2*half]``.
        """
        if fused.dim() > 2:
            fused = fused.reshape(fused.shape[0], -1)
        rows, cols = fused.shape
        two_h = 2 * half
        if rows == two_h:
            return fused.narrow(0, 0, half), fused.narrow(0, half, half)
        if cols == two_h:
            t = fused.transpose(0, 1).contiguous()
            return t.narrow(0, 0, half), t.narrow(0, half, half)
        raise ValueError(
            f"gate_up_proj expert slab shape {tuple(fused.shape)}: expected "
            f"one dim == {two_h} (2*half, half={half})"
        )

    @staticmethod
    def _gate_up_halves_for_split(
        half_param: int, tp_size: int
    ) -> list[int]:
        """Order matters: prefer full-checkpoint half before TP-local layout.

        HF stores ``gate_up_proj`` as ``[E, 2 * moe_intermediate, hidden]``
        (global). ``w13_weight`` uses ``2 * intermediate_size_per_partition``
        per rank, so ``half_param * tp_size`` matches the checkpoint fused dim.
        """
        out: list[int] = []
        if tp_size > 1:
            out.append(half_param * tp_size)
        out.append(half_param)
        return out

    def _expert_gate_up_to_w1_w3_multi(
        self, fused: torch.Tensor, half_candidates: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        last: Optional[ValueError] = None
        for h in half_candidates:
            try:
                return self._expert_gate_up_to_w1_w3(fused, h)
            except ValueError as e:
                last = e
        assert last is not None
        raise last

    def load_fused_expert_weights(
        self,
        original_name: str,
        name: str,
        params_dict: dict,
        loaded_weight: torch.Tensor,
        shard_id: str,
        num_experts: int,
    ) -> bool:
        # ``name`` is the mapped param path (…w13_weight); ``original_name`` is the
        # checkpoint key (…gate_up_proj). Only the latter contains gate_up_proj.
        param = params_dict[name]
        weight_loader = param.weight_loader
        loaded_local_expert = False

        if "gate_up_proj" in original_name:
            half_param = param.data[0].shape[0] // 2
            moe = getattr(weight_loader, "__self__", None)
            tp_size = int(getattr(moe, "tp_size", 1) or 1)
            half_candidates = self._gate_up_halves_for_split(half_param, tp_size)
            w = loaded_weight
            for expert_id in range(num_experts):
                slab = w[expert_id]
                w1_slab, w3_slab = self._expert_gate_up_to_w1_w3_multi(
                    slab, half_candidates
                )
                weight_loader(param, w1_slab, name, "w1", expert_id)
                weight_loader(param, w3_slab, name, "w3", expert_id)
                loaded_local_expert = True
        else:
            for expert_id in range(num_experts):
                weight_loader(
                    param,
                    loaded_weight[expert_id],
                    name,
                    shard_id,
                    expert_id,
                )
                loaded_local_expert = True

        return loaded_local_expert

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return load_model_in_plugin_mode(
            model=self,
            config=self.atom_config,
            prefix="",
            weights_mapper=_QWEN35_SGLANG_VL_HF_MAPPER,
            load_fused_expert_weights_fn=self.load_fused_expert_weights,
        )


EntryClass = [Qwen3_5MoeForConditionalGeneration, Qwen3_5ForConditionalGeneration]