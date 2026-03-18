from collections.abc import Iterable
from contextlib import nullcontext

import importlib
import torch
import torch.nn as nn
from torch.profiler import record_function
from aiter.dist.parallel_state import (
    get_pp_group,
    get_tp_group,
)
from vllm.config import VllmConfig, get_current_vllm_config_or_none
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.models.interfaces import (
    SupportsPP,
    SupportsQuant,
)
from vllm.model_executor.models.interfaces_base import (
    VllmModel,
    VllmModelForTextGeneration,
)
from vllm.sequence import IntermediateTensors

import atom  # noqa: F401
from atom.plugin.config import generate_atom_config_for_plugin_mode
from atom.model_loader.loader import load_model_in_plugin_mode

import logging

logger = logging.getLogger("atom")


_ATOM_MODEL_CLASSES: dict[str, str] = {
    "Qwen3ForCausalLM": "atom.models.qwen3:Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM": "atom.models.qwen3_moe:Qwen3MoeForCausalLM",
    "GptOssForCausalLM": "atom.models.gpt_oss:GptOssForCausalLM",
    "DeepseekV3ForCausalLM": "atom.models.deepseek_v2:DeepseekV3ForCausalLM",
    "Glm4MoeForCausalLM": "atom.models.glm4_moe:Glm4MoeForCausalLM",
}


def _get_atom_model_cls(model_arch: str) -> type:
    if model_arch is not None and model_arch in _ATOM_MODEL_CLASSES:
        model_ref = _ATOM_MODEL_CLASSES[model_arch]
    else:
        raise ValueError(f"The {model_arch} is not supported by ATOM OOT backend")

    module_path, class_name = model_ref.split(":", 1)
    return getattr(importlib.import_module(module_path), class_name)


def _prepare_env(atom_config) -> None:
    from atom.plugin.register import set_attn_cls, init_aiter_dist

    # set global attention class
    logger.info("Set global attention class")
    set_attn_cls()

    # init aiter dist for using aiter custom collective ops
    logger.info("Init aiter dist for using aiter custom collective ops")
    init_aiter_dist(config=atom_config)

    _patch_vllm_profile_labels()


def _is_torch_profile_enabled(vllm_config: VllmConfig) -> bool:
    profiler_config = getattr(vllm_config, "profiler_config", None)
    return profiler_config is not None and bool(
        getattr(profiler_config, "torch_profiler_dir", "")
    )


def _patch_vllm_profile_labels() -> None:
    from vllm.v1.worker import gpu_model_runner as gpu_model_runner_mod
    from vllm.v1 import utils as v1_utils_mod

    # Use vLLM's existing step boundary in execute_model():
    #
    #   with set_forward_context(...),
    #        record_function_or_nullcontext("gpu_model_runner: forward"):
    #       model_output = self._model_forward(...)
    #
    # This encloses the real model run for eager, piecewise, and full graph
    # modes, so a single dynamic label hook here is simpler than patching
    # separate execution paths. Note that gpu_model_runner imported
    # `record_function_or_nullcontext` by value, so patching only v1.utils is
    # not sufficient after import; we patch the local symbol in
    # gpu_model_runner as well.
    if not getattr(gpu_model_runner_mod, "_atom_step_label_patched", False):
        original_record_ctx = gpu_model_runner_mod.record_function_or_nullcontext
        original_utils_record_ctx = v1_utils_mod.record_function_or_nullcontext

        class _DynamicForwardRecordContext:
            def __init__(self, name: str, original_ctx):
                self.name = name
                self.original_ctx = original_ctx
                self.ctx = nullcontext()

            def __enter__(self):
                if self.name == "gpu_model_runner: forward":
                    vllm_config = get_current_vllm_config_or_none()
                    if (
                        vllm_config is not None
                        and _is_torch_profile_enabled(vllm_config)
                        and is_forward_context_available()
                    ):
                        record_label = _build_step_profiler_label()
                        if record_label is not None:
                            self.ctx = record_function(record_label)
                            return self.ctx.__enter__()

                self.ctx = self.original_ctx(self.name)
                return self.ctx.__enter__()

            def __exit__(self, exc_type, exc, tb):
                return self.ctx.__exit__(exc_type, exc, tb)

        def _wrapped_gpu_record_function_or_nullcontext(name: str):
            return _DynamicForwardRecordContext(name, original_record_ctx)

        def _wrapped_utils_record_function_or_nullcontext(name: str):
            return _DynamicForwardRecordContext(name, original_utils_record_ctx)

        gpu_model_runner_mod.record_function_or_nullcontext = (
            _wrapped_gpu_record_function_or_nullcontext
        )
        v1_utils_mod.record_function_or_nullcontext = (
            _wrapped_utils_record_function_or_nullcontext
        )
        gpu_model_runner_mod._atom_step_label_patched = True


def _get_step_attn_metadata_list():
    if not is_forward_context_available():
        return []

    attn_metadata = get_forward_context().attn_metadata
    if attn_metadata is None:
        return []

    if isinstance(attn_metadata, list):
        metadatas = []
        # In ubatch mode, vLLM stores one metadata dict per microbatch. We need
        # one representative metadata object per microbatch and aggregate them to
        # get the full-step request/token counts.
        for ubatch_attn_metadata in attn_metadata:
            if not ubatch_attn_metadata:
                continue
            metadata = next(iter(ubatch_attn_metadata.values()), None)
            if metadata is not None:
                metadatas.append(metadata)
        return metadatas

    if isinstance(attn_metadata, dict):
        metadata = next(iter(attn_metadata.values()), None)
        return [metadata] if metadata is not None else []

    return [attn_metadata]


def _build_step_profiler_label() -> str | None:
    attn_metadata_list = _get_step_attn_metadata_list()
    if not attn_metadata_list:
        return None

    num_actual_tokens = 0
    num_decodes = 0
    num_decode_tokens = 0
    num_prefills = 0
    num_extends = 0
    num_extend_tokens = 0
    num_prefill_tokens = 0

    for attn_metadata in attn_metadata_list:
        plugin_metadata = getattr(attn_metadata, "plugin_metadata", None)
        if plugin_metadata is None:
            continue

        actual_tokens_i = plugin_metadata.num_actual_tokens
        decodes_i = plugin_metadata.num_decodes
        decode_tokens_i = plugin_metadata.num_decode_tokens
        prefills_i = plugin_metadata.num_prefills
        extends_i = getattr(plugin_metadata, "num_extends", 0)
        extend_tokens_i = getattr(plugin_metadata, "num_extend_tokens", 0)
        prefill_tokens_i = getattr(
            plugin_metadata,
            "num_prefill_tokens",
            actual_tokens_i - decode_tokens_i - extend_tokens_i,
        )

        num_actual_tokens += actual_tokens_i
        num_decodes += decodes_i
        num_decode_tokens += decode_tokens_i
        num_prefills += prefills_i
        num_extends += extends_i
        num_extend_tokens += extend_tokens_i
        num_prefill_tokens += prefill_tokens_i

    if num_actual_tokens <= 0:
        return None

    total_reqs = num_decodes + num_prefills + num_extends
    if total_reqs <= 0:
        return None

    # OOT step naming policy:
    # - prefill wins if any prefill tokens exist
    # - otherwise extend wins if any extend tokens exist
    # - otherwise the step is decode
    # `bs` is total request count, `tok` is total token count,
    # and `p/e/d` are prefill/extend/decode token counts.
    if num_prefills > 0:
        step = "prefill"
    elif num_extends > 0:
        step = "extend"
    else:
        step = "decode"

    return (
        f"{step}[bs={total_reqs} tok={num_actual_tokens} "
        f"p={num_prefill_tokens} e={num_extend_tokens} d={num_decode_tokens}]"
    )


class ATOMModelBase(nn.Module, VllmModel, SupportsQuant, SupportsPP):
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        self.config = vllm_config.model_config.hf_config
        self.text_config = self.config.get_text_config()
        self.cache_config = vllm_config.cache_config
        self.device_config = vllm_config.device_config
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.quant_config = vllm_config.quant_config

        # Weights to skip in `self.load_weights`
        self.skip_prefixes: list[str] = []
        self.skip_substrs: list[str] = []
        self.ignore_unexpected_prefixes: list[str] = []
        self.ignore_unexpected_suffixes: list[str] = []

        self.atom_config = generate_atom_config_for_plugin_mode(vllm_config)

        _prepare_env(atom_config=self.atom_config)

        model_arch = vllm_config.model_config.architectures[0]
        model_cls = _get_atom_model_cls(model_arch)

        logger.info(f"Construct ATOM model {model_arch} for vLLM plugin mode")
        self.model = model_cls(self.atom_config)

        if self.model is None:
            model_arch = vllm_config.model_config.architectures[0]
            raise ValueError(
                f"The model {model_arch} is not supported by model impl backend atom"
            )

        # here init aiter dist for using aiter custom collective ops
        self.pp_group = get_pp_group()
        self.tp_group = get_tp_group()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        if not self.pp_group.is_first_rank:
            assert intermediate_tensors is not None
            input_ids = None
            inputs_embeds = intermediate_tensors["hidden_states"]

        # capture. This ensures attention_mla reads correct positions in graph mode.
        # This is only for mla attention in plugin mode.
        if "positions" in self.atom_config.compilation_config.static_forward_context:
            buf = self.atom_config.compilation_config.static_forward_context[
                "positions"
            ]
            buf[: positions.numel()].copy_(positions)
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )

        if not self.pp_group.is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        return hidden_states

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        loaded_weights_record = load_model_in_plugin_mode(
            model=self.model, config=self.atom_config, prefix="model."
        )
        return loaded_weights_record

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.model.compute_logits(hidden_states)
        return logits


class ATOMForCausalLM(ATOMModelBase, VllmModelForTextGeneration): ...


class ATOMMoEForCausalLM(ATOMModelBase, VllmModelForTextGeneration): ...
