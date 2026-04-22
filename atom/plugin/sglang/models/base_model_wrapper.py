"""ATOM model wrappers for SGLang external model loading.

Registers model architecture classes via SGLANG_EXTERNAL_MODEL_PACKAGE,
replacing sglang's built-in implementations with ATOM-optimized versions.

"""

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple, Union

import torch
from torch import nn

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from atom.plugin.sglang.utils.forward_context import SGLangForwardBatchMetadata

logger = logging.getLogger("atom.plugin.sglang.models")


@dataclass(frozen=True)
class ModelArchSpec:
    native_sglang_forward: bool = False
    wrapper_binds_gdn_context: bool = False
    apply_deepseek_sglang_patch: bool = False


_MODEL_ARCH_SPECS = {
    "DeepseekV3ForCausalLM": ModelArchSpec(apply_deepseek_sglang_patch=True),
    "Qwen3MoeForCausalLM": ModelArchSpec(),
    "Qwen3NextForCausalLM": ModelArchSpec(wrapper_binds_gdn_context=True),
    # Qwen3.5 keeps SGLang's native conditional-generation shell, so the wrapper
    # must pass through its forward_batch-style inputs instead of ATOM tensors.
    "Qwen3_5ForConditionalGeneration": ModelArchSpec(native_sglang_forward=True),
    "Qwen3_5MoeForConditionalGeneration": ModelArchSpec(native_sglang_forward=True),
}


def _ensure_config_vocab_size(config: Any) -> int:
    """Read and backfill top-level vocab_size for nested multimodal configs."""
    for candidate in (config, getattr(config, "text_config", None)):
        if candidate is not None and hasattr(candidate, "vocab_size"):
            vocab_size = candidate.vocab_size
            if not hasattr(config, "vocab_size"):
                setattr(config, "vocab_size", vocab_size)
            return vocab_size
    raise AttributeError(f"{type(config).__name__} does not define vocab_size")


class _AtomCausalLMBaseForSglang(nn.Module):
    """Base ATOM model wrapper conforming to sglang's model interface.

    Delegates model creation and weight loading to ATOM's plugin system,
    while providing the forward signature and LogitsProcessorOutput return
    type that sglang expects.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        logger.info("Initializing ATOM backend for %s", self.__class__.__name__)

        self.pp_group = get_pp_group()
        self.quant_config = quant_config
        self.config = config
        self.model_arch = getattr(config, "architectures", [""])[0]
        self.model_arch_spec = _MODEL_ARCH_SPECS.get(self.model_arch, ModelArchSpec())
        self.vocab_size = _ensure_config_vocab_size(config)
        self.unpadded_vocab_size = self.vocab_size

        import atom

        self.model = atom.prepare_model(config=config, engine="sglang")

        if self.model is None:
            raise ValueError(
                f"ATOM failed to create model for architecture {self.model_arch}"
            )

        self.logits_processor = LogitsProcessor(config)

        # Apply ds model-specific sglang patches (attn dispatch, weight hooks, etc.)
        # TODO: will remove this after sglang supports atom attention backend
        if self.model_arch_spec.apply_deepseek_sglang_patch:
            from atom.plugin.sglang.attention_backend.sgl_attention_mla import (
                setup_deepseek_for_sglang,
            )

            setup_deepseek_for_sglang(self.model)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        **model_kwargs: Any,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        metadata = SGLangForwardBatchMetadata.build(
            forward_batch,
            pp_proxy_tensors=pp_proxy_tensors,
            save_kv_cache=model_kwargs.get("save_kv_cache"),
        )
        with SGLangForwardBatchMetadata.bind(metadata):
            if self.model_arch_spec.native_sglang_forward:
                model_inputs = dict(
                    input_ids=input_ids,
                    positions=positions,
                    forward_batch=forward_batch,
                    get_embedding=get_embedding,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
            else:
                model_inputs = dict(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=SGLangForwardBatchMetadata.to_intermediate_tensors(
                        pp_proxy_tensors, metadata
                    ),
                    inputs_embeds=input_embeds,
                )
            if self.model_arch_spec.wrapper_binds_gdn_context:
                from atom.plugin.sglang.utils.gdn_context import (
                    SGLangGDNForwardContext,
                )

                with SGLangGDNForwardContext.bind(metadata):
                    hidden_states = self.model(**model_inputs)
            else:
                hidden_states = self.model(**model_inputs)

        if self.model_arch_spec.native_sglang_forward:
            return hidden_states

        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids,
                hidden_states,
                self.model.lm_head,
                forward_batch,
            )
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # The passed `weights` iterable from sglang is ignored because ATOM
        # uses its own weight loading pipeline (handling AITER-specific quant
        # formats, kv_b_proj splitting, etc.) that is incompatible with
        # sglang's default weight iterator.
        from atom.plugin.sglang.utils.loader import load_model_in_sglang_plugin_mode

        return load_model_in_sglang_plugin_mode(
            model=self.model,
            config=self.model.atom_config,
            prefix="model.",
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.compute_logits(hidden_states)

    def get_input_embeddings(self, *args: Any, **kwargs: Any):
        return self.model.get_input_embeddings(*args, **kwargs)

    def pad_input_ids(self, *args: Any, **kwargs: Any):
        return self.model.pad_input_ids(*args, **kwargs)

    def get_image_feature(self, *args: Any, **kwargs: Any):
        return self.model.get_image_feature(*args, **kwargs)

    def get_video_feature(self, *args: Any, **kwargs: Any):
        return self.model.get_video_feature(*args, **kwargs)

    def embed_input_ids(self, *args: Any, **kwargs: Any):
        return self.model.embed_input_ids(*args, **kwargs)


EntryClass = []
for _name in _MODEL_ARCH_SPECS:
    _cls = type(_name, (_AtomCausalLMBaseForSglang,), {})
    globals()[_name] = _cls
    EntryClass.append(_cls)
