"""ATOM Qwen3-next model wrappers for SGLang external model loading."""

from __future__ import annotations

from typing import Any, Union

import torch
from aiter.dist.parallel_state import get_pp_group

from atom.config import CompilationLevel
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.layernorm import GemmaRMSNorm as Qwen3NextRMSNorm
from atom.models.qwen3_next import (
    Qwen3NextAttention as _CoreQwen3NextAttention,
    Qwen3NextDecoderLayer as _CoreQwen3NextDecoderLayer,
    Qwen3NextForCausalLM as _CoreQwen3NextForCausalLM,
    Qwen3NextModel as _CoreQwen3NextModel,
)
from atom.models.utils import (
    IntermediateTensors,
    PPMissingLayer,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from atom.plugin.sglang.models.base_model_wrapper import _AtomCausalLMBaseForSglang
from atom.plugin.sglang.utils.gdn_forward_helper import sglang_gdn_bridge
from atom.utils.decorators import TorchCompileWrapperWithCustomDispatcher


class Qwen3NextSglangAttention(_CoreQwen3NextAttention):
    """SGLang-aware full attention that forwards ``forward_batch`` to RadixAttention."""

    def forward(
        self,
        positions: torch.Tensor,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)

        if self.attn_output_gate:
            q_gate, k, v = torch.split(
                qkv, [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = torch.split(
                qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1
            )

        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
            -1, self.num_heads * self.head_dim
        )
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
            -1, self.num_kv_heads * self.head_dim
        )

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, positions=positions, **model_kwargs)

        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        output[:] = self.o_proj(attn_output)
        return output


class Qwen3NextSglangDecoderLayer(_CoreQwen3NextDecoderLayer):
    """Reuse the core decoder layer and swap only the SGLang-specific attention path."""

    def __init__(
        self,
        atom_config,
        layer_type: str,
        prefix: str = "",
        layer_num: int = 0,
    ) -> None:
        super().__init__(atom_config, layer_type, prefix, layer_num)
        if self.layer_type == "full_attention":
            self.self_attn = Qwen3NextSglangAttention(
                atom_config,
                quant_config=atom_config.quant_config,
                prefix=f"{prefix}.self_attn",
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **model_kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        self_attention_output = torch.empty_like(hidden_states)
        if self.layer_type == "linear_attention":
            self.linear_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
            )
        elif self.layer_type == "full_attention":
            self.self_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
                positions=positions,
                **model_kwargs,
            )
        else:
            raise ValueError("Invalid layer_type")
        hidden_states = self_attention_output

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype) + 1
                )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                assert len(hidden_states.shape) == len(self.ffn_layer_scale.shape), (
                    f"shape must be the same {len(hidden_states.shape)}, "
                    f"{len(self.ffn_layer_scale.shape)}"
                )
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype) + 1
                )

        return hidden_states, residual


class Qwen3NextSglangModel(_CoreQwen3NextModel):
    """SGLang-aware Qwen3-next model without touching ``atom.models.qwen3_next``."""

    def __init__(self, atom_config, prefix: str = ""):
        super(_CoreQwen3NextModel, self).__init__()

        config = atom_config.hf_config
        self.config = config
        self.config.n_shared_experts = 1
        self.config.n_routed_experts = self.config.num_experts

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str, layer_num: int):
            return Qwen3NextSglangDecoderLayer(
                atom_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
                layer_num=layer_num,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
            layer_num_offset=0,
        )
        if get_pp_group().is_last_rank:
            self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        self.atom_config = atom_config
        self.do_not_compile = atom_config.compilation_config.level in [
            CompilationLevel.NO_COMPILATION,
            CompilationLevel.DYNAMO_AS_IS,
        ]
        if not self.do_not_compile:
            TorchCompileWrapperWithCustomDispatcher.__init__(
                self,
                vllm_config=atom_config,
                compilation_level=atom_config.compilation_config.level,
            )

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
            if get_pp_group().is_first_rank:
                if inputs_embeds is not None:
                    hidden_states = inputs_embeds
                else:
                    hidden_states = self.get_input_embeddings(input_ids)
                residual = None
            else:
                assert intermediate_tensors is not None
                hidden_states = intermediate_tensors["hidden_states"]
                residual = intermediate_tensors["residual"]

            for layer in self.layers[self.start_layer : self.end_layer]:
                hidden_states, residual = layer(
                    positions, hidden_states, residual, **model_kwargs
                )

            if not get_pp_group().is_last_rank:
                return IntermediateTensors(
                    {"hidden_states": hidden_states, "residual": residual}
                )
            hidden_states, _ = self.norm(hidden_states, residual)
            return hidden_states


class _AtomQwen3NextForCausalLMSglang(_CoreQwen3NextForCausalLM):
    """Plugin-local ATOM Qwen3-next model that reuses the core implementation shape."""

    def __init__(self, atom_config, prefix: str = ""):
        torch.nn.Module.__init__(self)
        config = atom_config.hf_config
        quant_config = atom_config.quant_config

        self.atom_config = atom_config
        self.config = config
        self.quant_config = quant_config
        self.packed_modules_mapping = dict(type(self).packed_modules_mapping)
        if self.quant_config.global_quant_config.quant_dtype == torch.bfloat16:
            self.packed_modules_mapping["in_proj_qkvz"] = ("in_proj_qkvzba", "qkvz")
            self.packed_modules_mapping["in_proj_ba"] = ("in_proj_qkvzba", "ba")
        if getattr(config, "mlp_only_layers", []):
            self.packed_modules_mapping["gate_up_proj"] = ["gate_proj", "up_proj"]

        self.model = Qwen3NextSglangModel(
            atom_config=atom_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **kwargs,
        )


class Qwen3NextForCausalLM(_AtomCausalLMBaseForSglang):
    """ATOM-backed Qwen3-next for SGLang using plugin-local subclasses only."""

    atom_model_cls = _AtomQwen3NextForCausalLMSglang


EntryClass = Qwen3NextForCausalLM
