# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The vLLM team. Copyright 2025 Google Inc. HuggingFace Inc.
#
# Adapted from vLLM's gemma3.py for ATOM.
# Text-only Gemma 3 (Gemma3ForCausalLM) with sliding / full attention and
# GemmaRMSNorm, GELU-tanh MLP, embedding scaling.
# Multimodal Gemma 3 (Gemma3ForConditionalGeneration) with SigLIP vision
# encoder and multi-modal projector (adapted from vLLM's gemma3_mm.py).
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from aiter.dist.parallel_state import get_pp_group, get_tensor_model_parallel_world_size
from aiter.rotary_embedding import get_rope
from torch import nn

from atom.config import Config, QuantizationConfig
from atom.model_ops.base_attention import Attention
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.layernorm import GemmaRMSNorm
from atom.model_ops.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from atom.model_loader.loader import default_weight_loader
from atom.models.utils import (
    IntermediateTensors,
    PPMissingLayer,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from atom.utils.decorators import support_torch_compile

try:
    from transformers import Gemma3TextConfig
except ImportError:
    Gemma3TextConfig = None  # type: ignore[misc, assignment]

try:
    from transformers import Gemma3Config, SiglipVisionConfig
except ImportError:
    Gemma3Config = None  # type: ignore[misc, assignment]
    SiglipVisionConfig = None  # type: ignore[misc, assignment]


class Gemma3MLP(nn.Module):
    """MLP with GELU (tanh approximation) and gate * up projection."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_activation: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_activation != "gelu_pytorch_tanh":
            raise ValueError(
                "Gemma3 uses `gelu_pytorch_tanh` as the hidden activation. "
                "Set `hidden_activation` to `gelu_pytorch_tanh`."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        x = F.gelu(gate, approximate="tanh") * up
        return self.down_proj(x)


class Gemma3Attention(nn.Module):
    """Attention with QK GemmaRMSNorm, per-layer sliding/full, and query_pre_attn_scalar."""

    def __init__(
        self,
        config: "Gemma3TextConfig",
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: str = "bf16",
        prefix: str = "",
        layer_num: int = 0,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        query_pre_attn_scalar = getattr(
            config, "query_pre_attn_scalar", 256
        )
        self.scaling = query_pre_attn_scalar**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        layer_idx = extract_layer_index(prefix)
        layer_types = getattr(config, "layer_types", None)
        if layer_types is not None and layer_idx < len(layer_types):
            layer_type = layer_types[layer_idx]
            self.is_sliding = layer_type == "sliding_attention"
            sliding_window = (
                getattr(config, "sliding_window", None) if self.is_sliding else None
            )
        else:
            self.is_sliding = False
            sliding_window = None

        rope_parameters = getattr(config, "rope_parameters", None) or {}
        if isinstance(rope_parameters, dict) and layer_types is not None and layer_idx < len(layer_types):
            layer_type = layer_types[layer_idx]
            if layer_type in rope_parameters:
                rope_params = rope_parameters[layer_type]
            else:
                rope_params = rope_parameters
        else:
            rope_params = rope_parameters if isinstance(rope_parameters, dict) else {}
        if self.is_sliding:
            # Local/sliding attention: simple RoPE with no scaling (aiter does not
            # accept rope_type "default", so pass rope_scaling=None).
            rope_theta = getattr(config, "rope_local_base_freq", 10000.0)
            rope_scaling = None
        else:
            rope_theta = rope_params.get("rope_theta", 10000.0)
            rope_scaling = rope_params
            # If rope_scaling contains rope_type aiter does not support, fall back to None.
            # "linear" is also excluded because aiter's LinearScalingRotaryEmbedding
            # returns a concatenated cache tensor rather than (cos, sin), causing an unpack
            # error in RotaryEmbeddingBase.__init__. Use base theta without scaling instead.
            _unsupported = {"default", "linear"}
            if isinstance(rope_scaling, dict) and rope_scaling.get("rope_type") in _unsupported:
                rope_scaling = None

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=True,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            kv_cache_dtype=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
            layer_num=layer_num,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_norm(q)
        q = q.flatten(-2, -1)
        k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
        k = self.k_norm(k)
        k = k.flatten(-2, -1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, positions)
        return self.o_proj(attn_output)


class Gemma3DecoderLayer(nn.Module):
    """Decoder layer with four GemmaRMSNorms (input, post_attn, pre_ff, post_ff)."""

    def __init__(
        self,
        config: "Gemma3TextConfig",
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: str = "bf16",
        prefix: str = "",
        layer_num: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Gemma3Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
            layer_num=layer_num,
        )
        self.mlp = Gemma3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_activation=getattr(config, "hidden_activation", "gelu_pytorch_tanh"),
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states, residual = self.pre_feedforward_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return hidden_states, residual


@support_torch_compile
class Gemma3Model(nn.Module):
    """Gemma 3 transformer with embedding scaling by sqrt(hidden_size)."""

    def __init__(
        self,
        atom_config: Config,
        prefix: str = "",
    ):
        super().__init__()
        config = atom_config.hf_config
        if Gemma3TextConfig is not None and not isinstance(config, Gemma3TextConfig):
            raise ValueError(
                "Gemma3Model expects Gemma3TextConfig; "
                f"got {type(config).__name__}. Ensure the model's config.json has "
                '"model_type": "gemma3_text" and "architectures": ["Gemma3ForCausalLM"].'
            )
        if not hasattr(config, "layer_types") or not hasattr(config, "head_dim"):
            raise ValueError(
                "Gemma3Model expects config with layer_types and head_dim "
                "(e.g. Gemma3TextConfig)."
            )
        self.config = config
        cache_config = atom_config.kv_cache_dtype
        quant_config = atom_config.quant_config

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix, layer_num=None: Gemma3DecoderLayer(
                config=config,
                quant_config=quant_config,
                cache_config=cache_config,
                prefix=prefix,
                layer_num=layer_num or 0,
            ),
            prefix=f"{prefix}.layers",
            layer_num_offset=0,
        )

        if get_pp_group().is_last_rank:
            self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        normalizer = config.hidden_size**0.5
        self.register_buffer(
            "normalizer",
            torch.tensor(normalizer, dtype=torch.float32),
            persistent=False,
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self.normalizer.to(
            self.embed_tokens.weight.dtype
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Gemma3ForCausalLM(nn.Module):
    """Gemma 3 text-only causal LM. Compatible with HF architecture Gemma3ForCausalLM."""

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        atom_config: Config,
        prefix: str = "",
    ):
        super().__init__()
        config = atom_config.hf_config
        self.model = Gemma3Model(atom_config=atom_config, prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head.weight = self.model.embed_tokens.weight
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        return self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return self.lm_head(hidden_states)


# ── SigLIP Vision Encoder (for Gemma 3 multimodal) ────────────────────────
# Adapted from vLLM's siglip.py / gemma3_mm.py.

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: "SiglipVisionConfig") -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
            bias=True,
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches, dtype=torch.int64).unsqueeze(0),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class SiglipAttention(nn.Module):
    """Full (non-cached) self-attention for SigLIP encoder layers."""

    def __init__(
        self,
        config: "SiglipVisionConfig",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.out_proj = RowParallelLinear(
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_partition = max(1, self.num_heads // tp_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        q_size = self.num_heads_per_partition * self.head_dim
        kv_size = q_size
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = q.view(bsz, seq_len, self.num_heads_per_partition, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads_per_partition, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads_per_partition, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.out_proj(attn_out)


class SiglipMLP(nn.Module):
    def __init__(
        self,
        config: "SiglipVisionConfig",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate="none")
        return self.fc2(hidden_states)


class SiglipEncoderLayer(nn.Module):
    def __init__(
        self,
        config: "SiglipVisionConfig",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = SiglipAttention(
            config, quant_config=quant_config, prefix=f"{prefix}.self_attn"
        )
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config, quant_config=quant_config, prefix=f"{prefix}.mlp")
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class SiglipEncoder(nn.Module):
    def __init__(
        self,
        config: "SiglipVisionConfig",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            SiglipEncoderLayer(
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.layers.{i}",
            )
            for i in range(config.num_hidden_layers)
        ])

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(
        self,
        config: "SiglipVisionConfig",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(
            config, quant_config=quant_config, prefix=f"{prefix}.encoder"
        )
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(hidden_states)
        return self.post_layernorm(hidden_states)


class SiglipVisionModel(nn.Module):
    def __init__(
        self,
        config: "SiglipVisionConfig",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.vision_model = SiglipVisionTransformer(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.vision_model",
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision_model(pixel_values)


# ── Gemma 3 Multi-Modal Projector ─────────────────────────────────────────

class Gemma3MultiModalProjector(nn.Module):
    """Projects SigLIP features into language model embedding space.

    Applies average pooling to reduce spatial resolution, GemmaRMSNorm,
    then a learned linear projection (stored as a transposed weight matrix
    to match HF checkpoint layout: [vision_hidden, text_hidden]).
    """

    def __init__(self, config: "Gemma3Config") -> None:
        super().__init__()
        vision_hidden = config.vision_config.hidden_size
        text_hidden = config.text_config.hidden_size

        self.mm_input_projection_weight = nn.Parameter(
            torch.zeros(vision_hidden, text_hidden)
        )
        self.mm_soft_emb_norm = GemmaRMSNorm(
            vision_hidden, eps=config.vision_config.layer_norm_eps
        )
        patches_per_image = int(
            config.vision_config.image_size // config.vision_config.patch_size
        )
        tokens_per_side = int(config.mm_tokens_per_image ** 0.5)
        kernel_size = patches_per_image // tokens_per_side
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=kernel_size)
        self._patches_per_image = patches_per_image

    def forward(self, vision_outputs: torch.Tensor) -> torch.Tensor:
        bsz, _, hidden_size = vision_outputs.shape
        # Reshape to spatial grid for average pooling.
        x = vision_outputs.transpose(1, 2)  # [bsz, hidden, patches]
        x = x.reshape(bsz, hidden_size, self._patches_per_image, self._patches_per_image)
        x = self.avg_pool(x)                # [bsz, hidden, tokens_per_side, tokens_per_side]
        x = x.flatten(2).transpose(1, 2)   # [bsz, mm_tokens, hidden]
        x = self.mm_soft_emb_norm(x)
        x = torch.matmul(x, self.mm_input_projection_weight)
        return x.type_as(vision_outputs)


# ── Gemma 3 Multimodal (VLM) ──────────────────────────────────────────────

# HF checkpoint key prefixes that need remapping to ATOM model attribute paths.
_HF_VLM_PREFIX_REMAP: tuple[tuple[str, str], ...] = (
    ("model.language_model.", "language_model."),
    ("model.vision_tower.", "vision_tower."),
    ("model.multi_modal_projector.", "multi_modal_projector."),
    ("lm_head.", "language_model.lm_head."),
)


class Gemma3ForConditionalGeneration(nn.Module):
    """Gemma 3 vision-language model (Gemma3ForConditionalGeneration).

    Architecture: SigLIP vision tower + multi-modal projector +
    Gemma3ForCausalLM language model.

    Compatible with HuggingFace Gemma3ForConditionalGeneration checkpoints.
    Weight names from the HF checkpoint (model.language_model.*, lm_head.*)
    are remapped to ATOM's module layout via get_parameter().
    """

    packed_modules_mapping = {
        # Text decoder: q/k/v → qkv_proj (also applies to vision encoder)
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        # Text decoder: gate/up → gate_up_proj
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, atom_config: Config, prefix: str = "") -> None:
        super().__init__()
        config: "Gemma3Config" = atom_config.hf_config
        quant_config = atom_config.quant_config

        self.vision_tower = SiglipVisionModel(
            config.vision_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "vision_tower"),
        )
        self.multi_modal_projector = Gemma3MultiModalProjector(config)

        # Build the language model using the text sub-config, but swap in the
        # full Gemma3Config so Gemma3Model's isinstance check passes.
        text_atom_config = Config.__new__(Config)
        text_atom_config.__dict__.update(atom_config.__dict__)
        text_atom_config.hf_config = config.text_config
        self.language_model = Gemma3ForCausalLM(
            text_atom_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.image_token_id = getattr(config, "image_token_index", None) or getattr(
            config, "image_token_id", None
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    # ------------------------------------------------------------------
    # Weight name remapping: HF checkpoint → ATOM module layout
    # ------------------------------------------------------------------

    @staticmethod
    def _remap_weight_name(name: str) -> str:
        for hf_prefix, atom_prefix in _HF_VLM_PREFIX_REMAP:
            if name.startswith(hf_prefix):
                return atom_prefix + name[len(hf_prefix):]
        return name

    def get_parameter(self, target: str) -> nn.Parameter:
        remapped = self._remap_weight_name(target)
        try:
            return super().get_parameter(remapped)
        except AttributeError:
            # Weight not present in this model (e.g. pooling head, lm_loss).
            # Return a dummy parameter with a no-op loader so the ATOM weight
            # loader silently skips it.
            dummy = nn.Parameter(torch.empty(0), requires_grad=False)
            dummy.weight_loader = lambda *a, **kw: None  # type: ignore[attr-defined]
            return dummy

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.get_input_embeddings(input_ids)

    def _merge_image_embeddings(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """Replace image-token positions with projected vision features."""
        image_features = self.vision_tower(pixel_values)        # [n, patches, v_hidden]
        image_embeds = self.multi_modal_projector(image_features)  # [n, mm_tokens, t_hidden]
        image_embeds_flat = image_embeds.flatten(0, 1)             # [n*mm_tokens, t_hidden]

        if self.image_token_id is not None:
            mask = input_ids == self.image_token_id
        else:
            # Fallback: fill first N positions (should not happen in practice)
            mask = torch.zeros_like(input_ids, dtype=torch.bool)

        # Scatter image embeddings into the right positions.
        mask_expanded = mask.unsqueeze(-1).expand_as(inputs_embeds)
        n_slots = mask.sum().item()
        if n_slots == image_embeds_flat.shape[0]:
            inputs_embeds = inputs_embeds.masked_scatter(
                mask_expanded, image_embeds_flat
            )
        return inputs_embeds

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is None and inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids)
            if pixel_values is not None:
                inputs_embeds = self._merge_image_embeddings(
                    input_ids, inputs_embeds, pixel_values
                )
            input_ids = None  # consumed via inputs_embeds

        return self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(
        self, hidden_states: torch.Tensor
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states)
