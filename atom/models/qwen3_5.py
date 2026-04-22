from collections.abc import Iterable

import torch
from torch import nn


from atom.config import QuantizationConfig, Config

from atom.model_ops.topK import is_rocm_aiter_fusion_shared_expert_enabled
from atom.model_ops.utils import atom_parameter
from atom.utils.decorators import support_torch_compile

from atom.model_ops.embed_head import VocabParallelEmbedding, ParallelLMHead
from atom.model_config.qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig

from atom.model_config.qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
)
from atom.model_ops.moe import FusedMoE
from atom.model_ops.linear import (
    MergedColumnParallelLinear,
)
from atom.plugin.prepare import is_sglang, is_vllm
from atom.model_ops.layernorm import GemmaRMSNorm as Qwen3_5RMSNorm
from atom.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextGatedDeltaNet,
    Qwen3NextModel,
    Qwen3NextSparseMoeBlock,
    Qwen3NextMLP,
    Qwen3NextDecoderLayer,
)

from atom.models.utils import (
    IntermediateTensors,
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    extract_layer_index,
)
from atom.model_ops.split_chunk import (
    fused_split_chunk_zeros,
    fused_split_chunk_zeros_qwen3_5_qkvzba,
)

if is_vllm():
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateShapeCalculator,
        MambaStateDtypeCalculator,
        MambaStateCopyFunc,
        MambaStateCopyFuncCalculator,
    )


def get_qwen3_5_text_config(atom_config: Config):
    hf_config = atom_config.hf_config
    return hf_config.text_config if hasattr(hf_config, "text_config") else hf_config


# Qwen3.5 MoE models have some checkpoints where expert weights are fused together in BF16 format, so we need special handling to load those weights into our per-expert parameters.
def detect_fused_expert_format(weight_name: str) -> bool:
    """Detect if weight is from fused expert checkpoint (BF16 format)."""
    # Qwen3.5 BF16 has: experts.gate_up_proj, experts.down_proj
    # Qwen3.5 FP8 has: experts.0.gate_proj, experts.0.up_proj, experts.0.down_proj
    return "experts.gate_up_proj" in weight_name or (
        "experts.down_proj" in weight_name
        and ".experts." in weight_name
        and weight_name.count(".experts.") == 1
    )


def get_fused_expert_mapping() -> list[tuple[str, str, str]]:
    """Return mapping for fused expert weights (BF16 format)."""
    # (param_name, weight_name, shard_id)
    return [
        ("experts.w13_weight", "experts.gate_up_proj", "w1"),  # Will be chunked
        ("experts.w2_weight", "experts.down_proj", "w2"),
    ]


def load_fused_expert_weights(
    original_name: str,
    name: str,
    params_dict: dict,
    loaded_weight: torch.Tensor,
    shard_id: str,
    num_experts: int,
) -> bool:
    """Load fused expert weights (BF16 format) into per-expert parameters.

    Args:
        original_name: Original weight name from checkpoint (e.g., "experts.gate_up_proj")
        name: Mapped parameter name (e.g., "experts.w13_weight")
        params_dict: Model parameters dict
        loaded_weight: The weight tensor to load
        shard_id: Shard identifier ("w1", "w2", "w3")
        num_experts: Number of experts

    Returns:
        True if weights were loaded successfully
    """
    param = params_dict[name]
    weight_loader = param.weight_loader
    loaded_local_expert = False

    # Special handling for gate_up_proj: chunk into gate and up
    if "gate_up_proj" in original_name:
        gate_weight, up_weight = loaded_weight.chunk(2, dim=-2)
        # Load gate part (w1)
        for expert_id in range(num_experts):
            try:
                success = weight_loader(
                    param,
                    gate_weight[expert_id],
                    name,
                    "w1",
                    expert_id,
                    return_success=True,
                )
                if success:
                    loaded_local_expert = True
            except TypeError:
                weight_loader(param, gate_weight[expert_id], name, "w1", expert_id)
                loaded_local_expert = True
        # Load up part (w3)
        for expert_id in range(num_experts):
            try:
                success = weight_loader(
                    param,
                    up_weight[expert_id],
                    name,
                    "w3",
                    expert_id,
                    return_success=True,
                )
                if success:
                    loaded_local_expert = True
            except TypeError:
                weight_loader(param, up_weight[expert_id], name, "w3", expert_id)
                loaded_local_expert = True
    else:
        # down_proj or other weights - no chunking
        for expert_id in range(num_experts):
            try:
                success = weight_loader(
                    param,
                    loaded_weight[expert_id],
                    name,
                    shard_id,
                    expert_id,
                    return_success=True,
                )
                if success:
                    loaded_local_expert = True
            except TypeError:
                weight_loader(
                    param, loaded_weight[expert_id], name, shard_id, expert_id
                )
                loaded_local_expert = True

    return loaded_local_expert


class Qwen3_5GatedDeltaNet(Qwen3NextGatedDeltaNet):

    def create_qkvz_proj(
        self,
        hidden_size: int,
        key_dim: int,
        value_dim: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:

        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[key_dim, key_dim, value_dim, value_dim],
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def create_ba_proj(
        self,
        hidden_size: int,
        num_v_heads: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:
        # Qwen3.5 has separate in_proj_b and in_proj_a weights in the
        # checkpoint, which are loaded into the fused in_proj_ba parameter
        # via stacked_params_mapping with shard_id 0 and 1 respectively.
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[num_v_heads] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def create_qkvzba_proj(self, quant_config, prefix):
        if self.quant_config.global_quant_config.quant_dtype == torch.bfloat16:
            self.in_proj_qkvzba = MergedColumnParallelLinear(
                input_size=self.hidden_size,
                output_sizes=[
                    self.key_dim,
                    self.key_dim,
                    self.value_dim,
                    self.value_dim,
                    self.num_v_heads,
                    self.num_v_heads,
                ],
                bias=False,
                quant_config=quant_config,
                prefix=prefix,
            )
        else:
            self.in_proj_qkvz = self.create_qkvz_proj(
                hidden_size=self.hidden_size,
                key_dim=self.key_dim,
                value_dim=self.value_dim,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj_qkvz",
            )

            self.in_proj_ba = self.create_ba_proj(
                hidden_size=self.hidden_size,
                num_v_heads=self.num_v_heads,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj_ba",
            )

    def fix_query_key_value_ordering(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
    ):
        raise NotImplementedError(
            "Qwen3.5 Series dont need to fix query key value ordering"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        num_tokens = hidden_states.size(0)

        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        if hasattr(self, "in_proj_qkvzba"):
            qkvzba = self.in_proj_qkvzba(hidden_states)
            k_heads_after_tp = self.num_k_heads // self.tp_size
            v_heads_after_tp = self.num_v_heads // self.tp_size
            mixed_qkv, z, b, a, core_attn_out = fused_split_chunk_zeros_qwen3_5_qkvzba(
                qkvzba,
                k_heads_after_tp,
                v_heads_after_tp,
                self.head_k_dim,
                self.head_v_dim,
            )
        else:
            mixed_qkvz = self.in_proj_qkvz(hidden_states)
            ba = self.in_proj_ba(hidden_states)

            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            num_v_heads_tp = self.num_v_heads // self.tp_size

            mixed_qkv, z, b, a, core_attn_out = fused_split_chunk_zeros(
                mixed_qkvz, ba, qkv_size, z_size, self.head_v_dim, num_v_heads_tp
            )

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        core_attn_out = self.attn(mixed_qkv, b, a, core_attn_out)

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        core_attn_out, maybe_scale = self.norm(core_attn_out, z)
        output[:num_tokens] = self.out_proj(core_attn_out, x_scale=maybe_scale)


class Qwen3_5DecoderLayer(Qwen3NextDecoderLayer):
    def __init__(
        self,
        atom_config,
        layer_type: str,
        prefix: str = "",
        layer_num: int = 0,
    ) -> None:
        super(Qwen3NextDecoderLayer, self).__init__()

        config = get_qwen3_5_text_config(atom_config)
        quant_config = atom_config.quant_config
        speculative_config = atom_config.speculative_config

        self.layer_type = layer_type
        self.layer_idx = layer_num

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(
                atom_config,
                quant_config=quant_config,
                speculative_config=speculative_config,
                prefix=f"{prefix}.linear_attn",
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                atom_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        # NOTE: Determine the MLP type based on the model type
        # Qwen3.5 use all layers for MLP / Qwen3.5-MoE use sparse MoE blocks
        if config.model_type == "qwen3_5_moe_text":
            self.mlp = Qwen3NextSparseMoeBlock(
                config,
                atom_config.quant_config,
                prefix=f"{prefix}.mlp",
            )
        elif config.model_type == "qwen3_5_text":
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            raise ValueError(f"Invalid model_type {config.model_type}")

        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = atom_parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                )
            )
            self.ffn_layer_scale = atom_parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                )
            )


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3_5Model(Qwen3NextModel):
    def __init__(self, *, atom_config, prefix: str = ""):
        super(Qwen3NextModel, self).__init__()
        config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig = get_qwen3_5_text_config(
            atom_config
        )

        self.config = config

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str, layer_num: int):
            return Qwen3_5DecoderLayer(
                atom_config=atom_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
                layer_num=layer_num,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen3_5ForCausalLMBase(nn.Module):

    def __init__(self, atom_config: Config, prefix: str = ""):
        config: Qwen3_5MoeTextConfig = get_qwen3_5_text_config(atom_config)
        self.atom_config = atom_config

        self.quant_config = atom_config.quant_config

        super().__init__()
        self.config = config
        self.model = Qwen3_5Model(
            atom_config=atom_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.lm_head(hidden_states)


class Qwen3_5ForCausalLM(Qwen3_5ForCausalLMBase):
    pass


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLMBase):
    def __init__(self, atom_config: Config, prefix: str = ""):
        config: Qwen3_5MoeTextConfig = get_qwen3_5_text_config(atom_config)
        config.n_shared_experts = 1
        config.n_routed_experts = config.num_experts
        super().__init__(atom_config=atom_config, prefix=prefix)
        self.config = config

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts
            + (
                self.config.n_shared_experts
                if is_rocm_aiter_fusion_shared_expert_enabled()
                else 0
            ),
        )


class Qwen3_5ForConditionalGenerationTextOnly(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkv": ("in_proj_qkvz", (0, 1, 2)),
        "in_proj_z": ("in_proj_qkvz", 3),
        "in_proj_b": ("in_proj_ba", 0),
        "in_proj_a": ("in_proj_ba", 1),
        ".gate.": (".gate.", 0),
        "shared_expert_gate": ("gate", 1),
    }
    weights_mapping = {
        "model.language_model.": "language_model.model.",
        "lm_head.": "language_model.lm_head.",
    }
    quant_exclude_name_mapping = {
        "model.language_model.": "model.",
    }
    skip_weight_prefixes = ["model.visual."]

    def __init__(self, atom_config: Config, prefix: str = ""):
        super().__init__()
        self.config = atom_config.hf_config
        self.visual = PPMissingLayer()
        self.language_model = Qwen3_5ForCausalLM(atom_config=atom_config, prefix="")
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.model.get_input_embeddings(input_ids)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: object,
    ):
        return self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)


class Qwen3_5MoeForConditionalGenerationTextOnly(
    Qwen3_5ForConditionalGenerationTextOnly
):
    def __init__(self, atom_config: Config, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = atom_config.hf_config
        self.visual = PPMissingLayer()
        self.language_model = Qwen3_5MoeForCausalLM(atom_config=atom_config, prefix="")
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def detect_fused_expert_format(self, weight_name: str) -> bool:
        """Detect if weight is from fused expert checkpoint (BF16 format)."""
        # Qwen3.5 BF16 has: experts.gate_up_proj, experts.down_proj
        # Qwen3.5 FP8 has: experts.0.gate_proj, experts.0.up_proj, experts.0.down_proj
        return detect_fused_expert_format(weight_name)

    def get_fused_expert_mapping(self) -> list[tuple[str, str, str]]:
        """Return mapping for fused expert weights (BF16 format)."""
        # (param_name, weight_name, shard_id)
        return get_fused_expert_mapping()

    def load_fused_expert_weights(
        self,
        original_name: str,
        name: str,
        params_dict: dict,
        loaded_weight: torch.Tensor,
        shard_id: str,
        num_experts: int,
    ) -> bool:
        """Load fused expert weights (BF16 format) into per-expert parameters.

        Args:
            original_name: Original weight name from checkpoint (e.g., "experts.gate_up_proj")
            name: Mapped parameter name (e.g., "experts.w13_weight")
            params_dict: Model parameters dict
            loaded_weight: The weight tensor to load
            shard_id: Shard identifier ("w1", "w2", "w3")
            num_experts: Number of experts

        Returns:
            True if weights were loaded successfully
        """
        return load_fused_expert_weights(
            original_name,
            name,
            params_dict,
            loaded_weight,
            shard_id,
            num_experts,
        )

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.language_model.get_expert_mapping()


########################################################
# Qwen3_5-Dense
########################################################

# ConditionalGeneration model scope should only works on plugin mode
if is_vllm():
    from vllm.config import VllmConfig
    from vllm.model_executor.models.qwen3_vl import (
        Qwen3VLMultiModalProcessor,
        Qwen3VLDummyInputsBuilder,
        Qwen3_VisionTransformer,
    )
    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5ProcessingInfo,
        Qwen3_5MoeProcessingInfo,
    )

    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5ForConditionalGeneration as vLLMQwen3_5,
    )
    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5MoeForConditionalGeneration as vLLMQwen3_5Moe,
    )
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.model_executor.models.interfaces import IsHybrid
    from atom.model_loader.loader import load_model_in_plugin_mode, WeightsMapper
    from atom.plugin.vllm.model_wrapper import ATOMForConditionalGeneration

    @MULTIMODAL_REGISTRY.register_processor(
        Qwen3VLMultiModalProcessor,
        info=Qwen3_5ProcessingInfo,
        dummy_inputs=Qwen3VLDummyInputsBuilder,
    )
    class Qwen3_5ForConditionalGeneration_(vLLMQwen3_5):
        packed_modules_mapping = {
            "q_proj": ("qkv_proj", "q"),
            "k_proj": ("qkv_proj", "k"),
            "v_proj": ("qkv_proj", "v"),
            "gate_proj": ("gate_up_proj", 0),
            "up_proj": ("gate_up_proj", 1),
            "gate_up_proj": ["gate_proj", "up_proj"],  # BF16 models: fused → split
            "in_proj_qkv": ("in_proj_qkvz", (0, 1, 2)),
            "in_proj_z": ("in_proj_qkvz", 3),
            "in_proj_b": ("in_proj_ba", 0),
            "in_proj_a": ("in_proj_ba", 1),
            ".gate.": (".gate.", 0),
            "shared_expert_gate": ("gate", 1),
        }

        hf_to_atom_mapper = WeightsMapper(
            orig_to_new_prefix={
                "model.visual.": "visual.",
                "lm_head.": "language_model.lm_head.",
                "model.language_model.": "language_model.model.",
            }
        )
        hf_to_vllm_mapper = hf_to_atom_mapper

        def __init__(self, atom_config: Config, prefix: str = "model"):
            # protocols have not __init__ method, so we need to use nn.Module.__init__
            nn.Module.__init__(self)
            config: Qwen3_5Config = atom_config.hf_config
            vllm_config = atom_config.plugin_config.vllm_config
            quant_config = vllm_config.quant_config
            multimodal_config = vllm_config.model_config.multimodal_config
            self.atom_config = atom_config
            if (
                self.atom_config.quant_config.global_quant_config.quant_dtype
                == torch.bfloat16
            ):
                self.packed_modules_mapping.pop("in_proj_qkv")
                self.packed_modules_mapping.pop("in_proj_b")
                self.packed_modules_mapping.pop("in_proj_a")
                self.packed_modules_mapping["in_proj_qkv"] = (
                    "in_proj_qkvzba",
                    (0, 1, 2),
                )
                self.packed_modules_mapping["in_proj_z"] = ("in_proj_qkvzba", (3))
                self.packed_modules_mapping["in_proj_b"] = ("in_proj_qkvzba", (4))
                self.packed_modules_mapping["in_proj_a"] = ("in_proj_qkvzba", (5))

            self.config = config
            self.multimodal_config = multimodal_config
            self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
            self.video_pruning_rate = multimodal_config.video_pruning_rate
            self.is_multimodal_pruning_enabled = (
                multimodal_config.is_multimodal_pruning_enabled()
            )
            with self._mark_tower_model(vllm_config, {"image", "video"}):
                self.visual = Qwen3_VisionTransformer(
                    config.vision_config,
                    norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "visual"),
                )

            with self._mark_language_model(vllm_config):
                self.language_model = Qwen3_5ForCausalLM(
                    atom_config=atom_config,
                    prefix=maybe_prefix("", "language_model"),
                )
            self.make_empty_intermediate_tensors = (
                self.language_model.make_empty_intermediate_tensors
            )

        def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
            # load weights in plugin mode and discard passed weights generator
            # here prefix is "model." because Qwen3ForCausalLM is constructed in model
            # wrapper class, so the name of loaded weights are prefixed with "model.".
            # The vLLM will check the name of the loaded weights to make sure all the
            # weights are loaded correctly
            loaded_weights_record = load_model_in_plugin_mode(
                model=self,
                config=self.atom_config,
                prefix="model.",
                weights_mapper=self.hf_to_atom_mapper,
            )
            return loaded_weights_record

    ########################################################
    # Qwen3_5-MoE
    ########################################################

    class Qwen3_5MoeForConditionalGeneration_(vLLMQwen3_5Moe):
        packed_modules_mapping = {
            "q_proj": ("qkv_proj", "q"),
            "k_proj": ("qkv_proj", "k"),
            "v_proj": ("qkv_proj", "v"),
            "gate_proj": ("gate_up_proj", 0),
            "up_proj": ("gate_up_proj", 1),
            "gate_up_proj": ["gate_proj", "up_proj"],  # BF16 models: fused → split
            "in_proj_qkv": ("in_proj_qkvz", (0, 1, 2)),
            "in_proj_z": ("in_proj_qkvz", 3),
            "in_proj_b": ("in_proj_ba", 0),
            "in_proj_a": ("in_proj_ba", 1),
            ".gate.": (".gate.", 0),
            "shared_expert_gate": ("gate", 1),
        }

        hf_to_atom_mapper = WeightsMapper(
            orig_to_new_prefix={
                "model.visual.": "visual.",
                "lm_head.": "language_model.lm_head.",
                "model.language_model.": "language_model.model.",
            }
        )

        def __init__(self, atom_config: Config, prefix: str = "model"):
            # protocols have not __init__ method, so we need to use nn.Module.__init__
            nn.Module.__init__(self)
            self.atom_config = atom_config
            vllm_config = atom_config.plugin_config.vllm_config
            atom_config.hf_config.text_config.n_shared_experts = 1
            atom_config.hf_config.text_config.n_routed_experts = (
                atom_config.hf_config.text_config.num_experts
            )
            config: Qwen3_5MoeConfig = atom_config.hf_config
            quant_config = vllm_config.quant_config
            multimodal_config = vllm_config.model_config.multimodal_config
            if (
                self.atom_config.quant_config.global_quant_config.quant_dtype
                == torch.bfloat16
            ):
                self.packed_modules_mapping.pop("in_proj_qkv")
                self.packed_modules_mapping.pop("in_proj_b")
                self.packed_modules_mapping.pop("in_proj_a")
                self.packed_modules_mapping["in_proj_qkv"] = (
                    "in_proj_qkvzba",
                    (0, 1, 2),
                )
                self.packed_modules_mapping["in_proj_z"] = ("in_proj_qkvzba", (3))
                self.packed_modules_mapping["in_proj_b"] = ("in_proj_qkvzba", (4))
                self.packed_modules_mapping["in_proj_a"] = ("in_proj_qkvzba", (5))

            self.config = config
            self.multimodal_config = multimodal_config
            self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
            self.video_pruning_rate = multimodal_config.video_pruning_rate
            self.is_multimodal_pruning_enabled = (
                multimodal_config.is_multimodal_pruning_enabled()
            )

            with self._mark_tower_model(vllm_config, {"image", "video"}):
                self.visual = Qwen3_VisionTransformer(
                    config.vision_config,
                    norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "visual"),
                )

            with self._mark_language_model(vllm_config):
                self.language_model = Qwen3_5MoeForCausalLM(
                    atom_config=atom_config, prefix=maybe_prefix("", "language_model")
                )

            self.make_empty_intermediate_tensors = (
                self.language_model.make_empty_intermediate_tensors
            )

        def detect_fused_expert_format(self, weight_name: str) -> bool:
            """Detect if weight is from fused expert checkpoint (BF16 format)."""
            # Qwen3.5 BF16 has: experts.gate_up_proj, experts.down_proj
            # Qwen3.5 FP8 has: experts.0.gate_proj, experts.0.up_proj, experts.0.down_proj
            return detect_fused_expert_format(weight_name)

        def get_fused_expert_mapping(self) -> list[tuple[str, str, str]]:
            """Return mapping for fused expert weights (BF16 format)."""
            # (param_name, weight_name, shard_id)
            return get_fused_expert_mapping()

        def load_fused_expert_weights(
            self,
            original_name: str,
            name: str,
            params_dict: dict,
            loaded_weight: torch.Tensor,
            shard_id: str,
            num_experts: int,
        ) -> bool:
            """Load fused expert weights (BF16 format) into per-expert parameters.

            Args:
                original_name: Original weight name from checkpoint (e.g., "experts.gate_up_proj")
                name: Mapped parameter name (e.g., "experts.w13_weight")
                params_dict: Model parameters dict
                loaded_weight: The weight tensor to load
                shard_id: Shard identifier ("w1", "w2", "w3")
                num_experts: Number of experts

            Returns:
                True if weights were loaded successfully
            """
            return load_fused_expert_weights(
                original_name,
                name,
                params_dict,
                loaded_weight,
                shard_id,
                num_experts,
            )

        def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
            # load weights in plugin mode and discard passed weights generator
            # here prefix is "model." because Qwen3ForCausalLM is constructed in model
            # wrapper class, so the name of loaded weights are prefixed with "model.".
            # The vLLM will check the name of the loaded weights to make sure all the
            # weights are loaded correctly
            loaded_weights_record = load_model_in_plugin_mode(
                model=self,
                config=self.atom_config,
                prefix="model.",
                weights_mapper=self.hf_to_atom_mapper,
                load_fused_expert_weights_fn=self.load_fused_expert_weights,
            )
            return loaded_weights_record

        def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
            return self.language_model.get_expert_mapping()

        def embed_multimodal(self, **kwargs):
            return super().embed_multimodal(**kwargs)

    @MULTIMODAL_REGISTRY.register_processor(
        Qwen3VLMultiModalProcessor,
        info=Qwen3_5ProcessingInfo,
        dummy_inputs=Qwen3VLDummyInputsBuilder,
    )
    class Qwen3_5ForConditionalGeneration(ATOMForConditionalGeneration, IsHybrid):

        packed_modules_mapping = {
            "q_proj": ("qkv_proj", "q"),
            "k_proj": ("qkv_proj", "k"),
            "v_proj": ("qkv_proj", "v"),
            "gate_proj": ("gate_up_proj", 0),
            "up_proj": ("gate_up_proj", 1),
            "gate_up_proj": ["gate_proj", "up_proj"],  # BF16 models: fused → split
            "in_proj_qkv": ("in_proj_qkvz", (0, 1, 2)),
            "in_proj_z": ("in_proj_qkvz", 3),
            "in_proj_b": ("in_proj_ba", 0),
            "in_proj_a": ("in_proj_ba", 1),
            ".gate.": (".gate.", 0),
            "shared_expert_gate": ("gate", 1),
        }

        hf_to_atom_mapper = WeightsMapper(
            orig_to_new_prefix={
                "lm_head.": "language_model.lm_head.",
                "model.language_model.": "language_model.model.",
            }
        )
        hf_to_vllm_mapper = hf_to_atom_mapper

        def embed_multimodal(self, **kwargs):
            return self.model.embed_multimodal(**kwargs)

        @classmethod
        def get_placeholder_str(cls, modality: str, i: int) -> str | None:
            if modality.startswith("image"):
                return "<|vision_start|><|image_pad|><|vision_end|>"
            if modality.startswith("video"):
                return "<|vision_start|><|video_pad|><|vision_end|>"

            raise ValueError("Only image or video modality is supported")

        @classmethod
        def get_mamba_state_dtype_from_config(
            cls,
            vllm_config: "VllmConfig",
        ) -> tuple[torch.dtype, torch.dtype]:
            return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
                vllm_config.model_config.dtype,
                vllm_config.cache_config.mamba_cache_dtype,
                vllm_config.cache_config.mamba_ssm_cache_dtype,
            )

        @classmethod
        def get_mamba_state_shape_from_config(
            cls, vllm_config: "VllmConfig"
        ) -> tuple[tuple[int, int], tuple[int, int]]:
            parallel_config = vllm_config.parallel_config
            hf_config = vllm_config.model_config.hf_text_config
            tp_size = parallel_config.tensor_parallel_size
            num_spec = (
                vllm_config.speculative_config.num_speculative_tokens
                if vllm_config.speculative_config
                else 0
            )
            return MambaStateShapeCalculator.gated_delta_net_state_shape(
                tp_size,
                hf_config.linear_num_key_heads,
                hf_config.linear_num_value_heads,
                hf_config.linear_key_head_dim,
                hf_config.linear_value_head_dim,
                hf_config.linear_conv_kernel_dim,
                num_spec,
            )

        @classmethod
        def get_mamba_state_copy_func(
            cls,
        ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
            return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

        def load_weights(
            self,
            weights: Iterable[tuple[str, torch.Tensor]],
        ) -> set[str]:
            return self.model.load_weights(weights)

    @MULTIMODAL_REGISTRY.register_processor(
        Qwen3VLMultiModalProcessor,
        info=Qwen3_5MoeProcessingInfo,
        dummy_inputs=Qwen3VLDummyInputsBuilder,
    )
    class Qwen3_5MoeForConditionalGeneration(Qwen3_5ForConditionalGeneration, IsHybrid):

        def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
            return self.model.get_expert_mapping()


if is_sglang():
    from typing import Any, Optional

    from aiter.dist.parallel_state import get_pp_group as _aiter_pp_group
    from atom.model_loader.loader import WeightsMapper
    from atom.plugin.config import generate_atom_config_for_plugin_mode
    from atom.plugin.sglang.utils.forward_context import SGLangForwardBatchMetadata
    from atom.plugin.sglang.utils.gdn_context import (
        SGLangGDNForwardContext,
    )
    from atom.plugin.sglang.utils.loader import (
        SGLangLoaderPatch,
        load_model_in_sglang_plugin_mode,
    )
    from sglang.srt.layers.quantization.base_config import (
        QuantizationConfig as SGLangQuantizationConfig,
    )
    from sglang.srt.model_executor.forward_batch_info import (
        ForwardBatch,
        PPProxyTensors,
    )
    from sglang.srt.models.qwen3_5 import (
        Qwen3_5ForConditionalGeneration as _SglQwen35VL,
        Qwen3_5MoeForConditionalGeneration as _SglQwen35MoeVL,
    )

    _PACKED_MODULES_MAPPING = {
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

    def _apply_bf16_in_proj_mapping(mapping: dict, atom_config: Any) -> dict:
        """Map split in-proj checkpoint keys onto the BF16 fused qkvzba layout."""
        if atom_config.quant_config.global_quant_config.quant_dtype != torch.bfloat16:
            return mapping

        mapping.pop("in_proj_qkvz", None)
        mapping.pop("in_proj_ba", None)
        mapping["in_proj_qkvzba"] = ("in_proj_qkvzba", None)
        mapping["in_proj_qkv"] = ("in_proj_qkvzba", (0, 1, 2))
        mapping["in_proj_z"] = ("in_proj_qkvzba", 3)
        mapping["in_proj_b"] = ("in_proj_qkvzba", 4)
        mapping["in_proj_a"] = ("in_proj_qkvzba", 5)
        return mapping

    _QWEN35_SGLANG_VL_HF_MAPPER = WeightsMapper(
        orig_to_new_substr={"attn.qkv.": "attn.qkv_proj."},
        orig_to_new_prefix={
            "model.visual.": "visual.",
            "model.language_model.model.": "model.",
            "model.language_model.lm_head.": "lm_head.",
            "model.language_model.": "model.",
            "lm_head.": "lm_head.",
        },
    )

    def _patch_qwen35_moe_text_for_sparse_moe_block(hf_config: Any) -> None:
        tc = getattr(hf_config, "text_config", None)
        if tc is None:
            tc = hf_config
        mt = getattr(tc, "model_type", "") or ""
        if "qwen3_5" not in mt or "moe" not in mt:
            return
        tc.n_shared_experts = 1
        if hasattr(tc, "num_experts"):
            tc.n_routed_experts = tc.num_experts

    def _remap_qwen35_quant_config_for_sglang_plugin(atom_config: Any) -> None:
        atom_config.quant_config.remap_layer_name(
            atom_config.hf_config,
            packed_modules_mapping=_apply_bf16_in_proj_mapping(
                dict(_PACKED_MODULES_MAPPING), atom_config
            ),
            weights_mapper=_QWEN35_SGLANG_VL_HF_MAPPER,
        )

    def _maybe_to_intermediate_tensors(pp_proxy_tensors: Optional[PPProxyTensors]):
        if pp_proxy_tensors is None:
            return None
        tensors = getattr(pp_proxy_tensors, "tensors", None)
        if tensors is None:
            return None
        return IntermediateTensors(dict(tensors))

    def _forward_qwen35_decoder_stack(
        decoder_stack: Qwen3_5Model,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        input_deepstack_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | IntermediateTensors:
        if input_deepstack_embeds is None or input_deepstack_embeds.numel() == 0:
            return decoder_stack(
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
            )

        if _aiter_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = decoder_stack.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        hs = decoder_stack.config.hidden_size
        for local_i, layer in enumerate(
            decoder_stack.layers[decoder_stack.start_layer : decoder_stack.end_layer]
        ):
            hidden_states, residual = layer(positions, hidden_states, residual)
            layer_num = decoder_stack.start_layer + local_i
            if (
                input_deepstack_embeds is not None
                and input_deepstack_embeds.numel() > 0
                and layer_num < 3
            ):
                sep = hs * layer_num
                hidden_states.add_(input_deepstack_embeds[:, sep : sep + hs])

        if not _aiter_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = decoder_stack.norm(hidden_states, residual)
        return hidden_states

    _QWEN35_SGLANG_LANGUAGE_MODEL_STACKS: dict[
        type[Qwen3_5ForCausalLMBase], type[nn.Module]
    ] = {}

    def _get_qwen35_language_model_stack_cls(
        atom_model_cls: type[Qwen3_5ForCausalLMBase],
    ) -> type[nn.Module]:
        stack_cls = _QWEN35_SGLANG_LANGUAGE_MODEL_STACKS.get(atom_model_cls)
        if stack_cls is not None:
            return stack_cls

        class _AtomQwen35LanguageModelAdapter(atom_model_cls):
            def __init__(
                self,
                config: Any,
                quant_config: Optional[SGLangQuantizationConfig] = None,
                prefix: str = "",
            ) -> None:
                atom_config = generate_atom_config_for_plugin_mode(config)
                _remap_qwen35_quant_config_for_sglang_plugin(atom_config)
                super().__init__(atom_config=atom_config, prefix="")
                self.quant_config = quant_config or atom_config.quant_config
                self.atom_config = atom_config

            @property
            def embed_tokens(self) -> nn.Module:
                return self.model.embed_tokens

            @property
            def layers(self) -> nn.Module:
                return self.model.layers

            @property
            def norm(self) -> nn.Module:
                return self.model.norm

            @property
            def start_layer(self) -> int:
                return self.model.start_layer

            @property
            def end_layer(self) -> int:
                return self.model.end_layer

            @property
            def vocab_size(self) -> int:
                return self.model.vocab_size

            def get_input_embeddings(
                self, input_ids: Optional[torch.Tensor] = None
            ) -> torch.Tensor | nn.Module:
                if input_ids is None:
                    return self.model.embed_tokens
                return self.model.get_input_embeddings(input_ids)

            def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
                return self.model.embed_input_ids(input_ids)

            def forward(
                self,
                input_ids: Optional[torch.Tensor],
                positions: torch.Tensor,
                intermediate_tensors: IntermediateTensors | None = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                forward_batch: Optional[ForwardBatch] = None,
                input_embeds: Optional[torch.Tensor] = None,
                pp_proxy_tensors: Optional[PPProxyTensors] = None,
                input_deepstack_embeds: Optional[torch.Tensor] = None,
                save_kv_cache: bool = True,
                **kwargs: Any,
            ):
                kwargs = dict(kwargs)
                if inputs_embeds is None:
                    inputs_embeds = input_embeds
                if inputs_embeds is None:
                    inputs_embeds = kwargs.pop("inputs_embeds", None)
                if intermediate_tensors is None:
                    intermediate_tensors = _maybe_to_intermediate_tensors(
                        pp_proxy_tensors
                    )
                del kwargs
                metadata = SGLangForwardBatchMetadata.build(
                    forward_batch,
                    pp_proxy_tensors=pp_proxy_tensors,
                    save_kv_cache=save_kv_cache,
                )
                with SGLangGDNForwardContext.bind(metadata):
                    out = _forward_qwen35_decoder_stack(
                        self.model,
                        input_ids,
                        positions,
                        intermediate_tensors=intermediate_tensors,
                        inputs_embeds=inputs_embeds,
                        input_deepstack_embeds=input_deepstack_embeds,
                    )
                if isinstance(out, IntermediateTensors):
                    return PPProxyTensors(dict(out.tensors))
                return out

        _QWEN35_SGLANG_LANGUAGE_MODEL_STACKS[atom_model_cls] = (
            _AtomQwen35LanguageModelAdapter
        )
        return _AtomQwen35LanguageModelAdapter

    class _Qwen3_5ConditionalGenerationSglangBase:
        packed_modules_mapping = _PACKED_MODULES_MAPPING
        atom_language_model_cls: type[Qwen3_5ForCausalLMBase] = Qwen3_5ForCausalLM

        def get_sglang_loader_patch(self) -> SGLangLoaderPatch:
            return SGLangLoaderPatch(weights_mapper=_QWEN35_SGLANG_VL_HF_MAPPER)

        def _prepare_sglang_root_config(self, config: Any) -> None:
            """Allow subclasses to patch the SGLang root config before init."""

        def __init__(
            self,
            config: Any,
            quant_config: Optional[SGLangQuantizationConfig] = None,
            prefix: str = "",
        ) -> None:
            self._prepare_sglang_root_config(config)
            stack_cls = _get_qwen35_language_model_stack_cls(
                type(self).atom_language_model_cls
            )
            super().__init__(
                config,
                quant_config,
                prefix,
                language_model_cls=stack_cls,
            )
            # SGLang's Qwen3VL wrapper stores the injected ATOM language model on
            # `self.model`, not `self.language_model`. Keep both names available
            # so the rest of ATOM's qwen3.5 helpers can stay unchanged.
            self.language_model = self.model
            self.atom_config = self.model.atom_config
            self.packed_modules_mapping = _apply_bf16_in_proj_mapping(
                dict(type(self).packed_modules_mapping), self.atom_config
            )
            self.make_empty_intermediate_tensors = (
                self.model.make_empty_intermediate_tensors
            )
            if quant_config is not None and hasattr(
                quant_config, "packed_modules_mapping"
            ):
                quant_config.packed_modules_mapping = self.packed_modules_mapping

        def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
            del weights
            return load_model_in_sglang_plugin_mode(
                model=self,
                config=self.atom_config,
                prefix="",
            )

    class Qwen3_5ForConditionalGeneration(
        _Qwen3_5ConditionalGenerationSglangBase, _SglQwen35VL
    ):
        pass

    class Qwen3_5MoeForConditionalGeneration(
        _Qwen3_5ConditionalGenerationSglangBase, _SglQwen35MoeVL
    ):
        atom_language_model_cls = Qwen3_5MoeForCausalLM

        def _prepare_sglang_root_config(self, config: Any) -> None:
            _patch_qwen35_moe_text_for_sparse_moe_block(config)

        def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
            return self.model.get_expert_mapping()

        def load_fused_expert_weights(
            self,
            original_name: str,
            name: str,
            params_dict: dict,
            loaded_weight: torch.Tensor,
            shard_id: str,
            num_experts: int,
        ) -> bool:
            return load_fused_expert_weights(
                original_name,
                name,
                params_dict,
                loaded_weight,
                shard_id,
                num_experts,
            )

        def get_sglang_loader_patch(self) -> SGLangLoaderPatch:
            return SGLangLoaderPatch(
                weights_mapper=_QWEN35_SGLANG_VL_HF_MAPPER,
                load_fused_expert_weights_fn=self.load_fused_expert_weights,
            )
