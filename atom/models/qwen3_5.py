from collections.abc import Iterable

import torch
from einops import rearrange
from torch import nn

import triton
import triton.language as tl

from aiter.dist.parallel_state import get_tensor_model_parallel_rank
from atom.config import QuantizationConfig, Config

from atom.model_ops.topK import is_rocm_aiter_fusion_shared_expert_enabled
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
from atom.plugin.prepare import is_vllm
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
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    extract_layer_index,
)


@triton.jit
def shard_qkvzba_kernel(
    qkvzba_ptr,  # [num_tokens,  (2 * num_k_heads * head_k_dim + 2 * num_v_heads * head_v_dim + 2 * num_v_heads)]
    qkv_ptr,  # [num_tokens, (2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim)]
    z_ptr,  # [num_tokens, num_v_heads * head_v_dim]
    b_ptr,  # [num_tokens, num_v_heads]
    a_ptr,  # [num_tokens, num_v_heads]
    num_k_heads: tl.constexpr,
    num_v_heads: tl.constexpr,
    head_k_dim: tl.constexpr,
    head_v_dim: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_id = tl.program_id(1)
    QKVZ_TOTAL_SIZE = 2 * num_k_heads * head_k_dim + 2 * num_v_heads * head_v_dim
    QKV_DIM_SIZE = 2 * head_k_dim + head_v_dim * num_v_heads // num_k_heads
    Z_DIM_SIZE = head_v_dim * num_v_heads // num_k_heads
    BA_TOTAL_SIZE = 2 * num_v_heads
    KV_HEAD_RATIO: tl.constexpr = num_v_heads // num_k_heads
    ROW_SIZE = QKVZ_TOTAL_SIZE + BA_TOTAL_SIZE
    if head_id == num_k_heads:  # ba
        load_ptr = qkvzba_ptr + token_id * ROW_SIZE + QKVZ_TOTAL_SIZE
        b_offset = (
            tl.arange(0, num_v_heads) // KV_HEAD_RATIO * KV_HEAD_RATIO * 2
            + tl.arange(0, num_v_heads) % KV_HEAD_RATIO
        )
        a_offset = b_offset + KV_HEAD_RATIO
        b_val = tl.load(load_ptr + b_offset)
        a_val = tl.load(load_ptr + a_offset)
        store_offset = tl.arange(0, num_v_heads)
        tl.store(b_ptr + token_id * num_v_heads + store_offset, b_val)
        tl.store(a_ptr + token_id * num_v_heads + store_offset, a_val)
    else:
        base_ptr = qkvzba_ptr + token_id * ROW_SIZE + head_id * QKV_DIM_SIZE
        qkv_base_ptr = qkv_ptr + token_id * (
            2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
        )
        k_dim_offset = tl.arange(0, head_k_dim)
        v_dim_offset = tl.arange(0, head_v_dim)
        q_val = tl.load(base_ptr + k_dim_offset)
        q_store_ptr = qkv_base_ptr + head_id * head_k_dim
        tl.store(q_store_ptr + k_dim_offset, q_val)

        k_val = tl.load(base_ptr + head_k_dim + k_dim_offset)
        k_store_ptr = qkv_base_ptr + num_k_heads * head_k_dim + head_id * head_k_dim
        tl.store(k_store_ptr + k_dim_offset, k_val)

        for sub_head in tl.static_range(0, KV_HEAD_RATIO):
            v_val = tl.load(
                base_ptr + 2 * head_k_dim + sub_head * head_v_dim + v_dim_offset
            )
            v_store_ptr = (
                qkv_base_ptr
                + 2 * num_k_heads * head_k_dim
                + (head_id * KV_HEAD_RATIO + sub_head) * head_v_dim
            )
            tl.store(v_store_ptr + v_dim_offset, v_val)

        base_ptr_z = (
            qkvzba_ptr
            + token_id * ROW_SIZE
            + num_k_heads * QKV_DIM_SIZE
            + head_id * Z_DIM_SIZE
        )
        for sub_head in tl.static_range(0, KV_HEAD_RATIO):
            z_val = tl.load(base_ptr_z + sub_head * head_v_dim + v_dim_offset)
            z_store_ptr = (
                z_ptr
                + token_id * num_v_heads * head_v_dim
                + (head_id * KV_HEAD_RATIO + sub_head) * head_v_dim
            )
            tl.store(z_store_ptr + v_dim_offset, z_val)


GDN = Qwen3NextGatedDeltaNet
if is_vllm():
    from vllm.config import get_current_vllm_config
    from vllm.model_executor.layers.mamba.abstract import MambaBase
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateShapeCalculator,
        MambaStateDtypeCalculator,
        MambaStateCopyFunc,
        MambaStateCopyFuncCalculator,
    )
    from vllm.distributed import get_tensor_model_parallel_rank

    class Qwen3NextGatedDeltaNetVllm(Qwen3NextGatedDeltaNet, MambaBase):

        def __init__(
            self, config, quant_config=None, speculative_config=None, prefix=""
        ):
            super().__init__(config, quant_config, speculative_config, prefix)
            self.model_config = config.plugin_config.vllm_config.model_config
            self.cache_config = config.plugin_config.vllm_config.cache_config
            self.tp_rank = get_tensor_model_parallel_rank()
            compilation_config = get_current_vllm_config().compilation_config
            if prefix in compilation_config.static_forward_context:
                raise ValueError(f"Duplicate layer name: {prefix}")
            print(f"register layer {prefix} to static forward context for Mamba")
            compilation_config.static_forward_context[prefix] = self

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

        def create_qkvzba_proj(self):
            self.in_proj_qkvz = self.create_qkvz_proj(
                hidden_size=self.hidden_size,
                key_dim=self.key_dim,
                value_dim=self.value_dim,
                quant_config=self.quant_config,
                prefix=f"{self.prefix}.in_proj_qkvz",
            )

            self.in_proj_ba = self.create_ba_proj(
                hidden_size=self.hidden_size,
                num_v_heads=self.num_v_heads,
                quant_config=self.quant_config,
                prefix=f"{self.prefix}.in_proj_ba",
            )

        def shard_qkvzba(
            self,
            qkvzba: torch.Tensor,
            num_k_heads: int,
            num_v_heads: int,
            head_k_dim: int,
            head_v_dim: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            num_tokens, dtype, device = qkvzba.shape[0], qkvzba.dtype, qkvzba.device
            mixed_qkv = torch.empty(
                [
                    num_tokens,
                    2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim,
                ],
                dtype=dtype,
                device=device,
            )
            z = torch.empty(
                [num_tokens, num_v_heads, head_v_dim], dtype=dtype, device=device
            )
            b = torch.empty([num_tokens, num_v_heads], dtype=dtype, device=device)
            a = torch.empty([num_tokens, num_v_heads], dtype=dtype, device=device)
            grid = (num_tokens, num_k_heads + 1)
            shard_qkvzba_kernel[grid](
                qkvzba,
                mixed_qkv,
                z,
                b,
                a,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
            )
            return mixed_qkv, z, b, a

        def shard_qkvzba_native(
            self,
            qkvzba: torch.Tensor,
            num_k_heads: int,
            num_v_heads: int,
            head_k_dim: int,
            head_v_dim: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            key_dim = num_k_heads * head_k_dim
            value_dim = num_v_heads * head_v_dim
            qkv_size = key_dim * 2 + value_dim
            z_size = value_dim
            b_size = num_v_heads
            a_size = num_v_heads
            mixed_qkv, z, b, a = qkvzba.split(
                [qkv_size, z_size, b_size, a_size], dim=-1
            )
            return (
                mixed_qkv.contiguous(),
                z.contiguous(),
                b.contiguous(),
                a.contiguous(),
            )

        def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
            return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
                self.model_config.dtype,
                self.cache_config.mamba_cache_dtype,
                self.cache_config.mamba_ssm_cache_dtype,
            )

        def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
            return MambaStateShapeCalculator.gated_delta_net_state_shape(
                self.tp_size,
                self.num_k_heads,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
                self.conv_kernel_size,
                self.num_spec,
            )

    GDN = Qwen3NextGatedDeltaNetVllm


class Qwen3_5GatedDeltaNet(GDN):
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
        mixed_qkvz = self.in_proj_qkvz(hidden_states)
        qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        z_size = self.value_dim // self.tp_size
        mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        ba = self.in_proj_ba(hidden_states)
        b, a = ba.chunk(2, dim=-1)

        b = b.contiguous()
        a = a.contiguous()
        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        # Note: we should not use torch.empty here like other attention backends,
        # see discussions in https://github.com/vllm-project/vllm/pull/28182
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        core_attn_out = self.attn(mixed_qkv, b, a, core_attn_out)

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        z_shape_og = z.shape
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens] = self.out_proj(core_attn_out)


class Qwen3_5DecoderLayer(Qwen3NextDecoderLayer):
    def __init__(
        self,
        atom_config,
        layer_type: str,
        prefix: str = "",
        layer_num: int = 0,
    ) -> None:
        super(Qwen3NextDecoderLayer, self).__init__()

        config = atom_config.hf_config.text_config
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
                atom_config.hf_config.text_config,
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
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
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
        config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig = (
            atom_config.hf_config.text_config
        )

        self.config = config
        self.config.n_shared_experts = 1
        self.config.n_routed_experts = self.config.num_experts

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


class Qwen3_5ForCausalLMBase(nn.Module):

    def __init__(self, atom_config: Config, prefix: str = ""):
        config: Qwen3_5Config = atom_config.hf_config.text_config
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
        print("hidden states shape: ", hidden_states.shape)
        return self.lm_head(hidden_states)


class Qwen3_5ForCausalLM(Qwen3_5ForCausalLMBase):
    pass


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLMBase):
    def __init__(self, atom_config: Config, prefix: str = ""):
        super().__init__(atom_config=atom_config, prefix=prefix)
        config: Qwen3_5TextConfig = atom_config.hf_config.text_config
        self.config = config
        self.config.n_shared_experts = 1
        self.config.n_routed_experts = self.config.num_experts
        # set MoE hyperparameters

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


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
                    prefix=maybe_prefix(prefix, "language_model"),
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
        # hf_to_vllm_mapper = hf_to_atom_mapper

        def __init__(self, atom_config: Config, prefix: str = "model"):
            # protocols have not __init__ method, so we need to use nn.Module.__init__
            nn.Module.__init__(self)
            self.atom_config = atom_config
            vllm_config = atom_config.plugin_config.vllm_config
            config: Qwen3_5MoeConfig = atom_config.hf_config
            quant_config = vllm_config.quant_config
            multimodal_config = vllm_config.model_config.multimodal_config

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

            # set MoE hyperparameters
            # self.set_moe_parameters()

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

        def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
            return self.language_model.get_expert_mapping()

        def embed_multimodal(self, **kwargs):
            return super().embed_multimodal(**kwargs)

    @MULTIMODAL_REGISTRY.register_processor(
        Qwen3VLMultiModalProcessor,
        info=Qwen3_5MoeProcessingInfo,
        dummy_inputs=Qwen3VLDummyInputsBuilder,
    )
    class Qwen3_5ForConditionalGeneration(ATOMForConditionalGeneration, IsHybrid):

        packed_modules_mapping = {
            "q_proj": ("qkv_proj", "q"),
            "k_proj": ("qkv_proj", "k"),
            "v_proj": ("qkv_proj", "v"),
            "gate_proj": ("gate_up_proj", 0),
            "up_proj": ("gate_up_proj", 1),
            "in_proj_qkv": ("in_proj_qkvz", (0, 1, 2)),
            "in_proj_z": ("in_proj_qkvz", 3),
            "in_proj_b": ("in_proj_ba", 0),
            "in_proj_a": ("in_proj_ba", 1),
            ".gate.": (".gate.", 0),
            "shared_expert_gate": ("gate", 1),
        }

        hf_to_atom_mapper = WeightsMapper(
            orig_to_new_prefix={
                # "model.visual.": "visual.",
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
