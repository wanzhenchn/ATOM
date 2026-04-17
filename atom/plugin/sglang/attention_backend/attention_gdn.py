# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.


import torch

from atom.model_ops.attentions.gdn_attn import GDNAttentionMetadata
from atom.model_ops.attention_gdn import GatedDeltaNet as AtomGatedDeltaNet
from atom.utils.forward_context import ForwardContext


class GatedDeltaNet(AtomGatedDeltaNet):
    """SGLang adapter over the shared ATOM GDN implementation."""

    def _resolve_runtime_state(
        self,
        forward_context: ForwardContext,
    ) -> tuple[GDNAttentionMetadata | None, torch.Tensor | None, torch.Tensor | None]:
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return None, None, None

        # SGLang bridge path: `kv_cache_data[layer_{i}]` already points at the
        # persistent shadow cache allocated in `gdn_forward_helper`, so there should
        # be no per-request layout conversion left here.
        gdn_metadata = getattr(attn_metadata, "gdn_metadata", None)
        if gdn_metadata is None:
            return None, None, None

        kv_cache_data = getattr(forward_context, "kv_cache_data", None) or {}
        layer_kv = kv_cache_data.get(f"layer_{self.layer_num}")
        if layer_kv is None:
            return None, None, None

        conv_state = layer_kv.k_cache
        if getattr(layer_kv, "mamba_conv_layout", "atom") == "sglang_rowmajor":
            assert conv_state.stride(-2) == 1, (
                "SGLang GDN bridge is expected to provide an ATOM-layout shadow cache "
                "with stride(-2)==1."
            )
        else:
            conv_state = conv_state.transpose(-1, -2)
        return gdn_metadata, conv_state, layer_kv.v_cache
