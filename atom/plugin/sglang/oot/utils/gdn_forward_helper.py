# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Helpers to wire SGLang ForwardBatch + hybrid mamba pool into ATOM ForwardContext for GatedDeltaNet."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional

import torch

from atom.config import KVCacheTensor, get_current_atom_config
from atom.model_ops.attentions.gdn_attn import (
    GDNAttentionMetadata,
    compute_causal_conv1d_metadata,
)
from atom.utils.forward_context import (
    AttentionMetaData,
    Context,
    _forward_kv_cache_context,
    reset_forward_context,
    set_forward_context,
    set_kv_cache_data,
)

logger = logging.getLogger(__name__)


def _linear_attn_backend(attn_backend: Any) -> Any:
    """HybridLinearAttnBackend wraps the GDN (MambaAttnBackendBase) instance."""
    return getattr(attn_backend, "linear_attn_backend", attn_backend)


def _build_kv_cache_data(forward_batch: Any) -> Dict[str, KVCacheTensor]:
    pool = forward_batch.req_to_token_pool
    mamba_map = getattr(pool, "mamba_map", None)
    if mamba_map is None:
        return {}

    out: Dict[str, KVCacheTensor] = {}
    for layer_id in mamba_map:
        layer_cache = pool.mamba2_layer_cache(layer_id)
        conv = layer_cache.conv[0]
        temporal = layer_cache.temporal
        out[f"layer_{layer_id}"] = KVCacheTensor(
            layer_num=layer_id,
            k_cache=conv,
            v_cache=temporal,
            k_scale=None,
            v_scale=None,
            mamba_conv_layout="sglang_rowmajor",
        )
    return out


def _build_gdn_metadata(forward_batch: Any, linear_backend: Any) -> Optional[GDNAttentionMetadata]:
    fm = getattr(linear_backend, "forward_metadata", None)
    if fm is None:
        return None

    mode = forward_batch.forward_mode
    if mode.is_target_verify():
        logger.warning(
            "SGLang GDN forward helper: TARGET_VERIFY is not supported; GDN metadata skipped."
        )
        return None

    device = fm.query_start_loc.device
    idx = fm.mamba_cache_indices.to(dtype=torch.int32, device=device)

    if mode.is_decode_or_idle():
        bs = forward_batch.batch_size
        return GDNAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decodes=bs,
            num_decode_tokens=bs,
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
            num_actual_tokens=bs,
            has_initial_state=None,
            spec_query_start_loc=None,
            non_spec_query_start_loc=fm.query_start_loc,
            spec_state_indices_tensor=None,
            non_spec_state_indices_tensor=idx,
            spec_sequence_masks=None,
            spec_token_indx=None,
            non_spec_token_indx=None,
            num_accepted_tokens=None,
            nums_dict=None,
            batch_ptr=None,
            token_chunk_offset_ptr=None,
        )

    if mode.is_extend():
        bs = forward_batch.batch_size
        seq_sum = forward_batch.seq_lens_sum
        epl = forward_batch.extend_prefix_lens
        if epl is None:
            has_initial_state = None
        else:
            has_initial_state = epl > 0
        nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
            fm.query_start_loc
        )
        return GDNAttentionMetadata(
            num_prefills=bs,
            num_prefill_tokens=seq_sum,
            num_decodes=0,
            num_decode_tokens=0,
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
            num_actual_tokens=seq_sum,
            has_initial_state=has_initial_state,
            spec_query_start_loc=None,
            non_spec_query_start_loc=fm.query_start_loc,
            spec_state_indices_tensor=None,
            non_spec_state_indices_tensor=idx,
            spec_sequence_masks=None,
            spec_token_indx=None,
            non_spec_token_indx=None,
            num_accepted_tokens=None,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )

    logger.warning(
        "SGLang GDN forward helper: unsupported forward_mode=%s; GDN metadata skipped.",
        mode,
    )
    return None


@contextmanager
def sglang_gdn_forward_context(forward_batch: Any):
    """Populate ATOM ForwardContext + kv_cache_data for one model forward."""
    if forward_batch is None:
        yield
        return

    linear_be = _linear_attn_backend(forward_batch.attn_backend)
    gmd = _build_gdn_metadata(forward_batch, linear_be)
    if gmd is None:
        yield
        return

    kv = _build_kv_cache_data(forward_batch)
    if not kv:
        yield
        return

    prev_kv = _forward_kv_cache_context.kv_cache_data
    try:
        set_kv_cache_data(kv)
        attn_md = AttentionMetaData()
        attn_md.gdn_metadata = gmd
        atom_config = get_current_atom_config()
        positions = forward_batch.positions
        is_prefill = forward_batch.forward_mode.is_prefill()
        ctx = Context(
            positions=positions,
            is_prefill=is_prefill,
            batch_size=forward_batch.batch_size,
            graph_bs=forward_batch.batch_size,
        )
        num_tok = (
            forward_batch.seq_lens_sum
            if forward_batch.forward_mode.is_extend()
            else forward_batch.batch_size
        )
        set_forward_context(
            attn_metadata=attn_md,
            atom_config=atom_config,
            context=ctx,
            num_tokens=num_tok,
        )
        yield
    finally:
        reset_forward_context()
        set_kv_cache_data(prev_kv if prev_kv is not None else {})
