# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Build SGLang GDN forward context for ATOM plugin execution."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Iterator, Optional

import torch

from atom.config import KVCacheTensor, get_current_atom_config
from atom.plugin.sglang.utils.forward_context import SGLangForwardBatchMetadata
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


@dataclass(frozen=True)
class SGLangGDNForwardContext:
    """Precomputed ATOM forward-context state derived from SGLang metadata."""

    _atom_conv_cache_by_layer: ClassVar[
        dict[tuple[int, int, tuple[int, ...], torch.dtype, torch.device], torch.Tensor]
    ] = {}

    forward_batch: Any
    gdn_metadata: GDNAttentionMetadata
    kv_cache_data: Dict[str, KVCacheTensor]
    context: Context
    num_tokens: int

    @staticmethod
    def _linear_attn_backend(attn_backend: Any) -> Any:
        """Unwrap the actual GDN backend from hybrid attention wrappers."""
        return getattr(attn_backend, "linear_attn_backend", attn_backend)

    @classmethod
    def _get_atom_conv_cache(
        cls, pool: Any, layer_id: int, layer_cache: Any
    ) -> torch.Tensor:
        """Return a reusable ATOM-layout shadow cache for one SGLang GDN conv state.

        SGLang stores the conv cache as row-major `[slot, D, W]`, while ATOM's
        `causal_conv1d_*` kernels require the same logical shape with `stride(-2)==1`.
        A per-forward layout fix would work but adds extra copies on every request.

        To avoid that steady-state overhead, we keep one persistent shadow tensor in
        the ATOM-preferred layout and reuse it across requests. Once created, this
        shadow cache is exposed through the forward context as `KVCacheTensor.k_cache`,
        so the attention backend can usually consume it directly without any per-request
        row-major to ATOM-layout conversion.

        The cache key is based on the long-lived pool object plus logical `layer_id`
        and tensor spec, rather than `id(layer_cache)`. This matters because
        `mamba2_layer_cache(layer_id)` may return a fresh frozen wrapper object on
        each call, even though it still points to the same underlying runtime cache.
        Keying by `id(layer_cache)` would therefore create one shadow tensor per
        request and eventually leak VRAM under load, while `pool + layer_id`
        preserves one reusable shadow tensor per actual layer cache.
        """
        raw = layer_cache.conv[0]
        _, d, w = raw.shape
        need_stride = (d * w, 1, d)
        cache_key = (id(pool), layer_id, tuple(raw.shape), raw.dtype, raw.device)
        shadow = cls._atom_conv_cache_by_layer.get(cache_key)
        if (
            shadow is None
            or shadow.shape != raw.shape
            or shadow.stride() != need_stride
            or shadow.dtype != raw.dtype
            or shadow.device != raw.device
        ):
            # Allocate once in ATOM layout, then reuse it for subsequent forwards.
            shadow = torch.empty_strided(
                raw.shape,
                need_stride,
                dtype=raw.dtype,
                device=raw.device,
            )
            # Seed the shadow cache from the current SGLang cache contents. After
            # this, the shadow cache is treated as the source of truth for ATOM's
            # GDN path and is reused across requests.
            shadow.copy_(raw)
            cls._atom_conv_cache_by_layer[cache_key] = shadow
        return shadow

    @classmethod
    def _build_kv_cache_tensors(cls, forward_batch: Any) -> Dict[str, KVCacheTensor]:
        """Map SGLang mamba layer caches to ATOM KV cache entries."""
        pool = forward_batch.req_to_token_pool
        mamba_map = getattr(pool, "mamba_map", None)
        if mamba_map is None:
            return {}

        out: Dict[str, KVCacheTensor] = {}
        for layer_id in mamba_map:
            layer_cache = pool.mamba2_layer_cache(layer_id)
            conv = cls._get_atom_conv_cache(pool, layer_id, layer_cache)
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

    @staticmethod
    def _build_context(forward_batch: Any) -> tuple[Context, int]:
        """Build the ATOM forward-context view for one SGLang forward batch."""
        mode = forward_batch.forward_mode
        is_prefill = mode.is_prefill()
        num_tokens = (
            forward_batch.seq_lens_sum if mode.is_extend() else forward_batch.batch_size
        )
        return (
            Context(
                positions=forward_batch.positions,
                is_prefill=is_prefill,
                batch_size=forward_batch.batch_size,
                graph_bs=forward_batch.batch_size,
            ),
            num_tokens,
        )

    @staticmethod
    def _build_gdn_metadata(
        forward_batch: Any, linear_backend: Any
    ) -> Optional[GDNAttentionMetadata]:
        """Build GDN metadata from SGLang attention runtime metadata."""
        fm = getattr(linear_backend, "forward_metadata", None)
        if fm is None:
            return None

        mode = forward_batch.forward_mode
        if mode.is_target_verify():
            logger.warning(
                "SGLang GDN forward context: TARGET_VERIFY is not supported; GDN metadata skipped."
            )
            return None

        device = fm.query_start_loc.device
        idx = fm.mamba_cache_indices.to(dtype=torch.int32, device=device)
        bs = forward_batch.batch_size
        common_kwargs = dict(
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
            spec_query_start_loc=None,
            non_spec_query_start_loc=fm.query_start_loc,
            spec_state_indices_tensor=None,
            non_spec_state_indices_tensor=idx,
            spec_sequence_masks=None,
            spec_token_indx=None,
            non_spec_token_indx=None,
            num_accepted_tokens=None,
        )

        if mode.is_decode_or_idle():
            return GDNAttentionMetadata(
                num_prefills=0,
                num_prefill_tokens=0,
                num_decodes=bs,
                num_decode_tokens=bs,
                num_actual_tokens=bs,
                has_initial_state=None,
                nums_dict=None,
                batch_ptr=None,
                token_chunk_offset_ptr=None,
                **common_kwargs,
            )

        if mode.is_extend():
            seq_sum = forward_batch.seq_lens_sum
            epl = forward_batch.extend_prefix_lens
            has_initial_state = None if epl is None else epl > 0
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(fm.query_start_loc)
            )
            return GDNAttentionMetadata(
                num_prefills=bs,
                num_prefill_tokens=seq_sum,
                num_decodes=0,
                num_decode_tokens=0,
                num_actual_tokens=seq_sum,
                has_initial_state=has_initial_state,
                nums_dict=nums_dict,
                batch_ptr=batch_ptr,
                token_chunk_offset_ptr=token_chunk_offset_ptr,
                **common_kwargs,
            )

        logger.warning(
            "SGLang GDN forward context: unsupported forward_mode=%s; GDN metadata skipped.",
            mode,
        )
        return None

    @classmethod
    def build(
        cls, forward_batch_or_metadata: Any
    ) -> Optional["SGLangGDNForwardContext"]:
        """Build all ATOM GDN forward-context inputs from SGLang metadata."""
        metadata = SGLangForwardBatchMetadata.build(forward_batch_or_metadata)
        if metadata is None or metadata.forward_batch is None:
            return None

        forward_batch = metadata.forward_batch
        linear_backend = cls._linear_attn_backend(forward_batch.attn_backend)
        gdn_metadata = cls._build_gdn_metadata(forward_batch, linear_backend)
        if gdn_metadata is None:
            return None

        kv_cache_data = cls._build_kv_cache_tensors(forward_batch)
        if not kv_cache_data:
            return None

        context, num_tokens = cls._build_context(forward_batch)
        return cls(
            forward_batch=forward_batch,
            gdn_metadata=gdn_metadata,
            kv_cache_data=kv_cache_data,
            context=context,
            num_tokens=num_tokens,
        )

    @classmethod
    @contextmanager
    def bind(cls, forward_batch_or_metadata: Any) -> Iterator[None]:
        """Attach one precomputed GDN forward context to the current forward."""
        forward_context = cls.build(forward_batch_or_metadata)
        if forward_context is None:
            yield
            return

        prev_kv = _forward_kv_cache_context.kv_cache_data
        try:
            set_kv_cache_data(forward_context.kv_cache_data)
            attn_md = AttentionMetaData()
            attn_md.gdn_metadata = forward_context.gdn_metadata
            set_forward_context(
                attn_metadata=attn_md,
                atom_config=get_current_atom_config(),
                context=forward_context.context,
                num_tokens=forward_context.num_tokens,
            )
            yield
        finally:
            reset_forward_context()
            set_kv_cache_data(prev_kv if prev_kv is not None else {})
