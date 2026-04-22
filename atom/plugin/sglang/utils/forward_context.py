"""Shared SGLang forward metadata helpers for ATOM plugin mode."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import ClassVar, Optional, Union

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors


@dataclass(frozen=True)
class SGLangForwardBatchMetadata:
    """Small context object for one SGLang model forward."""

    forward_batch: Optional[ForwardBatch]
    pp_proxy_tensors: Optional[PPProxyTensors] = None
    save_kv_cache: bool = True
    _current: ClassVar[ContextVar[Optional["SGLangForwardBatchMetadata"]]] = ContextVar(
        "atom_sglang_current_forward_batch_metadata",
        default=None,
    )

    @classmethod
    def current(cls) -> Optional["SGLangForwardBatchMetadata"]:
        """Return metadata bound to the current forward context, if any."""
        return cls._current.get()

    @classmethod
    def current_forward_batch(cls):
        """Convenience accessor for the current SGLang forward batch."""
        metadata = cls.current()
        return None if metadata is None else metadata.forward_batch

    @classmethod
    def build(
        cls,
        forward_batch: Optional[
            Union[ForwardBatch, "SGLangForwardBatchMetadata"]
        ] = None,
        *,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        save_kv_cache: bool | None = None,
    ) -> Optional["SGLangForwardBatchMetadata"]:
        """Build metadata from explicit inputs or fall back to current context."""
        if isinstance(forward_batch, cls):
            return forward_batch
        if forward_batch is None and pp_proxy_tensors is None and save_kv_cache is None:
            return cls.current()
        return cls(
            forward_batch=forward_batch,
            pp_proxy_tensors=pp_proxy_tensors,
            save_kv_cache=True if save_kv_cache is None else save_kv_cache,
        )

    @classmethod
    @contextmanager
    def bind(cls, metadata: Optional["SGLangForwardBatchMetadata"]):
        """Temporarily bind metadata so nested calls can read it from context."""
        token = cls._current.set(metadata)
        try:
            yield metadata
        finally:
            cls._current.reset(token)

    @staticmethod
    def to_intermediate_tensors(intermediate_tensors, metadata):
        """Convert proxy tensors in metadata into ATOM intermediate tensors."""
        if intermediate_tensors is not None or metadata is None:
            return intermediate_tensors
        pp_proxy_tensors = metadata.pp_proxy_tensors
        if pp_proxy_tensors is None:
            return intermediate_tensors
        tensors = getattr(pp_proxy_tensors, "tensors", None)
        if tensors is None:
            return intermediate_tensors
        from atom.models.utils import IntermediateTensors

        return IntermediateTensors(dict(tensors))
