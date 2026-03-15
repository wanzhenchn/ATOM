from collections.abc import (
    Callable,
)
from contextlib import ExitStack, contextmanager, nullcontext
from typing import (
    ClassVar,
    Literal,
    Protocol,
    TypeAlias,
    overload,
    runtime_checkable,
)

import torch
import torch.nn as nn
from .utils import common_prefix
from torch import Tensor

MultiModalEmbeddings: TypeAlias = list[Tensor] | Tensor | tuple[Tensor, ...]
"""
The output embeddings must be one of the following formats:

- A list or tuple of 2D tensors, where each tensor corresponds to
    each input multimodal data item (e.g, image).
- A single 3D tensor, with the batch dimension grouping the 2D tensors.
"""


def _require_is_multimodal(is_multimodal: Tensor | None) -> Tensor:
    """
    A helper function to be used in the context of
    [vllm.model_executor.models.interfaces.SupportsMultiModal.embed_input_ids][]
    to provide a better error message.
    """
    if is_multimodal is None:
        raise ValueError(
            "`embed_input_ids` now requires `is_multimodal` arg, "
            "please update your model runner according to "
            "https://github.com/vllm-project/vllm/pull/16229."
        )

    return is_multimodal


# Cache results of `SupportsMultiModal.get_language_model`
_language_model_by_module = dict[nn.Module, nn.Module]()


@runtime_checkable
class SupportsMultiModal(Protocol):
    """The interface required for all multi-modal models."""

    supports_multimodal: ClassVar[Literal[True]] = True
    """
    A flag that indicates this model supports multi-modal inputs.

    Note:
        There is no need to redefine this flag if this class is in the
        MRO of your model class.
    """

    supports_multimodal_raw_input_only: ClassVar[bool] = False
    """
    A flag that indicates this model supports multi-modal inputs and processes
    them in their raw form and not embeddings.
    """

    supports_encoder_tp_data: ClassVar[bool] = False
    """
    A flag that indicates whether this model supports
    `multimodal_config.mm_encoder_tp_mode="data"`.
    """

    requires_raw_input_tokens: ClassVar[bool] = False
    """
    A flag that indicates this model processes input id tokens
    in their raw form and not input embeddings.
    """

    _language_model_names: list[str] = []
    """
    Set internally by `_mark_language_model`.
    """

    _tower_model_names: list[str] = []
    """
    Set internally by `_mark_tower_model`.
    """

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        """
        Get the placeholder text for the `i`th `modality` item in the prompt.
        """
        ...

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """
        Returns multimodal embeddings generated from multimodal kwargs
        to be merged with text embeddings.

        Note:
            The returned multimodal embeddings must be in the same order as
            the appearances of their corresponding multimodal data item in the
            input prompt.
        """
        ...

    def get_language_model(self) -> nn.Module:
        """
        Returns the underlying language model used for text generation.

        This is typically the `torch.nn.Module` instance responsible for
        processing the merged multimodal embeddings and producing hidden states

        Returns:
            torch.nn.Module: The core language model component.
        """
        # Cached
        if self in _language_model_by_module:
            return _language_model_by_module[self]

        if self._language_model_names:
            mod = self
            for attr in common_prefix(
                [name.split(".") for name in self._language_model_names]
            ):
                if attr:
                    mod = getattr(mod, attr)

            if mod is not self and hasattr(mod, "embed_input_ids"):
                _language_model_by_module[self] = mod
                return mod

        # Fallback
        for mod in self.children():
            if hasattr(mod, "embed_input_ids"):
                _language_model_by_module[self] = mod
                return mod

        raise NotImplementedError(
            f"No language model found in {type(self).__name__}! "
            "You should initialize it via `_mark_language_model`."
        )

    @contextmanager
    def _mark_language_model(
        self,
        atom_config: nn.Module,
        *,
        targets: type[nn.Module] | tuple[type[nn.Module], ...] | None = None,
    ):
        """
        Mark each child module that was assigned to this model during this context
        as a language model component.

        Language model components are automatically skipped in `--mm-encoder-only`
        mode.

        If `targets` is set, instead include descendants that are an instance
        of `targets`, even if they aren't direct children.
        """
        from .utils import StageMissingLayer, collect_children, no_init_weights

        mm_config = atom_config.plugin_config.vllm_config.model_config.multimodal_config

        with collect_children(self, targets=targets) as children_names:  # noqa: SIM117
            with (
                no_init_weights(
                    self,
                    lambda mod: StageMissingLayer("language_model", mod),
                    targets=targets,
                )
                if mm_config.mm_encoder_only
                else nullcontext()
            ):
                yield

        self._language_model_names = children_names

    @contextmanager
    def _mark_tower_model(
        self,
        vllm_config: nn.Module,
        modalities: set[str] | str,
        *,
        targets: type[nn.Module] | tuple[type[nn.Module], ...] | None = None,
    ):
        """
        Mark each child module that was assigned to this model during this context
        as a tower model component.

        Tower model components are automatically skipped when `--limit-mm-per-prompt`
        is set to zero for all of their modalities.

        If `targets` is set, instead include descendants that are an instance
        of `targets`, even if they aren't direct children.
        """
        from .utils import StageMissingLayer, collect_children, no_init_weights

        if isinstance(modalities, str):
            modalities = {modalities}

        if modalities == {"image", "video"}:
            stage_name = "vision_tower"
        else:
            stage_name = "_".join([*modalities, "tower"])

        mm_config = vllm_config.model_config.multimodal_config

        with collect_children(self, targets=targets) as children_names:  # noqa: SIM117
            with (
                no_init_weights(
                    self,
                    lambda mod: StageMissingLayer(stage_name, mod),
                    targets=targets,
                )
                if all(mm_config.get_limit_per_prompt(m) == 0 for m in modalities)
                else nullcontext()
            ):
                yield

        self._tower_model_names = children_names

    @contextmanager
    def _mark_composite_model(
        self,
        vllm_config: nn.Module,
        *,
        language_targets: type[nn.Module] | tuple[type[nn.Module], ...],
        tower_targets: dict[str, type[nn.Module] | tuple[type[nn.Module], ...]],
    ):
        """
        Composite wrapper over `_mark_language_model` and
        `_mark_tower_model` by modality.
        """
        with ExitStack() as stack:
            stack.enter_context(
                self._mark_language_model(
                    vllm_config,
                    targets=language_targets,
                )
            )

            for modality, modality_targets in tower_targets.items():
                stack.enter_context(
                    self._mark_tower_model(
                        vllm_config,
                        modality,
                        targets=modality_targets,
                    )
                )

            yield

    def get_num_mm_encoder_tokens(self, num_image_tokens: int) -> int:
        """
        Implement this function to enable LoRA support
        for the tower module of the multi-modal model.
        Given the number of image tokens, output the number of
        multi-modal encoder tokens.
        """
        ...

    def get_num_mm_connector_tokens(self, num_vision_tokens: int) -> int:
        """
        Implement this function to enable LoRA support
        for the connector module of the multi-modal model.
        Given the number of vision tokens, output the number of
        multi-modal connector tokens.
        """
        ...

    @overload
    def embed_input_ids(self, input_ids: Tensor) -> Tensor: ...

    @overload
    def embed_input_ids(
        self,
        input_ids: Tensor,
        multimodal_embeddings: MultiModalEmbeddings,
        *,
        is_multimodal: torch.Tensor,
        handle_oov_mm_token: bool = False,
    ) -> Tensor: ...

    def _embed_text_input_ids(
        self,
        input_ids: Tensor,
        embed_input_ids: Callable[[Tensor], Tensor],
        *,
        is_multimodal: Tensor | None,
        handle_oov_mm_token: bool,
    ) -> Tensor:
        if handle_oov_mm_token and is_multimodal is not None:
            is_text = ~is_multimodal
            text_embeds = embed_input_ids(input_ids[is_text])

            return torch.empty(
                (input_ids.shape[0], text_embeds.shape[1]),
                dtype=text_embeds.dtype,
                device=text_embeds.device,
            ).masked_scatter_(is_text.unsqueeze_(-1), text_embeds)

        return embed_input_ids(input_ids)

    def embed_input_ids(
        self,
        input_ids: Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> Tensor:
        """
        Apply token embeddings to `input_ids`.

        If `multimodal_embeddings` is passed, scatter them into
        `input_ids` according to the mask `is_multimodal`.

        In case the multi-modal token IDs exceed the vocabulary size of
        the language model, you can set `handle_oov_mm_token=False`
        to avoid calling the language model's `embed_input_ids` method
        on those tokens. Note however that doing so increases memory usage
        as an additional buffer is needed to hold the input embeddings.
        """
        from .utils import _merge_multimodal_embeddings

        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.get_language_model().embed_input_ids,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=_require_is_multimodal(is_multimodal),
        )
