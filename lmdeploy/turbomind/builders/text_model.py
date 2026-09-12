# Copyright (c) OpenMMLab. All rights reserved.

from ..linear import round_up_output_groups
from ._base import _CPP_TO_TORCH, Builder, BuiltModule, ParallelGroup, SplitSide


class TextModelBuilder(Builder):
    """Builder for the root ModelWeight.

    Constructs a ModelWeight via ``_tm.create_module(ModelWeightConfig)``
    on each context (inherited Builder machinery), then attaches it to
    externally-owned ``ModelRoot`` sentinel handles as their
    ``text_model`` child during ``build()``.

    Owns the ``tok_embeddings`` (Tensor param) and ``output`` (LinearWeight
    child) commits on the ModelWeight via ``add_token_embeds`` /
    ``add_token_embeds_quant`` / ``add_token_embeds_quant4`` and
    ``add_lm_head`` / ``add_lm_head_shared``, driven by
    ``add_embedding_and_head``.
    """

    def __init__(self, config, ctx, *, root_handles,
                 tp: ParallelGroup, vocab_size):
        super().__init__(config, ctx)
        self.tp = tp
        self.config.tp_size = tp.size
        self._root_handles = root_handles
        self._vocab_size = vocab_size

    def build(self) -> BuiltModule:
        """Create ModelWeight via _tm.create_module (via super), then attach
        each per-GPU ModelWeight handle to its sentinel root via
        add_child_raw."""
        built = super().build()
        for i, (root, text_model) in enumerate(
                zip(self._root_handles, built.handles)):
            with self._ctx.devices[i]:
                root.add_child_raw('text_model', text_model)
        return built

    def add_token_embeds(self, tensor):
        """Commit the raw embedding lookup as the ``tok_embeddings`` root
        param.

        Shards along hidden (output) dim by ``self.tp.size``. No vocab padding —
        embedding lookup never indexes past ``vocab - 1``.
        """
        self._add_tensor('tok_embeddings', tensor,
                            split_side=SplitSide.OUTPUT)

    def add_token_embeds_quant(self, table, scales):
        """Commit an int8 embedding table + group scales.

        Both shard along hidden (OUTPUT). Requires hidden % (tp*group) == 0.
        """
        hidden = table.shape[-1]
        group = hidden // scales.shape[-1]
        if hidden % (self.tp.size * group) != 0:
            raise ValueError(f'int8 embedding: hidden={hidden} not divisible by '
                             f'tp*group={self.tp.size * group}')
        self._add_tensor('tok_embeddings', table, split_side=SplitSide.OUTPUT)
        self._add_tensor('tok_embeddings_scale', scales,
                         split_side=SplitSide.OUTPUT)

    def add_token_embeds_quant4(self, table, scales, zeros):
        """Commit an int4 embedding table (packed nibbles) + scales + zeros.

        ``table`` is uint8 [vocab, hidden/2]; the logical hidden is
        ``table.shape[-1] * 2``. All three shard along hidden (OUTPUT).
        """
        hidden = table.shape[-1] * 2
        group = hidden // scales.shape[-1]
        if hidden % (self.tp.size * group) != 0:
            raise ValueError(f'int4 embedding: hidden={hidden} not divisible by '
                             f'tp*group={self.tp.size * group}')
        self._add_tensor('tok_embeddings', table, split_side=SplitSide.OUTPUT)
        self._add_tensor('tok_embeddings_scale', scales,
                         split_side=SplitSide.OUTPUT)
        self._add_tensor('tok_embeddings_zero', zeros,
                         split_side=SplitSide.OUTPUT)

    def add_lm_head_shared(self):
        """Bind the tied output head to the shared embedding table (TP=1 only).

        Sets ``output_from_tok_embeddings`` on the ModelWeightConfig; no
        LinearWeight is created, so the table exists exactly once.
        """
        if self.tp.size != 1:
            raise RuntimeError(
                'shared embedding head supports tp=1 only; regenerate the '
                'sidecar or run with LMDEPLOY_DISABLE_EMBED_QUANT=1')
        self.config.output_from_tok_embeddings = True

    def add_lm_head(self, linear):
        """Pad lm-head vocab so each TP-local logits row is uint4-aligned."""
        _VECTOR_BYTES = 16
        itemsize = _CPP_TO_TORCH[self.config.data_type].itemsize
        div = (_VECTOR_BYTES // itemsize) * self.tp.size if self.tp.size > 1 else 1
        linear = round_up_output_groups(linear, self._vocab_size, div)
        self._add_linear('output', linear, split_side=SplitSide.OUTPUT)
