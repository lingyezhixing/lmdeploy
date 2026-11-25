# Copyright (c) OpenMMLab. All rights reserved.
"""HYV3 (Hy-MT3 / Hy3) TextModel for the new TurboMind pipeline.

HYV3 combines a Qwen3-style GQA attention block (QK RMSNorm + RoPE) with a
DeepSeek/GLM-style noaux_tc MoE block:

- sigmoid router scores with a float32 expert-selection bias
- top-k routed experts, optionally renormalized and scaled
- one shared expert per MoE layer, added outside the routed experts
- layer 0 is dense; ``first_k_dense_replace`` / ``mlp_layer_types`` select it

The checkpoint stores per-expert ``gate_proj`` / ``up_proj`` / ``down_proj``
tensors and names the router ``mlp.router.gate`` and the shared expert
``mlp.shared_mlp``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import HYV3Config

from ..builders import (
    AttentionBuilder,
    DecoderLayerBuilder,
    DecoderLayerConfig,
    FfnBuilder,
    ModuleListBuilder,
    ModuleListConfig,
    MoeBuilder,
    TextModelBuilder,
    _act_type_id,
)
from ..builders.linear import Linear
from ..text_model import TextModel
from ..weight_format import FP8Format, FP8PerTensorFormat
from .base import INPUT_MODELS
from .utils import (
    add_embedding_and_head,
    make_attention_config,
    make_ffn_config,
    make_model_weight_config,
    make_moe_config,
    reorder_rotary_emb,
)


@INPUT_MODELS.register_module(name='hy_v3')
class HyV3TextModel(TextModel):
    """Weight model for HYV3 (Hy-MT3 / Hy3)."""

    cfg: HYV3Config

    def __init__(self, cfg: HYV3Config, *, resolver):
        super().__init__(cfg, resolver=resolver)

        self._attn_cfg = make_attention_config(cfg)
        self._ffn_cfg = make_ffn_config(cfg, act_type=_act_type_id('silu'))

        self._n_experts = int(getattr(cfg, 'num_experts', 0) or 0)
        if self._n_experts > 0:
            self._moe_cfg = make_moe_config(
                cfg,
                experts_per_token=cfg.num_experts_per_tok,
                norm_topk_prob=bool(getattr(cfg, 'route_norm', True)),
                topk_method='noaux_tc',
                scoring_func=('sigmoid' if getattr(cfg, 'moe_router_use_sigmoid', True) else 'softmax'),
                routed_scale=float(getattr(cfg, 'router_scaling_factor', 1.0)),
                topk_group=1,
                n_group=1,
            )
            self._moe_cfg.expert_num = self._n_experts

    # ------------------------------------------------------------------
    # model() - walks the full hierarchy
    # ------------------------------------------------------------------

    def model(self, pfx):
        root_cfg = make_model_weight_config(self.cfg)
        builder = TextModelBuilder(
            root_cfg, self._ctx,
            root_handles=self._root_handles,
            tp=self._model_tp,
            vocab_size=self.cfg.vocab_size)
        add_embedding_and_head(self, builder, pfx, 'model.embed_tokens.weight',
                               tie=self.cfg.tie_word_embeddings, head_from_cpu=True)
        builder.norm = self.norm(pfx + 'model.norm')
        builder.layers = self.layers(pfx + 'model.layers')
        builder.build()

    # ------------------------------------------------------------------
    # Attention / FFN / MoE factories
    # ------------------------------------------------------------------

    @staticmethod
    def _row_scale(linear: Linear) -> Linear:
        if not isinstance(linear.weight_format, FP8PerTensorFormat):
            return linear
        weight = linear.tensors['weight']
        scale = linear.tensors['global_scale'].reshape(1, 1)
        scales = scale.expand(weight.shape[0] // 128, weight.shape[1]).contiguous()
        return Linear({'weight': weight, 'scales': scales}, FP8Format(block_out=1))

    def attn(self, pfx):
        q, k, v, o = [self._linear(pfx + f'{x}_proj') for x in 'qkvo']

        cfg = self._attn_cfg.clone()

        def reorder(x):
            return reorder_rotary_emb(x, cfg.head_dim, cfg.rope.dim, dtype=self._ctx.dtype)

        q, k = [reorder(x) for x in (q, k)]

        # Fused QKV needs one scale per output row; the three source tensors
        # have independent per-tensor scales.  Expanding them here keeps the
        # fused weights exact while using only ~200 KiB of scales per layer.
        q, k, v = [self._row_scale(x) for x in (q, k, v)]

        m = AttentionBuilder(cfg, self._ctx, tp=self._attn_tp)
        m.add_qkv_proj(q, k, v)
        m.add_o_proj(o)

        m.q_norm = self.norm(pfx + 'q_norm', reorder)
        m.k_norm = self.norm(pfx + 'k_norm', reorder)

        return m.build()

    def ffn(self, pfx, inter_size=None, is_expert=False, *, tp):
        w1, w3, w2 = [self._linear(pfx + f'{x}_proj') for x in ('gate', 'up', 'down')]

        cfg = self._ffn_cfg.clone()
        cfg.inter_size = (self.cfg.intermediate_size if inter_size is None else inter_size)
        cfg.is_expert = is_expert

        m = FfnBuilder(cfg, self._ctx, tp=tp)
        m.add_ffn(w1, w2, w3, fuse_gate_up=False)
        return m.build()

    def moe(self, pfx):
        cfg = self._moe_cfg.clone()

        m = MoeBuilder(cfg, self._ctx, ep=self._ep)
        m.add_gate('gate', self._linear(pfx + 'router.gate'))

        if getattr(self.cfg, 'moe_router_enable_expert_bias', False):
            correction = pfx.pop('expert_bias')
            m.add_param('score_correction_bias', correction)

        experts = ModuleListBuilder(ModuleListConfig(), self._ctx)
        for e in m.range(cfg.expert_num):
            experts[e] = self.ffn(
                pfx + 'experts' + e,
                inter_size=self.cfg.moe_intermediate_size,
                is_expert=True,
                tp=self._mlp_tp)
        m.experts = experts.build()

        num_shared = int(getattr(self.cfg, 'num_shared_experts', 0) or 0)
        if num_shared > 0:
            shared_inter = self.cfg.moe_intermediate_size * num_shared
            m.shared = self.ffn(pfx + 'shared_mlp', inter_size=shared_inter, tp=self._mlp_tp)

        return m.build()

    def layers(self, pfx):
        mlp_layer_types = getattr(self.cfg, 'mlp_layer_types', None)
        first_k_dense = int(getattr(self.cfg, 'first_k_dense_replace', 0) or 0)
        any_moe = self._n_experts > 0 and (
            mlp_layer_types is None or any(t == 'sparse' for t in mlp_layer_types))
        dense_tp = self._dense_tp if any_moe else self._mlp_tp

        layers = ModuleListBuilder(ModuleListConfig(), self._ctx)
        for i, p in pfx.slices(0, self.cfg.num_hidden_layers):
            d = DecoderLayerBuilder(DecoderLayerConfig(), self._ctx)
            d.attention_norm = self.norm(p + 'input_layernorm')
            d.attention = self.attn(p + 'self_attn')
            d.ffn_norm = self.norm(p + 'post_attention_layernorm')

            if mlp_layer_types is not None:
                is_moe = mlp_layer_types[i] == 'sparse'
            else:
                is_moe = self._n_experts > 0 and i >= first_k_dense

            if is_moe:
                d.moe_ffn = self.moe(p + 'mlp')
            else:
                d.feed_forward = self.ffn(p + 'mlp', tp=dense_tp)
            layers[i] = d.build()
        return layers.build()
