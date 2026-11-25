from transformers import HYV3Config

from lmdeploy.turbomind.models.base import INPUT_MODELS
from lmdeploy.turbomind.supported_models import SUPPORTED_ARCHS


def _tiny_config():
    return HYV3Config(
        num_hidden_layers=3,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=128,
        num_experts=128,
        num_experts_per_tok=8,
        num_shared_experts=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=1024,
        first_k_dense_replace=1,
        route_norm=True,
        router_scaling_factor=2.826,
        moe_router_use_sigmoid=True,
        moe_router_enable_expert_bias=True,
        qk_norm=True,
    )


def test_hy_v3_is_registered():
    assert SUPPORTED_ARCHS['HYV3ForCausalLM'] == 'hy_v3'
    model_cls = INPUT_MODELS.get('hy_v3')
    assert model_cls.__name__ == 'HyV3TextModel'


def test_hy_v3_moe_config_matches_checkpoint_semantics():
    cfg = _tiny_config()
    model = INPUT_MODELS.get('hy_v3')(cfg, resolver=None)
    assert model._moe_cfg.topk_method == 'noaux_tc'
    assert model._moe_cfg.scoring_func == 'sigmoid'
    assert model._moe_cfg.norm_topk_prob is True
    assert model._moe_cfg.routed_scale == 2.826
    assert model._moe_cfg.experts_per_token == 8
    assert model._moe_cfg.expert_num == 128
    assert model._moe_cfg.n_group == 1
    assert model._moe_cfg.topk_group == 1
