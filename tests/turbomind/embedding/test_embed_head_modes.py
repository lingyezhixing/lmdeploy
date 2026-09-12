# tests/turbomind/embedding/test_embed_head_modes.py
import pytest
import torch

from lmdeploy.turbomind.embed_quant import (
    dequant_int4_simple,
    dequant_int8,
    quantize_table,
    resolve_embed_head,
)


def R(**kw):
    base = dict(tie=True, sidecar_format=None, mode='auto', fmt='native',
                env_disabled=False, sm_version=89, tp_size=1,
                engine_dtype='fp16', hidden=2560)
    base.update(kw)
    return resolve_embed_head(**base)


@pytest.mark.parametrize('kwargs,action,fmt,constraint', [
    (dict(tie=False), 'native', None, 'untied'),
    (dict(mode='off'), 'native', None, '--embed-head off'),
    (dict(env_disabled=True), 'native', None, 'LMDEPLOY_DISABLE_EMBED_QUANT'),
    (dict(sidecar_format='int8'), 'sidecar', 'int8', 'offline sidecar'),
    (dict(sidecar_format='int8', fmt='int4'), 'sidecar', 'int8', 'offline sidecar'),  # sidecar wins
    (dict(sidecar_format='native'), 'sidecar', 'native', 'offline sidecar'),
    (dict(fmt='native'), 'shared', 'native', 'shared native'),
    (dict(fmt='int8'), 'shared_quant', 'int8', 'online int8'),
    (dict(fmt='int4'), 'shared_quant', 'int4', 'online int4'),
    (dict(sm_version=70), 'native', None, 'SM70'),                  # auto falls back
    (dict(tp_size=2), 'native', None, 'tp=2'),
    (dict(engine_dtype='other'), 'native', None, 'engine dtype other'),
    (dict(hidden=1000), 'native', None, '% 16 != 0'),
    (dict(fmt='int8', hidden=1040), 'error', None, '% 128 != 0'),
    (dict(mode='on', sm_version=70), 'error', None, 'SM70'),
    (dict(fmt='int8', hidden=1040, mode='on'), 'error', None, '% 128 != 0'),
    (dict(fmt='int8', tp_size=2), 'error', None, 'tp=2'),
    (dict(fmt='int4', sm_version=70), 'error', None, 'SM70'),       # explicit quant, older SM
    (dict(fmt='bf16'), 'error', None, 'invalid'),                   # removed format
    (dict(fmt='fp16'), 'error', None, 'invalid'),                   # removed format
    (dict(fmt='q8'), 'error', None, 'invalid'),                     # invalid format
    (dict(mode='bogus'), 'error', None, 'invalid'),                 # invalid mode
    (dict(mode='off', fmt='int8'), 'native', None, '--embed-head off'),  # off beats explicit quant
])
def test_decision_matrix(kwargs, action, fmt, constraint):
    plan = R(**kwargs)
    assert plan.action == action
    if fmt is not None:
        assert plan.table_format == fmt
    assert constraint in plan.reason
    if action == 'error':
        assert plan.error and 'embed head' in plan.error.lower()
        assert constraint in plan.error
    assert plan.reason  # every plan carries a reason for the log line


@pytest.mark.parametrize('fmt', ['int8', 'int4'])
def test_quantize_table_shapes_and_error(fmt):
    torch.manual_seed(0)
    x = (torch.randn(64, 256, dtype=torch.bfloat16) * 0.05)
    table, scale, zero = quantize_table(x, fmt)
    assert scale.shape == (64, 2)
    if fmt == 'int8':
        assert table.dtype == torch.int8 and table.shape == (64, 256)
        assert zero is None
        rec = dequant_int8(table, scale, 128)
    else:
        assert table.dtype == torch.uint8 and table.shape == (64, 128)
        assert zero is not None and zero.shape == (64, 2)
        rec = dequant_int4_simple(table, scale, zero, 128, 256)
    rel = (rec.float() - x.float()).norm() / x.float().norm()
    assert rel < (0.02 if fmt == 'int8' else 0.15)


def test_quantize_table_rejects_unknown_format():
    with pytest.raises(ValueError, match='unsupported embed head format'):
        quantize_table(torch.zeros(4, 256, dtype=torch.bfloat16), 'fp8')


def test_invalid_format_error_names_valid_values():
    plan = R(fmt='bf16')
    assert plan.action == 'error'
    assert 'native' in plan.error and 'int8' in plan.error and 'int4' in plan.error


def test_quantize_table_rejects_native():
    with pytest.raises(ValueError, match='unsupported embed head format'):
        quantize_table(torch.zeros(4, 256, dtype=torch.bfloat16), 'native')
