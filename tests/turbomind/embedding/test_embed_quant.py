import json

import pytest
import torch

from lmdeploy.turbomind.embed_quant import (
    SIDECAR_META_NAME,
    SIDECAR_NAME,
    dequant_int4_simple,
    dequant_int8,
    find_embed_key,
    load_sidecar_tensors,
    quantize_int4_simple,
    quantize_int8_sym,
    read_sidecar_meta,
    sidecar_enabled,
    write_sidecar,
)


def test_int8_roundtrip_quality():
    torch.manual_seed(0)
    x = torch.randn(256, 512, dtype=torch.bfloat16)
    q, s = quantize_int8_sym(x, 128)
    assert q.dtype == torch.int8 and q.shape == x.shape
    assert s.shape == (256, 4) and s.dtype == torch.bfloat16
    dq = dequant_int8(q, s, 128)
    rel = ((dq.float() - x.float()).norm() / x.float().norm()).item()
    assert rel < 0.02


def test_find_embed_key(tmp_path):
    from safetensors.torch import save_file
    save_file({'model.language_model.embed_tokens.weight': torch.zeros(4, 8)},
              str(tmp_path / 'model.safetensors'))
    assert find_embed_key(str(tmp_path)) == 'model.language_model.embed_tokens.weight'


def test_find_embed_key_missing_raises(tmp_path):
    from safetensors.torch import save_file
    save_file({'model.other.weight': torch.zeros(4, 8)},
              str(tmp_path / 'model.safetensors'))
    with pytest.raises(RuntimeError, match='no known embedding key'):
        find_embed_key(str(tmp_path))


def test_sidecar_v2_write_and_env(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    save_file({'model.embed_tokens.weight': torch.zeros(4, 8)},
              str(tmp_path / 'model.safetensors'))
    q8 = torch.zeros(4, 8, dtype=torch.int8)
    s8 = torch.ones(4, 1, dtype=torch.bfloat16)
    write_sidecar(str(tmp_path), 'model.embed_tokens.weight', 'int8',
                  {'model.embed_tokens.weight_i8': q8,
                   'model.embed_tokens.weight_i8_scale': s8},
                  head_from_embed=True, qa={'int8_rel_l2': 0.001})
    assert (tmp_path / SIDECAR_NAME).exists()
    assert 'lm_head.weight_from_embed' in load_sidecar_tensors(str(tmp_path))
    meta = json.loads((tmp_path / SIDECAR_META_NAME).read_text())
    assert meta['version'] == 2
    assert meta['table_format'] == 'int8'
    assert meta['head_from_embed'] is True
    assert sidecar_enabled(str(tmp_path))
    monkeypatch.setenv('LMDEPLOY_DISABLE_EMBED_QUANT', '1')
    assert not sidecar_enabled(str(tmp_path))


def test_sidecar_v1_rejected(tmp_path):
    (tmp_path / SIDECAR_META_NAME).write_text(json.dumps({'version': 1}))
    with pytest.raises(RuntimeError, match='regenerate'):
        read_sidecar_meta(str(tmp_path))


def test_int4_simple_roundtrip():
    torch.manual_seed(0)
    x = torch.randn(96, 256, dtype=torch.bfloat16)
    q, s, z = quantize_int4_simple(x, 128)
    assert q.dtype == torch.uint8 and q.shape == (96, 128)
    assert s.shape == (96, 2) and z.shape == (96, 2) and z.dtype == torch.uint8
    dq = dequant_int4_simple(q, s, z, 128, 256)
    rel = ((dq.float() - x.float()).norm() / x.float().norm()).item()
    assert rel < 0.12


@pytest.mark.parametrize('value', [3.0, -2.5, 0.0])
def test_int4_simple_constant_group_roundtrip(value):
    x = torch.full((2, 256), value, dtype=torch.bfloat16)
    q, s, z = quantize_int4_simple(x, 128)
    dq = dequant_int4_simple(q, s, z, 128, 256)
    torch.testing.assert_close(dq, x.float(), atol=1e-2, rtol=0)


def test_load_sidecar_tensors_absent(tmp_path):
    assert load_sidecar_tensors(str(tmp_path)) == {}


@pytest.mark.parametrize('meta_fmt,bits', [('bf16', 16), ('fp16', 16), (None, 16)])
def test_sidecar_meta_legacy_native_normalized(tmp_path, meta_fmt, bits):
    meta = {'version': 2, 'bits': bits}
    if meta_fmt is not None:
        meta['table_format'] = meta_fmt
    (tmp_path / SIDECAR_META_NAME).write_text(json.dumps(meta))
    assert read_sidecar_meta(str(tmp_path))['table_format'] == 'native'


def test_sidecar_meta_unknown_format_rejected(tmp_path):
    (tmp_path / SIDECAR_META_NAME).write_text(json.dumps(
        {'version': 2, 'table_format': 'q8', 'bits': 8}))
    with pytest.raises(RuntimeError, match='regenerate'):
        read_sidecar_meta(str(tmp_path))


def test_sidecar_write_native_sentinel_only(tmp_path):
    from safetensors.torch import save_file
    save_file({'model.embed_tokens.weight': torch.zeros(4, 8)},
              str(tmp_path / 'model.safetensors'))
    write_sidecar(str(tmp_path), 'model.embed_tokens.weight', 'native', {},
                  head_from_embed=True)
    meta = json.loads((tmp_path / SIDECAR_META_NAME).read_text())
    assert meta['table_format'] == 'native' and meta['bits'] == 16
    assert load_sidecar_tensors(str(tmp_path)).keys() == {'lm_head.weight_from_embed'}
