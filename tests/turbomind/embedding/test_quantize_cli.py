# tests/turbomind/embedding/test_quantize_cli.py
import json
import os
import sys

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'scripts'))
from quantize_embedding import main  # noqa: E402


def _make_model(tmp_path, vocab=64, hidden=256, tied=True, arch='Qwen3_5ForCausalLM'):
    (tmp_path / 'config.json').write_text(json.dumps(
        {'architectures': [arch], 'tie_word_embeddings': tied}))
    torch.manual_seed(0)
    save_file({'model.language_model.embed_tokens.weight':
               torch.randn(vocab, hidden, dtype=torch.bfloat16).to(torch.bfloat16)},
              str(tmp_path / 'model.safetensors'))


def test_cli_writes_v2_sidecar_tied(tmp_path):
    _make_model(tmp_path)
    rc = main(['--model', str(tmp_path), '--format', 'int8', '--chunk', '16'])
    assert rc == 0
    with safe_open(str(tmp_path / 'embed_quant.safetensors'), 'pt') as f:
        keys = set(f.keys())
        q = f.get_tensor('model.language_model.embed_tokens.weight_i8')
        s = f.get_tensor('model.language_model.embed_tokens.weight_i8_scale')
    assert q.dtype == torch.int8 and q.shape == (64, 256)
    assert s.shape == (64, 2)
    assert 'lm_head.weight_from_embed' in keys
    assert 'lm_head.weight_packed' not in keys
    meta = json.loads((tmp_path / 'embed_quant.json').read_text())
    assert meta['version'] == 2 and meta['table_format'] == 'int8'
    assert meta['qa']['int8_rel_l2'] < 0.02


def test_cli_untied_has_no_sentinel(tmp_path):
    _make_model(tmp_path, tied=False)
    rc = main(['--model', str(tmp_path), '--format', 'int8', '--chunk', '16'])
    assert rc == 0
    with safe_open(str(tmp_path / 'embed_quant.safetensors'), 'pt') as f:
        assert 'lm_head.weight_from_embed' not in set(f.keys())


def test_cli_rejects_non_default_group(tmp_path):
    _make_model(tmp_path)
    with pytest.raises(SystemExit, match='only --group 128'):
        main(['--model', str(tmp_path), '--group', '64'])
    assert not (tmp_path / 'embed_quant.safetensors').exists()
    assert not (tmp_path / 'embed_quant.json').exists()


def test_cli_default_native_sentinel_only(tmp_path):
    _make_model(tmp_path)
    rc = main(['--model', str(tmp_path)])
    assert rc == 0
    with safe_open(str(tmp_path / 'embed_quant.safetensors'), 'pt') as f:
        assert set(f.keys()) == {'lm_head.weight_from_embed'}
    meta = json.loads((tmp_path / 'embed_quant.json').read_text())
    assert meta['table_format'] == 'native' and meta['bits'] == 16


def test_cli_bits_alias_maps_to_native(tmp_path):
    _make_model(tmp_path)
    with pytest.warns(DeprecationWarning):
        rc = main(['--model', str(tmp_path), '--bits', '16'])
    assert rc == 0
    meta = json.loads((tmp_path / 'embed_quant.json').read_text())
    assert meta['table_format'] == 'native' and meta['bits'] == 16


def test_cli_bits_conflicts_with_format(tmp_path):
    _make_model(tmp_path)
    with pytest.raises(SystemExit, match='conflicts'):
        main(['--model', str(tmp_path), '--bits', '8', '--format', 'int4'])


def test_cli_bits4_table_and_sentinel(tmp_path):
    _make_model(tmp_path)
    rc = main(['--model', str(tmp_path), '--bits', '4'])
    assert rc == 0
    with safe_open(str(tmp_path / 'embed_quant.safetensors'), 'pt') as f:
        keys = set(f.keys())
        q = f.get_tensor('model.language_model.embed_tokens.weight_i4')
        s = f.get_tensor('model.language_model.embed_tokens.weight_i4_scale')
        z = f.get_tensor('model.language_model.embed_tokens.weight_i4_zero')
    assert q.dtype == torch.uint8 and q.shape == (64, 128)
    assert s.shape == (64, 2) and z.shape == (64, 2)
    assert 'lm_head.weight_from_embed' in keys
    meta = json.loads((tmp_path / 'embed_quant.json').read_text())
    assert meta['table_format'] == 'int4'
    assert meta['bits'] == 4
    assert 'int4_rel_l2' in meta['qa']


def test_cli_bits16_untied_rejected(tmp_path):
    _make_model(tmp_path, tied=False)
    with pytest.raises(SystemExit, match='requires tied embeddings'):
        main(['--model', str(tmp_path), '--bits', '16'])
