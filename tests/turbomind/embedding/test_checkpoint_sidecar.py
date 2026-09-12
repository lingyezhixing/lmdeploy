import json

import pytest
import torch
from safetensors.torch import save_file

from lmdeploy.turbomind.checkpoint import create_checkpoint


def _mini_model(tmp_path):
    (tmp_path / 'config.json').write_text(json.dumps({'tie_word_embeddings': True}))
    save_file({'model.embed_tokens.weight': torch.zeros(4, 8, dtype=torch.bfloat16)},
              str(tmp_path / 'model.safetensors'))


def _write_sidecar(tmp_path, tensors):
    save_file(tensors, str(tmp_path / 'embed_quant.safetensors'))
    (tmp_path / 'embed_quant.json').write_text(
        json.dumps({'version': 2, 'table_format': 'int8', 'head_from_embed': True}))


def test_sidecar_keys_merged(tmp_path):
    _mini_model(tmp_path)
    _write_sidecar(tmp_path, {
        'model.embed_tokens.weight_i8': torch.zeros(4, 8, dtype=torch.int8),
        'model.embed_tokens.weight_i8_scale': torch.ones(4, 1, dtype=torch.bfloat16),
        'lm_head.weight_from_embed': torch.ones(1, dtype=torch.uint8)})
    ckpt = create_checkpoint(str(tmp_path))
    assert ckpt.has('model.embed_tokens.weight_i8')
    assert ckpt.has('lm_head.weight_from_embed')


def test_get_cpu_returns_cpu_tensor(tmp_path):
    _mini_model(tmp_path)
    ckpt = create_checkpoint(str(tmp_path))
    t = ckpt.get_cpu('model.embed_tokens.weight')
    assert t.device.type == 'cpu' and t.dtype == torch.bfloat16
    assert ckpt.get_cpu('model.embed_tokens.weight', index=1).shape == (8,)


def test_sidecar_v1_rejected(tmp_path):
    _mini_model(tmp_path)
    (tmp_path / 'embed_quant.json').write_text('{"version": 1}')
    with pytest.raises(RuntimeError, match='regenerate'):
        create_checkpoint(str(tmp_path))


def test_sidecar_disabled_by_env(tmp_path, monkeypatch):
    _mini_model(tmp_path)
    _write_sidecar(tmp_path,
                   {'model.embed_tokens.weight_i8': torch.zeros(4, 8, dtype=torch.int8)})
    monkeypatch.setenv('LMDEPLOY_DISABLE_EMBED_QUANT', '1')
    ckpt = create_checkpoint(str(tmp_path))
    assert not ckpt.has('model.embed_tokens.weight_i8')


def test_sidecar_keys_pass_through_mappings(tmp_path):
    _mini_model(tmp_path)
    _write_sidecar(tmp_path,
                   {'model.embed_tokens.weight_i8': torch.zeros(4, 8, dtype=torch.int8)})
    ckpt = create_checkpoint(
        str(tmp_path), mappings=[lambda k: k.replace('model.', 'mapped_model.', 1)])
    assert ckpt.has('mapped_model.embed_tokens.weight_i8')
    assert not ckpt.has('model.embed_tokens.weight_i8')


def test_sidecar_skipped_when_embed_head_off(tmp_path):
    _mini_model(tmp_path)
    _write_sidecar(tmp_path, {
        'model.embed_tokens.weight_i8': torch.zeros(4, 8, dtype=torch.int8),
        'lm_head.weight_from_embed': torch.ones(1, dtype=torch.uint8)})
    ckpt = create_checkpoint(str(tmp_path), embed_head='off')
    assert not ckpt.has('model.embed_tokens.weight_i8')
    assert not ckpt.has('lm_head.weight_from_embed')


def test_model_loader_threads_embed_head(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import lmdeploy.turbomind.model_loader as model_loader

    captured = {}

    class FakeCkpt:
        def close(self):
            pass

    def fake_create_checkpoint(model_path, *, mappings=(), embed_head='auto'):
        captured['embed_head'] = embed_head
        return FakeCkpt()

    class FakeSourceModel:
        _loader_mappings = []

        def model(self, pfx):
            pass

    monkeypatch.setattr(model_loader, 'create_checkpoint', fake_create_checkpoint)
    loader = model_loader.ModelLoader.__new__(model_loader.ModelLoader)
    loader.model = FakeSourceModel()
    loader.model_path = str(tmp_path)
    loader.engine_config = SimpleNamespace(embed_head='off')
    loader.export()
    assert captured['embed_head'] == 'off'


def test_model_loader_export_iter_threads_embed_head(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import lmdeploy.turbomind.model_loader as model_loader

    captured = {}

    class FakeCkpt:
        def close(self):
            pass

    def fake_create_checkpoint(model_path, *, mappings=(), embed_head='auto'):
        captured['embed_head'] = embed_head
        return FakeCkpt()

    class FakeSourceModel:
        _loader_mappings = []

        def model(self, pfx):
            pass

    monkeypatch.setattr(model_loader, 'create_checkpoint', fake_create_checkpoint)
    loader = model_loader.ModelLoader.__new__(model_loader.ModelLoader)
    loader.model = FakeSourceModel()
    loader.model_path = str(tmp_path)
    loader.engine_config = SimpleNamespace(embed_head='off')
    gen = loader.export_iter()
    assert next(gen) == -1
    gen.close()
    assert captured['embed_head'] == 'off'
