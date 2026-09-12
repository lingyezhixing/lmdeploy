import gc
import json
import logging
import weakref
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import lmdeploy.turbomind.models.utils as utils_mod
from lmdeploy.turbomind.checkpoint import Prefix, create_checkpoint
from lmdeploy.turbomind.embed_quant import quantize_table
from lmdeploy.turbomind.models.utils import add_embedding_and_head


@contextmanager
def _capture_lmdeploy_logs():
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    lmdeploy_logger = logging.getLogger('lmdeploy')
    lmdeploy_logger.addHandler(handler)
    try:
        yield records
    finally:
        lmdeploy_logger.removeHandler(handler)


def _resolver(mode='auto', fmt='native'):
    return SimpleNamespace(embed_head=mode, embed_head_format=fmt)


class FakeBuilder:
    def __init__(self, *, tp_size=1):
        self.calls = []
        self.staged_devices = []
        self.tp = SimpleNamespace(size=tp_size)
        import _turbomind as _tm
        self.config = SimpleNamespace(data_type=_tm.DataType.TYPE_BF16)

    def add_token_embeds(self, t):
        self.calls.append(('embed', tuple(t.shape), str(t.dtype)))
        self.staged_devices.append(t.device.type)

    def add_token_embeds_quant(self, t, s):
        self.calls.append(('embed_q', tuple(t.shape), tuple(s.shape)))
        self.staged_devices += [t.device.type, s.device.type]

    def add_token_embeds_quant4(self, t, s, z):
        self.calls.append(('embed_q4', tuple(t.shape), tuple(s.shape), tuple(z.shape)))
        self.staged_devices += [t.device.type, s.device.type, z.device.type]

    def add_lm_head(self, lin):
        self.calls.append(('head',))

    def add_lm_head_shared(self):
        self.calls.append(('head_shared',))


class FakeModel:
    def __init__(self, pfx, resolver=None):
        self._pfx = pfx
        self._resolver = resolver

    def _linear(self, pfx):
        self._pfx = pfx
        return pfx


class FakePrefix:
    """Prefix double that records ``get`` calls and rejects the raw table."""

    def __init__(self, tensors, expensive=()):
        self._tensors = dict(tensors)
        self._expensive = set(expensive)
        self.requested = []
        self.requested_cpu = []

    def has(self, name='', sep='.'):
        return name in self._tensors

    def get(self, name='', sep='.'):
        self.requested.append(name)
        if name in self._expensive:
            raise AssertionError(f'raw tensor {name!r} was materialized')
        return self._tensors[name]

    def get_cpu(self, name='', sep='.'):
        self.requested_cpu.append(name)
        if name in self._expensive:
            raise AssertionError(f'raw tensor {name!r} was materialized')
        return self._tensors[name]

    def __add__(self, key):
        return key


def _model_dir(tmp_path, with_sidecar):
    (tmp_path / 'config.json').write_text(json.dumps({'tie_word_embeddings': True}))
    save_file({'model.embed_tokens.weight': torch.zeros(4, 8, dtype=torch.bfloat16)},
              str(tmp_path / 'model.safetensors'))
    if with_sidecar:
        save_file({'model.embed_tokens.weight_i8': torch.zeros(4, 8, dtype=torch.int8),
                   'model.embed_tokens.weight_i8_scale': torch.ones(4, 1, dtype=torch.bfloat16)},
                  str(tmp_path / 'embed_quant.safetensors'))
        (tmp_path / 'embed_quant.json').write_text(json.dumps(
            {'version': 2, 'table_format': 'int8', 'head_from_embed': True}))


def _model_dir_shared(tmp_path):
    (tmp_path / 'config.json').write_text(json.dumps({'tie_word_embeddings': True}))
    save_file({'model.embed_tokens.weight': torch.zeros(4, 8, dtype=torch.bfloat16)},
              str(tmp_path / 'model.safetensors'))
    save_file({'model.embed_tokens.weight_i8': torch.zeros(4, 8, dtype=torch.int8),
               'model.embed_tokens.weight_i8_scale': torch.ones(4, 1, dtype=torch.bfloat16),
               'lm_head.weight_from_embed': torch.ones(1, dtype=torch.uint8)},
              str(tmp_path / 'embed_quant.safetensors'))
    (tmp_path / 'embed_quant.json').write_text(json.dumps(
        {'version': 2, 'table_format': 'int8', 'head_from_embed': True}))


def _capable_model(tmp_path, hidden, *, tie=True):
    (tmp_path / 'config.json').write_text(json.dumps({'tie_word_embeddings': tie}))
    save_file({'model.embed_tokens.weight': torch.zeros(16, hidden, dtype=torch.bfloat16)},
              str(tmp_path / 'model.safetensors'))


def test_quantized_branch(tmp_path):
    _model_dir(tmp_path, with_sidecar=True)
    pfx = Prefix(create_checkpoint(str(tmp_path)))
    b, m = FakeBuilder(), FakeModel(pfx)
    add_embedding_and_head(m, b, pfx, 'model.embed_tokens.weight', tie=True)
    assert b.calls[0] == ('embed_q', (4, 8), (4, 1))
    assert ('head_shared',) in b.calls
    assert ('head',) not in b.calls


def test_plain_branch(tmp_path):
    _model_dir(tmp_path, with_sidecar=False)
    pfx = Prefix(create_checkpoint(str(tmp_path)))
    b, m = FakeBuilder(), FakeModel(pfx, _resolver(mode='off'))
    add_embedding_and_head(m, b, pfx, 'model.embed_tokens.weight', tie=True)
    assert b.calls == [('embed', (4, 8), 'torch.bfloat16'), ('head',)]
    assert m._pfx.prefix == 'model.embed_tokens'


def test_shared_head_branch(tmp_path):
    _model_dir_shared(tmp_path)
    pfx = Prefix(create_checkpoint(str(tmp_path)))
    b, m = FakeBuilder(), FakeModel(pfx)
    add_embedding_and_head(m, b, pfx, 'model.embed_tokens.weight', tie=True)
    assert ('embed_q', (4, 8), (4, 1)) in b.calls
    assert ('head_shared',) in b.calls
    assert ('head',) not in b.calls


@pytest.mark.parametrize('fmt', ['int8', 'int4'])
def test_sidecar_load_never_reads_raw_table(fmt):
    raw_key = 'model.embed_tokens.weight'
    if fmt == 'int8':
        tensors = {raw_key + '_i8': torch.zeros(4, 128, dtype=torch.int8),
                   raw_key + '_i8_scale': torch.ones(4, 1, dtype=torch.bfloat16)}
        expected = ('embed_q', (4, 128), (4, 1))
    else:
        tensors = {raw_key + '_i4': torch.zeros(4, 64, dtype=torch.uint8),
                   raw_key + '_i4_scale': torch.ones(4, 1, dtype=torch.bfloat16),
                   raw_key + '_i4_zero': torch.zeros(4, 1, dtype=torch.uint8)}
        expected = ('embed_q4', (4, 64), (4, 1), (4, 1))
    pfx = FakePrefix(tensors, expensive=[raw_key])
    b, m = FakeBuilder(), FakeModel(pfx)
    add_embedding_and_head(m, b, pfx, raw_key, tie=True)
    assert b.calls == [expected, ('head_shared',)]
    assert raw_key not in pfx.requested


@pytest.mark.parametrize('fmt', ['int8', 'int4'])
def test_untied_sidecar_quantizes_lookup_only(fmt):
    raw_key = 'model.embed_tokens.weight'
    if fmt == 'int8':
        tensors = {raw_key + '_i8': torch.zeros(4, 128, dtype=torch.int8),
                   raw_key + '_i8_scale': torch.ones(4, 1, dtype=torch.bfloat16)}
        expected = ('embed_q', (4, 128), (4, 1))
    else:
        tensors = {raw_key + '_i4': torch.zeros(4, 64, dtype=torch.uint8),
                   raw_key + '_i4_scale': torch.ones(4, 1, dtype=torch.bfloat16),
                   raw_key + '_i4_zero': torch.zeros(4, 1, dtype=torch.uint8)}
        expected = ('embed_q4', (4, 64), (4, 1), (4, 1))
    pfx = FakePrefix(tensors, expensive=[raw_key])
    b, m = FakeBuilder(), FakeModel(pfx)
    add_embedding_and_head(m, b, pfx, raw_key, tie=False)
    assert b.calls == [expected, ('head',)]
    assert raw_key not in pfx.requested
    assert m._pfx == 'lm_head'


def test_shared_head_success_sets_flag():
    from types import SimpleNamespace

    from lmdeploy.turbomind.builders.text_model import TextModelBuilder
    builder = TextModelBuilder.__new__(TextModelBuilder)
    builder.tp = SimpleNamespace(size=1)
    builder.config = SimpleNamespace(output_from_tok_embeddings=False)
    builder.add_lm_head_shared()
    assert builder.config.output_from_tok_embeddings is True


def test_tp_guard():
    from types import SimpleNamespace

    from lmdeploy.turbomind.builders.text_model import TextModelBuilder
    builder = TextModelBuilder.__new__(TextModelBuilder)
    builder.tp = SimpleNamespace(size=2)
    builder.config = SimpleNamespace(output_from_tok_embeddings=False)
    with pytest.raises(RuntimeError, match='tp=1'):
        builder.add_lm_head_shared()


def test_tied_head_ignores_legacy_packed_without_sidecar(tmp_path):
    (tmp_path / 'config.json').write_text(json.dumps({'tie_word_embeddings': True}))
    save_file({'model.embed_tokens.weight': torch.zeros(4, 8, dtype=torch.bfloat16),
               'lm_head.weight_packed': torch.zeros(4, 1, dtype=torch.int32)},
              str(tmp_path / 'model.safetensors'))
    pfx = Prefix(create_checkpoint(str(tmp_path)))
    b, m = FakeBuilder(), FakeModel(pfx, _resolver(mode='off'))
    add_embedding_and_head(m, b, pfx, 'model.embed_tokens.weight', tie=True)
    assert b.calls[0] == ('embed', (4, 8), 'torch.bfloat16')
    assert any(c[0] == 'head' for c in b.calls)
    assert m._pfx.prefix == 'model.embed_tokens'


def test_untied_head_uses_head_key(tmp_path):
    (tmp_path / 'config.json').write_text(json.dumps({'tie_word_embeddings': False}))
    save_file({'model.tok_embeddings.weight': torch.zeros(4, 8, dtype=torch.bfloat16)},
              str(tmp_path / 'model.safetensors'))
    pfx = Prefix(create_checkpoint(str(tmp_path)))
    b, m = FakeBuilder(), FakeModel(pfx)
    add_embedding_and_head(m, b, pfx, 'model.tok_embeddings.weight',
                           tie=False, head_key='output')
    assert b.calls[0] == ('embed', (4, 8), 'torch.bfloat16')
    assert any(c[0] == 'head' for c in b.calls)
    assert m._pfx.prefix == 'output'


def test_int4_shared_branch(tmp_path):
    (tmp_path / 'config.json').write_text(json.dumps({'tie_word_embeddings': True}))
    save_file({'model.embed_tokens.weight': torch.zeros(4, 8, dtype=torch.bfloat16)},
              str(tmp_path / 'model.safetensors'))
    save_file({'model.embed_tokens.weight_i4': torch.zeros(4, 4, dtype=torch.uint8),
               'model.embed_tokens.weight_i4_scale': torch.ones(4, 1, dtype=torch.bfloat16),
               'model.embed_tokens.weight_i4_zero': torch.zeros(4, 1, dtype=torch.uint8),
               'lm_head.weight_from_embed': torch.ones(1, dtype=torch.uint8)},
              str(tmp_path / 'embed_quant.safetensors'))
    (tmp_path / 'embed_quant.json').write_text(json.dumps(
        {'version': 2, 'table_format': 'int4', 'head_from_embed': True}))
    pfx = Prefix(create_checkpoint(str(tmp_path)))
    b, m = FakeBuilder(), FakeModel(pfx)
    add_embedding_and_head(m, b, pfx, 'model.embed_tokens.weight', tie=True)
    assert ('embed_q4', (4, 4), (4, 1), (4, 1)) in b.calls
    assert ('head_shared',) in b.calls


def test_shared_plan_binds_table_once(tmp_path, monkeypatch):
    _capable_model(tmp_path, hidden=16)
    monkeypatch.setattr(utils_mod, '_sm_version', lambda: 89)
    pfx = Prefix(create_checkpoint(str(tmp_path)))
    b, m = FakeBuilder(), FakeModel(pfx, _resolver(mode='auto'))
    add_embedding_and_head(m, b, pfx, 'model.embed_tokens.weight', tie=True)
    assert b.calls == [('embed', (16, 16), 'torch.bfloat16'), ('head_shared',)]
    assert ('head',) not in b.calls


@pytest.mark.parametrize('fmt', ['int8', 'int4'])
def test_shared_quant_plan_quantizes_from_cpu(tmp_path, monkeypatch, fmt):
    _capable_model(tmp_path, hidden=256)
    monkeypatch.setattr(utils_mod, '_sm_version', lambda: 89)
    if fmt == 'int8':
        sentinel = (torch.zeros(16, 256, dtype=torch.int8),
                    torch.ones(16, 2, dtype=torch.bfloat16),
                    None)
    else:
        sentinel = (torch.zeros(16, 128, dtype=torch.uint8),
                    torch.ones(16, 2, dtype=torch.bfloat16),
                    torch.zeros(16, 2, dtype=torch.uint8))
    seen = []

    def fake_quantize_embed_table(table, requested):
        seen.append((tuple(table.shape), requested, table.device.type))
        return sentinel

    monkeypatch.setattr(utils_mod, '_quantize_embed_table', fake_quantize_embed_table)
    pfx = Prefix(create_checkpoint(str(tmp_path)))
    b, m = FakeBuilder(), FakeModel(pfx, _resolver(fmt=fmt))
    add_embedding_and_head(m, b, pfx, 'model.embed_tokens.weight', tie=True)
    assert seen == [((16, 256), fmt, 'cpu')]
    if fmt == 'int8':
        assert ('embed_q', (16, 256), (16, 2)) in b.calls
        assert b.staged_devices == ['cpu', 'cpu']
    else:
        assert ('embed_q4', (16, 128), (16, 2), (16, 2)) in b.calls
        assert b.staged_devices == ['cpu', 'cpu', 'cpu']
    assert ('head_shared',) in b.calls
    assert ('head',) not in b.calls


@pytest.mark.parametrize('fmt', ['int8', 'int4'])
def test_quantize_embed_table_matches_gpu_reference(fmt):
    if not torch.cuda.is_available():
        pytest.skip('cuda')
    torch.manual_seed(0)
    table = torch.randn(64, 256, dtype=torch.bfloat16) * 0.05
    q, s, z = utils_mod._quantize_embed_table(table, fmt)
    ref_q, ref_s, ref_z = quantize_table(table.cuda(), fmt)
    assert q.device.type == 'cpu' and s.device.type == 'cpu'
    assert torch.equal(q, ref_q.cpu()) and torch.equal(s, ref_s.cpu())
    if z is None:
        assert ref_z is None
    else:
        assert z.device.type == 'cpu' and torch.equal(z, ref_z.cpu())


@pytest.mark.parametrize('fmt', ['int8', 'int4'])
def test_sidecar_stages_cpu_tensors(tmp_path, fmt):
    raw_key = 'model.embed_tokens.weight'
    if fmt == 'int8':
        tensors = {raw_key + '_i8': torch.zeros(4, 128, dtype=torch.int8),
                   raw_key + '_i8_scale': torch.ones(4, 1, dtype=torch.bfloat16)}
        expected_devices = ['cpu', 'cpu']
    else:
        tensors = {raw_key + '_i4': torch.zeros(4, 64, dtype=torch.uint8),
                   raw_key + '_i4_scale': torch.ones(4, 1, dtype=torch.bfloat16),
                   raw_key + '_i4_zero': torch.zeros(4, 1, dtype=torch.uint8)}
        expected_devices = ['cpu', 'cpu', 'cpu']
    pfx = FakePrefix(tensors, expensive=[raw_key])
    b, m = FakeBuilder(), FakeModel(pfx)
    add_embedding_and_head(m, b, pfx, raw_key, tie=True)
    assert b.staged_devices == expected_devices
    assert pfx.requested == []
    assert raw_key not in pfx.requested_cpu


def test_shared_stages_cpu_table(tmp_path, monkeypatch):
    _capable_model(tmp_path, hidden=16)
    monkeypatch.setattr(utils_mod, '_sm_version', lambda: 89)
    pfx = Prefix(create_checkpoint(str(tmp_path)))
    b, m = FakeBuilder(), FakeModel(pfx)
    add_embedding_and_head(m, b, pfx, 'model.embed_tokens.weight', tie=True)
    assert b.staged_devices == ['cpu']
    assert ('head_shared',) in b.calls


def test_native_legacy_stages_cpu_table(tmp_path):
    _capable_model(tmp_path, hidden=16)
    pfx = Prefix(create_checkpoint(str(tmp_path)))
    b, m = FakeBuilder(), FakeModel(pfx, _resolver(mode='off'))
    add_embedding_and_head(m, b, pfx, 'model.embed_tokens.weight', tie=True)
    assert b.staged_devices == ['cpu']
    assert ('head',) in b.calls


def test_native_plan_legacy_tie_head(tmp_path):
    _capable_model(tmp_path, hidden=16)
    pfx = Prefix(create_checkpoint(str(tmp_path)))
    b, m = FakeBuilder(), FakeModel(pfx, _resolver(mode='off'))
    add_embedding_and_head(m, b, pfx, 'model.embed_tokens.weight', tie=True)
    assert b.calls == [('embed', (16, 16), 'torch.bfloat16'), ('head',)]
    assert m._pfx.prefix == 'model.embed_tokens'


def test_env_disable_keeps_legacy_path(tmp_path, monkeypatch):
    _model_dir(tmp_path, with_sidecar=True)
    monkeypatch.setenv('LMDEPLOY_DISABLE_EMBED_QUANT', '1')
    pfx = Prefix(create_checkpoint(str(tmp_path)))
    b, m = FakeBuilder(), FakeModel(pfx)
    add_embedding_and_head(m, b, pfx, 'model.embed_tokens.weight', tie=True)
    assert b.calls == [('embed', (4, 8), 'torch.bfloat16'), ('head',)]
    assert m._pfx.prefix == 'model.embed_tokens'


def test_error_plan_raises_with_constraint(tmp_path, monkeypatch):
    _capable_model(tmp_path, hidden=16)
    monkeypatch.setattr(utils_mod, '_sm_version', lambda: 70)
    pfx = Prefix(create_checkpoint(str(tmp_path)))
    b, m = FakeBuilder(), FakeModel(pfx, _resolver(fmt='int8'))
    with pytest.raises(RuntimeError, match='SM70'):
        add_embedding_and_head(m, b, pfx, 'model.embed_tokens.weight', tie=True)
    assert b.calls == []


def test_sidecar_format_conflict_logs_one_warning(tmp_path):
    _model_dir_shared(tmp_path)
    pfx = Prefix(create_checkpoint(str(tmp_path)))
    b, m = FakeBuilder(), FakeModel(pfx, _resolver(fmt='int4'))
    with _capture_lmdeploy_logs() as records:
        add_embedding_and_head(m, b, pfx, 'model.embed_tokens.weight', tie=True)
    warnings = [r for r in records if r.levelno == logging.WARNING]
    infos = [r for r in records
             if r.levelno == logging.INFO and r.getMessage().startswith('embed head:')]
    assert len(warnings) == 1
    assert 'int4' in warnings[0].getMessage() and 'int8' in warnings[0].getMessage()
    assert len(infos) == 1
    assert ('head_shared',) in b.calls


def test_sidecar_native_format_no_warning(tmp_path):
    _model_dir_shared(tmp_path)
    pfx = Prefix(create_checkpoint(str(tmp_path)))
    b, m = FakeBuilder(), FakeModel(pfx, _resolver(fmt='native'))
    with _capture_lmdeploy_logs() as records:
        add_embedding_and_head(m, b, pfx, 'model.embed_tokens.weight', tie=True)
    assert not [r for r in records if r.levelno >= logging.WARNING]
    assert ('head_shared',) in b.calls


def test_sidecar_matching_format_no_warning(tmp_path):
    _model_dir_shared(tmp_path)
    pfx = Prefix(create_checkpoint(str(tmp_path)))
    b, m = FakeBuilder(), FakeModel(pfx, _resolver(fmt='int8'))
    with _capture_lmdeploy_logs() as records:
        add_embedding_and_head(m, b, pfx, 'model.embed_tokens.weight', tie=True)
    assert not [r for r in records if r.levelno >= logging.WARNING]
    infos = [r for r in records
             if r.levelno == logging.INFO and r.getMessage().startswith('embed head:')]
    assert len(infos) == 1
    assert ('head_shared',) in b.calls


def test_resolver_carries_embed_head_options():
    import _turbomind as _tm

    from lmdeploy.turbomind.weight_format import TrivialFormat, WeightFormatResolver
    r = WeightFormatResolver(data_type=_tm.DataType.TYPE_FP16, formats=[TrivialFormat()],
                             embed_head='on', embed_head_format='int4')
    assert r.embed_head == 'on' and r.embed_head_format == 'int4'
    d = WeightFormatResolver(data_type=_tm.DataType.TYPE_FP16, formats=[TrivialFormat()])
    assert d.embed_head == 'auto' and d.embed_head_format == 'native'


def _make_probe_builder():
    from lmdeploy.turbomind.builders._base import Builder

    class ProbeBuilder(Builder):
        def _create_handles(self):
            self._handles = []

        def _commit_child(self, name, handles):
            pass

        def _commit_tensor(self, name, tensor, split_side):
            pass

    config = SimpleNamespace(data_type=torch.float16)
    ctx = SimpleNamespace(devices=[], data_type=torch.float16, active_mask=(True,))
    return ProbeBuilder(config, ctx)


def test_build_clears_pending_staging():
    builder = _make_probe_builder()
    builder._add_child('child', [None])
    builder._add_tensor('weight', torch.zeros(4, 8), None)

    built = builder.build()

    assert builder._pending_tensors == {}
    assert builder._pending_children == {}
    assert list(built) == []
    # build() stays idempotent after the drain.
    assert list(builder.build()) == []


def test_build_releases_staged_tensor():
    builder = _make_probe_builder()
    tensor = torch.zeros(4, 8)
    ref = weakref.ref(tensor)
    builder._add_tensor('weight', tensor, None)
    del tensor

    builder.build()
    gc.collect()

    assert ref() is None, 'staged tensor kept alive after build()'


def test_native_sidecar_shared_branch(tmp_path):
    (tmp_path / 'config.json').write_text(json.dumps({'tie_word_embeddings': True}))
    save_file({'model.embed_tokens.weight': torch.zeros(4, 8, dtype=torch.bfloat16)},
              str(tmp_path / 'model.safetensors'))
    save_file({'lm_head.weight_from_embed': torch.ones(1, dtype=torch.uint8)},
              str(tmp_path / 'embed_quant.safetensors'))
    (tmp_path / 'embed_quant.json').write_text(json.dumps(
        {'version': 2, 'table_format': 'native', 'bits': 16, 'head_from_embed': True}))
    pfx = Prefix(create_checkpoint(str(tmp_path)))
    b, m = FakeBuilder(), FakeModel(pfx)
    add_embedding_and_head(m, b, pfx, 'model.embed_tokens.weight', tie=True)
    assert b.calls == [('embed', (4, 8), 'torch.bfloat16'), ('head_shared',)]


def test_native_shared_fp32_table_logs_engine_dtype(tmp_path, monkeypatch):
    (tmp_path / 'config.json').write_text(json.dumps({'tie_word_embeddings': True}))
    save_file({'model.embed_tokens.weight': torch.zeros(16, 16, dtype=torch.float32)},
              str(tmp_path / 'model.safetensors'))
    monkeypatch.setattr(utils_mod, '_sm_version', lambda: 89)
    pfx = Prefix(create_checkpoint(str(tmp_path)))
    b, m = FakeBuilder(), FakeModel(pfx)
    with _capture_lmdeploy_logs() as records:
        add_embedding_and_head(m, b, pfx, 'model.embed_tokens.weight', tie=True)
    assert b.calls == [('embed', (16, 16), 'torch.float32'), ('head_shared',)]
    infos = [r.getMessage() for r in records if r.levelno == logging.INFO]
    assert any('native table dtype torch.float32' in m for m in infos)
