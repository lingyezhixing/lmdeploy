import torch

from lmdeploy.turbomind import _tm
from lmdeploy.turbomind.builders.linear import Linear
from lmdeploy.turbomind.models.utils import reorder_rotary_emb
from lmdeploy.turbomind.weight_format import (
    FP8Format,
    FP8PerTensorFormat,
    TrivialFormat,
    WeightFormatResolver,
)


class _FakePrefix:
    def __init__(self, tensors):
        self.tensors = dict(tensors)

    def has(self, suffix, sep=''):
        return suffix in self.tensors

    def pop(self, suffix, sep='', index=None):
        return self.tensors.pop(suffix)


def _weight():
    return torch.empty(256, 512, dtype=torch.float8_e4m3fn)


def test_per_tensor_fp8_expands_scalar_scale():
    fmt = FP8PerTensorFormat()
    weight = _weight()
    scale = torch.tensor([0.5], dtype=torch.float32)
    assert fmt.accepts({'.weight': weight, '.weight_scale': scale})

    normalized_weight = fmt.normalize(weight, 'weight')
    normalized_scale = fmt.normalize(scale, 'scales')

    assert tuple(normalized_weight.shape) == (512, 256)
    assert tuple(normalized_scale.shape) == (4, 2)
    assert torch.equal(normalized_scale, torch.full((4, 2), 0.5))

    data_format = fmt.make_data_format()
    assert data_format.block_sizes == [128, 128]


def test_per_tensor_fp8_dequant_applies_scale():
    fmt = FP8PerTensorFormat()
    weight = (torch.randn(256, 512) * 0.02).to(torch.float8_e4m3fn)
    scale = torch.tensor([0.5], dtype=torch.float32)
    assert fmt.accepts({'.weight': weight, '.weight_scale': scale})

    tensors = {
        'weight': fmt.normalize(weight, 'weight'),
        'scales': fmt.normalize(scale, 'scales'),
    }
    result = fmt.dequant(tensors, torch.float16)
    expected = (weight.float() * scale).t().half()
    torch.testing.assert_close(result['weight'], expected, rtol=1e-3, atol=1e-3)


def test_reorder_keeps_fp8_block_scales_when_block_spans_heads():
    # head_dim=64, block_out=128: one scale block covers two heads.  RoPE
    # permutes rows inside each head, so the block scale is unchanged.
    weight = torch.zeros(2048, 128, dtype=torch.float8_e4m3fn)
    scales = torch.arange(16 * 1, dtype=torch.float32).reshape(16, 1)
    linear = Linear({'weight': weight, 'scales': scales}, FP8PerTensorFormat())

    reordered = reorder_rotary_emb(linear, 64, 64, dtype=torch.float16)

    assert torch.equal(reordered.tensors['scales'], scales)


def test_per_tensor_fp8_requires_scalar_scale():
    fmt = FP8PerTensorFormat()
    assert not fmt.accepts({'.weight': _weight()})
    assert not fmt.accepts({
        '.weight': _weight(),
        '.weight_scale': torch.ones(4, dtype=torch.float32),
    })


def test_resolver_prefers_per_tensor_fp8_over_trivial():
    weight = _weight()
    scale = torch.tensor([0.25], dtype=torch.float32)
    resolver = WeightFormatResolver(formats=[
        FP8Format(block_out=128),
        FP8PerTensorFormat(),
        TrivialFormat(weight_dtype=_tm.DataType.TYPE_FP16),
    ])
    fmt, tensors = resolver.resolve(_FakePrefix({
        '.weight': weight,
        '.weight_scale': scale,
    }))
    assert isinstance(fmt, FP8PerTensorFormat)
    assert set(tensors) == {'.weight', '.weight_scale'}
