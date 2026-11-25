import pytest
import torch

from lmdeploy.turbomind import _tm
from lmdeploy.turbomind.builders.linear import Linear
from lmdeploy.turbomind.linear import get_linear
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


def test_per_tensor_fp8_uses_global_scale():
    fmt = FP8PerTensorFormat()
    weight = _weight()
    scale = torch.tensor([0.5], dtype=torch.float32)
    assert fmt.accepts({'.weight': weight, '.weight_scale': scale})

    normalized_weight = fmt.normalize(weight, 'weight')
    normalized_scale = fmt.normalize(scale, 'global_scale')

    assert tuple(normalized_weight.shape) == (512, 256)
    assert tuple(normalized_scale.shape) == (1,)
    assert torch.equal(normalized_scale, scale)

    data_format = fmt.make_data_format()
    assert data_format.block_sizes == []


def test_per_tensor_fp8_dequant_applies_scale():
    fmt = FP8PerTensorFormat()
    weight = (torch.randn(256, 512) * 0.02).to(torch.float8_e4m3fn)
    scale = torch.tensor([0.5], dtype=torch.float32)
    assert fmt.accepts({'.weight': weight, '.weight_scale': scale})

    tensors = {
        'weight': fmt.normalize(weight, 'weight'),
        'global_scale': fmt.normalize(scale, 'global_scale'),
    }
    result = fmt.dequant(tensors, torch.float16)
    expected = (weight.float() * scale).t().half()
    torch.testing.assert_close(result['weight'], expected, rtol=1e-3, atol=1e-3)


def test_reorder_keeps_fp8_global_scale():
    weight = torch.zeros(2048, 128, dtype=torch.float8_e4m3fn)
    linear = Linear({'weight': weight, 'global_scale': torch.tensor([1.0])}, FP8PerTensorFormat())

    reordered = reorder_rotary_emb(linear, 64, 64, dtype=torch.float16)

    assert torch.equal(reordered.tensors['global_scale'], torch.tensor([1.0]))


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_global_scale_linear_matches_reference():
    torch.manual_seed(0)
    weight = (torch.randn(512, 256) * 0.02).to(torch.float8_e4m3fn)
    scale = torch.tensor([0.37], dtype=torch.float32)
    x = torch.randn(2, 256, dtype=torch.float16, device='cuda')

    fmt = FP8PerTensorFormat()
    assert fmt.accepts({'.weight': weight, '.weight_scale': scale})
    linear = get_linear()
    plan = linear.get_weight_plan(
        weight_format=fmt, dtype=torch.float16, input_dtype=torch.float16, output_dtype=torch.float16)
    prepared = linear.prepare_weight(weight.to('cuda'), plan=plan, global_scale=scale.to('cuda'))
    exec_plan = linear.get_exec_plan(x, prepared)
    actual, _ = linear(x, prepared, exec_plan=exec_plan)

    expected = (x.float() @ (weight.float().cuda() * scale.cuda()).t()).half()
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
