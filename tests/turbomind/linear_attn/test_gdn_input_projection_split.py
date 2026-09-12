"""DeltaNet input-projection split: [Q|K|V|Z] and [B|A] must be committed
as two separate linears so a high-precision gate pair cannot downgrade the
big projection to trivial (fp16) during mixed-format fusion."""
from types import SimpleNamespace

import torch

from lmdeploy.turbomind.builders.deltanet import DeltaNetBuilder
from lmdeploy.turbomind.linear import Linear
from lmdeploy.turbomind.weight_format import TrivialFormat, WeightFormat


class FakeQuantFormat(WeightFormat):
    name = 'fake-quant'
    suffix_map = {}
    weight_dtype = None
    has_zero_point = False

    def accepts(self, available):
        return True

    def normalize(self, x, kind):
        return x

    def dequant(self, tensors, data_type):
        return {'weight': tensors['weight']}


class ProbeDeltaNetBuilder(DeltaNetBuilder):

    def __init__(self, *, num_k_heads=2, num_v_heads=2):
        self.config = SimpleNamespace(
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            data_type=torch.float16,
        )
        self.tp = SimpleNamespace(size=1)
        self.committed = []

    def _add_linear(self, name, linear, split_side=None):
        self.committed.append((name, linear, split_side))


def _linear(weight_format, in_dim, out_dim, quant, fill):
    tensors = {'weight': torch.full((in_dim, out_dim), fill)}
    if quant:
        tensors['scales'] = torch.full((in_dim, out_dim), fill)
        tensors['zeros'] = torch.full((in_dim, out_dim), fill)
    return Linear(tensors=tensors, weight_format=weight_format)


def _projections(*, quant_gates):
    quant = FakeQuantFormat()
    trivial = TrivialFormat()
    qkv = _linear(quant, 3, 12, True, 1.0)
    z = _linear(quant, 3, 4, True, 2.0)
    b = _linear(quant if quant_gates else trivial, 3, 2,
                quant_gates, 3.0)
    a = _linear(quant if quant_gates else trivial, 3, 2,
                quant_gates, 4.0)
    return qkv, z, b, a


def test_split_keeps_big_projection_quantized_when_gates_are_trivial():
    builder = ProbeDeltaNetBuilder()
    qkv, z, b, a = _projections(quant_gates=False)

    builder.add_input_projections(in_proj_qkv=qkv, in_proj_z=z,
                                  in_proj_b=b, in_proj_a=a)

    names = [c[0] for c in builder.committed]
    assert names == ['in_proj_all', 'in_proj_ba']
    all_lin = builder.committed[0][1]
    ba_lin = builder.committed[1][1]
    assert all_lin.weight_format.name == 'fake-quant'
    assert tuple(all_lin.tensors['weight'].shape) == (3, 16)
    assert ba_lin.weight_format.name == 'trivial'
    assert tuple(ba_lin.tensors['weight'].shape) == (3, 4)


def test_split_keeps_gate_pair_quantized_when_checkpoint_quantized_them():
    builder = ProbeDeltaNetBuilder()
    qkv, z, b, a = _projections(quant_gates=True)

    builder.add_input_projections(in_proj_qkv=qkv, in_proj_z=z,
                                  in_proj_b=b, in_proj_a=a)

    assert builder.committed[0][1].weight_format.name == 'fake-quant'
    assert builder.committed[1][1].weight_format.name == 'fake-quant'


def test_gate_pair_layout_is_b_then_a():
    builder = ProbeDeltaNetBuilder()
    qkv, z, b, a = _projections(quant_gates=False)

    builder.add_input_projections(in_proj_qkv=qkv, in_proj_z=z,
                                  in_proj_b=b, in_proj_a=a)

    ba = builder.committed[1][1].tensors['weight']
    assert torch.equal(ba[:, :2], b.tensors['weight'])
    assert torch.equal(ba[:, 2:], a.tensors['weight'])


def test_missing_gates_skip_gate_commit():
    builder = ProbeDeltaNetBuilder()
    qkv, z, _, _ = _projections(quant_gates=False)

    builder.add_input_projections(in_proj_qkv=qkv, in_proj_z=z)

    assert [c[0] for c in builder.committed] == ['in_proj_all']


def test_out_proj_committed_after_gate_pair():
    builder = ProbeDeltaNetBuilder()
    qkv, z, b, a = _projections(quant_gates=False)
    out_proj = _linear(TrivialFormat(), 4, 3, False, 5.0)

    builder.add_input_projections(in_proj_qkv=qkv, in_proj_z=z,
                                  in_proj_b=b, in_proj_a=a,
                                  out_proj=out_proj)

    assert [c[0] for c in builder.committed] == [
        'in_proj_all', 'in_proj_ba', 'out_proj']
