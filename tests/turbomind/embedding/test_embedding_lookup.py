import pytest
import torch

from .reference import embedding_lookup_int4_ref, embedding_lookup_int8_ref, embedding_lookup_ref
from .turbomind_embedding import embedding_lookup, embedding_lookup_int4, embedding_lookup_int8

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda')


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('num,dim,group', [(1, 256, 128), (7, 512, 128), (33, 2560, 128)])
def test_lookup_matches_reference(dtype, num, dim, group):
    torch.manual_seed(0)
    vocab = 1000
    table = torch.randint(-127, 128, (vocab, dim), dtype=torch.int8, device='cuda')
    scales = torch.rand(vocab, dim // group, dtype=torch.float32, device='cuda') * 0.1
    scales = scales.to(dtype)
    ids = torch.randint(0, vocab, (num,), dtype=torch.int32, device='cuda')
    out = torch.empty(num, dim, dtype=dtype, device='cuda')
    embedding_lookup_int8(out, ids, table, scales, group)
    ref = embedding_lookup_int8_ref(ids, table, scales, group, dtype)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('num,dim,group', [(1, 256, 128), (7, 512, 128), (33, 2560, 128)])
def test_lookup_int4_matches_reference(dtype, num, dim, group):
    torch.manual_seed(0)
    vocab = 1000
    q = torch.randint(0, 256, (vocab, dim // 2), dtype=torch.uint8, device='cuda')
    scales = (torch.rand(vocab, dim // group, device='cuda') * 0.1).to(dtype)
    zeros = torch.randint(0, 16, (vocab, dim // group), dtype=torch.uint8, device='cuda')
    ids = torch.randint(0, vocab, (num,), dtype=torch.int32, device='cuda')
    out = torch.empty(num, dim, dtype=dtype, device='cuda')
    embedding_lookup_int4(out, ids, q, scales, zeros, group)
    ref = embedding_lookup_int4_ref(ids, q, scales, zeros, group, dtype)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('out_dtype,table_dtype', [
    (torch.float16, torch.float16), (torch.float16, torch.bfloat16),
    (torch.bfloat16, torch.float16), (torch.bfloat16, torch.bfloat16)])
def test_lookup_mixed_dtype_matches_reference(out_dtype, table_dtype):
    torch.manual_seed(0)
    vocab, dim, num = 1000, 2560, 33
    table = (torch.randn(vocab, dim, device='cuda') * 0.05).to(table_dtype)
    ids = torch.randint(0, vocab, (num,), dtype=torch.int32, device='cuda')
    out = torch.empty(num, dim, dtype=out_dtype, device='cuda')
    embedding_lookup(out, ids, table)
    ref = embedding_lookup_ref(ids, table, out_dtype)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


def test_lookup_bf16_table_fp16_out_is_exact():
    torch.manual_seed(0)
    vocab, dim, num = 512, 2560, 17
    table = (torch.randn(vocab, dim, device='cuda') * 0.05).to(torch.bfloat16)
    ids = torch.randint(0, vocab, (num,), dtype=torch.int32, device='cuda')
    out = torch.empty(num, dim, dtype=torch.float16, device='cuda')
    embedding_lookup(out, ids, table)
    assert torch.equal(out, table[ids].to(torch.float16))
