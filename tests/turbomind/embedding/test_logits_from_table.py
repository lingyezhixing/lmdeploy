import pytest
import torch

from .reference import logits_from_table_bf16_ref, logits_from_table_int4_mma_ref, logits_from_table_int8_mma_ref
from .turbomind_embedding import logits_from_table

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda'),
    pytest.mark.skipif(torch.cuda.get_device_capability() < (8, 0), reason='sm80+'),
]


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('tokens,vocab', [(1, 1000), (2, 256), (7, 512), (2, 999)])
def test_int8_matches_reference(dtype, tokens, vocab):
    dim, group = 2560, 128
    torch.manual_seed(0)
    x = (torch.randn(tokens, dim, device='cuda') * 0.1).to(dtype)
    table = torch.randint(-127, 128, (vocab, dim), dtype=torch.int8, device='cuda')
    scale = (torch.rand(vocab, dim // group, device='cuda') * 0.1).to(dtype)
    logits = torch.empty(tokens, vocab, dtype=dtype, device='cuda')
    zero = torch.empty(0, dtype=torch.uint8, device='cuda')
    logits_from_table(logits, x, table, scale, zero, group)
    ref = logits_from_table_int8_mma_ref(x, table, scale, group)
    torch.testing.assert_close(logits, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('tokens,vocab', [(1, 1000), (2, 256), (7, 512), (2, 999)])
def test_bf16_matches_reference(dtype, tokens, vocab):
    dim = 2560
    torch.manual_seed(0)
    x = (torch.randn(tokens, dim, device='cuda') * 0.1).to(dtype)
    table = ((torch.randn(vocab, dim, device='cuda') * 0.05).to(dtype))
    logits = torch.empty(tokens, vocab, dtype=dtype, device='cuda')
    scale = torch.empty(0, dtype=dtype, device='cuda')
    zero = torch.empty(0, dtype=torch.uint8, device='cuda')
    logits_from_table(logits, x, table, scale, zero, 128)
    ref = logits_from_table_bf16_ref(x, table)
    torch.testing.assert_close(logits, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('tokens,vocab', [(1, 1000), (2, 256), (7, 512), (2, 999)])
def test_int4_matches_reference(dtype, tokens, vocab):
    dim, group = 2560, 128
    torch.manual_seed(0)
    x = (torch.randn(tokens, dim, device='cuda') * 0.1).to(dtype)
    table = torch.randint(0, 256, (vocab, dim // 2), dtype=torch.uint8, device='cuda')
    scale = (torch.rand(vocab, dim // group, device='cuda') * 0.1).to(dtype)
    zero = torch.randint(0, 16, (vocab, dim // group), dtype=torch.uint8, device='cuda')
    logits = torch.empty(tokens, vocab, dtype=dtype, device='cuda')
    logits_from_table(logits, x, table, scale, zero, group)
    ref = logits_from_table_int4_mma_ref(x, table, scale, zero, group)
    torch.testing.assert_close(logits, ref, atol=2e-2, rtol=2e-2)
