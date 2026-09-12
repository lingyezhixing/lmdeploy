import pytest
import torch

from .reference import (
    logits_from_table_bf16_ref,
    logits_from_table_cast_ref,
    logits_from_table_int4_mma_ref,
    logits_from_table_int8_mma_ref,
)
from .turbomind_embedding import logits_from_table

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda'),
    pytest.mark.skipif(torch.cuda.get_device_capability() < (8, 0), reason='sm80+'),
]

DTYPES = [pytest.param(torch.bfloat16, id='bf16'), pytest.param(torch.float16, id='fp16')]


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('tokens,vocab', [(1, 1000), (2, 256), (7, 512), (2, 999),
                                          (8, 32768), (17, 1000), (33, 512)])
def test_float_mma_matches_reference(dtype, tokens, vocab):
    dim = 2560
    torch.manual_seed(0)
    x = (torch.randn(tokens, dim, device='cuda') * 0.1).to(dtype)
    table = (torch.randn(vocab, dim, device='cuda') * 0.05).to(dtype)
    scale = torch.empty(0, dtype=dtype, device='cuda')
    zero = torch.empty(0, dtype=torch.uint8, device='cuda')
    logits = torch.empty(tokens, vocab, dtype=dtype, device='cuda')
    logits_from_table(logits, x, table, scale, zero, 128)
    ref = logits_from_table_bf16_ref(x, table)
    torch.testing.assert_close(logits, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('dtype', DTYPES)
def test_float_mma_partial_k_stage(dtype):  # dim % 16 == 0 but dim % 32 != 0
    dim, tokens, vocab = 1040, 3, 999
    torch.manual_seed(0)
    x = (torch.randn(tokens, dim, device='cuda') * 0.1).to(dtype)
    table = (torch.randn(vocab, dim, device='cuda') * 0.05).to(dtype)
    scale = torch.empty(0, dtype=dtype, device='cuda')
    zero = torch.empty(0, dtype=torch.uint8, device='cuda')
    logits = torch.empty(tokens, vocab, dtype=dtype, device='cuda')
    logits_from_table(logits, x, table, scale, zero, 128)
    ref = logits_from_table_bf16_ref(x, table)
    torch.testing.assert_close(logits, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('tokens,vocab', [(1, 1000), (2, 256), (7, 512), (2, 999), (8, 32768)])
def test_int8_mma_matches_reference(dtype, tokens, vocab):
    dim, group = 2560, 128
    torch.manual_seed(0)
    x = (torch.randn(tokens, dim, device='cuda') * 0.1).to(dtype)
    table = torch.randint(-127, 128, (vocab, dim), dtype=torch.int8, device='cuda')
    scale = (torch.rand(vocab, dim // group, device='cuda') * 0.1).to(dtype)
    zero = torch.empty(0, dtype=torch.uint8, device='cuda')
    logits = torch.empty(tokens, vocab, dtype=dtype, device='cuda')
    logits_from_table(logits, x, table, scale, zero, group)
    ref = logits_from_table_int8_mma_ref(x, table, scale, group)
    torch.testing.assert_close(logits, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('tokens', [64, 65, 512])
def test_int8_mma_prefill_tiles(dtype, tokens):
    dim, group, vocab = 2560, 128, 1000
    torch.manual_seed(0)
    x = (torch.randn(tokens, dim, device='cuda') * 0.1).to(dtype)
    table = torch.randint(-127, 128, (vocab, dim), dtype=torch.int8, device='cuda')
    scale = (torch.rand(vocab, dim // group, device='cuda') * 0.1).to(dtype)
    zero = torch.empty(0, dtype=torch.uint8, device='cuda')
    logits = torch.empty(tokens, vocab, dtype=dtype, device='cuda')
    logits_from_table(logits, x, table, scale, zero, group)
    ref = logits_from_table_int8_mma_ref(x, table, scale, group)
    torch.testing.assert_close(logits, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('dtype', DTYPES)
def test_int8_mma_crosses_prefill_boundary(dtype):
    dim, group, vocab = 2560, 128, 1000
    tokens = 17
    torch.manual_seed(0)
    x = (torch.randn(tokens, dim, device='cuda') * 0.1).to(dtype)
    table = torch.randint(-127, 128, (vocab, dim), dtype=torch.int8, device='cuda')
    scale = (torch.rand(vocab, dim // group, device='cuda') * 0.1).to(dtype)
    zero = torch.empty(0, dtype=torch.uint8, device='cuda')
    logits = torch.empty(tokens, vocab, dtype=dtype, device='cuda')
    logits_from_table(logits, x, table, scale, zero, group)
    ref = logits_from_table_int8_mma_ref(x, table, scale, group)
    torch.testing.assert_close(logits, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('tokens,vocab', [(1, 1000), (2, 256), (7, 512), (2, 999), (8, 32768)])
def test_int4_mma_matches_reference(dtype, tokens, vocab):
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


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('tokens', [64, 65])
def test_int4_mma_prefill_tiles(dtype, tokens):
    dim, group, vocab = 2560, 128, 1000
    torch.manual_seed(0)
    x = (torch.randn(tokens, dim, device='cuda') * 0.1).to(dtype)
    table = torch.randint(0, 256, (vocab, dim // 2), dtype=torch.uint8, device='cuda')
    scale = (torch.rand(vocab, dim // group, device='cuda') * 0.1).to(dtype)
    zero = torch.randint(0, 16, (vocab, dim // group), dtype=torch.uint8, device='cuda')
    logits = torch.empty(tokens, vocab, dtype=dtype, device='cuda')
    logits_from_table(logits, x, table, scale, zero, group)
    ref = logits_from_table_int4_mma_ref(x, table, scale, zero, group)
    torch.testing.assert_close(logits, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('dtype', DTYPES)
def test_int4_mma_crosses_prefill_boundary(dtype):
    dim, group, vocab = 2560, 128, 1000
    tokens = 17
    torch.manual_seed(0)
    x = (torch.randn(tokens, dim, device='cuda') * 0.1).to(dtype)
    table = torch.randint(0, 256, (vocab, dim // 2), dtype=torch.uint8, device='cuda')
    scale = (torch.rand(vocab, dim // group, device='cuda') * 0.1).to(dtype)
    zero = torch.randint(0, 16, (vocab, dim // group), dtype=torch.uint8, device='cuda')
    logits = torch.empty(tokens, vocab, dtype=dtype, device='cuda')
    logits_from_table(logits, x, table, scale, zero, group)
    ref = logits_from_table_int4_mma_ref(x, table, scale, zero, group)
    torch.testing.assert_close(logits, ref, atol=2e-2, rtol=2e-2)


def test_unsupported_shape_fails_loudly():
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    code = (
        "import sys; sys.path.insert(0, r'" + str(repo / 'tests') + "');"
        "import torch;"
        "from turbomind.embedding.turbomind_embedding import logits_from_table;"
        "x=torch.zeros(2,1000,device='cuda',dtype=torch.float16);"
        "t=torch.zeros(16,1000,device='cuda',dtype=torch.float16);"
        "s=torch.empty(0,device='cuda',dtype=torch.float16);"
        "z=torch.empty(0,device='cuda',dtype=torch.uint8);"
        "o=torch.empty(2,16,device='cuda',dtype=torch.float16);"
        "logits_from_table(o,x,t,s,z,128)"
    )
    p = subprocess.run([sys.executable, '-c', code], cwd=str(repo), capture_output=True, text=True, timeout=120)
    assert p.returncode != 0
    assert 'dim % 16' in (p.stderr + p.stdout)


PAIRS = [(torch.float16, torch.float16), (torch.float16, torch.bfloat16),
         (torch.bfloat16, torch.float16), (torch.bfloat16, torch.bfloat16)]


@pytest.mark.parametrize('x_dtype,table_dtype', PAIRS)
@pytest.mark.parametrize('tokens,vocab', [(1, 1000), (7, 512), (33, 512)])
def test_float_mma_mixed_table_dtype(x_dtype, table_dtype, tokens, vocab):
    dim = 2560
    torch.manual_seed(0)
    x = (torch.randn(tokens, dim, device='cuda') * 0.1).to(x_dtype)
    table = (torch.randn(vocab, dim, device='cuda') * 0.05).to(table_dtype)
    scale = torch.empty(0, dtype=x_dtype, device='cuda')
    zero = torch.empty(0, dtype=torch.uint8, device='cuda')
    logits = torch.empty(tokens, vocab, dtype=x_dtype, device='cuda')
    logits_from_table(logits, x, table, scale, zero, 128)
    ref = logits_from_table_cast_ref(x, table)
    torch.testing.assert_close(logits, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('x_dtype,table_dtype', PAIRS)
def test_float_mma_mixed_table_dtype_one_hot_exact(x_dtype, table_dtype):
    dim, vocab = 2560, 1000
    torch.manual_seed(0)
    table = (torch.randn(vocab, dim, device='cuda') * 0.05).to(table_dtype)
    x = torch.zeros(1, dim, device='cuda', dtype=x_dtype)
    x[0, 7] = 1.0
    scale = torch.empty(0, dtype=x_dtype, device='cuda')
    zero = torch.empty(0, dtype=torch.uint8, device='cuda')
    logits = torch.empty(1, vocab, dtype=x_dtype, device='cuda')
    logits_from_table(logits, x, table, scale, zero, 128)
    expected = table.to(x_dtype)[:, 7].reshape(1, -1)
    assert torch.equal(logits, expected)
