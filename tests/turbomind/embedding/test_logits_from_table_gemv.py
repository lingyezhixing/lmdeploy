"""Tests for the GEMV decode path (tokens <= 16) of the shared-table head.

The mma path is covered by test_logits_from_table_mma.py; here we force the
GEMV implementation with ``impl=1`` and compare against the same torch
references.
"""
import pytest
import torch

from .reference import (
    logits_from_table_bf16_ref,
    logits_from_table_int4_mma_ref,
    logits_from_table_int8_mma_ref,
)
from .turbomind_embedding import logits_from_table

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda'),
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0), reason='sm80+'),
]

DTYPES = [pytest.param(torch.bfloat16, id='bf16'), pytest.param(torch.float16, id='fp16')]
TOKENS = [1, 2, 7, 8, 16]


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('tokens', TOKENS)
@pytest.mark.parametrize('dim', [2560, 384])
def test_int8_gemv_matches_reference(dtype, tokens, dim):
    group, vocab = 128, 999
    torch.manual_seed(0)
    x = (torch.randn(tokens, dim, device='cuda') * 0.1).to(dtype)
    table = torch.randint(-127, 128, (vocab, dim), dtype=torch.int8, device='cuda')
    scale = (torch.rand(vocab, dim // group, device='cuda') * 0.1).to(dtype)
    zero = torch.empty(0, dtype=torch.uint8, device='cuda')
    logits = torch.empty(tokens, vocab, dtype=dtype, device='cuda')
    logits_from_table(logits, x, table, scale, zero, group, impl=1)
    ref = logits_from_table_int8_mma_ref(x, table, scale, group)
    torch.testing.assert_close(logits, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('tokens', TOKENS)
@pytest.mark.parametrize('dim', [2560, 384])
def test_int4_gemv_matches_reference(dtype, tokens, dim):
    group, vocab = 128, 999
    torch.manual_seed(0)
    x = (torch.randn(tokens, dim, device='cuda') * 0.1).to(dtype)
    table = torch.randint(0, 256, (vocab, dim // 2), dtype=torch.uint8, device='cuda')
    scale = (torch.rand(vocab, dim // group, device='cuda') * 0.1).to(dtype)
    zero = torch.randint(0, 16, (vocab, dim // group), dtype=torch.uint8, device='cuda')
    logits = torch.empty(tokens, vocab, dtype=dtype, device='cuda')
    logits_from_table(logits, x, table, scale, zero, group, impl=1)
    ref = logits_from_table_int4_mma_ref(x, table, scale, zero, group)
    torch.testing.assert_close(logits, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('tokens', TOKENS)
@pytest.mark.parametrize('dim', [2560, 1040])
def test_float_gemv_matches_reference(dtype, tokens, dim):
    vocab = 999
    torch.manual_seed(0)
    x = (torch.randn(tokens, dim, device='cuda') * 0.1).to(dtype)
    table = (torch.randn(vocab, dim, device='cuda') * 0.05).to(dtype)
    scale = torch.empty(0, dtype=dtype, device='cuda')
    zero = torch.empty(0, dtype=torch.uint8, device='cuda')
    logits = torch.empty(tokens, vocab, dtype=dtype, device='cuda')
    logits_from_table(logits, x, table, scale, zero, 128, impl=1)
    ref = logits_from_table_bf16_ref(x, table)
    torch.testing.assert_close(logits, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('x_dtype,table_dtype',
                         [(torch.float16, torch.bfloat16), (torch.bfloat16, torch.float16)])
def test_float_gemv_mixed_table_dtype(x_dtype, table_dtype):
    from .reference import logits_from_table_cast_ref
    dim, vocab, tokens = 2560, 999, 7
    torch.manual_seed(0)
    x = (torch.randn(tokens, dim, device='cuda') * 0.1).to(x_dtype)
    table = (torch.randn(vocab, dim, device='cuda') * 0.05).to(table_dtype)
    scale = torch.empty(0, dtype=x_dtype, device='cuda')
    zero = torch.empty(0, dtype=torch.uint8, device='cuda')
    logits = torch.empty(tokens, vocab, dtype=x_dtype, device='cuda')
    logits_from_table(logits, x, table, scale, zero, 128, impl=1)
    ref = logits_from_table_cast_ref(x, table)
    torch.testing.assert_close(logits, ref, atol=2e-2, rtol=2e-2)


def test_gemv_rejects_prefill():
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    code = (
        "import sys; sys.path.insert(0, r'" + str(repo / 'tests') + "');"
        "import torch;"
        "from turbomind.embedding.turbomind_embedding import logits_from_table;"
        "dim, group, vocab, tokens = 2560, 128, 1000, 17;"
        "x=torch.zeros(tokens,dim,device='cuda',dtype=torch.float16);"
        "t=torch.zeros(vocab,dim,device='cuda',dtype=torch.int8);"
        "s=torch.zeros(vocab,dim//group,device='cuda',dtype=torch.float16);"
        "z=torch.empty(0,device='cuda',dtype=torch.uint8);"
        "o=torch.empty(tokens,vocab,device='cuda',dtype=torch.float16);"
        "logits_from_table(o,x,t,s,z,group,impl=1)"
    )
    p = subprocess.run([sys.executable, '-c', code], cwd=str(repo), capture_output=True, text=True, timeout=120)
    assert p.returncode != 0
    assert 'tokens <= 16' in (p.stderr + p.stdout)
