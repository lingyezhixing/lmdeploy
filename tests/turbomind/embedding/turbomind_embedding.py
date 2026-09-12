import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_LIB_DIR = os.path.join(_REPO_ROOT, 'lmdeploy', 'lib')
if _LIB_DIR not in sys.path:
    sys.path.insert(0, _LIB_DIR)

import _turbomind as _tm  # noqa: E402


def embedding_lookup_int8(out, ids, table, scales, group):
    _tm.embedding_lookup_int8(
        _tm.from_dlpack_with_strides(out),
        _tm.from_dlpack_with_strides(ids),
        _tm.from_dlpack_with_strides(table),
        _tm.from_dlpack_with_strides(scales),
        group,
        stream_ptr=int(torch.cuda.current_stream(out.device).cuda_stream),
    )
    return out


def embedding_lookup_int4(out, ids, table, scales, zeros, group):
    _tm.embedding_lookup_int4(
        _tm.from_dlpack_with_strides(out),
        _tm.from_dlpack_with_strides(ids),
        _tm.from_dlpack_with_strides(table),
        _tm.from_dlpack_with_strides(scales),
        _tm.from_dlpack_with_strides(zeros),
        group,
        stream_ptr=int(torch.cuda.current_stream(out.device).cuda_stream),
    )
    return out


def embedding_lookup(out, ids, table):
    _tm.embedding_lookup(
        _tm.from_dlpack_with_strides(out),
        _tm.from_dlpack_with_strides(ids),
        _tm.from_dlpack_with_strides(table),
        stream_ptr=int(torch.cuda.current_stream(out.device).cuda_stream),
    )
    return out


def logits_from_table(logits, x, table, scale, zero, group=128, impl=None):
    kwargs = {}
    if impl is not None:
        kwargs['impl'] = impl
    _tm.logits_from_table(
        _tm.from_dlpack_with_strides(logits),
        _tm.from_dlpack_with_strides(x),
        _tm.from_dlpack_with_strides(table),
        _tm.from_dlpack_with_strides(scale),
        _tm.from_dlpack_with_strides(zero),
        group,
        stream_ptr=int(torch.cuda.current_stream(x.device).cuda_stream),
        **kwargs,
    )
    return logits
