import torch

from lmdeploy.turbomind import _tm


def _from_torch(t):
    """Import a torch tensor into TurboMind.

    Zero-size tensors are replaced with a 1-element dummy: the unified DLPack
    importer rejects non-positive extents, and every kernel path that receives
    an empty scale/zero sidecar ignores it (``data_or(nullptr)``).
    """
    if t.numel() == 0:
        t = torch.zeros(1, dtype=t.dtype, device=t.device)
    return _tm.from_dlpack(t)


def embedding_lookup_int8(out, ids, table, scales, group):
    _tm.embedding_lookup_int8(
        _from_torch(out),
        _from_torch(ids),
        _from_torch(table),
        _from_torch(scales),
        group,
        stream_ptr=int(torch.cuda.current_stream(out.device).cuda_stream),
    )
    return out


def embedding_lookup_int4(out, ids, table, scales, zeros, group):
    _tm.embedding_lookup_int4(
        _from_torch(out),
        _from_torch(ids),
        _from_torch(table),
        _from_torch(scales),
        _from_torch(zeros),
        group,
        stream_ptr=int(torch.cuda.current_stream(out.device).cuda_stream),
    )
    return out


def embedding_lookup(out, ids, table):
    _tm.embedding_lookup(
        _from_torch(out),
        _from_torch(ids),
        _from_torch(table),
        stream_ptr=int(torch.cuda.current_stream(out.device).cuda_stream),
    )
    return out


def logits_from_table(logits, x, table, scale, zero, group=128):
    _tm.logits_from_table(
        _from_torch(logits),
        _from_torch(x),
        _from_torch(table),
        _from_torch(scale),
        _from_torch(zero),
        group,
        stream_ptr=int(torch.cuda.current_stream(x.device).cuda_stream),
    )
    return logits
