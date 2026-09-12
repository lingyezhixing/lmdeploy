import torch


def embedding_lookup_int8_ref(ids, table, scales, group, dtype):
    rows = table[ids].float()
    s = scales[ids].float()
    s = s.repeat_interleave(group, dim=-1)[:, :rows.shape[1]]
    return (rows * s).to(dtype)


def logits_from_table_int8_mma_ref(x, table, scale, group=128):
    w = (table.float() * scale.float().repeat_interleave(group, dim=-1)).to(x.dtype)
    return (x.float() @ w.float().t()).to(x.dtype)


def logits_from_table_bf16_ref(x, table):
    return (x.float() @ table.float().t()).to(x.dtype)


def embedding_lookup_int4_ref(ids, table, scales, zeros, group, dtype):
    rows = table[ids]
    even = (rows & 0xF).float()
    odd = (rows >> 4).float()
    q = torch.stack([even, odd], dim=-1).reshape(rows.shape[0], -1)
    s = scales[ids].float().repeat_interleave(group, dim=-1)
    z = zeros[ids].float().repeat_interleave(group, dim=-1)
    return ((q - z) * s).to(dtype)


def logits_from_table_int4_mma_ref(x, table, scale, zero, group=128):
    even = (table & 0xF).float()
    odd = (table >> 4).float()
    q = torch.stack([even, odd], dim=-1).reshape(table.shape[0], -1)
    w = ((q - zero.float().repeat_interleave(group, dim=-1))
         * scale.float().repeat_interleave(group, dim=-1)).to(x.dtype)
    return (x.float() @ w.float().t()).to(x.dtype)


def embedding_lookup_ref(ids, table, dtype):
    return table[ids].to(dtype)


def logits_from_table_cast_ref(x, table):
    return (x.float() @ table.to(x.dtype).float().t()).to(x.dtype)
