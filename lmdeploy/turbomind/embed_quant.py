# Copyright (c) OpenMMLab. All rights reserved.
"""Sidecar format + quantization math for the shared embedding head."""
from __future__ import annotations

import json
import os
import os.path as osp
from dataclasses import dataclass
from datetime import datetime, timezone

import torch
from safetensors.torch import load_file, save_file

GROUP_DEFAULT = 128
SIDECAR_NAME = 'embed_quant.safetensors'
SIDECAR_META_NAME = 'embed_quant.json'
SIDECAR_VERSION = 2
DISABLE_ENV = 'LMDEPLOY_DISABLE_EMBED_QUANT'

EMBED_I8_SUFFIX = '_i8'            # <embed_key>_i8
EMBED_I8_SCALE_SUFFIX = '_i8_scale'
EMBED_I4_SUFFIX = '_i4'            # <embed_key>_i4  [V, H/2] uint8, low nibble = even h
EMBED_I4_SCALE_SUFFIX = '_i4_scale'
EMBED_I4_ZERO_SUFFIX = '_i4_zero'
SHARED_HEAD_KEY = 'lm_head.weight_from_embed'   # uint8 [1] sentinel

BITS_TO_FORMAT = {16: 'native', 8: 'int8', 4: 'int4'}
FORMAT_TO_BITS = {v: k for k, v in BITS_TO_FORMAT.items()}
TABLE_FORMATS = ('native', 'int8', 'int4')
LEGACY_FORMAT_ALIASES = {'bf16': 'native', 'fp16': 'native'}

EMBED_KEY_PATTERNS = (
    'model.language_model.embed_tokens.weight',   # Qwen3.5 / InternS2
    'model.embed_tokens.weight',                  # Qwen2/3, Llama, GPT-OSS
    'model.tok_embeddings.weight',                # InternLM2
)


def sidecar_path(model_dir: str) -> str:
    return osp.join(model_dir, SIDECAR_NAME)


def meta_path(model_dir: str) -> str:
    return osp.join(model_dir, SIDECAR_META_NAME)


def read_sidecar_meta(model_dir: str) -> dict:
    if not osp.isfile(meta_path(model_dir)):
        raise RuntimeError(f'embed_quant meta not found: {meta_path(model_dir)}')
    try:
        with open(meta_path(model_dir)) as f:
            meta = json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f'malformed {meta_path(model_dir)}: {e}; regenerate with '
            f'scripts/quantize_embedding.py') from e
    if meta.get('version') != SIDECAR_VERSION:
        raise RuntimeError(
            f'unsupported embed_quant.json version {meta.get("version")} in '
            f'{model_dir}; regenerate with '
            f'scripts/quantize_embedding.py')
    fmt = meta.get('table_format')
    if fmt in LEGACY_FORMAT_ALIASES or (fmt is None and meta.get('bits') == 16):
        fmt = 'native'
        meta['table_format'] = fmt
    if fmt not in TABLE_FORMATS:
        raise RuntimeError(
            f'unsupported embed_quant.json table_format {fmt!r} in {model_dir}; '
            f'regenerate with scripts/quantize_embedding.py')
    return meta


def sidecar_enabled(model_dir: str) -> bool:
    if os.getenv(DISABLE_ENV, '0') == '1':
        return False
    return osp.isfile(meta_path(model_dir))


def find_embed_key(model_dir: str, override: str | None = None) -> str:
    if override:
        return override
    from safetensors import safe_open
    files = [f for f in os.listdir(model_dir) if f.endswith('.safetensors')]
    present = set()
    for name in files:
        if name == SIDECAR_NAME:
            continue
        with safe_open(osp.join(model_dir, name), 'pt') as f:
            present.update(f.keys())
    for key in EMBED_KEY_PATTERNS:
        if key in present:
            return key
    raise RuntimeError(
        f'no known embedding key found in {model_dir}; use --embed-key. '
        f'known: {EMBED_KEY_PATTERNS}')


def find_embed_file(model_dir: str, embed_key: str) -> str:
    """Return the safetensors shard that stores ``embed_key``."""
    from safetensors import safe_open
    for name in sorted(os.listdir(model_dir)):
        if not name.endswith('.safetensors') or name == SIDECAR_NAME:
            continue
        path = osp.join(model_dir, name)
        with safe_open(path, 'pt') as f:
            if embed_key in f.keys():
                return path
    raise RuntimeError(
        f'embedding tensor {embed_key!r} not found in any safetensors '
        f'under {model_dir}')


def quantize_int8_sym(x: torch.Tensor, group: int = GROUP_DEFAULT):
    *lead, hidden = x.shape
    ng = hidden // group
    x = x.reshape(*lead, ng, group)
    scale = (x.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(1e-8)
    q = torch.round(x / scale).clamp_(-128, 127).to(torch.int8)
    return q.reshape(*lead, hidden), scale.squeeze(-1).to(torch.bfloat16)


def dequant_int8(q: torch.Tensor, scale: torch.Tensor, group: int = GROUP_DEFAULT):
    *lead, hidden = q.shape
    ng = hidden // group
    return (q.reshape(*lead, ng, group).float()
            * scale.float().unsqueeze(-1)).reshape(*lead, hidden)


def quantize_int4_simple(x: torch.Tensor, group: int = GROUP_DEFAULT):
    """x [N, K] -> (q uint8 [N, K/2], scale bf16 [N, K/g], zero uint8 [N, K/g]).

    Low nibble of byte k//2 holds element k when k is even (little-endian).
    """
    n, k = x.shape
    ng = k // group
    xg = x.reshape(n, ng, group).float()
    mn = xg.amin(dim=-1)
    mx = xg.amax(dim=-1)
    rng = mx - mn
    scale = (rng / 15.0).clamp_min(1e-8)
    # A constant group would otherwise get a near-zero scale and a saturated
    # zero point, dequantizing to ~0 instead of the constant.
    degenerate = rng < 1e-8
    scale = torch.where(degenerate, (mx.abs() / 15.0).clamp_min(1e-8), scale)
    zp = torch.where(degenerate,
                     torch.where(mx < 0, torch.full_like(mx, 15.0), torch.zeros_like(mx)),
                     torch.round(-mn / scale).clamp_(0, 15))
    q = torch.round(xg / scale.unsqueeze(-1) + zp.unsqueeze(-1)).clamp_(0, 15).float()
    q = q.reshape(n, k).to(torch.uint8)
    packed = q[:, 0::2] | (q[:, 1::2] << 4)
    return packed.contiguous(), scale.to(torch.bfloat16), zp.to(torch.uint8)


def dequant_int4_simple(q: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor,
                        group: int, k: int) -> torch.Tensor:
    even = (q & 0xF).float()
    odd = (q >> 4).float()
    vals = torch.stack([even, odd], dim=-1).reshape(q.shape[0], k)
    s = scale.float().repeat_interleave(group, dim=-1)
    z = zero.float().repeat_interleave(group, dim=-1)
    return (vals - z) * s


def quantize_table(x: torch.Tensor, fmt: str):
    """Group-128 quantization shared by the offline writer and the loader.

    Returns (table, scale, zero); zero is None for int8.
    """
    if fmt == 'int8':
        q, s = quantize_int8_sym(x, GROUP_DEFAULT)
        return q, s, None
    if fmt == 'int4':
        q, s, z = quantize_int4_simple(x, GROUP_DEFAULT)
        return q, s, z
    raise ValueError(f'unsupported embed head format: {fmt!r}')


def write_sidecar(model_dir: str, embed_key: str, table_format: str,
                  tensors: dict, *, head_from_embed: bool, qa: dict | None = None) -> None:
    if table_format not in TABLE_FORMATS:
        raise ValueError(f'unsupported embed quant table format: {table_format!r}')
    tensors = dict(tensors)
    if head_from_embed:
        tensors[SHARED_HEAD_KEY] = torch.ones(1, dtype=torch.uint8)
    meta = {
        'version': SIDECAR_VERSION,
        'embed_key': embed_key,
        'table_format': table_format,
        'bits': FORMAT_TO_BITS[table_format],
        'group': GROUP_DEFAULT,
        'head_from_embed': head_from_embed,
        'created': datetime.now(timezone.utc).isoformat(),
        'qa': qa or {},
    }
    tmp_tensors = sidecar_path(model_dir) + '.tmp'
    tmp_meta = meta_path(model_dir) + '.tmp'
    try:
        save_file({k: v.contiguous() for k, v in tensors.items()}, tmp_tensors)
        os.replace(tmp_tensors, sidecar_path(model_dir))
        with open(tmp_meta, 'w') as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp_meta, meta_path(model_dir))
    except Exception:
        for tmp in (tmp_tensors, tmp_meta):
            if osp.isfile(tmp):
                os.remove(tmp)
        raise


def load_sidecar_tensors(model_dir: str) -> dict:
    if not osp.isfile(sidecar_path(model_dir)):
        return {}
    return load_file(sidecar_path(model_dir))


@dataclass(frozen=True)
class EmbedHeadPlan:
    action: str
    table_format: str
    reason: str
    error: str | None = None


def resolve_embed_head(*, tie: bool, sidecar_format: str | None, mode: str, fmt: str,
                       env_disabled: bool, sm_version: int, tp_size: int,
                       engine_dtype: str, hidden: int) -> EmbedHeadPlan:
    """Choose how the embedding table and the output head are wired.

    Returns a plan whose action is ``native`` (legacy two-tensor path:
    untied, ``--embed-head off``, env disable, or unmet constraints in auto
    mode), ``sidecar`` (offline quantized table + shared head), ``shared``
    (checkpoint table reused as the head), ``shared_quant`` (online group-128
    quantization), or ``error`` (explicit sharing constraints failed).
    """
    if env_disabled:
        return EmbedHeadPlan('native', '', 'env LMDEPLOY_DISABLE_EMBED_QUANT=1')
    if mode not in ('auto', 'on', 'off'):
        return EmbedHeadPlan('error', '', 'invalid mode',
                             error=f'embed head: invalid --embed-head {mode!r}')
    if not tie or mode == 'off':
        return EmbedHeadPlan('native', '', 'untied' if not tie else '--embed-head off')
    if fmt not in TABLE_FORMATS:
        return EmbedHeadPlan('error', '', 'invalid format',
                             error=f'embed head: invalid --embed-head-format {fmt!r}; '
                                   'valid values: native, int8, int4')
    if sidecar_format is not None and sidecar_format not in TABLE_FORMATS:
        return EmbedHeadPlan('error', '', 'invalid sidecar format',
                             error=f'embed head: invalid sidecar table format '
                                   f'{sidecar_format!r}; regenerate the sidecar')
    quantized = (sidecar_format or fmt) in ('int8', 'int4')
    constraints = []
    if sm_version < 80:
        constraints.append(f'SM{sm_version} < SM80 (mma head needs SM80+)')
    if tp_size != 1:
        constraints.append(f'tp={tp_size} != 1')
    if engine_dtype not in ('fp16', 'bf16'):
        constraints.append(f'engine dtype {engine_dtype}')
    if hidden % 16 != 0:
        constraints.append(f'hidden {hidden} % 16 != 0')
    if quantized and hidden % 128 != 0:
        constraints.append(f'hidden {hidden} % 128 != 0 (int8/int4 group 128)')
    if constraints:
        reason = '; '.join(constraints)
        if mode == 'on' or (quantized and sidecar_format is None):
            return EmbedHeadPlan('error', '', reason,
                                 error=f'embed head: cannot share ({reason}); '
                                       'set --embed-head off or regenerate an offline sidecar')
        return EmbedHeadPlan('native', '', reason)
    if sidecar_format is not None:
        return EmbedHeadPlan('sidecar', sidecar_format, 'offline sidecar')
    if quantized:
        return EmbedHeadPlan('shared_quant', fmt, f'online {fmt}')
    return EmbedHeadPlan('shared', 'native', 'shared native')
