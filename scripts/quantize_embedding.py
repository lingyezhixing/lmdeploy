#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""Offline single-table conversion for TurboMind (sidecar v2).

Usage:
  python scripts/quantize_embedding.py --model E:\\models\\LLM\\Qwen3.5-4B-AWQ --format int8
"""
import argparse
import json
import os.path as osp
import sys
import warnings

import torch
from safetensors import safe_open

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from lmdeploy.turbomind.embed_quant import (  # noqa: E402
    BITS_TO_FORMAT,
    EMBED_I4_SCALE_SUFFIX,
    EMBED_I4_SUFFIX,
    EMBED_I4_ZERO_SUFFIX,
    EMBED_I8_SCALE_SUFFIX,
    EMBED_I8_SUFFIX,
    SIDECAR_META_NAME,
    SIDECAR_NAME,
    dequant_int4_simple,
    dequant_int8,
    find_embed_file,
    find_embed_key,
    quantize_table,
    write_sidecar,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--format', dest='table_format',
                    choices=['native', 'int8', 'int4'], default=None)
    ap.add_argument('--bits', type=int, choices=[16, 8, 4], default=None,
                    help='deprecated alias of --format (16/8/4 -> native/int8/int4)')
    ap.add_argument('--group', type=int, default=128)
    ap.add_argument('--chunk', type=int, default=8192, help='rows per GPU batch')
    ap.add_argument('--embed-key', default=None)
    ap.add_argument('--qa-threshold', type=float, default=None,
                    help='max allowed relative L2 error; default 0.02 for int8, 0.12 for int4')
    args = ap.parse_args(argv)

    if args.group != 128:
        raise SystemExit(f'only --group 128 is supported, got {args.group}')

    cfg_path = osp.join(args.model, 'config.json')
    try:
        with open(cfg_path, encoding='utf-8') as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f'failed to read {cfg_path}: {e}')
    tied = bool(cfg.get('tie_word_embeddings', False))
    embed_key = find_embed_key(args.model, args.embed_key)
    embed_file = find_embed_file(args.model, embed_key)
    table_format = args.table_format
    if args.bits is not None:
        alias = BITS_TO_FORMAT[args.bits]
        if table_format is not None and table_format != alias:
            raise SystemExit(
                f'--bits {args.bits} conflicts with --format {table_format}')
        warnings.warn('--bits is deprecated; use --format '
                      f'{alias}', DeprecationWarning, stacklevel=2)
        table_format = alias
    if table_format is None:
        table_format = 'native'

    if table_format == 'native':
        if not tied:
            raise SystemExit('--format native requires tied embeddings (nothing to share)')
        write_sidecar(args.model, embed_key, table_format, {},
                      head_from_embed=True, qa={})
        print(f'wrote {SIDECAR_META_NAME} to {args.model} (native table, no tensors)')
        return 0

    dev = torch.device('cuda')

    if table_format == 'int4':
        q4_parts, s4_parts, z4_parts = [], [], []
        num4 = den4 = 0.0
        with safe_open(embed_file, 'pt') as f:
            if embed_key not in f.keys():
                raise SystemExit(f'{embed_key!r} not found in {embed_file}')
            t = f.get_tensor(embed_key)
            vocab, hidden = t.shape
            if hidden % args.group != 0:
                raise SystemExit(f'hidden {hidden} not divisible by group {args.group}')
            for r0 in range(0, vocab, args.chunk):
                x = t[r0:r0 + args.chunk].to(dev)
                q, s, z = quantize_table(x, table_format)
                dq = dequant_int4_simple(q, s, z, args.group, hidden)
                num4 += ((dq - x.float()).norm() ** 2).item()
                den4 += (x.float().norm() ** 2).item()
                q4_parts.append(q.cpu())
                s4_parts.append(s.cpu())
                z4_parts.append(z.cpu())
                print(f'rows {r0 + x.shape[0]}/{vocab}', flush=True)
        rel4 = (num4 / den4) ** 0.5
        print(f'int4 rel-L2 = {rel4:.5f}')
        qa_threshold = args.qa_threshold if args.qa_threshold is not None else 0.12
        if rel4 > qa_threshold:
            print(f'ERROR: int4 rel-L2 {rel4:.5f} > threshold {qa_threshold}', file=sys.stderr)
            return 1
        write_sidecar(args.model, embed_key, table_format,
                      {embed_key + EMBED_I4_SUFFIX: torch.cat(q4_parts, 0),
                       embed_key + EMBED_I4_SCALE_SUFFIX: torch.cat(s4_parts, 0),
                       embed_key + EMBED_I4_ZERO_SUFFIX: torch.cat(z4_parts, 0)},
                      head_from_embed=tied, qa={'int4_rel_l2': rel4})
        print(f'wrote {SIDECAR_NAME} + {SIDECAR_META_NAME} to {args.model}')
        return 0

    q8_parts, s8_parts = [], []
    num = den = 0.0
    with safe_open(embed_file, 'pt') as f:
        if embed_key not in f.keys():
            raise SystemExit(f'{embed_key!r} not found in {embed_file}')
        t = f.get_tensor(embed_key)
        vocab, hidden = t.shape
        if hidden % args.group != 0:
            raise SystemExit(f'hidden {hidden} not divisible by group {args.group}')
        for r0 in range(0, vocab, args.chunk):
            x = t[r0:r0 + args.chunk].to(dev)
            q, s, _ = quantize_table(x, table_format)
            dq = dequant_int8(q, s, args.group)
            num += ((dq - x.float()).norm() ** 2).item()
            den += (x.float().norm() ** 2).item()
            q8_parts.append(q.cpu())
            s8_parts.append(s.cpu())
            print(f'rows {r0 + x.shape[0]}/{vocab}', flush=True)

    q8 = torch.cat(q8_parts, dim=0)
    s8 = torch.cat(s8_parts, dim=0)
    rel = (num / den) ** 0.5
    print(f'int8 rel-L2 = {rel:.5f}')
    qa_threshold = args.qa_threshold if args.qa_threshold is not None else 0.02
    if rel > qa_threshold:
        print(f'ERROR: int8 rel-L2 {rel:.5f} > threshold {qa_threshold}', file=sys.stderr)
        return 1

    write_sidecar(args.model, embed_key, table_format,
                  {embed_key + EMBED_I8_SUFFIX: q8,
                   embed_key + EMBED_I8_SCALE_SUFFIX: s8},
                  head_from_embed=tied, qa={'int8_rel_l2': rel})
    print(f'wrote {SIDECAR_NAME} + {SIDECAR_META_NAME} to {args.model}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
