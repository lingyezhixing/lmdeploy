# Copyright (c) OpenMMLab. All rights reserved.
"""Shared fakes for auxiliary endpoint handler tests."""
from __future__ import annotations

import pytest
import torch
from fastapi import APIRouter

from lmdeploy.serve.openai.endpoints import auxiliary


class FakeTokenizer:

    def encode(self, text, **kwargs):
        return [] if text == '' else [1, 2, 3]


class FakeAsyncEngine:
    model_name = 'fake-model'

    def __init__(self, hidden_size=8):
        self.tokenizer = FakeTokenizer()
        self.hidden_size = hidden_size

    async def async_get_embeddings(self, input_ids):
        return [torch.ones(self.hidden_size) for _ in input_ids]

    async def async_get_rerank_scores(self, query, documents):
        return [(0.5, i) for i in range(len(documents))], 10


class FakeServerContext:

    def __init__(self, task):
        self.task = task
        self.async_engine = FakeAsyncEngine()


@pytest.fixture
def auxiliary_endpoints():
    """Build endpoint handlers for a given server task."""

    def _make(task='embed'):
        context = FakeServerContext(task)
        router = APIRouter()
        auxiliary.register(router, context)
        return {route.path: route.endpoint for route in router.routes}, context

    return _make
