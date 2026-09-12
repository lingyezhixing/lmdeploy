"""Backend/task validation for the embed and rerank pipelines.

The pytorch backend cannot output last hidden states (embedding) and drops
``output_logits='generation'`` (rerank), so both tasks must be rejected before
an engine is created.
"""
import asyncio
import inspect

import pytest

from lmdeploy.archs import get_task
from lmdeploy.serve.core import AsyncEngine


def test_get_task_keeps_upstream_positional_contract():
    params = inspect.signature(get_task).parameters
    assert list(params) == ['backend', 'model_path', 'trust_remote_code', 'backend_config', 'task']
    assert params['task'].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize('task', ['embed', 'rerank'])
def test_get_task_rejects_pytorch(task):
    with pytest.raises(ValueError, match='turbomind'):
        get_task('pytorch', 'unused-model-path', task=task)


@pytest.mark.parametrize('task', ['embed', 'rerank'])
def test_get_task_allows_turbomind(task):
    got_task, pipeline_class = get_task('turbomind', 'unused-model-path', task=task)
    assert got_task == task
    assert pipeline_class is AsyncEngine


def _bare_engine(backend):
    engine = AsyncEngine.__new__(AsyncEngine)
    engine.backend = backend
    return engine


@pytest.mark.parametrize('task', ['embed', 'rerank'])
def test_embedding_and_rerank_require_turbomind_backend(task):
    engine = _bare_engine('pytorch')
    method = engine.async_get_embeddings if task == 'embed' else engine.async_get_rerank_scores
    args = ([[1, 2, 3]],) if task == 'embed' else ('query', ['doc'])
    with pytest.raises(ValueError, match='turbomind'):
        asyncio.run(method(*args))
