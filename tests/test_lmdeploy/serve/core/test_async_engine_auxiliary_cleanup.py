# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from types import SimpleNamespace

import pytest

from lmdeploy.serve.core.async_engine import AsyncEngine
from lmdeploy.serve.managers import SessionManager


class _FakeHandle:

    async def async_cancel(self, session_id: int):
        return None

    async def async_end(self, session_id: int):
        return None


class _FakeEngine:

    def create_instance(self):
        return _FakeHandle()


class _FailingRun:

    async def __aenter__(self):
        raise RuntimeError('engine exploded')

    async def __aexit__(self, *exc_info):
        return False


def _engine(*, backend='turbomind', session_len=32):
    engine = AsyncEngine.__new__(AsyncEngine)
    engine.session_mgr = SessionManager()
    engine.session_mgr.build_request_handle_pool(_FakeEngine(), 1)
    engine.backend = backend
    engine.session_len = session_len
    engine.safe_run = lambda *args, **kwargs: _FailingRun()
    return engine


def test_embeddings_removes_sessions_when_engine_raises():

    async def _run():
        engine = _engine()
        with pytest.raises(RuntimeError, match='engine exploded'):
            await engine.async_get_embeddings([[1, 2, 3]])

        assert engine.session_mgr.sessions == {}

    asyncio.run(_run())


def test_rerank_removes_sessions_when_engine_raises():

    async def _run():
        engine = _engine()
        engine.tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1])
        with pytest.raises(RuntimeError, match='engine exploded'):
            await engine.async_get_rerank_scores('query', ['doc'])

        assert engine.session_mgr.sessions == {}

    asyncio.run(_run())
