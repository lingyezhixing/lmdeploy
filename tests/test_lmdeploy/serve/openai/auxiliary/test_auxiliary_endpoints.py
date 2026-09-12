# Copyright (c) OpenMMLab. All rights reserved.
"""Validation tests for the /v1/embeddings and /v1/rerank handlers."""
from __future__ import annotations

import asyncio
import json

from lmdeploy.serve.openai.protocol import EmbeddingsRequest, RerankRequest


def _error_message(response):
    assert response.status_code == 400, response.body
    return json.loads(response.body)['message']


def test_embeddings_rejects_empty_string_input(auxiliary_endpoints):
    endpoints, _ = auxiliary_endpoints('embed')

    response = asyncio.run(endpoints['/v1/embeddings'](EmbeddingsRequest(input='')))

    assert 'empty' in _error_message(response).lower()


def test_embeddings_rejects_non_positive_dimensions(auxiliary_endpoints):
    endpoints, _ = auxiliary_endpoints('embed')

    response = asyncio.run(endpoints['/v1/embeddings'](EmbeddingsRequest(input='hello', dimensions=0)))

    assert 'dimensions' in _error_message(response)


def test_embeddings_rejects_dimensions_larger_than_hidden_size(auxiliary_endpoints):
    endpoints, _ = auxiliary_endpoints('embed')

    response = asyncio.run(endpoints['/v1/embeddings'](EmbeddingsRequest(input='hello', dimensions=9)))

    assert 'dimensions' in _error_message(response)


def test_embeddings_truncates_to_requested_dimensions(auxiliary_endpoints):
    endpoints, _ = auxiliary_endpoints('embed')

    response = asyncio.run(endpoints['/v1/embeddings'](EmbeddingsRequest(input='hello', dimensions=4)))

    assert len(response['data'][0]['embedding']) == 4


def test_rerank_rejects_non_positive_top_n(auxiliary_endpoints):
    endpoints, _ = auxiliary_endpoints('rerank')

    response = asyncio.run(
        endpoints['/v1/rerank'](RerankRequest(query='q', documents=['a', 'b'], top_n=0)))

    assert 'top_n' in _error_message(response)
