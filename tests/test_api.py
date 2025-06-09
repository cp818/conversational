"""Integration tests for the RAG API endpoints."""

import json
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.api import app
from app.schema import ChatResponse, QueryResponse, IngestResponse


@pytest.fixture
def test_client():
    """Create a test client for the API."""
    return TestClient(app)


@pytest.fixture
def async_client():
    """Create an async client for the API."""
    return AsyncClient(app=app, base_url="http://test")


@pytest.fixture
def mock_retriever():
    """Mock the retriever."""
    with patch("app.api.get_retriever") as mock:
        retriever = AsyncMock()
        retriever.search.return_value = [
            {
                "id": "doc1", 
                "text": "This is a test document.",
                "metadata": {
                    "source": "test.pdf",
                    "page": 1
                },
                "score": 0.95
            }
        ]
        mock.return_value = retriever
        yield retriever


@pytest.fixture
def mock_generator():
    """Mock the generator."""
    with patch("app.api.get_generator") as mock:
        generator = AsyncMock()
        generator.generate_stream.return_value = AsyncMock(__aiter__=AsyncMock(return_value=iter([
            {"text": "This"},
            {"text": " is"},
            {"text": " a"},
            {"text": " generated"},
            {"text": " response."}
        ])))
        generator.generate.return_value = "This is a generated response."
        mock.return_value = generator
        yield generator


@pytest.fixture
def mock_embedding_client():
    """Mock the embedding client."""
    with patch("app.api.get_embedding_client") as mock:
        client = AsyncMock()
        client.embed_batch.return_value = [[0.1, 0.2, 0.3]]
        mock.return_value = client
        yield client


def test_health_check(test_client):
    """Test the health check endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@pytest.mark.asyncio
async def test_query_endpoint(async_client, mock_retriever, mock_generator):
    """Test the query endpoint."""
    # Set up the mock return values
    mock_retriever.search.return_value = [
        {
            "id": "doc1", 
            "text": "This is a test document.",
            "metadata": {
                "source": "test.pdf",
                "page": 1
            },
            "score": 0.95
        }
    ]
    mock_generator.generate.return_value = "This is a generated response."
    
    # Make the request
    response = await async_client.post(
        "/query",
        json={"query": "Test query", "max_results": 1, "similarity_threshold": 0.7}
    )
    
    # Verify the response
    assert response.status_code == 200
    data = response.json()
    assert data["generated_text"] == "This is a generated response."
    assert len(data["retrieved_contexts"]) == 1
    assert data["retrieved_contexts"][0]["text"] == "This is a test document."
    
    # Verify the right calls were made
    mock_retriever.search.assert_called_once()
    assert mock_retriever.search.call_args[0][0] == "Test query"
    assert mock_retriever.search.call_args[1]["top_k"] == 1
    assert mock_retriever.search.call_args[1]["similarity_threshold"] == 0.7
    
    mock_generator.generate.assert_called_once()
    assert "Test query" in mock_generator.generate.call_args[0][0]
    assert "This is a test document." in mock_generator.generate.call_args[0][0]


@pytest.mark.asyncio
async def test_ingest_endpoint(async_client, mock_retriever, mock_embedding_client):
    """Test the ingest endpoint."""
    # Set up the mock return values
    mock_retriever.upsert.return_value = 1
    
    # Make the request
    response = await async_client.post(
        "/ingest",
        json={
            "documents": [{
                "id": "test-doc-1",
                "text": "This is a test document content",
                "metadata": {
                    "source": "test.txt",
                    "author": "Test Author"
                }
            }]
        }
    )
    
    # Verify the response
    assert response.status_code == 200
    data = response.json()
    assert data["ingested_count"] == 1
    assert data["status"] == "success"
    
    # Verify the right calls were made
    mock_retriever.upsert.assert_called_once()
    assert len(mock_retriever.upsert.call_args[0][0]) == 1
    assert mock_retriever.upsert.call_args[0][0][0]["id"] == "test-doc-1"


@pytest.mark.parametrize("max_results", [1, 3, 5])
@pytest.mark.asyncio
async def test_query_parameter_max_results(async_client, mock_retriever, mock_generator, max_results):
    """Test max_results parameter in query endpoint."""
    # Set up mock return values for different max_results
    mock_docs = []
    for i in range(max_results):
        mock_docs.append({
            "id": f"doc{i}",
            "text": f"This is test document {i}.",
            "metadata": {
                "source": f"test{i}.pdf",
                "page": i + 1
            },
            "score": 0.95 - (i * 0.05)
        })
    mock_retriever.search.return_value = mock_docs
    
    # Make the request
    response = await async_client.post(
        "/query",
        json={"query": "Test query", "max_results": max_results}
    )
    
    # Verify response has correct number of results
    assert response.status_code == 200
    data = response.json()
    assert len(data["retrieved_contexts"]) == max_results
    
    # Verify retriever was called with correct parameters
    mock_retriever.search.assert_called_once()
    assert mock_retriever.search.call_args[1]["top_k"] == max_results


@pytest.mark.asyncio
async def test_chat_sse_endpoint():
    """Test the SSE chat endpoint.
    
    Note: This is a more complex test requiring mocking of the SSE response,
    so we're just providing a placeholder here. In a real implementation,
    you would test the SSE stream response.
    """
    pytest.skip("SSE endpoint test requires a more complex setup")


@pytest.mark.asyncio
async def test_chat_websocket_endpoint():
    """Test the WebSocket chat endpoint.
    
    Note: This is a more complex test requiring mocking of the WebSocket connection,
    so we're just providing a placeholder here. In a real implementation,
    you would test the WebSocket stream response.
    """
    pytest.skip("WebSocket endpoint test requires a more complex setup")


@pytest.mark.asyncio
async def test_latency_smoke(async_client, mock_retriever, mock_generator):
    """Smoke test for latency measurement."""
    if not pytest.smoke_tests_enabled:
        pytest.skip("Smoke tests not enabled")
    
    # Configure mock for fast response
    async def fast_search(*args, **kwargs):
        await asyncio.sleep(0.01)  # 10ms delay
        return [{
            "id": "doc1", 
            "text": "Fast test document.",
            "metadata": {"source": "test.pdf"},
            "score": 0.95
        }]
    
    async def fast_generate(*args, **kwargs):
        await asyncio.sleep(0.02)  # 20ms delay
        return "Fast generated response."
        
    mock_retriever.search.side_effect = fast_search
    mock_generator.generate.side_effect = fast_generate
    
    # Time the request
    start_time = asyncio.get_event_loop().time()
    response = await async_client.post("/query", json={"query": "Fast query test"})
    elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000
    
    # Verify response and timing
    assert response.status_code == 200
    assert response.json()["generated_text"] == "Fast generated response."
    
    # We should expect ~30-40ms for the total call with our mocks
    # But add some buffer for test environment variations
    assert elapsed_ms < 100, f"Query took {elapsed_ms}ms, which exceeds latency budget"
