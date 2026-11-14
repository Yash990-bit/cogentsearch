"""
Tests for CogentSearch FastAPI REST & SSE Streaming Endpoints.
Validates health check, document telemetry, preload corpus, chat query reasoning,
streaming events, and visual crop retrieval.
"""

import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "CogentSearch" in data["service"]


def test_list_documents_initially_empty():
    response = client.get("/api/documents")
    assert response.status_code == 200
    data = response.json()
    assert "total_documents" in data
    assert "documents" in data


def test_preload_enterprise_corpus_endpoint():
    response = client.post("/api/preload")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_documents"] >= 1
    assert data["total_chunks"] >= 1


def test_chat_query_endpoint():
    payload = {"query": "What are the quarterly financial metrics and revenue figures?"}
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 10
    assert "steps" in data
    assert len(data["steps"]) >= 4
    # Verify steps contain core agentic loop stages
    step_names = [s["step"] for s in data["steps"]]
    assert "rewrite" in step_names
    assert "hybrid_search" in step_names
    assert "rerank" in step_names
    assert "grade" in step_names
    assert "generate" in step_names
    assert "citations" in data
    assert len(data["citations"]) >= 1


def test_agent_sse_stream_endpoint():
    payload = {"query": "Explain cluster throughput benchmarks"}
    with client.stream("POST", "/api/agent/stream", json=payload) as response:
        assert response.status_code == 200
        lines = list(response.iter_lines())
        assert len(lines) > 0
        data_lines = [l for l in lines if l.startswith("data: ")]
        assert len(data_lines) >= 4
        # At least one event should have answer or step_start
        assert any("step_start" in l for l in data_lines)
        assert any("answer" in l for l in data_lines)


def test_get_bounding_box_crop_endpoint():
    # Test valid crop on sample PDF
    response = client.get(
        "/api/crop",
        params={
            "file": "doc_02_multipage_table.pdf",
            "page": 1,
            "x0": 50,
            "y0": 100,
            "x1": 400,
            "y1": 250,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 100
