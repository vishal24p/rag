from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import time

from fastapi.testclient import TestClient
import pytest

from src.api import app, get_query_service, _cached_component, _serialized_retriever
from src.query_service import QueryResult


@pytest.fixture
def client():
    calls = []
    result = QueryResult("request-1", "private", route="RAG", answer="answer",
                         contexts=[{"chunk_id": 2, "content": "second", "score": 0.2},
                                   {"chunk_id": 1, "content": "first", "score": 0.1}],
                         jev_decision_ms=10, total_request_ms=20)
    service = SimpleNamespace(run=lambda query: calls.append(query) or result)
    app.dependency_overrides[get_query_service] = lambda: service
    with TestClient(app) as test_client:
        yield test_client, calls, result
    app.dependency_overrides.clear()


def test_rag_preserves_sources_and_public_fields(client):
    api, calls, _ = client
    response = api.post("/api/query", json={"query": " original "})
    assert response.status_code == 200
    data = response.json()
    assert calls == [" original "]
    assert [chunk["chunk_id"] for chunk in data["contexts"]] == [2, 1]
    assert data["jev_decision_ms"] == 10 and data["total_request_ms"] == 20
    assert set(data) == {"request_id", "status", "route", "block_score", "blocked",
                         "answer", "contexts", "jev_decision_ms", "total_request_ms", "error"}


@pytest.mark.parametrize("body", [{}, {"query": ""}, {"query": " \n\t"},
                                    {"query": 123}, {"query": True},
                                    {"query": "x" * 10001}, {"query": "q", "history": []}])
def test_invalid_query_never_calls_pipeline(client, body):
    api, calls, _ = client
    assert api.post("/api/query", json=body).status_code == 422
    assert calls == []


@pytest.mark.parametrize("status,route,error,code", [
    ("ok", "DIRECT", None, 200),
    ("blocked", "RAG", None, 403),
    ("error", None, "gateway-timeout", 503),
    ("error", "RAG", "retrieval-error", 500),
    ("error", "DIRECT", "generation-error", 500),
])
def test_status_and_sources(client, status, route, error, code):
    api, _, result = client
    result.status, result.route, result.error = status, route, error
    result.blocked = status == "blocked"
    response = api.post("/api/query", json={"query": "q"})
    assert response.status_code == code
    assert response.json()["contexts"] == []
    assert response.json()["status"] == status


def test_health_never_calls_pipeline(client):
    api, calls, _ = client
    assert api.get("/api/health").json() == {"status": "ok"}
    assert calls == []


def test_query_accepts_exact_length_limit(client):
    api, calls, _ = client
    query = "x" * 10000
    assert api.post("/api/query", json={"query": query}).status_code == 200
    assert calls == [query]


@pytest.mark.parametrize("body", [None, [], {"query": None}, {"query": []}, {"query": {}}])
def test_query_rejects_invalid_json_types(client, body):
    import json

    api, calls, _ = client
    assert api.post("/api/query", content=json.dumps(body),
                    headers={"Content-Type": "application/json"}).status_code == 422
    assert calls == []


def test_malformed_json_never_calls_pipeline(client):
    api, calls, _ = client
    assert api.post("/api/query", content='{"query":',
                    headers={"Content-Type": "application/json"}).status_code == 422
    assert calls == []


@pytest.mark.parametrize("error,code", [("PRIVATE_EXCEPTION", 500),
                                         ("gateway-PRIVATE_EXCEPTION", 503)])
def test_unknown_errors_are_masked_and_contexts_removed(client, error, code):
    api, _, result = client
    result.status, result.error = "error", error
    response = api.post("/api/query", json={"query": "q"})
    assert response.status_code == code
    assert response.json()["error"] == "request-error"
    assert response.json()["contexts"] == []
    assert "PRIVATE_EXCEPTION" not in response.text


@pytest.mark.parametrize("field,value", [("chunk_id", True), ("chunk_id", "1"),
                                         ("content", 1), ("score", True),
                                         ("score", "0.5"), ("score", float("nan")),
                                         ("score", float("inf"))])
def test_context_rejects_invalid_types_and_nonfinite_scores(field, value):
    from pydantic import ValidationError
    from src.api import Context

    chunk = {"chunk_id": 1, "content": "passage", "score": 0.5}
    chunk[field] = value
    with pytest.raises(ValidationError):
        Context(**chunk)


def test_component_cache_is_lazy_and_serializes_first_construction():
    calls = []
    def build():
        calls.append(1)
        time.sleep(0.02)
        return object()
    factory = _cached_component(build)
    assert calls == []
    with ThreadPoolExecutor(max_workers=6) as workers:
        instances = list(workers.map(lambda _: factory(), range(6)))
    assert calls == [1]
    assert all(instance is instances[0] for instance in instances)


def test_failed_construction_can_retry():
    calls = []
    def build():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("initial failure")
        return object()
    factory = _cached_component(build)
    with pytest.raises(RuntimeError):
        factory()
    assert factory() is factory()
    assert len(calls) == 2


def test_serialized_retriever_prevents_concurrent_lazy_model_loads():
    loads = []
    loaded = False
    def retrieve(query):
        nonlocal loaded
        if not loaded:
            time.sleep(0.02)
            loads.append(1)
            loaded = True
        return ([], [])
    retriever = _serialized_retriever(SimpleNamespace(retrieve=retrieve))
    with ThreadPoolExecutor(max_workers=6) as workers:
        list(workers.map(retriever.retrieve, ["q"] * 6))
    assert loads == [1]
