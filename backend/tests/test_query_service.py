from types import SimpleNamespace

import pytest

from src.gateway import GatewayDecision
from src.query_service import QueryService


@pytest.mark.parametrize("route", ["RAG", "DIRECT"])
def test_block_stops_every_downstream_factory(route):
    def forbidden():
        raise AssertionError("downstream factory called")

    gateway = SimpleNamespace(decide=lambda query: GatewayDecision(0.6, route))
    result = QueryService(gateway, forbidden, forbidden, forbidden, forbidden).run("query")
    assert result.status == "blocked"
    assert result.blocked
    assert result.jev_decision_ms >= 0
    assert result.total_request_ms >= result.jev_decision_ms


def test_direct_skips_retrieval_and_rewrite():
    calls = []
    def forbidden(*args):
        raise AssertionError("RAG component called")
    gateway = SimpleNamespace(decide=lambda query: GatewayDecision(0.1, "DIRECT"))
    generator = SimpleNamespace(generate=lambda query: calls.append(query) or "hello")
    result = QueryService(gateway, forbidden, forbidden, lambda: generator, forbidden).run("hi")
    assert result.answer == "hello"
    assert result.route == "DIRECT"
    assert result.contexts == result.rrf == []
    assert calls == ["hi"]


def test_rag_preserves_retrieval_rewrite_and_original_generation_query():
    calls = []
    chunks = [{"content": "context", "chunk_id": 1, "score": 0.8}]
    retriever = SimpleNamespace(retrieve=lambda query: calls.append(("retrieve", query)) or (chunks, chunks))
    generator = SimpleNamespace(generate=lambda query, contexts: calls.append(("generate", query, contexts)) or "answer")
    gateway = SimpleNamespace(decide=lambda query: GatewayDecision(0.1, "RAG"))
    result = QueryService(gateway, lambda: retriever, lambda: generator, rewriter=lambda query: "rewritten").run("original")
    assert calls == [("retrieve", "rewritten"), ("generate", "original", chunks)]
    assert result.answer == "answer"
    assert result.rewritten_query == "rewritten"


@pytest.mark.parametrize("route", ["RAG", "DIRECT"])
def test_retrieve_only_never_constructs_generator(route):
    def forbidden():
        raise AssertionError("generator constructed")
    gateway = SimpleNamespace(decide=lambda query: GatewayDecision(0.0, route))
    retriever = SimpleNamespace(retrieve=lambda query: ([], []))
    result = QueryService(gateway, lambda: retriever, forbidden, forbidden, lambda query: query).run("q", retrieve_only=True)
    assert result.status == "ok"
    assert result.answer is None


def test_gateway_failure_is_safe_and_timed(caplog):
    def fail(query):
        raise RuntimeError("SECRET raw query")
    result = QueryService(gateway=SimpleNamespace(decide=fail)).run("private query")
    assert result.status == "error" and not result.blocked
    assert result.error == "gateway-error"
    assert result.jev_decision_ms >= 0
    assert "SECRET" not in caplog.text


def test_untrusted_gateway_error_message_is_not_logged(caplog):
    from src.gateway import GatewayError
    def fail(query):
        raise GatewayError("SECRET")
    result = QueryService(gateway=SimpleNamespace(decide=fail)).run("private")
    assert result.error == "gateway-error"
    assert "SECRET" not in caplog.text


def test_request_metrics_are_logged_without_raw_query(caplog):
    import logging
    gateway = SimpleNamespace(decide=lambda query: GatewayDecision(1, "DIRECT"))
    with caplog.at_level(logging.INFO, logger="src.query_service"):
        result = QueryService(gateway).run("private text")
    assert result.request_id in caplog.text
    assert "jev_decision_ms=" in caplog.text
    assert "total_request_ms=" in caplog.text
    assert "private text" not in caplog.text


def test_invalid_replacement_provider_decision_fails_closed():
    gateway = SimpleNamespace(decide=lambda query: SimpleNamespace(request_block=0, retrieval_route="UNKNOWN"))
    result = QueryService(gateway=gateway).run("q", retrieve_only=True)
    assert result.status == "error" and not result.blocked


def test_threshold_equality_is_allowed():
    gateway = SimpleNamespace(decide=lambda query: GatewayDecision(0.5, "DIRECT"))
    result = QueryService(gateway=gateway).run("q", retrieve_only=True)
    assert result.status == "ok" and not result.blocked


def test_downstream_error_has_safe_answer():
    def fail():
        raise RuntimeError("SECRET")
    gateway = SimpleNamespace(decide=lambda query: GatewayDecision(0, "DIRECT"))
    result = QueryService(gateway=gateway, direct_generator_factory=fail).run("q")
    assert result.status == "error"
    assert result.error == "generation-error"
    assert result.answer and "SECRET" not in result.answer
