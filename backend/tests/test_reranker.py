from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest


def _hit(chunk_id: int, content: str, score: float = 0.01) -> dict:
    return {"chunk_id": chunk_id, "content": content, "score": score, "source": "test"}


def _reranker(monkeypatch, tmp_path, payload=None, handler=None, *, api_key="test-key"):
    import src.retrieval.reranker as module
    from src.retrieval.reranker import JevReranker

    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Judge whether any phrase is needed to answer the question.", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "settings",
        replace(
            module.settings,
            jev_api_key=api_key,
            jev_api_url="https://example.test/systemone",
            jev_model="jev-test",
            jev_timeout_seconds=3,
            jev_rerank_prompt_path=str(prompt_path),
        ),
    )
    calls = []

    def respond(request):
        calls.append(request)
        if handler:
            return handler(request)
        return httpx.Response(200, json=payload)

    reranker = JevReranker(transport=httpx.MockTransport(respond))
    return reranker, calls


def test_reranker_batches_candidates_in_one_request_and_sorts_by_noul(monkeypatch, tmp_path):
    payload = {"answers": {
        "chunk_2": {"type": "noul", "noul": 0.6},
        "chunk_1": {"type": "noul", "noul": 0.9},
        "chunk_0": {"type": "noul", "noul": 0.2},
    }}
    reranker, calls = _reranker(monkeypatch, tmp_path, payload)
    candidates = [_hit(11, "chunk about pods"), _hit(12, "deployment rollout details"), _hit(13, "service routing")]

    out = reranker.rerank("How do deployments roll out?", candidates, top_n=2)

    assert len(calls) == 1
    request = calls[0]
    body = json.loads(request.content)
    assert request.url == "https://example.test/systemone"
    assert request.headers["Authorization"] == "Bearer test-key"
    assert body["model"] == "jev-test"
    assert body["state"] == "How do deployments roll out?"
    assert list(body["questions"]) == ["chunk_0", "chunk_1", "chunk_2"]
    assert all(question["type"] == "noul" for question in body["questions"].values())
    assert all("Judge whether any phrase" in q["instructions"] for q in body["questions"].values())
    assert all(c["content"] in body["questions"][f"chunk_{i}"]["instructions"] for i, c in enumerate(candidates))
    assert [item["chunk_id"] for item in out] == [12, 13]
    assert [item["score"] for item in out] == [0.9, 0.6]
    assert out[0]["source"] == "test"
    assert all(c["score"] == 0.01 for c in candidates)
    assert request.extensions["timeout"]["read"] == 3
    for i, candidate in enumerate(candidates):
        instructions = body["questions"][f"chunk_{i}"]["instructions"]
        assert all(other["content"] not in instructions for other in candidates if other is not candidate)


def test_reranker_empty_input_skips_request(monkeypatch, tmp_path):
    reranker, calls = _reranker(monkeypatch, tmp_path, {"answers": {}})
    assert reranker.rerank("q", []) == []
    assert calls == []


def test_empty_candidates_need_no_key_or_prompt(monkeypatch, tmp_path):
    import src.retrieval.reranker as module

    reranker, calls = _reranker(monkeypatch, tmp_path, api_key="")
    monkeypatch.setattr(module, "settings", replace(
        module.settings, jev_rerank_prompt_path=str(tmp_path / "missing.md"),
    ))
    assert reranker.rerank("q", []) == []
    assert calls == []


def test_directory_prompt_fails_before_http(monkeypatch, tmp_path):
    import src.retrieval.reranker as module
    from src.gateway import GatewayError

    reranker, calls = _reranker(monkeypatch, tmp_path)
    monkeypatch.setattr(module, "settings", replace(
        module.settings, jev_rerank_prompt_path=str(tmp_path),
    ))
    with pytest.raises(GatewayError, match="gateway-invalid-response"):
        reranker.rerank("q", [_hit(1, "chunk")])
    assert calls == []


def test_reranker_logs_counts_without_private_content(monkeypatch, tmp_path, caplog):
    import logging

    reranker, _ = _reranker(monkeypatch, tmp_path, {
        "answers": {"chunk_0": {"type": "noul", "noul": 0.5}},
    }, api_key="PRIVATE_KEY")
    with caplog.at_level(logging.INFO, logger="src.retrieval.reranker"):
        reranker.rerank("PRIVATE_QUERY", [_hit(1, "PRIVATE_CHUNK")])
    assert "in=1 out=1" in caplog.text
    assert "PRIVATE_" not in caplog.text


@pytest.mark.parametrize("answers", [
    {},
    {"chunk_0": {"type": "choice", "noul": 0.9}},
    {"chunk_0": {"type": "noul", "noul": True}},
    {"chunk_0": {"type": "noul", "noul": -0.1}},
    {"chunk_0": {"type": "noul", "noul": 1.1}},
    {"chunk_0": {"type": "noul", "noul": float("nan")}},
    {"chunk_0": {"type": "noul", "noul": float("inf")}},
    {"chunk_0": {"type": "noul", "noul": "0.9"}},
    {"chunk_0": {"type": "noul", "noul": 10**400}},
    {"chunk_0": {"type": "noul"}},
    {"chunk_0": None},
    {"chunk_0": {"type": "noul", "noul": 0.9}, "extra": {"type": "noul", "noul": 0.1}},
])
def test_reranker_rejects_invalid_answer_mapping(monkeypatch, tmp_path, answers):
    from src.gateway import GatewayError

    reranker, _ = _reranker(
        monkeypatch, tmp_path,
        handler=lambda request: httpx.Response(200, content=json.dumps({"answers": answers})),
    )
    with pytest.raises(GatewayError, match="gateway-invalid-response"):
        reranker.rerank("q", [_hit(1, "chunk")])


def test_reranker_rejects_more_than_32_candidates_before_request(monkeypatch, tmp_path):
    from src.gateway import GatewayError

    reranker, calls = _reranker(monkeypatch, tmp_path, {"answers": {}})
    with pytest.raises(GatewayError, match="gateway-invalid-response"):
        reranker.rerank("q", [_hit(i, f"chunk {i}") for i in range(33)])
    assert calls == []


@pytest.mark.parametrize("handler,expected", [
    (lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("secret query", request=request)), "gateway-timeout"),
    (lambda request: httpx.Response(503, text="secret response"), "gateway-http-error"),
    (lambda request: httpx.Response(302, headers={"Location": "https://secret.test"}), "gateway-http-error"),
    (lambda request: (_ for _ in ()).throw(httpx.ConnectError("secret URL", request=request)), "gateway-http-error"),
    (lambda request: httpx.Response(200, text="secret non-JSON response"), "gateway-invalid-response"),
])
def test_reranker_provider_failures_are_sanitized(monkeypatch, tmp_path, handler, expected):
    from src.gateway import GatewayError

    reranker, calls = _reranker(monkeypatch, tmp_path, handler=handler)
    with pytest.raises(GatewayError, match=expected) as error:
        reranker.rerank("private query", [_hit(1, "private chunk")])
    assert len(calls) == 1
    assert "secret" not in str(error.value)
    assert "private" not in str(error.value)


def test_reranker_missing_key_never_calls_provider(monkeypatch, tmp_path):
    from src.gateway import GatewayError

    reranker, calls = _reranker(monkeypatch, tmp_path, {"answers": {}}, api_key="")
    with pytest.raises(GatewayError, match="gateway-missing-key"):
        reranker.rerank("q", [_hit(1, "chunk")])
    assert calls == []


def test_reranker_missing_prompt_is_safe_and_never_calls_provider(monkeypatch, tmp_path):
    import src.retrieval.reranker as module
    from src.gateway import GatewayError
    from src.retrieval.reranker import JevReranker

    monkeypatch.setattr(module, "settings", replace(
        module.settings, jev_api_key="test-key", jev_rerank_prompt_path=str(tmp_path / "missing.md"),
    ))
    reranker = JevReranker(transport=httpx.MockTransport(lambda request: pytest.fail("unexpected request")))
    with pytest.raises(GatewayError, match="gateway-invalid-response"):
        reranker.rerank("q", [_hit(1, "chunk")])


@pytest.mark.parametrize("contents", [b"  \n", b"\xff"])
def test_reranker_invalid_prompt_is_safe(monkeypatch, tmp_path, contents):
    import src.retrieval.reranker as module
    from src.gateway import GatewayError

    reranker, calls = _reranker(monkeypatch, tmp_path)
    Path(module.settings.jev_rerank_prompt_path).write_bytes(contents)
    with pytest.raises(GatewayError, match="gateway-invalid-response"):
        reranker.rerank("q", [_hit(1, "chunk")])
    assert calls == []


def test_reranker_accepts_32_candidates_and_defaults_top_n(monkeypatch, tmp_path):
    import src.retrieval.reranker as module

    payload = {"answers": {f"chunk_{i}": {"type": "noul", "noul": i / 31} for i in range(32)}}
    reranker, calls = _reranker(monkeypatch, tmp_path, payload)
    monkeypatch.setattr(module, "settings", replace(module.settings, rerank_top_n=3))
    out = reranker.rerank("q", [_hit(i, f"chunk {i}") for i in range(32)])
    assert len(calls) == 1
    assert len(json.loads(calls[0].content)["questions"]) == 32
    assert [item["chunk_id"] for item in out] == [31, 30, 29]
    assert [item["score"] for item in out] == [1, 30 / 31, 29 / 31]


@pytest.mark.parametrize("payload", [None, [], {}, {"answers": []}, {"answers": None}])
def test_reranker_rejects_malformed_envelope(monkeypatch, tmp_path, payload):
    from src.gateway import GatewayError

    reranker, _ = _reranker(monkeypatch, tmp_path, payload)
    with pytest.raises(GatewayError, match="gateway-invalid-response"):
        reranker.rerank("q", [_hit(1, "chunk")])


def test_reranker_keeps_tie_order_and_duplicate_chunk_ids(monkeypatch, tmp_path):
    payload = {"answers": {
        "chunk_0": {"type": "noul", "noul": 0.5},
        "chunk_1": {"type": "noul", "noul": 0.5},
    }}
    reranker, _ = _reranker(monkeypatch, tmp_path, payload)
    candidates = [_hit(1, "first"), _hit(1, "second")]
    out = reranker.rerank("q", candidates, top_n=5)
    assert [item["content"] for item in out] == ["first", "second"]
    assert reranker.rerank("q", candidates, top_n=0) == []


def test_reranker_resolves_default_prompt_from_backend_root(monkeypatch, tmp_path):
    import src.retrieval.reranker as module

    reranker, calls = _reranker(monkeypatch, tmp_path, {"answers": {"chunk_0": {"type": "noul", "noul": 1}}})
    monkeypatch.setattr(module, "settings", replace(module.settings, jev_rerank_prompt_path="prompts/jev_reranker.md"))
    monkeypatch.chdir(tmp_path)
    reranker.rerank("q", [_hit(1, "chunk")])
    prompt = (Path(module.__file__).resolve().parents[2] / "prompts/jev_reranker.md").read_text(encoding="utf-8").strip()
    assert prompt in json.loads(calls[0].content)["questions"]["chunk_0"]["instructions"]


def test_reranked_retriever_defaults_to_jev_and_preserves_injection():
    from src.retrieval.reranker import JevReranker
    from src.retrieval.reranked_retriever import RerankedRetriever

    class Hybrid:
        def retrieve(self, query, **kwargs):
            return [_hit(i, f"chunk {i}") for i in range(4)]

    class Reranker:
        def rerank(self, query, candidates, top_n):
            assert query == "q"
            assert [item["chunk_id"] for item in candidates] == [0, 1]
            assert top_n == 1
            return candidates[:top_n]

    hybrid = Hybrid()
    assert isinstance(RerankedRetriever("unused", hybrid=hybrid).reranker, JevReranker)
    retriever = RerankedRetriever("unused", hybrid=hybrid, reranker=Reranker())
    pool, out = retriever.retrieve("q", rrf_top_n=2, top_n=1)
    assert len(pool) == 4
    assert out == pool[:1]


def test_reranked_retriever_uses_configured_retrieval_and_rerank_limits(monkeypatch):
    import src.retrieval.reranked_retriever as module

    monkeypatch.setattr(module, "settings", replace(
        module.settings, dense_top_k=7, sparse_top_k=9, rrf_top_n=4, rerank_top_n=2,
    ))

    class Hybrid:
        def retrieve(self, query, **kwargs):
            assert query == "q"
            assert kwargs == {"top_k_dense": 7, "top_k_sparse": 9, "top_k": 4}
            return [_hit(i, f"chunk {i}") for i in range(6)]

    class Reranker:
        def rerank(self, query, candidates, top_n):
            assert query == "q"
            assert len(candidates) == 4
            assert top_n == 2
            return candidates[:top_n]

    pool, out = module.RerankedRetriever("unused", hybrid=Hybrid(), reranker=Reranker()).retrieve("q")
    assert len(pool) == 6
    assert out == pool[:2]
