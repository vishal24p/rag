import httpx
import pytest

from src.gateway import GatewayError, OpenCodeJevProvider


def response(score=0.1, route="RAG"):
    return {"answers": {"request_block": {"type": "noul", "noul": score}, "retrieval_route": {"type": "choice", "choice": route, "confidence": 0.9, "probabilities": {"RAG": 0.9, "DIRECT": 0.1}}}}


def provider(payload, captured=None):
    def handle(request):
        if captured is not None:
            captured.append(request)
        return httpx.Response(200, json=payload)
    return OpenCodeJevProvider(api_key="test-key", transport=httpx.MockTransport(handle))


def test_single_request_two_questions_original_state():
    import json
    calls = []
    decision = provider(response(), calls).decide("original query")
    assert decision.request_block == 0.1 and decision.retrieval_route == "RAG"
    assert len(calls) == 1
    payload = json.loads(calls[0].content)
    assert payload["state"] == "original query"
    assert payload["model"] == "jev-1.13"
    assert len(payload["questions"]) == 2
    assert set(payload["questions"]) == {"request_block", "retrieval_route"}
    assert payload["questions"]["request_block"]["type"] == "noul"
    assert payload["questions"]["retrieval_route"]["type"] == "choice"
    assert set(payload["questions"]["retrieval_route"]["criteria"]) == {"RAG", "DIRECT"}
    assert "choices" not in payload["questions"]["retrieval_route"]


@pytest.mark.parametrize("score", [True, "0.1", -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_scores_rejected(score):
    # Exercise the parser directly: JSON transports reject nonfinite values too.
    with pytest.raises(GatewayError):
        OpenCodeJevProvider.parse_response(response(score))


@pytest.mark.parametrize("payload", [{}, {"answers": []}, response(route="other"), {"answers": {"request_block": {"type": "choice", "noul": 0.1}, "retrieval_route": {"type": "choice", "choice": "RAG"}}}])
def test_malformed_response_rejected(payload):
    with pytest.raises(GatewayError):
        provider(payload).decide("q")


def test_provider_failure_does_not_retry_or_expose_details():
    calls = []
    def handle(request):
        calls.append(request)
        raise httpx.ConnectError("secret credential query", request=request)
    gateway = OpenCodeJevProvider(api_key="test-key", transport=httpx.MockTransport(handle))
    with pytest.raises(GatewayError) as raised:
        gateway.decide("q")
    assert len(calls) == 1
    assert "secret" not in str(raised.value)


def test_missing_key_never_calls_provider(monkeypatch):
    calls = []
    gateway = OpenCodeJevProvider(api_key="", transport=httpx.MockTransport(lambda request: calls.append(request)))
    with pytest.raises(GatewayError):
        gateway.decide("q")
    assert calls == []


@pytest.mark.parametrize("field,value", [("confidence", True), ("confidence", float("nan")), ("probabilities", {"RAG": "0.8", "DIRECT": 0.2}), ("probabilities", [])])
def test_invalid_choice_metadata_rejected(field, value):
    payload = response()
    payload["answers"]["retrieval_route"][field] = value
    with pytest.raises(GatewayError):
        OpenCodeJevProvider.parse_response(payload)


def test_timeout_is_single_request():
    calls = []
    def handle(request):
        calls.append(request)
        raise httpx.ReadTimeout("secret", request=request)
    gateway = OpenCodeJevProvider(api_key="test", transport=httpx.MockTransport(handle))
    with pytest.raises(GatewayError, match="gateway-timeout"):
        gateway.decide("q")
    assert len(calls) == 1


@pytest.mark.parametrize("route,probabilities", [
    ("RAG", {"RAG": 0.9, "DIRECT": 0.9}),
    ("DIRECT", {"RAG": 0.9, "DIRECT": 0.1}),
])
def test_inconsistent_probability_distribution_rejected(route, probabilities):
    payload = response(route=route)
    payload["answers"]["retrieval_route"]["probabilities"] = probabilities
    with pytest.raises(GatewayError):
        OpenCodeJevProvider.parse_response(payload)


def test_tied_probabilities_allow_either_choice():
    payload = response(route="DIRECT")
    payload["answers"]["retrieval_route"]["probabilities"] = {"RAG": 0.5, "DIRECT": 0.5}
    assert OpenCodeJevProvider.parse_response(payload).retrieval_route == "DIRECT"
