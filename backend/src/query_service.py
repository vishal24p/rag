"""Single request boundary: gateway, route selection, and request timings."""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time
from uuid import uuid4

from src.config import settings
from src.gateway import GatewayDecision, GatewayError, GatewayProvider, OpenCodeJevProvider

log = logging.getLogger(__name__)
BLOCK_REFUSAL = "I can't help with that request."
SAFE_GATEWAY_ERRORS = {
    "gateway-missing-key", "gateway-invalid-timeout", "gateway-timeout",
    "gateway-http-error", "gateway-invalid-response", "gateway-invalid-score",
    "gateway-invalid-route",
}


@dataclass
class QueryResult:
    request_id: str
    query: str
    rewritten_query: str | None = None
    route: str | None = None
    block_score: float | None = None
    blocked: bool = False
    status: str = "ok"
    answer: str | None = None
    rrf: list = field(default_factory=list)
    contexts: list = field(default_factory=list)
    jev_decision_ms: float | None = None
    total_request_ms: float = 0.0
    error: str | None = None


def _retriever():
    from src.retrieval.reranked_retriever import RerankedRetriever
    return RerankedRetriever(settings.db_url)


def _generator():
    from src.generation.generator import Generator
    return Generator()


def _direct_generator():
    from src.generation.direct import DirectGenerator
    return DirectGenerator()


def _rewrite(query):
    from src.query_rewrite import rewrite_query
    return rewrite_query(query)


class QueryService:
    def __init__(
        self, gateway: GatewayProvider | None = None, retriever_factory=None,
        generator_factory=None, direct_generator_factory=None, rewriter=None,
    ):
        self.gateway = gateway
        self.retriever_factory = retriever_factory or _retriever
        self.generator_factory = generator_factory or _generator
        self.direct_generator_factory = direct_generator_factory or _direct_generator
        self.rewriter = rewriter or _rewrite

    def run(self, query: str, *, retrieve_only: bool = False) -> QueryResult:
        started = time.perf_counter()
        result = QueryResult(request_id=uuid4().hex, query=query)
        stage = "gateway"
        try:
            decision_started = time.perf_counter()
            try:
                decision = (self.gateway or OpenCodeJevProvider()).decide(query)
                decision = GatewayDecision(decision.request_block, decision.retrieval_route)
            finally:
                result.jev_decision_ms = (time.perf_counter() - decision_started) * 1000
            result.route = decision.retrieval_route
            result.block_score = decision.request_block
            if decision.request_block > settings.jev_block_threshold:
                result.blocked = True
                result.status = "blocked"
                result.answer = BLOCK_REFUSAL
            elif result.route == "RAG":
                stage = "retrieval"
                result.rewritten_query = self.rewriter(query)
                result.rrf, result.contexts = self.retriever_factory().retrieve(result.rewritten_query)
                if not retrieve_only:
                    stage = "generation"
                    result.answer = self.generator_factory().generate(query, result.contexts)
            elif not retrieve_only:
                stage = "generation"
                result.answer = self.direct_generator_factory().generate(query)
        except Exception as exc:
            result.status = "error"
            result.error = (
                str(exc)
                if isinstance(exc, GatewayError) and str(exc) in SAFE_GATEWAY_ERRORS
                else f"{stage}-error"
            )
            result.answer = "The request could not be processed. Please try again later."
        finally:
            result.total_request_ms = (time.perf_counter() - started) * 1000
            log.info(
                "request_result request_id=%s block_score=%s blocked=%s route=%s "
                "status=%s jev_decision_ms=%s total_request_ms=%.3f error=%s",
                result.request_id, result.block_score, result.blocked, result.route,
                result.status, result.jev_decision_ms, result.total_request_ms,
                result.error,
            )
        return result
