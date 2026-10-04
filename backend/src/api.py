"""Thin HTTP adapter for the existing stateless query service."""
from functools import wraps
from threading import Lock
from types import SimpleNamespace
from typing import Literal

from fastapi import Depends, FastAPI, Response
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from src.logging_setup import setup_logging
from src.query_service import QueryService, SAFE_GATEWAY_ERRORS

setup_logging()
app = FastAPI(title="Jev RAG API")


def _cached_component(build):
    """Cache successful construction; concurrent first callers share one instance."""
    lock = Lock()
    instance = None

    @wraps(build)
    def get():
        nonlocal instance
        with lock:
            if instance is None:
                instance = build()
            return instance
    return get


def _serialized_retriever(component):
    lock = Lock()

    def retrieve(query):
        # ponytail: serial retrieval protects lazy model loading; use a dedicated
        # initialization lock in the reranker if local request throughput matters.
        with lock:
            return component.retrieve(query)
    return SimpleNamespace(retrieve=retrieve)


@_cached_component
def _retriever():
    from src.config import settings
    from src.retrieval.reranked_retriever import RerankedRetriever
    return _serialized_retriever(RerankedRetriever(settings.db_url))


@_cached_component
def _generator():
    from src.generation.generator import Generator
    return Generator()


@_cached_component
def _direct_generator():
    from src.generation.direct import DirectGenerator
    return DirectGenerator()


@_cached_component
def get_query_service() -> QueryService:
    return QueryService(
        retriever_factory=_retriever, generator_factory=_generator,
        direct_generator_factory=_direct_generator,
    )


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: StrictStr = Field(min_length=1, max_length=10000)

    @field_validator("query")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class Context(BaseModel):
    chunk_id: int = Field(strict=True)
    content: StrictStr
    score: float = Field(strict=True, allow_inf_nan=False)


class QueryResponse(BaseModel):
    request_id: str
    status: Literal["ok", "blocked", "error"]
    route: Literal["RAG", "DIRECT"] | None
    block_score: float | None
    blocked: bool
    answer: str | None
    contexts: list[Context]
    jev_decision_ms: float | None
    total_request_ms: float
    error: str | None


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/query", response_model=QueryResponse)
def query(
    request: QueryRequest, response: Response,
    service: QueryService = Depends(get_query_service),
) -> QueryResponse:
    result = service.run(request.query)
    if result.status == "blocked":
        response.status_code = 403
    elif result.status == "error":
        response.status_code = 503 if (result.error or "").startswith("gateway-") else 500
    safe_errors = SAFE_GATEWAY_ERRORS | {"gateway-error", "retrieval-error", "generation-error"}
    error = (
        result.error if result.error in safe_errors
        else "request-error" if result.error else None
    )
    return QueryResponse(
        request_id=result.request_id, status=result.status, route=result.route,
        block_score=result.block_score, blocked=result.blocked, answer=result.answer,
        contexts=result.contexts if result.status == "ok" and result.route == "RAG" else [],
        jev_decision_ms=result.jev_decision_ms, total_request_ms=result.total_request_ms,
        error=error,
    )
