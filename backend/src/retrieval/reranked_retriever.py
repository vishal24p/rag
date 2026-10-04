"""Composite retriever: Hybrid (RRF) -> CrossEncoderReranker.

Pipeline:
   query
     -> HybridRetriever.retrieve(query, top_k_dense=N, top_k_sparse=N, top_k=rrf)
        (returns the configured post-RRF candidate list)
     -> CrossEncoderReranker.rerank(query, candidates[:rrf], top_n=final)
        (re-scores by query/chunk relevance and keeps the best)
"""
from __future__ import annotations

import logging
import time

from src.config import settings
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker

log = logging.getLogger(__name__)


class RerankedRetriever:
    def __init__(
        self,
        connection_string: str,
        hybrid: HybridRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.hybrid = hybrid or HybridRetriever(connection_string)
        self.reranker = reranker or CrossEncoderReranker()

    def retrieve(
        self,
        query: str,
        top_k_dense: int | None = None,
        top_k_sparse: int | None = None,
        rrf_top_n: int | None = None,
        top_n: int | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """Return (rrf_output, reranked_top_n).

        The first list lets callers inspect the RRF output. Only its
        configured post-RRF prefix is passed to the cross-encoder; the second
        list is the final context passed to the generator.
        """
        dense_limit = top_k_dense if top_k_dense is not None else settings.dense_top_k
        sparse_limit = top_k_sparse if top_k_sparse is not None else settings.sparse_top_k
        rrf_limit = rrf_top_n if rrf_top_n is not None else settings.rrf_top_n
        final = top_n if top_n is not None else settings.rerank_top_n

        log.info(
            "pipeline start: dense=%d sparse=%d rrf=%d rerank top_n=%d",
            dense_limit, sparse_limit, rrf_limit, final,
        )
        t0 = time.perf_counter()
        rrf_pool = self.hybrid.retrieve(
            query,
            top_k_dense=dense_limit,
            top_k_sparse=sparse_limit,
            top_k=rrf_limit,
        )
        rerank_input = rrf_pool[:rrf_limit]
        reranked = self.reranker.rerank(query, rerank_input, top_n=final)
        log.info(
            "pipeline end: rrf=%d -> reranked=%d (total %.1f ms)",
            len(rrf_pool), len(reranked),
            (time.perf_counter() - t0) * 1000.0,
        )
        return rrf_pool, reranked
