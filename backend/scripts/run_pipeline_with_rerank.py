r"""End-to-end demo: same query, RRF and reranked views side by side.

   1. RRF output
   2. RRF + Jev rerank

Usage (PowerShell, venv active):
    $env:PYTHONPATH = "C:\Users\visha\rag"
    python scripts\run_pipeline_with_rerank.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402
from src.logging_setup import setup_logging  # noqa: E402
from src.query_service import QueryService  # noqa: E402

QUERY = "How do H1, H2, H3 heading levels work in markdown?"


def show(name, hits, n_chars=160):
    print(f"\n== {name} ({len(hits)} hits) ==")
    for i, h in enumerate(hits, 1):
        preview = h["content"].replace("\n", " ")[:n_chars]
        print(f"  {i}. id={h['chunk_id']:<5}  score={h['score']:.4f}  | {preview}...")


def main():
    setup_logging()
    original_query = QUERY
    print(
        f"[cfg] dense_top_k={settings.dense_top_k} "
        f"sparse_top_k={settings.sparse_top_k} "
        f"rrf_top_n={settings.rrf_top_n} "
        f"rerank_model={settings.jev_model} "
        f"rerank_top_n={settings.rerank_top_n} "
        f"jev_model={settings.jev_model} "
        f"query_rewrite={'on' if settings.query_rewrite_enabled else 'off'} "
        f"query_rewrite_model={settings.query_rewrite_model}"
    )

    result = QueryService().run(original_query, retrieve_only=True)
    print(f"[request] id={result.request_id} status={result.status} route={result.route}")
    print(f"[timing] jev_decision_ms={result.jev_decision_ms} "
          f"total_request_ms={result.total_request_ms:.2f}")
    if result.status != "ok":
        print(result.answer)
        return 1 if result.status == "error" else 0
    if result.route == "DIRECT":
        print("[DIRECT] Retrieval not required; diagnostic finished without generation.")
        return 0
    query = result.rewritten_query
    if query != original_query:
        print(f"\n[rewrite] {original_query}\n     -> {query}")

    rrf_pool, reranked = result.rrf, result.contexts

    show(f"RRF pool (top {settings.rrf_top_n})", rrf_pool)
    show(f"RERANKED (final top {settings.rerank_top_n})", reranked)

    overlap_ids = {h["chunk_id"] for h in rrf_pool[: settings.rerank_top_n]}.intersection(
        {h["chunk_id"] for h in reranked}
    )
    print(f"\n[summary] rerank kept/added {len(overlap_ids)} of RRF top-{settings.rerank_top_n}; "
          f"swapped in {settings.rerank_top_n - len(overlap_ids)} from the RRF pool")
    return 0


if __name__ == "__main__":
    sys.exit(main())
