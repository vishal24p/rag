r"""Jev-gated RAG or DIRECT answer demo.

Reads the query from argv if given, otherwise prompts on stdin.
Prints the RRF pool, the reranked top-N, then the generator's answer.

Usage:
    python scripts\run_rag.py                # interactive prompt
    python scripts\run_rag.py "your question?"  # one-shot
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402
from src.logging_setup import setup_logging  # noqa: E402
from src.query_service import QueryService  # noqa: E402

setup_logging()
log = logging.getLogger("run_rag")

DEFAULT_QUERY = "How do H1, H2, H3 heading levels work in markdown?"
PROMPT_LABEL = "[you]"


def resolve_query(argv: list[str]) -> str:
    if len(argv) > 1 and argv[1].strip():
        return " ".join(argv[1:]).strip()
    try:
        return input(f"{PROMPT_LABEL} ").strip()
    except EOFError:
        return ""


def show(name: str, hits: list[dict]) -> None:
    print(f"\n== {name} ({len(hits)} hits) ==")
    for i, h in enumerate(hits, 1):
        preview = h["content"].replace("\n", " ")[:160]
        print(f"  {i}. id={h['chunk_id']:<5}  score={h['score']:.4f}  | {preview}...")


def main() -> int:
    original_query = resolve_query(sys.argv) or DEFAULT_QUERY
    print(f"\n[query] {original_query}")
    print(f"[cfg] model={settings.gen_model} temp={settings.gen_temperature} "
          f"max_tokens={settings.gen_max_tokens} "
          f"jev_model={settings.jev_model} "
          f"query_rewrite={'on' if settings.query_rewrite_enabled else 'off'} "
          f"query_rewrite_model={settings.query_rewrite_model}")

    result = QueryService().run(original_query)
    print(f"[request] id={result.request_id} status={result.status} "
          f"route={result.route} block_score={result.block_score}")
    print(f"[timing] jev_decision_ms={result.jev_decision_ms} "
          f"total_request_ms={result.total_request_ms:.2f}")
    if result.status == "ok" and result.route == "RAG":
        if result.rewritten_query != original_query:
            print(f"\n[rewrite] {original_query}\n     -> {result.rewritten_query}")
        show(f"RRF pool (top {settings.rrf_top_n})", result.rrf)
        show(f"RERANKED (final top {settings.rerank_top_n})", result.contexts)
    print(f"\n== {result.status.upper()} ==")
    print(result.answer)
    return 1 if result.status == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
