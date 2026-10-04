r"""Streamlit UI for the RAG pipeline (no chat memory).

Layout:
- Title and short subtitle.
- Free-form query input. On Enter / Send:
    * runs Jev -> RAG or DIRECT -> Generator fresh per submission
    * renders the user's question
    * shows a live "Pipeline status" panel + an in-flight log feed on the
      right as the pipeline runs
    * renders the answer
    * expandable panels for the RRF pool and final reranked context
    * citations panel with the [n] labels used by the answer
    * logs panel (kept open by default) with INFO lines from the pipeline

Run from PowerShell with:
    $env:PYTHONPATH = r"C:\Users\visha\rag"
    python -m streamlit run scripts\run_ui.py
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.config import settings  # noqa: E402
from src.generation.generator import Generator  # noqa: E402
from src.generation.prompts import QA_TEMPLATE, SYSTEM_PROMPT, format_context  # noqa: E402
from src.generation.direct import DirectGenerator, DIRECT_SYSTEM_PROMPT  # noqa: E402
from src.query_service import QueryService  # noqa: E402
from src.logging_setup import setup_logging  # noqa: E402
from src.query_rewrite import REWRITE_SYSTEM_PROMPT  # noqa: E402
from src.retrieval.reranked_retriever import RerankedRetriever  # noqa: E402
from src.ui.keys import derive_ragas_button_key as _derive_ragas_button_key

setup_logging()
_CITE_RE = re.compile(r"\[(\d+)\]")


class _MemoryHandler(logging.Handler):
    """One-shot handler that collects formatted records into a list.

    Captures DEBUG and above; we render the full timeline in the UI.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[str] = []
        self.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append(self.format(record))
        except Exception:
            pass


class _StreamingHandler(_MemoryHandler):
    """Forward each record to both records[] AND a Streamlit container."""

    def __init__(self, sink) -> None:
        super().__init__()
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        try:
            self._sink.text += "\n\n" + self.format(record)
        except Exception:
            pass


def collect_logs_with_status(callable_, status_box, log_box, prefix=""):
    """Run `callable_` while streaming log lines into `log_box`.

    Returns (result, captured_records_list).
    """
    handler = _StreamingHandler(log_box)
    root = logging.getLogger()
    # If the user set LOG_LEVEL=ERROR or higher in env, that's deliberate.
    # We do NOT bypass it; we only use the root logger as an event source.
    root.addHandler(handler)
    handler.setLevel(logging.DEBUG)
    try:
        result = callable_()
        if status_box is not None:
            status_box.update(label=f"{prefix}done", state="complete")
        return result, handler.records
    except Exception as exc:
        if status_box is not None:
            status_box.update(label=f"{prefix}failed: {exc}", state="error")
        raise
    finally:
        root.removeHandler(handler)


@st.cache_resource
def load_retriever() -> RerankedRetriever:
    return RerankedRetriever(settings.db_url)


@st.cache_resource
def load_generator() -> Generator:
    return Generator()


@st.cache_resource
def load_direct_generator() -> DirectGenerator:
    return DirectGenerator()


@st.cache_resource
def load_query_service() -> QueryService:
    return QueryService(
        retriever_factory=load_retriever,
        generator_factory=load_generator,
        direct_generator_factory=load_direct_generator,
    )


def _chunk_preview(content: str, limit: int = 280) -> str:
    preview = content.replace("\n", " ").strip()
    if len(preview) <= limit:
        return preview
    return f"{preview[:limit]}..."


def render_chunks(name: str, hits: list[dict]) -> None:
    if not hits:
        st.write(f"_{name}_: no results")
        return
    for i, h in enumerate(hits, 1):
        content = h.get("content", "")
        chunk_id = h.get("chunk_id", "<missing>")
        score = h.get("score")
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
        with st.expander(
            f"[{i}] id={chunk_id} score={score_text} - {_chunk_preview(content, 110)}",
            expanded=False,
        ):
            st.markdown(
                f"**rank:** `{i}`  \n"
                f"**chunk_id:** `{chunk_id}`  \n"
                f"**score:** `{score_text}`"
            )
            st.code(content, language="text")


def _render_generator_inspection(
    query: str,
    fed_chunks: list[dict],
    answer: str,
    route: str = "RAG",
) -> None:
    context_block = format_context(fed_chunks)
    user_prompt = (query if route == "DIRECT"
                   else QA_TEMPLATE.format(context=context_block, query=query))
    system_prompt = DIRECT_SYSTEM_PROMPT if route == "DIRECT" else SYSTEM_PROMPT

    with st.expander("Generator inspection", expanded=False):
        st.markdown(
            f"**generator model:** `{settings.gen_model}`  \n"
            f"**fed chunks:** `{len(fed_chunks)}`"
        )
        with st.expander("SYSTEM_PROMPT", expanded=False):
            st.code(system_prompt, language="text")
        with st.expander("Rendered user prompt", expanded=False):
            st.code(user_prompt, language="text")
        with st.expander("Generated answer/output", expanded=False):
            st.code(answer, language="text")


def _render_query_rewrite_inspection(
    query: str,
    rewritten_query: str | None,
) -> None:
    rewritten = rewritten_query or query
    with st.expander("Query rewrite inspection", expanded=False):
        st.markdown(
            f"**enabled:** `{settings.query_rewrite_enabled}`  \n"
            f"**rewrite model:** `{settings.query_rewrite_model}`"
        )
        with st.expander("REWRITE_SYSTEM_PROMPT", expanded=False):
            st.code(REWRITE_SYSTEM_PROMPT, language="text")
        st.markdown("**Original user query**")
        st.code(query, language="text")
        st.markdown("**Rewritten retrieval query**")
        st.code(rewritten, language="text")


def _ragas_input_bundle(
    query: str,
    contexts: list[str],
    answer: str,
) -> dict:
    return {
        "user_input": query,
        "response": answer,
        "retrieved_contexts": contexts,
        "configured_models": {
            "generator": settings.gen_model,
            "judge_llm": settings.judge_model,
            "evaluation_embeddings": settings.embed_model,
            "reranker": settings.rerank_model,
        },
        "input_counts": {
            "questions": 1,
            "contexts": len(contexts),
            "answers": 1,
        },
        "metrics": [
            "faithfulness",
            "answer_relevancy",
            "llm_context_precision_without_reference",
        ],
    }


def extract_used_citations(answer: str) -> list[int]:
    seen = []
    for m in _CITE_RE.finditer(answer):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def main() -> None:
    st.set_page_config(page_title="RAG (Hybrid + Rerank)", layout="wide")
    st.title("RAG with BM25 (ParadeDB) + Dense (pgvector) + Cross-Encoder Rerank")
    st.caption(
        "Jev checks each question and routes it through RAG or directly to the generator; "
        "no chat memory is kept across submissions."
    )

    # Pipeline config snapshot (read-only chip block)
    with st.expander("Pipeline settings", expanded=False):
        cfg_pairs = [
            ("Embedding model", settings.embed_model),
            ("Reranker model", settings.rerank_model),
            ("Dense top-K", str(settings.dense_top_k)),
            ("Sparse top-K", str(settings.sparse_top_k)),
            ("RRF top-N", str(settings.rrf_top_n)),
            ("Rerank top-N", str(settings.rerank_top_n)),
            ("model", settings.gen_model),
            ("temp", str(settings.gen_temperature)),
            ("max tokens", str(settings.gen_max_tokens)),
            ("Query rewrite", "on" if settings.query_rewrite_enabled else "off"),
            ("Query rewrite model", settings.query_rewrite_model),
            ("Jev model", settings.jev_model),
            ("Jev block threshold", str(settings.jev_block_threshold)),
        ]
        st.table({k: v for k, v in cfg_pairs})

    query = st.text_input(
        "Question",
        placeholder="e.g. How do H1, H2, H3 heading levels work in markdown?",
        label_visibility="collapsed",
    )
    send = st.button("Ask", type="primary")

    if send and query.strip():
        # Clear the prior answer before a new run, including refusals and failures.
        for key in list(st.session_state):
            if key.startswith("last_"):
                del st.session_state[key]
        status = st.status("Running Jev gateway and request pipeline...", expanded=True)
        live_log = st.empty()
        result, log_chunks = collect_logs_with_status(
            lambda: load_query_service().run(query), None, live_log,
        )
        if result.status != "ok":
            status.update(label="Request blocked" if result.blocked else "Request failed",
                          state="error")
            st.markdown("---")
            st.markdown(f"**You:** {query}")
            st.error(result.answer or "The request could not be completed. Please try again.")
            _render_request_metrics(result.route, result.jev_decision_ms, result.total_request_ms)
            return
        status.update(label="Answer ready", state="complete")
        st.session_state.update({
            "last_query": result.query,
            "last_rewritten_query": result.rewritten_query,
            "last_top5": result.contexts,
            "last_answer": result.answer,
            "last_rrf_pool": result.rrf,
            "last_log_chunks": log_chunks,
            "last_route": result.route,
            "last_jev_decision_ms": result.jev_decision_ms,
            "last_total_request_ms": result.total_request_ms,
        })

    last = _read_last_result()
    if last:
        _render_last_result(**last)
        if last["route"] == "RAG":
            _render_ragas_eval_block(last["query"], last["top5"])


def _render_request_metrics(route: str | None, jev_decision_ms: float,
                            total_request_ms: float) -> None:
    st.caption(f"Route: {route or 'unavailable'} · Jev decision: {jev_decision_ms:.1f} ms · "
               f"Total request: {total_request_ms:.1f} ms")


def _read_last_result() -> dict | None:
    """Pull the most recent run from session_state; None if absent."""
    if "last_query" not in st.session_state:
        return None
    return {
        "query": st.session_state["last_query"],
        "rewritten_query": st.session_state.get("last_rewritten_query"),
        "top5": st.session_state["last_top5"],
        "rrf_pool": st.session_state["last_rrf_pool"],
        "answer": st.session_state["last_answer"],
        "log_chunks": st.session_state["last_log_chunks"],
        "route": st.session_state["last_route"],
        "jev_decision_ms": st.session_state["last_jev_decision_ms"],
        "total_request_ms": st.session_state["last_total_request_ms"],
    }


def _render_last_result(query: str, rewritten_query: str | None,
                        top5: list[dict],
                        rrf_pool: list[dict], answer: str,
                        log_chunks: list[str], route: str,
                        jev_decision_ms: float, total_request_ms: float) -> None:
    """Render a cached query result. Reusable on every rerun."""
    st.markdown("---")
    st.markdown(f"**You:** {query}")
    _render_request_metrics(route, jev_decision_ms, total_request_ms)
    if route == "RAG" and rewritten_query and rewritten_query != query:
        st.caption(f"rewritten for retrieval: {rewritten_query}")

    st.markdown("**Answer:**")
    st.write(answer)
    if route == "RAG":
        _render_query_rewrite_inspection(query, rewritten_query)
    _render_generator_inspection(query, top5, answer, route)

    used = extract_used_citations(answer) if route == "RAG" else []
    if used:
        st.markdown("---")
        st.markdown(f"**Citations ({len(used)}):**")
        for n in used:
            hit = top5[n - 1] if 1 <= n <= len(top5) else None
            if hit:
                preview = hit["content"].replace("\n", " ")[:320]
                st.markdown(
                    f"- **[{n}]** id={hit['chunk_id']} (rerank score "
                    f"{hit['score']:.4f}) - {preview}..."
                )

    if route == "RAG":
        with st.expander("Retrieval details", expanded=False):
            st.markdown(f"##### RRF pool (top {settings.rrf_top_n} fused candidates)")
            render_chunks("RRF pool", rrf_pool)
            st.markdown(f"##### Reranked top {settings.rerank_top_n}")
            render_chunks("Reranked", top5)

    with st.expander("Logs (last run)", expanded=True):
        if log_chunks:
            st.code("\n".join(log_chunks), language="text")
        else:
            st.write("No log lines emitted for this query.")


def _render_ragas_eval_block(query: str, top5: list[dict]) -> None:
    """Below the answer: a button that runs RAGAS on the same Q + retrieval.

    Faithfulness, answer_relevancy, context precision without reference.
    No labelled doc set.
    Passes the (q, contexts, answer) we already produced, so the metric
    measures the answer the user just saw -- no extra pipeline run.

    RAGAS scores the same reranked context list that the LLM saw.
    """
    btn_key = _derive_ragas_button_key(query, top5)
    with st.expander("RAGAS eval (last answer)", expanded=False):
        answer = st.session_state.get("last_answer", "")
        contexts = [
            ch["content"] if isinstance(ch, dict) and "content" in ch else str(ch)
            for ch in top5
        ]
        input_bundle = _ragas_input_bundle(query, contexts, answer)
        with st.expander("Evaluation Inputs/Models", expanded=False):
            st.json(input_bundle)
        run = st.button("Run RAGAS on this answer", key=btn_key)
        if run:
            with st.spinner("Scoring with RAGAS..."):
                try:
                    from src.evaluation.judge import build_judge_llm, build_judge_embeddings
                    from src.evaluation.ragas_runner import score_with_ragas

                    judge = build_judge_llm()
                    emb = build_judge_embeddings()
                    ragas_log_box = st.empty()
                    score, ragas_logs = collect_logs_with_status(
                        lambda: score_with_ragas(query, contexts, answer, judge, emb),
                        None,
                        ragas_log_box,
                        prefix="RAGAS ",
                    )

                    mean = score["mean"]
                    cols = st.columns(3)
                    def display_metric(name: str) -> str:
                        value = mean.get(name)
                        return f"{value:.3f}" if value is not None else "N/A"

                    cols[0].metric("faithfulness", display_metric("faithfulness"))
                    cols[1].metric("answer_relevancy", display_metric("answer_relevancy"))
                    cols[2].metric(
                        "Context Precision Without Reference",
                        display_metric("llm_context_precision_without_reference"),
                    )
                    with st.expander("Raw RAGAS scores", expanded=False):
                        st.json(score.get("scores", {}))
                    details = score.get("details", {})
                    generated_questions = details.get("generated_questions", [])
                    with st.expander("Answer Relevancy Details", expanded=False):
                        answer_trace = details.get("answer_relevancy", {})
                        st.write(
                            "RAGAS generates questions from the answer, compares each "
                            "generated question with the original question using cosine "
                            "similarity, averages those similarities, and sets the final "
                            "score to zero only when a noncommittal flag is 1."
                        )
                        st.json(answer_trace)
                        if generated_questions:
                            st.write("Actual generated questions")
                            for question_number, generated in enumerate(generated_questions, 1):
                                st.write(f"{question_number}. {generated}")
                            if details.get("noncommittal"):
                                st.write({"noncommittal": details["noncommittal"]})
                        else:
                            st.write("No generated questions captured.")
                    context_trace = details.get("context_precision", {})
                    with st.expander("Context Precision Details", expanded=False):
                        st.write(
                            "Each retrieved chunk is judged as useful (1) or not useful (0) "
                            "for answering the question with the generated answer. "
                            "RAGAS then applies ranked average precision."
                        )
                        st.json(context_trace)
                    with st.expander("RAGAS logs", expanded=False):
                        concise_logs = details.get("logs", []) or ragas_logs
                        st.code("\n".join(concise_logs) if concise_logs else "No RAGAS logs.", language="text")
                    st.caption("elapsed (not measured for single-row runs)")
                except Exception as exc:
                    st.error(f"RAGAS run failed: {exc}")


if __name__ == "__main__":
    main()
