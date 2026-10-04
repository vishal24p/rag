"""Exercise UI branching without loading models or a Streamlit server."""
import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest


class FakeUI:
    def __init__(self):
        self.session_state = {"last_query": "old", "last_answer": "old answer"}
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args))
            if name == "text_input":
                return "new query"
            if name == "button":
                return True
            return self
        return call


def ui_namespace(result):
    path = Path(__file__).parents[1] / "scripts" / "streamlit_app.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and not node.decorator_list]
    ui = FakeUI()
    rendered = []
    evaluated = []
    namespace = {
        "st": ui,
        "settings": SimpleNamespace(**{name: "test" for name in (
            "embed_model", "rerank_model", "dense_top_k", "sparse_top_k",
            "rrf_top_n", "rerank_top_n", "gen_model", "gen_temperature",
            "gen_max_tokens", "query_rewrite_enabled", "query_rewrite_model",
            "jev_model", "jev_block_threshold", "jev_api_url",
        )}),
        "load_query_service": lambda: SimpleNamespace(run=lambda query: result),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    namespace["_actual_render_last_result"] = namespace["_render_last_result"]
    namespace["collect_logs_with_status"] = lambda fn, *a, **k: (fn(), ["log"])
    namespace["_render_last_result"] = lambda **kwargs: rendered.append(kwargs)
    namespace["_render_ragas_eval_block"] = lambda *args: evaluated.append(args)
    return namespace, ui, rendered, evaluated


def result(status="ok", route="DIRECT"):
    return SimpleNamespace(
        request_id="request", query="new query", rewritten_query=None,
        status=status, route=route, answer="answer", error="unavailable",
        rrf=[], contexts=[], jev_decision_ms=12.5, total_request_ms=30.5,
        block_score=0.1, blocked=status == "blocked",
    )


@pytest.mark.parametrize("status", ["blocked", "error"])
def test_failed_submission_clears_previous_answer(status):
    namespace, ui, rendered, evaluated = ui_namespace(result(status=status))
    namespace["main"]()
    assert "last_query" not in ui.session_state
    assert "last_answer" not in ui.session_state
    assert rendered == evaluated == []


@pytest.mark.parametrize("route", ["DIRECT", "RAG"])
def test_route_and_timings_persist_and_only_rag_can_evaluate(route):
    namespace, ui, rendered, evaluated = ui_namespace(result(route=route))
    namespace["main"]()
    assert rendered[0]["route"] == route
    assert rendered[0]["jev_decision_ms"] == 12.5
    assert rendered[0]["total_request_ms"] == 30.5
    assert bool(evaluated) is (route == "RAG")


def test_direct_inspection_uses_direct_prompt_without_retrieval_panels():
    namespace, ui, _, _ = ui_namespace(result())
    namespace.update({
        "DIRECT_SYSTEM_PROMPT": "direct system prompt",
        "SYSTEM_PROMPT": "rag system prompt",
        "format_context": lambda chunks: "",
        "_CITE_RE": re.compile(r"\[(\d+)\]"),
    })
    namespace["_actual_render_last_result"](
        query="new query", rewritten_query=None, top5=[], rrf_pool=[],
        answer="answer with [1]", log_chunks=[], route="DIRECT",
        jev_decision_ms=12.5, total_request_ms=30.5,
    )
    labels = [args[0] for _, args in ui.calls if args]
    assert "direct system prompt" in labels
    assert "new query" in labels
    assert "rag system prompt" not in labels
    assert "Retrieval details" not in labels
    assert "Query rewrite inspection" not in labels
    assert not any("Citations" in str(label) for label in labels)
