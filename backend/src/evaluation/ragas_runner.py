"""RAGAS evaluator.

Builds a ragas SingleTurnSample dataset and runs `ragas.evaluate` with the
three cheapest metrics (no labeled doc lists required):

  * faithfulness            was the answer supported by the retrieved context?
  * answer_relevancy        did the answer actually address the question?
  * context_precision       of retrieved chunks, how many were useful?

Returns a dict with `per_row` (list of metric scores per question) and
`mean` (the run summary).
"""
from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any, Iterable

from langchain_core.callbacks import BaseCallbackHandler
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.evaluation import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    LLMContextPrecisionWithoutReference,
    answer_relevancy,
    context_precision,
    faithfulness,
)

from src.evaluation.judge import build_judge_embeddings
from src.config import settings
from src.retrieval.reranked_retriever import RerankedRetriever

log = logging.getLogger(__name__)

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision")


class _AnswerRelevancyCapture(BaseCallbackHandler):
    """Capture structured answer-relevancy prompt outputs from one evaluation."""

    def __init__(self) -> None:
        self.generated_questions: list[str] = []
        self.noncommittal: list[Any] = []

    def on_chain_end(self, outputs: Any, **_: Any) -> None:
        try:
            values = outputs.get("output", outputs) if isinstance(outputs, Mapping) else outputs
            if not isinstance(values, (list, tuple)):
                values = [values]
            for value in values:
                if isinstance(value, Mapping):
                    question = value.get("question")
                    flag = value.get("noncommittal")
                else:
                    question = getattr(value, "question", None)
                    flag = getattr(value, "noncommittal", None)
                if isinstance(question, str) and flag is not None:
                    self.generated_questions.append(question)
                    self.noncommittal.append(flag)
        except Exception:
            # Observability must never affect scoring.
            return


class _ContextPrecisionCapture(BaseCallbackHandler):
    """Capture the judge's per-context usefulness decisions."""

    def __init__(self) -> None:
        self._inputs: dict[str, Any] = {}
        self.verdicts: list[dict[str, Any]] = []

    def on_chain_start(self, _: Any, inputs: Any, *, run_id: Any, **__: Any) -> None:
        if isinstance(inputs, Mapping):
            values = inputs.get("data", inputs)
            if isinstance(values, Mapping):
                self._inputs[str(run_id)] = dict(values)
            else:
                self._inputs[str(run_id)] = {
                    "question": getattr(values, "question", None),
                    "context": getattr(values, "context", None),
                    "answer": getattr(values, "answer", None),
                }

    def on_chain_end(self, outputs: Any, *, run_id: Any, **__: Any) -> None:
        inputs = self._inputs.pop(str(run_id), {})
        values = outputs.get("output", outputs) if isinstance(outputs, Mapping) else outputs
        if not isinstance(values, (list, tuple)):
            values = [values]

        for value in values:
            if isinstance(value, Mapping):
                verdict = value.get("verdict")
                reason = value.get("reason")
            else:
                verdict = getattr(value, "verdict", None)
                reason = getattr(value, "reason", None)
            if verdict is None:
                continue

            context = inputs.get("context")
            self.verdicts.append({
                "rank": len(self.verdicts) + 1,
                "context": context if isinstance(context, str) else str(context or ""),
                "verdict": int(verdict),
                "reason": reason or "",
            })


def _context_precision_trace(verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Reproduce RAGAS average-precision math for the captured verdicts."""
    relevant = 0
    contributions = []
    for item in verdicts:
        rank = item["rank"]
        verdict = 1 if item["verdict"] else 0
        relevant += verdict
        precision_at_rank = relevant / rank
        contribution = precision_at_rank * verdict
        item.update({
            "precision_at_rank": precision_at_rank,
            "contribution": contribution,
        })
        contributions.append(contribution)

    denominator = relevant
    score = sum(contributions) / denominator if denominator else None
    return {
        "formula": "sum(precision_at_rank × verdict) / sum(verdict)",
        "relevant_contexts": denominator,
        "contexts_evaluated": len(verdicts),
        "score_from_trace": score,
        "verdicts": verdicts,
    }


def _answer_relevancy_trace(
    query: str,
    generated_questions: list[str],
    noncommittal: list[Any],
    embeddings: Any,
) -> dict[str, Any]:
    """Expose the cosine-similarity calculation used by RAGAS."""
    trace: dict[str, Any] = {
        "formula": "mean(cosine_similarity(original_question, generated_question))",
        "generated_questions": [],
        "noncommittal_penalty_applied": bool(any(noncommittal)),
    }
    if not generated_questions:
        trace["cosine_average"] = None
        trace["final_score"] = None
        return trace

    try:
        query_vector = embeddings.embed_query(query)
        generated_vectors = embeddings.embed_documents(generated_questions)

        def cosine(left: Any, right: Any) -> float:
            dot = sum(float(a) * float(b) for a, b in zip(left, right))
            left_norm = sum(float(a) ** 2 for a in left) ** 0.5
            right_norm = sum(float(b) ** 2 for b in right) ** 0.5
            return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0

        similarities = [cosine(query_vector, vector) for vector in generated_vectors]
        trace["generated_questions"] = [
            {
                "question": generated,
                "noncommittal": noncommittal[index] if index < len(noncommittal) else None,
                "cosine_similarity": similarity,
            }
            for index, (generated, similarity) in enumerate(
                zip(generated_questions, similarities)
            )
        ]
        trace["cosine_average"] = sum(similarities) / len(similarities)
        trace["final_score"] = (
            0.0 if trace["noncommittal_penalty_applied"] else trace["cosine_average"]
        )
    except Exception as exc:
        trace["error"] = f"{type(exc).__name__}: {exc}"
        trace["cosine_average"] = None
        trace["final_score"] = None
    return trace


def run_rag_pipeline(query: str, conn_str: str) -> tuple[list[str], str]:
    """Return (context_passages, generated_answer) for one query."""
    rr = RerankedRetriever(conn_str)
    _, top = rr.retrieve(query)
    contexts = [c["content"] for c in top]
    from src.generation.generator import Generator
    answer = Generator().generate(query, top)
    return contexts, answer


def score_with_ragas(
    query: str,
    contexts: list[str],
    answer: str,
    judge_llm: LangchainLLMWrapper,
    judge_embeddings,
) -> dict:
    """Score one (q, retrieved, answer) triple with the three cheap metrics.

    Does NOT re-run the pipeline; the caller hands in the answers it just got.
    """
    log.info("RAGAS scoring triple (q len=%d, ctx=%d, ans len=%d)",
             len(query), len(contexts), len(answer))
    capture = _AnswerRelevancyCapture()
    context_capture = _ContextPrecisionCapture()
    logs = [f"input counts: questions=1 contexts={len(contexts)} answers=1"]
    ds = EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=query,
            response=answer,
            retrieved_contexts=contexts,
        ),
    ])
    live_context_precision = LLMContextPrecisionWithoutReference()
    metrics = [faithfulness, answer_relevancy, live_context_precision]
    live_metric_names = (
        "faithfulness",
        "answer_relevancy",
        live_context_precision.name,
    )

    try:
        result = evaluate(
            dataset=ds,
            metrics=metrics,
            llm=judge_llm,
            embeddings=judge_embeddings,
            callbacks=[capture, context_capture],
            raise_exceptions=False,
            show_progress=False,
        )
    except Exception as exc:
        logs.append(f"evaluation failed: {type(exc).__name__}")
        result = None

    def unavailable_as_none(value):
        if value is None:
            return None
        try:
            if value != value:  # NaN
                return None
        except (TypeError, ValueError):
            return None
        return value

    mean = {}
    for name in live_metric_names:
        try:
            values = result[name] if result is not None else None
        except (KeyError, TypeError):
            values = None
        if not values:
            mean[name] = None
        else:
            valid_values = [unavailable_as_none(value) for value in values]
            valid_values = [value for value in valid_values if value is not None]
            mean[name] = sum(valid_values) / len(valid_values) if valid_values else None

        logs.append(
            f"metric {name}: {'completed' if mean[name] is not None else 'unavailable'}"
        )

    trace = _context_precision_trace(context_capture.verdicts)
    answer_trace = _answer_relevancy_trace(
        query,
        capture.generated_questions,
        capture.noncommittal,
        judge_embeddings,
    )
    logs.append(
        "context precision trace: "
        f"evaluated={trace['contexts_evaluated']} "
        f"relevant={trace['relevant_contexts']} "
        f"score={trace['score_from_trace']}"
    )
    logs.append(
        "answer relevancy trace: "
        f"generated={len(answer_trace['generated_questions'])} "
        f"cosine_average={answer_trace['cosine_average']} "
        f"noncommittal_penalty={answer_trace['noncommittal_penalty_applied']} "
        f"final={answer_trace['final_score']}"
    )

    scores = result.scores[0] if result is not None and result.scores else {}
    return {
        "mean": mean,
        "scores": {
            name: unavailable_as_none(scores.get(name))
            for name in live_metric_names
        },
        "details": {
            "question": query,
            "original_question": query,
            "generated_questions": capture.generated_questions,
            "noncommittal": capture.noncommittal,
            "answer_relevancy": answer_trace,
            "context_precision": trace,
            "configured_models": {
                "generator": settings.gen_model,
                "judge_llm": settings.judge_model,
                "evaluation_embeddings": settings.embed_model,
                "reranker": settings.rerank_model,
            },
            "logs": logs,
        },
    }


def evaluate_pipeline(
    qa_set: list[dict],
    conn_str: str,
    judge_llm: LangchainLLMWrapper,
) -> dict:
    """Run the pipeline for each Q in qa_set, score with RAGAS, return
    {"per_row": [...], "mean": {...}, "elapsed_seconds": float}.
    """
    samples: list[SingleTurnSample] = []
    per_row: list[dict] = []
    t0 = time.perf_counter()
    for qa in qa_set:
        log.info("=== running QA id=%s ===", qa["id"])
        log.info("Q: %s", qa["question"][:120])
        contexts, answer = run_rag_pipeline(qa["question"], conn_str)
        samples.append(SingleTurnSample(
            user_input=qa["question"],
            response=answer,
            retrieved_contexts=contexts,
            reference=qa.get("ground_truth") or "",
            reference_contexts=qa.get("reference_contexts") or [],
        ))
        per_row.append({
            "id": qa["id"],
            "question": qa["question"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": qa.get("ground_truth"),
        })

    log.info("running ragas.evaluate (%d metrics)...", len(METRIC_NAMES))
    ds = EvaluationDataset(samples=samples)
    metrics = [faithfulness, answer_relevancy, context_precision]
    embeddings = build_judge_embeddings()
    result = evaluate(
        dataset=ds,
        metrics=metrics,
        llm=judge_llm,
        embeddings=embeddings,
        raise_exceptions=False,
        show_progress=False,
    )
    elapsed = time.perf_counter() - t0

    # scores is a list of dicts, one per row; order matches samples / per_row.
    scores = result.scores
    per_row_scored: list[dict] = []
    for row, score in zip(per_row, scores):
        per_row_scored.append({
            **{k: row[k] for k in ("id", "question", "ground_truth", "answer")},
            "contexts": row["contexts"],
            "scores": {name: score.get(name) for name in METRIC_NAMES},
        })

    mean = {
        name: (sum(result[name]) / len(result[name])) if result[name] else 0.0
        for name in METRIC_NAMES
    }
    return {
        "per_row": per_row_scored,
        "mean": mean,
        "elapsed_seconds": elapsed,
    }


def write_jsonl(records: Iterable[dict], path: str) -> None:
    import json
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
