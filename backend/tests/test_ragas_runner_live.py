from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace


if importlib.util.find_spec("ragas") is None:
    ragas = __import__("types").ModuleType("ragas")
    dataset_schema = __import__("types").ModuleType("ragas.dataset_schema")
    evaluation = __import__("types").ModuleType("ragas.evaluation")
    llms = __import__("types").ModuleType("ragas.llms")
    metrics = __import__("types").ModuleType("ragas.metrics")

    class EvaluationDataset:
        def __init__(self, samples):
            self.samples = samples

    class SingleTurnSample:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    dataset_schema.EvaluationDataset = EvaluationDataset
    dataset_schema.SingleTurnSample = SingleTurnSample
    evaluation.evaluate = None
    llms.LangchainLLMWrapper = object
    class LLMContextPrecisionWithoutReference:
        name = "llm_context_precision_without_reference"

    metrics.LLMContextPrecisionWithoutReference = LLMContextPrecisionWithoutReference
    metrics.answer_relevancy = object()
    metrics.context_precision = object()
    metrics.faithfulness = object()
    ragas.dataset_schema = dataset_schema
    ragas.evaluation = evaluation
    ragas.llms = llms
    ragas.metrics = metrics
    sys.modules.update({
        "ragas": ragas,
        "ragas.dataset_schema": dataset_schema,
        "ragas.evaluation": evaluation,
        "ragas.llms": llms,
        "ragas.metrics": metrics,
    })

    judge = __import__("types").ModuleType("src.evaluation.judge")
    judge.build_judge_embeddings = lambda: None
    judge.LangchainLLMWrapper = object
    sys.modules["src.evaluation.judge"] = judge
    retriever = __import__("types").ModuleType("src.retrieval.reranked_retriever")
    retriever.RerankedRetriever = object
    sys.modules["src.retrieval.reranked_retriever"] = retriever
else:
    # Import the project's compatibility setup before ragas_runner.  RAGAS
    # 0.2.7 imports the removed VertexAI module during collection.
    from src.evaluation import judge as _judge_compat  # noqa: F401

from src.evaluation import ragas_runner


class _FakeMetric:
    name = "llm_context_precision_without_reference"


class _FakeResult:
    scores = [{
        "faithfulness": 0.8,
        "answer_relevancy": 0.9,
        "llm_context_precision_without_reference": 0.7,
    }]

    def __getitem__(self, name):
        return [self.scores[0][name]]


def test_live_ragas_uses_without_reference_metric_and_sample(monkeypatch):
    captured = {}

    def fake_metric():
        metric = _FakeMetric()
        captured["metric"] = metric
        return metric

    def fake_evaluate(*, dataset, metrics, llm, embeddings, **kwargs):
        captured["sample"] = dataset.samples[0]
        captured["metrics"] = metrics
        captured["llm"] = llm
        captured["embeddings"] = embeddings
        captured["kwargs"] = kwargs
        return _FakeResult()

    monkeypatch.setattr(ragas_runner, "LLMContextPrecisionWithoutReference", fake_metric)
    monkeypatch.setattr(ragas_runner, "evaluate", fake_evaluate)

    result = ragas_runner.score_with_ragas(
        "What is RAG?",
        ["retrieved context"],
        "A generated answer.",
        SimpleNamespace(),
        SimpleNamespace(),
    )

    sample = captured["sample"]
    assert sample.user_input == "What is RAG?"
    assert sample.response == "A generated answer."
    assert sample.retrieved_contexts == ["retrieved context"]
    assert getattr(sample, "reference", None) is None
    assert getattr(sample, "reference_contexts", None) is None
    assert captured["metrics"][2] is captured["metric"]
    assert captured["kwargs"]["raise_exceptions"] is False
    metric_name = captured["metric"].name
    assert metric_name == "llm_context_precision_without_reference"
    assert result["mean"][metric_name] == 0.7
    assert result["scores"][metric_name] == 0.7


def test_live_ragas_captures_actual_answer_relevancy_outputs_and_details(monkeypatch):
    class Result(_FakeResult):
        pass

    def fake_evaluate(*, callbacks, **kwargs):
        callbacks[0].on_chain_end({
            "output": [
                SimpleNamespace(question="Generated one?", noncommittal=0),
                SimpleNamespace(question="Generated two?", noncommittal=1),
                SimpleNamespace(question="Generated three?", noncommittal=0),
            ]
        })
        return Result()

    monkeypatch.setattr(ragas_runner, "evaluate", fake_evaluate)
    monkeypatch.setattr(
        ragas_runner,
        "settings",
        SimpleNamespace(
            gen_model="generator",
            judge_model="judge",
            embed_model="embeddings",
            jev_model="reranker",
        ),
    )

    result = ragas_runner.score_with_ragas(
        "Original question",
        ["context"],
        "answer",
        SimpleNamespace(),
        SimpleNamespace(),
    )

    assert result["details"]["question"] == "Original question"
    assert result["details"]["original_question"] == "Original question"
    assert result["details"]["generated_questions"] == [
        "Generated one?", "Generated two?", "Generated three?"
    ]
    assert result["details"]["noncommittal"] == [0, 1, 0]
    assert result["details"]["configured_models"] == {
        "generator": "generator",
        "judge_llm": "judge",
        "evaluation_embeddings": "embeddings",
        "reranker": "reranker",
    }


def test_context_precision_trace_reproduces_ranked_average_precision():
    from src.evaluation.ragas_runner import _context_precision_trace

    trace = _context_precision_trace([
        {"rank": 1, "context": "first", "verdict": 1, "reason": "useful"},
        {"rank": 2, "context": "second", "verdict": 0, "reason": "off topic"},
        {"rank": 3, "context": "third", "verdict": 1, "reason": "useful"},
    ])

    assert trace["score_from_trace"] == 0.8333333333333333
    assert trace["verdicts"][1]["precision_at_rank"] == 0.5
    assert trace["verdicts"][2]["precision_at_rank"] == 2 / 3


def test_answer_relevancy_trace_shows_cosines_and_noncommittal_penalty():
    from src.evaluation.ragas_runner import _answer_relevancy_trace

    class Embeddings:
        def embed_query(self, _query):
            return [1.0, 0.0]

        def embed_documents(self, _questions):
            return [[0.8, 0.6], [0.6, 0.8]]

    trace = _answer_relevancy_trace(
        "original",
        ["generated one", "generated two"],
        [0, 0],
        Embeddings(),
    )

    assert [item["cosine_similarity"] for item in trace["generated_questions"]] == [0.8, 0.6]
    assert trace["cosine_average"] == 0.7
    assert trace["noncommittal_penalty_applied"] is False
    assert trace["final_score"] == 0.7
