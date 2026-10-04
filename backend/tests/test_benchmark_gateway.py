from scripts.benchmark_gateway import summarize


def test_summary_excludes_warmup_and_separates_failures_and_routes():
    records = [
        dict(phase="warmup", route="RAG", status="ok", jev_decision_ms=999, total_request_ms=9999),
        dict(phase="measured", route="RAG", status="ok", jev_decision_ms=10, total_request_ms=100),
        dict(phase="measured", route="RAG", status="ok", jev_decision_ms=20, total_request_ms=200),
        dict(phase="measured", route="DIRECT", status="ok", jev_decision_ms=5, total_request_ms=40),
        dict(phase="measured", route=None, status="error", jev_decision_ms=None, total_request_ms=1),
    ]
    summaries = {(row["route"], row["status"]): row for row in summarize(records)}
    assert summaries[("RAG", "ok")]["count"] == 2
    assert summaries[("RAG", "ok")]["total_request_ms"] == {"median": 150, "p95": 200}
    assert summaries[("DIRECT", "ok")]["count"] == 1
    assert "jev_decision_ms" not in summaries[(None, "error")]


def test_summary_separates_startup_from_warm_requests():
    records = [
        dict(phase="startup", route="RAG", status="ok", jev_decision_ms=30, total_request_ms=1000),
        dict(phase="measured", route="RAG", status="ok", jev_decision_ms=10, total_request_ms=100),
    ]
    summaries = {row["phase"]: row for row in summarize(records)}
    assert summaries["startup"]["total_request_ms"]["median"] == 1000
    assert summaries["measured"]["total_request_ms"]["median"] == 100
