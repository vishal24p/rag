from types import SimpleNamespace

from src.generation.direct import DIRECT_SYSTEM_PROMPT, DirectGenerator


def test_direct_reuses_client_and_own_prompt(monkeypatch):
    calls = []
    client = SimpleNamespace(chat=lambda messages: calls.append(messages) or "answer")
    monkeypatch.setattr("src.generation.direct.build_llm", lambda: client)
    generator = DirectGenerator()
    assert generator.llm is client
    assert generator.generate("hello") == "answer"
    assert calls[0][0].content == DIRECT_SYSTEM_PROMPT
    assert calls[0][1].content == "hello"
    assert "passages" not in DIRECT_SYSTEM_PROMPT.lower()


def test_direct_complete_fallback(monkeypatch):
    calls = []
    def fail(messages):
        raise RuntimeError("unsupported chat")
    client = SimpleNamespace(chat=fail, complete=lambda prompt, **kwargs: calls.append((prompt, kwargs)) or "fallback")
    monkeypatch.setattr("src.generation.direct.build_llm", lambda: client)
    assert DirectGenerator().generate("hello") == "fallback"
    assert "hello" in calls[0][0]
