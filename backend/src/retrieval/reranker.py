"""Rank retrieved chunks using one Jev batch of Noul relevance questions."""
from __future__ import annotations

import logging
from pathlib import Path

import httpx

from src.config import settings
from src.gateway import GatewayError, _probability

log = logging.getLogger(__name__)


class JevReranker:
    def __init__(self, *, transport=None) -> None:
        self.transport = transport

    def rerank(self, query: str, candidates: list[dict], top_n: int | None = None) -> list[dict]:
        if not candidates:
            return []
        if len(candidates) > 32:
            raise GatewayError("gateway-invalid-response")
        if not settings.jev_api_key.strip():
            raise GatewayError("gateway-missing-key")

        try:
            prompt_path = Path(settings.jev_rerank_prompt_path)
            if not prompt_path.is_absolute():
                prompt_path = Path(__file__).resolve().parents[2] / prompt_path
            prompt = prompt_path.read_text(encoding="utf-8").strip()
            if not prompt:
                raise ValueError
        except (OSError, ValueError):
            raise GatewayError("gateway-invalid-response") from None

        questions = {
            f"chunk_{i}": {
                "type": "noul",
                "instructions": f"{prompt}\n\nCandidate chunk {i}:\n{candidate['content']}",
                "criteria": {
                    "true": "The chunk contains at least one phrase useful for answering the question.",
                    "false": "No phrase in the chunk helps answer the question.",
                },
            }
            for i, candidate in enumerate(candidates)
        }
        try:
            with httpx.Client(
                timeout=settings.jev_timeout_seconds, transport=self.transport, follow_redirects=False,
            ) as client:
                response = client.post(
                    settings.jev_api_url,
                    headers={"Authorization": f"Bearer {settings.jev_api_key}"},
                    json={"model": settings.jev_model, "state": query, "questions": questions},
                )
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException:
            raise GatewayError("gateway-timeout") from None
        except httpx.HTTPError:
            raise GatewayError("gateway-http-error") from None
        except (ValueError, TypeError):
            raise GatewayError("gateway-invalid-response") from None

        try:
            if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
                raise ValueError
            answers = data["answers"]
            if set(answers) != set(questions):
                raise ValueError
            scored = []
            for i, candidate in enumerate(candidates):
                answer = answers[f"chunk_{i}"]
                if (
                    not isinstance(answer, dict)
                    or answer.get("type") != "noul"
                    or not _probability(answer.get("noul"))
                ):
                    raise ValueError
                scored.append({**candidate, "score": float(answer["noul"])})
        except (KeyError, ValueError, TypeError, OverflowError):
            raise GatewayError("gateway-invalid-response") from None

        scored.sort(key=lambda candidate: candidate["score"], reverse=True)
        keep = settings.rerank_top_n if top_n is None else top_n
        out = scored[:keep]
        log.info("Jev rerank in=%d out=%d", len(candidates), len(out))
        return out
