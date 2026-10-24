"""Centralized environment-driven config."""
import os
import math
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: int = _int_env("DB_PORT", 5432)
    db_name: str = os.getenv("DB_NAME", "rag")
    db_user: str = os.getenv("DB_USER", "rag")
    db_password: str = os.getenv("DB_PASSWORD", "")
    chunks_table: str = os.getenv("CHUNKS_TABLE", "chunks")

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embed_model: str = os.getenv("EMBED_MODEL", "nomic-embed-text-v2-moe")
    gen_model: str = os.getenv("GEN_MODEL", "minimax-m3:cloud")

    jev_api_key: str = os.getenv("JEV_API_KEY", "")
    jev_api_url: str = os.getenv("JEV_API_URL", "https://opencode.ai/zen/v1/systemone")
    jev_model: str = os.getenv("JEV_MODEL", "jev-1.13")
    jev_timeout_seconds: float = float(os.getenv("JEV_TIMEOUT_SECONDS", "10"))
    jev_block_threshold: float = float(os.getenv("JEV_BLOCK_THRESHOLD", "0.5"))
    jev_kb_description: str = os.getenv("JEV_KB_DESCRIPTION", "Kubernetes documentation")
    jev_rerank_prompt_path: str = os.getenv(
        "JEV_RERANK_PROMPT_PATH", "prompts/jev_reranker.md",
    )

    dense_top_k: int = _int_env("DENSE_TOP_K", 10)
    sparse_top_k: int = _int_env("SPARSE_TOP_K", 10)
    rrf_top_n: int = _int_env("RRF_TOP_N", 20)
    rerank_top_n: int = _int_env("RERANK_TOP_N", 5)

    # Query rewriting: runs after the gateway selects RAG, before retrieval.
    # Rewrites the user's single query into a clearer search query.
    # When disabled, the original query is passed to retrieval unchanged.
    query_rewrite_enabled: bool = os.getenv("QUERY_REWRITE_ENABLED", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }
    query_rewrite_model: str = os.getenv("QUERY_REWRITE_MODEL", "minimax-m3:cloud")

    gen_temperature: float = float(os.getenv("GEN_TEMPERATURE", "0.2"))
    gen_max_tokens: int = _int_env("GEN_MAX_TOKENS", 400)

    judge_model: str = os.getenv("JUDGE_MODEL", "glm-4.7:cloud")
    judge_temperature: float = float(os.getenv("JUDGE_TEMPERATURE", "0.0"))

    def __post_init__(self) -> None:
        if not math.isfinite(self.jev_timeout_seconds) or self.jev_timeout_seconds <= 0:
            raise ValueError("JEV_TIMEOUT_SECONDS must be finite and positive")
        if not math.isfinite(self.jev_block_threshold) or not 0 <= self.jev_block_threshold <= 1:
            raise ValueError("JEV_BLOCK_THRESHOLD must be finite and between 0 and 1")

    @property
    def db_url(self) -> str:
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


settings = Settings()
