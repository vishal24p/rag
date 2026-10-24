# RAG over Kubernetes Docs

Hybrid (BM25 + pgvector) RAG with a Jev request gateway through OpenCode Zen. React chat calls a thin FastAPI adapter; Ollama generates grounded RAG answers and DIRECT answers. The gateway requires a Zen API key.

## Stack

| Layer | Tool |
|---|---|
| Database + Index | ParadeDB (pgvector + BM25) |
| Embeddings | `nomic-embed-text-v2-moe` via Ollama |
| Semantic Chunker | LlamaIndex `SemanticSplitterNodeParser` |
| Hybrid Fusion | RRF (Reciprocal Rank Fusion, k=60) |
| Reranker | Jev Noul relevance scores through OpenCode Zen |
| Generator | `minimax-m3:cloud` via Ollama (served by Ollama locally; the `:cloud` tag is just the model name) |
| Request Gateway | Jev through OpenCode Zen: `request_block` + `retrieval_route` in one request |
| Eval | RAGAS (faithfulness, answer relevancy, context precision) |
| API | FastAPI |
| UI | React + TypeScript + Vite |

## Architecture

### V1 — Hybrid retrieval

The baseline: dense (pgvector) + sparse (BM25) retrieval fused by RRF, reranked, then answered. No safety gate yet — every query reaches retrieval and the LLM.

![V1 architecture](https://github.com/vishal24p/learning/raw/main/rag/assets/version1-arch.webp)

### Current — Jev gateway

Jev evaluates the original query before downstream processing. A block stops the request; RAG retains the existing rewrite/retrieval/generation path; DIRECT uses the same cached Ollama client with its own prompt.

```text
React → FastAPI → QueryService → Jev (request_block + retrieval_route)
        ├─ blocked → safe refusal
        ├─ RAG → rewrite → Dense + BM25 → RRF → Jev rerank → RAG generator
        └─ DIRECT → same Ollama client + direct prompt
```

Timeouts, invalid responses and provider failures stop the request with a service error. The gateway sends Jev the original query and a short KB scope description. On the RAG route, a separate reranking request sends the rewritten query as shared state and retrieved candidate chunk text in one Noul relevance question per chunk. These chunks leave the local backend for OpenCode Zen. The generator then receives the highest-scoring chunks.

## Version log

### V1 -- Hybrid retrieval, oversized chunks

<img src="https://github.com/vishal24p/learning/raw/main/rag/assets/version1.gif" alt="V1 demo" width="100%">

> [version1.gif](https://github.com/vishal24p/learning/blob/main/rag/assets/version1.gif) for the animated loop.

Chunker: SemanticSplitterNodeParser, buffer 2, breakpoint percentile 95. Tight window plus strict threshold meant splits almost never fired on docs where paragraphs only loosely related. Whole pages stayed glued. Average chunk size: **2,813 chars**.

Original V1 retrieval: dense `DENSE_TOP_K` + sparse `SPARSE_TOP_K` → RRF `RRF_TOP_N` →
cross-encoder rerank → `RERANK_TOP_N` to the LLM. The composite retriever
returns both the RRF output and the final reranked context, so callers do not
run hybrid retrieval twice.

```
Faithfulness      1.000
Answer Relevancy  0.386
Context Precision 0.333
```

Faithfulness at 1.0. Nothing fabricated; everything came from retrieved context. The 0.386 answer relevancy is the real failure. On "what is the use of Kubernetes?" retrieval pulled one Overview chunk covering orchestration, scaling, networking, deployments--all in one block. The model summarized it faithfully, but it never answered the question.

The reranker scored that chunk **-0.44**. Negative. It knew the chunk was off. Still landed in the final 5 because nothing better existed upstream.

**Diagnosis**: 2,813-character chunks bury the two sentences that actually answer the question inside twelve other topics. Dense retrieval can't isolate them. The reranker can't fix what the chunker broke.

### V1.1 -- Re-chunk + positive filter

<img src="https://github.com/vishal24p/learning/raw/main/rag/assets/version2.gif" alt="V2 demo" width="100%">

> [version2.gif](https://github.com/vishal24p/learning/blob/main/rag/assets/version2.gif) for the animated loop. (On disk the V1.1 demo is named `version2.gif`; the V1 demo is `version1.gif`.)

Chunker retuned to buffer 5, breakpoint percentile 80. Same corpus, same embed model, fresh database. 14,734 chunks. Average **1,045 chars**.

`filter_positive_chunks()` added post-rerank: anything below reranker score zero drops out before the LLM sees it. If every candidate is negative, keep the single best so the model still has grounding.

```
V1                              V1.1
Faithfulness      1.000        Faithfulness      0.889
Answer Relevancy  0.386        Answer Relevancy  0.661
Context Precision 0.333        Context Precision 1.000
```

Context precision 0.333 → 1.000. Every chunk in the context window is now relevant. Answer relevancy nearly doubles. Faithfulness drops from 1.000 → 0.889 because the model starts going beyond retrieved context when chunks are tight enough to answer from. Worth the trade-off; answers now answer the question.

### V2 -- Historical safety gate

A previous version used a local safe/unsafe classifier before retrieval. That implementation has been removed and replaced by the two-decision Jev gateway above.

The current version replaces the local cross-encoder with Jev reranking. Hybrid retrieval and RAG generation retain their existing interfaces. React uses the same request service as the CLI and benchmarks, through FastAPI. Streamlit has been retired.

## Run it

### Prerequisites

- **Python 3.11+** in a virtualenv (`python -m venv .venv`).
  - Install Python packages with `pip install -r backend/requirements.txt` from the repository root. See
    [`requirements.md`](backend/requirements.md) for the package inventory and
    non-Python prerequisites.
- **Ollama** running on `http://localhost:11434`, with these models pulled:
  - `ollama pull nomic-embed-text-v2-moe` (embeddings)
  - `ollama pull minimax-m3:cloud` (generator)
  - `ollama pull glm-4.7:cloud` (RAGAS judge; only needed for evaluation)
- **ParadeDB** (Postgres with the `vector` and `pg_search` extensions), listening
  on the port set in `.env` (default `56432`). Stand up ParadeDB yourself
  (Docker image `paradedb/paradedb`) and ensure the extensions are available.
- **Corpus**: the Kubernetes docs mirror under `website-main/content/en/docs`.
  This directory is gitignored and regenerated from upstream; clone the
  `kubernetes/website` repo there before reindexing.
- **`.env`**: copy `backend/.env.example` to `backend/.env` and set `DB_PASSWORD` and `JEV_API_KEY` (and adjust
  `DB_*` / model names as needed). `.env` is gitignored.

### Setup & launch

```powershell
# From C:\Users\visha\rag. Existing virtualenv and populated database can be reused.
uv pip install --python .venv\Scripts\python.exe -r backend\requirements.txt

# Only for a new database; do not reindex an existing populated database unnecessarily.
.\.venv\Scripts\python.exe backend\scripts\apply_migration.py

# Terminal 1: FastAPI
.\.venv\Scripts\python.exe -m uvicorn src.api:app --app-dir backend --host 127.0.0.1 --port 8000

# Terminal 2: React
cd frontend
npm install
npm run dev
```

The install command uses `uv`, which works with a virtualenv that has no `pip` module. A conventional pip-enabled virtualenv can instead use `.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt`.

Open http://127.0.0.1:5173. Node 22.18+ is required for frontend tests. Vite proxies `/api` to port 8000; the browser never receives the Zen key. API liveness is at http://127.0.0.1:8000/api/health and interactive API docs are at http://127.0.0.1:8000/docs. This setup serves the UI locally; production hosting is outside this version.

Chat sends the current query only. Displayed conversations reset on refresh; they do not supply model memory. Answers arrive as complete responses. Stop cancels waiting in the browser; inference already running on the backend may continue. Sources show actual retrieved chunks for successful RAG answers. DIRECT answers and refused/failed requests have no sources. Document uploads are not included.

`POST /api/query` accepts `{"query":"According to the Kubernetes docs, how do rolling updates work?"}`. It returns the answer, route, request ID, block decision, ordered contexts and both latency metrics. Successful answers use HTTP 200, refusals 403, gateway failures 503, downstream failures 500 and invalid inputs 422. Queries must be nonblank strings of at most 10,000 characters.

By default `.env.example` points at the committed schema (`DB_NAME=rag`,
`CHUNKS_TABLE=chunks`). If you keep a local-only v2 reindex helper, it can target
`rag_v2` / `chunks_v2`; set `.env` to those names after that local table exists.

CLI smoke tests (no UI) live under `scripts/`: `run_pipeline.py`,
`run_pipeline_with_rerank.py`, `run_rag.py`. Evaluation: `run_ragas.py`.

## Jev settings

Set these in `backend/.env`. Existing environment variables take precedence.

| Name | Default / meaning |
|---|---|
| `JEV_API_KEY` | Your OpenCode Zen key; empty stops requests safely |
| `JEV_API_URL` | `https://opencode.ai/zen/v1/systemone` |
| `JEV_MODEL` | `jev-1.13` |
| `JEV_TIMEOUT_SECONDS` | `10`, positive HTTP timeout; no automatic retries |
| `JEV_BLOCK_THRESHOLD` | `0.5`; block when the Noul probability is strictly greater |
| `JEV_KB_DESCRIPTION` | Short trusted description of the indexed corpus; default Kubernetes docs |
| `JEV_RERANK_PROMPT_PATH` | `prompts/jev_reranker.md`; editable Markdown instructions, resolved relative to `backend/` unless absolute |

`request_block` covers prompt injection, bypass/override attempts, hidden information or secrets, and clearly harmful/disallowed assistance. `retrieval_route` selects RAG or DIRECT using the KB scope. The gateway adapter lives in `backend/src/gateway.py`; the reranker lives in `backend/src/retrieval/reranker.py`.

Reranking uses one System One request for up to 32 candidates from `RRF_TOP_N`. Each question includes the Markdown prompt and only its target chunk. Returned Noul probabilities in `[0, 1]` replace candidate scores; chunks are sorted descending and limited by `RERANK_TOP_N`. No local reranker model or softmax setting is required. Empty candidate lists skip the request; more than 32 candidates, an unreadable or empty prompt, and malformed responses fail safely.

## Latency benchmark

From `backend/`:

```bash
python scripts/run_rag.py "According to the Kubernetes docs, how do rolling updates work?"
python scripts/benchmark_gateway.py --repeat 10 --warmup 1 --output jev-benchmark.jsonl
python scripts/benchmark_gateway.py --queries queries.txt --repeat 10 --warmup 0 --output cold-start.jsonl
python -m pytest tests -q
```

Reranker tests use mocked HTTP responses; they do not require a live Zen key or local reranker model.

Query files contain one query per line. The runner reuses loaded components and groups median/p95 by phase, actual route and outcome. Warmup rounds are excluded. With `--warmup 0`, the first round is labeled `startup` and summarized separately from subsequent `measured` rounds. Process startup does not guarantee a cold Ollama model; it may already be loaded by another client.

Each request logs its ID, block score/decision, route, failures, `jev_decision_ms` and `total_request_ms`. Total timing spans request service entry through answer/refusal/failure completion, including Jev and model initialization; it excludes HTTP queueing, serialization, browser/network time and UI rendering. Benchmark JSONL omits query text, answers and credentials.

Compare equivalent RAG queries against a saved existing-pipeline baseline; report DIRECT separately because it intentionally skips retrieval. A live baseline could not be captured during implementation: local Ollama and ParadeDB were unavailable. No live performance improvement is claimed. Run benchmarks with the Zen key and backend services configured.
