# Local Mind

A local-only retrieval-augmented generation system over a personal PDF corpus. We treat
the document set, the index, the chat log, and the long-term memory store as
state on the developer's machine; no request leaves the host at any point in the
default path. The system is a working research artifact, not a product — it exists
to make a small set of design choices about RAG visible and measurable.

---

## What this is

A self-contained pipeline that:

1. Parses PDFs from `pdfs/` into text and Markdown tables.
2. Chunks text at 1500 chars with 300 char overlap. Tables are not split.
3. Embeds chunks with `nomic-embed-text` (768 dim) via a local Ollama instance.
4. Persists vectors to a Chroma collection on disk (`chroma_db_local/`).
5. On a query, rewrites the question with a small model (`phi4-mini`), retrieves
   the top-4 chunks via MMR (fetch_k=15), pulls relevant items from a
   Qdrant-backed long-term memory store (Mem0), and streams an answer from a
   larger model (`minimax-m3:cloud` by default).
6. Writes the completed turn back to long-term memory off the response path.

Two LLM roles, one embedding role:

```
        ┌──────────────┐
        │  query       │  phi4-mini:latest
        │  rewriter    │  — temp 0
        └──────┬───────┘
               │ standalone question
               ▼
        ┌──────────────┐         ┌──────────────────┐
        │  retriever   │ ──────► │  Chroma (local)  │
        │  MMR k=4     │         │  768d cosine     │
        └──────┬───────┘         └──────────────────┘
               │ top-k chunks
               ▼
        ┌──────────────┐         ┌──────────────────┐
        │  generator   │ ◄────── │  Mem0 + Qdrant   │
        │  MAIN_MODEL  │         │  768d cosine     │
        └──────┬───────┘         └──────────────────┘
               │ streamed tokens
               ▼
        background: m.add(turn)  ──►  long-term memory
```

We deliberately use a small model for rewriting and memory extraction and a
larger one for synthesis. The asymmetry is the point: rewriting does not need
a 70B model, but synthesizing a grounded answer often does.

---

## What this is not

- Not a hosted service. There is no API key, no account, no server endpoint
  reachable from the public internet.
- Not a multi-user system. The user is hardcoded as `mihirmaru` in
  `config/settings.py`. Mem0 filters by this id; deleting the Qdrant directory
  is the only "logout".
- Not optimized for scale. The current implementation is single-process,
  CPU-tolerant, and bounded by the developer's hardware.
- Not a benchmark of any specific RAG technique. The query-rewriter, MMR
  retrieval, and Mem0 memory layer are each independently reasonable; we are
  not making claims about their joint optimality.

---

## Layout

```
config/         paths, model names, identity
ingestion/      hybrid PDF parser, recursive chunker, ingest driver
retrieval/      embeddings client, Chroma wrapper, MMR retriever
llm/            rewrite prompt, RAG prompt, chain factories
memory/         Mem0 + Qdrant configuration, save/search helpers
observability/  psutil-based CPU/RAM snapshot
interfaces/     FastAPI service, Streamlit client
pdfs/           input corpus (user-supplied)
chroma_db_local/        Chroma persistence (generated)
mem0_qdrant_data/       Qdrant local mode (generated)
rag_core.py     legacy monolithic equivalent of the modular split
app.py          legacy monolithic Streamlit UI
```

The two legacy files (`rag_core.py`, `app.py`) and the modular split
(`config/`, `ingestion/`, …, `interfaces/`) encode the same business logic.
The modular split is the canonical target. The legacy files are kept for
reference and for users who prefer a single-file entry point.

---

## Requirements

- Python ≥ 3.10
- Ollama running at `http://localhost:11434` (override with `OLLAMA_HOST`)
- The three models above pulled and resident

Pinned dependencies are in `requirements.txt`. The `langchain==1.2.x` and
`langchain-chroma==1.1.x` lines are recent at the time of writing; older
tutorials will not match this API surface.

---

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

ollama pull nomic-embed-text
ollama pull phi4-mini:latest
ollama pull minimax-m3:cloud

# index the corpus
python run_ingest.py

# CLI REPL
python rag_core.py

# or the Streamlit HUD
streamlit run app.py
```

To force a rebuild of the index, delete `chroma_db_local/` and
`mem0_qdrant_data/` before running `run_ingest.py`. Ingestion is otherwise
idempotent at the directory level; the legacy `rag_core.py` checks for an
existing DB, the modular `ingestion/ingest.py` does not — re-running it
appends.

---

## What to look at if you are reading the code

| If you care about | Read |
|---|---|
| How a PDF becomes a chunk | `ingestion/parser.py`, `ingestion/chunker.py` |
| How a question becomes an answer | `interfaces/api.py` end to end |
| The actual prompts | `llm/prompt.py` (use the 6-rule prompt in `rag_core.py`, not the compressed one) |
| Why memory writes are off the response path | `interfaces/api.py` `BackgroundTasks`; `app.py` `daemon=True` thread |
| The retrieval configuration | `retrieval/retriever.py` — MMR, k=4, fetch_k=15 |
| The Mem0 configuration | `memory/mem0_manager.py` — note the hardcoded 768 dim |
| Where telemetry is logged | `system_performance_log.csv`, `query_performance_log.csv`, `rag_metrics_log.csv` |

---

## Design notes we did not paper over

- The modular `RAG_PROMPT` is more terse than the legacy one. The legacy
  prompt explicitly enumerates six rules (quote numbers, show table
  calculations, synthesize vague questions, cite source + page). The
  modular prompt collapses these. We recommend the legacy version for
  real use; the modular one is a regression we have not yet fixed.
- The `interfaces/webui.py` UI does not expose the model selector or the
  retrieval-source preview that the legacy `app.py` does. The API supports
  both; the UI just does not surface them.
- `observability/metrics.py` (modular) drops GPU telemetry that exists in
  the legacy file. This is a regression.
- `run_ingest.py` re-ingests on every run. It should short-circuit when
  the collection is non-empty.
- There is no authentication on the FastAPI server. It is intended for
  localhost only.

These are listed in the engineering reconstruction document with the
specific files and lines. A reimplementer should treat the legacy files
as the behavior spec and the modular files as the cleaner target.

---

## Evaluation

A `benchmark.json` of the form:

```json
[
  {"question": "...", "expected_source": "report.pdf", "expected_page": 3}
]
```

will drive `run_benchmark()` (from the legacy REPL via the `benchmark`
command). It scores top-k retrieval accuracy only — answer quality is not
measured. There is no LLM-as-judge in this repo. We do not ship a golden
set; bring your own questions and expected sources.

---

## License

Apache 2.0. See `LICENSE`.
