# Local Mind

Local Mind is a local-first RAG application for searching, summarizing, and querying PDF-based knowledge using Ollama, Chroma, Mem0, and Streamlit. It is designed for experimentation with enterprise-style document analysis while keeping the core workflow local and privacy-friendly.

## Overview

This project combines:

- PDF ingestion and parsing
- Vector retrieval over local embeddings
- Conversational RAG using local or configured LLM models
- Memory-aware question answering through Mem0
- A small Streamlit dashboard for interactive use

It is intended for developers who want to run a document Q&A system on their own machine without depending on cloud-only APIs for the core path.

## Key Features

- Hybrid PDF parsing for both text and table-based content
- Local embeddings using Ollama (`nomic-embed-text`)
- Query rewriting for vague or follow-up questions
- Long-term memory search using Mem0
- Performance logging for CPU, RAM, and GPU telemetry
- Benchmarking and debugging tools for retrieval inspection
- Streamlit UI for an interactive chat experience

## Architecture

1. PDF files are loaded from the `pdfs/` directory.
2. The content is parsed and split into chunks.
3. Chroma stores the vector embeddings in `chroma_db_local/`.
4. Mem0 stores conversational memory in `mem0_qdrant_data/`.
5. The app retrieves relevant chunks and generates answers through the configured LLM chain.

## Repository Layout

- `app.py` — Streamlit chat interface
- `rag_core.py` — ingestion, retrieval, memory, prompt, and query logic
- `pdfs/` — source PDFs to be indexed
- `chroma_db_local/` — local vector database cache
- `mem0_qdrant_data/` — local memory database storage
- `requirement.txt` — Python dependencies
- `query_performance_log.csv`, `rag_metrics_log.csv`, `system_performance_log.csv` — runtime logs

## Requirements

Before using this project, make sure you have:

- Python 3.10 or newer
- Ollama installed and running locally
- Access to the following models if you want to use the default local path:
  - `nomic-embed-text`
  - `phi4-mini:latest`
  - `minimax-m3:cloud` (optional, depends on your configuration)

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirement.txt
```

### 3. Start Ollama

Make sure the required models are available in your local Ollama setup.

### 4. Add PDFs

Place your PDF documents in the `pdfs/` folder before running ingestion.

## Running the Project

### Launch the Streamlit UI

```bash
streamlit run app.py
```

### Run the CLI pipeline directly

```bash
python rag_core.py
```

## Usage Notes

- The first run will build or refresh the local vector database.
- Existing embeddings are reused on later runs when the DB is available.
- To force a fresh rebuild, remove the local database folders under `chroma_db_local/` and `mem0_qdrant_data/`.
- The system logs performance metrics during ingestion, query processing, and shutdown.

## Troubleshooting

- If Ollama models are unavailable, install them first with `ollama pull <model-name>`.
- If the app fails to start, confirm that dependencies are installed in the active Python environment.
- If retrieval seems weak, try re-ingesting the PDF set after cleaning the local vector store.

## License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.
