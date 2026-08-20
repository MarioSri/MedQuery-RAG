# MedQuery

MedQuery is a production-grade Retrieval-Augmented Generation (RAG) system for medical question answering. It retrieves evidence from an indexed medical corpus using hybrid semantic and keyword search, reranks candidates with a cross-encoder, and generates grounded answers through a provider-agnostic LLM interface. All state is persisted in PostgreSQL with pgvector.

## Overview

When a question arrives, MedQuery embeds it with a sentence transformer, retrieves candidate passages from PostgreSQL using a combination of HNSW vector search and BM25 keyword search, reranks the merged candidate set with a cross-encoder, and passes the top passages to the configured LLM. If no passage exceeds the configured similarity threshold, the system refuses to answer rather than hallucinating. Indexed documents include the MedQuAD corpus of NIH question-answer pairs, and users can add their own PDFs or raw text through the dashboard or API.

The system consists of two components:

- **Backend** — a FastAPI service implementing the RAG pipeline, persistence, and REST endpoints.
- **Frontend** — a single-page dashboard for querying the system, managing documents, and monitoring metrics.

## Features

- Hybrid retrieval combining HNSW vector search with BM25 keyword matching
- Cross-encoder reranking with a configurable similarity threshold and explicit insufficient-evidence responses
- Provider-agnostic LLM and embedding interfaces (Anthropic, OpenAI, Gemini, Ollama) configured entirely through environment variables
- Persistent storage with PostgreSQL and pgvector, including document and chunk metadata, page numbers, and per-document chunk counts
- Background corpus ingestion with progress tracking (MedQuAD)
- Conversation history with session-scoped retrieval context
- Live metrics endpoint covering queries, similarity scores, latency, and index statistics
- Dashboard with live-updating analytics, document management, and ranked source attribution

## Architecture

```text
User ──► Dashboard (HTML/CSS/JS) ──► FastAPI Backend
                                            │
                      ┌─────────────────────┼──────────────────────┐
                      │                     │                      │
              Query Embedding         Hybrid Retrieval        LLM Generation
              (SentenceTransformer)   (HNSW + BM25)           (Anthropic/OpenAI/
                      │                     │                   Gemini/Ollama)
                      │             CrossEncoder Rerank            │
                      │                     │                      │
                      └──────────── PostgreSQL + pgvector ◄────────┘
                                  (documents, chunks, conversations)
```

## Quick Start

Clone the repository and install dependencies:

```bash
git clone https://github.com/MarioSri/MedQuery-RAG.git
cd MedQuery-RAG
pip install -r requirements.txt
```

Set up PostgreSQL with the pgvector extension:

```sql
CREATE DATABASE medquery;
\c medquery
CREATE EXTENSION IF NOT EXISTS vector;
```

Create a `.env` file with your configuration:

```env
LLM_PROVIDER=anthropic
LLM_API_KEY=YOUR_LLM_API_KEY
EMBEDDING_PROVIDER=openai_local
EMBEDDING_API_KEY=YOUR_EMBEDDING_API_KEY
DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/medquery
HOST=0.0.0.0
PORT=8000
CORS_ORIGINS=*
```

Each provider uses its own credentials and model settings; see the Configuration section for the full list.

Start the server:

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Open `medquery-dashboard.html` in a browser to query the system and manage documents.

## REST API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check and version |
| POST | `/query` | Ask a question; returns answer, sources, and similarity scores |
| POST | `/conversations` | Start a conversation session for context-aware follow-ups |
| GET | `/conversations/{conversation_id}` | Retrieve conversation history |
| POST | `/documents/ingest` | Index raw text |
| POST | `/documents/upload` | Index an uploaded PDF |
| GET | `/documents` | List indexed documents with chunk counts |
| DELETE | `/documents/{doc_id}` | Remove a document and its chunks |
| GET | `/metrics` | Live pipeline and retrieval metrics |
| POST | `/ingest-medquad` | Trigger background MedQuAD corpus ingestion |

## Configuration

The backend is configured exclusively through environment variables, allowing the same deployment to run against different providers without code changes.

| Variable | Description | Example |
|---|---|---|
| `LLM_PROVIDER` | `anthropic`, `openai`, `gemini`, or `ollama` | `anthropic` |
| `LLM_API_KEY` | API key for the selected LLM provider | |
| `LLM_MODEL` | Model name (default varies by provider) | `claude-3-5-haiku-latest` |
| `EMBEDDING_PROVIDER` | `openai_local` (local SentenceTransformer) or `openai` | `openai_local` |
| `EMBEDDING_MODEL` | Embedding model name | `all-MiniLM-L6-v2` |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+psycopg://user:pass@localhost/medquery` |
| `RERANKER_ENABLED` | Enable cross-encoder reranking | `true` |
| `SIMILARITY_THRESHOLD` | Minimum score to answer; below this, the system refuses | `0.35` |
| `TOP_K` | Number of passages retrieved before reranking | `20` |

## Project Structure

```text
MedQuery-RAG/
├── main.py                  # FastAPI application entry point
├── routes.py                # REST endpoints
├── rag_pipeline.py          # RAG logic: chunking, retrieval, answering
├── db_store.py              # PostgreSQL + pgvector persistence layer
├── llm_providers.py         # Provider-agnostic LLM interface
├── embedding_providers.py   # Provider-agnostic embedding interface
├── reranker.py              # Cross-encoder reranking
├── config.py                # Environment-based configuration
├── models.py                # Pydantic request/response models
├── requirements.txt         # Python dependencies
├── serve_dashboard.py       # Static file server for the dashboard
├── medquery-dashboard.html  # Frontend dashboard
├── test_medquery.py         # Test suite
├── LICENSE
└── README.md
```

Diagnostic and scratch files are excluded from version control via `.gitignore`.

## Running Tests

```bash
python -m unittest test_medquery -v
```

The test suite covers chunking and overlap behavior, hybrid retrieval and reranking, the similarity threshold refusal path, zero-chunk ingestion rejection, document lifecycle operations, conversation memory, and metrics computation.

## Data

MedQuAD (Medical Question Answering Dataset) is a corpus of real medical questions and answers compiled from NIH websites. MedQuery ingests it through a background pipeline and answers domain questions by retrieving against this corpus together with any user-uploaded documents.

## Limitations and Future Work

The system targets single-instance deployment and currently has no authentication. Planned extensions include streaming responses, OCR support for scanned PDFs, metadata filtering, a formal evaluation harness (Recall/MRR, faithfulness), and containerized deployment.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for the full text.
