"""PostgreSQL + pgvector persistence layer for MedQuery-RAG.

Works with Supabase (managed Postgres + pgvector) or any local PostgreSQL 16+
with the pgvector extension — same API, connection string driven via
``config.DATABASE_URL``.

Schema (per the verified roadmap):
  documents(id, title, doc_type, source, status, created_at, updated_at)
  chunks(id, document_id, content, chunk_index, page_number, metadata JSONB,
         embedding vector(384), created_at)
  queries(id, question, answer, model, latency_ms, created_at)
  conversation_messages(id, session_id, role, content, created_at)

Retrieval:
  - Semantic: ORDER BY embedding <=> %s (cosine distance, normalized vectors)
  - Keyword:   to_tsvector(content) @@ plainto_tsquery(...)
  - Hybrid:    RRF fusion of the two ranked lists
"""
import json
import threading

import psycopg2
from pgvector.psycopg2 import register_vector
from pgvector import Vector

import config

_lock = threading.Lock()
_conn = None


def get_connection():
    """Thread-local connection so concurrent requests don't trample each other."""
    global _conn
    # psycopg2 connections are not thread-safe; use one per thread via a tiny pool.
    tid = threading.get_ident()
    if not hasattr(_conn_store, "conns"):
        _conn_store.conns = {}
    conn = _conn_store.conns.get(tid)
    if conn is None or conn.closed:
        conn = psycopg2.connect(config.DATABASE_URL)
        register_vector(conn)
        _conn_store.conns[tid] = conn
    # Autocommit mode: every read and write ends its own transaction, so an
    # idle-in-transaction connection can never hold a lock that blocks a
    # TRUNCATE on chunks/documents (this hung the original test suite).
    # psycopg2 refuses to flip autocommit mid-transaction, so end any open
    # transaction first.
    if not conn.autocommit:
        if conn.get_transaction_status() != psycopg2.extensions.TRANSACTION_STATUS_IDLE:
            conn.rollback()
        conn.autocommit = True
    return conn


class _ConnStore:
    pass


_conn_store = _ConnStore()


def ensure_schema():
    """Create the extension + tables + indexes if missing (idempotent).

    Important: the vector extension must exist BEFORE register_vector() can
    succeed, so we create the extension on a raw connection first.
    """
    raw = psycopg2.connect(config.DATABASE_URL)
    raw.autocommit = True
    try:
        raw.cursor().execute("CREATE EXTENSION IF NOT EXISTS vector;")
    except psycopg2.errors.InsufficientPrivilege:
        # Non-superuser role: extension must already be installed by the DBA
        # (e.g. `CREATE EXTENSION vector` in Supabase it ships pre-enabled).
        raw.close()
        raw = psycopg2.connect(config.DATABASE_URL)
        raw.autocommit = True
        cur = raw.cursor()
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        if not cur.fetchone():
            raise RuntimeError(
                "pgvector extension is not installed and the database role "
                "cannot install it. Install it as a superuser first: "
                "CREATE EXTENSION vector;"
            )
        cur.close()
    raw.close()
    conn = get_connection()  # register_vector now succeeds
    cur = conn.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS documents (
            id          TEXT PRIMARY KEY,
            title       TEXT NOT NULL,
            doc_type    TEXT NOT NULL DEFAULT 'General',
            source      TEXT DEFAULT 'manual',
            status      TEXT NOT NULL DEFAULT 'completed',
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id            TEXT PRIMARY KEY,
            document_id   TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            content       TEXT NOT NULL,
            chunk_index   INT NOT NULL,
            page_number   INT,
            metadata      JSONB NOT NULL DEFAULT '{{}}',
            embedding     vector({config.EMBEDDING_DIM}) NOT NULL,
            content_fts   tsvector,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE TABLE IF NOT EXISTS queries (
            id          BIGSERIAL PRIMARY KEY,
            question    TEXT NOT NULL,
            answer      TEXT,
            model       TEXT,
            latency_ms  REAL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE TABLE IF NOT EXISTS conversation_messages (
            id          BIGSERIAL PRIMARY KEY,
            session_id  TEXT NOT NULL,
            role        TEXT NOT NULL,
            content     TEXT NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)
    # Full-text search column kept in sync via trigger
    cur.execute("""
        CREATE OR REPLACE FUNCTION chunks_fts_update() RETURNS trigger AS $$
        BEGIN
            NEW.content_fts := to_tsvector('english', NEW.content);
            RETURN NEW;
        END; $$ LANGUAGE plpgsql;
        DROP TRIGGER IF EXISTS chunks_fts_trg ON chunks;
        CREATE TRIGGER chunks_fts_trg BEFORE INSERT OR UPDATE OF content
            ON chunks FOR EACH ROW EXECUTE FUNCTION chunks_fts_update();
        -- backfill existing rows
        UPDATE chunks SET content = content WHERE content_fts IS NULL;
    """)
    cur.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks
            USING hnsw (embedding vector_cosine_ops);
        CREATE INDEX IF NOT EXISTS idx_chunks_fts ON chunks USING gin (content_fts);
        CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks (document_id);
        CREATE INDEX IF NOT EXISTS idx_conv_session ON conversation_messages (session_id);
    """)
    conn.commit()
    cur.close()


# ── Documents ─────────────────────────────────────────────────────────────────

def list_documents():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT d.id, d.title, d.doc_type, d.source, d.status,
               COALESCE(c.cnt, 0) AS chunk_count
        FROM documents d
        LEFT JOIN (SELECT document_id, COUNT(*)::int AS cnt FROM chunks GROUP BY document_id) c
            ON c.document_id = d.id
        ORDER BY d.created_at;
    """)
    rows = cur.fetchall()
    cur.close()
    return [
        {"id": r[0], "title": r[1], "type": r[2], "source": r[3],
         "status": r[4], "chunk_count": r[5]}
        for r in rows
    ]


def delete_document(doc_id: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM chunks WHERE document_id = %s RETURNING id", (doc_id,))
    removed = len(cur.fetchall())
    cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
    deleted = cur.rowcount
    conn.commit()
    cur.close()
    return removed, deleted > 0


def get_document(doc_id: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, title, doc_type, source FROM documents WHERE id = %s", (doc_id,))
    row = cur.fetchone()
    cur.close()
    return row


# ── Chunks / vector operations ────────────────────────────────────────────────

def store_chunks(doc_id: str, items):
    """items: list of dicts {chunk_id, content, chunk_index, page_number, metadata, embedding}"""
    conn = get_connection()
    cur = conn.cursor()
    for it in items:
        cur.execute(
            "INSERT INTO chunks (id, document_id, content, chunk_index, page_number, metadata, embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (id) DO NOTHING",
            (it["chunk_id"], doc_id, it["content"], it["chunk_index"],
             it.get("page_number"), json.dumps(it.get("metadata", {})),
             Vector(it["embedding"])),
        )
    conn.commit()
    cur.close()


def delete_chunks_for_document(doc_id: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
    conn.commit()
    cur.close()


def chunk_stats():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM chunks;")
    n = cur.fetchone()[0]
    cur.close()
    return n


# ── Retrieval (semantic, keyword, hybrid) ─────────────────────────────────────

def _sem_results(query_vec, k):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, content, 1 - (embedding <=> %s) AS score FROM chunks "
        "ORDER BY embedding <=> %s LIMIT %s",
        (Vector(query_vec), Vector(query_vec), k),
    )
    rows = cur.fetchall()
    cur.close()
    return rows  # (id, content, cosine_score)


def _kw_results(query_text: str, k):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, content, ts_rank(content_fts, plainto_tsquery('english', %s)) AS score "
        "FROM chunks WHERE content_fts @@ plainto_tsquery('english', %s) "
        "ORDER BY score DESC LIMIT %s",
        (query_text, query_text, k),
    )
    rows = cur.fetchall()
    cur.close()
    return rows  # (id, content, bm25-ish rank)


def hybrid_retrieve(query_vec, query_text: str, candidate_k: int, top_k: int):
    """Reciprocal Rank Fusion of semantic + keyword ranked lists.

    Returns [(chunk_id, content, semantic_cosine_score)] — the score is always
    the semantic cosine similarity so callers can compare it against a fixed
    threshold. Keyword ranking only affects ORDER via RRF fusion.
    """
    sem = _sem_results(query_vec, candidate_k)
    kw = _kw_results(query_text, candidate_k)

    sem_scores = {cid: score for cid, _, score in sem}

    ranked = {}
    for rank, (cid, content, score) in enumerate(sem, 1):
        r = ranked.setdefault(cid, [0.0, content])
        r[0] += 1.0 / (60 + rank)
    for rank, (cid, content, score) in enumerate(kw, 1):
        r = ranked.setdefault(cid, [0.0, content])
        r[0] += 1.0 / (60 + rank)

    fused = sorted(ranked.items(), key=lambda kv: -kv[1][0])[:top_k]
    # ranked values: [rrf_score, content]; attach semantic cosine where known
    return [(cid, vals[1], round(sem_scores.get(cid, 0.0), 4))
            for cid, vals in fused]


def semantic_retrieve(query_vec, top_k: int):
    return [(cid, content, score) for cid, content, score in _sem_results(query_vec, top_k)]
