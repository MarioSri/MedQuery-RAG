"""MedQuery-RAG core pipeline — persistent, provider-agnostic version.

Architecture per the verified roadmap:

  FastAPI backend
       |
  RAG engine ── llm_providers.py (LLMProvider: Anthropic/OpenAI/Gemini/Ollama/Custom)
       |        embedding_providers.py (EmbeddingProvider: ST/OpenAI/HF/Custom)
  pgvector DB ── db_store.py (documents / chunks / queries / conversations)
       |
  retrieval: hybrid (vector + keyword, RRF) -> reranker -> threshold -> LLM

Knowledge storage != conversation memory (conversations kept in their own
table, used only for optional query rewriting).
"""
import re
import time
import uuid
from typing import Dict, List

import numpy as np

import config
import db_store
from embedding_providers import get_embedding_provider
from llm_providers import get_llm_provider
from reranker import get_reranker

# Module-level singletons (initialized once at startup)
embedding_model = None
llm = None
reranker = None

# Medically grounded system prompt
SYSTEM_PROMPT = (
    "You are MedQuery, a medical RAG assistant. Answer the question ONLY using "
    "the retrieved medical documents below. Cite the source title and page/chunk "
    "where each claim comes from, as inline markers like [1], [2]. If the retrieved "
    "information is insufficient to answer, say so plainly. Do not invent clinical "
    "facts. End with a reminder to consult a healthcare professional for personal "
    "medical decisions."
)

INSUFFICIENT_EVIDENCE = (
    "No sufficiently relevant information was found in the knowledge base to "
    "answer this question confidently. Please rephrase, add supporting documents, "
    "or consult a healthcare professional."
)


def initialize():
    global embedding_model, llm, reranker
    print("[INFO] Connecting to PostgreSQL + pgvector...")
    db_store.ensure_schema()
    print("[INFO] Loading embedding model...")
    embedding_model = get_embedding_provider()
    print(f"[OK] Embeddings ready -- {embedding_model.model_id} ({embedding_model.dimension} dims)")
    print("[INFO] Initializing LLM provider...")
    llm = get_llm_provider()
    print(f"[OK] LLM ready -- {llm.model_id}")
    print("[INFO] Initializing reranker...")
    reranker = get_reranker()
    print(f"[OK] Reranker ready -- {type(reranker).__name__}")


# ── Chunking ──────────────────────────────────────────────────────────────────

def _guard_min_length(content: str):
    """P1 contract: documents below a minimum length can never be indexed."""
    if len(content.strip()) < 41:
        raise ValueError("Document is too short to be indexed.")


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 120) -> List[str]:
    """Sentence-boundary chunking (unchanged algorithm from the original repo)."""
    if not text or not text.strip():
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        return []

    chunks = []
    current_chunk = ""
    last_boundary_end = 0

    def add_chunk(text_):
        text_ = text_.strip()
        if text_:
            chunks.append(text_)

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk = (current_chunk + " " + sentence).strip()
        else:
            if len(current_chunk) > 0:
                if len(current_chunk) >= min(40, chunk_size):
                    add_chunk(current_chunk)
                    tail = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    boundary_pos = re.search(r'(?<=[.!?])\s', tail)
                    if boundary_pos:
                        current_chunk = tail[boundary_pos.end():] + " " + sentence
                    else:
                        current_chunk = tail + " " + sentence
                    last_boundary_end = len(current_chunk)
                    continue
                else:
                    next_start = last_boundary_end + len(current_chunk)
            current_chunk = sentence
            last_boundary_end = len(current_chunk)

    if current_chunk:
        add_chunk(current_chunk)
    return chunks


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_document(doc_type: str, title: str, content: str,
                    source: str = "manual", chunks_with_pages=None) -> Dict:
    """Ingest a document into PostgreSQL + pgvector.

    chunks_with_pages: optional list of (content, page_number) from structured
    PDF extraction; falls back to chunk_text(content) when None.
    Raises ValueError if the content produces zero chunks (no silent drops).
    """
    global embedding_model
    if embedding_model is None:
        raise RuntimeError("Pipeline not initialized")

    _guard_min_length(content)
    doc_id = f"doc_{uuid.uuid4().hex[:8]}"

    if chunks_with_pages:
        text_chunks = [c for c, _ in chunks_with_pages]
    else:
        text_chunks = chunk_text(content)

    if not text_chunks:
        raise ValueError("Document is too short to be indexed.")

    # Embed all chunks
    new_emb = embedding_model.encode(text_chunks)

    items = []
    for i, tc in enumerate(text_chunks):
        page = chunks_with_pages[i][1] if chunks_with_pages else None
        items.append({
            "chunk_id": f"{doc_id}_chunk_{i:03d}",
            "content": tc,
            "chunk_index": i,
            "page_number": page,
            "metadata": {"doc_type": doc_type, "doc_title": title},
            "embedding": new_emb[i],
        })

    with db_store._lock:
        conn = db_store.get_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO documents (id, title, doc_type, source) VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (id) DO NOTHING",
            (doc_id, title, doc_type, source),
        )
        conn.commit()
        cur.close()
        db_store.store_chunks(doc_id, items)

    print(f"[INGEST] Added '{title}' ({doc_id}) — {len(items)} chunks")
    return {
        "doc_id": doc_id,
        "chunks_added": len(items),
        "total_chunks": db_store.chunk_stats(),
        "total_documents": len(db_store.list_documents()),
    }


def delete_document(doc_id: str) -> Dict:
    with db_store._lock:
        removed, deleted = db_store.delete_document(doc_id)
    return {"success": deleted, "chunks_removed": removed,
            "total_documents": len(db_store.list_documents()),
            "total_chunks": db_store.chunk_stats(),
            "message": "Document deleted" if deleted else f"Document {doc_id} not found"}


def get_documents() -> List[Dict]:
    return db_store.list_documents()


# ── Retrieval ─────────────────────────────────────────────────────────────────

def _retrieve_candidates(query: str):
    query_vec = embedding_model.encode([query])[0].tolist()
    if config.RERANKER.lower() in ("none", "off", "false", ""):
        results = db_store.hybrid_retrieve(
            query_vec, query,
            candidate_k=config.RETRIEVAL_TOP_K,
            top_k=config.RETRIEVAL_TOP_K,
        )
        return [(cid, content, score) for cid, content, score in results]
    # Reranking re-orders candidates; map each reranked chunk back to its
    # semantic cosine score so all returned scores stay comparable.
    candidates = db_store.hybrid_retrieve(
        query_vec, query,
        candidate_k=config.RETRIEVAL_CANDIDATE_K,
        top_k=config.RETRIEVAL_CANDIDATE_K,
    )
    cosine_of = {cid: score for cid, _, score in candidates}
    reranked = reranker.rerank(query, candidates)
    return [(cid, content, round(cosine_of.get(cid, 0.0), 4))
            for cid, content, _ in reranked]


def retrieve(query: str, top_k: int = None):
    top_k = top_k or config.RETRIEVAL_TOP_K
    results = _retrieve_candidates(query)[:top_k]
    if not results:
        return [], []
    # Scores are always semantic cosine similarity (0..1). Filter every
    # candidate below the evidence threshold so weak matches are not sent to
    # the LLM or displayed as citations. If no candidate survives, the caller
    # returns the explicit insufficient-evidence response.
    if config.THRESHOLD_ENABLED:
        results = [r for r in results if r[2] >= config.SIMILARITY_THRESHOLD]
    scores = [r[2] for r in results]
    return results, scores


# ── Query answering ───────────────────────────────────────────────────────────

def rag_answer(question: str, top_k: int = None, conversation=None) -> Dict:
    """Full RAG flow: retrieve -> threshold -> build context with citations -> LLM.

    conversation: optional list of {"role", "content"} used only to rewrite the
    query (conversation memory is stored separately from the knowledge base).
    """
    global llm
    start = time.time()
    top_k = top_k or config.RETRIEVAL_TOP_K

    # Optional lightweight query rewriting using conversation context
    effective_question = question
    if conversation and llm is not None and _needs_rewrite(conversation, question):
        try:
            rewrite = llm.generate(
                prompt=_rewrite_prompt(conversation, question),
                system="Rewrite the user's latest question into a self-contained, "
                       "retrieval-friendly question. Output only the rewritten question.",
            )["text"].strip()
            if rewrite:
                effective_question = rewrite
        except Exception as e:
            print(f"[WARN] Query rewrite failed ({e}); using original question")

    results, scores = retrieve(effective_question, top_k)
    latency_ms = round((time.time() - start) * 1000, 2)
    tokens_used = 0

    if not results:
        answer = INSUFFICIENT_EVIDENCE
        sources, chunks_out = [], []
    else:
        # Build numbered context with page/chunk citation metadata
        numbered = []
        sources = []
        chunks_out = []
        for i, (cid, content, score) in enumerate(results, 1):
            numbered.append(f"[{i}] {content}")
            src = _source_for_chunk(cid)
            src["rank"] = i
            src["similarity"] = round(score, 4)
            sources.append(src)
            chunks_out.append(content)

        context = "\n\n".join(numbered)
        prompt = f"Retrieved medical documents:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"
        try:
            resp = llm.generate(prompt, system=SYSTEM_PROMPT)
            answer = resp["text"]
            tokens_used = resp["usage"].get("input_tokens", 0) + resp["usage"].get("output_tokens", 0)
        except Exception as e:
            answer = f"(LLM unavailable: {type(e).__name__}) — retrieved {len(results)} relevant chunk(s): {context[:500]}"
            tokens_used = 0

    # Persist query log (separate from conversation memory)
    try:
        model_id = str(llm.model_id) if llm else "none"
        _log_query(question, answer, model_id, latency_ms)
    except Exception as e:
        print(f"[WARN] Query logging failed: {e}")

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": chunks_out,
        "similarity_scores": [round(s, 4) for s in scores],
        "latency_ms": latency_ms,
        "tokens_used": tokens_used,
    }


def _source_for_chunk(chunk_id: str):
    """Look up rich citation metadata for a chunk."""
    conn = db_store.get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT c.document_id, c.chunk_index, c.page_number, c.metadata, "
        "       d.title, d.doc_type, d.source "
        "FROM chunks c JOIN documents d ON d.id = c.document_id WHERE c.id = %s",
        (chunk_id,),
    )
    row = cur.fetchone()
    cur.close()
    if not row:
        return {"doc_id": chunk_id.rsplit("_chunk_", 1)[0]}
    doc_id, idx, page, meta, title, dtype, source = row
    return {
        "doc_id": doc_id, "chunk_index": idx, "page_number": page,
        "doc_title": title, "doc_type": dtype, "source": source,
    }


def _needs_rewrite(conversation, question: str) -> bool:
    """Heuristic: ambiguous standalone questions after a substantive conversation."""
    if len(conversation) < 2:
        return False
    vague = re.match(r"^(what|which|who|where|when|why|how|was|were|did|does|do|is|are)\b", question, re.I)
    short = len(question.split()) <= 6
    return bool(vague and short)


def _rewrite_prompt(conversation, question: str) -> str:
    history = "\n".join(
        f"{m['role']}: {m['content']}" for m in conversation[-6:]
    )
    return f"Conversation history:\n{history}\n\nLatest question: {question}\n\nRewrite as a self-contained question:"


def _log_query(question, answer, model, latency_ms):
    conn = db_store.get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO queries (question, answer, model, latency_ms) VALUES (%s, %s, %s, %s)",
        (question, answer[:4000], model, latency_ms),
    )
    conn.commit()
    cur.close()


def get_metrics() -> Dict:
    conn = db_store.get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*), COALESCE(AVG(latency_ms), 0) FROM queries;")
    total, avg_lat = cur.fetchone()
    cur.execute("""
        SELECT COUNT(DISTINCT document_id), COUNT(*) FROM chunks;
    """)
    ndocs, nchunks = cur.fetchone()
    # Random sampling inside a scalar subquery is not allowed in Postgres, so
    # fetch a random sample first and average the cosine similarities locally.
    if nchunks > 0:
        cur.execute("SELECT embedding FROM chunks ORDER BY RANDOM() LIMIT 100")
        embs = [row[0].to_list() if hasattr(row[0], "to_list") else [float(c) for c in row[0]] for row in cur.fetchall()]
        arr = np.array(embs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
        sims = (arr / norms).dot((arr[0] / norms[0]))
        avg_sim = float(sims.mean())
    else:
        avg_sim = 0.0
    cur.close()
    return {
        "total_queries": total,
        "documents_indexed": ndocs,
        "chunks_indexed": nchunks,
        "avg_latency_ms": round(avg_lat, 2),
        "avg_similarity_score": round(float(avg_sim), 4),
        "embedding_model": str(embedding_model.model_id) if embedding_model else config.EMBEDDING_MODEL,
        "embedding_dim": int(embedding_model.dimension) if embedding_model else config.EMBEDDING_DIM,
        "vector_db_type": "PostgreSQL + pgvector (hnsw)",
        "retrieval_mode": "hybrid (vector + keyword)",
        "reranker": config.RERANKER,
        "similarity_threshold": config.SIMILARITY_THRESHOLD if config.THRESHOLD_ENABLED else None,
        "llm_provider": str(llm.model_id) if llm else config.LLM_PROVIDER,
    }


# ── Conversation memory (kept separate from knowledge base) ──────────────────

def add_conversation_message(session_id: str, role: str, content: str):
    conn = db_store.get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO conversation_messages (session_id, role, content) VALUES (%s, %s, %s)",
        (session_id, role, content),
    )
    conn.commit()
    cur.close()


def get_conversation(session_id: str, max_messages: int = 20) -> List[Dict]:
    conn = db_store.get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT role, content FROM conversation_messages WHERE session_id = %s "
        "ORDER BY id DESC LIMIT %s",
        (session_id, max_messages),
    )
    rows = list(reversed(cur.fetchall()))
    cur.close()
    return [{"role": r, "content": c} for r, c in rows]
