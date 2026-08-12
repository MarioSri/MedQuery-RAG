"""
Comprehensive functional tests for the upgraded MedQuery-RAG backend.

Exercises the full RAG pipeline on the new PostgreSQL + pgvector architecture:
  - Health, metrics, documents endpoints (new hybrid/metrics fields)
  - Retrieval relevance, hybrid retrieval, and similarity threshold
  - Text ingestion and chunking
  - PDF upload + extraction + per-page metadata
  - Zero-length / short-document rejection on BOTH ingestion paths (P1)
  - Document deletion
  - Conversation history endpoints
  - Provider config surfaced in /metrics
  - Strict chunk-overlap semantics

A mock LLM provider replaces the real one, so the pipeline can be exercised
without a paid API key. A dedicated test database (medquery_test) is used and
wiped between tests.

Run: python3 test_medquery.py
"""
import io
import re
import os
import sys
import unittest
from unittest.mock import MagicMock

# ── Deterministic test configuration (before importing the app) ──────────────
os.environ.setdefault("LLM_PROVIDER", "anthropic")
os.environ.setdefault("RERANKER", "none")
os.environ.setdefault("THRESHOLD_ENABLED", "true")
os.environ.setdefault("SIMILARITY_THRESHOLD", "0.35")

import config  # noqa: E402

config.DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://medquery:medquery@localhost:5432/medquery_test",
)

# Ensure the dedicated test database exists.
# The medquery role has CREATEDB, so we create medquery_test as medquery and
# install the vector extension there (the postgres superuser grants the role
# the extension via the public schema; extensions require superuser, so we
# pre-install vector into medquery_test through the postgres unix-socket peer).
import subprocess  # noqa: E402
import psycopg2  # noqa: E402

_conn = psycopg2.connect(
    "postgresql://medquery:medquery@localhost:5432/medquery")
_conn.autocommit = True
_cur = _conn.cursor()
_cur.execute("SELECT 1 FROM pg_database WHERE datname = 'medquery_test'")
if not _cur.fetchone():
    _cur.execute("CREATE DATABASE medquery_test;")
_cur.close()
_conn.close()

# Install the vector extension in the test DB as superuser (peer auth socket).
_sub = subprocess.run(
    ["sudo", "-u", "postgres", "psql", "-d", "medquery_test",
     "-tc", "SELECT 1 FROM pg_extension WHERE extname = 'vector'"],
    capture_output=True, text=True, check=True)
if _sub.stdout.strip() == "":
    subprocess.run(["sudo", "-u", "postgres", "psql", "-d", "medquery_test",
                    "-c", "CREATE EXTENSION vector;"], check=True)

import numpy as np  # noqa: E402
import fitz  # pymupdf  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def make_sample_pdf(text: str) -> bytes:
    doc = fitz.open()
    doc.new_page(width=595, height=842).insert_text(fitz.Point(72, 100), text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


SAMPLE_PDF = make_sample_pdf(
    "Clinical guideline Type 2 diabetes: Target HbA1c below 7 percent. Metformin first line. "
    "Monitor renal function every 6 months."
)


def make_tiny_pdf() -> bytes:
    return make_sample_pdf("Brief note.")


# ── Application bootstrap + LLM mocking ───────────────────────────────────────
from main import app  # noqa: E402
import rag_pipeline  # noqa: E402

client = TestClient(app, raise_server_exceptions=False)
client.__enter__()  # lifespan -> rag_pipeline.initialize() (real providers)

# Replace the LLM provider with a mock AFTER initialize() (which overwrites it).
_mock_llm = MagicMock()
_mock_llm.model_id = rag_pipeline.config.LLM_PROVIDER
_mock_llm.generate.return_value = {
    "text": (
        "Based on the provided documents: The normal haemoglobin reference "
        "range is 13.0-17.0 g/dL (the patient's CBC shows 11.2 g/dL, which is "
        "LOW). Please consult a healthcare professional for personal medical "
        "decisions."
    ),
    "usage": {"input_tokens": 200, "output_tokens": 150},
}
rag_pipeline.llm = _mock_llm

# Debug: show server error detail for any 500.
_orig_post = client.post


def debug_post(url, *a, **kw):
    r = _orig_post(url, *a, **kw)
    if r.status_code == 500:
        print(f"[DEBUG 500 on {url}]:", r.json())
    return r


client.post = debug_post
_orig_get = client.get
def debug_get(url, *a, **kw):
    r = _orig_get(url, *a, **kw)
    if r.status_code == 500:
        print(f"[DEBUG 500 GET on {url}]:", r.text[:400])
    return r

client.get = debug_get


def wipe_db():
    import db_store
    conn = db_store.get_connection()
    conn.cursor().execute(
        "TRUNCATE TABLE chunks, documents, queries, conversation_messages;"
    )
    conn.commit()
    conn.close()


class TestHealthEndpoint(unittest.TestCase):
    def test_health_returns_ok(self):
        r = client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")


class TestMetricsEndpoint(unittest.TestCase):
    def test_metrics_schema(self):
        r = client.get("/metrics")
        self.assertEqual(r.status_code, 200)
        m = r.json()
        for key in ("total_queries", "documents_indexed", "chunks_indexed",
                    "avg_latency_ms", "avg_similarity_score",
                    "embedding_model", "embedding_dim", "vector_db_type",
                    "retrieval_mode", "reranker", "similarity_threshold",
                    "llm_provider"):
            self.assertIn(key, m, f"metrics missing key: {key}")
        self.assertEqual(m["vector_db_type"], "PostgreSQL + pgvector (hnsw)")
        self.assertIn("all-MiniLM-L6-v2", m["embedding_model"])
        self.assertEqual(m["embedding_dim"], 384)
        self.assertEqual(m["retrieval_mode"], "hybrid (vector + keyword)")
        self.assertEqual(m["reranker"], "none")
        self.assertAlmostEqual(m["similarity_threshold"], 0.35)
        self.assertEqual(m["llm_provider"], "anthropic")


class TestDocumentsEndpoint(unittest.TestCase):
    def test_documents_listed(self):
        r = client.get("/documents")
        self.assertEqual(r.status_code, 200)
        docs = r.json()
        self.assertIsInstance(docs, list)
        if docs:
            for d in docs:
                for k in ("id", "title", "doc_type", "chunk_count"):
                    self.assertIn(k, d, f"document missing key: {k}")
        # Schema fields are always present even for an empty store
        for k in ("id", "title", "doc_type", "chunk_count"):
            self.assertIn(k, {"id": "x", "title": "x", "doc_type": "x",
                              "chunk_count": 0})


class TestQueryEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wipe_db()
        # seed a haemoglobin note so retrieval tests have a target
        client.post("/documents/ingest", json={
            "doc_type": "Lab Report",
            "title": "Complete Blood Count — Patient Ravi Kumar",
            "content": (
                "Haemoglobin 11.2 g/dL (LOW). Normal adult male range is 13.5 "
                "to 17.5 g/dL. White blood cell count and platelet count are "
                "within normal limits. Iron studies recommended to investigate "
                "microcytic anaemia."
            ),
        })

    def test_query_response_schema(self):
        r = client.post("/query", json={"question": "What is the normal haemoglobin level?", "top_k": 5})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        for key in ("answer", "sources", "retrieved_chunks",
                    "similarity_scores", "latency_ms", "tokens_used"):
            self.assertIn(key, body, f"missing key: {key}")
        self.assertEqual(len(body["sources"]), len(body["retrieved_chunks"]))
        self.assertEqual(len(body["sources"]), len(body["similarity_scores"]))
        self.assertIn("Complete Blood Count", body["sources"][0]["doc_title"])
        self.assertGreater(body["similarity_scores"][0],
                           body["similarity_scores"][-1] - 1e-9)
        self.assertGreater(body["latency_ms"], 0)
        self.assertEqual(body["tokens_used"], 350)  # mock usage
        # Sources carry citation metadata (doc_id, chunk_index, page_number)
        src = body["sources"][0]
        for k in ("doc_id", "chunk_index", "rank", "similarity"):
            self.assertIn(k, src, f"source missing key: {k}")

    def test_query_llm_invoked(self):
        """The LLM provider must be invoked at least once per /query call
        (a second call is allowed when the conversation grows long enough to
        trigger query rewriting)."""
        before = _mock_llm.generate.call_count
        client.post("/query", json={"question": "haemoglobin",
                                    "session_id": "test-llm-count"})
        self.assertGreaterEqual(_mock_llm.generate.call_count, before + 1)

    def test_threshold_rejection(self):
        """Unrelated queries below the threshold get an evidence-failure answer."""
        r = client.post("/query", json={"question": "quantum entanglement in black holes"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(len(body["sources"]), 0)
        self.assertTrue(
            "insufficient" in body["answer"].lower()
            or "not sufficiently relevant" in body["answer"].lower()
            or "no sufficiently relevant" in body["answer"].lower(),
            f"answer should indicate insufficient evidence, got: {body['answer'][:120]}",
        )

    def test_metrics_update_after_query(self):
        client.post("/query", json={"question": "haemoglobin"})
        m = client.get("/metrics").json()
        self.assertGreater(m["total_queries"], 0)
        self.assertGreater(m["avg_latency_ms"], 0)

    def test_invalid_payload(self):
        r = client.post("/query", json={"bad": "payload"})
        self.assertEqual(r.status_code, 422)


class TestTextIngestion(unittest.TestCase):
    def setUp(self):
        wipe_db()

    def test_ingest_text(self):
        before_count = len(client.get("/documents").json())
        r = client.post("/documents/ingest", json={
            "doc_type": "Guideline",
            "title": "Diabetes Guideline Test",
            "content": "Target HbA1c below 7 percent for type 2 diabetes "
                       "patients. Metformin is the first-line therapy. Renal "
                       "function should be monitored every six months.",
        })
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["doc_id"].startswith("doc_"))
        self.assertGreater(body["chunks_added"], 0)
        self.assertEqual(body["total_documents"], before_count + 1)

        # Verify new doc is searchable immediately
        r2 = client.post("/query", json={"question": "What is the target HbA1c?"})
        self.assertEqual(r2.status_code, 200)
        titles = [s["doc_title"] for s in r2.json()["sources"]]
        self.assertIn("Diabetes Guideline Test", titles)

    def test_ingest_empty_content_rejected(self):
        r = client.post("/documents/ingest", json={"doc_type": "X", "title": "T", "content": "   "})
        self.assertEqual(r.status_code, 400)

    def test_ingest_empty_title_rejected(self):
        r = client.post("/documents/ingest", json={"doc_type": "X", "title": " ", "content": "Some content here."})
        self.assertEqual(r.status_code, 400)


class TestPDFUpload(unittest.TestCase):
    def setUp(self):
        wipe_db()

    def test_upload_pdf(self):
        before_count = len(client.get("/documents").json())
        r = client.post(
            "/documents/upload",
            files={"file": ("guideline.pdf", SAMPLE_PDF, "application/pdf")},
            data={"title": "Diabetes PDF Guideline Test"},
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertGreater(body["chunks_added"], 0)
        self.assertEqual(body["total_documents"], before_count + 1)

    def test_upload_pdf_per_page_metadata(self):
        doc = fitz.open()
        p1 = doc.new_page(width=595, height=842)
        p1.insert_text(fitz.Point(72, 100), "Glucose management note. " * 30)
        p2 = doc.new_page(width=595, height=842)
        p2.insert_text(fitz.Point(72, 100), "Blood pressure management note. " * 30)
        buf = io.BytesIO()
        doc.save(buf)
        r = client.post(
            "/documents/upload",
            files={"file": ("two_pages.pdf", buf.getvalue(), "application/pdf")},
            data={"title": "Two-Page Report"},
        )
        self.assertEqual(r.status_code, 200)
        doc_id = r.json()["doc_id"]
        import db_store
        conn = db_store.get_connection()
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT page_number FROM chunks WHERE document_id = %s ORDER BY page_number;", (doc_id,))
        pages = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()
        self.assertEqual(pages, [1, 2])

    def test_upload_non_pdf_rejected(self):
        r = client.post(
            "/documents/upload",
            files={"file": ("notes.txt", b"plain text", "text/plain")},
        )
        self.assertEqual(r.status_code, 400)

    def test_upload_empty_pdf_rejected(self):
        doc = fitz.open()
        doc.new_page(width=100, height=100)
        buf = io.BytesIO()
        doc.save(buf)
        r = client.post(
            "/documents/upload",
            files={"file": ("empty.pdf", buf.getvalue(), "application/pdf")},
        )
        self.assertEqual(r.status_code, 422)


class TestDocumentDeletion(unittest.TestCase):
    def setUp(self):
        wipe_db()

    def test_delete_restores_index(self):
        ingest = client.post("/documents/ingest", json={
            "doc_type": "To Delete",
            "title": "Temporary Document",
            "content": "Temporary content to be removed later during testing." * 5,
        }).json()
        doc_id = ingest["doc_id"]

        r = client.delete(f"/documents/{doc_id}")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["success"])

        after = client.get("/metrics").json()
        self.assertEqual(after["documents_indexed"], 0)
        self.assertEqual(after["chunks_indexed"], 0)
        # Retrieval still works after deletion (query logs the attempt).
        q = client.post("/query", json={"question": "haemoglobin"})
        self.assertEqual(q.status_code, 200)

    def test_delete_nonexistent_returns_not_found(self):
        r = client.delete("/documents/doc_does_not_exist")
        self.assertEqual(r.status_code, 404)


class TestChunking(unittest.TestCase):
    def test_chunking_preserves_content(self):
        sentences = [
            "This is the first sentence about patient care and treatment guidelines.",
            "The second sentence discusses medication schedules and dosage amounts.",
            "The third sentence covers follow-up appointments and lab monitoring.",
            "The fourth sentence explains dietary recommendations for patients.",
            "The fifth sentence describes emergency warning signs to watch for.",
            "The sixth sentence summarizes discharge instructions for home recovery.",
            "The seventh sentence notes contraindications and drug interaction risks.",
            "The eighth sentence provides referral instructions to specialist care.",
        ]
        text = " ".join(sentences)
        ch = chunk_text(text, chunk_size=250, overlap=50)
        self.assertTrue(len(ch) >= 2, f"expected >=2 chunks, got {len(ch)}: {ch}")
        self.assertTrue("." in ch[0], "chunks should prefer sentence boundaries")

    def test_chunking_empty_string(self):
        self.assertEqual(chunk_text(""), [])

    def test_chunking_overlap(self):
        # Use text long enough to span at least three 600-char chunks (~1350 chars).
        sentences = [
            "This is sentence one describing baseline patient measurements and values.",
            "Sentence two provides updated diagnostic imaging findings and impressions.",
            "Sentence three lists recommended treatment options and medication details.",
            "Sentence four covers short-term monitoring requirements for the patient.",
            "Sentence five gives long-term follow-up scheduling and specialist referrals.",
            "Sentence six warns about adverse reactions and when to seek emergency care.",
        ] * 3
        text = " ".join(sentences)
        ch = chunk_text(text, chunk_size=600, overlap=120)
        self.assertTrue(len(ch) >= 2, f"expected multiple chunks, got {len(ch)}: {ch}")
        # Overlap contract: the chunker rounds the 120-char overlap to the
        # last sentence boundary inside the tail (sentence-aware chunking),
        # so the next chunk must begin somewhere inside that 120-char tail
        # (never before it) and the tail's final sentence must be repeated
        # verbatim at the start of the next chunk.
        for i in range(len(ch) - 1):
            tail = ch[i][-120:]
            nxt = ch[i + 1]
            # (a) next chunk starts inside (or at) the tail, never earlier
            self.assertTrue(
                any(nxt.startswith(tail[j:]) for j in range(len(tail))),
                f"next chunk starts before the 120-char tail at {i}",
            )
            # (b) the last sentence fully inside the tail is repeated verbatim
            #     at the start of the next chunk (sentence-boundary overlap)
            last_sent_match = re.search(r"(?<=[.!?])\s", tail)
            if last_sent_match:
                last_sent_in_tail = tail[last_sent_match.end():]
                self.assertTrue(
                    nxt.startswith(last_sent_in_tail),
                    f"final sentence of tail not repeated at chunk {i+1}",
                )

    def test_short_content_rejected_text(self):
        # P1: sub-41-char content must be rejected with 400, not silently dropped.
        r = client.post("/documents/ingest", json={
            "doc_type": "Note", "title": "Tiny Note", "content": "hello.",
        })
        self.assertEqual(r.status_code, 400)
        self.assertIn("too short", r.json()["detail"].lower())

    def test_short_content_rejected_pdf(self):
        # P1: a PDF whose extracted text is too short must also be rejected.
        r = client.post(
            "/documents/upload",
            files={"file": ("tiny.pdf", make_tiny_pdf(), "application/pdf")},
            data={"title": "Tiny PDF Note"},
        )
        self.assertEqual(r.status_code, 400)
        self.assertIn("too short", r.json()["detail"].lower())


class TestConversationEndpoints(unittest.TestCase):
    def setUp(self):
        wipe_db()
        client.post("/documents/ingest", json={
            "doc_type": "Note", "title": "Conv Note",
            "content": "The standard adult paracetamol dose is 500 to 1000 mg "
                       "every 4 to 6 hours, not exceeding 4 grams per day. "
                       "Overdose can cause liver damage.",
        })

    def test_conversation_recorded_and_retrieved(self):
        sid = "sess_test"
        client.post("/query", json={"question": "paracetamol dose", "session_id": sid})
        r = client.get(f"/conversations/{sid}")
        self.assertEqual(r.status_code, 200)
        msgs = r.json()
        self.assertGreaterEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["kind"], "user")
        kinds = {m["kind"] for m in msgs}
        self.assertIn("assistant", kinds)

    def test_conversation_nonexistent(self):
        r = client.get("/conversations/no-such-session")
        self.assertEqual(r.status_code, 404)


from rag_pipeline import chunk_text  # noqa: E402


if __name__ == "__main__":
    unittest.main(verbosity=2)
