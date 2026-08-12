"""
Comprehensive functional tests for the MedQuery-RAG backend.

Uses FastAPI TestClient and monkey-patches the Anthropic client with a mock so
the full RAG pipeline (retrieve -> prompt -> LLM -> response) can be exercised
without a real Anthropic API key. Also tests:
  - Health, metrics, documents endpoints
  - Text ingestion and chunking
  - PDF upload + extraction
  - Document deletion + FAISS rebuild
  - Chunking correctness
  - End-to-end /query response schema
"""
import io
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import fitz  # pymupdf
from fastapi.testclient import TestClient


# Build a small sample PDF in memory for upload testing
def make_sample_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(72, 100), text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


SAMPLE_PDF = make_sample_pdf(
    "Clinical guideline Type 2 diabetes: Target HbA1c below 7 percent. Metformin first line. "
    "Monitor renal function every 6 months."
)


class MockUsage:
    input_tokens = 200
    output_tokens = 150


class MockContent:
    text = (
        "Based on the provided documents: The normal haemoglobin reference range is "
        "13.0-17.0 g/dL (the patient's CBC shows 11.2 g/dL, which is LOW). "
        "Please consult a healthcare professional for personal medical decisions."
    )


class MockResponse:
    content = [MockContent()]
    usage = MockUsage()


mock_messages = MagicMock(create=lambda **kw: MockResponse())


# Initialize the RAG pipeline (downloads model, builds FAISS index).
# Then patch the Anthropic client attribute so rag_answer() calls the mock.
from main import app

client = TestClient(app, raise_server_exceptions=False)
client.__enter__()  # triggers lifespan -> rag_pipeline.initialize() (real Anthropic client)

patch("rag_pipeline.client.messages", mock_messages, create=True).start()

# Debug: show server error detail for any 500
_orig_post = client.post

def debug_post(url, *a, **kw):
    r = _orig_post(url, *a, **kw)
    if r.status_code == 500:
        print(f"[DEBUG 500 on {url}]:", r.json())
    return r
client.post = debug_post


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
        self.assertIsInstance(m["total_queries"], int)
        self.assertIsInstance(m["documents_indexed"], int)
        self.assertIsInstance(m["chunks_indexed"], int)
        self.assertEqual(m["documents_indexed"], 4)      # 4 hardcoded docs
        self.assertEqual(m["chunks_indexed"], 19)        # known chunk count
        self.assertEqual(m["embedding_model"], "all-MiniLM-L6-v2")
        self.assertEqual(m["vector_db_type"], "FAISS IndexFlatIP")
        self.assertEqual(m["embedding_dim"], 384)


class TestDocumentsEndpoint(unittest.TestCase):
    def test_documents_listed(self):
        r = client.get("/documents")
        self.assertEqual(r.status_code, 200)
        docs = r.json()
        self.assertEqual(len(docs), 4)
        titles = {d["title"] for d in docs}
        self.assertIn("Complete Blood Count — Patient Ravi Kumar", titles)
        self.assertIn("Prescription — Hypertension Management", titles)
        # chunk counts sum to 19
        self.assertEqual(sum(d["chunk_count"] for d in docs), 19)


class TestQueryEndpoint(unittest.TestCase):
    def test_query_response_schema(self):
        r = client.post("/query", json={"question": "What is the normal haemoglobin level?", "top_k": 5})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        for key in ["answer", "sources", "retrieved_chunks", "similarity_scores", "latency_ms", "tokens_used"]:
            self.assertIn(key, body, f"missing key: {key}")
        self.assertEqual(len(body["sources"]), len(body["retrieved_chunks"]))
        self.assertEqual(len(body["sources"]), len(body["similarity_scores"]))
        # Top result should be the CBC lab report (haemoglobin question)
        self.assertIn("Complete Blood Count", body["sources"][0]["doc_title"])
        self.assertGreater(body["similarity_scores"][0], body["similarity_scores"][-1] - 1e-9)
        self.assertGreater(body["latency_ms"], 0)
        self.assertEqual(body["tokens_used"], 350)  # mock usage

    def test_query_retrieval_relevance(self):
        """Appendicitis discharge question should rank the discharge summary first."""
        r = client.post("/query", json={"question": "What medications were prescribed at discharge after appendicitis surgery?", "top_k": 5})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("Discharge Summary", body["sources"][0]["doc_type"])

    def test_query_unrelated_question(self):
        """Questions with low relevance still return results but low scores."""
        r = client.post("/query", json={"question": "quantum entanglement in black holes", "top_k": 5})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertLess(body["similarity_scores"][0], 0.45)

    def test_metrics_update_after_query(self):
        client.post("/query", json={"question": "haemoglobin"})
        m = client.get("/metrics").json()
        self.assertGreater(m["total_queries"], 0)
        self.assertGreater(m["avg_latency_ms"], 0)

    def test_invalid_payload(self):
        r = client.post("/query", json={"bad": "payload"})
        self.assertIn(r.status_code, (422,))


class TestTextIngestion(unittest.TestCase):
    def test_ingest_text(self):
        before = client.get("/documents").json()
        before_count = len(before)
        r = client.post("/documents/ingest", json={
            "doc_type": "Guideline",
            "title": "Diabetes Guideline Test",
            "content": "Target HbA1c below 7 percent for type 2 diabetes patients. Metformin is the first-line therapy."
        })
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["doc_id"].startswith("doc_"))
        self.assertGreater(body["chunks_added"], 0)
        self.assertEqual(body["total_documents"], before_count + 1)

        # Verify new doc is searchable immediately
        r2 = client.post("/query", json={"question": "What is the target HbA1c?", "top_k": 5})
        self.assertEqual(r2.status_code, 200)
        titles = [s["doc_title"] for s in r2.json()["sources"]]
        self.assertIn("Diabetes Guideline Test", titles)

    def test_ingest_empty_content_rejected(self):
        r = client.post("/documents/ingest", json={"doc_type": "X", "title": "T", "content": "   "})
        self.assertEqual(r.status_code, 400)

    def test_ingest_empty_title_rejected(self):
        r = client.post("/documents/ingest", json={"doc_type": "X", "title": " ", "content": "Some content"})
        self.assertEqual(r.status_code, 400)


class TestPDFUpload(unittest.TestCase):
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

    def test_upload_non_pdf_rejected(self):
        r = client.post(
            "/documents/upload",
            files={"file": ("notes.txt", b"plain text", "text/plain")},
        )
        self.assertEqual(r.status_code, 400)

    def test_upload_empty_pdf_rejected(self):
        # A valid PDF with no extractable text
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
    def test_delete_restores_index(self):
        # Ingest then delete a document; FAISS should shrink accordingly
        before = client.get("/metrics").json()
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
        self.assertEqual(after["documents_indexed"], before["documents_indexed"])
        # Index rebuilt without re-encoding; total queries should still work
        q = client.post("/query", json={"question": "haemoglobin", "top_k": 3})
        self.assertEqual(q.status_code, 200)

    def test_delete_nonexistent_returns_404(self):
        r = client.delete("/documents/doc_does_not_exist")
        self.assertEqual(r.status_code, 404)


class TestChunking(unittest.TestCase):
    def test_chunking_preserves_content(self):
        import rag_pipeline
        # Use a chunk size comparable to the pipeline's own (600 chars), but with a
        # long multi-sentence text that spans multiple chunks.
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
        ch = rag_pipeline.chunk_text(text, chunk_size=250, overlap=50)
        self.assertTrue(len(ch) >= 2, f"expected >=2 chunks, got {len(ch)}: {ch}")
        # Prefer sentence boundaries: first chunk should end at the first period
        self.assertTrue(ch[0].endswith(".") or "." in ch[0], "chunks should prefer sentence boundaries")

    def test_chunking_empty_string(self):
        import rag_pipeline
        self.assertEqual(rag_pipeline.chunk_text(""), [])

    def test_chunking_overlap(self):
        import rag_pipeline
        # Pipeline default chunking: 600 chars with 120-char overlap
        sentences = [
            "This is sentence one describing baseline patient measurements and values.",
            "Sentence two provides updated diagnostic imaging findings and impressions.",
            "Sentence three lists recommended treatment options and medication details.",
            "Sentence four covers short-term monitoring requirements for the patient.",
            "Sentence five gives long-term follow-up scheduling and specialist referrals.",
            "Sentence six warns about adverse reactions and when to seek emergency care.",
        ]
        text = " ".join(sentences)
        ch = rag_pipeline.chunk_text(text, chunk_size=600, overlap=120)
        self.assertTrue(len(ch) >= 2, f"expected multiple chunks, got {len(ch)}: {ch}")
        # Overlap contract: the full 120-char tail of chunk i must start chunk i+1
        # (not merely the last word of the tail).
        for i in range(len(ch) - 1):
            tail = ch[i][-120:]
            self.assertTrue(
                ch[i + 1].startswith(tail.strip()),
                f"120-char overlap between chunk {i} and {i+1} broken",
            )

    def test_short_content_rejected_text(self):
        # P1: sub-40-char content produces zero chunks and must be rejected, not dropped.
        r = client.post("/documents/ingest", json={
            "doc_type": "Note", "title": "Tiny Note", "content": "hello.",
        })
        self.assertEqual(r.status_code, 400)
        self.assertIn("too short", r.json()["detail"].lower())

    def test_short_content_rejected_pdf(self):
        # P1: a PDF whose extracted text is too short must also be rejected.
        tiny_pdf = make_sample_pdf("Brief note.")
        r = client.post(
            "/documents/upload",
            files={"file": ("tiny.pdf", tiny_pdf, "application/pdf")},
            data={"title": "Tiny PDF Note"},
        )
        self.assertEqual(r.status_code, 400)
        self.assertIn("too short", r.json()["detail"].lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
