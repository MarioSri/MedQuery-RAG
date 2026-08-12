from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from models import (
    QueryRequest, QueryResponse, MetricsResponse, DocumentInfo, SourceInfo,
    IngestTextRequest, IngestResponse, DeleteResponse
)
from typing import List
import rag_pipeline
import db_store

router = APIRouter()

MAX_PDF_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        session_id = req.session_id or "default"
        rag_pipeline.add_conversation_message(session_id, "user", req.question)
        conversation = rag_pipeline.get_conversation(session_id)
        result = rag_pipeline.rag_answer(req.question, top_k=req.top_k, conversation=conversation)
        rag_pipeline.add_conversation_message(session_id, "assistant", result["answer"])
        return QueryResponse(
            answer=result["answer"],
            sources=[SourceInfo(**s) for s in result["sources"]],
            retrieved_chunks=result["retrieved_chunks"],
            similarity_scores=result["similarity_scores"],
            latency_ms=result["latency_ms"],
            tokens_used=result["tokens_used"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=MetricsResponse)
def metrics():
    return rag_pipeline.get_metrics()


@router.get("/documents", response_model=List[DocumentInfo])
def documents():
    return rag_pipeline.get_documents()


# ── Live ingestion endpoints ─────────────────────────────────────────────────

@router.post("/documents/ingest", response_model=IngestResponse)
def ingest_text(req: IngestTextRequest):
    """Ingest a document from raw text — chunk, embed, store in PostgreSQL + pgvector."""
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="Document content cannot be empty")
    if not req.title.strip():
        raise HTTPException(status_code=400, detail="Document title is required")

    try:
        result = rag_pipeline.ingest_document(
            doc_type=req.doc_type.strip() or "General",
            title=req.title.strip(),
            content=req.content,
            source="manual",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return IngestResponse(**result)


@router.post("/documents/upload", response_model=IngestResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    doc_type: str = Form("Uploaded PDF"),
    title: str = Form("")
):
    """Upload a PDF — per-page extraction with page_number metadata per chunk."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    pdf_bytes = await file.read()
    if len(pdf_bytes) > MAX_PDF_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(pdf_bytes) / 1024 / 1024:.1f} MB). Maximum is 10 MB."
        )

    try:
        import fitz  # pymupdf
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_texts = []
        for page in doc:
            page_texts.append(page.get_text())
        doc.close()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to extract text from PDF: {str(e)}")

    if not any(page_texts):
        raise HTTPException(status_code=422, detail="PDF contains no extractable text")

    # Per-page chunks with page_number metadata (roadmap item: metadata + page tracking)
    chunks_with_pages = []
    for page_num, ptext in enumerate(page_texts, 1):
        text = ptext.strip()
        if not text:
            continue
        page_chunks = rag_pipeline.chunk_text(text)
        for pc in page_chunks:
            chunks_with_pages.append((pc, page_num))

    if not chunks_with_pages:
        raise HTTPException(status_code=400, detail="Document is too short to be indexed.")

    doc_title = title.strip() or file.filename.rsplit(".", 1)[0]

    try:
        result = rag_pipeline.ingest_document(
            doc_type=doc_type.strip() or "Uploaded PDF",
            title=doc_title,
            content="\n\n".join(c for c, _ in chunks_with_pages),
            source="pdf",
            chunks_with_pages=chunks_with_pages,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return IngestResponse(**result)


@router.get("/conversations/{session_id}")
def conversation_history(session_id: str):
    """Retrieve recorded conversation memory for a session."""
    msgs = rag_pipeline.get_conversation(session_id)
    if not msgs:
        raise HTTPException(status_code=404,
                            detail=f"No conversation found for session {session_id}")
    return [
        {"kind": "user" if m["role"] == "user" else "assistant",
         "content": m["content"],
         "role": m["role"]}
        for m in msgs
    ]


@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
def delete_document(doc_id: str):
    """Delete a document — remove chunks and vectors from the database."""
    result = rag_pipeline.delete_document(doc_id)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["message"])
    return DeleteResponse(**result)
