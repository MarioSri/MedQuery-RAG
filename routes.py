from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from models import (
    QueryRequest, QueryResponse, MetricsResponse, DocumentInfo, SourceInfo,
    IngestTextRequest, IngestResponse, DeleteResponse
)
from typing import List
import rag_pipeline

router = APIRouter()

MAX_PDF_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        result = rag_pipeline.rag_answer(req.question, top_k=req.top_k)
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
    """Ingest a document from raw text — chunk, embed, add to FAISS immediately."""
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="Document content cannot be empty")
    if not req.title.strip():
        raise HTTPException(status_code=400, detail="Document title is required")

    result = rag_pipeline.ingest_document(
        doc_type=req.doc_type.strip() or "General",
        title=req.title.strip(),
        content=req.content
    )
    return IngestResponse(**result)


@router.post("/documents/upload", response_model=IngestResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    doc_type: str = Form("Uploaded PDF"),
    title: str = Form("")
):
    """Upload a PDF file — extract text, chunk, embed, add to FAISS immediately."""
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Read and validate size
    pdf_bytes = await file.read()
    if len(pdf_bytes) > MAX_PDF_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(pdf_bytes) / 1024 / 1024:.1f} MB). Maximum is 10 MB."
        )

    # Extract text from PDF using PyMuPDF
    try:
        import fitz  # pymupdf
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        extracted_text = "\n\n".join(text_parts).strip()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to extract text from PDF: {str(e)}")

    if not extracted_text:
        raise HTTPException(status_code=422, detail="PDF contains no extractable text")

    # Use filename as title if not provided
    doc_title = title.strip() or file.filename.rsplit(".", 1)[0]

    result = rag_pipeline.ingest_document(
        doc_type=doc_type.strip() or "Uploaded PDF",
        title=doc_title,
        content=extracted_text
    )
    return IngestResponse(**result)


@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
def delete_document(doc_id: str):
    """Delete a document — remove chunks and rebuild FAISS index."""
    result = rag_pipeline.delete_document(doc_id)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["message"])
    return DeleteResponse(**result)
