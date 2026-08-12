from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    session_id: Optional[str] = "default"


class SourceInfo(BaseModel):
    doc_id: str
    doc_type: str
    doc_title: str
    chunk_index: int
    page_number: Optional[int] = None
    source: Optional[str] = None
    rank: int
    similarity: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    retrieved_chunks: List[str]
    similarity_scores: List[float]
    latency_ms: float
    tokens_used: int


class MetricsResponse(BaseModel):
    total_queries: int
    documents_indexed: int
    chunks_indexed: int
    avg_similarity_score: float
    avg_latency_ms: float
    embedding_model: str
    embedding_dim: int
    vector_db_type: str
    retrieval_mode: str
    reranker: str
    similarity_threshold: Optional[float]
    llm_provider: str


class DocumentInfo(BaseModel):
    id: str
    type: str
    title: str
    source: Optional[str] = None
    status: Optional[str] = "completed"
    chunk_count: int


# ── Live ingestion models ────────────────────────────────────────────────────

class IngestTextRequest(BaseModel):
    doc_type: str
    title: str
    content: str


class IngestResponse(BaseModel):
    doc_id: str
    chunks_added: int
    total_chunks: int
    total_documents: int


class DeleteResponse(BaseModel):
    success: bool
    message: str
    chunks_removed: Optional[int] = None
