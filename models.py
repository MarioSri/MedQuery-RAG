from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class SourceInfo(BaseModel):
    doc_type: str
    doc_title: str
    similarity_score: float


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
    vector_db_type: str
    embedding_dim: int


class DocumentInfo(BaseModel):
    id: str
    type: str
    title: str
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
