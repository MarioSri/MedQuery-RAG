import os
from dotenv import load_dotenv

load_dotenv()

# ── Server ────────────────────────────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# ── Database (Supabase or any PostgreSQL + pgvector instance) ─────────────────
# Supabase:  DATABASE_URL="postgresql://postgres:<password>@db.<project>.supabase.co:5432/postgres"
# Local:     DATABASE_URL="postgresql://medquery:medquery@localhost:5432/medquery"
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://medquery:medquery@localhost:5432/medquery")

# ── LLM provider interface (items 2/3 of the roadmap) ─────────────────────────
# Supported LLM providers: anthropic | openai | gemini | ollama | custom
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-3-5-haiku-latest")          # model id for the chosen provider
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CUSTOM_BASE_URL = os.getenv("CUSTOM_BASE_URL", "")                       # any OpenAI-compatible endpoint
CUSTOM_API_KEY = os.getenv("CUSTOM_API_KEY", "")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# ── Embedding provider interface ──────────────────────────────────────────────
# Supported embedding providers: sentence_transformers | openai | hf_api | custom
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")       # model id for the chosen provider
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))                   # MUST match the model
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")

# ── Retrieval tuning (items 5–7 of the roadmap) ──────────────────────────────
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))                 # final chunks sent to LLM
RETRIEVAL_CANDIDATE_K = int(os.getenv("RETRIEVAL_CANDIDATE_K", "15"))    # candidates before reranking
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))  # cosine; below this -> insufficient evidence
THRESHOLD_ENABLED = os.getenv("THRESHOLD_ENABLED", "true").lower() in ("1", "true", "yes")
HYBRID_WEIGHT_VECTOR = float(os.getenv("HYBRID_WEIGHT_VECTOR", "0.7"))   # vector vs keyword (1.0 - x = keyword)

# ── Reranking (item 6 of the roadmap) ────────────────────────────────────────
# reranker: none | cross_encoder | bm25_rerank (simple)
RERANKER = os.getenv("RERANKER", "cross_encoder")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
