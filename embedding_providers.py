"""Provider-independent embedding interface (roadmap item 3).

EmbeddingProvider is the abstract contract; the pipeline calls
``embedder.encode(texts)`` and never imports a vendor SDK directly.

Supported providers:
  - sentence_transformers: local HF model via sentence-transformers
  - openai:                text-embedding-3-small/large via the openai SDK
  - hf_api:                HF Inference API
  - custom:                any OpenAI-compatible /embeddings endpoint
"""
import numpy as np

import config


class EmbeddingProvider:
    """Abstract embedding interface."""

    def encode(self, texts, batch_size: int = 32, show_progress_bar: bool = False) -> np.ndarray:
        """Return a (n, dim) float32 array of NORMALIZED embeddings."""
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        raise NotImplementedError

    @property
    def model_id(self) -> str:
        raise NotImplementedError

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(arr, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return arr / norm


class SentenceTransformersProvider(EmbeddingProvider):
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(config.EMBEDDING_MODEL)
        self._dim = (
            self._model.get_embedding_dimension()
            if hasattr(self._model, "get_embedding_dimension")
            else self._model.get_sentence_embedding_dimension()
        )
        if config.EMBEDDING_DIM and self._dim != config.EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dim mismatch: model produces {self._dim} but config "
                f"declares {config.EMBEDDING_DIM}. Update EMBEDDING_DIM in config."
            )

    def encode(self, texts, batch_size=32, show_progress_bar=False):
        arr = np.array(self._model.encode(
            texts, batch_size=batch_size, show_progress_bar=show_progress_bar,
            normalize_embeddings=False,
        ), dtype="float32")
        return self._normalize(arr)

    @property
    def dimension(self):
        return self._dim

    @property
    def model_id(self):
        return f"sentence_transformers/{config.EMBEDDING_MODEL}"


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self):
        from openai import OpenAI
        self._client = OpenAI(api_key=config.OPENAI_API_KEY)

    def encode(self, texts, batch_size=32, show_progress_bar=False):
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = self._client.embeddings.create(
                model=config.EMBEDDING_MODEL, input=batch,
                encoding_format="float",
            )
            out.extend(np.array(item.embedding, dtype="float32") for item in resp.data)
        return self._normalize(np.array(out, dtype="float32"))

    @property
    def dimension(self):
        return config.EMBEDDING_DIM

    @property
    def model_id(self):
        return f"openai/{config.EMBEDDING_MODEL}"


class HFInferenceEmbeddingProvider(EmbeddingProvider):
    def __init__(self):
        import requests
        self._requests = requests
        if not config.HF_API_TOKEN:
            raise ValueError("HF_API_TOKEN is required for the hf_api embedding provider")

    def encode(self, texts, batch_size=32, show_progress_bar=False):
        url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{config.EMBEDDING_MODEL}"
        headers = {"Authorization": f"Bearer {config.HF_API_TOKEN}"}
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = self._requests.post(url, headers=headers, json={"inputs": batch})
            resp.raise_for_status()
            out.extend(np.array(v, dtype="float32") for v in resp.json())
        return self._normalize(np.array(out, dtype="float32"))

    @property
    def dimension(self):
        return config.EMBEDDING_DIM

    @property
    def model_id(self):
        return f"hf_api/{config.EMBEDDING_MODEL}"


class CustomEmbeddingProvider(EmbeddingProvider):
    """Any OpenAI-compatible /embeddings endpoint."""

    def __init__(self):
        from openai import OpenAI
        self._client = OpenAI(base_url=config.CUSTOM_BASE_URL, api_key=config.CUSTOM_API_KEY)

    def encode(self, texts, batch_size=32, show_progress_bar=False):
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = self._client.embeddings.create(
                model=config.EMBEDDING_MODEL, input=batch, encoding_format="float",
            )
            out.extend(np.array(item.embedding, dtype="float32") for item in resp.data)
        return self._normalize(np.array(out, dtype="float32"))

    @property
    def dimension(self):
        return config.EMBEDDING_DIM

    @property
    def model_id(self):
        return f"custom/{config.EMBEDDING_MODEL}"


PROVIDERS = {
    "sentence_transformers": SentenceTransformersProvider,
    "openai": OpenAIEmbeddingProvider,
    "hf_api": HFInferenceEmbeddingProvider,
    "custom": CustomEmbeddingProvider,
}


def get_embedding_provider() -> EmbeddingProvider:
    name = config.EMBEDDING_PROVIDER.lower()
    if name not in PROVIDERS:
        raise ValueError(f"Unknown embedding provider: {name}. Choose from {sorted(PROVIDERS)}")
    return PROVIDERS[name]()
