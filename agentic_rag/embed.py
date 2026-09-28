"""Embeddings behind a swappable provider, and the one asymmetry that is easy to
get wrong.

**The asymmetry.** `bge-small-en-v1.5` is trained *asymmetrically*: passages are
embedded bare, but a query must be prefixed with an instruction. Skip the prefix
and nothing breaks loudly — retrieval just gets quietly worse, which is the most
expensive kind of bug in a retrieval system. So the prefix lives here, applied by
`encode_queries()` and never by `encode_documents()`, and the two are separate
methods specifically so a caller cannot use the wrong one by accident.

**Why a provider protocol.** Local embeddings are free, offline, and private;
hosted ones (Voyage, OpenAI) retrieve better on long or technical text. That is a
tradeoff to *measure*, not to argue about — and this project happens to own an
eval harness, so `EMBED_PROVIDER` is a config flag and the benchmark decides.
`vector_id` is what makes that safe: it is stamped on the collection, so switching
provider cannot silently query one model's vectors with another model's embedding.

Vectors are L2-normalised, so cosine similarity and dot product agree.
"""

from __future__ import annotations

import os
import threading
from typing import Protocol, runtime_checkable

import numpy as np

from agentic_rag.config import CONFIG

# Prescribed by the BAAI/bge model card for retrieval.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@runtime_checkable
class EmbeddingProvider(Protocol):
    """What the store and retriever are allowed to assume about embeddings."""

    model_name: str
    dim: int

    @property
    def vector_id(self) -> str:
        """Stable identity of the vector space, stored alongside the vectors.

        Two providers, or two models from one provider, produce incomparable
        vectors. Recording this lets the store refuse a mismatch instead of
        returning confident nonsense.
        """
        ...

    def encode_documents(self, texts: list[str], show_progress: bool = False) -> np.ndarray: ...

    def encode_queries(self, texts: list[str], show_progress: bool = False) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


class LocalEmbedder:
    """sentence-transformers on this machine. Free, offline, no rate limits.

    **Calls are serialised, and that is not optional.** A `SentenceTransformer` makes
    no thread-safety promises, and calling one from several threads at once does not
    raise — the interpreter segfaults. Both callers here are multi-threaded: the API
    serves requests on a thread pool, and the benchmark harness runs questions
    concurrently. The lock lives on the embedder rather than at each call site so a
    new caller cannot forget it.

    The cost is nil in practice: encoding one query takes single-digit milliseconds
    against seconds per LLM call, so the lock is never the bottleneck.
    """

    def __init__(self, model_name: str | None = None, batch_size: int = 64) -> None:
        from sentence_transformers import SentenceTransformer  # slow import, keep it lazy

        self.model_name = model_name or CONFIG.embed_model
        self.batch_size = batch_size
        self.model = SentenceTransformer(self.model_name)
        self._lock = threading.Lock()
        # Renamed in sentence-transformers 6.x; keep working on both.
        get_dim = getattr(self.model, "get_embedding_dimension", None) or \
            self.model.get_sentence_embedding_dimension
        self.dim = int(get_dim())
        # Only the bge family wants the prefix. Applying it to a model that was
        # not trained with it is not neutral — it is noise prepended to the query.
        self._query_prefix = BGE_QUERY_PREFIX if "bge" in self.model_name.lower() else ""

    @property
    def vector_id(self) -> str:
        return f"local:{self.model_name}:{self.dim}"

    def _encode(self, texts: list[str], show_progress: bool) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        with self._lock:
            return self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )

    def encode_documents(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """No prefix — passages are embedded as-is."""
        return self._encode(texts, show_progress)

    def encode_queries(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Prefixed — this is the half that must not be skipped."""
        return self._encode([self._query_prefix + t for t in texts], show_progress)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_queries([text])[0]


class VoyageEmbedder:
    """Voyage AI embeddings. Better on long/technical text; costs per token.

    Voyage has the same asymmetry as bge but expresses it as an `input_type`
    parameter rather than a text prefix — same trap, different spelling, and the
    same reason documents and queries get separate methods here.
    """

    def __init__(self, model_name: str | None = None, batch_size: int = 128) -> None:
        import voyageai  # optional dependency: pip install voyageai

        key = os.environ.get("VOYAGE_API_KEY", "")
        if not key:
            raise RuntimeError("EMBED_PROVIDER=voyage but VOYAGE_API_KEY is not set.")
        self.model_name = model_name or os.environ.get("VOYAGE_MODEL", "voyage-3")
        self.batch_size = batch_size
        self.client = voyageai.Client(api_key=key)
        self.dim = int(os.environ.get("VOYAGE_DIM", "1024"))

    @property
    def vector_id(self) -> str:
        return f"voyage:{self.model_name}:{self.dim}"

    def _encode(self, texts: list[str], input_type: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            resp = self.client.embed(batch, model=self.model_name, input_type=input_type)
            out.extend(resp.embeddings)
        arr = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-12)

    def encode_documents(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        return self._encode(texts, "document")

    def encode_queries(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        return self._encode(texts, "query")

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_queries([text])[0]


_PROVIDERS = {"local": LocalEmbedder, "voyage": VoyageEmbedder}


def build_embedder(provider: str | None = None, model_name: str | None = None) -> EmbeddingProvider:
    """One place that maps a config string to a provider."""
    name = (provider or CONFIG.embed_provider).strip().lower()
    if name not in _PROVIDERS:
        raise ValueError(f"unknown EMBED_PROVIDER {name!r}; choose from {sorted(_PROVIDERS)}")
    return _PROVIDERS[name](model_name)


# Backwards-compatible alias: the eval scripts were written against `Embedder`.
Embedder = LocalEmbedder
