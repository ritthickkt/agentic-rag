"""The vector store: Qdrant, embedded on disk by default.

Embedded mode means a directory and no server to run, which keeps `uvicorn` the
only process you start. The cost is a hard one and worth stating: **the storage
directory takes an exclusive lock, so exactly one process may hold it.** Run the
API and a CLI ingest at the same time and the second one fails to open the store.
Set `QDRANT_URL` to point at a real Qdrant (`docker run -p 6333:6333 qdrant/qdrant`)
and that limit goes away. See DECISIONS D21.

Two guarantees this module exists to provide:

**Vector-space identity.** A collection is stamped with the `vector_id` of the
embedder that built it. Querying bge-small vectors with a Voyage query embedding
returns confident, plausible, wrong results — there is no error, just silently bad
retrieval. `ensure_collection` refuses the mismatch instead.

**Deletion actually deletes.** A user who removes a document expects its content to
stop appearing in answers. That means a filtered delete on `doc_id` with an index
on the field, not a tombstone the retriever is trusted to remember to skip.
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
from qdrant_client import QdrantClient, models

from agentic_rag.config import CONFIG
from agentic_rag.models import Chunk

# Namespace for turning our string chunk_ids into the UUIDs Qdrant requires as
# point ids. UUID5 is deterministic, so the same chunk_id always maps to the same
# point and a re-index is an overwrite rather than a duplicate.
_POINT_NAMESPACE = uuid.UUID("6f1b6c4e-6a3e-4f1a-9a6e-1f2c3d4e5f60")

# Stored as a payload on a single reserved point, because Qdrant collections have
# no user metadata slot of their own.
_META_POINT_ID = "00000000-0000-0000-0000-000000000001"


class VectorSpaceMismatch(Exception):
    """The collection was built by a different embedding model."""


def point_id_for(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


class VectorStore:
    def __init__(
        self,
        path: Path | None = None,
        url: str | None = None,
        collection: str | None = None,
    ) -> None:
        self.collection = collection or CONFIG.collection
        url = CONFIG.qdrant_url if url is None else url
        if url:
            self.client = QdrantClient(url=url)
            self.location = url
        else:
            path = Path(path or CONFIG.qdrant_path)
            path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))
            self.location = str(path)
        # Embedded Qdrant makes no thread-safety promises and FastAPI serves on a
        # thread pool. A search is ~10ms against seconds per LLM call, so
        # serialising costs nothing measurable.
        self._lock = threading.Lock()

    # --- lifecycle --------------------------------------------------------

    def ensure_collection(self, dim: int, vector_id: str) -> None:
        """Create the collection, or verify the existing one matches this embedder."""
        with self._lock:
            if self.client.collection_exists(self.collection):
                existing = self._read_vector_id()
                if existing and existing != vector_id:
                    raise VectorSpaceMismatch(
                        f"Collection {self.collection!r} was built with {existing!r} but this "
                        f"process embeds with {vector_id!r}. Querying across embedding models "
                        f"returns plausible nonsense. Re-index with "
                        f"`python scripts/rag.py reindex`, or point COLLECTION at a new name."
                    )
                if not existing:
                    self._write_vector_id(vector_id)
                return

            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )
            # Without an index on doc_id, filtered search and filtered delete
            # degrade to a full scan as the corpus grows. Embedded Qdrant ignores
            # payload indexes (it scans regardless), so we only ask for one when
            # talking to a server, rather than emit a warning on every startup.
            if CONFIG.qdrant_url:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name="doc_id",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            self._write_vector_id(vector_id, dim=dim)

    def _write_vector_id(self, vector_id: str, dim: int | None = None) -> None:
        info = self.client.get_collection(self.collection)
        size = dim or info.config.params.vectors.size
        self.client.upsert(
            collection_name=self.collection,
            points=[
                models.PointStruct(
                    id=_META_POINT_ID,
                    vector=[0.0] * size,
                    payload={"__meta__": True, "vector_id": vector_id, "doc_id": "__meta__"},
                )
            ],
        )

    def _read_vector_id(self) -> str:
        recs = self.client.retrieve(self.collection, ids=[_META_POINT_ID], with_payload=True)
        if not recs:
            return ""
        return (recs[0].payload or {}).get("vector_id", "")

    def drop(self) -> None:
        with self._lock:
            if self.client.collection_exists(self.collection):
                self.client.delete_collection(self.collection)

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:  # noqa: BLE001 — closing a store must never raise on shutdown
            pass

    # --- writes -----------------------------------------------------------

    def upsert_chunks(self, chunks: Sequence[Chunk], vectors: np.ndarray,
                      batch_size: int = 256) -> int:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        with self._lock:
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                self.client.upsert(
                    collection_name=self.collection,
                    points=[
                        models.PointStruct(
                            id=point_id_for(c.chunk_id),
                            vector=vectors[start + i].tolist(),
                            payload=c.to_payload(),
                        )
                        for i, c in enumerate(batch)
                    ],
                )
        return len(chunks)

    def delete_doc(self, doc_id: str) -> None:
        with self._lock:
            self.client.delete(
                collection_name=self.collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
                    )
                ),
            )

    # --- reads ------------------------------------------------------------

    def search(
        self, vector: np.ndarray, k: int | None = None, doc_ids: Sequence[str] | None = None
    ) -> list[tuple[Chunk, float]]:
        k = k or CONFIG.candidate_k
        flt = models.Filter(
            must_not=[models.IsEmptyCondition(is_empty=models.PayloadField(key="text"))],
            **(
                {"must": [models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(doc_ids)))]}
                if doc_ids
                else {}
            ),
        )
        with self._lock:
            hits = self.client.query_points(
                collection_name=self.collection,
                query=vector.tolist(),
                limit=k,
                with_payload=True,
                query_filter=flt,
            ).points
        out: list[tuple[Chunk, float]] = []
        for h in hits:
            payload = h.payload or {}
            if payload.get("__meta__"):
                continue
            out.append((Chunk.from_payload(payload), float(h.score)))
        return out

    def iter_chunks(self, doc_ids: Sequence[str] | None = None,
                    batch_size: int = 512) -> Iterator[Chunk]:
        """Full scan, used to (re)build the in-memory lexical index."""
        flt = (
            models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(doc_ids)))]
            )
            if doc_ids
            else None
        )
        offset = None
        while True:
            with self._lock:
                points, offset = self.client.scroll(
                    collection_name=self.collection,
                    scroll_filter=flt,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
            for p in points:
                payload = p.payload or {}
                if payload.get("__meta__") or not payload.get("text"):
                    continue
                yield Chunk.from_payload(payload)
            if offset is None:
                break

    def get_chunks(self, chunk_ids: Sequence[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        with self._lock:
            recs = self.client.retrieve(
                self.collection, ids=[point_id_for(c) for c in chunk_ids], with_payload=True
            )
        out: dict[str, Chunk] = {}
        for r in recs:
            payload = r.payload or {}
            if payload.get("text"):
                chunk = Chunk.from_payload(payload)
                out[chunk.chunk_id] = chunk
        return out

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        with self._lock:
            total = int(self.client.count(self.collection, exact=True).count)
        return max(total - 1, 0)  # exclude the reserved metadata point

    def info(self) -> dict[str, Any]:
        return {
            "collection": self.collection,
            "location": self.location,
            "exists": self.client.collection_exists(self.collection),
            "n_chunks": self.count(),
            "vector_id": self._read_vector_id() if self.client.collection_exists(self.collection) else "",
        }
