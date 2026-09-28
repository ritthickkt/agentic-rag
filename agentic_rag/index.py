"""The BENCHMARK's vector index. Superseded for the served app by `store.py`.

This module backs `eval/` and the HotpotQA comparison in `scripts/run_eval.py`
only. The served application uses `agentic_rag.store.VectorStore` plus
`agentic_rag.retrieve.Retriever`, which add hybrid search, payload provenance, and
per-document filtering. Both are kept because the benchmark's numbers were produced
against this code path, and quietly re-pointing it at a different retriever would
invalidate every result already recorded in `results/`.

Local mode means a directory on disk and no server to run, which keeps the
project to `python scripts/…`. The tradeoff is that the storage directory takes a
single-process lock, so the harness must do its retrieval in one process — it
does, with threads.

`search()` returns `eval.types.Passage` objects. That direction of dependency is
deliberate and is the shape D3 asks for: systems know about the harness's data
types, and the harness knows nothing about systems.
"""

from __future__ import annotations

import threading
from pathlib import Path

from qdrant_client import QdrantClient, models

from agentic_rag.config import CONFIG
from agentic_rag.embed import Embedder
from eval.types import Document, Passage


class Retriever:
    def __init__(
        self,
        path: Path | None = None,
        collection: str | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.path = Path(path or CONFIG.eval_qdrant_path)
        self.collection = collection or CONFIG.eval_collection
        self.embedder = embedder or Embedder()
        self.client = QdrantClient(path=str(self.path))
        # The harness runs questions concurrently because LLM calls dominate the
        # wall clock. Qdrant's local mode and the sentence-transformers model make
        # no thread-safety promises, so retrieval is serialised. It costs nothing:
        # a search is ~10ms against ~1-3s per LLM call.
        self._lock = threading.Lock()

    # --- build ------------------------------------------------------------

    def build(self, docs: list[Document], batch_size: int = 256, show_progress: bool = True) -> int:
        """(Re)create the collection and index every document."""
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=self.embedder.dim, distance=models.Distance.COSINE
            ),
        )

        vectors = self.embedder.encode_documents([d.text for d in docs], show_progress=show_progress)
        for start in range(0, len(docs), batch_size):
            chunk = docs[start : start + batch_size]
            self.client.upsert(
                collection_name=self.collection,
                points=[
                    models.PointStruct(
                        id=start + i,
                        vector=vectors[start + i].tolist(),
                        # doc_id is the join key for every retrieval metric.
                        payload={"doc_id": d.doc_id, "text": d.text},
                    )
                    for i, d in enumerate(chunk)
                ],
            )
        return self.count()

    def count(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)

    # --- query ------------------------------------------------------------

    def search(self, query: str, k: int | None = None) -> list[Passage]:
        k = k or CONFIG.top_k
        with self._lock:
            vec = self.embedder.encode_query(query)
            hits = self.client.query_points(
                collection_name=self.collection, query=vec.tolist(), limit=k, with_payload=True
            ).points
        return [
            Passage(doc_id=h.payload["doc_id"], text=h.payload["text"], score=float(h.score))
            for h in hits
        ]

    def close(self) -> None:
        self.client.close()
