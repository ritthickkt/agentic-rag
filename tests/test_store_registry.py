"""The two persistence layers, and the guarantees they exist to provide."""

from __future__ import annotations

import numpy as np
import pytest

from agentic_rag.models import Chunk, DocumentRecord
from agentic_rag.registry import Registry
from agentic_rag.store import VectorSpaceMismatch, VectorStore
from tests.conftest import FakeEmbedder


def chunks(n: int, doc_id: str = "doc1") -> list[Chunk]:
    return [
        Chunk(chunk_id=f"{doc_id}-{i}", doc_id=doc_id, text=f"passage number {i}",
              ordinal=i, page=i + 1, filename="f.pdf")
        for i in range(n)
    ]


class TestVectorStore:
    def test_roundtrip_preserves_provenance(self, store):
        """If page/heading are lost here, every citation downstream is a guess."""
        emb = FakeEmbedder()
        store.ensure_collection(emb.dim, emb.vector_id)
        cs = chunks(3)
        store.upsert_chunks(cs, emb.encode_documents([c.text for c in cs]))
        hits = store.search(emb.encode_query("passage number 1"), k=3)
        got = {c.chunk_id: c for c, _ in hits}
        assert got["doc1-1"].page == 2
        assert got["doc1-1"].filename == "f.pdf"

    def test_reupsert_is_idempotent(self, store):
        emb = FakeEmbedder()
        store.ensure_collection(emb.dim, emb.vector_id)
        cs = chunks(4)
        vecs = emb.encode_documents([c.text for c in cs])
        store.upsert_chunks(cs, vecs)
        store.upsert_chunks(cs, vecs)
        assert store.count() == 4      # deterministic ids => overwrite, not duplicate

    def test_delete_doc_removes_only_that_doc(self, store):
        emb = FakeEmbedder()
        store.ensure_collection(emb.dim, emb.vector_id)
        a, b = chunks(2, "docA"), chunks(3, "docB")
        store.upsert_chunks(a + b, emb.encode_documents([c.text for c in a + b]))
        store.delete_doc("docA")
        assert store.count() == 3
        assert {c.doc_id for c in store.iter_chunks()} == {"docB"}

    def test_doc_id_filter_scopes_search(self, store):
        emb = FakeEmbedder()
        store.ensure_collection(emb.dim, emb.vector_id)
        a, b = chunks(3, "docA"), chunks(3, "docB")
        store.upsert_chunks(a + b, emb.encode_documents([c.text for c in a + b]))
        hits = store.search(emb.encode_query("passage number 0"), k=10, doc_ids=["docB"])
        assert {c.doc_id for c, _ in hits} == {"docB"}

    def test_mismatched_embedder_is_refused(self, store):
        """Querying one model's vectors with another's returns confident nonsense."""
        store.ensure_collection(16, "fake:a:16")
        with pytest.raises(VectorSpaceMismatch, match="rag.py reindex"):
            store.ensure_collection(16, "fake:b:16")

    def test_metadata_point_is_not_a_search_result(self, store):
        emb = FakeEmbedder()
        store.ensure_collection(emb.dim, emb.vector_id)
        cs = chunks(2)
        store.upsert_chunks(cs, emb.encode_documents([c.text for c in cs]))
        assert store.count() == 2                                  # excludes the meta point
        assert all(c.chunk_id.startswith("doc1") for c in store.iter_chunks())

    def test_vector_count_mismatch_raises(self, store):
        emb = FakeEmbedder()
        store.ensure_collection(emb.dim, emb.vector_id)
        with pytest.raises(ValueError, match="chunks but"):
            store.upsert_chunks(chunks(3), np.zeros((2, emb.dim), dtype=np.float32))


class TestRegistry:
    def test_dedup_is_scoped_to_the_embedding_model(self, cfg):
        """Same bytes under a different embedder is a new index entry, not a dupe."""
        r = Registry(cfg.registry_path)
        rec = DocumentRecord(doc_id="a", filename="f.pdf", source_type="pdf",
                             content_hash="h1", vector_id="v1", status="ready")
        r.upsert(rec)
        assert r.find_by_hash("h1", "v1") is not None
        assert r.find_by_hash("h1", "v2") is None

    def test_only_ready_documents_dedupe(self, cfg):
        """A failed upload must not block re-uploading the fixed file."""
        r = Registry(cfg.registry_path)
        r.upsert(DocumentRecord(doc_id="a", filename="f.pdf", source_type="pdf",
                                content_hash="h1", vector_id="v1", status="failed"))
        assert r.find_by_hash("h1", "v1") is None

    def test_status_transitions_and_error_text(self, cfg):
        r = Registry(cfg.registry_path)
        r.upsert(DocumentRecord(doc_id="a", filename="f.pdf", source_type="pdf",
                                content_hash="h", status="pending"))
        r.set_status("a", "failed", "it was a scan")
        got = r.get("a")
        assert got.status == "failed" and "scan" in got.error

    def test_stats_counts_only_ready(self, cfg):
        r = Registry(cfg.registry_path)
        r.upsert(DocumentRecord(doc_id="a", filename="a", source_type="pdf",
                                content_hash="h1", status="ready", n_chunks=5))
        r.upsert(DocumentRecord(doc_id="b", filename="b", source_type="pdf",
                                content_hash="h2", status="failed", n_chunks=99))
        st = r.stats()
        assert st["n_documents"] == 1 and st["n_chunks"] == 5
        assert st["by_status"] == {"ready": 1, "failed": 1}
