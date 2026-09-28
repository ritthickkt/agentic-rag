"""Ingestion end to end, and the registry/store consistency it must maintain."""

from __future__ import annotations

import pytest

from agentic_rag.ingest.loaders import CorruptDocument, UnsupportedFormat


class TestIngest:
    def test_file_becomes_searchable_chunks(self, pipeline, retriever, sample_md):
        res = pipeline.ingest_bytes(sample_md, "vendors.md")
        assert res.document.status == "ready"
        assert res.document.n_chunks > 0
        assert retriever.store.count() == res.document.n_chunks
        assert res.document.title == "Vendor Report"

    def test_chunks_carry_the_filename_for_citations(self, pipeline, retriever, sample_md):
        pipeline.ingest_bytes(sample_md, "vendors.md")
        assert all(c.filename == "vendors.md" for c in retriever.store.iter_chunks())

    def test_reupload_is_a_no_op(self, pipeline, retriever, sample_md):
        """Duplicate chunks let one document occupy every slot in the top-k."""
        first = pipeline.ingest_bytes(sample_md, "vendors.md")
        n = retriever.store.count()
        second = pipeline.ingest_bytes(sample_md, "vendors.md")
        assert second.duplicate is True
        assert second.document.doc_id == first.document.doc_id
        assert retriever.store.count() == n

    def test_same_content_different_name_still_dedupes(self, pipeline, sample_md):
        pipeline.ingest_bytes(sample_md, "a.md")
        assert pipeline.ingest_bytes(sample_md, "b.md").duplicate is True

    def test_original_is_kept_so_citations_can_link_to_it(self, pipeline, sample_md):
        from pathlib import Path

        res = pipeline.ingest_bytes(sample_md, "vendors.md")
        assert Path(res.document.stored_path).is_file()

    def test_oversized_upload_is_rejected_before_any_row_is_written(self, pipeline, cfg):
        big = b"x" * (cfg.max_upload_mb * 1024 * 1024 + 1)
        with pytest.raises(ValueError, match="over the"):
            pipeline.ingest_bytes(big, "big.txt")
        assert pipeline.registry.list() == []

    def test_unsupported_format_leaves_no_row(self, pipeline):
        with pytest.raises(UnsupportedFormat):
            pipeline.ingest_bytes(b"data", "archive.zip")
        assert pipeline.registry.list() == []


class TestFailureHandling:
    def test_failed_parse_leaves_a_visible_row_with_the_reason(self, pipeline, retriever):
        """An upload that vanishes silently is the worst outcome: the user cannot
        tell whether it worked, retry sensibly, or report anything useful."""
        with pytest.raises(CorruptDocument):
            pipeline.ingest_bytes(b"%PDF-1.4 truncated", "broken.pdf")

        docs = pipeline.registry.list()
        assert len(docs) == 1
        assert docs[0].status == "failed"
        assert "damaged" in docs[0].error
        # No partial vectors: they would be retrieved and cited from a document the
        # UI shows as failed.
        assert retriever.store.count() == 0

    def test_a_failed_document_does_not_block_retrying_it(self, pipeline, sample_md):
        with pytest.raises(CorruptDocument):
            pipeline.ingest_bytes(b"%PDF-1.4 truncated", "doc.pdf")
        assert pipeline.ingest_bytes(sample_md, "doc.md").document.status == "ready"


class TestDeleteAndReconcile:
    def test_delete_removes_vectors_row_and_original(self, pipeline, retriever, sample_md):
        from pathlib import Path

        res = pipeline.ingest_bytes(sample_md, "vendors.md")
        path = Path(res.document.stored_path)
        assert pipeline.delete(res.document.doc_id) is True
        assert retriever.store.count() == 0
        assert pipeline.registry.get(res.document.doc_id) is None
        assert not path.exists()

    def test_delete_unknown_doc_returns_false(self, pipeline):
        assert pipeline.delete("nope") is False

    def test_reconcile_removes_orphaned_vectors(self, pipeline, retriever, sample_md):
        """Vectors whose registry row is gone would be cited from a document the user
        cannot see or delete — a crash between embedding and the status flip."""
        res = pipeline.ingest_bytes(sample_md, "vendors.md")
        pipeline.registry.delete(res.document.doc_id)   # simulate the lost row

        report = pipeline.reconcile()
        assert res.document.doc_id in report["orphaned_vectors_removed"]
        assert retriever.store.count() == 0

    def test_reconcile_flags_a_ready_document_with_no_vectors(self, pipeline, retriever, sample_md):
        res = pipeline.ingest_bytes(sample_md, "vendors.md")
        retriever.store.delete_doc(res.document.doc_id)  # simulate the lost vectors

        report = pipeline.reconcile()
        assert res.document.doc_id in report["documents_marked_failed"]
        assert pipeline.registry.get(res.document.doc_id).status == "failed"

    def test_reconcile_is_quiet_when_consistent(self, pipeline, sample_md):
        pipeline.ingest_bytes(sample_md, "vendors.md")
        report = pipeline.reconcile()
        assert report["orphaned_vectors_removed"] == []
        assert report["documents_marked_failed"] == []


class TestRetrievalIntegration:
    def test_new_document_is_immediately_findable_by_keyword(self, pipeline, retriever, sample_md):
        """The lexical index lives in memory. Without an explicit invalidation the
        document is findable by vector but invisible to exact-string search until
        restart — a bug that only shows for the queries BM25 exists to serve."""
        pipeline.ingest_bytes(sample_md, "vendors.md")
        hits = retriever.search("Tanaka", k=5)
        assert any("Tanaka" in h.chunk.text for h in hits)

    def test_deleted_document_stops_appearing(self, pipeline, retriever, sample_md):
        res = pipeline.ingest_bytes(sample_md, "vendors.md")
        pipeline.delete(res.document.doc_id)
        assert retriever.search("Tanaka", k=5) == []

    def test_doc_scoped_search(self, pipeline, retriever, sample_md):
        a = pipeline.ingest_bytes(sample_md, "a.md")
        b = pipeline.ingest_bytes(b"# Other\n\nCompletely unrelated content about boats.", "b.md")
        hits = retriever.search("content", k=10, doc_ids=[b.document.doc_id])
        assert hits and {h.chunk.doc_id for h in hits} == {b.document.doc_id}

    def test_stats_reflect_the_corpus(self, pipeline, sample_md):
        pipeline.ingest_bytes(sample_md, "vendors.md")
        st = pipeline.stats()
        assert st["n_documents"] == 1
        assert st["n_chunks"] > 0
        assert st["store"]["n_chunks"] == st["n_chunks"]
