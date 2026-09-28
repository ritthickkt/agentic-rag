"""Hybrid retrieval: tokenisation, RRF fusion, and cross-hop accumulation."""

from __future__ import annotations

from agentic_rag.models import Chunk, RetrievedChunk
from agentic_rag.retrieve import fuse_rrf, merge_unique, tokenize


def ch(i: int, text: str = "") -> Chunk:
    return Chunk(chunk_id=f"c{i}", doc_id="d", text=text or f"text {i}", ordinal=i)


class TestTokenize:
    def test_keeps_identifiers_intact(self):
        """These are exactly the queries dense retrieval is worst at."""
        assert tokenize("ERR_CONN_4021 in v2.1.3 and covid-19") == [
            "err_conn_4021", "v2.1.3", "covid-19"
        ]

    def test_drops_stopwords_and_single_chars(self):
        assert tokenize("What is the a b revenue of the company") == ["revenue", "company"]

    def test_empty_query_is_empty_not_an_error(self):
        assert tokenize("  the  ") == []


class TestFusion:
    def test_chunk_found_by_both_retrievers_wins(self):
        """The whole point of fusing: agreement beats a single high rank."""
        dense = [(ch(1), 0.9), (ch(2), 0.8), (ch(3), 0.7)]
        lexical = [(ch(3), 12.0), (ch(4), 5.0)]
        fused = fuse_rrf(dense, lexical, k=4)
        assert fused[0].chunk.chunk_id == "c3"
        assert fused[0].dense_rank == 3 and fused[0].lexical_rank == 1

    def test_rank_only_no_score_magnitudes(self):
        """BM25 scores are unbounded; fusion must not be sensitive to their scale."""
        dense = [(ch(1), 0.9)]
        small = fuse_rrf(dense, [(ch(2), 0.01)], k=2)
        huge = fuse_rrf(dense, [(ch(2), 9999.0)], k=2)
        assert [r.chunk.chunk_id for r in small] == [r.chunk.chunk_id for r in huge]
        assert small[0].score == huge[0].score

    def test_lexical_only_hit_still_surfaces(self):
        fused = fuse_rrf([(ch(1), 0.9)], [(ch(9), 3.0)], k=5)
        ids = [r.chunk.chunk_id for r in fused]
        assert "c9" in ids
        assert next(r for r in fused if r.chunk.chunk_id == "c9").dense_rank is None

    def test_truncates_to_k(self):
        dense = [(ch(i), 1.0 / i) for i in range(1, 11)]
        assert len(fuse_rrf(dense, [], k=3)) == 3

    def test_empty_inputs(self):
        assert fuse_rrf([], [], k=5) == []


class TestMergeAcrossHops:
    def test_later_hop_adds_without_discarding(self):
        """A loop that can lose evidence it already found is a bug, not a tradeoff."""
        first = [RetrievedChunk(chunk=ch(1), score=0.5, hop=0)]
        second = [RetrievedChunk(chunk=ch(2), score=0.4, hop=1)]
        merged, added = merge_unique(first, second)
        assert added == 1
        assert {r.chunk.chunk_id for r in merged} == {"c1", "c2"}

    def test_duplicate_keeps_best_score_but_earliest_hop(self):
        first = [RetrievedChunk(chunk=ch(1), score=0.3, hop=0)]
        second = [RetrievedChunk(chunk=ch(1), score=0.9, hop=1)]
        merged, added = merge_unique(first, second)
        assert added == 0 and len(merged) == 1
        assert merged[0].score == 0.9
        assert merged[0].hop == 0   # provenance: where the evidence entered

    def test_result_is_sorted_by_score(self):
        merged, _ = merge_unique(
            [RetrievedChunk(chunk=ch(1), score=0.2)],
            [RetrievedChunk(chunk=ch(2), score=0.8), RetrievedChunk(chunk=ch(3), score=0.5)],
        )
        assert [r.chunk.chunk_id for r in merged] == ["c2", "c3", "c1"]
