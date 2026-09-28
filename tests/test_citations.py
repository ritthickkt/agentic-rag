"""Citation verification: the check that makes a citation mean something."""

from __future__ import annotations

from agentic_rag.citations import (
    strip_markers,
    used_markers,
    verify_citations,
)
from agentic_rag.models import Chunk, RetrievedChunk


def rc(i: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id=f"c{i}", doc_id="d", text=text, ordinal=i, page=i,
                    filename="report.pdf"),
        score=1.0,
    )


CHUNKS = [
    rc(1, "Total revenue for Q3 2025 was $47.2 million, up 12 percent."),
    rc(2, "The CTO is Dr. Priya Raman, who joined in February 2024."),
    rc(3, "Engineering headcount grew from 88 to 104 this quarter."),
]


def test_exact_quote_verifies():
    r = verify_citations([{"passage": 1, "quote": "revenue for Q3 2025 was $47.2 million"}], CHUNKS)
    assert r.n_verified == 1 and r.n_unverified == 0
    assert r.citations[0].verified and r.citations[0].chunk_id == "c1"


def test_marker_equals_passage_number():
    """There is deliberately no second numbering for the two to disagree about."""
    r = verify_citations([{"passage": 3, "quote": "headcount grew from 88 to 104"}], CHUNKS)
    assert r.citations[0].marker == 3


def test_quote_in_wrong_passage_is_relocated_not_accepted_blindly():
    r = verify_citations([{"passage": 1, "quote": "The CTO is Dr. Priya Raman"}], CHUNKS)
    assert r.n_relocated == 1 and r.n_verified == 0
    assert r.citations[0].chunk_id == "c2"      # repointed to where the text really is
    assert r.citations[0].marker == 1           # but the prose still says [1]


def test_fabricated_quote_is_flagged_not_silently_shown():
    r = verify_citations(
        [{"passage": 2, "quote": "She previously led engineering at Northwind Systems"}], CHUNKS
    )
    assert r.n_unverified == 1
    assert r.citations[0].verified is False
    assert r.all_verified is False


def test_typography_drift_still_verifies():
    """Models 'copy verbatim' while normalising quotes and dashes."""
    r = verify_citations(
        [{"passage": 1, "quote": "Total revenue for Q3 2025 was “$47.2 million”, up 12 percent"}],
        CHUNKS,
    )
    assert r.n_verified == 1


def test_out_of_range_passage_is_dropped():
    r = verify_citations([{"passage": 99, "quote": "anything at all"}], CHUNKS)
    assert r.citations == [] and r.dropped_markers == [99]


def test_short_quote_is_not_fuzzy_matched():
    """'revenue' appears everywhere; a 7-char match is not evidence."""
    r = verify_citations([{"passage": 3, "quote": "revenue"}], CHUNKS)
    assert r.citations[0].verified is False


def test_duplicate_passage_entries_collapse():
    r = verify_citations(
        [{"passage": 1, "quote": "revenue for Q3 2025"}, {"passage": 1, "quote": "up 12 percent"}],
        CHUNKS,
    )
    assert len(r.citations) == 1


def test_used_markers_and_stripping():
    answer = "Revenue rose [1]. The CTO joined in 2024 [2]. Something unsourced [7]."
    assert used_markers(answer) == [1, 2, 7]
    cleaned = strip_markers(answer, [7])
    assert "[7]" not in cleaned and "[1]" in cleaned
    # Stripping must not leave a space before the full stop.
    assert " ." not in cleaned


def test_empty_quote_is_dropped():
    r = verify_citations([{"passage": 1, "quote": "   "}], CHUNKS)
    assert r.citations == [] and 1 in r.dropped_markers
