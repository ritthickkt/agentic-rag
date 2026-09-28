"""Chunking. The step whose mistakes are invisible at query time."""

from __future__ import annotations

import pytest

from agentic_rag.ingest.chunker import chunk_id_for, chunk_sections
from agentic_rag.models import Section


def test_short_section_becomes_one_chunk():
    chunks = chunk_sections([Section("a" * 100, page=3)], doc_id="d" * 32,
                            chunk_chars=300, overlap=50, min_chars=20)
    assert len(chunks) == 1
    assert chunks[0].page == 3


def test_oversized_section_is_split_and_fully_covered():
    text = ". ".join(f"Sentence number {i} about something" for i in range(40))
    chunks = chunk_sections([Section(text, page=1)], doc_id="d" * 32,
                            chunk_chars=200, overlap=40, min_chars=20)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    # Every chunk keeps the page it came from; losing this breaks every citation.
    assert {c.page for c in chunks} == {1}


def test_split_prefers_sentence_boundaries():
    text = "First sentence here. " * 30
    chunks = chunk_sections([Section(text)], doc_id="d" * 32,
                            chunk_chars=120, overlap=20, min_chars=20)
    # A boundary-respecting split leaves most chunks ending at a sentence end.
    ends_clean = sum(1 for c in chunks if c.text.rstrip().endswith("."))
    assert ends_clean >= len(chunks) - 1


def test_sections_are_never_merged_across_different_pages():
    """Packing two pages into one chunk would make its citation a coin flip."""
    sections = [Section("short one", page=1), Section("short two", page=2)]
    chunks = chunk_sections(sections, doc_id="d" * 32, chunk_chars=300,
                            overlap=50, min_chars=5)
    pages = [c.page for c in chunks]
    assert pages == [1, 2]
    assert all("short one" not in c.text or "short two" not in c.text for c in chunks)


def test_tiny_section_merges_forward_only_within_same_location():
    sections = [Section("Title", page=1, heading="H"),
                Section("Body text that is long enough to stand alone here.", page=1, heading="H")]
    chunks = chunk_sections(sections, doc_id="d" * 32, chunk_chars=300,
                            overlap=50, min_chars=30)
    assert len(chunks) == 1
    assert "Title" in chunks[0].text and "Body text" in chunks[0].text


def test_trailing_tiny_section_is_kept_not_dropped():
    """A 9-character page may hold the only date in the document."""
    sections = [Section("Long enough body of text to be its own chunk here.", page=1),
                Section("May 2024", page=2)]
    chunks = chunk_sections(sections, doc_id="d" * 32, chunk_chars=300,
                            overlap=50, min_chars=30)
    assert any("May 2024" in c.text for c in chunks)


def test_chunk_ids_are_deterministic_and_content_sensitive():
    a = chunk_id_for("doc1", 0, "hello")
    assert a == chunk_id_for("doc1", 0, "hello")   # re-ingest is idempotent
    assert a != chunk_id_for("doc1", 0, "hello!")  # edited text gets a new id
    assert a != chunk_id_for("doc1", 1, "hello")   # position matters


def test_overlap_must_be_smaller_than_budget():
    with pytest.raises(ValueError, match="overlap"):
        chunk_sections([Section("x" * 500)], doc_id="d", chunk_chars=100, overlap=100)


def test_overlap_makes_boundary_text_appear_twice():
    text = "A" * 150 + "THE-CRITICAL-FACT" + "B" * 150
    chunks = chunk_sections([Section(text)], doc_id="d" * 32,
                            chunk_chars=180, overlap=60, min_chars=10)
    # With no overlap the fact could be severed; with overlap it survives whole
    # somewhere. That is the entire reason overlap costs us storage.
    assert any("THE-CRITICAL-FACT" in c.text for c in chunks)
