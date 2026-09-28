"""`Section`s -> `Chunk`s. The least glamorous step, and the one that decides the
ceiling on everything downstream.

Retrieval can only ever return a chunk. So if the fact a user needs is split
across a boundary, no amount of clever agent looping will recover it — the loop
can retrieve more chunks, but it cannot un-split a sentence. Chunking is the one
step whose mistakes are invisible at query time and unfixable without re-indexing.

Three decisions, each of which is a real tradeoff:

**Split on structure first, size second.** A Section from the loader is already a
page, slide, sheet, or heading block. We only split it when it exceeds the budget,
and then we split at the strongest available boundary — paragraph, then sentence,
then whitespace, then mid-word as a last resort. A fixed-width window over the
whole document would be simpler and would cut through the middle of tables and
sentences for no reason.

**Never merge across sections.** Two adjacent pages could be packed into one chunk
to reduce their number, but then the chunk spans two page numbers and its citation
has to pick one and be wrong half the time. Provenance beats packing density.

**Overlap, but only within a section.** Overlap exists so a fact sitting on a
boundary appears whole in at least one chunk. It costs storage and creates
near-duplicate hits at query time (deduplicated later by `chunk_id`), which is a
price worth paying because a fact lost to a boundary is lost permanently.
"""

from __future__ import annotations

import hashlib
import re

from agentic_rag.config import CONFIG
from agentic_rag.models import Chunk, Section

# Ordered strongest-to-weakest. The first separator that actually appears in the
# oversized text is the one we split on.
_SEPARATORS = ("\n\n", "\n", ". ", "; ", ", ", " ")

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def chunk_id_for(doc_id: str, ordinal: int, text: str) -> str:
    """Deterministic id from (document, position, content).

    Deterministic so that re-ingesting an unchanged document produces the same
    ids, which makes a re-index an idempotent upsert instead of a duplicate set.
    Content is in the hash so an edited chunk gets a new id rather than silently
    keeping the old vector.
    """
    h = hashlib.sha1(f"{doc_id}:{ordinal}:{text}".encode("utf-8")).hexdigest()[:16]
    return f"{doc_id[:8]}-{ordinal:04d}-{h[:8]}"


def _split_oversized(text: str, budget: int, overlap: int) -> list[tuple[int, int]]:
    """Return (start, end) offsets covering `text`, each at most `budget` long.

    Offsets rather than strings, because the caller needs `char_start`/`char_end`
    relative to the section to build provenance — and slicing first and computing
    offsets afterwards is how off-by-one provenance bugs get in.
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    n = len(text)

    while pos < n:
        end = min(pos + budget, n)
        if end < n:
            # Look for the best boundary in the last third of the window: far
            # enough back to find one, not so far that chunks become tiny.
            window_start = pos + max(budget // 2, 1)
            cut = -1
            for sep in _SEPARATORS:
                found = text.rfind(sep, window_start, end)
                if found != -1:
                    cut = found + len(sep)
                    break
            if cut > pos:
                end = cut
        spans.append((pos, end))
        if end >= n:
            break
        # Step back by the overlap, but always make forward progress.
        pos = max(end - overlap, pos + 1)

    return spans


def chunk_sections(
    sections: list[Section],
    doc_id: str,
    filename: str = "",
    title: str = "",
    chunk_chars: int | None = None,
    overlap: int | None = None,
    min_chars: int | None = None,
) -> list[Chunk]:
    budget = chunk_chars or CONFIG.chunk_chars
    overlap = CONFIG.chunk_overlap if overlap is None else overlap
    min_chars = CONFIG.min_chunk_chars if min_chars is None else min_chars
    if overlap >= budget:
        raise ValueError(f"chunk_overlap ({overlap}) must be smaller than chunk_chars ({budget})")

    chunks: list[Chunk] = []
    carry: Section | None = None  # a section too small to stand alone

    for section in sections:
        text = section.text.strip()
        if not text:
            continue

        # A tiny section (a slide title, a one-line page) is merged forward into
        # the next one *only if they share a location*, so provenance survives.
        if carry is not None:
            if carry.page == section.page and carry.heading == section.heading:
                text = f"{carry.text}\n\n{text}"
                section = Section(text, page=section.page, heading=section.heading,
                                  ordinal=carry.ordinal)
            else:
                chunks.append(_make(carry, carry.text, 0, len(carry.text), doc_id,
                                    len(chunks), filename, title))
            carry = None

        if len(text) < min_chars:
            carry = Section(text, page=section.page, heading=section.heading,
                            ordinal=section.ordinal)
            continue

        if len(text) <= budget:
            chunks.append(_make(section, text, 0, len(text), doc_id, len(chunks),
                                filename, title))
            continue

        for start, end in _split_oversized(text, budget, overlap):
            piece = text[start:end].strip()
            if len(piece) >= min_chars:
                chunks.append(_make(section, piece, start, end, doc_id, len(chunks),
                                    filename, title))

    if carry is not None:
        # Trailing short section: keep it. A 40-character page may be the only
        # place a date appears, and dropping it to tidy the index is a data loss
        # the user will never be told about.
        chunks.append(_make(carry, carry.text, 0, len(carry.text), doc_id, len(chunks),
                            filename, title))

    return chunks


def _make(section: Section, text: str, start: int, end: int, doc_id: str,
          ordinal: int, filename: str, title: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id_for(doc_id, ordinal, text),
        doc_id=doc_id,
        text=text,
        ordinal=ordinal,
        page=section.page,
        heading=section.heading,
        char_start=start,
        char_end=end,
        filename=filename,
        title=title,
    )


def split_sentences(text: str) -> list[str]:
    """Used by citation verification, not by chunking. Crude on purpose."""
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]
