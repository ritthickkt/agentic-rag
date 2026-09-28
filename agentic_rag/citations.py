"""Verify that a citation's quote actually appears in the passage it cites.

This module is the difference between a system that cites and a system whose
citations mean something. An LLM asked for supporting quotes will occasionally
produce one that is *almost* in the text — a paraphrase, two sentences spliced,
the right fact attributed to the wrong passage. Rendered in a UI next to a
clickable source, that is worse than an uncited answer: it manufactures the
appearance of evidence, and it is the one failure mode a reader cannot catch
without going and reading the source themselves.

So every quote is checked against the chunk it claims to come from, and the result
is one of three states, all of which are reported rather than smoothed over:

- **verified** — the quote is in that chunk (after whitespace/quote normalisation).
- **relocated** — the quote is not in the cited chunk but *is* in another retrieved
  chunk. The marker is repointed to the chunk that actually contains it. A real and
  common off-by-one, and silently correctable, but still counted.
- **unverified** — not found anywhere in the retrieved context. The citation is
  kept but flagged `verified=False`, and the API response carries the count, so a
  caller can choose to reject the answer. It is not deleted, because hiding it
  would leave a marker in the prose pointing at nothing.

Matching is exact-after-normalisation first, then a token-overlap fallback with a
high threshold. The fallback exists because models reliably normalise typography
(curly quotes, ligatures, en dashes) while copying "verbatim"; the threshold is
high because a loose match here is precisely the failure this module exists to
catch.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from agentic_rag.models import Citation, RetrievedChunk

# Below this many characters a quote is not evidence of anything — "revenue" appears
# in every passage. Short quotes are therefore checked ONLY against the chunk the
# model actually named: they may not be relocated to another chunk, and they may not
# be fuzzy-matched. Both of those are searches, and searching a 7-character string
# across the context finds a hit almost every time regardless of whether the citation
# was real. (Found by `test_short_quote_is_not_fuzzy_matched`: "revenue" cited
# against passage 3 was being relocated to passage 1 and reported as verified.)
MIN_SEARCHABLE_QUOTE_CHARS = 12
# Token-overlap ratio required for the fuzzy path. Deliberately strict.
FUZZY_THRESHOLD = 0.85

_WS = re.compile(r"\s+")
_QUOTES = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "−": "-", "­": "",
})


def normalize(s: str) -> str:
    """Fold away the differences a model introduces while copying 'verbatim'."""
    s = unicodedata.normalize("NFKC", s).translate(_QUOTES)
    return _WS.sub(" ", s).strip().lower()


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize(s))


def _overlap(quote: str, text: str) -> float:
    """Fraction of the quote's tokens present, with multiplicity, in the text.

    Multiplicity matters: "the the the revenue" should not score as a perfect
    match against a passage containing one "the" and one "revenue".
    """
    q, t = Counter(_tokens(quote)), Counter(_tokens(text))
    if not q:
        return 0.0
    return sum((q & t).values()) / sum(q.values())


@dataclass
class VerificationReport:
    citations: list[Citation]
    n_verified: int = 0
    n_relocated: int = 0
    n_unverified: int = 0
    dropped_markers: list[int] = None  # markers pointing outside the passage range

    def __post_init__(self) -> None:
        if self.dropped_markers is None:
            self.dropped_markers = []

    @property
    def all_verified(self) -> bool:
        return self.n_unverified == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_verified": self.n_verified,
            "n_relocated": self.n_relocated,
            "n_unverified": self.n_unverified,
            "dropped_markers": self.dropped_markers,
        }


def verify_citations(
    raw_citations: Sequence[dict[str, Any]], chunks: Sequence[RetrievedChunk]
) -> VerificationReport:
    """Resolve model-reported citations against the passages it was shown.

    `raw_citations` are `{passage, quote}` dicts straight off the schema, and the
    passage number doubles as the marker — there is deliberately no second numbering
    for the two to disagree about (see `prompts/answer.py`). `chunks` is the same
    list, in the same order, that was numbered in the prompt.
    """
    report = VerificationReport(citations=[])
    normalized = [normalize(rc.chunk.text) for rc in chunks]
    seen_markers: set[int] = set()

    for raw in raw_citations:
        try:
            passage = int(raw.get("passage", 0))
        except (TypeError, ValueError):
            continue
        # The marker in the prose IS the passage number.
        marker = passage
        quote = str(raw.get("quote", "")).strip()
        if marker in seen_markers:
            continue  # one entry per passage; a repeat is noise

        # A marker pointing outside the passage list cannot be repaired by
        # guessing. Record it so the answer text can have the dangling marker
        # stripped rather than shown.
        if not quote or not (1 <= passage <= len(chunks)):
            report.dropped_markers.append(marker)
            continue

        idx = _locate(quote, normalized, preferred=passage - 1)
        if idx is None:
            rc = chunks[passage - 1]
            report.citations.append(_build(marker, rc, quote, verified=False))
            report.n_unverified += 1
        else:
            rc = chunks[idx]
            report.citations.append(_build(marker, rc, quote, verified=True))
            if idx == passage - 1:
                report.n_verified += 1
            else:
                report.n_relocated += 1
        seen_markers.add(marker)

    report.citations.sort(key=lambda c: c.marker)
    return report


def _locate(quote: str, normalized_chunks: Sequence[str], preferred: int) -> int | None:
    """Index of the chunk containing `quote`, preferring the cited one.

    The cited chunk is checked first and at any quote length. Every *other* chunk is
    a search, and a search only happens for a quote long enough to be distinctive —
    otherwise a generic word gets relocated to whichever chunk happens to contain it
    and the citation is stamped verified on no evidence at all.
    """
    nq = normalize(quote)
    if not nq:
        return None

    # 1. Exactly where the model said it was. Trusted at any length.
    if nq in normalized_chunks[preferred]:
        return preferred

    if len(nq) < MIN_SEARCHABLE_QUOTE_CHARS:
        return None

    others = [i for i in range(len(normalized_chunks)) if i != preferred]

    # 2. Exact, somewhere else: an off-by-one in the model's numbering.
    for i in others:
        if nq in normalized_chunks[i]:
            return i

    # 3. Fuzzy, for typography the model normalised while "copying verbatim".
    best_i, best_score = None, 0.0
    for i in [preferred, *others]:
        score = _overlap(quote, normalized_chunks[i])
        if score > best_score:
            best_i, best_score = i, score
    return best_i if best_score >= FUZZY_THRESHOLD else None


def _build(marker: int, rc: RetrievedChunk, quote: str, verified: bool) -> Citation:
    return Citation(
        marker=marker,
        chunk_id=rc.chunk.chunk_id,
        doc_id=rc.chunk.doc_id,
        filename=rc.chunk.filename,
        page=rc.chunk.page,
        heading=rc.chunk.heading,
        quote=quote,
        verified=verified,
    )


_MARKER = re.compile(r"\[(\d+)\]")


def strip_markers(answer: str, markers: Sequence[int]) -> str:
    """Remove citation markers that resolved to nothing.

    A `[5]` left in the prose with no citation behind it is a broken link in the
    UI, and it reads as evidence the answer does not have.
    """
    if not markers:
        return answer
    drop = {int(m) for m in markers}
    out = _MARKER.sub(lambda m: "" if int(m.group(1)) in drop else m.group(0), answer)
    return re.sub(r" +([.,;:])", r"\1", re.sub(r"[ \t]{2,}", " ", out)).strip()


def used_markers(answer: str) -> list[int]:
    """Markers actually present in the answer text, in order of first appearance."""
    seen: list[int] = []
    for m in _MARKER.finditer(answer):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen
