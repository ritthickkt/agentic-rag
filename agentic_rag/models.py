"""The data types the served application passes around.

Deliberately separate from `eval/types.py`. That module is the *benchmark's*
contract (a question, a gold answer, an abstention flag) and the benchmark must
stay able to score any system, including ones that predate this file. This module
is about documents a user actually uploaded: where a chunk came from, which page
it was on, and which chunk a sentence in the answer is standing on.

Everything here is plain data with `to_dict()`, because it all ends up either in
SQLite, in a Qdrant payload, or in a JSON response — and a type you can read with
`curl | jq` is a type you can debug.

The central invariant: **a chunk always knows where it came from.** `doc_id`,
`page`, and `char_start/char_end` are carried from the loader through chunking,
embedding, storage, retrieval, and into the answer's citations. Lose them at any
step and the product's main promise — "click the citation, see the source" —
quietly becomes a guess.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Literal

SourceType = Literal["pdf", "docx", "pptx", "xlsx", "markdown", "text", "html", "url"]
DocStatus = Literal["pending", "indexing", "ready", "failed"]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- loading ----------------------------------------------------------------


@dataclass(frozen=True)
class Section:
    """A contiguous run of text from a document, with where it came from.

    A loader's job is to produce these and nothing else. `page` is 1-based and
    None when the format has no pages (Markdown, HTML). `heading` is the nearest
    enclosing structural label — a Markdown heading, a slide title, a sheet name —
    and it is worth carrying because it is both a retrieval signal and the thing a
    human needs to see next to a citation.
    """

    text: str
    page: int | None = None
    heading: str | None = None
    ordinal: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- chunking ---------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """One indexed unit: the text that gets embedded, plus its provenance."""

    chunk_id: str
    doc_id: str
    text: str
    ordinal: int
    page: int | None = None
    heading: str | None = None
    char_start: int = 0
    char_end: int = 0
    # Denormalised onto the chunk so a search hit can be rendered without a
    # second lookup. Retrieval is on the hot path; a join per hit is not free.
    filename: str = ""
    title: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "ordinal": self.ordinal,
            "page": self.page,
            "heading": self.heading,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "filename": self.filename,
            "title": self.title,
        }

    @classmethod
    def from_payload(cls, p: dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=p["chunk_id"],
            doc_id=p["doc_id"],
            text=p["text"],
            ordinal=p.get("ordinal", 0),
            page=p.get("page"),
            heading=p.get("heading"),
            char_start=p.get("char_start", 0),
            char_end=p.get("char_end", 0),
            filename=p.get("filename", ""),
            title=p.get("title", ""),
        )

    @property
    def location(self) -> str:
        """Human-readable provenance, e.g. `report.pdf p.12 — Methodology`."""
        bits = [self.filename or self.doc_id]
        if self.page is not None:
            bits.append(f"p.{self.page}")
        if self.heading:
            bits.append(f"— {self.heading}")
        return " ".join(bits)


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk that came back from search, with *why* it came back.

    The per-retriever ranks are kept rather than collapsed into one score because
    "the lexical retriever found this and the dense one did not" is the single
    most useful thing to know when a hybrid system retrieves something strange.
    """

    chunk: Chunk
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None
    hop: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.chunk.to_payload(),
            "score": round(self.score, 5),
            "dense_rank": self.dense_rank,
            "lexical_rank": self.lexical_rank,
            "hop": self.hop,
        }


# --- documents --------------------------------------------------------------


@dataclass
class DocumentRecord:
    """Registry row for one ingested file. Mutable: status changes as it indexes."""

    doc_id: str
    filename: str
    source_type: SourceType
    content_hash: str
    n_bytes: int = 0
    n_chars: int = 0
    n_chunks: int = 0
    n_pages: int | None = None
    title: str = ""
    source_uri: str = ""
    stored_path: str = ""
    vector_id: str = ""
    status: DocStatus = "pending"
    error: str = ""
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- answering --------------------------------------------------------------


@dataclass(frozen=True)
class Citation:
    """One `[n]` marker in an answer, resolved back to its source.

    `quote` is the span the model claimed to be leaning on. It is verified against
    the chunk text before this object is built (see `citations.py`) — an
    unverifiable quote is a fabricated citation, which is strictly worse than no
    citation because it looks like evidence.
    """

    marker: int
    chunk_id: str
    doc_id: str
    filename: str
    page: int | None
    heading: str | None
    quote: str
    verified: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HopTrace:
    """What one iteration of the agent loop did. This is the explainability."""

    hop: int
    query: str
    n_new_chunks: int
    n_total_chunks: int
    sufficient: bool
    missing: str = ""
    relevant: list[int] = field(default_factory=list)
    reasoning: str = ""
    latency_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnswerResult:
    """Everything one question produced: the answer, its evidence, and the trace."""

    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    hops: list[HopTrace] = field(default_factory=list)
    abstained: bool = False
    abstain_reason: str = ""
    confidence: str = ""
    # Citation audit from `citations.verify_citations`: how many quotes were found
    # where the model said they were, how many had to be repointed, how many could
    # not be found at all. Surfaced in the API response rather than kept internal,
    # because a caller may reasonably refuse an answer whose citations did not check
    # out — and because a rising unverified count is the first sign of trouble.
    verification: dict[str, Any] = field(default_factory=dict)
    n_llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "confidence": self.confidence,
            "citations": [c.to_dict() for c in self.citations],
            "verification": self.verification,
            "chunks": [c.to_dict() for c in self.chunks],
            "hops": [h.to_dict() for h in self.hops],
            "usage": {
                "n_llm_calls": self.n_llm_calls,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "latency_s": round(self.latency_s, 2),
                "model": self.model,
            },
        }
