"""Hybrid retrieval: dense vectors + BM25 lexical, fused with RRF.

**Why not dense alone.** Embeddings match meaning, which is exactly what you want
for "what were the risks in the merger" and exactly what fails for `ERR_CONN_4021`,
`Section 14(b)`, a part number, or a surname the model has never seen. Those are
*lexical* needs: the user knows the string and wants the document containing it.
A dense model maps an unfamiliar token to something vaguely nearby and returns
paragraphs about the general vicinity of the topic. BM25 — keyword scoring with
term-rarity weighting — nails them. The two fail in different places, which is
precisely why fusing beats either.

**Why RRF rather than adding the scores.** Cosine similarity lives on roughly
0–1; BM25 is unbounded and corpus-dependent. Combining them numerically means
inventing a weight, and that weight is wrong the moment the corpus changes.
Reciprocal Rank Fusion throws the magnitudes away and uses only *rank*:
`score = sum over retrievers of 1/(k + rank)`. A chunk both retrievers rank
highly wins; a chunk one retriever loves and the other has never heard of still
places well. Nothing needs tuning per corpus.

**The lexical index is in memory and rebuilt on change.** `rank_bm25` holds the
whole tokenised corpus in RAM, which is fine into the low hundreds of thousands of
chunks and is honestly the wrong answer above that — at which point the fix is
Qdrant's server-side sparse vectors, not a bigger laptop. Stated here rather than
discovered later (DECISIONS D23).
"""

from __future__ import annotations

import re
import threading
from typing import Sequence

from agentic_rag.config import CONFIG
from agentic_rag.embed import EmbeddingProvider, build_embedder
from agentic_rag.models import Chunk, RetrievedChunk
from agentic_rag.store import VectorStore

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")

# Dropped from BM25 only. They carry real weight in the dense model, where the
# whole sentence is encoded, but in a bag-of-words scorer they only add noise.
_STOPWORDS = frozenset(
    """a an and are as at be been but by for from has have how i if in into is it its
    of on or that the their there these this to was were what when where which who
    why will with would you your""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, keeping internal - _ . joins.

    That last part is deliberate: `ERR_CONN_4021`, `v2.1.3`, and `covid-19` are
    exactly the queries dense retrieval is worst at, so the lexical half must not
    shatter them into meaningless pieces.
    """
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


class LexicalIndex:
    """BM25 over chunk text, rebuilt from the store when the corpus changes.

    `version` is bumped by the ingestion pipeline on every write. The retriever
    compares versions before searching, so a document ingested a second ago is
    lexically searchable immediately rather than after a restart.
    """

    def __init__(self) -> None:
        self._bm25 = None
        self._chunks: list[Chunk] = []
        self._version = -1
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def version(self) -> int:
        return self._version

    def rebuild(self, chunks: Sequence[Chunk], version: int) -> None:
        from rank_bm25 import BM25Okapi

        with self._lock:
            self._chunks = list(chunks)
            corpus = [tokenize(f"{c.heading or ''} {c.text}") for c in self._chunks]
            # BM25Okapi divides by average document length, so an empty corpus is
            # a ZeroDivisionError rather than an empty result.
            self._bm25 = BM25Okapi(corpus) if corpus else None
            self._version = version

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        with self._lock:
            if self._bm25 is None or not self._chunks:
                return []
            toks = tokenize(query)
            if not toks:
                return []
            scores = self._bm25.get_scores(toks)
            chunks = self._chunks
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        # A zero BM25 score means no query term appears at all. Those are not weak
        # matches, they are non-matches, and passing them to fusion would let
        # arbitrary chunks earn rank credit for nothing.
        return [(chunks[i], float(scores[i])) for i in ranked if scores[i] > 0.0]


class Retriever:
    """Dense + lexical + fusion, over one collection."""

    def __init__(
        self,
        store: VectorStore | None = None,
        embedder: EmbeddingProvider | None = None,
        lexical: LexicalIndex | None = None,
    ) -> None:
        self.embedder = embedder or build_embedder()
        self.store = store or VectorStore()
        self.store.ensure_collection(self.embedder.dim, self.embedder.vector_id)
        self.lexical = lexical if lexical is not None else LexicalIndex()
        self._corpus_version = 0

    # --- lexical index freshness -----------------------------------------

    def mark_dirty(self) -> None:
        """Called by the pipeline after any write, so the next search rebuilds."""
        self._corpus_version += 1

    def _ensure_lexical(self) -> None:
        if not CONFIG.hybrid:
            return
        if self.lexical.version != self._corpus_version:
            self.lexical.rebuild(list(self.store.iter_chunks()), self._corpus_version)

    # --- search -----------------------------------------------------------

    def dense(self, query: str, k: int, doc_ids: Sequence[str] | None = None):
        return self.store.search(self.embedder.encode_query(query), k=k, doc_ids=doc_ids)

    def search(
        self,
        query: str,
        k: int | None = None,
        doc_ids: Sequence[str] | None = None,
        hop: int = 0,
    ) -> list[RetrievedChunk]:
        k = k or CONFIG.top_k
        pool = max(CONFIG.candidate_k, k)

        dense_hits = self.dense(query, pool, doc_ids)
        if not CONFIG.hybrid:
            return [
                RetrievedChunk(chunk=c, score=s, dense_rank=i + 1, hop=hop)
                for i, (c, s) in enumerate(dense_hits[:k])
            ]

        self._ensure_lexical()
        lex_hits = self.lexical.search(query, pool)
        # Filtering after the BM25 scan rather than before keeps one code path;
        # the index is in memory, so the scan is cheap either way.
        if doc_ids:
            allowed = set(doc_ids)
            lex_hits = [(c, s) for c, s in lex_hits if c.doc_id in allowed]

        return fuse_rrf(dense_hits, lex_hits, k=k, rrf_k=CONFIG.rrf_k, hop=hop)


def fuse_rrf(
    dense_hits: Sequence[tuple[Chunk, float]],
    lexical_hits: Sequence[tuple[Chunk, float]],
    k: int,
    rrf_k: int = 60,
    hop: int = 0,
) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion. `rrf_k` flattens the curve; 60 is the standard value.

    Larger `rrf_k` makes ranks matter less and agreement matter more; smaller makes
    a single retriever's #1 dominate. 60 comes from the original RRF paper and is
    left alone deliberately — this is not a knob worth tuning by hand when the eval
    harness can answer the question properly.
    """
    fused: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    dense_rank: dict[str, int] = {}
    lex_rank: dict[str, int] = {}

    for rank, (chunk, _score) in enumerate(dense_hits, start=1):
        fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
        chunks[chunk.chunk_id] = chunk
        dense_rank.setdefault(chunk.chunk_id, rank)

    for rank, (chunk, _score) in enumerate(lexical_hits, start=1):
        fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
        chunks.setdefault(chunk.chunk_id, chunk)
        lex_rank.setdefault(chunk.chunk_id, rank)

    order = sorted(fused, key=lambda cid: (-fused[cid], dense_rank.get(cid, 10**6)))
    return [
        RetrievedChunk(
            chunk=chunks[cid],
            score=fused[cid],
            dense_rank=dense_rank.get(cid),
            lexical_rank=lex_rank.get(cid),
            hop=hop,
        )
        for cid in order[:k]
    ]


def merge_unique(
    existing: list[RetrievedChunk], new: list[RetrievedChunk]
) -> tuple[list[RetrievedChunk], int]:
    """Accumulate across hops, deduped by chunk_id, keeping the best score.

    A later hop *adds* evidence rather than replacing it. Dropping hop one's
    chunks would let the loop lose ground — which is a bug, not a tradeoff.
    """
    by_id = {rc.chunk.chunk_id: rc for rc in existing}
    added = 0
    for rc in new:
        prev = by_id.get(rc.chunk.chunk_id)
        if prev is None:
            by_id[rc.chunk.chunk_id] = rc
            added += 1
        elif rc.score > prev.score:
            # Keep the better score but remember the earliest hop that found it,
            # so the trace shows where evidence entered.
            by_id[rc.chunk.chunk_id] = RetrievedChunk(
                chunk=rc.chunk, score=rc.score, dense_rank=rc.dense_rank,
                lexical_rank=rc.lexical_rank, hop=prev.hop,
            )
    merged = sorted(by_id.values(), key=lambda rc: -rc.score)
    return merged, added
