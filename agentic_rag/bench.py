"""Adapter: make the served engine scoreable by the benchmark harness.

This module is the reason the eval survives the move from experiment to product.
`eval/` scores anything satisfying `question -> SystemAnswer` and knows nothing about
Qdrant, Anthropic, or chunking (DECISIONS D3). The served `Answerer` returns a much
richer `AnswerResult` — citations, page numbers, per-hop traces. So one small
translation makes the real system measurable against the same 420 labelled questions,
with the same abstention scoring, that the four benchmark rungs were measured on.

Why bother, when the app demonstrably works on a handful of documents: because
"it looked right on the five files I tried" is not a claim, and every change to a
prompt, a chunk size, or an embedding provider is a change that can quietly make
retrieval worse. The benchmark is the only thing here that can tell the difference
between a refactor and a regression.

**What the comparison is and is not.** Run on `hotpotqa_bench`, the product engine is
measured under the same corpus and the same questions as `naive`/`graded`/`agentic`.
It is *not* a like-for-like ablation of those rungs: it chunks differently (one
paragraph per chunk, so in practice the same), retrieves hybrid rather than dense-only,
and answers with citations rather than a bare span. Those differences are the product,
and they are the point — but they mean a delta against `agentic` is a delta between
two different systems, not the value of one isolated change. Read it as "is the thing
we ship better than the thing we measured", and nothing finer.
"""

from __future__ import annotations

from agentic_rag.agent import Answerer
from eval.types import Passage, SystemAnswer


class ProductSystem:
    """The served answerer, wearing the benchmark's interface."""

    def __init__(self, answerer: Answerer, name: str = "product") -> None:
        self.answerer = answerer
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, question: str) -> SystemAnswer:
        res = self.answerer.answer(question)

        # `doc_id` is the join key for every retrieval metric: the benchmark compares
        # it against a question's `gold_titles`. The bench index therefore stores the
        # HotpotQA article title as the doc_id, and a mismatch here would silently
        # zero out recall rather than raise.
        passages = tuple(
            Passage(doc_id=rc.chunk.doc_id, text=rc.chunk.text, score=rc.score)
            for rc in res.chunks
        )

        # A partial answer is an answer. `abstained` must mean "produced no answer",
        # because the metrics count an abstention on an answerable item as a miss —
        # and scoring a delivered partial answer as a refusal would understate the
        # system in exactly the direction that makes it look worse than it is.
        abstained = res.abstained and not res.answer.strip()

        verification = res.verification or {}
        return SystemAnswer(
            answer=res.answer,
            passages=passages,
            abstained=abstained,
            meta={
                "system": self._name,
                "n_llm_calls": res.n_llm_calls,
                "hops": len(res.hops),
                "queries": [h.query for h in res.hops],
                "input_tokens": res.input_tokens,
                "output_tokens": res.output_tokens,
                "latency_s": round(res.latency_s, 2),
                "confidence": res.confidence,
                "abstain_reason": res.abstain_reason,
                "abstained_by": ("answerer" if abstained else None),
                "raw_answer": res.answer,
                # Citation health per question. A run whose unverified count climbs is
                # regressing in a way accuracy alone would not show.
                "citations_verified": verification.get("n_verified", 0),
                "citations_relocated": verification.get("n_relocated", 0),
                "citations_unverified": verification.get("n_unverified", 0),
                "n_citations": len(res.citations),
                "grader_verdicts": [
                    {"hop": h.hop, "sufficient": h.sufficient, "missing": h.missing}
                    for h in res.hops
                ],
            },
        )
