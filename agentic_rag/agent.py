"""The agent loop: retrieve, try to answer, and on failure search for what is missing.

What makes this "agentic" is not that an LLM is involved — it is that the system
forms an opinion about its own evidence and acts on it. One pass of naive RAG
cannot tell the difference between "the corpus does not contain this" and "my first
query was phrased badly", so it answers from bad evidence either way. This loop can
tell the difference, because it asks.

**The loop, and why it is shaped like this:**

    retrieve(query) -> answer-or-declare-missing -> reformulate(missing) -> retrieve …

The critical design choice is that **answering and judging sufficiency are the same
call.** The obvious alternative — a separate "grader" call that gates the answer —
was built and measured in this project's benchmark phase, and it declined **17.8%
of questions where both required passages were provably already in the context
window.** A judge that never has to produce an answer is free to be maximally
cautious; asking for the answer and the verdict together means declining is a
decision made by something that actually tried.

**Evidence accumulates and is never discarded.** Hop two adds to hop one's chunks
rather than replacing them (`merge_unique`). A loop that can lose the evidence it
already found is a loop that can score worse than not looping.

**Reformulation targets the gap, not the question.** Re-running the original query
returns the same chunks and wastes the hop, so the next query is written against the
`missing` field from the previous attempt — which is precisely why `missing` is
required to name a specific fact rather than say "insufficient information".
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from agentic_rag.citations import strip_markers, used_markers, verify_citations
from agentic_rag.config import CONFIG
from agentic_rag.llm import LLM
from agentic_rag.models import AnswerResult, HopTrace, RetrievedChunk
from agentic_rag.prompts.answer import (
    ANSWER_SCHEMA,
    ANSWER_SYSTEM,
    answer_prompt,
    format_passages,
)
from agentic_rag.prompts.reformulate import (
    REFORMULATE_SCHEMA,
    REFORMULATE_SYSTEM,
    reformulate_prompt,
)
from agentic_rag.retrieve import Retriever, merge_unique

# Callback for streaming: (event_type, payload). The API turns these into
# server-sent events so the UI can show the loop working instead of a spinner —
# which matters when a three-hop answer takes eight seconds.
EventFn = Callable[[str, dict[str, Any]], None]


@dataclass
class Answerer:
    """The served question-answering system."""

    retriever: Retriever
    llm: LLM
    answer_llm: LLM | None = None
    max_hops: int | None = None
    top_k: int | None = None

    def __post_init__(self) -> None:
        # The answering model may be larger than the one rewriting queries:
        # reformulation is a small transformation, answering with citations is not.
        self.answer_llm = self.answer_llm or self.llm

    # --- public -----------------------------------------------------------

    def answer(
        self,
        question: str,
        doc_ids: Sequence[str] | None = None,
        on_event: EventFn | None = None,
    ) -> AnswerResult:
        t0 = time.time()
        max_hops = self.max_hops or CONFIG.max_hops
        k = self.top_k or CONFIG.top_k

        def emit(kind: str, **payload: Any) -> None:
            if on_event:
                on_event(kind, payload)

        result = AnswerResult(question=question, answer="", model=self.answer_llm.model)
        chunks: list[RetrievedChunk] = []
        query = question
        tried_queries = {query.strip().lower()}
        last_missing = ""

        for hop in range(max_hops):
            hop_t0 = time.time()
            emit("searching", hop=hop, query=query)

            new = self.retriever.search(query, k=k, doc_ids=doc_ids, hop=hop)
            chunks, n_added = merge_unique(chunks, new)
            emit("retrieved", hop=hop, n_new=n_added, n_total=len(chunks),
                 chunks=[rc.to_dict() for rc in new])

            if not chunks:
                # Nothing in the index matched at all. There is nothing for a model
                # to read, so we do not spend a call asking it to read nothing.
                result.abstained = True
                result.abstain_reason = (
                    "No passages matched this question. The index may be empty, or "
                    "the documents may not cover this topic."
                )
                result.hops.append(HopTrace(hop=hop, query=query, n_new_chunks=0,
                                            n_total_chunks=0, sufficient=False,
                                            missing="no passages retrieved",
                                            latency_s=round(time.time() - hop_t0, 2)))
                break

            # The context is capped: more passages is not monotonically better.
            # Past a dozen or so, precision drops and the model starts citing
            # loosely-related chunks because they are present.
            context = chunks[: CONFIG.max_context_chunks]
            is_final = hop == max_hops - 1

            emit("reading", hop=hop, n_passages=len(context))
            parsed, completion = self.answer_llm.complete_json_with_usage(
                answer_prompt(question, context,
                              final_attempt=is_final and CONFIG.answer_on_exhausted),
                schema=ANSWER_SCHEMA,
                system=ANSWER_SYSTEM,
                max_tokens=1400,
            )
            result.n_llm_calls += 1
            result.input_tokens += completion.input_tokens
            result.output_tokens += completion.output_tokens

            insufficient = bool(parsed.get("insufficient"))
            missing = (parsed.get("missing") or "").strip()
            answer_text = (parsed.get("answer") or "").strip()
            last_missing = missing or last_missing

            result.hops.append(HopTrace(
                hop=hop,
                query=query,
                n_new_chunks=n_added,
                n_total_chunks=len(chunks),
                sufficient=not insufficient,
                missing=missing,
                relevant=sorted({int(c.get("passage", 0)) for c in parsed.get("citations", [])
                                 if isinstance(c, dict)}),
                latency_s=round(time.time() - hop_t0, 2),
            ))

            # A model that sets insufficient but writes an answer anyway is
            # trusting its own prose over its own flag. Treat the answer as real:
            # the flag is advisory, the answer is the deliverable.
            if not insufficient and answer_text:
                self._finalize(result, answer_text, parsed, context)
                result.confidence = str(parsed.get("confidence") or "medium")
                emit("answer", answer=result.answer,
                     citations=[c.to_dict() for c in result.citations])
                break
            if insufficient and answer_text:
                self._finalize(result, answer_text, parsed, context)
                result.confidence = "low"
                result.abstain_reason = missing
                emit("answer", answer=result.answer,
                     citations=[c.to_dict() for c in result.citations], partial=True)
                break

            # Genuinely nothing to say from this evidence.
            if is_final or not missing:
                result.abstained = True
                result.abstain_reason = missing or (
                    "The retrieved passages do not contain the information needed "
                    "to answer this question."
                )
                emit("abstained", reason=result.abstain_reason)
                break

            emit("reformulating", hop=hop, missing=missing)
            next_query = self._reformulate(question, context, missing, result)
            if not next_query or next_query.strip().lower() in tried_queries:
                # Repeating a query returns the same chunks, so another hop would
                # burn a call to learn nothing. Stop and say why.
                result.abstained = True
                result.abstain_reason = missing
                emit("abstained", reason=result.abstain_reason)
                break
            tried_queries.add(next_query.strip().lower())
            query = next_query

        result.chunks = chunks
        result.latency_s = time.time() - t0
        if not result.abstain_reason and result.abstained:
            result.abstain_reason = last_missing
        emit("done", latency_s=round(result.latency_s, 2), n_llm_calls=result.n_llm_calls)
        return result

    # --- internals --------------------------------------------------------

    def _reformulate(
        self, question: str, context: Sequence[RetrievedChunk], missing: str,
        result: AnswerResult
    ) -> str:
        parsed, completion = self.llm.complete_json_with_usage(
            reformulate_prompt(question, format_passages(context), missing),
            schema=REFORMULATE_SCHEMA,
            system=REFORMULATE_SYSTEM,
            max_tokens=200,
        )
        result.n_llm_calls += 1
        result.input_tokens += completion.input_tokens
        result.output_tokens += completion.output_tokens
        return (parsed.get("query") or "").strip()

    def _finalize(
        self,
        result: AnswerResult,
        answer_text: str,
        parsed: dict[str, Any],
        context: Sequence[RetrievedChunk],
    ) -> None:
        """Verify citations, repair the answer text, and attach both."""
        report = verify_citations(parsed.get("citations") or [], context)
        result.citations = report.citations

        # Two kinds of broken marker get removed from the prose: ones whose
        # citation was unresolvable, and ones the model used without supplying a
        # citation at all. Either way the reader would be looking at a link to
        # nowhere and reading it as evidence.
        resolved = {c.marker for c in report.citations}
        dangling = [m for m in used_markers(answer_text) if m not in resolved]
        result.answer = strip_markers(answer_text, report.dropped_markers + dangling)
        result.verification = report.to_dict()
