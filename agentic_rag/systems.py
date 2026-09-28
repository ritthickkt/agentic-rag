"""The four systems under comparison, all satisfying `question -> SystemAnswer`.

The ladder exists so that a final "agentic beats naive by N points" can be
decomposed into *which change bought which points*:

  naive          retrieve once, answer. Not told it may decline.
  naive_abstain  identical, plus one sentence permitting abstention.
  graded         retrieve once, have the model grade its own evidence, and
                 abstain when the grader says the evidence is insufficient.
  agentic        graded, plus reformulate-and-retrieve-again on insufficiency.

Reading the gaps: naive -> naive_abstain is what a *sentence of prompt* is worth.
naive_abstain -> graded is what an explicit self-check is worth on top of that.
graded -> agentic is what a second retrieval hop is worth. Without the middle
rung, the prompt's contribution would be silently credited to the architecture —
a mistake that would make the whole project's headline claim wrong.

Every system reports `meta.n_llm_calls` and its token counts, because the
interesting result is a cost/quality tradeoff, not a single accuracy number.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from agentic_rag.config import CONFIG
from agentic_rag.index import Retriever
from agentic_rag.llm import LLM
from agentic_rag.prompts.baseline import (
    ABSTAIN_TOKEN,
    answer_prompt,
    answer_system,
    format_context,
)
from agentic_rag.prompts.grader import GRADER_SCHEMA, GRADER_SYSTEM, grader_prompt
from agentic_rag.prompts.reformulate import (
    REFORMULATE_SCHEMA,
    REFORMULATE_SYSTEM,
    reformulate_prompt,
)
from eval.types import Passage, SystemAnswer


def _is_abstention(text: str) -> bool:
    """Did the model emit the abstention token we asked for?

    Strictly the token, not a sentiment read of the text. A model that wanders
    off-script and says "the context doesn't mention this" is NOT counted here —
    that case is caught later by the judge as a `no_answer` verdict, and the two
    are kept separate on purpose: this one is the system doing what it was told,
    that one is the system going off-script. Collapsing them would hide how
    reliably the instruction is actually followed.
    """
    return text.strip().upper().startswith(ABSTAIN_TOKEN)


@dataclass
class NaiveRAG:
    """Retrieve top-k once, stuff the context, answer. The thing to beat."""

    retriever: Retriever
    llm: LLM
    allow_abstain: bool = False
    top_k: int | None = None

    @property
    def name(self) -> str:
        return "naive_abstain" if self.allow_abstain else "naive"

    def __call__(self, question: str) -> SystemAnswer:
        t0 = time.time()
        k = self.top_k or CONFIG.top_k
        passages = self.retriever.search(question, k=k)
        out = self.llm.complete(
            answer_prompt(question, passages),
            system=answer_system(self.allow_abstain),
            max_tokens=128,
        )
        abstained = self.allow_abstain and _is_abstention(out.text)
        return SystemAnswer(
            answer="" if abstained else out.text,
            passages=tuple(passages),
            abstained=abstained,
            meta={
                "system": self.name,
                "n_llm_calls": 1,
                "hops": 1,
                "queries": [question],
                "raw_answer": out.text,
                "input_tokens": out.input_tokens,
                "output_tokens": out.output_tokens,
                "latency_s": round(time.time() - t0, 2),
            },
        )


@dataclass
class AgenticRAG:
    """Retrieve, grade the evidence, optionally reformulate and retrieve again.

    With `use_reformulation=False` and `max_hops=1` this is the `graded` rung: one
    retrieval, one self-check, abstain if the check fails. Enabling both turns it
    into the full loop.

    Design note on the abstention path: when the grader says "insufficient" and we
    are out of hops, the system abstains **without** making an answer call. That is
    not just a saving — it is the architectural difference being measured. The
    naive system cannot decline because it never forms an opinion about its own
    evidence; this one declines because it did.
    """

    retriever: Retriever
    llm: LLM
    use_reformulation: bool = True
    max_hops: int = 3
    top_k: int | None = None
    name_override: str | None = None

    @property
    def name(self) -> str:
        if self.name_override:
            return self.name_override
        return "agentic" if self.use_reformulation and self.max_hops > 1 else "graded"

    def __call__(self, question: str) -> SystemAnswer:
        t0 = time.time()
        k = self.top_k or CONFIG.top_k
        n_calls = 0
        in_tok = out_tok = 0

        query = question
        queries: list[str] = []
        # Accumulates across hops, deduped by doc_id. A later hop adds evidence
        # rather than replacing it — dropping hop one's passages would make the
        # loop able to lose ground, which is a bug and not a tradeoff.
        seen: dict[str, Passage] = {}
        verdicts: list[dict[str, Any]] = []
        sufficient = False

        for hop in range(self.max_hops):
            queries.append(query)
            for p in self.retriever.search(query, k=k):
                seen.setdefault(p.doc_id, p)
            passages = sorted(seen.values(), key=lambda p: -(p.score or 0.0))

            grade, gout = self.llm.complete_json_with_usage(
                grader_prompt(question, passages),
                schema=GRADER_SCHEMA,
                system=GRADER_SYSTEM,
                max_tokens=400,
            )
            n_calls += 1
            in_tok += gout.input_tokens
            out_tok += gout.output_tokens
            verdicts.append({"hop": hop, **grade})
            sufficient = bool(grade.get("sufficient"))

            if sufficient or not self.use_reformulation or hop == self.max_hops - 1:
                break

            missing = (grade.get("missing") or "").strip()
            if not missing:
                # Grader said "not sufficient" but named nothing missing. There is
                # nothing to search for, so another hop would just re-retrieve the
                # same passages. Stop rather than burn a call for show.
                break
            ref, rout = self.llm.complete_json_with_usage(
                reformulate_prompt(question, format_context(passages), missing),
                schema=REFORMULATE_SCHEMA,
                system=REFORMULATE_SYSTEM,
                max_tokens=200,
            )
            n_calls += 1
            in_tok += rout.input_tokens
            out_tok += rout.output_tokens
            new_query = (ref.get("query") or "").strip()
            if not new_query or new_query == query:
                break
            query = new_query

        passages = sorted(seen.values(), key=lambda p: -(p.score or 0.0))
        meta: dict[str, Any] = {
            "system": self.name,
            "hops": len(queries),
            "queries": queries,
            "grader_verdicts": verdicts,
            "grader_sufficient": sufficient,
        }

        if not sufficient:
            # Abstain on the grader's word, with no answer call at all.
            meta.update({"n_llm_calls": n_calls, "abstained_by": "grader",
                         "input_tokens": in_tok, "output_tokens": out_tok,
                         "latency_s": round(time.time() - t0, 2)})
            return SystemAnswer(answer="", passages=tuple(passages), abstained=True, meta=meta)

        ans = self.llm.complete(
            answer_prompt(question, passages),
            system=answer_system(allow_abstain=True),
            max_tokens=128,
        )
        n_calls += 1
        abstained = _is_abstention(ans.text)
        meta.update({
            "n_llm_calls": n_calls,
            "raw_answer": ans.text,
            "abstained_by": "answerer" if abstained else None,
            "input_tokens": in_tok + ans.input_tokens,
            "output_tokens": out_tok + ans.output_tokens,
            "latency_s": round(time.time() - t0, 2),
        })
        return SystemAnswer(
            answer="" if abstained else ans.text,
            passages=tuple(passages),
            abstained=abstained,
            meta=meta,
        )


def build_system(name: str, retriever: Retriever, llm: LLM):
    """One place that maps a CLI name to a configured system."""
    if name == "naive":
        return NaiveRAG(retriever, llm, allow_abstain=False)
    if name == "naive_abstain":
        return NaiveRAG(retriever, llm, allow_abstain=True)
    if name == "graded":
        return AgenticRAG(retriever, llm, use_reformulation=False, max_hops=1)
    if name == "agentic":
        return AgenticRAG(retriever, llm, use_reformulation=True, max_hops=3)
    raise ValueError(f"unknown system {name!r}")


SYSTEM_NAMES = ("naive", "naive_abstain", "graded", "agentic")
