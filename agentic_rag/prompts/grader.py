"""The grader: does the retrieved evidence actually support an answer?

This is the first thing that makes a system "agentic" in any meaningful sense —
it inspects its own retrieval before committing to an answer, rather than
treating whatever came back as ground truth.

Two notes on the design:

`sufficient` is asked as a question about *the passages*, not about the model's
confidence. "Can these passages answer this?" is checkable; "are you sure?" is
not, and models are famously badly calibrated when asked the second one.

`missing` exists to feed the reformulation step. A grader that only says "no"
tells the next hop nothing; one that says "no — the passages identify the film
but say nothing about its director" gives the reformulator something to search
for. This is the whole mechanism by which multi-hop retrieval gets its second hop.
"""

from __future__ import annotations

from eval.types import Passage
from agentic_rag.prompts.baseline import format_context

GRADER_SYSTEM = """You assess whether a set of retrieved passages contains enough \
information to answer a question.

Judge the passages, not your own knowledge. If you happen to know the answer but \
the passages do not state it, the passages are NOT sufficient.

Many questions are multi-hop: they need a fact from one passage joined to a fact \
from another. Passages are sufficient only if every hop is covered."""

# Structured output — the response is guaranteed to validate against this, so the
# caller never has to defend against prose where a boolean was expected.
GRADER_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "1-based numbers of the passages that bear on the question",
        },
        "sufficient": {
            "type": "boolean",
            "description": "True only if these passages fully answer the question",
        },
        "missing": {
            "type": "string",
            "description": "If not sufficient, the specific fact still needed. Empty if sufficient.",
        },
    },
    "required": ["relevant", "sufficient", "missing"],
    "additionalProperties": False,
}


def grader_prompt(question: str, passages: list[Passage]) -> str:
    return f"""Question: {question}

Retrieved passages:

{format_context(passages)}

Assess whether these passages are sufficient to answer the question. If they are \
not, state precisely which fact is still missing."""
