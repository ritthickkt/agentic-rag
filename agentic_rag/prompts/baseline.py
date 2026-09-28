"""Prompts for answering from retrieved context.

The abstention instruction is a *variable* here, not a constant, and that is the
most important design choice in this file.

LEARNING step 1 committed to one deliberate handicap on the naive baseline: it is
not told it may decline to answer. But if the agentic system is also the only one
*allowed* to abstain, then any improvement on the unanswerable bucket is
unattributable — it could be the architecture noticing bad evidence, or it could
be nothing more than one extra sentence of prompt. Those are very different
claims and cost very different amounts of money.

So `ABSTAIN_RULE` is toggled independently of the architecture, which buys us a
control (`naive_abstain`) that isolates the prompt's contribution. Everything
else about the prompt is identical across systems, on purpose: the answer-format
instruction, the context layout, the ordering.
"""

from __future__ import annotations

from eval.types import Passage

ABSTAIN_TOKEN = "INSUFFICIENT_EVIDENCE"

_BASE = """You are a precise question-answering assistant.

Answer the question using only the context passages provided. Do not use outside \
knowledge, even if you are confident it is correct.

Give the shortest complete answer: a name, a date, a number, a short phrase, or \
"yes"/"no" for yes-no questions. No preamble, no explanation, no full sentences."""

ABSTAIN_RULE = f"""

If the context passages do not contain the information needed to answer, reply \
with exactly {ABSTAIN_TOKEN} and nothing else. Do not guess, and do not answer \
from your own knowledge."""


def answer_system(allow_abstain: bool) -> str:
    return _BASE + (ABSTAIN_RULE if allow_abstain else "")


def format_context(passages: list[Passage]) -> str:
    """Numbered, title-labelled passages.

    Titles are included because they are often part of the answer in HotpotQA
    (the bridge entity is frequently the article title), so hiding them would
    handicap every system equally but for no reason.
    """
    if not passages:
        return "(no passages retrieved)"
    return "\n\n".join(
        f"[{i}] {p.doc_id}\n{p.text}" for i, p in enumerate(passages, start=1)
    )


def answer_prompt(question: str, passages: list[Passage]) -> str:
    return f"""Context passages:

{format_context(passages)}

Question: {question}

Answer:"""
