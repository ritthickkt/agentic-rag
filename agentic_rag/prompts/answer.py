"""The answering prompt: grounded prose with citations that can be checked.

The product promise is "every claim points at a source you can open." That promise
is only worth anything if the pointer is **verifiable**, so the model is not asked
for a bibliography — it is asked, for each marker, to return the **verbatim quote**
it is leaning on. `citations.py` then checks that quote actually appears in the
passage. A citation whose quote cannot be found is dropped and flagged rather than
shown, because a fabricated citation is strictly worse than none: it looks like
evidence.

**A marker IS a passage number.** There is no separate numbering scheme, and that
is a correctness fix rather than a simplification. The first version of this prompt
let the model choose marker numbers and report which passage each one referred to —
and it promptly produced an answer where the quotes were all individually correct
but the markers were shifted by one relative to the claims they sat next to: `[4]`
appeared on a revenue claim while citation 4 quoted something else entirely. Every
quote verified, because each quote really was in the chunk it named; the mapping
from *claim* to *marker* was what broke, and no server-side check can catch that,
because it requires knowing which claim the author meant. Removing the second
numbering removes the only place that mismatch could live. Markers are therefore not
sequential in the prose — an answer may cite [2], [7], [3] — which is also what the
UI's passage list is numbered by, so they line up.

`insufficient` is part of the same call rather than a separate one. The earlier
version of this project learned that gating the answer behind a separate
"is the evidence good enough?" call made the system decline questions it could
answer — 17.8% of them, with the evidence provably in the context window. Asking
the model to answer *or* declare what is missing, in one call, means declining is a
decision made by something that has already tried to answer.
"""

from __future__ import annotations

from typing import Sequence

from agentic_rag.models import RetrievedChunk

ANSWER_SYSTEM = """You answer questions using only the numbered passages provided.

GROUNDING
- Use only the passages. Do not use outside knowledge, even when you are confident \
it is correct, and do not fill gaps with what is usually true.
- Every factual claim must carry a marker naming THE PASSAGE NUMBER it came from. \
A fact from passage 7 is cited [7]. Do not renumber, do not use [1] for "my first \
citation" — the marker is always the passage's own number.
- Two passages supporting one claim: [3][7].
- Place the marker AFTER the claim it supports, at the end of the sentence or \
clause, before the full stop: "Revenue was $47.2 million [4]." Never put a marker \
before the fact or in the middle of a phrase.
- If passages disagree, say so and cite both rather than silently picking one.

CITATIONS
- In `citations`, add one entry per passage number you cited, with `passage` set \
to that same number and `quote` set to the exact verbatim text from that passage \
which supports the claim. Copy it character for character. Do not paraphrase, do \
not tidy punctuation, do not join text from two places with an ellipsis.
- Quote the shortest span that actually contains the fact, roughly one sentence.

ANSWERING
- Lead with the answer. No preamble, no restating the question, no "based on the \
provided context".
- Be as brief as the question allows: a name, a date, a number, a sentence. Use a \
short list only when the answer genuinely has multiple parts.
- A question asking to compare, summarise, or explain gets prose; a question \
asking for a fact gets the fact.

WHEN THE PASSAGES DO NOT ANSWER IT
- Set `insufficient` to true, leave `answer` empty, and in `missing` name the \
specific fact that is absent — "the passages give the 2025 revenue but not 2024", \
not "insufficient information".
- Partially answerable counts as answerable: answer the part the passages support, \
cite it, and state plainly what they do not cover.
- Do not answer a question the passages merely touch on. A passage about the same \
company is not a passage about the question.

CONFIDENCE
- `high`: the passages state the answer directly.
- `medium`: the answer requires combining passages, or the wording is indirect.
- `low`: the passages support an answer only by inference, or partially."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "insufficient": {
            "type": "boolean",
            "description": "True if the passages cannot answer the question at all.",
        },
        "answer": {
            "type": "string",
            "description": "The answer, with [n] citation markers. Empty if insufficient.",
        },
        "citations": {
            "type": "array",
            "description": "One entry per passage number cited in the answer.",
            "items": {
                "type": "object",
                "properties": {
                    "passage": {
                        "type": "integer",
                        "description": "The passage number, matching the [n] marker in the answer.",
                    },
                    "quote": {
                        "type": "string",
                        "description": "Verbatim span copied from that passage.",
                    },
                },
                "required": ["passage", "quote"],
                "additionalProperties": False,
            },
        },
        "missing": {
            "type": "string",
            "description": "When insufficient or partial: the specific fact that is absent.",
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["insufficient", "answer", "citations", "missing", "confidence"],
    "additionalProperties": False,
}


def format_passages(chunks: Sequence[RetrievedChunk]) -> str:
    """Numbered passages, each labelled with its real source location.

    The location line is shown to the model as well as to the user. It is cheap
    context and it lets the model say "the 2024 handbook says X while the 2025 one
    says Y" instead of treating two conflicting passages as one muddled fact.
    """
    if not chunks:
        return "(no passages were retrieved)"
    blocks = []
    for i, rc in enumerate(chunks, start=1):
        blocks.append(f"[{i}] source: {rc.chunk.location}\n{rc.chunk.text}")
    return "\n\n".join(blocks)


FINAL_ATTEMPT_NOTE = """
This is the final set of passages; no further search will be run. If they support \
even a partial answer, give that answer, cite it, and state in `missing` what is \
still not covered. Only set `insufficient` if they genuinely contain nothing that \
bears on the question."""


def answer_prompt(
    question: str, chunks: Sequence[RetrievedChunk], final_attempt: bool = False
) -> str:
    """`final_attempt` is set on the last hop of the agent loop.

    It exists because of a specific measured bug in the earlier version of this
    project: when the loop ran out of hops, the system abstained unconditionally —
    on items where retrieval had in fact succeeded 84.2% of the time. "Ran out of
    budget" was being reported to the user as "there is no evidence". The note does
    not lower the grounding bar; it removes the incentive to hold out for a
    retrieval that is never going to happen."""
    tail = FINAL_ATTEMPT_NOTE if final_attempt else ""
    return f"""Passages:

{format_passages(chunks)}

Question: {question}

Answer from the passages above, citing each claim.{tail}"""
