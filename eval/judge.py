"""LLM-as-judge: decide whether a free-form answer is right.

Why a judge at all, when we already have exact match and F1: the systems generate
free text, and the gold answers are short spans. `token_f1("Argentina",
"Argentinian")` is **0.0**. So is "1994" vs "in 1994". String metrics systematically
under-credit correct answers, and they under-credit them *unevenly* across systems
— whichever system happens to be terser scores higher for reasons that have
nothing to do with being right. That is a bias with a direction, which is worse
than noise.

Why the judge takes an injected callable (DECISIONS D3, D12): `eval/` must not
import `agentic_rag/`. So this module defines the prompts, the schemas, and the
parsing, and the *script* hands it something that can talk to a model. The harness
stays a pure function of (system, questions, judge) and can be run with a stub
judge in a test.

Known bias, stated rather than hidden (DECISIONS D4): the judge model is currently
the same model that generates the answers, which inflates scores through
self-preference. It is tolerable because we care about movement between systems
that all share the same judge, and `scripts/check_judge.py` measures the residual
disagreement against the string metrics.
"""

from __future__ import annotations

from typing import Any, Protocol


class JsonCompleter(Protocol):
    """Whatever the script injects must look like this.

    `agentic_rag.llm.LLM.complete_json` satisfies it. So does a dict-returning
    stub, which is how the harness gets tested without spending money.
    """

    def __call__(
        self,
        prompt: str,
        schema: dict[str, Any],
        system: str | None = ...,
        max_tokens: int = ...,
    ) -> dict[str, Any]: ...


# --- answerable items: is the answer correct? -------------------------------

GRADE_SYSTEM = """You grade short answers against a reference answer.

Mark an answer `correct` when it conveys the same fact as the reference, even if \
the wording differs. "Argentinian" for "Argentina", "1994" for "in 1994", and \
"Desmond Tutu" for "Desmond Mpilo Tutu" are all correct — the reference is a \
short span, not a required phrasing.

Mark it `incorrect` when it names a different entity, gives a different value, or \
contradicts the reference. Extra correct detail is fine; a wrong specific is not.

Mark it `no_answer` when the response declines, says the information is not in \
the context, hedges without committing, or is empty. Do not judge whether \
declining was the right call — only that the response contains no answer."""

GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "incorrect", "no_answer"]},
        "reason": {"type": "string", "description": "One short clause."},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}


def grade_answer(
    complete_json: JsonCompleter, question: str, gold: str, answer: str
) -> dict[str, Any]:
    if not answer.strip():
        # No call needed, and no call means no judge variance on the easiest case.
        return {"verdict": "no_answer", "reason": "empty response"}
    prompt = f"""Question: {question}

Reference answer: {gold}

Response to grade: {answer}

Grade the response."""
    return complete_json(prompt, GRADE_SCHEMA, GRADE_SYSTEM, 200)


# --- unanswerable items: did it answer at all? ------------------------------

ABSTAIN_SYSTEM = """You decide whether a response commits to an answer or declines \
to answer.

`answered` — it states a fact, name, date, value, or yes/no as the answer, even \
hedged ("probably Argentina").
`no_answer` — it declines, says the context is insufficient, asks for more \
information, or is empty.

There is no reference answer here and you must not judge correctness. Only \
whether an answer was given."""

ABSTAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["answered", "no_answer"]},
        "reason": {"type": "string", "description": "One short clause."},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}


def classify_response(
    complete_json: JsonCompleter, question: str, answer: str
) -> dict[str, Any]:
    """For unanswerable items: answered, or declined?

    This exists because `SystemAnswer.abstained` only records the *structural*
    abstentions — the ones the system took deliberately. A naive system given no
    abstention instruction can still wander off-script and refuse, and counting
    that as "answered" would overstate its hallucination rate.
    """
    if not answer.strip():
        return {"verdict": "no_answer", "reason": "empty response"}
    prompt = f"""Question: {question}

Response: {answer}

Did this response commit to an answer, or decline?"""
    return complete_json(prompt, ABSTAIN_SCHEMA, ABSTAIN_SYSTEM, 200)
