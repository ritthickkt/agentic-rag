"""Query reformulation: turn "what's missing" into a better search query.

Why this step exists at all: the smoke test in `scripts/build_index.py` found that
a single dense query over the original question retrieves *at least one* gold
paragraph 100% of the time but *both* only ~73% of the time. That is the
signature of multi-hop failure — the question's wording looks like hop one, so
hop two's document never surfaces. Rewriting the query to describe the missing
fact is the cheapest available fix.

The prompt deliberately asks for keyword-style search text rather than a polite
question, because the retriever is a dense embedding model over encyclopedia
paragraphs, and those paragraphs are written as declarative statements.
"""

from __future__ import annotations

REFORMULATE_SYSTEM = """You write search queries for a document retrieval system \
over encyclopedia articles.

Given a question, the passages already retrieved, and the fact still missing, \
write one search query aimed at the MISSING fact.

Rules:
- Target the missing fact, not the part already covered. Repeating the original \
question will return the same passages again and waste the hop.
- Use the specific entity names discovered in the passages already retrieved. \
Resolving a vague reference ("the director of that film") into a concrete name is \
the single most valuable thing you can do here.
- Write it like the sentence you expect to find, not like a question."""

REFORMULATE_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "The new search query"},
    },
    "required": ["query"],
    "additionalProperties": False,
}


def reformulate_prompt(question: str, context: str, missing: str) -> str:
    """`context` is the already-formatted passage block.

    Taking a string rather than a list of passages is what lets one prompt serve
    both the served application (which retrieves `RetrievedChunk`s with page
    numbers) and the benchmark (which retrieves bare `Passage`s). The prompt is the
    thing being held constant across those two; the formatting is not.
    """
    return f"""Original question: {question}

Passages already retrieved:

{context}

Still missing: {missing}

Write one search query to find the missing fact."""
