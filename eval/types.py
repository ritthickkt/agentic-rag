"""The contract between a *system* and the *harness*.

This module is the whole reason `eval/` is a sibling of `agentic_rag/` instead of
a submodule (DECISIONS D3). Everything here is plain data. Nothing in this file
knows that Qdrant, Anthropic, or embeddings exist, and nothing in here may import
`agentic_rag`.

A system is any callable `str -> SystemAnswer`. That is the entire interface.
Naive RAG satisfies it; so will the agentic version; so does a stub that always
abstains (which is a genuinely useful control).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol


# --- evidence ---------------------------------------------------------------


@dataclass(frozen=True)
class Passage:
    """One retrieved chunk, as the system saw it.

    `doc_id` is what retrieval-quality metrics compare against a question's
    `gold_titles`, so it must be the corpus id, not a paraphrase of it.
    """

    doc_id: str
    text: str
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- what a system returns --------------------------------------------------


@dataclass(frozen=True)
class SystemAnswer:
    """A system's response: the text, the evidence behind it, and bookkeeping.

    `abstained` is recorded separately rather than sniffed out of `answer` with a
    regex. A metric that has to guess whether "I don't know" counts as an
    abstention is a metric measuring string formatting, not behaviour.
    """

    answer: str
    passages: tuple[Passage, ...] = ()
    abstained: bool = False
    # Free-form and never scored: n_llm_calls, prompt_tokens, latency_s, trace…
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "passages": [p.to_dict() for p in self.passages],
            "abstained": self.abstained,
            "meta": self.meta,
        }


class System(Protocol):
    """Structural type for anything the harness can evaluate."""

    def __call__(self, question: str) -> SystemAnswer: ...


SystemFn = Callable[[str], SystemAnswer]


# --- what the harness reads -------------------------------------------------


@dataclass(frozen=True)
class Question:
    """One labelled eval item, as written by `scripts/build_dataset.py`.

    `answer` is None exactly when `answerable` is False: for those items the
    corpus has had the supporting paragraphs removed, so the only correct
    behaviour is to abstain.

    `oracle_answer` keeps the original HotpotQA string even for unanswerable
    items, purely so a human can eyeball whether a system guessed right anyway.
    **No metric may read it.** Scoring an unanswerable item against a gold answer
    that is unreachable from the corpus is how you accidentally reward
    hallucination.
    """

    qid: str
    question: str
    answer: str | None
    answerable: bool
    gold_titles: tuple[str, ...]
    oracle_answer: str
    level: str = ""
    qtype: str = ""
    removed_titles: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["gold_titles"] = list(self.gold_titles)
        d["removed_titles"] = list(self.removed_titles)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Question":
        return cls(
            qid=d["qid"],
            question=d["question"],
            answer=d["answer"],
            answerable=d["answerable"],
            gold_titles=tuple(d["gold_titles"]),
            oracle_answer=d["oracle_answer"],
            level=d.get("level", ""),
            qtype=d.get("qtype", ""),
            removed_titles=tuple(d.get("removed_titles", ())),
        )


@dataclass(frozen=True)
class Document:
    """One corpus paragraph. `doc_id` is the HotpotQA article title."""

    doc_id: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Document":
        return cls(doc_id=d["doc_id"], text=d["text"])


# --- jsonl helpers ----------------------------------------------------------


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    n = 0
    with Path(path).open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_questions(path: Path) -> list[Question]:
    return [Question.from_dict(d) for d in read_jsonl(path)]


def load_documents(path: Path) -> list[Document]:
    return [Document.from_dict(d) for d in read_jsonl(path)]
