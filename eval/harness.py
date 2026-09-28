"""Run a system over a question set and produce a results file.

This module knows nothing about Qdrant, Anthropic, embeddings, or prompts. It
takes a callable and a list of `Question`s. That is the whole point of D3: when
Phase 2 reports "agentic beats naive", the harness contributed identically to both
numbers, because it is physically incapable of contributing differently.

Concurrency: questions are independent, and wall-clock time is dominated by API
latency, so items run in a thread pool. Results are reassembled in the original
question order — never in completion order — so two runs of the same set produce
byte-comparable files.
"""

from __future__ import annotations

import json
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from eval.metrics import (
    aggregate,
    exact_match,
    gold_fraction,
    recall_all,
    recall_any,
    token_f1,
)
from eval.types import Question, SystemAnswer

# (question, answer) -> {"verdict": ..., "reason": ...}
JudgeFn = Callable[[Question, SystemAnswer], dict[str, Any]]


@dataclass
class Run:
    name: str
    split: str
    items: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    # How many questions the run was *asked* to score, as opposed to how many it
    # managed to. Without this the two are indistinguishable in the output file.
    n_attempted: int = 0

    @property
    def completion_rate(self) -> float:
        return len(self.items) / self.n_attempted if self.n_attempted else 0.0

    @property
    def is_complete(self) -> bool:
        return self.n_attempted > 0 and len(self.items) == self.n_attempted

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "split": self.split,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            # First-class, and named so it cannot be skimmed past. A run that scored
            # 137 of 420 questions produces rates over 137 items, and those rates are
            # not the split's rates — the denominator shrank. `run_system` already
            # recorded every failure, but the printed summary looked identical to a
            # clean run, which is the same class of error as D6: a green light that
            # did not test the thing it claimed to.
            "complete": self.is_complete,
            "n_attempted": self.n_attempted,
            "n_scored": len(self.items),
            "completion_rate": round(self.completion_rate, 4),
            "metrics": self.metrics,
            "n_errors": len(self.errors),
            "errors": self.errors[:20],
            **self.extra,
            "items": self.items,
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        return path


def score_item(q: Question, ans: SystemAnswer, judge: dict[str, Any] | None) -> dict[str, Any]:
    """One row of the results file. Everything a later question can be asked of."""
    retrieved = [p.doc_id for p in ans.passages]
    meta = ans.meta or {}

    # String metrics only make sense against a reachable gold answer. For an
    # unanswerable item the gold is unreachable by construction, so EM/F1 against
    # it would be scoring the system on a hallucination target (D10). The same
    # comparison is kept under `oracle_f1`, labelled as a diagnostic.
    em = f1 = 0.0
    if q.answerable and q.answer:
        em = exact_match(ans.answer, q.answer)
        f1 = token_f1(ans.answer, q.answer)

    return {
        "qid": q.qid,
        "question": q.question,
        "answerable": q.answerable,
        "gold_answer": q.answer,
        "gold_titles": list(q.gold_titles),
        "qtype": q.qtype,
        "answer": ans.answer,
        "abstained": ans.abstained,
        "retrieved": retrieved,
        "em": em,
        "f1": f1,
        "oracle_f1": token_f1(ans.answer, q.oracle_answer) if ans.answer else 0.0,
        "recall_any": recall_any(retrieved, q.gold_titles),
        "recall_all": recall_all(retrieved, q.gold_titles),
        "gold_fraction": gold_fraction(retrieved, q.gold_titles),
        "judge_verdict": (judge or {}).get("verdict"),
        "judge_reason": (judge or {}).get("reason"),
        "n_llm_calls": meta.get("n_llm_calls", 0),
        "hops": meta.get("hops", 1),
        "queries": meta.get("queries", []),
        "input_tokens": meta.get("input_tokens", 0),
        "output_tokens": meta.get("output_tokens", 0),
        "latency_s": meta.get("latency_s", 0.0),
        "grader_verdicts": meta.get("grader_verdicts"),
        "abstained_by": meta.get("abstained_by"),
        "raw_answer": meta.get("raw_answer"),
        # Citation health, for systems that produce citations. Carried through as
        # plain passthrough: `eval/` does not know what a citation is, it just does
        # not drop fields a system chose to report.
        "n_citations": meta.get("n_citations"),
        "citations_verified": meta.get("citations_verified"),
        "citations_relocated": meta.get("citations_relocated"),
        "citations_unverified": meta.get("citations_unverified"),
        "confidence": meta.get("confidence"),
    }


def run_system(
    system: Callable[[str], SystemAnswer],
    questions: list[Question],
    name: str,
    split: str,
    judge: JudgeFn | None = None,
    workers: int = 8,
    on_progress: Callable[[int, int], None] | None = None,
) -> Run:
    run = Run(name=name, split=split)
    results: list[dict[str, Any] | None] = [None] * len(questions)

    def work(i_q: tuple[int, Question]) -> None:
        i, q = i_q
        try:
            ans = system(q.question)
            verdict = judge(q, ans) if judge else None
            results[i] = score_item(q, ans, verdict)
        except Exception as e:  # noqa: BLE001
            # One question failing must not take the run down — but it must also
            # never be silently dropped, or the denominator quietly shrinks and
            # every rate in the report is wrong. Record it and count it.
            run.errors.append({
                "qid": q.qid,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=3),
            })

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, _ in enumerate(pool.map(work, enumerate(questions)), start=1):
            if on_progress and (n % 20 == 0 or n == len(questions)):
                on_progress(n, len(questions))

    run.items = [r for r in results if r is not None]
    run.n_attempted = len(questions)
    run.metrics = aggregate(run.items)
    return run
