"""A run must be able to say how much of its split it actually scored.

This exists because of a real incident: 283 of 420 questions failed with a 400 from an
exhausted API balance, the harness recorded every failure faithfully, and the printed
summary still looked exactly like a clean run. The rates were over the surviving 137
items — a different denominator, and therefore a different question — and nothing in
the output said so.
"""

from __future__ import annotations

from eval.harness import Run, run_system
from eval.types import Question, SystemAnswer


def q(i: int, answerable: bool = True) -> Question:
    return Question(
        qid=f"q{i}", question=f"question {i}",
        answer="gold" if answerable else None, answerable=answerable,
        gold_titles=("T",), oracle_answer="gold",
    )


def test_complete_run_is_marked_complete():
    questions = [q(i) for i in range(5)]
    run = run_system(lambda s: SystemAnswer(answer="gold"), questions,
                     name="stub", split="dev", workers=2)
    assert run.is_complete is True
    assert run.completion_rate == 1.0
    d = run.to_dict()
    assert d["complete"] is True and d["n_scored"] == d["n_attempted"] == 5


def test_partial_run_is_marked_incomplete_with_the_real_denominator():
    questions = [q(i) for i in range(10)]

    def flaky(question: str) -> SystemAnswer:
        if int(question.split()[-1]) % 2:
            raise RuntimeError("credit balance is too low")
        return SystemAnswer(answer="gold")

    run = run_system(flaky, questions, name="stub", split="dev", workers=2)

    assert run.is_complete is False
    assert run.n_attempted == 10
    assert len(run.items) == 5
    assert run.completion_rate == 0.5
    d = run.to_dict()
    assert d["complete"] is False
    assert d["n_attempted"] == 10 and d["n_scored"] == 5
    assert d["n_errors"] == 5


def test_metrics_are_computed_over_scored_items_only():
    """Not a defect — but it is exactly why `complete` has to be in the file."""
    questions = [q(i) for i in range(4)]

    def half(question: str) -> SystemAnswer:
        if question.endswith(("2", "3")):
            raise RuntimeError("boom")
        return SystemAnswer(answer="gold")

    run = run_system(half, questions, name="stub", split="dev", workers=1)
    assert run.metrics["n"] == 2
    assert run.metrics["n"] != run.n_attempted


def test_empty_run_is_not_complete():
    run = Run(name="stub", split="dev")
    assert run.is_complete is False
    assert run.completion_rate == 0.0
