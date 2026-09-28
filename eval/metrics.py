"""Scoring. No LLM calls in this file — these are all string and set operations.

Split deliberately from `judge.py`: the metrics here are cheap, deterministic, and
reproducible by anyone with the results file. The judge is none of those things.
Keeping them apart means a reader can tell at a glance which numbers depend on a
model's opinion and which do not.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Iterable


# --- answer string matching (the standard SQuAD normalisation) ---------------


def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    """Bag-of-tokens F1 — partial credit for "Desmond Tutu" vs "Desmond Mpilo Tutu".

    Exact match alone is unfairly harsh on free-form generation: a model that
    answers correctly but with one extra middle name scores zero. F1 is why we
    report both.
    """
    p_toks = normalize_answer(pred).split()
    g_toks = normalize_answer(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = Counter(p_toks) & Counter(g_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(p_toks)
    recall = n_same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


# --- retrieval ---------------------------------------------------------------


def recall_any(retrieved: Iterable[str], gold: Iterable[str]) -> float:
    r, g = set(retrieved), set(gold)
    return float(bool(r & g))


def recall_all(retrieved: Iterable[str], gold: Iterable[str]) -> float:
    """The one that matters for multi-hop: did we get *every* gold paragraph?

    `recall_any` looks great on HotpotQA and means little — one of the two hops is
    usually lexically close to the question. Answering needs both.
    """
    r, g = set(retrieved), set(gold)
    return float(bool(g) and g <= r)


def gold_fraction(retrieved: Iterable[str], gold: Iterable[str]) -> float:
    r, g = set(retrieved), set(gold)
    return len(r & g) / len(g) if g else 0.0


# --- aggregation -------------------------------------------------------------


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def effective_abstention(row: dict[str, Any]) -> bool:
    """Did the system decline, by any route?

    `abstained` is the *structural* flag: the system deliberately declined, via the
    abstention token or the grader gate. But a system given no abstention
    instruction can still go off-script and refuse in prose, and counting that as
    "answered" would overstate its hallucination rate — flattering the systems we
    want to beat is as much a distortion as flattering the one we built.

    So the headline rate is "declined by any route", and the structural rate is
    reported next to it. The gap between them is itself informative: it measures
    how much of a system's abstention is instruction-following versus improvisation.
    """
    return bool(row.get("abstained")) or row.get("judge_verdict") == "no_answer"


def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll per-item records up into the report.

    The answerable and unanswerable buckets are scored on different things and are
    never averaged into one headline number. A single "accuracy" over both would
    let a system that abstains on everything look respectable, and a system that
    answers everything look respectable, while neither is doing the job.
    """
    ans = [it for it in items if it["answerable"]]
    unans = [it for it in items if not it["answerable"]]

    def bucket_retrieval(rows: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "recall_any": _mean([r["recall_any"] for r in rows]),
            "recall_all": _mean([r["recall_all"] for r in rows]),
            "gold_fraction": _mean([r["gold_fraction"] for r in rows]),
        }

    out: dict[str, Any] = {
        "n": len(items),
        "n_answerable": len(ans),
        "n_unanswerable": len(unans),
        "answerable": {
            **bucket_retrieval(ans),
            # Scored over ALL answerable items, with an abstention counted as a
            # miss — declining a question you could have answered is a failure,
            # and dropping those rows would reward over-abstention.
            "exact_match": _mean([r["em"] for r in ans]),
            "token_f1": _mean([r["f1"] for r in ans]),
            "judge_correct": _mean([float(r.get("judge_verdict") == "correct") for r in ans]),
            "abstain_rate": _mean([float(r["abstained"]) for r in ans]),
            "off_script_no_answer": _mean(
                [float(r.get("judge_verdict") == "no_answer" and not r["abstained"]) for r in ans]
            ),
        },
        "unanswerable": {
            # The headline for this bucket: the corpus cannot support an answer, so
            # declining is correct and anything else is a hallucination.
            "abstain_rate_effective": _mean([float(effective_abstention(r)) for r in unans]),
            "abstain_rate_structural": _mean([float(r["abstained"]) for r in unans]),
            "answered_without_evidence": _mean(
                [float(not effective_abstention(r)) for r in unans]
            ),
            "off_script_no_answer": _mean(
                [float(r.get("judge_verdict") == "no_answer" and not r["abstained"]) for r in unans]
            ),
            # DIAGNOSTIC ONLY (DECISIONS D10). When the system answered anyway,
            # how often did it match the answer we deleted the evidence for? A high
            # value means parametric memory is doing the work. This is never a
            # score — matching here is still a grounding failure.
            "lucky_guess_f1_when_answered": _mean(
                [r["oracle_f1"] for r in unans if not effective_abstention(r)]
            ),
        },
        "cost": {
            "llm_calls_per_question": _mean([float(r["n_llm_calls"]) for r in items]),
            "hops_per_question": _mean([float(r.get("hops", 1)) for r in items]),
            "input_tokens_per_question": _mean([float(r.get("input_tokens", 0)) for r in items]),
            "output_tokens_per_question": _mean([float(r.get("output_tokens", 0)) for r in items]),
            "latency_s_per_question": _mean([float(r.get("latency_s", 0)) for r in items]),
        },
    }

    # One combined number, defined explicitly rather than implied by an average:
    # got the answerable ones right AND declined the unanswerable ones.
    correct = [float(r.get("judge_verdict") == "correct") for r in ans]
    declined = [float(effective_abstention(r)) for r in unans]
    out["overall_behaviour_score"] = _mean(correct + declined)
    return out
