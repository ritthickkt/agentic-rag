"""Build the comparison table across results files, plus the diagnostics that
explain *why* the table looks the way it does.

    python scripts/compare.py [--split dev]

The diagnostics are the point. A table of four overall scores tells you which
system won; it does not tell you which component is the bottleneck, and without
that the next step is guesswork.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic_rag.config import CONFIG
from agentic_rag.systems import SYSTEM_NAMES

# The benchmark ladder, then the system actually shipped. `product` is listed last
# because it is not another rung: it differs from `agentic` in several ways at once
# (hybrid retrieval, combined answer/grade call, cited prose), so its column is "is
# the thing we ship better than the thing we measured", not the value of one change.
ALL_NAMES = (*SYSTEM_NAMES, "product")


def load_runs(split: str) -> dict[str, dict]:
    runs = {}
    for name in ALL_NAMES:
        path = CONFIG.results_dir / f"{name}_{split}.json"
        if not path.exists():
            continue
        run = json.loads(path.read_text(encoding="utf-8"))
        # A run that did not score its whole split has a smaller denominator than the
        # others, so putting it in the same table would compare two different
        # questions. Refuse it rather than print an asterisk nobody reads.
        if run.get("complete") is False:
            scored, attempted = run.get("n_scored", "?"), run.get("n_attempted", "?")
            print(f"SKIPPING {name}: incomplete run ({scored}/{attempted} questions "
                  f"scored). Re-run it before comparing.", file=sys.stderr)
            continue
        runs[name] = run
    return runs


def pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def table(runs: dict[str, dict]) -> None:
    rows = [
        ("answerable: judge correct", lambda m: m["answerable"]["judge_correct"]),
        ("answerable: exact match", lambda m: m["answerable"]["exact_match"]),
        ("answerable: token F1", lambda m: m["answerable"]["token_f1"]),
        ("answerable: recall_all (both golds)", lambda m: m["answerable"]["recall_all"]),
        ("answerable: abstained (a miss here)", lambda m: m["answerable"]["abstain_rate"]),
        ("unanswerable: declined", lambda m: m["unanswerable"]["abstain_rate_effective"]),
        ("unanswerable: answered anyway", lambda m: m["unanswerable"]["answered_without_evidence"]),
        ("OVERALL behaviour score", lambda m: m["overall_behaviour_score"]),
    ]
    names = list(runs)
    w = 37
    print(f"\n{'metric':<{w}} " + " ".join(f"{n:>14}" for n in names))
    print("-" * (w + 15 * len(names)))
    for label, fn in rows:
        if label.startswith("OVERALL"):
            print("-" * (w + 15 * len(names)))
        print(f"{label:<{w}} " + " ".join(f"{pct(fn(runs[n]['metrics'])):>14}" for n in names))

    print(f"\n{'cost':<{w}} " + " ".join(f"{n:>14}" for n in names))
    print("-" * (w + 15 * len(names)))
    for label, key in (("LLM calls / question", "llm_calls_per_question"),
                       ("retrieval hops / question", "hops_per_question"),
                       ("input tokens / question", "input_tokens_per_question"),
                       ("latency s / question", "latency_s_per_question")):
        print(f"{label:<{w}} " + " ".join(
            f"{runs[n]['metrics']['cost'][key]:>14.2f}" for n in names))
    print(f"{'system $ for the run':<{w}} " + " ".join(
        f"{runs[n]['system_usage']['cost_usd']:>14.3f}" for n in names))


def diagnose_grader(runs: dict[str, dict]) -> None:
    """The question the table raises: where does the retrieval gain go?

    If a system retrieves both gold paragraphs and then abstains anyway, the
    evidence was there and the grader threw it away. That is a grader
    false negative, and it is measurable exactly.
    """
    print("\n\n=== Why the retrieval gain does not convert ===")
    print("Answerable items only. 'evidence present' = both gold paragraphs retrieved.\n")
    hdr = f"{'system':<16}{'evid present':>13}{'abstained':>11}{'FALSE NEG':>11}{'correct|evid':>14}"
    print(hdr)
    print("-" * len(hdr))
    for name, run in runs.items():
        ans = [i for i in run["items"] if i["answerable"]]
        with_ev = [i for i in ans if i["recall_all"] == 1.0]
        abst = [i for i in ans if i["abstained"]]
        # The number that matters: had the evidence, declined anyway.
        false_neg = [i for i in with_ev if i["abstained"]]
        correct_given_ev = [i for i in with_ev if i["judge_verdict"] == "correct"]
        print(f"{name:<16}{len(with_ev) / len(ans):>12.1%}"
              f"{len(abst) / len(ans):>11.1%}"
              f"{len(false_neg) / len(with_ev):>11.1%}"
              f"{len(correct_given_ev) / len(with_ev):>14.1%}")

    print("\n  FALSE NEG   = of answerable items where BOTH gold paragraphs were")
    print("                retrieved, the share the system declined anyway.")
    print("                Every one of these is a question it could have answered.")
    print("  correct|evid = accuracy when the evidence was definitely present.")


def diagnose_hops(runs: dict[str, dict]) -> None:
    run = runs.get("agentic")
    if not run:
        return
    print("\n\n=== Did the extra hops earn their cost? (agentic, answerable) ===")
    ans = [i for i in run["items"] if i["answerable"]]
    by_hops: dict[int, list] = {}
    for i in ans:
        by_hops.setdefault(i["hops"], []).append(i)
    hdr = f"{'hops':>5}{'n':>6}{'recall_all':>12}{'judge correct':>15}{'abstained':>11}"
    print(hdr)
    print("-" * len(hdr))
    for h in sorted(by_hops):
        rows = by_hops[h]
        print(f"{h:>5}{len(rows):>6}"
              f"{sum(r['recall_all'] for r in rows) / len(rows):>11.1%}"
              f"{sum(r['judge_verdict'] == 'correct' for r in rows) / len(rows):>14.1%}"
              f"{sum(r['abstained'] for r in rows) / len(rows):>10.1%}")
    print("\n  Items needing more hops are the harder ones by construction, so these")
    print("  rows are not comparable to each other — read them against the naive")
    print("  baseline's recall_all, not against hop 1.")


def compare_to_naive(runs: dict[str, dict]) -> None:
    if "naive" not in runs:
        return
    base = runs["naive"]["metrics"]
    print("\n\n=== Movement vs the naive baseline (percentage points) ===")
    hdr = f"{'system':<16}{'answerable':>12}{'unanswerable':>14}{'overall':>10}{'cost x':>9}"
    print(hdr)
    print("-" * len(hdr))
    base_calls = base["cost"]["llm_calls_per_question"]
    for name, run in runs.items():
        if name == "naive":
            continue
        m = run["metrics"]
        d_ans = (m["answerable"]["judge_correct"] - base["answerable"]["judge_correct"]) * 100
        d_un = (m["unanswerable"]["abstain_rate_effective"]
                - base["unanswerable"]["abstain_rate_effective"]) * 100
        d_all = (m["overall_behaviour_score"] - base["overall_behaviour_score"]) * 100
        ratio = m["cost"]["llm_calls_per_question"] / base_calls
        print(f"{name:<16}{d_ans:>+11.1f}{d_un:>+14.1f}{d_all:>+10.1f}{ratio:>8.2f}x")


def diagnose_citations(runs: dict[str, dict]) -> None:
    """Citation health, for systems that produce citations.

    Accuracy alone cannot see this. A system can get steadily better at answering
    while getting steadily worse at pointing to where the answer came from, and the
    unverified rate is the only thing that shows it. Reported as a rate over citations,
    not over questions, because one answer may carry four.
    """
    cited = {n: r for n, r in runs.items()
             if any(i.get("n_citations") for i in r["items"])}
    if not cited:
        return
    print("\n\n=== Citation health ===")
    hdr = (f"{'system':<16}{'answers cited':>15}{'cites/answer':>14}"
           f"{'verified':>10}{'relocated':>11}{'UNVERIFIED':>12}")
    print(hdr)
    print("-" * len(hdr))
    for name, run in cited.items():
        items = run["items"]
        answered = [i for i in items if (i.get("n_citations") or 0) > 0]
        total = sum(i.get("n_citations") or 0 for i in items)
        ver = sum(i.get("citations_verified") or 0 for i in items)
        rel = sum(i.get("citations_relocated") or 0 for i in items)
        unv = sum(i.get("citations_unverified") or 0 for i in items)
        committed = [i for i in items if not i["abstained"] and i["answer"].strip()]
        print(f"{name:<16}{len(answered) / max(len(committed), 1):>14.1%}"
              f"{total / max(len(answered), 1):>14.2f}"
              f"{ver / max(total, 1):>9.1%}{rel / max(total, 1):>11.1%}"
              f"{unv / max(total, 1):>12.1%}")
    print("\n  verified   = the quote was found in the passage the model named.")
    print("  relocated  = found in a different retrieved passage; marker repointed.")
    print("  UNVERIFIED = found nowhere. A citation that cannot be checked is the")
    print("               one failure a reader cannot catch without reading the source.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=("dev", "test"))
    args = ap.parse_args()

    runs = load_runs(args.split)
    if not runs:
        print(f"no results for split {args.split}; run scripts/run_eval.py first")
        return 1
    n = {r["metrics"]["n"] for r in runs.values()}
    print(f"=== {args.split}: {len(runs)} systems, n={n} questions ===")
    if len(n) > 1:
        print("!! WARNING: runs cover different numbers of questions; not comparable")

    table(runs)
    compare_to_naive(runs)
    diagnose_grader(runs)
    diagnose_citations(runs)
    diagnose_hops(runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
