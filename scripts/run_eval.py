"""Run one system over one split and write results/<system>_<split>.json.

    python scripts/run_eval.py --system naive --split dev [--limit 40] [--workers 8]
    python scripts/run_eval.py --system all --split dev

This is the wiring script: it is the only place that knows both how to build a
system (`agentic_rag`) and how to score one (`eval`). Keeping that knowledge here
is what lets `eval/` stay free of any import from `agentic_rag/` (D3).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic_rag.config import CONFIG
from agentic_rag.index import Retriever
from agentic_rag.llm import LLM
from agentic_rag.systems import SYSTEM_NAMES, build_system
from eval.harness import run_system
from eval.judge import classify_response, grade_answer
from eval.types import Question, SystemAnswer, load_questions


def make_judge(judge_llm: LLM):
    """Route each item to the right judge question.

    Answerable items get graded against the gold answer. Unanswerable items have no
    reachable gold (D10), so they are only classified as answered-or-declined —
    the judge is never shown `oracle_answer`, so it cannot leak it into a verdict.
    """

    def judge(q: Question, ans: SystemAnswer) -> dict:
        if q.answerable and q.answer:
            return grade_answer(judge_llm.complete_json, q.question, q.answer, ans.answer)
        return classify_response(judge_llm.complete_json, q.question, ans.answer)

    return judge


# The four benchmark rungs plus the system we actually ship. `product` is scored
# through `agentic_rag.bench.ProductSystem`, over a separate index built by
# `scripts/build_bench_index.py`, because the served engine needs payload provenance
# and a lexical half that the original benchmark index does not carry.
ALL_SYSTEMS = (*SYSTEM_NAMES, "product")


def _build_product_system(system_llm: LLM):
    """Construct the served engine against the benchmark corpus."""
    from agentic_rag.agent import Answerer
    from agentic_rag.bench import ProductSystem
    from agentic_rag.retrieve import Retriever as HybridRetriever
    from agentic_rag.store import VectorStore
    from scripts.build_bench_index import BENCH_COLLECTION

    store = VectorStore(collection=BENCH_COLLECTION)
    if store.count() == 0:
        store.close()
        raise SystemExit(
            f"the {BENCH_COLLECTION!r} collection is empty.\n"
            f"run: python scripts/build_bench_index.py"
        )
    retriever = HybridRetriever(store=store)
    return ProductSystem(Answerer(retriever=retriever, llm=system_llm)), store


def run_one(name: str, questions: list[Question], split: str, retriever: Retriever | None,
            workers: int, no_cache: bool) -> dict:
    # Two separate LLM instances on purpose: the judge's tokens must not land in
    # the system's cost figure, or "agentic costs 3x naive" would be part judge.
    system_llm = LLM(use_cache=not no_cache)
    judge_llm = LLM(use_cache=not no_cache)
    product_store = None
    if name == "product":
        system, product_store = _build_product_system(system_llm)
    else:
        system = build_system(name, retriever, system_llm)

    print(f"\n=== {name} on {split} ({len(questions)} questions, {workers} workers) ===")
    t0 = time.time()

    def progress(done: int, total: int) -> None:
        rate = done / max(time.time() - t0, 1e-9)
        print(f"  {done}/{total}  ({rate:.1f} q/s)", flush=True)

    try:
        run = run_system(system, questions, name=name, split=split,
                         judge=make_judge(judge_llm), workers=workers, on_progress=progress)
    finally:
        if product_store is not None:
            product_store.close()
    run.extra = {
        "config": CONFIG.to_dict(),
        "system_usage": system_llm.usage.to_dict(),
        "judge_usage": judge_llm.usage.to_dict(),
        "wall_clock_s": round(time.time() - t0, 1),
    }
    path = run.save(CONFIG.results_dir / f"{name}_{split}.json")

    m = run.metrics
    print(f"  wrote {path.name} in {run.extra['wall_clock_s']}s "
          f"({len(run.errors)} errors)")
    if not run.is_complete:
        # Loud, before the numbers, because the numbers are not the split's numbers.
        # A 400 from an exhausted credit balance took out 283 of 420 questions once,
        # and the summary printed rates over the surviving 137 that looked exactly
        # like a clean run.
        print(f"  {'!' * 66}")
        print(f"  !! INCOMPLETE RUN — scored {run.n_scored if hasattr(run, 'n_scored') else len(run.items)}"
              f"/{run.n_attempted} questions ({run.completion_rate:.0%})")
        print(f"  !! Every rate below is over {len(run.items)} items, NOT {run.n_attempted}.")
        print(f"  !! Do not compare these numbers against a complete run.")
        print(f"  {'!' * 66}")
    print(f"  answerable   judge_correct {m['answerable']['judge_correct']:.1%}  "
          f"EM {m['answerable']['exact_match']:.1%}  "
          f"recall_all {m['answerable']['recall_all']:.1%}  "
          f"abstain {m['answerable']['abstain_rate']:.1%}")
    print(f"  unanswerable declined {m['unanswerable']['abstain_rate_effective']:.1%}  "
          f"answered_without_evidence {m['unanswerable']['answered_without_evidence']:.1%}")
    print(f"  overall {m['overall_behaviour_score']:.1%}  "
          f"calls/q {m['cost']['llm_calls_per_question']:.2f}  "
          f"system ${run.extra['system_usage']['cost_usd']:.3f} "
          f"+ judge ${run.extra['judge_usage']['cost_usd']:.3f}")
    if name == "product":
        rows = run.items
        cited = [r for r in rows if r.get("n_citations")]
        unver = sum(r.get("citations_unverified", 0) for r in rows)
        total = sum(r.get("n_citations", 0) for r in rows)
        print(f"  citations    {total} total across {len(cited)} answers, "
              f"{unver} unverified ({unver / max(total, 1):.1%})")
    if run.errors:
        print(f"  first error: {run.errors[0]['error']}")
    return run.extra


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="naive",
                    help=f"one of {', '.join(ALL_SYSTEMS)}, or 'all'")
    ap.add_argument("--split", default="dev", choices=("dev", "test"))
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (smoke tests)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-cache", action="store_true", help="bypass the LLM disk cache")
    args = ap.parse_args()

    if args.split == "test":
        # D11: the test split is read once, at the end. Make walking into it an
        # explicit act rather than a tab-completion accident.
        print("!! You are about to read the TEST split (DECISIONS D11).")
        print("!! Every number you look at while building should come from dev.")
        if input("   Type 'yes I mean it' to continue: ").strip() != "yes I mean it":
            print("   aborted")
            return 1

    questions = load_questions(CONFIG.data_dir / f"{args.split}.jsonl")
    if args.limit:
        questions = questions[: args.limit]

    names = list(ALL_SYSTEMS) if args.system == "all" else [args.system]
    for n in names:
        if n not in ALL_SYSTEMS:
            print(f"unknown system {n!r}; choose from {ALL_SYSTEMS}")
            return 1

    # One retriever shared across the benchmark rungs: same index, same embedder, no
    # chance of a config drift between runs (LEARNING step 1's third risk). Built
    # lazily, because `product` does not use it — and constructing it anyway loaded
    # the embedding model a second time and opened a second embedded Qdrant in the
    # same process, which aborted the interpreter outright.
    retriever: Retriever | None = None
    try:
        for n in names:
            if n != "product" and retriever is None:
                retriever = Retriever()
            run_one(n, questions, args.split, retriever, args.workers, args.no_cache)
    finally:
        if retriever is not None:
            retriever.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
