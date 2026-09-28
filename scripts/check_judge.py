"""Audit the LLM judge against the string metrics (DECISIONS D4).

The judge is the one number in this project that depends on a model's opinion, and
it is the same model family that produced the answers, so it is the most likely
place for a silent bias to live. This script does not assume the judge is right —
it finds every case where the judge and exact match disagree and prints them, so a
human can read a sample and decide.

Two disagreement classes, and they mean different things:

  judge=correct but F1=0.0  — the judge accepted an answer with no token overlap.
      Usually legitimate ("Argentinian"/"Argentina"), which is the entire reason
      the judge exists. But it is also exactly what a rubber-stamp judge looks
      like, so these must be eyeballed rather than trusted.

  judge=incorrect but EM=1.0 — the judge rejected an exact string match. Almost
      always a judge error, and the count should be near zero.

    python scripts/check_judge.py --system naive --split dev [--show 15]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic_rag.config import CONFIG
from eval.metrics import token_f1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="naive")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--show", type=int, default=12)
    args = ap.parse_args()

    path = CONFIG.results_dir / f"{args.system}_{args.split}.json"
    if not path.exists():
        print(f"missing {path}")
        return 1
    run = json.loads(path.read_text(encoding="utf-8"))
    ans = [i for i in run["items"] if i["answerable"] and not i["abstained"] and i["answer"]]

    lenient = [i for i in ans if i["judge_verdict"] == "correct" and i["f1"] == 0.0]
    harsh = [i for i in ans if i["judge_verdict"] == "incorrect" and i["em"] == 1.0]
    agree = [i for i in ans if (i["judge_verdict"] == "correct") == (i["f1"] > 0.5)]

    print(f"=== judge audit: {args.system} / {args.split} ===")
    print(f"  {len(ans)} answerable items with a committed answer")
    print(f"  judge agrees with F1>0.5:        {len(agree) / len(ans):.1%}")
    print(f"  judge=correct   but F1 == 0.0:   {len(lenient):4d}  ({len(lenient) / len(ans):.1%})")
    print(f"  judge=incorrect but EM == 1.0:   {len(harsh):4d}  ({len(harsh) / len(ans):.1%})")
    if len(harsh) / max(len(ans), 1) > 0.02:
        print("  !! more than 2% of exact matches rejected — inspect the judge prompt")

    for label, rows in (("judge=correct, F1=0 (read these)", lenient),
                        ("judge=incorrect, EM=1 (should be ~0)", harsh)):
        if not rows:
            continue
        print(f"\n--- {label} ---")
        for i in rows[: args.show]:
            print(f"  Q    {i['question'][:100]}")
            print(f"  gold {i['gold_answer']!r}")
            print(f"  got  {i['answer'][:90]!r}   (F1 {token_f1(i['answer'], i['gold_answer']):.2f})")
            print(f"  why  {i['judge_reason'][:100]}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
