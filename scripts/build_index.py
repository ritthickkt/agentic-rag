"""Embed data/corpus.jsonl into the Qdrant collection.

Run once after build_dataset.py. Reports a retrieval sanity check at the end:
recall@k of the gold paragraphs over the dev questions. That number is not a
result — it is a smoke test. If it came out near zero, something is wired wrong
(most likely the query prefix in embed.py) and no downstream number would mean
anything.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic_rag.config import CONFIG
from agentic_rag.index import Retriever
from eval.types import load_documents, load_questions


def main() -> int:
    corpus_path = CONFIG.data_dir / "corpus.jsonl"
    if not corpus_path.exists():
        print(f"missing {corpus_path} — run scripts/build_dataset.py first")
        return 1

    docs = load_documents(corpus_path)
    print(f"--- indexing {len(docs)} documents ---")
    print(f"  embed model: {CONFIG.embed_model}")
    print(f"  collection:  {CONFIG.collection} at {CONFIG.qdrant_path}")

    t0 = time.time()
    r = Retriever()
    n = r.build(docs)
    print(f"  indexed {n} points in {time.time() - t0:.1f}s")

    print("\n--- smoke test: recall@k on dev (not a result, a wiring check) ---")
    dev = [q for q in load_questions(CONFIG.data_dir / "dev.jsonl") if q.answerable]
    sample = dev[:100]
    hit_any = hit_all = 0
    for q in sample:
        got = {p.doc_id for p in r.search(q.question, k=CONFIG.top_k)}
        gold = set(q.gold_titles)
        hit_any += bool(got & gold)
        hit_all += gold <= got
    print(f"  k={CONFIG.top_k}, n={len(sample)} answerable dev questions")
    print(f"  at least one gold retrieved: {hit_any / len(sample):.1%}")
    print(f"  both golds retrieved:        {hit_all / len(sample):.1%}")
    if hit_any / len(sample) < 0.30:
        print("\nFAIL — recall this low means a wiring bug, not a hard dataset")
        return 1
    r.close()
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
