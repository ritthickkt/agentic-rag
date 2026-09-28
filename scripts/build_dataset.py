"""Build the eval set: a corpus of paragraphs + dev/test question files.

Source: HotpotQA `distractor`, validation split. Each source item ships its own
10 paragraphs (2 supporting, 8 distractors), so a corpus assembled from the items
we keep is guaranteed to contain the evidence for every answerable question.

How an *unanswerable* item is made
----------------------------------
Not by writing a weird question. We take a real HotpotQA question and **delete its
two supporting paragraphs from the corpus**. The question stays perfectly
sensible; the evidence simply is not there, so the only correct behaviour is to
abstain. That is the failure mode worth measuring, and it cannot be faked by a
system that pattern-matches on question wording.

The one hazard, and why the selection loop looks like it does
-------------------------------------------------------------
HotpotQA titles are shared across items. A paragraph deleted to make question A
unanswerable might be the *gold evidence* for question B, which would silently
turn B into an unanswerable item still labelled answerable — a corrupted label,
the worst kind of bug in an eval set. So selection maintains two disjoint sets
(`excluded` and `required`) and refuses any item that would cross them. The
invariant holds by construction, and is then re-checked against the files on disk.

Usage:  python scripts/build_dataset.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic_rag.config import CONFIG
from eval.types import Document, Question, load_documents, load_questions, write_jsonl

HF_DATASET = "hotpotqa/hotpot_qa"
HF_CONFIG = "distractor"
HF_SPLIT = "validation"


def paragraph_text(sentences: list[str]) -> str:
    return " ".join(s.strip() for s in sentences if s.strip()).strip()


def select_items(ds, n_questions: int, unanswerable_frac: float, seed: int):
    """Greedy single pass that cannot produce a corrupted label.

    Returns (items, excluded_titles) where each item is
    (source_example, answerable: bool, gold_titles: tuple[str, ...]).
    """
    n_unanswerable = round(n_questions * unanswerable_frac)
    n_answerable = n_questions - n_unanswerable

    order = list(range(len(ds)))
    random.Random(seed).shuffle(order)

    excluded: set[str] = set()   # titles deleted from the corpus
    required: set[str] = set()   # titles that must stay in the corpus
    items: list[tuple[dict, bool, tuple[str, ...]]] = []
    n_ans = n_unans = 0
    skipped_conflict = skipped_malformed = 0

    for idx in order:
        if n_ans >= n_answerable and n_unans >= n_unanswerable:
            break
        ex = ds[idx]
        gold = tuple(dict.fromkeys(ex["supporting_facts"]["title"]))  # dedup, keep order
        titles = ex["context"]["title"]
        # A handful of HotpotQA items reference a supporting title that is absent
        # from their own context. Unusable either way: skip rather than paper over.
        if len(gold) != 2 or any(t not in titles for t in gold):
            skipped_malformed += 1
            continue

        goldset = set(gold)
        if n_unans < n_unanswerable and not (goldset & required):
            items.append((ex, False, gold))
            excluded |= goldset
            n_unans += 1
        elif n_ans < n_answerable and not (goldset & excluded):
            items.append((ex, True, gold))
            required |= goldset
            n_ans += 1
        else:
            skipped_conflict += 1

    if n_ans < n_answerable or n_unans < n_unanswerable:
        raise RuntimeError(
            f"exhausted the source split: got {n_ans}/{n_answerable} answerable, "
            f"{n_unans}/{n_unanswerable} unanswerable"
        )

    print(f"  selected {n_ans} answerable + {n_unans} unanswerable")
    print(f"  skipped  {skipped_conflict} (title conflict), {skipped_malformed} (malformed)")
    return items, excluded


def build_corpus(items, excluded: set[str]) -> tuple[list[Document], int, int]:
    """Union of every kept item's 10 paragraphs, minus the deleted titles."""
    by_title: dict[str, str] = {}
    n_deleted = n_title_clash = 0
    for ex, _answerable, _gold in items:
        for title, sentences in zip(ex["context"]["title"], ex["context"]["sentences"]):
            if title in excluded:
                n_deleted += 1
                continue
            text = paragraph_text(sentences)
            if not text:
                continue
            if title in by_title:
                # Same title, different text across source items. Keep the first
                # (walk order is deterministic) so doc_id stays a unique key.
                if by_title[title] != text:
                    n_title_clash += 1
                continue
            by_title[title] = text
    docs = [Document(doc_id=t, text=x) for t, x in sorted(by_title.items())]
    return docs, n_deleted, n_title_clash


def to_questions(items, excluded: set[str]) -> list[Question]:
    out: list[Question] = []
    for ex, answerable, gold in items:
        out.append(
            Question(
                qid=ex["id"],
                question=ex["question"].strip(),
                answer=ex["answer"].strip() if answerable else None,
                answerable=answerable,
                gold_titles=gold,
                oracle_answer=ex["answer"].strip(),
                level=ex.get("level", ""),
                qtype=ex.get("type", ""),
                removed_titles=() if answerable else gold,
            )
        )
    return out


def split_dev_test(questions: list[Question], test_frac: float, seed: int):
    """Stratified on `answerable`, so both splits carry the same abstain rate."""
    rng = random.Random(seed + 1)
    dev: list[Question] = []
    test: list[Question] = []
    for flag in (True, False):
        bucket = [q for q in questions if q.answerable is flag]
        rng.shuffle(bucket)
        n_test = round(len(bucket) * test_frac)
        test += bucket[:n_test]
        dev += bucket[n_test:]
    rng.shuffle(dev)
    rng.shuffle(test)
    return dev, test


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def verify(corpus_path: Path, dev_path: Path, test_path: Path, expect_n: int) -> bool:
    """Re-read what we wrote and check the claims the build makes.

    Deliberately reads from disk, not from memory: the artefact that later steps
    consume is the file, and D6's lesson is that a check which tests something
    adjacent to its claim is worse than no check.
    """
    print("\n--- verify (from disk) ---")
    ok = True
    corpus_ids = {d.doc_id for d in load_documents(corpus_path)}
    dev = load_questions(dev_path)
    test = load_questions(test_path)
    qs = dev + test

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        if not cond:
            ok = False
        print(f"  {'ok     ' if cond else 'FAILED '} {label}{(' — ' + detail) if detail and not cond else ''}")

    check(f"{len(qs)} questions == n_questions", len(qs) == expect_n, f"got {len(qs)}")
    check("no qid overlap dev/test", not ({q.qid for q in dev} & {q.qid for q in test}))
    check("qids unique", len({q.qid for q in qs}) == len(qs))

    bad_ans = [q.qid for q in qs if q.answerable and not set(q.gold_titles) <= corpus_ids]
    check("every answerable question has both gold paragraphs in corpus",
          not bad_ans, f"{len(bad_ans)} broken, e.g. {bad_ans[:3]}")

    leaked = [q.qid for q in qs if not q.answerable and (set(q.gold_titles) & corpus_ids)]
    check("every unanswerable question has zero gold paragraphs in corpus",
          not leaked, f"{len(leaked)} leaked, e.g. {leaked[:3]}")

    check("unanswerable items carry answer=None",
          all(q.answer is None for q in qs if not q.answerable))
    check("answerable items carry a non-empty answer",
          all(q.answer for q in qs if q.answerable))

    for name, split in (("dev", dev), ("test", test)):
        frac = sum(1 for q in split if not q.answerable) / max(len(split), 1)
        check(f"{name} unanswerable frac {frac:.3f} ~= {CONFIG.unanswerable_frac}",
              abs(frac - CONFIG.unanswerable_frac) < 0.02, f"got {frac:.3f}")

    check("no empty corpus text", all(d.text for d in load_documents(corpus_path)))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite existing data files")
    args = ap.parse_args()

    data = CONFIG.data_dir
    data.mkdir(parents=True, exist_ok=True)
    corpus_path = data / "corpus.jsonl"
    dev_path = data / "dev.jsonl"
    test_path = data / "test.jsonl"
    manifest_path = data / "dataset_manifest.json"

    existing = [p for p in (corpus_path, dev_path, test_path) if p.exists()]
    if existing and not args.force:
        print("refusing to overwrite (pass --force):")
        for p in existing:
            print(f"  {p}")
        return 1

    from datasets import load_dataset  # slow import; keep it out of --help

    print(f"--- source: {HF_DATASET} / {HF_CONFIG} / {HF_SPLIT} ---")
    ds = load_dataset(HF_DATASET, HF_CONFIG, split=HF_SPLIT)
    print(f"  {len(ds)} source items")

    print("\n--- select ---")
    items, excluded = select_items(
        ds, CONFIG.n_questions, CONFIG.unanswerable_frac, CONFIG.seed
    )

    print("\n--- corpus ---")
    docs, n_deleted, n_clash = build_corpus(items, excluded)
    print(f"  {len(docs)} paragraphs, {n_deleted} deleted ({len(excluded)} titles), "
          f"{n_clash} same-title text clashes (first kept)")

    questions = to_questions(items, excluded)
    dev, test = split_dev_test(questions, CONFIG.test_frac, CONFIG.seed)
    print(f"\n--- split ---\n  dev {len(dev)}, test {len(test)}")

    write_jsonl(corpus_path, (d.to_dict() for d in docs))
    write_jsonl(dev_path, (q.to_dict() for q in dev))
    write_jsonl(test_path, (q.to_dict() for q in test))

    manifest = {
        "source": {"dataset": HF_DATASET, "config": HF_CONFIG, "split": HF_SPLIT,
                   "n_source_items": len(ds)},
        "counts": {
            "corpus_docs": len(docs),
            "deleted_titles": len(excluded),
            "dev": len(dev),
            "test": len(test),
            "dev_unanswerable": sum(1 for q in dev if not q.answerable),
            "test_unanswerable": sum(1 for q in test if not q.answerable),
            "levels": dict(Counter(q.level for q in questions)),
            "types": dict(Counter(q.qtype for q in questions)),
        },
        "sha256_16": {p.name: sha256(p) for p in (corpus_path, dev_path, test_path)},
        "config": CONFIG.to_dict(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    ok = verify(corpus_path, dev_path, test_path, CONFIG.n_questions)
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
