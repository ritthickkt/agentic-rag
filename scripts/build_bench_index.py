"""Index the HotpotQA benchmark corpus into the *product* vector store.

    python scripts/build_bench_index.py

Why this exists: `scripts/build_index.py` builds the index the four benchmark rungs
were measured against, using `agentic_rag/index.py`. That index has no payload
provenance and no lexical half, so the served engine cannot read it. This script puts
the same 5,670 paragraphs into a `VectorStore` collection the served `Retriever` can
query, which is what lets `run_eval.py --system product` score the thing we ship.

**One paragraph, one chunk, no splitting.** HotpotQA paragraphs are short, and running
them through the chunker would risk splitting one into two — at which point a
`doc_id` could be retrieved twice and `recall_all` would measure something subtly
different from what it measured before. The point of this index is that the *only*
thing that changed is the system under test.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic_rag.config import CONFIG
from agentic_rag.embed import build_embedder
from agentic_rag.ingest.chunker import chunk_id_for
from agentic_rag.models import Chunk
from agentic_rag.store import VectorStore
from eval.types import load_documents

BENCH_COLLECTION = "hotpotqa_bench"


def main() -> int:
    corpus_path = CONFIG.data_dir / "corpus.jsonl"
    if not corpus_path.is_file():
        print(f"error: {corpus_path} not found. Run scripts/build_dataset.py first.",
              file=sys.stderr)
        return 1

    docs = load_documents(corpus_path)
    print(f"loaded {len(docs)} paragraphs from {corpus_path.name}")

    embedder = build_embedder()
    store = VectorStore(collection=BENCH_COLLECTION)
    print(f"embedder {embedder.vector_id}\ncollection {BENCH_COLLECTION} at {store.location}")

    store.drop()  # rebuild from scratch: a partial benchmark index is worse than none
    store.ensure_collection(embedder.dim, embedder.vector_id)

    # doc_id stays the HotpotQA article title — it is the join key every retrieval
    # metric compares against `gold_titles`.
    chunks = [
        Chunk(
            chunk_id=chunk_id_for(d.doc_id, i, d.text),
            doc_id=d.doc_id,
            text=d.text,
            ordinal=i,
            filename=d.doc_id,
            title=d.doc_id,
        )
        for i, d in enumerate(docs)
    ]

    t0 = time.time()
    vectors = embedder.encode_documents([c.text for c in chunks], show_progress=True)
    print(f"embedded in {time.time() - t0:.0f}s")

    store.upsert_chunks(chunks, vectors)
    n = store.count()
    print(f"indexed {n} chunks")
    if n != len(docs):
        # Titles are unique in the corpus, so anything else means a collision or a
        # dropped batch — and a quietly short index would make recall look worse for
        # reasons that have nothing to do with the system under test.
        print(f"WARNING: expected {len(docs)} but the store holds {n}", file=sys.stderr)
    store.close()

    print(f"\nnow run:  python scripts/run_eval.py --system product --split dev")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
