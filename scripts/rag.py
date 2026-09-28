"""Command-line access to the same engine the API serves.

    python scripts/rag.py add report.pdf notes/*.md
    python scripts/rag.py add-url https://example.com/post
    python scripts/rag.py ask "what were the Q3 risks?"
    python scripts/rag.py ask "..." --doc <doc_id> --show-passages
    python scripts/rag.py search "ERR_CONN_4021"
    python scripts/rag.py ls
    python scripts/rag.py rm <doc_id>
    python scripts/rag.py stats
    python scripts/rag.py reindex

This exists for two reasons beyond convenience. Bulk ingestion belongs in a shell
loop, not in a browser file picker. And when an answer looks wrong, `search` shows
what retrieval returned without spending an LLM call — which is the first question
worth asking and the cheapest one to answer.

**The server must be stopped first.** Embedded Qdrant locks its directory to one
process, so this CLI and `uvicorn` cannot both hold it. Set `QDRANT_URL` to a Qdrant
server if you want both at once (DECISIONS D21).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic_rag.agent import Answerer
from agentic_rag.config import CONFIG
from agentic_rag.ingest.loaders import SUPPORTED_EXTENSIONS
from agentic_rag.ingest.pipeline import IngestionPipeline
from agentic_rag.llm import LLM
from agentic_rag.retrieve import Retriever

BOLD, DIM, CYAN, YELLOW, RED, GREEN, OFF = (
    "\033[1m", "\033[2m", "\033[36m", "\033[33m", "\033[31m", "\033[32m", "\033[0m"
)


def _engine() -> tuple[Retriever, IngestionPipeline]:
    retriever = Retriever()
    return retriever, IngestionPipeline(retriever)


# --- commands ---------------------------------------------------------------


def cmd_add(args: argparse.Namespace) -> int:
    retriever, pipeline = _engine()
    failures = 0
    try:
        for raw in args.paths:
            path = Path(raw)
            if path.is_dir():
                files = sorted(
                    f for f in path.rglob("*")
                    if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
                )
                if not files:
                    print(f"{YELLOW}no supported files under {path}{OFF}")
            else:
                files = [path]
            for f in files:
                try:
                    res = pipeline.ingest_path(f, on_progress=(
                        (lambda m, f=f: print(f"  {DIM}{f.name}: {m}{OFF}", flush=True))
                        if args.verbose else None
                    ))
                    tag = f"{DIM}duplicate{OFF}" if res.duplicate else f"{GREEN}indexed{OFF}"
                    print(f"{tag}  {f.name}  {res.document.n_chunks} chunks  "
                          f"{res.elapsed_s:.1f}s  {DIM}{res.document.doc_id}{OFF}")
                    for w in res.warnings:
                        print(f"  {YELLOW}warning: {w}{OFF}")
                except Exception as e:  # noqa: BLE001
                    failures += 1
                    print(f"{RED}failed{OFF}   {f.name}: {e}")
    finally:
        retriever.store.close()
    return 1 if failures else 0


def cmd_add_url(args: argparse.Namespace) -> int:
    retriever, pipeline = _engine()
    try:
        res = pipeline.ingest_url(args.url)
        tag = "duplicate" if res.duplicate else "indexed"
        print(f"{tag}  {res.document.title or args.url}  {res.document.n_chunks} chunks  "
              f"{DIM}{res.document.doc_id}{OFF}")
    except Exception as e:  # noqa: BLE001
        print(f"{RED}failed: {e}{OFF}")
        return 1
    finally:
        retriever.store.close()
    return 0


def _explain_upstream(e: Exception) -> str | None:
    """A readable line for the Anthropic failures a user will actually hit."""
    import anthropic

    if isinstance(e, anthropic.AuthenticationError):
        return "ANTHROPIC_API_KEY was rejected. Check the key in .env."
    if isinstance(e, anthropic.RateLimitError):
        return "Rate-limited by the Anthropic API. Retry shortly."
    if isinstance(e, anthropic.BadRequestError) and "credit balance" in str(e).lower():
        return ("The Anthropic account is out of credit, so answers cannot be generated.\n"
                "  Retrieval still works: try `python scripts/rag.py search \"...\"`.\n"
                "  Add credits at console.anthropic.com to restore answering.")
    if isinstance(e, (anthropic.APIConnectionError, anthropic.APITimeoutError)):
        return "Could not reach the Anthropic API. Check connectivity."
    return None


def cmd_ask(args: argparse.Namespace) -> int:
    retriever, _ = _engine()
    try:
        answerer = Answerer(
            retriever=retriever, llm=LLM(), max_hops=args.max_hops, top_k=args.top_k
        )

        def on_event(kind: str, data: dict) -> None:
            if kind == "searching":
                print(f"{DIM}  search hop {data['hop'] + 1}: {data['query']}{OFF}", flush=True)
            elif kind == "retrieved":
                print(f"{DIM}  +{data['n_new']} passages ({data['n_total']} total){OFF}", flush=True)
            elif kind == "reformulating":
                print(f"{DIM}  missing: {data['missing']}{OFF}", flush=True)

        try:
            res = answerer.answer(args.question, doc_ids=args.doc or None,
                                  on_event=on_event if not args.quiet else None)
        except Exception as e:  # noqa: BLE001
            explained = _explain_upstream(e)
            if explained is None:
                raise
            print(f"\n{RED}{explained}{OFF}")
            return 2

        print()
        if res.abstained and not res.answer:
            print(f"{YELLOW}No answer from your documents.{OFF}")
            print(f"  {res.abstain_reason}")
        else:
            print(f"{BOLD}{res.answer}{OFF}\n")
            for c in res.citations:
                loc = c.filename + (f" p.{c.page}" if c.page else "") + (f" — {c.heading}" if c.heading else "")
                mark = "" if c.verified else f"  {RED}[UNVERIFIED]{OFF}"
                print(f"  {CYAN}[{c.marker}]{OFF} {loc}{mark}")
                print(f'      {DIM}"{c.quote[:150]}"{OFF}')

        v = res.verification or {}
        print(f"\n{DIM}{res.confidence or 'n/a'} confidence · {len(res.hops)} hop(s) · "
              f"{res.n_llm_calls} calls · {res.input_tokens}/{res.output_tokens} tokens · "
              f"{res.latency_s:.1f}s · quotes {v.get('n_verified', 0)} ok, "
              f"{v.get('n_relocated', 0)} repointed, {v.get('n_unverified', 0)} unverified{OFF}")

        if args.show_passages:
            print(f"\n{BOLD}Passages seen:{OFF}")
            for i, rc in enumerate(res.chunks, start=1):
                tags = []
                if rc.dense_rank: tags.append(f"vec#{rc.dense_rank}")
                if rc.lexical_rank: tags.append(f"kw#{rc.lexical_rank}")
                print(f"  {i:2d}. {rc.chunk.location}  {DIM}{' '.join(tags)}{OFF}")
                print(f"      {DIM}{rc.chunk.text[:160]}{OFF}")
    finally:
        retriever.store.close()
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Retrieval only. The no-LLM diagnostic."""
    retriever, _ = _engine()
    try:
        hits = retriever.search(args.question, k=args.top_k or CONFIG.top_k,
                                doc_ids=args.doc or None)
        if not hits:
            print(f"{YELLOW}nothing retrieved{OFF}")
            return 0
        print(f"{DIM}hybrid={CONFIG.hybrid}  candidate_k={CONFIG.candidate_k}{OFF}\n")
        for i, rc in enumerate(hits, start=1):
            tags = []
            if rc.dense_rank: tags.append(f"vector #{rc.dense_rank}")
            if rc.lexical_rank: tags.append(f"keyword #{rc.lexical_rank}")
            print(f"{BOLD}{i:2d}. {rc.chunk.location}{OFF}  {DIM}{'  '.join(tags) or 'no rank'}{OFF}")
            print(f"    {rc.chunk.text[:230].replace(chr(10), ' ')}\n")
    finally:
        retriever.store.close()
    return 0


def cmd_ls(args: argparse.Namespace) -> int:
    retriever, pipeline = _engine()
    try:
        docs = pipeline.registry.list()
        if not docs:
            print(f"{DIM}no documents indexed{OFF}")
            return 0
        print(f"{BOLD}{'doc_id':34s} {'type':9s} {'chunks':>7s} {'status':9s} title{OFF}")
        for d in docs:
            colour = {"ready": GREEN, "failed": RED}.get(d.status, YELLOW)
            print(f"{DIM}{d.doc_id}{OFF} {d.source_type:9s} {d.n_chunks:7d} "
                  f"{colour}{d.status:9s}{OFF} {d.title or d.filename}")
            if d.error:
                print(f"  {RED}{d.error[:110]}{OFF}")
    finally:
        retriever.store.close()
    return 0


def cmd_rm(args: argparse.Namespace) -> int:
    retriever, pipeline = _engine()
    try:
        for doc_id in args.doc_ids:
            print(f"{GREEN}removed{OFF} {doc_id}" if pipeline.delete(doc_id)
                  else f"{RED}not found{OFF} {doc_id}")
    finally:
        retriever.store.close()
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    retriever, pipeline = _engine()
    try:
        st = pipeline.stats()
        print(f"{BOLD}corpus{OFF}")
        print(f"  documents   {st['n_documents']}")
        print(f"  chunks      {st['n_chunks']}")
        print(f"  characters  {st['n_chars']:,}")
        print(f"  by status   {st['by_status']}")
        print(f"{BOLD}store{OFF}")
        for k, v in st["store"].items():
            print(f"  {k:11s} {v}")
        print(f"{BOLD}retrieval{OFF}")
        print(f"  embedder    {retriever.embedder.vector_id}")
        print(f"  top_k       {CONFIG.top_k} (from {CONFIG.candidate_k} candidates)")
        print(f"  hybrid      {CONFIG.hybrid}")
        print(f"  max_hops    {CONFIG.max_hops}")
    finally:
        retriever.store.close()
    return 0


def cmd_reindex(args: argparse.Namespace) -> int:
    """Re-chunk and re-embed every stored original.

    Needed after changing `CHUNK_CHARS`, `EMBED_MODEL`, or a loader: the chunks in
    the store were produced by the old settings, and mixing the two in one
    collection means half the corpus is searched under rules that no longer apply.
    """
    retriever, pipeline = _engine()
    try:
        docs = [d for d in pipeline.registry.list() if d.stored_path or d.source_uri]
        skipped = [d for d in pipeline.registry.list() if not (d.stored_path or d.source_uri)]
        print(f"re-indexing {len(docs)} documents into {CONFIG.collection}")
        if skipped:
            print(f"{YELLOW}{len(skipped)} cannot be re-indexed (no stored original){OFF}")
        if not args.yes:
            if input("this deletes and rebuilds the collection. type 'yes': ").strip() != "yes":
                print("aborted")
                return 1

        retriever.store.drop()
        retriever.store.ensure_collection(retriever.embedder.dim, retriever.embedder.vector_id)
        for d in docs:
            pipeline.registry.delete(d.doc_id)
        retriever.mark_dirty()

        ok = 0
        for d in docs:
            try:
                if d.stored_path and Path(d.stored_path).is_file():
                    res = pipeline.ingest_bytes(Path(d.stored_path).read_bytes(), d.filename)
                else:
                    res = pipeline.ingest_url(d.source_uri)
                print(f"  {GREEN}ok{OFF} {d.filename} -> {res.document.n_chunks} chunks")
                ok += 1
            except Exception as e:  # noqa: BLE001
                print(f"  {RED}failed{OFF} {d.filename}: {e}")
        print(f"\n{ok}/{len(docs)} re-indexed, {retriever.store.count()} chunks total")
    finally:
        retriever.store.close()
    return 0


# --- wiring -----------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(prog="rag", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("add", help="index files or directories")
    p.add_argument("paths", nargs="+")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_add)

    p = sub.add_parser("add-url", help="fetch and index a web page")
    p.add_argument("url")
    p.set_defaults(fn=cmd_add_url)

    p = sub.add_parser("ask", help="ask a question")
    p.add_argument("question")
    p.add_argument("--doc", action="append", help="restrict to this doc_id (repeatable)")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--max-hops", type=int, default=None)
    p.add_argument("--show-passages", action="store_true")
    p.add_argument("-q", "--quiet", action="store_true", help="hide the agent's steps")
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("search", help="retrieval only, no LLM call")
    p.add_argument("question")
    p.add_argument("--doc", action="append")
    p.add_argument("--top-k", type=int, default=None)
    p.set_defaults(fn=cmd_search)

    sub.add_parser("ls", help="list indexed documents").set_defaults(fn=cmd_ls)

    p = sub.add_parser("rm", help="remove documents")
    p.add_argument("doc_ids", nargs="+")
    p.set_defaults(fn=cmd_rm)

    sub.add_parser("stats", help="corpus and config summary").set_defaults(fn=cmd_stats)

    p = sub.add_parser("reindex", help="re-chunk and re-embed everything")
    p.add_argument("-y", "--yes", action="store_true")
    p.set_defaults(fn=cmd_reindex)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
