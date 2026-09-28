"""The HTTP API, and the single-process constraint it is built around.

**One process owns the index.** Embedded Qdrant takes an exclusive lock on its
directory, so the store, the embedder, and the registry are created once at startup
and shared by every request. That is why they live on `app.state` rather than being
constructed per request: two `Retriever()` calls in one process would each load a
130MB model, and in two processes the second would fail to open the store at all.
Run with `--workers 1`, or set `QDRANT_URL` to a Qdrant server and the constraint
lifts (DECISIONS D21).

**Ingestion and answering are synchronous and CPU/IO-bound.** They are declared as
`def`, not `async def`, so FastAPI runs them on its thread pool instead of blocking
the event loop. An `async def` endpoint that spent 40 seconds embedding a PDF would
freeze every other request in the process, including the health check.

**Streaming is a queue bridge.** The agent reports progress through a synchronous
callback, which cannot `await`. So the agent runs in a worker thread and pushes
events onto a queue that an async generator drains into server-sent events. This
matters for the product, not just the demo: a three-hop answer takes several
seconds, and showing "searching → reading → reformulating" is the difference between
a system that looks stuck and one that visibly reasons.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from agentic_rag.agent import Answerer
from agentic_rag.config import CONFIG
from agentic_rag.ingest.loaders import (
    SUPPORTED_EXTENSIONS,
    CorruptDocument,
    EmptyDocument,
    FetchFailed,
    UnsupportedFormat,
)
from agentic_rag.ingest.pipeline import IngestionPipeline
from agentic_rag.llm import LLM
from agentic_rag.registry import Registry
from agentic_rag.retrieve import Retriever
from agentic_rag.store import VectorSpaceMismatch
from api.schemas import DeleteResponse, QueryRequest, UrlIngestRequest

VERSION = "1.0.0"
UI_DIR = Path(__file__).resolve().parent.parent / "ui"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("agentic_rag.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the expensive, shared, single-owner objects exactly once."""
    CONFIG.ensure_dirs()
    CONFIG.require_api_key()  # fail at boot, not on the first question
    log.info("loading embedder %s", CONFIG.embed_model)
    t0 = time.time()
    retriever = Retriever()
    registry = Registry()
    pipeline = IngestionPipeline(retriever, registry)
    llm = LLM()
    answer_llm = LLM(model=CONFIG.answer_model_id) if CONFIG.answer_model else llm

    app.state.retriever = retriever
    app.state.registry = registry
    app.state.pipeline = pipeline
    app.state.llm = llm
    app.state.answerer = Answerer(retriever=retriever, llm=llm, answer_llm=answer_llm)
    # A crash mid-ingest can leave vectors without a registry row, which would be
    # retrieved and cited from a document the user cannot see. Cheap to check, and
    # boot is the only moment we know nothing else is writing.
    app.state.reconciled = pipeline.reconcile()

    log.info(
        "ready in %.1fs — %d chunks, %d documents",
        time.time() - t0,
        retriever.store.count(),
        registry.stats()["n_documents"],
    )
    try:
        yield
    finally:
        retriever.store.close()


app = FastAPI(
    title="Agentic RAG",
    version=VERSION,
    description="Drop in documents, ask questions, get answers with verified citations.",
    lifespan=lifespan,
)

# Wide open because the default deployment is localhost. Narrow this before
# putting it on a network — there is no authentication in front of it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- errors -----------------------------------------------------------------


def _upstream_error(e: Exception) -> HTTPException | None:
    """Map an Anthropic SDK failure to something the operator can act on.

    These are not bugs in this service and they are not the caller's fault, so a bare
    500 with the SDK's own message is the wrong answer on both counts — it reads as
    "this application is broken" when the truth is "the account is out of credit" or
    "you are being rate-limited, retry". Same principle as D28: the status code is
    chosen by what the reader should do next.
    """
    import anthropic

    if isinstance(e, anthropic.AuthenticationError):
        return HTTPException(500, "ANTHROPIC_API_KEY was rejected. Check the key in .env.")
    if isinstance(e, anthropic.RateLimitError):
        return HTTPException(429, "Rate-limited by the Anthropic API. Retry shortly.")
    if isinstance(e, anthropic.BadRequestError):
        msg = str(e)
        if "credit balance" in msg.lower():
            # 402 rather than 500: nothing here is broken and retrying will not help
            # until someone tops up the account.
            return HTTPException(
                402,
                "The Anthropic account is out of credit, so answers cannot be "
                "generated. Retrieval still works — try POST /api/search. Add credits "
                "at console.anthropic.com to restore answering.",
            )
        return HTTPException(400, f"The model rejected the request: {msg[:300]}")
    if isinstance(e, (anthropic.APIConnectionError, anthropic.APITimeoutError)):
        return HTTPException(504, "Could not reach the Anthropic API. Check connectivity.")
    if isinstance(e, anthropic.APIStatusError):
        return HTTPException(502, f"Anthropic API error {e.status_code}. This is upstream.")
    return None


def _http_error(e: Exception) -> HTTPException:
    """Map an internal failure to a status code and a message a user can act on."""
    upstream = _upstream_error(e)
    if upstream is not None:
        return upstream
    if isinstance(e, UnsupportedFormat):
        return HTTPException(415, str(e))
    if isinstance(e, (EmptyDocument, CorruptDocument, FetchFailed)):
        return HTTPException(422, str(e))
    if isinstance(e, VectorSpaceMismatch):
        return HTTPException(409, str(e))
    if isinstance(e, ValueError):
        return HTTPException(400, str(e))
    if isinstance(e, FileNotFoundError):
        return HTTPException(404, str(e))
    log.exception("unhandled error")
    return HTTPException(500, f"{type(e).__name__}: {e}")


# --- documents --------------------------------------------------------------


@app.post("/api/documents", status_code=201)
def upload_document(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    """Ingest an uploaded file: parse, chunk, embed, index.

    Synchronous on purpose. A background job would return faster, but then the
    client has to poll and the common case — a handful of documents on a laptop —
    gains nothing but a state machine. The registry already records `indexing` and
    `failed`, so moving to a queue later does not change this contract.
    """
    if not file.filename:
        raise HTTPException(400, "no filename supplied")
    data = file.file.read()
    if not data:
        raise HTTPException(400, f"{file.filename} is empty")
    try:
        result = request.app.state.pipeline.ingest_bytes(data, file.filename)
    except Exception as e:  # noqa: BLE001
        raise _http_error(e) from e
    log.info("ingested %s (%d chunks, %.1fs)", file.filename,
             result.document.n_chunks, result.elapsed_s)
    return result.to_dict()


@app.post("/api/documents/url", status_code=201)
def ingest_url(request: Request, body: UrlIngestRequest) -> dict[str, Any]:
    try:
        result = request.app.state.pipeline.ingest_url(body.url)
    except Exception as e:  # noqa: BLE001
        raise _http_error(e) from e
    return result.to_dict()


@app.get("/api/documents")
def list_documents(request: Request) -> dict[str, Any]:
    registry: Registry = request.app.state.registry
    return {
        "documents": [d.to_dict() for d in registry.list()],
        "stats": request.app.state.pipeline.stats(),
    }


@app.get("/api/documents/{doc_id}")
def get_document(request: Request, doc_id: str) -> dict[str, Any]:
    rec = request.app.state.registry.get(doc_id)
    if rec is None:
        raise HTTPException(404, f"no document {doc_id}")
    return rec.to_dict()


@app.get("/api/documents/{doc_id}/file")
def download_document(request: Request, doc_id: str):
    """Serve the original file, so a citation can open the real source.

    This is why the pipeline keeps the uploaded bytes. A citation that says
    "report.pdf p.2" is useful; one that opens report.pdf at page 2 is the product.
    """
    rec = request.app.state.registry.get(doc_id)
    if rec is None:
        raise HTTPException(404, f"no document {doc_id}")
    if not rec.stored_path:
        raise HTTPException(404, f"{rec.filename} has no stored original (ingested from a URL)")
    path = Path(rec.stored_path)
    # The path comes from our own registry, but resolving and re-checking it costs
    # nothing and means a corrupted row cannot turn into an arbitrary file read.
    if not path.is_file() or CONFIG.upload_dir.resolve() not in path.resolve().parents:
        raise HTTPException(404, "stored file is missing")
    return FileResponse(path, filename=rec.filename)


@app.delete("/api/documents/{doc_id}")
def delete_document(request: Request, doc_id: str) -> DeleteResponse:
    deleted = request.app.state.pipeline.delete(doc_id)
    if not deleted:
        raise HTTPException(404, f"no document {doc_id}")
    log.info("deleted document %s", doc_id)
    return DeleteResponse(deleted=True, doc_id=doc_id)


# --- querying ---------------------------------------------------------------


@app.post("/api/query")
def query(request: Request, body: QueryRequest) -> dict[str, Any]:
    answerer: Answerer = request.app.state.answerer
    if body.top_k or body.max_hops:
        # Per-request overrides get their own Answerer rather than mutating the
        # shared one, which would leak one request's settings into every concurrent
        # request on the thread pool.
        answerer = Answerer(
            retriever=answerer.retriever,
            llm=answerer.llm,
            answer_llm=answerer.answer_llm,
            top_k=body.top_k,
            max_hops=body.max_hops,
        )
    try:
        result = answerer.answer(body.question, doc_ids=body.doc_ids)
    except Exception as e:  # noqa: BLE001
        raise _http_error(e) from e
    return result.to_dict()


@app.post("/api/query/stream")
async def query_stream(request: Request, body: QueryRequest) -> StreamingResponse:
    """Same answer, streamed as server-sent events.

    The agent runs in a worker thread and pushes events onto a plain queue; this
    coroutine drains the queue and writes SSE frames. The sentinel `None` marks the
    end, and an exception from the thread is forwarded as an `error` event rather
    than left to hang the connection open.
    """
    answerer: Answerer = request.app.state.answerer
    if body.top_k or body.max_hops:
        answerer = Answerer(
            retriever=answerer.retriever, llm=answerer.llm, answer_llm=answerer.answer_llm,
            top_k=body.top_k, max_hops=body.max_hops,
        )

    events: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue()

    def run() -> None:
        try:
            result = answerer.answer(
                body.question,
                doc_ids=body.doc_ids,
                on_event=lambda kind, payload: events.put((kind, payload)),
            )
            events.put(("result", result.to_dict()))
        except Exception as e:  # noqa: BLE001
            log.exception("stream query failed")
            # The stream has already returned 200, so the status code is no longer
            # available to carry meaning — the message has to do that work alone.
            upstream = _upstream_error(e)
            events.put(("error", {
                "error": upstream.detail if upstream else f"{type(e).__name__}: {e}",
                "status": upstream.status_code if upstream else 500,
            }))
        finally:
            events.put(None)

    threading.Thread(target=run, daemon=True).start()
    loop = asyncio.get_running_loop()

    async def frames() -> AsyncIterator[str]:
        while True:
            # `queue.get` blocks, so it goes to a thread — otherwise the event loop
            # stalls and nothing gets flushed to the client until the end.
            item = await loop.run_in_executor(None, events.get)
            if item is None:
                break
            kind, payload = item
            yield f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx would otherwise buffer the whole stream
        },
    )


@app.post("/api/search")
def search_only(request: Request, body: QueryRequest) -> dict[str, Any]:
    """Retrieval with no generation. The debugging endpoint.

    When an answer is wrong, the first question is always "was the evidence even
    there?" — and answering it should not cost an LLM call or make you read a trace.
    """
    retriever: Retriever = request.app.state.retriever
    hits = retriever.search(body.question, k=body.top_k or CONFIG.top_k, doc_ids=body.doc_ids)
    return {
        "question": body.question,
        "hybrid": CONFIG.hybrid,
        "n_results": len(hits),
        "results": [h.to_dict() for h in hits],
    }


# --- operations -------------------------------------------------------------


@app.get("/api/health")
def health(request: Request) -> dict[str, Any]:
    state = request.app.state
    store_info = state.retriever.store.info()
    doc_stats = state.registry.stats()
    # "Reachable" is not "working": a store with a collection but zero chunks
    # answers nothing, and saying `ok` would be the kind of green light this
    # project's notes specifically warn about.
    degraded = not store_info["exists"] or doc_stats["n_documents"] == 0
    return {
        "status": "degraded" if degraded else "ok",
        "version": VERSION,
        "embedding": {
            "provider": CONFIG.embed_provider,
            "model": state.retriever.embedder.model_name,
            "dim": state.retriever.embedder.dim,
            "vector_id": state.retriever.embedder.vector_id,
        },
        "store": store_info,
        "documents": doc_stats,
        "models": {"answer": CONFIG.answer_model_id, "utility": CONFIG.anthropic_model},
        "config": {
            "top_k": CONFIG.top_k,
            "candidate_k": CONFIG.candidate_k,
            "hybrid": CONFIG.hybrid,
            "max_hops": CONFIG.max_hops,
            "chunk_chars": CONFIG.chunk_chars,
            "chunk_overlap": CONFIG.chunk_overlap,
            "max_context_chunks": CONFIG.max_context_chunks,
        },
        "reconciled_at_boot": getattr(state, "reconciled", {}),
        "supported_formats": list(SUPPORTED_EXTENSIONS),
        "llm_usage_this_process": state.llm.usage.to_dict(),
    }


@app.post("/api/reconcile")
def reconcile(request: Request) -> dict[str, Any]:
    return request.app.state.pipeline.reconcile()


# --- UI ---------------------------------------------------------------------

if UI_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> HTMLResponse:
        return HTMLResponse((UI_DIR / "index.html").read_text(encoding="utf-8"))
