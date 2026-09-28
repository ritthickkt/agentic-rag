"""The ingestion pipeline: bytes in, searchable chunks out.

This is the one place that writes to both the registry and the vector store, so it
is also the one place responsible for keeping them consistent. The ordering below
is chosen for exactly that reason:

1. Hash the bytes and check for a duplicate. Re-uploading the same file is a
   no-op, not a second copy — duplicate chunks degrade retrieval by letting one
   document occupy every slot in the top-k.
2. Write a `pending` registry row **before** parsing. If parsing then crashes, the
   user sees a failed document with the reason, rather than an upload that
   vanished silently.
3. Parse, chunk, embed, upsert.
4. Flip to `ready` only once the vectors are actually in the store.

A document is therefore never `ready` without its vectors, and a failure always
leaves a row explaining itself. The remaining gap is honest and documented: a hard
kill between step 3 and step 4 leaves vectors with a non-`ready` row, so
`reconcile()` exists to find and clean those.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agentic_rag.config import CONFIG
from agentic_rag.ingest.chunker import chunk_sections
from agentic_rag.ingest.loaders import (
    CorruptDocument,
    EmptyDocument,
    FetchFailed,
    LoadResult,
    UnsupportedFormat,
    detect_source_type,
    load_bytes,
    load_url,
)
from agentic_rag.models import Chunk, DocumentRecord, sha256_hex, utcnow
from agentic_rag.registry import Registry
from agentic_rag.retrieve import Retriever


@dataclass
class IngestResult:
    document: DocumentRecord
    duplicate: bool = False
    warnings: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "document": self.document.to_dict(),
            "duplicate": self.duplicate,
            "warnings": self.warnings,
            "elapsed_s": round(self.elapsed_s, 2),
        }


class IngestionPipeline:
    def __init__(self, retriever: Retriever, registry: Registry | None = None) -> None:
        self.retriever = retriever
        self.registry = registry or Registry()
        CONFIG.ensure_dirs()

    # --- public entry points ----------------------------------------------

    def ingest_bytes(
        self,
        data: bytes,
        filename: str,
        save_original: bool = True,
        on_progress: Callable[[str], None] | None = None,
    ) -> IngestResult:
        t0 = time.time()
        max_bytes = CONFIG.max_upload_mb * 1024 * 1024
        if len(data) > max_bytes:
            raise ValueError(
                f"{filename} is {len(data) / 1e6:.1f}MB, over the {CONFIG.max_upload_mb}MB limit."
            )
        # Raises UnsupportedFormat before we write anything, so an unreadable
        # upload never leaves a row behind.
        source_type = detect_source_type(filename)
        content_hash = sha256_hex(data)
        vector_id = self.retriever.embedder.vector_id

        existing = self.registry.find_by_hash(content_hash, vector_id)
        if existing:
            return IngestResult(existing, duplicate=True, elapsed_s=time.time() - t0)

        doc_id = uuid.uuid4().hex
        stored_path = ""
        if save_original:
            # Keep the original so a citation can link to the real file, and so a
            # re-index after a chunking change does not need the user to re-upload.
            dest = CONFIG.upload_dir / f"{doc_id}{Path(filename).suffix.lower()}"
            dest.write_bytes(data)
            stored_path = str(dest)

        rec = DocumentRecord(
            doc_id=doc_id,
            filename=filename,
            source_type=source_type,
            content_hash=content_hash,
            n_bytes=len(data),
            stored_path=stored_path,
            vector_id=vector_id,
            status="pending",
        )
        self.registry.upsert(rec)

        return self._index(rec, lambda: load_bytes(data, filename), t0, on_progress)

    def ingest_path(self, path: str | Path, **kwargs: Any) -> IngestResult:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"{p} is not a file")
        return self.ingest_bytes(p.read_bytes(), p.name, **kwargs)

    def ingest_url(
        self, url: str, on_progress: Callable[[str], None] | None = None
    ) -> IngestResult:
        t0 = time.time()
        loaded = load_url(url)
        # The hash is of the extracted text, not the raw HTML: a page whose ad
        # markup changed on every fetch would otherwise never dedupe.
        body = "\n\n".join(s.text for s in loaded.sections)
        content_hash = sha256_hex(body.encode("utf-8"))
        vector_id = self.retriever.embedder.vector_id

        existing = self.registry.find_by_hash(content_hash, vector_id)
        if existing:
            return IngestResult(existing, duplicate=True, elapsed_s=time.time() - t0)

        rec = DocumentRecord(
            doc_id=uuid.uuid4().hex,
            filename=loaded.title or url,
            source_type="url",
            content_hash=content_hash,
            n_bytes=len(body.encode("utf-8")),
            source_uri=url,
            vector_id=vector_id,
            status="pending",
        )
        self.registry.upsert(rec)
        return self._index(rec, lambda: loaded, t0, on_progress)

    # --- the shared indexing path -----------------------------------------

    def _index(
        self,
        rec: DocumentRecord,
        load: Callable[[], LoadResult],
        t0: float,
        on_progress: Callable[[str], None] | None,
    ) -> IngestResult:
        def progress(msg: str) -> None:
            if on_progress:
                on_progress(msg)

        try:
            self.registry.set_status(rec.doc_id, "indexing")
            progress("parsing")
            loaded = load()

            progress("chunking")
            title = loaded.title or Path(rec.filename).stem
            chunks = chunk_sections(
                loaded.sections, doc_id=rec.doc_id, filename=rec.filename, title=title
            )
            if not chunks:
                raise EmptyDocument("Document parsed but produced no chunks above the minimum size.")

            progress(f"embedding {len(chunks)} chunks")
            vectors = self.retriever.embedder.encode_documents([c.text for c in chunks])

            progress("storing")
            self.retriever.store.upsert_chunks(chunks, vectors)
            # The lexical index is in memory; without this the new document is
            # findable by vector search but invisible to keyword search until
            # restart — a bug that only shows up for exact-string queries.
            self.retriever.mark_dirty()

            rec.title = title
            rec.n_chars = loaded.n_chars
            rec.n_chunks = len(chunks)
            rec.n_pages = loaded.n_pages
            rec.status = "ready"
            rec.error = ""
            rec.updated_at = utcnow()
            self.registry.upsert(rec)
            progress("ready")
            return IngestResult(rec, warnings=loaded.warnings, elapsed_s=time.time() - t0)

        except (UnsupportedFormat, EmptyDocument, CorruptDocument, FetchFailed, ValueError) as e:
            # Expected failures: the user's file is the problem, and the message is
            # written for them rather than for a log.
            self._fail(rec, f"{e}")
            raise
        except Exception as e:  # noqa: BLE001
            self._fail(rec, f"{type(e).__name__}: {e}")
            raise

    def _fail(self, rec: DocumentRecord, error: str) -> None:
        rec.status = "failed"
        rec.error = error[:1000]
        self.registry.upsert(rec)
        # Partial vectors from a crash mid-upsert would otherwise be retrieved and
        # cited against a document the UI shows as failed.
        try:
            self.retriever.store.delete_doc(rec.doc_id)
            self.retriever.mark_dirty()
        except Exception:  # noqa: BLE001 — never mask the original error
            pass

    # --- maintenance ------------------------------------------------------

    def delete(self, doc_id: str) -> bool:
        """Remove a document's vectors, registry row, and stored original."""
        rec = self.registry.get(doc_id)
        if rec is None:
            return False
        self.retriever.store.delete_doc(doc_id)
        self.retriever.mark_dirty()
        if rec.stored_path:
            Path(rec.stored_path).unlink(missing_ok=True)
        self.registry.delete(doc_id)
        return True

    def reconcile(self) -> dict[str, Any]:
        """Find and fix disagreements between the registry and the vector store.

        Two failure shapes, both from a process dying mid-ingest:
        - vectors for a doc_id the registry has no `ready` row for -> orphaned
          vectors, which would be retrieved and cited from a document the user
          cannot see or delete. Removed.
        - a `ready` row with zero vectors -> a document that answers nothing.
          Marked failed so it is visible and re-indexable.
        """
        ready = set(self.registry.ready_doc_ids())
        seen: dict[str, int] = {}
        for chunk in self.retriever.store.iter_chunks():
            seen[chunk.doc_id] = seen.get(chunk.doc_id, 0) + 1

        orphaned = [d for d in seen if d not in ready]
        for doc_id in orphaned:
            self.retriever.store.delete_doc(doc_id)

        empty = [d for d in ready if seen.get(d, 0) == 0]
        for doc_id in empty:
            self.registry.set_status(doc_id, "failed", "no vectors in the index; re-ingest needed")

        if orphaned or empty:
            self.retriever.mark_dirty()
        return {
            "orphaned_vectors_removed": orphaned,
            "documents_marked_failed": empty,
            "n_ready": len(ready),
            "n_chunks": sum(seen.values()),
        }

    def stats(self) -> dict[str, Any]:
        return {**self.registry.stats(), "store": self.retriever.store.info()}
