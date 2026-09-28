"""The document registry: SQLite, one row per ingested file.

Why a registry separate from the vector store at all — the vectors could carry
this metadata in their payloads. Three things make a real table worth it:

1. **A document exists before its vectors do.** Upload, parse, chunk, and embed
   take seconds to minutes for a large PDF. The UI has to show "indexing…" and,
   if it fails, *why*. A failed document has a row and no vectors, which is a
   state the vector store cannot represent.
2. **Deduplication needs a content hash lookup.** Re-uploading the same file
   should be recognised, not silently indexed twice — duplicate chunks quietly
   degrade retrieval by letting one document occupy every slot in the top-k.
3. **"List my documents" must not be a vector scan.** It is a table query.

SQLite because it is in the standard library, it is a single file you can open
with `sqlite3 storage/registry.sqlite`, and it survives a restart. Connections are
opened per operation with `check_same_thread=False` and WAL, because FastAPI
serves requests on a thread pool.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from agentic_rag.config import CONFIG
from agentic_rag.models import DocumentRecord, utcnow

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id        TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    source_type   TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    n_bytes       INTEGER DEFAULT 0,
    n_chars       INTEGER DEFAULT 0,
    n_chunks      INTEGER DEFAULT 0,
    n_pages       INTEGER,
    title         TEXT DEFAULT '',
    source_uri    TEXT DEFAULT '',
    stored_path   TEXT DEFAULT '',
    vector_id     TEXT DEFAULT '',
    status        TEXT NOT NULL DEFAULT 'pending',
    error         TEXT DEFAULT '',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
-- Dedup lookup is by (hash, vector_id): the same bytes embedded by a different
-- model is genuinely a different index entry, not a duplicate.
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash, vector_id);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at DESC);
"""

_FIELDS = (
    "doc_id", "filename", "source_type", "content_hash", "n_bytes", "n_chars",
    "n_chunks", "n_pages", "title", "source_uri", "stored_path", "vector_id",
    "status", "error", "created_at", "updated_at",
)


class Registry:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or CONFIG.registry_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        # WAL lets a reader and a writer coexist, which matters as soon as the UI
        # polls document status while an upload is still indexing.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- writes -----------------------------------------------------------

    def upsert(self, rec: DocumentRecord) -> DocumentRecord:
        rec.updated_at = utcnow()
        cols = ", ".join(_FIELDS)
        marks = ", ".join("?" for _ in _FIELDS)
        updates = ", ".join(f"{f}=excluded.{f}" for f in _FIELDS if f not in ("doc_id", "created_at"))
        with self._lock, self._connect() as conn:
            conn.execute(
                f"INSERT INTO documents ({cols}) VALUES ({marks}) "
                f"ON CONFLICT(doc_id) DO UPDATE SET {updates}",
                [getattr(rec, f) for f in _FIELDS],
            )
        return rec

    def set_status(self, doc_id: str, status: str, error: str = "") -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE documents SET status=?, error=?, updated_at=? WHERE doc_id=?",
                (status, error, utcnow(), doc_id),
            )

    def delete(self, doc_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
            return cur.rowcount > 0

    # --- reads ------------------------------------------------------------

    def get(self, doc_id: str) -> DocumentRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        return _to_record(row) if row else None

    def find_by_hash(self, content_hash: str, vector_id: str) -> DocumentRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE content_hash=? AND vector_id=? AND status='ready'",
                (content_hash, vector_id),
            ).fetchone()
        return _to_record(row) if row else None

    def list(self, status: str | None = None, limit: int = 500) -> list[DocumentRecord]:
        sql = "SELECT * FROM documents"
        params: list[Any] = []
        if status:
            sql += " WHERE status=?"
            params.append(status)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            return [_to_record(r) for r in conn.execute(sql, params).fetchall()]

    def ready_doc_ids(self) -> list[str]:
        with self._connect() as conn:
            return [
                r["doc_id"]
                for r in conn.execute(
                    "SELECT doc_id FROM documents WHERE status='ready'"
                ).fetchall()
            ]

    def stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) n, COALESCE(SUM(n_chunks),0) chunks, "
                "COALESCE(SUM(n_chars),0) chars, COALESCE(SUM(n_bytes),0) bytes "
                "FROM documents WHERE status='ready'"
            ).fetchone()
            by_status = {
                r["status"]: r["c"]
                for r in conn.execute(
                    "SELECT status, COUNT(*) c FROM documents GROUP BY status"
                ).fetchall()
            }
        return {
            "n_documents": row["n"],
            "n_chunks": row["chunks"],
            "n_chars": row["chars"],
            "n_bytes": row["bytes"],
            "by_status": by_status,
        }


def _to_record(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(**{k: row[k] for k in row.keys()})
