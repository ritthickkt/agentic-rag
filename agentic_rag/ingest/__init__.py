"""Document ingestion: bytes -> sections -> chunks -> vectors."""

from agentic_rag.ingest.chunker import chunk_sections, chunk_id_for
from agentic_rag.ingest.loaders import (
    CorruptDocument,
    EmptyDocument,
    FetchFailed,
    LoadResult,
    SUPPORTED_EXTENSIONS,
    UnsupportedFormat,
    detect_source_type,
    load_bytes,
    load_path,
    load_url,
)

__all__ = [
    "CorruptDocument",
    "EmptyDocument",
    "FetchFailed",
    "LoadResult",
    "SUPPORTED_EXTENSIONS",
    "UnsupportedFormat",
    "chunk_id_for",
    "chunk_sections",
    "detect_source_type",
    "load_bytes",
    "load_path",
    "load_url",
]
