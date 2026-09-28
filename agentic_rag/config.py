"""Single source of truth for every knob in the project.

Rule we follow everywhere: no module reads os.environ directly except this one.
A results file or an API response can then record `CONFIG.to_dict()` and you know
exactly what produced a number or an answer.

The config is split into three groups, because the project has three audiences:
serving (what the API uses per request), ingestion (what a document looks like
once it is in the index), and eval (the benchmark that guards regressions).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Config:
    # --- paths -------------------------------------------------------------
    root: Path = ROOT
    data_dir: Path = ROOT / "data"
    results_dir: Path = ROOT / "results"
    cache_dir: Path = ROOT / ".cache" / "llm"
    storage_dir: Path = ROOT / "storage"
    upload_dir: Path = ROOT / "storage" / "uploads"
    registry_path: Path = ROOT / "storage" / "registry.sqlite"
    qdrant_path: Path = ROOT / "storage" / "qdrant"

    # Set QDRANT_URL to talk to a Qdrant server instead of the embedded on-disk
    # store. Embedded mode takes an exclusive lock on its directory, so exactly
    # one process may hold it — fine for a single API server, a problem the
    # moment you want a worker too. See DECISIONS D21.
    qdrant_url: str = field(default_factory=lambda: _env_str("QDRANT_URL", ""))

    # --- models ------------------------------------------------------------
    anthropic_model: str = field(default_factory=lambda: _env_str("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"))
    # The answer model may differ from the cheap one used for grading/rewriting:
    # those are classification tasks, answering is not.
    answer_model: str = field(default_factory=lambda: _env_str("ANSWER_MODEL", ""))

    embed_provider: str = field(default_factory=lambda: _env_str("EMBED_PROVIDER", "local"))
    embed_model: str = field(default_factory=lambda: _env_str("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))

    # --- ingestion ---------------------------------------------------------
    # Chunk size is in characters, not tokens: the loaders produce characters, and
    # a character budget is the one you can reason about when reading a chunk in
    # the UI. ~1200 chars is roughly 300 tokens, comfortably inside bge's 512.
    chunk_chars: int = field(default_factory=lambda: _env_int("CHUNK_CHARS", 1200))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 200))
    min_chunk_chars: int = field(default_factory=lambda: _env_int("MIN_CHUNK_CHARS", 80))
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 50))

    # --- retrieval ---------------------------------------------------------
    collection: str = field(default_factory=lambda: _env_str("COLLECTION", "documents"))
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 8))
    # Hybrid retrieval pulls a wider pool from each retriever, then fuses and
    # truncates to top_k. Fusing after a k-sized cut would throw away the
    # documents fusion exists to rescue.
    candidate_k: int = field(default_factory=lambda: _env_int("CANDIDATE_K", 30))
    hybrid: bool = field(default_factory=lambda: _env_bool("HYBRID", True))
    rrf_k: int = field(default_factory=lambda: _env_int("RRF_K", 60))

    # --- agent loop --------------------------------------------------------
    max_hops: int = field(default_factory=lambda: _env_int("MAX_HOPS", 3))
    # D22: the final hop answers with whatever it holds instead of abstaining.
    answer_on_exhausted: bool = field(default_factory=lambda: _env_bool("ANSWER_ON_EXHAUSTED", True))
    max_context_chunks: int = field(default_factory=lambda: _env_int("MAX_CONTEXT_CHUNKS", 14))

    # --- eval (benchmark only; unused by the served app) -------------------
    seed: int = field(default_factory=lambda: _env_int("SEED", 42))
    n_questions: int = field(default_factory=lambda: _env_int("N_QUESTIONS", 600))
    unanswerable_frac: float = field(default_factory=lambda: _env_float("UNANSWERABLE_FRAC", 0.20))
    test_frac: float = field(default_factory=lambda: _env_float("TEST_FRAC", 0.30))
    eval_collection: str = "hotpotqa_paragraphs"
    # The benchmark keeps its own Qdrant directory. Embedded Qdrant locks a
    # directory per process, so sharing one with the served app would mean the API
    # and an eval run could never be up at the same time — and the two indexes hold
    # different things anyway (HotpotQA paragraphs vs the user's documents).
    eval_qdrant_path: Path = ROOT / "qdrant_storage"

    # --- derived -----------------------------------------------------------
    @property
    def answer_model_id(self) -> str:
        return self.answer_model or self.anthropic_model

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe snapshot, for embedding in a results file or /health."""
        d = asdict(self)
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}

    def ensure_dirs(self) -> None:
        for p in (self.storage_dir, self.upload_dir, self.cache_dir, self.results_dir):
            p.mkdir(parents=True, exist_ok=True)

    def require_api_key(self) -> str:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        # The .env.example placeholder is a real trap: it is *present*, so a naive
        # "is it set?" check passes and you only find out at the first API call.
        if key.endswith("...") or key == "sk-ant-":
            raise RuntimeError(
                "ANTHROPIC_API_KEY is still the placeholder from .env.example. "
                "Put your real key in .env."
            )
        return key


CONFIG = Config()
