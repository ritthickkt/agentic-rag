"""Shared fixtures. The important one is that **no test spends money.**

The engine takes its LLM and its embedder as arguments rather than constructing
them, which is what makes this possible: a fake embedder and a scripted LLM satisfy
the same interfaces the real ones do. That is not a testing convenience bolted on
afterwards — it is the same injection boundary the benchmark relies on, exercised a
second way.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agentic_rag.config as config_module


class FakeEmbedder:
    """Deterministic hash-based vectors. No model download, no GPU, no waiting.

    These vectors carry no semantics, so retrieval *quality* cannot be tested with
    them — only the plumbing: that a chunk goes in, comes back, keeps its payload,
    and is filtered correctly. Quality is what the benchmark in `eval/` is for, and
    conflating the two would give a suite that is slow and still proves nothing.
    """

    model_name = "fake"
    dim = 16

    @property
    def vector_id(self) -> str:
        return f"fake:{self.model_name}:{self.dim}"

    def _one(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        v = np.frombuffer(h[: self.dim], dtype=np.uint8).astype(np.float32) - 127.5
        return v / (np.linalg.norm(v) or 1.0)

    def encode_documents(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._one(t) for t in texts])

    def encode_queries(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        return self.encode_documents(texts)

    def encode_query(self, text: str) -> np.ndarray:
        return self._one(text)


class ScriptedLLM:
    """An LLM that returns queued responses instead of calling the API.

    Matches the parts of `agentic_rag.llm.LLM` the agent actually uses. Every call
    is recorded, so a test can assert on *how many* calls a control-flow path makes
    — which is the property the agent loop is most likely to get wrong, and the one
    that costs real money when it does.
    """

    def __init__(self, responses: list[dict[str, Any]], model: str = "scripted") -> None:
        self.responses = list(responses)
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def complete_json_with_usage(self, prompt: str, schema: dict, system: str | None = None,
                                 max_tokens: int = 256):
        from agentic_rag.llm import Completion

        self.calls.append({"prompt": prompt, "system": system})
        if not self.responses:
            raise AssertionError(
                f"ScriptedLLM ran out of responses after {len(self.calls)} calls"
            )
        return self.responses.pop(0), Completion(text="", input_tokens=10, output_tokens=5)

    def complete_json(self, prompt: str, schema: dict, system: str | None = None,
                      max_tokens: int = 256) -> dict[str, Any]:
        return self.complete_json_with_usage(prompt, schema, system, max_tokens)[0]


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """Point every path at a tmpdir, and propagate it to modules holding a ref.

    `CONFIG` is a frozen module-level singleton that other modules import by value,
    so rebinding `config.CONFIG` alone would leave them on the old object. Patching
    each importer is the honest cost of that design; the alternative (passing config
    everywhere) was rejected because it makes every call site noisier than the
    problem warrants.
    """
    import agentic_rag.agent
    import agentic_rag.ingest.chunker
    import agentic_rag.ingest.pipeline
    import agentic_rag.registry
    import agentic_rag.retrieve
    import agentic_rag.store

    test_cfg = replace(
        config_module.CONFIG,
        storage_dir=tmp_path,
        upload_dir=tmp_path / "uploads",
        registry_path=tmp_path / "registry.sqlite",
        qdrant_path=tmp_path / "qdrant",
        cache_dir=tmp_path / "cache",
        collection="test_docs",
        qdrant_url="",
        top_k=5,
        candidate_k=20,
        max_hops=3,
        chunk_chars=300,
        chunk_overlap=50,
        min_chunk_chars=20,
        max_context_chunks=10,
        hybrid=True,
        answer_on_exhausted=True,
    )
    for mod in (
        config_module, agentic_rag.store, agentic_rag.retrieve, agentic_rag.registry,
        agentic_rag.ingest.pipeline, agentic_rag.ingest.chunker, agentic_rag.agent,
    ):
        monkeypatch.setattr(mod, "CONFIG", test_cfg, raising=False)
    test_cfg.ensure_dirs()
    return test_cfg


@pytest.fixture
def store(cfg):
    from agentic_rag.store import VectorStore

    s = VectorStore(path=cfg.qdrant_path, url="", collection=cfg.collection)
    yield s
    s.close()


@pytest.fixture
def retriever(cfg, store):
    from agentic_rag.retrieve import Retriever

    return Retriever(store=store, embedder=FakeEmbedder())


@pytest.fixture
def pipeline(cfg, retriever):
    from agentic_rag.ingest.pipeline import IngestionPipeline
    from agentic_rag.registry import Registry

    return IngestionPipeline(retriever, Registry(cfg.registry_path))


@pytest.fixture
def sample_md() -> bytes:
    return (
        b"# Vendor Report\n\n"
        b"## Concentration\n\n"
        b"Tanaka Components Ltd supplies 63 percent of components, which the board "
        b"has flagged as a concentration risk requiring mitigation this year.\n\n"
        b"## Mitigation\n\n"
        b"The agreed cap is 40 percent by the end of Q1 2026, and Priya owns the "
        b"vendor diversification plan that gets us there.\n"
    )
