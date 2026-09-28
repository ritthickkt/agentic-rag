"""HTTP surface: status codes, error messages, and the streaming contract.

These tests deliberately skip the app's lifespan and populate `app.state` by hand.
That is not a shortcut around startup — it is the point. Startup loads a 130MB
embedding model and opens a Qdrant directory; a suite that paid that cost per test
would be slow enough that nobody runs it, and it would test the model rather than the
API. Constructing the state by hand means these tests assert exactly what the HTTP
layer contributes: validation, status codes, and serialisation.
"""

from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient

from agentic_rag.agent import Answerer
from agentic_rag.ingest.pipeline import IngestionPipeline
from agentic_rag.llm import Completion
from agentic_rag.registry import Registry
from tests.conftest import ScriptedLLM


class UsageStub:
    def to_dict(self):
        return {"calls": 0, "cache_hits": 0, "input_tokens": 0,
                "output_tokens": 0, "retries": 0, "cost_usd": 0.0}


class LLMStub(ScriptedLLM):
    """A ScriptedLLM that also carries the `usage` attribute `/api/health` reads."""

    def __init__(self, responses):
        super().__init__(responses)
        self.usage = UsageStub()


@pytest.fixture
def client(cfg, retriever, monkeypatch):
    import api.main as main

    registry = Registry(cfg.registry_path)
    pipeline = IngestionPipeline(retriever, registry)
    llm = LLMStub([])

    monkeypatch.setattr(main, "CONFIG", cfg)
    app = main.app
    app.state.retriever = retriever
    app.state.registry = registry
    app.state.pipeline = pipeline
    app.state.llm = llm
    app.state.answerer = Answerer(retriever=retriever, llm=llm)
    app.state.reconciled = {}
    # No `with`: the lifespan never runs, so no model is loaded.
    return TestClient(app)


def set_script(client, responses):
    llm = LLMStub(responses)
    client.app.state.llm = llm
    client.app.state.answerer = Answerer(retriever=client.app.state.retriever, llm=llm)
    return llm


class TestDocuments:
    def test_upload_returns_201_and_the_record(self, client, sample_md):
        r = client.post("/api/documents",
                        files={"file": ("vendors.md", io.BytesIO(sample_md), "text/markdown")})
        assert r.status_code == 201
        body = r.json()
        assert body["document"]["status"] == "ready"
        assert body["document"]["n_chunks"] > 0
        assert body["duplicate"] is False

    def test_reupload_reports_duplicate(self, client, sample_md):
        for _ in range(2):
            r = client.post("/api/documents",
                            files={"file": ("v.md", io.BytesIO(sample_md), "text/markdown")})
        assert r.json()["duplicate"] is True

    def test_unsupported_format_is_415_and_lists_what_works(self, client):
        r = client.post("/api/documents",
                        files={"file": ("a.zip", io.BytesIO(b"PK"), "application/zip")})
        assert r.status_code == 415
        assert ".pdf" in r.json()["detail"]

    def test_corrupt_file_is_422_not_500(self, client):
        r = client.post("/api/documents",
                        files={"file": ("x.pdf", io.BytesIO(b"%PDF-1.4 junk"), "application/pdf")})
        assert r.status_code == 422
        assert "damaged" in r.json()["detail"]

    def test_empty_upload_is_400(self, client):
        r = client.post("/api/documents",
                        files={"file": ("e.txt", io.BytesIO(b""), "text/plain")})
        assert r.status_code == 400

    def test_list_includes_stats(self, client, sample_md):
        client.post("/api/documents", files={"file": ("v.md", io.BytesIO(sample_md), "text/markdown")})
        body = client.get("/api/documents").json()
        assert len(body["documents"]) == 1
        assert body["stats"]["n_chunks"] > 0

    def test_delete_then_gone(self, client, sample_md):
        doc_id = client.post(
            "/api/documents", files={"file": ("v.md", io.BytesIO(sample_md), "text/markdown")}
        ).json()["document"]["doc_id"]
        assert client.delete(f"/api/documents/{doc_id}").json()["deleted"] is True
        assert client.get(f"/api/documents/{doc_id}").status_code == 404

    def test_delete_missing_is_404(self, client):
        assert client.delete("/api/documents/nope").status_code == 404

    def test_original_file_is_downloadable(self, client, sample_md):
        doc_id = client.post(
            "/api/documents", files={"file": ("v.md", io.BytesIO(sample_md), "text/markdown")}
        ).json()["document"]["doc_id"]
        r = client.get(f"/api/documents/{doc_id}/file")
        assert r.status_code == 200 and b"Tanaka" in r.content

    def test_url_must_be_http(self, client):
        r = client.post("/api/documents/url", json={"url": "file:///etc/passwd"})
        assert r.status_code == 422


class TestQuery:
    def test_answer_carries_citations_and_usage(self, client, sample_md):
        client.post("/api/documents", files={"file": ("v.md", io.BytesIO(sample_md), "text/markdown")})
        set_script(client, [{
            "insufficient": False,
            "answer": "Tanaka supplies 63 percent [1].",
            "citations": [{"passage": 1, "quote": "Tanaka Components Ltd supplies 63 percent"}],
            "missing": "", "confidence": "high",
        }])
        body = client.post("/api/query", json={"question": "Who supplies the most?"}).json()
        assert body["abstained"] is False
        assert body["citations"][0]["marker"] == 1
        assert body["usage"]["n_llm_calls"] == 1
        assert "verification" in body

    def test_abstention_is_a_200_with_a_reason(self, client, sample_md):
        """Declining is a correct outcome, not an error. A 4xx here would make every
        honest refusal look like a client mistake."""
        client.post("/api/documents", files={"file": ("v.md", io.BytesIO(sample_md), "text/markdown")})
        set_script(client, [{"insufficient": True, "answer": "", "citations": [],
                             "missing": "the dividend policy", "confidence": "low"}])
        # max_hops=1 so the single attempt is also the final one; with more hops the
        # loop would reformulate and this test would be asserting a different path.
        r = client.post("/api/query",
                        json={"question": "What is the dividend policy?", "max_hops": 1})
        assert r.status_code == 200
        assert r.json()["abstained"] is True
        assert r.json()["abstain_reason"] == "the dividend policy"

    def test_blank_question_is_422(self, client):
        assert client.post("/api/query", json={"question": "   "}).status_code == 422

    def test_bounds_are_enforced(self, client):
        """These forward into prompt construction and API spend."""
        assert client.post("/api/query", json={"question": "hi", "max_hops": 99}).status_code == 422
        assert client.post("/api/query", json={"question": "hi", "top_k": 0}).status_code == 422

    def test_doc_scope_is_honoured(self, client, sample_md):
        a = client.post("/api/documents",
                        files={"file": ("a.md", io.BytesIO(sample_md), "text/markdown")}).json()
        client.post("/api/documents",
                    files={"file": ("b.md", io.BytesIO(b"# Boats\n\nAll about sailing boats here."),
                                    "text/markdown")})
        set_script(client, [{"insufficient": False, "answer": "x", "citations": [],
                             "missing": "", "confidence": "high"}])
        body = client.post("/api/query", json={
            "question": "what is here?", "doc_ids": [a["document"]["doc_id"]]
        }).json()
        assert {c["doc_id"] for c in body["chunks"]} == {a["document"]["doc_id"]}


class TestSearchAndOps:
    def test_search_needs_no_llm(self, client, sample_md):
        """The first question when an answer is wrong is 'was the evidence there?'"""
        client.post("/api/documents", files={"file": ("v.md", io.BytesIO(sample_md), "text/markdown")})
        set_script(client, [])          # any LLM call would raise
        body = client.post("/api/search", json={"question": "Tanaka"}).json()
        assert body["n_results"] > 0
        assert "dense_rank" in body["results"][0]

    def test_health_reports_degraded_on_an_empty_index(self, client):
        """'Reachable' is not 'working'. An empty index answers nothing."""
        body = client.get("/api/health").json()
        assert body["status"] == "degraded"
        assert body["documents"]["n_documents"] == 0

    def test_health_ok_once_populated(self, client, sample_md):
        client.post("/api/documents", files={"file": ("v.md", io.BytesIO(sample_md), "text/markdown")})
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert body["embedding"]["vector_id"].startswith("fake:")
        assert ".pdf" in body["supported_formats"]

    def test_ui_is_served(self, client):
        r = client.get("/")
        assert r.status_code == 200 and "Agentic RAG" in r.text


class TestStreaming:
    def test_events_then_result(self, client, sample_md):
        client.post("/api/documents", files={"file": ("v.md", io.BytesIO(sample_md), "text/markdown")})
        set_script(client, [{
            "insufficient": False, "answer": "Tanaka [1].",
            "citations": [{"passage": 1, "quote": "Tanaka Components Ltd supplies 63 percent"}],
            "missing": "", "confidence": "high",
        }])
        with client.stream("POST", "/api/query/stream",
                           json={"question": "who supplies?"}) as r:
            assert r.status_code == 200
            assert "text/event-stream" in r.headers["content-type"]
            raw = "".join(r.iter_text())

        kinds = [l.split(" ", 1)[1] for l in raw.splitlines() if l.startswith("event: ")]
        assert kinds[0] == "searching"
        assert "result" in kinds
        payload = json.loads(raw.split("event: result\ndata: ")[1].splitlines()[0])
        assert payload["citations"][0]["verified"] is True

    def test_agent_failure_becomes_an_error_event_not_a_hang(self, client, sample_md):
        client.post("/api/documents", files={"file": ("v.md", io.BytesIO(sample_md), "text/markdown")})
        set_script(client, [])          # ScriptedLLM raises on first call
        with client.stream("POST", "/api/query/stream", json={"question": "q?"}) as r:
            raw = "".join(r.iter_text())
        assert "event: error" in raw
