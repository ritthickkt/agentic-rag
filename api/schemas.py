"""Request and response shapes for the HTTP API.

Pydantic models rather than loose dicts, for one reason that matters more than
tidiness: this is the boundary where untrusted input arrives. `top_k`, `max_hops`,
and `doc_ids` are all forwarded into retrieval and prompt construction, so they get
bounded here — a request asking for `max_hops=500` is a request to spend the
account's money, and a request asking for `top_k=100000` is a request to build a
prompt that will not fit.

Response models are deliberately *not* enforced as pydantic output types. The agent
already produces `to_dict()` payloads with a stable shape, and re-declaring every
nested field here would mean two definitions of the same thing drifting apart. The
documented shape lives on the dataclasses in `agentic_rag/models.py`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    # Restrict the search to specific documents — "ask this contract" rather than
    # "ask everything I have ever uploaded".
    doc_ids: list[str] | None = Field(default=None, max_length=200)
    top_k: int | None = Field(default=None, ge=1, le=50)
    max_hops: int | None = Field(default=None, ge=1, le=6)

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question cannot be blank")
        return v


class UrlIngestRequest(BaseModel):
    url: str = Field(min_length=8, max_length=2000)

    @field_validator("url")
    @classmethod
    def _http_only(cls, v: str) -> str:
        v = v.strip()
        # Without this, `file:///etc/passwd` and `http://169.254.169.254/` (the
        # cloud metadata endpoint) are both reachable by anyone who can reach the
        # API. Restricting the scheme is the minimum; a deployment exposed to the
        # internet needs egress rules too (see DECISIONS D24).
        if not v.lower().startswith(("http://", "https://")):
            raise ValueError("url must start with http:// or https://")
        return v


class DeleteResponse(BaseModel):
    deleted: bool
    doc_id: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    embedding: dict[str, Any]
    store: dict[str, Any]
    documents: dict[str, Any]
    models: dict[str, str]
    config: dict[str, Any]
