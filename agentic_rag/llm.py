"""The only place this project talks to Anthropic.

Three jobs, and they are here rather than scattered through the systems because
every one of them affects whether a number is trustworthy:

1. **Determinism.** `temperature=0` on every call. Two runs of the same system
   over the same questions should differ only because we changed something.
2. **A disk cache.** Keyed on everything that can change the output. Re-running
   the harness after editing a metric costs nothing and returns byte-identical
   generations, so a metric change is measured against a fixed set of outputs
   instead of a fresh sample of them.
3. **Usage accounting.** Every call's tokens are counted so a results file can
   state what the run cost. "The agentic version is better" means little without
   "and it costs 3.1x as much per question."

Note on the SDK (see DECISIONS D13): `anthropic` 1.x removed `temperature` from
the typed `messages.create()` signature, because the current frontier models
reject sampling parameters outright. Haiku 4.5 still honours it, so it goes
through `extra_body`.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic

from agentic_rag.config import CONFIG

# $ per million tokens, Haiku 4.5. Used only to report what a run cost.
PRICE_IN_PER_MTOK = 1.00
PRICE_OUT_PER_MTOK = 5.00


@dataclass
class Usage:
    """Mutable running total, shared by every call through one LLM instance."""

    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    retries: int = 0

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens / 1_000_000 * PRICE_IN_PER_MTOK
            + self.output_tokens / 1_000_000 * PRICE_OUT_PER_MTOK
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "retries": self.retries,
            "cost_usd": round(self.cost_usd, 4),
        }


@dataclass(frozen=True)
class Completion:
    text: str
    from_cache: bool = False
    input_tokens: int = 0
    output_tokens: int = 0


class LLM:
    """Thin, cached, deterministic wrapper around the Messages API."""

    def __init__(
        self,
        model: str | None = None,
        cache_dir: Path | None = None,
        use_cache: bool = True,
        max_retries: int = 5,
    ) -> None:
        self.model = model or CONFIG.anthropic_model
        self.cache_dir = Path(cache_dir or CONFIG.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache
        # The SDK already retries 429/5xx/connection errors with backoff. Let it,
        # rather than hand-rolling a second retry loop on top of its own.
        self.client = anthropic.Anthropic(
            api_key=CONFIG.require_api_key(), max_retries=max_retries, timeout=120.0
        )
        self.usage = Usage()
        self._lock = threading.Lock()

    # --- cache ------------------------------------------------------------

    def _cache_key(self, payload: dict[str, Any]) -> str:
        # sort_keys matters: an unsorted dump would make the key depend on
        # insertion order and silently miss every time.
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_read(self, key: str) -> dict[str, Any] | None:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None  # a truncated cache file is a miss, not a crash

    def _cache_write(self, key: str, record: dict[str, Any]) -> None:
        path = self.cache_dir / f"{key}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)  # atomic: concurrent workers never read a half file

    # --- calls ------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        schema: dict[str, Any] | None = None,
    ) -> Completion:
        """One user turn in, text out.

        `schema` enables structured outputs: the response is then guaranteed to
        validate against it, which is why the grader and judge use it instead of
        asking for JSON in the prompt and hoping.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "schema": schema,
        }
        key = self._cache_key(payload)

        if self.use_cache:
            hit = self._cache_read(key)
            if hit is not None:
                with self._lock:
                    self.usage.calls += 1
                    self.usage.cache_hits += 1
                    # Deliberately NOT added to usage.input_tokens: `Usage` tracks
                    # money actually spent by this run, and a cache hit spends
                    # none. The counts are still returned on the Completion,
                    # because per-question token accounting is a statement about
                    # what the architecture consumes, not about today's invoice.
                    # Without this, a cached rerun would report 0 tokens/question.
                return Completion(
                    text=hit["text"],
                    from_cache=True,
                    input_tokens=hit.get("input_tokens", 0),
                    output_tokens=hit.get("output_tokens", 0),
                )

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            # Removed from the typed signature in SDK 1.x; Haiku 4.5 still takes it.
            "extra_body": {"temperature": temperature},
        }
        if system:
            kwargs["system"] = system
        if schema:
            kwargs["output_config"] = {
                "format": {"type": "json_schema", "schema": schema}
            }

        resp = self._create_with_backoff(kwargs)

        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        with self._lock:
            self.usage.calls += 1
            self.usage.input_tokens += resp.usage.input_tokens
            self.usage.output_tokens += resp.usage.output_tokens

        if self.use_cache:
            self._cache_write(key, {
                "text": text,
                "model": resp.model,
                "key": key,
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            })

        return Completion(
            text=text,
            from_cache=False,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )

    def _create_with_backoff(self, kwargs: dict[str, Any]):
        """Outer guard for the case the SDK's own retries are exhausted.

        With ~20 workers hammering the API, a burst of 429s can outlast the SDK's
        default backoff. Losing a single question to an exception would leave a
        hole in a results file, which is worse than waiting.
        """
        delay = 2.0
        for attempt in range(4):
            try:
                return self.client.messages.create(**kwargs)
            except (anthropic.RateLimitError, anthropic.APIStatusError) as e:
                status = getattr(e, "status_code", 500)
                if isinstance(e, anthropic.APIStatusError) and status < 500 and status != 429:
                    raise  # a 400 will not fix itself by waiting
                if attempt == 3:
                    raise
                with self._lock:
                    self.usage.retries += 1
                time.sleep(delay)
                delay *= 2
            except anthropic.APIConnectionError:
                if attempt == 3:
                    raise
                with self._lock:
                    self.usage.retries += 1
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable")

    def complete_json_with_usage(
        self,
        prompt: str,
        schema: dict[str, Any],
        system: str | None = None,
        max_tokens: int = 256,
    ) -> tuple[dict[str, Any], Completion]:
        """Structured output, parsed, plus the raw Completion.

        Systems use this rather than `complete_json` so that grader and
        reformulator tokens land in the per-question cost figure. Leaving them out
        understated the agentic system's token use by roughly the entire grading
        step — the exact direction that would flatter the system we built.
        """
        out = self.complete(prompt, system=system, max_tokens=max_tokens, schema=schema)
        try:
            return json.loads(out.text), out
        except json.JSONDecodeError as e:
            raise ValueError(f"model returned non-JSON despite schema: {out.text[:200]!r}") from e

    def complete_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        system: str | None = None,
        max_tokens: int = 256,
    ) -> dict[str, Any]:
        """Structured output, parsed. Matches `eval.judge.JsonCompleter`."""
        return self.complete_json_with_usage(prompt, schema, system, max_tokens)[0]
