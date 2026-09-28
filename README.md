# Agentic RAG

Drop in documents. Ask questions. Get answers where **every claim carries a citation
you can click**, and where "your documents don't say" is a real answer rather than a
confident guess.

```
┌─ documents ──────────┐        ┌─ agent loop ─────────────────────────────┐
│ PDF DOCX PPTX XLSX   │        │                                          │
│ MD TXT HTML URL      │        │   retrieve ──► try to answer ──► done     │
└──────────┬───────────┘        │       ▲              │                    │
           │ parse              │       │         insufficient?             │
           ▼                    │       │              │                    │
   sections (page/heading)      │       └── search for ─┘                   │
           │ chunk              │           what's missing   (≤3 hops)      │
           ▼                    └──────────────────┬───────────────────────┘
   chunks ──► embed ──► Qdrant                     │
                 ▲                                 ▼
          BM25 lexical index            verify every quote against
                                        its source, then answer
```

---

## Why this is not another RAG demo

Three things, each of which is a thing that usually goes wrong.

**1. Citations are verified, not asserted.** The model returns the verbatim quote it
is relying on for each claim, and the server checks that the quote is actually in the
chunk it names. A quote found somewhere else gets the marker repointed; a quote found
nowhere is flagged in the response and rendered in amber in the UI. A fabricated
citation is worse than no citation, because it manufactures the appearance of
evidence — so the pipeline is built to catch it rather than to hope.

**2. Retrieval is hybrid, because embeddings cannot find `ERR_CONN_4021`.** Dense
vectors match meaning, which is what you want for "what were the merger risks" and
useless for an error code, a part number, or `Section 14(b)`. BM25 nails those and is
hopeless at paraphrase. They fail in different places, so both run and their results
are fused by rank (RRF) rather than by score — no per-corpus weight to tune.

**3. The system is benchmarked, not vibe-checked.** There is a 420-question labelled
dev set in which 20% of questions are deliberately **unanswerable** — real questions
whose supporting paragraphs were deleted from the corpus, so they read completely
normally and the only way to know is to actually inspect what retrieval returned. The
served engine is scored on that set by the same harness that scored four simpler
baselines.

| dev, 420 questions | naive | graded | agentic | **served engine** |
|---|---|---|---|---|
| answerable: correct | 72.3% | 60.4% | 70.8% | **78.9%** |
| answerable: retrieved both golds | 74.7% | 74.7% | 88.4% | **93.5%** |
| unanswerable: declined | 89.3% | 97.6% | 96.4% | **95.2%** |
| **overall behaviour score** | 75.7% | 67.9% | 76.0% | **82.1%** |
| LLM calls / question | 1.00 | 1.58 | 3.01 | **2.39** |

Citations on that run: 546 across 295 answers, **99.8% verified, 1 unverified**.
Confidence came out monotonically calibrated without being tuned for —
`high` 91.7% correct, `medium` 81.8%, `low` 66.7%.

`python scripts/run_eval.py --system product` is a regression gate, and it is the only
thing here that can tell a refactor from a regression.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then put your ANTHROPIC_API_KEY in it

python scripts/serve.py       # → http://127.0.0.1:8000
```

Open the UI, drag some files in, ask a question. Or stay in the terminal:

```bash
python scripts/rag.py add ~/Documents/contracts/          # a file, a glob, or a folder
python scripts/rag.py add-url https://example.com/post
python scripts/rag.py ask "what are the termination clauses?"
python scripts/rag.py search "ERR_CONN_4021"              # retrieval only, no LLM call
python scripts/rag.py ls
```

Only `ANTHROPIC_API_KEY` is required. Embeddings run locally by default — no second
API key, nothing leaves the machine except the passages sent to Claude to answer.

> **One process at a time.** The default vector store is embedded Qdrant, which locks
> its directory. Stop the server before running the CLI, or set `QDRANT_URL` to a
> Qdrant server and the limit disappears.

---

## What happens to a question

1. **Retrieve.** The question is embedded and searched against the vector store, and
   the same question is scored by BM25 over the chunk text. Each retriever returns 30
   candidates; RRF fuses them and the top 8 go forward.
2. **Answer, or say what's missing.** One call asks the model to answer with citations
   *or* to set `insufficient` and name the specific fact that is absent. These are the
   same call on purpose — see below.
3. **Loop if needed.** If evidence was insufficient, a second call rewrites the search
   query to target the named gap ("the director's nationality", not the original
   question), retrieval runs again, and the new chunks are **added** to the old ones.
   Up to 3 hops.
4. **Verify.** Every quote is checked against its chunk. Markers that resolve to
   nothing are stripped from the prose so there are no dead links.
5. **Return** the answer, the citations, the full retrieved set, and a per-hop trace of
   what the agent searched for and why.

### The one design decision worth arguing about

Answering and judging sufficiency are **the same LLM call**. The obvious alternative —
a separate "grader" that gates the answer — was built and measured here first. It
declined **17.8% of questions where both required passages were provably already in
the context window.** A judge that never has to produce an answer is free to be
maximally cautious. Asking for the answer and the verdict together means a refusal is
a decision made by something that actually tried.

Relatedly, the final hop is *told* it is the final hop. An earlier version abstained
unconditionally when it ran out of hops — on questions where retrieval had in fact
succeeded 84.2% of the time. It was reporting "ran out of budget" to the user as
"there is no evidence."

---

## Layout

```
agentic_rag/
  config.py        every knob, and the only module that reads os.environ
  models.py        the data types: Section, Chunk, Citation, AnswerResult
  ingest/
    loaders.py     bytes -> sections, one loader per format, provenance preserved
    chunker.py     sections -> chunks, split on structure before size
    pipeline.py    the only writer of both the registry and the vector store
  embed.py         pluggable embedding provider (local bge / Voyage)
  store.py         Qdrant wrapper; refuses a vector-space mismatch
  registry.py      SQLite: one row per document, with status and errors
  retrieve.py      dense + BM25 + RRF fusion
  agent.py         the loop
  citations.py     quote verification — the part that makes citations mean something
  prompts/         answer, grader, reformulate
  bench.py         adapter that makes the served engine scoreable by eval/
api/               FastAPI: upload, query, SSE streaming, search, health
ui/                single-page app, no framework
eval/              the benchmark harness — never imports agentic_rag/
scripts/           rag.py (CLI), serve.py, run_eval.py, build_*.py
tests/             115 tests, ~3s, no API calls
```

`eval/` being a sibling that cannot import `agentic_rag/` is a structural guarantee,
not a convention: the harness takes a callable `question -> SystemAnswer` and is
therefore physically incapable of helping one system more than another.

---

## API

| | |
|---|---|
| `POST /api/documents` | upload a file (multipart) |
| `POST /api/documents/url` | fetch and index a URL |
| `GET /api/documents` | list, with per-document status and chunk counts |
| `DELETE /api/documents/{id}` | remove vectors, row, and stored original |
| `GET /api/documents/{id}/file` | the original file, so a citation can open it |
| `POST /api/query` | ask; returns answer + citations + chunks + trace |
| `POST /api/query/stream` | the same, as server-sent events |
| `POST /api/search` | retrieval only, no LLM call — the debugging endpoint |
| `GET /api/health` | config, store, document counts, embedding identity |

Interactive docs at `/docs`.

```bash
curl -X POST localhost:8000/api/query -H 'Content-Type: application/json' \
  -d '{"question":"which vendor is a concentration risk?"}'
```

```jsonc
{
  "answer": "Tanaka Components Ltd is a concentration risk, accounting for 63 percent of component sourcing [1][4]. …",
  "citations": [
    {"marker": 1, "filename": "quarterly_report.pdf", "page": 2,
     "quote": "supplier concentration: 63 percent of component sourcing flows through a single vendor",
     "verified": true}
  ],
  "verification": {"n_verified": 4, "n_relocated": 0, "n_unverified": 0},
  "confidence": "high",
  "hops": [{"hop": 0, "query": "...", "sufficient": true}],
  "usage": {"n_llm_calls": 1, "input_tokens": 1236, "latency_s": 3.2}
}
```

A marker **is** a passage number. There is deliberately no second numbering scheme —
an earlier version let the model number its markers independently and it produced
answers where every quote was individually correct but the markers were shifted
relative to the claims they sat beside. No server-side check can catch that, so the
degree of freedom was removed instead.

Abstention returns **200**, not an error. Declining is a correct outcome.

---

## Configuration

Everything lives in `.env`; see `.env.example` for the full annotated list. The ones
that change behaviour most:

| | default | |
|---|---|---|
| `TOP_K` | 8 | passages given to the model per hop |
| `CANDIDATE_K` | 30 | pool per retriever before fusion |
| `HYBRID` | true | set false for dense-only |
| `MAX_HOPS` | 3 | 1 disables the loop entirely |
| `CHUNK_CHARS` | 1200 | ~300 tokens |
| `ANSWER_MODEL` | — | a stronger model for answering than for query rewriting |
| `EMBED_PROVIDER` | local | or `voyage` |

Changing `CHUNK_CHARS` or the embedding model invalidates the index. The store detects
the embedder change and refuses rather than returning plausible nonsense; run
`python scripts/rag.py reindex` to rebuild from the stored originals.

---

## The benchmark

```bash
python scripts/build_dataset.py       # 600 HotpotQA items, 20% evidence deleted
python scripts/build_index.py         # index for the four baseline rungs
python scripts/build_bench_index.py   # index the served engine can read
python scripts/run_eval.py --system product --split dev
python scripts/compare.py
```

Scoring is split in two and never averaged into one number, because a system that
abstains on everything and a system that answers everything can post the same blended
score while failing in opposite directions:

- **answerable** — judged correct by an LLM judge, with an abstention counted as a
  miss. Exact match is also reported but is near-zero for the served engine, which
  answers in cited prose rather than bare spans.
- **unanswerable** — did it decline? Counted as "declined by any route", because a
  system never told it may abstain still refuses in prose sometimes, and counting
  those as hallucinations would flatter the system under test by overstating its
  baseline.

`data/test.jsonl` exists and is read once, at the end. Every number looked at while
building comes from `dev`.

### The defect this currently has

Questions that use all three hops retrieve both gold paragraphs **97.5%** of the time
and still decline **65%** of the time. The evidence is provably in the context window
and the system will not commit to it. It is the single largest remaining loss, it is
written up in `DECISIONS.md` D29 with the next experiment and a prediction, and it is
not being fixed by tuning the prompt against the dev set until the number moves.

### Known limits, stated rather than discovered

- The benchmark corpus is 5,670 short paragraphs with ~8 plausible distractors per
  question. Retrieval there is **easier** than production RAG, so absolute numbers are
  optimistic and only the relative comparison is meaningful.
- The judge is currently the same model family that generates answers, which inflates
  scores through self-preference. `scripts/check_judge.py` measures the residual: ~2.5%
  too lenient, ~0% too harsh.
- The model may answer an "unanswerable" question correctly from memory. Abstention is
  therefore scored on *behaviour*, never on whether the guess matched.
- No OCR: a scanned PDF is reported as having no extractable text rather than indexed
  as silence.
- No authentication in front of the API. It binds to localhost by default.

---

## Development

```bash
pytest                                  # 115 tests, ~3s, zero API calls
pytest tests/test_agent.py -v            # agent control flow, scripted LLM
python scripts/rag.py search "query"     # what did retrieval return?
curl localhost:8000/api/health | jq      # what is actually configured?
```

Tests inject a fake embedder and a scripted LLM, so the whole suite runs offline and
free. That is the same injection boundary the benchmark relies on, exercised a second
way.

`DECISIONS.md` records every choice a number depends on, with what would make us
revisit it. `LEARNING.md` is the narrative: what was built, what broke, and what the
measurements actually said.
