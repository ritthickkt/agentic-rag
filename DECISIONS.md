# Decisions

Every choice that a number depends on. Format: what, why, what we'd revisit.

---

### D1 — Package manager: `venv` + `pip`, not `uv`
**Date:** 2026-09-20
`uv` is not installed on this machine (`which uv` → not found). Rather than add an
install step before line one of the project, we use the stdlib `venv` with `pip`.
`requirements.txt` is loose-pinned (`>=`) because this is a learning project where
a newer `sentence-transformers` is more helpful than a reproducible lockfile.
**Revisit if:** results stop reproducing across machines. Then pin exactly, or move
to `uv` + `uv.lock`.

---

### D2 — One config object, no scattered `os.environ`
**Date:** 2026-09-20
`agentic_rag/config.py` is the only module allowed to read the environment. Every
run writes `CONFIG.to_dict()` into its results file.
**Why:** the entire point of Phase 1 is trustworthy numbers. A number you can't
attribute to a specific `top_k` / model / seed is not a number, it's a vibe.
**Alternative rejected:** argparse flags per script. More flexible, but then the
settings live in shell history instead of in the results file.

---

### D3 — `eval/` does not import `agentic_rag/`
**Date:** 2026-09-20
The harness takes a *callable* `question -> SystemAnswer`. It knows nothing about
Qdrant, the baseline prompt, or Anthropic.
**Why:** in Phase 2 we compare naive RAG against agentic RAG. If the harness had
baseline-specific code in it, that comparison would be measuring the harness as
much as the systems.
**Cost:** slightly more ceremony now (a typed contract in `eval/types.py`).

---

### D4 — Generation model and judge model are the same env var
**Date:** 2026-09-20
`ANTHROPIC_MODEL`, default `claude-haiku-4-5-20251001`.
**Why:** cheap and fast for a first baseline; one knob to swap.
**Known risk — self-preference bias:** a model judging its own output tends to
score it higher than a neutral judge would. We are accepting this for now because
we mainly care about *relative* movement (naive vs agentic), which is affected
less than absolute scores.
**Revisit at step 5:** consider a stronger, different judge model
(e.g. `claude-sonnet-5`) and spot-check ~30 judgements by hand against the labels.

---

### D5 — Anthropic SDK is 1.x
**Date:** 2026-09-20
`pip` resolved `anthropic-1.7.0`. The 1.x line differs from 0.x (httpx2, awaited
async raw responses, Text Completions removed). Nothing we write touches those,
but code samples you find online written for `anthropic` 0.x may not apply.
Also installed: `torch 2.14.0` (CPU/MPS — pulled in by sentence-transformers),
`qdrant-client 1.19.1`, `datasets 5.0.1`.

---

### D6 — Setup check must reject the placeholder API key
**Date:** 2026-09-20
First version of `scripts/check_setup.py` tested only that `ANTHROPIC_API_KEY`
was non-empty, so the `sk-ant-...` placeholder copied from `.env.example` passed.
Now rejected explicitly.
**Why this matters beyond the bug:** it is the same class of error we are trying
to avoid in the eval harness — a check that reports success without testing the
thing it claims to test. A green light you can't trust is worse than a red one.

---

### D7 — An "unanswerable" question is a real question with its evidence deleted
**Date:** 2026-09-26
We do not hand-write unanswerable questions. We take a real HotpotQA item and
delete its two supporting paragraphs from the corpus. The question stays fluent
and well-formed; the evidence simply is not there.
**Why:** a hand-written unanswerable question ("What is the capital of Atlantis?")
is detectable from the question text alone. A system could score well on
abstention while having learned nothing about checking its own retrieval. With
deletion, the question is indistinguishable from an answerable one until you look
at what came back from the index — which is exactly the skill we want to measure.
**Revisit if:** we want to also cover the "question is nonsense" case. That's a
different failure mode and deserves its own bucket, not a blend with this one.

---

### D8 — Deletion cannot be allowed to corrupt an answerable label
**Date:** 2026-09-26
HotpotQA titles recur across items: a paragraph deleted to make question A
unanswerable may be the gold evidence for question B. Left unchecked, B is now
unanswerable *while still labelled answerable* — a silently wrong label, which is
worse than a missing one because every downstream number inherits it.
`select_items()` therefore maintains two disjoint sets (`excluded`, `required`)
and refuses any item that would cross them. 5 source items were skipped on this
basis. The invariant holds by construction **and** is re-verified against the
files on disk at the end of the build.
**Why verify from disk:** same lesson as D6. The artefact later steps consume is
the file, so that is what the check must read.

---

### D9 — Corpus is the union of kept items' paragraphs (5,670), not all of Wikipedia
**Date:** 2026-09-26
Each HotpotQA item ships 10 paragraphs (2 gold + 8 distractors). Our corpus is the
union over the 600 kept items, minus the 238 deleted titles.
**Why:** every answerable question's evidence is guaranteed present, so a retrieval
miss is a retrieval failure and not a corpus gap. It also embeds in minutes on this
laptop.
**Cost, stated plainly:** retrieval here is *easier* than production RAG — 5.7k
short paragraphs, with roughly 8 plausible distractors per question rather than
millions. Absolute recall/accuracy will be optimistic. This is acceptable because
Phase 2 is a *relative* comparison under identical conditions, but no number from
this project should be quoted as "what RAG scores."
**Revisit if:** we claim anything absolute. Then index the full HotpotQA Wikipedia
dump and re-run.

---

### D10 — Known contamination: the model may answer an unanswerable item from memory
**Date:** 2026-09-26
Deleting the paragraphs removes the evidence from *the corpus*, not from the
model's weights. Asked "Juan Martín del Potro is a professional of what
nationality?" with no retrieved evidence, Haiku may well say "Argentinian" and be
right — while doing precisely the thing we want to penalise: answering without
grounding.
**Consequence for metrics (step 4):** abstention on unanswerable items is scored on
*behaviour*, never on whether the guess happened to match `oracle_answer`. That
field exists for human inspection and no metric may read it. We will additionally
report "answered-without-evidence rate" so this failure is visible rather than
averaged away.
**Revisit if:** the unanswerable bucket looks suspiciously easy for the baseline.
That would be a signal that parametric memory is doing the work, and the honest fix
is a corpus whose facts postdate the model, not a prompt tweak.

---

### D11 — dev/test split is stratified and `test` stays sealed
**Date:** 2026-09-26
`TEST_FRAC=0.30` → dev 420 / test 180, stratified on `answerable` so both carry
the same 20% abstain rate. All items are HotpotQA `level: hard` (the distractor
validation split contains only hard items), and the bridge/comparison mix is
479/121 across both.
**Why stratify:** an unstratified 180-item draw could easily land at 15% or 26%
unanswerable, and then a test-set delta partly measures how many abstention
opportunities each split happened to contain.
**The rule:** every number we look at while building comes from `dev`. `test` is
read once, at the end of Phase 2, and whatever it says is what we report.

---

### D12 — The judge is injected, not imported
**Date:** 2026-09-26
`eval/judge.py` defines the judge's prompts, schemas, and parsing, but takes a
`JsonCompleter` callable. `scripts/run_eval.py` hands it
`agentic_rag.llm.LLM.complete_json`.
**Why:** D3 says `eval/` never imports `agentic_rag/`. A judge that constructed its
own Anthropic client would have broken that, and with it the guarantee that the
harness contributes identically to every system's score. It also means the harness
can be tested with a stub judge and no API spend, which is how `run_system` was
verified before a cent was spent.
**Also:** the judge gets its **own** `LLM` instance. Sharing one would fold judge
tokens into the system's cost figure, and "agentic costs 3x naive" would silently
be part judge.

---

### D13 — `temperature` goes through `extra_body` on SDK 1.x
**Date:** 2026-09-26
`anthropic` 1.7.0 removed `temperature` from the typed `messages.create()`
signature — the current frontier models reject sampling parameters outright.
Haiku 4.5 still honours it, so `llm.py` passes `extra_body={"temperature": 0.0}`.
Verified against the live API.
**Revisit if:** we move the generation model to Opus/Sonnet 5, where the parameter
is rejected with a 400. Determinism would then have to come from caching alone.
**Also settled by the same probe:** Haiku 4.5 supports structured outputs
(`output_config.format`), so the grader, reformulator, and judge return
schema-validated JSON instead of prose we parse hopefully. One less failure mode
between a model and a number.

---

### D14 — Model ID: pin the dated snapshot
**Date:** 2026-09-26
`claude-haiku-4-5` and `claude-haiku-4-5-20251001` both resolve to the same
snapshot (confirmed: `response.model` is the dated string either way). Current
Anthropic guidance is to use the unsuffixed alias. We keep the dated one in `.env`.
**Why:** an alias is a moving target, and every results file embeds this string as
its provenance. For a project whose output is numbers, "which model produced this"
must be answerable a year from now.

---

### D15 — The ablation ladder, and why `naive_abstain` exists
**Date:** 2026-09-26
Four systems: `naive`, `naive_abstain`, `graded`, `agentic`.
`naive_abstain` is `naive` plus one sentence of prompt permitting abstention.
**Why this rung is not optional:** LEARNING step 1 gave the baseline exactly one
handicap — it is not told it may decline. But if the agentic system is also the
only one *allowed* to decline, then its entire gain on the unanswerable bucket is
unattributable: it could be the architecture inspecting its evidence, or it could
be one sentence. The data settled it — that sentence alone is worth **+6.0 points**
on the unanswerable bucket, at zero extra cost. Without this control, those six
points would have been credited to the architecture, and the project's headline
claim would have been wrong.

---

### D16 — Abstention is scored as "declined by any route"
**Date:** 2026-09-26
`SystemAnswer.abstained` records only *structural* abstentions (the token, or the
grader gate). But `naive`, which is never told it may decline, still refuses in
prose sometimes — and `naive` declines **89.3%** of unanswerable items that way.
Counting those as hallucinations would have overstated the baseline's failure rate
by a wide margin.
So `eval/metrics.effective_abstention()` is `structural OR judge says no_answer`,
and both rates are reported side by side.
**Why it matters:** flattering the system you want to beat is as much a distortion
as flattering the one you built. Had we used the structural flag alone, `agentic`
would have looked like +96 points on that bucket instead of +7.1.

---

### D17 — `Usage` is money spent; `Completion` tokens are architecture cost
**Date:** 2026-09-26
The LLM disk cache now stores token counts and returns them on a hit, but a cache
hit does **not** add to `Usage`. Two different questions: "what did this run cost
me" (excludes cache hits) versus "what does this architecture consume per
question" (does not care whether we paid today).
**The bug this fixed:** the first full run reported `input tokens / question` as
**0.00** for `graded` and **148.93** for `agentic` — the grader and reformulator
tokens were never accumulated, and cached completions returned no counts at all.
Corrected figures: 804 / 1567 / 3572 for naive / graded / agentic. The error ran
in exactly the direction that flattered the system we built, which is the direction
these errors always seem to run.

---

### D18 — Finding: the grader, not retrieval, is the bottleneck. Not tuned on dev yet.
**Date:** 2026-09-26
Full dev results (420 questions). `agentic` beats `naive` on the overall behaviour
score by **+0.2 points** — a tie — while costing **3.01x** the LLM calls.
Underneath that tie:
- Retrieval improved a lot: `recall_all` 74.7% → **88.4%** (+13.7). Reformulation
  works, and does the thing it was designed to do.
- The gain does not convert. The grader declines on **17.8%** of answerable items
  *where both gold paragraphs were retrieved* — a false negative, a question it
  could have answered and threw away.
- Worst case is the exhausted loop: items that used all 3 hops still have 84.2%
  `recall_all` but abstain **94.7%** of the time. "Ran out of hops" is being
  treated as "no evidence", which is a bug in the control flow, not a property of
  the data.
**What we are deliberately NOT doing yet:** tuning the grader prompt against these
dev numbers until `agentic` wins. That is the overfitting trap LEARNING step 1
named, and the honest sequence is to fix the *identified defect* (the exhausted
loop abstains unconditionally) rather than to search prompt-space for a number.
**Revisit at the next step:** make the final hop answer with whatever evidence it
has rather than abstain, and re-measure. Prediction to check ourselves against:
that alone should recover most of the 22.3% answerable-abstain rate.

---

### D19 — Judge audit: ~2.5% lenient, ~0% harsh
**Date:** 2026-09-26
`scripts/check_judge.py` on dev: the judge agrees with token-F1>0.5 on 91.7%
(naive) and 88.9% (agentic) of committed answers. It rejected an exact string
match 0 times for naive and once for agentic (0.4%). It accepted a zero-F1 answer
2.3–2.7% of the time, and reading those by hand, most are correct
("fifty words"/"fifty-word", "Ulver"/"Kristoffer Rygg").
**One genuine judge error found:** for "What was the real name of the author…",
gold `Charles Lutwidge Dodgson`, the system answered `Lewis Carroll` and the judge
marked it correct — but the question asked for the *real* name, so the pen name is
wrong. Leniency of roughly this size applies to all systems equally, so it does not
threaten the comparison; it does mean absolute accuracy is overstated by low
single digits.
**Also noted:** some HotpotQA gold labels are worse than the model's answer (gold
`international football competition` vs answer `FIFA Women's World Cup`). A ceiling
on any absolute number from this dataset.

---

# Phase 3 — from benchmark to product

The project now serves a real application: drop in documents, ask questions, get cited
answers. The benchmark is retained and repointed at the served engine, so it becomes a
regression gate rather than the deliverable.

---

### D20 — No OCR. A scanned PDF is reported, not indexed as silence
**Date:** 2026-09-27
`pypdf` returns an empty string for a page that is an image. `load_pdf` counts those
pages, warns with the count when some are empty, and raises `EmptyDocument` with an
actionable message when *all* are.
**Why this over adding OCR:** Tesseract or a vision model would roughly triple the
dependency surface and make ingestion minutes-long per document, and the failure it
prevents is one the user can fix themselves once told. The unacceptable outcome is the
silent one — indexing a 200-page scan as zero chunks and reporting success. That is the
same class of error as D6: a green light you cannot trust.
**Revisit if:** users actually upload scans. Then OCR belongs in a background worker,
not in the request path.

---

### D21 — Embedded Qdrant, one process, with a documented escape hatch
**Date:** 2026-09-27
Default store is embedded Qdrant in `storage/qdrant`. It takes an **exclusive lock on
its directory**, so the API and the CLI cannot both hold it, and `uvicorn --workers 2`
would crash rather than load-balance. `scripts/serve.py` therefore does not offer
`--workers`, and `QDRANT_URL` switches to a real server when that limit matters.
**Why accept it:** the alternative is that "run the app" starts with "install Docker".
For the actual default use — one person, one laptop, a few hundred documents — a
directory and `python scripts/serve.py` is the right trade, and the constraint is one
line of config away from gone.
**Also settled here:** the benchmark keeps its own store directory (`qdrant_storage`,
via `CONFIG.eval_qdrant_path`). Sharing one would mean an eval run and the API could
never be up together, and the two hold unrelated corpora anyway.

---

### D22 — Answering and grading are one call, not two
**Date:** 2026-09-27
The served agent asks the model, in a single call, to answer with citations *or* to set
`insufficient` and name the missing fact. The separate grader that gates the answer —
the `graded` and `agentic` rungs — is retained only in `systems.py` for the benchmark.
**Why:** D18 measured what the separate grader costs. It declined **17.8% of answerable
items where both gold paragraphs were provably in the context window**, and items that
exhausted the hop budget abstained **94.7%** of the time while still holding both gold
paragraphs **84.2%** of the time. A judge that never has to produce an answer is free
to be maximally cautious, and "ran out of hops" was being reported to the user as
"there is no evidence."
**The two fixes, together:** sufficiency is decided by something that has already tried
to answer, and the final hop is *told* it is final (`ANSWER_ON_EXHAUSTED`) so partial
answers are delivered instead of withheld. D18 predicted this would recover most of the
22.3% answerable-abstain rate; the product run recorded in LEARNING is the check.
**Cost:** the ablation ladder no longer decomposes cleanly onto the served system. A
`product` vs `agentic` delta is a delta between two different systems, not the value of
one isolated change. Stated in `bench.py` so nobody reads it as an ablation.

---

### D23 — Hybrid retrieval, fused by rank; BM25 in memory
**Date:** 2026-09-27
Dense + BM25, combined with Reciprocal Rank Fusion (`RRF_K=60`).
**Why hybrid:** embeddings map an unseen token to something vaguely nearby, so
`ERR_CONN_4021`, `Section 14(b)`, and part numbers — cases where the user *knows the
string* — are exactly what dense retrieval is worst at. Verified on the fixture corpus:
the exact-code query ranks the right chunk #1 in both retrievers, while a paraphrase
query is carried by the dense half alone.
**Why RRF rather than weighted scores:** cosine sits on 0–1 and BM25 is unbounded and
corpus-dependent, so any numeric combination invents a weight that is wrong as soon as
the corpus changes. RRF uses only rank, so there is nothing to tune per corpus.
**The limit, stated up front:** `rank_bm25` holds the tokenised corpus in RAM and is
rebuilt on every write. Fine into the low hundreds of thousands of chunks; above that
the answer is Qdrant's server-side sparse vectors, not a bigger laptop.
**Also:** the tokeniser deliberately keeps internal `-`, `_`, and `.` so `v2.1.3` and
`covid-19` survive as single terms. Shattering them would remove the only advantage the
lexical half has.

---

### D24 — A marker IS a passage number
**Date:** 2026-09-27
The answer schema has no `marker` field. The model cites passage 7 as `[7]`, and
`citations` entries carry only `{passage, quote}`.
**The bug that forced this.** The first version let the model choose marker numbers and
report which passage each referred to. It produced: *"Tanaka accounts for [1][4] 63
percent … cap at 40 percent [2] … Priya drafting the plan [3]"* — where citation 4 was
the Priya quote, 2 was the spreadsheet row, and 3 was the 40% cap. **Every quote
verified**, because each really was in the chunk it named. What broke was the mapping
from *claim* to *marker*, and no server-side check can catch that: it requires knowing
which claim the author meant. Removing the second numbering removes the only place the
mismatch could live.
**Cost:** markers are not sequential in the prose — an answer may cite [2], [7], [3].
This is also how the UI numbers its retrieved-passages list, so they line up.
**The general lesson:** when a model must keep two representations consistent and only
one is checkable, delete one of them rather than validating the other.

---

### D25 — Citation verification has three outcomes, and all three are reported
**Date:** 2026-09-27
Each quote is checked against the chunk it cites. **verified** = found there.
**relocated** = found in a different retrieved chunk, so the marker is repointed and
the move is counted. **unverified** = found nowhere; the citation is kept but flagged
`verified: false`, rendered in amber, and counted in the API response.
**Why not drop an unverified citation:** the marker is already in the prose. Removing
the citation without removing the marker leaves a dead link; removing both silently
deletes the model's own admission that it was leaning on something. Surfacing it lets a
caller refuse the answer, and a rising unverified rate is the earliest signal of trouble.
**A bug the tests caught:** the length guard originally applied only to the fuzzy path,
so a 7-character quote like `"revenue"` cited against passage 3 was *relocated* to
passage 1 and stamped verified. Searching a short generic string across the context
finds a hit almost every time regardless of whether the citation was real. Short quotes
are now checked **only** against the chunk the model actually named.
**Known gap:** verification proves quote↔chunk, never quote↔claim. A correct quote
attached to the wrong sentence still passes. D24 removes the mechanical cause; the
semantic case remains open and is why `confidence` and the full passage list are
returned.

---

### D26 — The embedder serialises its own calls
**Date:** 2026-09-27
`LocalEmbedder` holds a lock around every `model.encode`.
**Why it is not optional:** `SentenceTransformer` makes no thread-safety promises, and
calling one from several threads does not raise — **the interpreter segfaults**. Found
by running the benchmark with 4 workers against the served engine (exit 139); it would
have hit the API identically, since FastAPI serves sync endpoints on a thread pool, on
the second concurrent question.
**Why on the embedder rather than at each call site:** there are already three callers
(retriever, ingestion pipeline, bench index builder) and adding a fourth should not
require knowing this. The cost is nil — encoding a query is single-digit milliseconds
against seconds per LLM call.

---

### D27 — Ingestion writes the registry row before parsing
**Date:** 2026-09-27
`pipeline._index` sets `pending`, then `indexing`, parses, chunks, embeds, upserts, and
only then flips to `ready`. A failure writes `failed` plus the reason and deletes any
partial vectors.
**Why this ordering:** a document is never `ready` without its vectors, and a failure
always leaves something on screen explaining itself. An upload that vanishes silently
is the worst outcome — the user cannot tell whether it worked, retry sensibly, or
report anything useful.
**The gap that remains, and its fix:** a hard kill between the upsert and the status
flip leaves vectors whose row is not `ready`. Those would be retrieved and cited from a
document the user cannot see or delete, so `reconcile()` runs at every boot and removes
orphaned vectors, and marks any `ready` row with zero vectors as failed.
**Also:** dedup is keyed on `(content_hash, vector_id)`, not hash alone. The same bytes
under a different embedding model is a genuinely different index entry, not a duplicate.

---

### D28 — Errors are typed by what the user should do about them
**Date:** 2026-09-27
`UnsupportedFormat` → 415. `EmptyDocument` / `CorruptDocument` / `FetchFailed` → 422.
`VectorSpaceMismatch` → 409. Only a genuine internal fault reaches 500.
**Why bother:** the first version let `PdfReadError: startxref not found` escape as a
500. A 500 tells the user the service is broken; the truth was "this file is damaged,
try re-exporting it", and only one of those is actionable. `FetchFailed` goes further
and maps the upstream status to advice — a 403 means "this site blocks automated
fetching, save the page and upload it", which is a different action from a 404.
**Related:** abstention returns **200**. Declining is a correct outcome, and a 4xx
there would make every honest refusal look like a client mistake.

---

### D29 — Result: the served engine wins, and the prediction was 40% right
**Date:** 2026-09-28
Full dev run, 420/420, zero errors. `product` scores **82.1%** overall against
`naive` 75.7% and `agentic` 76.0`%`, at **2.39** LLM calls/question versus `agentic`'s
3.01 — better *and* cheaper than the previous best. `recall_all` 93.5% (hybrid
retrieval), answerable accuracy 78.9%, unanswerable declined 95.2%, citations 99.8%
verified with 1 unverified out of 546.
**Checking the prediction.** D18/D22 predicted that killing the separate grader and
telling the final hop it is final would "recover most of the 22.3% answerable-abstain
rate". It recovered **8.9 of 22.3 points — 40%, not most.** Recorded as stated rather
than rounded up, because a prediction written in advance is only worth something if a
miss is reported as a miss.
**What is left, precisely.** Items using all 3 hops retrieve both gold paragraphs
**97.5%** of the time — the highest of any bucket — and still abstain **65.0%** of the
time (was 94.7% in D18). This is no longer the control-flow bug D22 fixed: the final
hop is told it is final and does answer when it can.

**Two hypotheses proposed and both killed by looking.** Recorded because the wrong
guesses are the useful part of this entry.

1. *"The final call reads its own two prior refusals."* **False, and false about code I
   had just written.** Every hop is a fresh single-turn call: `answer_prompt` contains
   the passages, the question, and the final-attempt note, and nothing else. There is no
   message history and no prior verdict anywhere in the prompt. The experiment this
   implied — "strip the earlier verdicts" — is a no-op, because that is already the
   behaviour.
2. *"The gold paragraphs are retrieved but pushed out of the context by
   `MAX_CONTEXT_CHUNKS=14`."* Checkable directly, and also false: of the 26 three-hop
   abstentions, **0** lost a gold paragraph to the cap. Gold chunks rank high under RRF,
   and a 3-hop item accumulates a median of 15 chunks against a cap of 14.

So **25 of the 26 declined with both gold paragraphs demonstrably in front of them.**
Reading those 25 by hand, they are not one failure but three:
- **Dataset noise.** "In what year did Farda Amiga win a race at Saratoga?" has gold
  `(foaled February 1, 1999)`, which is her birth date, not a year she won anything; the
  gold paragraphs do not contain the answer to the question asked. "Who has the highest
  scope in profession in Rob Thomas or James Dean Bradfield" is not a well-formed
  question. Declining these is **correct**, and no change can fix them. D19 already
  flagged that some HotpotQA labels are worse than the model's answer.
- **Absence-as-answer on comparisons.** "Are Steve Perry and Dennis Lyxzén both members
  of the same band?", gold `no`. The system wrote: *"The passages do not indicate that
  Steve Perry and Dennis Lyxzén have ever been members of the same band."* That sentence
  **is** the answer, and it filed it as missing evidence. For a yes/no comparison, not
  finding a shared attribute across two passages is exactly the evidence for "no" — the
  prompt currently gives the model no way to see that.
- **Genuine over-caution on joins.** The Kanye/Dwele item had "Power is from *My
  Beautiful Dark Twisted Fantasy*" and Dwele's credit on "Power" both in context, and
  declined because Dwele also appears on other albums. That is a join it could have made.

**Next experiments, in this order, predictions written first.**
*E0 — measure the ceiling before optimising toward it.* Classify all 25 by whether the
gold answer string is even present in its own gold paragraphs. Until that number exists,
"65% abstain at 3 hops" overstates the defect by an unknown amount, and any prompt change
will be credited with fixing questions that were never answerable.
*E1 — absence-as-answer.* Tell the answerer that for a yes/no or comparison question, the
absence of a shared or greater attribute across the passages is itself the answer.
Prediction: comparison-type abstention roughly halves; bridge-type barely moves. If
bridge moves too, the instruction is making the model less careful in general rather than
better at comparisons, and it should be reverted.
**Not doing:** tuning the answer prompt against these dev numbers until the figure
improves. That is the overfitting trap named in LEARNING step 1.
**Two caveats that travel with the headline.** `product` is not an ablation rung — it
changes retrieval, prompt structure, and output format at once, so its margin over
`agentic` is a gap between two systems, not the value of one change. And exact match
(0.0%) and token F1 (18.3%) are meaningless for it: it answers in cited prose against
gold answers that are short spans. `judge_correct` is the only accuracy metric that
survives the format change, which is exactly why D12's judge exists.
**Unexpected, and not tuned for:** confidence is monotonically calibrated —
`high` 91.7% correct (n=277), `medium` 81.8% (n=11), `low` 66.7% (n=3).

---

### D30 — An incomplete run is marked in the file and refused by the comparison
**Date:** 2026-09-28
A `Run` records `n_attempted`, `n_scored`, `completion_rate`, and a `complete` flag.
`run_eval.py` prints a banner above the metrics when a run did not finish, and
`compare.py` skips incomplete runs with a message instead of tabling them.
**The incident.** A dev run died at question 137 of 420 when the API credit balance hit
zero. Every one of the 283 failures was recorded faithfully — and `aggregate` computed
every rate over the 137 survivors, and the printed summary looked **identical to a
clean run**. The denominator had silently shrunk and nothing said so. The surviving
items also are not a random sample: they are whichever happened to run first.
**Why this is the same bug as D6 and not a new one.** `check_setup.py` passed on a
placeholder API key because "is it set?" is not "is it usable?". Here, "did every
question get an answer?" was never asked, only "did the run finish?" — and a harness
that reports success without testing the thing it claims to test is the failure this
project keeps rediscovering in new clothes.
**Also:** the partial file was moved to `results/partial/` with a README naming the
denominator, rather than deleted. Deleting it would have removed the evidence.
