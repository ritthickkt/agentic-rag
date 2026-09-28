# Learning log

One entry per step. Written so you could say it out loud in an interview.

---

## Step 1 — Project scaffold, config, and why the layout is shaped this way

### The concept: a *system* is a function, and evaluation is separate from it

The core design idea in this project is a boundary. On one side sits a **system**:
anything that takes a question and returns an answer plus the evidence it used. On
the other side sits a **harness**: something that takes a system and produces
numbers. The harness never knows what's inside the system.

Right now there will be exactly one system (naive RAG). Later there will be several
(naive, +grader, +reformulation, full agentic). Because they all satisfy the same
contract, comparing them is a one-line change, and every comparison is run under
identical conditions.

**Jargon defined:**
- **RAG** — Retrieval-Augmented Generation. Instead of asking a language model to
  answer from memory, you first *retrieve* relevant text from your own documents,
  paste it into the prompt, and ask the model to answer using only that. It exists
  because models hallucinate on facts they don't have, and because you can't
  retrain a model every time a document changes.
- **Baseline** — the simplest honest version of the system. Its job is not to be
  good. Its job is to be a fair thing to beat.
- **Harness** — the code that runs a system over a fixed question set and scores it.

### Why the layout is what it is

`agentic_rag/` is a library with no side effects on import. `scripts/` are thin
entrypoints. `eval/` is a *sibling*, not a submodule, and never imports the
baseline — that's a structural guarantee, not a promise. `data/` and `results/`
hold JSONL and JSON so you can read them with `head` instead of a debugger.

### What could go wrong (and this is the real lesson)

The failure mode that ruins RAG portfolio projects is **a baseline that was never
honest**. Two ways it happens:

1. **You tune the baseline while looking at the test set.** Then your final
   "agentic beats naive by 14 points" is partly just overfitting. This is why we
   split `dev`/`test` in step 2 and do not touch `test` until step 6 of a much
   later phase.
2. **You quietly make the baseline worse than it needs to be** so your clever
   system looks better. The specific temptation here is the unanswerable bucket:
   it would be easy to write a baseline prompt that never abstains *and* is
   badly written, then claim the agentic version fixed it. We're deliberately
   omitting the "say I don't know" instruction — but that's the *only* handicap,
   it's stated out loud, and the rest of the baseline prompt will be genuinely
   decent.

A third, quieter risk: **config drift**. You run the baseline with `k=5`, later run
the agentic version with `k=10`, and attribute a retrieval gain to the agent. This
is why every result file embeds the full config snapshot.

### One alternative we didn't choose

**A single `main.py` with everything in it**, or a framework like LangChain doing
the plumbing. The framework version is fewer lines today and much harder to reason
about at step 6, when a number looks wrong and you need to know whether the bug is
in retrieval, the prompt, or the metric. Plain modules mean every number has a
readable path back to the code that produced it. The tradeoff is real: we will
hand-write things (a retry loop, a disk cache) that a framework gives for free.

**Postscript, from actually running it:** the first `check_setup.py` passed on a
placeholder API key, because "the variable is set" is not the same question as
"the value is usable." Worth remembering when we write the metrics: most bad
evals fail this exact way — they measure something adjacent to what they claim.

---

## Step 2 — The eval set, and the typed boundary the harness scores through

### The concept: build the ruler before the thing you measure

Nothing in this step improves the system, because there is no system yet. That
ordering is deliberate. If you build the RAG pipeline first and the evaluation
second, you will — not out of dishonesty, just out of being human — design the
metric that flatters what you already built. Building the ruler first means the
system has to meet a standard it did not get to choose.

Two artefacts came out of this step: a **contract** (`eval/types.py`) and a
**dataset** (`data/`).

### The contract

`eval/types.py` defines four plain data types and one interface:

- `Question` — one labelled item: the question, the gold answer, whether it is
  answerable at all, and which corpus paragraphs count as evidence.
- `Document` — one corpus paragraph, keyed by `doc_id` (the article title).
- `Passage` — one *retrieved* chunk, as the system actually saw it, with its score.
- `SystemAnswer` — what a system returns: the answer text, the passages behind it,
  and an explicit `abstained` flag.
- `System` — the interface, and it is one line: `(question: str) -> SystemAnswer`.

Two of those choices carry real weight:

**`abstained` is a field, not a regex.** The tempting shortcut is to detect
abstention by searching the answer for "I don't know." Then the moment a model
phrases it as "the provided context does not specify," your metric quietly counts
a correct abstention as a wrong answer — and you are now measuring string
formatting rather than behaviour. The system declares what it did.

**`Passage.doc_id` must be the corpus id.** Retrieval metrics compare it against
`Question.gold_titles`. If a system returned a paraphrased or re-chunked id,
recall would read as zero while retrieval was in fact fine.

**Jargon defined:**
- **Abstain** — the system declines to answer, saying the evidence isn't there.
  For an unanswerable question this is the *correct* output, not a failure.
- **Gold / gold evidence** — the human-labelled correct answer, or the paragraphs
  a human marked as the ones you need. "Gold" just means "the label."
- **Distractor** — a paragraph that is topically nearby but does not contain the
  answer. Its job is to be a plausible wrong retrieval.
- **Stratified split** — splitting so each part keeps the same class proportions,
  instead of hoping a random draw lands even.

### The dataset, and the one idea in it worth stealing

600 questions from HotpotQA's distractor validation split: 480 answerable, 120 not.
Corpus: 5,670 paragraphs, the union of the kept items' own 10-paragraph bundles.

The interesting part is how an unanswerable item is made. **We take a real,
perfectly sensible question and delete its two supporting paragraphs from the
corpus.** The question reads exactly like an answerable one. The only way to tell
is to look at what retrieval brought back and notice it does not contain an answer
— which is precisely the skill the agentic version is supposed to have and the
naive baseline is supposed to lack.

Compare the lazy alternative: hand-writing unanswerable questions. "Who was the
first president of Atlantis?" is detectable from the question text alone, so a
system could ace the abstention bucket by learning the *style* of an impossible
question while never once checking its own evidence. The metric would go up and
mean nothing.

### What could go wrong (the actual lesson of this step)

**A silently wrong label.** HotpotQA reuses article titles across items. The
paragraph I delete to make question A unanswerable may be the gold evidence for
question B. Nothing crashes. Nothing warns. Question B is now unanswerable while
still labelled answerable, its gold answer unreachable, and every system will be
marked wrong on it forever. You would see it as "retrieval is oddly bad on some
items" and go optimise retrieval — chasing a bug in the ruler.

So selection keeps two disjoint sets, `excluded` (titles being deleted) and
`required` (titles that must survive), and refuses any item that would cross them.
5 source items got skipped on that basis. Then the build re-reads the three files
*from disk* and asserts both halves of the invariant: every answerable question has
both gold paragraphs present, every unanswerable one has zero. That check reads the
files rather than the in-memory objects for the D6 reason — the file is what later
steps consume, so the file is what must be checked.

**Contamination I can't engineer away.** Deleting a paragraph removes the fact from
the corpus, not from the model's weights. One of the unanswerable items asks del
Potro's nationality; Haiku almost certainly knows it is Argentina. It will answer
confidently with no evidence, and be right — while doing the exact thing we want to
catch. This is why abstention is scored on *behaviour* and never against
`oracle_answer`, and why step 4 will report an explicit
"answered-without-evidence" rate. The clean fix is a corpus whose facts postdate
the model's training; that is a bigger project, and stating the limitation is more
honest than pretending a prompt fixes it.

### One alternative we didn't choose

**SQuAD 2.0**, which ships human-written unanswerable questions for free. It would
have saved this entire selection routine. Rejected because SQuAD questions are
single-paragraph: retrieval is trivial, so there is nothing for a multi-step
agentic system to be *better at*. HotpotQA questions need two documents chained
together, which is where reformulation and iterative retrieval can actually earn
their cost. Choosing the easier dataset would have saved a day and removed the
phenomenon under study.

**Postscript, from actually running it:** the build reports `5 (title conflict)`
skips. Small number, and every one of them was a corrupted label that never made it
into the files.

---

## Step 3 — Retrieval, and the four systems

### The concept: one index, four systems, identical conditions

`agentic_rag/` is now a working RAG stack: `embed.py` (sentence embeddings),
`index.py` (Qdrant in local mode), `prompts/`, and `systems.py` (the four systems).
Every system shares one `Retriever` instance, so there is physically no way for a
`top_k` or embedding-model difference to sneak between two runs and get attributed
to the architecture — LEARNING step 1's third risk, closed structurally rather
than by remembering to be careful.

**Jargon defined:**
- **Dense retrieval** — encode the query and every document as vectors, return the
  documents whose vectors are closest. "Dense" because the vector is a few hundred
  floats that are all meaningful, versus a sparse keyword vector that is mostly
  zeros.
- **Asymmetric embedding model** — one trained so that queries and documents are
  encoded *differently*. `bge-small-en-v1.5` wants a prefix on the query and none
  on the document.
- **Reformulation** — rewriting the search query mid-run, usually to chase a fact
  the first query missed.
- **Ablation** — removing one component to see what it was contributing.

### The asymmetry trap, which is my favourite kind of bug

`bge-small-en-v1.5` expects queries prefixed with `"Represent this sentence for
searching relevant passages: "` and documents bare. Get this wrong and **nothing
fails**. No exception, no warning — retrieval just gets quietly worse, and every
number downstream is a little bit wrong in a way no test would catch.

So `embed.py` exposes `encode_documents()` and `encode_queries()` as separate
methods, and the prefix lives inside the query one. Not because that's tidier, but
because a single `encode()` with a boolean flag is a thing a caller can get wrong,
and this bug is invisible when you do.

The wiring check in `build_index.py` exists for the same reason: if recall came
back near zero, that is a bug, not a hard dataset, and it should stop the pipeline
rather than produce a confident-looking table.

### The number that shaped the whole design

The index smoke test, over 100 answerable dev questions at k=5:

- at least one gold paragraph retrieved: **100.0%**
- both gold paragraphs retrieved: **73.0%**

That gap *is* the multi-hop problem, quantified before writing a single agentic
component. These questions need two documents joined together. The question's
wording looks like hop one, so hop one always surfaces — and hop two's document,
which the question never describes, is missing a quarter of the time. That single
observation is why reformulation is the mechanism the agentic system is built
around, rather than something more elaborate I'd have guessed at.

### The ladder, and the control that saved the project from a wrong claim

Four systems, each one change apart: `naive` → `naive_abstain` → `graded` →
`agentic`. The middle rung is `naive` plus *one sentence* of prompt permitting
abstention, and it is the most important system in the project even though it is
the least interesting.

Here is the trap it defuses. Step 1 committed to one handicap on the baseline: it
isn't told it may decline. If the agentic system were the only one *allowed* to
decline, then every point it gained on unanswerable questions would have two
possible explanations — the architecture inspecting its own evidence, or one
sentence of English — and no way to separate them. I would have written "the
agentic system reduced hallucination by X" and been unable to defend it.

The measurement: that one sentence is worth **+6.0 points** on the unanswerable
bucket, for free. Six points that would otherwise have been credited to
architecture I wrote.

---

## Step 4 — Metrics, and the judge

### The concept: separate what a machine can check from what needs an opinion

`eval/metrics.py` contains no LLM calls — exact match, token F1, retrieval recall.
Anyone with the results file can reproduce every one of those numbers exactly.
`eval/judge.py` contains the part that needs a model's opinion. Splitting them by
file means a reader can see at a glance which numbers are reproducible arithmetic
and which are a model's judgement.

**Why a judge is needed at all:** `token_f1("Argentina", "Argentinian")` is
**0.0**. So is "1994" vs "in 1994". The systems generate free text and the gold
answers are short spans, so string metrics systematically under-credit correct
answers — and they do it *unevenly across systems*, since whichever system happens
to be terser scores higher for reasons unrelated to being right. That's not noise,
it's a bias with a direction.

**Why the judge is injected rather than imported:** `eval/` may not import
`agentic_rag/` (D3). So `judge.py` declares a `JsonCompleter` protocol and the
*script* supplies something that can talk to a model. Two payoffs, one expected and
one not: the harness can be tested with a stub judge for free (which is how it was
verified before any spend), and the judge gets its own `LLM` instance so its tokens
never contaminate the system's cost figure.

### The metric decision I'd defend hardest

Scoring abstention as **"declined by any route"** rather than by the system's own
flag.

`naive` is never told it may decline — yet it declines on **89.3%** of unanswerable
questions anyway, in prose, off its own bat. Had I scored only the structural flag,
`naive` would have registered 0% abstention, `agentic` 96%, and I'd have reported a
*ninety-point* improvement in hallucination avoidance. The true figure is **+7.1**.

The lesson generalizes past this project: flattering the system you want to beat is
exactly as much a distortion as flattering the one you built, and it is much harder
to notice, because the number comes out in the direction you were hoping for.

### And an error in that same direction, in my own code

The first full run reported `input tokens / question` as **0.00** for `graded` and
**148.93** for `agentic`. Both are impossible — those systems make strictly more
calls than `naive`'s 804. Two bugs: the grader and reformulator token counts were
never accumulated, and cached completions returned no counts at all.

Corrected: **804 / 1567 / 3572** tokens per question for naive / graded / agentic.
So the real cost multiple is about **4.4x in tokens**, not the ~0.2x the broken
table implied. The bug ran in precisely the direction that flattered the system I
had just built. They seem to.

The fix also forced a distinction worth keeping: `Usage` counts money actually
spent (a cache hit is free), while the per-question token figures describe what the
architecture consumes regardless of who paid. Two different questions that had been
sharing one counter.

---

## Step 5 — What the comparison actually showed

### The headline, stated the way the data actually came out

| | naive | naive_abstain | graded | agentic |
|---|---|---|---|---|
| answerable: judge correct | **72.3%** | 64.6% | 60.4% | 70.8% |
| answerable: recall_all | 74.7% | 74.7% | 74.7% | **88.4%** |
| answerable: abstained (a miss) | 0.0% | 27.4% | 33.3% | 22.3% |
| unanswerable: declined | 89.3% | 95.2% | **97.6%** | 96.4% |
| **overall behaviour score** | 75.7% | 70.7% | 67.9% | **76.0%** |
| LLM calls / question | 1.00 | 1.00 | 1.58 | 3.01x |

**The agentic system did not beat the baseline.** +0.2 points overall, for 3x the
calls and 4.4x the tokens. On a 420-question dev set that is a tie.

This is the result. I'm writing it up as the result rather than tuning until the
table flatters the component I spent the most time on, because the whole apparatus
in this repo — the sealed test split, the honest baseline, the stated handicap —
exists to make exactly this outcome reportable.

### But the interesting part is *why*, and the diagnostic answers it precisely

Retrieval got substantially better: `recall_all` **74.7% → 88.4%**. Reformulation
does the job it was designed for. The gain simply does not convert into answers.

The diagnostic in `scripts/compare.py` isolates the culprit by asking a question
the headline table cannot: *of the answerable items where both gold paragraphs were
retrieved, how often did the system decline anyway?*

- naive: 0.0% (it cannot decline)
- graded: 17.1%
- **agentic: 17.8%**

Nearly one in five questions where the evidence was provably sitting in the context
window, thrown away by the grader. And the accuracy-given-evidence column tells the
same story from the other side: 88.4% for naive versus 75.8% for agentic — the
agentic system is *worse* at converting good evidence, because a chunk of its good
evidence gets vetoed before the answer call.

The per-hop breakdown finds the actual defect:

| hops | n | recall_all | judge correct | abstained |
|---|---|---|---|---|
| 1 | 241 | 90.9% | 84.2% | 7.1% |
| 2 | 57 | 80.7% | 59.6% | 38.6% |
| 3 | 38 | 84.2% | **2.6%** | **94.7%** |

Items that used all three hops still retrieved both gold paragraphs **84.2%** of
the time — and abstained on **94.7%** of them. That is not a calibration problem
to be smoothed away with prompt tuning. It is a control-flow bug I wrote: when the
loop exhausts `max_hops`, the last grader verdict is still "insufficient", and the
system treats *running out of budget* as *having no evidence*. It declines while
holding the answer.

### What could go wrong (and what I'm refusing to do about it)

The tempting move is to soften the grader prompt and re-run until `agentic` wins.
Dev exists for iteration, so that isn't cheating outright — but searching
prompt-space for a number is how you end up with a system tuned to 420 specific
questions and a "14-point gain" that evaporates on `test`. That's the first failure
mode named in step 1, and it would be embarrassing to walk into it in step 5.

The honest next move is narrower: fix the *identified defect* — make the final hop
answer with whatever evidence it holds instead of abstaining unconditionally — and
re-measure. Written down in advance so I can be wrong in public: that change alone
should recover most of the 22.3% answerable-abstain rate and move `agentic` clearly
ahead of `naive` on the overall score. If it doesn't, the grader's judgement is
worse than its control flow and that's a different (and more interesting) problem.

`test` remains unread.

### One alternative we didn't choose

**Scoring only a single blended accuracy number.** It would have made this write-up
much shorter and the agentic system look fine. It also would have hidden every
finding above: the retrieval gain, the grader false negatives, and the exhausted-loop
bug all live in the *disagreement* between buckets. A system that abstains on
everything and a system that answers everything can post the same blended score
while failing in opposite directions.

---

## Step 6 — From a benchmark into a product

### The concept: the eval stops being the deliverable and becomes the guard rail

Everything up to here produced *numbers about four systems*. Nothing produced something
a person could use. This step builds the actual application — drop in documents, ask
questions, get cited answers — and repoints the benchmark at the thing being shipped.

That inversion is the whole idea. A demo that "worked on the five files I tried" is not
a claim about anything. Once `scripts/run_eval.py --system product` exists, every change
to a prompt, a chunk size, or an embedding provider can be checked against 420 labelled
questions, 20% of which are unanswerable. The eval is no longer the point; it is the
thing that tells a refactor apart from a regression.

**Jargon defined:**
- **Chunk** — a document is too big to hand a model whole, so it is cut into pieces.
  Retrieval can only ever return a chunk, so a fact split across a boundary is
  unreachable no matter how clever the rest of the system is.
- **Provenance** — for each chunk, which file, which page, which heading it came from.
  Carried end to end, or citations are guesses.
- **BM25** — keyword scoring that weights rare words more heavily. The lexical half of
  hybrid retrieval.
- **RRF (Reciprocal Rank Fusion)** — combine two ranked lists using only positions, not
  scores. Avoids inventing a weight between two incomparable number scales.
- **SSE (Server-Sent Events)** — a one-way stream from server to browser, used to show
  the agent's steps while it works rather than a spinner.

### What got built

Ingestion for seven formats, each emitting the smallest unit it has a location for — a
PDF page, a slide, a sheet, a Markdown heading block. A hybrid retriever. An agent loop
that retrieves, tries to answer, and on failure searches for the specific fact it said
was missing. A FastAPI service with a streaming endpoint, a single-page UI, and a CLI.
115 tests that run in three seconds and spend nothing, because the embedder and the LLM
are injected — the same boundary the benchmark already relied on, exercised a second way.

### The four bugs worth writing down

All four were found by building the thing, not by reasoning about it.

**1. Citations that all verified, and were all wrong.** The first answer schema let the
model pick its own marker numbers and report which passage each referred to. It
produced an answer where every quote really was in the chunk it named — verification
passed 4/4 — but the markers were shifted relative to the claims they sat next to.
`[4]` sat on a revenue claim while citation 4 quoted something else entirely. No
server-side check can catch this: it needs to know which claim the author *meant*. The
fix was not a better check, it was deleting the second numbering — a marker is now
simply the passage number. **When a model has to keep two representations consistent
and only one of them is checkable, remove one rather than validating the other.**

**2. A verifier that verified nothing.** A test asked what happens to a 7-character
quote, `"revenue"`, cited against the wrong passage. It was *relocated* to a passage
that did contain the word and stamped verified. Searching a short generic string across
the context finds a hit almost every time, whether or not the citation was real. The
length guard existed but only covered the fuzzy path, not exact relocation. This is the
same shape as the step-1 bug where `check_setup.py` passed on a placeholder API key: a
check that reports success without testing the thing it claims to test.

**3. A segfault hiding in a thread pool.** Running the benchmark against the served
engine with four workers killed the interpreter — exit 139, no traceback.
`SentenceTransformer` is not thread-safe and does not raise; it dies. The old
benchmark retriever had a lock around this and the comment explaining why, and I did
not carry it over when writing the new one. It would have hit production identically,
on the second concurrent question, because FastAPI serves sync endpoints on a thread
pool. The lock now lives on the embedder itself, so the next caller cannot forget it.

**4. `PdfReadError` as a 500.** A damaged PDF returned "500 Internal Server Error" with
a raw library message. A 500 tells the user the service is broken. The truth was "this
file is damaged, try re-exporting it" — and only one of those is something they can act
on. Every loader now funnels its library's exceptions into typed errors chosen by *what
the user should do*: 415 for a format we cannot read, 422 for a file we can identify but
not parse, and for a failed URL fetch, advice keyed to the upstream status (a 403 means
"this site blocks bots, save the page and upload it", which is a different action from
a 404).

### What could go wrong, and what the design does about it

The failure mode specific to a *citing* system is the one this phase is mostly built
around: an answer that looks sourced and is not. It is worse than an uncited answer
because it manufactures the appearance of evidence, and a reader cannot catch it
without going and reading the source — which is the work the citation was supposed to
save them. So verification has three outcomes and all three are reported, including the
bad one: unverified citations are kept, flagged in the API response, and rendered in
amber rather than quietly dropped. A rising unverified rate is the earliest signal that
something has gone wrong.

The gap that remains, stated plainly: verification proves **quote ↔ chunk**, never
**quote ↔ claim**. A correct quote attached to the wrong sentence still passes. Bug 1
removed the mechanical cause of that mismatch; the semantic version is open, and is why
the full retrieved passage list and a confidence level come back with every answer.

### The measurement

Full dev run, 420/420 questions, zero errors.

| | naive | naive_abstain | graded | agentic | **product** |
|---|---|---|---|---|---|
| answerable: judge correct | 72.3% | 64.6% | 60.4% | 70.8% | **78.9%** |
| answerable: recall_all | 74.7% | 74.7% | 74.7% | 88.4% | **93.5%** |
| answerable: abstained (a miss) | 0.0% | 27.4% | 33.3% | 22.3% | **13.4%** |
| unanswerable: declined | 89.3% | 95.2% | 97.6% | 96.4% | **95.2%** |
| **OVERALL** | 75.7% | 70.7% | 67.9% | 76.0% | **82.1%** |
| LLM calls / question | 1.00 | 1.00 | 1.58 | 3.01 | **2.39** |

The served engine beats the naive baseline by **+6.4 points** and the previous best
(`agentic`) by **+6.1**, while making **fewer** LLM calls than `agentic` did — 2.39
per question against 3.01. Hybrid retrieval lifted `recall_all` to 93.5%, and unlike
in step 5 the retrieval gain actually converted this time.

Two caveats that belong next to that number rather than in a footnote. **`product` is
not another rung on the ablation ladder** — it changes retrieval, prompt structure, and
output format all at once, so its margin over `agentic` is a gap between two different
systems, not the value of one isolated change. And **exact match and token F1 are
meaningless for it**: it answers in cited prose, not bare spans, so EM reads 0.0% and
F1 18.3% against gold answers that are short strings. `judge_correct` is the metric
that survives the format change.

### Checking the prediction I wrote down, and getting it partly wrong

D18 predicted, and D22 acted on, this: removing the separate grader and telling the
final hop it is final "should recover most of the 22.3% answerable-abstain rate."

It recovered **8.9 of those 22.3 points — about 40% of the gap, not most of it.**
Directionally right, materially overstated. Writing the prediction down beforehand is
what makes that a finding rather than a thing I could quietly round up.

The per-hop breakdown shows exactly what is left:

| hops | n | recall_all | judge correct | abstained |
|---|---|---|---|---|
| 1 | 237 | 92.8% | 91.6% | 0.0% |
| 2 | 59 | 93.2% | 59.3% | 32.2% |
| 3 | 40 | 97.5% | 32.5% | **65.0%** |

Items that use all three hops now retrieve both gold paragraphs **97.5%** of the time —
the best of any bucket — and still decline **65%** of the time. In step 5 that figure
was 94.7%, so the fix moved it a long way and did not finish the job. The evidence is
provably in the context window and the system will not commit to it.

That is no longer a control-flow bug; the final hop is told it is final and answers
when it can.

**I then guessed at the cause twice and was wrong twice**, which is the part of this
worth keeping. First guess: the final call must be reading its own two previous
refusals. That was wrong *about code I had just written* — every hop is a fresh
single-turn call containing the passages, the question, and nothing else. No history,
no prior verdicts. The experiment it implied was a no-op. Second guess: the gold
paragraphs are retrieved but pushed out of the 14-chunk context cap. Checkable in one
query, and also wrong — **0** of the 26 lost a gold paragraph to the cap.

So 25 of 26 declined with both gold paragraphs demonstrably in front of them. Reading
those 25 by hand instead of theorising about them, they are three different things:

- **Dataset noise.** "In what year did Farda Amiga win a race at Saratoga?" has gold
  `(foaled February 1, 1999)` — her birth date, not a year she won anything. "Who has
  the highest scope in profession in Rob Thomas or James Dean Bradfield" is not a
  well-formed question. Declining these is correct and unfixable.
- **Absence-as-answer.** "Are Steve Perry and Dennis Lyxzén both members of the same
  band?", gold `no`. The system wrote *"The passages do not indicate that Steve Perry
  and Dennis Lyxzén have ever been members of the same band"* — and filed it as missing
  evidence. That sentence **is** the answer. For a comparison, not finding a shared
  attribute is precisely the evidence for "no", and the prompt gives the model no way
  to see that.
- **Genuine over-caution.** The Kanye/Dwele item had both halves of the join in context
  and declined because Dwele also appears on other albums.

The lesson is the one the project keeps re-teaching: **the diagnostic beat the
hypothesis, both times, and the hand-read beat the diagnostic.** Two plausible
mechanisms, each of which would have justified a code change, and neither survived five
minutes of checking against the data. The next step is to *measure* how much of the 65%
is dataset noise before changing anything, because otherwise a prompt tweak gets credit
for questions that were never answerable.

**One thing that came out better than expected:** confidence is monotonically
calibrated. `high` → 91.7% correct (n=277), `medium` → 81.8% (n=11), `low` → 66.7%
(n=3). It was not tuned for; it was asked for and then measured.

**Citation health:** 546 citations across 295 answers, 99.8% verified, 0 relocated,
**1 unverified** (0.2%) — on "Who died first, Bryce Courtenay or Juan Carlos Onetti?",
where the answer was judged correct and one of two date citations could not be matched
verbatim. That is the number this whole phase was built to be able to state.

**Cost of the run:** $1.73 system + $0.14 judge for 420 questions.

`test` remains unread.

### One alternative we didn't choose

**A framework — LangChain or LlamaIndex — for the retrieval and agent plumbing.** It
would have replaced most of `retrieve.py`, `agent.py`, and the loaders with
configuration, and would have been a lot less code today. Three of the four bugs above
were found by reading and instrumenting code I had written: the marker mismatch was
visible because the citation payload is mine, the segfault was traceable because the
embedder call site is one function, and the error mapping was possible because every
loader exception passes through one place. Inside a framework those are a wrong-looking
answer, an unexplained crash, and a 500 — all three still real, all three much harder
to reach. The cost is honest and was paid: a retry loop, a disk cache, a chunker, and a
BM25 index that a framework would have handed over for free.
