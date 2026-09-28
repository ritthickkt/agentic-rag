/* Agentic RAG — client.
 *
 * Three jobs: keep the document list in sync, stream an answer while the agent
 * works, and make every citation lead somewhere real.
 *
 * The last one drives most of the code below. An answer arrives as text with [n]
 * markers plus a separate list of citations, each carrying the chunk it came from
 * and the verbatim quote. Rendering turns each marker into a button that opens the
 * source drawer with that quote highlighted inside the full passage. A marker whose
 * quote could not be verified server-side is rendered in amber rather than hidden,
 * because "the model cited something we could not find" is information the reader
 * is entitled to.
 */
"use strict";

const $  = (s) => document.querySelector(s);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

const state = {
  docs: [],
  selected: new Set(),   // empty = search everything
  busy: false,
  chunksById: new Map(), // chunk_id -> chunk, for the drawer
};

/* ── utilities ─────────────────────────────────────────────────────────── */

function toast(msg, kind = "") {
  const t = el("div", `toast ${kind}`, msg);
  $("#toasts").append(t);
  setTimeout(() => {
    t.style.transition = "opacity .25s";
    t.style.opacity = "0";
    setTimeout(() => t.remove(), 250);
  }, kind === "bad" ? 6000 : 3200);
}

const fmtBytes = (n) =>
  n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB`
           : n >= 1e3 ? `${Math.round(n / 1e3)} KB` : `${n} B`;

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body.detail) msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch { /* non-JSON error body; the status line is all we have */ }
    throw new Error(msg);
  }
  return res.status === 204 ? null : res.json();
}

/* ── documents ─────────────────────────────────────────────────────────── */

async function refreshDocs() {
  try {
    const data = await api("/api/documents");
    state.docs = data.documents;
    // A document deleted elsewhere must not linger in the scope filter, or the
    // next query silently searches a doc_id the store no longer has.
    const live = new Set(state.docs.map((d) => d.doc_id));
    for (const id of [...state.selected]) if (!live.has(id)) state.selected.delete(id);
    renderDocs(data.stats);
  } catch (e) {
    toast(`Could not load documents: ${e.message}`, "bad");
  }
}

function renderDocs(stats) {
  const list = $("#doc-list");
  list.replaceChildren();

  const ready = state.docs.filter((d) => d.status === "ready");
  $("#corpus-summary").textContent = ready.length
    ? `${ready.length} document${ready.length === 1 ? "" : "s"} · ${(stats?.n_chunks ?? 0).toLocaleString()} chunks`
    : "no documents yet";
  $("#doc-count").textContent = state.docs.length ? "Documents" : "";
  $("#toggle-all").hidden = ready.length < 2;
  $("#toggle-all").textContent = state.selected.size ? "Clear" : "Select all";
  $("#scope-note").textContent = state.selected.size
    ? `Searching ${state.selected.size} of ${ready.length}`
    : "Searching all documents";

  for (const d of state.docs) {
    const li = el("li", "doc" + (state.selected.has(d.doc_id) ? " picked" : ""));

    if (d.status === "ready") {
      const cb = el("input");
      cb.type = "checkbox";
      cb.checked = state.selected.has(d.doc_id);
      cb.addEventListener("change", () => {
        cb.checked ? state.selected.add(d.doc_id) : state.selected.delete(d.doc_id);
        renderDocs(stats);
      });
      li.append(cb);
    } else if (d.status === "failed") {
      li.append(el("span", "", "⚠"));
    } else {
      li.append(el("div", "spinner"));
    }

    const main = el("div", "doc-main");
    main.append(el("div", "doc-name", d.title || d.filename));
    const meta = el("div", "doc-meta");
    meta.append(el("span", "doc-kind", d.source_type));
    if (d.status === "ready") {
      meta.append(el("span", "", `${d.n_chunks} chunks`));
      if (d.n_pages) meta.append(el("span", "", `${d.n_pages} pages`));
      if (d.n_bytes) meta.append(el("span", "", fmtBytes(d.n_bytes)));
    } else {
      meta.append(el("span", "", d.status));
    }
    main.append(meta);
    if (d.error) main.append(el("div", "doc-err", d.error.slice(0, 160)));
    li.append(main);

    const del = el("button", "doc-del", "×");
    del.title = `Remove ${d.filename}`;
    del.addEventListener("click", async () => {
      if (!confirm(`Remove "${d.filename}" from the index?`)) return;
      try {
        await api(`/api/documents/${d.doc_id}`, { method: "DELETE" });
        state.selected.delete(d.doc_id);
        toast(`Removed ${d.filename}`);
        refreshDocs();
      } catch (e) { toast(e.message, "bad"); }
    });
    li.append(del);
    list.append(li);
  }
}

/* ── upload ────────────────────────────────────────────────────────────── */

async function uploadFiles(files) {
  for (const file of files) {
    // An optimistic row appears immediately: embedding a large PDF takes real
    // time, and an upload with no visible effect reads as a broken button.
    const pending = {
      doc_id: `pending-${crypto.randomUUID()}`, filename: file.name, title: file.name,
      source_type: file.name.split(".").pop().toLowerCase(), status: "indexing",
      n_bytes: file.size, n_chunks: 0, error: "",
    };
    state.docs.unshift(pending);
    renderDocs();

    const body = new FormData();
    body.append("file", file);
    try {
      const out = await api("/api/documents", { method: "POST", body });
      state.docs = state.docs.filter((d) => d.doc_id !== pending.doc_id);
      if (out.duplicate) {
        toast(`${file.name} is already indexed`);
      } else {
        toast(`${file.name} — ${out.document.n_chunks} chunks indexed`, "good");
        (out.warnings || []).forEach((w) => toast(w, "bad"));
      }
      await refreshDocs();
    } catch (e) {
      const row = state.docs.find((d) => d.doc_id === pending.doc_id);
      if (row) { row.status = "failed"; row.error = e.message; }
      renderDocs();
      toast(`${file.name}: ${e.message}`, "bad");
    }
  }
}

function wireUpload() {
  const zone = $("#drop-zone"), input = $("#file-input");
  zone.addEventListener("click", () => input.click());
  zone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
  });
  input.addEventListener("change", () => {
    if (input.files.length) uploadFiles([...input.files]);
    input.value = "";           // so re-picking the same file fires again
  });

  // dragover must be cancelled or the browser navigates to the dropped file
  for (const ev of ["dragenter", "dragover"]) {
    zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("over"); });
  }
  for (const ev of ["dragleave", "drop"]) {
    zone.addEventListener(ev, () => zone.classList.remove("over"));
  }
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    if (e.dataTransfer?.files?.length) uploadFiles([...e.dataTransfer.files]);
  });
  // Dropping anywhere but the zone should do nothing rather than leave the page.
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  $("#url-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const input = $("#url-input"), url = input.value.trim();
    if (!url) return;
    input.disabled = true;
    try {
      const out = await api("/api/documents/url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      toast(out.duplicate ? "Already indexed"
                          : `${out.document.title || url} — ${out.document.n_chunks} chunks`, "good");
      input.value = "";
      refreshDocs();
    } catch (err) {
      toast(err.message, "bad");
    } finally {
      input.disabled = false;
    }
  });

  $("#toggle-all").addEventListener("click", () => {
    if (state.selected.size) state.selected.clear();
    else state.docs.filter((d) => d.status === "ready").forEach((d) => state.selected.add(d.doc_id));
    renderDocs();
  });
}

/* ── rendering an answer ───────────────────────────────────────────────── */

const STEP_LABELS = {
  searching:     "search",
  retrieved:     "retrieved",
  reading:       "reading",
  reformulating: "rethinking",
  answer:        "answering",
  abstained:     "no answer",
};

function stepText(kind, d) {
  switch (kind) {
    case "searching":     return `hop ${d.hop + 1} — "${d.query}"`;
    case "retrieved":     return `${d.n_new} new passage${d.n_new === 1 ? "" : "s"} (${d.n_total} total)`;
    case "reading":       return `weighing ${d.n_passages} passage${d.n_passages === 1 ? "" : "s"}`;
    case "reformulating": return `missing: ${d.missing}`;
    case "answer":        return "drafting with citations";
    case "abstained":     return d.reason;
    default:              return "";
  }
}

/* Escape first, then substitute markers. Doing it the other way round would let a
 * document's own text inject markup into the page. */
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderAnswerHtml(text, citations) {
  const byMarker = new Map(citations.map((c) => [c.marker, c]));
  const paragraphs = escapeHtml(text).split(/\n{2,}/);

  return paragraphs.map((para) => {
    const lines = para.split("\n");
    const isList = lines.every((l) => /^\s*([-*•]|\d+[.)])\s+/.test(l)) && lines.length > 1;
    const withCites = (s) => s.replace(/\[(\d+)\]/g, (m, n) => {
      const c = byMarker.get(Number(n));
      if (!c) return "";                       // a marker with no citation is a dead link
      const cls = c.verified ? "cite" : "cite unverified";
      const tip = `${c.filename}${c.page ? ` · p.${c.page}` : ""}`;
      return `<button class="${cls}" data-marker="${n}" title="${escapeHtml(tip)}">${n}</button>`;
    });
    if (isList) {
      const items = lines.map((l) => `<li>${withCites(l.replace(/^\s*([-*•]|\d+[.)])\s+/, ""))}</li>`).join("");
      return /^\s*\d/.test(lines[0]) ? `<ol>${items}</ol>` : `<ul>${items}</ul>`;
    }
    return `<p>${withCites(lines.join(" "))}</p>`;
  }).join("");
}

function renderResult(turn, result) {
  turn.querySelector(".steps")?.remove();
  const body = el("div");

  if (result.abstained && !result.answer) {
    const box = el("div", "abstained");
    box.append(el("div", "abstained-label", "No answer from your documents"));
    box.append(el("p", "", result.abstain_reason || "The documents do not cover this."));
    body.append(box);
  } else {
    const ans = el("div", "answer");
    ans.innerHTML = renderAnswerHtml(result.answer, result.citations);
    ans.addEventListener("click", (e) => {
      const btn = e.target.closest(".cite");
      if (!btn) return;
      const c = result.citations.find((x) => x.marker === Number(btn.dataset.marker));
      if (c) openDrawer(c);
    });
    body.append(ans);
  }

  // The footer is the honesty layer: confidence, citation audit, and cost.
  const foot = el("div", "answer-foot");
  const v = result.verification || {};
  if (result.confidence) {
    const tone = { high: "good", medium: "", low: "warn" }[result.confidence] ?? "";
    foot.append(el("span", `chip ${tone}`, `${result.confidence} confidence`));
  }
  if (v.n_verified) foot.append(el("span", "chip good", `${v.n_verified} quote${v.n_verified === 1 ? "" : "s"} verified`));
  if (v.n_relocated) foot.append(el("span", "chip warn", `${v.n_relocated} repointed`));
  if (v.n_unverified) foot.append(el("span", "chip bad", `${v.n_unverified} unverified`));
  const u = result.usage || {};
  foot.append(el("span", "chip",
    `${result.hops.length} hop${result.hops.length === 1 ? "" : "s"} · ${u.n_llm_calls} calls · ${u.latency_s}s`));
  body.append(foot);

  // Full retrieval set, collapsed. This is what makes a wrong answer debuggable
  // without reading a log: you can see exactly what the model was shown.
  if (result.chunks?.length) {
    result.chunks.forEach((c) => state.chunksById.set(c.chunk_id, c));
    const det = el("details", "sources");
    det.append(el("summary", "", `${result.chunks.length} passages retrieved`));
    result.chunks.forEach((c, i) => {
      const row = el("div", "src");
      row.append(el("div", "src-rank", `${i + 1}`));
      const b = el("div", "src-body");
      const loc = el("div", "src-loc", c.filename + (c.page ? ` · p.${c.page}` : "") + (c.heading ? ` · ${c.heading}` : ""));
      loc.addEventListener("click", () => openDrawer({
        chunk_id: c.chunk_id, doc_id: c.doc_id, filename: c.filename,
        page: c.page, heading: c.heading, quote: "", verified: true,
      }));
      b.append(loc, el("div", "src-text", c.text));
      const tags = [];
      if (c.dense_rank)   tags.push(`vector #${c.dense_rank}`);
      if (c.lexical_rank) tags.push(`keyword #${c.lexical_rank}`);
      tags.push(`hop ${c.hop + 1}`);
      b.append(el("div", "src-tags", tags.join("  ·  ")));
      row.append(b);
      det.append(row);
    });
    body.append(det);
  }

  turn.append(body);
}

/* ── the source drawer ─────────────────────────────────────────────────── */

function openDrawer(citation) {
  const chunk = state.chunksById.get(citation.chunk_id);
  $("#drawer-title").textContent = citation.filename || "source";
  $("#drawer-loc").textContent = [
    citation.page ? `page ${citation.page}` : null,
    citation.heading || null,
  ].filter(Boolean).join(" · ");

  const body = $("#drawer-body");
  body.replaceChildren();

  if (citation.quote) {
    const q = el("div", "quote-block");
    q.append(el("div", "quote-label", citation.verified ? "Cited quote — found in source" : "Cited quote — NOT found in source"));
    q.append(el("div", "", `“${citation.quote}”`));
    body.append(q);
  }

  if (chunk) {
    body.append(el("div", "quote-label", "Full passage"));
    const p = el("div", "passage");
    // Highlight the quote inside the passage so the eye lands on it. Falls back to
    // the plain passage when the quote is not a literal substring — which is
    // exactly the case the server already flagged as unverified.
    const idx = citation.quote
      ? chunk.text.toLowerCase().indexOf(citation.quote.toLowerCase().trim())
      : -1;
    if (idx >= 0) {
      p.append(document.createTextNode(chunk.text.slice(0, idx)));
      const m = el("mark", "", chunk.text.slice(idx, idx + citation.quote.trim().length));
      p.append(m, document.createTextNode(chunk.text.slice(idx + citation.quote.trim().length)));
    } else {
      p.textContent = chunk.text;
    }
    body.append(p);
  }

  const doc = state.docs.find((d) => d.doc_id === citation.doc_id);
  if (doc?.stored_path || doc?.source_uri) {
    const actions = el("div", "drawer-actions");
    const a = el("a", "", doc.source_uri ? "Open original page ↗" : "Open original file ↗");
    a.href = doc.source_uri || `/api/documents/${citation.doc_id}/file`;
    a.target = "_blank";
    a.rel = "noopener";
    actions.append(a);
    body.append(actions);
  }

  $("#drawer").classList.add("open");
  $("#drawer").setAttribute("aria-hidden", "false");
}

function closeDrawer() {
  $("#drawer").classList.remove("open");
  $("#drawer").setAttribute("aria-hidden", "true");
}

/* ── asking ────────────────────────────────────────────────────────────── */

async function ask(question) {
  if (state.busy || !question.trim()) return;
  state.busy = true;
  $("#send").disabled = true;
  $("#empty-state")?.remove();

  const turn = el("div", "turn");
  turn.append(el("div", "q", question));
  const steps = el("div", "steps");
  turn.append(steps);
  $("#thread").append(turn);
  turn.scrollIntoView({ block: "start" });

  const addStep = (kind, data) => {
    steps.querySelectorAll(".step.live").forEach((s) => s.classList.remove("live"));
    const row = el("div", "step live");
    row.append(el("div", "step-dot"));
    row.append(el("div", "step-kind", STEP_LABELS[kind] || kind));
    row.append(el("div", "", stepText(kind, data)));
    steps.append(row);
    $("#thread").scrollTop = $("#thread").scrollHeight;
  };

  try {
    const res = await fetch("/api/query/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        doc_ids: state.selected.size ? [...state.selected] : null,
      }),
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);

    // Parse SSE by hand: EventSource cannot issue a POST, and the request carries
    // a JSON body (question + document scope) that does not belong in a URL.
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop();                    // last piece may be incomplete
      for (const frame of frames) {
        const kind = frame.match(/^event: (.+)$/m)?.[1];
        const raw  = frame.match(/^data: (.*)$/m)?.[1];
        if (!kind || raw == null) continue;
        const data = JSON.parse(raw);
        if (kind === "result") {
          renderResult(turn, data);
        } else if (kind === "error") {
          steps.remove();
          turn.append(el("div", "error-box", data.error));
        } else if (kind !== "done") {
          addStep(kind, data);
        }
      }
    }
    steps.querySelectorAll(".step.live").forEach((s) => s.classList.remove("live"));
  } catch (e) {
    steps.remove();
    turn.append(el("div", "error-box", `Request failed: ${e.message}`));
  } finally {
    state.busy = false;
    $("#send").disabled = false;
    $("#question").focus();
  }
}

function wireComposer() {
  const ta = $("#question");
  const grow = () => {
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 180)}px`;
  };
  ta.addEventListener("input", grow);
  ta.addEventListener("keydown", (e) => {
    // Enter sends, Shift+Enter makes a new line.
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); }
  });
  $("#composer").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = ta.value.trim();
    if (!q) return;
    ta.value = "";
    grow();
    ask(q);
  });

  $("#drawer-close").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
}

function renderExamples() {
  const box = $("#examples");
  if (!box) return;
  const ready = state.docs.filter((d) => d.status === "ready");
  const names = ready.slice(0, 2).map((d) => d.title || d.filename);
  const prompts = ready.length
    ? [`Summarise ${names[0]} in three points`,
       names[1] ? `What do ${names[0]} and ${names[1]} disagree about?` : "What are the key risks mentioned?",
       "What is not covered by these documents?"]
    : ["Upload a document to get started"];
  box.replaceChildren();
  for (const p of prompts) {
    const b = el("button", "example", p);
    b.type = "button";
    b.disabled = !ready.length;
    b.addEventListener("click", () => { $("#question").value = p; ask(p); });
    box.append(b);
  }
}

/* ── boot ──────────────────────────────────────────────────────────────── */

(async function init() {
  wireUpload();
  wireComposer();
  await refreshDocs();
  renderExamples();
  try {
    const h = await api("/api/health");
    if (h.status === "degraded" && h.documents.n_documents === 0) {
      // Not an error — an empty index is the expected first-run state.
      $("#corpus-summary").textContent = "no documents yet";
    }
  } catch { /* health is advisory; the app works without it */ }
  $("#question").focus();
})();
