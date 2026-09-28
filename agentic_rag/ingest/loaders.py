"""File bytes -> `Section`s. One loader per format, and nothing else in the file.

A loader's only job is to get text out along with **where in the document it
was**. It does not chunk, embed, or clean aggressively. Keeping it that narrow is
what makes adding a format a contained change rather than a refactor.

The rule that shapes every loader here: **emit the smallest unit the format gives
you a location for.** A PDF gives pages, so one Section per page. A Markdown file
gives headings, so one Section per heading block. A spreadsheet gives sheets, a
deck gives slides. The chunker downstream is then free to split or merge, but it
can never invent provenance that the loader threw away — and a citation that says
"somewhere in this 200-page PDF" is not a citation.

Text extraction from PDFs is genuinely lossy. `pypdf` returns nothing for scanned
pages, so `load_pdf` reports how many pages came back empty and the pipeline
surfaces that rather than indexing silence. OCR is a deliberate non-goal for now
(see DECISIONS D20).
"""

from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from agentic_rag.models import Section, SourceType

EXTENSION_TYPES: dict[str, SourceType] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".pptx": "pptx",
    ".xlsx": "xlsx",
    ".xlsm": "xlsx",
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".text": "text",
    ".rst": "text",
    ".csv": "text",
    ".log": "text",
    ".html": "html",
    ".htm": "html",
}

SUPPORTED_EXTENSIONS = tuple(sorted(EXTENSION_TYPES))


class UnsupportedFormat(Exception):
    """Raised for a file we cannot read, with the list of ones we can."""


class EmptyDocument(Exception):
    """Raised when a file parsed fine but yielded no usable text.

    A distinct exception on purpose: "I cannot read .zip" and "this PDF is a scan
    of a fax" are different problems for the user, and collapsing them into one
    error message sends people to the wrong fix.
    """


class CorruptDocument(Exception):
    """Raised when the file is the claimed format but the parser cannot read it.

    Its own type because the alternative is what the first version did: let
    `pypdf`'s `PdfReadError: startxref not found` escape as a 500. A 500 says "this
    service is broken"; the truth was "this file is damaged, try re-exporting it",
    and only one of those tells the user what to do. Every loader below funnels its
    library's exceptions through here.
    """


class FetchFailed(Exception):
    """Raised when a URL could not be fetched. Carries what the server said.

    Separate from CorruptDocument because the fix is different: a 403 means "this
    site refuses automated fetches, save the page and upload it", a 404 means "check
    the link", and a timeout means "try again". Reporting all three as a 500 tells
    the user their own service is broken, which is both unhelpful and untrue.
    """


def _parse(fn, label: str):
    """Run a parser, converting library-specific failures into CorruptDocument."""
    try:
        return fn()
    except (EmptyDocument, UnsupportedFormat):
        raise
    except Exception as e:  # noqa: BLE001 — the library's exception types are its own business
        raise CorruptDocument(
            f"This file could not be read as {label}. It may be damaged, encrypted, "
            f"or not actually {label}. ({type(e).__name__}: {e})"
        ) from e


@dataclass
class LoadResult:
    sections: list[Section]
    source_type: SourceType
    title: str = ""
    n_pages: int | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def n_chars(self) -> int:
        return sum(len(s.text) for s in self.sections)


# --- text hygiene -----------------------------------------------------------

_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}


def clean_text(s: str) -> str:
    """Normalise without destroying meaning.

    Deliberately conservative. Every "clever" cleaning step is a chance to delete
    something that turns out to matter — and because the same text is both
    embedded and shown next to a citation, anything removed here is removed from
    what the user is told the source says.

    What we do fix: Unicode normalisation (so "fi" as a ligature matches "fi" as
    two letters at query time), PDF hyphens splitting a word across a line break,
    non-breaking spaces, and runs of blank lines.
    """
    s = unicodedata.normalize("NFKC", s)
    for lig, repl in _LIGATURES.items():
        s = s.replace(lig, repl)
    s = s.replace(" ", " ").replace("​", "")
    # "informa-\ntion" -> "information". Only when a lowercase letter is split,
    # so hyphenated compounds at a line end ("well-\nknown") survive.
    s = re.sub(r"([a-z])-\n([a-z])", r"\1\2", s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def _nonempty(sections: list[Section]) -> list[Section]:
    return [s for s in sections if s.text.strip()]


# --- PDF --------------------------------------------------------------------


def load_pdf(data: bytes, filename: str = "") -> LoadResult:
    """One Section per page, because a page number is what a citation needs."""
    from pypdf import PdfReader

    reader = _parse(lambda: PdfReader(io.BytesIO(data)), "a PDF")
    if reader.is_encrypted:
        # An empty password unlocks a surprising number of "protected" PDFs.
        try:
            reader.decrypt("")
        except Exception as e:  # noqa: BLE001
            raise EmptyDocument(f"PDF is password-protected and could not be opened: {e}") from e

    sections: list[Section] = []
    empty_pages = 0
    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:  # noqa: BLE001 — one broken page must not lose the document
            raw = ""
        text = clean_text(raw)
        if text:
            sections.append(Section(text=text, page=i, ordinal=i - 1))
        else:
            empty_pages += 1

    warnings: list[str] = []
    n_pages = len(reader.pages)
    if empty_pages:
        warnings.append(
            f"{empty_pages} of {n_pages} pages had no extractable text "
            "(likely scanned images; OCR is not enabled)"
        )
    if not sections:
        raise EmptyDocument(
            f"No extractable text in any of {n_pages} pages. This is probably a "
            "scanned PDF, which needs OCR."
        )

    meta_title = ""
    try:
        meta_title = (reader.metadata or {}).get("/Title", "") or ""
    except Exception:  # noqa: BLE001
        pass
    title = clean_text(str(meta_title))[:200] or Path(filename).stem

    return LoadResult(sections, "pdf", title=title, n_pages=n_pages, warnings=warnings)


# --- Word -------------------------------------------------------------------


def load_docx(data: bytes, filename: str = "") -> LoadResult:
    """Group paragraphs under their nearest preceding heading.

    .docx has no pages — pagination is decided by the renderer, not stored — so
    trying to report a page number here would mean inventing one. Headings are the
    honest unit of location for this format.
    """
    import docx  # python-docx

    doc = _parse(lambda: docx.Document(io.BytesIO(data)), "a Word document")
    sections: list[Section] = []
    heading: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if buf:
            text = clean_text("\n\n".join(buf))
            if text:
                sections.append(Section(text=text, heading=heading, ordinal=len(sections)))
            buf.clear()

    for para in doc.paragraphs:
        text = (para.text or "").strip()
        style = (para.style.name or "") if para.style else ""
        if style.startswith("Heading") and text:
            flush()
            heading = text
            continue
        if text:
            buf.append(text)
    flush()

    # Tables carry a lot of the real content in business documents; skipping them
    # silently is a common and expensive omission.
    for t_i, table in enumerate(doc.tables, start=1):
        rows = [
            " | ".join(cell.text.strip() for cell in row.cells)
            for row in table.rows
            if any(c.text.strip() for c in row.cells)
        ]
        if rows:
            sections.append(
                Section(
                    text=clean_text("\n".join(rows)),
                    heading=f"Table {t_i}",
                    ordinal=len(sections),
                )
            )

    sections = _nonempty(sections)
    if not sections:
        raise EmptyDocument("No text found in the .docx (no paragraphs and no tables).")
    core_title = ""
    try:
        core_title = doc.core_properties.title or ""
    except Exception:  # noqa: BLE001
        pass
    return LoadResult(sections, "docx", title=core_title or Path(filename).stem)


# --- PowerPoint -------------------------------------------------------------


def load_pptx(data: bytes, filename: str = "") -> LoadResult:
    """One Section per slide; slide number doubles as the page for citations."""
    from pptx import Presentation

    prs = _parse(lambda: Presentation(io.BytesIO(data)), "a PowerPoint file")
    sections: list[Section] = []
    deck_title = ""

    for i, slide in enumerate(prs.slides, start=1):
        title = ""
        try:
            if slide.shapes.title is not None:
                title = (slide.shapes.title.text or "").strip()
        except Exception:  # noqa: BLE001
            pass
        if i == 1:
            deck_title = title

        parts: list[str] = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                txt = (shape.text_frame.text or "").strip()
                if txt and txt != title:
                    parts.append(txt)
            if getattr(shape, "has_table", False):
                parts.extend(
                    " | ".join(c.text.strip() for c in row.cells) for row in shape.table.rows
                )
        # Speaker notes often hold the argument the slide only gestures at.
        try:
            if slide.has_notes_slide:
                notes = (slide.notes_slide.notes_text_frame.text or "").strip()
                if notes:
                    parts.append(f"[Speaker notes] {notes}")
        except Exception:  # noqa: BLE001
            pass

        text = clean_text("\n".join(p for p in parts if p))
        if text or title:
            sections.append(
                Section(
                    text=text or title,
                    page=i,
                    heading=title or f"Slide {i}",
                    ordinal=i - 1,
                )
            )

    sections = _nonempty(sections)
    if not sections:
        raise EmptyDocument("No text found on any slide (an image-only deck needs OCR).")
    return LoadResult(
        sections, "pptx", title=deck_title or Path(filename).stem, n_pages=len(prs.slides)
    )


# --- Excel ------------------------------------------------------------------


def load_xlsx(data: bytes, filename: str = "", max_rows: int = 2000) -> LoadResult:
    """One Section per sheet, rows rendered as pipe-delimited lines.

    Spreadsheets are the worst fit for embedding-based retrieval — a grid of
    numbers has almost no semantic surface. Rows are emitted with their header row
    repeated as a prefix line per block so a retrieved fragment is still
    interpretable on its own, which is the minimum needed for a citation to mean
    anything.
    """
    from openpyxl import load_workbook

    wb = _parse(
        lambda: load_workbook(io.BytesIO(data), read_only=True, data_only=True),
        "an Excel workbook",
    )
    sections: list[Section] = []
    warnings: list[str] = []

    for sheet in wb.worksheets:
        rows: list[str] = []
        truncated = False
        for r_i, row in enumerate(sheet.iter_rows(values_only=True)):
            if r_i >= max_rows:
                truncated = True
                break
            cells = ["" if v is None else str(v).strip() for v in row]
            if any(cells):
                rows.append(" | ".join(cells).strip(" |"))
        if truncated:
            warnings.append(f"sheet {sheet.title!r} truncated at {max_rows} rows")
        if rows:
            sections.append(
                Section(
                    text=clean_text("\n".join(rows)),
                    heading=sheet.title,
                    ordinal=len(sections),
                )
            )
    wb.close()

    sections = _nonempty(sections)
    if not sections:
        raise EmptyDocument("Every sheet in the workbook was empty.")
    return LoadResult(sections, "xlsx", title=Path(filename).stem, warnings=warnings)


# --- Markdown / plain text --------------------------------------------------

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$", re.MULTILINE)


def load_markdown(data: bytes | str, filename: str = "") -> LoadResult:
    """Split on headings, so each Section is one semantically coherent block.

    This is why Markdown often retrieves *better* than PDF despite being simpler:
    the author already marked where the topic boundaries are, and splitting there
    beats any fixed-size window we could guess at.
    """
    text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
    text = clean_text(text)
    if not text:
        raise EmptyDocument("File is empty.")

    matches = list(_MD_HEADING.finditer(text))
    doc_title = ""
    sections: list[Section] = []

    if not matches:
        return LoadResult(
            [Section(text=text, ordinal=0)], "markdown", title=Path(filename).stem
        )

    # Text before the first heading is still content, not preamble to discard.
    if matches[0].start() > 0:
        lead = text[: matches[0].start()].strip()
        if lead:
            sections.append(Section(text=lead, ordinal=0))

    for i, m in enumerate(matches):
        level, heading = len(m.group(1)), m.group(2).strip()
        if level == 1 and not doc_title:
            doc_title = heading
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end() : end].strip()
        # The heading is prepended to the body it labels, because the body is what
        # gets embedded and "Revenue fell 12%" is far more findable when the chunk
        # also contains the words "Q3 Financial Results".
        block = f"{heading}\n\n{body}" if body else heading
        sections.append(Section(text=block, heading=heading, ordinal=len(sections)))

    return LoadResult(
        _nonempty(sections), "markdown", title=doc_title or Path(filename).stem
    )


def load_text(data: bytes | str, filename: str = "") -> LoadResult:
    text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
    text = clean_text(text)
    if not text:
        raise EmptyDocument("File is empty.")
    # Blank-line-separated blocks are the only structure plain text offers.
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    sections = [Section(text=b, ordinal=i) for i, b in enumerate(blocks)]
    return LoadResult(sections or [Section(text=text)], "text", title=Path(filename).stem)


# --- HTML -------------------------------------------------------------------


def load_html(data: bytes | str, filename: str = "", url: str = "") -> LoadResult:
    """Strip boilerplate, then extract the article body.

    Raw HTML-to-text is the wrong tool for a web page: nav bars, cookie banners, and
    footers are lexically rich and semantically worthless, so indexing them means
    "Accept all cookies" competes with your content at query time.

    The order here matters and was arrived at the hard way. Handing the raw page to
    trafilatura and trusting its boilerplate detection let `<nav>Home | Careers</nav>`
    through on short pages — its heuristics need enough body text to tell chrome from
    content, and a small page does not give them that. So the structural chrome is
    removed by tag first, which is unambiguous, and trafilatura then works on what is
    left. Its output is preferred when it finds something, because it is much better
    than `get_text()` at dropping in-article clutter; `get_text()` is the fallback.
    """
    from bs4 import BeautifulSoup

    html = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
    soup = BeautifulSoup(html, "html.parser")

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # Removed by tag, not by heuristic. None of these ever hold the document.
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header",
                     "aside", "form", "iframe", "svg", "button"]):
        tag.decompose()
    cleaned = str(soup)

    body = ""
    try:
        import trafilatura

        extracted = trafilatura.extract(cleaned, include_comments=False, include_tables=True)
        if extracted:
            body = extracted
        if not title:
            meta = trafilatura.extract_metadata(cleaned)
            if meta and meta.title:
                title = meta.title
    except Exception:  # noqa: BLE001 — fall through to the simpler path
        pass

    if not body:
        body = soup.get_text("\n")

    body = clean_text(body)
    if not body:
        raise EmptyDocument(
            "No article text found in the HTML — the page may render its content "
            "with JavaScript, which this loader does not execute."
        )
    # Extracted article text is Markdown-ish, so reuse the heading splitter.
    result = load_markdown(body, filename=filename)
    return LoadResult(
        result.sections,
        "url" if url else "html",
        title=title or result.title or (url or Path(filename).stem),
    )


# --- dispatch ---------------------------------------------------------------

_LOADERS = {
    "pdf": load_pdf,
    "docx": load_docx,
    "pptx": load_pptx,
    "xlsx": load_xlsx,
    "markdown": load_markdown,
    "text": load_text,
    "html": load_html,
}


def detect_source_type(filename: str) -> SourceType:
    ext = Path(filename).suffix.lower()
    if ext not in EXTENSION_TYPES:
        raise UnsupportedFormat(
            f"Cannot read {ext or 'files with no extension'!r}. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return EXTENSION_TYPES[ext]


def load_bytes(data: bytes, filename: str) -> LoadResult:
    """Entry point for an uploaded file."""
    return _LOADERS[detect_source_type(filename)](data, filename=filename)


def load_path(path: str | Path) -> LoadResult:
    p = Path(path)
    return load_bytes(p.read_bytes(), p.name)


def load_url(url: str, timeout: float = 20.0) -> LoadResult:
    """Fetch a URL and load it. PDFs served over HTTP are handled too."""
    import httpx

    # A bare tool name gets 403'd by a lot of sites, Wikipedia included. A
    # browser-shaped User-Agent that still identifies the tool is both honest and
    # actually served.
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; agentic-rag/1.0; +https://github.com/) "
            "Python-httpx document ingestion"
        ),
        "Accept": "text/html,application/xhtml+xml,application/pdf,text/plain;q=0.9,*/*;q=0.8",
        "Accept-Language": "en",
    }
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout, headers=headers) as client:
            resp = client.get(url)
            resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        hint = {
            401: "the page requires signing in",
            403: "the site blocks automated fetching — save the page and upload the file instead",
            404: "the page was not found; check the link",
            429: "the site is rate-limiting; wait and retry",
        }.get(code, "the server refused the request")
        raise FetchFailed(f"Could not fetch {url} — HTTP {code}: {hint}.") from e
    except httpx.TimeoutException as e:
        raise FetchFailed(f"Timed out after {timeout:.0f}s fetching {url}.") from e
    except httpx.HTTPError as e:
        raise FetchFailed(f"Could not reach {url}: {type(e).__name__}.") from e
    ctype = resp.headers.get("content-type", "").lower()

    if "application/pdf" in ctype or url.lower().split("?")[0].endswith(".pdf"):
        out = load_pdf(resp.content, filename=url)
        return LoadResult(out.sections, "url", title=out.title or url,
                          n_pages=out.n_pages, warnings=out.warnings)
    if "text/plain" in ctype:
        out = load_text(resp.content, filename=url)
        return LoadResult(out.sections, "url", title=url)
    return load_html(resp.text, filename=url, url=url)
