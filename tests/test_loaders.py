"""Loaders: text hygiene, provenance, and the errors users are meant to act on."""

from __future__ import annotations

import io

import pytest

from agentic_rag.ingest.loaders import (
    CorruptDocument,
    EmptyDocument,
    UnsupportedFormat,
    clean_text,
    detect_source_type,
    load_bytes,
    load_html,
    load_markdown,
    load_text,
)


class TestCleanText:
    def test_rejoins_hyphen_split_across_lines(self):
        assert clean_text("informa-\ntion") == "information"

    def test_keeps_real_hyphenated_compounds(self):
        """'well-\\nknown' must not become 'wellknown' — only lowercase splits join,
        and a compound whose second half is a word still reads correctly either way,
        so the conservative rule is the safe one."""
        assert "-" in clean_text("state-of-the-art system")

    def test_normalises_ligatures_so_queries_match(self):
        assert clean_text("ﬁle") == "file"

    def test_collapses_blank_lines_and_nbsp(self):
        assert clean_text("a b\n\n\n\nc") == "a b\n\nc"


class TestMarkdown:
    def test_splits_on_headings_and_prepends_them(self, ):
        r = load_markdown(b"# Doc\n\n## Alpha\n\nbody A\n\n## Beta\n\nbody B")
        headings = [s.heading for s in r.sections]
        assert "Alpha" in headings and "Beta" in headings
        alpha = next(s for s in r.sections if s.heading == "Alpha")
        # The heading is inside the embedded text: "body A" alone is far less findable.
        assert "Alpha" in alpha.text and "body A" in alpha.text

    def test_title_comes_from_the_h1(self):
        assert load_markdown(b"# Vendor Report\n\ntext").title == "Vendor Report"

    def test_text_before_the_first_heading_is_kept(self):
        r = load_markdown(b"preamble matter\n\n## Alpha\n\nbody")
        assert any("preamble" in s.text for s in r.sections)

    def test_no_headings_still_loads(self):
        r = load_markdown(b"just a paragraph of text")
        assert len(r.sections) == 1

    def test_empty_raises(self):
        with pytest.raises(EmptyDocument):
            load_markdown(b"   \n\n  ")


class TestHtml:
    HTML = """<html><head><title>Announcement</title></head><body>
      <nav>Home | Careers</nav><script>track()</script>
      <article><h1>New CTO</h1><p>Priya Raman joins as CTO in February 2024.</p></article>
      <footer>Accept all cookies. Copyright 2025.</footer></body></html>"""

    def test_strips_boilerplate(self):
        """Nav and cookie banners are lexically rich and semantically worthless —
        indexed, they compete with real content at query time."""
        r = load_html(self.HTML)
        body = " ".join(s.text for s in r.sections)
        assert "Priya Raman" in body
        assert "Accept all cookies" not in body
        assert "Careers" not in body

    def test_keeps_the_title(self):
        assert load_html(self.HTML).title in ("Announcement", "New CTO")

    def test_empty_html_raises(self):
        with pytest.raises(EmptyDocument):
            load_html("<html><body><nav>only nav</nav></body></html>")


class TestPlainText:
    def test_splits_on_blank_lines(self):
        r = load_text(b"para one\n\npara two\n\npara three")
        assert len(r.sections) == 3

    def test_sections_have_no_page_numbers(self):
        """Inventing a page number for a format that has none would make a citation
        precise and wrong, which is worse than imprecise and honest."""
        assert all(s.page is None for s in load_text(b"a\n\nb").sections)


class TestDispatchAndErrors:
    def test_detects_type_by_extension(self):
        assert detect_source_type("a.PDF") == "pdf"
        assert detect_source_type("notes.md") == "markdown"

    def test_unsupported_extension_names_what_is_supported(self):
        with pytest.raises(UnsupportedFormat, match=r"\.pdf"):
            detect_source_type("archive.zip")

    def test_no_extension_is_a_clear_error(self):
        with pytest.raises(UnsupportedFormat):
            detect_source_type("README")

    def test_corrupt_pdf_is_not_a_server_error(self):
        """`PdfReadError: startxref not found` as a 500 tells the user their service
        is broken. The truth is 'this file is damaged', and only one of those is
        actionable."""
        with pytest.raises(CorruptDocument, match="damaged"):
            load_bytes(b"%PDF-1.4 but truncated garbage", "broken.pdf")

    def test_fake_docx_is_not_a_server_error(self):
        with pytest.raises(CorruptDocument, match="Word document"):
            load_bytes(b"this is not a zip archive", "fake.docx")


class TestRealOfficeFormats:
    """Round-trip through the actual libraries, since the failure modes live there."""

    def test_docx_groups_under_headings_and_reads_tables(self):
        docx = pytest.importorskip("docx")
        d = docx.Document()
        d.add_heading("Leave Policy", 2)
        d.add_paragraph("Staff accrue 22 days of annual leave each year.")
        t = d.add_table(rows=2, cols=2)
        t.rows[0].cells[0].text = "Band"; t.rows[0].cells[1].text = "Days"
        t.rows[1].cells[0].text = "Senior"; t.rows[1].cells[1].text = "27 days"
        buf = io.BytesIO(); d.save(buf)

        r = load_bytes(buf.getvalue(), "handbook.docx")
        assert any(s.heading == "Leave Policy" for s in r.sections)
        # Tables carry much of the real content in business documents.
        assert any("Senior | 27 days" in s.text for s in r.sections)

    def test_pptx_uses_slide_number_as_page(self):
        pptx = pytest.importorskip("pptx")
        prs = pptx.Presentation()
        for title in ("Roadmap", "Reliability"):
            s = prs.slides.add_slide(prs.slide_layouts[1])
            s.shapes.title.text = title
            s.placeholders[1].text = f"content for {title}"
        buf = io.BytesIO(); prs.save(buf)

        r = load_bytes(buf.getvalue(), "deck.pptx")
        assert [s.page for s in r.sections] == [1, 2]
        assert r.n_pages == 2

    def test_xlsx_one_section_per_sheet(self):
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.Workbook()
        wb.active.title = "Revenue"; wb.active.append(["Region", "USD"]); wb.active.append(["EU", 18400000])
        ws2 = wb.create_sheet("Vendors"); ws2.append(["Vendor", "Share"]); ws2.append(["Tanaka", "63%"])
        buf = io.BytesIO(); wb.save(buf)

        r = load_bytes(buf.getvalue(), "financials.xlsx")
        assert {s.heading for s in r.sections} == {"Revenue", "Vendors"}
        assert any("Tanaka | 63%" in s.text for s in r.sections)
