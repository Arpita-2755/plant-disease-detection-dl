"""Small python-docx layer: headings, captions with fields, tables, code, sections."""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Emu, Inches, Pt, RGBColor
from PIL import Image

FONT = "Times New Roman"
BODY_W = 6.1     # inches of text width on A4 with the margins below


def set_font(style, size, bold=None, italic=None, color=None):
    style.font.name = FONT
    rpr = style.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.append(rf)
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(a), FONT)
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rf.get(qn(a)) is not None:
            del rf.attrib[qn(a)]
    style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if italic is not None:
        style.font.italic = italic
    style.font.color.rgb = RGBColor(*(color or (0, 0, 0)))


class Report:
    def __init__(self, refs: dict[str, str] | None = None):
        self.refs_in = refs or {}
        self.refs: dict[str, str] = {}
        self.missing: set[str] = set()
        self.ch = 0
        self.sec = self.sub = 0
        self.count = {"Figure": 0, "Table": 0}
        d = self.d = Document()
        s = d.sections[0]
        s.page_width, s.page_height = Cm(21.0), Cm(29.7)
        s.left_margin, s.right_margin = Cm(3.0), Cm(2.5)
        s.top_margin = s.bottom_margin = Cm(2.2)
        st = d.styles
        set_font(st["Normal"], 12)
        pf = st["Normal"].paragraph_format
        pf.line_spacing, pf.space_after = 1.5, Pt(4)
        for name, size, before, after in (("Heading 1", 16, 0, 14), ("Heading 2", 14, 10, 4), ("Heading 3", 12, 6, 3)):
            set_font(st[name], size, bold=True, italic=False)
            p = st[name].paragraph_format
            p.space_before, p.space_after, p.line_spacing, p.keep_with_next = Pt(before), Pt(after), 1.2, True
        set_font(st["Caption"], 10.5, bold=False, italic=False)
        st["Caption"].paragraph_format.line_spacing = 1.15
        for n in ("List Bullet", "List Number"):
            set_font(st[n], 12)
            st[n].paragraph_format.line_spacing = 1.5
            st[n].paragraph_format.space_after = Pt(1)

        from docx.enum.style import WD_STYLE_TYPE
        for n in ("toc 1", "toc 2", "table of figures"):
            t = st[n] if n in [x.name for x in st] else st.add_style(n, WD_STYLE_TYPE.PARAGRAPH, builtin=True)
            t.base_style = st["Normal"]
            set_font(t, 11)
            t.paragraph_format.line_spacing = 1.0
            t.paragraph_format.space_after = Pt(2)
            if n == "toc 2":
                t.paragraph_format.left_indent = Cm(0.6)
            t.paragraph_format.tab_stops.add_tab_stop(Cm(15.5), 2, 1)

    # ------------------------------------------------------------- inline text
    def ref(self, key: str) -> str:
        if key not in self.refs_in:
            self.missing.add(key)
        return self.refs_in.get(key, "?")

    def runs(self, p, text: str):
        """Inline markup: **bold**, *italic*, `code`, {ref:key}."""
        text = re.sub(r"\{ref:([\w-]+)\}", lambda m: self.ref(m[1]), text)
        for tok in re.split(r"(\*\*.+?\*\*|`.+?`|(?<!\w)\*[^*\s][^*]*?\*(?!\w))", text):
            if not tok:
                continue
            if tok.startswith("**"):
                p.add_run(tok[2:-2]).bold = True
            elif tok.startswith("`"):
                r = p.add_run(tok[1:-1])
                r.font.name = "Consolas"
                r._element.rPr.rFonts.set(qn("w:cs"), "Consolas")
                r.font.size = Pt(10.5)
            elif tok.startswith("*") and len(tok) > 2:
                p.add_run(tok[1:-1]).italic = True
            else:
                p.add_run(tok)
        return p

    def p(self, text: str, align="justify", size=None, after=None, indent=None, keep=False):
        p = self.d.add_paragraph()
        self.runs(p, " ".join(text.split()))
        p.alignment = {"justify": WD_ALIGN_PARAGRAPH.JUSTIFY, "left": WD_ALIGN_PARAGRAPH.LEFT,
                       "center": WD_ALIGN_PARAGRAPH.CENTER, "right": WD_ALIGN_PARAGRAPH.RIGHT}[align]
        if size:
            for r in p.runs:
                r.font.size = Pt(size)
        if after is not None:
            p.paragraph_format.space_after = Pt(after)
        if indent:
            p.paragraph_format.left_indent = Cm(indent)
        p.paragraph_format.keep_with_next = keep
        return p

    def ps(self, text: str):
        """Several paragraphs separated by blank lines."""
        for t in re.split(r"\n\s*\n", text.strip()):
            self.p(t)

    def bullets(self, items, style="List Bullet"):
        for t in items:
            p = self.d.add_paragraph(style=style)
            self.runs(p, " ".join(t.split()))
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    def equation(self, text: str, number: str | None = None):
        p = self.d.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(text)
        r.italic = True
        r.font.name = "Cambria Math"
        if number:
            p.add_run(f"        ({number})")

    # ---------------------------------------------------------------- headings
    def front(self, title: str, toc=True, new_page=True):
        """Unnumbered front-matter heading on a new page."""
        h = self.d.add_heading(title, 1) if toc else self.d.add_paragraph()
        if not toc:
            r = h.add_run(title)
            r.bold, r.font.size = True, Pt(16)
            h.paragraph_format.space_after = Pt(14)
        h.paragraph_format.page_break_before = new_page
        h.alignment = WD_ALIGN_PARAGRAPH.CENTER
        return h

    def chapter(self, title: str, page_break=True):
        self.ch += 1
        self.sec = self.sub = 0
        self.count = {"Figure": 0, "Table": 0}
        h = self.d.add_heading(f"Chapter {self.ch}: {title}", 1)
        h.paragraph_format.page_break_before = page_break
        return h

    def h2(self, title: str):
        self.sec += 1
        self.sub = 0
        self.d.add_heading(f"{self.ch}.{self.sec} {title}", 2)

    def h3(self, title: str):
        self.sub += 1
        self.d.add_heading(f"{self.ch}.{self.sec}.{self.sub} {title}", 3)

    # ------------------------------------------------------------------ fields
    def field(self, p, instr: str, text: str = ""):
        def el(t, **kw):
            e = OxmlElement(t)
            for k, v in kw.items():
                e.set(qn("w:" + k), v)
            return e

        r = p.add_run()
        r._r.append(el("w:fldChar", fldCharType="begin"))
        r2 = p.add_run()
        it = OxmlElement("w:instrText")
        it.set(qn("xml:space"), "preserve")
        it.text = instr
        r2._r.append(it)
        p.add_run()._r.append(el("w:fldChar", fldCharType="separate"))
        p.add_run(text)
        p.add_run()._r.append(el("w:fldChar", fldCharType="end"))

    def caption(self, kind: str, text: str, key: str | None, above: bool):
        self.count[kind] += 1
        n = self.count[kind]
        chap = str(self.ch) if self.ch else "A"
        chap = getattr(self, "chap_label", None) or chap
        p = self.d.add_paragraph(style="Caption")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run(f"{kind} {chap}.").bold = True
        self.field(p, f" SEQ {kind} \\* ARABIC" + (" \\r 1" if n == 1 else ""), str(n))
        p.add_run(f": {text}")
        p.paragraph_format.space_before = Pt(8 if above else 2)
        p.paragraph_format.space_after = Pt(4 if above else 12)
        p.paragraph_format.keep_with_next = above
        if key:
            self.refs[key] = f"{kind} {chap}.{n}"

    # ----------------------------------------------------------------- objects
    def figure(self, path, caption: str, key: str | None = None, width=BODY_W, max_h=7.6, border=False):
        w, h = Image.open(path).size
        width = min(width, max_h * w / h)
        p = self.d.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(8), Pt(4)
        pic = p.add_run().add_picture(str(path), width=Inches(width))
        pic._inline.docPr.set("descr", caption)
        if border:
            self.outline(pic)
        self.caption("Figure", caption, key, above=False)

    def pair(self, left, right, caption: str, key: str | None = None, sub=("", ""), width=3.0, border=True):
        """Two images side by side under one caption."""
        t = self.d.add_table(rows=1, cols=2)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for cell, img, lab in zip(t.rows[0].cells, (left, right), sub):
            cell.width = Inches(BODY_W / 2)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.BOTTOM
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.keep_with_next = True
            pic = p.add_run().add_picture(str(img), width=Inches(width))
            pic._inline.docPr.set("descr", f"{caption} {lab}")
            if border:
                self.outline(pic)
            q = cell.add_paragraph()
            q.alignment = WD_ALIGN_PARAGRAPH.CENTER
            q.paragraph_format.line_spacing = 1.0
            q.paragraph_format.space_after = Pt(2)
            q.paragraph_format.keep_with_next = True
            q.add_run(lab).font.size = Pt(10)
        t.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        self.caption("Figure", caption, key, above=False)

    @staticmethod
    def outline(pic):
        ln = OxmlElement("a:ln")
        ln.set("w", "6350")
        f = OxmlElement("a:solidFill")
        c = OxmlElement("a:srgbClr")
        c.set("val", "A6A6A6")
        f.append(c)
        ln.append(f)
        pic._inline.graphic.graphicData.pic.spPr.append(ln)

    def table(self, header, rows, caption=None, key=None, widths=None, size=10, align=None, bold_rows=()):
        if caption:
            self.caption("Table", caption, key, above=True)
        t = self.d.add_table(rows=1, cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        widths = widths or [BODY_W / len(header)] * len(header)
        scale = min(1.0, BODY_W / sum(widths))
        widths = [w * scale for w in widths]

        def fill(cell, text, head=False, bold=False, j=0):
            cell.width = Inches(widths[j])
            p = cell.paragraphs[0]
            p.paragraph_format.line_spacing = 1.1
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.space_before = Pt(1)
            self.runs(p, str(text))
            for r in p.runs:
                r.font.size = Pt(size)
                if head or bold:
                    r.bold = True
            if head or (align and align[j] == "c"):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if head:
                sh = OxmlElement("w:shd")
                sh.set(qn("w:val"), "clear")
                sh.set(qn("w:fill"), "D9E5F4")
                cell._tc.get_or_add_tcPr().append(sh)

        def tr_prop(row, tag):
            e = OxmlElement(tag)
            e.set(qn("w:val"), "true")
            row._tr.get_or_add_trPr().append(e)

        for j, h in enumerate(header):
            fill(t.rows[0].cells[j], h, head=True, j=j)
        tr_prop(t.rows[0], "w:tblHeader")
        tr_prop(t.rows[0], "w:cantSplit")
        for i, row in enumerate(rows):
            cells = t.add_row().cells
            for j, v in enumerate(row):
                fill(cells[j], v, bold=i in bold_rows, j=j)
            tr_prop(t.rows[-1], "w:cantSplit")
            if len(rows) <= 14 and i < len(rows) - 1:      # keep short tables on one page
                for c in cells:
                    for q in c.paragraphs:
                        q.paragraph_format.keep_with_next = True
        for c in t.rows[0].cells:
            c.paragraphs[0].paragraph_format.keep_with_next = True
        sp = self.d.add_paragraph()
        sp.paragraph_format.space_after = Pt(4)
        sp.paragraph_format.line_spacing = 0.7

    def code(self, text: str, size=8, title: str | None = None):
        if title:
            self.p(f"*{title}*", align="left", size=10.5, after=2, keep=True)
        t = self.d.add_table(rows=1, cols=1)
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = t.rows[0].cells[0]
        cell.width = Inches(BODY_W)
        sh = OxmlElement("w:shd")
        sh.set(qn("w:val"), "clear")
        sh.set(qn("w:fill"), "F5F5F3")
        cell._tc.get_or_add_tcPr().append(sh)
        p = cell.paragraphs[0]
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_after = Pt(0)
        lines = text.strip("\n").split("\n")
        if len(lines) <= 30:
            t.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        for i, line in enumerate(lines):
            r = p.add_run(line)
            r.font.name, r.font.size = "Consolas", Pt(size)
            if i < len(lines) - 1:
                r.add_break()
        sp = self.d.add_paragraph()
        sp.paragraph_format.space_after = Pt(4)
        sp.paragraph_format.line_spacing = 0.7

    # ---------------------------------------------------------------- sections
    def footer_pages(self, section, fmt: str, start: int | None):
        section.footer.is_linked_to_previous = False
        p = section.footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        self.field(p, " PAGE ", "1")
        for r in p.runs:
            r.font.size = Pt(11)
        pg = OxmlElement("w:pgNumType")
        pg.set(qn("w:fmt"), fmt)
        if start is not None:
            pg.set(qn("w:start"), str(start))
        section._sectPr.append(pg)

    def new_section(self):
        return self.d.add_section(WD_SECTION.NEW_PAGE)


def extract(path: Path, name: str, dedent=True) -> str:
    """Source of one top-level or nested function/class, exactly as in the repo."""
    import ast
    import textwrap

    src = path.read_text(encoding="utf8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            lines = src.split("\n")[node.lineno - 1 - len(node.decorator_list): node.end_lineno]
            return textwrap.dedent("\n".join(lines))
    raise KeyError(name)


def to_pdf(docx: Path, pdf: Path) -> int:
    """Update all fields (TOC, lists, captions) in Word and export to PDF."""
    import win32com.client as win32

    word = win32.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    try:
        doc = word.Documents.Open(str(docx))
        for _ in range(2):
            doc.Fields.Update()
            for toc in doc.TablesOfContents:
                toc.Update()
            for tof in doc.TablesOfFigures:
                tof.Update()
        pages = doc.ComputeStatistics(2)
        doc.Save()
        doc.ExportAsFixedFormat(str(pdf), 17, CreateBookmarks=1)
        doc.Close(False)
    finally:
        word.Quit()
    return pages
