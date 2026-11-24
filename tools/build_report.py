"""Build the project report (.docx and .pdf) from metrics.json and the plots.

  python tools/build_report.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "tools"))

import report_part1 as p1  # noqa: E402
import report_part2 as p2  # noqa: E402
import report_part3 as p3  # noqa: E402
from report_lib import Report, to_pdf  # noqa: E402

NAME = "Arpita_Mishra_12303670_INT422_Project_Report"


def build(M, refs):
    R = Report(refs)
    R.d.core_properties.title = p1.TITLE
    R.d.core_properties.author = "Arpita Mishra"
    R.d.core_properties.subject = "INT422 - Deep Learning project report"
    p1.cover(R)
    R.footer_pages(R.new_section(), "lowerRoman", 1)
    p1.front_matter(R, M)
    R.footer_pages(R.new_section(), "decimal", 1)
    for chapter in (p1.chapter1, p1.chapter2, p1.chapter3, p1.chapter4, p2.chapter5, p2.chapter6,
                    p2.chapter7, p3.chapter8, p3.chapter9, p3.chapter10):
        chapter(R, M)
    p3.references(R, M, p1.TODAY)
    p3.appendix(R, M)
    return R


def main() -> None:
    M = json.loads((ROOT / "metrics.json").read_text())
    first = build(M, {})               # pass 1 collects figure and table numbers
    final = build(M, first.refs)       # pass 2 resolves cross-references
    if final.missing:
        raise SystemExit(f"unresolved references: {final.missing}")
    out = ROOT / "report" / f"{NAME}.docx"
    final.d.save(out)
    pages = to_pdf(out, out.with_suffix(".pdf"))
    print(f"{out.name}: {pages} pages")


if __name__ == "__main__":
    main()
