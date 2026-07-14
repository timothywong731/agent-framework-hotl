"""Regenerate ``sample_data/docs/*.pdf`` from ``sample_data/docs_src/*.md``.

The committed PDFs are the pipeline's actual inputs; the markdown sources
exist so humans can edit the corpus. Re-run this script (and re-run the
sample-data tests) after any ``docs_src`` edit. Sources must stay ASCII:
fpdf2's built-in Helvetica is latin-1 only.
"""
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "sample_data" / "docs_src"
OUT = ROOT / "sample_data" / "docs"


def render(md_path: Path, pdf_path: Path) -> None:
    """Render one markdown source as a simple typeset PDF.

    Only two markdown constructs matter for the corpus: ``#``/``##``
    headings become bold sizes; everything else is body text. No nested
    formatting - the goal is a realistic-looking document, not fidelity.

    Args:
        md_path: Markdown source file (ASCII only).
        pdf_path: Destination PDF path (overwritten).

    Example:
        >>> render(SRC / "01_oms_application_architecture.md",
        ...        OUT / "01_oms_application_architecture.pdf")  # doctest: +SKIP
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    for line in md_path.read_text(encoding="utf-8").splitlines():
        # Heading prefixes switch the font; the prefix itself is stripped.
        if line.startswith("# "):
            pdf.set_font("Helvetica", "B", 16)
            line = line[2:]
        elif line.startswith("## "):
            pdf.set_font("Helvetica", "B", 13)
            line = line[3:]
        else:
            pdf.set_font("Helvetica", size=10)
        # multi_cell needs non-empty text; blank lines become a spacer row.
        pdf.multi_cell(0, 5, line if line.strip() else " ", new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(pdf_path))


def main() -> None:
    """Render every ``docs_src/*.md`` into ``docs/*.pdf`` and report each write."""
    OUT.mkdir(parents=True, exist_ok=True)
    for md in sorted(SRC.glob("*.md")):
        render(md, OUT / (md.stem + ".pdf"))
        print(f"wrote {OUT / (md.stem + '.pdf')}")


if __name__ == "__main__":
    main()
