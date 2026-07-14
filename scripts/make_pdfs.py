"""Regenerate sample_data/docs/*.pdf from sample_data/docs_src/*.md (fpdf2)."""
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "sample_data" / "docs_src"
OUT = ROOT / "sample_data" / "docs"


def render(md_path: Path, pdf_path: Path) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    for line in md_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            pdf.set_font("Helvetica", "B", 16)
            line = line[2:]
        elif line.startswith("## "):
            pdf.set_font("Helvetica", "B", 13)
            line = line[3:]
        else:
            pdf.set_font("Helvetica", size=10)
        pdf.multi_cell(0, 5, line if line.strip() else " ", new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(pdf_path))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for md in sorted(SRC.glob("*.md")):
        render(md, OUT / (md.stem + ".pdf"))
        print(f"wrote {OUT / (md.stem + '.pdf')}")


if __name__ == "__main__":
    main()
