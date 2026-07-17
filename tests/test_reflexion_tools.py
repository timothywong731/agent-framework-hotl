"""Corpus tools (traversal guard, text filter) and report tools (atomic write, flag)."""
from pathlib import Path

from reflexion_demo.tools import make_corpus_tools, make_report_tools


def _corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    (root / "docs_src").mkdir(parents=True)
    (root / "docs_src" / "strategy.md").write_text("Azure is strategic", encoding="utf-8")
    (root / "app.py").write_text("print('hi')", encoding="utf-8")
    (root / "notes.txt").write_text("note", encoding="utf-8")
    (root / "binary.pdf").write_bytes(b"%PDF-1.4")
    return root


def test_list_files_filters_to_text_suffixes(tmp_path):
    list_files, _ = make_corpus_tools(_corpus(tmp_path))
    listing = list_files()
    assert "docs_src/strategy.md" in listing
    assert "app.py" in listing
    assert "notes.txt" in listing
    assert "binary.pdf" not in listing


def test_read_file_reads_and_guards_traversal(tmp_path):
    root = _corpus(tmp_path)
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    _, read_file = make_corpus_tools(root)
    assert read_file("docs_src/strategy.md") == "Azure is strategic"
    assert read_file("../secret.txt").startswith("ERROR:")
    assert read_file("no/such.md").startswith("ERROR:")


def test_read_file_rejects_non_text_suffix(tmp_path):
    _, read_file = make_corpus_tools(_corpus(tmp_path))
    assert read_file("binary.pdf").startswith("ERROR:")


def test_read_file_truncates_oversized(tmp_path):
    root = _corpus(tmp_path)
    (root / "big.txt").write_text("x" * 25_000, encoding="utf-8")
    _, read_file = make_corpus_tools(root)
    text = read_file("big.txt")
    assert len(text) < 25_000
    assert text.endswith("... (truncated)")


def test_write_report_sets_flag_and_overwrites_atomically(tmp_path):
    report = tmp_path / "report.md"
    write_report, read_report, flag = make_report_tools(report)
    assert flag.written is False
    assert read_report().startswith("ERROR:")          # nothing written yet

    assert "saved" in write_report("# Draft 1").lower()
    assert flag.written is True
    assert report.read_text(encoding="utf-8") == "# Draft 1"
    assert read_report() == "# Draft 1"

    write_report("# Draft 2")                           # revision overwrites
    assert report.read_text(encoding="utf-8") == "# Draft 2"
    assert not list(tmp_path.glob("*.tmp"))             # no temp litter


def test_write_report_rejects_empty(tmp_path):
    write_report, _, flag = make_report_tools(tmp_path / "report.md")
    assert write_report("   ").startswith("ERROR:")
    assert flag.written is False
