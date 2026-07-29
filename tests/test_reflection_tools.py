"""Corpus tools (traversal guard, text filter) and the report writer.

Tools are called directly - a ``FunctionTool`` is callable and the repo's
existing tool tests do the same (see tests/test_reflexion_tools.py).
"""
from pathlib import Path

from reflection_demo.tools import make_corpus_tools, make_report_tools


def _corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    (root / "docs_src").mkdir(parents=True)
    (root / "docs_src" / "strategy.md").write_text("Azure is strategic", encoding="utf-8")
    (root / "app.py").write_text("print('hi')", encoding="utf-8")
    (root / "binary.pdf").write_bytes(b"%PDF-1.4")
    return root


def test_list_files_filters_to_text_suffixes(tmp_path):
    list_files, _ = make_corpus_tools(_corpus(tmp_path))
    listing = list_files()
    assert "docs_src/strategy.md" in listing
    assert "app.py" in listing
    assert "binary.pdf" not in listing


def test_read_file_reads_and_guards_traversal(tmp_path):
    root = _corpus(tmp_path)
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    _, read_file = make_corpus_tools(root)
    assert read_file("docs_src/strategy.md") == "Azure is strategic"
    assert read_file("../secret.txt").startswith("ERROR:")
    assert read_file("no/such.md").startswith("ERROR:")


def test_make_report_tools_returns_a_pair_with_no_reader(tmp_path):
    """Shape differs from reflexion's 3-tuple: nothing here reads the report."""
    result = make_report_tools(tmp_path / "report.md")
    assert len(result) == 2


def test_write_report_sets_the_flag_and_writes_atomically(tmp_path):
    report = tmp_path / "report.md"
    write_report, flag = make_report_tools(report)
    assert flag.written is False
    assert write_report("# Report\n").startswith("Report saved")
    assert flag.written is True
    assert report.read_text(encoding="utf-8") == "# Report\n"
    assert not (tmp_path / "report.md.tmp").exists()


def test_write_report_rejects_empty_markdown(tmp_path):
    write_report, flag = make_report_tools(tmp_path / "report.md")
    assert write_report("   ").startswith("ERROR:")
    assert flag.written is False
