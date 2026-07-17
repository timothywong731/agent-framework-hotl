"""Closure-bound tools. Docstrings are the descriptions the LLM sees.

Same idioms as the HOTL demo's tools: traversal-guarded resolution, oversized
reads truncated, failures returned as ``ERROR:`` strings (never raised) so
the framework feeds them back to the model.

Information parity: worker and reviewer get the IDENTICAL corpus binding from
:func:`make_corpus_tools`; the reviewer additionally reads the artifact under
review (``read_report``), the worker additionally writes it (``write_report``).
"""
import os
from pathlib import Path

from agent_framework import tool

TEXT_SUFFIXES = frozenset({".md", ".py", ".txt"})
_READ_CAP = 20_000


def atomic_write(path: Path, text: str) -> None:
    """Write via temp file + ``os.replace`` so readers never see a torn file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def make_corpus_tools(corpus_root: Path) -> list:
    """Build the shared read-only corpus pair, bound to one root.

    Args:
        corpus_root: Directory both agents may read; resolved once and used
            as the traversal guard boundary.

    Returns:
        ``[list_files, read_file]`` decorated tool functions.
    """
    root = Path(corpus_root).resolve()

    @tool(approval_mode="never_require")
    def list_files() -> str:
        """List every readable source file in the corpus as relative paths,
        one per line. Call this first to see what documentation and code is
        available."""
        files = sorted(
            p.relative_to(root).as_posix()
            for p in root.rglob("*")
            if p.is_file() and p.suffix in TEXT_SUFFIXES
        )
        return "\n".join(files) or "(empty corpus)"

    @tool(approval_mode="never_require")
    def read_file(path: str) -> str:
        """Read one corpus file by its relative path exactly as shown by
        list_files. Returns the full file contents."""
        target = (root / path).resolve()
        if not target.is_relative_to(root):
            return "ERROR: path escapes the corpus. Use a relative path from list_files."
        if target.suffix not in TEXT_SUFFIXES or not target.is_file():
            return f"ERROR: no such readable file: {path}. Call list_files to see valid paths."
        text = target.read_text(encoding="utf-8", errors="replace")
        if len(text) > _READ_CAP:
            text = text[:_READ_CAP] + "\n... (truncated)"
        return text

    return [list_files, read_file]


class ReportFlag:
    """Mutable cell recording whether write_report ran this turn.

    A fresh instance comes out of :func:`make_report_tools` per turn, so no
    reset discipline is needed anywhere.
    """

    def __init__(self) -> None:
        self.written = False


def make_report_tools(report_path: Path) -> tuple:
    """Build the report write/read pair bound to one run's report file.

    Args:
        report_path: ``output/reflexion_<ts>/report.md`` for this run.

    Returns:
        ``(write_report, read_report, flag)`` - the worker gets
        ``write_report``, the reviewer gets ``read_report``, the worker
        executor checks ``flag.written`` after each turn.
    """
    flag = ReportFlag()

    @tool(approval_mode="never_require")
    def write_report(markdown: str) -> str:
        """Save the complete migration report. Pass the FULL report as
        markdown - this overwrites any previous draft, so never send a
        fragment or a diff."""
        if not markdown.strip():
            return "ERROR: report must be non-empty markdown. Send the full report text."
        try:
            atomic_write(report_path, markdown)
        except OSError as exc:
            return f"ERROR: could not save the report ({exc})."
        flag.written = True
        return f"Report saved ({len(markdown)} chars)."

    @tool(approval_mode="never_require")
    def read_report() -> str:
        """Read the report under review exactly as the author saved it."""
        if not report_path.exists():
            return "ERROR: no report has been written yet."
        return report_path.read_text(encoding="utf-8", errors="replace")

    return write_report, read_report, flag
