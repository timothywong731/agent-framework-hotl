"""Closure-bound tools. Docstrings are the descriptions the LLM sees.

A deliberate copy of ``reflexion_demo/tools.py`` minus ``read_report``:
demo packages here are standalone and must be readable end to end without
tracing imports into a sibling (see the design spec, section 9).

No *tool* here reads the report file: the judge is handed the report's text
by ``judging.make_judge_predicate``, which reads it in plain Python, so the
judge needs no ``read_report`` and stays tool-less. The corpus is the only
channel it is denied - that asymmetry is the pattern.
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
    """Build the read-only corpus pair, bound to one root.

    Byte-for-byte the reflexion worker's corpus binding. That identity is
    the control in the experiment: only the critic differs.

    Args:
        corpus_root: Directory the agent may read; resolved once and used as
            the traversal guard boundary.

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
    """Mutable cell recording whether write_report ever ran this run.

    One instance per run, not per pass: the loop middleware drives every
    pass inside a single ``agent.run()``, so there is no per-pass boundary
    to reset on. It answers one question - did the report ever land - which
    is all the fallback in main.py needs.
    """

    def __init__(self) -> None:
        self.written = False


def make_report_tools(report_path: Path) -> tuple:
    """Build the report writer bound to one run's report file.

    Args:
        report_path: ``output/reflection_<ts>/report.md`` for this run.

    Returns:
        ``(write_report, flag)``. There is no reader *tool* - see the module
        docstring.
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

    # 2-tuple, where reflexion_demo's make_report_tools returns a 3-tuple
    # (write_report, read_report, flag): no participant here reads the report
    # back through a tool - the predicate hands the judge its text directly.
    return write_report, flag
