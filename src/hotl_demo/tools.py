"""The three agent tools. Docstrings below are the tool descriptions the LLM sees."""
from __future__ import annotations

from pathlib import Path

from agent_framework import tool

from .artifacts import ArtifactStore

# Repo root (src/hotl_demo/tools.py -> two parents up): the spec mandates a
# stable, CWD-independent steering-file path.
SCRATCHPAD_PATH = Path(__file__).resolve().parents[2] / "scratchpad.md"


def ensure_scratchpad(path: Path = SCRATCHPAD_PATH) -> None:
    if not path.exists():
        path.write_text("", encoding="utf-8")


def make_tools(store: ArtifactStore, phase: str, unit: str | None = None,
               scratchpad_path: Path = SCRATCHPAD_PATH) -> list:
    @tool(approval_mode="never_require")
    def read_scratchpad() -> str:
        """Read the human operator's scratchpad. It may contain steering guidance,
        priorities, or constraints for this assessment run. Always consult it
        before starting your work and follow any guidance it contains."""
        if scratchpad_path.exists():
            text = scratchpad_path.read_text(encoding="utf-8")
            if text.strip():
                return text
        return "The scratchpad is empty. No operator guidance provided."

    @tool(approval_mode="never_require")
    def raise_question(question: str, context: str, default_assumption: str) -> str:
        """Raise a question that requires human clarification or adjudication.
        Use when evidence conflicts or a decision-critical fact is missing.
        Provide the question, the evidence context, and the default assumption
        you will proceed with until a human answers. Returns the question id."""
        if not question.strip() or not default_assumption.strip():
            return "ERROR: question and default_assumption must both be non-empty. Retry with both."
        qid = store.raise_question(
            phase, unit, question.strip(), context.strip(), default_assumption.strip()
        )
        return f"Recorded {qid}. Proceed using your stated default assumption."

    @tool(approval_mode="never_require")
    def update_memory(key: str, value: str) -> str:
        """Record one finding in the shared long-term memory for this assessment.
        Call this once per key finding (3-8 times per phase). Use a short
        snake_case key (e.g. 'runtime', 'data_store', 'blockers') and a concise
        factual value."""
        if not key.strip() or not value.strip():
            return "ERROR: key and value must both be non-empty."
        store.update_memory(phase, unit, key.strip(), value.strip())
        return f"Memory updated: {key.strip()}"

    return [read_scratchpad, raise_question, update_memory]


def make_repo_tools(repo_dir: Path) -> list:
    """Exploration tools for deep_analysis: the analyzer walks the repo itself."""
    root = Path(repo_dir).resolve()

    @tool(approval_mode="never_require")
    def list_files() -> str:
        """List every file in the repository you are analyzing, as relative
        paths, one per line. Call this first to see the file tree."""
        files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
        return "\n".join(files) or "(empty repository)"

    @tool(approval_mode="never_require")
    def read_file(path: str) -> str:
        """Read one file from the repository by its relative path exactly as
        shown by list_files. Returns the full file contents."""
        target = (root / path).resolve()
        if not target.is_relative_to(root):
            return "ERROR: path escapes the repository. Use a relative path from list_files."
        if not target.is_file():
            return f"ERROR: no such file: {path}. Call list_files to see valid paths."
        text = target.read_text(encoding="utf-8", errors="replace")
        if len(text) > 20_000:
            text = text[:20_000] + "\n... (truncated)"
        return text

    return [list_files, read_file]
