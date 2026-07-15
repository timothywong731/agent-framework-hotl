"""Agent tools. Tool docstrings are the descriptions the LLM sees - they are
written for the model, not for human readers.

Two factories: :func:`make_tools` builds the three core tools every phase
agent carries (scratchpad steering, ledger raises, memory writes), bound to
the calling phase so agents cannot write outside their own section;
:func:`make_repo_tools` builds the exploration pair only deep_analysis
analyzers get, bound to their repo.
"""
from __future__ import annotations

from pathlib import Path

from agent_framework import tool

from .artifacts import ArtifactStore, Importance

# Repo root (src/hotl_demo/tools.py -> two parents up): the spec mandates a
# stable, CWD-independent steering-file path.
SCRATCHPAD_PATH = Path(__file__).resolve().parents[2] / "scratchpad.md"


def ensure_scratchpad(path: Path = SCRATCHPAD_PATH) -> None:
    """Create the steering file empty if missing; NEVER truncate an existing one.

    The human may have written guidance before the run started - that content
    must survive.

    Args:
        path: Scratchpad location; defaults to the repo-root file.

    Example:
        >>> ensure_scratchpad(Path("scratchpad.md"))  # doctest: +SKIP
    """
    if not path.exists():
        path.write_text("", encoding="utf-8")


def make_tools(store: ArtifactStore, phase: str, unit: str | None = None,
               scratchpad_path: Path = SCRATCHPAD_PATH) -> list:
    """Build the three core tools bound to one phase agent.

    The closures capture ``(store, phase, unit)`` so the LLM never supplies -
    and can never spoof - its own identity: a deep_analysis analyzer for
    ``oms-monolith`` physically cannot write another repo's memory section.

    Args:
        store: The run's shared artifact store.
        phase: Owning phase name (stamped onto ledger entries/memory writes).
        unit: Owning repo for analyzer instances, else ``None``.
        scratchpad_path: Steering file surfaced by ``read_scratchpad``.

    Returns:
        ``[read_scratchpad, raise_question, update_memory]`` - decorated
        tool functions ready to hand to an ``Agent``.

    Example:
        >>> read_scratchpad, raise_question, update_memory = make_tools(
        ...     store, "discovery")  # doctest: +SKIP
        >>> raise_question("In scope?", "docs omit recon", "in scope")  # doctest: +SKIP
        'Recorded q-1. Proceed using your stated default assumption.'
    """
    @tool(approval_mode="never_require")
    def read_scratchpad() -> str:
        """Read the human operator's scratchpad. It may contain steering guidance,
        priorities, or constraints for this assessment run. Always consult it
        before starting your work and follow any guidance it contains."""
        # The human picks the editor and therefore the encoding; this tool must
        # still return a string rather than raise (see read_file, same idiom).
        try:
            text = (
                scratchpad_path.read_text(encoding="utf-8", errors="replace")
                if scratchpad_path.exists()
                else ""
            )
        except OSError as exc:
            return f"ERROR: could not read the scratchpad ({exc}). Proceed without operator guidance."
        if text.strip():
            return text
        return "The scratchpad is empty. No operator guidance provided."

    @tool(approval_mode="never_require")
    def raise_question(question: str, context: str, default_assumption: str,
                       importance: str, impact: str) -> str:
        """Raise a question that requires human clarification or adjudication.
        Use when evidence conflicts or a decision-critical fact is missing.
        Provide: the question; the evidence context; the default assumption
        you will proceed with until a human answers; importance - exactly one
        of "high" (answer materially changes migration approach, scope, or
        cost), "medium" (affects one workstream or sequencing), or "low"
        (clarification that changes no decision); and impact - one or two
        sentences on how the human's answer would change the migration
        decision. Returns the question id."""
        # Validation failures return ERROR strings (never raise): the framework
        # feeds them back so the model can correct its call.
        if not question.strip() or not default_assumption.strip():
            return "ERROR: question and default_assumption must both be non-empty. Retry with both."
        if not impact.strip():
            return "ERROR: impact must explain how the human's answer would change the migration decision."
        try:
            level = Importance(importance.strip().lower())
        except ValueError:
            return "ERROR: importance must be exactly one of: high, medium, low."
        qid = store.raise_question(
            phase, unit, question.strip(), context.strip(), default_assumption.strip(),
            importance=level.value, impact=impact.strip(),
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
    """Build the exploration tools for one deep_analysis analyzer.

    The analyzer's prompt does not contain its repository - these two tools
    are how it sees the code, which is what makes the exploration genuinely
    agentic (and observable: any file-level finding must have come through
    ``read_file``).

    Args:
        repo_dir: The repository this analyzer owns; resolved once and used
            as the traversal guard boundary.

    Returns:
        ``[list_files, read_file]`` decorated tool functions.

    Example:
        >>> list_files, read_file = make_repo_tools(
        ...     Path("sample_data/repos/oms-monolith"))
        >>> "s3_uploader.py" in list_files()
        True
        >>> read_file("../oms-batch-recon/config.py").startswith("ERROR")
        True
    """
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
        # Resolve then containment-check: rejects ../ traversal and absolute
        # paths in one test, symlinks included.
        target = (root / path).resolve()
        if not target.is_relative_to(root):
            return "ERROR: path escapes the repository. Use a relative path from list_files."
        if not target.is_file():
            return f"ERROR: no such file: {path}. Call list_files to see valid paths."
        text = target.read_text(encoding="utf-8", errors="replace")
        # Cap pathological reads; sample repo files are far smaller.
        if len(text) > 20_000:
            text = text[:20_000] + "\n... (truncated)"
        return text

    return [list_files, read_file]
