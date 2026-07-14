"""The three agent tools. Docstrings below are the tool descriptions the LLM sees."""
from __future__ import annotations

from pathlib import Path

from agent_framework import tool

from .artifacts import ArtifactStore

SCRATCHPAD_PATH = Path("scratchpad.md")


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
