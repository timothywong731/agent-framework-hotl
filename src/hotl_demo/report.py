"""Final report: LLM-composed verdict + deterministic adjudication log.

The narrative (executive summary, scorecard, recommendation) is one LLM call;
the adjudication log is rendered from the ledger by plain code and appended
afterwards, so the human-decision record is accurate regardless of what the
model does.
"""
from __future__ import annotations

from typing import Any

from agent_framework import Agent, Executor, WorkflowContext, handler
from agent_framework.ollama import OllamaChatClient
from typing_extensions import Never

from .artifacts import ArtifactStore
from .phases import PROMPT_ENV, ReportTrigger


def _cell(text: str) -> str:
    """Make arbitrary text safe inside one markdown table cell.

    Args:
        text: Any string, possibly containing newlines or pipes.

    Returns:
        Whitespace-collapsed text with ``|`` escaped.

    Example:
        >>> _cell("a|b\\nc")
        'a\\\\|b c'
    """
    # LLM-authored strings may contain newlines/pipes that would break the row.
    return " ".join(str(text).split()).replace("|", "\\|")


def render_adjudication_log(ledger: list[dict]) -> str:
    """Render the ledger as a markdown table - the run's human-decision record.

    Resolution wording per status: ``answered`` shows the human's verbatim
    answer; ``declined`` and ``open`` show the default assumption that was
    applied instead.

    Args:
        ledger: All ledger entries, in raise order.

    Returns:
        A markdown table (or a sentence when no questions were raised).

    Example:
        >>> print(render_adjudication_log([{"id": "q-1", "phase": "discovery",
        ...     "unit": None, "question": "In scope?", "status": "declined",
        ...     "human_answer": None, "default_assumption": "in scope"}]))
        | id | phase | question | resolution |
        |---|---|---|---|
        | q-1 | discovery | In scope? | declined - default applied: in scope |
    """
    if not ledger:
        return "No questions were raised during this run.\n"
    lines = [
        "| id | phase | question | resolution |",
        "|---|---|---|---|",
    ]
    for e in ledger:
        where = f"{e['phase']}[{e['unit']}]" if e["unit"] else e["phase"]
        if e["status"] == "answered":
            resolution = f"answered: {e['human_answer']}"
        elif e["status"] == "declined":
            resolution = f"declined - default applied: {e['default_assumption']}"
        else:
            resolution = f"open - default assumption applied: {e['default_assumption']}"
        lines.append(f"| {e['id']} | {where} | {_cell(e['question'])} | {_cell(resolution)} |")
    return "\n".join(lines) + "\n"


def build_report_prompt(memory_text: str, reports: dict[str, str], ledger: list[dict]) -> str:
    """Render the final-report prompt via ``prompts/final_report.md``.

    Args:
        memory_text: Post-adjudication ``memory.json`` as pretty JSON.
        reports: Phase reports keyed by filename
            (from :meth:`ArtifactStore.read_all_reports`).
        ledger: Full ledger; summarized so the model treats human answers as
            authoritative.

    Returns:
        The complete prompt string for the report agent.
    """
    report_blocks = "\n\n".join(f"--- {name} ---\n{text}" for name, text in reports.items())
    adjudications = "\n".join(
        f"- {e['id']} [{e['status']}] {e['question']}"
        + (f" -> HUMAN: {e['human_answer']}" if e["human_answer"] else
           f" -> default: {e['default_assumption']}")
        for e in ledger
    ) or "none"
    return PROMPT_ENV.get_template("final_report.md").render(
        adjudications=adjudications, memory=memory_text, reports=report_blocks
    )


class FinalReportExecutor(Executor):
    """Terminal workflow node: writes ``final_report.md`` and yields its path.

    Example:
        >>> executor = FinalReportExecutor(store)  # doctest: +SKIP
    """

    def __init__(self, store: ArtifactStore, agent: Any | None = None) -> None:
        """Create the report agent (no tools - pure composition).

        Args:
            store: The run's shared artifact store.
            agent: Test seam - scripted stand-in replaces the Ollama-backed
                ``Agent`` when provided.
        """
        super().__init__(id="final_report")
        self._store = store
        self._agent = agent or Agent(
            client=OllamaChatClient(),
            name="final_report",
            instructions="You write crisp executive assessment reports.",
        )

    @handler
    async def on_report(self, trig: ReportTrigger, ctx: WorkflowContext[Never, str]) -> None:
        """Compose the final report and yield its path as the workflow output.

        Args:
            trig: The review gate's release signal.
            ctx: Workflow context; ``yield_output`` carries the report path
                back to the CLI runner.
        """
        ledger = self._store.read_ledger()
        prompt = build_report_prompt(
            self._store.memory_text(), self._store.read_all_reports(), ledger
        )
        result = await self._agent.run(prompt)
        text = (result.text or "(no text returned by the model)")
        # Deterministic appendix: the adjudication table comes from the ledger,
        # never from the model, so the decision record cannot be hallucinated.
        text += "\n\n## Adjudication log\n\n" + render_adjudication_log(ledger)
        path = self._store.write_report("final_report.md", text)
        print("  final_report: written")
        await ctx.yield_output(str(path))
